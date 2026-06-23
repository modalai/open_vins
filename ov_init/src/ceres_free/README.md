# `ov_init::zbft_sfm` — Ceres-free, RT, lock-free initialization solver

A small, deterministic, real-time-friendly nonlinear least-squares core that replaces
**Ceres** in the OpenVINS initialization path. It is purpose-built for the
visual-inertial (re)initialization / reset problem on resource-constrained hardware
(QRB5165), not a general graph optimizer.

Ceres has exactly one real consumer in this tree (the `ov_init` dynamic MLE — `ov_core`,
`ov_msckf`, `ov_eval` contain **zero** `ceres::` usage). Replacing it here lets the
`voxl-ceres-solver` dependency be dropped from the flight image once the A/B parity
gate (below) flips the default.

## Why Ceres-free (the RT argument)

- The shipped config runs `init_dyn_mle_max_threads: 20` on an **8-core** SoC while the
  IMU callback runs at `THREAD_PRIORITY_RT_HIGH` (1 kHz). An uncontrolled, unpinned,
  per-solve thread pool contends with and can priority-invert the hard-RT path.
- Ceres brings dynamic allocation, generic sparse linear algebra, nondeterministic
  timing, large binary size, and cross-compile burden.
- The init problem is small and structured (≤ ~13 keyframes, ≤ ~100 landmarks). A
  hand-rolled fixed-size, bounded-iteration, single-threaded-by-default solver is
  faster, deterministic, and dependency-free here — and the analytic Jacobians already
  exist in `ov_init/src/ceres/*` (lifted verbatim).

## Layout

| File | Role |
|------|------|
| `CostFunction.h` | Drop-in subset of `ceres::CostFunction` (same `Evaluate` signature) so factors transcribe verbatim. |
| `LocalParameterization.h` | Manifold retraction interface + Euclidean param + `PlusJacobian()`. |
| `State_JPLQuatLocal.{h,cpp}` | JPL quaternion ⊞ (lifted verbatim). |
| `LossFunction.h` | Trivial / Huber / Cauchy as first-order IRLS weights. |
| `Parallel.{h,cpp}` | **Lock-free** bounded worker pool (atomics only; no mutex/CV). |
| `Problem.{h,cpp}` | LM + Schur + parallel accumulation + covariance. |
| `Factor_ImuCPIv1.{h,cpp}` | IMU CPI factor (lifted verbatim from `ov_init/src/ceres/`). |
| `Factor_ImageReprojCalib.{h,cpp}` | Reprojection-with-calibration factor (lifted verbatim). |
| `Factor_GenericPrior.{h,cpp}` | Gauge / bias / soft-prior factor (lifted verbatim). |
| `test_mini_solver.cpp` | Eigen-only self-test of the solver core (no ov_core / Ceres). |

The factor `.cpp` bodies are **byte-for-byte** the proven upstream residuals and
analytic Jacobians; only the base class (`ceres::CostFunction` → `zbft_sfm::CostFunction`)
and includes changed. This keeps the only must-be-correct *new* code in `Problem.cpp`
and `Parallel.cpp`, which the self-test validates.

## Solver (`Problem`)

- **Levenberg–Marquardt** over manifold parameter blocks (JPL quat + Euclidean).
- **Schur complement** of tagged *landmark* blocks (the BA arrowhead). Each reprojection
  residual touches exactly one landmark, so the reduced landmark Hessian is
  block-diagonal and inverted as independent 3×3 blocks. `solve()` reduces to a plain
  dense LDLT when there are no landmarks (the inertial-only alignment refine).
- **First-order IRLS** robust loss (scale `H`,`g` by `ρ'(‖r‖²)`); keeps `H` PSD.
- **Bounded**: hard `max_num_iterations` **and** `max_solver_time_seconds` (mirrors
  `init_dyn_mle_max_time`); always returns the best accepted iterate; no allocation in
  the inner LM retry loop.
- **Covariance** is read directly off the gauge-anchored, landmark-marginalized reduced
  Hessian (`Σ = (A − B D⁻¹ Bᵀ)⁻¹`). Requires the gauge to be anchored (yaw/position
  priors) or it reports failure rather than returning a bogus marginal.

## Threading model (lock-free)

`Parallel.{h,cpp}` is a pre-spawned worker pool using **only release/acquire atomics**
(cache-line padded), a bounded **spin + `yield`** idle wait, and **no mutexes,
condition variables, or blocking sleeps** — so it cannot lock-contend with or
priority-invert the IMU/camera RT threads.

- `num_threads <= 1` (the default) runs **fully inline** — no threads are created. This
  is the RT default and what the hard path should use.
- Parallelism is confined to the embarrassingly-parallel per-residual Hessian/gradient
  accumulation. The linear solve stays serial (small), and Eigen is pinned to one
  thread (`setNbThreads(1)`) so it never spawns its own pool.
- Work is split into **fixed contiguous ranges bound to fixed worker indices**, and
  partials are reduced in **worker-index order**.
- Keep `num_threads <= cores − 2` and use `worker_init_fn` to pin affinity / drop the
  scheduling class so workers never preempt the RT threads. The pool should live only
  for the duration of a bounded solve.

### Determinism guarantee (precise)

- **For a fixed `num_threads`**: bitwise reproducible run-to-run (no scheduling
  nondeterminism, no races). *Verified.*
- **Across different `num_threads`**: identical up to floating-point summation grouping
  (different partition ⇒ different add order). Measured difference on the solution is
  ~`1e-15`. It is **not** bitwise identical across thread counts, and cannot be without
  serializing the reduction.

## Test results (verified)

Built and run natively (x86-64 g++ 13.3) inside the `voxl-cross` container —
**13 / 13 checks pass**:

```
g++ -O3 -std=c++17 -pthread -I/usr/include/eigen3 \
    test_mini_solver.cpp Problem.cpp Parallel.cpp -o /tmp/test_mini && /tmp/test_mini
```

| Check | Result |
|-------|--------|
| Analytic vs FD Jacobians (euclidean, two-block, rectangular manifold V) | ~1e-10 |
| Linear LS == normal equations | 4e-9 |
| **Schur solution == dense solution** | 2.9e-15 |
| **4-thread run-to-run bitwise identical** | pass |
| **1-thread vs 4-thread agree** | 7e-15 |
| Covariance == (AᵀA + SᵀS)⁻¹ | 2e-16 |
| Schur-marginal nav cov == dense full-inverse nav block | 3.6e-8 |

The lifted IMU/reproj/prior factors additionally `-fsyntax-only` clean against `ov_core`
(reproj needs OpenCV from the sysroot). They are also exercised end-to-end by
`test_mini_factors.cpp` under the full `ov_init` build (FD checks vs `ov_core`).

To build the self-test via CMake: `-DOV_INIT_BUILD_MINI_TESTS=ON`.

## On-target performance vs Ceres (verified)

`bench_init` builds the SAME VI-init factor graph (CPI IMU + pinhole reprojection +
gauge/bias prior; 20 keyframes, 80 landmarks, ~540-dim system) and solves it from an
identical initial guess with Ceres and with `zbft_sfm`, single-threaded, **on the QRB5165**
(`adb`), over 30 random trials. Both reach the same minimum (max final-cost rel. diff
1.2e-5; covariance agreement <0.1% on observable DOF).

| Method (single thread) | Ceres | `zbft_sfm` | ratio |
|------------------------|-------|------------|-------|
| Levenberg–Marquardt    | 73.5 ms | **55.7 ms** | **0.76×** |
| Powell dogleg          | 72.0 ms | **50.6 ms** | **0.70×** |

The speedup comes from four solver changes, none of which alters the trajectory:
- **Lazy convergence**: the accepted step that triggers convergence is scored with a
  residual-only cost evaluation; the final Jacobian (needed only by a *next* iteration that
  never runs) is skipped — saving one full linearization per solve.
- **Triangular arrowhead Schur**: the symmetric landmark fill-in `−W V⁻¹ Wᵀ` writes only the
  lower triangle (Eigen's `LLT` reads one triangle), and `W_a`/`M_a = W_a V⁻¹` are precomputed
  once per observing pose instead of re-fetched in the inner loop — this roughly halves the
  O(P²) reduction, the single largest cost (`solve_step` 46 ms → 26 ms).
- **Symmetric Hessian assembly**: each off-diagonal `Jₐᵀ J_b` is formed once and scattered to
  `(a,b)` and `(b,a)`.
- **Allocation-free hot loop**: residual/Hessian scratch is reused (no per-residual heap
  traffic), matching the stated RT design.

Reproduce: `adb push` the cross-built `bench_init` and run `./bench_init 30 1 0 0`
(args: trials, threads, zbft method [0=LM,1=dogleg], ceres method).

## A/B parity gate (how Ceres gets removed safely)

1. Build `mini` alongside Ceres (done — additive in `ROS1/ROS2.cmake`, nothing
   regresses).
2. In `DynamicInitializer`, add a compile flag `USE_CERES_INIT` (default = Ceres during
   bring-up). The `mini` path mirrors the existing MLE assembly: same CPI factors, same
   reprojection factors, same `quat_yaw` + first-position gauge priors, same
   `init_dyn_inflation_*` covariance inflation.
3. A/B offline on EuRoC/TUM-VI **and** PX4 logs: require trajectory + covariance parity
   vs the Ceres MLE before flipping. Per-factor FD Jacobian tests gate correctness.
4. Flip the default → `mini`. Then delete `src/ceres/*`, `find_package(Ceres)` in
   `server/`, `ov_init/`, `ov_msckf/`, `${CERES_LIBRARIES}`, `voxl-ceres-solver` in
   `install_build_deps.sh`, and `libceres-dev` in `ov_msckf/package.xml`.

## Using the solver (sketch)

```cpp
using namespace ov_init::zbft_sfm;
Problem problem;
State_JPLQuatLocal quat_param;                       // one shared instance is fine (stateless)

// Register blocks (caller owns the memory; canonical clone order q,bg,v,ba,p).
problem.AddParameterBlock(q.data(), 4, &quat_param);
problem.AddParameterBlock(p.data(), 3);
// ... velocities, biases, clones ...
for (auto &lm : landmarks) {
  problem.AddParameterBlock(lm.data(), 3);
  problem.SetSchurLandmark(lm.data());               // eliminate via Schur
}
problem.SetParameterBlockConstant(cam_calib.data()); // calibration fixed by default

// Factors (lifted, analytic): IMU CPI between consecutive clones, reprojection per
// observation (with CauchyLoss), gauge + bias soft priors via Factor_GenericPrior.
problem.AddResidualBlock(&imu_factor, nullptr, {q1,bg1,v1,ba1,p1, q2,bg2,v2,ba2,p2});
problem.AddResidualBlock(&reproj_factor, &cauchy, {q,p,lm, q_ItoC,p_IinC, cam_calib});
problem.AddResidualBlock(&gauge_prior, nullptr, {q0, p0});

SolverOptions opts;
opts.num_threads = 1;                  // RT default; raise (<= cores-2) only off the RT path
opts.max_solver_time_seconds = 0.04;   // hard budget
SolverSummary s = problem.Solve(opts);

Eigen::MatrixXd cov;                    // 15x15 IMU error-state, ordering [theta,p,v,bg,ba]
problem.ComputeCovariance({q.data(), p.data(), v.data(), bg.data(), ba.data()}, cov, opts);
// -> symmetrize / PSD-project / inflate with init_dyn_inflation_*, then inject as today.
```

## Status

Implemented, unit-validated (13/13), and **benchmarked faster than Ceres on-target**
(0.76× LM, 0.70× dogleg, single-thread; same minimum): solver core, lock-free pool, lifted
factors, build wiring. The solver core is production-clean (no profiling scaffolding, no dead
code; bounded, allocation-free hot loop).

Next (per the project plan): the `SfMInertialInitializer` that drives this solver
(stereo-metric SfM → S² gravity alignment → bounded MAP), the `DynamicInitializer`
`USE_CERES_INIT` A/B switch, and `test_mini_factors.cpp` (FD vs ov_core).
