# `ov_init::zbft_sfm` — Ceres-free, RT, lock-free initialization solver

> Contributor: Joao Leonardo Silva Cotta (@zauberflote1)

A small, deterministic, real-time-friendly nonlinear least-squares core that replaces
**Ceres** in the OpenVINS dynamic-initialization path, plus the S²-gravity MLE, gauge
anchoring, gravity gate/re-align, covariance recovery, and warm-start injection built on
top of it. It is purpose-built for the visual-inertial (re)initialization / reset problem
on resource-constrained hardware (QRB5165), not a general graph optimizer.

Ceres has exactly one real consumer in this tree (the `ov_init` dynamic MLE — `ov_core`,
`ov_msckf`, `ov_eval` contain **zero** `ceres::` usage). Replacing it here lets the
`voxl-ceres-solver` dependency be dropped from the flight image once the removal step
(below) is taken.

> **Status in one line:** the ceres-free path is **implemented, wired into
> `DynamicInitializer`, and ON BY DEFAULT** (`OV_INIT_CERES_FREE=ON`). It is what a normal
> build compiles and what the shipped `fpv` config runs. Ceres is retained only as a
> compile-time A/B fallback and has **not** been removed yet. See **Current status** at the
> bottom for what is verified vs. what is still open — do not assume "done" means
> "consistency-validated on hardware."

## Why Ceres-free (the RT argument)

- The init problem is small and structured (≤ ~13 keyframes, ≤ ~100 landmarks). A
  hand-rolled fixed-size, bounded-iteration, single-threaded-by-default solver is faster,
  deterministic, and dependency-free here — and the analytic Jacobians already exist in
  `ov_init/src/ceres/*` (lifted verbatim).
- Ceres brings dynamic allocation, generic sparse linear algebra, nondeterministic timing,
  large binary size, and cross-compile burden. An uncontrolled per-solve thread pool
  contends with and can priority-invert the hard-RT IMU/camera path.

## Layout

| File | Role |
|------|------|
| `CostFunction.h` | Drop-in subset of `ceres::CostFunction` (same `Evaluate` signature) so factors transcribe verbatim. |
| `LocalParameterization.h` | Manifold retraction interface + Euclidean param + **`GravityS2Parameterization`** (S²(G) gravity, 3-global/2-local) + `PlusJacobian()`. |
| `State_JPLQuatLocal.{h,cpp}` | JPL quaternion ⊞ (lifted verbatim). |
| `LossFunction.h` | Trivial / Huber / Cauchy as first-order IRLS weights. |
| `Parallel.{h,cpp}` | **Lock-free** bounded worker pool (atomics only; no mutex/CV). |
| `Problem.{h,cpp}` | LM (+ optional dogleg) + Schur + parallel accumulation + covariance. |
| `Factor_ImuCPIv1.{h,cpp}` | IMU CPI factor. Now carries an **11th, S²-gravity parameter block** so gravity is a free 2-DOF variable shared across all IMU factors (lifted from `ov_init/src/ceres/`, extended for gravity). |
| `Factor_ImageReprojCalib.{h,cpp}` | Reprojection-with-calibration factor (lifted verbatim). |
| `Factor_GenericPrior.{h,cpp}` | Gauge / bias / soft-prior factor (lifted verbatim). |

The factor residual/Jacobian bodies are the proven upstream code; the only must-be-correct
*new* code is `Problem.cpp`, `Parallel.cpp`, and the S² parameterization, all covered by the
self-tests below.

## Solver (`Problem`)

- **Levenberg–Marquardt** (default) over manifold parameter blocks (JPL quat + Euclidean +
  S² gravity). Dogleg is available (`SolverOptions::use_dogleg`) but LM is the wired default
  — it gave better results on real hardware.
- **Schur complement** of tagged *landmark* blocks (the BA arrowhead), inverted as
  independent 3×3 blocks. `solve()` reduces to a dense LDLT when there are no landmarks.
- **First-order IRLS** robust loss (scale `H`,`g` by `ρ'(‖r‖²)`); keeps `H` PSD.
- **Bounded**: hard `max_num_iterations` **and** `max_solver_time_seconds` (mirrors
  `init_dyn_mle_max_time`); always returns the best accepted iterate; allocation-free hot loop.
- **Covariance** is read off the gauge-anchored, landmark-marginalized reduced Hessian
  (`Σ = (A − B D⁻¹ Bᵀ)⁻¹`). `ComputeCovariance()` can slice **any** requested nav blocks in one
  factorization — the integration uses this to recover the full **joint** IMU+clones
  covariance for warm-start in a single shot (see below).

## S²-gravity MLE (the "S2" everyone asks about)

Gravity is optimized as a free direction on the sphere **S²(G)** (fixed magnitude `G =
gravity_mag`, 3 global coords, **2 local** tilt DOF) via `GravityS2Parameterization`:

- Retraction `g ⊞ δ = G·normalize(g + B(g)·δ)`, with `B(g) ∈ ℝ³ˣ²` an orthonormal tangent
  basis. Basis rule is deterministic and never degenerate (seed = axis least aligned with
  `ĝ` via `argmin_i |ĝ_i|`, then Gram–Schmidt) — valid at the poles included.
- One shared `var_gravity` block is added to the problem and **never fixed**, so all IMU CPI
  factors see the same optimized gravity. The 2 observable tilt DOF live here instead of in a
  `quat_yaw` gauge, so the first-pose gauge can use a **full 3-DOF orientation soft prior**
  (tight, ~0.057°) + position + bias priors rather than hard-fixing states — which is what lets
  init/reset succeed at aggressive/arbitrary attitudes.
- **Post-solve gravity gate + conditional re-align:** if optimized `g` is >30° from +Z the
  init is **rejected** (flip/corruption guard, protects downstream PX4 NED); if >0.1° all
  states are rotated (Rodrigues) so `g → +Z` before injection.
- **Covariance is recovered AT the optimum, before re-align**, then carried through the
  re-align by a closed-form similarity `T = blkdiag(I, R, R, I, I)` (per clone `blkdiag(I,R)`)
  — so the injected covariance stays consistent with the (possibly rotated) injected state
  instead of being re-linearized off-optimum.

## Warm-start injection

`init_warmstart_inject` (config; **`true` in the shipped `fpv` config**, default `false` in
code) makes the initializer return the full joint covariance over the IMU state **and all
window clones** (landmark-marginalized), which `StateHelper::set_initial_state_warmstart()`
uses to seed the EKF with those clones — so the filter can update immediately after a
(re)init instead of cold-starting and re-collecting a fresh ~2 s window. Gating lives in
`VioManagerHelper::try_to_initialize` (`warmstart_next_init`, set by `soft_reset`); on any
contract mismatch it falls back to the legacy IMU-only (15×15) cold-start seed. The
inflation is applied as a proper **congruence** `S·Σ·Sᵀ` extended over the clone blocks
(not a diagonal-only scale), and a new `init_dyn_inflation_pos` knob exists (default `1.0` =
no change).

## Threading model (lock-free)

`Parallel.{h,cpp}` is a pre-spawned worker pool using **only release/acquire atomics**,
bounded spin+`yield` idle wait, **no mutexes / CV / sleeps** — so it cannot lock-contend or
priority-invert the IMU/camera RT threads. `num_threads <= 1` runs **fully inline** (no
threads). Parallelism is confined to per-residual Hessian/gradient accumulation; the linear
solve is serial and Eigen is pinned to one thread. Determinism: **bitwise reproducible for a
fixed `num_threads`** (verified); across thread counts, identical up to FP summation grouping
(~1e-15). The shipped `fpv` config uses `init_dyn_mle_max_threads: 4`.

## Build wiring

- CMake option **`OV_INIT_CERES_FREE` (default `ON`)** → `add_definitions(-DUSE_CERES_FREE_INIT)`
  (`ov_init/CMakeLists.txt`). This macro selects the ceres-free path in
  `DynamicInitializer.cpp`; the Ceres path is the `#else` fallback. Build with
  `-DOV_INIT_CERES_FREE=OFF` to A/B against Ceres.
- `ceres_free/*.cpp` are compiled into `ov_init_lib` (`ov_init/cmake/ROS1.cmake`).
- **Ceres is still a hard dependency and has NOT been removed:** `find_package(Ceres REQUIRED)`,
  `${CERES_LIBRARIES}` on the link line, `voxl-ceres-solver` in `install_build_deps.sh`, and the
  `src/ceres/*` factors still dual-compiled as the fallback. (Note: because the active build
  `#ifdef`s the Ceres path out, the shipped `libov_init_lib.so` contains no `ceres::` symbols
  even though it links libceres.)

## Verification (reproduced natively — x86-64 g++ 13.3, this pass)

Eigen-only self-tests, built and run in-tree just now:

```
# solver core — 13/13 pass
g++ -O3 -std=c++17 -pthread -I/usr/include/eigen3 -Iceres_free \
    ceres_free/test_mini_solver.cpp ceres_free/Problem.cpp ceres_free/Parallel.cpp -o /tmp/t && /tmp/t

# warm-start covariance (shared-pointer joint slice + realign/inflation congruence) — 20/20 pass
g++ -O2 -std=c++17 -pthread -I/usr/include/eigen3 -Iceres_free \
    ceres_free/test_warmstart_cov.cpp ceres_free/Problem.cpp ceres_free/Parallel.cpp -o /tmp/tw && /tmp/tw
```

| Check | Result (this run) |
|-------|-------------------|
| Analytic vs FD Jacobians (euclidean, two-block, rectangular manifold V) | ~1e-10 |
| Schur solution == dense solution | 3.97e-15 |
| 4-thread run-to-run bitwise identical | pass |
| 1-thread vs 4-thread agree | 9.1e-15 |
| Covariance == (AᵀA + SᵀS)⁻¹ | 1.4e-16 |
| Schur-marginal nav cov == dense full-inverse nav block | 3.6e-8 |
| Warm-start joint slice (newest clone ≡ IMU pose, bit-identical) + realign/inflate congruence | 20/20 |

### On-target performance vs Ceres (previously measured on QRB5165; not re-run this pass)

`bench_init` builds the SAME VI-init factor graph and solves it with Ceres and with
`zbft_sfm`, single-threaded, on the QRB5165 over 30 trials — same minimum (max final-cost
rel. diff 1.2e-5). LM **55.7 ms vs 73.5 ms (0.76×)**, dogleg **50.6 ms vs 72.0 ms (0.70×)**.
Reproduce: `adb push` the cross-built `bench_init` and run `./bench_init 30 1 0 0`.

## Standalone validation binaries (manual build — NOT wired into CMake)

These live in `ov_init/src/` (+ `ceres_free/`) and are compiled by hand (recipes in each
file header), not by the CMake build:

| File | Purpose | Re-run this pass |
|------|---------|------------------|
| `ceres_free/test_mini_solver.cpp` | Eigen-only solver-core self-test | ✅ 13/13 |
| `ceres_free/test_warmstart_cov.cpp` | Warm-start joint-covariance / realign / inflation | ✅ 20/20 |
| `ceres_free/test_mini_factors.cpp` | FD checks of lifted factors vs `ov_core` | not run |
| `test_init_consistency.cpp` | Monte-Carlo **NEES** consistency gold standard (S² gravity) | ⚠️ ran — see below |
| `test_init_ab_compare.cpp` | Ceres vs ceres-free A/B on identical synthetic window | not run |
| `bench_zbft_s2.cpp` | S²-gravity MLE micro-benchmark | not run |

## Current status (code-accurate, honest)

**Done and default:**
- Ceres-free LM/Schur solver core, lock-free pool, covariance — verified (13/13, 20/20 this pass).
- S²-gravity MLE, full-orientation gauge prior, gravity gate + conditional re-align,
  covariance-at-optimum + similarity carry-through — implemented and `ON` by default.
- Warm-start joint-covariance injection (`DynamicInitializer` ↔ `StateHelper::set_initial_state_warmstart`
  ↔ `VioManagerHelper`) — implemented; enabled in the shipped `fpv` config.

**Open / not-yet-true (do not read the "Done" list as production-signed-off):**
- **Statistical consistency is NOT yet in band.** `test_init_consistency` (500 trials, inflation
  OFF, free-S²) reports **ANEES/15 ≈ 26** (ideal 1.0): the raw Laplace covariance is
  overconfident, dominated by accel-bias (per-block NEES ba ≈ 285), velocity (≈ 59) and
  position (≈ 16). The shipped config compensates with aggressive inflation
  (`init_dyn_inflation_vel: 150`, `ba: 150`, `ori: 30`) — but **position is still uninflated**
  (`init_dyn_inflation_pos` default 1.0) despite being ~5× overconfident. This inflation is a
  band-aid, not a consistency fix; it should be tuned on real reset logs.
- **Gravity flip-rejection gate is leaky.** Same test: **42/50** upside-down seeds rejected (want
  50/50). Eight flipped seeds passed the 30° gate in synthetic data.
- **Ceres not removed.** Still `find_package(Ceres REQUIRED)` + `voxl-ceres-solver` + dual-built
  fallback. Removal (delete `src/ceres/*`, drop `find_package`/`${CERES_LIBRARIES}`/`voxl-ceres-solver`/
  `libceres-dev`) is deliberately deferred until on-hardware A/B + consistency pass.
- Validation binaries are manual-build only (not in CI); `init_dyn_mle_opt_calib` calib
  injection is still a TODO in the injection block.

**Removal step (once consistency + on-hardware A/B pass):** flip nothing (already default), then
delete `src/ceres/*`, `find_package(Ceres)`, `${CERES_LIBRARIES}`, `voxl-ceres-solver` in
`install_build_deps.sh`, and `libceres-dev` in `ov_msckf/package.xml`.
