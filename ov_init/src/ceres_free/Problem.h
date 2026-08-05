/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * Ceres-free initialization backend (ov_init::zbft_sfm)
 * ------------------------------------------------
 * A small, deterministic, real-time-friendly nonlinear least-squares solver that
 * hosts the lifted analytic factors. It is NOT a general graph optimizer; it is
 * purpose-built for the VI-initialization problem:
 *
 *   - Levenberg-Marquardt over manifold parameter blocks (JPL quat + Euclidean).
 *   - Optional Schur complement of "landmark" blocks (the BA arrowhead), exploiting
 *     that each reprojection residual touches exactly one landmark (block-diagonal
 *     reduced landmark Hessian, inverted as independent 3x3 blocks).
 *   - Parallel per-thread Hessian/gradient accumulation with a deterministic,
 *     worker-ordered reduction (serial result == multi-threaded result, bitwise).
 *   - Bounded: hard iteration cap AND wall-clock budget; falls back to the best
 *     accepted iterate. No dynamic allocation inside the inner LM retry loop.
 *   - Marginal covariance recovered directly from the gauge-anchored reduced
 *     Hessian (landmarks marginalized) -- no separate covariance machinery.
 *
 * Ownership: the Problem does NOT own the CostFunction/LossFunction/
 * LocalParameterization pointers or the parameter-block memory; the caller keeps
 * them alive across Solve()/ComputeCovariance().
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_ZBFT_SFM_PROBLEM_H
#define OV_INIT_ZBFT_SFM_PROBLEM_H

#include <Eigen/Dense>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "CostFunction.h"
#include "LocalParameterization.h"
#include "LossFunction.h"

namespace ov_init {
namespace zbft_sfm {

struct SolverOptions {
  int max_num_iterations = 30;           ///< max accepted steps
  double max_solver_time_seconds = 0.05; ///< hard wall-clock budget (mirrors init_dyn_mle_max_time)
  int num_threads = 4;                   ///< accumulation workers; 1 => fully inline/deterministic (the RT default)

  /// Optional per-worker thread setup (CPU affinity / scheduling class) so workers
  /// never preempt the IMU/camera real-time threads. Called once per spawned worker.
  std::function<void(int worker_index)> worker_init_fn;

  // Levenberg-Marquardt damping (Madsen-Nielsen nu-update). Used only when use_dogleg=false.
  double initial_lambda = 1e-4; // = 1/initial_radius; matches Ceres' default initial_trust_region_radius=1e4
  double min_lambda = 1e-12;
  double max_lambda = 1e12;

  // Rejected-trial escalation: lambda *= nu; nu *= lm_nu_growth (Ceres LevenbergMarquardtStrategy
  // uses growth 2.0). On problems with a flat-valley regime change (the free-S2 gravity <->
  // accel-bias ambiguity) the Ceres schedule first lets lambda crash ~5 decades during the easy
  // phase (bounded below by min_lambda), then pays ~9 REJECTED trials -- each a full Schur solve
  // + residual pass -- climbing back. Raising min_lambda (e.g. 1e-7) bounds the crash and a larger
  // lm_nu_growth (e.g. 4.0) accelerates the climb. Defaults preserve Ceres-exact behavior.
  double lm_nu_growth = 2.0;

  // Powell dogleg trust region (default; the per-iteration win: ONE factorization per
  // linearization, rejected trials are cheap GN/Cauchy blends). use_dogleg=false -> LM.
  bool use_dogleg = false;
  double initial_radius = 1e4; // matches Ceres' DoglegStrategy default
  double max_radius = 1e16;

  double min_relative_decrease = 1e-3; ///< accept a step only if actual/predicted > this (matches Ceres)

  // Convergence tolerances
  double function_tolerance = 1e-6;  ///< relative cost decrease
  double gradient_tolerance = 1e-10; ///< max-norm of gradient
  double parameter_tolerance = 1e-8; ///< step 2-norm

  bool verbose = false;
  bool pin_eigen_single_thread = false; ///< Eigen::setNbThreads(1) during solve
};

struct SolverSummary {
  int iterations = 0;       ///< accepted LM/dogleg steps
  int successful_steps = 0; ///< == iterations that decreased the cost (kept for Ceres-summary parity)
  int rejected_steps = 0;   ///< trial steps rejected (damping increased; no Jacobian recomputed)
  int jacobian_evals = 0;   ///< full linearizations (residual + Jacobian + Hessian)
  int residual_evals = 0;   ///< residual-only (cost) evaluations
  double initial_cost = 0.0;
  double final_cost = 0.0;
  double solve_time_seconds = 0.0;
  // Wall-clock split of solve_time_seconds (timing instrumentation; ~zero overhead):
  double time_linearize_seconds = 0.0;    ///< residual+Jacobian+Hessian accumulation passes
  double time_linear_solve_seconds = 0.0; ///< damped (Schur) linear solves, incl. rejected trials
  double time_residual_seconds = 0.0;     ///< residual-only trial scoring (evaluate_cost)
  bool converged = false;
  /// True iff the wall-clock budget ended the solve (any of the in-loop checks).
  /// A binding time cap couples machine load into the ITERATE -- callers that
  /// need run-to-run bit-identity must treat a time-stopped solve as tainted
  /// and surface this flag (ov_zcalib evidence table does).
  bool time_stopped = false;
  std::string message;
};

class ParallelExecutor; // fwd

class Problem {
public:
  Problem() = default;
  virtual ~Problem(); // virtual: Problem is the reusable ceres-free base (ov_zcalib derives)

  /**
   * @brief Opt in to Ceres-style memory ownership: when enabled, the Problem
   * deletes the CostFunction / LossFunction / LocalParameterization pointers it
   * was given (deduplicated) on destruction. Off by default (callers that pass
   * stack-allocated factors must leave it off). Enable it to make this a drop-in
   * for `ceres::Problem`, which takes ownership of those objects.
   */
  void EnableOwnership(bool on = true) { owns_ = on; }

  /**
   * @brief Register a parameter block.
   * @param values      pointer to the block's ambient values (caller-owned, length global_size)
   * @param global_size ambient dimension (e.g. 4 for a JPL quaternion)
   * @param param       manifold parameterization, or nullptr for Euclidean (local==global)
   * @return internal block index
   */
  int AddParameterBlock(double *values, int global_size, const LocalParameterization *param = nullptr);

  void SetParameterBlockConstant(double *values);
  void SetParameterBlockVariable(double *values);

  /// Tag a (already-added) block as a Schur-eliminated landmark (typically 3-dof).
  void SetSchurLandmark(double *values);

  /// Add a residual block. loss==nullptr means squared (trivial) loss.
  void AddResidualBlock(const CostFunction *cost, const LossFunction *loss, const std::vector<double *> &blocks);

  /// Run the bounded LM solve. Returns a summary; parameter blocks are updated in place.
  SolverSummary Solve(const SolverOptions &options);

  /**
   * @brief Marginal covariance of the requested blocks (landmarks marginalized out),
   *        assembled in the requested order, in LOCAL (error-state) coordinates.
   *        Requires the gauge to be anchored (e.g. yaw/position priors) so the
   *        reduced Hessian is positive definite; returns false otherwise.
   *        All requested blocks must be non-landmark, non-constant.
   */
  bool ComputeCovariance(const std::vector<double *> &blocks, Eigen::MatrixXd &covariance, const SolverOptions &options);

  /**
   * @brief Reduced information + gradient of the requested blocks at the CURRENT iterate.
   *
   * Linearizes once (undamped) and marginalizes EVERY other variable block -- landmarks
   * and nuisances alike: Lambda = H_kk - H_kn H_nn^-1 H_nk and the matching reduced
   * gradient g_k - H_kn H_nn^-1 g_n of the robustified cost, in LOCAL (error-state)
   * coordinates, requested-block order. This is the per-window export of the VarPro
   * calibration fusion: the global step solves (sum Lambda_w + prior) dp = -(sum g_w).
   * Requires the nuisance system to be PD (per-window gauge anchored); returns false
   * otherwise. Additive API: Solve()/ComputeCovariance() behavior is unchanged.
   */
  /// Optional per-export convergence evidence (stationarity certificate).
  /// nuis_decrement = g_z' H_zz^{-1} g_z summed over ALL marginalized blocks
  /// (landmark + nav-nuisance terms; exact block-elimination identity) -- the
  /// model energy of the remaining inner correction: 2x the cost decrease a
  /// further inner solve could still achieve at this linearization. Computed
  /// from factorizations this export already forms (one extra O(nn^2) solve).
  struct ExportStats {
    double nuis_decrement = 0.0;
    double land_decrement = 0.0;
    double nuis_grad_inf = 0.0;
    // Veto diagnostics only: the nuisance LDLT's smallest pivot and dimension.
    // NaN pivot = factorization itself failed. Never consumed by solver logic;
    // excluded from the export-audit memcmp by name.
    double nuis_min_pivot = 0.0;
    int nuis_dim = 0;
    /// landmark directions rank-clamped by the spectral landmark elimination:
    /// the window's degenerate-landmark load. Diagnostic only.
    int clamped_dirs = 0;
  };
  // NOTE (export-on-accept): a "qn-only" variant of this export (empty keep
  // set, calibration columns never formed) was built and MEASURED as NOT
  // byte-equal: the smaller leading dimension of H (692 vs 722 on the probe
  // window) shifts column base alignment mod 32 bytes, and under -ffast-math
  // the SIMD head-peeling reassociates the accumulations -- H_zz drifts ~1 ulp
  // and q_n follows (~1e-10 rel). q_n feeds thresholded duel arbitration, so
  // cert-consuming evaluations keep the full export (see ov_zcalib JointCalib).
  // Do not resurrect the qn-only path without emitted-loop byte parity first.
  bool ExportReducedInformation(const std::vector<double *> &blocks, Eigen::MatrixXd &Lambda, Eigen::VectorXd &gred,
                                const SolverOptions &options, ExportStats *stats = nullptr);

protected: // protected (not private): the base-class seam for derived solvers (ov_zcalib)
  struct Block {
    double *data = nullptr;
    int gsize = 0;
    int lsize = 0;
    const LocalParameterization *param = nullptr;
    bool constant = false;
    bool landmark = false;
    bool tangent_leading_identity = true; // true iff V=[I;0], so J_local = J_ambient.leftCols(lsize)
    int offset = -1; // local offset in the assembled error-state vector (variable blocks only)
  };

  struct Residual {
    const CostFunction *cost = nullptr;
    const LossFunction *loss = nullptr;
    std::vector<int> blocks; // indices into blocks_
  };

  int block_index(double *values) const;
  void assign_ordering();                                    // fills offsets, n_nav_, n_land_, n_total_, land_diag_
  double evaluate_cost(ParallelExecutor &exec) const;        // 0.5 * sum rho(||r||^2) at current x (parallel, worker-ordered reduction)
  void linearize(Eigen::MatrixXd &H, Eigen::VectorXd &grad, double &cost, // GN Hessian + gradient + robustified cost, one pass
                 ParallelExecutor &exec) const;
  /// Solve (H + lambda*diag(H)) delta = -grad. Uses the visibility-aware arrowhead Schur
  /// complement when landmark blocks are present, else a plain damped dense Cholesky.
  bool solve_step(const Eigen::MatrixXd &H, const Eigen::VectorXd &grad, double lambda, Eigen::VectorXd &delta) const;
  void apply_delta(const Eigen::VectorXd &delta);    // x <- x (+) delta
  double snapshot(std::vector<double> &backup) const; // save variable-block ambient values; returns ||x|| (Ceres x_norm)
  void restore(const std::vector<double> &backup);   // restore variable-block ambient values

  std::vector<Block> blocks_;
  std::vector<Residual> residuals_;
  std::unordered_map<double *, int> index_;

  // Optional Ceres-style ownership of added factors/params (see EnableOwnership).
  bool owns_ = false;
  std::vector<const CostFunction *> owned_cost_;
  std::vector<const LossFunction *> owned_loss_;
  std::vector<const LocalParameterization *> owned_param_;

  // Preallocated solver scratch -- REAL-TIME: no heap allocation in the iteration loop.
  // (On aarch64/glibc, a fresh n_total x n_total MatrixXd each trial exceeds the 128 KB
  //  mmap threshold and would trigger mmap/munmap syscalls + page-zeroing every solve.)
  mutable std::vector<Eigen::MatrixXd> Hw_;    // per-worker Hessian accumulators
  mutable std::vector<Eigen::VectorXd> gw_;    // per-worker gradient accumulators
  mutable std::vector<double> cw_;             // per-worker cost accumulators
  mutable Eigen::MatrixXd Hd_, Hred_;          // damped full H (landmark-free path), reduced nav H (Schur path)
  mutable Eigen::VectorXd rhs_, da_;           // reduced rhs and nav step
  // (The Cholesky runs IN-PLACE on Hd_/Hred_ via LLT<Ref<MatrixXd>> -- no factor-storage copy per trial.)
  mutable Eigen::VectorXd dgn_, Hg_, Hdl_;     // dogleg: Gauss-Newton step, H*grad, the trust-region step
  mutable Eigen::VectorXd lm_delta_, lm_Dvec_; // LM: preallocated step and Marquardt damping diagonal

  // Arrowhead-Schur scratch (reused; no per-call heap traffic): for one landmark's adjacency,
  // schur_off_[k] = nav offset of adjacent block k, schur_W_[k] = H[landmark, off_k] (3 x lsize_k),
  // schur_Ma_[k] = W_k^T * V^-1 (lsize_k x 3). GENERAL block widths: an adjacent block may be a
  // clone pose (3) or an unlocked calibration block (e.g. 8-dof camera intrinsics) -- a fixed
  // Matrix3d scratch would silently drop every column beyond the third.
  mutable std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> schur_W_;
  mutable std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> schur_Ma_;
  mutable std::vector<Eigen::Matrix3d> schur_Vinv_; // per-landmark damped V^-1 (Schur pass -> back-substitution)
  mutable std::vector<int> schur_off_;

  // FIXED-SIZE (3x3) Schur scratch -- the fast path.
  //
  // In the inner solve EVERY block a landmark touches is a clone pose (q or p), so every
  // adjacency has lsize == 3, yet with the Dynamic scratch above the O(P^2) fill-in
  // `Hred.block(...) -= Ma * W_b` compiles to Eigen's dynamic gemm: runtime dispatch
  // wrapped around 54 flops. PROFILED on one production session record: 73.5 million such
  // products at 1.42 GFLOP/s (~2% of machine peak), 36% of solve_step's wall, while the
  // dense Cholesky beside it runs at 58% of peak -- the loop pays dispatch, not arithmetic.
  // (Ceres template-specializes its SchurEliminator on block shape for the same reason.)
  // Fixed-size buffers give the compiler 3x3 at compile time: full unroll, SIMD, no
  // dispatch. The Dynamic path stays for the EXPORT, where the calibration blocks are
  // variable and adjacencies really can be 1 (td), 3 (quat/pos) or 8 (camera) wide.
  mutable std::vector<Eigen::Matrix3d> schur_W3_, schur_Ma3_;
  std::vector<char> land_all3_; // per landmark: every adjacency is lsize 3 (=> fixed-size path)

  // ---- BORDERED-BANDED direct solver for the reduced nav system (analyze_band) ----
  //
  // The reduced nav Hessian is banded with a low-rank border, not dense:
  //   * IMU preintegration couples clone k to clone k+1 only           -> half-bandwidth 30
  //   * a landmark's Schur fill cliques the clones that SEE it         -> 15 * max_track_len
  //   * the S2 gravity block couples to EVERY clone and is registered LAST (WindowBA.cpp),
  //     landing at the end of the nav partition                        -> a rank-2 BORDER.
  //
  // Whether that is worth exploiting is a property of the WINDOW SHAPE: at 70 clones /
  // 40-obs tracks the fill reaches ~600 of 1052 dofs (57% dense) and banded buys nothing --
  // rejecting it there was correct. The flight shape caps tracks at 10 obs / 32 clones,
  // where the MEASURED half-bandwidth is 74 of 467 (16%): n*b^2 = 2.6 MFLOP vs n^3/3 =
  // 33.9 MFLOP, 13x fewer. So the choice is never hard-coded to a profile: analyze_band()
  // measures the actual fill and enables the banded path ONLY when it is cheaper; long-track
  // windows fall back to the dense LLT automatically. The band is also 280 KB instead of
  // 1.75 MB -- it fits in L2, which matters more on the target than the flop count does.
  mutable Eigen::MatrixXd band_AB_; // (b+1) x nb, lower band storage: AB(k,j) = B(j+k, j)
  mutable Eigen::MatrixXd band_C_;  // nb x nbord   border columns
  mutable Eigen::MatrixXd band_Y_;  // nb x nbord   = B^-1 C
  mutable Eigen::MatrixXd band_S_;  // nbord x nbord Schur complement of the border
  mutable Eigen::VectorXd band_z_;  // nb           scratch
  int band_half_ = 0;               ///< half-bandwidth of the leading banded block
  int band_nb_ = 0;                 ///< size of the leading banded block
  int band_border_ = 0;             ///< size of the trailing dense border (0 = pure band)
  bool use_banded_ = false;         ///< analyze_band()'s verdict for THIS window's structure

  /// Measure the reduced nav system's fill from the residual + landmark structure and decide
  /// whether the bordered-banded path beats the dense Cholesky. Structure-only: called once
  /// per ordering, never per trial.
  void analyze_band();
  std::vector<double> plus_tmp_; // apply_delta Plus() scratch, sized once to the max ambient block size

  // Diagnostics (reset each Solve), surfaced in SolverSummary: linearization / cost-eval counts.
  mutable int n_jac_evals_ = 0;
  mutable int n_res_evals_ = 0;

  // Ordering (computed in assign_ordering)
  int n_nav_ = 0;   // total local dim of navigation (non-landmark) variable blocks
  int n_land_ = 0;  // total local dim of landmark variable blocks
  int n_total_ = 0; // n_nav_ + n_land_
  std::vector<std::pair<int, int>> land_diag_; // (offset_within_land_partition, lsize) per landmark block
  std::vector<int> land_block_idx_;            // blocks_ index of each landmark (parallel to land_diag_)
  std::vector<std::vector<int>> land_adj_;     // per landmark: adjacent nav block indices (its observing poses)

  // LOWER-TRIANGLE Hessian storage: H (and the per-worker accumulators) only ever hold the
  // lower triangle of the used sparsity -- column j holds rows [j, col_tail_end_[j]). For nav
  // columns the tail runs to n_total_ (nav-nav lower + the landmark W^T strip); for a landmark
  // column it ends at its own 3x3 diagonal block (landmark-landmark coupling is structurally
  // zero). Everything above/outside is NEVER written, zeroed, reduced, or read -- that cuts the
  // dominant zero+reduce memory traffic of linearize() by ~2.5x and halves the scatter writes.
  // Consumers read via selfadjoint/transposed-lower access (see solve_step/ComputeCovariance).
  std::vector<int> col_tail_end_; // per column: one-past-the-end row of the used lower tail
};

} // namespace zbft_sfm
} // namespace ov_init

#endif // OV_INIT_ZBFT_SFM_PROBLEM_H
