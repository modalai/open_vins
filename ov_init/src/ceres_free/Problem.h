/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Contributor: Joao Leonardo Silva Cotta (@zauberflote1)
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
  bool converged = false;
  std::string message;
};

class ParallelExecutor; // fwd

class Problem {
public:
  Problem() = default;
  ~Problem();

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

private:
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
  void assign_ordering();                                   // fills offsets, n_nav_, n_land_, n_total_, land_diag_
  double evaluate_cost() const;                             // 0.5 * sum rho(||r||^2) at current x
  void linearize(Eigen::MatrixXd &H, Eigen::VectorXd &grad, double &cost, // GN Hessian + gradient + robustified cost, one pass
                 ParallelExecutor &exec) const;
  /// Solve (H + lambda*diag(H)) delta = -grad. Uses the visibility-aware arrowhead Schur
  /// complement when landmark blocks are present, else a plain damped dense Cholesky.
  bool solve_step(const Eigen::MatrixXd &H, const Eigen::VectorXd &grad, double lambda, Eigen::VectorXd &delta) const;
  void apply_delta(const Eigen::VectorXd &delta); // x <- x (+) delta
  void snapshot(std::vector<double> &backup) const; // save variable-block ambient values
  void restore(const std::vector<double> &backup);  // restore variable-block ambient values

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
  mutable Eigen::LLT<Eigen::MatrixXd> lltRed_; // Cholesky (SPD): faster than LDLT; the reduced H is PD via the gauge prior
  mutable Eigen::VectorXd dgn_, Hg_, Hdl_;     // dogleg: Gauss-Newton step, H*grad, the trust-region step
  mutable Eigen::VectorXd lm_delta_, lm_Dvec_; // LM: preallocated step and Marquardt damping diagonal

  // Arrowhead-Schur scratch (reused; no per-call heap traffic): for one landmark's adjacency,
  // schur_off_[k] = nav offset of pose block k, schur_W_[k] = H[off_k, landmark], schur_Ma_[k] = W_k * V^-1.
  mutable std::vector<Eigen::Matrix3d> schur_W_, schur_Ma_;
  mutable std::vector<int> schur_off_;

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
};

} // namespace zbft_sfm
} // namespace ov_init

#endif // OV_INIT_ZBFT_SFM_PROBLEM_H
