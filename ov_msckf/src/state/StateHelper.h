/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_STATE_HELPER_H
#define OV_MSCKF_STATE_HELPER_H

#include <Eigen/Eigen>
#include <map>
#include <memory>
#include "utils/InitializerPhysicalWarmResult.h"
#include "State.h"

namespace ov_type {
class Type;
class PoseJPL;
} // namespace ov_type

namespace ov_msckf {

class State;

/**
 * @brief Helper which manipulates the State and its covariance.
 *
 * In general, this class has all the core logic for an Extended Kalman Filter (EKF)-based system.
 * This has all functions that change the covariance along with addition and removing elements from the state.
 * All functions here are static, and thus are self-contained so that in the future multiple states could be tracked and updated.
 * We recommend you look directly at the code for this class for clarity on what exactly we are doing in each and the matching documentation
 * pages.
 */
class StateHelper {

public:
  /**
   * @brief Deep-copy an entire State into an independent new State.
   *
   * Produces a fully decoupled snapshot: the returned state shares no mutable object with the
   * source, so the source can keep evolving (or the copy can be fed forward on a branch) without
   * cross-contamination. Preserves the exact covariance (_Cov verbatim) and variable ordering
   * (each cloned variable keeps its local id), rebuilds every state map (_imu, _clones_IMU,
   * _features_SLAM, calib, intrinsics) to point at the cloned variables, preserves the
   * _calib_dt_CAMtoIMU alias into _calib_dt_CAMtoIMU_map, deep-copies the camera intrinsic
   * objects, and copies all non-covariance metadata (_clones_kinematics, _epoch_residuals,
   * _epoch_bridges, _timestamp, _options, _kin_miss_count).
   *
   * This is the core primitive behind the replay harness's snapshot / rewind / branch. It has
   * private access to _Cov and _variables, which is why it lives on StateHelper.
   *
   * @param state Source state to copy
   * @return A newly allocated, independent deep copy
   */
  static std::shared_ptr<State> clone_state(std::shared_ptr<State> state);

  /// Explicit ownership-only setup; no runtime option/caller enables sampled
  /// propagation yet. Allocates two persistent zero Vec6 slots once. Repeating
  /// with the same nonzero stream is a no-op; a different stream is rejected.
  static bool prepare_sampled_imu_boundary(std::shared_ptr<State> state, uint64_t stream_episode);

  /// First use is independent of all existing variables and starts at zero
  /// mean with the full finite symmetric PSD raw-axis prior (singular allowed).
  /// Active identical repeats return the same posterior without resetting it.
  /// Changed/retired identities, nonmonotonic records, and a third live sample
  /// are rejected without mutation. Admission/retirement reuse fixed slots.
  static std::shared_ptr<ov_type::Vec> admit_sampled_imu_noise(std::shared_ptr<State> state,
                                                             const State::SampledImuRecord &record);

  /// Integrate out the older of two live samples at the exact successor's raw
  /// knot, already accepted in State. The caller must have completed all direct
  /// factors/outputs using the retired sample. Other means and covariance
  /// blocks remain verbatim; the vacated permanent slot becomes deterministic
  /// zero. Generic marginalize cannot remove these reserved coordinates.
  static bool retire_sampled_imu_noise_at_knot(std::shared_ptr<State> state, uint64_t sequence);

  static bool valid_sampled_imu_record(const State::SampledImuRecord &record);

  /// Disabled-runtime sampled propagation transaction. Stages admission of
  /// both original records and the augmented covariance product before any
  /// mutation. Fixed IMU calibration only: Phi columns are [IMU15, left6, right6].
  /// Invalid records, nonfinite arithmetic or invalid independent Q preserve
  /// covariance, sample means/metadata and admission watermarks.
  static bool EKFPropagationSampled(std::shared_ptr<State> state,
                                    const std::array<State::SampledImuRecord, 2> &records,
                                    const Eigen::MatrixXd &Phi, const Eigen::MatrixXd &Q);

  /**
   * @brief Performs EKF propagation of the state covariance.
   *
   * The mean of the state should already have been propagated, thus just moves the covariance forward in time.
   * The new states that we are propagating the old covariance into, should be **contiguous** in memory.
   * The user only needs to specify the sub-variables that this block is a function of.
   * \f[
   * \tilde{\mathbf{x}}' =
   * \begin{bmatrix}
   * \boldsymbol\Phi_1 &
   * \boldsymbol\Phi_2 &
   * \boldsymbol\Phi_3
   * \end{bmatrix}
   * \begin{bmatrix}
   * \tilde{\mathbf{x}}_1 \\
   * \tilde{\mathbf{x}}_2 \\
   * \tilde{\mathbf{x}}_3
   * \end{bmatrix}
   * +
   * \mathbf{n}
   * \f]
   *
   * @param state Pointer to state
   * @param order_NEW Contiguous variables that have evolved according to this state transition
   * @param order_OLD Variable ordering used in the state transition
   * @param Phi State transition matrix (size order_NEW by size order_OLD)
   * @param Q Additive state propagation noise matrix (size order_NEW by size order_NEW)
   * @return False before changing covariance on invalid ownership, dimensions
   * or arithmetic. The caller owns mean staging/rollback. No full state-sized
   * covariance copy is made; allocation failure/concurrent mutation are excluded.
   */
  static bool EKFPropagation(std::shared_ptr<State> state, const std::vector<std::shared_ptr<ov_type::Type>> &order_NEW,
                             const std::vector<std::shared_ptr<ov_type::Type>> &order_OLD, const Eigen::MatrixXd &Phi,
                             const Eigen::MatrixXd &Q);

  /**
   * @brief Performs EKF update of the state (see @ref linear-meas page)
   * The optional temporal excitation gate applies a Schmidt update: full measurement columns
   * remain present, temporal means and their joint prior covariance stay fixed, and their
   * cross-covariance with updated variables follows the ordinary posterior.
   * @param state Pointer to state
   * @param H_order Variable ordering used in the compressed Jacobian
   * @param H Condensed Jacobian of updating measurement
   * @param res Residual of updating measurement
   * @param R Updating measurement covariance (PSD; the innovation must be SPD)
   * @return False on invalid arithmetic with live means, covariance and FEJ unchanged.
   * Allocation failure and concurrent state mutation are outside this contract.
   */
  static bool EKFUpdate(std::shared_ptr<State> state, const std::vector<std::shared_ptr<ov_type::Type>> &H_order, const Eigen::MatrixXd &H,
                        const Eigen::VectorXd &res, const Eigen::MatrixXd &R);

  /**
   * @brief This will set the initial covaraince of the specified state elements.
   * Will also ensure that proper cross-covariances are inserted.
   * @param state Pointer to state
   * @param covariance The covariance of the system state
   * @param order Order of the covariance matrix
   */
  static void set_initial_covariance(std::shared_ptr<State> state, const Eigen::MatrixXd &covariance,
                                     const std::vector<std::shared_ptr<ov_type::Type>> &order);

  /// Validate a recovered navigation covariance before handing it to the filter.
  /// Requires finite entries, positive diagonal, symmetry and PSD to roundoff.
  /// Exact singular joint covariances are permitted; no jitter is introduced.
  /// This is an initialization boundary check, not a per-update factorization.
  static bool valid_initial_covariance(const Eigen::Ref<const Eigen::MatrixXd> &covariance);
  static bool valid_initial_covariance(const Eigen::Ref<const Eigen::MatrixXd> &covariance, bool allow_zero_diagonal);

  /// Snapshot an empty physical episode's complete camera consider prior.
  /// Camera values must match FEJ. Nonzero bias/calibration cross covariance
  /// requires the exact immutable reset snapshot; other navigation cross terms
  /// are unsupported. This never changes state or a failed output.
  static bool make_initial_physical_warm_request(std::shared_ptr<State> state, uint64_t episode_id,
                                                ov_core::InitPhysicalWarmRequest &request,
                                                std::shared_ptr<const ov_core::InitPhysicalResetPrior> reset_prior = {});

  /// Stage a new physical reset episode and its complete bias/camera marginal.
  /// Retains camera values/Pcc, explicitly rebases FEJ, and never changes the
  /// old state or either failed output. Raw support provenance is caller-owned.
  static bool make_physical_reset_state(std::shared_ptr<State> state, uint64_t snapshot_id,
                                        const std::vector<double> &raw_watermarks, double imu_raw_cutoff,
                                        const Eigen::Matrix<double,6,1> &bias_rw_variance, int cause,
                                        std::shared_ptr<State> &replacement,
                                        std::shared_ptr<const ov_core::InitPhysicalResetPrior> &prior);

  /// Atomically install a bounded physical initializer result.
  /// Requires a fresh empty physical state and the caller's current nonzero
  /// episode. No augmentation/process Q is added. Consume the exact image
  /// receipt only after success. Camera consider priors retain their complete
  /// covariance and cross terms. A joint reset requires the caller's identical
  /// snapshot and new likelihood boundary; fitted calibration means are unsupported.
  static bool set_initial_state_physical_warm(std::shared_ptr<State> state,
                                             const ov_core::InitPhysicalWarmResult &result,
                                             uint64_t expected_episode_id,
                                             std::shared_ptr<const ov_core::InitPhysicalResetPrior> expected_reset_prior = {});

  /**
   * @brief Warm-start seed: inject the active IMU state **and** the initializer's window clones with
   * their full joint covariance. Subsequent updates may use only observations not already
   * assimilated by that initializer posterior.
   *
   * The dynamic initializer recovers the landmark-marginalized navigation covariance over the IMU and
   * every window clone. This grows the covariance once, registers each clone (id, _variables,
   * _clones_IMU), and copies the joint covariance (diagonals **and** cross-terms) into place. Contract:
   * @p covariance is in LOCAL error coordinates ordered `[IMU(15) | clone_0(6) | clone_1(6) | ...]`
   * with clones in ASCENDING time -- exactly @p clones_IMU's iteration order. The IMU must be the first
   * covariance block (id 0) and the clones must be valued but not yet present in the state.
   *
   * @param state Pointer to state (its _imu is assumed already valued by the initializer)
   * @param covariance Joint covariance, ordered [IMU, clones-ascending]
   * @param clones_IMU Window clone poses (valued), keyed by imaging time (ascending)
   * @return True on success; false without mutation on an invalid ownership or finite/PSD contract.
   */
  static bool set_initial_state_warmstart(std::shared_ptr<State> state, const Eigen::MatrixXd &covariance,
                                          const std::map<double, std::shared_ptr<ov_type::PoseJPL>> &clones_IMU);

  /**
   * @brief For a given set of variables, this will this will calculate a smaller covariance.
   *
   * That only includes the ones specified with all crossterms.
   * Thus the size of the return will be the summed dimension of all the passed variables.
   * Normal use for this is a chi-squared check before update (where you don't need the full covariance).
   *
   * @param state Pointer to state
   * @param small_variables Vector of variables whose marginal covariance is desired
   * @return Marginal covariance of the passed variables
   */
  static Eigen::MatrixXd get_marginal_covariance(std::shared_ptr<State> state,
                                                 const std::vector<std::shared_ptr<ov_type::Type>> &small_variables);

  /**
   * @brief This gets the full covariance matrix.
   *
   * Should only be used during simulation as operations on this covariance will be slow.
   * This will return a copy, so this cannot be used to change the covariance by design.
   * Please use the other interface functions in the StateHelper to progamatically change to covariance.
   *
   * @param state Pointer to state
   * @return Covariance of current state
   */
  static Eigen::MatrixXd get_full_covariance(std::shared_ptr<State> state);

  /// Read-only fixed output projection over [IMU15, slot0 noise6, slot1 noise6].
  /// Inactive noise columns must be zero. State covariance is the valid prior;
  /// singular output covariance is permitted, with floating-point roundoff.
  /// Optional state_cross must already be n by 12; no caller output is resized.
  /// Numerical/ownership/dimension refusal preserves all outputs and State.
  /// Fixed marginal scratch; optional full cross uses O(n times 12) scratch.
  static bool project_sampled_imu_output(std::shared_ptr<State> state, const Eigen::Matrix<double, 12, 27> &H,
                                         Eigen::Matrix<double, 12, 12> &covariance,
                                         Eigen::MatrixXd *state_cross = nullptr);

  /**
   * @brief Marginalizes a variable, properly modifying the ordering/covariances in the state
   *
   * This function can support any Type variable out of the box.
   * Right now the marginalization of a sub-variable/type is not supported.
   * For example if you wanted to just marginalize the orientation of a PoseJPL, that isn't supported.
   * We will first remove the rows and columns corresponding to the type (i.e. do the marginalization).
   * After we update all the type ids so that they take into account that the covariance has shrunk in parts of it.
   *
   * @param state Pointer to state
   * @param marg Pointer to variable to marginalize
   */
  static void marginalize(std::shared_ptr<State> state, std::shared_ptr<ov_type::Type> marg);

  /**
   * @brief Clones "variable to clone" and places it at end of covariance
   * @param state Pointer to state
   * @param variable_to_clone Pointer to variable that will be cloned
   */
  static std::shared_ptr<ov_type::Type> clone(std::shared_ptr<State> state, std::shared_ptr<ov_type::Type> variable_to_clone);

  /**
   * @brief Initializes new variable into covariance.
   *
   * Uses Givens to separate into updating and initializing systems (therefore system must be fed as isotropic).
   * If you are not isotropic first whiten your system (TODO: we should add a helper function to do this).
   * If your H_L Jacobian is already directly invertable, the just call the initialize_invertible() instead of this function.
   * Please refer to @ref update-delay page for detailed derivation.
   *
   * @param state Pointer to state
   * @param new_variable Pointer to variable to be initialized
   * @param H_order Vector of pointers in order they are contained in the condensed state Jacobian
   * @param H_R Jacobian of initializing measurements wrt variables in H_order
   * @param H_L Jacobian of initializing measurements wrt new variable
   * @param R Covariance of initializing measurements (isotropic)
   * @param res Residual of initializing measurements
   * @param chi_2_mult Value we should multiply the chi2 threshold by (larger means it will be accepted more measurements)
   */
  static bool initialize(std::shared_ptr<State> state, std::shared_ptr<ov_type::Type> new_variable,
                         const std::vector<std::shared_ptr<ov_type::Type>> &H_order, Eigen::MatrixXd &H_R, Eigen::MatrixXd &H_L,
                         Eigen::MatrixXd &R, Eigen::VectorXd &res, double chi_2_mult);

  /**
   * @brief Initializes new variable into covariance (H_L must be invertible)
   *
   * Please refer to @ref update-delay page for detailed derivation.
   * This is just the update assuming that H_L is invertable (and thus square) and isotropic noise.
   *
   * @param state Pointer to state
   * @param new_variable Pointer to variable to be initialized
   * @param H_order Vector of pointers in order they are contained in the condensed state Jacobian
   * @param H_R Jacobian of initializing measurements wrt variables in H_order
   * @param H_L Jacobian of initializing measurements wrt new variable (needs to be invertible)
   * @param R Covariance of initializing measurements
   * @param res Residual of initializing measurements
   * @return False on invalid or numerically rank-deficient input; state and the
   * proposed variable remain unchanged. True after successful augmentation.
   */
  static bool initialize_invertible(std::shared_ptr<State> state, std::shared_ptr<ov_type::Type> new_variable,
                                    const std::vector<std::shared_ptr<ov_type::Type>> &H_order, const Eigen::MatrixXd &H_R,
                                    const Eigen::MatrixXd &H_L, const Eigen::MatrixXd &R, const Eigen::VectorXd &res);

  /**
   * @brief Augment the state with a stochastic copy of the current IMU pose
   *
   * After propagation, normally we augment the state with an new clone that is at the new update timestep.
   * This augmentation clones the IMU pose and adds it to our state's clone map.
   * If we are doing time offset calibration we also make our cloning a function of the time offset.
   * Time offset logic is based on Li and Mourikis @cite Li2014IJRR.
   *
   * We can write the current clone at the true imu base clock time as the
   * follow: \f{align*}{
   * {}^{I_{t+t_d}}_G\bar{q} &= \begin{bmatrix}\frac{1}{2} {}^{I_{t+\hat{t}_d}}\boldsymbol\omega \tilde{t}_d \\
   * 1\end{bmatrix}\otimes{}^{I_{t+\hat{t}_d}}_G\bar{q} \\
   * {}^G\mathbf{p}_{I_{t+t_d}} &= {}^G\mathbf{p}_{I_{t+\hat{t}_d}} + {}^G\mathbf{v}_{I_{t+\hat{t}_d}}\tilde{t}_d
   * \f}
   * where we say that we have propagated our state up to the current estimated true imaging time for the current image,
   * \f${}^{I_{t+\hat{t}_d}}\boldsymbol\omega\f$ is the angular velocity at the end of propagation with biases removed.
   * This is off by some smaller error, so to get to the true imaging time in the imu base clock, we can append some small timeoffset error.
   * Thus the Jacobian in respect to our time offset during our cloning procedure is the following:
   * \f{align*}{
   * \frac{\partial {}^{I_{t+t_d}}_G\tilde{\boldsymbol\theta}}{\partial \tilde{t}_d} &= {}^{I_{t+\hat{t}_d}}\boldsymbol\omega \\
   * \frac{\partial {}^G\tilde{\mathbf{p}}_{I_{t+t_d}}}{\partial \tilde{t}_d} &= {}^G\mathbf{v}_{I_{t+\hat{t}_d}}
   * \f}
   *
   * @param state Pointer to state
   * @param last_w The estimated angular velocity at cloning time (current linearization)
   * @param last_w_fej The estimated angular velocity at cloning time (FEJ linearization)
   */
  static void augment_clone(std::shared_ptr<State> state, Eigen::Matrix<double, 3, 1> last_w,
                            Eigen::Matrix<double, 3, 1> last_w_fej);

  /**
   * Append an owned pose view with sparse Jacobian E_pose + [omega; v] e_clock^T.
   * The owner camera's active clock, including every state/clock cross block,
   * enters this deterministic augmentation. Fixed clocks add no clock column.
   * No independent noise or registry entry is added; the caller owns the handle.
   */
  static std::shared_ptr<ov_type::PoseJPL> augment_pose_view(std::shared_ptr<State> state, size_t clock_cam_id,
                                                           const Eigen::Vector3d &omega);

  /**
   * @brief Remove the oldest clone, if we have more then the max clone count!!
   *
   * This will marginalize the clone from our covariance, and remove it from our state.
   * This is mainly a helper function that we can call after each update.
   * It will marginalize the clone specified by State::margtimestep() which should return a clone timestamp.
   *
   * @param state Pointer to state
   */
  static void marginalize_old_clone(std::shared_ptr<State> state);

  /**
   * @brief Marginalize bad SLAM features
   * @param state Pointer to state
   */
  static void marginalize_slam(std::shared_ptr<State> state);

private:
  // Validate covariance ownership before QR, marginal extraction or assigning
  // the proposal a live ID. Does not allocate or touch measurement matrices.
  static bool valid_initialization_order(const std::shared_ptr<State> &state,
      const std::shared_ptr<ov_type::Type> &proposal,
      const std::vector<std::shared_ptr<ov_type::Type>> &order, int columns);

  /**
   * All function in this class should be static.
   * Thus an instance of this class cannot be created.
   */
  StateHelper() {}
};

} // namespace ov_msckf

#endif // OV_MSCKF_STATE_HELPER_H
