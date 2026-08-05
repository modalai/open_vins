/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * ov_zcalib: targetless VI calibrator (batch, ceres-free, filter-path-free)
 * ------------------------------------------------------------------------
 * ACI3-style calibration preintegration (after Yang et al., MVIS, IJRR 2024):
 * one sweep over the raw IMU samples of a window segment produces
 *   - the preintegrated mean (DeltaR, alpha, beta) with the intrinsic model applied,
 *   - the 15x15 CPI measurement covariance (theta, bg, beta, ba, alpha order),
 *   - analytic first-order Jacobian columns of the mean w.r.t. the FULL calibration
 *     vector: gyro/accel biases AND the IMU intrinsics (Dw6, Da6, R_ACCtoIMU, opt. Tg),
 * so a batch solver can iterate calibration parameters without re-integrating.
 *
 * Conventions are OpenVINS/JPL throughout (see ov_zcalib/README.md). The rotation
 * bias/parameter Jacobians are accumulated in the LEFT (end-frame) convention
 *   DeltaR(p) = Exp(J_theta * dp) * DeltaR(p_lin)
 * with the exact per-step increment R_step*(J + Jr(-w dt)*M_w*dt) (the J_r flavor
 * is pinned by the FD oracle in test_aci3_fd), and converted to the
 * Factor_ImuCPIv1 right/START-frame convention at emit time.
 *
 * This module is self-contained: it links ov_core only (quat_ops); it neither
 * includes nor links anything from ov_msckf (zero filter-path coupling).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_ACI_CALIB_PREINT_H
#define OV_ZCALIB_ACI_CALIB_PREINT_H

#include <Eigen/Dense>
#include <vector>

#include "../types/ImuIntrinsicModel.h"

namespace ov_zcalib {

/**
 * @brief One raw IMU sample (sensor-frame, uncorrected).
 */
struct RawImu {
  double timestamp = 0.0;
  Eigen::Vector3d wm = Eigen::Vector3d::Zero(); ///< raw gyro (sensor frame)
  Eigen::Vector3d am = Eigen::Vector3d::Zero(); ///< raw accel (sensor frame)
  double temp_c = 0.0; ///< sensor temperature (thermal gates/fingerprints; ignored by the preintegration)
};

/**
 * @brief IMU continuous-noise densities (kalibr-chain units).
 */
struct ImuNoise {
  double sigma_w = 1.6968e-04;  ///< gyro white  [rad/s/sqrt(Hz)]
  double sigma_wb = 1.9393e-05; ///< gyro bias RW [rad/s^2/sqrt(Hz)]
  double sigma_a = 2.0000e-03;  ///< accel white [m/s^2/sqrt(Hz)]
  double sigma_ab = 3.0000e-03; ///< accel bias RW [m/s^3/sqrt(Hz)]
};

/**
 * @brief ACI3 calibration preintegration over one clone-to-clone interval.
 *
 * Emitted quantities (all at the linearization point given to integrate()):
 *  - q_KtoK1 / alpha / beta: the CPIv1 measurement triplet (start-frame {K}).
 *  - P15: measurement covariance, CPIv1 residual order [theta, bg, beta, ba, alpha].
 *  - J_q, J_b(eta), J_a(lpha), H_b, H_a: bias Jacobians, Factor_ImuCPIv1 conventions.
 *  - Jq_pi / Jb_pi / Ja_pi: the ACI3 columns for the enabled intrinsic groups, in the
 *    SAME conventions as the bias Jacobians (theta columns already converted to the
 *    right/start-frame convention used by the factor's quaternion correction).
 *    Column layout = ImuIntrinsicModel::num_params() (dw6 | da6 | th_A3 | tg9 subset).
 */
struct AciPreintResult {
  double dt = 0.0;
  Eigen::Vector4d q_KtoK1 = Eigen::Vector4d(0, 0, 0, 1);
  Eigen::Vector3d alpha = Eigen::Vector3d::Zero();
  Eigen::Vector3d beta = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 15, 15> P15 = Eigen::Matrix<double, 15, 15>::Identity();
  // Bias Jacobians (Factor_ImuCPIv1 member names/conventions)
  Eigen::Matrix3d J_q = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_b = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_a = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d H_b = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d H_a = Eigen::Matrix3d::Zero();
  /// d(theta)/d(ba): identically ZERO unless the model carries Tg (ba reaches the rotation
  /// through Mw_ba = -Dw*Tg*Ma_ba). Consumed by the factor's tg branch only, so the legacy
  /// (Tg-off) arithmetic never touches it.
  Eigen::Matrix3d H_q = Eigen::Matrix3d::Zero();
  // ACI3 intrinsic columns (3 x n_pi each; n_pi = model.num_params())
  Eigen::Matrix<double, 3, Eigen::Dynamic> Jq_pi;
  Eigen::Matrix<double, 3, Eigen::Dynamic> Jb_pi;
  Eigen::Matrix<double, 3, Eigen::Dynamic> Ja_pi;
  // End-instant corrected kinematics (for time-offset transport columns downstream)
  Eigen::Vector3d w_end = Eigen::Vector3d::Zero();
  Eigen::Vector3d a_end = Eigen::Vector3d::Zero();
};

/**
 * @brief The single-sweep integrator. Stateless; all context passed per call.
 */
class AciCalibPreint {
public:
  /**
   * @brief Integrate raw samples on [t0, t1] at the given linearization point.
   * @param imu     raw samples covering [t0, t1] (boundary samples interpolated)
   * @param t0,t1   interval endpoints (IMU clock)
   * @param model   intrinsic model (values = linearization point) + enable flags
   * @param bg_lin  gyro bias linearization (raw sensor frame, pre-Dw)
   * @param ba_lin  accel bias linearization (raw sensor frame, pre-Da)
   * @param noise   continuous noise densities
   * @param out     result (see AciPreintResult)
   * @return false if fewer than 2 samples span the interval
   */
  /**
   * @param noise_model intrinsic model used ONLY for the covariance propagation
   *        (Phi/G bias-mixing). Freeze this at the fusion-entry calibration: if the
   *        weights are rebuilt with the live model each outer iteration, the
   *        Gauss-Newton gradient (which ignores dW/dp) chases the weights and the
   *        outer loop converges to a non-minimum. nullptr = use `model`.
   */
  static bool integrate(const std::vector<RawImu> &imu, double t0, double t1, const ImuIntrinsicModel &model,
                        const Eigen::Vector3d &bg_lin, const Eigen::Vector3d &ba_lin, const ImuNoise &noise, AciPreintResult &out,
                        const ImuIntrinsicModel *noise_model = nullptr);

  /**
   * @brief All clone intervals of a window in ONE chronological two-pointer
   *        sweep -- BIT-IDENTICAL to calling integrate() per interval.
   *
   * integrate() rescans the window's full IMU vector from index 0 and reserves
   * a full-size boundary-clamped copy for EVERY interval (O(N*S) comparisons +
   * O(N*S) allocation traffic per window evaluation). The sweep advances one
   * sample cursor monotonically and streams each step straight into the
   * per-interval accumulators: O(S + N), zero per-interval allocation. The
   * boundary interpolant at each interior clone time is derived from the same
   * straddling pair with the same lambda as integrate()'s two independent
   * scans, so every emitted field matches integrate() BYTE-EXACTLY (enforced
   * by the randomized oracle in test_preint_chain and the OV_ZCALIB_PREINT_AUDIT
   * runtime auditor; any edit to either loop body must preserve the pairing).
   *
   * @param clone_times strictly the window's clone stamps (>= 2)
   * @param out resized to clone_times.size()-1; out[k] covers [ct[k], ct[k+1]]
   * @param skip_cov true skips the covariance propagation entirely (P15 stays
   *        at its default) -- EXACT for consumers that never read P15 (the
   *        linear seeder): the mean/Jacobian recursions do not depend on P.
   * @return false if ANY interval fails (same per-interval conditions as
   *         integrate(); partial results in `out` are then meaningless)
   */
  static bool integrate_chain(const std::vector<RawImu> &imu, const std::vector<double> &clone_times, const ImuIntrinsicModel &model,
                              const Eigen::Vector3d &bg_lin, const Eigen::Vector3d &ba_lin, const ImuNoise &noise,
                              std::vector<AciPreintResult> &out, const ImuIntrinsicModel *noise_model = nullptr,
                              bool skip_cov = false);

  /**
   * @brief Closed-form ACI2 step integrals (Xi_1, Xi_2) and their within-step
   *        omega-sensitivities (T1 = d(Xi_1*a)/d(omega), T2 = d(Xi_2*a)/d(omega)),
   *        small-omega safe. Exposed for the FD oracle.
   */
  static void step_integrals(double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat, Eigen::Matrix3d &R_step,
                             Eigen::Matrix3d &Jr_neg, Eigen::Matrix3d &Xi_1, Eigen::Matrix3d &Xi_2, Eigen::Matrix3d &T1,
                             Eigen::Matrix3d &T2);
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_ACI_CALIB_PREINT_H
