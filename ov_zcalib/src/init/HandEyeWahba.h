/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * ov_zcalib: rotational hand-eye bootstrap (R_ItoC + gyro bias + fine td).
 *
 * Math (OpenVINS conventions; every port re-derived and oracle-pinned):
 * frames rigidly attached => {}^{C2}_{C1}R = R_ItoC {}^{I2}_{I1}R R_ItoC^T
 * exactly, so the relative-rotation logs obey the adjoint pair relation
 *      theta_C = R_ItoC * theta_I,      theta_X = Log({}^{X2}_{X1}R).
 * theta_I comes from gyro-only preintegration over the td-shifted frame
 * interval (per-step exp(-w_hat dt), midpoint samples, boundary-interpolated
 * — the AciCalibPreint stepping). The solve alternates
 *   (1) Wahba/Markley SVD for R_ItoC on unit log pairs (magnitude-weighted,
 *       reflection-corrected, sigma_min/sigma_max axis-diversity gate),
 *   (2) closed-form gyro-bias update via d(theta_I)/d(bg) = +dt*I (first
 *       order; re-preintegration between alternations makes it exact),
 * then one Hampel trim (median + 3*1.4826*MAD on residual norms) with refit
 * adopted only if the kept-set RMSE improves, and a fine td sweep that
 * re-preintegrates at each candidate (no Taylor shift needed) around the
 * xcorr seed, with boundary pinning reported. Estimating bg jointly is the
 * kalibr orientation-prior improvement over plain hand-eye; the coarse/fine
 * search shape and the gates follow the flight-tested VOXL calibrator.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_HAND_EYE_WAHBA_H
#define OV_ZCALIB_HAND_EYE_WAHBA_H

#include <Eigen/Dense>
#include <vector>

#include "../cpi/AciCalibPreint.h"

namespace ov_zcalib {

/// One camera frame pair: interval in CAMERA clock + the visual relative-rotation log.
struct HandEyePair {
  double t0 = 0.0, t1 = 0.0;                          ///< frame stamps (camera clock)
  Eigen::Vector3d theta_C = Eigen::Vector3d::Zero();  ///< Log({}^{C2}_{C1}R) from the front-end
  double weight = 1.0;                                ///< front-end quality (e.g. inlier ratio)
};

struct HandEyeConfig {
  double min_pair_rate = 0.15;  ///< rad/s: reject rotation-poor pairs
  double max_pair_rate = 6.0;   ///< rad/s: reject saturated/blurred pairs
  double ratio_min = 0.5;       ///< |theta_C|/|theta_I| sanity band (tracking-failure gate)
  double ratio_max = 1.8;
  double min_axis_diversity = 0.08; ///< sigma_min/sigma_max of the Wahba B matrix
  int min_pairs = 25;
  int bias_alternations = 4;
  /// false = hold bg at the seed (a SETTLE still baseline beats re-estimating it
  /// from translation-contaminated visual pairs; kalibr estimates bg only
  /// because it has no still period)
  bool estimate_bg = true;
  /// Physical sanity bound on the estimated |bg| [rad/s]. On close-range
  /// translation-heavy scenes the closed-form bias step soaks the DC of the
  /// rotation-only epipolar bias (observed 0.21 rad/s on the kalibr sample —
  /// absurd for any MEMS gyro); beyond this the solve REDOES the alternation
  /// with bg frozen at the seed (a 0-bias error costs ~0.3% of theta_I, and
  /// the per-window BA estimates bias as a state regardless).
  double max_bg_sane = 0.05;
  double td_fine_range = 0.004; ///< fine sweep half-width around the xcorr seed [s]
  double td_fine_step = 0.0002; ///< [s]
};

struct HandEyeResult {
  bool ok = false;
  Eigen::Vector4d q_ItoC = Eigen::Vector4d(0, 0, 0, 1); ///< JPL, quat_2_Rot(q) = R_ItoC
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();          ///< raw-frame gyro bias seed
  double td = 0.0;                                       ///< refined time offset [s]
  double rmse_rad = 0.0;                                 ///< residual RMSE on all usable pairs
  double axis_diversity = 0.0;                           ///< sigma_min/sigma_max at the solution
  int pairs_used = 0;
  int pairs_trimmed = 0;
  bool td_at_bound = false;
};

class HandEyeWahba {
public:
  /**
   * @brief Solve R_ItoC, bg, td from frame pairs + raw IMU.
   * @param imu      raw samples spanning all pair intervals (+ td search slack)
   * @param pairs    visual relative-rotation pairs (camera clock)
   * @param td_seed  coarse seed from TimeOffsetInit (fine sweep centers here)
   * @param bg_seed  gyro-bias seed (zero is fine; it is re-estimated)
   */
  static bool solve(const std::vector<RawImu> &imu, const std::vector<HandEyePair> &pairs, double td_seed,
                    const Eigen::Vector3d &bg_seed, const HandEyeConfig &cfg, HandEyeResult &out);

  /// Gyro-only preintegrated Log({}^{I2}_{I1}R) over [t0,t1] (IMU clock). Exposed for oracles.
  static bool theta_imu(const std::vector<RawImu> &imu, double t0, double t1, const Eigen::Vector3d &bg, Eigen::Vector3d &theta);
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_HAND_EYE_WAHBA_H
