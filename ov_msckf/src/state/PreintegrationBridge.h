/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
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

#ifndef OV_MSCKF_PREINTEGRATION_BRIDGE_H
#define OV_MSCKF_PREINTEGRATION_BRIDGE_H

#include <Eigen/Eigen>

namespace ov_msckf {

/**
 * @brief ACI2 preintegration bridge over a short KNOWN interval (epoch-anchored cloning).
 *
 * Composes a clone pose forward to a measurement instant EXACTLY (under the same
 * piecewise-constant-signal assumption the propagator makes), replacing first-order
 * omega/v extrapolation:
 *
 *   R_GtoI(m) = DR * R_GtoI(k)
 *   p(m) = p(k) + v(k)*dt + p_grav + R_GtoI(k)^T * alpha
 *   v(m) = v(k) + v_grav + R_GtoI(k)^T * beta
 *
 * Built once at the CURRENT bias estimates (ACI2 partial-fixed linearization); consumers
 * correct later bias motion to first order via J_b -- the IMU is never re-integrated.
 * Convention for the bias correction (validated by the finite-difference oracle test):
 *   DR(b0+db) = exp_so3(J_th * db) * DR(b0),  alpha += J_alpha*db,  beta += J_beta*db.
 *
 * @author Joao Leonardo Silva Cotta (@zauberflote1)
 */
struct PreintBridgeData {
  /// Integrated interval length (s)
  double dt = 0;
  /// Relative rotation I_k -> I_m (JPL: left-composes onto R_GtoI(k))
  Eigen::Matrix3d DR = Eigen::Matrix3d::Identity();
  /// Position / velocity preintegrals expressed in the I_k frame
  Eigen::Vector3d alpha = Eigen::Vector3d::Zero();
  Eigen::Vector3d beta = Eigen::Vector3d::Zero();
  /// Gravity contributions (world-frame constants): p_grav = -1/2 g dt^2, v_grav = -g dt
  Eigen::Vector3d p_grav = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_grav = Eigen::Vector3d::Zero();
  /// d(theta, alpha, beta)/d(bg, ba) at the build-time bias linearization (rows th,p,v; cols bg,ba)
  Eigen::Matrix<double, 9, 6> J_b = Eigen::Matrix<double, 9, 6>::Zero();
  /// Preintegration noise on (theta, alpha) -- available for measurement-noise inflation
  Eigen::Matrix<double, 6, 6> Q_tp = Eigen::Matrix<double, 6, 6>::Zero();
  /// Intrinsics/bias-corrected angular rate at the measurement instant (H_dt linearization)
  Eigen::Vector3d w_end = Eigen::Vector3d::Zero();
  /// Bias linearization points at build time (for the first-order J_b correction)
  Eigen::Vector3d bg0 = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba0 = Eigen::Vector3d::Zero();
  bool valid = false;
};

} // namespace ov_msckf

#endif // OV_MSCKF_PREINTEGRATION_BRIDGE_H
