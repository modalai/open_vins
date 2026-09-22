/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_CERES_QUATERNION_TANGENT_H
#define OV_INIT_CERES_QUATERNION_TANGENT_H

#include "utils/quat_ops.h"

namespace ov_init {

// JPL left retraction q_plus = Exp_q(delta) (*) q. For a unit quaternion,
// D = d q_plus / d delta and L = 4 D' satisfy L D = I and D L = I - q q'.
// Ceres needs genuine ambient Jacobians: an analytic tangent factor H becomes
// H L. The local solver's explicit packed-tangent convention is separate.
inline Eigen::Matrix<double, 4, 3> jpl_plus_jacobian(const Eigen::Vector4d &q) {
  Eigen::Matrix<double, 4, 3> D;
  D.topRows<3>() = .5 * (q(3) * Eigen::Matrix3d::Identity() + ov_core::skew_x(q.head<3>()));
  D.bottomRows<1>() = -.5 * q.head<3>().transpose();
  return D;
}

inline Eigen::Matrix<double, 3, 4> jpl_tangent_lift(const Eigen::Vector4d &q) {
  return 4. * jpl_plus_jacobian(q).transpose();
}

} // namespace ov_init
#endif
