/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_MSCKF_ZERO_VELOCITY_MODEL_H
#define OV_MSCKF_ZERO_VELOCITY_MODEL_H

#include <Eigen/Dense>
#include <cstdint>
#include <cstring>
#include "utils/quat_ops.h"

namespace ov_msckf {

// Fixed-intrinsic stationary IMU likelihood in raw sensor coordinates. Expressing
// both constraints here is equivalent to using the full correlated covariance of
// corrected gyro/accel residuals. Raw white noises are independent in this form.
class ZeroVelocityModel {
public:
  using Jacobian = Eigen::Matrix<double, 6, 9>; // q_GtoI (left JPL error), raw bg, raw ba
  using Residual = Eigen::Matrix<double, 6, 1>;

  bool set_calibration(const Eigen::Matrix3d &accel_map, const Eigen::Matrix3d &tg) {
    ready = false;
    if (!finite(accel_map) || !finite(tg)) return false;
    identity_accel = (accel_map.array() == Eigen::Matrix3d::Identity().array()).all();
    zero_tg = (tg.array() == 0.0).all();
    Eigen::FullPivLU<Eigen::Matrix3d> lu(accel_map);
    if (!lu.isInvertible() || !(accel_map.determinant() > 0.0) || !(lu.rcond() > 1e-12)) return false;
    accel_inverse = identity_accel ? Eigen::Matrix3d::Identity() : Eigen::Matrix3d(lu.inverse());
    Tg = tg;
    ready = finite(accel_inverse);
    return ready;
  }

  // force = R_current*g, force_jacobian = R_FEJ*g when FEJ is enabled.
  // innovation = -h; H differentiates h, as required by StateHelper::EKFUpdate.
  bool linearize(const Eigen::Vector3d &am, const Eigen::Vector3d &wm, const Eigen::Vector3d &bg,
                 const Eigen::Vector3d &ba, const Eigen::Vector3d &force, const Eigen::Vector3d &force_jacobian,
                 double weight_gyro, double weight_accel, Jacobian &H, Residual &innovation) const {
    if (!ready || !finite(am) || !finite(wm) || !finite(bg) || !finite(ba) || !finite(force) || !finite(force_jacobian) ||
        !finite_scalar(weight_gyro) || !finite_scalar(weight_accel) || !(weight_gyro > 0) || !(weight_accel > 0)) return false;
    Eigen::Vector3d h_w = wm - bg;
    Eigen::Vector3d h_a = am - ba - force;
    if (!zero_tg) h_w -= Tg * force;
    if (!identity_accel) h_a = am - ba - accel_inverse * force;
    innovation.head<3>() = -weight_gyro * h_w;
    innovation.tail<3>() = -weight_accel * h_a;
    H.setZero();
    const Eigen::Matrix3d force_skew = ov_core::skew_x(force_jacobian);
    if (!zero_tg) H.block<3,3>(0,0) = -weight_gyro * Tg * force_skew;
    H.block<3,3>(0,3) = -weight_gyro * Eigen::Matrix3d::Identity();
    H.block<3,3>(3,0) = -weight_accel * force_skew;
    if (!identity_accel) H.block<3,3>(3,0) = -weight_accel * accel_inverse * force_skew;
    H.block<3,3>(3,6) = -weight_accel * Eigen::Matrix3d::Identity();
    return finite(H) && finite(innovation);
  }

private:
  // The estimator is compiled with -ffast-math; use representation checks.
  static bool finite_scalar(double value) {
    std::uint64_t bits;
    static_assert(sizeof(bits) == sizeof(value), "64-bit IEEE double required");
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
  }
  template <typename Derived> static bool finite(const Eigen::MatrixBase<Derived> &value) {
    for (int c=0; c<value.cols(); ++c) for (int r=0; r<value.rows(); ++r)
      if (!finite_scalar(value(r,c))) return false;
    return true;
  }
  bool ready = false, identity_accel = true, zero_tg = true;
  Eigen::Matrix3d accel_inverse = Eigen::Matrix3d::Identity(), Tg = Eigen::Matrix3d::Zero();
};
} // namespace ov_msckf
#endif
