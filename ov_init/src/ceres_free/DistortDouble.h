/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * Double-precision forward camera distortion, shared by the ceres and
 * ceres-free reprojection factors. ov_core::CamBase::distort_d bottlenecks
 * through distort_f (float32): the returned pixel is quantized at ~3-6e-5 px,
 * which stairsteps optimization costs against the double-exact analytic
 * Jacobians and breaks finite-difference oracles run against the factors.
 * Same models as ov_core compute_distort_jacobian (all-double already); the
 * filter's own CamBase consumers are untouched.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_ZBFT_DISTORT_DOUBLE_H
#define OV_INIT_ZBFT_DISTORT_DOUBLE_H

#include <Eigen/Dense>
#include <cmath>

namespace ov_init {

/// Forward-distort a normalized image point at the given intrinsics
/// [fx fy cx cy k1 k2 k3 k4] (equidistant when is_fisheye, radtan otherwise).
inline Eigen::Vector2d distort_double(const Eigen::Matrix<double, 8, 1> &c, const Eigen::Vector2d &uv_norm, bool is_fisheye) {
  Eigen::Vector2d uv_dist;
  if (is_fisheye) {
    const double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
    const double theta = std::atan(r);
    const double theta_d =
        theta + c(4) * std::pow(theta, 3) + c(5) * std::pow(theta, 5) + c(6) * std::pow(theta, 7) + c(7) * std::pow(theta, 9);
    const double cdist = (r > 1e-8) ? theta_d / r : 1.0;
    uv_dist(0) = c(0) * uv_norm(0) * cdist + c(2);
    uv_dist(1) = c(1) * uv_norm(1) * cdist + c(3);
  } else {
    const double x = uv_norm(0), y = uv_norm(1);
    const double r2 = x * x + y * y;
    const double rad = 1.0 + c(4) * r2 + c(5) * r2 * r2;
    const double x1 = x * rad + 2.0 * c(6) * x * y + c(7) * (r2 + 2.0 * x * x);
    const double y1 = y * rad + c(6) * (r2 + 2.0 * y * y) + 2.0 * c(7) * x * y;
    uv_dist(0) = c(0) * x1 + c(2);
    uv_dist(1) = c(1) * y1 + c(3);
  }
  return uv_dist;
}

} // namespace ov_init

#endif // OV_INIT_ZBFT_DISTORT_DOUBLE_H
