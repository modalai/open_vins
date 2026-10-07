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
 * filter's forward projection retains its existing precision.
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

// Equidistant derivatives with the same forward model and optical-center
// branch as distort_double. The historical camera-object implementation used
// theta/r with r replaced by 1 near zero, incorrectly returning zero instead
// of diag(fx,fy) at the optical center.
// Use the pinhole limit there and a series nearby to avoid cancellation in
// (d(theta_d)/dr - theta_d/r)/r^2. All temporaries have fixed size.
inline void equidistant_jacobian_double(const Eigen::Matrix<double, 8, 1> &c, const Eigen::Vector2d &uv,
                                       Eigen::Matrix2d &Jn, Eigen::Matrix<double, 2, 8> *Jc = nullptr) {
  const double x = uv(0), y = uv(1), r2 = x*x + y*y;
  const double r = std::sqrt(r2);
  if (r <= 1e-8) {
    Jn << c(0), 0.0, 0.0, c(1);
    if (Jc) {
      Jc->setZero();
      (*Jc)(0,0) = x; (*Jc)(1,1) = y;
      (*Jc)(0,2) = (*Jc)(1,3) = 1.0;
    }
    return;
  }

  const double theta = std::atan(r), t2 = theta*theta, t4 = t2*t2, t6 = t4*t2, t8 = t4*t4;
  const double theta_d = theta*(1.0 + c(4)*t2 + c(5)*t4 + c(6)*t6 + c(7)*t8);
  const double s = theta_d/r;
  double h;
  if (r2 < 1e-8) {
    // theta_d/r = 1 + (k1-1/3)r^2 + (k2-k1+1/5)r^4 + O(r^6).
    h = 2.0*(c(4) - 1.0/3.0) + 4.0*(c(5) - c(4) + 1.0/5.0)*r2;
  } else {
    const double dtheta = 1.0 + 3.0*c(4)*t2 + 5.0*c(5)*t4 + 7.0*c(6)*t6 + 9.0*c(7)*t8;
    h = (dtheta/(1.0 + r2) - s)/r2;
  }
  Jn << c(0)*(s + h*x*x), c(0)*h*x*y,
        c(1)*h*x*y, c(1)*(s + h*y*y);
  if (Jc) {
    Jc->setZero();
    (*Jc)(0,0) = x*s; (*Jc)(1,1) = y*s;
    (*Jc)(0,2) = (*Jc)(1,3) = 1.0;
    const double fx_x_theta = c(0)*x*theta/r, fy_y_theta = c(1)*y*theta/r;
    (*Jc)(0,4) = fx_x_theta*t2; (*Jc)(1,4) = fy_y_theta*t2;
    (*Jc)(0,5) = fx_x_theta*t4; (*Jc)(1,5) = fy_y_theta*t4;
    (*Jc)(0,6) = fx_x_theta*t6; (*Jc)(1,6) = fy_y_theta*t6;
    (*Jc)(0,7) = fx_x_theta*t8; (*Jc)(1,7) = fy_y_theta*t8;
  }
}

// Fixed-size analytic radtan derivatives. No camera object, OpenCV matrix
// update, or dynamic Eigen allocation is needed for an individual residual.
// The 2x8 intrinsic Jacobian is optional: inner BA holds intrinsics constant.
inline void radtan_jacobian_double(const Eigen::Matrix<double, 8, 1> &c, const Eigen::Vector2d &uv,
                                  Eigen::Matrix2d &Jn, Eigen::Matrix<double, 2, 8> *Jc = nullptr) {
  const double x = uv(0), y = uv(1), xx = x*x, yy = y*y, xy = x*y;
  const double r2 = xx + yy, r4 = r2*r2;
  const double rad = 1.0 + c(4)*r2 + c(5)*r4;
  const double dr = 2.0*c(4) + 4.0*c(5)*r2;
  const double cross = dr*xy + 2.0*c(6)*x + 2.0*c(7)*y;
  Jn << c(0)*(rad + dr*xx + 2.0*c(6)*y + 6.0*c(7)*x), c(0)*cross,
        c(1)*cross, c(1)*(rad + dr*yy + 6.0*c(6)*y + 2.0*c(7)*x);
  if (Jc) {
    Jc->setZero();
    (*Jc)(0,0) = x*rad + 2.0*c(6)*xy + c(7)*(r2 + 2.0*xx);
    (*Jc)(1,1) = y*rad + c(6)*(r2 + 2.0*yy) + 2.0*c(7)*xy;
    (*Jc)(0,2) = (*Jc)(1,3) = 1.0;
    (*Jc)(0,4) = c(0)*x*r2; (*Jc)(0,5) = c(0)*x*r4;
    (*Jc)(1,4) = c(1)*y*r2; (*Jc)(1,5) = c(1)*y*r4;
    (*Jc)(0,6) = 2.0*c(0)*xy; (*Jc)(0,7) = c(0)*(r2 + 2.0*xx);
    (*Jc)(1,6) = c(1)*(r2 + 2.0*yy); (*Jc)(1,7) = 2.0*c(1)*xy;
  }
}

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
