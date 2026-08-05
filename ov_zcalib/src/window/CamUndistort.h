/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * ov_zcalib: Eigen-only pixel -> unit-bearing undistortion at the SEED
 * intrinsics (radtan k1..k4 = k1,k2,p1,p2 as in ov_core CamRadtan, or
 * equidistant k1..k4). Fixed-iteration Newton/fixed-point (deterministic,
 * allocation-free) — seed-level accuracy only; the MLE re-projects through
 * the live cam block and never uses these bearings.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_CAM_UNDISTORT_H
#define OV_ZCALIB_CAM_UNDISTORT_H

#include <Eigen/Dense>
#include <cmath>

namespace ov_zcalib {

class CamUndistort {
public:
  /// Raw distorted pixel -> unit bearing (camera frame). cam = [fx fy cx cy k1 k2 k3 k4].
  static Eigen::Vector3d bearing(const Eigen::Vector2d &uv, const Eigen::Matrix<double, 8, 1> &cam, bool fisheye) {
    const double xd = (uv(0) - cam(2)) / cam(0);
    const double yd = (uv(1) - cam(3)) / cam(1);
    double x = xd, y = yd;
    if (fisheye) {
      // equidistant: rd = theta(1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8)
      const double rd = std::sqrt(xd * xd + yd * yd);
      if (rd > 1e-12) {
        double th = rd; // Newton on f(th) = th*(1 + ...) - rd
        for (int it = 0; it < 8; ++it) {
          const double th2 = th * th, th4 = th2 * th2, th6 = th4 * th2, th8 = th4 * th4;
          const double poly = 1.0 + cam(4) * th2 + cam(5) * th4 + cam(6) * th6 + cam(7) * th8;
          const double dpoly = 2.0 * cam(4) * th + 4.0 * cam(5) * th * th2 + 6.0 * cam(6) * th * th4 + 8.0 * cam(7) * th * th6;
          const double f = th * poly - rd, df = poly + th * dpoly;
          if (std::abs(df) < 1e-12)
            break;
          th -= f / df;
        }
        const double s = std::tan(th) / rd;
        x = xd * s;
        y = yd * s;
      }
    } else {
      // radtan (k1, k2, p1, p2): fixed-point inverse
      const double k1 = cam(4), k2 = cam(5), p1 = cam(6), p2 = cam(7);
      for (int it = 0; it < 8; ++it) {
        const double r2 = x * x + y * y;
        const double rad = 1.0 + k1 * r2 + k2 * r2 * r2;
        const double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        const double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        x = (xd - dx) / rad;
        y = (yd - dy) / rad;
      }
    }
    return Eigen::Vector3d(x, y, 1.0).normalized();
  }
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_CAM_UNDISTORT_H
