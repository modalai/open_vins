/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * ov_zcalib: IMU intrinsic model (RSS 2020 "imu2"/kalibr variant + subsets)
 * ------------------------------------------------------------------------
 * Corrected signals (OpenVINS convention, biases in the RAW sensor frame):
 *
 *   a_hat = R_AtoI * Da * (a_m - b_a)
 *   w_hat =          Dw * (w_m - b_g - Tg * a_hat)      (R_GtoI = I: imu2)
 *
 * with Dw, Da UPPER-triangular (inverse scale/misalignment, Dw = Tw^-1), packed
 * column-wise as [d11, d12, d22, d13, d23, d33], and Tg the 9-parameter
 * g-sensitivity (row-major), OFF by default. Calibration groups are individually
 * selectable (kalibr-style subsets); the RSS 2020 over-parameterization rule is
 * structural here: R_GtoI is never a parameter (exactly one gyro/accel frame
 * rotation may be free, and this model fixes the gyro one).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_IMU_INTRINSIC_MODEL_H
#define OV_ZCALIB_IMU_INTRINSIC_MODEL_H

#include <Eigen/Dense>

#include "utils/quat_ops.h"

namespace ov_zcalib {

/**
 * @brief imu2 intrinsic model with kalibr-style calibration subsets.
 *
 * Parameter vector layout (enabled groups only, in this fixed order):
 *   [ dw6 | da6 | th_A (3, JPL local of q_AtoI) | tg9 ]
 */
class ImuIntrinsicModel {
public:
  // ---- values (linearization point / current estimate) ----
  Eigen::Matrix<double, 6, 1> dw = (Eigen::Matrix<double, 6, 1>() << 1, 0, 1, 0, 0, 1).finished();
  Eigen::Matrix<double, 6, 1> da = (Eigen::Matrix<double, 6, 1>() << 1, 0, 1, 0, 0, 1).finished();
  Eigen::Vector4d q_AtoI = Eigen::Vector4d(0, 0, 0, 1); ///< JPL, R_AtoI = quat_2_Rot(q_AtoI)
  Eigen::Matrix3d Tg = Eigen::Matrix3d::Zero();

  // ---- calibration-subset flags (kalibr-style) ----
  bool calib_dw = true;
  bool calib_da = true;
  bool calib_RAtoI = true;
  bool calib_tg = false;

  /// Upper-triangular expansion of a 6-vector [d11,d12,d22,d13,d23,d33]
  static Eigen::Matrix3d ut(const Eigen::Matrix<double, 6, 1> &d) {
    Eigen::Matrix3d D = Eigen::Matrix3d::Zero();
    D(0, 0) = d(0);
    D(0, 1) = d(1);
    D(1, 1) = d(2);
    D(0, 2) = d(3);
    D(1, 2) = d(4);
    D(2, 2) = d(5);
    return D;
  }

  /// Basis matrix E_k = dD/dd_k for the upper-triangular packing
  static Eigen::Matrix3d ut_basis(int k) {
    static const int rc[6][2] = {{0, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {2, 2}};
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    E(rc[k][0], rc[k][1]) = 1.0;
    return E;
  }

  int num_params() const { return (calib_dw ? 6 : 0) + (calib_da ? 6 : 0) + (calib_RAtoI ? 3 : 0) + (calib_tg ? 9 : 0); }
  int off_dw() const { return 0; }
  int off_da() const { return calib_dw ? 6 : 0; }
  int off_thA() const { return off_da() + (calib_da ? 6 : 0); }
  int off_tg() const { return off_thA() + (calib_RAtoI ? 3 : 0); }

  /**
   * @brief Correct one raw sample at the model's values.
   */
  void correct(const Eigen::Vector3d &wm, const Eigen::Vector3d &am, const Eigen::Vector3d &bg, const Eigen::Vector3d &ba,
               Eigen::Vector3d &w_hat, Eigen::Vector3d &a_hat) const {
    const Eigen::Matrix3d Dw = ut(dw), Da = ut(da);
    const Eigen::Matrix3d R_A = ov_core::quat_2_Rot(q_AtoI);
    a_hat = R_A * (Da * (am - ba));
    w_hat = Dw * (wm - bg - Tg * a_hat);
  }

  /**
   * @brief Per-sample mixing matrices: M_w = d(w_hat)/dp, M_a = d(a_hat)/dp (3 x num_params()),
   *        and the bias mixing d(w_hat)/dbg, d(w_hat)/dba, d(a_hat)/dba (d(a_hat)/dbg = 0).
   *        JPL left-error convention for th_A: R_AtoI = (I - skew(th)) * R_hat  =>
   *        d(a_hat)/dth = skew(a_hat).
   */
  void mixing(const Eigen::Vector3d &wm, const Eigen::Vector3d &am, const Eigen::Vector3d &bg, const Eigen::Vector3d &ba,
              Eigen::Matrix<double, 3, Eigen::Dynamic> &M_w, Eigen::Matrix<double, 3, Eigen::Dynamic> &M_a, Eigen::Matrix3d &Mw_bg,
              Eigen::Matrix3d &Mw_ba, Eigen::Matrix3d &Ma_ba) const {
    const Eigen::Matrix3d Dw = ut(dw), Da = ut(da);
    const Eigen::Matrix3d R_A = ov_core::quat_2_Rot(q_AtoI);
    const Eigen::Vector3d t = am - ba;
    const Eigen::Vector3d a_hat = R_A * (Da * t);
    const Eigen::Vector3d s = wm - bg - Tg * a_hat;
    const int n = num_params();
    M_w = Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, n);
    M_a = Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, n);
    // d(a_hat)/d(group), then chain d(w_hat)/d(group) = -Dw*Tg * d(a_hat)/d(group)
    if (calib_da) {
      for (int k = 0; k < 6; ++k)
        M_a.col(off_da() + k) = R_A * (ut_basis(k) * t);
    }
    if (calib_RAtoI)
      M_a.middleCols(off_thA(), 3) = ov_core::skew_x(a_hat);
    if (calib_dw) {
      for (int k = 0; k < 6; ++k)
        M_w.col(off_dw() + k) = ut_basis(k) * s;
    }
    if (calib_tg) {
      for (int k = 0; k < 9; ++k) {
        Eigen::Matrix3d Ek = Eigen::Matrix3d::Zero();
        Ek(k / 3, k % 3) = 1.0;
        M_w.col(off_tg() + k) = -Dw * (Ek * a_hat);
      }
    }
    if (!Tg.isZero())
      M_w -= Dw * Tg * M_a; // a_hat feedback into w_hat for da/th_A (and tg's own a_hat term above)
    // bias mixing
    Mw_bg = -Dw;
    Ma_ba = -R_A * Da;
    Mw_ba = -Dw * Tg * Ma_ba;
  }
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_IMU_INTRINSIC_MODEL_H
