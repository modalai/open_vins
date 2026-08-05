/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * ov_zcalib: reprojection factor with time-offset / rolling-shutter-readout blocks.
 * Wraps the proven Factor_ImageReprojCalib: the value transport applies the
 * first-order kinematic shift Delta = (td - td_lin) + (tr - tr_lin)*u to the
 * observing pose BEFORE delegating (R' = (I - skew(w*Delta)) R, p' = p + v*Delta,
 * the standard Li-Mourikis model with the window-solve clone kinematics), and the
 * td/tr columns are the exact chain J_td = J_theta*w + J_p*v (row-scaled by u for
 * tr). The pose-block Jacobians are reused from the inner factor (the O(|w|Delta)
 * transport of those columns only affects convergence rate, never the optimum).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_FACTOR_REPROJ_TD_H
#define OV_ZCALIB_FACTOR_REPROJ_TD_H

#include <Eigen/Dense>

#include "ceres_free/Factor_ImageReprojCalib.h"
#include "utils/quat_ops.h"

namespace ov_zcalib {

/**
 * @brief Reprojection with calibration + temporal blocks.
 *
 * Parameter blocks (2 residuals):
 *  [0] q_GtoIi (4) [1] p_IiinG (3) [2] p_FinG (3) [3] q_ItoC (4) [4] p_IinC (3)
 *  [5] cam intrinsics (8) [6] td (1) [7] tr (1)
 */
class Factor_ReprojTd : public ov_init::zbft_sfm::CostFunction {
public:
  ov_init::zbft_sfm::Factor_ImageReprojCalib inner;
  Eigen::Vector3d w_clone, v_clone; ///< clone kinematics (window frame; w in IMU frame)
  double u_frac = 0.5;              ///< row fraction m/M of this observation
  double td_lin = 0.0, tr_lin = 0.0;
  /// Constant offset [s] from this observation's own seed-mapped instant to its clone's stamp
  /// (CloneObs::dt_ref). Zero whenever the clone IS this observation's frame, which is the normal
  /// case; nonzero only for a frame merged onto a neighbouring camera's clone. It shifts the
  /// transport but not its derivative, so no Jacobian below changes.
  double dt_ref = 0.0;

  Factor_ReprojTd(const Eigen::Vector2d &uv, double pix_sigma, bool is_fisheye, const Eigen::Vector3d &w_at_clone,
                  const Eigen::Vector3d &v_at_clone, double u_row_frac, double td_lin_, double tr_lin_, double dt_ref_ = 0.0)
      : inner(uv, pix_sigma, is_fisheye), w_clone(w_at_clone), v_clone(v_at_clone), u_frac(u_row_frac), td_lin(td_lin_),
        tr_lin(tr_lin_), dt_ref(dt_ref_) {
    set_num_residuals(2);
    for (int s : {4, 3, 3, 4, 3, 8, 1, 1})
      mutable_parameter_block_sizes()->push_back(s);
  }

  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    const double td = parameters[6][0], tr = parameters[7][0];
    const double Delta = dt_ref + (td - td_lin) + (tr - tr_lin) * u_frac;

    // Transported observing pose (value-exact first-order kinematic shift)
    Eigen::Map<const Eigen::Vector4d> q(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> p(parameters[1]);
    Eigen::Vector4d dq;
    dq.head<3>() = 0.5 * w_clone * Delta;
    dq(3) = 1.0;
    dq /= dq.norm();
    Eigen::Vector4d q_t = ov_core::quat_multiply(dq, q);
    Eigen::Vector3d p_t = p + v_clone * Delta;

    const double *inner_params[6] = {q_t.data(), p_t.data(), parameters[2], parameters[3], parameters[4], parameters[5]};
    double *inner_jac[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    Eigen::Matrix<double, 2, 4, Eigen::RowMajor> Jq;
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> Jp;
    const bool need_j = (jacobians != nullptr);
    if (need_j) {
      inner_jac[0] = Jq.data();
      inner_jac[1] = Jp.data();
      for (int b = 2; b < 6; ++b)
        inner_jac[b] = jacobians ? jacobians[b] : nullptr;
    }
    if (!inner.Evaluate(inner_params, residuals, need_j ? inner_jac : nullptr))
      return false;
    if (!jacobians)
      return true;

    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(jacobians[0]);
      J = Jq; // first-order: transported-frame theta ~= clone theta (O(|w|Delta) transport omitted)
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
      J = Jp;
    }
    // Temporal columns: dr/dDelta = J_theta * w + J_p * v  (theta-left convention)
    const Eigen::Vector2d dr_dDelta = Jq.leftCols<3>() * w_clone + Jp * v_clone;
    if (jacobians[6]) {
      jacobians[6][0] = dr_dDelta(0);
      jacobians[6][1] = dr_dDelta(1);
    }
    if (jacobians[7]) {
      jacobians[7][0] = dr_dDelta(0) * u_frac;
      jacobians[7][1] = dr_dDelta(1) * u_frac;
    }
    return true;
  }
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_FACTOR_REPROJ_TD_H
