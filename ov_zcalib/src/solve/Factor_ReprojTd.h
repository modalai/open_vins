/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * ov_zcalib: reprojection factor with a time-offset block.
 * Wraps the proven Factor_ImageReprojCalib: the value transport applies the
 * exact-SO(3) kinematic shift Delta = dt_ref + (td - td_lin) to the observing
 * pose BEFORE delegating (R' = exp_so3(-w*Delta) R, p' = p + v*Delta).
 * The angular rate is a fixed clone linearization; velocity is the current
 * nuisance state. The exact columns are J_td = J_theta*w + J_p*v and
 * J_v = J_p*Delta. A clone's local attitude perturbation must
 * also be transported: E Exp(-dtheta) R = Exp(-E*dtheta) E R, with
 * E = exp_so3(-w*Delta). Thus J_clone_theta = J_theta * E.
 *
 * Rolling shutter enters through dt_ref, not through a parameter: the readout
 * time is a HARDWARE fact (HAL3 sensor mode), never estimated, so each
 * observation's row time (v/h - 0.5)*tr is a KNOWN constant folded into dt_ref
 * at factor construction (WindowBA). Frame timestamps anchor mid-frame: the
 * producer stamps HAL3 SOF + (readout + exposure)/2, and rows deviate by the
 * CENTERED fraction around that instant.
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
 *  [5] cam intrinsics (8) [6] td (1) [7] v_IiinG (3)
 */
class Factor_ReprojTd : public ov_init::zbft_sfm::CostFunction {
public:
  ov_init::zbft_sfm::Factor_ImageReprojCalib inner;
  Eigen::Vector3d w_clone; ///< fixed angular-rate linearization in the IMU frame
  double td_lin = 0.0;
  /// Constant offset [s] from this observation's own sampling instant to its clone's stamp:
  /// the centered rolling-shutter row time (v/h - 0.5)*tr_hw plus, for a frame merged onto a
  /// neighbouring camera's clone, the frame-merge offset (CloneObs::dt_ref).
  /// It shifts Delta while dDelta/dtd remains one.
  double dt_ref = 0.0;

  Factor_ReprojTd(const Eigen::Vector2d &uv, double pix_sigma, bool is_fisheye, const Eigen::Vector3d &w_at_clone,
                  double td_lin_, double dt_ref_ = 0.0)
      : inner(uv, pix_sigma, is_fisheye), w_clone(w_at_clone), td_lin(td_lin_), dt_ref(dt_ref_) {
    set_num_residuals(2);
    for (int s : {4, 3, 3, 4, 3, 8, 1, 3})
      mutable_parameter_block_sizes()->push_back(s);
  }

  // Calibration and the angular-rate linearization are fixed during a
  // WindowBA solve. Cache only their SO(3) exponential; the optimized
  // velocity is read from its parameter block on every evaluation.
  // Read-only during Evaluate (including parallel evaluation); off-stamp calls
  // use the exact original expression, so finite differences remain valid.
  void prepare_transport(double td) {
    prepared_delta_ = dt_ref + (td - td_lin);
    prepared_w_ = w_clone;
    prepared_dq_ = ov_core::rot_2_quat(ov_core::exp_so3(-w_clone * prepared_delta_));
    prepared_rotation_ = ov_core::quat_2_Rot(prepared_dq_);
    prepared_ = true;
  }

  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    const double td = parameters[6][0];
    const double Delta = dt_ref + (td - td_lin);

    // Transported observing pose: exact SO(3) composition under constant clone kinematics
    // (JPL: R(dq) = exp_so3(-w*Delta), the same map the filter's updaters apply)
    Eigen::Map<const Eigen::Vector4d> q(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> p(parameters[1]);
    Eigen::Map<const Eigen::Vector3d> v(parameters[7]);
    const bool prepared_match = prepared_ && Delta == prepared_delta_ && w_clone == prepared_w_;
    const Eigen::Vector4d dq = prepared_match ? prepared_dq_
                                             : ov_core::rot_2_quat(ov_core::exp_so3(-w_clone * Delta));
    Eigen::Vector4d q_t = ov_core::quat_multiply(dq, q);
    Eigen::Vector3d p_t = p + v * Delta;

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
      const Eigen::Matrix3d E = prepared_match ? prepared_rotation_ : ov_core::quat_2_Rot(dq);
      J.leftCols<3>().noalias() = Jq.leftCols<3>() * E;
      J.col(3).setZero(); // packed JPL local Jacobian
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
      J = Jp;
    }
    // Temporal column: dr/dDelta = J_theta * w + J_p * v  (theta-left convention)
    if (jacobians[6]) {
      const Eigen::Vector2d dr_dDelta = Jq.leftCols<3>() * w_clone + Jp * v;
      jacobians[6][0] = dr_dDelta(0);
      jacobians[6][1] = dr_dDelta(1);
    }
    if (jacobians[7]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[7]);
      J = Jp * Delta;
    }
    return true;
  }

private:
  bool prepared_ = false;
  double prepared_delta_ = 0.0;
  Eigen::Vector3d prepared_w_ = Eigen::Vector3d::Zero();
  Eigen::Vector4d prepared_dq_ = Eigen::Vector4d(0, 0, 0, 1);
  Eigen::Matrix3d prepared_rotation_ = Eigen::Matrix3d::Identity();
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_FACTOR_REPROJ_TD_H
