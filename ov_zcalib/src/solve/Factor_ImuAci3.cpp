/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * ov_zcalib: ACI3 preintegration factor (see Factor_ImuAci3.h).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "Factor_ImuAci3.h"

#include <cstdlib>
#include <iostream>

#include "utils/quat_ops.h"

using namespace ov_zcalib;

Factor_ImuAci3::Factor_ImuAci3(const AciPreintResult &meas, const ImuIntrinsicModel &model_lin, const Eigen::Vector3d &bg_lin_,
                               const Eigen::Vector3d &ba_lin_) {
  m = meas;
  bg_lin = bg_lin_;
  ba_lin = ba_lin_;
  dw_lin = model_lin.dw;
  da_lin = model_lin.da;
  qA_lin = model_lin.q_AtoI;
  tg_on = model_lin.calib_tg;
  tg_lin = model_lin.Tg;
  if (m.Jq_pi.cols() != (tg_on ? 24 : 15)) {
    std::cerr << "Factor_ImuAci3: preint columns (" << m.Jq_pi.cols() << ") do not match the model's intrinsic groups ("
              << (tg_on ? 24 : 15) << ": dw6|da6|thA3" << (tg_on ? "|tg9" : "") << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // sqrt information (identical construction to Factor_ImuCPIv1).
  // PARITY CONTRACT: this body must stay VERBATIM in this constructor. Moving
  // it into a helper changed its codegen context under -ffast-math and shifted
  // sqrtI by 1 ulp (measured 2026-07-11: deterministic +1-iteration trajectory
  // divergence entering at A0 vs the pre-P3 binary, propagating to a 1-ulp
  // committed-YAML delta). The preint cache therefore COPIES these members out
  // of a constructed factor (WindowBA) instead of recomputing them elsewhere —
  // cache bytes equal ctor bytes by construction, never by re-derivation.
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(15, 15);
  Eigen::MatrixXd information = m.P15.llt().solve(I);
  Eigen::LLT<Eigen::MatrixXd> lltOfI(information);
  sqrtI = lltOfI.matrixL().transpose();
  sqrtI_grav_fold = m.dt * sqrtI.block<15, 3>(0, 6) + (0.5 * m.dt * m.dt) * sqrtI.block<15, 3>(0, 12);

  set_num_residuals(15);
  for (int s : {4, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 6, 6, 4})
    mutable_parameter_block_sizes()->push_back(s);
  if (tg_on)
    mutable_parameter_block_sizes()->push_back(9);
}

Factor_ImuAci3::Factor_ImuAci3(const AciPreintResult &meas, const ImuIntrinsicModel &model_lin, const Eigen::Vector3d &bg_lin_,
                               const Eigen::Vector3d &ba_lin_, const Eigen::Matrix<double, 15, 15> &sqrtI_ext,
                               const Eigen::Matrix<double, 15, 3> &fold_ext) {
  m = meas;
  bg_lin = bg_lin_;
  ba_lin = ba_lin_;
  dw_lin = model_lin.dw;
  da_lin = model_lin.da;
  qA_lin = model_lin.q_AtoI;
  tg_on = model_lin.calib_tg;
  tg_lin = model_lin.Tg;
  if (m.Jq_pi.cols() != (tg_on ? 24 : 15)) {
    std::cerr << "Factor_ImuAci3: preint columns (" << m.Jq_pi.cols() << ") do not match the model's intrinsic groups ("
              << (tg_on ? 24 : 15) << ": dw6|da6|thA3" << (tg_on ? "|tg9" : "") << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  sqrtI = sqrtI_ext;
  sqrtI_grav_fold = fold_ext;

  set_num_residuals(15);
  for (int s : {4, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 6, 6, 4})
    mutable_parameter_block_sizes()->push_back(s);
  if (tg_on)
    mutable_parameter_block_sizes()->push_back(9);
}

bool Factor_ImuAci3::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  // Dispatch FIRST, so the legacy body below is the LITERAL pre-tg function under the same
  // codegen context (-ffast-math parity: measured 2026-07-13, in-body branching alone reflowed
  // the vectorizer and drifted the down-leg corpus in the 4th decimal).
  if (tg_on)
    return evaluate_tg_(parameters, residuals, jacobians);

  // ---- states ----
  Eigen::Vector4d q_1 = Eigen::Map<const Eigen::Vector4d>(parameters[0]);
  Eigen::Matrix3d R_1 = ov_core::quat_2_Rot(q_1);
  Eigen::Vector4d q_2 = Eigen::Map<const Eigen::Vector4d>(parameters[5]);
  Eigen::Vector3d b_w1 = Eigen::Map<const Eigen::Vector3d>(parameters[1]);
  Eigen::Vector3d b_a1 = Eigen::Map<const Eigen::Vector3d>(parameters[3]);
  Eigen::Vector3d v_1 = Eigen::Map<const Eigen::Vector3d>(parameters[2]);
  Eigen::Vector3d v_2 = Eigen::Map<const Eigen::Vector3d>(parameters[7]);
  Eigen::Vector3d p_1 = Eigen::Map<const Eigen::Vector3d>(parameters[4]);
  Eigen::Vector3d p_2 = Eigen::Map<const Eigen::Vector3d>(parameters[9]);
  Eigen::Vector3d gravity = Eigen::Map<const Eigen::Vector3d>(parameters[10]);

  // ---- calibration deltas from the preintegration linearization point ----
  Eigen::Matrix<double, 6, 1> dw_now = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[11]);
  Eigen::Matrix<double, 6, 1> da_now = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[12]);
  Eigen::Vector4d qA_now = Eigen::Map<const Eigen::Vector4d>(parameters[13]);
  Eigen::Matrix<double, 15, 1> dpi;
  dpi.segment<6>(0) = dw_now - dw_lin;
  dpi.segment<6>(6) = da_now - da_lin;
  dpi.segment<3>(12) = 2.0 * ov_core::quat_multiply(qA_now, ov_core::Inv(qA_lin)).head<3>();

  Eigen::Vector3d dbw = b_w1 - bg_lin;
  Eigen::Vector3d dba = b_a1 - ba_lin;

  // ---- first-order corrected measurement (bias + ACI3 intrinsic columns) ----
  Eigen::Vector3d th_corr = m.J_q * dbw + m.Jq_pi * dpi;
  Eigen::Vector4d q_b;
  q_b.head<3>() = 0.5 * th_corr;
  q_b(3) = 1.0;
  q_b /= q_b.norm();

  Eigen::Vector4d q_1_to_2 = ov_core::quat_multiply(q_2, ov_core::Inv(q_1));
  Eigen::Vector4d q_res_minus = ov_core::quat_multiply(q_1_to_2, ov_core::Inv(m.q_KtoK1));
  Eigen::Vector4d q_res_plus = ov_core::quat_multiply(q_res_minus, q_b);

  Eigen::Matrix<double, 15, 1> res;
  res.segment<3>(0) = 2 * q_res_plus.head<3>();
  res.segment<3>(3) = Eigen::Map<const Eigen::Vector3d>(parameters[6]) - b_w1;
  res.segment<3>(6) = R_1 * (v_2 - v_1 + gravity * m.dt) - m.J_b * dbw - m.H_b * dba - m.Jb_pi * dpi - m.beta;
  res.segment<3>(9) = Eigen::Map<const Eigen::Vector3d>(parameters[8]) - b_a1;
  res.segment<3>(12) = R_1 * (p_2 - p_1 - v_1 * m.dt + 0.5 * gravity * m.dt * m.dt) - m.J_a * dbw - m.H_a * dba - m.Ja_pi * dpi - m.alpha;
  res = sqrtI * res;
  for (int i = 0; i < 15; ++i)
    residuals[i] = res(i);

  if (!jacobians)
    return true;

  const Eigen::Matrix3d eye = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d Lq = q_res_minus(3) * eye - ov_core::skew_x(q_res_minus.head<3>()); // dtheta_res/d(th_corr)

  // Canonical 11 blocks (identical structure to Factor_ImuCPIv1)
  Eigen::Vector4d q_meas_plus = ov_core::quat_multiply(ov_core::Inv(m.q_KtoK1), q_b);
  Eigen::Matrix<double, 15, 30> Jc = Eigen::Matrix<double, 15, 30>::Zero();
  Jc.block<3, 3>(0, 0) = -((q_1_to_2(3) * eye - ov_core::skew_x(q_1_to_2.head<3>())) *
                               (q_meas_plus(3) * eye + ov_core::skew_x(q_meas_plus.head<3>())) -
                           q_1_to_2.head<3>() * q_meas_plus.head<3>().transpose());
  Jc.block<3, 3>(0, 15) = q_res_plus(3) * eye + ov_core::skew_x(q_res_plus.head<3>());
  Jc.block<3, 3>(0, 3) = Lq * m.J_q;
  Jc.block<3, 3>(3, 3) = -eye;
  Jc.block<3, 3>(3, 18) = eye;
  Jc.block<3, 3>(6, 0) = ov_core::skew_x(R_1 * (v_2 - v_1 + gravity * m.dt));
  Jc.block<3, 3>(6, 6) = -R_1;
  Jc.block<3, 3>(6, 21) = R_1;
  Jc.block<3, 3>(6, 3) = -m.J_b;
  Jc.block<3, 3>(6, 9) = -m.H_b;
  Jc.block<3, 3>(9, 9) = -eye;
  Jc.block<3, 3>(9, 24) = eye;
  Jc.block<3, 3>(12, 0) = ov_core::skew_x(R_1 * (p_2 - p_1 - v_1 * m.dt + 0.5 * gravity * m.dt * m.dt));
  Jc.block<3, 3>(12, 6) = -R_1 * m.dt;
  Jc.block<3, 3>(12, 12) = -R_1;
  Jc.block<3, 3>(12, 27) = R_1;
  Jc.block<3, 3>(12, 3) = -m.J_a;
  Jc.block<3, 3>(12, 9) = -m.H_a;
  Jc = sqrtI * Jc;

  auto store = [&](int j, int col, int width, int gsize) {
    if (!jacobians[j])
      return;
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[j], 15, gsize);
    J.setZero();
    J.leftCols(width) = Jc.middleCols(col, width);
  };
  store(0, 0, 3, 4);
  store(1, 3, 3, 3);
  store(2, 6, 3, 3);
  store(3, 9, 3, 3);
  store(4, 12, 3, 3);
  store(5, 15, 3, 4);
  store(6, 18, 3, 3);
  store(7, 21, 3, 3);
  store(8, 24, 3, 3);
  store(9, 27, 3, 3);
  if (jacobians[10]) {
    Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J_grav(jacobians[10], 15, 3);
    J_grav.noalias() = sqrtI_grav_fold * R_1;
  }

  // ---- ACI3 intrinsic blocks: identical structure to the bias-1 columns ----
  // dres/dpi = sqrtI * [ Lq*Jq_pi ; 0 ; -Jb_pi ; 0 ; -Ja_pi ]
  Eigen::Matrix<double, 15, 15> Jpi = Eigen::Matrix<double, 15, 15>::Zero();
  Jpi.block(0, 0, 3, 15) = Lq * m.Jq_pi;
  Jpi.block(6, 0, 3, 15) = -m.Jb_pi;
  Jpi.block(12, 0, 3, 15) = -m.Ja_pi;
  Jpi = sqrtI * Jpi;
  if (jacobians[11]) {
    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J(jacobians[11], 15, 6);
    J = Jpi.middleCols(0, 6);
  }
  if (jacobians[12]) {
    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J(jacobians[12], 15, 6);
    J = Jpi.middleCols(6, 6);
  }
  if (jacobians[13]) {
    Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> J(jacobians[13], 15, 4);
    J.setZero();
    J.leftCols(3) = Jpi.middleCols(12, 3);
  }
  return true;
}

// ---- Tg-enabled evaluation (n_pi = 24; parameter block [14] = tg9 in Matrix3d storage order).
// A SEPARATE function on purpose: the legacy body above must keep its exact emitted code (see the
// dispatch note), and this path is new arithmetic with no parity contract to honor. Structure
// mirrors the legacy body with dpi widened to 24 and the D12 ba->rotation coupling (H_q) applied
// to the theta correction and its ba-block Jacobian.
bool Factor_ImuAci3::evaluate_tg_(double const *const *parameters, double *residuals, double **jacobians) const {

  // ---- states ----
  Eigen::Vector4d q_1 = Eigen::Map<const Eigen::Vector4d>(parameters[0]);
  Eigen::Matrix3d R_1 = ov_core::quat_2_Rot(q_1);
  Eigen::Vector4d q_2 = Eigen::Map<const Eigen::Vector4d>(parameters[5]);
  Eigen::Vector3d b_w1 = Eigen::Map<const Eigen::Vector3d>(parameters[1]);
  Eigen::Vector3d b_a1 = Eigen::Map<const Eigen::Vector3d>(parameters[3]);
  Eigen::Vector3d v_1 = Eigen::Map<const Eigen::Vector3d>(parameters[2]);
  Eigen::Vector3d v_2 = Eigen::Map<const Eigen::Vector3d>(parameters[7]);
  Eigen::Vector3d p_1 = Eigen::Map<const Eigen::Vector3d>(parameters[4]);
  Eigen::Vector3d p_2 = Eigen::Map<const Eigen::Vector3d>(parameters[9]);
  Eigen::Vector3d gravity = Eigen::Map<const Eigen::Vector3d>(parameters[10]);

  // ---- calibration deltas from the preintegration linearization point ----
  Eigen::Matrix<double, 6, 1> dw_now = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[11]);
  Eigen::Matrix<double, 6, 1> da_now = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[12]);
  Eigen::Vector4d qA_now = Eigen::Map<const Eigen::Vector4d>(parameters[13]);
  Eigen::Matrix<double, 24, 1> dpi;
  dpi.segment<6>(0) = dw_now - dw_lin;
  dpi.segment<6>(6) = da_now - da_lin;
  dpi.segment<3>(12) = 2.0 * ov_core::quat_multiply(qA_now, ov_core::Inv(qA_lin)).head<3>();
  {
    // PI ORDER RECONCILIATION. The preint tg columns follow mixing()'s ROW-major element
    // enumeration (Ek(k/3, k%3) — kept at its legacy shape for -ffast-math parity), while the
    // parameter block is Tg.data() = Matrix3d COLUMN-major storage. The permutation lives HERE,
    // at the factor boundary, and nowhere else: dpi is built in PI order from the storage view,
    // and jacobians[14] maps pi columns back to storage-element columns below.
    const Eigen::Map<const Eigen::Matrix3d> Tg_now(parameters[14]);
    for (int k = 0; k < 9; ++k)
      dpi(15 + k) = Tg_now(k / 3, k % 3) - tg_lin(k / 3, k % 3);
  }

  Eigen::Vector3d dbw = b_w1 - bg_lin;
  Eigen::Vector3d dba = b_a1 - ba_lin;

  // ---- first-order corrected measurement (bias + ACI3 intrinsic columns + D12 H_q) ----
  Eigen::Vector3d th_corr = m.J_q * dbw + m.H_q * dba + m.Jq_pi * dpi;
  Eigen::Vector4d q_b;
  q_b.head<3>() = 0.5 * th_corr;
  q_b(3) = 1.0;
  q_b /= q_b.norm();

  Eigen::Vector4d q_1_to_2 = ov_core::quat_multiply(q_2, ov_core::Inv(q_1));
  Eigen::Vector4d q_res_minus = ov_core::quat_multiply(q_1_to_2, ov_core::Inv(m.q_KtoK1));
  Eigen::Vector4d q_res_plus = ov_core::quat_multiply(q_res_minus, q_b);

  Eigen::Matrix<double, 15, 1> res;
  res.segment<3>(0) = 2 * q_res_plus.head<3>();
  res.segment<3>(3) = Eigen::Map<const Eigen::Vector3d>(parameters[6]) - b_w1;
  res.segment<3>(6) = R_1 * (v_2 - v_1 + gravity * m.dt) - m.J_b * dbw - m.H_b * dba - m.Jb_pi * dpi - m.beta;
  res.segment<3>(9) = Eigen::Map<const Eigen::Vector3d>(parameters[8]) - b_a1;
  res.segment<3>(12) = R_1 * (p_2 - p_1 - v_1 * m.dt + 0.5 * gravity * m.dt * m.dt) - m.J_a * dbw - m.H_a * dba - m.Ja_pi * dpi - m.alpha;
  res = sqrtI * res;
  for (int i = 0; i < 15; ++i)
    residuals[i] = res(i);

  if (!jacobians)
    return true;

  const Eigen::Matrix3d eye = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d Lq = q_res_minus(3) * eye - ov_core::skew_x(q_res_minus.head<3>()); // dtheta_res/d(th_corr)

  Eigen::Vector4d q_meas_plus = ov_core::quat_multiply(ov_core::Inv(m.q_KtoK1), q_b);
  Eigen::Matrix<double, 15, 30> Jc = Eigen::Matrix<double, 15, 30>::Zero();
  Jc.block<3, 3>(0, 0) = -((q_1_to_2(3) * eye - ov_core::skew_x(q_1_to_2.head<3>())) *
                               (q_meas_plus(3) * eye + ov_core::skew_x(q_meas_plus.head<3>())) -
                           q_1_to_2.head<3>() * q_meas_plus.head<3>().transpose());
  Jc.block<3, 3>(0, 15) = q_res_plus(3) * eye + ov_core::skew_x(q_res_plus.head<3>());
  Jc.block<3, 3>(0, 3) = Lq * m.J_q;
  Jc.block<3, 3>(0, 9) = Lq * m.H_q; // D12: ba reaches the rotation through Tg
  Jc.block<3, 3>(3, 3) = -eye;
  Jc.block<3, 3>(3, 18) = eye;
  Jc.block<3, 3>(6, 0) = ov_core::skew_x(R_1 * (v_2 - v_1 + gravity * m.dt));
  Jc.block<3, 3>(6, 6) = -R_1;
  Jc.block<3, 3>(6, 21) = R_1;
  Jc.block<3, 3>(6, 3) = -m.J_b;
  Jc.block<3, 3>(6, 9) = -m.H_b;
  Jc.block<3, 3>(9, 9) = -eye;
  Jc.block<3, 3>(9, 24) = eye;
  Jc.block<3, 3>(12, 0) = ov_core::skew_x(R_1 * (p_2 - p_1 - v_1 * m.dt + 0.5 * gravity * m.dt * m.dt));
  Jc.block<3, 3>(12, 6) = -R_1 * m.dt;
  Jc.block<3, 3>(12, 12) = -R_1;
  Jc.block<3, 3>(12, 27) = R_1;
  Jc.block<3, 3>(12, 3) = -m.J_a;
  Jc.block<3, 3>(12, 9) = -m.H_a;
  Jc = sqrtI * Jc;

  auto store = [&](int j, int col, int width, int gsize) {
    if (!jacobians[j])
      return;
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[j], 15, gsize);
    J.setZero();
    J.leftCols(width) = Jc.middleCols(col, width);
  };
  store(0, 0, 3, 4);
  store(1, 3, 3, 3);
  store(2, 6, 3, 3);
  store(3, 9, 3, 3);
  store(4, 12, 3, 3);
  store(5, 15, 3, 4);
  store(6, 18, 3, 3);
  store(7, 21, 3, 3);
  store(8, 24, 3, 3);
  store(9, 27, 3, 3);
  if (jacobians[10]) {
    Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J_grav(jacobians[10], 15, 3);
    J_grav.noalias() = sqrtI_grav_fold * R_1;
  }

  Eigen::Matrix<double, 15, 24> Jpi = Eigen::Matrix<double, 15, 24>::Zero();
  Jpi.block(0, 0, 3, 24) = Lq * m.Jq_pi;
  Jpi.block(6, 0, 3, 24) = -m.Jb_pi;
  Jpi.block(12, 0, 3, 24) = -m.Ja_pi;
  Jpi = sqrtI * Jpi;
  if (jacobians[11]) {
    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J(jacobians[11], 15, 6);
    J = Jpi.middleCols(0, 6);
  }
  if (jacobians[12]) {
    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J(jacobians[12], 15, 6);
    J = Jpi.middleCols(6, 6);
  }
  if (jacobians[13]) {
    Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> J(jacobians[13], 15, 4);
    J.setZero();
    J.leftCols(3) = Jpi.middleCols(12, 3);
  }
  if (jacobians[14]) {
    Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J(jacobians[14], 15, 9);
    // storage element j sits at (row j%3, col j/3) = pi row-major index 3*(j%3) + j/3
    for (int j = 0; j < 9; ++j)
      J.col(j) = Jpi.col(15 + 3 * (j % 3) + j / 3);
  }
  return true;
}
