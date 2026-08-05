/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * ov_zcalib: ACI3 calibration preintegration sweep (see AciCalibPreint.h).
 *
 * The mean/bias recursions mirror the verified async-bridge sweep (left/end-frame
 * J_theta with the exact factored increment R_step*(J + Jr(-w dt)*M*dt)); the
 * intrinsic columns generalize the bias columns through per-sample mixing matrices;
 * the 15x15 covariance is propagated per step in (theta, alpha, beta, bg, ba) order
 * with the same Phi/G blocks the columns imply (noise enters exactly like the
 * corresponding bias), then permuted to the Factor_ImuCPIv1 residual order.
 * Every column and the covariance are pinned by the FD oracle in test_aci3_fd.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "AciCalibPreint.h"

#include <cmath>

#include "utils/quat_ops.h"

using namespace ov_zcalib;

void AciCalibPreint::step_integrals(double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat, Eigen::Matrix3d &R_step,
                                    Eigen::Matrix3d &Jr_neg, Eigen::Matrix3d &Xi_1, Eigen::Matrix3d &Xi_2, Eigen::Matrix3d &T1,
                                    Eigen::Matrix3d &T2) {

  // Closed-form ACI2 integrals (Yang et al.; identical math to the propagator's
  // compute_Xi_sum, re-implemented here so ov_zcalib carries zero ov_msckf coupling).
  const double w_norm = w_hat.norm();
  const double d_th = w_norm * dt;
  Eigen::Vector3d k_hat = Eigen::Vector3d::Zero();
  if (w_norm > 1e-12)
    k_hat = w_hat / w_norm;

  const Eigen::Matrix3d I_3x3 = Eigen::Matrix3d::Identity();
  const double d_t2 = dt * dt, d_t3 = d_t2 * dt;
  const double w_norm2 = w_norm * w_norm, w_norm3 = w_norm2 * w_norm;
  const double cos_dth = std::cos(d_th), sin_dth = std::sin(d_th);
  const double d_th2 = d_th * d_th, d_th3 = d_th2 * d_th;
  const Eigen::Matrix3d sK = ov_core::skew_x(k_hat);
  const Eigen::Matrix3d sK2 = sK * sK;
  const Eigen::Matrix3d sA = ov_core::skew_x(a_hat);

  R_step = ov_core::exp_so3(-w_hat * dt);
  Jr_neg = ov_core::Jr_so3(-w_hat * dt);

  Eigen::Matrix3d Xi_3, Xi_4;
  const bool small_w = (w_norm < 1.0 / 180 * M_PI / 2);
  if (!small_w) {
    Xi_1 = I_3x3 * dt + (1.0 - cos_dth) / w_norm * sK + (dt - sin_dth / w_norm) * sK2;
    Xi_2 = 1.0 / 2 * d_t2 * I_3x3 + (d_th - sin_dth) / w_norm2 * sK + (1.0 / 2 * d_t2 - (1.0 - cos_dth) / w_norm2) * sK2;
    Xi_3 = 1.0 / 2 * d_t2 * sA + (sin_dth - d_th) / w_norm2 * sA * sK + (sin_dth - d_th * cos_dth) / w_norm2 * sK * sA +
           (1.0 / 2 * d_t2 - (1.0 - cos_dth) / w_norm2) * sA * sK2 +
           (1.0 / 2 * d_t2 + (1.0 - cos_dth - d_th * sin_dth) / w_norm2) * (sK2 * sA + k_hat.dot(a_hat) * sK) -
           (3 * sin_dth - 2 * d_th - d_th * cos_dth) / w_norm2 * k_hat.dot(a_hat) * sK2;
    Xi_4 = 1.0 / 6 * d_t3 * sA + (2 * (1.0 - cos_dth) - d_th2) / (2 * w_norm3) * sA * sK +
           ((2 * (1.0 - cos_dth) - d_th * sin_dth) / w_norm3) * sK * sA + ((sin_dth - d_th) / w_norm3 + d_t3 / 6) * sA * sK2 +
           ((d_th - 2 * sin_dth + 1.0 / 6 * d_th3 + d_th * cos_dth) / w_norm3) * (sK2 * sA + k_hat.dot(a_hat) * sK) +
           (4 * cos_dth - 4 + d_th2 + d_th * sin_dth) / w_norm3 * k_hat.dot(a_hat) * sK2;
  } else {
    Xi_1 = dt * (I_3x3 + sin_dth * sK + (1.0 - cos_dth) * sK2);
    Xi_2 = 1.0 / 2 * dt * Xi_1;
    Xi_3 = 1.0 / 2 * d_t2 *
           (sA + sin_dth * (-sA * sK + sK * sA + k_hat.dot(a_hat) * sK2) + (1.0 - cos_dth) * (sA * sK2 + sK2 * sA + k_hat.dot(a_hat) * sK));
    Xi_4 = 1.0 / 3 * dt * Xi_3;
  }

  // Within-step omega-sensitivities of (Xi_1*a, Xi_2*a). The ACI2 Xi_3/Xi_4 are these
  // sensitivities contracted with the RAW-bias mixing d(w_hat)/d(bg) = -I, hence:
  T1 = -Xi_3;
  T2 = -Xi_4;
}

bool AciCalibPreint::integrate(const std::vector<RawImu> &imu, double t0, double t1, const ImuIntrinsicModel &model,
                               const Eigen::Vector3d &bg_lin, const Eigen::Vector3d &ba_lin, const ImuNoise &noise, AciPreintResult &out,
                               const ImuIntrinsicModel *noise_model) {
  const ImuIntrinsicModel &nm = noise_model ? *noise_model : model;

  if (!(t1 > t0) || imu.size() < 2)
    return false;

  // Collect boundary-interpolated samples spanning [t0, t1] (mirrors select_imu_readings)
  std::vector<RawImu> data;
  data.reserve(imu.size());
  for (size_t i = 0; i + 1 < imu.size(); ++i) {
    const RawImu &s0 = imu[i], &s1 = imu[i + 1];
    if (s1.timestamp <= t0 || s0.timestamp >= t1)
      continue;
    auto interp = [&](double t) {
      const double lam = (t - s0.timestamp) / (s1.timestamp - s0.timestamp);
      RawImu r;
      r.timestamp = t;
      r.wm = (1 - lam) * s0.wm + lam * s1.wm;
      r.am = (1 - lam) * s0.am + lam * s1.am;
      return r;
    };
    if (data.empty())
      data.push_back(s0.timestamp < t0 ? interp(t0) : s0);
    data.push_back(s1.timestamp > t1 ? interp(t1) : s1);
    if (s1.timestamp > t1)
      break;
  }
  if (data.size() < 2)
    return false;

  const int n_pi = model.num_params();
  const int n_cols = n_pi + 6; // [ intrinsic groups | bg | ba ]

  // Accumulators (start-frame alpha/beta; DR = R_{I_end <- I_start}); left-convention J_theta
  Eigen::Matrix3d DR = Eigen::Matrix3d::Identity();
  Eigen::Vector3d alpha = Eigen::Vector3d::Zero(), beta = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 3, Eigen::Dynamic> J_th(3, n_cols), J_al(3, n_cols), J_be(3, n_cols);
  J_th.setZero();
  J_al.setZero();
  J_be.setZero();

  // Covariance state order: [theta(0) alpha(3) beta(6) bg(9) ba(12)]
  Eigen::Matrix<double, 15, 15> P = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 15, 15> Phi;
  Eigen::Matrix<double, 15, 6> G;

  Eigen::Matrix<double, 3, Eigen::Dynamic> M_w, M_a;
  Eigen::Matrix3d Mw_bg, Mw_ba, Ma_ba;
  Eigen::Matrix3d R_step, Jr_neg, Xi_1, Xi_2, T1, T2;

  for (size_t i = 0; i + 1 < data.size(); ++i) {
    const double dt = data[i + 1].timestamp - data[i].timestamp;
    if (!(dt > 0))
      continue;

    // Midpoint raw signals; corrected via the intrinsic model at the linearization point
    const Eigen::Vector3d wm = 0.5 * (data[i].wm + data[i + 1].wm);
    const Eigen::Vector3d am = 0.5 * (data[i].am + data[i + 1].am);
    Eigen::Vector3d w_hat, a_hat;
    model.correct(wm, am, bg_lin, ba_lin, w_hat, a_hat);
    model.mixing(wm, am, bg_lin, ba_lin, M_w, M_a, Mw_bg, Mw_ba, Ma_ba);
    out.w_end = w_hat;
    out.a_end = a_hat;

    step_integrals(dt, w_hat, a_hat, R_step, Jr_neg, Xi_1, Xi_2, T1, T2);
    const Eigen::Matrix3d A = DR.transpose(); // step-start frame -> window-start frame
    const Eigen::Vector3d X1a = Xi_1 * a_hat;
    const Eigen::Vector3d X2a = Xi_2 * a_hat;
    const Eigen::Matrix3d AsX1 = A * ov_core::skew_x(X1a);
    const Eigen::Matrix3d AsX2 = A * ov_core::skew_x(X2a);

    // Stacked per-step mixing over [ pi | bg | ba ]
    Eigen::Matrix<double, 3, Eigen::Dynamic> Mw_all(3, n_cols), Ma_all(3, n_cols);
    Mw_all.leftCols(n_pi) = M_w;
    Ma_all.leftCols(n_pi) = M_a;
    Mw_all.middleCols(n_pi, 3) = Mw_bg;
    Ma_all.middleCols(n_pi, 3).setZero();
    Mw_all.middleCols(n_pi + 3, 3) = Mw_ba;
    Ma_all.middleCols(n_pi + 3, 3) = Ma_ba;

    // ---- parameter columns (consume PRE-update J_th/J_be, then advance; bridge order) ----
    // Chain signs: d(theta_step)/d(w_hat) = -Jr(+w dt)*dt (the bridge's +Jr already has the
    // raw-bias mixing d(w_hat)/d(bg) = -I folded in; here the mixing is explicit in Mw_all).
    J_al += J_be * dt + A * (Xi_2 * Ma_all + T2 * Mw_all) + AsX2 * J_th;
    J_be += A * (Xi_1 * Ma_all + T1 * Mw_all) + AsX1 * J_th;
    J_th = R_step * (J_th - Jr_neg * Mw_all * dt); // exact factored -Jr(+w dt) increment

    // ---- covariance: Phi/G share the identical per-step blocks (FROZEN noise model) ----
    if (noise_model) {
      Eigen::Matrix<double, 3, Eigen::Dynamic> Mw_n, Ma_n;
      nm.mixing(wm, am, bg_lin, ba_lin, Mw_n, Ma_n, Mw_bg, Mw_ba, Ma_ba);
    }
    Phi.setIdentity();
    Phi.block<3, 3>(0, 0) = R_step;
    Phi.block<3, 3>(0, 9) = -R_step * (Jr_neg * Mw_bg * dt);
    // Tg feeds a_hat into w_hat (Mw_ba = -Dw*Tg*Ma_ba), so ba -- and n_a, below -- reach the
    // ROTATION as well, exactly the way bg does. Both blocks are identically zero at Tg == 0,
    // which is the only reason they were omissible while Tg was forced off (defect D12).
    Phi.block<3, 3>(0, 12) = -R_step * (Jr_neg * Mw_ba * dt);
    Phi.block<3, 3>(3, 0) = AsX2;
    Phi.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity();
    Phi.block<3, 3>(3, 9) = A * (T2 * Mw_bg);
    Phi.block<3, 3>(3, 12) = A * (Xi_2 * Ma_ba + T2 * Mw_ba);
    Phi.block<3, 3>(6, 0) = AsX1;
    Phi.block<3, 3>(6, 9) = A * (T1 * Mw_bg);
    Phi.block<3, 3>(6, 12) = A * (Xi_1 * Ma_ba + T1 * Mw_ba);
    G.setZero();
    G.block<3, 3>(0, 0) = Phi.block<3, 3>(0, 9);  // n_g enters exactly like bg
    G.block<3, 3>(0, 3) = Phi.block<3, 3>(0, 12); // n_a enters exactly like ba
    G.block<3, 3>(3, 0) = Phi.block<3, 3>(3, 9);
    G.block<3, 3>(3, 3) = Phi.block<3, 3>(3, 12); // n_a enters exactly like ba
    G.block<3, 3>(6, 0) = Phi.block<3, 3>(6, 9);
    G.block<3, 3>(6, 3) = Phi.block<3, 3>(6, 12);
    Eigen::Matrix<double, 6, 6> Qd = Eigen::Matrix<double, 6, 6>::Zero();
    Qd.block<3, 3>(0, 0) = (noise.sigma_w * noise.sigma_w / dt) * Eigen::Matrix3d::Identity();
    Qd.block<3, 3>(3, 3) = (noise.sigma_a * noise.sigma_a / dt) * Eigen::Matrix3d::Identity();
    P = Phi * P * Phi.transpose() + G * Qd * G.transpose();
    P.block<3, 3>(9, 9) += (noise.sigma_wb * noise.sigma_wb * dt) * Eigen::Matrix3d::Identity();
    P.block<3, 3>(12, 12) += (noise.sigma_ab * noise.sigma_ab * dt) * Eigen::Matrix3d::Identity();

    // ---- mean (alpha consumes the PRE-update beta) ----
    alpha += beta * dt + A * X2a;
    beta += A * X1a;
    DR = R_step * DR;
  }

  // ---- emit in Factor_ImuCPIv1 conventions ----
  out.dt = t1 - t0;
  out.q_KtoK1 = ov_core::rot_2_quat(DR);
  out.alpha = alpha;
  out.beta = beta;
  // Bias Jacobians: theta column is already the factor's q_b convention (FD-oracle pinned)
  out.J_q = J_th.middleCols(n_pi, 3);
  out.H_b = J_be.middleCols(n_pi + 3, 3);
  out.H_a = J_al.middleCols(n_pi + 3, 3);
  out.H_q = J_th.middleCols(n_pi + 3, 3); // zero unless Tg (D12: ba -> rotation)
  out.J_b = J_be.middleCols(n_pi, 3);
  out.J_a = J_al.middleCols(n_pi, 3);
  // Intrinsic columns
  out.Jq_pi = J_th.leftCols(n_pi);
  out.Jb_pi = J_be.leftCols(n_pi);
  out.Ja_pi = J_al.leftCols(n_pi);
  // Covariance -> residual order [theta bg beta ba alpha]
  const int perm[5] = {0, 3, 2, 4, 1}; // source 3-blocks (theta, alpha, beta, bg, ba) -> target
  Eigen::Matrix<double, 15, 15> Pi = Eigen::Matrix<double, 15, 15>::Zero();
  const int src_of_tgt[5] = {0, 3, 2, 4, 1}; // target block t reads source block src_of_tgt[t]
  (void)perm;
  for (int t = 0; t < 5; ++t)
    Pi.block<3, 3>(3 * t, 3 * src_of_tgt[t]) = Eigen::Matrix3d::Identity();
  out.P15 = Pi * P * Pi.transpose();
  out.P15 += 1e-12 * Eigen::Matrix<double, 15, 15>::Identity();
  return true;
}

bool AciCalibPreint::integrate_chain(const std::vector<RawImu> &imu, const std::vector<double> &clone_times,
                                     const ImuIntrinsicModel &model, const Eigen::Vector3d &bg_lin, const Eigen::Vector3d &ba_lin,
                                     const ImuNoise &noise, std::vector<AciPreintResult> &out, const ImuIntrinsicModel *noise_model,
                                     bool skip_cov) {
  // BIT-PARITY CONTRACT with integrate(): the boundary clamping below mirrors
  // integrate()'s pair-scan (accept if s1.t > t0 && s0.t < t1; clamp the first
  // entry to t0 and the last to t1 with the SAME straddling pair and the SAME
  // lambda expression), and the step body is a verbatim copy of integrate()'s
  // sample loop. Any edit to either body must be mirrored and re-pinned by
  // test_preint_chain. Assumes chronologically NON-DECREASING sample stamps
  // (the select_imu_readings precondition integrate()'s scan shares).
  const ImuIntrinsicModel &nm = noise_model ? *noise_model : model;
  const int N = (int)clone_times.size();
  if (N < 2 || imu.size() < 2)
    return false;
  out.assign(N - 1, AciPreintResult());

  const int n_pi = model.num_params();
  const int n_cols = n_pi + 6; // [ intrinsic groups | bg | ba ]

  // Hoisted work buffers: the per-interval accumulators and the per-step
  // dynamic temporaries integrate() re-allocates (every column is fully
  // overwritten before use, so reuse is value-identical).
  Eigen::Matrix<double, 3, Eigen::Dynamic> J_th(3, n_cols), J_al(3, n_cols), J_be(3, n_cols);
  Eigen::Matrix<double, 15, 15> P;
  Eigen::Matrix<double, 15, 15> Phi;
  Eigen::Matrix<double, 15, 6> G;
  Eigen::Matrix<double, 3, Eigen::Dynamic> M_w, M_a, Mw_n, Ma_n;
  Eigen::Matrix<double, 3, Eigen::Dynamic> Mw_all(3, n_cols), Ma_all(3, n_cols);
  Eigen::Matrix3d Mw_bg, Mw_ba, Ma_ba;
  Eigen::Matrix3d R_step, Jr_neg, Xi_1, Xi_2, T1, T2;
  Eigen::Matrix<double, 6, 6> Qd = Eigen::Matrix<double, 6, 6>::Zero(); // off-diagonal blocks stay zero
  Eigen::Matrix<double, 15, 15> Pi = Eigen::Matrix<double, 15, 15>::Zero();
  const int src_of_tgt[5] = {0, 3, 2, 4, 1};
  for (int t = 0; t < 5; ++t)
    Pi.block<3, 3>(3 * t, 3 * src_of_tgt[t]) = Eigen::Matrix3d::Identity();

  size_t cur = 0; // monotone pair cursor: first pair (cur, cur+1) with imu[cur+1].t > interval start
  for (int k = 0; k + 1 < N; ++k) {
    const double t0 = clone_times[k], t1 = clone_times[k + 1];
    if (!(t1 > t0))
      return false;
    while (cur + 1 < imu.size() && imu[cur + 1].timestamp <= t0)
      ++cur;
    if (cur + 1 >= imu.size() || imu[cur].timestamp >= t1)
      return false; // no pair spans into [t0, t1]: integrate()'s data.size() < 2 failure

    // interval-local accumulators (reset per interval, allocations hoisted)
    Eigen::Matrix3d DR = Eigen::Matrix3d::Identity();
    Eigen::Vector3d alpha = Eigen::Vector3d::Zero(), beta = Eigen::Vector3d::Zero();
    J_th.setZero();
    J_al.setZero();
    J_be.setZero();
    P.setZero();
    AciPreintResult &o = out[k];

    // first entry: clamp to t0 from the cursor pair exactly as integrate()'s
    // first accepted pair (an interior clone boundary re-derives the SAME
    // interpolant bytes the previous interval's end clamp produced)
    RawImu x0;
    {
      const RawImu &s0 = imu[cur], &s1 = imu[cur + 1];
      if (s0.timestamp < t0) {
        const double lam = (t0 - s0.timestamp) / (s1.timestamp - s0.timestamp);
        x0.timestamp = t0;
        x0.wm = (1 - lam) * s0.wm + lam * s1.wm;
        x0.am = (1 - lam) * s0.am + lam * s1.am;
      } else {
        x0 = s0;
      }
    }

    for (size_t j = cur; j + 1 < imu.size() && imu[j].timestamp < t1; ++j) {
      const RawImu &s0 = imu[j], &s1 = imu[j + 1];
      RawImu x1;
      const bool clamp_end = (s1.timestamp > t1);
      if (clamp_end) {
        const double lam = (t1 - s0.timestamp) / (s1.timestamp - s0.timestamp);
        x1.timestamp = t1;
        x1.wm = (1 - lam) * s0.wm + lam * s1.wm;
        x1.am = (1 - lam) * s0.am + lam * s1.am;
      } else {
        x1 = s1;
      }

      const double dt = x1.timestamp - x0.timestamp;
      if (dt > 0) {
        // ---- step body: verbatim copy of integrate()'s sample loop ----
        const Eigen::Vector3d wm = 0.5 * (x0.wm + x1.wm);
        const Eigen::Vector3d am = 0.5 * (x0.am + x1.am);
        Eigen::Vector3d w_hat, a_hat;
        model.correct(wm, am, bg_lin, ba_lin, w_hat, a_hat);
        model.mixing(wm, am, bg_lin, ba_lin, M_w, M_a, Mw_bg, Mw_ba, Ma_ba);
        o.w_end = w_hat;
        o.a_end = a_hat;

        step_integrals(dt, w_hat, a_hat, R_step, Jr_neg, Xi_1, Xi_2, T1, T2);
        const Eigen::Matrix3d A = DR.transpose(); // step-start frame -> window-start frame
        const Eigen::Vector3d X1a = Xi_1 * a_hat;
        const Eigen::Vector3d X2a = Xi_2 * a_hat;
        const Eigen::Matrix3d AsX1 = A * ov_core::skew_x(X1a);
        const Eigen::Matrix3d AsX2 = A * ov_core::skew_x(X2a);

        Mw_all.leftCols(n_pi) = M_w;
        Ma_all.leftCols(n_pi) = M_a;
        Mw_all.middleCols(n_pi, 3) = Mw_bg;
        Ma_all.middleCols(n_pi, 3).setZero();
        Mw_all.middleCols(n_pi + 3, 3) = Mw_ba;
        Ma_all.middleCols(n_pi + 3, 3) = Ma_ba;

        J_al += J_be * dt + A * (Xi_2 * Ma_all + T2 * Mw_all) + AsX2 * J_th;
        J_be += A * (Xi_1 * Ma_all + T1 * Mw_all) + AsX1 * J_th;
        J_th = R_step * (J_th - Jr_neg * Mw_all * dt); // exact factored -Jr(+w dt) increment

        if (!skip_cov) {
          if (noise_model) {
            nm.mixing(wm, am, bg_lin, ba_lin, Mw_n, Ma_n, Mw_bg, Mw_ba, Ma_ba);
          }
          Phi.setIdentity();
          Phi.block<3, 3>(0, 0) = R_step;
          Phi.block<3, 3>(0, 9) = -R_step * (Jr_neg * Mw_bg * dt);
          Phi.block<3, 3>(0, 12) = -R_step * (Jr_neg * Mw_ba * dt); // see the serial path (D12)
          Phi.block<3, 3>(3, 0) = AsX2;
          Phi.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity();
          Phi.block<3, 3>(3, 9) = A * (T2 * Mw_bg);
          Phi.block<3, 3>(3, 12) = A * (Xi_2 * Ma_ba + T2 * Mw_ba);
          Phi.block<3, 3>(6, 0) = AsX1;
          Phi.block<3, 3>(6, 9) = A * (T1 * Mw_bg);
          Phi.block<3, 3>(6, 12) = A * (Xi_1 * Ma_ba + T1 * Mw_ba);
          G.setZero();
          G.block<3, 3>(0, 0) = Phi.block<3, 3>(0, 9);  // n_g enters exactly like bg
          G.block<3, 3>(0, 3) = Phi.block<3, 3>(0, 12); // n_a enters exactly like ba
          G.block<3, 3>(3, 0) = Phi.block<3, 3>(3, 9);
          G.block<3, 3>(3, 3) = Phi.block<3, 3>(3, 12); // n_a enters exactly like ba
          G.block<3, 3>(6, 0) = Phi.block<3, 3>(6, 9);
          G.block<3, 3>(6, 3) = Phi.block<3, 3>(6, 12);
          Qd.block<3, 3>(0, 0) = (noise.sigma_w * noise.sigma_w / dt) * Eigen::Matrix3d::Identity();
          Qd.block<3, 3>(3, 3) = (noise.sigma_a * noise.sigma_a / dt) * Eigen::Matrix3d::Identity();
          P = Phi * P * Phi.transpose() + G * Qd * G.transpose();
          P.block<3, 3>(9, 9) += (noise.sigma_wb * noise.sigma_wb * dt) * Eigen::Matrix3d::Identity();
          P.block<3, 3>(12, 12) += (noise.sigma_ab * noise.sigma_ab * dt) * Eigen::Matrix3d::Identity();
        }

        alpha += beta * dt + A * X2a;
        beta += A * X1a;
        DR = R_step * DR;
      }

      x0 = x1;
      if (clamp_end)
        break; // integrate() breaks after appending the t1 clamp
    }

    // ---- emit in Factor_ImuCPIv1 conventions (identical to integrate()) ----
    o.dt = t1 - t0;
    o.q_KtoK1 = ov_core::rot_2_quat(DR);
    o.alpha = alpha;
    o.beta = beta;
    o.J_q = J_th.middleCols(n_pi, 3);
    o.H_b = J_be.middleCols(n_pi + 3, 3);
    o.H_a = J_al.middleCols(n_pi + 3, 3);
    o.H_q = J_th.middleCols(n_pi + 3, 3); // zero unless Tg (D12: ba -> rotation)
    o.J_b = J_be.middleCols(n_pi, 3);
    o.J_a = J_al.middleCols(n_pi, 3);
    o.Jq_pi = J_th.leftCols(n_pi);
    o.Jb_pi = J_be.leftCols(n_pi);
    o.Ja_pi = J_al.leftCols(n_pi);
    if (!skip_cov) {
      o.P15 = Pi * P * Pi.transpose();
      o.P15 += 1e-12 * Eigen::Matrix<double, 15, 15>::Identity();
    }
  }
  return true;
}
