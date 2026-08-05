/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: linear window seeder implementation (see LinearSeed.h).
 *
 * The accel bias is a LINEAR unknown here: the ACI sweep exports the analytic
 * bias Jacobians H_a = d(alpha)/d(ba), H_b = d(beta)/d(ba), so the kinematic
 * unroll carries JS/JV recursions and the joint LS solves [v0, g, ba, {p_f}]
 * with a physical-scale Tikhonov prior on ba (sigma ~0.05 m/s^2) that keeps
 * short/degenerate windows well-posed. Leaving ba out costs ~0.5*|ba|*T^2 of
 * position drift (0.1+ m at 3 s) — the dominant seed error. The gyro bias
 * stays FIXED at the bootstrap value: it is already hand-eye-estimated, and
 * freeing it here couples into gravity on short windows.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "LinearSeed.h"

#include <algorithm>
#include <cmath>

#include "../init/RelRotProcrustes.h"
#include "utils/quat_ops.h"

using namespace ov_zcalib;

namespace {
/// Window-local gyro-bias refinement: consecutive-clone camera rotations from
/// the tracked bearings (Procrustes) vs the gyro chain, R_ItoC known from the
/// current calibration. Same closed-form dt-column update as the hand-eye.
Eigen::Vector3d bias_presolve(const WindowData &win, const SharedCalib &calib, const Eigen::Vector3d &bg0, int iters) {
  const int n_cams = calib.n_cams();
  std::vector<Eigen::Matrix3d> R_IC((size_t)n_cams);
  for (int c = 0; c < n_cams; ++c)
    R_IC[(size_t)c] = ov_core::quat_2_Rot(calib.cams[(size_t)c].q_ItoC);
  const int N = (int)win.clone_times.size();

  // Per-interval camera rotation logs (id-matched bearings, deterministic seed).
  //
  // Intervals are taken between consecutive clones OF THE SAME CAMERA, not between adjacent clones
  // of the merged timeline. Two cameras with no shared field of view share no feature id, so an
  // adjacent-clone pair that straddles cameras has nothing to match and yields nothing -- on an
  // interleaved rig that is EVERY pair, and the presolve would silently return the seed bias.
  struct Pair {
    Eigen::Vector3d th_C;
    double t0, t1;
    int cam;
  };
  std::vector<Pair> pairs;
  pairs.reserve((size_t)std::max(0, N - 1));
  std::vector<std::vector<int>> clones_of((size_t)n_cams);
  for (int k = 0; k < N; ++k)
    if (!win.obs[(size_t)k].empty()) {
      const int c = win.obs[(size_t)k].front().cam;
      if (c >= 0 && c < n_cams)
        clones_of[(size_t)c].push_back(k);
    }
  for (int c = 0; c < n_cams; ++c) {
    const std::vector<int> &ks = clones_of[(size_t)c];
    for (size_t i = 0; i + 1 < ks.size(); ++i) {
      const int k = ks[i], k2 = ks[i + 1];
      std::vector<Eigen::Vector3d> b0, b1;
      for (const CloneObs &oa : win.obs[(size_t)k])
        for (const CloneObs &ob : win.obs[(size_t)k2])
          if (oa.feat_id == ob.feat_id) {
            b0.push_back(oa.bearing);
            b1.push_back(ob.bearing);
            break;
          }
      RelRotEssential::Options ro;
      const RelRotEssential::Result rr = RelRotEssential::solve(b0, b1, (uint64_t)(win.clone_times[(size_t)k] * 1e6) + k, ro);
      if (!rr.ok)
        continue;
      Pair p;
      p.th_C = ov_core::log_so3(rr.R_C1toC2);
      p.t0 = win.clone_times[(size_t)k];
      p.t1 = win.clone_times[(size_t)k2];
      p.cam = c;
      pairs.push_back(p);
    }
  }
  Eigen::Vector3d bg = bg0;
  if (pairs.size() < 4)
    return bg;
  for (int it = 0; it < iters; ++it) {
    double denom = 0.0;
    Eigen::Vector3d num = Eigen::Vector3d::Zero();
    for (const Pair &p : pairs) {
      Eigen::Vector3d th_I;
      // gyro-only chain over the interval at the current bias (boundary-interp)
      Eigen::Matrix3d DR = Eigen::Matrix3d::Identity();
      bool any = false;
      for (size_t i = 0; i + 1 < win.imu.size(); ++i) {
        const RawImu &s0 = win.imu[i], &s1 = win.imu[i + 1];
        if (s1.timestamp <= p.t0 || s0.timestamp >= p.t1)
          continue;
        const double ta = std::max(s0.timestamp, p.t0), tb = std::min(s1.timestamp, p.t1);
        if (!(tb > ta))
          continue;
        DR = ov_core::exp_so3(-(0.5 * (s0.wm + s1.wm) - bg) * (tb - ta)) * DR;
        any = true;
      }
      if (!any)
        continue;
      th_I = ov_core::log_so3(DR);
      const double dt = p.t1 - p.t0;
      const double w = th_I.norm(); // rotation-rich intervals dominate
      num += w * dt * (R_IC[(size_t)p.cam].transpose() * p.th_C - th_I);
      denom += w * dt * dt;
    }
    if (denom > 1e-12)
      bg += num / denom;
  }
  return bg;
}
} // namespace

bool LinearSeed::seed_window(WindowData &win, const SharedCalib &calib, const Eigen::Vector3d &bg_boot, LinearSeedReport &rep) {
  return seed_window(win, calib, bg_boot, rep, LinearSeedConfig());
}

bool LinearSeed::seed_window(WindowData &win, const SharedCalib &calib, const Eigen::Vector3d &bg_boot, LinearSeedReport &rep,
                             const LinearSeedConfig &cfg) {
  rep = LinearSeedReport();
  const int N = (int)win.clone_times.size();
  const int F = (int)win.num_feats;
  if (N < 3 || F == 0)
    return false;

  Eigen::Vector3d bg_eff = bg_boot;
  if (cfg.bias_presolve) {
    bg_eff = bias_presolve(win, calib, bg_boot, cfg.bias_presolve_iters);
    rep.bg_shift = (bg_eff - bg_boot).norm();
    if (rep.bg_shift > cfg.max_bg_shift)
      bg_eff = bg_boot; // wild visual estimate: keep the boot value
  }

  // ---- gyro/accel preintegration chain at the current calibration ----
  ImuIntrinsicModel model_all = calib.imu;
  model_all.calib_dw = model_all.calib_da = model_all.calib_RAtoI = true;
  model_all.calib_tg = false;
  std::vector<Eigen::Vector4d> q(N, Eigen::Vector4d(0, 0, 0, 1)); // {}^{Ik}_{G}q, G = I0
  std::vector<Eigen::Vector3d> S(N, Eigen::Vector3d::Zero()), V(N, Eigen::Vector3d::Zero());
  std::vector<Eigen::Matrix3d> JS(N, Eigen::Matrix3d::Zero()), JV(N, Eigen::Matrix3d::Zero()); // d(S,V)/d(ba)
  std::vector<double> T(N, 0.0);
  {
    Eigen::Vector3d Sk = Eigen::Vector3d::Zero(), Vk = Eigen::Vector3d::Zero();
    Eigen::Matrix3d JSk = Eigen::Matrix3d::Zero(), JVk = Eigen::Matrix3d::Zero();
    for (int k = 0; k + 1 < N; ++k) {
      AciPreintResult pre;
      if (!AciCalibPreint::integrate(win.imu, win.clone_times[k], win.clone_times[k + 1], model_all, bg_eff, Eigen::Vector3d::Zero(),
                                     calib.noise, pre))
        return false;
      const Eigen::Matrix3d Rk_T = ov_core::quat_2_Rot(q[k]).transpose();
      Sk += Vk * pre.dt + Rk_T * pre.alpha;
      JSk += JVk * pre.dt + Rk_T * pre.H_a; // alpha(ba) = alpha + H_a * ba (ba_lin = 0)
      Vk += Rk_T * pre.beta;
      JVk += Rk_T * pre.H_b;
      q[k + 1] = ov_core::quat_multiply(pre.q_KtoK1, q[k]);
      S[k + 1] = Sk;
      V[k + 1] = Vk;
      JS[k + 1] = JSk;
      JV[k + 1] = JVk;
      T[k + 1] = T[k] + pre.dt;
    }
  }

  // ---- arrowhead normal equations over y = [v0(3), g(3), ba(3)] + {p_f} ----
  // The extrinsic enters PER OBSERVATION, not per window: a clone's observations come from the
  // camera that produced that clone, and the two cameras sit at different places on the rig.
  const int n_cams = calib.n_cams();
  std::vector<Eigen::Matrix3d> R_IC((size_t)n_cams);
  for (int c = 0; c < n_cams; ++c)
    R_IC[(size_t)c] = ov_core::quat_2_Rot(calib.cams[(size_t)c].q_ItoC);
  std::vector<Eigen::Matrix3d> Aff(F, Eigen::Matrix3d::Zero());
  std::vector<Eigen::Matrix<double, 3, 9>> Afy(F, Eigen::Matrix<double, 3, 9>::Zero());
  std::vector<Eigen::Vector3d> bf(F, Eigen::Vector3d::Zero());
  Eigen::Matrix<double, 9, 9> Ayy = Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::Matrix<double, 9, 1> by = Eigen::Matrix<double, 9, 1>::Zero();

  for (int k = 0; k < N; ++k) {
    const Eigen::Matrix3d R_GtoI_k = ov_core::quat_2_Rot(q[k]);
    for (const CloneObs &o : win.obs[k]) {
      const CamCalib &kc = calib.cams[(size_t)o.cam];
      const Eigen::Matrix3d RCk = R_IC[(size_t)o.cam] * R_GtoI_k;
      const Eigen::Matrix3d sb = ov_core::skew_x(o.bearing);
      const Eigen::Matrix3d K = sb * RCk;
      // K p_f - T K v0 + T^2/2 K g - K JS ba = K S - skew(b) p_IinC
      Eigen::Matrix<double, 3, 9> Ky;
      Ky.leftCols<3>() = -T[k] * K;
      Ky.middleCols<3>(3) = 0.5 * T[k] * T[k] * K;
      Ky.rightCols<3>() = -K * JS[k];
      const Eigen::Vector3d c = K * S[k] - sb * kc.p_IinC;
      const int f = (int)o.feat_id;
      Aff[f].noalias() += K.transpose() * K;
      Afy[f].noalias() += K.transpose() * Ky;
      bf[f].noalias() += K.transpose() * c;
      Ayy.noalias() += Ky.transpose() * Ky;
      by.noalias() += Ky.transpose() * c;
    }
  }
  // physical-scale prior on ba (keeps degenerate windows well-posed; zero-mean)
  Ayy.bottomRightCorner<3, 3>().diagonal().array() += 1.0 / (cfg.ba_prior_sigma * cfg.ba_prior_sigma);

  // ---- Schur on y; near-singular tracks deferred to the depth fallback ----
  std::vector<Eigen::Matrix3d> Aff_inv(F);
  std::vector<char> solvable(F, 0);
  Eigen::Matrix<double, 9, 9> Sch = Ayy;
  Eigen::Matrix<double, 9, 1> r = by;
  for (int f = 0; f < F; ++f) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(Aff[f]);
    if (eig.eigenvalues()(0) < cfg.min_block_eig * std::max(Aff[f].trace(), 1e-30))
      continue;
    solvable[f] = 1;
    Aff_inv[f] = eig.eigenvectors() * eig.eigenvalues().cwiseInverse().asDiagonal() * eig.eigenvectors().transpose();
    Sch.noalias() -= Afy[f].transpose() * (Aff_inv[f] * Afy[f]);
    r.noalias() -= Afy[f].transpose() * (Aff_inv[f] * bf[f]);
  }
  Eigen::LLT<Eigen::Matrix<double, 9, 9>> llt(Sch);
  if (llt.info() != Eigen::Success)
    return false;
  Eigen::Matrix<double, 9, 1> y = llt.solve(r);

  // health FIRST on the unconstrained gravity (the |g| deviation is the metric)
  rep.g_mag = y.segment<3>(3).norm();
  if (rep.g_mag < 1e-6 || std::abs(rep.g_mag - calib.grav_mag) > cfg.max_g_mag_dev * calib.grav_mag)
    return false;

  // ---- constrained re-solve on the |g| = grav_mag sphere tangent ----
  // Projecting only the direction after the fact breaks the seed's kinematic
  // self-consistency (the |g| gap lands in the un-robustified IMU factors as a
  // huge whitened residual and strands the micro-BA). Substitute
  // g = g0 + B*delta (g0 = grav_mag * ghat, B = tangent basis) into the SAME
  // normal equations and re-solve [v0, delta(2), ba] exactly.
  {
    const Eigen::Vector3d ghat = y.segment<3>(3) / rep.g_mag;
    const Eigen::Vector3d g0 = calib.grav_mag * ghat;
    // tangent basis via the least-aligned axis (the S2 parameterization rule)
    int amin = 0;
    for (int i = 1; i < 3; ++i)
      if (std::abs(ghat(i)) < std::abs(ghat(amin)))
        amin = i;
    const Eigen::Vector3d b1 = (Eigen::Vector3d::Unit(amin) - ghat * ghat(amin)).normalized();
    const Eigen::Vector3d b2 = ghat.cross(b1);
    Eigen::Matrix<double, 9, 8> M = Eigen::Matrix<double, 9, 8>::Zero();
    M.block<3, 3>(0, 0).setIdentity();
    M.block<3, 1>(3, 3) = b1;
    M.block<3, 1>(3, 4) = b2;
    M.block<3, 3>(6, 5).setIdentity();
    Eigen::Matrix<double, 9, 1> y0 = Eigen::Matrix<double, 9, 1>::Zero();
    y0.segment<3>(3) = g0;

    Eigen::Matrix<double, 8, 8> Sch8 = M.transpose() * Ayy * M;
    Eigen::Matrix<double, 8, 1> r8 = M.transpose() * (by - Ayy * y0);
    for (int f = 0; f < F; ++f) {
      if (!solvable[f])
        continue;
      const Eigen::Matrix<double, 3, 8> AfyM = Afy[f] * M;
      const Eigen::Vector3d bf0 = bf[f] - Afy[f] * y0;
      Sch8.noalias() -= AfyM.transpose() * (Aff_inv[f] * AfyM);
      r8.noalias() -= AfyM.transpose() * (Aff_inv[f] * bf0);
    }
    Eigen::LLT<Eigen::Matrix<double, 8, 8>> llt8(Sch8);
    if (llt8.info() != Eigen::Success)
      return false;
    const Eigen::Matrix<double, 8, 1> y8 = llt8.solve(r8);
    y = y0 + M * y8;
    // renormalize the second-order tangent overshoot exactly onto the sphere
    y.segment<3>(3) = calib.grav_mag * y.segment<3>(3).normalized();
  }
  const Eigen::Vector3d v0 = y.head<3>(), g = y.segment<3>(3), ba = y.tail<3>();

  rep.ba_mag = ba.norm();
  if (rep.ba_mag > cfg.max_ba_mag)
    return false;

  // bias-corrected kinematic terms used everywhere below
  auto S_at = [&](int k) { return Eigen::Vector3d(S[k] + JS[k] * ba); };
  auto V_at = [&](int k) { return Eigen::Vector3d(V[k] + JV[k] * ba); };
  auto p_at = [&](int k) { return Eigen::Vector3d(v0 * T[k] - 0.5 * g * T[k] * T[k] + S_at(k)); };

  // ---- back-substitution + depth stats + fallback ----
  std::vector<Eigen::Vector3d> feats(F, Eigen::Vector3d::Zero());
  std::vector<double> depths;
  depths.reserve(F);
  for (int f = 0; f < F; ++f) {
    if (!solvable[f])
      continue;
    feats[f] = Aff_inv[f] * (bf[f] - Afy[f] * y);
    rep.feats_solved++;
  }
  std::vector<char> depth_ok(F, 0);
  for (int k = 0; k < N; ++k) {
    const Eigen::Matrix3d R_GtoI_k = ov_core::quat_2_Rot(q[k]);
    const Eigen::Vector3d pk = p_at(k);
    for (const CloneObs &o : win.obs[k]) {
      const int f = (int)o.feat_id;
      if (!solvable[f] || depth_ok[f])
        continue;
      const CamCalib &kc = calib.cams[(size_t)o.cam];
      const double z = (R_IC[(size_t)o.cam] * R_GtoI_k * (feats[f] - pk) + kc.p_IinC)(2);
      if (z > 0.1) {
        depth_ok[f] = 1;
        depths.push_back(z);
      }
    }
  }
  rep.median_depth = cfg.fallback_depth;
  if (!depths.empty()) {
    std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());
    rep.median_depth = depths[depths.size() / 2];
  }
  std::vector<char> seeded = solvable;
  for (int k = 0; k < N && rep.feats_solved + rep.feats_fallback < F; ++k) {
    const Eigen::Vector3d pk = p_at(k);
    for (const CloneObs &o : win.obs[k]) {
      const int f = (int)o.feat_id;
      if (seeded[f])
        continue;
      // first observation: place at the median depth along the bearing, through ITS camera
      const CamCalib &kc = calib.cams[(size_t)o.cam];
      feats[f] = ov_core::quat_2_Rot(q[k]).transpose() *
                     (R_IC[(size_t)o.cam].transpose() * (rep.median_depth * o.bearing - kc.p_IinC)) +
                 pk;
      seeded[f] = 1;
      rep.feats_fallback++;
    }
  }
  if (rep.feats_solved == 0 || rep.feats_fallback > cfg.max_fallback_frac * F)
    return false;

  // ---- linear-model residual health (solvable tracks only) ----
  {
    double sum = 0.0;
    int n = 0;
    for (int k = 0; k < N; ++k) {
      const Eigen::Matrix3d R_GtoI_k = ov_core::quat_2_Rot(q[k]);
      const Eigen::Vector3d pk = p_at(k);
      for (const CloneObs &o : win.obs[k]) {
        if (!solvable[o.feat_id])
          continue;
        const CamCalib &kc = calib.cams[(size_t)o.cam];
        const Eigen::Vector3d pc = R_IC[(size_t)o.cam] * R_GtoI_k * (feats[o.feat_id] - pk) + kc.p_IinC;
        if (pc.norm() < 1e-9)
          continue;
        sum += std::acos(std::min(1.0, std::max(-1.0, pc.normalized().dot(o.bearing))));
        ++n;
      }
    }
    if (n == 0)
      return false;
    rep.mean_ang_resid = sum / n;
    // Depth-invariant primary gate; the angular gate remains a far-scene
    // backstop. The metric gate grows with the drift-budget ENVELOPE
    // (gate(T) = strict + eps_eff*g*T^2/4, see LinearSeedConfig): an
    // uncalibrated accel chain must leave that much kinematic drift in the
    // linear solution, and a fixed gate structurally rejects long windows on
    // exactly the factory-fresh rigs the calibrator exists for. Envelope
    // admissions are flagged for the session's post-A0 probation re-check —
    // geometry failures (FOE-degenerate triangulation, blur) blow far past
    // the envelope and stay rejected.
    const double Twin = (N > 0) ? T[N - 1] : 0.0;
    const double gate_m = cfg.max_mean_metric_resid + 0.25 * cfg.drift_budget_ms2 * Twin * Twin;
    if (rep.mean_ang_resid * rep.median_depth > gate_m && rep.mean_ang_resid > cfg.max_mean_ang_resid)
      return false;
    rep.envelope_only = (rep.mean_ang_resid * rep.median_depth > cfg.max_mean_metric_resid);
  }

  // ---- emit seeds (S2 gravity projection; window-frame states) ----
  win.seed_q = q;
  win.seed_v.resize(N);
  win.seed_p.resize(N);
  for (int k = 0; k < N; ++k) {
    win.seed_v[k] = v0 - g * T[k] + V_at(k);
    win.seed_p[k] = p_at(k);
  }
  win.seed_feats = feats;
  win.seed_grav = g; // already |g| = grav_mag from the constrained re-solve
  win.seed_bg = bg_eff;
  win.seed_ba = ba;
  win.has_seeds = true;
  rep.ok = true;
  return true;
}
