/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: rotational hand-eye bootstrap implementation (see HandEyeWahba.h).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "HandEyeWahba.h"

#include <algorithm>
#include <cmath>

#include "utils/quat_ops.h"

using namespace ov_zcalib;

bool HandEyeWahba::theta_imu(const std::vector<RawImu> &imu, double t0, double t1, const Eigen::Vector3d &bg, Eigen::Vector3d &theta) {
  if (!(t1 > t0) || imu.size() < 2)
    return false;
  // Boundary-interpolated samples spanning [t0, t1] (the AciCalibPreint stepping)
  std::vector<RawImu> data;
  for (size_t i = 0; i + 1 < imu.size(); ++i) {
    const RawImu &s0 = imu[i], &s1 = imu[i + 1];
    if (s1.timestamp <= t0 || s0.timestamp >= t1)
      continue;
    auto interp = [&](double t) {
      const double lam = (t - s0.timestamp) / (s1.timestamp - s0.timestamp);
      RawImu r;
      r.timestamp = t;
      r.wm = (1 - lam) * s0.wm + lam * s1.wm;
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
  Eigen::Matrix3d DR = Eigen::Matrix3d::Identity();
  for (size_t i = 0; i + 1 < data.size(); ++i) {
    const double dt = data[i + 1].timestamp - data[i].timestamp;
    if (!(dt > 0))
      continue;
    const Eigen::Vector3d w_hat = 0.5 * (data[i].wm + data[i + 1].wm) - bg;
    DR = ov_core::exp_so3(-w_hat * dt) * DR;
  }
  theta = ov_core::log_so3(DR);
  return true;
}

namespace {

struct PairEval {
  Eigen::Vector3d th_I, th_C;
  double dt = 0.0, w = 0.0;
  bool usable = false;
};

/// Wahba/Markley on the usable set: theta_C = R * theta_I. Returns diversity (0 on failure).
double wahba(const std::vector<PairEval> &pe, Eigen::Matrix3d &R) {
  Eigen::Matrix3d B = Eigen::Matrix3d::Zero();
  for (const auto &p : pe) {
    if (!p.usable)
      continue;
    const double ni = p.th_I.norm(), nc = p.th_C.norm();
    if (ni < 1e-9 || nc < 1e-9)
      continue;
    B += p.w * (p.th_C / nc) * (p.th_I / ni).transpose();
  }
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
  if (svd.singularValues()(0) < 1e-9)
    return 0.0;
  Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
  D(2, 2) = ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0) ? -1.0 : 1.0;
  R = svd.matrixU() * D * svd.matrixV().transpose();
  return svd.singularValues()(2) / svd.singularValues()(0);
}

struct AltOut {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  double rmse = -1.0, diversity = 0.0;
  int used = 0;
};

/// One full alternating solve at a fixed td. Returns rmse<0 on failure.
/// With `mask` given, pair admission is FROZEN to it (gates off): the fine td
/// sweep must compare RMSE over the SAME pair set, or the ingest gates re-select
/// different (luckier) subsets per candidate and the sweep minimizes selection
/// bias instead of time offset.
AltOut alternate(const std::vector<RawImu> &imu, const std::vector<HandEyePair> &pairs, double td, const Eigen::Vector3d &bg0,
                 const HandEyeConfig &cfg, std::vector<PairEval> *pe_out = nullptr, const std::vector<char> *mask = nullptr) {
  AltOut out;
  out.bg = bg0;
  std::vector<PairEval> pe(pairs.size());
  for (int alt = 0; alt < cfg.bias_alternations; ++alt) {
    // (re)preintegrate + gates at the current bias
    out.used = 0;
    for (size_t k = 0; k < pairs.size(); ++k) {
      PairEval &p = pe[k];
      p.usable = false;
      p.dt = pairs[k].t1 - pairs[k].t0;
      if (!(p.dt > 1e-6))
        continue;
      if (mask && !(*mask)[k])
        continue;
      if (!HandEyeWahba::theta_imu(imu, pairs[k].t0 + td, pairs[k].t1 + td, out.bg, p.th_I))
        continue;
      p.th_C = pairs[k].theta_C;
      if (!mask) {
        const double rate = p.th_I.norm() / p.dt;
        if (rate < cfg.min_pair_rate || rate > cfg.max_pair_rate)
          continue;
        const double ratio = p.th_C.norm() / std::max(p.th_I.norm(), 1e-12);
        if (ratio < cfg.ratio_min || ratio > cfg.ratio_max)
          continue;
      }
      p.w = pairs[k].weight * p.th_I.norm(); // rotation-rich pairs dominate the fit
      p.usable = true;
      ++out.used;
    }
    if (out.used < cfg.min_pairs)
      return out; // rmse stays -1
    out.diversity = wahba(pe, out.R);
    if (out.diversity < cfg.min_axis_diversity)
      return out;
    if (!cfg.estimate_bg)
      continue; // bg frozen at the (still-baseline) seed
    // closed-form bias step: theta_I(bg+db) ~= theta_I + dt*db  =>
    // min sum w || (th_C - R th_I) - R dt db ||^2, isotropic normal equations
    double denom = 0.0;
    Eigen::Vector3d num = Eigen::Vector3d::Zero();
    for (const auto &p : pe) {
      if (!p.usable)
        continue;
      num += p.w * p.dt * (out.R.transpose() * p.th_C - p.th_I);
      denom += p.w * p.dt * p.dt;
    }
    if (denom > 1e-12)
      out.bg += num / denom;
  }
  double sse = 0.0;
  int n = 0;
  for (const auto &p : pe) {
    if (!p.usable)
      continue;
    sse += (p.th_C - out.R * p.th_I).squaredNorm();
    ++n;
  }
  if (n == 0)
    return out;
  out.rmse = std::sqrt(sse / n);
  if (pe_out)
    *pe_out = pe;
  return out;
}

} // namespace

bool HandEyeWahba::solve(const std::vector<RawImu> &imu, const std::vector<HandEyePair> &pairs, double td_seed,
                         const Eigen::Vector3d &bg_seed, const HandEyeConfig &cfg, HandEyeResult &out) {
  out = HandEyeResult();

  // (a) gated alternating solve at the seed td
  std::vector<PairEval> pe;
  HandEyeConfig cfg_eff = cfg;
  AltOut base = alternate(imu, pairs, td_seed, bg_seed, cfg_eff, &pe);
  if (base.rmse >= 0.0 && cfg_eff.estimate_bg && (base.bg - bg_seed).norm() > cfg_eff.max_bg_sane) {
    // translation-bias soak: redo everything with the bias frozen at the seed
    cfg_eff.estimate_bg = false;
    base = alternate(imu, pairs, td_seed, bg_seed, cfg_eff, &pe);
  }
  if (base.rmse < 0.0)
    return false;

  // (b) Hampel trim at the seed td -> FROZEN pair mask for everything after
  std::vector<char> mask(pairs.size(), 0);
  for (size_t k = 0; k < pairs.size(); ++k)
    mask[k] = pe[k].usable;
  {
    std::vector<double> res;
    res.reserve(pe.size());
    for (const auto &p : pe)
      if (p.usable)
        res.push_back((p.th_C - base.R * p.th_I).norm());
    std::vector<double> sorted = res;
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    const double med = (n % 2) ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    std::vector<double> dev = res;
    for (double &d : dev)
      d = std::abs(d - med);
    std::sort(dev.begin(), dev.end());
    const double mad = (n % 2) ? dev[n / 2] : 0.5 * (dev[n / 2 - 1] + dev[n / 2]);
    const double gate = med + 3.0 * 1.4826 * mad;
    if (mad > 1e-9) {
      size_t ri = 0;
      int trimmed = 0;
      for (size_t k = 0; k < pairs.size(); ++k) {
        if (!pe[k].usable)
          continue;
        if (res[ri++] > gate) {
          mask[k] = 0;
          ++trimmed;
        }
      }
      int kept = 0;
      for (char m : mask)
        kept += m;
      if (trimmed > 0 && kept >= cfg.min_pairs) {
        const AltOut refit = alternate(imu, pairs, td_seed, base.bg, cfg_eff, nullptr, &mask);
        if (refit.rmse >= 0.0 && refit.rmse < base.rmse) {
          base = refit;
          out.pairs_trimmed = trimmed;
        } else {
          for (size_t k = 0; k < pairs.size(); ++k)
            mask[k] = pe[k].usable; // trim did not help; restore
        }
      }
    }
  }

  // (c) fine td sweep on the frozen set (exact re-preintegration per candidate)
  double best_td = td_seed;
  AltOut best = base;
  const int n_steps = std::max(0, (int)std::round(cfg.td_fine_range / std::max(cfg.td_fine_step, 1e-9)));
  for (int s = -n_steps; s <= n_steps; ++s) {
    if (s == 0)
      continue;
    const double td = td_seed + s * cfg.td_fine_step;
    const AltOut cand = alternate(imu, pairs, td, base.bg, cfg_eff, nullptr, &mask);
    if (cand.rmse >= 0.0 && cand.rmse < best.rmse) {
      best = cand;
      best_td = td;
    }
  }
  out.td_at_bound = (n_steps > 0) && (std::abs(std::abs(best_td - td_seed) - n_steps * cfg.td_fine_step) < 0.5 * cfg.td_fine_step);

  out.ok = true;
  out.q_ItoC = ov_core::rot_2_quat(best.R);
  out.bg = best.bg;
  out.td = best_td;
  out.rmse_rad = best.rmse;
  out.axis_diversity = best.diversity;
  out.pairs_used = best.used;
  return true;
}
