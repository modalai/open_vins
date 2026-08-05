/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: session orchestrator implementation (see CalibSessionRunner.h).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "CalibSessionRunner.h"

#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>

#include "../utils/YamlWriteback.h"
#include "ceres_free/Parallel.h"
#include "../window/CamUndistort.h"
#include "CalibSession.h"
#include "utils/quat_ops.h"

using namespace ov_zcalib;

CalibSessionRunner::CalibSessionRunner(const SessionConfig &cfg, const SessionSeed &seed) : cfg_(cfg), seed0_(seed) {
  calib_ = seed.calib;
  for (CamCalib &k : calib_.cams) {
    k.cam_mode = 0; // camera intrinsics enter only in the staged phase B
    // tr is NEVER estimated: it is the HAL3 hardware readout carried in the seed, a fixed
    // transport constant of every reprojection. A global-shutter camera has NO readout time,
    // so its tr is forced to zero no matter what the seed claims.
    if (!k.rolling)
      k.tr = 0.0;
  }
  // Tg (g-sensitivity): SESSION-level column switch + per-stage free flag, set together HERE and
  // then only ever narrowed. The flag stays frozen until the A1b full-chain gate certifies the
  // excitation (split-half / Wald) -- exactly the accel chain's doctrine -- but the preint COLUMN
  // WIDTH must be fixed for the whole session (persistent window graphs cannot change a factor's
  // parameter list in place), which is what tg_enabled pins. A frozen factory chain freezes tg
  // with it: earning a gyro cross-coupling while refusing to touch dw/da is not a coherent trust
  // model, and A1b (tg's only unlock) is skipped entirely on frozen chains anyway.
  calib_.imu.calib_tg = cfg_.free_tg && (calib_.imu.calib_dw || calib_.imu.calib_da || calib_.imu.calib_RAtoI);
  calib_.tg_enabled = calib_.imu.calib_tg;
  n_cams_ = calib_.n_cams();
  prev_frame_.assign((size_t)n_cams_, FrameObs());
  have_prev_frame_.assign((size_t)n_cams_, 0);
  pairs_.assign((size_t)n_cams_, {});
  rates_.assign((size_t)n_cams_, {});
  boot_thin_.assign((size_t)n_cams_, 0);
  boot_relrot_.assign((size_t)n_cams_, 0);
  boot_eiv_.assign((size_t)n_cams_, 0);
  boot_pairs_ok_.assign((size_t)n_cams_, 0);
  exp_sum_.assign((size_t)n_cams_, 0.0);
  exp_n_.assign((size_t)n_cams_, 0);
  rep_ = SessionReport();
  rep_.committed = calib_;
}

void CalibSessionRunner::enter_(RunnerState s, const char *why) {
  if (cfg_.verbose)
    std::printf("[session] -> %s (%s)\n",
                s == RunnerState::SETTLE          ? "SETTLE"
                : s == RunnerState::BOOTSTRAP     ? "BOOTSTRAP"
                : s == RunnerState::COLLECT       ? "COLLECT"
                : s == RunnerState::THERMAL_HOLD  ? "THERMAL_HOLD"
                : s == RunnerState::SOLVE_REFINE  ? "SOLVE_REFINE"
                : s == RunnerState::VERIFY        ? "VERIFY"
                : s == RunnerState::COMMIT        ? "COMMIT"
                : s == RunnerState::DONE          ? "DONE"
                                                  : "ABORT",
                why);
  state_ = s;
  if (s == RunnerState::COLLECT)
    past_bootstrap_.store(true, std::memory_order_relaxed); // ingest threads key decimation on this
  if (s == RunnerState::ABORT)
    rep_.abort_reason = why;
}

double CalibSessionRunner::temp_slope_() const {
  if (roll_.size() < 16)
    return 0.0;
  const RawImu &a = roll_.front(), &b = roll_.back();
  const double dt = b.timestamp - a.timestamp;
  return (dt > 1.0) ? (b.temp_c - a.temp_c) / dt : 0.0;
}

void CalibSessionRunner::feed_imu(const RawImu &s) {
  if (first_t_ < 0.0)
    first_t_ = s.timestamp;
  // rolling stats ring (~10 s: settle stillness + temp slope + thermal hold)
  roll_.push_back(s);
  rw_ += s.wm;
  rw2_ += s.wm.cwiseProduct(s.wm);
  while (!roll_.empty() && roll_.front().timestamp < s.timestamp - 10.0) {
    rw_ -= roll_.front().wm;
    rw2_ -= roll_.front().wm.cwiseProduct(roll_.front().wm);
    roll_.pop_front();
  }

  switch (state_) {
  case RunnerState::SETTLE: {
    // stillness over the last second of the ring
    Eigen::Vector3d sw = Eigen::Vector3d::Zero(), sw2 = Eigen::Vector3d::Zero();
    int n = 0;
    for (auto it = roll_.rbegin(); it != roll_.rend() && it->timestamp > s.timestamp - 1.0; ++it) {
      sw += it->wm;
      sw2 += it->wm.cwiseProduct(it->wm);
      ++n;
    }
    if (n >= 16) {
      const Eigen::Vector3d var = (sw2 / n - (sw / n).cwiseProduct(sw / n)).cwiseMax(0.0);
      const bool still = std::sqrt(var.sum()) < cfg_.settle_still_w;
      if (still) {
        if (still_since_ < 0.0) {
          still_since_ = s.timestamp;
          still_gyro_sum_.setZero();
          still_gyro_n_ = 0;
        }
        still_gyro_sum_ += s.wm;
        ++still_gyro_n_;
        if (s.timestamp - still_since_ >= cfg_.settle_min_still_s && std::abs(temp_slope_()) <= cfg_.settle_max_temp_slope) {
          bg0_ = still_gyro_sum_ / std::max(still_gyro_n_, 1);
          have_baseline_ = true;
          boot_t0_ = s.timestamp;
          enter_(RunnerState::BOOTSTRAP, "still baseline captured (bias + thermal)");
        }
      } else {
        still_since_ = -1.0;
      }
    }
    if (!have_baseline_ && s.timestamp - first_t_ > cfg_.settle_timeout_s) {
      bg0_.setZero();
      boot_t0_ = s.timestamp;
      enter_(RunnerState::BOOTSTRAP, "settle timeout: proceeding with bg=0 (hand-eye estimates it)");
    }
    break;
  }
  case RunnerState::BOOTSTRAP:
    boot_imu_.push_back(s);
    break;
  case RunnerState::COLLECT:
  case RunnerState::THERMAL_HOLD:
    harvester_->push_imu(s);
    break;
  default:
    break;
  }
}

void CalibSessionRunner::feed_frame(const FrameObs &f) {
  const size_t c = (size_t)f.cam;
  if (c >= (size_t)n_cams_)
    return; // a camera the session was never seeded for
  if (f.exposure_s > 0.f) {
    // Mean exposure is PER CAMERA and DIAGNOSTIC ONLY: stamps arrive already centered at
    // ingest (see SessionReport::mean_exposure_s), and each camera has its own AE history.
    exp_sum_[c] += (double)f.exposure_s;
    ++exp_n_[c];
  }
  switch (state_) {
  case RunnerState::SETTLE:
    prev_frame_[c] = f; // keep continuity so BOOTSTRAP starts pairing immediately
    have_prev_frame_[c] = 1;
    break;
  case RunnerState::BOOTSTRAP: {
    // Retroactive harvest buffer: these frames replay into the harvester the
    // moment bootstrap succeeds (see try_bootstrap_) -- otherwise the bootstrap
    // span is dead collection time (~half of a short flight log). RING at the
    // cap: the retro value lives in the FRESHEST span; keeping the oldest would
    // hand a long bootstrap its stalest minute and drop the motion that finally
    // passed the gate.
    if (cfg_.retro_harvest) {
      if ((int)boot_frames_.size() >= 4000)
        boot_frames_.pop_front();
      boot_frames_.push_back(f);
    }
    // Pairs are formed WITHIN a camera. Two cameras with no shared field of view share no feature
    // id, so a cross-camera pair would match nothing; and each camera has its own R_ItoC and its
    // own td, so each accumulates its own hand-eye and its own cross-correlation.
    if (have_prev_frame_[c]) {
      const FrameObs &prev = prev_frame_[c];
      const CamCalib &kc = seed0_.calib.cams[c];
      std::vector<Eigen::Vector3d> b0, b1;
      b0.reserve(f.pts.size());
      b1.reserve(f.pts.size());
      for (const FrameObsPoint &p : f.pts) {
        for (const FrameObsPoint &q : prev.pts) {
          if (q.id != p.id)
            continue;
          b0.push_back(CamUndistort::bearing(Eigen::Vector2d(q.u, q.v), kc.cam, kc.fisheye));
          b1.push_back(CamUndistort::bearing(Eigen::Vector2d(p.u, p.v), kc.cam, kc.fisheye));
          break;
        }
      }
      if ((int)b0.size() < cfg_.min_pair_matches) {
        ++boot_thin_[c];
      } else {
        RelRotEssential::Options ro;
        ro.min_pairs = cfg_.min_pair_matches;
        const RelRotEssential::Result rr =
            RelRotEssential::solve(b0, b1, (uint64_t)f.seq * 1000003ull + 17ull + 101ull * (uint64_t)c, ro);
        if (!rr.ok) {
          ++boot_relrot_[c];
        } else {
          const double tm0 = prev.timestamp + 0.5 * (double)prev.exposure_s;
          const double tm1 = f.timestamp + 0.5 * (double)f.exposure_s;
          HandEyePair hp;
          hp.t0 = tm0;
          hp.t1 = tm1;
          hp.theta_C = ov_core::log_so3(rr.R_C1toC2);
          // errors-in-variables gate/weight: the post-derotation residual is the
          // translation-flow scale; where it rivals |theta| the rotation-only
          // fit is biased, not noisy -- downweight, and drop the worst outright
          const double th = hp.theta_C.norm();
          const double eiv = rr.mean_resid_rad / std::max(th, 1e-9);
          if (eiv >= 0.30) {
            ++boot_eiv_[c];
          } else {
            hp.weight = rr.inlier_ratio * std::max(0.05, 1.0 - 4.0 * eiv);
            pairs_[c].push_back(hp);
            ++boot_pairs_ok_[c];
            CamRateSample cr;
            cr.t_mid = 0.5 * (tm0 + tm1);
            cr.rate = th / std::max(tm1 - tm0, 1e-6);
            cr.weight = hp.weight; // the xcorr votes with the SAME trust the hand-eye pairs carry
            rates_[c].push_back(cr);
          }
        }
      }
    }
    prev_frame_[c] = f;
    have_prev_frame_[c] = 1;
    try_bootstrap_(f.timestamp);
    break;
  }
  case RunnerState::COLLECT: {
    if (std::abs(temp_slope_()) > cfg_.thermal_hold_slope) {
      enter_(RunnerState::THERMAL_HOLD, "temperature ramping; pausing window collection");
      break;
    }
    if (f.timestamp - boot_t0_ > cfg_.collect_max_s)
      break; // stop opening; finish() runs the solve
    if (harvester_->push_frame(f)) {
      WindowData w;
      WindowMeta m;
      if (harvester_->pop_window(w, m))
        handle_window_(std::move(w), m);
    }
    break;
  }
  case RunnerState::THERMAL_HOLD:
    if (std::abs(temp_slope_()) <= 0.7 * cfg_.thermal_hold_slope)
      enter_(RunnerState::COLLECT, "temperature settled");
    break;
  default:
    break;
  }
}

void CalibSessionRunner::try_bootstrap_(double now) {
  // RECENCY: the gate judges the operator's LAST bootstrap_window_s of motion, not the whole
  // history. An early bad stretch (AE settling, blur, the pick-up) must age out instead of
  // capping the achievable peak forever -- and a long bootstrap must not grow its per-attempt
  // cost without bound. Sessions that pass inside the horizon never evict anything, so their
  // behaviour is byte-identical to the unwindowed path.
  if (cfg_.bootstrap_window_s > 0.0 && boot_t0_ >= 0.0) {
    const double t_lo = now - cfg_.bootstrap_window_s;
    for (int c = 0; c < n_cams_; ++c) {
      auto &pr = pairs_[(size_t)c];
      size_t np = 0;
      while (np < pr.size() && pr[np].t1 < t_lo)
        ++np;
      if (np)
        pr.erase(pr.begin(), pr.begin() + (long)np);
      auto &rt = rates_[(size_t)c];
      size_t nr = 0;
      while (nr < rt.size() && rt[nr].t_mid < t_lo)
        ++nr;
      if (nr)
        rt.erase(rt.begin(), rt.begin() + (long)nr);
    }
    // The IMU buffer keeps preintegration + td-search slack ahead of the oldest surviving pair;
    // the retro-harvest frames replay AGAINST boot_imu_, so they share its horizon.
    const double t_imu_lo = t_lo - cfg_.td_search_s - cfg_.handeye.td_fine_range - 0.25;
    size_t ni = 0;
    while (ni < boot_imu_.size() && boot_imu_[ni].timestamp < t_imu_lo)
      ++ni;
    if (ni)
      boot_imu_.erase(boot_imu_.begin(), boot_imu_.begin() + (long)ni);
    while (!boot_frames_.empty() && boot_frames_.front().timestamp < t_imu_lo)
      boot_frames_.pop_front();
  }

  // A camera is IN this session only if it is actually delivering frames -- a configured but dead
  // stream must not hold the whole rig in BOOTSTRAP forever.
  std::vector<int> active;
  for (int c = 0; c < n_cams_; ++c)
    if (!pairs_[(size_t)c].empty())
      active.push_back(c);
  int min_pairs = active.empty() ? 0 : (int)pairs_[(size_t)active.front()].size();
  for (int c : active)
    min_pairs = std::min(min_pairs, (int)pairs_[(size_t)c].size());

  // EVERY active camera must clear the pair floor: each one has its own R_ItoC and its own td, and
  // neither is recoverable from another camera's rotations.
  if (boot_t0_ < 0.0 || now - boot_t0_ < cfg_.bootstrap_min_span_s || active.empty() || min_pairs < cfg_.bootstrap_min_pairs) {
    if (boot_t0_ >= 0.0 && now - boot_t0_ > cfg_.bootstrap_timeout_s)
      enter_(RunnerState::ABORT, "bootstrap timeout: not enough rotation-diverse pairs (rotate about multiple axes)");
    return;
  }
  if (now - last_boot_try_ < 2.0)
    return; // xcorr + hand-eye per attempt; retry at 0.5 Hz, data timestamps only
  last_boot_try_ = now;

  // The gyro bias belongs to the IMU, not to a camera: estimate it ONCE, on the best-conditioned
  // camera (the one with the most pairs), and hold every other camera's hand-eye at that value.
  // Letting each camera refit its own bg would let them disagree about a quantity there is only
  // one of, and the disagreement would land in the extrinsics.
  int primary = active.front();
  for (int c : active)
    if (pairs_[(size_t)c].size() > pairs_[(size_t)primary].size())
      primary = c;

  std::vector<TimeOffsetResult> xr_all((size_t)n_cams_);
  std::vector<HandEyeResult> he_all((size_t)n_cams_);
  std::vector<char> xr_cert((size_t)n_cams_, 0); ///< accepted via the robust certificate (not the raw floor)
  Eigen::Vector3d bg_shared = bg0_;
  for (size_t pass = 0; pass < 2; ++pass) {
    for (int c : active) {
      const bool is_primary = (c == primary);
      if ((pass == 0) != is_primary)
        continue; // primary first (it fixes bg), then the rest at that bg

      // The cross-correlation is on rotation-RATE magnitude, which is invariant to R_ItoC -- so
      // each camera recovers its own td without needing its extrinsic first.
      TimeOffsetResult xr = TimeOffsetInit::solve(rates_[(size_t)c], boot_imu_, cfg_.td_search_s, 0.002);
      const bool sharp_ok = xr.peak_sharpness >= cfg_.xcorr_min_sharpness;
      const bool fast_ok = xr.ok && !xr.at_bound && xr.peak_corr >= cfg_.xcorr_min_peak && sharp_ok;
      // ROBUST CERTIFICATE -- a moderate peak whose LOCATION is reproducible identifies td exactly
      // as well as a tall one: the seed contract is only +/- td_fine_range (the hand-eye fine
      // sweep re-solves td by re-preintegration). Broadband pair noise (blur-slipped KLT tracks,
      // RS shear at rate reversals) depresses rho without moving the argmax; the FLAT ridge the
      // floor exists to reject cannot pass this -- its split halves land on different lags and its
      // trim buys nothing. Conditions: one Hampel trim clears the SAME floor while keeping >= 70%
      // of the evidence weight without moving the argmax, and both split halves independently
      // reproduce the lag (each showing a real ridge of its own). Measured need: a real session
      // pinned at peak 0.51-0.57 with a stable argmax for minutes -- a td a single-number floor
      // can never accept, although the fine sweep resolves it cleanly.
      const bool cert_ok = xr.ok && !xr.at_bound && sharp_ok && !fast_ok && xr.trim_consistent &&
                           xr.peak_trimmed >= cfg_.xcorr_min_peak && xr.trim_retention >= 0.70 &&
                           xr.td_split_delta >= 0.0 && xr.td_split_delta <= cfg_.handeye.td_fine_range &&
                           xr.split_min_peak >= 0.6 * cfg_.xcorr_min_peak;
      if (!fast_ok && !cert_ok) {
        if (cfg_.verbose) {
          char sd[32];
          if (xr.td_split_delta >= 0.0)
            std::snprintf(sd, sizeof(sd), "%.1f", 1e3 * xr.td_split_delta);
          else
            std::snprintf(sd, sizeof(sd), "n/a");
          std::printf("[session] bootstrap attempt rejected (cam %d): xcorr ok=%d at_bound=%d peak=%.2f (floor %.2f) "
                      "sharpness=%.2e (floor %.2e) | td=%+.1f ms trim=%.2f keep=%.0f%%%s split|dtd|=%s ms halves>=%.2f | "
                      "pairs=%d yield[thin=%ld relrot=%ld eiv=%ld ok=%ld]\n",
                      c, (int)xr.ok, (int)xr.at_bound, xr.peak_corr, cfg_.xcorr_min_peak, xr.peak_sharpness,
                      cfg_.xcorr_min_sharpness, 1e3 * xr.td, xr.peak_trimmed, 1e2 * xr.trim_retention,
                      xr.trim_consistent ? "" : " (argmax moved)", sd, xr.split_min_peak, (int)rates_[(size_t)c].size(),
                      boot_thin_[(size_t)c], boot_relrot_[(size_t)c], boot_eiv_[(size_t)c], boot_pairs_ok_[(size_t)c]);
        }
        if (now - boot_t0_ > cfg_.bootstrap_timeout_s)
          enter_(RunnerState::ABORT, (xr.ok && !xr.at_bound)
                                         ? "bootstrap timeout: xcorr peak weak/flat (td unidentifiable — vary the rotation rate)"
                                         : "bootstrap timeout: time-offset xcorr failed");
        return;
      }
      xr_cert[(size_t)c] = cert_ok ? 1 : 0;
      if (cert_ok && cfg_.verbose)
        std::printf("[session] bootstrap cam %d: xcorr accepted via ROBUST CERTIFICATE — peak %.2f is under the %.2f floor, "
                    "but the trimmed series clears it (%.2f, keeping %.0f%%) and the split halves agree on the lag "
                    "(|dtd| %.1f ms, weaker half %.2f): the ridge is real, just noisy. td seed %+.1f ms -> fine sweep.\n",
                    c, xr.peak_corr, cfg_.xcorr_min_peak, xr.peak_trimmed, 1e2 * xr.trim_retention, 1e3 * xr.td_split_delta,
                    xr.split_min_peak, 1e3 * xr.td);

      HandEyeConfig hec = cfg_.handeye;
      hec.estimate_bg = is_primary && !have_baseline_; // SETTLE still baseline beats visual-pair bg
      HandEyeResult he;
      if (!HandEyeWahba::solve(boot_imu_, pairs_[(size_t)c], xr.td, bg_shared, hec, he)) {
        if (now - boot_t0_ > cfg_.bootstrap_timeout_s)
          enter_(RunnerState::ABORT, "bootstrap timeout: hand-eye rejected (axis diversity)");
        return;
      }
      if (he.td_at_bound) {
        // fine-td refinement pinned at the sweep edge: the refined td is not a
        // maximum, keep accumulating pairs instead of seeding downstream with it
        if (now - boot_t0_ > cfg_.bootstrap_timeout_s)
          enter_(RunnerState::ABORT, "bootstrap timeout: hand-eye td refinement pinned at sweep bound");
        return;
      }
      xr_all[(size_t)c] = xr;
      he_all[(size_t)c] = he;
      if (is_primary)
        bg_shared = he.bg;
    }
  }

  rep_.xcorr = xr_all;
  rep_.handeye = he_all;
  for (int c : active) {
    calib_.cams[(size_t)c].q_ItoC = he_all[(size_t)c].q_ItoC;
    calib_.cams[(size_t)c].td = he_all[(size_t)c].td;
  }
  bg0_ = bg_shared;
  // No still baseline: per-window visual bias pre-solve carries the gyro bias,
  // the window bias prior widens to visual confidence, and the seed-admission
  // gate scales with the bootstrap's own residual level (the extrinsic seed
  // error is what the joint solve exists to remove -- early windows must pass).
  if (!have_baseline_) {
    cfg_.seed.bias_presolve = true;
    calib_.bg_prior_sigma = 0.03;
  }
  // One seed-admission gate for the rig, scaled by the WORST camera's bootstrap residual: a window
  // is admitted or not as a whole, so it must tolerate the least confident extrinsic seed in it.
  double worst_rmse = 0.0;
  for (int c : active)
    worst_rmse = std::max(worst_rmse, he_all[(size_t)c].rmse_rad);
  cfg_.seed.max_mean_ang_resid = std::max(cfg_.seed.max_mean_ang_resid, 5.0 * worst_rmse);
  if (cfg_.verbose) {
    std::printf("[session] seed gates: metric %.2f m (ang backstop %.1f mrad), bias presolve %s\n", cfg_.seed.max_mean_metric_resid,
                1e3 * cfg_.seed.max_mean_ang_resid, cfg_.seed.bias_presolve ? "ON" : "off");
    for (int c : active) {
      const HandEyeResult &he = he_all[(size_t)c];
      std::printf("[session] bootstrap cam %d%s: R_ItoC ok (rmse %.2e rad, diversity %.2f), td=%.3f ms, pairs=%d(-%d), "
                  "xcorr %.2f%s\n",
                  c, (c == primary) ? " [bg source]" : "", he.rmse_rad, he.axis_diversity, 1e3 * he.td, he.pairs_used,
                  he.pairs_trimmed, xr_all[(size_t)c].peak_corr, xr_cert[(size_t)c] ? " (robust certificate)" : "");
    }
    std::printf("[session] bootstrap: |bg|=%.4f (shared -- one IMU)\n", bg0_.norm());
  }
  // harvester snapshot = post-bootstrap calibration (bearings, td_ref)
  harvester_.reset(new WindowHarvester(cfg_.harvester, calib_));
  scorer_.reset(new WindowScorer(cfg_.scorer));
  slots_.resize(cfg_.scorer.capacity);
  slot_rep_.resize(cfg_.scorer.capacity);
  probation_.assign(cfg_.scorer.capacity, 0);
  // display fusion (prior vec mirrors the joint layout)
  {
    auto layout = calib_.free_blocks();
    const int np = calib_.local_dim();
    Eigen::VectorXd prior(np);
    std::vector<std::string> labels;
    int off = 0;
    for (auto &b : layout) {
      const double sg = cfg_.joint.prior_sigma.count(b.name) ? cfg_.joint.prior_sigma.at(b.name) : 1.0;
      for (int k = 0; k < b.lsize; ++k) {
        labels.push_back(b.name + "[" + std::to_string(k) + "]");
        prior(off + k) = sg;
      }
      off += b.lsize;
    }
    display_.reset(new CalibSession(np, prior, labels));
  }
  // Retroactive harvest: merge-replay the bootstrap-span streams into the
  // fresh harvester in timestamp order (IMU first on ties -- one fixed rule so
  // live == replay bit-identical). Windows close through the identical
  // handle_window_ path as live collection.
  if (cfg_.retro_harvest) {
    const size_t nf = boot_frames_.size(), ni = boot_imu_.size();
    size_t fi = 0, ii = 0;
    int replayed = 0, decimated = 0;
    // The live ingest feeds the tracker at the FULL sensor rate during bootstrap (the pair engine
    // needs it: halving the rate doubles per-pair KLT displacement) and only decimates to the
    // declared fps once COLLECT starts -- so the retro span decimates HERE, or the harvester's
    // drop accounting (nominal dt = 1/fps of the seed) would read a 2x-rate stream. Same accept
    // rule as the live gate (0.75*min_dt) so a jittered frame cannot cascade into rate-halving.
    // Streams already at the declared rate (old records, undecimated cams) pass untouched.
    std::vector<double> retro_last_fed((size_t)n_cams_, -1.0);
    while (fi < nf || ii < ni) {
      const bool take_imu = (ii < ni) && (fi >= nf || boot_imu_[ii].timestamp <= boot_frames_[fi].timestamp);
      if (take_imu) {
        harvester_->push_imu(boot_imu_[ii++]);
      } else {
        const FrameObs &bf = boot_frames_[fi++];
        const size_t bc = (size_t)bf.cam;
        const double fps = (bc < calib_.cams.size()) ? calib_.cams[bc].fps : 0.0;
        if (fps > 0.0 && retro_last_fed[bc] > 0.0 && bf.timestamp - retro_last_fed[bc] < 0.75 / fps) {
          ++decimated;
          continue;
        }
        retro_last_fed[bc] = bf.timestamp;
        if (harvester_->push_frame(bf)) {
          WindowData w;
          WindowMeta m;
          if (harvester_->pop_window(w, m)) {
            handle_window_(std::move(w), m);
            ++replayed;
          }
        }
      }
    }
    if (cfg_.verbose && nf > 0)
      std::printf("[session] retroactive harvest: %zu bootstrap frames / %zu imu replayed (%d decimated to the declared "
                  "rate), %d windows closed\n",
                  nf, ni, decimated, replayed);
    boot_frames_.clear();
    boot_frames_.shrink_to_fit();
  }
  enter_(RunnerState::COLLECT, "bootstrap complete; begin guided collection");
}

void CalibSessionRunner::handle_window_(WindowData &&w, const WindowMeta &m) {
  rep_.windows_harvested++;
  // pre-seed admission gates (fingerprint(9) = median first-to-last track
  // parallax): translation-free / far-field windows starve the seeder and the
  // export, and their failures otherwise surface only as silent drops
  if (cfg_.min_window_parallax > 0.0 && m.fingerprint(9) < cfg_.min_window_parallax) {
    rep_.windows_rejected_gate++;
    if (cfg_.verbose)
      std::printf("[session] window %d REJECTED by parallax floor: %.1f px < %.1f px\n", rep_.windows_harvested, m.fingerprint(9),
                  cfg_.min_window_parallax);
    return;
  }
  LinearSeedReport sr;
  const auto t_seed0 = std::chrono::steady_clock::now();
  const bool seeded_ok = LinearSeed::seed_window(w, calib_, bg0_, sr, cfg_.seed);
  adm_ev_.seed_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_seed0).count();
  if (!seeded_ok) {
    rep_.windows_rejected_seed++;
    if (cfg_.verbose) {
      // rejection diagnosis: angular rate + flow scale separate seed-model
      // failure (calibration-recoverable drift) from geometry failure
      // (FOE-degenerate push, exposure*flow motion blur)
      double wsum = 0.0;
      for (const RawImu &s : w.imu)
        wsum += s.wm.norm();
      const double wmean = w.imu.empty() ? 0.0 : wsum / (double)w.imu.size();
      std::printf("[session] window %d rate/flow: mean|w|=%.2f rad/s flow=%.1f px\n", rep_.windows_harvested, wmean, m.fingerprint(7));
    }
    if (cfg_.verbose)
      std::printf("[session] window %d REJECTED by seed gates: |g|=%.2f (%.2f) |ba|=%.3f ang=%.1f mrad metric=%.3f m "
                  "(depth %.2f) solved=%d/%d fallback=%d\n",
                  rep_.windows_harvested, sr.g_mag, calib_.grav_mag, sr.ba_mag, 1e3 * sr.mean_ang_resid,
                  sr.mean_ang_resid * sr.median_depth, sr.median_depth, sr.feats_solved, (int)w.num_feats, sr.feats_fallback);
    return;
  }
  // Two-stage admission (solver-time): the reservoir decision is fingerprint-
  // only, so probe it BEFORE paying the nonlinear window solve -- the probe and
  // the real decision are exact twins (nothing mutates in between).
  if (!scorer_->peek(m))
    return;
  WindowSolveReport wr;
  const auto t_ba0 = std::chrono::steady_clock::now();
  const bool adm_ok = WindowBA::solve_and_export(w, calib_, true, wr, cfg_.joint.window_max_iters, false, nullptr, store_.ensure(w.uid));
  adm_ev_.wall_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ba0).count();
  adm_ev_.passes++;
  adm_ev_.iters += wr.iterations;
  adm_ev_.preint_s += wr.t_preint;
  adm_ev_.inner_s += wr.t_inner;
  adm_ev_.export_s += wr.t_export;
  adm_ev_.factor_s += wr.t_factor;
  (wr.preint_hit ? adm_ev_.phit : adm_ev_.pmiss)++;
  adm_ev_.tstop += wr.time_stopped ? 1 : 0;
  if (!adm_ok) {
    rep_.windows_rejected_ba++; // counted rejection, never a silent drop
    return;
  }
  adm_ev_.accepted++;
  const ReservoirDecision d = scorer_->consider(m);
  if (!d.accepted)
    return;
  slots_[d.slot] = std::move(w);
  slot_rep_[d.slot] = wr;
  if (d.slot < (int)probation_.size()) {
    if (sr.envelope_only && !probation_[d.slot])
      rep_.windows_probation++;
    probation_[d.slot] = sr.envelope_only ? 1 : 0;
  }
  if (d.is_holdout)
    rep_.windows_holdout++;
  rep_.windows_retained = scorer_->size();
  // display-only fusion + guided prompt (weakest whitened eigenpair)
  if (display_ && !d.is_holdout && wr.Lambda.rows() == calib_.local_dim()) {
    display_->add_window_information(wr.Lambda);
    Eigen::VectorXd improve;
    display_->progress(improve, rep_.prompt);
    if (cfg_.verbose)
      std::printf("[session] window %d retained (slot %d%s): seed %.1f mrad, prompt: %s\n", rep_.windows_harvested, d.slot,
                  d.is_holdout ? ", holdout" : "", 1e3 * sr.mean_ang_resid, rep_.prompt.c_str());
  }
}

const SessionReport &CalibSessionRunner::finish() {
  const auto t0 = std::chrono::steady_clock::now();
  if (state_ == RunnerState::COLLECT || state_ == RunnerState::THERMAL_HOLD) {
    if (harvester_ && harvester_->flush()) {
      WindowData w;
      WindowMeta m;
      if (harvester_->pop_window(w, m))
        handle_window_(std::move(w), m);
    }
    if (harvester_)
      rep_.windows_invalidated = harvester_->windows_invalidated();
    solve_verify_commit_();
  } else if (state_ != RunnerState::ABORT) {
    enter_(RunnerState::ABORT, "stream ended before collection started");
  }
  rep_.t_total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  rep_.final_state = state_;
  print_evidence_(); // evidence table prints on every terminal path (DONE and ABORT alike)
  return rep_;
}

// evidence helpers ------------------------------------------------------------
static long vmrss_kb_() {
#ifdef __linux__
  FILE *f = std::fopen("/proc/self/status", "r");
  if (!f)
    return 0;
  char line[128];
  long kb = 0;
  while (std::fgets(line, sizeof(line), f))
    if (std::sscanf(line, "VmRSS: %ld", &kb) == 1)
      break;
  std::fclose(f);
  return kb;
#else
  return 0;
#endif
}

void CalibSessionRunner::note_stage_(const std::string &label, const JointReport &r) {
  StageEvidence e;
  e.label = label;
  e.passes = r.evaluation_passes;
  e.accepted = r.accepted_passes;
  e.windows = r.windows_used;
  e.dim_p = r.dim_p;
  e.wall_s = r.wall_s;
  e.seed_s = r.t_seed_sum;
  e.preint_s = r.t_preint_sum;
  e.inner_s = r.t_inner_sum;
  e.export_s = r.t_export_sum;
  e.iters = r.inner_iters_sum;
  e.warm = r.warm_evals;
  e.cold = r.cold_evals;
  e.cold_plateau = r.cold_plateau;
  e.cold_anchor = r.cold_anchor;
  e.cold_cert = r.cold_cert;
  e.cold_jump = r.cold_jump;
  e.cold_won = r.cold_won;
  e.cold_won_guard = r.cold_won_guard;
  e.phit = r.preint_hits;
  e.pmiss = r.preint_misses;
  e.factor_s = r.t_factor_sum;
  e.tstop = r.time_stops;
  e.merit = r.final_merit;
  e.qn_max = r.qn_max_final;
  e.stop_pass = r.stop_pass;
  e.hit_budget = r.hit_wall_budget;
  e.rss_kb = vmrss_kb_();
  rep_.evidence.push_back(std::move(e));
}

void CalibSessionRunner::print_evidence_() const {
  if (!cfg_.evidence_table || rep_.evidence.empty())
    return;
  std::printf("[evidence] stage          wall_s pass acc win dim   iters  warm  cold plat anch cert jump won(g)  phit pmis   seed preint  factr  inner export      merit  rss_mb\n");
  StageEvidence tot;
  for (const StageEvidence &e : rep_.evidence) {
    std::printf("[evidence] %-14s %6.2f%c %4d %3d %3d %3d %7ld %5ld %5ld %4ld %4ld %4ld %4ld %3ld(%ld) %5ld %4ld %6.2f %6.2f %6.2f %6.2f %6.2f %10.4e %7ld%s\n",
                e.label.c_str(), e.wall_s, e.hit_budget ? '!' : ' ', e.passes, e.accepted, e.windows, e.dim_p, e.iters, e.warm, e.cold,
                e.cold_plateau, e.cold_anchor, e.cold_cert, e.cold_jump, e.cold_won, e.cold_won_guard, e.phit, e.pmiss, e.seed_s,
                e.preint_s, e.factor_s, e.inner_s, e.export_s, e.merit, e.rss_kb / 1024, e.stop_pass >= 0 ? " ES" : "");
    tot.wall_s += e.wall_s;
    tot.passes += e.passes;
    tot.accepted += e.accepted;
    tot.iters += e.iters;
    tot.warm += e.warm;
    tot.cold += e.cold;
    tot.cold_plateau += e.cold_plateau;
    tot.cold_anchor += e.cold_anchor;
    tot.cold_cert += e.cold_cert;
    tot.cold_jump += e.cold_jump;
    tot.cold_won += e.cold_won;
    tot.cold_won_guard += e.cold_won_guard;
    tot.phit += e.phit;
    tot.pmiss += e.pmiss;
    tot.seed_s += e.seed_s;
    tot.preint_s += e.preint_s;
    tot.factor_s += e.factor_s;
    tot.inner_s += e.inner_s;
    tot.export_s += e.export_s;
  }
  std::printf("[evidence] %-14s %6.2f  %4d %3d          %7ld %5ld %5ld %4ld %4ld %4ld %4ld %3ld(%ld) %5ld %4ld %6.2f %6.2f %6.2f %6.2f %6.2f            %7ld\n",
              "TOTAL", tot.wall_s, tot.passes, tot.accepted, tot.iters, tot.warm, tot.cold, tot.cold_plateau, tot.cold_anchor,
              tot.cold_cert, tot.cold_jump, tot.cold_won, tot.cold_won_guard, tot.phit, tot.pmiss, tot.seed_s, tot.preint_s,
              tot.factor_s, tot.inner_s, tot.export_s, vmrss_kb_() / 1024);
  // Determinism taint check: the inner-solve wall hang-guard must never fire
  // on a healthy run. A firing returns a load-dependent iterate (duel
  // arbitration can then flip), so the run is NOT replay-deterministic --
  // measured as 1-ulp committed-YAML drift between quiet and loaded runs of
  // one binary under a 5 s per-iterate cap.
  long tstop_total = 0;
  for (const StageEvidence &e : rep_.evidence)
    tstop_total += e.tstop;
  if (tstop_total > 0) {
    std::printf("[evidence] WARNING: %ld inner solve(s) hit the wall hang-guard — numerics are load-coupled; "
                "do NOT use this run for A/B, replay-parity, or falsifier evidence (stages:",
                tstop_total);
    for (const StageEvidence &e : rep_.evidence)
      if (e.tstop > 0)
        std::printf(" %s=%ld", e.label.c_str(), e.tstop);
    std::printf(")\n");
  }
}


SessionReport::AccelGateVerdict CalibSessionRunner::wald_accel_gate_(const std::vector<WindowData> &fused,
                                                                     std::vector<WindowWarmState> &warm, double budget_left_s,
                                                                     SessionReport::AccelGateVerdict *tg_verdict) {
  using V = SessionReport::AccelGateVerdict;
  if (tg_verdict)
    *tg_verdict = V::WALD_UNOBSERVABLE; // safe default for every early abstention below
  const auto t_g0 = std::chrono::steady_clock::now();
  rep_.a_wald_windows = (int)fused.size();
  rep_.a_wald_dropped = 0;
  if (budget_left_s <= 0.0) {
    if (cfg_.verbose)
      std::printf("[session] wald accel gate: UNOBSERVABLE (budget exhausted before the gate pass)\n");
    return V::WALD_UNOBSERVABLE; // safe abstention, never unbudgeted work
  }
  // Widened layout at the A1a point: calib_RAtoI temporarily true so
  // free_blocks() adds the 3 qA columns; noise re-frozen at the A1a point
  // exactly as a half/A1b solve entry would freeze it.
  const bool sv_qa = calib_.imu.calib_RAtoI;
  const bool sv_tg = calib_.imu.calib_tg;
  const bool sv_nf = calib_.noise_frozen;
  const ImuIntrinsicModel sv_nl = calib_.noise_lin;
  calib_.imu.calib_RAtoI = true;
  if (calib_.tg_enabled)
    calib_.imu.calib_tg = true; // widen the judged layout with the tg columns the session carries
  calib_.noise_lin = calib_.imu;
  calib_.noise_frozen = true;
  // label per local dim + PER-GROUP gate index sets (derived, never hardcoded). The accel chain
  // and tg are judged SEPARATELY: an unidentifiable tg (e.g. a rig whose true Tg is ~0 fits it
  // from noise, so its halves disagree) must freeze TG, not veto da/qA -- measured on the S1
  // synthetic: a monolithic verdict froze the whole chain and cost 1.13 deg of q_AtoI. When one
  // group is judged, the other group's dofs sit in the NUISANCE set (prior-folded).
  std::vector<std::string> lab;
  std::vector<int> gidx_a, gidx_t, rest;
  for (auto &b : calib_.free_blocks())
    for (int k = 0; k < b.lsize; ++k)
      lab.push_back(b.name + "[" + std::to_string(k) + "]");
  const int n = (int)lab.size();
  for (int i = 0; i < n; ++i) {
    const std::string &L = lab[i];
    if (L == "da[1]" || L == "da[3]" || L == "da[4]" || L.rfind("q_AtoI", 0) == 0)
      gidx_a.push_back(i);
    else if (L.rfind("tg", 0) == 0)
      gidx_t.push_back(i);
    else
      rest.push_back(i);
  }
  // ---- ONE warm evaluation pass (parallel, disjoint slots, fixed-order reduce after) ----
  struct Slot {
    bool ok = false;
    double t0 = 0.0;
    Eigen::MatrixXd L;
    Eigen::VectorXd g;
    WindowSolveReport wr;
  };
  std::vector<Slot> slots(fused.size());
  for (const WindowData &w : fused)
    store_.ensure(w.uid); // resolve before the pool (resize race guard)
  const int nthreads = std::max(1, std::min(cfg_.joint.num_threads, (int)fused.size()));
  ov_init::zbft_sfm::ParallelExecutor pool(nthreads);
  pool.parallel_ranges((int)fused.size(), [&](int, int b0, int b1) {
    for (int wi = b0; wi < b1; ++wi) {
      Slot &sl = slots[wi];
      sl.t0 = fused[wi].clone_times.empty() ? 0.0 : fused[wi].clone_times.front();
      WindowWarmState wcopy = (wi < (int)warm.size()) ? warm[wi] : WindowWarmState{};
      SharedCalib ccopy = calib_; // per-worker copy: solve mutates nuisance-side scratch only, but keep it airtight
      bool ok = false;
      if (wcopy.valid) {
        ok = WindowBA::solve_and_export(fused[wi], ccopy, true, sl.wr, cfg_.joint.window_max_iters, false, &wcopy,
                                        store_.ensure(fused[wi].uid));
      } else {
        LinearSeedReport sr;
        WindowData wd = fused[wi];
        const bool seeded = LinearSeed::seed_window(wd, ccopy, wd.seed_bg, sr, cfg_.seed);
        if (seeded || wd.has_seeds)
          ok = WindowBA::solve_and_export(wd, ccopy, true, sl.wr, cfg_.joint.window_max_iters, false, nullptr,
                                          store_.ensure(wd.uid));
      }
      if (ok && (int)sl.wr.Lambda.rows() == n) {
        sl.L = std::move(sl.wr.Lambda);
        sl.g = std::move(sl.wr.gred);
        sl.ok = true;
      }
    }
  });
  // restore the shared state before any statistics (single authority: the
  // downstream freeze at the gate-closed path)
  calib_.imu.calib_RAtoI = sv_qa;
  calib_.imu.calib_tg = sv_tg;
  calib_.noise_frozen = sv_nf;
  calib_.noise_lin = sv_nl;
  // ---- fixed-order reduction over time-sorted halves ----
  std::vector<size_t> order(fused.size());
  for (size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return slots[a].t0 != slots[b].t0 ? slots[a].t0 < slots[b].t0 : a < b;
  });
  const size_t mid = order.size() / 2;
  Eigen::MatrixXd Lh[2] = {Eigen::MatrixXd::Zero(n, n), Eigen::MatrixXd::Zero(n, n)};
  Eigen::VectorXd gh[2] = {Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n)};
  int nwin[2] = {0, 0};
  for (size_t oi = 0; oi < order.size(); ++oi) {
    const Slot &sl = slots[order[oi]];
    if (!sl.ok) {
      rep_.a_wald_dropped++;
      continue;
    }
    const int h = (oi < mid) ? 0 : 1;
    Lh[h] += sl.L;
    gh[h] += sl.g;
    nwin[h]++;
  }
  // synthesized evidence row for the pass
  {
    JointReport jr;
    jr.evaluation_passes = 1;
    jr.accepted_passes = 1;
    jr.windows_used = nwin[0] + nwin[1];
    jr.dim_p = n;
    for (const Slot &sl : slots) {
      jr.t_preint_sum += sl.wr.t_preint;
      jr.t_inner_sum += sl.wr.t_inner;
      jr.t_export_sum += sl.wr.t_export;
      jr.t_factor_sum += sl.wr.t_factor;
      jr.inner_iters_sum += sl.wr.iterations;
      (sl.wr.preint_hit ? jr.preint_hits : jr.preint_misses)++;
      jr.time_stops += sl.wr.time_stopped ? 1 : 0;
    }
    jr.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_g0).count();
    note_stage_("A1b-wald", jr);
  }
  if (nwin[0] < 2 || nwin[1] < 2) {
    if (cfg_.verbose)
      std::printf("[session] wald accel gate: UNOBSERVABLE (half starvation %d/%d)\n", nwin[0], nwin[1]);
    return V::WALD_UNOBSERVABLE;
  }
  // ---- the subspace judge: fold nuisances, Schur-marginalize, whiten, size, decide. Runs once
  // for the accel chain (report-writing; its verdict is the function's return) and once for tg. ----
  auto sig_of = [&](const std::string &L) {
    const size_t br = L.find('[');
    const std::string nm = L.substr(0, br);
    auto it = cfg_.joint.prior_sigma.find(nm);
    return it != cfg_.joint.prior_sigma.end() ? it->second : 0.05;
  };
  auto judge = [&](const std::vector<int> &gidx, const std::vector<int> &oidx, const char *tag, bool write_rep) -> V {
  const int m = (int)gidx.size();
  if (m == 0)
    return V::WALD_UNOBSERVABLE;
  Eigen::MatrixXd Lf[2] = {Lh[0], Lh[1]}; // per-judge copies: the nuisance fold must not leak across judges
  for (int h = 0; h < 2; ++h)
    for (int i : oidx) {
      const double sg = sig_of(lab[i]);
      Lf[h](i, i) += 1.0 / (sg * sg);
    }
  // ---- Schur marginal onto the gate subspace, per half ----
  Eigen::MatrixXd Sh[2];
  Eigen::VectorXd ghat[2];
  for (int h = 0; h < 2; ++h) {
    Eigen::MatrixXd Loo(oidx.size(), oidx.size()), Lgo(m, oidx.size());
    Eigen::MatrixXd Lgg(m, m);
    Eigen::VectorXd go(oidx.size()), gg(m);
    for (size_t a = 0; a < oidx.size(); ++a) {
      go(a) = gh[h](oidx[a]);
      for (size_t b = 0; b < oidx.size(); ++b)
        Loo(a, b) = Lf[h](oidx[a], oidx[b]);
    }
    for (int a = 0; a < m; ++a) {
      gg(a) = gh[h](gidx[a]);
      for (size_t b = 0; b < oidx.size(); ++b)
        Lgo(a, b) = Lf[h](gidx[a], oidx[b]);
      for (int b = 0; b < m; ++b)
        Lgg(a, b) = Lf[h](gidx[a], gidx[b]);
    }
    Eigen::LDLT<Eigen::MatrixXd> ldl(Loo);
    if (ldl.info() != Eigen::Success) {
      if (cfg_.verbose)
        std::printf("[session] wald %s gate: UNOBSERVABLE (nuisance LDLT failed, half %d)\n", tag, h + 1);
      return V::WALD_UNOBSERVABLE;
    }
    Sh[h] = Lgg - Lgo * ldl.solve(Lgo.transpose());
    ghat[h] = gg - Lgo * ldl.solve(go);
  }
  // ---- whitening ----
  Eigen::VectorXd wg(m);
  for (int a = 0; a < m; ++a)
    wg(a) = sig_of(lab[gidx[a]]);
  // ---- per-session dispersion: kappa_hat over H0 replicates measured 13.8
  // with a 15x per-session spread -- a scalar deflation false-froze 81% of
  // honest sessions. The session estimates its OWN dispersion from the
  // scatter of time-quarter gate steps against their claimed covariances
  // (method of moments): under H0,
  // sum_q (dq - dbar)' A_q (dq - dbar) ~ kappa_true * chi2_{(J-1) r}, so the
  // ratio is kappa_sess with df = (J-1)*r. a_info_deflate becomes the FLOOR.
  // The quarter systems reuse the identical fold/Schur/whiten/project path.
  auto quarter_system = [&](size_t o0, size_t o1, Eigen::MatrixXd &Sq, Eigen::VectorXd &gq) {
    Eigen::MatrixXd Lq = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd gv = Eigen::VectorXd::Zero(n);
    int cnt = 0;
    for (size_t oi = o0; oi < o1 && oi < order.size(); ++oi) {
      const Slot &sl = slots[order[oi]];
      if (!sl.ok)
        continue;
      Lq += sl.L;
      gv += sl.g;
      cnt++;
    }
    if (cnt < 1)
      return false;
    for (int i : oidx) {
      const double sg = sig_of(lab[i]);
      Lq(i, i) += 1.0 / (sg * sg);
    }
    Eigen::MatrixXd Loo(oidx.size(), oidx.size()), Lgo(m, oidx.size()), Lgg(m, m);
    Eigen::VectorXd go(oidx.size()), gg(m);
    for (size_t a = 0; a < oidx.size(); ++a) {
      go(a) = gv(oidx[a]);
      for (size_t b = 0; b < oidx.size(); ++b)
        Loo(a, b) = Lq(oidx[a], oidx[b]);
    }
    for (int a = 0; a < m; ++a) {
      gg(a) = gv(gidx[a]);
      for (size_t b = 0; b < oidx.size(); ++b)
        Lgo(a, b) = Lq(gidx[a], oidx[b]);
      for (int b = 0; b < m; ++b)
        Lgg(a, b) = Lq(gidx[a], gidx[b]);
    }
    Eigen::LDLT<Eigen::MatrixXd> ldl(Loo);
    if (ldl.info() != Eigen::Success)
      return false;
    Sq = wg.asDiagonal() * (Lgg - Lgo * ldl.solve(Lgo.transpose())) * wg.asDiagonal();
    gq = wg.asDiagonal() * (gg - Lgo * ldl.solve(go));
    return true;
  };
  const double kap = cfg_.a_info_deflate;
  Eigen::MatrixXd St[2];
  Eigen::VectorXd gt[2];
  for (int h = 0; h < 2; ++h) {
    St[h] = wg.asDiagonal() * Sh[h] * wg.asDiagonal();
    gt[h] = wg.asDiagonal() * ghat[h];
  }
  // ---- (a) observability on the joint eigenbasis ----
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(St[0] + St[1]);
  if (eig.info() != Eigen::Success)
    return V::WALD_UNOBSERVABLE;
  std::vector<int> obs;
  double min_kept = 0.0, min_all = std::numeric_limits<double>::infinity();
  for (int j = 0; j < m; ++j) {
    Eigen::VectorXd v = eig.eigenvectors().col(j);
    int mi = 0;
    v.cwiseAbs().maxCoeff(&mi);
    if (v(mi) < 0)
      v = -v; // deterministic sign
    const double e = std::min(v.dot(St[0] * v), v.dot(St[1] * v)) / kap;
    min_all = std::min(min_all, e);
    if (e >= cfg_.a_obs_min_eig) {
      obs.push_back(j);
      min_kept = (obs.size() == 1) ? e : std::min(min_kept, e);
    }
  }
  const int r = (int)obs.size();
  if (write_rep) {
    rep_.a_wald_r = r;
    rep_.a_wald_min_eig = min_all;
  }
  if (r == 0) {
    if (cfg_.verbose)
      std::printf("[session] wald %s gate: UNOBSERVABLE r=0/%d (min-half whitened eig %.2f < %.2f)\n", tag, m, min_all,
                  cfg_.a_obs_min_eig);
    return V::WALD_UNOBSERVABLE;
  }
  Eigen::MatrixXd P(m, r);
  for (int a = 0; a < r; ++a) {
    Eigen::VectorXd v = eig.eigenvectors().col(obs[a]);
    int mi = 0;
    v.cwiseAbs().maxCoeff(&mi);
    if (v(mi) < 0)
      v = -v;
    P.col(a) = v;
  }
  // ---- (b) correlated Wald on the half-step disagreement ----
  Eigen::MatrixXd A1 = P.transpose() * St[0] * P, A2 = P.transpose() * St[1] * P;
  Eigen::VectorXd b1 = P.transpose() * gt[0], b2 = P.transpose() * gt[1];
  // LDLT.info() alone does not certify PD, and the kept-direction Rayleigh floor holds on
  // the JOINT eigenvectors, not on each half's restriction to the span -- a half can be
  // near-singular there, exploding delta_h into junk steps. An explicit min-eig floor
  // turns that into honest abstention (freeze = safe).
  auto pd_min_eig = [](const Eigen::MatrixXd &A) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    return es.info() == Eigen::Success ? es.eigenvalues()(0) : -1.0;
  };
  const double pd1 = pd_min_eig(A1), pd2 = pd_min_eig(A2);
  const double pd_floor = 1e-9 * std::max(1.0, std::max(A1.trace(), A2.trace()));
  if (pd1 <= pd_floor || pd2 <= pd_floor) {
    if (cfg_.verbose)
      std::printf("[session] wald %s gate: UNOBSERVABLE (half information not PD on the kept span: %.2e/%.2e)\n", tag, pd1, pd2);
    return V::WALD_UNOBSERVABLE;
  }
  Eigen::LDLT<Eigen::MatrixXd> l1(A1), l2(A2);
  if (l1.info() != Eigen::Success || l2.info() != Eigen::Success)
    return V::WALD_UNOBSERVABLE;
  const Eigen::VectorXd d1 = -l1.solve(b1), d2 = -l2.solve(b2);
  const Eigen::VectorXd d = d1 - d2;
  // ---- per-session kappa from WITHIN-HALF quarter scatter (df-corrected sizing):
  // each half's quarters scatter around that half's OWN GLS mean, never the global
  // one. The global construction folds the between-halves contrast -- the very
  // signal T tests -- into the (J-1)r df: under H1 kappa_hat inflates (E[k] ~
  // kappa + lambda/((J-1)r)) and the threshold self-widens against exactly that
  // alternative (measured: split froze at dqA 0.762 deg while T read 3.2); at K<8
  // (Jq=2) the two-group scatter identity even forces T = r whenever kappa_hat
  // clears the floor -- violent disagreement passes. Within-group scatter is
  // orthogonal to the contrast (Cochran): T/r ~ F_{r,df}, df = sum_h (n_h-1) r;
  // at Jq=2 there is no estimate and the floor + chi2 legacy sizing applies.
  // Under H0 both constructions agree (within-half df 12 vs 18 at r=6: slightly
  // wider thresholds, honest signal-side); the MC harness re-pins per shape.
  const size_t K = order.size();
  const int Jq = (K >= 8) ? 4 : 2;
  std::vector<Eigen::MatrixXd> Aq;
  std::vector<Eigen::VectorXd> dq;
  std::vector<int> hq; // owning half per surviving quarter (quarter boundaries tile the halves exactly)
  for (int q = 0; q < Jq; ++q) {
    const size_t q0 = (K * q) / Jq, q1 = (K * (q + 1)) / Jq;
    Eigen::MatrixXd Sq;
    Eigen::VectorXd gq;
    if (!quarter_system(q0, q1, Sq, gq))
      continue;
    Eigen::MatrixXd Ap = P.transpose() * Sq * P;
    Eigen::LDLT<Eigen::MatrixXd> lq(Ap);
    if (lq.info() != Eigen::Success || pd_min_eig(Ap) <= 1e-12 * std::max(1.0, Ap.trace()))
      continue; // a non-PD quarter drops (fewer df), never poisons the estimate
    Aq.push_back(Ap);
    dq.push_back(-lq.solve(P.transpose() * gq));
    hq.push_back(q < Jq / 2 ? 0 : 1);
  }
  double kap_sess = 0.0;
  int kdf = 0;
  {
    double qsum = 0.0;
    for (int h = 0; h < 2; ++h) {
      Eigen::MatrixXd Asum = Eigen::MatrixXd::Zero(r, r);
      Eigen::VectorXd Adsum = Eigen::VectorXd::Zero(r);
      int nh = 0;
      for (size_t q = 0; q < Aq.size(); ++q)
        if (hq[q] == h) {
          Asum += Aq[q];
          Adsum += Aq[q] * dq[q];
          nh++;
        }
      if (nh < 2)
        continue; // a half with a single quarter carries no within-half contrast
      const Eigen::VectorXd dbar_h = Asum.ldlt().solve(Adsum);
      for (size_t q = 0; q < Aq.size(); ++q)
        if (hq[q] == h) {
          const Eigen::VectorXd e = dq[q] - dbar_h;
          qsum += e.dot(Aq[q] * e);
        }
      kdf += (nh - 1) * r;
    }
    if (kdf > 0)
      kap_sess = qsum / kdf;
  }
  // floor at the configured deflation: the estimator protects against the
  // measured under-dispersion (kappa_hat 13.8, spread [1.8, 28.3] over H0
  // replicates), the floor protects against a lucky low draw re-inflating
  // the gate's confidence. Observability classification stays at the floor
  // (avoids kappa-estimate circularity); final sizing belongs to the
  // junk-injection study.
  const double kap_eff = std::max(kap, kap_sess);
  if (write_rep) {
    rep_.a_wald_kappa = kap_sess;
    rep_.a_wald_df = kdf;
  }
  const Eigen::MatrixXd C = kap_eff * (A1.inverse() + A2.inverse());
  const double T = d.dot(C.ldlt().solve(d));
  // F_{r,df}(0.99) sizing for the ESTIMATED dispersion (df from the quarter
  // scatter; bucketed floor lookup; falls back to chi2/r when no estimate =
  // the fixed-kappa legacy sizing). Approximate quantiles -- the MC harness
  // sizes the end-to-end rule empirically and a_wald_thresh_scale absorbs
  // residual calibration.
  static const double chi2_99[6] = {6.635, 9.210, 11.345, 13.277, 15.086, 16.812};
  static const double f99[4][6] = {
      {13.75, 10.92, 9.78, 9.15, 8.75, 8.47},  // df ~ 6
      {9.33, 6.93, 5.95, 5.41, 5.06, 4.82},    // df ~ 12
      {8.29, 5.93, 5.09, 4.58, 4.25, 4.02},    // df ~ 18
      {7.82, 5.61, 4.72, 4.22, 3.90, 3.67},    // df ~ 24+
  };
  double Tthr;
  if (kdf >= 6) {
    const int bi = kdf >= 24 ? 3 : kdf >= 18 ? 2 : kdf >= 12 ? 1 : 0;
    Tthr = cfg_.a_wald_thresh_scale * r * f99[bi][std::min(r, 6) - 1];
  } else {
    Tthr = cfg_.a_wald_thresh_scale * chi2_99[std::min(r, 6) - 1];
  }
  // ---- (c) cross-prediction, both directions (Satterthwaite + Wilson-Hilferty) ----
  auto xthr = [&](const Eigen::MatrixXd &A) {
    const Eigen::MatrixXd AC = A * C;
    const double m1 = 0.5 * AC.trace(), m2 = 0.5 * (AC * AC).trace();
    if (m1 <= 0 || m2 <= 0)
      return std::numeric_limits<double>::infinity();
    const double gfac = m2 / (2.0 * m1), nu = 2.0 * m1 * m1 / m2, z = 2.32635;
    const double q = nu * std::pow(1.0 - 2.0 / (9.0 * nu) + z * std::sqrt(2.0 / (9.0 * nu)), 3.0);
    return cfg_.a_wald_thresh_scale * gfac * q;
  };
  const double X12 = 0.5 * d.dot(A1 * d), X21 = 0.5 * d.dot(A2 * d);
  const double xt1 = xthr(A1), xt2 = xthr(A2);
  if (write_rep) {
    rep_.a_wald_T = T;
    rep_.a_wald_x12 = X12;
    rep_.a_wald_x21 = X21;
    rep_.a_wald_xthr1 = xt1;
    rep_.a_wald_xthr2 = xt2;
  }
  // ---- physical deadband / ceiling in PHYSICAL units ----
  const Eigen::VectorXd dp = wg.asDiagonal() * (P * d); // half-disagreement, physical
  double dda = 0.0, dtg = 0.0, dqa2 = 0.0;
  for (int a = 0; a < m; ++a) {
    if (lab[gidx[a]].rfind("q_AtoI", 0) == 0)
      dqa2 += dp(a) * dp(a);
    else if (lab[gidx[a]].rfind("tg", 0) == 0)
      dtg = std::max(dtg, std::abs(dp(a)));
    else
      dda = std::max(dda, std::abs(dp(a)));
  }
  const double dqa_deg = std::sqrt(dqa2) * 180.0 / M_PI;
  if (write_rep) {
    rep_.a_wald_dqa_deg = dqa_deg;
    rep_.a_wald_dda_off = dda;
  }
  const Eigen::VectorXd dJ = -(A1 + A2).ldlt().solve(b1 + b2);
  const Eigen::VectorXd pJ = wg.asDiagonal() * (P * dJ);
  double jqa2 = 0.0, jda = 0.0, jtg = 0.0;
  for (int a = 0; a < m; ++a) {
    if (lab[gidx[a]].rfind("q_AtoI", 0) == 0)
      jqa2 += pJ(a) * pJ(a);
    else if (lab[gidx[a]].rfind("tg", 0) == 0)
      jtg = std::max(jtg, std::abs(pJ(a)));
    else
      jda = std::max(jda, std::abs(pJ(a)));
  }
  const double jqa_deg = std::sqrt(jqa2) * 180.0 / M_PI;
  if (write_rep) {
    rep_.a_wald_jqa_deg = jqa_deg;
    rep_.a_wald_jda = jda;
  }
  // ---- decision order ----
  if (jqa_deg > cfg_.a_qa_phys_ceiling_deg || jda > cfg_.a_da_off_phys_ceiling || jtg > cfg_.a_tg_phys_ceiling) {
    if (cfg_.verbose)
      std::printf("[session] wald %s gate: PHYS-CEILING fused step qA %.2f deg / da_off %.4f / tg %.5f -> INCONSISTENT\n", tag, jqa_deg, jda, jtg);
    return V::WALD_INCONSISTENT;
  }
  // r < m must dominate ANY consistent outcome: ordered later, the physical deadband
  // could rescue a stat-failed session into CONSISTENT at r < m and unlock the full
  // chain on uncertified directions -- the time-stable-junk case the gate exists to
  // refuse. No unlock without every direction eigen-certified.
  if (r < m) {
    if (cfg_.verbose)
      std::printf("[session] wald %s gate: UNOBSERVABLE r=%d/%d (agreement cannot certify unobserved directions)\n", tag, r, m);
    return V::WALD_UNOBSERVABLE;
  }
  const bool stat_fail = (T > Tthr) || (X12 > xt1) || (X21 > xt2);
  if (stat_fail) {
    const bool deadband = dda <= cfg_.a_split_da_floor && dqa_deg <= cfg_.a_split_qa_floor_deg && dtg <= cfg_.a_split_tg_floor;
    if (cfg_.verbose)
      std::printf("[session] wald %s gate: T=%.1f/%.1f X=%.1f/%.1f (thr %.1f/%.1f) kap=%.1f df=%d dqA %.3f deg dda %.4f -> %s\n", tag, T,
                  Tthr, X12, X21, xt1, xt2, kap_eff, kdf, dqa_deg, dda,
                  deadband ? "CONSISTENT (deadband: physically irrelevant)" : "INCONSISTENT");
    return deadband ? V::WALD_CONSISTENT : V::WALD_INCONSISTENT;
  }
  if (cfg_.verbose)
    std::printf("[session] wald %s gate: CONSISTENT T=%.1f<=%.1f X %.1f/%.1f r=%d kap=%.1f df=%d dqA %.3f deg\n", tag, T, Tthr, X12, X21,
                r, kap_eff, kdf, dqa_deg);
  return V::WALD_CONSISTENT;
  };

  // ---- ACCEL verdict decides the chain (tg dofs are nuisance for it); tg gets its OWN verdict,
  // judged only when the chain certifies (A1b will not run otherwise). An unidentifiable tg must
  // freeze tg alone -- measured live: a monolithic 15-dof gate returned UNOBSERVABLE r=5/15 on
  // 122.9 deg of attitude spread and vetoed a certifiable accel chain. ----
  std::vector<int> o_accel = rest, o_tg = rest;
  o_accel.insert(o_accel.end(), gidx_t.begin(), gidx_t.end());
  o_tg.insert(o_tg.end(), gidx_a.begin(), gidx_a.end());
  const V va = judge(gidx_a, o_accel, "accel", true);
  if (tg_verdict) {
    *tg_verdict = V::WALD_UNOBSERVABLE;
    if (!gidx_t.empty() && va == V::WALD_CONSISTENT)
      *tg_verdict = judge(gidx_t, o_tg, "tg", false);
  }
  return va;
}

CalibSessionRunner::CollectStatus CalibSessionRunner::collect_status() {
  CollectStatus st;
  if (!scorer_)
    return st;
  const int np = calib_.local_dim();
  // Prior-whitening vector: the same one the solve builds, so the eigenvalue is
  // reported in the units the commit rule lives in (sigma_post/sigma_prior).
  Eigen::VectorXd prior(np);
  {
    auto layout = calib_.free_blocks();
    int off = 0;
    for (auto &b : layout) {
      const double sg = cfg_.joint.prior_sigma.count(b.name) ? cfg_.joint.prior_sigma.at(b.name) : 1.0;
      for (int k = 0; k < b.lsize; ++k)
        prior(off + k) = sg;
      off += b.lsize;
    }
  }
  std::vector<Eigen::MatrixXd> Lw;
  std::vector<std::pair<double, double>> spans;
  for (int slot = 0; slot < scorer_->size(); ++slot) {
    if (slots_[slot].clone_times.empty())
      continue;
    st.n_retained++;
    if (scorer_->is_holdout(slot)) {
      st.n_holdout++;
      continue; // holdouts are never fused -- they verify
    }
    if (slot_rep_[slot].Lambda.rows() != np)
      continue;
    Lw.push_back(prior.asDiagonal() * slot_rep_[slot].Lambda * prior.asDiagonal());
    spans.push_back({scorer_->meta(slot).t0, scorer_->meta(slot).t1});
  }
  st.n_fusable = (int)Lw.size();
  if (st.n_fusable == 0)
    return st;
  WindowScorer::select_logdet(Lw, spans, cfg_.select_K, cfg_.select_overlap_penalty, &st.min_eig);
  // Enough to FUSE (select_K) and to VERIFY (min_holdout) -- an unverifiable
  // session must never be allowed to cut over early.
  const bool enough = (st.n_fusable >= cfg_.select_K) && (st.n_holdout >= std::max(1, cfg_.min_holdout));
  st.ready = enough && (cfg_.collect_min_eig > 0.0) && (st.min_eig >= cfg_.collect_min_eig);
  return st;
}

std::vector<int> CalibSessionRunner::select_stage_(const char *tag, const std::vector<Eigen::MatrixXd> &Lw,
                                                   const std::vector<std::pair<double, double>> &spans,
                                                   const std::vector<int> &dof_idx,
                                                   const std::vector<Eigen::VectorXd> &feat, int K,
                                                   double *min_eig_out) const {
  const int nf = feat.empty() ? 0 : (int)feat[0].size();
  const int m = (int)dof_idx.size() + nf;
  std::vector<Eigen::MatrixXd> Ls(Lw.size());
  for (size_t c = 0; c < Lw.size(); ++c) {
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(m, m);
    for (size_t r = 0; r < dof_idx.size(); ++r)
      for (size_t q = 0; q < dof_idx.size(); ++q)
        M((int)r, (int)q) = Lw[c](dof_idx[r], dof_idx[q]);
    if (nf > 0)
      M.bottomRightCorner(nf, nf) = cfg_.stage_feat_gain * feat[c] * feat[c].transpose();
    Ls[c] = std::move(M);
  }
  double me = 0.0;
  std::vector<int> sel = WindowScorer::select_logdet(Ls, spans, K, cfg_.select_overlap_penalty, &me);
  if (min_eig_out)
    *min_eig_out = me;
  if (cfg_.verbose)
    std::printf("[session] stage-select %s: %d/%d windows (dofs %d + feat %d, whitened min-eig %.2e)\n", tag, (int)sel.size(),
                (int)Lw.size(), (int)dof_idx.size(), nf, me);
  return sel;
}

void CalibSessionRunner::solve_verify_commit_() {
  enter_(RunnerState::SOLVE_REFINE, "end of collection");
  const auto t_solve0 = std::chrono::steady_clock::now();
  rep_.mean_exposure_s.assign((size_t)n_cams_, 0.0);
  for (int c = 0; c < n_cams_; ++c)
    if (exp_n_[(size_t)c] > 0)
      rep_.mean_exposure_s[(size_t)c] = exp_sum_[(size_t)c] / (double)exp_n_[(size_t)c];
  if (adm_ev_.passes > 0) { // collection-side admission BA aggregate
    adm_ev_.label = "admit-collect";
    adm_ev_.windows = rep_.windows_retained;
    adm_ev_.rss_kb = vmrss_kb_();
    rep_.evidence.push_back(adm_ev_);
  }
  // Thread the SESSION seeder config (incl. bootstrap adaptations: bias
  // presolve, widened no-still-baseline gates) into every fusion/verify
  // re-seed -- a default-config re-seed would silently re-tighten the gates
  // in exactly the stages that produce and validate the committed answer.
  cfg_.joint.seed = cfg_.seed;
  // Session-wide solve deadline: every staged call gets the REMAINING budget
  // (see SessionConfig::solve_budget_s). Pass configs through arm_budget_.
  auto arm_budget = [&](const JointConfig &jcin) {
    JointConfig j = jcin;
    if (cfg_.solve_budget_s > 0.0) {
      const double left =
          cfg_.solve_budget_s - std::chrono::duration<double>(std::chrono::steady_clock::now() - t_solve0).count();
      j.max_wall_s = std::max(5.0, left);
    }
    return j;
  };
  // A-CHAIN LEGACY INVARIANT: A0/A1a/A1b (and the split halves) run the
  // LEGACY two-path solver -- no cert, no carry, no early-stop. The split-half
  // falsifier's bands were calibrated on stage-independent legacy arbitration,
  // and every newer solver mechanism measurably shifts the halves-vs-full
  // statistics in the A chain: carried seeds flip the S1 sim falsifier (dqA
  // 0.551 z 2.84 vs 0.235 z 0.91 carry-free); cert-in-A0 flips a real log
  // (dqA 0.436 z 1.14 vs legacy 0.301 z 0.78 -- real margins are thinner than
  // sim). A-stage cold paths are load-bearing (measured 4.9-15.5% gains); the
  // certificate's rent is B-stage-only (~85% plateau double-solves). So the
  // newer machinery lives in the B chain: cert on (IMU chain frozen there),
  // carry behind its default-off flag, stage-entry p/whitener/shape jumps
  // self-arbitrating through the stamps.
  if (cfg_.a_candidate)
    cfg_.joint.conv_stop = true; // candidate arms conv-stop in the B-chain configs derived from joint
  // p4: fused evaluation everywhere (the capped-regime certificate polices
  // the B chain on qn alone -- the inner_converged clause is legacy-regime
  // only; halves re-pin fused off below as always).
  if (cfg_.p4)
    cfg_.joint.fused_schur = true;
  JointConfig jc_legacyA = cfg_.joint;
  if (cfg_.a_carry)
    jc_legacyA.use_carry = true; // mode-1 experiment: see SessionConfig::a_carry
  jc_legacyA.use_cert = false;
  jc_legacyA.use_carry = false;
  jc_legacyA.early_stop = false;
  // a_candidate re-baseline (halves excluded below -- they keep the
  // falsifier's own legacy statistics in EVERY configuration).
  if (cfg_.a_candidate) {
    // conv-stop ONLY. Adding the certificate to the A chain was rejected:
    // cert-in-A re-flips the falsifier (dqA 0.435 z 1.14) and runs SLOWER
    // (181.8 vs 162 s: the A-stage duels are load-bearing arbitration, the
    // cert just re-routes them; the wrongly-frozen chain then quadruples the
    // cam phase downstream). The legacy two-path stays; only EXCESS PASSES go.
    jc_legacyA.conv_stop = true;
  }
  JointWarmCarry carry;
  const SharedCalib calib_postboot = calib_; // VERIFY reference + revert source

  // ---- candidate set: thermal-binned non-holdout slots with a valid export ----
  const int np = calib_.local_dim();
  Eigen::VectorXd prior(np);
  {
    auto layout = calib_.free_blocks();
    int off = 0;
    for (auto &b : layout) {
      const double sg = cfg_.joint.prior_sigma.count(b.name) ? cfg_.joint.prior_sigma.at(b.name) : 1.0;
      for (int k = 0; k < b.lsize; ++k)
        prior(off + k) = sg;
      off += b.lsize;
    }
  }
  // Ensure a held-out window exists: the every-k rule leaves 0 holdouts on
  // short reservoirs, and an unverifiable session must never commit. Windows
  // feed nothing before this point, so retro-designating the LEAST informative
  // retained window (min whitened logdet contribution) is exact and
  // deterministic -- it sacrifices the least fusion value for verifiability.
  if (scorer_) {
    int n_hold_avail = 0, n_valid = 0;
    for (int slot = 0; slot < scorer_->size(); ++slot) {
      if (slots_[slot].clone_times.empty())
        continue;
      if (scorer_->is_holdout(slot))
        n_hold_avail++;
      else if (slot_rep_[slot].Lambda.rows() == np)
        n_valid++;
    }
    // Top-up loop (generalizes the single retro-designation): reach
    // min_holdout holdouts, capped by reservoir size (~N/4, max 3), by
    // stratified information rank -- weakest first (legacy pick), then the
    // MEDIAN of the remainder (a representative window, so the small-n VERIFY
    // is not judged solely on the least-informative data). The first holdout
    // keeps the legacy floor (>= 2 valid, leaves >= 1 fused); later ones must
    // keep N_fused >= max(2, N_ret-3) -- verifiability never starves fusion.
    while (true) {
      int n_hold = 0, n_valid = 0;
      for (int slot = 0; slot < scorer_->size(); ++slot) {
        if (slots_[slot].clone_times.empty())
          continue;
        if (scorer_->is_holdout(slot))
          n_hold++;
        else if (slot_rep_[slot].Lambda.rows() == np)
          n_valid++;
      }
      const int n_ret = n_hold + n_valid;
      const int cap = std::max(1, std::min(3, n_ret / 4));
      const int target = std::max(1, std::min(std::max(cfg_.min_holdout, 1), cap));
      if (n_hold >= target)
        break;
      const bool ok_floor = (n_hold == 0) ? (n_valid >= 2) : (n_valid - 1 >= std::max(2, n_ret - 3));
      if (!ok_floor)
        break;
      std::vector<std::pair<double, int>> ranked; // (whitened logdet gain, slot) ascending
      for (int slot = 0; slot < scorer_->size(); ++slot) {
        if (slots_[slot].clone_times.empty() || scorer_->is_holdout(slot) || slot_rep_[slot].Lambda.rows() != np)
          continue;
        const Eigen::MatrixXd Lw1 = prior.asDiagonal() * slot_rep_[slot].Lambda * prior.asDiagonal();
        const double gain =
            Eigen::LDLT<Eigen::MatrixXd>(Eigen::MatrixXd::Identity(np, np) + Lw1).vectorD().array().log().sum();
        ranked.push_back({gain, slot});
      }
      if (ranked.empty())
        break;
      std::sort(ranked.begin(), ranked.end());
      const size_t pick = (n_hold == 0) ? 0 : ranked.size() / 2;
      scorer_->force_holdout(ranked[pick].second);
      rep_.windows_holdout++;
      if (cfg_.verbose)
        std::printf("[session] retro-designating holdout %d/%d: slot %d (%s, logdet %.2f)\n", n_hold + 1, target, ranked[pick].second,
                    n_hold == 0 ? "least informative" : "median rank", ranked[pick].first);
    }
  }
  std::vector<int> bin = scorer_ ? scorer_->thermal_bin() : std::vector<int>();
  std::vector<Eigen::MatrixXd> Lw;
  std::vector<std::pair<double, double>> spans;
  std::vector<int> cand_slot;
  for (int slot : bin) {
    if (slot_rep_[slot].Lambda.rows() != np)
      continue;
    Lw.push_back(prior.asDiagonal() * slot_rep_[slot].Lambda * prior.asDiagonal());
    spans.push_back({scorer_->meta(slot).t0, scorer_->meta(slot).t1});
    cand_slot.push_back(slot);
  }
  if (Lw.empty()) {
    enter_(RunnerState::ABORT, "no fusable windows survived (excitation/thermal gates)");
    return;
  }
  std::vector<int> sel = WindowScorer::select_logdet(Lw, spans, cfg_.select_K, cfg_.select_overlap_penalty, &rep_.min_eig_whitened);
  std::vector<WindowData> fused;
  std::vector<int> fused_slot; // provenance for the post-A0 probation re-check
  for (int i : sel) {
    fused.push_back(slots_[cand_slot[i]]);
    fused_slot.push_back(cand_slot[i]);
  }
  rep_.windows_fused = (int)fused.size();
  if (cfg_.verbose)
    std::printf("[session] fusing %d/%d windows (D-optimal, whitened min-eig %.2e)\n", rep_.windows_fused, (int)Lw.size(),
                rep_.min_eig_whitened);
  // Abort floor on the selected set: a near-prior eigenvalue means an entire
  // calibration subspace collected no excitation -- solving anyway lets the
  // staged phases walk through a degenerate valley before COMMIT can abstain.
  if (rep_.min_eig_whitened < cfg_.min_eig_floor) {
    enter_(RunnerState::ABORT, "selected windows leave an unexcited calibration subspace (whitened min-eig below floor)");
    return;
  }

  // Does ANY IMU-intrinsic block have free parameters this session? (A rig whose
  // per-unit Dw/Da/R_AtoI are trusted factory data seeds them and freezes them.)
  // Read BEFORE A0, which frees/refreezes the flags internally.
  const bool imu_chain_free = calib_.imu.calib_dw || calib_.imu.calib_da || calib_.imu.calib_RAtoI || calib_.imu.calib_tg;

  // ---- stage-specific D-optimal subsets (selection-side only) ----
  // Same candidate pool + whitened one-pass information as the master
  // selection above (which keeps the abort floor and is the fallback set).
  // Stage sets only re-route which windows each stage SOLVES.
  std::vector<WindowData> fused_a0, fused_a1, fused_b;
  std::vector<int> slot_a0, slot_a1, slot_b;
  std::vector<WindowData> *w_a0 = &fused, *w_a1 = &fused, *w_b = &fused;
  if (cfg_.stage_select) {
    std::vector<int> idx_a0, idx_a1, idx_b;
    {
      int off = 0;
      for (auto &b : calib_.free_blocks()) {
        const bool sA0 = (b.name == "q_ItoC" || b.name == "p_IinC" || b.name == "td");
        const bool sA1 = (b.name == "dw" || b.name == "da" || b.name == "q_AtoI" || b.name == "tg");
        const bool sB = (b.name == "cam");
        for (int k = 0; k < b.lsize; ++k) {
          if (sA0) idx_a0.push_back(off + k);
          if (sA1) idx_a1.push_back(off + k);
          if (sB) idx_b.push_back(off + k);
        }
        off += b.lsize;
      }
    }
    const int N = (int)Lw.size();
    // gravity-direction convention: SET-level rule mirroring the accel gate
    // (seeded and mean directions have opposite sign conventions; never mix)
    int n_gseed = 0;
    for (int i = 0; i < N; ++i) {
      const WindowData &w = slots_[cand_slot[i]];
      const double gn = w.has_seeds ? w.seed_grav.norm() : 0.0;
      if (std::isfinite(gn) && gn > 5.0 && gn < 15.0)
        n_gseed++;
    }
    const bool use_seed_dir = (n_gseed >= 2);
    std::vector<Eigen::VectorXd> fA0(N), fA1(N), fB(N);
    for (int i = 0; i < N; ++i) {
      const WindowData &w = slots_[cand_slot[i]];
      const Eigen::Matrix<double, 12, 1> nfp =
          scorer_->meta(cand_slot[i]).fingerprint.cwiseQuotient(cfg_.scorer.fp_scale);
      Eigen::VectorXd g0 = Eigen::VectorXd::Zero(4);
      g0.head<4>() << nfp(0), nfp(1), nfp(2), nfp(6);
      // A1: specific-force direction x magnitude (mirrors the accel gate formulas)
      Eigen::Vector3d m3 = Eigen::Vector3d::Zero();
      double s1 = 0.0, s2 = 0.0;
      for (const RawImu &s : w.imu) {
        m3 += s.am;
        const double a = s.am.norm();
        s1 += a;
        s2 += a * a;
      }
      const int nimu = (int)w.imu.size();
      Eigen::Vector3d ghat = Eigen::Vector3d::Zero();
      double dyn = 0.0;
      if (nimu >= 2) {
        s1 /= nimu;
        dyn = std::sqrt(std::max(0.0, s2 / nimu - s1 * s1));
        if (use_seed_dir) {
          const double gn = w.has_seeds ? w.seed_grav.norm() : 0.0;
          if (std::isfinite(gn) && gn > 5.0 && gn < 15.0)
            ghat = w.seed_grav / gn;
        } else {
          const double mn = (m3 / nimu).norm();
          if (std::isfinite(mn) && mn > 5.0 && mn < 15.0)
            ghat = m3 / (double)nimu / mn;
        }
      }
      Eigen::VectorXd g1(4);
      g1 << ghat, dyn / std::max(cfg_.a_full_dyn_gate, 1e-6);
      // B: per-camera radial/quadrant fractions about the CURRENT center
      // (identical geometry to the phase-B gate loop)
      Eigen::VectorXd gb = Eigen::VectorXd::Zero(5 * n_cams_);
      for (int c = 0; c < n_cams_; ++c) {
        const CamCalib &kc = calib_.cams[(size_t)c];
        const double cx = kc.cam(2), cy = kc.cam(3);
        const double r_max = std::hypot(std::max(cx, kc.img_w - cx), std::max(cy, kc.img_h - cy));
        double n_all = 0, n_far = 0, nq[4] = {0, 0, 0, 0};
        for (const auto &cl : w.obs)
          for (const CloneObs &o : cl) {
            if (o.cam != c)
              continue;
            n_all += 1.0;
            if (std::hypot(o.uv(0) - cx, o.uv(1) - cy) > 0.7 * r_max)
              n_far += 1.0;
            nq[(o.uv(0) >= cx ? 1 : 0) + (o.uv(1) >= cy ? 2 : 0)] += 1.0;
          }
        if (n_all > 0) {
          gb(5 * c + 0) = (n_far / n_all) / cfg_.k34_radial_gate;
          for (int q = 0; q < 4; ++q)
            gb(5 * c + 1 + q) = (nq[q] / n_all) / (4.0 * cfg_.cam_center_quadrant_gate);
        }
      }
      fA0[i] = g0;
      fA1[i] = g1;
      fB[i] = gb;
    }
    const int Ka0 = cfg_.select_K_a0 > 0 ? cfg_.select_K_a0 : cfg_.select_K;
    const int Ka1 = cfg_.select_K_a1 > 0 ? cfg_.select_K_a1 : cfg_.select_K;
    const int Kb = cfg_.select_K_b > 0 ? cfg_.select_K_b : cfg_.select_K;
    for (int i : select_stage_("A0", Lw, spans, idx_a0, fA0, Ka0, &rep_.min_eig_a0)) {
      fused_a0.push_back(slots_[cand_slot[i]]);
      slot_a0.push_back(cand_slot[i]);
    }
    if (!fused_a0.empty())
      w_a0 = &fused_a0;
    if (imu_chain_free) {
      for (int i : select_stage_("A1", Lw, spans, idx_a1, fA1, Ka1, &rep_.min_eig_a1)) {
        fused_a1.push_back(slots_[cand_slot[i]]);
        slot_a1.push_back(cand_slot[i]);
      }
      if (!fused_a1.empty())
        w_a1 = &fused_a1;
    }
    if (cfg_.cam_mode > 0) {
      for (int i : select_stage_("B", Lw, spans, idx_b, fB, Kb, &rep_.min_eig_b)) {
        fused_b.push_back(slots_[cand_slot[i]]);
        slot_b.push_back(cand_slot[i]);
      }
      if (!fused_b.empty())
        w_b = &fused_b;
    }
    rep_.windows_a0 = (int)w_a0->size();
    rep_.windows_a1 = (int)w_a1->size();
    rep_.windows_b = (int)w_b->size();
  }

  // ---- phase A0: extrinsics + td only (IMU intrinsics frozen) ----
  // The ext/td subset is strongly observable and well conditioned; entering the
  // full-p solve from ITS optimum avoids the stiff dw/da-coupled valley from a
  // bootstrap-grade start (measured: direct full-p stalls at the seed, staged
  // reaches truth). Same block-coordinate philosophy as the camera phase B.
  for (CamCalib &kc : calib_.cams)
    kc.cam_mode = 0;
  {
    const bool f_dw = calib_.imu.calib_dw, f_da = calib_.imu.calib_da, f_qa = calib_.imu.calib_RAtoI;
    const bool f_tg = calib_.imu.calib_tg;
    calib_.imu.calib_dw = calib_.imu.calib_da = calib_.imu.calib_RAtoI = false;
    calib_.imu.calib_tg = false;
    JointReport repA0;
    const bool okA0 = JointCalib::solve(*w_a0, calib_, arm_budget(jc_legacyA), repA0, nullptr, &store_); // A-chain legacy invariant
    note_stage_("A0-ext-td", repA0);
    if (!okA0) {
      enter_(RunnerState::ABORT, "joint solve (ext/td stage) failed");
      return;
    }
    // rep_.joint carries the posterior the COMMIT gates read, and is normally
    // written by whatever stage runs LAST (A1a/A1b/B). With the IMU chain frozen
    // and cam_mode 0 there is no later stage, so A0's posterior IS the session's
    // -- without this it stays default-constructed (zero-length sigma) and the
    // commit gates index off the end of it.
    if (!imu_chain_free)
      rep_.joint = repA0;
    calib_.imu.calib_dw = f_dw;
    calib_.imu.calib_da = f_da;
    calib_.imu.calib_RAtoI = f_qa;
    calib_.imu.calib_tg = f_tg;
  }
  const SharedCalib calib_a0 = calib_; // full-chain (A1b) entry point when the gate opens
  // ---- probation re-check (drift-budget envelope admissions) ----
  // Envelope windows were admitted on the theory that their metric drift is
  // CALIBRATION-recoverable. A0 just solved ext/td: re-seed each probation
  // window at the post-A0 point under the STRICT gate (budget=0). Recoverable
  // drift has collapsed by now and passes; geometry failures have not and are
  // dropped BEFORE the IMU-intrinsic stages where drift junk aliases into
  // da/qA. Re-seeding in place upgrades the survivors' seed provenance.
  {
    LinearSeedConfig strict = cfg_.seed;
    strict.drift_budget_ms2 = 0.0;
    std::vector<std::pair<std::vector<WindowData> *, std::vector<int> *>> sets = {{&fused, &fused_slot}};
    if (cfg_.stage_select) {
      if (w_a0 == &fused_a0) sets.push_back({&fused_a0, &slot_a0});
      if (w_a1 == &fused_a1) sets.push_back({&fused_a1, &slot_a1});
      if (w_b == &fused_b) sets.push_back({&fused_b, &slot_b});
    }
    std::map<int, bool> verdict; // slot -> strict re-seed pass (computed once)
    for (auto &sp : sets) {
      std::vector<WindowData> &vw = *sp.first;
      std::vector<int> &vs = *sp.second;
      for (int i = (int)vw.size() - 1; i >= 0; --i) {
        const int slot = vs[i];
        if (slot < 0 || slot >= (int)probation_.size() || !probation_[slot])
          continue;
        auto it = verdict.find(slot);
        if (it == verdict.end()) {
          LinearSeedReport prr;
          const bool pass = LinearSeed::seed_window(vw[i], calib_, vw[i].seed_bg, prr, strict);
          verdict[slot] = pass;
          if (!pass) {
            rep_.windows_probation_dropped++;
            slots_[slot].clone_times.clear(); // exclude from every downstream consumer
            if (cfg_.verbose)
              std::printf("[session] probation window (slot %d) DROPPED at post-A0 strict re-check: metric %.3f m ang %.1f mrad\n",
                          slot, prr.mean_ang_resid * prr.median_depth, 1e3 * prr.mean_ang_resid);
          } else if (cfg_.verbose) {
            std::printf("[session] probation window (slot %d) cleared at post-A0: metric %.3f m\n", slot,
                        prr.mean_ang_resid * prr.median_depth);
          }
        } else if (it->second) {
          LinearSeedReport prr; // identical inputs -> identical bytes: upgrade THIS copy's provenance too
          LinearSeed::seed_window(vw[i], calib_, vw[i].seed_bg, prr, strict);
        }
        if (!verdict[slot]) {
          vw.erase(vw.begin() + i);
          vs.erase(vs.begin() + i);
        }
      }
    }
    rep_.windows_fused = (int)fused.size();
    if (fused.empty() || w_a0->empty() || (imu_chain_free && w_a1->empty()) || (cfg_.cam_mode > 0 && w_b->empty())) {
      // OFF: every pointer aliases `fused`, and the legacy reason string is part of the
      // parity surface (abort_reason ships in the report).
      enter_(RunnerState::ABORT, cfg_.stage_select ? "probation re-check emptied a fused set (geometry-failure windows only)"
                                                   : "probation re-check emptied the fused set (geometry-failure windows only)");
      return;
    }
    if (cfg_.stage_select) {
      rep_.windows_a0 = (int)w_a0->size();
      rep_.windows_a1 = (int)w_a1->size();
      rep_.windows_b = (int)w_b->size();
    }
  }
  // ---- S_A1 half-balance guard: the split/wald falsifiers slice BY TIME ----
  if (cfg_.stage_select && cfg_.stage_a1_balance_guard && imu_chain_free && w_a1 == &fused_a1) {
    std::vector<std::pair<double, Eigen::Vector3d>> tdir;
    int n_gs = 0;
    for (const WindowData &w : fused_a1) {
      const double gn = w.has_seeds ? w.seed_grav.norm() : 0.0;
      if (std::isfinite(gn) && gn > 5.0 && gn < 15.0)
        n_gs++;
    }
    for (const WindowData &w : fused_a1) {
      Eigen::Vector3d m3 = Eigen::Vector3d::Zero();
      for (const RawImu &s : w.imu)
        m3 += s.am;
      Eigen::Vector3d d = Eigen::Vector3d::Zero();
      if (n_gs >= 2) {
        const double gn = w.has_seeds ? w.seed_grav.norm() : 0.0;
        if (std::isfinite(gn) && gn > 5.0 && gn < 15.0)
          d = w.seed_grav / gn;
      } else if (!w.imu.empty()) {
        const double mn = (m3 / (double)w.imu.size()).norm();
        if (std::isfinite(mn) && mn > 5.0 && mn < 15.0)
          d = m3 / (double)w.imu.size() / mn;
      }
      tdir.push_back({w.clone_times.empty() ? 0.0 : w.clone_times.front(), d});
    }
    std::sort(tdir.begin(), tdir.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    const size_t mid = tdir.size() / 2;
    double spread[2] = {0.0, 0.0};
    int nwin[2] = {(int)mid, (int)(tdir.size() - mid)};
    for (int h = 0; h < 2; ++h) {
      const size_t lo = h ? mid : 0, hi = h ? tdir.size() : mid;
      for (size_t i = lo; i < hi; ++i)
        for (size_t j = i + 1; j < hi; ++j)
          if (tdir[i].second.norm() > 0.5 && tdir[j].second.norm() > 0.5)
            spread[h] = std::max(spread[h],
                                 std::acos(std::min(1.0, std::max(-1.0, tdir[i].second.dot(tdir[j].second)))) * 180.0 / M_PI);
    }
    if (std::min(nwin[0], nwin[1]) < 3 || std::min(spread[0], spread[1]) < 0.5 * cfg_.a_full_att_gate_deg) {
      w_a1 = &fused;
      rep_.stage_a1_fallback = true;
      rep_.windows_a1 = (int)fused.size();
      if (cfg_.verbose)
        std::printf("[session] stage-select A1 GUARD: time-halves starved (n %d/%d, spread %.0f/%.0f deg) — master set\n",
                    nwin[0], nwin[1], spread[0], spread[1]);
    }
  }
  // ---- accel-chain excitation gate (decides A1b BEFORE any accel dof moves) ----
  // Attitude diversity: spread of the mean specific-force direction (body frame)
  // across the fused windows. Dynamics: mean within-window std of |a_m| (pure
  // rotation-with-gravity motion keeps |a_m| ~ g and cannot split da scale from
  // the per-window accel bias, let alone the off-diagonals/q_AtoI).
  {
    // Direction source: the SEEDED window gravity (BA/seeder-consistent, in the
    // window body frame) -- sustained kinematic acceleration cannot fake it the
    // way a raw accel mean can. Windows without a usable seed gravity fall back
    // to the mean specific force, but NEVER mixed with seeded directions (the
    // two have opposite sign conventions and would fake ~180 deg of spread).
    std::vector<Eigen::Vector3d> gseed, gmean;
    double dyn_sum = 0.0;
    int dyn_n = 0;
    for (const WindowData &w : *w_a1) {
      Eigen::Vector3d m = Eigen::Vector3d::Zero();
      double s1 = 0.0, s2 = 0.0;
      for (const RawImu &s : w.imu) {
        m += s.am;
        const double a = s.am.norm();
        s1 += a;
        s2 += a * a;
      }
      const int n = (int)w.imu.size();
      if (n < 2)
        continue;
      const double gn = w.has_seeds ? w.seed_grav.norm() : 0.0;
      if (std::isfinite(gn) && gn > 5.0 && gn < 15.0)
        gseed.push_back(w.seed_grav / gn);
      const double mn = (m / n).norm();
      if (std::isfinite(mn) && mn > 5.0 && mn < 15.0)
        gmean.push_back(m / (double)n / mn);
      s1 /= n;
      dyn_sum += std::sqrt(std::max(0.0, s2 / n - s1 * s1));
      ++dyn_n;
    }
    const std::vector<Eigen::Vector3d> &gdir = (gseed.size() >= 2) ? gseed : gmean;
    for (size_t i = 0; i < gdir.size(); ++i)
      for (size_t j = i + 1; j < gdir.size(); ++j)
        rep_.accel_att_spread_deg =
            std::max(rep_.accel_att_spread_deg, std::acos(std::min(1.0, std::max(-1.0, gdir[i].dot(gdir[j])))) * 180.0 / M_PI);
    rep_.accel_dyn_ms2 = (dyn_n > 0) ? dyn_sum / dyn_n : 0.0;
  }
  // tg's own unlock verdict (set by whichever gate machinery runs); consumed at the freeze
  // below and at A1b entry. tg may only open WITH the accel chain, never instead of it.
  bool tg_open = false;
  const bool a_pre_gate = (rep_.accel_att_spread_deg >= cfg_.a_full_att_gate_deg) && (rep_.accel_dyn_ms2 >= cfg_.a_full_dyn_gate) &&
                          ((int)w_a1->size() >= cfg_.a_full_min_windows) && calib_.imu.calib_da && calib_.imu.calib_RAtoI;
  if (cfg_.verbose && calib_.imu.calib_da)
    std::printf("[session] accel excitation: attitude spread %.1f deg, dynamics %.2f m/s^2, %d windows -> full-chain pre-gate %s\n",
                rep_.accel_att_spread_deg, rep_.accel_dyn_ms2, (int)w_a1->size(), a_pre_gate ? "open (split-half decides)" : "CLOSED");

  // ---- A-chain short circuit: nothing to unlock ----
  // The A1 stages exist ONLY to estimate the IMU intrinsic chain. When every IMU
  // block is frozen (a rig whose per-unit Dw/Da/R_AtoI are trusted factory data
  // and were SEEDED, not searched), A1a's free set collapses to ext/td -- i.e. it
  // degenerates into a redundant re-solve of what A0 just converged, and the
  // accel gates have nothing to adjudicate. Measured: that dead stage cost 3.7 s
  // of an 11.8 s host session (~26 s at the device budget scale), buying a
  // re-polish of blocks A0 already owns. Skip straight to phase B.
  // (imu_chain_free is read BEFORE A0, which frees/refreezes the flags itself.)
  if (!imu_chain_free && cfg_.verbose)
    std::printf("[session] IMU intrinsics frozen at the seed (factory chain): skipping A1a/A1b — ext/td are A0's, and the\n"
                "          accel gates have nothing to unlock\n");

  // ---- phase A1a: dw + da-DIAGONAL from the staged point (q_AtoI frozen; the
  // off-diagonals are information-frozen via the per-dof prior) ----
  JointConfig jc_imu = jc_legacyA; // A-chain legacy invariant
  jc_imu.use_da_prior_vec = true;
  jc_imu.da_prior_vec.setConstant(1e-9);
  {
    const double sda = cfg_.joint.prior_sigma.count("da") ? cfg_.joint.prior_sigma.at("da") : 0.02;
    jc_imu.da_prior_vec(0) = jc_imu.da_prior_vec(2) = jc_imu.da_prior_vec(5) = sda; // d11 d22 d33
  }
  std::vector<WindowWarmState> warm_a1a; // the Wald gate's linearization states (accepted-point optima)
  if (imu_chain_free) {
    const bool f_qa = calib_.imu.calib_RAtoI;
    calib_.imu.calib_RAtoI = false;
    // tg stays FREE through A1a as a NUISANCE (session flag; a frozen chain never gets here with
    // tg on). Its COMMIT still unlocks only through the A1b gate -- the gate-closure paths below
    // revert both the flag AND the value to seed. Why not frozen "like q_AtoI": conditioning the
    // dw/da-diag stage on tg = seed(0) is a MODEL statement the data can falsify -- on a rig with
    // a real part-class Tg the stage absorbs Tg*a_hat into dw/da-diag, and every downstream judge
    // inherits the contaminated entry (measured on a Tg-bearing synthetic: frozen-judge dqA 0.583
    // deg / worst z 1.65 vs 0.91 at Tg=0). A Tg~0 rig fits ~0 under the 1e-3 prior -- harmless by
    // construction.
    const bool okA1a =
        JointCalib::solve(*w_a1, calib_, arm_budget(jc_imu), rep_.joint, nullptr, &store_, &warm_a1a); // legacy via jc_imu(jc_legacyA)
    note_stage_("A1a-dw-dadiag", rep_.joint);
    if (!okA1a) {
      enter_(RunnerState::ABORT, "joint solve failed");
      return;
    }
    calib_.imu.calib_RAtoI = f_qa;
    if (cfg_.verbose && calib_.imu.calib_tg) {
      const Eigen::Map<const Eigen::Matrix<double, 9, 1>> tv(calib_.imu.Tg.data());
      std::printf("[session] A1a tg (nuisance): |Tg| %.3f deg/s @1g, el(storage):", calib_.imu.Tg.rowwise().norm().maxCoeff() * 9.81 * 180.0 / M_PI);
      for (int k = 0; k < 9; ++k)
        std::printf(" %+.2e", tv(k));
      std::printf("\n");
    }
  }

  // ---- A1b unlock: SPLIT-HALF consistency from the A1a point ----
  // The full accel chain is solved independently on the first and second
  // time-half of the fused windows. Real-sensor junk modes (bias/thermal/
  // vibration absorbed into weakly-excited dofs) are time-correlated and
  // disagree between halves; a genuinely observable chain reproduces within
  // its posteriors. This is the IMU-intrinsic falsifier made a gate.
  if (a_pre_gate && cfg_.a_gate_mode == 1) {
    // ---- Wald reduced-information gate DECIDES (flight-profile mode; the
    // split-half machinery below never runs). One widened warm evaluation
    // pass at the A1a point replaces the two nonlinear half-solves.
    const double budget_left =
        cfg_.solve_budget_s > 0.0
            ? cfg_.solve_budget_s - std::chrono::duration<double>(std::chrono::steady_clock::now() - t_solve0).count()
            : 1e9;
    SessionReport::AccelGateVerdict tg_v = SessionReport::AccelGateVerdict::WALD_UNOBSERVABLE;
    rep_.a_wald_verdict = wald_accel_gate_(*w_a1, warm_a1a, budget_left, &tg_v);
    rep_.a_full_open = (rep_.a_wald_verdict == SessionReport::AccelGateVerdict::WALD_CONSISTENT);
    if (calib_.tg_enabled)
      rep_.tg_gate_verdict = tg_v;
    tg_open = rep_.a_full_open && calib_.tg_enabled && tg_v == SessionReport::AccelGateVerdict::WALD_CONSISTENT;
    if (cfg_.verbose && calib_.tg_enabled && rep_.a_full_open)
      std::printf("[session] tg gate: %s\n", tg_open ? "CONSISTENT (tg unlocked with the chain)"
                                                      : "not certified (tg stays at its seed; the chain still opens)");
  } else if (a_pre_gate) {
    // fused is in D-optimal SELECTION order; the split must be by TIME so the
    // halves straddle the session's thermal/drift evolution (the junk mode).
    std::vector<size_t> torder(w_a1->size());
    for (size_t i = 0; i < torder.size(); ++i)
      torder[i] = i;
    std::sort(torder.begin(), torder.end(), [&](size_t a, size_t b) {
      const double ta = (*w_a1)[a].clone_times.empty() ? 0.0 : (*w_a1)[a].clone_times.front();
      const double tb = (*w_a1)[b].clone_times.empty() ? 0.0 : (*w_a1)[b].clone_times.front();
      return ta < tb;
    });
    const size_t mid = torder.size() / 2;
    std::vector<WindowData> h1, h2;
    for (size_t i = 0; i < torder.size(); ++i)
      (i < mid ? h1 : h2).push_back((*w_a1)[torder[i]]);
    SharedCalib c1 = calib_, c2 = calib_;
    // The CHAIN is judged with tg CONSTANT -- held at the fused A1a nuisance estimate, the SAME
    // value in both halves. Per-half tg fitting stays out of this judge (a junk tg fits noise
    // differently per half and drags each half's da/qA -- measured on S1: 0.23 deg ext / 1.13 deg
    // qA regression with per-half-tg halves), while holding it at ZERO on a rig with a real Tg
    // contaminates the halves the other way (measured: dqA 0.583 deg / worst z 1.65 vs 0.91 at
    // Tg=0). tg gets its OWN half pair below.
    c1.imu.calib_tg = false;
    c2.imu.calib_tg = false;
    JointReport r1, r2;
    // The FALSIFIER runs at its calibrated legacy operating point: the
    // split-half bands/floors were tuned against the legacy two-path solver,
    // and the certificate shifts the halves' convergence enough to flip
    // verdicts (measured: dqA 0.301 -> 0.491 on identical data, chain
    // wrongly frozen). Diagnostic solves are 8-window bounded -- correctness
    // of the GATE outranks their wall clock. Production stages keep the
    // newer machinery.
    JointConfig jc_half = cfg_.joint;
    jc_half.use_cert = false;
    jc_half.use_carry = false;
    jc_half.early_stop = false;
    jc_half.conv_stop = false; // the falsifier's statistics stay legacy in every configuration
    jc_half.fused_schur = false;
    jc_half.duel_on_accept = false;
    // The halves are independent programs (disjoint window subsets, separate
    // calib copies, separate reports); each solve's arithmetic is already
    // thread-count-invariant (fixed-range partition, worker-ordered
    // reduction), so running them concurrently changes no bytes in either --
    // it removes a control-flow accident worth ~22 s at the host shape
    // (wall = max of the halves instead of the sum). GUARD: resolve
    // every store slot on THIS thread first -- PreintStore::ensure may resize
    // by_uid, and the concurrent solves must only touch disjoint EXISTING
    // entries. Budgets are armed once, before launch, from the same instant
    // (identical max_wall_s in both configs -- order-independent).
    for (const WindowData &w : *w_a1)
      store_.ensure(w.uid);
    const JointConfig jc_h1 = arm_budget(jc_half), jc_h2 = arm_budget(jc_half);
    bool ok1 = false, ok2 = false;
    std::thread th2([&] { ok2 = JointCalib::solve(h2, c2, jc_h2, r2, nullptr, &store_); });
    ok1 = JointCalib::solve(h1, c1, jc_h1, r1, nullptr, &store_);
    th2.join();
    note_stage_("A1b-half1", r1);
    note_stage_("A1b-half2", r2);
    if (ok1 && ok2) {
      // per-dof posterior sigmas by label (band = k * rss + floor guards)
      auto sig_of = [](const JointReport &r, const std::string &lab) {
        for (size_t i = 0; i < r.labels.size(); ++i)
          if (r.labels[i] == lab)
            return r.sigma((int)i);
        return 0.0;
      };
      // Band per dof: sigma term (precision) OR signal-fraction term (the
      // scale-free criterion -- "the halves agree to a third of what they
      // claim to measure"). Junk modes scatter at the size of their own
      // claimed signal and fail both. One judge, reused verbatim across the
      // frozen-tg pair and the tg-free re-judge below.
      auto qangle = [](const Eigen::Vector4d &qa, const Eigen::Vector4d &qb) {
        return 2.0 * ov_core::quat_multiply(qa, ov_core::Inv(qb)).head<3>().norm() * 180.0 / M_PI;
      };
      struct ChainJudge {
        bool agree = false;
        double worst_z = 0.0, dqa = 0.0, qa_signal = 0.0, qa_band = 0.0;
      };
      auto judge_chain = [&](const SharedCalib &h1c, const SharedCalib &h2c, const JointReport &hr1, const JointReport &hr2,
                             const SharedCalib &ref) {
        ChainJudge j;
        j.agree = true;
        for (int k = 0; k < 6; ++k) {
          const std::string lab = "da[" + std::to_string(k) + "]";
          const double signal = std::max(std::abs(h1c.imu.da(k) - ref.imu.da(k)), std::abs(h2c.imu.da(k) - ref.imu.da(k)));
          const double band = std::max({cfg_.a_split_da_floor, cfg_.a_split_sigma_k * std::hypot(sig_of(hr1, lab), sig_of(hr2, lab)),
                                        cfg_.a_split_signal_frac * signal});
          const double z = std::abs(h1c.imu.da(k) - h2c.imu.da(k)) / band;
          j.worst_z = std::max(j.worst_z, z);
          j.agree = j.agree && (z <= 1.0);
        }
        double sq = 0.0;
        for (int k = 0; k < 3; ++k) {
          const std::string lab = "q_AtoI[" + std::to_string(k) + "]";
          sq = std::max(sq, std::hypot(sig_of(hr1, lab), sig_of(hr2, lab)));
        }
        j.dqa = qangle(h1c.imu.q_AtoI, h2c.imu.q_AtoI);
        j.qa_signal = std::max(qangle(h1c.imu.q_AtoI, ref.imu.q_AtoI), qangle(h2c.imu.q_AtoI, ref.imu.q_AtoI));
        j.qa_band = std::max({cfg_.a_split_qa_floor_deg, cfg_.a_split_sigma_k * sq * 180.0 / M_PI,
                              cfg_.a_split_signal_frac * j.qa_signal});
        j.worst_z = std::max(j.worst_z, j.dqa / j.qa_band);
        j.agree = j.agree && (j.dqa <= j.qa_band);
        return j;
      };
      ChainJudge jf = judge_chain(c1, c2, r1, r2, calib_); // the frozen-tg pair (the legacy judge, A1a-referenced)
      bool agree = jf.agree;
      bool rejudged = false;
      ChainJudge jr; // the tg-free re-judge (retry arbitration; stats reported when it runs)
      rep_.a_full_open = agree;
      // tg's OWN falsifier: a second half pair with tg free (the same freedom A1b would grant),
      // judged only on the tg elements. It runs when the chain certified (the tg unlock question)
      // AND when it did not -- the RETRY arbitration: the chain judge above CONDITIONS its halves
      // on tg = seed, and a rig with a REAL part-class Tg falsifies that conditioning (each
      // tg-frozen half absorbs Tg*a_hat into its own da/qA through its own excitation geometry --
      // measured on a Tg-bearing synthetic: worst z 1.27 vs 0.91 at Tg=0, chain wrongly frozen).
      // The wald judge (mode 1) marginalizes the tg columns and does not have this failure; here
      // the conditioning choice is arbitrated by tg's own falsifier: certify a REPRODUCIBLE Tg
      // first, then re-judge the chain on the SAME tg-free halves (zero extra solves). A junk tg
      // refuses at the tg pair and the legacy freeze stands byte-identically.
      if (calib_.tg_enabled) {
        // The tg pair enters from the A0 point -- A1b's OWN entry -- not the A1a point the frozen
        // pair uses. The pair's question is "will A1b's answer reproduce?", so the halves must be
        // solved under A1b's conditions: A1a's dw/da-diag were solved with tg frozen at seed, so
        // on a rig with a real Tg they carry its absorption, and halves entered there park in
        // scattered (qA, tg) basins (measured: dqA 0.583-0.630 deg vs the ~0.3 band, invariant to
        // the tg conditioning -- entry contamination, the same basin mechanism that moved A1b
        // itself from A1a-entry to A0-entry: 0.73 vs 0.57 deg).
        SharedCalib g1 = calib_a0, g2 = calib_a0;
        JointReport gr1, gr2;
        // (Budget note, measured: doubling the pair's outer budget moved nothing -- dqA 0.325 ->
        // 0.327 deg, both halves at their per-half stationary points either way. The residual
        // per-half qA spread is the documented flat valley at the per-half information level --
        // an EXCITATION property, not a solver-budget one -- so the pair keeps the stage budget.)
        const JointConfig jc_g1 = arm_budget(jc_half), jc_g2 = arm_budget(jc_half);
        bool gok1 = false, gok2 = false;
        std::thread gth2([&] { gok2 = JointCalib::solve(h2, g2, jc_g2, gr2, nullptr, &store_); });
        gok1 = JointCalib::solve(h1, g1, jc_g1, gr1, nullptr, &store_);
        gth2.join();
        note_stage_("A1b-tg-half1", gr1);
        note_stage_("A1b-tg-half2", gr2);
        if (gok1 && gok2) {
          bool agree_tg = true;
          const Eigen::Map<const Eigen::Matrix<double, 9, 1>> t1(g1.imu.Tg.data()), t2(g2.imu.Tg.data()),
              t0(calib_a0.imu.Tg.data()); // signal referenced to the pair's OWN entry (byte-equal to calib_'s seed tg)
          double worst_ztg = 0.0;
          int worst_ktg = 0;
          double worst_parts[4] = {0, 0, 0, 0}; // |d|, 3sig term, signal, band of the worst element
          // tg sigma term carries the MEASURED exported-Lambda under-dispersion (a_info_deflate,
          // VARIANCE semantics -- the same kappa the wald judge applies to the same exports): the
          // tg columns ride the chain nuisance, and a raw 3-sigma band refuses halves whose |d|
          // sits INSIDE the kappa-corrected posterior (measured: |d| 1.22e-4 vs 3*sqrt(2)*hypot
          // = 1.32e-4). da/qA bands stay untouched -- their calibration is the validated corpus's.
          const double kap_sig = std::sqrt(std::max(1.0, cfg_.a_info_deflate));
          for (int k = 0; k < 9; ++k) {
            const std::string lab = "tg[" + std::to_string(k) + "]";
            const double signal = std::max(std::abs(t1(k) - t0(k)), std::abs(t2(k) - t0(k)));
            const double sig3 = cfg_.a_split_sigma_k * kap_sig * std::hypot(sig_of(gr1, lab), sig_of(gr2, lab));
            const double band = std::max({cfg_.a_split_tg_floor, sig3, cfg_.a_split_signal_frac * signal});
            const double z = std::abs(t1(k) - t2(k)) / band;
            if (z > worst_ztg) {
              worst_ztg = z;
              worst_ktg = k;
              worst_parts[0] = std::abs(t1(k) - t2(k));
              worst_parts[1] = sig3;
              worst_parts[2] = signal;
              worst_parts[3] = band;
            }
            agree_tg = agree_tg && (z <= 1.0);
          }
          if (cfg_.verbose) { // which element carried the verdict, and which band term bound it
            std::printf("[session] split-half tg worst element [%d]: |d| %.2e vs band %.2e (floor %.1e | 3*sqrt(kap)*sig %.2e | %.2f*signal %.2e)\n",
                        worst_ktg, worst_parts[0], worst_parts[3], cfg_.a_split_tg_floor, worst_parts[1],
                        cfg_.a_split_signal_frac, cfg_.a_split_signal_frac * worst_parts[2]);
            std::printf("[session] split-half tg halves: |Tg1| %.3f |Tg2| %.3f deg/s @1g; half qA-from-entry %.3f / %.3f deg\n",
                        g1.imu.Tg.rowwise().norm().maxCoeff() * 9.81 * 180.0 / M_PI,
                        g2.imu.Tg.rowwise().norm().maxCoeff() * 9.81 * 180.0 / M_PI,
                        qangle(g1.imu.q_AtoI, calib_a0.imu.q_AtoI), qangle(g2.imu.q_AtoI, calib_a0.imu.q_AtoI));
          }
          rep_.tg_gate_verdict =
              agree_tg ? SessionReport::AccelGateVerdict::SPLIT_CONSISTENT : SessionReport::AccelGateVerdict::SPLIT_INCONSISTENT;
          if (agree) {
            tg_open = agree_tg; // legacy path: chain certified, tg unlocks with it (or not)
          } else if (agree_tg) {
            // RETRY: a reproducible Tg falsified the frozen-tg conditioning -- re-judge the chain
            // on the tg-free A0-entered halves already solved above (A1b's own conditions;
            // signal/claim referenced to THEIR entry). Opens BOTH or NEITHER (tg may only open
            // with the chain).
            rejudged = true;
            jr = judge_chain(g1, g2, gr1, gr2, calib_a0);
            if (jr.agree) {
              agree = true;
              rep_.a_full_open = true;
              tg_open = true;
            }
          }
          if (cfg_.verbose)
            std::printf("[session] split-half tg: worst z %.2f -> %s\n", worst_ztg,
                        agree_tg ? (tg_open ? "CONSISTENT (tg unlocks with the chain)" : "CONSISTENT, but the chain did not certify -> tg stays at its seed")
                                 : "not certified (tg stays at its seed)");
        } else {
          rep_.tg_gate_verdict = SessionReport::AccelGateVerdict::SPLIT_FAILED;
          if (cfg_.verbose)
            std::printf("[session] split-half tg: half-solve failed -> tg stays at its seed\n");
        }
      }
      rep_.a_wald_verdict =
          agree ? SessionReport::AccelGateVerdict::SPLIT_CONSISTENT : SessionReport::AccelGateVerdict::SPLIT_INCONSISTENT;
      if (cfg_.verbose) {
        std::printf("[session] split-half accel chain (tg-frozen judge): dqA %.3f deg (signal %.3f, band %.3f), worst z %.2f -> %s "
                    "[half qA-from-entry %.3f / %.3f deg]\n",
                    jf.dqa, jf.qa_signal, jf.qa_band, jf.worst_z,
                    jf.agree ? "CONSISTENT (A1b unlocked)"
                             : (rejudged ? "INCONSISTENT (conditioning suspect: real tg certified)" : "INCONSISTENT (chain frozen at A1a)"),
                    qangle(c1.imu.q_AtoI, calib_.imu.q_AtoI), qangle(c2.imu.q_AtoI, calib_.imu.q_AtoI));
        if (rejudged)
          std::printf("[session] split-half accel chain RE-JUDGED on the tg-free halves: dqA %.3f deg (signal %.3f, band %.3f), "
                      "worst z %.2f -> %s\n",
                      jr.dqa, jr.qa_signal, jr.qa_band, jr.worst_z,
                      jr.agree ? "CONSISTENT (A1b + tg unlocked)" : "INCONSISTENT (chain frozen at A1a; tg stays at its seed)");
      }
    } else {
      rep_.a_wald_verdict = SessionReport::AccelGateVerdict::SPLIT_FAILED;
      if (cfg_.verbose)
        std::printf("[session] split-half accel chain: half-solve failed -> chain frozen at A1a\n");
    }
    if (cfg_.a_gate_mode == 2) {
      // SHADOW: split-half decided above; the wald verdict is measured and
      // logged for the paired-verdict A/B record, never consulted.
      const double budget_left =
          cfg_.solve_budget_s > 0.0
              ? cfg_.solve_budget_s - std::chrono::duration<double>(std::chrono::steady_clock::now() - t_solve0).count()
              : 1e9;
      SessionReport::AccelGateVerdict sh_tg = SessionReport::AccelGateVerdict::WALD_UNOBSERVABLE;
      const auto shadow = wald_accel_gate_(*w_a1, warm_a1a, budget_left, calib_.tg_enabled ? &sh_tg : nullptr);
      if (cfg_.verbose) {
        auto vn = [](SessionReport::AccelGateVerdict v) {
          return v == SessionReport::AccelGateVerdict::WALD_CONSISTENT     ? "CONSISTENT"
                 : v == SessionReport::AccelGateVerdict::WALD_INCONSISTENT ? "INCONSISTENT"
                                                                           : "UNOBSERVABLE";
        };
        std::printf("[session] wald gate SHADOW verdict: %s (split-half decided %s)\n", vn(shadow),
                    rep_.a_full_open ? "CONSISTENT" : "INCONSISTENT");
        if (calib_.tg_enabled)
          std::printf("[session] wald tg SHADOW verdict: %s (split-half tg decided %s)\n", vn(sh_tg),
                      tg_open ? "CONSISTENT" : "not certified");
      }
    }
  }

  // Gate closed: q_AtoI AND tg stay at the seed for the REST of the session (phase B re-solves
  // the full vector and must not silently reopen them; commit/writeback ship the seed values
  // exactly like the --no-imu-intrinsics ablation). A chain that opened WITHOUT a certified tg
  // freezes tg alone. tg's VALUE reverts with its flag: A1a estimated it as a nuisance, and an
  // uncertified nuisance estimate must not ride into phase B / the commit walk as a "seed"
  // passenger (it would ship OUTSIDE committed_blocks).
  if (!rep_.a_full_open) {
    calib_.imu.calib_RAtoI = false;
    calib_.imu.calib_tg = false;
    calib_.imu.Tg = calib_postboot.imu.Tg;
  } else if (!tg_open) {
    calib_.imu.calib_tg = false;
    calib_.imu.Tg = calib_postboot.imu.Tg;
  }

  // ---- phase A1b: full accel chain (unlocked only), entered from the A0
  // point -- NOT the A1a point: da-diag + per-window biases have already
  // absorbed part of the misalignment signal there, parking q_AtoI in a
  // shallower basin (measured on the synthetic suite: 0.73 deg from A1a vs
  // 0.57 deg from A0). A1a remains the split-half midpoint and the
  // gate-closed product. ----
  if (rep_.a_full_open) {
    SharedCalib calib_a1a = calib_;
    calib_ = calib_a0; // restores session flags, INCLUDING calib_tg -- re-apply the tg verdict
    if (!tg_open)
      calib_.imu.calib_tg = false; // calib_a0's tg VALUE is the seed already (A0 never solves tg)
    JointReport repA1b;
    JointConfig jcA1b = jc_legacyA;
    if (cfg_.p4)
      jcA1b.fused_polish_accepts = 2; // close the capped-trajectory residual at the stage that feeds B
    const bool okA1b = JointCalib::solve(*w_a1, calib_, arm_budget(jcA1b), repA1b, nullptr, &store_); // A-chain legacy invariant
    note_stage_("A1b-full", repA1b);
    if (okA1b) {
      rep_.joint = repA1b;
    } else {
      calib_ = calib_a1a; // best-effort stage: the A1a point stands
      calib_.imu.calib_RAtoI = false;
      calib_.imu.calib_tg = false;
      calib_.imu.Tg = calib_postboot.imu.Tg; // calib_a1a carries A1a's nuisance tg -- revert with the flag
      rep_.a_full_open = false;
    }
  }
  // tg's final unlock state, ONE authority: the flag can only survive to here when the chain
  // certified, tg's own falsifier certified, and A1b accepted (every closed path froze it above).
  rep_.tg_open = calib_.imu.calib_tg;

  // ---- phase B: staged camera-intrinsic refinement AFTER temporal/IMU ----
  if (cfg_.cam_mode > 0) {
    // Radial-coverage gate for k3/k4 (fraction of obs beyond 0.7 * r_max), decided PER CAMERA: it
    // asks whether THIS camera actually saw its own image corners, and one camera reaching them
    // says nothing about another pointing somewhere else entirely.
    JointConfig jc = cfg_.joint;
    jc.use_cam_prior_vec = true;
    jc.cam_prior_vec.assign((size_t)n_cams_, (cfg_.cam_mode == 1) ? cfg_.cam_refine_prior : cfg_.cam_full_prior);
    std::vector<double> n_far((size_t)n_cams_, 0.0), n_all((size_t)n_cams_, 0.0);
    for (int c = 0; c < n_cams_; ++c) {
      const CamCalib &kc = calib_.cams[(size_t)c];
      const double cx = kc.cam(2), cy = kc.cam(3);
      const double r_max = std::hypot(std::max(cx, kc.img_w - cx), std::max(cy, kc.img_h - cy));
      double nq[4] = {0, 0, 0, 0};
      for (const WindowData &w : *w_b)
        for (const auto &obs : w.obs)
          for (const auto &o : obs) {
            if (o.cam != c)
              continue;
            n_all[(size_t)c] += 1.0;
            if (std::hypot(o.uv(0) - cx, o.uv(1) - cy) > 0.7 * r_max)
              n_far[(size_t)c] += 1.0;
            nq[(o.uv(0) >= cx ? 1 : 0) + (o.uv(1) >= cy ? 2 : 0)] += 1.0;
          }
      const bool k34_free = (n_all[(size_t)c] > 0.0) && (n_far[(size_t)c] / n_all[(size_t)c] >= cfg_.k34_radial_gate);
      if (k34_free) {
        jc.cam_prior_vec[(size_t)c](6) = jc.cam_prior_vec[(size_t)c](4); // open k3/k4 at the k1/k2 scale
        jc.cam_prior_vec[(size_t)c](7) = jc.cam_prior_vec[(size_t)c](5);
      }
      // C1 (quadrant-coverage center gate): cx/cy separate from distortion only when the data
      // BRACKETS the center -- with a quadrant starved, the center walks into a self-consistent
      // basin and commits (measured: cy +2.9 px shipped). Freeze cx/cy at seed through the
      // same prior-sigma mechanism as k3/k4; the frozen-dof exclusion keeps the cam block
      // committable with the frozen pair shipping seed values.
      const double minq = (n_all[(size_t)c] > 0.0)
                              ? std::min(std::min(nq[0], nq[1]), std::min(nq[2], nq[3])) / n_all[(size_t)c]
                              : 0.0;
      const bool center_free = minq >= cfg_.cam_center_quadrant_gate;
      if (!center_free && cfg_.cam_mode > 0) {
        jc.cam_prior_vec[(size_t)c](2) = 1e-9;
        jc.cam_prior_vec[(size_t)c](3) = 1e-9;
      }
      if (cfg_.verbose)
        std::printf("[session] phase B cam %d: cam_mode=%d, radial coverage %.1f%% -> k3/k4 %s; quadrant min %.1f%% -> cx/cy %s\n",
                    c, cfg_.cam_mode, 100.0 * n_far[(size_t)c] / std::max(n_all[(size_t)c], 1.0), k34_free ? "FREE" : "frozen",
                    100.0 * minq, center_free ? "FREE" : "frozen (center not bracketed)");
    }
    // the A-stage accel staging governs the joint polish too (gate-closed da
    // off-diagonals must not silently reopen inside the full vector)
    jc.use_da_prior_vec = jc_imu.use_da_prior_vec && !rep_.a_full_open;
    jc.da_prior_vec = jc_imu.da_prior_vec;
    SharedCalib calib_phaseA = calib_;
    JointReport repB;
    for (CamCalib &kc : calib_.cams)
      kc.cam_mode = cfg_.cam_mode;
    // ONE prior budget for the whole phase: every cam sub-solve (alternation
    // half-steps, settle, joint polish) anchors its free-dof priors at the
    // phase-A entry instead of re-centering per solve (see JointConfig).
    jc.use_cam_prior_center = true;
    jc.cam_prior_center.clear();
    for (const CamCalib &kc : calib_phaseA.cams)
      jc.cam_prior_center.push_back(kc.cam);
    // B-1: cam-only block passes (everything else frozen). The 8-dof camera
    // subspace is weakly conditioned against depth/td inside the full vector;
    // solving it alone first lets it take full-information steps before the
    // joint polish redistributes the residue. Inside the block, the pinhole row
    // and the radial polynomial correlate hard (|rho(fx,k1)| ~ 0.9 on
    // equidistant), so each alternation round solves (a) pinhole+k1/k2 with
    // k3/k4 frozen, then (b) distortion-only with the pinhole row frozen --
    // block-coordinate descent inside the camera block.
    {
      const bool f_dw = calib_.imu.calib_dw, f_da = calib_.imu.calib_da, f_qa = calib_.imu.calib_RAtoI;
      std::vector<char> f_ext, f_td;
      for (const CamCalib &kc : calib_.cams) {
        f_ext.push_back(kc.free_ext ? 1 : 0);
        f_td.push_back(kc.free_td ? 1 : 0);
      }
      calib_.imu.calib_dw = calib_.imu.calib_da = calib_.imu.calib_RAtoI = false;
      for (CamCalib &kc : calib_.cams)
        kc.free_ext = kc.free_td = false;
      if (cfg_.cam_alt_rounds > 0) {
        for (int round = 0; round < cfg_.cam_alt_rounds; ++round) {
          JointConfig ja = jc; // (a) pinhole + k1/k2; k3/k4 frozen this half-step
          for (auto &v : ja.cam_prior_vec)
            v(6) = v(7) = 1e-9;
          JointReport repBa;
          JointCalib::solve(*w_b, calib_, arm_budget(ja), repBa, &carry, &store_); // best-effort; the joint pass below is the arbiter
          note_stage_("B1a-pin-r" + std::to_string(round + 1), repBa);
          JointConfig jb = jc; // (b) distortion only; pinhole row frozen (k3/k4 per coverage gate)
          for (auto &v : jb.cam_prior_vec)
            v.head<4>().setConstant(1e-9);
          JointReport repBb;
          JointCalib::solve(*w_b, calib_, arm_budget(jb), repBb, &carry, &store_);
          note_stage_("B1b-dist-r" + std::to_string(round + 1), repBb);
          if (cfg_.verbose)
            for (int c = 0; c < n_cams_; ++c) {
              const Eigen::Matrix<double, 8, 1> &k = calib_.cams[(size_t)c].cam;
              std::printf("[session] phase B-1 alt round %d cam %d: [%.2f %.2f %.2f %.2f | %.5f %.5f %.5f %.5f]\n", round + 1, c,
                          k(0), k(1), k(2), k(3), k(4), k(5), k(6), k(7));
            }
        }
        // Settle pass: the rounds end distortion-last, which can leave the
        // pinhole row a half-oscillation from its conditional optimum (the
        // block-coordinate overshoot). One pinhole-only solve (all k frozen)
        // settles the row that carries the absolute acceptance gates before
        // the joint polish arbitrates. Skippable per profile (cam_settle):
        // measured merit-flat alongside round 2 on the host shape -- a
        // profile's own byte-level A/B decides whether it earns its ~9 s.
        if (cfg_.cam_settle) {
          JointConfig jp = jc;
          for (auto &v : jp.cam_prior_vec)
            v.tail<4>().setConstant(1e-9);
          JointReport repBp;
          JointCalib::solve(*w_b, calib_, arm_budget(jp), repBp, &carry, &store_);
          note_stage_("B1-settle", repBp);
        }
      } else {
        JointReport repB1; // monolithic fallback (alternation disabled)
        JointCalib::solve(*w_b, calib_, arm_budget(jc), repB1, &carry, &store_);
        note_stage_("B1-mono", repB1);
      }
      calib_.imu.calib_dw = f_dw;
      calib_.imu.calib_da = f_da;
      calib_.imu.calib_RAtoI = f_qa;
      for (int c = 0; c < n_cams_; ++c) {
        calib_.cams[(size_t)c].free_ext = (f_ext[(size_t)c] != 0);
        calib_.cams[(size_t)c].free_td = (f_td[(size_t)c] != 0);
      }
    }
    // B-2: joint polish with the camera block open
    JointConfig jb2 = jc;
    jb2.cert_open_imu = cfg_.b2_cert; // qn-policing replaces plateau/anchor (profile-gated A/B)
    const bool okB = JointCalib::solve(*w_b, calib_, arm_budget(jb2), repB, &carry, &store_);
    note_stage_("B2-polish", repB);
    // refinement-hurt detector: accept only if no cam dof of ANY camera moved > 3 prior-sigma.
    // Rig-wide, because a broken intrinsic aliases into ext/td and poisons the shared trajectory --
    // one camera going bad is enough to make the whole phase untrustworthy.
    bool cam_sane = okB;
    if (okB)
      for (int c = 0; c < n_cams_ && cam_sane; ++c)
        for (int k = 0; k < 8; ++k)
          if (std::abs(calib_.cams[(size_t)c].cam(k) - calib_phaseA.cams[(size_t)c].cam(k)) >
              3.0 * jc.cam_prior_vec[(size_t)c](k))
            cam_sane = false;
    if (!cam_sane) {
      calib_ = calib_phaseA; // revert the whole phase (cam aliases into ext/td when it breaks)
      if (cfg_.verbose)
        std::printf("[session] phase B REVERTED (refinement-hurt detector)\n");
    } else {
      rep_.joint = repB; // posterior of the full staged solve
    }
  }

  rep_.solved = calib_;
  rep_.t_solve_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_solve0).count();

  // ---- COMMIT decisions (precision ratio + absolute ceiling, frozen dofs
  // excluded), THEN the solved point, the resulting mixture, and the
  // leave-one-out variants all face the SAME held-out windows below ----
  const auto t_verify0 = std::chrono::steady_clock::now();
  enter_(RunnerState::VERIFY, "held-out reprojection check");
  SharedCalib out = calib_;
  // The revert point in the SAME layout as `out`: the values a block ships if it
  // does NOT commit. Flags are copied from `out` (a closed accel gate can freeze
  // calib_RAtoI mid-solve, so calib_postboot's own free-set may differ) -- only
  // the VALUES come from the seed.
  SharedCalib ref = calib_postboot;
  ref.imu.calib_dw = out.imu.calib_dw;
  ref.imu.calib_da = out.imu.calib_da;
  ref.imu.calib_RAtoI = out.imu.calib_RAtoI;
  ref.imu.calib_tg = out.imu.calib_tg; // layout lockstep: a mismatched flag silently shifts the block walk (measured: NaN moved-sigma on q_ItoC when a gate-closed session walked against a tg-on ref)
  // Per-camera flags too: free_blocks() is walked in lockstep across `out` and `ref` below, so the
  // two must emit the SAME layout. A single mismatched flag would silently shift the index
  // alignment and revert one block to another block's seed.
  for (int c = 0; c < n_cams_; ++c) {
    ref.cams[(size_t)c].free_ext = out.cams[(size_t)c].free_ext;
    ref.cams[(size_t)c].free_td = out.cams[(size_t)c].free_td;
    ref.cams[(size_t)c].cam_mode = out.cams[(size_t)c].cam_mode;
  }
  const double frozen_eps = 1e-8; // prior sigma at/below this = information-frozen dof
  {
    auto layout = out.free_blocks();
    auto layout_ref = ref.free_blocks();
    int off = 0;
    rep_.blocks.clear();
    for (size_t bi = 0; bi < layout.size(); ++bi) {
      auto &b = layout[bi];
      BlockCommit bc;
      bc.name = b.name;
      bc.cam = b.cam;
      // Movement off the seed, in LOCAL (tangent) coordinates -- the same units
      // the posterior sigma lives in, so |delta|/sigma is dimensionless.
      Eigen::VectorXd d = Eigen::VectorXd::Zero(b.lsize);
      if (b.is_quat) {
        const Eigen::Vector4d qs(b.ptr[0], b.ptr[1], b.ptr[2], b.ptr[3]);
        const double *pr = layout_ref[bi].ptr;
        const Eigen::Vector4d qr(pr[0], pr[1], pr[2], pr[3]);
        d = 2.0 * ov_core::quat_multiply(qs, ov_core::Inv(qr)).head<3>(); // small-angle JPL local
      } else {
        for (int k = 0; k < b.lsize; ++k)
          d(k) = b.ptr[k] - layout_ref[bi].ptr[k];
      }
      for (int k = 0; k < b.lsize; ++k) {
        const double sprior = rep_.joint.prior_sigma_vec(off + k);
        const double spost = rep_.joint.sigma(off + k);
        // An information-frozen dof has sigma_post ~= sigma_prior by
        // construction (data info ~0 vs 1/eps^2 prior): counting it pins
        // worst_ratio at ~commit_sigma_factor and the block can NEVER commit --
        // the k3/k4 freeze-vs-commit deadlock. Frozen dofs ship their seed
        // value inside a committed block by design.
        if (sprior > frozen_eps) {
          bc.worst_ratio = std::max(bc.worst_ratio, cfg_.commit_sigma_factor * spost / sprior);
          bc.worst_sigma = std::max(bc.worst_sigma, spost);
          bc.moved_sigma = std::max(bc.moved_sigma, std::abs(d(k)) / std::max(spost, 1e-300));
        }
        // (k continues; off advances after the block)
      }
      const auto ceil_it = cfg_.commit_abs_ceiling.find(b.name);
      bc.ceiling_ok = (ceil_it == cfg_.commit_abs_ceiling.end()) || (bc.worst_sigma <= ceil_it->second);
      // A block that the solve never moved off its seed has NOT been estimated --
      // its tight posterior only says "the data would have constrained it well",
      // not "the data determined it". Claiming it would ship the seed as a
      // calibration (and a writeback would then overwrite the system-of-record's
      // real value with it). Refuse, loudly.
      bc.not_estimated = (bc.moved_sigma < cfg_.commit_min_move_sigma);
      bc.committed = (bc.worst_ratio < 1.0) && bc.ceiling_ok && !bc.not_estimated;
      rep_.blocks.push_back(bc);
      off += b.lsize;
    }
  }
  // Atomic pair: q_ItoC and td move jointly in the solve (omega-coupled), so a
  // mixed state (new rotation, seed td) can be worse than either endpoint --
  // they commit together or revert together.
  // The pair is PER CAMERA: cam 0's rotation is coupled to cam 0's time offset, and to nothing on
  // cam 1 -- they are different physical quantities that happen to share a name.
  if (cfg_.commit_atomic_rot_td) {
    for (int c = 0; c < n_cams_; ++c) {
      BlockCommit *bq = nullptr, *bt = nullptr;
      for (auto &bc : rep_.blocks) {
        if (bc.cam != c)
          continue;
        if (bc.name == "q_ItoC")
          bq = &bc;
        if (bc.name == "td")
          bt = &bc;
      }
      if (bq && bt && bq->committed != bt->committed) {
        (bq->committed ? bq : bt)->atomic_reverted = true;
        bq->committed = bt->committed = false;
      }
    }
  }
  // Atomic pair: when the full accel chain was unlocked (A1b), da and q_AtoI
  // were estimated JOINTLY and are strongly coupled (the off-diagonals and the
  // accel-frame rotation trade against each other) -- shipping one with the
  // seed value of the other is a mixed state neither solve nor split-half
  // validated. When the chain is frozen, q_AtoI has no block and da(diag)
  // commits alone against its seed-consistent frozen complement.
  if (cfg_.commit_atomic_accel && rep_.a_full_open) {
    BlockCommit *ba = nullptr, *bq = nullptr;
    for (auto &bc : rep_.blocks) {
      if (bc.name == "da")
        ba = &bc;
      if (bc.name == "q_AtoI")
        bq = &bc;
    }
    if (ba && bq && ba->committed != bq->committed) {
      (ba->committed ? ba : bq)->atomic_reverted = true;
      ba->committed = bq->committed = false;
    }
  }
  auto revert_block = [](SharedCalib &dst, const SharedCalib &src, const std::string &name, int cam) {
    if (name == "dw")
      dst.imu.dw = src.imu.dw;
    else if (name == "da")
      dst.imu.da = src.imu.da;
    else if (name == "q_AtoI")
      dst.imu.q_AtoI = src.imu.q_AtoI;
    else if (name == "tg") // shared block (cam = -1): a refused tg must revert HERE, not fall
      dst.imu.Tg = src.imu.Tg; // through the per-camera guard below and ship its solved value
    else if (cam < 0 || (size_t)cam >= dst.cams.size())
      return; // a per-camera block with no camera is a layout bug, not a revert
    else if (name == "q_ItoC")
      dst.cams[(size_t)cam].q_ItoC = src.cams[(size_t)cam].q_ItoC;
    else if (name == "p_IinC")
      dst.cams[(size_t)cam].p_IinC = src.cams[(size_t)cam].p_IinC;
    else if (name == "td")
      dst.cams[(size_t)cam].td = src.cams[(size_t)cam].td;
    else if (name == "cam")
      dst.cams[(size_t)cam].cam = src.cams[(size_t)cam].cam;
  };
  bool any_reverted = false;
  for (auto &bc : rep_.blocks)
    if (!bc.committed) {
      revert_block(out, calib_postboot, bc.name, bc.cam);
      any_reverted = true;
    }

  // ---- unified holdout sweep ----
  // Candidates: [0] post-bootstrap seed, [1] fully-solved point, [2] committed
  // MIXTURE when blocks reverted (the mixture is what actually ships -- commit
  // decisions couple blocks, so it must beat the seed itself), then
  // leave-one-out variants per committed block (accuracy attribution). A
  // window where ANY candidate fails to re-seed/solve is dropped for ALL
  // candidates: one-sided failures previously inflated only that candidate's
  // cost (asymmetric walling -> spurious aborts).
  std::vector<SharedCalib> cand = {calib_postboot, calib_};
  const int i_mix = any_reverted ? 2 : 1;
  if (any_reverted)
    cand.push_back(out);
  std::vector<int> lobo_block; // rep_.blocks index per leave-one-out candidate
  const int lobo0 = (int)cand.size();
  if (cfg_.commit_attribution)
    for (size_t bi = 0; bi < rep_.blocks.size(); ++bi)
      if (rep_.blocks[bi].committed) {
        SharedCalib c = out;
        revert_block(c, calib_postboot, rep_.blocks[bi].name, rep_.blocks[bi].cam);
        cand.push_back(c);
        lobo_block.push_back((int)bi);
      }
  std::vector<double> csum(cand.size(), 0.0);
  int n_hold = 0, n_drop = 0;
  StageEvidence vev; // verify-sweep aggregate (seed+BA per holdout x candidate)
  vev.label = "verify-sweep";
  // The (holdout x candidate) evaluations are independent: each takes its own
  // WindowData + SharedCalib copies. Parallelized over the flattened job list
  // (the sweep was the last serial 8-9 s of the session); per-slot symmetric
  // drop + fixed-order accumulation happen serially after the pool joins, so
  // committed outputs are order-invariant. The preint store is NOT used here:
  // candidates sit at different calibrations, so the single-entry cache would
  // thrash serially and RACE in parallel (same uid, concurrent workers).
  std::vector<int> hslots;
  for (int slot = 0; slot < (int)slots_.size() && scorer_; ++slot)
    if (slot < scorer_->size() && scorer_->is_holdout(slot) && !slots_[slot].clone_times.empty())
      hslots.push_back(slot);
  const size_t C = cand.size();
  struct VJob {
    bool ok = false;
    double cost = 0.0, seed_s = 0.0;
    WindowSolveReport r;
  };
  std::vector<VJob> jobs(hslots.size() * C);
  {
    const int nthreads = std::max(1, std::min(cfg_.joint.num_threads, (int)jobs.size()));
    ov_init::zbft_sfm::ParallelExecutor vpool(nthreads);
    vpool.parallel_ranges((int)jobs.size(), [&](int, int b0, int b1) {
      for (int ji = b0; ji < b1; ++ji) {
        const int slot = hslots[ji / C];
        const size_t ci = ji % C;
        VJob &J = jobs[ji];
        // re-seed at EACH candidate calibration: harvest-time seeds are
        // p-stamped and would wall the evaluation against any calibration but
        // their own. On a seed GATE failure fall back to the harvest seeds:
        // the gate protects ADMISSION, not evaluation.
        SharedCalib cc = cand[ci];
        WindowData w = slots_[slot];
        LinearSeedReport sr;
        const auto t_vs0 = std::chrono::steady_clock::now();
        const bool seeded = LinearSeed::seed_window(w, cc, w.seed_bg, sr, cfg_.seed);
        J.seed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_vs0).count();
        J.ok = (seeded || w.has_seeds) &&
               WindowBA::solve_and_export(w, cc, false, J.r, cfg_.joint.window_max_iters, false, nullptr, nullptr);
        J.cost = J.r.cost_final;
      }
    });
  }
  for (size_t si = 0; si < hslots.size(); ++si) {
    bool all_ok = true;
    for (size_t ci = 0; ci < C; ++ci) {
      const VJob &J = jobs[si * C + ci];
      vev.passes++;
      vev.seed_s += J.seed_s;
      vev.iters += J.r.iterations;
      vev.preint_s += J.r.t_preint;
      vev.inner_s += J.r.t_inner;
      vev.export_s += J.r.t_export;
      vev.factor_s += J.r.t_factor;
      (J.r.preint_hit ? vev.phit : vev.pmiss)++;
      vev.tstop += J.r.time_stopped ? 1 : 0;
      if (J.ok)
        vev.accepted++;
      else
        all_ok = false;
    }
    if (!all_ok) {
      ++n_drop;
      continue;
    }
    for (size_t ci = 0; ci < C; ++ci)
      csum[ci] += jobs[si * C + ci].cost;
    // per-window paired improvement of the shipped mixture (small-n evidence:
    // one aggregate ratio hides a single window carrying the whole verdict)
    const double c0 = jobs[si * C + 0].cost;
    rep_.verify_window_ratio.push_back((c0 > 0.0) ? 1.0 - jobs[si * C + i_mix].cost / c0 : 0.0);
    ++n_hold;
  }
  rep_.verify_windows_used = n_hold;
  rep_.verify_windows_dropped = n_drop;
  rep_.t_verify_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_verify0).count();
  vev.wall_s = rep_.t_verify_s;
  vev.windows = n_hold;
  vev.rss_kb = vmrss_kb_();
  rep_.evidence.push_back(std::move(vev));
  if (n_hold == 0) {
    // silence here is not success: with zero evaluable holdout windows NOTHING
    // was verified -- refuse instead of committing blind
    rep_.committed = calib_postboot;
    enter_(RunnerState::ABORT, "VERIFY impossible: no held-out window evaluable");
    return;
  }
  rep_.holdout_cost_seed = csum[0];
  rep_.holdout_cost_committed = csum[1];
  rep_.verify_improve = (csum[0] > 0.0) ? 1.0 - csum[1] / csum[0] : -1.0;
  rep_.holdout_cost_mixture = csum[i_mix];
  rep_.mixture_improve = (csum[0] > 0.0) ? 1.0 - csum[i_mix] / csum[0] : -1.0;
  // Small-n honesty floors: a +5% aggregate on ONE window is weak evidence --
  // the required improvement rises as n_hold shrinks (the profile answer to a
  // starved session is MORE holdouts via min_holdout, not a softer gate).
  const double req_improve =
      (n_hold <= 1) ? cfg_.verify_min_improve_n1 : (n_hold == 2) ? cfg_.verify_min_improve_n2 : cfg_.verify_min_improve;
  rep_.verify_small_n = (n_hold <= 2);
  if (cfg_.verbose) {
    std::printf("[session] VERIFY: holdout %.4e -> solved %.4e (%.1f%%), mixture %.4e (%.1f%%), %d windows (%d dropped)%s\n", csum[0],
                csum[1], 100.0 * rep_.verify_improve, csum[i_mix], 100.0 * rep_.mixture_improve, n_hold, n_drop,
                rep_.verify_small_n ? " [small-n floors]" : "");
    if (rep_.verify_small_n) {
      std::printf("[session] VERIFY small-n: floor %.0f%%, per-window mixture ratios:", 100.0 * req_improve);
      for (double r : rep_.verify_window_ratio)
        std::printf(" %+.1f%%", 100.0 * r);
      std::printf("\n");
    }
  }
  if (rep_.verify_improve < req_improve) {
    rep_.committed = calib_postboot;
    enter_(RunnerState::ABORT, "VERIFY failed: solved calibration does not beat the seed on held-out windows");
    return;
  }
  if (any_reverted && rep_.mixture_improve < req_improve) {
    rep_.committed = calib_postboot;
    enter_(RunnerState::ABORT, "VERIFY failed: partially-committed mixture does not beat the seed on held-out windows");
    return;
  }
  for (size_t li = 0; li < lobo_block.size(); ++li)
    rep_.blocks[lobo_block[li]].holdout_delta = csum[lobo0 + li] - csum[i_mix];

  // ---- COMMIT: ship the verified mixture ----
  enter_(RunnerState::COMMIT, "gated partial commit");
  rep_.committed = out;
  if (!cfg_.out_yaml.empty()) {
    // Provenance travels WITH the values: the file carries a complete
    // calibration, but only these blocks were estimated by this session. Every
    // other field is a seed passenger and must not reach the system-of-record.
    std::vector<std::string> yes, no;
    for (const auto &bc : rep_.blocks)
      (bc.committed ? yes : no).push_back(bc.label());
    // Blocks with no free parameters at all (frozen IMU chain, cam_mode 0) never appear in the
    // layout -- they are seed passengers too, and a writeback must know that. The IMU chain is
    // asked once; the camera blocks are asked once PER CAMERA, because a block can be free on one
    // camera and absent on another. tr is ALWAYS in this list: the readout is HAL3 hardware
    // truth, never estimated, so camN_t_readout ships as a passenger for every camera.
    for (const char *nm : {"dw", "da", "q_AtoI", "tg"}) {
      bool present = false;
      for (const auto &bc : rep_.blocks)
        present = present || (bc.cam < 0 && bc.name == nm);
      if (!present)
        no.push_back(nm);
    }
    for (int c = 0; c < n_cams_; ++c)
      for (const char *nm : {"q_ItoC", "p_IinC", "td", "tr", "cam"}) {
        bool present = false;
        for (const auto &bc : rep_.blocks)
          present = present || (bc.cam == c && bc.name == nm);
        if (!present)
          no.push_back(std::string(nm) + "@" + std::to_string(c));
      }
    // rollback copy, then atomic write
    FILE *prev = std::fopen(cfg_.out_yaml.c_str(), "rb");
    if (prev) {
      std::fclose(prev);
      std::rename(cfg_.out_yaml.c_str(), (cfg_.out_yaml + ".rollback").c_str());
    }
    write_calib_yaml(cfg_.out_yaml, rep_.committed, rep_.mean_exposure_s, &yes, &no);
  }
  if (cfg_.verbose) {
    std::printf("[session] COMMIT:");
    for (auto &bc : rep_.blocks)
      std::printf(" %s=%s(r%.2f s%.2e m%.1f%s%s%s)", bc.label().c_str(), bc.committed ? "yes" : "SEED", bc.worst_ratio, bc.worst_sigma,
                  bc.moved_sigma, bc.ceiling_ok ? "" : " CEIL", bc.atomic_reverted ? " ATOMIC" : "",
                  bc.not_estimated ? " NOT-ESTIMATED" : "");
    std::printf("\n");
    if (cfg_.commit_attribution) {
      std::printf("[session] holdout leave-one-out deltas:");
      for (auto &bc : rep_.blocks)
        if (bc.committed)
          std::printf(" %s=%+.3e", bc.label().c_str(), bc.holdout_delta);
      std::printf("\n");
    }
    std::printf("[session] timings: solve %.2fs verify %.2fs\n", rep_.t_solve_s, rep_.t_verify_s);
  }
  enter_(RunnerState::DONE, "session complete");
}

bool CalibSessionRunner::run_replay(const std::string &record_path, const SessionConfig &cfg, SessionReport &out,
                                    const SeedOverride *seed_override) {
  SessionRecordReader rd;
  if (!rd.open(record_path))
    return false;
  // The recorded seed is authoritative for the CAMERA and the streams; the rest
  // may be patched so ONE recorded session can be scored across the seeding
  // matrix (blind vs chain-seeded, per block).
  SessionSeed seed = rd.seed();
  if (seed_override) {
    if (seed_override->imu)
      seed.calib.imu = *seed_override->imu;
    if (seed_override->Tg)
      seed.calib.imu.Tg = *seed_override->Tg; // after `imu`: Tg is its own axis, never a side effect
    if (seed_override->noise)
      seed.calib.noise = *seed_override->noise;
    const size_t c = (size_t)seed_override->cam;
    if (c < seed.calib.cams.size()) {
      if (seed_override->q_ItoC)
        seed.calib.cams[c].q_ItoC = *seed_override->q_ItoC;
      if (seed_override->p_IinC)
        seed.calib.cams[c].p_IinC = *seed_override->p_IinC;
      if (seed_override->td)
        seed.calib.cams[c].td = *seed_override->td;
    }
  }
  CalibSessionRunner runner(cfg, seed);
  bool is_imu = false;
  RawImu s;
  FrameObs f;
  while (rd.next(is_imu, s, f)) {
    if (is_imu)
      runner.feed_imu(s);
    else
      runner.feed_frame(f);
  }
  out = runner.finish();
  return true;
}
