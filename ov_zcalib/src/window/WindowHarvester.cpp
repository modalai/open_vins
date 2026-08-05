/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: streaming window harvester implementation (see WindowHarvester.h).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "WindowHarvester.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "CamUndistort.h"

using namespace ov_zcalib;

WindowHarvester::WindowHarvester(const HarvesterConfig &cfg, const SharedCalib &seed) : cfg_(cfg), seed_(seed) {
  imu_hist_span_ = cfg_.max_window_s + 2.0 * cfg_.imu_pad_s + 2.0; // window + pads + slack
  // Seed each camera's period from its DECLARED rate, so the first window of a session accounts
  // drops against the truth rather than against whatever the first two frames happened to show.
  // The EMA still tracks the real cadence from there.
  nominal_seed_.assign((size_t)seed_.n_cams(), 0.0);
  for (int c = 0; c < seed_.n_cams(); ++c)
    if (seed_.cams[(size_t)c].fps > 0.0)
      nominal_seed_[(size_t)c] = 1.0 / seed_.cams[(size_t)c].fps;
  nominal_dt_ = nominal_seed_;
  last_ts_.assign((size_t)seed_.n_cams(), -1.0);
  drops_in_window_.assign((size_t)seed_.n_cams(), 0);
  evicted_.assign((size_t)seed_.n_cams(), 0);
}

void WindowHarvester::push_imu(const RawImu &s) {
  imu_hist_.push_back(s);
  while (!imu_hist_.empty() && imu_hist_.front().timestamp < s.timestamp - imu_hist_span_)
    imu_hist_.pop_front();
  // rolling excitation sums
  excite_ring_.push_back(s);
  sw_ += s.wm;
  sw2_ += s.wm.cwiseProduct(s.wm);
  sa_ += s.am;
  sa2_ += s.am.cwiseProduct(s.am);
  while (!excite_ring_.empty() && excite_ring_.front().timestamp < s.timestamp - cfg_.excite_win_s) {
    const RawImu &o = excite_ring_.front();
    sw_ -= o.wm;
    sw2_ -= o.wm.cwiseProduct(o.wm);
    sa_ -= o.am;
    sa2_ -= o.am.cwiseProduct(o.am);
    excite_ring_.pop_front();
  }
}

void WindowHarvester::excitation(double &w_std, double &a_std) const {
  const double n = (double)excite_ring_.size();
  if (n < 8) {
    w_std = a_std = 0.0;
    return;
  }
  const Eigen::Vector3d vw = (sw2_ / n - (sw_ / n).cwiseProduct(sw_ / n)).cwiseMax(0.0);
  const Eigen::Vector3d va = (sa2_ / n - (sa_ / n).cwiseProduct(sa_ / n)).cwiseMax(0.0);
  w_std = std::sqrt(vw.sum());
  a_std = std::sqrt(va.sum());
}

bool WindowHarvester::push_frame(const FrameObs &f) {
  double w_std, a_std;
  excitation(w_std, a_std);
  const bool excited = (w_std >= cfg_.min_excite_w) && (a_std >= cfg_.min_excite_a);
  const bool tracking_ok = (int)f.pts.size() >= cfg_.min_feats;
  const size_t c = (size_t)f.cam;

  auto open_window = [&]() {
    open_ = true;
    frames_.clear();
    frames_.push_back(f);
    quiet_since_ = -1.0;
    nominal_dt_ = nominal_seed_; // back to the declared rates, not to "unknown"
    std::fill(last_ts_.begin(), last_ts_.end(), -1.0);
    std::fill(drops_in_window_.begin(), drops_in_window_.end(), 0);
    std::fill(evicted_.begin(), evicted_.end(), 0);
    if (c < last_ts_.size())
      last_ts_[c] = f.timestamp;
  };

  if (!open_) {
    if (excited && tracking_ok)
      open_window();
    return false;
  }

  // ---- open window: gap/drop accounting, PER CAMERA ----
  // Cadence is a property of a camera, not of the merged stream: a 60 Hz and a 30 Hz camera
  // interleave into inter-frame gaps that belong to neither of them, so a global nominal dt would
  // both mis-estimate the drop count and fire the split rule on healthy data. Each camera's gap is
  // measured against its OWN last frame and its own learned period.
  const double dt = (c < last_ts_.size() && last_ts_[c] >= 0.0) ? f.timestamp - last_ts_[c] : 0.0;
  int missing = 0;
  if (dt > 0.0 && c < nominal_dt_.size()) {
    if (nominal_dt_[c] <= 0.0)
      nominal_dt_[c] = dt;
    else if (dt < 2.5 * nominal_dt_[c])
      nominal_dt_[c] = 0.9 * nominal_dt_[c] + 0.1 * dt;
    missing = (nominal_dt_[c] > 0.0) ? std::max(0, (int)std::lround(dt / nominal_dt_[c]) - 1) : 0;
  }

  // A window splits on a discontinuity of the STREAM, not of one camera. Free-running cameras cover
  // for each other: while a saturated 60 Hz stream sheds frames the 30 Hz one keeps delivering, so
  // the trajectory is still observed and the window is still coherent. The gapping camera is simply
  // shedding frames -- which is charged to IT below, and evicts it if it sheds too many. Only a gap
  // in which NOTHING arrived, from ANY camera, is a real discontinuity.
  //
  // Splitting on one camera's gap is what turned the D0014 ingest overrun into a CALIBRATION bug:
  // 1522 shed hires frames fragmented the session into sub-min_window_s pieces, and the down camera
  // -- which had dropped nothing -- lost them all. On a one-camera rig dt_any == dt, so the rule
  // below is the old rule by construction, and every single-camera reference is untouched.
  const double dt_any = f.timestamp - frames_.back().timestamp;
  const bool sole_cam = (nominal_dt_.size() <= 1);
  const bool split = sole_cam ? (dt > cfg_.max_frame_gap_s || missing > cfg_.max_consec_drops)
                              : (dt_any > cfg_.max_frame_gap_s);
  if (split) {
    // close what we have, then treat this frame as a fresh opener
    close_window_(true);
    if (excited && tracking_ok)
      open_window();
    return have_out_;
  }
  if (c < drops_in_window_.size())
    drops_in_window_[c] += missing; // charged to the camera that dropped them, not to the rig

  if (c < last_ts_.size())
    last_ts_[c] = f.timestamp;
  frames_.push_back(f);
  const double dur = f.timestamp - frames_.front().timestamp;

  // quiet tracking for the close rule
  if (excited)
    quiet_since_ = -1.0;
  else if (quiet_since_ < 0.0)
    quiet_since_ = f.timestamp;

  const bool too_quiet = (quiet_since_ >= 0.0) && (f.timestamp - quiet_since_ >= cfg_.quiet_close_s);
  if (dur >= cfg_.max_window_s || too_quiet) {
    close_window_(true);
    return have_out_;
  }
  return false;
}

bool WindowHarvester::flush() {
  if (!open_)
    return false;
  close_window_(true);
  return have_out_;
}

void WindowHarvester::close_window_(bool valid) {
  open_ = false;
  if (!valid || frames_.size() < 2) {
    frames_.clear();
    return;
  }
  const double dur = frames_.back().timestamp - frames_.front().timestamp;
  // Each camera is answerable for ITS OWN frames: it is expected to deliver dur/nominal_dt of them,
  // and it is judged on the ones IT dropped. A camera that fails the gate is EVICTED from this
  // window; the window still stands on the cameras that are healthy, and dies only if none are.
  // (Pooling both into one rig-level fraction let a saturated 60 Hz stream invalidate windows in
  // which the 30 Hz camera's data was perfect -- see max_drop_frac.)
  double worst_frac = 0.0;
  int n_ok = 0;
  for (size_t c = 0; c < nominal_dt_.size(); ++c) {
    if (nominal_dt_[c] <= 0.0 || last_ts_[c] < 0.0)
      continue; // camera not present in this window at all
    const double exp_c = dur / nominal_dt_[c];
    const double frac_c = (exp_c > 1.0) ? (double)drops_in_window_[c] / exp_c : 0.0;
    evicted_[c] = (frac_c > cfg_.max_drop_frac) ? 1 : 0;
    if (!evicted_[c])
      ++n_ok;
    worst_frac = std::max(worst_frac, frac_c);
  }
  if (dur < cfg_.min_window_s || n_ok == 0) {
    ++invalidated_;
    frames_.clear();
    return;
  }
  WindowData w;
  WindowMeta m;
  m.drop_frac = worst_frac;
  if (assemble_(w, m)) {
    out_win_ = std::move(w);
    out_meta_ = m;
    have_out_ = true;
  } else {
    ++invalidated_;
  }
  frames_.clear();
}

bool WindowHarvester::pop_window(WindowData &out, WindowMeta &meta) {
  if (!have_out_)
    return false;
  out = std::move(out_win_);
  meta = out_meta_;
  have_out_ = false;
  return true;
}

bool WindowHarvester::assemble_(WindowData &out, WindowMeta &meta) {
  const int n_cams = seed_.n_cams();

  // ---- clone subsampling: stride each camera's OWN frame list, never the merged stream ----
  //
  // A uniform stride over the interleaved multi-camera stream ALIASES, and catastrophically. With a
  // 60 Hz and a 30 Hz camera the merged frames run c0,c0,c1,c0,c0,c1,... on a period of 3, so any
  // stride that is a multiple of 3 -- e.g. the stride 12 that a 4 s window and a 32-clone budget
  // produce -- selects c1 frames and NOTHING ELSE, and camera 0 silently vanishes from the window.
  // Striding per camera cannot alias, and it guarantees every camera is represented, which is the
  // entire point of solving them jointly.
  std::vector<std::vector<int>> per_cam((size_t)n_cams);
  for (int i = 0; i < (int)frames_.size(); ++i) {
    const int c = (int)frames_[i].cam;
    // An EVICTED camera (too many of its own frames dropped) contributes nothing: no clones, no
    // observations. It does not get to spend the window's clone budget on a stream full of holes,
    // and it does not get to take the window down with it.
    if (c >= 0 && c < n_cams && !evicted_[(size_t)c])
      per_cam[(size_t)c].push_back(i);
  }
  int n_present = 0;
  for (const auto &v : per_cam)
    if (!v.empty())
      ++n_present;
  if (n_present == 0)
    return false;

  // The window Cholesky is CUBIC in clones, so the clone budget belongs to the WINDOW, not to a
  // camera: SPLIT it across the cameras present, do not multiply it by them. Each camera then
  // carries its own tracks on its own clones, and the solve cost is unchanged by adding a camera.
  const int budget = std::max(2, cfg_.max_clones / n_present);
  std::vector<int> sel;
  for (int c = 0; c < n_cams; ++c) {
    const int nf = (int)per_cam[(size_t)c].size();
    if (nf == 0)
      continue;
    const int stride = std::max(1, (nf + budget - 1) / budget);
    for (int i = 0; i < nf; i += stride)
      sel.push_back(per_cam[(size_t)c][i]);
  }
  if ((int)sel.size() < cfg_.min_clones)
    return false;

  // ---- clone timeline ----
  //
  // A clone IS a frame: this is a BATCH problem, so unlike the filter it has no reason to anchor
  // non-reference frames onto a shared epoch and transport them across the residual. Each frame
  // gets its own clone at its own mid-exposure instant, mapped to the IMU clock through ITS OWN
  // camera's seed td (the offsets are not equal across cameras -- that is what we are solving for).
  //
  // The one case that needs care: two cameras free-run, so their frames occasionally land a hair
  // apart. Left as two clones that would hand the preintegration a ~0 s interval, which is
  // degenerate. So clones closer than kMinCloneDt are MERGED, and each observation records the
  // residual offset from its own instant to the merged clone's stamp (CloneObs::dt_ref) -- which
  // the reprojection then transports across exactly.
  constexpr double kMinCloneDt = 2e-3; // well below any frame period on these rigs
  std::vector<std::pair<double, int>> stamped; // (seed-mapped IMU time, frames_ index)
  stamped.reserve(sel.size());
  for (int i : sel) {
    const FrameObs &fr = frames_[(size_t)i];
    stamped.emplace_back(fr.timestamp + 0.5 * (double)fr.exposure_s + seed_.cams[(size_t)fr.cam].td, i);
  }
  std::sort(stamped.begin(), stamped.end());
  std::vector<std::vector<int>> clone_frames; // frames_ indices grouped per clone
  std::vector<double> clone_t;
  for (const auto &st : stamped) {
    if (!clone_t.empty() && st.first - clone_t.back() < kMinCloneDt)
      clone_frames.back().push_back(st.second);
    else {
      clone_t.push_back(st.first);
      clone_frames.push_back({st.second});
    }
  }
  const size_t n_clones = clone_t.size();
  if ((int)n_clones < cfg_.min_clones)
    return false;

  // ---- track filtering on the SELECTED clones ----
  // Feature ids are globally unique across cameras (ov_core hands them out from one atomic
  // counter) and a track never crosses cameras, so one flat map is exactly right here.
  std::unordered_map<uint32_t, int> track_len;
  for (int i : sel)
    for (const FrameObsPoint &p : frames_[(size_t)i].pts)
      track_len[p.id]++;
  std::unordered_map<uint32_t, int> id_map;
  std::unordered_map<uint32_t, int> id_used;
  for (auto &kv : track_len)
    if (kv.second >= cfg_.min_track_len)
      id_map.emplace(kv.first, -1); // assigned below in first-seen order (deterministic)
  if ((int)id_map.size() < cfg_.min_feats)
    return false;

  // ---- assemble clones ----
  int next_id = 0;
  out.obs.resize(n_clones);
  out.clone_times = clone_t;
  double flow_sum = 0.0;
  int flow_n = 0;
  std::vector<double> rows;
  std::unordered_map<uint32_t, Eigen::Vector2d> last_uv;
  std::unordered_map<uint32_t, double> last_t;
  std::unordered_map<uint32_t, Eigen::Vector3d> first_bear, last_bear;
  for (size_t k = 0; k < n_clones; ++k) {
    for (int fi : clone_frames[k]) {
      const FrameObs &fr = frames_[(size_t)fi];
      const size_t c = (size_t)fr.cam;
      const CamCalib &kc = seed_.cams[c];
      const double t_obs = fr.timestamp + 0.5 * (double)fr.exposure_s + kc.td;
      for (const FrameObsPoint &p : fr.pts) {
        auto it = id_map.find(p.id);
        if (it == id_map.end())
          continue;
        if (id_used[p.id] >= cfg_.max_track_len)
          continue; // drift cap
        if (it->second < 0)
          it->second = next_id++;
        id_used[p.id]++;
        CloneObs o;
        o.feat_id = (size_t)it->second;
        o.uv = Eigen::Vector2d((double)p.u, (double)p.v);
        o.cam = (int)c;
        o.dt_ref = t_obs - clone_t[k]; // 0 unless this frame was merged onto another camera's clone
        o.u_frac = std::min(1.0, std::max(0.0, (double)p.v / (double)kc.img_h));
        o.bearing = CamUndistort::bearing(o.uv, kc.cam, kc.fisheye);
        out.obs[k].push_back(o);
        rows.push_back(o.u_frac);
        auto lu = last_uv.find(p.id);
        if (lu != last_uv.end() && clone_t[k] > last_t[p.id]) {
          flow_sum += (o.uv - lu->second).norm() / (clone_t[k] - last_t[p.id]);
          ++flow_n;
        }
        last_uv[p.id] = o.uv;
        last_t[p.id] = clone_t[k];
        if (!first_bear.count(p.id))
          first_bear[p.id] = o.bearing;
        last_bear[p.id] = o.bearing;
      }
    }
  }
  out.num_feats = (size_t)next_id;
  if ((int)out.num_feats < cfg_.min_feats)
    return false;
  out.pix_sigma = cfg_.pix_sigma;
  out.td_ref.resize((size_t)n_cams);
  out.tr_ref.resize((size_t)n_cams);
  for (int c = 0; c < n_cams; ++c) {
    out.td_ref[(size_t)c] = seed_.cams[(size_t)c].td;
    // tr_ref is STRUCTURALLY ZERO, and that is not a shortcut -- it is what the clone timeline says.
    //
    // A clone is stamped at (t_sof + exposure/2 + td_seed): a per-FRAME instant, with no per-ROW
    // term in it. The observation on row u happened at (t_sof + exposure/2 + u*tr + td). So the
    // shift the factor must transport is
    //
    //     Delta = (td - td_seed) + u*tr        -- ABSOLUTE in tr, a DEVIATION only in td
    //
    // Seeding tr_ref with the camera's tr made the factor apply (tr - tr_seed)*u instead, which is
    // ZERO whenever tr is held at its seed: the hires rolling shutter was then not modelled AT ALL,
    // and td silently absorbed ~tr/2 of it (defect D10). Note VIO gets this right --
    // UpdaterHelper.cpp applies the full v_frac * t_readout, gated on the VALUE, not on whether the
    // readout is being estimated -- so the calibrator was handing VIO a td fitted under a different
    // camera model than VIO consumes it with.
    //
    // Zeroing it makes the existing formula exactly correct and leaves the Jacobian untouched
    // (d/dtr of u*tr and of (tr - tr_ref)*u are both u). Byte-identical on any global-shutter
    // camera, where tr == 0 either way.
    out.tr_ref[(size_t)c] = 0.0;
  }

  // ---- padded IMU slice ----
  const double t_lo = out.clone_times.front() - cfg_.imu_pad_s;
  const double t_hi = out.clone_times.back() + cfg_.imu_pad_s;
  for (const RawImu &s : imu_hist_)
    if (s.timestamp >= t_lo && s.timestamp <= t_hi)
      out.imu.push_back(s);
  if (out.imu.size() < 2 || out.imu.front().timestamp > out.clone_times.front() ||
      out.imu.back().timestamp < out.clone_times.back())
    return false;

  // ---- fingerprint ----
  meta.t0 = frames_.front().timestamp;
  meta.t1 = frames_.back().timestamp;
  meta.clones = (int)n_clones;
  meta.feats = (int)out.num_feats;
  {
    Eigen::Vector3d mw = Eigen::Vector3d::Zero(), ma = Eigen::Vector3d::Zero();
    for (const RawImu &s : out.imu) {
      mw += s.wm;
      ma += s.am;
    }
    const double n = (double)out.imu.size();
    mw /= n;
    ma /= n;
    Eigen::Vector3d vw = Eigen::Vector3d::Zero(), va = Eigen::Vector3d::Zero();
    double tmin = 1e30, tmax = -1e30, tsum = 0.0;
    for (const RawImu &s : out.imu) {
      vw += (s.wm - mw).cwiseProduct(s.wm - mw);
      va += (s.am - ma).cwiseProduct(s.am - ma);
      tmin = std::min(tmin, s.temp_c);
      tmax = std::max(tmax, s.temp_c);
      tsum += s.temp_c;
    }
    vw /= n;
    va /= n;
    meta.temp_mean = tsum / n;
    meta.temp_span = tmax - tmin;
    // gravity sweep: max angle between 0.25 s-mean accel directions across the window
    double sweep = 0.0;
    {
      std::vector<Eigen::Vector3d> dirs;
      Eigen::Vector3d acc = Eigen::Vector3d::Zero();
      int cnt = 0;
      double t_bin = out.imu.front().timestamp;
      for (const RawImu &s : out.imu) {
        acc += s.am;
        ++cnt;
        if (s.timestamp - t_bin >= 0.25) {
          if (acc.norm() > 1e-9)
            dirs.push_back(acc.normalized());
          acc.setZero();
          cnt = 0;
          t_bin = s.timestamp;
        }
      }
      for (size_t i = 0; i < dirs.size(); ++i)
        for (size_t j = i + 1; j < dirs.size(); ++j)
          sweep = std::max(sweep, std::acos(std::min(1.0, std::max(-1.0, dirs[i].dot(dirs[j])))));
    }
    // row coverage: std of u_frac
    double rmean = 0.0, rvar = 0.0;
    for (double u : rows)
      rmean += u;
    rmean /= std::max<size_t>(rows.size(), 1);
    for (double u : rows)
      rvar += (u - rmean) * (u - rmean);
    rvar /= std::max<size_t>(rows.size(), 1);
    // parallax proxy: median per-track first-to-last bearing angle
    std::vector<double> par;
    for (auto &kv : first_bear) {
      const Eigen::Vector3d &b0 = kv.second;
      const Eigen::Vector3d &b1 = last_bear[kv.first];
      par.push_back(std::acos(std::min(1.0, std::max(-1.0, b0.dot(b1)))));
    }
    double par_med = 0.0;
    if (!par.empty()) {
      std::nth_element(par.begin(), par.begin() + par.size() / 2, par.end());
      par_med = par[par.size() / 2];
    }
    meta.fingerprint << vw.cwiseSqrt(), va.cwiseSqrt(), sweep, (flow_n ? flow_sum / flow_n : 0.0), std::sqrt(rvar), par_med,
        (meta.t1 - meta.t0), meta.temp_span;
  }
  out.uid = uid_next_++; // only successfully assembled windows consume an id
  return true;
}
