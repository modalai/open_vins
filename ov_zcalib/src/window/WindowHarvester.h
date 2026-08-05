/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: streaming window harvester (the image-free front-end core).
 *
 * Consumes RawImu + FrameObs streams on the SESSION thread (the RT boundary is
 * the feeders' SPSC push; nothing here runs on a sensor callback). Excitation-
 * triggered variable-length windows (2-8 s): a window OPENS when the rolling
 * gyro/accel excitation clears the gate with healthy tracking, EXTENDS while
 * excited, and CLOSES on quiet, max duration, or a frame gap (>= 2 missing
 * frames or > max_frame_gap_s splits; > max_drop_frac of expected frames
 * invalidates). At close it assembles a solver-ready WindowData: clone
 * subsampling to the dense-solver budget, min/max track-length filtering (the
 * cap bounds correlated KLT drift), dense feature-id remap,
 * mid-exposure clone stamps mapped to the IMU clock at the SEED td (td_ref),
 * seed-intrinsics bearings, and the padded raw-IMU slice. Memory is bounded:
 * one fixed-capacity IMU ring + the open window's frame list; no allocation
 * in the steady per-sample path.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_WINDOW_HARVESTER_H
#define OV_ZCALIB_WINDOW_HARVESTER_H

#include <cstdint>
#include <deque>

#include "../utils/StreamTypes.h"
#include "../solve/WindowBA.h"

namespace ov_zcalib {

struct HarvesterConfig {
  // excitation gate (rolling std over excite_win_s)
  double min_excite_w = 0.20;  ///< rad/s
  double min_excite_a = 0.35;  ///< m/s^2
  double excite_win_s = 1.0;
  // window shape (3-5 s default per the solver clone budget; 2 s floor, 8 s
  // only with subsampling -- configure max_window_s up alongside max_clones)
  double min_window_s = 2.0;
  double max_window_s = 5.0;
  double quiet_close_s = 0.75; ///< close after this long below the gate
  // frame health
  double max_frame_gap_s = 0.100;
  int max_consec_drops = 1; ///< tolerated missing frames (>= 2 splits)
  /// Beyond this fraction of ITS OWN expected frames, a camera is EVICTED from the window -- the
  /// window itself survives on whatever cameras are still healthy, and is invalidated only when
  /// none are.
  ///
  /// Per camera by design: a pooled rig-level drop counter is not a health gate, it is a contagion
  /// on rigs whose cameras have different ingest costs. MEASURED (dual-camera rig, 60 Hz rolling +
  /// 30 Hz global sharing one tracker): the 60 Hz stream saturated the tracker and shed ~1500
  /// frames, invalidating 71 windows against 3 for the same session with that camera removed --
  /// the OTHER camera lost two thirds of its windows to a camera it shares no feature with
  /// (weakest direction 8.10 -> 3.23, held-out VERIFY 71.7% -> 60.2%). A camera must be
  /// answerable for its own frames and no one else's.
  double max_drop_frac = 0.10;
  // solver budgets
  int max_clones = 70;
  int min_clones = 12;
  int min_track_len = 3;
  int max_track_len = 40; ///< obs beyond this per track are dropped (drift cap)
  int min_feats = 12;
  double pix_sigma = 1.0;
  double imu_pad_s = 0.08; ///< slack around the window for td refinement
};

/// Per-window excitation fingerprint (reservoir similarity + gating material)
struct WindowMeta {
  Eigen::Matrix<double, 12, 1> fingerprint = Eigen::Matrix<double, 12, 1>::Zero();
  double t0 = 0.0, t1 = 0.0; ///< camera-clock span
  double temp_mean = 0.0, temp_span = 0.0;
  double drop_frac = 0.0;
  int clones = 0, feats = 0;
};

class WindowHarvester {
public:
  /// @param seed calibration snapshot at COLLECT start: bearings/undistortion and
  ///        the td/tr reference are frozen to it (replay determinism).
  WindowHarvester(const HarvesterConfig &cfg, const SharedCalib &seed);

  /// IMU samples (monotonic). Cheap: ring append + rolling-stat update.
  void push_imu(const RawImu &s);

  /// Tracked frame. Returns true when a window CLOSED and is ready to pop.
  bool push_frame(const FrameObs &f);

  /// Force-close any open window (end of session / state change).
  bool flush();

  /// Retrieve the closed window. Valid once after each push_frame()/flush() true.
  bool pop_window(WindowData &out, WindowMeta &meta);

  /// Rolling excitation (for SETTLE/UI): stds over the last excite_win_s.
  void excitation(double &w_std, double &a_std) const;

  int windows_invalidated() const { return invalidated_; }

private:
  void close_window_(bool valid);
  bool assemble_(WindowData &out, WindowMeta &meta);

  HarvesterConfig cfg_;
  SharedCalib seed_;

  // bounded raw-IMU history (ring; capacity set at construction)
  std::deque<RawImu> imu_hist_;
  double imu_hist_span_;

  // rolling excitation stats (exact windowed sums over a small ring)
  std::deque<RawImu> excite_ring_;
  Eigen::Vector3d sw_ = Eigen::Vector3d::Zero(), sw2_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d sa_ = Eigen::Vector3d::Zero(), sa2_ = Eigen::Vector3d::Zero();

  // open-window state. The window spans the WHOLE rig (all cameras, one IMU, one trajectory), but
  // cadence and gaps are per camera -- a 60 Hz and a 30 Hz stream interleave into inter-frame gaps
  // that belong to neither of them.
  bool open_ = false;
  std::vector<FrameObs> frames_; ///< every camera's frames, in arrival (= global time) order
  double quiet_since_ = -1.0;
  std::vector<double> nominal_seed_; ///< declared frame period, per camera (CamCalib::fps; 0 = unknown)
  std::vector<double> nominal_dt_;   ///< learned frame period, per camera (seeded from nominal_seed_)
  std::vector<double> last_ts_;      ///< last frame stamp, per camera (-1 = none yet this window)
  std::vector<int> drops_in_window_; ///< inferred missing frames, PER CAMERA (see max_drop_frac)
  std::vector<char> evicted_;        ///< cameras whose own drop fraction failed the gate this window

  // output slot
  bool have_out_ = false;
  WindowData out_win_;
  WindowMeta out_meta_;
  int invalidated_ = 0;
  // session-unique window ids (PreintStore index; 0 reserved = uncached).
  // Assigned at assembly in deterministic order, so they survive reservoir
  // moves and every fused/holdout/half subset copy.
  std::uint32_t uid_next_ = 1;
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_WINDOW_HARVESTER_H
