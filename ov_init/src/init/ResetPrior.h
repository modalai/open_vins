/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Contributor: Joao Leonardo Silva Cotta (@zauberflote1)
 *
 * Soft-reset hand-off buffer: carries the LIVE filter's bias estimates (+ marginal sigmas)
 * into the next dynamic initialization attempt. Rationale: the free-S2 initializer's dominant
 * ill-conditioning is the gravity <-> accel-bias ambiguity; at a soft reset the filter's
 * converged biases are available and (unless the reset was divergence-triggered) far better
 * than the static config seeds, collapsing that valley (sqrtVINS T-RO'25 assumes exactly this
 * "biases from prior calibration"; ORB-SLAM3 uses scheduled tight bias priors).
 *
 * Threading: the health thread arms the context inside soft_reset() (sensor callbacks already
 * drained); the detached init thread copies it out once per attempt. A plain mutex with
 * copy-in/copy-out is used -- neither side is on the RT path, and the critical section is a
 * ~14-double copy (same episode-scoped pattern as the warmstart_next_init atomic flag).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_RESETPRIOR_H
#define OV_INIT_RESETPRIOR_H

#include <Eigen/Dense>
#include <mutex>

namespace ov_init {

/// Snapshot of the live filter's bias state taken at soft-reset time (all-zero/invalid by default).
struct ResetBiasPrior {
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();       ///< gyroscope bias estimate at snapshot
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();       ///< accelerometer bias estimate at snapshot
  Eigen::Vector3d sigma_bg = Eigen::Vector3d::Zero(); ///< per-axis 1-sigma marginal of bg
  Eigen::Vector3d sigma_ba = Eigen::Vector3d::Zero(); ///< per-axis 1-sigma marginal of ba
  double t_snapshot = -1;                             ///< state timestamp at capture (camera clock)
  int cause = 0;                                      ///< 0 = client/unknown, 1 = divergence-triggered
  bool valid = false;                                 ///< false => consumers fall back to config seeds
};

/// Episode-scoped reset context shared between VioManager (producer) and the initializer (consumer).
class ResetContext {
public:
  /// Arm with a fresh snapshot (health thread, from soft_reset). Resets the failure counter.
  void arm(const ResetBiasPrior &p, bool short_window) {
    std::lock_guard<std::mutex> lk(mtx_);
    prior_ = p;
    short_window_ = short_window;
    fails_ = 0;
  }

  /// Clear everything (successful init or explicit teardown).
  void disarm() {
    std::lock_guard<std::mutex> lk(mtx_);
    prior_ = ResetBiasPrior();
    short_window_ = false;
    fails_ = 0;
  }

  /// Give up on the short window only (bias prior kept; it ages out via random-walk inflation).
  void disarm_window() {
    std::lock_guard<std::mutex> lk(mtx_);
    short_window_ = false;
  }

  /// Copy-out of the current prior (init thread, once per attempt).
  ResetBiasPrior prior() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return prior_;
  }

  /// True while a short-window reset episode is armed.
  bool short_window_armed() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return short_window_;
  }

  /// Count a failed dynamic-init attempt while armed; returns the new count.
  int bump_failed_attempts() {
    std::lock_guard<std::mutex> lk(mtx_);
    return ++fails_;
  }

private:
  mutable std::mutex mtx_;
  ResetBiasPrior prior_;
  bool short_window_ = false;
  int fails_ = 0;
};

} // namespace ov_init

#endif // OV_INIT_RESETPRIOR_H
