/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * Soft-reset hand-off buffer: carries the LIVE filter's bias estimates (+ marginal sigmas)
 * into the next dynamic initialization attempt. Rationale: the free-S2 initializer's dominant
 * ill-conditioning is the gravity <-> accel-bias ambiguity; at a soft reset the filter's
 * converged biases are available and (unless the reset was divergence-triggered) far better
 * than the static config seeds, collapsing that valley (the same assumption sqrtVINS [Peng et
 * al., T-RO 2025] makes with "biases from prior calibration"; ORB-SLAM3 uses scheduled tight
 * bias priors for the same reason).
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
  /// Arm with a fresh snapshot (health thread, from soft_reset).
  void arm(const ResetBiasPrior &p) {
    std::lock_guard<std::mutex> lk(mtx_);
    prior_ = p;
  }

  /// Clear the episode (successful init or explicit teardown).
  void disarm() {
    std::lock_guard<std::mutex> lk(mtx_);
    prior_ = ResetBiasPrior();
  }

  /// Copy-out of the current prior (init thread, once per attempt).
  ResetBiasPrior prior() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return prior_;
  }

private:
  mutable std::mutex mtx_;
  ResetBiasPrior prior_;
};

} // namespace ov_init

#endif // OV_INIT_RESETPRIOR_H
