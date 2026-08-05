/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: per-window preintegration value cache (P3).
 *
 * DEPENDENCE THEOREM (the reuse license). With Tg == 0 and bg_lin = ba_lin = 0
 * (structural in WindowBA), every AciPreintResult field is a function of
 * pi := (dw6, da6, q_AtoI) and the window's fixed sample stream + clone_times
 * ONLY. It is independent of td/tr/ext/cam (clone_times are td_ref-frozen; the
 * temporal transport lives in Factor_ReprojTd) and of all inner-solve states
 * (the factor's first-order CPI correction absorbs bias motion). A window's
 * preintegration therefore deduplicates across the warm/cold two-path at the
 * same p, across passes in frozen-pi stages, and across STAGES whose entry pi
 * bytes match — keyed by VALUE, never by pass index or stage name.
 *
 * KEY DISCIPLINE: bitwise equality (memcmp) on the exact 32-double value key
 * (live pi + noise-linearization pi), never an epsilon tolerance (hits would
 * become path-dependent) and never a hash (a collision is a silent wrong-pi
 * reuse that biases every block downstream). Well-defined because JointCalib's
 * restore(accepted_p) rewrites exact snapshot bytes. The window-content fields
 * guard the fixed-stream assumption: any future code that mutates win.imu or
 * win.clone_times post-harvest must be caught here, not silently reused.
 * Debug auditor: set OV_ZCALIB_PREINT_AUDIT=1 to recompute every hit and abort
 * on a byte mismatch (WindowBA).
 *
 * CONCURRENCY: entries are indexed by WindowData::uid (harvester-assigned,
 * monotone from 1; 0 = uncached). JointCalib resolves all slots on the main
 * thread before its pool starts; workers touch only their own windows' entries
 * under the fixed-range partition, and the pool join is the happens-before
 * edge. Lock-free by construction, serial == parallel bit-identical.
 *
 * A-CHAIN LEGACY INVARIANT COMPATIBILITY: this cache is pure memoization — a
 * hit returns the exact bytes recomputation would produce (replay-proven), so
 * it changes NO numbers and is legal in A0/A1a/A1b/split-halves, unlike the
 * P1 cert/carry/early-stop mechanisms, which change ARBITRATION.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_PREINT_CACHE_H
#define OV_ZCALIB_PREINT_CACHE_H

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "AciCalibPreint.h"

namespace ov_zcalib {

/**
 * @brief Bitwise value key for one window's preintegration.
 *
 * v[0..15]  = live pi (dw6 | da6 | q_AtoI4) — the mean/column linearization.
 * v[16..31] = noise-linearization pi (mirrors live pi when noise is unfrozen).
 * v[32..35] = ImuNoise sigmas (w, wb, a, ab) — they feed P15/the whitener;
 *             session-constant today, keyed so a future per-stage weighting
 *             change can never silently reuse a stale whitener.
 * v[36..54] = Tg extension, written ONLY when the session estimates Tg
 *             (SharedCalib::tg_enabled): live Tg (9) + noise-lin Tg (9) + a
 *             1.0 marker slot, so a 15-column entry can never serve a
 *             24-column request. Stays zero (the initializer) when tg is off
 *             — legacy keys keep their exact bytes.
 * The remaining fields are the window-content checksum (fixed-stream guard).
 */
struct PreintKey {
  double v[55] = {0};
  double clone_t0 = 0.0, clone_t1 = 0.0; ///< clone_times.front()/.back()
  double imu_t0 = 0.0, imu_t1 = 0.0;     ///< imu.front()/.back() timestamps
  std::uint32_t imu_n = 0, clone_n = 0;
  std::uint8_t noise_frozen = 0;

  bool operator==(const PreintKey &o) const {
    return std::memcmp(v, o.v, sizeof(v)) == 0 && std::memcmp(&clone_t0, &o.clone_t0, sizeof(double)) == 0 &&
           std::memcmp(&clone_t1, &o.clone_t1, sizeof(double)) == 0 && std::memcmp(&imu_t0, &o.imu_t0, sizeof(double)) == 0 &&
           std::memcmp(&imu_t1, &o.imu_t1, sizeof(double)) == 0 && imu_n == o.imu_n && clone_n == o.clone_n &&
           noise_frozen == o.noise_frozen;
  }
};

/// Bitwise equality of two preintegration results (cache auditor + parity test).
inline bool preint_bitwise_equal(const AciPreintResult &a, const AciPreintResult &b) {
  auto eqd = [](const double *x, const double *y, size_t n) { return std::memcmp(x, y, n * sizeof(double)) == 0; };
  if (a.Jq_pi.cols() != b.Jq_pi.cols() || a.Jb_pi.cols() != b.Jb_pi.cols() || a.Ja_pi.cols() != b.Ja_pi.cols())
    return false;
  return eqd(&a.dt, &b.dt, 1) && eqd(a.q_KtoK1.data(), b.q_KtoK1.data(), 4) && eqd(a.alpha.data(), b.alpha.data(), 3) &&
         eqd(a.beta.data(), b.beta.data(), 3) && eqd(a.P15.data(), b.P15.data(), 225) && eqd(a.J_q.data(), b.J_q.data(), 9) &&
         eqd(a.J_b.data(), b.J_b.data(), 9) && eqd(a.J_a.data(), b.J_a.data(), 9) && eqd(a.H_b.data(), b.H_b.data(), 9) &&
         eqd(a.H_a.data(), b.H_a.data(), 9) && eqd(a.H_q.data(), b.H_q.data(), 9) &&
         eqd(a.Jq_pi.data(), b.Jq_pi.data(), a.Jq_pi.size()) &&
         eqd(a.Jb_pi.data(), b.Jb_pi.data(), a.Jb_pi.size()) && eqd(a.Ja_pi.data(), b.Ja_pi.data(), a.Ja_pi.size()) &&
         eqd(a.w_end.data(), b.w_end.data(), 3) && eqd(a.a_end.data(), b.a_end.data(), 3);
}

/**
 * @brief One window's cached preintegration + whitener (single-entry: the last
 *        key wins — frozen-pi stages hit every pass, moving-pi stages dedup the
 *        two-path within a pass).
 */
struct WindowPreint {
  PreintKey mean_key, whit_key;
  bool has_means = false, has_whit = false;
  std::vector<AciPreintResult> pre;                   ///< per interval
  std::vector<Eigen::Matrix<double, 15, 15>> W;       ///< sqrt information per interval
  std::vector<Eigen::Matrix<double, 15, 3>> Wfold;    ///< gravity-fold columns per interval
  /// Persistent per-window problem graph (WindowBA-internal WindowGraph,
  /// type-erased; deleter bound at creation). Built once per window: the
  /// factor set and parameter registry are window-structure-fixed — per eval
  /// only VALUES and linearization members are rewritten (IMU factors on
  /// preint-key change, reproj clone kinematics always, prior anchors
  /// always). Same single-writer-per-uid discipline as the preint fields.
  std::shared_ptr<void> graph;
};

/**
 * @brief Session-lifetime store, indexed by WindowData::uid (0 = uncached).
 *        ensure() may RESIZE and is main-thread-only; take entry pointers only
 *        after every ensure() for the working set has run.
 */
struct PreintStore {
  std::vector<WindowPreint> by_uid;
  WindowPreint *ensure(std::uint32_t uid) {
    if (uid == 0)
      return nullptr;
    if (uid >= by_uid.size())
      by_uid.resize(uid + 1);
    return &by_uid[uid];
  }
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_PREINT_CACHE_H
