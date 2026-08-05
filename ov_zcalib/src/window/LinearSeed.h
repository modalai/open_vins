/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * ov_zcalib: linear window seeder (projector-free arrowhead Dong-Si).
 *
 * Given gyro-preintegrated clone rotations (bias-corrected at the bootstrap
 * bg, intrinsics at the current calibration) the CPI kinematics unroll to
 *      p_k = v0*T_k - 0.5*g*T_k^2 + S_k,   S/V by forward recursion
 *      S_{k+1} = S_k + V_k dt_k + R_k^T alpha_k,  V_{k+1} = V_k + R_k^T beta_k,
 * and each undistorted bearing b of feature f at clone k gives the LINEAR
 * constraint (K = skew(b) R_ItoC R_k)
 *      K p_f - T_k K v0 + 0.5 T_k^2 K g = K S_k - skew(b) p_IinC.
 * The joint LS over [v0, g, {p_f}] has arrowhead normal equations: per-feature
 * 3x3 blocks + a 6x6 Schur complement on [v0, g] — O(F) total, no projector,
 * no big matrix. Features whose 3x3 block is near-singular (short/parallax-
 * free tracks) fall back to a median-depth bearing seed after the solve.
 * The gravity magnitude is solved FREE and its deviation from the configured
 * |g| is returned as a seed-health metric before the S2 projection.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_LINEAR_SEED_H
#define OV_ZCALIB_LINEAR_SEED_H

#include "../solve/WindowBA.h"

namespace ov_zcalib {

struct LinearSeedReport {
  bool ok = false;
  double g_mag = 0.0;            ///< |g| of the unconstrained solve (health: vs grav_mag)
  double ba_mag = 0.0;           ///< |ba| of the linear solve (health: vs max_ba_mag)
  double mean_ang_resid = 0.0;   ///< mean bearing angular residual of the linear solution [rad]
  int feats_solved = 0;
  int feats_fallback = 0;        ///< near-singular tracks seeded at the median depth
  double median_depth = 0.0;
  double bg_shift = 0.0;         ///< |bg - bg_boot| found by the visual bias pre-solve [rad/s]
  /// Window admitted only through the drift-budget ENVELOPE (metric residual
  /// above the strict gate but within eps_eff*g*T^2/4): calibration-recoverable
  /// factory drift, flagged for the session's post-A0 probation re-check.
  bool envelope_only = false;
};

struct LinearSeedConfig {
  double min_block_eig = 1e-7;    ///< feature 3x3 admission (relative to trace)
  double max_g_mag_dev = 0.20;    ///< reject if | |g|-grav_mag | / grav_mag exceeds this
  double max_mean_ang_resid = 0.035; ///< angular backstop (far scenes; see metric gate)
  /// PRIMARY residual gate, depth-invariant: mean angular residual x median
  /// depth. A fixed angular gate is depth-blind — 0.1 m of kinematic drift is
  /// 1.4 deg at 4 m but 14 deg at 0.4 m (close-range kalibr-sample lesson).
  double max_mean_metric_resid = 0.10; ///< [m]
  double max_fallback_frac = 0.5; ///< reject if most tracks were unsolvable
  double fallback_depth = 4.0;    ///< used only when NO feature solved (degenerate window)
  double ba_prior_sigma = 0.05;   ///< [m/s^2] Tikhonov scale on the linear accel bias
  double max_ba_mag = 0.5;        ///< [m/s^2] reject wild bias solutions (health gate)
  /// Per-window gyro-bias pre-solve from consecutive-clone Procrustes rotations
  /// (window-local hand-eye). Rescues sessions with NO still baseline: a
  /// 0.01-0.02 rad/s unmodeled bias drifts the chain 60-100 mrad over a window
  /// and no [v0,g,ba,feats] combination can absorb it. Off when bg_boot comes
  /// from a still baseline (it is already better than the visual estimate).
  bool bias_presolve = false;
  int bias_presolve_iters = 2;
  double max_bg_shift = 0.06;     ///< [rad/s] sanity bound on the pre-solve correction
  /// Drift-budget envelope on the metric gate for FACTORY-FRESH rigs: an
  /// uncalibrated accel chain with effective scale/misalignment error eps_eff
  /// MUST produce ~eps_eff*g*T^2/4 of kinematic drift the linear seed cannot
  /// absorb — a fixed 0.10 m gate structurally rejects long windows on exactly
  /// the rigs the calibrator exists for. gate(T) = max_mean_metric_resid +
  /// 0.25*drift_budget_ms2*T^2 (drift_budget_ms2 ~= eps_eff*g, e.g. 0.10 for a
  /// 1% envelope). 0 = exact legacy behavior. Envelope admissions are flagged
  /// (LinearSeedReport::envelope_only) for post-A0 probation re-checks.
  double drift_budget_ms2 = 0.0;
};

class LinearSeed {
public:
  /**
   * @brief Fill win.seed_* from the linear solve (production replacement of the
   *        sim truth seeds). Uses win.obs[].bearing, win.imu, calib {q_ItoC,
   *        p_IinC, imu model, grav_mag} and the bootstrap gyro bias.
   * @return false (and win.has_seeds untouched) when the window fails the
   *         seed-health gates — the caller drops the window.
   */
  static bool seed_window(WindowData &win, const SharedCalib &calib, const Eigen::Vector3d &bg_boot, LinearSeedReport &rep);
  static bool seed_window(WindowData &win, const SharedCalib &calib, const Eigen::Vector3d &bg_boot, LinearSeedReport &rep,
                          const LinearSeedConfig &cfg);
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_LINEAR_SEED_H
