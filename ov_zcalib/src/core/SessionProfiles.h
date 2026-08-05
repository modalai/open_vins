/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: the VOXL device/host profile presets, shared VERBATIM between the
 * host CLI (ov_zcalibrate --voxl [--flight]) and the in-server live mode
 * (voxl-open-vins-server --calibrate [--calib-flight]). One definition site so
 * the two entry points can never drift apart: every value below is the one the
 * validated scorecards ran with (Sting v21 / down v1 ladders, 2026-07-11).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_SESSION_PROFILES_H
#define OV_ZCALIB_SESSION_PROFILES_H

#include "../utils/SessionRecord.h"
#include "CalibSessionRunner.h"

namespace ov_zcalib {

/// VOXL base profile (handheld/bench; kalibr-grade shape). Applies on top of
/// library defaults; CLI/experiment overrides apply AFTER this.
inline void apply_voxl_profile(SessionConfig &cfg) {
  cfg.settle_timeout_s = 10.0; // field logs rarely begin still enough for a full baseline
  // indoor handheld: close-range profile — at sub-metre depth, kinematic drift
  // is amplified into bearing error by 1/depth, shorter windows keep the
  // linear seed in-basin
  cfg.harvester.max_window_s = 2.5;
  cfg.harvester.quiet_close_s = 0.5;
  // ONE cam alternation round (P2a, 2026-07-11): the carry-free A/B on the
  // Sting host shape shows B round-2 is a byte-level no-op — committed YAML
  // IDENTICAL to alt=2 (same binary), every downstream stage (settle,
  // B2-polish) iteration-identical, falsifier identical — for −25 s of
  // solve wall (237→212 s class). The library default stays 2 (the suite
  // is tuned there); the old v8 "alt=1 is free" verdict was carry-tainted,
  // this one is carry-free with byte-level evidence. Flight profile is
  // cam_mode 0 (no cam refine) — alternation is structurally absent there,
  // so this is the only profile the knob affects.
  cfg.cam_alt_rounds = 1;
  // Settle pass OFF (same-binary A/B 2026-07-11): byte-identical committed
  // YAML with and without — the alternation anchor already owns the pinhole
  // row on this profile. Library default keeps settle (general-case seeds).
  cfg.cam_settle = false;
  // Certificate in B2-polish (same-binary A/B 2026-07-11): B2 48.3->10.9 s.
  // With qn-policed duels the stage accepts once and stops at its best
  // point — the 12-accept polish was merit-noise churn (P0's ~85%
  // double-solve diagnosis at full strength). Falsifier identical to the
  // digit, all 7 blocks commit at unchanged ratios, committed values move
  // 0.1 us / 0.1 mm / <=0.06 px (>=20x inside claim scales), and holdout
  // VERIFY *improves* 79.5 -> 79.9%.
  cfg.b2_cert = true;
  // Wald gate DECIDES (mode 1) on the HOST shape -- authority granted
  // 2026-07-11 late on three legs: (a) MC H0 size after the df-corrected
  // redesign (per-session kappa from quarter-step scatter + F thresholds:
  // false-freeze 81% -> ~8% at N=12; the scalar-kappa rule was unshippable);
  // (b) real-log operating point: 133.6 s with all 7 blocks at
  // print-precision parity, gate comfortable at T=6.0/24.1 (kappa_sess 5.1,
  // df 18) where the old rule sat at 15.1/16.8; (c) external truth: the
  // kalibr scorecard validates the chain this gate passes on this shape.
  // RESIDUAL, NAMED: the thin-shape junk-pass discordance (split freezes,
  // wald passes -- GN linearization understates nonlinear wandering) means
  // THIN/FLIGHT shapes keep split-half authority until the H1
  // junk-injection study; interim host-shape junk protection = r=6
  // observability floor + physical ceilings + cross-prediction + the
  // improve-on-seed VERIFY contract. Mode 2 remains one flag away.
  cfg.a_gate_mode = 1;
  // P4 fused evaluation in the A-chain (2026-07-11 late): 128.5 s, and the
  // capped-trajectory basin is CLOSER to kalibr external truth than the
  // legacy point (|dp| 1.085 vs ~1.28 mm, td 8.9 vs ~17 us; holdout VERIFY
  // 81.5% = best of day; cam deltas grow to 0.6-1.0 px, sub-gate). The
  // wald gate at the capped A1a point: T=5.1/24.1, kappa_sess 4.4. B-chain
  // stays unfused (cert/capped-regime conflict, measured).
  cfg.p4 = true;
  // Window shape stays at the library default (70 clones / 40-obs tracks).
  // MEASURED on this log (2026-07-10): thinning is the biggest wall-clock
  // lever (cubic: 32/10 solves 5-7x faster and meets the 60 s target) but
  // the thin shapes lose the information that disciplines the FLAT
  // calibration directions — Dw half-recovers (gyro gauge half-absorbed,
  // R_ItoC lands mid-gauge), and deeper outer budgets make it WORSE, not
  // better (24 outers @ 48/16: qA/da-offdiag/cam-center wander into
  // self-consistent-but-wrong basins, tighten their posteriors, and COMMIT
  // — the split-half even passes on the time-stable wrong point). The
  // at-kalibr scorecard (R 0.06-0.18 deg gauge-transported, p ~1.3 mm, td
  // ~10 us, Dw SVs within 1.5e-3) needs the full shape. Thin shapes belong
  // to the FLIGHT profile where those blocks are unlock-gated and the
  // P-wave (early-stop, Wald gate w/ cross-prediction, shared ba) pays for
  // the wall-clock properly.
  // Thermal effects in COLLECTION deactivated for now (operator directive
  // 2026-07-10): self-heating ICM rigs ramp continuously and the hold gate
  // thrashes COLLECT<->THERMAL_HOLD, dropping frames. The solve-side 6 degC
  // thermal bin still guards fusion consistency. Re-enable per-rig once
  // bench data justifies a slope threshold.
  cfg.thermal_hold_slope = 1e9;
  // NOTE (live sessions): collect_max_s stays at the library default here so
  // host replays of long logs are unchanged. A LIVE stream has no EOF -- the
  // live driver (voxl-open-vins-server --calibrate) owns end-of-collection and
  // cuts over into the solve on its own budget (40 s flight / 120 s handheld).
}

/// FLIGHT overlay (position_push study, 2026-07-10): a 41 s flight log starves
/// the handheld profile for STRUCTURAL reasons — settle timeout + bootstrap
/// eat 54% of the log, the 2.0-2.5 s window band + 0.75 s quiet-close skip
/// sub-2 s excitation bursts, and the strict metric seed gate rejects
/// factory-drift windows the calibrator exists for. The overlay: fast settle
/// handoff (stillness never comes in flight), dense short windows,
/// drift-budget envelope (+ post-A0 probation re-check), 2 holdouts + small-n
/// VERIFY floors, thin solver shape + budget for the <=60 s target. GATES ARE
/// PINNED WITH THE PROFILE (shape study: gate verdicts are shape/budget-
/// dependent); the accel chain stays excitation-gated and cam defaults to
/// mode 0 — flight data earns ext/td/dw, not intrinsics.
/// @param set_cam_mode0 pin cam_mode 0 (pass false when an explicit CLI
///        --cam-mode override must win over the profile default)
inline void apply_flight_overlay(SessionConfig &cfg, bool set_cam_mode0) {
  cfg.retro_harvest = true; // reclaim the bootstrap span (dead time on short flight logs)
  cfg.settle_timeout_s = 3.0;
  cfg.harvester.min_window_s = 1.2;
  cfg.harvester.max_window_s = 2.0;
  cfg.harvester.quiet_close_s = 0.4;
  cfg.harvester.max_clones = 32;
  cfg.harvester.max_track_len = 10;
  cfg.seed.drift_budget_ms2 = 0.10; // ~1% accel-chain envelope
  cfg.scorer.holdout_every = 4;
  cfg.min_holdout = 2;
  cfg.select_K = 10;
  cfg.solve_budget_s = 20.0;
  cfg.commit_attribution = false; // LOBO solves off the flight budget
  // W0 (permanent home; the server-side interim pin retires at the flip): thin/flight
  // shapes keep SPLIT-HALF authority. The voxl base grants wald mode-1 for the HOST
  // shape only; the measured thin-shape discordant pair (split froze dqA 0.762 deg
  // while wald under-read it) plus the W1 kappa-contamination finding keep mode 2
  // here — split-half DECIDES, wald verdicts log, the paired campaign accumulates —
  // until the H1 junk-injection study re-pins the corrected sizing per shape.
  cfg.a_gate_mode = 2;
  if (set_cam_mode0)
    cfg.cam_mode = 0;
}

/// ICM-class 1 kHz imu_apps CALIBRATION-weighting densities -- deliberately NOT the filter's.
///
/// The platform kalibr chains ship INFLATED noise for filter robustness. A FILTER wants that: an
/// over-stated sigma buys stability against unmodelled effects. A batch CALIBRATOR must not, because
/// sigma here is not a safety margin, it is the LEVER ARM between the IMU residuals and the camera
/// residuals. Inflating the IMU ~7x down-weights it ~48x in INFORMATION against the cameras -- and
/// then the gyro can no longer defend dw, which is a gyro parameter and (on a rig with no shared
/// field of view) the ONLY block the cameras share. A misspecified camera walks it out of family,
/// and every other camera inherits the damage through preintegration.
///
/// These four values used to be BYTE-IDENTICAL to the inflated chain -- the comment claimed a
/// de-inflation the code never performed. They are now the raw Allan fit carried in the chain's own
/// trailing comments (fpvhires/kalibr_imu_chain.yaml: "#w", "#wb", "#a", "#ab"), i.e. 6.9x / 1667x /
/// 5.9x / 43x below what was being used.
///
/// MEASURED, down truth log (log0004) vs the kalibr-grade reference:
///   inflated (old):  0.115 deg / 1.36 mm   VERIFY 50.6%
///   raw Allan (new): 0.025 deg / 0.54 mm   VERIFY 59.0%     <- 4.6x rotation, 2.5x position
/// and every block's posterior tightened (dw 0.21 -> 0.14, da 0.82 -> 0.38).
///
/// Override per rig with the four kalibr key names in ov_zcalib.yaml -- an Allan fit is a property of
/// the PART, and this is the one place a calibrator should be told about it.
inline void apply_voxl_noise_defaults(ImuNoise &n) {
  n.sigma_w = 1.3990944749616306e-4;
  n.sigma_wb = 4.1189724174615527e-7;
  n.sigma_a = 3.8947538150776763e-3;
  n.sigma_ab = 5.538346201712153e-5;
}

/// Rebuild the effective config from a record's PROFILE TAG (record v2). This
/// is what makes a session record replayable: the record carries the streams
/// AND the configuration that consumed them, so `--replay` reproduces the run
/// that produced it instead of silently scoring library defaults.
/// @return human-readable profile name (for the replay banner)
inline const char *apply_profile_tag(SessionConfig &cfg, SessionProfile profile, int cam_mode) {
  const char *name = "library defaults";
  switch (profile) {
  case SessionProfile::VOXL:
    apply_voxl_profile(cfg);
    name = "voxl";
    break;
  case SessionProfile::VOXL_FLIGHT:
    apply_voxl_profile(cfg);
    apply_flight_overlay(cfg, /*set_cam_mode0=*/true);
    name = "voxl+flight";
    break;
  case SessionProfile::EUROC:
    // EuRoC ingestion is not supported (VOXL data only). The enum value survives for
    // record-header compatibility; a record so tagged is refused at load.
    return nullptr;
  case SessionProfile::LIBRARY:
    break;
  }
  if (cam_mode >= 0)
    cfg.cam_mode = cam_mode; // the effective value the producing session ran
  return name;
}

/// The tracker knobs every validated VOXL run used (VoxlLogFeeder host KLT and
/// the in-server TrackOCL front-end feed the SAME solver: keep the observation
/// density identical across trackers so profiles transfer).
struct VoxlTrackerKnobs {
  int num_feats = 150;
  int fast_threshold = 15;
  int grid_x = 8, grid_y = 6;
  int min_px_dist = 10;
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_SESSION_PROFILES_H
