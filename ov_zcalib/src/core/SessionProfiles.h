/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: the VOXL device/host profile presets, shared VERBATIM between the
 * host CLI (ov_zcalibrate --voxl [--flight]) and the in-server live mode
 * (voxl-open-vins-server --calibrate [--calib-flight]). One definition site so
 * the two entry points can never drift apart: every value below is the one the
 * validated scorecards ran with.
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
  // Close-range (indoor handheld) window shape: at sub-metre depth, kinematic
  // drift is amplified into bearing error by 1/depth; shorter windows keep the
  // linear seed in-basin.
  cfg.harvester.max_window_s = 2.5;
  cfg.harvester.quiet_close_s = 0.5;
  // ONE cam alternation round: on this profile's shape a carry-free same-binary
  // A/B shows round 2 is a byte-level no-op (committed YAML identical to alt=2,
  // every downstream stage iteration-identical, falsifier identical) for ~25 s
  // less solve wall. The library default stays 2 (the suite is tuned there);
  // the flight profile is cam_mode 0, so this is the only profile the knob
  // affects. Judge alt-round savings carry-free only.
  cfg.cam_alt_rounds = 1;
  // Settle pass OFF: same-binary A/B is byte-identical with and without on
  // this profile -- the alternation anchor already owns the pinhole row.
  // Library default keeps settle (general-case seeds).
  cfg.cam_settle = false;
  // Certificate in B2-polish: with qn-policed duels the stage accepts once and
  // stops at its best point instead of merit-noise churn -- B2 48.3 -> 10.9 s,
  // falsifier identical to the digit, all 7 blocks commit at unchanged ratios,
  // committed values move 0.1 us / 0.1 mm / <=0.06 px (>=20x inside claim
  // scales), holdout VERIFY improves 79.5 -> 79.9%.
  cfg.b2_cert = true;
  // Wald gate DECIDES (mode 1) on the HOST shape. Authority rests on three
  // legs: (a) MC H0 size after the df-corrected sizing (per-session kappa from
  // quarter-step scatter + F thresholds: false-freeze 81% -> ~8% at N=12; a
  // scalar-kappa rule was unshippable); (b) real-log operating point: all 7
  // blocks at print-precision parity, gate comfortable at T=6.0 vs 24.1;
  // (c) external truth: the kalibr scorecard validates the chain this gate
  // passes on this shape. Named residual: thin shapes show a junk-pass
  // discordance (split freezes, wald passes -- GN linearization understates
  // nonlinear wandering), so THIN/FLIGHT shapes keep split-half authority
  // until the junk-injection study; interim host-shape junk protection is the
  // r=6 observability floor + physical ceilings + cross-prediction + the
  // improve-on-seed VERIFY contract. Mode 2 remains one flag away.
  cfg.a_gate_mode = 1;
  // Fused evaluation in the A-chain: the capped-trajectory basin is CLOSER to
  // kalibr external truth than the legacy point (|dp| 1.085 vs ~1.28 mm, td
  // 8.9 vs ~17 us; best holdout VERIFY 81.5%; cam deltas grow to 0.6-1.0 px,
  // sub-gate). B-chain stays unfused (certificate/capped-regime conflict,
  // measured).
  cfg.p4 = true;
  // Window shape stays at the library default (70 clones / 40-obs tracks).
  // Measured: thinning is the biggest wall-clock lever (cubic: 32/10 solves
  // 5-7x faster and meets the 60 s target) but thin shapes lose the
  // information that disciplines the FLAT calibration directions -- Dw
  // half-recovers (gyro gauge half-absorbed, R_ItoC lands mid-gauge), and
  // DEEPER outer budgets make it worse: qA/da-offdiag/cam-center wander into
  // self-consistent-but-wrong basins, tighten their posteriors, and COMMIT
  // (the split-half even passes on the time-stable wrong point). The
  // at-kalibr scorecard (R 0.06-0.18 deg gauge-transported, p ~1.3 mm, td
  // ~10 us, Dw SVs within 1.5e-3) needs the full shape; thin shapes belong to
  // the FLIGHT profile where those blocks are unlock-gated.
  // Thermal hold in COLLECTION disabled: self-heating ICM-class rigs ramp
  // continuously and the hold gate thrashes COLLECT<->THERMAL_HOLD, dropping
  // frames. The solve-side 6 degC thermal bin still guards fusion consistency.
  // Re-enable per-rig once bench data justifies a slope threshold.
  cfg.thermal_hold_slope = 1e9;
  // NOTE (live sessions): collect_max_s stays at the library default here so
  // host replays of long logs are unchanged. A LIVE stream has no EOF -- the
  // live driver (voxl-open-vins-server --calibrate) owns end-of-collection and
  // cuts over into the solve on its own budget (40 s flight / 120 s handheld).
}

/// FLIGHT overlay: a ~40 s flight log starves the handheld profile for
/// STRUCTURAL reasons -- settle timeout + bootstrap eat half the log, the
/// 2.0-2.5 s window band + 0.75 s quiet-close skip sub-2 s excitation bursts,
/// and the strict metric seed gate rejects the factory-drift windows the
/// calibrator exists for. The overlay: fast settle handoff (stillness never
/// comes in flight), dense short windows, drift-budget envelope (+ post-A0
/// probation re-check), 2 holdouts + small-n VERIFY floors, thin solver shape
/// + budget for the <=60 s target. GATES ARE PINNED WITH THE PROFILE (gate
/// verdicts are shape/budget-dependent); the accel chain stays
/// excitation-gated and cam defaults to mode 0 -- flight data earns
/// ext/td/dw, not intrinsics.
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
  // Thin/flight shapes keep SPLIT-HALF authority (the voxl base grants wald
  // mode 1 for the HOST shape only): a measured thin-shape discordant pair
  // (split froze dqA 0.762 deg while wald under-read it) keeps mode 2 here --
  // split-half DECIDES, wald verdicts are logged for the paired record --
  // until the junk-injection study re-pins the corrected sizing per shape.
  cfg.a_gate_mode = 2;
  if (set_cam_mode0)
    cfg.cam_mode = 0;
}

/// Historical ICM-class 1 kHz calibration-weighting defaults, separate from VINS tuning.
/// These values came from platform-chain trailing comments; they are not universal IMU
/// constants or the supplied same-board BMI270 Allan fit. Override them in ov_calib.yaml.
/// Noise sets the relative weight of camera/IMU evidence and the allowed bias drift.
/// Static Allan estimates depend on the sensor, driver filtering, rate and temperature;
/// they need not describe errors during calibration motion. Kalibr explicitly allows
/// inflation for unmodeled effects: https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
/// Validate weighting with independent recovery/prediction checks, not smaller sigmas alone.
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
