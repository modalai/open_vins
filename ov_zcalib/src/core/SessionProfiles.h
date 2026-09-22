/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * Shared VOXL profile presets for host replay and live calibration.
 * Explicit CLI or YAML settings may override these defaults.
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

/// VOXL bench profile. Apply caller overrides after this preset.
inline void apply_voxl_profile(SessionConfig &cfg) {
  cfg.settle_timeout_s = 10.0;
  // Short windows limit linear-seed drift for nearby handheld features.
  cfg.harvester.max_window_s = 2.5;
  cfg.harvester.quiet_close_s = 0.5;
  cfg.cam_alt_rounds = 1;
  cfg.cam_settle = false;
  cfg.b2_cert = true;
  // Nonlinear split halves decide; Wald remains diagnostic. Null-model sizing
  // alone does not establish rejection power under model mismatch. Explicit
  // host experiments may override this preset with --a-gate-mode 1.
  cfg.a_gate_mode = 2;
  cfg.p4 = true;
  // Retain the library's full window shape. The flight overlay bounds it below.
  // Disable collection-time thermal hold; solve-side thermal bins still apply.
  cfg.thermal_hold_slope = 1e9;
  // The live caller owns its collection deadline; replay ends at recorded EOF.
}

/// Flight overlay: smaller windows and a bounded solve. Split-half retains
/// authority; Wald is diagnostic. Callers may override the solve deadline and
/// camera mode after applying this preset.
/// @param set_cam_mode0 Set camera mode to zero unless the caller supplied one.
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
  // Split-half decides; Wald remains diagnostic.
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

/// Reconstruct a record's base profile and camera mode. Other nondefault
/// producing settings must be supplied separately; they are not serialized in
/// the profile tag.
/// @return Profile name, or nullptr for an unsupported legacy profile.
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

/// Default observation-density settings for the VOXL frontends.
struct VoxlTrackerKnobs {
  int num_feats = 150;
  int fast_threshold = 15;
  int grid_x = 8, grid_y = 6;
  int min_px_dist = 10;
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_SESSION_PROFILES_H
