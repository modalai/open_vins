/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: the CALIBRATOR profile (ov_zcalib.yaml) -- deliberately a separate file
 * from estimator_config.yaml, because the two answer different questions:
 *
 *   estimator_config.yaml  what the FILTER believes at boot and refines ONLINE.
 *   ov_zcalib.yaml          what the CALIBRATOR is GIVEN versus what it must EARN.
 *
 * The calibrator is a one-shot pre-warm for VIO. A filter wants a full prior on every
 * state; a calibrator told the answer will simply reproduce the answer. The seeding
 * policy is therefore the load-bearing part of this file, and its defaults are
 * MEASURED: one recorded reference-rig session scored across the full seed matrix,
 * everything else held fixed:
 *
 *   seed cam intrinsics   VERIFY 62.2%          per-unit factory cal; always right
 *   + seed imu intrinsics VERIFY 62.2 -> 73.3%  KEEP. starts the accel/gyro chain at the
 *                                               truth, and an unmoved block then ships
 *                                               the FACTORY value, not identity
 *   + seed R_ItoC         byte-identical NO-OP  the hand-eye bootstrap re-solves the
 *                                               rotation from scratch (and lands on the
 *                                               same one blind)
 *   + seed p_IinC         VERIFY 73.3 -> 48.6%  HARMFUL. the chain's lever arm is a
 *                                               MECHANICAL NOMINAL (~35 mm off) whose
 *                                               prior drags the solve toward it
 *   + seed td             VERIFY      -> 28.3%  HARMFUL. the shipped td is not per-unit.
 *
 * Defaults: seed the INTRINSICS (per-unit factory data the rig actually knows), earn
 * the EXTRINSICS and the TIME OFFSET blind. This file exists so the experiment can be
 * re-run on a different rig rather than taken on faith.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_CALIB_CONFIG_YAML_H
#define OV_ZCALIB_CALIB_CONFIG_YAML_H

#include <memory>
#include <string>

#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

#include "CalibSessionRunner.h"
#include "SessionProfiles.h"

namespace ov_zcalib {

/// What the session is GIVEN. Everything false here must be EARNED from the data.
struct SeedPolicy {
  bool cam_intrinsics = true;    ///< per-unit factory cal from the active chain
  bool imu_intrinsics = true;    ///< Dw/Da/R_AtoI, gauge-ported from the chain (kalibr <-> imu2)
  bool extrinsic_rotation = false; ///< no-op in practice: the hand-eye bootstrap re-solves it
  bool extrinsic_position = false; ///< HARMFUL: the chain's lever arm is a mechanical nominal
  bool time_offset = false;        ///< HARMFUL: the shipped td is not a per-unit value
  // NOTE: no readout knob. tr is a hardware fact (HAL3 sensor mode) that ALWAYS seeds from the
  // estimator config's camN_readout_time_s and is never estimated -- there is no policy to set.
  /// Tg (g-sensitivity) chain seed -- its OWN axis, deliberately decoupled from imu_intrinsics
  /// (the SeedOverride doctrine): "earn the dw/da/qA chain blind" must not silently zero the
  /// g-sensitivity the chain measured; with the seed discarded, |Tg|*g rides as a real gyro
  /// term for the shared dw block to absorb. The value is an INIT (estimate_tg earns it) --
  /// kalibr's own per-session Tg estimates scatter at the size of the value itself.
  bool tg = true;
};

/// What the session may ESTIMATE and COMMIT. A block that is seeded but NOT estimated
/// ships its seed and is reported as such (see the YAML's committed_blocks/seed_blocks).
struct EstimatePolicy {
  bool imu_intrinsics = true; ///< false => seed AND FREEZE: skips A1a/A1b entirely (~30% faster solve)
  int cam_intrinsics = 0;     ///< 0 fixed | 1 refine (tight priors) | 2 full (weak priors, gated)
  /// Estimate Tg (gyro g-sensitivity). EARNED per unit: the rig's own kalibr sessions scatter
  /// beyond |Tg| between runs, so no chain value deserves seed authority.
  /// Unlocks only through the A1b excitation gate + split-half/Wald falsifier; requires an
  /// estimable IMU chain (a frozen factory chain freezes tg with it).
  bool tg = true;
};

/// The whole calibrator profile: window shape + gates (SessionConfig), what is given
/// (SeedPolicy), what is earned (EstimatePolicy), and the session's own budgets.
struct CalibProfile {
  SessionConfig session;
  SeedPolicy seed;
  EstimatePolicy estimate;
  /// Calibration-WEIGHTING IMU densities -- deliberately not the filter's. sigma here is not a
  /// safety margin, it is the lever arm between the IMU and camera residuals: inflate the IMU and
  /// the gyro loses every argument with a camera, including the one about dw. See
  /// apply_voxl_noise_defaults().
  ImuNoise noise;
  /// Frames per second handed to the TRACKER, per camera (0 = the sensor's own rate). The calibrator
  /// clones far less than this -- the harvester subsamples each window to max_clones/n_cams -- so
  /// everything above what KLT needs is tracked at full cost and then discarded. On a multi-camera
  /// rig that waste is what saturates the (serialized, inline) tracker and sheds frames, and shed
  /// frames come back as destroyed windows. VIO itself tracks at 20-30 Hz on these rigs.
  double track_rate_hz = 30.0;
  std::string base = "flight"; ///< "flight" (thin, <=60 s class) | "bench" (full shape, kalibr-grade)
  double collect_budget_s = 40.0;    ///< live cutover ceiling (a live stream has no EOF)
  double collect_min_eig = 5.0;      ///< early cutover once the WEAKEST direction is this well determined
  bool cpu_performance_mode = true;  ///< measured 3.1x on the solve; restored on exit
};

/// Load a calibrator profile. A MISSING FILE IS NOT AN ERROR: the built-in defaults are the
/// measured-good configuration, and the YAML exists to override them. Returns false only if
/// the file exists but cannot be parsed, so a typo is loud rather than silently ignored.
inline bool load_calib_profile(const std::string &path, CalibProfile &out) {
  // ---- built-in defaults first (the measured-good configuration) ----
  out = CalibProfile();
  apply_voxl_profile(out.session);
  apply_flight_overlay(out.session, /*set_cam_mode0=*/true);
  apply_voxl_noise_defaults(out.noise);

  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) {
    PRINT_INFO("[ov_zcalib] no profile at %s -- using built-in defaults (seed intrinsics, earn ext/td)\n", path.c_str());
    return true;
  }
  std::fclose(f);

  auto parser = std::make_shared<ov_core::YamlParser>(path, /*fail_if_not_found=*/false);

  // ---- base shape ----
  parser->parse_config("profile", out.base, false);
  out.session = SessionConfig();
  apply_voxl_profile(out.session);
  if (out.base != "bench")
    apply_flight_overlay(out.session, /*set_cam_mode0=*/true);

  // ---- calibration-WEIGHTING IMU noise (the kalibr key names, so an Allan fit pastes in) ----
  parser->parse_config("gyroscope_noise_density", out.noise.sigma_w, false);
  parser->parse_config("gyroscope_random_walk", out.noise.sigma_wb, false);
  parser->parse_config("accelerometer_noise_density", out.noise.sigma_a, false);
  parser->parse_config("accelerometer_random_walk", out.noise.sigma_ab, false);

  // ---- seeding policy: what the calibrator is TOLD ----
  parser->parse_config("seed_cam_intrinsics", out.seed.cam_intrinsics, false);
  parser->parse_config("seed_imu_intrinsics", out.seed.imu_intrinsics, false);
  parser->parse_config("seed_extrinsic_rotation", out.seed.extrinsic_rotation, false);
  parser->parse_config("seed_extrinsic_position", out.seed.extrinsic_position, false);
  parser->parse_config("seed_time_offset", out.seed.time_offset, false);
  parser->parse_config("seed_tg", out.seed.tg, false);

  // ---- estimation policy: what it may EARN and COMMIT ----
  // (no readout key on either axis: tr is the fixed HAL3 hardware value, never estimated)
  parser->parse_config("estimate_imu_intrinsics", out.estimate.imu_intrinsics, false);
  parser->parse_config("estimate_cam_intrinsics", out.estimate.cam_intrinsics, false);
  parser->parse_config("estimate_tg", out.estimate.tg, false);
  out.session.cam_mode = out.estimate.cam_intrinsics;
  out.session.free_tg = out.estimate.tg;

  // ---- ingest ----
  parser->parse_config("track_rate_hz", out.track_rate_hz, false);

  // ---- session budgets ----
  parser->parse_config("collect_budget_s", out.collect_budget_s, false);
  parser->parse_config("collect_min_eig", out.collect_min_eig, false);
  parser->parse_config("solve_budget_s", out.session.solve_budget_s, false);
  parser->parse_config("num_threads", out.session.joint.num_threads, false);
  parser->parse_config("cpu_performance_mode", out.cpu_performance_mode, false);

  // ---- window shape (advanced; the profile above already sets a measured-good pair) ----
  parser->parse_config("max_clones", out.session.harvester.max_clones, false);
  parser->parse_config("max_track_len", out.session.harvester.max_track_len, false);
  parser->parse_config("select_k", out.session.select_K, false);
  parser->parse_config("stage_select", out.session.stage_select, false);
  parser->parse_config("select_k_a0", out.session.select_K_a0, false);
  parser->parse_config("select_k_a1", out.session.select_K_a1, false);
  parser->parse_config("select_k_b", out.session.select_K_b, false);
  parser->parse_config("min_holdout", out.session.min_holdout, false);

  // ---- commit gates (advanced) ----
  parser->parse_config("commit_sigma_factor", out.session.commit_sigma_factor, false);
  parser->parse_config("verify_min_improve", out.session.verify_min_improve, false);

  PRINT_INFO("[ov_zcalib] profile '%s' from %s\n", out.base.c_str(), path.c_str());
  PRINT_INFO("[ov_zcalib]   SEED   cam_intr=%d imu_intr=%d ext_R=%d ext_p=%d td=%d (tr: HAL3 hardware, always)\n",
             (int)out.seed.cam_intrinsics, (int)out.seed.imu_intrinsics, (int)out.seed.extrinsic_rotation,
             (int)out.seed.extrinsic_position, (int)out.seed.time_offset);
  PRINT_INFO("[ov_zcalib]   EARN   imu_intr=%d cam_mode=%d tg=%d   (everything not seeded is blind; tr never earned)\n",
             (int)out.estimate.imu_intrinsics, out.estimate.cam_intrinsics, (int)out.estimate.tg);
  PRINT_INFO("[ov_zcalib]   WEIGHT gyro %.4e/%.4e  accel %.4e/%.4e  (nd/rw -- calibration weighting, NOT the filter's)\n",
             out.noise.sigma_w, out.noise.sigma_wb, out.noise.sigma_a, out.noise.sigma_ab);
  return true;
}

} // namespace ov_zcalib

#endif // OV_ZCALIB_CALIB_CONFIG_YAML_H
