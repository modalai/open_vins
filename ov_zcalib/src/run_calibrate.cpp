/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalibrate: on-demand targetless VI calibrator CLI (host tool; never
 * co-resident with the filter). LIVE device sessions are NOT run from here:
 * voxl-open-vins-server --calibrate feeds the identical session runner
 * in-process from the server's own MPA drivers with the TrackOCL front-end
 * (the only tracker that exists on target), and records the same session
 * format this tool replays.
 *
 *   --replay <session.bin>   run the calibrator over a recorded session (CI path)
 *   --voxl <log000N_dir>     convert a voxl-logger log (KLT at the per-unit
 *                            intrinsics shipped inside the log), then calibrate
 *   --selftest               seed-identity writeback smoke test
 *
 * Common: --out <yaml> --cam-mode {0 fixed | 1 refine (default) | 2 full}
 *         --tr (estimate RS readout)
 *         --tr-seed <s> (HAL3 readout seed) --max-sec <s>
 *         --cam-pipe <name> --quiet
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <cstdio>
#include <cstring>
#include <iostream>

#include "utils/VoxlLogFeeder.h"
#include "utils/YamlWriteback.h"
#include "core/CalibSessionRunner.h"
#include "core/SessionProfiles.h"
#include "utils/quat_ops.h"

using namespace ov_zcalib;

static void print_report(const SessionReport &rep) {
  std::printf("\n================ ov_zcalibrate report ================\n");
  std::printf("state: %s%s%s\n", rep.final_state == RunnerState::DONE ? "DONE" : "ABORT",
              rep.abort_reason.empty() ? "" : " -- ", rep.abort_reason.c_str());
  for (size_t c = 0; c < rep.handeye.size(); ++c) {
    const HandEyeResult &he = rep.handeye[c];
    std::printf("bootstrap cam %zu: hand-eye rmse %.2e rad, diversity %.2f, td %.3f ms, pairs %d (trimmed %d)\n", c, he.rmse_rad,
                he.axis_diversity, 1e3 * he.td, he.pairs_used, he.pairs_trimmed);
  }
  std::printf("windows: harvested %d, retained %d (holdout %d), seed-rejected %d, invalidated %d, fused %d (min-eig %.2e)\n",
              rep.windows_harvested, rep.windows_retained, rep.windows_holdout, rep.windows_rejected_seed, rep.windows_invalidated,
              rep.windows_fused, rep.min_eig_whitened);
  std::printf("verify: holdout cost %.4e -> %.4e (improve %.1f%%)%s\n", rep.holdout_cost_seed, rep.holdout_cost_committed,
              100.0 * rep.verify_improve, rep.tr_hw_ok ? "" : "  [tr disagrees with the CONFIG readout seed -- seed suspect]");
  std::printf("blocks:");
  for (const auto &b : rep.blocks)
    std::printf("  %s=%s(%.2f)", b.label().c_str(), b.committed ? "COMMIT" : "seed", b.worst_ratio);
  std::printf("\n");
  if (rep.joint.sigma.size() > 0) {
    std::printf("posterior (1-sigma / prior):\n");
    for (int i = 0; i < rep.joint.sigma.size(); ++i)
      std::printf("  %-10s %.3e / %.3e\n", rep.joint.labels[i].c_str(), rep.joint.sigma(i), rep.joint.prior_sigma_vec(i));
  }
  for (size_t i = 0; i < rep.committed.cams.size(); ++i) {
    const CamCalib &k = rep.committed.cams[i];
    std::printf("committed cam %zu: td=%.4f ms  tr=%.4f ms  p_IinC=[%.4f %.4f %.4f]\n", i, 1e3 * k.td, 1e3 * k.tr, k.p_IinC(0),
                k.p_IinC(1), k.p_IinC(2));
  }
  std::printf("=====================================================\n");
}

int main(int argc, char **argv) {
  std::string replay_path, voxl_dir, out_yaml = "ov_zcalib_result.yaml";
  std::string cam_pipe = "tracking_front";
  std::string profile_ovr; ///< --profile voxl|flight|library (v1 records / deliberate re-scoring)
  bool selftest = false, quiet = false, free_tr = false, joint_verbose = false;
  bool flight = false, have_cam_mode = false;
  // experiment knobs (-1 = keep library/profile default)
  int outer_iters_ovr = -1, window_iters_ovr = -1, select_k_ovr = -1, cam_alt_ovr = -1;
  double max_window_ovr = -1.0, verify_min_ovr = -1.0, solve_budget_ovr = -1.0;
  int cam_settle_ovr = -1, b2_cert_ovr = -1, a_candidate_ovr = -1, a_gate_mode_ovr = -1, p4_ovr = -1;
  double conv_tol_ovr = -1.0;
  int max_clones_ovr = -1, max_track_ovr = -1, a_carry_ovr = -1, fused_iters_ovr = -1, duel_accept_ovr = -1, threads_ovr = -1;
  int stage_select_ovr = -1, k_a0_ovr = -1, k_a1_ovr = -1, k_b_ovr = -1;
  int eoa_ovr = -1; // --export-on-accept {0|1}: forensic within-binary A/B of the deferred-export mechanism
  int cam_mode = 1; // refine by default (plan decision); --cam-mode 0 pins the existing intrinsics
  double tr_seed = 0.0, max_sec = 0.0, grav_mag = 9.80665;
  // vcc mechanical seed for --voxl (RPY_parent_to_child intrinsic-XYZ [deg], T_child_wrt_parent [m])
  // Built-in LAST-RESORT reference (Stinger tracking_front nominals). The real
  // reference comes from the log's own etc/modalai/extrinsics.conf (the log is
  // self-describing) or explicit --ext-rpy/--ext-t; a log from another SKU
  // (e.g. Starling2 front = [0,90,90]) scored against these defaults reads a
  // phantom ~180 deg (measured on position_push before conf sourcing landed).
  double ext_rpy[3] = {0.0, 90.0, -90.0};
  double ext_t[3] = {0.0618, 0.0191, 0.022};
  bool have_ext_rpy = false, have_ext_t = false;
  double noise_ovr[4] = {0, 0, 0, 0};
  bool have_noise_ovr = false;
  bool no_imu_intrinsics = false;
  // --imu-seed dw0..dw5 da0..da5 qx qy qz qw : seed the IMU intrinsics (imu2 gauge,
  // ut()-packed) instead of identity. On device the server derives these from the
  // active chain via imu_chain_to_calib(); here they are explicit so a host replay
  // can A/B blind-vs-seeded intrinsics on the SAME recorded session.
  double imu_seed[16];
  bool have_imu_seed = false;
  // --tg tg0..tg8 : g-sensitivity, ROW-major, in the imu2 gauge (i.e. ALREADY conjugated,
  // Tg_r = Tg_chain * Q_w -- the same convention --imu-seed uses for dw/da). On device the server
  // ports it from the active chain; here it is explicit so a replay can A/B modelled-vs-discarded
  // g-sensitivity on the SAME recorded session.
  double tg_seed[9] = {0};
  bool have_tg = false;
  // --no-tg-est : freeze Tg at its seed (zero, or --tg) and keep the LEGACY 15-column layout —
  // the corpus-parity ablation for the estimated-Tg default.
  bool no_tg_est = false;
  // Seeding study (operator, 2026-07-12): which blocks EARN their value from the
  // data, and which should start from the rig's own chain? Each axis is separately
  // seedable on replay so one recorded session can be scored across the matrix.
  double seed_rot[4] = {0, 0, 0, 1}, seed_pos[3] = {0, 0, 0}, seed_td_v = 0.0;
  bool have_seed_rot = false, have_seed_pos = false, have_seed_td = false;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char *n) { return !std::strcmp(argv[i], n); };
    if (arg("--replay") && i + 1 < argc)
      replay_path = argv[++i];
    else if (arg("--voxl") && i + 1 < argc)
      voxl_dir = argv[++i];
    else if (arg("--ext-rpy") && i + 3 < argc) {
      ext_rpy[0] = std::atof(argv[++i]);
      ext_rpy[1] = std::atof(argv[++i]);
      ext_rpy[2] = std::atof(argv[++i]);
      have_ext_rpy = true;
    } else if (arg("--ext-t") && i + 3 < argc) {
      ext_t[0] = std::atof(argv[++i]);
      ext_t[1] = std::atof(argv[++i]);
      ext_t[2] = std::atof(argv[++i]);
      have_ext_t = true;
    } else if (arg("--grav") && i + 1 < argc)
      grav_mag = std::atof(argv[++i]);
    else if (arg("--noise") && i + 4 < argc) {
      // sigma_w sigma_wb sigma_a sigma_ab (continuous-time densities). The
      // platform kalibr chains ship INFLATED values (5x white / 10x RW) for
      // filter robustness; calibration weighting may want the raw Allan fit.
      noise_ovr[0] = std::atof(argv[++i]);
      noise_ovr[1] = std::atof(argv[++i]);
      noise_ovr[2] = std::atof(argv[++i]);
      noise_ovr[3] = std::atof(argv[++i]);
      have_noise_ovr = true;
    } else if (arg("--imu-seed") && i + 16 < argc) {
      for (int k = 0; k < 16; ++k)
        imu_seed[k] = std::atof(argv[++i]);
      have_imu_seed = true;
    } else if (arg("--tg") && i + 9 < argc) { // g-sensitivity, imu2 gauge, row-major
      for (int k = 0; k < 9; ++k)
        tg_seed[k] = std::atof(argv[++i]);
      have_tg = true;
    } else if (arg("--seed-rot") && i + 4 < argc) { // q_ItoC (JPL xyzw) from the chain
      for (int k = 0; k < 4; ++k)
        seed_rot[k] = std::atof(argv[++i]);
      have_seed_rot = true;
    } else if (arg("--seed-pos") && i + 3 < argc) { // p_IinC from the chain
      for (int k = 0; k < 3; ++k)
        seed_pos[k] = std::atof(argv[++i]);
      have_seed_pos = true;
    } else if (arg("--seed-td") && i + 1 < argc) { // td (MID-exposure convention)
      seed_td_v = std::atof(argv[++i]);
      have_seed_td = true;
    } else if (arg("--no-tg-est"))
      no_tg_est = true;
    else if (arg("--no-imu-intrinsics"))
      no_imu_intrinsics = true;
    else if (arg("--joint-verbose"))
      joint_verbose = true;
    else if (arg("--selftest"))
      selftest = true;
    else if (arg("--quiet"))
      quiet = true;
    else if (arg("--tr"))
      free_tr = true;
    else if (arg("--out") && i + 1 < argc)
      out_yaml = argv[++i];
    else if (arg("--cam-mode") && i + 1 < argc) {
      cam_mode = std::atoi(argv[++i]);
      have_cam_mode = true;
    } else if (arg("--flight"))
      flight = true;
    else if (arg("--tr-seed") && i + 1 < argc)
      tr_seed = std::atof(argv[++i]);
    else if (arg("--max-sec") && i + 1 < argc)
      max_sec = std::atof(argv[++i]);
    else if (arg("--cam-pipe") && i + 1 < argc)
      cam_pipe = argv[++i];
    else if (arg("--profile") && i + 1 < argc)
      profile_ovr = argv[++i]; // replay: state the profile for an UNTAGGED (v1) record, or re-score deliberately
    // ---- experiment knobs (P-wave / flight-profile studies; default = library/profile values) ----
    else if (arg("--outer-iters") && i + 1 < argc)
      outer_iters_ovr = std::atoi(argv[++i]);
    else if (arg("--window-max-iters") && i + 1 < argc)
      window_iters_ovr = std::atoi(argv[++i]);
    else if (arg("--select-k") && i + 1 < argc)
      select_k_ovr = std::atoi(argv[++i]);
    else if (arg("--stage-select"))
      stage_select_ovr = 1;
    else if (arg("--select-k-a0") && i + 1 < argc)
      k_a0_ovr = std::atoi(argv[++i]);
    else if (arg("--select-k-a1") && i + 1 < argc)
      k_a1_ovr = std::atoi(argv[++i]);
    else if (arg("--select-k-b") && i + 1 < argc)
      k_b_ovr = std::atoi(argv[++i]);
    else if (arg("--max-window-s") && i + 1 < argc)
      max_window_ovr = std::atof(argv[++i]);
    else if (arg("--cam-alt-rounds") && i + 1 < argc)
      cam_alt_ovr = std::atoi(argv[++i]);
    else if (arg("--verify-min-improve") && i + 1 < argc)
      verify_min_ovr = std::atof(argv[++i]);
    else if (arg("--solve-budget-s") && i + 1 < argc)
      solve_budget_ovr = std::atof(argv[++i]);
    else if (arg("--no-cam-settle"))
      cam_settle_ovr = 0;
    else if (arg("--b2-cert") && i + 1 < argc)
      b2_cert_ovr = std::atoi(argv[++i]);
    else if (arg("--a-candidate"))
      a_candidate_ovr = 1;
    else if (arg("--a-gate-mode") && i + 1 < argc)
      a_gate_mode_ovr = std::atoi(argv[++i]);
    else if (arg("--p4"))
      p4_ovr = 1;
    else if (arg("--conv-tol") && i + 1 < argc)
      conv_tol_ovr = std::atof(argv[++i]);
    else if (arg("--a-carry"))
      a_carry_ovr = 1;
    else if (arg("--fused-iters") && i + 1 < argc)
      fused_iters_ovr = std::atoi(argv[++i]);
    else if (arg("--duel-accept"))
      duel_accept_ovr = 1;
    else if (arg("--export-on-accept") && i + 1 < argc)
      eoa_ovr = std::atoi(argv[++i]); // forensic A/B: 0 = legacy inline exports (byte-parity instrument)
    else if (arg("--threads") && i + 1 < argc)
      threads_ovr = std::atoi(argv[++i]); // window-pool width: results are thread-count-invariant (fixed-range partition, worker-ordered fold)
    else if (arg("--max-clones") && i + 1 < argc)
      max_clones_ovr = std::atoi(argv[++i]);
    else if (arg("--max-track-len") && i + 1 < argc)
      max_track_ovr = std::atoi(argv[++i]);
    else {
      std::printf("unknown arg: %s\n", argv[i]);
      return 1;
    }
  }

  SessionConfig cfg;
  cfg.cam_mode = cam_mode;
  cfg.free_tr = free_tr;
  cfg.out_yaml = out_yaml;
  cfg.verbose = !quiet;
  cfg.joint.verbose = joint_verbose; // per-pass ACCEPT/damp lines + the per-stage timing split
  // Experiment overrides apply LAST (after any profile defaults) via this
  // helper called at each run-branch entry — CLI always wins over profiles.
  auto apply_cli_overrides = [&](SessionConfig &c) {
    if (no_tg_est)
      c.free_tg = false; // ablation: 15-column legacy layout, byte-identical to the pre-tg corpus
    if (outer_iters_ovr > 0)
      c.joint.outer_iterations = outer_iters_ovr;
    if (window_iters_ovr > 0)
      c.joint.window_max_iters = window_iters_ovr;
    if (select_k_ovr > 0)
      c.select_K = select_k_ovr;
    if (stage_select_ovr >= 0)
      c.stage_select = true;
    if (k_a0_ovr > 0)
      c.select_K_a0 = k_a0_ovr;
    if (k_a1_ovr > 0)
      c.select_K_a1 = k_a1_ovr;
    if (k_b_ovr > 0)
      c.select_K_b = k_b_ovr;
    if (max_window_ovr > 0.0)
      c.harvester.max_window_s = max_window_ovr;
    if (cam_alt_ovr >= 0)
      c.cam_alt_rounds = cam_alt_ovr;
    if (verify_min_ovr >= 0.0)
      c.verify_min_improve = verify_min_ovr;
    if (solve_budget_ovr >= 0.0)
      c.solve_budget_s = solve_budget_ovr;
    if (cam_settle_ovr >= 0)
      c.cam_settle = (cam_settle_ovr != 0);
    if (b2_cert_ovr >= 0)
      c.b2_cert = (b2_cert_ovr != 0);
    if (a_candidate_ovr >= 0)
      c.a_candidate = true;
    if (a_gate_mode_ovr >= 0)
      c.a_gate_mode = a_gate_mode_ovr;
    if (p4_ovr >= 0)
      c.p4 = true;
    if (conv_tol_ovr > 0.0)
      c.joint.conv_tol_rel = conv_tol_ovr;
    if (max_clones_ovr > 0)
      c.harvester.max_clones = max_clones_ovr;
    if (max_track_ovr > 0)
      c.harvester.max_track_len = max_track_ovr;
    if (a_carry_ovr >= 0)
      c.a_carry = true;
    if (fused_iters_ovr > 0)
      c.joint.fused_iters = fused_iters_ovr;
    if (duel_accept_ovr >= 0)
      c.joint.duel_on_accept = true;
    if (eoa_ovr >= 0)
      c.joint.export_on_accept = (eoa_ovr != 0);
    if (threads_ovr > 0)
      c.joint.num_threads = threads_ovr;
    if (outer_iters_ovr > 0 || window_iters_ovr > 0 || select_k_ovr > 0 || max_window_ovr > 0.0 || cam_alt_ovr >= 0 ||
        verify_min_ovr >= 0.0 || solve_budget_ovr >= 0.0 || cam_settle_ovr >= 0)
      std::printf("[cfg] experiment overrides: outer=%d win_iters=%d select_K=%d max_window=%.2fs cam_alt=%d verify_min=%.3f budget=%.1fs\n",
                  c.joint.outer_iterations, c.joint.window_max_iters, c.select_K, c.harvester.max_window_s, c.cam_alt_rounds,
                  c.verify_min_improve, c.solve_budget_s);
  };

  if (selftest) {
    SharedCalib c;
    if (!write_calib_yaml(out_yaml, c)) {
      std::printf("writeback failed\n");
      return 1;
    }
    std::printf("wrote seed-identity %s (run the ctest gates for the full pipeline checks)\n", out_yaml.c_str());
    return 0;
  }

  // ---- Seed overrides are built ONCE, above every ingest branch. They apply at RECORD-REPLAY
  // time inside run_replay (the record's seed is patched; the record itself stays pristine and
  // byte-replayable to the blind run), and the voxl path replays the record it just
  // converted -- so the SAME flags work identically on --replay and --voxl. They used to
  // be wired into --replay only, and --voxl SILENTLY IGNORED them: the 2026-07-13 Tg matrix ran 9
  // legs that were all byte-identical blind runs. An explicit operator flag must bind or refuse,
  // never no-op. ----
  ImuIntrinsicModel imu_ovr;
  if (have_imu_seed) {
    for (int k = 0; k < 6; ++k)
      imu_ovr.dw(k) = imu_seed[k];
    for (int k = 0; k < 6; ++k)
      imu_ovr.da(k) = imu_seed[6 + k];
    imu_ovr.q_AtoI = Eigen::Vector4d(imu_seed[12], imu_seed[13], imu_seed[14], imu_seed[15]);
    imu_ovr.q_AtoI.normalize();
    if (no_imu_intrinsics) // seeded AND frozen: trust the factory chain, solve ext/td only
      imu_ovr.calib_dw = imu_ovr.calib_da = imu_ovr.calib_RAtoI = false;
    std::printf("[replay] IMU intrinsics SEEDED (imu2 gauge)%s: Dw diag [%.5f %.5f %.5f] Da diag [%.5f %.5f %.5f] R_AtoI %.3f deg\n",
                no_imu_intrinsics ? " + FROZEN" : "", imu_ovr.dw(0), imu_ovr.dw(2), imu_ovr.dw(5), imu_ovr.da(0), imu_ovr.da(2),
                imu_ovr.da(5), 2.0 * imu_ovr.q_AtoI.head<3>().norm() * 180.0 / M_PI);
  }
  Eigen::Vector4d q_seed(seed_rot[0], seed_rot[1], seed_rot[2], seed_rot[3]);
  q_seed.normalize();
  const Eigen::Vector3d p_seed(seed_pos[0], seed_pos[1], seed_pos[2]);
  Eigen::Matrix3d Tg_ovr;
  Tg_ovr << tg_seed[0], tg_seed[1], tg_seed[2], tg_seed[3], tg_seed[4], tg_seed[5], tg_seed[6], tg_seed[7], tg_seed[8];
  ImuNoise noise_ovr_v;
  if (have_noise_ovr) {
    noise_ovr_v.sigma_w = noise_ovr[0];
    noise_ovr_v.sigma_wb = noise_ovr[1];
    noise_ovr_v.sigma_a = noise_ovr[2];
    noise_ovr_v.sigma_ab = noise_ovr[3];
  }
  CalibSessionRunner::SeedOverride ovr;
  ovr.imu = have_imu_seed ? &imu_ovr : nullptr;
  ovr.Tg = have_tg ? &Tg_ovr : nullptr;
  ovr.noise = have_noise_ovr ? &noise_ovr_v : nullptr;
  ovr.q_ItoC = have_seed_rot ? &q_seed : nullptr;
  ovr.p_IinC = have_seed_pos ? &p_seed : nullptr;
  ovr.td = have_seed_td ? &seed_td_v : nullptr;
  if (have_seed_rot || have_seed_pos || have_seed_td)
    std::printf("[replay] EXTRINSIC seeds: R_ItoC %s | p_IinC %s | td %s (unseeded blocks stay BLIND)\n",
                have_seed_rot ? "chain" : "blind", have_seed_pos ? "chain" : "blind", have_seed_td ? "chain" : "blind");
  if (have_tg)
    std::printf("[replay] Tg SEEDED (imu2 gauge): %.3f deg/s @1g\n", Tg_ovr.rowwise().norm().maxCoeff() * 9.81 * 180.0 / M_PI);
  if (have_noise_ovr)
    std::printf("[replay] calib-weighting noise OVERRIDE: w %.4e wb %.4e a %.4e ab %.4e\n", noise_ovr_v.sigma_w, noise_ovr_v.sigma_wb,
                noise_ovr_v.sigma_a, noise_ovr_v.sigma_ab);
  const bool any_ovr = have_imu_seed || have_tg || have_noise_ovr || have_seed_rot || have_seed_pos || have_seed_td;
  const CalibSessionRunner::SeedOverride *ovr_p = any_ovr ? &ovr : nullptr;


  if (!voxl_dir.empty()) {
#ifdef DISABLE_TRACK_KLT
    // Device build: the raw-log ingest front-end (VoxlLogFeeder/TrackKLT) is
    // host-only. On target the server records sessions in-process; this CLI
    // exists there for --replay of those records.
    std::printf("--voxl log ingest is host-only (device build strips TrackKLT); use --replay <session.bin>\n");
    return 1;
#else
    // voxl-logger MPA log (log000N/): the log is self-describing — per-unit
    // fisheye intrinsics ship inside it; the mechanical extrinsic seed comes
    // from --ext-rpy/--ext-t in the vcc convention, composed exactly as
    // VoxlConfigure does (R_ItoC = R_child_to_parent^T, p_IinC = -R_ItoC p_CinI).
    // The VOXL base profile + flight overlay live in SessionProfiles.h — ONE
    // definition shared verbatim with the in-server live mode
    // (voxl-open-vins-server --calibrate), so host studies and device
    // sessions can never drift apart. Evidence trail for every value is
    // in-header.
    apply_voxl_profile(cfg);
    if (flight) {
      apply_flight_overlay(cfg, /*set_cam_mode0=*/!have_cam_mode);
      std::printf("[voxl] FLIGHT profile overlay active (pinned gates; cam_mode=%d)\n", cfg.cam_mode);
    }
    SessionSeed seed;
    if (!VoxlLogFeeder::load_seed_intrinsics(voxl_dir, cam_pipe, seed))
      return 1;
    // BLIND protocol (operator directive): extrinsics.conf is the evaluation
    // TRUTH — never seed from it. The pipeline starts with NO mechanical
    // prior (identity rotation, zero lever arm) and must EARN R_ItoC/td in
    // bootstrap and p_IinC in the joint solve. The reference comes from the
    // log's own extrinsics.conf snapshot (per-SKU correct by construction);
    // --ext-rpy/--ext-t override it, the Stinger built-ins are last resort.
    VoxlLogFeeder::RefExtrinsics refx;
    const bool have_conf_ref = VoxlLogFeeder::load_ref_extrinsics(voxl_dir, cam_pipe, refx);
    const char *ref_src = (have_ext_rpy || have_ext_t) ? "--ext-rpy/--ext-t" : "built-in Stinger defaults";
    if (have_conf_ref && refx.have_cam) {
      if (!have_ext_rpy && !have_ext_t)
        ref_src = "log extrinsics.conf";
      if (!have_ext_rpy)
        for (int k = 0; k < 3; ++k)
          ext_rpy[k] = refx.cam_rpy[k];
      if (!have_ext_t)
        for (int k = 0; k < 3; ++k)
          ext_t[k] = refx.cam_t[k];
    }
    std::printf("[voxl] reference extrinsics (%s): RPY [%g %g %g] T [%g %g %g]%s\n", ref_src, ext_rpy[0], ext_rpy[1], ext_rpy[2],
                ext_t[0], ext_t[1], ext_t[2],
                refx.have_body ? "" : " (no body->imu_apps entry: body-frame print assumes Rx180)");
    const double d2r = M_PI / 180.0;
    const Eigen::Matrix3d R_cp = (Eigen::AngleAxisd(ext_rpy[0] * d2r, Eigen::Vector3d::UnitX()) *
                                  Eigen::AngleAxisd(ext_rpy[1] * d2r, Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(ext_rpy[2] * d2r, Eigen::Vector3d::UnitZ()))
                                     .toRotationMatrix();
    const Eigen::Matrix3d R_ItoC_ref = R_cp.transpose(); // reference ONLY, never seeded
    seed.calib.cams[0].q_ItoC = Eigen::Vector4d(0, 0, 0, 1);
    seed.calib.cams[0].p_IinC = Eigen::Vector3d::Zero();
    seed.calib.cams[0].td = 0.0;
    seed.calib.grav_mag = grav_mag;
    // ICM-class 1 kHz imu_apps: the platform kalibr-chain inflated Allan densities
    // (bench defaults are 6-120x optimistic for this hardware class); --noise
    // overrides for weighting studies (e.g. the raw un-inflated Allan fit)
    apply_voxl_noise_defaults(seed.calib.noise);
    if (have_noise_ovr) {
      seed.calib.noise.sigma_w = noise_ovr[0];
      seed.calib.noise.sigma_wb = noise_ovr[1];
      seed.calib.noise.sigma_a = noise_ovr[2];
      seed.calib.noise.sigma_ab = noise_ovr[3];
      std::printf("[voxl] IMU noise override: w=%.4e wb=%.4e a=%.4e ab=%.4e\n", noise_ovr[0], noise_ovr[1], noise_ovr[2], noise_ovr[3]);
    }
    seed.tr_hw_seed.assign(1, tr_seed);
    seed.calib.cams[0].rolling = (tr_seed > 0.0); // AR0144-class tracking cams are global shutter: 0
    if (no_imu_intrinsics) {
      // Attribution/ablation: freeze Dw/Da/R_AtoI at the seed (identity) so the
      // solve estimates ext/td/cam only. On weakly-excited data the IMU
      // intrinsics are barely identified and their co-estimation can BEND the
      // extrinsics through the A1 coupling (T-RO Sec-13 schedule made a flag).
      seed.calib.imu.calib_dw = seed.calib.imu.calib_da = seed.calib.imu.calib_RAtoI = false;
      std::printf("[voxl] IMU intrinsics FROZEN at seed (--no-imu-intrinsics)\n");
    }
    const std::string rec = out_yaml + ".voxl_session.bin";
    double mean_exp_s = 0.0;
    apply_cli_overrides(cfg);
    // The record carries the config it ran under (v2), so --replay of this bin
    // reproduces THIS session exactly instead of scoring library defaults.
    seed.profile = flight ? SessionProfile::VOXL_FLIGHT : SessionProfile::VOXL;
    seed.cam_mode = cfg.cam_mode;
    if (!VoxlLogFeeder::convert(voxl_dir, cam_pipe, seed, rec, max_sec, 150, &mean_exp_s))
      return 1;
    SessionReport rep;
    if (!CalibSessionRunner::run_replay(rec, cfg, rep, ovr_p)) {
      std::printf("replay of the converted record failed\n");
      return 1;
    }
    print_report(rep);
    // deltas vs the extrinsics.conf REFERENCE (the evaluation truth; NOT seeded)
    const Eigen::Matrix3d R_est = ov_core::quat_2_Rot(rep.committed.cams[0].q_ItoC);
    const double ang =
        std::acos(std::min(1.0, std::max(-1.0, ((R_est * R_ItoC_ref.transpose()).trace() - 1.0) / 2.0))) * 180.0 / M_PI;
    const Eigen::Vector3d p_CinI_est = -R_est.transpose() * rep.committed.cams[0].p_IinC;
    const Eigen::Vector3d dp_ref = p_CinI_est - Eigen::Vector3d(ext_t[0], ext_t[1], ext_t[2]);
    std::printf("[voxl] committed vs extrinsics.conf REFERENCE (blind, unseeded): dR %.3f deg | dp %.1f mm [%.1f %.1f %.1f] | td %.3f ms\n",
                ang, 1e3 * dp_ref.norm(), 1e3 * dp_ref(0), 1e3 * dp_ref(1), 1e3 * dp_ref(2), 1e3 * rep.committed.cams[0].td);
    // FRAME re-expressions for external comparison: the logged imu_apps stream
    // is the imu-server COMMON frame (imu0_rotate_common_frame). References
    // expressed against the vehicle BODY frame differ by the vcc body->imu_apps
    // rotation — PER RIG, from the log's conf (Stinger [180,0,0], Starling2
    // identity): print both so a kalibr/body-frame truth compares against the
    // right matrix. v_cam = R_ItoC * R_BtoI * v_body with R_BtoI =
    // (RxRyRz(body_rpy))^T in the vcc child->parent composition.
    {
      Eigen::Matrix3d R_BtoI = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix(); // legacy Rx180 fallback
      char bsrc[64] = "assumed Rx180";
      if (refx.have_body) {
        R_BtoI = (Eigen::AngleAxisd(refx.body_rpy[0] * d2r, Eigen::Vector3d::UnitX()) *
                  Eigen::AngleAxisd(refx.body_rpy[1] * d2r, Eigen::Vector3d::UnitY()) *
                  Eigen::AngleAxisd(refx.body_rpy[2] * d2r, Eigen::Vector3d::UnitZ()))
                     .toRotationMatrix()
                     .transpose();
        std::snprintf(bsrc, sizeof(bsrc), "conf RPY [%g %g %g]", refx.body_rpy[0], refx.body_rpy[1], refx.body_rpy[2]);
      }
      const Eigen::Matrix3d R_BtoC = R_est * R_BtoI;
      Eigen::IOFormat fmt(9, 0, ", ", "; ", "", "", "[", "]");
      std::cout << "[voxl] R_ItoC (imu_apps/common frame): " << R_est.format(fmt) << std::endl;
      std::cout << "[voxl] R_BtoC (body frame, via body->imu " << bsrc << "): " << R_BtoC.format(fmt) << std::endl;
      std::cout << "[voxl] p_IinC: " << rep.committed.cams[0].p_IinC.format(fmt) << "  (T_cam_wrt_imu = -R^T p = "
                << (-R_est.transpose() * rep.committed.cams[0].p_IinC).format(fmt) << ")" << std::endl;
    }
    // CONVENTION bridges for external (kalibr-style) comparison. Two silent
    // frame/clock disagreements measured to dominate naive scoring:
    //  (1) td: we stamp clones at MID-exposure (the optical instant); chains
    //      that keep start-of-exposure stamps leave exposure/2 inside td.
    //  (2) gauge: our IMU frame is GYRO-aligned (R_GtoI = I structural, Dw
    //      upper-tri closes the gauge); kalibr's body is ACCEL-aligned (M_a
    //      lower-tri, gyro carries M_w * C_gyro_i). The rotation between the
    //      gauges is PHYSICAL (gyro-vs-accel die misalignment, ~0.8 deg on
    //      ICM-class parts) and lives in our accel chain: with the forward map
    //      F_a = Da^-1 * R_AtoI^T (body -> accel measurement), LQ(F_a) = L * S
    //      gives R_ItoC^(accel-gauge) = R_ItoC * S^T. S is only as good as the
    //      estimated accel chain — when da/q_AtoI are gate-frozen, S = I and
    //      the gyro-gauge matrix IS the deliverable (VIO consumes Dw with it).
    {
      std::printf("[voxl] td (kalibr start-of-exposure convention): %.6f s = committed %.6f + mean_exposure/2 %.6f\n",
                  rep.committed.cams[0].td + 0.5 * mean_exp_s, rep.committed.cams[0].td, 0.5 * mean_exp_s);
      const Eigen::Matrix3d Da_c = ImuIntrinsicModel::ut(rep.committed.imu.da);
      const Eigen::Matrix3d R_A = ov_core::quat_2_Rot(rep.committed.imu.q_AtoI);
      const Eigen::Matrix3d Fa = Da_c.inverse() * R_A.transpose();
      Eigen::HouseholderQR<Eigen::Matrix3d> qr(Eigen::Matrix3d(Fa.transpose())); // Fa^T = Q~ R~  =>  Fa = R~^T Q~^T = L S
      Eigen::Matrix3d S = Eigen::Matrix3d(qr.householderQ()).transpose();
      Eigen::Matrix3d L = Eigen::Matrix3d(qr.matrixQR().triangularView<Eigen::Upper>()).transpose();
      for (int k = 0; k < 3; ++k)
        if (L(k, k) < 0) { // sign-canonical: positive scale diagonal, proper rotation
          L.col(k) *= -1.0;
          S.row(k) *= -1.0;
        }
      const double s_ang = std::acos(std::min(1.0, std::max(-1.0, (S.trace() - 1.0) / 2.0))) * 180.0 / M_PI;
      Eigen::IOFormat fmt(9, 0, ", ", "; ", "", "", "[", "]");
      if (s_ang < 0.02) {
        std::printf("[voxl] R_ItoC accel-gauge == gyro-gauge (accel chain frozen or aligned; gauge rotation %.3f deg)\n", s_ang);
      } else {
        const Eigen::Matrix3d R_acc = ov_core::quat_2_Rot(rep.committed.cams[0].q_ItoC) * S.transpose();
        std::printf("[voxl] R_ItoC (ACCEL-aligned/kalibr gauge, S=%.3f deg; inherits accel-chain error): ", s_ang);
        std::cout << R_acc.format(fmt) << std::endl;
      }
    }
    return rep.final_state == RunnerState::DONE ? 0 : 2;
#endif // DISABLE_TRACK_KLT
  }

  if (!replay_path.empty()) {
    // The record is SELF-DESCRIBING (v2): rebuild the producing session's
    // profile from its header before solving, so a device session
    // (voxl-open-vins-server --calibrate) replays here under the SAME
    // estimator it ran with on target. v1 records carry no tag and keep the
    // historical behavior (library defaults + explicit CLI flags). Explicit
    // CLI flags still win — they are applied after.
    SessionSeed hdr;
    if (!read_session_header(replay_path, hdr)) {
      std::printf("cannot open session record %s\n", replay_path.c_str());
      return 1;
    }
    SessionProfile prof = hdr.profile;
    const bool tagged = (hdr.profile != SessionProfile::LIBRARY);
    if (!profile_ovr.empty()) {
      if (profile_ovr == "voxl")
        prof = SessionProfile::VOXL;
      else if (profile_ovr == "flight")
        prof = SessionProfile::VOXL_FLIGHT;
      else if (profile_ovr == "library")
        prof = SessionProfile::LIBRARY;
      else {
        std::printf("unknown --profile '%s' (voxl|flight|library)\n", profile_ovr.c_str());
        return 1;
      }
    } else if (flight && hdr.profile == SessionProfile::VOXL) {
      prof = SessionProfile::VOXL_FLIGHT; // operator overlay on a base-profile record
    }
    const char *pname = apply_profile_tag(cfg, prof, have_cam_mode ? cam_mode : hdr.cam_mode);
    if (!pname) {
      std::printf("euroc-tagged records are not supported by ov_zcalibrate\n");
      return 1;
    }
    std::printf("[replay] profile: %s (cam_mode %d) -- %s\n", pname, cfg.cam_mode,
                !profile_ovr.empty() ? "from --profile (overrides the record)"
                : tagged             ? "FROM THE RECORD (self-describing, v2)"
                                     : "v1 record carries no profile tag: pass --profile voxl|flight to reproduce it");
    SessionReport rep;
    apply_cli_overrides(cfg);
    if (!CalibSessionRunner::run_replay(replay_path, cfg, rep, ovr_p)) {
      std::printf("cannot open session record %s\n", replay_path.c_str());
      return 1;
    }
    print_report(rep);
    return rep.final_state == RunnerState::DONE ? 0 : 2;
  }

  std::printf("usage: ov_zcalibrate --replay <session.bin> | --voxl <log_dir> | --selftest [options]\n");
  std::printf("(live device sessions run in-process: voxl-open-vins-server --calibrate)\n");
  return 1;
}
