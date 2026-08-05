/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: Monte-Carlo calibration harness for the Wald accel gate
 * (DESIGNS #1 test-plan item; SOLVER_TIME_METHOD / HANDOFF s7 blocker).
 *
 * H0 study: replicate synthetic sessions at TRUTH (stationary parameter,
 * correct model) with per-replicate measurement-noise seeds, run the full
 * production replay path with the Wald gate DECIDING (a_gate_mode=1), and
 * collect (T, r). Under H0 with honestly-dispersed exported information,
 * E[T_raw] = r; the measured over-claim makes E[T_raw] = kappa_true * r, so
 *
 *     kappa_hat = mean( T * kap_eff / r ),   kap_eff = max(floor, kap_sess)
 *
 * (T is computed deflated: C = kap_eff (A1^-1 + A2^-1); multiplying back
 * recovers the raw statistic). kappa_hat pins the a_info_deflate FLOOR PER
 * SHAPE — the 2026-07-11 thin-shape discordant pair (split freezes, wald
 * CONSISTENT via GN-linearized understatement) is exactly the failure class
 * this harness exists to bound before mode 1 takes profile authority. The H0
 * freeze rate estimates the test's size; the --h1 junk study its power. The
 * per-replicate stat line carries everything an OFFLINE (floor, scale)
 * re-sizer needs (T and the X thresholds rescale analytically in kap_eff and
 * a_wald_thresh_scale), so a full ROC sweep costs ONE collection run per
 * hypothesis — run it with --floor set to the sweep's LOWEST floor, so
 * observability reclassification (raw_eig / floor' >= a_obs_min_eig) only
 * ever REMOVES directions and every cell's verdict reconstructs exactly.
 * N here is small, so ctest asserts only loose sanity — the real study runs
 * standalone with --reps >= 30.
 *
 * The record generator is duplicated VERBATIM from test_session_e2e.cpp
 * (rich-excitation path only) so this harness cannot perturb the S-suite.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "utils/SessionRecord.h"
#include "core/CalibSessionRunner.h"
#include "sim/SynthWorld.h"
#include "utils/quat_ops.h"

using namespace ov_zcalib;

static int failures = 0;
#define CHECK(cond, ...)                                                                                                                   \
  do {                                                                                                                                     \
    if (!(cond)) {                                                                                                                         \
      std::printf("[FAIL] %s:%d: ", __FILE__, __LINE__);                                                                                   \
      std::printf(__VA_ARGS__);                                                                                                            \
      std::printf("\n");                                                                                                                   \
      failures++;                                                                                                                          \
    }                                                                                                                                      \
  } while (0)

// Duplicated from test_session_e2e.cpp (rich-excitation path; provenance
// comment there) — a shared testlib refactor is deliberate future work so the
// MC harness lands with zero S-suite blast radius.
static void write_session_record(const synth::Truth &tr, const std::string &path, double dur, double ex0, double ex1, unsigned rng,
                                 double h1_gyro_ramp = 0.0, double h1_accel_settle = 0.0) {
  synth::Trajectory tj;
  tj.excite_t0 = ex0;
  tj.excite_t1 = ex1;
  synth::StreamOptions so;
  so.dur = dur;
  so.excite = {{ex0, ex1}};
  std::vector<RawImu> imu;
  std::vector<FrameObs> frames;
  synth::make_streams(tr, tj, so, rng, imu, frames);
  if ((h1_gyro_ramp > 0.0 || h1_accel_settle > 0.0) && !imu.empty()) {
    // H1 junk injection (thermal-transient class): a slow gyro-bias ramp PLUS accel-bias
    // exponential transients. The original gyro-only 2e-5 rad/s/s ramp was measured NOT to be a
    // falsifier of the v3 class: per-window bg nuisances absorb it almost entirely (implied
    // dqA ~0.05-0.1 deg, 8/12 junk-passes at the 2026-07-13 sizing). The ACCEL term is the
    // mechanism the gate historically caught (v3 real-log freeze, dqA 0.867 deg): a body-frame
    // accel-bias error couples into qA/da through gravity leverage (thermal-VRE). What per-window
    // ba CANNOT absorb is transient CURVATURE; a single warmup settle parks that entirely in the
    // first half (measured here: dqA saturates ~0.33 deg and the joint A1a point absorbs the
    // settled tail — at 2x amplitude the verdict even flips to a junk-PASS). So the injection is
    // a transient PAIR: the warmup settle plus a SECOND onset at mid-session on an orthogonal
    // body axis (the thermal-event class: payload/motor power step, airflow change, VRE onset) —
    // each half then absorbs ITS transient through its own excitation geometry and the halves
    // disagree at the size of the junk. Magnitude is pinned so the implied half-disagreement
    // lands AT the v3 class (printed dqA ~0.6-0.9 deg on the gate subspace), not at a
    // nuisance-absorbable token nor at the phys-ceiling (>2 deg) where every cell trivially
    // refuses and the ROC cannot discriminate.
    const Eigen::Vector3d axis_w = Eigen::Vector3d(0.6, -0.5, 0.62).normalized();
    const Eigen::Vector3d axis_a = Eigen::Vector3d(0.55, 0.60, -0.58).normalized();
    const Eigen::Vector3d axis_a2 = axis_w.cross(axis_a).normalized(); // second thermal axis, skewed off the first
    const double t0 = imu.front().timestamp, tau = 25.0;
    const double t_mid = t0 + 0.5 * dur; // second onset at the gate's own half boundary (worst case for conditioning)
    for (RawImu &si : imu) {
      const double dt = si.timestamp - t0;
      si.wm += h1_gyro_ramp * dt * axis_w;
      si.am += h1_accel_settle * (1.0 - std::exp(-dt / tau)) * axis_a;
      if (si.timestamp > t_mid)
        si.am += h1_accel_settle * (1.0 - std::exp(-(si.timestamp - t_mid) / tau)) * axis_a2;
    }
  }

  SessionSeed ss;
  ss.calib = SharedCalib();
  Eigen::Vector4d dq;
  dq << 0.5 * 0.012, 0.5 * 0.009, 0.5 * -0.015, 1.0;
  ss.calib.cams[0].q_ItoC = ov_core::quat_multiply(dq / dq.norm(), tr.q_ItoC);
  ss.calib.cams[0].p_IinC = tr.p_IinC + Eigen::Vector3d(0.006, -0.004, 0.005);
  ss.calib.cams[0].cam = tr.cam;
  ss.calib.cams[0].td = 0.0;
  ss.calib.cams[0].tr = 0.0;
  ss.calib.cams[0].img_w = tr.img_w;
  ss.calib.cams[0].img_h = tr.img_h;

  SessionRecordWriter wr;
  wr.open(path, ss);
  size_t ii = 0;
  for (const FrameObs &f : frames) {
    while (ii < imu.size() && imu[ii].timestamp <= f.timestamp + 0.05)
      wr.write_imu(imu[ii++]);
    wr.write_frame(f);
  }
  while (ii < imu.size())
    wr.write_imu(imu[ii++]);
  wr.close();
}

int main(int argc, char **argv) {
  int reps = 3;  // ctest default: loose sanity at suite-compatible cost; the real study runs --reps >= 30
  int rep0 = 0;  // first replicate index (seed offset): shards a standalone study across processes
  bool p4 = false, h1 = false;
  double floor_ovr = -1.0, scale_ovr = -1.0, h1a_ovr = -1.0;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--reps") && i + 1 < argc)
      reps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--rep0") && i + 1 < argc)
      rep0 = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--p4"))
      p4 = true; // wald-verdict stability under the P4 solver (the composed-endgame validation)
    else if (!std::strcmp(argv[i], "--h1"))
      h1 = true; // junk-injection POWER study: the gate must refuse every replicate
    else if (!std::strcmp(argv[i], "--floor") && i + 1 < argc)
      floor_ovr = std::atof(argv[++i]); // a_info_deflate override (ROC collection runs at the sweep's lowest floor)
    else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc)
      scale_ovr = std::atof(argv[++i]); // a_wald_thresh_scale override
    else if (!std::strcmp(argv[i], "--h1a") && i + 1 < argc)
      h1a_ovr = std::atof(argv[++i]); // accel-settle amplitude override [m/s^2] (injection tuning only)
  }
  // H1 = the v3-class thermal transient: gyro ramp (2e-5 rad/s/s ~= 0.10 deg/s wander over the
  // session — absorbable alone, kept as the class's gyro component) + accel-bias transient pair
  // whose amplitude is pinned so the implied half-disagreement prints dqA ~0.6-0.9 deg (v3
  // measured 0.867 on real data). MEASURED on the seed-0 replicate: 0.12 -> dqA 0.51, 0.17 ->
  // 0.65, 0.20 -> 0.66 with kap_sess inflating 4.7 -> 5.3 (the response saturates into common
  // absorption while self-widening the threshold) — 0.17 is the v3-class point and the sharper
  // falsifier (T 45.8 vs 42.1). See write_session_record for the mechanism.
  const double h1_gyro = h1 ? 2e-5 : 0.0;
  const double h1_accel = h1 ? (h1a_ovr > 0.0 ? h1a_ovr : 0.17) : 0.0;

  const synth::Truth tr = synth::make_truth();
  std::vector<double> khat;
  int frozen = 0, unobservable = 0, aborted = 0;

  for (int rep = rep0; rep < rep0 + reps; ++rep) {
    const std::string rec = "/tmp/ov_zcalib_wald_mc_" + std::to_string(rep) + ".bin";
    write_session_record(tr, rec, 90.0, 5.0, 88.0, 9001u + 131u * (unsigned)rep, h1_gyro, h1_accel);

    SessionConfig cfg;
    cfg.harvester.pix_sigma = 0.5;
    cfg.out_yaml = "/tmp/ov_zcalib_wald_mc_" + std::to_string(rep) + ".yaml"; // per-rep: shards must not collide
    cfg.verbose = false;
    cfg.joint.verbose = false;
    cfg.cam_mode = 0;
    cfg.a_gate_mode = 1; // wald DECIDES: per-replicate cost drops (no halves) and the mode-1 H0 size is measured directly
    cfg.p4 = p4;
    if (floor_ovr > 0.0)
      cfg.a_info_deflate = floor_ovr;
    if (scale_ovr > 0.0)
      cfg.a_wald_thresh_scale = scale_ovr;

    SessionReport rp;
    const bool ok = CalibSessionRunner::run_replay(rec, cfg, rp);
    std::remove(rec.c_str());
    for (const char *sfx : {"", ".rollback", ".voxl_session.bin"})
      std::remove((cfg.out_yaml + sfx).c_str()); // tmp hygiene: no session bins left behind
    if (!ok || rp.final_state != RunnerState::DONE) {
      aborted++;
      std::printf("[mc %02d] ABORT (%s)\n", rep, rp.abort_reason.c_str());
      continue;
    }
    using V = SessionReport::AccelGateVerdict;
    if (rp.a_wald_verdict == V::WALD_UNOBSERVABLE || rp.a_wald_r <= 0) {
      unobservable++;
      std::printf("[mc %02d] UNOBSERVABLE r=%d mineig=%.6g\n", rep, rp.a_wald_r, rp.a_wald_min_eig);
      continue;
    }
    // kappa_hat per replicate: T is computed DEFLATED by kap_eff = max(floor, kap_sess) — the
    // session estimate, not the configured floor, is what divided T whenever it exceeded the
    // floor. Multiplying kap_eff back recovers T_raw; the old `T * floor / r` silently
    // under-reported kappa on every floor-unbound replicate.
    const double kap_eff = std::max(cfg.a_info_deflate, rp.a_wald_kappa);
    const double k = rp.a_wald_T * kap_eff / std::max(1, rp.a_wald_r);
    khat.push_back(k);
    if (!rp.a_full_open)
      frozen++;
    // One machine-parsable line per replicate: everything the offline (floor, scale) re-sizer
    // needs. T rescales as T' = T * kap_eff / max(floor', kap_sess); xthr' = xthr * scale' *
    // max(floor', kap_sess) / kap_eff; Tthr' from (r, kdf); mineig reclassifies as raw/floor'
    // (raw = mineig * floor_run); dqA/dda replay the deadband, jqa/jda the phys ceiling.
    std::printf("[mc %02d] verdict=%s T=%.8g r=%d kdf=%d kap_sess=%.8g kap_eff=%.8g X12=%.8g X21=%.8g xthr1=%.8g xthr2=%.8g "
                "mineig=%.6g dqA=%.4f dda=%.6g jqa=%.4f jda=%.6g kappa_rep=%.2f\n",
                rep, rp.a_full_open ? "CONSISTENT" : "FROZEN", rp.a_wald_T, rp.a_wald_r, rp.a_wald_df, rp.a_wald_kappa, kap_eff,
                rp.a_wald_x12, rp.a_wald_x21, rp.a_wald_xthr1, rp.a_wald_xthr2, rp.a_wald_min_eig, rp.a_wald_dqa_deg, rp.a_wald_dda_off,
                rp.a_wald_jqa_deg, rp.a_wald_jda, k);
  }

  double kmean = 0.0, kmin = 1e9, kmax = 0.0;
  for (double k : khat) {
    kmean += k;
    kmin = std::min(kmin, k);
    kmax = std::max(kmax, k);
  }
  if (!khat.empty())
    kmean /= (double)khat.size();
  std::printf("[mc] reps=%d evaluated=%zu kappa_hat mean=%.2f [%.2f, %.2f] | H0 freeze=%d unobservable=%d abort=%d\n", reps,
              khat.size(), kmean, khat.empty() ? 0.0 : kmin, khat.empty() ? 0.0 : kmax, frozen, unobservable, aborted);

  // ctest sanity gates (loose by design at small N; the standalone study is the calibrator):
  CHECK(aborted == 0, "replicates must complete (%d aborted)", aborted);
  if (h1) {
    // POWER gate: injected junk must never unlock the chain. FROZEN and UNOBSERVABLE both
    // count as detection (freeze-as-abstention doctrine); only CONSISTENT is a junk-pass.
    const int unlocked = (int)khat.size() - frozen;
    CHECK(unlocked == 0, "H1 junk-pass: %d replicate(s) unlocked under an injected bias ramp", unlocked);
  } else {
    CHECK(!khat.empty(), "no replicate reached the wald statistics");
    CHECK(kmean > 0.2 && kmean < 40.0, "kappa_hat %.2f outside sanity range", kmean);
  }

  if (failures == 0)
    std::printf("test_wald_mc: ALL PASS (kappa_hat %.2f over %zu H0 replicates)\n", kmean, khat.size());
  else
    std::printf("test_wald_mc: %d FAILURES\n", failures);
  return failures == 0 ? 0 : 1;
}
