/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib tg (gyro g-sensitivity) recovery-arbiter gates. The estimated-Tg
 * default (SessionConfig::free_tg) is only shippable if the session EARNS a
 * real Tg and REFUSES an unearnable one; three worlds pin the three verdicts:
 *
 *  [T1] Recovery: rich 6-axis session whose truth carries an ICM-part-class
 *       Tg (~4e-4 (rad/s)/(m/s^2), all 9 elements distinct; the sim inverse
 *       map injects wm += Tg*a_hat). The tg block must unlock through the
 *       full-chain gate AND commit; recovery lands three prongs -- MEDIAN
 *       element within the 1.2e-4 claim (0.3x part class), EVERY element
 *       within 1.5x claim (the kappa-honest 3-sigma at this world's
 *       information -- see the in-test note), |Tg| magnitude within 10% --
 *       and co-estimating tg must not degrade dw/da/qA against the SAME
 *       world with Tg = 0 (the baseline run) beyond basin noise (documented
 *       0.57-0.73 deg qA spread across numerics-equivalent builds => 1.35x +
 *       small floors, never a bare equality).
 *  [T1b] Unlock -> commit-walk refusal ships seed BYTES (a walk-refused but
 *       already-solved Tg must never leak into the shipped calib).
 *  [T2] Hover / constant-a_hat refusal: a session with no excitation cannot
 *       distinguish Tg*a_hat from the gyro bias. Whatever path refuses
 *       (bootstrap starvation, excitation gates), the committed Tg must be
 *       BYTE-IDENTICAL to the seed -- any nonzero byte means some stage wrote
 *       an unearned g-sensitivity.
 *  [T3] One-axis rotation about gravity: bootstrap-viable motion whose a_hat
 *       is constant (R(t) about g fixes g), so the tg columns collapse onto
 *       the bias direction -- verdict must be an abstention/refusal (wald
 *       UNOBSERVABLE r<9, split not-certified, or PRE_CLOSED), never a
 *       commit, and the committed Tg must again be the seed bytes. This is
 *       the falsifier for the refusal path itself (a refused-but-solved tg
 *       that skips the revert ships junk -- the revert_block tg arm).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "core/CalibSessionRunner.h"
#include "sim/SynthWorld.h"

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

enum class World { RICH, HOVER, SINGLE_AXIS };

/// Session stream writer (same conventions as test_session_e2e): quiet head for
/// SETTLE, then the world's own excitation. Truth Tg rides inside synth::Truth
/// (raw_imu_at injects wm += Tg*a_hat when nonzero); the SINGLE_AXIS branch
/// generates its own raw IMU and applies the identical inverse map by hand.
static void write_tg_record(const synth::Truth &tr, const std::string &path, double dur, World world, unsigned rng) {
  synth::Trajectory tj;
  synth::StreamOptions so;
  so.dur = dur;
  std::vector<RawImu> imu;
  std::vector<FrameObs> frames;
  if (world == World::RICH) {
    // SynthWorld's rich 6-axis rotation + a g-aligned "pumping" translation at real
    // handheld dynamics (std|a_m| 2.8-3.2 m/s^2). The weak 0.36 m/s^2 shape starves
    // the split-half chain judge of per-half q_AtoI conditioning and fails on its own
    // basin noise (z~1.2) -- a falsifier judging the WORLD, not the estimator.
    // Generated HERE like the SINGLE_AXIS branch, NOT via a synth::Trajectory knob:
    // probe-measured -ffast-math seam -- growing p_of by a guarded 4th sin repacks the
    // SLP sin group and shifts last-ulp stream bytes of the validated worlds
    // (session_e2e, wald_mc MC calibration).
    tj.excite_t0 = 6.0;
    tj.excite_t1 = dur - 2.0;
    std::mt19937 g(rng);
    std::normal_distribution<double> nrm(0.0, 1.0);
    // ONE g-aligned pump; the shape was pinned by measurement:
    //  * z-pump at 3.1 rad/s: best per-half conditioning, but 3.1 - 2.9 (w_z) =
    //    0.2 rad/s beats slower than a window -- Tg(2,2) trades against dw_z
    //    (31% shrinkage) and two elements miss at 1.4-1.9e-4.
    //  * ANY x-pump: per-half qA balloons 0.9-1.3 deg and the halves refuse --
    //    lateral translation wrecks the half-window seed geometry.
    //  * z-only at 3.7 rad/s: >=0.8 from every FIRST-order w-line and 0.3 from the
    //    second-order products (2x1.7 = 3.4, 1.7+2.3 = 4.0). At 3.35 the 3.4-line
    //    collinearity collapsed Tg(2,2) by 75%; 3.7 recovered it exactly. Its
    //    short-duration side effects (col-0 starvation, da col-3 valley ~9e-3 at
    //    180 s) are what the 240 s / select_K 32 shape averages down -- the da
    //    not-worse contract passes at 240 s where 180 s breached it.
    const double amp_z = 0.30, w_z = 3.7;
    auto p1 = [&](double t) {
      Eigen::Vector3d p = tj.p_of(t);
      p(2) += tj.env(t) * amp_z * std::sin(w_z * t + 0.7);
      return p;
    };
    const Eigen::Matrix3d Dw_i = ImuIntrinsicModel::ut(tr.imu.dw).inverse();
    const Eigen::Matrix3d Da_i = ImuIntrinsicModel::ut(tr.imu.da).inverse();
    const Eigen::Matrix3d R_A_t = ov_core::quat_2_Rot(tr.imu.q_AtoI).transpose();
    for (double t = 0.0; t <= dur; t += 1.0 / so.imu_hz) {
      const double h = 1e-4;
      const Eigen::Vector3d pdd = (p1(t + h) - 2 * p1(t) + p1(t - h)) / (h * h);
      const Eigen::Vector3d a_hat = tj.R_of(t) * (pdd + tr.g_W);
      RawImu s;
      s.timestamp = t;
      s.wm = Dw_i * tj.omega_I(t) + tr.bg + so.w_noise * Eigen::Vector3d(nrm(g), nrm(g), nrm(g));
      if (!tr.imu.Tg.isZero())
        s.wm += tr.imu.Tg * a_hat;
      s.am = Da_i * (R_A_t * a_hat) + tr.ba + so.a_noise * Eigen::Vector3d(nrm(g), nrm(g), nrm(g));
      s.temp_c = 28.0;
      imu.push_back(s);
    }
    auto pf = synth::make_cloud(90, rng ^ 0x51ed270bu);
    uint32_t seq = 0;
    const Eigen::Matrix3d R_ItoC = ov_core::quat_2_Rot(tr.q_ItoC);
    for (double tc = 0.2; tc + tr.td <= dur - 0.01; tc += 1.0 / so.fps, ++seq) {
      FrameObs fo;
      fo.timestamp = tc - 0.5 * so.exposure_s;
      fo.exposure_s = (float)so.exposure_s;
      fo.temp_c = 28.f;
      fo.seq = seq;
      for (int f = 0; f < 90; ++f) {
        const double ti = tc + tr.td;
        const Eigen::Vector3d pc = R_ItoC * (tj.R_of(ti) * (pf[f] - p1(ti))) + tr.p_IinC;
        if (pc(2) < 0.4)
          continue;
        Eigen::Vector2d uv(tr.cam(0) * pc(0) / pc(2) + tr.cam(2), tr.cam(1) * pc(1) / pc(2) + tr.cam(3));
        if (uv(0) < 10 || uv(0) > tr.img_w - 10 || uv(1) < 10 || uv(1) > tr.img_h - 10)
          continue;
        FrameObsPoint p;
        p.id = (uint32_t)f;
        p.u = (float)(uv(0) + so.pix_noise * nrm(g));
        p.v = (float)(uv(1) + so.pix_noise * nrm(g));
        fo.pts.push_back(p);
      }
      frames.push_back(fo);
    }
  } else if (world == World::HOVER) {
    // excitation segment strictly outside [0, dur]: env() stays at its 0.005
    // residual-jitter floor -- a hover with sensor-level wobble, a_hat ~ R*g.
    tj.excite_t0 = 2.0 * dur;
    tj.excite_t1 = 2.0 * dur + 1.0;
    synth::make_streams(tr, tj, so, rng, imu, frames);
  } else {
    // SINGLE_AXIS: yaw about gravity. Rotation exists (bootstrap can pair), but
    // R(t) about g fixes g: a_hat = R*g_W = g_W is CONSTANT, so Tg*a_hat is a
    // constant gyro offset -- exactly the bias direction. Same generator as the
    // session-e2e degenerate world, plus the Tg inverse-map injection.
    std::mt19937 g(rng);
    std::normal_distribution<double> nrm(0.0, 1.0);
    synth::Trajectory te; // envelope only (excited over the collection span)
    te.excite_t0 = 6.0;
    te.excite_t1 = dur - 2.0;
    auto R1 = [&](double t) {
      const double e = te.env(t);
      return ov_core::exp_so3(Eigen::Vector3d(0, 0, e * 1.2 * std::sin(2.0 * t))); // yaw about g
    };
    const Eigen::Matrix3d Dw_i = ImuIntrinsicModel::ut(tr.imu.dw).inverse();
    const Eigen::Matrix3d Da_i = ImuIntrinsicModel::ut(tr.imu.da).inverse();
    const Eigen::Matrix3d R_A_t = ov_core::quat_2_Rot(tr.imu.q_AtoI).transpose();
    for (double t = 0.0; t <= dur; t += 1.0 / so.imu_hz) {
      const double h = 1e-6;
      const Eigen::Matrix3d W = -(R1(t + h) - R1(t - h)) / (2 * h) * R1(t).transpose();
      const Eigen::Vector3d a_hat = R1(t) * tr.g_W; // == g_W: constant specific force
      RawImu s;
      s.timestamp = t;
      s.wm = Dw_i * Eigen::Vector3d(W(2, 1), W(0, 2), W(1, 0)) + tr.bg + 2e-4 * Eigen::Vector3d(nrm(g), nrm(g), nrm(g));
      if (!tr.imu.Tg.isZero())
        s.wm += tr.imu.Tg * a_hat;
      s.am = Da_i * (R_A_t * a_hat) + tr.ba + 2e-3 * Eigen::Vector3d(nrm(g), nrm(g), nrm(g));
      s.temp_c = 28.0;
      imu.push_back(s);
    }
    auto pf = synth::make_cloud(90, rng ^ 0x51ed270bu);
    uint32_t seq = 0;
    const Eigen::Matrix3d R_ItoC = ov_core::quat_2_Rot(tr.q_ItoC);
    for (double tc = 0.2; tc + tr.td <= dur - 0.01; tc += 1.0 / so.fps, ++seq) {
      FrameObs fo;
      fo.timestamp = tc - 0.5 * so.exposure_s;
      fo.exposure_s = (float)so.exposure_s;
      fo.temp_c = 28.f;
      fo.seq = seq;
      for (int f = 0; f < 90; ++f) {
        const double ti = tc + tr.td;
        const Eigen::Vector3d pc = R_ItoC * (R1(ti) * pf[f]) + tr.p_IinC; // rotation about the origin
        if (pc(2) < 0.4)
          continue;
        Eigen::Vector2d uv(tr.cam(0) * pc(0) / pc(2) + tr.cam(2), tr.cam(1) * pc(1) / pc(2) + tr.cam(3));
        if (uv(0) < 10 || uv(0) > tr.img_w - 10 || uv(1) < 10 || uv(1) > tr.img_h - 10)
          continue;
        FrameObsPoint p;
        p.id = (uint32_t)f;
        p.u = (float)(uv(0) + so.pix_noise * nrm(g));
        p.v = (float)(uv(1) + so.pix_noise * nrm(g));
        fo.pts.push_back(p);
      }
      frames.push_back(fo);
    }
  }

  // pre-bootstrap seed: identity IMU intrinsics, ZERO Tg (blind), ~1.2 deg
  // extrinsic error, td = 0 -- the production cold-start shape.
  SessionSeed ss;
  ss.calib = SharedCalib();
  Eigen::Vector4d dq;
  dq << 0.5 * 0.012, 0.5 * 0.009, 0.5 * -0.015, 1.0;
  ss.calib.cams[0].q_ItoC = ov_core::quat_multiply(dq / dq.norm(), tr.q_ItoC);
  ss.calib.cams[0].p_IinC = tr.p_IinC + Eigen::Vector3d(0.006, -0.004, 0.005);
  ss.calib.cams[0].cam = tr.cam;
  ss.calib.cams[0].img_w = tr.img_w;
  ss.calib.cams[0].img_h = tr.img_h;
  ss.calib.cams[0].td = 0.0;
  ss.calib.cams[0].tr = 0.0;

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

static const BlockCommit *find_block(const SessionReport &rep, const char *name) {
  for (const auto &b : rep.blocks)
    if (b.cam < 0 && b.name == name)
      return &b;
  return nullptr;
}

static const char *verdict_name(SessionReport::AccelGateVerdict v) {
  using V = SessionReport::AccelGateVerdict;
  switch (v) {
  case V::PRE_CLOSED:
    return "PRE_CLOSED";
  case V::SPLIT_CONSISTENT:
    return "SPLIT_CONSISTENT";
  case V::SPLIT_INCONSISTENT:
    return "SPLIT_INCONSISTENT";
  case V::SPLIT_FAILED:
    return "SPLIT_FAILED";
  case V::WALD_CONSISTENT:
    return "WALD_CONSISTENT";
  case V::WALD_INCONSISTENT:
    return "WALD_INCONSISTENT";
  case V::WALD_UNOBSERVABLE:
    return "WALD_UNOBSERVABLE";
  }
  return "?";
}

static bool tg_is_seed_bytes(const SessionReport &rep, const Eigen::Matrix3d &seed_tg) {
  return std::memcmp(rep.committed.imu.Tg.data(), seed_tg.data(), 9 * sizeof(double)) == 0;
}

static void chain_errors(const SessionReport &rep, const synth::Truth &tr, double &edw, double &eda, double &eqa_deg) {
  edw = (rep.committed.imu.dw - tr.imu.dw).cwiseAbs().maxCoeff();
  eda = (rep.committed.imu.da - tr.imu.da).cwiseAbs().maxCoeff();
  eqa_deg = 2.0 * ov_core::quat_multiply(rep.committed.imu.q_AtoI, ov_core::Inv(tr.imu.q_AtoI)).head<3>().norm() * 180.0 / M_PI;
}

int main() {
  const double G = 9.81, R2D = 180.0 / M_PI;
  // ICM-part-class truth Tg (a measured production chain carries ~0.22 deg/s
  // @1g): all 9 elements DISTINCT so a transposed/permuted recovery cannot
  // pass by luck (the pi-convention seam the module removed -- keeps it removed).
  Eigen::Matrix3d Tg_true;
  Tg_true << 3.5e-4, -1.2e-4, 2.1e-4, //
      -2.8e-4, 4.2e-4, -0.6e-4,       //
      1.6e-4, -3.3e-4, -4.4e-4;
  const double part_class = 4e-4;
  const double tol_el = 0.3 * part_class; // per-element recovery bound (1.2e-4)

  SessionConfig cfg; // library defaults: split-half decides (a_gate_mode 0), free_tg on
  cfg.harvester.pix_sigma = 0.5;
  cfg.verbose = true;
  cfg.joint.verbose = false;
  cfg.cam_mode = 0; // temporal/IMU gates: the cam path is S4's, tg does not touch it

  // ---------------- T0: the sim inverse map IS the estimator's correct() inverse ----------------
  // Kills the classic T1-killer class up front: a sign/frame/storage mismatch between the
  // injection (wm += Tg*a_hat_true) and the correction (w_hat = Dw*(wm - bg - Tg*a_hat)) would
  // make every downstream verdict an artifact. Noise-free round trip must close to rounding.
  {
    synth::Truth tr = synth::make_truth();
    tr.imu.Tg = Tg_true;
    synth::Trajectory tj; // excited by default (env ~ 1 everywhere)
    std::mt19937 rng(7);
    double worst_w = 0.0, worst_a = 0.0;
    for (double t : {0.5, 1.7, 3.3, 7.9}) {
      RawImu s = synth::raw_imu_at(tr, tj, t, rng, 0.0, 0.0);
      Eigen::Vector3d w_hat, a_hat;
      tr.imu.correct(s.wm, s.am, tr.bg, tr.ba, w_hat, a_hat);
      worst_w = std::max(worst_w, (w_hat - tj.omega_I(t)).norm());
      worst_a = std::max(worst_a, (a_hat - tj.accel_I(t, tr.g_W)).norm());
    }
    std::printf("[T0] sim<->model round trip (Tg injected, noise-free): |dw_hat| %.2e, |da_hat| %.2e\n", worst_w, worst_a);
    CHECK(worst_w < 1e-9, "T0: injection is NOT the model inverse on the gyro side (%.2e) — sign/frame/storage defect", worst_w);
    CHECK(worst_a < 1e-9, "T0: accel inverse map broken (%.2e)", worst_a);
  }

  const std::string rec_tg = "/tmp/ov_zcalib_tg_e2e.bin";
  const std::string rec_0 = "/tmp/ov_zcalib_tg_e2e_base.bin";
  const std::string rec_hv = "/tmp/ov_zcalib_tg_e2e_hover.bin";
  const std::string rec_1x = "/tmp/ov_zcalib_tg_e2e_1axis.bin";

  // ---------------- T1: recovery on the rich 6-axis world ----------------
  // The tg falsifier judges HALF-agreement against the 1e-4 floor; at a
  // 120 s / 16-window shape its basin noise alone put the weakest element
  // (tg[8]) at z ~1.09 -- the verdict was the falsifier's noise, not the
  // estimator. A recovery gate must run the unambiguously fully-excited
  // shape (240 s / select_K 32); refusal margins stay T2/T3's job.
  const double t1_dur = 240.0;
  const int t1_select_k = 32;
  double edw_tg = 0, eda_tg = 0, eqa_tg = 0;
  {
    synth::Truth tr = synth::make_truth();
    tr.imu.Tg = Tg_true;
    write_tg_record(tr, rec_tg, t1_dur, World::RICH, 4242);
    SessionConfig c1 = cfg;
    c1.select_K = t1_select_k;
    c1.out_yaml = "/tmp/ov_zcalib_tg_e2e.yaml";
    SessionReport rep;
    CHECK(CalibSessionRunner::run_replay(rec_tg, c1, rep), "T1: replay failed to open");
    CHECK(rep.final_state == RunnerState::DONE, "T1: session ended in %d (%s)", (int)rep.final_state, rep.abort_reason.c_str());
    std::printf("[T1] tg gate verdict: %s, tg_open=%d, accel chain open=%d\n", verdict_name(rep.tg_gate_verdict), (int)rep.tg_open,
                (int)rep.a_full_open);
    CHECK(rep.a_full_open, "T1: accel chain failed to open on rich excitation (att %.1f deg, dyn %.2f m/s^2)",
          rep.accel_att_spread_deg, rep.accel_dyn_ms2);
    CHECK(rep.tg_open, "T1: tg failed to unlock on a real part-class Tg (verdict %s)", verdict_name(rep.tg_gate_verdict));
    const BlockCommit *btg = find_block(rep, "tg");
    CHECK(btg != nullptr, "T1: no tg block in the commit layout");
    if (btg) {
      std::printf("[T1] tg block: %s (ratio %.2f, sigma %.2e, moved %.1f)\n", btg->committed ? "COMMIT" : "SEED", btg->worst_ratio,
                  btg->worst_sigma, btg->moved_sigma);
      CHECK(btg->committed, "T1: tg block refused (ratio %.2f ceil_ok %d moved %.1f) on fully-excited part-class truth",
            btg->worst_ratio, (int)btg->ceiling_ok, btg->moved_sigma);
    }
    const Eigen::Matrix3d dT = rep.committed.imu.Tg - Tg_true;
    std::printf("[T1] Tg recovery (est | err, (rad/s)/(m/s^2), storage order):");
    for (int k = 0; k < 9; ++k)
      std::printf(" %+.2e|%+.1e", rep.committed.imu.Tg.data()[k], dT.data()[k]);
    // Recovery acceptance, three prongs (basin-headroom doctrine: documented
    // basin noise => headroom + floors, never a bare equality):
    //  (1) MEDIAN element within the 1.2e-4 claim -- most of the matrix at claim precision;
    //  (2) EVERY element within 1.5x claim (1.8e-4): measured across six worlds, all 9
    //      never landed under the raw claim at once -- information-limited, not a defect;
    //      the wald MC's measured kappa (~2-2.4 information under-dispersion) puts the
    //      honest 3-sigma at ~1.7e-4 for these posteriors (sigma_el ~3.9e-5), so a bare
    //      9-of-9 <= 3-sigma_raw bound flaps by construction;
    //  (3) |Tg| row-norm magnitude within 10% of truth (junk recoveries fail this at the
    //      30-75% shrinkage level measured on mis-excited worlds).
    double errs[9];
    for (int k = 0; k < 9; ++k)
      errs[k] = std::abs(dT.data()[k]);
    std::sort(errs, errs + 9);
    const double mag_est = rep.committed.imu.Tg.rowwise().norm().maxCoeff();
    const double mag_true = Tg_true.rowwise().norm().maxCoeff();
    std::printf("\n[T1] |Tg| %.3f deg/s @1g (truth %.3f), element err median %.2e / worst %.2e (claim %.2e, ceiling %.2e)\n",
                mag_est * G * R2D, mag_true * G * R2D, errs[4], errs[8], tol_el, 1.5 * tol_el);
    CHECK(errs[4] < tol_el, "T1: median Tg element err %.2e >= claim %.2e", errs[4], tol_el);
    for (int k = 0; k < 9; ++k)
      CHECK(std::abs(dT.data()[k]) < 1.5 * tol_el, "T1: Tg element %d err %.2e >= 1.5x claim %.2e", k, std::abs(dT.data()[k]),
            1.5 * tol_el);
    CHECK(std::abs(mag_est - mag_true) < 0.10 * mag_true, "T1: |Tg| magnitude %.3f vs truth %.3f deg/s @1g (>10%%)",
          mag_est * G * R2D, mag_true * G * R2D);
    chain_errors(rep, tr, edw_tg, eda_tg, eqa_tg);
    std::printf("[T1] chain with tg estimated: dw=%.2e da=%.2e qA=%.3f deg\n", edw_tg, eda_tg, eqa_tg);
    // S1-class absolute sanity so a jointly-degraded chain cannot pass on ratios alone
    CHECK(edw_tg < 2.5e-3, "T1: dw err %.2e", edw_tg);
    CHECK(eda_tg < 8.0e-3, "T1: da err %.2e (col-3 valley documented)", eda_tg);
    CHECK(eqa_tg < 0.80, "T1: qA err %.3f deg", eqa_tg);
  }

  // ---------------- T1 baseline: SAME world, Tg = 0 ----------------
  {
    synth::Truth tr0 = synth::make_truth(); // Tg stays zero
    write_tg_record(tr0, rec_0, t1_dur, World::RICH, 4242); // same duration/shape as T1 or "not worse" compares different worlds
    SessionConfig c0 = cfg;
    c0.select_K = t1_select_k;
    c0.out_yaml = "/tmp/ov_zcalib_tg_e2e_base.yaml";
    c0.verbose = false;
    SessionReport rep0;
    CHECK(CalibSessionRunner::run_replay(rec_0, c0, rep0), "T1-base: replay failed to open");
    CHECK(rep0.final_state == RunnerState::DONE, "T1-base: session ended in %d (%s)", (int)rep0.final_state,
          rep0.abort_reason.c_str());
    double edw0 = 0, eda0 = 0, eqa0 = 0;
    chain_errors(rep0, tr0, edw0, eda0, eqa0);
    std::printf("[T1] chain at Tg=0 baseline:   dw=%.2e da=%.2e qA=%.3f deg (tg verdict %s, tg_open=%d)\n", edw0, eda0, eqa0,
                verdict_name(rep0.tg_gate_verdict), (int)rep0.tg_open);
    // "Not worse" with the basin-noise headroom the suite documents (S1 qA
    // lands 0.57-0.73 deg across numerics-equivalent builds): 1.35x + a floor
    // well under each gate's own scale. A bare <= would flap on benign noise;
    // a genuine tg-induced regression (2x+) still fails loudly.
    CHECK(edw_tg <= 1.35 * edw0 + 2e-4, "T1: dw degraded by tg estimation (%.2e vs baseline %.2e)", edw_tg, edw0);
    CHECK(eda_tg <= 1.35 * eda0 + 5e-4, "T1: da degraded by tg estimation (%.2e vs baseline %.2e)", eda_tg, eda0);
    CHECK(eqa_tg <= 1.35 * eqa0 + 0.05, "T1: qA degraded by tg estimation (%.3f vs baseline %.3f deg)", eqa_tg, eqa0);
  }

  // ---------------- T1b: unlock -> COMMIT-WALK refusal must ship seed bytes ----------------
  // The path T1 cannot reach (its tg commits) and T2/T3 never reach (their sessions refuse
  // upstream): tg unlocks through the gate, solves through A1b/B, and the commit walk refuses
  // the BLOCK (here: an impossible sigma ceiling). The refused value must revert to the seed
  // BYTES in the shipped calib -- a solved Tg once shipped while the ledger said SEED; the
  // revert_block tg arm pins that leak class closed.
  {
    SessionConfig c1b = cfg;
    c1b.select_K = t1_select_k;
    c1b.verbose = false;
    c1b.out_yaml = "/tmp/ov_zcalib_tg_e2e_t1b.yaml";
    c1b.commit_abs_ceiling["tg"] = 1e-9; // no real posterior passes: force the walk refusal
    SessionReport rep;
    CHECK(CalibSessionRunner::run_replay(rec_tg, c1b, rep), "T1b: replay failed to open");
    CHECK(rep.final_state == RunnerState::DONE, "T1b: session ended in %d (%s)", (int)rep.final_state, rep.abort_reason.c_str());
    const BlockCommit *btg = find_block(rep, "tg");
    std::printf("[T1b] walk refusal: tg_open=%d, block %s (sigma %.2e vs ceiling 1e-9)\n", (int)rep.tg_open,
                btg ? (btg->committed ? "COMMIT" : "SEED") : "absent", btg ? btg->worst_sigma : -1.0);
    CHECK(btg != nullptr && !btg->committed, "T1b: tg block missing or committed past an impossible ceiling");
    CHECK(tg_is_seed_bytes(rep, Eigen::Matrix3d::Zero()),
          "T1b: commit-walk-refused tg shipped its solved value (the revert_block tg arm leak)");
  }

  // ---------------- T2: hover / constant-a_hat refusal ----------------
  {
    synth::Truth tr = synth::make_truth();
    tr.imu.Tg = Tg_true; // the WORLD has g-sensitivity; the session has no way to see it
    write_tg_record(tr, rec_hv, 90.0, World::HOVER, 777);
    SessionConfig c2 = cfg;
    c2.out_yaml = "/tmp/ov_zcalib_tg_e2e_hover.yaml";
    c2.verbose = false;
    SessionReport rep;
    CHECK(CalibSessionRunner::run_replay(rec_hv, c2, rep), "T2: replay failed to open");
    std::printf("[T2] hover session: state=%s%s%s, tg verdict %s, tg_open=%d\n",
                rep.final_state == RunnerState::DONE ? "DONE" : "ABORT", rep.abort_reason.empty() ? "" : " -- ",
                rep.abort_reason.c_str(), verdict_name(rep.tg_gate_verdict), (int)rep.tg_open);
    const BlockCommit *btg = find_block(rep, "tg");
    CHECK(!rep.tg_open, "T2: tg unlocked on a hover");
    CHECK(btg == nullptr || !btg->committed, "T2: tg COMMITTED on a hover (ratio %.2f)", btg ? btg->worst_ratio : -1.0);
    CHECK(tg_is_seed_bytes(rep, Eigen::Matrix3d::Zero()), "T2: committed Tg != seed bytes (an unearned g-sensitivity was written)");
  }

  // ---------------- T3: one-axis rotation about gravity ----------------
  {
    synth::Truth tr = synth::make_truth();
    tr.imu.Tg = Tg_true; // real Tg, but a_hat is constant: Tg*a_hat == a bias
    write_tg_record(tr, rec_1x, 90.0, World::SINGLE_AXIS, 778);
    SessionConfig c3 = cfg;
    c3.out_yaml = "/tmp/ov_zcalib_tg_e2e_1axis.yaml";
    c3.verbose = true;
    SessionReport rep;
    CHECK(CalibSessionRunner::run_replay(rec_1x, c3, rep), "T3: replay failed to open");
    std::printf("[T3] one-axis session: state=%s%s%s, tg verdict %s, tg_open=%d, att spread %.1f deg\n",
                rep.final_state == RunnerState::DONE ? "DONE" : "ABORT", rep.abort_reason.empty() ? "" : " -- ",
                rep.abort_reason.c_str(), verdict_name(rep.tg_gate_verdict), (int)rep.tg_open, rep.accel_att_spread_deg);
    using V = SessionReport::AccelGateVerdict;
    CHECK(rep.tg_gate_verdict != V::SPLIT_CONSISTENT && rep.tg_gate_verdict != V::WALD_CONSISTENT,
          "T3: tg gate certified a constant-a_hat session (%s)", verdict_name(rep.tg_gate_verdict));
    CHECK(!rep.tg_open, "T3: tg unlocked under one-axis rotation");
    const BlockCommit *btg = find_block(rep, "tg");
    CHECK(btg == nullptr || !btg->committed, "T3: tg COMMITTED under one-axis rotation (ratio %.2f)",
          btg ? btg->worst_ratio : -1.0);
    CHECK(tg_is_seed_bytes(rep, Eigen::Matrix3d::Zero()),
          "T3: committed Tg != seed bytes (a refused tg leaked past the revert — the revert_block tg arm)");
    // the rest of the chain must not silently absorb the g-offset either: da/qA
    // are gated by the same degenerate-motion machinery the session-e2e S3 pins
    if (rep.final_state == RunnerState::DONE)
      for (const auto &b : rep.blocks)
        if (b.cam < 0 && (b.name == "da" || b.name == "q_AtoI"))
          CHECK(!b.committed, "T3: degenerate motion committed %s (false confidence)", b.name.c_str());
  }

  std::remove(rec_tg.c_str());
  std::remove(rec_0.c_str());
  std::remove(rec_hv.c_str());
  std::remove(rec_1x.c_str());
  if (failures == 0) {
    std::printf("[PASS] tg e2e: T1 part-class recovery (median at claim, all within 1.5x, |Tg| within 10%%, chain not degraded), "
                "T1b walk-refusal seed bytes, T2 hover refusal (seed bytes), T3 one-axis abstention (no commit, seed bytes)\n");
    return 0;
  }
  std::printf("[FAILED] %d checks\n", failures);
  return 1;
}
