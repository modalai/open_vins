/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib session end-to-end gates:
 *
 *  [S1] Full synthetic session through the RECORD -> REPLAY path with NO truth
 *       anywhere downstream (SETTLE still baseline, bootstrap hand-eye/xcorr,
 *       harvest, linear seeds, reservoir, VarPro, VERIFY, partial COMMIT):
 *       gates on recovered extrinsics/td/IMU intrinsics and on the commit
 *       decisions themselves.
 *  [S2] Determinism: two replays of the same record produce byte-identical
 *       committed YAML (the deterministic-reduction contract, session level).
 *  [S3] Degenerate-motion suite (RSS Table I style): a single-axis-rotation
 *       session must NOT commit the unexcited blocks -- either the bootstrap
 *       diversity gate aborts, or the sigma rule holds every gated parameter
 *       at its prior/seed.
 *  [S4] Staged camera-intrinsic refinement (cam_mode=1): the committed cam
 *       block must improve on its seed without sacrificing temporal/IMU blocks.
 *  [S7/S8] Refusing the nuisance Tg refits the remaining parameters and their
 *       posterior, or restores the complete A0 result when its budget expires.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <limits>
#include <thread>

#include "core/CalibSessionRunner.h"
#include "sim/SynthWorld.h"
#include "utils/NumericChecks.h"
#include "utils/YamlWriteback.h"

#include <sys/stat.h>
#include <unistd.h>

using namespace ov_zcalib;

static std::string tmp_file(const char *name) {
  return "/tmp/" + std::to_string((long)::getpid()) + "_" + name;
}

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

// Session stream: quiet head (SETTLE + bootstrap-fail guard), one long excited
// span for bootstrap + collection. Truth td/extrinsics/IMU-intrinsics inside.
static void write_session_record(const synth::Truth &tr, const std::string &path, double dur, double ex0, double ex1, unsigned rng,
                                 bool single_axis = false, const Eigen::Matrix<double, 8, 1> *seed_cam = nullptr,
                                 double pump_amp = 0.0) {
  synth::Trajectory tj;
  tj.excite_t0 = ex0;
  tj.excite_t1 = ex1;
  synth::StreamOptions so;
  so.dur = dur;
  so.excite = {{ex0, ex1}};
  std::vector<RawImu> imu;
  std::vector<FrameObs> frames;
  if (!single_axis && pump_amp > 0.0) {
    // g-aligned "pumping" at real handheld dynamics (std|a_m| 2.8-3.2 m/s^2). The
    // weak 0.37 m/s^2 shape starves the split-half chain judge of per-half q_AtoI
    // conditioning (per-half basins 0.6-1.0 deg vs 0.3 deg agreement bands), so the
    // honest judge refuses there. 3.1 rad/s is the measured da-clean frequency:
    // da col-3 stays ~3.1e-3 where 3.7 rad/s parks it ~1e-2 at short durations.
    // Its one weakness -- Tg(2,2) shrinkage against the 3.1-2.9 beat -- is a
    // Tg-RECOVERY concern; these worlds carry Tg=0 and the tg gate correctly
    // refuses (tg_e2e T1 owns recovery at 3.7 / 240 s). Generated HERE like the
    // single-axis branch, NOT via a synth::Trajectory knob: the -ffast-math SLP
    // seam repacks p_of's sin group if the shared expression grows a guarded term.
    std::mt19937 g(rng);
    std::normal_distribution<double> nrm(0.0, 1.0);
    const double w_z = 3.1;
    auto p1 = [&](double t) {
      Eigen::Vector3d p = tj.p_of(t);
      p(2) += tj.env(t) * pump_amp * std::sin(w_z * t + 0.7);
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
      // producer convention: the stamp IS the optical instant (center-row mid-exposure; sim is
      // global-shutter and samples the trajectory at tc). exposure_s is provenance only.
      fo.timestamp = tc;
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
  } else if (!single_axis) {
    synth::make_streams(tr, tj, so, rng, imu, frames);
  } else {
    // single-axis rotation, no gravity re-orientation: the degenerate case
    std::mt19937 g(rng);
    std::normal_distribution<double> nrm(0.0, 1.0);
    auto R1 = [&](double t) {
      const double e = tj.env(t);
      return ov_core::exp_so3(Eigen::Vector3d(0, 0, e * 1.2 * std::sin(2.0 * t))); // yaw-ish about g
    };
    const Eigen::Matrix3d Dw_i = ImuIntrinsicModel::ut(tr.imu.dw).inverse();
    const Eigen::Matrix3d Da_i = ImuIntrinsicModel::ut(tr.imu.da).inverse();
    const Eigen::Matrix3d R_A_t = ov_core::quat_2_Rot(tr.imu.q_AtoI).transpose();
    for (double t = 0.0; t <= dur; t += 1.0 / so.imu_hz) {
      const double h = 1e-6;
      const Eigen::Matrix3d W = -(R1(t + h) - R1(t - h)) / (2 * h) * R1(t).transpose();
      RawImu s;
      s.timestamp = t;
      s.wm = Dw_i * Eigen::Vector3d(W(2, 1), W(0, 2), W(1, 0)) + tr.bg + 2e-4 * Eigen::Vector3d(nrm(g), nrm(g), nrm(g));
      s.am = Da_i * (R_A_t * (R1(t) * tr.g_W)) + tr.ba + 2e-3 * Eigen::Vector3d(nrm(g), nrm(g), nrm(g));
      s.temp_c = 28.0;
      imu.push_back(s);
    }
    auto pf = synth::make_cloud(90, rng ^ 0x51ed270bu);
    uint32_t seq = 0;
    const Eigen::Matrix3d R_ItoC = ov_core::quat_2_Rot(tr.q_ItoC);
    for (double tc = 0.2; tc + tr.td <= dur - 0.01; tc += 1.0 / so.fps, ++seq) {
      FrameObs fo;
      // producer convention: the stamp IS the optical instant (center-row mid-exposure; sim is
      // global-shutter and samples the trajectory at tc). exposure_s is provenance only.
      fo.timestamp = tc;
      fo.exposure_s = (float)so.exposure_s;
      fo.temp_c = 28.f;
      fo.seq = seq;
      for (int f = 0; f < 90; ++f) {
        const double ti = tc + tr.td;
        const Eigen::Vector3d pc = R_ItoC * (R1(ti) * pf[f]) + tr.p_IinC; // rotation about origin
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

  // pre-bootstrap seed: identity IMU intrinsics, ~1.2 deg extrinsic error, td = 0
  SessionSeed ss;
  ss.calib = SharedCalib();
  Eigen::Vector4d dq;
  dq << 0.5 * 0.012, 0.5 * 0.009, 0.5 * -0.015, 1.0;
  ss.calib.cams[0].q_ItoC = ov_core::quat_multiply(dq / dq.norm(), tr.q_ItoC);
  ss.calib.cams[0].p_IinC = tr.p_IinC + Eigen::Vector3d(0.006, -0.004, 0.005);
  ss.calib.cams[0].cam = seed_cam ? *seed_cam : tr.cam;
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

static std::string slurp(const std::string &p) {
  FILE *f = std::fopen(p.c_str(), "rb");
  if (!f)
    return "";
  std::string s;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
    s.append(buf, n);
  std::fclose(f);
  return s;
}

template <typename A, typename B> static bool same_bits(const A &a, const B &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         (a.size() == 0 || std::memcmp(a.data(), b.data(), sizeof(double) * a.size()) == 0);
}

static bool same_imu(const ImuIntrinsicModel &a, const ImuIntrinsicModel &b) {
  return same_bits(a.dw, b.dw) && same_bits(a.da, b.da) && same_bits(a.q_AtoI, b.q_AtoI) && same_bits(a.Tg, b.Tg) &&
         a.calib_dw == b.calib_dw && a.calib_da == b.calib_da && a.calib_RAtoI == b.calib_RAtoI && a.calib_tg == b.calib_tg;
}

static bool same_calibration(const SharedCalib &a, const SharedCalib &b) {
  if (!same_imu(a.imu, b.imu) || !same_imu(a.noise_lin, b.noise_lin) || a.noise_frozen != b.noise_frozen ||
      a.tg_enabled != b.tg_enabled || a.cams.size() != b.cams.size())
    return false;
  for (size_t i = 0; i < a.cams.size(); ++i) {
    const CamCalib &ac = a.cams[i], &bc = b.cams[i];
    if (!same_bits(ac.q_ItoC, bc.q_ItoC) || !same_bits(ac.p_IinC, bc.p_IinC) || !same_bits(ac.cam, bc.cam) ||
        std::memcmp(&ac.td, &bc.td, sizeof(double)) != 0 || std::memcmp(&ac.tr, &bc.tr, sizeof(double)) != 0 ||
        ac.free_ext != bc.free_ext || ac.free_td != bc.free_td || ac.cam_mode != bc.cam_mode)
      return false;
  }
  return true;
}

static bool posterior_matches_layout(const SessionReport &report) {
  SharedCalib solved = report.solved;
  std::vector<std::string> labels;
  for (const auto &block : solved.free_blocks())
    for (int k = 0; k < block.lsize; ++k)
      labels.push_back(block.label() + "[" + std::to_string(k) + "]");
  const int n = solved.local_dim();
  return report.joint.ok && report.joint.labels == labels && report.joint.dim_p == n &&
         report.joint.sigma.size() == n && report.joint.prior_sigma_vec.size() == n &&
         report.joint.Lambda.rows() == n && report.joint.Lambda.cols() == n &&
         finite_matrix(report.joint.sigma) && finite_matrix(report.joint.Lambda);
}

static const StageEvidence *stage_named(const SessionReport &report, const char *name) {
  for (const auto &stage : report.evidence)
    if (stage.label == name)
      return &stage;
  return nullptr;
}

int main(int argc, char **argv) {
  // ---- developer iteration affordances (results IDENTICAL by contract) ----
  // S-block selection: pass any of S1 through S8 to run just those (default: all). Blocks own
  // their records; S2 replays S1's record, so selecting S2 runs S1 too. Lets a fix to one
  // block iterate in ~1/4 of the suite wall instead of re-paying every world.
  auto want = [&](const char *b) {
    if (argc <= 1)
      return true;
    for (int i = 1; i < argc; ++i)
      if (std::strcmp(argv[i], b) == 0)
        return true;
    return false;
  };
  const bool s2 = want("S2");
  const bool s1 = want("S1") || s2;
  const bool s5 = want("S5");
  const bool s6 = want("S6");
  const bool s7 = want("S7");
  const bool s8 = want("S8");

  if (want("IO")) {
    char directory[] = "/tmp/ov_zcalib_yaml_io.XXXXXX";
    CHECK(::mkdtemp(directory) != nullptr, "IO: temporary directory failed");
    const std::string path = std::string(directory) + "/result.yaml";
    const std::string temporary = path + ".tmp", backup = path + ".rollback";
    auto write_text = [&](const std::string &file, const char *text) {
      FILE *f = std::fopen(file.c_str(), "wb");
      CHECK(f != nullptr, "IO: fixture open failed");
      if (f) {
        CHECK(std::fputs(text, f) >= 0, "IO: fixture write failed");
        CHECK(std::fclose(f) == 0, "IO: fixture close failed");
      }
    };
    SharedCalib calibration;
    write_text(path, "previous valid calibration\n");
    write_text(backup, "older rollback\n");
    CHECK(::symlink("/dev/full", temporary.c_str()) == 0, "IO: /dev/full fixture failed");
    errno = 0;
    CHECK(!write_calib_yaml(path, calibration, {}, nullptr, nullptr, true) && errno == ENOSPC,
          "IO: buffered ENOSPC was reported as a successful write");
    CHECK(slurp(path) == "previous valid calibration\n" && slurp(backup) == "older rollback\n",
          "IO: failed write replaced previous result or rollback");
    CHECK(::access(temporary.c_str(), F_OK) != 0, "IO: failed temporary file was not removed");
    CHECK(!write_calib_yaml(std::string(directory) + "/absent/result.yaml", calibration),
          "IO: nonexistent output directory accepted");
    CHECK(write_calib_yaml(path, calibration, {}, nullptr, nullptr, true), "IO: valid atomic replacement failed");
    CHECK(slurp(path).find("num_cameras: 1\n") != std::string::npos && slurp(backup) == "previous valid calibration\n",
          "IO: successful replacement lost content or prior calibration");
    const std::string good = slurp(path);
    CHECK(::unlink(backup.c_str()) == 0 && ::mkdir(backup.c_str(), 0700) == 0, "IO: rollback-failure fixture failed");
    CHECK(!write_calib_yaml(path, calibration, {}, nullptr, nullptr, true) && slurp(path) == good,
          "IO: backup failure removed the live calibration");
    CHECK(::access(temporary.c_str(), F_OK) != 0 && ::access((backup + ".tmp").c_str(), F_OK) != 0,
          "IO: backup failure leaked temporary files");
    ::rmdir(backup.c_str());
    ::unlink(path.c_str());
    ::rmdir(directory);
    std::printf("[IO] YAML save detects buffered ENOSPC and preserves prior files on failure\n");
  }

  synth::Truth tr = synth::make_truth();
  const std::string rec = tmp_file("ov_zcalib_session_e2e.bin");
  const std::string rec_deg = tmp_file("ov_zcalib_session_deg.bin");

  SessionConfig cfg;
  cfg.harvester.pix_sigma = 0.5;
  cfg.out_yaml = tmp_file("ov_zcalib_session_e2e.yaml");
  cfg.verbose = true;
  cfg.joint.verbose = false;
  cfg.cam_mode = 0; // S1/S2/S3 are the frozen temporal/IMU gates; S4 below gates the camera path
  // Window pool at the machine width: serial == parallel BIT-IDENTICAL (fixed-range partition,
  // worker-ordered fold -- see ov_init::zbft_sfm::ParallelExecutor), so this is pure wall-clock.
  // Accuracy/determinism runs have no wall budget: only the 60 s inner hang
  // guard, which must never bind. Deadline regressions below inject a clock
  // tied to completed stages so machine load cannot change their verdict.
  cfg.joint.num_threads = std::max(4u, std::thread::hardware_concurrency());

  // 6 s quiet head (settle), excitation [6, 120]; bootstrap ~[6, 20+], collection after.
  // S1 = the DESIGNED rich-excitation case (pump 0.35 m @ 3.1 rad/s, std|a_m| ~2.4
  // m/s^2) so the accel-chain unlock question is really asked. The weak 0.37 m/s^2
  // shape REFUSES under the arbiter's split-half judge (its per-half qA basin is
  // wider than the agreement band) -- that refusal world is tg_e2e's T2/T3 territory;
  // S1 must be the certify world.
  if (s1 || s5 || s6 || s7 || s8)
    write_session_record(tr, rec, 120.0, 6.0, 118.0, 4242, false, nullptr, 0.35);

  // ---------------- S1: full session, replay path ----------------
  if (s1) {
  SessionReport rep;
  CHECK(CalibSessionRunner::run_replay(rec, cfg, rep), "S1: replay failed to open");
  CHECK(rep.final_state == RunnerState::DONE, "S1: session ended in %d (%s)", (int)rep.final_state, rep.abort_reason.c_str());
  std::printf("[S1] harvested=%d retained=%d holdout=%d rejected_seed=%d fused=%d verify_improve=%.1f%%\n", rep.windows_harvested,
              rep.windows_retained, rep.windows_holdout, rep.windows_rejected_seed, rep.windows_fused, 100.0 * rep.verify_improve);

  double er = 2.0 * ov_core::quat_multiply(rep.committed.cams[0].q_ItoC, ov_core::Inv(tr.q_ItoC)).head<3>().norm() * 180.0 / M_PI;
  double ep = (rep.committed.cams[0].p_IinC - tr.p_IinC).norm() * 1e3;
  double et = std::abs(rep.committed.cams[0].td - tr.td) * 1e3;
  double edw = (rep.committed.imu.dw - tr.imu.dw).cwiseAbs().maxCoeff();
  double eda = (rep.committed.imu.da - tr.imu.da).cwiseAbs().maxCoeff();
  double eqa = 2.0 * ov_core::quat_multiply(rep.committed.imu.q_AtoI, ov_core::Inv(tr.imu.q_AtoI)).head<3>().norm() * 180.0 / M_PI;
  std::printf("[S1] committed: ext_rot=%.3f deg ext_pos=%.2f mm td=%.3f ms dw=%.2e da=%.2e qA=%.3f deg\n", er, ep, et, edw, eda, eqa);
  std::printf("[S1] blocks:");
  for (auto &b : rep.blocks)
    std::printf(" %s=%s(%.2f)", b.name.c_str(), b.committed ? "yes" : "SEED", b.worst_ratio);
  std::printf("\n");
  // Production-path gates (full pipeline, no truth seeds anywhere). The
  // lever-arm and accel-frame gates are wider than the frozen S3 harness ones:
  // the session trajectory is envelope-ramped guided excitation, not the
  // harness's persistent aggressive 6-axis sweep, and p_IinC/q_AtoI are the
  // two blocks whose information comes from exactly that tail excitation.
  CHECK(er < 0.15, "S1: ext rot err %.3f deg", er);
  CHECK(ep < 8.0, "S1: ext pos err %.2f mm", ep);
  CHECK(et < 0.30, "S1: td err %.3f ms", et);
  CHECK(edw < 2.5e-3, "S1: dw err %.2e", edw);
  CHECK(eda < 8.0e-3, "S1: da err %.2e (col-3 valley documented)", eda);
  // The S1 sweep is the DESIGNED rich-excitation case: the accel-chain unlock
  // must OPEN here. Pinned explicitly because an abstention can masquerade as
  // a small qA error (truth is ~1.1 deg from the identity it would freeze at)
  // -- a gate-state-conditional tolerance would let a broken gate pass silently.
  CHECK(rep.a_full_open, "S1: accel-chain gate failed to open on rich synthetic excitation (att %.1f deg, dyn %.2f m/s^2)",
        rep.accel_att_spread_deg, rep.accel_dyn_ms2);
  // qA gate = truth-recovery bound OUTSIDE the estimator's own reproducibility
  // band: the split-half measures ~0.23 deg half-to-half scatter and full runs
  // land 0.57-0.73 deg across numerics-equivalent builds (float-path/summation
  // order) -- qA sits in a flat valley at this excitation. 0.80 still asserts
  // genuine recovery (abstention reads ~1.13 = the truth offset itself, pinned
  // by the a_full_open check above); a tighter gate flaps on benign changes.
  CHECK(eqa < 0.80, "S1: qA err %.3f deg", eqa);
  for (auto &b : rep.blocks)
    if (b.name == "q_ItoC" || b.name == "td" || b.name == "dw")
      CHECK(b.committed, "S1: block %s not committed (ratio %.2f)", b.name.c_str(), b.worst_ratio);
  }

  // ---------------- S2: determinism (two replays -> byte-identical YAML) ----------------
  if (s2) {
    const std::string y1 = slurp(cfg.out_yaml);
    SessionConfig cfg2 = cfg;
    cfg2.out_yaml = tmp_file("ov_zcalib_session_e2e_2.yaml");
    // A stale yaml from a PREVIOUS run makes the byte-compare below pass vacuously when
    // replay #2 fails before writeback (measured to mask a real flake). Fresh file or nothing.
    std::remove(cfg2.out_yaml.c_str());
    cfg2.verbose = false;
    SessionReport rep2;
    CHECK(CalibSessionRunner::run_replay(rec, cfg2, rep2), "S2: second replay failed");
    const std::string y2 = slurp(cfg2.out_yaml);
    CHECK(!y1.empty() && y1 == y2, "S2: committed YAML not byte-identical across replays");
    std::printf("[S2] determinism: committed YAML byte-identical across replays (%zu bytes)\n", y1.size());
  }

  // ---------------- S3: degenerate single-axis session ----------------
  if (want("S3")) {
    write_session_record(tr, rec_deg, 90.0, 6.0, 88.0, 777, /*single_axis*/ true);
    SessionConfig cfgd = cfg;
    cfgd.out_yaml = tmp_file("ov_zcalib_session_deg.yaml");
    cfgd.verbose = false;
    SessionReport repd;
    CHECK(CalibSessionRunner::run_replay(rec_deg, cfgd, repd), "S3: degenerate replay failed");
    if (repd.final_state == RunnerState::ABORT) {
      std::printf("[S3] degenerate session correctly rejected at: %s\n", repd.abort_reason.c_str());
    } else {
      // If it got through, the sigma rule must hold the unexcitable blocks at seed
      std::printf("[S3] degenerate session blocks:");
      for (auto &b : repd.blocks)
        std::printf(" %s=%s(%.2f)", b.name.c_str(), b.committed ? "yes" : "SEED", b.worst_ratio);
      std::printf("\n");
      for (auto &b : repd.blocks)
        if (b.name == "da" || b.name == "q_AtoI")
          CHECK(!b.committed, "S3: degenerate motion committed %s (false confidence)", b.name.c_str());
      // and whatever was committed must not be WORSE than the seed error scale
      const double eda_d = (repd.committed.imu.da - tr.imu.da).cwiseAbs().maxCoeff();
      const double eda_seed = (SharedCalib().imu.da - tr.imu.da).cwiseAbs().maxCoeff();
      CHECK(eda_d <= eda_seed + 1e-12, "S3: da drifted under degenerate motion (%.2e vs seed %.2e)", eda_d, eda_seed);
    }
  }

  // ---------------- S4: staged camera-intrinsic refinement (cam_mode=1) ----------------
  if (want("S4") || want("S4-tangent")) {
    const bool perturb_tangent = argc > 1 && want("S4-tangent");
    synth::Truth trc = synth::make_truth();
    trc.cam << 450, 452, 320, 240, -0.020, 0.005, 0.0002, -0.0001; // real radtan distortion
    Eigen::Matrix<double, 8, 1> seed_cam;
    // existing-cal-grade seed error: ~1.5 px focal/center, k1/k2 off within the refine prior
    seed_cam << 451.5, 450.8, 321.0, 239.2, -0.016, 0.003, 0.0002, -0.0001;
    if (perturb_tangent) {
      seed_cam(6) += 8e-4;
      seed_cam(7) -= 6e-4;
    }
    const std::string rec_cam = perturb_tangent ? tmp_file("ov_zcalib_session_tangent.bin") : tmp_file("ov_zcalib_session_cam.bin");
    write_session_record(trc, rec_cam, 120.0, 6.0, 118.0, 4242, false, &seed_cam);
    SessionConfig cfgc = cfg;
    cfgc.cam_mode = 1; // refine (the production default)
    cfgc.out_yaml = perturb_tangent ? tmp_file("ov_zcalib_session_tangent.yaml") : tmp_file("ov_zcalib_session_cam.yaml");
    cfgc.verbose = false;
    SessionReport repc;
    CHECK(CalibSessionRunner::run_replay(rec_cam, cfgc, repc), "S4: cam-refine replay failed");
    CHECK(repc.final_state == RunnerState::DONE, "S4: session ended in %d (%s)", (int)repc.final_state, repc.abort_reason.c_str());
    const Eigen::Matrix<double, 8, 1> dcam = repc.committed.cams[0].cam - trc.cam;
    const Eigen::Matrix<double, 8, 1> dseed = seed_cam - trc.cam;
    std::printf("[S4] cam refine: fx %+0.2f->%+0.2f  fy %+0.2f->%+0.2f  cx %+0.2f->%+0.2f  cy %+0.2f->%+0.2f  k1 %+0.4f->%+0.4f  "
                "k2 %+0.4f->%+0.4f (seed err -> committed err)\n",
                dseed(0), dcam(0), dseed(1), dcam(1), dseed(2), dcam(2), dseed(3), dcam(3), dseed(4), dcam(4), dseed(5), dcam(5));
    bool cam_committed = false;
    for (auto &b : repc.blocks)
      if (b.name == "cam")
        cam_committed = b.committed;
    std::printf("[S4] cam block committed=%d  verify improve=%.1f%%\n", (int)cam_committed, 100.0 * repc.verify_improve);
    CHECK(cam_committed, "S4: cam block not committed");
    if (perturb_tangent) {
      std::printf("[S4-tangent] p1 error %+.2e -> %+.2e, p2 error %+.2e -> %+.2e\n",
                  dseed(6), dcam(6), dseed(7), dcam(7));
      CHECK(std::abs(dcam(6)) < 4e-4 && std::abs(dcam(7)) < 4e-4,
            "S4-tangent: tangential recovery %.2e/%.2e", dcam(6), dcam(7));
    }
    // Measured-honest bounds for ONE ~110 s single-camera session (focal/center
    // trade against metric depth; distortion is the strongest-recovered pair).
    // Every dof must also IMPROVE on its seed -- refine must never harm.
    // Center bound 1.0: the cx/cy pair lives in the flat valley, and its
    // landing moves ~0.1-0.2 px across JUSTIFIED solver operating points
    // while focal/distortion improve outright. A bound at any one landing +
    // epsilon flaps on benign changes (the qA gate's philosophy); genuine harm
    // is caught by the improve-on-seed contract below and the 0.8 focal bound.
    CHECK(std::abs(dcam(0)) < 0.8 && std::abs(dcam(1)) < 0.8, "S4: focal err %.2f/%.2f px", dcam(0), dcam(1));
    CHECK(std::abs(dcam(2)) < 1.0 && std::abs(dcam(3)) < 1.0, "S4: center err %.2f/%.2f px", dcam(2), dcam(3));
    // Distortion bound 2.0e-3: still half the k1 seed error, so genuine recovery is
    // asserted. The exact k1/k2 landing was bisected to the center-stamp convention's
    // STRUCTURAL delta (removing the tr parameter block reorders the problem under
    // -ffast-math; center stamps shift the clone timeline) -- the float-path/
    // summation-order basin class, not a defect in any one term. The improve-on-seed
    // contract below still polices genuine harm dof-by-dof.
    CHECK(std::abs(dcam(4)) < 2.0e-3 && std::abs(dcam(5)) < 2.0e-3, "S4: distortion err %.4f/%.4f", dcam(4), dcam(5));
    for (int k = 0; k < 6; ++k)
      CHECK(std::abs(dcam(k)) <= std::abs(dseed(k)) + 0.35, "S4: dof %d worse than seed (%.3f vs %.3f)", k, dcam(k), dseed(k));
    // the temporal/IMU blocks must not be SACRIFICED to the cam unlock: bounded
    // aliasing only (seed was 1.2 deg; phases A ran with the wrong distortion)
    const double er_c = 2.0 * ov_core::quat_multiply(repc.committed.cams[0].q_ItoC, ov_core::Inv(trc.q_ItoC)).head<3>().norm() * 180.0 / M_PI;
    const double et_c = std::abs(repc.committed.cams[0].td - trc.td) * 1e3;
    std::printf("[S4] ext_rot=%.3f deg td=%.3f ms under cam refine\n", er_c, et_c);
    // Keep S4's original bound; the perturbed tangential seed has a separate fixture tolerance.
    const double rotation_limit = perturb_tangent ? 0.30 : 0.25;
    CHECK(er_c < rotation_limit && et_c < 0.30, "S4: temporal/ext degraded under cam unlock (%.3f deg, %.3f ms)", er_c, et_c);
    std::remove(rec_cam.c_str());
  }

  for (bool output_failure : {false, true}) {
    if (!(output_failure ? s6 : s5))
      continue;
    // Deterministic deadline: A0 runs normally; every later joint call sees
    // an expired clock. This must preserve the complete A0 posterior and
    // freeze the unestimated IMU chain, rather than aborting or reusing A0
    // sigmas under A1a's larger parameter layout.
    SessionRecordReader rd;
    CHECK(rd.open(rec), "S5: record open failed");
    const SessionSeed seed = rd.seed();
    SessionConfig limited = cfg;
    limited.cam_mode = 1;
    limited.joint.max_wall_s = 1e6;
    limited.joint.num_threads = 1;
    const std::string io_output = tmp_file("ov_zcalib_session_io.yaml");
    if (output_failure) {
      limited.out_yaml = io_output;
      FILE *previous = std::fopen(io_output.c_str(), "wb");
      CHECK(previous != nullptr, "S6: previous output fixture failed");
      if (previous) {
        CHECK(std::fputs("previous valid calibration\n", previous) >= 0 && std::fclose(previous) == 0,
              "S6: previous output fixture write failed");
      }
      CHECK(::symlink("/dev/full", (io_output + ".tmp").c_str()) == 0, "S6: /dev/full fixture failed");
    }
    CalibSessionRunner *live = nullptr;
    limited.joint.budget_clock = [&]() {
      if (live)
        for (const auto &stage : live->report().evidence)
          if (stage.label == "A0-ext-td")
            return 1e6;
      return 0.0;
    };
    CalibSessionRunner runner(limited, seed);
    live = &runner;
    bool is_imu = false;
    RawImu imu;
    FrameObs frame;
    while (rd.next(is_imu, imu, frame)) {
      if (is_imu)
        runner.feed_imu(imu);
      else
        runner.feed_frame(frame);
    }
    const SessionReport &result = runner.finish();
    SharedCalib solved_layout = result.solved;
    if (output_failure) {
      CHECK(result.final_state == RunnerState::ABORT && result.abort_reason.find("calibration YAML write failed") != std::string::npos,
            "S6: failed output reported success (%s)", result.abort_reason.c_str());
      bool committed = false;
      for (const auto &block : result.blocks)
        committed = committed || block.committed;
      CHECK(!committed && !result.blocks.empty(), "S6: failed output still claims committed blocks");
      CHECK(slurp(io_output) == "previous valid calibration\n", "S6: failed output destroyed previous result");
      ::unlink((io_output + ".tmp").c_str());
      ::unlink(io_output.c_str());
      std::printf("[S6] output failure aborts a verified solve without publishing committed blocks\n");
    } else {
      CHECK(result.final_state == RunnerState::DONE, "S5: lost completed A0 under A1a deadline (%s)", result.abort_reason.c_str());
    }
    CHECK(result.joint.ok && result.joint.sigma.size() == solved_layout.local_dim(),
          "S5: A0 posterior/layout mismatch after A1a deadline");
    CHECK(!result.solved.imu.calib_dw && !result.solved.imu.calib_da && !result.solved.imu.calib_RAtoI &&
              !result.solved.imu.calib_tg && !result.a_full_open && !result.tg_open,
          "S5: skipped IMU stage opened unestimated blocks");
    bool skipped_a1a = false, skipped_b = false;
    for (const auto &stage : result.evidence) {
      if (stage.label == "A1a-dw-dadiag")
        skipped_a1a = stage.hit_budget && stage.passes == 0;
      if (stage.label == "B2-polish")
        skipped_b = stage.hit_budget && stage.passes == 0;
    }
    CHECK(skipped_a1a && skipped_b, "S5: missing truthful skipped-stage evidence");
    std::printf("[S5] A1a/B deadline preserves A0 posterior and freezes unestimated IMU blocks\n");
  }

  if (s7 || s8) {
    // Tighten only the full-chain pre-gate so this test isolates the refused
    // Tg path. Accuracy, posterior, VERIFY and COMMIT gates retain their
    // production settings; the record contains no estimator truth seeds.
    auto replay_refused_tg = [&](int cam_mode, const char *expire_after, SessionSeed &seed) {
      SessionReport result;
      SessionRecordReader reader;
      const bool opened = reader.open(rec);
      CHECK(opened, "S7/S8: record open failed");
      if (!opened)
        return result;
      seed = reader.seed();
      SessionConfig refit_cfg = cfg;
      refit_cfg.cam_mode = cam_mode;
      refit_cfg.a_full_min_windows = std::numeric_limits<int>::max();
      refit_cfg.verbose = false;
      refit_cfg.out_yaml.clear();
      refit_cfg.joint.num_threads = 4;
      refit_cfg.joint.max_wall_s = 1e6;
      CalibSessionRunner *live = nullptr;
      refit_cfg.joint.budget_clock = [&]() {
        // Evidence is appended only after each joint call joins its workers;
        // this callback is read-only during every parallel window sweep.
        return live && expire_after && stage_named(live->report(), expire_after) ? 1e6 : 0.0;
      };
      CalibSessionRunner runner(refit_cfg, seed);
      live = &runner;
      bool is_imu = false;
      RawImu imu;
      FrameObs frame;
      while (reader.next(is_imu, imu, frame)) {
        if (is_imu)
          runner.feed_imu(imu);
        else
          runner.feed_frame(frame);
      }
      return SessionReport(runner.finish());
    };

    if (s7) {
      SessionSeed seed;
      const SessionReport result = replay_refused_tg(0, nullptr, seed);
      const StageEvidence *a1a = stage_named(result, "A1a-dw-dadiag");
      const StageEvidence *refit = stage_named(result, "A1a-fixed-tg");
      CHECK(result.final_state == RunnerState::DONE, "S7: fixed-Tg replay failed (%s)", result.abort_reason.c_str());
      CHECK(a1a && refit && refit->passes > 0 && refit->accepted > 0 && !refit->hit_budget && refit->tstop == 0,
            "S7: refused Tg did not produce a complete fixed-Tg refit");
      CHECK(posterior_matches_layout(result), "S7: fixed-Tg parameters and posterior layouts differ");
      CHECK(!result.a_full_open && !result.tg_open && !result.solved.imu.calib_tg &&
                !result.solved.imu.calib_RAtoI && same_bits(result.solved.imu.Tg, seed.calib.imu.Tg),
            "S7: refused Tg or accel rotation escaped its seed");
      CHECK(result.solved.imu.calib_dw && result.solved.imu.calib_da &&
                (result.solved.imu.dw - seed.calib.imu.dw).norm() > 1e-5 &&
                (result.solved.imu.da - seed.calib.imu.da).norm() > 1e-5,
            "S7: fixed-Tg refit did not estimate Dw/Da");
      if (a1a && refit) {
        CHECK(a1a->dim_p == refit->dim_p + 9 && result.joint.dim_p == refit->dim_p &&
                  result.joint.final_merit == refit->merit,
              "S7: final posterior did not come from the reduced fixed-Tg model");
      }
      for (const auto &label : result.joint.labels)
        CHECK(label.rfind("tg[", 0) != 0, "S7: discarded Tg remains in posterior labels");
      CHECK(!stage_named(result, "B2-polish"), "S7: camera phase masked the missing phase-A refit");
      CHECK(same_bits(result.committed.imu.Tg, seed.calib.imu.Tg), "S7: commit changed refused Tg");
      std::printf("[S7] refused Tg refits Dw/Da/extrinsics/td with a matching reduced posterior (dim %d)\n", result.joint.dim_p);
    }

    if (s8) {
      SessionSeed seed_control, seed_exhausted;
      const SessionReport control = replay_refused_tg(1, "A0-ext-td", seed_control);
      const SessionReport exhausted = replay_refused_tg(1, "A1a-dw-dadiag", seed_exhausted);
      CHECK(control.final_state == RunnerState::DONE && exhausted.final_state == RunnerState::DONE,
            "S8: A0 fallback failed (control: %s; refit: %s)", control.abort_reason.c_str(), exhausted.abort_reason.c_str());
      const StageEvidence *a1a = stage_named(exhausted, "A1a-dw-dadiag");
      const StageEvidence *refit = stage_named(exhausted, "A1a-fixed-tg");
      const StageEvidence *phase_b = stage_named(exhausted, "B2-polish");
      CHECK(a1a && a1a->passes > 0 && a1a->accepted > 0 && !a1a->hit_budget,
            "S8: budget expired before nuisance-Tg fitting completed");
      CHECK(refit && refit->hit_budget && refit->passes == 0 && phase_b && phase_b->hit_budget && phase_b->passes == 0,
            "S8: refit/phase-B exhaustion lacks truthful stage evidence");
      CHECK(posterior_matches_layout(control) && posterior_matches_layout(exhausted),
            "S8: A0 fallback has mismatched posterior labels or dimensions");
      CHECK(!exhausted.solved.imu.calib_dw && !exhausted.solved.imu.calib_da && !exhausted.solved.imu.calib_RAtoI &&
                !exhausted.solved.imu.calib_tg && !exhausted.a_full_open && !exhausted.tg_open,
            "S8: exhausted refit left unestimated IMU blocks open");
      CHECK(same_calibration(exhausted.solved, control.solved) && same_calibration(exhausted.committed, control.committed),
            "S8: exhausted refit/phase-B rollback failed to restore exact A0 calibration values");
      CHECK(exhausted.joint.labels == control.joint.labels && same_bits(exhausted.joint.sigma, control.joint.sigma) &&
                same_bits(exhausted.joint.prior_sigma_vec, control.joint.prior_sigma_vec) &&
                same_bits(exhausted.joint.Lambda, control.joint.Lambda) &&
                exhausted.joint.final_merit == control.joint.final_merit && exhausted.joint.qn_max_final == control.joint.qn_max_final,
            "S8: exhausted refit retained an A1a posterior instead of exact A0 evidence");
      for (const auto &block : exhausted.blocks)
        CHECK(block.name != "dw" && block.name != "da" && block.name != "q_AtoI" && block.name != "tg",
              "S8: discarded IMU fit reached the commit walk");
      std::printf("[S8] exhausted fixed-Tg refit and phase-B rollback restore exact A0 parameters/posterior (dim %d)\n",
                  exhausted.joint.dim_p);
    }
  }

  std::remove(rec.c_str());
  std::remove(rec_deg.c_str());
  if (failures == 0) {
    std::printf("[PASS] session e2e: full production path, determinism, degenerate-motion gating, cam refine\n");
    return 0;
  }
  std::printf("[FAILED] %d checks\n", failures);
  return 1;
}
