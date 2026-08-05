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
 *     kappa_hat = mean( T * kappa_config / r )
 *
 * (T is computed deflated: C = kappa_config (A1^-1 + A2^-1); multiplying back
 * recovers the raw statistic). kappa_hat pins a_info_deflate PER SHAPE — the
 * 2026-07-11 thin-shape discordant pair (split freezes, wald CONSISTENT via
 * GN-linearized understatement) is exactly the failure class this harness
 * exists to bound before mode 1 takes profile authority. The H0 freeze rate
 * estimates the test's size (nominal ~1-3% family-wise at the 0.99
 * thresholds; N here is small, so ctest asserts only loose sanity — the real
 * study runs standalone with --reps >= 30).
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
static void write_session_record(const synth::Truth &tr, const std::string &path, double dur, double ex0, double ex1, unsigned rng) {
  synth::Trajectory tj;
  tj.excite_t0 = ex0;
  tj.excite_t1 = ex1;
  synth::StreamOptions so;
  so.dur = dur;
  so.excite = {{ex0, ex1}};
  std::vector<RawImu> imu;
  std::vector<FrameObs> frames;
  synth::make_streams(tr, tj, so, rng, imu, frames);

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
  int reps = 3; // ctest default: loose sanity at suite-compatible cost; the real study runs --reps >= 30
  bool p4 = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--reps") && i + 1 < argc)
      reps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--p4"))
      p4 = true; // wald-verdict stability under the P4 solver (the composed-endgame validation)
  }

  const synth::Truth tr = synth::make_truth();
  std::vector<double> khat;
  int frozen = 0, unobservable = 0, aborted = 0;

  for (int rep = 0; rep < reps; ++rep) {
    const std::string rec = "/tmp/ov_zcalib_wald_mc_" + std::to_string(rep) + ".bin";
    write_session_record(tr, rec, 90.0, 5.0, 88.0, 9001u + 131u * (unsigned)rep);

    SessionConfig cfg;
    cfg.harvester.pix_sigma = 0.5;
    cfg.out_yaml = "/tmp/ov_zcalib_wald_mc.yaml";
    cfg.verbose = false;
    cfg.joint.verbose = false;
    cfg.cam_mode = 0;
    cfg.a_gate_mode = 1; // wald DECIDES: per-replicate cost drops (no halves) and the mode-1 H0 size is measured directly
    cfg.p4 = p4;

    SessionReport rp;
    const bool ok = CalibSessionRunner::run_replay(rec, cfg, rp);
    std::remove(rec.c_str());
    if (!ok || rp.final_state != RunnerState::DONE) {
      aborted++;
      std::printf("[mc %02d] ABORT (%s)\n", rep, rp.abort_reason.c_str());
      continue;
    }
    using V = SessionReport::AccelGateVerdict;
    if (rp.a_wald_verdict == V::WALD_UNOBSERVABLE || rp.a_wald_r <= 0) {
      unobservable++;
      std::printf("[mc %02d] UNOBSERVABLE r=%d\n", rep, rp.a_wald_r);
      continue;
    }
    const double k = rp.a_wald_T * cfg.a_info_deflate / std::max(1, rp.a_wald_r);
    khat.push_back(k);
    if (!rp.a_full_open)
      frozen++;
    std::printf("[mc %02d] verdict=%s T=%.2f r=%d kappa_rep=%.2f dqA=%.3f deg\n", rep, rp.a_full_open ? "CONSISTENT" : "FROZEN",
                rp.a_wald_T, rp.a_wald_r, k, rp.a_wald_dqa_deg);
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
  CHECK(aborted == 0, "H0 replicates must complete (%d aborted)", aborted);
  CHECK(!khat.empty(), "no replicate reached the wald statistics");
  CHECK(kmean > 0.2 && kmean < 40.0, "kappa_hat %.2f outside sanity range", kmean);

  if (failures == 0)
    std::printf("test_wald_mc: ALL PASS (kappa_hat %.2f over %zu H0 replicates)\n", kmean, khat.size());
  else
    std::printf("test_wald_mc: %d FAILURES\n", failures);
  return failures == 0 ? 0 : 1;
}
