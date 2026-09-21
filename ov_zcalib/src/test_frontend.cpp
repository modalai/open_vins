/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib front-end gates (production path pieces, synthetic truth):
 *
 *  [F1] LinearSeed: arrowhead Dong-Si seeds from bearings + gyro chain at the
 *       SEED calibration (wrong extrinsics/td/intrinsics, bootstrap-grade bg),
 *       gates on gravity/kinematic/feature seed error, determinism, and
 *       micro-BA convergence parity vs truth-seeded solves.
 *  [F2] WindowHarvester: excitation gating, gap split, drop invalidation,
 *       clone subsampling, track filtering, mid-exposure clone stamps.
 *  [F3] WindowScorer: diversity retention beats FIFO on fused logdet; greedy
 *       logdet selection; holdout protection; temperature-span gate.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <cstdio>
#include <cstring>
#include <csignal>
#include <limits>
#include <sys/resource.h>

#include "utils/SessionRecord.h"
#include "core/CalibSession.h"
#include "core/CalibSessionRunner.h"
#include "solve/JointCalib.h"
#include "sim/SynthWorld.h"
#include "window/LinearSeed.h"
#include "window/WindowHarvester.h"
#include "window/WindowScorer.h"

using namespace ov_zcalib;

namespace ov_zcalib {
struct LinearSeedBiasTestAccess {
  static Eigen::Vector3d solve(const WindowData &window, const SharedCalib &calib, const Eigen::Vector3d &bg, int iters) {
    return LinearSeed::bias_presolve(window, calib, bg, iters);
  }
};
/// Install known admission information without running a nonlinear calibration. All retention,
/// thermal eligibility, live precision and motion-prompt code below is the production path.
struct CalibCollectionTestAccess {
  static Eigen::VectorXd prepare(CalibSessionRunner &runner) {
    runner.scorer_.reset(new WindowScorer(runner.cfg_.scorer));
    runner.slots_.resize(runner.cfg_.scorer.capacity);
    runner.slot_rep_.resize(runner.cfg_.scorer.capacity);
    Eigen::VectorXd prior(runner.calib_.local_dim());
    std::vector<std::string> labels;
    int off = 0;
    for (const auto &b : runner.calib_.free_blocks()) {
      const auto found = runner.cfg_.joint.prior_sigma.find(b.name);
      const double sigma = found == runner.cfg_.joint.prior_sigma.end() ? 1.0 : found->second;
      for (int k = 0; k < b.lsize; ++k) {
        prior(off++) = sigma;
        labels.push_back(b.name + "[" + std::to_string(k) + "]");
      }
    }
    runner.display_.reset(new CalibSession(prior.size(), prior, labels));
    return prior;
  }
  static ReservoirDecision retain(CalibSessionRunner &runner, const WindowMeta &meta, const Eigen::MatrixXd &information) {
    const ReservoirDecision d = runner.scorer_->consider(meta);
    if (d.accepted) {
      runner.slots_[d.slot].clone_times = {meta.t0, meta.t1};
      runner.slot_rep_[d.slot].Lambda = information;
      runner.refresh_collection_guidance_();
    }
    return d;
  }
};
} // namespace ov_zcalib

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

static SharedCalib make_seed_calib(const synth::Truth &tr) {
  SharedCalib c;
  c.imu = ImuIntrinsicModel(); // identity intrinsics seed
  Eigen::Vector4d dq;
  dq << 0.5 * 0.012, 0.5 * 0.009, 0.5 * -0.015, 1.0; // ~1.2 deg extrinsic seed error
  c.cams[0].q_ItoC = ov_core::quat_multiply(dq / dq.norm(), tr.q_ItoC);
  c.cams[0].p_IinC = tr.p_IinC + Eigen::Vector3d(0.006, -0.004, 0.005);
  c.cams[0].cam = tr.cam;
  c.cams[0].img_w = tr.img_w;
  c.cams[0].img_h = tr.img_h;
  c.cams[0].td = 0.0; // 2.5 ms seed error vs truth
  c.cams[0].tr = 0.0;
  return c;
}

int main() {
  synth::Truth tr = synth::make_truth();

  // ---------------- F1: linear seeding at the SEED calibration ----------------
  {
    // Window laid down at the SEED td (0), truth td = 2.5 ms -- exactly production.
    WindowData w = synth::make_window(tr, 0.4, 3.0, 20.0, 800.0, 42, 0.5, /*td_ref*/ 0.0);
    CHECK(!w.has_seeds, "F1: synth window must arrive seedless");
    SharedCalib c = make_seed_calib(tr);
    const Eigen::Vector3d bg_boot = 1.2 * tr.bg; // bootstrap-grade bias seed
    LinearSeedReport rep;
    CHECK(LinearSeed::seed_window(w, c, bg_boot, rep), "F1: linear seed failed");
    CHECK(w.has_seeds, "F1: seeds not emitted");
    std::printf("[F1] |g|=%.3f (9.81) ang_resid=%.2f mrad solved=%d fallback=%d med_depth=%.2f\n", rep.g_mag, 1e3 * rep.mean_ang_resid,
                rep.feats_solved, rep.feats_fallback, rep.median_depth);

    // seed-vs-truth: gravity direction, per-clone kinematics (window frame = I0 at t0+td offsets absorbed)
    synth::Trajectory tj;
    tj.phase = 0.13 * 0.4;
    const double t0i = w.clone_times.front() + (tr.td - w.td_ref[0]); // true imaging time of clone 0
    const Eigen::Matrix3d R0 = tj.R_of(t0i);
    const Eigen::Vector3d g_true_win = R0 * tr.g_W;
    const double g_ang = std::acos(std::min(1.0, w.seed_grav.normalized().dot(g_true_win.normalized()))) * 180.0 / M_PI;
    double v_err = 0.0, p_err = 0.0, q_err = 0.0;
    for (size_t k = 0; k < w.clone_times.size(); ++k) {
      const double tki = w.clone_times[k] + (tr.td - w.td_ref[0]);
      const double h = 1e-5;
      const Eigen::Vector3d vW = (tj.p_of(tki + h) - tj.p_of(tki - h)) / (2 * h);
      v_err = std::max(v_err, (w.seed_v[k] - R0 * vW).norm());
      p_err = std::max(p_err, (w.seed_p[k] - R0 * (tj.p_of(tki) - tj.p_of(t0i))).norm());
      const Eigen::Matrix3d Rk_true = tj.R_of(tki) * R0.transpose();
      q_err = std::max(q_err, ov_core::log_so3(ov_core::quat_2_Rot(w.seed_q[k]) * Rk_true.transpose()).norm());
    }
    std::printf("[F1] seed errs: grav %.2f deg | q %.2f deg | v %.3f m/s | p %.3f m\n", g_ang, q_err * 180.0 / M_PI, v_err, p_err);
    CHECK(g_ang < 3.0, "F1: gravity seed err %.2f deg", g_ang);
    CHECK(q_err * 180.0 / M_PI < 1.5, "F1: orientation seed err %.2f deg", q_err * 180.0 / M_PI);
    CHECK(v_err < 0.20, "F1: velocity seed err %.3f", v_err);
    // seed-ACCURACY bound only: the residual carries the not-yet-calibrated td
    // (2.5 ms) + extrinsic (1.2 deg) truth gap by design; the convergence-parity
    // check below is the load-bearing gate.
    CHECK(p_err < 0.25, "F1: position seed err %.3f", p_err);
    CHECK(rep.feats_solved > (int)w.num_feats * 3 / 4, "F1: only %d/%d tracks solved", rep.feats_solved, (int)w.num_feats);

    // determinism
    WindowData w2 = synth::make_window(tr, 0.4, 3.0, 20.0, 800.0, 42, 0.5, 0.0);
    LinearSeedReport rep2;
    CHECK(LinearSeed::seed_window(w2, c, bg_boot, rep2), "F1: reseed failed");
    bool bit_id = (w2.seed_grav - w.seed_grav).norm() == 0.0 && w2.seed_q.size() == w.seed_q.size();
    for (size_t k = 0; bit_id && k < w.seed_q.size(); ++k)
      bit_id = (w2.seed_q[k] - w.seed_q[k]).norm() == 0.0 && (w2.seed_p[k] - w.seed_p[k]).norm() == 0.0;
    CHECK(bit_id, "F1: linear seed nondeterministic");

    // micro-BA basin parity: linear seeds vs truth seeds at the same seed calib
    WindowSolveReport r_lin;
    CHECK(WindowBA::solve_and_export(w, c, false, r_lin, 30, false), "F1: BA(linear seeds) failed");
    // truth-seeded twin (fill seeds from truth like the frozen e2e harness does)
    WindowData wt = synth::make_window(tr, 0.4, 3.0, 20.0, 800.0, 42, 0.5, 0.0);
    {
      std::mt19937 rng(9);
      wt.has_seeds = true;
      auto pf = synth::make_cloud(90, 42 ^ 0x9e3779b9u); // discarded; truth seeds are rebuilt from the window's own geometry
      (void)pf;
      // exact truth seeds by re-projecting the window's own geometry:
      // window frame = I0 at clone0 TRUE imaging time
      wt.seed_q.clear();
      wt.seed_v.clear();
      wt.seed_p.clear();
      for (size_t k = 0; k < wt.clone_times.size(); ++k) {
        const double tki = wt.clone_times[k] + (tr.td - wt.td_ref[0]);
        const Eigen::Matrix3d Rk = tj.R_of(tki) * R0.transpose();
        wt.seed_q.push_back(ov_core::rot_2_quat(Rk));
        const double h = 1e-5;
        wt.seed_v.push_back(R0 * (tj.p_of(tki + h) - tj.p_of(tki - h)) / (2 * h));
        wt.seed_p.push_back(R0 * (tj.p_of(tki) - tj.p_of(t0i)));
      }
      wt.seed_bg = tr.bg;
      wt.seed_ba = tr.ba;
      wt.seed_grav = g_true_win;
      // features: reuse the LinearSeed solutions; exact truth features are unnecessary for basin parity
      wt.seed_feats = w.seed_feats;
    }
    WindowSolveReport r_tru;
    CHECK(WindowBA::solve_and_export(wt, c, false, r_tru, 30, false), "F1: BA(truth seeds) failed");
    std::printf("[F1] micro-BA cost: linear-seeded %.4e vs truth-seeded %.4e (ratio %.3f)\n", r_lin.cost_final, r_tru.cost_final,
                r_lin.cost_final / std::max(r_tru.cost_final, 1e-300));
    CHECK(r_lin.cost_final < 1.25 * r_tru.cost_final, "F1: linear seeds land in a worse basin (%.3e vs %.3e)", r_lin.cost_final,
          r_tru.cost_final);
  }

  // Merged poses may carry bearings from differently oriented cameras. The
  // gyro-bias pre-solve must use each camera's own rotation pairs, regardless
  // of which camera's observations happen to come first in a merged pose.
  {
    auto merged_truth = tr;
    merged_truth.imu = ImuIntrinsicModel();
    SharedCalib c = make_seed_calib(merged_truth);
    c.imu = merged_truth.imu;
    c.cams[0].q_ItoC = tr.q_ItoC;
    c.cams[0].p_IinC = tr.p_IinC;
    c.cams[0].td = tr.td;
    c.cams.push_back(c.cams[0]);
    const Eigen::Matrix3d R = ov_core::exp_so3(Eigen::Vector3d(0, 0, 1.4));
    c.cams[1].q_ItoC = ov_core::rot_2_quat(R * ov_core::quat_2_Rot(tr.q_ItoC));
    c.cams[1].p_IinC = R * tr.p_IinC;
    WindowData a = synth::make_window(merged_truth, 2.0, 2.0, 10.0, 800.0, 73, 0.0, tr.td);
    WindowData b = a;
    const size_t nf = a.num_feats;
    a.num_feats = b.num_feats = 2 * nf;
    a.td_ref.push_back(tr.td);
    b.td_ref.push_back(tr.td);
    for (size_t k = 0; k < a.obs.size(); ++k) {
      std::vector<CloneObs> other = a.obs[k];
      for (auto &o : other) {
        o.cam = 1;
        o.feat_id += nf;
        o.bearing = R * o.bearing;
      }
      a.obs[k].insert(a.obs[k].end(), other.begin(), other.end());
      b.obs[k].insert(b.obs[k].begin(), other.begin(), other.end());
    }
    LinearSeedConfig cfg;
    cfg.bias_presolve = true;
    const Eigen::Vector3d bg = tr.bg + Eigen::Vector3d(0.01, -0.008, 0.005);
    LinearSeedReport ra, rb;
    const bool oka = LinearSeed::seed_window(a, c, bg, ra, cfg);
    const bool okb = LinearSeed::seed_window(b, c, bg, rb, cfg);
    std::printf("[F1-merged] bias shifts %.6f / %.6f, order delta %.6g\n", ra.bg_shift, rb.bg_shift,
                (a.seed_bg - b.seed_bg).norm());
    CHECK(oka && okb, "F1-merged: valid two-camera seed rejected");
    CHECK(ra.bg_shift > 1e-4 && ra.bg_shift <= cfg.max_bg_shift && (a.seed_bg - bg).norm() > 1e-4,
          "F1-merged: bias correction was not accepted");
    CHECK(oka && okb && std::abs(ra.bg_shift - rb.bg_shift) < 1e-12 && (a.seed_bg - b.seed_bg).norm() < 1e-12,
          "F1-merged: gyro bias depends on camera observation order");
  }

  // A non-default gravity magnitude must survive seeding and the S2 BA update.
  // Otherwise reading a calibration-specific value at ingest would have no effect.
  {
    auto local = tr;
    local.g_W *= 9.79321 / local.g_W.norm();
    SharedCalib c = make_seed_calib(local);
    c.grav_mag = local.g_W.norm();
    WindowData w = synth::make_window(local, 0.4, 3.0, 20.0, 800.0, 42, 0.5, 0.0);
    LinearSeedReport sr;
    CHECK(LinearSeed::seed_window(w, c, 1.2 * local.bg, sr), "F1-gravity: seed failed");
    CHECK(std::abs(w.seed_grav.norm() - c.grav_mag) < 1e-10, "F1-gravity: seed changed configured norm");
    WindowSolveReport br;
    WindowWarmState optimized;
    CHECK(WindowBA::solve_and_export(w, c, false, br, 30, false, &optimized), "F1-gravity: BA failed");
    CHECK(optimized.valid && std::abs(optimized.grav.norm() - c.grav_mag) < 1e-10,
          "F1-gravity: BA changed configured norm");
  }

  // The bias presolve must use the same raw-sensor bias convention and
  // intrinsic model as the ACI seed. Two cameras observe different optical
  // instants on merged clones; their current td values differ from td_ref.
  // The IMU grid deliberately misses the optical endpoints. Independent
  // analytic rotation truth (not the presolve's integration) supplies both
  // bearings and raw samples, so this catches sign/Dw/Tg and timestamp errors.
  // Pure rotation makes the visual rotation oracle exact; it cannot seed
  // metric translation/depth, so exercise the production presolve directly.
  {
    auto truth = tr;
    truth.imu.dw << 0.91, 0.034, 1.07, -0.026, 0.018, 0.95;
    truth.imu.da << 1.035, -0.02, 0.975, 0.017, -0.013, 1.02;
    truth.imu.Tg << 0.0030, -0.0011, 0.0007,
                   -0.0008, 0.0021, -0.0014,
                    0.0012, 0.0006, -0.0027;
    truth.imu.calib_tg = false; // fixed nonzero values still correct measurements
    truth.ba.setZero(); // the linear seeder's accel-bias linearization
    truth.bg << 0.013, -0.009, 0.006;
    truth.cam << 220, 222, 640, 400, 0, 0, 0, 0;
    truth.img_w = 1280;
    truth.img_h = 800;
    auto other = truth;
    const Eigen::Matrix3d R_other = ov_core::exp_so3(Eigen::Vector3d(0.05, -0.08, 0.65));
    other.q_ItoC = ov_core::rot_2_quat(R_other * ov_core::quat_2_Rot(truth.q_ItoC));
    other.p_IinC = R_other * truth.p_IinC;
    SharedCalib c = make_seed_calib(truth);
    c.imu = truth.imu;
    c.cams[0].q_ItoC = truth.q_ItoC;
    c.cams[0].p_IinC = truth.p_IinC;
    c.cams[0].td = 0.0143;
    c.cams.push_back(c.cams[0]);
    c.cams[1].q_ItoC = other.q_ItoC;
    c.cams[1].p_IinC = other.p_IinC;
    c.cams[1].td = -0.0097;
    synth::Trajectory tj;
    tj.phase = 0.27;
    WindowData base;
    base.td_ref = {0.0041, -0.0013};
    const auto cloud = synth::make_cloud(150, 9173);
    std::vector<int> feature_id(2 * cloud.size(), -1);
    int next_feature = 0;
    for (int k = 0; k < 26; ++k) {
      const double t = 0.731 + k / 13.0;
      base.clone_times.push_back(t);
      base.obs.emplace_back();
      for (int cam = 0; cam < 2; ++cam) {
        const double merge_dt = cam ? 0.00137 : 0.0;
        const double ti = t + merge_dt + c.cams[cam].td - base.td_ref[cam];
        const auto &ct = cam ? other : truth;
        for (size_t f = 0; f < cloud.size(); ++f) {
          const Eigen::Vector3d pc = ov_core::quat_2_Rot(ct.q_ItoC) * tj.R_of(ti) * cloud[f];
          if (pc.z() <= 0.4)
            continue;
          const Eigen::Vector2d uv(ct.cam(0) * pc.x() / pc.z() + ct.cam(2),
                                    ct.cam(1) * pc.y() / pc.z() + ct.cam(3));
          CloneObs obs;
          obs.cam = cam;
          int &id = feature_id[cam * cloud.size() + f];
          if (id < 0)
            id = next_feature++;
          obs.feat_id = (size_t)id;
          obs.dt_ref = merge_dt;
          obs.uv = uv;
          obs.bearing = Eigen::Vector3d((uv.x() - ct.cam(2)) / ct.cam(0),
                                        (uv.y() - ct.cam(3)) / ct.cam(1), 1.0).normalized();
          base.obs.back().push_back(obs);
        }
      }
    }
    base.num_feats = (size_t)next_feature;
    const Eigen::Matrix3d Dw_inv = ImuIntrinsicModel::ut(truth.imu.dw).inverse();
    const Eigen::Matrix3d Da_inv = ImuIntrinsicModel::ut(truth.imu.da).inverse();
    const Eigen::Matrix3d RA_transpose = ov_core::quat_2_Rot(truth.imu.q_AtoI).transpose();
    for (double ti = base.clone_times.front() - 0.07123; ti <= base.clone_times.back() + 0.07; ti += 1.0 / 337.0) {
      RawImu sample;
      sample.timestamp = ti;
      const Eigen::Vector3d accel = tj.R_of(ti) * truth.g_W;
      sample.wm = Dw_inv * tj.omega_I(ti) + truth.bg + truth.imu.Tg * accel;
      sample.am = Da_inv * (RA_transpose * accel);
      base.imu.push_back(sample);
    }
    double worst = 0.0;
    for (const Eigen::Vector3d &perturb : {Eigen::Vector3d(0.010, -0.008, 0.005),
                                          Eigen::Vector3d(-0.007, 0.006, -0.011)}) {
      const Eigen::Vector3d estimated = LinearSeedBiasTestAccess::solve(base, c, truth.bg + perturb, 3);
      const double err = (estimated - truth.bg).norm();
      worst = std::max(worst, err);
      CHECK(err < 2e-4, "F1-bias-model: raw gyro bias recovery error %.6g", err);
    }
    // An out-of-record time update must not turn a shortened integral into
    // a fictitious gyro-bias correction. No valid pairs => preserve input.
    SharedCalib outside = c;
    outside.cams[0].td += 10.0;
    outside.cams[1].td += 10.0;
    const Eigen::Vector3d kept = LinearSeedBiasTestAccess::solve(base, outside, truth.bg, 3);
    CHECK((kept - truth.bg).norm() == 0.0, "F1-bias-model: uncovered intervals changed the bias");
    std::printf("[F1-bias-model] nonidentity Dw/Da/qA/Tg + per-camera td updates: worst bias error %.6g rad/s\n", worst);
  }

  // ---------------- F2: harvester on live-style streams ----------------
  {
    // The harvester runs AFTER bootstrap in the session pipeline, so its seed
    // calibration carries hand-eye/xcorr accuracy (~0.3 deg / ~0.3 ms), not the
    // cold-start error (F1 gates the cold path).
    SharedCalib seed = make_seed_calib(tr);
    {
      Eigen::Vector4d dqb;
      dqb << 0.5 * 0.003, 0.5 * -0.004, 0.5 * 0.002, 1.0; // ~0.3 deg hand-eye residual
      seed.cams[0].q_ItoC = ov_core::quat_multiply(dqb / dqb.norm(), tr.q_ItoC);
      seed.cams[0].td = tr.td - 0.0003; // 0.3 ms bootstrap residual
    }
    HarvesterConfig hc;
    hc.pix_sigma = 0.5;

    auto run_stream = [&](const synth::StreamOptions &so, unsigned rng, std::vector<WindowData> &wins, std::vector<WindowMeta> &metas,
                          WindowHarvester *hv_out = nullptr) {
      synth::Trajectory tj;
      tj.excite_t0 = so.excite.empty() ? -1.0 : so.excite[0].first;
      tj.excite_t1 = so.excite.empty() ? -2.0 : so.excite[0].second;
      std::vector<RawImu> imu;
      std::vector<FrameObs> frames;
      synth::make_streams(tr, tj, so, rng, imu, frames);
      WindowHarvester hv(hc, seed);
      size_t ii = 0;
      for (const FrameObs &f : frames) {
        while (ii < imu.size() && imu[ii].timestamp <= f.timestamp + 0.05)
          hv.push_imu(imu[ii++]);
        if (hv.push_frame(f)) {
          WindowData w;
          WindowMeta m;
          if (hv.pop_window(w, m)) {
            wins.push_back(std::move(w));
            metas.push_back(m);
          }
        }
      }
      while (ii < imu.size())
        hv.push_imu(imu[ii++]);
      if (hv.flush()) {
        WindowData w;
        WindowMeta m;
        if (hv.pop_window(w, m)) {
          wins.push_back(std::move(w));
          metas.push_back(m);
        }
      }
      if (hv_out)
        *hv_out = hv;
      return hv.windows_invalidated();
    };

    // A: clean, excited [8, 18] of a 30 s stream
    synth::StreamOptions soA;
    soA.dur = 30.0;
    soA.excite = {{8.0, 18.0}};
    std::vector<WindowData> wA;
    std::vector<WindowMeta> mA;
    run_stream(soA, 101, wA, mA);
    std::printf("[F2A] windows=%d", (int)wA.size());
    for (auto &m : mA)
      std::printf(" [%.1f,%.1f]", m.t0, m.t1);
    std::printf("\n");
    CHECK(!wA.empty(), "F2A: no windows harvested from the excited segment");
    for (size_t i = 0; i < wA.size(); ++i) {
      CHECK(mA[i].t0 > 7.0 && mA[i].t1 < 20.5, "F2A: window [%.1f,%.1f] outside excitation", mA[i].t0, mA[i].t1);
      const double dur = mA[i].t1 - mA[i].t0;
      CHECK(dur >= hc.min_window_s - 1e-9 && dur <= hc.max_window_s + 0.2, "F2A: duration %.2f out of range", dur);
      CHECK((int)wA[i].clone_times.size() <= hc.max_clones, "F2A: %d clones over budget", (int)wA[i].clone_times.size());
      CHECK(wA[i].imu.front().timestamp <= wA[i].clone_times.front() && wA[i].imu.back().timestamp >= wA[i].clone_times.back(),
            "F2A: imu slice does not cover clones");
      for (auto &obs : wA[i].obs)
        for (auto &o : obs) {
          CHECK(o.u_frac >= -0.5 && o.u_frac <= 0.5, "F2A: u_frac %.3f (centered convention)", o.u_frac);
          CHECK(std::abs(o.bearing.norm() - 1.0) < 1e-9, "F2A: non-unit bearing");
        }
    }
    // end-to-end production window path: harvest -> linear seed -> micro-BA
    {
      LinearSeedReport rep;
      CHECK(LinearSeed::seed_window(wA[0], seed, 1.2 * tr.bg, rep), "F2A: linear seed on harvested window failed");
      WindowSolveReport r;
      CHECK(WindowBA::solve_and_export(wA[0], seed, false, r, 30, false), "F2A: BA on harvested window failed");
      std::printf("[F2A] harvested window: seed ang=%.2f mrad, BA cost=%.3e (%d it)\n", 1e3 * rep.mean_ang_resid, r.cost_final,
                  r.iterations);
      CHECK(rep.mean_ang_resid < 0.020, "F2A: post-bootstrap seed resid %.1f mrad", 1e3 * rep.mean_ang_resid);
      CHECK(r.cost_final < 8e3, "F2A: BA stranded at %.3e", r.cost_final);
    }

    // B: two consecutive dropped frames at t=11.0 -> split boundary there
    synth::StreamOptions soB = soA;
    // frame seq numbers: frames start at tc=0.2, 30 fps -> t=11 s ~ seq 324
    soB.drop_frames = {324, 325};
    std::vector<WindowData> wB;
    std::vector<WindowMeta> mB;
    run_stream(soB, 101, wB, mB);
    std::printf("[F2B] windows=%d (split at 11.0 expected)", (int)wB.size());
    for (auto &m : mB)
      std::printf(" [%.1f,%.1f]", m.t0, m.t1);
    std::printf("\n");
    bool has_split_boundary = false;
    for (auto &m : mB)
      has_split_boundary = has_split_boundary || (m.t1 > 10.7 && m.t1 < 11.05);
    bool a_has_boundary = false;
    for (auto &m : mA)
      a_has_boundary = a_has_boundary || (m.t1 > 10.7 && m.t1 < 11.05);
    CHECK(has_split_boundary && !a_has_boundary, "F2B: no split boundary at the injected gap");

    // C: 1-in-7 scattered drops across the WHOLE excitation -> every window
    // tolerates each 1-frame gap but dies on the drop fraction
    synth::StreamOptions soC = soA;
    for (int s = 234; s < 600; s += 7)
      soC.drop_frames.push_back(s);
    std::vector<WindowData> wC;
    std::vector<WindowMeta> mC;
    const int invC = run_stream(soC, 101, wC, mC);
    std::printf("[F2C] windows=%d invalidated=%d (drop-frac gate)\n", (int)wC.size(), invC);
    CHECK(wC.empty() && invC > 0, "F2C: heavy-drop stream not invalidated (%d windows)", (int)wC.size());

    // D: quiet stream -> nothing harvested
    synth::StreamOptions soD;
    soD.dur = 15.0;
    soD.excite = {}; // never excited
    std::vector<WindowData> wD;
    std::vector<WindowMeta> mD;
    run_stream(soD, 101, wD, mD);
    CHECK(wD.empty(), "F2D: %d windows from a quiet stream", (int)wD.size());
  }

  // Exposure is metadata once ingest has stamped the optical instant. Changing
  // it must not move clones, alter cross-camera merging, or shift IMU endpoints.
  // Exercise both near-synchronous (merged) and asynchronous camera frames.
  for (int nc : {1, 2}) {
    for (double stagger : {0.001, 0.006}) {
      SharedCalib seed = make_seed_calib(tr);
      seed.cams.resize((size_t)nc, seed.cams[0]);
      if (nc == 2)
        seed.cams[1].td = seed.cams[0].td - 0.0005;
      synth::StreamOptions so;
      so.dur = 12.0;
      so.excite = {{2.0, 10.0}};
      synth::Trajectory tj;
      tj.excite_t0 = 2.0;
      tj.excite_t1 = 10.0;
      std::vector<RawImu> imu;
      std::vector<FrameObs> frames;
      synth::make_streams(tr, tj, so, 103, imu, frames);
      if (nc == 2) {
        const size_t nf = frames.size();
        for (size_t i = 0; i < nf; ++i) {
          FrameObs f = frames[i];
          f.cam = 1;
          f.timestamp += stagger;
          for (auto &p : f.pts)
            p.id += 1000; // independent camera tracks
          frames.push_back(std::move(f));
        }
        std::sort(frames.begin(), frames.end(), [](const FrameObs &a, const FrameObs &b) {
          return a.timestamp < b.timestamp;
        });
      }
      auto harvest = [&](bool varying_exposure) {
        WindowHarvester hv(HarvesterConfig(), seed);
        std::vector<WindowData> wins;
        size_t ii = 0;
        auto pop = [&]() {
          WindowData w;
          WindowMeta m;
          if (hv.pop_window(w, m))
            wins.push_back(std::move(w));
        };
        for (FrameObs f : frames) {
          f.exposure_s = varying_exposure ? 0.002f + 0.003f * (float)((f.seq + 2 * f.cam) % 5) : 0.f;
          while (ii < imu.size() && imu[ii].timestamp <= f.timestamp + 0.05)
            hv.push_imu(imu[ii++]);
          if (hv.push_frame(f))
            pop();
        }
        if (hv.flush())
          pop();
        return wins;
      };
      const auto ref = harvest(false), varied = harvest(true);
      bool same = !ref.empty() && ref.size() == varied.size();
      bool timing_ok = true, saw_merged = false;
      for (size_t i = 0; same && i < ref.size(); ++i) {
        same = ref[i].clone_times == varied[i].clone_times && ref[i].obs.size() == varied[i].obs.size() &&
               ref[i].imu.size() == varied[i].imu.size();
        for (size_t k = 0; same && k < ref[i].obs.size(); ++k) {
          same = ref[i].obs[k].size() == varied[i].obs[k].size();
          for (size_t j = 0; same && j < ref[i].obs[k].size(); ++j) {
            const auto &a = ref[i].obs[k][j], &b = varied[i].obs[k][j];
            same = a.cam == b.cam && a.feat_id == b.feat_id && a.dt_ref == b.dt_ref;
            timing_ok = timing_ok && b.dt_ref >= 0.0 && b.dt_ref < 0.002;
            saw_merged = saw_merged || b.dt_ref > 0.0;
          }
        }
      }
      CHECK(same && timing_ok, "F2E: exposure changed %d-camera timing (stagger %.3f)", nc, stagger);
      CHECK(saw_merged == (nc == 2 && stagger < 0.002), "F2E: did not exercise expected clone merging");
    }
  }

  // ---------------- F3: reservoir + selection ----------------
  {
    ScorerConfig sc;
    sc.capacity = 12;
    WindowScorer scorer(sc);
    // 3 excitation clusters arriving CLUSTERED (FIFO would keep only the first)
    auto mk = [&](int cluster, int i) {
      WindowMeta m;
      m.fingerprint.setConstant(0.05);
      if (cluster == 0)
        m.fingerprint(0) = 0.8 + 0.01 * i; // gyro-x heavy
      else if (cluster == 1)
        m.fingerprint(6) = 0.9 + 0.01 * i; // gravity-sweep heavy
      else
        m.fingerprint(7) = 140.0 + 1.0 * i; // flow heavy
      m.t0 = 10.0 * i + cluster * 300.0;
      m.t1 = m.t0 + 3.0;
      m.temp_mean = 30.0;
      m.fingerprint(10) = 3.0;
      return m;
    };
    std::vector<int> slot_cluster(sc.capacity, -1);
    for (int cl = 0; cl < 3; ++cl)
      for (int i = 0; i < 20; ++i) {
        ReservoirDecision d = scorer.consider(mk(cl, i));
        if (d.accepted)
          slot_cluster[d.slot] = cl;
      }
    int have[3] = {0, 0, 0};
    for (int j = 0; j < scorer.size(); ++j)
      if (slot_cluster[j] >= 0)
        have[slot_cluster[j]]++;
    std::printf("[F3] reservoir cluster coverage: %d/%d/%d (FIFO baseline would be 12/0/0)\n", have[0], have[1], have[2]);
    CHECK(have[0] > 0 && have[1] > 0 && have[2] > 0, "F3: reservoir lost a cluster (%d/%d/%d)", have[0], have[1], have[2]);

    // holdout protection: slots flagged holdout must survive all later evictions
    std::vector<int> holdout_slots;
    for (int j = 0; j < scorer.size(); ++j)
      if (scorer.is_holdout(j))
        holdout_slots.push_back(j);
    CHECK(!holdout_slots.empty(), "F3: no holdout windows flagged");

    // greedy logdet selection: complementary information beats duplicates
    const int np = 4;
    std::vector<Eigen::MatrixXd> Lw;
    std::vector<std::pair<double, double>> spans;
    for (int c = 0; c < 6; ++c) {
      Eigen::MatrixXd L = Eigen::MatrixXd::Zero(np, np);
      L((c < 3) ? 0 : (c - 2), (c < 3) ? 0 : (c - 2)) = 50.0; // 0,0,0,1,2,3
      Lw.push_back(L);
      spans.push_back({100.0 * c, 100.0 * c + 3.0});
    }
    double min_eig = 0.0;
    std::vector<int> sel = WindowScorer::select_logdet(Lw, spans, 4, 0.5, &min_eig);
    std::printf("[F3] logdet selected:");
    for (int s : sel)
      std::printf(" %d", s);
    std::printf("  min_eig=%.2f\n", min_eig);
    bool has345 = std::count(sel.begin(), sel.end(), 3) && std::count(sel.begin(), sel.end(), 4) && std::count(sel.begin(), sel.end(), 5);
    CHECK(has345, "F3: greedy missed complementary candidates");
    CHECK(min_eig > 1.0, "F3: min-eig report wrong (%.2f)", min_eig);

    // A generic min-eigenvalue > 5 cannot certify Tg at 0.12 prior sigma.
    // Correlation with another calibration parameter matters: 1/sqrt(H_ii)
    // would incorrectly report 0.1 here, while the marginal is sqrt(100/1164).
    {
      Eigen::Matrix2d L;
      L << 99.0, 94.0, 94.0, 99.0;
      std::vector<Eigen::MatrixXd> info = {L};
      std::vector<std::pair<double, double>> times = {{0.0, 2.0}};
      Eigen::VectorXd sigma;
      double eig = 0.0;
      const auto ids = WindowScorer::select_logdet(info, times, 1, 0.5, &eig, &sigma);
      CHECK(ids.size() == 1 && std::abs(eig - 6.0) < 1e-10, "F3: correlated information selection");
      CHECK(sigma.size() == 2 && std::abs(sigma(0) - std::sqrt(100.0 / 1164.0)) < 1e-12,
            "F3: collection precision must marginalize correlated parameters");
      CHECK(sigma(0) > 0.12, "F3: weak Tg must not certify at the generic min-eig threshold");
      info[0] = 99.0 * Eigen::Matrix2d::Identity();
      WindowScorer::select_logdet(info, times, 1, 0.5, &eig, &sigma);
      CHECK(sigma.size() == 2 && std::abs(sigma(0) - 0.1) < 1e-12,
            "F3: independently excited Tg must clear the precision target");
      WindowScorer::select_logdet({}, {}, 1, 0.5, &eig, &sigma);
      CHECK(sigma.size() == 0 && eig == 0.0, "F3: empty selection must not retain previous precision/information");
      info[0].setZero();
      CHECK(WindowScorer::select_logdet(info, times, 1, 0.5, &eig, &sigma).empty() && eig == 0.0 && sigma.size() == 0,
            "F3: a prior alone must not be reported as a measured selection");
      for (double invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        info[0] = Eigen::Matrix2d::Identity();
        info[0](0, 0) = invalid;
        CHECK(WindowScorer::select_logdet(info, times, 1, 0.5, &eig, &sigma).empty() && sigma.size() == 0,
              "F3: invalid information must refuse even under fast-math");
      }
      info[0] = Eigen::MatrixXd::Identity(2, 3);
      CHECK(WindowScorer::select_logdet(info, times, 1, 0.5, &eig, &sigma).empty(), "F3: nonsquare information must refuse");
      info[0] = Eigen::Matrix2d::Identity();
      CHECK(WindowScorer::select_logdet(info, {}, 1, 0.5, &eig, &sigma).empty(), "F3: missing intervals must refuse");
      times[0].second = std::numeric_limits<double>::quiet_NaN();
      CHECK(WindowScorer::select_logdet(info, times, 1, 0.5, &eig, &sigma).empty(), "F3: invalid interval must refuse");
      times[0] = {0.0, 2.0};
      info.push_back(99.0 * Eigen::Matrix2d::Identity());
      times.push_back({3.0, 5.0});
      info[0](0, 0) = std::numeric_limits<double>::quiet_NaN();
      const auto valid_only = WindowScorer::select_logdet(info, times, 1, 0.5, &eig, &sigma);
      CHECK(valid_only.size() == 1 && valid_only[0] == 1 && sigma.size() == 2,
            "F3: invalid candidates must not discard valid independently scored windows");
    }

    // thermal binning
    {
      ScorerConfig sc2;
      sc2.capacity = 16;
      sc2.holdout_every = 100;
      WindowScorer s2(sc2);
      for (int i = 0; i < 5; ++i) {
        WindowMeta m = mk(0, i);
        m.temp_mean = 26.0 + 0.5 * i;
        s2.consider(m);
      }
      for (int i = 0; i < 3; ++i) {
        WindowMeta m = mk(1, i);
        m.temp_mean = 41.0 + 0.5 * i;
        s2.consider(m);
      }
      std::vector<int> bin = s2.thermal_bin();
      std::printf("[F3] thermal bin size %d/8 (5 cool + 3 hot, gate 6C)\n", (int)bin.size());
      CHECK(bin.size() == 5, "F3: thermal bin picked %d (want the 5-window cool bin)", (int)bin.size());
    }
  }

  // ---------------- F4: record -> replay bit-parity ----------------
  {
    SharedCalib seed = make_seed_calib(tr);
    HarvesterConfig hc;
    hc.pix_sigma = 0.5;

    synth::Trajectory tj;
    tj.excite_t0 = 6.0;
    tj.excite_t1 = 16.0;
    synth::StreamOptions so;
    so.dur = 20.0;
    so.excite = {{6.0, 16.0}};
    std::vector<RawImu> imu;
    std::vector<FrameObs> frames;
    synth::make_streams(tr, tj, so, 777, imu, frames);

    // live-style pass WITH the mandatory record mirror (arrival order = time order)
    SessionSeed ss;
    ss.calib = seed;
    const std::string rec_path = "/tmp/ov_zcalib_f4_session.bin";
    std::vector<WindowData> w_live;
    {
      SessionRecordWriter wr;
      CHECK(wr.open(rec_path, ss), "F4: record open failed");
      WindowHarvester hv(hc, seed);
      size_t ii = 0;
      for (const FrameObs &f : frames) {
        while (ii < imu.size() && imu[ii].timestamp <= f.timestamp + 0.05) {
          hv.push_imu(imu[ii]);
          wr.write_imu(imu[ii]);
          ++ii;
        }
        wr.write_frame(f);
        if (hv.push_frame(f)) {
          WindowData w;
          WindowMeta m;
          if (hv.pop_window(w, m))
            w_live.push_back(std::move(w));
        }
      }
      if (hv.flush()) {
        WindowData w;
        WindowMeta m;
        if (hv.pop_window(w, m))
          w_live.push_back(std::move(w));
      }
    }
    // replay pass from the record ONLY
    std::vector<WindowData> w_rep;
    {
      SessionSeed ss2;
      WindowHarvester *hv = nullptr;
      WindowHarvester hv_store(hc, seed); // rebuilt below once the seed is read
      bool first = true;
      auto on_imu = [&](const RawImu &s) { hv->push_imu(s); };
      auto on_frame = [&](const FrameObs &f) {
        if (hv->push_frame(f)) {
          WindowData w;
          WindowMeta m;
          if (hv->pop_window(w, m))
            w_rep.push_back(std::move(w));
        }
      };
      SessionRecordReader rd;
      CHECK(rd.open(rec_path), "F4: record reopen failed");
      ss2 = rd.seed();
      CHECK((ss2.calib.cams[0].cam - ss.calib.cams[0].cam).norm() == 0.0 && ss2.calib.cams[0].td == ss.calib.cams[0].td,
            "F4: seed roundtrip mismatch");
      CHECK(ss2.calib.cams[0].reprojection_sigma_px == 0.0, "F4: inherited scalar noise must survive roundtrip");
      hv_store = WindowHarvester(hc, ss2.calib); // the REPLAYED seed drives bearings/td_ref
      hv = &hv_store;
      bool is_imu = false;
      RawImu s;
      FrameObs f;
      while (rd.next(is_imu, s, f)) {
        if (is_imu)
          on_imu(s);
        else
          on_frame(f);
      }
      CHECK(!rd.failed() && rd.complete(), "F4: replay did not reach a valid completion footer");
      (void)first;
      if (hv->flush()) {
        WindowData w;
        WindowMeta m;
        if (hv->pop_window(w, m))
          w_rep.push_back(std::move(w));
      }
    }
    CHECK(w_live.size() == w_rep.size(), "F4: window count live %d vs replay %d", (int)w_live.size(), (int)w_rep.size());
    bool bitid = w_live.size() == w_rep.size() && !w_live.empty();
    for (size_t i = 0; bitid && i < w_live.size(); ++i) {
      const WindowData &a = w_live[i], &b = w_rep[i];
      CHECK(a.pix_sigma == 0.5 && b.pix_sigma == 0.5, "F4: library caller's scalar noise changed during replay");
      bitid = a.clone_times == b.clone_times && a.num_feats == b.num_feats && a.imu.size() == b.imu.size() && a.td_ref == b.td_ref;
      for (size_t k = 0; bitid && k < a.imu.size(); ++k)
        bitid = a.imu[k].timestamp == b.imu[k].timestamp && (a.imu[k].wm - b.imu[k].wm).norm() == 0.0 &&
                (a.imu[k].am - b.imu[k].am).norm() == 0.0;
      for (size_t c = 0; bitid && c < a.obs.size(); ++c) {
        bitid = a.obs[c].size() == b.obs[c].size();
        for (size_t o = 0; bitid && o < a.obs[c].size(); ++o)
          bitid = a.obs[c][o].feat_id == b.obs[c][o].feat_id && (a.obs[c][o].uv - b.obs[c][o].uv).norm() == 0.0 &&
                  (a.obs[c][o].bearing - b.obs[c][o].bearing).norm() == 0.0 && a.obs[c][o].u_frac == b.obs[c][o].u_frac;
      }
    }
    std::printf("[F4] live-vs-replay: %d windows, bit-identical=%d\n", (int)w_live.size(), (int)bitid);
    CHECK(bitid, "F4: live vs replay NOT bit-identical");
    std::remove(rec_path.c_str());
  }

  // ---------------- F5: camera-noise record versions and malformed headers ----------------
  {
    const std::string path = "/tmp/ov_zcalib_f5_session.bin";
    SessionSeed ss;
    ss.calib.cams.resize(2);
    ss.calib.cams[0].reprojection_sigma_px = 0.7;
    ss.calib.cams[1].reprojection_sigma_px = 1.4;
    ss.profile = SessionProfile::VOXL_FLIGHT;
    ss.cam_mode = 1;
    RawImu sample;
    sample.timestamp = 2.5;
    sample.wm = Eigen::Vector3d(0.1, 0.2, 0.3);
    sample.am = Eigen::Vector3d(1.0, 2.0, 3.0);
    sample.temp_c = 23.4;
    FrameObs frame;
    frame.timestamp = 3.0;
    frame.cam = 1;
    frame.seq = 17;
    frame.pts.resize(2);
    frame.pts[0].id = 9;
    frame.pts[0].u = 123.0f;
    frame.pts[1].id = 10;
    frame.pts[1].v = 234.0f;
    {
      SessionRecordWriter wr;
      CHECK(wr.open(path, ss), "F5: format-7 record open failed");
      CHECK(wr.write_imu(sample) && wr.write_frame(frame), "F5: record write failed");
      CHECK(wr.close() && !wr.failed(), "F5: completion footer/flush/close failed");
    }
    SessionRecordReader rd;
    auto check_stream = [&]() {
      bool is_imu = false;
      RawImu got_imu;
      FrameObs got_frame;
      CHECK(rd.next(is_imu, got_imu, got_frame) && is_imu && got_imu.timestamp == sample.timestamp &&
                (got_imu.wm - sample.wm).norm() == 0.0 && (got_imu.am - sample.am).norm() == 0.0,
            "F5: IMU stream offset/payload changed");
      CHECK(rd.next(is_imu, got_imu, got_frame) && !is_imu && got_frame.cam == 1 && got_frame.seq == 17 &&
                got_frame.timestamp == frame.timestamp && got_frame.pts.size() == 2 && got_frame.pts[1].v == 234.0f,
            "F5: camera stream offset/payload changed");
      CHECK(!rd.next(is_imu, got_imu, got_frame), "F5: unexpected trailing record");
      CHECK(!rd.failed(), "F5: valid stream failed: %s", rd.error().c_str());
      CHECK(rd.complete() == rd.requires_completion_marker(), "F5: legacy EOF must not claim certified completeness");
    };
    CHECK(rd.open(path), "F5: format-7 record reopen failed");
    CHECK(rd.seed().calib.cams.size() == 2 && rd.seed().calib.cams[0].reprojection_sigma_px == 0.7 &&
              rd.seed().calib.cams[1].reprojection_sigma_px == 1.4 && rd.seed().profile == ss.profile &&
              rd.seed().cam_mode == ss.cam_mode,
          "F5: distinct camera noise/profile roundtrip failed");
    check_stream();

    std::vector<unsigned char> bytes;
    FILE *file = std::fopen(path.c_str(), "rb");
    CHECK(file != nullptr, "F5: could not read fixture bytes");
    if (file) {
      std::fseek(file, 0, SEEK_END);
      const long size = std::ftell(file);
      std::rewind(file);
      if (size > 0) {
        bytes.resize((size_t)size);
        CHECK(std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size(), "F5: fixture read incomplete");
      }
      std::fclose(file);
    }
    auto write_bytes = [&](const std::vector<unsigned char> &data) {
      rd.close();
      FILE *out = std::fopen(path.c_str(), "wb");
      CHECK(out != nullptr, "F5: fixture write open failed");
      if (out) {
        CHECK(std::fwrite(data.data(), 1, data.size(), out) == data.size(), "F5: fixture write incomplete");
        std::fclose(out);
      }
    };
    // Independent description of the historical v5 layout: fixed prefix/shared chain,
    // two camera descriptors, then profile and camera mode. No pixel-noise bytes existed.
    const size_t v5_header = 3 * sizeof(uint32_t) + 30 * sizeof(double) +
                             2 * (18 * sizeof(double) + 2 * sizeof(uint8_t) + 2 * sizeof(uint32_t)) +
                             sizeof(uint8_t) + sizeof(int32_t);
    const size_t v6_header = v5_header + 2 * sizeof(double);
    const size_t footer_size = sizeof(uint8_t) + sizeof(uint32_t) + 2 * sizeof(uint64_t);
    CHECK(bytes.size() > v6_header, "F5: record shorter than independently described header");
    if (bytes.size() > v6_header) {
      uint32_t format = 0;
      std::memcpy(&format, bytes.data() + sizeof(uint32_t), sizeof(format));
      CHECK(format == 7, "F5: writer must use explicit format 7");
      std::vector<unsigned char> legacy = bytes;
      legacy.resize(legacy.size() - footer_size);
      format = 6;
      std::memcpy(legacy.data() + sizeof(uint32_t), &format, sizeof(format));
      write_bytes(legacy);
      CHECK(rd.open(path), "F5: historical format-6 header rejected");
      CHECK(rd.seed().calib.cams.size() == 2 && rd.seed().calib.cams[0].reprojection_sigma_px == 0.7 &&
                rd.seed().calib.cams[1].reprojection_sigma_px == 1.4,
            "F5: format-6 camera noise changed");
      check_stream();
      legacy.erase(legacy.begin() + v5_header, legacy.begin() + v6_header);
      format = 5;
      std::memcpy(legacy.data() + sizeof(uint32_t), &format, sizeof(format));
      write_bytes(legacy);
      CHECK(rd.open(path), "F5: historical format-5 header rejected");
      CHECK(rd.seed().calib.cams.size() == 2 && rd.seed().calib.cams[0].reprojection_sigma_px == 0.0 &&
                rd.seed().calib.cams[1].reprojection_sigma_px == 0.0,
            "F5: reused reader retained format-6 noise in format-5 replay");
      check_stream();

      for (size_t cut = v5_header; cut < v6_header; ++cut) {
        write_bytes(std::vector<unsigned char>(bytes.begin(), bytes.begin() + cut));
        CHECK(!rd.open(path), "F5: truncated camera-noise header accepted (%zu bytes)", cut);
      }
      for (uint64_t invalid : {UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
                               UINT64_C(0x7ff8000000000000), UINT64_C(0xbff0000000000000),
                               UINT64_C(0x8000000000000001)}) {
        std::vector<unsigned char> bad = bytes;
        std::memcpy(bad.data() + v5_header + sizeof(double), &invalid, sizeof(invalid));
        write_bytes(bad);
        CHECK(!rd.open(path), "F5: nonfinite/negative camera sigma accepted");
        SessionSeed bad_seed = ss;
        std::memcpy(&bad_seed.calib.cams[1].reprojection_sigma_px, &invalid, sizeof(invalid));
        SessionRecordWriter wr;
        CHECK(!wr.open(path, bad_seed), "F5: writer accepted nonfinite/negative camera sigma");
      }
      for (uint32_t unsupported : {4u, 8u}) {
        std::vector<unsigned char> bad = bytes;
        std::memcpy(bad.data() + sizeof(uint32_t), &unsupported, sizeof(unsupported));
        write_bytes(bad);
        CHECK(!rd.open(path), "F5: unsupported format %u accepted", unsupported);
      }
      auto check_read_failure = [&](const std::vector<unsigned char> &bad, const char *context) {
        write_bytes(bad);
        CHECK(rd.open(path), "F5: %s must retain a valid header", context);
        bool is_imu = false;
        RawImu got_imu;
        FrameObs got_frame;
        while (rd.next(is_imu, got_imu, got_frame)) {}
        CHECK(rd.failed() && !rd.complete() && !rd.error().empty(), "F5: %s treated as clean EOF", context);
        CHECK(!rd.next(is_imu, got_imu, got_frame) && rd.failed(), "F5: reader failure must stay sticky");
        SessionSeed ignored;
        CHECK(!replay_session(path, ignored, [](const RawImu &) {}, [](const FrameObs &) {}),
              "F5: replay_session swallowed %s", context);
      };
      // Every partial body length includes record-boundary EOF, short IMU/frame payloads,
      // and missing/partial footers. None is a complete format-7 session.
      for (size_t cut = v6_header; cut < bytes.size(); ++cut)
        check_read_failure(std::vector<unsigned char>(bytes.begin(), bytes.begin() + cut), "truncated format-7 body");
      std::vector<unsigned char> bad = bytes;
      bad.back() ^= 1; // impossible frame count in an otherwise complete footer
      check_read_failure(bad, "footer count mismatch");
      bad = bytes;
      bad[bytes.size() - footer_size + 1] ^= 1;
      check_read_failure(bad, "footer magic mismatch");
      bad = bytes;
      bad.push_back(0);
      check_read_failure(bad, "data after completion footer");
      bad = bytes;
      bad[v6_header] = 255;
      check_read_failure(bad, "unknown record tag");
      // Legacy compatibility must not preserve the old mid-record truncation bug.
      legacy.pop_back();
      check_read_failure(legacy, "truncated format-5 frame points");
      bad.assign(bytes.begin(), bytes.end() - footer_size - 1);
      format = 6;
      std::memcpy(bad.data() + sizeof(uint32_t), &format, sizeof(format));
      check_read_failure(bad, "truncated format-6 frame points");
    }
    rd.close();
    std::remove(path.c_str());
    // /dev/full catches an already-full filesystem while flushing the initial header.
    {
      SessionRecordWriter full;
      if (full.open("/dev/full", ss)) {
        full.write_imu(sample);
        CHECK(!full.flush(), "F5: disk-full flush falsely succeeded");
      }
      CHECK(full.failed() && !full.error().empty(), "F5: disk-full error was not exposed");
      const std::string first_error = full.error();
      CHECK(!full.write_frame(frame) && !full.close() && !full.close(), "F5: disk-full write/close failure was not sticky");
      CHECK(full.error() == first_error, "F5: later failure hid the original storage error");
      SessionRecordWriter close_full;
      close_full.open("/dev/full", ss);
      close_full.write_imu(sample);
      CHECK(!close_full.close() && close_full.failed(), "F5: buffered disk-full close falsely succeeded");
    }
    // Exhaust a regular file only AFTER open's header flush succeeds. A temporary soft size
    // limit exercises delayed stdio failure in both explicit flush and close, without filling
    // a real filesystem. Restore the process limit/handler before any assertions or logging.
    struct rlimit saved_limit;
    const bool have_limit = getrlimit(RLIMIT_FSIZE, &saved_limit) == 0;
    CHECK(have_limit, "F5: could not query file-size limit");
    if (have_limit) {
      for (bool fail_at_close : {false, true}) {
        SessionRecordWriter limited;
        CHECK(limited.open(path, ss), "F5: delayed failure fixture header failed");
        struct rlimit limit = saved_limit;
        limit.rlim_cur = (rlim_t)v6_header;
        const auto old_signal = std::signal(SIGXFSZ, SIG_IGN);
        const bool armed = setrlimit(RLIMIT_FSIZE, &limit) == 0;
        bool caught = false, sticky = false;
        if (armed) {
          limited.write_imu(sample);
          caught = fail_at_close ? !limited.close() : !limited.flush();
          sticky = limited.failed() && !limited.write_frame(frame) && !limited.close() && !limited.error().empty();
        }
        const bool restored = setrlimit(RLIMIT_FSIZE, &saved_limit) == 0;
        std::signal(SIGXFSZ, old_signal);
        CHECK(restored, "F5: could not restore file-size limit");
        if (!restored)
          return 1;
        CHECK(armed && caught && sticky, "F5: delayed storage failure at %s was not sticky", fail_at_close ? "close" : "flush");
        CHECK(rd.open(path), "F5: failed recording must still have its flushed header");
        bool is_imu = false;
        RawImu got_imu;
        FrameObs got_frame;
        while (rd.next(is_imu, got_imu, got_frame)) {}
        CHECK(rd.failed() && !rd.complete(), "F5: failed writer emitted a valid completion footer");
        rd.close();
      }
    }
    std::remove(path.c_str());
    std::printf("[F5] record integrity: v7 footer, v5/v6 compatibility, truncation/corruption and ENOSPC refusal\n");
  }

  // ---------------- F6: live screens/prompts use retained thermal-eligible evidence ----------------
  {
    SessionConfig cfg;
    cfg.verbose = false;
    cfg.scorer.capacity = 8;
    cfg.scorer.holdout_every = 4;
    cfg.scorer.temp_span_gate = 6.0;
    cfg.select_K = 2;
    cfg.min_holdout = 2;
    cfg.collect_min_eig = 5.0;
    cfg.joint.prior_sigma["dw"] = 1.0;
    cfg.joint.prior_sigma["tg"] = 1e-3;
    SessionSeed seed;
    seed.calib.imu.calib_dw = true;
    seed.calib.imu.calib_da = seed.calib.imu.calib_RAtoI = false;
    seed.calib.cams[0].free_ext = seed.calib.cams[0].free_td = false;
    CalibSessionRunner runner(cfg, seed);
    const Eigen::VectorXd prior = CalibCollectionTestAccess::prepare(runner);
    CHECK(prior.size() == 15, "F6: expected dw6 + tg9 test layout");
    Eigen::MatrixXd cool = (1000.0 * prior.array().square().inverse()).matrix().asDiagonal();
    cool(6, 6) = 0.0; // the first Tg/X coefficient is unobserved in the cool bin
    Eigen::MatrixXd hot = Eigen::MatrixXd::Zero(prior.size(), prior.size());
    hot(6, 6) = 1e12 / (prior(6) * prior(6)); // incompatible hot evidence would fill that gap
    for (int i = 0; i < 8; ++i) {
      WindowMeta meta;
      meta.t0 = 3.0 * i;
      meta.t1 = meta.t0 + 2.0;
      meta.temp_mean = i < 4 ? 30.0 : 45.0;
      meta.fingerprint(0) = 1.0;
      CHECK(CalibCollectionTestAccess::retain(runner, meta, i < 4 ? cool : hot).accepted,
            "F6: thermal fixture was not retained");
    }
    const auto status = runner.collect_status();
    CHECK(status.n_retained == 8 && status.n_holdout == 2 && status.n_fusable == 3,
          "F6: collection candidates must match the solve's thermal bin (%d/%d/%d)",
          status.n_retained, status.n_holdout, status.n_fusable);
    CHECK(!status.ready && std::abs(status.min_eig - 1.0) < 1e-10 && status.tg_sigma > status.tg_sigma_target,
          "F6: incompatible-temperature information incorrectly allowed early cutover");
    CHECK(runner.report().prompt.find("IMU X") != std::string::npos,
          "F6: hot-bin evidence hid the retained bin's weak force axis: %s", runner.report().prompt.c_str());
    // Verify this fixture would falsely pass if the two bins were pooled, as before the fix.
    std::vector<Eigen::MatrixXd> pooled = {prior.asDiagonal() * cool * prior.asDiagonal(),
                                           prior.asDiagonal() * hot * prior.asDiagonal()};
    double pooled_eig = 0.0;
    Eigen::VectorXd pooled_sigma;
    WindowScorer::select_logdet(pooled, {{0.0, 2.0}, {12.0, 14.0}}, 2, 0.5, &pooled_eig, &pooled_sigma);
    CHECK(pooled_eig > cfg.collect_min_eig &&
              pooled_sigma.tail(9).cwiseProduct(prior.tail(9)).maxCoeff() < status.tg_sigma_target,
          "F6: fixture must expose the former cross-temperature false-ready verdict");

    cfg.scorer.capacity = 2;
    cfg.scorer.holdout_every = 100;
    CalibSessionRunner eviction(cfg, seed);
    CalibCollectionTestAccess::prepare(eviction);
    Eigen::MatrixXd old = (1000.0 * prior.array().square().inverse()).matrix().asDiagonal();
    old(9, 9) = 0.0; // initially weak on Tg/Y
    WindowMeta meta;
    meta.temp_mean = 30.0;
    meta.t0 = 0.0;
    meta.t1 = 2.0;
    meta.fingerprint(0) = 1.0;
    CalibCollectionTestAccess::retain(eviction, meta, old);
    meta.t0 = 3.0;
    meta.t1 = 5.0;
    CalibCollectionTestAccess::retain(eviction, meta, Eigen::MatrixXd::Zero(prior.size(), prior.size()));
    CHECK(eviction.report().prompt.find("IMU Y") != std::string::npos, "F6: initial weak-axis fixture failed");
    meta.t0 = 6.0;
    meta.t1 = 8.0;
    meta.fingerprint.setZero();
    meta.fingerprint(1) = 1.0; // less redundant, replaces the old information in slot zero
    const auto replaced = CalibCollectionTestAccess::retain(eviction, meta, cool);
    CHECK(replaced.accepted && replaced.evicted_slot == 0, "F6: eviction fixture did not replace its old slot");
    CHECK(eviction.report().prompt.find("IMU X") != std::string::npos,
          "F6: evicted information remained in motion guidance: %s", eviction.report().prompt.c_str());
    std::printf("[F6] live information: thermal false-ready refused; evicted information removed from prompt\n");
  }

  if (failures == 0) {
    std::printf("[PASS] front-end gates green\n");
    return 0;
  }
  std::printf("[FAILED] %d checks\n", failures);
  return 1;
}
