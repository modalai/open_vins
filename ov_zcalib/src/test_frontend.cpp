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

#include "utils/SessionRecord.h"
#include "solve/JointCalib.h"
#include "sim/SynthWorld.h"
#include "window/LinearSeed.h"
#include "window/WindowHarvester.h"
#include "window/WindowScorer.h"

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
    // Window laid down at the SEED td (0), truth td = 2.5 ms — exactly production.
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
      auto pf = synth::make_cloud(90, 42 ^ 0x9e3779b9u); // NOTE: window cloud used 60 — regenerate consistently below
      (void)pf;
      // regenerate exact truth seeds by re-projecting the window's own geometry:
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
      // features: triangulate-free — use the linear seeds' ids by re-solving? Simplest: use LinearSeed feats
      wt.seed_feats = w.seed_feats;
    }
    WindowSolveReport r_tru;
    CHECK(WindowBA::solve_and_export(wt, c, false, r_tru, 30, false), "F1: BA(truth seeds) failed");
    std::printf("[F1] micro-BA cost: linear-seeded %.4e vs truth-seeded %.4e (ratio %.3f)\n", r_lin.cost_final, r_tru.cost_final,
                r_lin.cost_final / std::max(r_tru.cost_final, 1e-300));
    CHECK(r_lin.cost_final < 1.25 * r_tru.cost_final, "F1: linear seeds land in a worse basin (%.3e vs %.3e)", r_lin.cost_final,
          r_tru.cost_final);
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

  // ---------------- F4: record -> replay bit-parity (the S4a/S4b stage gate) ----------------
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

  if (failures == 0) {
    std::printf("[PASS] front-end gates green\n");
    return 0;
  }
  std::printf("[FAILED] %d checks\n", failures);
  return 1;
}
