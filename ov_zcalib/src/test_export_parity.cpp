/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib export-on-accept (eoa) byte-contract gates:
 *
 *  W1  Deferred export == inline export, kept-A (warm) shape: an eval that ran
 *      cost-only and was accepted is re-entered at its UNCHANGED optimum with
 *      the entry-faithful context (WindowBA state_at) -- Lambda/g/qn must be
 *      BYTE-EQUAL to the export the legacy path computed inline. Run twice:
 *      persistent graph (PreintStore slot, the production path) and the
 *      call-local graph (pc = nullptr).
 *  W2  Same, kept-B (cold/seeds) shape: re-entry with warm = nullptr rebuilds
 *      B's evaluation entry from the window seeds.
 *  W3  free_dim == Lambda.rows() (the duel-validity dimension witness). A
 *      calib-column-free qn-only stats path was tried and MEASURED not
 *      byte-equal (-ffast-math SIMD head-peel vs H's leading dimension), so
 *      cert stages keep inline path-A exports -- J2 pins that composition.
 *  J1-J3  JointCalib::solve ON vs OFF (export_on_accept) byte parity of the
 *      shipped posterior (calib values, Lambda, sigma, merit) AND of the duel/
 *      cold evidence counters, across the three arbitration regimes: legacy
 *      A-chain (cert structurally off: accel chain free), cert-on B-chain
 *      (frozen accel chain, ext/td/cam free), and fused (capped) evals.
 *
 * The synthetic harness is the test_calib_e2e generator (truth-perturbed
 * seeds; the gates here isolate the EXPORT mechanics, not seeding).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "cpi/PreintCache.h"
#include "solve/JointCalib.h"
#include "types/ImuIntrinsicModel.h"
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

// ---- byte comparators (parity means BITS, not tolerances) ----
static bool bits_eq(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
static bool mat_eq(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         (a.size() == 0 || std::memcmp(a.data(), b.data(), sizeof(double) * a.size()) == 0);
}
static bool vec_eq(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
  return a.size() == b.size() && (a.size() == 0 || std::memcmp(a.data(), b.data(), sizeof(double) * a.size()) == 0);
}

// ---------------- synthetic world (the test_calib_e2e generator) ----------------

struct Truth {
  ImuIntrinsicModel imu;
  Eigen::Vector4d q_ItoC;
  Eigen::Vector3d p_IinC;
  Eigen::Matrix<double, 8, 1> cam;
  double td = 0.0025;
  Eigen::Vector3d bg{0.002, -0.003, 0.001}, ba{0.02, -0.01, 0.015};
  Eigen::Vector3d g_W{0, 0, 9.81};
};

static Truth make_truth() {
  Truth t;
  t.imu.dw << 1.004, 0.002, 0.997, -0.003, 0.0015, 1.002;
  t.imu.da << 0.996, -0.0025, 1.003, 0.002, -0.001, 0.998;
  Eigen::Vector4d dq;
  dq << 0.5 * 0.010, 0.5 * -0.015, 0.5 * 0.008, 1.0;
  t.imu.q_AtoI = dq / dq.norm();
  Eigen::Vector4d qc;
  qc << 0.5 * 0.25, 0.5 * -0.18, 0.5 * 0.30, 1.0;
  t.q_ItoC = qc / qc.norm();
  t.p_IinC << 0.02, -0.03, 0.05;
  t.cam << 450, 452, 320, 240, 0, 0, 0, 0;
  return t;
}

static Eigen::Matrix3d R_of(double t, double ph) {
  return ov_core::exp_so3(Eigen::Vector3d(1.00 * std::sin(2.3 * t + ph), 0.90 * std::sin(1.7 * t + 1.3 * ph + 0.8),
                                          0.80 * std::sin(2.9 * t + 0.5 * ph + 1.9)));
}
static Eigen::Vector3d p_of(double t, double ph) {
  return Eigen::Vector3d(0.40 * std::sin(1.9 * t + ph), 0.35 * std::sin(2.6 * t + 0.7 * ph + 1.1), 0.30 * std::sin(1.3 * t + 2.2));
}

struct Synth {
  Truth tr;
  double ph = 0.0;
  Eigen::Vector3d omega_I(double t) const {
    const double h = 1e-6;
    Eigen::Matrix3d Rdot = (R_of(t + h, ph) - R_of(t - h, ph)) / (2 * h);
    Eigen::Matrix3d W = -Rdot * R_of(t, ph).transpose();
    return Eigen::Vector3d(W(2, 1), W(0, 2), W(1, 0));
  }
  Eigen::Vector3d accel_I(double t) const {
    const double h = 1e-4;
    Eigen::Vector3d pdd = (p_of(t + h, ph) - 2 * p_of(t, ph) + p_of(t - h, ph)) / (h * h);
    return R_of(t, ph) * (pdd + tr.g_W);
  }
};

static WindowData make_window(const Truth &tr, double phase, double dur, double fps, double imu_hz, unsigned seed) {
  Synth sy{tr, phase};
  WindowData w;
  w.pix_sigma = 0.5;
  std::mt19937 rng(seed);
  std::normal_distribution<double> nrm(0.0, 1.0);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);

  const Eigen::Matrix3d Dw_i = ImuIntrinsicModel::ut(tr.imu.dw).inverse();
  const Eigen::Matrix3d Da_i = ImuIntrinsicModel::ut(tr.imu.da).inverse();
  const Eigen::Matrix3d R_A_t = ov_core::quat_2_Rot(tr.imu.q_AtoI).transpose();
  for (double t = -0.05; t <= dur + 0.05 + tr.td; t += 1.0 / imu_hz) {
    RawImu s;
    s.timestamp = t;
    s.wm = Dw_i * sy.omega_I(t) + tr.bg;
    s.am = Da_i * (R_A_t * sy.accel_I(t)) + tr.ba;
    w.imu.push_back(s);
  }

  const int NF = 60;
  std::vector<Eigen::Vector3d> pf(NF);
  for (int f = 0; f < NF; ++f)
    pf[f] = Eigen::Vector3d(4.5 * uni(rng), 4.5 * uni(rng), 3.0 + 2.5 * uni(rng));
  w.num_feats = NF;

  const Eigen::Matrix3d R_ItoC = ov_core::quat_2_Rot(tr.q_ItoC);
  for (double t = 0.0; t <= dur + 1e-9; t += 1.0 / fps) {
    w.clone_times.push_back(t);
    w.obs.emplace_back();
    const double ti = t + tr.td;
    const Eigen::Matrix3d R = R_of(ti, phase);
    const Eigen::Vector3d p = p_of(ti, phase);
    for (int f = 0; f < NF; ++f) {
      const Eigen::Vector3d pc = R_ItoC * (R * (pf[f] - p)) + tr.p_IinC;
      if (pc(2) < 0.4)
        continue;
      Eigen::Vector2d uv(tr.cam(0) * pc(0) / pc(2) + tr.cam(2), tr.cam(1) * pc(1) / pc(2) + tr.cam(3));
      if (uv(0) < 10 || uv(0) > 630 || uv(1) < 10 || uv(1) > 470)
        continue;
      CloneObs o;
      o.feat_id = (size_t)f;
      o.uv = uv + w.pix_sigma * Eigen::Vector2d(nrm(rng), nrm(rng));
      o.u_frac = uv(1) / 480.0 - 0.5; // centered convention
      w.obs.back().push_back(o);
    }
  }

  const double t0c = w.clone_times.front();
  const Eigen::Matrix3d R0 = R_of(t0c, phase);
  const Eigen::Vector3d p0 = p_of(t0c, phase);
  w.has_seeds = true;
  for (double t : w.clone_times) {
    const Eigen::Matrix3d Rk = R_of(t, phase) * R0.transpose();
    Eigen::Vector4d qk = ov_core::rot_2_quat(Rk);
    Eigen::Vector4d dq;
    dq << 0.5 * 0.003 * nrm(rng), 0.5 * 0.003 * nrm(rng), 0.5 * 0.003 * nrm(rng), 1.0;
    w.seed_q.push_back(ov_core::quat_multiply(dq / dq.norm(), qk));
    const double h = 1e-5;
    const Eigen::Vector3d vW = (p_of(t + h, phase) - p_of(t - h, phase)) / (2 * h);
    w.seed_v.push_back(R0 * vW + 0.02 * Eigen::Vector3d(nrm(rng), nrm(rng), nrm(rng)));
    w.seed_p.push_back(R0 * (p_of(t, phase) - p0) + 0.005 * Eigen::Vector3d(nrm(rng), nrm(rng), nrm(rng)));
  }
  w.seed_q[0] = Eigen::Vector4d(0, 0, 0, 1);
  w.seed_p[0].setZero();
  w.seed_bg = tr.bg * 1.2;
  w.seed_ba = tr.ba * 1.1;
  w.seed_grav = R0 * tr.g_W;
  for (int f = 0; f < NF; ++f)
    w.seed_feats.push_back(R0 * (pf[f] - p0) + 0.02 * Eigen::Vector3d(nrm(rng), nrm(rng), nrm(rng)));
  return w;
}

static SharedCalib make_seed(const Truth &tr) {
  SharedCalib c;
  c.imu = ImuIntrinsicModel();
  Eigen::Vector4d dq;
  dq << 0.5 * 0.012, 0.5 * 0.009, 0.5 * -0.015, 1.0;
  c.cams[0].q_ItoC = ov_core::quat_multiply(dq / dq.norm(), tr.q_ItoC);
  c.cams[0].p_IinC = tr.p_IinC + Eigen::Vector3d(0.006, -0.004, 0.005);
  c.cams[0].cam = tr.cam;
  c.cams[0].td = 0.0;
  c.cams[0].tr = 0.0;
  return c;
}

// serialize every calibration double (free or frozen) for bitwise comparison
static std::vector<double> calib_bits(const SharedCalib &c) {
  std::vector<double> s;
  auto push = [&s](const double *p, int n) { s.insert(s.end(), p, p + n); };
  push(c.imu.dw.data(), 6);
  push(c.imu.da.data(), 6);
  push(c.imu.q_AtoI.data(), 4);
  for (const CamCalib &k : c.cams) {
    push(k.q_ItoC.data(), 4);
    push(k.p_IinC.data(), 3);
    push(k.cam.data(), 8);
    s.push_back(k.td);
    s.push_back(k.tr);
  }
  return s;
}

int main() {
  Truth tr = make_truth();

  // one rich window (the e2e suite's proven shape -- weaker/shorter windows sit
  // at the undamped-PD margin and die at entry even under legacy); uid arms
  // the production PreintStore path
  WindowData w = make_window(tr, 0.3, 3.0, 20.0, 800.0, 11);
  w.uid = 1;

  // ---------------- W1/W2/W3: window-level byte contracts ----------------
  for (int use_pc = 1; use_pc >= 0; --use_pc) {
    PreintStore store;
    WindowPreint *pc = use_pc ? store.ensure(w.uid) : nullptr;

    SharedCalib c = make_seed(tr); // full layout: dw/da/qA + ext + td + cam
    c.cams[0].cam_mode = 1;
    const int np = c.local_dim();

    // (a) warm-up solve at the seed p -> the accepted-baseline nuisance optimum z0
    WindowWarmState z0;
    WindowSolveReport r0;
    CHECK(WindowBA::solve_and_export(w, c, false, r0, 30, false, &z0, pc), "warm-up solve failed (pc=%d)", use_pc);
    CHECK(r0.free_dim == np, "free_dim %d != np %d on eval-only call", r0.free_dim, np);

    // (b) an outer step moves p (td transport + extrinsic + imu + cam all armed)
    SharedCalib c2 = c;
    c2.cams[0].td += 4e-4;
    c2.cams[0].p_IinC(0) += 2e-3;
    c2.cams[0].cam(0) += 0.3;
    c2.imu.dw(0) += 1e-3;

    // (c) kept-A inline (legacy): warm eval at c2 WITH the inline export
    WindowWarmState wA = z0;
    WindowSolveReport ri;
    CHECK(WindowBA::solve_and_export(w, c2, true, ri, 30, false, &wA, pc) && (int)ri.Lambda.rows() == np,
          "inline warm eval failed (pc=%d)", use_pc);

    // (d) kept-A deferred: re-enter with the SAME entry context (z0), override
    //     the state to the kept optimum (wA), zero inner iterations, export
    WindowWarmState entryA = z0;
    WindowSolveReport rd;
    CHECK(WindowBA::solve_and_export(w, c2, true, rd, 0, false, &entryA, pc, &wA) && (int)rd.Lambda.rows() == np,
          "deferred export re-entry failed (pc=%d)", use_pc);
    CHECK(mat_eq(ri.Lambda, rd.Lambda), "W1 pc=%d: deferred Lambda != inline Lambda (max |d| %.3e)", use_pc,
          (ri.Lambda - rd.Lambda).cwiseAbs().maxCoeff());
    CHECK(vec_eq(ri.gred, rd.gred), "W1 pc=%d: deferred gred != inline gred", use_pc);
    CHECK(bits_eq(ri.qn, rd.qn), "W1 pc=%d: deferred qn %.17e != inline qn %.17e", use_pc, rd.qn, ri.qn);
    std::printf("[W1] pc=%d kept-A deferred export BYTE-EQUAL to inline (np=%d, qn=%.3e)\n", use_pc, np, ri.qn);

    // (e) kept-B inline (legacy): cold eval from the window seeds at c2
    WindowWarmState wB;
    WindowSolveReport rbi;
    CHECK(WindowBA::solve_and_export(w, c2, true, rbi, 30, false, &wB, pc) && (int)rbi.Lambda.rows() == np,
          "inline cold eval failed (pc=%d)", use_pc);

    // (f) kept-B deferred: warm = nullptr rebuilds B's seed entry, state -> wB
    WindowSolveReport rbd;
    CHECK(WindowBA::solve_and_export(w, c2, true, rbd, 0, false, nullptr, pc, &wB) && (int)rbd.Lambda.rows() == np,
          "deferred cold export re-entry failed (pc=%d)", use_pc);
    CHECK(mat_eq(rbi.Lambda, rbd.Lambda), "W2 pc=%d: deferred(B) Lambda != inline(B) Lambda (max |d| %.3e)", use_pc,
          (rbi.Lambda - rbd.Lambda).cwiseAbs().maxCoeff());
    CHECK(vec_eq(rbi.gred, rbd.gred), "W2 pc=%d: deferred(B) gred != inline(B) gred", use_pc);
    CHECK(bits_eq(rbi.qn, rbd.qn), "W2 pc=%d: deferred(B) qn != inline(B) qn", use_pc);
    std::printf("[W2] pc=%d kept-B deferred export BYTE-EQUAL to inline\n", use_pc);

    // (g) free_dim contract: the eval-only dimension witness equals the
    //     exporting call's Lambda dimension (the duel-validity checks read it)
    CHECK(ri.free_dim == (int)ri.Lambda.rows(), "W3 pc=%d: free_dim %d != Lambda.rows %d", use_pc, ri.free_dim, (int)ri.Lambda.rows());

    // (h) W4 qn-only forensic probe (the measurement behind keeping cert-stage
    // exports inline, re-verified live): freeze every calibration block
    // (free_blocks() empty) and re-enter the SAME linearization point -- the
    // export forms H without kept columns (nk=0), i.e. the "qn-only fast
    // path" candidate. Its q_n equals the full export's mathematically but
    // need not in BITS: the smaller leading dimension re-peels SIMD
    // accumulations under -ffast-math. EVIDENCE print, not a contract -- eoa
    // never risks the inequality (the certificate consumes only full-export
    // bits); a compiler bump could legitimately close the gap.
    {
      SharedCalib cf = c2;
      cf.imu.calib_dw = cf.imu.calib_da = cf.imu.calib_RAtoI = false;
      cf.imu.calib_tg = false;
      cf.cams[0].free_ext = cf.cams[0].free_td = false;
      cf.cams[0].cam_mode = 0;
      CHECK(cf.local_dim() == 0, "probe calib not fully frozen (np=%d)", cf.local_dim());
      WindowWarmState entryQ = z0;
      WindowSolveReport rq;
      CHECK(WindowBA::solve_and_export(w, cf, true, rq, 0, false, &entryQ, pc, &wA) && rq.Lambda.rows() == 0,
            "qn-only probe export failed (pc=%d)", use_pc);
      const bool qbits = bits_eq(rq.qn, ri.qn);
      std::printf("[W4] pc=%d qn-only (empty-keep) vs full-export q_n: %s (qn_only %.17e full %.17e rel %.3e)\n", use_pc,
                  qbits ? "byte-EQUAL" : "NOT byte-equal", rq.qn, ri.qn,
                  std::abs(rq.qn - ri.qn) / std::max(std::abs(ri.qn), 1e-300));
    }
  }

  // ---------------- J1-J3: JointCalib ON/OFF byte parity ----------------
  auto joint_windows = [&](int n) {
    std::vector<WindowData> ws;
    for (int i = 0; i < n; ++i) {
      ws.push_back(make_window(tr, 0.3 + 0.9 * i, 3.0, 20.0, 800.0, 20 + (unsigned)i));
      ws.back().uid = (std::uint32_t)(i + 1);
    }
    return ws;
  };
  const std::vector<WindowData> ws6 = joint_windows(6);

  struct JointOut {
    bool ok = false;
    SharedCalib c;
    JointReport rep;
  };
  auto run_joint = [&](const SharedCalib &c_in, const JointConfig &cfg_in, bool eoa) {
    JointOut o;
    o.c = c_in;
    JointConfig cfg = cfg_in;
    cfg.export_on_accept = eoa;
    cfg.verbose = false;
    std::vector<WindowData> ws = ws6; // fresh copy: the solve mutates window seeds
    PreintStore st;
    o.ok = JointCalib::solve(ws, o.c, cfg, o.rep, nullptr, &st);
    return o;
  };
  auto check_pair = [&](const char *tag, const JointOut &off, const JointOut &on) {
    CHECK(off.ok && on.ok, "%s: solve failed (off %d on %d)", tag, (int)off.ok, (int)on.ok);
    if (!(off.ok && on.ok))
      return;
    const auto b_off = calib_bits(off.c), b_on = calib_bits(on.c);
    bool same = b_off.size() == b_on.size();
    for (size_t i = 0; same && i < b_off.size(); ++i)
      same = bits_eq(b_off[i], b_on[i]);
    CHECK(same, "%s: committed calib values differ ON vs OFF", tag);
    CHECK(mat_eq(off.rep.Lambda, on.rep.Lambda), "%s: shipped Lambda differs", tag);
    CHECK(vec_eq(off.rep.sigma, on.rep.sigma), "%s: shipped sigma differs", tag);
    CHECK(bits_eq(off.rep.final_merit, on.rep.final_merit), "%s: final merit differs (%.17e vs %.17e)", tag, off.rep.final_merit,
          on.rep.final_merit);
    CHECK(bits_eq(off.rep.qn_max_final, on.rep.qn_max_final), "%s: qn_max differs (%.17e vs %.17e)", tag, off.rep.qn_max_final,
          on.rep.qn_max_final);
    CHECK(off.rep.evaluation_passes == on.rep.evaluation_passes && off.rep.accepted_passes == on.rep.accepted_passes,
          "%s: pass counts differ (%d/%d vs %d/%d)", tag, on.rep.accepted_passes, on.rep.evaluation_passes, off.rep.accepted_passes,
          off.rep.evaluation_passes);
    // Evidence-counter contract: arbitration inputs and their order unchanged
    CHECK(off.rep.warm_evals == on.rep.warm_evals && off.rep.cold_evals == on.rep.cold_evals &&
              off.rep.cold_first == on.rep.cold_first && off.rep.cold_warmfail == on.rep.cold_warmfail &&
              off.rep.cold_jump == on.rep.cold_jump && off.rep.cold_strand == on.rep.cold_strand &&
              off.rep.cold_cert == on.rep.cold_cert && off.rep.cold_plateau == on.rep.cold_plateau &&
              off.rep.cold_anchor == on.rep.cold_anchor && off.rep.cold_won == on.rep.cold_won &&
              off.rep.cold_won_guard == on.rep.cold_won_guard && off.rep.cert_dual_confirms == on.rep.cert_dual_confirms,
          "%s: duel/cold evidence counters differ ON vs OFF", tag);
    CHECK(off.rep.windows_dead == on.rep.windows_dead, "%s: dead-window counts differ", tag);
    std::printf("[%s] ON==OFF byte parity: acc/pass %d/%d, warm %ld cold %ld (f%ld x%ld j%ld s%ld c%ld p%ld a%ld) won %ld, "
                "merit %.6e, export cpu %.3fs -> %.3fs\n",
                tag, on.rep.accepted_passes, on.rep.evaluation_passes, on.rep.warm_evals, on.rep.cold_evals, on.rep.cold_first,
                on.rep.cold_warmfail, on.rep.cold_jump, on.rep.cold_strand, on.rep.cold_cert, on.rep.cold_plateau, on.rep.cold_anchor,
                on.rep.cold_won, on.rep.final_merit, off.rep.t_export_sum, on.rep.t_export_sum);
  };

  { // J1: legacy A-chain shape (accel chain free -> certificate structurally off)
    SharedCalib c = make_seed(tr);
    JointConfig cfg;
    cfg.outer_iterations = 6;
    check_pair("J1 legacy", run_joint(c, cfg, false), run_joint(c, cfg, true));
  }
  { // J2: cert-on B-chain shape (frozen accel chain; ext/td/cam free; qn-only armed per warm eval)
    SharedCalib c = make_seed(tr);
    c.imu.calib_dw = c.imu.calib_da = c.imu.calib_RAtoI = false;
    c.cams[0].cam_mode = 1;
    JointConfig cfg;
    cfg.outer_iterations = 6;
    cfg.use_cert = true;
    check_pair("J2 cert", run_joint(c, cfg, false), run_joint(c, cfg, true));
  }
  { // J3: fused (capped) evals + cert + finalize
    SharedCalib c = make_seed(tr);
    c.imu.calib_dw = c.imu.calib_da = c.imu.calib_RAtoI = false;
    JointConfig cfg;
    cfg.outer_iterations = 6;
    cfg.use_cert = true;
    cfg.fused_schur = true;
    cfg.fused_warmup_passes = 1;
    cfg.fused_iters = 1;
    check_pair("J3 fused", run_joint(c, cfg, false), run_joint(c, cfg, true));
  }
  { // J4: duel_on_accept composition -- eoa SELF-DISARMS (the deferred-duel
    // fold is incremental: legacy's accepted linearization carries the duel
    // LOSER's export in its rounding, unreproducible without exporting the
    // loser). ON must equal OFF because BOTH run legacy inline exports.
    SharedCalib c = make_seed(tr);
    c.imu.calib_dw = c.imu.calib_da = c.imu.calib_RAtoI = false;
    JointConfig cfg;
    cfg.outer_iterations = 6;
    cfg.use_cert = true;
    cfg.duel_on_accept = true;
    check_pair("J4 duel-defer", run_joint(c, cfg, false), run_joint(c, cfg, true));
  }

  if (failures) {
    std::printf("EXPORT-PARITY GATES: %d FAILURE(S)\n", failures);
    return 1;
  }
  std::printf("EXPORT-PARITY GATES: all passed\n");
  return 0;
}
