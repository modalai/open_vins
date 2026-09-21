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
#include <limits>
#include <random>

#include "cpi/PreintCache.h"
#include "core/Verification.h"
#include "solve/JointCalib.h"
#include "solve/PosteriorChecks.h"
#include "solve/Factor_ReprojTd.h"
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
    s.push_back(k.reprojection_sigma_px);
  }
  return s;
}

static void test_fused_finalize_reports() {
  const Truth truth = make_truth();
  SharedCalib calib = make_seed(truth);
  calib.imu.calib_dw = calib.imu.calib_da = calib.imu.calib_RAtoI = false;
  calib.imu.calib_tg = false;
  calib.cams[0].free_td = false;
  calib.cams[0].td = 0.0; // zero transport: repeated exports have the same objective
  SharedCalib entry = calib;
  std::vector<WindowData> windows;
  for (int i = 0; i < 2; ++i) {
    windows.push_back(make_window(truth, 0.3 + 0.9 * i, 2.0, 20.0, 800.0, 80 + i));
    windows.back().uid = i + 1;
  }
  JointConfig cfg;
  cfg.outer_iterations = 3;
  cfg.window_max_iters = 100;
  cfg.fused_schur = true;
  cfg.fused_iters = 1;
  cfg.use_carry = true;
  cfg.verbose = false;
  JointReport report;
  JointWarmCarry carry;
  PreintStore store;
  CHECK(JointCalib::solve(windows, calib, cfg, report, &carry, &store) && carry.valid,
        "J9: fused finalization failed");
  if (!carry.valid) return;
  double actual_qn = 0.0, carried_cost = 0.0;
  for (size_t i = 0; i < windows.size(); ++i) {
    WindowData w = windows[i];
    const WindowBiasPrior physical_prior{w.seed_bg, w.seed_ba};
    const SeedSnap &s = carry.seeds[i];
    w.has_seeds = s.has;
    w.seed_q = s.q; w.seed_v = s.v; w.seed_p = s.p; w.seed_feats = s.feats;
    w.seed_grav = s.grav; w.seed_bg = s.bg; w.seed_ba = s.ba;
    WindowWarmState warm = carry.warm[i];
    WindowSolveReport evaluated;
    CHECK(WindowBA::solve_and_export(w, calib, true, evaluated, 0, false, &warm,
                                     store.ensure(w.uid), &carry.warm[i], nullptr, &physical_prior),
          "J9: returned-state export failed");
    actual_qn = std::max(actual_qn, evaluated.qn);
    // One additional iteration independently evaluates the carried state's
    // objective. At the already-tight result its cost change is negligible;
    // a stale capped-step cost differs by orders of magnitude more.
    WindowSolveReport polished;
    warm = carry.warm[i];
    CHECK(WindowBA::solve_and_export(w, calib, false, polished, 1, false, &warm, store.ensure(w.uid),
                                     nullptr, nullptr, &physical_prior),
          "J9: returned-state cost evaluation failed");
    CHECK(std::abs(carry.cost[i] - polished.cost_final) < 1e-6 * std::max(1.0, polished.cost_final),
          "J9: carried cost %.9g does not match final state %.9g", carry.cost[i], polished.cost_final);
    carried_cost += carry.cost[i];
  }
  double prior_cost = 0.0;
  const auto old_blocks = entry.free_blocks(), new_blocks = calib.free_blocks();
  int off = 0;
  for (size_t i = 0; i < new_blocks.size(); ++i) {
    const auto &b = new_blocks[i];
    Eigen::VectorXd d(b.lsize);
    if (b.is_quat)
      d = 2.0 * ov_core::quat_multiply(Eigen::Map<const Eigen::Vector4d>(b.ptr),
              ov_core::Inv(Eigen::Vector4d(Eigen::Map<const Eigen::Vector4d>(old_blocks[i].ptr)))).head<3>();
    else
      for (int k = 0; k < b.lsize; ++k) d(k) = b.ptr[k] - old_blocks[i].ptr[k];
    prior_cost += 0.5 * d.cwiseQuotient(report.prior_sigma_vec.segment(off, b.lsize)).squaredNorm();
    off += b.lsize;
  }
  CHECK(std::abs(report.qn_max_final - actual_qn) < 1e-8 * std::max(1.0, actual_qn),
        "J9: reported final qn %.9g != returned-state qn %.9g", report.qn_max_final, actual_qn);
  CHECK(std::abs(report.final_merit - carried_cost - prior_cost) < 1e-10 * std::max(1.0, report.final_merit),
        "J9: reported merit does not match final carried objective");
  std::printf("[J9] final qn %.9g, returned-state qn %.9g, merit %.9g\n",
              report.qn_max_final, actual_qn, report.final_merit);
}

static void test_posterior_validity() {
  Eigen::MatrixXd L = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd sigma;
  L(0, 0) = -1.0;
  Eigen::LDLT<Eigen::MatrixXd> old(L);
  const Eigen::MatrixXd old_cov = old.solve(Eigen::MatrixXd::Identity(2, 2));
  CHECK(old.info() == Eigen::Success && old_cov(0, 0) < 0.0,
        "J10: indefinite-matrix counterexample no longer exercises old LDLT path");
  CHECK(!posterior_sigmas(L, sigma) && sigma.size() == 0,
        "J10: indefinite information must not become a zero-sigma posterior");
  L(0, 0) = 0.0;
  CHECK(!posterior_sigmas(L, sigma), "J10: singular information accepted");
  for (std::uint64_t bits : {UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
                             UINT64_C(0x7ff8000000000000)}) {
    std::memcpy(&L(0, 0), &bits, sizeof(bits));
    CHECK(!posterior_sigmas(L, sigma), "J10: nonfinite information accepted under fast-math");
  }
  L << 4.0, 1.0, 1.0, 3.0;
  Eigen::LDLT<Eigen::MatrixXd> reference(L);
  const Eigen::MatrixXd covariance = reference.solve(Eigen::MatrixXd::Identity(2, 2));
  const Eigen::VectorXd expected = covariance.diagonal().cwiseMax(0.0).cwiseSqrt();
  CHECK(posterior_sigmas(L, sigma) && vec_eq(sigma, expected),
        "J10: valid positive posterior changed numerical values");
  L.setZero(); L(0, 0) = 1e18; L(1, 1) = 1e6;
  CHECK(posterior_sigmas(L, sigma), "J10: information-frozen prior rejected");
  std::printf("[J10] posterior rejects indefinite/singular/nonfinite information; positive path unchanged\n");
}

static void test_verification_objective() {
  const Truth truth = make_truth();
  WindowData w = make_window(truth, 0.3, 1.0, 20.0, 800.0, 77);
  w.uid = 77;
  SharedCalib c = make_seed(truth);
  WindowEvaluationContext context;
  CHECK(WindowBA::make_evaluation_context(w, c, context), "V1: reference context failed");
  CHECK(vec_eq(context.bias_prior.bg, w.seed_bg) && vec_eq(context.bias_prior.ba, w.seed_ba),
        "V1: reference prior differs from harvested physical bias prior");
  WindowWarmState initial;
  WindowSolveReport init;
  CHECK(WindowBA::solve_and_export(w, c, false, init, 1, false, &initial), "V1: initial state failed");
  auto score = [&](const WindowData &window, SharedCalib candidate, const WindowEvaluationContext *ctx) {
    WindowWarmState warm = initial;
    WindowSolveReport report;
    CHECK(WindowBA::solve_and_export(window, candidate, false, report, 1, false, &warm, nullptr, nullptr, ctx),
          "V1: scoring failed");
    return report.cost_final;
  };
  WindowData changed_init = w;
  changed_init.seed_bg += Eigen::Vector3d(0.05, -0.04, 0.03);
  changed_init.seed_ba += Eigen::Vector3d(0.2, -0.1, 0.15);
  const double fixed = score(w, c, &context);
  CHECK(bits_eq(fixed, score(changed_init, c, &context)),
        "V1: candidate initializer changed the fixed physical prior");
  CHECK(!bits_eq(score(w, c, nullptr), score(changed_init, c, nullptr)),
        "V1: failing control does not exercise the old initializer-dependent objective");
  SharedCalib noisy = c;
  noisy.noise.sigma_w *= 2.0;
  noisy.noise.sigma_a *= 2.0;
  CHECK(bits_eq(fixed, score(w, noisy, &context)), "V1: candidate covariance changed the fixed whitener");
  CHECK(!bits_eq(score(w, c, nullptr), score(w, noisy, nullptr)),
        "V1: failing control does not exercise the old candidate-dependent whitener");
  SharedCalib changed_model = c;
  changed_model.imu.dw(0) *= 1.1;
  CHECK(!bits_eq(fixed, score(w, changed_model, &context)),
        "V1: fixed weights incorrectly froze the candidate measurement model");
  // Hold the evaluated state fixed while changing only the entry values that
  // used to be captured as constant temporal reprojection velocities.
  SharedCalib timed = c;
  timed.cams[0].td = 0.003;
  PreintStore temporal_store;
  auto transport_score = [&](bool perturb, const WindowEvaluationContext *ctx, bool cached = false) {
    WindowWarmState entry = initial;
    if (perturb)
      for (auto &velocity : entry.v)
        velocity += Eigen::Vector3d(0.2, -0.15, 0.1);
    WindowSolveReport report;
    CHECK(WindowBA::solve_and_export(w, timed, false, report, 1, false, &entry,
                                     cached ? temporal_store.ensure(w.uid) : nullptr, &initial, ctx),
          "V1: temporal scoring failed");
    return report.cost_final;
  };
  CHECK(bits_eq(transport_score(false, &context), transport_score(true, &context)),
        "V1: entry velocity changed the fixed temporal objective");
  CHECK(bits_eq(transport_score(false, nullptr), transport_score(true, nullptr)),
        "V1: training reprojection still captures entry-dependent temporal velocity");
  const double temporal_fresh = transport_score(false, nullptr);
  CHECK(bits_eq(temporal_fresh, transport_score(false, nullptr, true)) &&
            bits_eq(temporal_fresh, transport_score(true, nullptr, true)),
        "V1: cached reprojection changed the live-velocity objective");
  // Rejected inputs must leave the persistent graph usable at the same state.
  for (int field = 0; field < 6; ++field) {
    WindowWarmState malformed = initial;
    if (field == 0) malformed.q.pop_back();
    if (field == 1) malformed.bg.pop_back();
    if (field == 2) malformed.v.pop_back();
    if (field == 3) malformed.ba.pop_back();
    if (field == 4) malformed.p.pop_back();
    if (field == 5) malformed.feats.pop_back();
    WindowSolveReport bad;
    CHECK(!WindowBA::solve_and_export(w, timed, false, bad, 1, false, &malformed,
                                      temporal_store.ensure(w.uid)),
          "V1: malformed warm field %d entered the cached graph", field);
    WindowWarmState valid = initial;
    CHECK(!WindowBA::solve_and_export(w, timed, false, bad, 1, false, &valid,
                                      temporal_store.ensure(w.uid), &malformed),
          "V1: malformed export state field %d entered the cached graph", field);
  }
  for (int field = 0; field < 4; ++field) {
    WindowData malformed = w;
    if (field == 0) malformed.seed_q.pop_back();
    if (field == 1) malformed.seed_v.pop_back();
    if (field == 2) malformed.seed_p.pop_back();
    if (field == 3) malformed.seed_feats.pop_back();
    WindowSolveReport bad;
    CHECK(!WindowBA::solve_and_export(malformed, timed, false, bad, 1, false, nullptr,
                                      temporal_store.ensure(w.uid)),
          "V1: malformed seed field %d entered the cached graph", field);
  }
  CHECK(bits_eq(temporal_fresh, transport_score(false, nullptr, true)),
        "V1: malformed state poisoned a persistent graph");
  CHECK(valid_verification_cost(true, false, fixed), "V2: valid cost rejected");
  CHECK(!valid_verification_cost(false, false, fixed) && !valid_verification_cost(true, true, fixed) &&
            !valid_verification_cost(true, false, -1.0), "V2: invalid solve accepted");
  for (std::uint64_t bits : {UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
                             UINT64_C(0x7ff8000000000000)}) {
    double bad;
    std::memcpy(&bad, &bits, sizeof(bits));
    CHECK(!valid_verification_cost(true, false, bad), "V2: nonfinite cost accepted under fast-math");
  }
  CHECK(verification_window_usable({1, 1, 0}, 2) && verification_window_usable({1, 1, 1, 0}, 3),
        "V2: optional attribution changed the authoritative holdout set");
  CHECK(!verification_window_usable({1, 0, 1}, 2) && !verification_window_usable({1, 1, 0}, 3) &&
            !verification_window_usable({1}, 2), "V2: required candidate failure accepted");
  std::printf("[V1/V2] common whitening/prior objective, candidate-dependent means, paired holdout validity\n");
}

static void test_training_bias_prior() {
  const Truth truth = make_truth();
  WindowData w = make_window(truth, 0.3, 1.0, 20.0, 800.0, 78);
  w.uid = 78; // nonzero UID is required to obtain an actual persistent graph
  SharedCalib c = make_seed(truth);
  const WindowBiasPrior prior{w.seed_bg, w.seed_ba};
  WindowData reseeded = w;
  reseeded.seed_bg += Eigen::Vector3d(0.05, -0.04, 0.03);
  reseeded.seed_ba += Eigen::Vector3d(0.2, -0.1, 0.15);
  WindowWarmState initial;
  WindowSolveReport init;
  CHECK(WindowBA::solve_and_export(w, c, false, init, 1, false, &initial), "P1: initial state failed");
  for (bool cached : {false, true}) {
    PreintStore store;
    auto evaluate = [&](const WindowData &window, SharedCalib calib, const WindowBiasPrior *physical,
                        const WindowEvaluationContext *context = nullptr) {
      // Both initializers are evaluated from the identical nuisance state.
      // One deterministic inner step also checks that the full objective,
      // derivatives and resulting state remain the same, not only a prior
      // factor evaluated in isolation.
      WindowWarmState warm = initial;
      WindowSolveReport report;
      CHECK(WindowBA::solve_and_export(window, calib, true, report, 1, false, &warm,
                                       cached ? store.ensure(window.uid) : nullptr, nullptr, context, physical),
            "P1: scoring failed cached=%d", cached);
      return report;
    };
    const WindowSolveReport fixed = evaluate(w, c, &prior);
    const WindowSolveReport moved = evaluate(reseeded, c, &prior);
    CHECK(bits_eq(fixed.cost_final, moved.cost_final) && mat_eq(fixed.Lambda, moved.Lambda) &&
              vec_eq(fixed.gred, moved.gred) && bits_eq(fixed.qn, moved.qn),
          "P1: initialization moved the frozen training objective cached=%d", cached);
    CHECK(bits_eq(fixed.cost_final, evaluate(w, c, nullptr).cost_final),
          "P1: explicit unchanged prior altered the legacy objective cached=%d", cached);
    const double old_changed = evaluate(reseeded, c, nullptr).cost_final;
    CHECK(!bits_eq(fixed.cost_final, old_changed),
          "P1: negative control did not exercise moving physical priors cached=%d", cached);
    SharedCalib reweighted = c;
    reweighted.noise.sigma_w *= 2.0;
    CHECK(!bits_eq(fixed.cost_final, evaluate(w, reweighted, &prior).cost_final),
          "P1: training bias override incorrectly froze measurement weights cached=%d", cached);
    WindowEvaluationContext context;
    CHECK(WindowBA::make_evaluation_context(w, c, context), "P1: verification context failed");
    const WindowBiasPrior conflicting{reseeded.seed_bg, reseeded.seed_ba};
    CHECK(bits_eq(evaluate(w, c, &prior, &context).cost_final,
                  evaluate(w, c, &conflicting, &context).cost_final),
          "P1: training override displaced the authoritative VERIFY prior cached=%d", cached);
    std::printf("[P1] cached=%d fixed cost %.12g, old moving-prior control %.12g\n",
                cached, fixed.cost_final, old_changed);
  }

  PreintStore prior_store;
  auto compare_prior_weights = [&](const SharedCalib &candidate, const char *name) {
    WindowSolveReport reports[2];
    for (int cached = 0; cached < 2; ++cached) {
      SharedCalib calib = candidate;
      WindowWarmState warm = initial;
      CHECK(WindowBA::solve_and_export(w, calib, true, reports[cached], 1, false, &warm,
                                       cached ? prior_store.ensure(w.uid) : nullptr, nullptr, nullptr, &prior),
            "P3: %s evaluation failed cached=%d", name, cached);
    }
    CHECK(bits_eq(reports[0].cost_final, reports[1].cost_final) && mat_eq(reports[0].Lambda, reports[1].Lambda) &&
              vec_eq(reports[0].gred, reports[1].gred) && bits_eq(reports[0].qn, reports[1].qn),
          "P3: cached %s weights differ from a fresh graph (cost %.12g vs %.12g)",
          name, reports[1].cost_final, reports[0].cost_final);
  };
  compare_prior_weights(c, "initial prior");
  SharedCalib prior_reweighted = c;
  prior_reweighted.bg_prior_sigma *= 2.0;
  compare_prior_weights(prior_reweighted, "changed bg sigma");
  prior_reweighted.ba_prior_sigma *= 2.0;
  compare_prior_weights(prior_reweighted, "changed ba sigma");
  compare_prior_weights(c, "restored prior sigmas");

  // A supplied, fixed Tg changes physical preintegration even with its
  // derivative columns disabled (the normal 15-column intrinsic layout).
  // Reuse the persistent graph while moving the live and frozen-noise values
  // independently; compare cost and the entire export to fresh integration.
  CHECK(!c.tg_enabled, "P3: fixed-Tg cache regression requires the 15-column layout");
  SharedCalib fixed_tg = c;
  fixed_tg.imu.Tg(0, 1) = 2e-3;
  fixed_tg.imu.Tg(2, 0) = -1e-3;
  compare_prior_weights(fixed_tg, "changed fixed Tg");
  fixed_tg.noise_frozen = true;
  fixed_tg.noise_lin = fixed_tg.imu;
  compare_prior_weights(fixed_tg, "frozen-noise fixed Tg");
  fixed_tg.noise_lin.Tg(1, 2) = 3e-3;
  compare_prior_weights(fixed_tg, "changed noise-only fixed Tg");
  compare_prior_weights(c, "restored fixed Tg");

  // Cached costs must be invalidated when physical priors change between
  // solve calls, including scales that leave all estimated parameters fixed.
  SharedCalib carried = make_seed(truth);
  carried.imu.calib_dw = carried.imu.calib_da = carried.imu.calib_RAtoI = carried.imu.calib_tg = false;
  std::vector<WindowData> windows{make_window(truth, 0.3, 3.0, 20.0, 800.0, 78),
                                make_window(truth, 1.0, 3.0, 20.0, 800.0, 79)};
  windows[0].uid = 78;
  windows[1].uid = 79;
  JointConfig cfg;
  cfg.outer_iterations = 1;
  cfg.use_carry = true;
  cfg.verbose = false;
  JointWarmCarry carry;
  PreintStore store;
  JointReport report;
  CHECK(JointCalib::solve(windows, carried, cfg, report, &carry, &store), "P2: first solve failed");
  report = JointReport();
  CHECK(JointCalib::solve(windows, carried, cfg, report, &carry, &store) && report.cold_jump == 0,
        "P2: unchanged physical priors invalidated compatible carry");
  auto expect_jump = [&](const char *name) {
    report = JointReport();
    CHECK(JointCalib::solve(windows, carried, cfg, report, &carry, &store) &&
              report.windows_used == (int)windows.size() && report.windows_dead == 0 &&
              report.cold_jump >= (long)windows.size(),
          "P2: changed %s reused stale prior costs (%ld jump duels)", name, report.cold_jump);
  };
  windows[0].seed_bg(0) += 0.01;
  expect_jump("bg mean");
  windows[1].seed_ba(1) += 0.02;
  expect_jump("ba mean");
  carried.bg_prior_sigma *= 2.0;
  expect_jump("bg sigma");
  carried.ba_prior_sigma *= 2.0;
  expect_jump("ba sigma");
  carried.noise.sigma_w *= 2.0;
  expect_jump("gyro noise density");
  carried.noise.sigma_wb *= 2.0;
  expect_jump("gyro bias random walk");
  carried.noise.sigma_a *= 2.0;
  expect_jump("accel noise density");
  carried.noise.sigma_ab *= 2.0;
  expect_jump("accel bias random walk");
  std::printf("[P2] physical bias prior means and scales invalidate carried costs\n");
}

int main(int argc, char **argv) {
  test_training_bias_prior();
  if (argc > 1 && std::string(argv[1]) == "--bias-priors-only")
    return failures ? 1 : 0;
  test_posterior_validity();
  test_verification_objective();
  test_fused_finalize_reports();
  if (argc > 1 && std::string(argv[1]) == "--fused-report-only")
    return failures ? 1 : 0;
  Truth tr = make_truth();

  // one rich window (the e2e suite's proven shape -- weaker/shorter windows sit
  // at the undamped-PD margin and die at entry even under legacy); uid arms
  // the production PreintStore path
  WindowData w = make_window(tr, 0.3, 3.0, 20.0, 800.0, 11);
  w.uid = 1;

  // Pixel sigma whitens BOTH residuals and every Jacobian, including td.
  // Test before robustification: a Cauchy cost and a Schur-complemented
  // camera/IMU posterior do not obey a simple inverse-square scaling law.
  {
    Eigen::Vector4d q(0, 0, 0, 1);
    Eigen::Vector3d p(0.1, -0.2, 0.05), feat(0.7, 0.4, 4.0), ext(0.02, 0.01, -0.03);
    Eigen::Vector3d velocity(0.4, 0.1, -0.2);
    Eigen::Matrix<double, 8, 1> cam = tr.cam;
    double td = 0.003;
    const double *params[] = {q.data(), p.data(), feat.data(), q.data(), ext.data(), cam.data(), &td, velocity.data()};
    auto evaluate = [&](bool fisheye, double sigma) {
      Factor_ReprojTd factor(Eigen::Vector2d(350, 260), sigma, fisheye,
                            Eigen::Vector3d(0.2, -0.3, 0.4), 0.0);
      factor.prepare_transport(td);
      Eigen::VectorXd values(60); // 2 residuals + 2 x (4+3+3+4+3+8+1+3) Jacobian entries
      double *jac[8];
      int offset = 2, block = 0;
      for (int size : {4, 3, 3, 4, 3, 8, 1, 3}) {
        jac[block++] = values.data() + offset;
        offset += 2 * size;
      }
      CHECK(factor.Evaluate(params, values.data(), jac), "N1: factor evaluation failed");
      return values;
    };
    for (bool fisheye : {false, true}) {
      const auto base = evaluate(fisheye, 0.5), doubled = evaluate(fisheye, 1.0);
      CHECK((2.0 * doubled - base).norm() <= 1e-12 * base.norm(), "N1: sigma scaling failed (fisheye=%d)", fisheye);
    }
    const auto cam0 = evaluate(false, 0.5), cam1 = evaluate(true, 0.75);
    const auto cam0_after = evaluate(false, 0.5), cam1_after = evaluate(true, 1.5);
    CHECK(vec_eq(cam0, cam0_after) && (2.0 * cam1_after - cam1).norm() <= 1e-12 * cam1.norm(),
          "N1: changing camera 1's sigma changed camera 0 or failed to scale camera 1");
    std::printf("[N1] per-camera pixel sigma scales residuals and all Jacobian blocks\n");
  }

  { // Persistent graph whitening must follow camera overrides AND window fallback.
    WindowData dual = w;
    dual.td_ref.resize(2, dual.td_ref[0]);
    for (auto &observations : dual.obs)
      for (CloneObs &o : observations)
        o.cam = (int)(o.feat_id % 2); // independent, camera-local tracks
    SharedCalib c = make_seed(tr);
    c.cams.resize(2, c.cams[0]);
    c.imu.calib_dw = c.imu.calib_da = c.imu.calib_RAtoI = false;
    PreintStore store;
    WindowPreint *pc = store.ensure(dual.uid);
    WindowWarmState fixed;
    WindowSolveReport warmup;
    CHECK(WindowBA::solve_and_export(dual, c, false, warmup, 30, false, &fixed, pc), "N2: warmup failed");
    auto export_at_fixed = [&](WindowPreint *cache) {
      WindowWarmState entry = fixed;
      WindowSolveReport result;
      CHECK(WindowBA::solve_and_export(dual, c, true, result, 0, false, &entry, cache, &fixed), "N2: fixed export failed");
      return result;
    };
    auto same_export = [&](const WindowSolveReport &a, const WindowSolveReport &b, const char *tag) {
      CHECK(a.Lambda.rows() == b.Lambda.rows() && a.gred.size() == b.gred.size(), "N2 %s: export shape mismatch", tag);
      if (a.Lambda.rows() != b.Lambda.rows() || a.gred.size() != b.gred.size())
        return;
      CHECK((a.Lambda - b.Lambda).norm() <= 1e-10 * std::max(1.0, b.Lambda.norm()) &&
                (a.gred - b.gred).norm() <= 1e-10 * std::max(1.0, b.gred.norm()) &&
                std::abs(a.qn - b.qn) <= 1e-10 * std::max(1.0, std::abs(b.qn)),
            "N2 %s: cached export differs from fresh whitening", tag);
    };
    const auto inherited = export_at_fixed(pc);
    c.cams[0].reprojection_sigma_px = c.cams[1].reprojection_sigma_px = dual.pix_sigma;
    const auto explicit_same = export_at_fixed(pc);
    CHECK(mat_eq(inherited.Lambda, explicit_same.Lambda) && vec_eq(inherited.gred, explicit_same.gred) &&
              bits_eq(inherited.qn, explicit_same.qn), "N2: explicit legacy sigma changed export bytes");
    c.cams[1].reprojection_sigma_px = 1.0;
    const auto independent = export_at_fixed(pc);
    CHECK(independent.preint_hit, "N2: camera-only reweighting invalidated IMU preintegration");
    same_export(independent, export_at_fixed(nullptr), "camera 1 override");
    CHECK((independent.Lambda - inherited.Lambda).norm() > 1e-6 * inherited.Lambda.norm(),
          "N2: camera 1 override did not affect its observations");
    c.cams[0].reprojection_sigma_px = c.cams[1].reprojection_sigma_px = 0.0;
    same_export(export_at_fixed(pc), inherited, "restore fallback");
    dual.pix_sigma = 0.8;
    const auto fallback_changed = export_at_fixed(pc);
    CHECK(fallback_changed.preint_hit, "N2: fallback reweighting invalidated IMU preintegration");
    same_export(fallback_changed, export_at_fixed(nullptr), "changed fallback");
    for (double invalid : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
      c.cams[1].reprojection_sigma_px = invalid;
      WindowSolveReport rejected;
      CHECK(!WindowBA::solve_and_export(dual, c, true, rejected, 0, false, nullptr, pc), "N2: invalid sigma accepted");
    }
    c.cams[1].reprojection_sigma_px = 0.0;
    same_export(export_at_fixed(pc), fallback_changed, "invalid request preserves graph");
    std::printf("[N2] two-camera cached whitening matches fresh export; fallback and IMU cache preserved\n");
  }

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
  auto check_pair = [&](const char *tag, const JointOut &off, const JointOut &on, bool same_work = true) {
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
    CHECK(off.rep.accepted_passes == on.rep.accepted_passes &&
              (!same_work || off.rep.evaluation_passes == on.rep.evaluation_passes),
          "%s: pass counts differ (%d/%d vs %d/%d)", tag, on.rep.accepted_passes, on.rep.evaluation_passes, off.rep.accepted_passes,
          off.rep.evaluation_passes);
    // Evidence-counter contract: arbitration inputs and their order unchanged
    CHECK(!same_work || (off.rep.warm_evals == on.rep.warm_evals && off.rep.cold_evals == on.rep.cold_evals &&
              off.rep.cold_first == on.rep.cold_first && off.rep.cold_warmfail == on.rep.cold_warmfail &&
              off.rep.cold_jump == on.rep.cold_jump && off.rep.cold_strand == on.rep.cold_strand &&
              off.rep.cold_cert == on.rep.cold_cert && off.rep.cold_plateau == on.rep.cold_plateau &&
              off.rep.cold_anchor == on.rep.cold_anchor && off.rep.cold_won == on.rep.cold_won &&
              off.rep.cold_won_guard == on.rep.cold_won_guard && off.rep.cert_dual_confirms == on.rep.cert_dual_confirms),
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
    const auto parallel = run_joint(c, cfg, true);
    cfg.num_threads = 1;
    const auto serial = run_joint(c, cfg, true);
    check_pair("J3 fused thread parity", serial, parallel);
    CHECK(serial.rep.inner_iters_sum == parallel.rep.inner_iters_sum && serial.rep.time_stops == parallel.rep.time_stops,
          "J3: final-pass counters depend on thread scheduling (%ld vs %ld inner iterations)",
          serial.rep.inner_iters_sum, parallel.rep.inner_iters_sum);
  }

  { // Carry must invalidate costs when frozen Tg changes, including its whitener.
    SharedCalib c = make_seed(tr);
    c.imu.calib_dw = c.imu.calib_da = c.imu.calib_RAtoI = c.imu.calib_tg = false;
    c.tg_enabled = true;
    JointConfig cfg;
    cfg.outer_iterations = 2;
    cfg.use_carry = true;
    cfg.verbose = false;
    auto ws = ws6;
    JointWarmCarry carry;
    JointReport first, moved;
    CHECK(JointCalib::solve(ws, c, cfg, first, &carry), "J5: initial carry solve failed");
    c.imu.Tg(0, 1) += 1e-4;
    CHECK(JointCalib::solve(ws, c, cfg, moved, &carry), "J5: changed-Tg solve failed");
    CHECK(moved.cold_jump >= (long)ws.size(), "J5: changed frozen Tg reused stale window costs (%ld jump duels)", moved.cold_jump);
  }
  { // Camera weighting is part of the accepted objective, even when no
    // estimated parameter, IMU whitener, or free-block layout has changed.
    SharedCalib c = make_seed(tr);
    c.imu.calib_dw = c.imu.calib_da = c.imu.calib_RAtoI = false;
    JointConfig cfg;
    cfg.outer_iterations = 1;
    cfg.use_carry = true;
    cfg.verbose = false;
    auto ws = ws6;
    JointWarmCarry carry;
    PreintStore store;
    JointReport first, unchanged, overridden, restored, fallback;
    CHECK(JointCalib::solve(ws, c, cfg, first, &carry, &store), "N3: initial carry solve failed");
    CHECK(JointCalib::solve(ws, c, cfg, unchanged, &carry, &store) && unchanged.cold_jump == 0,
          "N3: unchanged camera weighting demoted compatible carry");
    c.cams[0].reprojection_sigma_px = 0.75;
    CHECK(JointCalib::solve(ws, c, cfg, overridden, &carry, &store) && overridden.cold_jump >= (long)ws.size(),
          "N3: camera sigma override reused old costs (%ld jump duels)", overridden.cold_jump);
    c.cams[0].reprojection_sigma_px = 0.0;
    CHECK(JointCalib::solve(ws, c, cfg, restored, &carry, &store) && restored.cold_jump >= (long)ws.size(),
          "N3: restoring fallback reused overridden costs (%ld jump duels)", restored.cold_jump);
    for (WindowData &window : ws)
      window.pix_sigma = 0.8;
    CHECK(JointCalib::solve(ws, c, cfg, fallback, &carry, &store) && fallback.cold_jump >= (long)ws.size(),
          "N3: changed window fallback reused old costs (%ld jump duels)", fallback.cold_jump);
    std::printf("[N3] camera override and window fallback reweighting invalidate carried costs\n");
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

  { // Budget cancellation must preserve a WHOLE accepted objective, including
    // its window set and posterior. A fake elapsed clock interrupts the next
    // candidate after work has started; no machine-speed threshold is used.
    SharedCalib c = make_seed(tr);
    JointConfig cfg;
    cfg.outer_iterations = 1;
    cfg.num_threads = 1;
    cfg.use_cert = false;
    cfg.early_stop = false;
    const auto first = run_joint(c, cfg, true);
    for (bool eoa : {false, true}) {
      JointOut cut;
      cut.c = c;
      cfg.outer_iterations = 6;
      cfg.export_on_accept = eoa;
      cfg.verbose = false;
      cfg.max_wall_s = 100.0;
      int candidate_checks = 0;
      cfg.budget_clock = [&]() {
        return cut.rep.evaluation_passes >= 2 && ++candidate_checks >= 4 ? 100.0 : 0.0;
      };
      PreintStore st;
      cut.ok = JointCalib::solve(ws6, cut.c, cfg, cut.rep, nullptr, &st);
      CHECK(first.ok && cut.ok && cut.rep.hit_wall_budget, "J6 eoa=%d: no complete budget fallback", eoa);
      CHECK(calib_bits(first.c) == calib_bits(cut.c), "J6 eoa=%d: interrupted candidate leaked parameters", eoa);
      CHECK(mat_eq(first.rep.Lambda, cut.rep.Lambda) && vec_eq(first.rep.sigma, cut.rep.sigma),
            "J6 eoa=%d: interrupted candidate leaked a partial posterior", eoa);
      CHECK(bits_eq(first.rep.final_merit, cut.rep.final_merit) && first.rep.windows_used == cut.rep.windows_used &&
                first.rep.windows_dead == cut.rep.windows_dead && cut.rep.accepted_passes == 1,
            "J6 eoa=%d: interrupted candidate changed the accepted window set or merit", eoa);
    }
    // Predictive admission respects the previous stage's slowest-pass hint
    // before starting any work; ample/unlimited-budget parity is J1-J4 above.
    SharedCalib skipped = c;
    JointReport skipped_rep;
    cfg.budget_clock = [] { return 0.0; };
    cfg.budget_pass_hint_s = 100.0;
    CHECK(!JointCalib::solve(ws6, skipped, cfg, skipped_rep) && skipped_rep.hit_wall_budget &&
              skipped_rep.evaluation_passes == 0 && calib_bits(skipped) == calib_bits(c),
          "J6: stage started despite insufficient budget for its measured pass cost");

    // Fused capped evaluations cannot ship unless EVERY accepted window has
    // a tight final export. Interrupt after one final window and require
    // failure, entry values restored, and no carry/posterior publication.
    SharedCalib fused = c;
    JointReport fused_rep;
    JointWarmCarry carry;
    cfg.outer_iterations = 1;
    cfg.fused_schur = true;
    cfg.use_carry = true;
    cfg.budget_pass_hint_s = 0.0;
    int final_checks = 0;
    cfg.budget_clock = [&]() { return fused_rep.max_pass_s > 0.0 && ++final_checks >= 2 ? 100.0 : 0.0; };
    CHECK(!JointCalib::solve(ws6, fused, cfg, fused_rep, &carry) && fused_rep.hit_wall_budget &&
              !fused_rep.ok && fused_rep.sigma.size() == 0 && fused_rep.Lambda.size() == 0 && !carry.valid &&
              calib_bits(fused) == calib_bits(c),
          "J7: incomplete fused finalization exposed parameters or a partial posterior");
    std::printf("[J6/J7] deadline cancellation preserves complete accepted posterior; incomplete fused finalize rejected\n");
  }

  for (bool fused : {false, true}) {
    // An accepted confirmation must install its point AND its information;
    // a rejected confirmation must preserve the previous accepted bytes.
    // Disable only confirmation-forced q_n duels so a fixed accepted-pass
    // run follows the same accepted trajectory. A rejected final confirmation
    // pays one extra evaluation, so its work counters cannot be identical.
    SharedCalib c = make_seed(tr);
    c.imu.calib_dw = c.imu.calib_da = c.imu.calib_RAtoI = false;
    JointConfig cfg;
    cfg.outer_iterations = 20;
    cfg.num_threads = 1;
    cfg.use_cert = cfg.early_stop = true;
    cfg.fused_schur = fused;
    cfg.cert_qn_rel = 1e100;
    cfg.stop_k = 1;
    cfg.stop_step_winf = cfg.stop_merit_rel = cfg.stop_lambda_max = 1e6;
    cfg.verbose = false;
    const auto stopped = run_joint(c, cfg, true);
    CHECK(stopped.ok && stopped.rep.stopped_early, "J8 fused=%d: no early-stop confirmation to test", fused);
    cfg.early_stop = false;
    cfg.outer_iterations = stopped.rep.accepted_passes;
    const auto fixed = run_joint(c, cfg, true);
    const int extra_confirmation = stopped.rep.evaluation_passes - fixed.rep.evaluation_passes;
    CHECK(extra_confirmation == 0 || extra_confirmation == 1,
          "J8 fused=%d: unexpected confirmation work (%d extra evaluations)", fused, extra_confirmation);
    check_pair(fused ? "J8 fused early-stop" : "J8 early-stop", fixed, stopped, extra_confirmation == 0);
  }

  if (failures) {
    std::printf("EXPORT-PARITY GATES: %d FAILURE(S)\n", failures);
    return 1;
  }
  std::printf("EXPORT-PARITY GATES: all passed\n");
  return 0;
}
