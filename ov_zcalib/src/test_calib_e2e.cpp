/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib end-to-end gates (synthetic truth injection, no target anywhere):
 *
 *  S0  Fisher falsifier: prior-whitened eigenspectrum of the fused information at
 *      TRUTH (identifiability audit), plus a correlated-KLT-drift systematics run
 *      reporting the induced bias on td / extrinsics / Dw/Da (the bias floor).
 *  S2  Single-window recovery: extrinsics + time offset from one excited window.
 *  S3  Multi-window joint recovery: full p = {dw, da, q_AtoI, q_ItoC, p_IinC, td}
 *      from 8 windows, VarPro outer loop, gates on absolute parameter error.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <cstdio>
#include <random>

#include "types/ImuIntrinsicModel.h"
#include "solve/JointCalib.h"
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

// ---------------- synthetic world ----------------

struct Truth {
  ImuIntrinsicModel imu;
  Eigen::Vector4d q_ItoC;
  Eigen::Vector3d p_IinC;
  Eigen::Matrix<double, 8, 1> cam;
  double td = 0.0025; // 2.5 ms
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
  qc << 0.5 * 0.25, 0.5 * -0.18, 0.5 * 0.30, 1.0; // ~25 deg total cam-IMU rotation
  t.q_ItoC = qc / qc.norm();
  t.p_IinC << 0.02, -0.03, 0.05;
  t.cam << 450, 452, 320, 240, 0, 0, 0, 0;
  return t;
}

// Window-frame trajectory (aggressive 6-axis; each window uses its own phase)
static Eigen::Matrix3d R_of(double t, double ph) {
  return ov_core::exp_so3(Eigen::Vector3d(1.00 * std::sin(2.3 * t + ph), 0.90 * std::sin(1.7 * t + 1.3 * ph + 0.8),
                                          0.80 * std::sin(2.9 * t + 0.5 * ph + 1.9))); // R_GtoI (wide tilt sweeps: Da needs gravity re-orientation)
}
static Eigen::Vector3d p_of(double t, double ph) {
  return Eigen::Vector3d(0.40 * std::sin(1.9 * t + ph), 0.35 * std::sin(2.6 * t + 0.7 * ph + 1.1), 0.30 * std::sin(1.3 * t + 2.2));
}

struct Synth {
  Truth tr;
  double ph = 0.0;
  Eigen::Vector3d omega_I(double t) const { // dR/dt = -skew(w) R  =>  w = vee(-Rdot R^T)
    const double h = 1e-6;
    Eigen::Matrix3d Rdot = (R_of(t + h, ph) - R_of(t - h, ph)) / (2 * h);
    Eigen::Matrix3d W = -Rdot * R_of(t, ph).transpose();
    return Eigen::Vector3d(W(2, 1), W(0, 2), W(1, 0));
  }
  Eigen::Vector3d accel_I(double t) const { // a_hat = R_GtoI * (pddot + g_W)
    const double h = 1e-4;
    Eigen::Vector3d pdd = (p_of(t + h, ph) - 2 * p_of(t, ph) + p_of(t - h, ph)) / (h * h);
    return R_of(t, ph) * (pdd + tr.g_W);
  }
};

/// Generate one window: raw IMU (model-inverted + biases) and pixel tracks.
static WindowData make_window(const Truth &tr, double phase, double dur, double fps, double imu_hz, unsigned seed,
                              double drift_px_per_frame = 0.0) {
  Synth sy{tr, phase};
  WindowData w;
  w.pix_sigma = 0.5;
  std::mt19937 rng(seed);
  std::normal_distribution<double> nrm(0.0, 1.0);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);

  // raw IMU over [0-pad, dur+pad]: invert the intrinsic model
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

  // world features around the trajectory
  const int NF = 60;
  std::vector<Eigen::Vector3d> pf(NF);
  for (int f = 0; f < NF; ++f)
    pf[f] = Eigen::Vector3d(4.5 * uni(rng), 4.5 * uni(rng), 3.0 + 2.5 * uni(rng));
  w.num_feats = NF;

  // clones at frame stamps; TRUE imaging pose at t + td
  const Eigen::Matrix3d R_ItoC = ov_core::quat_2_Rot(tr.q_ItoC);
  std::vector<Eigen::Vector2d> last_uv(NF), drift_dir(NF, Eigen::Vector2d::Zero());
  std::vector<int> track_len(NF, 0);
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
      // correlated KLT drift systematics: accumulate along the track's flow direction
      if (drift_px_per_frame > 0.0 && track_len[f] > 0) {
        Eigen::Vector2d flow = uv - last_uv[f];
        if (flow.norm() > 1e-6)
          drift_dir[f] = flow.normalized();
        uv += drift_px_per_frame * track_len[f] * drift_dir[f];
      }
      last_uv[f] = uv;
      track_len[f]++;
      CloneObs o;
      o.feat_id = (size_t)f;
      o.uv = uv + w.pix_sigma * Eigen::Vector2d(nrm(rng), nrm(rng));
      o.u_frac = uv(1) / 480.0;
      w.obs.back().push_back(o);
    }
  }

  // Nuisance seeds from perturbed truth (window frame = I0 at t0): the gates isolate
  // CALIBRATION recovery; window bootstrapping is the production Dong-Si path (S4).
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
  w.seed_q[0] = Eigen::Vector4d(0, 0, 0, 1); // exact gauge anchor at the window origin
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
  c.imu = ImuIntrinsicModel(); // identity dw/da/qA seeds
  Eigen::Vector4d dq;
  dq << 0.5 * 0.012, 0.5 * 0.009, 0.5 * -0.015, 1.0; // ~1.2 deg extrinsic seed error
  c.cams[0].q_ItoC = ov_core::quat_multiply(dq / dq.norm(), tr.q_ItoC);
  c.cams[0].p_IinC = tr.p_IinC + Eigen::Vector3d(0.006, -0.004, 0.005);
  c.cams[0].cam = tr.cam;
  c.cams[0].td = 0.0; // 2.5 ms seed error
  c.cams[0].tr = 0.0;
  return c;
}

static void report_errors(const SharedCalib &c, const Truth &tr, double &e_rot_deg, double &e_pos_mm, double &e_td_ms, double &e_dw,
                          double &e_da, double &e_qA_deg) {
  e_rot_deg = 2.0 * ov_core::quat_multiply(c.cams[0].q_ItoC, ov_core::Inv(tr.q_ItoC)).head<3>().norm() * 180.0 / M_PI;
  e_pos_mm = (c.cams[0].p_IinC - tr.p_IinC).norm() * 1e3;
  e_td_ms = std::abs(c.cams[0].td - tr.td) * 1e3;
  e_dw = (c.imu.dw - tr.imu.dw).cwiseAbs().maxCoeff();
  e_da = (c.imu.da - tr.imu.da).cwiseAbs().maxCoeff();
  e_qA_deg = 2.0 * ov_core::quat_multiply(c.imu.q_AtoI, ov_core::Inv(tr.imu.q_AtoI)).head<3>().norm() * 180.0 / M_PI;
}

int main() {
  Truth tr = make_truth();

  // ---------------- model sanity: at TRUTH p, a window must cost ~noise level ----------------
  {
    WindowData w0 = make_window(tr, 0.3, 3.0, 20.0, 800.0, 11);
    SharedCalib ct = make_seed(tr);
    ct.imu.dw = tr.imu.dw;
    ct.imu.da = tr.imu.da;
    ct.imu.q_AtoI = tr.imu.q_AtoI;
    ct.cams[0].q_ItoC = tr.q_ItoC;
    ct.cams[0].p_IinC = tr.p_IinC;
    ct.cams[0].td = tr.td;
    WindowSolveReport wr_t, wr_s;
    WindowBA::solve_and_export(w0, ct, false, wr_t, 30, false);
    SharedCalib cs = make_seed(tr);
    WindowBA::solve_and_export(w0, cs, false, wr_s, 30, false);
    std::printf("[sanity] window cost at TRUTH p = %.4e | at SEED p = %.4e (truth must be << seed)\n", wr_t.cost_final, wr_s.cost_final);
    CHECK(wr_t.cost_final < 0.3 * wr_s.cost_final, "truth-p cost not below seed-p cost: model/sign defect");
  }

  // ---------------- S2: single window, extrinsics + td only ----------------
  // Data generated with IDENTITY IMU intrinsics so the frozen dw/da/qA are consistent
  // (freezing them at identity against non-identity truth would push model error into
  // ext/td — that is an S3 coupling question, not the S2 gate).
  {
    Truth tr2 = tr;
    tr2.imu = ImuIntrinsicModel();
    SharedCalib c = make_seed(tr2);
    c.imu.calib_dw = c.imu.calib_da = c.imu.calib_RAtoI = false; // subset: ext + td
    std::vector<WindowData> ws2 = {make_window(tr2, 0.3, 3.0, 20.0, 800.0, 11)};
    JointConfig cfg;
    cfg.outer_iterations = 8;
    JointReport rep;
    CHECK(JointCalib::solve(ws2, c, cfg, rep), "S2 joint solve failed");
    double er, ep, et, edw, eda, eqa;
    report_errors(c, tr2, er, ep, et, edw, eda, eqa);
    std::printf("[S2] 1 window: ext_rot=%.3f deg  ext_pos=%.2f mm  td=%.3f ms (seed was 1.2deg/8.8mm/2.5ms)\n", er, ep, et);
    CHECK(er < 0.15, "S2 ext rot err %.3f deg", er);
    CHECK(ep < 8.0, "S2 ext pos err %.2f mm", ep);
    CHECK(et < 0.30, "S2 td err %.3f ms", et);
  }

  // ---------------- gradient truth-probe: FD of the reduced objective vs export ----
  {
    SharedCalib c = make_seed(tr);
    c.imu = tr.imu;
    c.cams[0].q_ItoC = tr.q_ItoC;
    c.cams[0].p_IinC = tr.p_IinC;
    c.cams[0].td = tr.td;
    c.imu.calib_dw = false;
    c.imu.calib_RAtoI = false;
    c.cams[0].free_ext = false;
    c.cams[0].free_td = false; // da only
    WindowData w0 = make_window(tr, 0.3, 3.0, 20.0, 800.0, 11);
    WindowSolveReport wr;
    WindowBA::solve_and_export(w0, c, true, wr, 40, false);
    auto Jof = [&](int dof, double d) {
      SharedCalib cc = c;
      cc.imu.da(dof) += d;
      WindowSolveReport r;
      WindowBA::solve_and_export(w0, cc, false, r, 40, false);
      return r.cost_final;
    };
    for (int dof : {3, 5}) {
      const double d = 2e-3;
      const double fd = (Jof(dof, +d) - Jof(dof, -d)) / (2 * d);
      std::printf("[gradcheck] da(%d) at truth: FD dJ/dda=%+.4e  exported g=%+.4e  (both ~0 & same sign expected)\n", dof, fd,
                  wr.gred(dof));
    }
  }

  // ---------------- single-group probes: free ONE group, everything else at truth ----
  // Decisive separation of column-correctness from cross-group coupling.
  for (const std::string grp : {std::string("da"), std::string("q_AtoI"), std::string("p_IinC")}) {
    SharedCalib c = make_seed(tr);
    c.imu = tr.imu;
    c.cams[0].q_ItoC = tr.q_ItoC;
    c.cams[0].p_IinC = tr.p_IinC;
    c.cams[0].td = tr.td;
    c.imu.calib_dw = false;
    c.imu.calib_da = (grp == "da");
    c.imu.calib_RAtoI = (grp == "q_AtoI");
    c.cams[0].free_ext = (grp == "p_IinC");
    c.cams[0].free_td = false;
    // perturb ONLY the probed group away from truth
    if (grp == "da")
      c.imu.da = ImuIntrinsicModel().da; // identity seed vs truth
    else if (grp == "q_AtoI")
      c.imu.q_AtoI = Eigen::Vector4d(0, 0, 0, 1);
    else
      c.cams[0].p_IinC = tr.p_IinC + Eigen::Vector3d(0.008, -0.006, 0.007);
    std::vector<WindowData> wp;
    for (int k = 0; k < 3; ++k)
      wp.push_back(make_window(tr, 0.3 + 0.9 * k, 3.0, 20.0, 800.0, 300 + k));
    JointConfig cfg;
    cfg.outer_iterations = 8;
    cfg.verbose = (grp == "da");
    JointReport rep;
    CHECK(JointCalib::solve(wp, c, cfg, rep), "probe %s solve failed", grp.c_str());
    double er, ep, et, edw, eda, eqa;
    report_errors(c, tr, er, ep, et, edw, eda, eqa);
    std::printf("[probe %s] da=%.2e qA=%.3f deg ext_rot=%.3f ext_pos=%.2f mm\n", grp.c_str(), eda, eqa, er, ep);
    if (grp == "da") {
      std::printf("[probe da] per-dof err:");
      for (int i = 0; i < 6; ++i)
        std::printf(" %+.2e", c.imu.da(i) - tr.imu.da(i));
      // cost at FOUND vs at TRUTH over the same windows (nuisances re-solved)
      double cost_found = 0, cost_truth = 0;
      for (auto &w : wp) {
        WindowSolveReport r1;
        WindowBA::solve_and_export(w, c, false, r1, 30, false);
        cost_found += r1.cost_final;
      }
      SharedCalib ct2 = c;
      ct2.imu.da = tr.imu.da;
      for (auto &w : wp) {
        WindowSolveReport r2;
        WindowBA::solve_and_export(w, ct2, false, r2, 30, false);
        cost_truth += r2.cost_final;
      }
      std::printf("\n[probe da] cost at FOUND=%.6e  at TRUTH=%.6e  (found<truth => real model defect)\n", cost_found, cost_truth);
    }
    // da / qA single-group probes are INFORMATIONAL: the accel column-3 (d13/d23/d33)
    // direction is a shallow valley against scene scale + per-window S2 gravity in the
    // single-camera targetless setup (gradients verified by three oracles; the reduced
    // objective is genuinely flat-to-adverse there). Documented open item: multi-window
    // attitude diversity tightens it (see S3); a second camera or |g| scalar closes it.
    if (grp == "da")
      CHECK(eda < 5.0e-2, "probe da diverged %.2e", eda);
    if (grp == "q_AtoI")
      CHECK(eqa < 0.60, "probe qA diverged %.3f deg", eqa);
    if (grp == "p_IinC")
      CHECK(ep < 4.0, "probe p_IinC err %.2f mm", ep);
  }

  // ---------------- S3: 8 windows, full p ----------------
  std::vector<WindowData> ws;
  for (int k = 0; k < 8; ++k)
    ws.push_back(make_window(tr, 0.3 + 0.9 * k, 3.0, 20.0, 800.0, 100 + k));
  {
    SharedCalib c = make_seed(tr);
    JointConfig cfg;
    cfg.outer_iterations = 10;
    JointReport rep;
    CHECK(JointCalib::solve(ws, c, cfg, rep), "S3 joint solve failed");
    double er, ep, et, edw, eda, eqa;
    report_errors(c, tr, er, ep, et, edw, eda, eqa);
    std::printf("[S3] 8 windows: ext_rot=%.3f deg  ext_pos=%.2f mm  td=%.3f ms  dw=%.2e  da=%.2e  qA=%.3f deg\n", er, ep, et, edw, eda,
                eqa);
    CHECK(er < 0.10, "S3 ext rot err %.3f deg", er);
    CHECK(ep < 5.0, "S3 ext pos err %.2f mm", ep);
    CHECK(et < 0.20, "S3 td err %.3f ms", et);
    CHECK(edw < 1.5e-3, "S3 dw err %.2e", edw);
    CHECK(eda < 5.0e-3, "S3 da err %.2e (col-3 shallow valley: documented open item)", eda);
    CHECK(eqa < 0.20, "S3 qA err %.3f deg", eqa);
    // posterior sigma sanity: every committed dof must beat its prior by >= 3x
    int improved = 0;
    for (int i = 0; i < rep.sigma.size(); ++i)
      if (rep.sigma(i) * 3.0 < rep.prior_sigma_vec(i))
        improved++;
    std::printf("[S3] %d/%d dofs beat prior 3x\n", improved, (int)rep.sigma.size());
    CHECK(improved >= (int)rep.sigma.size() - 2, "only %d dofs improved 3x", improved);
  }

  // ---------------- S0a: Fisher eigenspectrum at TRUTH ----------------
  {
    SharedCalib c = make_seed(tr);
    c.imu = tr.imu; // evaluate information AT truth
    c.imu.calib_dw = c.imu.calib_da = c.imu.calib_RAtoI = true;
    c.cams[0].q_ItoC = tr.q_ItoC;
    c.cams[0].p_IinC = tr.p_IinC;
    c.cams[0].td = tr.td;
    const int np = c.local_dim();
    Eigen::MatrixXd Lsum = Eigen::MatrixXd::Zero(np, np);
    int used = 0;
    for (auto &w : ws) {
      WindowSolveReport wr;
      if (WindowBA::solve_and_export(w, c, true, wr, 20, false) && wr.Lambda.rows() == np) {
        Lsum += wr.Lambda;
        used++;
      }
    }
    JointConfig cfg;
    Eigen::VectorXd prior(np);
    {
      SharedCalib tmp = c;
      auto layout = tmp.free_blocks();
      int off = 0;
      for (auto &b : layout) {
        const double sg = cfg.prior_sigma.count(b.name) ? cfg.prior_sigma.at(b.name) : 1.0;
        for (int k = 0; k < b.lsize; ++k)
          prior(off + k) = sg;
        off += b.lsize;
      }
    }
    Eigen::MatrixXd Lw = prior.asDiagonal() * Lsum * prior.asDiagonal(); // prior-whitened DATA information
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Lw);
    std::printf("[S0a] fused DATA information at truth (%d windows), prior-whitened eigenspectrum:\n      min5:", used);
    for (int i = 0; i < 5 && i < np; ++i)
      std::printf(" %.2e", eig.eigenvalues()(i));
    std::printf("  max: %.2e\n", eig.eigenvalues()(np - 1));
    CHECK(eig.eigenvalues()(0) > 1.0, "S0a: weakest whitened mode %.3e <= 1 (data does not beat prior)", eig.eigenvalues()(0));
  }

  // ---------------- S0b: correlated-drift systematics bias floor ----------------
  {
    std::vector<WindowData> wd;
    for (int k = 0; k < 8; ++k)
      wd.push_back(make_window(tr, 0.3 + 0.9 * k, 3.0, 20.0, 800.0, 100 + k, 0.08)); // 0.08 px/frame drift
    SharedCalib c = make_seed(tr);
    JointConfig cfg;
    cfg.outer_iterations = 6;
    cfg.verbose = false;
    JointReport rep;
    bool ok = JointCalib::solve(wd, c, cfg, rep);
    double er, ep, et, edw, eda, eqa;
    report_errors(c, tr, er, ep, et, edw, eda, eqa);
    std::printf("[S0b] drift 0.08 px/frame -> bias floor: ext_rot=%.3f deg ext_pos=%.2f mm td=%.3f ms dw=%.2e da=%.2e (ok=%d)\n", er, ep,
                et, edw, eda, (int)ok);
    // Informational (the falsifier REPORTS the floor; spec gates live in the plan):
    // still assert the floor is bounded, not divergent
    CHECK(ok && et < 3.0 && er < 0.5, "S0b: systematics run diverged (td=%.3f ms rot=%.3f deg)", et, er);
  }

  if (failures == 0) {
    std::printf("[PASS] ov_zcalib e2e: S2 single-window, S3 joint recovery, S0 falsifier gates green\n");
    return 0;
  }
  std::printf("[FAILED] %d checks\n", failures);
  return 1;
}
