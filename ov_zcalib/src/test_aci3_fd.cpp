/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * Two-level FD oracle for the ACI3 calibration preintegration.
 *
 *  Level 1 (sweep): mean exactness vs dense sub-step integration, and every
 *    intrinsic/bias column of (DeltaR, alpha, beta) vs central differences of a
 *    full RE-INTEGRATION at perturbed parameters, at 800 Hz and 200 Hz under a
 *    1e-6 relative tolerance. Re-integration FD is the oracle of record: it
 *    cannot share a sign or Jr-flavor convention with the analytic accumulation.
 *  Level 2 (factor): analytic Jacobians of Factor_ImuAci3 vs central differences
 *    of its residual for ALL 14 parameter blocks (quaternions perturbed through
 *    the JPL local Plus). Pins every convention end to end.
 *
 * Standalone: g++ -O2 -std=c++17 -I<eigen> -I<ov_core/src> -I<ov_init/src> ...
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <cstdio>
#include <cmath>

#include "cpi/AciCalibPreint.h"
#include "solve/Factor_ImuAci3.h"
#include "solve/Factor_ReprojTd.h"
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

// Aggressive smooth raw signals (sensor frame)
static Eigen::Vector3d wm_of(double t) {
  return Eigen::Vector3d(1.2 * std::sin(35 * t), -0.8 * std::cos(21 * t) + 0.4, 0.9 * std::sin(13 * t + 0.7));
}
static Eigen::Vector3d am_of(double t) {
  return Eigen::Vector3d(1.5 * std::sin(9 * t) + 0.3, 9.6 + 0.8 * std::cos(17 * t), -1.1 * std::sin(23 * t) - 0.2);
}

static std::vector<RawImu> make_imu(double t0, double t1, double hz) {
  std::vector<RawImu> v;
  for (double t = t0 - 2.0 / hz; t <= t1 + 2.0 / hz; t += 1.0 / hz) {
    RawImu s;
    s.timestamp = t;
    s.wm = wm_of(t);
    s.am = am_of(t);
    v.push_back(s);
  }
  return v;
}

static ImuIntrinsicModel make_model() {
  ImuIntrinsicModel m;
  m.dw << 1.004, 0.002, 0.997, -0.003, 0.0015, 1.002;
  m.da << 0.996, -0.0025, 1.003, 0.002, -0.001, 0.998;
  Eigen::Vector3d th(0.010, -0.015, 0.008); // ~1 deg accel-frame tilt
  Eigen::Vector4d dq;
  dq.head<3>() = 0.5 * th;
  dq(3) = 1.0;
  m.q_AtoI = ov_core::quat_multiply(dq / dq.norm(), Eigen::Vector4d(0, 0, 0, 1));
  return m;
}

// Dense reference: fine-grid midpoint integration of the exact continuous signals
static void dense_ref(const std::vector<RawImu> &imu, double t0, double t1, const ImuIntrinsicModel &model, const Eigen::Vector3d &bg,
                      const Eigen::Vector3d &ba, Eigen::Matrix3d &DR, Eigen::Vector3d &alpha, Eigen::Vector3d &beta) {
  AciPreintResult r;
  ImuNoise nz;
  DR.setIdentity();
  alpha.setZero();
  beta.setZero();
  const int M = 400;
  std::vector<RawImu> sel;
  AciCalibPreint::integrate(imu, t0, t1, model, bg, ba, nz, r); // result unused; exercises boundary-interp selection
  // integrate the signal functions directly on a grid far finer than the raw rate
  double T = t1 - t0;
  int steps = (int)std::round(T * 100000.0); // ~10 us grid cap
  (void)steps;
  double t = t0;
  const double hstep = T / (double)(M * 64);
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  while (t < t1 - 1e-12) {
    double h = std::min(hstep, t1 - t);
    double tm = t + 0.5 * h;
    Eigen::Vector3d w_hat, a_hat;
    model.correct(wm_of(tm), am_of(tm), bg, ba, w_hat, a_hat);
    // exact ODE: dR/dt = -skew(w) R ; beta' = R^T a ; alpha' = beta
    alpha += beta * h + 0.5 * h * h * (R.transpose() * a_hat);
    beta += h * (R.transpose() * a_hat);
    R = ov_core::exp_so3(-w_hat * h) * R;
    t += h;
  }
  DR = R;
}

static void test_sweep(double hz, double tol_mean, double tol_col) {
  const double t0 = 10.0, t1 = 10.2; // 200 ms interval
  const Eigen::Vector3d bg(0.004, -0.010, 0.020), ba(0.03, -0.05, 0.01);
  ImuIntrinsicModel model = make_model();
  ImuNoise nz;
  auto imu = make_imu(t0, t1, hz);

  AciPreintResult r;
  CHECK(AciCalibPreint::integrate(imu, t0, t1, model, bg, ba, nz, r), "integrate failed");

  // ---- mean vs dense (dense uses the exact continuous signals; the sweep uses
  //      per-raw-step midpoints, so agreement is O(dt^2) discretization: gate loosely
  //      at 800 Hz where the discretization error is ~1e-5) ----
  Eigen::Matrix3d DRd;
  Eigen::Vector3d ald, bed;
  dense_ref(imu, t0, t1, model, bg, ba, DRd, ald, bed);
  const Eigen::Matrix3d DRs = ov_core::quat_2_Rot(r.q_KtoK1);
  const double e_R = ov_core::log_so3(DRs * DRd.transpose()).norm();
  const double e_a = (r.alpha - ald).norm() / std::max(1e-9, ald.norm());
  const double e_b = (r.beta - bed).norm() / std::max(1e-9, bed.norm());
  CHECK(e_R < tol_mean && e_a < tol_mean && e_b < tol_mean, "mean @%.0f Hz: eR=%.3e ea=%.3e eb=%.3e", hz, e_R, e_a, e_b);
  std::printf("[ok] mean vs dense @%.0f Hz: eR=%.2e ea=%.2e eb=%.2e\n", hz, e_R, e_a, e_b);

  // ---- columns vs re-integration (central differences over the FULL sweep) ----
  // Perturbation appliers for each of the 21 columns: [dw6 | da6 | thA3 | bg3 | ba3]
  auto integrate_pert = [&](int col, double eps, AciPreintResult &out) {
    ImuIntrinsicModel mm = model;
    Eigen::Vector3d bgp = bg, bap = ba;
    if (col < 6)
      mm.dw(col) += eps;
    else if (col < 12)
      mm.da(col - 6) += eps;
    else if (col < 15) {
      Eigen::Vector4d dq = Eigen::Vector4d::Zero();
      dq(col - 12) = 0.5 * eps;
      dq(3) = 1.0;
      mm.q_AtoI = ov_core::quat_multiply(dq / dq.norm(), model.q_AtoI);
    } else if (col < 18)
      bgp(col - 15) += eps;
    else
      bap(col - 18) += eps;
    AciCalibPreint::integrate(imu, t0, t1, mm, bgp, bap, nz, out);
  };

  double max_rel = 0.0;
  const double eps = 1e-5; // central-difference sweet spot ~cbrt(machine eps); 1e-6 is roundoff-limited
  for (int c = 0; c < 21; ++c) {
    AciPreintResult rp, rm;
    integrate_pert(c, +eps, rp);
    integrate_pert(c, -eps, rm);
    // theta column: left convention DeltaR(p+dp) = Exp(J dp) DeltaR(p)
    const Eigen::Vector3d fd_th =
        (ov_core::log_so3(ov_core::quat_2_Rot(rp.q_KtoK1) * DRs.transpose()) - ov_core::log_so3(ov_core::quat_2_Rot(rm.q_KtoK1) * DRs.transpose())) /
        (2 * eps);
    const Eigen::Vector3d fd_al = (rp.alpha - rm.alpha) / (2 * eps);
    const Eigen::Vector3d fd_be = (rp.beta - rm.beta) / (2 * eps);
    Eigen::Vector3d an_th, an_al, an_be;
    if (c < 15) {
      an_th = r.Jq_pi.col(c);
      an_al = r.Ja_pi.col(c);
      an_be = r.Jb_pi.col(c);
    } else if (c < 18) {
      an_th = r.J_q.col(c - 15);
      an_al = r.J_a.col(c - 15);
      an_be = r.J_b.col(c - 15);
    } else {
      an_th = Eigen::Vector3d::Zero();
      an_al = r.H_a.col(c - 18);
      an_be = r.H_b.col(c - 18);
    }
    auto rel = [](const Eigen::Vector3d &fd, const Eigen::Vector3d &an) {
      return (fd - an).norm() / std::max(1e-4, fd.norm());
    };
    const double e1 = rel(fd_th, an_th), e2 = rel(fd_al, an_al), e3 = rel(fd_be, an_be);
    max_rel = std::max({max_rel, e1, e2, e3});
    CHECK(e1 < tol_col && e2 < tol_col && e3 < tol_col, "col %d @%.0f Hz: th=%.3e al=%.3e be=%.3e", c, hz, e1, e2, e3);
  }
  std::printf("[ok] sweep columns @%.0f Hz: max rel err %.3e over 21 columns\n", hz, max_rel);
}

static void test_factor_fd() {
  const double t0 = 10.0, t1 = 10.2;
  const double hz = 800.0;
  const Eigen::Vector3d bg(0.004, -0.010, 0.020), ba(0.03, -0.05, 0.01);
  ImuIntrinsicModel model = make_model();
  ImuNoise nz;
  auto imu = make_imu(t0, t1, hz);
  AciPreintResult r;
  AciCalibPreint::integrate(imu, t0, t1, model, bg, ba, nz, r);
  Factor_ImuAci3 f(r, model, bg, ba);

  // States: q1/p1/v1 arbitrary; q2/v2/p2 from prediction; small bias deltas; gravity
  Eigen::Vector3d grav(0.05, -0.03, 9.81);
  grav = 9.81 * grav / grav.norm();
  Eigen::Vector4d q1(0.15, -0.22, 0.08, 1.0);
  q1 /= q1.norm();
  Eigen::Matrix3d R1 = ov_core::quat_2_Rot(q1);
  Eigen::Vector3d p1(0.3, -0.4, 1.2), v1(0.5, 0.2, -0.1);
  Eigen::Matrix3d DR = ov_core::quat_2_Rot(r.q_KtoK1);
  Eigen::Vector4d q2 = ov_core::quat_multiply(r.q_KtoK1, q1);
  Eigen::Vector3d v2 = v1 - grav * r.dt + R1.transpose() * r.beta + Eigen::Vector3d(0.011, -0.007, 0.004);
  Eigen::Vector3d p2 = p1 + v1 * r.dt - 0.5 * grav * r.dt * r.dt + R1.transpose() * r.alpha + Eigen::Vector3d(-0.006, 0.009, 0.005);
  (void)DR;
  Eigen::Vector3d bg1 = bg + Eigen::Vector3d(0.002, -0.001, 0.0015), bg2 = bg1 + Eigen::Vector3d(1e-4, -2e-4, 5e-5);
  Eigen::Vector3d ba1 = ba + Eigen::Vector3d(-0.01, 0.006, -0.004), ba2 = ba1 + Eigen::Vector3d(-3e-4, 1e-4, 2e-4);
  Eigen::Matrix<double, 6, 1> dw = model.dw, da = model.da;
  dw(0) += 0.002;
  da(2) -= 0.0015;
  Eigen::Vector4d qA = model.q_AtoI;

  double *params[14] = {q1.data(), bg1.data(), v1.data(), ba1.data(), p1.data(), q2.data(), bg2.data(),
                        v2.data(), ba2.data(), p2.data(), grav.data(), dw.data(), da.data(), qA.data()};
  const int gsize[14] = {4, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 6, 6, 4};
  const bool is_quat[14] = {true, false, false, false, false, true, false, false, false, false, false, false, false, true};

  // Analytic
  Eigen::Matrix<double, 15, 1> res0;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(14);
  double *jac[14];
  for (int b = 0; b < 14; ++b) {
    J[b].resize(15, gsize[b]);
    jac[b] = J[b].data();
  }
  f.Evaluate(params, res0.data(), jac);

  // FD per block (gravity: 3-dof euclidean FD but compare only the tangent-projected part;
  // it is exact in ambient coords for this factor since the residual is linear in gravity)
  const double eps = 1e-7;
  double max_rel = 0.0;
  for (int b = 0; b < 14; ++b) {
    const int loc = is_quat[b] ? 3 : gsize[b];
    for (int k = 0; k < loc; ++k) {
      std::vector<double> backup(params[b], params[b] + gsize[b]);
      Eigen::Matrix<double, 15, 1> rp, rm;
      auto apply = [&](double e) {
        if (is_quat[b]) {
          Eigen::Vector4d dq = Eigen::Vector4d::Zero();
          dq(k) = 0.5 * e;
          dq(3) = 1.0;
          Eigen::Map<Eigen::Vector4d> q(params[b]);
          q = ov_core::quat_multiply(dq / dq.norm(), Eigen::Map<const Eigen::Vector4d>(backup.data()));
        } else {
          params[b][k] = backup[k] + e;
        }
      };
      apply(+eps);
      f.Evaluate(params, rp.data(), nullptr);
      apply(-eps);
      f.Evaluate(params, rm.data(), nullptr);
      std::copy(backup.begin(), backup.end(), params[b]);
      const Eigen::Matrix<double, 15, 1> fd = (rp - rm) / (2 * eps);
      const Eigen::Matrix<double, 15, 1> an = J[b].col(k);
      const double rel = (fd - an).norm() / std::max(1.0, fd.norm());
      max_rel = std::max(max_rel, rel);
      CHECK(rel < 5e-6, "factor block %d col %d: rel err %.3e", b, k, rel);
    }
  }
  std::printf("[ok] factor FD: max rel err %.3e over all 14 blocks\n", max_rel);
}

// FD oracle for the reprojection factor's EQUIDISTANT (fisheye) branch at
// production-unit intrinsics. The synthetic e2e suites are radtan-only, so the
// fisheye distortion Jacobians (k1..k4 columns and their fx/theta chain) would
// otherwise ship untested at values real wide-FOV units exercise. Covers all
// 7 blocks of Factor_ReprojTd including the td transport column and the fixed
// dt_ref shift (the centered rolling-shutter row time folds in there);
// tolerance loosened under transport: the analytic pose columns reuse the
// transported-frame Jacobians, an O(|w| Delta) omission by design.
static void test_reproj_fd(bool fisheye, bool with_transport) {
  // Per-unit calibration of a real AR0144 fisheye (1280x800). The radtan
  // variant reuses the pinhole row with small k1/k2/p1/p2 as the harness
  // CONTROL: it isolates CamEqui-specific defects from factor plumbing.
  Eigen::Matrix<double, 8, 1> cam;
  if (fisheye)
    cam << 457.377, 457.147, 643.895, 409.041, 0.0679821, -5.607e-5, 0.0091765, -0.0051205;
  else
    cam << 457.377, 457.147, 643.895, 409.041, -0.28, 0.07, 1.9e-4, -3.1e-4;
  Eigen::Vector4d q_ItoC(0.353, -0.612, 0.352, 0.615); // representative real IMU-to-camera mounting
  q_ItoC /= q_ItoC.norm();
  Eigen::Vector3d p_IinC(0.0195, 0.0088, -0.0934);
  Eigen::Vector4d q_GtoI(0.11, -0.05, 0.21, 1.0);
  q_GtoI /= q_GtoI.norm();
  Eigen::Vector3d p_IinG(0.4, -0.3, 1.1);

  // Feature CONSTRUCTED in the camera frame off-axis (fisheye: ~55 deg, r ~ 450
  // px, deep in the equidistant polynomial; radtan: ~25 deg, inside its domain)
  const Eigen::Matrix3d R_GtoI = ov_core::quat_2_Rot(q_GtoI);
  const Eigen::Matrix3d R_ItoC = ov_core::quat_2_Rot(q_ItoC);
  const double off_deg = fisheye ? 55.0 : 25.0;
  Eigen::Vector3d dir_c(std::sin(off_deg * M_PI / 180.0), 0.15, std::cos(off_deg * M_PI / 180.0));
  Eigen::Vector3d p_FinC = 2.0 * dir_c.normalized();
  Eigen::Vector3d p_FinG = R_GtoI.transpose() * (R_ItoC.transpose() * (p_FinC - p_IinC)) + p_IinG;

  // Measurement = exact forward projection + sub-pixel offset (nonzero residual)
  const double xn = p_FinC(0) / p_FinC(2), yn = p_FinC(1) / p_FinC(2);
  Eigen::Vector2d uv;
  if (fisheye) {
    const double rr = std::sqrt(xn * xn + yn * yn), th = std::atan(rr);
    const double th_d =
        th * (1 + cam(4) * std::pow(th, 2) + cam(5) * std::pow(th, 4) + cam(6) * std::pow(th, 6) + cam(7) * std::pow(th, 8));
    uv << cam(0) * th_d / rr * xn + cam(2), cam(1) * th_d / rr * yn + cam(3);
  } else {
    const double r2 = xn * xn + yn * yn;
    const double rad = 1.0 + cam(4) * r2 + cam(5) * r2 * r2;
    uv << cam(0) * (xn * rad + 2 * cam(6) * xn * yn + cam(7) * (r2 + 2 * xn * xn)) + cam(2),
        cam(1) * (yn * rad + cam(6) * (r2 + 2 * yn * yn) + 2 * cam(7) * xn * yn) + cam(3);
  }
  uv += Eigen::Vector2d(0.3, -0.2);

  // Clone kinematics; the transported variant moves td away from the
  // linearization point AND carries a fixed dt_ref (the centered row time
  // (v/h - 0.5)*tr_hw of a rolling frame, a known constant), the exact
  // variant sits at Delta == 0 where EVERY column must match FD to double-FD
  // precision.
  const Eigen::Vector3d w_clone(1.2, -0.8, 0.9), v_clone(0.4, -0.3, 0.2);
  double td = with_transport ? 1.3e-3 : 1.0e-3;
  const double dt_ref = with_transport ? 2.0e-3 * (0.37 - 0.5) : 0.0; // row 0.37 of a tr=2 ms frame
  const double Delta = dt_ref + (td - 1.0e-3);
  Factor_ReprojTd f(uv, 1.0, fisheye, w_clone, v_clone, /*td_lin=*/1.0e-3, dt_ref);

  double *params[7] = {q_GtoI.data(), p_IinG.data(), p_FinG.data(), q_ItoC.data(), p_IinC.data(), cam.data(), &td};
  const int gsize[7] = {4, 3, 3, 4, 3, 8, 1};
  const bool is_quat[7] = {true, false, false, true, false, false, false};

  Eigen::Vector2d res0;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(7);
  double *jac[7];
  for (int b = 0; b < 7; ++b) {
    J[b].resize(2, gsize[b]);
    jac[b] = J[b].data();
  }
  CHECK(f.Evaluate(params, res0.data(), jac), "fisheye reproj Evaluate failed");
  CHECK(res0.norm() > 1e-3 && res0.norm() < 5.0, "fisheye residual magnitude off (%.3e px): geometry bug", res0.norm());

  const double eps = 1e-7;
  double max_rel = 0.0;
  for (int b = 0; b < 7; ++b) {
    const int loc = is_quat[b] ? 3 : gsize[b];
    for (int k = 0; k < loc; ++k) {
      std::vector<double> backup(params[b], params[b] + gsize[b]);
      Eigen::Vector2d rp, rm;
      auto apply = [&](double e) {
        if (is_quat[b]) {
          Eigen::Vector4d dq = Eigen::Vector4d::Zero();
          dq(k) = 0.5 * e;
          dq(3) = 1.0;
          Eigen::Map<Eigen::Vector4d> q(params[b]);
          q = ov_core::quat_multiply(dq / dq.norm(), Eigen::Map<const Eigen::Vector4d>(backup.data()));
        } else {
          params[b][k] = backup[k] + e;
        }
      };
      apply(+eps);
      f.Evaluate(params, rp.data(), nullptr);
      apply(-eps);
      f.Evaluate(params, rm.data(), nullptr);
      std::copy(backup.begin(), backup.end(), params[b]);
      const Eigen::Vector2d fd = (rp - rm) / (2 * eps);
      const Eigen::Vector2d an = J[b].col(k);
      const double rel = (fd - an).norm() / std::max(1.0, fd.norm());
      max_rel = std::max(max_rel, rel);
      // Tolerances by CONTRACT: everything is exact at Delta == 0; under
      // transport the clone-attitude column reuses the transported-frame
      // Jacobian (documented O(|w| Delta) omission), and the td chain
      // carries the same first-order slack.
      double tol = 5e-6;
      if (b == 0 || b >= 6)
        tol = std::max(tol, 3.0 * w_clone.norm() * std::abs(Delta));
      if (std::getenv("OV_FD_DEBUG") && rel >= tol)
        std::printf("  dbg b%d c%d: fd=[%.6e %.6e] an=[%.6e %.6e]\n", b, k, fd(0), fd(1), an(0), an(1));
      CHECK(rel < tol, "%s reproj block %d col %d: rel err %.3e (tol %.1e)", fisheye ? "fisheye" : "radtan", b, k, rel, tol);
    }
  }
  std::printf("[ok] %s reproj FD (%s, @ Stinger-class intrinsics): max rel err %.3e over all 7 blocks\n", fisheye ? "fisheye" : "radtan",
              with_transport ? "transported" : "Delta=0 exact", max_rel);
}

// ---- Tg-enabled variants: n_pi = 24 (dw6|da6|thA3|tg9). Pins the tg pi columns AND the D12
// ba->rotation coupling (Phi(0,12)/G(0,3)), which are only nonzero at Tg != 0. Tg at the real
// ICM part scale, all nine elements distinct so no column can borrow another's agreement. ----
static ImuIntrinsicModel make_model_tg() {
  ImuIntrinsicModel m = make_model();
  m.calib_tg = true;
  m.Tg << 3.3e-4, -1.2e-4, 0.9e-4, 2.1e-4, 0.7e-4, -3.8e-4, -1.5e-4, 2.9e-4, 2.2e-4;
  return m;
}

static void test_sweep_tg(double hz, double tol_mean, double tol_col) {
  const double t0 = 10.0, t1 = 10.2;
  const Eigen::Vector3d bg(0.004, -0.010, 0.020), ba(0.03, -0.05, 0.01);
  ImuIntrinsicModel model = make_model_tg();
  ImuNoise nz;
  auto imu = make_imu(t0, t1, hz);

  AciPreintResult r;
  CHECK(AciCalibPreint::integrate(imu, t0, t1, model, bg, ba, nz, r), "tg integrate failed");
  CHECK(r.Jq_pi.cols() == 24, "tg preint carries %d pi columns (want 24)", (int)r.Jq_pi.cols());

  Eigen::Matrix3d DRd;
  Eigen::Vector3d ald, bed;
  dense_ref(imu, t0, t1, model, bg, ba, DRd, ald, bed); // correct() applies Tg -> dense is Tg-exact
  const Eigen::Matrix3d DRs = ov_core::quat_2_Rot(r.q_KtoK1);
  const double e_R = ov_core::log_so3(DRs * DRd.transpose()).norm();
  const double e_a = (r.alpha - ald).norm() / std::max(1e-9, ald.norm());
  const double e_b = (r.beta - bed).norm() / std::max(1e-9, bed.norm());
  CHECK(e_R < tol_mean && e_a < tol_mean && e_b < tol_mean, "tg mean @%.0f Hz: eR=%.3e ea=%.3e eb=%.3e", hz, e_R, e_a, e_b);
  std::printf("[ok] tg mean vs dense @%.0f Hz: eR=%.2e ea=%.2e eb=%.2e\n", hz, e_R, e_a, e_b);

  // 30 columns: [dw6 | da6 | thA3 | tg9 (STORAGE order) | bg3 | ba3]
  auto integrate_pert = [&](int col, double eps, AciPreintResult &out) {
    ImuIntrinsicModel mm = model;
    Eigen::Vector3d bgp = bg, bap = ba;
    if (col < 6)
      mm.dw(col) += eps;
    else if (col < 12)
      mm.da(col - 6) += eps;
    else if (col < 15) {
      Eigen::Vector4d dq = Eigen::Vector4d::Zero();
      dq(col - 12) = 0.5 * eps;
      dq(3) = 1.0;
      mm.q_AtoI = ov_core::quat_multiply(dq / dq.norm(), model.q_AtoI);
    } else if (col < 24)
      mm.Tg.data()[col - 15] += eps; // pi order == Matrix3d storage order, end-to-end
    else if (col < 27)
      bgp(col - 24) += eps;
    else
      bap(col - 27) += eps;
    AciCalibPreint::integrate(imu, t0, t1, mm, bgp, bap, nz, out);
  };

  double max_rel = 0.0;
  const double eps = 1e-5;
  for (int c = 0; c < 30; ++c) {
    AciPreintResult rp, rm;
    integrate_pert(c, +eps, rp);
    integrate_pert(c, -eps, rm);
    const Eigen::Vector3d fd_th =
        (ov_core::log_so3(ov_core::quat_2_Rot(rp.q_KtoK1) * DRs.transpose()) - ov_core::log_so3(ov_core::quat_2_Rot(rm.q_KtoK1) * DRs.transpose())) /
        (2 * eps);
    const Eigen::Vector3d fd_al = (rp.alpha - rm.alpha) / (2 * eps);
    const Eigen::Vector3d fd_be = (rp.beta - rm.beta) / (2 * eps);
    Eigen::Vector3d an_th, an_al, an_be;
    if (c < 24) {
      an_th = r.Jq_pi.col(c);
      an_al = r.Ja_pi.col(c);
      an_be = r.Jb_pi.col(c);
    } else if (c < 27) {
      an_th = r.J_q.col(c - 24);
      an_al = r.J_a.col(c - 24);
      an_be = r.J_b.col(c - 24);
    } else {
      // D12: at Tg != 0, ba reaches the ROTATION through Mw_ba = -Dw*Tg*Ma_ba
      an_th = r.H_q.col(c - 27);
      an_al = r.H_a.col(c - 27);
      an_be = r.H_b.col(c - 27);
    }
    auto rel = [](const Eigen::Vector3d &fd, const Eigen::Vector3d &an) {
      return (fd - an).norm() / std::max(1e-4, fd.norm());
    };
    const double e1 = rel(fd_th, an_th), e2 = rel(fd_al, an_al), e3 = rel(fd_be, an_be);
    max_rel = std::max({max_rel, e1, e2, e3});
    CHECK(e1 < tol_col && e2 < tol_col && e3 < tol_col, "tg col %d @%.0f Hz: th=%.3e al=%.3e be=%.3e", c, hz, e1, e2, e3);
  }
  std::printf("[ok] tg sweep columns @%.0f Hz: max rel err %.3e over 30 columns\n", hz, max_rel);
}

static void test_factor_fd_tg() {
  const double t0 = 10.0, t1 = 10.2;
  const double hz = 800.0;
  const Eigen::Vector3d bg(0.004, -0.010, 0.020), ba(0.03, -0.05, 0.01);
  ImuIntrinsicModel model = make_model_tg();
  ImuNoise nz;
  auto imu = make_imu(t0, t1, hz);
  AciPreintResult r;
  AciCalibPreint::integrate(imu, t0, t1, model, bg, ba, nz, r);
  Factor_ImuAci3 f(r, model, bg, ba);

  Eigen::Vector3d grav(0.05, -0.03, 9.81);
  grav = 9.81 * grav / grav.norm();
  Eigen::Vector4d q1(0.15, -0.22, 0.08, 1.0);
  q1 /= q1.norm();
  Eigen::Matrix3d R1 = ov_core::quat_2_Rot(q1);
  Eigen::Vector3d p1(0.3, -0.4, 1.2), v1(0.5, 0.2, -0.1);
  Eigen::Vector4d q2 = ov_core::quat_multiply(r.q_KtoK1, q1);
  Eigen::Vector3d v2 = v1 - grav * r.dt + R1.transpose() * r.beta + Eigen::Vector3d(0.011, -0.007, 0.004);
  Eigen::Vector3d p2 = p1 + v1 * r.dt - 0.5 * grav * r.dt * r.dt + R1.transpose() * r.alpha + Eigen::Vector3d(-0.006, 0.009, 0.005);
  Eigen::Vector3d bg1 = bg + Eigen::Vector3d(0.002, -0.001, 0.0015), bg2 = bg1 + Eigen::Vector3d(1e-4, -2e-4, 5e-5);
  Eigen::Vector3d ba1 = ba + Eigen::Vector3d(-0.01, 0.006, -0.004), ba2 = ba1 + Eigen::Vector3d(-3e-4, 1e-4, 2e-4);
  Eigen::Matrix<double, 6, 1> dw = model.dw, da = model.da;
  dw(0) += 0.002;
  da(2) -= 0.0015;
  Eigen::Vector4d qA = model.q_AtoI;
  Eigen::Matrix3d tg = model.Tg;
  tg(0, 0) += 4e-5; // off the linearization point so dpi_tg is exercised, not just the column
  tg(2, 1) -= 6e-5;

  double *params[15] = {q1.data(), bg1.data(), v1.data(), ba1.data(), p1.data(), q2.data(), bg2.data(),
                        v2.data(), ba2.data(), p2.data(), grav.data(), dw.data(), da.data(), qA.data(),
                        tg.data()};
  const int gsize[15] = {4, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 6, 6, 4, 9};
  const bool is_quat[15] = {true, false, false, false, false, true, false, false, false, false, false, false, false, true, false};

  Eigen::Matrix<double, 15, 1> res0;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(15);
  double *jac[15];
  for (int b = 0; b < 15; ++b) {
    J[b].resize(15, gsize[b]);
    jac[b] = J[b].data();
  }
  f.Evaluate(params, res0.data(), jac);

  const double eps = 1e-7;
  double max_rel = 0.0;
  for (int b = 0; b < 15; ++b) {
    const int loc = is_quat[b] ? 3 : gsize[b];
    for (int k = 0; k < loc; ++k) {
      std::vector<double> backup(params[b], params[b] + gsize[b]);
      Eigen::Matrix<double, 15, 1> rp, rm;
      auto apply = [&](double e) {
        if (is_quat[b]) {
          Eigen::Vector4d dq = Eigen::Vector4d::Zero();
          dq(k) = 0.5 * e;
          dq(3) = 1.0;
          Eigen::Map<Eigen::Vector4d> q(params[b]);
          q = ov_core::quat_multiply(dq / dq.norm(), Eigen::Map<const Eigen::Vector4d>(backup.data()));
        } else {
          params[b][k] = backup[k] + e;
        }
      };
      apply(+eps);
      f.Evaluate(params, rp.data(), nullptr);
      apply(-eps);
      f.Evaluate(params, rm.data(), nullptr);
      std::copy(backup.begin(), backup.end(), params[b]);
      const Eigen::Matrix<double, 15, 1> fd = (rp - rm) / (2 * eps);
      const Eigen::Matrix<double, 15, 1> an = J[b].col(k);
      const double rel = (fd - an).norm() / std::max(1.0, fd.norm());
      max_rel = std::max(max_rel, rel);
      CHECK(rel < 5e-6, "tg factor block %d col %d: rel err %.3e", b, k, rel);
    }
  }
  std::printf("[ok] tg factor FD: max rel err %.3e over all 15 blocks\n", max_rel);
}

int main() {
  test_sweep(800.0, 2e-5, 1e-6);
  test_sweep(200.0, 3e-4, 1e-6);
  test_sweep_tg(800.0, 2e-5, 1e-6);
  test_factor_fd();
  test_factor_fd_tg();
  test_reproj_fd(false, false);
  test_reproj_fd(false, true);
  test_reproj_fd(true, false);
  test_reproj_fd(true, true);
  if (failures == 0) {
    std::printf("[PASS] AciCalibPreint: mean exact, 21 + 30 (tg) ACI3/bias columns == FD, factor Jacobians == FD (14 + 15-block)\n");
    return 0;
  }
  std::printf("[FAILED] %d checks\n", failures);
  return 1;
}
