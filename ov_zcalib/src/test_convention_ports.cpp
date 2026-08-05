/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib convention-port oracles (the sign/frame/ordering firewall gates):
 *
 *  [P1] RelRotProcrustes: bearing-pair relative rotation vs truth, with
 *       translation-induced flow present and 10% outliers injected.
 *  [P2] TimeOffsetInit: xcorr td seed vs injected truth, sub-step accuracy.
 *  [P3] HandEyeWahba: R_ItoC + bg + fine td joint recovery vs truth under
 *       noise + 15% wild outliers (Hampel), including the adjoint-pair sign
 *       convention (theta_C = R_ItoC * theta_I) and the JPL quat output.
 *  [P4] Degeneracy: single-axis rotation must be REJECTED by the
 *       axis-diversity gate, not returned with false confidence.
 *  [P5] Three-way model selection (Procrustes / homography / essential) on
 *       planar, close-3D, and far-field scenes; deterministic per seed.
 *  [P6] IMU-chain gauge port: kalibr/rpng chains must reproduce the same
 *       corrected signals up to one frame rotation (Tg conjugated, not zeroed).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <cstdio>
#include <random>

#include "init/HandEyeWahba.h"
#include "init/RelRotProcrustes.h"
#include "init/TimeOffsetInit.h"
#include "types/ImuChainConvert.h"
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

// Smooth 3-axis body rotation R_GtoI(t) and world angular rate helper
static Eigen::Matrix3d R_of(double t) {
  return ov_core::exp_so3(Eigen::Vector3d(0.9 * std::sin(2.1 * t), 0.8 * std::sin(1.6 * t + 0.7), 0.7 * std::sin(2.7 * t + 1.9)));
}
static Eigen::Vector3d omega_I(double t) { // body rate: dR/dt = -skew(w) R for R = R_GtoI
  const double h = 1e-6;
  const Eigen::Matrix3d W = -(R_of(t + h) - R_of(t - h)) / (2 * h) * R_of(t).transpose();
  return Eigen::Vector3d(W(2, 1), W(0, 2), W(1, 0));
}

int main() {
  std::mt19937 rng(7);
  std::normal_distribution<double> nrm(0.0, 1.0);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);

  // Truth
  Eigen::Vector4d dq;
  dq << 0.5 * 0.32, 0.5 * -0.21, 0.5 * 0.40, 1.0; // ~30 deg total
  const Eigen::Vector4d q_ItoC_t = dq / dq.norm();
  const Eigen::Matrix3d R_ItoC_t = ov_core::quat_2_Rot(q_ItoC_t);
  const Eigen::Vector3d bg_t(0.004, -0.006, 0.003);
  const double td_t = 0.0037; // 3.7 ms

  // Raw IMU stream @ 800 Hz over [0, 25] s (IMU clock)
  std::vector<RawImu> imu;
  for (double t = 0.0; t <= 25.0; t += 1.0 / 800.0) {
    RawImu s;
    s.timestamp = t;
    s.wm = omega_I(t) + bg_t + 2e-4 * Eigen::Vector3d(nrm(rng), nrm(rng), nrm(rng));
    imu.push_back(s);
  }

  // ---------------- P1: Procrustes relative rotation from bearings ----------------
  {
    // Two camera poses 33 ms apart with rotation AND translation; static cloud
    const double t0 = 4.10 + td_t, t1 = 4.10 + td_t + 1.0 / 30.0;
    const Eigen::Matrix3d R_C0 = R_ItoC_t * R_of(t0), R_C1 = R_ItoC_t * R_of(t1);
    const Eigen::Vector3d p0(0.0, 0.0, 0.0), p1(0.04, -0.02, 0.03); // 5+ cm hop
    const Eigen::Matrix3d R_true = R_C1 * R_C0.transpose();
    std::vector<Eigen::Vector3d> b0, b1;
    for (int f = 0; f < 80; ++f) {
      const Eigen::Vector3d pf(4.0 * uni(rng), 4.0 * uni(rng), 5.0 + 2.0 * uni(rng));
      Eigen::Vector3d v0 = R_C0 * (pf - p0), v1 = R_C1 * (pf - p1);
      if (v0(2) < 0.5 || v1(2) < 0.5)
        continue;
      v0.normalize();
      v1.normalize();
      // px-level direction noise
      v0 += 8e-4 * Eigen::Vector3d(nrm(rng), nrm(rng), nrm(rng));
      v1 += 8e-4 * Eigen::Vector3d(nrm(rng), nrm(rng), nrm(rng));
      v0.normalize();
      v1.normalize();
      b0.push_back(v0);
      b1.push_back(v1);
    }
    // inject 10% outliers
    for (size_t k = 0; k < b0.size() / 10; ++k) {
      b1[k * 10] = Eigen::Vector3d(uni(rng), uni(rng), 1.0).normalized();
    }
    RelRotProcrustes::Options opt;
    RelRotProcrustes::Result rr = RelRotProcrustes::solve(b0, b1, /*rng_seed*/ 4100000000ull, opt);
    CHECK(rr.ok, "P1: procrustes failed");
    const double err_deg = std::acos(std::min(1.0, std::max(-1.0, ((rr.R_C1toC2 * R_true.transpose()).trace() - 1.0) / 2.0))) * 180.0 / M_PI;
    std::printf("[P1] rel-rot err %.4f deg (inliers %d/%d, translation present)\n", err_deg, rr.inliers, (int)b0.size());
    // Rotation-only fit with a 5 cm hop at ~5 m depth: the translation-induced bias
    // is real physics (~0.5 deg here) -- SEED-level bound; the hand-eye consumer
    // averages it down over hundreds of direction-diverse pairs (see P3).
    CHECK(err_deg < 0.70, "P1: rel rot err %.3f deg", err_deg);

    // determinism: same seed -> bit-identical result
    RelRotProcrustes::Result rr2 = RelRotProcrustes::solve(b0, b1, 4100000000ull, opt);
    CHECK((rr2.R_C1toC2 - rr.R_C1toC2).norm() == 0.0, "P1: nondeterministic RANSAC");
  }

  // ---------------- build frame pairs over the session (30 Hz camera) ----------------
  std::vector<HandEyePair> pairs;
  std::vector<CamRateSample> rates;
  {
    std::mt19937 prng(21);
    const double fps = 30.0;
    for (double tc = 0.5; tc + 1.0 / fps <= 24.0; tc += 1.0 / fps) {
      // camera clock tc maps to IMU time tc + td
      const double ti0 = tc + td_t, ti1 = tc + 1.0 / fps + td_t;
      const Eigen::Matrix3d Rrel_I = R_of(ti1) * R_of(ti0).transpose(); // {}^{I2}_{I1}R
      const Eigen::Matrix3d Rrel_C = R_ItoC_t * Rrel_I * R_ItoC_t.transpose();
      HandEyePair p;
      p.t0 = tc;
      p.t1 = tc + 1.0 / fps;
      p.theta_C = ov_core::log_so3(Rrel_C) + 6e-4 * Eigen::Vector3d(nrm(prng), nrm(prng), nrm(prng));
      p.weight = 1.0;
      pairs.push_back(p);
      CamRateSample r;
      r.t_mid = tc + 0.5 / fps;
      r.rate = p.theta_C.norm() * fps;
      rates.push_back(r);
    }
  }

  // ---------------- P2: xcorr time offset ----------------
  {
    TimeOffsetResult tr = TimeOffsetInit::solve(rates, imu, 0.10, 0.002);
    CHECK(tr.ok, "P2: xcorr failed");
    std::printf("[P2] xcorr td=%.4f ms (truth %.4f), peak=%.3f sharp=%.1e bound=%d\n", 1e3 * tr.td, 1e3 * td_t, tr.peak_corr,
                tr.peak_sharpness, (int)tr.at_bound);
    CHECK(std::abs(tr.td - td_t) < 7e-4, "P2: td err %.3f ms", 1e3 * std::abs(tr.td - td_t));
    CHECK(!tr.at_bound, "P2: peak at bound");
    CHECK(tr.peak_corr > 0.9, "P2: weak peak %.3f", tr.peak_corr);

    // ---------------- P3: hand-eye joint recovery (+15% outliers) ----------------
    std::vector<HandEyePair> dirty = pairs;
    std::mt19937 orng(5);
    for (size_t k = 0; k < dirty.size(); k += 7) { // ~14% wild
      dirty[k].theta_C = 0.3 * Eigen::Vector3d(nrm(orng), nrm(orng), nrm(orng));
    }
    HandEyeConfig cfg;
    HandEyeResult he;
    CHECK(HandEyeWahba::solve(imu, dirty, tr.td, Eigen::Vector3d::Zero(), cfg, he), "P3: hand-eye failed");
    const double rot_err_deg = 2.0 * ov_core::quat_multiply(he.q_ItoC, ov_core::Inv(q_ItoC_t)).head<3>().norm() * 180.0 / M_PI;
    std::printf("[P3] R_ItoC err %.4f deg | bg err %.2e rad/s | td=%.4f ms (truth %.4f) | rmse=%.2e rad | trimmed %d | pairs %d\n",
                rot_err_deg, (he.bg - bg_t).norm(), 1e3 * he.td, 1e3 * td_t, he.rmse_rad, he.pairs_trimmed, he.pairs_used);
    CHECK(rot_err_deg < 0.25, "P3: rotation err %.3f deg", rot_err_deg);
    CHECK((he.bg - bg_t).norm() < 1.5e-3, "P3: bg err %.2e", (he.bg - bg_t).norm());
    CHECK(std::abs(he.td - td_t) < 4e-4, "P3: td err %.3f ms", 1e3 * std::abs(he.td - td_t));
    CHECK(!he.td_at_bound, "P3: td at fine bound");
    CHECK(he.pairs_trimmed > 0, "P3: Hampel trimmed nothing despite injected outliers");
  }

  // ---------------- P4: degenerate single-axis motion must be rejected ----------------
  {
    std::vector<RawImu> imu1;
    std::vector<HandEyePair> pairs1;
    for (double t = 0.0; t <= 10.0; t += 1.0 / 800.0) {
      RawImu s;
      s.timestamp = t;
      s.wm = Eigen::Vector3d(1.2 * std::sin(2.0 * t), 0, 0) + bg_t;
      imu1.push_back(s);
    }
    auto R1_of = [&](double t) { return ov_core::exp_so3(Eigen::Vector3d(-1.2 / 2.0 * (std::cos(2.0 * t) - 1.0), 0, 0)); };
    for (double tc = 0.5; tc + 1.0 / 30.0 <= 9.5; tc += 1.0 / 30.0) {
      const Eigen::Matrix3d Rrel_I = R1_of(tc + 1.0 / 30.0) * R1_of(tc).transpose();
      HandEyePair p;
      p.t0 = tc;
      p.t1 = tc + 1.0 / 30.0;
      p.theta_C = ov_core::log_so3(R_ItoC_t * Rrel_I * R_ItoC_t.transpose());
      pairs1.push_back(p);
    }
    HandEyeConfig cfg;
    HandEyeResult he;
    const bool ok = HandEyeWahba::solve(imu1, pairs1, 0.0, Eigen::Vector3d::Zero(), cfg, he);
    std::printf("[P4] single-axis: ok=%d (must be 0; diversity gate)\n", (int)ok);
    CHECK(!ok, "P4: degenerate motion accepted (diversity %.3f)", he.axis_diversity);
  }

  // ---------------- P5: three-way model selection (Procrustes / H / E) ----------------
  {
    std::mt19937 prng(31);
    const double t0 = 6.20 + td_t, t1 = 6.20 + td_t + 1.0 / 30.0;
    const Eigen::Matrix3d R_C0 = R_ItoC_t * R_of(t0), R_C1 = R_ItoC_t * R_of(t1);
    const Eigen::Matrix3d R_true = R_C1 * R_C0.transpose();
    auto gen = [&](auto make_pt, const Eigen::Vector3d &hop, std::vector<Eigen::Vector3d> &b0, std::vector<Eigen::Vector3d> &b1v) {
      for (int f = 0; f < 90; ++f) {
        const Eigen::Vector3d pf = make_pt(f);
        Eigen::Vector3d v0 = R_C0 * pf, v1 = R_C1 * (pf - hop);
        if (v0(2) < 0.2 || v1(2) < 0.2)
          continue;
        v0.normalize();
        v1.normalize();
        v0 += 8e-4 * Eigen::Vector3d(nrm(prng), nrm(prng), nrm(prng));
        v1 += 8e-4 * Eigen::Vector3d(nrm(prng), nrm(prng), nrm(prng));
        b0.push_back(v0.normalized());
        b1v.push_back(v1.normalized());
      }
    };
    auto ang_err = [&](const Eigen::Matrix3d &R) {
      return std::acos(std::min(1.0, std::max(-1.0, ((R * R_true.transpose()).trace() - 1.0) / 2.0))) * 180.0 / M_PI;
    };
    std::uniform_real_distribution<double> u2(-1.0, 1.0);

    // (a) PLANAR wall at ~0.8 m + real translation: the 8-point-E degenerate case
    {
      std::vector<Eigen::Vector3d> b0, b1v;
      const Eigen::Vector3d nrm_pl = R_C0.transpose() * Eigen::Vector3d(0.05, -0.03, 1.0).normalized();
      auto pt = [&](int f) -> Eigen::Vector3d { // points on the plane n.p = 0.8 (camera-0-ish frame)
        (void)f;
        const Eigen::Vector3d q(0.6 * u2(prng), 0.45 * u2(prng), 0.0);
        const Eigen::Vector3d base = R_C0.transpose() * q; // spread roughly parallel to the image
        return Eigen::Vector3d(base + nrm_pl * (0.8 - nrm_pl.dot(base)));
      };
      gen(pt, Eigen::Vector3d(0.03, -0.02, 0.015), b0, b1v);
      RelRotEssential::Result rh = RelRotEssential::solve(b0, b1v, 620000001ull);
      RelRotEssential::Options no_h;
      no_h.enable_homography = false;
      RelRotEssential::Result re = RelRotEssential::solve(b0, b1v, 620000001ull, no_h);
      std::printf("[P5a] planar wall: model=%d err=%.3f deg (E-only would give %.3f deg) inl=%d/%d\n", rh.model, ang_err(rh.R_C1toC2),
                  ang_err(re.R_C1toC2), rh.inliers, (int)b0.size());
      CHECK(rh.ok && rh.model == RelRotEssential::MODEL_HOMOGRAPHY, "P5a: planar scene did not select homography (model %d)", rh.model);
      CHECK(ang_err(rh.R_C1toC2) < 0.35, "P5a: homography R err %.3f deg", ang_err(rh.R_C1toC2));
      RelRotEssential::Result rh2 = RelRotEssential::solve(b0, b1v, 620000001ull);
      CHECK((rh2.R_C1toC2 - rh.R_C1toC2).norm() == 0.0 && rh2.model == rh.model, "P5a: nondeterministic");
    }
    // (b) general close-range 3D scene: essential must win
    {
      std::vector<Eigen::Vector3d> b0, b1v;
      auto pt = [&](int f) { return Eigen::Vector3d(R_C0.transpose() * Eigen::Vector3d(1.5 * u2(prng), 1.1 * u2(prng), 1.5 + 1.2 * u2(prng))); };
      gen(pt, Eigen::Vector3d(0.04, -0.025, 0.02), b0, b1v);
      RelRotEssential::Result r3 = RelRotEssential::solve(b0, b1v, 620000002ull);
      std::printf("[P5b] 3D close scene: model=%d err=%.3f deg inl=%d/%d\n", r3.model, ang_err(r3.R_C1toC2), r3.inliers, (int)b0.size());
      CHECK(r3.ok && r3.model == RelRotEssential::MODEL_ESSENTIAL, "P5b: 3D scene did not select essential (model %d)", r3.model);
      CHECK(ang_err(r3.R_C1toC2) < 0.35, "P5b: essential R err %.3f deg", ang_err(r3.R_C1toC2));
    }
    // (c) far-field / sub-noise parallax: rotation-only parsimony
    {
      std::vector<Eigen::Vector3d> b0, b1v;
      auto pt = [&](int f) { return Eigen::Vector3d(R_C0.transpose() * Eigen::Vector3d(6.0 * u2(prng), 4.5 * u2(prng), 12.0 + 4.0 * u2(prng))); };
      gen(pt, Eigen::Vector3d(0.003, -0.002, 0.001), b0, b1v);
      RelRotEssential::Result rp = RelRotEssential::solve(b0, b1v, 620000003ull);
      std::printf("[P5c] far field: model=%d err=%.3f deg\n", rp.model, ang_err(rp.R_C1toC2));
      CHECK(rp.ok && rp.model == RelRotEssential::MODEL_PROCRUSTES, "P5c: far scene did not select Procrustes (model %d)", rp.model);
      CHECK(ang_err(rp.R_C1toC2) < 0.25, "P5c: Procrustes R err %.3f deg", ang_err(rp.R_C1toC2));
    }
  }

  // ---------------- P6: IMU-chain gauge port (kalibr/rpng -> ov_zcalib imu2) ----------------
  // The chain seed crosses a GAUGE boundary: kalibr's IMU frame is the ACCEL
  // frame (lower-tri Dw/Da + R_GYROtoIMU), ov_zcalib's is the GYRO frame
  // (upper-tri + R_AtoI). The invariant that MUST hold is physical: both gauges
  // reproduce the SAME corrected signals up to ONE frame rotation. A repacking
  // (copying triangles across) would silently zero the misalignment and pass a
  // naive equality test -- so this asserts the physics, not the numbers.
  {
    std::mt19937 crng(99);
    // A kalibr-gauge chain: LOWER-tri Dw/Da (scale+misalignment) and a ~0.8 deg
    // gyro-vs-accel die misalignment (the real ICM-class figure).
    Eigen::Matrix3d Dw_k = Eigen::Matrix3d::Identity(), Da_k = Eigen::Matrix3d::Identity();
    Dw_k << 1.0021, 0, 0, -0.0032, 0.9985, 0, 0.0027, -0.0041, 1.0043;
    Da_k << 0.9974, 0, 0, 0.0019, 1.0032, 0, -0.0026, 0.0035, 0.9951;
    // g-sensitivity at real ICM-class magnitude (~4e-4 (rad/s)/(m/s^2), a measured production
    // chain). Tg multiplies a_hat, the ONE quantity whose frame the gauge change moves, so the
    // port must CONJUGATE it (Tg_r = Tg_k * Q_w) -- neither zero it nor copy it across.
    Eigen::Matrix3d Tg_k;
    Tg_k << 1.69362e-4, -7.07768e-5, 2.75611e-5, 1.20919e-4, 2.41152e-6, 1.30676e-4, -7.29277e-5, -3.83020e-4, 9.93289e-5;
    const Eigen::Matrix3d R_G = ov_core::exp_so3(Eigen::Vector3d(0.0092, -0.0051, 0.0068)); // ~0.71 deg
    const Eigen::Matrix3d R_A = Eigen::Matrix3d::Identity();                                // kalibr: accel frame IS the IMU frame

    ImuIntrinsicModel imu;
    imu_chain_to_calib(Dw_k, Da_k, R_G, R_A, Tg_k, imu);
    const Eigen::Matrix3d Dw_r = ImuIntrinsicModel::ut(imu.dw), Da_r = ImuIntrinsicModel::ut(imu.da);
    const Eigen::Matrix3d R_AtoI = ov_core::quat_2_Rot(imu.q_AtoI);

    // Structure: ov_zcalib's matrices MUST be upper-triangular with positive scale
    CHECK(std::abs(Dw_r(1, 0)) + std::abs(Dw_r(2, 0)) + std::abs(Dw_r(2, 1)) < 1e-12, "P6: Dw not upper-triangular");
    CHECK(std::abs(Da_r(1, 0)) + std::abs(Da_r(2, 0)) + std::abs(Da_r(2, 1)) < 1e-12, "P6: Da not upper-triangular");
    CHECK(Dw_r(0, 0) > 0 && Dw_r(1, 1) > 0 && Dw_r(2, 2) > 0, "P6: Dw diagonal not sign-canonical");

    // PHYSICS: for random raw samples, the two gauges must agree up to ONE fixed
    // rotation R (the gyro->accel frame change) on BOTH channels simultaneously.
    Eigen::Matrix3d R_frame = Eigen::Matrix3d::Zero();
    double worst_w = 0.0, worst_a = 0.0;
    for (int k = 0; k < 200; ++k) {
      const Eigen::Vector3d wm(nrm(crng), nrm(crng), nrm(crng)), am(nrm(crng), nrm(crng), 9.81 + nrm(crng));
      // Tg feeds the CORRECTED accel back into the gyro in BOTH gauges (w_hat = D*(w_m - Tg*a_hat)),
      // so the invariance below is what actually pins the conjugation.
      const Eigen::Vector3d a_K = R_A * Da_k * am, w_K = R_G * Dw_k * (wm - Tg_k * a_K);   // kalibr gauge
      const Eigen::Vector3d a_R = R_AtoI * Da_r * am, w_R = Dw_r * (wm - imu.Tg * a_R);    // ov_zcalib gauge
      if (k == 0) { // recover the frame rotation from the gyro channel of the first sample set
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero(), B = Eigen::Matrix3d::Zero();
        std::mt19937 srng(1);
        for (int j = 0; j < 3; ++j) {
          const Eigen::Vector3d v(nrm(srng), nrm(srng), nrm(srng));
          A.col(j) = Dw_r * v;
          B.col(j) = R_G * Dw_k * v;
        }
        R_frame = B * A.inverse();
      }
      worst_w = std::max(worst_w, (w_K - R_frame * w_R).norm());
      worst_a = std::max(worst_a, (a_K - R_frame * a_R).norm());
    }
    const double frame_deg = std::acos(std::min(1.0, std::max(-1.0, (R_frame.trace() - 1.0) / 2.0))) * 180.0 / M_PI;
    std::printf("[P6] chain gauge port: frame rotation %.3f deg | max |dw_hat| %.2e | max |da_hat| %.2e | R_AtoI %.3f deg\n", frame_deg,
                worst_w, worst_a, 2.0 * imu.q_AtoI.head<3>().norm() * 180.0 / M_PI);
    CHECK(worst_w < 1e-12, "P6: gyro channel not gauge-invariant (%.2e)", worst_w);
    CHECK(worst_a < 1e-12, "P6: accel channel not gauge-invariant (%.2e)", worst_a);
    CHECK(R_frame.transpose() * R_frame - Eigen::Matrix3d::Identity() == Eigen::Matrix3d::Zero() ||
              (R_frame.transpose() * R_frame - Eigen::Matrix3d::Identity()).norm() < 1e-10,
          "P6: recovered frame map is not a rotation");
    // The misalignment is PHYSICAL: it must survive the port, not be zeroed.
    CHECK(2.0 * imu.q_AtoI.head<3>().norm() * 180.0 / M_PI > 0.2, "P6: gyro-vs-accel misalignment was zeroed by the port");
    // Tg likewise. The gyro channel above is only gauge-invariant if Tg was conjugated by the SAME
    // frame rotation the QR produced -- zeroing it breaks worst_w outright.
    CHECK(imu.Tg.norm() > 1e-6, "P6: Tg was zeroed by the port");
    const double tg_err = (imu.Tg - Tg_k * R_frame).norm();
    std::printf("[P6] Tg conjugation: |Tg_r - Tg_k*Q_w| %.2e | %.3f deg/s @1g\n", tg_err,
                imu.Tg.rowwise().norm().maxCoeff() * 9.81 * 180.0 / M_PI);
    CHECK(tg_err < 1e-12, "P6: Tg not conjugated by the frame rotation (%.2e)", tg_err);

    // Same-gauge identity: an rpng chain (R_GYROtoIMU = I, Dw already upper-tri)
    // must round-trip EXACTLY -- the port is a no-op in its own gauge.
    {
      Eigen::Matrix3d Dw_u = Eigen::Matrix3d::Identity(), Da_u = Eigen::Matrix3d::Identity();
      Dw_u << 1.0021, -0.0032, 0.0027, 0, 0.9985, -0.0041, 0, 0, 1.0043;
      Da_u << 0.9974, 0.0019, -0.0026, 0, 1.0032, 0.0035, 0, 0, 0.9951;
      const Eigen::Matrix3d R_Ac = ov_core::exp_so3(Eigen::Vector3d(0.004, -0.002, 0.006));
      ImuIntrinsicModel r;
      imu_chain_to_calib(Dw_u, Da_u, Eigen::Matrix3d::Identity(), R_Ac, Tg_k, r);
      const double edw = (ImuIntrinsicModel::ut(r.dw) - Dw_u).norm();
      const double eda = (ImuIntrinsicModel::ut(r.da) - Da_u).norm();
      const double eqa = (ov_core::quat_2_Rot(r.q_AtoI) - R_Ac).norm();
      const double etg = (r.Tg - Tg_k).norm(); // Q_w = I in its own gauge, so Tg passes through
      std::printf("[P6] rpng round-trip (no-op gauge): |dDw| %.2e |dDa| %.2e |dR_AtoI| %.2e |dTg| %.2e\n", edw, eda, eqa, etg);
      CHECK(edw < 1e-12 && eda < 1e-12 && eqa < 1e-12 && etg < 1e-12, "P6: rpng chain did not round-trip exactly");
    }
  }

  if (failures == 0) {
    std::printf("[PASS] convention ports: Procrustes, xcorr td, hand-eye Wahba, 3-way model selection (P5), IMU-chain gauge (P6)\n");
    return 0;
  }
  std::printf("[FAILED] %d checks\n", failures);
  return 1;
}
