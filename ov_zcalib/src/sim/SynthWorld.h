/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib test library: synthetic targetless world shared by the front-end /
 * session gates (test_calib_e2e keeps its own frozen copy — do not merge).
 * Generates raw IMU (intrinsic-model-inverted + biases), pixel tracks with
 * bearings, harvest-ready WindowData WITHOUT truth seeds (the production
 * LinearSeed path is the unit under test), and live-style RawImu/FrameObs
 * streams with quiet/excited phases, frame drops and exposure stamps for the
 * harvester and record/replay gates.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_TESTLIB_SYNTH_WORLD_H
#define OV_ZCALIB_TESTLIB_SYNTH_WORLD_H

#include <random>

#include "../utils/StreamTypes.h"
#include "../solve/WindowBA.h"
#include "utils/quat_ops.h"

namespace ov_zcalib {
namespace synth {

struct Truth {
  ImuIntrinsicModel imu;
  Eigen::Vector4d q_ItoC = Eigen::Vector4d(0, 0, 0, 1);
  Eigen::Vector3d p_IinC = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 8, 1> cam = (Eigen::Matrix<double, 8, 1>() << 450, 452, 320, 240, 0, 0, 0, 0).finished();
  double td = 0.0025;
  Eigen::Vector3d bg{0.002, -0.003, 0.001}, ba{0.02, -0.01, 0.015};
  Eigen::Vector3d g_W{0, 0, 9.81};
  int img_w = 640, img_h = 480;
};

inline Truth make_truth() {
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
  return t;
}

/// Excitation envelope: 1 inside excited segments, ramps at the edges.
/// Trajectory shape: aggressive 6-axis inside excitation, near-still outside.
struct Trajectory {
  double phase = 0.0;
  double excite_t0 = -1e9, excite_t1 = 1e9; ///< excited segment (single segment per Trajectory)
  double ramp = 0.5;

  double env(double t) const {
    // residual jitter so "quiet" is not mathematically zero motion, but small
    // enough that a real SETTLE still-detector (~0.03 rad/s) accepts it
    if (t < excite_t0 || t > excite_t1)
      return 0.005;
    const double up = std::min(1.0, (t - excite_t0) / ramp);
    const double dn = std::min(1.0, (excite_t1 - t) / ramp);
    // C1 smoothstep: a piecewise-LINEAR ramp has impulsive env-dotdot at the
    // knees, i.e. true-accel spikes the IMU sampling cannot represent — the
    // preintegration then disagrees with the projected geometry by design.
    const double u = std::min(up, dn);
    const double s = u * u * (3.0 - 2.0 * u);
    return s * 0.995 + 0.005;
  }
  Eigen::Matrix3d R_of(double t) const { // R_GtoI
    const double e = env(t);
    return ov_core::exp_so3(Eigen::Vector3d(e * 1.00 * std::sin(2.3 * t + phase), e * 0.90 * std::sin(1.7 * t + 1.3 * phase + 0.8),
                                            e * 0.80 * std::sin(2.9 * t + 0.5 * phase + 1.9)));
  }
  Eigen::Vector3d p_of(double t) const {
    const double e = env(t);
    return Eigen::Vector3d(e * 0.40 * std::sin(1.9 * t + phase), e * 0.35 * std::sin(2.6 * t + 0.7 * phase + 1.1),
                           e * 0.30 * std::sin(1.3 * t + 2.2));
  }
  Eigen::Vector3d omega_I(double t) const { // w = vee(-Rdot R^T), R = R_GtoI
    const double h = 1e-6;
    const Eigen::Matrix3d W = -(R_of(t + h) - R_of(t - h)) / (2 * h) * R_of(t).transpose();
    return Eigen::Vector3d(W(2, 1), W(0, 2), W(1, 0));
  }
  Eigen::Vector3d accel_I(double t, const Eigen::Vector3d &g_W) const { // a_hat = R_GtoI (pddot + g)
    const double h = 1e-4;
    const Eigen::Vector3d pdd = (p_of(t + h) - 2 * p_of(t) + p_of(t - h)) / (h * h);
    return R_of(t) * (pdd + g_W);
  }
};

/// Raw IMU sample at time t (intrinsic model inverted, biases added).
/// g-sensitivity: the imu2 forward map is w_hat = Dw*(wm - bg - Tg*a_hat) with
/// a_hat = R_AtoI*Da*(am - ba), so the INVERSE injects wm += Tg*a_hat_true. Guarded
/// on a nonzero truth Tg so Tg=0 worlds stay byte-identical to the validated gates.
inline RawImu raw_imu_at(const Truth &tr, const Trajectory &tj, double t, std::mt19937 &rng, double w_noise = 0.0, double a_noise = 0.0,
                         double temp_c = 30.0) {
  std::normal_distribution<double> nrm(0.0, 1.0);
  const Eigen::Matrix3d Dw_i = ImuIntrinsicModel::ut(tr.imu.dw).inverse();
  const Eigen::Matrix3d Da_i = ImuIntrinsicModel::ut(tr.imu.da).inverse();
  const Eigen::Matrix3d R_A_t = ov_core::quat_2_Rot(tr.imu.q_AtoI).transpose();
  RawImu s;
  s.timestamp = t;
  s.wm = Dw_i * tj.omega_I(t) + tr.bg + w_noise * Eigen::Vector3d(nrm(rng), nrm(rng), nrm(rng));
  if (!tr.imu.Tg.isZero())
    s.wm += tr.imu.Tg * tj.accel_I(t, tr.g_W);
  s.am = Da_i * (R_A_t * tj.accel_I(t, tr.g_W)) + tr.ba + a_noise * Eigen::Vector3d(nrm(rng), nrm(rng), nrm(rng));
  s.temp_c = temp_c;
  return s;
}

/// Project the static cloud into the camera at IMU-time ti. Returns pixel + validity.
/// Applies the truth radtan distortion (k1, k2, p1, p2 in cam(4..7)); with all-zero
/// coefficients this is bit-identical to the pinhole used by the frozen gates.
inline bool project(const Truth &tr, const Trajectory &tj, double ti, const Eigen::Vector3d &pf, Eigen::Vector2d &uv) {
  const Eigen::Matrix3d R_ItoC = ov_core::quat_2_Rot(tr.q_ItoC);
  const Eigen::Vector3d pc = R_ItoC * (tj.R_of(ti) * (pf - tj.p_of(ti))) + tr.p_IinC;
  if (pc(2) < 0.4)
    return false;
  double x = pc(0) / pc(2), y = pc(1) / pc(2);
  if (tr.cam.tail<4>().cwiseAbs().sum() > 0.0) {
    const double k1 = tr.cam(4), k2 = tr.cam(5), p1 = tr.cam(6), p2 = tr.cam(7);
    const double r2 = x * x + y * y;
    const double rad = 1.0 + k1 * r2 + k2 * r2 * r2;
    const double xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    x = xd;
    y = yd;
  }
  uv = Eigen::Vector2d(tr.cam(0) * x + tr.cam(2), tr.cam(1) * y + tr.cam(3));
  return uv(0) >= 10 && uv(0) <= tr.img_w - 10 && uv(1) >= 10 && uv(1) <= tr.img_h - 10;
}

/// Static world cloud around the trajectory
inline std::vector<Eigen::Vector3d> make_cloud(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::vector<Eigen::Vector3d> pf(n);
  for (int f = 0; f < n; ++f)
    pf[f] = Eigen::Vector3d(4.5 * uni(rng), 4.5 * uni(rng), 3.0 + 2.5 * uni(rng));
  return pf;
}

/**
 * @brief Harvest-ready window (bearings filled, NO truth seeds).
 * Clone times are laid down at the truth td (i.e. td_ref = truth td) unless
 * td_ref_override is given — pass the SEED td to exercise the temporal
 * transport exactly like production.
 */
inline WindowData make_window(const Truth &tr, double t_start, double dur, double fps, double imu_hz, unsigned seed,
                              double pix_noise = 0.5, double td_ref_override = -1e9, double phase = 0.0) {
  Trajectory tj;
  tj.phase = phase + 0.13 * t_start;
  WindowData w;
  w.pix_sigma = pix_noise;
  std::mt19937 rng(seed);
  std::normal_distribution<double> nrm(0.0, 1.0);

  const double td_ref = (td_ref_override > -1e8) ? td_ref_override : tr.td;
  w.td_ref.assign(1, td_ref); // the synthetic world is a single-camera rig
  w.tr_ref.assign(1, 0.0);

  for (double t = t_start - 0.06; t <= t_start + dur + 0.06 + std::abs(tr.td) + 0.01; t += 1.0 / imu_hz)
    w.imu.push_back(raw_imu_at(tr, tj, t, rng, 2e-4, 2e-3));

  const int NF = 60;
  auto pf = make_cloud(NF, seed ^ 0x9e3779b9u);
  std::vector<int> id_map(NF, -1);
  int next_id = 0;
  for (double tc = t_start; tc <= t_start + dur + 1e-9; tc += 1.0 / fps) {
    // camera stamp tc; TRUE imaging time tc + td; clone laid down at tc + td_ref
    w.clone_times.push_back(tc + td_ref);
    w.obs.emplace_back();
    for (int f = 0; f < NF; ++f) {
      Eigen::Vector2d uv;
      if (!project(tr, tj, tc + tr.td, pf[f], uv))
        continue;
      if (id_map[f] < 0)
        id_map[f] = next_id++;
      CloneObs o;
      o.feat_id = (size_t)id_map[f];
      o.uv = uv + pix_noise * Eigen::Vector2d(nrm(rng), nrm(rng));
      o.u_frac = uv(1) / tr.img_h;
      o.bearing = Eigen::Vector3d((o.uv(0) - tr.cam(2)) / tr.cam(0), (o.uv(1) - tr.cam(3)) / tr.cam(1), 1.0).normalized();
      w.obs.back().push_back(o);
    }
  }
  w.num_feats = (size_t)next_id;
  return w;
}

/// Live-style streams over [0, dur]: IMU @ imu_hz and tracked frames @ fps with
/// drop injection. Frame pts use stable track ids from the cloud index.
struct StreamOptions {
  double dur = 60.0, fps = 30.0, imu_hz = 800.0;
  double pix_noise = 0.5;
  double w_noise = 2e-4, a_noise = 2e-3;
  std::vector<std::pair<double, double>> excite; ///< excited segments (single-traj envelope per segment NOT supported; use one)
  std::vector<int> drop_frames;                  ///< frame seq numbers to drop
  double exposure_s = 0.004;
  double temp_c0 = 28.0, temp_slope = 0.0; ///< deg C per second (thermal-gate tests)
};

inline void make_streams(const Truth &tr, const Trajectory &tj, const StreamOptions &so, unsigned seed, std::vector<RawImu> &imu,
                         std::vector<FrameObs> &frames) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nrm(0.0, 1.0);
  imu.clear();
  frames.clear();
  for (double t = 0.0; t <= so.dur; t += 1.0 / so.imu_hz)
    imu.push_back(raw_imu_at(tr, tj, t, rng, so.w_noise, so.a_noise, so.temp_c0 + so.temp_slope * t));
  const int NF = 90;
  auto pf = make_cloud(NF, seed ^ 0x51ed270bu);
  uint32_t seq = 0;
  for (double tc = 0.2; tc + tr.td <= so.dur - 0.01; tc += 1.0 / so.fps, ++seq) {
    bool dropped = false;
    for (int d : so.drop_frames)
      if ((uint32_t)d == seq)
        dropped = true;
    if (dropped)
      continue;
    FrameObs fo;
    // stamp = START of exposure; the trajectory is sampled at mid-exposure
    fo.timestamp = tc - 0.5 * so.exposure_s;
    fo.exposure_s = (float)so.exposure_s;
    fo.temp_c = (float)(so.temp_c0 + so.temp_slope * tc);
    fo.seq = seq;
    for (int f = 0; f < NF; ++f) {
      Eigen::Vector2d uv;
      if (!project(tr, tj, tc + tr.td, pf[f], uv))
        continue;
      FrameObsPoint p;
      p.id = (uint32_t)f;
      p.u = (float)(uv(0) + so.pix_noise * nrm(rng));
      p.v = (float)(uv(1) + so.pix_noise * nrm(rng));
      fo.pts.push_back(p);
    }
    frames.push_back(fo);
  }
}

} // namespace synth
} // namespace ov_zcalib

#endif // OV_ZCALIB_TESTLIB_SYNTH_WORLD_H
