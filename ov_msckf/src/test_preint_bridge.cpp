/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 *
 * ACI2 PreintegrationBridge oracle tests (host, CTest).
 *
 * 1. MEAN: the bridge composition must reproduce dense zeroth-order integration of the same
 *    piecewise-constant signals (reference: fine-substep JPL integration) to numerical precision.
 * 2. J_b: the ANALYTIC bias Jacobians must match finite-difference rebuilds (FD is the test
 *    oracle only -- production math stays closed-form per the project policy).
 * 3. CORRECTION CONVENTION: bridge(b0) corrected via J_b to b1 must match bridge(b1).
 */

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>

#include "state/Propagator.h"
#include "state/State.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_msckf;

static int failures = 0;
#define CHECK(cond, ...)                                                                                                                   \
  do {                                                                                                                                     \
    if (!(cond)) {                                                                                                                         \
      failures++;                                                                                                                          \
      std::printf("[FAIL] %s:%d: ", __func__, __LINE__);                                                                                   \
      std::printf(__VA_ARGS__);                                                                                                            \
      std::printf("\n");                                                                                                                   \
    }                                                                                                                                      \
  } while (0)

// Build a state with no intrinsic distortion and the given biases
static std::shared_ptr<State> make_state(const Eigen::Vector3d &bg, const Eigen::Vector3d &ba) {
  StateOptions opts;
  opts.num_cameras = 1;
  opts.do_calib_imu_intrinsics = false;
  auto state = std::make_shared<State>(opts);
  Eigen::Matrix<double, 16, 1> imu_x = Eigen::Matrix<double, 16, 1>::Zero();
  imu_x(3) = 1.0; // unit quaternion
  imu_x.block(10, 0, 3, 1) = bg;
  imu_x.block(13, 0, 3, 1) = ba;
  state->_imu->set_value(imu_x);
  state->_imu->set_fej(imu_x);
  return state;
}

// Synthetic IMU stream: smoothly varying rates so no term integrates to zero by accident
static std::vector<ov_core::ImuData> make_imu(double t0, double t1, double rate_hz) {
  std::vector<ov_core::ImuData> v;
  const double dt = 1.0 / rate_hz;
  for (double t = t0 - 2 * dt; t <= t1 + 2 * dt; t += dt) {
    ov_core::ImuData s;
    s.timestamp = t;
    s.wm = Eigen::Vector3d(0.8 * std::sin(3 * t), -1.1 * std::cos(2 * t), 0.6 * std::sin(5 * t) + 0.4);
    s.am = Eigen::Vector3d(0.9 * std::cos(4 * t), 9.81 + 0.5 * std::sin(2 * t), -0.7 * std::sin(3 * t));
    v.push_back(s);
  }
  return v;
}

// Reference: dense zeroth-order (piecewise-constant, midpoint-signal) JPL integration of the SAME
// interval, matching the bridge's signal model with fine substeps of the per-sample intervals
static void reference_integrate(const std::vector<ov_core::ImuData> &imu, double t0, double t1, const Eigen::Vector3d &bg,
                                const Eigen::Vector3d &ba, Eigen::Matrix3d &DR, Eigen::Vector3d &alpha, Eigen::Vector3d &beta) {
  auto data = Propagator::select_imu_readings(imu, t0, t1, false);
  DR.setIdentity();
  alpha.setZero();
  beta.setZero();
  for (size_t i = 0; i + 1 < data.size(); i++) {
    const double DT = data.at(i + 1).timestamp - data.at(i).timestamp;
    const Eigen::Vector3d w = 0.5 * (data.at(i).wm + data.at(i + 1).wm) - bg;
    const Eigen::Vector3d a = 0.5 * (data.at(i).am + data.at(i + 1).am) - ba;
    const int n_sub = 2000;
    const double dts = DT / n_sub;
    for (int k = 0; k < n_sub; k++) {
      const Eigen::Matrix3d A = DR.transpose();
      alpha += beta * dts + 0.5 * A * a * dts * dts;
      beta += A * a * dts;
      DR = ov_core::exp_so3(-w * dts) * DR;
    }
  }
}

static void test_mean_exactness() {
  const double t0 = 10.000, t1 = 10.020; // 20 ms bridge @ 800 Hz
  const Eigen::Vector3d bg(0.01, -0.02, 0.015), ba(0.05, 0.02, -0.04);
  auto state = make_state(bg, ba);
  NoiseManager noises;
  Propagator prop(noises, 9.81);
  auto imu = make_imu(t0, t1, 800.0);
  for (auto &s : imu)
    prop.feed_imu(s);

  Propagator::BridgeData bd;
  CHECK(prop.compute_bridge(state, t0, t1, bd), "bridge build failed");

  Eigen::Matrix3d DR_ref;
  Eigen::Vector3d al_ref, be_ref;
  reference_integrate(imu, t0, t1, bg, ba, DR_ref, al_ref, be_ref);

  const double rot_err = ov_core::log_so3(bd.DR * DR_ref.transpose()).norm();
  CHECK(rot_err < 1e-9, "DR mismatch: %.3e rad", rot_err);
  CHECK((bd.alpha - al_ref).norm() < 1e-9, "alpha mismatch: %.3e (|alpha|=%.3e)", (bd.alpha - al_ref).norm(), al_ref.norm());
  CHECK((bd.beta - be_ref).norm() < 5e-7, "beta mismatch: %.3e (|beta|=%.3e)", (bd.beta - be_ref).norm(), be_ref.norm());
  std::printf("[ok] mean_exactness: DR %.2e rad, alpha %.2e, beta %.2e vs dense reference\n", rot_err, (bd.alpha - al_ref).norm(),
              (bd.beta - be_ref).norm());
}

// The J_th_g increment must be Jr(+w dt), not Jr(-w dt): the two differ by O(|w| dt) relative,
// so the oracle runs a low-rate case (200 Hz -> ~6e-3 with the wrong flavor) under a tolerance
// (1e-6) far below that gap but far above the exact form's residual (~4e-9 measured).
static void test_bias_jacobians_fd(double rate_hz, double tol) {
  const double t0 = 20.000, t1 = 20.030; // 30 ms
  const Eigen::Vector3d bg(0.004, -0.01, 0.02), ba(0.03, -0.05, 0.01);
  NoiseManager noises;
  auto imu = make_imu(t0, t1, rate_hz);

  auto build = [&](const Eigen::Vector3d &bgi, const Eigen::Vector3d &bai, Propagator::BridgeData &out) {
    auto state = make_state(bgi, bai);
    Propagator prop(noises, 9.81);
    for (auto &s : imu)
      prop.feed_imu(s);
    return prop.compute_bridge(state, t0, t1, out);
  };

  Propagator::BridgeData b0;
  CHECK(build(bg, ba, b0), "bridge build failed");

  const double eps = 1e-5;
  double max_rel = 0.0;
  for (int j = 0; j < 6; j++) {
    Eigen::Vector3d bgp = bg, bap = ba, bgm = bg, bam = ba;
    if (j < 3) {
      bgp(j) += eps;
      bgm(j) -= eps;
    } else {
      bap(j - 3) += eps;
      bam(j - 3) -= eps;
    }
    Propagator::BridgeData bp, bm;
    CHECK(build(bgp, bap, bp) && build(bgm, bam, bm), "FD rebuild failed");
    // theta rows: d(theta)/db with DR(b) = exp_so3(J_th db) DR(b0) => log(DR(b+) DR(b-)^T)/2eps
    Eigen::Vector3d fd_th = ov_core::log_so3(bp.DR * bm.DR.transpose()) / (2 * eps);
    Eigen::Vector3d fd_al = (bp.alpha - bm.alpha) / (2 * eps);
    Eigen::Vector3d fd_be = (bp.beta - bm.beta) / (2 * eps);
    Eigen::Matrix<double, 9, 1> fd, an;
    fd << fd_th, fd_al, fd_be;
    an = b0.J_b.col(j);
    const double scale = std::max(1e-6, fd.norm());
    const double rel = (fd - an).norm() / scale;
    max_rel = std::max(max_rel, rel);
    CHECK(rel < tol, "J_b col %d @ %.0f Hz: rel err %.3e (fd norm %.3e)", j, rate_hz, rel, fd.norm());
  }
  std::printf("[ok] bias_jacobians_fd @ %.0f Hz: max relative error %.3e over 6 columns\n", rate_hz, max_rel);
}

static void test_correction_convention() {
  const double t0 = 30.000, t1 = 30.025;
  const Eigen::Vector3d bg(0.002, -0.006, 0.01), ba(0.02, -0.03, 0.015);
  const Eigen::Vector3d dbg(0.004, 0.003, -0.005), dba(0.02, -0.015, 0.01); // realistic bias motion
  NoiseManager noises;
  auto imu = make_imu(t0, t1, 800.0);
  auto build = [&](const Eigen::Vector3d &bgi, const Eigen::Vector3d &bai, Propagator::BridgeData &out) {
    auto state = make_state(bgi, bai);
    Propagator prop(noises, 9.81);
    for (auto &s : imu)
      prop.feed_imu(s);
    return prop.compute_bridge(state, t0, t1, out);
  };
  Propagator::BridgeData b0, b1;
  CHECK(build(bg, ba, b0) && build(bg + dbg, ba + dba, b1), "builds failed");

  Eigen::Matrix<double, 6, 1> db;
  db << dbg, dba;
  const Eigen::Matrix3d DR_corr = ov_core::exp_so3(b0.J_b.block(0, 0, 3, 6) * db) * b0.DR;
  const Eigen::Vector3d al_corr = b0.alpha + b0.J_b.block(3, 0, 3, 6) * db;
  const Eigen::Vector3d be_corr = b0.beta + b0.J_b.block(6, 0, 3, 6) * db;

  const double rot_err = ov_core::log_so3(DR_corr * b1.DR.transpose()).norm();
  const double al_err = (al_corr - b1.alpha).norm();
  const double be_err = (be_corr - b1.beta).norm();
  // First-order correction: residual must be second order in |db| (~|db|^2 * dt scale)
  CHECK(rot_err < 1e-6, "corrected DR err %.3e rad", rot_err);
  CHECK(al_err < 5e-7, "corrected alpha err %.3e", al_err);
  CHECK(be_err < 5e-7, "corrected beta err %.3e", be_err);
  std::printf("[ok] correction_convention: rot %.2e rad, alpha %.2e, beta %.2e residual after J_b correction\n", rot_err, al_err, be_err);
}

int main() {
  ov_core::Printer::setPrintLevel("WARNING");
  test_mean_exactness();
  test_bias_jacobians_fd(800.0, 1e-6);
  test_bias_jacobians_fd(200.0, 1e-6);
  test_correction_convention();
  if (failures == 0) {
    std::printf("[PASS] PreintegrationBridge: mean exact, analytic J_b == FD oracle, correction convention verified\n");
    return 0;
  }
  std::printf("[FAILED] PreintegrationBridge: %d failed checks\n", failures);
  return 1;
}
