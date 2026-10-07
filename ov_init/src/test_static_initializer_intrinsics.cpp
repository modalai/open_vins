/*
 * Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Stationary raw-sensor fixtures check the physical initialization equations,
 * independent of the initializer's implementation and triangular IMU model.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "static/StaticInitializer.h"
#include "types/IMU.h"
#include "utils/helper.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"

namespace {
int failures = 0;
void check(bool ok, const char *message) {
  if (!ok) { ++failures; std::printf("FAIL: %s\n", message); }
}
struct Result {
  bool ok = false;
  double time = -1.0;
  Eigen::MatrixXd P;
  std::shared_ptr<ov_type::IMU> imu = std::make_shared<ov_type::IMU>();
  std::vector<std::shared_ptr<ov_type::Type>> order;
};
using Samples = std::shared_ptr<std::vector<ov_core::ImuData>>;
Samples readings(const Eigen::Vector3d &am, const Eigen::Vector3d &wm, double variation = 0.0, bool only_newer = false) {
  auto out = std::make_shared<std::vector<ov_core::ImuData>>();
  for (int k = 0; k <= 400; ++k) {
    ov_core::ImuData x; x.timestamp = 10.0 + k / 200.0; x.am = am; x.wm = wm;
    if (!only_newer || k > 200) x.am.x() += (k % 2 ? variation : -variation);
    out->push_back(x);
  }
  return out;
}
Result run(ov_init::InertialInitializerOptions options, Samples samples, bool wait_for_jerk = false) {
  options.init_window_time = 2.0;
  auto db = std::make_shared<ov_core::FeatureDatabase>();
  ov_init::StaticInitializer init(options, db, samples);
  Result r; r.P = Eigen::MatrixXd::Constant(2, 2, 7.0);
  r.ok = init.initialize(r.time, r.P, r.order, r.imu, wait_for_jerk);
  if (!r.ok) {
    check(r.time == -1.0 && r.P.rows() == 2 && (r.P.array() == 7.0).all() && r.order.empty(),
          "refusal leaves output state/covariance ownership untouched");
  }
  return r;
}
void stationary(const Result &r, const Eigen::Matrix3d &A, const Eigen::Matrix3d &Tg, const Eigen::Matrix3d &gyro_map,
                const Eigen::Vector3d &am, const Eigen::Vector3d &wm, double g) {
  check(r.ok, "stationary calibrated sample accepted"); if (!r.ok) return;
  const Eigen::Vector3d a = A * (am - r.imu->bias_a());
  const Eigen::Vector3d w = gyro_map * (wm - r.imu->bias_g() - Tg * a);
  const Eigen::Vector3d acceleration_global = r.imu->Rot().transpose() * a - Eigen::Vector3d(0, 0, g);
  check(acceleration_global.norm() < 2e-12, "stationary acceleration propagates to zero in global frame");
  check(w.norm() < 2e-12, "stationary corrected angular rate is zero including Tg");
  check((r.imu->Rot().col(2) - (A * am).normalized()).norm() < 1e-12, "tilt follows calibrated specific force");
  check((r.imu->value() - r.imu->fej()).norm() == 0, "FEJ matches calibrated static seed");
  check(r.P.rows() == 15 && r.P.cols() == 15 && r.order.size() == 1 && r.order.front() == r.imu,
        "static covariance retains 15D raw-bias ordering");
  std::printf("stationary residual acceleration=%.3e gyro=%.3e\n", acceleration_global.norm(), w.norm());
}
} // namespace

int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  const Eigen::Vector3d am(-0.125, 0.0625, -10.0), wm(0.03125, -0.015625, 0.0078125);
  const auto data = readings(am, wm);
  ov_init::InertialInitializerOptions identity;
  auto id = run(identity, data);
  stationary(id, Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity(), am, wm, identity.gravity_mag);
  // Legacy identity equations and binary-exact constant samples pin existing
  // state/covariance behavior without a fitted trajectory or a tolerance change.
  Eigen::Matrix3d R_legacy;
  ov_init::InitializerHelper::gram_schmidt(am / am.norm(), R_legacy);
  const auto q_legacy = ov_core::rot_2_quat(R_legacy);
  const Eigen::Vector3d ba_legacy = am - ov_core::quat_2_Rot(q_legacy) * Eigen::Vector3d(0, 0, identity.gravity_mag);
  Eigen::Matrix<double, 16, 1> legacy = Eigen::Matrix<double, 16, 1>::Zero();
  legacy.head<4>() = q_legacy; legacy.segment<3>(10) = wm; legacy.tail<3>() = ba_legacy;
  check(id.ok && std::memcmp(id.imu->value().data(), legacy.data(), 16 * sizeof(double)) == 0,
        "identity calibration retains exact legacy mean bytes");
  Eigen::MatrixXd P_legacy = std::pow(.02, 2) * Eigen::MatrixXd::Identity(15, 15);
  P_legacy.block<3,3>(3,3) = std::pow(.05, 2) * Eigen::Matrix3d::Identity();
  P_legacy.block<3,3>(6,6) = std::pow(.01, 2) * Eigen::Matrix3d::Identity();
  check(id.P.rows() == 15 && std::memcmp(id.P.data(), P_legacy.data(), 225 * sizeof(double)) == 0,
        "identity calibration retains exact legacy covariance bytes");
  check(id.time == 11.0, "legacy window timestamp preserved");

  ov_init::InertialInitializerOptions calibrated = identity;
  Eigen::Matrix3d upper; upper << 1.03,.015,-.012, 0,.97,.008, 0,0,1.02;
  calibrated.init_imu_accel_map = ov_core::exp_so3(Eigen::Vector3d(.2,-.12,.1)) * upper;
  calibrated.init_imu_tg << 4e-4,-2e-4,1e-4, 3e-4,5e-4,-2e-4, -1e-4,3e-4,2e-4;
  const Eigen::Matrix3d gyro_map = ov_core::exp_so3(Eigen::Vector3d(-.03,.02,.01)) * upper;
  auto first = run(calibrated, data);
  stationary(first, calibrated.init_imu_accel_map, calibrated.init_imu_tg, gyro_map, am, wm, calibrated.gravity_mag);

  // A pure body-gauge change rotates corrected vectors but must preserve raw
  // biases, acceptance, and camera motion after removing the arbitrary yaw.
  const Eigen::Matrix3d S = ov_core::exp_so3(Eigen::Vector3d(-.31,.18,.23));
  auto changed = calibrated;
  changed.init_imu_accel_map = S * calibrated.init_imu_accel_map;
  changed.init_imu_tg = calibrated.init_imu_tg * S.transpose();
  auto second = run(changed, data);
  stationary(second, changed.init_imu_accel_map, changed.init_imu_tg, S * gyro_map, am, wm, calibrated.gravity_mag);
  if (first.ok && second.ok) {
    check((first.imu->bias_a() - second.imu->bias_a()).norm() < 2e-12 &&
          (first.imu->bias_g() - second.imu->bias_g()).norm() < 2e-12, "raw bias estimates invariant under body gauge");
    const Eigen::Matrix3d yaw = first.imu->Rot().transpose() * S.transpose() * second.imu->Rot();
    check((yaw * Eigen::Vector3d::UnitZ() - Eigen::Vector3d::UnitZ()).norm() < 1e-12,
          "gauge-equivalent static attitudes differ only by free global yaw");
    const Eigen::Matrix3d camera_R = ov_core::exp_so3(Eigen::Vector3d(.6,.2,-.3));
    check((camera_R * S.transpose() * second.imu->Rot() * yaw.transpose() - camera_R * first.imu->Rot()).norm() < 1e-12,
          "camera relative attitude agrees after shared-yaw alignment");
    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(15,15); J.block<3,3>(0,0) = S;
    check((J * first.P * J.transpose() - second.P).norm() < 1e-14, "static covariance is gauge equivalent with raw bias blocks");
  }
  // Gate reference remains the same physical raw mounting direction. A rigid
  // sensor-coordinate change cannot bypass the legacy 5-degree tilt limit.
  const Eigen::Vector3d tilted(.25,0,-1);
  check(!run(calibrated, readings(10*tilted,wm)).ok && !run(changed, readings(10*tilted,wm)).ok,
        "tilted fixture refused in both gauges without widening gate");
  check(!run(calibrated, data, true).ok, "no jerk still refuses initialization when required");
  check(run(calibrated, readings(am,wm,2.0,true), true).ok && run(changed, readings(am,wm,2.0,true), true).ok,
        "new-window jerk acceptance is gauge invariant");
  check(!run(calibrated, readings(am,wm,2.0)).ok && !run(changed, readings(am,wm,2.0)).ok,
        "moving-window refusal is gauge invariant");
  auto scaled = identity; scaled.init_imu_accel_map(0,0)=2.0;
  check(run(identity, readings(am,wm,.6)).ok && !run(scaled, readings(am,wm,.6)).ok,
        "stationarity threshold measures corrected acceleration variation");

  // Refuse bad calibration/data before publishing a mean or covariance, also
  // in the production fast-math build where std::isfinite is not reliable.
  auto bad = calibrated; bad.init_imu_accel_map.row(2).setZero(); check(!run(bad,data).ok,"singular accel map refused");
  bad = calibrated; bad.init_imu_accel_map(0,0)=std::numeric_limits<double>::quiet_NaN(); check(!run(bad,data).ok,"NaN accel map refused");
  bad = calibrated; bad.init_imu_tg(1,2)=std::numeric_limits<double>::infinity(); check(!run(bad,data).ok,"infinite Tg refused");
  bad = identity; bad.init_imu_accel_map(2,2)=-1; check(!run(bad,data).ok,"reflected accel map refused");
  bad = identity; bad.init_imu_accel_map(2,2)=1e-16; check(!run(bad,data).ok,"ill-conditioned accel map refused");
  check(!run(identity,readings(Eigen::Vector3d::Zero(),wm)).ok,"zero mean gravity refused");
  auto nonfinite=readings(am,wm); nonfinite->at(200).wm.x()=std::numeric_limits<double>::quiet_NaN();
  check(!run(calibrated,nonfinite).ok,"nonfinite raw IMU refused");
  std::printf("STATIC_INIT_INTRINSICS %s failures=%d\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
