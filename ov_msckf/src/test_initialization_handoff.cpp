/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstdio>
#include <cstring>
#include <limits>
#include "core/VioManager.h"
#include "init/InertialInitializer.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/print.h"

// Fault injection only in this executable: the real manager calls this symbol
// instead of the optimizer. This lets us test the production caller transaction
// independently of whether one particular dataset happens to make MLE fail.
namespace fixture {
int fault = -1, calls = 0;
bool joint = false;
Eigen::Matrix<double, 16, 1> mean() {
  Eigen::Matrix<double, 16, 1> x = Eigen::Matrix<double, 16, 1>::Zero();
  x(3) = 1.; x(4) = .4; x(7) = .3; x(11) = .02;
  return x;
}
}
bool ov_init::InertialInitializer::initialize(
    double &timestamp, Eigen::MatrixXd &covariance,
    std::vector<std::shared_ptr<ov_type::Type>> &order, std::shared_ptr<ov_type::IMU> imu,
    std::map<double, std::shared_ptr<ov_type::PoseJPL>> &clones,
    std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> &features, bool) {
  ++fixture::calls;
  timestamp = 12.;
  auto value = fixture::mean(), fej = value;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  if (fixture::fault == 5) value(7) = nan;
  if (fixture::fault == 6) fej(13) = inf;
  if (fixture::fault == 7) value.head<4>().setZero();
  if (fixture::fault == 8) timestamp = inf;
  if (fixture::fault == 9) timestamp = nan;
  imu->set_value(value); imu->set_fej(fej);
  clones.clear(); features.clear(); order = {imu};
  if (fixture::joint) {
    for (double time : {11.875, 12.}) {
      auto pose = std::make_shared<ov_type::PoseJPL>();
      pose->set_value(fixture::mean().head<7>()); pose->set_fej(pose->value());
      clones.emplace(time, pose); order.push_back(pose);
    }
  }
  covariance = .01 * Eigen::MatrixXd::Identity(fixture::joint ? 27 : 15, fixture::joint ? 27 : 15);
  if (fixture::joint) {
    // The newest clone is exactly the IMU pose: a legitimate singular joint.
    covariance.bottomRows(6) = covariance.topRows(6).eval();
    covariance.rightCols(6) = covariance.leftCols(6).eval();
  }
  if (fixture::fault == 0) covariance(0, 0) = nan;
  if (fixture::fault == 1) covariance(0, 0) = inf;
  if (fixture::fault == 2) covariance(0, 1) = covariance(1, 0) = .1;
  if (fixture::fault == 3) covariance(0, 1) = .001;
  if (fixture::fault == 4) covariance(2, 2) = -1.;
  if (fixture::fault == 11) covariance = Eigen::MatrixXd::Identity(14, 14);
  if (fixture::fault == 12) covariance = Eigen::MatrixXd::Identity(15, 14);
  if (fixture::fault == 13) covariance = Eigen::MatrixXd::Identity(15, 16);
  if (fixture::fault == 14) covariance = Eigen::MatrixXd::Identity(16, 16);
  if (fixture::fault == 15) fej.head<4>().setZero(), imu->set_fej(fej);
  if (fixture::fault == 16) covariance(0, 0) = 0.;
  if (fixture::fault == 17) covariance(0, 1) = inf;
  if (fixture::fault == 18) covariance(1, 0) = nan;
  if (fixture::fault == 19) covariance.resize(0, 0);
  // Invalid *clone* covariance is allowed to fall back to a valid IMU marginal.
  if (fixture::fault == 20) covariance(15, 15) = nan;
  return fixture::fault != 10;
}

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char *why) {
  ++checks;
  if (!ok) { ++failures; std::printf("FAIL fault=%d joint=%d: %s\n", fixture::fault, fixture::joint, why); }
}
template <class A, class B> bool same(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
      std::memcmp(a.derived().data(), b.derived().data(), sizeof(double) * a.size()) == 0;
}
ov_msckf::VioManagerOptions options() {
  ov_msckf::VioManagerOptions p;
  p.state_options.num_cameras = p.init_options.num_cameras = 1;
  p.state_options.max_clone_size = 11;
  p.state_options.max_slam_features = p.state_options.max_aruco_features = 0;
  p.state_options.do_calib_camera_timeoffset = true;
  p.epoch_mode = p.async_frame_clones = false;
  p.use_stereo = p.use_aruco = p.use_gpu = p.try_zupt = false;
  p.use_multi_threading_pubs = p.use_multi_threading_subs = false;
  p.num_opencv_threads = 0;
  p.init_options.init_warmstart_inject = true;
  p.num_pts = p.init_options.init_max_features = 20;
  p.gravity_mag = 9.81;
  p.vec_dw << 1, 0, 1, 0, 0, 1; p.vec_da = p.vec_dw; p.vec_tg.setZero();
  p.q_ACCtoIMU << 0, 0, 0, 1; p.q_GYROtoIMU = p.q_ACCtoIMU;
  auto camera = std::make_shared<ov_core::CamRadtan>(320, 240);
  Eigen::VectorXd intrinsic(8), extrinsic(7);
  intrinsic << 220, 230, 160, 120, -.08, .01, .002, -.003;
  extrinsic << 0, 0, 0, 1, 0, 0, 0;
  camera->set_value(intrinsic);
  p.camera_intrinsics[0] = p.init_options.camera_intrinsics[0] = camera;
  p.camera_extrinsics[0] = p.init_options.camera_extrinsics[0] = extrinsic;
  p.camera_imu_dt[0] = .125; p.camera_readout_time[0] = 0.; p.camera_shutter_rolling[0] = false;
  return p;
}
class Manager : public ov_msckf::VioManager {
public:
  Manager() : Manager(options()) {}
  explicit Manager(ov_msckf::VioManagerOptions p) : VioManager(p) {}
  void attempt(bool warm) {
    warmstart_next_init.store(warm);
    ov_core::CameraData message; message.timestamp = 12.; message.sensor_ids = {0};
    try_to_initialize(message);
  }
  bool succeeded() const { return thread_init_success.load(); }
  bool warm_armed() const { return warmstart_next_init.load(); }
};
void invalid_result(int bad, bool warm) {
  fixture::fault = bad; fixture::joint = warm;
  Manager manager;
  auto state = manager.get_state();
  const auto value = state->_imu->value(), fej = state->_imu->fej();
  const auto cov = ov_msckf::StateHelper::get_full_covariance(state);
  const double timestamp = state->_timestamp, endpoint = state->_imu_endpoint;
  const bool endpoint_valid = state->_imu_endpoint_valid;
  auto db = manager.get_track_feats()->get_feature_database();
  db->update_feature(100, 11., 0, 1, 2, 3, 4);
  manager.attempt(warm);
  check(!manager.succeeded(), "invalid initializer output is not accepted");
  check(same(value, state->_imu->value()) && same(fej, state->_imu->fej()),
        "failed handoff does not leak staged IMU mean or FEJ into live state");
  check(same(cov, ov_msckf::StateHelper::get_full_covariance(state)) && state->_clones_IMU.empty(),
        "failed handoff leaves covariance and clone ownership unchanged");
  check(state->_timestamp == timestamp && state->_imu_endpoint == endpoint && state->_imu_endpoint_valid == endpoint_valid,
        "failed handoff leaves accepted endpoint unchanged");
  check(bool(db->get_feature(100)) && manager.warm_armed() == warm,
        "failed handoff neither consumes observations nor disarms a reset episode");
  fixture::fault = -1;
  manager.attempt(warm);
  check(manager.succeeded() && same(state->_imu->value(), fixture::mean()),
        "a valid retry after refusal succeeds without a reset");
}
void valid_result(bool warm, bool invalid_clone_only = false) {
  fixture::fault = invalid_clone_only ? 20 : -1; fixture::joint = warm;
  Manager manager; manager.attempt(warm); const auto state = manager.get_state();
  check(manager.succeeded() && same(state->_imu->value(), fixture::mean()) && same(state->_imu->fej(), fixture::mean()),
        "valid IMU posterior is installed exactly");
  check(state->_timestamp == 12. && state->_imu_endpoint == 12.125 && state->_imu_endpoint_valid,
        "valid handoff establishes the physical endpoint");
  const bool expect_warm = warm && !invalid_clone_only;
  check(state->_clones_IMU.size() == (expect_warm ? 2 : 0), "warm or validated cold fallback has the intended ownership");
  if (expect_warm)
    check((state->_clones_kinematics.rbegin()->second.vel - fixture::mean().segment<3>(7)).norm() == 0.,
          "newest warm clone kinematics use the accepted initializer velocity");
  const auto cov = ov_msckf::StateHelper::get_full_covariance(state);
  check((cov.topLeftCorner(15, 15) - .01 * Eigen::MatrixXd::Identity(15, 15)).norm() == 0.,
        "valid cold/warm marginal is not regularized or rescaled");
}
}
int main(int argc, char **) {
  ov_core::Printer::setPrintLevel("ERROR");
  // The historical caller has unchecked Eigen block extraction; avoid an OOB
  // negative control while still proving invalid numeric output is accepted.
  const bool old_safe = argc > 1;
  for (bool warm : {false, true})
    for (int bad = 0; bad < 20; ++bad) {
      if (old_safe && (bad >= 11 && bad <= 14 || bad == 19)) continue;
      invalid_result(bad, warm);
    }
  valid_result(false); valid_result(true); valid_result(true, true);
  check(fixture::calls > 20, "fault injection exercised the production manager's initializer calls");
  std::printf("INITIALIZATION_HANDOFF %s checks=%d failures=%d\n", failures ? "FAIL" : "PASS", checks, failures);
  return failures ? 1 : 0;
}
