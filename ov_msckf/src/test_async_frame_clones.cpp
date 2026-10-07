/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstdio>
#include <fstream>
#include <limits>
#include <unistd.h>

#include "core/VioManager.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/Propagator.h"
#include "state/StateHelper.h"
#include "sim/SimulationTruthTime.h"

using namespace ov_msckf;
namespace {
int failures = 0;
void check(bool ok, const char *name) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
  failures += !ok;
}

VioManagerOptions options(bool enabled, int cameras = 2, bool stereo = false) {
  VioManagerOptions p;
  p.state_options.num_cameras = cameras;
  p.state_options.max_clone_size = 4;
  p.state_options.max_slam_features = 0;
  p.state_options.max_aruco_features = 0;
  p.state_options.imu_model = StateOptions::RPNG;
  p.async_frame_clones = enabled;
  p.epoch_mode = true; // new policy takes precedence only for independent cameras
  p.use_stereo = stereo;
  p.use_aruco = false;
  p.use_gpu = false;
  p.try_zupt = false;
  p.num_opencv_threads = 0;
  p.use_multi_threading_pubs = p.use_multi_threading_subs = false;
  p.vec_dw << 1, 0, 1, 0, 0, 1;
  p.vec_da = p.vec_dw;
  p.vec_tg.setZero();
  p.q_ACCtoIMU << 0, 0, 0, 1;
  p.q_GYROtoIMU = p.q_ACCtoIMU;
  p.init_options.num_cameras = cameras;
  p.init_options.use_stereo = stereo;
  for (int c = 0; c < cameras; ++c) {
    auto camera = std::make_shared<ov_core::CamRadtan>(320, 240);
    Eigen::VectorXd intrinsic(8);
    intrinsic << 220, 220, 160, 120, 0, 0, 0, 0;
    camera->set_value(intrinsic);
    Eigen::VectorXd extrinsic(7);
    extrinsic << 0, 0, 0, 1, .1*c, 0, 0;
    p.camera_intrinsics[c] = camera;
    p.camera_extrinsics[c] = extrinsic;
    p.init_options.camera_intrinsics[c] = camera;
    p.init_options.camera_extrinsics[c] = extrinsic;
    p.camera_imu_dt[c] = 0.0;
  }
  return p;
}

struct Manager : VioManager {
  using VioManager::VioManager;
  const VioManagerOptions &resolved() const { return params; }
};

void seed(Manager &manager) {
  Eigen::Matrix<double,17,1> x = Eigen::Matrix<double,17,1>::Zero();
  x(0) = 1.0;
  x(4) = 1.0;
  x(8) = 0.5;
  manager.initialize_with_gt(x);
  for (int i = 0; i <= 160; ++i) {
    ov_core::ImuData sample;
    sample.timestamp = .95 + .0025*i;
    sample.wm.setZero();
    sample.am << 0, 0, 9.81;
    manager.feed_measurement_imu(sample);
  }
}

void option_checks() {
  for (int cameras : {1, 2, 3}) for (bool stereo : {false, true}) for (bool enabled : {false, true}) {
    auto p = options(enabled, cameras, stereo);
    const bool independent = enabled && cameras > 1 && !stereo;
    const int expected = independent ? 4*cameras : 4;
    check(p.state_options.configure_clone_policy(enabled, stereo) && p.state_options.max_pose_clones() == expected &&
          p.state_options.max_clone_size == 4 && p.use_async_frame_clones() == independent &&
          p.use_epoch_clones() == (!p.synchronize_camera_timestamps() && !independent),
          "default/single/stereo/independent policies derive one finite total capacity");
    for (int reset = 0; reset < 3; ++reset)
      check(p.state_options.configure_clone_policy(enabled, stereo) && p.state_options.max_pose_clones() == expected &&
            p.state_options.max_clone_size == 4, "repeated derivation never compounds the configured track length");
  }
  StateOptions invalid;
  invalid.max_clone_size = 0;
  check(!invalid.configure_clone_policy(false, true), "zero configured clone count refused");
  invalid.max_clone_size = 3;
  invalid.num_cameras = 0;
  check(!invalid.configure_clone_policy(true, false), "zero camera count refused");
  invalid.num_cameras = 2;
  invalid.max_clone_size = std::numeric_limits<int>::max();
  check(!invalid.configure_clone_policy(true, false), "overflowing total pose capacity refused");

  char path[] = "/tmp/ov-frame-clones-XXXXXX";
  const int fd = mkstemp(path);
  if (fd < 0) { check(false, "temporary options file"); return; }
  close(fd);
  for (int setting : {-1, 0, 1}) {
    { std::ofstream out(path); out << "%YAML:1.0\n---\nfixture: 1\n";
      if (setting >= 0) out << "async_frame_clones: " << (setting ? "true" : "false") << "\n";
    }
    auto parser = std::make_shared<ov_core::YamlParser>(path);
    auto p = options(false);
    p.resolve_camera_epoch_mode(parser);
    check(p.async_frame_clones == (setting == 1) && parser->successful(), "new YAML policy defaults off and obeys explicit opt-in");
  }
  unlink(path);
}

void manager_checks() {
  auto excessive=options(false);
  excessive.epoch_mode=true;
  excessive.state_options.max_clone_size=33;
  bool rejected=false;
  try { Manager invalid(excessive); }
  catch(const std::invalid_argument &) { rejected=true; }
  check(rejected,"undeclared-rate epoch capacity also obeys the resource bound before state allocation");
  auto raw_options = options(true), epoch_options = options(false);
  Manager raw(raw_options), epoch(epoch_options);
  check(raw.get_state()->_options.max_pose_clones() == 8 && epoch.get_state()->_options.max_pose_clones() == 8,
        "frame and stochastic epoch policies allocate their bounded exposure windows");
  check(!raw.get_state()->uses_physical_clones() && epoch.get_state()->_options.stochastic_epoch_transport,
        "epoch resolves to stochastic physical ownership, with no snapping policy");
  seed(raw); seed(epoch);
  check(raw.get_propagator()->propagate_and_clone(raw.get_state(),1.01),"raw reference clone propagation succeeds");
  check(raw.get_propagator()->propagate_and_clone(raw.get_state(),1.023) && raw.get_state()->_clones_IMU.count(1.023),
        "independent frame keeps its own raw stochastic clone");
  epoch.feed_measurement_simulation(1.01,{0},{{}});
  epoch.feed_measurement_simulation(1.023,{1},{{}});
  check(epoch.get_state()->find_pose(0,1.01) && epoch.get_state()->find_pose(1,1.023) &&
        !epoch.get_state()->find_pose(1,1.01) && epoch.get_state()->_clones_IMU.empty(),
        "epoch preserves each camera raw key and covariance-owned exposure");
  const std::vector<double> truth_offsets{.03125,.25};
  check(simulation_truth_time(*epoch.get_state(),truth_offsets)==epoch.get_state()->imu_endpoint() &&
        epoch.get_state()->imu_endpoint()==1.023,
        "epoch truth scoring uses the accepted endpoint, independently of later clock relabeling");
  StateOptions alternate_clock;
  alternate_clock.num_cameras = 2;
  alternate_clock.cam_imu_dt_ref_camid = 1;
  State alternate_state(alternate_clock);
  alternate_state._timestamp = 5.0;
  check(simulation_truth_time(alternate_state, truth_offsets) == 5.25,
        "simulation truth time follows the configured reference camera");
  for (int reset = 0; reset < 3; ++reset) {
    raw.soft_reset();
    check(raw.get_state()->_options.max_clone_size == 4 && raw.get_state()->_options.max_pose_clones() == 8 &&
          raw.resolved().state_options.max_clone_size == 4 && raw.get_state()->_clones_IMU.empty(),
          "real soft reset preserves configured track length and total pose capacity");
  }
}

void covariance_and_capacity_checks() {
  StateOptions derived, reference;
  derived.num_cameras = reference.num_cameras = 2;
  derived.max_clone_size = 3;
  reference.max_clone_size = 6;
  check(derived.configure_clone_policy(true, false) && reference.configure_clone_policy(false, false), "covariance fixture capacities");
  auto a = std::make_shared<State>(derived), b = std::make_shared<State>(reference);
  a->_timestamp = b->_timestamp = 1.0;
  Eigen::MatrixXd L = Eigen::MatrixXd::Identity(15,15);
  for (int i = 0; i < 15; ++i) for (int j = 0; j < i; ++j) L(i,j) = .03*std::sin(i+2*j);
  const Eigen::MatrixXd initial = .01*L*L.transpose();
  for (auto state : {a,b}) {
    StateHelper::set_initial_covariance(state, initial, {state->_imu});
    Eigen::VectorXd td(1); td << .125;
    state->cam_imu_dt_var(0)->set_value(td); state->cam_imu_dt_var(0)->set_fej(td);
    td << -.25;
    state->cam_imu_dt_var(1)->set_value(td); state->cam_imu_dt_var(1)->set_fej(td);
  }
  NoiseManager noise;
  Propagator pa(noise, 9.81), pb(noise, 9.81);
  for (int i = 0; i <= 120; ++i) {
    ov_core::ImuData sample;
    sample.timestamp = 1.1 + .002*i;
    sample.wm << .03, -.02, .04;
    sample.am << .1, -.2, 9.81;
    pa.feed_imu(sample); pb.feed_imu(sample);
  }
  const std::vector<double> times{1.01, 1.023, 1.043, 1.060, 1.075, 1.091, 1.11, 1.129};
  for (size_t k = 0; k < times.size(); ++k) {
    check(pa.propagate_and_clone(a, times[k]) && pb.propagate_and_clone(b, times[k]), "raw frame normal propagation succeeds");
    const Eigen::MatrixXd P = StateHelper::get_full_covariance(a);
    const int clone_id = a->_clones_IMU.at(times[k])->id();
    check(P.block(0,clone_id,clone_id,6) == P.block(0,0,clone_id,6) &&
          P.block(clone_id,clone_id,6,6) == P.topLeftCorner(6,6),
          "stochastic clone copies complete pose/cross covariance with no extra independent Q");
    if (k > 0)
      check(P.block(15,clone_id,6,6).norm() > .01, "consecutive frame clones retain their shared state cross-covariance");
    check(P == StateHelper::get_full_covariance(b) && a->_imu->value() == b->_imu->value(),
          "derived-cap policy leaves ordinary propagation mean and covariance exactly unchanged");
    StateHelper::marginalize_old_clone(a); StateHelper::marginalize_old_clone(b);
    check(a->_clones_IMU.size() == std::min<size_t>(k+1,6) &&
          StateHelper::get_full_covariance(a) == StateHelper::get_full_covariance(b),
          "oldest clone marginalizes exactly at the separate total capacity");
  }
  check(a->cam_imu_dt_delta(1) == -.375,
        "raw-frame policy preserves per-camera relative td and adds no bridge state");
}

void immature_track_check() {
  auto p = options(true);
  Manager manager(p);
  seed(manager);
  for (int frame = 0; frame < 6; ++frame) {
    const int camera = frame % 2;
    const double time = 1.01 + .02*frame;
    const Eigen::Vector3d point = Eigen::Vector3d(.4,.2,4.0) - Eigen::Vector3d(.5*(time-1.0),0,0) +
                                  p.camera_extrinsics.at(camera).tail<3>();
    const Eigen::Vector2f uv = p.camera_intrinsics.at(camera)->distort_f((point.head<2>()/point.z()).cast<float>());
    Eigen::VectorXf observation(2); observation = uv;
    manager.feed_measurement_simulation(time, {camera}, {{{size_t(100+camera), observation}}});
  }
  const auto features = manager.get_track_feats()->get_feature_database()->get_internal_data();
  size_t complete = 0;
  bool original_times = true;
  for (const auto &entry : features)
    for (const auto &camera : entry.second->timestamps) {
      if (camera.second.size() == 3) ++complete;
      std::vector<double> expected;
      for (int frame = int(camera.first); frame < 6; frame += 2)
        expected.push_back(1.01 + .02*frame);
      original_times = original_times && camera.second == expected;
      std::printf("ASYNC_CAMERA_HISTORY camera=%zu observations=%zu oldest=%.9f newest=%.9f\n", camera.first,
                  camera.second.size(), camera.second.empty() ? -1.0 : camera.second.front(),
                  camera.second.empty() ? -1.0 : camera.second.back());
    }
  check(manager.get_state()->_clones_IMU.size() == 6 && complete == 2 && original_times,
        "six aggregate frame clones do not consume either view's three-observation immature track");
}
} // namespace

int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  option_checks();
  manager_checks();
  covariance_and_capacity_checks();
  immature_track_check();
  std::printf("ASYNC_FRAME_CLONES %s failures=%d\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
