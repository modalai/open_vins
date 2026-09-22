/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <utility>
#include <vector>

#include "core/VioManager.h"
#include "state/State.h"
#include "state/StateHelper.h"

using namespace ov_msckf;
using ov_core::CameraData;
using ov_core::ImuData;

namespace {
int failures = 0, checks = 0;
void check(bool passed, const char *name) {
  ++checks;
  failures += !passed;
  std::printf("[%s] %s\n", passed ? "PASS" : "FAIL", name);
}

bool near(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b, double tolerance = 1e-12) {
  return a.rows() == b.rows() && a.cols() == b.cols() && a.allFinite() && b.allFinite() &&
         (a.size() == 0 || (a - b).cwiseAbs().maxCoeff() < tolerance);
}

VioManagerOptions options(int cap = 2, double dt0 = .015625, double dt1 = -.0078125) {
  VioManagerOptions p;
  p.state_options.num_cameras = 2;
  p.state_options.max_clone_size = cap;
  p.state_options.max_slam_features = p.state_options.max_aruco_features = 0;
  p.state_options.physical_camera_clones = true;
  p.state_options.do_calib_camera_timeoffset = true;
  p.state_options.imu_model = StateOptions::RPNG;
  p.epoch_mode = p.async_frame_clones = false;
  p.use_stereo = p.use_aruco = p.use_gpu = p.try_zupt = false;
  p.use_multi_threading_pubs = p.use_multi_threading_subs = false;
  p.num_opencv_threads = 0;
  p.num_pts = p.init_options.init_max_features = 20;
  p.async_ring_size = 16;
  p.async_guard = .001953125;
  p.async_stale_factor = 0.;
  p.gravity_mag = 9.81;
  p.vec_dw << 1, 0, 1, 0, 0, 1;
  p.vec_da = p.vec_dw;
  p.vec_tg.setZero();
  p.q_ACCtoIMU << 0, 0, 0, 1;
  p.q_GYROtoIMU = p.q_ACCtoIMU;
  p.init_options.num_cameras = 2;
  p.init_options.use_stereo = false;
  for (int c = 0; c < 2; ++c) {
    auto camera = std::make_shared<ov_core::CamRadtan>(320, 240);
    Eigen::VectorXd intrinsic(8), extrinsic(7);
    intrinsic << 220, 220, 160, 120, 0, 0, 0, 0;
    extrinsic << 0, 0, 0, 1, .1 * c, 0, 0;
    camera->set_value(intrinsic);
    p.camera_intrinsics[c] = p.init_options.camera_intrinsics[c] = camera;
    p.camera_extrinsics[c] = p.init_options.camera_extrinsics[c] = extrinsic;
    p.camera_imu_dt[c] = c == 0 ? dt0 : dt1;
    p.camera_readout_time[c] = 0.;
    p.camera_shutter_rolling[c] = false;
  }
  return p;
}

void seed(VioManager &manager, double physical_time = 100.) {
  Eigen::Matrix<double, 17, 1> x = Eigen::Matrix<double, 17, 1>::Zero();
  x(0) = physical_time - manager.get_state()->cam_imu_dt_ref();
  x(4) = 1.;
  x(8) = .2;
  manager.initialize_with_gt(x);
}

CameraData frame(double raw, std::vector<int> cameras) {
  CameraData message;
  message.timestamp = raw;
  message.sensor_ids = std::move(cameras);
  for (size_t i = 0; i < message.sensor_ids.size(); ++i) {
    message.images.push_back(cv::Mat::zeros(240, 320, CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(240, 320, CV_8UC1));
  }
  return message;
}

void group(VioManager &manager, double physical_time) {
  for (int c = 0; c < 2; ++c)
    manager.feed_measurement_camera(frame(physical_time - manager.get_state()->cam_imu_dt(c), {c}));
}

// Binary-exact clock grid makes equality/grouping checks independent of decimal rounding.
std::vector<ImuData> samples(double first, double last) {
  std::vector<ImuData> out;
  const int n = static_cast<int>(std::llround((last - first) * 256.));
  for (int i = 0; i <= n; ++i) {
    ImuData z;
    z.timestamp = first + i / 256.;
    z.wm.setZero();
    z.am << 0, 0, 9.81;
    out.push_back(z);
  }
  return out;
}

struct Event {
  int camera;
  double raw, physical;
  bool processed;
};
void record(VioManager &manager, std::vector<Event> &events) {
  manager.set_camera_processed_callback([&](const CameraData &msg, bool processed) {
    for (int c : msg.sensor_ids)
      events.push_back({c, msg.timestamp, manager.get_state()->imu_endpoint(), processed});
    return true;
  });
}

bool equal_state(const std::shared_ptr<State> &a, const std::shared_ptr<State> &b) {
  if (a->_timestamp != b->_timestamp || a->imu_endpoint() != b->imu_endpoint() ||
      !near(a->_imu->value(), b->_imu->value()) || !near(a->_imu->fej(), b->_imu->fej()) ||
      !near(StateHelper::get_full_covariance(a), StateHelper::get_full_covariance(b)) ||
      a->_exposure_poses.size() != b->_exposure_poses.size())
    return false;
  for (size_t i = 0; i < a->_exposure_poses.size(); ++i) {
    const auto &x = a->_exposure_poses[i], &y = b->_exposure_poses[i];
    if (x.camera_id != y.camera_id || x.raw_time != y.raw_time || x.imu_time != y.imu_time ||
        x.pose->id() != y.pose->id() || !near(x.pose->value(), y.pose->value()) || !near(x.pose->fej(), y.pose->fej()))
      return false;
  }
  return true;
}

void same_endpoint_covariance() {
  auto p = options();
  VioManager manager(p);
  seed(manager);
  std::vector<Event> events;
  record(manager, events);
  auto expected = StateHelper::clone_state(manager.get_state());
  Propagator oracle(p.imu_noises, p.gravity_mag);
  const auto imu = samples(99.96875, 100.0703125);
  for (const auto &z : imu)
    oracle.feed_imu(z);
  constexpr double end = 100.0625;
  Propagator::EndpointKinematics k;
  const bool propagated = oracle.propagate_to_imu(expected, end, end - p.camera_imu_dt.at(0), k);
  check(propagated, "independent explicit endpoint propagation succeeds");
  if (propagated)
    for (size_t c = 0; c < 2; ++c)
      StateHelper::augment_pose_view(expected, c, k.omega);
  group(manager, end);
  manager.feed_measurement_batch_imu(imu);
  const auto state = manager.get_state();
  const double raw0 = end - p.camera_imu_dt.at(0), raw1 = end - p.camera_imu_dt.at(1);
  auto a = state->find_pose(0, raw0), b = state->find_pose(1, raw1);
  check(events.size() == 2 && events[0].processed && events[1].processed &&
            events[0].physical == end && events[1].physical == end,
        "different raw stamps at one physical endpoint complete together");
  check(a && b && a != b && !state->find_pose(0, raw1) && !state->find_pose(1, raw0) &&
            near(a->value(), b->value()),
        "camera-owned raw keys resolve distinct stochastic views with equal physical pose");
  check(near(StateHelper::get_full_covariance(state), StateHelper::get_full_covariance(expected)) &&
            near(state->_imu->value(), expected->_imu->value()),
        "public caller matches one propagation plus both full owner-clock augmentations");
  check(state->clone_count() == 2 && state->_clones_IMU.empty() && state->_epoch_bridges.empty() &&
            state->_epoch_residuals.empty() && manager.get_camera_buffer()->count_released() == 1 &&
            manager.get_camera_buffer()->count_physical_views() == 2,
        "physical group adds two views and no legacy clone or deterministic bridge");
}

void physical_order_and_split() {
  auto p = options(4, .125, 0.);
  VioManager manager(p);
  seed(manager);
  std::vector<Event> events;
  record(manager, events);
  manager.feed_measurement_camera(frame(100.125, {0}));    // physical 100.25
  manager.feed_measurement_camera(frame(100.1875, {1}));   // physical 100.1875
  manager.feed_measurement_camera(frame(100.3125, {0, 1})); // two physical endpoints
  manager.feed_measurement_batch_imu(samples(99.96875, 100.4453125));
  check(events.size() == 4 && events[0].processed && events[1].processed &&
            events[0].camera == 1 && events[1].camera == 0 &&
            events[0].raw > events[1].raw && events[0].physical < events[1].physical,
        "manager processes physical order when camera raw order is reversed");
  check(events.size() == 4 && events[2].processed && events[3].processed &&
            events[2].raw == events[3].raw && events[2].camera == 1 &&
            events[3].camera == 0 && events[2].physical == 100.3125 && events[3].physical == 100.4375,
        "one packed image message splits by owner clock without changing either raw key");
  auto a = manager.get_state()->find_pose(0, 100.3125), b = manager.get_state()->find_pose(1, 100.3125);
  std::printf("PACKED_OWNER_DIAGNOSTIC a=%d b=%d dx=%.17g views=%llu\n", bool(a), bool(b),
              a && b ? (a->pos() - b->pos()).x() : 0.,
              static_cast<unsigned long long>(manager.get_camera_buffer()->count_physical_views()));
  for (const auto &event : events)
    std::printf("PACKED_EVENT camera=%d raw=%.17g physical=%.17g processed=%d\n", event.camera, event.raw,
                event.physical, event.processed);
  check(a && b && std::abs((a->pos() - b->pos()).x() - .025) < 1e-12 &&
            manager.get_camera_buffer()->count_physical_views() == 4,
        "equal raw keys retain camera-specific physical poses and consume four views exactly once");
}

void bounded_history() {
  auto p = options();
  VioManager manager(p);
  seed(manager);
  const int base_dimension = StateHelper::get_full_covariance(manager.get_state()).rows();
  const size_t cap = manager.get_state()->_options.max_pose_clones();
  double next_imu = 99.96875;
  for (int i = 1; i <= 12; ++i) {
    const double end = 100. + i / 16.;
    group(manager, end);
    manager.feed_measurement_batch_imu(samples(next_imu, end + .0078125));
    next_imu = end + .01171875;
    const auto state = manager.get_state();
    const Eigen::MatrixXd P = StateHelper::get_full_covariance(state);
    check(state->clone_count() == std::min<size_t>(2 * i, cap) &&
              P.rows() == base_dimension + 6 * static_cast<int>(state->clone_count()) && near(P, P.transpose()),
          "repeated complete groups keep every pose and covariance dimension within the finite cap");
    std::set<std::pair<size_t, double>> keys;
    bool registry_ok = true;
    double previous = -1.;
    for (const auto &view : state->_exposure_poses) {
      registry_ok &= keys.emplace(view.camera_id, view.raw_time).second && view.imu_time >= previous &&
                     state->find_pose(view.camera_id, view.raw_time) == view.pose;
      previous = view.imu_time;
    }
    for (int c = 0; c < 2; ++c) {
      registry_ok &= bool(state->find_pose(c, end - p.camera_imu_dt.at(c)));
      for (int old = 1; old <= i - static_cast<int>(cap / 2); ++old)
        registry_ok &= !state->find_pose(c, 100. + old / 16. - p.camera_imu_dt.at(c));
    }
    check(registry_ok, "marginalization removes every retired camera alias and preserves both newest owners");
  }
  check(manager.get_camera_buffer()->count_physical_views() == 24 &&
            manager.get_state()->_options.max_clone_size == 2 && cap == 4,
        "bounded camera history neither loses views nor expands configured per-view track length");
}

void coverage_then_continuation() {
  auto p = options();
  VioManager delayed(p), fresh(p);
  seed(delayed);
  seed(fresh);
  std::vector<Event> events;
  record(delayed, events);
  delayed.feed_measurement_batch_imu(samples(99.96875, 100.03125));
  const auto before = StateHelper::clone_state(delayed.get_state());
  group(delayed, 100.0625);
  delayed.feed_measurement_batch_imu(samples(100.03515625, 100.046875));
  check(events.empty() && equal_state(delayed.get_state(), before),
        "insufficient physical IMU endpoint coverage leaves queued views and filter state untouched");
  delayed.feed_measurement_batch_imu(samples(100.05078125, 100.0703125));
  group(fresh, 100.0625);
  fresh.feed_measurement_batch_imu(samples(99.96875, 100.0703125));
  check(events.size() == 2 && equal_state(delayed.get_state(), fresh.get_state()),
        "covered continuation consumes pending group once and matches a fresh uninterrupted caller");
}

void callback_snapshot_and_reset() {
  auto p = options();
  VioManager manager(p);
  seed(manager);
  std::vector<Event> events;
  std::shared_ptr<VioManager::Snapshot> first;
  manager.set_camera_processed_callback([&](const CameraData &msg, bool processed) {
    for (int c : msg.sensor_ids)
      events.push_back({c, msg.timestamp, manager.get_state()->imu_endpoint(), processed});
    if (processed && !first) {
      first = manager.snapshot();
      return false;
    }
    return true;
  });
  group(manager, 100.0625);
  group(manager, 100.125);
  manager.feed_measurement_batch_imu(samples(99.96875, 100.1328125));
  check(first && events.size() == 2 && events[0].processed && events[1].processed &&
            manager.get_state()->imu_endpoint() == 100.0625 && first->state->clone_count() == 2,
        "a pause requested by first callback finishes its complete group and defers the next group");
  check(first && first->tracked_camera_times.size() == 2 &&
            first->tracked_camera_times[0] == 100.0625 - p.camera_imu_dt.at(0) &&
            first->tracked_camera_times[1] == 100.0625 - p.camera_imu_dt.at(1),
        "first callback snapshot already contains both camera-owned raw replay cursors");
  if (!first || first->state->clone_count() != 2)
    return;
  const auto trigger = samples(100.13671875, 100.13671875);
  manager.feed_measurement_batch_imu(trigger);
  const auto completed = StateHelper::clone_state(manager.get_state());
  check(events.size() == 4 && completed->clone_count() == 4 && completed->imu_endpoint() == 100.125,
        "subsequent IMU feed resumes the deferred physical group once");
  group(manager, 100.1875); // not covered; restore must dispose both pending components
  manager.restore(first, {});
  check(events.size() == 6 && !events[4].processed && !events[5].processed &&
            equal_state(manager.get_state(), first->state) &&
            manager.get_state()->_exposure_poses.front().pose != first->state->_exposure_poses.front().pose,
        "restore disposes queued cameras and independently restores physical owners and covariance");
  // Resume each camera from its own raw cursor. A scalar raw cutoff would skip
  // or replay a different camera when the offsets or physical order differ.
  for (int c = 0; c < 2; ++c) {
    const double raw = 100.125 - p.camera_imu_dt.at(c);
    check(raw > first->tracked_camera_times.at(c), "replay continuation advances its own camera cursor");
    manager.feed_measurement_camera(frame(raw, {c}));
  }
  manager.feed_measurement_batch_imu(trigger);
  check(events.size() == 8 && equal_state(manager.get_state(), completed),
        "snapshot branch repeats the same physical endpoint, owner poses and full joint covariance");
  auto propagator = manager.get_propagator();
  for (int reset = 0; reset < 2; ++reset) {
    manager.clear_camera_buffers();
    manager.soft_reset();
    check(!manager.initialized() && manager.get_state()->clone_count() == 0 &&
              manager.get_propagator() == propagator && manager.get_state()->_options.max_clone_size == 2 &&
              manager.get_state()->_options.max_pose_clones() == 4,
          "soft reset empties camera owners without replacing IMU history or multiplying pose capacity");
    const double start = 101. + reset;
    seed(manager, start);
    VioManager fresh(p);
    seed(fresh, start);
    const auto imu = samples(start - .03125, start + .0703125);
    group(manager, start + .0625);
    group(fresh, start + .0625);
    manager.feed_measurement_batch_imu(imu);
    fresh.feed_measurement_batch_imu(imu);
    check(equal_state(manager.get_state(), fresh.get_state()),
          "new navigation episode continues like a fresh manager at the same physical seed");
  }
}

void configured_temporal_gate() {
  auto p = options();
  p.state_options.dt_calib_gate = true;
  VioManager manager(p);
  seed(manager);
  std::vector<Event> events;
  record(manager, events);
  group(manager, 100.0625);
  manager.feed_measurement_batch_imu(samples(99.96875, 100.0703125));
  check(manager.get_state()->_options.dt_calib_gate && events.size() == 2 &&
            events[0].processed && events[1].processed && manager.get_state()->clone_count() == 2,
        "configured physical manager with temporal calibration gate processes a covered group");
}

std::shared_ptr<ov_type::Landmark> add_landmark(const std::shared_ptr<State> &state, size_t id, int camera, double raw,
                                              ov_type::LandmarkRepresentation::Representation representation) {
  const int dimension = representation == ov_type::LandmarkRepresentation::ANCHORED_INVERSE_DEPTH_SINGLE ? 1 : 3;
  auto landmark = std::make_shared<ov_type::Landmark>(dimension);
  landmark->_featid = id;
  landmark->_unique_camera_id = camera;
  landmark->_anchor_cam_id = camera;
  landmark->_anchor_clone_timestamp = raw;
  landmark->_feat_representation = representation;
  landmark->set_from_xyz(Eigen::Vector3d(.3, .1, 4.), false);
  landmark->set_from_xyz(Eigen::Vector3d(.31, .09, 4.02), true);
  StateHelper::initialize_invertible(state, landmark, {state->_imu}, Eigen::MatrixXd::Zero(dimension, 15),
      Eigen::MatrixXd::Identity(dimension, dimension), .01 * Eigen::MatrixXd::Identity(dimension, dimension),
      Eigen::VectorXd::Zero(dimension));
  state->_features_SLAM[id] = landmark;
  return landmark;
}

Eigen::Vector3d landmark_world(const std::shared_ptr<State> &state, const std::shared_ptr<ov_type::Landmark> &landmark, bool fej) {
  if (!ov_type::LandmarkRepresentation::is_relative_representation(landmark->_feat_representation))
    return landmark->get_xyz(fej);
  const auto anchor = state->pose_for_camera(landmark->_anchor_cam_id, landmark->_anchor_clone_timestamp);
  const auto extrinsic = state->_calib_IMUtoCAM.at(landmark->_anchor_cam_id);
  return (fej ? anchor->Rot_fej() : anchor->Rot()).transpose() * extrinsic->Rot().transpose() *
      (landmark->get_xyz(fej) - extrinsic->pos()) + (fej ? anchor->pos_fej() : anchor->pos());
}

void aruco_anchor_lifetime() {
  using Rep = ov_type::LandmarkRepresentation;
  for (bool fej : {false, true})
    for (auto representation : {Rep::ANCHORED_3D, Rep::ANCHORED_FULL_INVERSE_DEPTH,
                                Rep::ANCHORED_MSCKF_INVERSE_DEPTH, Rep::ANCHORED_INVERSE_DEPTH_SINGLE}) {
      auto p = options(1);
      p.state_options.do_fej = fej;
      p.state_options.max_aruco_features = 10;
      p.state_options.feat_rep_aruco = representation;
      p.use_aruco = true;
      VioManager manager(p);
      seed(manager);
      group(manager, 100.0625);
      manager.feed_measurement_batch_imu(samples(99.96875, 100.0703125));
      auto state = manager.get_state();
      const double raw0 = 100.0625 - p.camera_imu_dt.at(0), raw1 = 100.0625 - p.camera_imu_dt.at(1);
      auto orphan = add_landmark(state, 1, 0, raw0, representation);
      auto valid = add_landmark(state, 2, 1, raw1, representation);
      auto global = add_landmark(state, 3, 0, raw0, Rep::GLOBAL_3D);
      add_landmark(state, 100, 0, raw0, representation); // ordinary SLAM control at the same retiring owner
      valid->should_marg = global->should_marg = true;   // normal tracking loss must not evict valid tags
      const Eigen::Vector3d world = landmark_world(state, valid, false), frozen = landmark_world(state, valid, true);
      std::vector<std::shared_ptr<ov_type::Type>> order{state->_imu, state->cam_imu_dt_var(0), state->cam_imu_dt_var(1)};
      for (const auto &view : state->_exposure_poses) order.push_back(view.pose);
      for (const auto &entry : state->_features_SLAM) order.push_back(entry.second);
      std::sort(order.begin(), order.end(), [](const auto &a, const auto &b) { return a->id() < b->id(); });
      const int n = state->max_covariance_size();
      Eigen::MatrixXd L = Eigen::MatrixXd::Identity(n, n);
      for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) L(i, j) = .02 * std::sin(.4 * i + .7 * j);
      StateHelper::set_initial_covariance(state, .001 * L * L.transpose(), order);

      // Independent retained-index marginal after the same endpoint propagation.
      // There are no detected image features, so no visual update can mask an
      // incorrect lifetime or accidentally repair the full covariance ordering.
      auto expected = StateHelper::clone_state(state);
      Propagator oracle(p.imu_noises, p.gravity_mag);
      oracle.restore(manager.get_propagator()->capture());
      const auto imu = samples(100.07421875, 100.1328125);
      for (const auto &z : imu) oracle.feed_imu(z);
      Propagator::EndpointKinematics rates;
      check(oracle.propagate_to_imu(expected, 100.125, 100.125 - p.camera_imu_dt.at(0), rates),
            "ArUco lifetime covariance oracle reaches next physical owner");
      StateHelper::augment_pose_view(expected, 1, rates.omega);
      const Eigen::MatrixXd full = StateHelper::get_full_covariance(expected);
      std::vector<bool> retained(full.rows(), true);
      for (const auto &variable : std::vector<std::shared_ptr<ov_type::Type>>{
               expected->_exposure_poses.front().pose, expected->_features_SLAM.at(1), expected->_features_SLAM.at(100)})
        for (int i = variable->id(); i < variable->id() + variable->size(); ++i) retained[i] = false;
      std::vector<int> indices;
      for (int i = 0; i < full.rows(); ++i) if (retained[i]) indices.push_back(i);
      Eigen::MatrixXd marginal(indices.size(), indices.size());
      for (size_t i = 0; i < indices.size(); ++i) for (size_t j = 0; j < indices.size(); ++j)
        marginal(i, j) = full(indices[i], indices[j]);

      manager.feed_measurement_camera(frame(100.125 - p.camera_imu_dt.at(1), {1}));
      manager.feed_measurement_batch_imu(imu);
      state = manager.get_state();
      check(!state->find_pose(0, raw0) && !state->_features_SLAM.count(1) && !state->_features_SLAM.count(100) && orphan->id() == -1,
            "actual manager removes anchored tag and ordinary landmark before their final camera owner retires");
      check(state->_features_SLAM.count(2) && state->_features_SLAM.count(3) && valid->should_marg && global->should_marg &&
                near(landmark_world(state, valid, false), world) && near(landmark_world(state, valid, true), frozen),
            "tracking-loss flags preserve valid anchored and global persistent ArUco tags");
      check(near(StateHelper::get_full_covariance(state), marginal) && near(state->_imu->value(), expected->_imu->value()),
            "orphan and owner removal preserves the exact retained joint covariance with nonzero cross blocks");
      bool export_ok = false;
      try { export_ok = manager.get_features_ARUCO().size() == 2; }
      catch (const std::out_of_range &) {}
      check(export_ok, "public ArUco export has no dangling anchor after real caller marginalization");
      if (state->_features_SLAM.count(1))
        continue; // the negative control is already proven; do not use its dangling pose
      manager.feed_measurement_camera(frame(100.1875 - p.camera_imu_dt.at(1), {1}));
      manager.feed_measurement_batch_imu(samples(100.13671875, 100.1953125));
      check(state->_features_SLAM.count(2) && state->_features_SLAM.count(3) && valid->_anchor_clone_timestamp != raw1 &&
                near(landmark_world(state, valid, false), world) && near(landmark_world(state, valid, true), frozen) &&
                manager.get_features_ARUCO().size() == 2,
            "valid persistent tag reanchors and exports correctly as its own camera continues");
    }
}
} // namespace

int main(int argc, char **argv) {
  ov_core::Printer::setPrintLevel("ERROR");
  if (!(argc == 2 && std::strcmp(argv[1], "--orphan-only") == 0)) {
    same_endpoint_covariance();
    physical_order_and_split();
    bounded_history();
    coverage_then_continuation();
    callback_snapshot_and_reset();
    configured_temporal_gate();
  }
  aruco_anchor_lifetime();
  std::printf("PHYSICAL_MANAGER %s checks=%d failures=%d\n", failures ? "FAIL" : "PASS", checks, failures);
  return failures ? 1 : 0;
}
