/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/VioManager.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "track/TrackSIM.h"

using namespace ov_msckf;
using ov_core::CameraData;
using ov_core::FeatureObservations;
using ov_core::ImuData;

namespace {
int checks = 0, failures = 0;
void check(bool passed, const char *name) {
  ++checks;
  failures += !passed;
  std::printf("[%s] %s\n", passed ? "PASS" : "FAIL", name);
}
bool near(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b, double tolerance = 1e-12) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         (a.size() == 0 || (a - b).cwiseAbs().maxCoeff() < tolerance);
}
VioManagerOptions options(double dt0 = .125, double dt1 = 0.) {
  VioManagerOptions p;
  p.state_options.num_cameras = 2;
  p.state_options.max_clone_size = 11;
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
  for (int camera_id = 0; camera_id < 2; ++camera_id) {
    auto camera = std::make_shared<ov_core::CamRadtan>(320, 240);
    Eigen::VectorXd intrinsic(8), extrinsic(7);
    intrinsic << 220, 230, 160, 120, -.08, .01, .002, -.003;
    extrinsic << 0, 0, 0, 1, .1 * camera_id, 0, 0;
    camera->set_value(intrinsic);
    p.camera_intrinsics[camera_id] = p.init_options.camera_intrinsics[camera_id] = camera;
    p.camera_extrinsics[camera_id] = p.init_options.camera_extrinsics[camera_id] = extrinsic;
    p.camera_imu_dt[camera_id] = camera_id == 0 ? dt0 : dt1;
    p.camera_readout_time[camera_id] = 0.;
    p.camera_shutter_rolling[camera_id] = false;
  }
  return p;
}
void seed(VioManager &manager) {
  manager.prepare_observation_replay();
  Eigen::Matrix<double, 17, 1> x = Eigen::Matrix<double, 17, 1>::Zero();
  x(0) = 100.;
  x(4) = 1.;
  x(8) = .2;
  manager.initialize_with_gt_imu(x);
}
std::vector<ImuData> samples(double last = 100.75) {
  std::vector<ImuData> out;
  for (int i = -8; 100. + i / 256. <= last; ++i) {
    ImuData z;
    z.timestamp = 100. + i / 256.;
    z.wm.setZero();
    z.am << 0, 0, 9.81;
    out.push_back(z);
  }
  return out;
}
FeatureObservations points(size_t id) {
  Eigen::VectorXf uv(2);
  uv << 211.25f, 89.75f;
  return {{id, uv}};
}
void queue(VioManager &manager, double raw, int camera, size_t id) {
  manager.feed_measurement_simulation_queued(raw, {camera}, {points(id)});
}
std::shared_ptr<ov_core::Feature> feature(VioManager &manager, size_t raw_id) {
  // TrackSIM reserves 4 * max_aruco_features + 1 ids, exactly as its legacy API.
  return manager.get_track_feats()->get_feature_database()->get_feature(raw_id + 1);
}
struct Event { int camera; double raw, physical; bool processed; };
void record(VioManager &manager, std::vector<Event> &events) {
  manager.set_camera_processed_callback([&](const CameraData &message, bool processed) {
    for (int camera : message.sensor_ids)
      events.push_back({camera, message.timestamp, manager.get_state()->imu_endpoint(), processed});
    return true;
  });
}

void physical_order_oracle() {
  auto p = options();
  VioManager queued(p), oracle(p), old_direct(p);
  seed(queued); seed(oracle); seed(old_direct);
  std::vector<Event> events;
  record(queued, events);
  queue(queued, 100.125, 0, 10);   // arrives first, physical .250
  queue(queued, 100.1875, 1, 20); // arrives second, physical .1875
  queued.feed_measurement_simulation_queued(100.3125, {0, 1}, {points(30), points(40)});
  const auto imu = samples();
  queued.feed_measurement_batch_imu(imu);
  queued.finish_observation_replay();
  oracle.feed_measurement_batch_imu(imu);
  oracle.feed_measurement_simulation(100.1875, {1}, {points(20)});
  oracle.feed_measurement_simulation(100.125, {0}, {points(10)});
  oracle.feed_measurement_simulation(100.3125, {1}, {points(40)});
  oracle.feed_measurement_simulation(100.3125, {0}, {points(30)});
  check(events.size() == 4 && events[0].camera == 1 && events[0].raw == 100.1875 &&
            events[0].physical == 100.1875 && events[1].camera == 0 && events[1].physical == 100.25 &&
            events[2].camera == 1 && events[2].physical == 100.3125 &&
            events[3].camera == 0 && events[3].physical == 100.4375,
        "queued raw-order reversal and packed split follow actual physical exposure order");
  const auto a = queued.get_state(), b = oracle.get_state();
  check(near(a->_imu->value(), b->_imu->value()) && near(a->_imu->fej(), b->_imu->fej()) &&
            near(StateHelper::get_full_covariance(a), StateHelper::get_full_covariance(b)),
        "queued state and full covariance match independently physically ordered direct oracle");
  check(a->clone_count() == 4 && a->find_pose(0, 100.125) && a->find_pose(1, 100.1875) &&
            a->find_pose(0, 100.3125) && a->find_pose(1, 100.3125) &&
            !a->find_pose(1, 100.125) && !a->find_pose(0, 100.1875),
        "physical views retain exact camera/raw identities without aliasing");
  old_direct.feed_measurement_batch_imu(imu);
  old_direct.feed_measurement_simulation(100.125, {0}, {points(10)});
  old_direct.feed_measurement_simulation(100.1875, {1}, {points(20)});
  check(!old_direct.get_state()->find_pose(1, 100.1875) && old_direct.get_state()->find_pose(0, 100.125),
        "negative control: old arrival-ordered immediate replay loses the earlier physical owner");
}

void complete_group_and_calibration() {
  auto p = options();
  p.downsample_cameras = true; // observations already use the loaded camera's pixel coordinates
  VioManager manager(p);
  seed(manager);
  int callbacks = 0;
  bool complete = true, pixels = true, empty_images = true;
  manager.set_camera_processed_callback([&](const CameraData &message, bool processed) {
    if (!processed) { complete = false; return true; }
    ++callbacks;
    const auto snapshot = manager.snapshot();
    const auto f0 = feature(manager, 100), f1 = feature(manager, 200);
    complete &= f0 && f1 && snapshot->tracked_camera_times == std::vector<double>({100.125, 100.25}) &&
                manager.get_state()->find_pose(0, 100.125) && manager.get_state()->find_pose(1, 100.25);
    empty_images &= message.images.size() == 1 && message.images[0].empty() &&
                    message.observations.size() == 1 && message.observations[0]->size() == 1;
    for (int camera = 0; camera < 2; ++camera) {
      const auto f = camera == 0 ? f0 : f1;
      if (!f) { pixels = false; continue; }
      const cv::Point2f expected = manager.get_state()->_cam_intrinsics_cameras.at(camera)->undistort_cv({211.25f, 89.75f});
      pixels &= f->uvs.at(camera)[0](0) == 211.25f && f->uvs.at(camera)[0](1) == 89.75f &&
                std::abs(f->uvs_norm.at(camera)[0](0) - expected.x) < 1e-7 &&
                std::abs(f->uvs_norm.at(camera)[0](1) - expected.y) < 1e-7;
    }
    return true;
  });
  queue(manager, 100.125, 0, 100);
  queue(manager, 100.25, 1, 200);
  // Change the camera model after enqueue, before consumption: normalization must use this model.
  auto camera = manager.get_state()->_cam_intrinsics_cameras.at(1);
  Eigen::VectorXd changed = camera->get_value();
  changed(0) += 19.; changed(4) += .025;
  camera->set_value(changed);
  manager.feed_measurement_batch_imu(samples());
  manager.finish_observation_replay();
  check(callbacks == 2 && complete, "first physical-group callback/snapshot sees all cameras tracked and updated");
  check(pixels, "original distorted pixels survive and are undistorted at consumption using the active camera model");
  check(empty_images, "observation transport carries only empty image headers even with downsampling enabled");
}

void online_clock_and_eof() {
  auto p = options();
  VioManager manager(p);
  seed(manager);
  std::vector<Event> events;
  manager.set_camera_processed_callback([&](const CameraData &message, bool processed) {
    for (int camera : message.sensor_ids)
      events.push_back({camera, message.timestamp, manager.get_state()->imu_endpoint(), processed});
    if (processed && events.size() == 1) {
      Eigen::VectorXd dt(1); dt << .25;
      manager.get_state()->_calib_dt_CAMtoIMU_map.at(0)->set_value(dt);
    }
    return true;
  });
  queue(manager, 100.25, 0, 1);
  queue(manager, 100.3125, 1, 2);
  queue(manager, 100.375, 0, 3);
  queue(manager, 100.4375, 1, 4);
  manager.feed_measurement_batch_imu(samples());
  manager.finish_observation_replay();
  check(events.size() == 4 && events[0].camera == 1 && events[0].physical == 100.3125 &&
            events[1].camera == 1 && events[1].physical == 100.4375 &&
            events[2].camera == 0 && events[2].physical == 100.5 &&
            events[3].camera == 0 && events[3].physical == 100.625,
        "online clock changes reorder later groups without a sort-once schedule");
  const auto &views = manager.get_state()->_exposure_poses;
  check(views.size() == 4 && views.front().raw_time == 100.3125 && views.front().imu_time == 100.3125,
        "previously created owner endpoint stays immutable after online clock update");

  VioManager tail(p);
  seed(tail);
  std::vector<Event> tail_events;
  record(tail, tail_events);
  queue(tail, 101., 0, 500);
  tail.feed_measurement_batch_imu(samples(100.5));
  check(tail_events.empty(), "uncovered observation is held before EOF");
  tail.finish_observation_replay();
  check(tail_events.size() == 1 && !tail_events[0].processed && !feature(tail, 500) &&
            tail.get_state()->clone_count() == 0 && tail.get_state()->imu_endpoint() == 100.,
        "EOF disposes uncovered tail without tracking, propagation or extrapolation");
  queue(tail, 101.125, 1, 501);
  check(tail_events.size() == 2 && !tail_events.back().processed && tail.get_camera_buffer()->count_drop_finished() == 1,
        "closed replay rejects subsequent input through the normal disposal callback");
}

void input_contract_and_legacy() {
  auto p = options();
  VioManager late(p);
  const auto original_tracker = late.get_track_feats();
  late.feed_measurement_batch_imu(samples(100.03125));
  bool rejected = false;
  try { late.prepare_observation_replay(); } catch (const std::logic_error &) { rejected = true; }
  check(rejected && late.get_track_feats() == original_tracker,
        "late preparation is refused before replacing tracker/initializer with accumulated IMU");

  VioManager manager(p);
  seed(manager);
  rejected = false;
  CameraData image;
  image.timestamp = 100.125; image.sensor_ids = {0};
  image.images.resize(1); image.masks.resize(1);
  try { manager.feed_measurement_camera(image); } catch (const std::invalid_argument &) { rejected = true; }
  check(rejected && manager.get_camera_buffer()->count_pushed() == 0, "prepared replay rejects image/observation mode mixing");
  rejected = false;
  auto bad = points(5); bad[0].second(1) = std::numeric_limits<float>::quiet_NaN();
  try { manager.feed_measurement_simulation_queued(100.125, {0, 1}, {points(4), bad}); }
  catch (const std::invalid_argument &) { rejected = true; }
  check(rejected && manager.get_camera_buffer()->count_pushed() == 0,
        "whole queued bundle validates finite pixels before any camera is enqueued");
  rejected = false;
  try { manager.feed_measurement_simulation_queued(100.125, {0, 0}, {points(4), points(5)}); }
  catch (const std::invalid_argument &) { rejected = true; }
  check(rejected && manager.get_camera_buffer()->count_pushed() == 0, "repeated camera slots are rejected atomically");

  auto legacy_options = options(0., 0.);
  legacy_options.state_options.physical_camera_clones = false;
  VioManager direct(legacy_options), queued(legacy_options);
  seed(direct); seed(queued);
  const auto imu = samples(100.25);
  direct.feed_measurement_batch_imu(imu);
  direct.feed_measurement_simulation(100.125, {0, 1}, {points(1), points(2)});
  queued.feed_measurement_simulation_queued(100.125, {0, 1}, {points(1), points(2)});
  queued.feed_measurement_batch_imu(imu);
  queued.finish_observation_replay();
  check(near(direct.get_state()->_imu->value(), queued.get_state()->_imu->value()) &&
            near(StateHelper::get_full_covariance(direct.get_state()), StateHelper::get_full_covariance(queued.get_state())) &&
            direct.get_state()->clone_count() == queued.get_state()->clone_count(),
        "legacy synchronized queued path matches preserved immediate simulation state and covariance");
}

void forced_sync_dual_mono_replay() {
  auto p = options(0., 0.);
  p.state_options.physical_camera_clones = false;
  p.force_camera_sync = true;
  VioManager queued(p), oracle(p);
  seed(queued); seed(oracle);
  int callbacks = 0;
  bool complete = true;
  queued.set_camera_processed_callback([&](const CameraData &message, bool processed) {
    ++callbacks;
    complete &= processed && message.sensor_ids == std::vector<int>({0, 1}) &&
                message.timestamp == 100.125 && feature(queued, 10) && feature(queued, 20);
    return true;
  });
  queue(queued, 100.125, 1, 20);
  const auto imu = samples(100.25);
  queued.feed_measurement_batch_imu(imu);
  check(callbacks == 0, "forced-sync replay waits for the unseen reference camera even with zero stale interval");
  queue(queued, 100.125, 0, 10);
  queued.finish_observation_replay();
  oracle.feed_measurement_batch_imu(imu);
  oracle.feed_measurement_simulation(100.125, {0, 1}, {points(10), points(20)});
  check(callbacks == 1 && complete && !p.use_stereo,
        "forced-sync dual-mono observation replay processes both cameras in one complete callback");
  check(near(queued.get_state()->_imu->value(), oracle.get_state()->_imu->value()) &&
            near(StateHelper::get_full_covariance(queued.get_state()), StateHelper::get_full_covariance(oracle.get_state())),
        "forced-sync queued state and covariance match a simultaneous dual-mono measurement oracle");
}

void shared_feature_clock_ownership() {
  auto p = options(0., .125);
  VioManager manager(p);
  seed(manager);
  // A stereo/shared id is lost by camera 0 but remains current in camera 1. The latter's
  // raw clock is earlier, although these observations belong to the same physical group.
  // Continue beyond five clone views so the actual MSCKF feature-selection branch runs.
  for (int group = 1; group <= 3; ++group) {
    const double end = 100. + group / 8.;
    manager.feed_measurement_simulation_queued(end, {0},
        {group < 3 ? points(700) : FeatureObservations{}});
    queue(manager, end - .125, 1, 700);
  }
  manager.feed_measurement_batch_imu(samples());
  manager.finish_observation_replay();
  const auto kept = feature(manager, 700);
  check(kept && !kept->to_delete && kept->timestamps.count(1) &&
            kept->timestamps.at(1).back() == 100.25 && manager.get_state()->clone_count() == 6,
        "real MSCKF selection preserves shared feature current in an earlier raw camera clock");
}
} // namespace

int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  physical_order_oracle();
  complete_group_and_calibration();
  online_clock_and_eof();
  input_contract_and_legacy();
  forced_sync_dual_mono_replay();
  shared_feature_clock_ownership();
  std::printf("OBSERVATION_REPLAY_RESULT checks=%d failures=%d\n", checks, failures);
  return failures ? 1 : 0;
}
