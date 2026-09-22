/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 * A gated optimizer exercises the real manager's snapshot, ownership, import
 * and lifecycle paths without depending on a nonlinear solve's wall time.
 */
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
#include "core/VioManager.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "init/InertialInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Landmark.h"

namespace fixture {
std::mutex mutex;
std::condition_variable changed;
bool entered = false, release = false, block = true, throw_result = false, bad_covariance = false;
bool private_input = false, stable_input = false;
int calls = 0;
std::thread::id worker_id;
std::shared_ptr<ov_core::FeatureDatabase> live_database;
Eigen::Matrix<double, 16, 1> mean() {
  Eigen::Matrix<double, 16, 1> x = Eigen::Matrix<double, 16, 1>::Zero();
  x(3) = 1.; x(4) = .4; x(7) = .3; x(11) = .02;
  return x;
}
void reset(bool hold = true) {
  std::lock_guard<std::mutex> lock(mutex);
  entered = release = private_input = stable_input = throw_result = bad_covariance = false;
  block = hold; calls = 0;
}
void await_entry() {
  std::unique_lock<std::mutex> lock(mutex);
  if (!changed.wait_for(lock, std::chrono::seconds(10), [] { return entered; }))
    throw std::runtime_error("initializer did not enter");
}
void unblock() {
  std::lock_guard<std::mutex> lock(mutex);
  release = true;
  changed.notify_all();
}
}

// Interpose only the solver entry, retaining make_attempt and all production
// manager transaction code. Access from this member also verifies every input
// passed to a worker is independent of the producer's storage.
bool ov_init::InertialInitializer::initialize(
    double &timestamp, Eigen::MatrixXd &covariance,
    std::vector<std::shared_ptr<ov_type::Type>> &order, std::shared_ptr<ov_type::IMU> imu,
    std::map<double, std::shared_ptr<ov_type::PoseJPL>> &clones,
    std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> &features, bool) {
  const auto samples = *imu_data;
  const auto tracks = _db->clone_features();
  const auto calibration = params.camera_intrinsics.at(0)->get_value();
  const auto prior = reset_ctx->prior();
  {
    std::unique_lock<std::mutex> lock(fixture::mutex);
    ++fixture::calls;
    fixture::worker_id = std::this_thread::get_id();
    fixture::private_input = is_attempt && _db != fixture::live_database;
    fixture::entered = true;
    fixture::changed.notify_all();
    if (fixture::block)
      fixture::changed.wait(lock, [] { return fixture::release; });
  }
  bool unchanged = samples.size() == imu_data->size() && tracks.size() == _db->size() &&
                   calibration == params.camera_intrinsics.at(0)->get_value() &&
                   prior.bg == reset_ctx->prior().bg && prior.valid == reset_ctx->prior().valid;
  for (size_t i = 0; unchanged && i < samples.size(); ++i)
    unchanged = samples[i].timestamp == imu_data->at(i).timestamp &&
                samples[i].wm == imu_data->at(i).wm && samples[i].am == imu_data->at(i).am;
  for (const auto &track : tracks) {
    auto current = _db->get_feature(track.first);
    unchanged = unchanged && current && current->timestamps == track.second->timestamps;
    if (current)
      for (const auto &camera : track.second->uvs)
        unchanged = unchanged && camera.second == current->uvs.at(camera.first) &&
                    track.second->uvs_norm.at(camera.first) == current->uvs_norm.at(camera.first);
  }
  fixture::stable_input = unchanged;
  if (fixture::throw_result)
    throw std::runtime_error("injected optimizer exception");
  timestamp = 12.;
  imu->set_value(fixture::mean()); imu->set_fej(fixture::mean());
  order = {imu};
  for (double time : {11.875, 12.}) {
    auto pose = std::make_shared<ov_type::PoseJPL>();
    pose->set_value(fixture::mean().head<7>()); pose->set_fej(pose->value());
    clones.emplace(time, pose); order.push_back(pose);
  }
  features.emplace(100, std::make_shared<ov_type::Landmark>(3));
  covariance = .01 * Eigen::MatrixXd::Identity(27, 27);
  covariance.bottomRows(6) = covariance.topRows(6).eval();
  covariance.rightCols(6) = covariance.leftCols(6).eval();
  if (fixture::bad_covariance)
    covariance(0, 0) = -1.;
  return true;
}

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char *why) {
  ++checks;
  if (!ok) { ++failures; std::printf("FAIL: %s\n", why); }
}
template <class A, class B> bool same(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
      std::memcmp(a.derived().data(), b.derived().data(), sizeof(double) * a.size()) == 0;
}
ov_msckf::VioManagerOptions options(bool async = true) {
  ov_msckf::VioManagerOptions p;
  p.state_options.num_cameras = p.init_options.num_cameras = 1;
  p.state_options.max_clone_size = 4;
  p.state_options.max_slam_features = p.state_options.max_aruco_features = 0;
  p.epoch_mode = p.async_frame_clones = false;
  p.use_stereo = p.use_aruco = p.use_gpu = p.try_zupt = false;
  p.use_multi_threading_pubs = false; p.use_multi_threading_subs = async;
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
  explicit Manager(ov_msckf::VioManagerOptions p = options()) : VioManager(p) {}
  bool attempt(double time = 12.) {
    ov_core::CameraData message; message.timestamp = time; message.sensor_ids = {0};
    return try_to_initialize(message);
  }
  void prime() {
    fixture::live_database = trackFEATS->get_feature_database();
    for (double time : {11.875, 12., 12.02}) {
      fixture::live_database->update_feature(100, time, 0, 1, 2, 3, 4);
      fixture::live_database->update_feature(101, time, 0, 5, 6, 7, 8);
    }
    for (int i = 0; i <= 600; ++i) {
      ov_core::ImuData sample; sample.timestamp = 10. + .005 * i;
      sample.wm.setZero(); sample.am << 0, 0, 9.81;
      propagator->feed_imu(sample); initializer->feed_imu(sample);
    }
    warmstart_next_init.store(true);
  }
  void finish_worker_only() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!initialization_completed()) {
      if (std::chrono::steady_clock::now() > deadline)
        throw std::runtime_error("initializer did not finish");
      std::this_thread::yield();
    }
  }
  bool success() const { return thread_init_success.load(); }
  bool pending() const { return thread_init_running.load() || initialization_attempt; }
  bool worker_exists() const { return initialization_thread.joinable(); }
  std::thread::id worker() const { return initialization_thread.get_id(); }
  size_t queued() const { return camera_queue_init.size() + camera_queue_init_owners.size(); }
  void mutate_inputs() {
    for (int i = 0; i < 256; ++i) {
      ov_core::ImuData sample; sample.timestamp = 13. + .001 * i;
      sample.wm.setConstant(.1); sample.am << 0, 0, 9.81;
      initializer->feed_imu(sample, 11.);
      initializer->feed_imu_batch({sample}, 11.);
      fixture::live_database->update_feature(100, 12.1 + .001 * i, 0, 9, 10, 11, 12);
    }
    auto values = state->_cam_intrinsics_cameras.at(0)->get_value(); values(0) += 1.;
    state->_cam_intrinsics_cameras.at(0)->set_value(values);
    ov_init::ResetBiasPrior prior; prior.valid = true; prior.bg.setConstant(.05);
    initializer->set_reset_prior(prior);
  }
};

void overlap() {
  fixture::reset();
  Manager manager; manager.prime();
  const auto state = manager.get_state();
  const auto mean = state->_imu->value(), fej = state->_imu->fej();
  const auto covariance = ov_msckf::StateHelper::get_full_covariance(state);
  const auto endpoint = state->_imu_endpoint;
  check(!manager.attempt(), "async launch remains an initialization frame");
  fixture::await_entry();
  manager.mutate_inputs();
  for (int i = 0; i < 500; ++i)
    manager.attempt(12.03 + .0002 * i);
  check(manager.queued() == 4 && fixture::calls == 1, "one pending worker and bounded catch-up window during a long solve");
  check(same(mean, state->_imu->value()) && same(fej, state->_imu->fej()) &&
        same(covariance, ov_msckf::StateHelper::get_full_covariance(state)) && state->_imu_endpoint == endpoint,
        "blocked optimizer cannot change live navigation, FEJ, covariance or endpoint");
  fixture::unblock(); manager.finish_worker_only();
  check(fixture::private_input && fixture::stable_input, "worker owns immutable feature, raw timestamp, IMU, calibration and prior snapshots");
  check(!manager.success() && same(mean, state->_imu->value()) && same(covariance, ov_msckf::StateHelper::get_full_covariance(state)),
        "worker completion alone never imports or propagates live state");
  check(manager.attempt(12.2) && manager.success() && !manager.pending(), "consumer validates and imports completed output exactly once");
  check(state->_imu_endpoint == 12.03 + .0002 * 499 + .125, "consumer catches up through retained camera endpoints");
  const auto used = fixture::live_database->get_feature(100), unused = fixture::live_database->get_feature(101);
  check(used && std::find(used->timestamps.at(0).begin(), used->timestamps.at(0).end(), 12.) == used->timestamps.at(0).end() &&
        std::find(used->timestamps.at(0).begin(), used->timestamps.at(0).end(), 12.02) != used->timestamps.at(0).end() &&
        unused && std::find(unused->timestamps.at(0).begin(), unused->timestamps.at(0).end(), 12.) != unused->timestamps.at(0).end(),
        "warm import consumes exact used rows and preserves unused and later observations");
}

void invalid_and_retry(bool exception) {
  fixture::reset();
  Manager manager; manager.prime();
  auto state = manager.get_state();
  const auto mean = state->_imu->value(), fej = state->_imu->fej();
  const auto covariance = ov_msckf::StateHelper::get_full_covariance(state);
  fixture::throw_result = exception; fixture::bad_covariance = !exception;
  manager.attempt(); fixture::await_entry(); fixture::unblock(); manager.finish_worker_only();
  const auto worker = manager.worker();
  check(!manager.attempt(12.1) && !manager.success() && !manager.pending(), "invalid/throwing async result is reaped without acceptance");
  check(manager.worker_exists(), "failed attempts retain one idle worker for retry");
  check(same(mean, state->_imu->value()) && same(fej, state->_imu->fej()) &&
        same(covariance, ov_msckf::StateHelper::get_full_covariance(state)) && state->_clones_IMU.empty(),
        "async rejection is atomic for navigation, FEJ and covariance ownership");
  fixture::reset();
  manager.attempt(); fixture::await_entry();
  check(manager.worker() == worker, "retry reuses the owned worker without creating a per-frame thread");
  fixture::unblock(); manager.finish_worker_only();
  check(manager.attempt(12.1), "a refused async attempt can retry successfully");
}

void lifecycle(int operation) {
  fixture::reset();
  auto manager = std::make_unique<Manager>(); manager->prime();
  const auto before = manager->get_state();
  const auto mean = before->_imu->value();
  const auto snapshot = manager->snapshot();
  manager->attempt(); fixture::await_entry();
  std::promise<void> started;
  auto started_future = started.get_future();
  auto action = std::async(std::launch::async, [&] {
    started.set_value();
    if (operation == 0) manager->soft_reset();
    if (operation == 1) manager->restore(snapshot, {});
    if (operation == 2) manager.reset();
    if (operation == 3) {
      Eigen::Matrix<double, 17, 1> truth; truth(0) = 20.; truth.tail<16>() = fixture::mean();
      manager->initialize_with_gt_imu(truth);
    }
  });
  started_future.wait();
  check(action.wait_for(std::chrono::milliseconds(25)) == std::future_status::timeout,
        "reset/restore/destructor/GT waits for its owned blocked worker");
  fixture::unblock(); action.get();
  if (operation != 3)
    check(same(mean, before->_imu->value()), "abandoned attempt cannot mutate the old state during lifecycle transition");
  if (manager) {
    check(!manager->pending() && !manager->worker_exists(), "lifecycle transition joins and discards every pending result");
    if (operation == 3) {
      check(manager->get_state()->imu_endpoint() == 20., "GT endpoint cannot be overwritten by a stale initialization result");
    } else {
      check(!manager->success() && manager->get_state()->_clones_IMU.empty(), "reset/restore never inherits old initialization success");
      fixture::reset(); manager->attempt(); fixture::await_entry(); fixture::unblock(); manager->finish_worker_only();
      check(manager->attempt(), "new lifecycle episode can initialize independently");
    }
  }
}

void synchronous() {
  fixture::reset(false);
  Manager manager(options(false)); manager.prime();
  const auto caller = std::this_thread::get_id();
  check(!manager.attempt() && manager.success() && !manager.pending() && !manager.worker_exists(),
        "synchronous initialization preserves trigger-frame contract without creating a worker");
  check(fixture::worker_id == caller && fixture::private_input, "synchronous mode runs inline on a private snapshot");
}
}
int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  overlap(); invalid_and_retry(false); invalid_and_retry(true);
  for (int operation = 0; operation < 4; ++operation) lifecycle(operation);
  synchronous();
  std::printf("INITIALIZATION_ASYNC %s checks=%d failures=%d\n", failures ? "FAIL" : "PASS", checks, failures);
  return failures ? 1 : 0;
}
