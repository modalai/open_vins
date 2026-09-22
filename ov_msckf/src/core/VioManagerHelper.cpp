/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
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

#include "VioManager.h"

#include <cmath>
#include <condition_variable>
#include <limits>

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "types/LandmarkRepresentation.h"
#include "utils/ChronoProf.h"
#include "utils/print.h"

#include "init/InertialInitializer.h"
#include "init/InitializerCameraClock.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/LegacyExposure.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

void VioManager::initialize_with_gt(Eigen::Matrix<double, 17, 1> imustate) {
  const double imu_time = imustate(0) + state->cam_imu_dt_ref();
  initialize_with_gt_at_endpoint(imustate, imu_time);
}

void VioManager::initialize_with_gt_imu(Eigen::Matrix<double, 17, 1> imustate) {
  const double imu_time = imustate(0);
  imustate(0) = imu_time - state->cam_imu_dt_ref();
  initialize_with_gt_at_endpoint(imustate, imu_time);
}

void VioManager::initialize_with_gt_at_endpoint(Eigen::Matrix<double, 17, 1> imustate, double imu_time) {
  stop_initialization();

  // Initialize the system
  state->_imu->set_value(imustate.block(1, 0, 16, 1));
  state->_imu->set_fej(imustate.block(1, 0, 16, 1));

  // Fix the global yaw and position gauge freedoms
  // TODO: Why does this break out simulation consistency metrics?
  std::vector<std::shared_ptr<ov_type::Type>> order = {state->_imu};
  Eigen::MatrixXd Cov = std::pow(0.02, 2) * Eigen::MatrixXd::Identity(state->_imu->size(), state->_imu->size());
  Cov.block(0, 0, 3, 3) = std::pow(0.017, 2) * Eigen::Matrix3d::Identity(); // q
  Cov.block(3, 3, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity();  // p
  Cov.block(6, 6, 3, 3) = std::pow(0.01, 2) * Eigen::Matrix3d::Identity();  // v (static)
  StateHelper::set_initial_covariance(state, Cov, order);

  // Set the state time
  state->_timestamp = imustate(0, 0);
  state->_imu_endpoint = imu_time;
  state->_imu_endpoint_valid = true;
  startup_time = imustate(0, 0);
  startup_imu_time = state->imu_endpoint();
  is_initialized_vio = true;

  // Cleanup any features older then the initialization time
  if (state->uses_physical_clones()) {
    for (int camera = 0; camera < state->_options.num_cameras; ++camera) {
      const double cutoff = std::nextafter(imu_time - state->cam_imu_dt(camera), std::numeric_limits<double>::infinity());
      trackFEATS->get_feature_database()->cleanup_measurements_camera(camera, cutoff);
      if (trackARUCO)
        trackARUCO->get_feature_database()->cleanup_measurements_camera(camera, cutoff);
    }
  } else {
    trackFEATS->get_feature_database()->cleanup_measurements(state->_timestamp);
    if (trackARUCO)
      trackARUCO->get_feature_database()->cleanup_measurements(state->_timestamp);
  }

  // Print what we init'ed with
  PRINT_DEBUG(GREEN "[INIT]: INITIALIZED FROM GROUNDTRUTH FILE!!!!!\n" RESET);
  PRINT_DEBUG(GREEN "[INIT]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, state->_imu->quat()(0), state->_imu->quat()(1),
              state->_imu->quat()(2), state->_imu->quat()(3));
  PRINT_DEBUG(GREEN "[INIT]: bias gyro = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_g()(0), state->_imu->bias_g()(1),
              state->_imu->bias_g()(2));
  PRINT_DEBUG(GREEN "[INIT]: velocity = %.4f, %.4f, %.4f\n" RESET, state->_imu->vel()(0), state->_imu->vel()(1), state->_imu->vel()(2));
  PRINT_DEBUG(GREEN "[INIT]: bias accel = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_a()(0), state->_imu->bias_a()(1),
              state->_imu->bias_a()(2));
  PRINT_DEBUG(GREEN "[INIT]: position = %.4f, %.4f, %.4f\n" RESET, state->_imu->pos()(0), state->_imu->pos()(1), state->_imu->pos()(2));
}

struct VioManager::InitializationAttempt {
  std::shared_ptr<ov_init::InertialInitializer> input;
  std::shared_ptr<ov_type::IMU> imu;
  std::weak_ptr<State> state_owner;
  uint64_t generation = 0;
  bool wait_for_jerk = true;
  bool success = false;
  bool threw = false;
  std::atomic<bool> complete{false};
  double elapsed_seconds = 0.;
  double timestamp = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd covariance;
  std::vector<std::shared_ptr<ov_type::Type>> order;
  std::map<double, std::shared_ptr<ov_type::PoseJPL>> clones;
  std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> features;
};

// One job slot, no backlog. An idle worker is retained for failed attempts,
// which avoids creating another OS thread for each sparse startup frame.
struct VioManager::InitializationWorker {
  std::mutex mutex;
  std::condition_variable changed;
  bool stop = false;
  std::shared_ptr<InitializationAttempt> next;
};

VioManager::~VioManager() { stop_initialization(); }

void VioManager::stop_initialization_worker() {
  // Reset/restore/destruction are quiescent consumer operations. The worker
  // never needs a manager/state lock, so joining cannot deadlock on that lock.
  if (initialization_worker) {
    {
      std::lock_guard<std::mutex> lock(initialization_worker->mutex);
      initialization_worker->stop = true;
      initialization_worker->next.reset();
    }
    initialization_worker->changed.notify_one();
  }
  if (initialization_thread.joinable())
    initialization_thread.join();
  initialization_worker.reset();
}

void VioManager::stop_initialization() {
  stop_initialization_worker();
  initialization_attempt.reset();
  ++initialization_generation;
  thread_init_running.store(false);
  thread_init_success.store(false);
  camera_queue_init.clear();
  camera_queue_init_owners.clear();
}

void VioManager::queue_initialization_camera(const ov_core::CameraData &message) {
  const size_t capacity = static_cast<size_t>(std::max(1, params.state_options.max_pose_clones()));
  if (state->uses_physical_clones()) {
    for (int camera : message.sensor_ids) {
      if (camera_queue_init_owners.size() == capacity)
        camera_queue_init_owners.pop_front();
      camera_queue_init_owners.emplace_back(static_cast<size_t>(camera), message.timestamp);
    }
  } else {
    if (camera_queue_init.size() == capacity)
      camera_queue_init.pop_front();
    camera_queue_init.push_back(message.timestamp);
  }
}

void VioManager::compute_initialization(const std::shared_ptr<InitializationAttempt> &attempt) {
  const auto begin = ov_core::prof_now();
  try {
    attempt->success = attempt->input->initialize(attempt->timestamp, attempt->covariance, attempt->order,
        attempt->imu, attempt->clones, attempt->features, attempt->wait_for_jerk);
  } catch (...) {
    // Never terminate the process or publish a partial output from a worker.
    attempt->success = false;
    attempt->threw = true;
  }
  attempt->elapsed_seconds = ov_core::prof_s(begin, ov_core::prof_now());
  attempt->complete.store(true, std::memory_order_release);
}

void VioManager::run_initialization_worker(const std::shared_ptr<InitializationWorker> &worker) {
  for (;;) {
    std::shared_ptr<InitializationAttempt> attempt;
    {
      std::unique_lock<std::mutex> lock(worker->mutex);
      worker->changed.wait(lock, [&] { return worker->stop || worker->next; });
      if (worker->stop)
        return;
      attempt = std::move(worker->next);
    }
    compute_initialization(attempt);
  }
}

bool VioManager::initialization_completed() const {
  return initialization_attempt && initialization_attempt->complete.load(std::memory_order_acquire);
}

bool VioManager::try_to_initialize(const ov_core::CameraData &message) {
  // Every caller is the existing serialized VIO/IMU consumer. In particular,
  // the ROS consumer's state lock now covers the entire import and catch-up.
  if (thread_init_success.load())
    return true;
  if (initialization_attempt) {
    if (!initialization_completed()) {
      queue_initialization_camera(message);
      return false;
    }
    auto completed = std::move(initialization_attempt);
    thread_init_running.store(false);
    return finish_initialization(completed);
  }

  auto attempt = std::make_shared<InitializationAttempt>();
  attempt->input = initializer->make_attempt();
  attempt->imu = std::static_pointer_cast<ov_type::IMU>(state->_imu->clone());
  attempt->state_owner = state;
  attempt->generation = ++initialization_generation;
  const auto &options = state->_options;
  const bool camera_consider = options.do_calib_camera_pose || options.do_calib_camera_intrinsics || options.do_calib_camera_timeoffset;
  bool supported_physical = state->uses_physical_clones() && params.init_options.init_warmstart_inject &&
      !options.do_calib_camera_readout && !options.do_calib_imu_intrinsics && !options.do_calib_imu_g_sensitivity;
  const auto joint_reset=initializer->physical_reset_prior();
  if (camera_consider && warmstart_next_init.load() && !joint_reset) supported_physical=false;
  for (const auto &readout : state->_calib_camera_readout)
    supported_physical = supported_physical && readout.second &&
        ov_init::finite_initializer_time(readout.second->value()(0)) && readout.second->value()(0) == 0.;
  if (supported_physical) {
    ov_core::InitPhysicalWarmRequest request;
    if (!StateHelper::make_initial_physical_warm_request(state,attempt->generation,request,joint_reset)) {
      PRINT_WARNING(YELLOW "[init]: physical warm calibration snapshot is not a supported retained prior\n" RESET);
      return false;
    }
    const bool requested=attempt->input->request_physical_warmstart(request);
    if(joint_reset && !requested)return false;
  } else if(joint_reset) return false;

  attempt->wait_for_jerk = (updaterZUPT == nullptr);
  if (!params.use_multi_threading_subs) {
    // Deterministic/replay mode pays no thread creation or join overhead.
    compute_initialization(attempt);
    finish_initialization(attempt);
  } else {
    initialization_attempt = attempt;
    thread_init_running.store(true);
    try {
      if (!initialization_worker) {
        auto worker = std::make_shared<InitializationWorker>();
        initialization_thread = std::thread(&VioManager::run_initialization_worker, worker);
        initialization_worker = std::move(worker);
      }
      {
        std::lock_guard<std::mutex> lock(initialization_worker->mutex);
        assert(!initialization_worker->next);
        initialization_worker->next = attempt;
      }
      initialization_worker->changed.notify_one();
    } catch (...) {
      initialization_attempt.reset();
      thread_init_running.store(false);
      throw;
    }
  }
  // Preserve the synchronous caller contract: the triggering camera is still
  // an initialization frame; the next camera performs the first filter update.
  return false;
}

bool VioManager::finish_initialization(const std::shared_ptr<InitializationAttempt> &attempt) {
  bool success = attempt->success && attempt->generation == initialization_generation &&
                 attempt->state_owner.lock() == state;
  if (attempt->threw)
    PRINT_WARNING(YELLOW "[init]: initializer threw; discarding detached output\n" RESET);
  const double timestamp = attempt->timestamp;
  const auto &covariance = attempt->covariance;
  const auto &initialized_imu = attempt->imu;
  const auto &clones_IMU = attempt->clones;
  const auto &features_SLAM = attempt->features;
  const auto *physical_result = attempt->input->physical_warm_result();
  const auto joint_reset=initializer->physical_reset_prior();
  if(attempt->input->reset_prior().filter!=initializer->reset_prior().filter)success=false;
  if(attempt->input->physical_reset_prior()!=joint_reset ||
      (joint_reset && (!physical_result || physical_result->reset_prior!=joint_reset)))success=false;
  if (success) {
    const int nimu = state->_imu->size();
    const int njoint = nimu + 6 * static_cast<int>(clones_IMU.size());
    const Eigen::Matrix<double, 16, 1> value = initialized_imu->value(), fej = initialized_imu->fej();
    bool valid = ov_init::finite_initializer_time(timestamp) &&
                 ov_init::finite_initializer_time(timestamp + state->cam_imu_dt_ref()) &&
                 covariance.rows() == covariance.cols() &&
                 (covariance.rows() == nimu || covariance.rows() == njoint);
    for (int i = 0; valid && i < value.size(); ++i)
      valid = ov_init::finite_initializer_time(value(i)) && ov_init::finite_initializer_time(fej(i));
    // Quaternion normalization is an output contract, not a new motion gate.
    valid = valid && std::abs(value.head<4>().squaredNorm() - 1.) <= 1e-6 &&
                     std::abs(fej.head<4>().squaredNorm() - 1.) <= 1e-6;
    if (valid)
      valid = StateHelper::valid_initial_covariance(covariance.topLeftCorner(nimu, nimu), physical_result != nullptr);
    if (!valid) {
      PRINT_WARNING(YELLOW "[init]: rejected invalid mean, time or IMU covariance; retaining uninitialized state\n" RESET);
      success = false;
    }
  }

  bool physical_warm=false;
  if (success && physical_result) {
    physical_warm=StateHelper::set_initial_state_physical_warm(state,*physical_result,attempt->generation,joint_reset);
    if (!physical_warm && (joint_reset || !physical_result->consider.empty())) {
      // A stale/malformed retained prior cannot be replaced by a cold marginal
      // with zero cross terms. Keep the entire live state and exact image rows.
      PRINT_WARNING(YELLOW "[init]: rejected conditional camera handoff; retaining the uninitialized state\n" RESET);
      success=false;
    }
  }
  // If we have initialized successfully we will set the covariance and state elements as needed
  if (success) {

    // The physical result stages all allocations and validates complete
    // ownership/covariance before installing any mean, clock or Type identity.
    // Explicit warm start also works at first boot. Camera consider import was
    // accepted above as one transaction, including its retained calibration.
    bool legacy_warm = false;
    if (!physical_warm) {
      state->_imu->set_value(initialized_imu->value());
      state->_imu->set_fej(initialized_imu->fej());
      const int njoint = state->_imu->size() + 6 * static_cast<int>(clones_IMU.size());
      legacy_warm = !state->uses_physical_clones() && warmstart_next_init.load() && params.init_options.init_warmstart_inject &&
          !clones_IMU.empty() && covariance.rows() == njoint &&
          static_cast<int>(clones_IMU.size()) <= params.state_options.max_pose_clones();
      if (legacy_warm)
        legacy_warm = StateHelper::set_initial_state_warmstart(state, covariance, clones_IMU);
      if (!legacy_warm) {
        std::vector<std::shared_ptr<ov_type::Type>> imu_order = {state->_imu};
        StateHelper::set_initial_covariance(state, covariance.topLeftCorner(state->_imu->size(), state->_imu->size()), imu_order);
      }
      state->_timestamp = timestamp;
      state->_imu_endpoint = timestamp + state->cam_imu_dt_ref();
      state->_imu_endpoint_valid = true;
    }
    // Preserve the exact physical endpoint; reversing a reference-clock label
    // can lose an ULP and move future camera ownership across the boundary.
    startup_time = state->_timestamp;
    startup_imu_time = state->imu_endpoint();

    // Retain unused observations supported by the warm clone window, but consume the exact
    // image factors already represented in its posterior. Keeping those pixels would count
    // their likelihood twice in the next MSCKF/SLAM update. In this equal-clock warm path,
    // the returned feature IDs and selected clone keys identify precisely that factor set.
    // Cleanup stays inside the database mutex and preserves later asynchronous arrivals.
    size_t consumed_init_rows = 0;
    if (physical_warm) {
      consumed_init_rows = trackFEATS->get_feature_database()->cleanup_measurements_exact_observations(
          physical_result->consumed_observations);
    }
    double clean_time = legacy_warm ? std::nextafter(clones_IMU.begin()->first, -std::numeric_limits<double>::infinity())
                                    : state->_timestamp;
    if (state->uses_physical_clones()) {
      for (int camera = 0; camera < state->_options.num_cameras; ++camera) {
        double cutoff = std::nextafter(state->imu_endpoint() - state->cam_imu_dt(camera),
                                      std::numeric_limits<double>::infinity());
        if (physical_warm) {
          for (const auto &owner : physical_result->owners)
            if (owner.camera_id == static_cast<size_t>(camera)) cutoff = std::min(cutoff, owner.raw_time);
        }
        trackFEATS->get_feature_database()->cleanup_measurements_camera(camera, cutoff);
        if (trackARUCO)
          trackARUCO->get_feature_database()->cleanup_measurements_camera(camera, cutoff);
      }
    } else {
      trackFEATS->get_feature_database()->cleanup_measurements(clean_time);
      if (trackARUCO)
        trackARUCO->get_feature_database()->cleanup_measurements(clean_time);
    }
    if (legacy_warm) {
      std::vector<size_t> initialized_feature_ids;
      std::vector<double> initialized_pose_times;
      initialized_feature_ids.reserve(features_SLAM.size());
      initialized_pose_times.reserve(clones_IMU.size());
      for (const auto &feature : features_SLAM) initialized_feature_ids.push_back(feature.first);
      for (const auto &clone : clones_IMU) initialized_pose_times.push_back(clone.first);
      consumed_init_rows = trackFEATS->get_feature_database()->cleanup_measurements_exact_for_features(
          initialized_feature_ids, initialized_pose_times);
    }
    trackFEATS->set_num_features(std::floor(static_cast<double>(params.num_pts)));
    if (physical_warm) {
      PRINT_INFO(GREEN "[init]: PHYSICAL WARM-START injected %zu exposure owners, joint cov %dx%d (consumed %zu exact init observations)\n" RESET,
          physical_result->owners.size(), static_cast<int>(physical_result->joint_covariance.rows()),
          static_cast<int>(physical_result->joint_covariance.cols()), consumed_init_rows);
    } else if (legacy_warm) {
      PRINT_INFO(GREEN "[init]: WARM-START injected %d clones, joint cov %dx%d (consumed %zu init observations; unused window rows retained)\n" RESET,
          static_cast<int>(clones_IMU.size()), static_cast<int>(covariance.rows()), static_cast<int>(covariance.cols()), consumed_init_rows);
    }

    // If we are moving then don't do zero velocity update4
    if (state->_imu->vel().norm() > params.zupt_max_velocity) {
      has_moved_since_zupt = true;
    }

    // Else we are good to go, print out our stats
    PRINT_INFO(GREEN "[init]: successful initialization in %.4f seconds\n" RESET, attempt->elapsed_seconds);
    PRINT_INFO(GREEN "[init]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, state->_imu->quat()(0), state->_imu->quat()(1),
               state->_imu->quat()(2), state->_imu->quat()(3));
    PRINT_INFO(GREEN "[init]: bias gyro = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_g()(0), state->_imu->bias_g()(1),
               state->_imu->bias_g()(2));
    PRINT_INFO(GREEN "[init]: velocity = %.4f, %.4f, %.4f\n" RESET, state->_imu->vel()(0), state->_imu->vel()(1), state->_imu->vel()(2));
    PRINT_INFO(GREEN "[init]: bias accel = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_a()(0), state->_imu->bias_a()(1),
               state->_imu->bias_a()(2));
    PRINT_INFO(GREEN "[init]: position = %.4f, %.4f, %.4f\n" RESET, state->_imu->pos()(0), state->_imu->pos()(1), state->_imu->pos()(2));

    // Remove any camera times that are order then the initialized time
    // This can happen if the initialization has taken a while to perform
    std::vector<double> camera_timestamps_to_init;
    for (size_t i = 0; i < camera_queue_init.size(); i++) {
      if (camera_queue_init.at(i) > timestamp) {
        camera_timestamps_to_init.push_back(camera_queue_init.at(i));
      }
    }

    // Now we have initialized we will propagate the state to the current timestep
    // In general this should be ok as long as the initialization didn't take too long to perform
    // Propagating over multiple seconds will become an issue if the initial biases are bad
    if (state->uses_physical_clones()) {
      std::stable_sort(camera_queue_init_owners.begin(), camera_queue_init_owners.end(), [&](const auto &a, const auto &b) {
        return a.second + state->cam_imu_dt(a.first) < b.second + state->cam_imu_dt(b.first);
      });
      for (const auto &owner : camera_queue_init_owners) {
        const double target = owner.second + state->cam_imu_dt(owner.first);
        if (target < state->imu_endpoint() || state->find_pose(owner.first, owner.second))
          continue;
        Propagator::EndpointKinematics endpoint;
        if (!propagator->propagate_to_imu(state, target, target - state->cam_imu_dt_ref(), endpoint))
          continue;
        State::ExposurePose view;
        view.camera_id = owner.first;
        view.raw_time = owner.second;
        view.imu_time = target;
        view.pose = StateHelper::augment_pose_view(state, owner.first, endpoint.omega);
        view.kinematics.omega = endpoint.omega;
        view.kinematics.omega_fej = endpoint.omega_fej;
        view.kinematics.vel = state->_imu->vel();
        view.kinematics.vel_fej = state->_imu->vel_fej();
        state->_exposure_poses.push_back(std::move(view));
        StateHelper::marginalize_old_clone(state);
      }
    } else {
      for (double camera_time : camera_timestamps_to_init) {
        if (!propagator->propagate_and_clone(state, camera_time))
          continue;
        StateHelper::marginalize_old_clone(state);
      }
    }
    PRINT_DEBUG(YELLOW "[init]: moved the state forward %.2f seconds\n" RESET, state->_timestamp - timestamp);
    // Soft-reset re-init episode is over; the next init cold-starts unless another soft_reset() arms it.
    warmstart_next_init.store(false);
    initializer->clear_reset_prior(); // disarm the bias-prior episode too (lives in ov_init)
    thread_init_success = true;
    stop_initialization_worker();
    camera_queue_init.clear();
    camera_queue_init_owners.clear();

  } else {
    PRINT_DEBUG(YELLOW "[init]: failed initialization in %.4f seconds\n" RESET, attempt->elapsed_seconds);
    thread_init_success = false;
    camera_queue_init.clear();
    camera_queue_init_owners.clear();
  }
  return thread_init_success.load();
}

void VioManager::retriangulate_active_tracks(const ov_core::CameraData &message) {

  // Start timing
  ov_core::ProfTime retri_rT1, retri_rT2, retri_rT3;
  retri_rT1 = ov_core::prof_now();

  // Clear old active track data
  assert(!message.sensor_ids.empty() && state->find_pose(message.sensor_ids.front(), message.timestamp));
  active_tracks_time = message.timestamp;
  active_image = cv::Mat();
  trackFEATS->display_active(active_image, 255, 255, 255, 255, 255, 255, " ");
  if (!active_image.empty()) {
    active_image = active_image(cv::Rect(0, 0, message.images.at(0).cols, message.images.at(0).rows));
  }
  active_tracks_posinG.clear();
  active_tracks_uvd.clear();

  // Current active tracks in our frontend
  // TODO: should probably assert here that these are at the message time...
  auto last_obs = trackFEATS->get_last_obs();
  auto last_ids = trackFEATS->get_last_ids();

  // New set of linear systems that only contain the latest track info
  std::map<size_t, Eigen::Matrix3d> active_feat_linsys_A_new;
  std::map<size_t, Eigen::Vector3d> active_feat_linsys_b_new;
  std::map<size_t, int> active_feat_linsys_count_new;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG_new;

  // Append our new observations for each camera
  std::map<size_t, cv::Point2f> feat_uvs_in_cam0;
  for (auto const &cam_id : message.sensor_ids) {

    // IMU historical clone
    Eigen::Matrix3d R_GtoI = state->pose_for_camera(cam_id, active_tracks_time)->Rot();
    Eigen::Vector3d p_IinG = state->pose_for_camera(cam_id, active_tracks_time)->pos();
    const double dt_cam_delta = state->uses_physical_clones() ? 0.0 : state->cam_imu_dt_delta(cam_id) + state->epoch_residual(cam_id, active_tracks_time);
    if (std::abs(dt_cam_delta) > 1e-10 && state->clone_kinematics(cam_id, active_tracks_time) != nullptr) {
      const State::CloneKinematics &kin = *state->clone_kinematics(cam_id, active_tracks_time);
      const bool has_bridge = state->epoch_bridge(cam_id, active_tracks_time) != nullptr;
      if (legacy_exposure::uses_body_velocity(*state, cam_id, has_bridge)) {
        legacy_exposure::warp_pose(R_GtoI, p_IinG, state->pose_for_camera(cam_id, active_tracks_time)->Rot_fej(),
                                    kin.vel, kin.omega, dt_cam_delta);
      } else {
        R_GtoI = exp_so3(-kin.omega * dt_cam_delta) * R_GtoI;
        p_IinG = p_IinG + kin.vel * dt_cam_delta;
      }
    }

    // Calibration for this cam_id
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(cam_id)->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(cam_id)->pos();

    // Convert current CAMERA position relative to global
    Eigen::Matrix3d R_GtoCi = R_ItoC * R_GtoI;
    Eigen::Vector3d p_CiinG = p_IinG - R_GtoCi.transpose() * p_IinC;

    // Loop through each measurement
    assert(last_obs.find(cam_id) != last_obs.end());
    assert(last_ids.find(cam_id) != last_ids.end());
    for (size_t i = 0; i < last_obs.at(cam_id).size(); i++) {

      // Record this feature uv if is seen from cam0
      size_t featid = last_ids.at(cam_id).at(i);
      cv::Point2f pt_d = last_obs.at(cam_id).at(i).pt;
      if (cam_id == 0) {
        feat_uvs_in_cam0[featid] = pt_d;
      }

      // Skip this feature if it is a SLAM feature (the state estimate takes priority)
      if (state->_features_SLAM.find(featid) != state->_features_SLAM.end()) {
        continue;
      }

      // Get the UV coordinate normal
      cv::Point2f pt_n = state->_cam_intrinsics_cameras.at(cam_id)->undistort_cv(pt_d);
      Eigen::Matrix<double, 3, 1> b_i;
      b_i << pt_n.x, pt_n.y, 1;
      b_i = R_GtoCi.transpose() * b_i;
      b_i = b_i / b_i.norm();
      Eigen::Matrix3d Bperp = skew_x(b_i);

      // Append to our linear system
      Eigen::Matrix3d Ai = Bperp.transpose() * Bperp;
      Eigen::Vector3d bi = Ai * p_CiinG;
      if (active_feat_linsys_A.find(featid) == active_feat_linsys_A.end()) {
        active_feat_linsys_A_new.insert({featid, Ai});
        active_feat_linsys_b_new.insert({featid, bi});
        active_feat_linsys_count_new.insert({featid, 1});
      } else {
        active_feat_linsys_A_new[featid] = Ai + active_feat_linsys_A[featid];
        active_feat_linsys_b_new[featid] = bi + active_feat_linsys_b[featid];
        active_feat_linsys_count_new[featid] = 1 + active_feat_linsys_count[featid];
      }

      // For this feature, recover its 3d position if we have enough observations!
      if (active_feat_linsys_count_new.at(featid) > 3) {

        // Recover feature estimate
        Eigen::Matrix3d A = active_feat_linsys_A_new[featid];
        Eigen::Vector3d b = active_feat_linsys_b_new[featid];
        Eigen::MatrixXd p_FinG = A.colPivHouseholderQr().solve(b);
        Eigen::MatrixXd p_FinCi = R_GtoCi * (p_FinG - p_CiinG);

        // Check A and p_FinCi
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
        Eigen::MatrixXd singularValues;
        singularValues.resize(svd.singularValues().rows(), 1);
        singularValues = svd.singularValues();
        double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

        // If we have a bad condition number, or it is too close
        // Then set the flag for bad (i.e. set z-axis to nan)
        if (std::abs(condA) <= params.featinit_options.max_cond_number && p_FinCi(2, 0) >= params.featinit_options.min_dist &&
            p_FinCi(2, 0) <= params.featinit_options.max_dist && ov_core::numeric::finite_matrix(p_FinCi)) {
          active_tracks_posinG_new[featid] = p_FinG;
        }
      }
    }
  }
  size_t total_triangulated = active_tracks_posinG.size();

  // Update active set of linear systems
  active_feat_linsys_A = active_feat_linsys_A_new;
  active_feat_linsys_b = active_feat_linsys_b_new;
  active_feat_linsys_count = active_feat_linsys_count_new;
  active_tracks_posinG = active_tracks_posinG_new;
  retri_rT2 = ov_core::prof_now();

  // Return if no features
  if (active_tracks_posinG.empty() && state->_features_SLAM.empty())
    return;

  // Append our SLAM features we have
  for (const auto &feat : state->_features_SLAM) {
    Eigen::Vector3d p_FinG = feat.second->get_xyz(false);
    if (LandmarkRepresentation::is_relative_representation(feat.second->_feat_representation)) {
      // Assert that we have an anchor pose for this feature
      assert(feat.second->_anchor_cam_id != -1);
      // Get calibration for our anchor camera
      Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feat.second->_anchor_cam_id)->Rot();
      Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feat.second->_anchor_cam_id)->pos();
      // Anchor pose orientation and position
      Eigen::Matrix3d R_GtoI = state->pose_for_camera(feat.second->_anchor_cam_id, feat.second->_anchor_clone_timestamp)->Rot();
      Eigen::Vector3d p_IinG = state->pose_for_camera(feat.second->_anchor_cam_id, feat.second->_anchor_clone_timestamp)->pos();
      // Feature in the global frame
      p_FinG = R_GtoI.transpose() * R_ItoC.transpose() * (feat.second->get_xyz(false) - p_IinC) + p_IinG;
    }
    active_tracks_posinG[feat.second->_featid] = p_FinG;
  }

  // Calibration of the first camera (cam0)
  std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(0);
  std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(0);
  Eigen::Matrix<double, 3, 3> R_ItoC = calibration->Rot();
  Eigen::Matrix<double, 3, 1> p_IinC = calibration->pos();

  // Get current IMU clone state
  std::shared_ptr<PoseJPL> clone_Ii = state->find_pose(0, active_tracks_time);
  if (!clone_Ii)
    return; // This raw frame belongs to a different camera; no cam0 image to project into.
  Eigen::Matrix3d R_GtoIi = clone_Ii->Rot();
  Eigen::Vector3d p_IiinG = clone_Ii->pos();

  // 4. Next we can update our variable with the global position
  //    We also will project the features into the current frame
  for (const auto &feat : active_tracks_posinG) {

    // For now skip features not seen from current frame
    // TODO: should we publish other features not tracked in cam0??
    if (feat_uvs_in_cam0.find(feat.first) == feat_uvs_in_cam0.end())
      continue;

    // Calculate the depth of the feature in the current frame
    // Project SLAM feature and non-cam0 features into the current frame of reference
    Eigen::Vector3d p_FinIi = R_GtoIi * (feat.second - p_IiinG);
    Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
    double depth = p_FinCi(2);
    Eigen::Vector2d uv_dist;
    if (feat_uvs_in_cam0.find(feat.first) != feat_uvs_in_cam0.end()) {
      uv_dist << (double)feat_uvs_in_cam0.at(feat.first).x, (double)feat_uvs_in_cam0.at(feat.first).y;
    } else {
      Eigen::Vector2d uv_norm;
      uv_norm << p_FinCi(0) / depth, p_FinCi(1) / depth;
      uv_dist = state->_cam_intrinsics_cameras.at(0)->distort_d(uv_norm);
    }

    // Skip if not valid (i.e. negative depth, or outside of image)
    if (depth < 0.1) {
      continue;
    }

    // Skip if not valid (i.e. negative depth, or outside of image)
    int width = state->_cam_intrinsics_cameras.at(0)->w();
    int height = state->_cam_intrinsics_cameras.at(0)->h();
    if (uv_dist(0) < 0 || (int)uv_dist(0) >= width || uv_dist(1) < 0 || (int)uv_dist(1) >= height) {
      // PRINT_DEBUG("feat %zu -> depth = %.2f | u_d = %.2f | v_d = %.2f\n",(*it2)->featid,depth,uv_dist(0),uv_dist(1));
      continue;
    }

    // Finally construct the uv and depth
    Eigen::Vector3d uvd;
    uvd << uv_dist, depth;
    active_tracks_uvd.insert({feat.first, uvd});
  }
  retri_rT3 = ov_core::prof_now();

  // Timing information
  PRINT_ALL(CYAN "[RETRI-TIME]: %.4f seconds for triangulation (%zu tri of %zu active)\n" RESET,
            ov_core::prof_s(retri_rT1, retri_rT2), total_triangulated, active_feat_linsys_A.size());
  PRINT_ALL(CYAN "[RETRI-TIME]: %.4f seconds for re-projection into current\n" RESET, ov_core::prof_s(retri_rT2, retri_rT3));
  PRINT_ALL(CYAN "[RETRI-TIME]: %.4f seconds total\n" RESET, ov_core::prof_s(retri_rT1, retri_rT3));
}

cv::Mat VioManager::get_historical_viz_image() {

  // Return if not ready yet
  if (state == nullptr || trackFEATS == nullptr)
    return cv::Mat();

  // Build an id-list of what features we should highlight (i.e. SLAM)
  std::vector<size_t> highlighted_ids;
  for (const auto &feat : state->_features_SLAM) {
    highlighted_ids.push_back(feat.first);
  }

  // Text we will overlay if needed
  std::string overlay = (did_zupt_update) ? "zvupt" : "";
  overlay = (!is_initialized_vio) ? "init" : overlay;

  // Get the current active tracks
  cv::Mat img_history;
  trackFEATS->display_history(img_history, 255, 255, 0, 255, 255, 255, highlighted_ids, overlay);
  if (trackARUCO != nullptr) {
    trackARUCO->display_history(img_history, 0, 255, 255, 255, 255, 255, highlighted_ids, overlay);
    // trackARUCO->display_active(img_history, 0, 255, 255, 255, 255, 255, overlay);
  }

  // Finally return the image
  return img_history;
}

std::vector<Eigen::Vector3d> VioManager::get_features_SLAM() {
  std::vector<Eigen::Vector3d> slam_feats;
  for (auto &f : state->_features_SLAM) {
    if ((int)f.first <= 4 * state->_options.max_aruco_features)
      continue;
    if (ov_type::LandmarkRepresentation::is_relative_representation(f.second->_feat_representation)) {
      // Assert that we have an anchor pose for this feature
      assert(f.second->_anchor_cam_id != -1);
      // Get calibration for our anchor camera
      Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->pos();
      // Anchor pose orientation and position
      Eigen::Matrix<double, 3, 3> R_GtoI = state->pose_for_camera(f.second->_anchor_cam_id, f.second->_anchor_clone_timestamp)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinG = state->pose_for_camera(f.second->_anchor_cam_id, f.second->_anchor_clone_timestamp)->pos();
      // Feature in the global frame
      slam_feats.push_back(R_GtoI.transpose() * R_ItoC.transpose() * (f.second->get_xyz(false) - p_IinC) + p_IinG);
    } else {
      slam_feats.push_back(f.second->get_xyz(false));
    }
  }
  return slam_feats;
}

std::vector<Eigen::Vector3d> VioManager::get_features_ARUCO() {
  std::vector<Eigen::Vector3d> aruco_feats;
  for (auto &f : state->_features_SLAM) {
    if ((int)f.first > 4 * state->_options.max_aruco_features)
      continue;
    if (ov_type::LandmarkRepresentation::is_relative_representation(f.second->_feat_representation)) {
      // Assert that we have an anchor pose for this feature
      assert(f.second->_anchor_cam_id != -1);
      // Get calibration for our anchor camera
      Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->pos();
      // Anchor pose orientation and position
      Eigen::Matrix<double, 3, 3> R_GtoI = state->pose_for_camera(f.second->_anchor_cam_id, f.second->_anchor_clone_timestamp)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinG = state->pose_for_camera(f.second->_anchor_cam_id, f.second->_anchor_clone_timestamp)->pos();
      // Feature in the global frame
      aruco_feats.push_back(R_GtoI.transpose() * R_ItoC.transpose() * (f.second->get_xyz(false) - p_IinC) + p_IinG);
    } else {
      aruco_feats.push_back(f.second->get_xyz(false));
    }
  }
  return aruco_feats;
}
