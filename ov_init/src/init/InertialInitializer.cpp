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

#include "InertialInitializer.h"
#include "InitializerCameraClock.h"
#include "PhysicalResetWindow.h"
#include "MarginalResetPrior.h"
#include "dynamic/RawImuCpi.h"
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"

#include <algorithm>
#include <array>
#include <stdexcept>

#include "dynamic/DynamicInitializer.h"
#include "static/StaticInitializer.h"

#include "feat/FeatureHelper.h"
#include "types/Type.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_init;

InertialInitializer::InertialInitializer(InertialInitializerOptions &params_, std::shared_ptr<ov_core::FeatureDatabase> db)
    : params(params_), _db(db) {

  // Vector of our IMU data
  imu_data = std::make_shared<std::vector<ov_core::ImuData>>();

  // Soft-reset hand-off context (bias prior episode; armed from VioManager::soft_reset)
  reset_ctx = std::make_shared<ResetContext>();

  // Runtime calibration mutates the live camera objects. Initializer options
  // own their calibration values for the whole lifetime of this episode.
  for (auto &camera : params.camera_intrinsics)
    camera.second = camera.second->clone();
}

void InertialInitializer::set_reset_prior(const ResetBiasPrior &prior) {
  if(prior.joint) {
    if(!set_physical_reset_prior(prior.joint))throw std::invalid_argument("unsupported joint initializer reset prior");
  } else reset_ctx->arm(prior);
}

bool InertialInitializer::set_physical_reset_prior(std::shared_ptr<const InitPhysicalResetPrior> prior) {
#ifndef USE_CERES_FREE_INIT
  return false;
#else
  if (!prior || !params.init_warmstart_inject || !params.init_dyn_use || !params.init_dyn_reset_prior_use ||
      params.init_dyn_mle_opt_calib || params.init_dyn_fix_ba_on_reset || !valid_physical_reset_prior(*prior) ||
      prior->calibration.size()!=size_t(params.num_cameras)) return false;
  RawImuCpiModel model;
  if (!model.set_calibration(prior->imu_accel_map,prior->imu_gyro_map,prior->imu_tg)) return false;
  Eigen::Matrix<double,6,1> expected_rw;
  expected_rw.head<3>().setConstant(params.sigma_wb*params.sigma_wb);
  expected_rw.tail<3>().setConstant(params.sigma_ab*params.sigma_ab);
  if (!(expected_rw.array()==prior->bias_rw_variance.array()).all()) return false;
  const double scale=prior->cause==1 ? params.init_dyn_reset_prior_divergence_infl : 1.;
  const std::array<double,6> gates{{params.init_dyn_reset_prior_max_bg,params.init_dyn_reset_prior_max_ba,
      params.init_dyn_reset_prior_max_sigma_bg,params.init_dyn_reset_prior_max_sigma_ba,
      params.init_dyn_reset_prior_sigma_floor_bg,params.init_dyn_reset_prior_sigma_floor_ba}};
  for(double gate:gates)if(!ov_core::numeric::finite(gate) || gate<0.)return false;
  if (!ov_core::numeric::finite(scale) || scale<1. ||
      prior->bias_mean.head<3>().norm()>params.init_dyn_reset_prior_max_bg ||
      prior->bias_mean.tail<3>().norm()>params.init_dyn_reset_prior_max_ba ||
      scale*scale*prior->bias_covariance.diagonal().head<3>().maxCoeff()>params.init_dyn_reset_prior_max_sigma_bg*params.init_dyn_reset_prior_max_sigma_bg ||
      scale*scale*prior->bias_covariance.diagonal().tail<3>().maxCoeff()>params.init_dyn_reset_prior_max_sigma_ba*params.init_dyn_reset_prior_max_sigma_ba)
    return false;
  ResetBiasPrior snapshot;
  snapshot.bg=prior->bias_mean.head<3>(); snapshot.ba=prior->bias_mean.tail<3>();
  snapshot.sigma_bg=prior->bias_covariance.diagonal().head<3>().cwiseSqrt();
  snapshot.sigma_ba=prior->bias_covariance.diagonal().tail<3>().cwiseSqrt();
  snapshot.t_snapshot=prior->imu_endpoint; snapshot.cause=prior->cause; snapshot.valid=true;
  snapshot.joint=std::move(prior); reset_ctx->arm(snapshot);
  return true;
#endif
}

void InertialInitializer::clear_reset_prior() { reset_ctx->disarm(); }

void InertialInitializer::feed_imu(const ov_core::ImuData &message, double oldest_time) {

  std::lock_guard<std::mutex> lock(imu_data_mtx);

  // Append it to our vector
  imu_data->emplace_back(message);

  // Sort our imu data (handles any out of order measurements)
  // std::sort(imu_data->begin(), imu_data->end(), [](const IMUDATA i, const IMUDATA j) {
  //    return i.timestamp < j.timestamp;
  //});

  // Loop through and delete imu messages that are older than our requested time
  // std::cout << "INIT: imu_data.size() " << imu_data->size() << std::endl;
  if (oldest_time != -1) {
    // Stable compaction also preserves the legacy handling of out-of-order
    // samples, without shifting the entire history once per removed sample.
    imu_data->erase(std::remove_if(imu_data->begin(), imu_data->end(),
                                   [oldest_time](const ImuData &sample) { return sample.timestamp < oldest_time; }),
                    imu_data->end());
  }
}

void InertialInitializer::feed_imu_batch(const std::vector<ov_core::ImuData>& messages, double oldest_time) {
    
    //ADDING THIS GUARD TO PREVENT CRASH FROM EMPTY VECTOR
    if (messages.empty()) return;
    std::lock_guard<std::mutex> lock(imu_data_mtx);
    
    // Insert all measurements at once
    //have to use insert...
    imu_data->insert(imu_data->end(), messages.begin(), messages.end());
    
    // Clean old measurements if needed
    if (oldest_time != -1) {
        imu_data->erase(std::remove_if(imu_data->begin(), imu_data->end(),
                                      [oldest_time](const ImuData &sample) { return sample.timestamp < oldest_time; }),
                        imu_data->end());
    }
}

std::shared_ptr<InertialInitializer> InertialInitializer::make_attempt() {

  auto db = _db->clone();
  auto reset=reset_ctx->prior();
  auto attempt_params=params;
  if (reset.joint) {
    // The reset State rebases FEJ at its current calibration. Use that same chart
    // before any clock-dependent pruning or policy decision, not configuration.
    const auto &joint=*reset.joint;
    attempt_params.camera_intrinsics.clear(); attempt_params.camera_extrinsics.clear();
    attempt_params.camera_imu_dt.clear();
    for (const auto &camera:joint.calibration) {
      std::shared_ptr<CamBase> model;
      if(camera.fisheye) model=std::make_shared<CamEqui>(camera.width,camera.height);
      else model=std::make_shared<CamRadtan>(camera.width,camera.height);
      model->set_value(camera.intrinsics);
      attempt_params.camera_intrinsics.emplace(camera.camera_id,std::move(model));
      attempt_params.camera_extrinsics.emplace(camera.camera_id,camera.extrinsics);
      attempt_params.camera_imu_dt.emplace(camera.camera_id,camera.clock_mean);
    }
    attempt_params.calib_camimu_dt=joint.calibration.at(joint.reference_camera_id).clock_mean;
    attempt_params.init_imu_accel_map=joint.imu_accel_map;
    attempt_params.init_imu_gyro_map=joint.imu_gyro_map;
    attempt_params.init_imu_tg=joint.imu_tg;
    std::vector<InitObservationKey> previous;
    for(const auto &feature:db->get_internal_data())
      for(const auto &camera:feature.second->timestamps)
        for(double raw:camera.second)
          if(!physical_reset_future_row(joint,camera.first,raw))
            previous.push_back({feature.first,camera.first,raw});
    db->cleanup_measurements_exact_observations(previous);
  }
  if(params.init_dyn_reset_prior_use && reset.valid && reset.filter && !reset.joint) {
    if(!valid_filter_reset_prior(*reset.filter,params.num_cameras) || params.init_dyn_mle_opt_calib || params.init_dyn_fix_ba_on_reset)
      reset.valid=false;
    else {
      std::vector<InitObservationKey> previous;
      for(const auto &feature:db->get_internal_data())
        for(const auto &camera:feature.second->timestamps) {
          const auto offset=params.camera_imu_dt.find(camera.first);
          const double td=offset==params.camera_imu_dt.end() ? params.calib_camimu_dt : offset->second;
          for(double raw:camera.second)
            if(!marginal_reset_future_row(*reset.filter,camera.first,raw,td))previous.push_back({feature.first,camera.first,raw});
        }
      db->cleanup_measurements_exact_observations(previous);
    }
  }
  auto attempt = std::make_shared<InertialInitializer>(attempt_params, db);
  attempt->is_attempt = true;
  attempt->set_reset_prior(reset);

  // Get the newest and oldest timestamps we will try to initialize between!
  double newest_cam_time = -1;
  for (auto const &feat : db->get_internal_data()) {
    for (auto const &camtimepair : feat.second->timestamps) {
      for (auto const &time : camtimepair.second) {
        newest_cam_time = std::max(newest_cam_time, time);
      }
    }
  }
  double oldest_time = newest_cam_time - attempt_params.init_window_time - 0.10;
  InitializerCameraClock camera_clock;
  const bool valid_clock = camera_clock.configure(attempt_params.camera_imu_dt, attempt_params.calib_camimu_dt);

  // The private dynamic window is in the reference clock. Its newest time can
  // be as early as newest_raw + min_delta, and an observation inside it can
  // have raw time t_reference - max_delta. Retain that entire raw interval.
  // Equal offsets preserve the original cleanup boundary and sample sequence.
  const double raw_oldest_time = camera_clock.unequal_offsets
                                     ? oldest_time - (camera_clock.max_delta - camera_clock.min_delta)
                                     : oldest_time;
  const double imu_oldest_time = raw_oldest_time + camera_clock.min_imu_offset;
  const bool prune = valid_clock && newest_cam_time >= 0 && oldest_time >= 0 &&
                     finite_initializer_time(raw_oldest_time) && finite_initializer_time(imu_oldest_time);
  if (prune) {
    _db->cleanup_measurements(raw_oldest_time);
    db->cleanup_measurements(raw_oldest_time);
  }
  {
    std::lock_guard<std::mutex> lock(imu_data_mtx);
    if (prune) {
      auto it_imu = imu_data->begin();
      while (it_imu != imu_data->end() && it_imu->timestamp < imu_oldest_time)
        ++it_imu;
      // Keep an interpolation predecessor for the earliest retained observation.
      if (camera_clock.unequal_offsets && it_imu != imu_data->begin())
        --it_imu;
      imu_data->erase(imu_data->begin(), it_imu);
    }
    *attempt->imu_data = *imu_data;
  }
  return attempt;
}

bool InertialInitializer::request_physical_warmstart(const ov_core::InitPhysicalWarmRequest &request) {
#ifndef USE_CERES_FREE_INIT
  return false;
#else
  if (!is_attempt || !params.init_warmstart_inject || !params.init_dyn_use || params.init_dyn_mle_opt_calib ||
      !request.episode_id || !request.max_retained_owners ||
      (params.init_dyn_reset_prior_use && reset_ctx && reset_ctx->prior().valid && !reset_ctx->prior().joint)) return false;
  const auto joint=physical_reset_prior();
  if (request.reset_prior!=joint || (joint && !valid_physical_reset_prior(*joint))) return false;
  physical_request = std::make_unique<ov_core::InitPhysicalWarmRequest>(request);
  return true;
#endif
}

bool InertialInitializer::initialize(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<ov_type::Type>> &order,
                                     std::shared_ptr<ov_type::IMU> t_imu,
                                     std::map<double, std::shared_ptr<ov_type::PoseJPL>> &clones_IMU,
                                     std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> &features_SLAM, bool wait_for_jerk) {

  // The public synchronous API gets the same isolation as an owned async
  // attempt. In particular, disparity must never read shallow live features.
  if (!is_attempt)
    return make_attempt()->initialize(timestamp, covariance, order, t_imu, clones_IMU, features_SLAM, wait_for_jerk);

  if (physical_reset_prior() && (!physical_request || physical_request->reset_prior!=physical_reset_prior())) return false;
  double newest_cam_time = -1;
  for (const auto &feature : _db->get_internal_data())
    for (const auto &camera : feature.second->timestamps)
      for (double time : camera.second)
        newest_cam_time = std::max(newest_cam_time, time);
  if (newest_cam_time < 0 || newest_cam_time - params.init_window_time - 0.10 < 0)
    return false;
  InitializerCameraClock camera_clock;
  if (!camera_clock.configure(params.camera_imu_dt, params.calib_camimu_dt)) {
    PRINT_WARNING(YELLOW "[init]: invalid camera-to-IMU time offsets\n" RESET);
    return false;
  }

  // Compute the disparity of the system at the current timestep
  // If disparity is zero or negative we will always use the static initializer
  bool disparity_detected_moving_1to0 = false;
  bool disparity_detected_moving_2to1 = false;
  if (params.init_max_disparity > 0) {

    // Get the disparity statistics from this image to the previous
    // Only compute the disparity for the oldest half of the initialization period
    double newest_time_allowed = newest_cam_time - 0.5 * params.init_window_time;
    int num_features0 = 0;
    int num_features1 = 0;
    double avg_disp0, avg_disp1;
    double var_disp0, var_disp1;
    FeatureHelper::compute_disparity(_db, avg_disp0, var_disp0, num_features0, newest_time_allowed);
    FeatureHelper::compute_disparity(_db, avg_disp1, var_disp1, num_features1, newest_cam_time, newest_time_allowed);

    // Return if we can't compute the disparity
    int feat_thresh = 15;
    if (num_features0 < feat_thresh || num_features1 < feat_thresh) {
      PRINT_WARNING(YELLOW "[init]: not enough feats to compute disp: %d,%d < %d\n" RESET, num_features0, num_features1, feat_thresh);
      return false;
    }

    // Check if it passed our check!
    PRINT_INFO(YELLOW "[init]: disparity is %.3f,%.3f (%.2f thresh)\n" RESET, avg_disp0, avg_disp1, params.init_max_disparity);
    disparity_detected_moving_1to0 = (avg_disp0 > params.init_max_disparity);
    disparity_detected_moving_2to1 = (avg_disp1 > params.init_max_disparity);
  }

  // Use our static initializer!
  // CASE1: if our disparity says we were static in last window and have moved in the newest, we have a jerk
  // CASE2: if both disparities are below the threshold, then the platform has been stationary during both periods
  bool has_jerk = (!disparity_detected_moving_1to0 && disparity_detected_moving_2to1);
  bool is_still = (!disparity_detected_moving_1to0 && !disparity_detected_moving_2to1);
  if (((has_jerk && wait_for_jerk) || (is_still && !wait_for_jerk)) && params.init_imu_thresh > 0.0) {
    // A static/cold marginal cannot replace the selected correlated episode.
    if (physical_reset_prior()) return false;
    PRINT_DEBUG(GREEN "[init]: USING STATIC INITIALIZER METHOD!\n" RESET);
    // Static init produces no clones/features -> caller cold-starts (warm-start applies to dynamic only).
    clones_IMU.clear();
    features_SLAM.clear();
    StaticInitializer init_static(params, _db, imu_data);
    return init_static.initialize(timestamp, covariance, order, t_imu, wait_for_jerk);
  } else if (params.init_dyn_use && !is_still) {
    PRINT_DEBUG(GREEN "[init]: USING DYNAMIC INITIALIZER METHOD!\n" RESET);
    // Forward the caller's maps so the recovered window clones + the joint covariance survive (the
    // dynamic init previously dropped these locals here -> the filter always cold-started).
    DynamicInitializer init_dynamic(params, _db, imu_data, reset_ctx);
    if (physical_request) {
      auto result = std::make_unique<ov_core::InitPhysicalWarmResult>();
      if (!init_dynamic.initialize(timestamp, covariance, order, t_imu, clones_IMU, features_SLAM,
                                   physical_request.get(), result.get())) return false;
      physical_result = std::move(result);
      return true;
    }
    return init_dynamic.initialize(timestamp, covariance, order, t_imu, clones_IMU, features_SLAM);
  } else {
    std::string msg = (has_jerk) ? "" : "no accel jerk detected";
    msg += (has_jerk || is_still) ? "" : ", ";
    msg += (is_still) ? "" : "platform moving too much";
    PRINT_INFO(YELLOW "[init]: failed static init: %s\n" RESET, msg.c_str());
  }
  return false;
}
