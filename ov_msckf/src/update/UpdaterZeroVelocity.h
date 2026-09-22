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

#ifndef OV_MSCKF_UPDATER_ZEROVELOCITY_H
#define OV_MSCKF_UPDATER_ZEROVELOCITY_H

#include <memory>
#include <utility>
#include <vector>

#include "utils/sensor_data.h"

#include "UpdaterOptions.h"
#include "state/State.h"
#include "utils/NoiseManager.h"

namespace ov_core {
class Feature;
class FeatureDatabase;
} // namespace ov_core
namespace ov_type {
class Landmark;
} // namespace ov_type

namespace ov_msckf {

class State;
class Propagator;

/**
 * @brief Will try to *detect* and then update using zero velocity assumption.
 *
 * Consider the case that a VIO unit remains stationary for a period time.
 * Typically this can cause issues in a monocular system without SLAM features since no features can be triangulated.
 * Additional, if features could be triangulated (e.g. stereo) the quality can be poor and hurt performance.
 * If we can detect the cases where we are stationary then we can leverage this to prevent the need to do feature update during this period.
 * The main application would be using this on a **wheeled vehicle** which needs to stop (e.g. stop lights or parking).
 */
class UpdaterZeroVelocity {

public:
  /**
   * @brief Default constructor for our zero velocity detector and updater.
   * @param options Updater options (chi2 multiplier)
   * @param noises imu noise characteristics (continuous time)
   * @param db Feature tracker database with all features in it
   * @param prop Propagator class object which can predict the state forward in time
   * @param gravity_mag Global gravity magnitude of the system (normally 9.81)
   * @param zupt_max_velocity Max velocity we should consider to do a update with
   * @param zupt_noise_multiplier Multiplier of our IMU noise matrix (default should be 1.0)
   * @param zupt_max_disparity Max disparity we should consider to do a update with
   * @param prop_window Time window (seconds) to keep IMU measurements beyond the oldest needed time
   */
  UpdaterZeroVelocity(UpdaterOptions &options, NoiseManager &noises, std::shared_ptr<ov_core::FeatureDatabase> db,
                      std::shared_ptr<Propagator> prop, double gravity_mag, double zupt_max_velocity, double zupt_noise_multiplier,
                      double zupt_max_disparity, double prop_window = 0.10);

  /**
   * @brief Feed function for inertial data
   * @param message Contains our timestamp and inertial information
   * @param oldest_time Time that we can discard measurements before
   */
  void feed_imu(const ov_core::ImuData &message, double oldest_time = -1) {

    // Append it to our vector
    imu_data.emplace_back(message);

    // Sort our imu data (handles any out of order measurements)
    // std::sort(imu_data.begin(), imu_data.end(), [](const IMUDATA i, const IMUDATA j) {
    //    return i.timestamp < j.timestamp;
    //});

    // Clean old measurements
    // std::cout << "ZVUPT: imu_data.size() " << imu_data.size() << std::endl;
    clean_old_imu_measurements(oldest_time - _prop_window);
  }

  /**
   * @brief This will remove any IMU measurements that are older then the given measurement time
   * @param oldest_time Time that we can discard measurements before (in IMU clock)
   */
  void clean_old_imu_measurements(double oldest_time) {
    if (oldest_time < 0)
      return;
    auto it0 = imu_data.begin();
    while (it0 != imu_data.end()) {
      if (it0->timestamp < oldest_time) {
        it0 = imu_data.erase(it0);
      } else {
        it0++;
      }
    }
  }

  /**
   * @brief Will first detect if the system is zero velocity, then will update.
   * @param state State of the filter
   * @param timestamp Next camera timestamp we want to see if we should propagate to.
   * @return True if the system is currently at zero velocity
   */
  bool try_update(std::shared_ptr<State> state, double timestamp);

  using RawCameraKeys = std::vector<std::pair<size_t, double>>;

  /// Explicit physical endpoint; the reference label is committed only on
  /// acceptance. Physical-clone mode requires the current owner raw keys so
  /// disparity and accepted-track cleanup retain camera-specific ownership.
  bool try_update_at_imu(std::shared_ptr<State> state, double target_imu, double reference_timestamp,
                         const RawCameraKeys &raw_keys = {});

  struct SampledStationaryFactor {
    // Columns: current attitude, raw bg, raw ba, original raw sample noise6.
    Eigen::Matrix<double, 6, 15> H = Eigen::Matrix<double, 6, 15>::Zero();
    Eigen::Matrix<double, 6, 1> residual = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> independent_R = Eigen::Matrix<double, 6, 6>::Zero();
  };

  /// Disabled-runtime proof API. Linearize one admitted ORIGINAL raw record at
  /// its exact accepted knot, with fixed IMU calibration and do_fej=false.
  /// independent_stationarity_covariance is an explicit model discrepancy;
  /// sensor uncertainty is already owned by the sample Vec6 and is not R.
  /// Zero independent covariance is supported. Refusal leaves out unchanged.
  bool linearize_sampled_at_knot(std::shared_ptr<State> state, const State::SampledImuRecord &record,
                                const Eigen::Matrix<double, 6, 6> &independent_stationarity_covariance,
                                SampledStationaryFactor &out) const;

  /// Apply the same-knot factor once, using the existing chi-square/velocity
  /// gates and the full state/sample posterior. No interval bias Q, camera
  /// history, track cleanup or timestamp advancement occurs here. The sampled
  /// Propagator must already have evolved the state to this knot. The caller
  /// serializes State access and restores this updater together with State.
  /// Unsupported/numeric/gate rejection leaves state, receipt, cache and out
  /// unchanged. No runtime configuration invokes this proof caller.
  bool try_update_sampled_at_knot(std::shared_ptr<State> state, const State::SampledImuRecord &record,
                                 const Eigen::Matrix<double, 6, 6> &independent_stationarity_covariance,
                                 SampledStationaryFactor *out = nullptr);

  struct CameraHistory {
    double previous_raw = 0.0;
    double accepted_raw = 0.0;
    bool has_previous = false;
    int accepted_count = 0;
  };

  struct Snapshot {
    std::vector<ov_core::ImuData> imu_data;
    std::vector<CameraHistory> camera_history;
    double last_prop_time_offset = 0.0;
    bool have_last_prop_time_offset = false;
    double last_zupt_state_timestamp = 0.0;
    int last_zupt_count = 0;
    uint64_t sampled_stream_episode = 0;
    uint64_t sampled_last_sequence = 0;
    uint64_t sampled_last_timestamp_bits = 0;
  };

  /// Serialized estimator lifecycle helpers; retain buffered raw IMU on reset.
  void reset_for_new_state();
  Snapshot capture() const;
  void restore(const Snapshot &snapshot);

  /**
   * @brief Feed a batch of IMU measurements with a single lock
   * @param messages Vector of IMU measurements to process
   * @param oldest_time Oldest timestamp to keep measurements for
   */
  void feed_imu_batch(const std::vector<ov_core::ImuData>& messages, double oldest_time = -1);

protected:
  /// Options used during update (chi2 multiplier)
  UpdaterOptions _options;

  /// Container for the imu noise values
  NoiseManager _noises;

  /// Feature tracker database with all features in it
  std::shared_ptr<ov_core::FeatureDatabase> _db;

  /// Our propagator!
  std::shared_ptr<Propagator> _prop;

  /// Gravity vector
  Eigen::Vector3d _gravity;

  /// Max velocity (m/s) that we should consider a zupt with
  double _zupt_max_velocity = 1.0;

  /// Multiplier of our IMU noise matrix (default should be 1.0)
  double _zupt_noise_multiplier = 1.0;

  /// Max disparity (pixels) that we should consider a zupt with
  double _zupt_max_disparity = 1.0;

  /// Time window (seconds) to keep IMU measurements beyond the oldest needed time
  double _prop_window = 0.10;

  /// Our history of IMU messages (time, angular, linear)
  std::vector<ov_core::ImuData> imu_data;

  /// Dense camera-ID history, allocated on first covered physical-mode request.
  std::vector<CameraHistory> camera_history;

  /// Estimate for time offset at last propagation time
  double last_prop_time_offset = 0.0;
  bool have_last_prop_time_offset = false;

  /// Last timestamp we did zero velocity update with
  double last_zupt_state_timestamp = 0.0;

  /// Number of times we have called update
  int last_zupt_count = 0;

  /// One-use likelihood receipt for this updater's current State episode.
  /// Vec slot pointers are reused, so only immutable sample identity is valid.
  uint64_t sampled_stream_episode = 0;
  uint64_t sampled_last_sequence = 0;
  uint64_t sampled_last_timestamp_bits = 0;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_ZEROVELOCITY_H
