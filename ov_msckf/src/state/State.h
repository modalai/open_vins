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

#ifndef OV_MSCKF_STATE_H
#define OV_MSCKF_STATE_H

#include "utils/finite.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "PreintegrationBridge.h"
#include "StateOptions.h"
#include "cam/CamBase.h"
#include "types/IMU.h"
#include "types/Landmark.h"
#include "types/PoseJPL.h"
#include "types/Type.h"
#include "types/Vec.h"
#include "utils/InitializerPhysicalWarmResult.h"

namespace ov_msckf {

/**
 * @brief State of our filter
 *
 * This state has all the current estimates for the filter.
 * This system is modeled after the MSCKF filter, thus we have a sliding window of clones.
 * We additionally have more parameters for online estimation of calibration and SLAM features.
 * We also have the covariance of the system, which should be managed using the StateHelper class.
 */
class State {

public:
  /// Immutable first-use description of one original raw-axis IMU record.
  /// This ownership foundation does not enable a sampled-noise runtime mode.
  struct SampledImuRecord {
    uint64_t stream_episode = 0, sequence = 0;
    double timestamp = 0.;
    Eigen::Matrix<double, 6, 1> measured = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> prior = Eigen::Matrix<double, 6, 6>::Zero();
  };

  struct SampledImuSlot {
    bool active = false;
    SampledImuRecord record;
    // Always allocated in the covariance after explicit preparation. Inactive
    // slots have zero mean, FEJ, covariance and cross rows. FEJ remains zero in
    // the raw noise chart; value is the inferred sample-noise posterior mean.
    std::shared_ptr<ov_type::Vec> noise;
  };

  bool has_sampled_imu_boundary() const { return _sampled_imu_stream_episode != 0; }
  const std::array<SampledImuSlot, 2> &sampled_imu_slots() const { return _sampled_imu_slots; }

  /// Per-clone kinematic metadata (velocity and corrected angular rate at clone time, plus their
  /// FEJ twins). Consumed by the per-camera time-offset / rolling-shutter measurement models as the
  /// CLONE-SIDE Jacobian linearization point. Metadata only -- never in the state vector/covariance.
  ///
  /// Design rationale: freezing these at augment time is exactly what FEJ prescribes for
  /// clone-anchored Jacobians (the fej twins), and cheaper by ~2x in clone-block covariance work
  /// than carrying velocity inside each clone. Value-side pose shifts to measurement time must NOT
  /// rely on the cached velocity over long gaps -- the preintegration bridge integrates the actual
  /// IMU samples for that; the cached values remain only linearization points. omega is the
  /// intrinsics/bias-corrected gyro at the clone instant (select_imu_readings boundary-interpolated),
  /// matching the dnc_dt = [w; v] augment Jacobian. Lifecycle: augment_clone stores, marginalize
  /// erases, warmstart restore rebuilds by finite differences; consumers must use tolerant lookups.
  struct CloneKinematics {
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();
    Eigen::Vector3d vel_fej = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega_fej = Eigen::Vector3d::Zero();
  };

  /// One stochastic exposure view. Even equal nominal exposure times have
  /// separate views when their camera clock errors are independent. Raw keys
  /// never change after insertion; the view's ordinary EKF correction carries
  /// its clock correlation. Entries are retained in physical creation order.
  struct ExposurePose {
    size_t camera_id = 0;
    double raw_time = -1.0;
    double imu_time = -1.0;
    std::shared_ptr<ov_type::PoseJPL> pose;
    CloneKinematics kinematics;
  };

  bool uses_physical_clones() const { return _options.physical_camera_clones; }
  size_t clone_count() const { return uses_physical_clones() ? _exposure_poses.size() : _clones_IMU.size(); }

  /// A pre-first-propagation state may still carry the initializer's camera
  /// clock label. After any accepted propagation/ZUPT, the stored endpoint is
  /// authoritative and cannot be relabeled by a camera-clock EKF update.
  double imu_endpoint() const {
    return _imu_endpoint_valid ? _imu_endpoint : _timestamp + cam_imu_dt_ref();
  }

  std::shared_ptr<ov_type::PoseJPL> find_pose(size_t camera_id, double raw_time) const {
    if (!uses_physical_clones()) {
      auto it = _clones_IMU.find(raw_time);
      return it == _clones_IMU.end() ? nullptr : it->second;
    }
    for (const auto &view : _exposure_poses)
      if (view.camera_id == camera_id && ov_core::initializer_time_bits(view.raw_time) == ov_core::initializer_time_bits(raw_time))
        return view.pose;
    return nullptr;
  }

  std::shared_ptr<ov_type::PoseJPL> pose_for_camera(size_t camera_id, double raw_time) const {
    auto pose = find_pose(camera_id, raw_time);
    if (!pose)
      throw std::out_of_range("camera observation has no retained exposure pose");
    return pose;
  }

  const CloneKinematics *clone_kinematics(size_t camera_id, double raw_time) const {
    if (!uses_physical_clones()) {
      auto it = _clones_kinematics.find(raw_time);
      return it == _clones_kinematics.end() ? nullptr : &it->second;
    }
    for (const auto &view : _exposure_poses)
      if (view.camera_id == camera_id && ov_core::initializer_time_bits(view.raw_time) == ov_core::initializer_time_bits(raw_time))
        return &view.kinematics;
    return nullptr;
  }

  /// Bounded allocation-free traversal. Legacy mode exposes its shared clone
  /// window to every camera; physical mode exposes only that camera's owners.
  template <typename Fn> void for_each_clone(size_t camera_id, Fn fn) const {
    if (uses_physical_clones()) {
      for (const auto &view : _exposure_poses)
        if (view.camera_id == camera_id)
          fn(view.raw_time, view.pose);
    } else {
      for (const auto &entry : _clones_IMU)
        fn(entry.first, entry.second);
    }
  }

  double latest_clone_time(size_t camera_id) const {
    double time = -std::numeric_limits<double>::infinity();
    for_each_clone(camera_id, [&](double t, const std::shared_ptr<ov_type::PoseJPL> &) { time = std::max(time, t); });
    return time;
  }

  /**
   * @brief Default Constructor (will initialize variables to defaults)
   * @param options_ Options structure containing filter options
   */
  State(StateOptions &options_);

  ~State() {}

  /**
   * @brief Will return the timestep that we will marginalize next.
   * As of right now, since we are using a sliding window, this is the oldest clone.
   * But if you wanted to do a keyframe system, you could selectively marginalize clones.
   * @return timestep of clone we will marginalize
   */
  double margtimestep() {
    std::lock_guard<std::mutex> lock(_mutex_state);
    if (uses_physical_clones())
      return _exposure_poses.empty() ? INFINITY : _exposure_poses.front().raw_time;
    double time = INFINITY;
    for (const auto &clone_imu : _clones_IMU) {
      if (clone_imu.first < time) {
        time = clone_imu.first;
      }
    }
    return time;
  }

  /**
   * @brief Calculates the current max size of the covariance
   * @return Size of the current covariance matrix
   */
  int max_covariance_size() { return (int)_Cov.rows(); }

  /**
   * @brief Gyroscope and accelerometer intrinsic matrix (scale imperfection and axis misalignment)
   *
   * If kalibr model, lower triangular of the matrix is used
   * If rpng model, upper triangular of the matrix is used
   *
   * @return 3x3 matrix of current imu gyroscope / accelerometer intrinsics
   */
  static Eigen::Matrix3d Dm(StateOptions::ImuModel imu_model, const Eigen::MatrixXd &vec) {
    assert(vec.rows() == 6);
    assert(vec.cols() == 1);
    Eigen::Matrix3d D_matrix = Eigen::Matrix3d::Identity();
    if (imu_model == StateOptions::ImuModel::KALIBR) {
      D_matrix << vec(0), 0, 0, vec(1), vec(3), 0, vec(2), vec(4), vec(5);
    } else {
      D_matrix << vec(0), vec(1), vec(3), 0, vec(2), vec(4), 0, 0, vec(5);
    }
    return D_matrix;
  }

  /**
   * @brief Gyroscope gravity sensitivity
   *
   * For both kalibr and rpng models, this a 3x3 that is column-wise filled.
   *
   * @return 3x3 matrix of current gravity sensitivity
   */
  static Eigen::Matrix3d Tg(const Eigen::MatrixXd &vec) {
    assert(vec.rows() == 9);
    assert(vec.cols() == 1);
    Eigen::Matrix3d Tg = Eigen::Matrix3d::Zero();
    Tg << vec(0), vec(3), vec(6), vec(1), vec(4), vec(7), vec(2), vec(5), vec(8);
    return Tg;
  }

  /**
   * @brief Calculates the error state size for imu intrinsics.
   *
   * This is used to construct our state transition which depends on if we are estimating calibration.
   * 15 if doing intrinsics, another +9 if doing grav sensitivity
   *
   * @return size of error state
   */
  int imu_intrinsic_size() const {
    int sz = 0;
    if (_options.do_calib_imu_intrinsics) {
      sz += 15;
      if (_options.do_calib_imu_g_sensitivity) {
        sz += 9;
      }
    }
    return sz;
  }

  /// Returns the camera id used as propagation/cloning time reference
  int cam_imu_dt_ref_camid() const { return _options.cam_imu_dt_ref_camid; }

  /// Returns per-camera timeoffset variable (falls back to reference variable if camera id not found)
  std::shared_ptr<ov_type::Vec> cam_imu_dt_var(size_t cam_id) const {
    auto it = _calib_dt_CAMtoIMU_map.find(cam_id);
    if (it != _calib_dt_CAMtoIMU_map.end()) {
      return it->second;
    }
    return _calib_dt_CAMtoIMU;
  }

  /// Returns current per-camera dt value
  double cam_imu_dt(size_t cam_id) const { return cam_imu_dt_var(cam_id)->value()(0); }

  /// Returns FEJ per-camera dt value
  double cam_imu_dt_fej(size_t cam_id) const { return cam_imu_dt_var(cam_id)->fej()(0); }

  /// Returns current reference camera dt value
  double cam_imu_dt_ref() const { return cam_imu_dt((size_t)cam_imu_dt_ref_camid()); }

  /// Returns FEJ reference camera dt value
  double cam_imu_dt_ref_fej() const { return cam_imu_dt_fej((size_t)cam_imu_dt_ref_camid()); }

  /// Returns delta dt between camera and reference camera
  double cam_imu_dt_delta(size_t cam_id) const { return cam_imu_dt(cam_id) - cam_imu_dt_ref(); }

  /// Returns FEJ delta dt between camera and reference camera
  double cam_imu_dt_delta_fej(size_t cam_id) const { return cam_imu_dt_fej(cam_id) - cam_imu_dt_ref_fej(); }

  /// Returns min dt across all cameras
  double cam_imu_dt_min() const {
    double dt_min = std::numeric_limits<double>::infinity();
    for (int i = 0; i < _options.num_cameras; i++) {
      dt_min = std::min(dt_min, cam_imu_dt((size_t)i));
    }
    return ov_core::numeric::finite(dt_min) ? dt_min : cam_imu_dt_ref();
  }

  /// Returns max dt among specific camera ids
  double cam_imu_dt_max_for_ids(const std::vector<int> &cam_ids) const {
    if (cam_ids.empty()) {
      return cam_imu_dt_ref();
    }
    double dt_max = -std::numeric_limits<double>::infinity();
    for (const int cam_id : cam_ids) {
      if (cam_id < 0) {
        continue;
      }
      dt_max = std::max(dt_max, cam_imu_dt((size_t)cam_id));
    }
    return ov_core::numeric::finite(dt_max) ? dt_max : cam_imu_dt_ref();
  }

  /**
   * @brief True when the current window's motion cannot observe the temporal calibration states.
   *
   * MVIS-style degenerate-motion gate: under (near-)static, constant-velocity, or pure-slow-rotation
   * windows the cam-IMU time offsets / readout are weakly observable and their estimates wander,
   * injecting correlated error (hover is the canonical case). The gate checks the CLONE WINDOW's
   * excitation from the stored kinematics: peak |omega| and the velocity spread (a window-scale
   * acceleration proxy). StateHelper freezes dt/readout gain ROWS while degenerate (a Schmidt
   * update). The full measurement model and their prior uncertainty remain in the update.
   */
  bool dt_calib_degenerate() const {
    if (!_options.dt_calib_gate) {
      return false;
    }
    if (uses_physical_clones()) {
      if (_exposure_poses.empty()) {
        return true;
      }
      double omega_max = 0.0;
      Eigen::Vector3d vel_mean = Eigen::Vector3d::Zero();
      for (const auto &view : _exposure_poses) {
        omega_max = std::max(omega_max, view.kinematics.omega.norm());
        vel_mean += view.kinematics.vel;
      }
      if (omega_max >= _options.dt_calib_gate_min_omega) {
        return false;
      }
      vel_mean /= static_cast<double>(_exposure_poses.size());
      double vel_spread = 0.0;
      for (const auto &view : _exposure_poses) {
        vel_spread = std::max(vel_spread, (view.kinematics.vel - vel_mean).norm());
      }
      return vel_spread < _options.dt_calib_gate_min_vel_spread;
    }
    if (_clones_kinematics.empty()) {
      return true; // no evidence of excitation -> freeze
    }
    double omega_max = 0.0;
    Eigen::Vector3d vel_mean = Eigen::Vector3d::Zero();
    for (const auto &kv : _clones_kinematics) {
      omega_max = std::max(omega_max, kv.second.omega.norm());
      vel_mean += kv.second.vel;
    }
    if (omega_max >= _options.dt_calib_gate_min_omega) {
      return false;
    }
    vel_mean /= (double)_clones_kinematics.size();
    double vel_spread = 0.0;
    for (const auto &kv : _clones_kinematics) {
      vel_spread = std::max(vel_spread, (kv.second.vel - vel_mean).norm());
    }
    return vel_spread < _options.dt_calib_gate_min_vel_spread;
  }

  /// Count of measurement-model lookups that found no clone kinematics (telemetry; expected only
  /// briefly after a warm-started reset window; the consumers degrade to zero correction/columns)
  uint64_t _kin_miss_count = 0;

  /// Mutex for locking access to the state
  std::mutex _mutex_state;

  /// Current timestamp (should be the last update time in camera clock frame!)
  double _timestamp = -1;

  double _imu_endpoint = -1.0;
  bool _imu_endpoint_valid = false;

  /// Accepted physical initialization transaction, copied with snapshots.
  uint64_t _initialization_episode_id = 0;

  /// Struct containing filter options
  StateOptions _options;

  /// Pointer to the "active" IMU state (q_GtoI, p_IinG, v_IinG, bg, ba)
  std::shared_ptr<ov_type::IMU> _imu;

  /// Map between imaging times and clone poses (q_GtoIi, p_IiinG)
  std::map<double, std::shared_ptr<ov_type::PoseJPL>> _clones_IMU;

  /// Bounded owner registry for physical-camera mode. StateHelper owns pose
  /// covariance insertion/removal; VioManager owns the raw observation aliases.
  std::vector<ExposurePose> _exposure_poses;

  /// Our current set of SLAM features (3d positions)
  std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> _features_SLAM;

  /// Time offset base IMU to camera (t_imu = t_cam + t_off); ALIASES the reference camera's
  /// entry in _calib_dt_CAMtoIMU_map (kept for the single-offset call sites)
  std::shared_ptr<ov_type::Vec> _calib_dt_CAMtoIMU;

  /// Per-camera time offset base IMU to camera (t_imu = t_cam_i + t_off_i)
  std::unordered_map<size_t, std::shared_ptr<ov_type::Vec>> _calib_dt_CAMtoIMU_map;

  /// Rolling shutter readout time per camera (seconds, full frame top to bottom; 0 = global
  /// shutter). CENTERED convention: row v samples at stamp + (v/h - 0.5)*readout, so image
  /// timestamps MUST anchor mid-frame (producers stamp HAL3 SOF + (readout + exposure)/2)
  std::unordered_map<size_t, std::shared_ptr<ov_type::Vec>> _calib_camera_readout;

  /// Kinematic metadata at each clone time (metadata, not in state/covariance)
  std::map<double, CloneKinematics> _clones_kinematics;

  /// Epoch-mode KNOWN time residuals: for a frame of camera c snapped onto the epoch clone at
  /// time t, _epoch_residuals[t][c] = t_raw - t_epoch (its true sampling instant relative to the
  /// clone, in the camera clock). Consumed additively in the measurement models' dt_total.
  /// Metadata only; erased with the clone at marginalization.
  std::map<double, std::map<size_t, double>> _epoch_residuals;

  /// KNOWN epoch time residual for (camera, clone time); 0 when none was recorded
  double epoch_residual(size_t cam_id, double clone_time) const {
    if (uses_physical_clones())
      return 0.0;
    auto it = _epoch_residuals.find(clone_time);
    if (it == _epoch_residuals.end()) {
      return 0.0;
    }
    auto it2 = it->second.find(cam_id);
    return (it2 == it->second.end()) ? 0.0 : it2->second;
  }

  /// ACI2 preintegration bridges for epoch-snapped frames, keyed like _epoch_residuals.
  /// Built once per (camera, epoch) at bind time; erased with the clone.
  std::map<double, std::map<size_t, PreintBridgeData>> _epoch_bridges;

  /// Bridge lookup for (camera, clone time); nullptr when none exists (first-order fallback)
  const PreintBridgeData *epoch_bridge(size_t cam_id, double clone_time) const {
    if (uses_physical_clones())
      return nullptr;
    auto it = _epoch_bridges.find(clone_time);
    if (it == _epoch_bridges.end()) {
      return nullptr;
    }
    auto it2 = it->second.find(cam_id);
    return (it2 == it->second.end() || !it2->second.valid) ? nullptr : &it2->second;
  }

  /// Calibration poses for each camera (R_ItoC, p_IinC)
  std::unordered_map<size_t, std::shared_ptr<ov_type::PoseJPL>> _calib_IMUtoCAM;

  /// Camera intrinsics
  std::unordered_map<size_t, std::shared_ptr<ov_type::Vec>> _cam_intrinsics;

  /// Camera intrinsics camera objects
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> _cam_intrinsics_cameras;

  /// Gyroscope IMU intrinsics (scale imperfection and axis misalignment)
  std::shared_ptr<ov_type::Vec> _calib_imu_dw;

  /// Accelerometer IMU intrinsics (scale imperfection and axis misalignment)
  std::shared_ptr<ov_type::Vec> _calib_imu_da;

  /// Gyroscope gravity sensitivity
  std::shared_ptr<ov_type::Vec> _calib_imu_tg;

  /// Rotation from gyroscope frame to the "IMU" accelerometer frame (kalibr model)
  std::shared_ptr<ov_type::JPLQuat> _calib_imu_GYROtoIMU;

  /// Rotation from accelerometer to the "IMU" gyroscope frame frame (rpng model)
  std::shared_ptr<ov_type::JPLQuat> _calib_imu_ACCtoIMU;

protected:
  /// Reusable Schmidt-update scratch, not estimator state. Allocated only on a requested freeze;
  /// bounded by two scalar temporal variables per configured camera. Rebuilt after a snapshot.
  std::vector<int> _temporal_schmidt_ids;
  std::vector<double> _temporal_schmidt_prior;

private:
  // Define that the state helper is a friend class of this class
  // This will allow it to access the below functions which should normally not be called
  // This prevents a developer from thinking that the "insert clone" will actually correctly add it to the covariance
  friend class StateHelper;

  // Two permanent covariance slots avoid allocation and dimension changes at
  // every raw knot. Only StateHelper admits/retires immutable sample metadata.
  std::array<SampledImuSlot, 2> _sampled_imu_slots;
  uint64_t _sampled_imu_stream_episode = 0;
  uint64_t _sampled_imu_last_sequence = 0;
  double _sampled_imu_last_timestamp = 0.;

  /// Covariance of all active variables
  Eigen::MatrixXd _Cov;

  /// Vector of variables
  std::vector<std::shared_ptr<ov_type::Type>> _variables;
};

} // namespace ov_msckf

#endif // OV_MSCKF_STATE_H
