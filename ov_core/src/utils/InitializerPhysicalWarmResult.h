/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_CORE_INITIALIZER_PHYSICAL_WARM_RESULT_H
#define OV_CORE_INITIALIZER_PHYSICAL_WARM_RESULT_H

#include <Eigen/Core>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace ov_core {

inline uint64_t initializer_time_bits(double time) {
  uint64_t bits;
  std::memcpy(&bits, &time, sizeof(bits));
  return bits;
}

struct InitObservationKey {
  size_t feature_id = 0;
  size_t camera_id = 0;
  double raw_time = 0.; // Original bits, never recovered by subtracting a clock.
  bool operator<(const InitObservationKey &other) const {
    if (feature_id != other.feature_id) return feature_id < other.feature_id;
    if (camera_id != other.camera_id) return camera_id < other.camera_id;
    return initializer_time_bits(raw_time) < initializer_time_bits(other.raw_time);
  }
};

struct InitExposureOwner {
  size_t camera_id = 0;
  double raw_time = 0.;
  double nominal_imu_time = 0.;
  uint32_t graph_node = 0;
  Eigen::Matrix<double, 7, 1> pose_mean = (Eigen::Matrix<double, 7, 1>() << 0., 0., 0., 1., 0., 0., 0.).finished();
  Eigen::Vector3d velocity_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d omega_body = Eigen::Vector3d::Zero();
};

struct InitFixedCameraCalibration {
  size_t camera_id = 0;
  double clock_mean = 0.;
  bool fisheye = false;
  int width = 0, height = 0;
  Eigen::Matrix<double, 7, 1> extrinsics;
  Eigen::Matrix<double, 8, 1> intrinsics;
};

// Semantic calibration keys survive a different live Type order/local-id layout.
// Fresh camera priors and explicit joint reset snapshots share these keys.
enum class InitCameraCalibrationKind { Clock, Extrinsics, Intrinsics };
struct InitCameraCalibrationBlock {
  size_t camera_id = 0;
  InitCameraCalibrationKind kind = InitCameraCalibrationKind::Clock;
  Eigen::VectorXd mean, fej;
  int local_size() const {
    switch (kind) {
      case InitCameraCalibrationKind::Clock: return 1;
      case InitCameraCalibrationKind::Extrinsics: return 6;
      case InitCameraCalibrationKind::Intrinsics: return 8;
    }
    return 0;
  }
  int value_size() const { return kind == InitCameraCalibrationKind::Extrinsics ? 7 : local_size(); }
};

// One immutable reset marginal. Its camera chart is retained into the new
// episode, with FEJ explicitly rebased to the current mean. A new likelihood
// must be later than BOTH the physical endpoint and its camera's raw watermark.
struct InitPhysicalResetPrior {
  uint64_t snapshot_id = 0;
  double imu_endpoint = 0.;
  double imu_raw_cutoff = 0.; // Last raw support used at the accepted endpoint.
  Eigen::Matrix<double,6,1> bias_mean = Eigen::Matrix<double,6,1>::Zero();
  Eigen::Matrix<double,6,6> bias_covariance = Eigen::Matrix<double,6,6>::Zero();
  Eigen::MatrixXd bias_calibration_covariance;
  Eigen::Matrix<double,6,1> bias_rw_variance = Eigen::Matrix<double,6,1>::Zero();
  std::vector<InitCameraCalibrationBlock> consider;
  Eigen::MatrixXd calibration_covariance;
  std::vector<InitFixedCameraCalibration> calibration;
  std::vector<double> raw_watermarks;
  size_t reference_camera_id = 0;
  Eigen::Matrix3d imu_accel_map = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d imu_gyro_map = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d imu_tg = Eigen::Matrix3d::Zero();
  int cause = 0;
};

struct InitPhysicalWarmRequest {
  uint64_t episode_id = 0; // Nonzero, owned and checked by the caller.
  size_t max_retained_owners = 0;
  // Empty preserves the existing fixed-calibration contract. Otherwise the
  // entire uncertain camera calibration state is retained, with unchanged
  // means/FEJ and full covariance (including off-diagonal/singular support).
  std::vector<InitCameraCalibrationBlock> consider;
  Eigen::MatrixXd calibration_covariance;
  std::shared_ptr<const InitPhysicalResetPrior> reset_prior;
};

/** Global-shutter, fixed-mean conditional initializer result.
 *
 * Ordering is IMU15, 6D owners in physical-time/camera/raw-bit order, then the
 * semantic consider calibration order. Calibration means/FEJ/Pcc are retained.
 * Coincident owners are distinct output rows and may have a singular covariance.
 * Includes the gravity-dependent output map and declared navigation inflation.
 * The optional reset snapshot retains the bias/camera marginal and requires
 * future-only image/IMU support. Estimated IMU/readout calibration, fitted
 * calibration means, a shared sampled-noise boundary, bridge Q and another
 * update from the consumed pixels remain unsupported.
 */
struct InitPhysicalWarmResult {
  uint64_t episode_id = 0;
  double accepted_imu_endpoint = 0.;
  double reference_clock_label = 0.;
  double reference_clock_mean = 0.;
  uint32_t graph_node_count = 0;
  Eigen::Matrix<double, 16, 1> imu_mean;
  std::vector<InitExposureOwner> owners;
  std::vector<InitObservationKey> consumed_observations;
  std::vector<InitFixedCameraCalibration> calibration;
  Eigen::Matrix3d imu_accel_map = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d imu_gyro_map = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d imu_tg = Eigen::Matrix3d::Zero();
  std::vector<InitCameraCalibrationBlock> consider;
  Eigen::MatrixXd calibration_covariance;
  std::shared_ptr<const InitPhysicalResetPrior> reset_prior;
  double reset_first_imu_time = 0.;
  double reset_first_imu_support_time = 0.;
  // Congruence applied to each output pose's [theta,p] errors. Calibration is
  // never inflated. Needed to validate stochastic same-node owner identities.
  Eigen::Matrix<double, 6, 1> pose_error_scale = Eigen::Matrix<double, 6, 1>::Ones();
  Eigen::MatrixXd joint_covariance;
};

} // namespace ov_core
#endif
