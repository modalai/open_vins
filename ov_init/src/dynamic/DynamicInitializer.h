/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
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

#ifndef OV_INIT_DYNAMICINITIALIZER_H
#define OV_INIT_DYNAMICINITIALIZER_H

#include "init/InertialInitializerOptions.h"
#include "init/ResetPrior.h"

namespace ov_core {
class FeatureDatabase;
struct ImuData;
class Feature;
class CpiV1;
} // namespace ov_core
namespace ov_type {
class Type;
class IMU;
class PoseJPL;
class Landmark;
class Vec;
} // namespace ov_type

namespace ov_init {

/**
 * @brief Initializer for a dynamic visual-inertial system.
 *
 * This implementation that will try to recover the initial conditions of the system.
 * Additionally, we will try to recover the covariance of the system.
 * To initialize with arbitrary motion:
 * 1. Preintegrate our system to get the relative rotation change (biases assumed known)
 * 2. Construct linear system with features to recover velocity (solve with |g| constraint)
 * 3. Perform a large MLE with all calibration and recover the covariance.
 *
 * Method is based on this work (see this [tech report](https://pgeneva.com/downloads/reports/tr_init.pdf) for a high level walk through):
 *
 * > Dong-Si, Tue-Cuong, and Anastasios I. Mourikis.
 * > "Estimator initialization in vision-aided inertial navigation with unknown camera-IMU calibration."
 * > 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2012.
 *
 * - https://ieeexplore.ieee.org/abstract/document/6386235
 * - https://tdongsi.github.io/download/pubs/2011_VIO_Init_TR.pdf
 * - https://pgeneva.com/downloads/reports/tr_init.pdf
 *
 */
class DynamicInitializer {
public:
  /**
   * @brief Default constructor
   * @param params_ Parameters loaded from either ROS or CMDLINE
   * @param db Feature tracker database with all features in it
   * @param imu_data_ Shared pointer to our IMU vector of historical information
   * @param reset_ctx_ Optional soft-reset hand-off context (live bias prior episode)
   */
  explicit DynamicInitializer(const InertialInitializerOptions &params_, std::shared_ptr<ov_core::FeatureDatabase> db,
                              std::shared_ptr<std::vector<ov_core::ImuData>> imu_data_,
                              std::shared_ptr<ResetContext> reset_ctx_ = nullptr)
      : params(params_), _db(db), imu_data(imu_data_), reset_ctx(reset_ctx_) {}

  /**
   * @brief Try to get the initialized system
   *
   * @param[out] timestamp Timestamp we have initialized the state at (last imu state)
   * @param[out] covariance Calculated covariance of the returned state
   * @param[out] order Order of the covariance matrix
   * @param _imu Pointer to the "active" IMU state (q_GtoI, p_IinG, v_IinG, bg, ba)
   * @param _clones_IMU Map between imaging times and clone poses (q_GtoIi, p_IiinG)
   * @param _features_SLAM Our current set of SLAM features (3d positions)
   * @return True if we have successfully initialized our system
   */
  bool initialize(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<ov_type::Type>> &order,
                  std::shared_ptr<ov_type::IMU> &_imu, std::map<double, std::shared_ptr<ov_type::PoseJPL>> &_clones_IMU,
                  std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> &_features_SLAM);

private:
  /**
   * @brief Feature-less linear seed (sqrtVINS Stage-A style): recovers ONLY [v_I0, gravity_I0]
   * from a 6x6 system -- no feature positions in the state, O(pairs x co-observed) assembly.
   *
   * Per keyframe pair (and camera combo), the gyro-preintegrated rotation makes each co-observed
   * feature's epipolar-plane normal n = b_i x b_j (bearings in I0) perpendicular to the camera
   * translation direction t: the smallest eigenvector of M = sum n n^T gives t, and the other two
   * eigenvectors e annihilate the unknown scale, so projecting the preintegrated position equation
   * onto e yields 2 scale-free rows per pair in [v, g]. |g| is enforced with the same Dong-Si
   * polynomial machinery on the velocity-eliminated 3x3 system, then g is projected to the sphere.
   * Robust to low parallax (features are never inverted) -- this is the reset-window fast path.
   *
   * Gates (init_dyn_fl_*): per-pair min co-observed features + M eigen-ratio (t well defined) +
   * 3sigma-MAD outlier filter on n^T t (one refit); global min pairs, cond(N), |g| tolerance,
   * and static detection (median parallax / implied translation) -> false so the caller can fall
   * back (method=2) or fail out (method=1).
   *
   * @return true and fills v_I0inI0 / gravity_inI0 on success
   */
  bool linear_seed_featureless(const std::unordered_map<size_t, std::shared_ptr<ov_core::Feature>> &features,
                               const std::map<size_t, int> &map_features_num_meas, int min_num_meas,
                               const std::map<double, bool> &map_camera_times,
                               const std::map<double, std::shared_ptr<ov_core::CpiV1>> &map_camera_cpi_I0toIi,
                               double oldest_camera_time, Eigen::Vector3d &v_I0inI0, Eigen::Vector3d &gravity_inI0) const;

  /// Initialization parameters
  InertialInitializerOptions params;

  /// Feature tracker database with all features in it
  std::shared_ptr<ov_core::FeatureDatabase> _db;

  /// Our history of IMU messages (time, angular, linear)
  std::shared_ptr<std::vector<ov_core::ImuData>> imu_data;

  /// Soft-reset hand-off context (may be null; then config bias seeds are used)
  std::shared_ptr<ResetContext> reset_ctx;
};

} // namespace ov_init

#endif // OV_INIT_DYNAMICINITIALIZER_H
