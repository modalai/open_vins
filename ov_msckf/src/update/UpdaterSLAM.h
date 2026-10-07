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

#ifndef OV_MSCKF_UPDATER_SLAM_H
#define OV_MSCKF_UPDATER_SLAM_H

#include <Eigen/Eigen>
#include <memory>
#include <unordered_map>
#include <vector>

#include "feat/FeatureInitializerOptions.h"

#include "UpdaterOptions.h"

namespace ov_core {
class Feature;
class FeatureInitializer;
} // namespace ov_core
namespace ov_type {
class Landmark;
} // namespace ov_type

namespace ov_msckf {

class State;
struct StereoMatchConfidence; // see RejectStats.h; only used here as a pointer-to-map param

/**
 * @brief Will compute the system for our sparse SLAM features and update the filter.
 *
 * This class is responsible for performing delayed feature initialization, SLAM update, and
 * SLAM anchor change for anchored feature representations.
 */
class UpdaterSLAM {

public:
  /**
   * @brief Default constructor for our SLAM updater
   *
   * Our updater has a feature initializer which we use to initialize features as needed.
   * Also the options allow for one to tune the different parameters for update.
   *
   * @param options_slam Updater options (include measurement noise value) for SLAM features
   * @param options_aruco Updater options (include measurement noise value) for ARUCO features
   * @param feat_init_options Feature initializer options
   */
  UpdaterSLAM(UpdaterOptions &options_slam, UpdaterOptions &options_aruco, ov_core::FeatureInitializerOptions &feat_init_options);

  /**
   * @brief Given tracked SLAM features, this will try to use them to update the state.
   * @param state State of the filter
   * @param feature_vec Features that can be used for update
   */
  void update(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec);

  /**
   * @brief Given max track features, this will try to use them to initialize them in the state.
   * @param state State of the filter
   * @param feature_vec Features that can be used for update
   * @param stereo_confidence DIAGNOSTIC ONLY: optional featid -> matcher-confidence lookup
   *        (from TrackOCL::stereo_confidence_map(), copied by VioManager since UpdaterSLAM
   *        doesn't otherwise see the tracker) attached to reinit diagnostic log lines. Null
   *        if not available; does not affect estimation, only ReinitEvent logging.
   */
  void delayed_init(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec,
                     const std::unordered_map<size_t, StereoMatchConfidence> *stereo_confidence = nullptr);

  /**
   * @brief Will change SLAM feature anchors if it will be marginalized
   *
   * Makes sure that if any clone is about to be marginalized, it changes anchor representation.
   * By default, this will shift the anchor into the newest IMU clone and keep the camera calibration anchor the same.
   *
   * @param state State of the filter
   */
  void change_anchors(std::shared_ptr<State> state);

protected:
  /**
   * @brief Shifts landmark anchor to new clone
   * @param state State of filter
   * @param landmark landmark whose anchor is being shifter
   * @param new_anchor_timestamp Clone timestamp we want to move to
   * @param new_cam_id Which camera frame we want to move to
   */
  /// False leaves the old anchor, current/FEJ coordinates and covariance intact.
  bool perform_anchor_change(std::shared_ptr<State> state, std::shared_ptr<ov_type::Landmark> landmark, double new_anchor_timestamp,
                             size_t new_cam_id);

  /// Options used during update for slam features
  UpdaterOptions _options_slam;

  /// Options used during update for aruco features
  UpdaterOptions _options_aruco;

  /// Feature initializer class object
  std::shared_ptr<ov_core::FeatureInitializer> initializer_feat;

  /// Reused pre-batch virtual camera poses, indexed by camera * clone_count +
  /// sorted clone index. Capacity follows the configured window bound and is
  /// retained across calls; global-only landmark initialization leaves it alone.
  struct VirtualAnchorPose {
    Eigen::Matrix3d R_GtoC;
    Eigen::Vector3d p_CinG;
  };
  std::vector<VirtualAnchorPose, Eigen::aligned_allocator<VirtualAnchorPose>> _virtual_anchor_scratch;

  // Row-time triangulation and a later stereo-to-mono retry must use the same
  // pre-batch calibration and endpoint motion. Allocate only for RS, then reuse
  // the configured camera/window capacity; no per-feature pose maps are retained.
  struct RowCameraSnapshot {
    Eigen::Matrix3d R_ItoC;
    Eigen::Vector3d p_IinC;
    double readout = 0.0;
    double inverse_height = 0.0;
    bool active = false;
  };
  struct RowMotionSnapshot {
    Eigen::Vector3d omega;
    Eigen::Vector3d velocity;
    bool available = false;
  };
  std::vector<RowCameraSnapshot, Eigen::aligned_allocator<RowCameraSnapshot>> _row_camera_scratch;
  std::vector<RowMotionSnapshot, Eigen::aligned_allocator<RowMotionSnapshot>> _row_motion_scratch;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_SLAM_H
