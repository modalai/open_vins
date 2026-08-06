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

#ifndef OV_MSCKF_STATE_OPTIONS_H
#define OV_MSCKF_STATE_OPTIONS_H

#include <map>

#include "types/LandmarkRepresentation.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace ov_msckf {

/**
 * @brief Struct which stores all our filter options
 */
struct StateOptions {

  /// Bool to determine whether or not to do first estimate Jacobians
  bool do_fej = true;

  /// Numerical integration methods
  enum IntegrationMethod { DISCRETE, RK4, ANALYTICAL };

  /// What type of numerical integration is used during propagation
  IntegrationMethod integration_method = IntegrationMethod::RK4;

  /// Bool to determine whether or not to calibrate imu-to-camera pose
  bool do_calib_camera_pose = false;

  /// Bool to determine whether or not to calibrate camera intrinsics
  bool do_calib_camera_intrinsics = false;

  /// Bool to determine whether or not to calibrate camera to IMU time offset
  bool do_calib_camera_timeoffset = false;

  /// Reference camera id whose cam-IMU time offset drives propagation/cloning; other cameras'
  /// offsets are estimated as deltas against this one at measurement time
  int cam_imu_dt_ref_camid = 0;

  /// Bool to determine whether or not to calibrate camera rolling shutter readout time
  bool do_calib_camera_readout = false;

  /// Which cameras carry an ESTIMATED readout state when do_calib_camera_readout is on.
  /// Filled from the per-camera shutter declarations: only rolling-shutter cameras are
  /// estimated; a global-shutter camera's readout stays pinned at zero (no spurious DOF).
  /// Empty map = legacy behavior (every camera estimated).
  std::map<size_t, bool> camera_estimate_readout;

  /// Initial 1-sigma prior (seconds) on an estimated readout time. The legacy 1ms is right for
  /// refining a CALIBRATED readout; recovering from a datasheet-grade seed needs a wider prior
  /// (a 12ms-error seed under a 1ms prior is a 12-sigma fight the filter effectively never wins).
  double calib_cam_readout_init_sigma = 0.001;

  /// Analytic IMU-bias columns from the preintegration bridge (see VioManagerOptions)
  bool epoch_bridge_bias_cols = true;

  /// Rolling-shutter row-anchor convention: which image row the frame stamp refers to. Row v
  /// samples at stamp + (v/h - rs_row_anchor) * readout. Parsed from "rs_convention":
  ///   top    (anchor 0.0) -- stamp is the raw HAL3 SOF (top row, start of readout). The
  ///                          pre-flip system convention; replays of that era's recordings
  ///                          use it to reproduce their trajectories.
  ///   center (anchor 0.5) -- stamp is the center-row mid-exposure instant (the default;
  ///                          the producer anchors stamps to match, see frame_timestamp_s)
  ///   bottom (anchor 1.0) -- stamp refers to the last row (producers that stamp end-of-frame)
  /// The stamp producer derives its anchoring from this SAME key, so stamps and filter model
  /// cannot disagree.
  double rs_row_anchor = 0.5;

  /// Freeze dt/readout Jacobian columns while the window motion is degenerate for temporal
  /// calibration (MVIS degenerate motions: static / constant velocity / slow pure rotation).
  /// Opt-in (default off): rigs enable it explicitly in their estimator config.
  bool dt_calib_gate = false;

  /// Gate threshold: window peak |omega| (rad/s) below which rotation provides no dt excitation
  double dt_calib_gate_min_omega = 0.10;

  /// Gate threshold: window velocity spread (m/s) below which translation provides no dt excitation
  double dt_calib_gate_min_vel_spread = 0.10;

  /// Bool to determine whether or not to calibrate the IMU intrinsics
  bool do_calib_imu_intrinsics = false;

  /// Bool to determine whether or not to calibrate the Gravity sensitivity
  bool do_calib_imu_g_sensitivity = false;

  /// IMU intrinsic models
  enum ImuModel { KALIBR, RPNG };

  /// What model our IMU intrinsics are
  ImuModel imu_model = ImuModel::KALIBR;

  /// Max clone size of sliding window
  int max_clone_size = 11;

  /// Max number of estimated SLAM features
  int max_slam_features = 25;

  /// Max number of SLAM features we allow to be included in a single EKF update.
  int max_slam_in_update = 1000;

  /// Max number of MSCKF features we will use at a given image timestep.
  int max_msckf_in_update = 1000;

  /// Max number of estimated ARUCO features
  int max_aruco_features = 1024;

  /// Number of distinct cameras that we will observe features in
  int num_cameras = 1;

  /// What representation our features are in (msckf features)
  ov_type::LandmarkRepresentation::Representation feat_rep_msckf = ov_type::LandmarkRepresentation::Representation::GLOBAL_3D;

  /// What representation our features are in (slam features)
  ov_type::LandmarkRepresentation::Representation feat_rep_slam = ov_type::LandmarkRepresentation::Representation::GLOBAL_3D;

  /// What representation our features are in (aruco tag features)
  ov_type::LandmarkRepresentation::Representation feat_rep_aruco = ov_type::LandmarkRepresentation::Representation::GLOBAL_3D;

  /// Nice print function of what parameters we have loaded
  void print(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    if (parser != nullptr) {
      parser->parse_config("use_fej", do_fej);

      // Integration method
      std::string integration_str = "rk4";
      parser->parse_config("integration", integration_str);
      if (integration_str == "discrete") {
        integration_method = IntegrationMethod::DISCRETE;
      } else if (integration_str == "rk4") {
        integration_method = IntegrationMethod::RK4;
      } else if (integration_str == "analytical") {
        integration_method = IntegrationMethod::ANALYTICAL;
      } else {
        PRINT_ERROR(RED "invalid imu integration model: %s\n" RESET, integration_str.c_str());
        PRINT_ERROR(RED "please select a valid model: discrete, rk4, analytical\n" RESET);
        std::exit(EXIT_FAILURE);
      }

      // Calibration booleans
      parser->parse_config("calib_cam_extrinsics", do_calib_camera_pose);
      parser->parse_config("calib_cam_intrinsics", do_calib_camera_intrinsics);
      parser->parse_config("calib_cam_timeoffset", do_calib_camera_timeoffset);
      parser->parse_config("cam_imu_dt_ref_camid", cam_imu_dt_ref_camid, false);
      parser->parse_config("calib_cam_readout", do_calib_camera_readout, false);
      parser->parse_config("calib_cam_readout_init_sigma", calib_cam_readout_init_sigma, false);
      if (calib_cam_readout_init_sigma <= 0.0) {
        PRINT_ERROR(RED "calib_cam_readout_init_sigma must be positive (got %.6f)\n" RESET, calib_cam_readout_init_sigma);
        std::exit(EXIT_FAILURE);
      }
      std::string rs_convention_str = "center";
      parser->parse_config("rs_convention", rs_convention_str, false);
      if (rs_convention_str == "top") {
        rs_row_anchor = 0.0;
      } else if (rs_convention_str == "center") {
        rs_row_anchor = 0.5;
      } else if (rs_convention_str == "bottom") {
        rs_row_anchor = 1.0;
      } else {
        PRINT_ERROR(RED "invalid rs_convention: '%s'\n" RESET, rs_convention_str.c_str());
        PRINT_ERROR(RED "please select a valid convention: top, center, bottom\n" RESET);
        if (rs_convention_str.empty()) {
          // The classic cause of an EMPTY read: a ':' inside the key's TRAILING comment --
          // cv::FileStorage splits there and the node parses as a map. Own-line comments are safe.
          PRINT_ERROR(RED "an empty value usually means a ':' in the key's trailing comment "
                          "(cv::FileStorage splits on it); move the comment to its own line\n" RESET);
        }
        std::exit(EXIT_FAILURE);
      }
      parser->parse_config("dt_calib_gate", dt_calib_gate, false);
      parser->parse_config("dt_calib_gate_min_omega", dt_calib_gate_min_omega, false);
      parser->parse_config("dt_calib_gate_min_vel_spread", dt_calib_gate_min_vel_spread, false);
      parser->parse_config("calib_imu_intrinsics", do_calib_imu_intrinsics);
      parser->parse_config("calib_imu_g_sensitivity", do_calib_imu_g_sensitivity);

      // State parameters
      parser->parse_config("max_clones", max_clone_size);
      parser->parse_config("max_slam", max_slam_features);
      parser->parse_config("max_slam_in_update", max_slam_in_update);
      parser->parse_config("max_msckf_in_update", max_msckf_in_update);
      parser->parse_config("num_aruco", max_aruco_features);
      parser->parse_config("max_cameras", num_cameras);

      // Feature representations
      std::string rep1 = ov_type::LandmarkRepresentation::as_string(feat_rep_msckf);
      parser->parse_config("feat_rep_msckf", rep1);
      feat_rep_msckf = ov_type::LandmarkRepresentation::from_string(rep1);
      std::string rep2 = ov_type::LandmarkRepresentation::as_string(feat_rep_slam);
      parser->parse_config("feat_rep_slam", rep2);
      feat_rep_slam = ov_type::LandmarkRepresentation::from_string(rep2);
      std::string rep3 = ov_type::LandmarkRepresentation::as_string(feat_rep_aruco);
      parser->parse_config("feat_rep_aruco", rep3);
      feat_rep_aruco = ov_type::LandmarkRepresentation::from_string(rep3);

      // IMU model
      std::string imu_model_str = "kalibr";
      parser->parse_external("relative_config_imu", "imu0", "model", imu_model_str);
      if (imu_model_str == "kalibr" || imu_model_str == "calibrated") {
        imu_model = ImuModel::KALIBR;
      } else if (imu_model_str == "rpng") {
        imu_model = ImuModel::RPNG;
      } else {
        PRINT_ERROR(RED "invalid imu model: %s\n" RESET, imu_model_str.c_str());
        PRINT_ERROR(RED "please select a valid model: kalibr, rpng\\n" RESET);
        std::exit(EXIT_FAILURE);
      }
      if (imu_model_str == "calibrated" && (do_calib_imu_intrinsics || do_calib_imu_g_sensitivity)) {
        PRINT_ERROR(RED "calibrated IMU model selected, but requested calibration!\n" RESET);
        PRINT_ERROR(RED "please select what model you have: kalibr, rpng\n" RESET);
        std::exit(EXIT_FAILURE);
      }
    }
    PRINT_DEBUG("  - use_fej: %d\n", do_fej);
    PRINT_DEBUG("  - integration: %d\n", integration_method);
    PRINT_DEBUG("  - calib_cam_extrinsics: %d\n", do_calib_camera_pose);
    PRINT_DEBUG("  - calib_cam_intrinsics: %d\n", do_calib_camera_intrinsics);
    PRINT_DEBUG("  - calib_cam_timeoffset: %d\n", do_calib_camera_timeoffset);
    PRINT_DEBUG("  - cam_imu_dt_ref_camid: %d\n", cam_imu_dt_ref_camid);
    PRINT_DEBUG("  - calib_cam_readout: %d\n", do_calib_camera_readout);
    PRINT_DEBUG("  - calib_cam_readout_init_sigma: %.4f\n", calib_cam_readout_init_sigma);
    PRINT_DEBUG("  - rs_convention: %s (row anchor %.1f)\n",
                rs_row_anchor == 0.0 ? "top" : (rs_row_anchor == 1.0 ? "bottom" : "center"), rs_row_anchor);
    PRINT_DEBUG("  - calib_imu_intrinsics: %d\n", do_calib_imu_intrinsics);
    PRINT_DEBUG("  - calib_imu_g_sensitivity: %d\n", do_calib_imu_g_sensitivity);
    PRINT_DEBUG("  - imu_model: %d\n", imu_model);
    PRINT_DEBUG("  - max_clones: %d\n", max_clone_size);
    PRINT_DEBUG("  - max_slam: %d\n", max_slam_features);
    PRINT_DEBUG("  - max_slam_in_update: %d\n", max_slam_in_update);
    PRINT_DEBUG("  - max_msckf_in_update: %d\n", max_msckf_in_update);
    PRINT_DEBUG("  - max_aruco: %d\n", max_aruco_features);
    PRINT_DEBUG("  - max_cameras: %d\n", num_cameras);
    PRINT_DEBUG("  - feat_rep_msckf: %s\n", ov_type::LandmarkRepresentation::as_string(feat_rep_msckf).c_str());
    PRINT_DEBUG("  - feat_rep_slam: %s\n", ov_type::LandmarkRepresentation::as_string(feat_rep_slam).c_str());
    PRINT_DEBUG("  - feat_rep_aruco: %s\n", ov_type::LandmarkRepresentation::as_string(feat_rep_aruco).c_str());
  }
};

} // namespace ov_msckf

#endif // OV_MSCKF_STATE_OPTIONS_H