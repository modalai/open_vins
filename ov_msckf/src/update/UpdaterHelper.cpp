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

#include "UpdaterHelper.h"

#include <algorithm>
#include <cinttypes>

#include "state/State.h"

#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

void UpdaterHelper::get_feature_jacobian_representation(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                        std::vector<Eigen::MatrixXd> &H_x, std::vector<std::shared_ptr<Type>> &x_order) {

  // Global XYZ representation
  if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D) {
    H_f.resize(3, 3);
    H_f.setIdentity();
    return;
  }

  // Global inverse depth representation
  if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_FULL_INVERSE_DEPTH) {

    // Get the feature linearization point
    Eigen::Matrix<double, 3, 1> p_FinG = (state->_options.do_fej) ? feature.p_FinG_fej : feature.p_FinG;

    // Get inverse depth representation (should match what is in Landmark.cpp)
    double g_rho = 1 / p_FinG.norm();
    double g_phi = std::acos(g_rho * p_FinG(2));
    // double g_theta = std::asin(g_rho*p_FinG(1)/std::sin(g_phi));
    double g_theta = std::atan2(p_FinG(1), p_FinG(0));
    Eigen::Matrix<double, 3, 1> p_invFinG;
    p_invFinG(0) = g_theta;
    p_invFinG(1) = g_phi;
    p_invFinG(2) = g_rho;

    // Get inverse depth bearings
    double sin_th = std::sin(p_invFinG(0, 0));
    double cos_th = std::cos(p_invFinG(0, 0));
    double sin_phi = std::sin(p_invFinG(1, 0));
    double cos_phi = std::cos(p_invFinG(1, 0));
    double rho = p_invFinG(2, 0);

    // Construct the Jacobian
    H_f.resize(3, 3);
    H_f << -(1.0 / rho) * sin_th * sin_phi, (1.0 / rho) * cos_th * cos_phi, -(1.0 / (rho * rho)) * cos_th * sin_phi,
        (1.0 / rho) * cos_th * sin_phi, (1.0 / rho) * sin_th * cos_phi, -(1.0 / (rho * rho)) * sin_th * sin_phi, 0.0,
        -(1.0 / rho) * sin_phi, -(1.0 / (rho * rho)) * cos_phi;
    return;
  }

  //======================================================================
  //======================================================================
  //======================================================================

  // Assert that we have an anchor pose for this feature
  assert(feature.anchor_cam_id != -1);

  // Anchor pose orientation and position, and camera calibration for our anchor camera
  Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
  Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
  Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
  Eigen::Vector3d p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
  Eigen::Vector3d p_FinA = feature.p_FinA;

  // Camera time offset delta for anchor camera (consistent with get_feature_jacobian_full);
  // the KNOWN epoch residual of the anchor observation adds to both linearizations
  const double dt_epoch_anc = state->epoch_residual((size_t)feature.anchor_cam_id, feature.anchor_clone_timestamp);
  double dt_camoff_anc = state->cam_imu_dt_delta(feature.anchor_cam_id) + dt_epoch_anc;
  double dt_camoff_anc_lin = (state->_options.do_calib_camera_timeoffset
      ? state->cam_imu_dt_delta_fej(feature.anchor_cam_id) : state->cam_imu_dt_delta(feature.anchor_cam_id)) + dt_epoch_anc;

  // Rolling shutter readout for anchor camera (map is populated for every camera at construction;
  // id >= 0 iff this camera's readout is an ESTIMATED state -- declared rolling under calib)
  std::shared_ptr<Vec> readout_anc = state->_calib_camera_readout.at(feature.anchor_cam_id);
  const bool readout_anc_est = readout_anc->id() >= 0;
  double t_readout_anc = readout_anc->value()(0);
  double t_readout_anc_lin = readout_anc_est ? readout_anc->fej()(0) : t_readout_anc;
  Eigen::Vector3d omega_anc_val = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_anc_val = Eigen::Vector3d::Zero();
  Eigen::Vector3d omega_anc_lin = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_anc_lin = Eigen::Vector3d::Zero();
  double v_frac_anc = 0.0;
  bool need_anchor_correction = readout_anc_est || state->_options.do_calib_camera_timeoffset ||
      std::abs(dt_camoff_anc) > 1e-10 || std::abs(dt_camoff_anc_lin) > 1e-10 ||
      std::abs(t_readout_anc) > 1e-10 || std::abs(t_readout_anc_lin) > 1e-10;
  if (need_anchor_correction) {
    // Get kinematics at anchor clone time (needed for both time offset and RS corrections).
    // Tolerant lookup: a clone restored by a warm-started reset may have no kinematics -- degrade
    // to zero correction/columns for it (counted; the clone marginalizes out within one window).
    auto kin_anc_it = state->_clones_kinematics.find(feature.anchor_clone_timestamp);
    if (kin_anc_it != state->_clones_kinematics.end()) {
      const State::CloneKinematics &kin_anc = kin_anc_it->second;
      omega_anc_val = kin_anc.omega;
      v_anc_val = kin_anc.vel;
      if (state->_options.do_fej) {
        omega_anc_lin = kin_anc.omega_fej;
        v_anc_lin = kin_anc.vel_fej;
      } else {
        omega_anc_lin = omega_anc_val;
        v_anc_lin = v_anc_val;
      }
    } else {
      state->_kin_miss_count++;
      if (state->_kin_miss_count == 1 || state->_kin_miss_count % 256 == 0) {
        PRINT_WARNING(YELLOW "UpdaterHelper: no clone kinematics at %.6f (miss #%" PRIu64 "); zero dt/RS correction\n" RESET,
                      feature.anchor_clone_timestamp, state->_kin_miss_count);
      }
    }

    // Compute centered v_frac from anchor observation pixel row (RS component; stamps anchor mid-frame)
    auto ts_it = feature.timestamps.find(feature.anchor_cam_id);
    if (ts_it != feature.timestamps.end()) {
      const auto &anc_timestamps = ts_it->second;
      auto it = std::find(anc_timestamps.begin(), anc_timestamps.end(), feature.anchor_clone_timestamp);
      if (it != anc_timestamps.end()) {
        size_t anc_idx = std::distance(anc_timestamps.begin(), it);
        const auto &anc_uvs = feature.uvs.at(feature.anchor_cam_id);
        double v_pixel_anc = (double)anc_uvs.at(anc_idx)(1);
        double inv_img_h_anc = 1.0 / (double)state->_cam_intrinsics_cameras.at(feature.anchor_cam_id)->h();
        v_frac_anc = v_pixel_anc * inv_img_h_anc - 0.5;
      }
    }

    // Apply combined time offset delta and RS correction to anchor pose (value)
    double dt_total_anc = dt_camoff_anc;
    if (std::abs(t_readout_anc) > 1e-10) {
      dt_total_anc += v_frac_anc * t_readout_anc;
    }
    if (std::abs(dt_total_anc) > 1e-10) {
      R_GtoI = exp_so3(-omega_anc_val * dt_total_anc) * R_GtoI;
      p_IinG = p_IinG + v_anc_val * dt_total_anc;
    }
  }

  // If I am doing FEJ, I should FEJ the anchor states (should we fej calibration???)
  // Also get the FEJ position of the feature if we are
  if (state->_options.do_fej) {
    // "Best" feature in the global frame
    Eigen::Vector3d p_FinG_best = R_GtoI.transpose() * R_ItoC.transpose() * (feature.p_FinA - p_IinC) + p_IinG;
    // Transform the best into our anchor frame using FEJ
    R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot_fej();
    p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos_fej();
    // Apply combined time offset delta and RS correction to FEJ anchor pose
    if (need_anchor_correction) {
      double dt_total_anc_fej = dt_camoff_anc_lin;
      if (std::abs(t_readout_anc_lin) > 1e-10) {
        dt_total_anc_fej += v_frac_anc * t_readout_anc_lin;
      }
      if (std::abs(dt_total_anc_fej) > 1e-10) {
        R_GtoI = exp_so3(-omega_anc_lin * dt_total_anc_fej) * R_GtoI;
        p_IinG = p_IinG + v_anc_lin * dt_total_anc_fej;
      }
    }
    p_FinA = (R_GtoI.transpose() * R_ItoC.transpose()).transpose() * (p_FinG_best - p_IinG) + p_IinC;
  }
  Eigen::Matrix3d R_CtoG = R_GtoI.transpose() * R_ItoC.transpose();

  // Jacobian for our anchor pose
  Eigen::Matrix<double, 3, 6> H_anc;
  H_anc.block(0, 0, 3, 3).noalias() = -R_GtoI.transpose() * skew_x(R_ItoC.transpose() * (p_FinA - p_IinC));
  H_anc.block(0, 3, 3, 3).setIdentity();

  // Add anchor Jacobians to our return vector
  x_order.push_back(state->_clones_IMU.at(feature.anchor_clone_timestamp));
  H_x.push_back(H_anc);

  // Get calibration Jacobians (for anchor clone)
  if (state->_options.do_calib_camera_pose) {
    Eigen::Matrix<double, 3, 6> H_calib;
    H_calib.block(0, 0, 3, 3).noalias() = -R_CtoG * skew_x(p_FinA - p_IinC);
    H_calib.block(0, 3, 3, 3) = -R_CtoG;
    x_order.push_back(state->_calib_IMUtoCAM.at(feature.anchor_cam_id));
    H_x.push_back(H_calib);
  }

  // Anchor readout time calibration Jacobian
  // p_FinG = R_GtoI^T * R_ItoC^T * (p_FinA - p_IinC) + p_IinG
  // d(p_FinG)/d(t_rd) = (R_GtoI^T * [ω]× * R_ItoC^T * (p_FinA - p_IinC) + v) * v_frac
  // Column skipped while the window motion is degenerate for temporal calibration (values still used)
  if (readout_anc_est && !state->dt_calib_degenerate()) {
    Eigen::Vector3d p_FinI_anc = R_ItoC.transpose() * (p_FinA - p_IinC);
    Eigen::Matrix<double, 3, 1> H_readout_anc;
    H_readout_anc = (R_GtoI.transpose() * skew_x(omega_anc_lin) * p_FinI_anc + v_anc_lin) * v_frac_anc;
    x_order.push_back(state->_calib_camera_readout.at(feature.anchor_cam_id));
    H_x.push_back(H_readout_anc);
  }

  // If we are doing anchored XYZ feature
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_3D) {
    H_f = R_CtoG;
    return;
  }

  // If we are doing full inverse depth
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_FULL_INVERSE_DEPTH) {

    // Get inverse depth representation (should match what is in Landmark.cpp)
    double a_rho = 1 / p_FinA.norm();
    double a_phi = std::acos(a_rho * p_FinA(2));
    double a_theta = std::atan2(p_FinA(1), p_FinA(0));
    Eigen::Matrix<double, 3, 1> p_invFinA;
    p_invFinA(0) = a_theta;
    p_invFinA(1) = a_phi;
    p_invFinA(2) = a_rho;

    // Using anchored inverse depth
    double sin_th = std::sin(p_invFinA(0, 0));
    double cos_th = std::cos(p_invFinA(0, 0));
    double sin_phi = std::sin(p_invFinA(1, 0));
    double cos_phi = std::cos(p_invFinA(1, 0));
    double rho = p_invFinA(2, 0);
    // assert(p_invFinA(2,0)>=0.0);

    // Jacobian of anchored 3D position wrt inverse depth parameters
    Eigen::Matrix<double, 3, 3> d_pfinA_dpinv;
    d_pfinA_dpinv << -(1.0 / rho) * sin_th * sin_phi, (1.0 / rho) * cos_th * cos_phi, -(1.0 / (rho * rho)) * cos_th * sin_phi,
        (1.0 / rho) * cos_th * sin_phi, (1.0 / rho) * sin_th * cos_phi, -(1.0 / (rho * rho)) * sin_th * sin_phi, 0.0,
        -(1.0 / rho) * sin_phi, -(1.0 / (rho * rho)) * cos_phi;
    H_f = R_CtoG * d_pfinA_dpinv;
    return;
  }

  // If we are doing the MSCKF version of inverse depth
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH) {

    // Get inverse depth representation (should match what is in Landmark.cpp)
    Eigen::Matrix<double, 3, 1> p_invFinA_MSCKF;
    p_invFinA_MSCKF(0) = p_FinA(0) / p_FinA(2);
    p_invFinA_MSCKF(1) = p_FinA(1) / p_FinA(2);
    p_invFinA_MSCKF(2) = 1 / p_FinA(2);

    // Using the MSCKF version of inverse depth
    double alpha = p_invFinA_MSCKF(0, 0);
    double beta = p_invFinA_MSCKF(1, 0);
    double rho = p_invFinA_MSCKF(2, 0);

    // Jacobian of anchored 3D position wrt inverse depth parameters
    Eigen::Matrix<double, 3, 3> d_pfinA_dpinv;
    d_pfinA_dpinv << (1.0 / rho), 0.0, -(1.0 / (rho * rho)) * alpha, 0.0, (1.0 / rho), -(1.0 / (rho * rho)) * beta, 0.0, 0.0,
        -(1.0 / (rho * rho));
    H_f = R_CtoG * d_pfinA_dpinv;
    return;
  }

  /// CASE: Estimate single depth of the feature using the initial bearing
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

    // Get inverse depth representation (should match what is in Landmark.cpp)
    double rho = 1.0 / p_FinA(2);
    Eigen::Vector3d bearing = rho * p_FinA;

    // Jacobian of anchored 3D position wrt inverse depth parameters
    Eigen::Vector3d d_pfinA_drho;
    d_pfinA_drho << -(1.0 / (rho * rho)) * bearing;
    H_f = R_CtoG * d_pfinA_drho;
    return;
  }

  // Failure, invalid representation that is not programmed
  assert(false);
}

void UpdaterHelper::get_feature_jacobian_full(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                              Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order) {

  // Total number of measurements for this feature
  int total_meas = 0;
  for (auto const &pair : feature.timestamps) {
    total_meas += (int)pair.second.size();
  }

  // Compute the size of the states involved with this feature
  int total_hx = 0;
  std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;
  bool include_dt_ref_col = false;

  // Reference camera timeoffset term is used by all camera measurements as dt_cam - dt_ref
  std::shared_ptr<Vec> dt_ref_var;
  if (state->_options.do_calib_camera_timeoffset) {
    dt_ref_var = state->cam_imu_dt_var((size_t)state->cam_imu_dt_ref_camid());
  }

  for (auto const &pair : feature.timestamps) {

    // Our extrinsics and intrinsics
    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first);
    std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(pair.first);

    // If doing calibration extrinsics
    if (state->_options.do_calib_camera_pose) {
      map_hx.insert({calibration, total_hx});
      x_order.push_back(calibration);
      total_hx += calibration->size();
    }

    // If doing calibration intrinsics
    if (state->_options.do_calib_camera_intrinsics) {
      map_hx.insert({distortion, total_hx});
      x_order.push_back(distortion);
      total_hx += distortion->size();
    }

    // If doing camera-imu timeoffset calibration
    if (state->_options.do_calib_camera_timeoffset) {
      std::shared_ptr<Vec> dt_cam = state->cam_imu_dt_var(pair.first);
      if (dt_cam.get() != dt_ref_var.get() && map_hx.find(dt_cam) == map_hx.end()) {
        include_dt_ref_col = true;
        map_hx.insert({dt_cam, total_hx});
        x_order.push_back(dt_cam);
        total_hx += dt_cam->size();
      }
    }

    // If this camera's readout is an estimated state (declared rolling under calib)
    {
      std::shared_ptr<Vec> readout = state->_calib_camera_readout.at(pair.first);
      if (readout->id() >= 0 && map_hx.find(readout) == map_hx.end()) {
        map_hx.insert({readout, total_hx});
        x_order.push_back(readout);
        total_hx += readout->size();
      }
    }

    // Loop through all measurements for this specific camera
    for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

      // Add this clone if it is not added already
      std::shared_ptr<PoseJPL> clone_Ci = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
      if (map_hx.find(clone_Ci) == map_hx.end()) {
        map_hx.insert({clone_Ci, total_hx});
        x_order.push_back(clone_Ci);
        total_hx += clone_Ci->size();
      }
    }
  }

  // Add reference camera dt column only when at least one non-reference camera contributed measurements
  if (state->_options.do_calib_camera_timeoffset && include_dt_ref_col && map_hx.find(dt_ref_var) == map_hx.end()) {
    map_hx.insert({dt_ref_var, total_hx});
    x_order.push_back(dt_ref_var);
    total_hx += dt_ref_var->size();
  }

  // If any observation rides a preintegration bridge, its pose depends on the IMU biases: add the
  // bias columns once (analytic H_bias from the bridge Jacobians)
  int bias_g_hx_col = -1, bias_a_hx_col = -1;
  {
    bool any_bridge = false;
    for (auto const &pair : state->_options.epoch_bridge_bias_cols ? feature.timestamps : decltype(feature.timestamps)()) {
      for (size_t m = 0; m < pair.second.size() && !any_bridge; m++) {
        any_bridge = (state->epoch_bridge(pair.first, pair.second.at(m)) != nullptr);
      }
    }
    if (any_bridge) {
      bias_g_hx_col = total_hx;
      map_hx.insert({state->_imu->bg(), total_hx});
      x_order.push_back(state->_imu->bg());
      total_hx += state->_imu->bg()->size();
      bias_a_hx_col = total_hx;
      map_hx.insert({state->_imu->ba(), total_hx});
      x_order.push_back(state->_imu->ba());
      total_hx += state->_imu->ba()->size();
    }
  }

  // If we are using an anchored representation, make sure that the anchor is also added
  if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {

    // Assert we have a clone
    assert(feature.anchor_cam_id != -1);

    // Add this anchor if it is not added already
    std::shared_ptr<PoseJPL> clone_Ai = state->_clones_IMU.at(feature.anchor_clone_timestamp);
    if (map_hx.find(clone_Ai) == map_hx.end()) {
      map_hx.insert({clone_Ai, total_hx});
      x_order.push_back(clone_Ai);
      total_hx += clone_Ai->size();
    }

    // Also add its calibration if we are doing calibration
    if (state->_options.do_calib_camera_pose) {
      // Add this anchor if it is not added already
      std::shared_ptr<PoseJPL> clone_calib = state->_calib_IMUtoCAM.at(feature.anchor_cam_id);
      if (map_hx.find(clone_calib) == map_hx.end()) {
        map_hx.insert({clone_calib, total_hx});
        x_order.push_back(clone_calib);
        total_hx += clone_calib->size();
      }
    }
  }

  //=========================================================================
  //=========================================================================

  const double dt_ref_val = state->cam_imu_dt_ref();
  const double dt_ref_lin = state->_options.do_calib_camera_timeoffset ? state->cam_imu_dt_ref_fej() : dt_ref_val;
  const int dt_ref_hx_col = (state->_options.do_calib_camera_timeoffset && map_hx.find(dt_ref_var) != map_hx.end())
                                ? (int)map_hx.at(dt_ref_var)
                                : -1;
  // Freeze the temporal-calibration COLUMNS (dt and readout) while the window motion is degenerate
  // for them; their current values are still applied to the poses above/below
  const bool dt_rs_cols_active = !state->dt_calib_degenerate();

  // Calculate the position of this feature in the global frame
  // If anchored, then we need to calculate the position of the feature in the global
  Eigen::Vector3d p_FinG = feature.p_FinG;
  if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    // Assert that we have an anchor pose for this feature
    assert(feature.anchor_cam_id != -1);
    // Get calibration for our anchor camera
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
    // Anchor pose orientation and position
    Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
    Eigen::Vector3d p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
    // Apply async camera timeoffset correction and rolling shutter correction for anchor pose
    double dt_camoff_anc =
        state->cam_imu_dt_delta(feature.anchor_cam_id) + state->epoch_residual((size_t)feature.anchor_cam_id, feature.anchor_clone_timestamp);
    double t_readout_anc = state->_calib_camera_readout.at(feature.anchor_cam_id)->value()(0);
    double dt_total_anc = dt_camoff_anc;
    if (std::abs(t_readout_anc) > 1e-10) {
      const auto &anc_timestamps = feature.timestamps.at(feature.anchor_cam_id);
      auto it = std::find(anc_timestamps.begin(), anc_timestamps.end(), feature.anchor_clone_timestamp);
      if (it != anc_timestamps.end()) {
        size_t anc_idx = std::distance(anc_timestamps.begin(), it);
        const auto &anc_uvs = feature.uvs.at(feature.anchor_cam_id);
        double v_pixel_anc = (double)anc_uvs.at(anc_idx)(1);
        double inv_img_h_anc = 1.0 / (double)state->_cam_intrinsics_cameras.at(feature.anchor_cam_id)->h();
        dt_total_anc += (v_pixel_anc * inv_img_h_anc - 0.5) * t_readout_anc;
      }
    }
    if (std::abs(dt_total_anc) > 1e-10) {
      auto kin_anc_it = state->_clones_kinematics.find(feature.anchor_clone_timestamp);
      if (kin_anc_it != state->_clones_kinematics.end()) {
        R_GtoI = exp_so3(-kin_anc_it->second.omega * dt_total_anc) * R_GtoI;
        p_IinG = p_IinG + kin_anc_it->second.vel * dt_total_anc;
      } else {
        state->_kin_miss_count++;
      }
    }
    // Feature in the global frame
    p_FinG = R_GtoI.transpose() * R_ItoC.transpose() * (feature.p_FinA - p_IinC) + p_IinG;
  }

  // Calculate the position of this feature in the global frame FEJ
  // If anchored, then we can use the "best" p_FinG since the value of p_FinA does not matter
  Eigen::Vector3d p_FinG_fej = feature.p_FinG_fej;
  if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    p_FinG_fej = p_FinG;
  }

  //=========================================================================
  //=========================================================================

  // Allocate our residual and Jacobians
  int c = 0;
  int jacobsize = (feature.feat_representation != LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
  res = Eigen::VectorXd::Zero(2 * total_meas);
  H_f = Eigen::MatrixXd::Zero(2 * total_meas, jacobsize);
  H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);

  // Derivative of p_FinG in respect to feature representation.
  // This only needs to be computed once and thus we pull it out of the loop
  Eigen::MatrixXd dpfg_dlambda;
  std::vector<Eigen::MatrixXd> dpfg_dx;
  std::vector<std::shared_ptr<Type>> dpfg_dx_order;
  UpdaterHelper::get_feature_jacobian_representation(state, feature, dpfg_dlambda, dpfg_dx, dpfg_dx_order);

  // Assert that all the ones in our order are already in our local jacobian mapping
#ifndef NDEBUG
  for (auto &type : dpfg_dx_order) {
    assert(map_hx.find(type) != map_hx.end());
  }
#endif

  // Loop through each camera for this feature
  for (auto const &pair : feature.timestamps) {
    size_t cam_id = pair.first;
    const auto &cam_timestamps = pair.second;
    const auto &cam_uvs = feature.uvs.at(cam_id);

    // Our calibration between the IMU and CAMi frames (all maps populated at construction)
    std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(cam_id);
    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(cam_id);
    std::shared_ptr<CamBase> camera_model = state->_cam_intrinsics_cameras.at(cam_id);
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();
    int img_h = camera_model->h();
    double inv_img_h = 1.0 / (double)img_h;

    std::shared_ptr<Vec> readout = state->_calib_camera_readout.at(cam_id);
    const bool readout_est = readout->id() >= 0;
    double t_readout = readout->value()(0);
    double t_readout_lin = readout_est ? readout->fej()(0) : t_readout;
    std::shared_ptr<Vec> dt_cam_var = state->cam_imu_dt_var(cam_id);
    double dt_camoff = dt_cam_var->value()(0) - dt_ref_val;
    double dt_camoff_lin = (state->_options.do_calib_camera_timeoffset ? dt_cam_var->fej()(0) : dt_cam_var->value()(0)) - dt_ref_lin;
    bool rs_on_value = std::abs(t_readout) > 1e-10;
    bool rs_on_linearization = std::abs(t_readout_lin) > 1e-10;
    bool dt_on_value = std::abs(dt_camoff) > 1e-10;
    bool dt_on_linearization = std::abs(dt_camoff_lin) > 1e-10;
    bool need_rs_terms = readout_est || state->_options.do_calib_camera_timeoffset || rs_on_value ||
                         rs_on_linearization || dt_on_value || dt_on_linearization;
    int readout_hx_col = -1;
    if (readout_est) {
      readout_hx_col = map_hx.at(readout);
    }
    int dt_cam_hx_col = -1;
    if (state->_options.do_calib_camera_timeoffset && map_hx.find(dt_cam_var) != map_hx.end()) {
      dt_cam_hx_col = map_hx.at(dt_cam_var);
    }

    // Loop through all measurements for this specific camera
    for (size_t m = 0; m < cam_timestamps.size(); m++) {

      //=========================================================================
      //=========================================================================

      // Get current IMU clone state
      std::shared_ptr<PoseJPL> clone_Ii = state->_clones_IMU.at(cam_timestamps.at(m));
      Eigen::Matrix3d R_GtoIi = clone_Ii->Rot();
      Eigen::Vector3d p_IiinG = clone_Ii->pos();

      // Time correction of this observation's pose to its true sampling instant.
      // With a bridge (epoch-snapped frame): EXACT ACI2 composition over the KNOWN residual,
      // bias-corrected to first order via J_b (never re-integrated); only the small ESTIMATED
      // parts (dt delta drift since build + RS row time) compose as exact SO(3) under constant
      // bridge-ENDPOINT kinematics (Jacobian columns stay first-order). Without a bridge: the
      // same constant-kinematics model over the whole dt_total. Tolerant kinematics lookup
      // throughout (warm-restored clones may have none).
      const double clone_time = cam_timestamps.at(m);
      const double dt_epoch = state->epoch_residual(cam_id, clone_time);
      const PreintBridgeData *bridge = state->epoch_bridge(cam_id, clone_time);
      const bool need_obs_terms = need_rs_terms || std::abs(dt_epoch) > 1e-12;
      // Estimated shift: with a bridge the KNOWN residual is integrated exactly, so it drops out
      double dt_total = dt_camoff + ((bridge == nullptr) ? dt_epoch : 0.0);
      Eigen::Vector3d omega_clone_val = Eigen::Vector3d::Zero();
      Eigen::Vector3d v_clone_val = Eigen::Vector3d::Zero();
      Eigen::Vector3d omega_clone_lin = Eigen::Vector3d::Zero();
      Eigen::Vector3d v_clone_lin = Eigen::Vector3d::Zero();
      bool have_clone_kin = false;
      double v_pixel = (double)cam_uvs.at(m)(1);
      // Bridge-endpoint kinematics for the Jacobian columns / extra shifts (filled below)
      Eigen::Vector3d omega_end_lin = Eigen::Vector3d::Zero();
      Eigen::Vector3d v_end_lin = Eigen::Vector3d::Zero();
      Eigen::Matrix3d R_clone_lin = Eigen::Matrix3d::Identity(); // clone rotation at the Jacobian linearization
      if (need_obs_terms) {
        auto kin_it = state->_clones_kinematics.find(clone_time);
        if (kin_it != state->_clones_kinematics.end()) {
          have_clone_kin = true;
          omega_clone_val = kin_it->second.omega;
          v_clone_val = kin_it->second.vel;
          if (state->_options.do_fej) {
            omega_clone_lin = kin_it->second.omega_fej;
            v_clone_lin = kin_it->second.vel_fej;
          } else {
            omega_clone_lin = omega_clone_val;
            v_clone_lin = v_clone_val;
          }
        } else {
          state->_kin_miss_count++;
          if (state->_kin_miss_count == 1 || state->_kin_miss_count % 256 == 0) {
            PRINT_WARNING(YELLOW "UpdaterHelper: no clone kinematics at %.6f (miss #%" PRIu64 "); zero dt/RS correction\n" RESET,
                          clone_time, state->_kin_miss_count);
          }
        }
        if (rs_on_value) {
          dt_total += (v_pixel * inv_img_h - 0.5) * t_readout;
        }
        if (bridge != nullptr) {
          // EXACT composition of the clone pose over the known residual, with the first-order
          // bias correction at the current estimates (ACI2 partial-fixed linearization)
          Eigen::Matrix<double, 6, 1> db;
          db.head<3>() = state->_imu->bias_g() - bridge->bg0;
          db.tail<3>() = state->_imu->bias_a() - bridge->ba0;
          const Eigen::Vector3d d_th = bridge->J_b.block(0, 0, 3, 6) * db;
          const Eigen::Vector3d d_al = bridge->J_b.block(3, 0, 3, 6) * db;
          const Eigen::Vector3d d_be = bridge->J_b.block(6, 0, 3, 6) * db;
          const Eigen::Matrix3d R_clone_val = R_GtoIi;
          R_GtoIi = ov_core::exp_so3(d_th) * bridge->DR * R_clone_val;
          p_IiinG = p_IiinG + v_clone_val * bridge->dt + bridge->p_grav + R_clone_val.transpose() * (bridge->alpha + d_al);
          const Eigen::Vector3d v_end_val = v_clone_val + bridge->v_grav + R_clone_val.transpose() * (bridge->beta + d_be);
          // Linearization kinematics for the temporal columns (FEJ path overwrites when active)
          omega_end_lin = bridge->w_end;
          v_end_lin = v_end_val;
          R_clone_lin = R_clone_val;
          // Remaining ESTIMATED shift, exact SO(3) under constant endpoint kinematics
          if (have_clone_kin && std::abs(dt_total) > 1e-10) {
            R_GtoIi = exp_so3(-bridge->w_end * dt_total) * R_GtoIi;
            p_IiinG = p_IiinG + v_end_val * dt_total;
          }
        } else if (have_clone_kin && std::abs(dt_total) > 1e-10) {
          R_GtoIi = exp_so3(-omega_clone_val * dt_total) * R_GtoIi;
          p_IiinG = p_IiinG + v_clone_val * dt_total;
        }
      }

      // Get current feature in the IMU
      Eigen::Vector3d p_FinIi = R_GtoIi * (p_FinG - p_IiinG);

      // Project the current feature into the current frame of reference
      Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
      Eigen::Vector2d uv_norm;
      uv_norm << p_FinCi(0) / p_FinCi(2), p_FinCi(1) / p_FinCi(2);

      // Distort the normalized coordinates (radtan or fisheye)
      Eigen::Vector2d uv_dist;
      uv_dist = camera_model->distort_d(uv_norm);

      // Our residual
      Eigen::Vector2d uv_m;
      uv_m << (double)cam_uvs.at(m)(0), (double)cam_uvs.at(m)(1);
      res.block(2 * c, 0, 2, 1) = uv_m - uv_dist;

      //=========================================================================
      //=========================================================================

      // If we are doing first estimate Jacobians, then overwrite with the first estimates
      if (state->_options.do_fej) {
        R_GtoIi = clone_Ii->Rot_fej();
        p_IiinG = clone_Ii->pos_fej();
        // Apply async timeoffset and RS correction to FEJ values. With a bridge, the KNOWN
        // residual composes at the FIXED build-time linearization (b0), which is exactly what
        // FEJ prescribes; the estimated remainder composes exactly on SO(3) at the endpoint kinematics.
        R_clone_lin = R_GtoIi;
        if (need_obs_terms) {
          double dt_total_fej = dt_camoff_lin + ((bridge == nullptr) ? dt_epoch : 0.0);
          if (rs_on_linearization) {
            dt_total_fej += (v_pixel * inv_img_h - 0.5) * t_readout_lin;
          }
          if (bridge != nullptr) {
            const Eigen::Matrix3d R_clone_fej = R_GtoIi;
            R_GtoIi = bridge->DR * R_clone_fej;
            p_IiinG = p_IiinG + v_clone_lin * bridge->dt + bridge->p_grav + R_clone_fej.transpose() * bridge->alpha;
            v_end_lin = v_clone_lin + bridge->v_grav + R_clone_fej.transpose() * bridge->beta;
            omega_end_lin = bridge->w_end;
            if (have_clone_kin && std::abs(dt_total_fej) > 1e-10) {
              R_GtoIi = exp_so3(-omega_end_lin * dt_total_fej) * R_GtoIi;
              p_IiinG = p_IiinG + v_end_lin * dt_total_fej;
            }
          } else if (have_clone_kin && std::abs(dt_total_fej) > 1e-10) {
            R_GtoIi = exp_so3(-omega_clone_lin * dt_total_fej) * R_GtoIi;
            p_IiinG = p_IiinG + v_clone_lin * dt_total_fej;
          }
        }
        // R_ItoC = calibration->Rot_fej();
        // p_IinC = calibration->pos_fej();
        p_FinIi = R_GtoIi * (p_FinG_fej - p_IiinG);
        p_FinCi = R_ItoC * p_FinIi + p_IinC;
        // uv_norm << p_FinCi(0)/p_FinCi(2),p_FinCi(1)/p_FinCi(2);
        // cam_d = state->get_intrinsics_CAM(pair.first)->fej();
      }

      // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
      Eigen::MatrixXd dz_dzn, dz_dzeta;
      camera_model->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

      // Normalized coordinates in respect to projection function
      Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
      dzn_dpfc << 1 / p_FinCi(2), 0, -p_FinCi(0) / (p_FinCi(2) * p_FinCi(2)), 0, 1 / p_FinCi(2), -p_FinCi(1) / (p_FinCi(2) * p_FinCi(2));

      // Derivative of p_FinCi in respect to p_FinIi
      Eigen::MatrixXd dpfc_dpfg = R_ItoC * R_GtoIi;

      // Derivative of p_FinCi in respect to camera clone state
      Eigen::MatrixXd dpfc_dclone = Eigen::MatrixXd::Zero(3, 6);
      dpfc_dclone.block(0, 0, 3, 3).noalias() = R_ItoC * skew_x(p_FinIi);
      dpfc_dclone.block(0, 3, 3, 3) = -dpfc_dpfg;

      //=========================================================================
      //=========================================================================

      // Precompute some matrices
      Eigen::MatrixXd dz_dpfc = dz_dzn * dzn_dpfc;
      Eigen::MatrixXd dz_dpfg = dz_dpfc * dpfc_dpfg;

      // CHAINRULE: get the total feature Jacobian
      H_f.block(2 * c, 0, 2, H_f.cols()).noalias() = dz_dpfg * dpfg_dlambda;

      // CHAINRULE: get state clone Jacobian
      H_x.block(2 * c, map_hx[clone_Ii], 2, clone_Ii->size()).noalias() = dz_dpfc * dpfc_dclone;

      // CHAINRULE: loop through all extra states and add their
      // NOTE: we add the Jacobian here as we might be in the anchoring pose for this measurement
      for (size_t i = 0; i < dpfg_dx_order.size(); i++) {
        H_x.block(2 * c, map_hx[dpfg_dx_order.at(i)], 2, dpfg_dx_order.at(i)->size()).noalias() += dz_dpfg * dpfg_dx.at(i);
      }

      //=========================================================================
      //=========================================================================

      // Derivative of p_FinCi in respect to camera calibration (R_ItoC, p_IinC)
      if (state->_options.do_calib_camera_pose) {

        // Calculate the Jacobian
        Eigen::MatrixXd dpfc_dcalib = Eigen::MatrixXd::Zero(3, 6);
        dpfc_dcalib.block(0, 0, 3, 3) = skew_x(p_FinCi - p_IinC);
        dpfc_dcalib.block(0, 3, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();

        // Chainrule it and add it to the big jacobian
        H_x.block(2 * c, map_hx[calibration], 2, calibration->size()).noalias() += dz_dpfc * dpfc_dcalib;
      }

      // Derivative of measurement in respect to distortion parameters
      if (state->_options.do_calib_camera_intrinsics) {
        H_x.block(2 * c, map_hx[distortion], 2, distortion->size()) = dz_dzeta;
      }

      // Temporal-column linearization kinematics: bridge endpoint when available (the sampling
      // instant), else the clone-time cache
      const Eigen::Vector3d &w_col = (bridge != nullptr) ? omega_end_lin : omega_clone_lin;
      const Eigen::Vector3d &v_col = (bridge != nullptr) ? v_end_lin : v_clone_lin;

      // Derivative of measurement in respect to rolling shutter readout time
      // (column skipped when the window motion is degenerate or this clone has no kinematics)
      if (readout_hx_col >= 0 && dt_rs_cols_active && have_clone_kin) {
        double v_frac = v_pixel * inv_img_h - 0.5;
        Eigen::Vector3d dpfI_dtrd = -(skew_x(w_col) * p_FinIi + R_GtoIi * v_col) * v_frac;
        H_x.block(2 * c, readout_hx_col, 2, 1).noalias() += dz_dpfc * R_ItoC * dpfI_dtrd;
      }

      // Derivative of measurement in respect to camera-imu timeoffsets (dz/d(dt_cam) = +dz_ddt and
      // dz/d(dt_ref) = -dz_ddt since the effective quantity is dt_cam - dt_ref; the reference
      // camera's own measurements contribute nothing by construction)
      if (state->_options.do_calib_camera_timeoffset && dt_rs_cols_active && have_clone_kin && dt_cam_hx_col >= 0 &&
          dt_ref_hx_col >= 0 && dt_cam_hx_col != dt_ref_hx_col) {
        Eigen::Vector3d dpfI_ddt = -(skew_x(w_col) * p_FinIi + R_GtoIi * v_col);
        Eigen::Matrix<double, 2, 1> dz_ddt = dz_dpfc * R_ItoC * dpfI_ddt;
        H_x.block(2 * c, dt_cam_hx_col, 2, 1).noalias() += dz_ddt;
        H_x.block(2 * c, dt_ref_hx_col, 2, 1).noalias() -= dz_ddt;
      }

      // Analytic IMU-bias columns from the bridge: the composed pose depends on the biases through
      // the preintegration (d theta_m/db = J_th, d p_m/db = R_k^T J_alpha), a consistency term the
      // first-order schemes drop entirely
      if (bridge != nullptr && bias_g_hx_col >= 0 && bias_a_hx_col >= 0) {
        const Eigen::Matrix<double, 3, 6> dth_db = bridge->J_b.block(0, 0, 3, 6);
        const Eigen::Matrix<double, 3, 6> dal_db = bridge->J_b.block(3, 0, 3, 6);
        const Eigen::Matrix<double, 2, 6> H_bias =
            dz_dpfc * (R_ItoC * skew_x(p_FinIi) * dth_db + (-R_ItoC * R_GtoIi) * (R_clone_lin.transpose() * dal_db));
        H_x.block(2 * c, bias_g_hx_col, 2, 3).noalias() += H_bias.block(0, 0, 2, 3);
        H_x.block(2 * c, bias_a_hx_col, 2, 3).noalias() += H_bias.block(0, 3, 2, 3);
      }

      // Move the Jacobian and residual index forward
      c++;
    }
  }
}

void UpdaterHelper::nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

  // Apply the left nullspace of H_f to all variables
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_f.cols(); ++n) {
    for (int m = (int)H_f.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_x.block(m - 1, 0, 2, H_x.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
  // NOTE: need to eigen3 eval here since this experiences aliasing!
  // H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
  H_x = H_x.block(H_f.cols(), 0, H_x.rows() - H_f.cols(), H_x.cols()).eval();
  res = res.block(H_f.cols(), 0, res.rows() - H_f.cols(), res.cols()).eval();

  // Sanity check
  assert(H_x.rows() == res.rows());
}

void UpdaterHelper::measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

  // Return if H_x is a fat matrix (there is no need to compress in this case)
  if (H_x.rows() <= H_x.cols())
    return;

  // Do measurement compression through givens rotations
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_x.cols(); n++) {
    for (int m = (int)H_x.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_x(m - 1, n), H_x(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_x.block(m - 1, n, 2, H_x.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // If H is a fat matrix, then use the rows
  // Else it should be same size as our state
  int r = std::min(H_x.rows(), H_x.cols());

  // Construct the smaller jacobian and residual after measurement compression
  assert(r <= H_x.rows());
  H_x.conservativeResize(r, H_x.cols());
  res.conservativeResize(r, res.cols());
}
