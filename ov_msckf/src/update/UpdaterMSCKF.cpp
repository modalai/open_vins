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

#include "UpdaterMSCKF.h"

#include "UpdaterHelper.h"
#include "RejectStats.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <cmath>

#include "utils/ChronoProf.h"
#include "utils/chi_square/chi_squared_quantile_table_0_95.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterMSCKF::UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options) : _options(options) {

  // Save our raw pixel noise squared
  _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Chi-squared 0.95 gating thresholds come from the baked table
  // (utils/chi_square/, bit-identical to the boost::math values this used to compute here)
}

void UpdaterMSCKF::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  ov_core::ProfTime rT0, rT1, rT2, rT3, rT4, rT5;
  rT0 = ov_core::prof_now();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // 1. Clean all feature measurements and make sure they all have valid clone times
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {

    // Clean the feature
    (*it0)->clean_old_measurements(clonetimes);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // Remove if we don't have enough
    if (ct_meas < 2) {
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = ov_core::prof_now();

  // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
  // RS row-anchor convention (rs_convention: top 0.0 / center 0.5 / bottom 1.0)
  const double rs_row_anchor = state->_options.rs_row_anchor;
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  for (const auto &clone_calib : state->_calib_IMUtoCAM) {

    // For this camera, create the vector of camera poses
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    const double dt_cam_delta = state->cam_imu_dt_delta(clone_calib.first);
    for (const auto &clone_imu : state->_clones_IMU) {

      // Get current IMU pose corrected to this camera's sampling instant: EXACT bridge
      // composition when the frame was epoch-snapped (bias correction unnecessary at
      // triangulation accuracy), else the constant-kinematics exact-SO(3) model over the full correction
      Eigen::Matrix<double, 3, 3> R_GtoIi = clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_IiinG = clone_imu.second->pos();
      const PreintBridgeData *br = state->epoch_bridge(clone_calib.first, clone_imu.first);
      const double dt_extra = dt_cam_delta + ((br == nullptr) ? state->epoch_residual(clone_calib.first, clone_imu.first) : 0.0);
      auto kin_it = state->_clones_kinematics.find(clone_imu.first);
      const bool have_kin = (kin_it != state->_clones_kinematics.end());
      if (br != nullptr && have_kin) {
        const Eigen::Matrix3d R_clone = R_GtoIi;
        R_GtoIi = br->DR * R_clone;
        p_IiinG = p_IiinG + kin_it->second.vel * br->dt + br->p_grav + R_clone.transpose() * br->alpha;
        if (std::abs(dt_extra) > 1e-10) {
          const Eigen::Vector3d v_end = kin_it->second.vel + br->v_grav + R_clone.transpose() * br->beta;
          R_GtoIi = exp_so3(-br->w_end * dt_extra) * R_GtoIi;
          p_IiinG = p_IiinG + v_end * dt_extra;
        }
      } else if (std::abs(dt_extra) > 1e-10) {
        if (have_kin) {
          R_GtoIi = exp_so3(-kin_it->second.omega * dt_extra) * R_GtoIi;
          p_IiinG = p_IiinG + kin_it->second.vel * dt_extra;
        } else {
          state->_kin_miss_count++;
        }
      }

      // Get current camera pose
      Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * R_GtoIi;
      Eigen::Matrix<double, 3, 1> p_CioinG = p_IiinG - R_GtoCi.transpose() * clone_calib.second->pos();

      // Append to our map
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }

    // Append to our map
    clones_cam.insert({clone_calib.first, clones_cami});
  }

  // Check if any camera has rolling shutter (for triangulation RS correction)
  bool has_rolling_shutter = false;
  for (const auto &pair : state->_calib_camera_readout) {
    if (std::abs(pair.second->value()(0)) > 1e-10) {
      has_rolling_shutter = true;
      break;
    }
  }

  // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
  RejectCounters rc; // DIAGNOSTIC: stereo-vs-mono gate-level reject accounting
  auto it1 = feature_vec.begin();
  while (it1 != feature_vec.end()) {

    // Apply per-observation rolling shutter correction to clone poses for this feature
    std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam_rs;
    auto *clones_for_tri = &clones_cam;
    if (has_rolling_shutter) {
      clones_cam_rs = clones_cam;
      for (const auto &obs_pair : (*it1)->timestamps) {
        size_t cam_id = obs_pair.first;
        if (state->_calib_camera_readout.find(cam_id) == state->_calib_camera_readout.end()) continue;
        double t_readout = state->_calib_camera_readout.at(cam_id)->value()(0);
        if (std::abs(t_readout) < 1e-10) continue;
        if (state->_cam_intrinsics_cameras.find(cam_id) == state->_cam_intrinsics_cameras.end()) continue;
        double inv_img_h = 1.0 / (double)state->_cam_intrinsics_cameras.at(cam_id)->h();
        Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(cam_id)->Rot();
        Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(cam_id)->pos();
        for (size_t m = 0; m < obs_pair.second.size(); m++) {
          double clone_time = obs_pair.second.at(m);
          if (clones_cam_rs.find(cam_id) == clones_cam_rs.end()) continue;
          if (clones_cam_rs.at(cam_id).find(clone_time) == clones_cam_rs.at(cam_id).end()) continue;
          if (state->_clones_kinematics.find(clone_time) == state->_clones_kinematics.end()) continue;
          double v_pixel = (double)(*it1)->uvs.at(cam_id).at(m)(1);
          double dt_rs = (v_pixel * inv_img_h - rs_row_anchor) * t_readout;
          if (std::abs(dt_rs) < 1e-10) continue;
          // Recover IMU pose from camera pose (undo camera transform)
          Eigen::Matrix3d R_GtoCi = clones_cam_rs.at(cam_id).at(clone_time).Rot();
          Eigen::Vector3d p_CiinG = clones_cam_rs.at(cam_id).at(clone_time).pos();
          Eigen::Matrix3d R_GtoIi = R_ItoC.transpose() * R_GtoCi;
          Eigen::Vector3d p_IiinG = p_CiinG + R_GtoCi.transpose() * p_IinC;
          // Apply RS correction in IMU frame -- at the SAME kinematics the residual path
          // transports this row time with: bridge ENDPOINT (w_end, v_end) when the frame was
          // epoch-snapped, else the clone-time cache. Triangulating the row warp at clone-time
          // kinematics while the update linearizes it at the endpoint left the landmark init
          // inconsistent with the measurement model at O((w_end - w_clone) * dt_rs).
          const State::CloneKinematics &kin = state->_clones_kinematics.at(clone_time);
          Eigen::Vector3d w_rs = kin.omega;
          Eigen::Vector3d v_rs = kin.vel;
          const PreintBridgeData *br_rs = state->epoch_bridge(cam_id, clone_time);
          if (br_rs != nullptr && state->_clones_IMU.find(clone_time) != state->_clones_IMU.end()) {
            const Eigen::Matrix3d R_clone = state->_clones_IMU.at(clone_time)->Rot();
            w_rs = br_rs->w_end;
            v_rs = kin.vel + br_rs->v_grav + R_clone.transpose() * br_rs->beta;
          }
          R_GtoIi = exp_so3(-w_rs * dt_rs) * R_GtoIi;
          p_IiinG = p_IiinG + v_rs * dt_rs;
          // Recompute camera pose
          R_GtoCi = R_ItoC * R_GtoIi;
          p_CiinG = p_IiinG - R_GtoCi.transpose() * p_IinC;
          clones_cam_rs[cam_id][clone_time] = FeatureInitializer::ClonePose(R_GtoCi, p_CiinG);
        }
      }
      clones_for_tri = &clones_cam_rs;
    }

    // DIAGNOSTIC: feature is "stereo" this update if observed in >1 camera.
    bool is_stereo = (*it1)->timestamps.size() > 1;
    if (is_stereo) rc.s_n++; else rc.m_n++;

    // Triangulate the feature and remove if it fails
    FeatureInitializer::FailReason tri_reason = FeatureInitializer::FailReason::NONE;
    bool success_tri = true;
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(*it1, *clones_for_tri);
    } else {
      success_tri = initializer_feat->single_triangulation(*it1, *clones_for_tri, &tri_reason);
    }

    // Gauss-newton refine the feature
    FeatureInitializer::FailReason gn_reason = FeatureInitializer::FailReason::NONE;
    bool success_refine = true;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(*it1, *clones_for_tri, &gn_reason);
    }

    // Remove the feature if not a success
    if (!success_tri || !success_refine) {
      if (kEnableRejectDiag) {
        using FR = FeatureInitializer::FailReason;
        FR r = (!success_tri) ? tri_reason : gn_reason;
        switch (r) {
          case FR::TRI_COND:    is_stereo ? rc.s_tri_cond++  : rc.m_tri_cond++;  break;
          case FR::TRI_DEPTH:   is_stereo ? rc.s_tri_depth++ : rc.m_tri_depth++; break;
          case FR::TRI_NAN:     is_stereo ? rc.s_tri_nan++   : rc.m_tri_nan++;   break;
          case FR::GN_BASELINE: is_stereo ? rc.s_gn_base++   : rc.m_gn_base++;   break;
          case FR::GN_DEPTH:    is_stereo ? rc.s_gn_depth++  : rc.m_gn_depth++;  break;
          case FR::GN_NAN:      is_stereo ? rc.s_gn_nan++    : rc.m_gn_nan++;     break;
          default: break; // 1d path or unclassified
        }
      }
      (*it1)->to_delete = true;
      it1 = feature_vec.erase(it1);
      continue;
    }
    it1++;
  }
  rT2 = ov_core::prof_now();

  // Calculate the max possible measurement size
  size_t max_meas_size = 0;
  for (size_t i = 0; i < feature_vec.size(); i++) {
    for (const auto &pair : feature_vec.at(i)->timestamps) {
      max_meas_size += 2 * feature_vec.at(i)->timestamps[pair.first].size();
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
  size_t max_hx_size = state->max_covariance_size();
  for (auto &landmark : state->_features_SLAM) {
    max_hx_size -= landmark.second->size();
  }

  // Large Jacobian and residual of *all* features for this update
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

  // 4. Compute linear system for each feature, nullspace project, and reject
  auto it2 = feature_vec.begin();
  while (it2 != feature_vec.end()) {

    // Convert our feature into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = (*it2)->featid;
    feat.uvs = (*it2)->uvs;
    feat.uvs_norm = (*it2)->uvs_norm;
    feat.timestamps = (*it2)->timestamps;

    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    feat.feat_representation = state->_options.feat_rep_msckf;
    if (state->_options.feat_rep_msckf == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = (*it2)->anchor_cam_id;
      feat.anchor_clone_timestamp = (*it2)->anchor_clone_timestamp;
      feat.p_FinA = (*it2)->p_FinA;
      feat.p_FinA_fej = (*it2)->p_FinA;
    } else {
      feat.p_FinG = (*it2)->p_FinG;
      feat.p_FinG_fej = (*it2)->p_FinG;
    }

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // Nullspace project
    UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

    /// Chi2 distance check
    // Per-feature measurement noise: a feature seen in >1 camera (stereo) is
    // weighted with a larger sigma than a mono feature, because cross-camera ZNCC
    // matches are noisier than same-camera KLT temporal tracks. sigma_pix_sq_stereo
    // falls back to sigma_pix_sq when no stereo value is configured.
    bool is_stereo = feat.timestamps.size() > 1;
    double sigma2_f = is_stereo ? _options.sigma_pix_sq_stereo : _options.sigma_pix_sq;
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
    S.diagonal() += sigma2_f * Eigen::VectorXd::Ones(S.rows());
    double chi2 = res.dot(S.llt().solve(res));

    // Threshold from the baked quantile table (full reachable dof range; no runtime solve)
    double chi2_check = ov_core::chi_squared_quantile_0_95((int)res.rows());

    // Check if we should delete or not
    if (chi2 > _options.chi2_multipler * chi2_check) {
      if (kEnableRejectDiag) { if (is_stereo) rc.s_chi2++; else rc.m_chi2++; }
      (*it2)->to_delete = true;
      it2 = feature_vec.erase(it2);
      // PRINT_DEBUG("featid = %d\n", feat.featid);
      // PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options.chi2_multipler*chi2_check);
      // std::stringstream ss;
      // ss << "res = " << std::endl << res.transpose() << std::endl;
      // PRINT_DEBUG(ss.str().c_str());
      continue;
    }
    if (kEnableRejectDiag) { if (is_stereo) rc.s_accept++; else rc.m_accept++; }

    // Whiten this feature's rows by 1/sigma_f so the stacked system has isotropic
    // unit noise. Required because the global measurement compression and the EKF
    // update below assume R = I. For a mono feature (sigma2_f == sigma_pix_sq) this
    // is identical to the previous R = sigma_pix_sq * I formulation.
    double inv_sigma_f = 1.0 / std::sqrt(sigma2_f);
    H_x *= inv_sigma_f;
    res *= inv_sigma_f;

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;
    for (const auto &var : Hx_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
      ct_hx += var->size();
    }

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();
    it2++;
  }
  rT3 = ov_core::prof_now();

  // DIAGNOSTIC: emit per-update gate-reject accounting (state ts for yaw align).
  if (kEnableRejectDiag) {
    log_reject_stats(state->_timestamp, "MSCKF", rc);
  }

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  for (size_t f = 0; f < feature_vec.size(); f++) {
    feature_vec[f]->to_delete = true;
  }

  // Return if we don't have anything and resize our matrices
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  res_big.conservativeResize(ct_meas, 1);
  Hx_big.conservativeResize(ct_meas, ct_jacob);

  // 5. Perform measurement compression
  UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
  if (Hx_big.rows() < 1) {
    return;
  }
  rT4 = ov_core::prof_now();

  // Each feature's rows were whitened by 1/sigma_f above (per-feature stereo/mono
  // noise), so the stacked & compressed system already has unit isotropic noise.
  Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

  // 6. With all good features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  rT5 = ov_core::prof_now();

  // Debug print timing information
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to clean\n", ov_core::prof_s(rT0, rT1));
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to triangulate\n", ov_core::prof_s(rT1, rT2));
  PRINT_ALL("[MSCKF-UP]: %.4f seconds create system (%d features)\n", ov_core::prof_s(rT2, rT3), (int)feature_vec.size());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds compress system\n", ov_core::prof_s(rT3, rT4));
  PRINT_ALL("[MSCKF-UP]: %.4f seconds update state (%d size)\n", ov_core::prof_s(rT4, rT5), (int)res_big.rows());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds total\n", ov_core::prof_s(rT1, rT5));
}
