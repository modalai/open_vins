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

#include "UpdaterSLAM.h"

#include "UpdaterHelper.h"
#include "RejectStats.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <cmath>
#include <iterator>
#include <set>

#include "utils/ChronoProf.h"
#include "utils/chi_square/chi_squared_quantile_table_0_95.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterSLAM::UpdaterSLAM(UpdaterOptions &options_slam, UpdaterOptions &options_aruco, ov_core::FeatureInitializerOptions &feat_init_options)
    : _options_slam(options_slam), _options_aruco(options_aruco) {

  // Save our raw pixel noise squared
  _options_slam.sigma_pix_sq = std::pow(_options_slam.sigma_pix, 2);
  _options_aruco.sigma_pix_sq = std::pow(_options_aruco.sigma_pix, 2);

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Chi-squared 0.95 gating thresholds come from the baked table
  // (utils/chi_square/, bit-identical to the boost::math values this used to compute here)
}

void UpdaterSLAM::delayed_init(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec,
                               const std::unordered_map<size_t, StereoMatchConfidence> *stereo_confidence) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  ov_core::ProfTime rT0, rT1, rT2, rT3;
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
          default: break;
        }
      }
      (*it1)->to_delete = true;
      it1 = feature_vec.erase(it1);
      continue;
    }
    it1++;
  }
  rT2 = ov_core::prof_now();

  // 4. Compute linear system for each feature, nullspace project, and reject.
  // try_init() builds the linear system for an ALREADY-TRIANGULATED feature `f`
  // (f.p_FinA / f.anchor_cam_id set) and attempts the delayed SLAM init; on success
  // it inserts the landmark into the state and returns true. Factored into a lambda
  // so stereo->mono graceful degrade can call it a second time on the
  // mono-stripped feature when the stereo form fails the chi2 gate.
  auto try_init = [&](ov_core::Feature &f) -> bool {
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = f.featid;
    feat.uvs = f.uvs;
    feat.uvs_norm = f.uvs_norm;
    feat.timestamps = f.timestamps;
    feat.quality = f.quality;

    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    auto feat_rep =
        ((int)feat.featid < state->_options.max_aruco_features) ? state->_options.feat_rep_aruco : state->_options.feat_rep_slam;
    feat.feat_representation = feat_rep;
    if (feat_rep == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = f.anchor_cam_id;
      feat.anchor_clone_timestamp = f.anchor_clone_timestamp;
      feat.p_FinA = f.p_FinA;
      feat.p_FinA_fej = f.p_FinA;
    } else {
      feat.p_FinG = f.p_FinG;
      feat.p_FinG_fej = f.p_FinG;
    }

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // If we are doing the single feature representation, then we need to remove the bearing portion
    // To do so, we project the bearing portion onto the state and depth Jacobians and the residual.
    // This allows us to directly initialize the feature as a depth-old feature
    if (feat_rep == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

      // Append the Jacobian in respect to the depth of the feature
      Eigen::MatrixXd H_xf = H_x;
      H_xf.conservativeResize(H_x.rows(), H_x.cols() + 1);
      H_xf.block(0, H_x.cols(), H_x.rows(), 1) = H_f.block(0, H_f.cols() - 1, H_f.rows(), 1);
      H_f.conservativeResize(H_f.rows(), H_f.cols() - 1);

      // Nullspace project the bearing portion
      // This takes into account that we have marginalized the bearing already
      // Thus this is crucial to ensuring estimator consistency as we are not taking the bearing to be true
      UpdaterHelper::nullspace_project_inplace(H_f, H_xf, res);

      // Split out the state portion and feature portion
      H_x = H_xf.block(0, 0, H_xf.rows(), H_xf.cols() - 1);
      H_f = H_xf.block(0, H_xf.cols() - 1, H_xf.rows(), 1);
    }

    // Create feature pointer (we will always create it of size three since we initialize the single invese depth as a msckf anchored
    // representation)
    int landmark_size = (feat_rep == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 1 : 3;
    auto landmark = std::make_shared<Landmark>(landmark_size);
    landmark->_featid = feat.featid;
    landmark->_feat_representation = feat_rep;
    landmark->_unique_camera_id = f.anchor_cam_id;
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      landmark->_anchor_cam_id = feat.anchor_cam_id;
      landmark->_anchor_clone_timestamp = feat.anchor_clone_timestamp;
      landmark->set_from_xyz(feat.p_FinA, false);
      landmark->set_from_xyz(feat.p_FinA_fej, true);
    } else {
      landmark->set_from_xyz(feat.p_FinG, false);
      landmark->set_from_xyz(feat.p_FinG_fej, true);
    }
    landmark->_quality = feat.quality;

    bool is_aruco = (int)feat.featid < state->_options.max_aruco_features;
    bool is_stereo = feat.timestamps.size() > 1; // for the reject-diag accounting below
    double sigma_pix_sq = is_aruco ? _options_aruco.sigma_pix_sq : _options_slam.sigma_pix_sq;
    Eigen::MatrixXd R = sigma_pix_sq * Eigen::MatrixXd::Identity(res.rows(), res.rows());
    double chi2_multipler = is_aruco ? _options_aruco.chi2_multipler : _options_slam.chi2_multipler;
    if (StateHelper::initialize(state, landmark, Hx_order, H_x, H_f, R, res, chi2_multipler)) {
      state->_features_SLAM.insert({f.featid, landmark});

      // DIAGNOSTIC: confirm/refute whether "landmarked too close" flicker traces to a
      // feature being marginalized then re-triangulated from a thin/short-baseline
      // observation set. See RejectStats.h ReinitEvent for details.
      if (kEnableReinitDiag) {
        std::set<double> distinct_ts;
        int n_obs = 0;
        for (auto const &pair : f.timestamps) {
          n_obs += (int)pair.second.size();
          for (double t : pair.second)
            distinct_ts.insert(t);
        }
        ReinitEvent ev;
        ev.meas_ts = state->_timestamp;
        ev.featid = f.featid;
        ev.is_stereo = is_stereo;
        ev.is_reinit = reinit_mark_and_check(f.featid);
        ev.n_cams = (int)f.timestamps.size();
        ev.n_obs = n_obs;
        ev.n_distinct_ts = (int)distinct_ts.size();
        ev.time_span_s = distinct_ts.empty() ? 0.0 : (*distinct_ts.rbegin() - *distinct_ts.begin());
        ev.depth_anchor = f.p_FinA(2);

        // For the single-instant 2-cam stereo case: recompute the SAME baseline vector
        // FeatureInitializer used (anchor pose vs. the other camera's pose, both read
        // straight from clones_cam) to check whether the triangulation actually saw the
        // true known extrinsic separation, or something anomalous (a pose/lookup bug).
        if (ev.n_cams == 2 && ev.n_distinct_ts == 1) {
          try {
            double t0 = *distinct_ts.begin();
            size_t other_cam = ((int)f.timestamps.begin()->first == f.anchor_cam_id) ? std::next(f.timestamps.begin())->first
                                                                                      : f.timestamps.begin()->first;
            FeatureInitializer::ClonePose anchor_pose = clones_cam.at(f.anchor_cam_id).at(f.anchor_clone_timestamp);
            FeatureInitializer::ClonePose other_pose = clones_cam.at(other_cam).at(t0);
            Eigen::Vector3d p_CiinA = anchor_pose.Rot() * (other_pose.pos() - anchor_pose.pos());
            ev.baseline_norm = p_CiinA.norm();
          } catch (const std::exception &e) {
            ev.baseline_norm = -1.0;
          }
        }

        // Matcher's own confidence for this feature's L/R correspondence, if VioManager
        // supplied one (see UpdaterSLAM.h delayed_init doc). Left at ReinitEvent's "no
        // data" sentinel defaults if unavailable.
        if (stereo_confidence != nullptr) {
          auto it_conf = stereo_confidence->find(f.featid);
          if (it_conf != stereo_confidence->end()) {
            ev.peak_zncc = it_conf->second.peak_zncc;
            ev.margin = it_conf->second.margin;
            ev.lr_err = it_conf->second.lr_err;
          }
        }

        log_reinit_event(ev);
      }

      return true;
    }
    return false;
  };

  // build_mono: strip a feature to its anchor camera's observations and re-triangulate
  // as mono. Returns the mono Feature (p_FinA set) or nullptr if it cannot triangulate
  // (e.g. no parallax during hover). Used by the stereo->mono chi2-fail demote below.
  auto build_mono = [&](ov_core::Feature &f) -> std::shared_ptr<ov_core::Feature> {
    auto mono = std::make_shared<ov_core::Feature>(f); // deep copy of the feature
    int anchor = f.anchor_cam_id;                      // the cam with most obs (set in loop 1)
    // remove EVERY camera's observations except the anchor's
    for (auto mit = mono->timestamps.begin(); mit != mono->timestamps.end();) {
      if ((int)mit->first != anchor) {
        mono->uvs.erase(mit->first);
        mono->uvs_norm.erase(mit->first);
        mit = mono->timestamps.erase(mit);
      } else {
        ++mit;
      }
    }
    // re-triangulate the now single-camera (temporal) track
    bool ok = mono->timestamps.count(anchor) && mono->timestamps.at(anchor).size() >= 2 &&
              initializer_feat->single_triangulation(mono, clones_cam) &&
              (!initializer_feat->config().refine_features || initializer_feat->single_gaussnewton(mono, clones_cam));
    return ok ? mono : nullptr;
  };
  auto mark_demoted = [&](size_t featid) {
    auto lm = state->_features_SLAM.find(featid);
    if (lm != state->_features_SLAM.end())
      lm->second->demoted_to_mono = true; // keep it mono for life (see UpdaterSLAM::update)
  };

  auto it2 = feature_vec.begin();
  while (it2 != feature_vec.end()) {
    bool was_stereo = (*it2)->timestamps.size() > 1;
    bool inserted = try_init(**it2);
    bool demoted = false;

    // stereo->mono graceful degrade on chi2 failure. Retry a failed
    // stereo feature using only the anchor camera's observations so its monocular
    // value isn't discarded; flag it so the update path keeps it mono for life.
    if (!inserted && was_stereo && kEnableStereoToMonoDemote) {
      auto mono = build_mono(**it2);
      if (mono && try_init(*mono)) { inserted = true; demoted = true; mark_demoted((*it2)->featid); }
    }

    if (kEnableRejectDiag) {
      if (inserted) {
        if (demoted)         { if (was_stereo) rc.s_demote++; else rc.m_demote++; }
        else if (was_stereo) rc.s_accept++;
        else rc.m_accept++;
      } else {
        if (was_stereo) rc.s_chi2++;
        else rc.m_chi2++;
      }
    }
    (*it2)->to_delete = true;
    if (inserted)
      it2++;
    else
      it2 = feature_vec.erase(it2);
  }
  rT3 = ov_core::prof_now();

  // DIAGNOSTIC: per-update gate-reject accounting (state ts for yaw align).
  if (kEnableRejectDiag)
    log_reject_stats(state->_timestamp, "SLAM_INIT", rc);

  // Debug print timing information
  if (!feature_vec.empty()) {
    PRINT_ALL("[SLAM-DELAY]: %.4f seconds to clean\n", ov_core::prof_s(rT0, rT1));
    PRINT_ALL("[SLAM-DELAY]: %.4f seconds to triangulate\n", ov_core::prof_s(rT1, rT2));
    PRINT_ALL("[SLAM-DELAY]: %.4f seconds initialize (%d features)\n", ov_core::prof_s(rT2, rT3), (int)feature_vec.size());
    PRINT_ALL("[SLAM-DELAY]: %.4f seconds total\n", ov_core::prof_s(rT1, rT3));
  }
}

void UpdaterSLAM::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  ov_core::ProfTime rT0, rT1, rT2, rT3;
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

    // Get the landmark and its representation
    // For single depth representation we need at least two measurement
    // This is because we do nullspace projection
    std::shared_ptr<Landmark> landmark = state->_features_SLAM.at((*it0)->featid);
    int required_meas = (landmark->_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 2 : 1;

    // Remove if we don't have enough
    if (ct_meas < 1) {
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else if (ct_meas < required_meas) {
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = ov_core::prof_now();

  // Calculate the max possible measurement size
  size_t max_meas_size = 0;
  for (size_t i = 0; i < feature_vec.size(); i++) {
    for (const auto &pair : feature_vec.at(i)->timestamps) {
      max_meas_size += 2 * feature_vec.at(i)->timestamps[pair.first].size();
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  size_t max_hx_size = state->max_covariance_size();

  // Large Jacobian, residual, and measurement noise of *all* features for this update
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(max_meas_size, max_meas_size);
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

  // 4. Compute linear system for each feature, nullspace project, and reject
  RejectCounters rc; // DIAGNOSTIC: stereo-vs-mono chi2 reject accounting (existing SLAM feats)
  auto it2 = feature_vec.begin();
  while (it2 != feature_vec.end()) {

    // Ensure we have the landmark and it is the same
    assert(state->_features_SLAM.find((*it2)->featid) != state->_features_SLAM.end());
    assert(state->_features_SLAM.at((*it2)->featid)->_featid == (*it2)->featid);

    // Get our landmark from the state
    std::shared_ptr<Landmark> landmark = state->_features_SLAM.at((*it2)->featid);

    // Convert the state landmark into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = (*it2)->featid;
    feat.uvs = (*it2)->uvs;
    feat.uvs_norm = (*it2)->uvs_norm;
    feat.timestamps = (*it2)->timestamps;
    feat.quality = (*it2)->quality;
    // a landmark admitted via stereo->mono graceful degrade must keep
    // using ONLY its anchor camera; otherwise we'd re-impose the failing stereo
    // constraint here and evict it. Strip the non-anchor cameras' observations.
    if (kEnableStereoToMonoDemote && landmark->demoted_to_mono) {
      for (auto mit = feat.timestamps.begin(); mit != feat.timestamps.end();) {
        if ((int)mit->first != landmark->_anchor_cam_id) {
          feat.uvs.erase(mit->first);
          feat.uvs_norm.erase(mit->first);
          mit = feat.timestamps.erase(mit);
        } else {
          ++mit;
        }
      }
    }
    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    feat.feat_representation = landmark->_feat_representation;
    if (landmark->_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = landmark->_anchor_cam_id;
      feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
      feat.p_FinA = landmark->get_xyz(false);
      // (*it2)->p_FinA = feat.p_FinA;
      feat.p_FinA_fej = landmark->get_xyz(true);
      (*it2)->p_FinA = feat.p_FinA_fej;
    } else {
      feat.p_FinG = landmark->get_xyz(false);
      feat.p_FinG_fej = landmark->get_xyz(true);
      (*it2)->p_FinG = feat.p_FinG_fej;
    }

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // Place Jacobians in one big Jacobian, since the landmark is already in our state vector
    Eigen::MatrixXd H_xf = H_x;
    if (landmark->_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

      // Append the Jacobian in respect to the depth of the feature
      H_xf.conservativeResize(H_x.rows(), H_x.cols() + 1);
      H_xf.block(0, H_x.cols(), H_x.rows(), 1) = H_f.block(0, H_f.cols() - 1, H_f.rows(), 1);
      H_f.conservativeResize(H_f.rows(), H_f.cols() - 1);

      // Nullspace project the bearing portion
      // This takes into account that we have marginalized the bearing already
      // Thus this is crucial to ensuring estimator consistency as we are not taking the bearing to be true
      UpdaterHelper::nullspace_project_inplace(H_f, H_xf, res);

    } else {

      // Else we have the full feature in our state, so just append it
      H_xf.conservativeResize(H_x.rows(), H_x.cols() + H_f.cols());
      H_xf.block(0, H_x.cols(), H_x.rows(), H_f.cols()) = H_f;
    }

    // Append to our Jacobian order vector
    std::vector<std::shared_ptr<Type>> Hxf_order = Hx_order;
    Hxf_order.push_back(landmark);

    // Chi2 distance check
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hxf_order);
    Eigen::MatrixXd S = H_xf * P_marg * H_xf.transpose();
    // sigma_pix_sq is reused below for this feature's R_big block.
    bool is_aruco = (int)feat.featid < state->_options.max_aruco_features;
    bool is_stereo = feat.timestamps.size() > 1; // for the reject-diag accounting below
    double sigma_pix_sq = is_aruco ? _options_aruco.sigma_pix_sq : _options_slam.sigma_pix_sq;
    S.diagonal() += sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
    double chi2 = res.dot(S.llt().solve(res));

    // Get our threshold (we precompute up to 500 but handle the case that it is more)
    // Threshold from the baked quantile table (full reachable dof range; no runtime solve)
    double chi2_check = ov_core::chi_squared_quantile_0_95((int)res.rows());

    // Check if we should delete or not
    double chi2_multipler = is_aruco ? _options_aruco.chi2_multipler : _options_slam.chi2_multipler;
    if (chi2 > chi2_multipler * chi2_check) {
      if (kEnableRejectDiag) { if (is_stereo) rc.s_chi2++; else rc.m_chi2++; }
      if ((int)feat.featid < state->_options.max_aruco_features) {
        PRINT_WARNING(YELLOW "[SLAM-UP]: rejecting aruco tag %d for chi2 thresh (%.3f > %.3f)\n" RESET, (int)feat.featid, chi2,
                      chi2_multipler * chi2_check);
      } else {
        landmark->update_fail_count++;
      }
      (*it2)->to_delete = true;
      it2 = feature_vec.erase(it2);
      continue;
    }
    if (kEnableRejectDiag) { if (is_stereo) rc.s_accept++; else rc.m_accept++; }

    // Debug print when we are going to update the aruco tags
    if ((int)feat.featid < state->_options.max_aruco_features) {
      PRINT_DEBUG("[SLAM-UP]: accepted aruco tag %d for chi2 thresh (%.3f < %.3f)\n", (int)feat.featid, chi2, chi2_multipler * chi2_check);
    }

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;
    for (const auto &var : Hxf_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_xf.rows(), var->size()) = H_xf.block(0, ct_hx, H_xf.rows(), var->size());
      ct_hx += var->size();
    }

    // Our isotropic measurement noise
    R_big.block(ct_meas, ct_meas, res.rows(), res.rows()) *= sigma_pix_sq;

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();
    it2++;
  }
  rT2 = ov_core::prof_now();

  // DIAGNOSTIC: per-update chi2-reject accounting for existing SLAM features.
  if (kEnableRejectDiag)
    log_reject_stats(state->_timestamp, "SLAM_UPD", rc);

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
  R_big.conservativeResize(ct_meas, ct_meas);

  // 5. With all good SLAM features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  rT3 = ov_core::prof_now();

  // Debug print timing information
  PRINT_ALL("[SLAM-UP]: %.4f seconds to clean\n", ov_core::prof_s(rT0, rT1));
  PRINT_ALL("[SLAM-UP]: %.4f seconds creating linear system\n", ov_core::prof_s(rT1, rT2));
  PRINT_ALL("[SLAM-UP]: %.4f seconds to update (%d feats of %d size)\n", ov_core::prof_s(rT2, rT3), (int)feature_vec.size(),
            (int)Hx_big.rows());
  PRINT_ALL("[SLAM-UP]: %.4f seconds total\n", ov_core::prof_s(rT1, rT3));
}

void UpdaterSLAM::change_anchors(std::shared_ptr<State> state) {

  // Return if we do not have enough clones
  if ((int)state->_clones_IMU.size() <= state->_options.max_clone_size) {
    return;
  }

  // Get the marginalization timestep, and change the anchor for any feature seen from it
  // NOTE: for now we have anchor the feature in the same camera as it is before
  // NOTE: this also does not change the representation of the feature at all right now
  double marg_timestep = state->margtimestep();
  for (auto &f : state->_features_SLAM) {
    // Skip any features that are in the global frame
    if (f.second->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D ||
        f.second->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_FULL_INVERSE_DEPTH)
      continue;
    // Else lets see if it is anchored in the clone that will be marginalized
    assert(marg_timestep <= f.second->_anchor_clone_timestamp);
    if (f.second->_anchor_clone_timestamp == marg_timestep) {
      perform_anchor_change(state, f.second, state->_timestamp, f.second->_anchor_cam_id);
    }
  }
}

void UpdaterSLAM::perform_anchor_change(std::shared_ptr<State> state, std::shared_ptr<Landmark> landmark, double new_anchor_timestamp,
                                        size_t new_cam_id) {

  // Assert that this is an anchored representation
  assert(LandmarkRepresentation::is_relative_representation(landmark->_feat_representation));
  assert(landmark->_anchor_cam_id != -1);

  // Create current feature representation
  UpdaterHelper::UpdaterHelperFeature old_feat;
  old_feat.featid = landmark->_featid;
  old_feat.feat_representation = landmark->_feat_representation;
  old_feat.anchor_cam_id = landmark->_anchor_cam_id;
  old_feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
  old_feat.p_FinA = landmark->get_xyz(false);
  old_feat.p_FinA_fej = landmark->get_xyz(true);

  // Get Jacobians of p_FinG wrt old representation
  Eigen::MatrixXd H_f_old;
  std::vector<Eigen::MatrixXd> H_x_old;
  std::vector<std::shared_ptr<Type>> x_order_old;
  UpdaterHelper::get_feature_jacobian_representation(state, old_feat, H_f_old, H_x_old, x_order_old);

  // Create future feature representation
  UpdaterHelper::UpdaterHelperFeature new_feat;
  new_feat.featid = landmark->_featid;
  new_feat.feat_representation = landmark->_feat_representation;
  new_feat.anchor_cam_id = new_cam_id;
  new_feat.anchor_clone_timestamp = new_anchor_timestamp;

  //==========================================================================
  //==========================================================================

  // OLD: anchor camera position and orientation
  Eigen::Matrix<double, 3, 3> R_GtoIOLD = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->Rot();
  Eigen::Matrix<double, 3, 3> R_GtoOLD = state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->Rot() * R_GtoIOLD;
  Eigen::Matrix<double, 3, 1> p_OLDinG = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->pos() -
                                         R_GtoOLD.transpose() * state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->pos();

  // NEW: anchor camera position and orientation
  Eigen::Matrix<double, 3, 3> R_GtoINEW = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->Rot();
  Eigen::Matrix<double, 3, 3> R_GtoNEW = state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->Rot() * R_GtoINEW;
  Eigen::Matrix<double, 3, 1> p_NEWinG = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->pos() -
                                         R_GtoNEW.transpose() * state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->pos();

  // Calculate transform between the old anchor and new one
  Eigen::Matrix<double, 3, 3> R_OLDtoNEW = R_GtoNEW * R_GtoOLD.transpose();
  Eigen::Matrix<double, 3, 1> p_OLDinNEW = R_GtoNEW * (p_OLDinG - p_NEWinG);
  new_feat.p_FinA = R_OLDtoNEW * landmark->get_xyz(false) + p_OLDinNEW;

  //==========================================================================
  //==========================================================================

  // OLD: anchor camera position and orientation
  Eigen::Matrix<double, 3, 3> R_GtoIOLD_fej = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->Rot_fej();
  Eigen::Matrix<double, 3, 3> R_GtoOLD_fej = state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->Rot() * R_GtoIOLD_fej;
  Eigen::Matrix<double, 3, 1> p_OLDinG_fej = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->pos_fej() -
                                             R_GtoOLD_fej.transpose() * state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->pos();

  // NEW: anchor camera position and orientation
  Eigen::Matrix<double, 3, 3> R_GtoINEW_fej = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->Rot_fej();
  Eigen::Matrix<double, 3, 3> R_GtoNEW_fej = state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->Rot() * R_GtoINEW_fej;
  Eigen::Matrix<double, 3, 1> p_NEWinG_fej = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->pos_fej() -
                                             R_GtoNEW_fej.transpose() * state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->pos();

  // Calculate transform between the old anchor and new one
  Eigen::Matrix<double, 3, 3> R_OLDtoNEW_fej = R_GtoNEW_fej * R_GtoOLD_fej.transpose();
  Eigen::Matrix<double, 3, 1> p_OLDinNEW_fej = R_GtoNEW_fej * (p_OLDinG_fej - p_NEWinG_fej);
  new_feat.p_FinA_fej = R_OLDtoNEW_fej * landmark->get_xyz(true) + p_OLDinNEW_fej;

  // Get Jacobians of p_FinG wrt new representation
  Eigen::MatrixXd H_f_new;
  std::vector<Eigen::MatrixXd> H_x_new;
  std::vector<std::shared_ptr<Type>> x_order_new;
  UpdaterHelper::get_feature_jacobian_representation(state, new_feat, H_f_new, H_x_new, x_order_new);

  //==========================================================================
  //==========================================================================

  // New phi order is just the landmark
  std::vector<std::shared_ptr<Type>> phi_order_NEW;
  phi_order_NEW.push_back(landmark);

  // Loop through all our orders and append them
  std::vector<std::shared_ptr<Type>> phi_order_OLD;
  int current_it = 0;
  std::map<std::shared_ptr<Type>, int> Phi_id_map;
  for (const auto &var : x_order_old) {
    if (Phi_id_map.find(var) == Phi_id_map.end()) {
      Phi_id_map.insert({var, current_it});
      phi_order_OLD.push_back(var);
      current_it += var->size();
    }
  }
  for (const auto &var : x_order_new) {
    if (Phi_id_map.find(var) == Phi_id_map.end()) {
      Phi_id_map.insert({var, current_it});
      phi_order_OLD.push_back(var);
      current_it += var->size();
    }
  }
  Phi_id_map.insert({landmark, current_it});
  phi_order_OLD.push_back(landmark);
  current_it += landmark->size();

  // Anchor change Jacobian
  int phisize = (new_feat.feat_representation != LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
  Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(phisize, current_it);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(phisize, phisize);

  // Inverse of our new representation
  // pf_new_error = Hfnew^{-1}*(Hfold*pf_olderror+Hxold*x_olderror-Hxnew*x_newerror)
  Eigen::MatrixXd H_f_new_inv;
  if (phisize == 1) {
    H_f_new_inv = 1.0 / H_f_new.squaredNorm() * H_f_new.transpose();
  } else {
    H_f_new_inv = H_f_new.colPivHouseholderQr().solve(Eigen::Matrix<double, 3, 3>::Identity());
  }

  // Place Jacobians for old anchor
  for (size_t i = 0; i < H_x_old.size(); i++) {
    Phi.block(0, Phi_id_map.at(x_order_old[i]), phisize, x_order_old[i]->size()).noalias() += H_f_new_inv * H_x_old[i];
  }

  // Place Jacobians for old feat
  Phi.block(0, Phi_id_map.at(landmark), phisize, phisize) = H_f_new_inv * H_f_old;

  // Place Jacobians for new anchor
  for (size_t i = 0; i < H_x_new.size(); i++) {
    Phi.block(0, Phi_id_map.at(x_order_new[i]), phisize, x_order_new[i]->size()).noalias() -= H_f_new_inv * H_x_new[i];
  }

  // Perform covariance propagation
  StateHelper::EKFPropagation(state, phi_order_NEW, phi_order_OLD, Phi, Q);

  // Set state from new feature
  landmark->_featid = new_feat.featid;
  landmark->_feat_representation = new_feat.feat_representation;
  landmark->_anchor_cam_id = new_feat.anchor_cam_id;
  landmark->_anchor_clone_timestamp = new_feat.anchor_clone_timestamp;
  landmark->set_from_xyz(new_feat.p_FinA, false);
  landmark->set_from_xyz(new_feat.p_FinA_fej, true);
  landmark->has_had_anchor_change = true;
}
