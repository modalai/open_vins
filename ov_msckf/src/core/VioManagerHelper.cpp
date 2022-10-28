/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2022 Patrick Geneva
 * Copyright (C) 2018-2022 Guoquan Huang
 * Copyright (C) 2018-2022 OpenVINS Contributors
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

#include "VioManager.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "init/InertialInitializer.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/LandmarkRepresentation.h"
#include "utils/print.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

void VioManager::initialize_with_gt(Eigen::Matrix<double, 17, 1> imustate) {
    // Initialize the system
    state->_imu->set_value(imustate.block(1, 0, 16, 1));
    state->_imu->set_fej(imustate.block(1, 0, 16, 1));

    // Fix the global yaw and position gauge freedoms
    // TODO: Why does this break out simulation consistency metrics?
    std::vector<std::shared_ptr<ov_type::Type>> order = {state->_imu};
    Eigen::MatrixXd Cov = std::pow(0.02, 2) * Eigen::MatrixXd::Identity(state->_imu->size(), state->_imu->size());
    Cov.block(3, 3, 3, 3) = std::pow(0.017, 2) * Eigen::Matrix3d::Identity();  // q
    Cov.block(3, 3, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity();   // p
    Cov.block(6, 6, 3, 3) = std::pow(0.01, 2) * Eigen::Matrix3d::Identity();   // v (static)
    StateHelper::set_initial_covariance(state, Cov, order);

    // Set the state time
    state->_timestamp = imustate(0, 0);
    startup_time = imustate(0, 0);
    is_initialized_vio = true;

    // Cleanup any features older then the initialization time
    trackFEATS->get_feature_database()->cleanup_measurements(state->_timestamp);
    if (trackARUCO != nullptr) {
        trackARUCO->get_feature_database()->cleanup_measurements(state->_timestamp);
    }

    // Print what we init'ed with
    PRINT_DEBUG(GREEN "[INIT]: INITIALIZED FROM GROUNDTRUTH FILE!!!!!\n" RESET);
    PRINT_DEBUG(GREEN "[INIT]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, state->_imu->quat()(0), state->_imu->quat()(1),
                state->_imu->quat()(2), state->_imu->quat()(3));
    PRINT_DEBUG(GREEN "[INIT]: bias gyro = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_g()(0), state->_imu->bias_g()(1),
                state->_imu->bias_g()(2));
    PRINT_DEBUG(GREEN "[INIT]: velocity = %.4f, %.4f, %.4f\n" RESET, state->_imu->vel()(0), state->_imu->vel()(1), state->_imu->vel()(2));
    PRINT_DEBUG(GREEN "[INIT]: bias accel = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_a()(0), state->_imu->bias_a()(1),
                state->_imu->bias_a()(2));
    PRINT_DEBUG(GREEN "[INIT]: position = %.4f, %.4f, %.4f\n" RESET, state->_imu->pos()(0), state->_imu->pos()(1), state->_imu->pos()(2));
}

bool VioManager::try_to_initialize(const ov_core::CameraData &message) {
    // Directly return if the initialization thread is running
    // Note that we lock on the queue since we could have finished an update
    // And are using this queue to propagate the state forward. We should wait in this case
    if (thread_init_running) {
        std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
        camera_queue_init.push_back(message.timestamp);
        return false;
    }

    // If the thread was a success, then return success!
    if (thread_init_success) {
        return true;
    }

    // Run the initialization in a second thread so it can go as slow as it desires
    thread_init_running = true;
    std::thread thread([&] {
        // Returns from our initializer
        double timestamp;
        Eigen::MatrixXd covariance;
        std::vector<std::shared_ptr<ov_type::Type>> order;
        auto init_rT1 = boost::posix_time::microsec_clock::local_time();

        // Try to initialize the system
        // We will wait for a jerk if we do not have the zero velocity update enabled
        // Otherwise we can initialize right away as the zero velocity will handle the stationary case
        bool wait_for_jerk = (updaterZUPT == nullptr);
        bool success = initializer->initialize(timestamp, covariance, order, state->_imu, wait_for_jerk);

        // If we have initialized successfully we will set the covariance and state elements as needed
        // TODO: set the clones and SLAM features here so we can start updating right away...
        if (success) {
            // Set our covariance (state should already be set in the initializer)
            StateHelper::set_initial_covariance(state, covariance, order);

            // Set the state time
            state->_timestamp = timestamp;
            startup_time = timestamp;

            // Cleanup any features older than the initialization time
            // Also increase the number of features to the desired amount during estimation
            // NOTE: we will split the total number of features over all cameras uniformly
            trackFEATS->get_feature_database()->cleanup_measurements(state->_timestamp);
            // TURI -> better to have n points tracked per cam
            trackFEATS->set_num_features(params.num_pts);
            if (trackARUCO != nullptr) {
                trackARUCO->get_feature_database()->cleanup_measurements(state->_timestamp);
            }

            // If we are moving then don't do zero velocity update4
            if (state->_imu->vel().norm() > params.zupt_max_velocity) {
                has_moved_since_zupt = true;
            }

            // Else we are good to go, print out our stats
            auto init_rT2 = boost::posix_time::microsec_clock::local_time();
            PRINT_INFO(GREEN "[init]: successful initialization in %.4f seconds\n" RESET,
                       (init_rT2 - init_rT1).total_microseconds() * 1e-6);
            PRINT_INFO(GREEN "[init]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, state->_imu->quat()(0), state->_imu->quat()(1),
                       state->_imu->quat()(2), state->_imu->quat()(3));
            PRINT_INFO(GREEN "[init]: bias gyro = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_g()(0), state->_imu->bias_g()(1),
                       state->_imu->bias_g()(2));
            PRINT_INFO(GREEN "[init]: velocity = %.4f, %.4f, %.4f\n" RESET, state->_imu->vel()(0), state->_imu->vel()(1),
                       state->_imu->vel()(2));
            PRINT_INFO(GREEN "[init]: bias accel = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_a()(0), state->_imu->bias_a()(1),
                       state->_imu->bias_a()(2));
            PRINT_INFO(GREEN "[init]: position = %.4f, %.4f, %.4f\n" RESET, state->_imu->pos()(0), state->_imu->pos()(1),
                       state->_imu->pos()(2));

            // Remove any camera times that are order then the initialized time
            // This can happen if the initialization has taken a while to perform
            std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
            std::vector<double> camera_timestamps_to_init;
            for (size_t i = 0; i < camera_queue_init.size(); i++) {
                if (camera_queue_init.at(i) > timestamp) {
                    camera_timestamps_to_init.push_back(camera_queue_init.at(i));
                }
            }

            // Now we have initialized we will propagate the state to the current timestep
            // In general this should be ok as long as the initialization didn't take too long to perform
            // Propagating over multiple seconds will become an issue if the initial biases are bad
            size_t clone_rate = (size_t)((double)camera_timestamps_to_init.size() / (double)params.state_options.max_clone_size) + 1;
            for (size_t i = 0; i < camera_timestamps_to_init.size(); i += clone_rate) {
                propagator->propagate_and_clone(state, camera_timestamps_to_init.at(i));
                StateHelper::marginalize_old_clone(state);
            }
            PRINT_DEBUG(YELLOW "[init]: moved the state forward %.2f seconds\n" RESET, state->_timestamp - timestamp);
            thread_init_success = true;
            camera_queue_init.clear();

        } else {
            auto init_rT2 = boost::posix_time::microsec_clock::local_time();
            PRINT_DEBUG(YELLOW "[init]: failed initialization in %.4f seconds\n" RESET, (init_rT2 - init_rT1).total_microseconds() * 1e-6);
            thread_init_success = false;
            std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
            camera_queue_init.clear();
        }

        // Finally, mark that the thread has finished running
        thread_init_running = false;
    });

    // If we are single threaded, then run single threaded
    // Otherwise detach this thread so it runs in the background!
    if (!params.use_multi_threading_subs) {
        thread.join();
    } else {
        thread.detach();
    }
    return false;
}

void VioManager::retriangulate_active_tracks(const ov_core::CameraData &message) {
    // Start timing
    boost::posix_time::ptime retri_rT1, retri_rT2, retri_rT3, retri_rT4, retri_rT5;
    retri_rT1 = boost::posix_time::microsec_clock::local_time();

    // Clear old active track data
    active_tracks_time = state->_timestamp;
    // active_image = message.images.at(0).clone();
    active_tracks_posinG.clear();
    active_tracks_uvd.clear();

    // Get all features which are tracked in the current frame
    // NOTE: This database should have all features from all trackers already in it
    // NOTE: it also has the complete history so we shouldn't see jumps from deleting measurements
    std::vector<std::shared_ptr<Feature>> active_features = trackDATABASE->features_containing_older(state->_timestamp);

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    std::vector<double> clonetimes;
    for (const auto &clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    //    Also remove any that we are unable to triangulate (due to not having enough measurements)
    auto it0 = active_features.begin();
    while (it0 != active_features.end()) {

        // Skip if it is a SLAM feature since it already is already going to be added
        if (state->_features_SLAM.find((*it0)->featid) != state->_features_SLAM.end()) {
            it0 = active_features.erase(it0);
            continue;
        }

        // Clean the feature
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements
        int ct_meas = 0;
        for (const auto &pair : (*it0)->timestamps) {
            ct_meas += (*it0)->timestamps[pair.first].size();
        }

        // Remove if we don't have enough and am not a SLAM feature which doesn't need triangulation
        if (ct_meas < (int)std::max(4.0, std::floor(state->_options.max_clone_size * 2.0 / 5.0))) {
            it0 = active_features.erase(it0);
        } else {
            it0++;
        }
    }
    retri_rT2 = boost::posix_time::microsec_clock::local_time();

    // Return if no features
    if (active_features.empty() && state->_features_SLAM.empty()) {
        return;
    }

    // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
    std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
    for (const auto &clone_calib : state->_calib_IMUtoCAM) {
        // For this camera, create the vector of camera poses
        std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
        for (const auto &clone_imu : state->_clones_IMU) {
            // Get current camera pose
            Eigen::Matrix3d R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
            Eigen::Vector3d p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();

            // Append to our map
            clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
        }

        // Append to our map
        clones_cam.insert({clone_calib.first, clones_cami});
    }
    retri_rT3 = boost::posix_time::microsec_clock::local_time();

    // 3. Try to triangulate all features that have measurements
    auto it1 = active_features.begin();
    while (it1 != active_features.end()) {
        // Triangulate the feature and remove if it fails
        bool success_tri = true;
        if (active_tracks_initializer->config().triangulate_1d) {
            success_tri = active_tracks_initializer->single_triangulation_1d(*it1, clones_cam);
        } else {
            success_tri = active_tracks_initializer->single_triangulation(*it1, clones_cam);
        }

        // Remove the feature if not a success
        if (!success_tri) {
            it1 = active_features.erase(it1);
            continue;
        }
        it1++;
    }
    retri_rT4 = boost::posix_time::microsec_clock::local_time();

    // Return if no features
    if (active_features.empty() && state->_features_SLAM.empty()) {
        return;
    }

    // Points which we have in the global frame
    for (const auto &feat : active_features) {
        active_tracks_posinG[feat->featid] = feat->p_FinG;
    }
    for (const auto &feat : state->_features_SLAM) {
        Eigen::Vector3d p_FinG = feat.second->get_xyz(false);
        if (LandmarkRepresentation::is_relative_representation(feat.second->_feat_representation)) {
            // Assert that we have an anchor pose for this feature
            assert(feat.second->_anchor_cam_id != -1);
            assert(feat.second->_anchor_clone_timestamp != -1);
            // Get calibration for our anchor camera
            Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feat.second->_anchor_cam_id)->Rot();
            Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feat.second->_anchor_cam_id)->pos();
            // Anchor pose orientation and position
            Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feat.second->_anchor_clone_timestamp)->Rot();
            Eigen::Vector3d p_IinG = state->_clones_IMU.at(feat.second->_anchor_clone_timestamp)->pos();
            // Feature in the global frame
            p_FinG = R_GtoI.transpose() * R_ItoC.transpose() * (feat.second->get_xyz(false) - p_IinC) + p_IinG;
        }
        active_tracks_posinG[feat.second->_featid] = p_FinG;
    }

    // Calibration of the first camera (cam0)
    std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(0);
    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(0);
    Eigen::Matrix<double, 3, 3> R_ItoC = calibration->Rot();
    Eigen::Matrix<double, 3, 1> p_IinC = calibration->pos();

    // Get current IMU clone state
    std::shared_ptr<PoseJPL> clone_Ii = state->_clones_IMU.at(active_tracks_time);
    Eigen::Matrix3d R_GtoIi = clone_Ii->Rot();
    Eigen::Vector3d p_IiinG = clone_Ii->pos();

    // 4. Next we can update our variable with the global position
    //    We also will project the features into the current frame
    for (const auto &feat : active_tracks_posinG) {

        // Project the current feature into the current frame of reference
        Eigen::Vector3d p_FinIi = R_GtoIi * (feat.second - p_IiinG);
        Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
        double depth = p_FinCi(2);
        Eigen::Vector2d uv_norm, uv_dist;
        uv_norm << p_FinCi(0) / depth, p_FinCi(1) / depth;
        uv_dist = state->_cam_intrinsics_cameras.at(0)->distort_d(uv_norm);

        // Skip if not valid (i.e. negative depth, or outside of image)
        if (depth < 0.1) {
            continue;
        }

        // Skip if not valid (i.e. negative depth, or outside of image)
        int width = state->_cam_intrinsics_cameras.at(0)->w();
        int height = state->_cam_intrinsics_cameras.at(0)->h();
        if (uv_dist(0) < 0 || (int)uv_dist(0) >= width || uv_dist(1) < 0 || (int)uv_dist(1) >= height) {
            // PRINT_DEBUG("feat %zu -> depth = %.2f | u_d = %.2f | v_d = %.2f\n",(*it2)->featid,depth,uv_dist(0),uv_dist(1));
            continue;
        }

        //////////////////////////////////////////////////////////////////
        // MAI NOTE
        // calculating depth error each time we retriangulate a feature
        /////////////////////////////////////////////////////////////////
        auto curr_feat =  trackDATABASE->get_feature(feat.first, false);

        double rho = 1 / curr_feat->p_FinA(2);
        double alpha = curr_feat->p_FinA(0) / curr_feat->p_FinA(2);
        double beta = curr_feat->p_FinA(1) / curr_feat->p_FinA(2);
        
        double error = active_tracks_initializer->compute_error(clones_cam, curr_feat, alpha, beta, rho);

        // Finally construct the uv and depth
        // MAI NOTE: also includes depth error now (last entry)
        Eigen::Vector4d uvd;
        uvd << uv_dist, depth, error;

        active_tracks_uvd.insert({feat.first, uvd});
    }
    retri_rT5 = boost::posix_time::microsec_clock::local_time();

    // Timing information
    PRINT_DEBUG(CYAN "[RETRI-TIME]: %.4f seconds for cleaning\n" RESET, (retri_rT2 - retri_rT1).total_microseconds() * 1e-6);
    PRINT_DEBUG(CYAN "[RETRI-TIME]: %.4f seconds for triangulate setup\n" RESET, (retri_rT3 - retri_rT2).total_microseconds() * 1e-6);
    PRINT_DEBUG(CYAN "[RETRI-TIME]: %.4f seconds for triangulation\n" RESET, (retri_rT4 - retri_rT3).total_microseconds() * 1e-6);
    PRINT_DEBUG(CYAN "[RETRI-TIME]: %.4f seconds for re-projection\n" RESET, (retri_rT5 - retri_rT4).total_microseconds() * 1e-6);
    PRINT_DEBUG(CYAN "[RETRI-TIME]: %.4f seconds total\n" RESET, (retri_rT5 - retri_rT1).total_microseconds() * 1e-6);
}

cv::Mat VioManager::get_historical_viz_image() {

    // Return if not ready yet
    if (state == nullptr || trackFEATS == nullptr)
        return cv::Mat();

    // Build an id-list of what features we should highlight (i.e. SLAM)
    std::vector<size_t> highlighted_ids;
    for (const auto &feat : state->_features_SLAM) {
        highlighted_ids.push_back(feat.first);
    }

    // Text we will overlay if needed
    std::string overlay = (did_zupt_update) ? "zvupt" : "";
    overlay = (!is_initialized_vio) ? "init" : overlay;

    // Get the current active tracks
    cv::Mat img_history;
    trackFEATS->display_history(img_history, 255, 255, 0, 255, 255, 255, highlighted_ids, overlay);
    if (trackARUCO != nullptr) {
        trackARUCO->display_history(img_history, 0, 255, 255, 255, 255, 255, highlighted_ids, overlay);
        // trackARUCO->display_active(img_history, 0, 255, 255, 255, 255, 255, overlay);
    }

    // Finally return the image
    return img_history;
}

std::vector<Eigen::Vector3d> VioManager::get_features_SLAM() {
    std::vector<Eigen::Vector3d> slam_feats;
    for (auto &f : state->_features_SLAM) {
        // why all this aruco bullshit everywhere
        // if ((int)f.first <= 4 * state->_options.max_aruco_features)
        // continue;
        if (ov_type::LandmarkRepresentation::is_relative_representation(f.second->_feat_representation)) {
            // Assert that we have an anchor pose for this feature
            assert(f.second->_anchor_cam_id != -1);
            assert(f.second->_anchor_clone_timestamp != -1);
            // Get calibration for our anchor camera
            Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->Rot();
            Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->pos();
            // Anchor pose orientation and position
            Eigen::Matrix<double, 3, 3> R_GtoI = state->_clones_IMU.at(f.second->_anchor_clone_timestamp)->Rot();
            Eigen::Matrix<double, 3, 1> p_IinG = state->_clones_IMU.at(f.second->_anchor_clone_timestamp)->pos();
            // Feature in the global frame
            slam_feats.push_back(R_GtoI.transpose() * R_ItoC.transpose() * (f.second->get_xyz(false) - p_IinC) + p_IinG);
        } else {
            slam_feats.push_back(f.second->get_xyz(false));
        }
    }
    return slam_feats;
}

std::vector<output_feature> VioManager::get_pixel_loc_features() {
    std::vector<output_feature> feats;

    // Build an id-list of our "in state" features
    // i.e. SLAM and last msckf update features
    std::vector<size_t> slam_ids;
    std::vector<size_t> msckf_ids(MSCKF_ids);
    std::vector<Eigen::Vector3d> good_features_MSCKF_clone(good_features_MSCKF);
    std::vector<Eigen::Vector3d> slam_feats_clone = get_features_SLAM();


    for (const auto &feat : state->_features_SLAM) {
        slam_ids.push_back(feat.second->_featid);
    }

    // get our full covariance matrix here
    Eigen::MatrixXd cov = StateHelper::get_full_covariance(state);

    std::vector<std::shared_ptr<Feature>> feats_to_draw;
    feats_to_draw = trackDATABASE->features_containing(state->_timestamp, false, false);

    for (size_t i = 0; i < feats_to_draw.size(); i++) {
        output_feature of;
        of.cam_id = feats_to_draw[i]->anchor_cam_id == -1 ? feats_to_draw[i]->first_id : feats_to_draw[i]->anchor_cam_id;
        of.id = feats_to_draw[i]->featid;
        Eigen::Vector2f pt_e = feats_to_draw[i]->uvs.at(of.cam_id).back();
        of.pix_loc[0] = pt_e[0]; 
        of.pix_loc[1] = pt_e[1];

        auto iter = std::find(slam_ids.begin(), slam_ids.end(), feats_to_draw[i]->featid);
        auto iter2 = std::find(msckf_ids.begin(), msckf_ids.end(), feats_to_draw[i]->featid);

        // slam
        if (iter != slam_ids.end()){
            auto index = std::distance(slam_ids.begin(), iter);
            of.point_quality = OV_HIGH;
            of.tsf[0] = slam_feats_clone[index](0);
            of.tsf[1] = slam_feats_clone[index](1);
            of.tsf[2] = slam_feats_clone[index](2);
            // as of now, only the slam features have a recoverable covariance
            Eigen::MatrixXf::Map(reinterpret_cast<float*>(of.p_tsf), 3, 3) = cov.block(0, state->_features_SLAM.at(feats_to_draw[i]->featid)->id(), state->_features_SLAM.at(feats_to_draw[i]->featid)->size(), state->_features_SLAM.at(feats_to_draw[i]->featid)->size()).cast<float>();
        }
        // msckf
        else if (iter2 != msckf_ids.end()){
            auto index = std::distance(msckf_ids.begin(), iter2);
            of.point_quality = OV_MEDIUM;
            of.tsf[0] = good_features_MSCKF_clone[index](0);
            of.tsf[1] = good_features_MSCKF_clone[index](1);
            of.tsf[2] = good_features_MSCKF_clone[index](2);
        }
        // oos, no 3d projections yet
        else {
            of.point_quality = OV_LOW;
        }

        if (of.point_quality != OV_LOW){
            if (active_tracks_uvd.find(feats_to_draw[i]->featid) != active_tracks_uvd.end()) {
                Eigen::Vector4d uvd = Eigen::Vector4d::Zero();
                uvd = active_tracks_uvd.at(feats_to_draw[i]->featid);
                of.depth = uvd(2); // (u,v,depth, error)
                of.depth_error_stddev = uvd(3);
            }
        }

        // special case: we are using feats for zupt state updates rather than msckf
        if (!has_moved_since_zupt && did_zupt_update && of.point_quality == OV_LOW) of.point_quality = OV_MEDIUM;

        feats.push_back(of);
    }

    return feats;
}

int VioManager::pickup_lost_slam_feats(std::vector<std::shared_ptr<Feature>> &new_feats){
    auto it0 = state->_features_SLAM_lost.begin();

    for (size_t i = 0; i < state->_cam_intrinsics.size(); i++){ // loop through cameras
        // for (size_t i = 0; i < state->_features_SLAM_lost.size(); i++){
        while (it0 != state->_features_SLAM_lost.end()) {
        
            // Calibration of the first camera (cam0)
            std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(i);
            std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(i);
            Eigen::Matrix<double, 3, 3> R_ItoC = calibration->Rot();
            Eigen::Matrix<double, 3, 1> p_IinC = calibration->pos();

            // Get current IMU clone state
            std::shared_ptr<PoseJPL> clone_Ii = state->_clones_IMU.at(state->_timestamp);
            Eigen::Matrix3d R_GtoIi = clone_Ii->Rot();
            Eigen::Vector3d p_IiinG = clone_Ii->pos();

            // Project the current feature into the current frame of reference
            Eigen::Vector3d p_FinIi = R_GtoIi * ((*it0)->get_xyz(true) - p_IiinG);
            Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
            double depth = p_FinCi(2);
            Eigen::Vector2d uv_norm, uv_dist;
            uv_norm << p_FinCi(0) / depth, p_FinCi(1) / depth;
            uv_dist = state->_cam_intrinsics_cameras.at(i)->distort_d(uv_norm);

            // Skip if not valid (i.e. negative depth, or outside of image)
            if (depth < 0.1) {
                it0++;
                continue;
            }

            // Skip if not valid (i.e. negative depth, or outside of image)
            int width = state->_cam_intrinsics_cameras.at(i)->w();
            int height = state->_cam_intrinsics_cameras.at(i)->h();
            if (uv_dist(0) < 0 || (int)uv_dist(0) >= width || uv_dist(1) < 0 || (int)uv_dist(1) >= height) {
                // PRINT_DEBUG("feat %zu -> depth = %.2f | u_d = %.2f | v_d = %.2f\n",(*it2)->featid,depth,uv_dist(0),uv_dist(1));
                it0++;
                continue;
            }

            // so uv_dist is the uv coordinates of this feature in our current image frame
            // we just need to loop through all features at the LAST state timestamp, get a group of the closest ones, and compute hamming distance
            // 20x20 patch around projected uv coordinates
            static int patch_size = 40;
            int patch_x = uv_dist(0) - patch_size/2;
            int patch_y = uv_dist(1) - patch_size/2;

            if (patch_x < 0) patch_x = 0;
            if (patch_y < 0) patch_y = 0;

            std::vector<std::shared_ptr<Feature>> feats_last_update;
            feats_last_update = trackFEATS->get_feature_database()->features_containing(state->_timestamp, false, false);

            // DANGER: this can pull up a slam landmark
            // if we have already re-identified this landmark, what do we do?
            
            std::vector<std::shared_ptr<Feature>> feats_to_compare;
            for (size_t j = 0; j < feats_last_update.size(); j++){
                Eigen::Vector2f pt_e = feats_last_update[j]->uvs.at(feats_last_update[j]->anchor_cam_id == -1 ? feats_last_update[j]->first_id : feats_last_update[j]->anchor_cam_id).back();

                if (pt_e[0] < patch_x) continue;
                if (pt_e[1] < patch_y) continue;
                if (pt_e[0] > patch_x + patch_size) continue;
                if (pt_e[1] > patch_y + patch_size) continue;

                feats_to_compare.push_back(feats_last_update[j]);
            }

            if (feats_to_compare.empty()){
                it0++;
                continue;
            }


            int best_dist = std::numeric_limits<int>::max();
            int best_index = -1;
            int best_id;
            for (size_t j = 0; j < feats_to_compare.size(); j++){
                int dist = DescriptorDistance(feats_to_compare[j]->descriptor, (*it0)->descriptor);
                if(dist < best_dist) {
                    best_dist = dist;
                    best_index = j;
                    best_id = feats_to_compare[j]->featid;
                }
            }

            // now, we need to REINSERT THE LOST SLAM FEAT INTO OUR SLAM FEATS AGAIN WITH THE NEW ID

            (*it0)->_featid = best_id;
            (*it0)->should_marg = false;
            // need a pair, with first being featid??_ and second the landmark
            state->_features_SLAM.insert({best_id, (*it0)});

            new_feats.push_back(feats_to_compare[best_index]);

            it0 = state->_features_SLAM_lost.erase(it0);

            // need an iterator here to efficiently delete from lost features vec now

        }
    }

    if (!new_feats.empty()) fprintf(stderr, "adding %lu re-found slam feats, %lu still lost\n", new_feats.size(), state->_features_SLAM_lost.size());

    return 0;
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int VioManager::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

std::vector<Eigen::Vector3d> VioManager::get_features_ARUCO() {
    std::vector<Eigen::Vector3d> aruco_feats;
    for (auto &f : state->_features_SLAM) {
        if ((int)f.first > 4 * state->_options.max_aruco_features)
            continue;
        if (ov_type::LandmarkRepresentation::is_relative_representation(f.second->_feat_representation)) {
            // Assert that we have an anchor pose for this feature
            assert(f.second->_anchor_cam_id != -1);
            assert(f.second->_anchor_clone_timestamp != -1);

            // Get calibration for our anchor camera
            Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->Rot();
            Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->pos();
            // Anchor pose orientation and position
            Eigen::Matrix<double, 3, 3> R_GtoI = state->_clones_IMU.at(f.second->_anchor_clone_timestamp)->Rot();
            Eigen::Matrix<double, 3, 1> p_IinG = state->_clones_IMU.at(f.second->_anchor_clone_timestamp)->pos();
            // Feature in the global frame
            aruco_feats.push_back(R_GtoI.transpose() * R_ItoC.transpose() * (f.second->get_xyz(false) - p_IinC) + p_IinG);
        } else {
            aruco_feats.push_back(f.second->get_xyz(false));
        }
    }
    return aruco_feats;
}
