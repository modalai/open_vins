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

#include "TrackCVP.h"

#include <opencv2/features2d.hpp>

#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"

using namespace ov_core;

void TrackCVP::feed_new_camera(const CameraData &message) {
    // Error check that we have all the data
    if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
        PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
        PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
        PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
        PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
        std::exit(EXIT_FAILURE);
    }

    // Either call our stereo or monocular version
    // If we are doing binocular tracking, then we should parallize our tracking
    size_t num_images = message.images.size();
    if (num_images == 1) {
        feed_KLT_monocular(message, 0);
        // feed_orb_monocular(message, 0);
    } else if (num_images == 2 && use_stereo) {
        // feed_stereo(message, 0, 1);
        fprintf(stderr, "STEREO IS NOT SUPPORTED YET\n");
    } else if (!use_stereo) {
        parallel_for_(cv::Range(0, (int)num_images), LambdaBody([&](const cv::Range &range) {
                          for (int i = range.start; i < range.end; i++) {
                              feed_KLT_monocular(message, i);
                              // feed_orb_monocular(message, i);
                          }
                      }));
    } else {
        PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, we only support mono or stereo tracking", num_images);
        std::exit(EXIT_FAILURE);
    }
}

void TrackCVP::feed_orb_monocular(const CameraData &message, size_t msg_id) {

    // Start timing
    rT1 = boost::posix_time::microsec_clock::local_time();

    // Lock this data feed for this camera
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Histogram equalize
    cv::Mat img, mask;
    // if (histogram_method == HistogramMethod::HISTOGRAM) {
    // cv::equalizeHist(message.images.at(msg_id), img);
    // } else if (histogram_method == HistogramMethod::CLAHE) {
    // double eq_clip_limit = 10.0;
    // cv::Size eq_win_size = cv::Size(8, 8);
    // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    // clahe->apply(message.images.at(msg_id), img);
    // } else {
    cv::Mat _img = message.images.at(msg_id);
    // now do a lpf over the image
    // cv::gaussianBlur(_img, img, 11);
    cv::GaussianBlur(_img, img, cv::Size(7, 7), 0, 0, cv::BORDER_DEFAULT);

    // }
    mask = message.masks.at(msg_id);

    // If we are the first frame (or have lost tracking), initialize our descriptors
    if (pts_last.find(cam_id) == pts_last.end() || pts_last[cam_id].empty()) {
        std::vector<cv::KeyPoint> good_left;
        std::vector<size_t> good_ids_left;
        mcv_dcm_pos_t positions[MCV_MAX_POS_BUF_SIZE];
        // this needs our keypoints
        perform_orb_detection_monocular(img, mask, good_ids_left, good_left, true, cam_id, positions);
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_last[cam_id] = img;
        img_mask_last[cam_id] = mask;
        pts_last[cam_id] = good_left;
        ids_last[cam_id] = good_ids_left;
        prev_positions[cam_id] = positions;
        return;
    }

    // Our new points and ids
    mcv_dcm_pos_t positions[MCV_MAX_POS_BUF_SIZE];
    std::vector<size_t> ids_new;
    std::vector<cv::KeyPoint> pts_new;

    auto rT0 = boost::posix_time::microsec_clock::local_time();

    // First, extract new points for this image
    perform_orb_detection_monocular(img, mask, ids_new, pts_new, false, cam_id, positions);
    rT2 = boost::posix_time::microsec_clock::local_time();

    // Our matches temporally
    std::vector<cv::DMatch> matches_ll;

    // Lets match temporally
    robust_orb_match(img, positions, pts_new.size(), img_last[cam_id], prev_positions[cam_id], pts_last[cam_id].size(), cam_id, cam_id,
                     matches_ll);
    rT3 = boost::posix_time::microsec_clock::local_time();

    // Get our "good tracks"
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;

    // Count how many we have tracked from the last time
    int num_tracklast = 0;
    int num_non_matched = 0;

    // Loop through all current left to right points
    // We want to see if any of theses have matches to the previous frame
    // If we have a match new->old then we want to use that ID instead of the new one
    for (size_t i = 0; i < pts_new.size(); i++) {

        // Loop through all left matches, and find the old "train" id
        int idll = -1;
        for (size_t j = 0; j < matches_ll.size(); j++) {
            if (matches_ll[j].trainIdx == (int)i) {
                idll = matches_ll[j].queryIdx;
            }
        }

        // Then lets replace the current ID with the old ID if found
        // Else just append the current feature and its unique ID
        good_left.push_back(pts_new[i]);
        if (idll != -1) {
            good_ids_left.push_back(ids_last[cam_id][idll]);
            num_tracklast++;
        } else {
            // if (num_non_matched + num_tracklast < num_features){
            num_non_matched++;
            good_ids_left.push_back(ids_new[i]);
            // }
        }
    }
    rT4 = boost::posix_time::microsec_clock::local_time();

    // Update our feature database, with theses new observations
    for (size_t i = 0; i < good_left.size(); i++) {
        cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
        database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                                 npt_l.y);
    }

    // Debug info
    PRINT_DEBUG("LtoL = %d | good = %d | fromlast = %d\n", (int)matches_ll.size(), (int)good_left.size(), num_tracklast);

    // Move forward in time
    {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_last[cam_id] = img;
        img_mask_last[cam_id] = mask;
        pts_last[cam_id] = good_left;
        ids_last[cam_id] = good_ids_left;
        prev_positions[cam_id] = positions;
    }
    rT5 = boost::posix_time::microsec_clock::local_time();

    // Our timing information
    PRINT_DEBUG("[TIME-DESC]: %.4f seconds for preprocessing\n", (rT0 - rT1).total_microseconds() * 1e-6);
    PRINT_DEBUG("[TIME-DESC]: %.4f seconds for detection\n", (rT2 - rT0).total_microseconds() * 1e-6);
    PRINT_DEBUG("[TIME-DESC]: %.4f seconds for matching\n", (rT3 - rT2).total_microseconds() * 1e-6);
    PRINT_DEBUG("[TIME-DESC]: %.4f seconds for merging\n", (rT4 - rT3).total_microseconds() * 1e-6);
    PRINT_DEBUG("[TIME-DESC]: %.4f seconds for feature DB update (%d features)\n", (rT5 - rT4).total_microseconds() * 1e-6,
                (int)good_left.size());
    PRINT_DEBUG("[TIME-DESC]: %.4f seconds for total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}

void TrackCVP::perform_orb_detection_monocular(const cv::Mat &img0, const cv::Mat &mask0, std::vector<size_t> &ids0,
                                               std::vector<cv::KeyPoint> &pts0, bool reference_set_empty, size_t cam_id,
                                               mcv_dcm_pos_t *positions) {

    // Assert that we need features
    assert(pts0.empty());

    // Extract our features (using cvpFPX)
    int n_points;
    uint32_t max_score;
    int ret = mcv_fpx_process(img0.data, mcv_features[cam_id], &max_score, &n_points);
    // fprintf(stderr, "EXTRACTED %d POINTS\n", n_points);
    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less then grid_px_size points away then existing features
    cv::Size size((int)((float)img0.cols / (float)min_px_dist), (int)((float)img0.rows / (float)min_px_dist));
    cv::Mat grid_2d = cv::Mat::zeros(size, CV_8UC1);

    // For all good matches, lets append to our returned vectors
    // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
    // NOTE: this is due to the fact that we select update features based on feat id
    // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
    // NOTE: not sure how to remove... maybe a better way?
    int actual_index = 0;
    for (int i = 0; i < n_points; i++) {
        // Get current point, check that it is in bounds
        int x = mcv_features[cam_id][i].x;
        int y = mcv_features[cam_id][i].y;
        int score = mcv_features[cam_id][i].score;

        // fprintf(stderr, "feat score: %d\n", score);
        // if (score < 1000) continue;

        int x_grid = (int)((float)x / (float)min_px_dist);
        int y_grid = (int)((float)y / (float)min_px_dist);
        if (x_grid < 0 || x_grid >= size.width || y_grid < 0 || y_grid >= size.height || x < 0 || x >= img0.cols || y < 0 ||
            y >= img0.rows) {
            continue;
        }
        // TODO check if this is within a masked region!!!!s
        // TURI

        // Check if this keypoint is near another point
        if (grid_2d.at<uint8_t>(y_grid, x_grid) > 127)
            continue;
        // Else we are good, append our keypoints
        positions[actual_index].x = mcv_features[cam_id][i].x;
        positions[actual_index].y = mcv_features[cam_id][i].y;
        actual_index++;

        // also append to our keypoint vector
        pts0.push_back(cv::KeyPoint(mcv_features[cam_id][i].x, mcv_features[cam_id][i].y, 1.));
        // Set our IDs to be unique IDs here, will later replace with corrected ones, after temporal matching
        size_t temp = ++currid;
        ids0.push_back(temp);
        // grid_2d.at<uint8_t>(y_grid, x_grid) = 255;
    }

    // fprintf(stderr, "KEEPING %d EXTRACTED FEATS OUT OF %d\n", actual_index, n_points);

    // if our reference set is empty, that means we need to calculate the descriptors for this batch of features now
    // then we have something to match against next round
    if (reference_set_empty) {
        if (mcv_dcm_calc(img0.data, positions, actual_index, cam_id)) {
            return;
        }
    }
}

void TrackCVP::robust_orb_match(const cv::Mat &img1, mcv_dcm_pos_t *positions1, size_t n_pos1, const cv::Mat &img0,
                                mcv_dcm_pos_t *positions0, size_t n_pos0, size_t id0, size_t id1, std::vector<cv::DMatch> &matches) {

    // Our 1to2 and 2to1 match vectors
    mcv_dcm_match_t matches_0_1[MCV_MAX_MATCH_BUF_SIZE];
    mcv_dcm_match_t matches_1_0[MCV_MAX_MATCH_BUF_SIZE];
    int n_matches_0, n_matches_1;

    // last arg is true to update reference descriptor set
    if (mcv_dcm_match(img1.data, positions1, n_pos1, matches_0_1, &n_matches_0, id0, true)) {
        fprintf(stderr, "EARLY RET\n");
        return;
    }

    std::vector<cv::Point2f> pts0_rsc, pts1_rsc;
    std::vector<cv::DMatch> matches_good;
    std::vector<cv::DMatch> matches_1_to_2, matches_2_to_1;

    for (int i = 0; i < n_matches_0; i++) {
        uint16_t idx = matches_0_1[i].index;
        uint16_t score = matches_0_1[i].score;
        if (score < 20) {
            matches_good.push_back(cv::DMatch(idx, i, 1));
            pts0_rsc.push_back(cv::Point2f(positions0[idx].x, positions0[idx].y));
            pts1_rsc.push_back(cv::Point2f(positions1[i].x, positions1[i].y));
        }
    }

    // If we don't have enough points for ransac just return empty
    if (pts0_rsc.size() < 10)
        return;

    // Normalize these points, so we can then do ransac
    // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
    std::vector<cv::Point2f> pts0_n, pts1_n;
    for (size_t i = 0; i < pts0_rsc.size(); i++) {
        pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0_rsc.at(i)));
        pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1_rsc.at(i)));
    }

    // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
    std::vector<uchar> mask_rsc;
    double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
    double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
    double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
    cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 0.9, 0.999, mask_rsc);

    // Loop through all good matches, and only append ones that have passed RANSAC
    for (size_t i = 0; i < matches_good.size(); i++) {
        // Skip if bad ransac id
        if (mask_rsc[i] != 1)
            continue;
        // Else, lets append this match to the return array!
        matches.push_back(matches_good.at(i));
    }

    // fprintf(stderr, "MATCHED %d FEATS, REJECTED %d via RANSAC, REJECTED %d via SCORE\n", matches.size(),  matches_good.size() -
    // matches.size(), n_matches_0 -  matches_good.size());
}

void TrackCVP::feed_KLT_monocular(const CameraData &message, size_t msg_id) {

    // Get our image objects for this image
    // Start timing
    rT1 = boost::posix_time::microsec_clock::local_time();

    // Lock this data feed for this camera
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    cv::Mat img, mask;
    img = message.images.at(msg_id);
    mask = message.masks.at(msg_id);

    // Extract the new image pyramid
    std::vector<cv::Mat> imgpyr;
    cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);
    rT2 = boost::posix_time::microsec_clock::local_time();

    // If we didn't have any successful tracks last time, just extract this time
    // This also handles, the tracking initalization on the first call to this extractor
    if (pts_last[cam_id].empty()) {
        // Detect new features
        std::vector<cv::KeyPoint> good_left;
        std::vector<size_t> good_ids_left;
        perform_KLT_detection_monocular(imgpyr, mask, good_left, good_ids_left, msg_id);
        // Save the current image and pyramid
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_last[cam_id] = img;
        img_pyramid_last[cam_id] = imgpyr;
        img_mask_last[cam_id] = mask;
        pts_last[cam_id] = good_left;
        ids_last[cam_id] = good_ids_left;
        return;
    }

    // First we should make that the last images have enough features so we can do KLT
    // This will "top-off" our number of tracks so always have a constant number
    auto pts_left_old = pts_last[cam_id];
    auto ids_left_old = ids_last[cam_id];
    perform_KLT_detection_monocular(img_pyramid_last[cam_id], img_mask_last[cam_id], pts_left_old, ids_left_old, msg_id);
    rT3 = boost::posix_time::microsec_clock::local_time();

    // Our return success masks, and predicted new features
    std::vector<uchar> mask_ll;
    std::vector<cv::KeyPoint> pts_left_new = pts_left_old;

    // Lets track temporally
    perform_KLT_matching(img_pyramid_last[cam_id], imgpyr, pts_left_old, pts_left_new, cam_id, cam_id, mask_ll);
    assert(pts_left_new.size() == ids_left_old.size());
    rT4 = boost::posix_time::microsec_clock::local_time();

    // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
    if (mask_ll.empty()) {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_last[cam_id] = img;
        img_pyramid_last[cam_id] = imgpyr;
        img_mask_last[cam_id] = mask;
        pts_last[cam_id].clear();
        ids_last[cam_id].clear();
        PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
        return;
    }

    // Get our "good tracks"
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;

    // Loop through all left points
    for (size_t i = 0; i < pts_left_new.size(); i++) {
        // fprintf(stderr, "mask: %u\n", mask_ll[i]);
        // Ensure we do not have any bad KLT tracks (i.e., points are negative)
        if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= img.cols ||
            (int)pts_left_new.at(i).pt.y >= img.rows)
            continue;
        // Check if it is in the mask
        // NOTE: mask has max value of 255 (white) if it should be
        if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
            continue;
        // If it is a good track, and also tracked from left to right
        if (mask_ll[i]) {
            good_left.push_back(pts_left_new[i]);
            good_ids_left.push_back(ids_left_old[i]);
        }
    }
    // fprintf(stderr, "matched %d new points, keeping %d\n", (int)mask_ll.size(), (int)good_left.size());

    // Update our feature database, with theses new observations
    for (size_t i = 0; i < good_left.size(); i++) {
        cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
        // fprintf(stderr, "inserting into feature db at time %6.5f\n", message.timestamp);
        database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                                 npt_l.y);
    }

    // Move forward in time
    {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_last[cam_id] = img;
        img_pyramid_last[cam_id] = imgpyr;
        img_mask_last[cam_id] = mask;
        pts_last[cam_id] = good_left;
        ids_last[cam_id] = good_ids_left;
    }
    rT5 = boost::posix_time::microsec_clock::local_time();

    // Timing information
    PRINT_DEBUG("[TIME-KLT]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
    PRINT_DEBUG("[TIME-KLT]: %.4f seconds for detection\n", (rT3 - rT2).total_microseconds() * 1e-6);
    PRINT_DEBUG("[TIME-KLT]: %.4f seconds for temporal klt\n", (rT4 - rT3).total_microseconds() * 1e-6);
    PRINT_DEBUG("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", (rT5 - rT4).total_microseconds() * 1e-6,
                (int)good_left.size());
    PRINT_DEBUG("[TIME-KLT]: %.4f seconds for total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}

static bool compare_response(mcv_fpx_feature_t first, mcv_fpx_feature_t second) { return first.score > second.score; }

// instead, just use the zone mode and only add in the best n_features per grid, starting at like 2
// maybe can just configure the zone mode to use the same griding system and do it for us.
// test how zone mode outputs features, figure out a nice way to split them up
// determine which features are good using that?
void TrackCVP::perform_KLT_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                               std::vector<size_t> &ids0, size_t msg_id) {

    auto start_grid_t = boost::posix_time::microsec_clock::local_time();
    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less than grid_px_size points away then existing features
    cv::Size size_close((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                        (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
    cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
    float size_x = (float)img0pyr.at(0).cols / (float)grid_x;
    float size_y = (float)img0pyr.at(0).rows / (float)grid_y;
    cv::Size size_grid(grid_x, grid_y); // width x height
    cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
    cv::Mat mask0_updated = mask0.clone();
    auto it0 = pts0.begin();
    auto it1 = ids0.begin();
    while (it0 != pts0.end()) {
        // Get current left keypoint, check that it is in bounds
        cv::KeyPoint kpt = *it0;
        int x = (int)kpt.pt.x;
        int y = (int)kpt.pt.y;
        int edge = 10;
        if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate mask coordinates for close points
        int x_close = (int)(kpt.pt.x / (float)min_px_dist);
        int y_close = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate what grid cell this feature is in
        int x_grid = std::floor(kpt.pt.x / size_x);
        int y_grid = std::floor(kpt.pt.y / size_y);
        if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Check if this keypoint is near another point
        if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Now check if it is in a mask area or not
        // NOTE: mask has max value of 255 (white) if it should be
        if (mask0.at<uint8_t>(y, x) > 127) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Else we are good, move forward to the next point
        grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
        if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) {
            grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
        }
        // Append this to the local mask of the image
        if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
            cv::Point pt1(x - min_px_dist, y - min_px_dist);
            cv::Point pt2(x + min_px_dist, y + min_px_dist);
            cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255));
        }
        it0++;
        it1++;
    }

    // First compute how many more features we need to extract from this image
    // If we don't need any features, just return
    int num_featsneeded = num_features - (int)pts0.size();
    // fprintf(stderr, "previous points: %d, need %d\n", (int)pts0.size(), num_featsneeded);
    if (num_featsneeded < 20)
        return;

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask0_grid;
    cv::resize(mask0, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    // this is just to know which areas actually matter, since cvp extracts everywhere regardless
    int num_features_grid = (int)((double)num_featsneeded / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(num_features_grid));

    auto end_grid_t = boost::posix_time::microsec_clock::local_time();

    int n_points;
    uint32_t max_score;

    int ret = mcv_fpx_process(img0pyr[0].data, mcv_features[msg_id], &max_score, &n_points);
    if (ret) {
        fprintf(stderr, "[ERROR-FPX] EXTRACTING FEATURES FAILED\n");
    }

    auto start_refine_t = boost::posix_time::microsec_clock::local_time();

    // basically, sort our returned features, do the same grid checks, and only add the amount of features needed
    std::vector<std::vector<mcv_fpx_feature_t>> feat_vec(grid_y, std::vector<mcv_fpx_feature_t>(grid_x));

    int max_allowed_y = img0pyr[0].rows / grid_y;
    int prev_index = -1;
    int start_index = 0;
    int vec_index = 0;
    int grid_points = 0;
    for (int i = 0; i < n_points; i++) {
        // if y is within the grid cell range, we're good
        if (mcv_features[msg_id][i].y <= max_allowed_y && mcv_features[msg_id][i].y >= max_allowed_y - img0pyr[0].rows / grid_y) {
            prev_index = i;
            grid_points++;
            continue;
        } else {
            if (prev_index >= 0) {
                feat_vec[vec_index].resize(grid_points);
                feat_vec[vec_index].assign(mcv_features[msg_id] + start_index, mcv_features[msg_id] + prev_index);
                std::sort(feat_vec[vec_index].begin(), feat_vec[vec_index].end(), compare_response);
                vec_index++;
            }
            grid_points = 1;
            max_allowed_y += img0pyr[0].rows / grid_y;
            prev_index = i;
            start_index = i;
            if (vec_index > grid_y - 1) {
                // fprintf(stderr, "SKIPPED %d POINTS\n", n_points - i);
                break;
            }
        }
    }

    std::vector<cv::Point2f> good_feats;

    // fprintf(stderr, "need %d features per grid\n", num_features_grid);
    // now, our feat_vec should be gridded up just like everything else
    // so can basically check the first point in each grid.
    // if valid, pick the best n from that grid and add, otherwise skip
    for (int i = 0; i < feat_vec.size(); i++) {
        for (int j = 0; j < feat_vec[i].size(); j++) {
            if ((int)grid_2d_grid.at<uint8_t>(j, i) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(j, i) != 255) {
                // now add the best num_features_grid points I guess
                for (size_t k = 0; k < (size_t)num_features_grid && i < feat_vec[i].size(); k++) {
                    // fprintf(stderr, "adding feature %d with condition of %d\n", k, num_features_grid);
                    good_feats.push_back(cv::Point2f(feat_vec[i][k].x, feat_vec[i][k].y));
                }
                break;
            } else {
                // fprintf(stderr, "SKIPPING GRID SECTION\n");
                continue;
            }
        }
    }

    auto end_refine_t = boost::posix_time::microsec_clock::local_time();

    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;

    // Now, reject features that are close a current feature
    for (int i = 0; i < good_feats.size(); i++) {
        int x_grid = (int)(good_feats[i].x / (float)min_px_dist);
        int y_grid = (int)(good_feats[i].y / (float)min_px_dist);

        if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height) {
            continue;
        }

        // See if there is a point at this location
        if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127) {
            continue;
        }
        // lets add it!
        kpts0_new.push_back(
            cv::KeyPoint(good_feats[i].x, good_feats[i].y, 1)); // need to fake a keypoint here, going to construct with mcv feature data
        pts0_new.push_back(cv::Point2f(good_feats[i].x, good_feats[i].y)); // cv point2f
    }

    // Sub-pixel refinement parameters
    cv::Size win_size = cv::Size(5, 5);
    cv::Size zero_zone = cv::Size(-1, -1);
    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.001);

    // Finally get sub-pixel for all extracted features
    cv::cornerSubPix(img0pyr.at(0), pts0_new, win_size, zero_zone, term_crit);

    // Loop through and record only ones that are valid
    // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
    // NOTE: this is due to the fact that we select update features based on feat id
    // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
    // NOTE: not sure how to remove... maybe a better way?
    for (size_t i = 0; i < pts0_new.size(); i++) {
        // update the uv coordinates
        kpts0_new.at(i).pt = pts0_new.at(i);
        // append the new uv coordinate
        pts0.push_back(kpts0_new.at(i));
        // move id foward and append this new point
        size_t temp = ++currid;
        ids0.push_back(temp);
    }

    // fprintf(stderr, "adding %d new features\n", (int)pts0_new.size());

    PRINT_DEBUG("[TIME-KLT]: %.4f seconds for grid\n", (end_grid_t - start_grid_t).total_microseconds() * 1e-6);
    PRINT_DEBUG("[TIME-KLT]: %.4f seconds for refine\n", (end_refine_t - start_refine_t).total_microseconds() * 1e-6);
}

void TrackCVP::perform_KLT_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr,
                                    std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1,
                                    std::vector<uchar> &mask_out) {
    // We must have equal vectors
    assert(kpts0.size() == kpts1.size());

    // Return if we don't have any points
    if (kpts0.empty() || kpts1.empty())
        return;

    // Convert keypoints into points (stupid opencv stuff)
    std::vector<cv::Point2f> pts0, pts1;
    for (size_t i = 0; i < kpts0.size(); i++) {
        pts0.push_back(kpts0.at(i).pt);
        pts1.push_back(kpts1.at(i).pt);
    }

    // If we don't have enough points for ransac just return empty
    // We set the mask to be all zeros since all points failed RANSAC
    if (pts0.size() < 10) {
        for (size_t i = 0; i < pts0.size(); i++)
            mask_out.push_back((uchar)0);
        return;
    }

    auto opt_time = boost::posix_time::microsec_clock::local_time();

    // Now do KLT tracking to get the valid new points
    std::vector<uchar> mask_klt;
    std::vector<float> error;
    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
    auto opt_finish = boost::posix_time::microsec_clock::local_time();

    // if this takes too long, we have a problem
    // gonna give it 20 ms to complete, otherwise we terminating
    // if we terminate when iot actually should solve, this is bad
    // if ((opt_finish - opt_time).total_microseconds() * 1e-6 >= 0.03) {
    //   fprintf(stderr, "RETURNING OPT, %6.5f\n", (opt_finish - opt_time).total_microseconds() * 1e-6);
    //   for (size_t i = 0; i < pts0.size(); i++)
    //     mask_out.push_back((uchar)0);
    //   return;
    // }

    // Normalize these points, so we can then do ransac
    // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
    std::vector<cv::Point2f> pts0_n, pts1_n;
    for (size_t i = 0; i < pts0.size(); i++) {
        pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0.at(i)));
        pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1.at(i)));
    }
    auto ran_time = boost::posix_time::microsec_clock::local_time();

    // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
    std::vector<uchar> mask_rsc;
    double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
    double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
    double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
    cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc);
    auto ran_finish = boost::posix_time::microsec_clock::local_time();

    // Loop through and record only ones that are valid
    for (size_t i = 0; i < mask_klt.size(); i++) {
        auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
        mask_out.push_back(mask);
    }

    // Copy back the updated positions
    for (size_t i = 0; i < pts0.size(); i++) {
        kpts0.at(i).pt = pts0.at(i);
        kpts1.at(i).pt = pts1.at(i);
    }

    PRINT_DEBUG("[TIME-MATCH]: %.4f seconds for optical flow with %d points\n", (opt_finish - opt_time).total_microseconds() * 1e-6,
                pts0.size());
    PRINT_DEBUG("[TIME-MATCH]: %.4f seconds for RANSAC\n", (ran_finish - ran_time).total_microseconds() * 1e-6);
}

// void TrackCVP::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {

//     // Start timing
//     rT1 = boost::posix_time::microsec_clock::local_time();

//     // Lock this data feed for this camera
//     size_t cam_id_left = message.sensor_ids.at(msg_id_left);
//     size_t cam_id_right = message.sensor_ids.at(msg_id_right);
//     std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
//     std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

//     // Histogram equalize images
//     cv::Mat img_left, img_right, mask_left, mask_right;
//     if (histogram_method == HistogramMethod::HISTOGRAM) {
//         cv::equalizeHist(message.images.at(msg_id_left), img_left);
//         cv::equalizeHist(message.images.at(msg_id_right), img_right);
//     } else if (histogram_method == HistogramMethod::CLAHE) {
//         double eq_clip_limit = 10.0;
//         cv::Size eq_win_size = cv::Size(8, 8);
//         cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
//         clahe->apply(message.images.at(msg_id_left), img_left);
//         clahe->apply(message.images.at(msg_id_right), img_right);
//     } else {
//         img_left = message.images.at(msg_id_left);
//         img_right = message.images.at(msg_id_right);
//     }
//     mask_left = message.masks.at(msg_id_left);
//     mask_right = message.masks.at(msg_id_right);

//     // If we are the first frame (or have lost tracking), initialize our descriptors
//     if (pts_last[cam_id_left].empty() || pts_last[cam_id_right].empty()) {
//         std::vector<cv::KeyPoint> good_left, good_right;
//         std::vector<size_t> good_ids_left, good_ids_right;
//         cv::Mat good_desc_left, good_desc_right;
//         perform_detection_stereo(img_left, img_right, mask_left, mask_right, good_left, good_right, good_desc_left, good_desc_right,
//                                  cam_id_left, cam_id_right, good_ids_left, good_ids_right);
//         std::lock_guard<std::mutex> lckv(mtx_last_vars);
//         img_last[cam_id_left] = img_left;
//         img_last[cam_id_right] = img_right;
//         img_mask_last[cam_id_left] = mask_left;
//         img_mask_last[cam_id_right] = mask_right;
//         pts_last[cam_id_left] = good_left;
//         pts_last[cam_id_right] = good_right;
//         ids_last[cam_id_left] = good_ids_left;
//         ids_last[cam_id_right] = good_ids_right;
//         desc_last[cam_id_left] = good_desc_left;
//         desc_last[cam_id_right] = good_desc_right;
//         return;
//     }

//     // Our new keypoints and descriptor for the new image
//     std::vector<cv::KeyPoint> pts_left_new, pts_right_new;
//     cv::Mat desc_left_new, desc_right_new;
//     std::vector<size_t> ids_left_new, ids_right_new;

//     // First, extract new descriptors for this new image
//     perform_detection_stereo(img_left, img_right, mask_left, mask_right, pts_left_new, pts_right_new, desc_left_new, desc_right_new,
//                              cam_id_left, cam_id_right, ids_left_new, ids_right_new);
//     rT2 = boost::posix_time::microsec_clock::local_time();

//     // Our matches temporally
//     std::vector<cv::DMatch> matches_ll, matches_rr;
//     parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
//                       for (int i = range.start; i < range.end; i++) {
//                           bool is_left = (i == 0);
//                           robust_match(pts_last[is_left ? cam_id_left : cam_id_right], is_left ? pts_left_new : pts_right_new,
//                                        desc_last[is_left ? cam_id_left : cam_id_right], is_left ? desc_left_new : desc_right_new,
//                                        is_left ? cam_id_left : cam_id_right, is_left ? cam_id_left : cam_id_right,
//                                        is_left ? matches_ll : matches_rr);
//                       }
//                   }));
//     rT3 = boost::posix_time::microsec_clock::local_time();

//     // Get our "good tracks"
//     std::vector<cv::KeyPoint> good_left, good_right;
//     std::vector<size_t> good_ids_left, good_ids_right;
//     cv::Mat good_desc_left, good_desc_right;

//     // Points must be of equal size
//     assert(pts_last[cam_id_left].size() == pts_last[cam_id_right].size());
//     assert(pts_left_new.size() == pts_right_new.size());

//     // Count how many we have tracked from the last time
//     int num_tracklast = 0;

//     // Loop through all current left to right points
//     // We want to see if any of theses have matches to the previous frame
//     // If we have a match new->old then we want to use that ID instead of the new one
//     for (size_t i = 0; i < pts_left_new.size(); i++) {

//         // Loop through all left matches, and find the old "train" id
//         int idll = -1;
//         for (size_t j = 0; j < matches_ll.size(); j++) {
//             if (matches_ll[j].trainIdx == (int)i) {
//                 idll = matches_ll[j].queryIdx;
//             }
//         }

//         // Loop through all left matches, and find the old "train" id
//         int idrr = -1;
//         for (size_t j = 0; j < matches_rr.size(); j++) {
//             if (matches_rr[j].trainIdx == (int)i) {
//                 idrr = matches_rr[j].queryIdx;
//             }
//         }

//         // If we found a good stereo track from left to left, and right to right
//         // Then lets replace the current ID with the old ID
//         // We also check that we are linked to the same past ID value
//         if (idll != -1 && idrr != -1 && ids_last[cam_id_left][idll] == ids_last[cam_id_right][idrr]) {
//             good_left.push_back(pts_left_new[i]);
//             good_right.push_back(pts_right_new[i]);
//             good_desc_left.push_back(desc_left_new.row((int)i));
//             good_desc_right.push_back(desc_right_new.row((int)i));
//             good_ids_left.push_back(ids_last[cam_id_left][idll]);
//             good_ids_right.push_back(ids_last[cam_id_right][idrr]);
//             num_tracklast++;
//         } else {
//             // Else just append the current feature and its unique ID
//             good_left.push_back(pts_left_new[i]);
//             good_right.push_back(pts_right_new[i]);
//             good_desc_left.push_back(desc_left_new.row((int)i));
//             good_desc_right.push_back(desc_right_new.row((int)i));
//             good_ids_left.push_back(ids_left_new[i]);
//             good_ids_right.push_back(ids_left_new[i]);
//         }
//     }
//     rT4 = boost::posix_time::microsec_clock::local_time();

//     //===================================================================================
//     //===================================================================================

//     // Update our feature database, with theses new observations
//     for (size_t i = 0; i < good_left.size(); i++) {
//         // Assert that our IDs are the same
//         assert(good_ids_left.at(i) == good_ids_right.at(i));
//         // Try to undistort the point
//         cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
//         cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
//         // Append to the database
//         database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y,
//         npt_l.x,
//                                  npt_l.y);
//         database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y,
//                                  npt_r.x, npt_r.y);
//     }

//     // Debug info
//     // PRINT_DEBUG("LtoL = %d | RtoR = %d | LtoR = %d | good = %d | fromlast = %d\n", (int)matches_ll.size(),
//     //       (int)matches_rr.size(),(int)ids_left_new.size(),(int)good_left.size(),num_tracklast);

//     // Move forward in time
//     {
//         std::lock_guard<std::mutex> lckv(mtx_last_vars);
//         img_last[cam_id_left] = img_left;
//         img_last[cam_id_right] = img_right;
//         img_mask_last[cam_id_left] = mask_left;
//         img_mask_last[cam_id_right] = mask_right;
//         pts_last[cam_id_left] = good_left;
//         pts_last[cam_id_right] = good_right;
//         ids_last[cam_id_left] = good_ids_left;
//         ids_last[cam_id_right] = good_ids_right;
//         desc_last[cam_id_left] = good_desc_left;
//         desc_last[cam_id_right] = good_desc_right;
//     }
//     rT5 = boost::posix_time::microsec_clock::local_time();

//     // Our timing information
//     // PRINT_DEBUG("[TIME-DESC]: %.4f seconds for detection\n",(rT2-rT1).total_microseconds() * 1e-6);
//     // PRINT_DEBUG("[TIME-DESC]: %.4f seconds for matching\n",(rT3-rT2).total_microseconds() * 1e-6);
//     // PRINT_DEBUG("[TIME-DESC]: %.4f seconds for merging\n",(rT4-rT3).total_microseconds() * 1e-6);
//     // PRINT_DEBUG("[TIME-DESC]: %.4f seconds for feature DB update (%d features)\n",(rT5-rT4).total_microseconds() * 1e-6,
//     // (int)good_left.size()); PRINT_DEBUG("[TIME-DESC]: %.4f seconds for total\n",(rT5-rT1).total_microseconds() * 1e-6);
// }

// void TrackCVP::perform_detection_stereo(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &mask0, const cv::Mat &mask1,
//                                                std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1, cv::Mat &desc0,
//                                                cv::Mat &desc1, size_t cam_id0, size_t cam_id1, std::vector<size_t> &ids0,
//                                                std::vector<size_t> &ids1) {

//     // Assert that we need features
//     assert(pts0.empty());
//     assert(pts1.empty());

//     // Extract our features (use FAST with griding), and their descriptors
//     std::vector<cv::KeyPoint> pts0_ext, pts1_ext;
//     cv::Mat desc0_ext, desc1_ext;
//     parallel_for_(
//         cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
//             for (int i = range.start; i < range.end; i++) {
//                 bool is_left = (i == 0);
//                 Grider_FAST::perform_griding(is_left ? img0 : img1, is_left ? mask0 : mask1, is_left ? pts0_ext : pts1_ext, num_features,
//                                              grid_x, grid_y, threshold, true);
//                 (is_left ? orb0 : orb1)->compute(is_left ? img0 : img1, is_left ? pts0_ext : pts1_ext, is_left ? desc0_ext : desc1_ext);
//             }
//         }));

//     // Do matching from the left to the right image
//     std::vector<cv::DMatch> matches;
//     robust_match(pts0_ext, pts1_ext, desc0_ext, desc1_ext, cam_id0, cam_id1, matches);

//     // Create a 2D occupancy grid for this current image
//     // Note that we scale this down, so that each grid point is equal to a set of pixels
//     // This means that we will reject points that less then grid_px_size points away then existing features
//     cv::Size size0((int)((float)img0.cols / (float)min_px_dist), (int)((float)img0.rows / (float)min_px_dist));
//     cv::Mat grid_2d_0 = cv::Mat::zeros(size0, CV_8UC1);
//     cv::Size size1((int)((float)img1.cols / (float)min_px_dist), (int)((float)img1.rows / (float)min_px_dist));
//     cv::Mat grid_2d_1 = cv::Mat::zeros(size1, CV_8UC1);

//     // For all good matches, lets append to our returned vectors
//     for (size_t i = 0; i < matches.size(); i++) {

//         // Get our ids
//         int index_pt0 = matches.at(i).queryIdx;
//         int index_pt1 = matches.at(i).trainIdx;

//         // Get current left/right keypoint, check that it is in bounds
//         cv::KeyPoint kpt0 = pts0_ext.at(index_pt0);
//         cv::KeyPoint kpt1 = pts1_ext.at(index_pt1);
//         int x0 = (int)kpt0.pt.x;
//         int y0 = (int)kpt0.pt.y;
//         int x0_grid = (int)(kpt0.pt.x / (float)min_px_dist);
//         int y0_grid = (int)(kpt0.pt.y / (float)min_px_dist);
//         if (x0_grid < 0 || x0_grid >= size0.width || y0_grid < 0 || y0_grid >= size0.height || x0 < 0 || x0 >= img0.cols || y0 < 0 ||
//             y0 >= img0.rows) {
//             continue;
//         }
//         int x1 = (int)kpt1.pt.x;
//         int y1 = (int)kpt1.pt.y;
//         int x1_grid = (int)(kpt1.pt.x / (float)min_px_dist);
//         int y1_grid = (int)(kpt1.pt.y / (float)min_px_dist);
//         if (x1_grid < 0 || x1_grid >= size1.width || y1_grid < 0 || y1_grid >= size1.height || x1 < 0 || x1 >= img0.cols || y1 < 0 ||
//             y1 >= img0.rows) {
//             continue;
//         }

//         // Check if this keypoint is near another point
//         if (grid_2d_0.at<uint8_t>(y0_grid, x0_grid) > 127 || grid_2d_1.at<uint8_t>(y1_grid, x1_grid) > 127)
//             continue;

//         // Append our keypoints and descriptors
//         pts0.push_back(pts0_ext[index_pt0]);
//         pts1.push_back(pts1_ext[index_pt1]);
//         desc0.push_back(desc0_ext.row(index_pt0));
//         desc1.push_back(desc1_ext.row(index_pt1));

//         // Set our IDs to be unique IDs here, will later replace with corrected ones, after temporal matching
//         size_t temp = ++currid;
//         ids0.push_back(temp);
//         ids1.push_back(temp);
//     }
// }