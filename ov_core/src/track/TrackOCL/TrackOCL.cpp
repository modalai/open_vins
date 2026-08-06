/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
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

#include "TrackOCL.h"

#include <algorithm>
#include <unordered_set>
#include "../Grider_FAST.h"
#include "../Grider_GRID.h"
#include "Grider_OCL.h"
#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"
#include <modal_flow/ocl/StereoMatcherCL.hpp>
#include "utils/print.h"

using namespace ov_core;

int test_feed_all = 0;

// Compile-time toggle for the per-call [STEREO ZNCC] diagnostic prints in
// perform_detection_stereo (top-off accept summary, top-off accept=0 per-gate
// breakdown, mono-promote accept summary, mono-promote accept=0 breakdown).
// Off by default so production logs stay clean; flip to true when investigating
// accept-rate / lr_err / margin distributions. The entire stats-and-printf
// block is dead-stripped under `if constexpr` when false, so there is zero
// runtime cost in the off configuration.
static constexpr bool kEnableStereoZnccDiag = false;


void TrackOCL::enable_zncc_stereo_matcher(const modal_flow::StereoCalib &calib_in,
                                          float z_min, float z_max)
{
    auto stm = std::make_unique<modal_flow::ocl::StereoMatcherCL>(dev_);
    stm->set_lr_thresh(5.0f);
    mgr_.set_stereo_matcher(std::move(stm));

    modal_flow::StereoCalib c = calib_in;
    c.z_min = z_min;
    c.z_max = z_max;
    mgr_.set_stereo_calibration(c);

    stereo_cam_id_left_  = (size_t)c.left;
    stereo_cam_id_right_ = (size_t)c.right;
    printf("[TrackOCL] ZNCC-band stereo matcher enabled (src cam %zu -> dst cam %zu, z=[%.2f,%.1f]m)\n",
           stereo_cam_id_left_, stereo_cam_id_right_, (double)z_min, (double)z_max);
}

void TrackOCL::feed_new_camera(const CameraData &message)
{

    // Error check that we have all the data
    if (message.sensor_ids.empty() ||
        (message.sensor_ids.size() != message.images.size()) ||
        (message.images.size() != message.masks.size()))
    {
        PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
        PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
        PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
        PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
        std::exit(EXIT_FAILURE);
    }

    // Preprocessing steps that we do not parallelize
    // NOTE: DO NOT PARALLELIZE THESE!
    // NOTE: These seem to be much slower if you parallelize them...
    rT1 = prof_now();

    size_t num_images = message.images.size();

    // Stability guard: drop single-cam "MONO-fallback" messages when configured
    // for stereo. CameraQueueFusion emits them at startup before both cam frames
    // have landed; routing them to feed_monocular has triggered cv::resize() on
    // empty masks and OclPyramidRing ABA throws. Stereo VIO can't use single-cam
    // frames anyway. One-shot warning so we know if they fire at steady state.
    if (use_stereo && num_images < 2)
    {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr,
                    "[TrackOCL] dropping MONO-fallback CameraData (num_images=%zu) "
                    "because TrackOCL is in stereo mode -- this should only happen at "
                    "startup before both cams have landed a frame. Subsequent drops "
                    "are silent.\n", num_images);
            warned = true;
        }
        return;
    }

    for (size_t msg_id = 0; msg_id < num_images; msg_id++)
    {
        // Lock this data feed for this camera
        size_t cam_id = message.sensor_ids.at(msg_id);
        std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

        modal_flow::Frame frame = message.img_frames[msg_id];

        // upload image to flow manager
        if (img_buf_next_[cam_id]) {
            if (img_buf_prev_[cam_id]) {
                mgr_.release_pyramid((modal_flow::CameraId)cam_id, img_buf_prev_[cam_id]);
            }
            img_buf_prev_[cam_id] = img_buf_next_[cam_id];
        }
        img_buf_next_[cam_id] = mgr_.acquire_pyramid_buf((modal_flow::CameraId)cam_id);
        mgr_.upload_frame_to_buf(frame, img_buf_next_[cam_id]);
    }

    // Either call our stereo or monocular version
    // If we are doing binocular tracking, then we should parallize our tracking
    if (num_images == 1)
    {
        feed_monocular(message, 0);
    }
    else if (num_images == 2 && use_stereo)
    {
        feed_stereo(message, 0, 1);
    }
    else if (!use_stereo)
    {
        // NOTE: opencv::parallel_for() seems to be less efficient than direct loop
        for (int i = 0; i < (int)num_images; i++)
        {
            feed_monocular(message, i);
        }
    }
    else
    {
        PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, we only support mono or stereo tracking", num_images);
        std::exit(EXIT_FAILURE);
    }
}

static int64_t _apps_time_monotonic_ns()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts))
    {
        fprintf(stderr, "ERROR calling clock_gettime\n");
        return -1;
    }
    return (int64_t)ts.tv_sec * 1000000000 + (int64_t)ts.tv_nsec;
}

void TrackOCL::feed_monocular(const CameraData &message, size_t msg_id)
{
    // Lock this data feed for this camera
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Get our image objects for this image
    cv::Mat mask = message.masks.at(msg_id);
    
    std::pair<int, int> dims = mgr_.get_cam_dim(cam_id);
    int cam_width  = std::get<0>(dims);
    int cam_height = std::get<1>(dims);
    
    rT2 = prof_now();
    int64_t t2 = _apps_time_monotonic_ns();
    // If we didn't have any successful tracks last time, just extract this time
    // This also handles, the tracking initalization on the first call to this extractor
    if (pts_last[cam_id].empty())
    {
        // Detect new features
        std::vector<cv::KeyPoint> good_left;
        std::vector<size_t> good_ids_left;
        perform_detection_monocular(img_buf_next_[cam_id], mask, good_left, good_ids_left, cam_id);

        // Save the current image and pyramid
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_mask_last[cam_id] = mask;
        pts_last[cam_id] = good_left;
        ids_last[cam_id] = good_ids_left;

        return;
    }

    // First we should make that the last images have enough features so we can do KLT
    // This will "top-off" our number of tracks so always have a constant number
    auto pts_left_old = pts_last[cam_id];
    auto ids_left_old = ids_last[cam_id];

    perform_detection_monocular(img_buf_prev_[cam_id], img_mask_last[cam_id], pts_left_old, ids_left_old, cam_id);
    rT3 = prof_now();
    int64_t t3 = _apps_time_monotonic_ns();

    // Our return success masks, and predicted new features
    std::vector<uchar> mask_ll;
    std::vector<cv::KeyPoint> pts_left_new = pts_left_old;

    // Lets track temporally
    perform_matching(img_buf_prev_[cam_id], img_buf_next_[cam_id], pts_left_old, pts_left_new, cam_id, cam_id, mask_ll);
    assert(pts_left_new.size() == ids_left_old.size());
    int64_t t4 = _apps_time_monotonic_ns();
    rT4 = prof_now();

    // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
    if (mask_ll.empty())
    {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_mask_last[cam_id] = mask;
        pts_last[cam_id].clear();
        ids_last[cam_id].clear();
        PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
        return;
    }

    // Get our "good tracks"
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;
    // printf("cam_id: %d, pts tracked: %zu\n", cam_id, pts_left_new.size());

    // Loop through all left points
    for (size_t i = 0; i < pts_left_new.size(); i++)
    {
        // Ensure we do not have any bad KLT tracks (i.e., points are negative)
        if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= cam_width ||
            (int)pts_left_new.at(i).pt.y >= cam_height)
            continue;
        // Check if it is in the mask
        // NOTE: mask has max value of 255 (white) if it should be
        if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
            continue;
        // If it is a good track, and also tracked from left to right
        if (mask_ll[i])
        {
            good_left.push_back(pts_left_new[i]);
            good_ids_left.push_back(ids_left_old[i]);
        }
    }
    // printf("cam_id: %d, good tracks: %zu\n", cam_id, good_left.size());

    // Update our feature database, with theses new observations
    for (size_t i = 0; i < good_left.size(); i++)
    {
        cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
        database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                                 npt_l.y);
    }

    // Move forward in time
    {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_mask_last[cam_id] = mask;
        pts_last[cam_id] = good_left;
        ids_last[cam_id] = good_ids_left;
    }
    int64_t t5 = _apps_time_monotonic_ns();
    rT5 = prof_now();

    // Timing prints in milliseconds
    auto dt = [](int64_t a, int64_t b){ return double(b - a) / 1e6; };

    // printf("[TIME-KLT]: %.3f ms for pyramid\n", dt(t1, t2));
    // printf("[TIME-KLT]: %.3f ms for detection\n", dt(t2, t3));
    // printf("[TIME-KLT]: %.3f ms for temporal klt\n", dt(t3, t4));
    // printf("[TIME-KLT]: %.3f ms for feature DB update (%d features)\n",
    //        dt(t4, t5), (int)good_left.size());
    // printf("[TIME-KLT]: %.3f ms total\n", dt(t1, t5));
}

void TrackOCL::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right)
{
    // Lock this data feed for this camera
    size_t cam_id_left = message.sensor_ids.at(msg_id_left);
    size_t cam_id_right = message.sensor_ids.at(msg_id_right);
    std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
    std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

    // Get our mask objects for this image.
    // NOTE: the OCL path keeps images/pyramids on the GPU (img_buf_*_), so the CPU
    // img_curr / img_pyramid_curr maps are never populated here (unlike CPU TrackKLT).
    cv::Mat mask_left = message.masks.at(msg_id_left);
    cv::Mat mask_right = message.masks.at(msg_id_right);

    std::pair<int, int> dims_left = mgr_.get_cam_dim(cam_id_left);
    int cam_width_left   = std::get<0>(dims_left);
    int cam_height_left = std::get<1>(dims_left);

    std::pair<int, int> dims_right = mgr_.get_cam_dim(cam_id_right);
    int cam_width_right  = std::get<0>(dims_right);
    int cam_height_right = std::get<1>(dims_right);

    // NOTE: do NOT re-upload frames here. feed_new_camera() already swapped prev<-next
    // and uploaded the CURRENT frame for BOTH cameras before dispatching to feed_stereo().
    // Re-uploading collapses prev==next (both = current frame), so temporal KLT matches a
    // frame against itself -> 0.000 disparity (stuck in ZUPT). feed_monocular() relies on
    // the same feed_new_camera() upload; mirror that behavior here.
    
    rT2 = prof_now();
    int64_t t2 = _apps_time_monotonic_ns();

    if (pts_last[cam_id_left].empty() && pts_last[cam_id_right].empty()) {
        // Track into the new image
        std::vector<cv::KeyPoint> good_left, good_right;
        std::vector<size_t> good_ids_left, good_ids_right;
        perform_detection_stereo(img_buf_next_[cam_id_left], img_buf_next_[cam_id_right], mask_left, mask_right,
                                 cam_id_left, cam_id_right, good_left, good_right, good_ids_left, good_ids_right);
        // Save the current image and pyramid
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        // img_last / img_pyramid_last are unused by the OCL tracker (images stay on the GPU)
        img_mask_last[cam_id_left] = mask_left;
        img_mask_last[cam_id_right] = mask_right;
        pts_last[cam_id_left] = good_left;
        pts_last[cam_id_right] = good_right;
        ids_last[cam_id_left] = good_ids_left;
        ids_last[cam_id_right] = good_ids_right;
        return;
    }

    // First we should make that the last images have enough features so we can do KLT
    // This will "top-off" our number of tracks so always have a constant number
    int pts_before_detect = (int)pts_last[cam_id_left].size();
    auto pts_left_old = pts_last[cam_id_left];
    auto pts_right_old = pts_last[cam_id_right];
    auto ids_left_old = ids_last[cam_id_left];
    auto ids_right_old = ids_last[cam_id_right];
    perform_detection_stereo(img_buf_prev_[cam_id_left], img_buf_prev_[cam_id_right], 
                             img_mask_last[cam_id_left], img_mask_last[cam_id_right],
                             cam_id_left, cam_id_right, pts_left_old, pts_right_old, ids_left_old, ids_right_old);
    rT3 = prof_now();

    // Our return success masks, and predicted new features
    std::vector<uchar> mask_ll, mask_rr;
    std::vector<cv::KeyPoint> pts_left_new = pts_left_old;
    std::vector<cv::KeyPoint> pts_right_new = pts_right_old;

    perform_matching(img_buf_prev_[cam_id_left], img_buf_next_[cam_id_left], pts_left_old, pts_left_new, cam_id_left, cam_id_left, mask_ll);
    perform_matching(img_buf_prev_[cam_id_right], img_buf_next_[cam_id_right], pts_right_old, pts_right_new, cam_id_right, cam_id_right, mask_rr);

    rT4 = prof_now();

    // left to right matching
    // TODO: we should probably still do this to reject outliers
    // TODO: maybe we should collect all tracks that are in both frames and make they pass this?
    // std::vector<uchar> mask_lr;
    // perform_matching(imgpyr_left, imgpyr_right, pts_left_new, pts_right_new, cam_id_left, cam_id_right, mask_lr);
    rT5 = prof_now();

    // If any of our masks are empty, that means we didn't have enough to do ransac, so just return
    if (mask_ll.empty() && mask_rr.empty()) {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        // img_last / img_pyramid_last are unused by the OCL tracker (images stay on the GPU)
        img_mask_last[cam_id_left] = mask_left;
        img_mask_last[cam_id_right] = mask_right;
        pts_last[cam_id_left].clear();
        pts_last[cam_id_right].clear();
        ids_last[cam_id_left].clear();
        ids_last[cam_id_right].clear();
        PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
        return;
    }

    // Get our "good tracks"
    std::vector<cv::KeyPoint> good_left, good_right;
    std::vector<size_t> good_ids_left, good_ids_right;

    // Loop through all left points
    for (size_t i = 0; i < pts_left_new.size(); i++) {
        // Ensure we do not have any bad KLT tracks (i.e., points are negative)
        if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x > cam_width_left ||
            (int)pts_left_new.at(i).pt.y > cam_height_left)
        continue;
        // See if we have the same feature in the right
        bool found_right = false;
        size_t index_right = 0;
        for (size_t n = 0; n < ids_right_old.size(); n++) {
        if (ids_left_old.at(i) == ids_right_old.at(n)) {
            found_right = true;
            index_right = n;
            break;
        }
        }
        // If it is a good track, and also tracked from left to right
        // Else track it as a mono feature in just the left image
        if (mask_ll[i] && found_right && mask_rr[index_right]) {
        // Ensure we do not have any bad KLT tracks (i.e., points are negative)
        if (pts_right_new.at(index_right).pt.x < 0 || pts_right_new.at(index_right).pt.y < 0 ||
            (int)pts_right_new.at(index_right).pt.x >= cam_width_right || (int)pts_right_new.at(index_right).pt.y >= cam_height_right)
            continue;
        good_left.push_back(pts_left_new.at(i));
        good_right.push_back(pts_right_new.at(index_right));
        good_ids_left.push_back(ids_left_old.at(i));
        good_ids_right.push_back(ids_right_old.at(index_right));
        // PRINT_DEBUG("adding to stereo - %u , %u\n", ids_left_old.at(i), ids_right_old.at(index_right));
        } else if (mask_ll[i]) {
        good_left.push_back(pts_left_new.at(i));
        good_ids_left.push_back(ids_left_old.at(i));
        // PRINT_DEBUG("adding to left - %u \n",ids_left_old.at(i));
        }
    }

    // Loop through all right points
    for (size_t i = 0; i < pts_right_new.size(); i++) {
        // Ensure we do not have any bad KLT tracks (i.e., points are negative)
        if (pts_right_new.at(i).pt.x < 0 || pts_right_new.at(i).pt.y < 0 || (int)pts_right_new.at(i).pt.x >= cam_width_right ||
            (int)pts_right_new.at(i).pt.y >= cam_height_right)
        continue;
        // See if we have the same feature in the right
        bool added_already = (std::find(good_ids_right.begin(), good_ids_right.end(), ids_right_old.at(i)) != good_ids_right.end());
        // If it has not already been added as a good feature, add it as a mono track
        if (mask_rr[i] && !added_already) {
        good_right.push_back(pts_right_new.at(i));
        good_ids_right.push_back(ids_right_old.at(i));
        // PRINT_DEBUG("adding to right - %u \n", ids_right_old.at(i));
        }
    }

    // Update our feature database, with theses new observations
    for (size_t i = 0; i < good_left.size(); i++) {
        cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
        database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                                npt_l.y);
    }
    for (size_t i = 0; i < good_right.size(); i++) {
        cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
        database->update_feature(good_ids_right.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                                npt_r.y);
    }

    // Move forward in time
    {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        // img_last / img_pyramid_last are unused by the OCL tracker (images stay on the GPU)
        img_mask_last[cam_id_left] = mask_left;
        img_mask_last[cam_id_right] = mask_right;
        pts_last[cam_id_left] = good_left;
        pts_last[cam_id_right] = good_right;
        ids_last[cam_id_left] = good_ids_left;
        ids_last[cam_id_right] = good_ids_right;
    }
    rT6 = prof_now();

    //  // Timing information
    PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", prof_s(rT1, rT2));
    PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%d detected)\n", prof_s(rT2, rT3),
                (int)pts_last[cam_id_left].size() - pts_before_detect);
    PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", prof_s(rT3, rT4));
    PRINT_ALL("[TIME-KLT]: %.4f seconds for stereo klt\n", prof_s(rT4, rT5));
    PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", prof_s(rT5, rT6),
                (int)good_left.size());
    PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", prof_s(rT1, rT6));
}

void TrackOCL::perform_detection_monocular(modal_flow::BufferId& buf_id, const cv::Mat &mask0,
                                           std::vector<cv::KeyPoint> &pts0,
                                           std::vector<size_t> &ids0, int cam_id)
{

    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less than grid_px_size points away then existing features

    int64_t rT1 = _apps_time_monotonic_ns();
    
    int img_width  = mask0.cols;
    int img_height = mask0.rows;

    cv::Size size_close((int)((float)img_width  / (float)min_px_dist),
                        (int)((float)img_height / (float)min_px_dist)); // width x height
    cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
    float size_x = (float)img_width  / (float)grid_x;
    float size_y = (float)img_height / (float)grid_y;
    cv::Size size_grid(grid_x, grid_y); // width x height
    cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
    cv::Mat mask0_updated = mask0.clone();
    auto it0 = pts0.begin();
    auto it1 = ids0.begin();
    while (it0 != pts0.end())
    {
        // Get current left keypoint, check that it is in bounds
        cv::KeyPoint kpt = *it0;
        int x = (int)kpt.pt.x;
        int y = (int)kpt.pt.y;
        int edge = 10;
        if (x < edge || x >= img_width - edge || y < edge || y >= img_height - edge)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate mask coordinates for close points
        int x_close = (int)(kpt.pt.x / (float)min_px_dist);
        int y_close = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate what grid cell this feature is in
        int x_grid = std::floor(kpt.pt.x / size_x);
        int y_grid = std::floor(kpt.pt.y / size_y);
        if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Check if this keypoint is near another point
        if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Now check if it is in a mask area or not
        // NOTE: mask has max value of 255 (white) if it should be
        if (mask0.at<uint8_t>(y, x) > 127)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Else we are good, move forward to the next point
        grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
        if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255)
        {
            grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
        }
        // Append this to the local mask of the image
        if (x - min_px_dist >= 0 && x + min_px_dist < img_width && y - min_px_dist >= 0 && y + min_px_dist < img_height)
        {
            cv::Point pt1(x - min_px_dist, y - min_px_dist);
            cv::Point pt2(x + min_px_dist, y + min_px_dist);
            cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255));
        }
        it0++;
        it1++;
    }

    int64_t rT2 = _apps_time_monotonic_ns();

    // First compute how many more features we need to extract from this image
    // If we don't need any features, just return
    double min_feat_percent = 0.50;
    int num_featsneeded = num_features - (int)pts0.size();
    if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
        return;

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask0_grid;
    cv::resize(mask0, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid.cols; x++)
    {
        for (int y = 0; y < grid_2d_grid.rows; y++)
        {
            if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255)
            {
                valid_locs.emplace_back(x, y);
            }
        }
    }
    std::vector<cv::KeyPoint> pts0_ext;
    Grider_OCL::perform_griding_use_flow(mgr_, cam_id, buf_id, mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);
    
    int64_t rT3 = _apps_time_monotonic_ns();

    // Now, reject features that are close to a current feature
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext)
    {
        // Check that it is in bounds
        int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
        int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
            continue;
        // See if there is a point at this location
        if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
            continue;
        // Else lets add it!
        kpts0_new.push_back(kpt);
        pts0_new.push_back(kpt.pt);
        grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
    }

    // Loop through and record only ones that are valid
    // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
    // NOTE: this is due to the fact that we select update features based on feat id
    // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
    // NOTE: not sure how to remove... maybe a better way?
    for (size_t i = 0; i < pts0_new.size(); i++)
    {
        // update the uv coordinates
        kpts0_new.at(i).pt = pts0_new.at(i);
        // append the new uv coordinate
        pts0.push_back(kpts0_new.at(i));
        // move id foward and append this new point
        size_t temp = ++currid;
        ids0.push_back(temp);
    }
    int64_t rT4 = _apps_time_monotonic_ns();
    // printf("[TIME-DTCT]: %.4f seconds for grid creation\n", (rT2 - rT1) * 1e-6);
    // printf("[TIME-DTCT]: %.4f seconds for grid detection\n", (rT3 - rT2) * 1e-6);
    // printf("[TIME-DTCT]: %.4f seconds for feature rejection\n", (rT4 - rT3) * 1e-6);
}

void TrackOCL::perform_detection_stereo(modal_flow::BufferId buf_id_left, modal_flow::BufferId buf_id_right,
                                        const cv::Mat &mask0, const cv::Mat &mask1,
                                        size_t cam_id_left, size_t cam_id_right,
                                        std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1,
                                        std::vector<size_t> &ids0, std::vector<size_t> &ids1)
{
    int img_width0  = mask0.cols;
    int img_height0 = mask0.rows;
    int img_width1  = mask1.cols;
    int img_height1 = mask1.rows;

    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less then grid_px_size points away then existing features
    cv::Size size_close0((int)((float)img_width0 / (float)min_px_dist),
                        (int)((float)img_height0 / (float)min_px_dist)); // width x height
    cv::Mat grid_2d_close0 = cv::Mat::zeros(size_close0, CV_8UC1);
    float size_x0 = (float)img_width0 / (float)grid_x;
    float size_y0 = (float)img_height0 / (float)grid_y;
    cv::Size size_grid0(grid_x, grid_y); // width x height
    cv::Mat grid_2d_grid0 = cv::Mat::zeros(size_grid0, CV_8UC1);
    cv::Mat mask0_updated = mask0.clone();
    auto it0 = pts0.begin();
    auto it1 = ids0.begin();
    while (it0 != pts0.end()) {
        // Get current left keypoint, check that it is in bounds
        cv::KeyPoint kpt = *it0;
        int x = (int)kpt.pt.x;
        int y = (int)kpt.pt.y;
        int edge = 10;
        if (x < edge || x >= img_width0 - edge || y < edge || y >= img_height0 - edge) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate mask coordinates for close points
        int x_close = (int)(kpt.pt.x / (float)min_px_dist);
        int y_close = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_close < 0 || x_close >= size_close0.width || y_close < 0 || y_close >= size_close0.height) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate what grid cell this feature is in
        int x_grid = std::floor(kpt.pt.x / size_x0);
        int y_grid = std::floor(kpt.pt.y / size_y0);
        if (x_grid < 0 || x_grid >= size_grid0.width || y_grid < 0 || y_grid >= size_grid0.height) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Check if this keypoint is near another point
        if (grid_2d_close0.at<uint8_t>(y_close, x_close) > 127) {
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
        grid_2d_close0.at<uint8_t>(y_close, x_close) = 255;
        if (grid_2d_grid0.at<uint8_t>(y_grid, x_grid) < 255) {
            grid_2d_grid0.at<uint8_t>(y_grid, x_grid) += 1;
        }
        // Append this to the local mask of the image
        if (x - min_px_dist >= 0 && x + min_px_dist < img_width0 && y - min_px_dist >= 0 && y + min_px_dist < img_height0) {
            cv::Point pt1(x - min_px_dist, y - min_px_dist);
            cv::Point pt2(x + min_px_dist, y + min_px_dist);
            cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
        }
        it0++;
        it1++;
    }

    // First compute how many more features we need to extract from this image
    double min_feat_percent = 0.50;
    int num_featsneeded_0 = num_features - (int)pts0.size();

    // LEFT: if we need features we should extract them in the current frame
    // LEFT: we will also try to track them from this frame over to the right frame
    // LEFT: in the case that we have two features that are the same, then we should merge them
    if (num_featsneeded_0 > std::min(20, (int)(min_feat_percent * num_features))) {
        // We also check a downsampled mask such that we don't extract in areas where it is all masked!
        cv::Mat mask0_grid;
        cv::resize(mask0, mask0_grid, size_grid0, 0.0, 0.0, cv::INTER_NEAREST);

        // Create grids we need to extract from and then extract our features (use fast with griding)
        int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
        int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
        std::vector<std::pair<int, int>> valid_locs;
        for (int x = 0; x < grid_2d_grid0.cols; x++) {
            for (int y = 0; y < grid_2d_grid0.rows; y++) {
                if ((int)grid_2d_grid0.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
                    valid_locs.emplace_back(x, y);
                }
            }
        }
        std::vector<cv::KeyPoint> pts0_ext;
        Grider_OCL::perform_griding_use_flow(mgr_, cam_id_left, buf_id_left, mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

        // Now, reject features that are close a current feature
        std::vector<cv::KeyPoint> kpts0_new;
        std::vector<cv::Point2f> pts0_new;
        for (auto &kpt : pts0_ext) {
            // Check that it is in bounds
            int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
            int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
            if (x_grid < 0 || x_grid >= size_close0.width || y_grid < 0 || y_grid >= size_close0.height)
                continue;
            // See if there is a point at this location
            if (grid_2d_close0.at<uint8_t>(y_grid, x_grid) > 127)
                continue;
            // Else lets add it!
            grid_2d_close0.at<uint8_t>(y_grid, x_grid) = 255;
            kpts0_new.push_back(kpt);
            pts0_new.push_back(kpt.pt);
        }


        // Project the new left features into the right image via the ZNCC
        // epipolar-band matcher (set up once at startup by
        // enable_zncc_stereo_matcher). The matcher does forward+reverse ZNCC
        // along the calibrated epipolar curve and gates on peak / margin /
        // L-R round-trip consistency; per-feature confidence is stashed in
        // stereo_confidence_ for downstream EKF measurement weighting.
        std::vector<cv::KeyPoint> kpts1_new;
        std::vector<cv::Point2f>  pts1_new;
        kpts1_new = kpts0_new;
        pts1_new  = pts0_new;

        if (pts0_new.empty()) {
            // nothing to project -- fall through to the right-image dedupe loop below
        } else if (cam_id_left  == stereo_cam_id_left_ &&
                   cam_id_right == stereo_cam_id_right_) {
            // Pre-compute normalized cam0 bearings for every new left feature.
            // camera_calib already exists in the base class and exposes the
            // ov_core fisheye/radtan undistort -- exact match for what the
            // GPU matcher expects (it does no iterative undistort itself).
            modal_flow::StereoMatchInput in{};
            in.left_cam_id   = (modal_flow::CameraId)cam_id_left;
            in.right_cam_id  = (modal_flow::CameraId)cam_id_right;
            // Match against the buffer this function was called for, NOT
            // img_buf_next_. pts0_new and buf_id_* refer to the same frame
            // (prev on subsequent-frame calls). Using img_buf_next_ would
            // align positions to prev but images to next -- lr_err then
            // measures inter-frame motion (~8-60 px) instead of the round-trip
            // residual. perform_matching below carries accepted prev-frame
            // pairs forward to current via temporal KLT.
            in.left_img_buf  = buf_id_left;
            in.right_img_buf = buf_id_right;
            in.left_points  .reserve(pts0_new.size());
            in.left_bearings.reserve(pts0_new.size());
            for (const auto &p : pts0_new) {
                Eigen::Vector2f n = camera_calib.at(cam_id_left)->undistort_f(Eigen::Vector2f(p.x, p.y));
                in.left_points  .push_back({p.x, p.y, 0.f});
                in.left_bearings.push_back({n(0), n(1), 0.f});
            }

            modal_flow::StereoMatchResult res = mgr_.match_stereo(in);

            // Per-call validation print so we can see L<->R correlation working live.
            // Gated by kEnableStereoZnccDiag (compile-time, off in production).
            if constexpr (kEnableStereoZnccDiag) {
                int    n_accept = 0;
                double sum_disp = 0, sum_peak = 0, sum_lr = 0;
                for (size_t i = 0; i < pts0_new.size(); i++) {
                    if (!res.status[i]) continue;
                    float dx = res.right_points[i].x - pts0_new[i].x;
                    float dy = res.right_points[i].y - pts0_new[i].y;
                    sum_disp += std::sqrt(dx * dx + dy * dy);
                    sum_peak += res.peak_zncc[i];
                    sum_lr   += res.lr_err[i];
                    n_accept++;
                }
                if (n_accept > 0) {
                    printf("[STEREO ZNCC] cam %zu->%zu  new=%zu  accepted=%d (%.0f%%)  "
                           "disp=%.1fpx  peakZNCC=%.2f  lrErr=%.2fpx\n",
                           cam_id_left, cam_id_right, pts0_new.size(), n_accept,
                           100.0 * n_accept / pts0_new.size(),
                           sum_disp / n_accept, sum_peak / n_accept, sum_lr / n_accept);
                } else {
                    // Per-gate breakdown when no features accept. Thresholds must
                    // mirror StereoMatcherCL's current settings (0.60 / 0.10 / 5.0).
                    // We report both marginal counts and the joint (peak∧margin and
                    // peak∧margin∧lr) -- marginals add to N even when joint is 0
                    // because the three gates can have disjoint pass populations.
                    const float zncc_min   = 0.60f;
                    const float margin_min = 0.10f;
                    const float lr_thresh  = 5.0f;
                    int    n_oob = 0, n_peak_ok = 0, n_marg_ok = 0, n_lr_ok = 0;
                    int    n_joint_pm = 0, n_full = 0;
                    int    n_valid = 0;
                    float  peak_min =  1e9f, peak_max = -1e9f, peak_sum = 0.f;
                    float  marg_min =  1e9f, marg_max = -1e9f, marg_sum = 0.f;
                    float  lr_min   =  1e9f, lr_max   = -1e9f, lr_sum   = 0.f;
                    for (size_t i = 0; i < pts0_new.size(); i++) {
                        const float p  = res.peak_zncc[i];
                        const float m  = res.margin[i];
                        const float lr = res.lr_err[i];
                        if (p <= -1.5f) { n_oob++; continue; }
                        n_valid++;
                        if (p > peak_max) peak_max = p; if (p < peak_min) peak_min = p; peak_sum += p;
                        if (m > marg_max) marg_max = m; if (m < marg_min) marg_min = m; marg_sum += m;
                        if (lr > lr_max)  lr_max  = lr; if (lr < lr_min)  lr_min  = lr; lr_sum  += lr;
                        const bool peak_ok = (p  >= zncc_min);
                        const bool marg_ok = (m  >= margin_min);
                        const bool lr_ok   = (lr <  lr_thresh);
                        if (peak_ok) n_peak_ok++;
                        if (marg_ok) n_marg_ok++;
                        if (lr_ok)   n_lr_ok++;
                        if (peak_ok && marg_ok)            n_joint_pm++;
                        if (peak_ok && marg_ok && lr_ok)   n_full++;
                    }
                    printf("[STEREO ZNCC] cam %zu->%zu  new=%zu  accepted=0  oob=%d  "
                           "peak>=%.2f:%d/%d  margin>=%.2f:%d/%d  lr<%.1fpx:%d/%d  "
                           "JOINT pm:%d/%d  full:%d/%d\n",
                           cam_id_left, cam_id_right, pts0_new.size(), n_oob,
                           zncc_min, n_peak_ok, n_valid, margin_min, n_marg_ok, n_valid,
                           lr_thresh, n_lr_ok, n_valid,
                           n_joint_pm, n_valid, n_full, n_valid);
                    if (n_valid > 0) {
                        printf("[STEREO ZNCC]   new stats  peak[%.2f/%.2f/%.2f]  "
                               "margin[%.2f/%.2f/%.2f]  lr_px[%.1f/%.1f/%.1f]\n",
                               peak_min, peak_sum / n_valid, peak_max,
                               marg_min, marg_sum / n_valid, marg_max,
                               lr_min,   lr_sum   / n_valid, lr_max);
                    }
                }
            }

            for (size_t i = 0; i < pts0_new.size(); i++) {
                bool oob_left = ((int)pts0_new.at(i).x < 0 || (int)pts0_new.at(i).x >= img_width0 ||
                                 (int)pts0_new.at(i).y < 0 || (int)pts0_new.at(i).y >= img_height0);
                if (oob_left) continue;

                if (res.status[i]) {
                    cv::Point2f rpt(res.right_points[i].x, res.right_points[i].y);
                    bool oob_right = ((int)rpt.x < 0 || (int)rpt.x >= img_width1 ||
                                      (int)rpt.y < 0 || (int)rpt.y >= img_height1);
                    if (!oob_right) {
                        kpts0_new.at(i).pt = pts0_new.at(i);
                        kpts1_new.at(i).pt = rpt;
                        pts0.push_back(kpts0_new.at(i));
                        pts1.push_back(kpts1_new.at(i));
                        size_t temp = ++currid;
                        ids0.push_back(temp);
                        ids1.push_back(temp);
                        // Stash confidence for the eventual EKF measurement-noise weighting.
                        stereo_confidence_[temp] = StereoConfidence{
                            res.peak_zncc[i], res.margin[i], res.lr_err[i]};
                        continue;
                    }
                }
                // Match rejected (or right oob) -> still record as a mono left feature.
                kpts0_new.at(i).pt = pts0_new.at(i);
                pts0.push_back(kpts0_new.at(i));
                ids0.push_back(++currid);
            }
        } else {
            // ZNCC matcher not bound to this cam pair -- unreachable in normal
            // operation since enable_zncc_stereo_matcher is called at startup
            // whenever use_stereo is true. As a defensive fallback, record the
            // FAST corners as mono-left features so they're temporally tracked;
            // the mono->stereo promote pass below will upgrade them once the
            // matcher is bound.
            for (size_t i = 0; i < pts0_new.size(); i++) {
                bool oob_left = ((int)pts0_new.at(i).x < 0 || (int)pts0_new.at(i).x >= img_width0 ||
                                 (int)pts0_new.at(i).y < 0 || (int)pts0_new.at(i).y >= img_height0);
                if (oob_left) continue;
                kpts0_new.at(i).pt = pts0_new.at(i);
                pts0.push_back(kpts0_new.at(i));
                ids0.push_back(++currid);
            }
        }
    }

    // Mono->stereo promote pass: re-run match_stereo on every existing mono-left
    // track (ids0[i] not in ids1) and upgrade successful matches to full stereo
    // pairs by appending the right point to pts1/ids1 with the same id. The
    // right-image dedup loop below preserves features whose id is in ids0 via
    // its is_stereo branch, so newly-promoted pairs survive cleanup.
    if (cam_id_left  == stereo_cam_id_left_ &&
        cam_id_right == stereo_cam_id_right_ &&
        !pts0.empty())
    {
        std::unordered_set<size_t> right_ids(ids1.begin(), ids1.end());

        modal_flow::StereoMatchInput in{};
        in.left_cam_id   = (modal_flow::CameraId)cam_id_left;
        in.right_cam_id  = (modal_flow::CameraId)cam_id_right;
        // Use this function's buf_id parameters; see the longer comment on the
        // top-off match call above for why img_buf_next_ would mis-align.
        in.left_img_buf  = buf_id_left;
        in.right_img_buf = buf_id_right;

        // Map kept index in `in` -> index in pts0/ids0 so we can write back.
        std::vector<size_t> src_idx;
        src_idx.reserve(pts0.size());
        in.left_points  .reserve(pts0.size());
        in.left_bearings.reserve(pts0.size());
        for (size_t i = 0; i < pts0.size(); i++) {
            if (right_ids.count(ids0[i])) continue;            // already stereo
            const cv::Point2f &p = pts0[i].pt;
            // Skip features too close to the border for the matcher's 11x11 patch.
            // BORDER=12 in the kernel; use a safe 15 here. Out-of-bounds and edge
            // features cause the matcher to return the OOB sentinel anyway -- skipping
            // them up-front saves a kernel slot and keeps the accept-rate print honest.
            if (p.x < 15 || p.x >= img_width0 - 15 ||
                p.y < 15 || p.y >= img_height0 - 15) continue;
            Eigen::Vector2f n = camera_calib.at(cam_id_left)->undistort_f(Eigen::Vector2f(p.x, p.y));
            in.left_points  .push_back({p.x, p.y, 0.f});
            in.left_bearings.push_back({n(0), n(1), 0.f});
            src_idx.push_back(i);
        }

        if (!in.left_points.empty()) {
            modal_flow::StereoMatchResult res = mgr_.match_stereo(in);
            // Accumulators are only consumed by the kEnableStereoZnccDiag print
            // block below; tagged maybe_unused so off-builds don't warn.
            [[maybe_unused]] int    n_promote = 0;
            [[maybe_unused]] double sum_disp = 0, sum_peak = 0, sum_lr = 0;
            for (size_t k = 0; k < in.left_points.size(); k++) {
                if (!res.status[k]) continue;
                cv::Point2f rpt(res.right_points[k].x, res.right_points[k].y);
                if ((int)rpt.x < 0 || (int)rpt.x >= img_width1 ||
                    (int)rpt.y < 0 || (int)rpt.y >= img_height1) continue;
                size_t i = src_idx[k];
                cv::KeyPoint rkpt = pts0[i]; // copy keypoint attributes (size, octave, etc.)
                rkpt.pt = rpt;
                pts1.push_back(rkpt);
                ids1.push_back(ids0[i]);     // SAME id == becomes a stereo pair
                stereo_confidence_[ids0[i]] = StereoConfidence{
                    res.peak_zncc[k], res.margin[k], res.lr_err[k]};
                float dx = rpt.x - pts0[i].pt.x;
                float dy = rpt.y - pts0[i].pt.y;
                sum_disp += std::sqrt(dx * dx + dy * dy);
                sum_peak += res.peak_zncc[k];
                sum_lr   += res.lr_err[k];
                n_promote++;
            }
            if constexpr (kEnableStereoZnccDiag) {
                if (n_promote > 0) {
                    printf("[STEREO ZNCC] mono->stereo promote: %d/%zu (%.0f%%)  "
                           "disp=%.1fpx  peakZNCC=%.2f  lrErr=%.2fpx  (cam %zu->%zu)\n",
                           n_promote, in.left_points.size(),
                           100.0 * n_promote / in.left_points.size(),
                           sum_disp / n_promote, sum_peak / n_promote, sum_lr / n_promote,
                           cam_id_left, cam_id_right);
                } else {
                    // Mirrors the per-gate breakdown on the new-features path; see the
                    // comment there for why we report joint counts alongside marginals.
                    const float zncc_min   = 0.60f;
                    const float margin_min = 0.10f;
                    const float lr_thresh  = 5.0f;
                    int    n_oob = 0, n_peak_ok = 0, n_marg_ok = 0, n_lr_ok = 0;
                    int    n_joint_pm = 0, n_full = 0;
                    int    n_valid = 0;
                    float  peak_min =  1e9f, peak_max = -1e9f, peak_sum = 0.f;
                    float  marg_min =  1e9f, marg_max = -1e9f, marg_sum = 0.f;
                    float  lr_min   =  1e9f, lr_max   = -1e9f, lr_sum   = 0.f;
                    for (size_t k = 0; k < in.left_points.size(); k++) {
                        const float p  = res.peak_zncc[k];
                        const float m  = res.margin[k];
                        const float lr = res.lr_err[k];
                        if (p <= -1.5f) { n_oob++; continue; }
                        n_valid++;
                        if (p > peak_max) peak_max = p; if (p < peak_min) peak_min = p; peak_sum += p;
                        if (m > marg_max) marg_max = m; if (m < marg_min) marg_min = m; marg_sum += m;
                        if (lr > lr_max)  lr_max  = lr; if (lr < lr_min)  lr_min  = lr; lr_sum  += lr;
                        const bool peak_ok = (p  >= zncc_min);
                        const bool marg_ok = (m  >= margin_min);
                        const bool lr_ok   = (lr <  lr_thresh);
                        if (peak_ok) n_peak_ok++;
                        if (marg_ok) n_marg_ok++;
                        if (lr_ok)   n_lr_ok++;
                        if (peak_ok && marg_ok)            n_joint_pm++;
                        if (peak_ok && marg_ok && lr_ok)   n_full++;
                    }
                    printf("[STEREO ZNCC] mono->stereo promote: 0/%zu  oob=%d  "
                           "peak>=%.2f:%d/%d  margin>=%.2f:%d/%d  lr<%.1fpx:%d/%d  "
                           "JOINT pm:%d/%d  full:%d/%d\n",
                           in.left_points.size(), n_oob,
                           zncc_min, n_peak_ok, n_valid, margin_min, n_marg_ok, n_valid,
                           lr_thresh, n_lr_ok, n_valid,
                           n_joint_pm, n_valid, n_full, n_valid);
                    if (n_valid > 0) {
                        printf("[STEREO ZNCC]   promote stats  peak[%.2f/%.2f/%.2f]  "
                               "margin[%.2f/%.2f/%.2f]  lr_px[%.1f/%.1f/%.1f]\n",
                               peak_min, peak_sum / n_valid, peak_max,
                               marg_min, marg_sum / n_valid, marg_max,
                               lr_min,   lr_sum   / n_valid, lr_max);
                    }
                }
            }
        }
    }

    // RIGHT: Now summarise the number of tracks in the right image
    // RIGHT: We will try to extract some monocular features if we have the room
    // RIGHT: This will also remove features if there are multiple in the same location
    cv::Size size_close1((int)((float)img_width1 / (float)min_px_dist), (int)((float)img_height1 / (float)min_px_dist));
    cv::Mat grid_2d_close1 = cv::Mat::zeros(size_close1, CV_8UC1);
    float size_x1 = (float)img_width1  / (float)grid_x;
    float size_y1 = (float)img_height1 / (float)grid_y;
    cv::Size size_grid1(grid_x, grid_y); // width x height
    cv::Mat grid_2d_grid1 = cv::Mat::zeros(size_grid1, CV_8UC1);
    cv::Mat mask1_updated = mask0.clone();
    it0 = pts1.begin();
    it1 = ids1.begin();
    
    while (it0 != pts1.end()) {
        // Get current left keypoint, check that it is in bounds
        cv::KeyPoint kpt = *it0;
        int x = (int)kpt.pt.x;
        int y = (int)kpt.pt.y;
        int edge = 10;
        if (x < edge || x >= img_width1 - edge || y < edge || y >= img_height1 - edge) {
            it0 = pts1.erase(it0);
            it1 = ids1.erase(it1);
            continue;
        }
        // Calculate mask coordinates for close points
        int x_close = (int)(kpt.pt.x / (float)min_px_dist);
        int y_close = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_close < 0 || x_close >= size_close1.width || y_close < 0 || y_close >= size_close1.height) {
            it0 = pts1.erase(it0);
            it1 = ids1.erase(it1);
            continue;
        }
        // Calculate what grid cell this feature is in
        int x_grid = std::floor(kpt.pt.x / size_x1);
        int y_grid = std::floor(kpt.pt.y / size_y1);
        if (x_grid < 0 || x_grid >= size_grid1.width || y_grid < 0 || y_grid >= size_grid1.height) {
            it0 = pts1.erase(it0);
            it1 = ids1.erase(it1);
            continue;
        }
        // Check if this keypoint is near another point
        // NOTE: if it is *not* a stereo point, then we will not delete the feature
        // NOTE: this means we might have a mono and stereo feature near each other, but that is ok
        bool is_stereo = (std::find(ids0.begin(), ids0.end(), *it1) != ids0.end());
        if (grid_2d_close1.at<uint8_t>(y_close, x_close) > 127 && !is_stereo) {
            it0 = pts1.erase(it0);
            it1 = ids1.erase(it1);
            continue;
        }
        // Now check if it is in a mask area or not
        // NOTE: mask has max value of 255 (white) if it should be
        if (mask1.at<uint8_t>(y, x) > 127) {
            it0 = pts1.erase(it0);
            it1 = ids1.erase(it1);
            continue;
        }
        // Else we are good, move forward to the next point
        grid_2d_close1.at<uint8_t>(y_close, x_close) = 255;
        if (grid_2d_grid1.at<uint8_t>(y_grid, x_grid) < 255) {
            grid_2d_grid1.at<uint8_t>(y_grid, x_grid) += 1;
        }

        // Append this to the local mask of the image
        if (x - min_px_dist >= 0 && x + min_px_dist < img_width1 && y - min_px_dist >= 0 && y + min_px_dist < img_height1) {
            cv::Point pt1(x - min_px_dist, y - min_px_dist);
            cv::Point pt2(x + min_px_dist, y + min_px_dist);
            cv::rectangle(mask1_updated, pt1, pt2, cv::Scalar(255), -1);
        }
        it0++;
        it1++;
    }

    // RIGHT: if we need features we should extract them in the current frame
    // RIGHT: note that we don't track them to the left as we already did left->right tracking above
    int num_featsneeded_1 = num_features - (int)pts1.size();
    if (num_featsneeded_1 > std::min(20, (int)(min_feat_percent * num_features))) {

        // We also check a downsampled mask such that we don't extract in areas where it is all masked!
        cv::Mat mask1_grid;
        cv::resize(mask1, mask1_grid, size_grid1, 0.0, 0.0, cv::INTER_NEAREST);

        // Create grids we need to extract from and then extract our features (use fast with griding)
        int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
        int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
        std::vector<std::pair<int, int>> valid_locs;
        for (int x = 0; x < grid_2d_grid1.cols; x++) {
            for (int y = 0; y < grid_2d_grid1.rows; y++) {
                if ((int)grid_2d_grid1.at<uint8_t>(y, x) < num_features_grid_req && (int)mask1_grid.at<uint8_t>(y, x) != 255) {
                valid_locs.emplace_back(x, y);
                }
            }
        }
        std::vector<cv::KeyPoint> pts1_ext;
        Grider_OCL::perform_griding_use_flow(mgr_, cam_id_right, buf_id_right, mask1_updated, valid_locs, pts1_ext, num_features, grid_x, grid_y, threshold, true);

        // Now, reject features that are close a current feature
        for (auto &kpt : pts1_ext) {
            // Check that it is in bounds
            int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
            int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
            if (x_grid < 0 || x_grid >= size_close1.width || y_grid < 0 || y_grid >= size_close1.height)
                continue;
            // See if there is a point at this location
            if (grid_2d_close1.at<uint8_t>(y_grid, x_grid) > 127)
                continue;
            // Else lets add it!
            pts1.push_back(kpt);
            size_t temp = ++currid;
            ids1.push_back(temp);
            grid_2d_close1.at<uint8_t>(y_grid, x_grid) = 255;
        }
    }
        
    return;
}

void TrackOCL::perform_matching(modal_flow::BufferId buf0, modal_flow::BufferId buf1, std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out)
{

    // We must have equal vectors
    assert(kpts0.size() == kpts1.size());

    // Return if we don't have any points
    if (kpts0.empty() || kpts1.empty())
        return;

    // Convert keypoints into points (stupid opencv stuff)
    std::vector<cv::Point2f> pts0, pts1;
    std::vector<modal_flow::Keypoint> pts_in;
    std::vector<float> pts_out;
    for (size_t i = 0; i < kpts0.size(); i++)
    {
        pts0.push_back(kpts0.at(i).pt);
        pts1.push_back(kpts1.at(i).pt);

        modal_flow::Keypoint kp;
        kp.x = kpts0.at(i).pt.x;
        kp.y = kpts0.at(i).pt.y;
        kp.score = 0.f;
        pts_in.push_back(kp);

        // for gpu run
        pts_out.push_back(kpts0.at(i).pt.x);
        pts_out.push_back(kpts0.at(i).pt.y);
    }

    // If we don't have enough points for ransac just return empty
    // We set the mask to be all zeros since all points failed RANSAC
    if (pts0.size() < 10)
    {
        for (size_t i = 0; i < pts0.size(); i++)
            mask_out.push_back((uchar)0);
        return;
    }

    std::vector<uchar> mask_klt;
    int64_t t0 = _apps_time_monotonic_ns();

    modal_flow::TrackingBatch track_batch;
    modal_flow::TrackOptions topt;

    size_t cam_id = id0;
    std::pair<int, int> dims = mgr_.get_cam_dim(cam_id);

    std::vector<modal_flow::TrackInput> track_in(1);
    track_in[0].prev_cam_id = id0;
    track_in[0].next_cam_id = id1;
    track_in[0].prev_img_buf = buf0;
    track_in[0].next_img_buf = buf1;
    track_in[0].prev_points = pts_in;

    auto res = mgr_.track_many(track_in);

    int64_t t1 = _apps_time_monotonic_ns();
    // printf("[TIME-KLT-INTERNAL]: run_track = %.3f ms\n", (t1 - t0) / 1e6);

    int n_points = res[0].next_points.size();
    mask_klt.resize(n_points);

    for (int i = 0; i < n_points; i++)
    {
        modal_flow::Keypoint point = res[0].next_points[i];
        pts1[i] = (cv::Point2f){point.x, point.y};
        mask_klt[i] = res[0].status[i];
    }

    std::vector<cv::Point2f> pts0_keep, pts1_keep;
    std::vector<int>         keep_idx;        // map back to original i

    pts0_keep.reserve(pts0.size());
    pts1_keep.reserve(pts1.size());
    keep_idx.reserve(pts0.size());

    for (size_t i = 0; i < pts0.size(); ++i) {
        if (mask_klt[i]) {                    // only keep successfully tracked points
            pts0_keep.push_back(pts0[i]);
            pts1_keep.push_back(pts1[i]);
            keep_idx.push_back((int)i);
        }
    }

    // Normalize these points, so we can then do ransac
    // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
    std::vector<cv::Point2f> pts0_n, pts1_n;
    for (size_t i = 0; i < pts0_keep.size(); i++)
    {
        pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0_keep.at(i)));
        pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1_keep.at(i)));
    }
    int64_t t2 = _apps_time_monotonic_ns();
    // printf("[TIME-KLT-INTERNAL]: undistort = %.3f ms\n", (t2 - t1) / 1e6);

    // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
    std::vector<uchar> mask_rsc;
    double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
    double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
    double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
    cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.95, 50, mask_rsc);
    // cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc);

    // int inliers = 0, outliers = 0;
    // for (auto m : mask_rsc) {
    //     if (m) inliers++;
    //     else   outliers++;
    // }

    // printf("[RANSAC] Inliers = %d, Outliers = %d, Total = %zu\n",
    //     inliers, outliers, mask_rsc.size());

    // printf("[RANSAC] Dumping %zu point pairs:\n", pts0_n.size());
    // for (size_t i = 0; i < pts0_n.size(); i++) {
    //     printf("  [%zu] (%.3f, %.3f) -> (%.3f, %.3f), status=%d\n",
    //         i, pts0_n[i].x, pts0_n[i].y,
    //         pts1_n[i].x, pts1_n[i].y,
    //         mask_rsc[i]);
    // }

    int64_t t3 = _apps_time_monotonic_ns();
    // printf("[TIME-KLT-INTERNAL]: perform_matching total = %6.3f ms,  RANSAC = %6.3f ms\n", (t3 - t0) / 1e6, (t3 - t2) / 1e6);

    // Loop through and record only ones that are valid
    // Expand compact RANSAC mask back to original indexing
    mask_out.assign(pts0.size(), (uchar)0);

    size_t M = std::min(mask_rsc.size(), keep_idx.size());
    for (size_t j = 0; j < M; ++j) {
        if (mask_rsc[j]) {
            int i_orig = keep_idx[j];
            // mask_klt[i_orig] is already true for kept points; AND for clarity
            mask_out[i_orig] = (uchar)1;
        }
    }

    // Copy back the updated positions
    for (size_t i = 0; i < pts0.size(); i++)
    {
        kpts0.at(i).pt = pts0.at(i);
        kpts1.at(i).pt = pts1.at(i);
    }
}
