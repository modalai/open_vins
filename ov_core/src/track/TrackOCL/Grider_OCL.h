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

#ifndef OV_CORE_GRIDER_OCL_H
#define OV_CORE_GRIDER_OCL_H

#include <Eigen/Eigen>
#include <functional>
#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "utils/opencv_lambda_body.h"
#include <modal_flow_track_manager.h>
#include <modal_flow/ocl/ManagerCL.hpp>
#include <modal_flow/Types.hpp>
namespace ov_core {

/**
 * @brief Extracts FAST features in a grid pattern.
 *
 * As compared to just extracting fast features over the entire image,
 * we want to have as uniform of extractions as possible over the image plane.
 * Thus we split the image into a bunch of small grids, and extract points in each.
 * We then pick enough top points in each grid so that we have the total number of desired points.
 */
class Grider_OCL {

  public:
    /**
     * @brief Compare keypoints based on their response value.
     * @param first First keypoint
     * @param second Second keypoint
     *
     * We want to have the keypoints with the highest values!
     * See: https://stackoverflow.com/a/10910921
     */
    static bool compare_response(cv::KeyPoint first, cv::KeyPoint second) { return first.response > second.response; }


    template<typename pt>
    struct cmp_pt
    {
        bool operator ()(const pt& a, const pt& b) const { return a.y < b.y || (a.y == b.y && a.x < b.x); }
    };

    /**
     * @brief This function will perform grid extraction using FAST.
     * @param img Image we will do FAST extraction on
     * @param mask Region of the image we do not want to extract features in (255 = do not detect features)
     * @param valid_locs Valid 2d grid locations we will extract in (instead of the whole image)
     * @param pts vector of extracted points we will return
     * @param num_features max number of features we want to extract
     * @param grid_x size of grid in the x-direction / u-direction
     * @param grid_y size of grid in the y-direction / v-direction
     * @param threshold FAST threshold paramter (10 is a good value normally)
     * @param nonmaxSuppression if FAST should perform non-max suppression (true normally)
     *
     * Given a specified grid size, this will try to extract fast features from each grid.
     * It will then return the best from each grid in the return vector.
     */
    static void perform_griding(VoxlTrackingManager* tracker, const cv::Mat &img, const cv::Mat &mask, const std::vector<std::pair<int, int>> &valid_locs,
                                std::vector<cv::KeyPoint> &pts, int num_features, int grid_x, int grid_y, int threshold,
                                bool nonmaxSuppression) {

        // Return if there is nothing to extract
        if (valid_locs.empty())
            return;

        // We want to have equally distributed features
        // NOTE: If we have more grids than number of total points, we calc the biggest grid we can do
        // NOTE: Thus if we extract 1 point per grid we have
        // NOTE:    -> 1 = num_features / (grid_x * grid_y)
        // NOTE:    -> grid_x = ratio * grid_y (keep the original grid ratio)
        // NOTE:    -> grid_y = sqrt(num_features / ratio)
        if (num_features < grid_x * grid_y) {
            double ratio = (double)grid_x / (double)grid_y;
            grid_y = std::ceil(std::sqrt(num_features / ratio));
            grid_x = std::ceil(grid_y * ratio);
        }
        int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
        assert(grid_x > 0);
        assert(grid_y > 0);
        assert(num_features_grid > 0);

        // Calculate the size our extraction boxes should be
        int size_x = img.cols / grid_x;
        int size_y = img.rows / grid_y;

        // Make sure our sizes are not zero
        assert(size_x > 0);
        assert(size_y > 0);

        std::vector<std::vector<cv::KeyPoint>> collection(valid_locs.size());
        int* kp_out = (int*)malloc((tracker->maxPointsExtracted() * 3 + 1) * sizeof(int));

        for (int r = 0 ; r < (int)valid_locs.size(); r++)
        {
            // Calculate what cell xy value we are in
            auto grid = valid_locs.at(r);
            int x = grid.first * size_x;
            int y = grid.second * size_y;
        

            // Skip if we are out of bounds
            if (x + size_x > img.cols || y + size_y > img.rows)
                continue;


            // create grid_img from img and x,y,size,size_y
            cv::Rect img_roi = cv::Rect(x, y, size_x, size_y);
            cv::Mat grid_img_roi(img(img_roi));
            cv::Mat grid_img = grid_img_roi.clone();

            memset(kp_out, 0, (tracker->maxPointsExtracted() * 3 + 1) * sizeof(int));

            tracker->runExtraction(grid_img, size_x, size_y, threshold, kp_out);

            std::vector<cv::KeyPoint> pts_new;

            int counter = std::min(tracker->maxPointsExtracted(), kp_out[0]);
            if (counter == 0) continue;

            cv::Point3i* pt2 = (cv::Point3i*)(kp_out + 1);
            std::sort(pt2, pt2 + counter, cmp_pt<cv::Point3i>());

            for( int i = 0; i < counter; i++ )
                pts_new.push_back(cv::KeyPoint((float)pt2[i].x, (float)pt2[i].y, 7.f, -1, (float)pt2[i].z));
            
            std::sort(pts_new.begin(), pts_new.end(), Grider_FAST::compare_response);

            for (size_t i = 0; i < (size_t)num_features_grid && i < pts_new.size(); i++) {

                // Create keypoint
                cv::KeyPoint pt_cor = pts_new.at(i);
                pt_cor.pt.x += (float)x;
                pt_cor.pt.y += (float)y;

                // Reject if out of bounds (shouldn't be possible...)
                if ((int)pt_cor.pt.x < 0 || (int)pt_cor.pt.x > img.cols || (int)pt_cor.pt.y < 0 ||
                    (int)pt_cor.pt.y > img.rows)
                    continue;

                // Check if it is in the mask region
                // NOTE: mask has max value of 255 (white) if it should be removed
                if (mask.at<uint8_t>((int)pt_cor.pt.y, (int)pt_cor.pt.x) > 127)
                    continue;

                collection.at(r).push_back(pt_cor);
            }
        }
        free(kp_out);

        // Combine all the collections into our single vector
        for (size_t r = 0; r < collection.size(); r++) {
            pts.insert(pts.end(), collection.at(r).begin(), collection.at(r).end());
        }

        // Return if no points
        if (pts.empty())
            return;

        // Sub-pixel refinement parameters
        cv::Size win_size = cv::Size(5, 5);
        cv::Size zero_zone = cv::Size(-1, -1);
        cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001);

        // Get vector of points
        std::vector<cv::Point2f> pts_refined;
        for (size_t i = 0; i < pts.size(); i++) {
            pts_refined.push_back(pts.at(i).pt);
        }

        // Finally run sub pixel refinement (cv::cornerSubPix)
        int res = tracker->refinePoints(static_cast<int>(pts_refined.size()),
                                reinterpret_cast<float*>(pts_refined.data()),
                                win_size.width,
                                term_crit.maxCount,
                                static_cast<float>(term_crit.epsilon));
        if (res != 0) {
            std::cerr << "Subpixel refinement failed." << std::endl;
            return;
        }

        // Save the refined points!
        for (size_t i = 0; i < pts.size(); i++) {
            pts.at(i).pt = pts_refined.at(i);
        }
    }

    static void perform_griding_use_flow(modal_flow::ocl::ManagerCL& mgr, int cam_id, modal_flow::BufferId buf_id, const cv::Mat &mask, const std::vector<std::pair<int, int>> &valid_locs,
                            std::vector<cv::KeyPoint> &pts, int num_features, int grid_x, int grid_y, int threshold,
                            bool nonmaxSuppression) {

        // Return if there is nothing to extract
        if (valid_locs.empty())
            return;

        // We want to have equally distributed features
        // NOTE: If we have more grids than number of total points, we calc the biggest grid we can do
        // NOTE: Thus if we extract 1 point per grid we have
        // NOTE:    -> 1 = num_features / (grid_x * grid_y)
        // NOTE:    -> grid_x = ratio * grid_y (keep the original grid ratio)
        // NOTE:    -> grid_y = sqrt(num_features / ratio)
        if (num_features < grid_x * grid_y) {
            double ratio = (double)grid_x / (double)grid_y;
            grid_y = std::ceil(std::sqrt(num_features / ratio));
            grid_x = std::ceil(grid_y * ratio);
        }
        int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
        assert(grid_x > 0);
        assert(grid_y > 0);
        assert(num_features_grid > 0);

        int img_width  = mask.cols;
        int img_height = mask.rows;

        // Calculate the size our extraction boxes should be
        int size_x = img_width  / grid_x;
        int size_y = img_height / grid_y;

        // Make sure our sizes are not zero
        assert(size_x > 0);
        assert(size_y > 0);

        auto pack_cell = [&](int cx, int cy)
        {
            return size_t(cx) * size_t(grid_y) + size_t(cy);
        };

        // restrict to valid cells
        std::unordered_set<size_t> valid_set;
        valid_set.reserve(valid_locs.size());
        for (auto &c : valid_locs) valid_set.insert(pack_cell(c.first, c.second));
        
        // run detection here        
        modal_flow::DetectOptions dopt;
        dopt.border_x = 3;
        dopt.border_y = 3;
        dopt.threshold = threshold;

        dopt.use_grid_detect = true;
        dopt.horizontal_grid_cells = grid_x;
        dopt.vertical_grid_cells   = grid_y;
        dopt.grid_cells_to_search.clear();
        for (auto &c : valid_locs) dopt.grid_cells_to_search.push_back({c.first, c.second});

        dopt.use_nms = nonmaxSuppression;

        std::vector<modal_flow::DetectInput> detect_in(1);
        detect_in[0].cam_id  = cam_id;
        detect_in[0].img_buf = buf_id;
        detect_in[0].opts = dopt;

        // run the detection
        auto results = mgr.detect_many(detect_in);
        if (results.empty()) return;
        auto& det = results[0];

        // sort detections by score descending
        std::sort(det.keypoints.begin(), det.keypoints.end(),
                [](auto &a, auto &b) { return a.score > b.score; });

        // printf("in grid flow function, points; %d, grid_x: %d, grid_y; %d, num_feats_per_grid: %d\n", det.keypoints.size(), grid_x, grid_y, num_features_grid);

        std::unordered_map<size_t, int> picked_count;
        picked_count.reserve(valid_set.size());

        pts.clear();
        pts.reserve(size_t(grid_x * grid_y * num_features_grid));

        for (const auto& k : det.keypoints)
        {
            const int x = int(std::floor(k.x));
            const int y = int(std::floor(k.y));
            if (x <= 0 || x >= img_width || y <= 0 || y >= img_height) continue;

            // mask reject (255 == remove)
            if (!mask.empty() && mask.at<uint8_t>(y, x) > 127) continue;

            const int cx = x / size_x;
            const int cy = y / size_y;
            if (cx < 0 || cx >= grid_x || cy < 0 || cy >= grid_y) continue;

            const size_t cell = pack_cell(cx, cy);
            if (!valid_set.count(cell)) continue;

            int& cnt = picked_count[cell];
            if (cnt >= num_features_grid) continue;

            // accept
            cv::KeyPoint kp;
            kp.pt.x = float(x);
            kp.pt.y = float(y);

            // accept it
            pts.push_back(kp);
            ++cnt;

            // early exit
            if ((int)pts.size() >= int(valid_set.size()) * num_features_grid) break;
        }

        // Combine all the collections into our single vector
        // for (size_t r = 0; r < collection.size(); r++) {
        //     pts.insert(pts.end(), collection.at(r).begin(), collection.at(r).end());
        // }

        // Return if no points
        if (pts.empty())
            return;

        // Sub-pixel refinement parameters
        // cv::Size win_size = cv::Size(5, 5);
        // cv::Size zero_zone = cv::Size(-1, -1);
        // cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001);

        // // Get vector of points
        // std::vector<cv::Point2f> pts_refined;
        // for (size_t i = 0; i < pts.size(); i++) {
        //     pts_refined.push_back(pts.at(i).pt);
        // }

        // // Finally run sub pixel refinement (cv::cornerSubPix)
        // int res = tracker->refinePoints(static_cast<int>(pts_refined.size()),
        //                         reinterpret_cast<float*>(pts_refined.data()),
        //                         win_size.width,
        //                         term_crit.maxCount,
        //                         static_cast<float>(term_crit.epsilon));
        // if (res != 0) {
        //     std::cerr << "Subpixel refinement failed." << std::endl;
        //     return;
        // }

        // Save the refined points!
        // for (size_t i = 0; i < pts.size(); i++) {
        //     pts.at(i).pt = pts_refined.at(i);
        // }
    }
};

} // namespace ov_core

#endif /* OV_CORE_GRIDER_GRID_H */
