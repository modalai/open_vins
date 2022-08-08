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

#ifndef OV_CORE_TRACK_CVP_H
#define OV_CORE_TRACK_CVP_H

#include "TrackBase.h"
#include <modalcv.h>

namespace ov_core {

/**
 * @brief Descriptor-based visual tracking
 *
 * Here we use descriptor matching to track features from one frame to the next.
 * We track both temporally, and across stereo pairs to get stereo constraints.
 * Right now we use ORB descriptors as we have found it is the fastest when computing descriptors.
 * Tracks are then rejected based on a ratio test and ransac.
 */
class TrackCVP : public TrackBase {

  public:
    /**
     * @brief Public constructor with configuration variables
     * @param cameras camera calibration object which has all camera intrinsics in it
     * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
     * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
     * @param stereo if we should do stereo feature tracking or binocular
     * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
     * @param fast_threshold FAST detection threshold
     * @param gridx size of grid in the x-direction / u-direction
     * @param gridy size of grid in the y-direction / v-direction
     * @param minpxdist features need to be at least this number pixels away from each other
     * @param knnratio matching ratio needed (smaller value forces top two descriptors during match to be more different)
     */
    explicit TrackCVP(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                             HistogramMethod histmethod, int fast_threshold, int gridx, int gridy, int minpxdist, std::vector<mcv_fpx_feature_t*> *mcv_features_in)
        : TrackBase(cameras, numfeats, numaruco, stereo, histmethod), threshold(fast_threshold), grid_x(gridx), grid_y(gridy),
          min_px_dist(minpxdist),  mcv_features(*mcv_features_in){
            prev_positions.resize(cameras.size());
            for (size_t i = 0; i < prev_positions.size(); i++){
                // init our positions vector
                prev_positions[i] = (mcv_dcm_pos_t*)malloc(sizeof(mcv_dcm_pos_t) * MCV_MAX_POS_BUF_SIZE);
            }
          }


    void feed_new_camera(const CameraData &message) override;

  protected:

    void feed_orb_monocular(const CameraData &message, size_t msg_id);

    void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);


    void perform_orb_detection_monocular(const cv::Mat &img0, const cv::Mat &mask0, std::vector<size_t> &ids0, std::vector<cv::KeyPoint> &pts0,
                                     bool reference_set_empty, size_t cam_id, mcv_dcm_pos_t* positions);


    void perform_detection_stereo(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &mask0, const cv::Mat &mask1,
                                  std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1, cv::Mat &desc0, cv::Mat &desc1,
                                  size_t cam_id0, size_t cam_id1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);


    void robust_orb_match(const cv::Mat &img1, mcv_dcm_pos_t* positions1, size_t n_pos1,
                      const cv::Mat &img0, mcv_dcm_pos_t* positions0, size_t n_pos0,
                      size_t id0, size_t id1, std::vector<cv::DMatch> &matches);

    /////
    void feed_KLT_monocular(const CameraData &message, size_t msg_id);
    void perform_KLT_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                           std::vector<size_t> &ids0, size_t msg_id);
    void perform_KLT_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                        std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out);
    // Timing variables
    boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

    // Parameters for our FAST grid detector
    int threshold;
    int grid_x;
    int grid_y;

    // Minimum pixel distance to be "far away enough" to be a different extracted feature
    int min_px_dist;
    
    // vector of mcv features, pre-allocated before being assigned
    std::vector<mcv_fpx_feature_t*> mcv_features;

    // vector of previous positions, pre-allocated before assigned
    std::vector<mcv_dcm_pos_t*> prev_positions;

    // How many pyramid levels to track
    int pyr_levels = 3;
    cv::Size win_size = cv::Size(20, 20);

    // Last set of image pyramids
    std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;
    std::map<size_t, cv::Mat> img_curr;
    std::map<size_t, std::vector<cv::Mat>> img_pyramid_curr;


};

} // namespace ov_core

#endif /* OV_CORE_TRACK_CVP_H */
