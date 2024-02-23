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

#ifndef OV_CORE_SENSOR_DATA_H
#define OV_CORE_SENSOR_DATA_H

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ov_core {

/**
 * @brief Struct for a single imu measurement (time, wm, am)
 */
struct ImuData {

    /// Timestamp of the reading
    double timestamp;

    /// Gyroscope reading, angular velocity (rad/s)
    Eigen::Matrix<double, 3, 1> wm;

    /// Accelerometer reading, linear acceleration (m/s^2)
    Eigen::Matrix<double, 3, 1> am;

    /// Sort function to allow for using of STL containers
    bool operator<(const ImuData &other) const { return timestamp < other.timestamp; }
};

/**
 * @brief Struct for a collection of camera measurements.
 *
 * For each image we have a camera id and timestamp that it occured at.
 * If there are multiple cameras we will treat it as pair-wise stereo tracking.
 */
struct CameraData {

    /// Timestamp of the reading
    double timestamp;

    /// Camera ids for each of the images collected
    std::vector<int> sensor_ids;

    /// Raw image we have collected for each camera
    std::vector<cv::Mat> images;

    /// Tracking masks for each camera we have
    std::vector<cv::Mat> masks;

    /// Sort function to allow for using of STL containers
    bool operator<(const CameraData &other) const {
        if (timestamp == other.timestamp) {
            int id = *std::min_element(sensor_ids.begin(), sensor_ids.end());
            int id_other = *std::min_element(other.sensor_ids.begin(), other.sensor_ids.end());
            return id < id_other;
        } else {
            return timestamp < other.timestamp;
        }
    }
};

typedef struct ExtFeature {
    size_t id;
    size_t cam_id;
    float u;
    float v;
    unsigned char descriptor[32] = {0};
} __attribute((packed))__ExtFeature;

//TODO ============= DEPRECATED
typedef struct MaiFeature {
    size_t id;
    size_t cam_id;
    float x;
    float y;
    float u;
    float v;
    unsigned char descriptor[32] = {0};
} __attribute((packed))__MaiFeature;

struct ProcessedCameraData {
    /// Timestamp of the reading
    double timestamp;

    /// Camera ids for each of the images collected
    std::vector<int> sensor_ids;

    // no mask or image for now

    /// Features Extracted per cam
    std::vector<MaiFeature> feats;

    /// Sort function to allow for using of STL containers
    bool operator<(const ProcessedCameraData &other) const {
        if (timestamp == other.timestamp) {
            int id = *std::min_element(sensor_ids.begin(), sensor_ids.end());
            int id_other = *std::min_element(other.sensor_ids.begin(), other.sensor_ids.end());
            return id < id_other;
        } else {
            return timestamp < other.timestamp;
        }
    }
};
//TODO ============= DEPRECATED


} // namespace ov_core

#endif // OV_CORE_SENSOR_DATA_H