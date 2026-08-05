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

#ifndef OV_CORE_SENSOR_DATA_H
#define OV_CORE_SENSOR_DATA_H

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <vector>
#if HAVE_OPENCL
#include <CL/cl.h>
#include <modal_flow/Types.hpp>
#endif

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

#if HAVE_OPENCL
  // Device memory references for each camera
  std::vector<cl_mem> cl_images;
  std::vector<modal_flow::Frame> img_frames;
#endif

  /// Tracking masks for each camera we have
  std::vector<cv::Mat> masks;

  /// Exposure time [s] of each image (0 when the source does not publish one).
  ///
  /// Per-FRAME, not per-session: auto-exposure moves during a run, and a consumer that stamps at
  /// mid-exposure (t + exposure/2) needs the exposure of THIS frame. It rides on the message
  /// because the thread holding the driver metadata is not, in general, the thread that consumes
  /// the image -- an async ingest hands frames to a different consumer entirely.
  ///
  /// *** INERT TO THE ESTIMATOR -- DO NOT CONSUME THIS IN VIO. ***
  /// The filter's time base is `timestamp` AS DELIVERED (start of exposure), and the calibrated
  /// `calib_camimu_dt` it runs with is defined in exactly that convention. Shifting camera times
  /// by exposure/2 anywhere in the estimator would DOUBLE-COUNT the offset the calibrator already
  /// folded into td, silently biasing every clone. The calibrator (ov_zcalib) stamps its own clones
  /// at mid-exposure internally and converts back to start-of-exposure before it writes td out --
  /// that conversion is the ONLY place this field belongs. VIO must ignore it.
  std::vector<float> exposures;

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

} // namespace ov_core

#endif // OV_CORE_SENSOR_DATA_H