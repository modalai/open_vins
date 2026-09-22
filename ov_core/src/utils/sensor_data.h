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
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <memory>
#include <utility>
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

/// Stable feature id and original distorted pixel coordinates. Normalization is performed
/// by TrackSIM using the camera calibration active when this observation is consumed.
using FeatureObservations = std::vector<std::pair<size_t, Eigen::VectorXf>>;

/**
 * @brief Struct for a collection of camera measurements.
 *
 * For each image we have a camera id and timestamp that it occured at.
 * Multiple cameras may use independent mono tracking or stereo association.
 */
struct CameraData {

  /// Timestamp of the reading
  double timestamp;

  /// Optional hardware exposure-start stamp used ONLY to associate forced-sync views.
  /// Producers retain each view's own timestamp above. The synchronized consumer
  /// assigns the reference view's timestamp to the complete group. -1 means the
  /// source has no separate trigger stamp (already-synchronized messages still work).
  int64_t sync_timestamp_ns = -1;

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
  /// Per-FRAME, not per-session: auto-exposure moves during a run. It rides on the message
  /// because the thread holding the driver metadata is not, in general, the thread that consumes
  /// the image -- an async ingest hands frames to a different consumer entirely.
  ///
  /// *** PROVENANCE ONLY -- DO NOT APPLY ANOTHER EXPOSURE SHIFT DOWNSTREAM. ***
  /// The PRODUCER already applied it: `timestamp` is the frame's center-row mid-exposure instant
  /// (HAL3 start-of-exposure + (readout + exposure)/2, stamped at ingest), the calibrated
  /// `calib_camimu_dt` is defined against that convention, and per-row rolling-shutter time is
  /// the CENTERED deviation (v/h - 0.5) * t_readout around it. Shifting camera times by any
  /// exposure or readout term downstream DOUBLE-COUNTS what the stamp already contains, silently
  /// biasing every clone. Forced-sync VIO instead uses the reference view's producer
  /// timestamp for the entire group, deliberately approximating coincident exposures.
  /// Each view's original exposure remains here for diagnostics and session evidence.
  std::vector<float> exposures;

  /// Optional observation-replay payload, parallel to sensor_ids (empty for ordinary images).
  /// One immutable vector is owned per camera frame. Buffer split/bundle/drop operations share
  /// or move this owner, never copy feature vectors; the queue bounds the number of live frames.
  /// Replay images/masks contain aligned empty cv::Mat headers, not fabricated sensor images.
  std::vector<std::shared_ptr<const FeatureObservations>> observations;

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
