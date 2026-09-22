/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_PHYSICAL_RESET_WINDOW_H
#define OV_INIT_PHYSICAL_RESET_WINDOW_H

#include "ConditionalBiasPrior.h"
#include "utils/InitializerPhysicalWarmResult.h"
#include "utils/sensor_data.h"
#include <algorithm>
#include <set>

namespace ov_init {

inline bool valid_physical_reset_prior(const ov_core::InitPhysicalResetPrior &prior) {
  using ov_core::numeric::finite;
  using ov_core::numeric::finite_matrix;
  if(!prior.snapshot_id || !finite(prior.imu_endpoint) || !finite(prior.imu_raw_cutoff) ||
      prior.imu_raw_cutoff<prior.imu_endpoint || prior.calibration.empty() ||
      prior.raw_watermarks.size()!=prior.calibration.size() || prior.reference_camera_id>=prior.calibration.size() ||
      (prior.cause!=0 && prior.cause!=1) || !finite_matrix(prior.bias_mean) || !finite_matrix(prior.bias_rw_variance) ||
      (prior.bias_rw_variance.array()<0.).any() || !finite_matrix(prior.imu_accel_map) ||
      !finite_matrix(prior.imu_gyro_map) || !finite_matrix(prior.imu_tg)) return false;
  for(size_t i=0;i<prior.calibration.size();++i) {
    const auto &camera=prior.calibration[i];
    if(camera.camera_id!=i || !finite(camera.clock_mean) || !finite_matrix(camera.extrinsics) ||
        !finite_matrix(camera.intrinsics) || std::abs(camera.extrinsics.head<4>().norm()-1.)>1e-10 ||
        camera.width<=0 || camera.height<=0 ||
        (!finite(prior.raw_watermarks[i]) && prior.raw_watermarks[i]!=-std::numeric_limits<double>::infinity())) return false;
  }
  int dimension=0;std::set<std::pair<size_t,ov_core::InitCameraCalibrationKind>> keys;
  for(const auto &block:prior.consider) {
    if(block.camera_id>=prior.calibration.size() || !block.local_size() || block.mean.size()!=block.value_size() ||
        block.fej.size()!=block.value_size() || !finite_matrix(block.mean) || !finite_matrix(block.fej) ||
        !(block.mean.array()==block.fej.array()).all() || !keys.emplace(block.camera_id,block.kind).second) return false;
    const auto &camera=prior.calibration[block.camera_id];Eigen::VectorXd mean;
    if(block.kind==ov_core::InitCameraCalibrationKind::Clock)mean=Eigen::VectorXd::Constant(1,camera.clock_mean);
    else if(block.kind==ov_core::InitCameraCalibrationKind::Extrinsics)mean=camera.extrinsics;
    else mean=camera.intrinsics;
    if(!(mean.array()==block.mean.array()).all())return false;
    dimension+=block.local_size();
  }
  return prior.calibration_covariance.rows()==dimension &&
      conditional_bias::valid_joint(prior.bias_covariance,prior.bias_calibration_covariance,prior.calibration_covariance);
}

// Recover the support actually needed at the accepted endpoint from the raw
// retained stream. A newer buffered horizon is not evidence for this boundary.
// Missing bracketing provenance declines joint mode before reset is committed.
inline bool physical_reset_raw_cutoff(const std::vector<ov_core::ImuData> &samples,double endpoint,double &cutoff) {
  if(!ov_core::numeric::finite(endpoint) || samples.empty())return false;
  for(size_t i=0;i<samples.size();++i)
    if(!ov_core::numeric::finite(samples[i].timestamp) ||
        (i && !(samples[i].timestamp>samples[i-1].timestamp)))return false;
  const auto right=std::lower_bound(samples.begin(),samples.end(),endpoint,
                                   [](const ov_core::ImuData &sample,double t){return sample.timestamp<t;});
  if(right==samples.end() || (right==samples.begin() && right->timestamp!=endpoint) ||
      !ov_core::numeric::finite_matrix(right->wm) || !ov_core::numeric::finite_matrix(right->am)) return false;
  if(right->timestamp!=endpoint && (!ov_core::numeric::finite_matrix(std::prev(right)->wm) ||
      !ov_core::numeric::finite_matrix(std::prev(right)->am)))return false;
  cutoff=right->timestamp;return true;
}

// The enclosing snapshot is validated once per attempt, not once per pixel.
inline bool physical_reset_future_row(const ov_core::InitPhysicalResetPrior &prior,size_t camera,double raw) {
  return camera<prior.calibration.size() && camera<prior.raw_watermarks.size() && ov_core::numeric::finite(raw) &&
      ov_core::numeric::finite(raw+prior.calibration[camera].clock_mean) && raw>prior.raw_watermarks[camera] &&
      raw+prior.calibration[camera].clock_mean>prior.imu_endpoint;
}

} // namespace ov_init
#endif
