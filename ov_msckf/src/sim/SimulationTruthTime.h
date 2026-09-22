/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_MSCKF_SIMULATION_TRUTH_TIME_H
#define OV_MSCKF_SIMULATION_TRUTH_TIME_H

#include <vector>
#include "state/State.h"

namespace ov_msckf {

/// The estimated IMU pose belongs to the returned state key in the reference
/// camera clock. Use the TRUE reference offset for a simulation oracle, never
/// the observing camera's raw frame stamp/offset or the estimated calibration.
inline double simulation_truth_time(const State &state, const std::vector<double> &true_camera_offsets) {
  if (state.uses_physical_clones())
    return state.imu_endpoint();
  return state._timestamp + true_camera_offsets.at((size_t)state.cam_imu_dt_ref_camid());
}

} // namespace ov_msckf
#endif // OV_MSCKF_SIMULATION_TRUTH_TIME_H
