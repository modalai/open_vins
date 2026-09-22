/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_MSCKF_LEGACY_EXPOSURE_H
#define OV_MSCKF_LEGACY_EXPOSURE_H

#include "state/State.h"
#include "utils/quat_ops.h"

namespace ov_msckf {
namespace legacy_exposure {

// This bounded deterministic model applies only to a fixed zero-readout camera
// without a preintegration bridge. An estimated readout remains outside this
// model even when its current estimate is zero.
inline bool uses_body_velocity(const State &state, size_t camera, bool has_bridge) {
  const auto &readout = state._calib_camera_readout.at(camera);
  return !state.uses_physical_clones() && !has_bridge && readout->id() < 0 && readout->value()(0) == 0.0;
}

// The cache remains world-valued for its other consumers. Its exposure-model
// coordinate is the fixed vector u = R_FEJ * v_cache, expressed in the clone's
// frozen FEJ chart. Thus v(R) = R^T u transforms with the estimated world frame.
// This does not retain independent velocity, bias, or process-noise uncertainty.
inline Eigen::Vector3d world_velocity(const Eigen::Matrix3d &R, const Eigen::Matrix3d &R_fej,
                                      const Eigen::Vector3d &cached_velocity) {
  return R.transpose() * (R_fej * cached_velocity);
}

inline void warp_pose(Eigen::Matrix3d &R, Eigen::Vector3d &p, const Eigen::Matrix3d &R_fej,
                      const Eigen::Vector3d &cached_velocity, const Eigen::Vector3d &omega, double dt) {
  p += world_velocity(R, R_fej, cached_velocity) * dt;
  R = ov_core::exp_so3(-omega * dt) * R;
}

} // namespace legacy_exposure
} // namespace ov_msckf
#endif
