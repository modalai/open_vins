/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_RAW_IMU_CPI_H
#define OV_INIT_RAW_IMU_CPI_H

#include <memory>
#include <vector>
#include "cpi/CpiV1.h"
#include "utils/sensor_data.h"

namespace ov_init {

// Completed CPI snapshot in the physical IMU frame with RAW sensor biases.
// All means/Jacobians/P_meas are ready for a raw-bias graph factor. Do not feed
// additional samples after completion: CpiV1's internal integration coordinates
// are corrected-sensor biases, whereas this exported snapshot uses raw biases.
struct RawBiasCpiV1 : ov_core::CpiV1 {
  using ov_core::CpiV1::CpiV1;
  Eigen::Matrix3d H_q = Eigen::Matrix3d::Zero(); // rotation correction wrt raw ba
};

// Fixed calibrated IMU model:
// a = A*(am-ba), w = G*(wm-bg-Tg*a), with raw independent white/RW noises.
// The optimizer, reset priors and injected state stay in raw bias coordinates.
class RawImuCpiModel {
public:
  bool set_calibration(const Eigen::Matrix3d &A, const Eigen::Matrix3d &G, const Eigen::Matrix3d &Tg);

  bool correct(const Eigen::Vector3d &wm, const Eigen::Vector3d &am,
               const Eigen::Vector3d &bg, const Eigen::Vector3d &ba,
               Eigen::Vector3d &w, Eigen::Vector3d &a) const;

  // sigma order: gyro white, gyro RW, accel white, accel RW. Both full white
  // and RW covariance blocks are transported, including their off-diagonals.
  // The output is replaced only on success. Identity calibration follows the
  // original arithmetic path to preserve its exact means/covariance bytes.
  bool preintegrate(const std::vector<ov_core::ImuData> &readings,
                    const Eigen::Vector3d &bg, const Eigen::Vector3d &ba,
                    const Eigen::Vector4d &sigma, std::shared_ptr<RawBiasCpiV1> &output) const;

private:
  bool ready_ = false, identity_ = true;
  Eigen::Matrix3d A_ = Eigen::Matrix3d::Identity(), G_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d Tg_ = Eigen::Matrix3d::Zero(), C_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix<double, 12, 12> corrected_noise_from_raw_ = Eigen::Matrix<double,12,12>::Identity();
  Eigen::Matrix<double, 15, 15> raw_error_from_corrected_ = Eigen::Matrix<double,15,15>::Identity();
};
} // namespace ov_init
#endif
