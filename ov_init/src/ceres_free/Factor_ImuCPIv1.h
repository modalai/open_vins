/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * Ceres-free initialization backend (ov_init::zbft_sfm)
 * ------------------------------------------------
 * Lifted from ov_init/src/ceres/Factor_ImuCPIv1.{h,cpp}. The residual and the
 * analytic Jacobians are IDENTICAL to the proven upstream factor; only the base
 * class (ceres::CostFunction -> zbft_sfm::CostFunction) and the include changed.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_ZBFT_SFM_FACTOR_IMUCPIV1_H
#define OV_INIT_ZBFT_SFM_FACTOR_IMUCPIV1_H

#include <Eigen/Dense>

#include "CostFunction.h"

namespace ov_init {
namespace zbft_sfm {

/**
 * @brief Factor for IMU continuous preintegration version 1 (CPI v1).
 *
 * Parameter blocks (canonical VINS clone order), 15 residuals:
 *   [0] q_GtoI1 (4)  [1] bg_1 (3)  [2] v_I1inG (3)  [3] ba_1 (3)  [4] p_I1inG (3)
 *   [5] q_GtoI2 (4)  [6] bg_2 (3)  [7] v_I2inG (3)  [8] ba_2 (3)  [9] p_I2inG (3)
 */
class Factor_ImuCPIv1 : public CostFunction {
public:
  // Preintegrated measurements and time interval
  Eigen::Vector3d alpha;
  Eigen::Vector3d beta;
  Eigen::Vector4d q_breve;
  double dt;

  // Preintegration linearization points
  Eigen::Vector3d b_w_lin_save;
  Eigen::Vector3d b_a_lin_save;

  // Preintegrated bias jacobians
  Eigen::Matrix3d J_q; // orientation wrt bias w
  Eigen::Matrix3d J_a; // position wrt bias w
  Eigen::Matrix3d J_b; // velocity wrt bias w
  Eigen::Matrix3d H_a; // position wrt bias a
  Eigen::Matrix3d H_b; // velocity wrt bias a

  // Sqrt of the preintegration information
  Eigen::Matrix<double, 15, 15> sqrtI_save;

  // Gravity
  Eigen::Vector3d grav_save;

  Factor_ImuCPIv1(double deltatime, Eigen::Vector3d &grav, Eigen::Vector3d &alpha, Eigen::Vector3d &beta, Eigen::Vector4d &q_KtoK1,
                  Eigen::Vector3d &ba_lin, Eigen::Vector3d &bg_lin, Eigen::Matrix3d &J_q, Eigen::Matrix3d &J_beta, Eigen::Matrix3d &J_alpha,
                  Eigen::Matrix3d &H_beta, Eigen::Matrix3d &H_alpha, Eigen::Matrix<double, 15, 15> &covariance);

  virtual ~Factor_ImuCPIv1() {}

  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;
};

} // namespace zbft_sfm
} // namespace ov_init

#endif // OV_INIT_ZBFT_SFM_FACTOR_IMUCPIV1_H
