/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * Ceres-free initialization backend (ov_init::zbft_sfm)
 * ------------------------------------------------
 * Lifted from ov_init/src/ceres/Factor_ImageReprojCalib.{h,cpp}; residual and
 * analytic Jacobians verbatim. Base class swapped to zbft_sfm::CostFunction.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_ZBFT_SFM_FACTOR_IMAGEREPROJCALIB_H
#define OV_INIT_ZBFT_SFM_FACTOR_IMAGEREPROJCALIB_H

#include <Eigen/Dense>

#include "CostFunction.h"
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "utils/quat_ops.h"

namespace ov_init {
namespace zbft_sfm {

/**
 * @brief Feature bearing (raw pixel) reprojection factor with calibration.
 *
 * Parameter blocks (2 residuals):
 *   [0] q_GtoIi (4)  [1] p_IiinG (3)  [2] p_FinG (3)  [3] q_ItoC (4)  [4] p_IinC (3)  [5] cam intrinsics (8)
 * Typically blocks [3..5] are held constant (calibration fixed) during a reset.
 */
class Factor_ImageReprojCalib : public CostFunction {
public:
  // Measurement observation of the feature (raw pixel coordinates)
  Eigen::Vector2d uv_meas;

  // Measurement noise
  double pix_sigma = 1.0;
  Eigen::Matrix<double, 2, 2> sqrtQ;

  // If distortion model is fisheye or radtan
  bool is_fisheye = false;

  // If value of 1 then this residual adds to the problem, otherwise if zero it is "gated"
  double gate = 1.0;

  Factor_ImageReprojCalib(const Eigen::Vector2d &uv_meas_, double pix_sigma_, bool is_fisheye_);

  virtual ~Factor_ImageReprojCalib() {}

  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;
};

} // namespace zbft_sfm
} // namespace ov_init

#endif // OV_INIT_ZBFT_SFM_FACTOR_IMAGEREPROJCALIB_H
