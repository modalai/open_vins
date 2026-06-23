/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * Ceres-free initialization backend (ov_init::zbft_sfm)
 * ------------------------------------------------
 * Lifted from ov_init/src/ceres/Factor_GenericPrior.{h,cpp}; residual and
 * analytic Jacobians verbatim. Base class swapped to zbft_sfm::CostFunction. This is
 * the mechanism for the gauge prior (quat_yaw + first-position), bias priors,
 * and any soft prior (gravity tilt, velocity) used by the reset initializer.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_ZBFT_SFM_FACTOR_GENERICPRIOR_H
#define OV_INIT_ZBFT_SFM_FACTOR_GENERICPRIOR_H

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "CostFunction.h"

namespace ov_init {
namespace zbft_sfm {

/**
 * @brief Factor for a generic Gaussian prior on a set of states.
 *
 * cost = sqrtI * (x [-] x_lin) + b   with   b = sqrtI^{-T} * prior_grad.
 * Supported per-block types: "quat" (3-dof), "quat_yaw" (1-dof, z error only),
 * "vec1", "vec3", "vec8". The parameter blocks are passed in the same order as
 * x_type, with ambient sizes 4/4/1/3/8 respectively.
 */
class Factor_GenericPrior : public CostFunction {
public:
  /// State linearization point (stacked ambient values, column vector)
  Eigen::MatrixXd x_lin;

  /// Type of each variable ("quat", "quat_yaw", "vec1", "vec3", "vec8")
  std::vector<std::string> x_type;

  /// Square-root information and constant offset (precomputed in the constructor)
  Eigen::MatrixXd sqrtI;
  Eigen::MatrixXd b;

  Factor_GenericPrior(const Eigen::MatrixXd &x_lin_, const std::vector<std::string> &x_type_, const Eigen::MatrixXd &prior_Info,
                      const Eigen::MatrixXd &prior_grad);

  virtual ~Factor_GenericPrior() {}

  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;
};

} // namespace zbft_sfm
} // namespace ov_init

#endif // OV_INIT_ZBFT_SFM_FACTOR_GENERICPRIOR_H
