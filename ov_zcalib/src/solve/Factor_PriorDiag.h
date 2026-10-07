/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: tiny diagonal prior factors (gauge anchoring / seed priors).
 * Euclidean: r = W^(1/2) (x - x0). JPL quaternion: r = W^(1/2) * 2*vec(q ⊗ q0^-1)
 * (left-error convention, matching the solver's quaternion local parameterization).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_FACTOR_PRIOR_DIAG_H
#define OV_ZCALIB_FACTOR_PRIOR_DIAG_H

#include <Eigen/Dense>

#include "ceres_free/CostFunction.h"
#include "utils/quat_ops.h"

namespace ov_zcalib {

/// Euclidean diagonal prior on an n-dof block
class Factor_PriorEuclid : public ov_init::zbft_sfm::CostFunction {
public:
  Eigen::VectorXd x0, w; ///< prior mean and per-axis 1/sigma
  Factor_PriorEuclid(const Eigen::VectorXd &x0_, const Eigen::VectorXd &sigma) : x0(x0_) {
    w = sigma.cwiseInverse();
    set_num_residuals((int)x0.size());
    mutable_parameter_block_sizes()->push_back((int)x0.size());
  }
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    const int n = (int)x0.size();
    Eigen::Map<const Eigen::VectorXd> x(parameters[0], n);
    Eigen::Map<Eigen::VectorXd> r(residuals, n);
    r = w.asDiagonal() * (x - x0);
    if (jacobians && jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[0], n, n);
      J.setZero();
      J.diagonal() = w;
    }
    return true;
  }
};

/// JPL quaternion prior (3-dof residual on a 4-ambient block, leading-identity local)
class Factor_PriorQuatJPL : public ov_init::zbft_sfm::CostFunction {
public:
  Eigen::Vector4d q0;
  Eigen::Vector3d w;
  Factor_PriorQuatJPL(const Eigen::Vector4d &q0_, const Eigen::Vector3d &sigma) : q0(q0_), w(sigma.cwiseInverse()) {
    set_num_residuals(3);
    mutable_parameter_block_sizes()->push_back(4);
  }
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    Eigen::Map<const Eigen::Vector4d> q(parameters[0]);
    const Eigen::Vector4d dq = ov_core::quat_multiply(q, ov_core::Inv(q0));
    Eigen::Map<Eigen::Vector3d> r(residuals);
    r = w.asDiagonal() * (2.0 * dq.head<3>());
    if (jacobians && jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();
      // d(2*vec(dq))/d(theta_left) = dq4*I + skew(vec(dq)) (exact to first order; FD-pinned)
      J.leftCols(3) = w.asDiagonal() * (dq(3) * Eigen::Matrix3d::Identity() + ov_core::skew_x(dq.head<3>()));
    }
    return true;
  }
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_FACTOR_PRIOR_DIAG_H
