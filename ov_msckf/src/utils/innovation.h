/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_MSCKF_INNOVATION_H
#define OV_MSCKF_INNOVATION_H

#include <Eigen/Cholesky>
#include <limits>
#include "utils/finite.h"

namespace ov_msckf {
// A failed factorization or non-finite statistic is not evidence of a small
// innovation. Keep the same lower-triangle LLT and quadratic form as the valid
// path; storage-bit checks also work in the production -ffast-math build.
inline bool innovation_chi2(const Eigen::MatrixXd &covariance, const Eigen::VectorXd &residual,
                            double &statistic) {
  statistic = std::numeric_limits<double>::infinity();
  if (residual.rows() == 0 || covariance.rows() != residual.rows() || covariance.cols() != residual.rows() ||
      !ov_core::numeric::finite_matrix(covariance) || !ov_core::numeric::finite_matrix(residual))
    return false;
  Eigen::LLT<Eigen::MatrixXd> factor(covariance);
  if (factor.info() != Eigen::Success)
    return false;
  const Eigen::VectorXd weighted = factor.solve(residual);
  if (!ov_core::numeric::finite_matrix(weighted))
    return false;
  const double value = residual.dot(weighted);
  if (!ov_core::numeric::finite(value) || value < 0.0)
    return false;
  statistic = value;
  return true;
}

inline bool valid_innovation_limit(double limit) {
  return ov_core::numeric::finite(limit) && limit >= 0.0;
}
} // namespace ov_msckf
#endif
