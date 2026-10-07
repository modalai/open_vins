#ifndef OV_ZCALIB_POSTERIOR_CHECKS_H
#define OV_ZCALIB_POSTERIOR_CHECKS_H

#include <Eigen/Dense>
#include "utils/NumericChecks.h"

namespace ov_zcalib {

// The accepted information includes proper calibration priors, so it must be
// positive definite. LDLT success alone also accepts indefinite matrices.
// Clamping a negative inverse diagonal to zero would report false certainty.
inline bool posterior_sigmas(const Eigen::MatrixXd &information, Eigen::VectorXd &sigma) {
  sigma.resize(0);
  const Eigen::Index n = information.rows();
  if (n == 0 || information.cols() != n)
    return false;
  for (Eigen::Index k = 0; k < information.size(); ++k)
    if (!finite_scalar(information.data()[k]))
      return false;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(information);
  if (ldlt.info() != Eigen::Success)
    return false;
  for (Eigen::Index k = 0; k < n; ++k)
    if (!finite_scalar(ldlt.vectorD()(k)) || !(ldlt.vectorD()(k) > 0.0))
      return false;
  const Eigen::MatrixXd covariance = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
  if (ldlt.info() != Eigen::Success)
    return false;
  for (Eigen::Index k = 0; k < covariance.size(); ++k)
    if (!finite_scalar(covariance.data()[k]))
      return false;
  for (Eigen::Index k = 0; k < n; ++k)
    if (!(covariance(k, k) > 0.0))
      return false;
  sigma = covariance.diagonal().cwiseSqrt();
  return true;
}

} // namespace ov_zcalib
#endif
