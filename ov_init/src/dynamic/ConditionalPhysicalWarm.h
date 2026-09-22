/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_CONDITIONAL_PHYSICAL_WARM_H
#define OV_INIT_CONDITIONAL_PHYSICAL_WARM_H

#include <Eigen/Eigenvalues>
#include <limits>
#include "GravityAlignment.h"

namespace ov_init {
namespace conditional_warm {

inline bool valid_psd(const Eigen::MatrixXd &P) {
  if (P.rows() != P.cols() || !gravity_export::finite(P)) return false;
  if (!P.rows()) return true;
  const double scale = P.cwiseAbs().maxCoeff();
  const double tolerance = 128. * std::numeric_limits<double>::epsilon() * P.rows() * scale;
  if ((P-P.transpose()).cwiseAbs().maxCoeff() > tolerance) return false;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(.5*(P+P.transpose()), Eigen::EigenvaluesOnly);
  return eigen.info() == Eigen::Success && gravity_export::finite(eigen.eigenvalues()) &&
         eigen.eigenvalues().minCoeff() >= -tolerance;
}

// Q and S use UNIQUE solved-block rows. C expands/re-aligns those rows (including
// gravity); D contains owner-only direct clock derivatives in the output chart.
// No calibration likelihood or bridge/process noise is added here. This is the
// error covariance of the existing conditional estimator, not a fitted posterior.
inline bool assemble(const Eigen::MatrixXd &Q, const Eigen::MatrixXd &S, const Eigen::MatrixXd &C,
                     const Eigen::MatrixXd &D, const Eigen::MatrixXd &Pc, const Eigen::VectorXd &inflation,
                     Eigen::MatrixXd &output) {
  const int n = C.rows(), k = Pc.rows();
  if (n <= 0 || Q.rows() != Q.cols() || C.cols() != Q.rows() || S.rows() != Q.rows() || S.cols() != k ||
      D.rows() != n || D.cols() != k || inflation.size() != n || !gravity_export::finite(Q) ||
      !gravity_export::finite(S) || !gravity_export::finite(C) || !gravity_export::finite(D) ||
      !gravity_export::finite(inflation) || (inflation.array() <= 0.).any() || !valid_psd(Pc)) return false;
  const Eigen::MatrixXd V = inflation.asDiagonal() * (C*S+D);
  const Eigen::MatrixXd L = inflation.asDiagonal()*C;
  Eigen::MatrixXd next = Eigen::MatrixXd::Zero(n+k,n+k);
  next.topLeftCorner(n,n).noalias() = L*Q*L.transpose();
  next.topLeftCorner(n,n).noalias() += V*Pc*V.transpose();
  next.topRightCorner(n,k).noalias() = V*Pc;
  next.bottomLeftCorner(k,n) = next.topRightCorner(n,k).transpose();
  next.topLeftCorner(n,n) = .5*(next.topLeftCorner(n,n)+next.topLeftCorner(n,n).transpose()).eval();
  next.bottomRightCorner(k,k) = Pc; // retain every original calibration covariance entry
  if (!valid_psd(next)) return false;
  output = std::move(next);
  return true;
}

} // namespace conditional_warm
} // namespace ov_init
#endif
