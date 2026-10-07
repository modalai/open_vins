/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_CONDITIONAL_BIAS_PRIOR_H
#define OV_INIT_CONDITIONAL_BIAS_PRIOR_H

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include "utils/finite.h"

namespace ov_init {
namespace conditional_bias {

using Matrix6 = Eigen::Matrix<double,6,6>;
using Vector6 = Eigen::Matrix<double,6,1>;

// Rank decisions use correlation coordinates. Raw clock, pixel and extrinsic
// units must not decide which stochastic calibration directions exist.
inline bool normalized_covariance(const Eigen::MatrixXd &P,Eigen::MatrixXd &correlation,Eigen::VectorXd &scales) {
  if(P.rows()!=P.cols() || !ov_core::numeric::finite_matrix(P)) return false;
  const int n=P.rows();
  scales=Eigen::VectorXd::Ones(n);
  for(int i=0;i<n;++i) {
    if(P(i,i)<0.) return false;
    if(P(i,i)==0.) {
      // A deterministic variable has no covariance with a stochastic one.
      if(!P.row(i).isZero(0.) || !P.col(i).isZero(0.)) return false;
    } else scales(i)=std::sqrt(P(i,i));
  }
  correlation.resize(n,n);
  for(int r=0;r<n;++r) for(int c=0;c<n;++c) correlation(r,c)=P(r,c)/scales(r)/scales(c);
  if(!ov_core::numeric::finite_matrix(correlation)) return false;
  const double roundoff=128.*std::numeric_limits<double>::epsilon()*std::max(1,n);
  if(n && (correlation-correlation.transpose()).cwiseAbs().maxCoeff()>roundoff) return false;
  correlation=(.5*(correlation+correlation.transpose())).eval();
  return true;
}

inline bool valid_joint(const Matrix6 &Pbb,const Eigen::MatrixXd &Pbc,const Eigen::MatrixXd &Pcc) {
  const int k=Pcc.rows();
  if(Pbc.rows()!=6 || Pbc.cols()!=k || Pcc.cols()!=k) return false;
  Eigen::MatrixXd joint(6+k,6+k);
  joint.topLeftCorner<6,6>()=Pbb;joint.topRightCorner(6,k)=Pbc;
  joint.bottomLeftCorner(k,6)=Pbc.transpose();joint.bottomRightCorner(k,k)=Pcc;
  Eigen::MatrixXd R;Eigen::VectorXd scales;
  if(!normalized_covariance(joint,R,scales)) return false;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(R,Eigen::EigenvaluesOnly);
  const double roundoff=128.*std::numeric_limits<double>::epsilon()*(6+k);
  if(eigen.info()!=Eigen::Success || !ov_core::numeric::finite_matrix(eigen.eigenvalues()) ||
      eigen.eigenvalues().minCoeff() < -roundoff) return false;
  // PSD tolerance alone can hide a tiny, but first-order, range defect: its
  // negative joint eigenvalue is quadratic in the invalid cross covariance.
  // Check supported calibration directions explicitly before selecting reset.
  if(k) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> calibration(R.bottomRightCorner(k,k));
    if(calibration.info()!=Eigen::Success || !ov_core::numeric::finite_matrix(calibration.eigenvalues()) ||
        !ov_core::numeric::finite_matrix(calibration.eigenvectors()))return false;
    const double tolerance=128.*std::numeric_limits<double>::epsilon()*k;
    if(calibration.eigenvalues().minCoeff() < -tolerance)return false;
    Eigen::MatrixXd projector=Eigen::MatrixXd::Zero(k,k);
    for(int i=0;i<k;++i)if(calibration.eigenvalues()(i)>tolerance)
      projector.noalias()+=calibration.eigenvectors().col(i)*calibration.eigenvectors().col(i).transpose();
    const Eigen::MatrixXd cross=R.topRightCorner(6,k),outside=cross-cross*projector;
    if(!ov_core::numeric::finite_matrix(outside) || outside.cwiseAbs().maxCoeff()>tolerance*std::max(1.,cross.norm()))return false;
  }
  return true;
}

struct Conditioned {
  Matrix6 bias_covariance=Matrix6::Zero();
  Eigen::MatrixXd bias_calibration_covariance;
  Matrix6 covariance=Matrix6::Zero();
  Eigen::MatrixXd regression; // E[db | dc] = regression * dc, on prior support
  Matrix6 sqrt_information=Matrix6::Zero();
  int calibration_rank=0;
};

// The covariance and regression describe the first NEW graph node, not its
// newest node. Gap random walk, declared divergence inflation and sigma floors
// are explicit model operations. No jitter is used in either factorization.
inline bool condition(const Matrix6 &Pbb,const Eigen::MatrixXd &Pbc,const Eigen::MatrixXd &Pcc,
                      double gap,const Vector6 &rw_variance,double bias_inflation,
                      const Vector6 &sigma_floor,Conditioned &output) {
  using ov_core::numeric::finite_matrix;
  const Eigen::Vector2d scalars(gap,bias_inflation);
  if(!finite_matrix(scalars) || gap<0. || bias_inflation<1. || !finite_matrix(rw_variance) ||
      (rw_variance.array()<0.).any() || !finite_matrix(sigma_floor) || (sigma_floor.array()<0.).any() ||
      !valid_joint(Pbb,Pbc,Pcc)) return false;
  const int k=Pcc.rows();
  Conditioned next;
  next.bias_covariance=Pbb;
  next.bias_covariance.diagonal()+=gap*rw_variance;
  next.bias_covariance*=bias_inflation*bias_inflation;
  next.bias_calibration_covariance=bias_inflation*Pbc;
  for(int i=0;i<6;++i) next.bias_covariance(i,i)+=std::max(0.,sigma_floor(i)*sigma_floor(i)-next.bias_covariance(i,i));
  Eigen::MatrixXd R,Rb;Eigen::VectorXd scales,scales_b;
  if(!normalized_covariance(Pcc,R,scales) || !normalized_covariance(next.bias_covariance,Rb,scales_b)) return false;
  Eigen::MatrixXd U=next.bias_calibration_covariance;
  for(int r=0;r<6;++r) for(int c=0;c<k;++c) U(r,c)=U(r,c)/scales_b(r)/scales(c);
  if(!finite_matrix(U)) return false;
  Eigen::MatrixXd inverse=Eigen::MatrixXd::Zero(k,k),projector=Eigen::MatrixXd::Zero(k,k);
  const double rank_roundoff=128.*std::numeric_limits<double>::epsilon()*std::max(1,k);
  if(k) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(R);
    if(eigen.info()!=Eigen::Success || !finite_matrix(eigen.eigenvalues()) || !finite_matrix(eigen.eigenvectors()) ||
        eigen.eigenvalues().minCoeff() < -rank_roundoff) return false;
    for(int i=0;i<k;++i) if(eigen.eigenvalues()(i)>rank_roundoff) {
      const Eigen::VectorXd direction=eigen.eigenvectors().col(i);
      inverse.noalias()+=direction*direction.transpose()/eigen.eigenvalues()(i);
      projector.noalias()+=direction*direction.transpose();++next.calibration_rank;
    }
    const Eigen::MatrixXd outside=U-U*projector;
    if(!finite_matrix(outside) || outside.cwiseAbs().maxCoeff()>rank_roundoff*std::max(1.,U.norm())) return false;
  }
  const Eigen::MatrixXd normalized_regression=U*inverse;
  Eigen::MatrixXd Q=Rb-normalized_regression*U.transpose();
  Q=(.5*(Q+Q.transpose())).eval();
  if(!finite_matrix(Q)) return false;
  // No infinite-information deterministic bias factor is represented by this
  // bounded API. A singular conditional bias prior requires another contract.
  Eigen::SelfAdjointEigenSolver<Matrix6> spectrum(Q,Eigen::EigenvaluesOnly);
  if(spectrum.info()!=Eigen::Success || !finite_matrix(spectrum.eigenvalues()) ||
      spectrum.eigenvalues().minCoeff()<=128.*std::numeric_limits<double>::epsilon()*6.) return false;
  Eigen::LLT<Matrix6> factor(Q);
  if(factor.info()!=Eigen::Success) return false;
  next.covariance=scales_b.asDiagonal()*Q*scales_b.asDiagonal();
  next.regression=scales_b.asDiagonal()*normalized_regression;
  for(int c=0;c<k;++c) next.regression.col(c)/=scales(c);
  next.sqrt_information=factor.matrixL().solve(Matrix6::Identity())*scales_b.cwiseInverse().asDiagonal();
  if(!finite_matrix(next.covariance) || !finite_matrix(next.regression) || !finite_matrix(next.sqrt_information)) return false;
  output=std::move(next);
  return true;
}

} // namespace conditional_bias
} // namespace ov_init
#endif
