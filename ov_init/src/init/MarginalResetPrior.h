/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_MARGINAL_RESET_PRIOR_H
#define OV_INIT_MARGINAL_RESET_PRIOR_H
#include "ResetPrior.h"
#include "ConditionalBiasPrior.h"

namespace ov_init {
inline bool valid_filter_reset_prior(const ResetFilterPrior &p,size_t cameras) {
  using ov_core::numeric::finite;
  if(!finite(p.imu_endpoint) || !finite(p.imu_raw_cutoff) || p.imu_raw_cutoff<p.imu_endpoint ||
      !cameras || p.raw_watermarks.size()!=cameras ||
      !conditional_bias::valid_joint(p.covariance,Eigen::MatrixXd(6,0),Eigen::MatrixXd(0,0)))return false;
  for(double raw:p.raw_watermarks)
    if(!finite(raw) && raw!=-std::numeric_limits<double>::infinity())return false;
  return true;
}

inline bool marginal_reset_future_row(const ResetFilterPrior &p,size_t camera,double raw,double offset) {
  using ov_core::numeric::finite;
  return camera<p.raw_watermarks.size() && finite(raw) && finite(offset) && finite(raw+offset) &&
      raw>p.raw_watermarks[camera] && raw+offset>p.imu_endpoint;
}

// The prior factor belongs to the FIRST graph node. CPI owns bias RW after it.
// The legacy scalar timestamp is explicitly a reference-camera label; a live
// producer instead supplies the immutable accepted physical endpoint.
inline bool condition_marginal_reset_prior(const ResetBiasPrior &p,double first_imu,double reference_offset,
                                           const conditional_bias::Vector6 &rw,double inflation,
                                           const conditional_bias::Vector6 &floors,
                                           conditional_bias::Conditioned &out) {
  using ov_core::numeric::finite;
  using ov_core::numeric::finite_matrix;
  if(!p.valid || p.joint || !finite(first_imu) || !finite(reference_offset) || !finite_matrix(p.bg) ||
      !finite_matrix(p.ba) || (p.cause!=0 && p.cause!=1))return false;
  const double endpoint=p.filter ? p.filter->imu_endpoint : p.t_snapshot+reference_offset;
  if(!finite(endpoint) || first_imu<endpoint)return false;
  conditional_bias::Matrix6 P;
  if(p.filter)P=p.filter->covariance;
  else {
    if(!finite_matrix(p.sigma_bg) || !finite_matrix(p.sigma_ba) ||
        (p.sigma_bg.array()<0.).any() || (p.sigma_ba.array()<0.).any())return false;
    P.setZero();P.diagonal().head<3>()=p.sigma_bg.array().square().matrix();
    P.diagonal().tail<3>()=p.sigma_ba.array().square().matrix();
  }
  return conditional_bias::condition(P,Eigen::MatrixXd(6,0),Eigen::MatrixXd(0,0),first_imu-endpoint,rw,
                                      p.cause==1 ? inflation : 1.,floors,out);
}
}
#endif
