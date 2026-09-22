/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "RawImuCpi.h"
#include <cstdint>
#include <cstring>

namespace ov_init {
namespace {
bool finite_scalar(double value) {
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value), "IEEE binary64 required");
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
template<class Derived> bool finite(const Eigen::MatrixBase<Derived> &value) {
  for (int col=0; col<value.cols(); ++col)
    for (int row=0; row<value.rows(); ++row)
      if (!finite_scalar(value(row,col))) return false;
  return true;
}
} // namespace

bool RawImuCpiModel::set_calibration(const Eigen::Matrix3d &A, const Eigen::Matrix3d &G, const Eigen::Matrix3d &Tg) {
  ready_ = false;
  if (!finite(A) || !finite(G) || !finite(Tg)) return false;
  const Eigen::FullPivLU<Eigen::Matrix3d> lu_a(A), lu_g(G);
  if (!lu_a.isInvertible() || !lu_g.isInvertible() || !(A.determinant()>0) || !(G.determinant()>0) ||
      !(lu_a.rcond()>1e-12) || !(lu_g.rcond()>1e-12)) return false;
  A_=A; G_=G; Tg_=Tg; C_=-G*Tg*A;
  identity_ = (A.array()==Eigen::Matrix3d::Identity().array()).all() &&
              (G.array()==Eigen::Matrix3d::Identity().array()).all() && (Tg.array()==0.0).all();

  // B=[G,-G*Tg*A;0,A] maps [raw gyro;raw accel] to corrected readings and
  // their biases. Apply the same B to each independent white/RW noise pair.
  corrected_noise_from_raw_.setZero();
  corrected_noise_from_raw_.block<3,3>(0,0)=G;
  corrected_noise_from_raw_.block<3,3>(0,6)=C_;
  corrected_noise_from_raw_.block<3,3>(3,3)=G;
  corrected_noise_from_raw_.block<3,3>(3,9)=C_;
  corrected_noise_from_raw_.block<3,3>(6,6)=A;
  corrected_noise_from_raw_.block<3,3>(9,9)=A;

  // CPI error order is [theta,bg,beta,ba,alpha]. Only the bias blocks need
  // pulling back; corrected theta/alpha/beta already use the physical IMU frame.
  // B^-1=[G^-1,Tg;0,A^-1], retaining every bias/kinematics cross-covariance.
  raw_error_from_corrected_.setIdentity();
  raw_error_from_corrected_.block<3,3>(3,3)=lu_g.inverse();
  raw_error_from_corrected_.block<3,3>(3,9)=Tg;
  raw_error_from_corrected_.block<3,3>(9,9)=lu_a.inverse();
  ready_=finite(C_) && finite(corrected_noise_from_raw_) && finite(raw_error_from_corrected_);
  return ready_;
}

bool RawImuCpiModel::correct(const Eigen::Vector3d &wm, const Eigen::Vector3d &am,
                            const Eigen::Vector3d &bg, const Eigen::Vector3d &ba,
                            Eigen::Vector3d &w, Eigen::Vector3d &a) const {
  if (!ready_ || !finite(wm) || !finite(am) || !finite(bg) || !finite(ba)) return false;
  if (identity_) { w=wm-bg; a=am-ba; }
  else { a=A_*(am-ba); w=G_*(wm-bg-Tg_*a); }
  return finite(w) && finite(a);
}

bool RawImuCpiModel::preintegrate(const std::vector<ov_core::ImuData> &readings,
                                 const Eigen::Vector3d &bg, const Eigen::Vector3d &ba,
                                 const Eigen::Vector4d &sigma, std::shared_ptr<RawBiasCpiV1> &output) const {
  if (!ready_ || readings.size()<2 || !finite(bg) || !finite(ba) || !finite(sigma) || !(sigma.array()>0).all()) return false;
  for (size_t i=0; i<readings.size(); ++i)
    if (!finite_scalar(readings[i].timestamp) || !finite(readings[i].wm) || !finite(readings[i].am) ||
        (i && !(readings[i].timestamp>readings[i-1].timestamp))) return false;
  auto cpi=std::make_shared<RawBiasCpiV1>(sigma(0),sigma(1),sigma(2),sigma(3),true);
  if (identity_) {
    cpi->setLinearizationPoints(bg,ba);
    for (size_t i=0;i+1<readings.size();++i)
      cpi->feed_IMU(readings[i].timestamp,readings[i+1].timestamp,readings[i].wm,readings[i].am,readings[i+1].wm,readings[i+1].am);
  } else {
    cpi->setLinearizationPoints(G_*bg+C_*ba,A_*ba);
    cpi->Q_c=(corrected_noise_from_raw_*cpi->Q_c*corrected_noise_from_raw_.transpose()).eval();
    for (size_t i=0;i+1<readings.size();++i) {
      const auto &x=readings[i], &y=readings[i+1];
      cpi->feed_IMU(x.timestamp,y.timestamp,G_*x.wm+C_*x.am,A_*x.am,G_*y.wm+C_*y.am,A_*y.am);
    }
    // Chain rule back to the RAW bias columns. Tg couples accelerometer bias
    // to orientation as well as velocity/position; omitting H_q loses that term.
    cpi->H_q=cpi->J_q*C_;
    cpi->H_a=(cpi->J_a*C_+cpi->H_a*A_).eval();
    cpi->H_b=(cpi->J_b*C_+cpi->H_b*A_).eval();
    cpi->J_q=(cpi->J_q*G_).eval();
    cpi->J_a=(cpi->J_a*G_).eval();
    cpi->J_b=(cpi->J_b*G_).eval();
    cpi->P_meas=(raw_error_from_corrected_*cpi->P_meas*raw_error_from_corrected_.transpose()).eval();
    cpi->P_meas=(0.5*(cpi->P_meas+cpi->P_meas.transpose())).eval();
    cpi->setLinearizationPoints(bg,ba);
  }
  if (!finite_scalar(cpi->DT) || !(cpi->DT>0) || !finite(cpi->R_k2tau) || !finite(cpi->q_k2tau) ||
      !finite(cpi->alpha_tau) || !finite(cpi->beta_tau) || !finite(cpi->J_q) || !finite(cpi->H_q) ||
      !finite(cpi->J_a) || !finite(cpi->J_b) || !finite(cpi->H_a) || !finite(cpi->H_b) || !finite(cpi->P_meas)) return false;
  Eigen::LLT<Eigen::Matrix<double,15,15>> covariance_check(cpi->P_meas);
  if (covariance_check.info()!=Eigen::Success) return false;
  output=std::move(cpi);
  return true;
}
} // namespace ov_init
