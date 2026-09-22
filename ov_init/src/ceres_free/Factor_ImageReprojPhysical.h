/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_FACTOR_IMAGE_REPROJ_PHYSICAL_H
#define OV_INIT_FACTOR_IMAGE_REPROJ_PHYSICAL_H

#include "Factor_ImageReprojCalib.h"
#include "dynamic/GravityAlignment.h"

namespace ov_init {
namespace zbft_sfm {

// A fixed nominal IMU graph node owns the navigation variables. Its camera
// exposure at c0+dc has R=Exp(-omega*dc)R0 and p=p0+v0*dc. The initializer holds
// c=c0 throughout Solve; the final conditional export requests its analytic
// derivative. CPI times and graph membership never move with this clock block.
// Parameter order is the ordinary six reprojection blocks, then c, v, bg, ba.
// The nonzero-dc definition permits independent fixed-graph derivative tests;
// no nonlinear re-solve is needed by the production handoff.
class Factor_ImageReprojPhysical final : public CostFunction {
public:
  Factor_ImageReprojPhysical(const Eigen::Vector2d &uv, double sigma, bool fisheye, double clock_mean,
                            const Eigen::Vector3d &wm, const Eigen::Vector3d &am,
                            const Eigen::Matrix3d &A, const Eigen::Matrix3d &G, const Eigen::Matrix3d &Tg)
      : image_(uv,sigma,fisheye), clock_mean_(clock_mean), G_(G), C_(G*Tg*A), w0_(G*(wm-Tg*A*am)) {
    set_num_residuals(2);
    *mutable_parameter_block_sizes() = {4,3,3,4,3,8,1,3,3,3};
  }

  bool Evaluate(double const *const *p, double *residual, double **jac) const override {
    const double dt = p[6][0]-clock_mean_;
    if (!gravity_export::finite(dt)) return false;
    const bool need_clock = jac && jac[6];
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();
    if (dt != 0. || need_clock) {
      omega = w0_-G_*Eigen::Map<const Eigen::Vector3d>(p[8])+C_*Eigen::Map<const Eigen::Vector3d>(p[9]);
      if (!gravity_export::finite(omega) || !gravity_export::finite(Eigen::Map<const Eigen::Vector3d>(p[7]))) return false;
    }
    const double *image_parameters[6] = {p[0],p[1],p[2],p[3],p[4],p[5]};
    Eigen::Vector4d q;
    Eigen::Vector3d position;
    Eigen::Matrix3d E = Eigen::Matrix3d::Identity();
    if (dt != 0.) {
      E = ov_core::exp_so3(-dt*omega);
      q = ov_core::rot_2_quat(E*ov_core::quat_2_Rot(Eigen::Map<const Eigen::Vector4d>(p[0])));
      position = Eigen::Map<const Eigen::Vector3d>(p[1])+dt*Eigen::Map<const Eigen::Vector3d>(p[7]);
      image_parameters[0] = q.data(); image_parameters[1] = position.data();
    }
    if (!jac) return image_.Evaluate(image_parameters,residual,nullptr);

    // At the nominal chart the extra navigation columns are exactly zero and
    // all ordinary image residual/Jacobian arithmetic stays on its old path.
    for (int i = 7; i < 10; ++i)
      if (jac[i]) Eigen::Map<Eigen::Matrix<double,2,3,Eigen::RowMajor>>(jac[i]).setZero();
    if (dt == 0. && !need_clock) return image_.Evaluate(image_parameters,residual,jac);

    Eigen::Matrix<double,2,4,Eigen::RowMajor> Jq;
    Eigen::Matrix<double,2,3,Eigen::RowMajor> Jp;
    double *image_jacobians[6] = {Jq.data(),Jp.data(),jac[2],jac[3],jac[4],jac[5]};
    if (!image_.Evaluate(image_parameters,residual,image_jacobians)) return false;
    if (jac[0]) {
      Eigen::Map<Eigen::Matrix<double,2,4,Eigen::RowMajor>> out(jac[0]);
      out.leftCols<3>() = Jq.leftCols<3>()*E; out.col(3).setZero();
    }
    if (jac[1]) Eigen::Map<Eigen::Matrix<double,2,3,Eigen::RowMajor>>{jac[1]} = Jp;
    if (need_clock) Eigen::Map<Eigen::Vector2d>{jac[6]} = Jq.leftCols<3>()*omega+Jp*Eigen::Map<const Eigen::Vector3d>(p[7]);
    if (dt != 0.) {
      if (jac[7]) Eigen::Map<Eigen::Matrix<double,2,3,Eigen::RowMajor>>{jac[7]} = dt*Jp;
      const Eigen::Matrix<double,2,3> Jomega = dt*Jq.leftCols<3>()*ov_core::Jl_so3(-dt*omega);
      if (jac[8]) Eigen::Map<Eigen::Matrix<double,2,3,Eigen::RowMajor>>{jac[8]} = -Jomega*G_;
      if (jac[9]) Eigen::Map<Eigen::Matrix<double,2,3,Eigen::RowMajor>>{jac[9]} = Jomega*C_;
    }
    return gravity_export::finite(Jq) && gravity_export::finite(Jp) &&
           (!need_clock || gravity_export::finite(Eigen::Map<const Eigen::Vector2d>(jac[6])));
  }

private:
  Factor_ImageReprojCalib image_;
  double clock_mean_;
  Eigen::Matrix3d G_, C_;
  Eigen::Vector3d w0_;
};

} // namespace zbft_sfm
} // namespace ov_init
#endif
