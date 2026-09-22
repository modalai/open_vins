/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_GRAVITY_ALIGNMENT_H
#define OV_INIT_GRAVITY_ALIGNMENT_H

#include <Eigen/Dense>
#include <cstdint>
#include <cstring>
#include "utils/quat_ops.h"

namespace ov_init {
namespace gravity_export {
inline bool finite(double value) {
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
template <typename Derived> inline bool finite(const Eigen::MatrixBase<Derived> &value) {
  for (Eigen::Index c=0;c<value.cols();++c)
    for (Eigen::Index r=0;r<value.rows();++r)
      if (!finite(value(r,c))) return false;
  return true;
}

// Smooth shortest rotation taking finite g to +Z. The Rodrigues expression is
// well-conditioned at the pole itself: R=I there, but its derivative is nonzero.
inline bool alignment_rotation(const Eigen::Vector3d &g, Eigen::Matrix3d &R) {
  if (!finite(g)) return false;
  const double norm=g.norm();
  if (!finite(norm) || norm<=1e-12) return false;
  const Eigen::Vector3d u=g/norm;
  const double denominator=1.0+u.z();
  if (!finite(denominator) || denominator<=1e-10) return false;
  const Eigen::Matrix3d V=ov_core::skew_x(u.cross(Eigen::Vector3d::UnitZ()));
  const Eigen::Matrix3d out=Eigen::Matrix3d::Identity()+V+(V*V)/denominator;
  if (!finite(out)) return false;
  R=out;
  return true;
}

// Left rotation derivative of the shortest rotation R(g) taking g to +Z:
// R(g (+) dg) = Exp(J_left * dg) * R(g) + O(||dg||^2).
// tangent is the SAME 3x2 local S2 basis used by the covariance export.
// Valid at zero tilt too; nominal alignment never removes gravity uncertainty.
inline bool alignment_left_jacobian(const Eigen::Vector3d &g, const Eigen::Matrix<double,3,2> &tangent,
                                    const Eigen::Matrix3d &R, Eigen::Matrix<double,3,2> &J_left) {
  if (!finite(g) || !finite(tangent) || !finite(R)) return false;
  const double norm=g.norm();
  if (!finite(norm) || norm<=1e-12) return false;
  const Eigen::Vector3d u=g/norm, pole=Eigen::Vector3d::UnitZ();
  const double denominator=1.0+u.z();
  if (!finite(denominator) || denominator<=1e-10) return false;
  const Eigen::Matrix3d V=ov_core::skew_x(u.cross(pole));
  Eigen::Matrix<double,3,2> out;
  for (int j=0;j<2;++j) {
    const Eigen::Vector3d du=(tangent.col(j)-u*u.dot(tangent.col(j)))/norm;
    const Eigen::Matrix3d dV=ov_core::skew_x(du.cross(pole));
    const Eigen::Matrix3d dR=dV+(dV*V+V*dV)/denominator-V*V*(du.z()/(denominator*denominator));
    const Eigen::Matrix3d A=dR*R.transpose();
    out.col(j)<<.5*(A(2,1)-A(1,2)),.5*(A(0,2)-A(2,0)),.5*(A(1,0)-A(0,1));
  }
  if (!finite(out)) return false;
  J_left=out;
  return true;
}

// Fill [orientation,position] output rows for one pose into J=[T,U]. All state
// means supplied here are PRE-rotation. The final two input columns are gravity.
inline void pose_rows(Eigen::MatrixXd &J, int row, const Eigen::Matrix3d &R_GtoI,
                      const Eigen::Vector3d &p_IinG, const Eigen::Matrix3d &R_align,
                      const Eigen::Matrix<double,3,2> &J_left) {
  const Eigen::Index gravity_column=J.cols()-2;
  J.block<3,3>(row,row).setIdentity();
  J.block<3,3>(row+3,row+3)=R_align;
  J.block<3,2>(row,gravity_column)=R_GtoI*R_align.transpose()*J_left;
  J.block<3,2>(row+3,gravity_column)=-ov_core::skew_x(R_align*p_IinG)*J_left;
}
inline void velocity_rows(Eigen::MatrixXd &J, int row, const Eigen::Vector3d &v_IinG,
                          const Eigen::Matrix3d &R_align, const Eigen::Matrix<double,3,2> &J_left) {
  J.block<3,3>(row,row)=R_align;
  J.block<3,2>(row,J.cols()-2)=-ov_core::skew_x(R_align*v_IinG)*J_left;
}
} // namespace gravity_export
} // namespace ov_init
#endif
