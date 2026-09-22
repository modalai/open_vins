/* Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <Eigen/Dense>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#ifdef OV_TEST_CERES
#include "ceres/Factor_ImuCPIv1.h"
using Factor = ov_init::Factor_ImuCPIv1;
#else
#include "ceres_free/Factor_ImuCPIv1.h"
using Factor = ov_init::zbft_sfm::Factor_ImuCPIv1;
#endif
#include "utils/finite.h"

int main(int argc,char **argv) {
  const std::string mode=argc>1?argv[1]:"valid";
  Eigen::Vector3d grav(0,0,9.81), a=Eigen::Vector3d::Zero(), b=a, ba=a, bg=a;
  Eigen::Vector4d q(0,0,0,1);
  Eigen::Matrix3d Jq=Eigen::Matrix3d::Zero(), Jb=Jq, Ja=Jq, Hb=Jq, Ha=Jq, Hq=Jq;
  Eigen::Matrix<double,15,15> P=Eigen::Matrix<double,15,15>::Identity();
  double dt=.02;
  // Opaque runtime input: fast-math is allowed to discard a compile-time NaN
  // assignment. Inject the actual nonfinite bits that a corrupt input carries.
  volatile std::uint64_t nan_input=UINT64_C(0x7ff8000000000000);
  volatile std::uint64_t inf_input=UINT64_C(0x7ff0000000000000);
  const std::uint64_t nan_bits=nan_input, inf_bits=inf_input;
  double nan, inf;
  std::memcpy(&nan,&nan_bits,sizeof(nan));
  std::memcpy(&inf,&inf_bits,sizeof(inf));
  if(mode=="cov_nan") P(0,0)=nan;
  if(mode=="cov_inf") P(1,1)=inf;
  if(mode=="cov_ninf") P(2,2)=-inf;
  if(mode=="cov_zero") P.setZero();
  if(mode=="cov_singular") P(7,7)=0;
  if(mode=="cov_indefinite") P(3,3)=-1;
  if(mode=="cov_asymmetric") P(2,3)=.2;
  if(mode=="dt_nan") dt=nan;
  if(mode=="dt_zero") dt=0;
  if(mode=="dt_negative") dt=-.1;
  if(mode=="alpha_nan") a(1)=nan;
  if(mode=="beta_inf") b(2)=inf;
  if(mode=="quat_zero") q.setZero();
  if(mode=="quat_nan") q(0)=nan;
  if(mode=="gravity_inf") grav(0)=inf;
  if(mode=="bias_nan") ba(1)=nan;
  if(mode=="jac_nan") Jq(0,0)=nan;
  if(mode=="tg_jac_inf") Hq(0,1)=inf;
  if(mode=="dt_nan" && ov_core::numeric::finite(dt)) {
    std::fprintf(stderr,"invalid fixture: dt_nan did not supply nonfinite input\n");
    return 2;
  }
  Factor f(dt,grav,a,b,q,ba,bg,Jq,Jb,Ja,Hb,Ha,P,Hq);
  Eigen::Vector4d qi(0,0,0,1); Eigen::Vector3d zero=Eigen::Vector3d::Zero(), gravity(0,0,9.81);
  std::array<const double*,11> p{{qi.data(),zero.data(),zero.data(),zero.data(),zero.data(),qi.data(),zero.data(),zero.data(),zero.data(),zero.data(),gravity.data()}};
  Eigen::Matrix<double,15,1> res;res.setConstant(17);
  bool accepted=f.Evaluate(p.data(),res.data(),nullptr);
  bool ok=mode=="valid" ? accepted&&ov_core::numeric::finite_matrix(res) : !accepted&&(res.array()==17).all();
  std::printf("IMU factor %s: %s (accepted=%d)\n",mode.c_str(),ok?"PASS":"FAIL",accepted);
  return ok?0:1;
}
