/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Independent raw-coordinate ODE covariance and physical IMU fixtures.
 */
#include <cstdio>
#include <cstring>
#include <limits>
#include "dynamic/RawImuCpi.h"
#ifdef USE_CERES_FREE_INIT
#include "ceres_free/Factor_ImuCPIv1.h"
#else
#include "ceres/Factor_ImuCPIv1.h"
#endif

namespace {
using namespace ov_init;
using M15=Eigen::Matrix<double,15,15>;
using M12=Eigen::Matrix<double,12,12>;
using G15=Eigen::Matrix<double,15,12>;
int failures=0;
void check(bool good,const char *message){if(!good){++failures;std::printf("FAIL: %s\n",message);}}
struct Fixture {
  Eigen::Matrix3d A,G,Tg;
  Eigen::Vector3d bg{.013,-.021,.008},ba{.024,-.016,.011};
  Eigen::Vector4d sigma{.002,.0002,.03,.002};
  std::vector<ov_core::ImuData> data;
  Fixture() {
    Eigen::Matrix3d D;D<<1.04,.025,-.012,0,.96,.019,0,0,1.02;
    A=ov_core::exp_so3(Eigen::Vector3d(.23,-.16,.07))*D;
    D<<.93,-.014,.021,0,1.07,.012,0,0,.98;
    G=ov_core::exp_so3(Eigen::Vector3d(-.14,.08,.11))*D;
    Tg<<.004,-.002,.001, .003,.002,-.004, -.002,.005,.003;
    for(int k=0;k<=320;++k){
      const double t=k/800.0;
      const Eigen::Vector3d w(.7+.2*std::sin(1.3*t),-.4+.1*std::cos(1.9*t),.3+.15*std::sin(.7*t));
      const Eigen::Vector3d a(1.2*std::sin(1.6*t),.8*std::cos(1.2*t),-9.81+.4*std::sin(1.1*t));
      ov_core::ImuData x;x.timestamp=10+t;x.am=A.inverse()*a+ba;x.wm=G.inverse()*w+bg+Tg*a;data.push_back(x);
    }
  }
};
std::shared_ptr<RawBiasCpiV1> integrate(const Fixture &f,const Eigen::Vector3d &bg,const Eigen::Vector3d &ba) {
  RawImuCpiModel model;check(model.set_calibration(f.A,f.G,f.Tg),"valid model configured");
  std::shared_ptr<RawBiasCpiV1> result;
  check(model.preintegrate(f.data,bg,ba,f.sigma,result),"raw calibrated CPI succeeds");return result;
}

M15 native_raw_covariance(const Fixture &f) {
  // Direct raw-coordinate covariance ODE, independently of the change-of-basis
  // implementation. Noise order [nw,nwb,na,nab]; errors [theta,bg,beta,ba,alpha].
  const Eigen::Matrix3d C=-f.G*f.Tg*f.A;
  M12 Q=M12::Zero();for(int block=0;block<4;++block)Q.block<3,3>(3*block,3*block)=f.sigma(block)*f.sigma(block)*Eigen::Matrix3d::Identity();
  M15 P=M15::Zero();Eigen::Matrix3d R=Eigen::Matrix3d::Identity();
  for(size_t i=0;i+1<f.data.size();++i){
    const double dt=f.data[i+1].timestamp-f.data[i].timestamp;
    const Eigen::Vector3d a=f.A*(.5*(f.data[i].am+f.data[i+1].am)-f.ba);
    const Eigen::Vector3d w=f.G*(.5*(f.data[i].wm+f.data[i+1].wm)-f.bg-f.Tg*a);
    const Eigen::Matrix3d Rm=ov_core::exp_so3(-.5*dt*w)*R,R1=ov_core::exp_so3(-dt*w)*R;
    auto derivative=[&](const Eigen::Matrix3d &Rt,const M15 &Pt)->M15{
      M15 F=M15::Zero();F.block<3,3>(0,0)=-ov_core::skew_x(w);F.block<3,3>(0,3)=-f.G;F.block<3,3>(0,9)=-C;
      F.block<3,3>(6,0)=-Rt.transpose()*ov_core::skew_x(a);F.block<3,3>(6,9)=-Rt.transpose()*f.A;F.block<3,3>(12,6).setIdentity();
      G15 N=G15::Zero();N.block<3,3>(0,0)=-f.G;N.block<3,3>(0,6)=-C;N.block<3,3>(3,3).setIdentity();
      N.block<3,3>(6,6)=-Rt.transpose()*f.A;N.block<3,3>(9,9).setIdentity();
      return F*Pt+Pt*F.transpose()+N*Q*N.transpose();
    };
    const M15 k1=derivative(R,P),k2=derivative(Rm,P+.5*dt*k1),k3=derivative(Rm,P+.5*dt*k2),k4=derivative(R1,P+dt*k3);
    P+=dt/6*(k1+2*k2+2*k3+k4);P=(.5*(P+P.transpose())).eval();R=R1;
  }
  return P;
}

void physical_and_jacobians(const Fixture &f,const std::shared_ptr<RawBiasCpiV1> &c) {
  if(!c)return;
  ov_core::CpiV1 truth(f.sigma(0),f.sigma(1),f.sigma(2),f.sigma(3),true);
  truth.setLinearizationPoints(Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero());
  ov_core::CpiV1 wrong(f.sigma(0),f.sigma(1),f.sigma(2),f.sigma(3),true);wrong.setLinearizationPoints(f.bg,f.ba);
  for(size_t i=0;i+1<f.data.size();++i){
    const auto &x=f.data[i],&y=f.data[i+1];
    const Eigen::Vector3d ax=f.A*(x.am-f.ba),ay=f.A*(y.am-f.ba);
    truth.feed_IMU(x.timestamp,y.timestamp,f.G*(x.wm-f.bg-f.Tg*ax),ax,f.G*(y.wm-f.bg-f.Tg*ay),ay);
    wrong.feed_IMU(x.timestamp,y.timestamp,x.wm,x.am,y.wm,y.am);
  }
  check((c->R_k2tau-truth.R_k2tau).norm()<1e-12 && (c->alpha_tau-truth.alpha_tau).norm()<1e-12 &&
        (c->beta_tau-truth.beta_tau).norm()<1e-12,"calibrated means match independent physical corrected-force integration");
  check((wrong.R_k2tau-c->R_k2tau).norm()>.01 && (wrong.beta_tau-c->beta_tau).norm()>.1,"legacy identity-input model fails this nonidentity/Tg fixture");
  const M15 oracle=native_raw_covariance(f);
  const double cov_error=(oracle-c->P_meas).norm()/oracle.norm();
  check(cov_error<1e-10,"complete pulled-back covariance matches independent native raw-coordinate RK4 ODE");
  check(c->P_meas.block<3,3>(0,6).norm()>1e-6,"fixture includes gyro/accel coupled uncertainty");
  check((c->b_w_lin-f.bg).norm()==0 && (c->b_a_lin-f.ba).norm()==0,"exported linearization points remain raw biases");
  double error=0;
  for(int k=0;k<6;++k){
    Eigen::Vector3d bgp=f.bg,bgm=f.bg,bap=f.ba,bam=f.ba;const double eps=2e-6;
    if(k<3){bgp(k)+=eps;bgm(k)-=eps;}else{bap(k-3)+=eps;bam(k-3)-=eps;}
    auto plus=integrate(f,bgp,bap),minus=integrate(f,bgm,bam);if(!plus||!minus)continue;
    const Eigen::Vector3d dq=ov_core::log_so3(plus->R_k2tau*minus->R_k2tau.transpose())/(2*eps);
    const Eigen::Vector3d da=(plus->alpha_tau-minus->alpha_tau)/(2*eps),db=(plus->beta_tau-minus->beta_tau)/(2*eps);
    error=std::max(error,(dq-(k<3?c->J_q:c->H_q).col(k%3)).norm());
    error=std::max(error,(da-(k<3?c->J_a:c->H_a).col(k%3)).norm());
    error=std::max(error,(db-(k<3?c->J_b:c->H_b).col(k%3)).norm());
  }
  std::printf("raw CPI relative covariance error %.3e, bias Jacobian FD error %.3e\n",cov_error,error);
  check(error<2e-7,"all six raw bias columns including rotation/ba match finite differences");
}

void gauge_and_identity(const Fixture &f,const std::shared_ptr<RawBiasCpiV1> &c) {
  if(!c)return;
  const Eigen::Matrix3d S=ov_core::exp_so3(Eigen::Vector3d(.31,-.18,.24));Fixture b=f;
  b.A=S*f.A;b.G=S*f.G;b.Tg=f.Tg*S.transpose();auto d=integrate(b,b.bg,b.ba);if(!d)return;
  M15 J=M15::Identity();J.block<3,3>(0,0)=S;J.block<3,3>(6,6)=S;J.block<3,3>(12,12)=S;
  check((d->R_k2tau-S*c->R_k2tau*S.transpose()).norm()<2e-12 && (d->alpha_tau-S*c->alpha_tau).norm()<2e-12 &&
        (d->beta_tau-S*c->beta_tau).norm()<2e-12,"physical CPI means are body-gauge equivariant");
  check((d->P_meas-J*c->P_meas*J.transpose()).norm()<1e-14,"all CPI covariance cross blocks obey body-gauge congruence");
  check((d->J_q-S*c->J_q).norm()<1e-12 && (d->H_q-S*c->H_q).norm()<1e-12 &&
        (d->J_a-S*c->J_a).norm()<1e-12 && (d->H_a-S*c->H_a).norm()<1e-12,"raw bias Jacobians obey body-gauge transform");
  Fixture id=f;id.A.setIdentity();id.G.setIdentity();id.Tg.setZero();auto identity=integrate(id,id.bg,id.ba);
  ov_core::CpiV1 old(id.sigma(0),id.sigma(1),id.sigma(2),id.sigma(3),true);old.setLinearizationPoints(id.bg,id.ba);
  for(size_t i=0;i+1<id.data.size();++i){const auto&x=id.data[i];const auto&y=id.data[i+1];old.feed_IMU(x.timestamp,y.timestamp,x.wm,x.am,y.wm,y.am);}
  check(identity && std::memcmp(identity->P_meas.data(),old.P_meas.data(),225*sizeof(double))==0 &&
        std::memcmp(identity->alpha_tau.data(),old.alpha_tau.data(),3*sizeof(double))==0 &&
        std::memcmp(identity->R_k2tau.data(),old.R_k2tau.data(),9*sizeof(double))==0,"identity mean/covariance bytes match legacy CPI");
}

void factor_bias_jacobians(const std::shared_ptr<RawBiasCpiV1>&c) {
  if(!c)return;
  Eigen::Vector3d gravity(0,0,9.81);
#ifdef USE_CERES_FREE_INIT
  using InputFactor = zbft_sfm::Factor_ImuCPIv1;
#else
  using InputFactor = ov_init::Factor_ImuCPIv1;
#endif
  InputFactor factor(c->DT,gravity,c->alpha_tau,c->beta_tau,c->q_k2tau,c->b_a_lin,c->b_w_lin,
                                  c->J_q,c->J_b,c->J_a,c->H_b,c->H_a,c->P_meas,c->H_q);
  std::vector<Eigen::VectorXd> values(11);
  for(int i=0;i<11;++i)values[i]=Eigen::VectorXd::Zero(i==0||i==5?4:3);
  values[0]=ov_core::rot_2_quat(ov_core::exp_so3(Eigen::Vector3d(.13,-.17,.21)));
  values[5]=ov_core::rot_2_quat(ov_core::exp_so3(Eigen::Vector3d(-.08,.05,-.11))*c->R_k2tau*ov_core::quat_2_Rot(values[0]));
  values[1]=c->b_w_lin+Eigen::Vector3d(.7,-.4,.2);values[6]=values[1]+Eigen::Vector3d(.002,-.001,.003);
  values[3]=c->b_a_lin+Eigen::Vector3d(.3,-.2,.4);values[8]=values[3]+Eigen::Vector3d(.01,.02,-.01);
  values[7]<<.3,-.2,.1;values[9]<<.1,-.3,.2;values[10]=gravity;
  std::vector<const double*> parameters;std::vector<std::vector<double>> jac(11);std::vector<double*> ptrs;
  for(auto &v:values)parameters.push_back(v.data());
  for(int i=0;i<11;++i){jac[i].resize(15*values[i].size());ptrs.push_back(jac[i].data());}
  Eigen::Matrix<double,15,1> residual;check(factor.Evaluate(parameters.data(),residual.data(),ptrs.data()),"raw CPI factor evaluates");
  double error=0;
  for(int block:{1,3,6,8})for(int k=0;k<3;++k){
    const double old=values[block](k),eps=2e-6;Eigen::Matrix<double,15,1> plus,minus;
    values[block](k)=old+eps;factor.Evaluate(parameters.data(),plus.data(),nullptr);
    values[block](k)=old-eps;factor.Evaluate(parameters.data(),minus.data(),nullptr);values[block](k)=old;
    Eigen::Map<Eigen::Matrix<double,15,3,Eigen::RowMajor>> J(jac[block].data());
    error=std::max(error,((plus-minus)/(2*eps)-J.col(k)).norm()/std::max(1.0,J.col(k).norm()));
  }
  std::printf("raw factor nonzero-bias Jacobian relative FD error %.3e\n",error);
  check(error<2e-8,"factor raw bg/ba Jacobians include normalized nonzero orientation correction");
}

void invalid_inputs(Fixture f) {
  RawImuCpiModel model;Eigen::Matrix3d bad=f.G;bad.row(1).setZero();check(!model.set_calibration(f.A,bad,f.Tg),"singular gyro map rejected");
  bad=f.A;bad(0,0)=std::numeric_limits<double>::quiet_NaN();check(!model.set_calibration(bad,f.G,f.Tg),"nonfinite accel map rejected under fast math");
  check(model.set_calibration(f.A,f.G,f.Tg),"valid model restored");std::shared_ptr<RawBiasCpiV1> output;
  f.data[10].timestamp=f.data[9].timestamp;check(!model.preintegrate(f.data,f.bg,f.ba,f.sigma,output)&&!output,"nonmonotone input rejects without publishing CPI");
}
} // namespace
int main(){Fixture f;auto c=integrate(f,f.bg,f.ba);physical_and_jacobians(f,c);gauge_and_identity(f,c);factor_bias_jacobians(c);invalid_inputs(f);
  std::printf("RAW_IMU_CPI %s failures=%d\n",failures?"FAIL":"PASS",failures);return failures?1:0;}
