/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterZeroVelocity.h"
#include "update/ZeroVelocityModel.h"
#include "utils/print.h"

namespace {
using namespace ov_msckf;
using Model = ZeroVelocityModel;
using M9 = Eigen::Matrix<double,9,9>;
using M6 = Eigen::Matrix<double,6,6>;
int failures=0;
void check(bool ok,const char *why) { if(!ok){++failures;std::printf("FAIL: %s\n",why);} }
struct Fixture {
  Eigen::Matrix3d A, Tg, G, R, Rf;
  Eigen::Vector3d am,wm,bg,ba,g=Eigen::Vector3d(0,0,9.81);
  double ww=23.0,wa=3.7;
  Fixture() {
    Eigen::Matrix3d D; D<<1.03,.012,-.007, 0,.98,.009, 0,0,1.02;
    A=ov_core::exp_so3(Eigen::Vector3d(.14,-.08,.03))*D;
    G=ov_core::exp_so3(Eigen::Vector3d(-.07,.02,.04))*D;
    Tg<<4e-4,-2e-4,3e-4, 5e-4,1e-4,-3e-4, -2e-4,6e-4,3e-4;
    R=ov_core::exp_so3(Eigen::Vector3d(.24,-.19,.07));
    Rf=ov_core::exp_so3(Eigen::Vector3d(-.05,.02,.03))*R;
    bg<<.015,-.011,.005; ba<<.02,-.03,.01;
    am=ba+A.inverse()*R*g+Eigen::Vector3d(.01,-.007,.012);
    wm=bg+Tg*R*g+Eigen::Vector3d(.0003,-.0002,.0001);
  }
};
bool linearize(const Fixture &f,Model::Jacobian &H,Model::Residual &r,bool fej) {
  Model m; if(!m.set_calibration(f.A,f.Tg)) return false;
  return m.linearize(f.am,f.wm,f.bg,f.ba,f.R*f.g,(fej?f.Rf:f.R)*f.g,f.ww,f.wa,H,r);
}
void jacobian_checks(const Fixture &f) {
  double worst=0;
  for(bool fej:{false,true}) {
    Model::Jacobian H; Model::Residual r;
    check(linearize(f,H,r,fej),"valid stationary model linearizes");
    // Differentiate the measurement at the actual linearization pose (FEJ or
    // current); the innovation itself must always use the current pose.
    Fixture reference=f; if(fej) reference.R=f.Rf;
    for(int col=0;col<9;++col) {
      Fixture plus=reference,minus=reference;
      const double eps=2e-7;
      if(col<3) {
        Eigen::Vector3d dx=Eigen::Vector3d::Zero();dx(col)=eps;
        plus.R=ov_core::exp_so3(-dx)*reference.R;
        minus.R=ov_core::exp_so3(dx)*reference.R;
      } else if(col<6) {plus.bg(col-3)+=eps;minus.bg(col-3)-=eps;}
      else {plus.ba(col-6)+=eps;minus.ba(col-6)-=eps;}
      Model::Jacobian ignored;Model::Residual rp,rm;
      check(linearize(plus,ignored,rp,false)&&linearize(minus,ignored,rm,false),"FD perturbed model valid");
      worst=std::max(worst,(H.col(col)+(rp-rm)/(2*eps)).norm());
    }
    Model::Jacobian Hcurrent;Model::Residual rcurrent;
    check(linearize(f,Hcurrent,rcurrent,false),"current linearization valid");
    check((r-rcurrent).norm()==0,"FEJ affects Jacobian but preserves current-pose innovation");
    if(fej) check((H-Hcurrent).norm()>.1,"FEJ fixture genuinely separates current and frozen pose");
  }
  std::printf("ZUPT Jacobian maximum absolute FD error %.3e\n",worst);
  check(worst<1e-7,"all orientation and raw-bias Jacobians match FD with and without FEJ");
}
void identity_checks(Fixture f) {
  f.A.setIdentity();f.Tg.setZero();f.G.setIdentity();
  f.R.setIdentity();f.Rf.setIdentity();f.g<<0,0,8;f.ww=16;f.wa=4;
  f.am<<.125,-.25,8.5;f.ba<<.03125,.0625,.125;f.wm<<.125,-.0625,.25;f.bg<<.03125,.125,-.125;
  Model::Jacobian H;Model::Residual r;check(linearize(f,H,r,true),"identity model valid");
  Model::Jacobian oldH=Model::Jacobian::Zero();Model::Residual oldr;
  const Eigen::Vector3d a=f.am-f.ba,w=f.wm-f.bg;
  oldr.head<3>()=-f.ww*w;oldr.tail<3>()=-f.wa*(a-f.R*f.g);
  oldH.block<3,3>(0,3)=-f.ww*Eigen::Matrix3d::Identity();
  oldH.block<3,3>(3,0)=-f.wa*ov_core::skew_x(f.Rf*f.g);
  oldH.block<3,3>(3,6)=-f.wa*Eigen::Matrix3d::Identity();
  check(H==oldH && r==oldr,"identity/Tg-zero model recovers exact legacy values on binary-exact fixture");
}
void whitening_and_gauge(const Fixture &f) {
  Model::Jacobian H;Model::Residual r;check(linearize(f,H,r,true),"fixed model valid");
  M6 W=M6::Zero();W.topLeftCorner<3,3>()=f.ww*Eigen::Matrix3d::Identity();W.bottomRightCorner<3,3>()=f.wa*Eigen::Matrix3d::Identity();
  const M6 Winv=W.inverse();
  // Independent oracle: propagate the complete raw covariance through the
  // corrected measurement map, including the gyro/accel cross-covariance.
  M6 B=M6::Zero();B.topLeftCorner<3,3>()=f.G;B.topRightCorner<3,3>()=-f.G*f.Tg*f.A;B.bottomRightCorner<3,3>()=f.A;
  const M6 C=B*Winv*Winv.transpose()*B.transpose();
  Model::Residual corrected;
  const Eigen::Vector3d ahat=f.A*(f.am-f.ba);
  corrected.head<3>()=-f.G*(f.wm-f.bg-f.Tg*ahat);
  corrected.tail<3>()=-(ahat-f.R*f.g);
  const Model::Jacobian Hc=B*Winv*H;
  const auto solve=C.llt();
  check(std::abs(corrected.dot(solve.solve(corrected))-r.squaredNorm())<1e-11,"raw whitening equals full corrected covariance cost");
  check((Hc.transpose()*solve.solve(Hc)-H.transpose()*H).norm()<1e-9,"raw whitening preserves full corrected covariance information");
  check((Hc.transpose()*solve.solve(corrected)-H.transpose()*r).norm()<1e-10,"raw whitening preserves score with accel-noise correlation");
  check(C.topRightCorner<3,3>().norm()>1e-7,"fixture has material gyro/accel noise correlation");

  const Eigen::Matrix3d S=ov_core::exp_so3(Eigen::Vector3d(-.27,.11,.19));
  Fixture b=f;b.A=S*f.A;b.Tg=f.Tg*S.transpose();b.G=S*f.G;b.R=S*f.R;b.Rf=S*f.Rf;
  Model::Jacobian Hb;Model::Residual rb;check(linearize(b,Hb,rb,true),"second body gauge valid");
  M9 J=M9::Identity();J.topLeftCorner<3,3>()=S;
  check((r-rb).norm()<1e-12 && (Hb-H*J.transpose()).norm()<1e-12,"body gauge preserves raw innovation and transports orientation columns");
  M9 L=M9::Identity();for(int i=0;i<9;++i)for(int j=0;j<i;++j)L(i,j)=.03*std::sin(2*i+j);
  const M9 P=.01*L*L.transpose(),Pb=J*P*J.transpose();
  const M6 innovation=H*P*H.transpose()+M6::Identity(),innovation_b=Hb*Pb*Hb.transpose()+M6::Identity();
  const Eigen::Matrix<double,9,6> K=P*H.transpose()*innovation.inverse(),Kb=Pb*Hb.transpose()*innovation_b.inverse();
  const M9 posterior=P-K*H*P,posterior_b=Pb-Kb*Hb*Pb;
  check(std::abs(r.dot(innovation.llt().solve(r))-rb.dot(innovation_b.llt().solve(rb)))<1e-12,"chi-square gate invariant under body gauge");
  check((Kb*rb-J*K*r).norm()<1e-12 && (posterior_b-J*posterior*J.transpose()).norm()<1e-12,
        "EKF increment and covariance invariant including cross blocks");
}

std::shared_ptr<State> state_for_update(bool kalibr,const Eigen::Matrix3d &Ra,const Eigen::Matrix3d &D,const Eigen::Matrix3d &Tg,
                                     const Eigen::Matrix3d &R,const Eigen::Matrix3d &Rf,const Eigen::MatrixXd &P) {
  StateOptions options;options.num_cameras=1;options.imu_model=kalibr?StateOptions::KALIBR:StateOptions::RPNG;options.do_fej=true;
  auto s=std::make_shared<State>(options);s->_timestamp=1.0;
  Eigen::Matrix<double,6,1> da,dw;
  if(kalibr) {da<<D(0,0),0,0,D(1,1),0,D(2,2);dw<<1,0,0,1,0,1;}
  else {da<<D(0,0),0,D(1,1),0,0,D(2,2);dw<<1,0,1,0,0,1;}
  Eigen::Matrix<double,9,1> tv;tv<<Tg.col(0),Tg.col(1),Tg.col(2);
  s->_calib_imu_da->set_value(da);s->_calib_imu_dw->set_value(dw);s->_calib_imu_tg->set_value(tv);
  s->_calib_imu_ACCtoIMU->set_value(ov_core::rot_2_quat(kalibr?Eigen::Matrix3d::Identity():Ra));
  s->_calib_imu_GYROtoIMU->set_value(ov_core::rot_2_quat(kalibr?Eigen::Matrix3d(Ra.transpose()):Eigen::Matrix3d::Identity()));
  Eigen::Matrix<double,16,1> x=Eigen::Matrix<double,16,1>::Zero();
  x.head<4>()=ov_core::rot_2_quat(R);x.segment<3>(10)<<.015,-.011,.005;x.tail<3>()<<.02,-.03,.01;
  s->_imu->set_value(x);x.head<4>()=ov_core::rot_2_quat(Rf);s->_imu->set_fej(x);
  StateHelper::set_initial_covariance(s,P,{s->_imu});return s;
}
void production_update(const Fixture &f,bool zero_residual) {
  const Eigen::Matrix3d Ra=ov_core::exp_so3(Eigen::Vector3d(.16,-.11,.04)),S=Ra.transpose();
  const Eigen::Matrix3d D=Eigen::Vector3d(1.03,.97,1.01).asDiagonal(),A=Ra*D;
  Eigen::MatrixXd L=Eigen::MatrixXd::Identity(15,15);
  for(int i=0;i<15;++i)for(int j=0;j<i;++j)L(i,j)=.025*std::sin(3*i+j);
  const Eigen::MatrixXd P=.01*L*L.transpose();Eigen::MatrixXd J=Eigen::MatrixXd::Identity(15,15);J.topLeftCorner<3,3>()=S;
  auto a=state_for_update(false,Ra,D,f.Tg,f.R,f.Rf,P);
  auto b=state_for_update(true,Ra,D,f.Tg*S.transpose(),S*f.R,S*f.Rf,J*P*J.transpose());
  const auto initial=a->_imu->value();
  const Eigen::Vector3d force=a->_imu->Rot()*f.g;
  Eigen::Vector3d am=a->_imu->bias_a()+A.inverse()*force,wm=a->_imu->bias_g()+f.Tg*force;
  if(!zero_residual){am+=Eigen::Vector3d(.003,-.002,.001);wm+=Eigen::Vector3d(.0002,-.0001,.0001);}
  NoiseManager noise;noise.sigma_a=.03;noise.sigma_w=.003;noise.sigma_ab=.001;noise.sigma_wb=.0001;
  UpdaterOptions opts;opts.chi2_multipler=1;
  auto dba=std::make_shared<ov_core::FeatureDatabase>(),dbb=std::make_shared<ov_core::FeatureDatabase>();
  auto prop=std::make_shared<Propagator>(noise,9.81);
  UpdaterZeroVelocity ua(opts,noise,dba,prop,9.81,.03,1,0),ub(opts,noise,dbb,prop,9.81,.03,1,0);
  for(int i=0;i<=10;++i){ov_core::ImuData sample;sample.timestamp=1+i*.01;sample.am=am;sample.wm=wm;ua.feed_imu(sample);ub.feed_imu(sample);}
  const bool accepted_a=ua.try_update(a,1.1),accepted_b=ub.try_update(b,1.1);
  check(accepted_a&&accepted_b,"real ZUPT startup update accepts in RPNG and equivalent Kalibr gauge");
  if(accepted_a&&accepted_b){
    check((b->_imu->Rot()-S*a->_imu->Rot()).norm()<1e-11,"real ZUPT corrected attitude is gauge equivalent");
    check((b->_imu->value().block<12,1>(4,0)-a->_imu->value().block<12,1>(4,0)).norm()<1e-11,"real ZUPT raw biases/global position/velocity are gauge invariant");
    check((StateHelper::get_full_covariance(b)-J*StateHelper::get_full_covariance(a)*J.transpose()).norm()<1e-11,
          "real ZUPT posterior and cross-covariance are gauge equivalent");
    check(a->_timestamp==1.1&&b->_timestamp==1.1,"accepted startup ZUPT advances both clocks");
    if(zero_residual)check((a->_imu->value()-initial).norm()<1e-12,"calibrated stationary seed does not drift on first ZUPT");
    else check((a->_imu->value()-initial).norm()>1e-5,"nonzero production fixture actually updates the state");
  }
}
void invalid_checks(Fixture f) {
  Model model;Eigen::Matrix3d bad=f.A;bad.row(2).setZero();check(!model.set_calibration(bad,f.Tg),"singular A refused");
  bad=f.A;bad(0,0)=std::numeric_limits<double>::quiet_NaN();check(!model.set_calibration(bad,f.Tg),"NaN A refused under fast math");
  bad=f.Tg;bad(1,2)=std::numeric_limits<double>::infinity();check(!model.set_calibration(f.A,bad),"infinite Tg refused under fast math");
  check(model.set_calibration(f.A,f.Tg),"valid model restored");Model::Jacobian H;Model::Residual r;
  check(!model.linearize(f.am,f.wm,f.bg,f.ba,f.R*f.g,f.Rf*f.g,0,f.wa,H,r),"zero whitening weight refused");
  f.am(0)=std::numeric_limits<double>::quiet_NaN();
  check(!model.linearize(f.am,f.wm,f.bg,f.ba,f.R*f.g,f.Rf*f.g,f.ww,f.wa,H,r),"nonfinite measurement refused");
}
void identity_update_receipt(const Fixture &f,const char *path) {
  const Eigen::MatrixXd P=.01*Eigen::MatrixXd::Identity(15,15);
  auto state=state_for_update(false,Eigen::Matrix3d::Identity(),Eigen::Matrix3d::Identity(),Eigen::Matrix3d::Zero(),f.R,f.Rf,P);
  NoiseManager noise;noise.sigma_a=.03;noise.sigma_w=.003;noise.sigma_ab=.001;noise.sigma_wb=.0001;
  UpdaterOptions opts;opts.chi2_multipler=1;
  auto db=std::make_shared<ov_core::FeatureDatabase>();auto prop=std::make_shared<Propagator>(noise,9.81);
  UpdaterZeroVelocity update(opts,noise,db,prop,9.81,.03,1,0);
  const Eigen::Vector3d am=state->_imu->bias_a()+state->_imu->Rot()*f.g+Eigen::Vector3d(.003,-.002,.001);
  const Eigen::Vector3d wm=state->_imu->bias_g()+Eigen::Vector3d(.0002,-.0001,.0001);
  for(int i=0;i<=10;++i){ov_core::ImuData sample;sample.timestamp=1+i*.01;sample.am=am;sample.wm=wm;update.feed_imu(sample);}
  check(update.try_update(state,1.1),"identity production ZUPT accepts legacy parity fixture");
  if(path){
    std::ofstream file(path,std::ios::binary);const auto value=state->_imu->value(),cov=StateHelper::get_full_covariance(state);
    file.write(reinterpret_cast<const char *>(value.data()),value.size()*sizeof(double));
    file.write(reinterpret_cast<const char *>(cov.data()),cov.size()*sizeof(double));
    check(file.good(),"identity production state/covariance receipt written");
  }
}
} // namespace
int main(int argc,char **argv){ov_core::Printer::setPrintLevel("ERROR");Fixture f;jacobian_checks(f);identity_checks(f);whitening_and_gauge(f);
  production_update(f,false);production_update(f,true);invalid_checks(f);
  identity_update_receipt(f,argc>1?argv[1]:nullptr);
  std::printf("ZERO_VELOCITY_INTRINSICS %s failures=%d\n",failures?"FAIL":"PASS",failures);return failures?1:0;}
