/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include "state/State.h"
#include "state/StateHelper.h"
#include "state/Propagator.h"
#include "update/UpdaterSLAM.h"
#include "types/Landmark.h"
#include "utils/print.h"

#ifndef TEST_OLD_VOID_API
#define TEST_OLD_VOID_API 0
#endif
using namespace ov_core;
using namespace ov_msckf;
using namespace ov_type;
namespace {
using Mat=Eigen::MatrixXd;
int checks=0,failures=0;
struct ProbePropagator : Propagator {
  using Propagator::Propagator;
  bool cache_valid() const { return cache_imu_valid.load(); }
  Eigen::Matrix<double,16,1> cached_mean() const { return cache_state_est; }
  Eigen::Matrix<double,15,15> cached_cov() const { return cache_state_covariance; }
  double cached_time() const { return cache_state_time; }
};
struct SlamProbe : UpdaterSLAM {
  using UpdaterSLAM::UpdaterSLAM;
  using UpdaterSLAM::perform_anchor_change;
};
void check(bool ok,const char *why){++checks;if(!ok){++failures;std::printf("FAIL: %s\n",why);}}
bool same(const Mat&a,const Mat&b){return a.rows()==b.rows()&&a.cols()==b.cols()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0;}
double stored(std::uint64_t bits){volatile std::uint64_t source=bits;bits=source;double value;std::memcpy(&value,&bits,8);return value;}
bool propagate(const std::shared_ptr<State>&s,const std::vector<std::shared_ptr<Type>>&next,
               const std::vector<std::shared_ptr<Type>>&old,const Mat&Phi,const Mat&Q){
#if TEST_OLD_VOID_API
  StateHelper::EKFPropagation(s,next,old,Phi,Q);return true;
#else
  return StateHelper::EKFPropagation(s,next,old,Phi,Q);
#endif
}
struct Fixture {
  std::shared_ptr<State>s;
  std::vector<std::shared_ptr<Type>> next,old,all;
  Mat P,Phi,Q;
  Fixture(){
    StateOptions o;o.num_cameras=2;o.do_calib_camera_timeoffset=true;s=std::make_shared<State>(o);
    all={s->_imu,s->cam_imu_dt_var(0),s->cam_imu_dt_var(1)};
    next={s->_imu->bg(),s->_imu->ba()};old={s->_imu->p(),s->_imu->ba(),s->_imu->bg(),s->cam_imu_dt_var(1)};
    const int n=s->max_covariance_size();Mat L=Mat::Identity(n,n);
    for(int i=0;i<n;++i)for(int j=0;j<i;++j)L(i,j)=.04*std::sin(.8*i+.5*j);
    P=.2*L*L.transpose();StateHelper::set_initial_covariance(s,P,all);
    Phi.resize(6,10);for(int i=0;i<6;++i)for(int j=0;j<10;++j)Phi(i,j)=.2*std::sin(.7+i+.4*j);
    Mat A(6,2);for(int i=0;i<6;++i){A(i,0)=.1*std::sin(i+.5);A(i,1)=.1*std::cos(.7*i);}
    Q=A*A.transpose();
  }
};
void valid_case(int mode){
  Fixture f;if(mode==1)f.Q.setZero();if(mode==2)f.Q=.1*Mat::Identity(6,6);
  const int n=f.P.rows();Mat T=Mat::Identity(n,n);T.middleRows(9,6).setZero();int c=0;
  for(const auto&v:f.old){T.block(9,v->id(),6,v->size())+=f.Phi.middleCols(c,v->size());c+=v->size();}
  Mat expected=T*f.P*T.transpose();expected.block(9,9,6,6)+=f.Q;
  const Mat mean=f.s->_imu->value(),fej=f.s->_imu->fej();
  check(propagate(f.s,f.next,f.old,f.Phi,f.Q),"valid correlated, zero and diagonal covariance propagation succeeds");
  const Mat actual=StateHelper::get_full_covariance(f.s);
  check((actual-expected).cwiseAbs().maxCoeff()<2e-13,"full covariance matches independent dense coordinate map");
  check(same(mean,f.s->_imu->value())&&same(fej,f.s->_imu->fej()),"covariance propagation does not change any mean or FEJ");
}
void invalid_case(int mode){
  Fixture f;StateOptions opt;auto foreign=std::make_shared<State>(opt);auto supplied=f.s;
  if(mode==0)f.Phi(0,0)=stored(UINT64_C(0x7ff8000000000001));
  if(mode==1)f.Q(0,0)=stored(UINT64_C(0x7ff0000000000000));
  if(mode==2)f.Q=-Mat::Identity(6,6);
  if(mode==3)f.Q(0,1)+=.1;
  if(mode==4){f.Q=Mat::Identity(6,6);f.Q(0,1)=f.Q(1,0)=2.;}
  if(mode==5)f.old[0]=foreign->_imu->p();
  if(mode==6)f.next[0]=foreign->_imu->bg();
  if(mode==7){auto stale=std::make_shared<Vec>(3);stale->set_local_id(3);f.old[0]=stale;}
  if(mode==8)f.next[0]=f.s->_imu->p();
  if(mode==9)f.Phi.conservativeResize(6,9);
  if(mode==10)supplied.reset();
  if(mode==11)f.old[0].reset();
  if(mode==12)f.next.clear();
  if(mode==13)f.Phi.setConstant(1e308);
  if(mode==14)f.next[0].reset();
  const Mat prior=StateHelper::get_full_covariance(f.s),mean=f.s->_imu->value(),fej=f.s->_imu->fej(),other=foreign->_imu->value();
  check(!propagate(supplied,f.next,f.old,f.Phi,f.Q),"invalid ownership, dimensions or arithmetic is refused");
  check(same(prior,StateHelper::get_full_covariance(f.s)),"refusal preserves complete prior covariance");
  check(same(mean,f.s->_imu->value())&&same(fej,f.s->_imu->fej())&&same(other,foreign->_imu->value()),"refusal preserves both owners and FEJ");
}
void endpoint_case(int method,bool corrupt){
  StateOptions o;o.do_fej=false;o.integration_method=static_cast<StateOptions::IntegrationMethod>(method);
  auto s=std::make_shared<State>(o);s->_timestamp=3.;s->_imu_endpoint=3.;s->_imu_endpoint_valid=true;
  NoiseManager noise;ProbePropagator p(noise,9.81);
  for(int i=0;i<4;++i){ImuData d;d.timestamp=3.+.01*i;d.wm<<.2,-.1,.3;d.am<<.1,.2,9.81;
    if(corrupt&&i==2)d.wm.x()=1e250;p.feed_imu(d,-1.);}
  const Mat mean=s->_imu->value(),fej=s->_imu->fej(),prior=StateHelper::get_full_covariance(s);
  Propagator::EndpointKinematics out;out.omega.setConstant(13.);out.omega_fej.setConstant(14.);
  const auto before=p.capture();const bool cache_before=p.cache_valid();
  const bool accepted=p.propagate_to_imu(s,3.03,2.9,out);
  check(accepted!=corrupt,"covered finite propagation succeeds; arithmetic overflow refuses");
  if(corrupt){
    check(same(mean,s->_imu->value())&&same(fej,s->_imu->fej())&&same(prior,StateHelper::get_full_covariance(s)),"failed multi-step propagation rolls back navigation and covariance");
    check(s->_timestamp==3.&&s->imu_endpoint()==3.&&s->_imu_endpoint_valid,"failure preserves accepted camera and IMU clocks");
    check((out.omega.array()==13.).all()&&(out.omega_fej.array()==14.).all(),"failure preserves caller endpoint output");
    const auto after=p.capture();
    check(before.have_last_prop_time_offset==after.have_last_prop_time_offset&&before.last_prop_time_offset==after.last_prop_time_offset&&cache_before==p.cache_valid(),"failure preserves propagation clock and cache metadata");
  }else check(s->_timestamp==2.9&&s->imu_endpoint()==3.03,"successful propagation commits exact endpoint and reference label");
}
void anchor_case(bool single,bool bad){
  StateOptions o;o.do_fej=true;o.max_clone_size=1;auto s=std::make_shared<State>(o);
  const Eigen::Vector3d zero=Eigen::Vector3d::Zero();s->_timestamp=1.;
  StateHelper::augment_clone(s,zero,zero);
  Mat pose=s->_imu->value();pose.block<3,1>(4,0)<<1.,2.,3.;s->_imu->set_value(pose);s->_imu->set_fej(pose);
  s->_timestamp=2.;StateHelper::augment_clone(s,zero,zero);
  const int d=single?1:3;auto l=std::make_shared<Landmark>(d);
  l->_feat_representation=single?LandmarkRepresentation::ANCHORED_INVERSE_DEPTH_SINGLE:LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
  l->_featid=47;l->_anchor_cam_id=0;l->_anchor_clone_timestamp=1.;
  const Eigen::Vector3d xyz(1.,2.,bad?3.:5.);l->set_from_xyz(xyz,false);l->set_from_xyz(xyz,true);
  check(StateHelper::initialize_invertible(s,l,{s->_imu},Mat::Zero(d,15),Mat::Identity(d,d),.01*Mat::Identity(d,d),Eigen::VectorXd::Zero(d)),"anchor fixture has a valid finite owned landmark");
  s->_features_SLAM[47]=l;UpdaterOptions a,b;FeatureInitializerOptions f;SlamProbe u(a,b,f);
  const Mat prior=StateHelper::get_full_covariance(s),mean=l->value(),fej=l->fej();
  Eigen::Vector3d bearing=Eigen::Vector3d::Zero(),fej_bearing=Eigen::Vector3d::Zero();
  if(single){bearing=l->uv_norm_zero;fej_bearing=l->uv_norm_zero_fej;}
#if TEST_OLD_VOID_API
  u.perform_anchor_change(s,l,2.,0);const bool accepted=true;
#else
  const bool accepted=u.perform_anchor_change(s,l,2.,0);
#endif
  check(accepted!=bad,"singular new inverse-depth chart is refused; valid chart succeeds");
  if(bad){
    check(same(prior,StateHelper::get_full_covariance(s))&&same(mean,l->value())&&same(fej,l->fej()),"failed reanchor preserves full covariance and current/FEJ coordinates");
    check(l->_anchor_clone_timestamp==1.&&!l->has_had_anchor_change&&(!single||(same(bearing,l->uv_norm_zero)&&same(fej_bearing,l->uv_norm_zero_fej))),"failed reanchor retains old anchor identity and both bearings");
    u.change_anchors(s);
    check(s->_features_SLAM.empty()&&s->max_covariance_size()==prior.rows()-d,"retirement removes a landmark whose reparameterization failed");
  }else {
    check(l->_anchor_clone_timestamp==2.&&l->has_had_anchor_change,"valid reanchor commits new owner metadata");
    check((l->get_xyz(false)+pose.block<3,1>(4,0)-xyz).norm()<1e-13,"valid reanchor preserves the world point");
  }
}
void cached_case(bool warm){
  StateOptions o;auto s=std::make_shared<State>(o);s->_timestamp=3.;s->_imu_endpoint=3.;s->_imu_endpoint_valid=true;
  NoiseManager noise;ProbePropagator p(noise,9.81);
  for(int i=0;i<4;++i){ImuData d;d.timestamp=3.+.01*i;d.wm<<.2,-.1,.3;d.am<<.1,.2,9.81;if(i==2)d.wm.x()=1e250;p.feed_imu(d,-1.);}
  Eigen::Matrix<double,13,1> output=Eigen::Matrix<double,13,1>::Constant(17.);
  Eigen::Matrix<double,12,12> covariance=Eigen::Matrix<double,12,12>::Identity();
  if(warm)check(p.fast_state_propagate(s,3.01,output,covariance),"cache prefix prediction succeeds");
  const auto old_output=output;const auto old_cov=covariance;
  Eigen::Matrix<double,16,1> cache_mean=Eigen::Matrix<double,16,1>::Zero();
  Eigen::Matrix<double,15,15> cache_cov=Eigen::Matrix<double,15,15>::Zero();
  double time=0.;if(warm){cache_mean=p.cached_mean();cache_cov=p.cached_cov();time=p.cached_time();}
  const bool valid=p.cache_valid();const Mat state_mean=s->_imu->value(),prior=StateHelper::get_full_covariance(s);
  check(!p.fast_state_propagate(s,3.03,output,covariance),"cached output rejects a finite-signal arithmetic overflow");
  check(same(old_output,output)&&same(old_cov,covariance),"failed prediction leaves published output unchanged");
  check(valid==p.cache_valid()&&(!warm||(time==p.cached_time()&&same(cache_mean,p.cached_mean())&&same(cache_cov,p.cached_cov()))),"failed prediction preserves the last accepted cache");
  check(same(state_mean,s->_imu->value())&&same(prior,StateHelper::get_full_covariance(s)),"prediction never mutates the live filter");
}
}
int main(int argc,char**argv){
  Printer::setPrintLevel("ERROR");const std::string selected=argc>1?argv[1]:"all";
  for(int i=0;i<3;++i)if(selected=="all"||selected=="valid")valid_case(i);
  for(int i=0;i<15;++i)if(selected=="all"||selected=="invalid"+std::to_string(i))invalid_case(i);
  for(int m:{int(StateOptions::DISCRETE),int(StateOptions::ANALYTICAL),int(StateOptions::RK4)})
    for(bool bad:{false,true})if(selected=="all"||selected=="endpoint"+std::to_string(m)+(bad?"bad":"good"))endpoint_case(m,bad);
  for(bool single:{false,true})for(bool bad:{false,true})
    if(selected=="all"||selected=="anchor"+std::to_string(single)+(bad?"bad":"good"))anchor_case(single,bad);
  for(bool warm:{false,true})if(selected=="all"||selected=="cache"+std::to_string(warm))cached_case(warm);
  check(checks>0,"requested case exists");std::printf("PROPAGATION_ATOMIC %s checks=%d failures=%d\n",failures?"FAIL":"PASS",checks,failures);return failures?1:0;
}
