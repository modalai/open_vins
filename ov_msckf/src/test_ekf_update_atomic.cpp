/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include "cam/CamRadtan.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Landmark.h"
#include "update/UpdaterZeroVelocity.h"
#include "utils/print.h"

#ifndef TEST_OLD_VOID_API
#define TEST_OLD_VOID_API 0
#endif
using namespace ov_core;
using namespace ov_msckf;
using namespace ov_type;
namespace {
using Mat=Eigen::MatrixXd;
using Vector=Eigen::VectorXd;
int checks=0,failures=0;
double max_covariance=0.,max_mean=0.;
void check(bool ok,const char*why){++checks;if(!ok){++failures;std::printf("FAIL: %s\n",why);}}
bool same(const Mat&a,const Mat&b){return a.rows()==b.rows()&&a.cols()==b.cols()&&std::memcmp(a.data(),b.data(),sizeof(double)*a.size())==0;}
double invalid(){volatile std::uint64_t raw=UINT64_C(0x7ff8000000000001);std::uint64_t bits=raw;double out;std::memcpy(&out,&bits,sizeof(out));return out;}
bool update(const std::shared_ptr<State>&s,const std::vector<std::shared_ptr<Type>>&order,const Mat&H,const Vector&r,const Mat&R){
#if TEST_OLD_VOID_API
  StateHelper::EKFUpdate(s,order,H,r,R);return true;
#else
  return StateHelper::EKFUpdate(s,order,H,r,R);
#endif
}
struct ProbeState:State {
  std::vector<std::shared_ptr<Type>> inventory;
  explicit ProbeState(StateOptions&o):State(o){
    inventory={_imu};
    for(const auto&v:std::vector<std::shared_ptr<Type>>{_calib_imu_dw,_calib_imu_da,_calib_imu_tg,_calib_imu_GYROtoIMU,_calib_imu_ACCtoIMU})
      if(v->id()>=0)inventory.push_back(v);
    for(int c=0;c<o.num_cameras;++c)for(const auto&v:std::vector<std::shared_ptr<Type>>{cam_imu_dt_var(c),_calib_IMUtoCAM.at(c),_cam_intrinsics.at(c),_calib_camera_readout.at(c)})
      if(v->id()>=0)inventory.push_back(v);
    std::sort(inventory.begin(),inventory.end(),[](const auto&a,const auto&b){return a->id()<b->id();});
  }
  const std::vector<std::shared_ptr<Type>>&variables()const{return inventory;}
};
struct Snapshot {
  Mat covariance;std::vector<Mat>values,fej;std::vector<int>ids;std::vector<const Type*>objects;
  explicit Snapshot(const std::shared_ptr<ProbeState>&s):covariance(StateHelper::get_full_covariance(s)){
    for(const auto&v:s->variables()){values.push_back(v->value());fej.push_back(v->fej());ids.push_back(v->id());objects.push_back(v.get());}
  }
  bool unchanged(const std::shared_ptr<ProbeState>&s)const{
    if(!same(covariance,StateHelper::get_full_covariance(s))||values.size()!=s->variables().size())return false;
    for(size_t i=0;i<values.size();++i){const auto&v=s->variables()[i];if(v.get()!=objects[i]||v->id()!=ids[i]||!same(v->value(),values[i])||!same(v->fej(),fej[i]))return false;}
    return true;
  }
};
std::shared_ptr<ProbeState> make_state(bool rich=false){
  StateOptions o;o.do_fej=false;o.do_calib_camera_pose=rich;o.do_calib_camera_intrinsics=rich;
  o.do_calib_camera_timeoffset=rich;o.do_calib_imu_intrinsics=rich;o.do_calib_imu_g_sensitivity=rich;
  auto s=std::make_shared<ProbeState>(o);
  Vector intr(8);intr<<400,405,320,240,.01,-.001,.002,-.003;
  s->_cam_intrinsics.at(0)->set_value(intr);s->_cam_intrinsics.at(0)->set_fej(intr);
  auto cam=std::make_shared<CamRadtan>(640,480);cam->set_value(intr);s->_cam_intrinsics_cameras[0]=cam;
  const int n=s->max_covariance_size();Mat L=Mat::Identity(n,n);
  for(int i=0;i<n;++i)for(int j=0;j<i;++j)L(i,j)=.08*std::sin(.4*i+.7*j);
  StateHelper::set_initial_covariance(s,.2*L*L.transpose(),s->variables());return s;
}
Vector retract_oracle(const Mat&value,const Vector&dx){
  if(value.rows()==dx.rows())return value+dx;
  // Independent JPL product written as vector/scalar operations.
  Eigen::Vector4d dq;dq<<.5*dx.head<3>(),1.;dq.normalize();
  const Eigen::Vector3d a=dq.head<3>(),b=value.topRows(3);const double sa=dq(3),sb=value(3);
  Eigen::Vector4d q;q.head<3>()=sa*b+sb*a-a.cross(b);q(3)=sa*sb-a.dot(b);
  if(q(3)<0.)q=-q;q.normalize();Vector out=value;out.head<4>()=q;
  if(dx.rows()>3)out.tail(dx.rows()-3)+=dx.tail(dx.rows()-3);return out;
}
void valid_case(int mode){
  auto s=make_state(true);const int n=s->max_covariance_size(),m=7;
  std::vector<std::shared_ptr<Type>> order=s->variables();std::reverse(order.begin(),order.end());
  Mat Hd(m,n);for(int i=0;i<m;++i)for(int j=0;j<n;++j)Hd(i,j)=.1*std::sin(.2+.81*i+.43*j)+(i==j?.8:0.);
  Mat H(m,n);int c=0;for(const auto&v:order){H.middleCols(c,v->size())=Hd.middleCols(v->id(),v->size());c+=v->size();}
  Mat R=.05*Mat::Identity(m,m);
  if(mode==1){Mat L=Mat::Identity(m,m);for(int i=1;i<m;++i)L(i,i-1)=.25;R=.05*L*L.transpose();}
  if(mode==2)R.setZero();
  Vector r(m);for(int i=0;i<m;++i)r(i)=.003*std::cos(.4*i);
  const Mat P=StateHelper::get_full_covariance(s),S=Hd*P*Hd.transpose()+R;
  const Mat K=S.ldlt().solve(Hd*P).transpose(),A=Mat::Identity(n,n)-K*Hd;
  const Mat expected=A*P*A.transpose()+K*R*K.transpose();const Vector dx=K*r;
  Snapshot before(s);std::vector<Vector>means;for(const auto&v:s->variables())means.push_back(retract_oracle(v->value(),dx.segment(v->id(),v->size())));
  check(update(s,order,H,r,R),"valid diagonal, correlated or exact-constraint update succeeds");
  const double error=(expected-StateHelper::get_full_covariance(s)).cwiseAbs().maxCoeff();max_covariance=std::max(max_covariance,error);
  check(error<4e-13,"complete covariance agrees with independent dense Joseph oracle");
  for(size_t i=0;i<s->variables().size();++i){const auto&v=s->variables()[i];const double e=(v->value()-means[i]).cwiseAbs().maxCoeff();max_mean=std::max(max_mean,e);
    check(e<4e-13,"Type mean matches independent vector/JPL retraction");check(same(v->fej(),before.fej[i])&&v->id()==before.ids[i]&&v.get()==before.objects[i],"FEJ, IDs and object identity are preserved");}
  check(same(s->_cam_intrinsics.at(0)->value(),s->_cam_intrinsics_cameras.at(0)->get_value()),"online camera model receives committed intrinsic mean");
}
void invalid_case(int mode){
  auto s=make_state();Mat H=Mat::Zero(3,15),R=.1*Mat::Identity(3,3);H.middleCols(3,3).setIdentity();Vector r=Vector::Constant(3,.01);
  if(mode==0)H(0,0)=invalid();
  if(mode==1)r(0)=invalid();
  if(mode==2)R(0,0)=invalid();
  if(mode==3)R(0,0)=-.01;
  if(mode==4){H.setZero();R.setZero();}
  if(mode==5)H(0,3)=1e308;
  if(mode==6){R(0,1)=.01;}
  if(mode==7){R(0,1)=R(1,0)=.11;}
  if(mode==8){H*=.01;R*=1e-8;r.setConstant(1e308);}
  if(mode==9){auto x=s->_imu->value();x(4)=1.7e308;s->_imu->set_value(x);r.setConstant(1e308);}
  if(mode==10){H.setZero();H.leftCols(3).setIdentity();r.setConstant(1e308);}
  if(mode==11){auto x=s->_imu->value();x(4)=invalid();s->_imu->set_value(x);}
  // These two corrupted finite priors force failure after the triangular
  // write, proving lower-triangle rollback rather than only preflight guards.
  if(mode==14){Mat P=StateHelper::get_full_covariance(s);P(6,6)=-1.;StateHelper::set_initial_covariance(s,P,{s->_imu});}
  if(mode==15){Mat P=StateHelper::get_full_covariance(s);P(6,3)=P(3,6)=1e308;StateHelper::set_initial_covariance(s,P,{s->_imu});R*=20.;r.setZero();}
  std::vector<std::shared_ptr<Type>> order{s->_imu};
  if(mode==12){auto foreign=std::make_shared<Vec>(15);foreign->set_local_id(0);order={foreign};}
  if(mode==13){auto foreign=std::make_shared<Vec>(3);foreign->set_local_id(s->_imu->p()->id());order={foreign};H=Mat::Identity(3,3);}
  const Snapshot before(s);
  check(!update(s,order,H,r,R),"invalid factor, ownership, posterior or mean proposal is rejected");
  check(before.unchanged(s),"failed update preserves complete covariance, all means, FEJ and ownership bytes");
}
void invalid_camera(){
  auto s=make_state(true);const Snapshot before(s);Mat H=Mat::Zero(2,8);H.topLeftCorner(2,2).setIdentity();
  check(!update(s,{s->_cam_intrinsics.at(0)},H,Vector::Constant(2,-2000.),.01*Mat::Identity(2,2)),"negative proposed focal length is rejected before updating cached camera model");
  check(before.unchanged(s)&&same(s->_cam_intrinsics.at(0)->value(),s->_cam_intrinsics_cameras.at(0)->get_value()),"invalid camera proposal leaves both Type and projection model unchanged");
}
void delayed_tail_failure(){
  auto s=make_state();auto x=s->_imu->value();x(4)=invalid();s->_imu->set_value(x);const Snapshot before(s);
  auto v=std::make_shared<Vec>(3);Vector mean(3);mean<<.1,.2,.3;v->set_value(mean);v->set_fej(mean);
  Mat Hx=Mat::Zero(4,3),Hf=Mat::Zero(4,3),R=.1*Mat::Identity(4,4);Hf.topRows(3).setIdentity();Hx(3,0)=1.;Vector r=Vector::Zero(4);
  check(!StateHelper::initialize(s,v,{s->_imu->bg()},Hx,Hf,R,r,1.),"actual delayed initializer rejects failed residual update");
  check(before.unchanged(s)&&v->id()==-1&&same(v->value(),mean)&&same(v->fej(),mean),"failed post-augmentation update restores proposal and prior ownership/covariance");
}
void zupt_failure(){
  auto s=make_state();s->_timestamp=1.;NoiseManager noise;noise.sigma_w=.003;noise.sigma_a=.03;noise.sigma_wb=.0001;noise.sigma_ab=.001;
  UpdaterOptions options;options.chi2_multipler=1.;auto db=std::make_shared<FeatureDatabase>();auto prop=std::make_shared<Propagator>(noise,9.81);
  for(int id=0;id<25;++id)for(double t:{1.,1.1,1.2,1.3})db->update_feature(id,t,0,100+id,100,0,0);
  UpdaterZeroVelocity zupt(options,noise,db,prop,9.81,.03,1.,.1);
  for(int i=0;i<=31;++i){ImuData d;d.timestamp=1.+.01*i;d.wm.setZero();d.am<<0,0,9.81;zupt.feed_imu(d);}
  check(zupt.try_update(s,1.1)&&zupt.try_update(s,1.2),"actual ZUPT valid prefix establishes two accepted events");
  auto x=s->_imu->value();x(4)=invalid();s->_imu->set_value(x);const Snapshot before(s);const double endpoint=s->imu_endpoint();
  const auto old_history=zupt.capture();
  check(!zupt.try_update(s,1.3),"actual ZUPT reports a backend proposal failure");
  check(before.unchanged(s)&&s->_timestamp==1.2&&s->imu_endpoint()==endpoint,"failed ZUPT rolls back bias random walk and retains endpoint");
  const auto feature=db->get_feature(0);check(feature&&std::find(feature->timestamps.at(0).begin(),feature->timestamps.at(0).end(),1.2)!=feature->timestamps.at(0).end(),"failed ZUPT does not delete prior accepted image measurements");
  const auto history=zupt.capture();check(history.last_zupt_count==0&&history.last_zupt_state_timestamp==0.&&history.last_prop_time_offset==old_history.last_prop_time_offset,"failure breaks stationarity streak without advancing offset");
}
}
int main(int argc,char**argv){Printer::setPrintLevel("ERROR");const std::string selected=argc>1?argv[1]:"all";
  if(selected=="all"||selected=="valid")for(int m=0;m<3;++m)valid_case(m);
  for(int m=0;m<16;++m)if(selected=="all"||selected=="invalid"+std::to_string(m))invalid_case(m);
  if(selected=="all"||selected=="camera")invalid_camera();
  if(selected=="all"||selected=="delayed")delayed_tail_failure();
  if(selected=="all"||selected=="zupt")zupt_failure();
  check(checks>0,"selected test case exists");std::printf("EKF_UPDATE_ATOMIC %s checks=%d failures=%d max_cov=%.12g max_mean=%.12g\n",failures?"FAIL":"PASS",checks,failures,max_covariance,max_mean);return failures?1:0;
}
