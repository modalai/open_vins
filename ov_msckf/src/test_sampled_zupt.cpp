/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "state/Propagator.h"
#include "state/StateHelper.h"
#include "update/UpdaterZeroVelocity.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

using namespace ov_msckf;
using namespace ov_core;
namespace {
using Mat=Eigen::MatrixXd; using Vec=Eigen::VectorXd; using V3=Eigen::Vector3d; using M3=Eigen::Matrix3d;
using V6=Eigen::Matrix<double,6,1>; using M6=Eigen::Matrix<double,6,6>;
using Record=State::SampledImuRecord; using Factor=UpdaterZeroVelocity::SampledStationaryFactor;
int checks=0,failures=0; double max_fd=0.,max_cov=0.,max_mean=0.;
void check(bool ok,const char *why) { ++checks; if(!ok){++failures;std::printf("FAIL: %s\n",why);} }
template<class A,class B> bool same(const Eigen::MatrixBase<A>&a,const Eigen::MatrixBase<B>&b) {
  const Mat aa=a,bb=b; return aa.rows()==bb.rows()&&aa.cols()==bb.cols()&&std::memcmp(aa.data(),bb.data(),aa.size()*sizeof(double))==0;
}
struct Probe:Propagator {
  using Propagator::Propagator;
  void seed_cache(){cache_imu_valid=true;cache_state_time=913.;}
  bool cache_valid()const{return cache_imu_valid.load();}
};
NoiseManager noise(){NoiseManager n;n.sigma_w=.003;n.sigma_a=.04;n.sigma_wb=.015;n.sigma_ab=.03;return n;}
std::vector<std::shared_ptr<ov_type::Type>> variables(const std::shared_ptr<State>&s) {
  std::vector<std::shared_ptr<ov_type::Type>> out{s->_imu,s->cam_imu_dt_var(0),s->cam_imu_dt_var(1)};
  for(const auto &v:s->sampled_imu_slots())if(v.noise)out.push_back(v.noise);
  for(const auto &v:s->_exposure_poses)out.push_back(v.pose);
  std::sort(out.begin(),out.end(),[](const auto&a,const auto&b){return a->id()<b->id();});return out;
}
std::shared_ptr<State> state(StateOptions::IntegrationMethod method=StateOptions::ANALYTICAL,
                             StateOptions::ImuModel model=StateOptions::RPNG,bool calibrated=true) {
  StateOptions o;o.integration_method=method;o.imu_model=model;o.do_fej=false;o.num_cameras=2;o.physical_camera_clones=true;
  o.do_calib_camera_timeoffset=true;o.max_clone_size=6;o.configure_clone_policy(false,false);
  auto s=std::make_shared<State>(o);s->_timestamp=0.;s->_imu_endpoint=0.;s->_imu_endpoint_valid=true;
  if(calibrated) {
    V6 dw,da;if(model==StateOptions::RPNG){dw<<1.07,.018,.94,-.012,.021,1.025;da<<.97,-.024,1.055,.016,-.019,1.02;}
    else{dw<<1.07,.018,-.012,.94,.021,1.025;da<<.97,-.024,.016,1.055,-.019,1.02;}
    s->_calib_imu_dw->set_value(dw);s->_calib_imu_da->set_value(da);
    M3 tg;tg<<.028,-.014,.013,.012,.027,-.015,-.016,.011,.024;
    Eigen::Matrix<double,9,1> tv;tv<<tg.col(0),tg.col(1),tg.col(2);s->_calib_imu_tg->set_value(tv);
    auto q=model==StateOptions::RPNG?s->_calib_imu_ACCtoIMU:s->_calib_imu_GYROtoIMU;
    q->set_value(rot_2_quat(exp_so3(V3(.16,-.11,.07))));
    Vec x=s->_imu->value();x.head<4>()=rot_2_quat(exp_so3(V3(.31,-.22,.18)));
    x.segment<3>(10)<<.013,-.021,.009;x.segment<3>(13)<<.035,-.026,.019;s->_imu->set_value(x);s->_imu->set_fej(x);
  }
  check(StateHelper::prepare_sampled_imu_boundary(s,73),"fixture prepares bounded sampled owners");return s;
}
struct Calibration {
  M3 A,W,T;
  Calibration(const std::shared_ptr<State>&s){A=s->_calib_imu_ACCtoIMU->Rot()*State::Dm(s->_options.imu_model,s->_calib_imu_da->value());
    W=s->_calib_imu_GYROtoIMU->Rot()*State::Dm(s->_options.imu_model,s->_calib_imu_dw->value());T=State::Tg(s->_calib_imu_tg->value());}
};
Record record(const std::shared_ptr<State>&s,uint64_t seq,double time,bool calibrated=true) {
  Record r;r.stream_episode=73;r.sequence=seq;r.timestamp=time;Calibration c(s);
  V3 force=s->_imu->Rot()*V3(0,0,calibrated?9.81:0.);
  r.measured.head<3>()=s->_imu->bias_g()+c.T*force;r.measured.tail<3>()=s->_imu->bias_a()+c.A.inverse()*force;
  if(calibrated) {
    r.measured+=(V6()<<.004,-.003,.002,.015,-.007,.009).finished();
    M6 L=M6::Identity()*.2;for(int i=0;i<6;++i)for(int j=0;j<i;++j)L(i,j)=.012*(i+j+1);r.prior=L*L.transpose();
  } else {r.measured(3)+=.025+.003*seq;r.prior=M6::Identity()*.05;}
  return r;
}
std::shared_ptr<ov_type::Vec> owner(const std::shared_ptr<State>&s,uint64_t sequence) {
  for(const auto &slot:s->sampled_imu_slots())if(slot.active&&slot.record.sequence==sequence)return slot.noise;
  return nullptr;
}
std::shared_ptr<UpdaterZeroVelocity> updater(const std::shared_ptr<Probe>&p,NoiseManager n=noise(),double gravity=9.81,double multiplier=1.) {
  UpdaterOptions o;o.chi2_multipler=1.;return std::make_shared<UpdaterZeroVelocity>(o,n,std::make_shared<FeatureDatabase>(),p,gravity,1.,multiplier,1.);
}
V6 raw_constraint(const std::shared_ptr<State>&s,const Record&r,double gravity=9.81) {
  Calibration c(s);V6 h=r.measured-owner(s,r.sequence)->value();const V3 force=s->_imu->Rot()*V3(0,0,gravity);
  h.head<3>()-=s->_imu->bias_g()+c.T*force;h.tail<3>()-=s->_imu->bias_a()+c.A.inverse()*force;return h;
}
bool same_state(const std::shared_ptr<State>&a,const std::shared_ptr<State>&b) {
  if(!same(StateHelper::get_full_covariance(a),StateHelper::get_full_covariance(b)) ||
      initializer_time_bits(a->_timestamp)!=initializer_time_bits(b->_timestamp) ||
      initializer_time_bits(a->imu_endpoint())!=initializer_time_bits(b->imu_endpoint()) ||a->_imu_endpoint_valid!=b->_imu_endpoint_valid)return false;
  const auto av=variables(a),bv=variables(b);if(av.size()!=bv.size())return false;
  for(size_t i=0;i<av.size();++i)if(!same(av[i]->value(),bv[i]->value())||!same(av[i]->fej(),bv[i]->fej()))return false;
  for(int i=0;i<2;++i){const auto &x=a->sampled_imu_slots()[i],&y=b->sampled_imu_slots()[i];
    if(x.active!=y.active||x.record.sequence!=y.record.sequence||x.record.stream_episode!=y.record.stream_episode||
       !same(x.record.measured,y.record.measured)||!same(x.record.prior,y.record.prior))return false;}
  return true;
}
bool same_receipt(const UpdaterZeroVelocity::Snapshot&a,const UpdaterZeroVelocity::Snapshot&b) {
  return a.sampled_stream_episode==b.sampled_stream_episode&&a.sampled_last_sequence==b.sampled_last_sequence&&
    a.sampled_last_timestamp_bits==b.sampled_last_timestamp_bits&&a.last_zupt_count==b.last_zupt_count&&
    a.last_zupt_state_timestamp==b.last_zupt_state_timestamp&&a.last_prop_time_offset==b.last_prop_time_offset&&
    a.have_last_prop_time_offset==b.have_last_prop_time_offset&&a.camera_history.size()==b.camera_history.size();
}
void compare_mean(const std::shared_ptr<State>&s,const std::shared_ptr<State>&expected) {
  const auto a=variables(s),b=variables(expected);double error=0.;
  for(size_t i=0;i<a.size();++i)error=std::max(error,(a[i]->value()-b[i]->value()).norm());
  max_mean=std::max(max_mean,error);check(error<2e-9,"all state/sample/old-pose means equal independent conditioning and retraction");
}

void factor_and_conditioning() {
  for(auto method:{StateOptions::ANALYTICAL,StateOptions::DISCRETE})for(auto model:{StateOptions::RPNG,StateOptions::KALIBR})for(bool exact:{true,false}) {
    auto s=state(method,model);auto r=record(s,1,0.);auto next=record(s,2,.1);
    auto first=StateHelper::admit_sampled_imu_noise(s,r),second=StateHelper::admit_sampled_imu_noise(s,next);
    first->set_value((V6()<<.003,-.002,.001,.012,-.006,.004).finished());second->set_value(V6::Constant(.002));
    State::ExposurePose view;view.pose=StateHelper::augment_pose_view(s,1,V3(.1,-.2,.3));s->_exposure_poses.push_back(view);
    const int n=s->max_covariance_size();Mat L=Mat::Zero(n,n);
    for(int i=0;i<n;++i)for(int j=0;j<=i;++j)L(i,j)=i==j?.04+.001*i:.002*std::sin(i+.4*j);
    StateHelper::set_initial_covariance(s,L*L.transpose(),variables(s));
    auto p=std::make_shared<Probe>(noise(),9.81);auto z=updater(p);Factor f;
    M6 R=M6::Zero();if(!exact){M6 B=M6::Identity()*.008;B(4,1)=.002;R=B*B.transpose();}
    check(z->linearize_sampled_at_knot(s,r,R,f),"calibrated original-record factor linearizes at exact knot");
    check((f.H.leftCols<3>()*(s->_imu->Rot()*V3::UnitZ())).norm()<1e-12,
          "stationary raw likelihood adds no direct global-yaw information");
    // Resolve truncation and subtraction error independently of the analytic
    // H: three central steps must show second-order convergence. Richardson
    // removes that leading term without relaxing the derivative tolerance.
    auto central=[&](double eps){Mat result(6,15);
      for(int col=0;col<15;++col){auto a=StateHelper::clone_state(s),b=StateHelper::clone_state(s);Vec d=Vec::Zero(15);
        if(col<9){const int id=col<3?col:col+6;d(id)=eps;a->_imu->update(d);b->_imu->update(-d);}
        else{auto ap=owner(a,1),bp=owner(b,1);Vec av=ap->value(),bv=bp->value();av(col-9)+=eps;bv(col-9)-=eps;ap->set_value(av);bp->set_value(bv);}
        result.col(col)=(raw_constraint(a,r)-raw_constraint(b,r))/(2*eps);
      }return result;
    };
    const Mat coarse=central(4e-4),middle=central(2e-4),fine=central(1e-4);
    const double convergence=(coarse-middle).norm()/(middle-fine).norm();
    check(convergence>3.9&&convergence<4.1,"independent central derivatives exhibit second-order step convergence");
    const Mat fd=(4.*fine-middle)/3.;
    const double error=(f.H-fd).cwiseAbs().maxCoeff();max_fd=std::max(max_fd,error);
    std::printf("FD_CONVERGENCE method=%d model=%d exact_R=%d ratio=%.12g fine_error=%.12g extrapolated_error=%.12g\n",
                int(method),int(model),int(exact),convergence,(f.H-fine).cwiseAbs().maxCoeff(),error);
    check(error<2e-8 && (f.residual+raw_constraint(s,r)).norm()<1e-13,"raw posterior, JPL gravity and Tg columns match independent nonlinear finite differences");
    Mat H=Mat::Zero(6,n);H.leftCols<3>()=fd.leftCols<3>();H.middleCols<6>(9)=fd.middleCols<6>(3);H.middleCols(first->id(),6)=fd.rightCols<6>();
    const Mat P=StateHelper::get_full_covariance(s),S=H*P*H.transpose()+R,K=P*H.transpose()*S.inverse();const Vec residual=-raw_constraint(s,r);
    const Mat T=Mat::Identity(n,n)-K*H,expected_P=T*P*T.transpose()+K*R*K.transpose();const Vec delta=K*residual;
    auto expected=StateHelper::clone_state(s);for(const auto &v:variables(expected))v->update(delta.segment(v->id(),v->size()));
    auto missing_mean=StateHelper::clone_state(s),independent=StateHelper::clone_state(s),double_bias=StateHelper::clone_state(s),density_control=StateHelper::clone_state(s);
    auto double_sensor=StateHelper::clone_state(s);
    const auto before=StateHelper::clone_state(s);p->seed_cache();Factor returned;
    check(z->try_update_sampled_at_knot(s,r,R,&returned),"actual sampled ZUPT caller accepts common-noise stationary factor");
    const double covariance_error=(StateHelper::get_full_covariance(s)-expected_P).cwiseAbs().maxCoeff();max_cov=std::max(max_cov,covariance_error);
    check(covariance_error<2e-9,"actual full covariance equals independent dense common-noise conditioning");compare_mean(s,expected);
    check(s->_timestamp==before->_timestamp&&s->imu_endpoint()==before->imu_endpoint()&&!p->cache_valid()&&
      same(s->_imu->fej(),before->_imu->fej()),"accepted knot update changes no timestamp or FEJ and invalidates prediction cache");
    check(same(returned.independent_R,R),"raw sample prior is not copied into independent measurement R");
    auto altered=noise();altered.sigma_w=1e3;altered.sigma_a=2e3;altered.sigma_wb=300.;altered.sigma_ab=400.;
    auto densities=updater(p,altered,9.81,500.);
    check(densities->try_update_sampled_at_knot(density_control,r,R)&&same_state(density_control,s),
          "sensor density, legacy multiplier and bias RW parameters have no already-propagated sampled factor effect");
    owner(missing_mean,1)->set_value(V6::Zero());auto missing=updater(p);
    check(missing->try_update_sampled_at_knot(missing_mean,r,R)&&
      (missing_mean->_imu->value()-s->_imu->value()).norm()>1e-5,"negative actual caller omitting inferred raw-noise mean changes update");
    const std::vector<std::shared_ptr<ov_type::Type>> wrong_order{independent->_imu->q(),independent->_imu->bg(),independent->_imu->ba()};
    check(StateHelper::EKFUpdate(independent,wrong_order,f.H.leftCols(9),f.residual,R+r.prior),"independent-noise predecessor control forms a valid ordinary update");
    const double independence_error=(StateHelper::get_full_covariance(independent)-StateHelper::get_full_covariance(s)).norm();
    check(independence_error>1e-4,"negative independent-R predecessor loses common sample posterior/cross covariance");
    check(StateHelper::EKFUpdate(double_sensor,{double_sensor->_imu->q(),double_sensor->_imu->bg(),double_sensor->_imu->ba(),owner(double_sensor,1)},
                                f.H,f.residual,R+r.prior)&&
      (StateHelper::get_full_covariance(double_sensor)-StateHelper::get_full_covariance(s)).norm()>1e-4,
      "negative duplicate sensor R double counts an already admitted common sample");
    Mat biasQ=Mat::Identity(6,6)*.001;
    StateHelper::EKFPropagation(double_bias,{double_bias->_imu->bg(),double_bias->_imu->ba()},{double_bias->_imu->bg(),double_bias->_imu->ba()},Mat::Identity(6,6),biasQ);
    auto twice=updater(p);check(twice->try_update_sampled_at_knot(double_bias,r,R)&&
      (StateHelper::get_full_covariance(double_bias)-StateHelper::get_full_covariance(s)).norm()>1e-5,
      "negative extra bias random walk double counting changes the posterior");
  }
}

// Independent all-original-record Gaussian. In this zero-gravity fixture mean
// motion is along x and gyro means stay zero; the full attitude/bias covariance
// still evolves through a constant 15D generator, including cross terms.
struct Dense {
  Mat P;Vec mean;std::vector<Record> records;std::vector<int> pose_ids;
  Dense(const std::shared_ptr<State>&s,const std::vector<Record>&r):records(r){int n=17+6*r.size();P=Mat::Zero(n,n);mean=Vec::Zero(n);
    P.topLeftCorner(17,17)=StateHelper::get_full_covariance(s).topLeftCorner(17,17);
    mean.segment<12>(3)=s->_imu->value().block<12,1>(4,0);for(size_t i=0;i<r.size();++i)P.block<6,6>(17+6*i,17+6*i)=r[i].prior;}
  int raw(uint64_t seq)const{return 17+6*(seq-1);}
  void propagate(const Propagator::SampledImuSegment&seg,StateOptions::IntegrationMethod method){const double dt=seg.time1-seg.time0;
    const Eigen::Vector2d w=.5*(seg.weights0+seg.weights1);const V6 sampled=w(0)*(seg.records[0].measured-mean.segment<6>(raw(seg.records[0].sequence)))+
      w(1)*(seg.records[1].measured-mean.segment<6>(raw(seg.records[1].sequence)));
    const V3 a=sampled.tail<3>()-mean.segment<3>(12);Mat generator=Mat::Zero(15,15);
    generator.block<3,3>(0,9)=-M3::Identity();generator.block<3,3>(3,6).setIdentity();generator.block<3,3>(6,0)=-skew_x(a);generator.block<3,3>(6,12)=-M3::Identity();
    Mat Q=Mat::Zero(15,15);const auto n=noise();Q.block<3,3>(9,9)=n.sigma_wb*n.sigma_wb*M3::Identity();Q.block<3,3>(12,12)=n.sigma_ab*n.sigma_ab*M3::Identity();
    Mat F;
    if(method==StateOptions::ANALYTICAL){Mat V=Mat::Zero(30,30);V.topLeftCorner(15,15)=generator;V.topRightCorner(15,15)=Q;V.bottomRightCorner(15,15)=-generator.transpose();
      const Mat E=(dt*V).exp();F=E.topLeftCorner(15,15);Q=E.topRightCorner(15,15)*F.transpose();}
    else{F=Mat::Identity(15,15)+dt*generator;F.block<3,3>(3,0)=-.5*dt*dt*skew_x(a);F.block<3,3>(3,12)=-.5*dt*dt*M3::Identity();Q*=dt;}
    Mat B=F.middleCols(9,6);B.bottomRows(6).setZero();Mat T=Mat::Identity(P.rows(),P.cols());T.topRows(15).setZero();T.topLeftCorner(15,15)=F;
    T.block(0,raw(seg.records[0].sequence),15,6)=w(0)*B;T.block(0,raw(seg.records[1].sequence),15,6)=w(1)*B;
    P=(T*P*T.transpose()).eval();P.topLeftCorner(15,15)+=Q;
    mean.segment<3>(3)+=dt*mean.segment<3>(6)+.5*dt*dt*a;mean.segment<3>(6)+=dt*a;
  }
  void condition(const Mat&H,const Vec&res,const Mat&R){const Mat K=P*H.transpose()*(H*P*H.transpose()+R).inverse();
    mean+=K*res;const Mat T=Mat::Identity(P.rows(),P.cols())-K*H;P=(T*P*T.transpose()+K*R*K.transpose()).eval();}
  void stationary(const Record&r){Mat H=Mat::Zero(6,P.rows());H.middleCols<6>(9)=-M6::Identity();H.middleCols(raw(r.sequence),6)=-M6::Identity();
    condition(H,mean.segment<6>(9)+mean.segment<6>(raw(r.sequence))-r.measured,M6::Zero());}
  void clone(const std::shared_ptr<State>&s){const int n=P.rows();Mat T=Mat::Zero(n+6,n);T.topRows(n).setIdentity();T.bottomRows(6).leftCols(6).setIdentity();
    T.block<3,1>(n+3,15)=mean.segment<3>(6);P=(T*P*T.transpose()).eval();mean.conservativeResize(n+6);mean.segment<6>(n)=mean.head<6>();pose_ids.push_back(n);
    State::ExposurePose v;v.camera_id=0;v.imu_time=s->imu_endpoint();v.raw_time=s->_timestamp;v.pose=StateHelper::augment_pose_view(s,0,V3::Zero());s->_exposure_poses.push_back(v);}
  void visual(const std::shared_ptr<State>&s){Mat H=Mat::Zero(1,P.rows());H(0,pose_ids.front()+3)=1.;H(0,6)=.3;
    const Mat R=Mat::Constant(1,1,.0005);Vec residual=Vec::Constant(1,.008-(H*mean)(0));condition(H,residual,R);
    Mat local=Mat::Zero(1,9);local(0,3)=1.;local(0,6)=.3;
    check(StateHelper::EKFUpdate(s,{s->_exposure_poses.front().pose,s->_imu->v()},local,residual,R),"delayed pose constraint accepts via actual generic update");}
  void compare(const std::shared_ptr<State>&s){std::vector<int> map(s->max_covariance_size(),-1);for(int i=0;i<17;++i)map[i]=i;
    for(const auto &slot:s->sampled_imu_slots())if(slot.active)for(int j=0;j<6;++j)map[slot.noise->id()+j]=raw(slot.record.sequence)+j;
    for(size_t i=0;i<pose_ids.size();++i)for(int j=0;j<6;++j)map[s->_exposure_poses[i].pose->id()+j]=pose_ids[i]+j;
    Mat expected=Mat::Zero(map.size(),map.size());for(size_t i=0;i<map.size();++i)for(size_t j=0;j<map.size();++j)if(map[i]>=0&&map[j]>=0)expected(i,j)=P(map[i],map[j]);
    const double ce=(StateHelper::get_full_covariance(s)-expected).cwiseAbs().maxCoeff();max_cov=std::max(max_cov,ce);check(ce<3e-9,"sampled propagation/ZUPT/retirement/visual sequence equals all-raw dense oracle");
    double me=(s->_imu->value().block<12,1>(4,0)-mean.segment<12>(3)).norm();me=std::max(me,s->_imu->quat().head<3>().norm());
    for(const auto &slot:s->sampled_imu_slots())if(slot.active)me=std::max(me,(slot.noise->value()-mean.segment<6>(raw(slot.record.sequence))).norm());
    for(size_t i=0;i<pose_ids.size();++i)me=std::max(me,(s->_exposure_poses[i].pose->pos()-mean.segment<3>(pose_ids[i]+3)).norm());
    for(int i=0;i<2;++i)me=std::max(me,std::abs(s->cam_imu_dt_var(i)->value()(0)-mean(15+i)));
    max_mean=std::max(max_mean,me);check(me<3e-9,"nonzero raw-noise and bias means feed later actual sampled prediction consistently");}
};
void linked_sequence(){for(auto method:{StateOptions::ANALYTICAL,StateOptions::DISCRETE}){
  auto s=state(method,StateOptions::RPNG,false);Mat P=Mat::Identity(17,17)*.003;
  // Cross-correlate x position/velocity/bias and clock, preserving gyro-accel
  // mean separation so this independent scalar-mean oracle remains exact.
  Eigen::VectorXd v=Vec::Zero(17);v(3)=.04;v(6)=.03;v(12)=.02;v(15)=.01;P+=v*v.transpose();
  StateHelper::set_initial_covariance(s,P,{s->_imu,s->cam_imu_dt_var(0),s->cam_imu_dt_var(1)});
  std::vector<Record> records;for(int i=0;i<4;++i)records.push_back(record(s,i+1,.1*i,false));Dense dense(s,records);
  auto p=std::make_shared<Probe>(noise(),0.);auto z=updater(p,noise(),0.);StateHelper::admit_sampled_imu_noise(s,records[0]);
  dense.stationary(records[0]);check(z->try_update_sampled_at_knot(s,records[0],M6::Zero()),"first exact-knot sampled constraint uses original owner");dense.compare(s);
  const auto a=s->sampled_imu_slots()[0].noise,b=s->sampled_imu_slots()[1].noise;
  std::vector<Propagator::SampledImuSegment> segments;segments.reserve(4);double time=0.;
  for(double target:{.023,.057,records[1].timestamp,.14,records[2].timestamp,records[3].timestamp}){
    check(Propagator::select_sampled_imu_readings(records,time,target,segments),"actual selector retains original records across camera cuts");
    for(const auto &seg:segments){dense.propagate(seg,method);Propagator::EndpointKinematics endpoint;
      check(p->propagate_sampled_segment(s,seg,target,endpoint),"actual sampled propagation reaches requested camera/raw knot");dense.compare(s);}
    if(target<.1){dense.clone(s);dense.compare(s);}
    auto raw=std::find_if(records.begin(),records.end(),[&](const Record&r){return initializer_time_bits(r.timestamp)==initializer_time_bits(target);});
    if(raw!=records.end()){
      const auto previous=raw-1;Factor refused;refused.residual.setConstant(987.);
      const auto before=StateHelper::clone_state(s);const auto receipt=z->capture();
      check(!owner(s,previous->sequence)&&!z->linearize_sampled_at_knot(s,*previous,M6::Zero(),refused)&&
        !z->try_update_sampled_at_knot(s,*previous,M6::Zero())&&same_state(s,before)&&same_receipt(receipt,z->capture())&&
        refused.residual(0)==987.,"retired raw record cannot be reinterpreted at a newer bias time or reusable slot");
      dense.stationary(*raw);check(z->try_update_sampled_at_knot(s,*raw,M6::Zero()),"current right-knot ZUPT consumes retained correlated owner");dense.compare(s);
    }
    else if(target==.14){dense.visual(s);dense.compare(s);
      check(owner(s,3)&&owner(s,3)->value().norm()>1e-7,"delayed visual data infer a nonzero future-knot sample mean before its ZUPT");}
    check(s->sampled_imu_slots()[0].noise==a&&s->sampled_imu_slots()[1].noise==b,"ZUPT adds no raw owners and retirement reuses fixed coordinates");time=target;
  }
  check(s->max_covariance_size()==41,"two raw slots and two camera poses remain bounded through multiple ZUPTs");
}}

void rejection_and_receipts(){
  auto s=state();auto r=record(s,1,0.);StateHelper::admit_sampled_imu_noise(s,r);auto p=std::make_shared<Probe>(noise(),9.81);auto z=updater(p);
  Factor out;out.H.setConstant(123.);out.residual.setConstant(456.);out.independent_R.setConstant(789.);
  auto reject=[&](const Record&input,const M6&R){auto before=StateHelper::clone_state(s);const auto receipt=z->capture();p->seed_cache();
    check(!z->try_update_sampled_at_knot(s,input,R,&out)&&same_state(s,before)&&same_receipt(receipt,z->capture())&&p->cache_valid()&&
      out.H(0,0)==123.&&out.residual(0)==456.&&out.independent_R(0,0)==789.,"refusal preserves state, raw owners, receipt, cache and diagnostics");};
  for(int i=0;i<13;++i){auto bad=r;M6 R=M6::Zero();
    if(i==0)bad.timestamp=std::nextafter(0.,1.);
    if(i==1)bad.measured(0)+=.01;
    if(i==2)bad.sequence=9;
    if(i==3)R(0,0)=-1.;
    if(i==4)R(0,0)=std::numeric_limits<double>::infinity();
    if(i==5)s->_options.do_fej=true;
    if(i==6)s->_options.integration_method=StateOptions::RK4;
    if(i==7)s->_options.do_calib_imu_intrinsics=true;
    if(i==8)s->_imu_endpoint_valid=false;
    auto original=s->_imu->value();
    if(i==9){auto x=original;x(7)=2.;s->_imu->set_value(x);}
    if(i==10){auto x=original;x(10)+=100.;s->_imu->set_value(x);}
    if(i==11){R.setIdentity();R(0,1)=R(1,0)=2.;}
    if(i==12){R.setIdentity();R(0,1)=.1;}
    reject(bad,R);s->_imu->set_value(original);s->_options.do_fej=false;s->_options.integration_method=StateOptions::ANALYTICAL;
    s->_options.do_calib_imu_intrinsics=false;s->_imu_endpoint_valid=true;
  }
  const auto dw=s->_calib_imu_dw->value(),da=s->_calib_imu_da->value(),tg=s->_calib_imu_tg->value();
  s->_calib_imu_dw->set_value(Vec::Zero(6));reject(r,M6::Zero());s->_calib_imu_dw->set_value(dw);
  s->_calib_imu_da->set_value(Vec::Zero(6));reject(r,M6::Zero());s->_calib_imu_da->set_value(da);
  auto invalid_tg=tg;invalid_tg(0)=std::numeric_limits<double>::quiet_NaN();s->_calib_imu_tg->set_value(invalid_tg);
  reject(r,M6::Zero());s->_calib_imu_tg->set_value(tg);
  // The local factor/gate is valid, but a nonfinite retained nuisance mean
  // makes the generic update refuse. Its false result must not consume a use.
  const auto clock=s->cam_imu_dt_var(0)->value();Vec invalid_clock=clock;invalid_clock(0)=std::numeric_limits<double>::quiet_NaN();
  s->cam_imu_dt_var(0)->set_value(invalid_clock);reject(r,M6::Zero());s->cam_imu_dt_var(0)->set_value(clock);
  const auto before=StateHelper::clone_state(s);const auto checkpoint=z->capture();
  check(z->try_update_sampled_at_knot(s,r,M6::Zero()),"rejected requests did not consume the valid same-knot factor");
  auto accepted=StateHelper::clone_state(s);const auto used=z->capture();reject(r,M6::Zero());
  z->restore(checkpoint);s=StateHelper::clone_state(before);
  check(z->try_update_sampled_at_knot(s,r,M6::Zero())&&same_state(s,accepted),"paired State/updater rewind reproduces one-use update exactly");
  z->restore(used);reject(r,M6::Zero());
  auto fresh=state();auto newr=record(fresh,1,0.);StateHelper::admit_sampled_imu_noise(fresh,newr);
  z->reset_for_new_state();check(z->try_update_sampled_at_knot(fresh,newr,M6::Zero()),"new-State lifecycle reset clears only old episode factor receipt");
  // S=0 with exact stationarity is not invertible. No jitter or artificial
  // sensor R is introduced merely to make it pass.
  s=state();r=record(s,1,0.);StateHelper::admit_sampled_imu_noise(s,r);z=updater(p);
  StateHelper::set_initial_covariance(s,Mat::Zero(s->max_covariance_size(),s->max_covariance_size()),variables(s));reject(r,M6::Zero());
}
} // namespace
int main(){Printer::setPrintLevel(Printer::WARNING);factor_and_conditioning();linked_sequence();rejection_and_receipts();
  std::printf("SAMPLED_ZUPT %d/%d max_factor_FD=%.12g max_full_cov=%.12g max_mean=%.12g\n",checks-failures,checks,max_fd,max_cov,max_mean);return failures?1:0;}
