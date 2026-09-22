/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureHelper.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterZeroVelocity.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_msckf;
using namespace ov_type;
namespace {
int failures = 0, checks = 0;
double max_covariance_error = 0.0;
void check(bool ok, const char *why) {
  ++checks;
  if (!ok) { ++failures; std::printf("FAIL: %s\n", why); }
}
template <typename A, typename B> bool same(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b) {
  const Eigen::MatrixXd aa = a, bb = b;
  return aa.rows() == bb.rows() && aa.cols() == bb.cols() &&
         std::memcmp(aa.data(), bb.data(), sizeof(double) * aa.size()) == 0;
}
NoiseManager noises() {
  NoiseManager n; n.sigma_w=.003; n.sigma_a=.03; n.sigma_wb=.0001; n.sigma_ab=.001; return n;
}
struct Probe : Propagator {
  Probe() : Propagator(noises(), 9.81) {}
  bool cached() const { return cache_imu_valid.load(); }
  double cache_time() const { return cache_state_time; }
  Eigen::MatrixXd cache_mean() const { return cache_state_est; }
  Eigen::MatrixXd cache_cov() const { return cache_state_covariance; }
};
std::shared_ptr<State> make_state(bool clocks=true, bool physical=true, bool moving=true) {
  StateOptions o; o.num_cameras=2; o.imu_model=StateOptions::RPNG; o.do_fej=false;
  o.integration_method=StateOptions::DISCRETE; o.do_calib_camera_timeoffset=clocks;
  o.physical_camera_clones=physical; o.max_clone_size=1;
  check(o.configure_clone_policy(false,false), "bounded physical pose policy configured");
  auto s=std::make_shared<State>(o); s->_timestamp=10.;
  Eigen::Matrix<double,16,1> x=s->_imu->value();
  if (moving) x.segment<3>(7)<<.3,-.2,.1;
  s->_imu->set_value(x); s->_imu->set_fej(x);
  return s;
}
void clock_value(const std::shared_ptr<State> &s, size_t camera, double value) {
  Eigen::VectorXd v(1);v<<value; s->cam_imu_dt_var(camera)->set_value(v); s->cam_imu_dt_var(camera)->set_fej(v);
}
ImuData sample(double t, bool stationary=false) {
  ImuData z; z.timestamp=t;
  z.wm = stationary ? Eigen::Vector3d::Zero() : Eigen::Vector3d(.2+.03*(t-10.),-.1,.15);
  z.am = stationary ? Eigen::Vector3d(0.,0.,9.81) : Eigen::Vector3d(.2,-.3,9.9);
  return z;
}
void feed(Propagator &p, bool stationary=false) {
  for(int i=0;i<=50;++i) p.feed_imu(sample(9.98+.01*i,stationary));
}
bool state_equal(const std::shared_ptr<State> &a, const std::shared_ptr<State> &b) {
  return a->_timestamp==b->_timestamp && a->_imu_endpoint==b->_imu_endpoint &&
         a->_imu_endpoint_valid==b->_imu_endpoint_valid && same(a->_imu->value(),b->_imu->value()) &&
         same(a->_imu->fej(),b->_imu->fej()) && same(StateHelper::get_full_covariance(a),StateHelper::get_full_covariance(b));
}
bool history_equal(const UpdaterZeroVelocity::Snapshot &a,const UpdaterZeroVelocity::Snapshot &b) {
  if(a.last_prop_time_offset!=b.last_prop_time_offset || a.have_last_prop_time_offset!=b.have_last_prop_time_offset ||
     a.last_zupt_state_timestamp!=b.last_zupt_state_timestamp || a.last_zupt_count!=b.last_zupt_count ||
     a.camera_history.size()!=b.camera_history.size() || a.imu_data.size()!=b.imu_data.size()) return false;
  for(size_t i=0;i<a.camera_history.size();++i) {
    const auto &x=a.camera_history[i], &y=b.camera_history[i];
    if(x.previous_raw!=y.previous_raw || x.accepted_raw!=y.accepted_raw || x.has_previous!=y.has_previous ||
       x.accepted_count!=y.accepted_count) return false;
  }
  return true;
}

void endpoint_ownership() {
  auto s=make_state(); clock_value(s,0,.004); Probe p; feed(p);
  auto before=StateHelper::clone_state(s); const auto descriptor=p.capture();
  Propagator::EndpointKinematics k; k.omega.setConstant(123.); k.omega_fej.setConstant(456.); const auto sentinel=k;
  for(double end : {9.99,10.9,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}) {
    check(!p.propagate_to_imu(s,end,10.02,k),"explicit invalid/uncovered interval refused");
    const auto after=p.capture();
    check(state_equal(s,before) && same(k.omega,sentinel.omega) && same(k.omega_fej,sentinel.omega_fej) &&
          after.have_last_prop_time_offset==descriptor.have_last_prop_time_offset &&
          after.last_prop_time_offset==descriptor.last_prop_time_offset,"failed explicit request is atomic for state, clock owner and output");
  }
  clock_value(s,0,.007); auto fresh=StateHelper::clone_state(s); Probe q; feed(q); Propagator::EndpointKinematics kq;
  check(p.propagate_to_imu(s,10.05,10.043,k) && q.propagate_to_imu(fresh,10.05,10.043,kq),"covered explicit endpoint succeeds after failures");
  check(state_equal(s,fresh) && same(k.omega,kq.omega) && s->_clones_IMU.empty() && s->_exposure_poses.empty(),
        "failed-then-valid equals fresh and explicit API creates no implicit clone");
  check(s->imu_endpoint()==10.05 && s->_imu_endpoint_valid,"accepted physical endpoint is explicit");

  // Online td changes cannot move an already accepted navigation state in time.
  clock_value(s,0,.03); clock_value(fresh,0,.03);
  check(s->imu_endpoint()==10.05 && s->_timestamp+s->cam_imu_dt_ref()!=s->imu_endpoint(),"td update cannot relabel physical state time");
  Eigen::Matrix<double,13,1> out,outq; Eigen::Matrix<double,12,12> cov,covq;
  auto physical_start=StateHelper::clone_state(s); physical_start->_timestamp=10.05; clock_value(physical_start,0,0.);
  Probe oracle; feed(oracle);
  check(p.fast_state_propagate(s,10.075,out,cov) && oracle.fast_state_propagate(physical_start,10.075,outq,covq) &&
        same(out,outq) && same(cov,covq),"fast prediction starts at authoritative endpoint despite changed reference clock");
  const auto cache_mean=p.cache_mean(),cache_cov=p.cache_cov(); const auto out0=out; const auto cov0=cov;
  check(!p.fast_state_propagate(s,10.9,out,cov) && p.cached() && same(cache_mean,p.cache_mean()) &&
        same(cache_cov,p.cache_cov()) && same(out,out0) && same(cov,cov0),"failed fast request preserves existing cache and caller outputs");
  check(p.propagate_to_imu(s,10.08,10.05,k) && !p.cached(),"accepted navigation propagation invalidates future prediction cache");

  auto exact=StateHelper::clone_state(s); const auto P=StateHelper::get_full_covariance(s);
  check(p.propagate_to_imu(s,10.08,10.05,k) && state_equal(s,exact) && same(P,StateHelper::get_full_covariance(s)),
        "same-time second owner adds no process noise or navigation motion");
  check((k.omega-sample(10.08).wm).norm()<1e-13,"same-time owner receives corrected covered endpoint rate");
  Probe missing;
  check(!missing.propagate_to_imu(s,10.08,10.05,k) && state_equal(s,exact),"same-time owner still requires a covered endpoint sample");
  Probe corrupt; auto z=sample(10.08);z.wm(1)=std::numeric_limits<double>::quiet_NaN();corrupt.feed_imu(z);
  check(!corrupt.propagate_to_imu(s,10.08,10.05,k) && state_equal(s,exact),"same-time nonfinite signal rejected under fast-math");

  // State and propagator snapshots restore the accepted endpoint jointly.
  const auto ps=p.capture(); auto saved=StateHelper::clone_state(s); auto continuation=StateHelper::clone_state(saved); Probe restored;
  restored.restore(ps); check(p.propagate_to_imu(s,10.11,10.08,k) && restored.propagate_to_imu(continuation,10.11,10.08,kq) &&
      state_equal(s,continuation),"snapshot continuation retains endpoint rather than recomputing from mutable td");
  p.reset_for_new_state(); auto reset=make_state();clock_value(reset,0,-.003);auto reset_oracle=StateHelper::clone_state(reset);Probe rp;feed(rp);
  check(p.propagate_to_imu(reset,10.02,10.023,k) && rp.propagate_to_imu(reset_oracle,10.02,10.023,kq) && state_equal(reset,reset_oracle),
        "new navigation episode uses its initializer endpoint and retained IMU independently of old clock history");
}

Eigen::MatrixXd rich_covariance(int n) {
  Eigen::MatrixXd L=Eigen::MatrixXd::Identity(n,n);
  for(int i=0;i<n;++i) for(int j=0;j<=i;++j) L(i,j)+=.04*std::sin(.3+.7*i+.2*j);
  return .01*L*L.transpose();
}
void owner_covariance(bool clocks) {
  auto s=make_state(clocks); s->_imu_endpoint=10.1;s->_imu_endpoint_valid=true;s->_timestamp=10.098;
  std::vector<std::shared_ptr<Type>> order{s->_imu};
  if(clocks) {order.push_back(s->cam_imu_dt_var(0));order.push_back(s->cam_imu_dt_var(1));}
  const int n=s->max_covariance_size();const auto P=rich_covariance(n);StateHelper::set_initial_covariance(s,P,order);
  const Eigen::Vector3d omega(.3,-.2,.5);Eigen::Matrix<double,6,1>d;d<<omega,s->_imu->vel();
  // Independent geometric derivative: camera time advances the body rotation
  // by Exp(-omega dt), while PoseJPL's positive error rotates by Exp(-dtheta).
  for(double epsilon : {1e-4,1e-5,1e-6}) {
    Eigen::Matrix<double,6,1> derivative;
    derivative.head<3>() = -(log_so3(exp_so3(-omega*epsilon))-log_so3(exp_so3(omega*epsilon)))/(2*epsilon);
    derivative.tail<3>() = ((s->_imu->pos()+s->_imu->vel()*epsilon)-(s->_imu->pos()-s->_imu->vel()*epsilon))/(2*epsilon);
    check((derivative-d).norm()<1e-9,"owner-clock augmentation sign and coordinates match physical pose-time finite differences");
  }
  Eigen::MatrixXd A=Eigen::MatrixXd::Zero(n+12,n);A.topRows(n).setIdentity();
  for(int c=0;c<2;++c) {
    A.block<6,6>(n+6*c,s->_imu->pose()->id()).setIdentity();
    if(clocks)A.block<6,1>(n+6*c,s->cam_imu_dt_var(c)->id())+=d;
    const auto view=StateHelper::augment_pose_view(s,c,omega);
    State::ExposurePose owner;owner.camera_id=c;owner.raw_time=10.098;owner.imu_time=10.1;owner.pose=view;
    owner.kinematics.omega=omega;owner.kinematics.vel=s->_imu->vel();s->_exposure_poses.push_back(owner);
  }
  const Eigen::MatrixXd expected=A*P*A.transpose(), actual=StateHelper::get_full_covariance(s);
  const double error=(expected-actual).norm();max_covariance_error=std::max(max_covariance_error,error);
  check(error<1e-13,"same-time independent owner-clock views match dense stacked full-correlation Gaussian oracle");
  check(s->_exposure_poses[0].pose!=s->_exposure_poses[1].pose && s->_exposure_poses[0].pose->id()!=s->_exposure_poses[1].pose->id() &&
        same(s->_exposure_poses[0].pose->value(),s->_exposure_poses[1].pose->value()),"equal-time owners are separate Types with equal nominal poses");
  Eigen::Matrix<double,6,6> difference=actual.block<6,6>(n,n)+actual.block<6,6>(n+6,n+6)-actual.block<6,6>(n,n+6)-actual.block<6,6>(n+6,n);
  check(clocks ? difference.norm()>1e-5 : difference.norm()<1e-14,"independent clock uncertainty survives equal nominal exposure time");
  check(s->_clones_IMU.empty(),"owned pose augmentation does not insert legacy timestamp aliases");
  auto copy=StateHelper::clone_state(s);
  check(state_equal(s,copy) && copy->_exposure_poses.size()==2 && copy->pose_for_camera(0,10.098)!=s->pose_for_camera(0,10.098) &&
        copy->pose_for_camera(0,10.098)->id()==s->pose_for_camera(0,10.098)->id(),"snapshot deep-resolves owner handles into copied covariance variables");
  auto old=s->_exposure_poses.front().pose;
  const auto extra=StateHelper::augment_pose_view(s,0,omega);State::ExposurePose owner;owner.camera_id=0;owner.raw_time=10.11;
  owner.imu_time=10.112;owner.pose=extra;s->_exposure_poses.push_back(owner);
  const auto before=StateHelper::get_full_covariance(s);const int remove=old->id();
  Eigen::MatrixXd retained(before.rows()-6,before.cols()-6);
  for(int r=0;r<retained.rows();++r)for(int c=0;c<retained.cols();++c)retained(r,c)=before(r+(r>=remove?6:0),c+(c>=remove?6:0));
  StateHelper::marginalize_old_clone(s);
  check(s->_exposure_poses.size()==2 && old->id()==-1 && s->find_pose(0,10.098)==nullptr && s->find_pose(1,10.098)!=nullptr &&
        same(retained,StateHelper::get_full_covariance(s)),"bounded marginalization removes one owner and exact covariance block, retaining equal-time other camera");
  const auto copyP=StateHelper::get_full_covariance(copy);Eigen::VectorXd dx=Eigen::VectorXd::Constant(6,.001);old->update(dx);
  check(same(copyP,StateHelper::get_full_covariance(copy)) && !same(old->value(),copy->pose_for_camera(0,10.098)->value()),
        "snapshot owns independent pose memory and covariance");
  const int before_invalid=s->max_covariance_size(); bool threw=false;
  try {StateHelper::augment_pose_view(s,100,omega);}catch(const std::out_of_range&){threw=true;}
  check(threw && s->max_covariance_size()==before_invalid,"invalid clock owner cannot silently alias the reference or mutate covariance");
}

struct StationaryFixture {
  std::shared_ptr<State>s=make_state(false,true,false);
  std::shared_ptr<Probe>p=std::make_shared<Probe>();
  std::shared_ptr<FeatureDatabase>db=std::make_shared<FeatureDatabase>();
  NoiseManager n=noises();UpdaterOptions options;
  std::unique_ptr<UpdaterZeroVelocity>z;
  StationaryFixture(double disparity=0.) {
    options.chi2_multipler=1.;z=std::make_unique<UpdaterZeroVelocity>(options,n,db,p,9.81,.05,1.,disparity);
    for(int i=0;i<=50;++i) {auto data=sample(9.98+.01*i,true);p->feed_imu(data);z->feed_imu(data);}
  }
};
void zupt_ownership() {
  StationaryFixture f,g; clock_value(f.s,0,.004);clock_value(g.s,0,.004);
  const auto before=StateHelper::clone_state(f.s);const auto history=f.z->capture();
  check(!f.z->try_update_at_imu(f.s,10.9,10.896,{{0,10.896}}) && state_equal(f.s,before) && history_equal(history,f.z->capture()),
        "ZUPT coverage refusal preserves all endpoint, detection and camera-key history");
  Eigen::Matrix<double,16,1> x=f.s->_imu->value();x.segment<3>(7)<<1.,0.,0.;f.s->_imu->set_value(x);
  const auto moving=StateHelper::clone_state(f.s);
  check(!f.z->try_update_at_imu(f.s,10.04,10.036,{{0,10.036}}) && state_equal(f.s,moving) && !f.s->_imu_endpoint_valid,
        "ZUPT statistical rejection cannot commit endpoint or covariance");
  x.segment<3>(7).setZero();f.s->_imu->set_value(x);
  check(f.z->try_update_at_imu(f.s,10.04,10.036,{{0,10.036}}) && g.z->try_update_at_imu(g.s,10.04,10.036,{{0,10.036}}) &&
        state_equal(f.s,g.s),"ZUPT rejected-then-valid equals fresh numerical update");
  check(f.s->_imu_endpoint_valid && f.s->imu_endpoint()==10.04,"accepted ZUPT owns its exact IMU endpoint");
  // Both propagation and the next ZUPT start there even if current reference td changes.
  clock_value(f.s,0,.018);clock_value(g.s,0,.018);
  auto zero_clock=StateHelper::clone_state(f.s);zero_clock->_timestamp=10.04;zero_clock->_imu_endpoint_valid=false;
  clock_value(zero_clock,0,0.);Probe direct;feed(direct,true);
  Propagator::EndpointKinematics k,kq;
  check(f.p->propagate_to_imu(f.s,10.06,10.042,k) && g.p->propagate_to_imu(g.s,10.06,10.042,kq) && state_equal(f.s,g.s),
        "ZUPT-to-propagation continuation consumes shared accepted endpoint");
  check(direct.propagate_to_imu(zero_clock,10.06,10.042,kq) && state_equal(f.s,zero_clock),
        "ZUPT continuation agrees with independent zero-clock state initialized at the accepted physical endpoint");
  Eigen::Matrix<double,13,1> out;Eigen::Matrix<double,12,12>cov;
  check(f.p->fast_state_propagate(f.s,10.10,out,cov)&&f.p->cached(),"future cache initialized before accepted ZUPT");
  check(f.z->try_update_at_imu(f.s,10.08,10.062,{{0,10.062}}) && !f.p->cached(),"propagation-to-ZUPT accepts exact interval and invalidates fast cache");
  const auto statesnap=StateHelper::clone_state(f.s);const auto propsnap=f.p->capture();const auto zuptsnap=f.z->capture();
  auto copy=StateHelper::clone_state(statesnap);StationaryFixture branch;branch.p->restore(propsnap);branch.z->restore(zuptsnap);
  check(f.z->try_update_at_imu(f.s,10.12,10.102,{{0,10.102}}) && branch.z->try_update_at_imu(copy,10.12,10.102,{{0,10.102}}) &&
        state_equal(f.s,copy) && history_equal(f.z->capture(),branch.z->capture()),"ZUPT state/IMU/key snapshot restores an identical continuation");
  f.z->reset_for_new_state();f.p->reset_for_new_state();auto reset=make_state(false,true,false);StationaryFixture reset_oracle;
  check(f.z->try_update_at_imu(reset,10.03,10.03,{{1,10.03}}) && reset_oracle.z->try_update_at_imu(reset_oracle.s,10.03,10.03,{{1,10.03}}) &&
        state_equal(reset,reset_oracle.s),"ZUPT reset retains IMU but no old episode detection or raw-key ownership");
}

void database_ownership() {
  auto db=std::make_shared<FeatureDatabase>();
  for(size_t id=0;id<24;++id)for(size_t c=0;c<2;++c)for(int k=0;k<4;++k)
    db->update_feature(id,10.+.01*k,c,float(id+10*c+k),float(20+k),0.,0.);
  const auto other=db->get_feature(0)->timestamps.at(1);
  db->cleanup_measurements_exact_camera(0,10.01);
  check(db->get_feature(0)->timestamps.at(0).size()==3 && db->get_feature(0)->timestamps.at(1)==other,
        "exact camera cleanup retains same-raw observations of every other camera");
  db->cleanup_measurements_camera(0,10.025);
  auto f=db->get_feature(0);
  check(f->timestamps.at(0).size()==1 && f->timestamps.at(0)[0]==10.03 && f->uvs.at(0)[0](0)==3.f &&
        f->uvs_norm.at(0).size()==1 && f->timestamps.at(1)==other,"camera cutoff compacts parallel arrays once without cross-camera deletion");
  check(db->features_containing_camera(0,10.01).empty() && db->features_containing_camera(1,10.01).size()==24,
        "exact feature query retains camera/raw key ownership");
  double mean=0.,stddev=0.;int count=0;
  FeatureHelper::compute_disparity(db,10.,10.02,mean,stddev,count,1);
  check(count==24 && std::abs(mean-std::sqrt(8.))<1e-6,"camera-filtered disparity uses requested camera's actual raw pixel motions");
  FeatureHelper::compute_disparity(db,10.,10.02,mean,stddev,count,0);
  check(count==0 && mean==-1. && stddev==-1.,"empty filtered disparity has explicit finite sentinel statistics");

  // A camera with one track cannot get the same vote as one with 30 tracks.
  // Mean pooled disparity=(30*0+1*10)/31 < 1, while an equal-camera mean is 5.
  StationaryFixture pooled(1.);
  for(size_t c=0;c<2;++c)for(size_t id=0;id<(c==0?30u:1u);++id)for(int k=1;k<=3;++k)
    pooled.db->update_feature(id,10.+.02*k,c,float(id+(c==1?10*k:0)),20.,0.,0.);
  check(pooled.z->try_update_at_imu(pooled.s,10.02,10.02,{{0,10.02},{1,10.02}}),"first physical stationary group establishes per-camera histories");
  Eigen::Matrix<double,16,1> x=pooled.s->_imu->value();x.segment<3>(7)<<1.,0.,0.;pooled.s->_imu->set_value(x);
  check(pooled.z->try_update_at_imu(pooled.s,10.04,10.04,{{0,10.04},{1,10.04}}),"physical multi-owner disparity is weighted by feature counts");
  // A camera-only accepted third event removes only its prior accepted key.
  x=pooled.s->_imu->value();x.segment<3>(7).setZero();pooled.s->_imu->set_value(x);
  check(pooled.z->try_update_at_imu(pooled.s,10.06,10.06,{{0,10.06}}),"third camera-owned stationary event accepted");
  check(pooled.db->features_containing_camera(0,10.04).empty() && pooled.db->features_containing_camera(1,10.04).size()==1,
        "accepted ZUPT cleanup cannot delete other camera at equal raw timestamp");
  const auto before=StateHelper::clone_state(pooled.s);const auto history=pooled.z->capture();
  check(!pooled.z->try_update_at_imu(pooled.s,10.08,10.08,{{0,10.08},{0,10.08}}) && state_equal(pooled.s,before) &&
        history_equal(history,pooled.z->capture()),"duplicate physical owner keys are rejected atomically");
}
} // namespace
int main() {
  Printer::setPrintLevel("ERROR");
  endpoint_ownership();owner_covariance(true);owner_covariance(false);zupt_ownership();database_ownership();
  std::printf("IMU_ENDPOINT_VIEWS %s checks=%d failures=%d maxCovarianceError=%.17g\n",failures?"FAIL":"PASS",checks,failures,max_covariance_error);
  return failures?1:0;
}
