/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstdio>
#include <cstring>
#include <limits>
#include "core/VioManager.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"

namespace {
using namespace ov_msckf;
using ov_core::ImuData;
int failures = 0;
void check(bool condition, const char *label) {
  if (!condition) { ++failures; std::printf("FAIL: %s\n", label); }
}
template<class A, class B> bool same(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         std::memcmp(a.derived().data(), b.derived().data(), sizeof(double)*a.size()) == 0;
}
ImuData sample(double t) {
  ImuData m; m.timestamp=t;
  m.wm << .1+.02*t, -.04+.003*t*t, .03;
  m.am << .3+.1*t, -.2, 9.81+.002*t*t;
  return m;
}
std::shared_ptr<State> state() {
  StateOptions options; options.num_cameras=2; options.imu_model=StateOptions::RPNG;
  options.integration_method=StateOptions::DISCRETE; options.do_fej=false;
  auto s=std::make_shared<State>(options); s->_timestamp=10.;
  Eigen::Matrix<double,16,1> x=s->_imu->value(); x.segment<3>(7)<<.2,-.1,.05;
  s->_imu->set_value(x); s->_imu->set_fej(x);
  return s;
}
void td(const std::shared_ptr<State> &s,double value) {
  Eigen::VectorXd v(1);v<<value;s->cam_imu_dt_var(0)->set_value(v);s->cam_imu_dt_var(0)->set_fej(v);
}
void feed(Propagator &p) { for (int i=0;i<=10;++i) p.feed_imu(sample(10.+.01*i)); }
struct Probe : Propagator {
  Probe():Propagator(NoiseManager(),9.81){}
  bool cached() const{return cache_imu_valid.load();}
  double time() const{return cache_state_time;}
  double offset() const{return cache_t_off;}
  Eigen::MatrixXd mean() const{return cache_state_est;}
  Eigen::MatrixXd cov() const{return cache_state_covariance;}
};

void selector_checks() {
  const std::vector<ImuData> data{sample(10.),sample(11.),sample(12.),sample(13.)};
  for (const auto &interval : std::vector<std::pair<double,double>>{{10.,13.},{10.,11.},{10.2,10.7},{10.2,11.},
                                                                {10.2,12.7},{11.,11.7},{11.,13.},{12.2,12.7}}) {
    auto selected=Propagator::select_imu_readings(data,interval.first,interval.second,false);
    check(selected.size()>=2 && selected.front().timestamp==interval.first && selected.back().timestamp==interval.second,
          "covered interval has exact endpoints, including within one sample interval");
    for (size_t i=0;i<selected.size();++i) {
      if(i)check(selected[i].timestamp>selected[i-1].timestamp,"selector emits strictly increasing samples");
      const double t=selected[i].timestamp;
      auto exact=std::find_if(data.begin(),data.end(),[t](const ImuData &s){return s.timestamp==t;});
      if(exact!=data.end()) {
        check(same(selected[i].wm,exact->wm)&&same(selected[i].am,exact->am),"exact endpoint/interior sample values remain bit-identical");
      } else {
        size_t left=0;while(data[left+1].timestamp<t)++left;
        const double f=(t-data[left].timestamp)/(data[left+1].timestamp-data[left].timestamp);
        const Eigen::Vector3d w=(1-f)*data[left].wm+f*data[left+1].wm;
        const Eigen::Vector3d a=(1-f)*data[left].am+f*data[left+1].am;
        check(same(selected[i].wm,w)&&same(selected[i].am,a),"boundary interpolation retains the covered-sample arithmetic");
      }
    }
  }
  const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
  for (const auto &interval : std::vector<std::pair<double,double>>{{9.9,11.},{10.1,13.1},{13.1,13.2},{10.,10.},
                                                                 {11.,10.},{nan,11.},{10.,nan},{-inf,11.},{10.,inf}})
    check(Propagator::select_imu_readings(data,interval.first,interval.second,false).empty(),"invalid/uncovered interval refused without extrapolation");
  check(Propagator::select_imu_readings({},10.,11.,false).empty(),"empty buffer refused");
  check(Propagator::select_imu_readings({sample(10.)},10.,11.,false).empty(),"single sample refused");
  for(int kind=0;kind<5;++kind) {
    auto bad=data;
    if(kind==0)bad[1].timestamp=bad[0].timestamp;
    if(kind==1)bad[1].timestamp=9.;
    if(kind==2)bad[1].timestamp=nan;
    if(kind==3)bad[1].wm(1)=nan;
    if(kind==4)bad[1].am(2)=inf;
    check(Propagator::select_imu_readings(bad,10.,13.,false).empty(),"nonfinite signal/stamp or nonmonotone buffer refused under fast-math");
  }
}

void full_atomic_checks() {
  auto s=state();td(s,.004);Probe p;feed(p);
  const auto before=s->_imu->value(),fej=s->_imu->fej(),P=StateHelper::get_full_covariance(s);
  const auto ps=p.capture();
  for(double end : {10.2,9.9,10.,std::numeric_limits<double>::quiet_NaN()}) {
    check(!p.propagate_and_clone(s,end),"full propagation refuses uncovered/backwards/nonfinite request");
    const auto after=p.capture();
    check(s->_timestamp==10. && s->_clones_IMU.empty() && s->_clones_kinematics.empty() &&
          same(s->_imu->value(),before) && same(s->_imu->fej(),fej) && same(StateHelper::get_full_covariance(s),P) &&
          after.have_last_prop_time_offset==ps.have_last_prop_time_offset && after.last_prop_time_offset==ps.last_prop_time_offset,
          "failed full propagation preserves state/covariance/clones/time-offset ownership");
  }
  // A rejected first call must not latch a now-stale offset. Compare continuation
  // to a fresh propagator with exactly the same accepted inputs.
  td(s,.007);auto fresh=state();td(fresh,.007);Probe q;feed(q);
  check(p.propagate_and_clone(s,10.04)&&q.propagate_and_clone(fresh,10.04),"valid full propagation after failure succeeds");
  check(same(s->_imu->value(),fresh->_imu->value())&&same(StateHelper::get_full_covariance(s),StateHelper::get_full_covariance(fresh)),
        "failed-then-valid full continuation is bit-identical to fresh");
  const auto value=s->_imu->value(),cov=StateHelper::get_full_covariance(s);const auto settled=p.capture();
  check(!p.propagate_and_clone(s,10.2)&&same(value,s->_imu->value())&&same(cov,StateHelper::get_full_covariance(s))&&
        p.capture().last_prop_time_offset==settled.last_prop_time_offset,"later failed full propagation also leaves established ownership unchanged");
  // Missing the beginning must refuse too, even if the endpoint is in the buffer.
  auto late=state();Probe r;r.feed_imu(sample(10.01));r.feed_imu(sample(10.1));
  check(!r.propagate_and_clone(late,10.05)&&late->_timestamp==10.&&late->_clones_IMU.empty(),"missing initial bracket never clones an unpropagated state");
}

void fast_atomic_checks() {
  auto s=state();td(s,.004);Probe p;feed(p);
  Eigen::Matrix<double,13,1> output=Eigen::Matrix<double,13,1>::Constant(123.);
  Eigen::Matrix<double,12,12> covariance=Eigen::Matrix<double,12,12>::Constant(456.);
  const auto out0=output;const auto cov0=covariance;
  check(!p.fast_state_propagate(s,10.2,output,covariance)&&!p.cached()&&same(output,out0)&&same(covariance,cov0),
        "failed first fast request leaves cache uninitialized and caller outputs untouched");
  td(s,.007);auto fresh=state();td(fresh,.007);Probe q;feed(q);
  Eigen::Matrix<double,13,1> fresh_out;Eigen::Matrix<double,12,12> fresh_cov;
  check(p.fast_state_propagate(s,10.04,output,covariance)&&q.fast_state_propagate(fresh,10.04,fresh_out,fresh_cov),"valid fast request after failure succeeds");
  check(same(output,fresh_out)&&same(covariance,fresh_cov)&&same(p.mean(),q.mean())&&same(p.cov(),q.cov()),
        "failed-then-valid fast continuation is bit-identical to fresh");
  const auto mean=p.mean(),P=p.cov();const auto old_out=output;const auto old_cov=covariance;const double time=p.time(),offset=p.offset();
  for(double end : {10.2,10.03,10.04,std::numeric_limits<double>::quiet_NaN()}) {
    check(!p.fast_state_propagate(s,end,output,covariance)&&p.cached()&&p.time()==time&&p.offset()==offset&&
          same(p.mean(),mean)&&same(p.cov(),P)&&same(output,old_out)&&same(covariance,old_cov),
          "failed later fast request leaves the complete cache and outputs untouched");
  }
  check(p.fast_state_propagate(s,10.06,output,covariance)&&q.fast_state_propagate(fresh,10.06,fresh_out,fresh_cov)&&
        same(output,fresh_out)&&same(covariance,fresh_cov),"fast continuation remains identical after repeated rejected requests");
}

void bridge_checks() {
  auto s=state();Probe p;feed(p);Propagator::BridgeData bridge;
  const auto value=s->_imu->value(),P=StateHelper::get_full_covariance(s);
  check(p.compute_bridge(s,10.002,10.008,bridge)&&bridge.valid&&bridge.dt==10.008-10.002,
        "bridge is valid when both endpoints lie inside one buffered interval");
  for(const auto &endpoints:std::vector<std::pair<double,double>>{{9.999,10.05},{10.02,10.101},{10.02,10.02},{10.04,10.02},
                                                                {10.02,std::numeric_limits<double>::infinity()}})
    check(!p.compute_bridge(s,endpoints.first,endpoints.second,bridge)&&!bridge.valid&&
          same(s->_imu->value(),value)&&same(StateHelper::get_full_covariance(s),P),"uncovered bridge refuses and invalidates only its output payload");
}

VioManagerOptions manager_options() {
  VioManagerOptions o; o.state_options.num_cameras=2;o.state_options.max_slam_features=0;o.state_options.max_aruco_features=0;
  o.state_options.imu_model=StateOptions::RPNG;o.use_stereo=false;o.epoch_mode=false;o.use_aruco=false;o.use_gpu=false;o.try_zupt=false;
  o.num_opencv_threads=0;o.use_multi_threading_pubs=o.use_multi_threading_subs=false;
  o.async_guard=.002;o.vec_dw<<1,0,1,0,0,1;o.vec_da=o.vec_dw;o.vec_tg.setZero();
  o.q_ACCtoIMU<<0,0,0,1;o.q_GYROtoIMU=o.q_ACCtoIMU;
  o.init_options.num_cameras=2;o.init_options.use_stereo=false;
  for(int c=0;c<2;++c) {
    auto camera=std::make_shared<ov_core::CamRadtan>(320,240);Eigen::VectorXd intr(8);intr<<220,220,160,120,0,0,0,0;camera->set_value(intr);
    Eigen::VectorXd pose(7);pose<<0,0,0,1,.1*c,0,0;
    o.camera_intrinsics[c]=camera;o.camera_extrinsics[c]=pose;o.init_options.camera_intrinsics[c]=camera;o.init_options.camera_extrinsics[c]=pose;
    o.camera_imu_dt[c]=c==0?.02:-.02;
  }
  return o;
}
void consumer_guard_check() {
  auto options=manager_options();VioManager manager(options);
  int processed=0;manager.set_camera_processed_callback([&](const ov_core::CameraData &,bool ok){processed+=ok;return true;});
  for(int i=0;i<=10;++i)manager.feed_measurement_imu(sample(.90+.01*i));
  ov_core::CameraData frame;frame.timestamp=1.0;frame.sensor_ids={1};
  frame.images.emplace_back(cv::Mat::zeros(240,320,CV_8UC1));frame.masks.emplace_back(cv::Mat::zeros(240,320,CV_8UC1));
  manager.feed_measurement_camera(frame);
  manager.feed_measurement_imu(sample(1.01));
  check(manager.get_camera_buffer()->count_released()==0&&processed==0,"public camera drain waits for reference-clock endpoint when camera td is smaller");
  manager.feed_measurement_imu(sample(1.025));
  check(manager.get_camera_buffer()->count_released()==1&&processed==1,"same raw camera frame releases once the reference-clock endpoint is covered");
}
void physical_initialization_clock_check() {
  auto options = manager_options();
  options.state_options.physical_camera_clones = true;
  options.camera_imu_dt[0] = .125;
  options.camera_imu_dt[1] = -.25;
  VioManager manager(options), reference(options), legacy_clock_api(options);
  auto db = manager.get_track_feats()->get_feature_database();
  for (int camera = 0; camera < 2; ++camera) {
    const double raw_seed_time = 10. - options.camera_imu_dt.at(camera);
    for (double delta : {-.125, 0., .125})
      db->update_feature(41, raw_seed_time + delta, camera, 160, 120, 0, 0);
  }
  Eigen::Matrix<double,17,1> initial = Eigen::Matrix<double,17,1>::Zero();
  initial(0) = 10.; initial(4) = 1.; initial(8) = .2;
  manager.initialize_with_gt_imu(initial);
  reference.initialize_with_gt_imu(initial);
  check(manager.get_state()->imu_endpoint() == 10. && manager.get_state()->_timestamp == 9.875 &&
        manager.snapshot()->startup_imu_time == 10.,
        "IMU-clock initializer pins both navigation and startup endpoint without an estimated-clock shift");
  const auto feature = db->get_feature(41);
  check(feature && feature->timestamps.at(0) == std::vector<double>{10.} &&
        feature->timestamps.at(1) == std::vector<double>{10.375},
        "physical initialization removes consumed keys in each camera clock, including exact endpoint keys");
  legacy_clock_api.initialize_with_gt(initial);
  check(legacy_clock_api.get_state()->_timestamp == 10. && legacy_clock_api.get_state()->imu_endpoint() == 10.125,
        "reference-camera-clock initialization API preserves its existing timestamp contract");

  // A clock correction after seeding must not move the accepted IMU instant.
  td(manager.get_state(), .5);
  for (int i = 0; i <= 20; ++i) {
    manager.feed_measurement_imu(sample(9.99 + .01 * i));
    reference.feed_measurement_imu(sample(9.99 + .01 * i));
  }
  Propagator::EndpointKinematics a, b;
  check(manager.get_propagator()->propagate_to_imu(manager.get_state(), 10.0625, 9.5625, a) &&
        reference.get_propagator()->propagate_to_imu(reference.get_state(), 10.0625, 9.9375, b) &&
        same(manager.get_state()->_imu->value(), reference.get_state()->_imu->value()) &&
        same(StateHelper::get_full_covariance(manager.get_state()), StateHelper::get_full_covariance(reference.get_state())),
        "first propagation uses the same physical interval after an online camera-clock correction");
}
void soft_reset_ownership_check() {
  auto old_options=manager_options(),fresh_options=manager_options();
  VioManager reused(old_options),fresh(fresh_options);
  Eigen::Matrix<double,17,1> initial=Eigen::Matrix<double,17,1>::Zero();
  initial(0)=10.;initial(4)=1.;initial(8)=.2;
  reused.initialize_with_gt(initial);
  // The previous episode has moved the reference clock away from configured td.
  td(reused.get_state(),.04);
  for(int i=0;i<=20;++i)reused.feed_measurement_imu(sample(10.+.01*i));
  auto p=reused.get_propagator();
  check(p->propagate_and_clone(reused.get_state(),10.03),"old reset fixture latches calibrated reference offset");
  Eigen::Matrix<double,13,1> output,fresh_output;
  Eigen::Matrix<double,12,12> covariance,fresh_covariance;
  check(p->fast_state_propagate(reused.get_state(),10.09,output,covariance),"old reset fixture populates fast prediction cache");
  const auto before=p->capture();
  reused.soft_reset();
  const auto after=p->capture();
  bool imu_same=before.imu_data.size()==after.imu_data.size();
  for(size_t i=0;i<before.imu_data.size()&&imu_same;++i)
    imu_same=before.imu_data[i].timestamp==after.imu_data[i].timestamp&&same(before.imu_data[i].wm,after.imu_data[i].wm)&&
             same(before.imu_data[i].am,after.imu_data[i].am);
  check(reused.get_propagator()==p&&imu_same&&!after.have_last_prop_time_offset&&after.last_prop_time_offset==0.&&
        reused.get_state()->cam_imu_dt_ref()==.02,
        "soft reset preserves propagator identity/raw IMU but discards previous episode time ownership");
  // Simulate successful reinitialization with a different mean/time, then compare
  // both public propagation APIs against a fresh manager with identical inputs.
  initial(0)=10.1;initial(5)=.4;initial(8)=-.1;
  reused.initialize_with_gt(initial);fresh.initialize_with_gt(initial);
  for(int i=0;i<=20;++i)fresh.feed_measurement_imu(sample(10.+.01*i));
  check(p->fast_state_propagate(reused.get_state(),10.16,output,covariance)&&
        fresh.get_propagator()->fast_state_propagate(fresh.get_state(),10.16,fresh_output,fresh_covariance)&&
        same(output,fresh_output)&&same(covariance,fresh_covariance),
        "first fast prediction after real reset is bit-identical to a fresh episode");
  check(p->propagate_and_clone(reused.get_state(),10.15)&&fresh.get_propagator()->propagate_and_clone(fresh.get_state(),10.15)&&
        same(reused.get_state()->_imu->value(),fresh.get_state()->_imu->value())&&
        same(StateHelper::get_full_covariance(reused.get_state()),StateHelper::get_full_covariance(fresh.get_state())),
        "first full propagation after real reset uses the new configured offset exactly once");
}
} // namespace
int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  selector_checks();full_atomic_checks();fast_atomic_checks();bridge_checks();consumer_guard_check();soft_reset_ownership_check();
  physical_initialization_clock_check();
  std::printf("IMU_COVERAGE %s failures=%d\n",failures?"FAIL":"PASS",failures);
  return failures?1:0;
}
