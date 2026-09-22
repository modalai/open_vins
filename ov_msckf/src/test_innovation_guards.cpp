/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include "cam/CamRadtan.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Landmark.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"
#include "utils/innovation.h"
#include "utils/print.h"

using namespace ov_core;
using namespace ov_msckf;
using namespace ov_type;
namespace {
using Mat = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
int checks = 0, failures = 0;
// Construct invalid input at runtime. Under -ffast-math a compile-time NaN
// in a scalar conditional can be simplified away before it reaches the API.
double from_bits(std::uint64_t pattern) {
  volatile std::uint64_t stored=pattern;
  const std::uint64_t bits=stored;
  double value; std::memcpy(&value,&bits,sizeof(value)); return value;
}
const double nan_value = from_bits(UINT64_C(0x7ff8000000000001));
const double inf_value = from_bits(UINT64_C(0x7ff0000000000000));
void check(bool ok, const char *message) {
  ++checks;
  if (!ok) { ++failures; std::printf("FAIL: %s\n", message); }
}
bool same(const Mat &a, const Mat &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         std::memcmp(a.data(), b.data(), sizeof(double) * a.size()) == 0;
}
void quadratic_oracle() {
  for (int n : {1, 3, 9, 27, 60}) for (double scale : {.001, 1., 1000.}) {
    Mat L = Mat::Zero(n,n); Vector z(n);
    for (int i=0; i<n; ++i) {
      z(i) = .2 * std::cos(.7*i);
      L(i,i) = scale * (1.+.1*i);
      for (int j=0; j<i; ++j) L(i,j) = .03 * scale * std::sin(.4*i+.6*j);
    }
    const Mat S = L * L.transpose(); const Vector r = L*z;
    double statistic;
    check(innovation_chi2(S,r,statistic), "valid correlated innovation has an SPD solve");
    check(std::abs(statistic-z.squaredNorm()) < 1e-12*(1+z.squaredNorm()),
          "quadratic statistic equals known whitened residual energy");
  }
  for (int mode=0; mode<7; ++mode) {
    Mat S=Mat::Identity(3,3); Vector r=Vector::Ones(3);
    if (mode==0) S(0,0)=nan_value;
    if (mode==1) S(0,0)=inf_value;
    if (mode==2) r(0)=nan_value;
    if (mode==3) r(0)=inf_value;
    if (mode==4) S(0,0)=-1.;
    if (mode==5) S(0,0)=0.;
    if (mode==6) r(0)=1e308;
    double statistic=0.;
    check(!innovation_chi2(S,r,statistic), "invalid, singular, indefinite or overflowing innovation is rejected");
    check(!numeric::finite(statistic), "rejected quadratic does not report a valid small statistic");
  }
  check(!valid_innovation_limit(nan_value) && !valid_innovation_limit(inf_value) && !valid_innovation_limit(-1.),
        "invalid acceptance limit cannot disable a gate");
}
void delayed_invalid(int mode) {
  StateOptions o; auto s=std::make_shared<State>(o);
  Mat P=.2*Mat::Identity(15,15);
  if (mode==8) P(3,3)=-2.;
  StateHelper::set_initial_covariance(s,P,{s->_imu});
  Mat Hx=Mat::Zero(4,3), Hf=Mat::Zero(4,3), R=Mat::Identity(4,4);
  Hf.topRows(3).setIdentity(); Hx(3,0)=1.; Vector r=Vector::Zero(4); double multiplier=1.;
  if (mode==0) r(0)=nan_value;
  if (mode==1) r(3)=inf_value;
  if (mode==2) Hx(0,0)=nan_value;
  if (mode==3) Hf(0,0)=nan_value;
  if (mode==4) R*=nan_value;
  if (mode==5) R.setZero();
  if (mode==6) R=-R;
  if (mode==7) multiplier=nan_value;
  if (mode==7) check(!numeric::finite(multiplier), "NaN threshold reaches the delayed-init API as stored bits");
  const Mat before=StateHelper::get_full_covariance(s), mean=s->_imu->value();
  auto landmark=std::make_shared<Vec>(3);
  check(!StateHelper::initialize(s,landmark,{s->_imu->p()},Hx,Hf,R,r,multiplier),
        "actual delayed initialization rejects invalid innovation input");
  check(same(before,StateHelper::get_full_covariance(s)) && same(mean,s->_imu->value()) && landmark->id()==-1,
        "rejected delayed initialization preserves state, covariance and variable ownership");
}
void zupt_case(int mode, bool disparity) {
  StateOptions o; auto s=std::make_shared<State>(o); s->_timestamp=1.;
  Mat P=.01*Mat::Identity(15,15);
  if (mode==1) P(9,9)=nan_value;
  if (mode==2) P(9,9)=-1.;
  StateHelper::set_initial_covariance(s,P,{s->_imu});
  NoiseManager noise; noise.sigma_w=.003; noise.sigma_a=.03; noise.sigma_wb=.0001; noise.sigma_ab=.001;
  UpdaterOptions options; options.chi2_multipler=1.;
  auto db=std::make_shared<FeatureDatabase>(); auto prop=std::make_shared<Propagator>(noise,9.81);
  if (disparity) for (int id=0; id<25; ++id) for (double t : {1.,1.1})
    db->update_feature(id,t,0,100+id,100,0,0);
  const double multiplier=mode==3?nan_value:1.;
  if (mode==3) check(!numeric::finite(multiplier), "NaN noise multiplier reaches the ZUPT constructor as stored bits");
  UpdaterZeroVelocity updater(options,noise,db,prop,9.81,.03,multiplier,.1);
  for (int i=0; i<=10; ++i) {
    ImuData d; d.timestamp=1.+.01*i; d.am=Eigen::Vector3d(0,0,9.81); d.wm.setZero();
    if (mode==4 && i==5) d.wm.x()=nan_value;
    if (mode==5 && i>=4) d.wm.x()=(i%2 ? -.2 : .2);
    updater.feed_imu(d);
  }
  const Mat before=StateHelper::get_full_covariance(s), mean=s->_imu->value();
  const bool expected=mode==0 || (mode==5 && disparity);
  check(updater.try_update(s,1.1)==expected, "actual ZUPT retains valid disparity policy but rejects invalid arithmetic");
  if (!expected) check(s->_timestamp==1. && same(before,StateHelper::get_full_covariance(s)) && same(mean,s->_imu->value()),
                       "invalid or moving ZUPT leaves state and covariance untouched");
}
struct Visual {
  std::shared_ptr<State> s;
  Eigen::Vector3d point{.7,.4,4.5};
  std::vector<std::shared_ptr<Type>> variables;
  Visual() {
    StateOptions o; o.num_cameras=1; o.do_fej=false; o.max_aruco_features=0;
    o.feat_rep_msckf=LandmarkRepresentation::GLOBAL_3D; o.feat_rep_slam=LandmarkRepresentation::GLOBAL_3D;
    s=std::make_shared<State>(o);
    Vector intr(8); intr<<400,405,320,240,0,0,0,0;
    s->_cam_intrinsics[0]->set_value(intr); s->_cam_intrinsics[0]->set_fej(intr);
    auto cam=std::make_shared<CamRadtan>(640,480); cam->set_value(intr); s->_cam_intrinsics_cameras[0]=cam;
    variables.push_back(s->_imu);
    for (int k=0; k<6; ++k) {
      auto x=s->_imu->value(); x.block<3,1>(4,0)<<.15*k,.02*k*k,0.; s->_imu->set_value(x); s->_imu->set_fej(x);
      s->_timestamp=1.+.1*k; StateHelper::augment_clone(s,Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero());
      variables.push_back(s->_clones_IMU.at(s->_timestamp));
    }
    StateHelper::set_initial_covariance(s,.01*Mat::Identity(s->max_covariance_size(),s->max_covariance_size()),variables);
  }
  std::shared_ptr<Feature> track() {
    auto f=std::make_shared<Feature>(); f->featid=123; f->quality=1.; int k=0;
    for (const auto &view:s->_clones_IMU) {
      const Eigen::Vector3d local=view.second->Rot()*(point-view.second->pos());
      Eigen::Vector2f uv(400*local.x()/local.z()+320+.03*std::sin(k),405*local.y()/local.z()+240+.04*std::cos(k));
      Eigen::Vector2f un((uv.x()-320)/400.,(uv.y()-240)/405.);
      f->timestamps[0].push_back(view.first); f->uvs[0].push_back(uv); f->uvs_norm[0].push_back(un); ++k;
    }
    return f;
  }
};
void visual_case(bool slam, bool invalid) {
  Visual v; UpdaterOptions options, aruco; FeatureInitializerOptions init;
  if (slam) {
    UpdaterSLAM seed(options,aruco,init); std::vector<std::shared_ptr<Feature>> tracks{v.track()};
    seed.delayed_init(v.s,tracks);
    check(v.s->_features_SLAM.count(123)==1, "valid visual fixture initializes the actual persistent landmark");
    if (!v.s->_features_SLAM.count(123)) return;
  }
  if (invalid) options.sigma_pix=nan_value;
  const Mat before=StateHelper::get_full_covariance(v.s), mean=v.s->_imu->value();
  std::vector<std::shared_ptr<Feature>> tracks{v.track()};
  if (slam) { UpdaterSLAM updater(options,aruco,init); updater.update(v.s,tracks); }
  else { UpdaterMSCKF updater(options,init); updater.update(v.s,tracks); }
  check(tracks.size()==(invalid?0:1), "actual visual update rejects invalid noise and accepts the matched valid track");
  if (invalid) check(same(before,StateHelper::get_full_covariance(v.s)) && same(mean,v.s->_imu->value()),
                     "invalid MSCKF or SLAM gate leaves state and covariance unchanged");
  else check(!same(before,StateHelper::get_full_covariance(v.s)), "valid visual control performs a nontrivial update");
}
} // namespace
int main(int argc, char **argv) {
  Printer::setPrintLevel("ERROR");
  const std::string selected=argc>1?argv[1]:"all";
  if (selected=="all" || selected=="quadratic") quadratic_oracle();
  for (int i=0;i<9;++i) if (selected=="all" || selected=="delayed"+std::to_string(i)) delayed_invalid(i);
  for (int i=0;i<6;++i) for (bool disparity:{false,true})
    if (selected=="all" || selected=="zupt"+std::to_string(i)+(disparity?"d":"")) zupt_case(i,disparity);
  for (bool slam:{false,true}) for (bool invalid:{false,true})
    if (selected=="all" || selected==std::string(slam?"slam":"msckf")+(invalid?"_invalid":"_valid")) visual_case(slam,invalid);
  check(checks>0,"requested test case exists");
  std::printf("INNOVATION_GUARDS %s checks=%d failures=%d\n",failures?"FAIL":"PASS",checks,failures);
  return failures?1:0;
}
