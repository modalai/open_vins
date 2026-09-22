/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "cam/CamRadtan.h"
#include "feat/Feature.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterHelper.h"
#include "update/UpdaterMSCKF.h"
#include "utils/print.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace ov_core;
using namespace ov_msckf;
using namespace ov_type;
namespace {
using Mat = Eigen::MatrixXd;
using VecX = Eigen::VectorXd;
using V3 = Eigen::Vector3d;
using HF = UpdaterHelper::UpdaterHelperFeature;
int checks = 0, failures = 0;
double max_joseph = 0., max_mean = 0., max_visual = 0.;
void check(bool pass, const char *what) {
  ++checks;
  if (!pass) { ++failures; std::printf("FAIL: %s\n", what); }
}
template <typename A, typename B> bool same(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b) {
  const Mat aa = a, bb = b;
  return aa.rows() == bb.rows() && aa.cols() == bb.cols() &&
         std::memcmp(aa.data(), bb.data(), sizeof(double) * aa.size()) == 0;
}
struct ProbeState : State {
  using State::State;
  size_t id_capacity() const { return _temporal_schmidt_ids.capacity(); }
  size_t prior_capacity() const { return _temporal_schmidt_prior.capacity(); }
  const void *id_data() const { return _temporal_schmidt_ids.data(); }
  const void *prior_data() const { return _temporal_schmidt_prior.data(); }
};
std::vector<std::shared_ptr<Type>> variables(const std::shared_ptr<State> &s) {
  std::vector<std::shared_ptr<Type>> out{s->_imu};
  for (int c = 0; c < s->_options.num_cameras; ++c) {
    for (const auto &v : std::vector<std::shared_ptr<Type>>{s->cam_imu_dt_var(c), s->_calib_IMUtoCAM.at(c),
             s->_cam_intrinsics.at(c), s->_calib_camera_readout.at(c)})
      if (v->id() >= 0) out.push_back(v);
  }
  if (s->uses_physical_clones()) for (const auto &v : s->_exposure_poses) out.push_back(v.pose);
  else for (const auto &v : s->_clones_IMU) out.push_back(v.second);
  for (const auto &v : s->_features_SLAM) out.push_back(v.second);
  std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a->id() < b->id(); });
  return out;
}
std::vector<int> temporal_ids(const std::shared_ptr<State> &s) {
  std::vector<int> ids;
  for (int c = 0; c < s->_options.num_cameras; ++c)
    for (const auto &v : {s->cam_imu_dt_var(c), s->_calib_camera_readout.at(c)})
      if (v->id() >= 0) ids.push_back(v->id());
  return ids;
}
Mat frozen_block(const Mat &P, const std::vector<int> &ids) {
  Mat out(ids.size(), ids.size());
  for (size_t i = 0; i < ids.size(); ++i) for (size_t j = 0; j < ids.size(); ++j) out(i,j) = P(ids[i],ids[j]);
  return out;
}
Mat dense_H(const std::shared_ptr<State> &s, const std::vector<std::shared_ptr<Type>> &order, const Mat &H) {
  Mat out = Mat::Zero(H.rows(), s->max_covariance_size());
  int col = 0;
  for (const auto &v : order) { out.middleCols(v->id(), v->size()) += H.middleCols(col, v->size()); col += v->size(); }
  return out;
}
Mat rich_covariance(int n) {
  Mat A(n,n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) A(i,j) = .14 * std::sin(.7 + .31*i + .19*j);
  A.diagonal().array() += 1.;
  return .002 * A * A.transpose();
}
std::shared_ptr<ProbeState> algebra_state(bool gate, bool physical = false, bool temporal = true) {
  StateOptions o; o.num_cameras = 3; o.do_fej = false; o.dt_calib_gate = gate;
  o.do_calib_camera_timeoffset = temporal; o.do_calib_camera_readout = temporal;
  o.camera_estimate_readout = {{0,true},{1,false},{2,true}};
  o.physical_camera_clones = physical;
  check(o.configure_clone_policy(false,false), "algebra state has a valid bounded pose policy");
  auto s = std::make_shared<ProbeState>(o);
  for (int c = 0; c < 3; ++c) {
    VecX v(1); v << .002*(c+1); s->cam_imu_dt_var(c)->set_value(v); s->cam_imu_dt_var(c)->set_fej(v);
    v << .006 + .001*c; s->_calib_camera_readout.at(c)->set_value(v); s->_calib_camera_readout.at(c)->set_fej(v);
  }
  StateHelper::set_initial_covariance(s, rich_covariance(s->max_covariance_size()), variables(s));
  return s;
}
void apply_delta(const std::shared_ptr<State> &s, const VecX &dx) {
  for (const auto &v : variables(s)) v->update(dx.segment(v->id(), v->size()));
}
double mean_error(const std::shared_ptr<State> &a, const std::shared_ptr<State> &b) {
  const auto av = variables(a), bv = variables(b); double out = 0.;
  if (av.size() != bv.size()) return 1e100;
  for (size_t i = 0; i < av.size(); ++i) out = std::max(out, (av[i]->value()-bv[i]->value()).cwiseAbs().maxCoeff());
  return out;
}
bool same_means(const std::shared_ptr<State> &a, const std::shared_ptr<State> &b) {
  const auto av = variables(a), bv = variables(b);
  if (av.size() != bv.size()) return false;
  for (size_t i = 0; i < av.size(); ++i) if (!same(av[i]->value(), bv[i]->value())) return false;
  return true;
}
struct Joseph { Mat P, K; };
Joseph joseph(const Mat &P, const Mat &H, const Mat &R, const std::vector<int> &frozen) {
  // Independent dense Joseph oracle: no small-block restoration or triangular subtract.
  const Mat S = H * P * H.transpose() + R;
  Mat K = S.llt().solve(H * P).transpose();
  for (int i : frozen) K.row(i).setZero();
  const Mat A = Mat::Identity(P.rows(), P.cols()) - K * H;
  return {A * P * A.transpose() + K * R * K.transpose(), K};
}
void legacy_update(const std::shared_ptr<State> &s, const std::vector<std::shared_ptr<Type>> &order,
                   const Mat &H, const VecX &res, const Mat &R) {
  // Verbatim pre-Schmidt algebra, using public covariance access for the test.
  // Retain the former inverse-based backend as a numerical cross-check. The
  // direct gain solve deliberately changes floating-point reduction order.
  Mat P = StateHelper::get_full_covariance(s);
  Mat M_a = Mat::Zero(P.rows(), res.rows());
  std::vector<int> H_id; int current_it = 0;
  for (const auto &v : order) { H_id.push_back(current_it); current_it += v->size(); }
  for (const auto &v : variables(s)) {
    Mat M_i = Mat::Zero(v->size(), res.rows());
    for (size_t i = 0; i < order.size(); ++i)
      M_i.noalias() += P.block(v->id(), order[i]->id(), v->size(), order[i]->size()) *
                       H.block(0,H_id[i],H.rows(),order[i]->size()).transpose();
    M_a.block(v->id(),0,v->size(),res.rows()) = M_i;
  }
  Mat P_small = StateHelper::get_marginal_covariance(s,order);
  Mat S(R.rows(),R.rows());
  S.triangularView<Eigen::Upper>() = H * P_small * H.transpose();
  S.triangularView<Eigen::Upper>() += R;
  Mat Sinv = Mat::Identity(R.rows(),R.rows());
  S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
  Mat K = M_a * Sinv.selfadjointView<Eigen::Upper>();
  P.triangularView<Eigen::Upper>() -= K * M_a.transpose();
  P = P.selfadjointView<Eigen::Upper>();
  StateHelper::set_initial_covariance(s,P,variables(s));
  apply_delta(s,K*res);
}
std::vector<std::shared_ptr<Type>> measure_order(const std::shared_ptr<State> &s, bool indirect) {
  if (indirect) return {s->_imu->v(),s->_imu->q(),s->_imu->p()};
  return {s->_calib_camera_readout.at(2),s->_imu->p(),s->cam_imu_dt_var(1),s->_imu->q(),
          s->_calib_camera_readout.at(0),s->cam_imu_dt_var(0),s->cam_imu_dt_var(2)};
}
Mat measurement(const std::vector<std::shared_ptr<Type>> &order, int step) {
  int n = 0; for (const auto &v : order) n += v->size();
  Mat H(8,n);
  for (int i = 0; i < H.rows(); ++i) for (int j = 0; j < H.cols(); ++j)
    H(i,j) = std::sin(.21*i+.63*j+.09*step) + .2*std::cos(.34*i-.27*j+.12*step);
  return H;
}
void covariance_and_parity() {
  for (bool physical : {false,true}) for (bool indirect : {false,true}) {
    auto s = algebra_state(true,physical); const auto ids = temporal_ids(s);
    check(ids.size()==5 && ids[0]!=ids[2], "only estimated temporal states are frozen; reference alias is not duplicated");
    const Mat initial_ff = frozen_block(StateHelper::get_full_covariance(s),ids);
    check(std::abs(initial_ff(0,2))>1e-5, "oracle has nonzero between-camera clock covariance");
    check(s->id_capacity()==0 && s->prior_capacity()==0, "temporal scratch is initially unallocated");
    const void *id_ptr = nullptr, *prior_ptr = nullptr; size_t id_cap = 0, prior_cap = 0;
    for (int step = 0; step < 16; ++step) {
      const auto order = measure_order(s,indirect); const Mat H = measurement(order,step), Hd = dense_H(s,order,H);
      VecX r(8); for (int i = 0; i < 8; ++i) r(i)=.002*std::cos(.8*i+.19*step);
      const Mat R = .03*Mat::Identity(8,8); const Mat prior = StateHelper::get_full_covariance(s);
      const auto expected = joseph(prior,Hd,R,ids), ordinary = joseph(prior,Hd,R,{});
      auto mean = StateHelper::clone_state(s); apply_delta(mean,expected.K*r);
      StateHelper::EKFUpdate(s,order,H,r,R);
      const Mat actual = StateHelper::get_full_covariance(s);
      const double error = (actual-expected.P).cwiseAbs().maxCoeff(); max_joseph = std::max(max_joseph,error);
      max_mean = std::max(max_mean,mean_error(s,mean));
      check(error<3e-13 && mean_error(s,mean)<3e-14, "complete covariance and mean match dense Schmidt Joseph oracle");
      check(same(frozen_block(actual,ids),initial_ff), "complete correlated temporal prior block stays bit-identical over repeated updates");
      check((actual.block(0,ids[0],15,1)-ordinary.P.block(0,ids[0],15,1)).norm()<1e-13,
            "active-temporal cross covariance remains the ordinary posterior");
      if (step==0) {
        check((actual.block(0,ids[0],15,1)-prior.block(0,ids[0],15,1)).norm()>1e-6,
              "cross covariance is updated, not incorrectly restored to its prior");
        check((ordinary.K*r)(ids[0])!=0., "ordinary update would change a clock, including through indirect-only H");
        id_ptr=s->id_data(); prior_ptr=s->prior_data(); id_cap=s->id_capacity(); prior_cap=s->prior_capacity();
      } else check(id_ptr==s->id_data() && prior_ptr==s->prior_data() && id_cap==s->id_capacity() && prior_cap==s->prior_capacity(),
                   "repeated freezes reuse both bounded scratch allocations");
      Eigen::SelfAdjointEigenSolver<Mat> eigen(actual);
      check(eigen.info()==Eigen::Success && eigen.eigenvalues().minCoeff()>-2e-14, "repeated Schmidt posterior is symmetric PSD");
    }
    check(id_cap<=6 && prior_cap<=36, "scratch capacity is bounded by configured camera temporal dimension");
    // Snapshot copies estimator state, not transient scratch; it must continue identically.
    auto copy=StateHelper::clone_state(s); const auto order=measure_order(s,indirect), other_order=measure_order(copy,indirect);
    const Mat H=measurement(order,40), R=.03*Mat::Identity(8,8); const VecX r=VecX::Constant(8,.001);
    StateHelper::EKFUpdate(s,order,H,r,R); StateHelper::EKFUpdate(copy,other_order,H,r,R);
    check(same(StateHelper::get_full_covariance(s),StateHelper::get_full_covariance(copy)) && same_means(s,copy),
          "snapshot continuation rebuilds scratch without changing estimator results");
  }
  for (bool temporal : {false,true}) {
    auto off=algebra_state(false,false,temporal), on=algebra_state(true,false,temporal);
    State::CloneKinematics excited; excited.omega=V3(.1,0.,0.); on->_clones_kinematics[1.]=excited;
    auto old=StateHelper::clone_state(off);
    for(int step=0;step<6;++step) {
      const auto oo=measure_order(off,true), no=measure_order(on,true), lo=measure_order(old,true);
      const Mat H=measurement(oo,step), R=.03*Mat::Identity(8,8); const VecX r=VecX::Constant(8,.001);
      const Mat prior = StateHelper::get_full_covariance(off);
      const auto expected = joseph(prior,dense_H(off,oo,H),R,{});
      auto expected_mean = StateHelper::clone_state(off); apply_delta(expected_mean,expected.K*r);
      StateHelper::EKFUpdate(off,oo,H,r,R); StateHelper::EKFUpdate(on,no,H,r,R); legacy_update(old,lo,H,r,R);
      // Backend arithmetic may change; the independent dense Joseph contract
      // and exact equality between the two unfrozen policies must still hold.
      check((StateHelper::get_full_covariance(off)-expected.P).cwiseAbs().maxCoeff()<3e-13 &&
                mean_error(off,expected_mean)<3e-14,
            "gate-off covariance and means match the independent dense Joseph oracle");
      check((StateHelper::get_full_covariance(off)-StateHelper::get_full_covariance(old)).cwiseAbs().maxCoeff()<3e-13 &&
                mean_error(off,old)<3e-14,
            "direct gain solve agrees numerically with the former inverse backend");
      check(same(StateHelper::get_full_covariance(off),StateHelper::get_full_covariance(on)) && same_means(off,on),
            "enabled but excited policy has exact gate-off parity");
    }
    check(off->id_capacity()==0 && off->prior_capacity()==0 && on->id_capacity()==0 && on->prior_capacity()==0,
          "gate-off and never-frozen paths allocate no temporal scratch");
  }
}
void excitation_policy() {
  auto legacy=algebra_state(true), physical=algebra_state(true,true);
  check(legacy->dt_calib_degenerate() && physical->dt_calib_degenerate(), "both empty owner windows freeze by the unchanged policy");
  for(int i=0;i<3;++i) {
    State::CloneKinematics k; k.vel=V3(.5,0.,0.); k.omega=V3(.01,0.,0.);
    legacy->_clones_kinematics[1.+i]=k;
    State::ExposurePose e; e.camera_id=i%2; e.raw_time=1.+i; e.kinematics=k; physical->_exposure_poses.push_back(e);
  }
  check(legacy->dt_calib_degenerate() && physical->dt_calib_degenerate(), "constant translation and slow rotation freeze in either owner model");
  legacy->_clones_kinematics.begin()->second.omega.x()=.1; physical->_exposure_poses[0].kinematics.omega.x()=.1;
  check(!legacy->dt_calib_degenerate() && !physical->dt_calib_degenerate(), "existing inclusive omega threshold unfreezes both owner models");
  legacy->_clones_kinematics.begin()->second.omega.setZero(); physical->_exposure_poses[0].kinematics.omega.setZero();
  legacy->_clones_kinematics.begin()->second.vel.x()=.8; physical->_exposure_poses[0].kinematics.vel.x()=.8;
  check(!legacy->dt_calib_degenerate() && !physical->dt_calib_degenerate(), "existing velocity-spread threshold unfreezes both owner models");
  physical->_exposure_poses.clear(); physical->_clones_kinematics=legacy->_clones_kinematics;
  check(physical->dt_calib_degenerate(), "physical excitation ignores stale legacy kinematic caches");
  physical->_options.dt_calib_gate=false;
  check(!physical->dt_calib_degenerate(), "disabled policy returns before either owner-window scan");
}
struct Visual {
  std::shared_ptr<State> s;
  std::vector<double> raw;
  V3 point{.7,.4,4.5};
  explicit Visual(bool physical, bool fej=false) {
    StateOptions o; o.num_cameras=2; o.max_clone_size=6; o.do_fej=fej; o.dt_calib_gate=true;
    o.do_calib_camera_timeoffset=true; o.do_calib_camera_readout=!physical; o.physical_camera_clones=physical;
    o.imu_model=StateOptions::RPNG; o.integration_method=StateOptions::DISCRETE;
    o.feat_rep_msckf=LandmarkRepresentation::GLOBAL_3D;
    check(o.configure_clone_policy(false,false),"visual fixture has a valid bounded pose policy");
    s=std::make_shared<State>(o);s->_timestamp=9.98;
    for(int c=0;c<2;++c) {
      VecX intr(8);intr<<400,405,320,240,0,0,0,0;auto camera=std::make_shared<CamRadtan>(640,480);camera->set_value(intr);
      s->_cam_intrinsics_cameras[c]=camera;s->_cam_intrinsics[c]->set_value(intr);s->_cam_intrinsics[c]->set_fej(intr);
      Eigen::Matrix<double,7,1> extr=s->_calib_IMUtoCAM[c]->value();extr(4)=.09*c;
      s->_calib_IMUtoCAM[c]->set_value(extr);s->_calib_IMUtoCAM[c]->set_fej(extr);
      VecX td(1);td<<(c?-.002:.003);s->cam_imu_dt_var(c)->set_value(td);s->cam_imu_dt_var(c)->set_fej(td);
      if(!physical){td<<.006+.001*c;s->_calib_camera_readout[c]->set_value(td);s->_calib_camera_readout[c]->set_fej(td);}
    }
    auto x=s->_imu->value();x.block<3,1>(7,0)<<.65,.08,.04;s->_imu->set_value(x);s->_imu->set_fej(x);
    NoiseManager noise;Propagator p(noise,9.81);
    for(int k=0;k<=170;++k){ImuData z;z.timestamp=9.95+.005*k;z.wm=V3(.01,-.005,.008);z.am=V3(0.,0.,9.81);p.feed_imu(z);}
    for(int k=0;k<6;++k) {
      const double t=10.+.1*k;raw.push_back(t);
      if(physical)for(int c:{1,0}) {
        const double endpoint=t+s->cam_imu_dt(c);Propagator::EndpointKinematics rates;
        check(p.propagate_to_imu(s,endpoint,endpoint-s->cam_imu_dt_ref(),rates),"physical visual fixture reaches each camera endpoint");
        State::ExposurePose v;v.camera_id=c;v.raw_time=t;v.imu_time=endpoint;v.pose=StateHelper::augment_pose_view(s,c,rates.omega);
        v.kinematics.omega=rates.omega;v.kinematics.omega_fej=rates.omega_fej;v.kinematics.vel=s->_imu->vel();v.kinematics.vel_fej=s->_imu->vel_fej();
        s->_exposure_poses.push_back(v);
      } else p.propagate_and_clone(s,t);
    }
    check(s->dt_calib_degenerate(),"visual motion satisfies the original freeze thresholds");
  }
  std::shared_ptr<Feature> track(bool noise=true) const {
    auto f=std::make_shared<Feature>();f->featid=517;f->quality=1.;
    for(int c=0;c<2;++c)for(size_t k=0;k<raw.size();++k) {
      auto view=s->pose_for_camera(c,raw[k]), extr=s->_calib_IMUtoCAM[c];
      V3 local=extr->Rot()*view->Rot()*(point-view->pos())+extr->pos();
      Eigen::Vector2f uv;uv<<400*local.x()/local.z()+320,405*local.y()/local.z()+240;
      if(noise){uv.x()+=.06*std::sin(.9*k+c);uv.y()+=.045*std::cos(.7*k-c);}
      Eigen::Vector2f un;un<<(uv.x()-320)/400.,(uv.y()-240)/405.;
      f->timestamps[c].push_back(raw[k]);f->uvs[c].push_back(uv);f->uvs_norm[c].push_back(un);
    }
    return f;
  }
  HF feature()const {
    const auto t=track();HF f;f.featid=t->featid;f.timestamps=t->timestamps;f.uvs=t->uvs;f.uvs_norm=t->uvs_norm;
    f.feat_representation=LandmarkRepresentation::GLOBAL_3D;f.p_FinG=point;f.p_FinG_fej=point;return f;
  }
};
void visual_update() {
  for(bool physical:{false,true})for(bool fej:{false,true}) {
    Visual f(physical,fej);auto off=StateHelper::clone_state(f.s);off->_options.dt_calib_gate=false;
    Mat Hf,Hx,Hfo,Hxo;VecX r,ro;std::vector<std::shared_ptr<Type>> order,order_off;
    auto feat=f.feature();UpdaterHelper::get_feature_jacobian_full(f.s,feat,Hf,Hx,r,order);
    UpdaterHelper::get_feature_jacobian_full(off,feat,Hfo,Hxo,ro,order_off);
    check(same(Hf,Hfo)&&same(Hx,Hxo)&&same(r,ro),"visual Jacobian/residual retains the complete model under temporal freeze");
    const auto ids=temporal_ids(f.s);const Mat Hd=dense_H(f.s,order,Hx);
    if(!physical) for(int id:ids) check(Hd.col(id).norm()>1e-3,"legacy estimated dt/readout columns remain nonzero while frozen");
    else for(int id:ids) check(Hd.col(id).norm()==0.,"physical clocks act only through owned-pose covariance, with no duplicate visual columns");
    UpdaterHelper::nullspace_project_inplace(Hf,Hx,r);
    UpdaterHelper::measurement_compress_inplace(Hx,r);
    const Mat prior=StateHelper::get_full_covariance(f.s), R=Mat::Identity(r.size(),r.size());
    const auto expected=joseph(prior,dense_H(f.s,order,Hx),R,ids);
    StateHelper::EKFUpdate(f.s,order,Hx,r,R);
    const Mat actual=StateHelper::get_full_covariance(f.s);
    const double error=(actual-expected.P).cwiseAbs().maxCoeff();max_visual=std::max(max_visual,error);
    check(error<3e-13,"projected/compressed real visual system agrees with dense Schmidt Joseph covariance");
    check(same(frozen_block(actual,ids),frozen_block(prior,ids)),"visual update preserves the full frozen temporal uncertainty block");
  }
  for(bool fej:{false,true}) {
    Visual f(true,fej);auto off=StateHelper::clone_state(f.s);off->_options.dt_calib_gate=false;
    const auto ids=temporal_ids(f.s);const Mat prior=StateHelper::get_full_covariance(f.s);
    const double dt0=f.s->cam_imu_dt(0),dt1=f.s->cam_imu_dt(1);
    UpdaterOptions options;FeatureInitializerOptions init;UpdaterMSCKF frozen(options,init),ordinary(options,init);
    std::vector<std::shared_ptr<Feature>> a{f.track()},b{f.track()};frozen.update(f.s,a);ordinary.update(off,b);
    check(a.size()==1&&b.size()==1,"actual physical MSCKF accepts the matched tracks with unchanged noise and gates");
    check(f.s->cam_imu_dt(0)==dt0&&f.s->cam_imu_dt(1)==dt1,"actual physical MSCKF freezes correlated clock means");
    const Mat actual=StateHelper::get_full_covariance(f.s), standard=StateHelper::get_full_covariance(off);
    check(same(frozen_block(actual,ids),frozen_block(prior,ids)),"actual physical MSCKF freezes correlated clock uncertainty");
    Mat expected=standard;for(int i:ids)for(int j:ids)expected(i,j)=prior(i,j);
    check((actual-expected).norm()<1e-13,"actual MSCKF active and clock cross blocks match ordinary posterior");
    check(std::abs(off->cam_imu_dt(0)-dt0)+std::abs(off->cam_imu_dt(1)-dt1)>1e-10,
          "matched gate-off MSCKF control demonstrates a nontrivial indirect clock correction");
  }
}
} // namespace
int main() {
  Printer::setPrintLevel("ERROR");covariance_and_parity();excitation_policy();visual_update();
  std::printf("TEMPORAL_SCHMIDT %s checks=%d failures=%d maxJoseph=%.17g maxMean=%.17g maxVisual=%.17g\n",
              failures?"FAIL":"PASS",checks,failures,max_joseph,max_mean,max_visual);
  return failures?1:0;
}
