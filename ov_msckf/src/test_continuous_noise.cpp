/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "cam/CamRadtan.h"
#include "feat/Feature.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Host-only allocation receipt for the actual shared-library kernel. Eigen's
// runtime malloc switch would require rebuilding every linked Eigen TU with
// matching instrumentation. Interpose the glibc entry points instead; disabled
// during the algebra/caller tests and on platforms without these entry points.
#if defined(__GLIBC__)
static bool record_heap_calls = false;
static size_t recorded_heap_calls = 0;
extern "C" void *__libc_malloc(size_t);
extern "C" void *__libc_calloc(size_t, size_t);
extern "C" void *__libc_realloc(void *, size_t);
extern "C" void *malloc(size_t size) noexcept {
  if (record_heap_calls) ++recorded_heap_calls;
  return __libc_malloc(size);
}
extern "C" void *calloc(size_t count, size_t size) noexcept {
  if (record_heap_calls) ++recorded_heap_calls;
  return __libc_calloc(count, size);
}
extern "C" void *realloc(void *pointer, size_t size) noexcept {
  if (record_heap_calls) ++recorded_heap_calls;
  return __libc_realloc(pointer, size);
}
#endif

using namespace ov_core;
using namespace ov_msckf;
namespace {
using Mat = Eigen::MatrixXd;
using M3 = Eigen::Matrix3d;
using V3 = Eigen::Vector3d;
using M15 = Eigen::Matrix<double, 15, 15>;
int checks = 0, failures = 0;
double max_relative_q = 0., max_relative_split = 0., max_caller = 0.;
void check(bool pass, const char *why) {
  ++checks;
  if (!pass) { ++failures; std::printf("FAIL: %s\n", why); }
}
double relative(const Mat &actual, const Mat &expected) {
  // Standard-deviation normalization prevents the large velocity diagonal
  // from hiding small position/bias blocks or unit-dependent cancellation.
  Eigen::VectorXd scale(expected.rows());
  for (int i = 0; i < scale.size(); ++i) scale(i) = 1. / std::sqrt(std::max(1e-40, expected(i, i)));
  return (scale.asDiagonal() * (actual - expected) * scale.asDiagonal()).cwiseAbs().maxCoeff();
}
struct Probe : Propagator {
  using Propagator::Propagator;
  using Propagator::predict_and_compute;
  using Propagator::compute_Qd_analytic;
  using Propagator::compute_Xi_sum;
};
NoiseManager noise(int component = -1) {
  NoiseManager n;
  n.sigma_w = component < 0 || component == 0 ? .004 : 0.;
  n.sigma_a = component < 0 || component == 1 ? .037 : 0.;
  n.sigma_wb = component < 0 || component == 2 ? .002 : 0.;
  n.sigma_ab = component < 0 || component == 3 ? .02 : 0.;
  return n;
}
std::shared_ptr<State> state(StateOptions::IntegrationMethod method, bool identity = false,
                             StateOptions::ImuModel model = StateOptions::RPNG, bool estimate_calib = false) {
  StateOptions o;
  o.integration_method = method; o.imu_model = model; o.do_fej = false;
  o.num_cameras = 2; o.max_clone_size = 6; o.max_aruco_features = 0;
  o.do_calib_imu_intrinsics = o.do_calib_imu_g_sensitivity = estimate_calib;
  o.physical_camera_clones = true;
  check(o.configure_clone_policy(false, false), "bounded physical clone policy configures");
  auto s = std::make_shared<State>(o); s->_timestamp = 10.;
  if (!identity) {
    Eigen::Matrix<double, 6, 1> dw, da;
    if (model == StateOptions::RPNG) {
      dw << 1.07, .018, .94, -.012, .021, 1.025;
      da << .97, -.024, 1.055, .016, -.019, 1.02;
    } else {
      dw << 1.07, .018, -.012, .94, .021, 1.025;
      da << .97, -.024, .016, 1.055, -.019, 1.02;
    }
    s->_calib_imu_dw->set_value(dw); s->_calib_imu_dw->set_fej(dw);
    s->_calib_imu_da->set_value(da); s->_calib_imu_da->set_fej(da);
    M3 tg; tg << .008, -.004, .003, .002, .007, -.005, -.006, .001, .004;
    Eigen::Matrix<double, 9, 1> tv; tv << tg.col(0), tg.col(1), tg.col(2);
    s->_calib_imu_tg->set_value(tv); s->_calib_imu_tg->set_fej(tv);
    const Eigen::Vector4d qa = rot_2_quat(exp_so3(V3(.16, -.11, .07)));
    const auto rotation = model == StateOptions::RPNG ? s->_calib_imu_ACCtoIMU : s->_calib_imu_GYROtoIMU;
    rotation->set_value(qa); rotation->set_fej(qa);
  }
  Eigen::Matrix<double, 16, 1> x = s->_imu->value();
  x.head<4>() = rot_2_quat(identity ? M3::Identity() : exp_so3(V3(.31, -.22, .18)));
  x.segment<3>(4) << .3, -.4, .15; x.segment<3>(7) << .65, .08, .04;
  x.segment<3>(10) << .013, -.021, .009; x.segment<3>(13) << .035, -.026, .019;
  s->_imu->set_value(x); s->_imu->set_fej(x);
  return s;
}
struct Model {
  M3 W, A, Tg;
  explicit Model(const std::shared_ptr<State> &s) {
    W = s->_calib_imu_GYROtoIMU->Rot() * State::Dm(s->_options.imu_model, s->_calib_imu_dw->value());
    A = s->_calib_imu_ACCtoIMU->Rot() * State::Dm(s->_options.imu_model, s->_calib_imu_da->value());
    Tg = State::Tg(s->_calib_imu_tg->value());
  }
};
ImuData sample(const std::shared_ptr<State> &s, double t, const V3 &w, const V3 &a) {
  const Model m(s); ImuData z; z.timestamp = t;
  z.am = m.A.inverse() * a + s->_imu->bias_a();
  z.wm = m.W.inverse() * w + s->_imu->bias_g() + m.Tg * a;
  return z;
}
struct Reference { M15 F, Q; };
Reference dense(const std::shared_ptr<State> &s, const NoiseManager &n, double dt,
                 const V3 &w, const V3 &a, const M3 &output_rotation) {
  // Independent continuous ODE in [theta, R(t) dp_G, R(t) dv_G, bg_raw, ba_raw].
  // Constant body signals make this 15D generator constant. No production Xi,
  // propagation F/G, quadrature nodes or impulse-response code is used here.
  const Model m(s); const M3 O = -skew_x(w);
  Mat A = Mat::Zero(15, 15), L = Mat::Zero(15, 12);
  A.block<3, 3>(0, 0) = O;
  A.block<3, 3>(0, 9) = -m.W;
  A.block<3, 3>(0, 12) = m.W * m.Tg * m.A;
  A.block<3, 3>(3, 3) = O; A.block<3, 3>(3, 6).setIdentity();
  A.block<3, 3>(6, 0) = -skew_x(a); A.block<3, 3>(6, 6) = O;
  A.block<3, 3>(6, 12) = -m.A;
  L.block<3, 3>(0, 0) = -m.W; L.block<3, 3>(0, 3) = m.W * m.Tg * m.A;
  L.block<3, 3>(6, 3) = -m.A; L.block<3, 3>(9, 6).setIdentity(); L.block<3, 3>(12, 9).setIdentity();
  Eigen::Matrix<double, 12, 1> q;
  q << V3::Constant(n.sigma_w * n.sigma_w), V3::Constant(n.sigma_a * n.sigma_a),
      V3::Constant(n.sigma_wb * n.sigma_wb), V3::Constant(n.sigma_ab * n.sigma_ab);
  Mat van_loan = Mat::Zero(30, 30);
  van_loan.topLeftCorner(15, 15) = A;
  van_loan.topRightCorner(15, 15) = L * q.asDiagonal() * L.transpose();
  van_loan.bottomRightCorner(15, 15) = -A.transpose();
  const Mat E = (dt * van_loan).exp();
  const M3 R0 = s->_options.do_fej ? s->_imu->Rot_fej() : s->_imu->Rot();
  const M3 R1 = exp_so3(-w * dt) * R0;
  M15 T0 = M15::Identity(), T1 = M15::Identity();
  T0.block<3, 3>(3, 3) = T0.block<3, 3>(6, 6) = R0;
  T1.block<3, 3>(0, 0) = output_rotation * R1.transpose();
  T1.block<3, 3>(3, 3) = T1.block<3, 3>(6, 6) = R1.transpose();
  Reference out;
  out.F = T1 * E.topLeftCorner(15, 15) * T0;
  out.Q = T1 * E.topRightCorner(15, 15) * E.topLeftCorner(15, 15).transpose() * T1.transpose();
  out.Q = (0.5 * (out.Q + out.Q.transpose())).eval();
  return out;
}
void stationary_scaling() {
  for (auto method : {StateOptions::ANALYTICAL, StateOptions::RK4})
    for (double dt : {.0001, .00125, .01, .1}) {
      auto s = state(method, true); const auto n = noise(); Probe p(n, 9.81); Mat F, Q;
      p.predict_and_compute(s, sample(s, 10., V3::Zero(), V3::Zero()),
                              sample(s, 10. + dt, V3::Zero(), V3::Zero()), F, Q);
      // Preserve the represented sample endpoints under -ffast-math: directly
      // writing (10+dt)-10 may reassociate to dt and skip timestamp rounding.
      volatile double represented_end = 10. + dt;
      const double t = represented_end - 10.;
      const double a2 = n.sigma_a * n.sigma_a, ab2 = n.sigma_ab * n.sigma_ab;
      const double w2 = n.sigma_w * n.sigma_w, wb2 = n.sigma_wb * n.sigma_wb;
      M15 expected = M15::Zero();
      for (int k = 0; k < 3; ++k) {
        expected(k, k) = w2 * t + wb2 * t*t*t / 3.;
        expected(3+k, 3+k) = a2 * t*t*t / 3. + ab2 * std::pow(t, 5) / 20.;
        expected(6+k, 6+k) = a2 * t + ab2 * t*t*t / 3.;
        expected(3+k, 6+k) = expected(6+k, 3+k) = a2 * t*t / 2. + ab2 * std::pow(t, 4) / 8.;
        expected(k, 9+k) = expected(9+k, k) = -wb2 * t*t / 2.;
        expected(3+k, 12+k) = expected(12+k, 3+k) = -ab2 * t*t*t / 6.;
        expected(6+k, 12+k) = expected(12+k, 6+k) = -ab2 * t*t / 2.;
        expected(9+k, 9+k) = wb2 * t; expected(12+k, 12+k) = ab2 * t;
      }
      const double error=relative(Q,expected);
      if(error>=2e-12)std::printf("STATIONARY method=%d dt=%.17g normalized_error=%.6e max_abs=%.6e\n",
                                method,t,error,(Q-expected).cwiseAbs().maxCoeff());
      check(error < 2e-12, "stationary continuous covariance has dt^3/3 and exact white/bias cross blocks at every sample period");
      check(std::abs(Q(3, 3) - a2*t*t*t/4.) > .08*a2*t*t*t,
            "negative interval-average predecessor is separated from continuous position variance");
    }
  auto s = state(StateOptions::DISCRETE, true); Probe p(noise(1), 9.81); Mat F, Q;
  p.predict_and_compute(s, sample(s, 10., V3::Zero(), V3::Zero()), sample(s, 10.125, V3::Zero(), V3::Zero()), F, Q);
  check(std::abs(Q(3,3) - std::pow(noise(1).sigma_a,2)*std::pow(.125,3)/4.) < 1e-20,
        "explicit DISCRETE interval-constant covariance is preserved");
}
void dense_oracle() {
  const V3 direction = V3(.7, -.5, .4).normalized(), accel(1.2, -.8, 9.6);
  for (auto method : {StateOptions::ANALYTICAL, StateOptions::RK4})
    for (auto model : {StateOptions::RPNG, StateOptions::KALIBR})
      for (int component : {-1, 0, 1, 2, 3})
        for (double angle : {0., .00001, .05, .099999, .100001, .25, .5}) {
          const double dt = .03125; const V3 w = direction * (angle / dt);
          auto s = state(method, false, model, true), before = StateHelper::clone_state(s);
          Probe p(noise(component), 9.81); Mat F, Q;
          p.predict_and_compute(s, sample(s, 10., w, accel), sample(s, 10. + dt, w, accel), F, Q);
          const Reference ref = dense(before, noise(component), dt, w, accel, s->_imu->Rot());
          const double error = relative(Q.topLeftCorner(15, 15), ref.Q);
          max_relative_q = std::max(max_relative_q, error);
          if (error >= 2e-8) std::printf("DENSE method=%d model=%d noise=%d angle=%.6g normalized_error=%.4e\n", method, model, component, angle, error);
          check(error < 2e-8, "calibrated white and bias-RW covariance matches independent dense continuous ODE");
          check((Q.bottomRows(Q.rows()-15).norm() + Q.rightCols(Q.cols()-15).norm()) == 0.,
                "estimated fixed calibration gets no duplicate process noise");
          check(Eigen::SelfAdjointEigenSolver<Mat>(Q).eigenvalues().minCoeff() > -1e-16,
                "positive quadrature preserves PSD including singular isolated noise components");
          if (method == StateOptions::ANALYTICAL)
            check((F.topLeftCorner(15,15) - ref.F).cwiseAbs().maxCoeff() < 1e-10,
                  "declared constant continuous generator reproduces the production analytic transition");
        }
  // Separated current/FEJ rotation and nonconstant endpoints use the analytic
  // linearization's average signals and final output chart, not a fresh chart.
  for (auto method : {StateOptions::ANALYTICAL, StateOptions::RK4}) {
    auto s = state(method); s->_options.do_fej = true;
    Eigen::Matrix<double,16,1> x = s->_imu->fej();
    x.head<4>() = rot_2_quat(exp_so3(V3(.03,-.02,.01)) * s->_imu->Rot()); s->_imu->set_fej(x);
    auto before = StateHelper::clone_state(s); const double dt=.03125;
    const V3 w(.6,-.2,.4), a(1.1,-.9,9.7), dw(.2,-.3,.1), da(.4,-.1,.3);
    Probe p(noise(),9.81); Mat F,Q;
    p.predict_and_compute(s,sample(s,10.,w-dw,a-da),sample(s,10.+dt,w+dw,a+da),F,Q);
    check(relative(Q,dense(before,noise(),dt,w,a,s->_imu->Rot()).Q)<2e-9,
          "continuous Q follows the existing FEJ/average-signal/final-attitude analytic chart");
  }
}
void bias_impulse_small_angles() {
  // The inherited transition uses Jr=I below 1e-6 rad. Continuous bias forcing
  // still has first-order attitude/bias cross terms there; the Q integral must
  // not inherit that identity approximation at individual quadrature nodes.
  const V3 direction = V3(.7, -.5, .4).normalized(), accel(1.2, -.8, 9.6);
  const double dt = .03125;
  for (auto method : {StateOptions::ANALYTICAL, StateOptions::RK4})
    for (int component : {2, 3})
      for (double angle : {0., 1e-8, 1e-7, 5e-7, .999999e-6, 1.00001e-6, 2e-6, 3e-6, 1e-5}) {
        const V3 w = direction * (angle / dt);
        auto s = state(method), before = StateHelper::clone_state(s);
        Probe p(noise(component), 9.81); Mat F, Q;
        p.predict_and_compute(s, sample(s, 10., w, accel), sample(s, 10. + dt, w, accel), F, Q);
        const double error = relative(Q, dense(before, noise(component), dt, w, accel, s->_imu->Rot()).Q);
        max_relative_q = std::max(max_relative_q, error);
        if (error >= 2e-8)
          std::printf("SMALL_BIAS method=%d noise=%d angle=%.9g normalized_error=%.4e\n", method, component, angle, error);
        check(error < 2e-8, "continuous bias impulses retain first-order cross covariance across the inherited tiny-angle Jr branch");
      }
}
void split_and_gauge() {
  for (auto method : {StateOptions::ANALYTICAL, StateOptions::RK4})
    for (double rate : {0., .7, 8.}) {
      auto whole=state(method),split=StateHelper::clone_state(whole);
      const V3 w=rate*V3(.7,-.5,.4),a(1.2,-.8,9.6);const auto n=noise();
      Probe p(n,9.81),q(n,9.81);Mat F,Q,Fs,Qs;
      const auto first=sample(whole,10.,w,a),last=sample(whole,10.03125,w,a);
      p.predict_and_compute(whole,first,last,F,Q);
      M15 accumulated= M15::Zero(),transition=M15::Identity();
      double t=10.;for(double end:{10.00390625,10.01171875,10.03125}) {
        q.predict_and_compute(split,sample(split,t,w,a),sample(split,end,w,a),Fs,Qs);
        accumulated=(Fs*accumulated*Fs.transpose()+Qs).eval();transition=(Fs*transition).eval();t=end;
      }
      const double error=relative(accumulated,Q);max_relative_split=std::max(max_relative_split,error);
      // RK4 mean error makes exact semigroup equality inappropriate; this
      // tolerance bounds that declared discretization at |omega|dt <= .238.
      check(error<(method==StateOptions::ANALYTICAL?2e-9:2e-6),
            "exposure splitting preserves the continuous process integral to its mean/quadrature order");
      if(method==StateOptions::ANALYTICAL)
        check((transition-F).norm()<1e-10,"constant analytic transition obeys interval composition");
    }
  auto s=state(StateOptions::ANALYTICAL),rotated=StateHelper::clone_state(s);
  const M3 yaw=exp_so3(V3(0.,0.,.8));Eigen::Matrix<double,16,1> x=rotated->_imu->value();
  x.head<4>()=rot_2_quat(s->_imu->Rot()*yaw.transpose());x.segment<3>(4)=yaw*s->_imu->pos();x.segment<3>(7)=yaw*s->_imu->vel();
  rotated->_imu->set_value(x);rotated->_imu->set_fej(x);
  const V3 w(.8,-.4,.2),a(1.,-.7,9.7);Probe p(noise(),9.81);Mat F,Q,Fr,Qr;
  p.predict_and_compute(s,sample(s,10.,w,a),sample(s,10.03125,w,a),F,Q);
  p.predict_and_compute(rotated,sample(rotated,10.,w,a),sample(rotated,10.03125,w,a),Fr,Qr);
  M15 T=M15::Identity();T.block<3,3>(3,3)=T.block<3,3>(6,6)=yaw;
  check(relative(Qr,T*Q*T.transpose())<1e-12,"world yaw changes only world position/velocity noise coordinates; raw biases stay raw");
}
void owned_pose_and_visual() {
  for (auto method : {StateOptions::ANALYTICAL,StateOptions::RK4}) {
    auto s=state(method);const V3 w(.2,-.12,.16),a(.1,-.08,9.81);const auto n=noise();
    M15 L=M15::Identity();for(int i=0;i<15;++i)for(int j=0;j<i;++j)L(i,j)=.025*std::sin(i+2*j);
    Mat expected=.001*L*L.transpose();StateHelper::set_initial_covariance(s,expected,{s->_imu});
    for(int c=0;c<2;++c) {
      Eigen::VectorXd intrinsic(8);intrinsic<<400,405,320,240,0,0,0,0;auto camera=std::make_shared<CamRadtan>(640,480);camera->set_value(intrinsic);
      s->_cam_intrinsics_cameras[c]=camera;s->_cam_intrinsics[c]->set_value(intrinsic);s->_cam_intrinsics[c]->set_fej(intrinsic);
      Eigen::Matrix<double,7,1> extrinsic;extrinsic<<0,0,0,1,.08*c,0,0;
      s->_calib_IMUtoCAM[c]->set_value(extrinsic);s->_calib_IMUtoCAM[c]->set_fej(extrinsic);
    }
    Propagator p(n,9.81);p.feed_imu(sample(s,10.,w,a));p.feed_imu(sample(s,11.,w,a));
    double previous=10.;std::vector<double> raw;
    for(int k=0;k<6;++k) {
      const double endpoint=10.03125+.0625*k;raw.push_back(endpoint);
      auto before=StateHelper::clone_state(s);Propagator::EndpointKinematics rates;
      check(p.propagate_to_imu(s,endpoint,endpoint,rates),"actual propagator reaches interpolated physical exposure endpoint");
      const Reference ref=dense(before,n,endpoint-previous,w,a,s->_imu->Rot());
      Mat transition=Mat::Identity(expected.rows(),expected.cols());transition.topLeftCorner(15,15)=ref.F;
      expected=(transition*expected*transition.transpose()).eval();expected.topLeftCorner(15,15)+=ref.Q;
      for(int c=0;c<2;++c) {
        if(c==1)check(p.propagate_to_imu(s,endpoint,endpoint,rates),"second same-endpoint camera adds no process interval");
        const int old=expected.rows();Mat J=Mat::Zero(old+6,old);J.topRows(old).setIdentity();J.block(old,0,6,6).setIdentity();
        expected=(J*expected*J.transpose()).eval();
        State::ExposurePose owner;owner.camera_id=c;owner.raw_time=endpoint;owner.imu_time=endpoint;
        owner.pose=StateHelper::augment_pose_view(s,c,rates.omega);owner.kinematics.omega=rates.omega;owner.kinematics.omega_fej=rates.omega_fej;
        owner.kinematics.vel=s->_imu->vel();owner.kinematics.vel_fej=s->_imu->vel_fej();s->_exposure_poses.push_back(owner);
        const double error=(StateHelper::get_full_covariance(s)-expected).cwiseAbs().maxCoeff();max_caller=std::max(max_caller,error);
        check(error<2e-10,"owned physical poses and all historical cross blocks match dense continuous joint propagation");
      }
      previous=endpoint;
    }
    check(s->max_covariance_size()==15+12*6&&s->_exposure_poses.size()==12,"process repair adds no state or per-feature covariance blocks");
    auto reference=StateHelper::clone_state(s);
    std::vector<std::shared_ptr<ov_type::Type>> order{reference->_imu};for(const auto &v:reference->_exposure_poses)order.push_back(v.pose);
    StateHelper::set_initial_covariance(reference,expected,order);
    auto make_track=[&]() {
      auto f=std::make_shared<Feature>();f->featid=123;f->quality=1.;const V3 point(.9,.25,4.8);
      for(int c=0;c<2;++c)for(double t:raw) {
        const auto pose=s->pose_for_camera(c,t),extr=s->_calib_IMUtoCAM.at(c);
        const V3 local=extr->Rot()*pose->Rot()*(point-pose->pos())+extr->pos();
        Eigen::Vector2f uv;uv<<400.*local.x()/local.z()+320.+.02*std::sin(t*3+c),405.*local.y()/local.z()+240.+.015*std::cos(t*4-c);
        Eigen::Vector2f un;un<<(uv.x()-320.)/400.,(uv.y()-240.)/405.;
        f->timestamps[c].push_back(t);f->uvs[c].push_back(uv);f->uvs_norm[c].push_back(un);
      }return f;
    };
    std::vector<std::shared_ptr<Feature>> tracks{make_track()},oracle_tracks{make_track()};
    UpdaterOptions opts;FeatureInitializerOptions init;UpdaterMSCKF update(opts,init),oracle_update(opts,init);
    const Mat prior=StateHelper::get_full_covariance(s);update.update(s,tracks);oracle_update.update(reference,oracle_tracks);
    check(tracks.size()==1&&oracle_tracks.size()==1&&(StateHelper::get_full_covariance(s)-prior).norm()>1e-8,
          "actual MSCKF triangulates and updates the continuously propagated owned exposure views under unchanged gates");
    check((StateHelper::get_full_covariance(s)-StateHelper::get_full_covariance(reference)).cwiseAbs().maxCoeff()<2e-10&&
          (s->_imu->value()-reference->_imu->value()).norm()<2e-10,
          "actual visual posterior agrees with independent dense joint process prior without extra bridge noise");
  }
}
void resources(bool timing) {
  auto s=state(StateOptions::ANALYTICAL);Probe p(noise(),9.81);
  const double dt=.00125;const V3 w(.8,-.5,.4),a(1.2,-.8,9.6);
  Eigen::Matrix<double,3,18> xi;p.compute_Xi_sum(s,dt,w,a,xi);
  const Eigen::Vector4d q=rot_2_quat(exp_so3(-dt*w)*s->_imu->Rot());
  volatile double checksum=p.compute_Qd_analytic(s,dt,w,a,q,xi)(0,0);
  const int repeats=timing?20000:200;
  const auto start=std::chrono::steady_clock::now();
#if defined(__GLIBC__)
  recorded_heap_calls=0;record_heap_calls=true;
#endif
  for(int i=0;i<repeats;++i)checksum+=p.compute_Qd_analytic(s,dt,w,a,q,xi)(0,0);
#if defined(__GLIBC__)
  record_heap_calls=false;
  check(recorded_heap_calls==0,"continuous process kernel makes no heap allocations after setup");
  std::printf("CONTINUOUS_NOISE_RESOURCE calls=%d heap_calls=%zu fixed_nodes=6 max_explicit_matrix=15x15 added_state_dim=0\n",
              repeats,recorded_heap_calls);
#endif
  if(timing)std::printf("CONTINUOUS_NOISE_HOST_TIMING mean_us=%.6f calls=%d (host-only; valid only without competing workloads)\n",
                        std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats,repeats);
  check(checksum>0,"resource receipt executes the actual calibrated production kernel");
}
} // namespace
int main(int argc,char **argv) {
  Printer::setPrintLevel("ERROR");
  const bool skip_resources=argc>1&&std::strcmp(argv[1],"--skip-resources")==0;
  const bool small_angles=argc>1&&std::strcmp(argv[1],"--small-angles")==0;
  if(argc>1&&!skip_resources&&!small_angles) {
    resources(std::strcmp(argv[1],"--benchmark")==0);
    return failures?1:0;
  }
  if (small_angles) {
    bias_impulse_small_angles();
  } else {
    stationary_scaling();dense_oracle();bias_impulse_small_angles();split_and_gauge();owned_pose_and_visual();
    if(!skip_resources)resources(false);
  }
  std::printf("CONTINUOUS_NOISE checks=%d failures=%d max_normalized_Q=%.4e max_normalized_split=%.4e max_caller_abs=%.4e\n",
              checks,failures,max_relative_q,max_relative_split,max_caller);
  return failures?1:0;
}
