/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

namespace {
using namespace ov_msckf;
using namespace ov_core;
using M3 = Eigen::Matrix3d;
using V3 = Eigen::Vector3d;
using Mat = Eigen::MatrixXd;
int failures = 0;
void check(bool condition, const char *message) {
  if (!condition) { ++failures; std::printf("FAIL: %s\n", message); }
}
struct Probe : Propagator {
  using Propagator::Propagator;
  using Propagator::predict_and_compute;
  using Propagator::compute_Xi_sum;
  const Mat &cached_covariance() const { return cache_state_covariance; }
  double cached_time() const { return cache_state_time; }
};

NoiseManager noises() {
  NoiseManager n;
  n.sigma_w = .004; n.sigma_a = .037; n.sigma_wb = .0002; n.sigma_ab = .002;
  return n;
}

std::shared_ptr<State> make_state(StateOptions::ImuModel model, StateOptions::IntegrationMethod method,
                                  int calibration, bool identity = false) {
  StateOptions options;
  options.imu_model = model; options.integration_method = method; options.do_fej = false;
  options.num_cameras = 1; options.do_calib_imu_intrinsics = calibration > 0;
  options.do_calib_imu_g_sensitivity = calibration > 1;
  auto s = std::make_shared<State>(options); s->_timestamp = 10.;
  Eigen::Matrix<double,6,1> dw, da;
  if (model == StateOptions::KALIBR) {
    dw << 1.07,.018,-.012,.94,.021,1.025;
    da << .97,-.024,.016,1.055,-.019,1.02;
  } else {
    dw << 1.07,.018,.94,-.012,.021,1.025;
    da << .97,-.024,1.055,.016,-.019,1.02;
  }
  if (identity) {
    if (model == StateOptions::KALIBR) dw << 1,0,0,1,0,1;
    else dw << 1,0,1,0,0,1;
    da = dw;
  }
  s->_calib_imu_dw->set_value(dw); s->_calib_imu_dw->set_fej(dw);
  s->_calib_imu_da->set_value(da); s->_calib_imu_da->set_fej(da);
  M3 tg;
  tg << .008,-.004,.003, .002,.007,-.005, -.006,.001,.004;
  if (identity) tg.setZero();
  Eigen::Matrix<double,9,1> tv; tv << tg.col(0),tg.col(1),tg.col(2);
  s->_calib_imu_tg->set_value(tv); s->_calib_imu_tg->set_fej(tv);
  const Eigen::Vector4d qa = rot_2_quat(identity ? M3::Identity() : exp_so3(V3(.16,-.11,.07)));
  if (model == StateOptions::RPNG) { s->_calib_imu_ACCtoIMU->set_value(qa); s->_calib_imu_ACCtoIMU->set_fej(qa); }
  else { s->_calib_imu_GYROtoIMU->set_value(qa); s->_calib_imu_GYROtoIMU->set_fej(qa); }
  Eigen::Matrix<double,16,1> x;
  x << rot_2_quat(exp_so3(V3(.31,-.22,.18))), .3,-.4,.15, .2,-.1,.08, .013,-.021,.009, .035,-.026,.019;
  s->_imu->set_value(x); s->_imu->set_fej(x);
  return s;
}

std::vector<std::shared_ptr<ov_type::Type>> ordered(const std::shared_ptr<State> &s) {
  std::vector<std::shared_ptr<ov_type::Type>> result{s->_imu};
  if (s->_options.do_calib_imu_intrinsics) {
    result.push_back(s->_calib_imu_dw); result.push_back(s->_calib_imu_da);
    if (s->_options.do_calib_imu_g_sensitivity) result.push_back(s->_calib_imu_tg);
    result.push_back(s->_options.imu_model == StateOptions::KALIBR ? s->_calib_imu_GYROtoIMU : s->_calib_imu_ACCtoIMU);
  }
  return result;
}

void perturb(const std::shared_ptr<State> &s, int column, double amount) {
  for (const auto &v : ordered(s)) {
    if (column < v->size()) {
      Eigen::VectorXd dx = Eigen::VectorXd::Zero(v->size()); dx(column) = amount; v->update(dx); return;
    }
    column -= v->size();
  }
  check(false, "perturbation column belongs to propagation state");
}

Eigen::VectorXd difference(const std::shared_ptr<State> &a, const std::shared_ptr<State> &b) {
  Eigen::VectorXd result = Eigen::VectorXd::Zero(15+a->imu_intrinsic_size());
  result.head<3>() = -log_so3(a->_imu->Rot()*b->_imu->Rot().transpose());
  result.segment<12>(3) = a->_imu->value().block<12,1>(4,0)-b->_imu->value().block<12,1>(4,0);
  const auto av = ordered(a), bv = ordered(b); int offset = 15;
  for (size_t i=1; i<av.size(); ++i) {
    if (av[i]->size()==3) result.segment<3>(offset) = -log_so3(quat_2_Rot(av[i]->value())*quat_2_Rot(bv[i]->value()).transpose());
    else result.segment(offset,av[i]->size()) = av[i]->value()-bv[i]->value();
    offset += av[i]->size();
  }
  return result;
}

ov_core::ImuData sample(const std::shared_ptr<State> &s, double angular_rate) {
  ov_core::ImuData m; m.timestamp=s->_timestamp;
  m.am = V3(1.2,-.8,9.6)+s->_imu->bias_a();
  const M3 a=s->_calib_imu_ACCtoIMU->Rot()*State::Dm(s->_options.imu_model,s->_calib_imu_da->value());
  const M3 w=s->_calib_imu_GYROtoIMU->Rot()*State::Dm(s->_options.imu_model,s->_calib_imu_dw->value());
  m.wm=s->_imu->bias_g()+State::Tg(s->_calib_imu_tg->value())*a*(m.am-s->_imu->bias_a())+
       w.inverse()*(angular_rate*V3(.7,-.5,.4));
  return m;
}

void transition_fd(StateOptions::ImuModel model, StateOptions::IntegrationMethod method, int calibration, double rate) {
  auto nominal=make_state(model,method,calibration); auto first=sample(nominal,rate), last=first;
  const double dt=.02, eps=2e-6; last.timestamp+=dt;
  Probe p(noises(),9.81); Mat f,q; p.predict_and_compute(nominal,first,last,f,q);
  Mat fd=Mat::Zero(f.rows(),f.cols()); Mat raw_fd=Mat::Zero(f.rows(),6);
  for(int col=0;col<f.cols();++col) {
    auto plus=make_state(model,method,calibration), minus=make_state(model,method,calibration);
    perturb(plus,col,eps); perturb(minus,col,-eps); Mat ignored1,ignored2;
    p.predict_and_compute(plus,first,last,ignored1,ignored2); p.predict_and_compute(minus,first,last,ignored1,ignored2);
    fd.col(col)=difference(plus,minus)/(2*eps);
  }
  for(int col=0;col<6;++col) {
    auto plus=make_state(model,method,calibration), minus=make_state(model,method,calibration);
    auto fp=first,lp=last,fm=first,lm=last;
    if(col<3){fp.wm(col)+=eps;lp.wm(col)+=eps;fm.wm(col)-=eps;lm.wm(col)-=eps;}
    else{fp.am(col-3)+=eps;lp.am(col-3)+=eps;fm.am(col-3)-=eps;lm.am(col-3)-=eps;}
    Mat ignored1,ignored2; p.predict_and_compute(plus,fp,lp,ignored1,ignored2); p.predict_and_compute(minus,fm,lm,ignored1,ignored2);
    raw_fd.col(col)=difference(plus,minus)/(2*eps);
  }
  const auto n=noises(); Eigen::Matrix<double,6,1> variances;
  variances << V3::Constant(n.sigma_w*n.sigma_w/dt),V3::Constant(n.sigma_a*n.sigma_a/dt);
  Mat q_fd=raw_fd*variances.asDiagonal()*raw_fd.transpose();
  q_fd.block<3,3>(9,9)+=n.sigma_wb*n.sigma_wb*dt*M3::Identity();
  q_fd.block<3,3>(12,12)+=n.sigma_ab*n.sigma_ab*dt*M3::Identity();
  // A common perturbation of both readings differentiates an interval-constant
  // draw. Keep testing that raw calibrated gain for every mean integrator; it
  // is not the covariance of continuous white noise. The latter is checked
  // against an independent dense ODE oracle in test_continuous_noise.
  Mat raw_gain=f.block(0,9,f.rows(),6); raw_gain.bottomRows(f.rows()-9).setZero();
  Mat q_constant=raw_gain*variances.asDiagonal()*raw_gain.transpose();
  q_constant.block<3,3>(9,9)+=n.sigma_wb*n.sigma_wb*dt*M3::Identity();
  q_constant.block<3,3>(12,12)+=n.sigma_ab*n.sigma_ab*dt*M3::Identity();
  const double error=(f-fd).cwiseAbs().maxCoeff(),qerror=(q_constant-q_fd).norm()/q_fd.norm();
  int worst=0; for(int i=1;i<f.cols();++i) if((f-fd).col(i).norm()>(f-fd).col(worst).norm())worst=i;
  std::printf("FD model=%d method=%d calib=%d rate=%.4g maxF=%.3e col=%d relQ=%.3e\n",model,method,calibration,rate,error,worst,qerror);
  check(error<2e-7,"full propagation transition matches finite differences of its mean");
  check(qerror<2e-6,"raw gyro/accel finite differences reproduce the calibrated interval-constant noise gain");
  if(method==StateOptions::DISCRETE)
    check((q-q_fd).norm()/q_fd.norm()<2e-6,"DISCRETE retains its interval-constant and endpoint-bias noise convention");
  check(q.block<3,3>(0,6).norm()>1e-9,"nonzero Tg fixture exercises orientation/acceleration noise cross-covariance");
}

void fej_gauge(StateOptions::IntegrationMethod method) {
  auto s=make_state(StateOptions::RPNG,method,2); s->_options.do_fej=true;
  Eigen::Matrix<double,16,1> x=s->_imu->value(); x.head<4>()=rot_2_quat(exp_so3(V3(.03,-.02,.01))*s->_imu->Rot());
  x.segment<3>(4)+=V3(.02,.01,-.03); x.segment<3>(7)+=V3(-.01,.02,.04); s->_imu->set_fej(x);
  const V3 g(0,0,9.81); Eigen::VectorXd nk=Eigen::VectorXd::Zero(15+s->imu_intrinsic_size());
  nk.head<3>()=s->_imu->Rot_fej()*g; nk.segment<3>(3)=-skew_x(s->_imu->pos_fej())*g;
  nk.segment<3>(6)=-skew_x(s->_imu->vel_fej())*g;
  auto first=sample(s,1.),last=first;last.timestamp+=.02;Probe p(noises(),9.81);Mat f,q;p.predict_and_compute(s,first,last,f,q);
  Eigen::VectorXd next=Eigen::VectorXd::Zero(nk.size());next.head<3>()=s->_imu->Rot()*g;
  next.segment<3>(3)=-skew_x(s->_imu->pos())*g;next.segment<3>(6)=-skew_x(s->_imu->vel())*g;
  check((f*nk-next).norm()<1e-11,"FEJ propagation preserves the global-yaw gauge direction with separated current/FEJ state");
}

void bridge_fd(StateOptions::ImuModel model, bool identity) {
  auto build=[&](int column,double delta) {
    auto s=make_state(model,StateOptions::ANALYTICAL,0,identity);
    if(column>=0)perturb(s,9+column,delta);
    Probe p(noises(),9.81);
    auto base=make_state(model,StateOptions::ANALYTICAL,0,identity);
    for(int i=0;i<=12;++i){auto m=sample(base,.8+.03*i);m.timestamp+=i*.005;m.am+=V3(.03*i,-.02*i,.01*i);p.feed_imu(m);}
    Propagator::BridgeData result;check(p.compute_bridge(s,10.,10.06,result),"bridge interval covered");return result;
  };
  const auto nominal=build(-1,0);Eigen::Matrix<double,9,6> fd;const double eps=2e-5;
  for(int col=0;col<6;++col){const auto plus=build(col,eps),minus=build(col,-eps);
    fd.block<3,1>(0,col)=log_so3(plus.DR*minus.DR.transpose())/(2*eps);
    fd.block<3,1>(3,col)=(plus.alpha-minus.alpha)/(2*eps);
    fd.block<3,1>(6,col)=(plus.beta-minus.beta)/(2*eps);}
  const double error=(nominal.J_b-fd).cwiseAbs().maxCoeff();
  std::printf("BRIDGE model=%d identity=%d maxFD=%.3e\n",model,identity,error);
  check(error<1e-7,"bridge full raw bg/ba Jacobian matches finite differences through all calibrated substeps");
  if(!identity)check(nominal.J_b.block<3,3>(0,3).norm()>1e-4,"fixed Tg induces bridge orientation sensitivity to raw accel bias");
}

void fast_covariance(StateOptions::ImuModel model, bool identity) {
  auto slow=make_state(model,StateOptions::DISCRETE,0,identity),fast=make_state(model,StateOptions::DISCRETE,0,identity);
  Mat l=Mat::Identity(15,15);for(int i=0;i<15;++i)for(int j=0;j<i;++j)l(i,j)=.07*std::sin(2*i+j);
  const Mat initial=.002*l*l.transpose();StateHelper::set_initial_covariance(fast,initial,{fast->_imu});
  auto first=sample(slow,.9),last=first;last.timestamp+=.02;
  Probe reference(noises(),9.81),probe(noises(),9.81);Mat f,q;reference.predict_and_compute(slow,first,last,f,q);
  probe.feed_imu(first);probe.feed_imu(last);Eigen::Matrix<double,13,1> predicted;Eigen::Matrix<double,12,12> covariance;
  check(probe.fast_state_propagate(fast,last.timestamp,predicted,covariance),"fast interval covered");
  const double error=(probe.cached_covariance()-(f*initial*f.transpose()+q)).cwiseAbs().maxCoeff();
  std::printf("FAST model=%d identity=%d maxCov=%.3e\n",model,identity,error);
  check(error<1e-12,"fast fixed-calibration covariance equals the audited discrete propagation model");
  check((quat_2_Rot(predicted.head<4>())-slow->_imu->Rot()).norm()<1e-12 &&
        (predicted.segment<3>(4)-slow->_imu->pos()).norm()<1e-12 &&
        (predicted.segment<3>(7)-slow->_imu->Rot()*slow->_imu->vel()).norm()<1e-12,
        "fast propagation preserves calibrated mean and returns body-frame velocity");

  // Differentiate the published coordinate map at the propagated state. The
  // output-noise convention is explicitly independent of that state; this is
  // not a sampled-white-noise oracle for endpoint/propagation correlations.
  auto evaluate=[&](int state_column,int noise_column,double amount) {
    auto s=make_state(model,StateOptions::DISCRETE,0,identity);
    s->_imu->set_value(slow->_imu->value());
    if(state_column>=0)perturb(s,state_column,amount);
    auto m=last;
    if(noise_column>=0 && noise_column<3)m.wm(noise_column)+=amount;
    if(noise_column>=3)m.am(noise_column-3)+=amount;
    const M3 a=s->_calib_imu_ACCtoIMU->Rot()*State::Dm(model,s->_calib_imu_da->value());
    const M3 w=s->_calib_imu_GYROtoIMU->Rot()*State::Dm(model,s->_calib_imu_dw->value());
    Eigen::Matrix<double,13,1> value;
    value<<s->_imu->quat(),s->_imu->pos(),s->_imu->Rot()*s->_imu->vel(),
      w*(m.wm-s->_imu->bias_g()-State::Tg(s->_calib_imu_tg->value())*a*(m.am-s->_imu->bias_a()));
    return value;
  };
  auto delta=[](const Eigen::Matrix<double,13,1> &a,const Eigen::Matrix<double,13,1> &b) {
    Eigen::Matrix<double,12,1> dx;
    dx.head<3>()=-log_so3(quat_2_Rot(a.head<4>())*quat_2_Rot(b.head<4>()).transpose());
    dx.tail<9>()=a.tail<9>()-b.tail<9>();return dx;
  };
  const double eps=2e-6,dt=last.timestamp-first.timestamp;
  Eigen::Matrix<double,12,15> js;
  Eigen::Matrix<double,12,6> jn;
  for(int col=0;col<15;++col)js.col(col)=delta(evaluate(col,-1,eps),evaluate(col,-1,-eps))/(2*eps);
  for(int col=0;col<6;++col)jn.col(col)=delta(evaluate(-1,col,eps),evaluate(-1,col,-eps))/(2*eps);
  const auto n=noises();Eigen::Matrix<double,6,1> variance;
  variance << V3::Constant(n.sigma_w*n.sigma_w/dt),V3::Constant(n.sigma_a*n.sigma_a/dt);
  Eigen::Matrix<double,12,12> expected=js*(f*initial*f.transpose()+q)*js.transpose()+jn*variance.asDiagonal()*jn.transpose();
  const double output_error=(covariance-expected).cwiseAbs().maxCoeff();
  std::printf("FAST_OUTPUT model=%d identity=%d maxMapCovFD=%.3e independent_output_noise_approximation\n",model,identity,output_error);
  check(output_error<2e-10,"published marginal coordinate transform and independent output-noise maps match FD");
  check(Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,12,12>>(covariance).eigenvalues().minCoeff()>-1e-13,
        "published full covariance remains positive semidefinite");
}

void fast_clock() {
  auto s=make_state(StateOptions::RPNG,StateOptions::DISCRETE,0);
  auto expected=make_state(StateOptions::RPNG,StateOptions::DISCRETE,0);
  Eigen::Matrix<double,1,1> td;td<<.011;s->_calib_dt_CAMtoIMU->set_value(td);
  auto first=sample(s,1.);first.timestamp+=td(0);auto last=first;last.timestamp+=.02;auto extra=last;extra.timestamp+=.03;
  Probe p(noises(),9.81),reference(noises(),9.81);p.feed_imu(first);p.feed_imu(last);p.feed_imu(extra);
  Eigen::Matrix<double,13,1> out;Eigen::Matrix<double,12,12> covariance;Mat f,q;
  reference.predict_and_compute(expected,first,last,f,q);
  check(p.fast_state_propagate(s,last.timestamp,out,covariance),"first fast call uses IMU endpoint with nonzero camera offset");
  check(p.cached_time()==last.timestamp && (out.segment<3>(4)-expected->_imu->pos()).norm()<1e-12,
        "camera offset converts cached start only, never the requested IMU endpoint");
  auto next=last;next.timestamp+=.01;reference.predict_and_compute(expected,last,next,f,q);
  check(p.fast_state_propagate(s,next.timestamp,out,covariance),"subsequent fast call uses cached IMU clock");
  check(p.cached_time()==next.timestamp && (out.segment<3>(4)-expected->_imu->pos()).norm()<1e-12,
        "first and subsequent fast calls integrate contiguous IMU-time intervals");
}

void defaults() {
  for(auto model:{StateOptions::KALIBR,StateOptions::RPNG}) {
    StateOptions o;o.imu_model=model;auto s=std::make_shared<State>(o);
    for(const auto &v:{s->_calib_imu_dw,s->_calib_imu_da}) {
      check((State::Dm(model,v->value())-M3::Identity()).norm()==0,"fresh State calibration defaults to identity in each packing");
      check((State::Dm(model,v->fej())-M3::Identity()).norm()==0,"fresh FEJ calibration defaults to identity in each packing");
    }
  }
}

void integration_oracle() {
  const double dt=.025;const V3 direction=V3(.7,-.5,.4).normalized(),a(1.4,-.9,9.7);
  auto s=make_state(StateOptions::RPNG,StateOptions::ANALYTICAL,0);Probe p(noises(),9.81);
  auto quadrature=[&](const V3 &w) {
    // Simpson integration of the rotation itself is independent of Xi's
    // closed forms/series and tests the actual mean, not just self-consistency.
    const int subdivisions=400;Eigen::Matrix<double,3,6> result=Eigen::Matrix<double,3,6>::Zero();
    for(int i=0;i<=subdivisions;++i) {
      const double t=dt*i/subdivisions,weight=(i==0||i==subdivisions)?1:(i%2?4:2);
      const M3 r=exp_so3(w*t);
      result.leftCols<3>()+=weight*r;result.rightCols<3>()+=weight*(dt-t)*r;
    }
    return Eigen::Matrix<double,3,6>(result*(dt/(3*subdivisions)));
  };
  double mean_error=0,derivative_error=0;
  for(double rate:{0.,.001,.006,.01,3.99999,4.00001,30.}) {
    const V3 w=rate*direction;Eigen::Matrix<double,3,18> xi;p.compute_Xi_sum(s,dt,w,a,xi);
    mean_error=std::max(mean_error,(xi.block<3,6>(0,3)-quadrature(w)).cwiseAbs().maxCoeff());
    for(int col=0;col<3;++col) {
      V3 plus=w,minus=w;const double eps=2e-5;plus(col)+=eps;minus(col)-=eps;
      const Eigen::Matrix<double,3,6> fd=(quadrature(plus)-quadrature(minus))/(2*eps);
      derivative_error=std::max(derivative_error,(xi.block<3,3>(0,12).col(col)+fd.leftCols<3>()*a).cwiseAbs().maxCoeff());
      derivative_error=std::max(derivative_error,(xi.block<3,3>(0,15).col(col)+fd.rightCols<3>()*a).cwiseAbs().maxCoeff());
    }
  }
  std::printf("XI_QUADRATURE mean=%.3e derivatives=%.3e (zero, small, branch boundary, large angle)\n",mean_error,derivative_error);
  check(mean_error<1e-12,"Xi mean matches independent quadrature across zero and the small-angle branch boundary");
  check(derivative_error<2e-9,"Xi bias derivatives match independent quadrature derivatives across all angle regimes");
  volatile double checksum=0;
  for(double rate:{0.,1.,30.}) {
    const int repeats=20000;const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<repeats;++i){Eigen::Matrix<double,3,18> xi;p.compute_Xi_sum(s,dt,rate*direction,a,xi);checksum+=xi(0,3);}
    const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
    std::printf("XI_BENCH rate=%.3g host_only_mean_us=%.3f repeats=%d\n",rate,us,repeats);
  }
  check(checksum>0,"integration benchmark evaluations retained");
}
} // namespace

int main() {
  Printer::setPrintLevel("ERROR");defaults();
  for(auto model:{StateOptions::KALIBR,StateOptions::RPNG}) {
    for(auto method:{StateOptions::DISCRETE,StateOptions::RK4,StateOptions::ANALYTICAL})
      for(int calibration:{0,1,2})for(double rate:{1.0,0.0,.006}) transition_fd(model,method,calibration,rate);
    for(bool identity:{false,true}){bridge_fd(model,identity);fast_covariance(model,identity);}
  }
  for(auto method:{StateOptions::DISCRETE,StateOptions::RK4,StateOptions::ANALYTICAL})fej_gauge(method);
  fast_clock();
  integration_oracle();
  std::printf("PROPAGATOR_INTRINSICS %s failures=%d\n",failures?"FAIL":"PASS",failures);return failures?1:0;
}
