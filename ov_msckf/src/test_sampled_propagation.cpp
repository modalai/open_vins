/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "state/Propagator.h"
#include "state/StateHelper.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#if defined(__GLIBC__) && (defined(__x86_64__) || defined(__aarch64__))
#include <pthread.h>
#endif

#ifdef __GLIBC__
static thread_local bool count_heap=false;
static thread_local size_t heap_calls=0,heap_bytes=0,heap_largest=0;
static void allocation(size_t n) {
  if(count_heap) { ++heap_calls; heap_bytes+=n; heap_largest=std::max(heap_largest,n); }
}
extern "C" void *__libc_malloc(size_t);
extern "C" void *__libc_calloc(size_t,size_t);
extern "C" void *__libc_realloc(void*,size_t);
extern "C" void *__libc_memalign(size_t,size_t);
extern "C" void *malloc(size_t n) noexcept { allocation(n); return __libc_malloc(n); }
extern "C" void *calloc(size_t n,size_t s) noexcept { allocation(n*s); return __libc_calloc(n,s); }
extern "C" void *realloc(void *p,size_t n) noexcept { allocation(n); return __libc_realloc(p,n); }
extern "C" void *aligned_alloc(size_t a,size_t n) noexcept { allocation(n); return __libc_memalign(a,n); }
extern "C" int posix_memalign(void **p,size_t a,size_t n) noexcept {
  if(a<sizeof(void*) || (a&(a-1)))return EINVAL;
  allocation(n); void *v=__libc_memalign(a,n); if(!v)return ENOMEM; *p=v; return 0;
}
#endif

using namespace ov_msckf;
using namespace ov_core;
namespace {
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
using V6 = Eigen::Matrix<double,6,1>;
using X = Eigen::Matrix<double,16,1>;
using Segment = Propagator::SampledImuSegment;
using Record = State::SampledImuRecord;
using Linearization = Propagator::SampledPropagationLinearization;
int checks=0, failures=0;
double max_fd=0., max_dense=0., max_mean=0., max_q=0.;
void check(bool ok, const char *why) { ++checks; if (!ok) { ++failures; std::printf("FAIL: %s\n",why); } }
template<class A,class B> bool same(const Eigen::MatrixBase<A> &a,const Eigen::MatrixBase<B> &b) {
  const Mat aa=a,bb=b;
  return aa.rows()==bb.rows() && aa.cols()==bb.cols() && std::memcmp(aa.data(),bb.data(),aa.size()*sizeof(double))==0;
}
struct Probe : Propagator {
  using Propagator::Propagator;
  using Propagator::predict_and_compute;
  using Propagator::predict_mean_rk4;
  void seed_cache() { cache_imu_valid=true; cache_state_time=913.; }
  bool cache_valid() const { return cache_imu_valid.load(); }
  double cache_time() const { return cache_state_time; }
};
NoiseManager noises(bool bias=true) {
  NoiseManager n; n.sigma_w=.004; n.sigma_a=.037; n.sigma_wb=bias?.012:0.; n.sigma_ab=bias?.035:0.; return n;
}
std::shared_ptr<State> make_state(StateOptions::IntegrationMethod method,StateOptions::ImuModel model=StateOptions::RPNG,
                                bool identity=false, bool prepare=true) {
  StateOptions o; o.integration_method=method; o.imu_model=model; o.do_fej=false;
  o.num_cameras=2; o.do_calib_camera_timeoffset=true; o.physical_camera_clones=true; o.max_clone_size=6;
  o.configure_clone_policy(false,false);
  auto s=std::make_shared<State>(o); s->_timestamp=0.; s->_imu_endpoint=0.; s->_imu_endpoint_valid=true;
  V6 dw,da;
  if(model==StateOptions::RPNG) { dw<<1.07,.018,.94,-.012,.021,1.025; da<<.97,-.024,1.055,.016,-.019,1.02; }
  else { dw<<1.07,.018,-.012,.94,.021,1.025; da<<.97,-.024,.016,1.055,-.019,1.02; }
  if(identity) { if(model==StateOptions::RPNG) dw<<1,0,1,0,0,1; else dw<<1,0,0,1,0,1; da=dw; }
  s->_calib_imu_dw->set_value(dw); s->_calib_imu_da->set_value(da);
  M3 tg; tg<<.028,-.014,.013,.012,.027,-.015,-.016,.011,.024; if(identity)tg.setZero();
  Eigen::Matrix<double,9,1> tv; tv<<tg.col(0),tg.col(1),tg.col(2); s->_calib_imu_tg->set_value(tv);
  auto rotation=model==StateOptions::RPNG?s->_calib_imu_ACCtoIMU:s->_calib_imu_GYROtoIMU;
  rotation->set_value(rot_2_quat(identity?M3::Identity():exp_so3(V3(.16,-.11,.07))));
  X x=s->_imu->value(); x.head<4>()=rot_2_quat(identity?M3::Identity():exp_so3(V3(.31,-.22,.18)));
  x.segment<3>(4)<<.3,-.4,.15; x.segment<3>(7)<<.65,.08,.04;
  x.segment<3>(10)<<.013,-.021,.009; x.segment<3>(13)<<.035,-.026,.019;
  s->_imu->set_value(x); s->_imu->set_fej(x);
  const int n=s->max_covariance_size(); Mat L=Mat::Zero(n,n);
  for(int i=0;i<n;++i) for(int j=0;j<=i;++j) L(i,j)=i==j?.04+.002*i:.0015*std::sin(i+.4*j);
  StateHelper::set_initial_covariance(s,L*L.transpose(),{s->_imu,s->cam_imu_dt_var(0),s->cam_imu_dt_var(1)});
  if(prepare) check(StateHelper::prepare_sampled_imu_boundary(s,73),"two persistent slots prepare");
  return s;
}
struct Calibration {
  M3 A,W,T;
  explicit Calibration(const std::shared_ptr<State> &s) {
    A=s->_calib_imu_ACCtoIMU->Rot()*State::Dm(s->_options.imu_model,s->_calib_imu_da->value());
    W=s->_calib_imu_GYROtoIMU->Rot()*State::Dm(s->_options.imu_model,s->_calib_imu_dw->value());
    T=State::Tg(s->_calib_imu_tg->value());
  }
};
Record record(uint64_t seq,double t,const std::shared_ptr<State> &s,double rate=.8) {
  Record r; r.stream_episode=73; r.sequence=seq; r.timestamp=t; Calibration c(s);
  V3 a(1.2+.1*seq,-.8+.04*seq,9.6), w=rate*V3(.7,-.5,.4).normalized();
  r.measured.tail<3>()=c.A.inverse()*a+s->_imu->bias_a();
  r.measured.head<3>()=c.W.inverse()*w+s->_imu->bias_g()+c.T*a;
  Eigen::Matrix<double,6,6> L=Eigen::Matrix<double,6,6>::Zero();
  for(int i=0;i<6;++i) for(int j=0;j<=i;++j) L(i,j)=i==j?.08+.02*i:.008*(i+j+1);
  r.prior=L*L.transpose(); return r;
}
Segment segment(const std::vector<Record> &r,double a,double b) {
  std::vector<Segment> out; out.reserve(r.size());
  check(Propagator::select_sampled_imu_readings(r,a,b,out) && out.size()==1,"actual selector returns original pair");
  return out.front();
}
V6 posterior(const std::shared_ptr<State> &s,uint64_t sequence) {
  for(const auto &slot:s->sampled_imu_slots()) if(slot.active && slot.record.sequence==sequence)return slot.noise->value();
  return V6::Zero();
}
X retract(X x,const Vec &d) {
  Eigen::Vector4d dq; dq<<.5*d.head<3>(),1.; dq.normalize();
  x.head<4>()=quat_multiply(dq,x.head<4>()); x.tail<12>()+=d.segment<12>(3); return x;
}
Vec difference(const X &a,const X &b) {
  Vec d(15); d.head<3>()=-log_so3(quat_2_Rot(a.head<4>())*quat_2_Rot(b.head<4>()).transpose());
  d.tail<12>()=a.tail<12>()-b.tail<12>(); return d;
}
// Independent RK4 implementation integrates the full global quaternion, not
// the production incremental quaternion. Scalar quaternion ODE/rotation
// formulas avoid production Omega, quat_multiply, stage/tangent helpers.
X reference_rk4(const X &x, const Calibration &c, const Segment &s, const V6 &n0, const V6 &n1) {
  using Stage = Eigen::Matrix<double, 10, 1>;
  const double dt = s.time1 - s.time0;
  Stage y0; y0 << x.head<4>(), x.segment<3>(4), x.segment<3>(7);
  auto rhs = [&](double fraction, Stage y) -> Stage {
    const Eigen::Vector2d weights = (1. - fraction) * s.weights0 + fraction * s.weights1;
    const V6 raw = weights(0) * (s.records[0].measured - n0) + weights(1) * (s.records[1].measured - n1);
    const V3 a = c.A * (raw.tail<3>() - x.segment<3>(13));
    const V3 w = c.W * (raw.head<3>() - x.segment<3>(10) - c.T * a);
    // Do not canonicalize intermediate global quaternions: the ODE remains
    // smooth through the q_scalar=0 representation boundary.
    y.head<4>() /= y.head<4>().norm();
    const double qx=y(0), qy=y(1), qz=y(2), qs=y(3);
    Stage derivative;
    derivative(0)=.5*(qs*w.x()+qy*w.z()-qz*w.y());
    derivative(1)=.5*(qs*w.y()+qz*w.x()-qx*w.z());
    derivative(2)=.5*(qs*w.z()+qx*w.y()-qy*w.x());
    derivative(3)=-.5*(qx*w.x()+qy*w.y()+qz*w.z());
    M3 rotation;
    rotation << 2.*qs*qs-1.+2.*qx*qx, 2.*qx*qy+2.*qs*qz, 2.*qx*qz-2.*qs*qy,
                2.*qx*qy-2.*qs*qz, 2.*qs*qs-1.+2.*qy*qy, 2.*qy*qz+2.*qs*qx,
                2.*qx*qz+2.*qs*qy, 2.*qy*qz-2.*qs*qx, 2.*qs*qs-1.+2.*qz*qz;
    derivative.segment<3>(4)=y.tail<3>();
    derivative.tail<3>()=rotation.transpose()*a-V3(0.,0.,9.81);
    return derivative;
  };
  const Stage k1=dt*rhs(0.,y0), k2=dt*rhs(.5,y0+.5*k1),
              k3=dt*rhs(.5,y0+.5*k2), k4=dt*rhs(1.,y0+k3);
  Stage y=y0+(k1+2.*k2+2.*k3+k4)/6.;
  y.head<4>()/=y.head<4>().norm();
  if(y(3)<0.)y.head<4>()*=-1.;
  X result=x; result.head<10>()=y; return result;
}

// Independent mean: AngleAxis exponential and 40-term integral series, never
// production Xi/F/G. Differentiation below uses JPL retraction at the input.
X reference_mean(const X &x,const Calibration &c,const Segment &s,const V6 &n0,const V6 &n1,
                 StateOptions::IntegrationMethod method,V3 *angular=nullptr,V3 *acceleration=nullptr) {
  const double dt=s.time1-s.time0; const Eigen::Vector2d weights=.5*(s.weights0+s.weights1);
  const V6 raw=weights(0)*(s.records[0].measured-n0)+weights(1)*(s.records[1].measured-n1);
  const V3 a=c.A*(raw.tail<3>()-x.segment<3>(13));
  const V3 w=c.W*(raw.head<3>()-x.segment<3>(10)-c.T*a); const M3 R=quat_2_Rot(x.head<4>());
  const V3 v=x.segment<3>(7), g(0,0,9.81), delta=-w*dt;
  const M3 dR=delta.norm()>0.?Eigen::AngleAxisd(delta.norm(),delta.normalized()).toRotationMatrix():M3::Identity();
  X out=x; out.head<4>()=rot_2_quat(dR*R);
  M3 I1=dt*M3::Identity(), I2=.5*dt*dt*M3::Identity();
  if(method==StateOptions::ANALYTICAL) {
    M3 power=M3::Identity(); double f1=dt,f2=.5*dt*dt;
    for(int k=1;k<=40;++k) { power=(power*skew_x(w*dt)).eval(); f1/=k+1.; f2/=k+2.; I1+=f1*power; I2+=f2*power; }
  }
  out.segment<3>(7)=v+R.transpose()*I1*a-g*dt;
  out.segment<3>(4)=x.segment<3>(4)+dt*v+R.transpose()*I2*a-.5*dt*dt*g;
  if(angular)*angular=w; if(acceleration)*acceleration=a;
  return method==StateOptions::RK4 ? reference_rk4(x,c,s,n0,n1) : out;
}
Mat reference_derivative(const X &x,const Calibration &c,const Segment &s,const V6 &n0,const V6 &n1,
                         StateOptions::IntegrationMethod method) {
  const double eps=2e-6; Mat f(15,27);
  for(int j=0;j<27;++j) {
    X xp=x,xm=x; V6 p0=n0,m0=n0,p1=n1,m1=n1;
    if(j<15) { Vec d=Vec::Zero(15); d(j)=eps; xp=retract(x,d); xm=retract(x,-d); }
    else if(j<21) { p0(j-15)+=eps; m0(j-15)-=eps; }
    else { p1(j-21)+=eps; m1(j-21)-=eps; }
    f.col(j)=difference(reference_mean(xp,c,s,p0,p1,method),reference_mean(xm,c,s,m0,m1,method))/(2*eps);
  }
  return f;
}
Mat bias_oracle(const X &x,const Calibration &c,const Segment &s,const V6 &n0,const V6 &n1,
                StateOptions::IntegrationMethod method,const NoiseManager &noise) {
  const double dt=s.time1-s.time0;
  Mat q=Mat::Zero(15,15);
  q.block<3,3>(9,9)=noise.sigma_wb*noise.sigma_wb*M3::Identity();
  q.block<3,3>(12,12)=noise.sigma_ab*noise.sigma_ab*M3::Identity();
  if(method==StateOptions::DISCRETE)return dt*q;
  V3 w,a; const X next=reference_mean(x,c,s,n0,n1,method,&w,&a);
  Mat f=Mat::Zero(15,15); const M3 O=-skew_x(w);
  f.block<3,3>(0,0)=O; f.block<3,3>(0,9)=-c.W; f.block<3,3>(0,12)=c.W*c.T*c.A;
  f.block<3,3>(3,3)=O; f.block<3,3>(3,6).setIdentity();
  f.block<3,3>(6,0)=-skew_x(a); f.block<3,3>(6,6)=O; f.block<3,3>(6,12)=-c.A;
  Mat v=Mat::Zero(30,30); v.topLeftCorner(15,15)=f; v.topRightCorner(15,15)=q; v.bottomRightCorner(15,15)=-f.transpose();
  const Mat e=(dt*v).exp(); Mat T=Mat::Identity(15,15);
  const V3 angle=-dt*w;
  const M3 increment=angle.norm()>0.?Eigen::AngleAxisd(angle.norm(),angle.normalized()).toRotationMatrix():M3::Identity();
  const M3 analytic_end=increment*quat_2_Rot(x.head<4>());
  T.block<3,3>(0,0)=quat_2_Rot(next.head<4>())*analytic_end.transpose();
  T.block<3,3>(3,3)=T.block<3,3>(6,6)=analytic_end.transpose();
  return T*e.topRightCorner(15,15)*e.topLeftCorner(15,15).transpose()*T.transpose();
}

void derivatives() {
  for(auto method:{StateOptions::ANALYTICAL,StateOptions::DISCRETE,StateOptions::RK4})
    for(auto model:{StateOptions::RPNG,StateOptions::KALIBR})
      for(double angle:{0.,1e-8,3e-7,1e-6,.099999,.100001,.49}) {
        auto s=make_state(method,model); const double dt=.03125;
        std::vector<Record> r{record(1,0.,s,angle/(.8*dt)),record(2,dt,s,angle/(.8*dt))};
        // Same acceleration makes the requested angular increment exact.
        r[1].measured=r[0].measured;
        auto seg=segment(r,0.,dt*.8);
        const V6 n0=(V6()<<.003,-.002,.001,.012,-.006,.004).finished(), n1=-.4*n0;
        // Keep the corrected rate at the requested branch-test value despite
        // nonzero posterior means; otherwise those means hide the Jr limit.
        const Eigen::Vector2d blend=.5*(seg.weights0+seg.weights1);
        for(auto &v:r)v.measured+=blend(0)*n0+blend(1)*n1;
        seg.records={r[0],r[1]};
        auto left=StateHelper::admit_sampled_imu_noise(s,r[0]),right=StateHelper::admit_sampled_imu_noise(s,r[1]);
        left->set_value(n0); right->set_value(n1);
        auto before=StateHelper::clone_state(s); Calibration c(s); const X x=s->_imu->value();
        const Mat expected=reference_derivative(x,c,seg,n0,n1,method), q=bias_oracle(x,c,seg,n0,n1,method,noises());
        Probe p(noises(),9.81); Propagator::EndpointKinematics endpoint; Linearization result;
        check(p.propagate_sampled_segment(s,seg,seg.time1,endpoint,&result),"actual sampled caller accepts calibrated current-mean segment");
        const double err=(result.Phi-expected).cwiseAbs().maxCoeff(); max_fd=std::max(max_fd,err);
        check(err<2e-8,"all 27 actual mean tangent columns match independent finite differences");
        const double meanerr=difference(s->_imu->value(),reference_mean(x,c,seg,n0,n1,method)).norm();
        max_mean=std::max(max_mean,meanerr); check(meanerr<2e-12,"actual nonlinear mean matches independent integral");
        Vec scale(15); for(int i=0;i<15;++i)scale(i)=1./std::sqrt(std::max(1e-40,q(i,i)));
        const double qe=(scale.asDiagonal()*(result.independent_Q-q)*scale.asDiagonal()).cwiseAbs().maxCoeff(); max_q=std::max(max_q,qe);
        check(qe<2e-8,"independent Q matches dense bias-only Van Loan integral or declared discrete kick");
        // Differentiate the actual public caller as a second, implementation-
        // independent check that the diagnostic Phi is the map it commits.
        Mat actual_fd(15,27);
        for(int j=0;j<27;++j) {
          auto plus=StateHelper::clone_state(before),minus=StateHelper::clone_state(before); const double eps=2e-6;
          if(j<15) { Vec d=Vec::Zero(15); d(j)=eps; plus->_imu->update(d); minus->_imu->update(-d); }
          else {
            const auto seq=r[j<21?0:1].sequence;
            for(auto b:{plus,minus}) for(const auto &slot:b->sampled_imu_slots()) if(slot.record.sequence==seq) {
              auto value=slot.noise->value(); value((j-15)%6)+=b==plus?eps:-eps; slot.noise->set_value(value);
            }
          }
          const bool ok=p.propagate_sampled_segment(plus,seg,seg.time1,endpoint) && p.propagate_sampled_segment(minus,seg,seg.time1,endpoint);
          check(ok,"actual perturbed public caller accepts tangent fixture");
          actual_fd.col(j)=difference(plus->_imu->value(),minus->_imu->value())/(2*eps);
        }
        const double actualerr=(result.Phi-actual_fd).cwiseAbs().maxCoeff(); max_fd=std::max(max_fd,actualerr);
        check(actualerr<2e-8,"Phi differentiates the actual linked caller including raw posterior means");
        auto huge=noises(); huge.sigma_w=100.; huge.sigma_a=200.; Probe control(huge,9.81);
        check(control.propagate_sampled_segment(before,seg,seg.time1,endpoint) &&
              same(StateHelper::get_full_covariance(before),StateHelper::get_full_covariance(s)),
              "configured sensor densities have no sampled covariance effect (no double Q)");
      }
}

// Dense oracle retains every original raw sample for the whole experiment.
// Production holds only two reusable slots; comparisons project this larger
// joint Gaussian onto the currently retained variables. No production Phi is
// used to evolve the oracle and no discarded record is reintroduced as a prior.
struct Dense {
  Mat P; Vec mean; X nav; std::vector<Eigen::Matrix<double,7,1>> poses;
  std::vector<Record> records; Calibration calibration;
  Dense(const std::shared_ptr<State> &s,const std::vector<Record> &r):nav(s->_imu->value()),records(r),calibration(s) {
    const int initial=17, n=initial+6*r.size(); P=Mat::Zero(n,n); mean=Vec::Zero(n);
    P.topLeftCorner(initial,initial)=StateHelper::get_full_covariance(s).topLeftCorner(initial,initial);
    mean.segment<12>(3)=nav.tail<12>();
    for(size_t i=0;i<r.size();++i)P.block<6,6>(initial+6*i,initial+6*i)=r[i].prior;
  }
  int raw(uint64_t sequence) const { for(size_t i=0;i<records.size();++i)if(records[i].sequence==sequence)return 17+6*i; return -1; }
  std::vector<int> indices(const std::shared_ptr<State> &s) const {
    std::vector<int> ids(s->max_covariance_size(),-1); for(int i=0;i<17;++i)ids[i]=i;
    for(const auto &slot:s->sampled_imu_slots()) if(slot.active)for(int j=0;j<6;++j)ids[slot.noise->id()+j]=raw(slot.record.sequence)+j;
    for(size_t k=0;k<s->_exposure_poses.size();++k)for(int j=0;j<6;++j)ids[s->_exposure_poses[k].pose->id()+j]=17+6*records.size()+6*k+j;
    return ids;
  }
  void compare(const std::shared_ptr<State> &s,const char *why) {
    const auto map=indices(s); Mat expected=Mat::Zero(map.size(),map.size());
    for(size_t i=0;i<map.size();++i)for(size_t j=0;j<map.size();++j)if(map[i]>=0&&map[j]>=0)expected(i,j)=P(map[i],map[j]);
    double ce=(StateHelper::get_full_covariance(s)-expected).cwiseAbs().maxCoeff(); max_dense=std::max(max_dense,ce);
    check(ce<3e-9,why);
    double me=difference(s->_imu->value(),nav).norm();
    for(const auto &slot:s->sampled_imu_slots())if(slot.active)me=std::max(me,(slot.noise->value()-mean.segment<6>(raw(slot.record.sequence))).norm());
    for(size_t i=0;i<poses.size();++i)me=std::max(me,(s->_exposure_poses[i].pose->value()-poses[i]).norm());
    for(int i=0;i<2;++i)me=std::max(me,std::abs(s->cam_imu_dt_var(i)->value()(0)-mean(15+i)));
    max_mean=std::max(max_mean,me); check(me<3e-9,"all retained means match independent nonlinear/dense update oracle");
  }
  void propagate(const Segment &seg,StateOptions::IntegrationMethod method,const NoiseManager &noise) {
    const int a=raw(seg.records[0].sequence), b=raw(seg.records[1].sequence); const V6 n0=mean.segment<6>(a),n1=mean.segment<6>(b);
    Mat f=reference_derivative(nav,calibration,seg,n0,n1,method), T=Mat::Identity(P.rows(),P.cols());
    T.topRows(15).setZero(); T.topLeftCorner(15,15)=f.leftCols(15);
    T.block(0,a,15,6)=f.middleCols(15,6); T.block(0,b,15,6)=f.rightCols(6);
    const Mat Q=bias_oracle(nav,calibration,seg,n0,n1,method,noise);
    P=(T*P*T.transpose()).eval(); P.topLeftCorner(15,15)+=Q;
    nav=reference_mean(nav,calibration,seg,n0,n1,method); mean.segment<12>(3)=nav.tail<12>();
  }
  void clone(const std::shared_ptr<State> &s,int camera,const V3 &omega) {
    const int n=P.rows(); Mat T=Mat::Zero(n+6,n); T.topRows(n).setIdentity(); T.bottomRows(6).leftCols(6).setIdentity();
    T.block<3,1>(n,15+camera)=omega; T.block<3,1>(n+3,15+camera)=nav.segment<3>(7);
    P=(T*P*T.transpose()).eval(); mean.conservativeResize(n+6); mean.tail(6).setZero(); mean.tail<3>()=nav.segment<3>(4);
    Eigen::Matrix<double,7,1> pose; pose<<nav.head<4>(),nav.segment<3>(4); poses.push_back(pose);
    State::ExposurePose exposure; exposure.camera_id=camera; exposure.imu_time=s->imu_endpoint(); exposure.raw_time=s->_timestamp;
    exposure.pose=StateHelper::augment_pose_view(s,camera,omega); s->_exposure_poses.push_back(exposure);
  }
  void observe(const std::shared_ptr<State> &s,int k) {
    // A delayed pose/velocity visual constraint, with full generic EKF update
    // reaching noise owners solely through previously established cross terms.
    std::vector<std::shared_ptr<ov_type::Type>> order{s->_exposure_poses.front().pose,s->_imu->v()};
    Mat H=Mat::Zero(3,9); H.block<3,3>(0,3).setIdentity(); H.rightCols<3>()=.23*M3::Identity();
    Mat full=Mat::Zero(3,P.rows()); full.block<3,3>(0,17+6*records.size()+3).setIdentity(); full.block<3,3>(0,6)=.23*M3::Identity();
    const Mat R=.0004*Mat::Identity(3,3), S=full*P*full.transpose()+R;
    const Mat K=P*full.transpose()*S.inverse(); V3 z(.49+.006*k,-.38+.003*k,.18-.002*k);
    const Vec residual=z-full*mean, delta=K*residual;
    nav=retract(nav,delta.head(15)); mean+=delta; mean.segment<12>(3)=nav.tail<12>();
    for(size_t i=0;i<poses.size();++i) {
      int id=17+6*records.size()+6*i; Eigen::Vector4d dq; dq<<.5*delta.segment<3>(id),1.; dq.normalize();
      poses[i].head<4>()=quat_multiply(dq,poses[i].head<4>()); poses[i].tail<3>()+=delta.segment<3>(id+3);
    }
    const Mat I=Mat::Identity(P.rows(),P.cols())-K*full; P=(I*P*I.transpose()+K*R*K.transpose()).eval();
    StateHelper::EKFUpdate(s,order,H,residual,R);
  }
};

void linked_dense() {
  for(auto method:{StateOptions::ANALYTICAL,StateOptions::DISCRETE,StateOptions::RK4}) {
    auto s=make_state(method); std::vector<Record> r;
    for(int i=0;i<4;++i)r.push_back(record(i+1,.08*i,s));
    Dense dense(s,r); Probe p(noises(),9.81); Propagator::EndpointKinematics endpoint;
    const auto slot0=s->sampled_imu_slots()[0].noise,slot1=s->sampled_imu_slots()[1].noise;
    const double times[]={.023,.057,.08,.109,.16,.24}; double previous=0.;
    for(int k=0;k<6;++k) {
      auto seg=segment(r,previous,times[k]); dense.propagate(seg,method,noises());
      check(p.propagate_sampled_segment(s,seg,times[k],endpoint),"actual two-camera cuts propagate with original records");
      dense.compare(s,"full live/old pose/sample cross covariance matches all-raw dense oracle");
      if(k<3) { dense.clone(s,k%2,endpoint.omega); dense.compare(s,"camera pose retains both raw-noise cross blocks"); }
      dense.observe(s,k); dense.compare(s,"intervening and delayed visual update conditions retained common noise");
      if(k==1)check(posterior(s,1).norm()>1e-6 && posterior(s,2).norm()>1e-6,"visual constraints infer nonzero means for both reused samples");
      if(k==1) {
        auto missing=StateHelper::clone_state(s),retained=StateHelper::clone_state(s);
        for(const auto &slot:missing->sampled_imu_slots())if(slot.active)slot.noise->set_value(V6::Zero());
        auto next=segment(r,times[k],times[k+1]);
        check(p.propagate_sampled_segment(missing,next,next.time1,endpoint) && p.propagate_sampled_segment(retained,next,next.time1,endpoint),
              "both actual posterior-mean control callers propagate");
        const double omitted=difference(missing->_imu->value(),retained->_imu->value()).norm();
        std::printf("OMITTED_MEAN method=%d actual_prediction_error=%.12g\n",method,omitted);
        check(omitted>1e-6,"negative actual caller omitting inferred sample means changes future prediction");
      }
      check(s->sampled_imu_slots()[0].noise==slot0 && s->sampled_imu_slots()[1].noise==slot1,
            "knot retirement/reuse keeps fixed Vec identities and 12 reserved coordinates");
      previous=times[k];
    }
    check(s->max_covariance_size()==17+12+18,"sample storage stays 12D through successive raw knots and three retained camera poses");
    check(!StateHelper::admit_sampled_imu_noise(s,r[0]),"retired original identity cannot be introduced as fresh independent noise");
  }
}

void rejection() {
  auto s=make_state(StateOptions::ANALYTICAL); std::vector<Record> r{record(1,0.,s),record(2,.1,s),record(3,.2,s)};
  auto seg=segment(r,0.,.03); Probe p(noises(),9.81); p.seed_cache(); Propagator::EndpointKinematics out; out.omega.setConstant(123.);
  Linearization linear; linear.Phi=Mat::Constant(1,1,42.); linear.independent_Q=Mat::Constant(1,1,43.);
  auto unchanged=[&](const Segment &bad) {
    auto before=StateHelper::clone_state(s); const auto snapshot=p.capture();
    const bool accepted=p.propagate_sampled_segment(s,bad,bad.time1,out,&linear);
    bool identical=same(s->_imu->value(),before->_imu->value()) && same(StateHelper::get_full_covariance(s),StateHelper::get_full_covariance(before)) &&
        s->_timestamp==before->_timestamp && s->imu_endpoint()==before->imu_endpoint() &&
        p.capture().have_last_prop_time_offset==snapshot.have_last_prop_time_offset && p.cache_valid() && p.cache_time()==913. &&
        out.omega(0)==123. && linear.Phi.rows()==1 && linear.Phi(0,0)==42.;
    for(int i=0;i<2;++i) identical=identical && s->sampled_imu_slots()[i].active==before->sampled_imu_slots()[i].active &&
        s->sampled_imu_slots()[i].record.sequence==before->sampled_imu_slots()[i].record.sequence &&
        same(s->sampled_imu_slots()[i].noise->value(),before->sampled_imu_slots()[i].noise->value());
    check(!accepted && identical,"declared refusal leaves covariance, mean, owners, endpoint, cache and outputs intact");
  };
  for(int which=0;which<13;++which) {
    auto bad=seg;
    if(which==0)bad.weights0(0)=.5;
    if(which==1)bad.time0=std::nextafter(0.,1.);
    if(which==2)bad.records[1].prior(0,0)=-1.;
    if(which==3)bad.records[1].stream_episode=74;
    if(which==4)bad.records[1].sequence=1;
    if(which==5)bad.records[1].measured(0)=std::numeric_limits<double>::infinity();
    if(which==6)bad.records[1].measured(0)=1e308;
    if(which==7)bad.records[0].prior(0,0)=std::numeric_limits<double>::quiet_NaN();
    if(which==8)bad.weights1(1)=std::numeric_limits<double>::quiet_NaN();
    if(which==9)s->_options.do_fej=true;
    if(which==10)s->_options.integration_method=static_cast<StateOptions::IntegrationMethod>(3);
    if(which==11)s->_options.do_calib_imu_intrinsics=true;
    if(which==12)s->_imu_endpoint_valid=false;
    unchanged(bad);
    s->_options.do_fej=false; s->_options.integration_method=StateOptions::ANALYTICAL; s->_options.do_calib_imu_intrinsics=false; s->_imu_endpoint_valid=true;
  }
  check(p.propagate_sampled_segment(s,seg,.03,out),"failed second-record admission did not advance watermark");
  p.seed_cache(); out.omega.setConstant(123.); linear.Phi=Mat::Constant(1,1,42.);
  auto repeated=segment(r,.03,.05); repeated.records[0].measured(0)+=1.; unchanged(repeated);
  check(p.propagate_sampled_segment(s,segment(r,.03,.05),.05,out),"unchanged original pair reuses posterior after rejected mutation");
  auto fresh=make_state(StateOptions::ANALYTICAL); Propagator::EndpointKinematics unused;
  check(!p.propagate_to_imu(fresh,.03,.03,unused),"legacy continuous propagation refuses explicit sampled ownership");
  for(const auto &record:r) { ImuData sample; sample.timestamp=record.timestamp; sample.wm=record.measured.head<3>();
    sample.am=record.measured.tail<3>(); p.feed_imu(sample); }
  unused.omega.setConstant(321.);
  check(!p.propagate_to_imu(fresh,fresh->imu_endpoint(),fresh->_timestamp,unused) && unused.omega(0)==321.,
        "zero-duration covered endpoint cannot bypass sampled refusal or overwrite rates");
  Propagator::BridgeData bridge; bridge.dt=123.; bridge.valid=true; bridge.alpha.setConstant(456.);
  check(!p.compute_bridge(fresh,0.,.1,bridge) && bridge.valid && bridge.dt==123. && bridge.alpha(0)==456.,
        "legacy bridge refuses sampled owners before clearing or replacing output");
  auto ordinary=make_state(StateOptions::ANALYTICAL,StateOptions::RPNG,false,false);
  check(p.propagate_to_imu(ordinary,ordinary->imu_endpoint(),ordinary->_timestamp,unused)&&unused.omega(0)!=321.,
        "same covered zero-duration endpoint succeeds for ordinary ownership");
  check(p.compute_bridge(ordinary,0.,.1,bridge) && bridge.valid && bridge.dt==.1,
        "covered ordinary-state bridge remains available as a positive control");
  Eigen::Matrix<double,13,1> state_out; Eigen::Matrix<double,12,12> cov_out;
  check(!p.fast_state_propagate(fresh,.03,state_out,cov_out),"legacy fast output refuses sampled covariance approximation");
  std::vector<Segment> selected{seg}; const auto original=selected.front(); const auto *storage=selected.data();
  check(!Propagator::select_sampled_imu_readings(r,0.,.2,selected) && selected.data()==storage && selected.size()==1 && selected[0].time1==original.time1,
        "insufficient reserved selection capacity leaves caller output untouched");
  selected.reserve(3); const auto *reserved=selected.data();
  check(Propagator::select_sampled_imu_readings(r,.025,.18,selected) && selected.size()==2 && selected.data()==reserved &&
        selected[0].records[1].sequence==selected[1].records[0].sequence && selected[0].time1==.1 && selected[1].time0==.1,
        "selector keeps immutable shared record and exact knot across reserved segments");
  auto badr=r; badr[2].sequence=1; const auto saved=selected;
  check(!Propagator::select_sampled_imu_readings(badr,0.,.2,selected) && selected.size()==saved.size() && selected[0].time0==saved[0].time0,
        "nonmonotonic original sequence refuses atomically");
  // Exercise failure after metadata preview, inside the covariance transaction.
  const auto old=StateHelper::get_full_covariance(fresh);
  Mat overflow=Mat::Constant(15,27,1e308), Q=Mat::Zero(15,15);
  check(!StateHelper::EKFPropagationSampled(fresh,{r[0],r[1]},overflow,Q) && same(old,StateHelper::get_full_covariance(fresh)) &&
        !fresh->sampled_imu_slots()[0].active && !fresh->sampled_imu_slots()[1].active,
        "overflowing full covariance product refuses without admitting staged records");
}

void rk4_controls() {
  for(auto model:{StateOptions::RPNG,StateOptions::KALIBR}) for(double scalar:{1.,1e-10,-1e-10}) {
    auto s=make_state(StateOptions::RK4,model); Calibration c(s);
    X x=s->_imu->value(); x.head<3>()=std::sqrt(1.-scalar*scalar)*V3(.3,-.7,.2).normalized(); x(3)=scalar;
    s->_imu->set_value(x); s->_imu->set_fej(x); s->_imu_endpoint=s->_timestamp=.006;
    const V6 n0=(V6()<<.01,-.02,.015,.03,-.012,.005).finished(),n1=-.7*n0;
    std::vector<Record> r{record(1,0.,s),record(2,.05,s)};
    const V3 a0(1.2,-.4,9.6),a1(-2.4,3.2,10.1),w0(10.,-6.,5.),w1(-11.,8.,-6.);
    for(int k=0;k<2;++k) {
      const V3 a=k?a1:a0,w=k?w1:w0;
      r[k].measured.tail<3>()=c.A.inverse()*a+x.segment<3>(13);
      r[k].measured.head<3>()=c.W.inverse()*w+x.segment<3>(10)+c.T*a;
      r[k].measured+=k?n1:n0;
      StateHelper::admit_sampled_imu_noise(s,r[k])->set_value(k?n1:n0);
    }
    const auto seg=segment(r,.006,.034); const auto before=StateHelper::clone_state(s);
    const Mat expected=reference_derivative(x,c,seg,n0,n1,StateOptions::RK4);
    const X expected_mean=reference_rk4(x,c,seg,n0,n1);
    Probe p(noises(),9.81); Propagator::EndpointKinematics endpoint; Linearization linear;
    check(p.propagate_sampled_segment(s,seg,seg.time1,endpoint,&linear),"RK4 accepts varying calibrated original-pair signals and quaternion hemisphere crossings");
    const double error=(linear.Phi-expected).cwiseAbs().maxCoeff(); max_fd=std::max(max_fd,error);
    check(error<2e-8 && difference(s->_imu->value(),expected_mean).norm()<2e-12,
          "RK4 full tangent and nonlinear mean match independent global-quaternion stages near normalization sign boundary");
    // Compare actual linked legacy RK4 mean and its old frozen-average tangent.
    // The legacy mean agrees, but that tangent is not the derivative of RK4.
    ImuData lo,hi;lo.timestamp=seg.time0;hi.timestamp=seg.time1;
    const V6 raw0=seg.weights0(0)*(r[0].measured-n0)+seg.weights0(1)*(r[1].measured-n1);
    const V6 raw1=seg.weights1(0)*(r[0].measured-n0)+seg.weights1(1)*(r[1].measured-n1);
    lo.wm=raw0.head<3>();lo.am=raw0.tail<3>();hi.wm=raw1.head<3>();hi.am=raw1.tail<3>();
    auto legacy=StateHelper::clone_state(before);Mat oldF,oldQ;p.predict_and_compute(legacy,lo,hi,oldF,oldQ);
    check(difference(legacy->_imu->value(),s->_imu->value()).norm()<2e-12,"new sampled mean matches actual legacy RK4 integration");
    Mat wrong(15,27);wrong.leftCols(15)=oldF;const Eigen::Vector2d average=.5*(seg.weights0+seg.weights1);
    wrong.middleCols(15,6)=average(0)*oldF.middleCols(9,6);wrong.rightCols(6)=average(1)*oldF.middleCols(9,6);
    wrong.bottomRightCorner(6,12).setZero(); // Raw measurement errors do not drive the bias means.
    const double missing=(wrong-expected).cwiseAbs().maxCoeff();
    std::printf("RK4_FROZEN_TANGENT_NEGATIVE model=%d scalar=%.3g error=%.12g\n",model,scalar,missing);
    check(missing>1e-4,"negative frozen-average F and noise weights fail the actual RK4 derivative");
    // A second oracle differentiates the real public caller at every input.
    for(int j=0;j<27;++j) {
      auto plus=StateHelper::clone_state(before),minus=StateHelper::clone_state(before);const double eps=2e-6;
      if(j<15){Vec d=Vec::Zero(15);d(j)=eps;plus->_imu->update(d);minus->_imu->update(-d);}
      else for(auto state:{plus,minus})for(const auto&slot:state->sampled_imu_slots())if(slot.record.sequence==r[j<21?0:1].sequence){
        auto mean=slot.noise->value();mean((j-15)%6)+=state==plus?eps:-eps;slot.noise->set_value(mean);
      }
      check(p.propagate_sampled_segment(plus,seg,seg.time1,endpoint)&&p.propagate_sampled_segment(minus,seg,seg.time1,endpoint),
            "perturbed RK4 caller accepts each current navigation and original-noise coordinate");
      const double fd=(linear.Phi.col(j)-difference(plus->_imu->value(),minus->_imu->value())/(2*eps)).cwiseAbs().maxCoeff();
      max_fd=std::max(max_fd,fd);check(fd<2e-8,"RK4 exact chain-rule column matches finite differences of the linked caller");
    }
  }
  // Endpoint cap, nonunit input and nonfinite inputs must reject atomically.
  // Opposing rates have zero average, so the inherited average guard alone
  // would not protect RK4 stage normalization or the declared integration cap.
  for(int which=0;which<6;++which) {
    auto s=make_state(StateOptions::RK4,StateOptions::RPNG,true);const X x=s->_imu->value();
    std::vector<Record> r{record(1,0.,s),record(2,.02,s)};
    for(int k=0;k<2;++k){r[k].measured.head<3>()=x.segment<3>(10);r[k].measured.tail<3>()=x.segment<3>(13);}
    const double peak=which==0?.49:.500001;
    r[0].measured(0)+=peak/.02;r[1].measured(0)-=peak/.02;
    if(which>=2){r[0].measured.head<3>()=x.segment<3>(10);r[1].measured.head<3>()=x.segment<3>(10);}
    if(which==2){auto v=x;v.head<4>()*=std::sqrt(1.+5e-10);s->_imu->set_value(v);}
    if(which==3){auto v=x;v.head<4>()*=std::sqrt(1.+2e-9);s->_imu->set_value(v);}
    if(which==5)s->_options.do_fej=true;
    const auto before=StateHelper::clone_state(s);auto seg=segment(r,0.,.02);
    if(which==4)seg.records[0].measured(3)=std::numeric_limits<double>::quiet_NaN();
    Probe p(noises(false),9.81);p.seed_cache();const auto snapshot=p.capture();
    Propagator::EndpointKinematics endpoint;endpoint.omega.setConstant(123.);endpoint.omega_fej.setConstant(124.);
    Linearization linear;linear.Phi=Mat::Constant(1,1,321.);linear.independent_Q=Mat::Constant(1,1,322.);
    const bool accepted=p.propagate_sampled_segment(s,seg,seg.time1,endpoint,&linear);
    if(which==0||which==2){
      check(accepted,"RK4 accepts bounded opposing endpoints and a near-unit initial quaternion inside the existing guard");
      const Calibration c(before);
      check(difference(s->_imu->value(),reference_rk4(before->_imu->value(),c,seg,V6::Zero(),V6::Zero())).norm()<2e-12,
            "accepted normalization-boundary control matches independent full-quaternion RK4");
    }else{
      check(!accepted&&same(s->_imu->value(),before->_imu->value())&&same(s->_imu->fej(),before->_imu->fej())&&
            same(StateHelper::get_full_covariance(s),StateHelper::get_full_covariance(before))&&s->imu_endpoint()==before->imu_endpoint()&&
            s->_timestamp==before->_timestamp&&p.cache_valid()&&p.cache_time()==913.&&
            p.capture().have_last_prop_time_offset==snapshot.have_last_prop_time_offset&&
            endpoint.omega(0)==123.&&endpoint.omega_fej(0)==124.&&linear.Phi.rows()==1&&linear.Phi(0,0)==321.&&linear.independent_Q(0,0)==322.&&
            !s->sampled_imu_slots()[0].active&&!s->sampled_imu_slots()[1].active,
            "RK4 rejection preserves full state, FEJ, owner registry, accepted clocks, cache and caller outputs");
    }
  }
}

void split_controls() {
  for(auto method:{StateOptions::ANALYTICAL,StateOptions::DISCRETE,StateOptions::RK4}) {
    auto full=make_state(method,StateOptions::RPNG,true),split=StateHelper::clone_state(full);
    std::vector<Record> r{record(1,0.,full,0.),record(2,.12,full,0.)}; r[1].measured=r[0].measured;
    // Only x acceleration sample error; zero nominal acceleration/gravity leaves
    // an exactly linear scalar subsystem independent of orientation errors.
    for(auto &v:r) { v.measured.head<3>()=full->_imu->bias_g(); v.measured.tail<3>()=full->_imu->bias_a(); v.prior.setZero(); v.prior(3,3)=.4; }
    auto seed=StateHelper::get_full_covariance(full); seed.setZero();
    std::vector<std::shared_ptr<ov_type::Type>> order{full->_imu,full->cam_imu_dt_var(0),full->cam_imu_dt_var(1),full->sampled_imu_slots()[0].noise,full->sampled_imu_slots()[1].noise};
    std::sort(order.begin(),order.end(),[](const auto&a,const auto&b){return a->id()<b->id();});
    StateHelper::set_initial_covariance(full,seed,order); split=StateHelper::clone_state(full);
    const auto start=StateHelper::clone_state(full);
    Probe p(noises(false),9.81); Propagator::EndpointKinematics endpoint;
    auto whole=segment(r,0.,.12); check(p.propagate_sampled_segment(full,whole,.12,endpoint),"whole interval raw-error control propagates");
    double t=0.; Mat independent=Mat::Zero(2,2); Eigen::Matrix<double,2,2> J=Eigen::Matrix2d::Identity();
    for(double end:{.027,.063,.12}) {
      auto part=segment(r,t,end); check(p.propagate_sampled_segment(split,part,end,endpoint),"same original raw pair supports multiple interior cuts");
      const double dt=end-t; J<<1,dt,0,1; const Eigen::Vector2d b(.5*dt*dt,dt),w=.5*(part.weights0+part.weights1);
      independent=(J*independent*J.transpose()+.4*w.squaredNorm()*b*b.transpose()).eval(); t=end;
    }
    const auto P=StateHelper::get_full_covariance(split); const double T=.12;
    // Piecewise endpoint-average integration has exact velocity integral for
    // linearly interpolated raw errors. Its position coefficient is midpoint
    // quadrature, so cuts improve it; they do not define new independent draws.
    check(std::abs(P(6,6)-.2*T*T)<1e-13,"sample velocity variance is R*dt^2 and invariant under camera cuts");
    check(std::abs(P(6,6)-StateHelper::get_full_covariance(full)(6,6))<1e-13,"split/unsplit velocity covariance preserves common sample identity");
    check(std::abs(P(6,6)-independent(1,1))>1e-3,"negative fresh-independent-noise-per-cut predecessor loses shared covariance");
    const double exact_position=.4*std::pow(T,4)*(1./9.+1./36.);
    const double fullerror=std::abs(StateHelper::get_full_covariance(full)(3,3)-exact_position);
    const double spliterror=std::abs(P(3,3)-exact_position);
    if(method==StateOptions::RK4)
      check(fullerror<1e-14 && spliterror<1e-14,"RK4 integrates scalar linearly-interpolated sample position coefficients exactly across cuts");
    else check(spliterror<fullerror && spliterror>1e-10,"position converges with mean discretization; no false exact split invariance claim");
    double previous_error=0.;
    for(int pieces:{4,8,16}) {
      auto refined=StateHelper::clone_state(start); double begin=0.;
      for(int piece=1;piece<=pieces;++piece) {
        const double end=piece*T/pieces; auto part=segment(r,begin,end);
        check(p.propagate_sampled_segment(refined,part,end,endpoint),"refined original-pair prediction accepts canonical cut"); begin=end;
      }
      const auto refined_P=StateHelper::get_full_covariance(refined);
      const double error=std::abs(refined_P(3,3)-exact_position);
      check(std::abs(refined_P(6,6)-.2*T*T)<1e-13,"raw velocity scaling remains exact at every refinement level");
      if(method==StateOptions::RK4)check(error<1e-14,"RK4 scalar position covariance stays exact at every split refinement");
      else if(previous_error)check(previous_error/error>3.8 && previous_error/error<4.1,
                              "position covariance converges at the declared second-order midpoint rate");
      previous_error=error;
    }
    for(double interval:{.005,.02}) {
      auto scaled=StateHelper::clone_state(start); auto readings=r; readings[1].timestamp=interval;
      auto part=segment(readings,0.,interval);
      check(p.propagate_sampled_segment(scaled,part,interval,endpoint) &&
            std::abs(StateHelper::get_full_covariance(scaled)(6,6)-.2*interval*interval)<1e-14,
            "explicit raw-sample prior yields quadratic sample-period variance, not density divided by fragment dt");
    }
    const auto n0=V6::Constant(.02),n1=V6::Constant(-.01); Calibration c(split);
    check(difference(reference_mean(split->_imu->value(),c,whole,n0,n1,method),reference_mean(split->_imu->value(),c,whole,V6::Zero(),V6::Zero(),method)).norm()>1e-4,
          "negative omitted posterior mean has a material subsequent prediction error");
  }
}

#if defined(__GLIBC__) && (defined(__x86_64__) || defined(__aarch64__))
struct StackMeasurement {
  unsigned char *storage;
  size_t bytes;
  std::shared_ptr<State> state;
  Propagator *propagator;
  std::array<Segment,3> segments;
  size_t touched[4]{};
  bool accepted[3]{};
  unsigned control=0;
};
__attribute__((noinline)) unsigned stack_positive_control() {
  volatile unsigned char probe[32768];
  for(size_t i=0;i<sizeof probe;++i)probe[i]=static_cast<unsigned char>(i);
  return probe[0]+probe[32767];
}
void *measure_stack(void *opaque) {
  auto &m=*static_cast<StackMeasurement*>(opaque);
  Propagator::EndpointKinematics endpoint;
  for(int step=0;step<4;++step) {
    uintptr_t pointer;
#if defined(__x86_64__)
    asm volatile("mov %%rsp, %0" : "=r"(pointer));
#else
    asm volatile("mov %0, sp" : "=r"(pointer));
#endif
    // Paint only unused stack below the live entry frame; leave the ABI red
    // zone and memset's tiny frame untouched. The stack is privately supplied
    // to this test thread, and the parent only joins while it is in use.
    auto *begin=m.storage+4096,*end=reinterpret_cast<unsigned char*>(pointer-512);
    if(end<=begin || end>=m.storage+m.bytes)std::abort();
    std::memset(begin,0xa5,static_cast<size_t>(end-begin));
    if(step==0)m.control=stack_positive_control();
    else m.accepted[step-1]=m.propagator->propagate_sampled_segment(m.state,m.segments[step-1],m.segments[step-1].time1,endpoint);
    auto *first=begin;while(first<end&&*first==0xa5)++first;
    m.touched[step]=first==end?0:pointer-reinterpret_cast<uintptr_t>(first);
  }
  return nullptr;
}
#endif
void stack_resources() {
#if defined(__GLIBC__) && (defined(__x86_64__) || defined(__aarch64__))
  for(int clones:{0,20,100}) {
    auto s=make_state(StateOptions::RK4);
    for(int k=0;k<clones;++k){State::ExposurePose pose;pose.pose=StateHelper::augment_pose_view(s,k%2,V3::Zero());s->_exposure_poses.push_back(pose);}
    Probe p(noises(),9.81);std::vector<Record> r{record(1,0.,s),record(2,.05,s),record(3,.1,s)};
    StackMeasurement measurement{};measurement.bytes=1024*1024;measurement.state=s;measurement.propagator=&p;
    measurement.segments={segment(r,0.,.03),segment(r,.03,.05),segment(r,.05,.075)};
    void *storage=nullptr;
    if(posix_memalign(&storage,4096,measurement.bytes)!=0)std::abort();measurement.storage=static_cast<unsigned char*>(storage);
    pthread_attr_t attributes;pthread_attr_init(&attributes);
    if(pthread_attr_setstack(&attributes,storage,measurement.bytes)!=0)std::abort();
    pthread_t thread;if(pthread_create(&thread,&attributes,measure_stack,&measurement)!=0)std::abort();
    pthread_attr_destroy(&attributes);pthread_join(thread,nullptr);
    check(measurement.control==255&&measurement.touched[0]>=32768,"stack watermark positive control detects a real32KiB callee frame");
    check(measurement.accepted[0]&&measurement.accepted[1]&&measurement.accepted[2],"stack probe runs actual RK4 admission, retirement and next-pair reuse");
    std::printf("RK4_STACK n=%d admission_touched=%zu retirement_touched=%zu next_pair_touched=%zu control_touched=%zu\n",
      s->max_covariance_size(),measurement.touched[1],measurement.touched[2],measurement.touched[3],measurement.touched[0]);
    // This is touched-stack evidence including out-of-line Eigen calls, not a
    // portable stack guarantee. Compiler .su files separately bound frames.
    check(*std::max_element(measurement.touched+1,measurement.touched+4)<measurement.bytes/2,
          "actual RK4 stack use stays within the provided bounded test stack");
    std::free(storage);
  }
#endif
}

void resources() {
#ifdef __GLIBC__
  for(auto method:{StateOptions::ANALYTICAL,StateOptions::RK4}) for(int clones:{0,20,100}) {
    auto s=make_state(method);
    for(int k=0;k<clones;++k) {
      State::ExposurePose view; view.pose=StateHelper::augment_pose_view(s,k%2,V3::Zero()); s->_exposure_poses.push_back(view);
    }
    const int n=s->max_covariance_size(); const auto a=s->sampled_imu_slots()[0].noise,b=s->sampled_imu_slots()[1].noise;
    std::vector<Record> r{record(1,0.,s),record(2,1.,s)}; std::vector<Segment> selected; selected.reserve(1);
    heap_calls=heap_bytes=heap_largest=0; count_heap=true;
    const Mat positive=StateHelper::get_full_covariance(s);
    count_heap=false; check(heap_calls>0 && heap_largest>=size_t(8*n*n),"allocation interception positive control sees actual dense copy");
    Probe p(noises(),9.81); Propagator::EndpointKinematics endpoint;
    size_t max_calls=0,max_bytes=0,max_single=0,selection_calls=0;
    double begin=0.;
    for(int k=1;k<=200;++k) {
      const double end=k/200.;
      heap_calls=heap_bytes=heap_largest=0; count_heap=true;
      const bool selected_ok=Propagator::select_sampled_imu_readings(r,begin,end,selected);
      count_heap=false; selection_calls+=heap_calls;
      heap_calls=heap_bytes=heap_largest=0; count_heap=true;
      const bool propagated=selected_ok && p.propagate_sampled_segment(s,selected.front(),end,endpoint);
      count_heap=false;
      max_calls=std::max(max_calls,heap_calls); max_bytes=std::max(max_bytes,heap_bytes); max_single=std::max(max_single,heap_largest);
      check(propagated,"bounded-storage caller propagates each fragment"); begin=end;
    }
    std::printf("RESOURCES method=%d n=%d calls=200 max_heap_calls=%zu max_requested_bytes=%zu largest_request=%zu selector_heap_calls=%zu\n",
                method,n,max_calls,max_bytes,max_single,selection_calls);
    check(selection_calls==0,"pre-reserved original-record selector makes no heap allocations");
    // These are storage bounds, not host latency thresholds. Permit fixed
    // 15D validation/packing work plus a linear n-by-15 covariance workspace.
    check(max_calls<=100 && max_bytes<=size_t(1024*n+100000) && max_single<=size_t(8*std::max(225,15*n)+64),
          "per-segment allocation count and requested storage have fixed/linear bounds, with no n-by-n scratch");
    check(s->max_covariance_size()==n && s->sampled_imu_slots()[0].noise==a && s->sampled_imu_slots()[1].noise==b,
          "fragment count does not grow covariance or create owner Types");
  }
#endif
}
} // namespace
int main() {
  Printer::setPrintLevel(Printer::WARNING);
  derivatives(); linked_dense(); rejection(); rk4_controls(); split_controls(); resources(); stack_resources();
  std::printf("SAMPLED PROPAGATION: %d/%d checks passed, max_FD=%.12g max_dense=%.12g max_mean=%.12g bias_Q_relative=%.12g\n",
              checks-failures,checks,max_fd,max_dense,max_mean,max_q);
  return failures?1:0;
}
