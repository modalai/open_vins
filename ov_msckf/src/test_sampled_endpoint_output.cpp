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
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>

#ifdef __GLIBC__
static thread_local bool count_heap=false;
static thread_local size_t heap_calls=0,heap_bytes=0,heap_largest=0;
static void allocation(size_t n){if(count_heap){++heap_calls;heap_bytes+=n;heap_largest=std::max(heap_largest,n);}}
extern "C" void *__libc_malloc(size_t);
extern "C" void *__libc_calloc(size_t,size_t);
extern "C" void *__libc_realloc(void*,size_t);
extern "C" void *__libc_memalign(size_t,size_t);
extern "C" void *malloc(size_t n) noexcept{allocation(n);return __libc_malloc(n);}
extern "C" void *calloc(size_t n,size_t s) noexcept{allocation(n*s);return __libc_calloc(n,s);}
extern "C" void *realloc(void*p,size_t n) noexcept{allocation(n);return __libc_realloc(p,n);}
extern "C" void *aligned_alloc(size_t a,size_t n) noexcept{allocation(n);return __libc_memalign(a,n);}
extern "C" int posix_memalign(void **p,size_t a,size_t n) noexcept{
  if(a<sizeof(void*)||(a&(a-1)))return EINVAL;allocation(n);void*v=__libc_memalign(a,n);if(!v)return ENOMEM;*p=v;return 0;
}
#endif

using namespace ov_msckf;
using namespace ov_core;
namespace {
using Mat=Eigen::MatrixXd;using Vec=Eigen::VectorXd;using V3=Eigen::Vector3d;using M3=Eigen::Matrix3d;
using V6=Eigen::Matrix<double,6,1>;using M6=Eigen::Matrix<double,6,6>;using X=Eigen::Matrix<double,16,1>;
using Y=Eigen::Matrix<double,13,1>;using C=Eigen::Matrix<double,12,12>;
using Output=Propagator::SampledStateOutput;using Record=State::SampledImuRecord;
int checks=0,failures=0;double max_fd_cov=0.,max_cross=0.,max_mean=0.;
void check(bool ok,const char*why){++checks;if(!ok){++failures;std::printf("FAIL: %s\n",why);}}
template<class A,class B>bool same(const Eigen::MatrixBase<A>&a,const Eigen::MatrixBase<B>&b){
  Mat aa=a,bb=b;return aa.rows()==bb.rows()&&aa.cols()==bb.cols()&&std::memcmp(aa.data(),bb.data(),aa.size()*sizeof(double))==0;
}
bool same(const Output&a,const Output&b){
  bool equal=same(a.mean,b.mean)&&same(a.covariance,b.covariance)&&same(a.weights,b.weights)&&
    initializer_time_bits(a.imu_time)==initializer_time_bits(b.imu_time)&&a.stream_episode==b.stream_episode&&a.support_count==b.support_count;
  for(int i=0;i<2;++i)equal&=a.support[i].sequence==b.support[i].sequence&&initializer_time_bits(a.support[i].timestamp)==initializer_time_bits(b.support[i].timestamp);
  return equal;
}
struct Probe:Propagator{using Propagator::Propagator;void seed_cache(){cache_imu_valid=true;cache_state_time=913.;}
  bool cached()const{return cache_imu_valid.load();}};
NoiseManager noise(){NoiseManager n;n.sigma_w=.004;n.sigma_a=.037;n.sigma_wb=.012;n.sigma_ab=.035;return n;}
std::vector<std::shared_ptr<ov_type::Type>> variables(const std::shared_ptr<State>&s){
  std::vector<std::shared_ptr<ov_type::Type>> out{s->_imu,s->cam_imu_dt_var(0),s->cam_imu_dt_var(1)};
  for(const auto&v:s->sampled_imu_slots())if(v.noise)out.push_back(v.noise);
  for(const auto&v:s->_exposure_poses)out.push_back(v.pose);
  std::sort(out.begin(),out.end(),[](const auto&a,const auto&b){return a->id()<b->id();});return out;
}
std::shared_ptr<ov_type::Vec> owner(const std::shared_ptr<State>&s,uint64_t seq){
  for(const auto&v:s->sampled_imu_slots())if(v.active&&v.record.sequence==seq)return v.noise;return nullptr;
}
struct Calibration{M3 A,W,T;explicit Calibration(const std::shared_ptr<State>&s){
  A=s->_calib_imu_ACCtoIMU->Rot()*State::Dm(s->_options.imu_model,s->_calib_imu_da->value());
  W=s->_calib_imu_GYROtoIMU->Rot()*State::Dm(s->_options.imu_model,s->_calib_imu_dw->value());T=State::Tg(s->_calib_imu_tg->value());}};
std::shared_ptr<State> state(StateOptions::ImuModel model=StateOptions::RPNG){
  StateOptions o;o.do_fej=false;o.imu_model=model;o.integration_method=StateOptions::ANALYTICAL;
  o.num_cameras=2;o.do_calib_camera_timeoffset=true;o.physical_camera_clones=true;o.max_clone_size=6;o.configure_clone_policy(false,false);
  auto s=std::make_shared<State>(o);s->_timestamp=0.;s->_imu_endpoint=0.;s->_imu_endpoint_valid=true;
  V6 dw,da;if(model==StateOptions::RPNG){dw<<1.07,.018,.94,-.012,.021,1.025;da<<.97,-.024,1.055,.016,-.019,1.02;}
  else{dw<<1.07,.018,-.012,.94,.021,1.025;da<<.97,-.024,.016,1.055,-.019,1.02;}
  s->_calib_imu_dw->set_value(dw);s->_calib_imu_da->set_value(da);M3 t;t<<.028,-.014,.013,.012,.027,-.015,-.016,.011,.024;
  Eigen::Matrix<double,9,1> tv;tv<<t.col(0),t.col(1),t.col(2);s->_calib_imu_tg->set_value(tv);
  auto q=model==StateOptions::RPNG?s->_calib_imu_ACCtoIMU:s->_calib_imu_GYROtoIMU;q->set_value(rot_2_quat(exp_so3(V3(.16,-.11,.07))));
  X x=s->_imu->value();x.head<4>()=rot_2_quat(exp_so3(V3(.31,-.22,.18)));x.segment<3>(4)<<.3,-.4,.15;
  x.segment<3>(7)<<.065,.018,.009;x.segment<3>(10)<<.013,-.021,.009;x.segment<3>(13)<<.035,-.026,.019;
  s->_imu->set_value(x);s->_imu->set_fej(x);Mat P=Mat::Identity(17,17)*.003;
  Vec v=Vec::LinSpaced(17,-.02,.03);P+=v*v.transpose();StateHelper::set_initial_covariance(s,P,variables(s));
  check(StateHelper::prepare_sampled_imu_boundary(s,73),"fixture prepares two permanent raw coordinates");return s;
}
std::vector<Record> records(const std::shared_ptr<State>&s){Calibration c(s);const V3 force=s->_imu->Rot()*V3(0,0,9.81);
  std::vector<Record> out;for(int k=0;k<3;++k){Record r;r.stream_episode=73;r.sequence=k+1;r.timestamp=.1*k;
    r.measured.head<3>()=s->_imu->bias_g()+c.T*force+V3(.004,-.003,.002)*(k+1);
    r.measured.tail<3>()=s->_imu->bias_a()+c.A.inverse()*force+V3(.018,-.012,.009)*(k+1);
    M6 L=M6::Identity()*.15;for(int i=0;i<6;++i)for(int j=0;j<i;++j)L(i,j)=.008*(i+j+1);r.prior=L*L.transpose();out.push_back(r);
  }return out;
}
Y independent_mean(const std::shared_ptr<State>&s,const std::vector<Record>&r,double time){
  auto right=std::lower_bound(r.begin(),r.end(),time,[](const Record&a,double t){return a.timestamp<t;});V6 raw;
  if(initializer_time_bits(right->timestamp)==initializer_time_bits(time))raw=right->measured-owner(s,right->sequence)->value();
  else{const auto&left=*(right-1);const double a=(time-left.timestamp)/(right->timestamp-left.timestamp);
    raw=(1-a)*(left.measured-owner(s,left.sequence)->value())+a*(right->measured-owner(s,right->sequence)->value());}
  Calibration c(s);Y y;y.head<4>()=s->_imu->quat();y.segment<3>(4)=s->_imu->pos();y.segment<3>(7)=s->_imu->Rot()*s->_imu->vel();
  y.tail<3>()=c.W*(raw.head<3>()-s->_imu->bias_g()-c.T*c.A*(raw.tail<3>()-s->_imu->bias_a()));return y;
}
template<class Function>Mat derivative(const std::shared_ptr<State>&s,int rows,Function f,double eps){
  Mat H=Mat::Zero(rows,s->max_covariance_size());
  for(int group=0;group<3;++group){auto var=group==0?std::static_pointer_cast<ov_type::Type>(s->_imu):
      std::static_pointer_cast<ov_type::Type>(s->sampled_imu_slots()[group-1].noise);
    for(int col=0;col<var->size();++col){auto a=StateHelper::clone_state(s),b=StateHelper::clone_state(s);Vec d=Vec::Zero(var->size());d(col)=eps;
      if(group==0){a->_imu->update(d);b->_imu->update(-d);}else{a->sampled_imu_slots()[group-1].noise->update(d);b->sampled_imu_slots()[group-1].noise->update(-d);}
      H.col(var->id()+col)=(f(a)-f(b))/(2*eps);
    }
  }return H;
}
Mat output_derivative(const std::shared_ptr<State>&s,const std::vector<Record>&r){const M3 rotation=s->_imu->Rot();
  auto f=[&](const std::shared_ptr<State>&x){Y y=independent_mean(x,r,s->imu_endpoint());Eigen::Matrix<double,12,1> z;
    Eigen::AngleAxisd relative(x->_imu->Rot()*rotation.transpose());z.head<3>()=-relative.angle()*relative.axis();z.tail<9>()=y.tail<9>();return z;};
  const Mat coarse=derivative(s,12,f,4e-4),middle=derivative(s,12,f,2e-4),fine=derivative(s,12,f,1e-4);
  const double ratio=(coarse-middle).norm()/(middle-fine).norm();check(ratio>3.85&&ratio<4.15,"output nonlinear finite differences converge quadratically");
  return (4*fine-middle)/3.;
}
Output compare(const std::shared_ptr<State>&s,const std::vector<Record>&r,const std::shared_ptr<Probe>&p,
               const Mat*oracle_p=nullptr,const std::shared_ptr<State>&oracle_state=nullptr){
  const auto expected=oracle_state?oracle_state:s;const Mat H=output_derivative(expected,r);
  const Mat P=oracle_p?*oracle_p:StateHelper::get_full_covariance(s);Mat cross(s->max_covariance_size(),12);double*storage=cross.data();Output out;
  const auto before=StateHelper::get_full_covariance(s);const auto time=initializer_time_bits(s->imu_endpoint());
  check(p->sampled_state_at_endpoint(s,s->imu_endpoint(),out,&cross),"actual current-output caller accepts retained exact-knot/interior support");
  const double ce=(out.covariance-H*P*H.transpose()).cwiseAbs().maxCoeff();const double xe=(cross-P*H.transpose()).cwiseAbs().maxCoeff();
  const double me=(out.mean-independent_mean(expected,r,expected->imu_endpoint())).norm();max_fd_cov=std::max(max_fd_cov,ce);max_cross=std::max(max_cross,xe);max_mean=std::max(max_mean,me);
  check(ce<3e-10&&xe<3e-10&&me<3e-10,"mean, complete output covariance and every retained-state cross equal independent dense oracle");
  check(cross.data()==storage&&same(before,StateHelper::get_full_covariance(s))&&initializer_time_bits(s->imu_endpoint())==time,
        "projection leaves State and caller cross allocation unchanged");
  Output marginal;check(p->sampled_state_at_endpoint(s,s->imu_endpoint(),marginal)&&same(out,marginal),"optional cross does not change projected mean/covariance/receipt");
  const auto method=s->_options.integration_method;for(auto m:{StateOptions::ANALYTICAL,StateOptions::DISCRETE,StateOptions::RK4}){
    s->_options.integration_method=m;Output other;check(p->sampled_state_at_endpoint(s,s->imu_endpoint(),other)&&same(out,other),
      "identical posterior has identical algebraic output independent of integration option");}s->_options.integration_method=method;
  auto alternative=noise();alternative.sigma_w=1e3;alternative.sigma_a=2e3;alternative.sigma_wb=300.;alternative.sigma_ab=400.;Probe densities(alternative,1e3);Output other;
  check(densities.sampled_state_at_endpoint(s,s->imu_endpoint(),other)&&same(out,other),"output adds no density-based sensor R or bias Q");return out;
}
void propagate(const std::shared_ptr<State>&s,const std::vector<Record>&r,const std::shared_ptr<Probe>&p,double target){
  std::vector<Propagator::SampledImuSegment> segments;segments.reserve(3);
  check(Propagator::select_sampled_imu_readings(r,s->imu_endpoint(),target,segments),"actual selection keeps immutable original support");
  for(const auto&seg:segments){Propagator::EndpointKinematics endpoint;check(p->propagate_sampled_segment(s,seg,target,endpoint),"actual sampled propagation accepts output endpoint");}
}
struct DenseUpdate{Mat P;std::shared_ptr<State> state;};
DenseUpdate condition(const std::shared_ptr<State>&s,const Mat&H,const Vec&res,const Mat&R){
  Mat P=StateHelper::get_full_covariance(s);const Mat K=P*H.transpose()*(H*P*H.transpose()+R).inverse(),T=Mat::Identity(P.rows(),P.rows())-K*H;
  auto expected=StateHelper::clone_state(s);const Vec dx=K*res;for(const auto&v:variables(expected))v->update(dx.segment(v->id(),v->size()));
  return {T*P*T.transpose()+K*R*K.transpose(),expected};
}
void visual(const std::shared_ptr<State>&s,const std::vector<Record>&r,const std::shared_ptr<Probe>&p){
  Mat H=Mat::Zero(1,s->max_covariance_size()),local=Mat::Zero(1,9);H(0,s->_exposure_poses.front().pose->id()+3)=1.;H(0,6)=.3;
  local(0,3)=1.;local(0,6)=.3;const Vec residual=Vec::Constant(1,.009);const Mat R=Mat::Constant(1,1,.0001);
  const auto expected=condition(s,H,residual,R);const auto before=compare(s,r,p);p->seed_cache();
  check(StateHelper::EKFUpdate(s,{s->_exposure_poses.front().pose,s->_imu->v()},local,residual,R),"actual visual update conditions current sample correlations");
  const auto after=compare(s,r,p,&expected.P,expected.state);
  check(p->cached()&&(after.mean-before.mean).norm()>1e-5,"same-time evaluator ignores deliberately stale legacy cache after visual conditioning");
}
void controls(const std::shared_ptr<State>&s,const std::vector<Record>&r,const std::shared_ptr<Probe>&p){
  Output good;check(p->sampled_state_at_endpoint(s,s->imu_endpoint(),good),"counterfactual baseline current output exists");
  auto omitted=StateHelper::clone_state(s);for(const auto&slot:omitted->sampled_imu_slots())if(slot.active)slot.noise->set_value(V6::Zero());Output wrong;
  check(p->sampled_state_at_endpoint(omitted,omitted->imu_endpoint(),wrong)&&(wrong.mean-good.mean).norm()>1e-7,
        "negative actual caller discarding inferred original noise means changes output");
  auto decorrelated=StateHelper::clone_state(s);Mat P=StateHelper::get_full_covariance(s);std::vector<int> ids;
  for(const auto&slot:s->sampled_imu_slots())for(int j=0;j<6;++j)ids.push_back(slot.noise->id()+j);
  for(int i:ids)for(int j=0;j<P.rows();++j)if(std::find(ids.begin(),ids.end(),j)==ids.end())P(i,j)=P(j,i)=0.;
  StateHelper::set_initial_covariance(decorrelated,P,variables(decorrelated));
  check(p->sampled_state_at_endpoint(decorrelated,decorrelated->imu_endpoint(),wrong)&&(wrong.covariance-good.covariance).norm()>1e-7,
        "negative actual caller dropping state/raw cross covariance changes output");
  for(int i:ids)for(int j:ids)P(i,j)=0.;for(const auto&slot:s->sampled_imu_slots())if(slot.active)P.block<6,6>(slot.noise->id(),slot.noise->id())=slot.record.prior;
  StateHelper::set_initial_covariance(decorrelated,P,variables(decorrelated));
  check(p->sampled_state_at_endpoint(decorrelated,decorrelated->imu_endpoint(),wrong)&&(wrong.covariance-good.covariance).norm()>1e-6,
        "negative actual fresh independent original priors lose posterior common-noise output covariance");
}
void sequence(){for(auto model:{StateOptions::RPNG,StateOptions::KALIBR}){
  auto s=state(model);const auto r=records(s);auto p=std::make_shared<Probe>(noise(),9.81);
  StateHelper::admit_sampled_imu_noise(s,r[0]);auto old_owner=owner(s,1);auto initial=compare(s,r,p);
  check(initial.support_count==1&&initial.support[0].sequence==1&&initial.weights(0)==1.,"exact raw knot receipt has one original identity");
  propagate(s,r,p,.037);State::ExposurePose pose;pose.pose=StateHelper::augment_pose_view(s,0,independent_mean(s,r,s->imu_endpoint()).tail<3>());
  s->_exposure_poses.push_back(pose);visual(s,r,p);auto interior=compare(s,r,p);
  check(interior.support_count==2&&interior.support[0].sequence==1&&interior.support[1].sequence==2&&
    (interior.weights-Eigen::Vector2d(.63,.37)).norm()<1e-15,"interior receipt uses original endpoints and exact time-derived weights");controls(s,r,p);
  propagate(s,r,p,.1);const auto before=compare(s,r,p);Calibration c(s);
  auto stationary=[&](const std::shared_ptr<State>&x){const V3 force=x->_imu->Rot()*V3(0,0,9.81);V6 h=r[1].measured-owner(x,2)->value();
    h.head<3>()-=x->_imu->bias_g()+c.T*force;h.tail<3>()-=x->_imu->bias_a()+c.A.inverse()*force;return h;};
  const Mat middle=derivative(s,6,stationary,2e-4),fine=derivative(s,6,stationary,1e-4);const Mat H=(4*fine-middle)/3.;
  const auto expected=condition(s,H,-stationary(s),M6::Zero());UpdaterOptions options;options.chi2_multipler=1.;auto n=noise();
  UpdaterZeroVelocity z(options,n,std::make_shared<FeatureDatabase>(),p,9.81,1.,1.,1.);p->seed_cache();
  check(z.try_update_sampled_at_knot(s,r[1],M6::Zero()),"actual exact-knot ZUPT conditions the same retained raw reading");
  auto after=compare(s,r,p,&expected.P,expected.state);Mat cross(s->max_covariance_size(),12);
  check(p->sampled_state_at_endpoint(s,s->imu_endpoint(),after,&cross)&&!p->cached()&&(after.mean-before.mean).norm()>1e-5,
        "same-time output refreshes after successful ZUPT and its cache invalidation");
  check(after.mean.tail<3>().norm()<2e-12&&after.covariance.bottomRightCorner<3,3>().norm()<2e-12&&cross.rightCols<3>().norm()<2e-12,
        "exact stationarity removes angular output uncertainty/cross without adding independent sensor noise");
  visual(s,r,p);propagate(s,r,p,.16);const auto current=compare(s,r,p);
  check(owner(s,3)==old_owner&&current.support[0].sequence==2&&current.support[1].sequence==3,
        "reused Vec address is not reused as a historical output identity");
  Output sentinel=current;sentinel.mean.setConstant(123.);const auto saved=sentinel;Mat stale_cross=Mat::Constant(s->max_covariance_size(),12,456.);
  check(!p->sampled_state_at_endpoint(s,.037,sentinel,&stale_cross)&&same(sentinel,saved)&&(stale_cross.array()==456.).all(),
        "historical output request refuses after retirement and reuse without overwriting any output");
}}
void refusals(){auto s=state();const auto r=records(s);auto p=std::make_shared<Probe>(noise(),9.81);StateHelper::admit_sampled_imu_noise(s,r[0]);
  Output out;out.mean.setConstant(123.);out.covariance.setConstant(456.);const auto original=out;
  auto reject=[&](double time,Mat &cross){const auto covariance=StateHelper::get_full_covariance(s);const auto mean=s->_imu->value();Mat prior=cross;double*storage=cross.data();p->seed_cache();
    check(!p->sampled_state_at_endpoint(s,time,out,&cross)&&same(out,original)&&same(prior,cross)&&cross.data()==storage&&
      same(covariance,StateHelper::get_full_covariance(s))&&same(mean,s->_imu->value())&&p->cached(),"invalid output request is atomic for State, receipt, cross allocation and legacy cache");};
  Mat cross=Mat::Constant(s->max_covariance_size(),12,789.);reject(std::nextafter(0.,1.),cross);reject(std::numeric_limits<double>::quiet_NaN(),cross);
  Mat wrong_rows=Mat::Constant(s->max_covariance_size()+1,12,789.),wrong_cols=Mat::Constant(s->max_covariance_size(),11,789.);reject(0.,wrong_rows);reject(0.,wrong_cols);
  s->_options.do_fej=true;reject(0.,cross);s->_options.do_fej=false;s->_options.do_calib_imu_intrinsics=true;reject(0.,cross);s->_options.do_calib_imu_intrinsics=false;
  s->_options.do_calib_imu_g_sensitivity=true;reject(0.,cross);s->_options.do_calib_imu_g_sensitivity=false;s->_imu_endpoint_valid=false;reject(0.,cross);s->_imu_endpoint_valid=true;
  auto raw=owner(s,1);Vec value=raw->value(),bad=value;bad(0)=std::numeric_limits<double>::infinity();raw->set_value(bad);reject(0.,cross);raw->set_value(value);
  const auto da=s->_calib_imu_da->value(),dw=s->_calib_imu_dw->value(),tg=s->_calib_imu_tg->value();
  s->_calib_imu_da->set_value(Vec::Zero(6));reject(0.,cross);s->_calib_imu_da->set_value(da);s->_calib_imu_dw->set_value(Vec::Zero(6));reject(0.,cross);s->_calib_imu_dw->set_value(dw);
  Vec bad_tg=tg;bad_tg(0)=std::numeric_limits<double>::quiet_NaN();s->_calib_imu_tg->set_value(bad_tg);reject(0.,cross);s->_calib_imu_tg->set_value(tg);
  const int id=raw->id();raw->set_local_id(-1);reject(0.,cross);raw->set_local_id(id);
  auto imu=s->_imu;s->_imu=std::dynamic_pointer_cast<ov_type::IMU>(imu->clone());s->_imu->set_local_id(imu->id());reject(0.,cross);s->_imu=imu;
  s->_imu_endpoint=.05;reject(.05,cross);s->_imu_endpoint=0.;
  Eigen::Matrix<double,12,27> H=Eigen::Matrix<double,12,27>::Zero();C covariance=C::Constant(111.);
  const int inactive=s->sampled_imu_slots()[0].active?1:0;H(0,15+6*inactive)=1.;
  check(!StateHelper::project_sampled_imu_output(s,H,covariance,&cross)&&(covariance.array()==111.).all()&&(cross.array()==789.).all(),
        "projection rejects nonzero dependence on inactive reserved coordinates");
  H.setZero();H(0,0)=1e308;
  check(!StateHelper::project_sampled_imu_output(s,H,covariance,&cross)&&(covariance.array()==111.).all()&&(cross.array()==789.).all(),
        "overflowing projection is rejected before either output is committed");
  auto empty=state();Output other;other.mean.setConstant(222.);
  check(!p->sampled_state_at_endpoint(empty,0.,other)&&other.mean(0)==222.,"prepared State without raw support cannot emit an independent-noise substitute");
}
void resources(){
#ifdef __GLIBC__
  for(int poses:{0,20,100}){auto s=state();auto r=records(s);auto p=std::make_shared<Probe>(noise(),9.81);propagate(s,r,p,.05);
    for(int i=0;i<poses;++i){State::ExposurePose pose;pose.pose=StateHelper::augment_pose_view(s,0,V3::Zero());s->_exposure_poses.push_back(pose);}
    const int n=s->max_covariance_size();const auto slot0=s->sampled_imu_slots()[0].noise,slot1=s->sampled_imu_slots()[1].noise;
    Output out;Mat cross(n,12);double*storage=cross.data();check(p->sampled_state_at_endpoint(s,.05,out,&cross),"resource fixture warms linked evaluator");
    for(bool full:{false,true}){size_t calls=0,bytes=0,largest=0;bool accepted=true;
      for(int repeat=0;repeat<200;++repeat){heap_calls=heap_bytes=heap_largest=0;count_heap=true;
        const bool ok=p->sampled_state_at_endpoint(s,.05,out,full?&cross:nullptr);count_heap=false;
        accepted&=ok;calls=std::max(calls,heap_calls);bytes=std::max(bytes,heap_bytes);largest=std::max(largest,heap_largest);}
      check(accepted&&largest<=(full?128*size_t(n)+4096:4096)&&bytes<=(full?1024*size_t(n)+16384:32768),
            "actual projection allocations are fixed for marginal and linear for optional cross, never full-state quadratic scratch");
      std::printf("OUTPUT_RESOURCES n=%d full_cross=%d calls=200 max_heap_calls=%zu max_bytes=%zu largest=%zu\n",n,int(full),calls,bytes,largest);
    }
    check(s->max_covariance_size()==n&&s->sampled_imu_slots()[0].noise==slot0&&s->sampled_imu_slots()[1].noise==slot1&&cross.data()==storage,
          "repeated output introduces no State coordinates, raw owners or caller cross resize");
  }
#endif
}
}
int main(){Printer::setPrintLevel(Printer::WARNING);sequence();refusals();resources();
  std::printf("SAMPLED_ENDPOINT_OUTPUT %d/%d max_FD_cov=%.12g max_full_cross=%.12g max_mean=%.12g\n",checks-failures,checks,max_fd_cov,max_cross,max_mean);return failures?1:0;}
