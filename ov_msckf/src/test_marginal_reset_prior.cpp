/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#define main fixed_warm_fixture_main
#include "test_physical_warm_manager.cpp"
#undef main
#include "init/MarginalResetPrior.h"

namespace {
void brownian_prior() {
  ov_init::ResetBiasPrior prior;prior.valid=true;prior.t_snapshot=10.;prior.bg=bg;prior.ba=ba;
  prior.sigma_bg<<.02,.03,.04;prior.sigma_ba<<.05,.06,.07;
  ov_init::conditional_bias::Vector6 rw,floors;rw<<.001,.002,.003,.004,.005,.006;floors.setZero();
  ov_init::conditional_bias::Conditioned result;
  const double td=.127,gap=.4,first=prior.t_snapshot+td+gap,duration=2.;
  check(ov_init::condition_marginal_reset_prior(prior,first,td,rw,3.,floors,result),"external reference-clock prior conditions at the first physical node");
  Eigen::Matrix<double,6,6> priorP=Eigen::Matrix<double,6,6>::Zero();
  priorP.diagonal()<<prior.sigma_bg.array().square().matrix(),prior.sigma_ba.array().square().matrix();
  Eigen::Matrix<double,6,6> expected=priorP;expected.diagonal()+=gap*rw;
  check(near(expected,result.covariance,1e-14),"first-node prior agrees with independent Brownian increment covariance");
  Eigen::Matrix<double,6,6> final=result.covariance;final.diagonal()+=duration*rw;
  Eigen::Matrix<double,6,6> direct=priorP;direct.diagonal()+=(gap+duration)*rw;
  check(near(final,direct,1e-14),"first-node gap plus graph CPI equals one complete random walk");
  auto wrong=direct;wrong.diagonal()+=duration*rw;
  check((wrong-final).norm()>.01,"aging to last node and applying CPI twice is detected");
  check(!ov_init::condition_marginal_reset_prior(prior,first-gap-.01,td,rw,3.,floors,result),"a prior from after the first graph node cannot be aged backwards by clamping to zero");
  auto owned=std::make_shared<ov_init::ResetFilterPrior>();owned->imu_endpoint=10.8;owned->imu_raw_cutoff=10.80125;
  owned->raw_watermarks={10.7,10.9};owned->covariance=priorP;owned->covariance(0,3)=owned->covariance(3,0)=.0004;
  prior.filter=owned;prior.t_snapshot=-100.; // stale camera label must be irrelevant
  check(ov_init::valid_filter_reset_prior(*owned,2),"full correlated producer prior and cutoffs validate");
  check(ov_init::condition_marginal_reset_prior(prior,11.,99.,rw,3.,floors,result),"live reset age uses the captured endpoint, independent of camera clock changes");
  expected=owned->covariance;expected.diagonal()+=.2*rw;
  check(near(expected,result.covariance,1e-14) && std::abs(result.covariance(0,3)-.0004)<1e-15,"gyro/accel bias cross covariance survives aging and conditioning");
  const Eigen::Matrix<double,6,6> identity=result.sqrt_information*expected*result.sqrt_information.transpose();
  check(near(identity,Eigen::Matrix<double,6,6>::Identity(),1e-13),"full prior information whitens independent covariance to identity");
  prior.cause=1;
  check(ov_init::condition_marginal_reset_prior(prior,11.,99.,rw,3.,floors,result) && near(result.covariance,9.*expected,1e-14),"divergence inflation is a full covariance congruence");
  check(!ov_init::marginal_reset_future_row(*owned,0,10.7,.2) && ov_init::marginal_reset_future_row(*owned,0,std::nextafter(10.7,INFINITY),.2),
        "exact raw watermarks block reused images even after a clock shift moves their nominal exposure forward");
  for(int bad=0;bad<8;++bad) {
    auto p=prior;auto raw=std::make_shared<ov_init::ResetFilterPrior>(*owned);p.filter=raw;
    auto noise=rw;double t=11.;
    if(bad==0)raw->covariance(0,0)=-1.;
    if(bad==1)raw->covariance(0,3)=10.;
    if(bad==2)raw->imu_endpoint=std::numeric_limits<double>::quiet_NaN();
    if(bad==3)t=10.7;
    if(bad==4)noise(0)=-1.;
    if(bad==5)p.bg(0)=std::numeric_limits<double>::infinity();
    if(bad==6)p.cause=3;
    if(bad==7)p.valid=false;
    const auto saved=result.covariance;
    check(!ov_init::condition_marginal_reset_prior(p,t,.127,noise,3.,floors,result) && same(saved,result.covariance),"malformed prior cannot publish a partial aged covariance");
  }
}

VioManagerOptions legacy_reset_options(double td=.125) {
  auto p=options();p.state_options.physical_camera_clones=false;p.use_stereo=p.init_options.use_stereo=true;
  for(int camera=0;camera<2;++camera)p.camera_imu_dt[camera]=p.init_options.camera_imu_dt[camera]=td;
  p.state_options.do_calib_camera_timeoffset=p.state_options.do_calib_camera_pose=p.state_options.do_calib_camera_intrinsics=false;
  return p;
}
class LegacyResetManager : public Manager {
public:
  explicit LegacyResetManager(double td=.125):Manager(legacy_reset_options(td)){}
  ov_init::ResetBiasPrior prior() const {return initializer->reset_prior();}
  void prime_clock(bool single) {
    for(int frame=0;frame<=72;++frame)for(size_t camera=0;camera<2;++camera) {
      const double t=frame/30.+(camera==1 && frame%2 ? .013 : 0.);
      for(size_t feature=0;feature<80;++feature)add(feature,camera,10.+t-state->cam_imu_dt(camera),frame);
    }
    const auto input=samples(-160,1936);
    if(single)for(const auto &sample:input)feed_measurement_imu(sample);
    else feed_measurement_batch_imu(input);
  }
  bool advance(double endpoint) {
    // Legacy cold import starts with no retained clones. Exercise its public
    // image path until the configured three-clone window can perform an update.
    int next=1937;
    for(int tick:{1952,1976,2000}) {
      queue_group(tick,80);feed_measurement_batch_imu(samples(next,tick+11));next=tick+12;
    }
    if(!initialized())return false;
    Propagator::EndpointKinematics kinematics;
    const bool ok=propagator->propagate_to_imu(state,endpoint,endpoint-state->cam_imu_dt_ref(),kinematics);
    return ok;
  }
  void set_bias_covariance(const Eigen::Matrix<double,6,6> &Pbb) {
    // Public setter for an independent known prior in the real running State.
    auto P=StateHelper::get_full_covariance(state);P.middleRows(9,6).setZero();P.middleCols(9,6).setZero();P.block<6,6>(9,9)=Pbb;
    std::vector<std::shared_ptr<ov_type::Type>> order{state->_imu};
    for(const auto &clone:state->_clones_IMU)order.push_back(clone.second);
    std::sort(order.begin(),order.end(),[](const auto &a,const auto &b){return a->id()<b->id();});
    StateHelper::set_initial_covariance(state,P,order);
    auto value=state->_imu->value();value.block<6,1>(10,0)<<bg,ba;state->_imu->set_value(value);
  }
  void future_window() {
    for(int frame=0;frame<=72;++frame)for(size_t camera=0;camera<2;++camera) {
      const double t=2.6+frame/30.+(camera==1 && frame%2 ? .013 : 0.);
      for(size_t feature=0;feature<80;++feature)add(feature,camera,10.+t-state->cam_imu_dt(camera),90+frame);
    }
    feed_measurement_batch_imu(samples(2012,4016));
  }
  void set_readout(double value) {
    Eigen::VectorXd readout(1);readout(0)=value;state->_calib_camera_readout.at(1)->set_value(readout);
  }
};
void clock_pruning() {
  for(double td:{-.125,0.,.125})for(bool single:{false,true}) {
    LegacyResetManager manager(td);manager.prime_clock(single);manager.attempt();manager.finish_worker_only();
    const bool accepted=manager.attempt() && manager.success();
    check(accepted,single ? "single-sample IMU input keeps the physical initialization window for either clock sign" :
                           "batch IMU input keeps the physical initialization window for either clock sign");
    if(accepted)check(std::abs(manager.get_state()->imu_endpoint()-12.4)<1e-12,
                      "initialization endpoint is independent of the reference camera label");
  }
}
void capture_guards() {
  for(int invalid=0;invalid<5;++invalid) {
    LegacyResetManager manager(0.);manager.prime();manager.attempt();manager.finish_worker_only();
    const bool accepted=manager.attempt() && manager.success() && manager.advance(12.5004);
    check(accepted,"reset rejection control first completes the real initialization and image update");
    if(!accepted)continue;
    Eigen::Matrix<double,6,6> Pbb=Eigen::Matrix<double,6,6>::Identity()*.0001;
    if(invalid==0)Pbb(0,0)=-1.;
    if(invalid==1)Pbb(0,3)=Pbb(3,0)=1.;
    if(invalid==2)Pbb(0,0)=std::numeric_limits<double>::quiet_NaN();
    manager.set_bias_covariance(Pbb);
    if(invalid>=3)manager.set_readout(invalid==3 ? .01 : std::numeric_limits<double>::infinity());
    manager.soft_reset();const auto prior=manager.prior();
    check(!prior.valid && !prior.filter && !prior.joint,
          "invalid full bias covariance or unsupported readout cannot manufacture a reusable reset prior");
  }
}
void actual_legacy_reset() {
  LegacyResetManager manager;manager.prime();manager.attempt();manager.finish_worker_only();
  check(manager.attempt() && manager.success(),"actual legacy first boot succeeds");if(!manager.success())return;
  check(manager.advance(12.5004),"ordinary propagation accepts the prior's physical endpoint");
  Eigen::Matrix<double,6,6> L=Eigen::Matrix<double,6,6>::Zero();L.diagonal()<<.006,.007,.005,.025,.03,.035;
  L(3,0)=.004;L(4,1)=-.003;const Eigen::Matrix<double,6,6> Pbb=L*L.transpose();manager.set_bias_covariance(Pbb);
  manager.soft_reset();auto prior=manager.prior();
  check(prior.valid && prior.filter && !prior.joint,"actual nonphysical soft reset captures an owned full bias prior");if(!prior.filter)return;
  check(same(prior.filter->covariance,Pbb) && prior.filter->imu_endpoint==12.5004 && prior.filter->imu_raw_cutoff>12.5004 &&
        prior.filter->imu_raw_cutoff<12.502,"capture keeps full covariance and actual interpolation support, not camera time or buffered horizon");
  const auto before=rows(manager.database());
  check(!manager.attempt(),"old legacy reset window produces no immediate result");manager.finish_worker_only();
  check(!manager.attempt() && !manager.success() && rows(manager.database())==before,"legacy posterior-conditioned image/CPI likelihood cannot be reused");
  const auto snapshot=manager.snapshot();manager.restore(snapshot,{});
  check(manager.prior().filter==prior.filter && manager.reset_armed(),"pending legacy reset snapshot preserves the immutable prior");
  manager.future_window();check(!manager.attempt(),"future-only legacy worker launches");manager.finish_worker_only();
  check(manager.attempt() && manager.success(),"real legacy graph and consumer reinitialize from new measurements");if(!manager.success())return;
  const auto state=manager.get_state();
  check(StateHelper::valid_initial_covariance(StateHelper::get_full_covariance(state),true),"legacy reset output covariance is PSD");
  check((state->_imu->Rot()*Eigen::Vector3d::UnitZ()-rotation(5.)*Eigen::Vector3d::UnitZ()).norm()<.02 &&
        (state->_imu->Rot()*state->_imu->vel()-rotation(5.)*velocity(5.)).norm()<.04,"legacy reset recovers independent analytic tilt and body velocity");
  std::printf("MARGINAL_RESET_ENDPOINT %.9f filter %.9f camera %.9f\n",state->imu_endpoint(),prior.filter->imu_endpoint,prior.t_snapshot);
}
}
int main() {
  ov_core::Printer::setPrintLevel(ov_core::Printer::WARNING);
  try{brownian_prior();clock_pruning();capture_guards();actual_legacy_reset();}
  catch(const std::exception &error){check(false,error.what());}
  std::printf("MARGINAL_RESET_PRIOR %s checks=%d failures=%d\n",failures?"FAIL":"PASS",checks,failures);
  return failures?1:0;
}
