/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Actual nonlinear initializer/reset/async consumer; no solver interposition.
 */
#define main fixed_warm_fixture_main
#include "test_physical_warm_manager.cpp"
#undef main
#include "init/PhysicalResetWindow.h"

namespace {
VioManagerOptions reset_options() {
  auto p=options();
  p.state_options.do_calib_camera_timeoffset=true;
  p.state_options.do_calib_camera_pose=true;
  p.state_options.do_calib_camera_intrinsics=true;
  return p;
}
std::shared_ptr<ov_type::Type> reset_type(const std::shared_ptr<State> &state,const InitCameraCalibrationBlock &block) {
  if(block.kind==InitCameraCalibrationKind::Clock)return state->cam_imu_dt_var(block.camera_id);
  if(block.kind==InitCameraCalibrationKind::Extrinsics)return state->_calib_IMUtoCAM.at(block.camera_id);
  return state->_cam_intrinsics.at(block.camera_id);
}
std::vector<std::shared_ptr<ov_type::Type>> all_types(const std::shared_ptr<State> &state) {
  std::vector<std::shared_ptr<ov_type::Type>> types{state->_imu};
  for(int camera=0;camera<state->_options.num_cameras;++camera) {
    for(const auto &type:std::vector<std::shared_ptr<ov_type::Type>>{state->cam_imu_dt_var(camera),state->_calib_IMUtoCAM.at(camera),state->_cam_intrinsics.at(camera)})
      if(type->id()>=0)types.push_back(type);
  }
  for(const auto &owner:state->_exposure_poses)types.push_back(owner.pose);
  std::sort(types.begin(),types.end(),[](const auto &a,const auto &b){return a->id()<b->id();});
  return types;
}
class ResetManager : public Manager {
public:
  explicit ResetManager(VioManagerOptions p=reset_options()):Manager(p){}
  std::shared_ptr<const InitPhysicalResetPrior> prior() const {return initializer->physical_reset_prior();}
  void queue_at(double endpoint,int frame) {
    for(size_t camera=0;camera<2;++camera) {
      FeatureObservations obs;
      for(size_t feature=0;feature<80;++feature) {
        Eigen::VectorXf uv(2);uv=pixel(feature,camera,endpoint-10.,frame);obs.emplace_back(feature,uv);
      }
      feed_measurement_simulation_queued(endpoint-state->cam_imu_dt(camera),{int(camera)},{std::move(obs)});
    }
  }
  void future_window() {
    for(int frame=0;frame<=72;++frame)
      for(size_t camera=0;camera<2;++camera) {
        const double t=2.6+frame/30.+(camera==1 && frame%2 ? .013 : 0.);
        for(size_t feature=0;feature<80;++feature)
          add(feature,camera,10.+t-state->cam_imu_dt(camera),90+frame);
      }
    feed_measurement_batch_imu(samples(2012,4016));
  }
  bool reference(InitPhysicalWarmResult &result,bool permute=false) {
    auto input=initializer->make_attempt();InitPhysicalWarmRequest request;
    if(!StateHelper::make_initial_physical_warm_request(state,1,request,prior()))return false;
    if(permute) {
      const auto blocks=request.consider;const auto covariance=request.calibration_covariance;
      std::vector<int> starts,indices;int total=0;
      for(const auto &block:blocks){starts.push_back(total);total+=block.local_size();}
      request.consider.clear();
      for(size_t i=blocks.size();i-->0;) {
        request.consider.push_back(blocks[i]);
        for(int j=0;j<blocks[i].local_size();++j)indices.push_back(starts[i]+j);
      }
      for(int r=0;r<total;++r)for(int c=0;c<total;++c)request.calibration_covariance(r,c)=covariance(indices[r],indices[c]);
    }
    if(!input->request_physical_warmstart(request))return false;
    double timestamp=-1.;Eigen::MatrixXd covariance;std::vector<std::shared_ptr<ov_type::Type>> order;
    auto imu=std::make_shared<ov_type::IMU>();std::map<double,std::shared_ptr<ov_type::PoseJPL>> clones;
    std::unordered_map<size_t,std::shared_ptr<ov_type::Landmark>> landmarks;
    if(!input->initialize(timestamp,covariance,order,imu,clones,landmarks,true) || !input->physical_warm_result())return false;
    result=*input->physical_warm_result();return true;
  }
  void replace_reset_prior(std::shared_ptr<const InitPhysicalResetPrior> prior) {
    if(!initializer->set_physical_reset_prior(std::move(prior)))throw std::runtime_error("fixture replacement prior rejected");
  }
  void forget_provenance() { auto snapshot=propagator->capture();snapshot.imu_data.clear();propagator->restore(snapshot); }
  void install_joint_test_marginal(bool singular) {
    auto P=StateHelper::get_full_covariance(state);
    std::vector<int> cols;std::vector<int> clock_cols;
    for(int camera=0;camera<2;++camera) {
      for(const auto &type:std::vector<std::shared_ptr<ov_type::Type>>{state->cam_imu_dt_var(camera),state->_calib_IMUtoCAM.at(camera),state->_cam_intrinsics.at(camera)}) {
        if(type==state->cam_imu_dt_var(camera))clock_cols.push_back(cols.size());
        for(int j=0;j<type->size();++j)cols.push_back(type->id()+j);
      }
      // A changed current chart with its old FEJ still present is a real reset
      // input. Rebase is checked independently below.
      auto clock=state->cam_imu_dt_var(camera)->value();clock(0)+=.002;
      state->cam_imu_dt_var(camera)->set_value(clock);
      auto pose=state->_calib_IMUtoCAM.at(camera)->value();pose(4)+=1e-5;
      state->_calib_IMUtoCAM.at(camera)->set_value(pose);
      auto intr=state->_cam_intrinsics.at(camera)->value();intr(0)+=.001;
      state->_cam_intrinsics.at(camera)->set_value(intr);state->_cam_intrinsics_cameras.at(camera)->set_value(intr);
    }
    const int k=cols.size();Eigen::MatrixXd L=Eigen::MatrixXd::Identity(k,k);
    for(int r=0;r<k;++r)for(int c=0;c<k;++c)L(r,c)+=.02*std::sin(.2+r+.7*c);
    for(int r=0;r<k;++r)L.row(r)*=r%15==0 ? .002 : (r%15<7 ? .001 : (r%15<11 ? .1 : .0001));
    if(singular)L.row(clock_cols[1])=L.row(clock_cols[0]);
    Eigen::MatrixXd B(6,k);Eigen::Matrix<double,6,6> Q=Eigen::Matrix<double,6,6>::Identity();
    for(int r=0;r<6;++r) {
      const double sigma=r<3 ? .006 : .03;
      Q.row(r)*=sigma;
      for(int c=0;c<k;++c)B(r,c)=sigma*.15*std::sin(.3+r+.9*c)/std::sqrt(double(k));
    }
    Q(3,0)=.004;Q(4,1)=-.003;Q(2,0)=.001;
    const Eigen::MatrixXd Pcc=L*L.transpose(),Pbc=B*L.transpose(),Pbb=Q*Q.transpose()+B*B.transpose();
    for(int c:cols){P.row(c).setZero();P.col(c).setZero();}
    P.middleRows(9,6).setZero();P.middleCols(9,6).setZero();P.block<6,6>(9,9)=Pbb;
    for(int r=0;r<k;++r) {
      P.block<6,1>(9,cols[r])=Pbc.col(r);P.block<1,6>(cols[r],9)=Pbc.col(r).transpose();
      for(int c=0;c<k;++c)P(cols[r],cols[c])=Pcc(r,c);
    }
    StateHelper::set_initial_covariance(state,P,all_types(state));
    auto value=state->_imu->value();value.block<6,1>(10,0)<<bg,ba;state->_imu->set_value(value);
    check(StateHelper::valid_initial_covariance(P,true),"controlled full marginal and untouched navigation marginal form a valid joint state");
  }
};

void window_controls() {
  auto stream=samples(0,20);double cutoff=-123.;
  const double endpoint=stream[5].timestamp+.0004;
  check(ov_init::physical_reset_raw_cutoff(stream,endpoint,cutoff) && cutoff==stream[6].timestamp && cutoff!=stream.back().timestamp,
        "accepted endpoint between raw records owns the right interpolation sample, not the latest buffered horizon");
  check(stream[6].timestamp>endpoint && !(stream[6].timestamp>cutoff),
        "endpoint-only new-IMU cutoff would incorrectly reuse the right interpolation record");
  check(ov_init::physical_reset_raw_cutoff(stream,stream[5].timestamp,cutoff) && cutoff==stream[5].timestamp,
        "an exact raw endpoint needs no following interpolation record");
  for(int bad=0;bad<5;++bad) {
    auto candidate=stream;double time=endpoint;
    if(bad==0)candidate.erase(candidate.begin(),candidate.begin()+6);
    if(bad==1)candidate.resize(6);
    if(bad==2)candidate[6].timestamp=candidate[5].timestamp;
    if(bad==3)candidate[5].am(0)=std::numeric_limits<double>::quiet_NaN();
    if(bad==4)time=std::numeric_limits<double>::infinity();
    cutoff=-123.;check(!ov_init::physical_reset_raw_cutoff(candidate,time,cutoff) && cutoff==-123.,
                      "missing or invalid interpolation provenance declines atomically");
  }
  InitPhysicalResetPrior p;p.imu_endpoint=12.5;p.raw_watermarks={12.375};
  InitFixedCameraCalibration c;c.clock_mean=.127;p.calibration={c};
  check(12.375+c.clock_mean>p.imu_endpoint && !ov_init::physical_reset_future_row(p,0,12.375),
        "clock correction cannot recycle an old raw image whose nominal physical time crossed the reset endpoint");
  check(!ov_init::physical_reset_future_row(p,0,std::nextafter(12.375,0.)) &&
        ov_init::physical_reset_future_row(p,0,std::nextafter(12.375,INFINITY)),
        "dual future cutoffs retain exact raw-bit neighbour semantics");
  p.calibration[0].clock_mean=.12;
  check(!ov_init::physical_reset_future_row(p,0,12.377),"new raw time with old physical exposure is still excluded");
}

void invalid_reset_imports(const std::shared_ptr<State> &seed,const InitPhysicalWarmResult &result) {
  for(int bad=0;bad<8;++bad) {
    auto state=StateHelper::clone_state(seed);auto candidate=result;auto expected=result.reset_prior;
    if(bad==0)candidate.reset_prior.reset();
    if(bad==1)expected=std::make_shared<InitPhysicalResetPrior>(*expected);
    if(bad==2)candidate.reset_first_imu_support_time=expected->imu_raw_cutoff;
    if(bad==3)candidate.reset_first_imu_time=std::nextafter(candidate.reset_first_imu_support_time,0.);
    if(bad==4)candidate.consumed_observations.front().raw_time=expected->raw_watermarks[candidate.consumed_observations.front().camera_id];
    if(bad==5)candidate.reset_first_imu_time=std::numeric_limits<double>::quiet_NaN();
    if(bad==6) {auto b=state->_imu->value();b(10)+=1e-8;state->_imu->set_value(b);}
    if(bad==7) {auto P=StateHelper::get_full_covariance(state);P(9,15)+=1e-12;P(15,9)+=1e-12;StateHelper::set_initial_covariance(state,P,all_types(state));}
    const auto P=StateHelper::get_full_covariance(state),mean=state->_imu->value(),fej=state->_imu->fej();const auto imu=state->_imu;
    check(!StateHelper::set_initial_state_physical_warm(state,candidate,1,expected),"invalid correlated reset import rejects");
    check(imu==state->_imu && same(P,StateHelper::get_full_covariance(state)) && same(mean,state->_imu->value()) &&
          same(fej,state->_imu->fej()) && !state->_imu_endpoint_valid && state->clone_count()==0,
          "reset rejection preserves handles, means, FEJ, covariance, endpoint and owners");
  }
}

void capture_controls(const std::shared_ptr<State> &live,const std::shared_ptr<const InitPhysicalResetPrior> &prior) {
  for(int bad=0;bad<12;++bad) {
    auto candidate=StateHelper::clone_state(live);double cutoff=prior->imu_raw_cutoff;auto watermarks=prior->raw_watermarks;
    if(bad==0)cutoff=std::nextafter(prior->imu_endpoint,0.);
    if(bad==1)candidate->_options.do_calib_imu_intrinsics=true;
    if(bad==2)candidate->_options.do_calib_camera_readout=true;
    if(bad==3){auto v=candidate->_calib_camera_readout.at(0)->value();v(0)=.001;candidate->_calib_camera_readout.at(0)->set_value(v);}
    if(bad==4){auto v=candidate->_calib_IMUtoCAM.at(0)->value();v.block<4,1>(0,0)*=2.;candidate->_calib_IMUtoCAM.at(0)->set_value(v);}
    if(bad==5){auto P=StateHelper::get_full_covariance(candidate);P(9,15)=std::numeric_limits<double>::quiet_NaN();StateHelper::set_initial_covariance(candidate,P,all_types(candidate));}
    if(bad==6){auto P=StateHelper::get_full_covariance(candidate);P(9,15)=100.*std::sqrt(P(9,9)*P(15,15));StateHelper::set_initial_covariance(candidate,P,all_types(candidate));}
    if(bad==7){auto v=candidate->_cam_intrinsics.at(0)->value();v(0)+=1.;candidate->_cam_intrinsics.at(0)->set_value(v);}
    if(bad==8)check(StateHelper::prepare_sampled_imu_boundary(candidate,75),"unsupported sampled-boundary producer control is valid");
    if(bad==9)watermarks[0]=std::numeric_limits<double>::quiet_NaN();
    if(bad==10)watermarks[0]=std::numeric_limits<double>::infinity();
    if(bad==11)watermarks.pop_back();
    const auto P=StateHelper::get_full_covariance(candidate);const auto mean=candidate->_imu->value();
    std::shared_ptr<State> replacement=live;auto output=prior;
    const bool accepted=StateHelper::make_physical_reset_state(candidate,74,watermarks,cutoff,prior->bias_rw_variance,0,replacement,output);
    if(accepted)std::printf("CAPTURE_UNEXPECTED_ACCEPT case=%d clock_id=%d imu_cutoff=%.17g endpoint=%.17g\n",bad,candidate->cam_imu_dt_var(0)->id(),cutoff,candidate->imu_endpoint());
    check(!accepted,"unsupported or invalid reset producer refuses before publishing replacement");
    check(replacement==live && output==prior && same(P,StateHelper::get_full_covariance(candidate)) && same(mean,candidate->_imu->value()),
          "rejected capture leaves both output handles and source mean/covariance unchanged");
  }
  for(int mode=0;mode<4;++mode) {
    auto p=reset_options().init_options;
    if(mode==0)p.init_dyn_mle_opt_calib=true;
    if(mode==1)p.init_dyn_fix_ba_on_reset=true;
    if(mode==2)p.init_dyn_reset_prior_use=false;
    if(mode==3)p.sigma_wb*=2.;
    ov_init::InertialInitializer unsupported(p,std::make_shared<FeatureDatabase>());
    check(!unsupported.set_physical_reset_prior(prior) && !unsupported.physical_reset_prior(),
          "fitted means, hard-frozen bias, disabled prior and mismatched runtime RW remain explicit unsupported modes");
  }
}

void actual_reset(bool singular) {
  ResetManager manager;manager.prime();manager.attempt();manager.finish_worker_only();
  check(manager.attempt() && manager.success(),"real camera-consider first boot completes before reset");
  if(!manager.success())return;
  manager.queue_at(12.5004,80);manager.feed_measurement_batch_imu(samples(1937,2011));
  auto old=manager.get_state();
  check(manager.initialized() && std::abs(old->imu_endpoint()-12.5004)<1e-8,"actual queued propagation accepts an endpoint between raw IMU samples");
  if(!singular) {
    const auto accepted=manager.snapshot();manager.forget_provenance();manager.soft_reset();
    check(!manager.prior() && !manager.initialized(),"actual missing accepted-endpoint provenance declines joint policy before reset selection");
    manager.restore(accepted,{});old=manager.get_state();
  }
  manager.install_joint_test_marginal(singular);
  const auto old_cov=StateHelper::get_full_covariance(old);const auto old_endpoint=old->imu_endpoint();
  manager.soft_reset(singular ? VioManager::SoftResetCause::DIVERGENCE : VioManager::SoftResetCause::CLIENT);const auto prior=manager.prior();
  check(prior && manager.reset_armed() && !manager.initialized(),"actual soft reset selects the joint conditional episode");
  if(!prior)return;
  if(!singular)capture_controls(old,prior);
  auto state=manager.get_state();
  check(prior->imu_endpoint==old_endpoint && prior->imu_raw_cutoff>prior->imu_endpoint && prior->imu_raw_cutoff<12.502 &&
        same(prior->bias_covariance,old_cov.block<6,6>(9,9).eval()) && prior->bias_calibration_covariance.norm()>1e-7,
        "capture retains authoritative endpoint, right raw support and the complete bias/calibration marginal");
  bool chart=true;
  for(const auto &block:prior->consider) {
    const auto type=reset_type(state,block),previous=reset_type(old,block);
    chart&=same(type->value(),previous->value()) && same(type->fej(),type->value());
  }
  check(chart,"new episode keeps current camera means and explicitly rebases their FEJ");
  check(state->_cam_intrinsics_cameras.at(0)==old->_cam_intrinsics_cameras.at(0),"reset retains the frontend camera object identity");
  for(size_t camera=0;camera<2;++camera)
    check(prior->raw_watermarks[camera]+prior->calibration[camera].clock_mean>prior->imu_endpoint &&
          !ov_init::physical_reset_future_row(*prior,camera,prior->raw_watermarks[camera]),
          "actual online clock shift moves a used boundary row past the physical endpoint; immutable raw cutoff still excludes it");
  const auto before=rows(manager.database());const auto seed=StateHelper::clone_state(state);
  check(!manager.attempt(),"old posterior-conditioned window cannot immediately reinitialize");manager.finish_worker_only();
  check(!manager.attempt() && !manager.success() && rows(manager.database())==before &&
        same(StateHelper::get_full_covariance(state),StateHelper::get_full_covariance(seed)),
        "failed old-window attempt preserves joint state and observation rows without cold substitution");
  const auto checkpoint=manager.snapshot();
  check(checkpoint->physical_reset_prior==prior,"preinit snapshot owns the immutable joint reset cutoffs");
  manager.restore(checkpoint,{});state=manager.get_state();
  check(manager.prior()==prior && manager.reset_armed() && !manager.initialized(),"restore re-arms the same reset chart, support and raw cutoffs");
  manager.future_window();
  if(!singular) {
    const auto future_snapshot=manager.snapshot();const auto future_rows=rows(manager.database());
    check(!manager.attempt(),"restore control launches a real joint-reset worker from future data");
    manager.restore(future_snapshot,{});state=manager.get_state();
    check(!manager.pending() && !manager.success() && manager.prior()==prior && rows(manager.database())==future_rows &&
          same(StateHelper::get_full_covariance(state),StateHelper::get_full_covariance(future_snapshot->state)),
          "restore joins/discards an abandoned joint worker and retains immutable support, raw cutoffs and the complete pending state");
  }
  InitPhysicalWarmResult result;
  const bool solved=manager.reference(result);check(solved,"real future-only conditional reset graph solves and exports");
  if(!solved)return;
  check(result.reset_prior==prior && result.reset_first_imu_support_time>prior->imu_raw_cutoff &&
        result.reset_first_imu_time>=result.reset_first_imu_support_time && result.accepted_imu_endpoint>result.reset_first_imu_time,
        "reset graph uses future raw interpolation support and ages the bias prior only to its first node");
  bool future=!result.consumed_observations.empty();
  for(const auto &row:result.consumed_observations)future&=ov_init::physical_reset_future_row(*prior,row.camera_id,row.raw_time);
  check(future && same(result.calibration_covariance,prior->calibration_covariance),"all exported likelihood receipts are new and full calibration covariance is retained once");
  invalid_reset_imports(StateHelper::clone_state(state),result);
  if(!singular) {
    InitPhysicalWarmResult permuted;const bool solved_permuted=manager.reference(permuted,true);
    auto alternate=StateHelper::clone_state(state);
    check(solved_permuted && StateHelper::set_initial_state_physical_warm(alternate,permuted,1,prior),
          "real joint bias factor retains its semantic columns under a permuted camera request and import");
    if(solved_permuted) {
      std::vector<std::shared_ptr<ov_type::Type>> types;for(const auto &block:prior->consider)types.push_back(reset_type(alternate,block));
      check(same(StateHelper::get_marginal_covariance(alternate,types),prior->calibration_covariance) && near(permuted.imu_mean,result.imu_mean),
            "permuted conditional reset preserves canonical Pc and the independently solved navigation mean");
    }
  }
  check(!manager.attempt(),"real joint reset worker launches");manager.finish_worker_only();
  auto stale=std::make_shared<InitPhysicalResetPrior>(*prior);++stale->snapshot_id;manager.replace_reset_prior(stale);
  const auto pending_rows=rows(manager.database());const auto pending_cov=StateHelper::get_full_covariance(state);
  check(!manager.attempt() && !manager.success() && rows(manager.database())==pending_rows &&
        same(pending_cov,StateHelper::get_full_covariance(state)),"stale immutable reset identity rejects an actual completed worker atomically");
  manager.replace_reset_prior(prior);
  check(!manager.attempt(),"joint reset can retry after stale output without discarding its prior");
  const auto preserved=append_arrivals(manager,result);const auto rows_before=rows(manager.database());
  manager.finish_worker_only();
  check(rows(manager.database())==rows_before && same(pending_cov,StateHelper::get_full_covariance(state)),
        "successful worker still cannot consume rows or publish covariance before consumer import");
  check(manager.attempt() && manager.success() && !manager.prior(),"actual consumer imports conditional reset posterior and disarms the prior");
  if(!manager.success())return;
  check((state->_imu->Rot()*Eigen::Vector3d::UnitZ()-rotation(5.)*Eigen::Vector3d::UnitZ()).norm()<.02 &&
        (state->_imu->Rot()*state->_imu->vel()-rotation(5.)*velocity(5.)).norm()<.04,
        "reset result agrees with independent analytic tilt and body velocity in its new gauge");
  const auto after=rows(manager.database());bool consumed=true,retained=true;
  for(const auto &row:result.consumed_observations)consumed&=!after.count(key(row));
  for(const auto &row:preserved)retained&=after.count(row)==1;
  check(consumed && retained,"reset consumes exact new likelihood rows once and preserves unused/future arrivals");
  std::vector<std::shared_ptr<ov_type::Type>> calib;for(const auto &block:prior->consider)calib.push_back(reset_type(state,block));
  check(same(StateHelper::get_marginal_covariance(state,calib),prior->calibration_covariance),"actual reset consumer preserves full, possibly singular, calibration covariance");
  check(state->clone_count()<=size_t(state->_options.max_pose_clones()) && StateHelper::valid_initial_covariance(StateHelper::get_full_covariance(state),true),
        "accepted reset posterior is bounded and PSD");
  size_t processed=0,dropped=0;manager.set_camera_processed_callback([&](const CameraData &m,bool accepted){(accepted?processed:dropped)+=m.sensor_ids.size();return true;});
  int next=4017;
  for(int frame=0;frame<4;++frame) {
    const int tick=4080+40*frame;manager.queue_at(10.+tick/800.,170+frame);manager.feed_measurement_batch_imu(samples(next,tick+10));next=tick+11;
    check(manager.initialized() && state->clone_count()<=size_t(state->_options.max_pose_clones()) &&
          StateHelper::valid_initial_covariance(StateHelper::get_full_covariance(state),true),"ordinary reset continuation keeps a bounded PSD camera-owner state");
  }
  manager.finish_observation_replay();check(processed==8 && dropped==0,"all reset continuation views complete through public queued callbacks");
  manager.set_camera_processed_callback({});
  std::printf("RESET_CASE singular=%d receipts=%zu first=%.9f endpoint=%.9f\n",singular,result.consumed_observations.size(),result.reset_first_imu_time,result.accepted_imu_endpoint);
}
} // namespace
#ifndef OV_RESET_FIXTURE_ONLY
int main() {
  ov_core::Printer::setPrintLevel(ov_core::Printer::WARNING);
  try{window_controls();actual_reset(false);actual_reset(true);}
  catch(const std::exception &error){check(false,error.what());}
  std::printf("RESET_CONDITIONAL_WARMSTART %s checks=%d failures=%d\n",failures?"FAIL":"PASS",checks,failures);
  return failures?1:0;
}
#endif
