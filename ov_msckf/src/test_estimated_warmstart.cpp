/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Reuse the independently visible analytic scene and real async manager fixture.
 * No initializer interposition or manufactured solver result is used here.
 */
#define main fixed_warm_manager_fixture_main
#include "test_physical_warm_manager.cpp"
#undef main

namespace {
std::shared_ptr<ov_type::Type> calibration_type(const std::shared_ptr<State> &state,const InitCameraCalibrationBlock &block) {
  if (block.kind==InitCameraCalibrationKind::Clock) return state->cam_imu_dt_var(block.camera_id);
  if (block.kind==InitCameraCalibrationKind::Extrinsics) return state->_calib_IMUtoCAM.at(block.camera_id);
  return state->_cam_intrinsics.at(block.camera_id);
}
std::vector<std::shared_ptr<ov_type::Type>> calibration_types(const std::shared_ptr<State> &state) {
  InitPhysicalWarmRequest snapshot;
  if (!StateHelper::make_initial_physical_warm_request(state,1,snapshot)) throw std::runtime_error("fresh camera prior snapshot failed");
  std::vector<std::shared_ptr<ov_type::Type>> output;
  for (const auto &block : snapshot.consider) output.push_back(calibration_type(state,block));
  return output;
}
Eigen::MatrixXd canonical_covariance(const std::shared_ptr<State> &state,const InitPhysicalWarmResult &result) {
  std::vector<int> indices;
  for (int i=0;i<15;++i) indices.push_back(i);
  for (const auto &owner : result.owners) {
    const auto pose=state->find_pose(owner.camera_id,owner.raw_time);
    if (!pose) throw std::runtime_error("missing imported owner");
    for (int j=0;j<6;++j) indices.push_back(pose->id()+j);
  }
  for (const auto &block : result.consider) {
    const auto type=calibration_type(state,block);
    for (int j=0;j<type->size();++j) indices.push_back(type->id()+j);
  }
  const auto P=StateHelper::get_full_covariance(state);
  Eigen::MatrixXd output(indices.size(),indices.size());
  for (size_t r=0;r<indices.size();++r) for (size_t c=0;c<indices.size();++c) output(r,c)=P(indices[r],indices[c]);
  return output;
}

class EstimatedManager : public Manager {
public:
  explicit EstimatedManager(VioManagerOptions p) : Manager(p) {}
  void set_prior(bool singular=false,double scale=1.) {
    const auto types=calibration_types(state);
    InitPhysicalWarmRequest snapshot;
    if (!StateHelper::make_initial_physical_warm_request(state,1,snapshot)) throw std::runtime_error("snapshot unavailable");
    const int n=snapshot.calibration_covariance.rows();
    Eigen::MatrixXd L=Eigen::MatrixXd::Identity(n,n);
    for (int r=0;r<n;++r) for (int c=0;c<n;++c) L(r,c)+=.035*std::sin(.8*r+.7*c+.2);
    L=snapshot.calibration_covariance.diagonal().cwiseSqrt().asDiagonal()*L;
    if (singular) {
      int offset=0,first=-1;
      for (const auto &block : snapshot.consider) {
        if (block.kind==InitCameraCalibrationKind::Clock) {
          if (first<0) first=offset; else L.row(offset)=L.row(first);
        }
        offset+=block.local_size();
      }
    }
    StateHelper::set_initial_covariance(state,scale*(L*L.transpose()),types);
  }
  bool solve_reference(InitPhysicalWarmResult &result,bool permute=true,size_t cap=0) {
    auto input=initializer->make_attempt();
    InitPhysicalWarmRequest request;
    if (!StateHelper::make_initial_physical_warm_request(state,1,request)) return false;
    if (cap) request.max_retained_owners=cap;
    if (permute) {
      auto previous=request.consider;
      std::vector<int> offsets; int total=0;
      for (const auto &block : previous) { offsets.push_back(total); total+=block.local_size(); }
      std::vector<int> indices;
      request.consider.clear();
      for (size_t i=previous.size();i-->0;) {
        request.consider.push_back(previous[i]);
        for (int j=0;j<previous[i].local_size();++j) indices.push_back(offsets[i]+j);
      }
      const auto P=request.calibration_covariance;
      for (int r=0;r<total;++r) for (int c=0;c<total;++c) request.calibration_covariance(r,c)=P(indices[r],indices[c]);
    }
    if (!input->request_physical_warmstart(request)) return false;
    auto imu=std::make_shared<ov_type::IMU>();
    double time=-1.; Eigen::MatrixXd covariance;
    std::vector<std::shared_ptr<ov_type::Type>> order;
    std::map<double,std::shared_ptr<ov_type::PoseJPL>> clones;
    std::unordered_map<size_t,std::shared_ptr<ov_type::Landmark>> features;
    if (!input->initialize(time,covariance,order,imu,clones,features,true) || !input->physical_warm_result()) return false;
    result=*input->physical_warm_result();
    check(covariance.rows()==15 && near(covariance,result.joint_covariance.topLeftCorner(15,15)),
          "public initializer output remains the unambiguous IMU marginal");
    return true;
  }
  bool can_request() {
    InitPhysicalWarmRequest request;
    return StateHelper::make_initial_physical_warm_request(state,1,request) && initializer->make_attempt()->request_physical_warmstart(request);
  }
};

VioManagerOptions estimated_options(bool full=true) {
  auto p=options(); p.state_options.do_calib_camera_timeoffset=true;
  p.state_options.do_calib_camera_pose=full; p.state_options.do_calib_camera_intrinsics=full;
  return p;
}

void conditional_continuation(EstimatedManager &manager,int calibration_dimension) {
  size_t processed=0,dropped=0;
  std::map<std::pair<size_t,uint64_t>,double> accepted_endpoints;
  manager.set_camera_processed_callback([&](const CameraData &message,bool accepted) {
    processed+=accepted ? message.sensor_ids.size() : 0;
    dropped+=accepted ? 0 : message.sensor_ids.size();
    if(accepted) for(int camera:message.sensor_ids) {
      const auto state=manager.get_state();
      const auto pose=state->find_pose(camera,message.timestamp);
      const auto owner=std::find_if(state->_exposure_poses.begin(),state->_exposure_poses.end(),
          [&](const State::ExposurePose &view) { return view.pose==pose; });
      check(pose && owner!=state->_exposure_poses.end() && owner->imu_time<=state->imu_endpoint(),
            "processed callback resolves the exact queued camera/raw key to its accepted physical owner");
      if(owner!=state->_exposure_poses.end())
        accepted_endpoints[{size_t(camera),initializer_time_bits(message.timestamp)}]=owner->imu_time;
    }
    return true;
  });
  int next_imu=1937;
  for(int frame=0;frame<10;++frame) {
    const int tick=2000+25*frame;
    const auto state=manager.get_state();
    const double nominal_endpoint=10.+tick/800.;
    // These are the exact keys queue_group publishes. An EKF clock correction
    // after that publication must not be used to reconstruct a historical key.
    const std::array<double,2> raw{{nominal_endpoint-state->cam_imu_dt(0),nominal_endpoint-state->cam_imu_dt(1)}};
    manager.queue_group(tick,75+frame);
    manager.feed_measurement_batch_imu(samples(next_imu,tick+10)); next_imu=tick+11;
    const auto a=state->find_pose(0,raw[0]),b=state->find_pose(1,raw[1]);
    bool retained=a && b && a!=b;
    for(size_t camera=0;camera<2;++camera) {
      const auto accepted=accepted_endpoints.find({camera,initializer_time_bits(raw[camera])});
      const auto owner=std::find_if(state->_exposure_poses.begin(),state->_exposure_poses.end(),
          [&](const State::ExposurePose &view) { return view.camera_id==camera &&
            initializer_time_bits(view.raw_time)==initializer_time_bits(raw[camera]); });
      retained&=accepted!=accepted_endpoints.end() && owner!=state->_exposure_poses.end() &&
          owner->imu_time==accepted->second;
    }
    check(retained,"online calibration updates preserve distinct owners and their accepted endpoint/key identities");
    const auto P=StateHelper::get_full_covariance(state);
    check(state->clone_count()<=static_cast<size_t>(state->_options.max_pose_clones()) &&
          P.rows()==15+calibration_dimension+6*static_cast<int>(state->clone_count()),
          "conditional continuation retains calibration dimensions in a bounded owner window");
    check(StateHelper::valid_initial_covariance(P,true),
          "ordinary propagation, image updates and marginalization preserve a PSD conditional joint state");
  }
  manager.finish_observation_replay();
  check(processed==20 && dropped==0 && accepted_endpoints.size()==20 && manager.initialized(),
        "all conditional post-import observations complete through the public queued manager caller");
  manager.set_camera_processed_callback({});
}

void invalid_imports(const std::shared_ptr<State> &fresh,const InitPhysicalWarmResult &result) {
  for(int bad=0;bad<14;++bad) {
    auto state=StateHelper::clone_state(fresh); auto candidate=result;
    switch(bad) {
      case 0: candidate.calibration_covariance(0,0)*=1.1; break;
      case 1: candidate.joint_covariance(0,20)=std::numeric_limits<double>::quiet_NaN(); break;
      case 2: candidate.joint_covariance(12,12)=-1.; break;
      case 3: candidate.consider.front().mean(0)+=.01; break;
      case 4: candidate.consider.front().fej(0)+=.01; break;
      case 5: candidate.consider.front()=candidate.consider.back(); break;
      case 6: candidate.consider.pop_back(); break;
      case 7: candidate.pose_error_scale*=2.; break;
      case 8: { const int row=15+6*(candidate.owners.size()-1); candidate.joint_covariance(row,row)+=.01; break; }
      case 9: candidate.joint_covariance.bottomRightCorner(1,1)(0,0)+=.001; break;
      case 10: candidate.owners.back().raw_time=std::nextafter(candidate.owners.back().raw_time,0.); break;
      case 11: candidate.episode_id=2; break;
      case 12: {
        auto type=calibration_type(state,candidate.consider.front()); auto fej=type->fej(); fej(0)+=.001; type->set_fej(fej); break;
      }
      case 13: {
        auto P=StateHelper::get_full_covariance(state); P(0,15)=P(15,0)=1e-10;
        auto types=calibration_types(state); types.insert(types.begin(),state->_imu);
        StateHelper::set_initial_covariance(state,P,types); break;
      }
    }
    const auto P=StateHelper::get_full_covariance(state); const auto mean=state->_imu->value(),fej=state->_imu->fej();
    const auto imu=state->_imu; const auto clock=state->cam_imu_dt_var(0);
    check(!StateHelper::set_initial_state_physical_warm(state,candidate,1),"invalid or stale conditional import rejects");
    check(state->_imu==imu && state->cam_imu_dt_var(0)==clock && same(P,StateHelper::get_full_covariance(state)) &&
          same(mean,state->_imu->value()) && same(fej,state->_imu->fej()) && !state->_imu_endpoint_valid &&
          state->_initialization_episode_id==0 && state->clone_count()==0,
          "conditional rejection is atomic for handles, means, FEJ, covariance, endpoint and owners");
  }
}

void actual_conditional(bool singular,bool full) {
  EstimatedManager manager(estimated_options(full)); manager.prime(); manager.set_prior(singular);
  auto state=manager.get_state();
  const auto fresh=StateHelper::clone_state(state);
  const auto types=calibration_types(state);
  InitPhysicalWarmRequest initial_prior;
  check(StateHelper::make_initial_physical_warm_request(state,1,initial_prior),"fresh complete calibration prior can be captured");
  std::vector<Eigen::VectorXd> means,fejs;
  for(const auto &type:types) { means.push_back(type->value()); fejs.push_back(type->fej()); }
  InitPhysicalWarmResult result;
  const bool solved=manager.solve_reference(result);
  check(solved,"actual dynamic initializer exports the complete permuted camera conditional posterior");
  if(!solved) return;
  const int navigation=15+6*result.owners.size(),k=result.calibration_covariance.rows();
  check(k==(full?30:2) && result.joint_covariance.rows()==navigation+k &&
        result.owners.size()==size_t(state->_options.max_pose_clones()) && result.accepted_imu_endpoint==12.4,
        "real conditional output is bounded and retains exact fixed graph endpoint");
  check(result.joint_covariance.topRightCorner(15,k).norm()>1e-8,
        "real navigation/calibration sensitivity creates nonzero retained cross covariance");
  auto direct=StateHelper::clone_state(state);
  check(StateHelper::set_initial_state_physical_warm(direct,result,1),"permuted semantic calibration order imports into existing Type order");
  check(same(canonical_covariance(direct,result),result.joint_covariance),"canonical joint covariance is scattered exactly once without added Q or jitter");
  const auto expected=result.joint_covariance;
  bool found=false;
  std::map<size_t,int> clock_columns;int offset=0;
  for(const auto &block:result.consider) { if(block.kind==InitCameraCalibrationKind::Clock) clock_columns[block.camera_id]=offset; offset+=block.local_size(); }
  for(size_t i=1;i<result.owners.size();++i) {
    const auto &a=result.owners[i-1],&b=result.owners[i];
    if(a.nominal_imu_time!=b.nominal_imu_time) continue;
    found=true;
    Eigen::MatrixXd D=Eigen::MatrixXd::Zero(6,k);
    Eigen::Matrix<double,6,1> rate; rate<<a.omega_body,a.velocity_world;
    rate=result.pose_error_scale.asDiagonal()*rate;
    D.col(clock_columns.at(a.camera_id))=rate; D.col(clock_columns.at(b.camera_id))=-rate;
    const int ra=15+6*(i-1),rb=15+6*i;
    const Eigen::MatrixXd difference=expected.block(ra,ra,6,6)+expected.block(rb,rb,6,6)-expected.block(ra,rb,6,6)-expected.block(rb,ra,6,6);
    check(near(difference,D*result.calibration_covariance*D.transpose(),1e-9) &&
          (singular ? difference.norm()<1e-9 : difference.norm()>1e-9),
          "actual coincident owners have exactly their independent or common-only clock prior support");
    check(direct->find_pose(a.camera_id,a.raw_time)!=direct->find_pose(b.camera_id,b.raw_time),"coincident stochastic owners retain different mutable handles");
  }
  check(found,"actual solver graph contains coincident endpoints from unequal raw clocks");
  if(!singular) {
    invalid_imports(fresh,result);
    InitPhysicalWarmResult bounded;
    const bool smaller=manager.solve_reference(bounded,true,3);
    check(smaller && bounded.owners.size()==3,"conditional export applies the caller owner cap before its final covariance pass");
    if(smaller) {
      std::vector<int> indices;
      for(int i=0;i<15;++i)indices.push_back(i);
      for(int i=navigation-18;i<navigation;++i)indices.push_back(i);
      for(int i=navigation;i<navigation+k;++i)indices.push_back(i);
      Eigen::MatrixXd selected(indices.size(),indices.size());
      for(size_t r=0;r<indices.size();++r)for(size_t c=0;c<indices.size();++c)selected(r,c)=result.joint_covariance(indices[r],indices[c]);
      check(near(bounded.joint_covariance,selected,2e-7),"bounded output is the exact marginal row selection of the complete conditional covariance");
    }
  }
  check(!manager.attempt(),"real manager launches a detached conditional initialization on fresh-boot opt-in");
  const auto preserved=append_arrivals(manager,result); const auto before=rows(manager.database());
  const auto before_cov=StateHelper::get_full_covariance(state); const auto before_mean=state->_imu->value();
  manager.finish_worker_only();
  check(same(before_cov,StateHelper::get_full_covariance(state)) && same(before_mean,state->_imu->value()) && rows(manager.database())==before,
        "completed conditional worker cannot mutate live prior, navigation or image ownership");
  check(manager.attempt() && manager.success(),"actual manager accepts the estimated-camera physical warm result");
  if(!manager.success()) return;
  check(near(canonical_covariance(state,result),result.joint_covariance,2e-7),"actual manager installs complete conditional navigation/calibration covariance");
  for(size_t i=0;i<types.size();++i) check(calibration_type(state,initial_prior.consider[i])==types[i] &&
      same(types[i]->value(),means[i]) && same(types[i]->fej(),fejs[i]),"estimated calibration handles, means and FEJ are retained exactly");
  const auto remaining=rows(manager.database()); bool consumed=true,retained=true;
  for(const auto &row:result.consumed_observations) consumed&=remaining.count(key(row))==0;
  for(const auto &row:preserved) retained&=remaining.count(row)==1;
  check(consumed && retained,"successful conditional import consumes exact likelihood rows once and retains unused/future arrivals");
  const auto after=canonical_covariance(state,result);
  check(same(after.bottomRightCorner(k,k).eval(),result.calibration_covariance),"complete calibration prior including correlations survives actual manager handoff");
  const double accepted=state->imu_endpoint(); auto clock=state->cam_imu_dt_var(0)->value();clock(0)+=1e-6;
  state->cam_imu_dt_var(0)->set_value(clock);
  check(state->imu_endpoint()==accepted,"a later online clock mean cannot relabel the accepted fixed IMU endpoint");
  conditional_continuation(manager,k);
  std::printf("ESTIMATED_CASE singular=%d full=%d owners=%zu calibration=%d\n",singular,full,result.owners.size(),k);
}

void stale_manager_prior() {
  EstimatedManager manager(estimated_options()); manager.prime(); manager.set_prior();
  auto state=manager.get_state();
  check(!manager.attempt(),"stale-prior control launches a real conditional worker");
  manager.finish_worker_only(); manager.set_prior(false,1.01);
  const auto P=StateHelper::get_full_covariance(state);const auto mean=state->_imu->value();const auto before=rows(manager.database());
  check(!manager.attempt() && !manager.success() && !manager.pending(),"stale calibration prior rejects the actual consumer handoff without cold zero-cross fallback");
  check(same(P,StateHelper::get_full_covariance(state)) && same(mean,state->_imu->value()) && rows(manager.database())==before &&
        !state->_imu_endpoint_valid && state->clone_count()==0 && state->_initialization_episode_id==0,
        "actual stale-prior rejection retains the entire live state and exact unconsumed rows");
  check(!manager.attempt(),"rejected conditional attempt can retry from a new complete prior snapshot");
  manager.finish_worker_only();
  check(manager.attempt() && manager.success(),"fresh retry accepts without replaying a previously imported likelihood");
}

void unsupported_contracts() {
  auto p=estimated_options();p.init_options.init_dyn_mle_opt_calib=true;
  EstimatedManager fitted(p);check(!fitted.can_request(),"fitted calibration means remain outside the conditional contract");
  EstimatedManager reset(estimated_options());reset.arm_prior();check(!reset.can_request(),"marginal-only reset prior cannot pretend to preserve a joint calibration prior");
  auto state=reset.get_state(); InitPhysicalWarmRequest request;request.episode_id=77;
  for(int unsupported=0;unsupported<3;++unsupported) {
    auto candidate=StateHelper::clone_state(state);
    if(unsupported==0)candidate->_options.do_calib_imu_intrinsics=true;
    if(unsupported==1)candidate->_options.do_calib_imu_g_sensitivity=true;
    if(unsupported==2)candidate->_options.do_calib_camera_readout=true;
    check(!StateHelper::make_initial_physical_warm_request(candidate,1,request) && request.episode_id==77,
          "unsupported IMU/readout uncertainty rejects without publishing a partial request");
  }
}
} // namespace

int main() {
  ov_core::Printer::setPrintLevel(ov_core::Printer::WARNING);
  try { actual_conditional(false,true); actual_conditional(true,false); stale_manager_prior(); unsupported_contracts(); }
  catch(const std::exception &error) { check(false,error.what()); }
  std::printf("ESTIMATED_WARMSTART %s checks=%d failures=%d\n",failures?"FAIL":"PASS",checks,failures);
  return failures?1:0;
}
