/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Fixed-camera reset exercises the real graph and consumer, not a mocked solve.
 */
#define OV_RESET_FIXTURE_ONLY
#include "test_reset_conditional_warmstart.cpp"
#undef OV_RESET_FIXTURE_ONLY

namespace {
VioManagerOptions fixed_reset_options() {
  auto p=options();
  p.state_options.do_calib_camera_timeoffset=false;
  p.state_options.do_calib_camera_pose=false;
  p.state_options.do_calib_camera_intrinsics=false;
  return p;
}

void fixed_reset(bool divergence) {
  ResetManager manager(fixed_reset_options());manager.prime();manager.attempt();manager.finish_worker_only();
  check(manager.attempt() && manager.success(),"fixed-camera first boot completes before reset");
  if(!manager.success())return;
  manager.queue_at(12.5004,80);manager.feed_measurement_batch_imu(samples(1937,2011));
  auto old=manager.get_state();
  check(manager.initialized() && std::abs(old->imu_endpoint()-12.5004)<1e-8,"fixed-camera accepted endpoint is between raw IMU records");
  Eigen::Matrix<double,6,6> L=Eigen::Matrix<double,6,6>::Zero();
  L.diagonal()<<.006,.007,.005,.025,.03,.035;
  L(3,0)=.004;L(4,1)=-.003;L(2,0)=.001;
  const Eigen::Matrix<double,6,6> expected=L*L.transpose();
  auto P=StateHelper::get_full_covariance(old);P.middleRows(9,6).setZero();P.middleCols(9,6).setZero();P.block<6,6>(9,9)=expected;
  StateHelper::set_initial_covariance(old,P,all_types(old));
  auto x=old->_imu->value();x.block<6,1>(10,0)<<bg,ba;old->_imu->set_value(x);
  check(StateHelper::valid_initial_covariance(P,true),"independently constructed gyro/accel bias prior and navigation marginal are PSD");
  const double endpoint=old->imu_endpoint();
  manager.soft_reset(divergence ? VioManager::SoftResetCause::DIVERGENCE : VioManager::SoftResetCause::CLIENT);
  const auto prior=manager.prior();
  check(prior && !manager.initialized() && manager.reset_armed(),"fixed camera reset selects the complete six-dimensional prior");
  if(!prior)return;
  auto state=manager.get_state();
  check(prior->consider.empty() && prior->calibration_covariance.rows()==0 && prior->bias_calibration_covariance.rows()==6 &&
        prior->bias_calibration_covariance.cols()==0 && StateHelper::get_full_covariance(state).rows()==15,
        "no estimated camera block or calibration state is invented for fixed-camera reset");
  check(same(expected,prior->bias_covariance) && same(expected,StateHelper::get_full_covariance(state).block<6,6>(9,9).eval()),
        "all six bias correlations survive capture and staging");
  check(prior->imu_endpoint==endpoint && prior->imu_raw_cutoff>endpoint && prior->imu_raw_cutoff<12.502,
        "fixed-camera prior owns physical endpoint and actual right interpolation support");
  const auto old_rows=rows(manager.database());const auto staged=StateHelper::get_full_covariance(state);
  check(!manager.attempt(),"old window starts no accepted reset result");manager.finish_worker_only();
  check(!manager.attempt() && !manager.success() && rows(manager.database())==old_rows && same(staged,StateHelper::get_full_covariance(state)),
        "posterior-conditioned old observations cannot be reused with the retained bias prior");
  const auto snapshot=manager.snapshot();manager.restore(snapshot,{});state=manager.get_state();
  check(manager.prior()==prior && manager.reset_armed(),"fixed-camera pending reset snapshot retains the identical prior and cutoffs");
  manager.future_window();InitPhysicalWarmResult result;
  const bool solved=manager.reference(result);check(solved,"real future-only fixed-camera reset graph solves");
  if(!solved)return;
  check(result.reset_prior==prior && result.consider.empty() && result.calibration_covariance.rows()==0 &&
        result.reset_first_imu_support_time>prior->imu_raw_cutoff && result.reset_first_imu_time>=result.reset_first_imu_support_time,
        "fixed-camera export carries the exact reset identity and future raw support");
  bool fresh=!result.consumed_observations.empty();
  for(const auto &row:result.consumed_observations)fresh&=ov_init::physical_reset_future_row(*prior,row.camera_id,row.raw_time);
  check(fresh,"every consumed fixed-camera observation is future-only in both raw and physical time");
  ov_init::conditional_bias::Conditioned conditioned;
  Eigen::Matrix<double,6,1> floors=Eigen::Matrix<double,6,1>::Zero();
  const double scale=divergence ? fixed_reset_options().init_options.init_dyn_reset_prior_divergence_infl : 1.;
  const double gap=result.reset_first_imu_time-prior->imu_endpoint;
  check(ov_init::conditional_bias::condition(expected,Eigen::MatrixXd(6,0),Eigen::MatrixXd(0,0),gap,
        prior->bias_rw_variance,scale,floors,conditioned),"empty-calibration conditional factor accepts the full six-dimensional bias covariance");
  Eigen::Matrix<double,6,6> analytic=expected;analytic.diagonal()+=gap*prior->bias_rw_variance;analytic*=scale*scale;
  check(near(analytic,conditioned.covariance,1e-14) && conditioned.regression.cols()==0 && conditioned.calibration_rank==0,
        "first-node gap covariance agrees with independent Brownian bias transition");
  Eigen::Matrix<double,6,6> wrong=analytic;
  wrong.diagonal()+=(result.accepted_imu_endpoint-result.reset_first_imu_time)*prior->bias_rw_variance*scale*scale;
  check((wrong-conditioned.covariance).norm()>1e-7,"aging to final node before adding CPI would count graph random walk twice");
  auto diagonal=expected.diagonal().asDiagonal().toDenseMatrix();
  check((diagonal-expected).norm()>1e-5,"discarding gyro/accel cross covariance is a detected incorrect control");
  auto bad=result;bad.reset_prior.reset();const auto untouched=StateHelper::get_full_covariance(state);
  check(!StateHelper::set_initial_state_physical_warm(state,bad,1,prior) && same(untouched,StateHelper::get_full_covariance(state)),
        "missing fixed-camera prior identity rejects import atomically");
  check(!manager.attempt(),"future fixed-camera reset worker launches");manager.finish_worker_only();
  check(manager.attempt() && manager.success() && !manager.prior(),"real consumer imports the full-bias reset and disarms the prior");
  if(!manager.success())return;
  check((state->_imu->Rot()*Eigen::Vector3d::UnitZ()-rotation(5.)*Eigen::Vector3d::UnitZ()).norm()<.02 &&
        (state->_imu->Rot()*state->_imu->vel()-rotation(5.)*velocity(5.)).norm()<.04,
        "reset recovers independent analytic tilt and body velocity in its new gauge");
  size_t processed=0,dropped=0;manager.set_camera_processed_callback([&](const CameraData &m,bool accepted){(accepted?processed:dropped)+=m.sensor_ids.size();return true;});
  int next=4017;
  for(int frame=0;frame<4;++frame) {
    const int tick=4080+40*frame;manager.queue_at(10.+tick/800.,170+frame);manager.feed_measurement_batch_imu(samples(next,tick+10));next=tick+11;
    check(manager.initialized() && state->clone_count()<=size_t(state->_options.max_pose_clones()) &&
          StateHelper::valid_initial_covariance(StateHelper::get_full_covariance(state),true),"fixed-camera continuation retains bounded PSD state");
  }
  manager.finish_observation_replay();check(processed==8 && dropped==0,"all future fixed-camera queued views are processed exactly once");
  manager.set_camera_processed_callback({});
  std::printf("FIXED_RESET_CASE divergence=%d receipts=%zu first=%.9f endpoint=%.9f\n",divergence,result.consumed_observations.size(),result.reset_first_imu_time,result.accepted_imu_endpoint);
}
}
int main() {
  ov_core::Printer::setPrintLevel(ov_core::Printer::WARNING);
  try{fixed_reset(false);fixed_reset(true);}
  catch(const std::exception &error){check(false,error.what());}
  std::printf("RESET_FIXED_WARMSTART %s checks=%d failures=%d\n",failures?"FAIL":"PASS",checks,failures);
  return failures?1:0;
}
