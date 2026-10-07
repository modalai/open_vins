/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "state/StateHelper.h"
#include "utils/print.h"
#include <cstdio>
#include <random>

using namespace ov_msckf;
using namespace ov_type;
using M = Eigen::MatrixXd;
using V = Eigen::VectorXd;

// Many features see the SAME transport increment. They cannot average that
// increment down as independent pixel noise. Score the old six-DOF pose after
// observing the transported pose, including an omission negative control.
static bool shared_transport_nees(double &anees, double &omitted_anees) {
  StateOptions options; options.num_cameras=2; options.stochastic_epoch_transport=true;
  auto state=std::make_shared<State>(options);
  constexpr double prior_variance=.005, transport_variance=.02, pixel_variance=.0025;
  StateHelper::set_initial_covariance(state,prior_variance*M::Identity(15,15),{state->_imu});
  const auto old_pose=StateHelper::augment_pose_view(state,0,Eigen::Vector3d::Zero());
  M process=.0001*M::Identity(15,15);
  process.topLeftCorner(6,6)=transport_variance*M::Identity(6,6);
  if(!StateHelper::EKFPropagation(state,{state->_imu},{state->_imu},M::Identity(15,15),process)) return false;
  const auto transported_pose=StateHelper::augment_pose_view(state,1,Eigen::Vector3d::Zero());
  M pixels=M::Zero(24,6);
  for(int i=0;i<24;++i) pixels(i,i%6)=1.+.1*(i/6);
  const M noise=pixel_variance*M::Identity(24,24);
  const M prior=StateHelper::get_full_covariance(state);
  M H=M::Zero(24,prior.rows()); H.middleCols(transported_pose->id(),6)=pixels;
  const M cross=prior*H.transpose();
  const M innovation=H*cross+noise;
  const M gain=innovation.ldlt().solve(cross.transpose()).transpose();
  const M expected=(prior-gain*cross.transpose()).eval();
  if(!StateHelper::EKFUpdate(state,{transported_pose},pixels,V::Zero(24),noise)) return false;
  const M posterior=StateHelper::get_full_covariance(state);
  if(!ov_core::numeric::finite_matrix(posterior) || (posterior-expected).cwiseAbs().maxCoeff()>2e-11) return false;
  const M kept=posterior.block(old_pose->id(),old_pose->id(),6,6);
  const M old_gain=gain.middleRows(old_pose->id(),6);
  const M old_prior=prior_variance*M::Identity(6,6);
  const M omitted_cross=old_prior*pixels.transpose();
  const M omitted_gain=(pixels*omitted_cross+noise).ldlt().solve(omitted_cross.transpose()).transpose();
  const M omitted_posterior=old_prior-omitted_gain*omitted_cross.transpose();
  const auto correct_factor=kept.ldlt();
  const auto omitted_factor=omitted_posterior.ldlt();
  std::mt19937_64 random(916733);
  std::normal_distribution<double> gaussian;
  anees=omitted_anees=0.;
  constexpr int replicates=5000;
  for(int i=0;i<replicates;++i) {
    V old_error(6), increment(6), measurement_noise(24);
    for(int j=0;j<6;++j) {
      old_error(j)=std::sqrt(prior_variance)*gaussian(random);
      increment(j)=std::sqrt(transport_variance)*gaussian(random);
    }
    for(int j=0;j<24;++j) measurement_noise(j)=std::sqrt(pixel_variance)*gaussian(random);
    const V measurement=pixels*(old_error+increment)+measurement_noise;
    const V error=old_error-old_gain*measurement;
    const V omitted_error=old_error-omitted_gain*measurement;
    anees+=error.dot(correct_factor.solve(error));
    omitted_anees+=omitted_error.dot(omitted_factor.solve(omitted_error));
  }
  anees/=replicates; omitted_anees/=replicates;
  // Four-sigma envelope for 5000 independent six-dimensional Gaussian draws.
  // This is a linear-model ownership test, not nonlinear VINS certification.
  return anees>5.8 && anees<6.2 && omitted_anees>18.;
}

// Independent dense oracle: all clock, historical velocity, pose and process
// cross blocks are kept. Production augmentation/propagation is never used
// to construct the expected covariance.
int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  std::mt19937_64 random(71619);
  std::normal_distribution<double> gaussian;
  int checks=0, failures=0;
  double maximum=0.;
  const auto check=[&](double error, double tolerance=2e-11) {
    ++checks; maximum=std::max(maximum,error);
    if(!ov_core::numeric::finite(error) || error>tolerance) ++failures;
  };
  for(int trial=0;trial<40;++trial) {
    StateOptions options; options.num_cameras=2; options.max_clone_size=1;
    options.do_calib_camera_timeoffset=true; options.stochastic_epoch_transport=true;
    auto state=std::make_shared<State>(options);
    const int n=state->max_covariance_size();
    M root(n,n);
    for(int row=0;row<n;++row) for(int col=0;col<n;++col) root(row,col)=.03*gaussian(random);
    M expected=root*root.transpose()+.01*M::Identity(n,n);
    StateHelper::set_initial_covariance(state,expected,{state->_imu,state->cam_imu_dt_var(0),state->cam_imu_dt_var(1)});
    Eigen::Matrix<double,16,1> value=state->_imu->value();
    value.segment<3>(7)<<1.2,-.4,.3;
    state->_imu->set_value(value); state->_imu->set_fej(value);
    const Eigen::Vector3d omega(.4,-.2,.7), acceleration(.8,.3,-.2);
    for(int epoch=0;epoch<2;++epoch) {
      if(epoch) {
        M transition=M::Identity(15,15);
        transition.block<3,3>(3,6)=.017*Eigen::Matrix3d::Identity();
        transition.block<3,3>(0,9)=-.017*Eigen::Matrix3d::Identity();
        transition.block<3,3>(6,12)=-.017*Eigen::Matrix3d::Identity();
        M noise=.0001*M::Identity(15,15);
        M dense=M::Identity(expected.rows(),expected.cols());
        dense.topLeftCorner(15,15)=transition;
        expected=(dense*expected*dense.transpose()).eval();
        expected.topLeftCorner(15,15)+=noise;
        if(!StateHelper::EKFPropagation(state,{state->_imu},{state->_imu},transition,noise)) ++failures;
      }
      const int old=expected.rows();
      const int size=epoch ? 6 : 9;
      M augmentation=M::Zero(old+size,old);
      augmentation.topRows(old).setIdentity();
      augmentation.block(old,0,6,6).setIdentity();
      if(!epoch) augmentation.block(old+6,6,3,3).setIdentity();
      const int clock=state->cam_imu_dt_var(epoch)->id();
      augmentation.block<3,1>(old,clock)=omega;
      augmentation.block<3,1>(old+3,clock)=state->_imu->vel();
      if(!epoch) augmentation.block<3,1>(old+6,clock)=acceleration;
      expected=(augmentation*expected*augmentation.transpose()).eval();
      state->_timestamp=1.+epoch;
      State::ExposurePose owner;
      owner.camera_id=epoch; owner.raw_time=1.+epoch; owner.imu_time=1.+epoch;
      if(!epoch) {
        const auto motion=StateHelper::augment_motion_view(state,epoch,omega,acceleration);
        owner.pose=motion.first; owner.velocity=motion.second;
      } else owner.pose=StateHelper::augment_pose_view(state,epoch,omega);
      owner.kinematics.omega=owner.kinematics.omega_fej=omega;
      owner.kinematics.vel=owner.kinematics.vel_fej=state->_imu->vel();
      state->append_exposure_pose(owner);
      check((StateHelper::get_full_covariance(state)-expected).cwiseAbs().maxCoeff());
    }
    // A late factor on the old velocity must correct the current navigation
    // state and every correlated pose. Compare the complete dense posterior.
    const auto old_velocity=state->clone_velocity(0,1.);
    M H=M::Zero(2,expected.rows());
    Eigen::Matrix<double,2,3> local; local<<.3,-.2,.4,-.1,.6,.2;
    H.block(0,old_velocity->id(),2,3)=local;
    M noise=.04*M::Identity(2,2);
    const M cross=expected*H.transpose();
    const M innovation=H*cross+noise;
    expected=(expected-cross*innovation.ldlt().solve(cross.transpose())).eval();
    if(!StateHelper::EKFUpdate(state,{old_velocity},local,V::Zero(2),noise)) ++failures;
    check((StateHelper::get_full_covariance(state)-expected).cwiseAbs().maxCoeff());
    auto snapshot=StateHelper::clone_state(state);
    check((StateHelper::get_full_covariance(snapshot)-expected).cwiseAbs().maxCoeff());
    if(snapshot->clone_velocity(0,1.)==old_velocity || snapshot->clone_velocity(0,1.)->id()!=old_velocity->id()) ++failures;
    V change=V::Ones(3)*.2;
    snapshot->clone_velocity(0,1.)->update(change);
    check((old_velocity->value()-state->_imu->vel()).norm());
    // Remove the old pose and velocity together. This must be a true marginal
    // (submatrix), and the snapshot must keep its independent live owners.
    std::vector<int> retained;
    const int old_pose=state->pose_for_camera(0,1.)->id();
    const int old_v=old_velocity->id();
    for(int i=0;i<expected.rows();++i)
      if(!(i>=old_pose && i<old_pose+6) && !(i>=old_v && i<old_v+3)) retained.push_back(i);
    M marginal(retained.size(),retained.size());
    for(size_t i=0;i<retained.size();++i) for(size_t j=0;j<retained.size();++j)
      marginal(i,j)=expected(retained[i],retained[j]);
    StateHelper::marginalize_old_clone(state);
    check((StateHelper::get_full_covariance(state)-marginal).cwiseAbs().maxCoeff());
    if(state->find_pose(0,1.) || !state->find_pose(1,2.) || !snapshot->clone_velocity(0,1.)) ++failures;
  }
  StateOptions window; window.num_cameras=2; window.max_clone_size=11;
  window.stochastic_epoch_transport=true;
  if(!window.configure_clone_policy(false,false) || !window.configure_epoch_window({{0,0.},{1,0.}}) ||
      window.max_pose_clones()!=22 || !window.configure_epoch_window({{0,60.},{1,30.}}) ||
      window.max_pose_clones()!=33 || window.max_clone_size!=11) ++failures;
  if(window.configure_epoch_window({{0,600.},{1,1.}}) ||
      window.configure_epoch_window({{0,60.},{1,std::numeric_limits<double>::quiet_NaN()}})) ++failures;
  window.max_epoch_clones=21;
  if(window.configure_epoch_window({{0,60.}}) || window.configure_epoch_window({{1,-30.}})) ++failures;
  double anees=0., omitted_anees=0.;
  if(!shared_transport_nees(anees,omitted_anees)) ++failures;
  std::printf("EPOCH_VELOCITY_OWNER checks=%d failures=%d dense_max=%.9g shared_pose_anees=%.9g omission_anees=%.9g\n",
              checks,failures,maximum,anees,omitted_anees);
  return failures ? 1 : 0;
}
