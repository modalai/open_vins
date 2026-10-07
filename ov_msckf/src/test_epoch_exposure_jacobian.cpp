/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "cam/CamRadtan.h"
#include "state/StateHelper.h"
#include "update/UpdaterHelper.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include <cstdio>

using namespace ov_msckf;
using namespace ov_core;
using namespace ov_type;
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;

// Independent double-precision projection of the declared deterministic
// transport model. Observed row is held fixed; no production Jacobian or
// production projection is differentiated.
static Eigen::Vector2d predict(const std::shared_ptr<State> &s,
                               const UpdaterHelper::UpdaterHelperFeature &f, bool fej=false) {
  const auto pose = s->pose_for_camera(1,10.);
  const auto &kin = *s->clone_kinematics(1,10.);
  M3 R = fej ? pose->Rot_fej() : pose->Rot();
  V3 p = fej ? pose->pos_fej() : pose->pos();
  V3 v = s->velocity_for_camera(1,10.,fej), w = fej ? kin.omega_fej : kin.omega;
  double tau = (s->uses_physical_clones() ? 0. :
      (fej ? s->cam_imu_dt_var(1)->fej()(0)-s->cam_imu_dt_var(0)->fej()(0) : s->cam_imu_dt_delta(1))) +
      (double(f.uvs.at(1).at(0)(1))/480. - .5) *
      (fej ? s->_calib_camera_readout.at(1)->fej()(0) : s->_calib_camera_readout.at(1)->value()(0));
  R = exp_so3(-w*tau)*R;
  p += v*tau;
  const auto cal = s->_calib_IMUtoCAM.at(1);
  const V3 z = cal->Rot()*R*((fej ? f.p_FinG_fej : f.p_FinG)-p)+cal->pos();
  // Deliberately independent pinhole projection; the fixture has zero distortion.
  return {400.*z.x()/z.z()+320., 410.*z.y()/z.z()+240.};
}

int main() {
  Printer::setPrintLevel("ERROR");
  int checks=0, failures=0;
  double max_pose=0., max_bias=0., max_clock=0., max_gauge=0.;
  for (bool physical : {false,true}) for (bool fej : {false,true}) for (bool velocity_owner : {false,true}) for (bool biased : {false,true})
    for (double readout : {0.012,0.016})
    for (double offset : {-0.012,0.006}) for (double row : {24.,312.,456.}) {
      StateOptions options; options.num_cameras=2; options.do_fej=fej;
      if(velocity_owner!=physical) continue;
      options.stochastic_epoch_transport=physical;
      options.do_calib_camera_timeoffset=true; options.do_calib_camera_readout=true;
      auto state=std::make_shared<State>(options);
      for (int c=0;c<2;++c) {
        auto camera=std::make_shared<CamRadtan>(640,480);
        Eigen::Matrix<double,8,1> k; k << 400.,410.,320.,240.,0.,0.,0.,0.;
        camera->set_value(k); state->_cam_intrinsics_cameras[c]=camera;
        state->_cam_intrinsics[c]->set_value(k); state->_cam_intrinsics[c]->set_fej(k);
        Eigen::Matrix<double,7,1> extrinsic;
        extrinsic << rot_2_quat(exp_so3(V3(.03,-.04,.015))), .08,.01,-.02;
        state->_calib_IMUtoCAM[c]->set_value(extrinsic);
        state->_calib_IMUtoCAM[c]->set_fej(extrinsic);
        Eigen::VectorXd clock(1); clock << (c ? offset : 0.);
        state->cam_imu_dt_var(c)->set_value(clock); state->cam_imu_dt_var(c)->set_fej(clock);
        Eigen::VectorXd rd(1); rd << (c ? readout : 0.);
        state->_calib_camera_readout[c]->set_value(rd); state->_calib_camera_readout[c]->set_fej(rd);
      }
      Eigen::Matrix<double,16,1> imu=state->_imu->value();
      imu.head<4>()=rot_2_quat(exp_so3(V3(.11,-.06,.17)));
      imu.segment<3>(4)=V3(.2,-.15,.12); imu.segment<3>(7)=V3(1.3,-.7,.2);
      state->_imu->set_value(imu); state->_imu->set_fej(imu); state->_timestamp=10.;
      const V3 omega(.7,-.4,.9);
      if(physical) {
        State::ExposurePose owner; owner.camera_id=1; owner.raw_time=10.; owner.imu_time=10.+offset;
        const auto motion=StateHelper::augment_motion_view(state,1,omega,V3(.2,-.1,.05));
        owner.pose=motion.first; owner.velocity=motion.second;
        owner.kinematics.omega=owner.kinematics.omega_fej=omega;
        owner.kinematics.vel=owner.kinematics.vel_fej=state->_imu->vel();
        state->append_exposure_pose(owner);
      } else StateHelper::augment_clone(state,omega,omega);
      if(biased) {
        Eigen::VectorXd change=Eigen::VectorXd::Zero(15);
        change.segment<3>(9)<<.03,-.02,.01;
        change.segment<3>(12)<<.05,-.03,.02;
        state->_imu->update(change);
      }
      if(fej) {
        Eigen::VectorXd change(6); change<<.004,-.002,.003,.006,-.004,.002;
        state->pose_for_camera(1,10.)->update(change);
      }
      UpdaterHelper::UpdaterHelperFeature f; f.featid=5;
      f.feat_representation=LandmarkRepresentation::GLOBAL_3D;
      f.p_FinG=f.p_FinG_fej=V3(1.3,.7,5.4);
      f.timestamps[1]={10.}; f.uvs[1]={Eigen::Vector2f(400.,float(row))};
      f.uvs_norm[1]={Eigen::Vector2f(.2,.1)};
      Eigen::MatrixXd Hf,Hx; Eigen::VectorXd residual;
      std::vector<std::shared_ptr<Type>> order;
      UpdaterHelper::get_feature_jacobian_full(state,f,Hf,Hx,residual,order);
      const auto expected=predict(state,f);
      ++checks;
      if (!ov_core::numeric::finite_matrix(residual) ||
          (residual-(f.uvs.at(1).at(0).cast<double>()-expected)).norm()>1e-9) ++failures;
      const double step=1e-6;
      int col=0;
      for (const auto &variable:order) {
        for(int axis=0;axis<variable->size();++axis,++col) {
          auto plus=StateHelper::clone_state(state), minus=StateHelper::clone_state(state);
          const auto perturb=[&](const std::shared_ptr<State> &copy,double amount) {
            const auto pose=copy->pose_for_camera(1,10.);
            std::shared_ptr<Type> target;
            if (variable==state->pose_for_camera(1,10.)) target=pose;
            else if(velocity_owner && variable==state->clone_velocity(1,10.)) target=copy->clone_velocity(1,10.);
            else if (variable==state->_imu->bg()) target=copy->_imu->bg();
            else if (variable==state->_imu->ba()) target=copy->_imu->ba();
            else if (variable==state->_calib_camera_readout.at(1)) target=copy->_calib_camera_readout.at(1);
            else if (variable==state->cam_imu_dt_var(1)) target=copy->cam_imu_dt_var(1);
            else if (variable==state->cam_imu_dt_var(0)) target=copy->cam_imu_dt_var(0);
            else std::abort();
            Eigen::VectorXd delta=Eigen::VectorXd::Zero(target->size()); delta(axis)=amount;
            if(fej) target->set_value(target->fej());
            target->update(delta);
            if(fej) target->set_fej(target->value());
          };
          perturb(plus,step); perturb(minus,-step);
          const Eigen::Vector2d fd=(predict(plus,f,fej)-predict(minus,f,fej))/(2.*step);
          const double error=(fd-Hx.col(col)).cwiseAbs().maxCoeff();
          if(variable==state->pose_for_camera(1,10.)) max_pose=std::max(max_pose,error);
          else if(variable==state->_imu->bg()||variable==state->_imu->ba()) max_bias=std::max(max_bias,error);
          else max_clock=std::max(max_clock,error);
          ++checks;
          if(!ov_core::numeric::finite(error) || error>2e-5) {
            if(failures<12) std::printf("FD_FAIL fej=%d velocity=%d biased=%d rd=%.3f dt=%.3f row=%.0f type_id=%d axis=%d error=%.9g\n",fej,velocity_owner,biased,readout,offset,row,variable->id(),axis,error);
            ++failures;
          }
        }
      }
      if(velocity_owner) {
        Eigen::MatrixXd gauge=Eigen::MatrixXd::Zero(Hx.cols(),4);
        int start=0;
        const auto pose=state->pose_for_camera(1,10.);
        const V3 g(0.,0.,1.);
        for(const auto &variable:order) {
          if(variable==pose) {
            gauge.block<3,3>(start+3,0).setIdentity();
            gauge.block<3,1>(start,3)=(fej ? pose->Rot_fej() : pose->Rot())*g;
            gauge.block<3,1>(start+3,3)=-skew_x(fej ? pose->pos_fej() : pose->pos())*g;
          } else if(variable==state->clone_velocity(1,10.))
            gauge.block<3,1>(start,3)=-skew_x(state->velocity_for_camera(1,10.,fej))*g;
          start+=variable->size();
        }
        Eigen::Matrix<double,3,4> feature_gauge;
        feature_gauge.leftCols<3>().setIdentity();
        feature_gauge.rightCols<1>()=-skew_x(fej ? f.p_FinG_fej : f.p_FinG)*g;
        const double error=(Hx*gauge+Hf*feature_gauge).cwiseAbs().maxCoeff();
        max_gauge=std::max(max_gauge,error); ++checks;
        if(!ov_core::numeric::finite(error) || error>1e-9) ++failures;
      }
    }
  std::printf("EPOCH_EXPOSURE_JACOBIAN checks=%d failures=%d max_pose=%.9g max_bias=%.9g max_clock=%.9g max_gauge=%.9g\n",checks,failures,max_pose,max_bias,max_clock,max_gauge);
  return failures ? 1 : 0;
}
