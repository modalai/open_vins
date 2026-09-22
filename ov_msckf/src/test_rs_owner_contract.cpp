/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "cam/CamRadtan.h"
#include "core/VioManager.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterHelper.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ov_core;
using namespace ov_msckf;
namespace {
int checks = 0, failures = 0;
void check(bool good, const char *name) {
  ++checks;
  if (!good) { ++failures; std::printf("FAIL: %s\n", name); }
}
}

int main() {
  Printer::setPrintLevel("ERROR");
  StateOptions options;
  options.num_cameras = 2;
  options.physical_camera_clones = true;
  options.do_fej = false;
  options.max_aruco_features = 0;
  check(options.configure_clone_policy(false, false), "physical owner fixture configures");
  auto state = std::make_shared<State>(options);
  const Eigen::Vector3d omega(.8, -.6, .3), velocity(.7, -.1, .05), position(.3, -.2, .1);
  const Eigen::Matrix3d rotation = exp_so3(Eigen::Vector3d(.2, -.1, .05));
  Eigen::Matrix<double, 16, 1> imu = state->_imu->value();
  imu.head<4>() = rot_2_quat(rotation); imu.segment<3>(4) = position; imu.segment<3>(7) = velocity;
  state->_imu->set_value(imu); state->_imu->set_fej(imu);
  Eigen::VectorXd intrinsics(8); intrinsics << 400, 405, 320, 240, 0, 0, 0, 0;
  Eigen::Matrix<double, 7, 1> extrinsic; extrinsic << 0, 0, 0, 1, 0, 0, 0;
  for (int camera = 0; camera < 2; ++camera) {
    auto model = std::make_shared<CamRadtan>(640, 480); model->set_value(intrinsics);
    state->_cam_intrinsics_cameras[camera] = model;
    state->_cam_intrinsics[camera]->set_value(intrinsics); state->_cam_intrinsics[camera]->set_fej(intrinsics);
    state->_calib_IMUtoCAM[camera]->set_value(extrinsic); state->_calib_IMUtoCAM[camera]->set_fej(extrinsic);
  }
  State::ExposurePose owner;
  owner.camera_id = 1; owner.raw_time = .02; owner.imu_time = .018;
  owner.pose = StateHelper::augment_pose_view(state, 1, omega);
  owner.kinematics.omega = owner.kinematics.omega_fej = omega;
  owner.kinematics.vel = owner.kinematics.vel_fej = velocity;
  state->_exposure_poses.push_back(owner);
  Eigen::VectorXd readout(1); readout << .016;
  state->_calib_camera_readout[1]->set_value(readout); state->_calib_camera_readout[1]->set_fej(readout);

  double maximum_unsupported_error = 0.;
  for (double row : {40., 240., 420.}) {
    const double delta = (row / 480. - .5) * readout(0);
    const Eigen::Matrix3d row_rotation = exp_so3(-omega * delta) * rotation;
    const Eigen::Vector3d row_position = position + velocity * delta;
    const Eigen::Vector3d ray((330.-320.)/400., (row-240.)/405., 1.);
    const Eigen::Vector3d point = row_position + row_rotation.transpose() * (4.8 * ray);
    UpdaterHelper::UpdaterHelperFeature feature;
    feature.featid = 1; feature.feat_representation = ov_type::LandmarkRepresentation::GLOBAL_3D;
    feature.p_FinG = feature.p_FinG_fej = point;
    feature.timestamps[1] = {owner.raw_time};
    feature.uvs[1] = {Eigen::Vector2f(330., row)};
    feature.uvs_norm[1] = {Eigen::Vector2f(ray.x(), ray.y())};
    Eigen::MatrixXd Hf, Hx; Eigen::VectorXd residual;
    std::vector<std::shared_ptr<ov_type::Type>> order;
    UpdaterHelper::get_feature_jacobian_full(state, feature, Hf, Hx, residual, order);
    if (row == 240.) {
      check(residual.norm() < 1e-8, "GS/reference-row projection remains correct");
    } else {
      maximum_unsupported_error = std::max(maximum_unsupported_error, residual.norm());
      check(residual.norm() > .1, "bypassing the RS guard aliases a distinct row to the frame pose");
    }
    check(order.size() == 1 && order.front() == owner.pose, "existing physical lookup carries only the frame owner");
    Eigen::VectorXd changed(1); changed << .04;
    state->_calib_camera_readout[1]->set_value(changed);
    Eigen::MatrixXd Hf_changed, Hx_changed; Eigen::VectorXd residual_changed;
    std::vector<std::shared_ptr<ov_type::Type>> order_changed;
    UpdaterHelper::get_feature_jacobian_full(state, feature, Hf_changed, Hx_changed, residual_changed, order_changed);
    check((residual-residual_changed).norm() == 0. && (Hx-Hx_changed).norm() == 0.,
          "existing physical visual path intentionally ignores row warps; this is not RS support");
    state->_calib_camera_readout[1]->set_value(readout);
  }

  VioManagerOptions params;
  params.state_options = options;
  params.use_stereo = params.use_gpu = params.use_aruco = params.try_zupt = false;
  params.num_opencv_threads = 0;
  params.init_options.num_cameras = options.num_cameras;
  params.init_options.use_stereo = false;
  for (int camera = 0; camera < 2; ++camera) {
    params.camera_intrinsics[camera] = params.init_options.camera_intrinsics[camera] = state->_cam_intrinsics_cameras[camera];
    params.camera_extrinsics[camera] = params.init_options.camera_extrinsics[camera] = extrinsic;
    params.camera_imu_dt[camera] = 0.;
    params.camera_readout_time[camera] = camera == 1 ? readout(0) : 0.;
    params.camera_shutter_rolling[camera] = camera == 1;
  }
  bool rejected = false;
  try { VioManager manager(params); }
  catch (const std::invalid_argument &error) {
    rejected = std::string(error.what()).find("requires global-shutter") != std::string::npos;
  }
  check(rejected, "public manager still rejects unsupported physical rolling shutter");
  std::printf("RS_OWNER_CONTRACT checks=%d failures=%d unsupported_row_error_pixels=%.8g runtime_rs_enabled=0\n",
              checks, failures, maximum_unsupported_error);
  return failures ? 1 : 0;
}
