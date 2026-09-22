/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include "cam/CamRadtan.h"
#include "dynamic/DynamicInitializer.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "init/InitializerCameraClock.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace {
using namespace ov_core;
using namespace ov_msckf;
int checks = 0, failures = 0;
void check(bool good, const char *message) {
  ++checks;
  if (!good) { ++failures; std::printf("FAIL: %s\n", message); }
}
template <class A, class B> bool same(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         std::memcmp(a.derived().data(), b.derived().data(), sizeof(double) * a.size()) == 0;
}
InitPhysicalWarmResult fixture() {
  InitPhysicalWarmResult result;
  result.episode_id = 7;
  result.accepted_imu_endpoint = 2.; result.reference_clock_mean = .125; result.reference_clock_label = 1.875;
  result.graph_node_count = 2;
  result.imu_mean.setZero(); result.imu_mean(3) = 1.; result.imu_mean.segment<3>(7) << .3, .2, .1;
  for (size_t camera = 0; camera < 2; ++camera) {
    InitFixedCameraCalibration calibration;
    calibration.camera_id = camera; calibration.clock_mean = camera ? -.25 : .125;
    calibration.width = 640; calibration.height = 480;
    calibration.extrinsics.setZero(); calibration.extrinsics(3) = 1.; calibration.extrinsics(4) = .1 * camera;
    calibration.intrinsics << 400., 405., 320., 240., .01, 0., 0., 0.;
    result.calibration.push_back(calibration);
  }
  for (int i = 0; i < 3; ++i) {
    InitExposureOwner owner;
    owner.camera_id = i == 1 ? 1 : 0;
    owner.nominal_imu_time = i == 2 ? 2. : 1.5; owner.graph_node = i == 2 ? 1 : 0;
    owner.raw_time = owner.nominal_imu_time - result.calibration[owner.camera_id].clock_mean;
    owner.pose_mean = result.imu_mean.head<7>();
    if (i < 2) owner.pose_mean.tail<3>() << -.2, .1, .03;
    owner.velocity_world = i < 2 ? Eigen::Vector3d(.1,-.2,.4) : result.imu_mean.segment<3>(7).eval();
    owner.omega_body = Eigen::Vector3d(.05,.1,-.02);
    result.owners.push_back(owner);
    result.consumed_observations.push_back({42, owner.camera_id, owner.raw_time});
  }
  result.consumed_observations.push_back({42, 0, .375}); // older fitted row with no retained owner
  std::sort(result.consumed_observations.begin(), result.consumed_observations.end());
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(33, 21);
  for (int r = 0; r < 21; ++r)
    for (int c = 0; c < 21; ++c)
      L(r,c) = .001 * std::sin(.4*r + .7*c) + (r == c ? .02 + .001*r : 0.);
  L.middleRows(21,6) = L.middleRows(15,6).eval();
  L.bottomRows(6) = L.topRows(6).eval();
  result.joint_covariance = L * L.transpose();
  return result;
}
std::shared_ptr<State> make_state(const InitPhysicalWarmResult &result, int capacity = 3) {
  StateOptions options; options.physical_camera_clones = true; options.num_cameras = result.calibration.size();
  options.max_clone_size = capacity;
  auto state = std::make_shared<State>(options);
  for (const auto &calibration : result.calibration) {
    const size_t camera = calibration.camera_id;
    auto model = std::make_shared<CamRadtan>(calibration.width, calibration.height);
    model->set_value(calibration.intrinsics); state->_cam_intrinsics_cameras[camera] = model;
    state->_cam_intrinsics.at(camera)->set_value(calibration.intrinsics);
    state->_cam_intrinsics.at(camera)->set_fej(calibration.intrinsics);
    state->_calib_IMUtoCAM.at(camera)->set_value(calibration.extrinsics);
    state->_calib_IMUtoCAM.at(camera)->set_fej(calibration.extrinsics);
    Eigen::Matrix<double,1,1> td; td << calibration.clock_mean;
    state->cam_imu_dt_var(camera)->set_value(td); state->cam_imu_dt_var(camera)->set_fej(td);
  }
  return state;
}
void raw_keys_and_receipts() {
  FeatureDatabase database;
  const double raw = 0x1.fffffffffffecp-1;
  database.update_feature(42, raw, 0, 10, 11, 12, 13);
  database.update_feature(42, raw, 1, 20, 21, 22, 23);
  database.update_feature(42, 2.1, 0, 30, 31, 32, 33); // uncovered future image
  auto features = database.clone_features();
  ov_init::InitializerFeatureTimeBounds bounds;
  ov_init::InitializerRawTimeSidecar sidecar;
  check(ov_init::prepare_physical_initializer_features(features, {{0,.005},{1,-.125}}, .005, .1, 2., 2., bounds, sidecar),
        "physical chart accepts unequal clocks and a covering IMU bracket");
  const double graph_time = features.at(42)->timestamps.at(0)[0];
  check(initializer_time_bits(sidecar.at(42).at(0)[0]) == initializer_time_bits(raw) &&
        initializer_time_bits(graph_time - .005) != initializer_time_bits(raw),
        "original one-ULP raw key survives; subtraction negative control loses identity");
  check(bounds.newest_ref_time == graph_time && bounds.observation_count == 2 &&
        features.at(42)->uvs.at(0)[0] == Eigen::Vector2f(10,11), "covered physical horizon and pixel alignment");
  const double reconstructed = graph_time - .005;
  check(database.cleanup_measurements_exact_observations({{42,0,reconstructed}}) == 0,
        "wrong reverse-retimed key cannot consume the original pixel");
  check(database.cleanup_measurements_exact_observations({{42,0,raw}}) == 1 &&
        database.get_feature(42)->timestamps.at(1).size() == 1 && database.get_feature(42)->timestamps.at(0)[0] == 2.1,
        "receipt consumption preserves another camera and future rows");
  check(database.cleanup_measurements_exact_observations({{42,0,raw}}) == 0, "receipt consumption is idempotent");
  auto invalid = database.clone_features(); invalid.at(42)->uvs.at(0).clear();
  const auto old_bounds = bounds; const auto old_keys = sidecar;
  check(!ov_init::prepare_physical_initializer_features(invalid, {}, 0., .1, 2., 2., bounds, sidecar) &&
        old_keys == sidecar && bounds.observation_count == old_bounds.observation_count && invalid.at(42)->timestamps.at(0)[0] == 2.1,
        "malformed input rejects without changing chart, bounds or sidecar");
}
void staged_import() {
  const auto result = fixture(); auto state = make_state(result);
  check(StateHelper::set_initial_state_physical_warm(state, result, 7), "singular equal-endpoint owners import");
  check(same(StateHelper::get_full_covariance(state), result.joint_covariance), "full joint covariance copied exactly with no Q or jitter");
  check(state->clone_count() == 3 && state->_clones_IMU.empty() && state->_features_SLAM.empty() &&
        state->_exposure_poses[0].pose != state->_exposure_poses[1].pose &&
        state->_exposure_poses[0].pose->id() == 15 && state->_exposure_poses[1].pose->id() == 21,
        "coincident exposures retain distinct handles and covariance rows");
  for (const auto &owner : result.owners) {
    const auto pose = state->find_pose(owner.camera_id, owner.raw_time);
    const auto kin = state->clone_kinematics(owner.camera_id, owner.raw_time);
    check(pose && same(pose->value(), owner.pose_mean) && same(pose->fej(), owner.pose_mean), "accepted owner value and FEJ");
    check(kin && same(kin->vel, owner.velocity_world) && same(kin->omega, owner.omega_body) && same(kin->vel_fej, owner.velocity_world),
          "historical optimized velocity and endpoint gyro retained without a secant");
  }
  auto snapshot = StateHelper::clone_state(state);
  check(snapshot->_initialization_episode_id == 7 && snapshot->imu_endpoint() == 2. &&
        snapshot->_exposure_poses[0].pose != state->_exposure_poses[0].pose &&
        same(StateHelper::get_full_covariance(snapshot), result.joint_covariance), "snapshot preserves endpoint, episode and independent owners");
  Eigen::Matrix<double,1,1> changed; changed << .3; state->cam_imu_dt_var(0)->set_value(changed);
  check(state->imu_endpoint() == 2., "later clock mean cannot relabel accepted IMU endpoint");
  const auto old_cov = StateHelper::get_full_covariance(state);
  check(!StateHelper::set_initial_state_physical_warm(state, result, 7) && same(old_cov, StateHelper::get_full_covariance(state)),
        "repeat import cannot duplicate a live window");
  // Positive semidefinite also permits deterministic zero diagonal directions.
  auto semidefinite = result;
  semidefinite.joint_covariance.row(12).setZero(); semidefinite.joint_covariance.col(12).setZero();
  check(StateHelper::set_initial_state_physical_warm(make_state(result), semidefinite, 7), "valid zero-diagonal PSD is accepted without artificial variance");
  check(!StateHelper::valid_initial_covariance(semidefinite.joint_covariance), "legacy positive-diagonal guard is a negative control for deterministic PSD");

  for (int bad = 0; bad < 30; ++bad) {
    auto candidate = result; auto target = make_state(result, bad == 0 ? 2 : 3);
    switch (bad) {
      case 1: candidate.episode_id = 8; break;
      case 2: candidate.joint_covariance(0,16) = std::numeric_limits<double>::quiet_NaN(); break;
      case 3: candidate.joint_covariance(16,0) = std::numeric_limits<double>::infinity(); break;
      case 4: candidate.joint_covariance(0,1) += .1; break;
      case 5: candidate.joint_covariance(12,12) = -1.; break;
      case 6: candidate.imu_mean(7) = std::numeric_limits<double>::quiet_NaN(); break;
      case 7: candidate.owners[0].raw_time = std::nextafter(candidate.owners[0].raw_time, 0.); break;
      case 8: candidate.owners[1].camera_id = 0; candidate.owners[1].raw_time = candidate.owners[0].raw_time; break;
      case 9: std::swap(candidate.owners[0], candidate.owners[1]); break;
      case 10: candidate.owners[1].graph_node = 1; break;
      case 11: candidate.accepted_imu_endpoint = 1.9; break;
      case 12: candidate.reference_clock_label += .01; break;
      case 13: candidate.calibration[1].clock_mean += .01; break;
      case 14: candidate.calibration[0].intrinsics(0) += 1.; break;
      case 15: candidate.imu_gyro_map(0,0) += .01; break;
      case 16: target->_options.do_calib_camera_timeoffset = true; break;
      case 17: target->_options.do_calib_camera_pose = true; break;
      case 18: target->_options.do_calib_camera_intrinsics = true; break;
      case 19: target->_options.do_calib_imu_intrinsics = true; break;
      case 20: target->_options.do_calib_imu_g_sensitivity = true; break;
      case 21: target->_options.do_calib_camera_readout = true; break;
      case 22: { Eigen::Matrix<double,1,1> readout; readout << .005; target->_calib_camera_readout.at(0)->set_value(readout); break; }
      case 23: candidate.consumed_observations.clear(); break;
      case 24: candidate.consumed_observations.push_back(candidate.consumed_observations.back()); break;
      case 25: candidate.owners[1].velocity_world.x() += .1; break;
      case 26: candidate.owners[0].pose_mean(3) = 0.; break;
      case 27: candidate.joint_covariance = candidate.joint_covariance.topLeftCorner(32,32).eval(); break;
      case 28: target->_options.physical_camera_clones = false; break;
      case 29: candidate.joint_covariance(21,21) += .001; break; // spurious independent bridge Q at a shared endpoint
    }
    const auto covariance_before = StateHelper::get_full_covariance(target); const auto mean_before = target->_imu->value();
    const auto imu_before = target->_imu;
    check(!StateHelper::set_initial_state_physical_warm(target, candidate, 7), "malformed/unsupported physical contract rejects");
    check(target->_imu == imu_before && same(mean_before, target->_imu->value()) && same(covariance_before, StateHelper::get_full_covariance(target)) &&
          target->_exposure_poses.empty() && !target->_imu_endpoint_valid && target->_initialization_episode_id == 0 && target->_timestamp == -1.,
          "rejected import leaves all state ownership and values unchanged");
  }
}

Eigen::Matrix3d rotation(double t) {
  using A = Eigen::AngleAxisd;
  return (A(.28*std::sin(1.2*t), Eigen::Vector3d::UnitZ()) * A(.45+.32*std::sin(2.2*t), Eigen::Vector3d::UnitY()) *
          A(M_PI+.22*std::sin(1.7*t), Eigen::Vector3d::UnitX())).toRotationMatrix();
}
Eigen::Vector3d position(double t) { return {.18*std::sin(1.7*t), .13*(std::cos(2.1*t)-1.), .10*std::sin(1.3*t)}; }
Eigen::Vector3d velocity(double t) { return {.306*std::cos(1.7*t), -.273*std::sin(2.1*t), .13*std::cos(1.3*t)}; }
Eigen::Vector3d acceleration(double t) { return {-.5202*std::sin(1.7*t), -.5733*std::cos(2.1*t), -.169*std::sin(1.3*t)}; }

void actual_dynamic_export() {
  ov_init::InertialInitializerOptions options;
  options.init_window_time = 2.; options.init_max_features = 80; options.init_dyn_num_pose = 11;
  options.init_dyn_mle_max_iter = 80; options.init_dyn_mle_max_time = 30.; options.init_dyn_mle_max_threads = 1;
  options.init_warmstart_inject = true; options.num_cameras = 2; options.calib_camimu_dt = .125;
  options.camera_imu_dt = {{0,.125},{1,-.25}};
  options.sigma_w = .001; options.sigma_wb = .0001; options.sigma_a = .01; options.sigma_ab = .001;
  const Eigen::Vector3d bg(.008,-.005,.004), ba(.012,-.009,.006);
  options.init_dyn_bias_g = bg + Eigen::Vector3d(.001,-.001,.0005);
  options.init_dyn_bias_a = ba + Eigen::Vector3d(.005,-.004,.002);
  const Eigen::Matrix3d Ric = Eigen::AngleAxisd(M_PI,Eigen::Vector3d::UnitX()).toRotationMatrix();
  for (int camera = 0; camera < 2; ++camera) {
    auto model = std::make_shared<CamRadtan>(1280,800);
    Eigen::Matrix<double,8,1> intr; intr << 458.,463.,640.,400.,.06,-.01,.002,-.0005; model->set_value(intr);
    options.camera_intrinsics[camera] = model;
    Eigen::VectorXd extrinsics(7); extrinsics.head<4>() = rot_2_quat(Ric); extrinsics.tail<3>() << .12*camera,0.,0.;
    options.camera_extrinsics[camera] = extrinsics;
  }
  auto imu_data = std::make_shared<std::vector<ImuData>>();
  for (int i = -160; i <= 1936; ++i) {
    const double t = i/800.; const double h = 1e-5;
    ImuData sample; sample.timestamp = 10.+t;
    sample.wm = -log_so3(rotation(t+h)*rotation(t-h).transpose())/(2*h) + bg;
    sample.am = rotation(t)*(acceleration(t)+Eigen::Vector3d(0,0,9.81)) + ba;
    imu_data->push_back(sample);
  }
  auto database = std::make_shared<FeatureDatabase>();
  for (int frame = 0; frame <= 72; ++frame)
    for (int camera = 0; camera < 2; ++camera)
      for (int feature = 0; feature < 80; ++feature) {
        const double t = frame/30. + (camera == 1 && frame%2 ? .013 : 0.);
        const Eigen::Vector3d point(-1.6+.4*(feature%9),-1.2+.4*((feature/9)%7),4.+.37*(feature%11));
        const Eigen::Vector3d pc = Ric*rotation(t)*(point-position(t)) + options.camera_extrinsics.at(camera).tail<3>();
        const Eigen::Vector2f normalized = (pc.head<2>()/pc.z()).cast<float>();
        Eigen::Vector2f pixel = options.camera_intrinsics.at(camera)->distort_f(normalized);
        pixel.x() += .02*std::sin(1.7*feature+.6*frame); pixel.y() += .02*std::cos(.8*feature+.9*frame);
        const auto tracked = options.camera_intrinsics.at(camera)->undistort_f(pixel);
        database->update_feature(feature,10.+t-options.camera_imu_dt.at(camera),camera,pixel.x(),pixel.y(),tracked.x(),tracked.y());
      }
  const auto source = database->clone_features(); const size_t imu_count = imu_data->size();
  const double imu_first = imu_data->front().timestamp;
  auto reset = std::make_shared<ov_init::ResetContext>();
  auto run = [&](const ov_init::InertialInitializerOptions &configuration, size_t cap, InitPhysicalWarmResult &output) {
    ov_init::DynamicInitializer initializer(configuration,database,imu_data,reset);
    double timestamp = -99.; Eigen::MatrixXd covariance = Eigen::MatrixXd::Constant(1,1,99.);
    auto imu = std::make_shared<ov_type::IMU>(); const auto original_imu = imu;
    std::vector<std::shared_ptr<ov_type::Type>> order;
    std::map<double,std::shared_ptr<ov_type::PoseJPL>> clones;
    std::unordered_map<size_t,std::shared_ptr<ov_type::Landmark>> landmarks;
    InitPhysicalWarmRequest request; request.episode_id = 9; request.max_retained_owners = cap;
    const bool success = initializer.initialize(timestamp,covariance,order,imu,clones,landmarks,&request,&output);
    check(imu_data->size() == imu_count && imu_data->front().timestamp == imu_first, "physical solve does not prune shared IMU input");
    if (success)
      check(imu == original_imu && covariance.rows() == 15 && order.size() == 1 && order[0] == imu && clones.empty() && landmarks.empty() &&
            timestamp == output.reference_clock_label, "physical result remains separate from legacy marginal outputs");
    else
      check(imu == original_imu && timestamp == -99. && covariance.rows() == 1 && covariance(0,0) == 99. && order.empty(),
            "failed physical solve leaves legacy outputs unchanged");
    return success;
  };
  InitPhysicalWarmResult full;
#ifdef USE_CERES_FREE_INIT
  const bool solved = run(options,64,full);
  check(solved, "actual conditional initializer accepts unequal physical clocks");
  if (!solved) return;
  check(full.owners.size() > 4 && full.accepted_imu_endpoint <= imu_data->back().timestamp &&
        full.accepted_imu_endpoint == 12.4, "all graph owners export at covered fixed endpoint");
  const Eigen::Matrix3d R = quat_2_Rot(full.imu_mean.head<4>());
  const double endpoint_t = full.accepted_imu_endpoint - 10.;
  check((R*Eigen::Vector3d::UnitZ()-rotation(endpoint_t)*Eigen::Vector3d::UnitZ()).norm() < .02 &&
        (R*full.imu_mean.segment<3>(7)-rotation(endpoint_t)*velocity(endpoint_t)).norm() < .04,
        "physical IMU mean matches independent analytic tilt/body velocity at accepted IMU endpoint");
  for (const auto &owner : full.owners)
    check((quat_2_Rot(owner.pose_mean.head<4>())*owner.velocity_world -
           rotation(owner.nominal_imu_time-10.)*velocity(owner.nominal_imu_time-10.)).norm() < .05,
          "historical owner velocity agrees with analytic trajectory, not latest IMU velocity");
  size_t shared = 0;
  for (size_t i = 1; i < full.owners.size(); ++i)
    if (full.owners[i].nominal_imu_time == full.owners[i-1].nominal_imu_time) {
      ++shared;
      check(full.owners[i].camera_id != full.owners[i-1].camera_id && full.owners[i].graph_node == full.owners[i-1].graph_node &&
            (full.joint_covariance.middleRows(15+6*i,6)-full.joint_covariance.middleRows(15+6*(i-1),6)).norm() < 1e-14,
            "same fixed graph node exports distinct perfectly correlated owner rows");
    }
  check(shared > 0, "actual initializer exercised equal endpoints with different camera/raw owners");
  InitPhysicalWarmResult bounded;
  check(run(options,4,bounded) && bounded.owners.size() == 4 && bounded.joint_covariance.rows() == 39,
        "retained output covariance is bounded independently of graph size");
  if (bounded.owners.size() != 4) return;
  Eigen::MatrixXd selector = Eigen::MatrixXd::Zero(39,full.joint_covariance.rows());
  selector.topLeftCorner(15,15).setIdentity(); selector.bottomRightCorner(24,24).setIdentity();
  const Eigen::MatrixXd oracle = selector * full.joint_covariance * selector.transpose();
  check((bounded.joint_covariance-oracle).cwiseAbs().maxCoeff() < 1e-12 && same(bounded.imu_mean,full.imu_mean),
        "bounded owner marginal equals full gravity-mapped covariance row selection");
  check(bounded.consumed_observations.size() == full.consumed_observations.size(), "receipt still contains all fitted rows after output marginalization");
  auto target = make_state(bounded,4);
  check(StateHelper::set_initial_state_physical_warm(target,bounded,9), "actual dynamic output passes staged StateHelper import");
  size_t old_rows = 0;
  for (const auto &key : bounded.consumed_observations) {
    const auto &times = source.at(key.feature_id)->timestamps.at(key.camera_id);
    check(std::any_of(times.begin(),times.end(),[&](double time) { return initializer_time_bits(time) == initializer_time_bits(key.raw_time); }),
          "every consumed key preserves the original raw double bits");
    if (!target->find_pose(key.camera_id,key.raw_time)) ++old_rows;
  }
  check(old_rows > 0, "fitted rows without retained owners remain in receipt");
  check(database->cleanup_measurements_exact_observations(bounded.consumed_observations) == bounded.consumed_observations.size(),
        "accepted exact receipt consumes all and only actual image likelihood rows");
  check(database->cleanup_measurements_exact_observations(bounded.consumed_observations) == 0, "actual solve receipt has no repeat likelihood");
#else
  check(!run(options,4,full), "alternate Ceres backend explicitly rejects unsupported physical export");
#endif
  for (int bad = 0; bad < 3; ++bad) {
    auto unsupported = options;
    if (bad == 0) unsupported.init_dyn_mle_opt_calib = true;
    if (bad == 1) { ov_init::ResetBiasPrior prior; prior.valid = true; reset->arm(prior); }
    if (bad == 2) { reset->disarm(); unsupported.init_dyn_grav_gate_deg = -1.; }
    InitPhysicalWarmResult unchanged; unchanged.episode_id = 123;
    check(!run(unsupported,4,unchanged) && unchanged.episode_id == 123 && unchanged.owners.empty(),
          "unsupported fitted calibration/reset prior or rejected solve preserves output receipt");
  }
}
} // namespace

int main() {
  ov_core::Printer::setPrintLevel("WARNING");
  raw_keys_and_receipts(); staged_import(); actual_dynamic_export();
  std::printf("physical warmstart: %d/%d checks passed\n",checks-failures,checks);
  return failures ? 1 : 0;
}
