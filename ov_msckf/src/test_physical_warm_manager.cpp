/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 * The real InertialInitializer policy, nonlinear dynamic solve and manager
 * transaction run here. There is no solver interposition or manufactured
 * posterior. A separate detached solve is the import/receipt oracle; analytic
 * motion independently checks the endpoint and historical owner kinematics.
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#include "cam/CamRadtan.h"
#include "core/VioManager.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "init/InertialInitializer.h"
#include "init/InitializerGeometry.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Landmark.h"
#include "utils/InitializerPhysicalWarmResult.h"
#include "utils/print.h"

namespace {
using namespace ov_core;
using namespace ov_msckf;
int checks = 0, failures = 0;
void check(bool ok, const char *message) {
  ++checks;
  if (!ok) { ++failures; std::printf("FAIL: %s\n", message); }
}
template <class A, class B> bool same(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
      std::memcmp(a.derived().data(), b.derived().data(), sizeof(double) * a.size()) == 0;
}
template <class A, class B> bool near(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b, double tolerance = 1e-8) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
      (a - b).norm() <= tolerance * std::max(1., b.norm());
}
Eigen::Matrix3d rotation(double t) {
  using A = Eigen::AngleAxisd;
  return (A(.28*std::sin(1.2*t), Eigen::Vector3d::UnitZ()) * A(.45+.32*std::sin(2.2*t), Eigen::Vector3d::UnitY()) *
          A(M_PI+.22*std::sin(1.7*t), Eigen::Vector3d::UnitX())).toRotationMatrix();
}
Eigen::Vector3d position(double t) { return {.18*std::sin(1.7*t), .13*(std::cos(2.1*t)-1.), .10*std::sin(1.3*t)}; }
Eigen::Vector3d velocity(double t) { return {.306*std::cos(1.7*t), -.273*std::sin(2.1*t), .13*std::cos(1.3*t)}; }
Eigen::Vector3d acceleration(double t) { return {-.5202*std::sin(1.7*t), -.5733*std::cos(2.1*t), -.169*std::sin(1.3*t)}; }
const Eigen::Vector3d bg(.008,-.005,.004), ba(.012,-.009,.006);

VioManagerOptions options() {
  VioManagerOptions p;
  p.state_options.num_cameras = p.init_options.num_cameras = 2;
  p.state_options.max_clone_size = 3; // manager resolves this to six camera-owned poses
  p.state_options.max_slam_features = p.state_options.max_aruco_features = 0;
  p.state_options.physical_camera_clones = true;
  p.state_options.imu_model = StateOptions::RPNG;
  p.epoch_mode = p.async_frame_clones = false;
  p.use_stereo = p.use_aruco = p.use_gpu = p.try_zupt = false;
  p.use_multi_threading_pubs = false; p.use_multi_threading_subs = true;
  p.num_opencv_threads = 0;
  p.num_pts = p.init_options.init_max_features = 80;
  p.async_ring_size = 16; p.async_guard = .001953125; p.async_stale_factor = 0.;
  p.gravity_mag = 9.81;
  p.vec_dw << 1, 0, 1, 0, 0, 1; p.vec_da = p.vec_dw; p.vec_tg.setZero();
  p.q_ACCtoIMU << 0, 0, 0, 1; p.q_GYROtoIMU = p.q_ACCtoIMU;
  p.imu_noises.sigma_w = p.init_options.sigma_w = .001;
  p.imu_noises.sigma_wb = p.init_options.sigma_wb = .0001;
  p.imu_noises.sigma_a = p.init_options.sigma_a = .01;
  p.imu_noises.sigma_ab = p.init_options.sigma_ab = .001;
  p.init_options.use_stereo = false;
  p.init_options.init_dyn_use = p.init_options.init_warmstart_inject = true;
  p.init_options.init_window_time = 2.; p.init_options.init_dyn_num_pose = 11;
  p.init_options.init_dyn_mle_max_iter = 80; p.init_options.init_dyn_mle_max_time = 30.;
  p.init_options.init_dyn_mle_max_threads = 1;
  p.init_options.init_dyn_bias_g = bg + Eigen::Vector3d(.001,-.001,.0005);
  p.init_options.init_dyn_bias_a = ba + Eigen::Vector3d(.005,-.004,.002);
  const auto Ric = Eigen::AngleAxisd(M_PI,Eigen::Vector3d::UnitX()).toRotationMatrix();
  for (int camera = 0; camera < 2; ++camera) {
    auto model = std::make_shared<CamRadtan>(1280,800);
    Eigen::Matrix<double,8,1> intrinsic; intrinsic << 458.,463.,640.,400.,.06,-.01,.002,-.0005;
    model->set_value(intrinsic);
    Eigen::VectorXd extrinsic(7); extrinsic.head<4>() = rot_2_quat(Ric); extrinsic.tail<3>() << .12*camera,0.,0.;
    p.camera_intrinsics[camera] = p.init_options.camera_intrinsics[camera] = model;
    p.camera_extrinsics[camera] = p.init_options.camera_extrinsics[camera] = extrinsic;
    p.camera_imu_dt[camera] = camera ? -.25 : .125;
    p.camera_readout_time[camera] = 0.; p.camera_shutter_rolling[camera] = false;
  }
  return p;
}

std::vector<ImuData> samples(int first, int last) {
  std::vector<ImuData> output;
  for (int i = first; i <= last; ++i) {
    const double t = i/800., h = 1e-5;
    ImuData sample; sample.timestamp = 10.+t;
    sample.wm = -log_so3(rotation(t+h)*rotation(t-h).transpose())/(2*h) + bg;
    sample.am = rotation(t)*(acceleration(t)+Eigen::Vector3d(0,0,9.81)) + ba;
    output.push_back(sample);
  }
  return output;
}

// Raw bits are part of the key; equal physical endpoints never merge cameras.
using RowKey = std::tuple<size_t,size_t,uint64_t>;
using Rows = std::map<RowKey,std::array<float,4>>;
RowKey key(size_t feature, size_t camera, double raw) { return {feature,camera,initializer_time_bits(raw)}; }
RowKey key(const InitObservationKey &row) { return key(row.feature_id,row.camera_id,row.raw_time); }
double raw_time(const RowKey &row) {
  const uint64_t bits = std::get<2>(row); double raw;
  std::memcpy(&raw,&bits,sizeof(raw)); return raw;
}
Rows rows(const std::shared_ptr<FeatureDatabase> &database) {
  Rows result;
  for (const auto &entry : database->clone_features())
    for (const auto &camera : entry.second->timestamps)
      for (size_t i = 0; i < camera.second.size(); ++i) {
        const auto &uv = entry.second->uvs.at(camera.first).at(i);
        const auto &norm = entry.second->uvs_norm.at(camera.first).at(i);
        result.emplace(key(entry.first,camera.first,camera.second[i]),std::array<float,4>{{uv.x(),uv.y(),norm.x(),norm.y()}});
      }
  return result;
}

class Manager : public VioManager {
public:
  explicit Manager(VioManagerOptions p = options()) : VioManager(p) { prepare_observation_replay(); }
  std::shared_ptr<FeatureDatabase> database() { return trackFEATS->get_feature_database(); }
  bool attempt() {
    CameraData message; message.timestamp = 12.4-state->cam_imu_dt(0); message.sensor_ids = {0};
    return try_to_initialize(message);
  }
  bool success() const { return thread_init_success.load(); }
  bool reset_armed() const { return warmstart_next_init.load(); }
  bool pending() const { return thread_init_running.load() || initialization_attempt; }
  void finish_worker_only() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(45);
    while (!initialization_completed()) {
      if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("actual initializer did not complete");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  Eigen::Vector2f pixel(size_t feature, size_t camera, double t, int frame) const {
    // Keep the fixed scene in both cameras' visible field over the complete
    // initialization and continuation trajectory. Off-sensor radtan points can
    // cross a non-injective distortion region and corrupt the linear seed.
    const Eigen::Vector3d point(.4+.4*(feature%9),-1.2+.4*((feature/9)%7),4.+.37*(feature%11));
    const auto &extrinsic = params.camera_extrinsics.at(camera);
    const Eigen::Vector3d pc = quat_2_Rot(extrinsic.head<4>())*rotation(t)*(point-position(t)) + extrinsic.tail<3>();
    if (!ov_init::InitializerGeometry::valid_camera_point(pc))
      throw std::runtime_error("synthetic point is outside the forward camera half-space");
    const Eigen::Vector2f true_ray = (pc.head<2>()/pc.z()).cast<float>();
    const auto &model = params.camera_intrinsics.at(camera);
    Eigen::Vector2f uv = model->distort_f(true_ray);
    const auto visible = [&](const Eigen::Vector2f &pixel) {
      return pixel.x() >= 0.f && pixel.y() >= 0.f && pixel.x() < model->w() && pixel.y() < model->h();
    };
    if (!visible(uv) || (model->undistort_f(uv)-true_ray).norm() > 1e-5f)
      throw std::runtime_error("synthetic point must be visible with an invertible camera ray");
    uv.x() += .02*std::sin(1.7*feature+.6*frame); uv.y() += .02*std::cos(.8*feature+.9*frame);
    if (!visible(uv)) throw std::runtime_error("synthetic pixel noise leaves the sensor");
    return uv;
  }
  void add(size_t feature, size_t camera, double raw, int frame = 0) {
    const Eigen::Vector2f uv = pixel(feature,camera,raw+state->cam_imu_dt(camera)-10.,frame);
    const auto norm = params.camera_intrinsics.at(camera)->undistort_f(uv);
    database()->update_feature(feature,raw,camera,uv.x(),uv.y(),norm.x(),norm.y());
  }
  void prime() {
    for (int frame = 0; frame <= 72; ++frame)
      for (size_t camera = 0; camera < 2; ++camera) {
        const double t = frame/30. + (camera == 1 && frame%2 ? .013 : 0.);
        for (size_t feature = 0; feature < 80; ++feature)
          add(feature,camera,10.+t-state->cam_imu_dt(camera),frame);
      }
    feed_measurement_batch_imu(samples(-160,1936));
  }
  bool detached_reference(InitPhysicalWarmResult &result) {
    auto input = initializer->make_attempt();
    InitPhysicalWarmRequest request; request.episode_id = 1; request.max_retained_owners = state->_options.max_pose_clones();
    if (!input->request_physical_warmstart(request)) return false;
    auto imu = std::make_shared<ov_type::IMU>();
    double timestamp = -1.; Eigen::MatrixXd covariance;
    std::vector<std::shared_ptr<ov_type::Type>> order;
    std::map<double,std::shared_ptr<ov_type::PoseJPL>> clones;
    std::unordered_map<size_t,std::shared_ptr<ov_type::Landmark>> features;
    if (!input->initialize(timestamp,covariance,order,imu,clones,features,true) || !input->physical_warm_result()) return false;
    result = *input->physical_warm_result(); return true;
  }
  void arm_prior() {
    ov_init::ResetBiasPrior prior; prior.valid = true; prior.bg = bg; prior.ba = ba; prior.t_snapshot = 10.;
    prior.sigma_bg.setConstant(.02); prior.sigma_ba.setConstant(.1); initializer->set_reset_prior(prior);
  }
  void change_live_fixed_intrinsics() {
    auto value = state->_cam_intrinsics.at(0)->value(); value(0) += 1.;
    state->_cam_intrinsics.at(0)->set_value(value); state->_cam_intrinsics.at(0)->set_fej(value);
    state->_cam_intrinsics_cameras.at(0)->set_value(value);
  }
  void queue_group(int tick, int frame) {
    for (size_t camera = 0; camera < 2; ++camera) {
      const double t = tick/800.;
      FeatureObservations observations;
      for (size_t feature = 0; feature < 80; ++feature) {
        Eigen::VectorXf uv(2); uv = pixel(feature,camera,t,frame); observations.emplace_back(feature,uv);
      }
      feed_measurement_simulation_queued(10.+t-state->cam_imu_dt(camera),{static_cast<int>(camera)},{std::move(observations)});
    }
  }
};

std::vector<RowKey> append_arrivals(Manager &manager, const InitPhysicalWarmResult &result) {
  std::vector<RowKey> preserved;
  for (size_t camera = 0; camera < 2; ++camera) {
    const auto owner = std::find_if(result.owners.rbegin(),result.owners.rend(),[camera](const InitExposureOwner &o) { return o.camera_id == camera; });
    if (owner == result.owners.rend()) throw std::runtime_error("fixture has no retained camera owner");
    // New feature at an existing owner; one-ULP neighbour of an assimilated
    // latest feature; and a later exposure. All arrive after the solve snapshot.
    const std::array<std::pair<size_t,double>,3> arrivals{{
        {1000+camera,owner->raw_time},
        {0,std::nextafter(owner->raw_time,std::numeric_limits<double>::infinity())},
        {2000+camera,result.accepted_imu_endpoint+.025-manager.get_state()->cam_imu_dt(camera)}}};
    for (const auto &arrival : arrivals) {
      manager.add(arrival.first,camera,arrival.second,72);
      preserved.push_back(key(arrival.first,camera,arrival.second));
    }
  }
  return preserved;
}

void continuation(Manager &manager) {
  size_t processed = 0, dropped = 0;
  manager.set_camera_processed_callback([&](const CameraData &message,bool accepted) {
    processed += accepted ? message.sensor_ids.size() : 0;
    dropped += accepted ? 0 : message.sensor_ids.size(); return true;
  });
  int next_imu = 1937;
  for (int frame = 0; frame < 10; ++frame) {
    const int tick = 2000+25*frame;
    manager.queue_group(tick,75+frame);
    manager.feed_measurement_batch_imu(samples(next_imu,tick+10)); next_imu = tick+11;
    const auto state = manager.get_state();
    const double endpoint = 10.+tick/800.;
    auto a = state->find_pose(0,endpoint-state->cam_imu_dt(0));
    auto b = state->find_pose(1,endpoint-state->cam_imu_dt(1));
    check(state->imu_endpoint() == endpoint && a && b && a != b,
          "ordinary queued continuation preserves each equal-endpoint exposure owner");
    const auto covariance = StateHelper::get_full_covariance(state);
    check(state->clone_count() <= static_cast<size_t>(state->_options.max_pose_clones()) &&
          covariance.rows() == 15+6*static_cast<int>(state->clone_count()) &&
          StateHelper::valid_initial_covariance(covariance,true),
          "ordinary propagation, image update and marginalization keep a bounded PSD window");
  }
  manager.finish_observation_replay();
  check(processed == 20 && dropped == 0 && manager.initialized(), "all post-import observations pass through the public queued manager caller");
  manager.set_camera_processed_callback({});
}

void fresh_boot() {
  Manager manager; manager.prime();
  const auto state = manager.get_state();
  InitPhysicalWarmResult reference;
  const bool solved = manager.detached_reference(reference);
  check(solved, "real policy and dynamic solver produce a fixed-calibration reference result");
  if (!solved) return;
  check(!manager.reset_armed() && state->_initialization_episode_id == 0 && state->clone_count() == 0,
        "fresh-boot fixture has no soft-reset warm flag or preexisting owner");
  check(reference.owners.size() == static_cast<size_t>(state->_options.max_pose_clones()) && reference.accepted_imu_endpoint == 12.4,
        "real export is bounded to the requested six owners at a covered physical horizon");
  const auto old_mean = state->_imu->value(), old_fej = state->_imu->fej();
  const auto old_covariance = StateHelper::get_full_covariance(state);
  check(!manager.attempt(), "fresh-boot opt-in launches an actual asynchronous initialization");
  const auto preserved = append_arrivals(manager,reference);
  const auto before_import = rows(manager.database());
  manager.finish_worker_only();
  check(!manager.success() && same(old_mean,state->_imu->value()) && same(old_fej,state->_imu->fej()) &&
        same(old_covariance,StateHelper::get_full_covariance(state)) && !state->_imu_endpoint_valid &&
        state->_initialization_episode_id == 0 && state->clone_count() == 0,
        "completed real worker cannot publish mean, FEJ, endpoint, owner or covariance before consumer import");
  check(rows(manager.database()) == before_import, "completed real worker consumes no live observation, including concurrent arrivals");
  bool all_receipts_present = !reference.consumed_observations.empty();
  for (const auto &receipt : reference.consumed_observations) all_receipts_present &= before_import.count(key(receipt)) == 1;
  check(all_receipts_present, "every exact assimilated row remains present before successful import");
  check(manager.attempt() && manager.success() && !manager.pending(), "consumer accepts the actual physical warm result on first boot");
  if (!manager.success()) return;
  check(state->_initialization_episode_id != 0 && state->imu_endpoint() == reference.accepted_imu_endpoint &&
        state->_timestamp == reference.reference_clock_label && state->_exposure_poses.size() == reference.owners.size() &&
        state->_clones_IMU.empty() && state->_epoch_bridges.empty(), "manager installs physical owners and endpoint without legacy clones or bridge state");
  check(near(StateHelper::get_full_covariance(state),reference.joint_covariance) && near(state->_imu->value(),reference.imu_mean),
        "manager imports complete conditional joint posterior, without another process-noise increment");
  const Eigen::Matrix3d R = state->_imu->Rot();
  check((R*Eigen::Vector3d::UnitZ()-rotation(2.4)*Eigen::Vector3d::UnitZ()).norm() < .02 &&
        (R*state->_imu->vel()-rotation(2.4)*velocity(2.4)).norm() < .04,
        "accepted manager mean agrees with independent analytic tilt and body velocity");
  size_t shared_endpoints = 0;
  const auto imported_covariance = StateHelper::get_full_covariance(state);
  for (size_t i = 0; i < reference.owners.size(); ++i) {
    const auto &owner = reference.owners[i]; const auto pose = state->find_pose(owner.camera_id,owner.raw_time);
    const auto kin = state->clone_kinematics(owner.camera_id,owner.raw_time);
    check(pose && near(pose->value(),owner.pose_mean) && same(pose->value(),pose->fej()) && kin &&
          near(kin->vel,owner.velocity_world) && near(kin->omega,owner.omega_body) &&
          (quat_2_Rot(owner.pose_mean.head<4>())*kin->vel -
           rotation(owner.nominal_imu_time-10.)*velocity(owner.nominal_imu_time-10.)).norm() < .05,
          "raw-key lookup retains each optimized historical owner pose, FEJ, velocity and gyro");
    if (i && reference.owners[i-1].nominal_imu_time == owner.nominal_imu_time) {
      ++shared_endpoints;
      const auto &previous = reference.owners[i-1];
      check(owner.camera_id != previous.camera_id && pose != state->find_pose(previous.camera_id,previous.raw_time) &&
            near(imported_covariance.middleRows(15+6*i,6),imported_covariance.middleRows(15+6*(i-1),6),1e-12),
            "unequal raw clocks at one endpoint retain distinct perfectly correlated owner rows");
    }
  }
  check(shared_endpoints > 0, "actual bounded warm window contains coincident physical endpoints from different cameras");

  std::set<RowKey> consumed;
  for (const auto &receipt : reference.consumed_observations) consumed.insert(key(receipt));
  std::map<size_t,double> oldest;
  for (const auto &owner : reference.owners) {
    const auto found = oldest.find(owner.camera_id);
    if (found == oldest.end() || owner.raw_time < found->second) oldest[owner.camera_id] = owner.raw_time;
  }
  Rows expected = before_import;
  for (auto it = expected.begin(); it != expected.end();) {
    if (consumed.count(it->first) || raw_time(it->first) < oldest.at(std::get<1>(it->first))) it = expected.erase(it);
    else ++it;
  }
  const auto remaining = rows(manager.database());
  check(remaining == expected, "manager cleanup equals exact receipt consumption plus per-camera retained-window age pruning");
  bool absent = true, retained = true;
  for (const auto &receipt : consumed) absent &= remaining.count(receipt) == 0;
  for (const auto &arrival : preserved) retained &= remaining.count(arrival) == 1;
  check(absent && retained, "assimilated likelihood rows are gone; unused owner rows, one-ULP neighbours and future arrivals survive");
  continuation(manager);
}

void rejected_export() {
  Manager manager; manager.prime();
  InitPhysicalWarmResult reference;
  const bool solved = manager.detached_reference(reference);
  check(solved, "invalid-export control has a real independently valid navigation marginal");
  if (!solved) return;
  manager.attempt(); const auto preserved = append_arrivals(manager,reference);
  const auto before = rows(manager.database());
  manager.finish_worker_only();
  check(rows(manager.database()) == before, "export awaiting rejection still consumes no live measurement");
  // This is a legitimate import-boundary mismatch. The solved snapshot and
  // IMU marginal remain valid, while its fixed calibration contract is stale.
  manager.change_live_fixed_intrinsics();
  check(manager.attempt() && manager.success(), "rejected physical export falls back to its independently valid cold navigation marginal");
  const auto state = manager.get_state();
  check(state->_initialization_episode_id == 0 && state->clone_count() == 0 && state->_clones_IMU.empty() &&
        near(state->_imu->value(),reference.imu_mean) &&
        near(StateHelper::get_full_covariance(state),reference.joint_covariance.topLeftCorner(15,15)),
        "rejected export leaves no partially installed owner, episode or joint covariance");
  Rows expected = before;
  for (auto it = expected.begin(); it != expected.end();) {
    const double cutoff = std::nextafter(state->imu_endpoint()-state->cam_imu_dt(std::get<1>(it->first)),
                                        std::numeric_limits<double>::infinity());
    if (raw_time(it->first) < cutoff) it = expected.erase(it); else ++it;
  }
  const auto after = rows(manager.database());
  check(after == expected && after.count(preserved[2]) == 1 && after.count(preserved[5]) == 1,
        "cold fallback performs its normal endpoint pruning and preserves concurrent future rows");
}

void unsupported_scope(int scope, double reference_clock = .125) {
  auto p = options();
  // Exercise the same physical motion in two reference-clock charts. The
  // legacy exclusive lower-window pruning leaves an 11th node at the spacing
  // boundary; timestamp subtraction roundoff must not discard that node.
  p.camera_imu_dt[0] = reference_clock;
  if (scope == 0) p.state_options.do_calib_camera_timeoffset = true;
  if (scope == 1) {
    p.camera_readout_time[0] = .005; p.camera_shutter_rolling[0] = true;
    bool rejected = false;
    try { Manager invalid(p); }
    catch (const std::invalid_argument &error) {
      rejected = std::string(error.what()).find("requires global-shutter") != std::string::npos;
    }
    check(rejected, "physical manager intentionally rejects rolling-shutter configuration before initialization");
    return;
  }
  if (scope == 3) p.init_options.init_warmstart_inject = false;
  Manager manager(p); manager.prime();
  if (scope == 2) manager.arm_prior();
  manager.attempt();
  // This observation arrives after the detached input was captured.
  manager.add(9999,0,12.5-manager.get_state()->cam_imu_dt(0));
  const auto before = rows(manager.database());
  manager.finish_worker_only();
  check(rows(manager.database()) == before && !manager.get_state()->_imu_endpoint_valid,
        "unsupported warm scope still isolates the actual cold solve until consumer commit");
  const bool accepted = manager.attempt();
  std::printf("COLD_SCOPE scope=%d reference=%g accepted=%d owners=%zu episode=%llu\n",scope,reference_clock,accepted,manager.get_state()->clone_count(),
              static_cast<unsigned long long>(manager.get_state()->_initialization_episode_id));
  check(accepted && manager.success(), "supported estimated clocks and unsupported warm scopes retain successful initialization");
  if (!accepted) return;
  const auto state = manager.get_state();
  if (scope == 0) {
    check(state->_initialization_episode_id != 0 && state->clone_count() > 0 && state->_clones_IMU.empty() &&
          StateHelper::get_full_covariance(state).block(0,state->cam_imu_dt_var(0)->id(),15,1).norm() > 1e-10,
          "fresh estimated clocks install a physical window with actual navigation/clock cross covariance");
  } else {
    check(state->_initialization_episode_id == 0 && state->clone_count() == 0 && state->_clones_IMU.empty() &&
          StateHelper::valid_initial_covariance(StateHelper::get_full_covariance(state),true),
          "unsupported physical warm scope cannot silently install a zero-sensitivity conditional window");
  }
  check(rows(manager.database()).count(key(9999,0,12.5-state->cam_imu_dt(0))) == 1,
        "unsupported-scope cold fallback preserves future asynchronous measurements");
}
} // namespace

int main() {
  ov_core::Printer::setPrintLevel(ov_core::Printer::WARNING);
  try {
    fresh_boot();
    rejected_export();
    for (int scope = 0; scope < 4; ++scope) unsupported_scope(scope);
    unsupported_scope(0,0.);
  } catch (const std::exception &error) {
    check(false,error.what());
  }
  std::printf("PHYSICAL_WARM_MANAGER %s checks=%d failures=%d\n",failures ? "FAIL" : "PASS",checks,failures);
  return failures ? 1 : 0;
}
