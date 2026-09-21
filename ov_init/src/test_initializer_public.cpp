/*
 * Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 * End-to-end initialization without a tracker or device. Camera observations
 * and IMU readings come from an analytic trajectory, independently of CPI.
 * Optional argv[1] writes the returned state/covariance for historical A/B.
 */
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "init/InertialInitializer.h"
#include "types/IMU.h"
#include "types/Landmark.h"
#include "types/PoseJPL.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace {
int failures = 0;
void check(bool ok, const char *message) {
  if (!ok) { std::printf("[FAIL] %s\n", message); ++failures; }
}
Eigen::Matrix3d rotation(double t, bool moving) {
  using A = Eigen::AngleAxisd;
  if (!moving) return A(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();
  return (A(.28*std::sin(1.2*t), Eigen::Vector3d::UnitZ()) *
          A(.45+.32*std::sin(2.2*t), Eigen::Vector3d::UnitY()) *
          A(M_PI+.22*std::sin(1.7*t), Eigen::Vector3d::UnitX())).toRotationMatrix();
}
Eigen::Vector3d position(double t) { return {.18*std::sin(1.7*t), .13*(std::cos(2.1*t)-1.0), .10*std::sin(1.3*t)}; }
Eigen::Vector3d velocity(double t) { return {.306*std::cos(1.7*t), -.273*std::sin(2.1*t), .13*std::cos(1.3*t)}; }
Eigen::Vector3d acceleration(double t) { return {-.5202*std::sin(1.7*t), -.5733*std::cos(2.1*t), -.169*std::sin(1.3*t)}; }

struct Result {
  bool ok = false;
  double timestamp = -1, seconds = 0;
  Eigen::VectorXd state;
  Eigen::MatrixXd covariance;
  size_t clones = 0;
  std::uint64_t input_hash = 14695981039346656037ull;
};

void hash_value(std::uint64_t &hash, double value) {
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 8; ++i) { hash ^= (bits >> (8*i)) & 0xff; hash *= 1099511628211ull; }
}

Result run(bool moving, bool fisheye, int cameras, bool warm, bool reset_prior, int threads, double angle_gate = 85.0,
           bool expected_success = true) {
  ov_init::InertialInitializerOptions options;
  options.init_window_time = 2.0;
  options.init_max_features = 80;
  options.init_max_disparity = 1.0;
  options.init_dyn_use = true;
  options.init_dyn_mle_max_iter = 80;
  options.init_dyn_mle_max_threads = threads;
  options.init_dyn_mle_max_time = 20.0; // Numerical test: no machine-load-dependent iterate.
  options.init_dyn_num_pose = 11;
  options.init_gravity_max_angle = angle_gate;
  options.init_warmstart_inject = warm;
  options.num_cameras = cameras;
  const Eigen::Vector3d bg(.008, -.005, .004), ba(.012, -.009, .006);
  // Modest deliberate seed error exercises the MLE, not only its linear seed.
  options.init_dyn_bias_g = bg + Eigen::Vector3d(.001, -.001, .0005);
  options.init_dyn_bias_a = ba + Eigen::Vector3d(.005, -.004, .002);
  options.sigma_w = .001; options.sigma_wb = .0001;
  options.sigma_a = .01; options.sigma_ab = .001;
  options.sigma_pix = 1.0;
  Eigen::Matrix<double, 8, 1> intr;
  intr << 458.0, 463.0, 640.0, 400.0, .06, -.01, .002, -.0005;
  const Eigen::Matrix3d Ric = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();
  for (int c = 0; c < cameras; ++c) {
    std::shared_ptr<ov_core::CamBase> cam;
    if (fisheye) cam = std::make_shared<ov_core::CamEqui>(1280, 800);
    else cam = std::make_shared<ov_core::CamRadtan>(1280, 800);
    cam->set_value(intr);
    options.camera_intrinsics[c] = cam;
    Eigen::VectorXd ext(7);
    ext.head<4>() = ov_core::rot_2_quat(Ric);
    ext.tail<3>() = Eigen::Vector3d(.12*c, 0, 0);
    options.camera_extrinsics[c] = ext;
  }
  auto database = std::make_shared<ov_core::FeatureDatabase>();
  ov_init::InertialInitializer initializer(options, database);
  Result result;
  if (reset_prior) {
    ov_init::ResetBiasPrior prior;
    prior.bg = bg; prior.ba = ba;
    prior.sigma_bg.setConstant(.006); prior.sigma_ba.setConstant(.025);
    prior.t_snapshot = 10.0; prior.valid = true;
    initializer.set_reset_prior(prior);
  }
  // 800 Hz IMU with a bracket before/after the complete image window.
  for (int i = -160; i <= 1936; ++i) {
    const double t = i/800.0;
    ov_core::ImuData imu;
    imu.timestamp = 10.0+t;
    if (moving) {
      const double h = 1e-5;
      imu.wm = -ov_core::log_so3(rotation(t+h, true)*rotation(t-h, true).transpose())/(2*h)+bg;
      imu.am = rotation(t, true)*(acceleration(t)+Eigen::Vector3d(0, 0, options.gravity_mag))+ba;
    } else {
      imu.wm = bg;
      imu.am = rotation(t, false)*Eigen::Vector3d(0, 0, options.gravity_mag);
    }
    initializer.feed_imu(imu);
    hash_value(result.input_hash, imu.timestamp);
    for (int k = 0; k < 3; ++k) { hash_value(result.input_hash, imu.wm(k)); hash_value(result.input_hash, imu.am(k)); }
  }
  for (int frame = 0; frame <= 72; ++frame) {
    const double t = frame/30.0;
    for (int c = 0; c < cameras; ++c) {
      for (int f = 0; f < options.init_max_features; ++f) {
        const Eigen::Vector3d point(-1.6+.4*(f%9), -1.2+.4*((f/9)%7), 4.0+.37*(f%11));
        const Eigen::Vector3d camera_point = Ric*rotation(t, moving)*(point-(moving ? position(t) : Eigen::Vector3d::Zero())) +
                                            options.camera_extrinsics.at(c).tail<3>();
        const Eigen::Vector2f normalized = (camera_point.head<2>()/camera_point.z()).cast<float>();
        Eigen::Vector2f pixel = options.camera_intrinsics.at(c)->distort_f(normalized);
        if (moving) {
          pixel.x() += .02*std::sin(1.7*f+.6*frame);
          pixel.y() += .02*std::cos(.8*f+.9*frame);
        }
        const Eigen::Vector2f tracked = options.camera_intrinsics.at(c)->undistort_f(pixel);
        database->update_feature(f, 10.0+t, c, pixel.x(), pixel.y(), tracked.x(), tracked.y());
        for (double value : {double(f), 10.0+t, double(c), double(pixel.x()), double(pixel.y()), double(tracked.x()), double(tracked.y())})
          hash_value(result.input_hash, value);
      }
    }
  }
  std::shared_ptr<ov_type::IMU> imu = std::make_shared<ov_type::IMU>();
  std::vector<std::shared_ptr<ov_type::Type>> order;
  std::map<double, std::shared_ptr<ov_type::PoseJPL>> clones;
  std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> landmarks;
  const auto start = std::chrono::steady_clock::now();
  result.ok = initializer.initialize(result.timestamp, result.covariance, order, imu, clones, landmarks, moving);
  result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  result.state = imu->value(); result.clones = clones.size();
  check(result.ok == expected_success, "public initializer expected acceptance");
  if (!result.ok) return result;
  const int expected_size = warm && moving ? 15+6*static_cast<int>(clones.size()) : 15;
  check(result.covariance.rows() == expected_size && result.covariance.cols() == expected_size, "covariance layout");
  check(result.state.allFinite() && result.covariance.allFinite(), "finite state and covariance");
  check((imu->value()-imu->fej()).norm() == 0.0, "initialized IMU FEJ equals value");
  check(!order.empty() && order.front() == imu, "IMU first in covariance ordering");
  check(order.size() == (warm && moving ? 1+clones.size() : 1), "clone covariance ordering size");
  check(result.covariance.diagonal().minCoeff() > 0.0, "positive covariance diagonal");
  check((result.covariance-result.covariance.transpose()).norm() < 1e-8*result.covariance.norm(), "covariance symmetric");
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(result.covariance);
  check(eig.info() == Eigen::Success && eig.eigenvalues().minCoeff() >= -1e-9*eig.eigenvalues().maxCoeff(), "joint covariance positive semidefinite");
  size_t oi = 1;
  for (const auto &clone : clones) {
    check((clone.second->value()-clone.second->fej()).norm() == 0.0, "clone FEJ equals value");
    if (warm && moving) check(order.at(oi++) == clone.second, "clones ordered by ascending timestamp");
  }
  const double t = result.timestamp-10.0;
  const double tilt_error = (imu->Rot()*Eigen::Vector3d::UnitZ()-rotation(t, moving)*Eigen::Vector3d::UnitZ()).norm();
  Eigen::Vector3d true_body_velocity = Eigen::Vector3d::Zero();
  if (moving) true_body_velocity = rotation(t, true)*velocity(t);
  const double velocity_error = (imu->Rot()*imu->vel()-true_body_velocity).norm();
  check(tilt_error < .02, "gravity direction within 1.2 degrees of analytic truth");
  check(velocity_error < .04, "body velocity within 4 cm/s of analytic truth");
  check((imu->bias_g()-bg).norm() < .005, "gyro bias within 5 mrad/s of analytic truth");
  std::printf("PUBLIC_INIT moving=%d fisheye=%d cameras=%d warm=%d reset=%d threads=%d wall_s=%.9f tilt_error=%.9g velocity_error=%.9g covariance_trace=%.12g clones=%zu\n",
              moving, fisheye, cameras, warm, reset_prior, threads, result.seconds, tilt_error, velocity_error, result.covariance.trace(), result.clones);
  return result;
}

void dump(std::ostream &out, const char *name, const Result &result) {
  out << name << " " << result.ok << " " << result.timestamp << " " << result.clones << " " << result.input_hash << "\n";
  out << result.state.transpose() << "\n" << result.covariance << "\n";
}
} // namespace

int main(int argc, char **argv) {
  ov_core::Printer::setPrintLevel("WARNING");
  std::ofstream receipt;
  if (argc > 1) {
    receipt.open(argv[1]);
    if (!receipt.is_open()) { std::perror("initializer receipt"); return EXIT_FAILURE; }
    receipt << std::setprecision(17);
  }
  const int repeats = argc > 2 ? std::max(1, std::atoi(argv[2])) : 1;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    const Result stationary = run(false, true, 2, false, false, 4);
    const Result radtan = run(true, false, 1, false, false, 4);
    const Result fisheye = run(true, true, 2, false, false, 4);
    const Result warm = run(true, true, 2, true, false, 4);
    const Result reset = run(true, true, 2, true, true, 4);
    const Result narrow_gate = run(true, true, 2, true, false, 4, 15.0, false);
    if (fisheye.ok && warm.ok) {
      check((fisheye.state-warm.state).norm() < 1e-10, "warmstart preserves the IMU estimate");
      check((fisheye.covariance-warm.covariance.topLeftCorner(15,15)).norm() < 1e-10, "warmstart preserves the IMU covariance");
    }
    if (receipt.is_open() && repeat == 0) {
      dump(receipt, "static", stationary); dump(receipt, "radtan", radtan);
      dump(receipt, "fisheye", fisheye); dump(receipt, "warm", warm); dump(receipt, "reset", reset);
      dump(receipt, "narrow_gate", narrow_gate);
    }
  }
  std::printf("Public initializer checks: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
