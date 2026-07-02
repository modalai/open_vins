/*
 * R2 gate: end-to-end A/B of the Stage-1 linear seed methods on the REAL
 * DynamicInitializer::initialize (feature DB + IMU stream in, EKF seed out).
 * Contributor: Joao Leonardo Silva Cotta (@zauberflote1)
 *
 *   method 0 = Dong-Si (features-in-state, legacy)
 *   method 1 = feature-less epipolar-normal 6x6 [v,g] seed (sqrtVINS Stage-A style)
 *   method 2 = feature-less with in-call Dong-Si fallback
 *
 * The sim integrates the ground-truth trajectory from the CPI recursion itself (IMU residual
 * is exactly zero at GT -- same construction as bench_zbft_s2 / test_init_consistency), places
 * landmarks in front of the camera, and projects pixel-noise observations at exactly the pose
 * times the initializer will select. Metrics: success rate, gravity-in-IMU angle error, |v|
 * error, bias errors, wall time.
 *
 * Build (host; OpenCV headers only via ov_core sensor_data.h; USE_CERES_FREE_INIT selects zbft):
 *   OVC=../../ov_core/src
 *   g++ -O2 -std=c++17 -pthread -DUSE_CERES_FREE_INIT -I/usr/include/eigen3 -I/usr/include/opencv4 \
 *       -I. -I$OVC test_linear_seed.cpp dynamic/DynamicInitializer.cpp \
 *       ceres_free/Problem.cpp ceres_free/Parallel.cpp ceres_free/State_JPLQuatLocal.cpp \
 *       ceres_free/Factor_ImuCPIv1.cpp ceres_free/Factor_GenericPrior.cpp \
 *       ceres_free/Factor_ImageReprojCalib.cpp \
 *       $OVC/cpi/CpiV1.cpp $OVC/feat/Feature.cpp $OVC/feat/FeatureDatabase.cpp \
 *       $OVC/feat/FeatureInitializer.cpp $OVC/utils/print.cpp \
 *       -lpthread -o /tmp/test_linear_seed && /tmp/test_linear_seed 10 2.0 12
 *
 * Usage: test_linear_seed [K_trials=10] [window_s=2.0] [num_pose=12]
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "cam/CamRadtan.h"
#include "cpi/CpiV1.h"
#include "feat/FeatureDatabase.h"
#include "types/IMU.h"
#include "types/Landmark.h"
#include "types/PoseJPL.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"

#include "dynamic/DynamicInitializer.h"
#include "init/InertialInitializerOptions.h"

using namespace ov_core;
using namespace ov_init;

static Eigen::Vector3d randn3(std::mt19937 &rng, double s) {
  std::normal_distribution<double> N(0, 1);
  return Eigen::Vector3d(s * N(rng), s * N(rng), s * N(rng));
}

struct SimOut {
  std::shared_ptr<FeatureDatabase> db;
  std::shared_ptr<std::vector<ImuData>> imu;
  // GT at the NEWEST camera time (the state the initializer reports)
  Eigen::Matrix3d R_WtoI_gt; // world (gravity-aligned, sim frame) to newest IMU
  Eigen::Vector3d v_gt_W;    // in the sim world frame
  Eigen::Vector3d bg_gt, ba_gt;
  Eigen::Vector3d grav_W = Eigen::Vector3d(0, 0, -9.81); // z-DOWN IMU mount (VOXL): reaction/g-var points -z, matching the +Z... -Z direction gate
};

// Simulate: trajectory integrated from the CPI outputs themselves, landmarks projected at the
// exact pose times, observations at pixel-noise level fed into a FeatureDatabase.
static SimOut make_sim(std::mt19937 &rng, const InertialInitializerOptions &params, double window_s, int n_pose, int n_feats) {
  SimOut out;
  out.db = std::make_shared<FeatureDatabase>();
  out.imu = std::make_shared<std::vector<ImuData>>();
  out.bg_gt = randn3(rng, 0.01);
  out.ba_gt = randn3(rng, 0.05);

  const double imu_hz = 500.0;
  const double t0 = 10.0; // arbitrary epoch
  const double margin = 0.3;
  const double dt_pose = window_s / (double)n_pose;
  std::vector<double> pose_times;
  for (int i = 0; i <= n_pose; i++)
    pose_times.push_back(t0 + i * dt_pose);
  const double t_end = pose_times.back();

  // IMU signal (sinusoidal body rates + specific force), biases added
  std::uniform_real_distribution<double> freq(0.8, 2.2), amp(0.5, 1.1);
  const double wx_f = freq(rng), wy_f = freq(rng), wz_f = freq(rng);
  const double wx_a = amp(rng), wy_a = amp(rng), wz_a = amp(rng);
  auto w_body = [&](double t) {
    return Eigen::Vector3d(wx_a * std::sin(wx_f * (t - t0)), wy_a * std::sin(wy_f * (t - t0) + 1.0),
                           wz_a * std::sin(wz_f * (t - t0) + 2.0));
  };
  auto a_body = [&](double t) {
    return Eigen::Vector3d(0.6 * std::sin(0.9 * (t - t0)), 0.4 * std::cos(1.3 * (t - t0)), 0.3 * std::sin(0.6 * (t - t0)));
  };
  // The IMU measures specific force f = R_WtoI*(a_W - g_W)... we CONSTRUCT the world trajectory
  // from the CPI recursion, so define the measurement first and integrate it (residual-free GT).
  for (double t = t0 - margin; t <= t_end + margin + 1e-9; t += 1.0 / imu_hz) {
    ImuData m;
    m.timestamp = t;
    m.wm = w_body(t) + out.bg_gt;
    // z-down mount: hovering accelerometer reads -9.81 on body z (specific force = -g_true);
    // adding it keeps the CPI-integrated trajectory bounded (hover-ish with wiggle)
    m.am = a_body(t) + Eigen::Vector3d(0, 0, -9.81) + out.ba_gt;
    out.imu->push_back(m);
  }

  // Integrate GT states at pose times via CPI (values-only), from identity/zero initial state
  std::vector<Eigen::Matrix3d> R_WtoI(pose_times.size());
  std::vector<Eigen::Vector3d> p_W(pose_times.size()), v_W(pose_times.size());
  R_WtoI[0] = Eigen::Matrix3d::Identity();
  p_W[0] = Eigen::Vector3d::Zero();
  v_W[0] = randn3(rng, 0.2);
  for (size_t k = 1; k < pose_times.size(); k++) {
    CpiV1 cpi(params.sigma_w, params.sigma_wb, params.sigma_a, params.sigma_ab, true);
    cpi.setLinearizationPoints(out.bg_gt, out.ba_gt);
    // feed IMU in [pose_times[k-1], pose_times[k]]
    for (size_t i = 0; i + 1 < out.imu->size(); i++) {
      const double ta = out.imu->at(i).timestamp, tb = out.imu->at(i + 1).timestamp;
      if (tb <= pose_times[k - 1] || ta >= pose_times[k])
        continue;
      // clamp to the interval with linear interpolation at the boundaries
      ImuData m0 = out.imu->at(i), m1 = out.imu->at(i + 1);
      auto interp = [&](double t) {
        const double lam = (t - ta) / (tb - ta);
        ImuData mm;
        mm.timestamp = t;
        mm.wm = (1 - lam) * m0.wm + lam * m1.wm;
        mm.am = (1 - lam) * m0.am + lam * m1.am;
        return mm;
      };
      if (m0.timestamp < pose_times[k - 1])
        m0 = interp(pose_times[k - 1]);
      if (m1.timestamp > pose_times[k])
        m1 = interp(pose_times[k]);
      cpi.feed_IMU(m0.timestamp, m1.timestamp, m0.wm, m0.am, m1.wm, m1.am);
    }
    const double DT = cpi.DT;
    p_W[k] = p_W[k - 1] + v_W[k - 1] * DT - 0.5 * out.grav_W * DT * DT + R_WtoI[k - 1].transpose() * cpi.alpha_tau;
    v_W[k] = v_W[k - 1] - out.grav_W * DT + R_WtoI[k - 1].transpose() * cpi.beta_tau;
    R_WtoI[k] = cpi.R_k2tau * R_WtoI[k - 1];
  }
  out.R_WtoI_gt = R_WtoI.back();
  out.v_gt_W = v_W.back();

  // Landmarks: in front of cam0 (R_ItoC from params) at the FIRST pose, depths 3..8 m
  const Eigen::Matrix3d R_ItoC = quat_2_Rot(params.camera_extrinsics.at(0).block(0, 0, 4, 1));
  const Eigen::Vector3d p_IinC = params.camera_extrinsics.at(0).block(4, 0, 3, 1);
  std::uniform_real_distribution<double> Ux(-1.5, 1.5), Uz(3.0, 8.0);
  std::vector<Eigen::Vector3d> lms(n_feats);
  for (int j = 0; j < n_feats; j++) {
    Eigen::Vector3d p_C(Ux(rng), Ux(rng), Uz(rng));
    lms[j] = R_WtoI[0].transpose() * (R_ItoC.transpose() * (p_C - p_IinC)) + p_W[0];
  }

  // Observations at every pose time (pixel noise), fed into the database
  const double fx = 320, fy = 320, cx = 320, cy = 240;
  std::normal_distribution<double> Npx(0, 1.0);
  for (size_t k = 0; k < pose_times.size(); k++) {
    for (int j = 0; j < n_feats; j++) {
      const Eigen::Vector3d p_C = R_ItoC * (R_WtoI[k] * (lms[j] - p_W[k])) + p_IinC;
      if (p_C(2) < 0.3)
        continue;
      const double un = p_C(0) / p_C(2) + Npx(rng) / fx;
      const double vn = p_C(1) / p_C(2) + Npx(rng) / fy;
      out.db->update_feature((size_t)(j + 1), pose_times[k], 0, (float)(fx * un + cx), (float)(fy * vn + cy), (float)un, (float)vn);
    }
  }
  return out;
}

int main(int argc, char **argv) {
  const int K = (argc > 1) ? atoi(argv[1]) : 10;
  const double window_s = (argc > 2) ? atof(argv[2]) : 2.0;
  const int n_pose = (argc > 3) ? atoi(argv[3]) : 12;
  const int n_feats = 60;
  ov_core::Printer::setPrintLevel("WARNING");

  std::printf("====== Stage-1 linear seed A/B (REAL DynamicInitializer) ======\n");
  std::printf("Trials: %d | window %.2fs | %d poses | %d feats\n\n", K, window_s, n_pose, n_feats);

  for (int method : {0, 1, 2}) {
    int ok_count = 0;
    double sum_ms = 0, sum_gerr = 0, sum_verr = 0, sum_bg = 0, sum_ba = 0;
    for (int trial = 0; trial < K; trial++) {
      std::mt19937 rng(7000 + trial);

      InertialInitializerOptions params;
      params.init_window_time = window_s;
      params.init_dyn_num_pose = n_pose;
      params.init_max_features = n_feats;
      params.gravity_mag = 9.81;
      params.sigma_w = 1e-3;
      params.sigma_wb = 1e-4;
      params.sigma_a = 1e-2;
      params.sigma_ab = 1e-3;
      params.sigma_pix = 1.0;
      params.init_dyn_min_deg = 1.0;
      params.init_gravity_max_angle = 85.0;
      params.init_dyn_mle_max_iter = 30;
      params.init_dyn_mle_max_time = 5.0;
      params.init_dyn_mle_max_threads = 1;
      params.init_dyn_grav_prior_sigma = 0.5;
      params.init_dyn_grav_gate_deg = 89.0; // sim attitude is arbitrary; disable the +Z gate
      params.init_warmstart_inject = false;
      params.init_dyn_linear_method = method;
      params.num_cameras = 1;
      // cam0: identity extrinsics, radtan zero distortion
      Eigen::VectorXd ext = Eigen::VectorXd::Zero(7);
      ext(3) = 1.0; // q_ItoC = identity (x,y,z,w)
      params.camera_extrinsics[0] = ext;
      auto cam = std::make_shared<ov_core::CamRadtan>(640, 480);
      Eigen::MatrixXd intr = Eigen::MatrixXd::Zero(8, 1);
      intr << 320, 320, 320, 240, 0, 0, 0, 0;
      cam->set_value(intr);
      params.camera_intrinsics[0] = cam;

      SimOut sim = make_sim(rng, params, window_s, n_pose, n_feats);

      DynamicInitializer init(params, sim.db, sim.imu);
      double timestamp = -1;
      Eigen::MatrixXd covariance;
      std::vector<std::shared_ptr<ov_type::Type>> order;
      auto imu_t = std::make_shared<ov_type::IMU>();
      std::map<double, std::shared_ptr<ov_type::PoseJPL>> clones;
      std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> feats_slam;

      const auto tt0 = std::chrono::steady_clock::now();
      const bool ok = init.initialize(timestamp, covariance, order, imu_t, clones, feats_slam);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tt0).count();
      sum_ms += ms;
      if (!ok)
        continue;
      ok_count++;

      // Gravity direction in the newest IMU frame: est vs GT (init world has gravity = +Z)
      const Eigen::Matrix3d R_GtoI_est = quat_2_Rot(imu_t->quat());
      const Eigen::Vector3d g_inI_est = R_GtoI_est * Eigen::Vector3d(0, 0, 1);
      const Eigen::Vector3d g_inI_gt = (sim.R_WtoI_gt * sim.grav_W).normalized();
      const double gerr = std::acos(std::min(1.0, std::max(-1.0, g_inI_est.dot(g_inI_gt)))) * 180.0 / M_PI;
      sum_gerr += gerr;
      sum_verr += std::abs(imu_t->vel().norm() - sim.v_gt_W.norm());
      sum_bg += (imu_t->bias_g() - sim.bg_gt).norm();
      sum_ba += (imu_t->bias_a() - sim.ba_gt).norm();
    }
    const double iN = (ok_count > 0) ? 1.0 / ok_count : 0.0;
    std::printf("method %d: success %2d/%2d | time %7.2f ms | grav-in-I err %7.4f deg | |v| err %.4f | bg %.4f | ba %.4f\n", method,
                ok_count, K, sum_ms / K, sum_gerr * iN, sum_verr * iN, sum_bg * iN, sum_ba * iN);
  }
  std::printf("\n====== DONE ======\n");
  return 0;
}
