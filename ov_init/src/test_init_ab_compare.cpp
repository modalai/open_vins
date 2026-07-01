/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Contributor: Joao Leonardo Silva Cotta (@zauberflote1)
 *
 * A/B comparison test: Ceres-based MLE vs ceres-free (ov_init::zbft_sfm) MLE.
 * ----------------------------------------------------------------------------
 * This ROS-free test generates a synthetic VIO window and runs BOTH
 * initialization backends (using the EXACT same factor implementations)
 * on the SAME data to compare timing, accuracy, cost, and covariance.
 *
 * Compile:
 *   cd ov_init/src
 *   OVC=../../ov_core/src
 *   g++ -O2 -std=c++17 -pthread -I/usr/include/eigen3 -I. -I$OVC \
 *       $(pkg-config --cflags opencv4) \
 *       test_init_ab_compare.cpp \
 *       ceres_free/Problem.cpp ceres_free/Parallel.cpp \
 *       ceres_free/State_JPLQuatLocal.cpp ceres_free/Factor_ImuCPIv1.cpp \
 *       ceres_free/Factor_GenericPrior.cpp ceres_free/Factor_ImageReprojCalib.cpp \
 *       ceres/State_JPLQuatLocal.cpp ceres/Factor_ImuCPIv1.cpp \
 *       ceres/Factor_GenericPrior.cpp ceres/Factor_ImageReprojCalib.cpp \
 *       $OVC/cpi/CpiV1.cpp \
 *       -lceres -lglog -lpthread $(pkg-config --libs opencv4) \
 *       -o /tmp/test_init_ab && /tmp/test_init_ab
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <Eigen/Dense>

// ov_core
#include "cpi/CpiV1.h"
#include "utils/quat_ops.h"

// Ceres backend (uses exact same factor code)
#include <ceres/ceres.h>
#include "ceres/Factor_GenericPrior.h"
#include "ceres/Factor_ImageReprojCalib.h"
#include "ceres/Factor_ImuCPIv1.h"
#include "ceres/State_JPLQuatLocal.h"

// ceres-free backend (uses exact same factor code, different solver)
#include "ceres_free/Factor_GenericPrior.h"
#include "ceres_free/Factor_ImageReprojCalib.h"
#include "ceres_free/Factor_ImuCPIv1.h"
#include "ceres_free/LocalParameterization.h"
#include "ceres_free/LossFunction.h"
#include "ceres_free/Problem.h"
#include "ceres_free/State_JPLQuatLocal.h"

using namespace ov_core;

// ===========================================================================
// Test configuration
// ===========================================================================
struct TestConfig {
  int num_poses = 10;
  double dt_pose = 0.1;
  int num_features = 50;
  int obs_per_feature = 5;

  double sigma_w = 0.005;
  double sigma_wb = 0.0001;
  double sigma_a = 0.05;
  double sigma_ab = 0.001;
  double sigma_pix = 1.0;

  double imu_rate = 200.0;
  double gravity_mag = 9.81;

  double fx = 400.0, fy = 400.0, cx = 320.0, cy = 240.0;
  Eigen::Vector4d q_ItoC = Eigen::Vector4d(0, 0, 0, 1);
  Eigen::Vector3d p_IinC = Eigen::Vector3d::Zero();

  int max_iterations = 100;
  int num_threads = 4;
  double max_solver_time = 10.0;
};

// ===========================================================================
// Ground truth state
// ===========================================================================
struct GroundTruthState {
  double timestamp;
  Eigen::Vector4d q_GtoI;
  Eigen::Vector3d p_IinG;
  Eigen::Vector3d v_IinG;
  Eigen::Vector3d bias_g;
  Eigen::Vector3d bias_a;
};

// ===========================================================================
// Synthetic data generation
// ===========================================================================
class SyntheticWindow {
public:
  std::vector<GroundTruthState> gt_states;
  std::vector<std::shared_ptr<CpiV1>> cpis;
  std::vector<Eigen::Vector3d> features_inG;
  std::vector<std::vector<std::pair<int, Eigen::Vector2d>>> observations;
  Eigen::Vector3d gravity;

  void generate(const TestConfig &cfg, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> noise_w(0.0, cfg.sigma_w);
    std::normal_distribution<double> noise_a(0.0, cfg.sigma_a);
    std::normal_distribution<double> noise_pix(0.0, cfg.sigma_pix);
    std::uniform_real_distribution<double> depth_dist(2.0, 10.0);

    gravity << 0.0, 0.0, cfg.gravity_mag;

    // Trajectory is DERIVED by integrating the CPIs below (not imposed analytically), so the
    // IMU-factor residual is exactly zero at ground truth by construction. Body-frame angular
    // velocity + specific force are smooth and time-varying; specific force ~ +g along body-z
    // (a tilting hover) keeps the short window bounded and features in view while the rotation
    // makes the 2 gravity-tilt DOF observable.
    gt_states.resize(cfg.num_poses);
    Eigen::Vector3d bias_g(0.01, -0.005, 0.008);
    Eigen::Vector3d bias_a(0.05, -0.03, 0.02);
    auto omega_body = [](double t) {
      return Eigen::Vector3d(0.5 * std::sin(1.3 * t), 0.6 * std::sin(1.1 * t + 0.5), 0.4 * std::sin(0.9 * t + 1.0));
    };
    auto accel_body = [&](double t) {
      return Eigen::Vector3d(0.6 * std::sin(0.7 * t), 0.4 * std::cos(0.9 * t), cfg.gravity_mag + 0.3 * std::sin(0.6 * t));
    };

    gt_states[0].timestamp = 0.0;
    gt_states[0].q_GtoI = Eigen::Vector4d(0, 0, 0, 1);
    gt_states[0].p_IinG = Eigen::Vector3d::Zero();
    gt_states[0].v_IinG = Eigen::Vector3d(0.1, -0.05, 0.0);
    gt_states[0].bias_g = bias_g;
    gt_states[0].bias_a = bias_a;

    // Integrate the CPI between consecutive camera times; derive GT from the preintegration
    // (the exact kinematics Factor_ImuCPIv1 uses => zero IMU residual at GT).
    cpis.resize(cfg.num_poses);
    cpis[0] = nullptr;
    const double dt_imu = 1.0 / cfg.imu_rate;
    for (int i = 1; i < cfg.num_poses; i++) {
      auto cpi = std::make_shared<CpiV1>(cfg.sigma_w, cfg.sigma_wb, cfg.sigma_a, cfg.sigma_ab, true);
      cpi->setLinearizationPoints(bias_g, bias_a);
      double t0 = (i - 1) * cfg.dt_pose;
      double t1 = i * cfg.dt_pose;
      for (double t = t0; t < t1 - 1e-9; t += dt_imu) {
        double tn = std::min(t + dt_imu, t1);
        Eigen::Vector3d wm0 = omega_body(t) + bias_g, am0 = accel_body(t) + bias_a;
        Eigen::Vector3d wm1 = omega_body(tn) + bias_g, am1 = accel_body(tn) + bias_a;
        cpi->feed_IMU(t, tn, wm0, am0, wm1, am1);
      }
      cpis[i] = cpi;

      const Eigen::Matrix3d R_GtoIm1 = quat_2_Rot(gt_states[i - 1].q_GtoI);
      const double DT = cpi->DT;
      gt_states[i].timestamp = t1;
      gt_states[i].q_GtoI = quatnorm(quat_multiply(cpi->q_k2tau, gt_states[i - 1].q_GtoI));
      gt_states[i].v_IinG = gt_states[i - 1].v_IinG - gravity * DT + R_GtoIm1.transpose() * cpi->beta_tau;
      gt_states[i].p_IinG = gt_states[i - 1].p_IinG + gt_states[i - 1].v_IinG * DT - 0.5 * gravity * DT * DT +
                            R_GtoIm1.transpose() * cpi->alpha_tau;
      gt_states[i].bias_g = bias_g;
      gt_states[i].bias_a = bias_a;
    }

    // Generate 3D features - distributed across the trajectory
    features_inG.resize(cfg.num_features);
    Eigen::Matrix3d R_ItoC_feat = quat_2_Rot(cfg.q_ItoC);

    for (int f = 0; f < cfg.num_features; f++) {
      // Pick a reference pose for this feature (spread across all poses)
      int ref_pose = f % cfg.num_poses;
      double depth = depth_dist(gen);

      // Generate feature in camera FOV (normalized image coords [-0.4, 0.4])
      std::uniform_real_distribution<double> uv_dist(-0.4, 0.4);
      double u_norm = uv_dist(gen);
      double v_norm = uv_dist(gen);

      Eigen::Vector3d p_FinC(u_norm * depth, v_norm * depth, depth);
      Eigen::Matrix3d R_GtoI = quat_2_Rot(gt_states[ref_pose].q_GtoI);
      Eigen::Vector3d p_IinG = gt_states[ref_pose].p_IinG;

      Eigen::Vector3d p_FinI = R_ItoC_feat.transpose() * (p_FinC - cfg.p_IinC);
      features_inG[f] = R_GtoI.transpose() * p_FinI + p_IinG;
    }

    // Generate observations - all features visible from all poses if in FOV
    observations.resize(cfg.num_poses);
    Eigen::Matrix3d R_ItoC = quat_2_Rot(cfg.q_ItoC);

    for (int i = 0; i < cfg.num_poses; i++) {
      Eigen::Matrix3d R_GtoI = quat_2_Rot(gt_states[i].q_GtoI);
      Eigen::Vector3d p_IinG = gt_states[i].p_IinG;

      for (int f = 0; f < cfg.num_features; f++) {
        Eigen::Vector3d p_FinI = R_GtoI * (features_inG[f] - p_IinG);
        Eigen::Vector3d p_FinC = R_ItoC * p_FinI + cfg.p_IinC;

        // Check if in front of camera and within reasonable depth
        if (p_FinC(2) < 0.5 || p_FinC(2) > 20.0)
          continue;

        double u = cfg.fx * p_FinC(0) / p_FinC(2) + cfg.cx;
        double v = cfg.fy * p_FinC(1) / p_FinC(2) + cfg.cy;

        // Check if in image with margin
        if (u < 10 || u > 630 || v < 10 || v > 470)
          continue;

        u += noise_pix(gen);
        v += noise_pix(gen);

        observations[i].push_back({f, Eigen::Vector2d(u, v)});
      }
    }

    printf("[SyntheticWindow] Generated %d poses, %d features\n", cfg.num_poses, cfg.num_features);
    int total_obs = 0;
    for (const auto &obs : observations)
      total_obs += obs.size();
    printf("[SyntheticWindow] Total observations: %d\n", total_obs);
  }
};

// ===========================================================================
// Results structure
// ===========================================================================
struct SolverResults {
  std::string name;
  double time_setup_ms;
  double time_solve_ms;
  double time_cov_ms;
  double initial_cost;
  double final_cost;
  int iterations;
  bool converged;

  Eigen::Vector3d err_ori_deg;
  Eigen::Vector3d err_pos;
  Eigen::Vector3d err_vel;
  Eigen::Vector3d err_bg;
  Eigen::Vector3d err_ba;

  double nees_total;
  double nees_ori;
  double nees_pos;
  double nees_vel;

  Eigen::Matrix<double, 15, 1> cov_diag;
};

// ===========================================================================
// Run Ceres backend
// ===========================================================================
SolverResults runCeres(const SyntheticWindow &data, const TestConfig &cfg) {
  SolverResults res;
  res.name = "Ceres";

  auto t_start = std::chrono::high_resolution_clock::now();

  ceres::Problem problem;

  std::vector<double *> vars_ori(cfg.num_poses);
  std::vector<double *> vars_pos(cfg.num_poses);
  std::vector<double *> vars_vel(cfg.num_poses);
  std::vector<double *> vars_bg(cfg.num_poses);
  std::vector<double *> vars_ba(cfg.num_poses);
  std::vector<double *> vars_feat(cfg.num_features);

  auto *var_calib_ori = new double[4];
  auto *var_calib_pos = new double[3];
  auto *var_calib_cam = new double[8];

  for (int j = 0; j < 4; j++)
    var_calib_ori[j] = cfg.q_ItoC(j);
  for (int j = 0; j < 3; j++)
    var_calib_pos[j] = cfg.p_IinC(j);
  var_calib_cam[0] = cfg.fx;
  var_calib_cam[1] = cfg.fy;
  var_calib_cam[2] = cfg.cx;
  var_calib_cam[3] = cfg.cy;
  for (int j = 4; j < 8; j++)
    var_calib_cam[j] = 0.0;

  problem.AddParameterBlock(var_calib_ori, 4, new ov_init::State_JPLQuatLocal());
  problem.AddParameterBlock(var_calib_pos, 3);
  problem.AddParameterBlock(var_calib_cam, 8);
  problem.SetParameterBlockConstant(var_calib_ori);
  problem.SetParameterBlockConstant(var_calib_pos);
  problem.SetParameterBlockConstant(var_calib_cam);

  std::mt19937 gen(123);
  std::normal_distribution<double> perturb_pos(0.0, 0.001);  // 1mm position perturbation
  std::normal_distribution<double> perturb_vel(0.0, 0.001);  // 1mm/s velocity perturbation
  std::normal_distribution<double> perturb_feat(0.0, 0.01);  // 1cm feature perturbation

  for (int i = 0; i < cfg.num_poses; i++) {
    vars_ori[i] = new double[4];
    vars_pos[i] = new double[3];
    vars_vel[i] = new double[3];
    vars_bg[i] = new double[3];
    vars_ba[i] = new double[3];

    for (int j = 0; j < 4; j++)
      vars_ori[i][j] = data.gt_states[i].q_GtoI(j);
    for (int j = 0; j < 3; j++) {
      vars_pos[i][j] = data.gt_states[i].p_IinG(j) + perturb_pos(gen);
      vars_vel[i][j] = data.gt_states[i].v_IinG(j) + perturb_vel(gen);
      vars_bg[i][j] = data.gt_states[i].bias_g(j);
      vars_ba[i][j] = data.gt_states[i].bias_a(j);
    }

    problem.AddParameterBlock(vars_ori[i], 4, new ov_init::State_JPLQuatLocal());
    problem.AddParameterBlock(vars_pos[i], 3);
    problem.AddParameterBlock(vars_vel[i], 3);
    problem.AddParameterBlock(vars_bg[i], 3);
    problem.AddParameterBlock(vars_ba[i], 3);
  }

  for (int f = 0; f < cfg.num_features; f++) {
    vars_feat[f] = new double[3];
    for (int j = 0; j < 3; j++)
      vars_feat[f][j] = data.features_inG[f](j) + perturb_feat(gen);
    problem.AddParameterBlock(vars_feat[f], 3);
  }

  // Gauge prior
  {
    Eigen::MatrixXd x_lin = Eigen::MatrixXd::Zero(13, 1);
    for (int j = 0; j < 4; j++)
      x_lin(j) = vars_ori[0][j];
    for (int j = 0; j < 3; j++) {
      x_lin(4 + j) = vars_pos[0][j];
      x_lin(7 + j) = vars_bg[0][j];
      x_lin(10 + j) = vars_ba[0][j];
    }
    Eigen::MatrixXd prior_grad = Eigen::MatrixXd::Zero(10, 1);
    Eigen::MatrixXd prior_Info = Eigen::MatrixXd::Identity(10, 10);
    prior_Info.block(0, 0, 4, 4) *= 1.0 / std::pow(1e-5, 2);
    prior_Info.block(4, 4, 3, 3) *= 1.0 / std::pow(0.05, 2);
    prior_Info.block(7, 7, 3, 3) *= 1.0 / std::pow(0.10, 2);

    std::vector<std::string> x_types = {"quat_yaw", "vec3", "vec3", "vec3"};
    std::vector<double *> factor_params = {vars_ori[0], vars_pos[0], vars_bg[0], vars_ba[0]};
    problem.AddResidualBlock(new ov_init::Factor_GenericPrior(x_lin, x_types, prior_Info, prior_grad), nullptr, factor_params);
  }

  // IMU factors
  Eigen::Vector3d gravity = data.gravity;
  for (int i = 1; i < cfg.num_poses; i++) {
    auto cpi = data.cpis[i];
    std::vector<double *> params = {vars_ori[i - 1], vars_bg[i - 1], vars_vel[i - 1], vars_ba[i - 1], vars_pos[i - 1],
                                    vars_ori[i],     vars_bg[i],     vars_vel[i],     vars_ba[i],     vars_pos[i]};
    auto *factor =
        new ov_init::Factor_ImuCPIv1(cpi->DT, gravity, cpi->alpha_tau, cpi->beta_tau, cpi->q_k2tau, cpi->b_a_lin, cpi->b_w_lin, cpi->J_q,
                                     cpi->J_b, cpi->J_a, cpi->H_b, cpi->H_a, cpi->P_meas);
    problem.AddResidualBlock(factor, nullptr, params);
  }

  // Reprojection factors
  for (int i = 0; i < cfg.num_poses; i++) {
    for (const auto &obs : data.observations[i]) {
      int f = obs.first;
      Eigen::Vector2d uv = obs.second;
      std::vector<double *> params = {vars_ori[i], vars_pos[i], vars_feat[f], var_calib_ori, var_calib_pos, var_calib_cam};
      auto *factor = new ov_init::Factor_ImageReprojCalib(uv, cfg.sigma_pix, false);
      problem.AddResidualBlock(factor, new ceres::CauchyLoss(1.0), params);
    }
  }

  auto t_setup = std::chrono::high_resolution_clock::now();
  res.time_setup_ms = std::chrono::duration<double, std::milli>(t_setup - t_start).count();

  // Solve
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.num_threads = cfg.num_threads;
  options.max_solver_time_in_seconds = cfg.max_solver_time;
  options.max_num_iterations = cfg.max_iterations;
  options.function_tolerance = 1e-5;
  options.gradient_tolerance = 1e-9;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  auto t_solve = std::chrono::high_resolution_clock::now();
  res.time_solve_ms = std::chrono::duration<double, std::milli>(t_solve - t_setup).count();

  res.initial_cost = summary.initial_cost;
  res.final_cost = summary.final_cost;
  res.iterations = summary.iterations.size();
  res.converged = (summary.termination_type == ceres::CONVERGENCE);

  // Covariance
  int last = cfg.num_poses - 1;
  std::vector<std::pair<const double *, const double *>> cov_blocks;
  cov_blocks.push_back({vars_ori[last], vars_ori[last]});
  cov_blocks.push_back({vars_pos[last], vars_pos[last]});
  cov_blocks.push_back({vars_vel[last], vars_vel[last]});
  cov_blocks.push_back({vars_bg[last], vars_bg[last]});
  cov_blocks.push_back({vars_ba[last], vars_ba[last]});

  ceres::Covariance::Options cov_options;
  cov_options.algorithm_type = ceres::DENSE_SVD;
  cov_options.apply_loss_function = true;
  cov_options.num_threads = cfg.num_threads;
  ceres::Covariance covariance(cov_options);

  Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Zero();
  if (covariance.Compute(cov_blocks, &problem)) {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> tmp;
    covariance.GetCovarianceBlockInTangentSpace(vars_ori[last], vars_ori[last], tmp.data());
    cov.block(0, 0, 3, 3) = tmp;
    covariance.GetCovarianceBlockInTangentSpace(vars_pos[last], vars_pos[last], tmp.data());
    cov.block(3, 3, 3, 3) = tmp;
    covariance.GetCovarianceBlockInTangentSpace(vars_vel[last], vars_vel[last], tmp.data());
    cov.block(6, 6, 3, 3) = tmp;
    covariance.GetCovarianceBlockInTangentSpace(vars_bg[last], vars_bg[last], tmp.data());
    cov.block(9, 9, 3, 3) = tmp;
    covariance.GetCovarianceBlockInTangentSpace(vars_ba[last], vars_ba[last], tmp.data());
    cov.block(12, 12, 3, 3) = tmp;
  }

  auto t_cov = std::chrono::high_resolution_clock::now();
  res.time_cov_ms = std::chrono::duration<double, std::milli>(t_cov - t_solve).count();

  // Errors
  const auto &gt = data.gt_states[last];
  Eigen::Matrix3d R_gt = quat_2_Rot(gt.q_GtoI);
  Eigen::Matrix3d R_est = quat_2_Rot(Eigen::Map<Eigen::Vector4d>(vars_ori[last]));
  res.err_ori_deg = -log_so3(R_gt * R_est.transpose()) * 180.0 / M_PI;
  res.err_pos = gt.p_IinG - Eigen::Map<Eigen::Vector3d>(vars_pos[last]);
  res.err_vel = gt.v_IinG - Eigen::Map<Eigen::Vector3d>(vars_vel[last]);
  res.err_bg = gt.bias_g - Eigen::Map<Eigen::Vector3d>(vars_bg[last]);
  res.err_ba = gt.bias_a - Eigen::Map<Eigen::Vector3d>(vars_ba[last]);

  // NEES
  Eigen::Matrix<double, 15, 1> err;
  err.segment<3>(0) = -log_so3(R_gt * R_est.transpose());
  err.segment<3>(3) = res.err_pos;
  err.segment<3>(6) = res.err_vel;
  err.segment<3>(9) = res.err_bg;
  err.segment<3>(12) = res.err_ba;

  Eigen::Matrix<double, 15, 15> info = cov.inverse();
  res.nees_total = (err.transpose() * info * err)(0, 0);
  res.nees_ori = (err.segment<3>(0).transpose() * info.block<3, 3>(0, 0) * err.segment<3>(0))(0, 0);
  res.nees_pos = (err.segment<3>(3).transpose() * info.block<3, 3>(3, 3) * err.segment<3>(3))(0, 0);
  res.nees_vel = (err.segment<3>(6).transpose() * info.block<3, 3>(6, 6) * err.segment<3>(6))(0, 0);

  res.cov_diag = cov.diagonal();

  // Cleanup
  for (int i = 0; i < cfg.num_poses; i++) {
    delete[] vars_ori[i];
    delete[] vars_pos[i];
    delete[] vars_vel[i];
    delete[] vars_bg[i];
    delete[] vars_ba[i];
  }
  for (int f = 0; f < cfg.num_features; f++)
    delete[] vars_feat[f];
  delete[] var_calib_ori;
  delete[] var_calib_pos;
  delete[] var_calib_cam;

  return res;
}

// ===========================================================================
// Run ceres-free backend
// ===========================================================================
SolverResults runCeresFree(const SyntheticWindow &data, const TestConfig &cfg) {
  SolverResults res;
  res.name = "CeresFree";

  auto t_start = std::chrono::high_resolution_clock::now();

  ov_init::zbft_sfm::Problem problem;
  problem.EnableOwnership();

  std::vector<double *> vars_ori(cfg.num_poses);
  std::vector<double *> vars_pos(cfg.num_poses);
  std::vector<double *> vars_vel(cfg.num_poses);
  std::vector<double *> vars_bg(cfg.num_poses);
  std::vector<double *> vars_ba(cfg.num_poses);
  std::vector<double *> vars_feat(cfg.num_features);

  // Gravity on S²
  auto *var_gravity = new double[3];
  var_gravity[0] = 0.0;
  var_gravity[1] = 0.0;
  var_gravity[2] = cfg.gravity_mag;
  problem.AddParameterBlock(var_gravity, 3, new ov_init::zbft_sfm::GravityS2Parameterization(cfg.gravity_mag));

  auto *var_calib_ori = new double[4];
  auto *var_calib_pos = new double[3];
  auto *var_calib_cam = new double[8];

  for (int j = 0; j < 4; j++)
    var_calib_ori[j] = cfg.q_ItoC(j);
  for (int j = 0; j < 3; j++)
    var_calib_pos[j] = cfg.p_IinC(j);
  var_calib_cam[0] = cfg.fx;
  var_calib_cam[1] = cfg.fy;
  var_calib_cam[2] = cfg.cx;
  var_calib_cam[3] = cfg.cy;
  for (int j = 4; j < 8; j++)
    var_calib_cam[j] = 0.0;

  problem.AddParameterBlock(var_calib_ori, 4, new ov_init::zbft_sfm::State_JPLQuatLocal());
  problem.AddParameterBlock(var_calib_pos, 3);
  problem.AddParameterBlock(var_calib_cam, 8);
  problem.SetParameterBlockConstant(var_calib_ori);
  problem.SetParameterBlockConstant(var_calib_pos);
  problem.SetParameterBlockConstant(var_calib_cam);

  // Same initial perturbation as Ceres
  std::mt19937 gen(123);
  std::normal_distribution<double> perturb_pos(0.0, 0.001);
  std::normal_distribution<double> perturb_vel(0.0, 0.001);
  std::normal_distribution<double> perturb_feat(0.0, 0.01);

  for (int i = 0; i < cfg.num_poses; i++) {
    vars_ori[i] = new double[4];
    vars_pos[i] = new double[3];
    vars_vel[i] = new double[3];
    vars_bg[i] = new double[3];
    vars_ba[i] = new double[3];

    for (int j = 0; j < 4; j++)
      vars_ori[i][j] = data.gt_states[i].q_GtoI(j);
    for (int j = 0; j < 3; j++) {
      vars_pos[i][j] = data.gt_states[i].p_IinG(j) + perturb_pos(gen);
      vars_vel[i][j] = data.gt_states[i].v_IinG(j) + perturb_vel(gen);
      vars_bg[i][j] = data.gt_states[i].bias_g(j);
      vars_ba[i][j] = data.gt_states[i].bias_a(j);
    }

    problem.AddParameterBlock(vars_ori[i], 4, new ov_init::zbft_sfm::State_JPLQuatLocal());
    problem.AddParameterBlock(vars_pos[i], 3);
    problem.AddParameterBlock(vars_vel[i], 3);
    problem.AddParameterBlock(vars_bg[i], 3);
    problem.AddParameterBlock(vars_ba[i], 3);
  }

  for (int f = 0; f < cfg.num_features; f++) {
    vars_feat[f] = new double[3];
    for (int j = 0; j < 3; j++)
      vars_feat[f][j] = data.features_inG[f](j) + perturb_feat(gen);
    problem.AddParameterBlock(vars_feat[f], 3);
    problem.SetSchurLandmark(vars_feat[f]);
  }

  // Gauge prior (S² gravity path)
  {
    Eigen::MatrixXd x_lin = Eigen::MatrixXd::Zero(13, 1);
    for (int j = 0; j < 4; j++)
      x_lin(j) = vars_ori[0][j];
    for (int j = 0; j < 3; j++) {
      x_lin(4 + j) = vars_pos[0][j];
      x_lin(7 + j) = vars_bg[0][j];
      x_lin(10 + j) = vars_ba[0][j];
    }
    Eigen::MatrixXd prior_grad = Eigen::MatrixXd::Zero(12, 1);
    Eigen::MatrixXd prior_Info = Eigen::MatrixXd::Identity(12, 12);
    prior_Info.block(0, 0, 3, 3) *= 1.0 / std::pow(0.001, 2);
    prior_Info.block(3, 3, 3, 3) *= 1.0 / std::pow(0.001, 2);
    prior_Info.block(6, 6, 3, 3) *= 1.0 / std::pow(0.05, 2);
    prior_Info.block(9, 9, 3, 3) *= 1.0 / std::pow(0.10, 2);

    std::vector<std::string> x_types = {"quat", "vec3", "vec3", "vec3"};
    std::vector<double *> factor_params = {vars_ori[0], vars_pos[0], vars_bg[0], vars_ba[0]};
    problem.AddResidualBlock(new ov_init::zbft_sfm::Factor_GenericPrior(x_lin, x_types, prior_Info, prior_grad), nullptr, factor_params);
  }

  // IMU factors (with S² gravity)
  Eigen::Vector3d gravity = data.gravity;
  for (int i = 1; i < cfg.num_poses; i++) {
    auto cpi = data.cpis[i];
    std::vector<double *> params = {vars_ori[i - 1], vars_bg[i - 1], vars_vel[i - 1], vars_ba[i - 1], vars_pos[i - 1],
                                    vars_ori[i],     vars_bg[i],     vars_vel[i],     vars_ba[i],     vars_pos[i],
                                    var_gravity};
    auto *factor =
        new ov_init::zbft_sfm::Factor_ImuCPIv1(cpi->DT, gravity, cpi->alpha_tau, cpi->beta_tau, cpi->q_k2tau, cpi->b_a_lin, cpi->b_w_lin,
                                               cpi->J_q, cpi->J_b, cpi->J_a, cpi->H_b, cpi->H_a, cpi->P_meas);
    problem.AddResidualBlock(factor, nullptr, params);
  }

  // Reprojection factors
  for (int i = 0; i < cfg.num_poses; i++) {
    for (const auto &obs : data.observations[i]) {
      int f = obs.first;
      Eigen::Vector2d uv = obs.second;
      std::vector<double *> params = {vars_ori[i], vars_pos[i], vars_feat[f], var_calib_ori, var_calib_pos, var_calib_cam};
      auto *factor = new ov_init::zbft_sfm::Factor_ImageReprojCalib(uv, cfg.sigma_pix, false);
      problem.AddResidualBlock(factor, new ov_init::zbft_sfm::CauchyLoss(1.0), params);
    }
  }

  auto t_setup = std::chrono::high_resolution_clock::now();
  res.time_setup_ms = std::chrono::duration<double, std::milli>(t_setup - t_start).count();

  // Solve
  ov_init::zbft_sfm::SolverOptions options;
  options.num_threads = cfg.num_threads;
  options.max_solver_time_seconds = cfg.max_solver_time;
  options.max_num_iterations = cfg.max_iterations;
  options.function_tolerance = 1e-5;
  options.gradient_tolerance = 1e-9;

  ov_init::zbft_sfm::SolverSummary summary = problem.Solve(options);

  auto t_solve = std::chrono::high_resolution_clock::now();
  res.time_solve_ms = std::chrono::duration<double, std::milli>(t_solve - t_setup).count();

  res.initial_cost = summary.initial_cost;
  res.final_cost = summary.final_cost;
  res.iterations = summary.iterations;
  res.converged = summary.converged;

  // Covariance
  int last = cfg.num_poses - 1;
  std::vector<double *> cov_blocks = {vars_ori[last], vars_pos[last], vars_vel[last], vars_bg[last], vars_ba[last]};
  Eigen::MatrixXd cov_dyn = Eigen::MatrixXd::Zero(15, 15);
  problem.ComputeCovariance(cov_blocks, cov_dyn, options);
  Eigen::Matrix<double, 15, 15> cov = cov_dyn;

  auto t_cov = std::chrono::high_resolution_clock::now();
  res.time_cov_ms = std::chrono::duration<double, std::milli>(t_cov - t_solve).count();

  // Errors
  const auto &gt = data.gt_states[last];
  Eigen::Matrix3d R_gt = quat_2_Rot(gt.q_GtoI);
  Eigen::Matrix3d R_est = quat_2_Rot(Eigen::Map<Eigen::Vector4d>(vars_ori[last]));
  res.err_ori_deg = -log_so3(R_gt * R_est.transpose()) * 180.0 / M_PI;
  res.err_pos = gt.p_IinG - Eigen::Map<Eigen::Vector3d>(vars_pos[last]);
  res.err_vel = gt.v_IinG - Eigen::Map<Eigen::Vector3d>(vars_vel[last]);
  res.err_bg = gt.bias_g - Eigen::Map<Eigen::Vector3d>(vars_bg[last]);
  res.err_ba = gt.bias_a - Eigen::Map<Eigen::Vector3d>(vars_ba[last]);

  // NEES
  Eigen::Matrix<double, 15, 1> err;
  err.segment<3>(0) = -log_so3(R_gt * R_est.transpose());
  err.segment<3>(3) = res.err_pos;
  err.segment<3>(6) = res.err_vel;
  err.segment<3>(9) = res.err_bg;
  err.segment<3>(12) = res.err_ba;

  Eigen::Matrix<double, 15, 15> info = cov.inverse();
  res.nees_total = (err.transpose() * info * err)(0, 0);
  res.nees_ori = (err.segment<3>(0).transpose() * info.block<3, 3>(0, 0) * err.segment<3>(0))(0, 0);
  res.nees_pos = (err.segment<3>(3).transpose() * info.block<3, 3>(3, 3) * err.segment<3>(3))(0, 0);
  res.nees_vel = (err.segment<3>(6).transpose() * info.block<3, 3>(6, 6) * err.segment<3>(6))(0, 0);

  res.cov_diag = cov.diagonal();

  // Cleanup
  for (int i = 0; i < cfg.num_poses; i++) {
    delete[] vars_ori[i];
    delete[] vars_pos[i];
    delete[] vars_vel[i];
    delete[] vars_bg[i];
    delete[] vars_ba[i];
  }
  for (int f = 0; f < cfg.num_features; f++)
    delete[] vars_feat[f];
  delete[] var_gravity;
  delete[] var_calib_ori;
  delete[] var_calib_pos;
  delete[] var_calib_cam;

  return res;
}

// ===========================================================================
// Print and compare
// ===========================================================================
void printResults(const SolverResults &r) {
  printf("\n=== %s Results ===\n", r.name.c_str());
  printf("Timing:  setup=%.2f ms  solve=%.2f ms  cov=%.2f ms  total=%.2f ms\n", r.time_setup_ms, r.time_solve_ms, r.time_cov_ms,
         r.time_setup_ms + r.time_solve_ms + r.time_cov_ms);
  printf("Solver:  %d iters  converged=%s  cost %.4e => %.4e\n", r.iterations, r.converged ? "yes" : "NO", r.initial_cost, r.final_cost);
  printf("Errors:  ori=[%.3f,%.3f,%.3f] deg  pos=[%.4f,%.4f,%.4f] m\n", r.err_ori_deg(0), r.err_ori_deg(1), r.err_ori_deg(2), r.err_pos(0),
         r.err_pos(1), r.err_pos(2));
  printf("         vel=[%.4f,%.4f,%.4f] m/s  bg=[%.5f,%.5f,%.5f] rad/s\n", r.err_vel(0), r.err_vel(1), r.err_vel(2), r.err_bg(0),
         r.err_bg(1), r.err_bg(2));
  printf("NEES:    total=%.2f  ori=%.2f  pos=%.2f  vel=%.2f  (ideal=15,3,3,3)\n", r.nees_total, r.nees_ori, r.nees_pos, r.nees_vel);
}

void compareResults(const SolverResults &ceres, const SolverResults &zbft) {
  printf("\n========================================\n");
  printf("          A/B COMPARISON SUMMARY\n");
  printf("========================================\n");

  double speedup = ceres.time_solve_ms / zbft.time_solve_ms;
  printf("\nTiming (solve only):\n");
  printf("  Ceres:     %.2f ms\n", ceres.time_solve_ms);
  printf("  CeresFree: %.2f ms  (%.2fx %s)\n", zbft.time_solve_ms, speedup > 1 ? speedup : 1.0 / speedup,
         speedup > 1 ? "faster" : "slower");

  double total_speedup = (ceres.time_setup_ms + ceres.time_solve_ms + ceres.time_cov_ms) /
                         (zbft.time_setup_ms + zbft.time_solve_ms + zbft.time_cov_ms);
  printf("\nTiming (total):\n");
  printf("  Ceres:     %.2f ms\n", ceres.time_setup_ms + ceres.time_solve_ms + ceres.time_cov_ms);
  printf("  CeresFree: %.2f ms  (%.2fx %s)\n", zbft.time_setup_ms + zbft.time_solve_ms + zbft.time_cov_ms,
         total_speedup > 1 ? total_speedup : 1.0 / total_speedup, total_speedup > 1 ? "faster" : "slower");

  printf("\nFinal cost:\n");
  printf("  Ceres:     %.6e\n", ceres.final_cost);
  printf("  CeresFree: %.6e\n", zbft.final_cost);
  double cost_diff_pct = 100.0 * std::abs(ceres.final_cost - zbft.final_cost) / std::max(ceres.final_cost, 1e-10);
  printf("  Diff:      %.2e (%.2f%%)\n", std::abs(ceres.final_cost - zbft.final_cost), cost_diff_pct);

  printf("\nPosition error (m):\n");
  printf("  Ceres:     |e|=%.5f\n", ceres.err_pos.norm());
  printf("  CeresFree: |e|=%.5f\n", zbft.err_pos.norm());

  printf("\nNEES (ideal=15 total, 3 per component):\n");
  printf("  Ceres:     total=%.2f  ori=%.2f  pos=%.2f  vel=%.2f\n", ceres.nees_total, ceres.nees_ori, ceres.nees_pos, ceres.nees_vel);
  printf("  CeresFree: total=%.2f  ori=%.2f  pos=%.2f  vel=%.2f\n", zbft.nees_total, zbft.nees_ori, zbft.nees_pos, zbft.nees_vel);

  // Verdict
  printf("\n========================================\n");
  printf("                VERDICT\n");
  printf("========================================\n");

  bool zbft_faster = total_speedup > 1.0;
  bool zbft_accurate = zbft.err_pos.norm() <= ceres.err_pos.norm() * 1.1;
  bool zbft_converged = zbft.converged;
  bool zbft_cost_ok = zbft.final_cost <= ceres.final_cost * 1.01;

  if (zbft_faster && zbft_accurate && zbft_converged && zbft_cost_ok) {
    printf("  [PASS] CeresFree outperforms or matches Ceres!\n");
    printf("         - %.2fx faster\n", total_speedup);
    printf("         - Accuracy within tolerance\n");
    printf("         - Cost within 1%%\n");
  } else {
    printf("  [INFO] Comparison details:\n");
    printf("         - Speed: %s (%.2fx)\n", zbft_faster ? "BETTER" : "WORSE", zbft_faster ? total_speedup : 1.0 / total_speedup);
    printf("         - Accuracy: %s\n", zbft_accurate ? "GOOD" : "DEGRADED");
    printf("         - Converged: %s\n", zbft_converged ? "YES" : "NO");
    printf("         - Cost: %s\n", zbft_cost_ok ? "GOOD" : "HIGHER");
  }
  printf("========================================\n");
}

// ===========================================================================
// Ground-truth residual sanity check: with the trajectory integrated from the CPIs, every
// IMU-factor residual must vanish at GT and every reprojection residual must sit at the pixel
// noise floor. If GT is not (near) the cost minimum, the whole A/B is meaningless.
// ===========================================================================
void checkGroundTruthResiduals(const SyntheticWindow &data, const TestConfig &cfg) {
  Eigen::Vector4d q_ItoC = cfg.q_ItoC;
  Eigen::Vector3d p_IinC = cfg.p_IinC;
  Eigen::Matrix<double, 8, 1> camvals;
  camvals << cfg.fx, cfg.fy, cfg.cx, cfg.cy, 0, 0, 0, 0;
  Eigen::Vector3d gravity = data.gravity;

  // IMU factors (ceres-free) at GT -> residual must be ~0
  double max_imu_res = 0.0;
  for (int i = 1; i < cfg.num_poses; i++) {
    auto cpi = data.cpis[i];
    ov_init::zbft_sfm::Factor_ImuCPIv1 f(cpi->DT, gravity, cpi->alpha_tau, cpi->beta_tau, cpi->q_k2tau, cpi->b_a_lin, cpi->b_w_lin,
                                         cpi->J_q, cpi->J_b, cpi->J_a, cpi->H_b, cpi->H_a, cpi->P_meas);
    const GroundTruthState &a = data.gt_states[i - 1];
    const GroundTruthState &b = data.gt_states[i];
    Eigen::Vector4d qa = a.q_GtoI, qb = b.q_GtoI;
    Eigen::Vector3d bga = a.bias_g, va = a.v_IinG, baa = a.bias_a, pa = a.p_IinG;
    Eigen::Vector3d bgb = b.bias_g, vb = b.v_IinG, bab = b.bias_a, pb = b.p_IinG;
    Eigen::Vector3d g = gravity;
    const double *params[11] = {qa.data(), bga.data(), va.data(), baa.data(), pa.data(), qb.data(),
                                bgb.data(), vb.data(), bab.data(), pb.data(), g.data()};
    Eigen::Matrix<double, 15, 1> res;
    f.Evaluate(params, res.data(), nullptr);
    max_imu_res = std::max(max_imu_res, res.norm());
  }

  // Reprojection factors (ceres-free) at GT -> residual must be ~pixel noise
  double sse_pix = 0.0;
  int nobs = 0;
  for (int i = 0; i < cfg.num_poses; i++) {
    Eigen::Vector4d qi = data.gt_states[i].q_GtoI;
    Eigen::Vector3d pi = data.gt_states[i].p_IinG;
    for (const auto &obs : data.observations[i]) {
      Eigen::Vector3d pf = data.features_inG[obs.first];
      ov_init::zbft_sfm::Factor_ImageReprojCalib f(obs.second, cfg.sigma_pix, false);
      const double *params[6] = {qi.data(), pi.data(), pf.data(), q_ItoC.data(), p_IinC.data(), camvals.data()};
      Eigen::Vector2d res;
      f.Evaluate(params, res.data(), nullptr);
      sse_pix += (res * cfg.sigma_pix).squaredNorm(); // un-whiten back to pixels
      nobs++;
    }
  }
  double rms_pix = std::sqrt(sse_pix / std::max(1, nobs));

  printf("\n[GT-check] max IMU-factor residual at GT: %.3e   (expect ~0)\n", max_imu_res);
  printf("[GT-check] reprojection RMS at GT: %.3f px over %d obs   (expect ~sigma_pix=%.2f)\n", rms_pix, nobs, cfg.sigma_pix);
  if (max_imu_res > 1e-5)
    printf("[GT-check] *** WARNING: IMU residual nonzero at GT -- generator/factor convention mismatch! ***\n");
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char **argv) {
  printf("====================================================\n");
  printf("  Dynamic Init A/B Comparison: Ceres vs CeresFree\n");
  printf("====================================================\n");

  TestConfig cfg;
  if (argc > 1)
    cfg.num_poses = std::atoi(argv[1]);
  if (argc > 2)
    cfg.num_features = std::atoi(argv[2]);

  printf("\nConfiguration:\n");
  printf("  Poses: %d\n", cfg.num_poses);
  printf("  Features: %d\n", cfg.num_features);
  printf("  Window: %.1f sec\n", cfg.num_poses * cfg.dt_pose);
  printf("  Threads: %d\n", cfg.num_threads);

  SyntheticWindow data;
  data.generate(cfg);

  // Sanity: GT must be (near) the cost minimum, or the A/B below is meaningless.
  checkGroundTruthResiduals(data, cfg);

  printf("\nRunning Ceres backend...\n");
  SolverResults ceres_results = runCeres(data, cfg);
  printResults(ceres_results);

  printf("\nRunning CeresFree backend...\n");
  SolverResults zbft_results = runCeresFree(data, cfg);
  printResults(zbft_results);

  compareResults(ceres_results, zbft_results);

  return (zbft_results.converged && ceres_results.converged) ? 0 : 1;
}
