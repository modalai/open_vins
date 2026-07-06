/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * Phase-3 (warm-start) covariance validation for the Ceres-free solver core
 * (ov_init::zbft_sfm). Depends ONLY on Eigen + the solver core (Problem.cpp,
 * Parallel.cpp) -- NOT on ov_core / ROS / OpenCV -- so it builds and runs anywhere:
 *
 *   g++ -O2 -std=c++17 -pthread -I/usr/include/eigen3 \
 *       test_warmstart_cov.cpp Problem.cpp Parallel.cpp -o /tmp/test_warm && /tmp/test_warm
 *
 * It validates the two mechanisms the warm-start injection (DynamicInitializer +
 * StateHelper::set_initial_state_warmstart) relies on, which the existing tests do
 * NOT cover:
 *
 *  (A) SHARED-POINTER JOINT SLICE. The newest window clone reuses the SAME solver
 *      parameter pointers as the active IMU pose, so ComputeCovariance over
 *      [imu..., clone_0, ..., clone_N(==imu pose)] must return a clone_N block that
 *      is bit-identical to the imu pose block, with cross(imu, clone_N) == the imu
 *      pose auto-covariance. Also cross-checked against the dense full inverse.
 *
 *  (B) RE-ALIGN SIMILARITY + CONGRUENCE INFLATION over the [IMU(15) | clones(6)]
 *      layout (a verbatim copy of the DynamicInitializer transforms): symmetry +
 *      SPD preserved, untouched blocks (theta, biases) unchanged, and the newest
 *      clone stays equal to the imu pose (the consistency the EKF requires).
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later version.
 */

#include <cstdio>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "LocalParameterization.h"
#include "LossFunction.h"
#include "Problem.h"

using namespace ov_init::zbft_sfm;

// ---------------------------------------------------------------------------- harness
static int g_failures = 0;
static int g_checks = 0;
static void check_lt(double value, double tol, const char *name) {
  ++g_checks;
  if (!(value < tol) || !std::isfinite(value)) {
    std::printf("  [FAIL] %-52s  value=%.3e  tol=%.1e\n", name, value, tol);
    ++g_failures;
  } else {
    std::printf("  [ ok ] %-52s  value=%.3e\n", name, value);
  }
}
static void check_true(bool cond, const char *name) {
  ++g_checks;
  if (!cond) {
    std::printf("  [FAIL] %s\n", name);
    ++g_failures;
  } else {
    std::printf("  [ ok ] %s\n", name);
  }
}

// ---------------------------------------------------------------------------- factors
// r = sqrt(w) * (x - x0): gauge anchor on a single 3-block.
class PriorFactor : public CostFunction {
public:
  PriorFactor(const Eigen::Vector3d &x0, double w) : x0_(x0), s_(std::sqrt(w)) {
    set_num_residuals(3);
    mutable_parameter_block_sizes()->push_back(3);
  }
  bool Evaluate(double const *const *p, double *r, double **J) const override {
    Eigen::Map<const Eigen::Vector3d> x(p[0]);
    Eigen::Map<Eigen::Vector3d> res(r);
    res = s_ * (x - x0_);
    if (J && J[0]) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> j(J[0]);
      j = s_ * Eigen::Matrix3d::Identity();
    }
    return true;
  }

private:
  Eigen::Vector3d x0_;
  double s_;
};

// r = (l - p) - z: a "bearing-ish" observation touching exactly one pose + one landmark
// (the arrowhead structure the Schur path exploits). Linear, so the solve is exact.
class ObsFactor : public CostFunction {
public:
  explicit ObsFactor(const Eigen::Vector3d &z) : z_(z) {
    set_num_residuals(3);
    mutable_parameter_block_sizes()->push_back(3); // pose
    mutable_parameter_block_sizes()->push_back(3); // landmark
  }
  bool Evaluate(double const *const *p, double *r, double **J) const override {
    Eigen::Map<const Eigen::Vector3d> pose(p[0]);
    Eigen::Map<const Eigen::Vector3d> land(p[1]);
    Eigen::Map<Eigen::Vector3d> res(r);
    res = (land - pose) - z_;
    if (J && J[0]) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> j0(J[0]);
      j0 = -Eigen::Matrix3d::Identity();
    }
    if (J && J[1]) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> j1(J[1]);
      j1 = Eigen::Matrix3d::Identity();
    }
    return true;
  }

private:
  Eigen::Vector3d z_;
};

// ---------------------------------------------------------------------------- (A) shared-pointer joint slice
static void test_shared_pointer_joint(std::mt19937 &rng) {
  std::printf("[test] (A) shared-pointer joint covariance (newest clone == imu pose)\n");
  std::normal_distribution<double> N(0.0, 1.0);

  const int P = 4; // pose blocks p[0..3]; treat p[3] as BOTH the active "imu pose" and the newest clone
  const int L = 6; // landmark blocks
  std::vector<Eigen::Vector3d> poses(P), lands(L);
  for (auto &x : poses) x = Eigen::Vector3d(N(rng), N(rng), N(rng));
  for (auto &x : lands) x = Eigen::Vector3d(N(rng), N(rng), N(rng));

  // Ground-truth geometry to generate consistent (noise-free) measurements, so GT is the minimizer.
  std::vector<Eigen::Vector3d> gtP(P), gtL(L);
  for (int i = 0; i < P; i++) gtP[i] = Eigen::Vector3d(N(rng), N(rng), N(rng));
  for (int j = 0; j < L; j++) gtL[j] = Eigen::Vector3d(N(rng), N(rng), N(rng));

  std::vector<ObsFactor> obs;
  obs.reserve(P * L);
  for (int i = 0; i < P; i++)
    for (int j = 0; j < L; j++)
      obs.emplace_back(gtL[j] - gtP[i]);
  PriorFactor prior(gtP[0], 1e4); // anchor the gauge (position + the global translation null space)

  auto build = [&](bool tag_landmarks, Problem &problem) {
    for (int i = 0; i < P; i++)
      problem.AddParameterBlock(poses[i].data(), 3);
    for (int j = 0; j < L; j++)
      problem.AddParameterBlock(lands[j].data(), 3);
    if (tag_landmarks)
      for (int j = 0; j < L; j++)
        problem.SetSchurLandmark(lands[j].data());
    int k = 0;
    for (int i = 0; i < P; i++)
      for (int j = 0; j < L; j++)
        problem.AddResidualBlock(&obs[k++], nullptr, {poses[i].data(), lands[j].data()});
    problem.AddResidualBlock(&prior, nullptr, {poses[0].data()});
  };

  // Solve once (Schur path) to land at the optimum.
  {
    Problem problem;
    build(true, problem);
    SolverOptions opts;
    opts.num_threads = 1;
    problem.Solve(opts);
  }

  // Request the JOINT covariance with the imu-pose pointer (p[3]) appearing as BOTH the leading "imu"
  // block and the trailing "newest clone" -- exactly how DynamicInitializer assembles the request
  // (the newest clone reuses ceres_vars_ori/pos[newest]). Layout: [imu(p3) | c0 c1 c2 c3(==p3)].
  Problem problem;
  build(true, problem);
  SolverOptions opts;
  opts.num_threads = 1;
  std::vector<double *> blocks = {poses[3].data(), poses[0].data(), poses[1].data(), poses[2].data(), poses[3].data()};
  Eigen::MatrixXd J;
  bool ok = problem.ComputeCovariance(blocks, J, opts);
  check_true(ok && J.rows() == 15 && J.cols() == 15, "joint covariance computed (15x15: imu + 4 clones)");

  const Eigen::Matrix3d imu = J.block(0, 0, 3, 3);          // imu pose (p3)
  const Eigen::Matrix3d newest = J.block(12, 12, 3, 3);     // newest clone (p3 again)
  const Eigen::Matrix3d cross = J.block(0, 12, 3, 3);       // cross(imu, newest)
  check_lt((newest - imu).norm(), 1e-12, "newest-clone block == imu pose block (bit-identical)");
  check_lt((cross - imu).norm(), 1e-12, "cross(imu, newest clone) == imu pose auto-cov");
  check_lt((J - J.transpose()).norm(), 1e-12, "joint covariance symmetric");
  // The duplicated variable makes (imu - newest) deterministic -> that direction has exactly zero
  // variance: Var(imu - newest) = Saa - Sab - Sba + Sbb = 0. This is the correct singular structure
  // OpenVINS' augment_clone also produces (a clone is a copy of the imu pose).
  check_lt((imu - cross - cross.transpose() + newest).norm(), 1e-12, "Var(imu - newest clone) == 0");

  // Cross-check the clone sub-block (c0..c2) against an independent dense full-inverse marginal.
  Eigen::MatrixXd Jdense;
  {
    Problem dproblem;
    build(false, dproblem); // untagged -> plain dense path, independent code route
    bool ok2 = dproblem.ComputeCovariance({poses[0].data(), poses[1].data(), poses[2].data(), poses[3].data()}, Jdense, opts);
    check_true(ok2, "dense full-inverse marginal computed");
  }
  // J rows 3..14 are clones c0,c1,c2,c3 == poses p0,p1,p2,p3 == Jdense in order.
  check_lt((J.block(3, 3, 12, 12) - Jdense).norm(), 1e-6, "joint clone block == dense full-inverse nav marginal");
}

// ------------------------------------------------- (B) realign similarity + congruence inflation indexing
// Verbatim copies of the DynamicInitializer transforms over the [IMU(15) | clones(6)] layout, applied
// here to a known matrix so an indexing slip is caught host-side (those transforms live in
// DynamicInitializer.cpp, which needs the full ov_init build to compile).
static Eigen::MatrixXd apply_realign_T(const Eigen::MatrixXd &cov, const Eigen::Matrix3d &R, int n_clones) {
  Eigen::MatrixXd T = Eigen::MatrixXd::Identity(cov.rows(), cov.cols());
  T.block(3, 3, 3, 3) = R; // IMU p
  T.block(6, 6, 3, 3) = R; // IMU v
  for (int c = 0; c < n_clones; ++c)
    T.block(15 + 6 * c + 3, 15 + 6 * c + 3, 3, 3) = R; // clone p
  return (T * cov * T.transpose()).eval();
}
static Eigen::MatrixXd apply_inflation_S(const Eigen::MatrixXd &cov, double ko, double kp, double kv, double kbg, double kba) {
  Eigen::VectorXd sd = Eigen::VectorXd::Ones(cov.rows());
  sd.segment<3>(0).setConstant(std::sqrt(ko));
  sd.segment<3>(3).setConstant(std::sqrt(kp));
  sd.segment<3>(6).setConstant(std::sqrt(kv));
  sd.segment<3>(9).setConstant(std::sqrt(kbg));
  sd.segment<3>(12).setConstant(std::sqrt(kba));
  for (int o = 15; o + 6 <= (int)cov.rows(); o += 6) {
    sd.segment<3>(o).setConstant(std::sqrt(ko));
    sd.segment<3>(o + 3).setConstant(std::sqrt(kp));
  }
  return (sd.asDiagonal() * cov * sd.asDiagonal()).eval();
}

static void test_transform_indexing(std::mt19937 &rng) {
  std::printf("[test] (B) realign similarity + congruence inflation over [IMU(15)|clones(6)]\n");
  std::normal_distribution<double> N(0.0, 1.0);
  const int K = 3;             // clones
  const int n = 15 + 6 * K;    // 33

  // A random SPD joint cov whose NEWEST clone (last 6) is, by construction, the IMU pose (first 6):
  // build SPD over [imu(15) | c0 | c1] (size 27), then append the newest clone as a copy of the imu
  // pose rows/cols (this is the singular "clone == imu pose" structure of a fresh augmentation).
  Eigen::MatrixXd A = Eigen::MatrixXd::NullaryExpr(27, 27, [&](Eigen::Index, Eigen::Index) { return N(rng); });
  Eigen::MatrixXd base = A.transpose() * A + 27 * Eigen::MatrixXd::Identity(27, 27); // SPD
  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(n, n);
  cov.topLeftCorner(27, 27) = base;
  cov.block(0, 27, 27, 6) = base.block(0, 0, 27, 6);   // cross(everything, newest) = cross(everything, imu pose)
  cov.block(27, 0, 6, 27) = base.block(0, 0, 6, 27);
  cov.block(27, 27, 6, 6) = base.block(0, 0, 6, 6);    // newest clone block = imu pose block
  cov = (0.5 * (cov + cov.transpose())).eval();

  auto newest_eq_imu = [&](const Eigen::MatrixXd &M, const char *tag) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s: newest clone == imu pose", tag);
    check_lt((M.block(27, 27, 6, 6) - M.block(0, 0, 6, 6)).norm(), 1e-9, buf);
    std::snprintf(buf, sizeof(buf), "%s: cross(imu, newest) == imu pose", tag);
    check_lt((M.block(0, 27, 6, 6) - M.block(0, 0, 6, 6)).norm(), 1e-9, buf);
  };
  newest_eq_imu(cov, "input");

  // Realign by a non-trivial rotation. R orthogonal -> SPD + symmetry preserved.
  Eigen::Matrix3d R = Eigen::AngleAxisd(0.4, Eigen::Vector3d(0.3, -0.7, 0.5).normalized()).toRotationMatrix();
  Eigen::MatrixXd covT = apply_realign_T(cov, R, K);
  check_lt((covT - covT.transpose()).norm(), 1e-9, "T: symmetric");
  check_true(Eigen::LLT<Eigen::MatrixXd>(covT.block(0, 0, 27, 27)).info() == Eigen::Success, "T: leading 27x27 still SPD");
  // theta (0..2) and biases (9..14) are NOT rotated -> untouched.
  check_lt((covT.block(0, 0, 3, 3) - cov.block(0, 0, 3, 3)).norm(), 1e-9, "T: imu theta block unchanged");
  check_lt((covT.block(9, 9, 6, 6) - cov.block(9, 9, 6, 6)).norm(), 1e-9, "T: imu bias blocks unchanged");
  // imu p (3..5) -> R * Spp * R^T.
  check_lt((covT.block(3, 3, 3, 3) - R * cov.block(3, 3, 3, 3) * R.transpose()).norm(), 1e-9, "T: imu p rotated by R");
  newest_eq_imu(covT, "T"); // the consistency the EKF requires must survive the transform

  // Congruence inflation, then re-check the same invariants.
  Eigen::MatrixXd covS = apply_inflation_S(covT, 30.0, 1.0, 150.0, 10.0, 150.0);
  check_lt((covS - covS.transpose()).norm(), 1e-6, "S: symmetric");
  check_true(Eigen::LLT<Eigen::MatrixXd>(covS.block(0, 0, 27, 27)).info() == Eigen::Success, "S: leading 27x27 still SPD");
  newest_eq_imu(covS, "S");
}

int main() {
  std::printf("==== ov_init::zbft_sfm warm-start covariance tests ====\n");
  std::mt19937 rng(7);
  test_shared_pointer_joint(rng);
  test_transform_indexing(rng);
  std::printf("==== %d checks, %d failures ====\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
