/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Contributor: Joao Leonardo Silva Cotta (@zauberflote1)
 *
 * Finite-difference validation of the LIFTED analytic factors (ov_init::zbft_sfm)
 * against ov_core. Confirms the residual/Jacobian transcription is correct.
 *
 * The IMU-CPI and generic-prior factors depend only on header-only ov_core
 * (quat_ops.h) + CpiV1, so this can be COMPILED AND RUN natively:
 *
 *   OVC=../../../ov_core/src
 *   g++ -O2 -std=c++17 -I/usr/include/eigen3 -I$OVC \
 *       test_mini_factors.cpp State_JPLQuatLocal.cpp Factor_ImuCPIv1.cpp \
 *       Factor_GenericPrior.cpp $OVC/cpi/CpiV1.cpp -o /tmp/test_factors && /tmp/test_factors
 *
 * The reprojection factor additionally needs the OpenCV-backed camera models, so
 * it is covered by the aarch64 ov_init_lib cross-build + -fsyntax-only checks.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "Factor_GenericPrior.h"
#include "Factor_ImuCPIv1.h"
#include "LocalParameterization.h"
#include "State_JPLQuatLocal.h"

#include "cpi/CpiV1.h"
#include "utils/quat_ops.h"

using namespace ov_init::zbft_sfm;

static int g_failures = 0;
static void check_lt(double v, double tol, const char *name) {
  if (!(v < tol) || !std::isfinite(v)) {
    std::printf("  [FAIL] %-44s value=%.3e tol=%.1e\n", name, v, tol);
    ++g_failures;
  } else {
    std::printf("  [ ok ] %-44s value=%.3e\n", name, v);
  }
}

// Evaluate residual only, with the given ambient block values.
static Eigen::VectorXd eval_res(const CostFunction &f, const std::vector<Eigen::VectorXd> &xs) {
  std::vector<const double *> p(xs.size());
  for (size_t i = 0; i < xs.size(); ++i)
    p[i] = xs[i].data();
  Eigen::VectorXd r(f.num_residuals());
  f.Evaluate(p.data(), r.data(), nullptr);
  return r;
}

// Max abs error between analytic local Jacobians and central finite differences.
// lps[k] == nullptr means the block is Euclidean (local == ambient).
static double fd_error(const CostFunction &f, std::vector<Eigen::VectorXd> xs, const std::vector<const LocalParameterization *> &lps) {
  const int nres = f.num_residuals();
  const int nb = (int)xs.size();
  const double eps = 1e-6;

  std::vector<const double *> p(nb);
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(nb);
  std::vector<double *> jp(nb, nullptr);
  for (int k = 0; k < nb; ++k) {
    p[k] = xs[k].data();
    J[k].resize(nres, xs[k].size());
    jp[k] = J[k].data();
  }
  Eigen::VectorXd r0(nres);
  f.Evaluate(p.data(), r0.data(), jp.data());

  double max_err = 0.0;
  for (int k = 0; k < nb; ++k) {
    const int g = (int)xs[k].size();
    const int l = lps[k] ? lps[k]->LocalSize() : g;
    Eigen::MatrixXd V = lps[k] ? lps[k]->PlusJacobian(xs[k].data()) : Eigen::MatrixXd::Identity(g, g);
    Eigen::MatrixXd Jl = J[k] * V; // analytic local (nres x l)
    for (int j = 0; j < l; ++j) {
      Eigen::VectorXd d = Eigen::VectorXd::Zero(l);
      d(j) = eps;
      std::vector<Eigen::VectorXd> xp = xs, xm = xs;
      if (lps[k]) {
        xp[k].resize(g);
        xm[k].resize(g);
        Eigen::VectorXd dp = d, dm = -d;
        lps[k]->Plus(xs[k].data(), dp.data(), xp[k].data());
        lps[k]->Plus(xs[k].data(), dm.data(), xm[k].data());
      } else {
        xp[k] = xs[k] + d;
        xm[k] = xs[k] - d;
      }
      Eigen::VectorXd col = (eval_res(f, xp) - eval_res(f, xm)) / (2.0 * eps);
      max_err = std::max(max_err, (col - Jl.col(j)).cwiseAbs().maxCoeff());
    }
  }
  return max_err;
}

static Eigen::Vector4d rand_quat(std::mt19937 &rng) {
  std::normal_distribution<double> N(0, 1);
  Eigen::Vector4d q(N(rng), N(rng), N(rng), N(rng));
  return ov_core::quatnorm(q);
}

static void test_imu_cpi(std::mt19937 &rng) {
  std::printf("[test] Factor_ImuCPIv1 analytic Jacobians vs finite differences\n");
  std::normal_distribution<double> N(0, 1);

  // Build a CPI measurement from a short IMU stream.
  Eigen::Vector3d bg_lin(0.01, -0.02, 0.005), ba_lin(0.1, 0.05, -0.08);
  ov_core::CpiV1 cpi(0.005, 1e-4, 0.02, 1e-3);
  cpi.setLinearizationPoints(bg_lin, ba_lin);
  double t = 0.0, dt = 0.01;
  Eigen::Vector3d w(0.3, -0.2, 0.15), a(0.2, -0.1, 9.9);
  for (int k = 0; k < 12; ++k) {
    Eigen::Vector3d w1 = w + 0.02 * Eigen::Vector3d(N(rng), N(rng), N(rng));
    Eigen::Vector3d a1 = a + 0.05 * Eigen::Vector3d(N(rng), N(rng), N(rng));
    cpi.feed_IMU(t, t + dt, w, a, w1, a1);
    w = w1;
    a = a1;
    t += dt;
  }

  Eigen::Vector3d grav(0, 0, 9.81);
  Factor_ImuCPIv1 f(cpi.DT, grav, cpi.alpha_tau, cpi.beta_tau, cpi.q_k2tau, cpi.b_a_lin, cpi.b_w_lin, cpi.J_q, cpi.J_b, cpi.J_a, cpi.H_b,
                    cpi.H_a, cpi.P_meas);

  // States set near the preintegrated prediction so the residual is ~0 -- the regime
  // the factor actually operates in during initialization. (Block order per clone:
  // q, bg, v, ba, p; 11th block: gravity on S².)
  // The analytic CPI Jacobian is a linearization, so the FD check is performed at a
  // representative operating point, not at random ~180-deg-apart states.
  State_JPLQuatLocal quat;
  GravityS2Parameterization grav_s2(grav.norm());
  Eigen::Vector4d q1 = rand_quat(rng);
  Eigen::Matrix3d R1 = ov_core::quat_2_Rot(q1);
  Eigen::Vector4d q2 = ov_core::quatnorm(ov_core::quat_multiply(cpi.q_k2tau, q1)); // zero orientation residual
  Eigen::Vector3d v1(0.4, -0.3, 0.2);
  Eigen::Vector3d v2 = v1 - grav * cpi.DT + R1.transpose() * cpi.beta_tau;
  Eigen::Vector3d p1(0.1, 0.2, -0.15);
  Eigen::Vector3d p2 = p1 + v1 * cpi.DT - 0.5 * grav * cpi.DT * cpi.DT + R1.transpose() * cpi.alpha_tau;
  std::vector<Eigen::VectorXd> xs(11);
  xs[0] = q1; xs[1] = bg_lin; xs[2] = v1; xs[3] = ba_lin; xs[4] = p1;
  xs[5] = q2; xs[6] = bg_lin; xs[7] = v2; xs[8] = ba_lin; xs[9] = p2;
  xs[10] = grav; // gravity (S² param, 3-global, 2-local)
  std::vector<const LocalParameterization *> lps = {&quat, nullptr, nullptr, nullptr, nullptr,
                                                    &quat, nullptr, nullptr, nullptr, nullptr,
                                                    &grav_s2};
  check_lt(fd_error(f, xs, lps), 1e-5, "Factor_ImuCPIv1 (incl. S2 gravity)");
}

static void test_generic_prior(std::mt19937 &rng) {
  std::printf("[test] Factor_GenericPrior analytic Jacobians vs finite differences\n");

  // Gauge-style prior: quat_yaw + position (matches the dynamic-init first-pose prior).
  {
    Eigen::Vector4d q0 = rand_quat(rng);
    Eigen::Vector3d p0(0.3, -0.1, 0.2);
    Eigen::MatrixXd x_lin(7, 1);
    x_lin.block(0, 0, 4, 1) = q0;
    x_lin.block(4, 0, 3, 1) = p0;
    std::vector<std::string> types = {"quat_yaw", "vec3"};
    Eigen::MatrixXd info = Eigen::MatrixXd::Identity(4, 4);
    info(0, 0) = 1.0 / (1e-3 * 1e-3);
    info.block(1, 1, 3, 3) *= 1.0 / (0.05 * 0.05);
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(4, 1);
    Factor_GenericPrior f(x_lin, types, info, grad);

    State_JPLQuatLocal quat;
    std::vector<Eigen::VectorXd> xs = {rand_quat(rng), Eigen::VectorXd((Eigen::Vector3d() << 0.31, -0.12, 0.18).finished())};
    std::vector<const LocalParameterization *> lps = {&quat, nullptr};
    check_lt(fd_error(f, xs, lps), 1e-5, "Factor_GenericPrior (quat_yaw + vec3)");
  }

  // Full-orientation prior (quat type, 3-dof).
  {
    Eigen::Vector4d q0 = rand_quat(rng);
    Eigen::MatrixXd x_lin(4, 1);
    x_lin.block(0, 0, 4, 1) = q0;
    std::vector<std::string> types = {"quat"};
    Eigen::MatrixXd info = Eigen::MatrixXd::Identity(3, 3) * (1.0 / (0.01 * 0.01));
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(3, 1);
    Factor_GenericPrior f(x_lin, types, info, grad);

    State_JPLQuatLocal quat;
    std::vector<Eigen::VectorXd> xs = {rand_quat(rng)};
    std::vector<const LocalParameterization *> lps = {&quat};
    check_lt(fd_error(f, xs, lps), 1e-5, "Factor_GenericPrior (quat)");
  }
}

// Standalone validation of the S² gravity retraction + tangent basis (independent of any
// factor): (a) B(g) is orthonormal and ⊥ g, (b) the retraction stays on the sphere of radius
// G for any step size, (c) ComputeJacobian == d Plus/d delta at delta=0 (central difference).
static void check_gravity_s2_at(const Eigen::Vector3d &g_dir, double G, const char *label) {
  Eigen::Vector3d g = G * g_dir.normalized(); // put it on the sphere
  GravityS2Parameterization param(G);
  const LocalParameterization &lp = param; // 2-arg Eigen Plus() wrapper lives on the base (hidden by the override)
  Eigen::Vector3d ghat = g.normalized();
  char name[96];

  // (a) tangent basis: orthonormal columns, both ⊥ g
  Eigen::MatrixXd B = param.PlusJacobian(g.data()); // 3x2, == ComputeJacobian at delta=0
  std::snprintf(name, sizeof(name), "S2 %-14s BtB == I2", label);
  check_lt((B.transpose() * B - Eigen::Matrix2d::Identity()).cwiseAbs().maxCoeff(), 1e-9, name);
  std::snprintf(name, sizeof(name), "S2 %-14s B cols perp g", label);
  check_lt((B.transpose() * ghat).cwiseAbs().maxCoeff(), 1e-9, name);

  // (b) retraction stays on the sphere across a wide range of step magnitudes
  std::mt19937 rng(12345);
  std::normal_distribution<double> N(0, 1);
  double max_sphere_err = 0.0;
  for (double scale : {1e-3, 1e-1, 1.0, 3.0})
    for (int t = 0; t < 8; ++t) {
      Eigen::Vector2d d(scale * N(rng), scale * N(rng));
      max_sphere_err = std::max(max_sphere_err, std::abs(lp.Plus(g,d).norm() - G));
    }
  std::snprintf(name, sizeof(name), "S2 %-14s |Plus| == G", label);
  check_lt(max_sphere_err, 1e-9, name);

  // (c) analytic Jacobian == central difference of Plus wrt delta at delta=0
  const double eps = 1e-6;
  double max_jac_err = 0.0;
  for (int j = 0; j < 2; ++j) {
    Eigen::Vector2d dp = Eigen::Vector2d::Zero(), dm = Eigen::Vector2d::Zero();
    dp(j) = eps;
    dm(j) = -eps;
    Eigen::Vector3d col = (lp.Plus(g,dp) - lp.Plus(g,dm)) / (2.0 * eps);
    max_jac_err = std::max(max_jac_err, (col - B.col(j)).cwiseAbs().maxCoeff());
  }
  std::snprintf(name, sizeof(name), "S2 %-14s J == dPlus/ddelta", label);
  check_lt(max_jac_err, 1e-6, name);
}

static void test_gravity_s2() {
  std::printf("[test] GravityS2Parameterization manifold (retraction + tangent basis)\n");
  const double G = 9.81;
  auto deg = [](double d) { return d * M_PI / 180.0; };
  check_gravity_s2_at(Eigen::Vector3d(0, 0, 1), G, "pole +Z");
  check_gravity_s2_at(Eigen::Vector3d(0, std::sin(deg(5)), std::cos(deg(5))), G, "tilt 5deg");
  check_gravity_s2_at(Eigen::Vector3d(0, std::sin(deg(30)), std::cos(deg(30))), G, "tilt 30deg");
  check_gravity_s2_at(Eigen::Vector3d(0, std::sin(deg(89)), std::cos(deg(89))), G, "tilt 89deg");
  check_gravity_s2_at(Eigen::Vector3d(0, 0, -1), G, "flipped -Z");
  // exercise each argmin-axis branch of the tangent-basis seed selection
  check_gravity_s2_at(Eigen::Vector3d(0.99, 0.10, 0.10), G, "mostly X");
  check_gravity_s2_at(Eigen::Vector3d(0.10, 0.99, 0.10), G, "mostly Y");
  check_gravity_s2_at(Eigen::Vector3d(0.10, 0.10, 0.99), G, "mostly Z");
  check_gravity_s2_at(Eigen::Vector3d(0.5001, 0.5, 0.707), G, "argmin edge");
}

int main() {
  std::printf("==== ov_init::zbft_sfm lifted-factor FD tests (vs ov_core) ====\n");
  std::mt19937 rng(7);
  test_imu_cpi(rng);
  test_generic_prior(rng);
  test_gravity_s2();
  std::printf("==== %d failures ====\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
