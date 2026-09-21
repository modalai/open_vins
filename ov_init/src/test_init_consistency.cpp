/*
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Matched Gaussian factor/solver Monte Carlo for ov_init::zbft_sfm with S² gravity.
 * NO OpenCV / NO Ceres dependency.
 *
 * Samples the full correlated CPI factor residual, Gaussian pixels and Gaussian prior
 * centers with the covariance used by the solver. Checks the local GN covariance of the
 * newest [θ, p, v, bg, ba] against its estimation errors, retaining the [11,20] ANEES band
 * around the first-order Gaussian expectation of 15. Nonlinear finite-sample coverage is
 * not an exact chi-square identity. This does NOT test raw noisy-IMU propagation or call
 * the public DynamicInitializer entry point, and does not certify its covariance.
 *
 * Also checks marginal block NEES, convergence, SPD/finite covariance, realignment,
 * recovery from flipped initial guesses and rejection of fixed invalid final gravity.
 *
 * Build (mirrors bench_zbft_s2; no OpenCV/Ceres):
 *   SFM=../../..; CF=ceres_free; OVC=$SFM/ov_core/src
 *   g++ -O2 -std=c++17 -pthread -I/usr/include/eigen3 -I. -I$OVC \
 *       test_init_consistency.cpp \
 *       ceres_free/Problem.cpp ceres_free/Parallel.cpp ceres_free/State_JPLQuatLocal.cpp \
 *       ceres_free/Factor_ImuCPIv1.cpp ceres_free/Factor_GenericPrior.cpp \
 *       $OVC/cpi/CpiV1.cpp -lpthread -o /tmp/test_consistency && /tmp/test_consistency
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "ceres_free/CostFunction.h"
#include "ceres_free/Factor_GenericPrior.h"
#include "ceres_free/Factor_ImuCPIv1.h"
#include "ceres_free/LocalParameterization.h"
#include "ceres_free/LossFunction.h"
#include "ceres_free/Problem.h"
#include "ceres_free/State_JPLQuatLocal.h"

#include "cpi/CpiV1.h"
#include "utils/quat_ops.h"

using ov_core::log_so3;
using ov_core::quat_2_Rot;
using ov_core::quat_multiply;
using ov_core::quatnorm;
using ov_core::rot_2_quat;
using ov_core::skew_x;
using namespace ov_init::zbft_sfm;

static bool finite_scalar(double value) {
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value), "64-bit IEEE double required");
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
template <typename Derived> static bool finite_matrix(const Eigen::MatrixBase<Derived> &value) {
  for (Eigen::Index j = 0; j < value.cols(); ++j)
    for (Eigen::Index i = 0; i < value.rows(); ++i)
      if (!finite_scalar(value(i, j))) return false;
  return true;
}
static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

using Vector15 = Eigen::Matrix<double, 15, 1>;
using Matrix15 = Eigen::Matrix<double, 15, 15>;

// ---------------------------------------------------------------------------
// Plain pinhole reprojection factor (no distortion); camera frame == IMU frame.
// Blocks: q_GtoIi(4), p_IiinG(3), p_FinG(3).
// ---------------------------------------------------------------------------
class PinholeFactor : public CostFunction {
public:
  Eigen::Vector2d uv;
  double fx, fy, cx, cy, isig;
  PinholeFactor(const Eigen::Vector2d &uv_, double fx_, double fy_, double cx_, double cy_, double sigma_px)
      : uv(uv_), fx(fx_), fy(fy_), cx(cx_), cy(cy_), isig(1.0 / sigma_px) {
    set_num_residuals(2);
    mutable_parameter_block_sizes()->push_back(4);
    mutable_parameter_block_sizes()->push_back(3);
    mutable_parameter_block_sizes()->push_back(3);
  }
  bool Evaluate(double const *const *p, double *res, double **jac) const override {
    Eigen::Matrix3d R = quat_2_Rot(Eigen::Map<const Eigen::Vector4d>(p[0]));
    Eigen::Vector3d pI = Eigen::Map<const Eigen::Vector3d>(p[1]);
    Eigen::Vector3d pf = Eigen::Map<const Eigen::Vector3d>(p[2]);
    Eigen::Vector3d fc = R * (pf - pI);
    double iz = 1.0 / fc(2);
    res[0] = isig * (fx * fc(0) * iz + cx - uv(0));
    res[1] = isig * (fy * fc(1) * iz + cy - uv(1));
    if (jac) {
      Eigen::Matrix<double, 2, 3> Hp;
      Hp << fx * iz, 0, -fx * fc(0) * iz * iz, 0, fy * iz, -fy * fc(1) * iz * iz;
      Hp *= isig;
      if (jac[0]) {
        Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(jac[0]);
        J.setZero();
        J.block(0, 0, 2, 3) = Hp * skew_x(R * (pf - pI));
      }
      if (jac[1]) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jac[1]);
        J = -Hp * R;
      }
      if (jac[2]) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jac[2]);
        J = Hp * R;
      }
    }
    return true;
  }
};

// ---------------------------------------------------------------------------
// A generative factor model. Clean CPIs provide means/Jacobians/covariances; each
// next truth state is constructed from one full correlated 15D residual draw.
// Bias random walks are part of that draw and have a separate state at each node.
// ---------------------------------------------------------------------------
struct Sim {
  int N = 10, M = 60;
  double dt = 0.1, fx = 400, fy = 400, cx = 320, cy = 240, sigma_px = 1.0;
  double imu_rate = 200.0, gmag = 9.81;
  double sw = 0.005, swb = 1e-4, sa = 0.02, sab = 1e-3; // CPI noise spectral densities
  Eigen::Vector3d grav;
  std::vector<Eigen::Vector4d> q;    // GT q_GtoIi
  std::vector<Eigen::Vector3d> p, v, bg, ba; // GT
  std::vector<Eigen::Vector3d> lm;   // GT landmarks (global)
  std::vector<std::shared_ptr<ov_core::CpiV1>> cpi; // nominal factor CPIs (size N)
  std::vector<std::vector<std::pair<int, Eigen::Vector2d>>> obs; // NOISY pixel observations
  double max_residual_mismatch = 0.0;
};

static Eigen::Vector3d randn3(std::mt19937 &rng, double s) {
  std::normal_distribution<double> N(0, 1);
  return Eigen::Vector3d(s * N(rng), s * N(rng), s * N(rng));
}

static Sim make_sim(std::mt19937 &rng) {
  Sim s;
  s.grav = Eigen::Vector3d(0, 0, s.gmag);

  auto omega_body = [](double t) {
    return Eigen::Vector3d(0.5 * std::sin(1.3 * t), 0.6 * std::sin(1.1 * t + 0.5), 0.4 * std::sin(0.9 * t + 1.0));
  };
  auto accel_body = [&](double t) { return Eigen::Vector3d(0.6 * std::sin(0.7 * t), 0.4 * std::cos(0.9 * t), s.gmag + 0.3 * std::sin(0.6 * t)); };

  s.q.resize(s.N);
  s.p.resize(s.N);
  s.v.resize(s.N);
  s.bg.resize(s.N);
  s.ba.resize(s.N);
  s.q[0] = Eigen::Vector4d(0, 0, 0, 1);
  s.p[0] = Eigen::Vector3d::Zero();
  s.v[0] = randn3(rng, 0.1);
  s.bg[0] = randn3(rng, 0.01);
  s.ba[0] = randn3(rng, 0.05);

  s.cpi.resize(s.N);
  s.cpi[0] = nullptr;
  const double dti = 1.0 / s.imu_rate;
  std::normal_distribution<double> N01(0, 1);
  for (int i = 1; i < s.N; ++i) {
    auto clean = std::make_shared<ov_core::CpiV1>(s.sw, s.swb, s.sa, s.sab, true);
    clean->setLinearizationPoints(s.bg[i - 1], s.ba[i - 1]);
    double t0 = (i - 1) * s.dt, t1 = i * s.dt;
    for (double t = t0; t < t1 - 1e-9; t += dti) {
      double tn = std::min(t + dti, t1);
      Eigen::Vector3d w0 = omega_body(t) + s.bg[i - 1], a0 = accel_body(t) + s.ba[i - 1];
      Eigen::Vector3d w1 = omega_body(tn) + s.bg[i - 1], a1 = accel_body(tn) + s.ba[i - 1];
      clean->feed_IMU(t, tn, w0, a0, w1, a1);
    }
    s.cpi[i] = clean;
    require(finite_matrix(clean->P_meas), "nonfinite CPI covariance");
    Eigen::LLT<Matrix15> chol(clean->P_meas);
    require(chol.info() == Eigen::Success, "CPI covariance is not SPD");
    Vector15 z;
    for (int k = 0; k < 15; ++k) z(k) = N01(rng);
    const Vector15 noise = chol.matrixL() * z;
    // Exact inverse of the factor's 2*q_error.xyz orientation residual. The
    // earlier normalized [0.5*dtheta,1] perturbation is only a first-order inverse.
    const Eigen::Vector3d half_theta = 0.5 * noise.head<3>();
    require(half_theta.squaredNorm() < 1.0, "orientation draw outside residual chart");
    Eigen::Vector4d q_error;
    q_error << half_theta, std::sqrt(1.0 - half_theta.squaredNorm());
    Eigen::Matrix3d R0 = quat_2_Rot(s.q[i - 1]);
    double DT = clean->DT;
    s.q[i] = quatnorm(quat_multiply(q_error, quat_multiply(clean->q_k2tau, s.q[i - 1])));
    s.bg[i] = s.bg[i - 1] + noise.segment<3>(3);
    s.v[i] = s.v[i - 1] - s.grav * DT + R0.transpose() * (clean->beta_tau + noise.segment<3>(6));
    s.ba[i] = s.ba[i - 1] + noise.segment<3>(9);
    s.p[i] = s.p[i - 1] + s.v[i - 1] * DT - 0.5 * s.grav * DT * DT +
             R0.transpose() * (clean->alpha_tau + noise.tail<3>());
    Factor_ImuCPIv1 factor(DT, s.grav, clean->alpha_tau, clean->beta_tau, clean->q_k2tau,
                          clean->b_a_lin, clean->b_w_lin, clean->J_q, clean->J_b, clean->J_a,
                          clean->H_b, clean->H_a, clean->P_meas);
    const double *truth[] = {s.q[i - 1].data(), s.bg[i - 1].data(), s.v[i - 1].data(), s.ba[i - 1].data(), s.p[i - 1].data(),
                            s.q[i].data(), s.bg[i].data(), s.v[i].data(), s.ba[i].data(), s.p[i].data(), s.grav.data()};
    Vector15 actual;
    require(factor.Evaluate(truth, actual.data(), nullptr), "truth residual evaluation failed");
    const Vector15 expected = factor.sqrtI_save * noise;
    const double mismatch = (actual - expected).lpNorm<Eigen::Infinity>();
    s.max_residual_mismatch = std::max(s.max_residual_mismatch, mismatch);
    require(finite_matrix(actual) && finite_matrix(expected) && mismatch < 1e-8,
            "constructed truth does not reproduce the drawn whitened CPI residual");
    require((factor.sqrtI_save * clean->P_meas * factor.sqrtI_save.transpose() - Matrix15::Identity()).cwiseAbs().maxCoeff() < 1e-9 &&
                std::abs(expected.squaredNorm() - z.squaredNorm()) < 1e-8 * (1.0 + z.squaredNorm()),
            "CPI draw and factor whitening do not represent the same covariance");
  }

  // Landmarks in front of cam0 at pose 0, mapped to global; noisy pixel observations.
  std::uniform_real_distribution<double> U(-0.4, 0.4), depth(3.0, 9.0);
  std::normal_distribution<double> npx(0, s.sigma_px);
  s.lm.resize(s.M);
  s.obs.resize(s.N);
  const Eigen::Matrix3d R0 = quat_2_Rot(s.q[0]);
  for (int j = 0; j < s.M; ++j) {
    double d = depth(rng);
    Eigen::Vector3d pcam(U(rng) * d, U(rng) * d, d);
    s.lm[j] = R0.transpose() * pcam + s.p[0];
  }
  for (int i = 0; i < s.N; ++i) {
    Eigen::Matrix3d R = quat_2_Rot(s.q[i]);
    for (int j = 0; j < s.M; ++j) {
      Eigen::Vector3d fc = R * (s.lm[j] - s.p[i]);
      if (fc(2) < 0.5)
        continue;
      double u = s.fx * fc(0) / fc(2) + s.cx, vv = s.fy * fc(1) / fc(2) + s.cy;
      if (u < 5 || u > 635 || vv < 5 || vv > 475)
        continue;
      s.obs[i].push_back({j, Eigen::Vector2d(u + npx(rng), vv + npx(rng))});
    }
  }
  return s;
}

struct InitGuess {
  std::vector<Eigen::Vector4d> q;
  std::vector<Eigen::Vector3d> p, v, lm, bg, ba;
  Eigen::Vector3d grav, gravity_prior;
  Eigen::Matrix<double, 12, 1> prior_residual;
};
static Eigen::Vector4d perturb_q(const Eigen::Vector4d &q, const Eigen::Vector3d &dth) {
  Eigen::Vector4d dq;
  dq << 0.5 * dth, 1.0;
  return quatnorm(quat_multiply(quatnorm(dq), q));
}
// Optional mismatch experiments retain explicit argv knobs. The default draw
// matches the factor prior; intentional inflation/mismatched priors are not a
// calibrated-Gaussian coverage experiment and may fail the unchanged band.
static double g_ba_prior_sigma = 0.10;
static double g_ba_seed_err = 0.10;

static InitGuess make_init(const Sim &s, std::mt19937 &rng, double grav_perturb_deg = 0.0) {
  InitGuess g;
  g.q = s.q;
  g.p = s.p;
  g.v = s.v;
  g.lm = s.lm;
  for (int i = 0; i < s.N; ++i) {
    if (i > 0) {
      g.q[i] = perturb_q(s.q[i], randn3(rng, 0.03));
      g.p[i] = s.p[i] + randn3(rng, 0.03);
    }
    g.v[i] = s.v[i] + randn3(rng, 0.08);
  }
  for (int j = 0; j < s.M; ++j)
    g.lm[j] = s.lm[j] + randn3(rng, 0.08);
  for (int j = 0; j < 4; ++j) g.prior_residual.segment<3>(3 * j) = randn3(rng, 1.0);
  g.q[0] = rot_2_quat(quat_2_Rot(s.q[0]) * ov_core::exp_so3(-0.001 * g.prior_residual.head<3>()));
  g.p[0] = s.p[0] - 0.001 * g.prior_residual.segment<3>(3);
  g.bg.assign(s.N, s.bg[0] - 0.05 * g.prior_residual.segment<3>(6));
  g.ba.assign(s.N, s.ba[0] - g_ba_seed_err * g.prior_residual.tail<3>());
  g.prior_residual.tail<3>() *= g_ba_seed_err / g_ba_prior_sigma;
  g.gravity_prior = s.grav + randn3(rng, 0.5);
  g.grav = s.grav;
  if (grav_perturb_deg > 0) {
    Eigen::Vector3d axis = randn3(rng, 1.0).normalized();
    g.grav = Eigen::AngleAxisd(grav_perturb_deg * M_PI / 180.0, axis) * s.grav;
  }
  return g;
}

// Full-orientation first-pose prior matching the real ceres-free DynamicInitializer path.
static Factor_GenericPrior *make_prior_s2(const double *q0, const double *p0, const double *bg0, const double *ba0) {
  Eigen::MatrixXd x_lin = Eigen::MatrixXd::Zero(13, 1);
  for (int j = 0; j < 4; j++)
    x_lin(j) = q0[j];
  for (int j = 0; j < 3; j++) {
    x_lin(4 + j) = p0[j];
    x_lin(7 + j) = bg0[j];
    x_lin(10 + j) = ba0[j];
  }
  Eigen::MatrixXd info = Eigen::MatrixXd::Identity(12, 12);
  info.block(0, 0, 3, 3) *= 1.0 / std::pow(0.001, 2); // orientation (full quat)
  info.block(3, 3, 3, 3) *= 1.0 / std::pow(0.001, 2); // position
  info.block(6, 6, 3, 3) *= 1.0 / std::pow(0.05, 2);  // bias_g
  info.block(9, 9, 3, 3) *= 1.0 / std::pow(g_ba_prior_sigma, 2);  // bias_a (reset mode: tightened)
  std::vector<std::string> types = {"quat", "vec3", "vec3", "vec3"};
  return new Factor_GenericPrior(x_lin, types, info, Eigen::MatrixXd::Zero(12, 1));
}

struct Result {
  bool ok = false, cov_ok = false, rejected_flip = false;
  int iters = 0;
  std::string msg;
  Eigen::Matrix<double, 15, 1> err = Eigen::Matrix<double, 15, 1>::Zero();
  Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  double grav_err_deg = 0, final_cost = 0, decrement = 0;
};

// Run the Gaussian factor graph (free S² gravity, full-quat prior, per-node biases),
// recover the 15x15 covariance of the newest IMU state, and form the error vs GT.
static Result run_init(const Sim &s, const InitGuess &g, int gmode, bool fixed_bad_gravity = false) {
  // gmode: 0 = free S² gravity; 1 = free + weak gravity prior; 2 = gravity FIXED at GT
  Result r;
  std::vector<Eigen::Vector4d> q = g.q;
  std::vector<Eigen::Vector3d> p = g.p, v = g.v, lm = g.lm, bg = g.bg, ba = g.ba;
  Eigen::Vector3d grav = (gmode == 2) ? s.grav : g.grav;
  if (fixed_bad_gravity) grav = -s.grav;

  Problem problem;
  problem.EnableOwnership();

  auto *gs2 = new GravityS2Parameterization(s.gmag);
  problem.AddParameterBlock(grav.data(), 3, gs2);
  if (gmode == 2 || fixed_bad_gravity)
    problem.SetParameterBlockConstant(grav.data());
  if (gmode == 1) {
    Eigen::MatrixXd glin(3, 1);
    glin = g.gravity_prior; // noisy observation, independent of the optimizer's starting guess
    Eigen::MatrixXd ginfo = Eigen::MatrixXd::Identity(3, 3) / std::pow(0.5, 2); // σ≈0.5 m/s² ≈ 3°
    std::vector<std::string> gt = {"vec3"};
    problem.AddResidualBlock(new Factor_GenericPrior(glin, gt, ginfo, Eigen::MatrixXd::Zero(3, 1)), nullptr, {grav.data()});
  }
  for (int i = 0; i < s.N; ++i) {
    auto *qparam = new State_JPLQuatLocal();
    problem.AddParameterBlock(q[i].data(), 4, qparam);
    problem.AddParameterBlock(p[i].data(), 3);
    problem.AddParameterBlock(v[i].data(), 3);
    problem.AddParameterBlock(bg[i].data(), 3);
    problem.AddParameterBlock(ba[i].data(), 3);
    if (i == 0) {
      auto *prior = make_prior_s2(g.q[0].data(), g.p[0].data(), g.bg[0].data(), g.ba[0].data());
      const double *truth[] = {s.q[0].data(), s.p[0].data(), s.bg[0].data(), s.ba[0].data()};
      Eigen::Matrix<double, 12, 1> residual;
      prior->Evaluate(truth, residual.data(), nullptr);
      require(finite_matrix(residual) && (residual - g.prior_residual).lpNorm<Eigen::Infinity>() < 1e-9,
              "sampled prior center does not reproduce its drawn residual");
      problem.AddResidualBlock(prior, nullptr, {q[0].data(), p[0].data(), bg[0].data(), ba[0].data()});
    }
    if (i > 0) {
      auto c = s.cpi[i];
      auto *f = new Factor_ImuCPIv1(c->DT, grav, c->alpha_tau, c->beta_tau, c->q_k2tau, c->b_a_lin, c->b_w_lin, c->J_q, c->J_b, c->J_a,
                                    c->H_b, c->H_a, c->P_meas);
      problem.AddResidualBlock(f, nullptr,
                               {q[i - 1].data(), bg[i - 1].data(), v[i - 1].data(), ba[i - 1].data(), p[i - 1].data(), q[i].data(), bg[i].data(),
                                v[i].data(), ba[i].data(), p[i].data(), grav.data()});
    }
  }
  for (int j = 0; j < s.M; ++j) {
    problem.AddParameterBlock(lm[j].data(), 3);
    problem.SetSchurLandmark(lm[j].data());
  }
  for (int i = 0; i < s.N; ++i)
    for (auto &o : s.obs[i])
      problem.AddResidualBlock(new PinholeFactor(o.second, s.fx, s.fy, s.cx, s.cy, s.sigma_px), nullptr,
                               {q[i].data(), p[i].data(), lm[o.first].data()});

  SolverOptions o;
  o.use_dogleg = false;
  o.num_threads = 1; // deterministic for NEES
  o.max_num_iterations = 200; // converge the test model; not the production initializer's budget
  o.max_solver_time_seconds = std::numeric_limits<double>::max(); // iteration-bounded; machine load must not censor trials
  o.function_tolerance = 1e-8;
  o.gradient_tolerance = 1e-9;
  SolverSummary sum = problem.Solve(o);
  r.ok = sum.converged && !sum.time_stopped && finite_scalar(sum.final_cost);
  r.iters = sum.iterations;
  r.msg = sum.message;
  r.final_cost = sum.final_cost;

  // The direction gate evaluates the FINAL gravity, never the starting guess.
  // This is a fixture check, not the public initializer's entry-point gates.
  if (!finite_matrix(grav) || !(grav.norm() > 0.0)) { r.ok = false; return r; }
  Eigen::Vector3d gexp(0, 0, s.gmag);
  double ang = std::acos(std::min(1.0, std::max(-1.0, grav.dot(gexp) / (grav.norm() * gexp.norm())))) * 180.0 / M_PI;
  r.grav_err_deg = ang;
  if (ang > 30.0) {
    r.rejected_flip = true;
    return r;
  }
  for (int i = 0; i < s.N; ++i)
    r.ok = r.ok && finite_matrix(q[i]) && finite_matrix(p[i]) && finite_matrix(v[i]) && finite_matrix(bg[i]) && finite_matrix(ba[i]);
  for (const auto &point : lm) r.ok = r.ok && finite_matrix(point);
  if (!r.ok) return r;

  // Do not accept the solver's "max damping (stationary)" label without
  // checking the remaining GN correction. The tolerance is the solver's own
  // requested relative cost tolerance, expressed as twice predicted decrease.
  Eigen::MatrixXd no_keep;
  Eigen::VectorXd no_gradient;
  Problem::ExportStats stationarity;
  if (!problem.ExportReducedInformation({}, no_keep, no_gradient, o, &stationarity) ||
      !finite_scalar(stationarity.nuis_decrement) || stationarity.nuis_decrement < 0.0) {
    r.ok = false;
    r.msg += "; invalid stationarity export";
    return r;
  }
  r.decrement = stationarity.nuis_decrement;
  if (r.decrement > 2.0 * o.function_tolerance * std::max(1.0, r.final_cost)) {
    r.ok = false;
    r.msg += "; remaining GN correction exceeds requested tolerance";
    return r;
  }

  int L = s.N - 1;
  std::vector<double *> blk = {q[L].data(), p[L].data(), v[L].data(), bg[L].data(), ba[L].data()};
  Eigen::MatrixXd C;
  if (problem.ComputeCovariance(blk, C, o) && C.rows() == 15 && C.cols() == 15 && finite_matrix(C) &&
      (C - C.transpose()).cwiseAbs().maxCoeff() < 1e-10 * std::max(1.0, C.cwiseAbs().maxCoeff())) {
    Eigen::LLT<Matrix15> chol(C);
    r.cov_ok = chol.info() == Eigen::Success;
    r.cov = C;
  }

  // Error of the newest IMU state vs GT, in the solver's local error coordinates.
  Eigen::Matrix3d R_gt = quat_2_Rot(s.q[L]), R_est = quat_2_Rot(q[L]);
  r.rotation = R_est;
  r.err.segment<3>(0) = -log_so3(R_gt * R_est.transpose());
  r.err.segment<3>(3) = s.p[L] - p[L];
  r.err.segment<3>(6) = s.v[L] - v[L];
  r.err.segment<3>(9) = s.bg[L] - bg[L];
  r.err.segment<3>(12) = s.ba[L] - ba[L];
  r.ok = r.ok && finite_matrix(r.err);
  return r;
}

// Validate the Phase-1 re-align covariance transform T = blkdiag(I, R, R, I, I): the error of a
// world-frame-re-aligned estimate vs the re-aligned truth must equal T times the original error
// (in particular delta-theta is INVARIANT -> T_theta_theta = I). If T were wrong (e.g. R on theta),
// the orientation rows would mismatch. This is the numerical check behind the production fix.
static bool check_realign_transform() {
  std::printf("[test] re-align covariance transform T=blkdiag(I,R,R,I,I) (Phase-1 fix)\n");
  std::mt19937 rng(55);
  std::uniform_real_distribution<double> U(0, 1);
  double maxerr = 0;
  for (int t = 0; t < 300; ++t) {
    Eigen::Vector4d q_gt = quatnorm(Eigen::Vector4d(U(rng) - 0.5, U(rng) - 0.5, U(rng) - 0.5, U(rng) - 0.5));
    Eigen::Vector4d q_est = perturb_q(q_gt, randn3(rng, 0.05));
    Eigen::Vector3d p_gt = randn3(rng, 1), p_est = p_gt + randn3(rng, 0.05);
    Eigen::Vector3d v_gt = randn3(rng, 1), v_est = v_gt + randn3(rng, 0.05);
    Eigen::Vector3d bg_gt = randn3(rng, 0.01), bg_est = bg_gt + randn3(rng, 0.005);
    Eigen::Vector3d ba_gt = randn3(rng, 0.05), ba_est = ba_gt + randn3(rng, 0.02);

    Eigen::Matrix<double, 15, 1> e0;
    e0.segment<3>(0) = -log_so3(quat_2_Rot(q_gt) * quat_2_Rot(q_est).transpose());
    e0.segment<3>(3) = p_gt - p_est;
    e0.segment<3>(6) = v_gt - v_est;
    e0.segment<3>(9) = bg_gt - bg_est;
    e0.segment<3>(12) = ba_gt - ba_est;

    // Random world-frame re-align R (5-25 deg), applied as production does: R_new = R_old * R^T.
    Eigen::Vector3d axis = randn3(rng, 1).normalized();
    Eigen::Matrix3d R = Eigen::AngleAxisd((5.0 + 20.0 * U(rng)) * M_PI / 180.0, axis).toRotationMatrix();
    auto realign_q = [&](const Eigen::Vector4d &q) { return rot_2_quat(quat_2_Rot(q) * R.transpose()); };

    Eigen::Matrix<double, 15, 1> e1;
    e1.segment<3>(0) = -log_so3(quat_2_Rot(realign_q(q_gt)) * quat_2_Rot(realign_q(q_est)).transpose());
    e1.segment<3>(3) = R * p_gt - R * p_est;
    e1.segment<3>(6) = R * v_gt - R * v_est;
    e1.segment<3>(9) = bg_gt - bg_est;
    e1.segment<3>(12) = ba_gt - ba_est;

    Eigen::Matrix<double, 15, 15> T = Eigen::Matrix<double, 15, 15>::Identity();
    T.block(3, 3, 3, 3) = R;
    T.block(6, 6, 3, 3) = R;
    maxerr = std::max(maxerr, (e1 - T * e0).cwiseAbs().maxCoeff());
  }
  std::printf("  max |err_realigned - T*err| over 300 cases: %.3e  %s\n\n", maxerr, maxerr < 1e-9 ? "[ok]" : "[FAIL]");
  return finite_scalar(maxerr) && maxerr < 1e-9;
}

static double nees(const Vector15 &error, const Matrix15 &covariance) {
  require(finite_matrix(error) && finite_matrix(covariance), "nonfinite NEES input");
  Eigen::LLT<Matrix15> chol(covariance);
  require(chol.info() == Eigen::Success, "NEES covariance is not SPD");
  const double value = error.dot(chol.solve(error));
  require(finite_scalar(value) && value >= 0.0, "invalid joint NEES");
  return value;
}

static double marginal_nees(const Vector15 &error, const Matrix15 &covariance, int offset) {
  // A marginal precision is inverse(C_bb), not the bb block of inverse(C).
  const Eigen::Matrix3d marginal = covariance.block<3, 3>(offset, offset);
  Eigen::LLT<Eigen::Matrix3d> chol(marginal);
  require(chol.info() == Eigen::Success, "marginal covariance is not SPD");
  const Eigen::Vector3d e = error.segment<3>(offset);
  const double score = e.dot(chol.solve(e));
  require(finite_scalar(score) && score >= 0.0, "invalid marginal NEES");
  return score;
}

static int run_test(int argc, char **argv) {
  const bool realign_ok = check_realign_transform();
  // Analytic correlated example: marginal p_x variance is 1, while the full
  // precision's p_x diagonal is 1/(1-.8^2). Catch the previous metric directly.
  Matrix15 correlated = Matrix15::Identity();
  correlated(3, 6) = correlated(6, 3) = 0.8;
  Vector15 unit_error = Vector15::Zero();
  unit_error(3) = 1.0;
  require(std::abs(marginal_nees(unit_error, correlated, 3) - 1.0) < 1e-12 &&
              std::abs(nees(unit_error, correlated) - 1.0 / 0.36) < 1e-12,
          "correlated covariance marginal/joint NEES regression");
  int K = (argc > 1) ? atoi(argv[1]) : 500;
  int gmode = (argc > 2) ? atoi(argv[2]) : 0;     // 0=free, 1=free+prior, 2=fixed@GT
  int do_inflate = (argc > 3) ? atoi(argv[3]) : 0; // apply the production init_dyn_inflation_* as a congruence
  g_ba_prior_sigma = (argc > 4) ? atof(argv[4]) : 0.10; // ba prior sigma (0.10 legacy; ~0.02 reset-prior mode)
  g_ba_seed_err = (argc > 5) ? atof(argv[5]) : g_ba_prior_sigma;
  require(K > 0 && gmode >= 0 && gmode <= 2 && finite_scalar(g_ba_prior_sigma) && g_ba_prior_sigma > 0.0 &&
              finite_scalar(g_ba_seed_err) && g_ba_seed_err >= 0.0, "invalid Monte Carlo arguments");
  const char *mname[3] = {"free-S2", "free-S2 + grav-prior", "gravity-FIXED@GT"};
  std::printf("==== Gaussian factor/solver NEES (%d trials, gravity=%s, inflation=%s) ====\n", K,
              mname[gmode], do_inflate ? "ON" : "off");
  std::printf("scope: full correlated CPI residual draws, Gaussian pixels and matched prior centers; not raw CPI/public initializer\n");
  if (do_inflate || g_ba_seed_err != g_ba_prior_sigma)
    std::printf("intentional covariance/prior mismatch requested: the ideal-15 expectation no longer applies\n");

  double sum_nees = 0, sum_th = 0, sum_p = 0, sum_v = 0, sum_bg = 0, sum_ba = 0;
  double sum_grav_err = 0, sum_iters = 0;
  double max_residual_mismatch = 0, max_decrement = 0;
  std::string msg0;
  int ok = 0, cov_ok = 0, used = 0;
  for (int t = 0; t < K; ++t) {
    std::mt19937 rng(2000 + t);
    Sim s = make_sim(rng);
    max_residual_mismatch = std::max(max_residual_mismatch, s.max_residual_mismatch);
    InitGuess g = make_init(s, rng, 0.0);
    Result r = run_init(s, g, gmode);
    ok += r.ok;
    sum_iters += r.iters;
    if (t == 0)
      msg0 = r.msg;
    sum_grav_err += r.grav_err_deg;
    max_decrement = std::max(max_decrement, r.decrement);
    if (!r.ok || !r.cov_ok || r.rejected_flip) {
      std::printf("FAIL trial %d: solve=%d covariance=%d gravity_reject=%d qn=%.6g cost=%.6g %s\n", t,
                  (int)r.ok, (int)r.cov_ok, (int)r.rejected_flip, r.decrement, r.final_cost, r.msg.c_str());
      continue;
    }
    cov_ok++;
    if (do_inflate) {
      // Production inflation as a proper congruence S·Cov·Sᵀ (Phase-1 form): orientation/velocity
      // variance ×10, gyro/accel-bias variance ×100, position unscaled. (init_dyn_inflation_*)
      Eigen::Matrix<double, 15, 1> sd;
      sd.segment<3>(0).setConstant(std::sqrt(10.0));
      sd.segment<3>(3).setConstant(1.0);
      sd.segment<3>(6).setConstant(std::sqrt(10.0));
      sd.segment<3>(9).setConstant(10.0);
      sd.segment<3>(12).setConstant(10.0);
      Eigen::Matrix<double, 15, 15> S = sd.asDiagonal();
      r.cov = (S * r.cov * S).eval();
    }
    const double joint_nees = nees(r.err, r.cov);
    used++;
    sum_nees += joint_nees;
    double *marginal_sums[] = {&sum_th, &sum_p, &sum_v, &sum_bg, &sum_ba};
    for (int b = 0; b < 5; ++b) *marginal_sums[b] += marginal_nees(r.err, r.cov, 3 * b);
  }
  double iU = 1.0 / std::max(1, used);
  double anees = sum_nees * iU;
  std::printf("converged: %d/%d   cov_ok: %d   used: %d   mean iters: %.1f   trial0 msg: \"%s\"\n", ok, K, cov_ok, used,
              sum_iters / std::max(1, K), msg0.c_str());
  std::printf("mean gravity err: %.4f deg\n", sum_grav_err / std::max(1, K));
  std::printf("max whitened CPI truth/draw mismatch: %.3e   max remaining GN energy: %.6g\n", max_residual_mismatch, max_decrement);
  std::printf("ANEES (total, ideal=15):   %.3f   [ANEES/15 = %.3f, ideal 1.0]\n", anees, anees / 15.0);
  std::printf("  marginal mean NEES (ideal=3 each):  theta=%.3f  p=%.3f  v=%.3f  bg=%.3f  ba=%.3f\n", sum_th * iU, sum_p * iU,
              sum_v * iU, sum_bg * iU, sum_ba * iU);
  bool consistent = (anees > 11.0 && anees < 20.0); // generous band around 15
  std::printf("CONSISTENCY: %s  (unchanged ANEES band [11,20])\n", consistent ? "PASS (in band)" : "OUT OF BAND");

  // An erroneous initial guess is recoverable. Keep the same measurements and
  // physical prior centers; changing the gravity guess does not change a prior.
  int flips = 50, matched = 0, other_basin = 0, refused = 0, references_ok = 0, rejected_bad_final = 0;
  double accepted_flip_nees = 0;
  bool known_recovery_ok = false;
  for (int t = 0; t < flips; ++t) {
    std::mt19937 rng(9000 + t);
    Sim s = make_sim(rng);
    InitGuess g = make_init(s, rng, 0.0);
    const Result reference = run_init(s, g, 0);
    const bool reference_ok = reference.ok && reference.cov_ok && !reference.rejected_flip;
    references_ok += reference_ok;
    g.grav = s.gmag * (-s.grav + randn3(rng, 0.5)).normalized();
    Result r = run_init(s, g, 0); // explicitly free S² for this recovery check
    if (r.ok && r.cov_ok && !r.rejected_flip) {
      accepted_flip_nees += nees(r.err, r.cov);
      bool same_optimum = false;
      if (reference_ok) {
        Vector15 difference = r.err - reference.err;
        difference.head<3>() = -log_so3(reference.rotation * r.rotation.transpose());
        const double state_distance = nees(difference, reference.cov);
        const double cost_distance = std::abs(r.final_cost - reference.final_cost);
        // Comparison precision, not a new physical acceptance gate: 100x the
        // solve's relative cost tolerance and 0.032 marginal-sigma state norm.
        same_optimum = cost_distance < 1e-6 * std::max(1.0, reference.final_cost) && state_distance < 1e-3;
        if (t == 0) {
          known_recovery_ok = same_optimum;
          std::printf("known recoverable seed 9000: |cost difference| %.3e, state Mahalanobis^2 %.3e %s\n",
                      cost_distance, state_distance, same_optimum ? "PASS" : "FAIL");
        }
      }
      if (same_optimum) ++matched;
      else ++other_basin;
    } else {
      ++refused;
    }
    const Result bad = run_init(s, g, 0, true);
    if (bad.rejected_flip && !bad.cov_ok) ++rejected_bad_final;
  }
  std::printf("ADVERSARIAL FLIPPED STARTS (diagnostic): %d match ordinary optimum, %d finite/SPD other basins, %d refused\n",
              matched, other_basin, refused);
  std::printf("  accepted-flip ANEES %.3f; selection/basin diagnostic, not calibrated coverage\n",
              accepted_flip_nees / std::max(1, matched + other_basin));
  std::printf("  LIMITATION: convergence + SPD + final gravity <30deg do not guarantee the correct basin; public initializer not exercised\n");
  std::printf("FIXED INVALID FINAL GRAVITY: %d/%d rejected without covariance\n", rejected_bad_final, flips);

  return (realign_ok && consistent && ok == K && cov_ok == K && used == K && references_ok == flips &&
          known_recovery_ok && rejected_bad_final == flips) ? 0 : 1;
}

int main(int argc, char **argv) {
  try { return run_test(argc, argv); }
  catch (const std::exception &error) {
    std::printf("FAIL: %s\n", error.what());
    return 1;
  }
}
