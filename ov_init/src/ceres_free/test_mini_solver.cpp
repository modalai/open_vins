/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * Standalone validation of the Ceres-free solver CORE (ov_init::zbft_sfm). It depends
 * ONLY on Eigen + the solver core (Problem.cpp, Parallel.cpp) -- NOT on ov_core --
 * so it can be compiled and run anywhere:
 *
 *   g++ -O2 -std=c++17 -pthread -I/usr/include/eigen3 \
 *       test_mini_solver.cpp Problem.cpp Parallel.cpp -o /tmp/test_mini && /tmp/test_mini
 *
 * It exercises: analytic Jacobians vs finite differences (incl. a non-identity
 * manifold V), linear-LS optimality, Schur-vs-dense equivalence, lock-free
 * parallel determinism (run-to-run bitwise; cross-thread to round-off), and
 * covariance (Schur marginalization vs dense full-inverse nav block).
 *
 * The lifted IMU/reprojection/prior factors (which need ov_core) are validated
 * separately by test_mini_factors.cpp under the full ov_init build.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <cstdio>
#include <chrono>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include "LocalParameterization.h"
#include "LandmarkQr.h"
#include "LossFunction.h"
#include "Problem.h"

using namespace ov_init::zbft_sfm;

// ----------------------------------------------------------------------------
// Tiny test harness
// ----------------------------------------------------------------------------
static int g_failures = 0;
static int g_checks = 0;
static void check_lt(double value, double tol, const char *name) {
  ++g_checks;
  if (!(value < tol) || !std::isfinite(value)) {
    std::printf("  [FAIL] %-46s  value=%.3e  tol=%.1e\n", name, value, tol);
    ++g_failures;
  } else {
    std::printf("  [ ok ] %-46s  value=%.3e\n", name, value);
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

// ----------------------------------------------------------------------------
// Self-contained factors (Euclidean blocks unless noted)
// ----------------------------------------------------------------------------

// r = A x - b  (single block of size A.cols())
class LinearFactor : public CostFunction {
public:
  LinearFactor(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) : A_(A), b_(b) {
    set_num_residuals((int)A.rows());
    mutable_parameter_block_sizes()->push_back((int)A.cols());
  }
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    Eigen::Map<const Eigen::VectorXd> x(parameters[0], A_.cols());
    Eigen::Map<Eigen::VectorXd>(residuals, A_.rows()) = A_ * x - b_;
    if (jacobians && jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(jacobians[0], A_.rows(), A_.cols()) = A_;
    }
    return true;
  }

private:
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
};

// r = w * (xa - xb - d)   (two blocks of size 3) -- the BA "observation" coupling
class DiffFactor : public CostFunction {
public:
  DiffFactor(const Eigen::Vector3d &d, double w) : d_(d), w_(w) {
    set_num_residuals(3);
    mutable_parameter_block_sizes()->push_back(3);
    mutable_parameter_block_sizes()->push_back(3);
  }
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    Eigen::Map<const Eigen::Vector3d> xa(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> xb(parameters[1]);
    Eigen::Map<Eigen::Vector3d> res(residuals);
    res = w_ * (xa - xb - d_);
    if (jacobians) {
      if (jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J0(jacobians[0]);
        J0 = w_ * Eigen::Matrix3d::Identity();
      }
      if (jacobians[1]) {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J1(jacobians[1]);
        J1 = -w_ * Eigen::Matrix3d::Identity();
      }
    }
    return true;
  }

private:
  Eigen::Vector3d d_;
  double w_;
};

// r = S * (x - x0)   (single block) -- a Gaussian anchor / gauge prior
class AnchorFactor : public CostFunction {
public:
  AnchorFactor(const Eigen::MatrixXd &S, const Eigen::VectorXd &x0) : S_(S), x0_(x0) {
    set_num_residuals((int)S.rows());
    mutable_parameter_block_sizes()->push_back((int)x0.size());
  }
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    Eigen::Map<const Eigen::VectorXd> x(parameters[0], x0_.size());
    Eigen::Map<Eigen::VectorXd>(residuals, S_.rows()) = S_ * (x - x0_);
    if (jacobians && jacobians[0])
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(jacobians[0], S_.rows(), x0_.size()) = S_;
    return true;
  }

private:
  Eigen::MatrixXd S_;
  Eigen::VectorXd x0_;
};

// Rectangular manifold: global 3, local 2 (perturb first two coords). Exercises a
// non-identity plus-Jacobian V = [[1,0],[0,1],[0,0]] in the assembly path.
class ProjParam : public LocalParameterization {
public:
  bool Plus(const double *x, const double *d, double *xpd) const override {
    xpd[0] = x[0] + d[0];
    xpd[1] = x[1] + d[1];
    xpd[2] = x[2];
    return true;
  }
  bool ComputeJacobian(const double *, double *j) const override {
    // row-major 3x2
    j[0] = 1; j[1] = 0;
    j[2] = 0; j[3] = 1;
    j[4] = 0; j[5] = 0;
    return true;
  }
  int GlobalSize() const override { return 3; }
  int LocalSize() const override { return 2; }
};

// ----------------------------------------------------------------------------
// Finite-difference Jacobian check (perturbs in LOCAL coordinates via Plus)
// ----------------------------------------------------------------------------
static Eigen::VectorXd eval_res(const CostFunction &f, const std::vector<Eigen::VectorXd> &xs) {
  std::vector<const double *> p(xs.size());
  for (size_t i = 0; i < xs.size(); ++i)
    p[i] = xs[i].data();
  Eigen::VectorXd r(f.num_residuals());
  f.Evaluate(p.data(), r.data(), nullptr);
  return r;
}

static double fd_jacobian_error(const CostFunction &f, std::vector<Eigen::VectorXd> xs,
                                const std::vector<const LocalParameterization *> &lps) {
  const int nres = f.num_residuals();
  const int nb = (int)xs.size();
  const double eps = 1e-6;

  // Analytic ambient Jacobians.
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
    Eigen::MatrixXd Jl = J[k] * V; // nres x l (analytic, local)

    for (int j = 0; j < l; ++j) {
      Eigen::VectorXd dl = Eigen::VectorXd::Zero(l);
      dl(j) = eps;
      std::vector<Eigen::VectorXd> xp = xs, xm = xs;
      if (lps[k]) {
        xp[k].resize(g);
        xm[k].resize(g);
        Eigen::VectorXd dlp = dl, dlm = -dl;
        lps[k]->Plus(xs[k].data(), dlp.data(), xp[k].data());
        lps[k]->Plus(xs[k].data(), dlm.data(), xm[k].data());
      } else {
        xp[k] = xs[k] + dl;
        xm[k] = xs[k] - dl;
      }
      Eigen::VectorXd col = (eval_res(f, xp) - eval_res(f, xm)) / (2.0 * eps);
      max_err = std::max(max_err, (col - Jl.col(j)).cwiseAbs().maxCoeff());
    }
  }
  return max_err;
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------
static void test_fd_jacobians(std::mt19937 &rng) {
  std::printf("[test] finite-difference Jacobians\n");
  std::normal_distribution<double> N(0.0, 1.0);
  auto randv = [&](int n) { Eigen::VectorXd v(n); for (int i = 0; i < n; ++i) v(i) = N(rng); return v; };
  auto randm = [&](int r, int c) { Eigen::MatrixXd m(r, c); for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) m(i, j) = N(rng); return m; };

  {
    LinearFactor f(randm(4, 3), randv(4));
    std::vector<Eigen::VectorXd> xs = {randv(3)};
    std::vector<const LocalParameterization *> lps = {nullptr};
    check_lt(fd_jacobian_error(f, xs, lps), 1e-6, "LinearFactor jacobian (euclidean)");
  }
  {
    DiffFactor f(Eigen::Vector3d(0.1, -0.2, 0.3), 1.7);
    std::vector<Eigen::VectorXd> xs = {randv(3), randv(3)};
    std::vector<const LocalParameterization *> lps = {nullptr, nullptr};
    check_lt(fd_jacobian_error(f, xs, lps), 1e-6, "DiffFactor jacobian (two blocks)");
  }
  {
    ProjParam proj;
    LinearFactor f(randm(2, 3), randv(2));
    std::vector<Eigen::VectorXd> xs = {randv(3)};
    std::vector<const LocalParameterization *> lps = {&proj};
    check_lt(fd_jacobian_error(f, xs, lps), 1e-6, "rectangular manifold V (global 3, local 2)");
  }
}

static void test_linear_solve(std::mt19937 &rng) {
  std::printf("[test] linear least-squares optimality\n");
  std::normal_distribution<double> N(0.0, 1.0);
  Eigen::MatrixXd A(5, 3);
  Eigen::VectorXd b(5);
  for (int i = 0; i < 5; ++i) { for (int j = 0; j < 3; ++j) A(i, j) = N(rng); b(i) = N(rng); }
  Eigen::MatrixXd S = 0.1 * Eigen::MatrixXd::Identity(3, 3);
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3);

  Eigen::Matrix<double, 3, 1> x = Eigen::Vector3d(1, 1, 1);
  LinearFactor lf(A, b);
  AnchorFactor af(S, x0);
  Problem problem;
  problem.AddParameterBlock(x.data(), 3);
  problem.AddResidualBlock(&lf, nullptr, {x.data()});
  problem.AddResidualBlock(&af, nullptr, {x.data()});
  SolverOptions opts;
  opts.num_threads = 1;
  SolverSummary s = problem.Solve(opts);

  // Closed form: (A^T A + S^T S) x = A^T b  (x0 = 0)
  Eigen::Matrix3d H = A.transpose() * A + S.transpose() * S;
  Eigen::Vector3d xstar = H.ldlt().solve(A.transpose() * b);
  check_true(s.converged, "linear solve converged");
  check_lt((x - xstar).norm(), 1e-6, "linear solve matches normal equations");
}

// Build a small BA problem in `data`. cams 0..M-1, landmarks M..M+K-1, all 3-dof.
struct BA {
  int M, K;
  std::vector<Eigen::Vector3d> init;
  std::vector<Eigen::Vector3d> obs_d; // per (i,j) pair
  std::vector<std::pair<int, int>> pairs;
  Eigen::MatrixXd S; // camera anchor sqrt-info
};
static BA make_ba(std::mt19937 &rng, int M, int K) {
  std::normal_distribution<double> N(0.0, 1.0);
  BA ba;
  ba.M = M;
  ba.K = K;
  ba.S = 0.3 * Eigen::MatrixXd::Identity(3, 3);
  for (int i = 0; i < M + K; ++i)
    ba.init.emplace_back(Eigen::Vector3d(N(rng), N(rng), N(rng)));
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < K; ++j) {
      ba.pairs.emplace_back(i, M + j);
      ba.obs_d.emplace_back(Eigen::Vector3d(N(rng), N(rng), N(rng)));
    }
  return ba;
}
// Solve the BA problem; returns the stacked solution. Factors live in the provided vectors.
static Eigen::VectorXd solve_ba(const BA &ba, bool use_schur, int num_threads, std::vector<Eigen::Vector3d> &data,
                                std::vector<DiffFactor> &diffs, std::vector<AnchorFactor> &anchors, SolverSummary *out = nullptr,
                                const SolverOptions *test_options = nullptr) {
  data = ba.init;
  diffs.clear();
  anchors.clear();
  diffs.reserve(ba.pairs.size());
  anchors.reserve(ba.M);
  for (size_t p = 0; p < ba.pairs.size(); ++p)
    diffs.emplace_back(ba.obs_d[p], 1.0);
  for (int i = 0; i < ba.M; ++i)
    anchors.emplace_back(ba.S, ba.init[i]); // anchor cams near their start (fixes the global gauge)

  Problem problem;
  for (int i = 0; i < ba.M + ba.K; ++i)
    problem.AddParameterBlock(data[i].data(), 3);
  if (use_schur)
    for (int j = 0; j < ba.K; ++j)
      problem.SetSchurLandmark(data[ba.M + j].data());
  for (size_t p = 0; p < ba.pairs.size(); ++p)
    problem.AddResidualBlock(&diffs[p], nullptr, {data[ba.pairs[p].first].data(), data[ba.pairs[p].second].data()});
  for (int i = 0; i < ba.M; ++i)
    problem.AddResidualBlock(&anchors[i], nullptr, {data[i].data()});

  // Schur-vs-dense is selected by whether landmarks are tagged via SetSchurLandmark (above),
  // not by an option: untagged landmark blocks are solved in the plain dense path.
  SolverOptions opts = test_options ? *test_options : SolverOptions();
  opts.num_threads = num_threads;
  SolverSummary s = problem.Solve(opts);
  if (out)
    *out = s;

  Eigen::VectorXd x(3 * (ba.M + ba.K));
  for (int i = 0; i < ba.M + ba.K; ++i)
    x.segment(3 * i, 3) = data[i];
  return x;
}

static void test_schur_vs_dense(std::mt19937 &rng) {
  std::printf("[test] Schur vs dense equivalence\n");
  BA ba = make_ba(rng, 4, 6);
  std::vector<Eigen::Vector3d> d1, d2;
  std::vector<DiffFactor> f1, f2;
  std::vector<AnchorFactor> a1, a2;
  SolverSummary s1, s2;
  Eigen::VectorXd x_dense = solve_ba(ba, false, 1, d1, f1, a1, &s1);
  Eigen::VectorXd x_schur = solve_ba(ba, true, 1, d2, f2, a2, &s2);
  check_true(s1.converged && s2.converged, "BA converged (dense & schur)");
  check_lt((x_dense - x_schur).norm(), 1e-9, "Schur solution == dense solution");
}

static void test_parallel_determinism(std::mt19937 &rng) {
  std::printf("[test] lock-free parallel determinism\n");
  BA ba = make_ba(rng, 5, 12);
  std::vector<Eigen::Vector3d> d;
  std::vector<DiffFactor> f;
  std::vector<AnchorFactor> a;

  // Compare completed numerical solves. A production wall deadline can end
  // different runs at different iterates under host scheduling contention.
  // Keep the same iteration cap and all convergence/comparison tolerances.
  SolverOptions complete;
  complete.max_solver_time_seconds = std::numeric_limits<double>::max();
  SolverSummary s1, s4a, s4b;
  Eigen::VectorXd x1 = solve_ba(ba, true, 1, d, f, a, &s1, &complete);
  Eigen::VectorXd x4a = solve_ba(ba, true, 4, d, f, a, &s4a, &complete);
  Eigen::VectorXd x4b = solve_ba(ba, true, 4, d, f, a, &s4b, &complete);

  check_true(s1.converged && !s1.time_stopped, "1-thread comparison solve completes");
  check_true(s4a.converged && !s4a.time_stopped, "first 4-thread comparison solve completes");
  check_true(s4b.converged && !s4b.time_stopped, "second 4-thread comparison solve completes");

  // Run-to-run with a fixed thread count must be bitwise identical (no races).
  check_true((x4a.array() == x4b.array()).all(), "4-thread run-to-run bitwise identical");
  // Across thread counts: identical up to floating-point summation grouping.
  check_lt((x1 - x4a).norm(), 1e-9, "1-thread vs 4-thread agree to round-off");
}

static void test_production_deadline() {
  std::printf("[test] production deadline remains active\n");
  class SlowLinearFactor : public LinearFactor {
  public:
    using LinearFactor::LinearFactor;
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
      // Deliberately expensive input evaluation, not a timing benchmark. The
      // unchanged default 50 ms deadline must stop before the first update.
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      return LinearFactor::Evaluate(parameters, residuals, jacobians);
    }
  };
  for (bool dogleg : {false, true}) {
    double value = 2.;
    SlowLinearFactor factor(Eigen::MatrixXd::Identity(1, 1), Eigen::VectorXd::Zero(1));
    Problem problem;
    problem.AddParameterBlock(&value, 1);
    problem.AddResidualBlock(&factor, nullptr, {&value});
    SolverOptions options;
    options.num_threads = 1;
    options.use_dogleg = dogleg;
    const auto result = problem.Solve(options);
    check_true(result.time_stopped && !result.converged, "default production deadline reports an incomplete solve");
    check_true(result.iterations == 0 && value == 2., "deadline before first update preserves the parameter");
  }
}

static void test_covariance(std::mt19937 &rng) {
  std::printf("[test] covariance (linear analytic + Schur-vs-dense marginal)\n");
  std::normal_distribution<double> N(0.0, 1.0);

  // (a) Linear: covariance == (A^T A + S^T S)^{-1}
  {
    Eigen::MatrixXd A(6, 3);
    Eigen::VectorXd b(6);
    for (int i = 0; i < 6; ++i) { for (int j = 0; j < 3; ++j) A(i, j) = N(rng); b(i) = N(rng); }
    Eigen::MatrixXd S = 0.5 * Eigen::MatrixXd::Identity(3, 3);
    Eigen::Vector3d x = Eigen::Vector3d::Zero();
    LinearFactor lf(A, b);
    AnchorFactor af(S, Eigen::Vector3d::Zero());
    Problem problem;
    problem.AddParameterBlock(x.data(), 3);
    problem.AddResidualBlock(&lf, nullptr, {x.data()});
    problem.AddResidualBlock(&af, nullptr, {x.data()});
    SolverOptions opts;
    opts.num_threads = 2;
    problem.Solve(opts);
    Eigen::MatrixXd cov;
    bool ok = problem.ComputeCovariance({x.data()}, cov, opts);
    Eigen::Matrix3d Hexp = A.transpose() * A + S.transpose() * S;
    Eigen::Matrix3d Cexp = Hexp.inverse();
    check_true(ok, "covariance computed (linear)");
    check_lt((cov - Cexp).norm(), 1e-7, "covariance == (A^T A + S^T S)^{-1}");
  }

  // (b) BA: Schur-marginalized nav covariance == dense full-inverse nav block.
  {
    BA ba = make_ba(rng, 3, 5);
    std::vector<Eigen::Vector3d> d;
    std::vector<DiffFactor> f;
    std::vector<AnchorFactor> a;
    solve_ba(ba, true, 1, d, f, a); // converge; leaves `d` at the solution

    // Rebuild two problems at the SAME converged data; one tags landmarks, one does not.
    auto build = [&](bool tag, Eigen::MatrixXd &cov) {
      Problem problem;
      for (int i = 0; i < ba.M + ba.K; ++i)
        problem.AddParameterBlock(d[i].data(), 3);
      if (tag)
        for (int j = 0; j < ba.K; ++j)
          problem.SetSchurLandmark(d[ba.M + j].data());
      for (size_t p = 0; p < ba.pairs.size(); ++p)
        problem.AddResidualBlock(&f[p], nullptr, {d[ba.pairs[p].first].data(), d[ba.pairs[p].second].data()});
      for (int i = 0; i < ba.M; ++i)
        problem.AddResidualBlock(&a[i], nullptr, {d[i].data()});
      std::vector<double *> cams;
      for (int i = 0; i < ba.M; ++i)
        cams.push_back(d[i].data());
      SolverOptions opts;
      return problem.ComputeCovariance(cams, cov, opts);
    };
    Eigen::MatrixXd cov_schur, cov_dense;
    bool ok1 = build(true, cov_schur);
    bool ok2 = build(false, cov_dense);
    check_true(ok1 && ok2, "covariance computed (BA, schur & dense)");
    // The Schur path regularizes landmark blocks with a tiny ridge; expect ~6-digit
    // agreement with the dense full-inverse nav block, not exact equality.
    check_lt((cov_schur - cov_dense).norm(), 1e-6, "Schur-marginal nav cov == dense full-inverse nav block");
  }
}

// Multiple ambient blocks, including repeated pointers and constant blocks.
class ExportLinearFactor : public CostFunction {
public:
  ExportLinearFactor(std::vector<Eigen::MatrixXd> J, Eigen::VectorXd r) : J_(std::move(J)), r_(std::move(r)) {
    set_num_residuals((int)r_.size());
    for (const auto &j : J_)
      mutable_parameter_block_sizes()->push_back((int)j.cols());
  }
  bool Evaluate(double const *const *p, double *r, double **j) const override {
    Eigen::Map<Eigen::VectorXd> out(r, r_.size());
    out = r_;
    for (size_t k = 0; k < J_.size(); ++k) {
      out.noalias() += J_[k] * Eigen::Map<const Eigen::VectorXd>(p[k], J_[k].cols());
      if (j && j[k])
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(j[k], J_[k].rows(), J_[k].cols()) = J_[k];
    }
    return true;
  }
private:
  std::vector<Eigen::MatrixXd> J_;
  Eigen::VectorXd r_;
};

class ZeroWeightLoss : public LossFunction {
public:
  void Evaluate(double, double out[2]) const override { out[0] = 7.0; out[1] = 0.0; }
};

class ExportInspectableProblem : public Problem {
public:
  bool projected(Eigen::MatrixXd &H, Eigen::VectorXd &g, landmark_qr::Evidence &e, int threads) {
    assign_ordering();
    ParallelExecutor exec(threads);
    return landmark_qr::assemble(blocks_, residuals_, land_block_idx_, land_adj_, n_nav_, exec, H, g, e);
  }
};

static void test_qr_exports(std::mt19937 &rng) {
  std::printf("[test] rank-aware export against independent full-row SVD\n");
  std::normal_distribution<double> normal;
  auto random = [&](int r, int c) {
    Eigen::MatrixXd M(r, c);
    for (int j = 0; j < c; ++j)
      for (int i = 0; i < r; ++i)
        M(i, j) = normal(rng);
    return M;
  };
  const char *qr_mode = std::getenv("OV_ZCALIB_EXPORT_QR");
  const bool production_qr = qr_mode != nullptr && std::strcmp(qr_mode, "1") == 0 &&
                             std::getenv("OV_ZCALIB_EXPORT_LEGACY") == nullptr &&
                             std::getenv("OV_ZCALIB_EXPORT_AUDIT") == nullptr;
  for (int obs : {1, 2, 3, 10, 40}) {
    for (int kind = 0; kind < 6; ++kind) {
      const int m = 2 * obs;
      Eigen::MatrixXd B = random(m, 3), A = random(m, 6);
      if (kind == 1) B.col(2).setZero();
      if (kind == 2) { B.col(1).setZero(); B.col(2).setZero(); }
      if (kind == 3) B *= 1e-9; // all-dust absolute floor
      if (kind == 5 && m >= 3) {
        // A cancellation-sensitive feature: large information in directions
        // eliminated from A, while the surviving information remains O(1).
        Eigen::HouseholderQR<Eigen::MatrixXd> q(B);
        B = q.householderQ() * Eigen::MatrixXd::Identity(m, 3);
        B.col(0) *= 4e5; B.col(1) *= 3.9e5; B.col(2) *= 58.;
        A.noalias() += B * random(3, 6);
      }
      const Eigen::VectorXd residual = random(m, 1);
      Eigen::MatrixXd Bw = B, Aw = A;
      Eigen::VectorXd rw = residual;
      double cost = 0.0;
      CauchyLoss cauchy(2.0);
      ZeroWeightLoss zero;
      ProjParam param;
      Eigen::Vector3d q = Eigen::Vector3d::Zero(), p = Eigen::Vector3d::Zero(), feature = Eigen::Vector3d::Zero();
      Eigen::Vector3d feature_only = Eigen::Vector3d::Zero();
      Eigen::Vector2d fixed(0.2, -0.3);
      double scalar = 0.0;
      ExportInspectableProblem problem;
      problem.AddParameterBlock(q.data(), 3, &param); // local 2, ambient 3
      problem.AddParameterBlock(p.data(), 3);
      problem.AddParameterBlock(&scalar, 1);
      problem.AddParameterBlock(feature.data(), 3);
      problem.SetSchurLandmark(feature.data());
      problem.AddParameterBlock(feature_only.data(), 3);
      problem.SetSchurLandmark(feature_only.data());
      problem.AddParameterBlock(fixed.data(), 2);
      problem.SetParameterBlockConstant(fixed.data());
      problem.SetSchurLandmark(fixed.data()); // a constant landmark stays outside elimination
      std::vector<ExportLinearFactor> factors;
      factors.reserve(obs);
      for (int ob = 0; ob < obs; ++ob) {
        const int row = 2 * ob;
        const double sigma = (ob % 2) ? 2.2 : 0.45;
        const bool zero_weight = kind == 4 || (ob == 1 && kind == 2);
        const LossFunction *loss = zero_weight ? static_cast<LossFunction *>(&zero) : &cauchy;
        Eigen::MatrixXd Jq = Eigen::MatrixXd::Zero(2, 3);
        Jq.leftCols(2) = A.block(row, 0, 2, 2) / sigma;
        const Eigen::MatrixXd Jp = A.block(row, 2, 2, 3) / sigma;
        const Eigen::MatrixXd Js = A.block(row, 5, 2, 1) / sigma;
        const Eigen::MatrixXd Jb = B.middleRows(row, 2) / sigma;
        const Eigen::MatrixXd Jfixed = random(2, 2);
        factors.emplace_back(std::vector<Eigen::MatrixXd>{Jq, 0.4 * Jp, Js, Jb, Jfixed, 0.6 * Jp},
                             residual.segment(row, 2) / sigma - Jfixed * fixed);
        problem.AddResidualBlock(&factors.back(), loss,
                                 {q.data(), p.data(), &scalar, feature.data(), fixed.data(), p.data()});
        double rho[2];
        loss->Evaluate(residual.segment(row, 2).squaredNorm() / (sigma * sigma), rho);
        const double scale = std::sqrt(std::max(0.0, rho[1])) / sigma;
        Bw.middleRows(row, 2) *= scale;
        Aw.middleRows(row, 2) *= scale;
        rw.segment(row, 2) *= scale;
        cost += 0.5 * rho[0];
      }
      // Feature-only residual: must affect cost/decrement/rank metadata without
      // inventing navigation columns or being combined with another feature.
      Eigen::Matrix3d only_B = Eigen::Matrix3d::Zero(); only_B(0, 0) = 3.0;
      const Eigen::Vector3d only_r(0.2, 0.4, 0.6);
      ExportLinearFactor only({only_B}, only_r);
      problem.AddResidualBlock(&only, nullptr, {feature_only.data()});
      cost += 0.5 * only_r.squaredNorm();
      // Independent navigation prior includes nuisance/calibration cross terms.
      const Eigen::MatrixXd prior = random(8, 6);
      Eigen::MatrixXd prior_q = Eigen::MatrixXd::Zero(8, 3);
      prior_q.leftCols(2) = prior.leftCols(2);
      ExportLinearFactor anchor({prior_q, prior.middleCols(2, 3), prior.rightCols(1)}, Eigen::VectorXd::Zero(8));
      problem.AddResidualBlock(&anchor, nullptr, {q.data(), p.data(), &scalar});

      // Independent oracle: full SVD of the original weighted m x 3 B, not
      // the implementation's QR followed by a 3 x 3 SVD.
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(Bw, Eigen::ComputeFullU);
      int rank = 0;
      if (svd.singularValues()(0) > 1e-6)
        for (int i = 0; i < svd.singularValues().size(); ++i)
          if (svd.singularValues()(i) > 1e-4 * svd.singularValues()(0)) ++rank;
      const Eigen::MatrixXd null = svd.matrixU().rightCols(m - rank);
      const Eigen::MatrixXd projected_A = null.transpose() * Aw;
      const Eigen::VectorXd projected_r = null.transpose() * rw;
      const Eigen::MatrixXd expected_H = projected_A.transpose() * projected_A + prior.transpose() * prior;
      const Eigen::VectorXd expected_g = projected_A.transpose() * projected_r;
      const double decrement = (svd.matrixU().leftCols(rank).transpose() * rw).squaredNorm() + 0.04;
      Eigen::MatrixXd H;
      Eigen::VectorXd g;
      landmark_qr::Evidence evidence;
      check_true(problem.projected(H, g, evidence, 2), "QR graph assembled");
      check_lt((H - expected_H).norm() / std::max(1.0, expected_H.norm()), 2e-9, "projected information agrees with SVD");
      check_lt((g - expected_g).norm() / std::max(1.0, expected_g.norm()), 2e-9, "projected gradient agrees with SVD");
      check_lt(std::abs(evidence.cost - cost), 1e-11, "robust residual-only objective constant retained");
      check_lt(std::abs(evidence.land_decrement - decrement), 1e-9, "landmark cost decrement agrees with SVD");
      check_true(evidence.clamped_dirs == (3 - rank) + 2, "rank and all-dust metadata preserved");
      for (int t = 0; t < 3; ++t) {
        const Eigen::VectorXd dx = random(6, 1);
        const double actual = evidence.cost - 0.5 * evidence.land_decrement + g.dot(dx) + 0.5 * dx.dot(H * dx);
        const double oracle = cost - 0.5 * decrement + expected_g.dot(dx) + 0.5 * dx.dot(expected_H * dx);
        check_lt(std::abs(actual - oracle) / std::max(1.0, std::abs(oracle)), 2e-9, "profiled robust quadratic objective agrees");
      }
      if (production_qr) {
        SolverOptions opts; opts.num_threads = 2;
        Eigen::MatrixXd Lambda, covariance;
        Eigen::VectorXd reduced_g;
        Problem::ExportStats stats;
        check_true(problem.ExportReducedInformation({&scalar, q.data()}, Lambda, reduced_g, opts, &stats), "production export computed (permuted, nontrailing keep)");
        const Eigen::MatrixXd full_cov = expected_H.inverse();
        const int keep[3] = {5, 0, 1};
        Eigen::Matrix3d cov_keep;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) cov_keep(a, b) = full_cov(keep[a], keep[b]);
        const Eigen::Matrix3d expected_L = cov_keep.inverse();
        const Eigen::VectorXd full_step = full_cov * expected_g;
        const Eigen::Vector3d expected_gr = expected_L * Eigen::Vector3d(full_step(5), full_step(0), full_step(1));
        check_lt((Lambda - expected_L).norm() / std::max(1.0, expected_L.norm()), 3e-9, "full nuisance-profile information agrees");
        check_lt((reduced_g - expected_gr).norm() / std::max(1.0, expected_gr.norm()), 3e-9, "full nuisance-profile gradient agrees");
        check_true(problem.ComputeCovariance({&scalar, q.data()}, covariance, opts), "production marginal covariance computed");
        check_lt((covariance - cov_keep).norm() / std::max(1.0, cov_keep.norm()), 3e-9, "marginal covariance agrees with oracle");
        const Eigen::Vector3d gn = expected_g.segment<3>(2);
        const double qn = decrement + gn.dot(expected_H.block<3, 3>(2, 2).ldlt().solve(gn));
        check_lt(std::abs(stats.nuis_decrement - qn) / std::max(1.0, qn), 3e-9, "complete nuisance decrement agrees");
        check_true(stats.clamped_dirs == evidence.clamped_dirs, "production rank evidence agrees");
        if (obs == 40 && kind == 5) {
          stats = Problem::ExportStats();
          stats.land_decrement = 1.25;
          stats.clamped_dirs = 2;
          check_true(problem.ExportReducedInformation({q.data(), p.data(), &scalar}, Lambda, reduced_g, opts, &stats),
                     "export with no navigation nuisance succeeds");
          check_lt((Lambda - expected_H).norm() / expected_H.norm(), 3e-9, "no-nuisance information agrees");
          check_lt(std::abs(stats.nuis_decrement - 1.25 - decrement), 1e-9, "additive decrement evidence preserved");
          check_true(stats.clamped_dirs == evidence.clamped_dirs + 2, "additive rank evidence preserved");
        }
      }
    }
  }

  Eigen::Vector3d nav = Eigen::Vector3d::Zero(), f1 = nav, f2 = nav;
  ExportInspectableProblem invalid;
  for (double *p : {nav.data(), f1.data(), f2.data()}) invalid.AddParameterBlock(p, 3);
  invalid.SetSchurLandmark(f1.data()); invalid.SetSchurLandmark(f2.data());
  DiffFactor coupled(Eigen::Vector3d::Zero(), 1.0);
  invalid.AddResidualBlock(&coupled, nullptr, {f1.data(), f2.data()});
  Eigen::MatrixXd H; Eigen::VectorXd g; landmark_qr::Evidence evidence;
  check_true(!invalid.projected(H, g, evidence, 1), "coupled variable landmarks fail explicitly");
  if (production_qr) {
    SolverOptions opts;
    check_true(!invalid.ExportReducedInformation({nav.data()}, H, g, opts), "production export rejects coupled landmarks");
  }

  for (double ratio : {0.999999e-4, 1.000001e-4}) {
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(4, 3), Ar = random(4, 3);
    B(0, 0) = 1.; B(1, 1) = 0.01; B(2, 2) = ratio;
    int rank = -1; double decrement = -1.;
    check_true(landmark_qr::project(B, Ar, rank, decrement), "near-cutoff QR projection succeeds");
    check_true(rank == (ratio < 1e-4 ? 2 : 3), "relative information rank cutoff preserved");
  }
}

// Inject invalid values after arithmetic so fast-math cannot erase their role
// in the factor implementation. The export must reject them before SVD/LDLT.
class InvalidExportFactor : public CostFunction {
public:
  InvalidExportFactor(int site, double bad) : site_(site), bad_(bad) {
    set_num_residuals(4);
    *mutable_parameter_block_sizes() = {1, 3};
  }
  bool Evaluate(double const *const *, double *r, double **j) const override {
    Eigen::Map<Eigen::Vector4d>(r).setConstant(0.25);
    if (j && j[0]) Eigen::Map<Eigen::Vector4d>(j[0]).setOnes();
    if (j && j[1]) {
      Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> B(j[1]);
      B.setZero(); B.topRows(3).setIdentity();
    }
    if (site_ == 0) r[0] = bad_;
    if (site_ == 1 && j && j[0]) j[0][0] = bad_;
    if (site_ == 2 && j && j[1]) j[1][0] = bad_;
    if (site_ == 3) r[0] = 1e200; // finite residual, overflowing squared norm
    if (site_ == 4 && j && j[0]) j[0][3] = 1e200; // overflowing projected information
    if (site_ == 5 && j && j[1]) { j[1][0] = 1e308; j[1][3] = 1e308; }
    return true;
  }
private:
  int site_;
  double bad_;
};

class InvalidExportLoss : public LossFunction {
public:
  InvalidExportLoss(bool derivative, double bad) : derivative_(derivative), bad_(bad) {}
  void Evaluate(double s, double out[2]) const override {
    out[0] = derivative_ ? s : bad_;
    out[1] = derivative_ ? bad_ : 1.0;
  }
private:
  bool derivative_;
  double bad_;
};

class InvalidExportParam : public LocalParameterization {
public:
  explicit InvalidExportParam(double bad) : bad_(bad) {}
  bool Plus(const double *x, const double *d, double *out) const override { out[0] = x[0] + d[0]; return true; }
  bool ComputeJacobian(const double *, double *out) const override { out[0] = bad_; return true; }
  int GlobalSize() const override { return 1; }
  int LocalSize() const override { return 1; }
private:
  double bad_;
};

static void test_qr_invalid_values() {
  std::printf("[test] QR finite checks survive -ffast-math\n");
  const char *mode = std::getenv("OV_ZCALIB_EXPORT_QR");
  const bool production_qr = mode && std::strcmp(mode, "1") == 0 &&
                             !std::getenv("OV_ZCALIB_EXPORT_LEGACY") && !std::getenv("OV_ZCALIB_EXPORT_AUDIT");
  for (std::uint64_t bits : {UINT64_C(0x7ff8000000000001), UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000)}) {
    double bad;
    std::memcpy(&bad, &bits, sizeof(bad));
    check_true(!landmark_qr::finite_scalar(bad), "bitwise finite predicate rejects NaN/Inf");
    for (int site = 0; site < 9; ++site) {
      double nav = 0.0;
      Eigen::Vector3d feature = Eigen::Vector3d::Zero();
      InvalidExportFactor factor(site, bad);
      InvalidExportLoss loss(site == 7, bad);
      InvalidExportParam param(bad);
      ExportInspectableProblem problem;
      problem.AddParameterBlock(&nav, 1, site == 8 ? &param : nullptr);
      problem.AddParameterBlock(feature.data(), 3);
      problem.SetSchurLandmark(feature.data());
      problem.AddResidualBlock(&factor, (site == 6 || site == 7) ? &loss : nullptr, {&nav, feature.data()});
      Eigen::MatrixXd H; Eigen::VectorXd g; landmark_qr::Evidence evidence;
      check_true(!problem.projected(H, g, evidence, 1), "invalid residual/J/loss/basis/overflow fails closed");
      if (production_qr) {
        SolverOptions opts;
        check_true(!problem.ExportReducedInformation({&nav}, H, g, opts), "invalid production information export rejected");
        check_true(!problem.ComputeCovariance({&nav}, H, opts), "invalid production covariance export rejected");
      }
    }
  }
}

int main() {
  std::printf("==== ov_init::zbft_sfm ceres-free solver core tests ====\n");
  std::mt19937 rng(42);
  test_fd_jacobians(rng);
  test_linear_solve(rng);
  test_schur_vs_dense(rng);
  test_parallel_determinism(rng);
  test_production_deadline();
  test_covariance(rng);
  test_qr_exports(rng);
  test_qr_invalid_values();
  std::printf("==== %d checks, %d failures ====\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
