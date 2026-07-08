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
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "LocalParameterization.h"
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
                                std::vector<DiffFactor> &diffs, std::vector<AnchorFactor> &anchors, SolverSummary *out = nullptr) {
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
  SolverOptions opts;
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

  Eigen::VectorXd x1 = solve_ba(ba, true, 1, d, f, a);
  Eigen::VectorXd x4a = solve_ba(ba, true, 4, d, f, a);
  Eigen::VectorXd x4b = solve_ba(ba, true, 4, d, f, a);

  // Run-to-run with a fixed thread count must be bitwise identical (no races).
  check_true((x4a.array() == x4b.array()).all(), "4-thread run-to-run bitwise identical");
  // Across thread counts: identical up to floating-point summation grouping.
  check_lt((x1 - x4a).norm(), 1e-9, "1-thread vs 4-thread agree to round-off");
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

int main() {
  std::printf("==== ov_init::zbft_sfm ceres-free solver core tests ====\n");
  std::mt19937 rng(42);
  test_fd_jacobians(rng);
  test_linear_solve(rng);
  test_schur_vs_dense(rng);
  test_parallel_determinism(rng);
  test_covariance(rng);
  std::printf("==== %d checks, %d failures ====\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
