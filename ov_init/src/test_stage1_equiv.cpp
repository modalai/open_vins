/*
 * Stage-1 equivalence gate (host-only, no OpenCV / no Ceres / no ROS).
 * Contributor: Joao Leonardo Silva Cotta (@zauberflote1)
 *
 * Validates the two behavior-identical Stage-1 rewrites in DynamicInitializer::initialize:
 *
 *  (1) CPI single-sweep + composition: the I0->Ik preintegration VALUES (DT, R, alpha, beta)
 *      composed from consecutive-interval CPIs must match
 *        (a) a running CPI fed the same interval-split readings  -> exact (associativity),
 *        (b) the OLD direct I0->Ik build (one un-split sweep)    -> equal up to the boundary
 *            interpolation splitting effect, O(dt_sample^3) per interior boundary. The new
 *            path is the one consistent with the MLE's interval factors.
 *
 *  (2) Projector-free arrowhead Dong-Si: on randomized observation structures, the
 *      normal-equation reduction (per-feature 3x3 elimination + 3x3 velocity Schur) must
 *      reproduce the OLD dense-projector D, d, the chosen lambda, and the recovered
 *      [features, velocity, gravity] state.
 *
 * Build (OpenCV headers needed transitively by ov_core sensor_data.h; nothing is linked):
 *   OVC=../../ov_core/src
 *   g++ -O2 -std=c++17 -I/usr/include/eigen3 -I/usr/include/opencv4 -I. -I$OVC \
 *       test_stage1_equiv.cpp $OVC/cpi/CpiV1.cpp -o /tmp/test_stage1_equiv && /tmp/test_stage1_equiv
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "cpi/CpiV1.h"
#include "utils/sensor_data.h"

#include "utils/helper.h"

static int n_checks = 0, n_fails = 0;
static void check(const char *name, double val, double tol) {
  n_checks++;
  bool ok = std::isfinite(val) && val <= tol;
  if (!ok)
    n_fails++;
  std::printf("  [%s] %-55s value=%.3e  tol=%.1e\n", ok ? " ok " : "FAIL", name, val, tol);
}

// ---------------------------------------------------------------------------------------------
// (1) CPI composition equivalence
// ---------------------------------------------------------------------------------------------
static void test_cpi_composition() {
  std::printf("[test] CPI single-sweep composition vs direct I0->Ik build\n");

  // Synthetic IMU at 500 Hz over 2.0 s with varied rotation (biases folded in)
  const double hz = 500.0, T = 2.0;
  const Eigen::Vector3d bg(0.01, -0.02, 0.005), ba(0.05, 0.02, -0.03);
  std::vector<ov_core::ImuData> imu;
  for (int k = 0; k <= (int)(T * hz) + 2; k++) {
    double t = k / hz;
    ov_core::ImuData m;
    m.timestamp = t;
    m.wm = Eigen::Vector3d(1.2 * std::sin(2.1 * t), 0.9 * std::sin(1.3 * t + 1.0), 1.1 * std::sin(2.9 * t + 2.0)) + bg;
    m.am = Eigen::Vector3d(0.5 * std::sin(0.8 * t), 0.3 * std::cos(1.2 * t), 9.81 + 0.2 * std::sin(0.5 * t)) + ba;
    imu.push_back(m);
  }

  // Camera times (K poses, NOT aligned to IMU samples so boundary interpolation is exercised)
  const int K = 9;
  std::vector<double> cam_times;
  for (int i = 0; i < K; i++)
    cam_times.push_back(0.0713 + i * (T - 0.15) / (K - 1));

  auto make_cpi = [&]() {
    auto c = std::make_shared<ov_core::CpiV1>(1e-3, 1e-4, 1e-2, 1e-3, true);
    c->setLinearizationPoints(bg, ba);
    return c;
  };
  auto feed = [&](std::shared_ptr<ov_core::CpiV1> &c, double t0, double t1) {
    auto r = ov_init::InitializerHelper::select_imu_readings(imu, t0, t1);
    for (size_t k = 0; k + 1 < r.size(); k++)
      c->feed_IMU(r[k].timestamp, r[k + 1].timestamp, r[k].wm, r[k].am, r[k + 1].wm, r[k + 1].am);
  };

  // (a) split-fed running CPI (associativity reference), (b) old direct build, (c) composition
  double max_a_alpha = 0, max_a_beta = 0, max_a_R = 0, max_a_DT = 0;
  double max_b_alpha = 0, max_b_beta = 0, max_b_R = 0;

  auto running = make_cpi(); // fed interval-by-interval (same readings the intervals use)
  double comp_DT = 0.0;
  Eigen::Matrix3d comp_R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d comp_alpha = Eigen::Vector3d::Zero(), comp_beta = Eigen::Vector3d::Zero();

  for (int i = 1; i < K; i++) {
    // interval CPI (what the new code builds)
    auto ci = make_cpi();
    feed(ci, cam_times[i - 1], cam_times[i]);
    // running CPI gets the SAME split readings
    feed(running, cam_times[i - 1], cam_times[i]);
    // compose (the new code's update)
    comp_alpha += comp_beta * ci->DT + comp_R.transpose() * ci->alpha_tau;
    comp_beta += comp_R.transpose() * ci->beta_tau;
    comp_R = ci->R_k2tau * comp_R;
    comp_DT += ci->DT;
    // old direct build (un-split single sweep from t0)
    auto direct = make_cpi();
    feed(direct, cam_times[0], cam_times[i]);

    max_a_alpha = std::max(max_a_alpha, (comp_alpha - running->alpha_tau).norm());
    max_a_beta = std::max(max_a_beta, (comp_beta - running->beta_tau).norm());
    max_a_R = std::max(max_a_R, (comp_R - running->R_k2tau).norm());
    max_a_DT = std::max(max_a_DT, std::abs(comp_DT - running->DT));
    max_b_alpha = std::max(max_b_alpha, (comp_alpha - direct->alpha_tau).norm());
    max_b_beta = std::max(max_b_beta, (comp_beta - direct->beta_tau).norm());
    max_b_R = std::max(max_b_R, (comp_R - direct->R_k2tau).norm());
  }

  // (a) exact associativity: fp-roundoff only
  check("compose == split-fed running CPI: alpha", max_a_alpha, 1e-10);
  check("compose == split-fed running CPI: beta", max_a_beta, 1e-10);
  check("compose == split-fed running CPI: R", max_a_R, 1e-10);
  check("compose == split-fed running CPI: DT", max_a_DT, 1e-10);
  // (b) vs old un-split direct build: boundary-splitting effect only
  check("compose vs OLD direct build: alpha", max_b_alpha, 1e-6);
  check("compose vs OLD direct build: beta", max_b_beta, 1e-6);
  check("compose vs OLD direct build: R", max_b_R, 1e-6);
}

// ---------------------------------------------------------------------------------------------
// (2) Dong-Si constrained solve: old dense projector vs new arrowhead normal equations
// ---------------------------------------------------------------------------------------------
struct Obs {
  int feat;
  Eigen::Matrix<double, 2, 3> Y;
  Eigen::Vector2d b;
  double DT;
};

static void run_dongsi_case(std::mt19937 &rng, int F, int K, bool consistent) {
  std::normal_distribution<double> N(0, 1);
  std::uniform_real_distribution<double> U(0.05, 2.0);
  const double gravity_mag = 9.81;

  // Ground truth for the measurement-consistent mode: the end-to-end (own-lambda) checks are
  // only meaningful when the |g|=G root is well separated, i.e. on a physically consistent
  // instance (b = A*x_gt + small noise). On a fully random b, several polynomial roots can tie
  // on the constraint cost and 1e-14 Gram noise flips the argmin -- exactly as it already did
  // between two runs of the OLD code (unordered_map summation order).
  std::vector<Eigen::Vector3d> gt_f(F);
  for (int f = 0; f < F; f++)
    gt_f[f] = Eigen::Vector3d(2.0 * N(rng), 2.0 * N(rng), 5.0 + 2.0 * std::abs(N(rng)));
  Eigen::Vector3d gt_v(0.5 * N(rng), 0.5 * N(rng), 0.5 * N(rng));
  Eigen::Vector3d gt_g = Eigen::Vector3d(0.3 * N(rng), 0.3 * N(rng), -1.0).normalized() * gravity_mag;

  // Random per-pose DT and per-(feature,pose) observations with generic Y
  std::vector<double> DTs(K);
  for (int k = 0; k < K; k++)
    DTs[k] = (k == 0) ? 0.0 : U(rng) + DTs[k - 1];
  std::vector<Obs> obs;
  for (int f = 0; f < F; f++) {
    for (int k = 0; k < K; k++) {
      Obs o;
      o.feat = f;
      for (int r = 0; r < 2; r++)
        for (int c = 0; c < 3; c++)
          o.Y(r, c) = N(rng);
      o.DT = DTs[k];
      if (consistent) {
        const double c_v = -o.DT, c_g = 0.5 * o.DT * o.DT;
        o.b = o.Y * (gt_f[f] + c_v * gt_v + c_g * gt_g) + 1e-3 * Eigen::Vector2d(N(rng), N(rng));
      } else {
        o.b = Eigen::Vector2d(N(rng), N(rng));
      }
      obs.push_back(o);
    }
  }
  const int sysz = 3 * F + 6;
  const int M = 2 * (int)obs.size();

  // ---- OLD path: dense A/b, Gram inverse, measurement-sized projector ----
  const auto t_old0 = std::chrono::steady_clock::now();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(M, sysz);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(M);
  int im = 0;
  for (auto const &o : obs) {
    A.block(im, 3 * o.feat, 2, 3) = o.Y;
    A.block(im, 3 * F + 0, 2, 3) = -o.DT * o.Y;
    A.block(im, 3 * F + 3, 2, 3) = 0.5 * o.DT * o.DT * o.Y;
    b.segment(im, 2) = o.b;
    im += 2;
  }
  Eigen::MatrixXd A1 = A.leftCols(sysz - 3);
  Eigen::MatrixXd A1A1_inv = (A1.transpose() * A1).llt().solve(Eigen::MatrixXd::Identity(sysz - 3, sysz - 3));
  Eigen::MatrixXd A2 = A.rightCols(3);
  Eigen::MatrixXd Temp = A2.transpose() * (Eigen::MatrixXd::Identity(M, M) - A1 * A1A1_inv * A1.transpose());
  Eigen::MatrixXd D_old = Temp * A2;
  Eigen::MatrixXd d_old = Temp * b;
  const double ms_old = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_old0).count();

  // ---- NEW path: per-observation normal-equation blocks + arrowhead reduction ----
  const auto t_new0 = std::chrono::steady_clock::now();
  std::vector<Eigen::Matrix3d> NE_F(F, Eigen::Matrix3d::Zero()), NE_C(F, Eigen::Matrix3d::Zero()), NE_G(F, Eigen::Matrix3d::Zero());
  std::vector<Eigen::Vector3d> NE_yf(F, Eigen::Vector3d::Zero());
  Eigen::Matrix3d NE_V = Eigen::Matrix3d::Zero(), NE_W = Eigen::Matrix3d::Zero(), NE_GG = Eigen::Matrix3d::Zero();
  Eigen::Vector3d NE_yv = Eigen::Vector3d::Zero(), NE_yg = Eigen::Vector3d::Zero();
  for (auto const &o : obs) {
    const Eigen::Matrix3d S = o.Y.transpose() * o.Y;
    const Eigen::Vector3d t = o.Y.transpose() * o.b;
    const double c_v = -o.DT, c_g = 0.5 * o.DT * o.DT;
    NE_F[o.feat] += S;
    NE_C[o.feat] += c_v * S;
    NE_G[o.feat] += c_g * S;
    NE_yf[o.feat] += t;
    NE_V += (c_v * c_v) * S;
    NE_W += (c_v * c_g) * S;
    NE_GG += (c_g * c_g) * S;
    NE_yv += c_v * t;
    NE_yg += c_g * t;
  }
  Eigen::Matrix3d Vs = NE_V, Rvg = NE_W;
  Eigen::Vector3d rv = NE_yv;
  std::vector<Eigen::Matrix3d> Finv(F);
  for (int j = 0; j < F; j++) {
    Finv[j] = NE_F[j].inverse();
    const Eigen::Matrix3d CtFinv = NE_C[j].transpose() * Finv[j];
    Vs.noalias() -= CtFinv * NE_C[j];
    Rvg.noalias() -= CtFinv * NE_G[j];
    rv.noalias() -= CtFinv * NE_yf[j];
  }
  const Eigen::LLT<Eigen::Matrix3d> Vs_llt(Vs);
  const Eigen::Matrix3d Xvg = Vs_llt.solve(Rvg);
  const Eigen::Vector3d xv = Vs_llt.solve(rv);
  std::vector<Eigen::Matrix3d> Xfg(F);
  std::vector<Eigen::Vector3d> xf(F);
  Eigen::MatrixXd D_new = NE_GG, d_new = NE_yg;
  D_new.noalias() -= NE_W.transpose() * Xvg;
  d_new.noalias() -= NE_W.transpose() * xv;
  for (int j = 0; j < F; j++) {
    Xfg[j] = Finv[j] * (NE_G[j] - NE_C[j] * Xvg);
    xf[j] = Finv[j] * (NE_yf[j] - NE_C[j] * xv);
    D_new.noalias() -= NE_G[j].transpose() * Xfg[j];
    d_new.noalias() -= NE_G[j].transpose() * xf[j];
  }
  const double ms_new = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_new0).count();
  std::printf("  [time] F=%d K=%d: assembly+reduction OLD %.3f ms -> NEW %.3f ms (%.0fx)\n", F, K, ms_old, ms_new,
              ms_old / std::max(1e-9, ms_new));

  // ---- Shared lambda machinery (as in DynamicInitializer) on each D/d, then recovery ----
  // NOTE on the comparison: the polynomial-root selection amplifies 1e-14 input differences
  // (Wilkinson root conditioning), and the OLD path itself had run-to-run 1e-14 Gram noise
  // from the unordered_map summation order -- so raw g-vs-g equality was never well-defined.
  // The meaningful equivalence checks are (i) identical recovery at the SAME lambda (isolates
  // the algebra), and (ii) identical end-to-end least-squares cost at each path's own lambda.
  auto solve_constrained = [&](const Eigen::MatrixXd &D, const Eigen::MatrixXd &d, double &lam_out) -> Eigen::Vector3d {
    Eigen::MatrixXd Dm = D;
    Eigen::Matrix<double, 7, 1> coeff = ov_init::InitializerHelper::compute_dongsi_coeff(Dm, d, gravity_mag);
    Eigen::Matrix<double, 6, 6> cm = Eigen::Matrix<double, 6, 6>::Zero();
    cm.diagonal(-1).setOnes();
    cm.col(5) = -coeff.reverse().head(6);
    Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> solver(cm, false);
    bool found = false;
    double lam = -1, cost_min = INFINITY;
    Eigen::MatrixXd I3 = Eigen::MatrixXd::Identity(3, 3);
    for (int i = 0; i < solver.eigenvalues().size(); i++) {
      if (solver.eigenvalues()(i).imag() != 0)
        continue;
      double l = solver.eigenvalues()(i).real();
      Eigen::VectorXd g = (D - l * I3).llt().solve(I3) * d;
      double cost = std::abs(g.norm() - gravity_mag);
      if (!found || cost < cost_min) {
        found = true;
        lam = l;
        cost_min = cost;
      }
    }
    lam_out = lam;
    Eigen::VectorXd g = (D - lam * I3).llt().solve(I3) * d;
    return g;
  };
  double lam_old = 0.0, lam_new = 0.0;
  const Eigen::Vector3d g_old_v = solve_constrained(D_old, d_old, lam_old);
  const Eigen::Vector3d g_new_v = solve_constrained(D_new, d_new, lam_new);

  // Recovery helpers for each path
  auto recover_old = [&](const Eigen::Vector3d &g) -> Eigen::VectorXd {
    return -A1A1_inv * A1.transpose() * A2 * g + A1A1_inv * A1.transpose() * b;
  };
  auto recover_new = [&](const Eigen::Vector3d &g) -> Eigen::VectorXd {
    Eigen::VectorXd x1(3 * F + 3);
    for (int j = 0; j < F; j++)
      x1.segment(3 * j, 3) = xf[j] - Xfg[j] * g;
    x1.segment(3 * F, 3) = xv - Xvg * g;
    return x1;
  };
  // End-to-end objective on the ground-truth dense system
  auto ls_cost = [&](const Eigen::VectorXd &x1, const Eigen::Vector3d &g) -> double {
    Eigen::VectorXd x(sysz);
    x.head(3 * F + 3) = x1;
    x.tail(3) = g;
    return (A * x - b).squaredNorm();
  };

  char name[128];
  std::snprintf(name, sizeof(name), "F=%d K=%d: |D_new-D_old| (rel)", F, K);
  check(name, (D_new - D_old).norm() / std::max(1.0, D_old.norm()), 1e-9);
  std::snprintf(name, sizeof(name), "F=%d K=%d: |d_new-d_old| (rel)", F, K);
  check(name, (d_new - d_old).norm() / std::max(1.0, d_old.norm()), 1e-9);
  // (i) same-lambda algebra equivalence: both reductions and both recoveries at lam_old
  {
    Eigen::MatrixXd I3 = Eigen::MatrixXd::Identity(3, 3);
    Eigen::Vector3d g_o = (D_old - lam_old * I3).llt().solve(I3) * d_old;
    Eigen::Vector3d g_n = (D_new - lam_old * I3).llt().solve(I3) * d_new;
    std::snprintf(name, sizeof(name), "F=%d K=%d: same-lambda |g_new-g_old|", F, K);
    check(name, (g_n - g_o).norm() / std::max(1.0, g_o.norm()), 1e-8);
    std::snprintf(name, sizeof(name), "F=%d K=%d: same-g |x1_new-x1_old| (rel)", F, K);
    check(name, (recover_new(g_o) - recover_old(g_o)).norm() / std::max(1.0, recover_old(g_o).norm()), 1e-9);
  }
  // (ii) solution-quality equivalence at each path's own lambda (consistent instances only;
  //      on random-b instances the root argmin is not well defined -- see note above)
  if (consistent) {
    const double cost_old = ls_cost(recover_old(g_old_v), g_old_v);
    const double cost_new = ls_cost(recover_new(g_new_v), g_new_v);
    std::snprintf(name, sizeof(name), "F=%d K=%d: LS-cost rel diff", F, K);
    check(name, std::abs(cost_new - cost_old) / std::max(1e-12, cost_old), 1e-8);
    std::snprintf(name, sizeof(name), "F=%d K=%d: |g|-constraint diff", F, K);
    check(name, std::abs(std::abs(g_new_v.norm() - gravity_mag) - std::abs(g_old_v.norm() - gravity_mag)), 1e-6);
    std::snprintf(name, sizeof(name), "F=%d K=%d: gravity vs GT (sanity, deg)", F, K);
    double ang = std::acos(std::min(1.0, std::abs(g_new_v.normalized().dot(gt_g.normalized())))) * 180.0 / M_PI;
    check(name, ang, 1.0);
  }
}

static void test_dongsi_equiv() {
  std::printf("[test] arrowhead normal-equation Dong-Si vs dense projector\n");
  std::mt19937 rng(42);
  run_dongsi_case(rng, 8, 5, true);    // minimal-ish, consistent
  run_dongsi_case(rng, 50, 8, true);   // fpv-shaped, consistent
  run_dongsi_case(rng, 75, 13, true);  // the on-target logged shape, consistent
  run_dongsi_case(rng, 50, 8, false);  // adversarial random-b: algebra checks only
}

int main() {
  test_cpi_composition();
  test_dongsi_equiv();
  std::printf("==== %d checks, %d failures ====\n", n_checks, n_fails);
  return (n_fails == 0) ? 0 : 1;
}
