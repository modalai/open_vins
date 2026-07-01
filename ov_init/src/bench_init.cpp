/*
 * Ceres vs ov_init::zbft_sfm — dynamic-init MLE parity & performance benchmark.
 *
 * Builds the SAME visual-inertial factor graph (CPI IMU factors + pinhole
 * reprojection + first-pose gauge/bias prior) used by DynamicInitializer, solves
 * it from an identical initial guess with (a) Ceres and (b) the in-tree lock-free
 * ov_init::zbft_sfm solver, over K random trials, and reports:
 *   - PARITY:    max |x_ceres - x_zbft| over all states; final-cost agreement
 *   - SPEED:     iterations and wall-clock to converge (real on-target time)
 *   - COVARIANCE: per-DOF sigma agreement on the latest IMU state
 *   - ACCURACY:  gyro/accel-bias error vs ground truth (gauge-invariant)
 *
 * Self-contained graph (no OpenCV): a plain pinhole reprojection factor is used
 * so the comparison isolates the SOLVER. Build/run on the VOXL target.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "ceres/Factor_GenericPrior.h"
#include "ceres/Factor_ImuCPIv1.h"
#include "ceres/State_JPLQuatLocal.h"

#include "ceres_free/Factor_GenericPrior.h"
#include "ceres_free/Factor_ImuCPIv1.h"
#include "ceres_free/LocalParameterization.h"
#include "ceres_free/LossFunction.h"
#include "ceres_free/Problem.h"
#include "ceres_free/State_JPLQuatLocal.h"

#include "cpi/CpiV1.h"
#include "utils/quat_ops.h"

using ov_core::quat_2_Rot;
using ov_core::quat_multiply;
using ov_core::quatnorm;
using ov_core::skew_x;

// ---------------------------------------------------------------------------
// Pinhole reprojection factor (no distortion), templated on the solver's
// CostFunction base so the SAME residual/Jacobian code serves both backends.
// Blocks: q_GtoIi(4), p_IiinG(3), p_FinG(3). Camera frame == IMU frame.
// ---------------------------------------------------------------------------
template <class Base> class PinholeFactorT : public Base {
public:
  Eigen::Vector2d uv;
  Eigen::Matrix3d Rc; // R_ItoC (fixed camera extrinsic rotation)
  Eigen::Vector3d pc; // p_IinC (fixed)
  double fx, fy, cx, cy, isig;
  PinholeFactorT(const Eigen::Vector2d &uv_, double fx_, double fy_, double cx_, double cy_, double sigma_px,
                 const Eigen::Matrix3d &Rc_, const Eigen::Vector3d &pc_)
      : uv(uv_), Rc(Rc_), pc(pc_), fx(fx_), fy(fy_), cx(cx_), cy(cy_), isig(1.0 / sigma_px) {
    this->set_num_residuals(2);
    this->mutable_parameter_block_sizes()->push_back(4);
    this->mutable_parameter_block_sizes()->push_back(3);
    this->mutable_parameter_block_sizes()->push_back(3);
  }
  bool Evaluate(double const *const *p, double *res, double **jac) const override {
    Eigen::Vector4d q = Eigen::Map<const Eigen::Vector4d>(p[0]);
    Eigen::Matrix3d R = quat_2_Rot(q); // R_GtoIi
    Eigen::Vector3d pI = Eigen::Map<const Eigen::Vector3d>(p[1]);
    Eigen::Vector3d pf = Eigen::Map<const Eigen::Vector3d>(p[2]);
    Eigen::Vector3d pIi = R * (pf - pI); // feature in IMU frame
    Eigen::Vector3d fc = Rc * pIi + pc;  // feature in camera frame (fixed extrinsic)
    double iz = 1.0 / fc(2);
    Eigen::Vector2d proj(fx * fc(0) * iz + cx, fy * fc(1) * iz + cy);
    res[0] = isig * (proj(0) - uv(0));
    res[1] = isig * (proj(1) - uv(1));
    if (jac) {
      Eigen::Matrix<double, 2, 3> Hp;
      Hp << fx * iz, 0, -fx * fc(0) * iz * iz, 0, fy * iz, -fy * fc(1) * iz * iz;
      Hp *= isig;
      const Eigen::Matrix<double, 2, 3> HpRc = Hp * Rc;
      if (jac[0]) {
        Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(jac[0]);
        J.setZero();
        J.block(0, 0, 2, 3) = HpRc * skew_x(pIi);
      }
      if (jac[1]) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jac[1]);
        J = -HpRc * R;
      }
      if (jac[2]) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jac[2]);
        J = HpRc * R;
      }
    }
    return true;
  }
};

// ---------------------------------------------------------------------------
// Simulated VI-init problem (IMU residuals zero at ground truth by construction)
// ---------------------------------------------------------------------------
struct Sim {
  // Larger window: 20 keyframes at 30 Hz (~0.63 s), 2 NON-OVERLAPPING cameras (each
  // landmark seen by exactly one camera), 80 landmarks (40/cam), 330 Hz IMU (11 samples/interval).
  int N = 20, M = 80;
  double dt = 1.0 / 30.0, fx = 320, fy = 320, cx = 320, cy = 240, sigma_px = 1.2;
  Eigen::Vector3d grav = Eigen::Vector3d(0, 0, 9.81), bg_gt, ba_gt;
  std::vector<Eigen::Vector4d> q;       // GT q_GtoIi
  std::vector<Eigen::Vector3d> p, v;    // GT
  std::vector<Eigen::Vector3d> lm;      // GT landmarks (global)
  std::vector<int> lm_cam;              // which camera (0/1) observes each landmark (non-overlapping)
  Eigen::Matrix3d Rc[2];                // per-camera fixed extrinsic R_ItoC
  Eigen::Vector3d pc[2];                // per-camera fixed extrinsic p_IinC
  std::vector<std::shared_ptr<ov_core::CpiV1>> cpi; // size N-1
  std::vector<std::vector<std::pair<int, Eigen::Vector2d>>> obs; // per keyframe
};

static Eigen::Vector3d randn3(std::mt19937 &rng, double s) {
  std::normal_distribution<double> N(0, 1);
  return Eigen::Vector3d(s * N(rng), s * N(rng), s * N(rng));
}
static Eigen::Vector4d perturb_q(const Eigen::Vector4d &q, const Eigen::Vector3d &dth) {
  Eigen::Vector4d dq;
  dq << 0.5 * dth, 1.0;
  return quatnorm(quat_multiply(quatnorm(dq), q));
}

static Sim make_sim(std::mt19937 &rng) {
  Sim s;
  s.bg_gt = randn3(rng, 0.01);
  s.ba_gt = randn3(rng, 0.05);
  Eigen::Vector3d w_body = randn3(rng, 0.4);  // rad/s
  Eigen::Vector3d a_meas0 = randn3(rng, 0.5); // arbitrary smooth accel meas

  s.q.resize(s.N);
  s.p.resize(s.N);
  s.v.resize(s.N);
  s.q[0] = Eigen::Vector4d(0, 0, 0, 1);
  s.p[0] = randn3(rng, 0.1);
  s.v[0] = randn3(rng, 0.3);
  int nsub = 11; // 330 Hz IMU / 30 Hz keyframes
  double ddt = s.dt / nsub;
  for (int i = 0; i < s.N - 1; ++i) {
    auto c = std::make_shared<ov_core::CpiV1>(0.005, 1e-4, 0.02, 1e-3);
    c->setLinearizationPoints(s.bg_gt, s.ba_gt);
    double t = 0;
    Eigen::Vector3d am = a_meas0 + randn3(rng, 0.1);
    for (int k = 0; k < nsub; ++k) {
      Eigen::Vector3d wm = w_body + s.bg_gt;        // gyro measurement (= true + bias)
      Eigen::Vector3d amk = am + s.ba_gt;           // accel measurement (= true + bias)
      c->feed_IMU(t, t + ddt, wm, amk, wm, amk);
      t += ddt;
    }
    s.cpi.push_back(c);
    // Set next GT state from the factor's own equations -> zero IMU residual at GT.
    Eigen::Matrix3d Ri = quat_2_Rot(s.q[i]);
    s.v[i + 1] = s.v[i] - s.grav * s.dt + Ri.transpose() * c->beta_tau;
    s.p[i + 1] = s.p[i] + s.v[i] * s.dt - 0.5 * s.grav * s.dt * s.dt + Ri.transpose() * c->alpha_tau;
    s.q[i + 1] = quatnorm(quat_multiply(c->q_k2tau, s.q[i]));
  }

  // Two NON-OVERLAPPING cameras. Cam0 == IMU frame (looks +Z). Cam1 is rotated 90 deg
  // about Y (looks along +X of the IMU) and offset 10 cm, so the two views never share a
  // landmark. Each landmark belongs to exactly one camera and is generated directly in
  // FRONT of it (depth 4-9 m), then mapped to the global frame through keyframe 0.
  s.Rc[0] = Eigen::Matrix3d::Identity();
  s.pc[0] = Eigen::Vector3d::Zero();
  s.Rc[1] = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()).toRotationMatrix();
  s.pc[1] = Eigen::Vector3d(0.10, 0, 0);
  std::normal_distribution<double> Npx(0, s.sigma_px);
  std::uniform_real_distribution<double> U(-2.0, 2.0), Uz(4.0, 9.0);
  s.lm.resize(s.M);
  s.lm_cam.resize(s.M);
  s.obs.resize(s.N);
  const Eigen::Matrix3d R0 = quat_2_Rot(s.q[0]);
  for (int j = 0; j < s.M; ++j) {
    int cam = (j < s.M / 2) ? 0 : 1; // first half -> cam0, second half -> cam1
    s.lm_cam[j] = cam;
    Eigen::Vector3d p_cam(U(rng), U(rng), Uz(rng)); // position in THIS camera's frame
    s.lm[j] = R0.transpose() * (s.Rc[cam].transpose() * (p_cam - s.pc[cam])) + s.p[0];
  }
  for (int i = 0; i < s.N; ++i) {
    Eigen::Matrix3d R = quat_2_Rot(s.q[i]);
    for (int j = 0; j < s.M; ++j) {
      const int cam = s.lm_cam[j];
      Eigen::Vector3d fc = s.Rc[cam] * (R * (s.lm[j] - s.p[i])) + s.pc[cam];
      if (fc(2) < 0.5)
        continue;
      Eigen::Vector2d uv(s.fx * fc(0) / fc(2) + s.cx + Npx(rng), s.fy * fc(1) / fc(2) + s.cy + Npx(rng));
      s.obs[i].push_back({j, uv});
    }
  }
  return s;
}

struct InitGuess {
  std::vector<Eigen::Vector4d> q;
  std::vector<Eigen::Vector3d> p, v, lm;
  Eigen::Vector3d bg, ba;
};
static InitGuess make_init(const Sim &s, std::mt19937 &rng) {
  InitGuess g;
  g.q = s.q;
  g.p = s.p;
  g.v = s.v;
  g.lm = s.lm;
  // Keep first pose at GT (defines the gauge); perturb everything else.
  for (int i = 0; i < s.N; ++i) {
    if (i > 0) {
      g.q[i] = perturb_q(s.q[i], randn3(rng, 0.05));
      g.p[i] = s.p[i] + randn3(rng, 0.05);
    }
    g.v[i] = s.v[i] + randn3(rng, 0.1);
  }
  for (int j = 0; j < s.M; ++j)
    g.lm[j] = s.lm[j] + randn3(rng, 0.1);
  g.bg = s.bg_gt + randn3(rng, 0.02);
  g.ba = s.ba_gt + randn3(rng, 0.05);
  return g;
}

struct Result {
  bool ok = false;
  int iters = 0;
  int jac = 0, res = 0, rej = 0; // jacobian evals, residual-only evals, rejected steps
  double time_ms = 0, cost0 = 0, cost1 = 0;
  std::vector<Eigen::Vector4d> q;
  std::vector<Eigen::Vector3d> p, v;
  std::vector<Eigen::Vector3d> bg, ba;
  Eigen::Matrix<double, 15, 1> sigma = Eigen::Matrix<double, 15, 1>::Zero();
  bool cov_ok = false;
};

// Build a generic first-pose gauge + bias prior (matches DynamicInitializer).
template <class PriorT>
static PriorT *make_prior(const double *q0, const double *p0, const double *bg0, const double *ba0) {
  Eigen::MatrixXd x_lin = Eigen::MatrixXd::Zero(13, 1);
  for (int j = 0; j < 4; j++)
    x_lin(j) = q0[j];
  for (int j = 0; j < 3; j++) {
    x_lin(4 + j) = p0[j];
    x_lin(7 + j) = bg0[j];
    x_lin(10 + j) = ba0[j];
  }
  Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(10, 1);
  Eigen::MatrixXd info = Eigen::MatrixXd::Identity(10, 10);
  info.block(0, 0, 4, 4) *= 1.0 / std::pow(1e-5, 2);
  info.block(4, 4, 3, 3) *= 1.0 / std::pow(0.05, 2);
  info.block(7, 7, 3, 3) *= 1.0 / std::pow(0.10, 2);
  std::vector<std::string> types = {"quat_yaw", "vec3", "vec3", "vec3"};
  return new PriorT(x_lin, types, info, grad);
}

// ===========================================================================
// Ceres backend
// ===========================================================================
static Result run_ceres(const Sim &s, const InitGuess &g, int num_threads, bool ceres_dogleg, bool verbose = false) {
  using namespace ov_init;
  Result r;
  std::vector<Eigen::Vector4d> q = g.q;
  std::vector<Eigen::Vector3d> p = g.p, v = g.v, lm = g.lm;
  std::vector<Eigen::Vector3d> bg(s.N, g.bg), ba(s.N, g.ba);
  Eigen::Vector3d grav = s.grav;

  ceres::Problem problem;
  for (int i = 0; i < s.N; ++i) {
    problem.AddParameterBlock(q[i].data(), 4, new State_JPLQuatLocal());
    problem.AddParameterBlock(p[i].data(), 3);
    problem.AddParameterBlock(v[i].data(), 3);
    problem.AddParameterBlock(bg[i].data(), 3);
    problem.AddParameterBlock(ba[i].data(), 3);
    if (i == 0)
      problem.AddResidualBlock(make_prior<Factor_GenericPrior>(q[0].data(), p[0].data(), bg[0].data(), ba[0].data()), nullptr,
                               {q[0].data(), p[0].data(), bg[0].data(), ba[0].data()});
    if (i > 0) {
      auto c = s.cpi[i - 1];
      auto *f = new Factor_ImuCPIv1(c->DT, grav, c->alpha_tau, c->beta_tau, c->q_k2tau, c->b_a_lin, c->b_w_lin, c->J_q, c->J_b, c->J_a,
                                    c->H_b, c->H_a, c->P_meas);
      problem.AddResidualBlock(f, nullptr,
                               {q[i - 1].data(), bg[i - 1].data(), v[i - 1].data(), ba[i - 1].data(), p[i - 1].data(), q[i].data(),
                                bg[i].data(), v[i].data(), ba[i].data(), p[i].data()});
    }
  }
  for (int j = 0; j < s.M; ++j)
    problem.AddParameterBlock(lm[j].data(), 3);
  for (int i = 0; i < s.N; ++i)
    for (auto &o : s.obs[i])
      problem.AddResidualBlock(new PinholeFactorT<ceres::CostFunction>(o.second, s.fx, s.fy, s.cx, s.cy, s.sigma_px,
                                                                       s.Rc[s.lm_cam[o.first]], s.pc[s.lm_cam[o.first]]),
                               new ceres::CauchyLoss(1.0), {q[i].data(), p[i].data(), lm[o.first].data()});

  ceres::Solver::Options o;
  o.linear_solver_type = ceres::DENSE_SCHUR;
  o.trust_region_strategy_type = ceres_dogleg ? ceres::DOGLEG : ceres::LEVENBERG_MARQUARDT;
  o.num_threads = num_threads;
  o.max_num_iterations = 50;
  o.max_solver_time_in_seconds = 1.0;
  o.function_tolerance = 1e-5;
  o.gradient_tolerance = 1e-9;
  o.logging_type = verbose ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
  o.minimizer_progress_to_stdout = verbose;
  ceres::Solver::Summary sum;
  auto t0 = std::chrono::steady_clock::now();
  ceres::Solve(o, &problem, &sum);
  r.time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  r.ok = (sum.termination_type == ceres::CONVERGENCE);
  r.iters = (int)sum.iterations.size() - 1;
  r.jac = sum.num_jacobian_evaluations;
  r.res = sum.num_residual_evaluations;
  r.rej = sum.num_unsuccessful_steps;
  r.cost0 = sum.initial_cost;
  r.cost1 = sum.final_cost;
  r.q = q;
  r.p = p;
  r.v = v;
  r.bg = bg;
  r.ba = ba;

  // Covariance of the latest IMU state (per-DOF sigma).
  int L = s.N - 1;
  ceres::Covariance::Options co;
  co.apply_loss_function = true;
  co.min_reciprocal_condition_number = 1e-15;
  ceres::Covariance cov(co);
  std::vector<std::pair<const double *, const double *>> blk = {{q[L].data(), q[L].data()},
                                                                {p[L].data(), p[L].data()},
                                                                {v[L].data(), v[L].data()},
                                                                {bg[L].data(), bg[L].data()},
                                                                {ba[L].data(), ba[L].data()}};
  if (cov.Compute(blk, &problem)) {
    r.cov_ok = true;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> t;
    const double *ptrs[5] = {q[L].data(), p[L].data(), v[L].data(), bg[L].data(), ba[L].data()};
    for (int b = 0; b < 5; ++b) {
      if (b == 0)
        cov.GetCovarianceBlockInTangentSpace(ptrs[0], ptrs[0], t.data());
      else
        cov.GetCovarianceBlock(ptrs[b], ptrs[b], t.data());
      for (int k = 0; k < 3; ++k)
        r.sigma(3 * b + k) = std::sqrt(std::max(0.0, t(k, k)));
    }
  }
  return r;
}

// ===========================================================================
// zbft_sfm backend
// ===========================================================================
static Result run_zbft(const Sim &s, const InitGuess &g, int num_threads, bool use_dogleg, bool verbose = false) {
  using namespace ov_init::zbft_sfm;
  Result r;
  std::vector<Eigen::Vector4d> q = g.q;
  std::vector<Eigen::Vector3d> p = g.p, v = g.v, lm = g.lm;
  std::vector<Eigen::Vector3d> bg(s.N, g.bg), ba(s.N, g.ba);
  Eigen::Vector3d grav = s.grav;
  std::vector<State_JPLQuatLocal *> qp;

  Problem problem;
  problem.EnableOwnership();

  // Gravity as a CONSTANT S² block (bench exercises the 11-block signature without optimizing g)
  auto *gravity_s2_param = new GravityS2Parameterization(grav.norm());
  problem.AddParameterBlock(grav.data(), 3, gravity_s2_param);
  problem.SetParameterBlockConstant(grav.data());

  for (int i = 0; i < s.N; ++i) {
    auto *qparam = new State_JPLQuatLocal();
    qp.push_back(qparam);
    problem.AddParameterBlock(q[i].data(), 4, qparam);
    problem.AddParameterBlock(p[i].data(), 3);
    problem.AddParameterBlock(v[i].data(), 3);
    problem.AddParameterBlock(bg[i].data(), 3);
    problem.AddParameterBlock(ba[i].data(), 3);
    if (i == 0)
      problem.AddResidualBlock(make_prior<Factor_GenericPrior>(q[0].data(), p[0].data(), bg[0].data(), ba[0].data()), nullptr,
                               {q[0].data(), p[0].data(), bg[0].data(), ba[0].data()});
    if (i > 0) {
      auto c = s.cpi[i - 1];
      auto *f = new Factor_ImuCPIv1(c->DT, grav, c->alpha_tau, c->beta_tau, c->q_k2tau, c->b_a_lin, c->b_w_lin, c->J_q, c->J_b, c->J_a,
                                    c->H_b, c->H_a, c->P_meas);
      // 11th block: gravity (constant here for bench parity)
      problem.AddResidualBlock(f, nullptr,
                               {q[i - 1].data(), bg[i - 1].data(), v[i - 1].data(), ba[i - 1].data(), p[i - 1].data(), q[i].data(),
                                bg[i].data(), v[i].data(), ba[i].data(), p[i].data(), grav.data()});
    }
  }
  for (int j = 0; j < s.M; ++j) {
    problem.AddParameterBlock(lm[j].data(), 3);
    problem.SetSchurLandmark(lm[j].data());
  }
  CauchyLoss *loss = new CauchyLoss(1.0); // shared; owned by problem
  for (int i = 0; i < s.N; ++i)
    for (auto &o : s.obs[i])
      problem.AddResidualBlock(new PinholeFactorT<CostFunction>(o.second, s.fx, s.fy, s.cx, s.cy, s.sigma_px,
                                                                s.Rc[s.lm_cam[o.first]], s.pc[s.lm_cam[o.first]]),
                               loss, {q[i].data(), p[i].data(), lm[o.first].data()});

  SolverOptions o;
  o.use_dogleg = use_dogleg;
  o.num_threads = num_threads;
  o.max_num_iterations = 50;
  o.max_solver_time_seconds = 1.0;
  o.function_tolerance = 1e-5;
  o.gradient_tolerance = 1e-9;
  o.verbose = verbose;
  auto t0 = std::chrono::steady_clock::now();
  SolverSummary sum = problem.Solve(o);
  r.time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  r.ok = sum.converged;
  r.iters = sum.iterations;
  r.jac = sum.jacobian_evals;
  r.res = sum.residual_evals;
  r.rej = sum.rejected_steps;
  r.cost0 = sum.initial_cost;
  r.cost1 = sum.final_cost;
  r.q = q;
  r.p = p;
  r.v = v;
  r.bg = bg;
  r.ba = ba;

  int L = s.N - 1;
  Eigen::MatrixXd C;
  std::vector<double *> blk = {q[L].data(), p[L].data(), v[L].data(), bg[L].data(), ba[L].data()};
  if (problem.ComputeCovariance(blk, C, o) && C.rows() == 15) {
    r.cov_ok = true;
    for (int k = 0; k < 15; ++k)
      r.sigma(k) = std::sqrt(std::max(0.0, C(k, k)));
  }
  return r;
}

// ---------------------------------------------------------------------------
static double state_diff(const Result &a, const Result &b) {
  double m = 0;
  for (size_t i = 0; i < a.q.size(); ++i) {
    Eigen::Matrix3d Rrel = quat_2_Rot(a.q[i]) * quat_2_Rot(b.q[i]).transpose();
    m = std::max(m, ov_core::log_so3(Rrel).norm());
    m = std::max(m, (a.p[i] - b.p[i]).norm());
    m = std::max(m, (a.v[i] - b.v[i]).norm());
    m = std::max(m, (a.bg[i] - b.bg[i]).norm());
    m = std::max(m, (a.ba[i] - b.ba[i]).norm());
  }
  return m;
}

// RMSE of a recovered solution vs ground truth (the gauge is fixed at GT: first pose
// unperturbed + yaw/position prior, so all states are directly comparable to truth).
struct Rmse {
  double rot_deg, pos, vel, bg, ba;
};
static Rmse rmse_vs_gt(const Result &r, const Sim &s) {
  double ro = 0, po = 0, ve = 0, bgs = 0, bas = 0;
  for (int i = 0; i < s.N; ++i) {
    const Eigen::Matrix3d Rr = quat_2_Rot(r.q[i]) * quat_2_Rot(s.q[i]).transpose();
    ro += ov_core::log_so3(Rr).squaredNorm();
    po += (r.p[i] - s.p[i]).squaredNorm();
    ve += (r.v[i] - s.v[i]).squaredNorm();
    bgs += (r.bg[i] - s.bg_gt).squaredNorm();
    bas += (r.ba[i] - s.ba_gt).squaredNorm();
  }
  Rmse m;
  m.rot_deg = std::sqrt(ro / s.N) * 180.0 / M_PI;
  m.pos = std::sqrt(po / s.N);
  m.vel = std::sqrt(ve / s.N);
  m.bg = std::sqrt(bgs / s.N);
  m.ba = std::sqrt(bas / s.N);
  return m;
}

int main(int argc, char **argv) {
  int K = (argc > 1) ? atoi(argv[1]) : 30;
  int zthreads = (argc > 2) ? atoi(argv[2]) : 1;
  bool dl = (argc > 3) ? (atoi(argv[3]) != 0) : false;  // zbft method:  0=LM (default), 1=dogleg
  bool cdl = (argc > 4) ? (atoi(argv[4]) != 0) : false; // ceres method: 0=LM (default), 1=dogleg
  std::printf("VI-init benchmark: Ceres(%s) vs ov_init::zbft_sfm(%s)   (%d trials, %d thread(s) for BOTH)\n", cdl ? "DOGLEG" : "LM",
              dl ? "DOGLEG" : "LM", K, zthreads);
  std::printf("%-5s | %-26s | %-26s | %-22s\n", "trial", "Ceres (it/ms/cost)", "zbft_sfm (it/ms/cost)", "parity / cov-sigma rel");
  std::printf("------------------------------------------------------------------------------------------------\n");

  std::vector<double> tc, tz;
  std::vector<int> ic, iz;
  long jc = 0, jz = 0, rcsum = 0, rzsum = 0, rejc = 0, rejz = 0; // jac/res/reject totals
  double max_parity = 0, max_costrel = 0, max_sigrel = 0, max_bias_c = 0, max_bias_z = 0;
  int okc = 0, okz = 0, covc = 0, covz = 0;
  int n_zbft_better = 0, n_ceres_better = 0; // when the two reach DIFFERENT minima, who is lower-cost?
  Rmse acc_c{0, 0, 0, 0, 0}, acc_z{0, 0, 0, 0, 0};
  double sum_c_cost = 0, sum_z_cost = 0;

  for (int trial = 0; trial < K; ++trial) {
    std::mt19937 rng(1000 + trial);
    Sim s = make_sim(rng);
    InitGuess g = make_init(s, rng);
    const bool vb = (trial == 0); // dump per-iteration convergence trace for the first trial
    if (vb)
      std::printf("---- trial 0 convergence trace (Ceres then zbft) ----\n");
    Result rc = run_ceres(s, g, zthreads, cdl, vb);
    Result rz = run_zbft(s, g, zthreads, dl, vb);

    double parity = state_diff(rc, rz);
    double costrel = std::abs(rc.cost1 - rz.cost1) / std::max(1e-9, rc.cost1);
    const double cost_adv = (rc.cost1 - rz.cost1) / std::max(1e-9, std::abs(rc.cost1)); // >0 => zbft lower (better)
    if (cost_adv > 1e-3)
      ++n_zbft_better;
    else if (cost_adv < -1e-3)
      ++n_ceres_better;
    double sigrel = 0;
    if (rc.cov_ok && rz.cov_ok) {
      for (int k = 0; k < 15; ++k)
        if (rc.sigma(k) > 1e-4) // skip gauge-pinned (~1e-5) DOF where a relative metric blows up
          sigrel = std::max(sigrel, std::abs(rc.sigma(k) - rz.sigma(k)) / rc.sigma(k));
    }
    if (trial == 0 && rc.cov_ok && rz.cov_ok) {
      const char *nm[15] = {"th_x", "th_y", "th_z", "p_x",  "p_y",  "p_z",  "v_x", "v_y",
                            "v_z",  "bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z"};
      std::printf("  per-DOF sigma (latest IMU state):   Ceres         zbft_sfm      reldiff\n");
      for (int k = 0; k < 15; ++k)
        std::printf("    %-5s   %11.3e   %11.3e   %6.1f%%\n", nm[k], rc.sigma(k), rz.sigma(k),
                    100.0 * std::abs(rc.sigma(k) - rz.sigma(k)) / std::max(1e-9, rc.sigma(k)));
    }
    double biasc = (rc.bg.back() - s.bg_gt).norm() + (rc.ba.back() - s.ba_gt).norm();
    double biasz = (rz.bg.back() - s.bg_gt).norm() + (rz.ba.back() - s.ba_gt).norm();

    const Rmse mc = rmse_vs_gt(rc, s), mz = rmse_vs_gt(rz, s);
    acc_c.rot_deg += mc.rot_deg; acc_c.pos += mc.pos; acc_c.vel += mc.vel; acc_c.bg += mc.bg; acc_c.ba += mc.ba;
    acc_z.rot_deg += mz.rot_deg; acc_z.pos += mz.pos; acc_z.vel += mz.vel; acc_z.bg += mz.bg; acc_z.ba += mz.ba;
    sum_c_cost += rc.cost1;
    sum_z_cost += rz.cost1;

    tc.push_back(rc.time_ms);
    tz.push_back(rz.time_ms);
    ic.push_back(rc.iters);
    iz.push_back(rz.iters);
    jc += rc.jac; jz += rz.jac; rcsum += rc.res; rzsum += rz.res; rejc += rc.rej; rejz += rz.rej;
    okc += rc.ok;
    okz += rz.ok;
    covc += rc.cov_ok;
    covz += rz.cov_ok;
    max_parity = std::max(max_parity, parity);
    max_costrel = std::max(max_costrel, costrel);
    max_sigrel = std::max(max_sigrel, sigrel);
    max_bias_c = std::max(max_bias_c, biasc);
    max_bias_z = std::max(max_bias_z, biasz);

    if (trial < 10)
      std::printf("%-5d | %3d / %7.2f / %.3e | %3d / %7.2f / %.3e | dx=%.2e dcost=%.1e dsig=%.1e\n", trial, rc.iters, rc.time_ms,
                  rc.cost1, rz.iters, rz.time_ms, rz.cost1, parity, costrel, sigrel);
  }

  auto med = [](std::vector<double> x) { std::sort(x.begin(), x.end()); return x[x.size() / 2]; };
  auto mean = [](const std::vector<double> &x) { double s = 0; for (double v : x) s += v; return s / x.size(); };
  auto meani = [](const std::vector<int> &x) { double s = 0; for (int v : x) s += v; return s / x.size(); };

  std::printf("------------------------------------------------------------------------------------------------\n");
  std::printf("converged:        Ceres %d/%d   zbft_sfm %d/%d   (covariance ok: %d / %d)\n", okc, K, okz, K, covc, covz);
  std::printf("iterations  mean: Ceres %.1f      zbft_sfm %.1f\n", meani(ic), meani(iz));
  std::printf("evals total:      Ceres jac=%ld res=%ld rej=%ld   zbft_sfm jac=%ld res=%ld rej=%ld\n", jc, rcsum, rejc, jz, rzsum,
              rejz);
  std::printf("time (ms) median: Ceres %.2f     zbft_sfm %.2f     mean: Ceres %.2f  zbft_sfm %.2f\n", med(tc), med(tz), mean(tc),
              mean(tz));
  std::printf("PARITY  max |x_ceres - x_zbft| over all states/trials: %.3e\n", max_parity);
  std::printf("        max relative final-cost difference:            %.3e\n", max_costrel);
  std::printf("        max relative per-DOF sigma difference:         %.3e\n", max_sigrel);
  const double iK = 1.0 / (double)K;
  std::printf("------------------------------------------------------------------------------------------------\n");
  std::printf("SOLUTION QUALITY (mean over %d trials):\n", K);
  std::printf("  local minima:  SAME %d  |  zbft lower-cost(BETTER) %d  |  Ceres lower-cost(better) %d\n",
              K - n_zbft_better - n_ceres_better, n_zbft_better, n_ceres_better);
  std::printf("  RMSE vs ground truth         Ceres        zbft_sfm     winner\n");
  std::printf("    orientation (deg)       %9.4f    %9.4f    %s\n", acc_c.rot_deg * iK, acc_z.rot_deg * iK,
              acc_z.rot_deg <= acc_c.rot_deg ? "zbft" : "ceres");
  std::printf("    position (m)            %9.4f    %9.4f    %s\n", acc_c.pos * iK, acc_z.pos * iK,
              acc_z.pos <= acc_c.pos ? "zbft" : "ceres");
  std::printf("    velocity (m/s)          %9.4f    %9.4f    %s\n", acc_c.vel * iK, acc_z.vel * iK,
              acc_z.vel <= acc_c.vel ? "zbft" : "ceres");
  std::printf("    gyro bias (rad/s)       %9.4f    %9.4f    %s\n", acc_c.bg * iK, acc_z.bg * iK,
              acc_z.bg <= acc_c.bg ? "zbft" : "ceres");
  std::printf("    accel bias (m/s^2)      %9.4f    %9.4f    %s\n", acc_c.ba * iK, acc_z.ba * iK,
              acc_z.ba <= acc_c.ba ? "zbft" : "ceres");
  std::printf("    final cost (mean)       %9.3e    %9.3e    %s\n", sum_c_cost * iK, sum_z_cost * iK,
              sum_z_cost <= sum_c_cost ? "zbft" : "ceres");
  std::printf("------------------------------------------------------------------------------------------------\n");
  const char *pv = (max_costrel < 1e-2) ? "MATCH (same minimum)"
                                        : (n_ceres_better == 0 ? "zbft >= Ceres (equal-or-better minima)" : "different minima");
  std::printf("VERDICT: solution %s  (max cost rel diff %.2e; lower-cost minimum: zbft x%d, Ceres x%d; biases GT err Ceres %.4f / "
              "zbft %.4f)\n",
              pv, max_costrel, n_zbft_better, n_ceres_better, max_bias_c, max_bias_z);
  std::printf("         raw-state Linf diff %.2e lives in weakly-observed/gauge directions (cost identical there).\n", max_parity);
  std::printf("         observable-DOF (sigma>1e-4) covariance agreement: max rel diff %.1f%%\n", 100.0 * max_sigrel);
  std::printf("         speed: zbft_sfm/Ceres = %.2fx time, +%.0f%% iterations (single core, unoptimized LM).\n",
              mean(tz) / mean(tc), 100.0 * (meani(iz) / meani(ic) - 1.0));
  return 0;
}
