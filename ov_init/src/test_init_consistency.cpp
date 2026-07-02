/*
 * Monte-Carlo NEES consistency gold standard for the ceres-free (ov_init::zbft_sfm)
 * S²-gravity dynamic initializer. NO OpenCV / NO Ceres dependency.
 * Contributor: Joao Leonardo Silva Cotta (@zauberflote1)
 *
 * Validates that the covariance recovered by Problem::ComputeCovariance() is statistically
 * consistent with the actual estimation error distribution. For a consistent estimator the
 * normalized estimation error squared (NEES = eᵀ Σ⁻¹ e) over the 15-DOF IMU state
 * [θ, p, v, bg, ba] is χ²(15)-distributed, so the average NEES (ANEES) → 15 (ANEES/15 → 1).
 *
 * It also (a) reports per-block NEES, which incidentally pins the orientation error-frame
 * convention (e_θ matching the covariance ⇒ NEES_θ ≈ 3), and (b) checks that a flipped /
 * upside-down gravity seed is rejected by the post-solve gravity gate.
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
#include <memory>
#include <random>
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
// Simulated VI-init problem: GT trajectory integrated from NOISE-FREE IMU, plus the
// measurements the solver actually sees -- separately integrated NOISY CPIs and noisy pixels.
// ---------------------------------------------------------------------------
struct Sim {
  int N = 10, M = 60;
  double dt = 0.1, fx = 400, fy = 400, cx = 320, cy = 240, sigma_px = 1.0;
  double imu_rate = 200.0, gmag = 9.81;
  double sw = 0.005, swb = 1e-4, sa = 0.02, sab = 1e-3; // CPI noise spectral densities
  Eigen::Vector3d grav, bg_gt, ba_gt;
  std::vector<Eigen::Vector4d> q;    // GT q_GtoIi
  std::vector<Eigen::Vector3d> p, v; // GT
  std::vector<Eigen::Vector3d> lm;   // GT landmarks (global)
  std::vector<std::shared_ptr<ov_core::CpiV1>> cpi; // NOISY measurement CPIs (size N)
  std::vector<std::vector<std::pair<int, Eigen::Vector2d>>> obs; // NOISY pixel observations
};

static Eigen::Vector3d randn3(std::mt19937 &rng, double s) {
  std::normal_distribution<double> N(0, 1);
  return Eigen::Vector3d(s * N(rng), s * N(rng), s * N(rng));
}

static Sim make_sim(std::mt19937 &rng) {
  Sim s;
  s.grav = Eigen::Vector3d(0, 0, s.gmag);
  s.bg_gt = randn3(rng, 0.01);
  s.ba_gt = randn3(rng, 0.05);

  auto omega_body = [](double t) {
    return Eigen::Vector3d(0.5 * std::sin(1.3 * t), 0.6 * std::sin(1.1 * t + 0.5), 0.4 * std::sin(0.9 * t + 1.0));
  };
  auto accel_body = [&](double t) { return Eigen::Vector3d(0.6 * std::sin(0.7 * t), 0.4 * std::cos(0.9 * t), s.gmag + 0.3 * std::sin(0.6 * t)); };

  s.q.resize(s.N);
  s.p.resize(s.N);
  s.v.resize(s.N);
  s.q[0] = Eigen::Vector4d(0, 0, 0, 1);
  s.p[0] = Eigen::Vector3d::Zero();
  s.v[0] = randn3(rng, 0.1);

  s.cpi.resize(s.N);
  s.cpi[0] = nullptr;
  const double dti = 1.0 / s.imu_rate, rt = std::sqrt(s.imu_rate);
  std::normal_distribution<double> N01(0, 1);
  for (int i = 1; i < s.N; ++i) {
    // Clean CPI -> defines GT; noisy CPI -> the measurement the solver uses.
    auto clean = std::make_shared<ov_core::CpiV1>(s.sw, s.swb, s.sa, s.sab, true);
    auto noisy = std::make_shared<ov_core::CpiV1>(s.sw, s.swb, s.sa, s.sab, true);
    clean->setLinearizationPoints(s.bg_gt, s.ba_gt);
    noisy->setLinearizationPoints(s.bg_gt, s.ba_gt);
    double t0 = (i - 1) * s.dt, t1 = i * s.dt;
    for (double t = t0; t < t1 - 1e-9; t += dti) {
      double tn = std::min(t + dti, t1);
      Eigen::Vector3d w0 = omega_body(t) + s.bg_gt, a0 = accel_body(t) + s.ba_gt;
      Eigen::Vector3d w1 = omega_body(tn) + s.bg_gt, a1 = accel_body(tn) + s.ba_gt;
      clean->feed_IMU(t, tn, w0, a0, w1, a1);
      Eigen::Vector3d nw0 = w0 + s.sw * rt * Eigen::Vector3d(N01(rng), N01(rng), N01(rng));
      Eigen::Vector3d na0 = a0 + s.sa * rt * Eigen::Vector3d(N01(rng), N01(rng), N01(rng));
      Eigen::Vector3d nw1 = w1 + s.sw * rt * Eigen::Vector3d(N01(rng), N01(rng), N01(rng));
      Eigen::Vector3d na1 = a1 + s.sa * rt * Eigen::Vector3d(N01(rng), N01(rng), N01(rng));
      noisy->feed_IMU(t, tn, nw0, na0, nw1, na1);
    }
    s.cpi[i] = noisy;
    // Derive GT from the CLEAN preintegration (zero IMU residual at GT vs the clean factor).
    Eigen::Matrix3d R0 = quat_2_Rot(s.q[i - 1]);
    double DT = clean->DT;
    s.q[i] = quatnorm(quat_multiply(clean->q_k2tau, s.q[i - 1]));
    s.v[i] = s.v[i - 1] - s.grav * DT + R0.transpose() * clean->beta_tau;
    s.p[i] = s.p[i - 1] + s.v[i - 1] * DT - 0.5 * s.grav * DT * DT + R0.transpose() * clean->alpha_tau;
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
  std::vector<Eigen::Vector3d> p, v, lm;
  Eigen::Vector3d bg, ba, grav;
};
static Eigen::Vector4d perturb_q(const Eigen::Vector4d &q, const Eigen::Vector3d &dth) {
  Eigen::Vector4d dq;
  dq << 0.5 * dth, 1.0;
  return quatnorm(quat_multiply(quatnorm(dq), q));
}
// Reset-prior mode knobs (argv): ba first-pose prior sigma (0.10 = legacy config-seed mode;
// ~0.02 = soft-reset tightened prior) and the ba seed error the sim draws (0.02 = legacy;
// larger with a tight sigma = the corrupted-prior failure mode the production gates exist for).
static double g_ba_prior_sigma = 0.10;
static double g_ba_seed_err = 0.03; // matches the legacy MC seed draw

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
  g.bg = s.bg_gt + randn3(rng, 0.01);
  g.ba = s.ba_gt + randn3(rng, g_ba_seed_err);
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
  double grav_err_deg = 0;
};

// Run the ceres-free init (free S² gravity, full-quat prior, shared biases, Cauchy reproj),
// recover the 15x15 covariance of the newest IMU state, and form the error vs GT.
static Result run_init(const Sim &s, const InitGuess &g, int gmode) {
  // gmode: 0 = free S² gravity; 1 = free + weak gravity prior; 2 = gravity FIXED at GT
  Result r;
  std::vector<Eigen::Vector4d> q = g.q;
  std::vector<Eigen::Vector3d> p = g.p, v = g.v, lm = g.lm;
  Eigen::Vector3d bg = g.bg, ba = g.ba;
  Eigen::Vector3d grav = (gmode == 2) ? s.grav : g.grav;

  Problem problem;
  problem.EnableOwnership();

  auto *gs2 = new GravityS2Parameterization(s.gmag);
  problem.AddParameterBlock(grav.data(), 3, gs2);
  if (gmode == 2)
    problem.SetParameterBlockConstant(grav.data()); // known-good baseline (matches Ceres to 0.1%)
  if (gmode == 1) {
    Eigen::MatrixXd glin(3, 1);
    glin << grav(0), grav(1), grav(2);
    Eigen::MatrixXd ginfo = Eigen::MatrixXd::Identity(3, 3) / std::pow(0.5, 2); // σ≈0.5 m/s² ≈ 3°
    std::vector<std::string> gt = {"vec3"};
    problem.AddResidualBlock(new Factor_GenericPrior(glin, gt, ginfo, Eigen::MatrixXd::Zero(3, 1)), nullptr, {grav.data()});
  }
  problem.AddParameterBlock(bg.data(), 3);
  problem.AddParameterBlock(ba.data(), 3);

  std::vector<State_JPLQuatLocal *> qp;
  for (int i = 0; i < s.N; ++i) {
    auto *qparam = new State_JPLQuatLocal();
    qp.push_back(qparam);
    problem.AddParameterBlock(q[i].data(), 4, qparam);
    problem.AddParameterBlock(p[i].data(), 3);
    problem.AddParameterBlock(v[i].data(), 3);
    if (i == 0)
      problem.AddResidualBlock(make_prior_s2(q[0].data(), p[0].data(), bg.data(), ba.data()), nullptr,
                               {q[0].data(), p[0].data(), bg.data(), ba.data()});
    if (i > 0) {
      auto c = s.cpi[i];
      auto *f = new Factor_ImuCPIv1(c->DT, grav, c->alpha_tau, c->beta_tau, c->q_k2tau, c->b_a_lin, c->b_w_lin, c->J_q, c->J_b, c->J_a,
                                    c->H_b, c->H_a, c->P_meas);
      problem.AddResidualBlock(f, nullptr,
                               {q[i - 1].data(), bg.data(), v[i - 1].data(), ba.data(), p[i - 1].data(), q[i].data(), bg.data(),
                                v[i].data(), ba.data(), p[i].data(), grav.data()});
    }
  }
  for (int j = 0; j < s.M; ++j) {
    problem.AddParameterBlock(lm[j].data(), 3);
    problem.SetSchurLandmark(lm[j].data());
  }
  auto *loss = new CauchyLoss(1.0);
  for (int i = 0; i < s.N; ++i)
    for (auto &o : s.obs[i])
      problem.AddResidualBlock(new PinholeFactor(o.second, s.fx, s.fy, s.cx, s.cy, s.sigma_px), loss, {q[i].data(), p[i].data(), lm[o.first].data()});

  SolverOptions o;
  o.use_dogleg = false;
  o.num_threads = 1; // deterministic for NEES
  o.max_num_iterations = 50;
  o.max_solver_time_seconds = 1.0;
  o.function_tolerance = 1e-5;
  o.gradient_tolerance = 1e-9;
  SolverSummary sum = problem.Solve(o);
  r.ok = sum.converged;
  r.iters = sum.iterations;
  r.msg = sum.message;

  // Gravity gate: reject a flipped/badly-tilted gravity (matches DynamicInitializer).
  Eigen::Vector3d gexp(0, 0, s.gmag);
  double ang = std::acos(std::min(1.0, std::max(-1.0, grav.dot(gexp) / (grav.norm() * gexp.norm())))) * 180.0 / M_PI;
  r.grav_err_deg = ang;
  if (ang > 30.0) {
    r.rejected_flip = true;
    return r;
  }

  int L = s.N - 1;
  std::vector<double *> blk = {q[L].data(), p[L].data(), v[L].data(), bg.data(), ba.data()};
  Eigen::MatrixXd C;
  if (problem.ComputeCovariance(blk, C, o) && C.rows() == 15) {
    r.cov_ok = true;
    r.cov = C;
  }

  // Error of the newest IMU state vs GT, in the solver's local error coordinates.
  Eigen::Matrix3d R_gt = quat_2_Rot(s.q[L]), R_est = quat_2_Rot(q[L]);
  r.err.segment<3>(0) = -log_so3(R_gt * R_est.transpose());
  r.err.segment<3>(3) = s.p[L] - p[L];
  r.err.segment<3>(6) = s.v[L] - v[L];
  r.err.segment<3>(9) = s.bg_gt - bg;
  r.err.segment<3>(12) = s.ba_gt - ba;
  return r;
}

// Validate the Phase-1 re-align covariance transform T = blkdiag(I, R, R, I, I): the error of a
// world-frame-re-aligned estimate vs the re-aligned truth must equal T times the original error
// (in particular delta-theta is INVARIANT -> T_theta_theta = I). If T were wrong (e.g. R on theta),
// the orientation rows would mismatch. This is the numerical check behind the production fix.
static void check_realign_transform() {
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
}

// χ²(15) two-sided 99% interval (approx): [4.60, 32.80]; the K-trial mean tightens around 15.
int main(int argc, char **argv) {
  check_realign_transform();
  int K = (argc > 1) ? atoi(argv[1]) : 500;
  int gmode = (argc > 2) ? atoi(argv[2]) : 0;     // 0=free, 1=free+prior, 2=fixed@GT
  int do_inflate = (argc > 3) ? atoi(argv[3]) : 0; // apply the production init_dyn_inflation_* as a congruence
  g_ba_prior_sigma = (argc > 4) ? atof(argv[4]) : 0.10; // ba prior sigma (0.10 legacy; ~0.02 reset-prior mode)
  g_ba_seed_err = (argc > 5) ? atof(argv[5]) : 0.03;    // ba seed error (0.03 legacy; raise w/ tight sigma = corrupted prior)
  const char *mname[3] = {"free-S2", "free-S2 + grav-prior", "gravity-FIXED@GT"};
  std::printf("==== zbft_sfm init NEES consistency (%d trials, gravity=%s, inflation=%s) ====\n", K,
              mname[(gmode < 0 || gmode > 2) ? 0 : gmode], do_inflate ? "ON" : "off");

  double sum_nees = 0, sum_th = 0, sum_p = 0, sum_v = 0, sum_bg = 0, sum_ba = 0;
  double sum_grav_err = 0, sum_iters = 0;
  std::string msg0;
  int ok = 0, cov_ok = 0, used = 0;
  for (int t = 0; t < K; ++t) {
    std::mt19937 rng(2000 + t);
    Sim s = make_sim(rng);
    InitGuess g = make_init(s, rng, 0.0);
    Result r = run_init(s, g, gmode);
    ok += r.ok;
    sum_iters += r.iters;
    if (t == 0)
      msg0 = r.msg;
    sum_grav_err += r.grav_err_deg;
    if (!r.cov_ok)
      continue;
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
    Eigen::Matrix<double, 15, 15> info = r.cov.inverse();
    double nees = (r.err.transpose() * info * r.err)(0, 0);
    if (!std::isfinite(nees) || nees < 0)
      continue;
    used++;
    sum_nees += nees;
    sum_th += (r.err.segment<3>(0).transpose() * info.block<3, 3>(0, 0) * r.err.segment<3>(0))(0, 0);
    sum_p += (r.err.segment<3>(3).transpose() * info.block<3, 3>(3, 3) * r.err.segment<3>(3))(0, 0);
    sum_v += (r.err.segment<3>(6).transpose() * info.block<3, 3>(6, 6) * r.err.segment<3>(6))(0, 0);
    sum_bg += (r.err.segment<3>(9).transpose() * info.block<3, 3>(9, 9) * r.err.segment<3>(9))(0, 0);
    sum_ba += (r.err.segment<3>(12).transpose() * info.block<3, 3>(12, 12) * r.err.segment<3>(12))(0, 0);
  }
  double iU = 1.0 / std::max(1, used);
  double anees = sum_nees * iU;
  std::printf("converged: %d/%d   cov_ok: %d   used: %d   mean iters: %.1f   trial0 msg: \"%s\"\n", ok, K, cov_ok, used,
              sum_iters / std::max(1, K), msg0.c_str());
  std::printf("mean gravity err: %.4f deg\n", sum_grav_err / std::max(1, K));
  std::printf("ANEES (total, ideal=15):   %.3f   [ANEES/15 = %.3f, ideal 1.0]\n", anees, anees / 15.0);
  std::printf("  per-block mean NEES (ideal=3 each):  theta=%.3f  p=%.3f  v=%.3f  bg=%.3f  ba=%.3f\n", sum_th * iU, sum_p * iU,
              sum_v * iU, sum_bg * iU, sum_ba * iU);
  bool consistent = (anees > 11.0 && anees < 20.0); // generous band around 15
  std::printf("CONSISTENCY: %s  (tight band [13.5,16.5]; gross-fail outside [11,20])\n", consistent ? "PASS (in band)" : "OUT OF BAND");

  // Flip rejection: a flipped gravity seed (~170°) must be gated out.
  int flips = 50, rejected = 0;
  for (int t = 0; t < flips; ++t) {
    std::mt19937 rng(9000 + t);
    Sim s = make_sim(rng);
    InitGuess g = make_init(s, rng, 0.0);
    g.grav = -s.grav + randn3(rng, 0.5); // upside-down
    Result r = run_init(s, g, gmode);
    rejected += r.rejected_flip ? 1 : 0;
  }
  std::printf("FLIP REJECTION: %d/%d rejected by gravity gate  (want %d/%d)\n", rejected, flips, flips, flips);

  return (consistent && rejected == flips) ? 0 : 1;
}
