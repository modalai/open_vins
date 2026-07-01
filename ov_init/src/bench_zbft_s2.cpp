/*
 * S² gravity MLE benchmark (zbft_sfm only, no Ceres dependency).
 * Contributor: Joao Leonardo Silva Cotta (@zauberflote1)
 * Tests the S² gravity parameterization on synthetic VI-init problems.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>

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

// Pinhole reprojection factor (no distortion)
class PinholeFactor : public ov_init::zbft_sfm::CostFunction {
public:
  Eigen::Vector2d uv;
  Eigen::Matrix3d Rc;
  Eigen::Vector3d pc;
  double fx, fy, cx, cy, isig;
  PinholeFactor(const Eigen::Vector2d &uv_, double fx_, double fy_, double cx_, double cy_, double sigma_px,
                const Eigen::Matrix3d &Rc_, const Eigen::Vector3d &pc_)
      : uv(uv_), Rc(Rc_), pc(pc_), fx(fx_), fy(fy_), cx(cx_), cy(cy_), isig(1.0 / sigma_px) {
    set_num_residuals(2);
    mutable_parameter_block_sizes()->push_back(4);
    mutable_parameter_block_sizes()->push_back(3);
    mutable_parameter_block_sizes()->push_back(3);
  }
  bool Evaluate(double const *const *p, double *res, double **jac) const override {
    Eigen::Vector4d q = Eigen::Map<const Eigen::Vector4d>(p[0]);
    Eigen::Matrix3d R = quat_2_Rot(q);
    Eigen::Vector3d pI = Eigen::Map<const Eigen::Vector3d>(p[1]);
    Eigen::Vector3d pf = Eigen::Map<const Eigen::Vector3d>(p[2]);
    Eigen::Vector3d pIi = R * (pf - pI);
    Eigen::Vector3d fc = Rc * pIi + pc;
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

struct Sim {
  int N = 40, M = 120;  // Longer window: 40 keyframes = 1.3s
  double dt = 1.0 / 30.0, fx = 320, fy = 320, cx = 320, cy = 240, sigma_px = 1.2;
  Eigen::Vector3d grav = Eigen::Vector3d(0, 0, 9.81), bg_gt, ba_gt;
  std::vector<Eigen::Vector4d> q;
  std::vector<Eigen::Vector3d> p, v;
  std::vector<Eigen::Vector3d> lm;
  std::vector<int> lm_cam;
  Eigen::Matrix3d Rc[2];
  Eigen::Vector3d pc[2];
  std::vector<std::shared_ptr<ov_core::CpiV1>> cpi;
  std::vector<std::vector<std::pair<int, Eigen::Vector2d>>> obs;
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

  s.q.resize(s.N);
  s.p.resize(s.N);
  s.v.resize(s.N);
  s.q[0] = Eigen::Vector4d(0, 0, 0, 1);
  s.p[0] = randn3(rng, 0.1);
  s.v[0] = randn3(rng, 0.3);
  int nsub = 11;
  double ddt = s.dt / nsub;

  // Generate a trajectory with VARYING rotation to make gravity observable
  // Use sinusoidal angular velocity patterns on each axis
  std::uniform_real_distribution<double> freq_dist(1.0, 3.0);
  std::uniform_real_distribution<double> amp_dist(0.8, 1.5);  // More aggressive rotation
  double wx_freq = freq_dist(rng), wy_freq = freq_dist(rng), wz_freq = freq_dist(rng);
  double wx_amp = amp_dist(rng), wy_amp = amp_dist(rng), wz_amp = amp_dist(rng);
  double ax_amp = 0.5 + 0.3 * std::abs(std::normal_distribution<double>(0, 1)(rng));

  for (int i = 0; i < s.N - 1; ++i) {
    auto c = std::make_shared<ov_core::CpiV1>(0.005, 1e-4, 0.02, 1e-3);
    c->setLinearizationPoints(s.bg_gt, s.ba_gt);
    double t = 0;
    double t_global = i * s.dt;
    for (int k = 0; k < nsub; ++k) {
      double tk = t_global + k * ddt;
      // Time-varying angular velocity (makes gravity observable through varied orientations)
      Eigen::Vector3d w_body(wx_amp * std::sin(wx_freq * tk),
                             wy_amp * std::sin(wy_freq * tk + 1.0),
                             wz_amp * std::sin(wz_freq * tk + 2.0));
      Eigen::Vector3d wm = w_body + s.bg_gt;
      // Time-varying specific force
      Eigen::Vector3d a_body(ax_amp * std::sin(0.8 * tk), 0.3 * std::cos(1.2 * tk), 0.2 * std::sin(0.5 * tk));
      Eigen::Vector3d am = a_body + s.ba_gt;
      c->feed_IMU(t, t + ddt, wm, am, wm, am);
      t += ddt;
    }
    s.cpi.push_back(c);
    Eigen::Matrix3d Ri = quat_2_Rot(s.q[i]);
    s.v[i + 1] = s.v[i] - s.grav * s.dt + Ri.transpose() * c->beta_tau;
    s.p[i + 1] = s.p[i] + s.v[i] * s.dt - 0.5 * s.grav * s.dt * s.dt + Ri.transpose() * c->alpha_tau;
    s.q[i + 1] = quatnorm(quat_multiply(c->q_k2tau, s.q[i]));
  }

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
    int cam = (j < s.M / 2) ? 0 : 1;
    s.lm_cam[j] = cam;
    Eigen::Vector3d p_cam(U(rng), U(rng), Uz(rng));
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
  Eigen::Vector3d bg, ba, grav;
};
static InitGuess make_init(const Sim &s, std::mt19937 &rng, bool perturb_gravity) {
  InitGuess g;
  g.q = s.q;
  g.p = s.p;
  g.v = s.v;
  g.lm = s.lm;
  g.grav = s.grav;
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
  if (perturb_gravity) {
    // Perturb gravity direction by up to 15 degrees
    Eigen::Vector3d axis = randn3(rng, 1.0).normalized();
    std::uniform_real_distribution<double> angle_dist(0.05, 0.26); // 3-15 deg
    double angle = angle_dist(rng);
    Eigen::AngleAxisd rot(angle, axis);
    g.grav = rot * s.grav;
  }
  return g;
}

struct Result {
  bool ok = false;
  int iters = 0;
  double time_ms = 0, cost0 = 0, cost1 = 0;
  std::vector<Eigen::Vector4d> q;
  std::vector<Eigen::Vector3d> p, v;
  Eigen::Vector3d bg, ba;  // Shared biases
  Eigen::Vector3d grav_final;
  double grav_err_deg = 0;
};

template <class PriorT>
static PriorT *make_prior(const double *q0, const double *p0, const double *bg0, const double *ba0) {
  // Use FULL orientation prior (quat, 3-DOF) to fix the gauge AND allow gravity to be observable
  // The gravity direction is then constrained relative to this fixed world frame.
  Eigen::MatrixXd x_lin = Eigen::MatrixXd::Zero(13, 1);
  for (int j = 0; j < 4; j++)
    x_lin(j) = q0[j];
  for (int j = 0; j < 3; j++) {
    x_lin(4 + j) = p0[j];
    x_lin(7 + j) = bg0[j];
    x_lin(10 + j) = ba0[j];
  }
  Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(12, 1);
  Eigen::MatrixXd info = Eigen::MatrixXd::Identity(12, 12);
  // Tight prior on full orientation (σ=0.001 rad ≈ 0.06°)
  info.block(0, 0, 3, 3) *= 1.0 / std::pow(0.001, 2);
  // Tight prior on position (σ=0.001 m)
  info.block(3, 3, 3, 3) *= 1.0 / std::pow(0.001, 2);
  // Moderate prior on gyro bias (σ=0.01 rad/s)
  info.block(6, 6, 3, 3) *= 1.0 / std::pow(0.01, 2);
  // Moderate prior on accel bias (σ=0.05 m/s²)
  info.block(9, 9, 3, 3) *= 1.0 / std::pow(0.05, 2);
  std::vector<std::string> types = {"quat", "vec3", "vec3", "vec3"};
  return new PriorT(x_lin, types, info, grad);
}

static Result run_zbft(const Sim &s, const InitGuess &g, int num_threads, bool use_dogleg,
                       bool optimize_gravity, bool verbose = false) {
  using namespace ov_init::zbft_sfm;
  Result r;
  std::vector<Eigen::Vector4d> q = g.q;
  std::vector<Eigen::Vector3d> p = g.p, v = g.v, lm = g.lm;
  // SHARED biases (single set for entire window, like real DynamicInitializer)
  Eigen::Vector3d bg = g.bg, ba = g.ba;
  Eigen::Vector3d grav = g.grav;

  Problem problem;
  problem.EnableOwnership();

  // Gravity as S² block with weak prior toward initial direction
  auto *gravity_s2_param = new GravityS2Parameterization(s.grav.norm());
  problem.AddParameterBlock(grav.data(), 3, gravity_s2_param);
  if (!optimize_gravity) {
    problem.SetParameterBlockConstant(grav.data());
  } else {
    // Add weak prior on gravity to prevent drift due to gravity-bias ambiguity
    // σ = 0.5 m/s² ≈ 3° direction uncertainty
    Eigen::MatrixXd grav_lin(3, 1);
    grav_lin << grav(0), grav(1), grav(2);
    Eigen::MatrixXd grav_info = Eigen::MatrixXd::Identity(3, 3) / std::pow(0.5, 2);
    Eigen::MatrixXd grav_grad = Eigen::MatrixXd::Zero(3, 1);
    std::vector<std::string> grav_types = {"vec3"};
    problem.AddResidualBlock(new Factor_GenericPrior(grav_lin, grav_types, grav_info, grav_grad), nullptr, {grav.data()});
  }

  // Shared bias blocks
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
      problem.AddResidualBlock(make_prior<Factor_GenericPrior>(q[0].data(), p[0].data(), bg.data(), ba.data()), nullptr,
                               {q[0].data(), p[0].data(), bg.data(), ba.data()});
    if (i > 0) {
      auto c = s.cpi[i - 1];
      auto *f = new Factor_ImuCPIv1(c->DT, grav, c->alpha_tau, c->beta_tau, c->q_k2tau, c->b_a_lin, c->b_w_lin, c->J_q, c->J_b, c->J_a,
                                    c->H_b, c->H_a, c->P_meas);
      // Use SHARED biases (same bg, ba for all factors)
      problem.AddResidualBlock(f, nullptr,
                               {q[i - 1].data(), bg.data(), v[i - 1].data(), ba.data(), p[i - 1].data(), q[i].data(),
                                bg.data(), v[i].data(), ba.data(), p[i].data(), grav.data()});
    }
  }
  for (int j = 0; j < s.M; ++j) {
    problem.AddParameterBlock(lm[j].data(), 3);
    problem.SetSchurLandmark(lm[j].data());
  }
  CauchyLoss *loss = new CauchyLoss(1.0);
  for (int i = 0; i < s.N; ++i)
    for (auto &o : s.obs[i])
      problem.AddResidualBlock(new PinholeFactor(o.second, s.fx, s.fy, s.cx, s.cy, s.sigma_px,
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

  Eigen::Vector3d grav_init = grav;
  auto t0 = std::chrono::steady_clock::now();
  SolverSummary sum = problem.Solve(o);

  if (verbose && optimize_gravity) {
    std::printf("  Gravity: init=[%.4f,%.4f,%.4f] |%.4f| -> final=[%.4f,%.4f,%.4f] |%.4f|\n",
                grav_init(0), grav_init(1), grav_init(2), grav_init.norm(),
                grav(0), grav(1), grav(2), grav.norm());
    std::printf("  GT grav: [%.4f,%.4f,%.4f]\n", s.grav(0), s.grav(1), s.grav(2));
  }
  r.time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  r.ok = sum.converged;
  r.iters = sum.iterations;
  r.cost0 = sum.initial_cost;
  r.cost1 = sum.final_cost;
  r.q = q;
  r.p = p;
  r.v = v;
  r.bg = bg;
  r.ba = ba;
  r.grav_final = grav;

  // Compute gravity error
  Eigen::Vector3d grav_gt_norm = s.grav.normalized();
  Eigen::Vector3d grav_est_norm = grav.normalized();
  double dot = grav_gt_norm.dot(grav_est_norm);
  r.grav_err_deg = std::acos(std::min(1.0, std::abs(dot))) * 180.0 / M_PI;

  return r;
}

struct Rmse {
  double rot_deg, pos, vel, bg, ba;
};
static Rmse rmse_vs_gt(const Result &r, const Sim &s) {
  double ro = 0, po = 0, ve = 0;
  for (int i = 0; i < s.N; ++i) {
    const Eigen::Matrix3d Rr = quat_2_Rot(r.q[i]) * quat_2_Rot(s.q[i]).transpose();
    ro += ov_core::log_so3(Rr).squaredNorm();
    po += (r.p[i] - s.p[i]).squaredNorm();
    ve += (r.v[i] - s.v[i]).squaredNorm();
  }
  Rmse m;
  m.rot_deg = std::sqrt(ro / s.N) * 180.0 / M_PI;
  m.pos = std::sqrt(po / s.N);
  m.vel = std::sqrt(ve / s.N);
  // Shared biases - error is just the norm of the difference
  m.bg = (r.bg - s.bg_gt).norm();
  m.ba = (r.ba - s.ba_gt).norm();
  return m;
}

int main(int argc, char **argv) {
  int K = (argc > 1) ? atoi(argv[1]) : 30;
  int threads = (argc > 2) ? atoi(argv[2]) : 1;
  bool dl = (argc > 3) ? (atoi(argv[3]) != 0) : false;

  std::printf("====== S² Gravity MLE Benchmark (zbft_sfm) ======\n");
  std::printf("Trials: %d | Threads: %d | Method: %s\n\n", K, threads, dl ? "DOGLEG" : "LM");

  // Test 1: Gravity fixed (baseline)
  std::printf("--- Test 1: Gravity FIXED at GT (baseline) ---\n");
  {
    double sum_time = 0, sum_rot = 0, sum_pos = 0, sum_vel = 0, sum_bg = 0, sum_ba = 0;
    int ok = 0;
    for (int t = 0; t < K; ++t) {
      std::mt19937 rng(1000 + t);
      Sim s = make_sim(rng);
      InitGuess g = make_init(s, rng, false);
      Result r = run_zbft(s, g, threads, dl, false, t == 0);
      if (r.ok) ok++;
      sum_time += r.time_ms;
      Rmse m = rmse_vs_gt(r, s);
      sum_rot += m.rot_deg; sum_pos += m.pos; sum_vel += m.vel; sum_bg += m.bg; sum_ba += m.ba;
    }
    double iK = 1.0 / K;
    std::printf("  Converged: %d/%d\n", ok, K);
    std::printf("  Time (mean): %.2f ms\n", sum_time * iK);
    std::printf("  RMSE vs GT:  rot=%.4f deg  pos=%.4f m  vel=%.4f m/s  bg=%.5f  ba=%.5f\n",
                sum_rot * iK, sum_pos * iK, sum_vel * iK, sum_bg * iK, sum_ba * iK);
  }

  // Test 2: Gravity optimized on S² (starting from GT)
  std::printf("\n--- Test 2: Gravity OPTIMIZED on S² (init from GT) ---\n");
  {
    double sum_time = 0, sum_rot = 0, sum_pos = 0, sum_vel = 0, sum_bg = 0, sum_ba = 0, sum_grav_err = 0;
    int ok = 0;
    for (int t = 0; t < K; ++t) {
      std::mt19937 rng(1000 + t);
      Sim s = make_sim(rng);
      InitGuess g = make_init(s, rng, false);
      Result r = run_zbft(s, g, threads, dl, true, t == 0);
      if (r.ok) ok++;
      sum_time += r.time_ms;
      sum_grav_err += r.grav_err_deg;
      Rmse m = rmse_vs_gt(r, s);
      sum_rot += m.rot_deg; sum_pos += m.pos; sum_vel += m.vel; sum_bg += m.bg; sum_ba += m.ba;
    }
    double iK = 1.0 / K;
    std::printf("  Converged: %d/%d\n", ok, K);
    std::printf("  Time (mean): %.2f ms\n", sum_time * iK);
    std::printf("  Gravity error (mean): %.4f deg\n", sum_grav_err * iK);
    std::printf("  RMSE vs GT:  rot=%.4f deg  pos=%.4f m  vel=%.4f m/s  bg=%.5f  ba=%.5f\n",
                sum_rot * iK, sum_pos * iK, sum_vel * iK, sum_bg * iK, sum_ba * iK);
  }

  // Test 3: Gravity optimized on S² (starting from perturbed)
  std::printf("\n--- Test 3: Gravity OPTIMIZED on S² (init perturbed 3-15°) ---\n");
  {
    double sum_time = 0, sum_rot = 0, sum_pos = 0, sum_vel = 0, sum_bg = 0, sum_ba = 0;
    double sum_grav_err_init = 0, sum_grav_err_final = 0;
    int ok = 0;
    for (int t = 0; t < K; ++t) {
      std::mt19937 rng(1000 + t);
      Sim s = make_sim(rng);
      InitGuess g = make_init(s, rng, true);

      // Compute initial gravity error
      double init_err = std::acos(std::min(1.0, std::abs(g.grav.normalized().dot(s.grav.normalized())))) * 180.0 / M_PI;
      sum_grav_err_init += init_err;

      Result r = run_zbft(s, g, threads, dl, true, t == 0);
      if (r.ok) ok++;
      sum_time += r.time_ms;
      sum_grav_err_final += r.grav_err_deg;
      Rmse m = rmse_vs_gt(r, s);
      sum_rot += m.rot_deg; sum_pos += m.pos; sum_vel += m.vel; sum_bg += m.bg; sum_ba += m.ba;

      if (t < 5) {
        std::printf("    Trial %d: init_g_err=%.2f° -> final_g_err=%.4f° (cost %.2e -> %.2e)\n",
                    t, init_err, r.grav_err_deg, r.cost0, r.cost1);
      }
    }
    double iK = 1.0 / K;
    std::printf("  Converged: %d/%d\n", ok, K);
    std::printf("  Time (mean): %.2f ms\n", sum_time * iK);
    std::printf("  Gravity error: init=%.2f° -> final=%.4f° (%.1fx improvement)\n",
                sum_grav_err_init * iK, sum_grav_err_final * iK, sum_grav_err_init / std::max(1e-6, sum_grav_err_final));
    std::printf("  RMSE vs GT:  rot=%.4f deg  pos=%.4f m  vel=%.4f m/s  bg=%.5f  ba=%.5f\n",
                sum_rot * iK, sum_pos * iK, sum_vel * iK, sum_bg * iK, sum_ba * iK);
  }

  std::printf("\n====== DONE ======\n");
  return 0;
}
