/*
 * Known-truth recovery oracle for the optional connected-bias calibration
 * solver. These bounds test numerical recovery in a synthetic world; they
 * are not a device-specific Tg accuracy or production acceptance claim.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "sim/SynthWorld.h"
#include "solve/BiasLinkedCalib.h"
#include "utils/NumericChecks.h"

using namespace ov_zcalib;

namespace {

int failures = 0;

void check(bool ok, const char *why) {
  if (!ok) {
    ++failures;
    std::printf("FAIL: %s\n", why);
  }
}

// A single continuous physical trajectory, cut into windows with positive
// gaps. SynthWorld::make_window adds 0.13*t_start to phase; cancel that
// explicitly instead of silently stitching unrelated motion segments.
WindowData truth_seeded_window(const synth::Truth &truth, double start, unsigned rng) {
  constexpr double duration = 1.2;
  constexpr double fps = 15.0;
  WindowData w = synth::make_window(truth, start, duration, fps, 800.0, rng,
                                  0.08, truth.td, -0.13 * start);
  synth::Trajectory trajectory;
  const double first = w.clone_times.front();
  const Eigen::Matrix3d R0 = trajectory.R_of(first);
  const Eigen::Vector3d p0 = trajectory.p_of(first);
  const double h = 1e-5;
  w.has_seeds = true;
  w.seed_bg = truth.bg;
  w.seed_ba = truth.ba;
  w.seed_grav = R0 * truth.g_W;
  for (size_t k = 0; k < w.clone_times.size(); ++k) {
    const double t = w.clone_times[k];
    Eigen::Vector4d q = ov_core::rot_2_quat(trajectory.R_of(t) * R0.transpose());
    Eigen::Vector3d p = R0 * (trajectory.p_of(t) - p0);
    Eigen::Vector3d v = R0 * ((trajectory.p_of(t + h) - trajectory.p_of(t - h)) / (2.0 * h));
    if (k != 0) {
      // Exercise nuisance convergence without making the production
      // initializer's basin part of this solver-level truth oracle.
      const Eigen::Vector3d d(2e-4 * std::sin(t), -3e-4 * std::cos(t), 1e-4 * std::sin(2.0 * t));
      Eigen::Vector4d dq;
      dq << 0.5 * d, 1.0;
      q = ov_core::quat_multiply(dq.normalized(), q);
      p += Eigen::Vector3d(3e-4, -2e-4, 1e-4);
      v += Eigen::Vector3d(-3e-4, 2e-4, 1e-4);
    }
    w.seed_q.push_back(q);
    w.seed_p.push_back(p);
    w.seed_v.push_back(v);
  }

  // Reproduce make_window's first-visible feature ordering. Seeds are only
  // initial guesses: the solver still estimates every landmark and clone.
  const auto cloud = synth::make_cloud(60, rng ^ 0x9e3779b9u);
  std::vector<int> ids(cloud.size(), -1);
  w.seed_feats.resize(w.num_feats);
  size_t next = 0;
  for (double t : w.clone_times) {
    for (size_t f = 0; f < cloud.size(); ++f) {
      Eigen::Vector2d uv;
      if (ids[f] >= 0 || !synth::project(truth, trajectory, t, cloud[f], uv))
        continue;
      ids[f] = static_cast<int>(next);
      if (next < w.seed_feats.size())
        w.seed_feats[next] = R0 * (cloud[f] - p0);
      ++next;
    }
  }
  check(next == w.num_feats, "synthetic feature ordering disagrees with generator");
  return w;
}

SharedCalib blind_tg_calibration(const synth::Truth &truth) {
  SharedCalib c;
  c.imu = truth.imu;
  c.imu.calib_dw = false;
  c.imu.calib_da = false;
  c.imu.calib_RAtoI = false;
  c.imu.calib_tg = true;
  c.imu.Tg.setZero();
  c.tg_enabled = true;
  c.cams[0].q_ItoC = truth.q_ItoC;
  c.cams[0].p_IinC = truth.p_IinC;
  c.cams[0].cam = truth.cam;
  c.cams[0].td = truth.td;
  c.cams[0].img_w = truth.img_w;
  c.cams[0].img_h = truth.img_h;
  c.cams[0].free_ext = false;
  c.cams[0].free_td = false;
  c.cams[0].cam_mode = 0;
  c.bg_prior_sigma = 0.05;
  c.ba_prior_sigma = 0.5;
  return c;
}

JointConfig oracle_config() {
  JointConfig cfg;
  cfg.outer_iterations = 18;
  cfg.window_max_iters = 60;
  cfg.num_threads = 3;
  cfg.verbose = false;
  // Well outside the injected Tg magnitude: recovery must come from data.
  cfg.prior_sigma["tg"] = 5e-3;
  return cfg;
}

// Invert the ENTIRE joint matrix, using an independently equilibrated LLT.
// This checks covariance, not merely whether the report repeated its own
// Schur expression. In particular it detects conditional-Hpp reporting.
Eigen::MatrixXd check_posterior(const JointReport &rep, const BiasLinkedReport &detail, int windows) {
  const int np = 9, dim = np + 12 * windows;
  check(rep.ok && rep.dim_p == np && rep.windows_used == windows, "posterior dimensions/window accounting wrong");
  check(detail.nodes == 2 * windows && detail.gap_links == windows - 1,
        "bias chain has the wrong number of endpoint nodes/gap links");
  check(detail.joint_information.rows() == dim && detail.joint_information.cols() == dim &&
            detail.variable_scales.size() == dim && rep.Lambda.rows() == np && rep.Lambda.cols() == np &&
            rep.sigma.size() == np, "posterior matrix shape wrong");
  if (detail.joint_information.rows() != dim || detail.joint_information.cols() != dim ||
      rep.Lambda.rows() != np || rep.Lambda.cols() != np || rep.sigma.size() != np)
    return Eigen::MatrixXd();
  const Eigen::MatrixXd &H = detail.joint_information;
  check(finite_matrix(H) && finite_matrix(rep.Lambda) && finite_matrix(rep.sigma), "posterior contains nonfinite values");
  check((H.diagonal().array() > 0.0).all() && (rep.sigma.array() > 0.0).all(), "nonpositive posterior diagonal/sigma");
  if (!finite_matrix(H) || (H.diagonal().array() <= 0.0).any()) return Eigen::MatrixXd();
  check((H - H.transpose()).norm() < 1e-12 * H.norm(), "joint information is asymmetric");
  const Eigen::VectorXd scale = H.diagonal().cwiseSqrt().cwiseInverse();
  const Eigen::MatrixXd equilibrated = scale.asDiagonal() * H * scale.asDiagonal();
  Eigen::LLT<Eigen::MatrixXd> llt(equilibrated);
  check(llt.info() == Eigen::Success, "full joint posterior is not positive definite");
  if (llt.info() != Eigen::Success) return Eigen::MatrixXd();
  const Eigen::MatrixXd covariance = scale.asDiagonal() * llt.solve(Eigen::MatrixXd::Identity(dim, dim)) * scale.asDiagonal();
  const Eigen::MatrixXd C = covariance.topLeftCorner(np, np);
  Eigen::LLT<Eigen::MatrixXd> report_llt(rep.Lambda);
  check(report_llt.info() == Eigen::Success, "reported marginal information is not positive definite");
  if (report_llt.info() != Eigen::Success) return Eigen::MatrixXd();
  const Eigen::MatrixXd report_cov = report_llt.solve(Eigen::MatrixXd::Identity(np, np));
  const double covariance_error = (C - report_cov).norm() / C.norm();
  const double sigma_error = (C.diagonal().cwiseSqrt() - rep.sigma).norm() / rep.sigma.norm();
  check(finite_matrix(covariance) && covariance_error < 2e-5 && sigma_error < 2e-5,
        "independent full-joint covariance disagrees with reported bias-marginal posterior");
  check(!detail.accepted_merit.empty() && detail.accepted_merit.size() == (size_t)rep.accepted_passes + 1,
        "accepted merit history missing entries");
  for (size_t k = 0; k < detail.accepted_merit.size(); ++k) {
    check(finite_scalar(detail.accepted_merit[k]), "accepted nonfinite merit");
    if (k) check(detail.accepted_merit[k] < detail.accepted_merit[k - 1], "accepted merit increased or stayed constant");
  }
  if (!detail.accepted_merit.empty())
    check(rep.final_merit == detail.accepted_merit.back(), "reported merit differs from accepted point");
  std::printf("  full-joint covariance relative error %.3e, sigma error %.3e, qn %.3e\n",
              covariance_error, sigma_error, rep.qn_max_final);
  return C;
}

void check_boundaries(const BiasLinkedReport &detail, const std::vector<WindowWarmState> &warm) {
  check(detail.boundary.size() == warm.size(), "boundary output and warm states differ in size");
  if (detail.boundary.size() != warm.size()) return;
  for (size_t i = 0; i < warm.size(); ++i) {
    const auto &w = warm[i];
    const auto &b = detail.boundary[i];
    check(b.imu_weights == nullptr, "returned boundary retained solve-local IMU weight storage");
    check(w.valid && !w.bg.empty() && !w.ba.empty(), "missing accepted local state");
    if (!w.valid || w.bg.empty() || w.ba.empty()) continue;
    check((w.bg.front() - b.bg_first).squaredNorm() == 0.0 &&
              (w.ba.front() - b.ba_first).squaredNorm() == 0.0 &&
              (w.bg.back() - b.bg_last).squaredNorm() == 0.0 &&
              (w.ba.back() - b.ba_last).squaredNorm() == 0.0,
          "accepted local endpoints do not equal accepted outer bias nodes");
  }
}

void recovery_case(bool inject_tg) {
  synth::Truth truth = synth::make_truth();
  if (inject_tg)
    truth.imu.Tg << 3.0e-4, -1.8e-4, 2.2e-4,
                   -2.4e-4, 3.5e-4, 1.2e-4,
                    1.6e-4, -2.8e-4, -3.2e-4;
  else
    truth.imu.Tg.setZero();
  std::vector<WindowData> windows;
  for (int i = 0; i < 6; ++i) {
    windows.push_back(truth_seeded_window(truth, 1.3 * i, 810 + i));
    windows.back().uid = 1 + i;
  }
  // Input order is not temporal order. Node sorting must still connect the
  // chronological endpoints and put output states back in input order.
  std::rotate(windows.begin(), windows.begin() + 2, windows.end());
  // The physical prior is independent and intentionally misses truth; broad
  // widths prevent an exact truth anchor from solving the bias/Tg ambiguity.
  const WindowBiasPrior prior;
  SharedCalib calib = blind_tg_calibration(truth);
  JointConfig cfg = oracle_config();
  JointReport rep;
  BiasLinkedReport detail;
  std::vector<WindowWarmState> warm;
  std::printf("%s Tg oracle\n", inject_tg ? "known nonzero" : "zero control");
  const bool ok = BiasLinkedCalib::solve(windows, calib, cfg, rep, prior, &detail, nullptr, &warm);
  check(ok, inject_tg ? "known-Tg solve failed" : "zero-Tg solve failed");
  if (!ok) return;
  const Eigen::MatrixXd C = check_posterior(rep, detail, windows.size());
  check_boundaries(detail, warm);
  int initial_priors = 0;
  for (size_t i = 0; i < detail.boundary.size(); ++i)
    if (detail.boundary[i].include_first_prior) {
      ++initial_priors;
      check(windows[i].uid == 1, "physical initial prior was attached to input order instead of earliest time");
    }
  check(initial_priors == 1, "physical bias prior must occur exactly once");
  const Eigen::Matrix3d error = calib.imu.Tg - truth.imu.Tg;
  const double max_error = error.cwiseAbs().maxCoeff();
  check(rep.accepted_passes >= 2, "recovery did not take meaningful outer steps");
  check(max_error < 8e-5, inject_tg ? "known-Tg recovery error exceeds 8e-5" : "zero-Tg control invents Tg above 8e-5");
  check(rep.sigma.maxCoeff() < 3e-4, "excited oracle did not meaningfully identify Tg");
  if (inject_tg)
    check(error.norm() < 0.2 * truth.imu.Tg.norm(), "known-Tg estimate did not improve sufficiently from zero");
  if (C.rows() == 9) {
    const Eigen::Map<const Eigen::Matrix<double, 9, 1>> err(error.data());
    const double mahalanobis2 = err.dot(C.ldlt().solve(err));
    check(finite_scalar(mahalanobis2) && mahalanobis2 < 36.0, "truth lies outside broad joint posterior recovery bound");
    std::printf("  max Tg error %.6e, sigma max %.6e, truth Mahalanobis^2 %.3f, passes %d/%d\n",
                max_error, rep.sigma.maxCoeff(), mahalanobis2, rep.accepted_passes, rep.evaluation_passes);
  }

  if (inject_tg) {
    // Invalid graph requests must be rejected before exposing a candidate.
    // Actual clone spans own images, including BOTH endpoints.
    const Eigen::Matrix3d before = calib.imu.Tg;
    auto invalid = windows;
    invalid[1] = invalid[0];
    check(!BiasLinkedCalib::solve(invalid, calib, cfg, rep, prior, &detail) && !rep.ok &&
              (calib.imu.Tg - before).squaredNorm() == 0.0 && detail.joint_information.size() == 0,
          "overlapping clone spans accepted or mutated calibration");
    invalid = windows;
    invalid[1].clone_times.front() = invalid[0].clone_times.back();
    check(!BiasLinkedCalib::solve(invalid, calib, cfg, rep, prior) && !rep.ok &&
              (calib.imu.Tg - before).squaredNorm() == 0.0,
          "touching clone spans accepted or mutated calibration");
    WindowBiasPrior bad_prior = prior;
    bad_prior.bg(0) = std::numeric_limits<double>::quiet_NaN();
    check(!BiasLinkedCalib::solve(windows, calib, cfg, rep, bad_prior) && !rep.ok &&
              (calib.imu.Tg - before).squaredNorm() == 0.0,
          "nonfinite physical prior accepted or mutated calibration");
    JointConfig bad_cfg = cfg;
    bad_cfg.max_wall_s = std::numeric_limits<double>::quiet_NaN();
    check(!BiasLinkedCalib::solve(windows, calib, bad_cfg, rep, prior) && !rep.ok &&
              (calib.imu.Tg - before).squaredNorm() == 0.0,
          "nonfinite wall budget silently disabled the deadline");
    bad_cfg = cfg;
    bad_cfg.max_wall_s = 1.0;
    bad_cfg.budget_clock = []() { return std::numeric_limits<double>::quiet_NaN(); };
    check(!BiasLinkedCalib::solve(windows, calib, bad_cfg, rep, prior) && !rep.ok &&
              (calib.imu.Tg - before).squaredNorm() == 0.0,
          "nonfinite budget clock silently disabled the deadline");
    bad_cfg = cfg;
    bad_cfg.step_cap["tg"] = std::numeric_limits<double>::infinity();
    check(!BiasLinkedCalib::solve(windows, calib, bad_cfg, rep, prior) && !rep.ok &&
              (calib.imu.Tg - before).squaredNorm() == 0.0,
          "nonfinite active step cap accepted or mutated calibration");
    bad_cfg = cfg;
    bad_cfg.max_backtracks = -1;
    check(!BiasLinkedCalib::solve(windows, calib, bad_cfg, rep, prior) && !rep.ok &&
              (calib.imu.Tg - before).squaredNorm() == 0.0,
          "negative backtrack count accepted or mutated calibration");
  }
}

// Fixed attitude and constant acceleration give one constant specific-force
// vector for every sample. Each Tg row can trade exactly against constant bg;
// temporal bias links cannot manufacture the missing excitation.
WindowData constant_force_window(const synth::Truth &truth, double start, unsigned uid) {
  const Eigen::Vector3d velocity(0.10, -0.05, 0.02), acceleration(0.25, -0.18, 0.08);
  const auto position = [&](double t) { return (velocity * t + 0.5 * acceleration * t * t).eval(); };
  const Eigen::Vector3d force = acceleration + truth.g_W;
  const Eigen::Vector3d p0 = position(start);
  std::vector<Eigen::Vector3d> points;
  for (int iy = -2; iy <= 2; ++iy)
    for (int ix = -3; ix <= 3; ++ix)
      points.emplace_back(0.45 * ix, 0.40 * iy, 4.5 + 0.3 * ((ix + iy + 5) % 3));
  WindowData w;
  w.uid = uid;
  w.pix_sigma = 0.1;
  w.has_seeds = true;
  w.seed_bg = truth.bg;
  w.seed_ba = truth.ba;
  w.seed_grav = truth.g_W;
  w.num_feats = points.size();
  for (const auto &p : points) w.seed_feats.push_back(p - p0);
  const Eigen::Matrix3d Ric = ov_core::quat_2_Rot(truth.q_ItoC);
  for (int k = 0; k <= 8; ++k) {
    const double t = start + 0.075 * k;
    w.clone_times.push_back(t);
    w.obs.emplace_back();
    w.seed_q.emplace_back(0.0, 0.0, 0.0, 1.0);
    w.seed_p.push_back(position(t) - p0);
    w.seed_v.push_back(velocity + acceleration * t);
    for (size_t f = 0; f < points.size(); ++f) {
      const Eigen::Vector3d pc = Ric * (points[f] - position(t)) + truth.p_IinC;
      CloneObs observation;
      observation.feat_id = f;
      observation.uv = Eigen::Vector2d(truth.cam(0) * pc(0) / pc(2) + truth.cam(2),
                                       truth.cam(1) * pc(1) / pc(2) + truth.cam(3));
      w.obs.back().push_back(observation);
    }
  }
  const Eigen::Vector3d raw_a = ImuIntrinsicModel::ut(truth.imu.da).inverse() *
      ov_core::quat_2_Rot(truth.imu.q_AtoI).transpose() * force + truth.ba;
  for (int k = -8; k <= 488; ++k) {
    RawImu imu;
    imu.timestamp = start + k / 800.0;
    imu.wm = truth.bg + truth.imu.Tg * force;
    imu.am = raw_a;
    w.imu.push_back(imu);
  }
  return w;
}

void constant_force_case() {
  synth::Truth truth = synth::make_truth();
  truth.td = 0.0;
  truth.q_ItoC = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  truth.p_IinC.setZero();
  truth.bg.setZero();
  truth.ba.setZero();
  truth.imu.Tg.setZero();
  SharedCalib calib = blind_tg_calibration(truth);
  calib.bg_prior_sigma = 1e3;
  JointConfig cfg = oracle_config();
  cfg.outer_iterations = 4;
  cfg.prior_sigma["tg"] = 1e-3;
  const WindowBiasPrior prior;
  std::vector<WindowData> windows;
  for (unsigned i = 0; i < 3; ++i)
    windows.push_back(constant_force_window(truth, 0.7 * i, 20 + i));
  JointReport rep;
  BiasLinkedReport detail;
  std::printf("constant-specific-force degeneracy oracle\n");
  const bool ok = BiasLinkedCalib::solve(windows, calib, cfg, rep, prior, &detail);
  check(ok, "constant-force solve failed");
  if (!ok) return;
  const Eigen::MatrixXd C = check_posterior(rep, detail, windows.size());
  if (C.rows() != 9) return;
  const Eigen::Vector3d force = Eigen::Vector3d(0.25, -0.18, 0.08) + truth.g_W;
  Eigen::VectorXd direction = Eigen::VectorXd::Zero(9);
  for (int col = 0; col < 3; ++col) direction(3 * col) = force(col) / force.norm();
  const double marginal_sigma = std::sqrt(direction.dot(C * direction));
  Eigen::LLT<Eigen::MatrixXd> conditional(detail.joint_information.topLeftCorner(9, 9));
  check(conditional.info() == Eigen::Success, "conditional Tg curvature is not positive definite");
  if (conditional.info() != Eigen::Success) return;
  const double conditional_sigma = std::sqrt(direction.dot(conditional.solve(direction)));
  check(marginal_sigma > 0.9 * cfg.prior_sigma.at("tg"), "constant force falsely identified bias-confounded Tg");
  check(marginal_sigma > 3.0 * conditional_sigma, "degeneracy fixture did not distinguish marginal and conditional Tg uncertainty");
  check(rep.sigma.minCoeff() > 0.9 * cfg.prior_sigma.at("tg"), "constant-force Tg posterior shrank without excitation");
  std::printf("  force-direction Tg sigma: marginal %.6e, conditional %.6e, prior %.6e\n",
              marginal_sigma, conditional_sigma, cfg.prior_sigma.at("tg"));
}

} // namespace

int main() {
  recovery_case(true);
  recovery_case(false);
  constant_force_case();
  std::printf("%s connected-bias calibration (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
