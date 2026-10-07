/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * WindowBA boundary-bias separator oracle: exact legacy cache isolation,
 * conditional endpoint preservation, prior ownership, Schur associativity,
 * and finite differences of the conditional window objective.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "sim/SynthWorld.h"
#include "utils/NumericChecks.h"

using namespace ov_zcalib;

namespace {
int failures = 0;
void check(bool ok, const char *message) {
  if (!ok) {
    ++failures;
    std::printf("FAIL: %s\n", message);
  }
}

template <typename A, typename B> bool same_bits(const A &a, const B &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         std::memcmp(a.data(), b.data(), sizeof(double) * a.size()) == 0;
}
bool same_scalar(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

bool same_state(const WindowWarmState &a, const WindowWarmState &b) {
  if (a.valid != b.valid || a.q.size() != b.q.size() || a.feats.size() != b.feats.size() || !same_bits(a.grav, b.grav))
    return false;
  for (size_t k = 0; k < a.q.size(); ++k)
    if (!same_bits(a.q[k], b.q[k]) || !same_bits(a.bg[k], b.bg[k]) || !same_bits(a.v[k], b.v[k]) ||
        !same_bits(a.ba[k], b.ba[k]) || !same_bits(a.p[k], b.p[k]))
      return false;
  for (size_t f = 0; f < a.feats.size(); ++f)
    if (!same_bits(a.feats[f], b.feats[f]))
      return false;
  return true;
}

bool endpoints_match(const WindowWarmState &state, const WindowBoundaryBias &boundary) {
  return state.valid && same_bits(state.bg.front(), boundary.bg_first) && same_bits(state.ba.front(), boundary.ba_first) &&
         same_bits(state.bg.back(), boundary.bg_last) && same_bits(state.ba.back(), boundary.ba_last);
}

struct Fixture {
  WindowData window;
  SharedCalib calib;
  WindowBiasPrior prior;
};

Fixture make_fixture() {
  Fixture f;
  synth::Truth truth = synth::make_truth();
  truth.td = 0.0; // no temporal transport approximation in the derivative oracle
  truth.imu.Tg << 2e-4, -1e-4, 3e-4, -2e-4, 1e-4, 1e-4, 1e-4, 2e-4, -2e-4;
  synth::Trajectory trajectory;
  trajectory.phase = 0.2;
  const double duration = 0.8;
  const Eigen::Matrix3d R0 = trajectory.R_of(0.0);
  const Eigen::Vector3d p0 = trajectory.p_of(0.0);
  const Eigen::Matrix3d Ric = ov_core::quat_2_Rot(truth.q_ItoC);
  std::vector<Eigen::Vector3d> world_points;
  for (int iy = -2; iy <= 2; ++iy)
    for (int ix = -3; ix <= 3; ++ix) {
      const double depth = 5.0 + 0.3 * ((ix + iy + 5) % 3);
      const Eigen::Vector3d pc(0.18 * ix * depth, 0.15 * iy * depth, depth);
      world_points.push_back(p0 + R0.transpose() * Ric.transpose() * (pc - truth.p_IinC));
    }
  auto &w = f.window;
  w.uid = 1;
  w.pix_sigma = 0.5;
  w.has_seeds = true;
  w.num_feats = world_points.size();
  w.seed_bg = truth.bg;
  w.seed_ba = truth.ba;
  w.seed_grav = R0 * truth.g_W;
  for (const auto &point : world_points)
    w.seed_feats.push_back(R0 * (point - p0));
  std::mt19937 rng(512);
  for (int i = -8; i <= 8 + (int)(duration * 800); ++i)
    w.imu.push_back(synth::raw_imu_at(truth, trajectory, i / 800.0, rng));
  for (int k = 0; k <= 8; ++k) {
    const double t = 0.1 * k;
    w.clone_times.push_back(t);
    w.obs.emplace_back();
    w.seed_q.push_back(ov_core::rot_2_quat(trajectory.R_of(t) * R0.transpose()));
    w.seed_p.push_back(R0 * (trajectory.p_of(t) - p0));
    const double dt = 1e-5;
    w.seed_v.push_back(R0 * (trajectory.p_of(t + dt) - trajectory.p_of(t - dt)) / (2.0 * dt));
    for (size_t id = 0; id < world_points.size(); ++id) {
      Eigen::Vector2d uv;
      if (!synth::project(truth, trajectory, t, world_points[id], uv))
        continue;
      CloneObs observation;
      observation.feat_id = id;
      observation.uv = uv;
      w.obs.back().push_back(observation);
    }
  }
  f.calib.imu = truth.imu;
  f.calib.imu.calib_dw = f.calib.imu.calib_da = f.calib.imu.calib_RAtoI = false;
  f.calib.imu.calib_tg = f.calib.tg_enabled = true;
  f.calib.imu.Tg *= 0.8;
  f.calib.cams[0].q_ItoC = truth.q_ItoC;
  f.calib.cams[0].p_IinC = truth.p_IinC;
  f.calib.cams[0].cam = truth.cam;
  f.calib.cams[0].free_ext = f.calib.cams[0].free_td = false;
  f.prior = WindowBiasPrior{truth.bg, truth.ba};
  return f;
}

WindowBoundaryBias boundary_of(const WindowWarmState &state) {
  WindowBoundaryBias boundary;
  boundary.bg_first = state.bg.front();
  boundary.ba_first = state.ba.front();
  boundary.bg_last = state.bg.back();
  boundary.ba_last = state.ba.back();
  return boundary;
}

double &coordinate(WindowBoundaryBias &boundary, int i) {
  if (i < 3) return boundary.bg_first(i);
  if (i < 6) return boundary.ba_first(i - 3);
  if (i < 9) return boundary.bg_last(i - 6);
  return boundary.ba_last(i - 9);
}

void run_tests() {
  Fixture f = make_fixture();
  const int np = f.calib.local_dim();
  WindowPreint pc;
  WindowWarmState fixed;
  WindowSolveReport initialized;
  check(WindowBA::solve_and_export(f.window, f.calib, true, initialized, 60, false, &fixed, &pc,
                                   nullptr, nullptr, &f.prior), "legacy fixture solve failed");
  if (!initialized.ok || !fixed.valid) return;
  const void *legacy_graph = pc.graph.get();
  WindowBoundaryBias boundary = boundary_of(fixed);
  auto export_at = [&](const WindowBoundaryBias *b, WindowSolveReport &report, WindowPreint *cache = nullptr) {
    WindowWarmState warm = fixed;
    return WindowBA::solve_and_export(f.window, f.calib, true, report, 0, false, &warm, cache,
                                      &fixed, nullptr, &f.prior, b);
  };

  // Every eliminated block and every kept cross term comes from the real
  // WindowBA. Eliminating the exported endpoints densely must reproduce the
  // independent-window export at the identical (possibly nonstationary) point.
  WindowSolveReport legacy, separated;
  check(export_at(nullptr, legacy, &pc), "legacy fixed-point export failed");
  check(export_at(&boundary, separated, &pc), "separated export failed");
  if (!legacy.ok || !separated.ok) return;
  check(separated.free_dim == np + 12 && separated.Lambda.rows() == np + 12 && separated.gred.size() == np + 12,
        "separator dimension must append exactly 12 coordinates");
  check(separated.preint_hit && pc.graph.get() == legacy_graph, "boundary call lost preintegration reuse or replaced legacy graph");
  check(separated.Lambda.topRightCorner(np, 12).norm() > 1.0, "Tg/bias cross information was discarded");
  const Eigen::MatrixXd Hbb = separated.Lambda.bottomRightCorner(12, 12);
  Eigen::LDLT<Eigen::MatrixXd> ldlt(Hbb);
  check(ldlt.info() == Eigen::Success && ldlt.vectorD().minCoeff() > 0.0, "boundary Hessian cannot be marginalized");
  if (ldlt.info() != Eigen::Success) return;
  const Eigen::MatrixXd Hpb = separated.Lambda.topRightCorner(np, 12);
  const Eigen::MatrixXd dense_L = separated.Lambda.topLeftCorner(np, np) - Hpb * ldlt.solve(Hpb.transpose());
  const Eigen::VectorXd dense_g = separated.gred.head(np) - Hpb * ldlt.solve(separated.gred.tail(12));
  const double schur_L_error = (dense_L - legacy.Lambda).norm() / std::max(1.0, legacy.Lambda.norm());
  const double schur_g_error = (dense_g - legacy.gred).norm() / std::max(1.0, legacy.gred.norm());
  check(schur_L_error < 2e-6 && schur_g_error < 2e-6, "dense endpoint elimination disagrees with legacy Schur export");

  // The single physical anchor is evidence only on the FIRST kept biases.
  // Turning it off must leave pose anchors, eliminated states and cross terms
  // unchanged; this also pins the calibration-first boundary ordering.
  boundary.include_first_prior = false;
  WindowSolveReport unanchored;
  check(export_at(&boundary, unanchored, &pc), "prior-free boundary export failed");
  if (!unanchored.ok) return;
  Eigen::MatrixXd prior_L = Eigen::MatrixXd::Zero(np + 12, np + 12);
  Eigen::VectorXd prior_g = Eigen::VectorXd::Zero(np + 12);
  const double wg = 1.0 / (f.calib.bg_prior_sigma * f.calib.bg_prior_sigma);
  const double wa = 1.0 / (f.calib.ba_prior_sigma * f.calib.ba_prior_sigma);
  prior_L.block<3, 3>(np, np).diagonal().setConstant(wg);
  prior_L.block<3, 3>(np + 3, np + 3).diagonal().setConstant(wa);
  prior_g.segment<3>(np) = wg * (boundary.bg_first - f.prior.bg);
  prior_g.segment<3>(np + 3) = wa * (boundary.ba_first - f.prior.ba);
  check((separated.Lambda - unanchored.Lambda - prior_L).norm() < 5e-7 * prior_L.norm(),
        "first-prior toggle changed information outside the first-bias blocks");
  check((separated.gred - unanchored.gred - prior_g).norm() < 5e-7 * std::max(1.0, prior_g.norm()),
        "first-prior toggle changed gradient outside the first-bias blocks");
  check(same_scalar(separated.qn, unanchored.qn), "first-bias anchor changed the conditional nuisance certificate");

  // A legacy caller may reuse its persistent graph after arbitrarily many
  // boundary exports. Neither constancy nor suppressed prior weights may leak.
  WindowWarmState before = fixed, after = fixed;
  WindowSolveReport before_report, after_report;
  check(WindowBA::solve_and_export(f.window, f.calib, true, before_report, 1, false, &before, &pc,
                                   nullptr, nullptr, &f.prior), "legacy baseline step failed");
  check(export_at(&boundary, unanchored, &pc), "interleaved boundary call failed");
  check(WindowBA::solve_and_export(f.window, f.calib, true, after_report, 1, false, &after, &pc,
                                   nullptr, nullptr, &f.prior, nullptr), "legacy step after boundary call failed");
  check(same_scalar(before_report.cost_final, after_report.cost_final) && same_scalar(before_report.qn, after_report.qn) &&
            same_bits(before_report.Lambda, after_report.Lambda) && same_bits(before_report.gred, after_report.gred) &&
            same_state(before, after) && before_report.free_dim == after_report.free_dim,
        "default-off legacy solve/export changed after boundary call");

  // Boundary training may freeze the full covariance transport separately
  // from its physical bias prior and pose anchors. Start from copied legacy
  // weights, then deliberately change both their values and unrelated
  // held-out metadata so a silently ignored/misrouted context cannot pass.
  WindowEvaluationContext weights;
  check(WindowBA::make_evaluation_context(f.window, f.calib, weights), "fixed IMU weights construction failed");
  weights.imu_sqrt_info = pc.W;
  weights.imu_gravity_fold = pc.Wfold;
  const auto cached_W = pc.W;
  const auto cached_Wfold = pc.Wfold;
  const PreintKey cached_whit_key = pc.whit_key;
  auto unchanged_whiteners = [&]() {
    if (pc.W.size() != cached_W.size() || pc.Wfold.size() != cached_Wfold.size() || !(pc.whit_key == cached_whit_key))
      return false;
    for (size_t i = 0; i < pc.W.size(); ++i)
      if (!same_bits(pc.W[i], cached_W[i]) || !same_bits(pc.Wfold[i], cached_Wfold[i]))
        return false;
    return true;
  };
  WindowBoundaryBias weighted_boundary = boundary_of(fixed);
  weighted_boundary.imu_weights = &weights;
  WindowSolveReport weighted_original;
  check(export_at(&weighted_boundary, weighted_original, &pc), "copied fixed-weight export failed");
  check(same_bits(weighted_original.Lambda, separated.Lambda) && same_bits(weighted_original.gred, separated.gred) &&
            same_scalar(weighted_original.qn, separated.qn), "copied weights changed the boundary objective");
  for (size_t i = 0; i < weights.imu_sqrt_info.size(); ++i) {
    const double scale = i % 2 ? 0.65 : 1.7;
    weights.imu_sqrt_info[i] *= scale;
    weights.imu_gravity_fold[i] *= scale;
  }
  WindowSolveReport weighted, weighted_no_prior;
  check(export_at(&weighted_boundary, weighted, &pc), "custom fixed-weight export failed");
  check((weighted.Lambda - separated.Lambda).norm() > 1e-3 * separated.Lambda.norm(),
        "custom fixed IMU weights did not affect exported information");
  weighted_boundary.include_first_prior = false;
  check(export_at(&weighted_boundary, weighted_no_prior, &pc), "custom-weight prior-free export failed");
  check((weighted.Lambda - weighted_no_prior.Lambda - prior_L).norm() < 5e-7 * prior_L.norm() &&
            (weighted.gred - weighted_no_prior.gred - prior_g).norm() < 5e-7 * std::max(1.0, prior_g.norm()),
        "fixed IMU weights changed ownership or multiplicity of the explicit first-bias prior");
  weighted_boundary.include_first_prior = true;

  auto weighted_step = [&](const WindowBoundaryBias &b, WindowWarmState &state, WindowSolveReport &report) {
    state = fixed;
    return WindowBA::solve_and_export(f.window, f.calib, true, report, 1, false, &state, &pc,
                                      nullptr, nullptr, &f.prior, &b);
  };
  WindowWarmState plain_state, weighted_state, metadata_state;
  WindowSolveReport plain_cost, weighted_cost, metadata_cost;
  check(weighted_step(boundary_of(fixed), plain_state, plain_cost), "plain boundary cost solve failed");
  check(weighted_step(weighted_boundary, weighted_state, weighted_cost), "fixed-weight cost solve failed");
  check(std::abs(weighted_cost.cost_final - plain_cost.cost_final) > 1e-6 * std::max(1.0, plain_cost.cost_final),
        "custom fixed IMU weights did not affect the conditional cost");
  weights.bias_prior.bg += Eigen::Vector3d(.4, -.5, .6);
  weights.bias_prior.ba += Eigen::Vector3d(4, -5, 6);
  weights.p_anchor += Eigen::Vector3d(10, 20, 30);
  weights.q_anchor = ov_core::rot_2_quat(ov_core::exp_so3(Eigen::Vector3d(.1, -.2, .3)));
  check(weighted_step(weighted_boundary, metadata_state, metadata_cost), "fixed-weight metadata isolation solve failed");
  check(same_scalar(weighted_cost.cost_final, metadata_cost.cost_final) && same_state(weighted_state, metadata_state) &&
            same_bits(weighted_cost.Lambda, metadata_cost.Lambda) && same_bits(weighted_cost.gred, metadata_cost.gred),
        "boundary training consumed the fixed-weight context's held-out priors or pose anchors");
  check(pc.has_whit && pc.graph.get() == legacy_graph && unchanged_whiteners(),
        "custom fixed weights contaminated the legacy graph or whitener cache");

  // A changed mean invalidates ordinary cached whiteners; fixed weights must
  // never be published under that new key. Test an entirely empty cache too.
  WindowPreint fixed_only_cache;
  WindowSolveReport fixed_only;
  check(export_at(&weighted_boundary, fixed_only, &fixed_only_cache), "fixed-weight solve with empty cache failed");
  check(fixed_only_cache.has_means && !fixed_only_cache.has_whit && fixed_only_cache.W.empty() &&
            fixed_only_cache.Wfold.empty() && !fixed_only_cache.graph,
        "fixed-weight solve filled the ordinary graph/whitener cache");
  SharedCalib changed_mean = f.calib;
  changed_mean.imu.Tg(0, 0) += 8e-4;
  WindowSolveReport changed_export;
  check(WindowBA::solve_and_export(f.window, changed_mean, true, changed_export, 0, false, nullptr, &pc,
                                   &fixed, nullptr, &f.prior, &weighted_boundary), "fixed-weight changed-mean export failed");
  check(!pc.has_whit && unchanged_whiteners() && pc.graph.get() == legacy_graph,
        "fixed weights were cached as ordinary whiteners for a changed mean");
  WindowWarmState restored = fixed;
  WindowSolveReport restored_report;
  check(WindowBA::solve_and_export(f.window, f.calib, true, restored_report, 1, false, &restored, &pc,
                                   nullptr, nullptr, &f.prior), "legacy solve after fixed-weight cache misses failed");
  check(same_scalar(restored_report.cost_final, before_report.cost_final) && same_scalar(restored_report.qn, before_report.qn) &&
            same_bits(restored_report.Lambda, before_report.Lambda) && same_bits(restored_report.gred, before_report.gred) &&
            same_state(restored, before) && pc.has_whit && unchanged_whiteners(),
        "interleaved fixed-weight calls changed legacy solve/export bytes");

  // An ordinary boundary call may populate W at another calibration while
  // leaving the legacy graph at its old factor key. Returning to that graph
  // must republish ITS weights, not relabel the boundary call's cache bytes.
  WindowBoundaryBias ordinary_boundary = boundary_of(fixed);
  check(WindowBA::solve_and_export(f.window, changed_mean, true, changed_export, 0, false, nullptr, &pc,
                                   &fixed, nullptr, &f.prior, &ordinary_boundary), "ordinary changed-mean boundary export failed");
  WindowSolveReport legacy_restored, boundary_restored;
  check(export_at(nullptr, legacy_restored, &pc), "legacy return after changed-mean boundary export failed");
  check(pc.has_whit && unchanged_whiteners(), "legacy graph relabeled stale boundary whiteners with its own key");
  check(export_at(&ordinary_boundary, boundary_restored, &pc), "boundary export after legacy cache repair failed");
  check(same_bits(boundary_restored.Lambda, separated.Lambda) && same_bits(boundary_restored.gred, separated.gred),
        "boundary calibration round trip changed the original information");

  WindowEvaluationContext invalid_weights;
  WindowBoundaryBias invalid_weight_boundary = weighted_boundary;
  invalid_weight_boundary.imu_weights = &invalid_weights;
  auto reject_weights = [&](const char *message) {
    WindowSolveReport report;
    report.ok = true;
    check(!export_at(&invalid_weight_boundary, report, &pc) && !report.ok, message);
    check(pc.has_whit && unchanged_whiteners() && pc.graph.get() == legacy_graph,
          "invalid fixed weights mutated the legacy cache");
  };
  invalid_weights = weights;
  invalid_weights.imu_sqrt_info.pop_back();
  reject_weights("short fixed IMU weight vector accepted");
  invalid_weights = weights;
  invalid_weights.imu_gravity_fold.pop_back();
  reject_weights("short fixed gravity-fold vector accepted");
  invalid_weights = weights;
  invalid_weights.imu_sqrt_info[0](1, 2) = std::numeric_limits<double>::quiet_NaN();
  reject_weights("NaN fixed IMU weight accepted under fast-math");
  invalid_weights = weights;
  invalid_weights.imu_gravity_fold[0](2, 1) = std::numeric_limits<double>::infinity();
  reject_weights("infinite fixed gravity-fold weight accepted under fast-math");

  // Test supplied boundaries that disagree with BOTH warm and state_at.
  boundary.include_first_prior = true;
  boundary.bg_first += Eigen::Vector3d(8e-5, -6e-5, 4e-5);
  boundary.bg_last += Eigen::Vector3d(-4e-5, 5e-5, -7e-5);
  boundary.ba_first += Eigen::Vector3d(.002, -.001, .002);
  boundary.ba_last += Eigen::Vector3d(-.001, .002, -.001);
  WindowWarmState conditioned = fixed;
  WindowSolveReport conditional;
  check(WindowBA::solve_and_export(f.window, f.calib, true, conditional, 60, false, &conditioned, &pc,
                                   &fixed, nullptr, &f.prior, &boundary), "conditional inner solve failed");
  check(endpoints_match(conditioned, boundary), "inner solve mutated prescribed endpoint values");
  WindowSolveReport deferred;
  WindowWarmState overwritten = fixed;
  check(WindowBA::solve_and_export(f.window, f.calib, true, deferred, 0, false, &overwritten, &pc,
                                   &conditioned, nullptr, &f.prior, &boundary), "conditional deferred export failed");
  check(endpoints_match(overwritten, boundary) && same_bits(conditional.Lambda, deferred.Lambda) &&
            same_bits(conditional.gred, deferred.gred) && same_scalar(conditional.qn, deferred.qn),
        "deferred export changed prescribed endpoints or accepted information");

  // The immutable physical prior is independent of subsequent initializer
  // means, even while its endpoint is an explicit outer variable.
  WindowData reseeded = f.window;
  reseeded.seed_bg += Eigen::Vector3d(.1, -.2, .3);
  reseeded.seed_ba += Eigen::Vector3d(.2, -.3, .4);
  WindowSolveReport reseeded_report;
  check(WindowBA::solve_and_export(reseeded, f.calib, true, reseeded_report, 0, false, nullptr, &pc,
                                   &conditioned, nullptr, &f.prior, &boundary), "reseeded boundary export failed");
  check(same_bits(reseeded_report.Lambda, deferred.Lambda) && same_bits(reseeded_report.gred, deferred.gred),
        "boundary candidate recentered its immutable physical prior");

  // Envelope derivative: reoptimize the free nuisances at +/- each endpoint
  // coordinate, and difference the REAL robust cost. No hand-built factor
  // or implementation-copy Jacobian participates in this check.
  double max_derivative_error = 0.0;
  for (int k = 0; k < 12; ++k) {
    const double eps = (k % 6 < 3) ? 1e-6 : 1e-5;
    double cost[2];
    for (int sign = 0; sign < 2; ++sign) {
      WindowBoundaryBias trial = boundary;
      coordinate(trial, k) += sign ? eps : -eps;
      WindowWarmState warm = conditioned;
      WindowSolveReport report;
      check(WindowBA::solve_and_export(f.window, f.calib, false, report, 80, false, &warm, &pc,
                                       nullptr, nullptr, &f.prior, &trial), "finite-difference conditional solve failed");
      check(endpoints_match(warm, trial) && report.free_dim == np + 12,
            "eval-only boundary call changed endpoints or report dimension");
      cost[sign] = report.cost_final;
    }
    const double fd = (cost[1] - cost[0]) / (2.0 * eps);
    const double analytic = conditional.gred(np + k);
    const double relative = std::abs(fd - analytic) / std::max(1.0, std::max(std::abs(fd), std::abs(analytic)));
    max_derivative_error = std::max(max_derivative_error, relative);
  }
  check(max_derivative_error < 2e-3, "conditional reduced bias gradient failed finite differences");

  WindowSolveReport rejected;
  rejected.ok = true;
  WindowBoundaryBias invalid = boundary;
  invalid.ba_last(1) = std::numeric_limits<double>::quiet_NaN();
  check(!WindowBA::solve_and_export(f.window, f.calib, true, rejected, 0, false, nullptr, &pc,
                                    nullptr, nullptr, &f.prior, &invalid) && !rejected.ok,
        "NaN boundary accepted under fast-math");
  invalid = boundary;
  invalid.bg_first(0) = std::numeric_limits<double>::infinity();
  check(!WindowBA::solve_and_export(f.window, f.calib, true, rejected, 0, false, nullptr, &pc,
                                    nullptr, nullptr, &f.prior, &invalid), "infinite boundary accepted");
  WindowEvaluationContext evaluation;
  check(!WindowBA::solve_and_export(f.window, f.calib, true, rejected, 0, false, nullptr, &pc,
                                    nullptr, &evaluation, &f.prior, &boundary), "unsupported evaluation+boundary combination accepted");
  WindowData too_short = f.window;
  too_short.clone_times.resize(2);
  check(!WindowBA::solve_and_export(too_short, f.calib, true, rejected, 0, false, nullptr, &pc,
                                    nullptr, nullptr, &f.prior, &boundary), "two-clone boundary window accepted");
  std::printf("Schur relative errors Lambda=%.3e g=%.3e; conditional gradient relative error=%.3e\n",
              schur_L_error, schur_g_error, max_derivative_error);
}
} // namespace

int main() {
  run_tests();
  std::printf("%s window boundary-bias separator (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
