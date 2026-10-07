/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/print.h"

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char *message) {
  ++checks;
  if (!ok) { ++failures; std::printf("FAIL: %s\n", message); }
}
template <class A, class B> bool same(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
      std::memcmp(a.derived().data(), b.derived().data(), sizeof(double) * a.size()) == 0;
}
using Poses = std::map<double, std::shared_ptr<ov_type::PoseJPL>>;
std::shared_ptr<ov_msckf::State> make_state(int cap = 11) {
  ov_msckf::StateOptions options;
  options.num_cameras = 2;
  options.max_clone_size = cap;
  options.do_calib_camera_timeoffset = true;
  return std::make_shared<ov_msckf::State>(options);
}
Poses poses() {
  Poses result;
  for (int i = 0; i < 2; ++i) {
    auto pose = std::make_shared<ov_type::PoseJPL>();
    Eigen::Matrix<double, 7, 1> value;
    value << 0, 0, 0, 1, -.1 * (1 - i), .04 * (1 - i), 0;
    pose->set_value(value); pose->set_fej(value);
    result.emplace(9.875 + .125 * i, pose);
  }
  return result;
}
Eigen::MatrixXd joint_covariance() {
  // A genuinely singular joint prior: the newest clone equals the IMU pose,
  // while the older clone has distinct correlated uncertainty.
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(27, 21);
  for (int row = 0; row < 21; ++row)
    for (int col = 0; col < 21; ++col)
      L(row, col) = .001 * std::sin(.2 + .7 * row + .3 * col) + (row == col ? .02 + .001 * row : 0.);
  L.bottomRows(6) = L.topRows(6).eval();
  return L * L.transpose();
}
void covariance_handoff() {
  using ov_msckf::StateHelper;
  auto state = make_state(); auto window = poses(); const auto input = joint_covariance();
  const auto prior = StateHelper::get_full_covariance(state);
  const int old_size = prior.rows();
  Eigen::MatrixXd select = Eigen::MatrixXd::Zero(old_size + 12, 27);
  select.topLeftCorner(15, 15).setIdentity();
  select.block(old_size, 15, 12, 12).setIdentity();
  Eigen::MatrixXd oracle = select * input * select.transpose();
  oracle.block(15, 15, old_size - 15, old_size - 15) = prior.bottomRightCorner(old_size - 15, old_size - 15);
  check(StateHelper::set_initial_state_warmstart(state, input, window),
        "valid singular joint IMU/clone covariance is accepted");
  const auto actual = StateHelper::get_full_covariance(state);
  check(actual.rows() == oracle.rows() && (actual - oracle).cwiseAbs().maxCoeff() < 1e-17,
        "warm injection matches an independent full covariance embedding including all cross terms");
  check(state->_clones_IMU.size() == 2 && state->_clones_kinematics.size() == 2 &&
        state->_clones_IMU.begin()->second->id() == old_size && state->_clones_IMU.rbegin()->second->id() == old_size + 6,
        "warm ownership and covariance IDs match ascending clone order");
  const auto saved = StateHelper::get_full_covariance(state);
  check(!StateHelper::set_initial_state_warmstart(state, input, poses()) &&
        same(saved, StateHelper::get_full_covariance(state)), "repeated warm injection cannot duplicate a live window");

  const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
  for (int bad = 0; bad < 12; ++bad) {
    auto target = make_state(bad == 10 ? 1 : 11); auto candidates = poses();
    Eigen::MatrixXd cov = input;
    if (bad == 0) cov(0, 16) = nan;
    if (bad == 1) cov(16, 0) = nan;
    if (bad == 2) cov(0, 0) = inf;
    if (bad == 3) cov(1, 2) = -inf;
    if (bad == 4) { cov.setIdentity(); cov(0, 1) = cov(1, 0) = 2.; }
    if (bad == 5) cov(0, 1) += .01;
    if (bad == 6) candidates.rbegin()->second = candidates.begin()->second;
    if (bad == 7) candidates.begin()->second->set_local_id(0);
    if (bad == 8) {
      auto value = candidates.begin()->second->value(); value(4) = nan;
      candidates.begin()->second->set_value(value);
    }
    if (bad == 9) {
      auto pose = candidates.rbegin()->second; candidates.erase(std::prev(candidates.end()));
      candidates.emplace(inf, pose);
    }
    if (bad == 11) cov = cov.topLeftCorner(26, 26).eval();
    const auto before = StateHelper::get_full_covariance(target);
    const auto imu_before = target->_imu->value();
    std::vector<int> input_ids; for (const auto &entry : candidates) input_ids.push_back(entry.second->id());
    check(!StateHelper::set_initial_state_warmstart(target, cov, candidates),
          "malformed warm covariance/ownership is refused under production fast-math");
    bool ids_same = true; size_t i = 0;
    for (const auto &entry : candidates) ids_same &= entry.second->id() == input_ids.at(i++);
    check(ids_same && target->_clones_IMU.empty() && target->_clones_kinematics.empty() &&
          same(before, StateHelper::get_full_covariance(target)) && same(imu_before, target->_imu->value()),
          "refused warm injection leaves covariance, means and input ownership unchanged");
  }
}
void exact_measurement_consumption() {
  ov_core::FeatureDatabase db;
  for (size_t id : {10, 20, 30})
    for (size_t camera : {0, 1})
      for (double t : {1., 1.125, 1.25, 1.5})
        db.update_feature(id, t, camera, float(id + t), float(camera + t), float(t), float(-t));
  // A later tracker append to an assimilated feature is independent and must survive.
  db.update_feature(10, 2., 1, 501, 502, 503, 504);
  const auto before = db.clone_features();
  const size_t count = db.cleanup_measurements_exact_for_features({10, 20, 999}, {1., 1.25});
  check(count == 8, "only accepted feature IDs at selected pose times are consumed");
  for (const auto &entry : before) {
    const auto after = db.get_feature(entry.first);
    check(bool(after), "feature remains when unused or future rows exist");
    if (!after) continue;
    for (const auto &camera : entry.second->timestamps) {
      size_t retained = 0;
      for (size_t i = 0; i < camera.second.size(); ++i) {
        const double t = camera.second[i];
        if (entry.first != 30 && (t == 1. || t == 1.25)) continue;
        check(after->timestamps.at(camera.first).at(retained) == t &&
              after->uvs.at(camera.first).at(retained) == entry.second->uvs.at(camera.first).at(i) &&
              after->uvs_norm.at(camera.first).at(retained) == entry.second->uvs_norm.at(camera.first).at(i),
              "raw timestamp, pixel and normalized-pixel arrays stay aligned and exact");
        ++retained;
      }
      check(after->timestamps.at(camera.first).size() == retained, "assimilated pixels cannot be reused by a later updater");
    }
  }
  check(db.cleanup_measurements_exact_for_features({10, 20}, {1., 1.25}) == 0, "consumption is idempotent");
  db.update_feature(99, 3., 0, 1, 2, 3, 4);
  check(db.cleanup_measurements_exact_for_features({99, 99}, {3.}) == 1 && !db.get_feature(99),
        "fully consumed tracks are removed, including duplicate input IDs without double counting");
}
void independent_camera_loss() {
  ov_core::FeatureDatabase db;
  const auto add = [&](size_t id, size_t camera, double time) { db.update_feature(id, time, camera, 1, 2, 3, 4); };
  add(10, 0, 100.); add(10, 1, 90.);      // lost in 0, still alive in 1
  add(20, 0, 100.); add(20, 1, 89.875);  // lost in both
  add(30, 1, 89.875);                    // belongs only to the other update
  add(40, 0, 100.25);                    // a newer async tracking append
  const auto ids = [](const std::vector<std::shared_ptr<ov_core::Feature>> &features) {
    std::vector<size_t> result;
    for (const auto &feature : features) result.push_back(feature->featid);
    std::sort(result.begin(), result.end());
    return result;
  };
  const std::vector<double> clocks{100.125, 90.};
  check(ids(db.features_lost_after_camera_updates(clocks, {0})) == std::vector<size_t>{20},
        "loss on one raw camera clock cannot consume a shared track still alive in another");
  check(ids(db.features_lost_after_camera_updates(clocks, {1})) == std::vector<size_t>({20, 30}),
        "truly lost tracks are detected despite a different camera's numerically larger raw time");
  const auto old = ids(db.features_not_containing_newer(clocks[0]));
  check(std::find(old.begin(), old.end(), 10) != old.end(),
        "old global-clock predicate reproduces premature shared-track consumption");
  db.get_feature(20)->to_delete = true;
  check(ids(db.features_lost_after_camera_updates(clocks, {0, 1}, false, true)) == std::vector<size_t>{30},
        "owner-aware loss preserves skip-deleted semantics");
  check(ids(db.features_lost_after_camera_updates(clocks, {1}, true, true)) == std::vector<size_t>{30} && !db.get_feature(30),
        "owner-aware removal removes only the selected expired track");
}
} // namespace
int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  covariance_handoff(); exact_measurement_consumption(); independent_camera_loss();
  std::printf("WARMSTART_HANDOFF %s checks=%d failures=%d\n", failures ? "FAIL" : "PASS", checks, failures);
  return failures ? 1 : 0;
}
