/* Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "init/InitializerCameraClock.h"
#include "utils/helper.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
using namespace ov_init;
int failures = 0;
void check(bool ok, const char *name) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
  failures += !ok;
}
bool same_double(double a, double b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

std::shared_ptr<ov_core::Feature> feature(size_t id) {
  auto result = std::make_shared<ov_core::Feature>();
  result->featid = id;
  result->to_delete = false;
  result->anchor_clone_timestamp = -1.0;
  result->p_FinA.setZero();
  result->p_FinG.setZero();
  return result;
}
void append(ov_core::Feature &f, size_t camera, double raw, const Eigen::Vector2f &uv) {
  f.timestamps[camera].push_back(raw);
  f.uvs[camera].push_back(300.0f * uv + Eigen::Vector2f(320, 240));
  f.uvs_norm[camera].push_back(uv);
}
InitializerFeatureMap copy(const InitializerFeatureMap &source) {
  InitializerFeatureMap result;
  for (const auto &entry : source)
    result.emplace(entry.first, std::make_shared<ov_core::Feature>(*entry.second));
  return result;
}
bool same_measurements(const InitializerFeatureMap &a, const InitializerFeatureMap &b) {
  if (a.size() != b.size()) return false;
  for (const auto &entry : a) {
    if (!b.count(entry.first)) return false;
    const auto &left = *entry.second;
    const auto &right = *b.at(entry.first);
    if (left.featid != right.featid || left.timestamps.size() != right.timestamps.size()) return false;
    for (const auto &camera : left.timestamps) {
      if (!right.timestamps.count(camera.first)) return false;
      const auto &times = right.timestamps.at(camera.first);
      if (camera.second.size() != times.size()) return false;
      for (size_t i = 0; i < times.size(); ++i) {
        if (!same_double(camera.second[i], times[i]) ||
            left.uvs.at(camera.first)[i] != right.uvs.at(camera.first)[i] ||
            left.uvs_norm.at(camera.first)[i] != right.uvs_norm.at(camera.first)[i]) return false;
      }
    }
  }
  return true;
}

Eigen::Vector2f moving_projection(double imu_time) {
  const double angle = 0.3 * imu_time;
  Eigen::Matrix3d rotation;
  rotation << std::cos(angle), -std::sin(angle), 0, std::sin(angle), std::cos(angle), 0, 0, 0, 1;
  const Eigen::Vector3d point = rotation * (Eigen::Vector3d(1, 2, 5) - Eigen::Vector3d(0.2, -0.1, 0) * imu_time);
  return (point.head<2>() / point.z()).cast<float>();
}

void physical_clock_check() {
  constexpr double reference_td = 0.0625;
  const std::map<size_t, double> offsets{{0, reference_td}, {1, -0.125}, {2, 0.1875}};
  InitializerCameraClock clock;
  check(clock.configure(offsets, reference_td) && clock.unequal_offsets, "per-camera clock configuration");
  InitializerFeatureMap original;
  original.emplace(71, feature(71));
  for (const auto &camera : offsets)
    for (double physical_time : {1.0, 2.0, 3.0})
      append(*original.at(71), camera.first, physical_time - camera.second, moving_projection(physical_time));
  const auto untouched = copy(original);
  auto local = copy(original);
  InitializerFeatureTimeBounds bounds;
  check(retime_initializer_features(local, clock, bounds), "private feature conversion succeeds");
  double model_error = 0, old_error = 0;
  bool exact_time = true, coordinates_unchanged = true;
  for (const auto &camera : offsets) {
    for (size_t i = 0; i < 3; ++i) {
      const double raw = original.at(71)->timestamps.at(camera.first)[i];
      const double ref = local.at(71)->timestamps.at(camera.first)[i];
      exact_time = exact_time && ref + reference_td == raw + camera.second;
      const auto &uv = local.at(71)->uvs_norm.at(camera.first)[i];
      model_error += (moving_projection(ref + reference_td) - uv).squaredNorm();
      old_error += (moving_projection(raw + reference_td) - uv).squaredNorm();
      coordinates_unchanged = coordinates_unchanged && uv == original.at(71)->uvs_norm.at(camera.first)[i] &&
          local.at(71)->uvs.at(camera.first)[i] == original.at(71)->uvs.at(camera.first)[i];
    }
  }
  check(exact_time && model_error == 0.0, "all cameras project at the exact physical IMU time");
  check(old_error > 1e-4, "negative control detects adding reference td to every raw camera stamp");
  check(coordinates_unchanged && local.at(71)->featid == 71, "feature identities and distorted/normalized coordinates preserved");
  check(same_measurements(original, untouched), "shared source measurements remain byte-identical");
  check(bounds.observation_count == 9 && bounds.oldest_ref_time == 0.9375 && bounds.newest_ref_time == 2.9375,
        "corrected observation bounds describe the common reference clock");
  check(clock.to_reference(99, 4.0) == 4.0, "missing camera entry uses the reference offset");
}

void identity_and_horizon_checks() {
  InitializerCameraClock equal, legacy;
  check(equal.configure({{0, 0.125}, {1, 0.125}}, 0.125) && !equal.unequal_offsets &&
        legacy.configure({}, 0.125) && !legacy.unequal_offsets, "equal and absent camera maps retain legacy clock");
  InitializerFeatureMap local;
  local.emplace(5, feature(5));
  for (double t : {-0.0, 0.0, 1.23456789012345}) append(*local.at(5), 1, t, Eigen::Vector2f(0.1f, -0.2f));
  const auto before = copy(local);
  const double *storage = local.at(5)->timestamps.at(1).data();
  InitializerFeatureTimeBounds bounds;
  check(retime_initializer_features(local, equal, bounds) && same_measurements(local, before) &&
        local.at(5)->timestamps.at(1).data() == storage, "equal-offset path preserves timestamp bits and storage");
  check(retime_initializer_features(local, legacy, bounds) && same_measurements(local, before), "empty map exact legacy parity");

  InitializerCameraClock skew;
  check(skew.configure({{0, 0.125}, {1, 0.375}}, 0.125), "horizon fixture clock");
  InitializerFeatureMap late;
  late.emplace(9, feature(9));
  for (size_t cam : {0u, 1u})
    for (double raw : {2.0, 2.5, 3.0}) append(*late.at(9), cam, raw, Eigen::Vector2f(float(raw), float(cam)));
  // A positive offset can move the newest observation beyond the received raw
  // timeline. The final state must also have an IMU sample at/after its time.
  constexpr double newest_raw = 3.0, newest_imu = 3.0;
  const double horizon = std::min(newest_raw, newest_imu - skew.reference_td);
  check(retime_initializer_features(late, skew, bounds, horizon), "future reference-time observations are clipped");
  check(bounds.newest_ref_time == 2.75 && bounds.newest_ref_time <= newest_raw &&
        bounds.newest_ref_time + skew.reference_td <= newest_imu && bounds.observation_count == 4,
        "newest returned reference time has received camera coverage and an IMU bracket");
  check(late.at(9)->timestamps.at(1) == std::vector<double>({2.25, 2.75}) &&
        late.at(9)->uvs_norm.at(1).size() == 2 && late.at(9)->uvs_norm.at(1)[1] == Eigen::Vector2f(2.5f, 1.0f),
        "clipping keeps timestamp/pixel associations and camera order");
  check(retime_initializer_features(late, legacy, bounds, 2.25) && late.at(9)->timestamps.at(1).back() == 2.25,
        "an observation exactly at the coverage horizon is retained");
}

void invalid_checks() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const double large = std::numeric_limits<double>::max();
  InitializerCameraClock clock;
  for (double bad : {nan, inf, -inf}) {
    check(!clock.configure({{0, bad}}, 0.0) && !clock.valid, "nonfinite camera offset rejected under fast-math");
    check(!clock.configure({}, bad) && !clock.valid, "nonfinite reference offset rejected under fast-math");
  }
  check(!clock.configure({{0, large}}, -large), "overflowing offset difference rejected");
  check(!clock.configure({{0, -large}, {1, large}}, 0.0), "overflowing offset span rejected");
  check(clock.configure({{1, 0.25}}, 0.0), "valid clock can be configured after failure");
  InitializerFeatureMap original;
  original.emplace(3, feature(3));
  append(*original.at(3), 1, 1.0, Eigen::Vector2f::Zero());
  for (double bad : {nan, inf, -inf}) {
    auto local = copy(original);
    local.at(3)->timestamps.at(1).push_back(bad);
    local.at(3)->uvs.at(1).push_back(Eigen::Vector2f::Zero());
    local.at(3)->uvs_norm.at(1).push_back(Eigen::Vector2f::Zero());
    auto before = copy(local);
    InitializerFeatureTimeBounds bounds;
    bounds.observation_count = 123;
    check(!retime_initializer_features(local, clock, bounds) && same_measurements(local, before) && bounds.observation_count == 123,
          "invalid timestamp fails transactionally without mutating any output");
  }
  for (double bad : {nan, -inf}) {
    auto local = copy(original);
    InitializerFeatureTimeBounds bounds;
    check(!retime_initializer_features(local, clock, bounds, bad) && same_measurements(local, original), "invalid horizon rejected");
  }
  auto mismatched = copy(original);
  mismatched.at(3)->uvs_norm.at(1).clear();
  InitializerFeatureTimeBounds bounds;
  check(!retime_initializer_features(mismatched, clock, bounds) && mismatched.at(3)->timestamps.at(1)[0] == 1.0,
        "mismatched timestamp/pixel arrays rejected before mutation");
  InitializerFeatureMap null_feature{{1, nullptr}};
  check(!retime_initializer_features(null_feature, clock, bounds), "null feature rejected");
  check(clock.configure({{1, large}}, 0.0), "large finite shift configuration");
  auto overflow = copy(original);
  overflow.at(3)->timestamps.at(1)[0] = large;
  check(!retime_initializer_features(overflow, clock, bounds) && overflow.at(3)->timestamps.at(1)[0] == large,
        "overflowing corrected time rejected before mutation");
}

ov_core::ImuData linear_imu(double time) {
  ov_core::ImuData data;
  data.timestamp = time;
  data.wm = Eigen::Vector3d(1 + 2*time, 3 - time, 4 + 0.5*time);
  data.am = Eigen::Vector3d(3 - time, 2 + time, 5 - 2*time);
  return data;
}

void imu_interval_checks() {
  using Samples = std::vector<ov_core::ImuData>;
  Samples source;
  for (double t : {0.0, 0.5, 1.0, 1.5, 2.0}) source.push_back(linear_imu(t));
  for (const auto &interval : std::vector<std::pair<double, double>>{
           {0.0, 2.0}, {0.125, 1.875}, {0.125, 0.375}, {1.625, 1.875}, {0.5, 2.0}}) {
    const auto samples = InitializerHelper::select_imu_readings(source, interval.first, interval.second);
    bool exact_bounds = samples.size() >= 2 && same_double(samples.front().timestamp, interval.first) &&
                        same_double(samples.back().timestamp, interval.second);
    Eigen::Vector3d integrated_w = Eigen::Vector3d::Zero(), integrated_a = Eigen::Vector3d::Zero();
    for (size_t i = 1; i < samples.size(); ++i) {
      const double dt = samples[i].timestamp - samples[i-1].timestamp;
      exact_bounds = exact_bounds && dt > 0.0;
      integrated_w += 0.5*dt*(samples[i].wm + samples[i-1].wm);
      integrated_a += 0.5*dt*(samples[i].am + samples[i-1].am);
    }
    const auto first = linear_imu(interval.first), last = linear_imu(interval.second);
    const double duration = interval.second - interval.first;
    check(exact_bounds && (integrated_w - 0.5*duration*(first.wm+last.wm)).norm() < 1e-13 &&
          (integrated_a - 0.5*duration*(first.am+last.am)).norm() < 1e-13,
          "IMU selection covers exact endpoints and the analytic complete interval integral");
  }
  const auto exact = InitializerHelper::select_imu_readings(source, 0.0, 2.0);
  bool exact_samples = exact.size() == source.size();
  for (size_t i = 0; exact_samples && i < source.size(); ++i)
    exact_samples = same_double(exact[i].timestamp, source[i].timestamp) && exact[i].wm == source[i].wm && exact[i].am == source[i].am;
  check(exact_samples, "exact buffered endpoints preserve all original IMU sample values");
  for (const auto &interval : std::vector<std::pair<double,double>>{{0.0, 2.0}, {0.25, 0.75}}) {
    const auto two = InitializerHelper::select_imu_readings(Samples{source.front(), source.back()}, interval.first, interval.second);
    check(two.size() == 2 && two.front().timestamp == interval.first && two.back().timestamp == interval.second,
          "two-sample buffer supports exact endpoints and an interval strictly inside one IMU bracket");
  }
  for (const auto &interval : std::vector<std::pair<double,double>>{
           {-0.001, 1.0}, {1.0, 2.001}, {1.0, 1.0}, {2.0, 1.0},
           {std::numeric_limits<double>::quiet_NaN(), 1.0}, {0.0, std::numeric_limits<double>::infinity()}})
    check(InitializerHelper::select_imu_readings(source, interval.first, interval.second).empty(),
          "missing bracket or invalid interval refuses partial propagation");
  check(InitializerHelper::select_imu_readings({}, 0.0, 1.0).empty() &&
        InitializerHelper::select_imu_readings(Samples{source.front()}, 0.0, 1.0).empty(), "insufficient IMU history rejected");
  auto duplicate = source;
  duplicate[2].timestamp = duplicate[1].timestamp;
  check(InitializerHelper::select_imu_readings(duplicate, 0.0, 2.0).empty(), "duplicate IMU timestamps rejected before interpolation");
  auto unordered = source;
  std::swap(unordered[1], unordered[2]);
  check(InitializerHelper::select_imu_readings(unordered, 0.0, 2.0).empty(), "unordered IMU timestamps rejected before interpolation");
  auto nonfinite = source;
  nonfinite[2].timestamp = std::numeric_limits<double>::quiet_NaN();
  check(InitializerHelper::select_imu_readings(nonfinite, 0.0, 2.0).empty(), "nonfinite IMU timestamp rejected under fast-math");
}
} // namespace

int main() {
  physical_clock_check();
  identity_and_horizon_checks();
  invalid_checks();
  imu_interval_checks();
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
