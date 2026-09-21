/* Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Exercise production buffer updates against the historical erase-loop result.
 */
#include <cstdio>
#include "dynamic/DynamicInitializer.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "init/InertialInitializer.h"
#include "types/IMU.h"
#include "types/Landmark.h"
#include "types/PoseJPL.h"
#include "utils/sensor_data.h"

namespace {
using Samples = std::vector<ov_core::ImuData>;
int failures = 0;
void check(bool ok, const char *name) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++failures;
}
struct InspectInitializer : ov_init::InertialInitializer {
  using InertialInitializer::InertialInitializer;
  const Samples &samples() const { return *imu_data; }
};
Samples make_samples(const std::vector<double> &times) {
  Samples result;
  for (double t : times) {
    ov_core::ImuData s;
    s.timestamp = t;
    s.wm.setConstant(static_cast<double>(result.size())+.1);
    s.am.setConstant(static_cast<double>(result.size())+.7);
    result.push_back(s);
  }
  return result;
}
bool equal(const Samples &a, const Samples &b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].timestamp != b[i].timestamp || a[i].wm != b[i].wm || a[i].am != b[i].am) return false;
  return true;
}
// The old implementation is deliberately retained only as the test oracle.
Samples historical_all(Samples samples, double cutoff) {
  auto it = samples.begin();
  while (it != samples.end()) {
    if (it->timestamp < cutoff) it = samples.erase(it);
    else ++it;
  }
  return samples;
}
Samples historical_prefix(Samples samples, double cutoff) {
  auto it = samples.begin();
  while (it != samples.end() && it->timestamp < cutoff) it = samples.erase(it);
  return samples;
}

void feed_checks() {
  ov_init::InertialInitializerOptions options;
  auto db = std::make_shared<ov_core::FeatureDatabase>();
  for (const std::vector<double> &times : {std::vector<double>{}, {0.0}, {0.0, .1, .2}, {1.0, 1.2, 1.3},
                                          {.2, 1.0, .3, 1.4, 1.0, 1.5}}) {
    const Samples samples = make_samples(times);
    InspectInitializer single(options, db), batch(options, db);
    single.feed_imu_batch(samples);
    batch.feed_imu_batch(samples);
    const Samples appended = make_samples({.4, 1.0, 1.2});
    Samples one_expected = samples;
    one_expected.push_back(appended.front());
    single.feed_imu(appended.front(), 1.0);
    check(equal(single.samples(), historical_all(one_expected, 1.0)), "single feed stable all-old removal including equality");
    Samples batch_expected = samples;
    batch_expected.insert(batch_expected.end(), appended.begin(), appended.end());
    batch.feed_imu_batch(appended, 1.0);
    check(equal(batch.samples(), historical_all(batch_expected, 1.0)), "batch feed stable all-old removal including unsorted data");
    InspectInitializer empty_batch(options, db);
    empty_batch.feed_imu_batch(samples);
    empty_batch.feed_imu_batch({}, 1.0);
    check(equal(empty_batch.samples(), samples), "empty batch preserves history without pruning");
    empty_batch.feed_imu(appended.front(), -1);
    check(equal(empty_batch.samples(), one_expected), "disabled pruning preserves all samples");
  }
}

void prefix_checks() {
  ov_init::InertialInitializerOptions options;
  options.init_window_time = 1.0;
  options.init_max_disparity = 1.0;
  constexpr double newest = 3.25;
  for (bool dynamic : {false, true}) {
    const double cutoff = newest-options.init_window_time-(dynamic ? 0.0 : .10);
    for (const std::vector<double> &times : {std::vector<double>{}, {cutoff-1.0}, {cutoff, cutoff+1.0},
                                            {cutoff-1.0, cutoff, cutoff-.1, cutoff+.2, cutoff}}) {
      const Samples samples = make_samples(times);
      auto db = std::make_shared<ov_core::FeatureDatabase>();
      db->update_feature(1, newest, 0, 320, 240, 0, 0);
      double timestamp = -1;
      Eigen::MatrixXd covariance;
      std::vector<std::shared_ptr<ov_type::Type>> order;
      auto imu = std::make_shared<ov_type::IMU>();
      std::map<double, std::shared_ptr<ov_type::PoseJPL>> clones;
      std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> landmarks;
      Samples actual;
      if (dynamic) {
        auto history = std::make_shared<Samples>(samples);
        ov_init::DynamicInitializer initializer(options, db, history);
        check(!initializer.initialize(timestamp, covariance, order, imu, clones, landmarks), "sparse dynamic fixture rejects before solve");
        actual = *history;
      } else {
        InspectInitializer initializer(options, db);
        initializer.feed_imu_batch(samples);
        check(!initializer.initialize(timestamp, covariance, order, imu, clones, landmarks), "sparse public fixture rejects before solve");
        actual = initializer.samples();
      }
      check(equal(actual, historical_prefix(samples, cutoff)), "initialization preserves exact historical prefix boundary");
      if (samples.size() == 5)
        check(!equal(actual, historical_all(samples, cutoff)), "negative control detects accidental all-old removal in prefix path");
    }
  }
}
} // namespace

int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  feed_checks(); prefix_checks();
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
