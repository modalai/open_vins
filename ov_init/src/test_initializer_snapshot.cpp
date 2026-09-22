/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <atomic>
#include <cstdio>
#include <cstring>
#include <future>
#include <thread>
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "init/InertialInitializer.h"
#include "types/IMU.h"
#include "types/PoseJPL.h"
#include "types/Landmark.h"
#include "utils/sensor_data.h"

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char *why) {
  ++checks;
  if (!ok) { ++failures; std::printf("FAIL: %s\n", why); }
}
template <class A, class B> bool same(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
      std::memcmp(a.derived().data(), b.derived().data(), sizeof(double) * a.size()) == 0;
}
struct Result {
  bool success;
  double time = -1.;
  std::shared_ptr<ov_type::IMU> imu = std::make_shared<ov_type::IMU>();
  Eigen::MatrixXd covariance;
};
Result run(ov_init::InertialInitializer &initializer) {
  Result result;
  std::vector<std::shared_ptr<ov_type::Type>> order;
  std::map<double, std::shared_ptr<ov_type::PoseJPL>> clones;
  std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> features;
  result.success = initializer.initialize(result.time, result.covariance, order, result.imu, clones, features, false);
  return result;
}
ov_core::ImuData sample(double time) {
  ov_core::ImuData value; value.timestamp = time; value.wm.setZero(); value.am << 0, 0, -9.81;
  return value;
}
void tracks(const std::shared_ptr<ov_core::FeatureDatabase> &db, double time) {
  for (size_t id = 0; id < 20; ++id)
    db->update_feature(id, time, 0, 10 + id, 20, .1, .2);
}
}
int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  ov_init::InertialInitializerOptions options;
  options.init_window_time = 2.; options.init_max_disparity = 1.;
  options.init_imu_thresh = 1.; options.init_dyn_use = false;
  auto db = std::make_shared<ov_core::FeatureDatabase>();
  ov_init::InertialInitializer initializer(options, db);
  for (int i = 0; i <= 400; ++i) initializer.feed_imu(sample(2. + .005 * i));
  for (int i = 0; i <= 16; ++i) tracks(db, 2. + .125 * i);
  auto frozen = initializer.make_attempt();
  const auto before = run(*frozen);
  check(before.success, "real stationary policy and static initializer accept the frozen input");

  // Feed and prune both producer stores after capture, including enough new
  // IMU to reallocate the original vector. The retained result is bit-exact.
  for (int i = 1; i <= 1000; ++i) {
    initializer.feed_imu(sample(4. + .001 * i), 3.);
    tracks(db, 4. + .001 * i);
  }
  db->cleanup_measurements(4.5);
  const auto after = run(*frozen);
  check(after.success && before.time == after.time && same(before.imu->value(), after.imu->value()) &&
        same(before.imu->fej(), after.imu->fej()) && same(before.covariance, after.covariance),
        "real initializer snapshot survives subsequent track/IMU mutation and pruning bit-exactly");

  // Deterministic rendezvous starts three owners together. This exercises the
  // public snapshot + disparity path while distinct camera/IMU producers keep
  // appending. A sanitizer can observe every production memory access here.
  auto concurrent_db = std::make_shared<ov_core::FeatureDatabase>();
  ov_init::InertialInitializer concurrent_initializer(options, concurrent_db);
  for (int i = 0; i <= 400; ++i) concurrent_initializer.feed_imu(sample(3. + .005 * i));
  for (int i = 0; i <= 16; ++i) tracks(concurrent_db, 3. + .125 * i);
  std::promise<void> start;
  auto ready = start.get_future().share();
  std::atomic<int> producers{2};
  std::thread imu_producer([&] {
    ready.wait();
    for (int i = 0; i < 512; ++i) {
      concurrent_initializer.feed_imu(sample(5. + .0002 * i), 2.);
      concurrent_initializer.feed_imu_batch({sample(5. + .0002 * i + .0001)}, 2.);
    }
    --producers;
  });
  std::thread camera_producer([&] {
    ready.wait();
    for (int i = 0; i < 512; ++i) tracks(concurrent_db, 5. + .0002 * i);
    --producers;
  });
  start.set_value();
  for (int i = 0; i < 32 || producers.load(); ++i) {
    const auto result = run(concurrent_initializer);
    check(result.success && result.covariance.rows() == 15 && result.covariance.allFinite() &&
          std::abs(result.imu->quat().squaredNorm() - 1.) < 1e-12,
          "concurrent public policy completes a valid real static solve throughout producer overlap");
  }
  imu_producer.join(); camera_producer.join();
  auto independent = db->clone();
  tracks(db, 10.);
  for (const auto &feature : independent->get_internal_data()) {
    check(feature.second->timestamps.at(0).back() < 10. &&
          feature.second->uvs.at(0).size() == feature.second->timestamps.at(0).size() &&
          feature.second->uvs_norm.at(0).size() == feature.second->timestamps.at(0).size(),
          "database snapshot preserves aligned raw and normalized observation rows independently");
  }
  std::printf("INITIALIZER_SNAPSHOT %s checks=%d failures=%d\n", failures ? "FAIL" : "PASS", checks, failures);
  return failures ? 1 : 0;
}
