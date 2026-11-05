// SPDX-License-Identifier: GPL-3.0-or-later
#include <cstdio>
#include "core/CalibSessionRunner.h"

// Ordinary sensor_msgs/Imu has no temperature field. Its existing unavailable
// placeholder must permit a still baseline before any timeout fallback.
int main() {
  ov_zcalib::SessionConfig config;
  config.verbose = false;
  config.settle_timeout_s = 60.0;
  ov_zcalib::CalibSessionRunner runner(config, ov_zcalib::SessionSeed{});
  for (int i = 0; i < 500; ++i) {
    ov_zcalib::RawImu sample;
    sample.timestamp = 0.005 * i;
    sample.am << 0.0, 0.0, 9.81;
    // No temperature supplied, just as in the ROS capture adapters.
    runner.feed_imu(sample);
  }
  if (runner.state() != ov_zcalib::RunnerState::BOOTSTRAP) {
    std::fprintf(stderr, "FAIL: temperature-free IMU could not complete the still baseline\n");
    return 1;
  }
  std::printf("PASS: ordinary IMU without temperature reaches bootstrap before settle timeout\n");
  return 0;
}
