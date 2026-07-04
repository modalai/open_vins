/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 * @author Joao Leonardo Silva Cotta (@zauberflote1)
 *
 * AsyncCameraBuffer unit tests (host, CTest): threaded producers with delivery jitter, camera
 * death + staleness unblocking, camera restart, bit-equal bundling, ring overflow disposal
 * accounting, and the global release-order invariant.
 */

#include <atomic>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "core/AsyncCameraBuffer.h"
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

using namespace ov_msckf;

static int failures = 0;
#define CHECK(cond, ...)                                                                                                                   \
  do {                                                                                                                                     \
    if (!(cond)) {                                                                                                                         \
      failures++;                                                                                                                          \
      std::printf("[FAIL] %s:%d: ", __func__, __LINE__);                                                                                   \
      std::printf(__VA_ARGS__);                                                                                                            \
      std::printf("\n");                                                                                                                   \
    }                                                                                                                                      \
  } while (0)

static ov_core::CameraData make_frame(int cam, double ts) {
  ov_core::CameraData m;
  m.timestamp = ts;
  m.sensor_ids.push_back(cam);
  m.images.emplace_back();
  m.masks.emplace_back();
  return m;
}

// Scenario 1: two jittered 30 Hz producers, phase-shifted -> strictly increasing release order,
// every frame released once IMU covers it, zero drops of any kind.
static void test_ordered_release_two_producers() {
  std::atomic<uint64_t> disposed{0};
  AsyncCameraBuffer::Options o;
  o.ring_capacity = 64;
  AsyncCameraBuffer buf(2, o, [&](const ov_core::CameraData &) { disposed++; });

  const int N = 300;
  const double P = 1.0 / 30.0, phase1 = 0.0073;
  // Producers paced in wall time (1 ms per frame) like real camera pipes; occasional extra jitter
  std::atomic<bool> done0{false}, done1{false};
  std::thread p0([&] {
    for (int k = 0; k < N; k++) {
      buf.push(make_frame(0, 100.0 + k * P));
      std::this_thread::sleep_for(std::chrono::microseconds(1000 + ((k % 7 == 0) ? 400 : 0)));
    }
    done0 = true;
  });
  std::thread p1([&] {
    for (int k = 0; k < N; k++) {
      buf.push(make_frame(1, 100.0 + phase1 + k * P));
      std::this_thread::sleep_for(std::chrono::microseconds(1000 + ((k % 5 == 0) ? 600 : 0)));
    }
    done1 = true;
  });

  std::vector<double> released;
  std::vector<int> released_cams;
  auto dtfn = [](const std::vector<int> &) { return 0.005; };
  auto sink = [&](ov_core::CameraData &&m) {
    released.push_back(m.timestamp);
    released_cams.push_back(m.sensor_ids.front());
    return true;
  };
  // Consumer: IMU time always ahead of the stream (gate never limits); drain while producers run
  while (!done0.load() || !done1.load()) {
    buf.drain(1000.0, dtfn, sink);
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  p0.join();
  p1.join();
  // Stream tail: the last frames hold until the peer camera exceeds its staleness window
  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  buf.drain(1000.0, dtfn, sink);

  CHECK(released.size() == (size_t)(2 * N), "released %zu of %d frames", released.size(), 2 * N);
  for (size_t i = 1; i < released.size(); i++) {
    CHECK(released[i] > released[i - 1], "release order violated at %zu (%.6f <= %.6f)", i, released[i], released[i - 1]);
  }
  CHECK(disposed.load() == 0, "unexpected disposals: %llu", (unsigned long long)disposed.load());
  CHECK(buf.count_drop_late() == 0 && buf.count_drop_full() == 0 && buf.count_drop_bogus() == 0, "unexpected drop counters");
  std::printf("[ok] ordered_release_two_producers: %zu frames, strict order, 0 drops\n", released.size());
}

// Scenario 2: camera 1 dies mid-run -> after its staleness window, camera 0 keeps releasing;
// camera 1 restarts later (with a timestamp jump) and is re-admitted in order.
static void test_camera_death_and_restart() {
  std::atomic<uint64_t> disposed{0};
  AsyncCameraBuffer::Options o;
  o.ring_capacity = 64;
  o.stale_factor = 1.5;
  AsyncCameraBuffer buf(2, o, [&](const ov_core::CameraData &) { disposed++; });
  auto dtfn = [](const std::vector<int> &) { return 0.0; };
  const double P = 1.0 / 30.0;

  // Both alive: establish EMA periods (need >=2 pushes per cam)
  for (int k = 0; k < 5; k++) {
    buf.push(make_frame(0, 200.0 + k * P));
    buf.push(make_frame(1, 200.0 + 0.005 + k * P));
  }
  size_t got = 0;
  buf.drain(210.0, dtfn, [&](ov_core::CameraData &&) {
    got++;
    return true;
  });
  // The final cam1 frame is HELD: cam0 is live and could still deliver an older frame
  CHECK(got == 9, "warmup released %zu (expect 9: last cam1 frame held by live cam0)", got);

  // Camera 1 dead: camera 0 alone. First drains are BLOCKED (cam1 live, might deliver older),
  // then unblock once cam1's silence exceeds 1.5 * EMA period (~50 ms wall time).
  for (int k = 5; k < 10; k++)
    buf.push(make_frame(0, 200.0 + k * P));
  size_t before = 0;
  buf.drain(210.0, dtfn, [&](ov_core::CameraData &&) {
    before++;
    return true;
  });
  // Only the residual cam1 frame releases (cam0 now shows newer frames); cam0's own frames are
  // HELD because cam1 is still inside its staleness window
  CHECK(before == 1, "released %zu (expect 1: residual cam1 frame only)", before);
  std::this_thread::sleep_for(std::chrono::milliseconds(80)); // > 1.5 * 33 ms
  size_t after = 0;
  buf.drain(210.0, dtfn, [&](ov_core::CameraData &&) {
    after++;
    return true;
  });
  CHECK(after == 5, "cam0 released %zu of 5 after cam1 went stale", after);

  // Camera 1 restarts ahead in time: accepted, and ordering with cam0 resumes once cam1's
  // follow-up frame proves it has passed cam0's pending timestamp
  buf.push(make_frame(1, 200.0 + 12 * P));
  buf.push(make_frame(0, 200.0 + 12.5 * P));
  buf.push(make_frame(1, 200.0 + 13 * P));
  std::vector<double> tail;
  buf.drain(220.0, dtfn, [&](ov_core::CameraData &&m) {
    tail.push_back(m.timestamp);
    return true;
  });
  CHECK(tail.size() >= 2 && tail[0] < tail[1], "restart ordering wrong (%zu released)", tail.size());
  std::printf("[ok] camera_death_and_restart: staleness unblocked, restart re-admitted in order\n");
}

// Scenario 3: bit-equal timestamps bundle into one multi-sensor message (synced/stereo path)
static void test_bundling_equal_timestamps() {
  AsyncCameraBuffer::Options o;
  AsyncCameraBuffer buf(2, o, nullptr);
  auto dtfn = [](const std::vector<int> &) { return 0.0; };
  for (int k = 0; k < 10; k++) {
    double ts = 300.0 + k * 0.0333333;
    buf.push(make_frame(0, ts));
    buf.push(make_frame(1, ts)); // bit-equal
  }
  size_t events = 0, sensors = 0;
  buf.drain(400.0, dtfn, [&](ov_core::CameraData &&m) {
    events++;
    sensors += m.sensor_ids.size();
    CHECK(m.sensor_ids.size() == 2, "event not bundled (ids=%zu)", m.sensor_ids.size());
    return true;
  });
  CHECK(events == 10 && sensors == 20, "bundling wrong: %zu events / %zu sensors", events, sensors);
  CHECK(buf.count_bundled() == 10, "bundled counter %llu", (unsigned long long)buf.count_bundled());
  std::printf("[ok] bundling_equal_timestamps: 10 events x 2 sensors\n");
}

// Scenario 4: late frames (older than the last release) are dropped+disposed, never re-ordered;
// ring overflow disposes exactly the overflowing frames.
static void test_late_and_overflow_disposal() {
  std::atomic<uint64_t> disposed{0};
  AsyncCameraBuffer::Options o;
  o.ring_capacity = 8;
  AsyncCameraBuffer buf(2, o, [&](const ov_core::CameraData &) { disposed++; });
  auto dtfn = [](const std::vector<int> &) { return 0.0; };

  buf.push(make_frame(0, 400.0));
  buf.push(make_frame(1, 400.5)); // cam1 well ahead -> lets cam0's frame release
  size_t got = 0;
  buf.drain(500.0, dtfn, [&](ov_core::CameraData &&) {
    got++;
    return true;
  });
  CHECK(got == 1, "released %zu (expect 1: cam1 held by live cam0)", got);
  // cam0 goes stale (single push -> conservative 20 Hz default period, 1.5x = 75 ms)
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  buf.drain(500.0, dtfn, [&](ov_core::CameraData &&) {
    got++;
    return true;
  });
  CHECK(got == 2, "released %zu of 2 after cam0 staleness", got);

  // Late: cam0 delivers a frame OLDER than the last release (400.5)
  buf.push(make_frame(0, 400.2));
  buf.drain(500.0, dtfn, [&](ov_core::CameraData &&) {
    CHECK(false, "late frame was released");
    return true;
  });
  CHECK(buf.count_drop_late() == 1, "late counter %llu", (unsigned long long)buf.count_drop_late());
  CHECK(disposed.load() == 1, "late frame not disposed");

  // Overflow: capacity 8 -> pushing 12 without draining disposes 4 (the newest ones)
  for (int k = 0; k < 12; k++)
    buf.push(make_frame(0, 501.0 + k * 0.01));
  CHECK(buf.count_drop_full() == 4, "full-drop counter %llu", (unsigned long long)buf.count_drop_full());
  CHECK(disposed.load() == 5, "disposal accounting %llu (expect 5)", (unsigned long long)disposed.load());
  std::printf("[ok] late_and_overflow_disposal: exact disposal accounting\n");
}

// Scenario 5 (wiring, needs a config path argv[1]): a real VioManager driven only through the
// PUBLIC API -- camera pushes + one IMU batch -> frames drain on the IMU feed in order, and the
// processed-callback fires on the same thread for each.
static void test_viomanager_wiring(const char *config_path) {
  ov_core::Printer::setPrintLevel("ERROR");
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  VioManagerOptions params;
  params.print_and_load(parser);
  params.num_opencv_threads = 0;
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;
  auto vm = std::make_shared<VioManager>(params);

  std::vector<double> processed;
  vm->set_camera_processed_callback([&](const ov_core::CameraData &msg, bool ok_frame) {
    if (ok_frame)
      processed.push_back(msg.timestamp);
    return true;
  });

  auto make_img_frame = [&](int cam, double ts) {
    ov_core::CameraData m;
    m.timestamp = ts;
    m.sensor_ids.push_back(cam);
    m.images.emplace_back(cv::Mat::zeros(params.camera_intrinsics.at(cam)->h(), params.camera_intrinsics.at(cam)->w(), CV_8UC1));
    m.masks.emplace_back(cv::Mat::zeros(params.camera_intrinsics.at(cam)->h(), params.camera_intrinsics.at(cam)->w(), CV_8UC1));
    return m;
  };

  // Async frames pushed BEFORE any IMU: nothing may process yet. The third frame (cam0, newer
  // than cam1's) is what proves cam0 has passed cam1's timestamp so cam1's frame can release;
  // it itself stays held (cam1 live, older last-enqueued) -- the ordering contract.
  vm->feed_measurement_camera(make_img_frame(0, 500.000));
  vm->feed_measurement_camera(make_img_frame(1, 500.007));
  vm->feed_measurement_camera(make_img_frame(0, 500.033));
  CHECK(processed.empty(), "frames processed before any IMU arrived");

  // One IMU batch past the frames (+offsets+guard) -> the two ordering-decided frames drain
  std::vector<ov_core::ImuData> batch;
  for (int k = 0; k < 200; k++) {
    ov_core::ImuData s;
    s.timestamp = 499.9 + k * 0.00125; // 800 Hz, ends at 500.149
    s.wm = Eigen::Vector3d::Zero();
    s.am = Eigen::Vector3d(0, 0, 9.81);
    batch.push_back(s);
  }
  vm->feed_measurement_batch_imu(batch, 0.0);
  CHECK(processed.size() == 2, "processed %zu of 2 ordering-decided frames after IMU batch", processed.size());
  if (processed.size() == 2) {
    CHECK(processed[0] < processed[1], "frames processed out of order");
    CHECK(std::abs(processed[1] - 500.007) < 1e-9, "cam1's frame did not release (got %.6f)", processed[1]);
  }
  std::printf("[ok] viomanager_wiring: public-API push -> IMU-feed drain -> callback, in order\n");
}

int main(int argc, char **argv) {
  test_ordered_release_two_producers();
  test_camera_death_and_restart();
  test_bundling_equal_timestamps();
  test_late_and_overflow_disposal();
  if (argc > 1) {
    test_viomanager_wiring(argv[1]);
  } else {
    std::printf("[skip] viomanager_wiring (no config path given)\n");
  }
  if (failures == 0) {
    std::printf("[PASS] AsyncCameraBuffer: all scenarios green\n");
    return 0;
  }
  std::printf("[FAILED] AsyncCameraBuffer: %d failed checks\n", failures);
  return 1;
}
