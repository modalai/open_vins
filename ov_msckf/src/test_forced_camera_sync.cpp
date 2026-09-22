/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#include "core/AsyncCameraBuffer.h"

using ov_core::CameraData;
using ov_msckf::AsyncCameraBuffer;

static int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
  std::printf("FAIL %s:%d: %s\n", __func__, __LINE__, #condition); ++failures; \
} } while (0)

class Buffer : public AsyncCameraBuffer {
public:
  using AsyncCameraBuffer::AsyncCameraBuffer;
  void age_pending() {
    for (auto &s : streams)
      s->staged_since = mono_now() - 10.;
  }
};

static AsyncCameraBuffer::Options options(int reference = 0) {
  AsyncCameraBuffer::Options o;
  o.sync_reference_camera = reference;
  return o;
}

static CameraData frame(int camera, int64_t start, int64_t exposure = 4000000) {
  CameraData m;
  m.timestamp = 1e-9 * static_cast<double>(start + exposure / 2);
  m.sync_timestamp_ns = start;
  m.sensor_ids = {camera};
  m.exposures = {static_cast<float>(1e-9 * exposure)};
  m.images.emplace_back(1, 1, CV_8UC1, cv::Scalar(camera + 1));
  m.masks.emplace_back();
  return m;
}

static double zero_offset(const std::vector<int> &) { return 0.; }

static void shared_reference_time() {
  for (int reference : {0, 1}) for (int first : {0, 1}) {
    size_t disposed = 0;
    Buffer b(2, options(reference), [&](const CameraData &) { ++disposed; });
    b.prepare_recorded_input();
    std::vector<CameraData> output;
    auto sink = [&](CameraData &&m) { output.push_back(std::move(m)); return true; };
    for (int i = 0; i < 3; ++i) {
      const int64_t start = 10000000000LL + i * 33000000LL;
      CameraData input[2] = {frame(0, start, 4000000 + i * 2000000),
                             frame(1, start, 16000000 - i * 4000000)};
      CHECK(b.push(input[first]));
      b.drain(20., zero_offset, sink);
      CHECK(output.size() == static_cast<size_t>(i));
      b.age_pending(); // replay speed must not expire a real matching frame
      b.drain(20., zero_offset, sink);
      CHECK(output.size() == static_cast<size_t>(i));
      CHECK(b.push(input[1 - first]));
      b.drain(20., zero_offset, sink);
      CHECK(output.size() == static_cast<size_t>(i + 1));
      if (output.size() != static_cast<size_t>(i + 1)) continue;
      const auto &m = output.back();
      CHECK(m.timestamp == input[reference].timestamp);
      CHECK(m.sync_timestamp_ns == start);
      CHECK(m.sensor_ids == std::vector<int>({0, 1}));
      CHECK(m.images.size() == 2 && m.exposures.size() == 2);
      if (m.images.size() != 2 || m.exposures.size() != 2) continue;
      for (int c = 0; c < 2; ++c) {
        CHECK(m.images[c].data == input[c].images[0].data);
        CHECK(m.exposures[c] == input[c].exposures[0]);
      }
    }
    CHECK(disposed == 0 && b.count_drop_unpaired() == 0);
  }
}

static void missing_mates() {
  size_t disposed = 0;
  Buffer b(2, options(), [&](const CameraData &m) { disposed += m.sensor_ids.size(); });
  b.prepare_recorded_input();
  std::vector<CameraData> output;
  auto sink = [&](CameraData &&m) { output.push_back(std::move(m)); return true; };
  CHECK(b.push(frame(0, 10000000000LL)));
  CHECK(b.push(frame(1, 10033000000LL))); // next capture cannot substitute for a missing mate
  b.drain(20., zero_offset, sink);
  CHECK(output.empty() && disposed == 1);
  CHECK(b.push(frame(0, 10033000000LL)));
  b.drain(20., zero_offset, sink);
  CHECK(output.size() == 1);
  CHECK(b.push(frame(1, 10066000000LL)));
  b.finish_all();
  b.drain(20., zero_offset, sink);
  CHECK(output.size() == 1 && disposed == 2 && b.count_drop_unpaired() == 2);

  Buffer live(2, options(), [&](const CameraData &m) { disposed += m.sensor_ids.size(); });
  CHECK(live.push(frame(1, 10000000000LL)));
  live.drain(20., zero_offset, sink);
  live.age_pending();
  live.drain(20., zero_offset, sink);
  CHECK(output.size() == 1 && live.count_drop_unpaired() == 1); // no reference fallback
}

static void coverage_and_reset() {
  size_t disposed = 0, released = 0;
  const auto o = options();
  Buffer b(2, o, [&](const CameraData &m) { disposed += m.sensor_ids.size(); });
  const auto reference = frame(0, 10000000000LL, 18000000);
  const auto peer = frame(1, 10000000000LL, 2000000);
  auto offset = [](const std::vector<int> &ids) {
    CHECK(ids == std::vector<int>({0, 1}));
    return .007; // largest camera offset, even though the reference owns the stamp
  };
  auto sink = [&](CameraData &&m) {
    CHECK(m.timestamp == reference.timestamp);
    ++released;
    return false; // consuming a group may pause the drain
  };
  CHECK(b.push(peer) && b.push(reference));
  const double endpoint = reference.timestamp + .007 + o.guard;
  b.drain(endpoint - 1e-9, offset, sink);
  CHECK(released == 0 && disposed == 0);
  b.drain(endpoint, offset, sink);
  CHECK(released == 1 && disposed == 0);
  CHECK(b.push(frame(0, 10033000000LL)) && b.push(frame(1, 10033000000LL)));
  b.drain(10., offset, sink); // complete group staged, IMU has not covered it
  b.clear();
  CHECK(disposed == 2);
  b.clear();
  CHECK(disposed == 2); // no double disposal of the cached normalized group
  CHECK(b.push(reference) && b.push(peer)); // reset permits replay from the first capture
  b.drain(endpoint, offset, sink);
  CHECK(released == 2 && disposed == 2);
}

static void already_synchronized_and_opt_out() {
  for (bool synchronize : {false, true}) {
    auto o = options();
    if (!synchronize) o.sync_reference_camera = -1;
    Buffer b(2, o, nullptr);
    auto a = frame(0, 10000000000LL), c = frame(1, 10000000000LL, 16000000);
    if (synchronize) {
      // Observation records already contain shared times, but no raw trigger key.
      a.sync_timestamp_ns = c.sync_timestamp_ns = -1;
      c.timestamp = a.timestamp;
    }
    CHECK(b.push(c) && b.push(a));
    b.finish_all();
    std::vector<CameraData> output;
    b.drain(20., zero_offset, [&](CameraData &&m) { output.push_back(std::move(m)); return true; });
    CHECK(output.size() == (synchronize ? 1u : 2u));
    if (output.empty()) continue;
    CHECK(output[0].timestamp == a.timestamp);
    if (!synchronize && output.size() == 2) CHECK(output[1].timestamp == c.timestamp);
    if (synchronize) CHECK(output[0].sensor_ids.size() == 2);
  }
}

static void exact_trigger_keys_and_invalid_input() {
  size_t disposed = 0, released = 0;
  Buffer b(2, options(), [&](const CameraData &) { ++disposed; });
  auto sink = [&](CameraData &&) { ++released; return true; };
  // These distinct nanosecond keys round to the same double. They must not pair.
  auto a = frame(0, 10000000000LL), c = frame(1, 10000000000LL);
  a.sync_timestamp_ns = 1800000000000000000LL;
  c.sync_timestamp_ns = a.sync_timestamp_ns + 1;
  CHECK(b.push(a) && b.push(c));
  b.finish_all();
  b.drain(20., zero_offset, sink);
  CHECK(released == 0 && disposed == 2 && b.count_drop_unpaired() == 2);
  b.clear();
  a = frame(0, 10000000000LL);
  a.exposures.push_back(.003f); // malformed parallel metadata must not escape
  CHECK(b.push(a));
  c = frame(1, 10000000000LL);
  c.timestamp = std::numeric_limits<double>::quiet_NaN();
  CHECK(b.push(c));
  b.drain(20., zero_offset, sink);
  CHECK(released == 0 && disposed == 4 && b.count_drop_bogus() == 2);
}

static void physical_offsets_keep_the_common_camera_clock() {
  auto o = options(1);
  o.physical_order = true;
  Buffer b(2, o, nullptr);
  b.prepare_recorded_input();
  const auto a = frame(0, 10000000000LL), c = frame(1, 10000000000LL, 12000000);
  CHECK(b.push(a) && b.push(c));
  b.finish_all();
  size_t views = 0;
  double previous = -1.;
  b.drain_physical(20., [](int camera) { return camera == 0 ? .020 : -.010; },
    [&](std::vector<CameraData> &group, double endpoint) {
      CHECK(endpoint > previous);
      previous = endpoint;
      for (const auto &m : group) {
        CHECK(m.timestamp == c.timestamp);
        CHECK(m.sync_timestamp_ns == c.sync_timestamp_ns);
        views += m.sensor_ids.size();
      }
      return true;
    });
  CHECK(views == 2);
}

int main() {
  shared_reference_time();
  missing_mates();
  coverage_and_reset();
  already_synchronized_and_opt_out();
  exact_trigger_keys_and_invalid_input();
  physical_offsets_keep_the_common_camera_clock();
  std::printf("FORCED_CAMERA_SYNC %s failures=%d\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
