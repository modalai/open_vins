/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>
#include "core/AsyncCameraBuffer.h"

using ov_core::CameraData;
using ov_msckf::AsyncCameraBuffer;
namespace {
int failures = 0;
size_t checks = 0;
void check(bool ok, const char *why) {
  ++checks;
  if (!ok) { ++failures; std::printf("FAIL: %s\n", why); }
}
CameraData frame(double raw, std::initializer_list<int> cameras, int token) {
  CameraData out;
  out.timestamp = raw;
  for (int camera : cameras) {
    out.sensor_ids.push_back(camera);
    out.images.emplace_back(1, 1, CV_32SC1, cv::Scalar(token));
    out.masks.emplace_back(1, 1, CV_32SC1, cv::Scalar(camera));
    out.exposures.push_back(static_cast<float>(token));
    ++token;
  }
  return out;
}
struct Ledger {
  std::map<int, int> handled;
  size_t released = 0, discarded = 0;
  void take(const CameraData &data, bool dropped) {
    for (size_t i = 0; i < data.images.size(); ++i) {
      if (data.images[i].empty()) continue;
      const int token = data.images[i].at<int>(0);
      ++handled[token];
      if (dropped) ++discarded; else ++released;
      if (data.sensor_ids.size() == data.images.size()) {
        if (data.masks.size() == data.images.size())
          check(data.masks[i].empty() || data.masks[i].at<int>(0) == data.sensor_ids[i],
                "split mask stays aligned with its camera");
        if (data.exposures.size() == data.images.size())
          check(data.exposures[i] == token, "split exposure provenance stays aligned with image");
      }
    }
  }
  void exactly_once(size_t expected) {
    check(handled.size() == expected, "every input image handle reaches one terminal owner");
    for (auto entry : handled) check(entry.second == 1, "image handle is consumed or disposed exactly once");
  }
};
struct ObservationLedger {
  std::map<size_t, const ov_core::FeatureObservations *> owners;
  std::map<size_t, int> handled;
  std::vector<std::weak_ptr<const ov_core::FeatureObservations>> lifetimes;
  size_t released = 0, discarded = 0;
  CameraData frame(double raw, std::initializer_list<int> cameras, size_t token) {
    CameraData out;
    out.timestamp = raw;
    for (int camera : cameras) {
      auto payload = std::make_shared<ov_core::FeatureObservations>();
      Eigen::VectorXf uv(2); uv << static_cast<float>(token), static_cast<float>(camera);
      payload->emplace_back(token, uv);
      owners[token++] = payload.get();
      lifetimes.push_back(payload);
      out.sensor_ids.push_back(camera);
      out.images.emplace_back(); out.masks.emplace_back();
      out.observations.push_back(std::move(payload));
    }
    return out;
  }
  void take(const CameraData &message, bool dropped) {
    if (!dropped)
      check(message.observations.size() == message.sensor_ids.size(), "released observation slots stay camera aligned");
    for (size_t i = 0; i < message.observations.size(); ++i) {
      const auto &payload = message.observations[i];
      if (!payload) continue; // malformed-input disposer also owns any valid remaining slots
      check(payload->size() == 1 && payload->front().second.size() == 2, "observation payload contents remain immutable");
      const size_t token = payload->front().first;
      check(owners.at(token) == payload.get() && payload->front().second(0) == static_cast<float>(token),
            "split and bundle move the exact shared observation owner without copying feature vectors");
      if (!dropped)
        check(payload->front().second(1) == message.sensor_ids.at(i) && message.images.at(i).empty() && message.masks.at(i).empty(),
              "queued observations retain camera identity and empty image headers");
      ++handled[token];
      if (dropped) ++discarded; else ++released;
    }
  }
  void complete() {
    check(handled.size() == owners.size(), "every observation payload reaches one terminal owner");
    for (const auto &entry : handled) check(entry.second == 1, "observation payload is consumed or disposed exactly once");
    for (const auto &payload : lifetimes) check(payload.expired(), "buffer releases its observation owner after terminal handling");
  }
};
struct Buffer : AsyncCameraBuffer {
  using AsyncCameraBuffer::AsyncCameraBuffer;
  void stale(int camera) { physical_cameras.at(camera)->last_push_mono.store(mono_now() - 1000.); }
  size_t group_capacity() const { return physical_group.capacity(); }
  const void *group_storage() const { return physical_group.data(); }
  size_t staged_views() const {
    size_t count = 0;
    for (const auto &camera : physical_cameras) count += camera->size;
    return count;
  }
  size_t stage_capacity() const {
    size_t count = 0;
    for (const auto &camera : physical_cameras) count += camera->pending.size();
    return count;
  }
};
AsyncCameraBuffer::Options options(size_t capacity = 4) {
  AsyncCameraBuffer::Options out;
  out.physical_order = true;
  out.ring_capacity = capacity;
  out.guard = 0;
  out.stale_factor = 0;
  return out;
}
void recorded_input_ordering() {
  for (bool recorded : {false, true}) {
    Ledger ledger;
    Buffer buffer(2, options(), [&](const CameraData &m) { ledger.take(m, true); });
    if (recorded) buffer.prepare_recorded_input();
    auto lookup = [](int camera) { return camera == 0 ? .125 : 0.; };
    std::vector<double> endpoints;
    auto sink = [&](std::vector<CameraData> &group, double endpoint) {
      endpoints.push_back(endpoint);
      for (auto &m : group) ledger.take(m, false);
      return true;
    };
    buffer.push(frame(10., {0}, 5001));
    buffer.stale(0); buffer.stale(1); // deterministic compute-stall control; no sleep
    buffer.drain_physical(11., lookup, sink);
    check(endpoints.size() == (recorded ? 0u : 1u),
          "recording waits for unseen logical cameras; live startup policy is unchanged");
    bool rejected = false;
    try { buffer.prepare_recorded_input(); } catch (const std::logic_error &) { rejected = true; }
    check(rejected, "ordering policy cannot change after producer input");
    buffer.push(frame(10.0625, {1}, 5002));
    buffer.stale(0); buffer.stale(1);
    buffer.drain_physical(11., lookup, sink);
    check(endpoints == std::vector<double>({recorded ? 10.0625 : 10.125}),
          "recorded watermark prevents the live-timeout late-frame counterexample");
    buffer.push(frame(10.1875, {1}, 5003));
    buffer.stale(0); buffer.stale(1);
    buffer.drain_physical(11., lookup, sink);
    if (recorded)
      check(endpoints == std::vector<double>({10.0625, 10.125}),
            "empty recorded camera still holds its peer despite wall staleness");
    buffer.finish_all();
    buffer.drain_physical(11., lookup, sink);
    check(ledger.released == (recorded ? 3u : 2u) && ledger.discarded == (recorded ? 0u : 1u),
          "recorded EOF delivers every view exactly once; live timeout still trades a late view for progress");
    buffer.clear();
    endpoints.clear();
    buffer.push(frame(2., {0}, 5004));
    buffer.stale(0); buffer.stale(1);
    buffer.drain_physical(3., lookup, sink);
    check(endpoints.size() == (recorded ? 0u : 1u), "reset preserves recorded ordering policy");
    buffer.finish_camera(1); // configured camera absent from this replay segment
    buffer.drain_physical(3., lookup, sink);
    check(endpoints == std::vector<double>({2.125}), "explicit absent-camera EOF unblocks a recording");
    ledger.exactly_once(4);
  }
  auto raw_options = options(); raw_options.physical_order = false;
  Buffer raw(1, raw_options, {});
  bool rejected = false;
  try { raw.prepare_recorded_input(); } catch (const std::logic_error &) { rejected = true; }
  check(rejected, "physical recorded ordering cannot silently change the legacy raw merge");
}
void ordering_and_grouping() {
  Ledger ledger;
  Buffer buffer(2, options(), [&](const CameraData &m) { ledger.take(m, true); });
  std::vector<double> dt{.25, 0};
  auto lookup = [&](int c) { return dt.at(c); };
  buffer.push(frame(10., {0}, 1));
  buffer.push(frame(10.125, {1}, 2));
  std::vector<double> endpoints, raws;
  buffer.drain_physical(11., lookup, [&](std::vector<CameraData> &group, double endpoint) {
    endpoints.push_back(endpoint);
    check(group.size() == 1, "unequal physical endpoints remain separate groups");
    raws.push_back(group.front().timestamp);
    ledger.take(group.front(), false);
    return true;
  });
  check(endpoints == std::vector<double>({10.125, 10.25}), "physical ordering can reverse raw-clock ordering");
  check(raws == std::vector<double>({10.125, 10.}), "raw keys are immutable");
  dt = {.25, .125};
  buffer.push(frame(20., {0}, 3));
  buffer.push(frame(20.125, {1}, 4));
  size_t groups = 0;
  buffer.drain_physical(21., lookup, [&](std::vector<CameraData> &group, double endpoint) {
    ++groups;
    check(endpoint == 20.25 && group.size() == 2, "equal physical/different raw views form one group without timestamp merging");
    check(group[0].timestamp == 20. && group[1].timestamp == 20.125, "group preserves each camera's original key");
    for (auto &m : group) ledger.take(m, false);
    dt = {5., -5.}; // selected members/endpoint must stay fixed through this callback
    return false;
  });
  check(groups == 1, "whole group delivered once before pause");
  dt = {.125, .125};
  buffer.push(frame(30., {0,1}, 5));
  buffer.drain_physical(31., lookup, [&](std::vector<CameraData> &group, double endpoint) {
    check(endpoint == 30.125 && group.size() == 1 && group[0].sensor_ids.size() == 2,
          "equal-raw equal-physical packed stereo remains one message");
    ledger.take(group[0], false);
    return true;
  });
  ledger.exactly_once(6);
  check(buffer.count_physical_views() == 6, "physical view counter does not reduce frame count");
}
void packed_backlog() {
  Ledger ledger;
  Buffer buffer(2, options(), [&](const CameraData &m) { ledger.take(m, true); });
  for (int k = 0; k < 3; ++k) buffer.push(frame(40. + .125*k, {0,1}, 10+2*k));
  std::vector<double> actual;
  buffer.drain_physical(42., [](int c) { return c == 0 ? .5 : 0.; },
      [&](std::vector<CameraData> &group, double endpoint) {
        actual.push_back(endpoint);
        check(group.size() == 1 && group[0].sensor_ids.size() == 1, "packed unequal-offset source splits by camera");
        ledger.take(group[0], false);
        return true;
      });
  check(actual == std::vector<double>({40.,40.125,40.25,40.5,40.625,40.75}),
        "later packed-message early cameras are not hidden behind an earlier message's late camera");
  check(buffer.stage_capacity() == 8 && buffer.staged_views() == 0, "physical staging remains bounded by cameras times ring capacity");
  ledger.exactly_once(6);
}
void mixed_producer_heads() {
  Ledger ledger;
  Buffer buffer(2, options(), [&](const CameraData &m) { ledger.take(m, true); });
  buffer.push(frame(100.125, {0}, 2000));
  buffer.push(frame(100.1875, {1}, 2001));
  buffer.push(frame(100.3125, {0,1}, 2002));
  std::vector<double> endpoints, raws;
  std::vector<int> cameras;
  buffer.drain_physical(100.5, [](int c) { return c == 0 ? .125 : 0.; },
      [&](std::vector<CameraData> &group, double endpoint) {
        endpoints.push_back(endpoint);
        for (auto &m : group) {
          raws.push_back(m.timestamp);
          cameras.push_back(m.sensor_ids.at(0));
          ledger.take(m, false);
        }
        return true;
      });
  check(endpoints == std::vector<double>({100.1875,100.25,100.3125,100.4375}) &&
        cameras == std::vector<int>({1,0,1,0}) && raws == std::vector<double>({100.1875,100.125,100.3125,100.3125}),
        "raw producer-head merge preserves earlier independent view before a later packed message");
  check(ledger.discarded == 0 && buffer.count_drop_late() == 0 && buffer.count_physical_views() == 4,
        "mixed independent/packed monotone inputs consume every camera view");
  ledger.exactly_once(4);
}
void full_queue_disjoint_head() {
  Ledger ledger;
  Buffer buffer(2, options(1), [&](const CameraData &m) { ledger.take(m, true); });
  auto lookup = [](int c) { return c == 0 ? .5 : 0.; };
  std::vector<double> endpoints;
  auto sink = [&](std::vector<CameraData> &group, double endpoint) {
    endpoints.push_back(endpoint);
    for (auto &m : group) ledger.take(m, false);
    return true;
  };
  buffer.push(frame(110., {0}, 2010));
  buffer.drain_physical(110.25, lookup, sink); // full logical queue, no IMU coverage yet
  buffer.push(frame(110.125, {0}, 2011));      // remains owned by source staging
  buffer.push(frame(110.25, {1}, 2012));
  buffer.drain_physical(111., lookup, sink);
  check(endpoints == std::vector<double>({110.25,110.5,110.625}),
        "a full earlier source queue does not block an unrelated producer's earlier physical frame");
  check(ledger.discarded == 0 && buffer.staged_views() == 0 && buffer.count_physical_views() == 3,
        "partial source ownership survives bounded logical-camera backpressure");
  ledger.exactly_once(3);
}
void bounded_invalid_replenishment() {
  Ledger ledger;
  Buffer *active = nullptr;
  bool replenish = false;
  int next_token = 2022;
  const double invalid = std::numeric_limits<double>::quiet_NaN();
  Buffer buffer(1, options(2), [&](const CameraData &m) {
    ledger.take(m, true);
    if (replenish)
      active->push(frame(invalid, {0}, next_token++));
  });
  active = &buffer;
  buffer.push(frame(invalid, {0}, 2020));
  buffer.push(frame(invalid, {0}, 2021));
  replenish = true;
  buffer.drain_physical(120., [](int) { return 0.; }, [&](std::vector<CameraData> &, double) {
    check(false, "invalid replenished messages must never reach the sink"); return true;
  });
  replenish = false;
  check(buffer.count_drop_bogus() == 2 && next_token == 2024,
        "one drain processes only its bounded entry budget when invalid inputs keep arriving");
  buffer.clear();
  ledger.exactly_once(4);
}
void clock_updates_and_late_raw() {
  Ledger ledger;
  Buffer buffer(3, options(), [&](const CameraData &m) { ledger.take(m, true); });
  std::vector<double> dt{0., .5, .75};
  auto lookup = [&](int c) { return dt.at(c); };
  for (int c = 0; c < 3; ++c) buffer.push(frame(50., {c}, 20+c));
  std::vector<double> actual;
  buffer.drain_physical(52., lookup, [&](std::vector<CameraData> &group, double endpoint) {
    actual.push_back(endpoint);
    for (auto &m : group) ledger.take(m, false);
    if (actual.size() == 1) { dt[1] = .875; dt[2] = .625; }
    return true;
  });
  check(actual == std::vector<double>({50.,50.625,50.875}), "pending heads reorder after online clock updates");
  dt = {0., .25, .5};
  buffer.push(frame(60., {0}, 23));
  buffer.push(frame(60., {1}, 24));
  buffer.drain_physical(62., lookup, [&](std::vector<CameraData> &group, double) {
    for (auto &m : group) ledger.take(m, false);
    dt[1] = -.125;
    return true;
  });
  check(buffer.count_drop_late() == 1, "clock update moving a pending view behind consumed physical time drops it");
  dt[0] = 2.;
  buffer.push(frame(59., {0}, 25));
  buffer.drain_physical(63., lookup, [&](std::vector<CameraData> &, double) {
    check(false, "raw-regressing frame must not become admissible through a changed offset"); return true;
  });
  check(buffer.count_drop_late() == 2, "per-camera raw monotonicity is independent of physical endpoint ordering");
  buffer.clear();
  dt[0] = 0.;
  buffer.push(frame(5., {0}, 26));
  buffer.drain_physical(6., lookup, [&](std::vector<CameraData> &group, double endpoint) {
    check(endpoint == 5., "clear resets physical and raw-history endpoints");
    for (auto &m : group) ledger.take(m, false);
    return true;
  });
  ledger.exactly_once(7);
}
void coverage_stale_and_pause() {
  Ledger ledger;
  auto o = options(); o.guard = .002; o.stale_factor = 1.5;
  Buffer buffer(2, o, [&](const CameraData &m) { ledger.take(m, true); });
  buffer.push(frame(70., {0}, 30));
  size_t calls = 0;
  auto sink = [&](std::vector<CameraData> &group, double) {
    ++calls; for (auto &m : group) ledger.take(m, false); return true;
  };
  buffer.drain_physical(70.125, [](int) { return .125; }, sink);
  check(calls == 0, "physical IMU coverage includes interpolation guard");
  buffer.drain_physical(70.125 + .002, [](int) { return .125; }, sink);
  check(calls == 1, "exact covered guard endpoint releases");
  buffer.push(frame(80., {0}, 31));
  buffer.push(frame(80.125, {1}, 32));
  buffer.drain_physical(81., [](int) { return 0.; }, sink);
  check(calls == 2, "live camera with no next physical head holds the peer");
  buffer.stale(0);
  buffer.drain_physical(81., [](int) { return 0.; }, sink);
  check(calls == 3, "stale camera unblocks the pending physical head");
  buffer.push(frame(80.25, {0}, 33));
  buffer.drain_physical(81., [](int) { return 0.; }, sink);
  check(calls == 3, "restarted camera still respects a live peer's order");
  buffer.push(frame(80.375, {1}, 34));
  buffer.drain_physical(81., [](int) { return 0.; }, [&](std::vector<CameraData> &group, double endpoint) {
    ++calls; check(endpoint == 80.25, "restart physical order");
    for (auto &m : group) ledger.take(m, false); return false;
  });
  check(calls == 4 && buffer.staged_views() == 1, "false pauses after consuming the complete group");
  buffer.clear();
  ledger.exactly_once(5);
}
void bounded_disposal_and_reset_in_callback() {
  Ledger ledger;
  Buffer buffer(2, options(2), [&](const CameraData &m) { ledger.take(m, true); });
  check(buffer.push(frame(90., {0,1}, 40)), "first packed push");
  check(buffer.push(frame(90.125, {0,1}, 42)), "second packed push");
  check(!buffer.push(frame(90.25, {0,1}, 44)), "full source ring remains bounded");
  const auto capacity = buffer.group_capacity();
  const auto storage = buffer.group_storage();
  buffer.drain_physical(90.01, [](int c) { return c == 0 ? 0. : .5; },
      [&](std::vector<CameraData> &group, double endpoint) {
        check(endpoint == 90. && group.size() == 1, "only covered first component releases");
        buffer.clear(); // must not invalidate the borrowed consumed group
        check(group.size() == 1 && group[0].sensor_ids == std::vector<int>({0}), "reset keeps in-flight callback group alive");
        ledger.take(group[0], false);
        return false;
      });
  check(buffer.count_drop_full() == 1 && ledger.released == 1 && ledger.discarded == 5, "split/overflow/clear disposal counts components exactly once");
  check(buffer.group_capacity() == capacity && buffer.group_storage() == storage, "group vector storage is reused");
  ledger.exactly_once(6);
}
void invalid_clocks_and_messages() {
  Ledger ledger;
  Buffer buffer(2, options(), [&](const CameraData &m) { ledger.take(m, true); });
  buffer.push(frame(100., {0}, 50));
  size_t calls = 0;
  auto sink = [&](std::vector<CameraData> &group, double) { ++calls; for (auto &m : group) ledger.take(m,false); return true; };
  buffer.drain_physical(101., [](int) { return std::numeric_limits<double>::quiet_NaN(); }, sink);
  buffer.drain_physical(std::numeric_limits<double>::infinity(), [](int) { return 0.; }, sink);
  check(calls == 0 && ledger.discarded == 0, "invalid clock/coverage refuses without consuming frames under fast-math");
  buffer.drain_physical(101., [](int) { return 0.; }, sink);
  check(calls == 1, "queued frame survives invalid clock refusal");
  buffer.push(frame(std::numeric_limits<double>::quiet_NaN(), {0}, 51));
  buffer.push(frame(101., {0,0}, 52));
  auto malformed = frame(102., {0,1}, 54); malformed.masks.pop_back();
  buffer.push(malformed);
  buffer.drain_physical(103., [](int) { return 0.; }, sink);
  check(buffer.count_drop_bogus() == 3, "nonfinite timestamps/duplicate cameras/misaligned component vectors are rejected");
  buffer.clear();
  ledger.exactly_once(6);
}
void observation_ownership() {
  ObservationLedger ledger;
  Buffer buffer(2, options(), [&](const CameraData &m) { ledger.take(m, true); });
  buffer.push(ledger.frame(300., {0}, 3000));
  buffer.push(ledger.frame(300.125, {1}, 3001));
  buffer.push(ledger.frame(300.25, {0,1}, 3002));
  std::vector<double> endpoints;
  buffer.drain_physical(301., [](int c) { return c == 0 ? .25 : 0.; },
      [&](std::vector<CameraData> &group, double endpoint) {
        endpoints.push_back(endpoint);
        if (endpoint == 300.25)
          check(group.size() == 2 && group[0].timestamp != group[1].timestamp,
                "observation cameras at one physical endpoint retain their distinct raw frame keys");
        for (const auto &message : group) ledger.take(message, false);
        return true;
      });
  check(endpoints == std::vector<double>({300.125,300.25,300.5}) && ledger.released == 4 && ledger.discarded == 0,
        "immutable observations follow the same physical ordering and packed splitting as images");
  buffer.push(ledger.frame(302., {0}, 3004));
  buffer.push(ledger.frame(302., {1}, 3005));
  buffer.drain_physical(303., [](int) { return 0.; }, [&](std::vector<CameraData> &group, double) {
    check(group.size() == 1 && group[0].observations.size() == 2, "same-raw observation owners bundle without deep copies");
    ledger.take(group[0], false); return true;
  });
  ledger.complete();
}
void finished_input(bool physical) {
  ObservationLedger ledger;
  auto o = options(); o.physical_order = physical; o.stale_factor = 100000.; o.guard = .002;
  Buffer buffer(2, o, [&](const CameraData &m) { ledger.take(m, true); });
  auto drain = [&](double newest) {
    if (physical)
      buffer.drain_physical(newest, [](int) { return 0.; }, [&](std::vector<CameraData> &group, double) {
        for (const auto &message : group) ledger.take(message, false);
        return true;
      });
    else
      buffer.drain(newest, [](const std::vector<int> &) { return 0.; }, [&](CameraData &&message) {
        ledger.take(message, false); return true;
      });
  };
  buffer.push(ledger.frame(310., {0}, 3100));
  buffer.push(ledger.frame(310.125, {1}, 3101));
  drain(311.);
  check(ledger.released == 1, "live unfinished camera holds its peer's unknown-future ordering");
  check(buffer.finish_camera(0) && !buffer.finish_camera(2), "logical camera EOF validates its camera index");
  drain(311.);
  check(ledger.released == 2, "finished camera releases its covered peer without wall-clock sleeping");
  check(!buffer.push(ledger.frame(310.25, {0}, 3102)) && buffer.count_drop_finished() == 1,
        "push after camera EOF is rejected and disposed exactly once");
  buffer.push(ledger.frame(311., {1}, 3103));
  buffer.finish_all();
  drain(310.75);
  check(ledger.released == 2 && ledger.discarded == 1, "EOF never substitutes for missing physical IMU coverage");
  buffer.discard_pending();
  check(ledger.discarded == 2, "terminal uncovered observation tail reaches the ordinary disposer");
  check(!buffer.push(ledger.frame(312., {1}, 3104)) && buffer.count_drop_finished() == 2,
        "discarding terminal tails does not reopen finished input");
  buffer.clear();
  check(buffer.push(ledger.frame(2., {1}, 3105)), "clear reopens finished cameras and admits a rewound replay");
  drain(3.);
  check(ledger.released == 3 && ledger.discarded == 3, "reopened replay preserves terminal accounting");
  ledger.complete();
}
void payload_shape_and_kind(bool physical) {
  ObservationLedger observations;
  Ledger images;
  auto o = options(); o.physical_order = physical;
  auto terminal = [&](const CameraData &message, bool dropped) {
    if (message.observations.empty()) images.take(message, dropped); else observations.take(message, dropped);
  };
  Buffer buffer(2, o, [&](const CameraData &m) { terminal(m, true); });
  auto malformed = observations.frame(320., {0}, 3200);
  malformed.observations.push_back(nullptr);
  check(!buffer.push(malformed), "misaligned observation payload vector is refused before enqueue");
  malformed = CameraData(); malformed.timestamp = 320.; malformed.sensor_ids = {1}; malformed.observations = {nullptr};
  check(!buffer.push(malformed) && buffer.count_drop_bogus() == 2, "null observation payload is refused before enqueue");
  malformed = CameraData();
  buffer.push(frame(321., {0}, 3201));
  buffer.push(observations.frame(321., {1}, 3202));
  size_t calls = 0;
  if (physical)
    buffer.drain_physical(322., [](int) { return 0.; }, [&](std::vector<CameraData> &group, double) {
      check(group.size() == 2, "physical group keeps image and observation message kinds separate");
      for (const auto &message : group) { ++calls; terminal(message, false); }
      return true;
    });
  else
    buffer.drain(322., [](const std::vector<int> &) { return 0.; }, [&](CameraData &&message) {
      ++calls; terminal(message, false); return true;
    });
  check(calls == 2 && images.released == 1 && observations.released == 1,
        "same raw instant with different payload kinds never creates mixed/null observation slots");
  observations.complete(); images.exactly_once(1);
}
void finished_pause_and_packed_rejection() {
  ObservationLedger ledger;
  Buffer buffer(2, options(), [&](const CameraData &m) { ledger.take(m, true); });
  buffer.push(ledger.frame(330., {0,1}, 3300));
  buffer.push(ledger.frame(330.25, {0,1}, 3302));
  buffer.finish_camera(1);
  check(!buffer.push(ledger.frame(330.5, {0,1}, 3304)) && buffer.count_drop_finished() == 1,
        "a packed push containing a finished camera is rejected as one complete owned message");
  buffer.finish_all();
  size_t calls = 0;
  buffer.drain_physical(331., [](int c) { return c == 0 ? 0. : .125; },
      [&](std::vector<CameraData> &group, double endpoint) {
        ++calls; check(endpoint == 330., "EOF pause consumes the earliest covered group only");
        for (const auto &message : group) ledger.take(message, false);
        return false;
      });
  check(calls == 1 && ledger.released == 1 && ledger.discarded == 2,
        "EOF honors deliberate sink pause even when later groups have complete IMU coverage");
  buffer.discard_pending();
  check(ledger.discarded == 5, "explicit terminal disposal also reports covered frames left by a deliberate pause");
  ledger.complete();
}
void threaded_producers(bool force_early_eof) {
  // This checks recorded ordering under arbitrary host scheduling. A live
  // stream may legitimately expire after a wall-clock stall, and a producer
  // may fill its bounded ring while the consumer is descheduled. Neither is
  // a zero-drop contract. Fit the entire finite fixture and use explicit EOF;
  // independent deterministic tests above cover live expiry and overflow.
  constexpr int frames_per_camera = 100;
  auto o = options(frames_per_camera + 1);
  std::atomic<size_t> disposed{0};
  Buffer buffer(2, o, [&](const CameraData &m) { disposed.fetch_add(m.images.size()); });
  buffer.prepare_recorded_input();
  std::atomic<int> ready{0}, done{0};
  std::atomic<bool> first_finished{false};
  auto producer = [&](int camera) {
    for (int k = 0; k < frames_per_camera; ++k) {
      if (force_early_eof && camera == 1 && k == 1)
        while (!first_finished.load()) std::this_thread::yield();
      buffer.push(frame(200. + .03125*k + .015625*camera, {camera}, 1000 + 200*camera + k));
      if (k == 0) {
        ready.fetch_add(1);
        while (ready.load() < 2) std::this_thread::yield();
      }
      std::this_thread::sleep_for(std::chrono::microseconds(150 + ((k+camera)%4)*30));
    }
    buffer.finish_camera(camera);
    if (camera == 0) first_finished.store(true);
    done.fetch_add(1);
  };
  std::thread camera0(producer, 0), camera1(producer, 1);
  while (ready.load() < 2) std::this_thread::yield();
  Ledger ledger;
  double previous = -1.;
  auto lookup = [](int camera) { return camera == 0 ? .0078125 : -.00390625; };
  auto sink = [&](std::vector<CameraData> &group, double endpoint) {
    check(endpoint > previous, "threaded physical release order is strict");
    previous = endpoint;
    for (auto &m : group) ledger.take(m, false);
    return true;
  };
  while (done.load() < 2) {
    buffer.drain_physical(210., lookup, sink);
    std::this_thread::sleep_for(std::chrono::microseconds(70));
  }
  camera0.join(); camera1.join();
  buffer.drain_physical(210., lookup, sink);
  check(disposed.load() == 0 && buffer.count_drop_late() == 0 && buffer.count_drop_full() == 0,
        "threaded recorded producers and early EOF lose no physical views");
  check(buffer.count_physical_views() == 2*frames_per_camera, "threaded physical view count equals every input frame");
  ledger.exactly_once(2*frames_per_camera);
}
} // namespace
int main() {
  recorded_input_ordering();
  ordering_and_grouping();
  packed_backlog();
  mixed_producer_heads();
  full_queue_disjoint_head();
  bounded_invalid_replenishment();
  clock_updates_and_late_raw();
  coverage_stale_and_pause();
  bounded_disposal_and_reset_in_callback();
  invalid_clocks_and_messages();
  observation_ownership();
  for (bool physical : {false, true}) { finished_input(physical); payload_shape_and_kind(physical); }
  finished_pause_and_packed_rejection();
  threaded_producers(false);
  threaded_producers(true);
  std::printf("PHYSICAL_CAMERA_BUFFER %s checks=%zu failures=%d\n", failures ? "FAIL" : "PASS", checks, failures);
  return failures ? 1 : 0;
}
