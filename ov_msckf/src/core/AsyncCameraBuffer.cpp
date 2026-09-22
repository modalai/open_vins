/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
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


#include "AsyncCameraBuffer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iterator>
#include <stdexcept>

#include "utils/colors.h"
#include "utils/print.h"

using namespace ov_msckf;

namespace {
// This project builds with fast-math: inspect the IEEE exponent instead of a
// floating-point isfinite expression that the compiler may assume is true.
bool finite_clock(double value) {
  uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value), "64-bit camera clock");
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

template <typename T> void take_component(std::vector<T> &source, size_t index, std::vector<T> &dest) {
  if (!source.empty()) {
    dest.push_back(std::move(source[index]));
    source.erase(source.begin() + index);
  }
}

// Move ownership of exactly one camera, including its external image handles.
// The remaining packed source is still owned by the buffer.
void take_camera(ov_core::CameraData &source, size_t index, ov_core::CameraData &dest) {
  if (source.sensor_ids.size() == 1) {
    dest = std::move(source);
    source = ov_core::CameraData();
    return;
  }
  dest = ov_core::CameraData();
  dest.timestamp = source.timestamp;
  dest.sync_timestamp_ns = source.sync_timestamp_ns;
  dest.sensor_ids.push_back(source.sensor_ids[index]);
  source.sensor_ids.erase(source.sensor_ids.begin() + index);
  take_component(source.images, index, dest.images);
  take_component(source.masks, index, dest.masks);
  take_component(source.exposures, index, dest.exposures);
  take_component(source.observations, index, dest.observations);
#if HAVE_OPENCL
  take_component(source.img_frames, index, dest.img_frames);
  take_component(source.cl_images, index, dest.cl_images);
#endif
}

template <typename T> void append_components(std::vector<T> &dest, std::vector<T> &source) {
  dest.insert(dest.end(), std::make_move_iterator(source.begin()), std::make_move_iterator(source.end()));
  source.clear();
}

template <typename T>
void append_optional_components(std::vector<T> &dest, std::vector<T> &source, size_t old_count, size_t added_count) {
  if (dest.empty() && source.empty())
    return;
  if (dest.empty())
    dest.resize(old_count);
  if (source.empty())
    source.resize(added_count);
  append_components(dest, source);
}

void append_cameras(ov_core::CameraData &dest, ov_core::CameraData &source) {
  const size_t old_count = dest.sensor_ids.size(), added_count = source.sensor_ids.size();
  append_optional_components(dest.images, source.images, old_count, added_count);
  append_optional_components(dest.masks, source.masks, old_count, added_count);
  append_optional_components(dest.exposures, source.exposures, old_count, added_count);
  append_optional_components(dest.observations, source.observations, old_count, added_count);
#if HAVE_OPENCL
  append_optional_components(dest.img_frames, source.img_frames, old_count, added_count);
  append_optional_components(dest.cl_images, source.cl_images, old_count, added_count);
#endif
  append_components(dest.sensor_ids, source.sensor_ids);
}
} // namespace

double AsyncCameraBuffer::mono_now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

AsyncCameraBuffer::AsyncCameraBuffer(int num_cams, Options options, Disposer disposer) : opts(options), dispose(std::move(disposer)) {
  streams.reserve(std::max(1, num_cams));
  for (int i = 0; i < std::max(1, num_cams); i++) {
    streams.emplace_back(std::make_unique<Stream>(opts.ring_capacity));
    // Declared nominal period seeds the staleness EMA (the stream's own cadence replaces it
    // after two frames: the producer EMA blends 90/10 from this starting point)
    if ((size_t)i < opts.initial_periods.size() && opts.initial_periods[i] > 0) {
      streams.back()->ema_period.store(opts.initial_periods[i], std::memory_order_relaxed);
    }
  }
  if (opts.sync_reference_camera >= 0) {
    if (static_cast<size_t>(opts.sync_reference_camera) >= streams.size())
      throw std::invalid_argument("forced-sync reference camera is out of range");
    synchronized_source = std::make_unique<Stream>(1);
    sync_matches.reserve(streams.size());
    sync_present.resize(streams.size());
  }
  if (opts.physical_order) {
    physical_cameras.reserve(streams.size());
    physical_offsets.resize(streams.size());
    physical_group.reserve(streams.size());
    for (size_t i = 0; i < streams.size(); ++i) {
      physical_cameras.emplace_back(std::make_unique<PhysicalCamera>(std::max<size_t>(1, opts.ring_capacity)));
      if (i < opts.initial_periods.size() && opts.initial_periods[i] > 0)
        physical_cameras.back()->ema_period.store(opts.initial_periods[i], std::memory_order_relaxed);
    }
  }
}

bool AsyncCameraBuffer::push(const ov_core::CameraData &msg) {

  // Ring selected by the first sensor id (a multi-sensor stereo message comes from ONE pipe
  // thread, so it stays single-producer on that ring)
  if (msg.sensor_ids.empty() || msg.sensor_ids.front() < 0 || (size_t)msg.sensor_ids.front() >= streams.size()) {
    if (dispose)
      dispose(msg);
    counter_drop_bogus.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (!msg.observations.empty() &&
      (msg.observations.size() != msg.sensor_ids.size() ||
       std::any_of(msg.observations.begin(), msg.observations.end(), [](const auto &payload) { return !payload; }))) {
    if (dispose)
      dispose(msg);
    counter_drop_bogus.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  for (int camera : msg.sensor_ids) {
    if (camera >= 0 && static_cast<size_t>(camera) < streams.size() &&
        streams[camera]->finished.load(std::memory_order_acquire)) {
      if (dispose)
        dispose(msg);
      counter_drop_finished.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
  }
  Stream &s = *streams.at(msg.sensor_ids.front());

  // Producer-side stream meta (single writer): EMA of the frame period + arrival time
  if (s.prev_push_ts > 0 && msg.timestamp > s.prev_push_ts) {
    double period = msg.timestamp - s.prev_push_ts;
    double ema = s.ema_period.load(std::memory_order_relaxed);
    s.ema_period.store((ema > 0) ? (0.9 * ema + 0.1 * period) : period, std::memory_order_relaxed);
  }
  s.prev_push_ts = msg.timestamp;

  if (!s.ring.push(msg)) {
    // Consumer stalled for > ring_capacity frames: drop THIS frame (the ring already holds the
    // older, more releasable ones; the consumer realizes drop-oldest semantics on catch-up)
    if (dispose)
      dispose(msg);
    counter_drop_full.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  // Publish meta AFTER the frame is visible so the consumer never waits on a frame that is not there
  s.last_enqueued_ts.store(msg.timestamp, std::memory_order_release);
  s.last_push_mono.store(mono_now(), std::memory_order_release);
  if (opts.physical_order && finite_clock(msg.timestamp)) {
    const double pushed_at = mono_now();
    for (const int camera : msg.sensor_ids) {
      if (camera < 0 || static_cast<size_t>(camera) >= physical_cameras.size())
        continue; // the physical consumer rejects the complete malformed message
      auto &meta = *physical_cameras[camera];
      const double previous = meta.last_enqueued_raw.load(std::memory_order_relaxed);
      if (finite_clock(previous) && msg.timestamp > previous) {
        const double period = msg.timestamp - previous;
        const double ema = meta.ema_period.load(std::memory_order_relaxed);
        meta.ema_period.store(ema > 0 ? .9 * ema + .1 * period : period, std::memory_order_relaxed);
      }
      meta.last_enqueued_raw.store(msg.timestamp, std::memory_order_release);
      meta.last_push_mono.store(pushed_at, std::memory_order_release);
    }
  }
  counter_pushed.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool AsyncCameraBuffer::stage_synchronized(double newest_imu_time, size_t &pop_budget) {
  if (synchronized_source->has_staged)
    return true;
  const double now = mono_now();
  auto order_time = [](const ov_core::CameraData &m) {
    return m.sync_timestamp_ns >= 0 ? 1e-9 * static_cast<double>(m.sync_timestamp_ns) : m.timestamp;
  };
  auto earlier = [&](const ov_core::CameraData &a, const ov_core::CameraData &b) {
    if (a.sync_timestamp_ns >= 0 && b.sync_timestamp_ns >= 0)
      return a.sync_timestamp_ns < b.sync_timestamp_ns;
    return order_time(a) < order_time(b);
  };
  auto same_trigger = [](const ov_core::CameraData &a, const ov_core::CameraData &b) {
    if (a.sync_timestamp_ns >= 0 || b.sync_timestamp_ns >= 0)
      return a.sync_timestamp_ns >= 0 && a.sync_timestamp_ns == b.sync_timestamp_ns;
    return a.timestamp == b.timestamp;
  };
  auto discard = [&](Stream &s, std::atomic<uint64_t> &counter) {
    if (dispose)
      dispose(s.staged);
    s.staged = ov_core::CameraData();
    s.has_staged = false;
    counter.fetch_add(1, std::memory_order_relaxed);
  };

  while (true) {
    Stream *head = nullptr;
    for (auto &sp : streams) {
      auto &s = *sp;
      while (!s.has_staged && pop_budget > 0 && s.ring.pop(s.staged)) {
        --pop_budget;
        s.has_staged = true;
        s.staged_since = now;
        const auto &m = s.staged;
        const size_t n = m.sensor_ids.size();
        auto aligned = [n](size_t size) { return size == 0 || size == n; };
        bool valid = n > 0 && finite_clock(m.timestamp) && m.sync_timestamp_ns >= -1 &&
                     aligned(m.images.size()) && aligned(m.masks.size()) && aligned(m.exposures.size()) &&
                     aligned(m.observations.size());
#if HAVE_OPENCL
        valid = valid && aligned(m.img_frames.size()) && aligned(m.cl_images.size());
#endif
        for (size_t i = 0; i < n; ++i)
          valid = valid && m.sensor_ids[i] >= 0 && static_cast<size_t>(m.sensor_ids[i]) < streams.size() &&
                  std::find(m.sensor_ids.begin(), m.sensor_ids.begin() + i, m.sensor_ids[i]) == m.sensor_ids.begin() + i;
        if (!valid || m.timestamp > newest_imu_time + opts.bogus_future) {
          discard(s, counter_drop_bogus);
          continue;
        }
        const bool late = m.sync_timestamp_ns >= 0 ? m.sync_timestamp_ns <= last_sync_start_ns
                                                  : m.timestamp <= last_sync_timestamp;
        if (late)
          discard(s, counter_drop_late);
      }
      if (s.has_staged && (!head || earlier(s.staged, head->staged)))
        head = &s;
    }
    if (!head)
      return false;

    sync_matches.clear();
    std::fill(sync_present.begin(), sync_present.end(), false);
    const ov_core::CameraData *reference = nullptr;
    bool valid = true;
    for (auto &sp : streams) {
      if (!sp->has_staged || !same_trigger(sp->staged, head->staged))
        continue;
      sync_matches.push_back(sp.get());
      const auto &m = sp->staged;
      valid = valid && m.observations.empty() == head->staged.observations.empty();
      for (int camera : m.sensor_ids) {
        valid = valid && !sync_present[camera];
        sync_present[camera] = true;
        if (camera == opts.sync_reference_camera)
          reference = &m;
      }
    }
    if (!valid) {
      for (auto *s : sync_matches)
        discard(*s, counter_drop_bogus);
      continue;
    }

    bool complete = true, impossible = false;
    for (size_t camera = 0; camera < streams.size(); ++camera) {
      if (sync_present[camera])
        continue;
      complete = false;
      const auto &s = *streams[camera];
      // A staged later trigger proves this group lost a mate. Otherwise allow
      // a producer that landed just after staging to be seen on the next drain.
      if (s.has_staged) {
        impossible = true;
        continue;
      }
      // Observe EOF before checking the ring: its release follows the final
      // push, so a finished producer's last burst must still be staged.
      const bool finished = s.finished.load(std::memory_order_acquire);
      if (s.ring.read_available() != 0)
        return false;
      if (finished) {
        impossible = true;
        continue;
      }
      const double period = s.ema_period.load(std::memory_order_relaxed);
      if (!recorded_input && now - head->staged_since >= opts.stale_factor * (period > 0 ? period : .05))
        impossible = true;
    }
    if (!complete) {
      if (!impossible)
        return false;
      for (auto *s : sync_matches)
        discard(*s, counter_drop_unpaired);
      continue;
    }

    // Membership came from the hardware trigger, never from exposure length or
    // callback order. Keep camera order and each exposure/handle unchanged.
    const double timestamp = reference->timestamp;
    if (timestamp <= last_sync_timestamp) {
      for (auto *s : sync_matches)
        discard(*s, counter_drop_late);
      continue;
    }
    auto &out = synchronized_source->staged;
    out = std::move(sync_matches.front()->staged);
    sync_matches.front()->has_staged = false;
    for (size_t i = 1; i < sync_matches.size(); ++i) {
      append_cameras(out, sync_matches[i]->staged);
      sync_matches[i]->staged = ov_core::CameraData();
      sync_matches[i]->has_staged = false;
      counter_bundled.fetch_add(1, std::memory_order_relaxed);
    }
    out.timestamp = timestamp;
    last_sync_timestamp = timestamp;
    if (out.sync_timestamp_ns >= 0)
      last_sync_start_ns = out.sync_timestamp_ns;
    synchronized_source->has_staged = true;
    return true;
  }
}

void AsyncCameraBuffer::drain(double newest_imu_time, const DtMaxFn &dt_max_for_ids, const SinkFn &sink) {

  if (synchronized_source) {
    if (!finite_clock(newest_imu_time))
      return;
    size_t pop_budget = 0;
    for (const auto &source : streams)
      pop_budget += source->ring.read_available();
    while (stage_synchronized(newest_imu_time, pop_budget)) {
      auto &source = *synchronized_source;
      const double offset = dt_max_for_ids(source.staged.sensor_ids);
      if (!finite_clock(offset) || source.staged.timestamp + offset + opts.guard > newest_imu_time)
        return;
      ov_core::CameraData out = std::move(source.staged);
      source.staged = ov_core::CameraData();
      source.has_staged = false;
      last_released_ts = out.timestamp;
      counter_released.fetch_add(1, std::memory_order_relaxed);
      if (!sink(std::move(out)))
        return;
    }
    return;
  }

  const double now = mono_now();

  while (true) {

    // Fill every empty staging slot from its ring, discarding frames that can no longer be
    // released in order (late) or carry broken timestamps (far future)
    for (auto &sp : streams) {
      Stream &s = *sp;
      while (!s.has_staged && s.ring.pop(s.staged)) {
        if (s.staged.timestamp <= last_released_ts && last_released_ts >= 0) {
          if (dispose)
            dispose(s.staged);
          counter_drop_late.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        if (s.staged.timestamp > newest_imu_time + opts.bogus_future) {
          if (dispose)
            dispose(s.staged);
          counter_drop_bogus.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        s.has_staged = true;
      }
    }

    // Find the globally-earliest staged frame
    Stream *head = nullptr;
    for (auto &sp : streams) {
      if (sp->has_staged && (head == nullptr || sp->staged.timestamp < head->staged.timestamp)) {
        head = sp.get();
      }
    }
    if (head == nullptr) {
      return;
    }
    const double t_head = head->staged.timestamp;

    // IMU gate: enough inertial data must exist past the frame's sampling instant
    if (t_head + dt_max_for_ids(head->staged.sensor_ids) + opts.guard > newest_imu_time) {
      return;
    }

    // Ordering gate: every other LIVE camera must have shown its hand (staged frame at or after
    // t_head), or be stale (silent > stale_factor * its EMA period) so it cannot block. A camera
    // that never pushed does not block either (startup, disabled stream).
    for (auto &sp : streams) {
      Stream &s = *sp;
      if (&s == head || s.has_staged) {
        // staged.timestamp >= t_head by construction (head is the minimum staged time)
        continue;
      }
      const bool finished = s.finished.load(std::memory_order_acquire);
      const double last_ts = s.last_enqueued_ts.load(std::memory_order_acquire);
      // The producer may have filled an empty ring after this drain's staging
      // pass. Its newest timestamp says nothing about the OLDEST queued frame.
      // Check the ring after acquiring both publication/EOF metadata so a final
      // burst cannot be skipped either. Yield once; the next drain stages its
      // actual head. This never chases a concurrent producer in a retry loop.
      if (s.ring.read_available() != 0) {
        if (last_ts == t_head)
          counter_held_pair.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      if (finished)
        continue;
      if (last_ts < 0) {
        continue; // never alive
      }
      if (last_ts > t_head) {
        continue; // this camera has only LATER frames; no bit-equal partner for t_head is coming,
                  // so releasing the head now cannot orphan a same-timestamp mate.
      }
      if (last_ts == t_head) {
        // Different payload kinds stay separate at the same raw instant. A
        // source already consumed at this instant is not an in-flight mate.
        if (last_released_ts == t_head)
          continue;
        // The mate's SAME-timestamp frame is already in this stream's ring but was not staged this
        // drain (it landed just after the staging step -- the "split across iterations" case). HOLD
        // one drain so it stages and BUNDLES with the head, instead of releasing the head unpaired
        // (feed_new_camera drops single-cam frames in stereo mode -> [HEALTH] Dropped camera frame).
        // Bounded: next drain pops it (push publishes last_enqueued_ts only AFTER the ring push, so
        // last_ts==t_head guarantees the frame is present). Restores the CameraQueueFusion re-pair fix.
        counter_held_pair.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      const double last_mono = s.last_push_mono.load(std::memory_order_acquire);
      double period = s.ema_period.load(std::memory_order_relaxed);
      period = (period > 0) ? period : 0.05; // conservative 20 Hz default before the EMA settles
      if (now - last_mono < opts.stale_factor * period) {
        return; // live camera may still deliver an older frame: hold ordering
      }
      // stale: don't block on it (it re-arms on its next arrival)
    }

    // Bundle every staged frame with a bit-equal timestamp into one multi-sensor message
    ov_core::CameraData out = std::move(head->staged);
    head->has_staged = false;
    for (auto &sp : streams) {
      Stream &s = *sp;
      if (!s.has_staged || sp.get() == head || s.staged.timestamp != t_head ||
          s.staged.observations.empty() != out.observations.empty()) {
        continue;
      }
      out.sensor_ids.insert(out.sensor_ids.end(), s.staged.sensor_ids.begin(), s.staged.sensor_ids.end());
      out.images.insert(out.images.end(), s.staged.images.begin(), s.staged.images.end());
      out.masks.insert(out.masks.end(), s.staged.masks.begin(), s.staged.masks.end());
      out.exposures.insert(out.exposures.end(), s.staged.exposures.begin(), s.staged.exposures.end());
      append_components(out.observations, s.staged.observations);
#if HAVE_OPENCL
      out.img_frames.insert(out.img_frames.end(), s.staged.img_frames.begin(), s.staged.img_frames.end());
      out.cl_images.insert(out.cl_images.end(), s.staged.cl_images.begin(), s.staged.cl_images.end());
#endif
      s.has_staged = false;
      counter_bundled.fetch_add(1, std::memory_order_relaxed);
    }

    last_released_ts = t_head;
    counter_released.fetch_add(1, std::memory_order_relaxed);
    if (!sink(std::move(out))) {
      return; // consumer asked to pause (e.g. reset in progress)
    }
  }
}

void AsyncCameraBuffer::drain_physical(double newest_imu_time, const DtFn &dt_for_camera, const GroupSinkFn &sink) {
  if (!opts.physical_order || !finite_clock(newest_imu_time))
    return;

  // Do not chase a producer indefinitely if it keeps replacing malformed
  // inputs while we drain. Already-staged partial messages need no extra pop.
  size_t pop_budget = 0;
  for (const auto &source : streams)
    pop_budget += source->ring.read_available();

  while (true) {
    // One clock snapshot owns selection, coverage and every member of the group.
    // A callback can update calibration; the next loop samples the new values.
    for (size_t c = 0; c < physical_cameras.size(); ++c) {
      physical_offsets[c] = dt_for_camera(static_cast<int>(c));
      if (!finite_clock(physical_offsets[c]))
        return;
    }

    // Merge the producer heads by RAW time before splitting them. A later
    // packed message in ring 0 may contain camera 1: exhausting ring 0 first
    // would incorrectly mark an earlier camera-1 frame in ring 1 as late.
    // Partial packed messages retain ownership in their existing source slot.
    // Skip a blocked source only for this pass, so full queues apply bounded
    // backpressure without blocking an unrelated camera's available head.
    for (auto &sp : streams) {
      sp->physical_stage_blocked = false;
    }
    if (synchronized_source)
      synchronized_source->physical_stage_blocked = false;
    while (true) {
      Stream *head = nullptr;
      if (synchronized_source) {
        if (!synchronized_source->physical_stage_blocked && stage_synchronized(newest_imu_time, pop_budget))
          head = synchronized_source.get();
      } else {
        for (auto &sp : streams) {
          if (!sp->has_staged && pop_budget > 0 && sp->ring.pop(sp->staged)) {
            --pop_budget;
            sp->has_staged = true;
          }
          if (!sp->has_staged || sp->physical_stage_blocked)
            continue;
          if (head == nullptr || (!finite_clock(sp->staged.timestamp) && finite_clock(head->staged.timestamp)) ||
              (finite_clock(sp->staged.timestamp) && finite_clock(head->staged.timestamp) &&
               sp->staged.timestamp < head->staged.timestamp))
            head = sp.get();
        }
      }
      if (head == nullptr)
        break;
      auto &source = *head;
      auto &msg = source.staged;
      const size_t cameras = msg.sensor_ids.size();
      auto aligned = [cameras](size_t n) { return n == 0 || n == cameras; };
      bool valid = cameras > 0 && finite_clock(msg.timestamp) && aligned(msg.images.size()) &&
                   aligned(msg.masks.size()) && aligned(msg.exposures.size()) && aligned(msg.observations.size());
#if HAVE_OPENCL
      valid = valid && aligned(msg.img_frames.size()) && aligned(msg.cl_images.size());
#endif
      for (size_t i = 0; i < cameras; ++i) {
        const int camera = msg.sensor_ids[i];
        valid = valid && camera >= 0 && static_cast<size_t>(camera) < physical_cameras.size() &&
                std::find(msg.sensor_ids.begin(), msg.sensor_ids.begin() + i, camera) == msg.sensor_ids.begin() + i;
      }
      if (!valid) {
        if (finite_clock(msg.timestamp))
          for (const int camera : msg.sensor_ids)
            if (camera >= 0 && static_cast<size_t>(camera) < physical_cameras.size())
              physical_cameras[camera]->last_seen_raw = std::max(physical_cameras[camera]->last_seen_raw, msg.timestamp);
        if (dispose)
          dispose(msg);
        msg = ov_core::CameraData();
        source.has_staged = false;
        counter_drop_bogus.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      for (size_t i = 0; i < msg.sensor_ids.size();) {
        const int camera = msg.sensor_ids[i];
        auto &queue = *physical_cameras[camera];
        const double raw = msg.timestamp;
        const double physical = raw + physical_offsets[camera];
        const bool late = raw <= queue.last_staged_raw;
        const bool bogus = !finite_clock(physical) || physical > newest_imu_time + opts.bogus_future;
        if (late || bogus) {
          ov_core::CameraData dropped;
          take_camera(msg, i, dropped);
          queue.last_seen_raw = std::max(queue.last_seen_raw, raw);
          if (dispose)
            dispose(dropped);
          (late ? counter_drop_late : counter_drop_bogus).fetch_add(1, std::memory_order_relaxed);
        } else if (queue.size < queue.pending.size()) {
          auto &slot = queue.pending[(queue.begin + queue.size) % queue.pending.size()];
          take_camera(msg, i, slot);
          ++queue.size;
          queue.last_staged_raw = raw;
          queue.last_seen_raw = std::max(queue.last_seen_raw, raw);
        } else {
          ++i; // this camera's bounded queue is full; keep its component staged
        }
      }
      if (!msg.sensor_ids.empty())
        source.physical_stage_blocked = true;
      else
        source.has_staged = false;
    }

    // Clock updates may move previously staged views behind the consumed
    // endpoint. Never retimestamp them or apply them at a later physical pose.
    bool discarded = false;
    for (size_t c = 0; c < physical_cameras.size(); ++c) {
      auto &queue = *physical_cameras[c];
      while (queue.size > 0) {
        auto &front = queue.pending[queue.begin];
        const double physical = front.timestamp + physical_offsets[c];
        const bool late = physical <= last_released_physical;
        const bool bogus = !finite_clock(physical) || physical > newest_imu_time + opts.bogus_future;
        if (!late && !bogus)
          break;
        if (dispose)
          dispose(front);
        front = ov_core::CameraData();
        queue.begin = (queue.begin + 1) % queue.pending.size();
        --queue.size;
        discarded = true;
        (late ? counter_drop_late : counter_drop_bogus).fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (discarded)
      continue; // refill freed queues before making an ordering decision

    double endpoint = std::numeric_limits<double>::infinity();
    for (size_t c = 0; c < physical_cameras.size(); ++c) {
      const auto &queue = *physical_cameras[c];
      if (queue.size > 0)
        endpoint = std::min(endpoint, queue.pending[queue.begin].timestamp + physical_offsets[c]);
    }
    if (!finite_clock(endpoint) || endpoint + opts.guard > newest_imu_time)
      return;

    // Every live logical camera must either expose its next physical endpoint,
    // prove it has advanced past this one, or become stale. This also covers
    // cameras sharing one packed-message producer ring.
    const double now = mono_now();
    for (size_t c = 0; c < physical_cameras.size(); ++c) {
      const auto &queue = *physical_cameras[c];
      if (queue.size > 0)
        continue;
      const bool finished = streams[c]->finished.load(std::memory_order_acquire);
      // Forced-sync input is published here only after a complete group adopted
      // its reference timestamp. Individual producer midpoints are not ordering
      // watermarks in that policy.
      const double last_raw = synchronized_source ? last_sync_timestamp
                                                  : queue.last_enqueued_raw.load(std::memory_order_acquire);
      if (!finite_clock(last_raw)) {
        if (recorded_input && !finished)
          return; // a not-yet-fed recording camera may own an earlier physical instant
        continue;
      }
      if (last_raw > queue.last_seen_raw) {
        // Published after the staging pass. Stage it on the next drain before
        // releasing an endpoint that might have a same-time or earlier mate.
        if (last_raw + physical_offsets[c] == endpoint)
          counter_held_pair.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      if (synchronized_source && pop_budget == 0) {
        // A producer may publish its final burst after our bounded snapshot.
        // Normalize that burst before EOF can waive physical-time ordering.
        for (const auto &source : streams) {
          static_cast<void>(source->finished.load(std::memory_order_acquire));
          if (source->ring.read_available() != 0)
            return;
        }
      }
      // EOF ends future arrivals; it does not discard the producer's final
      // queued burst when that burst landed after the bounded staging snapshot.
      if (finished)
        continue;
      if (last_raw + physical_offsets[c] >= endpoint)
        continue;
      if (recorded_input)
        return; // offline compute latency is not evidence of a missing camera
      double period = queue.ema_period.load(std::memory_order_relaxed);
      period = period > 0 ? period : .05;
      if (now - queue.last_push_mono.load(std::memory_order_acquire) < opts.stale_factor * period)
        return;
    }

    physical_group.clear();
    size_t views = 0;
    for (size_t c = 0; c < physical_cameras.size(); ++c) {
      auto &queue = *physical_cameras[c];
      if (queue.size == 0 || queue.pending[queue.begin].timestamp + physical_offsets[c] != endpoint)
        continue;
      auto &frame = queue.pending[queue.begin];
      auto same_raw = std::find_if(physical_group.begin(), physical_group.end(),
                                   [&](const ov_core::CameraData &other) {
                                     return other.timestamp == frame.timestamp &&
                                            other.observations.empty() == frame.observations.empty();
                                   });
      if (same_raw == physical_group.end()) {
        physical_group.push_back(std::move(frame));
      } else {
        append_cameras(*same_raw, frame);
        counter_bundled.fetch_add(1, std::memory_order_relaxed);
      }
      frame = ov_core::CameraData();
      queue.begin = (queue.begin + 1) % queue.pending.size();
      --queue.size;
      ++views;
    }
    last_released_physical = endpoint;
    counter_released.fetch_add(1, std::memory_order_relaxed);
    counter_physical_views.fetch_add(views, std::memory_order_relaxed);
    physical_in_sink = true;
    const bool keep_draining = sink(physical_group, endpoint);
    physical_in_sink = false;
    physical_group.clear(); // callback owns all handles; never dispose again
    if (!keep_draining)
      return;
  }
}

void AsyncCameraBuffer::prepare_recorded_input() {
  if ((!opts.physical_order && !synchronized_source) || count_pushed() != 0)
    throw std::logic_error("recorded camera ordering requires physical or synchronized mode before camera input");
  recorded_input = true;
}

bool AsyncCameraBuffer::finish_camera(size_t camera_id) {
  if (camera_id >= streams.size())
    return false;
  streams[camera_id]->finished.store(true, std::memory_order_release);
  return true;
}

void AsyncCameraBuffer::finish_all() {
  for (const auto &stream : streams)
    stream->finished.store(true, std::memory_order_release);
}

void AsyncCameraBuffer::discard_pending() {
  if (synchronized_source && synchronized_source->has_staged) {
    if (dispose)
      dispose(synchronized_source->staged);
    synchronized_source->staged = ov_core::CameraData();
    synchronized_source->has_staged = false;
  }
  for (auto &sp : streams) {
    Stream &s = *sp;
    if (s.has_staged) {
      if (dispose)
        dispose(s.staged);
      s.staged = ov_core::CameraData();
      s.has_staged = false;
    }
    ov_core::CameraData msg;
    while (s.ring.pop(msg)) {
      if (dispose)
        dispose(msg);
    }
  }
  for (auto &camera : physical_cameras) {
    auto &queue = *camera;
    while (queue.size > 0) {
      auto &front = queue.pending[queue.begin];
      if (dispose)
        dispose(front);
      front = ov_core::CameraData();
      queue.begin = (queue.begin + 1) % queue.pending.size();
      --queue.size;
    }
    queue.begin = 0;
  }
  if (!physical_in_sink)
    physical_group.clear();
}

void AsyncCameraBuffer::clear() {
  discard_pending();
  for (auto &stream : streams)
    stream->finished.store(false, std::memory_order_release);
  last_released_ts = -1.0;
  last_sync_start_ns = -1;
  last_sync_timestamp = -std::numeric_limits<double>::infinity();
  for (auto &camera : physical_cameras) {
    camera->last_staged_raw = -std::numeric_limits<double>::infinity();
    if (recorded_input) {
      // Recorded producers are quiescent at reset/rewind. Their previous
      // episode's high-water mark cannot prove that the new segment has
      // already delivered an earlier view. Do not reset live producer-owned
      // publication metadata: those producers may still be running.
      camera->last_enqueued_raw.store(-std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
      camera->last_push_mono.store(-1., std::memory_order_relaxed);
    }
    camera->last_seen_raw = camera->last_enqueued_raw.load(std::memory_order_acquire);
  }
  last_released_physical = -std::numeric_limits<double>::infinity();
}
