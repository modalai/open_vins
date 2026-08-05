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

#include "utils/colors.h"
#include "utils/print.h"

using namespace ov_msckf;

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
  counter_pushed.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void AsyncCameraBuffer::drain(double newest_imu_time, const DtMaxFn &dt_max_for_ids, const SinkFn &sink) {

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
      const double last_ts = s.last_enqueued_ts.load(std::memory_order_acquire);
      if (last_ts < 0) {
        continue; // never alive
      }
      if (last_ts >= t_head) {
        continue; // its >= t_head frame is in flight into the ring; we will see it next drain
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
      if (!s.has_staged || sp.get() == head || s.staged.timestamp != t_head) {
        continue;
      }
      out.sensor_ids.insert(out.sensor_ids.end(), s.staged.sensor_ids.begin(), s.staged.sensor_ids.end());
      out.images.insert(out.images.end(), s.staged.images.begin(), s.staged.images.end());
      out.masks.insert(out.masks.end(), s.staged.masks.begin(), s.staged.masks.end());
      out.exposures.insert(out.exposures.end(), s.staged.exposures.begin(), s.staged.exposures.end());
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

void AsyncCameraBuffer::clear() {
  for (auto &sp : streams) {
    Stream &s = *sp;
    if (s.has_staged) {
      if (dispose)
        dispose(s.staged);
      s.has_staged = false;
    }
    ov_core::CameraData msg;
    while (s.ring.pop(msg)) {
      if (dispose)
        dispose(msg);
    }
  }
  last_released_ts = -1.0;
}
