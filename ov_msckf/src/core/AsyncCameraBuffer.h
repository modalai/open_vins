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

#ifndef OV_MSCKF_ASYNC_CAMERA_BUFFER_H
#define OV_MSCKF_ASYNC_CAMERA_BUFFER_H

/**
 * @author Joao Leonardo Silva Cotta (@zauberflote1)
 *
 * Lock-free multi-camera ingest for asynchronous (unsynced) camera streams.
 *
 * One wait-free SPSC ring per camera stream (producer = that camera's callback thread,
 * consumer = the single IMU/VIO thread) plus a k-way merge that releases frames in GLOBAL
 * timestamp order, gated on IMU availability:
 *
 *   release head t  iff  t + dt_max(sensor_ids) + guard <= newest_imu_time
 *                   AND  every other LIVE camera has a staged frame >= t or is stale
 *
 * A camera is "live" once it has pushed a frame and its silence is shorter than
 * stale_factor * its EMA frame period -- a dead/slow camera never blocks the others
 * (it re-arms on its next frame). Bit-equal staged timestamps are bundled into one
 * multi-sensor CameraData (the synced-stereo path). Frames older than the last release
 * are dropped (counted) so the consumer NEVER sees time go backwards; the estimator's
 * out-of-order gate becomes a backstop that should read zero.
 *
 * No mutexes, no CAS loops, no new threads: SPSC rings + single-writer atomics only.
 * All drops flow through a disposer callback so the owner can release external image
 * handles (e.g. cl_mem) exactly once for every frame that entered the buffer.
 */

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <boost/lockfree/spsc_queue.hpp>

#include "utils/sensor_data.h"

namespace ov_msckf {

class AsyncCameraBuffer {

public:
  struct Options {
    /// Per-camera ring capacity (frames). 16 @ 30 Hz is >0.5 s of consumer stall headroom.
    size_t ring_capacity = 16;
    /// Extra IMU margin (seconds) beyond the per-camera offset before a frame is releasable
    /// (covers the propagator's boundary interpolation need).
    double guard = 0.002;
    /// A camera is stale after this many EMA frame periods of silence.
    double stale_factor = 1.5;
    /// Frames stamped further than this into the IMU future are broken timestamps: dropped.
    double bogus_future = 1.0;
  };

  /// Called for every frame the buffer discards (ring full / late / bogus). NEVER called for
  /// frames handed to the sink. Runs on the calling thread (producer for ring-full, consumer
  /// otherwise) -- must be thread-safe and cheap (release handles, count).
  using Disposer = std::function<void(const ov_core::CameraData &msg)>;

  /// Returns the max cam-IMU offset among the given sensor ids (state-owned knowledge).
  using DtMaxFn = std::function<double(const std::vector<int> &sensor_ids)>;

  /// Consumes one released frame; return false to pause draining (frame already consumed).
  using SinkFn = std::function<bool(ov_core::CameraData &&msg)>;

  AsyncCameraBuffer(int num_cams, Options options, Disposer disposer);

  /// Disposes anything still queued (so external image handles are never leaked on teardown)
  ~AsyncCameraBuffer() { clear(); }

  /**
   * @brief Producer push (wait-free; one fixed producer thread per camera stream).
   * The ring is selected by msg.sensor_ids.front(); a multi-sensor (stereo) message from its
   * single pipe thread uses its first id's ring, preserving SPSC. On a full ring the frame is
   * disposed and counted (consumer stalled >ring_capacity frames -- effectively unreachable).
   * @return false if the frame was dropped (ring full or invalid ids)
   */
  bool push(const ov_core::CameraData &msg);

  /**
   * @brief Consumer drain (single consumer thread): releases every frame whose ordering is
   * decided, in global timestamp order, into the sink.
   * @param newest_imu_time Newest IMU timestamp fed to the estimator (IMU clock)
   * @param dt_max_for_ids Per-camera offset lookup (see DtMaxFn)
   * @param sink Frame consumer; return false to pause this drain round
   */
  void drain(double newest_imu_time, const DtMaxFn &dt_max_for_ids, const SinkFn &sink);

  /// Dispose every queued/staged frame (consumer thread; use on reset)
  void clear();

  /// @name Telemetry (single-writer; read from anywhere)
  /// @{
  uint64_t count_pushed() const { return counter_pushed.load(std::memory_order_relaxed); }
  uint64_t count_released() const { return counter_released.load(std::memory_order_relaxed); }
  uint64_t count_bundled() const { return counter_bundled.load(std::memory_order_relaxed); }
  uint64_t count_drop_full() const { return counter_drop_full.load(std::memory_order_relaxed); }
  uint64_t count_drop_late() const { return counter_drop_late.load(std::memory_order_relaxed); }
  uint64_t count_drop_bogus() const { return counter_drop_bogus.load(std::memory_order_relaxed); }
  /// @}

protected:
  /// Monotonic wall time in seconds (steady clock; staleness only, never estimator time)
  static double mono_now();

  /// Per-camera stream: SPSC ring + producer-written meta (single-writer atomics)
  struct Stream {
    explicit Stream(size_t capacity) : ring(capacity) {}
    boost::lockfree::spsc_queue<ov_core::CameraData> ring;
    std::atomic<double> last_enqueued_ts{-1.0}; // newest frame timestamp pushed (camera clock)
    std::atomic<double> last_push_mono{-1.0};   // monotonic push time (staleness)
    std::atomic<double> ema_period{-1.0};       // EMA of the frame period (producer-side)
    // Consumer-only staging slot (head of this stream, popped but not yet released)
    ov_core::CameraData staged;
    bool has_staged = false;
    double prev_push_ts = -1.0; // producer-only scratch for the EMA
  };

  Options opts;
  Disposer dispose;
  std::vector<std::unique_ptr<Stream>> streams;

  /// Timestamp of the last frame handed to the sink (consumer-only; release monotonicity)
  double last_released_ts = -1.0;

  std::atomic<uint64_t> counter_pushed{0}, counter_released{0}, counter_bundled{0};
  std::atomic<uint64_t> counter_drop_full{0}, counter_drop_late{0}, counter_drop_bogus{0};
};

} // namespace ov_msckf

#endif // OV_MSCKF_ASYNC_CAMERA_BUFFER_H
