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

#ifndef OV_MSCKF_ASYNC_CAMERA_BUFFER_H
#define OV_MSCKF_ASYNC_CAMERA_BUFFER_H

/**
 *
 * Lock-free multi-camera ingest with asynchronous or forced-sync timestamps.
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
 * Forced-sync mode instead requires a complete group with identical hardware
 * exposure-start keys, assigns its reference camera's timestamp to every view,
 * and applies the same IMU coverage gate. Missing mates expire without releasing
 * a partial group. Recorded input uses stream progress/EOF instead of wall time.
 * This timing policy is independent of stereo feature association.
 *
 * No mutexes, no CAS loops, no new threads: SPSC rings + single-writer atomics only.
 * All drops flow through a disposer callback so the owner can release external image
 * handles (e.g. cl_mem) exactly once for every frame that entered the buffer.
 */

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "utils/LockFreeSpsc.h"

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
    /// Optional per-camera NOMINAL frame period seeds (seconds) for the staleness EMA, indexed
    /// by camera id. Until a stream's own EMA settles, its declared period (instead of the
    /// conservative 20 Hz default) decides how long the merge holds ordering for it.
    std::vector<double> initial_periods;
    /// Enable camera-wise metadata and bounded staging for drain_physical().
    /// Kept off for the legacy raw-clock merge.
    bool physical_order = false;
    /// Explicit common-time grouping, independent of stereo feature matching.
    /// -1 preserves asynchronous input. Otherwise all configured cameras must
    /// supply the same hardware exposure-start key; this camera owns group time.
    int sync_reference_camera = -1;
  };

  /// Called for every frame the buffer discards (ring full / late / bogus). NEVER called for
  /// frames handed to the sink. Runs on the calling thread (producer for ring-full, consumer
  /// otherwise) -- must be thread-safe and cheap (release handles, count).
  using Disposer = std::function<void(const ov_core::CameraData &msg)>;

  /// Returns the max cam-IMU offset among the given sensor ids (state-owned knowledge).
  using DtMaxFn = std::function<double(const std::vector<int> &sensor_ids)>;

  /// Consumes one released frame; return false to pause draining (frame already consumed).
  using SinkFn = std::function<bool(ov_core::CameraData &&msg)>;

  /// Nominal camera-to-IMU offset for one camera. Sampled once per selection loop.
  using DtFn = std::function<double(int camera_id)>;
  /// Consumes a complete equal-physical-time group. Raw timestamps stay unchanged;
  /// only members with equal raw timestamps may be bundled into one CameraData.
  /// The sink owns/disposes every component, and false pauses after consumption.
  using GroupSinkFn = std::function<bool(std::vector<ov_core::CameraData> &group, double nominal_imu_endpoint)>;

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

  /// Physical-clock merge. Requires Options::physical_order. Offsets are frozen
  /// for each group and resampled after its callback, so online clock updates can
  /// reorder pending cameras. Each camera's raw sequence remains monotonic.
  void drain_physical(double newest_imu_time, const DtFn &dt_for_camera, const GroupSinkFn &sink);

  /// Select recorded-input ordering before any camera input, on a quiescent
  /// consumer. Physical ordering then waits for every logical camera's next
  /// timestamp or explicit EOF, including cameras not seen yet. Host compute
  /// time cannot make a recorded camera stale. Declare absent cameras finished;
  /// bounded queues still apply. clear() preserves this replay policy.
  /// Requires physical_order or sync_reference_camera; in forced-sync mode EOF
  /// discards incomplete groups. The legacy raw-clock merge is unchanged.
  void prepare_recorded_input();

  /// Declare a logical camera complete after its producer's final push. Finished
  /// cameras no longer hold peers waiting for a future frame; IMU coverage and
  /// timestamp order still apply. Returns false for an invalid camera id.
  bool finish_camera(size_t camera_id);

  /// End all input after producers are quiescent (offline replay EOF).
  void finish_all();

  /// Dispose remaining queued/staged frames without reopening finished inputs
  /// or resetting the consumed time boundary. At terminal EOF, call after a
  /// final drain to report uncovered tails through the normal disposer. A sink
  /// that deliberately paused may also leave covered frames in this tail.
  void discard_pending();

  /// Dispose every queued/staged frame (consumer thread; use on reset)
  /// Reopens finished cameras and resets consumed/raw time boundaries.
  /// Recorded-input producers must be quiescent; their publication watermarks
  /// are cleared for a possible rewind. Reassert EOF for still-absent cameras.
  void clear();

  /// @name Telemetry (single-writer; read from anywhere)
  /// @{
  uint64_t count_pushed() const { return counter_pushed.load(std::memory_order_relaxed); }
  uint64_t count_released() const { return counter_released.load(std::memory_order_relaxed); }
  uint64_t count_bundled() const { return counter_bundled.load(std::memory_order_relaxed); }
  uint64_t count_drop_full() const { return counter_drop_full.load(std::memory_order_relaxed); }
  uint64_t count_drop_late() const { return counter_drop_late.load(std::memory_order_relaxed); }
  uint64_t count_drop_bogus() const { return counter_drop_bogus.load(std::memory_order_relaxed); }
  uint64_t count_drop_finished() const { return counter_drop_finished.load(std::memory_order_relaxed); }
  uint64_t count_drop_unpaired() const { return counter_drop_unpaired.load(std::memory_order_relaxed); }
  /// Times the merge held the head one drain to bundle a same-timestamp mate that was in-flight
  /// (should track the stereo pair rate; a nonzero, growing value = the re-pair fix is active).
  uint64_t count_held_pair() const { return counter_held_pair.load(std::memory_order_relaxed); }
  uint64_t count_physical_views() const { return counter_physical_views.load(std::memory_order_relaxed); }
  /// @}

protected:
  /// Monotonic wall time in seconds (steady clock; staleness only, never estimator time)
  static double mono_now();

  /// Per-camera stream: SPSC ring + producer-written meta (single-writer atomics)
  struct Stream {
    explicit Stream(size_t capacity) : ring(capacity) {}
    ov_core::SpscRing<ov_core::CameraData> ring;
    std::atomic<double> last_enqueued_ts{-1.0}; // newest frame timestamp pushed (camera clock)
    std::atomic<double> last_push_mono{-1.0};   // monotonic push time (staleness)
    std::atomic<double> ema_period{-1.0};       // EMA of the frame period (producer-side)
    // Consumer-only staging slot (head of this stream, popped but not yet released)
    ov_core::CameraData staged;
    bool has_staged = false;
    double staged_since = 0.; // consumer-only wait bound for a forced-sync mate
    bool physical_stage_blocked = false; // consumer-only, reset for each bounded staging pass
    std::atomic<bool> finished{false}; // indexed by logical camera, including packed-message members
    double prev_push_ts = -1.0; // producer-only scratch for the EMA
  };

  Options opts;
  Disposer dispose;
  std::vector<std::unique_ptr<Stream>> streams;

  /// One consumer-owned normalized group, also used by physical-clock staging.
  /// Its tiny unused ring keeps staging/partial-message ownership identical.
  std::unique_ptr<Stream> synchronized_source;
  std::vector<Stream *> sync_matches;
  std::vector<bool> sync_present;
  int64_t last_sync_start_ns = -1;
  double last_sync_timestamp = -std::numeric_limits<double>::infinity();
  bool stage_synchronized(double newest_imu_time, size_t &pop_budget);

  /// A bounded consumer-only staging ring for each logical camera. Packed source
  /// messages can split across physical instants without hiding the next frame
  /// of an earlier camera behind a later component in the same producer ring.
  struct PhysicalCamera {
    explicit PhysicalCamera(size_t capacity) : pending(capacity) {}
    std::vector<ov_core::CameraData> pending;
    size_t begin = 0, size = 0;
    double last_staged_raw = -std::numeric_limits<double>::infinity();
    double last_seen_raw = -std::numeric_limits<double>::infinity();
    std::atomic<double> last_enqueued_raw{-std::numeric_limits<double>::infinity()};
    std::atomic<double> last_push_mono{-1.0};
    std::atomic<double> ema_period{-1.0};
  };
  std::vector<std::unique_ptr<PhysicalCamera>> physical_cameras;
  std::vector<double> physical_offsets;
  std::vector<ov_core::CameraData> physical_group;
  double last_released_physical = -std::numeric_limits<double>::infinity();
  bool physical_in_sink = false;
  bool recorded_input = false; // configured before input, read only by the consumer

  /// Timestamp of the last frame handed to the sink (consumer-only; release monotonicity)
  double last_released_ts = -1.0;

  std::atomic<uint64_t> counter_pushed{0}, counter_released{0}, counter_bundled{0};
  std::atomic<uint64_t> counter_drop_full{0}, counter_drop_late{0}, counter_drop_bogus{0};
  std::atomic<uint64_t> counter_drop_finished{0};
  std::atomic<uint64_t> counter_drop_unpaired{0};
  std::atomic<uint64_t> counter_held_pair{0};
  std::atomic<uint64_t> counter_physical_views{0};
};

} // namespace ov_msckf

#endif // OV_MSCKF_ASYNC_CAMERA_BUFFER_H
