/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * Ceres-free initialization backend (ov_init::zbft_sfm)
 * ------------------------------------------------
 * LOCK-FREE bounded worker pool for the parallel Hessian accumulation in
 * Problem::Solve(). No mutexes, no condition variables, no blocking sleeps:
 * dispatch and completion are signalled through release/acquire atomics, and
 * idle workers busy-wait with an architecture PAUSE/YIELD hint. Rationale for a
 * real-time, resource-constrained target (QRB5165):
 *
 *   - No mutex/CV  => no lock contention and no priority inversion against the
 *     1 kHz IMU thread or the camera thread.
 *   - num_workers <= 1 => fully inline; NO threads are created (the RT default).
 *   - Work is split into FIXED contiguous ranges bound to FIXED worker indices,
 *     so a caller reducing per-worker partials in worker order is bit-identical
 *     regardless of thread count or scheduling (serial == parallel).
 *   - The executor is meant to live only for the duration of a bounded solve
 *     (tens of ms, off the RT path); workers busy-yield during that window and
 *     are torn down immediately after. Keep num_workers <= (cores - 2) so the
 *     IMU/camera real-time threads always have a core.
 *   - worker_init(index) lets the integrator pin affinity / drop scheduling class.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_ZBFT_SFM_PARALLEL_H
#define OV_INIT_ZBFT_SFM_PARALLEL_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

namespace ov_init {
namespace zbft_sfm {

class ParallelExecutor {
public:
  /**
   * @param num_threads Desired worker count; clamped to [1, hardware_concurrency()].
   *                    <=1 means "run inline, never create threads".
   * @param worker_init Optional per-worker setup (affinity/priority), invoked once on
   *                    each spawned worker (indices 1..num_workers-1).
   */
  explicit ParallelExecutor(int num_threads, std::function<void(int)> worker_init = {});
  ~ParallelExecutor();

  ParallelExecutor(const ParallelExecutor &) = delete;
  ParallelExecutor &operator=(const ParallelExecutor &) = delete;

  /// Number of logical workers (>=1). Worker 0 is always the calling thread.
  int num_workers() const { return num_workers_; }

  /// Balanced contiguous range [begin,end) for a worker given problem size n.
  static std::pair<int, int> worker_range(int worker, int num_workers, int n) {
    const int base = n / num_workers;
    const int extra = n % num_workers;
    const int b = worker * base + (worker < extra ? worker : extra);
    const int e = b + base + (worker < extra ? 1 : 0);
    return {b, e};
  }

  /**
   * @brief Split [0,n) into num_workers() fixed contiguous ranges and run
   *        body(worker_index, begin, end) on each; returns when all complete.
   *        Worker 0's range runs on the calling thread. body must not throw.
   */
  void parallel_ranges(int n, const std::function<void(int worker, int begin, int end)> &body);

  /**
   * @brief Dynamically scheduled loop: workers atomically claim the next index
   *        until [0,n) is exhausted. body(worker_index, i) runs exactly once per i.
   *
   * WHY, over parallel_ranges: a fixed contiguous partition balances only equal-cost
   * tasks. Window solves are not -- clone count, observation count and LM iteration
   * count all vary (a warm-vs-cold duel alone can double one), so a static split
   * gates every outer pass on the slowest range (measured: 149 s of thread-CPU
   * completing in 59 s wall on 4 workers = 63% efficiency).
   *
   * DETERMINISM IS UNAFFECTED -- structurally, not aspirationally. Task i writes
   * only slot i and is a pure function of (task data, p), so WHICH worker runs it
   * cannot change WHAT it computes; callers reduce the partials in a SERIAL,
   * INDEX-ORDERED fold after the region (see JointCalib's "serial fold").
   * serial == parallel == any schedule, bit for bit. (Strictly WEAKER than
   * parallel_ranges' contract, which pins ranges to worker indices -- needed only
   * when a worker accumulates its own partial, which these call sites do not.)
   *
   * Feed indices LONGEST-FIRST (LPT) for a near-optimal makespan on unequal tasks.
   */
  void parallel_dynamic(int n, const std::function<void(int worker, int i)> &body);

private:
  void worker_loop(int worker_index);

  // Cache-line padded atomic to prevent false sharing between workers' done flags.
  struct alignas(64) PaddedSeq {
    std::atomic<uint64_t> v{0};
    char pad[64 - sizeof(std::atomic<uint64_t>)] = {};
  };

  int num_workers_ = 1;
  std::vector<std::thread> threads_; // spawned workers (indices 1..num_workers_-1)
  std::function<void(int)> worker_init_;

  // Job payload: written by the dispatcher BEFORE the release store to dispatch_seq_,
  // read by workers AFTER the acquire load -> visibility is guaranteed by that pairing.
  const std::function<void(int, int, int)> *job_ = nullptr;
  int job_n_ = 0;
  uint64_t gen_ = 0; // dispatch counter; only the (single) dispatcher thread mutates it

  alignas(64) std::atomic<uint64_t> dispatch_seq_{0};
  alignas(64) std::atomic<bool> stop_{false};
  std::vector<PaddedSeq> done_seq_; // per-worker completion generation (index 0 unused)
};

} // namespace zbft_sfm
} // namespace ov_init

#endif // OV_INIT_ZBFT_SFM_PARALLEL_H
