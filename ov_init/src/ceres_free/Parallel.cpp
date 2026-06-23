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
 */

#include "Parallel.h"

#include <algorithm>

using namespace ov_init::zbft_sfm;

namespace {
// Architecture relax hint for busy-wait loops (lock-free spin, not a blocking wait).
inline void cpu_relax() {
#if defined(__aarch64__) || defined(__arm__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  asm volatile("pause" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_acq_rel);
#endif
}
} // namespace

ParallelExecutor::ParallelExecutor(int num_threads, std::function<void(int)> worker_init)
    : worker_init_(std::move(worker_init)) {
  int hw = (int)std::thread::hardware_concurrency();
  if (hw < 1)
    hw = 1;
  num_workers_ = std::max(1, std::min(num_threads, hw));
  if (num_workers_ <= 1) {
    num_workers_ = 1;
    return; // inline mode: never create threads
  }

  // Per-worker completion flags (index 0 unused; worker 0 is the dispatcher).
  done_seq_ = std::vector<PaddedSeq>(num_workers_);

  threads_.reserve(num_workers_ - 1);
  for (int i = 1; i < num_workers_; ++i)
    threads_.emplace_back(&ParallelExecutor::worker_loop, this, i);
}

ParallelExecutor::~ParallelExecutor() {
  if (threads_.empty())
    return;
  // Signal stop and bump the dispatch generation to release any spinning worker.
  stop_.store(true, std::memory_order_release);
  ++gen_;
  dispatch_seq_.store(gen_, std::memory_order_release);
  for (auto &t : threads_) {
    if (t.joinable())
      t.join();
  }
}

void ParallelExecutor::worker_loop(int w) {
  if (worker_init_)
    worker_init_(w);

  uint64_t last = 0;
  uint64_t idle = 0;
  while (true) {
    const uint64_t s = dispatch_seq_.load(std::memory_order_acquire);
    if (stop_.load(std::memory_order_acquire))
      return;

    if (s != last) {
      // New job: job_/job_n_ are visible thanks to the acquire on dispatch_seq_.
      const std::function<void(int, int, int)> *job = job_;
      const int n = job_n_;
      last = s;
      if (job != nullptr) {
        const auto r = worker_range(w, num_workers_, n);
        (*job)(w, r.first, r.second);
      }
      done_seq_[w].v.store(s, std::memory_order_release);
      idle = 0;
    } else {
      // Lock-free busy-wait. The pool only lives for the duration of a bounded
      // solve, so this never spins for long, and num_workers <= cores-2 keeps a
      // core free for the RT threads.
      cpu_relax();
      if ((++idle & 0xFFF) == 0)
        std::this_thread::yield();
    }
  }
}

void ParallelExecutor::parallel_ranges(int n, const std::function<void(int, int, int)> &body) {
  // Inline fast path: zero synchronization, fully deterministic.
  if (num_workers_ <= 1) {
    body(0, 0, n);
    return;
  }

  // Publish the job, then release-store the new generation so workers observe it.
  job_ = &body;
  job_n_ = n;
  ++gen_;
  const uint64_t gen = gen_;
  dispatch_seq_.store(gen, std::memory_order_release);

  // Worker 0's slice runs on the calling thread.
  const auto r0 = worker_range(0, num_workers_, n);
  body(0, r0.first, r0.second);

  // Lock-free wait for spawned workers to reach this generation.
  for (int w = 1; w < num_workers_; ++w) {
    uint64_t spins = 0;
    while (done_seq_[w].v.load(std::memory_order_acquire) < gen) {
      cpu_relax();
      if ((++spins & 0xFFF) == 0)
        std::this_thread::yield();
    }
  }
  job_ = nullptr;
}
