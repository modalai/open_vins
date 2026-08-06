/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_core: lock-free single-producer/single-consumer ring, the boost::lockfree::spsc_queue
 * replacement for the camera ingest path. Same discipline as ov_zcalib's SpscRing (the
 * proven live-feeder ring this generalizes): acquire/release atomics only, no locks, no
 * allocation after construction, alignas(128) control indices (Kryo-585 big.LITTLE
 * destructive-interference size) so producer and consumer never false-share.
 *
 * Two deliberate differences from boost::lockfree::spsc_queue, both for the embedded target:
 *   - any capacity is accepted and rounded UP to a power of two (index masking, no modulo)
 *   - pop() MOVES the element out and resets the slot, so a popped frame's pixel buffers are
 *     released immediately instead of pinning up to capacity frames of memory in the ring
 *     (boost destroys popped elements; a plain slot-copy ring would not)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_CORE_LOCK_FREE_SPSC_H
#define OV_CORE_LOCK_FREE_SPSC_H

#include <atomic>
#include <cstddef>
#include <utility>
#include <vector>

namespace ov_core {

template <typename T> class SpscRing {
public:
  explicit SpscRing(size_t capacity) : mask_(round_up_pow2(capacity) - 1), buf_(mask_ + 1) {
    static_assert(sizeof(head_pad_) >= 128, "pad");
  }

  /// Producer side. Returns false when full (caller counts the drop; never blocks).
  bool push(const T &v) {
    const size_t h = head_.load(std::memory_order_relaxed);
    const size_t t = tail_.load(std::memory_order_acquire);
    if (h - t > mask_)
      return false;
    buf_[h & mask_] = v;
    head_.store(h + 1, std::memory_order_release);
    return true;
  }

  /// Producer side, move-in overload (no deep copy of frame payloads).
  bool push(T &&v) {
    const size_t h = head_.load(std::memory_order_relaxed);
    const size_t t = tail_.load(std::memory_order_acquire);
    if (h - t > mask_)
      return false;
    buf_[h & mask_] = std::move(v);
    head_.store(h + 1, std::memory_order_release);
    return true;
  }

  /// Consumer side. Returns false when empty. Moves the element out and RESETS the slot so
  /// payload buffers (cv::Mat refcounts, vectors) release now, not at overwrite time.
  bool pop(T &v) {
    const size_t t = tail_.load(std::memory_order_relaxed);
    const size_t h = head_.load(std::memory_order_acquire);
    if (t == h)
      return false;
    v = std::move(buf_[t & mask_]);
    buf_[t & mask_] = T();
    tail_.store(t + 1, std::memory_order_release);
    return true;
  }

  size_t read_available() const {
    return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
  }

private:
  static size_t round_up_pow2(size_t n) {
    size_t p = 1;
    while (p < n)
      p <<= 1;
    return p;
  }

  alignas(128) std::atomic<size_t> head_{0};
  char head_pad_[128];
  alignas(128) std::atomic<size_t> tail_{0};
  char tail_pad_[128];
  const size_t mask_;
  std::vector<T> buf_;
};

} // namespace ov_core

#endif // OV_CORE_LOCK_FREE_SPSC_H
