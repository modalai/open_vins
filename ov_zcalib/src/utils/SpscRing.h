/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: lock-free single-producer/single-consumer ring for the live feeders.
 * QRB5165 hot-path rules: acquire/release atomics only, no locks, no allocation
 * after construction, and alignas(128) on the control indices (Kryo-585 big.LITTLE
 * destructive-interference size) so producer and consumer never false-share.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_SPSC_RING_H
#define OV_ZCALIB_SPSC_RING_H

#include <atomic>
#include <cstddef>
#include <vector>

namespace ov_zcalib {

template <typename T> class SpscRing {
public:
  explicit SpscRing(size_t capacity_pow2) : mask_(capacity_pow2 - 1), buf_(capacity_pow2) {
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

  /// Consumer side. Returns false when empty.
  bool pop(T &v) {
    const size_t t = tail_.load(std::memory_order_relaxed);
    const size_t h = head_.load(std::memory_order_acquire);
    if (t == h)
      return false;
    v = buf_[t & mask_];
    tail_.store(t + 1, std::memory_order_release);
    return true;
  }

  size_t size() const {
    return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
  }

private:
  alignas(128) std::atomic<size_t> head_{0};
  char head_pad_[128];
  alignas(128) std::atomic<size_t> tail_{0};
  char tail_pad_[128];
  const size_t mask_;
  std::vector<T> buf_;
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_SPSC_RING_H
