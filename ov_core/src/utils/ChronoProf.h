/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_core: profiling clock, the boost::posix_time replacement for the rT1..rT7 timing
 * instrumentation. std::chrono::steady_clock instead of microsec_clock::local_time():
 * the boost call resolved the local TIMEZONE on every sample (getenv("TZ") + localtime
 * machinery, which can take a lock inside glibc) -- on the estimator thread, several
 * times per frame. steady_clock is a raw monotonic counter read: cheaper, lock-free,
 * and immune to wall-clock steps, which is what elapsed-time profiling wanted anyway.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_CORE_CHRONO_PROF_H
#define OV_CORE_CHRONO_PROF_H

#include <chrono>

namespace ov_core {

/// Time point type for the rT* profiling members
using ProfTime = std::chrono::steady_clock::time_point;

/// One profiling sample (monotonic)
inline ProfTime prof_now() { return std::chrono::steady_clock::now(); }

/// Elapsed seconds a -> b (replaces (b - a).total_microseconds() * 1e-6)
inline double prof_s(const ProfTime &a, const ProfTime &b) {
  return std::chrono::duration<double>(b - a).count();
}

/// Elapsed microseconds a -> b (replaces (b - a).total_microseconds())
inline double prof_us(const ProfTime &a, const ProfTime &b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

} // namespace ov_core

#endif // OV_CORE_CHRONO_PROF_H
