/* Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_INITIALIZER_POSE_SELECTION_H
#define OV_INIT_INITIALIZER_POSE_SELECTION_H

#include <algorithm>
#include <cmath>
#include <limits>
#include "utils/finite.h"

namespace ov_init {
// This is only the pose-subsampling heuristic, not timestamp identity or IMU
// coverage. Subtracting/retiming floating timestamps can round an exactly
// spaced frame just below the desired interval. Allow their arithmetic error
// at the current clock scale without merging keys or moving any observation.
inline bool initializer_pose_spacing(double nearest_distance, double interval, double clock_scale) {
  if (!ov_core::numeric::finite(nearest_distance) || nearest_distance < 0. ||
      !ov_core::numeric::finite(interval) || interval <= 0. || !ov_core::numeric::finite(clock_scale)) return false;
  if (nearest_distance == 0. || nearest_distance >= interval) return true;
  const double roundoff = 8. * std::numeric_limits<double>::epsilon() * std::max(std::abs(clock_scale), interval);
  return interval - nearest_distance <= roundoff;
}
} // namespace ov_init
#endif
