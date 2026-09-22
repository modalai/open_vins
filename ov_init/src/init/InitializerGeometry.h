/*
 * Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_INITIALIZER_GEOMETRY_H
#define OV_INIT_INITIALIZER_GEOMETRY_H

#include <Eigen/Core>
#include <cstdint>
#include <cstring>

namespace ov_init {

// Domain of the initializer's forward pinhole/equidistant projection. Dividing
// by z also gives finite pixels for a point behind the camera, but that is not
// a valid observed ray. Test the optimized geometry before exporting covariance
// or injecting a seed; finite/SPD covariance and a plausible gravity direction
// do not establish cheirality. No data-dependent depth threshold is imposed.
struct InitializerGeometry {
  static bool valid_camera_point(const Eigen::Vector3d &point) {
    // std::isfinite can be optimized away by the target's -ffast-math flags.
    for (int i = 0; i < 3; ++i) {
      std::uint64_t bits;
      const double coordinate = point(i);
      static_assert(sizeof(bits) == sizeof(coordinate), "requires binary64 coordinates");
      std::memcpy(&bits, &coordinate, sizeof(bits));
      if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000))
        return false;
    }
    return point.z() > 0.0;
  }
};

} // namespace ov_init
#endif // OV_INIT_INITIALIZER_GEOMETRY_H
