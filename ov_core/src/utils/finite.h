/* Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_CORE_FINITE_H
#define OV_CORE_FINITE_H
#include <cstdint>
#include <cstring>
namespace ov_core { namespace numeric {
// Check storage bits: -ffast-math may erase std::isfinite and x == x guards.
inline bool finite(double value) {
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value), "binary64 required");
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
inline bool finite(float value) {
  std::uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value), "binary32 required");
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}
template <typename Matrix> inline bool finite_matrix(const Matrix &value) {
  for (decltype(value.cols()) j = 0; j < value.cols(); ++j)
    for (decltype(value.rows()) i = 0; i < value.rows(); ++i)
      if (!finite(value(i, j))) return false;
  return true;
}
}} // namespace ov_core::numeric
#endif
