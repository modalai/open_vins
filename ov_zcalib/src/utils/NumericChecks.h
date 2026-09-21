#ifndef OV_ZCALIB_NUMERIC_CHECKS_H
#define OV_ZCALIB_NUMERIC_CHECKS_H

#include <cstdint>
#include <cstring>

namespace ov_zcalib {

// std::isfinite may be folded to true under this project's -ffast-math.
// Validation must inspect the representation before doing ordered arithmetic.
inline bool finite_scalar(double value) {
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value), "64-bit IEEE double required");
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

template <typename Matrix> inline bool finite_matrix(const Matrix &value) {
  for (decltype(value.cols()) j = 0; j < value.cols(); ++j)
    for (decltype(value.rows()) i = 0; i < value.rows(); ++i)
      if (!finite_scalar(value(i, j)))
        return false;
  return true;
}

} // namespace ov_zcalib
#endif
