#ifndef OV_ZCALIB_VERIFICATION_H
#define OV_ZCALIB_VERIFICATION_H

#include <cstddef>
#include <vector>
#include "utils/NumericChecks.h"

namespace ov_zcalib {

inline bool valid_verification_cost(bool solved, bool time_stopped, double cost) {
  return solved && !time_stopped && finite_scalar(cost) && cost >= 0.0;
}

// Optional leave-one-block-out diagnostics cannot remove a window from the
// authoritative seed/solution/mixture comparison.
inline bool verification_window_usable(const std::vector<char> &valid, std::size_t required_candidates) {
  if (required_candidates < 2 || required_candidates > valid.size())
    return false;
  for (std::size_t i = 0; i < required_candidates; ++i)
    if (!valid[i])
      return false;
  return true;
}

} // namespace ov_zcalib
#endif
