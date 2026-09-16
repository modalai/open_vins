#ifndef OV_ZCALIB_CAMERA_REFINEMENT_H
#define OV_ZCALIB_CAMERA_REFINEMENT_H

#include <Eigen/Core>
#include <algorithm>
#include <cmath>

namespace ov_zcalib {

using CameraPrior = Eigen::Matrix<double, 8, 1>;

// Equidistant k3/k4 need radial coverage. Radtan p1/p2 need coverage around
// both image axes to separate tangential distortion from principal-point shifts.
inline CameraPrior camera_refinement_prior(bool fisheye, const CameraPrior &base, double far_fraction,
                                           double quadrant_min, double radial_gate, double center_gate,
                                           double tangent_sigma) {
  CameraPrior p = base;
  if (fisheye) {
    if (far_fraction >= radial_gate) {
      p(6) = p(4);
      p(7) = p(5);
    }
  } else {
    p.tail<2>().setConstant(quadrant_min >= center_gate ? tangent_sigma : 1e-9);
  }
  if (quadrant_min < center_gate)
    p.segment<2>(2).setConstant(1e-9);
  return p;
}

// Freeze unresolved pairs at factory values. The caller must refit the joint
// problem and recompute its marginal posterior before making a commit decision.
inline bool freeze_unresolved_camera_pairs(const CameraPrior &sigma, double factor, CameraPrior &prior,
                                           CameraPrior &value, const CameraPrior &factory) {
  bool changed = false;
  for (int first : {0, 2, 4, 6}) {
    bool weak = false;
    for (int k = first; k < first + 2; ++k)
      if (prior(k) > 1e-8 && !(factor * sigma(k) < prior(k)))
        weak = true;
    if (!weak)
      continue;
    for (int k = first; k < first + 2; ++k) {
      prior(k) = 1e-9;
      value(k) = factory(k);
    }
    changed = true;
  }
  return changed;
}

} // namespace ov_zcalib
#endif
