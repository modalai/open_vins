/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * Ceres-free initialization backend (ov_init::zbft_sfm)
 * ------------------------------------------------
 * Lifted verbatim from ov_init/src/ceres/State_JPLQuatLocal.{h,cpp}; only the
 * base class changed (ceres::LocalParameterization -> zbft_sfm::LocalParameterization).
 * JPL left-multiplicative quaternion retraction, GlobalSize=4, LocalSize=3.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_ZBFT_SFM_STATE_JPLQUATLOCAL_H
#define OV_INIT_ZBFT_SFM_STATE_JPLQUATLOCAL_H

#include "LocalParameterization.h"

namespace ov_init {
namespace zbft_sfm {

/**
 * @brief JPL quaternion local (manifold) parameterization.
 *
 * q <-- [d_th/2; 1] (x) q   (JPL left-multiplicative update)
 * The plus-Jacobian is [I_3; 0_{1x3}] (4x3), so the solver reduces an ambient
 * (N x 4) Jacobian to a local (N x 3) one by selecting its first three columns.
 */
class State_JPLQuatLocal : public LocalParameterization {
public:
  bool Plus(const double *x, const double *delta, double *x_plus_delta) const override;
  bool ComputeJacobian(const double *x, double *jacobian) const override;
  int GlobalSize() const override { return 4; }
  int LocalSize() const override { return 3; }
  bool tangent_is_leading_identity() const override { return true; } // V = [I₃; 0]
};

} // namespace zbft_sfm
} // namespace ov_init

#endif // OV_INIT_ZBFT_SFM_STATE_JPLQUATLOCAL_H
