/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_INIT_CERES_JPLQUATLOCAL_H
#define OV_INIT_CERES_JPLQUATLOCAL_H

#include <ceres/ceres.h>
#include <ceres/version.h>

namespace ov_init {

#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
/**
 * @brief JPL quaternion CERES Manifold (Ceres 2.1+)
 */
class State_JPLQuatLocal : public ceres::Manifold {
public:
  int AmbientSize() const override { return 4; }
  int TangentSize() const override { return 3; }

  /**
   * @brief State update function for a JPL quaternion representation.
   */
  bool Plus(const double *x, const double *delta, double *x_plus_delta) const override;

  /**
   * @brief Computes the jacobian of Plus w.r.t. delta at delta=0
   */
  bool PlusJacobian(const double *x, double *jacobian) const override;

  /**
   * @brief Computes y_minus_x = y [-] x in the tangent space
   */
  bool Minus(const double *y, const double *x, double *y_minus_x) const override;

  /**
   * @brief Computes the jacobian of Minus w.r.t. y
   */
  bool MinusJacobian(const double *x, double *jacobian) const override;
};

#else
/**
 * @brief JPL quaternion CERES state parameterization (Ceres 1.x)
 */
class State_JPLQuatLocal : public ceres::LocalParameterization {
public:
  /**
   * @brief State update function for a JPL quaternion representation.
   */
  bool Plus(const double *x, const double *delta, double *x_plus_delta) const override;

  /**
   * @brief Computes the jacobian in respect to the local parameterization
   */
  bool ComputeJacobian(const double *x, double *jacobian) const override;

  int GlobalSize() const override { return 4; };

  int LocalSize() const override { return 3; };
};
#endif

} // namespace ov_init

#endif // OV_INIT_CERES_JPLQUATLOCAL_H