/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Contributor: Joao Leonardo Silva Cotta (@zauberflote1)
 *
 * Ceres-free initialization backend (ov_init::zbft_sfm)
 * ------------------------------------------------
 * Drop-in replacement for the small subset of ceres::CostFunction that the
 * OpenVINS initialization factors actually use. By matching the method names
 * and the Evaluate() signature exactly, the proven analytic factors under
 * ov_init/src/ceres/ can be transcribed onto this base class with only a
 * base-class + include swap and NO change to the residual/Jacobian math.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_ZBFT_SFM_COSTFUNCTION_H
#define OV_INIT_ZBFT_SFM_COSTFUNCTION_H

#include <vector>

namespace ov_init {
namespace zbft_sfm {

/**
 * @brief Minimal analytic cost function interface (Ceres-API compatible subset).
 *
 * Semantics are identical to ceres::CostFunction:
 *   - residuals is a `num_residuals()`-length array.
 *   - parameters[i] points to the i-th parameter block (GLOBAL / ambient size,
 *     i.e. parameter_block_sizes()[i] doubles).
 *   - jacobians may be nullptr (residual-only eval). If non-null, jacobians[i]
 *     may individually be nullptr; otherwise it is a row-major
 *     (num_residuals x parameter_block_sizes()[i]) buffer of d(residual)/d(block_i)
 *     in AMBIENT coordinates. The solver maps ambient -> local via the block's
 *     LocalParameterization (see LocalParameterization.h).
 *
 * Implementations MUST be thread-safe / const: Evaluate() is called concurrently
 * from worker threads during parallel Hessian accumulation, so it must not mutate
 * shared state (all configuration lives in const members set at construction).
 */
class CostFunction {
public:
  virtual ~CostFunction() = default;

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const = 0;

  int num_residuals() const { return num_residuals_; }
  const std::vector<int> &parameter_block_sizes() const { return parameter_block_sizes_; }

protected:
  void set_num_residuals(int num_residuals) { num_residuals_ = num_residuals; }
  std::vector<int> *mutable_parameter_block_sizes() { return &parameter_block_sizes_; }

private:
  int num_residuals_ = 0;
  std::vector<int> parameter_block_sizes_;
};

} // namespace zbft_sfm
} // namespace ov_init

#endif // OV_INIT_ZBFT_SFM_COSTFUNCTION_H
