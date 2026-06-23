/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * Ceres-free initialization backend (ov_init::zbft_sfm)
 * ------------------------------------------------
 * Robust loss functions used as iteratively-reweighted-least-squares (IRLS)
 * weights. Evaluate(s, out) returns out[0]=rho(s), out[1]=rho'(s) for
 * s = ||residual||^2. The solver uses rho'(s) as the per-residual weight
 * (first-order IRLS); this matches the down-weighting behaviour of Ceres'
 * Cauchy/Huber losses and keeps the Gauss-Newton Hessian positive semidefinite.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_ZBFT_SFM_LOSSFUNCTION_H
#define OV_INIT_ZBFT_SFM_LOSSFUNCTION_H

#include <cmath>

namespace ov_init {
namespace zbft_sfm {

/// Base robust loss. out[0]=rho(s), out[1]=rho'(s)>=0 with s=||r||^2.
class LossFunction {
public:
  virtual ~LossFunction() = default;
  virtual void Evaluate(double s, double out[2]) const = 0;
};

/// rho(s) = s  (standard least squares).
class TrivialLoss : public LossFunction {
public:
  void Evaluate(double s, double out[2]) const override {
    out[0] = s;
    out[1] = 1.0;
  }
};

/// Huber loss with threshold a (on the residual norm). rho1 in (0,1].
class HuberLoss : public LossFunction {
public:
  explicit HuberLoss(double a) : b_(a * a) {}
  void Evaluate(double s, double out[2]) const override {
    if (s <= b_) {
      out[0] = s;
      out[1] = 1.0;
    } else {
      const double r = std::sqrt(s);
      out[0] = 2.0 * std::sqrt(b_) * r - b_;
      out[1] = std::sqrt(b_) / r;
    }
  }

private:
  double b_; // a^2
};

/// Cauchy loss with scale a. rho(s) = a^2 * log(1 + s/a^2). This is what the
/// upstream DynamicInitializer MLE uses on its reprojection residuals.
class CauchyLoss : public LossFunction {
public:
  explicit CauchyLoss(double a) : c2_(a * a), inv_c2_(1.0 / (a * a)) {}
  void Evaluate(double s, double out[2]) const override {
    const double sum = 1.0 + s * inv_c2_;
    out[0] = c2_ * std::log(sum);
    out[1] = 1.0 / sum;
  }

private:
  double c2_;
  double inv_c2_;
};

} // namespace zbft_sfm
} // namespace ov_init

#endif // OV_INIT_ZBFT_SFM_LOSSFUNCTION_H
