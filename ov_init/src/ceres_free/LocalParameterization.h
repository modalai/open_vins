/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * Ceres-free initialization backend (ov_init::zbft_sfm)
 * ------------------------------------------------
 * Drop-in for the subset of ceres::LocalParameterization the factors use, plus
 * the trivial Euclidean parameterization and a convenience PlusJacobian() that
 * returns the (global x local) plus-Jacobian as an Eigen matrix for the solver.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_ZBFT_SFM_LOCALPARAMETERIZATION_H
#define OV_INIT_ZBFT_SFM_LOCALPARAMETERIZATION_H

#include <Eigen/Dense>
#include <vector>

namespace ov_init {
namespace zbft_sfm {

/**
 * @brief Manifold retraction interface (Ceres-API compatible subset).
 *
 * Plus():            x_plus_delta = x (+) delta   (global<-global,local)
 * ComputeJacobian(): row-major (GlobalSize x LocalSize) of d(x (+) delta)/d(delta) at delta=0.
 */
class LocalParameterization {
public:
  virtual ~LocalParameterization() = default;

  virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const = 0;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const = 0;
  virtual int GlobalSize() const = 0;
  virtual int LocalSize() const = 0;

  /// Convenience: the (GlobalSize x LocalSize) plus-Jacobian V at x, as an Eigen matrix.
  Eigen::MatrixXd PlusJacobian(const double *x) const {
    const int g = GlobalSize();
    const int l = LocalSize();
    std::vector<double> buf((size_t)g * (size_t)l, 0.0);
    ComputeJacobian(x, buf.data());
    Eigen::MatrixXd V(g, l);
    for (int r = 0; r < g; ++r)
      for (int c = 0; c < l; ++c)
        V(r, c) = buf[(size_t)r * (size_t)l + (size_t)c]; // ComputeJacobian writes row-major
    return V;
  }

  /// Convenience Eigen wrapper for Plus().
  Eigen::VectorXd Plus(const Eigen::VectorXd &x, const Eigen::VectorXd &delta) const {
    Eigen::VectorXd out(GlobalSize());
    Plus(x.data(), delta.data(), out.data());
    return out;
  }
};

/**
 * @brief Trivial Euclidean parameterization: x_plus_delta = x + delta, V = I.
 */
class EuclideanParameterization : public LocalParameterization {
public:
  explicit EuclideanParameterization(int size) : size_(size) {}
  bool Plus(const double *x, const double *delta, double *x_plus_delta) const override {
    for (int i = 0; i < size_; ++i)
      x_plus_delta[i] = x[i] + delta[i];
    return true;
  }
  bool ComputeJacobian(const double *, double *jacobian) const override {
    // Row-major identity (size_ x size_)
    for (int r = 0; r < size_; ++r)
      for (int c = 0; c < size_; ++c)
        jacobian[(size_t)r * (size_t)size_ + (size_t)c] = (r == c) ? 1.0 : 0.0;
    return true;
  }
  int GlobalSize() const override { return size_; }
  int LocalSize() const override { return size_; }

private:
  int size_;
};

} // namespace zbft_sfm
} // namespace ov_init

#endif // OV_INIT_ZBFT_SFM_LOCALPARAMETERIZATION_H
