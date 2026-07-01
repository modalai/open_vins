/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Contributor: Joao Leonardo Silva Cotta (@zauberflote1)
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

  /// True iff the plus-Jacobian V has the form [I_local; 0], so J_local = J_ambient.leftCols(lsize).
  /// JPL quat and Euclidean satisfy this; S² gravity does not. Default false (safe: always use J*V).
  virtual bool tangent_is_leading_identity() const { return false; }

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
  bool tangent_is_leading_identity() const override { return true; }

private:
  int size_;
};

/**
 * @brief S²(G) gravity parameterization: 3-global, 2-local, fixed magnitude G.
 *
 * The gravity vector g lives on the sphere S²(G) with known radius G = gravity_mag.
 * Retraction: g_plus = G * normalize(g + B(g)*delta), where B(g) is the 3×2 tangent
 * basis (orthonormal columns spanning the plane ⊥ ghat = g/G).
 *
 * Tangent basis rule (deterministic, O(1)): reference axis = argmin_i |ghat_i|
 * (the axis least aligned with ghat), then two Gram–Schmidt passes for orthonormality.
 * This is never degenerate for any reachable g, including the +Z pole.
 */
class GravityS2Parameterization : public LocalParameterization {
public:
  explicit GravityS2Parameterization(double gravity_mag) : G_(gravity_mag) {}

  bool Plus(const double *x, const double *delta, double *x_plus_delta) const override {
    // g_plus = G * normalize(g + B(g)*delta)
    Eigen::Map<const Eigen::Vector3d> g(x);
    Eigen::Map<const Eigen::Vector2d> d(delta);
    Eigen::Matrix<double, 3, 2> B;
    compute_tangent_basis(g, B);
    Eigen::Vector3d gp = g + B * d;
    double n = gp.norm();
    if (n < 1e-12)
      n = 1e-12;
    Eigen::Vector3d result = (G_ / n) * gp;
    x_plus_delta[0] = result(0);
    x_plus_delta[1] = result(1);
    x_plus_delta[2] = result(2);
    return true;
  }

  bool ComputeJacobian(const double *x, double *jacobian) const override {
    // At delta=0, d(g_plus)/d(delta) = B(g) (the tangent basis), row-major 3×2.
    Eigen::Map<const Eigen::Vector3d> g(x);
    Eigen::Matrix<double, 3, 2> B;
    compute_tangent_basis(g, B);
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 2; ++c)
        jacobian[r * 2 + c] = B(r, c);
    return true;
  }

  int GlobalSize() const override { return 3; }
  int LocalSize() const override { return 2; }
  bool tangent_is_leading_identity() const override { return false; }

  double gravity_mag() const { return G_; }

private:
  double G_;

  /// Compute orthonormal tangent basis B(g) ∈ ℝ³ˣ², columns span the plane ⊥ ghat.
  /// Rule: seed = axis least aligned with ghat (argmin_i |ghat_i|), then Gram–Schmidt.
  void compute_tangent_basis(const Eigen::Vector3d &g, Eigen::Matrix<double, 3, 2> &B) const {
    Eigen::Vector3d ghat = g.normalized();

    // Pick the reference axis least aligned with ghat (avoids degeneracy everywhere)
    int min_idx = 0;
    double min_val = std::abs(ghat(0));
    for (int i = 1; i < 3; ++i) {
      if (std::abs(ghat(i)) < min_val) {
        min_val = std::abs(ghat(i));
        min_idx = i;
      }
    }
    Eigen::Vector3d seed = Eigen::Vector3d::Zero();
    seed(min_idx) = 1.0;

    // First basis vector: orthonormalize seed against ghat
    Eigen::Vector3d b1 = seed - ghat.dot(seed) * ghat;
    b1.normalize();

    // Second basis vector: cross product (already orthonormal)
    Eigen::Vector3d b2 = ghat.cross(b1);
    b2.normalize(); // numerically tighten

    B.col(0) = b1;
    B.col(1) = b2;
  }
};

} // namespace zbft_sfm
} // namespace ov_init

#endif // OV_INIT_ZBFT_SFM_LOCALPARAMETERIZATION_H
