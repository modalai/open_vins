/* Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace {
using namespace ov_core;
using V3 = Eigen::Vector3d;
using Poses = std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>>;
using Scalar = long double;
using Ref3 = Eigen::Matrix<Scalar, 3, 1>;
using Design = Eigen::Matrix<Scalar, Eigen::Dynamic, 3>;
using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
struct Probe : FeatureInitializer {
  using FeatureInitializer::FeatureInitializer;
  using FeatureInitializer::compute_error;
};
int checks = 0, failures = 0;
double max_relative_cost = 0., max_solution_error = 0.;
void check(bool ok, const char *message) {
  ++checks;
  if (!ok) { ++failures; std::printf("FAIL: %s\n", message); }
}

struct Fixture {
  Poses poses;
  std::shared_ptr<Feature> feature = std::make_shared<Feature>();
  Design design;
  Values observed;
  Eigen::Matrix3d rotation;
  V3 origin;
  Fixture(V3 inverse, double baseline, bool tilted, double noise) {
    rotation = tilted ? Eigen::AngleAxisd(.37, V3(.3, -.4, .5).normalized()).toRotationMatrix() : Eigen::Matrix3d::Identity();
    origin = tilted ? V3(.31, -.27, .19) : V3::Zero();
    const std::array<Eigen::Vector2d, 5> centers{{{0., 0.}, {.25, .125}, {.5, -.25}, {.75, .5}, {-.5, -.125}}};
    design.resize(10, 3); observed.resize(10);
    feature->anchor_cam_id = 0; feature->anchor_clone_timestamp = 10.;
    for (size_t i = 0; i < centers.size(); ++i) {
      const V3 local(baseline * centers[i](0), baseline * centers[i](1), 0.);
      const double time = 10. + .1 * i;
      poses[0].emplace(time, FeatureInitializer::ClonePose(rotation, origin + rotation.transpose() * local));
      Eigen::Vector2f measured;
      measured << inverse(0) - local(0) * inverse(2) + noise * std::sin(.6 + 1.7 * i),
                  inverse(1) - local(1) * inverse(2) + noise * std::cos(.9 + 1.3 * i);
      feature->timestamps[0].push_back(time);
      feature->uvs_norm[0].push_back(measured);
      feature->uvs[0].push_back(measured);
      // Independent affine projection in the common camera frame. The oracle
      // solves the complete tall design in long double, not normal equations.
      design.row(2 * i) << 1.L, 0.L, -Scalar(local(0));
      design.row(2 * i + 1) << 0.L, 1.L, -Scalar(local(1));
      observed(2 * i) = Scalar(measured(0)); observed(2 * i + 1) = Scalar(measured(1));
    }
  }
  void seed(const V3 &inverse) {
    feature->p_FinA << inverse(0) / inverse(2), inverse(1) / inverse(2), 1. / inverse(2);
    feature->p_FinG = rotation.transpose() * feature->p_FinA + origin;
  }
  V3 inverse() const {
    const V3 &p = feature->p_FinA;
    return {p(0) / p(2), p(1) / p(2), 1. / p(2)};
  }
  Scalar cost(const V3 &value) const { return (observed - design * value.cast<Scalar>()).squaredNorm(); }
};

void plateau() {
  FeatureInitializerOptions options;
  Probe initializer(options);
  const V3 truth(.5, .25, .25);
  Fixture fixture(truth, 1., false, 0.);
  const V3 seed = truth + V3(std::ldexp(1., -32), -std::ldexp(1., -33), std::ldexp(1., -34));
  const Scalar expected = fixture.cost(seed);
  const double actual = initializer.compute_error(fixture.poses, fixture.feature, seed(0), seed(1), seed(2));
  std::printf("PLATEAU actual_cost=%.17g reference_cost=%.21Lg\n", actual, expected);
  check(expected > 0.L && actual > 0., "sub-float projection changes retain nonzero objective");
  check(std::abs(Scalar(actual) - expected) < 1e-12L * expected, "objective equals independent dyadic quadratic");
  const double h = std::ldexp(1., -36);
  V3 plus = seed, minus = seed; plus(0) += h; minus(0) -= h;
  const double derivative = (initializer.compute_error(fixture.poses, fixture.feature, plus(0), plus(1), plus(2)) -
                             initializer.compute_error(fixture.poses, fixture.feature, minus(0), minus(1), minus(2))) / (2. * h);
  const Ref3 gradient = 2.L * fixture.design.transpose() * (fixture.design * seed.cast<Scalar>() - fixture.observed);
  check(std::abs(Scalar(derivative) - gradient(0)) < 1e-12L * std::abs(gradient(0)),
        "cost derivative matches continuous least-squares gradient below float ULP");
  fixture.seed(seed);
  check(initializer.single_gaussnewton(fixture.feature, fixture.poses), "small valid refinement is accepted");
  const Scalar after = fixture.cost(fixture.inverse());
  std::printf("PLATEAU_REFINEMENT before=%.21Lg after=%.21Lg\n", expected, after);
  check(after < expected * 1e-4L, "default refinement follows nonzero sub-float gradient");
}

void noisy() {
  FeatureInitializerOptions options;
  options.max_runs = 30; options.min_dx = 1e-13; options.min_dcost = 1e-13;
  Probe initializer(options);
  for (bool tilted : {false, true}) for (double baseline : {.5, 1., 2.})
    for (double normalized_x : {1e-4, .5, 1.3}) {
      const V3 truth(normalized_x, .2, .25);
      Fixture fixture(truth, baseline, tilted, 2e-4);
      const Ref3 oracle = fixture.design.colPivHouseholderQr().solve(fixture.observed);
      const V3 query = oracle.cast<double>() + V3(1e-5, -2e-5, 3e-5);
      const Scalar expected = fixture.cost(query);
      const double actual = initializer.compute_error(fixture.poses, fixture.feature, query(0), query(1), query(2));
      const double relative = double(std::abs(Scalar(actual) - expected) / expected);
      max_relative_cost = std::max(max_relative_cost, relative);
      check(relative < 2e-10, "noisy cost preserves fixed float observations with double predictions");
      fixture.seed(oracle.cast<double>() + V3(.012, -.009, .006));
      check(initializer.single_gaussnewton(fixture.feature, fixture.poses), "valid noisy refinement converges");
      const double solution_error = (fixture.inverse().cast<Scalar>() - oracle).cwiseAbs().maxCoeff();
      max_solution_error = std::max(max_solution_error, solution_error);
      check(solution_error < 2e-10, "refinement equals independent tall long-double least squares");
      const V3 world = fixture.rotation.transpose() * fixture.feature->p_FinA + fixture.origin;
      check((fixture.feature->p_FinG - world).norm() < 2e-13, "anchor/global output means agree");
    }
}
} // namespace

int main() {
  plateau(); noisy();
  std::printf("FEATURE_REFINEMENT checks=%d failures=%d max_relative_cost=%.12g max_solution_error=%.12g\n",
              checks, failures, max_relative_cost, max_solution_error);
  return failures ? 1 : 0;
}
