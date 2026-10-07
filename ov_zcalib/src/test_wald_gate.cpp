#include "core/WaldObservability.h"

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace ov_zcalib;

int main() {
  int failures = 0;
  auto check = [&](bool ok, const char *why) {
    if (!ok) {
      ++failures;
      std::printf("FAIL: %s\n", why);
    }
  };
  {
    Eigen::MatrixXd Lgg = 4.0 * Eigen::MatrixXd::Identity(2, 2), reduced;
    Eigen::VectorXd gg = Eigen::VectorXd::Ones(2), gradient;
    check(wald_marginalize(Lgg, Eigen::MatrixXd(2, 0), Eigen::MatrixXd(0, 0), gg,
                           Eigen::VectorXd(0), reduced, gradient) && reduced.isApprox(Lgg) && gradient.isApprox(gg),
          "an empty nuisance set must preserve direct information and gradient");
    Eigen::MatrixXd Lgo(2, 1), Loo(1, 1);
    Lgo << 1.0, 2.0;
    Loo << 2.0;
    Eigen::VectorXd go = Eigen::VectorXd::Constant(1, 3.0);
    check(wald_marginalize(Lgg, Lgo, Loo, gg, go, reduced, gradient) &&
              reduced.isApprox(Lgg - 0.5 * Lgo * Lgo.transpose()) && gradient.isApprox(gg - 1.5 * Lgo.col(0)),
          "valid nuisance profiling must retain its information and gradient cross terms");
    Loo(0, 0) = -2.0;
    check(!wald_marginalize(Lgg, Lgo, Loo, gg, go, reduced, gradient),
          "an unbounded nuisance quadratic must not create positive gate information");
  }
  {
    Eigen::MatrixXd information = Eigen::MatrixXd::Identity(2, 2);
    Eigen::LDLT<Eigen::MatrixXd> factor;
    check(wald_positive_ldlt(information, factor), "positive nuisance information must remain valid");
    information(0, 0) = -1.0;
    const Eigen::LDLT<Eigen::MatrixXd> legacy(information);
    check(legacy.info() == Eigen::Success, "indefinite counterexample must pass the former LDLT-success check");
    check(!wald_positive_ldlt(information, factor), "indefinite nuisance minimization must not become Wald evidence");
    information(0, 0) = 0.0;
    check(!wald_positive_ldlt(information, factor), "singular nuisance minimization must refuse");
    information(0, 0) = std::numeric_limits<double>::quiet_NaN();
    check(!wald_positive_ldlt(information, factor), "NaN nuisance information must refuse under fast-math");
    information(0, 0) = std::numeric_limits<double>::infinity();
    check(!wald_positive_ldlt(information, factor), "infinite nuisance information must refuse under fast-math");
    check(wald_statistics_valid(0.0, 28.9, 0.0, 2.0, 83.7, 126.9), "valid zero disagreement must remain valid");
    for (int i = 0; i < 6; ++i) {
      for (double invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.0}) {
        double values[6] = {1.0, 28.9, 1.0, 2.0, 83.7, 126.9};
        values[i] = invalid;
        check(!wald_statistics_valid(values[0], values[1], values[2], values[3], values[4], values[5]),
              "every invalid statistic/threshold must abstain instead of passing an ordered comparison");
      }
    }
  }
  constexpr double kappa = 2.0, floor = 4.5;
  for (int n : {2, 6, 9}) {
    Eigen::MatrixXd first = 10.0 * Eigen::MatrixXd::Identity(n, n), second = first;
    first(0, 1) = first(1, 0) = 9.9;
    second(0, 1) = second(1, 0) = -9.9;
    // The pooled system is 20 I. Reproduce the old Rayleigh test exactly on
    // its returned basis: every coordinate clears 4.5 after kappa deflation.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> pooled(first + second);
    bool old_test_passes = true;
    for (int j = 0; j < n; ++j) {
      const Eigen::VectorXd v = pooled.eigenvectors().col(j);
      old_test_passes = old_test_passes && std::min(v.dot(first * v), v.dot(second * v)) / kappa >= floor;
    }
    check(old_test_passes, "counterexample must pass the former pooled-basis test");
    const Eigen::MatrixXd P = pooled.eigenvectors();
    const WaldHalfObservability weak = wald_half_observability(P.transpose() * first * P, P.transpose() * second * P,
                                                              kappa, floor);
    check(weak.spectra_ok && !weak.meets_floor, "opposing half cross terms must not certify a weak direction");
    check(std::abs(weak.min_deflated_eig - .05) < 1e-12, "information floor must retain variance/kappa normalization");

    first(0, 1) = first(1, 0) = .5;
    second(0, 1) = second(1, 0) = -.5;
    const WaldHalfObservability strong = wald_half_observability(first, second, kappa, floor);
    check(strong.spectra_ok && strong.meets_floor && std::abs(strong.min_deflated_eig - 4.75) < 1e-12,
          "well-observed positive control must continue to pass");
    std::printf("Wald floor dim=%d: former basis passes weak case, true min-half %.6g; positive %.6g\n",
                n, weak.min_deflated_eig, strong.min_deflated_eig);
  }

  Eigen::Matrix2d first, second;
  first << 10.0, 9.9, 9.9, 10.0;
  second << 10.0, -9.9, -9.9, 10.0;
  for (double angle : {0.0, .13, .5, .7853981633974483, 1.1}) {
    Eigen::Matrix2d Q;
    Q << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
    const auto weak = wald_half_observability(Q.transpose() * first * Q, Q.transpose() * second * Q, kappa, floor);
    check(weak.spectra_ok && !weak.meets_floor && std::abs(weak.min_deflated_eig - .05) < 1e-12,
          "observability must be invariant to an orthogonal basis change within the tested span");
    const auto scaled = wald_half_observability(3.0 * Q.transpose() * first * Q, 3.0 * Q.transpose() * second * Q,
                                              3.0 * kappa, floor);
    check(scaled.spectra_ok && !scaled.meets_floor && std::abs(scaled.min_deflated_eig - .05) < 1e-12,
          "matching information and variance scaling must preserve the decision");
  }
  first.setIdentity();
  first *= kappa * floor;
  check(wald_half_observability(first, first, kappa, floor).meets_floor, "the existing inclusive floor must remain inclusive");
  first(0, 0) -= .01;
  check(!wald_half_observability(first, first, kappa, floor).meets_floor, "a mode below the same floor must refuse");
  first(0, 0) = std::numeric_limits<double>::quiet_NaN();
  check(!wald_half_observability(first, second, kappa, floor).spectra_ok, "nonfinite half information must refuse under fast-math");
  first(0, 0) = std::numeric_limits<double>::infinity();
  check(!wald_half_observability(first, second, kappa, floor).spectra_ok, "infinite half information must refuse under fast-math");
  first(0, 0) = 0.0;
  check(!wald_half_observability(first, first, kappa, floor).meets_floor, "singular half information must refuse");
  first(0, 0) = -1.0;
  check(!wald_half_observability(first, first, kappa, floor).meets_floor, "indefinite half information must refuse");
  check(!wald_half_observability(Eigen::MatrixXd(), Eigen::MatrixXd(), kappa, floor).spectra_ok,
        "empty projected spans must refuse");

  // Preserve every existing lower-dimensional table entry and the denominator
  // bucket policy. The new Tg dimensions must match independent quantiles.
  const double old_chi2[6] = {6.635, 9.210, 11.345, 13.277, 15.086, 16.812};
  const double old_f[4][6] = {{13.75, 10.92, 9.78, 9.15, 8.75, 8.47}, {9.33, 6.93, 5.95, 5.41, 5.06, 4.82},
                            {8.29, 5.93, 5.09, 4.58, 4.25, 4.02}, {7.82, 5.61, 4.72, 4.22, 3.90, 3.67}};
  double worst_quantile_error = 0.0;
  for (int r = 1; r <= 9; ++r) {
    for (int df : {0, 5, 6, 11, 12, 17, 18, 23, 24, 80}) {
      const int bucket = df >= 24 ? 3 : df >= 18 ? 2 : df >= 12 ? 1 : 0;
      const double threshold = wald_statistic_threshold(r, df, 1.0);
      if (r <= 6) {
        const double previous = df >= 6 ? r * old_f[bucket][r - 1] : old_chi2[r - 1];
        check(threshold == previous, "r<=6 thresholds and dispersion buckets must remain exactly unchanged");
      } else {
        const double reference = df >= 6
            ? r * boost::math::quantile(boost::math::fisher_f_distribution<double>(r, 6 * (bucket + 1)), .99)
            : boost::math::quantile(boost::math::chi_squared_distribution<double>(r), .99);
        const double error = std::abs(threshold - reference);
        worst_quantile_error = std::max(worst_quantile_error, error);
        check(error < 1e-10, "r=7..9 thresholds must use the correct numerator degrees of freedom at probability .99");
      }
      check(std::abs(wald_statistic_threshold(r, df, 1.5) - 1.5 * threshold) < 1e-12,
            "existing threshold scale must apply unchanged");
    }
  }
  check(!finite_scalar(wald_statistic_threshold(0, 18, 1.0)) && !finite_scalar(wald_statistic_threshold(10, 18, 1.0)),
        "unsupported Wald dimensions must not silently clamp to another table column");
  std::printf("Wald r=9: chi-square %.12g, df18 F threshold %.12g; max Boost quantile error %.3g\n",
              wald_statistic_threshold(9, 0, 1.0), wald_statistic_threshold(9, 18, 1.0), worst_quantile_error);
  std::printf("%s Wald whole-span information floor: cancellation, 6/9-dof, positive, basis/scaling invariance and invalid input\n",
              failures ? "FAIL" : "PASS");
  std::printf("%s Wald threshold dimensions: legacy r<=6 exact, r=7..9 chi-square/F99, buckets and scale preserved\n",
              failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
