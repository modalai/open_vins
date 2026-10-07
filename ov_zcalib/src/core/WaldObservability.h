#ifndef OV_ZCALIB_WALD_OBSERVABILITY_H
#define OV_ZCALIB_WALD_OBSERVABILITY_H

#include <Eigen/Dense>
#include <algorithm>
#include <limits>

#include "../utils/NumericChecks.h"

namespace ov_zcalib {

template <typename Derived> inline bool wald_finite(const Eigen::MatrixBase<Derived> &value) {
  for (Eigen::Index j = 0; j < value.cols(); ++j)
    for (Eigen::Index i = 0; i < value.rows(); ++i)
      if (!finite_scalar(value(i, j)))
        return false;
  return true;
}

// A successful Eigen LDLT can still factor an indefinite matrix. Such a
// nuisance quadratic has no finite minimum and cannot be Schur-marginalized
// into evidence for an acceptance decision.
inline bool wald_positive_ldlt(const Eigen::MatrixXd &information, Eigen::LDLT<Eigen::MatrixXd> &factor) {
  if (information.rows() == 0 || information.rows() != information.cols() || !wald_finite(information))
    return false;
  factor.compute(information);
  if (factor.info() != Eigen::Success)
    return false;
  for (Eigen::Index i = 0; i < factor.vectorD().size(); ++i)
    if (!finite_scalar(factor.vectorD()(i)) || !(factor.vectorD()(i) > 0.0))
      return false;
  return true;
}

inline bool wald_marginalize(const Eigen::MatrixXd &Lgg, const Eigen::MatrixXd &Lgo, const Eigen::MatrixXd &Loo,
                            const Eigen::VectorXd &gg, const Eigen::VectorXd &go,
                            Eigen::MatrixXd &information, Eigen::VectorXd &gradient) {
  if (Lgg.rows() == 0 || Lgg.rows() != Lgg.cols() || Lgo.rows() != Lgg.rows() || Lgo.cols() != Loo.rows() ||
      Loo.rows() != Loo.cols() || gg.size() != Lgg.rows() || go.size() != Loo.rows() ||
      !wald_finite(Lgg) || !wald_finite(Lgo) || !wald_finite(gg) || !wald_finite(go))
    return false;
  information = Lgg;
  gradient = gg;
  // No nuisance coordinates is a valid direct information system.
  if (Loo.rows() != 0) {
    Eigen::LDLT<Eigen::MatrixXd> factor;
    if (!wald_positive_ldlt(Loo, factor))
      return false;
    information.noalias() -= Lgo * factor.solve(Lgo.transpose());
    gradient.noalias() -= Lgo * factor.solve(go);
  }
  return wald_finite(information) && wald_finite(gradient);
}

// Ordered comparisons with NaN are not a rejection policy, particularly
// under -ffast-math. All quadratic statistics and thresholds must first be
// finite and nonnegative, including the cross-prediction thresholds.
inline bool wald_statistics_valid(double statistic, double threshold, double cross_first, double cross_second,
                                  double cross_threshold_first, double cross_threshold_second) {
  for (double value : {statistic, threshold, cross_first, cross_second, cross_threshold_first, cross_threshold_second})
    if (!finite_scalar(value) || value < 0.0)
      return false;
  return true;
}

struct WaldHalfObservability {
  bool spectra_ok = false;
  bool meets_floor = false;
  double min_eig_first = -std::numeric_limits<double>::infinity();
  double min_eig_second = -std::numeric_limits<double>::infinity();
  double min_deflated_eig = -std::numeric_limits<double>::infinity();
};

/// Check the entire projected span in BOTH prior-whitened half systems.
/// A diagonal/Rayleigh test in the pooled eigenbasis is insufficient: opposite
/// cross terms can cancel in the sum while either half has a weak direction.
/// variance_scale has the existing kappa semantics: information is divided by it.
inline WaldHalfObservability wald_half_observability(const Eigen::MatrixXd &first, const Eigen::MatrixXd &second,
                                                    double variance_scale, double information_floor) {
  WaldHalfObservability out;
  if (first.rows() == 0 || first.rows() != first.cols() || second.rows() != first.rows() || second.cols() != first.cols() ||
      !finite_scalar(variance_scale) || variance_scale <= 0.0 || !finite_scalar(information_floor) || information_floor < 0.0)
    return out;
  for (Eigen::Index i = 0; i < first.size(); ++i)
    if (!finite_scalar(first.data()[i]) || !finite_scalar(second.data()[i]))
      return out;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> e1(first), e2(second);
  if (e1.info() != Eigen::Success || e2.info() != Eigen::Success)
    return out;
  out.min_eig_first = e1.eigenvalues()(0);
  out.min_eig_second = e2.eigenvalues()(0);
  out.min_deflated_eig = std::min(out.min_eig_first, out.min_eig_second) / variance_scale;
  out.spectra_ok = finite_scalar(out.min_eig_first) && finite_scalar(out.min_eig_second) && finite_scalar(out.min_deflated_eig);
  out.meets_floor = out.spectra_ok && out.min_deflated_eig >= information_floor;
  return out;
}

/// Existing 99% Wald sizing, including its denominator-df bucket policy.
/// Keep the historical r<=6 entries exactly; Tg's r=7..9 must use its own
/// numerator degrees of freedom rather than silently indexing the r=6 column.
inline double wald_statistic_threshold(int dimensions, int dispersion_df, double scale) {
  if (dimensions < 1 || dimensions > 9 || !finite_scalar(scale) || scale < 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  static const double chi2_99[9] = {
      6.635, 9.210, 11.345, 13.277, 15.086, 16.812,
      18.475306906582357, 20.090235029663233, 21.665994333461924};
  static const double f99[4][9] = {
      {13.75, 10.92, 9.78, 9.15, 8.75, 8.47, 8.259995270968982, 8.101651366738700, 7.9761213666233575}, // df ~ 6
      {9.33, 6.93, 5.95, 5.41, 5.06, 4.82, 4.639502446564337, 4.499365280847432, 4.387509963180189}, // df ~ 12
      {8.29, 5.93, 5.09, 4.58, 4.25, 4.02, 3.840638659897973, 3.705421881172038, 3.5970739135457515}, // df ~ 18
      {7.82, 5.61, 4.72, 4.22, 3.90, 3.67, 3.4959275204932747, 3.362867119949481, 3.2559850744613916}, // df ~ 24+
  };
  if (dispersion_df >= 6) {
    const int bucket = dispersion_df >= 24 ? 3 : dispersion_df >= 18 ? 2 : dispersion_df >= 12 ? 1 : 0;
    return scale * dimensions * f99[bucket][dimensions - 1];
  }
  return scale * chi2_99[dimensions - 1];
}

} // namespace ov_zcalib

#endif // OV_ZCALIB_WALD_OBSERVABILITY_H
