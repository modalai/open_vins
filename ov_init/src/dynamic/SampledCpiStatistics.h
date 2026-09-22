/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_SAMPLED_CPI_STATISTICS_H
#define OV_INIT_SAMPLED_CPI_STATISTICS_H

#include <array>
#include <cstdint>
#include <Eigen/Core>

namespace ov_init {

// Linear sufficient statistics. A caller must supply the nonlinear CPI's
// transition and original-record derivatives, then preserve retained-record
// correlations through the solve and import. This helper does not enable a
// runtime initializer. Error order: [theta,bg,beta,ba,alpha]; raw noise: [g,a].
class SampledCpiStatistics {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Vector6 = Eigen::Matrix<double, 6, 1>;
  using Vector15 = Eigen::Matrix<double, 15, 1>;
  using Matrix6 = Eigen::Matrix<double, 6, 6>;
  using Matrix15 = Eigen::Matrix<double, 15, 15>;
  using Jacobian = Eigen::Matrix<double, 15, 6>;

  struct Record {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    uint64_t stream_episode = 0, sequence = 0;
    double timestamp = 0.;
    Vector6 measured = Vector6::Zero();
    Matrix6 prior = Matrix6::Zero(); // original zero-mean raw prior, never Q/dt
    Vector6 noise_linearization = Vector6::Zero();
  };

  struct Step {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::array<Record, 2> records;
    double time0 = 0., time1 = 0.;
    Eigen::Vector2d weights0 = Eigen::Vector2d::Zero(), weights1 = Eigen::Vector2d::Zero();
    Matrix15 transition = Matrix15::Identity();
    // These already include the exact producer's stage/interpolation weights.
    // The accumulator cannot infer them from a continuous CPI marginal.
    std::array<Jacobian, 2> noise = {Jacobian::Zero(), Jacobian::Zero()};
    Matrix15 independent_covariance = Matrix15::Zero();
  };

  struct Owner {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Record record;
    Jacobian derivative = Jacobian::Zero();
    bool start_support = false, end_support = false;
  };

  struct Output {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time0 = 0., time1 = 0.;
    uint64_t steps = 0, eliminated_records = 0;
    unsigned owner_count = 0;
    std::array<Owner, 4> owners;
    Matrix15 transition = Matrix15::Identity();
    Vector15 offset = Vector15::Zero();
    Matrix15 conditional_covariance = Matrix15::Zero();
  };

  // Original records are independent unless kept as cut variables by the
  // caller's joint prior. The first support is pinned. Repeated substeps retain
  // the same record; after its successor knot an unpinned record is eliminated
  // once. An adjacent pair must reuse exactly the previous right record.
  // Success and rejection allocate no heap memory. Failure changes nothing.
  bool append(const Step &step);

  // e1 = transition*e0 + sum T_i*(n_i-mu_i) + offset + epsilon.
  // conditional_covariance is Cov(epsilon), EXCLUDING retained record priors.
  // Incoming navigation/boundary correlations belong in the caller's joint
  // model; a separate independent covariance addition would lose them.
  bool export_statistics(Output &output) const;

private:
  Output statistics_;
  uint64_t left_sequence_ = 0, right_sequence_ = 0;
};

} // namespace ov_init
#endif
