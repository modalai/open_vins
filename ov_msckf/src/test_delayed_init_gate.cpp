/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cmath>
#include <cstdio>
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Vec.h"
#include "utils/print.h"

namespace {
int failures = 0;
void check(bool ok, const char *why) {
  if (!ok) {
    ++failures;
    std::printf("FAIL: %s\n", why);
  }
}

// A three-dimensional unknown landmark explains the first three orthogonal
// measurement modes. Only the remaining modes carry a goodness-of-fit test.
// Rotate the complete system so the production Givens elimination must recover
// that split rather than simply selecting the supplied residual tail.
void gate_case(int residual_dof, double statistic, bool expected_accept) {
  using namespace ov_msckf;
  StateOptions options;
  auto state = std::make_shared<State>(options);
  StateHelper::set_initial_covariance(state, .2 * Eigen::MatrixXd::Identity(15, 15), {state->_imu});
  const Eigen::MatrixXd before_cov = StateHelper::get_full_covariance(state);
  const Eigen::MatrixXd before_imu = state->_imu->value();
  const int rows = 3 + residual_dof;
  Eigen::MatrixXd Hx = Eigen::MatrixXd::Zero(rows, 3);
  Eigen::MatrixXd Hf = Eigen::MatrixXd::Zero(rows, 3);
  Hf.topRows(3).setIdentity();
  Hx.topRows(3) = .4 * Eigen::Matrix3d::Identity();
  Eigen::VectorXd residual = Eigen::VectorXd::Zero(rows);
  residual.head(3) << 2., -3., 1.; // explained by the unrestricted new variable
  if (residual_dof > 0) {
    Hx(3, 0) = 1.;
    residual(3) = std::sqrt(1.2 * statistic);
  }
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(rows, rows);
  for (int i = 0; i + 1 < rows; ++i) {
    const double angle = .31 + .07 * i;
    Eigen::JacobiRotation<double> rotation(std::cos(angle), std::sin(angle));
    Q.applyOnTheLeft(i, i + 1, rotation);
  }
  Hx = (Q * Hx).eval();
  Hf = (Q * Hf).eval();
  residual = (Q * residual).eval();
  Eigen::MatrixXd noise = Eigen::MatrixXd::Identity(rows, rows);
  auto variable = std::make_shared<ov_type::Vec>(3);
  const bool accepted = StateHelper::initialize(state, variable, {state->_imu->p()}, Hx, Hf, noise, residual, 1.);
  check(accepted == expected_accept, "gate uses residual degrees of freedom after landmark elimination");
  if (!expected_accept && !accepted) {
    check(StateHelper::get_full_covariance(state) == before_cov && state->_imu->value() == before_imu && variable->id() == -1,
          "rejected delayed initialization leaves state and covariance unchanged");
  }
  if (expected_accept && accepted) {
    check(state->max_covariance_size() == before_cov.rows() + 3, "accepted landmark adds exactly its three coordinates");
    if (residual_dof == 0)
      check(state->_imu->value() == before_imu, "exactly determined initialization does not update the existing state");
  }
}
} // namespace

int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  // Independent 95% chi-square quantiles: df=1: 3.84146; df=3: 7.81473.
  // Both rejected statistics would pass when the three fitted dimensions are
  // incorrectly counted (df=4: 9.48773; df=6: 12.59159).
  gate_case(1, 5., false);
  gate_case(3, 10., false);
  gate_case(1, 3., true);
  gate_case(3, 7., true);
  gate_case(0, 0., true);
  std::printf("DELAYED_INIT_GATE %s failures=%d\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
