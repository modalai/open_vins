/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cmath>
#include <cstdio>
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterHelper.h"
#include "update/UpdaterZeroVelocity.h"
#include "utils/chi_square/chi_squared_quantile_table_0_95.h"
#include "utils/print.h"

namespace {
int failures = 0;
void check(bool ok, const char *why) {
  if (!ok) { ++failures; std::printf("FAIL: %s\n", why); }
}
void dense_oracle(int rows, int cols, bool rank_deficient, double variance) {
  Eigen::MatrixXd H(rows, cols), L = Eigen::MatrixXd::Identity(cols, cols);
  Eigen::VectorXd r(rows);
  for (int i = 0; i < rows; ++i) {
    r(i) = std::cos(1.3 * i) + .2 * std::sin(.7 * i);
    for (int j = 0; j < cols; ++j) H(i,j) = std::sin(.3 + .31 * i + .7 * j) + .07 * (i == j);
  }
  if (rank_deficient && cols > 1) H.col(cols - 1) = H.col(0);
  for (int i = 0; i < cols; ++i) for (int j = 0; j < i; ++j) L(i,j) = .1 * std::sin(i + 2 * j);
  const Eigen::MatrixXd P = L * L.transpose();
  const Eigen::MatrixXd S = H * P * H.transpose() + variance * Eigen::MatrixXd::Identity(rows, rows);
  const auto factor = S.llt();
  const double full_chi2 = r.dot(factor.solve(r));
  const Eigen::VectorXd full_dx = P * H.transpose() * factor.solve(r);
  const Eigen::MatrixXd full_cov = P - P * H.transpose() * factor.solve(H * P);
  Eigen::MatrixXd compressed = H;
  Eigen::VectorXd residual = r;
  double tail_energy = -1;
  ov_msckf::UpdaterHelper::measurement_compress_inplace(compressed, residual, tail_energy);
  const Eigen::MatrixXd Sc = compressed * P * compressed.transpose() + variance * Eigen::MatrixXd::Identity(residual.rows(), residual.rows());
  const auto fc = Sc.llt();
  const double chi2 = residual.dot(fc.solve(residual)) + tail_energy / variance;
  const Eigen::VectorXd dx = P * compressed.transpose() * fc.solve(residual);
  const Eigen::MatrixXd cov = P - P * compressed.transpose() * fc.solve(compressed * P);
  check(std::abs(chi2 - full_chi2) < 1e-10 * (1 + full_chi2), "compressed innovation statistic equals full dense statistic");
  check((dx - full_dx).norm() < 1e-10 && (cov - full_cov).norm() < 1e-10, "compression preserves state update and covariance");
  check(tail_energy >= 0 && (rows > cols || tail_energy == 0), "tail energy initialized for fat/square matrices");
  Eigen::MatrixXd legacy = H; Eigen::VectorXd legacy_r = r;
  ov_msckf::UpdaterHelper::measurement_compress_inplace(legacy, legacy_r);
  check(legacy == compressed && legacy_r == residual, "two-argument compression preserves API and values");
}

void production_motion(bool alternating) {
  using namespace ov_msckf;
  StateOptions config; config.num_cameras = 1; config.imu_model = StateOptions::KALIBR;
  auto state = std::make_shared<State>(config); state->_timestamp = 1;
  Eigen::Matrix<double,16,1> x = Eigen::Matrix<double,16,1>::Zero(); x(3) = 1;
  state->_imu->set_value(x); state->_imu->set_fej(x);
  StateHelper::set_initial_covariance(state, .01 * Eigen::MatrixXd::Identity(15,15), {state->_imu});
  NoiseManager noise; noise.sigma_w = .003; noise.sigma_a = .03; noise.sigma_wb = .0001; noise.sigma_ab = .001;
  UpdaterOptions options; options.chi2_multipler = 1;
  auto database = std::make_shared<ov_core::FeatureDatabase>();
  auto prop = std::make_shared<Propagator>(noise, 9.81);
  UpdaterZeroVelocity update(options, noise, database, prop, 9.81, .03, 1, 0);
  for (int i = 0; i <= 10; ++i) {
    ov_core::ImuData imu; imu.timestamp = 1 + .01 * i;
    imu.am = Eigen::Vector3d(0,0,9.81);
    imu.wm = Eigen::Vector3d(alternating && i >= 4 ? (i % 2 ? -.2 : .2) : 0, 0, 0);
    update.feed_imu(imu);
  }
  const auto before = state->_imu->value();
  const bool accepted = update.try_update(state, 1.1);
  check(accepted != alternating, alternating ? "actual ZUPT rejects zero-mean alternating angular motion" : "actual ZUPT accepts stationary control");
  if (alternating) check(state->_timestamp == 1 && state->_imu->value() == before, "rejected motion leaves state untouched");
}
void large_quantiles() {
  // Independent scipy.stats.chi2.ppf(.95, dof) reference values. The lookup
  // range remains exact; the large-dof approximation avoids saturation.
  const int dofs[] = {4096,5000,10000,100000,2147483647};
  const double reference[] = {4246.0034833375275,5165.614518675803,10233.748897677937,100736.736177319,2147591445.264288};
  double previous = ov_core::chi_squared_quantile_0_95(4095);
  for (int i = 0; i < 5; ++i) {
    const double q = ov_core::chi_squared_quantile_0_95(dofs[i]);
    check(q > previous && std::abs(q / reference[i] - 1) < 5.5e-8, "large-dof threshold grows and matches independent quantile reference");
    previous = q;
  }
}
}
int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  for (double variance : {.2, 1., 20.}) for (bool deficient : {false, true}) {
    dense_oracle(60,9,deficient,variance);
    dense_oracle(9,9,deficient,variance);
    dense_oracle(6,9,deficient,variance);
  }
  production_motion(false); production_motion(true); large_quantiles();
  std::printf("ZUPT_COMPRESSION %s failures=%d\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
