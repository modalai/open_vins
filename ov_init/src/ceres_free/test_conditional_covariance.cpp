// Conditional initialization uncertainty: independent dense least-squares and
// finite-difference re-solve controls. No camera or IMU truth is assumed here.
#include "Problem.h"

#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

using namespace ov_init::zbft_sfm;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace {
int checks = 0, failures = 0;
double max_cov = 0.0, max_sensitivity = 0.0, max_fd = 0.0;
void check(bool ok, const char *message) {
  ++checks;
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
  }
}
double error(const MatrixXd &a, const MatrixXd &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols())
    return std::numeric_limits<double>::infinity();
  return a.size() ? (a - b).cwiseAbs().maxCoeff() : 0.0;
}
bool bit_equal(const MatrixXd &a, const MatrixXd &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

class LinearFactor final : public CostFunction {
public:
  std::vector<MatrixXd> A;
  VectorXd target;
  LinearFactor(std::vector<MatrixXd> matrices, VectorXd y) : A(std::move(matrices)), target(std::move(y)) {
    set_num_residuals(target.size());
    for (const auto &matrix : A)
      mutable_parameter_block_sizes()->push_back(matrix.cols());
  }
  bool Evaluate(double const *const *p, double *r, double **jac) const override {
    Eigen::Map<VectorXd> residual(r, target.size());
    residual = -target;
    for (size_t i = 0; i < A.size(); ++i) {
      residual.noalias() += A[i] * Eigen::Map<const VectorXd>(p[i], A[i].cols());
      if (jac && jac[i])
        Eigen::Map<RowMatrix>(jac[i], A[i].rows(), A[i].cols()) = A[i];
    }
    return true;
  }
};

void exercise(bool rank_deficient, bool robust, int threads) {
  std::mt19937 rng(8301);
  std::normal_distribution<double> normal;
  auto random = [&](int rows, int cols) {
    MatrixXd x(rows, cols);
    for (int i = 0; i < x.size(); ++i)
      x.data()[i] = normal(rng);
    return x;
  };
  // Dense oracle order: x=(a3,n2,f3,g3), c=(c2,d1). The solver registration,
  // requested output and consider orders intentionally differ from this order.
  MatrixXd A = MatrixXd::Zero(25, 11), B = random(25, 3);
  VectorXd target = random(25, 1);
  A.block(0, 0, 9, 5) = random(9, 5);
  A.block(0, 5, 9, 3) = random(9, 3);
  A.block(9, 0, 11, 5) = random(11, 5);
  A.block(9, 8, 11, 3) = random(11, 3);
  A.block(20, 0, 5, 5).setIdentity();
  // This prior is (x - T c), so its B columns explicitly encode a correlated
  // state/calibration prior rather than assuming independent marginal sigmas.
  B.bottomRows(5) *= -0.15;
  if (rank_deficient)
    A.col(7).setZero();

  double a[3] = {}, n[2] = {}, f[3] = {}, g[3] = {}, c[2] = {0.1, -0.2}, d[1] = {0.05};
  Problem problem;
  problem.AddParameterBlock(f, 3);
  problem.SetSchurLandmark(f);
  problem.AddParameterBlock(c, 2);
  problem.SetParameterBlockConstant(c);
  problem.AddParameterBlock(a, 3);
  problem.AddParameterBlock(d, 1);
  problem.SetParameterBlockConstant(d);
  problem.AddParameterBlock(g, 3);
  problem.SetSchurLandmark(g);
  problem.AddParameterBlock(n, 2);
  LinearFactor first({A.block(0, 0, 9, 3), A.block(0, 3, 9, 2), A.block(0, 5, 9, 3),
                      B.block(0, 0, 9, 2), B.block(0, 2, 9, 1)}, target.head(9));
  LinearFactor second({A.block(9, 0, 11, 3), A.block(9, 3, 11, 2), A.block(9, 8, 11, 3),
                       B.block(9, 0, 11, 2), B.block(9, 2, 11, 1)}, target.segment(9, 11));
  LinearFactor prior({A.block(20, 0, 5, 3), A.block(20, 3, 5, 2),
                      B.block(20, 0, 5, 2), B.block(20, 2, 5, 1)}, target.tail(5));
  HuberLoss loss(0.6);
  problem.AddResidualBlock(&first, robust ? &loss : nullptr, {a, n, f, c, d});
  problem.AddResidualBlock(&second, robust ? &loss : nullptr, {a, n, g, c, d});
  problem.AddResidualBlock(&prior, nullptr, {a, n, c, d});
  SolverOptions options;
  options.num_threads = threads;
  options.max_solver_time_seconds = 10.0;
  options.max_num_iterations = 60;
  options.function_tolerance = 1e-14;
  options.parameter_tolerance = 1e-13;
  options.gradient_tolerance = 1e-12;

  MatrixXd Aw = A, Bw = B;
  if (robust) {
    Eigen::Vector3d calibration(c[0], c[1], d[0]);
    const VectorXd residual = B * calibration - target;
    for (const auto span : {std::pair<int, int>{0, 9}, {9, 11}}) {
      const double weight = std::sqrt(std::min(1.0, 0.6 / residual.segment(span.first, span.second).norm()));
      Aw.middleRows(span.first, span.second) *= weight;
      Bw.middleRows(span.first, span.second) *= weight;
    }
  }
  // Independent rectangular minimum-norm QR. No problem Schur blocks or rank
  // helper are used by this oracle; a zero landmark direction is harmless.
  const MatrixXd gain = Aw.completeOrthogonalDecomposition().solve(MatrixXd::Identity(25, 25));
  const MatrixXd denseQ = gain * gain.transpose(), denseS = -gain * Bw;
  const int rows[] = {3, 4, 0, 1, 2}, cols[] = {2, 0, 1};
  MatrixXd expectedQ(5, 5), expectedS(5, 3);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j)
      expectedQ(i, j) = denseQ(rows[i], rows[j]);
    for (int j = 0; j < 3; ++j)
      expectedS(i, j) = denseS(rows[i], cols[j]);
  }
  MatrixXd Q, S, before, after;
  check(problem.ComputeCovariance({n, a}, before, options), "ordinary conditional covariance exists");
  check(problem.ComputeConditionalCovariance({n, a}, {d, c}, Q, S, options), "conditional export succeeds");
  max_cov = std::max(max_cov, error(Q, expectedQ));
  max_sensitivity = std::max(max_sensitivity, error(S, expectedS));
  check(error(Q, expectedQ) < 2e-10, "requested covariance matches independent rectangular QR");
  check(error(S, expectedS) < 2e-10, "landmark-eliminated sensitivity matches independent rectangular QR");
  check(error(Q, before) < 2e-10, "adding consider columns does not change conditional covariance");
  check(problem.ComputeCovariance({n, a}, after, options) && bit_equal(before, after), "ordinary export unchanged after consider export");
  check(c[0] == 0.1 && c[1] == -0.2 && d[0] == 0.05, "consider means remain fixed");
  check(Eigen::Map<Eigen::Vector3d>(a).isZero(0) && Eigen::Map<Eigen::Vector2d>(n).isZero(0), "navigation means are untouched");

  Eigen::Matrix3d Pc;
  Pc << 0.02, 0.003, -0.001, 0.003, 0.015, 0.002, -0.001, 0.002, 0.01;
  const MatrixXd joint = Q + S * Pc * S.transpose();
  check(error(joint, expectedQ + expectedS * Pc * expectedS.transpose()) < 2e-10,
        "retained correlated calibration uncertainty matches dense estimator error");
  check((joint - Q).norm() > 1e-4, "zero-sensitivity negative control changes covariance");
  MatrixXd subsetQ, subsetS;
  check(problem.ComputeConditionalCovariance({a}, {d, c}, subsetQ, subsetS, options) &&
        error(subsetQ, expectedQ.bottomRightCorner(3, 3)) < 2e-10 &&
        error(subsetS, expectedS.bottomRows(3)) < 2e-10, "unrequested navigation variables remain marginalized");
  MatrixXd emptyQ, emptyS;
  check(problem.ComputeConditionalCovariance({n, a}, {}, emptyQ, emptyS, options) &&
        error(emptyQ, before) < 2e-10 && emptyS.rows() == 5 && emptyS.cols() == 0, "empty consider set reduces to ordinary covariance");

  if (!rank_deficient && !robust) {
    const double h = 2e-5;
    MatrixXd finite_difference(5, 3);
    double *coordinates[] = {d, c, c + 1};
    for (int j = 0; j < 3; ++j) {
      const double original = *coordinates[j];
      VectorXd plus(5), minus(5);
      for (int sign : {1, -1}) {
        std::fill_n(a, 3, 0.0); std::fill_n(n, 2, 0.0);
        std::fill_n(f, 3, 0.0); std::fill_n(g, 3, 0.0);
        *coordinates[j] = original + sign * h;
        const auto solved = problem.Solve(options);
        check(!solved.time_stopped, "finite-difference solve completes without timing cap");
        VectorXd &value = sign > 0 ? plus : minus;
        value << n[0], n[1], a[0], a[1], a[2];
        check(*coordinates[j] == original + sign * h, "subsequent solve still holds consider variable constant");
      }
      *coordinates[j] = original;
      finite_difference.col(j) = (plus - minus) / (2.0 * h);
    }
    max_fd = std::max(max_fd, error(finite_difference, S));
    check(error(finite_difference, S) < 5e-7, "conditional-fit sensitivity matches independent re-solves");
  }

  // Input validation and failed information assembly must not publish partial
  // output or accidentally release fixed calibration for a later solve.
  MatrixXd sentinelQ = MatrixXd::Constant(2, 2, 42), sentinelS = MatrixXd::Constant(2, 1, -7);
  double unregistered = 0.0;
  for (const auto &bad : std::vector<std::vector<double *>>{{c, c}, {a}, {f}, {&unregistered}}) {
    Q = sentinelQ; S = sentinelS;
    check(!problem.ComputeConditionalCovariance({a}, bad, Q, S, options) && bit_equal(Q, sentinelQ) && bit_equal(S, sentinelS),
          "invalid consider request is rejected atomically");
  }
  Q = sentinelQ; S = sentinelS;
  check(!problem.ComputeConditionalCovariance({a, a}, {c}, Q, S, options) && bit_equal(Q, sentinelQ) && bit_equal(S, sentinelS),
        "duplicate output request is rejected atomically");
  check(!problem.ComputeConditionalCovariance({a}, {c}, Q, Q, options) && bit_equal(Q, sentinelQ), "aliased output matrices are rejected");
  first.A[0](0, 0) = std::numeric_limits<double>::quiet_NaN();
  check(!problem.ComputeConditionalCovariance({a}, {c}, Q, S, options) && bit_equal(Q, sentinelQ) && bit_equal(S, sentinelS),
        "nonfinite linearization is rejected atomically under optimized math");
  first.A[0](0, 0) = A(0, 0);
  check(problem.ComputeCovariance({a}, after, options), "ordinary covariance recovers after failed conditional export");
}
} // namespace

int main() {
  for (bool rank_deficient : {false, true})
    for (bool robust : {false, true})
      for (int threads : {1, 3})
        exercise(rank_deficient, robust, threads);
  std::printf("conditional covariance: %d/%d passed; max Q %.3e S %.3e re-solve FD %.3e\n",
              checks - failures, checks, max_cov, max_sensitivity, max_fd);
  return failures ? 1 : 0;
}
