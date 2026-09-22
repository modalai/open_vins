// Invalid factor evaluations cannot become convergence or finite certification.
#include "Problem.h"
#include "LandmarkQr.h"

#include <cstdio>
#include <cstring>
#include <limits>

using namespace ov_init::zbft_sfm;
namespace {
int checks = 0, failures = 0;
void check(bool value, const char *why) {
  ++checks;
  if (!value) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", why);
  }
}
class FaultFactor final : public CostFunction {
public:
  int mode = 0;
  bool after_step = false;
  FaultFactor() { set_num_residuals(1); mutable_parameter_block_sizes()->push_back(1); }
  bool Evaluate(double const *const *p, double *r, double **j) const override {
    r[0] = p[0][0] - 1.0;
    if (j && j[0]) j[0][0] = 1.0;
    if (after_step && p[0][0] == 0.0) return true;
    if (mode == 1) return false;
    if (mode == 2) r[0] = std::numeric_limits<double>::quiet_NaN();
    if (mode == 3) r[0] = std::numeric_limits<double>::infinity();
    if (mode == 4 && j && j[0]) j[0][0] = std::numeric_limits<double>::quiet_NaN();
    if (mode == 5) r[0] = 1e200; // finite input, overflowing norm
    if (mode == 6 && j && j[0]) j[0][0] = 1e200; // finite input, overflowing information
    return true;
  }
};
class FaultLoss final : public LossFunction {
public:
  int mode = 0;
  void Evaluate(double s, double rho[2]) const override {
    rho[0] = s;
    rho[1] = 1.0;
    if (mode == 1) rho[0] = std::numeric_limits<double>::quiet_NaN();
    if (mode == 2) rho[1] = std::numeric_limits<double>::quiet_NaN();
  }
};
bool unchanged(const Eigen::MatrixXd &M) { return M.rows() == 1 && M.cols() == 1 && M(0, 0) == 123.0; }

void initial_failures(bool dogleg, int threads) {
  for (int mode = 1; mode <= 8; ++mode) {
    double x = 0.0;
    Problem problem;
    FaultFactor factor;
    FaultLoss loss;
    HuberLoss huber(0.1);
    factor.mode = mode <= 6 ? mode : 0;
    loss.mode = mode > 6 ? mode - 6 : 0;
    problem.AddParameterBlock(&x, 1);
    problem.AddResidualBlock(&factor, mode > 6 ? static_cast<LossFunction *>(&loss) : &huber, {&x});
    SolverOptions options;
    options.use_dogleg = dogleg;
    options.num_threads = threads;
    options.max_solver_time_seconds = 10.0;
    const auto result = problem.Solve(options);
    check(!result.converged && result.successful_steps == 0 && x == 0.0, "invalid initial factor cannot mutate mean or converge");
    check(result.message == "invalid initial factor evaluation", "invalid initial result has explicit cause");
    Eigen::MatrixXd cov = Eigen::MatrixXd::Constant(1, 1, 123.0);
    check(!problem.ComputeCovariance({&x}, cov, options) && unchanged(cov), "invalid covariance export is atomic");
    Eigen::MatrixXd information = Eigen::MatrixXd::Constant(1, 1, 123.0);
    Eigen::VectorXd gradient = Eigen::VectorXd::Constant(1, 456.0);
    check(!problem.ExportReducedInformation({&x}, information, gradient, options) && unchanged(information) &&
          gradient.size() == 1 && gradient(0) == 456.0, "invalid reduced information cannot certify a partial model");
    factor.mode = loss.mode = 0;
    const auto retry = problem.Solve(options);
    check(retry.converged && std::abs(x - 1.0) < 1e-6, "valid retry succeeds after invalid factor refusal");
    check(problem.ComputeCovariance({&x}, cov, options) && std::abs(cov(0, 0) - 1.0) < 1e-10, "retry covariance is unchanged by prior failure");
  }
}

void trial_failures(bool dogleg, int threads) {
  for (int mode : {1, 2, 3, 4}) {
    double x = 0.0;
    FaultFactor factor;
    factor.mode = mode;
    factor.after_step = true;
    Problem problem;
    problem.AddParameterBlock(&x, 1);
    problem.AddResidualBlock(&factor, nullptr, {&x});
    SolverOptions options;
    options.use_dogleg = dogleg;
    options.num_threads = threads;
    options.max_solver_time_seconds = 10.0;
    options.max_num_iterations = 3; // exercise relinearization after a valid cost trial
    options.function_tolerance = 0.0;
    options.parameter_tolerance = 0.0;
    const auto result = problem.Solve(options);
    check(!result.converged && !result.time_stopped, "invalid trials cannot masquerade as stationary convergence");
    check(x == 0.0 && result.successful_steps == 0, "invalid trial or accepted-step derivative restores prior iterate");
    check(landmark_qr::finite_scalar(result.final_cost) && result.final_cost == 0.5,
          "refused trial retains the last valid objective");
    factor.mode = 0;
    options.max_num_iterations = 30;
    options.function_tolerance = 1e-12;
    options.parameter_tolerance = 1e-12;
    const auto retry = problem.Solve(options);
    check(retry.converged && std::abs(x - 1.0) < 1e-6, "valid continuation succeeds after invalid trial rollback");
  }
}
} // namespace

int main() {
  for (bool dogleg : {false, true})
    for (int threads : {1, 3}) {
      initial_failures(dogleg, threads);
      trial_failures(dogleg, threads);
    }
  std::printf("invalid factor contract: %d/%d assertions passed\n", checks - failures, checks);
  return failures ? 1 : 0;
}
