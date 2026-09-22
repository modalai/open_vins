/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "solve/JointCalib.h"
#include "utils/NumericChecks.h"
#include <cstdint>
#include <cstdio>
#include <cstring>

// Fault injection is confined to the window-solver boundary. JointCalib is the
// real production implementation, including its parallel reduction, acceptance,
// stationarity certificate and returned carry. No real sensor fit is claimed.
namespace {
enum class Mode { finite, reference, invalid_cost, invalid_gradient, invalid_finalize };
Mode mode = Mode::finite;
double injected = 0.;
int evaluations = 0, checks = 0, failures = 0;
void check(bool pass, const char *why) {
  ++checks;
  if (!pass) { ++failures; std::printf("FAIL: %s\n", why); }
}
double from_bits(std::uint64_t bits) {
  double x; std::memcpy(&x, &bits, sizeof(x)); return x;
}
ov_zcalib::SharedCalib calibration() {
  ov_zcalib::SharedCalib c;
  c.imu.calib_dw = c.imu.calib_da = c.imu.calib_RAtoI = c.imu.calib_tg = false;
  c.cams[0].free_ext = false;
  c.cams[0].free_td = true;
  return c;
}
ov_zcalib::JointConfig config() {
  ov_zcalib::JointConfig c;
  c.verbose = false;
  c.num_threads = 1;
  c.outer_iterations = 2;
  c.cert_min_windows = 1;
  c.export_on_accept = false;
  c.use_carry = true;
  c.prior_sigma["td"] = 1.;
  return c;
}
}

namespace ov_zcalib {
bool LinearSeed::seed_window(WindowData &w, const SharedCalib &, const Eigen::Vector3d &, LinearSeedReport &,
                             const LinearSeedConfig &) {
  w.has_seeds = true;
  return true;
}
bool WindowBA::solve_and_export(const WindowData &, SharedCalib &c, bool, WindowSolveReport &r, int, bool,
                               WindowWarmState *warm, WindowPreint *, const WindowWarmState *,
                               const WindowEvaluationContext *, const WindowBiasPrior *, const WindowBoundaryBias *) {
  const bool already_warm = warm && warm->valid;
  ++evaluations;
  r.ok = true;
  r.free_dim = c.local_dim();
  r.Lambda = Eigen::MatrixXd::Identity(r.free_dim, r.free_dim);
  r.gred = Eigen::VectorXd::Constant(r.free_dim, c.cams[0].td - .01);
  r.cost_final = 100. + .5 * (c.cams[0].td - .01) * (c.cams[0].td - .01);
  r.inner_converged = true;
  r.qn = mode == Mode::reference && already_warm ? .01 : .001;
  r.iterations = 3;
  if (mode == Mode::invalid_cost || (mode == Mode::invalid_finalize && already_warm)) r.cost_final = injected;
  if (mode == Mode::invalid_gradient) r.gred(0) = injected;
  if (warm) warm->valid = true;
  return true;
}
}

int main() {
  using namespace ov_zcalib;
  std::vector<WindowData> windows(1);
  for (bool export_on_accept : {false, true}) {
    auto c = calibration(); auto cfg = config(); cfg.outer_iterations = 1; cfg.export_on_accept = export_on_accept;
    JointReport report; JointWarmCarry carry;
    mode = Mode::finite; evaluations = 0;
    check(JointCalib::solve(windows,c,cfg,report,&carry), "finite objective still solves");
    check(report.ok && report.accepted_passes == 1 && finite_scalar(report.final_merit), "finite accepted report is complete");
    check(carry.valid && carry.qn_ref.size() == 1 && finite_scalar(carry.qn_ref[0]) && carry.qn_ref[0] == .001,
          "first accepted nuisance decrement replaces the unarmed infinity reference");
  }
  {
    auto c = calibration(); auto cfg = config(); JointReport report; JointWarmCarry carry;
    mode = Mode::reference; evaluations = 0;
    check(JointCalib::solve(windows,c,cfg,report,&carry), "certificate-controlled objective solves");
    check(report.cold_cert == 1, "growth above the first accepted nuisance reference triggers a confirming cold solve");
  }
  for (std::uint64_t bits : {UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),UINT64_C(0x7ff8000000000000)}) {
    injected = from_bits(bits);
    for (Mode m : {Mode::invalid_cost, Mode::invalid_gradient, Mode::invalid_finalize}) {
      for (bool export_on_accept : {false, true}) {
        mode = m; evaluations = 0;
        auto c = calibration(); auto cfg = config(); cfg.outer_iterations = 1; cfg.export_on_accept = export_on_accept;
        cfg.fused_schur = m == Mode::invalid_finalize;
        JointReport report; JointWarmCarry carry; std::vector<WindowWarmState> output(1); output[0].grav.setConstant(7.);
        check(!JointCalib::solve(windows,c,cfg,report,&carry,nullptr,&output), "nonfinite objective/gradient/finalization is rejected");
        check(!report.ok && !carry.valid && output.size() == 1 && (output[0].grav.array() == 7.).all(),
              "failed numerical stage publishes neither warm state nor valid carry");
        check(c.cams[0].td == 0., "failure retains the last accepted calibration mean");
      }
    }
  }
  std::printf("joint numeric: %d/%d checks passed\n",checks-failures,checks);
  return failures ? 1 : 0;
}
