#include <cstdio>
#include <limits>
#include "solve/CameraRefinement.h"
#include "solve/JointCalib.h"
#include "core/CalibSession.h"

using namespace ov_zcalib;

int main() {
  int failures = 0;
  auto check = [&](bool ok, const char *why) {
    if (!ok) {
      ++failures;
      std::printf("FAIL: %s\n", why);
    }
  };
  const CameraPrior base = JointConfig::default_cam_prior();
  // A centered scene need not reach the corners to constrain tangential
  // distortion. Equidistant's high-order radial tail still needs that reach.
  const auto rt = camera_refinement_prior(false, base, .03, .2, .12, .1, .001);
  const auto eq = camera_refinement_prior(true, base, .03, .2, .12, .1, .001);
  check(rt(6) == .001 && rt(7) == .001, "radtan tangential pair incorrectly corner-gated");
  check(eq(6) == 1e-9 && eq(7) == 1e-9, "fisheye radial tail opened without coverage");
  const auto edge = camera_refinement_prior(false, base, .8, .02, .12, .1, .001);
  check(edge(2) == 1e-9 && edge(3) == 1e-9 && edge(6) == 1e-9 && edge(7) == 1e-9,
        "one-sided radtan coverage must freeze center and tangential pairs");
  const auto wide = camera_refinement_prior(true, base, .8, .2, .12, .1, .001);
  check(wide(6) == base(4) && wide(7) == base(5), "fisheye coverage no longer opens radial tail");

  CameraPrior prior = rt;
  CameraPrior factory;
  factory << 450, 452, 320, 240, -.02, .005, .0002, -.0001;
  CameraPrior value = factory + CameraPrior::Constant(.01);
  CameraPrior sigma;
  sigma << .85, .6, .2, .2, .001, .001, .0001, .0001;
  check(freeze_unresolved_camera_pairs(sigma, 3.0, prior, value, factory), "weak focal pair not detected");
  check(value.head<2>() == factory.head<2>() && prior.head<2>() == CameraPrior::Constant(1e-9).head<2>(),
        "factory focal pair must be restored together before refit");
  check(value.tail<6>() != factory.tail<6>(), "observable center/distortion discarded with focal pair");
  check(!freeze_unresolved_camera_pairs(sigma, 3.0, prior, value, factory),
        "frozen pair counted against commit precision");
  sigma(6) = std::numeric_limits<double>::infinity();
  check(freeze_unresolved_camera_pairs(sigma, 3.0, prior, value, factory) && value.tail<2>() == factory.tail<2>(),
        "invalid precision must restore both tangential values");

  // Regression from the two-fisheye 005959 recording: both lens blocks miss
  // commit precision, but several intrinsic pairs ARE constrained. Restore
  // unresolved factory pairs before the complete joint refit, preserving the
  // radial updates rather than discarding every lens parameter or loosening
  // the camera-rotation ceiling. Each camera must keep its own mask and seed.
  CameraPrior fish_prior[2] = {eq, eq};
  CameraPrior fish_factory[2], fish_value[2], fish_sigma[2];
  fish_factory[0] << 457.376815, 457.147014, 643.895337, 409.040725, .067982103, -.000056072, .009176459, -.005120486;
  fish_factory[1] << 473.312492, 473.301741, 648.542520, 430.477762, .051487572, .017550368, -.012124795, .003899055;
  fish_value[0] << 460.53, 462.54, 643.04, 407.88, .06125, .00401, .009176459, -.005120486;
  fish_value[1] << 471.20, 470.38, 648.95, 427.81, .06179, .01841, -.012124795, .003899055;
  fish_sigma[0] << .6098, .6261, .8168, .7954, .001795, .001037, 1e-9, 1e-9;
  fish_sigma[1] << .7326, .6798, 1.240, .7671, .002149, .001275, 1e-9, 1e-9;
  const CameraPrior fish_candidate[2] = {fish_value[0], fish_value[1]};
  for (int c = 0; c < 2; ++c) {
    check(freeze_unresolved_camera_pairs(fish_sigma[c], 3.0, fish_prior[c], fish_value[c], fish_factory[c]),
          "unresolved fisheye center must trigger a conditional joint refit");
    check(fish_prior[c].segment<2>(2).maxCoeff() <= 1e-8 &&
              fish_value[c].segment<2>(2) == fish_factory[c].segment<2>(2),
          "fisheye center pair must return to that camera's factory seed");
    check(fish_prior[c].segment<2>(4) == eq.segment<2>(4) &&
              fish_value[c].segment<2>(4) == fish_candidate[c].segment<2>(4),
          "constrained fisheye radial pair must remain free for joint refit");
    check(fish_prior[c].tail<2>() == eq.tail<2>() && fish_value[c].tail<2>() == fish_factory[c].tail<2>(),
          "coverage-frozen radial tail must remain fixed and must not veto refinement");
    check(!freeze_unresolved_camera_pairs(fish_sigma[c], 3.0, fish_prior[c], fish_value[c], fish_factory[c]),
          "already-conditioned fisheye pairs must not request another refit");
  }
  check(fish_prior[0].head<2>() == eq.head<2>() && fish_value[0].head<2>() == fish_candidate[0].head<2>(),
        "front camera's constrained focal pair must survive the down camera's focal failure");
  check(fish_prior[1].head<2>().maxCoeff() <= 1e-8 && fish_value[1].head<2>() == fish_factory[1].head<2>(),
        "down camera's unresolved focal pair must use its own factory seed");

  SharedCalib c;
  const auto before = c.cams[0].cam;
  JointConfig cfg;
  cfg.max_wall_s = -1;
  JointReport report;
  check(!JointCalib::solve({}, c, cfg, report) && report.hit_wall_budget && c.cams[0].cam == before,
        "exhausted budget must skip without changing calibration");
  CalibSession display(1, Eigen::VectorXd::Ones(1), {"tg[6]"});
  Eigen::VectorXd improve;
  std::string prompt;
  display.progress(improve, prompt);
  check(prompt.find("IMU Z") != std::string::npos, "Tg guidance must use its column-major force axis");
  std::printf("%s camera refinement policy and exhausted-deadline tests\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
