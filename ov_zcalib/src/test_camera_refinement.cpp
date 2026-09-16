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
