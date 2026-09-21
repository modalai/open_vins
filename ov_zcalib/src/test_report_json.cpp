#include "utils/ReportJson.h"

#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include <boost/property_tree/json_parser.hpp>

int main() {
  ov_zcalib::SessionReport rep;
  ov_zcalib::ReportMeta meta;
  rep.t_solve_s = std::numeric_limits<double>::quiet_NaN();
  rep.t_verify_s = std::numeric_limits<double>::infinity();
  rep.t_total_s = -std::numeric_limits<double>::infinity();
  rep.abort_reason = "camera \"left\"\nmissing\\stream\t";
  const std::string json = ov_zcalib::report_to_json(rep, meta);
  if (json.find("\"solve_budget_s\"") != std::string::npos) {
    std::fprintf(stderr, "FAIL report JSON: unknown solver budget must not be invented\n");
    return 1;
  }
  for (const char *expected : {"\"t_solve_s\":null", "\"t_verify_s\":null", "\"t_total_s\":null",
                               "\"reprojection_sigma_px\":null", "\"reprojection_sigma_uses_window_default\":true",
                               "camera \\\"left\\\"\\nmissing\\\\stream\\t"}) {
    if (json.find(expected) == std::string::npos) {
      std::fprintf(stderr, "FAIL report JSON: missing %s\n", expected);
      return 1;
    }
  }
  rep.t_solve_s = 0.12345678901234567;
  rep.committed.grav_mag = 9.79321;
  meta.gravity_source = "calibration_profile";
  meta.solve_budget_s = 120.0;
  const std::string finite = ov_zcalib::report_to_json(rep, meta);
  for (const char *expected : {"\"sigma_semantics\":\"local_curvature_precision\"",
                               "\"coverage_calibrated\":false", "\"coverage_kappa\":null", "\"solve_budget_s\":120",
                               "\"verdict_semantics\":\"configured_policy_certification\"",
                               "\"r_semantics\":\"directions_above_configured_information_floor\""}) {
    if (finite.find(expected) == std::string::npos) {
      std::fprintf(stderr, "FAIL report JSON: posterior coverage metadata missing: %s\n", expected);
      return 1;
    }
  }
  const std::string key = "\"t_solve_s\":";
  const size_t offset = finite.find(key);
  if (offset == std::string::npos || std::stod(finite.substr(offset + key.size())) != rep.t_solve_s) {
    std::fprintf(stderr, "FAIL report JSON: double did not round-trip\n");
    return 1;
  }
  const std::string gravity_key = "\"gravity_mag\":";
  const size_t gravity_offset = finite.find(gravity_key);
  if (gravity_offset == std::string::npos ||
      std::stod(finite.substr(gravity_offset + gravity_key.size())) != rep.committed.grav_mag ||
      finite.find("\"gravity_source\":\"calibration_profile\"") == std::string::npos) {
    std::fprintf(stderr, "FAIL report JSON: calibration gravity/provenance missing\n");
    return 1;
  }
  ov_zcalib::SessionConfig cfg;
  cfg.harvester.pix_sigma = 0.5;
  ov_zcalib::SessionSeed seed;
  seed.calib.cams.resize(2);
  seed.calib.cams[1].reprojection_sigma_px = 1.25;
  ov_zcalib::CalibSessionRunner runner(cfg, seed);
  const auto &resolved = runner.report().camera_pixel_sigmas;
  if (resolved.size() != 2 || resolved[0] != 0.5 || resolved[1] != 1.25) {
    std::fprintf(stderr, "FAIL report JSON: runner lost effective scalar/camera measurement noise\n");
    return 1;
  }
  const std::string cameras = ov_zcalib::report_to_json(runner.report(), meta);
  for (const char *expected : {"\"reprojection_sigma_px\":0.5,\"reprojection_sigma_uses_window_default\":true",
                               "\"reprojection_sigma_px\":1.25,\"reprojection_sigma_uses_window_default\":false"}) {
    if (cameras.find(expected) == std::string::npos) {
      std::fprintf(stderr, "FAIL report JSON: effective camera measurement noise missing: %s\n", expected);
      return 1;
    }
  }
  cfg.harvester.pix_sigma = 1.0;
  ov_zcalib::CalibSessionRunner legacy_runner(cfg, seed);
  if (legacy_runner.report().camera_pixel_sigmas[0] != 1.0) {
    std::fprintf(stderr, "FAIL report JSON: legacy live/replay default must remain 1px\n");
    return 1;
  }
  // Actual file I/O: a full output filesystem must not replace an earlier
  // complete report or claim success after buffered output fails at close.
  char directory[] = "/tmp/zcalib-report-json-XXXXXX";
  if (!::mkdtemp(directory)) {
    std::perror("mkdtemp");
    return 1;
  }
  int failures = 0;
  auto check = [&](bool ok, const char *why) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL report JSON: %s\n", why); }
  };
  const std::string path = std::string(directory) + "/report.json";
  const std::string tmp = path + ".tmp";
  auto read_report = [&]() {
    std::ifstream in(path);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  };
  std::string error = "stale error";
  check(ov_zcalib::write_report_json(path, rep, meta, &error) && error.empty(), "normal atomic report write failed");
  const std::string saved = read_report();
  check(saved == finite, "published report differs from complete serialized result");
  try {
    std::istringstream input(saved);
    boost::property_tree::ptree parsed;
    boost::property_tree::read_json(input, parsed);
  } catch (const boost::property_tree::json_parser_error &) {
    check(false, "published report is not valid JSON");
  }
  for (size_t extra_bytes : {size_t(0), size_t(262144)}) {
    check(::symlink("/dev/full", tmp.c_str()) == 0, "could not arrange ENOSPC fault");
    auto failed_rep = rep;
    failed_rep.abort_reason.assign(extra_bytes, 'x');
    const bool ok = ov_zcalib::write_report_json(path, failed_rep, meta, &error);
    const int saved_errno = errno;
    check(!ok && saved_errno == ENOSPC, "full filesystem must fail and preserve ENOSPC");
    check(error.find(std::strerror(ENOSPC)) != std::string::npos && error.find(path) != std::string::npos,
          "I/O failure must identify the filesystem error and report path");
    check(read_report() == saved, "failed output replaced the prior complete report");
    check(::access(tmp.c_str(), F_OK) != 0, "failed temporary report was not cleaned up");
  }
  check(::mkdir(tmp.c_str(), 0700) == 0, "could not arrange open failure");
  check(!ov_zcalib::write_report_json(path, rep, meta, &error) && error.find("cannot open") != std::string::npos,
        "open failure must reach the caller");
  check(read_report() == saved, "open failure replaced the prior report");
  ::rmdir(tmp.c_str());
  ::unlink(path.c_str());
  ::rmdir(directory);
  if (failures)
    return 1;
  std::puts("PASS report JSON");
  return 0;
}
