#include "utils/ReportJson.h"

#include <cstdio>
#include <limits>
#include <string>

int main() {
  ov_zcalib::SessionReport rep;
  ov_zcalib::ReportMeta meta;
  rep.t_solve_s = std::numeric_limits<double>::quiet_NaN();
  rep.t_verify_s = std::numeric_limits<double>::infinity();
  rep.t_total_s = -std::numeric_limits<double>::infinity();
  rep.abort_reason = "camera \"left\"\nmissing\\stream\t";
  const std::string json = ov_zcalib::report_to_json(rep, meta);
  for (const char *expected : {"\"t_solve_s\":null", "\"t_verify_s\":null", "\"t_total_s\":null",
                               "camera \\\"left\\\"\\nmissing\\\\stream\\t"}) {
    if (json.find(expected) == std::string::npos) {
      std::fprintf(stderr, "FAIL report JSON: missing %s\n", expected);
      return 1;
    }
  }
  rep.t_solve_s = 0.12345678901234567;
  const std::string finite = ov_zcalib::report_to_json(rep, meta);
  const std::string key = "\"t_solve_s\":";
  const size_t offset = finite.find(key);
  if (offset == std::string::npos || std::stod(finite.substr(offset + key.size())) != rep.t_solve_s) {
    std::fprintf(stderr, "FAIL report JSON: double did not round-trip\n");
    return 1;
  }
  std::puts("PASS report JSON");
  return 0;
}
