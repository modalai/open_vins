/* Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "init/InitializerPoseSelection.h"
#include <array>
#include <cstdio>
#include <set>
#include <vector>

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char *what) { ++checks; if (!ok) { ++failures; std::printf("FAIL: %s\n", what); } }
bool accept(double distance, double interval, double clock) {
#ifdef OV_TEST_OLD_SPACING
  return distance >= interval || distance == 0.;
#else
  return ov_init::initializer_pose_spacing(distance, interval, clock);
#endif
}
void select(double origin, double reference) {
  std::array<std::vector<double>,2> camera_times;
  for (int camera = 0; camera < 2; ++camera)
    for (int frame = 0; frame <= 72; ++frame) {
      const double offset = camera ? -.25 : reference;
      // Storage barriers model the actual raw-camera double and later retiming.
      volatile double raw = origin + frame/30. + (camera && frame%2 ? .013 : 0.) - offset;
      volatile double shift = offset - reference;
      camera_times[camera].push_back(raw + shift);
    }
  const double newest = std::max(camera_times[0].back(),camera_times[1].back());
  const double oldest = newest - 2.;
  std::set<double> chosen{newest};
  // Same chronological camera traversal as the recorded two-camera fixture.
  for (int camera : {1,0})
    for (double time : camera_times[camera]) {
      if (time <= oldest) continue; // preserve the legacy exclusive lower bound
      double distance = std::numeric_limits<double>::infinity();
      for (double existing : chosen) distance = std::min(distance,std::abs(time-existing));
      if (accept(distance,2./12.,newest)) chosen.insert(time);
    }
  std::printf("spacing origin=%.3f reference=%.3f selected=%zu\n",origin,reference,chosen.size());
  check(chosen.size() >= 11 && chosen.size() <= 13, "roundoff must not remove a required pose at the cadence boundary");
  check(*chosen.begin() > oldest && *chosen.rbegin() == newest, "selection preserves the original window boundaries");
}
} // namespace

int main() {
  const double target = 2./12.;
  const double represented_gap = 12.4 - 12.233333333333334;
  check(accept(represented_gap,target,12.4), "represented 1/6-second gap retains the eligible pose");
  check(!accept(target-1e-6,target,12.4), "genuinely shorter gap is rejected");
  check(accept(0.,target,12.4), "exact existing key can reuse its pose");
  check(!accept(1e-14,target,12.4), "nearby distinct keys are not merged");
  check(accept(target+.001,target,12.4), "larger gap remains eligible");
  for (double origin : {10.,1000.,1000000.,1000000000.})
    for (double reference : {0.,.125}) select(origin,reference);
  std::printf("INITIALIZER_SPACING %s checks=%d failures=%d\n",failures?"FAIL":"PASS",checks,failures);
  return failures ? 1 : 0;
}
