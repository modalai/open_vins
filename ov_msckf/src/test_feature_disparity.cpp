/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "feat/FeatureHelper.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
namespace {
int checks=0, failures=0;
void check(bool ok, const char *why) {
  ++checks; if (!ok) { ++failures; std::printf("FAIL: %s\n",why); }
}
bool finite_value(double x) {
  std::uint64_t u; std::memcpy(&u,&x,sizeof(u));
  return (u & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
void append(const std::shared_ptr<ov_core::FeatureDatabase> &db, size_t id, size_t cam, float dx) {
  db->update_feature(id,1.,cam,100,200,0,0);
  db->update_feature(id,2.,cam,100+dx,200,0,0);
}
}
int main() {
  for (int samples : {0,1,2}) {
    auto db=std::make_shared<ov_core::FeatureDatabase>();
    if (samples>0) append(db,1,0,3);
    if (samples>1) append(db,2,0,4);
    for (bool pair : {false,true}) {
      double mean=123., spread=456.; int count=789;
      if (pair) ov_core::FeatureHelper::compute_disparity(db,1.,2.,mean,spread,count);
      else ov_core::FeatureHelper::compute_disparity(db,mean,spread,count);
      check(finite_value(mean)&&finite_value(spread),"all-camera statistics are finite for zero/one/two samples");
      if (samples<2) check(mean==-1.&&spread==-1.&&count==0,"insufficient unfiltered data retains the declared unavailable sentinel");
      else check(mean==3.5&&std::abs(spread-std::sqrt(.5))<1e-12&&count==2,"two-sample mean and sample standard deviation are unchanged");
    }
  }
  auto db=std::make_shared<ov_core::FeatureDatabase>();
  append(db,1,0,5); append(db,1,1,12);
  for (int cam : {0,1,2}) {
    double mean=123., spread=456.; int count=789;
    ov_core::FeatureHelper::compute_disparity(db,1.,2.,mean,spread,count,cam);
    check(finite_value(mean)&&finite_value(spread),"camera-filtered statistics remain finite");
    check(cam==2 ? count==0&&mean==-1.&&spread==-1. : count==1&&mean==(cam?12.:5.)&&spread==0.,
          "per-camera singleton is retained for pooled statistics without mixing camera keys");
  }
  double mean=123., spread=456.; int count=789;
  ov_core::FeatureHelper::compute_disparity(db,mean,spread,count,1.5,1.);
  check(finite_value(mean)&&finite_value(spread)&&mean==-1.&&spread==-1.&&count==0,"time selection with no complete displacement returns unavailable statistics");
  std::printf("disparity: %d/%d checks passed\n",checks-failures,checks);
  return failures?1:0;
}
