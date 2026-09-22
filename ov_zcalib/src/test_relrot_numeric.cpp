/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Header-only production seed oracle. OV_RELROT_HEADER can select an archived
 * pre-fix header for actual old/new fast-math negative controls; no substitute
 * solver is used. --valid prints deterministic floating-point receipts.
 */
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>

#ifndef OV_RELROT_HEADER
#define OV_RELROT_HEADER "init/RelRotProcrustes.h"
#endif
#include OV_RELROT_HEADER
#include "utils/NumericChecks.h"

namespace {
using namespace ov_zcalib;
int checks = 0, failures = 0;
void check(bool good, const char *message) {
  ++checks;
  if (!good) { ++failures; std::printf("FAIL: %s\n",message); }
}
double from_bits(uint64_t bits) {
  double result; std::memcpy(&result,&bits,sizeof(result)); return result;
}
bool same(const Eigen::Matrix3d &a, const Eigen::Matrix3d &b) {
  return std::memcmp(a.data(),b.data(),9*sizeof(double)) == 0;
}
double angle(const Eigen::Matrix3d &a, const Eigen::Matrix3d &b) {
  return std::acos(std::min(1.,std::max(-1.,.5*((a*b.transpose()).trace()-1.))))*180./M_PI;
}
struct Scene {
  Eigen::Matrix3d R;
  std::vector<Eigen::Vector3d> first, second;
  std::vector<int> indices;
};
Scene scene(int model) {
  Scene data;
  data.R = (Eigen::AngleAxisd(.045,Eigen::Vector3d(.3,-.5,1.).normalized()) *
            Eigen::AngleAxisd(-.018,Eigen::Vector3d::UnitY())).toRotationMatrix();
  const Eigen::Vector3d translation = model == 0 ? Eigen::Vector3d::Zero().eval() : Eigen::Vector3d(.045,-.030,.025);
  std::mt19937 rng(31+model);
  std::uniform_real_distribution<double> uniform(-1.,1.);
  std::normal_distribution<double> normal(0.,1.);
  for (int i = 0; i < 120; ++i) {
    const double x = .6*uniform(rng), y = .45*uniform(rng);
    const double z = model == 1 ? .8+.05*x-.03*y : 1.5+1.15*uniform(rng);
    const Eigen::Vector3d point(x,y,z);
    Eigen::Vector3d first = point.normalized(), second = (data.R*point+translation).normalized();
    first += 4e-4*Eigen::Vector3d(normal(rng),normal(rng),normal(rng));
    second += 4e-4*Eigen::Vector3d(normal(rng),normal(rng),normal(rng));
    data.first.push_back(first.normalized()); data.second.push_back(second.normalized());
  }
  data.indices.resize(data.first.size()); std::iota(data.indices.begin(),data.indices.end(),0);
  return data;
}
void print_result(int model, const RelRotEssential::Result &result) {
  std::printf("VALID requested=%d selected=%d ok=%d inliers=%d ratio=%a residual=%a",model,result.model,result.ok,
              result.inliers,result.inlier_ratio,result.mean_resid_rad);
  for (int i = 0; i < 9; ++i) std::printf(" %a",result.R_C1toC2.data()[i]);
  std::printf("\n");
}
void valid_scenes() {
  const std::array<int,3> expected{{RelRotEssential::MODEL_PROCRUSTES,RelRotEssential::MODEL_HOMOGRAPHY,RelRotEssential::MODEL_ESSENTIAL}};
  for (int model = 0; model < 3; ++model) {
    const auto data = scene(model);
    const auto result = RelRotEssential::solve(data.first,data.second,620000001ull+model);
    const auto repeat = RelRotEssential::solve(data.first,data.second,620000001ull+model);
    print_result(model,result);
    check(result.ok && result.model == expected[model], "actual noisy bearing scene chooses rotation, homography or essential as appropriate");
    check(finite_matrix(result.R_C1toC2) && angle(result.R_C1toC2,data.R) < .35 &&
          std::abs(result.R_C1toC2.determinant()-1.) < 1e-8 &&
          (result.R_C1toC2.transpose()*result.R_C1toC2-Eigen::Matrix3d::Identity()).norm() < 1e-8,
          "selected noisy real-header rotation remains proper and agrees with independent motion truth");
    check(same(result.R_C1toC2,repeat.R_C1toC2) && result.inliers == repeat.inliers && result.model == repeat.model,
          "same timestamp seed preserves deterministic hypotheses and result");
    Eigen::Matrix3d fitted, decomposed;
    if (model == 1) {
      check(RelRotEssential::fit_H(data.first,data.second,data.indices,fitted) &&
            RelRotEssential::decompose_H(fitted,data.first,data.second,data.indices,decomposed) &&
            angle(decomposed,data.R) < .35, "actual DLT and homography decomposition accept a noisy translated plane");
    }
    if (model == 2) {
      check(RelRotEssential::fit_E(data.first,data.second,data.indices,fitted) &&
            RelRotEssential::decompose(fitted,data.first,data.second,data.indices,decomposed) &&
            angle(decomposed,data.R) < .35, "actual essential fit and cheirality decomposition accept noisy nonplanar parallax");
    }
  }
  auto data = scene(0);
  Eigen::Matrix3d R;
  check(RelRotEssential::decompose_H(2.*data.R,data.first,data.second,data.indices,R) && angle(R,data.R) < 1e-5,
        "pure-rotation homography polar branch remains valid");
  for (size_t i = 0; i < data.second.size(); i += 10)
    data.second[i] = Eigen::Vector3d(.4,-.8,1.).normalized();
  const auto robust = RelRotProcrustes::solve(data.first,data.second,620000004ull);
  check(robust.ok && robust.inliers >= 100 && angle(robust.R_C1toC2,data.R) < .1,
        "actual Procrustes RANSAC still handles ordinary finite outliers");
}

void bad_ray_control() {
  auto data = scene(0);
  // Choose a seed whose first three-point sample avoids index zero. The old
  // solver then silently accepts the remaining 119 rays as a usable seed.
  constexpr uint64_t seed = 620000001ull;
  std::mt19937 rng(static_cast<unsigned>(seed));
  auto idx = data.indices;
  bool avoids_bad_ray = true;
  for (int i = 0; i < 3; ++i) {
    std::uniform_int_distribution<int> distribution(i,static_cast<int>(idx.size())-1);
    std::swap(idx[i],idx[distribution(rng)]); avoids_bad_ray &= idx[i] != 0;
  }
  check(avoids_bad_ray,"old-header counterexample avoids feeding NaN into its first hypothesis SVD");
  data.second[0](0) = from_bits(UINT64_C(0x7ff8000000001234));
  const auto rotation = RelRotProcrustes::solve(data.first,data.second,seed);
  const auto selected = RelRotEssential::solve(data.first,data.second,seed);
  std::printf("BAD_RAY procrustes_ok=%d selected_ok=%d model=%d\n",rotation.ok,selected.ok,selected.model);
  check(!rotation.ok && !selected.ok,"one malformed bearing rejects the seed instead of hiding in the RANSAC outliers");
}

void bad_matrix_control() {
  const auto data = scene(1);
  Eigen::Matrix3d H = Eigen::Matrix3d::Identity(), before = 7.*Eigen::Matrix3d::Identity(), output = before;
  H(0,0) = from_bits(UINT64_C(0x7ff8000000001234));
  const bool accepted = RelRotEssential::decompose_H(H,data.first,data.second,data.indices,output);
  std::printf("BAD_MATRIX accepted=%d output_finite=%d unchanged=%d\n",accepted,finite_matrix(output),same(output,before));
  check(!accepted && same(output,before),"NaN homography rejects before SVD and leaves caller output untouched");
}

void invalid_inputs() {
  const auto data = scene(2);
  const Eigen::Matrix3d before = 7.*Eigen::Matrix3d::Identity();
  const std::array<double,4> invalid{{from_bits(UINT64_C(0x7ff8000000001234)),
      from_bits(UINT64_C(0x7ff0000000000000)),from_bits(UINT64_C(0xfff0000000000000)),1e308}};
  for (double value : invalid) {
    auto bad = data.second; bad[0](0) = value;
    Eigen::Matrix3d E = before, H = before, rotation = before;
    check(!RelRotEssential::fit_E(data.first,bad,data.indices,E) && same(E,before),"invalid essential fit ray rejects before SVD with atomic output");
    check(!RelRotEssential::fit_H(data.first,bad,data.indices,H) && same(H,before),"invalid homography fit ray rejects before SVD with atomic output");
    check(!RelRotEssential::decompose(data.R,data.first,bad,data.indices,rotation) && same(rotation,before),
          "essential cheirality refuses invalid bearings before decomposition");
    check(!RelRotEssential::decompose_H(data.R,data.first,bad,data.indices,rotation) && same(rotation,before),
          "homography cheirality refuses invalid bearings before decomposition");
    check(!finite_matrix(RelRotProcrustes::procrustes(data.first,bad)),"direct Procrustes signals invalid bearings without an SVD");
    check(!RelRotEssential::solve(data.first,bad,9).ok && !RelRotProcrustes::solve(data.first,bad,9).ok,
          "both public seed solvers return recoverable failure for nonfinite or overflow rays");
  }
  for (int kind = 0; kind < 4; ++kind) {
    auto bad = data.second; auto indices = data.indices;
    if (kind == 0) bad.pop_back();
    if (kind == 1) bad[0].setZero();
    if (kind == 2) indices[0] = -1;
    if (kind == 3) indices[0] = static_cast<int>(bad.size());
    Eigen::Matrix3d E = before, H = before, R = before;
    check(!RelRotEssential::fit_E(data.first,bad,indices,E) && same(E,before) &&
          !RelRotEssential::fit_H(data.first,bad,indices,H) && same(H,before),
          "fit helpers reject missing, zero or out-of-range paired bearings atomically");
    check(!RelRotEssential::decompose(data.R,data.first,bad,indices,R) && same(R,before) &&
          !RelRotEssential::decompose_H(data.R,data.first,bad,indices,R) && same(R,before),
          "decomposition helpers reject missing, zero or out-of-range paired bearings atomically");
  }
  for (double value : invalid) {
    Eigen::Matrix3d matrix = data.R, R = before; matrix(0,0) = value;
    check(!RelRotEssential::decompose_H(matrix,data.first,data.second,data.indices,R) && same(R,before),
          "nonfinite or overflowing homography decomposition is recoverable and atomic");
    if (!finite_scalar(value))
      check(!RelRotEssential::decompose(matrix,data.first,data.second,data.indices,R) && same(R,before),
            "nonfinite essential matrix rejects before SVD");
    check(RelRotEssential::epi_ang(matrix,data.first[0],data.second[0]) >= .004 &&
          RelRotEssential::transfer_ang(matrix,data.first[0],data.second[0]) >= .004,
          "nonfinite or overflow residual cannot be clamped into an inlier");
  }
  RelRotProcrustes::Options po; RelRotEssential::Options eo;
  po.inlier_threshold_rad = eo.inlier_threshold_rad = invalid[0];
  check(!RelRotProcrustes::solve(data.first,data.second,7,po).ok && !RelRotEssential::solve(data.first,data.second,7,eo).ok,
        "nonfinite angular options cannot produce an accepted model");
  po = RelRotProcrustes::Options(); eo = RelRotEssential::Options();
  po.min_inlier_ratio = eo.min_inlier_ratio = invalid[1];
  check(!RelRotProcrustes::solve(data.first,data.second,7,po).ok && !RelRotEssential::solve(data.first,data.second,7,eo).ok,
        "nonfinite inlier ratio rejects before a floating-to-integer conversion");
  const std::vector<Eigen::Vector3d> empty, one{Eigen::Vector3d::UnitZ()};
  po.min_pairs = eo.min_pairs = 0; po.min_inlier_ratio = eo.min_inlier_ratio = .5;
  check(!RelRotProcrustes::solve(empty,empty,7,po).ok && !RelRotProcrustes::solve(one,one,7,po).ok &&
        !RelRotEssential::solve(empty,empty,7,eo).ok && !RelRotEssential::solve(one,one,7,eo).ok,
        "insufficient RANSAC sample size remains recoverable even with a zero min-pairs option");
}
} // namespace

int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "all";
  if (mode == "all" || mode == "--valid") valid_scenes();
  if (mode == "all" || mode == "--bad-ray") bad_ray_control();
  if (mode == "all" || mode == "--bad-matrix") bad_matrix_control();
  if (mode == "all") invalid_inputs();
  std::printf("RELROT_NUMERIC %s checks=%d failures=%d\n",failures ? "FAIL" : "PASS",checks,failures);
  return failures ? 1 : 0;
}
