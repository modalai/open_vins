/* Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "utils/finite.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

using namespace ov_core;
using Poses = std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>>;

static double invalid_input(bool infinity) {
  volatile std::uint64_t input = infinity ? UINT64_C(0x7ff0000000000000) : UINT64_C(0x7ff8000000000000);
  const std::uint64_t bits = input;
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static bool run(const std::string &mode) {
  FeatureInitializerOptions options;
  FeatureInitializer initializer(options);
  auto feat = std::make_shared<Feature>();
  feat->featid = 3;
  feat->to_delete = false;
  Poses poses;
  const Eigen::Vector3d truth(.7, -.3, 4.0);
  for (size_t camera = 0; camera != 2; ++camera) {
    for (int frame = 0; frame != 4 - int(camera); ++frame) {
      const double time = 10.0 + frame * .1;
      Eigen::Vector3d position(.25 * frame, .20 * camera, .025 * frame);
      Eigen::Matrix3d rotation = Eigen::AngleAxisd(.04 * frame - .07 * camera, Eigen::Vector3d::UnitY()).toRotationMatrix();
      poses[camera].emplace(time, FeatureInitializer::ClonePose(rotation, position));
      const Eigen::Vector3d point = rotation * (truth - position);
      Eigen::Vector2f observation = (point.head<2>() / point.z()).cast<float>();
      feat->timestamps[camera].push_back(time);
      feat->uvs_norm[camera].push_back(observation);
      feat->uvs[camera].push_back(observation);
    }
  }
  const bool one_d = mode.find("1d") != std::string::npos;
  const bool gn = mode.find("gn_") == 0;
  if (mode.find("empty") == 0) {
    feat->timestamps.clear();
    feat->timestamps[0] = {};
  }
  if (mode.find("ray_nan") == 0) feat->uvs_norm[0][0](0) = static_cast<float>(invalid_input(false));
  if (mode.find("ray_inf") == 0) feat->uvs_norm[1][0](1) = static_cast<float>(invalid_input(true));
  if (mode.find("pose_nan") == 0) poses[1].at(10.1)._pos(1) = invalid_input(false);
  if (mode == "zero_baseline_1d") {
    for (auto &camera : poses)
      for (auto &pose : camera.second) { pose.second._pos.setZero(); pose.second._Rot.setIdentity(); }
    for (auto &camera : feat->uvs_norm)
      for (auto &uv : camera.second) uv = Eigen::Vector2f(0, 0);
  }
  bool accepted;
  if (gn) {
    feat->anchor_cam_id = 0;
    feat->anchor_clone_timestamp = feat->timestamps[0].back();
    auto &anchor = poses.at(0).at(feat->anchor_clone_timestamp);
    feat->p_FinA = anchor._Rot * (truth - anchor._pos);
    if (mode == "gn_refine") feat->p_FinA += Eigen::Vector3d(.03, -.02, .1);
    if (mode == "gn_nan_seed") feat->p_FinA.x() = invalid_input(false);
    if (mode == "gn_inf_seed") feat->p_FinA.z() = invalid_input(true);
    if (mode == "gn_zero_depth") feat->p_FinA.z() = 0;
    if (mode == "gn_negative_depth") feat->p_FinA *= -1;
    if (mode == "gn_nan_observation") feat->uvs_norm[1][0](0) = static_cast<float>(invalid_input(false));
    accepted = initializer.single_gaussnewton(feat, poses);
  } else {
    accepted = one_d ? initializer.single_triangulation_1d(feat, poses) : initializer.single_triangulation(feat, poses);
  }
  const bool valid = mode == "valid_3d" || mode == "valid_1d" || mode == "gn_exact" || mode == "gn_refine";
  if (!valid) return !accepted;
  const double error = (feat->p_FinG - truth).norm();
  std::printf("truth error %.12g m\n", error);
  return accepted && numeric::finite_matrix(feat->p_FinG) && error < 5e-5;
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  bool passed = false;
  try { passed = run(argv[1]); }
  catch (const std::exception &error) { std::fprintf(stderr, "unexpected exception: %s\n", error.what()); }
  std::printf("feature numeric %s: %s\n", argv[1], passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
