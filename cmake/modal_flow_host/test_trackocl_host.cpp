/*
 * TrackOCL on the host OpenCL harness, compared with TrackKLT on identical frames.
 *
 *   test_trackocl_host                    synthetic texture translating (+2.0, +1.0) px/frame
 *   test_trackocl_host DIR [--frames N]   real 8-bit frames (*.png, sorted by name)
 *
 * Checks that TrackOCL keeps a sane number of features with stable ids and, for
 * the synthetic sequence, that its matched displacement equals the true shift.
 * Exit codes: 0 pass, 1 fail, 77 skipped (no OpenCL platform/device).
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "cam/CamRadtan.h"
#include "track/TrackOCL/TrackOCL.h"
#ifndef DISABLE_TRACK_KLT
#include "track/TrackKLT.h"
#endif
#include "utils/print.h"
#include "utils/sensor_data.h"

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ov_core;

namespace {

struct Stats {
  std::string name;
  std::vector<double> counts, persistence, ms, err_px;
};

bool have_opencl_device() {
  cl_uint platforms = 0;
  if (clGetPlatformIDs(0, nullptr, &platforms) != CL_SUCCESS || platforms == 0)
    return false;
  std::vector<cl_platform_id> ids(platforms);
  clGetPlatformIDs(platforms, ids.data(), nullptr);
  for (cl_platform_id p : ids) {
    cl_uint devices = 0;
    if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &devices) == CL_SUCCESS && devices > 0)
      return true;
  }
  return false;
}

std::vector<cv::Mat> synthetic_frames(int count, cv::Point2d step) {
  cv::RNG rng(7);
  cv::Mat noise(720, 960, CV_32F);
  rng.fill(noise, cv::RNG::UNIFORM, 0, 1);
  cv::GaussianBlur(noise, noise, cv::Size(0, 0), 2.5);
  cv::Mat base;
  cv::normalize(noise, base, 0, 255, cv::NORM_MINMAX, CV_8U);
  for (int i = 0; i < 400; ++i) // corners for FAST
    cv::rectangle(base, cv::Rect(rng.uniform(0, 940), rng.uniform(0, 700), rng.uniform(4, 18), rng.uniform(4, 18)),
                  cv::Scalar(rng.uniform(0, 2) ? 255 : 0), cv::FILLED);
  std::vector<cv::Mat> frames;
  for (int i = 0; i < count; ++i) {
    cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, -160 + step.x * i, 0, 1, -120 + step.y * i);
    cv::Mat frame;
    cv::warpAffine(base, frame, shift, cv::Size(640, 480), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    frames.push_back(frame);
  }
  return frames;
}

std::vector<cv::Mat> directory_frames(const std::string &dir, int limit) {
  std::vector<std::string> files;
  for (const auto &entry : std::filesystem::directory_iterator(dir))
    if (entry.path().extension() == ".png")
      files.push_back(entry.path().string());
  std::sort(files.begin(), files.end());
  std::vector<cv::Mat> frames;
  for (const auto &file : files) {
    if ((int)frames.size() >= limit)
      break;
    cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
    if (!image.empty())
      frames.push_back(image);
  }
  return frames;
}

Stats run(const std::string &name, const std::shared_ptr<TrackBase> &tracker, const std::vector<cv::Mat> &frames,
          bool synthetic, cv::Point2d step) {
  Stats stats{name, {}, {}, {}, {}};
  std::unordered_map<size_t, cv::Point2f> previous;
  for (size_t i = 0; i < frames.size(); ++i) {
    CameraData message;
    message.timestamp = 1.0 + (double)i / 30.0;
    message.sensor_ids = {0};
    message.images = {frames[i]};
    message.masks = {cv::Mat::zeros(frames[i].size(), CV_8UC1)};
    const auto t0 = std::chrono::steady_clock::now();
    tracker->feed_new_camera(message);
    stats.ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    auto obs = tracker->get_last_obs()[0];
    auto ids = tracker->get_last_ids()[0];
    std::unordered_map<size_t, cv::Point2f> current;
    size_t kept = 0;
    std::vector<double> errors;
    for (size_t k = 0; k < ids.size() && k < obs.size(); ++k) {
      current[ids[k]] = obs[k].pt;
      auto it = previous.find(ids[k]);
      if (it != previous.end()) {
        ++kept;
        if (synthetic) // warpAffine maps content forward: it moves by +step every frame
          errors.push_back(cv::norm(cv::Point2d(obs[k].pt - it->second) - step));
      }
    }
    if (i >= 5) { // let the detector fill the grid first
      stats.counts.push_back((double)ids.size());
      stats.persistence.push_back(previous.empty() ? 0.0 : (double)kept / (double)previous.size());
      if (!errors.empty()) {
        std::nth_element(errors.begin(), errors.begin() + errors.size() / 2, errors.end());
        stats.err_px.push_back(errors[errors.size() / 2]);
      }
    }
    previous = std::move(current);
  }
  return stats;
}

double mean(const std::vector<double> &v) {
  return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double median(std::vector<double> v) {
  if (v.empty())
    return 0.0;
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

void print(const Stats &s) {
  std::printf("  %-9s features %6.1f  id persistence %5.1f%%  median step error %s  %7.2f ms/frame (median %.2f)\n",
              s.name.c_str(), mean(s.counts), 100.0 * mean(s.persistence),
              s.err_px.empty() ? "   n/a" : (std::to_string(median(s.err_px)).substr(0, 5) + " px").c_str(), mean(s.ms),
              median(s.ms));
}

} // namespace

int main(int argc, char **argv) {
  Printer::setPrintLevel("WARNING");
  std::string dir;
  int limit = 150;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--frames" && i + 1 < argc)
      limit = std::max(10, std::atoi(argv[++i]));
    else
      dir = arg;
  }
  if (!have_opencl_device()) {
    std::printf("SKIP: no OpenCL platform/device (install an ICD, e.g. pocl-opencl-icd, or set OCL_ICD_VENDORS)\n");
    return 77;
  }
  const cv::Point2d step(2.0, 1.0);
  const bool synthetic = dir.empty();
  std::vector<cv::Mat> frames;
  if (!synthetic) {
    if (!std::filesystem::is_directory(dir)) {
      std::printf("SKIP: %s is not a directory\n", dir.c_str());
      return 77;
    }
    frames = directory_frames(dir, limit);
  } else {
    frames = synthetic_frames(std::min(limit, 90), step);
  }
  if (frames.size() < 10) {
    std::printf("FAIL: need at least 10 frames (got %zu)\n", frames.size());
    return 1;
  }
  const int w = frames[0].cols, h = frames[0].rows;
  auto camera = std::make_shared<CamRadtan>(w, h);
  Eigen::Matrix<double, 8, 1> intrinsics;
  intrinsics << 0.7 * w, 0.7 * w, 0.5 * w, 0.5 * h, 0, 0, 0, 0;
  camera->set_value(intrinsics);
  std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras{{0, camera}};
  std::printf("%zu frames %dx%d (%s)\n", frames.size(), w, h, synthetic ? "synthetic, known shift" : dir.c_str());

  Stats ocl;
  try {
    auto tracker = std::make_shared<TrackOCL>(cameras, 200, 0, false, 20, 5, 5, 10);
    ocl = run("TrackOCL", tracker, frames, synthetic, step);
  } catch (const std::exception &error) {
    std::printf("FAIL: TrackOCL: %s\n", error.what());
    return 1;
  }
  print(ocl);
#ifndef DISABLE_TRACK_KLT
  auto klt = std::make_shared<TrackKLT>(cameras, 200, 0, false, TrackBase::NONE, 20, 5, 5, 10);
  print(run("TrackKLT", klt, frames, synthetic, step));
#endif

  bool ok = mean(ocl.counts) >= 50.0 && mean(ocl.persistence) >= 0.6;
  if (synthetic)
    ok = ok && !ocl.err_px.empty() && median(ocl.err_px) < 0.5;
  std::printf("%s\n", ok ? "PASS" : "FAIL: TrackOCL feature count / id persistence / motion below thresholds");
  return ok ? 0 : 1;
}
