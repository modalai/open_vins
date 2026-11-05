// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * ov_zcalib_track: ROS-free session capture from a sensor stream.
 *
 *   <producer> | ov_zcalib_track OUT.bin [--features N] [--sigma-w X]
 *                                        [--sigma-wb X] [--sigma-a X]
 *                                        [--sigma-ab X]
 *
 * Reads the ov_core sensor stream (utils/sensor_stream.h) on stdin and runs the
 * same CaptureCore (default TrackKLT frontend, SessionRecordWriter) as the
 * ROS capture nodes, so a bag or voxl-logger log converted offline produces
 * the identical record a live capture would. Frame stamps arrive already
 * centered on the center-row mid-exposure instant, so they are used verbatim;
 * each frame's exposure is recorded as provenance. IMU temperature is passed
 * through; NaN (unavailable) becomes the record's legacy 0 placeholder.
 *
 * Exit status 0 only for a complete, footer-terminated record. On any error
 * the partial output is removed so it can never be mistaken for a capture.
 */

#include "ros/CaptureCore.h"
#include "utils/sensor_stream.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace ov_core::sensor_stream;

void usage() {
  std::fprintf(stderr,
               "usage: <producer> | ov_zcalib_track OUT.bin [--features N] "
               "[--sigma-w X] [--sigma-wb X] [--sigma-a X] [--sigma-ab X]\n"
               "Reads an ov_core sensor stream (OVSS v1) on stdin and writes a "
               "format-7 ov_zcalib session.\n");
}

double parse_double(const char *flag, const char *text) {
  char *end = nullptr;
  const double value = std::strtod(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(value))
    throw std::runtime_error(std::string(flag) + " expects a number");
  return value;
}

int run(int argc, char **argv) {
  std::string out;
  int features = 200;
  ov_zcalib::ImuNoise noise;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char *flag) -> const char * {
      if (i + 1 >= argc)
        throw std::runtime_error(std::string(flag) + " needs a value");
      return argv[++i];
    };
    if (arg == "-h" || arg == "--help") {
      usage();
      return 0;
    } else if (arg == "--features")
      features = int(parse_double("--features", value("--features")));
    else if (arg == "--sigma-w")
      noise.sigma_w = parse_double("--sigma-w", value("--sigma-w"));
    else if (arg == "--sigma-wb")
      noise.sigma_wb = parse_double("--sigma-wb", value("--sigma-wb"));
    else if (arg == "--sigma-a")
      noise.sigma_a = parse_double("--sigma-a", value("--sigma-a"));
    else if (arg == "--sigma-ab")
      noise.sigma_ab = parse_double("--sigma-ab", value("--sigma-ab"));
    else if (!arg.empty() && arg[0] == '-')
      throw std::runtime_error("unknown option " + arg);
    else if (out.empty())
      out = arg;
    else
      throw std::runtime_error("only one output path may be given");
  }
  if (out.empty()) {
    usage();
    return 2;
  }

  std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
  SensorStreamReader reader(stdin);
  reader.read_header();

  // Camera records precede all data: collect them, then build the core.
  std::vector<StreamCamera> cameras;
  StreamRecord record = reader.next();
  while (record == StreamRecord::Camera) {
    cameras.push_back(reader.camera());
    record = reader.next();
  }
  if (cameras.empty())
    throw std::runtime_error("the stream declares no cameras");
  std::vector<bool> seen(cameras.size(), false);
  for (const auto &c : cameras) {
    if (c.cam >= cameras.size() || seen[c.cam])
      throw std::runtime_error(
          "camera ids must be unique and contiguous from 0");
    seen[c.cam] = true;
  }
  std::vector<ov_zcalib::CaptureCore::CameraTiming> timing(cameras.size());
  for (const auto &c : cameras)
    timing[c.cam] = {c.readout_s, 0.0, false};

  std::unique_ptr<ov_zcalib::CaptureCore> core;
  bool created = false;
  try {
    core =
        std::make_unique<ov_zcalib::CaptureCore>(out, timing, features, noise);
    created = true;
    for (const auto &c : cameras) {
      if (c.fx <= 0 || c.fy <= 0)
        throw std::runtime_error(
            "camera " + std::to_string(c.cam) + " (" + c.name +
            ") has no calibration; intrinsics are required for tracking");
      core->configure(c.width, c.height, {c.fx, c.fy, c.cx, c.cy}, c.distortion,
                      c.model == StreamModel::Equidistant ? "equidistant"
                                                          : "plumb_bob",
                      c.cam);
      std::printf("[track] camera %u: %s %dx%d readout %.6f s\n", c.cam,
                  c.name.c_str(), c.width, c.height, c.readout_s);
    }
    if (!core->ready())
      throw std::runtime_error("cameras could not be configured");

    for (; record != StreamRecord::End; record = reader.next()) {
      if (record == StreamRecord::Camera)
        throw std::runtime_error("camera records must precede sensor data");
      if (record == StreamRecord::Imu) {
        const StreamImu &s = reader.imu();
        ov_zcalib::RawImu sample;
        sample.timestamp = s.t;
        sample.wm << s.w[0], s.w[1], s.w[2];
        sample.am << s.a[0], s.a[1], s.a[2];
        sample.temp_c = std::isfinite(s.temperature_c) ? s.temperature_c : 0.0;
        core->imu(sample);
      } else {
        StreamFrame &f = reader.frame();
        if (f.cam >= cameras.size())
          throw std::runtime_error("frame for undeclared camera " +
                                   std::to_string(f.cam));
        const cv::Mat gray(f.height, f.width, CV_8UC1, f.pixels.data());
        core->image(gray, f.t, f.cam, double(f.exposure_s));
      }
    }
    if (!core->close())
      throw std::runtime_error("capture is incomplete");
  } catch (...) {
    core.reset(); // the writer's destructor finalizes the file; remove it below
    if (created) {
      std::error_code ignored;
      std::filesystem::remove(out, ignored);
    }
    throw;
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  try {
    return run(argc, argv);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[track] error: %s\n", error.what());
    return 1;
  }
}
