// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef OV_ZCALIB_CAPTURE_CORE_H
#define OV_ZCALIB_CAPTURE_CORE_H

#include "utils/NumericChecks.h"
#include "utils/SessionRecord.h"
#include "utils/TrackerObs.h"
#include <cam/CamEqui.h>
#include <cam/CamRadtan.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <track/TrackKLT.h>
#include <unordered_map>
#include <vector>

namespace ov_zcalib {

// ROS adapters only translate messages. Tracking, pixels, timestamps and the
// persisted seed use the same production types as VoxlLogFeeder.
class CaptureCore {
public:
  struct CameraTiming {
    double readout = 0, exposure = 0;
    bool start_stamps = false;
  };
  CaptureCore(std::string path, double readout, double exposure,
              bool start_stamps, int features, ImuNoise noise)
      : CaptureCore(
            std::move(path),
            std::vector<CameraTiming>{{readout, exposure, start_stamps}},
            features, noise) {}

  CaptureCore(std::string path, std::vector<CameraTiming> timing, int features,
              ImuNoise noise)
      : path_(std::move(path)), timing_(std::move(timing)),
        features_(features) {
    if (path_.empty() || std::filesystem::exists(path_))
      throw std::runtime_error("record_path must name a new file (existing "
                               "files are never overwritten)");
    if (timing_.empty() || timing_.size() > 16 || features_ < 20)
      throw std::runtime_error(
          "capture requires 1-16 cameras and at least 20 features");
    for (const auto &t : timing_)
      if (!finite_scalar(t.readout) || t.readout < 0 ||
          !finite_scalar(t.exposure) || t.exposure < 0)
        throw std::runtime_error("invalid camera readout or exposure");
    for (double value :
         {noise.sigma_w, noise.sigma_wb, noise.sigma_a, noise.sigma_ab})
      if (!finite_scalar(value) || value <= 0)
        throw std::runtime_error(
            "IMU noise densities must be finite and positive");
    seed_.calib.noise = noise;
    seed_.profile = SessionProfile::LIBRARY;
    seed_.cam_mode = 1;
    seed_.calib.cams.resize(timing_.size());
    last_frame_.assign(timing_.size(), -1e300);
    camera_counts_.assign(timing_.size(), 0);
  }

  bool ready() const { return tracker_ != nullptr; }
  void configure(int width, int height, const std::vector<double> &intrinsics,
                 const std::vector<double> &distortion,
                 const std::string &model, size_t camera = 0) {
    if (camera >= timing_.size())
      throw std::runtime_error("unknown camera index");
    if (ready() || models_.count(camera))
      return;
    if (width < 16 || height < 16 || intrinsics.size() != 4 ||
        intrinsics[0] <= 0 || intrinsics[1] <= 0)
      throw std::runtime_error("CameraInfo requires a calibrated pinhole "
                               "camera and a valid resolution");
    if (model != "plumb_bob" && model != "equidistant" && !model.empty())
      throw std::runtime_error(
          "supported distortion models are plumb_bob (radtan) and equidistant");
    for (size_t i = 4; i < distortion.size(); ++i)
      if (!finite_scalar(distortion[i]) || distortion[i] != 0)
        throw std::runtime_error("nonzero distortion coefficients beyond the "
                                 "supported four cannot be discarded");
    auto &cam = seed_.calib.cams[camera];
    cam.img_w = width;
    cam.img_h = height;
    cam.tr = timing_[camera].readout;
    cam.rolling = cam.tr > 0;
    cam.fisheye = model == "equidistant";
    for (int i = 0; i < 4; ++i) {
      cam.cam(i) = intrinsics[i];
      cam.cam(i + 4) = i < (int)distortion.size() ? distortion[i] : 0;
    }
    if (!finite_matrix(cam.cam))
      throw std::runtime_error("non-finite camera calibration");
    std::shared_ptr<ov_core::CamBase> model_ptr;
    if (cam.fisheye)
      model_ptr = std::make_shared<ov_core::CamEqui>(width, height);
    else
      model_ptr = std::make_shared<ov_core::CamRadtan>(width, height);
    model_ptr->set_value(cam.cam);
    models_[camera] = model_ptr;
    std::printf("[capture] Camera %zu: %dx%d, %s stamps, readout %.6f s\n",
                camera, width, height,
                timing_[camera].start_stamps ? "start-of-exposure -> center"
                                             : "center (used verbatim)",
                cam.tr);
    if (models_.size() != timing_.size())
      return;
    // One tracker owns the global feature-ID counter. Cameras retain
    // independent timelines and tracks; stereo matching is not fabricated for
    // asynchronous RGB.
    tracker_ = std::make_unique<ov_core::TrackKLT>(
        models_, features_, 0, false,
        ov_core::TrackBase::HistogramMethod::HISTOGRAM, 15, 8, 6, 10);
    if (!writer_.open(path_, seed_))
      throw std::runtime_error(writer_.error());
    std::printf("[capture] READY TrackKLT; %zu cameras; %s\n", timing_.size(),
                path_.c_str());
  }

  void imu(const RawImu &sample) {
    if (!ready())
      return;
    if (!finite_scalar(sample.timestamp) || !finite_matrix(sample.wm) ||
        !finite_matrix(sample.am) || sample.timestamp <= last_imu_)
      throw std::runtime_error("IMU samples must be finite and strictly "
                               "increasing; check the sensor clock");
    if (!writer_.write_imu(sample))
      throw std::runtime_error(writer_.error());
    last_imu_ = sample.timestamp;
    ++imu_count;
  }

  FrameObs image(const cv::Mat &gray, double stamp, size_t camera = 0) {
    if (!ready())
      return {};
    if (camera >= timing_.size())
      throw std::runtime_error("unknown camera index");
    return image(gray, stamp, camera, timing_[camera].exposure);
  }

  /// Per-frame exposure (auto-exposure sources). It recenters start-of-exposure
  /// stamps and is recorded as the frame's exposure; center stamps are used
  /// verbatim.
  FrameObs image(const cv::Mat &gray, double stamp, size_t camera,
                 double exposure) {
    if (!ready())
      return {};
    if (camera >= timing_.size())
      throw std::runtime_error("unknown camera index");
    if (!finite_scalar(exposure) || exposure < 0)
      throw std::runtime_error("invalid frame exposure");
    const auto &cam = seed_.calib.cams[camera];
    const auto &timing = timing_[camera];
    const double center =
        stamp + (timing.start_stamps ? 0.5 * (timing.readout + exposure) : 0.0);
    if (gray.type() != CV_8UC1 || gray.cols != cam.img_w ||
        gray.rows != cam.img_h)
      throw std::runtime_error(
          "image dimensions/encoding do not match CameraInfo");
    if (!finite_scalar(center) || center <= last_frame_[camera])
      throw std::runtime_error(
          "camera stamps must be finite and strictly increasing");
    ov_core::CameraData data;
    data.timestamp = center;
    data.sensor_ids = {int(camera)};
    data.images = {gray};
    data.masks = {cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1)};
    tracker_->feed_new_camera(data);
    FrameObs frame = make_frame_obs(*tracker_, camera, center, float(exposure),
                                    0.f, uint32_t(camera_counts_[camera]));
    if (!writer_.write_frame(frame))
      throw std::runtime_error(writer_.error());
    last_frame_[camera] = center;
    ++camera_counts_[camera];
    ++frame_count;
    if (frame_count % 30 == 0) {
      if (!writer_.flush())
        throw std::runtime_error(writer_.error());
      std::printf("[capture] %llu frames / %llu IMU / %zu tracks\n",
                  (unsigned long long)frame_count,
                  (unsigned long long)imu_count, frame.pts.size());
    }
    return frame;
  }

  bool close() {
    if (!writer_.close()) {
      std::fprintf(stderr, "[capture] %s\n", writer_.error().c_str());
      return false;
    }
    if (!ready() || frame_count == 0 || imu_count == 0) {
      std::fprintf(stderr, "[capture] no complete camera/IMU capture; check "
                           "CameraInfo and topic names\n");
      return false;
    }
    for (size_t camera = 0; camera < camera_counts_.size(); ++camera)
      if (!camera_counts_[camera]) {
        std::fprintf(stderr, "[capture] camera %zu received no images\n",
                     camera);
        return false;
      }
    std::printf("[capture] COMPLETE %llu frames / %llu IMU\n",
                (unsigned long long)frame_count, (unsigned long long)imu_count);
    return true;
  }
  uint64_t imu_count = 0, frame_count = 0;

private:
  std::string path_;
  std::vector<CameraTiming> timing_;
  std::vector<double> last_frame_;
  std::vector<uint64_t> camera_counts_;
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> models_;
  int features_;
  double last_imu_ = -1e300;
  SessionSeed seed_;
  SessionRecordWriter writer_;
  std::unique_ptr<ov_core::TrackKLT> tracker_;
};
} // namespace ov_zcalib
#endif
