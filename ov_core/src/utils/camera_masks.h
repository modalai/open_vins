#pragma once

#include "utils/sensor_data.h"
#include <opencv2/imgproc.hpp>
#include <map>
#include <stdexcept>

namespace ov_core {

// A Mat over a transient/raw pointer has no owner to retain. Snapshot only that
// case; ordinary immutable Mat/ROI masks keep their reference-counted storage.
inline cv::Mat retain_camera_mask(const cv::Mat &mask) {
  return !mask.empty() && mask.u == nullptr ? mask.clone() : mask;
}

// Masks are immutable exclusion images aligned with sensor_ids/images. cv::Mat
// retains their owners across asynchronous split/bundle operations. Producers
// replace masks on updates instead of modifying storage queued frames still use.
inline bool camera_masks_valid(const CameraData &message) {
  if (!message.observations.empty()) return true;
  const size_t n = message.sensor_ids.size();
  if (n == 0 || message.images.size() != n || (!message.masks.empty() && message.masks.size() != n)) return false;
  for (size_t i = 0; i < n; ++i) {
    if (message.images[i].empty()) return false;
    if (!message.masks.empty() && !message.masks[i].empty() &&
        (message.masks[i].type() != CV_8UC1 || message.masks[i].size() != message.images[i].size())) return false;
  }
  return true;
}

inline void prepare_camera_masks(CameraData &message, const std::map<size_t, cv::Mat> &configured = {}) {
  if (!camera_masks_valid(message)) throw std::invalid_argument("camera masks must be CV_8UC1 and match their images");
  if (!message.observations.empty()) return;
  if (message.masks.empty()) message.masks.resize(message.sensor_ids.size());
  for (size_t i = 0; i < message.sensor_ids.size(); ++i) {
    auto &mask = message.masks[i];
    const auto it = configured.find(message.sensor_ids[i]);
    if (it != configured.end()) {
      const auto &fixed = it->second;
      if (fixed.empty() || fixed.type() != CV_8UC1 || fixed.size() != message.images[i].size())
        throw std::invalid_argument("configured camera mask does not match tracking image");
      if (mask.empty()) mask = fixed;
      else if (mask.data != fixed.data || mask.step != fixed.step) {
        // New storage: never mutate an input mask retained by queued/history frames.
        cv::Mat combined;
        cv::bitwise_or(mask, fixed, combined);
        mask = std::move(combined);
      }
    }
    if (mask.empty()) mask = cv::Mat::zeros(message.images[i].size(), CV_8UC1);
  }
}

// A mask is categorical, so Gaussian pyrDown must not dilute exclusion values.
inline void downsample_camera(CameraData &message, size_t i) {
#if HAVE_OPENCL
  if (!message.img_frames.empty() && message.img_frames.at(i).img.handle_type != modal_flow::ExternalType::None)
    throw std::invalid_argument("external-only camera images do not support CPU downsampling");
#endif
  const auto size = cv::Size(message.images.at(i).cols / 2, message.images.at(i).rows / 2);
  cv::Mat image;
  cv::pyrDown(message.images.at(i), image, size);
  message.images[i] = std::move(image);
  if (!message.masks.empty() && !message.masks[i].empty()) {
    cv::Mat mask;
    cv::resize(message.masks[i], mask, size, 0, 0, cv::INTER_NEAREST);
    message.masks[i] = std::move(mask);
  }
#if HAVE_OPENCL
  // Rebind the GPU upload to the owned reduced pixels, rather than the original
  // full-resolution ImageView. External-only images must be rejected by the caller.
  if (!message.img_frames.empty()) {
    auto &view = message.img_frames.at(i).img;
    view.desc = {size.width, size.height, modal_flow::PixelFormat::R8, static_cast<int>(message.images[i].step)};
    view.data = message.images[i].data;
    // Retain camera/time identity and the original view's release callback.
  }
#endif
}

inline void prepare_camera_for_tracking(CameraData &message, bool downsample,
                                        const std::map<size_t, cv::Mat> &configured = {}) {
  if (!camera_masks_valid(message)) throw std::invalid_argument("invalid camera image/mask geometry");
  if (!message.observations.empty()) return;
#if HAVE_OPENCL
  if (downsample)
    for (const auto &frame : message.img_frames)
      if (frame.img.handle_type != modal_flow::ExternalType::None)
        throw std::invalid_argument("external-only camera images do not support CPU downsampling");
#endif
  if (downsample)
    for (size_t i = 0; i < message.sensor_ids.size(); ++i) downsample_camera(message, i);
  prepare_camera_masks(message, configured);
}

} // namespace ov_core
