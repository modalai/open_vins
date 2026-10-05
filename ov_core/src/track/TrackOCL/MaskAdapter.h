#pragma once

#include <modal_flow/Mask.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <stdexcept>

namespace ov_core {

inline modal_flow::MaskView flow_mask_view(const cv::Mat &mask) {
  if (mask.empty()) return {};
  if (mask.type() != CV_8UC1) throw std::invalid_argument("TrackOCL mask must be CV_8UC1");
  return {mask.data, mask.cols, mask.rows, mask.step};
}

inline void remove_masked_keypoints(const cv::Mat &mask, std::vector<cv::KeyPoint> &points,
                                    std::vector<size_t> &ids) {
  const auto view = flow_mask_view(mask);
  if (points.size() != ids.size()) throw std::invalid_argument("feature and id counts differ");
  size_t kept = 0;
  for (size_t i = 0; i < points.size(); ++i)
    if (view.allows(points[i].pt.x, points[i].pt.y)) {
      points[kept] = points[i]; ids[kept++] = ids[i];
    }
  points.resize(kept); ids.resize(kept);
}

// A sampled masked pixel cannot disable an entire partially usable grid cell.
// Clear cells stop at their first pixel; fully excluded cells are skipped by FAST.
inline cv::Mat fully_masked_grid(const cv::Mat &mask, cv::Size grid) {
  if (grid.width <= 0 || grid.height <= 0 || mask.empty() || mask.type() != CV_8UC1)
    throw std::invalid_argument("invalid detection mask/grid");
  const int cell_w = mask.cols / grid.width, cell_h = mask.rows / grid.height;
  if (cell_w <= 0 || cell_h <= 0) throw std::invalid_argument("mask smaller than detection grid");
  cv::Mat result(grid, CV_8UC1, cv::Scalar(0));
  for (int cy = 0; cy < grid.height; ++cy) for (int cx = 0; cx < grid.width; ++cx) {
    bool blocked = true;
    for (int y = cy * cell_h; y < (cy + 1) * cell_h && blocked; ++y) {
      const auto *row = mask.ptr<uint8_t>(y);
      for (int x = cx * cell_w; x < (cx + 1) * cell_w; ++x)
        if (row[x] <= 127) { blocked = false; break; }
    }
    if (blocked) result.at<uint8_t>(cy, cx) = 255;
  }
  return result;
}

} // namespace ov_core
