/* Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_INITIALIZER_CAMERA_CLOCK_H
#define OV_INIT_INITIALIZER_CAMERA_CLOCK_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>

#include "feat/Feature.h"
#include "utils/InitializerPhysicalWarmResult.h"

namespace ov_init {

// These builds may use -ffast-math, which removes std::isfinite checks.
inline bool finite_initializer_time(double value) {
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

/** Fixed conversion from individual camera clocks to the initializer's reference clock.
 *
 * t_imu = t_raw_cam + td_cam = t_reference + td_reference.
 * Missing entries retain the legacy scalar offset. A zero shift returns the
 * original double directly, including for the all-equal legacy configuration.
 */
struct InitializerCameraClock {
  bool valid = false;
  bool unequal_offsets = false;
  double reference_td = 0.0;
  double min_delta = 0.0;
  double max_delta = 0.0;
  double min_imu_offset = 0.0;

  bool configure(const std::map<size_t, double> &camera_imu_dt, double ref_td) {
    valid = false;
    if (!finite_initializer_time(ref_td))
      return false;
    InitializerCameraClock next;
    next.reference_td = next.min_imu_offset = ref_td;
    for (const auto &entry : camera_imu_dt) {
      if (!finite_initializer_time(entry.second))
        return false;
      const double delta = entry.second - ref_td;
      if (!finite_initializer_time(delta))
        return false;
      next.deltas.emplace(entry.first, delta);
      next.unequal_offsets = next.unequal_offsets || delta != 0.0;
      next.min_delta = std::min(next.min_delta, delta);
      next.max_delta = std::max(next.max_delta, delta);
      next.min_imu_offset = std::min(next.min_imu_offset, entry.second);
    }
    if (!finite_initializer_time(next.max_delta - next.min_delta))
      return false;
    next.valid = true;
    *this = std::move(next);
    return true;
  }

  double to_reference(size_t camera, double raw_time) const {
    const auto it = deltas.find(camera);
    return it == deltas.end() || it->second == 0.0 ? raw_time : raw_time + it->second;
  }

private:
  std::map<size_t, double> deltas;
};

struct InitializerFeatureTimeBounds {
  double newest_ref_time = -1.0;
  double oldest_ref_time = std::numeric_limits<double>::infinity();
  size_t observation_count = 0;
};

using InitializerFeatureMap = std::unordered_map<size_t, std::shared_ptr<ov_core::Feature>>;

using InitializerRawTimeSidecar = std::map<size_t, std::map<size_t, std::vector<double>>>;

/** Freeze a private physical-time graph chart and its raw observation sidecar.
 * Bounds come only from covered physical exposure endpoints in this snapshot.
 * Validation is atomic; row compaction keeps both coordinate arrays and original
 * raw keys aligned. The legacy reference-clock helper below is unchanged.
 */
inline bool prepare_physical_initializer_features(
    InitializerFeatureMap &features, const std::map<size_t, double> &camera_offsets,
    double reference_offset, double first_imu, double last_imu, double window,
    InitializerFeatureTimeBounds &bounds, InitializerRawTimeSidecar &raw_keys) {
  if (!finite_initializer_time(reference_offset) || !finite_initializer_time(first_imu) ||
      !finite_initializer_time(last_imu) || !finite_initializer_time(window) ||
      !(last_imu > first_imu) || !(window > 0.)) return false;
  for (const auto &entry : camera_offsets)
    if (!finite_initializer_time(entry.second)) return false;
  const auto endpoint = [&](size_t camera, double raw) {
    const auto it = camera_offsets.find(camera);
    return raw + (it == camera_offsets.end() ? reference_offset : it->second);
  };
  double newest = -std::numeric_limits<double>::infinity();
  for (const auto &entry : features) {
    if (!entry.second || entry.second->featid != entry.first) return false;
    const auto &feature = *entry.second;
    for (const auto &camera : feature.timestamps) {
      const auto uv = feature.uvs.find(camera.first), un = feature.uvs_norm.find(camera.first);
      if (uv == feature.uvs.end() || un == feature.uvs_norm.end() ||
          uv->second.size() != camera.second.size() || un->second.size() != camera.second.size()) return false;
      std::set<uint64_t> seen;
      for (size_t i = 0; i < camera.second.size(); ++i) {
        const double raw = camera.second[i], time = endpoint(camera.first, raw);
        if (!finite_initializer_time(raw) || !finite_initializer_time(time) ||
            !seen.insert(ov_core::initializer_time_bits(raw)).second) return false;
        for (int j = 0; j < 2; ++j)
          if (!finite_initializer_time(uv->second[i](j)) || !finite_initializer_time(un->second[i](j))) return false;
        if (time >= first_imu && time <= last_imu) newest = std::max(newest, time);
      }
    }
  }
  if (!finite_initializer_time(newest)) return false;
  const double oldest = std::max(first_imu, newest - window);
  InitializerFeatureTimeBounds next;
  InitializerRawTimeSidecar next_keys;
  for (auto it = features.begin(); it != features.end();) {
    auto &feature = *it->second;
    size_t remaining = 0;
    for (auto &camera : feature.timestamps) {
      auto &uv = feature.uvs.at(camera.first), &un = feature.uvs_norm.at(camera.first);
      auto &keys = next_keys[it->first][camera.first];
      keys.reserve(camera.second.size());
      size_t keep = 0;
      for (size_t i = 0; i < camera.second.size(); ++i) {
        const double raw = camera.second[i], time = endpoint(camera.first, raw);
        if (time < oldest || time > last_imu) continue;
        keys.push_back(raw);
        camera.second[keep] = time;
        if (keep != i) { uv[keep] = uv[i]; un[keep] = un[i]; }
        if (next.observation_count++ == 0) next.newest_ref_time = next.oldest_ref_time = time;
        else { next.newest_ref_time = std::max(next.newest_ref_time, time); next.oldest_ref_time = std::min(next.oldest_ref_time, time); }
        ++keep;
      }
      camera.second.resize(keep); uv.resize(keep); un.resize(keep);
      remaining += keep;
    }
    if (!remaining) { next_keys.erase(it->first); it = features.erase(it); }
    else ++it;
  }
  bounds = next;
  raw_keys = std::move(next_keys);
  return next.observation_count != 0;
}

/** Retime a PRIVATE measurement snapshot, never the live FeatureDatabase map.
 *
 * FeatureDatabase::clone_features() supplies the required deep copy. Feature
 * identities, camera identities and pixel/normalized coordinates are preserved.
 * Measurements after latest_ref_time are removed with their aligned coordinates.
 * Anchor/triangulation metadata is not used by the initializer and is not changed.
 *
 * Validate the entire input before changing it. On failure both the map and
 * bounds are unchanged. For equal offsets and an unbounded horizon, all existing
 * timestamp and coordinate storage is left untouched.
 */
inline bool retime_initializer_features(
    InitializerFeatureMap &features, const InitializerCameraClock &clock, InitializerFeatureTimeBounds &bounds,
    double latest_ref_time = std::numeric_limits<double>::infinity()) {
  std::uint64_t limit_bits;
  std::memcpy(&limit_bits, &latest_ref_time, sizeof(limit_bits));
  const bool bounded = limit_bits != UINT64_C(0x7ff0000000000000);
  if (!clock.valid || (bounded && !finite_initializer_time(latest_ref_time)))
    return false;

  InitializerFeatureTimeBounds next;
  for (const auto &entry : features) {
    if (!entry.second)
      return false;
    const auto &feature = *entry.second;
    for (const auto &camera : feature.timestamps) {
      const auto uv = feature.uvs.find(camera.first);
      const auto uv_norm = feature.uvs_norm.find(camera.first);
      if (uv == feature.uvs.end() || uv_norm == feature.uvs_norm.end() ||
          uv->second.size() != camera.second.size() || uv_norm->second.size() != camera.second.size())
        return false;
      for (double raw_time : camera.second) {
        if (!finite_initializer_time(raw_time))
          return false;
        const double time = clock.to_reference(camera.first, raw_time);
        if (!finite_initializer_time(time))
          return false;
        if (bounded && time > latest_ref_time)
          continue;
        if (next.observation_count == 0)
          next.newest_ref_time = next.oldest_ref_time = time;
        else {
          next.newest_ref_time = std::max(next.newest_ref_time, time);
          next.oldest_ref_time = std::min(next.oldest_ref_time, time);
        }
        ++next.observation_count;
      }
    }
  }

  if (clock.unequal_offsets || bounded) {
    for (auto &entry : features) {
      auto &feature = *entry.second;
      for (auto &camera : feature.timestamps) {
        auto &uv = feature.uvs.at(camera.first);
        auto &uv_norm = feature.uvs_norm.at(camera.first);
        size_t keep = 0;
        for (size_t i = 0; i < camera.second.size(); ++i) {
          const double time = clock.to_reference(camera.first, camera.second[i]);
          if (bounded && time > latest_ref_time)
            continue;
          camera.second[keep] = time;
          if (keep != i) {
            uv[keep] = std::move(uv[i]);
            uv_norm[keep] = std::move(uv_norm[i]);
          }
          ++keep;
        }
        camera.second.resize(keep);
        uv.resize(keep);
        uv_norm.resize(keep);
      }
    }
  }
  bounds = next;
  return true;
}

} // namespace ov_init
#endif // OV_INIT_INITIALIZER_CAMERA_CLOCK_H
