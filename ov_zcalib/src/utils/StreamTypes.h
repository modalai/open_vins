/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: stream-level types shared by the feeders, the recorder and the
 * harvester. FrameObs is the post-tracking camera sample: the session record
 * stores tracked observations (compact, ~KB/s), NOT images -- replay is then
 * bit-identical by construction and a full session stays under the retention
 * budget. Bearings are NOT stored: the consumer recomputes them from the
 * recorded seed intrinsics, so live and replay share one code path.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_STREAM_TYPES_H
#define OV_ZCALIB_STREAM_TYPES_H

#include <cstdint>
#include <vector>

namespace ov_zcalib {

/// One tracked point in a frame (raw DISTORTED pixels; id stable along the track)
struct FrameObsPoint {
  uint32_t id = 0;
  float u = 0.f, v = 0.f;
};

/// One post-tracking camera frame sample.
///
/// A frame is only well-defined once you know WHICH camera produced it: a rig with two
/// asynchronous, non-overlapping cameras has two extrinsics, two time offsets and two
/// intrinsic sets sharing ONE IMU, so `cam` selects which calibration block a point
/// reprojects through. Feature ids stay globally unique across cameras (ov_core::TrackBase
/// hands out ids from a single atomic counter), so tracks never collide -- but a track never
/// crosses cameras either, and `cam` is what makes that structural instead of accidental.
struct FrameObs {
  /// Center-row mid-exposure instant, camera clock [s]: the PRODUCER (server camera ingest /
  /// VoxlLogFeeder) stamps HAL3 start-of-exposure + (readout + exposure)/2 before anything
  /// downstream sees the frame. Consumers use it verbatim -- re-adding exposure or readout
  /// double-counts. Rows deviate from it by the centered fraction (v/h - 0.5) * readout.
  double timestamp = 0.0;
  float exposure_s = 0.f;  ///< exposure time [s] -- provenance/diagnostics only (already in timestamp)
  float temp_c = 0.f;      ///< IMU temperature snapshot at ingest (0 if unavailable)
  uint32_t seq = 0;        ///< producer frame counter (gap/drop detection)
  uint32_t cam = 0;        ///< camera id (index into the per-camera calibration blocks)
  std::vector<FrameObsPoint> pts;
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_STREAM_TYPES_H
