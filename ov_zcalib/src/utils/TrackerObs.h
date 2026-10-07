/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: tracker -> FrameObs adapter. The solver consumes post-tracking
 * observations only (raw DISTORTED pixels, stable ids), so ANY ov_core
 * TrackBase front-end plugs in here: TrackKLT on the host (log conversion,
 * tests), TrackOCL inside voxl-open-vins-server on the device (the only
 * tracker that exists on target -- DISABLE_TRACK_KLT strips KLT there).
 * Tracker choice therefore never touches solver parity: device sessions
 * record TrackOCL observations, host replays them tracker-free.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_TRACKER_OBS_H
#define OV_ZCALIB_TRACKER_OBS_H

#include <track/TrackBase.h>

#include "StreamTypes.h"

namespace ov_zcalib {

/// Snapshot the tracker's last-frame observations for one camera into a
/// FrameObs (call right after feed_new_camera returns; TrackBase guards the
/// last-vars under its own mutex). Pixels are the tracker's raw distorted
/// coordinates at native resolution.
inline FrameObs make_frame_obs(ov_core::TrackBase &tracker, size_t cam_id, double timestamp, float exposure_s, float temp_c,
                               uint32_t seq) {
  FrameObs fo;
  fo.timestamp = timestamp;
  fo.exposure_s = exposure_s;
  fo.temp_c = temp_c;
  fo.seq = seq;
  fo.cam = (uint32_t)cam_id;
  auto obs = tracker.get_last_obs();
  auto ids = tracker.get_last_ids();
  const auto &kps = obs[cam_id];
  const auto &kid = ids[cam_id];
  fo.pts.reserve(kps.size());
  for (size_t i = 0; i < kps.size() && i < kid.size(); ++i) {
    FrameObsPoint p;
    p.id = (uint32_t)kid[i];
    p.u = kps[i].pt.x;
    p.v = kps[i].pt.y;
    fo.pts.push_back(p);
  }
  return fo;
}

} // namespace ov_zcalib

#endif // OV_ZCALIB_TRACKER_OBS_H
