/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: voxl-logger feeder — converts a VOXL MPA log (log000N/run/mpa/
 * {imu_apps,<cam_pipe>}/data.csv + per-frame PNGs) into a standard session
 * record through the same record->replay path as every other source. The log
 * is largely SELF-DESCRIBING: the per-unit camera intrinsics ship inside the
 * log at data/modalai/opencv_<cam>_intrinsics.yml (fisheye/equidistant aware);
 * only the mechanical extrinsic seed and IMU noise come from the caller.
 *
 * Host-only (needs OpenCV imread); no MPA dependency.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_VOXL_LOG_FEEDER_H
#define OV_ZCALIB_VOXL_LOG_FEEDER_H

#include <string>

#include "SessionRecord.h"

namespace ov_zcalib {

class VoxlLogFeeder {
public:
  /**
   * @brief Load the per-unit camera intrinsics that ship inside the log
   *        (data/modalai/opencv_<base>_intrinsics.yml, where <base> is the
   *        camera pipe name with its stream suffixes stripped) into the seed:
   *        cam vector, fisheye flag, image size.
   * @return false when the yml is missing/unreadable (seed untouched).
   */
  static bool load_seed_intrinsics(const std::string &log_dir, const std::string &cam_pipe, SessionSeed &seed);

  /**
   * @brief Convert log_dir (a voxl-logger log000N directory) into a session
   *        record: IMU csv (timestamp/accel/gyro/temp) + camera csv
   *        (timestamp/exposure) + PNGs through TrackKLT at the seed
   *        intrinsics (CamEqui when seed.calib.cam_fisheye). Timestamps are
   *        rebased to the first IMU sample; frame stamps are start-of-exposure
   *        with the logged per-frame exposure.
   * @param mean_exposure_s optional out: mean logged exposure [s] of the
   *        converted frames — the exact td convention bridge (ov_zcalib stamps
   *        clones at MID-exposure; kalibr-style start-of-exposure chains leave
   *        exposure/2 inside td, so td_kalibr ~= td_ours + mean_exposure/2).
   */
  static bool convert(const std::string &log_dir, const std::string &cam_pipe, const SessionSeed &seed, const std::string &record_out,
                      double max_seconds, int num_feats, double *mean_exposure_s = nullptr);

  /// Reference extrinsics parsed from the log's own etc/modalai/extrinsics.conf
  /// (voxl-logger snapshots the rig config — the log is self-describing).
  /// Evaluation REFERENCE only, never a seed (blind protocol).
  struct RefExtrinsics {
    bool have_cam = false;              ///< imu_apps -> <cam_pipe base> entry found
    bool have_body = false;             ///< body -> imu_apps entry found
    double cam_rpy[3] = {0, 0, 0};      ///< RPY_parent_to_child [deg], vcc convention
    double cam_t[3] = {0, 0, 0};        ///< T_child_wrt_parent [m]
    double body_rpy[3] = {0, 0, 0};     ///< body -> imu_apps RPY [deg] (Stinger Rx180, Starling identity)
  };
  /**
   * @brief Parse the rig extrinsics snapshot inside the log. The cam entry is
   *        matched by the LONGEST conf child name that prefixes cam_pipe
   *        (pipe names carry stream suffixes: tracking_front_misp_norm ->
   *        tracking_front). Returns false when the conf is missing/unparsable.
   */
  static bool load_ref_extrinsics(const std::string &log_dir, const std::string &cam_pipe, RefExtrinsics &out);
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_VOXL_LOG_FEEDER_H
