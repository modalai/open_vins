/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: calibration writeback (kalibr-chain fields consumed by the VOXL
 * config sync). Atomic write: temp file + rename; the caller keeps a rollback.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_YAML_WRITEBACK_H
#define OV_ZCALIB_YAML_WRITEBACK_H

#include <cstdio>
#include <string>

#include "../solve/WindowBA.h"
#include "utils/quat_ops.h"

namespace ov_zcalib {

/**
 * @param mean_exposure_s session-mean camera exposure [s], PER CAMERA (negative/absent = unknown).
 *        DIAGNOSTIC only: every stamp in the system is center-row mid-exposure (producers apply
 *        SOF + (readout + exposure)/2 at ingest), so timeshift_cam_imu is directly consumable by
 *        VIO with no exposure arithmetic anywhere downstream.
 *
 * Layout: the IMU chain is written once (there is one IMU), then each camera's own block under a
 * camN_ prefix. Flat keys, because that is what an OpenCV FileStorage consumer can actually read.
 */
inline bool write_calib_yaml(const std::string &path, SharedCalib &c, const std::vector<double> &mean_exposure_s = {},
                             const std::vector<std::string> *committed = nullptr,
                             const std::vector<std::string> *at_seed = nullptr) {
  const std::string tmp = path + ".tmp";
  FILE *f = std::fopen(tmp.c_str(), "w");
  if (!f)
    return false;
  const Eigen::Matrix3d Dw = ImuIntrinsicModel::ut(c.imu.dw);
  const Eigen::Matrix3d Da = ImuIntrinsicModel::ut(c.imu.da);
  const Eigen::Matrix3d R_A = ov_core::quat_2_Rot(c.imu.q_AtoI);
  auto mat = [&](const std::string &k, const Eigen::Matrix3d &M) {
    std::fprintf(f, "%s: [%.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f]\n", k.c_str(), M(0, 0), M(0, 1), M(0, 2), M(1, 0),
                 M(1, 1), M(1, 2), M(2, 0), M(2, 1), M(2, 2));
  };
  std::fprintf(f, "%%YAML:1.0\n# ov_zcalib writeback (imu2 model: Dw/Da upper-tri inverse intrinsics, R_ACCtoIMU)\n"
                  "# GAUGE: the IMU frame is GYRO-aligned (R_GtoI = I structural; Dw upper-tri\n"
                  "# closes the rotational gauge). Kalibr-style chains are ACCEL-aligned (M_a\n"
                  "# lower-tri): their R_ItoC differs by the physical gyro-vs-accel misalignment\n"
                  "# (~0.8 deg on ICM-class dies). Consume R_ItoC together with Dw/Da/R_ACCtoIMU.\n"
                  "# TD: frame stamps system-wide are CENTER-ROW MID-EXPOSURE (producers stamp\n"
                  "# HAL3 SOF + (readout + exposure)/2 at ingest), so timeshift_cam_imu here is\n"
                  "# directly what VIO consumes. Legacy raw-SOF chains (kalibr) differ by\n"
                  "# (readout + exposure)/2. t_readout is the HAL3 hardware value, never estimated.\n"
                  "# The IMU chain below is SHARED by every camera; each camera then has its own\n"
                  "# extrinsic, time offset, readout and intrinsics under its camN_ prefix.\n");
  mat("Dw", Dw);
  mat("Da", Da);
  mat("R_ACCtoIMU", R_A);
  // g-sensitivity in the SAME gauge as Dw/Da (w_hat = Dw*(w_m - bg - Tg*a_hat)); row-major here
  // for human eyes regardless of internal storage. Ships its seed unless the tg block committed.
  mat("Tg", c.imu.Tg);
  std::fprintf(f, "num_cameras: %d\n", c.n_cams());
  std::fprintf(f, "td_convention: center_row_mid_exposure\n");
  for (int n = 0; n < c.n_cams(); ++n) {
    const CamCalib &k = c.cams[(size_t)n];
    const std::string pre = "cam" + std::to_string(n) + "_";
    mat(pre + "R_ItoC", ov_core::quat_2_Rot(k.q_ItoC));
    std::fprintf(f, "%sp_IinC: [%.9f, %.9f, %.9f]\n", pre.c_str(), k.p_IinC(0), k.p_IinC(1), k.p_IinC(2));
    std::fprintf(f, "%stimeshift_cam_imu: %.9f\n", pre.c_str(), k.td);
    const double expo = ((size_t)n < mean_exposure_s.size()) ? mean_exposure_s[(size_t)n] : -1.0;
    if (expo >= 0.0)
      std::fprintf(f, "%smean_exposure_s: %.9f\n", pre.c_str(), expo); // diagnostic (AE evidence)
    std::fprintf(f, "%st_readout: %.9f\n", pre.c_str(), k.tr); // HAL3 hardware value, pass-through
    std::fprintf(f, "%scam_k: [%.6f, %.6f, %.6f, %.6f]\n%scam_d: [%.9f, %.9f, %.9f, %.9f]\n", pre.c_str(), k.cam(0), k.cam(1),
                 k.cam(2), k.cam(3), pre.c_str(), k.cam(4), k.cam(5), k.cam(6), k.cam(7));
  }
  // ---- PROVENANCE (load-bearing for writeback) ----
  // This file always carries EVERY field, because a consumer needs a complete
  // calibration. But only the blocks listed in `committed_blocks` were ESTIMATED
  // and passed this session's gates; the rest are the values the session was
  // SEEDED with (factory chain / bootstrap) and carry NO new information.
  // A writeback that ignores this and pushes the whole file into the
  // system-of-record will overwrite good per-unit factory data (e.g. a real Dw)
  // with a seed passenger. Touch `committed_blocks` ONLY.
  if (committed || at_seed) {
    auto list = [&](const char *key, const std::vector<std::string> *v) {
      std::fprintf(f, "%s: [", key);
      for (size_t i = 0; v && i < v->size(); ++i)
        std::fprintf(f, "%s%s", i ? ", " : "", (*v)[i].c_str());
      std::fprintf(f, "]\n");
    };
    list("committed_blocks", committed);
    list("seed_blocks", at_seed);
  }
  std::fclose(f);
  return std::rename(tmp.c_str(), path.c_str()) == 0;
}

} // namespace ov_zcalib

#endif // OV_ZCALIB_YAML_WRITEBACK_H
