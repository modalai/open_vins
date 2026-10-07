/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: session record read/write (see SessionRecord.h).
 * Fixed little-endian layout, shared by the aarch64 target and the x86 CI
 * host (both little-endian; a big-endian port would need byte swaps here).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "SessionRecord.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unistd.h>

using namespace ov_zcalib;

namespace {
constexpr uint32_t kMagic = 0x5343564Fu; // "OVCS"

/// Format 7 retains format 6's camera-noise header and requires a completion footer. Formats
/// 5/6 remain readable for existing evidence, but cannot certify a record-boundary EOF as a
/// complete session. Older binaries reject format 7 rather than silently ignoring its footer.
constexpr uint32_t kFormat = 7;
constexpr uint32_t kCameraNoiseFormat = 6;
constexpr uint32_t kLegacyFormat = 5; // Hardware tr; center-row mid-exposure frame stamps.
constexpr uint8_t kRecImu = 0;
constexpr uint8_t kRecFrame = 1;
constexpr uint8_t kRecEnd = 2;
constexpr uint32_t kEndMagic = 0x4543564Fu; // "OVCE"

template <typename T> bool wr(FILE *f, const T &v) { return std::fwrite(&v, sizeof(T), 1, f) == 1; }
template <typename T> bool rd(FILE *f, T &v) { return std::fread(&v, sizeof(T), 1, f) == 1; }

/// Zero is an internal legacy-window fallback; configured camera noise must be positive.
/// Inspect IEEE bits because -ffast-math can remove std::isfinite and NaN comparisons.
bool valid_reprojection_sigma(double value) {
  uint64_t bits;
  static_assert(sizeof(value) == sizeof(bits) && std::numeric_limits<double>::is_iec559,
                "session records require IEEE 754 binary64");
  std::memcpy(&bits, &value, sizeof(bits));
  const uint64_t magnitude = bits & UINT64_C(0x7fffffffffffffff);
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000) &&
         ((bits & UINT64_C(0x8000000000000000)) == 0 || magnitude == 0);
}

/// Header: magic | format | n_cams | shared IMU chain | N x camera | profile | cam_mode
///         | N x reprojection_sigma_px (formats 6/7).
/// The rig shape leads, so the reader knows how much follows before it reads any of it.
bool write_seed(FILE *f, const SessionSeed &s) {
  const SharedCalib &c = s.calib;
  if (c.cams.empty() || c.cams.size() > 16)
    return false;
  for (const CamCalib &cam : c.cams)
    if (!valid_reprojection_sigma(cam.reprojection_sigma_px))
      return false;
  const uint32_t n_cams = (uint32_t)c.cams.size();
  bool ok = wr(f, kMagic) && wr(f, kFormat) && wr(f, n_cams);
  // ---- shared: the one IMU ----
  for (int i = 0; i < 6; ++i)
    ok = ok && wr(f, c.imu.dw(i));
  for (int i = 0; i < 6; ++i)
    ok = ok && wr(f, c.imu.da(i));
  for (int i = 0; i < 4; ++i)
    ok = ok && wr(f, c.imu.q_AtoI(i));
  for (int i = 0; i < 9; ++i)
    ok = ok && wr(f, c.imu.Tg.data()[i]);
  ok = ok && wr(f, c.grav_mag);
  ok = ok && wr(f, c.noise.sigma_w) && wr(f, c.noise.sigma_wb) && wr(f, c.noise.sigma_a) && wr(f, c.noise.sigma_ab);
  // ---- per camera ----
  for (uint32_t n = 0; n < n_cams; ++n) {
    const CamCalib &k = c.cams[n];
    for (int i = 0; i < 8; ++i)
      ok = ok && wr(f, k.cam(i));
    const uint8_t fe = k.fisheye ? 1 : 0;
    ok = ok && wr(f, fe);
    for (int i = 0; i < 4; ++i)
      ok = ok && wr(f, k.q_ItoC(i));
    for (int i = 0; i < 3; ++i)
      ok = ok && wr(f, k.p_IinC(i));
    ok = ok && wr(f, k.td) && wr(f, k.tr);
    const uint32_t w = (uint32_t)k.img_w, h = (uint32_t)k.img_h;
    ok = ok && wr(f, w) && wr(f, h);
    ok = ok && wr(f, k.fps);
    const uint8_t roll = k.rolling ? 1 : 0;
    ok = ok && wr(f, roll);
  }
  const uint8_t prof = (uint8_t)s.profile;
  const int32_t cm = (int32_t)s.cam_mode;
  ok = ok && wr(f, prof) && wr(f, cm);
  for (const CamCalib &cam : c.cams)
    ok = ok && wr(f, cam.reprojection_sigma_px);
  return ok;
}

bool read_seed(FILE *f, SessionSeed &s, uint32_t &read_format) {
  uint32_t magic = 0, format = 0, n_cams = 0;
  if (!rd(f, magic) || magic != kMagic || !rd(f, format))
    return false;
  if (format != kLegacyFormat && format != kCameraNoiseFormat && format != kFormat) {
    std::fprintf(stderr, "[ov_zcalib] record format %u, this build reads %u, %u and %u -- re-run the session.\n", format,
                 kLegacyFormat, kCameraNoiseFormat, kFormat);
    return false;
  }
  if (!rd(f, n_cams) || n_cams == 0 || n_cams > 16)
    return false;
  SharedCalib &c = s.calib;
  c.cams.assign(n_cams, CamCalib());
  bool ok = true;
  for (int i = 0; i < 6; ++i)
    ok = ok && rd(f, c.imu.dw(i));
  for (int i = 0; i < 6; ++i)
    ok = ok && rd(f, c.imu.da(i));
  for (int i = 0; i < 4; ++i)
    ok = ok && rd(f, c.imu.q_AtoI(i));
  for (int i = 0; i < 9; ++i)
    ok = ok && rd(f, c.imu.Tg.data()[i]);
  ok = ok && rd(f, c.grav_mag);
  ok = ok && rd(f, c.noise.sigma_w) && rd(f, c.noise.sigma_wb) && rd(f, c.noise.sigma_a) && rd(f, c.noise.sigma_ab);
  for (uint32_t n = 0; n < n_cams && ok; ++n) {
    CamCalib &k = c.cams[n];
    for (int i = 0; i < 8; ++i)
      ok = ok && rd(f, k.cam(i));
    uint8_t fe = 0;
    ok = ok && rd(f, fe);
    k.fisheye = (fe != 0);
    for (int i = 0; i < 4; ++i)
      ok = ok && rd(f, k.q_ItoC(i));
    for (int i = 0; i < 3; ++i)
      ok = ok && rd(f, k.p_IinC(i));
    ok = ok && rd(f, k.td) && rd(f, k.tr);
    uint32_t w = 0, h = 0;
    ok = ok && rd(f, w) && rd(f, h);
    k.img_w = (int)w;
    k.img_h = (int)h;
    ok = ok && rd(f, k.fps);
    uint8_t roll = 0;
    ok = ok && rd(f, roll);
    k.rolling = (roll != 0);
  }
  uint8_t prof = 0;
  int32_t cm = -1;
  ok = ok && rd(f, prof) && rd(f, cm);
  if (ok && prof <= (uint8_t)SessionProfile::EUROC)
    s.profile = (SessionProfile)prof;
  s.cam_mode = (int)cm;
  for (CamCalib &cam : c.cams) {
    // Set explicitly even for format 5, including when a reader is reused after format 6.
    cam.reprojection_sigma_px = 0.0;
    if (format >= kCameraNoiseFormat)
      ok = ok && rd(f, cam.reprojection_sigma_px) && valid_reprojection_sigma(cam.reprojection_sigma_px);
  }
  if (ok)
    read_format = format;
  return ok;
}
} // namespace

bool SessionRecordWriter::open(const std::string &path, const SessionSeed &seed) {
  close();
  failed_ = false;
  error_.clear();
  imu_count_ = frame_count_ = 0;
  f_ = std::fopen(path.c_str(), "wb");
  if (!f_)
    return fail_("could not open session record");
  errno = 0;
  if (!write_seed(f_, seed)) {
    fail_("invalid or unwritable session record header");
    close();
    return false;
  }
  // Detect an already-full filesystem before collection asks the operator to move.
  if (!flush()) {
    close();
    return false;
  }
  return true;
}

bool SessionRecordWriter::fail_(const char *message) {
  if (!failed_) {
    failed_ = true;
    error_ = message;
    if (errno)
      error_ += std::string(": ") + std::strerror(errno);
  }
  return false;
}

bool SessionRecordWriter::write_imu(const RawImu &s) {
  if (failed_)
    return false;
  if (!f_)
    return fail_("session record is not open");
  bool ok = wr(f_, kRecImu) && wr(f_, s.timestamp);
  for (int i = 0; i < 3; ++i)
    ok = ok && wr(f_, s.wm(i));
  for (int i = 0; i < 3; ++i)
    ok = ok && wr(f_, s.am(i));
  ok = ok && wr(f_, s.temp_c);
  if (!ok)
    return fail_("could not write session IMU sample");
  ++imu_count_;
  return true;
}

bool SessionRecordWriter::write_frame(const FrameObs &f) {
  if (failed_)
    return false;
  if (!f_)
    return fail_("session record is not open");
  if (f.pts.size() > 100000) {
    errno = 0;
    return fail_("session frame exceeds record point limit");
  }
  bool ok = wr(f_, kRecFrame) && wr(f_, f.timestamp) && wr(f_, f.exposure_s) && wr(f_, f.temp_c) &&
            wr(f_, f.seq) && wr(f_, f.cam);
  const uint32_t n = (uint32_t)f.pts.size();
  ok = ok && wr(f_, n);
  if (n)
    ok = ok && std::fwrite(f.pts.data(), sizeof(FrameObsPoint), n, f_) == n;
  if (!ok)
    return fail_("could not write session camera frame");
  ++frame_count_;
  return true;
}

bool SessionRecordWriter::flush() {
  if (!f_)
    return !failed_;
  if (std::fflush(f_) != 0)
    fail_("could not flush session record");
  return !failed_;
}

bool SessionRecordWriter::close() {
  if (!f_)
    return !failed_;
  // Resolve buffered data errors before declaring the stream complete.
  flush();
  if (!failed_ && ::fsync(::fileno(f_)) != 0)
    fail_("could not sync session record data");
  // No completion marker after a failed write. Counts also protect against a misplaced footer.
  if (!failed_ && !(wr(f_, kRecEnd) && wr(f_, kEndMagic) && wr(f_, imu_count_) && wr(f_, frame_count_)))
    fail_("could not write session completion footer");
  flush();
  if (!failed_ && ::fsync(::fileno(f_)) != 0)
    fail_("could not sync session completion footer");
  if (std::fclose(f_) != 0)
    fail_("could not close session record");
  f_ = nullptr;
  return !failed_;
}

bool SessionRecordReader::open(const std::string &path) {
  close();
  failed_ = ended_ = complete_ = false;
  error_.clear();
  format_ = 0;
  imu_count_ = frame_count_ = 0;
  f_ = std::fopen(path.c_str(), "rb");
  if (!f_)
    return fail_("could not open session record");
  SessionSeed candidate;
  if (!read_seed(f_, candidate, format_)) {
    fail_("invalid or truncated session record header");
    close();
    return false;
  }
  seed_ = candidate;
  return true;
}

bool SessionRecordReader::fail_(const char *message) {
  if (!failed_)
    error_ = message;
  failed_ = ended_ = true;
  complete_ = false;
  return false;
}

bool SessionRecordReader::next(bool &is_imu, RawImu &imu, FrameObs &frame) {
  if (!f_ || failed_ || ended_)
    return false;
  const int type = std::fgetc(f_);
  if (type == EOF) {
    if (std::ferror(f_))
      return fail_("I/O error while reading session record");
    if (format_ == kFormat)
      return fail_("session record is incomplete: missing completion footer");
    ended_ = true; // Legacy clean EOF: replayable, but completeness is unknown.
    return false;
  }
  if (type == kRecImu) {
    is_imu = true;
    bool ok = rd(f_, imu.timestamp);
    for (int i = 0; i < 3; ++i)
      ok = ok && rd(f_, imu.wm(i));
    for (int i = 0; i < 3; ++i)
      ok = ok && rd(f_, imu.am(i));
    ok = ok && rd(f_, imu.temp_c);
    if (!ok)
      return fail_("truncated or unreadable session IMU sample");
    ++imu_count_;
    return true;
  }
  if (type == kRecFrame) {
    is_imu = false;
    uint32_t n = 0;
    bool ok = rd(f_, frame.timestamp) && rd(f_, frame.exposure_s) && rd(f_, frame.temp_c) && rd(f_, frame.seq) &&
              rd(f_, frame.cam) && rd(f_, n);
    if (!ok)
      return fail_("truncated or unreadable session frame header");
    if (n > 100000)
      return fail_("invalid session frame point count");
    frame.pts.resize(n);
    if (n && std::fread(frame.pts.data(), sizeof(FrameObsPoint), n, f_) != n)
      return fail_("truncated or unreadable session frame points");
    ++frame_count_;
    return true;
  }
  if (type == kRecEnd && format_ == kFormat) {
    uint32_t magic = 0;
    uint64_t imu_count = 0, frame_count = 0;
    if (!rd(f_, magic) || !rd(f_, imu_count) || !rd(f_, frame_count))
      return fail_("truncated or unreadable session completion footer");
    if (magic != kEndMagic || imu_count != imu_count_ || frame_count != frame_count_)
      return fail_("invalid session completion footer or sample counts");
    if (std::fgetc(f_) != EOF)
      return fail_("unexpected data after session completion footer");
    if (std::ferror(f_))
      return fail_("I/O error after session completion footer");
    complete_ = ended_ = true;
    return false;
  }
  return fail_("unknown session record tag");
}

void SessionRecordReader::close() {
  if (f_) {
    std::fclose(f_);
    f_ = nullptr;
  }
}
