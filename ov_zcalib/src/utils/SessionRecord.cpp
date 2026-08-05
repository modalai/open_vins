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

#include <cstring>

using namespace ov_zcalib;

namespace {
constexpr uint32_t kMagic = 0x5343564Fu; // "OVCS"

/// The record is a SESSION ARTIFACT, not an interchange format: the binary that writes it is the
/// binary that reads it, and a session that needs replaying is re-run or re-fed from its log. So
/// there is no version ladder here and there must not become one -- kFormat is a single fingerprint
/// the reader demands EXACTLY, and a mismatch is a loud error rather than a compatibility branch.
/// Back-compat shims on an internal format buy nothing and cost a permanent tax on every field
/// added afterwards (each one grows another `if (version >= N)` that must be reasoned about forever).
/// Bump this whenever the layout below changes; old records then fail cleanly instead of silently
/// decoding as garbage.
constexpr uint32_t kFormat = 4; // 4: +imu.Tg (9, storage order) in the shared chain block
constexpr uint8_t kRecImu = 0;
constexpr uint8_t kRecFrame = 1;

template <typename T> bool wr(FILE *f, const T &v) { return std::fwrite(&v, sizeof(T), 1, f) == 1; }
template <typename T> bool rd(FILE *f, T &v) { return std::fread(&v, sizeof(T), 1, f) == 1; }

/// Header layout: magic | format | n_cams | shared IMU chain | N x camera | profile tag.
/// The rig shape leads, so the reader knows how much follows before it reads any of it.
bool write_seed(FILE *f, const SessionSeed &s) {
  const SharedCalib &c = s.calib;
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
    const double trhw = (n < s.tr_hw_seed.size()) ? s.tr_hw_seed[n] : 0.0;
    ok = ok && wr(f, trhw);
  }
  const uint8_t prof = (uint8_t)s.profile;
  const int32_t cm = (int32_t)s.cam_mode;
  ok = ok && wr(f, prof) && wr(f, cm);
  return ok;
}

bool read_seed(FILE *f, SessionSeed &s) {
  uint32_t magic = 0, format = 0, n_cams = 0;
  if (!rd(f, magic) || magic != kMagic || !rd(f, format))
    return false;
  if (format != kFormat) {
    std::fprintf(stderr, "[ov_zcalib] record format %u, this build reads %u -- re-run the session.\n", format, kFormat);
    return false;
  }
  if (!rd(f, n_cams) || n_cams == 0 || n_cams > 16)
    return false;
  SharedCalib &c = s.calib;
  c.cams.assign(n_cams, CamCalib());
  s.tr_hw_seed.assign(n_cams, 0.0);
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
    ok = ok && rd(f, s.tr_hw_seed[n]);
  }
  uint8_t prof = 0;
  int32_t cm = -1;
  ok = ok && rd(f, prof) && rd(f, cm);
  if (ok && prof <= (uint8_t)SessionProfile::EUROC)
    s.profile = (SessionProfile)prof;
  s.cam_mode = (int)cm;
  return ok;
}
} // namespace

bool SessionRecordWriter::open(const std::string &path, const SessionSeed &seed) {
  close();
  f_ = std::fopen(path.c_str(), "wb");
  if (!f_)
    return false;
  if (!write_seed(f_, seed)) {
    close();
    return false;
  }
  return true;
}

void SessionRecordWriter::write_imu(const RawImu &s) {
  if (!f_)
    return;
  wr(f_, kRecImu);
  wr(f_, s.timestamp);
  for (int i = 0; i < 3; ++i)
    wr(f_, s.wm(i));
  for (int i = 0; i < 3; ++i)
    wr(f_, s.am(i));
  wr(f_, s.temp_c);
}

void SessionRecordWriter::write_frame(const FrameObs &f) {
  if (!f_)
    return;
  wr(f_, kRecFrame);
  wr(f_, f.timestamp);
  wr(f_, f.exposure_s);
  wr(f_, f.temp_c);
  wr(f_, f.seq);
  wr(f_, f.cam); // v3
  const uint32_t n = (uint32_t)f.pts.size();
  wr(f_, n);
  if (n)
    std::fwrite(f.pts.data(), sizeof(FrameObsPoint), n, f_);
}

void SessionRecordWriter::close() {
  if (f_) {
    std::fclose(f_);
    f_ = nullptr;
  }
}

bool SessionRecordReader::open(const std::string &path) {
  close();
  f_ = std::fopen(path.c_str(), "rb");
  if (!f_)
    return false;
  if (!read_seed(f_, seed_)) {
    close();
    return false;
  }
  return true;
}

bool SessionRecordReader::next(bool &is_imu, RawImu &imu, FrameObs &frame) {
  if (!f_)
    return false;
  uint8_t type = 0;
  if (!rd(f_, type))
    return false;
  if (type == kRecImu) {
    is_imu = true;
    bool ok = rd(f_, imu.timestamp);
    for (int i = 0; i < 3; ++i)
      ok = ok && rd(f_, imu.wm(i));
    for (int i = 0; i < 3; ++i)
      ok = ok && rd(f_, imu.am(i));
    ok = ok && rd(f_, imu.temp_c);
    return ok;
  }
  if (type == kRecFrame) {
    is_imu = false;
    uint32_t n = 0;
    bool ok = rd(f_, frame.timestamp) && rd(f_, frame.exposure_s) && rd(f_, frame.temp_c) && rd(f_, frame.seq) &&
              rd(f_, frame.cam) && rd(f_, n);
    if (!ok || n > 100000)
      return false;
    frame.pts.resize(n);
    return n == 0 || std::fread(frame.pts.data(), sizeof(FrameObsPoint), n, f_) == n;
  }
  return false;
}

void SessionRecordReader::close() {
  if (f_) {
    std::fclose(f_);
    f_ = nullptr;
  }
}
