/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: session record — the MANDATORY disk mirror of every live session
 * and the replay input (CI path). The record stores the post-tracking streams
 * (RawImu + FrameObs) in ARRIVAL order plus the seed-calibration snapshot the
 * live session used, so a replay reproduces the downstream computation
 * bit-identically (same numbers in, same deterministic reduction). Images are
 * NOT stored by design (budget; the tracker already ran). Little-endian
 * fixed-layout records, no compression: a 10-min session is ~40 MB IMU +
 * ~15 MB tracks. Live-vs-replay bit-parity is the S4a/S4b stage gate.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_SESSION_RECORD_H
#define OV_ZCALIB_SESSION_RECORD_H

#include <cstdio>
#include <string>

#include "../solve/WindowBA.h"
#include "StreamTypes.h"

namespace ov_zcalib {

/// The effective PROFILE a session ran under. A record that carries only the seed is NOT
/// replayable — the config is half the computation, and replaying a device session under library
/// defaults silently scores a DIFFERENT estimator (measured: the euroc record aborts instantly on
/// replay). The profile tag makes the record self-describing, so `--replay <bin>` reproduces the
/// session that produced it, by construction.
enum class SessionProfile : uint8_t { LIBRARY = 0, VOXL = 1, VOXL_FLIGHT = 2, EUROC = 3 };

/// Seed snapshot stored in the record header (what the live session ran with)
struct SessionSeed {
  SharedCalib calib; ///< seed calibration for the whole rig: N cameras + the one IMU
  /// HAL3 frame_readout_time_ns seed [s], per camera. The solved tr is cross-checked against it at
  /// VERIFY -- it is a hardware fact, so a solved value far from it means the session is wrong, not
  /// the datasheet. (Image geometry lives on CamCalib, where it belongs: it is per camera too.)
  std::vector<double> tr_hw_seed;
  /// The effective configuration. A seed-only record is NOT replayable -- the config is half the
  /// computation (see SessionProfile).
  SessionProfile profile = SessionProfile::LIBRARY;
  int cam_mode = -1; ///< effective cam_mode (-1 = take the profile's own default)
};

class SessionRecordWriter {
public:
  ~SessionRecordWriter() { close(); }
  bool open(const std::string &path, const SessionSeed &seed);
  void write_imu(const RawImu &s);
  void write_frame(const FrameObs &f);
  void close();
  bool is_open() const { return f_ != nullptr; }

private:
  FILE *f_ = nullptr;
};

class SessionRecordReader {
public:
  ~SessionRecordReader() { close(); }
  bool open(const std::string &path);
  const SessionSeed &seed() const { return seed_; }
  /// Sequential pull. Exactly one of imu/frame is filled per true return.
  bool next(bool &is_imu, RawImu &imu, FrameObs &frame);
  void close();

private:
  FILE *f_ = nullptr;
  SessionSeed seed_;
};

/// Read ONLY the header of a record (seed + profile tag). Used by the replay
/// CLI to reconstruct the session's configuration before it builds the runner.
inline bool read_session_header(const std::string &path, SessionSeed &out) {
  SessionRecordReader rd;
  if (!rd.open(path))
    return false;
  out = rd.seed();
  return true;
}

/// Pump a record through arbitrary sinks in recorded arrival order.
template <typename ImuSink, typename FrameSink>
inline bool replay_session(const std::string &path, SessionSeed &seed_out, ImuSink &&on_imu, FrameSink &&on_frame) {
  SessionRecordReader rd;
  if (!rd.open(path))
    return false;
  seed_out = rd.seed();
  bool is_imu = false;
  RawImu s;
  FrameObs f;
  while (rd.next(is_imu, s, f)) {
    if (is_imu)
      on_imu(s);
    else
      on_frame(f);
  }
  return true;
}

} // namespace ov_zcalib

#endif // OV_ZCALIB_SESSION_RECORD_H
