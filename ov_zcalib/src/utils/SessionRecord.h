/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: session record -- the MANDATORY disk mirror of every live session
 * and the replay input (CI path). The record stores the post-tracking streams
 * (RawImu + FrameObs) in ARRIVAL order plus the seed-calibration snapshot the
 * live session used, so a replay reproduces the downstream computation
 * bit-identically (same numbers in, same deterministic reduction). Images are
 * NOT stored by design (budget; the tracker already ran). Little-endian
 * fixed-layout records, no compression: a 10-min session is ~40 MB IMU +
 * ~15 MB tracks. Live-vs-replay bit-parity is a hard release gate.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_SESSION_RECORD_H
#define OV_ZCALIB_SESSION_RECORD_H

#include <cstdio>
#include <cstdint>
#include <string>

#include "../solve/WindowBA.h"
#include "StreamTypes.h"

namespace ov_zcalib {

/// The effective PROFILE a session ran under. A record that carries only the seed is NOT
/// replayable -- the config is half the computation, and replaying a device session under library
/// defaults silently scores a DIFFERENT estimator (measured: the euroc record aborts instantly on
/// replay). The profile tag makes the record self-describing, so `--replay <bin>` reproduces the
/// session that produced it, by construction.
enum class SessionProfile : uint8_t { LIBRARY = 0, VOXL = 1, VOXL_FLIGHT = 2, EUROC = 3 };

/// Seed snapshot stored in the record header (what the live session ran with).
/// The rolling-shutter readout needs no separate field: CamCalib::tr IS the HAL3 hardware value
/// (never estimated), stored per camera inside the calib block like every other hardware fact.
/// Format 6 also records each camera's reprojection_sigma_px override. Zero inherits the scalar
/// window noise supplied by the replay profile/caller, as all format-5 recordings did. Live
/// callers resolve their configured camera values before recording; replay does not reread YAML.
/// Format 7 requires a completion footer, so a disk-full prefix cannot masquerade as a complete
/// session. Formats 5/6 remain readable, with completeness explicitly unknown at ordinary EOF.
struct SessionSeed {
  SharedCalib calib; ///< seed calibration for the whole rig: N cameras + the one IMU
  /// The effective configuration. A seed-only record is NOT replayable -- the config is half the
  /// computation (see SessionProfile).
  SessionProfile profile = SessionProfile::LIBRARY;
  int cam_mode = -1; ///< effective cam_mode (-1 = take the profile's own default)
};

class SessionRecordWriter {
public:
  ~SessionRecordWriter() { close(); }
  bool open(const std::string &path, const SessionSeed &seed);
  bool write_imu(const RawImu &s);
  bool write_frame(const FrameObs &f);
  bool flush();
  /// Flush, check close, and write a completion footer only if every prior write succeeded.
  /// Call explicitly and check the result: the destructor cannot report storage failures.
  bool close();
  bool is_open() const { return f_ != nullptr; }
  bool failed() const { return failed_; }
  const std::string &error() const { return error_; }

private:
  bool fail_(const char *message);
  FILE *f_ = nullptr;
  bool failed_ = false;
  std::string error_;
  uint64_t imu_count_ = 0, frame_count_ = 0;
};

class SessionRecordReader {
public:
  ~SessionRecordReader() { close(); }
  bool open(const std::string &path);
  const SessionSeed &seed() const { return seed_; }
  /// Sequential pull. Exactly one of imu/frame is filled per true return.
  /// A false return is EOF/completion OR an error; always inspect failed() after the loop.
  bool next(bool &is_imu, RawImu &imu, FrameObs &frame);
  bool failed() const { return failed_; }
  const std::string &error() const { return error_; }
  /// True only after a valid format-7 footer and EOF. Legacy formats have unknown completeness.
  bool complete() const { return complete_; }
  bool requires_completion_marker() const { return format_ == 7; }
  void close();

private:
  bool fail_(const char *message);
  FILE *f_ = nullptr;
  SessionSeed seed_;
  uint32_t format_ = 0;
  bool failed_ = false, ended_ = false, complete_ = false;
  std::string error_;
  uint64_t imu_count_ = 0, frame_count_ = 0;
};

/// Read ONLY the header of a record (seed + profile tag). Used by the replay
/// CLI to reconstruct the session's configuration before it builds the runner.
inline bool read_session_header(const std::string &path, SessionSeed &out) {
  SessionRecordReader rd;
  if (!rd.open(path))
    return false;
  out = rd.seed();
  return !rd.failed();
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
  return !rd.failed();
}

} // namespace ov_zcalib

#endif // OV_ZCALIB_SESSION_RECORD_H
