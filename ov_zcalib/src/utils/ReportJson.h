/**
 * @file ReportJson.h
 * @brief Serialize calibration values, gate evidence, and caller provenance.
 *
 * Called after finish(), outside collection and solve. Report I/O failure
 * does not change the calibration verdict. Finite doubles use round-trip
 * precision; NaN and infinities serialize as null.
 */

#ifndef OV_ZCALIB_REPORT_JSON_H
#define OV_ZCALIB_REPORT_JSON_H

#include <string>
#include <vector>

#include "core/CalibSessionRunner.h"

namespace ov_zcalib {

/// Source and runtime metadata supplied by the embedding application.
struct ReportMeta {
  int schema = 1;

  // ---- provenance: what produced the observations ----
  std::string source_kind; ///< "live" | "log" | "record"
  std::string source_path; ///< log dir / record path ("" for live)
  std::string tracker;     ///< "TrackOCL" | "TrackKLT" | "none (record)"
  std::string machine;     ///< uname machine, e.g. "aarch64" / "x86_64"
  std::string binary;      ///< argv[0] basename
  std::string started_utc; ///< ISO-8601, session start

  // ---- what was asked for ----
  std::string profile_tag; ///< "voxl" | "voxl+flight" | "library"
  std::string config_path; ///< the calibrator profile consumed
  std::string out_yaml;    ///< where the result landed
  std::string record_path; ///< session record mirror ("" if disabled)
  std::string gravity_source; ///< "estimator_yaml" | "calibration_profile" | "session_record"
  int cam_mode = -1;

  // ---- per-camera identity (index == sensor id) ----
  std::vector<std::string> cam_names;
  /// Whether VIO currently flies this camera. Empty = unknown (the server's
  /// own --calibrate does not ask); sized with cam_names otherwise.
  std::vector<int> cam_vio_enabled;
  /// Which rung of the seed fallback supplied this camera's intrinsics:
  /// "chain" | "lens_cal" | "pipe_info" | "cli".
  std::vector<std::string> cam_seed_source;

  // ---- coexistence: what the session did about a running VIO ----
  int vins_running = -1;  ///< -1 unknown, 0 no, 1 yes
  std::string governor;   ///< "perf" | "kept" | "unavailable"
  std::string solve_when; ///< "now" | "deferred"
  int threads = 0;
  double collect_track_rate_hz = 0.0;

  // ---- drop accounting the sink owns (the runner never sees it) ----
  //
  // Bootstrap image shedding thins initialization evidence; collection losses
  // can break windows. Keep them separate. IMU loss matters in every phase.
  long long imu_drops = -1;            ///< damage in any phase
  long long frame_drops_collect = -1;  ///< damage: observations lost during COLLECT
  long long frame_drops_preboot = -1;  ///< benign: SETTLE/BOOTSTRAP shed
  long long image_stall_collect = -1;  ///< damage: bridge ring full during COLLECT
  long long image_stall_preboot = -1;  ///< benign
  long long ordering_gate_benign = -1; ///< late/bogus frames the solver must not see
  double span_collect_s = -1.0;
  double span_preboot_s = -1.0;
  /// Configured solver wall limit only; 0 unlimited, -1 not supplied by caller.
  /// Collection and held-out verification have separate durations.
  double solve_budget_s = -1.0;
};

/// Serialize a finished session; non-finite doubles become null.
std::string report_to_json(const SessionReport &rep, const ReportMeta &meta);

/// Write report_to_json() to `path` atomically (tmp + rename), so a reader can
/// never observe a half-written report. Returns false on I/O failure; the
/// caller must report incomplete output, not successful persistence. When supplied,
/// error receives the failed operation/path and OS error; errno is preserved.
bool write_report_json(const std::string &path, const SessionReport &rep, const ReportMeta &meta,
                       std::string *error = nullptr);

/// Human-readable name of an accel-gate verdict, as it appears in the JSON and
/// in any UI that renders it.
const char *accel_verdict_name(SessionReport::AccelGateVerdict v);

/// Human-readable name of a runner state.
const char *runner_state_name(RunnerState s);

} // namespace ov_zcalib

#endif // OV_ZCALIB_REPORT_JSON_H
