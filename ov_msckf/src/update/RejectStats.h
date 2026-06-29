#ifndef OV_MSCKF_REJECT_STATS_H
#define OV_MSCKF_REJECT_STATS_H

// DIAGNOSTIC (stereo-vs-dual-mono investigation): per-update-call accounting of
// where features are dropped in the estimator, split by stereo (observed in >1
// camera this frame) vs mono, and by which gate fired:
//   - triangulation:  TRI_COND / TRI_DEPTH / TRI_NAN  (single_triangulation)
//   - refine:         GN_BASELINE / GN_DEPTH / GN_NAN (single_gaussnewton)
//   - chi2:           update Mahalanobis gate
// One line per updater call (~30 Hz), tagged with the state timestamp so it can
// be aligned offline with yaw rate from the ov_extended pipe. Mirrors TrackOCL's
// note_track_stats() logging. Compile-gated by kEnableRejectDiag so it is fully
// dead-stripped in production.
//
// Localizes the stereo yaw-collapse: TrackOCL hands the EKF healthy feature
// counts in both modes, but in stereo the EKF-consumed count collapses during
// yaw. This tells us which estimator gate is doing the rejecting.

#include <cstdio>
#include <mutex>
#include <unistd.h>

namespace ov_msckf {

// Flip to false to dead-strip all reject diagnostics (zero runtime cost).
static constexpr bool kEnableRejectDiag = true;

// when a STEREO (multi-camera) feature fails the update chi2 gate, retry
// it using only its anchor camera's observations (a reliable mono track) before
// discarding it -- "graceful degrade to mono". Robust to the undiagnosable sub-pixel
// stereo stiffness that makes chi2 fail even on good matches; recovers the feature's
// monocular value instead of throwing the whole landmark away. Off => old behavior.
static constexpr bool kEnableStereoToMonoDemote = true;

// Per-call counters. "s_" = stereo (feat seen in >1 cam this frame), "m_" = mono.
struct RejectCounters {
  int s_n = 0, m_n = 0;                       // features considered
  int s_tri_cond = 0, m_tri_cond = 0;
  int s_tri_depth = 0, m_tri_depth = 0;
  int s_tri_nan = 0, m_tri_nan = 0;
  int s_gn_base = 0, m_gn_base = 0;
  int s_gn_depth = 0, m_gn_depth = 0;
  int s_gn_nan = 0, m_gn_nan = 0;
  int s_chi2 = 0, m_chi2 = 0;
  int s_accept = 0, m_accept = 0;
  int s_demote = 0, m_demote = 0;              // stereo demoted to mono after chi2 failure
};

// Append one summary line. stage is e.g. "MSCKF" / "SLAM_INIT" / "SLAM_UPD".
inline void log_reject_stats(double meas_ts, const char *stage, const RejectCounters &c) {
  if (!kEnableRejectDiag)
    return;
  static std::mutex mtx;
  static FILE *fp = nullptr;
  static bool tried = false;
  std::lock_guard<std::mutex> lk(mtx);
  if (!tried) {
    tried = true;
    fp = fopen("/run/voxl-open-vins-reject-stats.log", "w"); // truncate once per process
    if (fp == nullptr)
      fp = fopen("/tmp/voxl-open-vins-reject-stats.log", "w");
    if (fp != nullptr) {
      fprintf(fp, "# voxl-open-vins-server per-update feature-reject accounting (pid=%d)\n", (int)getpid());
      fprintf(fp, "# meas_ts_s, stage, kind(S|M), n, tri_cond, tri_depth, tri_nan, "
                  "gn_base, gn_depth, gn_nan, chi2, accept, demote\n");
      fflush(fp);
    }
  }
  if (fp == nullptr)
    return;
  fprintf(fp, "%.6f, %s, S, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", meas_ts, stage,
          c.s_n, c.s_tri_cond, c.s_tri_depth, c.s_tri_nan, c.s_gn_base, c.s_gn_depth, c.s_gn_nan, c.s_chi2, c.s_accept, c.s_demote);
  fprintf(fp, "%.6f, %s, M, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", meas_ts, stage,
          c.m_n, c.m_tri_cond, c.m_tri_depth, c.m_tri_nan, c.m_gn_base, c.m_gn_depth, c.m_gn_nan, c.m_chi2, c.m_accept, c.m_demote);
  fflush(fp);
}

} // namespace ov_msckf

#endif // OV_MSCKF_REJECT_STATS_H
