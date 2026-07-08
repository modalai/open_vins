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
#include <unordered_set>

namespace ov_msckf {

// Flip to false to dead-strip all reject diagnostics (zero runtime cost).
static constexpr bool kEnableRejectDiag = true;

// when a STEREO (multi-camera) feature fails the update chi2 gate, retry
// it using only its anchor camera's observations (a reliable mono track) before
// discarding it -- "graceful degrade to mono". Robust to the undiagnosable sub-pixel
// stereo stiffness that makes chi2 fail even on good matches; recovers the feature's
// monocular value instead of throwing the whole landmark away. Off => old behavior.
static constexpr bool kEnableStereoToMonoDemote = true;

// EXPERIMENT (2026-07, time-diversity gate): matcher-threshold tightening (zncc_min/
// margin_min) hit diminishing returns for the "stereo feat landmarked too close"
// problem -- tightening cuts good and bad matches roughly proportionally rather than
// selectively. This gate instead rejects a stereo delayed_init candidate outright
// (before triangulation even runs) if its observations span fewer than this many
// DISTINCT timestamps -- i.e. a pure single-instant L/R pair (n_distinct_ts==1) with
// zero temporal parallax is refused a persistent SLAM landmark, regardless of how
// confident the match looked. A rejected candidate is simply dropped this cycle (it
// can still be reconsidered once it accumulates more parallax); it is NOT
// auto-demoted to mono here. 1 = off (today's behavior, thin single-instant pairs
// allowed); bump to see the tradeoff between fewer bad promotions and lower stereo
// SLAM_INIT yield (both counted in RejectCounters.s_thin_ts vs s_accept below).
static constexpr int kMinStereoDistinctTs = 2;

// Per-call counters. "s_" = stereo (feat seen in >1 cam this frame), "m_" = mono.
struct RejectCounters {
  int s_n = 0, m_n = 0;                       // features considered
  int s_thin_ts = 0;                          // stereo-only: rejected by kMinStereoDistinctTs before triangulation
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
      fprintf(fp, "# meas_ts_s, stage, kind(S|M), n, thin_ts, tri_cond, tri_depth, tri_nan, "
                  "gn_base, gn_depth, gn_nan, chi2, accept, demote\n");
      fflush(fp);
    }
  }
  if (fp == nullptr)
    return;
  fprintf(fp, "%.6f, %s, S, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", meas_ts, stage, c.s_n, c.s_thin_ts, c.s_tri_cond,
          c.s_tri_depth, c.s_tri_nan, c.s_gn_base, c.s_gn_depth, c.s_gn_nan, c.s_chi2, c.s_accept, c.s_demote);
  fprintf(fp, "%.6f, %s, M, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", meas_ts, stage, c.m_n, 0, c.m_tri_cond, c.m_tri_depth,
          c.m_tri_nan, c.m_gn_base, c.m_gn_depth, c.m_gn_nan, c.m_chi2, c.m_accept, c.m_demote);
  fflush(fp);
}

// DIAGNOSTIC (2026-07, "stereo feat landmarked too close" investigation): a feature
// with solid L/R correspondence was observed to hold a correct, stable triangulated
// position for hundreds of cycles, then discontinuously jump to a position roughly
// half its true depth over a couple of cycles, before slowly drifting back -- i.e.
// the EKF-consumed position (not just the match) went wrong. The suspected mechanism:
// the landmark got marginalized out of state->_features_SLAM and was re-triangulated
// from scratch via delayed_init/try_init using only whatever thin/short-baseline
// observations remained in the current clone window (possibly just the current-instant
// stereo pair, with no temporal parallax), rather than resuming its prior well-conditioned
// mono history. This logs one line per successful delayed_init insert so a REINIT
// (featid previously seen in SLAM state, now reappearing after being dropped) can be
// distinguished from a FRESH (first-ever) insert, and its observation time-diversity /
// triangulated depth cross-checked against the moment a bad jump is seen downstream.
// Compile-gated (kEnableReinitDiag) so it is fully dead-stripped when not needed.
static constexpr bool kEnableReinitDiag = false;

// Per-feature stereo-match confidence, mirroring ov_core::TrackOCL::StereoConfidence
// but kept as an independent type so this diagnostic header (included from
// UpdaterSLAM.h/.cpp) doesn't need to pull in TrackOCL.h and its OpenCL dependencies.
// VioManager (which already includes TrackOCL.h) copies from
// TrackOCL::stereo_confidence_map() into this type before calling delayed_init.
struct StereoMatchConfidence {
  float peak_zncc = -2.0f; // forward peak ZNCC, in [-1, 1]; -2 = no data recorded
  float margin = -1.0f;    // peak - runner_up; uniqueness signal
  float lr_err = -1.0f;    // px residual of right->left round-trip
};

struct ReinitEvent {
  double meas_ts = 0.0;
  size_t featid = 0;
  bool is_stereo = false;
  bool is_reinit = false;     // true => this featid was previously in _features_SLAM and got dropped
  int n_cams = 0;
  int n_obs = 0;
  int n_distinct_ts = 0;      // distinct measurement timestamps across all cameras used for this triangulation
  double time_span_s = 0.0;   // max_ts - min_ts of those measurements (0 => pure single-instant stereo, no temporal parallax at all)
  double depth_anchor = 0.0;  // triangulated depth (p_FinA.z) in the anchor camera frame
  // For the n_cams==2 && n_distinct_ts==1 case only: the actual baseline vector norm (meters)
  // between the anchor camera pose and the other camera's pose used by single_triangulation,
  // read straight from clones_cam -- i.e. the real extrinsic separation the geometry was
  // computed with. -1 if not applicable/not computed. If this consistently matches the known
  // physical stereo baseline, the pose lookup is exonerated and a bad depth here points at the
  // L/R correspondence itself; if it's anomalous, the pose lookup is the bug.
  double baseline_norm = -1.0;
  // Matcher's OWN confidence for the L/R correspondence behind this feature, looked up
  // by featid from TrackOCL::stereo_confidence_map() at the moment delayed_init ran.
  // Sentinel defaults (peak_zncc=-2, margin=lr_err=-1) mean "no data" (e.g. tracker not
  // in use, or this feature's pair wasn't freshly matched this exact cycle). This is
  // what tells apart "matcher is confident but wrong" (a real scene ambiguity -- fix
  // belongs on the estimator/corroboration side) from "matcher is unconfident and
  // wrong" (gates just need tightening further).
  float peak_zncc = -2.0f;
  float margin = -1.0f;
  float lr_err = -1.0f;
};

// Returns true if featid has been inserted into SLAM state before (i.e. this is a
// re-init after being dropped), and records featid as seen either way. Persists for
// the life of the process, same lifetime assumption as RejectStats' static FILE*.
inline bool reinit_mark_and_check(size_t featid) {
  static std::unordered_set<size_t> ever_in_slam;
  auto res = ever_in_slam.insert(featid);
  return !res.second; // insert() reports .second==false when the key already existed
}

inline void log_reinit_event(const ReinitEvent &e) {
  if (!kEnableReinitDiag)
    return;
  static std::mutex mtx;
  static FILE *fp = nullptr;
  static bool tried = false;
  std::lock_guard<std::mutex> lk(mtx);
  if (!tried) {
    tried = true;
    fp = fopen("/run/voxl-open-vins-reinit-stats.log", "w"); // truncate once per process
    if (fp == nullptr)
      fp = fopen("/tmp/voxl-open-vins-reinit-stats.log", "w");
    if (fp != nullptr) {
      fprintf(fp, "# voxl-open-vins-server SLAM delayed_init insert events (pid=%d)\n", (int)getpid());
      fprintf(fp, "# meas_ts_s, featid, stereo(0|1), reinit(0|1), n_cams, n_obs, n_distinct_ts, time_span_s, depth_anchor_m, "
                  "baseline_norm_m, peak_zncc, margin, lr_err_px\n");
      fflush(fp);
    }
  }
  if (fp == nullptr)
    return;
  fprintf(fp, "%.6f, %zu, %d, %d, %d, %d, %d, %.4f, %.4f, %.5f, %.4f, %.4f, %.4f\n", e.meas_ts, e.featid, e.is_stereo ? 1 : 0,
          e.is_reinit ? 1 : 0, e.n_cams, e.n_obs, e.n_distinct_ts, e.time_span_s, e.depth_anchor, e.baseline_norm, e.peak_zncc, e.margin,
          e.lr_err);
  fflush(fp);
}

} // namespace ov_msckf

#endif // OV_MSCKF_REJECT_STATS_H
