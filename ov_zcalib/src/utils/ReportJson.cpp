/**
 * @file ReportJson.cpp
 * @brief SessionReport -> JSON. See ReportJson.h.
 *
 * Serialize after CalibSessionRunner::finish(), outside collection and solve.
 */

#include "utils/ReportJson.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "utils/quat_ops.h"

namespace ov_zcalib {

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

namespace {

/// JSON string escape. Control characters below 0x20 must be \u-escaped or the
/// document is invalid; an abort_reason built from strerror() can carry them.
void emit_str(std::string &o, const std::string &s) {
  o += '"';
  for (char c : s) {
    switch (c) {
    case '"':
      o += "\\\"";
      break;
    case '\\':
      o += "\\\\";
      break;
    case '\n':
      o += "\\n";
      break;
    case '\r':
      o += "\\r";
      break;
    case '\t':
      o += "\\t";
      break;
    case '\b':
      o += "\\b";
      break;
    case '\f':
      o += "\\f";
      break;
    default:
      if ((unsigned char)c < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)(unsigned char)c);
        o += buf;
      } else {
        o += c;
      }
    }
  }
  o += '"';
}

/// Round-trip precision; encode NaN and infinities as JSON null.
void emit_num(std::string &o, double v) {
  // This target uses -ffast-math, which can optimize away std::isfinite.
  uint64_t bits;
  static_assert(sizeof(v) == sizeof(bits) && std::numeric_limits<double>::is_iec559,
                "report JSON requires IEEE 754 binary64");
  std::memcpy(&bits, &v, sizeof(bits));
  if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000)) {
    o += "null";
    return;
  }
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  o += buf;
}

void emit_int(std::string &o, long long v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%lld", v);
  o += buf;
}

void emit_bool(std::string &o, bool v) { o += v ? "true" : "false"; }

/// Key emitters. `sep` carries the leading comma so callers never have to
/// track "is this the first field" by hand.
struct Obj {
  std::string &o;
  bool first = true;
  explicit Obj(std::string &out) : o(out) { o += '{'; }
  void close() { o += '}'; }
  std::string &key(const char *k) {
    if (!first)
      o += ',';
    first = false;
    emit_str(o, k);
    o += ':';
    return o;
  }
  void s(const char *k, const std::string &v) { emit_str(key(k), v); }
  void n(const char *k, double v) { emit_num(key(k), v); }
  void i(const char *k, long long v) { emit_int(key(k), v); }
  void b(const char *k, bool v) { emit_bool(key(k), v); }
  /// Optional string: omitted entirely when empty, so a consumer can tell
  /// "not applicable" from "empty value".
  void s_opt(const char *k, const std::string &v) {
    if (!v.empty())
      s(k, v);
  }
  /// Optional number: omitted when the caller's sentinel says "unknown".
  void n_opt(const char *k, double v, double unknown) {
    if (v != unknown)
      n(k, v);
  }
  void i_opt(const char *k, long long v, long long unknown) {
    if (v != unknown)
      i(k, v);
  }
};

void emit_vec(std::string &o, const double *v, int n) {
  o += '[';
  for (int i = 0; i < n; i++) {
    if (i)
      o += ',';
    emit_num(o, v[i]);
  }
  o += ']';
}

void emit_vecd(std::string &o, const std::vector<double> &v) {
  o += '[';
  for (size_t i = 0; i < v.size(); i++) {
    if (i)
      o += ',';
    emit_num(o, v[i]);
  }
  o += ']';
}

void emit_eigen_vec(std::string &o, const Eigen::VectorXd &v) {
  o += '[';
  for (int i = 0; i < v.size(); i++) {
    if (i)
      o += ',';
    emit_num(o, v(i));
  }
  o += ']';
}

/// A 3x3 as a flat row-major 9-list -- the SAME layout the result YAML uses
/// (YamlWriteback's mat()), so a consumer parses one convention, not two.
void emit_mat3(std::string &o, const Eigen::Matrix3d &M) {
  o += '[';
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) {
      if (r || c)
        o += ',';
      emit_num(o, M(r, c));
    }
  o += ']';
}

/// Symmetric matrix as its packed UPPER triangle, row-major:
/// (0,0),(0,1)..(0,n-1),(1,1).. -- n(n+1)/2 entries instead of n^2. Lambda is
/// symmetric by construction and at the archival window shape n can reach ~60,
/// so this roughly halves the largest field in the document.
void emit_sym_packed(std::string &o, const Eigen::MatrixXd &M) {
  const int n = (int)M.rows();
  o += '[';
  bool first = true;
  for (int r = 0; r < n; r++)
    for (int c = r; c < n; c++) {
      if (!first)
        o += ',';
      first = false;
      emit_num(o, M(r, c));
    }
  o += ']';
}

} // namespace

// ---------------------------------------------------------------------------
// enum names
// ---------------------------------------------------------------------------

const char *accel_verdict_name(SessionReport::AccelGateVerdict v) {
  using V = SessionReport::AccelGateVerdict;
  switch (v) {
  case V::PRE_CLOSED:
    return "PRE_CLOSED";
  case V::SPLIT_CONSISTENT:
    return "SPLIT_CONSISTENT";
  case V::SPLIT_INCONSISTENT:
    return "SPLIT_INCONSISTENT";
  case V::PRECISION_WEAK:
    return "PRECISION_WEAK";
  case V::SPLIT_FAILED:
    return "SPLIT_FAILED";
  case V::WALD_CONSISTENT:
    return "WALD_CONSISTENT";
  case V::WALD_INCONSISTENT:
    return "WALD_INCONSISTENT";
  case V::WALD_UNOBSERVABLE:
    return "WALD_UNOBSERVABLE";
  }
  return "UNKNOWN";
}

const char *runner_state_name(RunnerState s) {
  switch (s) {
  case RunnerState::SETTLE:
    return "SETTLE";
  case RunnerState::BOOTSTRAP:
    return "BOOTSTRAP";
  case RunnerState::COLLECT:
    return "COLLECT";
  case RunnerState::THERMAL_HOLD:
    return "THERMAL_HOLD";
  case RunnerState::SOLVE_REFINE:
    return "SOLVE_REFINE";
  case RunnerState::VERIFY:
    return "VERIFY";
  case RunnerState::COMMIT:
    return "COMMIT";
  case RunnerState::DONE:
    return "DONE";
  case RunnerState::ABORT:
    return "ABORT";
  }
  return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// sections
// ---------------------------------------------------------------------------

namespace {

void emit_camera_calib(std::string &o, const CamCalib &k) {
  Obj c(o);
  c.key("R_ItoC");
  emit_mat3(o, ov_core::quat_2_Rot(k.q_ItoC));
  c.key("q_ItoC");
  emit_vec(o, k.q_ItoC.data(), 4);
  c.key("p_IinC");
  emit_vec(o, k.p_IinC.data(), 3);
  c.n("timeshift_cam_imu", k.td);
  c.n("t_readout", k.tr);
  c.key("cam_k");
  emit_vec(o, k.cam.data(), 4);
  c.key("cam_d");
  emit_vec(o, k.cam.data() + 4, 4);
  c.b("fisheye", k.fisheye);
  c.b("rolling", k.rolling);
  c.i("img_w", k.img_w);
  c.i("img_h", k.img_h);
  c.n("fps", k.fps);
  c.i("cam_mode", k.cam_mode);
  c.close();
}

void emit_shared_calib(std::string &o, const SharedCalib &sc) {
  Obj s(o);
  s.key("Dw");
  emit_mat3(o, ImuIntrinsicModel::ut(sc.imu.dw));
  s.key("Da");
  emit_mat3(o, ImuIntrinsicModel::ut(sc.imu.da));
  s.key("R_ACCtoIMU");
  emit_mat3(o, ov_core::quat_2_Rot(sc.imu.q_AtoI));
  s.key("Tg");
  emit_mat3(o, sc.imu.Tg);
  s.b("tg_enabled", sc.tg_enabled);
  s.n("grav_mag", sc.grav_mag);
  s.key("cams");
  o += '[';
  for (size_t i = 0; i < sc.cams.size(); i++) {
    if (i)
      o += ',';
    emit_camera_calib(o, sc.cams[i]);
  }
  o += ']';
  s.close();
}

void emit_blocks(std::string &o, const SessionReport &rep) {
  o += '[';
  for (size_t i = 0; i < rep.blocks.size(); i++) {
    if (i)
      o += ',';
    const BlockCommit &b = rep.blocks[i];
    Obj j(o);
    j.s("name", b.name);
    j.i("cam", b.cam);
    j.s("label", b.label());
    j.b("committed", b.committed);
    j.n("worst_ratio", b.worst_ratio);
    j.n("worst_sigma", b.worst_sigma);
    j.b("ceiling_ok", b.ceiling_ok);
    j.n("moved_sigma", b.moved_sigma);
    j.b("not_estimated", b.not_estimated);
    j.b("atomic_reverted", b.atomic_reverted);
    j.n("holdout_delta", b.holdout_delta);
    j.close();
  }
  o += ']';
}

void emit_stages(std::string &o, const SessionReport &rep) {
  o += '[';
  for (size_t i = 0; i < rep.evidence.size(); i++) {
    if (i)
      o += ',';
    const StageEvidence &e = rep.evidence[i];
    Obj j(o);
    j.s("label", e.label);
    j.i("passes", e.passes);
    j.i("accepted", e.accepted);
    j.i("windows", e.windows);
    j.i("dim_p", e.dim_p);
    j.n("wall_s", e.wall_s);
    j.n("seed_s", e.seed_s);
    j.n("preint_s", e.preint_s);
    j.n("inner_s", e.inner_s);
    j.n("export_s", e.export_s);
    j.n("factor_s", e.factor_s);
    j.i("iters", e.iters);
    j.i("warm", e.warm);
    j.i("cold", e.cold);
    j.i("preint_hit", e.phit);
    j.i("preint_miss", e.pmiss);
    j.i("tstop", e.tstop);
    j.n("merit", e.merit);
    j.n("qn_max", e.qn_max);
    j.i("stop_pass", e.stop_pass);
    j.b("hit_budget", e.hit_budget);
    j.i("rss_kb", e.rss_kb);
    j.close();
  }
  o += ']';
}

void emit_bootstrap(std::string &o, const SessionReport &rep) {
  o += '[';
  const size_t n = rep.handeye.size();
  for (size_t i = 0; i < n; i++) {
    if (i)
      o += ',';
    const HandEyeResult &h = rep.handeye[i];
    Obj j(o);
    j.i("cam", (long long)i);
    j.b("ok", h.ok);
    j.n("rmse_rad", h.rmse_rad);
    j.n("axis_diversity", h.axis_diversity);
    j.i("pairs_used", h.pairs_used);
    j.i("pairs_trimmed", h.pairs_trimmed);
    j.n("td_s", h.td);
    j.b("td_at_bound", h.td_at_bound);
    j.b("time_provisional", h.ok && i < rep.xcorr.size() && !rep.xcorr[i].temporally_consistent(rep.td_fine_range_s) &&
                                !(i < rep.epipolar_time.size() && rep.epipolar_time[i].ok));
    j.key("q_ItoC");
    emit_vec(o, h.q_ItoC.data(), 4);
    j.key("bg");
    emit_vec(o, h.bg.data(), 3);
    if (i < rep.xcorr.size()) {
      const TimeOffsetResult &x = rep.xcorr[i];
      j.key("xcorr");
      Obj xo(o);
      xo.b("ok", x.ok);
      xo.n("td_s", x.td);
      xo.n("peak_corr", x.peak_corr);
      xo.n("peak_sharpness", x.peak_sharpness);
      xo.b("at_bound", x.at_bound);
      xo.n("peak_trimmed", x.peak_trimmed);
      xo.n("trim_retention", x.trim_retention);
      xo.b("trim_consistent", x.trim_consistent);
      xo.n("td_split_delta", x.td_split_delta);
      xo.n("split_min_peak", x.split_min_peak);
      xo.n("temporal_split_delta", x.temporal_split_delta);
      xo.n("temporal_min_peak", x.temporal_min_peak);
      xo.b("temporal_consistent", x.temporally_consistent(rep.td_fine_range_s));
      // The verdict and the rules it was judged by -- so a plot never invents its own threshold.
      xo.b("certified", i < rep.xcorr_certified.size() && rep.xcorr_certified[i] != 0);
      xo.n("peak_floor", rep.xcorr_min_peak);
      xo.n("split_tol_s", rep.td_fine_range_s);
      xo.close();
    }
    if (i < rep.epipolar_time.size() && rep.epipolar_time[i].attempted) {
      const EpipolarTimeResult &e = rep.epipolar_time[i];
      j.key("geometric_time");
      Obj g(o);
      g.b("ok", e.ok);
      g.n("td_s", e.td);
      g.n("seed_td_s", e.seed_td);
      g.n("split_delta_s", e.split_delta);
      g.n("conditional_sigma_s", e.sigma_td);
      g.n("cost_before", e.cost_before);
      g.n("cost_after", e.cost_after);
      g.n("wall_s", e.wall_s);
      g.i("pairs", e.pairs);
      g.close();
    }
    // The curve behind those scalars: correlation vs lag, the kalibr-style td
    // plot. `corr[k]` is the correlation at lag `-search + k*step`. The scan
    // leaves lags with fewer than 8 overlapping samples at its -2.0 sentinel;
    // those are emitted as null so a plot draws a GAP there instead of a
    // correlation of -2, which would swamp the y-axis and hide the ridge.
    if (i < rep.xcorr_curve.size() && !rep.xcorr_curve[i].corr.empty()) {
      const XcorrCurve &cv = rep.xcorr_curve[i];
      j.key("xcorr_curve");
      Obj co(o);
      co.n("search_s", cv.search);
      co.n("step_s", cv.step);
      co.i("n_lags", (long long)cv.corr.size());
      co.key("corr");
      o += '[';
      for (size_t k = 0; k < cv.corr.size(); k++) {
        if (k)
          o += ',';
        if (cv.corr[k] <= -2.0)
          o += "null";
        else
          emit_num(o, cv.corr[k]);
      }
      o += ']';
      co.close();
    }
    j.close();
  }
  o += ']';
}

void emit_meta(std::string &o, const ReportMeta &m, const SessionReport &rep) {
  Obj s(o);
  s.key("source");
  {
    Obj src(o);
    src.s_opt("kind", m.source_kind);
    src.s_opt("path", m.source_path);
    src.s_opt("tracker", m.tracker);
    src.s_opt("machine", m.machine);
    src.s_opt("binary", m.binary);
    src.close();
  }
  s.s_opt("profile", m.profile_tag);
  s.s_opt("config_path", m.config_path);
  s.s_opt("out_yaml", m.out_yaml);
  s.s_opt("record_path", m.record_path);
  s.s_opt("started_utc", m.started_utc);
  if (m.cam_mode >= 0)
    s.i("cam_mode", m.cam_mode);
  s.n("t_solve_s", rep.t_solve_s);
  s.n("t_verify_s", rep.t_verify_s);
  s.n("t_total_s", rep.t_total_s);
  s.n_opt("span_collect_s", m.span_collect_s, -1.0);
  s.n_opt("span_preboot_s", m.span_preboot_s, -1.0);
  s.key("coexistence");
  {
    Obj cx(o);
    if (m.vins_running >= 0)
      cx.b("vins_running", m.vins_running != 0);
    cx.s_opt("governor", m.governor);
    cx.s_opt("solve_when", m.solve_when);
    if (m.threads > 0)
      cx.i("threads", m.threads);
    if (m.collect_track_rate_hz > 0)
      cx.n("collect_track_rate_hz", m.collect_track_rate_hz);
    cx.close();
  }
  s.close();
}

void emit_cameras(std::string &o, const ReportMeta &m, const SessionReport &rep) {
  // The camera IDENTITY list is meta-driven (the runner knows sensor ids, not
  // names). Size on whichever the caller actually supplied.
  size_t n = m.cam_names.size();
  if (n == 0)
    n = rep.committed.cams.size();
  o += '[';
  for (size_t i = 0; i < n; i++) {
    if (i)
      o += ',';
    Obj j(o);
    j.i("id", (long long)i);
    if (i < m.cam_names.size())
      j.s("name", m.cam_names[i]);
    if (i < m.cam_vio_enabled.size())
      j.b("vio_enabled", m.cam_vio_enabled[i] != 0);
    if (i < m.cam_seed_source.size())
      j.s("seed_source", m.cam_seed_source[i]);
    if (i < rep.committed.cams.size()) {
      const CamCalib &k = rep.committed.cams[i];
      j.i("img_w", k.img_w);
      j.i("img_h", k.img_h);
      j.n("fps", k.fps);
      j.b("rolling", k.rolling);
      j.n("t_readout", k.tr);
      j.b("fisheye", k.fisheye);
    }
    if (i < rep.mean_exposure_s.size())
      j.n("mean_exposure_s", rep.mean_exposure_s[i]);
    j.close();
  }
  o += ']';
}

} // namespace

// ---------------------------------------------------------------------------
// top level
// ---------------------------------------------------------------------------

std::string report_to_json(const SessionReport &rep, const ReportMeta &meta) {
  std::string o;
  // A flight-shape report is a few tens of kB; the archival shape with a full
  // Lambda and a long evidence table can reach a few hundred. One reserve, no
  // reallocation storm.
  o.reserve(256 * 1024);

  Obj root(o);
  root.i("schema", meta.schema);

  // ---- verdict ----
  root.key("verdict");
  {
    Obj v(o);
    v.s("state", runner_state_name(rep.final_state));
    v.s_opt("abort_reason", rep.abort_reason);
    v.i("exit_code", rep.final_state == RunnerState::DONE ? 0 : 2);
    v.s_opt("prompt", rep.prompt);
    v.close();
  }

  // ---- tainted: any stage whose inner solve hit the wall hang guard ----
  // Load leaked into the numerics; the run is not replay-deterministic and
  // must not be used for A/B or falsifier work. Hoisted to the top level so a
  // consumer cannot miss it.
  {
    long long tstop_total = 0;
    for (const auto &e : rep.evidence)
      tstop_total += e.tstop;
    root.b("tainted", tstop_total != 0);
    if (tstop_total != 0)
      root.i("tainted_inner_solves", tstop_total);
  }

  // ---- session / provenance ----
  root.key("session");
  emit_meta(o, meta, rep);

  root.key("cameras");
  emit_cameras(o, meta, rep);

  // ---- collection ----
  root.key("collection");
  {
    Obj c(o);
    c.i("harvested", rep.windows_harvested);
    c.i("retained", rep.windows_retained);
    c.i("holdout", rep.windows_holdout);
    c.i("fused", rep.windows_fused);
    c.i("rejected_seed", rep.windows_rejected_seed);
    c.i("rejected_gate", rep.windows_rejected_gate);
    c.i("rejected_ba", rep.windows_rejected_ba);
    c.i("invalidated", rep.windows_invalidated);
    c.i("probation", rep.windows_probation);
    c.i("probation_dropped", rep.windows_probation_dropped);
    c.n("min_eig_whitened", rep.min_eig_whitened);
    c.i("windows_a0", rep.windows_a0);
    c.i("windows_a1", rep.windows_a1);
    c.i("windows_b", rep.windows_b);
    c.n("min_eig_a0", rep.min_eig_a0);
    c.n("min_eig_a1", rep.min_eig_a1);
    c.n("min_eig_b", rep.min_eig_b);
    c.b("stage_a1_fallback", rep.stage_a1_fallback);
    // Phase-split, never summed -- see ReportMeta. `damaging` is the one-bit
    // answer to "must this session be re-run?".
    c.key("drops");
    {
      Obj d(o);
      d.i_opt("imu", meta.imu_drops, -1);
      d.i_opt("frames_collect", meta.frame_drops_collect, -1);
      d.i_opt("frames_preboot", meta.frame_drops_preboot, -1);
      d.i_opt("image_stall_collect", meta.image_stall_collect, -1);
      d.i_opt("image_stall_preboot", meta.image_stall_preboot, -1);
      d.i_opt("ordering_gate_benign", meta.ordering_gate_benign, -1);
      if (meta.imu_drops >= 0)
        d.b("damaging", meta.imu_drops > 0 || meta.frame_drops_collect > 0 || meta.image_stall_collect > 0);
      d.close();
    }
    c.close();
  }

  // ---- bootstrap (hand-eye + time offset, per camera) ----
  root.key("bootstrap");
  emit_bootstrap(o, rep);

  // ---- gates ----
  root.key("gates");
  {
    Obj g(o);
    g.key("accel");
    {
      Obj a(o);
      a.s("verdict", accel_verdict_name(rep.a_wald_verdict));
      a.b("full_open", rep.a_full_open);
      a.n("att_spread_deg", rep.accel_att_spread_deg);
      a.n("dyn_ms2", rep.accel_dyn_ms2);
      a.key("wald");
      {
        Obj w(o);
        w.i("r", rep.a_wald_r);
        w.n("T", rep.a_wald_T);
        w.i("df", rep.a_wald_df);
        w.n("kappa", rep.a_wald_kappa);
        w.n("x12", rep.a_wald_x12);
        w.n("x21", rep.a_wald_x21);
        w.n("xthr1", rep.a_wald_xthr1);
        w.n("xthr2", rep.a_wald_xthr2);
        w.n("jqa_deg", rep.a_wald_jqa_deg);
        w.n("jda", rep.a_wald_jda);
        w.n("min_eig", rep.a_wald_min_eig);
        w.n("dqa_deg", rep.a_wald_dqa_deg);
        w.n("dda_off", rep.a_wald_dda_off);
        w.i("windows", rep.a_wald_windows);
        w.i("dropped", rep.a_wald_dropped);
        w.close();
      }
      a.close();
    }
    g.key("tg");
    {
      Obj t(o);
      t.s("verdict", accel_verdict_name(rep.tg_gate_verdict));
      t.b("open", rep.tg_open);
      t.n("conditional_sigma", rep.tg_conditional_sigma);
      t.close();
    }
    g.close();
  }

  // ---- per-block commit decisions ----
  root.key("blocks");
  emit_blocks(o, rep);

  // ---- posterior ----
  // labels/sigma/prior_sigma drive the "did we learn anything" bar chart;
  // Lambda drives the correlation heatmap and any 2x2 uncertainty ellipse.
  root.key("posterior");
  {
    Obj p(o);
    p.b("ok", rep.joint.ok);
    p.i("dim_p", rep.joint.dim_p);
    p.i("windows_used", rep.joint.windows_used);
    p.i("windows_dead", rep.joint.windows_dead);
    p.i("evaluation_passes", rep.joint.evaluation_passes);
    p.i("accepted_passes", rep.joint.accepted_passes);
    p.n("final_merit", rep.joint.final_merit);
    p.n("qn_max_final", rep.joint.qn_max_final);
    p.n("last_step_norm", rep.joint.last_step_norm);
    p.b("hit_wall_budget", rep.joint.hit_wall_budget);
    p.b("stopped_early", rep.joint.stopped_early);
    p.i("stop_pass", rep.joint.stop_pass);
    p.i("time_stops", rep.joint.time_stops);
    p.key("labels");
    {
      o += '[';
      for (size_t i = 0; i < rep.joint.labels.size(); i++) {
        if (i)
          o += ',';
        emit_str(o, rep.joint.labels[i]);
      }
      o += ']';
    }
    p.key("sigma");
    emit_eigen_vec(o, rep.joint.sigma);
    p.key("prior_sigma");
    emit_eigen_vec(o, rep.joint.prior_sigma_vec);
    if (rep.joint.Lambda.rows() > 0 && rep.joint.Lambda.rows() == rep.joint.Lambda.cols()) {
      p.key("lambda");
      Obj L(o);
      L.i("n", rep.joint.Lambda.rows());
      L.s("layout", "packed_upper_row_major");
      L.key("data");
      emit_sym_packed(o, rep.joint.Lambda);
      L.close();
    }
    p.close();
  }

  // ---- verify ----
  root.key("verify");
  {
    Obj v(o);
    v.n("holdout_cost_seed", rep.holdout_cost_seed);
    v.n("holdout_cost_committed", rep.holdout_cost_committed);
    v.n("holdout_cost_mixture", rep.holdout_cost_mixture);
    v.n("improve", rep.verify_improve);
    v.n("mixture_improve", rep.mixture_improve);
    v.i("windows_used", rep.verify_windows_used);
    v.i("windows_dropped", rep.verify_windows_dropped);
    v.b("small_n", rep.verify_small_n);
    v.key("per_window_ratio");
    emit_vecd(o, rep.verify_window_ratio);
    v.close();
  }

  // ---- per-stage evidence ----
  root.key("stages");
  emit_stages(o, rep);

  // ---- values ----
  // `committed` is what ships (uncommitted blocks reverted to seed);
  // `solved` is the raw post-VarPro point, diagnostics only. Both are emitted
  // because the delta between them is exactly what the gates refused.
  root.key("values");
  {
    Obj v(o);
    v.key("committed");
    emit_shared_calib(o, rep.committed);
    v.key("solved");
    emit_shared_calib(o, rep.solved);
    v.close();
  }

  // ---- provenance: the ONLY safe writeback filter ----
  root.key("provenance");
  {
    Obj pv(o);
    pv.key("committed_blocks");
    o += '[';
    {
      bool first = true;
      for (const auto &b : rep.blocks)
        if (b.committed) {
          if (!first)
            o += ',';
          first = false;
          emit_str(o, b.label());
        }
    }
    o += ']';
    pv.key("seed_blocks");
    o += '[';
    {
      bool first = true;
      for (const auto &b : rep.blocks)
        if (!b.committed) {
          if (!first)
            o += ',';
          first = false;
          emit_str(o, b.label());
        }
    }
    o += ']';
    pv.close();
  }

  root.close();
  o += '\n';
  return o;
}

bool write_report_json(const std::string &path, const SessionReport &rep, const ReportMeta &meta) {
  const std::string body = report_to_json(rep, meta);

  // Atomic: a reader (the portal polls this directory) must never see a
  // half-written report. Same tmp+rename discipline as the result YAML.
  const std::string tmp = path + ".tmp";
  FILE *f = std::fopen(tmp.c_str(), "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "[calib] WARNING: cannot open %s: %s\n", tmp.c_str(), std::strerror(errno));
    return false;
  }
  const size_t n = std::fwrite(body.data(), 1, body.size(), f);
  const bool short_write = (n != body.size());
  if (std::fclose(f) != 0 || short_write) {
    std::fprintf(stderr, "[calib] WARNING: short write on %s\n", tmp.c_str());
    std::remove(tmp.c_str());
    return false;
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::fprintf(stderr, "[calib] WARNING: cannot rename %s -> %s: %s\n", tmp.c_str(), path.c_str(),
                 std::strerror(errno));
    std::remove(tmp.c_str());
    return false;
  }
  return true;
}

} // namespace ov_zcalib
