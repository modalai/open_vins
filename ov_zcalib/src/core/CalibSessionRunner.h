/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: session orchestrator (SETTLE -> BOOTSTRAP -> COLLECT/THERMAL_HOLD
 * -> SOLVE_REFINE -> VERIFY -> COMMIT).
 *
 * Live and replay share this object through the push interface (feed_imu /
 * feed_frame / finish): on-device the session thread pops the feeder's SPSC
 * rings and pushes here; on host the replay pump pushes the recorded streams.
 * Per-window seeding, information export and reservoir admission run on this
 * caller thread. Live and replay use the same stages, but matching input alone
 * does not guarantee identical results: overrides, deadlines and numerical
 * execution also matter. Streaming fusion is for collection diagnostics.
 * The staged JointCalib solves supply the accepted calibration/posterior;
 * held-out verification and per-block rules decide the committed mixture.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_CALIB_SESSION_RUNNER_H
#define OV_ZCALIB_CALIB_SESSION_RUNNER_H

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "../init/HandEyeWahba.h"
#include "../init/RelRotProcrustes.h"
#include "../init/TimeOffsetInit.h"
#include "../init/EpipolarTimeInit.h"
#include "../utils/SessionRecord.h"
#include "../solve/JointCalib.h"
#include "../window/LinearSeed.h"
#include "../window/WindowHarvester.h"
#include "../window/WindowScorer.h"
#include "CalibSession.h"

namespace ov_zcalib {

enum class RunnerState { SETTLE, BOOTSTRAP, COLLECT, THERMAL_HOLD, SOLVE_REFINE, VERIFY, COMMIT, DONE, ABORT };

struct SessionConfig {
  HarvesterConfig harvester;
  ScorerConfig scorer;
  JointConfig joint;
  HandEyeConfig handeye;
  LinearSeedConfig seed;
  // SETTLE
  double settle_still_w = 0.03;      ///< rad/s rolling std counted as still
  double settle_min_still_s = 2.0;   ///< quiet span for the bias/temp baseline
  double settle_timeout_s = 60.0;
  double settle_max_temp_slope = 0.5 / 60.0; ///< deg C/s (pre-warm rule)
  // BOOTSTRAP
  double bootstrap_min_span_s = 12.0;
  int bootstrap_min_pairs = 150;
  double bootstrap_timeout_s = 90.0;
  double td_search_s = 0.08;
  bool bootstrap_epipolar = true; ///< geometrically refine time-unstable xcorr seeds before harvest
  int min_pair_matches = 12;
  /// Bootstrap xcorr/hand-eye history [s]; zero retains the whole session.
  /// Older motion ages out of both the evidence and per-attempt workspace.
  double bootstrap_window_s = 45.0;
  // COLLECT
  /// Replay buffered bootstrap IMU/frames into the newly created harvester.
  /// This changes the admitted window set; the flight overlay enables it.
  bool retro_harvest = false;
  double collect_max_s = 240.0;
  /// EARLY CUTOVER (live sessions): stop collecting as soon as the reservoir's
  /// D-optimal selection reaches this whitened min-eigenvalue AND holds enough
  /// windows to fuse + verify. Estimated Tg must also reach its marginal
  /// precision target; a generic min-eigenvalue of 5 does not certify Tg's
  /// absolute ceiling. 0 = disabled (collect the whole budget).
  /// This is a collection stopping heuristic, not a commit or accuracy verdict;
  /// correlated or repeated windows need not add independent information.
  double collect_min_eig = 0.0;
  double thermal_hold_slope = 1.5 / 60.0; ///< deg C/s: pause window opening above this
  // SOLVE / camera staging
  int select_K = 18;
  double select_overlap_penalty = 0.5;
  /// Optional stage-specific D-optimal subsets of the retained reservoir.
  /// Admission and reservoir retention are unchanged; holdouts stay excluded.
  /// When disabled, every stage receives the master selection.
  bool stage_select = false;
  int select_K_a0 = 0; ///< A0 (ext/td/tr) budget; 0 = select_K
  int select_K_a1 = 0; ///< A1a/A1b + accel/tg gates budget; 0 = select_K (keep >= a_full_min_windows)
  int select_K_b = 0;  ///< phase-B budget; 0 = select_K
  double stage_feat_gain = 1.0; ///< weight of the stage feature Gram vs the Fisher sub-block
  /// Fall back to the master set for the A1 family when S_A1's time-halves are
  /// direction-starved (the split/wald machinery quarters BY TIME; a
  /// direction-optimal subset clustered in time would fail a certifiable chain).
  bool stage_a1_balance_guard = true;
  /// Session-wide SOLVE budget [s] (0 = unlimited). One deadline shared by
  /// EVERY staged JointCalib call (A0/A1a/split-halves/A1b/B passes): each call
  /// receives the remaining time as its max_wall_s; exhausted stages are skipped.
  /// JointCalib reserves measured pass time and cancels incomplete candidates
  /// between window operations, preserving its last complete posterior.
  /// Indivisible factorizations and scheduling delays remain a soft-bound
  /// limitation. Truncated half-solves never certify a gate. The
  /// per-call joint.max_wall_s remains available but is overridden when this
  /// is set -- staging multiplied the call count, so only a shared deadline
  /// applies the requested solve budget to the whole session.
  double solve_budget_s = 0.0;
  /// 0 fixed | 1 refine (tight priors from the existing cal -- the default)
  /// | 2 full (weak priors; gated, loud). Intrinsics unlock only in
  /// phase B AFTER temporal/IMU converge (RS/td residue aliases into k1/f
  /// otherwise) and ship only if the block beats its prior 3x AND the
  /// refinement-hurt detector stays quiet.
  int cam_mode = 1;
  double radtan_tangent_refine_sigma = 0.001; ///< p1/p2, independent of radial k1/k2 units
  double radtan_tangent_full_sigma = 0.01;
  double k34_radial_gate = 0.12; ///< min fraction of obs beyond 0.7*r_max to free k3/k4
  /// Minimum per-quadrant fraction about this camera's current center.
  /// Below it, constrain cx/cy to their seed while judging the remaining dofs.
  double cam_center_quadrant_gate = 0.10;
  Eigen::Matrix<double, 8, 1> cam_refine_prior = (Eigen::Matrix<double, 8, 1>() << 2, 2, 2, 2, 0.01, 0.01, 1e-9, 1e-9).finished();
  Eigen::Matrix<double, 8, 1> cam_full_prior = (Eigen::Matrix<double, 8, 1>() << 20, 20, 20, 20, 0.1, 0.1, 1e-9, 1e-9).finished();
  /// Camera coordinate rounds in phase B: pinhole+k1/k2 with the final
  /// distortion pair held, then distortion-only with pinhole held. Coverage
  /// gates remain active. Zero selects one monolithic camera-block solve.
  /// Round count changes the reached solution and must be evaluated per rig.
  int cam_alt_rounds = 2;
  /// Run a pinhole-only settle solve after camera coordinate alternation.
  bool cam_settle = true;
  /// Permit the local stationarity heuristic in B2 joint polish when the
  /// IMU chain is open. It does not replace final verification or commit gates.
  bool b2_cert = false;
  /// Experimental outer Newton-decrement stopping for A0/A1a/A1b and the
  /// B-stage configurations. This does not enable an A-stage certificate.
  /// Split halves keep their separate stopping and arbitration settings.
  bool a_candidate = false;
  /// Use capped fused evaluations for staged solves except split halves.
  /// Changes to the optimization path can change gate statistics; enabling
  /// this does not establish either null sizing or model-error rejection.
  bool p4 = false;
  /// Historical carry experiment request. Current session A-stage policy
  /// explicitly disables carry after reading it; this flag does not enable it.
  bool a_carry = false;
  /// Full accel-chain fitting needs attitude and dynamic excitation before
  /// the consistency gate runs. Split modes independently fit the temporal
  /// halves; mode 1 instead uses the local Wald test. Agreement under either
  /// model is not independent accuracy validation. A1a fits diagonal Da.
  double a_full_att_gate_deg = 45.0;  ///< pre-gate: min pairwise angle between window gravity dirs (body frame)
  double a_full_dyn_gate = 0.2;       ///< pre-gate floor: mean within-window std of |a_m| [m/s^2] (near-static guard)
  int a_full_min_windows = 6;         ///< pre-gate: enough fused windows for two meaningful halves
  double a_split_sigma_k = 3.0;       ///< half-agreement band: k * sqrt(sig1^2 + sig2^2) per dof
  double a_split_da_floor = 3e-3;     ///< historical absolute agreement floor for Da dofs
  double a_split_qa_floor_deg = 0.1;  ///< band floor for the q_AtoI angle
  /// Absolute split-half difference floor for Tg [(rad/s)/(m/s^2)]. Historical
  /// policy (commit 3e95384): 0.3 times a 4e-4 ICM reference scale, supported by
  /// synthetic half differences around 1.1-1.2e-4. This is not a BMI270 accuracy
  /// validation. A difference floor and a one-sigma posterior ceiling describe
  /// different statistics even when their numeric values happen to match.
  double a_split_tg_floor = 1.2e-4;
  /// Agreement band also includes this fraction of the halves' departure
  /// from their shared entry. It supplements raw local-posterior bands;
  /// time-stable model errors can still pass a consistency test.
  double a_split_signal_frac = 0.34;
  // Local Wald gate: widen one warm evaluation at the accepted A1a point,
  // then form marginal contrast, quadratic cross-prediction and information
  // checks for the six accel-chain dofs. Tg has its own nine-dof subspace.
  // Mode 1 is experimental authority; supported profiles use mode 2.
  int a_gate_mode = 0;           ///< 0 split decides; 1 Wald decides; 2 split decides with Wald diagnostics
  double a_info_deflate = 2.0;   ///< covariance inflation floor; H0 sizing alone does not validate H1 rejection
  double a_obs_min_eig = 4.5;    ///< per-half prior-whitened eigenvalue floor over the entire tested span, after kappa deflation
  double a_wald_thresh_scale = 1.0;
  double a_qa_phys_ceiling_deg = 2.0;  ///< historical fused-step guard relative to entry; not an absolute sensor limit
  double a_da_off_phys_ceiling = 0.02;
  /// Historical fused-step guard for Tg elements relative to the entry value,
  /// motivated by an ICM reference chain and between-session scatter. This is
  /// an acceptance heuristic, not an absolute physical bound for every IMU.
  double a_tg_phys_ceiling = 1.5e-3;
  // Identifiability gates (diagnostics with AUTHORITY)
  double xcorr_min_peak = 0.6;      ///< normalized xcorr peak floor: a flat correlation ridge
                                    ///< (near-constant |w|) yields a noisy td seed that must not pass
  double xcorr_min_sharpness = 0.0; ///< curvature floor at the xcorr peak (0 = off; grid-scale dependent)
  double min_eig_floor = 1.0;       ///< whitened min-eig ABORT floor on the selected window set
                                    ///< (catastrophic-only: committability needs ~(commit_sigma_factor)^2 per dof)
  double min_window_parallax = 0.0; ///< admission floor on median first-to-last bearing angle [rad]
                                    ///< (0 = off; enable on bench data -- far-field/translation-free guard)
  // VERIFY / COMMIT
  double verify_min_improve = 0.05;  ///< held-out cost must improve by >= 5%
  /// Require larger held-out improvement when only one or two windows are
  /// available. Extra repeated observations are not independent validation.
  double verify_min_improve_n1 = 0.15;
  double verify_min_improve_n2 = 0.10;
  /// Retro-holdout top-up target: force_holdout weakest-information retained
  /// windows until n_hold >= min(min_holdout, clamp(N_ret/4,1,3)) while
  /// keeping N_fused >= max(2, N_ret-3). 1 = legacy single retro-designation.
  int min_holdout = 1;
  double commit_sigma_factor = 3.0;  ///< block commits only if 3*sigma_post < sigma_prior for all dofs
  /// Minimum movement from the seed in posterior-sigma units. This prevents
  /// local information alone from labeling an unmoved block as estimated.
  /// It is a small movement guard, not a significance or accuracy test.
  double commit_min_move_sigma = 0.1;
  /// Absolute posterior ceilings [local units] per block: commit additionally
  /// requires sigma_post <= ceiling on every non-frozen dof. This ties the
  /// commit rule to separate absolute precision targets. Tg's
  /// 1.2e-4 ceiling has the ICM/synthetic provenance above; it is a local one-sigma
  /// precision policy, not an empirically calibrated accuracy interval or a
  /// BMI270-specific limit. Passing the split-half floor does not imply passing
  /// this separate precision test.
  std::map<std::string, double> commit_abs_ceiling = {{"q_ItoC", 1.7e-3}, {"p_IinC", 2.5e-3}, {"td", 2.5e-4}, {"tg", 1.2e-4}};
  /// q_ItoC and td commit/revert TOGETHER: they move jointly in the solve, and
  /// a mixed state (new rotation, seed td) can be worse than either endpoint.
  bool commit_atomic_rot_td = true;
  /// da and q_AtoI commit/revert together when the full accel chain was
  /// unlocked (A1b): the off-diagonals and the accel-frame rotation trade
  /// against each other, and neither the solve nor the split-half falsifier
  /// validated a mixed state.
  bool commit_atomic_accel = true;
  /// Leave-one-block-out held-out cost deltas for committed blocks; these
  /// measure predictive contribution, not error against physical truth.
  bool commit_attribution = true;
  // Readout is a supplied fixed value (camN_readout_time_s -> seed CamCalib::tr),
  // used by reprojection timing. This session neither estimates nor validates it.
  /// Estimate nine-dof gyro g-sensitivity when the IMU chain is estimable.
  /// Tg may move as an A1a nuisance but must pass the full-chain/Tg gates and
  /// the commit rules before shipping. Seeding Tg does not certify its value.
  bool free_tg = true;
  bool tg_precision_screen = true; ///< skip split Tg solves when A1a conditional precision already misses commit
  std::string out_yaml = "ov_zcalib_result.yaml";
  bool verbose = true;
  /// Print the per-stage cost table ([evidence] lines) at the end of the
  /// session solve. Counters are always collected (they are cheap and ship in
  /// SessionReport::evidence); this only gates the print.
  bool evidence_table = true;
};

struct BlockCommit {
  std::string name; ///< the PHYSICAL block name ("td"); see SharedCalib::BlockRef::name
  int cam = -1;     ///< -1 = shared (the IMU chain); otherwise the camera this block belongs to
  /// Report/provenance identity, unique across cameras ("td@1"). This is what the writeback's
  /// committed_blocks / seed_blocks lists carry.
  std::string label() const { return cam < 0 ? name : name + "@" + std::to_string(cam); }
  bool committed = false;
  double worst_ratio = 0.0;  ///< max over NON-FROZEN dofs of 3*sigma_post/sigma_prior (<1 commits)
  double worst_sigma = 0.0;  ///< max posterior sigma over non-frozen dofs [local units]
  bool ceiling_ok = true;    ///< worst_sigma <= commit_abs_ceiling (when configured for the block)
  double moved_sigma = 0.0;  ///< max |x_solved - x_seed| / sigma_post over non-frozen dofs (0 = the solve never moved it)
  bool not_estimated = false; ///< refused because the solve never moved it off its seed (see commit_min_move_sigma)
  bool atomic_reverted = false; ///< reverted only because its atomic partner failed
  double holdout_delta = 0.0;   ///< leave-one-out: holdout cost with this block reverted minus committed-mixture cost (>0 = block helps)
};

/// Evidence table row: one per staged JointCalib solve, plus aggregated rows
/// for the collection-side admission BAs and the verify sweep. Where a row
/// comes from a JointCalib call the fields mirror JointReport; for BA-family
/// rows passes/accepted count solves attempted/succeeded and the warm/cold
/// columns stay zero. Timing fields are summed thread-CPU (> wall when parallel).
struct StageEvidence {
  std::string label;
  int passes = 0;
  int accepted = 0;
  int windows = 0, dim_p = 0;
  double wall_s = 0, seed_s = 0, preint_s = 0, inner_s = 0, export_s = 0;
  double max_pass_s = 0.0; ///< observed cost used to admit later staged passes
  long iters = 0, warm = 0, cold = 0, cold_plateau = 0, cold_anchor = 0, cold_won = 0, cold_won_guard = 0;
  long cold_cert = 0, cold_jump = 0; ///< certificate / carry-jump duels
  long phit = 0, pmiss = 0;   ///< preint-cache hits / misses (window solves)
  double factor_s = 0.0;      ///< IMU-factor construction thread-CPU
  long tstop = 0;             ///< inner solves ended by the wall hang-guard (0 = healthy;
                              ///< nonzero = load leaked into numerics, run tainted for A/B)
  double merit = 0.0, qn_max = 0.0;
  int stop_pass = -1; ///< pass where the early-stop fired (-1 = ran full)
  bool hit_budget = false;
  long rss_kb = 0; ///< VmRSS after the stage (0 if unavailable)
};

struct SessionReport {
  RunnerState final_state = RunnerState::ABORT;
  std::string abort_reason;
  // bootstrap, per camera (each has its own hand-eye and its own time offset)
  std::vector<HandEyeResult> handeye;
  std::vector<TimeOffsetResult> xcorr;
  std::vector<EpipolarTimeResult> epipolar_time;
  /// Per camera: the coarse peak passed the trim/interleaved gate.
  std::vector<char> xcorr_certified;
  /// Effective coarse-peak floor and split tolerance for report consumers.
  double xcorr_min_peak = 0.0;
  double td_fine_range_s = 0.0;
  /// Raw lag curves captured once at bootstrap (81 samples at default settings).
  std::vector<XcorrCurve> xcorr_curve;
  // collection
  int windows_harvested = 0, windows_retained = 0, windows_holdout = 0, windows_rejected_seed = 0, windows_invalidated = 0;
  int windows_rejected_gate = 0; ///< pre-seed admission gates (parallax floor etc.)
  int windows_rejected_ba = 0;   ///< admission BA failures (counted, never a silent drop)
  int windows_probation = 0;         ///< retained via the drift-budget envelope (probation)
  int windows_probation_dropped = 0; ///< probation windows failing the post-A0 strict re-check
  bool verify_small_n = false;       ///< VERIFY decided under the small-n floors (n_hold <= 2)
  std::vector<double> verify_window_ratio; ///< per-holdout paired improvement of the mixture
  // solve
  int windows_fused = 0;
  double min_eig_whitened = 0.0;
  // stage-specific selection (0/false when stage_select is off)
  int windows_a0 = 0, windows_a1 = 0, windows_b = 0;
  double min_eig_a0 = 0.0, min_eig_a1 = 0.0, min_eig_b = 0.0;
  bool stage_a1_fallback = false; ///< balance guard reverted the A1 family to the master set
  // accel-chain excitation gate telemetry (A1b decision)
  double accel_att_spread_deg = 0.0; ///< max pairwise angle between window gravity dirs
  double accel_dyn_ms2 = 0.0;        ///< mean within-window std of |a_m|
  bool a_full_open = false;          ///< full accel chain (da off-diag + q_AtoI) unlocked
  // Wald gate verdict + statistics (modes 1/2; PRE_CLOSED when the cheap
  // pre-gate never admitted the question)
  // WALD_UNOBSERVABLE is the legacy wire name for failure to certify under
  // configured information/numerical/budget checks, not a proof of algebraic rank loss.
  enum class AccelGateVerdict { PRE_CLOSED, SPLIT_CONSISTENT, SPLIT_INCONSISTENT, SPLIT_FAILED, WALD_CONSISTENT, WALD_INCONSISTENT, WALD_UNOBSERVABLE, PRECISION_WEAK };
  AccelGateVerdict a_wald_verdict = AccelGateVerdict::PRE_CLOSED;
  /// Tg's separate verdict. Mode 1 tests its local Wald subspace only after
  /// the accel chain passes. Split modes use a Tg-free pair initialized at A0;
  /// if Tg agrees, those halves can also re-judge a rejected chain that was
  /// conditioned on A1a's common nuisance Tg. A consistent Tg verdict alone
  /// does not unlock or commit Tg: the chain and later checks must also pass.
  AccelGateVerdict tg_gate_verdict = AccelGateVerdict::PRE_CLOSED;
  double tg_conditional_sigma = 0.0; ///< optimistic A1a Tg sigma before the full-chain split solves
  bool tg_open = false; ///< tg unlocked WITH the chain and survived A1b (the commit machinery still gates the block)
  int a_wald_r = 0;                  ///< pooled-basis candidate dimension (of 6); the whole-span floor is also required
  double a_wald_T = 0.0;             ///< dispersion-scaled local Wald statistic; finite-df sizing when estimated
  double a_wald_x12 = 0.0, a_wald_x21 = 0.0; ///< cross-prediction excesses
  /// Cross-prediction thresholds AT THE RUN CONFIG. X12/X21 are kappa-free, but their
  /// Satterthwaite thresholds are exactly proportional to kap_eff * a_wald_thresh_scale
  /// (AC = kap_eff * A*(A1^-1+A2^-1): gfac scales, nu is invariant) -- recorded so the MC
  /// harness can re-size the rule offline across (floor, scale) cells without re-solving.
  double a_wald_xthr1 = 0.0, a_wald_xthr2 = 0.0;
  /// Fused-step physical magnitudes (the phys-ceiling inputs). Config-invariant across
  /// (kappa, thresh_scale): the offline re-sizer needs them to replay the ceiling branch.
  double a_wald_jqa_deg = 0.0, a_wald_jda = 0.0;
  double a_wald_min_eig = 0.0;       ///< min-half whitened information bound used by the gate, after kappa deflation
  double a_wald_dqa_deg = 0.0;       ///< implied half-disagreement rotation angle
  double a_wald_dda_off = 0.0;       ///< implied half-disagreement max |da_offdiag|
  double a_wald_kappa = 0.0;         ///< per-session dispersion estimate (quarter-scatter method of moments)
  int a_wald_df = 0;                 ///< within-half scatter degrees of freedom: sum_h (n_h-1)*r
  int a_wald_windows = 0, a_wald_dropped = 0;
  JointReport joint;
  // verify (all candidates evaluated on the IDENTICAL held-out window set)
  double holdout_cost_seed = 0.0, holdout_cost_committed = 0.0;
  double verify_improve = 0.0;
  double holdout_cost_mixture = 0.0; ///< cost of the partially-committed mixture (== committed when nothing reverted)
  double mixture_improve = 0.0;
  int verify_windows_used = 0;    ///< holdout windows where seed/solution/mixture solved
  int verify_windows_dropped = 0; ///< paired windows dropped for a required candidate failure
  /// Session-mean camera exposure [s], PER CAMERA -- DIAGNOSTIC ONLY. Frame timestamps arrive
  /// already anchored at center-row mid-exposure (the producer applies SOF + (readout+exposure)/2
  /// at ingest), so the committed td needs NO exposure conversion for any consumer: every stamp in
  /// the system is in the same convention. Kept in the report/YAML because the session's exposure
  /// range is real evidence about the record (AE behavior, lighting).
  std::vector<double> mean_exposure_s;
  /// Effective reprojection measurement standard deviations [px], PER CAMERA. Resolved once
  /// from camera overrides or the harvester's scalar fallback; these are weights, not accuracy.
  std::vector<double> camera_pixel_sigmas;
  // timings (wall clock)
  double t_solve_s = 0.0, t_verify_s = 0.0, t_total_s = 0.0;
  // commit
  std::vector<BlockCommit> blocks;
  SharedCalib committed; ///< final calibration (uncommitted blocks reverted to seed)
  SharedCalib solved;    ///< raw post-VarPro calibration BEFORE verify/commit gating (diagnostics)
  std::string prompt;    ///< last guided-excitation prompt (live UX)
  // evidence rows (admission BAs, every staged solve, verify sweep)
  std::vector<StageEvidence> evidence;
};

class CalibSessionRunner {
public:
  CalibSessionRunner(const SessionConfig &cfg, const SessionSeed &seed);

  /// Push interface (live session thread or replay pump).
  void feed_imu(const RawImu &s);
  void feed_frame(const FrameObs &f);

  /// End of stream: run SOLVE_REFINE -> VERIFY -> COMMIT. Returns final report.
  const SessionReport &finish();

  /// Stop before solving/publishing when the source or mandatory record failed.
  void abort(const std::string &reason) { enter_(RunnerState::ABORT, reason.c_str()); }

  RunnerState state() const { return state_; }
  const SessionReport &report() const { return rep_; }

  /// True after the first COLLECT entry; never unset. Ingest threads may read
  /// this relaxed atomic to enable post-bootstrap decimation. It does not
  /// synchronize access to the report or other runner state.
  bool bootstrap_done() const { return past_bootstrap_.load(std::memory_order_relaxed); }

  /// Information-based collection screen over eligible retained windows.
  /// Reports the D-optimal selection's whitened minimum eigenvalue and the
  /// marginal Tg precision screen. Readiness is a stopping heuristic; it does
  /// not predict the final nonlinear posterior or establish physical accuracy.
  /// Excitation, verification and commit checks still run after collection.
  struct CollectStatus {
    int n_retained = 0;   ///< windows in the reservoir
    int n_holdout = 0;    ///< of those, reserved for VERIFY
    int n_fusable = 0;    ///< valid exports in the solve's eligible thermal bin
    double min_eig = 0.0; ///< whitened min-eig of the D-optimal selection over them
    double tg_sigma = -1.0; ///< worst marginal Tg sigma; -1 = unavailable
    double tg_sigma_target = 0.0; ///< precision needed for early cutover; 0 = Tg not estimated
    bool ready = false;   ///< enough windows, min-eig target AND estimated-Tg precision
  };
  CollectStatus collect_status();
  /// Per-block seed patches for a replay: which blocks start from the rig's own
  /// chain instead of blind, so ONE recorded session can be scored across the
  /// seeding matrix (what must be earned from the data vs what should be given).
  /// The recorded CAMERA seed and the streams always stay authoritative. The IMU
  /// chain is shared; the per-camera fields apply to `cam`.
  struct SeedOverride {
    const ImuIntrinsicModel *imu = nullptr; ///< dw/da/q_AtoI (+ their calib_* freeze flags)
    /// g-sensitivity, imu2 gauge (already conjugated: Tg_r = Tg_chain * Q_w). SEPARATE from `imu`
    /// on purpose -- Tg and the dw/da chain are independent seeding axes, and patching one must not
    /// silently reset the other to the model default.
    const Eigen::Matrix3d *Tg = nullptr;
    /// Calibration-WEIGHTING densities. The sigma ratio between the IMU and the cameras is what
    /// decides which one wins an argument about dw, so it is a first-class axis of the seeding
    /// matrix -- not a detail. Recorded sessions carry whatever the producer weighted at.
    const ImuNoise *noise = nullptr;
    int cam = 0; ///< which camera the extrinsic/td overrides below apply to
    const Eigen::Vector4d *q_ItoC = nullptr;
    const Eigen::Vector3d *p_IinC = nullptr;
    const double *td = nullptr;
  };

  /// Convenience: full replay run from a session record.
  static bool run_replay(const std::string &record_path, const SessionConfig &cfg, SessionReport &out,
                         const SeedOverride *seed_override = nullptr);

private:
  friend struct CalibCollectionTestAccess; ///< deterministic information-only collection regressions
  void enter_(RunnerState s, const char *why);
  void try_bootstrap_(double now);
  void handle_window_(WindowData &&w, const WindowMeta &m);
  /// One eligibility rule for collection screens, prompts and the final D-optimal solve pool.
  std::vector<int> retained_solve_candidates_();
  void refresh_collection_guidance_();
  void solve_verify_commit_();
  double temp_slope_() const;
  void note_stage_(const std::string &label, const JointReport &r);
  void print_evidence_() const;
  /// The Wald gate's one-evaluation pass (modes 1/2). Returns the ACCEL-chain verdict and fills
  /// the a_wald_* report fields; when tg_verdict is given (and the session estimates tg), the
  /// SAME window-evaluation pass is judged a second time on the tg subspace -- an unidentifiable
  /// tg must freeze tg alone, never veto the chain. budget_left_s <= 0 => safe abstention.
  SessionReport::AccelGateVerdict wald_accel_gate_(const std::vector<WindowData> &fused, std::vector<WindowWarmState> &warm,
                                                   double budget_left_s, SessionReport::AccelGateVerdict *tg_verdict = nullptr);
  /// Stage selection: greedy D-optimal (WindowScorer::select_logdet)
  /// over blkdiag( Lw[dof_idx,dof_idx], stage_feat_gain * g g^T ) per
  /// candidate. Deterministic (fixed candidate order, no RNG). Empty result =
  /// caller falls back to the master set.
  std::vector<int> select_stage_(const char *tag, const std::vector<Eigen::MatrixXd> &Lw,
                                 const std::vector<std::pair<double, double>> &spans, const std::vector<int> &dof_idx,
                                 const std::vector<Eigen::VectorXd> &feat, int K, double *min_eig_out) const;

  SessionConfig cfg_;
  SessionSeed seed0_;    ///< pre-bootstrap seed (the record-header seed)
  SharedCalib calib_;    ///< working calibration (bootstrap updates it)
  int n_cams_ = 1;       ///< cameras in this session (fixed at construction; SharedCalib::cams size)
  RunnerState state_ = RunnerState::SETTLE;
  SessionReport rep_;

  // mean exposure, PER CAMERA (diagnostic only; see SessionReport::mean_exposure_s)
  std::vector<double> exp_sum_;
  std::vector<long> exp_n_;
  // collection-side admission BA aggregate (one evidence row at table time)
  StageEvidence adm_ev_;
  // Value-keyed preintegration reuse across admission, staged solves and
  // verification. Split halves access disjoint existing window slots; any
  // storage growth must finish before concurrent calls begin.
  PreintStore store_;
  // Optional bootstrap replay into the newly created harvester uses tracked
  // observations and boot_imu_; it does not repeat image tracking. The frame
  // deque has a 4000-entry cap; payload memory also depends on points per frame.
  std::deque<FrameObs> boot_frames_;
  std::vector<char> probation_; ///< per-slot: retained via the drift envelope
  // SETTLE
  double first_t_ = -1.0, still_since_ = -1.0;
  bool have_baseline_ = false;
  Eigen::Vector3d bg0_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d still_gyro_sum_ = Eigen::Vector3d::Zero();
  int still_gyro_n_ = 0;

  // rolling excitation/temperature (settle + thermal-hold)
  std::deque<RawImu> roll_;
  Eigen::Vector3d rw_ = Eigen::Vector3d::Zero(), rw2_ = Eigen::Vector3d::Zero();

  // BOOTSTRAP accumulation. Rotation pairs and camera-rate samples are PER CAMERA: two cameras with
  // no shared field of view share no feature id, so a pair can only ever be formed within one
  // camera -- and each camera has its own R_ItoC and its own td to recover from them. The IMU
  // stream and the gyro bias are shared, because there is one IMU.
  std::vector<std::vector<HandEyePair>> pairs_;
  std::vector<std::vector<CamRateSample>> rates_;
  std::vector<RawImu> boot_imu_;
  std::vector<FrameObs> prev_frame_;
  std::vector<char> have_prev_frame_;
  double boot_t0_ = -1.0;
  double last_boot_try_ = -1e9; ///< retry throttle (xcorr + hand-eye are not per-frame cheap)
  // Pair-yield accounting, PER CAMERA (cumulative): names the bottleneck in the reject line.
  // thin = consecutive-frame id matches below min_pair_matches (KLT dying between frames);
  // relrot = RelRotEssential returned !ok (geometry unfittable); eiv = errors-in-variables gate
  // dropped the pair (translation flow rivals |theta|). Healthy yield + a low peak points at
  // per-pair noise (blur, RS) rather than tracking starvation.
  std::vector<long> boot_thin_, boot_relrot_, boot_eiv_, boot_pairs_ok_;
  std::atomic<bool> past_bootstrap_{false}; ///< set on COLLECT entry; read by ingest threads

  // COLLECT
  std::unique_ptr<WindowHarvester> harvester_;
  std::unique_ptr<WindowScorer> scorer_;
  std::vector<WindowData> slots_;          ///< reservoir storage (scorer slot -> window)
  std::vector<WindowSolveReport> slot_rep_; ///< display-Lambda per slot
  std::unique_ptr<CalibSession> display_;
  double last_display_t_ = -1.0;
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_CALIB_SESSION_RUNNER_H
