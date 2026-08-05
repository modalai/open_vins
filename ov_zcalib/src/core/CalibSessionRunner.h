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
 * All per-window work (linear seed, display-Lambda micro-BA, reservoir
 * admission) runs SYNCHRONOUSLY on this thread: windows close every few
 * seconds and cost ~100-300 ms on target, so the duty cycle stays low, the
 * rings upstream absorb the burst, and — decisive — live and replay execute
 * the IDENTICAL deterministic computation (the S4 bit-parity gate extends to
 * the committed answer). The streaming one-pass fusion drives DISPLAY ONLY;
 * the committed calibration always comes from the end-of-session VarPro
 * (JointCalib) over the D-optimal window selection, then VERIFY on held-out
 * windows gates a PARTIAL commit (only blocks whose posterior beats the prior
 * by commit_sigma_factor; the rest stay at seed and are reported).
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
  int min_pair_matches = 12;
  /// Evidence recency horizon [s] for the xcorr/hand-eye buffers (0 = whole session). The gate
  /// judges the operator's RECENT motion: an early bad stretch (AE settling, blur, the pick-up)
  /// must age out instead of capping the achievable peak forever — measured on D0014
  /// (2026-07-13) the cumulative peak crawled 0.52->0.57 over a minute because the prefix never
  /// left the sum. Also bounds per-attempt cost and memory on a long bootstrap. Sessions whose
  /// bootstrap passes within the horizon are byte-identical to the unwindowed behaviour.
  double bootstrap_window_s = 45.0;
  // COLLECT
  /// Merge-replay the bootstrap-span frames+imu into the harvester once it
  /// exists (see try_bootstrap_). MEASURED 2026-07-10: transformative on
  /// starved flight logs (position_push front: 4->10 harvested, ABORT->COMMIT)
  /// but HARMFUL on rich handheld sessions (Sting: 2 early cold-start windows
  /// joined the fused set, flipped split-half INCONSISTENT at dqA 0.823 deg
  /// vs 0.305 band — freezing the accel chain — and dragged cam 1.4-1.8 px
  /// off kalibr). Default OFF preserves the validated rich-session path; the
  /// --flight profile turns it on.
  bool retro_harvest = false;
  double collect_max_s = 240.0;
  /// EARLY CUTOVER (live sessions): stop collecting as soon as the reservoir's
  /// D-optimal selection reaches this whitened min-eigenvalue AND holds enough
  /// windows to fuse + verify. 0 = disabled (collect the whole budget).
  ///
  /// Why an eigenvalue and not a window count: the weakest direction is what
  /// gates committability, and 20 near-duplicate windows can leave it as bare as
  /// 5. The one-pass Lambda used here UNDER-states the information the full
  /// nonlinear solve extracts (measured: sessions committing every block sat at
  /// min-eig 4.8-6.3), so this threshold is a STOPPING heuristic, deliberately
  /// below the theoretical commit-grade value (9 = the 3-sigma rule in whitened
  /// units) -- the real gates still adjudicate at solve time.
  double collect_min_eig = 0.0;
  double thermal_hold_slope = 1.5 / 60.0; ///< deg C/s: pause window opening above this
  // SOLVE / S5 camera staging
  int select_K = 18;
  double select_overlap_penalty = 0.5;
  /// Task 8: stage-specific D-optimal subsets over the retained reservoir.
  /// SELECTION-SIDE ONLY: admission fingerprints and reservoir retention are
  /// parity-frozen (WindowScorer untouched); holdouts never enter any set
  /// (thermal_bin excludes them). OFF = the single master selection feeds
  /// every stage — byte-identical legacy path.
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
  /// receives the remaining time as its max_wall_s (floored so late stages
  /// still make progress). JointCalib stops at its best accepted point when
  /// exceeded, so a tight budget degrades polish, never consistency. The
  /// per-call joint.max_wall_s remains available but is overridden when this
  /// is set — staging multiplied the call count, so only a shared deadline
  /// bounds the session (flight profiles set this to meet the <=60 s target).
  double solve_budget_s = 0.0;
  /// 0 fixed | 1 refine (tight priors from the existing cal — the DEFAULT per
  /// the plan) | 2 full (weak priors; gated, loud). Intrinsics unlock only in
  /// phase B AFTER temporal/IMU converge (RS/td residue aliases into k1/f
  /// otherwise) and ship only if the block beats its prior 3x AND the
  /// refinement-hurt detector stays quiet.
  int cam_mode = 1;
  double k34_radial_gate = 0.12; ///< min fraction of obs beyond 0.7*r_max to free k3/k4
  /// C1 (center gate): min per-quadrant fraction of this camera's fused observations, quadrants
  /// taken about the current (cx, cy). Below it the data never brackets the center and cx/cy walk
  /// into self-consistent junk (the v5 wander: cy +2.9 px COMMITTED); freeze them at seed via the
  /// k3/k4 sigma mechanism -- D2 exclusion keeps the block committable with frozen dofs at seed.
  double cam_center_quadrant_gate = 0.10;
  Eigen::Matrix<double, 8, 1> cam_refine_prior = (Eigen::Matrix<double, 8, 1>() << 2, 2, 2, 2, 0.01, 0.01, 1e-9, 1e-9).finished();
  Eigen::Matrix<double, 8, 1> cam_full_prior = (Eigen::Matrix<double, 8, 1>() << 20, 20, 20, 20, 0.1, 0.1, 1e-9, 1e-9).finished();
  /// Camera-block coordinate alternation inside phase B (operator directive):
  /// fx<->k1 correlate at |rho|~0.9 on equidistant, so a monolithic 8-dof pass
  /// walks the shared valley slowly and drifts the pinhole row. Each round runs
  /// (a) pinhole+k1/k2 with k3/k4 frozen, then (b) distortion-only k1..k4 with
  /// the pinhole row frozen (k3/k4 still behind the radial-coverage gate).
  /// 0 disables (monolithic B-1, the pre-directive behavior).
  /// Cam alternation rounds. SEED QUALITY IS THE MODERATOR (2026-07-11):
  /// on a well-seeded rig (Sting v8_alt1 A/B, per-unit cal reproj 0.31 px)
  /// one round reproduces the two-round kalibr scorecard exactly and saves
  /// ~30 s — but on existing-cal-grade seeds (S4 sim, 1.5 px off: the
  /// production cam-refine case) round 2 is load-bearing for the flattest
  /// dof (cy +0.70 vs +0.94 px, carry-free). Library default serves the
  /// general case; drop to 1 via --cam-alt-rounds when the rig's existing
  /// cal is known-good. (An alt-rounds saving measured UNDER the warm-carry
  /// is an artifact: carried basins made round 2 a no-op at any setting.)
  int cam_alt_rounds = 2;
  /// Run the pinhole-only settle pass after the alternation rounds. P0
  /// measured settle merit-flat alongside round 2; the round-2 A/B proved a
  /// byte-level no-op. Profiles may drop settle (~9 s host) after the same
  /// byte-level A/B on their shape.
  bool cam_settle = true;
  /// Arm the stationarity certificate in B2-polish (IMU chain open there, so
  /// JointConfig::cert_open_imu is required): replaces the legacy
  /// plateau/anchor double-solves P0 measured as ~85% waste in that stage.
  /// Off by default; profiles enable after their falsifier+scorecard A/B.
  bool b2_cert = false;
  /// s7 RE-BASELINE CANDIDATE (default off; --a-candidate CLI): the A-chain
  /// stages (A0/A1a/A1b-full) run the certificate instead of legacy
  /// plateau/anchor, and every staged solve arms the Newton-decrement
  /// conv-stop. The split HALVES stay legacy two-path unconditionally (they
  /// ARE the falsifier). This deliberately moves the A-chain statistics the
  /// legacy invariant pins — verdicts are adjudicated by the s7 protocol
  /// (suite verdicts + both logs + kalibr-gauge envelope), never assumed.
  bool a_candidate = false;
  /// P4 fused evaluation in every staged solve EXCEPT the split halves (the
  /// mode-0/2 falsifier keeps its legacy statistics in every configuration).
  /// At host mode-1 the operative falsifier is the wald gate, whose stability
  /// under the P4 solver is measured by test_wald_mc --p4 (composed endgame).
  bool p4 = false;
  /// A-chain warm-carry (mode-1 experiment): carry nuisance optima across
  /// A0->A1a->A1b. REJECTION HISTORY: carried seeds poisoned the SPLIT-HALF
  /// falsifier (S1 sim flip) -- which never runs at mode-1, where the wald
  /// gate + kalibr scoring adjudicate instead. B-chain carry stays OFF
  /// unconditionally (S4 cy flat-dof walk, measured).
  bool a_carry = false;
  // Accel-chain unlock (A1b): the full accel intrinsic chain (da off-diagonals
  /// + q_AtoI) needs specific-force DIRECTION diversity (attitude spread) to be
  /// identifiable at all — the cheap pre-gate below — but on real sensors the
  /// weakly-excited dofs also absorb unmodeled systematics (-1%-level scale
  /// junk measured on a handheld ICM while kalibr resolves 0.3-0.6%). The
  /// authoritative gate is therefore SPLIT-HALF CONSISTENCY: the chain is
  /// solved independently on the first and second time-half of the fused
  /// windows and unlocks only when both halves agree within their posteriors
  /// (time split, not interleaved: the junk mode tracks self-heating/drift).
  /// Diagonal da always solves (A1a).
  double a_full_att_gate_deg = 45.0;  ///< pre-gate: min pairwise angle between window gravity dirs (body frame)
  double a_full_dyn_gate = 0.2;       ///< pre-gate floor: mean within-window std of |a_m| [m/s^2] (near-static guard)
  int a_full_min_windows = 6;         ///< pre-gate: enough fused windows for two meaningful halves
  double a_split_sigma_k = 3.0;       ///< half-agreement band: k * sqrt(sig1^2 + sig2^2) per dof
  double a_split_da_floor = 3e-3;     ///< band floor for da dofs: agreement is only demanded at the scale of
                                      ///< the accuracy CLAIM (~0.3%); tighter floors freeze chains whose halves
                                      ///< agree 8x better than the suite's own documented da valley tolerance
  double a_split_qa_floor_deg = 0.1;  ///< band floor for the q_AtoI angle
  /// Band floor for tg dofs [(rad/s)/(m/s^2)]: agreement is demanded at the scale of the accuracy
  /// CLAIM — the a_split_da_floor doctrine — i.e. 0.3x the MEASURED 4e-4 part class (D0014 chain
  /// 0.22 dps/g), the per-element bound the recovery gate and the commit ceiling ship. The old
  /// 1e-4 sat below the claim and let the falsifier's own noise adjudicate: measured on the T1
  /// synthetic (real all-9-distinct part-class Tg), the halves' worst |d| held at 1.1-1.2e-4 from
  /// 8 to 12 windows/half — inside the claim, refused by the floor. Kalibr's BETWEEN-session
  /// scatter (+/-5e-4) stays 4x above this floor: scatter-class junk still refuses.
  double a_split_tg_floor = 1.2e-4;
  /// Signal-fraction term of the agreement band: halves also agree when their
  /// disagreement is below this fraction of the SIGNAL they claim (deviation
  /// from the frozen A1a value). Scale-free falsifier: exported posteriors are
  /// precision-only and under-disperse (measured ~2x on the synthetic suite),
  /// so a pure sigma band wrongly freezes a well-observed chain; junk modes
  /// have scatter ~ signal and still fail this ratio.
  double a_split_signal_frac = 0.34;
  // ---- Wald reduced-information accel gate (DESIGNS #1 / S11): replaces the
  // two nonlinear half-solves with ONE widened warm evaluation pass at the
  // A1a accepted point + marginal Wald / cross-prediction / observability
  // statistics on the 6-dof gate subspace {da off-diag, qA}. Sequencing
  // theorem (2026-07-11 adjudication): the half-solves' launch sensitivity is
  // what pins A1a's pass count — this gate dissolves that constraint.
  int a_gate_mode = 0;           ///< 0 split-half decides (v15-exact); 1 wald decides; 2 shadow (split decides, wald logged)
  double a_info_deflate = 2.0;   ///< kappa: measured exported-Lambda under-dispersion, VARIANCE semantics (MC harness re-pins per shape)
  double a_obs_min_eig = 4.5;    ///< per-half prior-whitened eigenvalue floor along joint eigendirections
  double a_wald_thresh_scale = 1.0;
  double a_qa_phys_ceiling_deg = 2.0;  ///< fused-step ceiling (vs 0.776 deg die misalignment + ICM cross-axis spec class)
  double a_da_off_phys_ceiling = 0.02;
  /// Fused-step ceiling for tg elements. ANCHORED TO THE MEASURED PART CLASS, not the datasheet
  /// typ: the D0014 chain carries |Tg| ~ 4e-4 (0.22 dps/g) and kalibr's own per-session estimates
  /// scatter +/-5e-4 -- a blind (zero-seeded) session recovering the REAL value implies a fused
  /// step of ~4-6e-4, which the old 5e-4 ceiling refused as "unphysical" (measured: front log
  /// refused at 5.9e-4). 3x the part class = 1.5e-3: junk basins still hit it, physics does not.
  double a_tg_phys_ceiling = 1.5e-3;
  // Identifiability gates (diagnostics with AUTHORITY; previously computed-but-unused)
  double xcorr_min_peak = 0.6;      ///< normalized xcorr peak floor: a flat correlation ridge
                                    ///< (near-constant |w|) yields a noisy td seed that must not pass
  double xcorr_min_sharpness = 0.0; ///< curvature floor at the xcorr peak (0 = off; grid-scale dependent)
  double min_eig_floor = 1.0;       ///< whitened min-eig ABORT floor on the selected window set
                                    ///< (catastrophic-only: committability needs ~(commit_sigma_factor)^2 per dof)
  double min_window_parallax = 0.0; ///< admission floor on the fingerprint median parallax [px]
                                    ///< (0 = off; enable on bench data — far-field/translation-free guard)
  // VERIFY / COMMIT
  double verify_min_improve = 0.05;  ///< held-out cost must improve by >= 5%
  /// Small-n honesty floors: a +5% improvement on ONE holdout window is weak
  /// evidence (measured flight sessions sit exactly there). With n_hold <= 2
  /// the improvement floor rises (n=1 -> n1, n=2 -> n2); the profile answer
  /// to a starved session is MORE holdouts (min_holdout), not a softer gate.
  double verify_min_improve_n1 = 0.15;
  double verify_min_improve_n2 = 0.10;
  /// Retro-holdout top-up target: force_holdout weakest-information retained
  /// windows until n_hold >= min(min_holdout, clamp(N_ret/4,1,3)) while
  /// keeping N_fused >= max(2, N_ret-3). 1 = legacy single retro-designation.
  int min_holdout = 1;
  double tr_hw_tol = 0.15;           ///< |tr - tr_hw_seed| / tr_hw_seed gate (when both exist)
  double commit_sigma_factor = 3.0;  ///< block commits only if 3*sigma_post < sigma_prior for all dofs
  /// A block must be DISTINGUISHABLE from the value it would otherwise ship (the
  /// revert point: its seed) before it may claim a calibration. The 3-sigma rule
  /// above is PRECISION-only — it asks "is the posterior tight?", never "did the
  /// data actually move this block?" — and a block the solver never stepped still
  /// accumulates information at the linearization point, so its posterior
  /// collapses and it COMMITS AT ITS SEED. Measured on D0014 (2026-07-12): a
  /// budget-truncated A1a took zero accepted steps, and dw/da were reported
  /// `COMMIT` while carrying identity — which a writeback would then have written
  /// over the rig's real factory Dw. The bar here is deliberately low (did it move
  /// at all, in posterior units), not a significance test: it fires only on the
  /// pathology, and every converged block clears it by orders of magnitude.
  double commit_min_move_sigma = 0.1;
  /// Absolute posterior ceilings [local units] per block: commit additionally
  /// requires sigma_post <= ceiling on every non-frozen dof. This ties the
  /// commit rule to the acceptance targets (the 3x-prior rule alone admits
  /// sigmas 3-5x looser than the flight acceptance numbers). tg's ceiling is the
  /// CLAIM scale — 0.3x the measured 4e-4 part class, the same 1.2e-4 the split
  /// floor demands agreement at (a_split_tg_floor doctrine): a pair certified at
  /// claim-scale agreement must not then be refused for claim-scale precision
  /// (the old 1.0e-4 pin predated the floor re-pin and sat below the claim).
  std::map<std::string, double> commit_abs_ceiling = {{"q_ItoC", 1.7e-3}, {"p_IinC", 2.5e-3}, {"td", 2.5e-4}, {"tg", 1.2e-4}};
  /// q_ItoC and td commit/revert TOGETHER: they move jointly in the solve, and
  /// a mixed state (new rotation, seed td) can be worse than either endpoint.
  bool commit_atomic_rot_td = true;
  /// da and q_AtoI commit/revert together when the full accel chain was
  /// unlocked (A1b): the off-diagonals and the accel-frame rotation trade
  /// against each other, and neither the solve nor the split-half falsifier
  /// validated a mixed state.
  bool commit_atomic_accel = true;
  /// Leave-one-block-out holdout deltas for committed blocks (accuracy-side
  /// falsifier; costs a few extra held-out window solves).
  bool commit_attribution = true;
  bool free_tr = false;              ///< estimate rolling-shutter readout
  /// C3 tr gates (rolling cams). Row-coverage floor: std of observed u_frac over the fused set
  /// must clear this before tr stays free (uniform rows = 0.289; below it, tr aliases td and the
  /// session would earn a junk readout while biasing td -- the D10 mechanism, refused up front).
  double tr_row_cov_floor = 0.20;
  /// td-aliasing ceiling at commit: |rho(td,tr)| from the joint posterior above this refuses the
  /// tr commit (the pair is one dof wearing two names; ship the seed, keep td honest).
  double tr_td_corr_ceiling = 0.95;
  /// Estimate Tg (gyro g-sensitivity, 9 dof). EARNED, never trusted from a chain: the rig's own
  /// kalibr sessions scatter beyond the value's magnitude (measured 2026-07-13, |Tg(D1)-Tg(F3)|
  /// at the size of Tg itself), so a seed is an init and the session must certify its own
  /// estimate. Unlocks ONLY through the A1b full-chain gate (accel excitation pre-gate +
  /// split-half / Wald falsifier) and commits through the standard machinery + its ceiling.
  /// Requires an estimable IMU chain (a frozen factory chain freezes tg with it).
  bool free_tg = true;
  std::string out_yaml = "ov_zcalib_result.yaml";
  bool verbose = true;
  /// P0 evidence: print the per-stage cost table ([evidence] lines) at the end
  /// of the session solve. Counters are always collected (they are cheap and
  /// ship in SessionReport::evidence); this only gates the print.
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

/// P0 evidence: one row per staged JointCalib solve, plus aggregated rows for
/// the collection-side admission BAs and the verify sweep. Where a row comes
/// from a JointCalib call the fields mirror JointReport; for BA-family rows
/// passes/accepted count solves attempted/succeeded and the warm/cold columns
/// stay zero. Timing fields are summed thread-CPU (> wall when parallel).
struct StageEvidence {
  std::string label;
  int passes = 0;
  int accepted = 0;
  int windows = 0, dim_p = 0;
  double wall_s = 0, seed_s = 0, preint_s = 0, inner_s = 0, export_s = 0;
  long iters = 0, warm = 0, cold = 0, cold_plateau = 0, cold_anchor = 0, cold_won = 0, cold_won_guard = 0;
  long cold_cert = 0, cold_jump = 0; ///< P1 certificate / carry-jump duels
  long phit = 0, pmiss = 0;   ///< P3 preint-cache hits / misses (window solves)
  double factor_s = 0.0;      ///< P3: IMU-factor construction thread-CPU (was untimed)
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
  // collection
  int windows_harvested = 0, windows_retained = 0, windows_holdout = 0, windows_rejected_seed = 0, windows_invalidated = 0;
  int windows_rejected_gate = 0; ///< pre-seed admission gates (parallax floor etc.)
  int windows_rejected_ba = 0;   ///< admission BA failures (previously a silent drop)
  int windows_probation = 0;         ///< retained via the drift-budget envelope (probation)
  int windows_probation_dropped = 0; ///< probation windows failing the post-A0 strict re-check
  bool verify_small_n = false;       ///< VERIFY decided under the small-n floors (n_hold <= 2)
  std::vector<double> verify_window_ratio; ///< per-holdout paired improvement of the mixture
  // solve
  int windows_fused = 0;
  double min_eig_whitened = 0.0;
  // Task 8 stage-specific selection (0/false when stage_select is off)
  int windows_a0 = 0, windows_a1 = 0, windows_b = 0;
  double min_eig_a0 = 0.0, min_eig_a1 = 0.0, min_eig_b = 0.0;
  bool stage_a1_fallback = false; ///< balance guard reverted the A1 family to the master set
  // accel-chain excitation gate telemetry (A1b decision)
  double accel_att_spread_deg = 0.0; ///< max pairwise angle between window gravity dirs
  double accel_dyn_ms2 = 0.0;        ///< mean within-window std of |a_m|
  bool a_full_open = false;          ///< full accel chain (da off-diag + q_AtoI) unlocked
  // Wald gate verdict + statistics (modes 1/2; PRE_CLOSED when the cheap
  // pre-gate never admitted the question)
  enum class AccelGateVerdict { PRE_CLOSED, SPLIT_CONSISTENT, SPLIT_INCONSISTENT, SPLIT_FAILED, WALD_CONSISTENT, WALD_INCONSISTENT, WALD_UNOBSERVABLE };
  AccelGateVerdict a_wald_verdict = AccelGateVerdict::PRE_CLOSED;
  /// tg's OWN gate verdict. Mode 1: the wald tg-subspace judge (runs only when the chain
  /// certifies). Split modes: the tg half pair — which also runs when the frozen-tg chain judge
  /// REFUSED, because a certified-reproducible Tg falsifies that judge's tg=seed conditioning and
  /// arbitrates a re-judge on the tg-free halves. PRE_CLOSED = the question was never admitted
  /// (accel pre-gate closed, no solve, or the session does not estimate tg). A CONSISTENT verdict
  /// with tg_open=false means tg reproduced but the chain never certified (tg opens only WITH it).
  AccelGateVerdict tg_gate_verdict = AccelGateVerdict::PRE_CLOSED;
  bool tg_open = false; ///< tg unlocked WITH the chain and survived A1b (the commit machinery still gates the block)
  int a_wald_r = 0;                  ///< observable gate-subspace dimension (of 6)
  double a_wald_T = 0.0;             ///< correlated Wald statistic (chi^2_r under H0)
  double a_wald_x12 = 0.0, a_wald_x21 = 0.0; ///< cross-prediction excesses
  /// Cross-prediction thresholds AT THE RUN CONFIG. X12/X21 are kappa-free, but their
  /// Satterthwaite thresholds are exactly proportional to kap_eff * a_wald_thresh_scale
  /// (AC = kap_eff * A*(A1^-1+A2^-1): gfac scales, nu is invariant) — recorded so the MC
  /// harness can re-size the rule offline across (floor, scale) cells without re-solving.
  double a_wald_xthr1 = 0.0, a_wald_xthr2 = 0.0;
  /// Fused-step physical magnitudes (the phys-ceiling inputs). Config-invariant across
  /// (kappa, thresh_scale): the offline re-sizer needs them to replay the ceiling branch.
  double a_wald_jqa_deg = 0.0, a_wald_jda = 0.0;
  double a_wald_min_eig = 0.0;       ///< min-half whitened eig along the weakest joint direction
  double a_wald_dqa_deg = 0.0;       ///< implied half-disagreement rotation angle
  double a_wald_dda_off = 0.0;       ///< implied half-disagreement max |da_offdiag|
  double a_wald_kappa = 0.0;         ///< per-session dispersion estimate (quarter-scatter method of moments)
  int a_wald_df = 0;                 ///< its degrees of freedom (J-1)*r
  int a_wald_windows = 0, a_wald_dropped = 0;
  JointReport joint;
  // verify (all candidates evaluated on the IDENTICAL held-out window set)
  double holdout_cost_seed = 0.0, holdout_cost_committed = 0.0;
  double verify_improve = 0.0;
  double holdout_cost_mixture = 0.0; ///< cost of the partially-committed mixture (== committed when nothing reverted)
  double mixture_improve = 0.0;
  int verify_windows_used = 0;    ///< holdout windows where EVERY candidate solved
  int verify_windows_dropped = 0; ///< holdout windows dropped symmetrically (any candidate failed)
  bool tr_hw_ok = true;
  /// Session-mean camera exposure [s], PER CAMERA. The committed td is in the MID-EXPOSURE
  /// convention (clones stamped at the optical instant); consumers that stamp frames at
  /// start-of-exposure (the raw MPA/kalibr convention) must use td + mean_exposure_s/2. Written to
  /// the YAML as an explicit field pair. Per camera because both terms are: cameras run their own
  /// auto-exposure and have their own td.
  std::vector<double> mean_exposure_s;
  // timings (wall clock)
  double t_solve_s = 0.0, t_verify_s = 0.0, t_total_s = 0.0;
  // commit
  std::vector<BlockCommit> blocks;
  SharedCalib committed; ///< final calibration (uncommitted blocks reverted to seed)
  SharedCalib solved;    ///< raw post-VarPro calibration BEFORE verify/commit gating (diagnostics)
  std::string prompt;    ///< last guided-excitation prompt (live UX)
  // P0 evidence rows (admission BAs, every staged solve, verify sweep)
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

  RunnerState state() const { return state_; }
  const SessionReport &report() const { return rep_; }

  /// True once the session has entered COLLECT (bootstrap accepted). Safe to read from OTHER
  /// threads (relaxed atomic): the server's ingest decimation keys on it — the hand-eye pair
  /// engine needs the full sensor rate (halving it doubles per-pair KLT displacement; measured
  /// D0014 2026-07-13: cam 0 xcorr peak 0.2 at 30 Hz vs 0.55 at 60 Hz, same rig and operator),
  /// while the saturation the decimation prevents is a COLLECT-phase problem (window
  /// destruction). Never unset: THERMAL_HOLD and the solve states are all post-bootstrap.
  bool bootstrap_done() const { return past_bootstrap_.load(std::memory_order_relaxed); }

  /// Live collection-sufficiency probe: "if I stopped collecting RIGHT NOW, how
  /// much information would the solve actually have?"
  ///
  /// A live stream has no EOF, so something must decide when to stop. Stopping
  /// on a CLOCK collects until the budget expires regardless of whether the data
  /// became informative 20 s ago (or never will). This runs the SAME D-optimal
  /// selection the solve will run, over the reservoir as it currently stands,
  /// and reports its whitened min-eigenvalue (E-optimality: the WEAKEST
  /// calibration direction). The driver stops when the weakest direction is
  /// already well determined and there are enough windows to fuse AND verify.
  ///
  /// This only decides WHEN TO STOP COLLECTING. Every accuracy gate (seed gates,
  /// excitation gates, VERIFY, commit ceilings) still runs afterwards, unchanged
  /// -- so an early cutover can cost polish, never correctness.
  struct CollectStatus {
    int n_retained = 0;   ///< windows in the reservoir
    int n_holdout = 0;    ///< of those, reserved for VERIFY
    int n_fusable = 0;    ///< of those, with a valid export (the solve's candidates)
    double min_eig = 0.0; ///< whitened min-eig of the D-optimal selection over them
    bool ready = false;   ///< enough windows AND min_eig >= cfg.collect_min_eig
  };
  CollectStatus collect_status();
  /// Per-block seed patches for a replay: which blocks start from the rig's own
  /// chain instead of blind. The recorded CAMERA seed and the streams always stay
  /// authoritative — this exists so ONE recorded session can be scored across the
  /// seeding matrix (what earns its value from the data vs what should be given).
  /// Patch a recorded seed so ONE recorded session can be scored across the seeding matrix (blind
  /// vs chain-seeded, per block). The IMU chain is shared; the per-camera fields apply to `cam`.
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
  void enter_(RunnerState s, const char *why);
  void try_bootstrap_(double now);
  void handle_window_(WindowData &&w, const WindowMeta &m);
  void solve_verify_commit_();
  double temp_slope_() const;
  void note_stage_(const std::string &label, const JointReport &r);
  void print_evidence_() const;
  /// The Wald gate's one-evaluation pass (modes 1/2). Returns the ACCEL-chain verdict and fills
  /// the a_wald_* report fields; when tg_verdict is given (and the session estimates tg), the
  /// SAME window-evaluation pass is judged a second time on the tg subspace — an unidentifiable
  /// tg must freeze tg alone, never veto the chain. budget_left_s <= 0 => safe abstention.
  SessionReport::AccelGateVerdict wald_accel_gate_(const std::vector<WindowData> &fused, std::vector<WindowWarmState> &warm,
                                                   double budget_left_s, SessionReport::AccelGateVerdict *tg_verdict = nullptr);
  /// Task 8 stage selection: greedy D-optimal (WindowScorer::select_logdet)
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

  // mean exposure, PER CAMERA (td convention bridge; see SessionReport)
  std::vector<double> exp_sum_;
  std::vector<long> exp_n_;
  // P0 evidence: collection-side admission BA aggregate (one row at table time)
  StageEvidence adm_ev_;
  // P3 session preint store (PreintCache.h): pure value-keyed memoization —
  // hits return the exact bytes recomputation would produce (replay-proven),
  // so it is A-CHAIN-LEGACY-INVARIANT-compatible and threads through EVERY
  // window solve: admission BAs pre-fill at the seed pi (== the A0/A1a entry
  // key), the B-1 family shares one preint per window across its five solves,
  // split halves use disjoint windows, VERIFY's shared-pi candidates collapse.
  PreintStore store_;
  // Retroactive harvest: frames buffered during BOOTSTRAP are merge-replayed
  // (with boot_imu_) into the harvester the moment it exists — on a 41 s
  // flight log SETTLE+BOOTSTRAP otherwise eat ~54% of the collection span.
  // RESOURCE CONTRACT (operator directive): FrameObs is the POST-TRACKING
  // point set (id,u,v — 12 B/point, ~2 KB/frame, no image payload). The
  // tracker ran ONCE per frame in the feeder; replay is harvester bookkeeping
  // only — NO re-detection, NO KLT re-run, ~8 MB worst-case at the 4000 cap.
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
