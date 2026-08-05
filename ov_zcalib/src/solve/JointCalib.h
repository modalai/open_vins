/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: cross-window VarPro fusion on the shared calibration block.
 * Windows are conditionally independent given p, so information ADDS exactly:
 * each outer iteration re-preintegrates and re-solves every window at the current
 * p (WindowBA), sums the exported (Lambda_w, g_w), applies the global seed priors
 * ONCE, solves the small dense step (sum Lambda + Lambda_prior) dp = -(sum g +
 * g_prior), and retracts p on its manifolds. The one-pass fused estimate is
 * display-only by contract; the committed answer is the converged outer loop.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_JOINT_CALIB_H
#define OV_ZCALIB_JOINT_CALIB_H

#include <map>
#include <string>
#include <vector>

#include "../window/LinearSeed.h"
#include "WindowBA.h"

namespace ov_zcalib {

struct JointConfig {
  int outer_iterations = 12; ///< ACCEPTED outer steps (damped steps are small; see lm_lambda)
  int window_max_iters = 30;
  int max_backtracks = 6; ///< consecutive damped retries before stopping at the best point
  bool verbose = true;
  // ---- P1 stationarity certificate (replaces the plateau/anchor cold-solve
  // triggers when on; use_cert=false + use_carry=false + early_stop=false is
  // bit-identical legacy behavior for A/B). The certificate accepts a warm
  // result WITHOUT the duplicate fresh-seed solve when the inner solve exited
  // at genuine stationarity AND the remaining nuisance Newton decrement q_n is
  // negligible against the window cost and its own accepted reference.
  // P0-measured motivation: cam-only stages double-solved ~85% of window
  // evals (plateau trigger) for 0.05-0.34% cost noise; A-stage cold solves
  // stay reachable through the strand guard + q_n growth + jump duels.
  bool use_cert = true;
  /// Extend the certificate to solves with the IMU chain OPEN (calib_da /
  /// calib_RAtoI free — the B2-polish class). The stage-aware guard exists
  /// because the cert's q_n bands were validated with the chain frozen; B2
  /// therefore ran LEGACY plateau/anchor, which P0 measured as ~85%
  /// double-solves for 0.05-0.34% cost noise (plat 273/24 passes on the v14
  /// Sting run). Off by default — a profile enables it only after its own
  /// falsifier + scorecard + S4 A/B (the qn statistic polices exactly the
  /// stale-gradient case the plateau over-approximates, and the flat-cam-dof
  /// discipline P0 worried about is owned by the alternation anchor + the
  /// 3-sigma cam sanity gate).
  bool cert_open_imu = false;
  double cert_qn_rel = 2e-6;    ///< q_n ceiling as a fraction of window cost (1% of the 1e-4 acceptance band)
  double cert_ref_growth = 2.0; ///< q_n may grow at most this factor over its accepted reference
  double cert_agree_rel = 1e-3; ///< dual-pass agreement band that refreshes the q_n reference
  /// Certificate needs enough windows for per-window basin noise to average
  /// out of the fused step (the duels it removes were also BASIN arbitration:
  /// keep-cheaper across two inits). Measured: a 3-window p_IinC probe drifts
  /// 4.0 -> 4.8 mm under the certificate while 16-window stages IMPROVE; and
  /// small-N solves are cheap, so legacy duels there cost ~nothing.
  int cert_min_windows = 6;
  // ---- P1 scaled outer early-stop: stop after stop_k consecutive accepted
  // steps that are stable (post-cap whitened step, relative merit change,
  // lambda level) + one deterministic cold stop-confirmation pass.
  // DEFAULT OFF by evidence: never fired on Sting real data (zero runtime
  // rent), and its one measured firing (S4 sim B1-settle, pass 11/24) cut the
  // flat-dof cam polish short — cy landed +1.03 px vs the 0.9 gate, a
  // ~0.01 px/pass creep the step/merit stability tests cannot see (q_n
  // certifies NUISANCE stationarity, not flat-calib-dof progress). The
  // confirmation passes also burned ~85 duels per cam session. Re-enable only
  // with per-log-class evidence.
  bool early_stop = false;
  int stop_k = 2;
  // ---- Convergence-terminated outers (the A-chain candidate program's
  // enabler; SOLVER_TIME_METHOD s10a). Stop when the OUTER Newton decrement
  // lambda^2 = g^T (Lambda+Pi)^-1 g at the accepted point stays below
  // conv_tol_rel * max(merit, 1) for conv_k consecutive accepted steps (after
  // conv_min_accepts). Scale-free and per-dof-blind in the right way: the
  // decrement IS the predicted attainable merit reduction, so flat-dof creep
  // that still buys merit keeps iterating (the S4 case the retired
  // merit-watching early_stop cut short), while churn below the acceptance
  // band's own noise floor stops (the B2 case). Default OFF: pass counts
  // change everywhere it is armed -> falsifier re-pin class (s7).
  bool conv_stop = false;
  int conv_min_accepts = 3;
  int conv_k = 2;
  double conv_tol_rel = 1e-5;
  // ---- P4 fused evaluation (SOLVER_TIME_METHOD s4, scoped variant): after
  // fused_warmup_passes, warm-path evaluations run ONE inner iteration and
  // export -- the exported gred = gk - Hkn Hnn^-1 gn is the nuisance-corrected
  // joint-Newton reduced gradient (first-order exact off-optimum, the
  // Problem.cpp export contract), so the outer's damped step drives p and the
  // single inner step at the NEXT eval is the z back-substitution. Cold paths
  // and the first pass stay FULL solves (basin escapes + honest entry);
  // plateau/anchor triggers are structurally meaningless at capped evals and
  // are disabled under the flag (strand/warmfail/jump duels remain). A
  // commit-grade FINALIZE pass (tight solves + exports at the final accepted
  // point) rebuilds the reported linearization before return -- committed
  // posteriors never come from a capped eval. Validation: MC wald-verdict
  // stability + both-logs operating point (the mode-1 composed endgame).
  bool fused_schur = false;
  int fused_warmup_passes = 1;
  /// Inner iterations for capped warm evals. 1 = pure back-substitution proxy;
  /// it lacks the -Hnn^-1 Hnp dp feedforward (the p-step's effect on z), so in
  /// p-moving stages the warm path loses cold duels (A0 measured: cold wins
  /// 140/156). 2 gives the composed step a second Newton correction.
  int fused_iters = 1;
  /// Defer QUALITY duels (strand/cert/plateau/anchor/jump on healthy warm
  /// evals) until the candidate is ACCEPTED on warm-only merit: a rejected
  /// candidate's duel results are discarded entirely, so ~half of the
  /// designed two-path cost was spent arbitrating points that never ship
  /// (measured: A0 cold 154/304 evals in every variant). RESCUE duels
  /// (no-warm first evals, warm failures) stay inline — they are the only
  /// path for those windows. Trade, named: a candidate rejected on warm-only
  /// merit could have been accepted after duel improvements; it is re-seeded
  /// at the next pass instead. Off by default; mode-1 profiles arm it after
  /// wald + kalibr adjudication.
  bool duel_on_accept = false;
  /// Last N ACCEPTED steps run UNCAPPED (full inner solves): the fused
  /// trajectory re-converges toward the legacy point before the stage exits.
  /// Measured need: the capped A-chain lands a shifted point (p z +1.7 mm)
  /// whose different B entry costs +38 s downstream — polish only the stage
  /// whose output feeds B (A1b-full), not the gate-validated interior stages.
  int fused_polish_accepts = 0;
  /// Export-on-accept (t7 / dossier R4): window evaluations run COST-ONLY —
  /// no reduced-information export — and after a candidate is ACCEPTED on
  /// merit, ONE export pass produces the (Lambda, g, qn) the outer consumes,
  /// at the unchanged kept optima. Rejected-pass exports were write-only
  /// dead state (measured: export ~21% of window thread-CPU x ~41-45% of
  /// passes rejected), and duel LOSERS exported too when only the winner's
  /// bytes could ever be folded. Two sources at accept, both legacy bits:
  ///   * cert-on stages keep path A's INLINE export (the P1 certificate
  ///     consumes wrA.qn before the accept decision, and a calib-column-free
  ///     "qn-only" statistic was MEASURED 1-ulp off — removing the kept
  ///     columns changes H's leading dimension, the SIMD head-peel under
  ///     -ffast-math reassociates, H_zz drifts ~1 ulp and q_n ~1e-10 rel;
  ///     see Problem::ExportReducedInformation's note. q_n feeds thresholded
  ///     duel arbitration, so it must be the export's exact bits). Kept-A
  ///     windows there REUSE the stored eval export; the eoa savings on
  ///     cert stages are the path-B and duel-loser exports.
  ///   * everything else re-enters via WindowBA state_at: transport
  ///     linearization w_clone/v_clone and gauge anchors re-pinned from the
  ///     KEPT path's entry context (warm strand for kept-A, the window's
  ///     [possibly re-seeded] seeds for kept-B), state overridden to the
  ///     kept optimum, zero inner iterations — byte-equal to the inline
  ///     export legacy computed at eval time (unit-pinned: W1/W2 in
  ///     test_export_parity).
  /// Arbitration inputs and order are UNCHANGED: costs, duels, counters and
  /// the accepted-point linearization are byte-identical ON vs OFF (within-
  /// binary replay parity on the oracle front + down-flight records; see the
  /// t7 branch derivation). Named residual: a window whose UNDAMPED export
  /// system is non-PD at an eval is discovered by legacy AT the eval (okA/okB
  /// false -> duel/dead/veto) but by eoa only at the accept-time export on
  /// non-cert paths — the window then dies at the accepted point (loud
  /// warning) and the accepted merit is re-derived over the survivors in
  /// legacy accumulation order. Not observed on either oracle record.
  /// COMPOSITION: duel_on_accept DISARMS this flag (legacy inline exports
  /// run) — the deferred-duel fold is incremental (Lsum += d_L - L on a
  /// winner), so legacy's accepted linearization carries the LOSER's export
  /// in its rounding; a rebuilt direct sum is last-ulp different, and exact
  /// parity would need the loser exported too — the very work eoa elides.
  /// J4 in test_export_parity pins the composition.
  bool export_on_accept = true;
  double stop_step_winf = 0.01;  ///< ||dp/prior_sigma||_inf below this = stable step
  double stop_merit_rel = 3e-4;  ///< relative merit change below this = stable
  double stop_lambda_max = 1e-2; ///< lambda must be at/below this (not climbing a wall)
  /// Consume/produce a JointWarmCarry when the session provides one.
  /// DEFAULT OFF by evidence (both sims, isolated by A/B with cert held on):
  /// warm-carrying nuisance states across staged free-set boundaries anchors
  /// each sub-problem in its predecessor's basin — the warm strand WINS the
  /// entry duel on cost while sitting in a subtly biased basin, so cost
  /// arbitration cannot police it. S1 sim: post-A1a state under carry flips
  /// the split-half falsifier (dqA 0.551/0.265 vs 0.235 carry-free). S4 sim:
  /// B-chain carry walks the flattest cam dof (cy +1.03 px vs +0.70
  /// carry-free, gate 0.9) at ANY cert/early-stop/alt-rounds setting. Its
  /// duel count is neutral ('first' -> 'jump' one-for-one), so the runtime
  /// rent is inner-iteration savings only — not worth a basin bias. Full
  /// consume stays sound (and available) for SAME-shape re-solves:
  /// p_stamp + noise_stamp + layout_sig witnessed.
  bool use_carry = false;
  /// Seeder config for the per-evaluation re-seeds. The SESSION must thread its
  /// (possibly bootstrap-adapted) LinearSeedConfig here — the historical default
  /// silently dropped bias_presolve and the widened no-still-baseline gates in
  /// exactly the stages that produce the committed answer.
  LinearSeedConfig seed;
  /// Worker threads for the per-window evaluation loop. Windows are independent
  /// given p and results are reduced in fixed window order, so serial ==
  /// parallel BIT-IDENTICAL (see ov_init::zbft_sfm::ParallelExecutor). <=1 runs
  /// inline (no threads created) — the RT default for on-target flight profiles.
  int num_threads = 4;
  /// Wall-clock budget for ONE solve() call [s]; 0 = unlimited. When exceeded
  /// the loop stops at the best accepted point (never mid-evaluation), so the
  /// report stays consistent. Flight profiles set this to meet the <=60 s
  /// session target (collection included).
  double max_wall_s = 0.0;
  /// Absolute per-dof per-outer step caps: the ACI3 mean correction and the
  /// temporal transport are FIRST-ORDER in dp, so an outer step must stay
  /// inside first-order validity regardless of how confident the fused
  /// information is (re-preintegration refreshes between outers). These caps
  /// bind together with the whitened 3-prior-sigma trust region (min of both).
  /// tg cap/prior scales: ICM-class g-sensitivity elements sit at ~1e-4-5e-4 (rad/s)/(m/s^2)
  /// (D0014 chain 0.23 deg/s @1g; the Sting kalibr sessions scatter +/-5e-4 between runs). The
  /// 2e-4 step cap is half the part-class scale (first-order validity, mirrors dw's ratio); the
  /// 1e-3 prior admits any physical value at <1 sigma from a blind (zero) start.
  std::map<std::string, double> step_cap = {{"dw", 5e-3},     {"da", 5e-3},  {"q_AtoI", 5e-3}, {"q_ItoC", 0.01},
                                            {"p_IinC", 0.01}, {"td", 1e-3},  {"tr", 5e-4},     {"cam", 1.0},
                                            {"tg", 2e-4}};
  /// Global seed priors (1-sigma), applied ONCE at fusion. Group name -> sigma.
  std::map<std::string, double> prior_sigma = {{"dw", 0.02},    {"da", 0.02}, {"q_AtoI", 0.02}, {"q_ItoC", 0.05},
                                               {"p_IinC", 0.05}, {"td", 0.005}, {"tr", 0.005},  {"cam", 2.0},
                                               {"tg", 1e-3}};
  /// Per-dof camera prior override [fx fy cx cy k1 k2 k3 k4] (S5 staging: refine
  /// mode = tight priors; a ~1e-9 sigma is an information-level FREEZE, which is
  /// how the radial-coverage gate holds k3/k4 without a per-dof constancy API).
  /// PER CAMERA: which intrinsic dofs this stage opens, and how tightly. Per camera because the
  /// decision is data-driven -- the k3/k4 radial-coverage gate asks whether THIS camera actually
  /// saw the image corners, and one camera having coverage says nothing about another's.
  bool use_cam_prior_vec = false;
  std::vector<Eigen::Matrix<double, 8, 1>> cam_prior_vec;
  /// Default intrinsic prior [fx fy cx cy k1 k2 k3 k4]: k3/k4 information-frozen until the radial
  /// gate opens them.
  static Eigen::Matrix<double, 8, 1> default_cam_prior() {
    return (Eigen::Matrix<double, 8, 1>() << 2, 2, 2, 2, 0.01, 0.01, 1e-9, 1e-9).finished();
  }
  /// Per-dof da prior override [d11 d12 d22 d13 d23 d33] (upper-tri packing).
  /// The accel intrinsic chain splits by conditioning: the scale DIAGONAL is
  /// driven by gravity magnitude across attitudes, while the off-diagonals and
  /// q_AtoI need genuine dynamic excitation and otherwise absorb bias/gravity
  /// residue (measured: -1%-level junk on handheld rigs while kalibr resolves
  /// 0.3-0.6%). The session's A1a stage information-freezes the off-diagonals
  /// (sigma ~1e-9) unless the excitation gate opens the full chain (A1b).
  bool use_da_prior_vec = false;
  Eigen::Matrix<double, 6, 1> da_prior_vec = (Eigen::Matrix<double, 6, 1>() << 0.02, 1e-9, 0.02, 1e-9, 1e-9, 0.02).finished();
  /// Anchor the FREE cam dofs' prior at an explicit center instead of the
  /// solve-entry values. Sequential cam sub-solves (B-1 alternation, settle,
  /// B-2 polish) otherwise re-center the prior at every entry — a random-walk
  /// prior that legitimizes block-coordinate drift instead of arbitrating it;
  /// the anchor gives the whole phase ONE prior budget. Information-frozen
  /// dofs (sigma <= 1e-8) still center at entry: a freeze must HOLD the
  /// current value, never yank it back to the anchor mid-alternation.
  /// One anchor PER CAMERA (each has its own factory intrinsics, so there is no shared center to
  /// anchor at). The prior MASK above stays single: which intrinsic dofs the alternation opens is
  /// a policy about the model, identical for every camera that has one.
  bool use_cam_prior_center = false;
  std::vector<Eigen::Matrix<double, 8, 1>> cam_prior_center;
};

struct JointReport {
  bool ok = false;
  int windows_used = 0;
  Eigen::VectorXd sigma;             ///< posterior 1-sigma per local dof (block order)
  Eigen::VectorXd prior_sigma_vec;   ///< matching prior sigmas (for improvement ratios)
  std::vector<std::string> labels;   ///< per local dof
  Eigen::MatrixXd Lambda;            ///< fused information (whitened checks downstream)
  double last_step_norm = 0.0;
  int evaluation_passes = 0;         ///< evaluations spent (re-seed+solve+export sweeps)
  int windows_dead = 0;              ///< windows dropped at an accepted point (never candidates)
  double wall_s = 0.0;               ///< wall clock of this solve() call
  bool hit_wall_budget = false;      ///< stopped by max_wall_s (best accepted point shipped)
  // summed thread-CPU split across all window evaluations (> wall_s when parallel)
  double t_seed_sum = 0.0, t_preint_sum = 0.0, t_inner_sum = 0.0, t_export_sum = 0.0;
  long inner_iters_sum = 0;
  // P0 evidence: warm/cold two-path counters. A cold run is the fresh-seed
  // path B; its cause is attributed by priority first > warmfail > strand >
  // plateau > anchor, so cold_anchor counts PURE periodic re-anchors and
  // cold_plateau counts plateau fires that were not otherwise suspect — the
  // two triggers a P1 stationarity certificate would replace. cold_won_guard
  // says how often those triggered solves actually beat the warm result.
  long warm_evals = 0;           ///< path-A (warm-start) window solves run
  long cold_evals = 0;           ///< path-B (fresh-seed) window solves run
  long cold_first = 0;           ///< cold cause: no warm state yet (first evaluation)
  long cold_warmfail = 0;        ///< cold cause: warm solve failed outright
  long cold_strand = 0;          ///< cold cause: warm cost stranded above accepted*(1+guard)
  long cold_plateau = 0;         ///< cold cause: plateau trigger (legacy path, use_cert=false)
  long cold_anchor = 0;          ///< cold cause: periodic re-anchor only (legacy path)
  long cold_cert = 0;            ///< cold cause: stationarity certificate failed (use_cert)
  long cold_jump = 0;            ///< cold cause: carry stamp mismatch duel (stage-entry p jump)
  long cert_dual_confirms = 0;   ///< dual passes agreeing within cert_agree_rel (q_n ref refresh)
  long cold_won = 0;             ///< cold result kept over a VALID warm result
  long cold_won_guard = 0;       ///< subset of cold_won with plateau/anchor/cert/jump cause
  double cold_gain_relsum = 0.0; ///< sum of (costA-costB)/costA over cold_won
  // P3 preint cache evidence: window solves that reused / refilled the cached
  // preintegration, and total IMU-factor construction thread-CPU (whitener
  // build or fetch — previously untimed).
  long preint_hits = 0;
  long preint_misses = 0;
  double t_factor_sum = 0.0;
  /// Inner solves ended by the wall-clock hang guard: MUST be 0 on a healthy
  /// run — a nonzero count means machine load leaked into the iterate and the
  /// run is tainted for A/B, replay-parity, and falsifier purposes (the
  /// session evidence table prints a loud warning).
  long time_stops = 0;
  int accepted_passes = 0;       ///< accepted outer steps (passes - accepted = rejected/vetoed)
  int dim_p = 0;                 ///< free shared-parameter local dims this solve
  double final_merit = 0.0;      ///< merit at the shipped accepted point (0 if none)
  bool stopped_early = false;    ///< early-stop fired (stable steps + confirmation pass)
  int stop_pass = -1;            ///< pass index where the early-stop fired
  double qn_max_final = 0.0;     ///< max window q_n at the shipped accepted point
};

/// Warm-state carry ACROSS staged JointCalib calls on the SAME fused window
/// set (index-aligned). Stages re-solved every window from scratch before P1;
/// the carry hands each stage the previous stage's accepted nuisance optima
/// as warm-strand inits. FULL consume (seeds + comparable cost_acc/qn_ref, no
/// entry duels) requires (a) shared values bitwise-unchanged (p_stamp), (b)
/// the recording stage's ENTRY imu values equal to the consuming stage's
/// entry values (noise_stamp) — the ACI whitener freezes noise_lin from
/// calib.imu at solve entry, so a stage that MOVED imu values (A1a/A1b)
/// invalidates cost comparability even when its exit stamp matches — AND (c)
/// an identical free-set signature (layout_sig). Across a free-set boundary
/// the carry demotes to WARM-ONLY: warm states init the warm strand, but
/// seeds stay stage-fresh and every window runs one 'jump' duel at entry
/// (cause 'j'). Kept-path seeds must never cross a free-set boundary: the
/// restricted stage's arbitration would anchor BOTH strands of the expanded
/// stage's duels, removing the fresh-seed cold anchor (measured: S1 sim
/// split-half dqA 0.551 deg INCONSISTENT with carried seeds vs 0.235 deg
/// CONSISTENT without; Sting real data insensitive).
/// Seed-field snapshot (anchors of record — never the full window payload).
struct SeedSnap {
  bool has = false;
  std::vector<Eigen::Vector4d> q;
  std::vector<Eigen::Vector3d> v, p, feats;
  Eigen::Vector3d grav = Eigen::Vector3d::Zero(), bg = Eigen::Vector3d::Zero(), ba = Eigen::Vector3d::Zero();
};

struct JointWarmCarry {
  bool valid = false;
  std::vector<double> p_stamp;              ///< all shared values at the recording accepted point
  Eigen::Matrix<double, 16, 1> noise_stamp; ///< recording stage's ENTRY imu values (dw6 da6 qA4)
  std::string layout_sig;                   ///< recording stage's free-set signature ("name:gsize;")
  std::vector<WindowWarmState> warm;        ///< accepted nuisance optima per window
  std::vector<SeedSnap> seeds;              ///< seed anchors of record per window
  std::vector<double> cost;                 ///< accepted window costs
  std::vector<double> qn_ref;               ///< accepted q_n references
};

class JointCalib {
public:
  /**
   * @param store optional session preint store (PreintCache.h). Pure
   *        memoization — a hit returns the exact bytes recomputation would
   *        produce, so it is legal in the A-chain legacy stages (unlike
   *        cert/carry/early-stop, which change arbitration). Slots resolve by
   *        WindowData::uid on the main thread before the pool starts.
   */
  /**
   * @param warm_out optional: on success receives the per-window nuisance
   *        optima AT the final accepted point, aligned to the input window
   *        order (valid=false for dead / never-accepted windows). CONTRACT:
   *        warm_acc is promoted only on accepted passes and restore(accepted_p)
   *        rewrites the shared values, so the returned states correspond
   *        exactly to the shipped point — the Wald gate's one-evaluation-pass
   *        linearization depends on this (do not weaken via early-return
   *        paths without moving the hook).
   */
  static bool solve(const std::vector<WindowData> &windows, SharedCalib &calib, const JointConfig &cfg, JointReport &rep,
                    JointWarmCarry *carry = nullptr, PreintStore *store = nullptr,
                    std::vector<WindowWarmState> *warm_out = nullptr);
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_JOINT_CALIB_H
