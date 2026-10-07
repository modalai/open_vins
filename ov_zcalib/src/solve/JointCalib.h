/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: cross-window VarPro fusion on the shared calibration block.
 * Under its independent-window model, each outer evaluation re-preintegrates
 * and re-solves windows at the shared parameters, sums reduced information
 * and gradients, applies the global seed priors once, and retracts a damped
 * shared step. Overlapping or correlated data need separate modeling.
 * A successful return contains an accepted point and matching posterior;
 * iteration or time limits can stop before convergence. Session verification
 * and commit gates remain responsible for accepting that result.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_JOINT_CALIB_H
#define OV_ZCALIB_JOINT_CALIB_H

#include <map>
#include <functional>
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
  // Local stationarity heuristic for deciding whether a warm window needs
  // a fresh-seed comparison. With fused_schur off, it requires inner
  // convergence and a bounded nuisance Newton decrement. With fused_schur
  // on, it uses the decrement alone, including during full-budget warmup.
  // Other rescue, stranding and jump checks still apply.
  // This heuristic does not certify the nonlinear basin or physical accuracy.
  bool use_cert = true;
  /// Allow the heuristic when Da or R_AtoI is free. Otherwise those stages
  /// use the plateau/periodic fresh-seed comparisons. Session stages may
  /// override this independently of the standalone solver default.
  bool cert_open_imu = false;
  double cert_qn_rel = 2e-6;    ///< q_n ceiling as a fraction of window cost (1% of the 1e-4 acceptance band)
  double cert_ref_growth = 2.0; ///< q_n may grow at most this factor over its accepted reference
  double cert_agree_rel = 1e-3; ///< dual-pass agreement band that refreshes the q_n reference
  /// Minimum input window count before the stationarity heuristic applies.
  int cert_min_windows = 6;
  // Optional stopping after stable accepted steps and a cold confirmation
  // pass, only where the stationarity heuristic is enabled. Small steps or
  // merit changes do not prove that weak directions are accurately fitted.
  // Disabled by default; enabling changes the solve path.
  bool early_stop = false;
  int stop_k = 2;
  // Optional local-GN convergence stop: 0.5*g^T*(Lambda+Pi)^-1*g must stay
  // below conv_tol_rel*max(merit,1) for conv_k accepted steps, after
  // conv_min_accepts. The predicted reduction is a local model quantity,
  // not an exact nonlinear optimality or accuracy certificate.
  bool conv_stop = false;
  int conv_min_accepts = 3;
  int conv_k = 2;
  double conv_tol_rel = 1e-5;
  // ---- Fused (capped) evaluation: after fused_warmup_passes, warm-path
  // evaluations run fused_iters inner iterations and export -- the exported
  // gred = gk - Hkn Hnn^-1 gn is the nuisance-corrected joint-Newton reduced
  // gradient of the linearized least-squares model. First-order accuracy for
  // the NONLINEAR reduced gradient additionally requires negligible residual
  // curvature; a GN decrement alone does not prove an inexact-Newton bound.
  // The shared outer step and later nuisance corrections alternate. Cold
  // paths and warmup passes use full inner budgets. Capped evaluations skip
  // plateau/periodic comparisons but retain rescue, stranding and jump checks.
  // A final full-budget solve/export refreshes the reported linearization;
  // failure to finish it is not permission to report a capped posterior.
  bool fused_schur = false;
  int fused_warmup_passes = 1;
  /// Inner iterations for capped warm evaluations. These are correction
  /// steps, not an exact nuisance solution or implicit differentiation.
  int fused_iters = 1;
  /// Experimental deferral of healthy-warm fresh-seed comparisons until
  /// warm-only merit accepts the candidate. Rescue solves remain immediate.
  /// A rejected warm candidate might have improved with a fresh seed, so this
  /// changes arbitration. Incompatible capped evaluations disable it.
  bool duel_on_accept = false;
  /// Last N accepted steps use the full inner budget. This provides extra
  /// refinement without guaranteeing convergence to an uncapped trajectory.
  int fused_polish_accepts = 0;
  /// Defer reduced-information export until candidate acceptance. A warm
  /// path needing q_n for the stationarity check still exports inline; other
  /// paths export at their retained nuisance solution. Export failure vetoes
  /// a candidate, or removes an invalid entry window. Deferred-duel mode
  /// disables this optimization because its incremental information fold
  /// also uses the losing export.
  bool export_on_accept = true;
  double stop_step_winf = 0.01;  ///< ||dp/prior_sigma||_inf below this = stable step
  double stop_merit_rel = 3e-4;  ///< relative merit change below this = stable
  double stop_lambda_max = 1e-2; ///< lambda must be at/below this (not climbing a wall)
  /// Optional nuisance carry across calls. Comparable values, noise and
  /// layout stamps permit full reuse; other accepted carry becomes warm-only
  /// and receives a fresh-seed entry comparison. Cross-stage basins can still
  /// change even when local merit improves. Disabled by default.
  bool use_carry = false;
  /// Seeder config for the per-evaluation re-seeds. The SESSION must thread its
  /// (possibly bootstrap-adapted) LinearSeedConfig here: a default-constructed
  /// config silently drops bias_presolve and the widened no-still-baseline
  /// gates in exactly the stages that produce the committed answer.
  LinearSeedConfig seed;
  /// Per-window workers; results fold in fixed window order. Counts <=1 run
  /// inline. Equal inputs do not guarantee identical results when deadlines,
  /// compilers or floating-point execution differ; compare completed solves.
  int num_threads = 4;
  /// Wall-clock budget for ONE solve() call [s]; 0 = unlimited, <0 = skip.
  /// Admit another complete pass only when its measured cost, with headroom,
  /// fits. Deadline checks between window operations discard an incomplete
  /// candidate as a whole; only a complete accepted posterior may ship.
  /// A window factorization is not preemptible: this is a cooperative budget,
  /// not a hard real-time guarantee under arbitrary scheduling delays.
  double max_wall_s = 0.0;
  double budget_pass_hint_s = 0.0; ///< slowest complete pass measured by earlier session stages
  /// Optional elapsed-seconds clock for deterministic deadline tests. Empty
  /// uses steady_clock; report timings always retain actual wall time. A
  /// supplied callback must support concurrent reads when num_threads > 1.
  std::function<double()> budget_clock;
  /// Absolute per-dof per-outer step caps: the ACI3 mean correction and the
  /// temporal transport are FIRST-ORDER in dp, so an outer step must stay
  /// inside first-order validity regardless of how confident the fused
  /// information is (re-preintegration refreshes between outers). These caps
  /// bind together with the whitened 3-prior-sigma trust region (min of both).
  /// Tg's 2e-4 cap and 1e-3 prior are historical ICM/reference policy scales,
  /// not universal MEMS bounds or BMI270 accuracy validation. They control
  /// local optimization and seeding, separately from the commit ceiling.
  std::map<std::string, double> step_cap = {{"dw", 5e-3},     {"da", 5e-3},  {"q_AtoI", 5e-3}, {"q_ItoC", 0.01},
                                            {"p_IinC", 0.01}, {"td", 1e-3},  {"cam", 1.0},     {"tg", 2e-4}};
  /// Global seed priors (1-sigma), applied ONCE at fusion. Group name -> sigma.
  /// (no "tr" entry on either map: the rolling-shutter readout is a fixed hardware input from
  /// HAL3, never a free block, so it has no step to cap and no prior to weigh)
  std::map<std::string, double> prior_sigma = {{"dw", 0.02},    {"da", 0.02},  {"q_AtoI", 0.02}, {"q_ItoC", 0.05},
                                               {"p_IinC", 0.05}, {"td", 0.005}, {"cam", 2.0},    {"tg", 1e-3}};
  /// Per-dof camera prior override [fx fy cx cy k1 k2 k3 k4] (refine mode =
  /// tight priors; a ~1e-9 sigma is an information-level FREEZE, which is
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
  /// A1a constrains the off-diagonals with tight priors while fitting the
  /// diagonal scales. The full-chain gate controls their later release.
  /// Gravity, bias and unmodeled dynamics can couple into either set.
  bool use_da_prior_vec = false;
  Eigen::Matrix<double, 6, 1> da_prior_vec = (Eigen::Matrix<double, 6, 1>() << 0.02, 1e-9, 0.02, 1e-9, 1e-9, 0.02).finished();
  /// Anchor the FREE cam dofs' prior at an explicit center instead of the
  /// solve-entry values. Sequential cam sub-solves (B-1 alternation, settle,
  /// B-2 polish) otherwise re-center the prior at every entry -- a random-walk
  /// prior that legitimizes block-coordinate drift instead of arbitrating it;
  /// the anchor gives the whole phase ONE prior budget. Information-frozen
  /// dofs (sigma <= 1e-8) still center at entry: a freeze must HOLD the
  /// current value, never yank it back to the anchor mid-alternation.
  /// Each camera has its own center and per-dof prior mask; coverage decisions
  /// for one camera do not establish support for another.
  bool use_cam_prior_center = false;
  std::vector<Eigen::Matrix<double, 8, 1>> cam_prior_center;
};

struct JointReport {
  bool ok = false;
  int windows_used = 0;
  Eigen::VectorXd sigma;             ///< raw local-curvature 1-sigma per dof; not calibrated accuracy coverage
  Eigen::VectorXd prior_sigma_vec;   ///< matching prior sigmas (for improvement ratios)
  std::vector<std::string> labels;   ///< per local dof
  Eigen::MatrixXd Lambda;            ///< fused information (whitened checks downstream)
  double last_step_norm = 0.0;
  int evaluation_passes = 0;         ///< evaluations spent (re-seed+solve+export sweeps)
  int windows_dead = 0;              ///< windows dropped at an accepted point (never candidates)
  double wall_s = 0.0;               ///< wall clock of this solve() call
  double max_pass_s = 0.0;           ///< slowest evaluation/export pass, for downstream admission
  bool hit_wall_budget = false;      ///< stopped by max_wall_s (best accepted point shipped)
  // summed thread-CPU split across all window evaluations (> wall_s when parallel)
  double t_seed_sum = 0.0, t_preint_sum = 0.0, t_inner_sum = 0.0, t_export_sum = 0.0;
  long inner_iters_sum = 0;
  // Warm/cold two-path counters. A cold run is the fresh-seed path B; its
  // cause is attributed by priority first > warmfail > strand > plateau >
  // anchor, so cold_anchor counts PURE periodic re-anchors and cold_plateau
  // counts plateau fires that were not otherwise suspect -- the two triggers
  // the stationarity certificate replaces. cold_won_guard says how often
  // those triggered solves actually beat the warm result.
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
  // Preint cache evidence: window solves that reused / refilled the cached
  // preintegration, and total IMU-factor construction thread-CPU (whitener
  // build or fetch).
  long preint_hits = 0;
  long preint_misses = 0;
  double t_factor_sum = 0.0;
  /// Inner solves stopped by the wall-clock guard. Their accepted iterates
  /// depend on available runtime; compare this count and completion status
  /// before interpreting replay or thread-count differences.
  long time_stops = 0;
  int accepted_passes = 0;       ///< accepted outer steps (passes - accepted = rejected/vetoed)
  int dim_p = 0;                 ///< free shared-parameter local dims this solve
  double final_merit = 0.0;      ///< merit at the shipped accepted point (0 if none)
  bool stopped_early = false;    ///< early-stop fired (stable steps + confirmation pass)
  int stop_pass = -1;            ///< pass index where the early-stop fired
  double qn_max_final = 0.0;     ///< max window q_n at the shipped accepted point
};

/// Warm-state carry ACROSS staged JointCalib calls on the SAME fused window
/// set (index-aligned): hands each stage the previous stage's accepted
/// nuisance optima as warm-strand inits instead of re-solving every window
/// from scratch. FULL consume (seeds + comparable cost_acc/qn_ref, no
/// entry duels) requires (a) shared values bitwise-unchanged (p_stamp), (b)
/// the recording stage's ENTRY imu values equal to the consuming stage's
/// entry values (noise_stamp) -- the ACI whitener freezes noise_lin from
/// calib.imu at solve entry, so a stage that MOVED imu values (A1a/A1b)
/// invalidates cost comparability even when its exit stamp matches -- AND (c)
/// an identical free-set signature (layout_sig). Across a free-set boundary
/// the carry demotes to WARM-ONLY: warm states init the warm strand, but
/// seeds stay stage-fresh and every window runs one 'jump' duel at entry
/// (cause 'j'). Kept-path seeds must never cross a free-set boundary: the
/// restricted stage's arbitration would otherwise initialize both paths of
/// the expanded stage and remove its independent fresh-seed comparison.
/// Seed-field snapshot (anchors only, without full window payload).
struct SeedSnap {
  bool has = false;
  std::vector<Eigen::Vector4d> q;
  std::vector<Eigen::Vector3d> v, p, feats;
  Eigen::Vector3d grav = Eigen::Vector3d::Zero(), bg = Eigen::Vector3d::Zero(), ba = Eigen::Vector3d::Zero();
};

struct JointWarmCarry {
  bool valid = false;
  std::vector<double> p_stamp;              ///< all shared values at the recording accepted point
  Eigen::Matrix<double, 25, 1> noise_stamp; ///< recording stage's ENTRY imu values (dw6 da6 qA4 Tg9)
  std::string layout_sig;                   ///< recording stage's free-set signature ("name:gsize;")
  std::vector<WindowWarmState> warm;        ///< accepted nuisance optima per window
  std::vector<SeedSnap> seeds;              ///< seed anchors of record per window
  std::vector<double> cost;                 ///< accepted window costs
  std::vector<double> qn_ref;               ///< accepted q_n references
};

class JointCalib {
public:
  /**
   * @param store optional value-keyed preintegration cache. Reuse must match
   *        the interval, calibration and noise inputs. Resolve WindowData::uid
   *        slots on the calling thread before workers access distinct entries.
   */
  /**
   * @param warm_out optional: on success receives the per-window nuisance
   *        optima AT the final accepted point, aligned to the input window
   *        order (valid=false for dead / never-accepted windows). CONTRACT:
   *        warm_acc is promoted only on accepted passes and restore(accepted_p)
   *        rewrites the shared values, so the returned states correspond
   *        exactly to the shipped point -- the Wald gate's one-evaluation-pass
   *        linearization depends on this (do not weaken via early-return
   *        paths without moving the hook).
   */
  static bool solve(const std::vector<WindowData> &windows, SharedCalib &calib, const JointConfig &cfg, JointReport &rep,
                    JointWarmCarry *carry = nullptr, PreintStore *store = nullptr,
                    std::vector<WindowWarmState> *warm_out = nullptr);
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_JOINT_CALIB_H
