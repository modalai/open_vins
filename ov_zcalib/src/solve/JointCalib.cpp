/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: cross-window VarPro fusion (see JointCalib.h).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "JointCalib.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

#include "ceres_free/Parallel.h"

#include "../window/LinearSeed.h"
#include "utils/quat_ops.h"

using namespace ov_zcalib;

bool JointCalib::solve(const std::vector<WindowData> &windows, SharedCalib &calib, const JointConfig &cfg, JointReport &rep,
                       JointWarmCarry *carry, PreintStore *store, std::vector<WindowWarmState> *warm_out) {

  const auto t_entry = std::chrono::steady_clock::now();
  auto elapsed_s = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_entry).count(); };

  auto layout = calib.free_blocks();
  const int np = calib.local_dim();
  if (np == 0 || windows.empty())
    return false;

  // Freeze the noise linearization at fusion entry (weights must not chase p)
  calib.noise_lin = calib.imu;
  calib.noise_frozen = true;
  // Entry whitener stamp (carry validity): cost values are comparable across
  // stages ONLY under the identical noise_lin — which freezes HERE, from the
  // entry imu values, not from the recording stage's exit (verifier-caught:
  // A1a/A1b move imu, so their exit stamp matches the next entry while the
  // whitener does not).
  Eigen::Matrix<double, 16, 1> entry_noise;
  entry_noise << calib.imu.dw, calib.imu.da, calib.imu.q_AtoI;
  // Full-vector parameter stamp (33 doubles: ALL calib blocks, free or
  // frozen). Window costs depend on the frozen blocks too, and the staged
  // calls free DIFFERENT subsets — a free-subset stamp can never match across
  // a stage boundary and would demote every consume to jump duels. The full
  // vector matches exactly when the calib object is untouched between calls,
  // which is the condition under which carried costs are valid.
  auto full_stamp = [&calib]() {
    std::vector<double> s;
    s.reserve(16 + 17 * calib.cams.size()); // 16 shared IMU dofs + 17 per camera
    auto push = [&s](const double *p, int n) { s.insert(s.end(), p, p + n); };
    push(calib.imu.dw.data(), 6);
    push(calib.imu.da.data(), 6);
    push(calib.imu.q_AtoI.data(), 4);
    for (const CamCalib &k : calib.cams) {
      push(k.q_ItoC.data(), 4);
      push(k.p_IinC.data(), 3);
      push(k.cam.data(), 8);
      s.push_back(k.td);
      s.push_back(k.tr);
    }
    return s;
  };
  // Free-set signature: full consume is only sound between calls solving the
  // SAME problem shape (kept-path seeds and cost references are arbitration
  // products of that shape — see JointWarmCarry docs). Keyed on the per-camera
  // LABEL, so two cameras' blocks of the same physical kind stay distinguishable.
  std::string layout_sig;
  for (auto &b : layout)
    layout_sig += b.label() + ":" + std::to_string(b.gsize) + ";";

  // Seeds (for the once-applied global priors) and labels
  std::vector<std::vector<double>> seed;
  rep.labels.clear();
  rep.prior_sigma_vec.resize(np);
  {
    int off = 0;
    for (auto &b : layout) {
      seed.emplace_back(b.ptr, b.ptr + b.gsize);
      // Both the intrinsic prior MASK and its CENTER are per-camera: the mask because the k3/k4
      // radial gate is a question about what THIS camera actually saw, the center because each
      // camera has its own factory cal.
      const bool has_cam_prior =
          (b.name == "cam" && cfg.use_cam_prior_vec && b.cam >= 0 && (size_t)b.cam < cfg.cam_prior_vec.size());
      if (has_cam_prior && cfg.use_cam_prior_center && (size_t)b.cam < cfg.cam_prior_center.size())
        for (int k = 0; k < 8; ++k)
          if (cfg.cam_prior_vec[(size_t)b.cam](k) > 1e-8) // free dof: anchor; frozen dof: hold at entry
            seed.back()[k] = cfg.cam_prior_center[(size_t)b.cam](k);
      // Policy maps are keyed on the PHYSICAL name, not the per-camera label: a prior sigma or a
      // step cap describes the block's physics, and applies to every camera that has one.
      const double sg = cfg.prior_sigma.count(b.name) ? cfg.prior_sigma.at(b.name) : 1.0;
      for (int k = 0; k < b.lsize; ++k) {
        rep.labels.push_back(b.label() + "[" + std::to_string(k) + "]");
        rep.prior_sigma_vec(off + k) = has_cam_prior                               ? cfg.cam_prior_vec[(size_t)b.cam](k)
                                       : (b.name == "da" && cfg.use_da_prior_vec) ? cfg.da_prior_vec(k)
                                                                                  : sg;
      }
      off += b.lsize;
    }
  }

  // free-block value snapshot/restore (backtracking on true-cost increase)
  auto snapshot = [&]() {
    std::vector<std::vector<double>> s;
    for (auto &b : layout)
      s.emplace_back(b.ptr, b.ptr + b.gsize);
    return s;
  };
  auto restore = [&](const std::vector<std::vector<double>> &s) {
    for (size_t i = 0; i < layout.size(); ++i)
      std::copy(s[i].begin(), s[i].end(), layout[i].ptr);
  };

  // local deviation of the CURRENT p from the seed, per block (shared by the
  // prior fold and the prior-cost term of the merit)
  auto local_dev = [&](size_t bi) {
    auto &b = layout[bi];
    Eigen::VectorXd dloc(b.lsize);
    if (b.is_quat) {
      Eigen::Map<const Eigen::Vector4d> qn(b.ptr);
      Eigen::Map<const Eigen::Vector4d> q0(seed[bi].data());
      dloc = 2.0 * ov_core::quat_multiply(qn, ov_core::Inv(Eigen::Vector4d(q0))).head<3>();
    } else {
      for (int k = 0; k < b.lsize; ++k)
        dloc(k) = b.ptr[k] - seed[bi][k];
    }
    return dloc;
  };

  // Prior cost at the current p. The damped step is computed from the
  // prior-AUGMENTED system, so acceptance must judge the same objective —
  // comparing data cost alone accepts points that are not minima of the
  // objective whose curvature ships in rep.sigma.
  auto prior_cost_now = [&]() {
    double c = 0.0;
    int off = 0;
    for (size_t bi = 0; bi < layout.size(); ++bi) {
      const Eigen::VectorXd dloc = local_dev(bi);
      for (int k = 0; k < layout[bi].lsize; ++k) {
        const double z = dloc(k) / rep.prior_sigma_vec(off + k);
        c += 0.5 * z * z;
      }
      off += layout[bi].lsize;
    }
    return c;
  };

  // manifold retraction of a local step onto the free blocks
  auto apply_dp = [&](const Eigen::VectorXd &dp) {
    int off = 0;
    for (auto &b : layout) {
      if (b.is_quat) {
        Eigen::Vector4d dq;
        dq.head<3>() = 0.5 * dp.segment<3>(off);
        dq(3) = 1.0;
        dq /= dq.norm();
        Eigen::Map<Eigen::Vector4d> qcur(b.ptr);
        qcur = ov_core::quat_multiply(dq, Eigen::Vector4d(qcur));
      } else {
        for (int k = 0; k < b.lsize; ++k)
          b.ptr[k] += dp(off + k);
      }
      off += b.lsize;
    }
  };

  // Outer-level Levenberg step from the fused reduced system. The exported
  // Lambda is GAUSS-NEWTON information: in the tightly-coupled dw/da/td/grav
  // valley the true curvature carries second-order residual terms GN cannot
  // see (measured ~10-30x stiffer at linear-seed residual levels), so a raw
  // Newton step overshoots the valley wall. Damping bends the step toward
  // gradient descent using the SAME (Lambda, g) — a rejected candidate costs
  // one evaluation pass, never a new linearization.
  Eigen::MatrixXd Lsum(np, np);
  Eigen::MatrixXd accepted_L = Eigen::MatrixXd::Zero(np, np);
  Eigen::VectorXd gsum(np);
  Eigen::VectorXd accepted_g = Eigen::VectorXd::Zero(np);
  double prev_merit = std::numeric_limits<double>::infinity();
  std::vector<std::vector<double>> accepted_p = snapshot();

  // Working copies + GUARDED warm starts (the S1 two-start design). Naive
  // warm-carrying strands: LM from the previous optimum can sit above the
  // shifted valley floor at the new p and the outer loop stalls. Cold
  // re-seeding every pass avoids that but leaves %-level inner hysteresis in
  // the merit (re-seeds at nearby p take different inner paths), which on real
  // data exceeds the acceptance band and stalls the outer loop in reject
  // loops. The guard takes both: solve from the ACCEPTED point's nuisance
  // optimum (continuous, few inner iterations); if that fails or lands
  // suspiciously above the window's accepted cost, ALSO solve from a fresh
  // seed at the current p and keep the cheaper result. Bias/gauge priors are
  // anchored at the p-independent window seed, so both paths minimize the
  // same objective.
  std::vector<WindowData> work(windows.begin(), windows.end());
  std::vector<char> dead(work.size(), 0); // failed at an accepted point: never fusable
  // P3 preint-cache slots, resolved by uid on the MAIN thread (ensure() may
  // resize the store; pointers are taken only after every ensure() ran).
  // Workers touch only their own windows' entries (fixed-range partition).
  std::vector<WindowPreint *> pslot(work.size(), nullptr);
  if (store) {
    for (const WindowData &w : work)
      store->ensure(w.uid);
    for (size_t wi = 0; wi < work.size(); ++wi)
      if (work[wi].uid)
        pslot[wi] = &store->by_uid[work[wi].uid];
  }
  std::vector<WindowWarmState> warm_acc(work.size());  // nuisance optima at the accepted point
  std::vector<WindowWarmState> warm_cand(work.size()); // chosen result of the current evaluation
  std::vector<WindowWarmState> warm_dual(work.size()); // duel_on_accept: deferred path-B optima
  std::vector<double> cost_acc(work.size(), std::numeric_limits<double>::infinity());
  const double strand_guard = 0.05; // warm result above accepted cost by more than this -> try a fresh seed too
  // P1 certificate state: accepted q_n references (+inf = no reference yet ->
  // only the cost-relative ceiling applies) and the accepted q_n per window.
  std::vector<double> qn_ref(work.size(), std::numeric_limits<double>::infinity());
  std::vector<double> qn_acc(work.size(), 0.0);
  std::vector<char> jump(work.size(), 0); // carry stamp mismatch: force one duel at entry
  // Seed anchors of record (promoted at ACCEPTANCE only, per kept path — a
  // rejected candidate's path-B re-seed must never leak into the carry).
  std::vector<SeedSnap> seeds_acc(work.size());
  auto snap_seeds = [](const WindowData &w, SeedSnap &out) {
    out.has = w.has_seeds;
    out.q = w.seed_q;
    out.v = w.seed_v;
    out.p = w.seed_p;
    out.feats = w.seed_feats;
    out.grav = w.seed_grav;
    out.bg = w.seed_bg;
    out.ba = w.seed_ba;
  };
  auto apply_seeds = [](const SeedSnap &in, WindowData &w) {
    if (!in.has)
      return;
    w.has_seeds = true;
    w.seed_q = in.q;
    w.seed_v = in.v;
    w.seed_p = in.p;
    w.seed_feats = in.feats;
    w.seed_grav = in.grav;
    w.seed_bg = in.bg;
    w.seed_ba = in.ba;
  };
  // ---- carry consume (stage-entry warm start across staged calls) ----
  const bool carry_on = (carry != nullptr) && cfg.use_carry;
  if (carry_on && carry->valid && carry->warm.size() == work.size()) {
    const std::vector<double> now = full_stamp();
    bool stamp_match = (now.size() == carry->p_stamp.size());
    if (stamp_match)
      for (size_t i = 0; i < now.size(); ++i)
        if (carry->p_stamp[i] != now[i]) { // bitwise
          stamp_match = false;
          break;
        }
    bool noise_match = true;
    for (int k = 0; k < 16; ++k)
      if (carry->noise_stamp(k) != entry_noise(k)) { // bitwise: whitener identity
        noise_match = false;
        break;
      }
    const bool sig_match = (carry->layout_sig == layout_sig);
    const bool full = stamp_match && noise_match && sig_match;
    for (size_t wi = 0; wi < work.size(); ++wi) {
      warm_acc[wi] = carry->warm[wi];
      if (full) { // kept-path seeds only persist within the shape that arbitrated them
        apply_seeds(carry->seeds[wi], work[wi]);
        seeds_acc[wi] = carry->seeds[wi];
      }
      cost_acc[wi] = full ? carry->cost[wi] : std::numeric_limits<double>::infinity();
      qn_ref[wi] = full ? carry->qn_ref[wi] : std::numeric_limits<double>::infinity();
      jump[wi] = full ? 0 : 1; // demoted carry: one arbitration duel at entry (warm vs stage-fresh cold)
    }
    if (cfg.verbose)
      std::printf("[joint] carry consumed: %s (stamp %s, whitener %s, shape %s)\n",
                  full ? "FULL (costs comparable)" : "warm-only (jump duels)", stamp_match ? "match" : "moved",
                  noise_match ? "match" : "moved", sig_match ? "match" : "moved");
  }

  // Per-window evaluation slots, reduced IN WINDOW ORDER after the parallel
  // sweep — serial == parallel bit-identical (fixed ranges, no shared writes).
  struct EvalSlot {
    bool attempted = false, ok = false;
    Eigen::MatrixXd L;
    Eigen::VectorXd g;
    double cost = 0.0;
    double t_seed = 0.0, t_preint = 0.0, t_inner = 0.0, t_export = 0.0;
    int iters = 0;
    // P0 evidence (per pass): which paths ran, why cold fired, whether it won
    char ran_warm = 0, cold_cause = 0, cold_win = 0;
    double cold_gain = 0.0;
    // P1: kept-result q_n, kept path ('A'/'B'/0), dual agreement, and the
    // pre-path-B seed stash (path A's anchors of record when A is kept on a
    // pass where B re-seeded — verifier correction: the snapshot must match
    // the KEPT path's objective, not the post-re-seed state)
    double qn = 0.0;
    char kept_path = 0, dual_agree = 0;
    SeedSnap seeds_pre;
    // P3: preint cache hits/misses this pass + factor-construction thread-CPU
    int phit = 0, pmiss = 0;
    double t_factor = 0.0;
    // wall-clock hang-guard firings (must stay 0; nonzero = load-tainted run)
    int tstop = 0;
    // duel_on_accept: deferred-duel bookkeeping (phase-B runs post-accept)
    char deferred_cause = 0;
    bool d_ran = false, d_ok = false, d_win = false;
    double d_cost = 0.0, d_qn = 0.0, d_seed = 0.0, d_preint = 0.0, d_inner = 0.0, d_export = 0.0, d_factor = 0.0;
    long d_iters = 0;
    int d_phit = 0, d_pmiss = 0, d_tstop = 0;
    Eigen::MatrixXd d_L;
    Eigen::VectorXd d_g;
  };
  std::vector<EvalSlot> slots(work.size());
  const int nthreads = std::max(1, std::min(cfg.num_threads, (int)work.size()));
  ov_init::zbft_sfm::ParallelExecutor pool(nthreads);

  // LPT (longest-processing-time-first) order for the DYNAMIC window schedule.
  // Window solves cost wildly different amounts -- the nuisance state is
  // 15*clones + 3*landmarks and the dense factorization is CUBIC in it, so a
  // 32-clone/300-obs window can cost several times a 12-clone/80-obs one. A
  // static contiguous partition gates every pass on whichever range drew the
  // heavy ones (measured: 63% parallel efficiency, ~22 s/solve spent waiting).
  // Handing the EXPENSIVE windows out first leaves only cheap ones to fill the
  // tail, which is the classic 4/3-optimal makespan greedy.
  //
  // The cost proxy is static and deterministic (no timing feedback), so the
  // dispatch order is identical on every run and every machine. Results are
  // bit-identical to the static schedule regardless -- window i writes only slot
  // i, and the fold below is serial in index order (see Parallel.h).
  std::vector<int> order(work.size());
  for (size_t i = 0; i < work.size(); ++i)
    order[i] = (int)i;
  {
    std::vector<double> cost(work.size(), 0.0);
    for (size_t i = 0; i < work.size(); ++i) {
      const double nc = (double)work[i].clone_times.size(); // dense nav block is cubic in clones
      double nobs = 0.0;
      for (const auto &co : work[i].obs)
        nobs += (double)co.size();
      cost[i] = nc * nc * nc + 30.0 * nobs; // factorization + assembly/Schur
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cost[a] > cost[b]; });
  }

  double lm_lambda = 1e-2;
  int rejects_in_a_row = 0;
  bool have_lin = false; // (accepted_L, accepted_g) valid at accepted_p
  const int max_evals = cfg.outer_iterations * (1 + cfg.max_backtracks);
  int accepted_steps = 0;
  // P1 early-stop state: last applied step's post-cap whitened norm, predicted
  // reduction (undamped model), lambda, cap flag; consecutive-stable counter
  // and the deterministic cold stop-confirmation pass.
  // STAGE-AWARE certificate eligibility (the P0 verdict, confirmed by the
  // synthetic suite): the legacy plateau/anchor cold-solves are pure waste
  // ONLY where the shallow subspace (da off-diag, q_AtoI) is frozen — cam/
  // ext/td stages. Where the accel chain is FREE, the cold cross-checks are
  // load-bearing (P0: 4.9-15.5% mean cold-win gains; calib_e2e S3 regressed
  // 0.074->0.100 deg ext / ->0.321 deg qA under a global certificate), so
  // those solves keep the full legacy two-path.
  // (single-window solves excluded: with N=1 the fused step IS the window
  // step, the second-order argument collapses, and the calib_e2e p_IinC probe
  // measured 4.0 -> 4.8 mm under a global certificate)
  const bool cert_on = cfg.use_cert && (cfg.cert_open_imu || (!calib.imu.calib_da && !calib.imu.calib_RAtoI)) &&
                       (int)windows.size() >= cfg.cert_min_windows;
  // duel_on_accept is MEASURED-INCOMPATIBLE with fused (capped) evals: the
  // strand duels are what make candidate merits comparable to the accepted
  // baseline (full-vs-full); deferring them leaves 1-iter candidate merits
  // against a full-solve baseline -> every candidate 'rises' -> entry
  // deadlock (Sting: td frozen at bootstrap, zero accepts). The loose-vs-
  // loose baseline variant is the scoped follow-on; until then the flag
  // self-disarms under fused_schur.
  JointConfig cfg_eff = cfg;
  if (cfg_eff.duel_on_accept && cfg_eff.fused_schur) {
    if (cfg.verbose)
      std::printf("[joint] duel_on_accept disarmed: incompatible with fused evals (incomparable merits)\n");
    cfg_eff.duel_on_accept = false;
  }
  const bool duel_defer = cfg_eff.duel_on_accept;
  const bool estop_on = cfg.early_stop && cert_on;
  int stable_run = 0;
  int conv_run = 0; // conv-stop consecutive below-tolerance accepted steps
  bool confirm_pending = false, confirm_active = false;
  double pend_winf = 0.0, pend_pred = 0.0, pend_lambda = 0.0;
  bool pend_capped = false, have_pending = false;
  for (int pass = 0; pass < max_evals && accepted_steps < cfg.outer_iterations; ++pass) {
    if (cfg.max_wall_s > 0.0 && elapsed_s() > cfg.max_wall_s) {
      rep.hit_wall_budget = true;
      if (cfg.verbose)
        std::printf("[joint] wall budget %.1fs hit after %d passes -> stop at best accepted point\n", cfg.max_wall_s, pass);
      break;
    }
    // stop-confirmation pass: force one duel on every window whose accepted
    // q_n is material — the fixed-order, deterministic cross-check that a
    // stable-looking outer point is not resting on under-converged nuisances
    // (the v5 lesson: time-stable junk defeats pure consistency)
    confirm_active = confirm_pending;
    confirm_pending = false;
    if (confirm_active)
      for (size_t wi = 0; wi < work.size(); ++wi)
        if (qn_acc[wi] > cfg.cert_qn_rel * std::max(cost_acc[wi], 1e-300))
          jump[wi] = 1;
    // ---- evaluate at the current p: (seed-if-cold + solve nuisances + export) per window ----
    rep.evaluation_passes++;
    pool.parallel_dynamic((int)work.size(), [&](int /*worker*/, int k) {
      {
        const int wi = order[k]; // LPT: expensive windows are handed out first
        EvalSlot &s = slots[wi];
        s.attempted = false;
        s.ok = false;
        s.t_seed = s.t_preint = s.t_inner = s.t_export = 0.0;
        s.iters = 0;
        s.ran_warm = s.cold_cause = s.cold_win = 0;
        s.cold_gain = 0.0;
        s.qn = 0.0;
        s.kept_path = s.dual_agree = 0;
        s.seeds_pre.has = false;
        s.phit = s.pmiss = 0;
        s.t_factor = 0.0;
        s.tstop = 0;
        s.deferred_cause = 0;
        s.d_ran = s.d_ok = s.d_win = false;
        if (dead[wi])
          return;
        s.attempted = true;
        // ---- path A (continuity): solve from the accepted point's optimum ----
        WindowSolveReport wrA;
        WindowWarmState wA = warm_acc[wi]; // copy: the solve mutates its warm state
        bool okA = false;
        const bool had_warm = wA.valid;
        if (wA.valid) {
          const bool cap_now = cfg.fused_schur && pass >= cfg.fused_warmup_passes &&
                               accepted_steps < cfg.outer_iterations - cfg.fused_polish_accepts;
          const int itA = cap_now ? cfg.fused_iters : cfg.window_max_iters;
          okA = WindowBA::solve_and_export(work[wi], calib, true, wrA, itA, false, &wA, pslot[wi]) &&
                (int)wrA.Lambda.rows() == np;
          s.t_preint += wrA.t_preint;
          s.t_inner += wrA.t_inner;
          s.t_export += wrA.t_export;
          s.t_factor += wrA.t_factor;
          s.iters += wrA.iterations;
          (wrA.preint_hit ? s.phit : s.pmiss)++;
          s.tstop += wrA.time_stopped ? 1 : 0;
          s.ran_warm = 1;
        }
        // ---- path B (fresh seed). Under the P1 CERTIFICATE (use_cert): the
        // duplicate solve runs only on genuine suspicion — warm failure,
        // non-stationary inner exit, cost stranding, material nuisance Newton
        // decrement (q_n polices exactly the stale-shallow-gradient case the
        // legacy plateau trigger over-approximated: the exported gred is the
        // first-order-corrected VarPro gradient, so the unpoliced residual is
        // the second-order term q_n/2), or a carry jump duel. Legacy triggers
        // (plateau <=2 iters, every-3rd-pass anchor) remain reachable under
        // use_cert=false for bit-identical A/B. ----
        const bool anchor_pass = !cert_on && !cfg.fused_schur && (pass % 3 == 0);
        const bool plateau = !cert_on && !cfg.fused_schur && okA && wrA.iterations <= 2;
        const bool strand = okA && !(wrA.cost_final <= cost_acc[wi] * (1.0 + strand_guard));
        bool cert_fail = false;
        if (cert_on && okA && !strand) {
          const double qn_band =
              std::max(cfg.cert_qn_rel * wrA.cost_final, std::isfinite(qn_ref[wi]) ? cfg.cert_ref_growth * qn_ref[wi] : 0.0);
          // Capped-regime certificate (P4): a capped eval is structurally
          // !inner_converged, but qn — the nuisance Newton decrement the
          // export computes AT the current point — measures stationarity
          // regardless of how many iterations produced the point. Under
          // fused_schur the cert polices on qn alone; in the legacy regime a
          // non-stationary exit stays suspect as before.
          cert_fail = (!cfg.fused_schur && !wrA.inner_converged) || wrA.qn > qn_band;
        }
        const bool suspect = !okA || anchor_pass || plateau || cert_fail || jump[wi] || strand;
        // duel_on_accept: quality duels on HEALTHY warm evals defer until the
        // candidate is accepted (rescue duels stay inline — only path there)
        const bool defer = duel_defer && suspect && okA && had_warm;
        if (defer)
          s.deferred_cause = jump[wi] ? 'j' : strand ? 's' : cert_fail ? 'c' : plateau ? 'p' : 'a';
        WindowSolveReport wrB;
        WindowWarmState wB; // invalid: forces the seed init inside the solve
        bool okB = false;
        if (suspect && !defer) {
          snap_seeds(work[wi], s.seeds_pre); // path A's anchors of record (pre-re-seed)
          const auto t_s0 = std::chrono::steady_clock::now();
          LinearSeedReport sr;
          // Re-seed at the current p; on a GATE failure fall back to the
          // window's existing seeds (harvest/bootstrap/sim provenance) — the
          // fallback is load-bearing: stage entries arrive at a moved p where
          // marginal windows gate-fail, and killing them starves the very
          // stages (A1 IMU intrinsics) the windows were collected for.
          const bool seeded = LinearSeed::seed_window(work[wi], calib, work[wi].seed_bg, sr, cfg.seed);
          s.t_seed += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_s0).count();
          if (seeded || work[wi].has_seeds) {
            okB = WindowBA::solve_and_export(work[wi], calib, true, wrB, cfg.window_max_iters, false, &wB, pslot[wi]) &&
                  (int)wrB.Lambda.rows() == np;
            s.t_preint += wrB.t_preint;
            s.t_inner += wrB.t_inner;
            s.t_export += wrB.t_export;
            s.t_factor += wrB.t_factor;
            s.iters += wrB.iterations;
            (wrB.preint_hit ? s.phit : s.pmiss)++;
            s.tstop += wrB.time_stopped ? 1 : 0;
            // P0/P1 cause attribution, priority first > warmfail > jump > strand > cert > plateau > anchor
            s.cold_cause = !had_warm ? 'f'
                           : !okA    ? 'x'
                           : jump[wi] ? 'j'
                           : strand   ? 's'
                           : cert_fail ? 'c'
                           : plateau ? 'p'
                                     : 'a';
          }
        }
        // ---- keep the cheaper valid result ----
        const bool useA = okA && (!okB || wrA.cost_final <= wrB.cost_final);
        if (okA && okB && !useA) {
          s.cold_win = 1;
          s.cold_gain = (wrA.cost_final - wrB.cost_final) / std::max(wrA.cost_final, 1e-300);
        }
        if (okA && okB && std::abs(wrA.cost_final - wrB.cost_final) <= cfg.cert_agree_rel * std::max(wrA.cost_final, 1e-300))
          s.dual_agree = 1; // confirming duel: refresh the q_n reference at acceptance
        s.kept_path = useA ? 'A' : (okB ? 'B' : 0);
        s.qn = useA ? wrA.qn : wrB.qn;
        WindowSolveReport &wr = useA ? wrA : wrB;
        if (okA || okB) {
          s.L = std::move(wr.Lambda);
          s.g = std::move(wr.gred);
          s.cost = wr.cost_final;
          warm_cand[wi] = useA ? std::move(wA) : std::move(wB);
          s.ok = true;
        }
      }
    });
    std::fill(jump.begin(), jump.end(), 0); // entry/confirmation duels fire exactly once
    // ---- fixed-order reduction + failure semantics ----
    // A window that fails at an ACCEPTED point (incl. the entry point) is dead:
    // it can never join the fused sum, and excluding it keeps the merit
    // comparable across passes. A window that solved at the accepted point but
    // fails at a CANDIDATE p VETOES the candidate (cost = +inf): the merit
    // stays a function of p over a FIXED window set, never a silent subset.
    Lsum.setZero();
    gsum.setZero();
    rep.windows_used = 0;
    double cost_total = 0.0;
    bool veto = false;
    for (size_t wi = 0; wi < work.size(); ++wi) {
      const EvalSlot &s = slots[wi];
      if (!s.attempted)
        continue;
      rep.t_seed_sum += s.t_seed;
      rep.t_preint_sum += s.t_preint;
      rep.t_inner_sum += s.t_inner;
      rep.t_export_sum += s.t_export;
      rep.t_factor_sum += s.t_factor;
      rep.preint_hits += s.phit;
      rep.preint_misses += s.pmiss;
      rep.time_stops += s.tstop;
      rep.inner_iters_sum += s.iters;
      rep.warm_evals += s.ran_warm ? 1 : 0;
      rep.cert_dual_confirms += s.dual_agree ? 1 : 0;
      if (s.cold_cause) {
        rep.cold_evals++;
        switch (s.cold_cause) {
        case 'f': rep.cold_first++; break;
        case 'x': rep.cold_warmfail++; break;
        case 'j': rep.cold_jump++; break;
        case 's': rep.cold_strand++; break;
        case 'c': rep.cold_cert++; break;
        case 'p': rep.cold_plateau++; break;
        default:  rep.cold_anchor++; break;
        }
        if (s.cold_win) {
          rep.cold_won++;
          rep.cold_gain_relsum += s.cold_gain;
          if (s.cold_cause == 'p' || s.cold_cause == 'a' || s.cold_cause == 'c')
            rep.cold_won_guard++;
        }
      }
      if (!s.ok) {
        if (!have_lin) {
          dead[wi] = 1; // entry-point failure: drop for the whole solve
          continue;
        }
        veto = true; // candidate failure: reject this candidate (re-seeded again next pass)
        continue;
      }
      Lsum += s.L;
      gsum += s.g;
      cost_total += s.cost;
      rep.windows_used++;
    }
    if (rep.windows_used == 0)
      return false;
    double merit = cost_total + prior_cost_now();
    if (!std::isfinite(merit))
      veto = true; // a NaN/inf evaluation must never be accepted (NaN defeats comparisons)

    // ---- duel_on_accept: the deferred quality duels run ONLY for a candidate
    // that already wins on warm-only merit; their improvements can only lower
    // window costs, so acceptance is monotone (never revoked). Counters land
    // here (the main reduction ran without them). ----
    if (duel_defer && !veto && merit <= prev_merit * (1.0 + 1e-4)) {
      bool any_deferred = false;
      for (size_t wi = 0; wi < work.size(); ++wi)
        if (slots[wi].attempted && slots[wi].ok && slots[wi].deferred_cause)
          any_deferred = true;
      if (any_deferred) {
        pool.parallel_dynamic((int)work.size(), [&](int /*worker*/, int k) {
          {
            const int wi = order[k];
            EvalSlot &s = slots[wi];
            if (!s.attempted || !s.ok || !s.deferred_cause || dead[wi])
              return;
            s.d_ran = true;
            snap_seeds(work[wi], s.seeds_pre); // path A's anchors of record (pre-re-seed)
            const auto t_s0 = std::chrono::steady_clock::now();
            LinearSeedReport sr;
            const bool seeded = LinearSeed::seed_window(work[wi], calib, work[wi].seed_bg, sr, cfg.seed);
            s.d_seed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_s0).count();
            if (!(seeded || work[wi].has_seeds))
              return;
            WindowSolveReport wrB;
            WindowWarmState wB;
            s.d_ok = WindowBA::solve_and_export(work[wi], calib, true, wrB, cfg.window_max_iters, false, &wB, pslot[wi]) &&
                     (int)wrB.Lambda.rows() == np;
            s.d_preint = wrB.t_preint;
            s.d_inner = wrB.t_inner;
            s.d_export = wrB.t_export;
            s.d_factor = wrB.t_factor;
            s.d_iters = wrB.iterations;
            (wrB.preint_hit ? s.d_phit : s.d_pmiss)++;
            s.d_tstop = wrB.time_stopped ? 1 : 0;
            if (s.d_ok) {
              s.d_cost = wrB.cost_final;
              s.d_qn = wrB.qn;
              s.d_L = std::move(wrB.Lambda);
              s.d_g = std::move(wrB.gred);
              warm_dual[wi] = std::move(wB);
            }
          }
        });
        // serial fold: counters + incremental (Lsum, gsum, cost) updates
        for (size_t wi = 0; wi < work.size(); ++wi) {
          EvalSlot &s = slots[wi];
          if (!s.d_ran)
            continue;
          rep.t_seed_sum += s.d_seed;
          rep.t_preint_sum += s.d_preint;
          rep.t_inner_sum += s.d_inner;
          rep.t_export_sum += s.d_export;
          rep.t_factor_sum += s.d_factor;
          rep.preint_hits += s.d_phit;
          rep.preint_misses += s.d_pmiss;
          rep.time_stops += s.d_tstop;
          rep.inner_iters_sum += s.d_iters;
          rep.cold_evals++;
          switch (s.deferred_cause) {
          case 'j': rep.cold_jump++; break;
          case 's': rep.cold_strand++; break;
          case 'c': rep.cold_cert++; break;
          case 'p': rep.cold_plateau++; break;
          default:  rep.cold_anchor++; break;
          }
          // the duel RAN: mark the cause regardless of winner so the accept
          // block promotes the PRE-re-seed anchors for kept-A windows (the
          // verifier-correction semantics; omitting this drifts the bias/gauge
          // priors onto post-re-seed anchors and collapses the session)
          s.cold_cause = s.deferred_cause;
          if (s.d_ok && std::abs(s.cost - s.d_cost) <= cfg.cert_agree_rel * std::max(s.cost, 1e-300)) {
            s.dual_agree = 1;
            rep.cert_dual_confirms++;
          }
          if (s.d_ok && s.d_cost < s.cost) {
            s.d_win = 1;
            rep.cold_won++;
            rep.cold_gain_relsum += (s.cost - s.d_cost) / std::max(s.cost, 1e-300);
            if (s.deferred_cause == 'p' || s.deferred_cause == 'a' || s.deferred_cause == 'c')
              rep.cold_won_guard++;
            Lsum += s.d_L - s.L;
            gsum += s.d_g - s.g;
            cost_total += s.d_cost - s.cost;
            s.L = std::move(s.d_L);
            s.g = std::move(s.d_g);
            s.cost = s.d_cost;
            s.qn = s.d_qn;
            s.cold_cause = s.deferred_cause;
            s.cold_win = 1;
            s.kept_path = 'B';
            warm_cand[wi] = std::move(warm_dual[wi]);
          }
        }
        merit = cost_total + prior_cost_now(); // monotone <= warm-only merit
      }
    }

    // small tolerance absorbs residual inner-solver hysteresis
    if (veto || merit > prev_merit * (1.0 + 1e-4)) {
      restore(accepted_p);
      lm_lambda = std::min(lm_lambda * 8.0, 1e5);
      ++rejects_in_a_row;
      if (cfg.verbose)
        std::printf("[joint] pass %d: %s (%.4e > %.4e) -> damp lambda=%.1e\n", pass, veto ? "candidate VETOED (window failure)" : "merit rose",
                    merit, prev_merit, lm_lambda);
      if (estop_on && confirm_active) {
        // the forced-duel confirmation pass could not even produce an
        // acceptable candidate: the accepted point stands as stationary
        rep.stopped_early = true;
        rep.stop_pass = pass;
        if (cfg.verbose)
          std::printf("[joint] early-stop CONFIRMED at pass %d (confirmation pass rejected)\n", pass);
        restore(accepted_p);
        break;
      }
      if (rejects_in_a_row > cfg.max_backtracks)
        break; // fully damped and still rising: stop at best
      // recompute the candidate from the ACCEPTED linearization with more damping
      if (!have_lin)
        break;
    } else {
      // ---- accept the current point; fold the seed priors; store linearization ----
      const double merit_before = prev_merit; // for the early-stop stability test
      rejects_in_a_row = 0;
      prev_merit = merit;
      accepted_p = snapshot();
      // promote the accepted evaluation's nuisance optima to the warm baseline
      for (size_t wi = 0; wi < work.size(); ++wi)
        if (slots[wi].attempted && slots[wi].ok) {
          warm_acc[wi] = std::move(warm_cand[wi]);
          cost_acc[wi] = slots[wi].cost;
          qn_acc[wi] = slots[wi].qn;
          // q_n reference: first acceptance arms the growth clause; confirming
          // duels (paths agreed within cert_agree_rel) refresh it. Without the
          // init the growth clause never fires and the certificate degrades to
          // the absolute band alone.
          if (!std::isfinite(qn_ref[wi]) || slots[wi].dual_agree)
            qn_ref[wi] = slots[wi].qn;
          // seed anchors of record, per KEPT path (verifier correction: a pass
          // where B re-seeded but A was kept must promote A's PRE-re-seed
          // anchors — the objective A minimized — not the post-B state)
          if (slots[wi].kept_path == 'A' && slots[wi].cold_cause != 0)
            seeds_acc[wi] = slots[wi].seeds_pre;
          else
            snap_seeds(work[wi], seeds_acc[wi]);
        }
      lm_lambda = std::max(lm_lambda * 0.25, 1e-4);
      // ---- P1 early-stop: consecutive stable accepted steps + confirmation ----
      if (estop_on && have_pending) {
        const double actual = std::isfinite(merit_before) ? merit_before - merit : 0.0;
        const double rel = std::abs(actual) / std::max(merit, 1.0);
        const bool stable = !pend_capped && pend_winf <= cfg.stop_step_winf && rel <= cfg.stop_merit_rel &&
                            pend_lambda <= cfg.stop_lambda_max;
        if (confirm_active) {
          if (rel <= cfg.stop_merit_rel) {
            // the forced duels bought nothing: genuinely stationary
            rep.stopped_early = true;
            rep.stop_pass = pass;
            if (cfg.verbose)
              std::printf("[joint] early-stop CONFIRMED at pass %d (confirmation duels moved merit %.2e rel)\n", pass, rel);
            break;
          }
          stable_run = 0; // confirmation found real progress: keep iterating
        } else if (stable) {
          if (++stable_run >= cfg.stop_k)
            confirm_pending = true;
        } else {
          stable_run = 0;
        }
      }
      {
        int off = 0;
        for (size_t bi = 0; bi < layout.size(); ++bi) {
          const Eigen::VectorXd dloc = local_dev(bi);
          for (int k = 0; k < layout[bi].lsize; ++k) {
            const double info = 1.0 / (rep.prior_sigma_vec(off + k) * rep.prior_sigma_vec(off + k));
            Lsum(off + k, off + k) += info;
            gsum(off + k) += info * dloc(k);
          }
          off += layout[bi].lsize;
        }
      }
      accepted_L = Lsum;
      accepted_g = gsum;
      have_lin = true;
      ++accepted_steps;
      if (cfg.verbose)
        std::printf("[joint] pass %d: ACCEPT windows=%d merit=%.4e (data %.4e) lambda=%.1e\n", pass, rep.windows_used, merit, cost_total,
                    lm_lambda);
      // ---- conv-stop: outer Newton decrement at the accepted point ----
      if (cfg.conv_stop && accepted_steps >= cfg.conv_min_accepts) {
        Eigen::LDLT<Eigen::MatrixXd> ldn(accepted_L);
        if (ldn.info() == Eigen::Success) {
          const double lam2 = accepted_g.dot(ldn.solve(accepted_g)); // = predicted 2x attainable reduction
          const bool below = 0.5 * lam2 < cfg.conv_tol_rel * std::max(merit, 1.0);
          conv_run = below ? conv_run + 1 : 0;
          if (conv_run >= cfg.conv_k) {
            rep.stopped_early = true;
            rep.stop_pass = pass;
            if (cfg.verbose)
              std::printf("[joint] conv-stop at pass %d: decrement/2 %.3e < %.1e x merit for %d accepted steps\n", pass, 0.5 * lam2,
                          cfg.conv_tol_rel, cfg.conv_k);
            break;
          }
        }
      }
    }

    // ---- damped step from the accepted linearization ----
    Eigen::MatrixXd Ld = accepted_L;
    Ld.diagonal() += lm_lambda * accepted_L.diagonal().cwiseMax(1e-12);
    Eigen::LDLT<Eigen::MatrixXd> ldlt(Ld);
    if (ldlt.info() != Eigen::Success)
      return false;
    Eigen::VectorXd dp = ldlt.solve(-accepted_g);
    const double wnorm = (dp.cwiseQuotient(rep.prior_sigma_vec)).lpNorm<Eigen::Infinity>();
    const double trust = 3.0; // max step = 3 prior-sigmas per dof
    if (wnorm > trust)
      dp *= trust / wnorm;
    {
      // per-block absolute caps (single global scale so the step DIRECTION holds)
      double scale = 1.0;
      int off = 0;
      for (auto &b : layout) {
        const auto it = cfg.step_cap.find(b.name);
        if (it != cfg.step_cap.end()) {
          const double m = dp.segment(off, b.lsize).cwiseAbs().maxCoeff();
          if (m > it->second)
            scale = std::min(scale, it->second / m);
        }
        off += b.lsize;
      }
      dp *= scale;
      pend_capped = (wnorm > trust) || (scale < 1.0);
    }
    // early-stop pending stats for the NEXT pass's stability test: post-cap
    // whitened step, predicted reduction on the UNDAMPED accepted model
    pend_winf = (dp.cwiseQuotient(rep.prior_sigma_vec)).lpNorm<Eigen::Infinity>();
    pend_pred = -(accepted_g.dot(dp) + 0.5 * dp.dot(accepted_L * dp));
    pend_lambda = lm_lambda;
    have_pending = true;
    rep.last_step_norm = dp.norm();
    if (rep.last_step_norm < 1e-10)
      break;
    apply_dp(dp);
  }
  // report the last ACCEPTED point: its Lambda/sigma were evaluated exactly
  // there, and a trailing un-evaluated (or rejected) step must not ship
  restore(accepted_p);
  // ---- P4 finalize: capped evals carried the outer loop; the SHIPPED
  // linearization (commit gates read rep sigmas) must be commit-grade. One
  // tight pass at the accepted point from the accepted warm states, fixed-
  // order reduction, replacing accepted_L/accepted_g and the warm states.
  if (cfg.fused_schur && have_lin) {
    std::vector<Eigen::MatrixXd> Lf(work.size());
    std::vector<Eigen::VectorXd> gf(work.size());
    std::vector<char> okf(work.size(), 0);
    pool.parallel_dynamic((int)work.size(), [&](int, int k) {
      {
        const int wi = order[k];
        if (dead[wi] || !warm_acc[wi].valid)
          return;
        WindowSolveReport wrf;
        WindowWarmState wf = warm_acc[wi];
        if (WindowBA::solve_and_export(work[wi], calib, true, wrf, cfg.window_max_iters, false, &wf, pslot[wi]) &&
            (int)wrf.Lambda.rows() == np) {
          Lf[wi] = std::move(wrf.Lambda);
          gf[wi] = std::move(wrf.gred);
          warm_acc[wi] = std::move(wf);
          okf[wi] = 1;
          rep.inner_iters_sum += wrf.iterations;
          rep.t_preint_sum += wrf.t_preint;
          rep.t_inner_sum += wrf.t_inner;
          rep.t_export_sum += wrf.t_export;
          rep.t_factor_sum += wrf.t_factor;
          rep.time_stops += wrf.time_stopped ? 1 : 0;
        }
      }
    });
    Eigen::MatrixXd Lfin = Eigen::MatrixXd::Zero(np, np);
    Eigen::VectorXd gfin = Eigen::VectorXd::Zero(np);
    int nfin = 0;
    for (size_t wi = 0; wi < work.size(); ++wi)
      if (okf[wi]) {
        Lfin += Lf[wi];
        gfin += gf[wi];
        nfin++;
      }
    if (nfin >= 1) {
      // fold the seed priors exactly as the accepted-pass reduction does
      int off = 0;
      for (size_t bi = 0; bi < layout.size(); ++bi) {
        const Eigen::VectorXd dloc = local_dev(bi);
        for (int k = 0; k < layout[bi].lsize; ++k) {
          const double info = 1.0 / (rep.prior_sigma_vec(off + k) * rep.prior_sigma_vec(off + k));
          Lfin(off + k, off + k) += info;
          gfin(off + k) += info * dloc(k);
        }
        off += layout[bi].lsize;
      }
      accepted_L = Lfin;
      accepted_g = gfin;
      if (cfg.verbose)
        std::printf("[joint] P4 finalize: %d windows tight at the accepted point\n", nfin);
    }
  }
  if (warm_out)
    *warm_out = warm_acc; // == nuisance optima at accepted_p (promotion contract above)
  rep.windows_dead = (int)std::count(dead.begin(), dead.end(), (char)1);
  rep.wall_s = elapsed_s();
  rep.accepted_passes = accepted_steps;
  rep.dim_p = np;
  if (std::isfinite(prev_merit))
    rep.final_merit = prev_merit;
  for (size_t wi = 0; wi < work.size(); ++wi)
    if (!dead[wi])
      rep.qn_max_final = std::max(rep.qn_max_final, qn_acc[wi]);
  // ---- carry write-back (accepted point only; pre-linearization failures
  // leave the container untouched — the stale stamp self-arbitrates at the
  // next consume, e.g. the A1b-failure restore re-matching the A1a stamp) ----
  if (carry_on && have_lin) {
    carry->p_stamp = full_stamp(); // calib holds accepted_p (restored above); frozen blocks never moved
    carry->noise_stamp = entry_noise;
    carry->layout_sig = layout_sig;
    carry->warm = warm_acc;
    carry->seeds = seeds_acc;
    carry->cost = cost_acc;
    carry->qn_ref = qn_ref;
    carry->valid = true;
  }
  if (cfg.verbose) {
    std::printf("[joint] done: %d accepted / %d passes, wall %.2fs | thread-cpu: seed %.2f preint %.2f inner %.2f (%ld iters) export %.2f\n",
                accepted_steps, rep.evaluation_passes, rep.wall_s, rep.t_seed_sum, rep.t_preint_sum, rep.t_inner_sum, rep.inner_iters_sum,
                rep.t_export_sum);
    std::printf("[joint] paths: warm %ld cold %ld (first %ld fail %ld jump %ld strand %ld cert %ld plateau %ld anchor %ld) cold-won %ld "
                "(guard %ld, mean gain %.3f%%) dual-agree %ld%s qn_max %.2e\n",
                rep.warm_evals, rep.cold_evals, rep.cold_first, rep.cold_warmfail, rep.cold_jump, rep.cold_strand, rep.cold_cert,
                rep.cold_plateau, rep.cold_anchor, rep.cold_won, rep.cold_won_guard,
                rep.cold_won ? 100.0 * rep.cold_gain_relsum / (double)rep.cold_won : 0.0, rep.cert_dual_confirms,
                rep.stopped_early ? " EARLY-STOP" : "", rep.qn_max_final);
  }
  if (!have_lin)
    return false; // nothing was ever accepted: no linearization, no posterior to ship

  Eigen::LDLT<Eigen::MatrixXd> ldlt(accepted_L);
  if (ldlt.info() != Eigen::Success)
    return false;
  const Eigen::MatrixXd Sigma = ldlt.solve(Eigen::MatrixXd::Identity(np, np));
  rep.sigma = Sigma.diagonal().cwiseMax(0.0).cwiseSqrt();
  rep.Lambda = accepted_L;
  rep.ok = true;
  return true;
}
