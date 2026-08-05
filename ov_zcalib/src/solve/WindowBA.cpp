/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: per-window micro-BA + reduced-information export (see WindowBA.h).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "WindowBA.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#include "ceres_free/LocalParameterization.h"
#include "ceres_free/Problem.h"
#include "ceres_free/State_JPLQuatLocal.h"

#include "Factor_ImuAci3.h"
#include "Factor_PriorDiag.h"
#include "Factor_ReprojTd.h"

#include "utils/quat_ops.h"

using namespace ov_zcalib;
using namespace ov_init::zbft_sfm;

namespace {
// Persistent per-window problem: the graph STRUCTURE (parameter
// registry, factor set, bindings) depends only on the fixed window content,
// so it is built ONCE and revisited with value/linearization RESETS -- the
// per-eval rebuild was the dominant remaining cost at 1-iteration fused
// evals. Parameter storage and an internal SharedCalib live here so every
// registered pointer stays stable across calls and calib objects.
struct WindowGraph {
  Problem problem;
  State_JPLQuatLocal quat_local;
  std::unique_ptr<GravityS2Parameterization> s2_local;
  CauchyLoss cauchy{2.0};
  std::vector<Eigen::Vector4d> q;
  std::vector<Eigen::Vector3d> bg, ba, v, p, feats;
  Eigen::Vector3d grav = Eigen::Vector3d::Zero();
  SharedCalib calib; // stable-address calib copy (values/flags mirrored per call)
  std::vector<CostFunction *> owned;
  std::vector<Factor_ImuAci3 *> imu_f;
  std::vector<Factor_ReprojTd *> rp_f;
  std::vector<int> rp_clone;
  Factor_PriorQuatJPL *pq = nullptr;
  Factor_PriorEuclid *pp = nullptr, *pbg = nullptr, *pba = nullptr;
  PreintKey factor_key;
  bool has_factor_key = false, built = false;
  ~WindowGraph() {
    for (auto *f : owned)
      delete f;
  }
};
} // namespace



bool WindowBA::solve_and_export(const WindowData &win, SharedCalib &calib, bool export_info, WindowSolveReport &rep, int max_iters,
                                bool verbose, WindowWarmState *warm, WindowPreint *pc, const WindowWarmState *state_at) {

  const int N = (int)win.clone_times.size();
  if (N < 3 || win.num_feats == 0)
    return false;

  // ---- persistent graph acquisition (one uniform path; pc==nullptr gets a
  // call-local graph so admission/tests behave exactly like production) ----
  std::unique_ptr<WindowGraph> local_g;
  WindowGraph *Gp;
  if (pc) {
    if (!pc->graph)
      pc->graph = std::shared_ptr<void>(new WindowGraph, [](void *g) { delete static_cast<WindowGraph *>(g); });
    Gp = static_cast<WindowGraph *>(pc->graph.get());
  } else {
    local_g.reset(new WindowGraph);
    Gp = local_g.get();
  }
  WindowGraph &G = *Gp;
  if (!G.built) {
    G.q.assign(N, Eigen::Vector4d(0, 0, 0, 1));
    G.bg.assign(N, Eigen::Vector3d::Zero());
    G.ba.assign(N, Eigen::Vector3d::Zero());
    G.v.assign(N, Eigen::Vector3d::Zero());
    G.p.assign(N, Eigen::Vector3d::Zero());
    G.feats.assign(win.num_feats, Eigen::Vector3d::Zero());
  }
  // ---- states (window frame {G} = first-clone IMU frame) ----
  std::vector<Eigen::Vector4d> &q = G.q;
  std::vector<Eigen::Vector3d> &bg = G.bg, &ba = G.ba;
  std::vector<Eigen::Vector3d> &v = G.v, &p = G.p;
  std::vector<Eigen::Vector3d> &feats = G.feats;
  std::vector<char> feat_seen(win.num_feats, 0);
  if (!(warm && warm->valid)) {
    // cold entry: reset carried values to the canonical cold init (the old
    // call-local initializers) so seed fallbacks start where they used to
    std::fill(q.begin(), q.end(), Eigen::Vector4d(0, 0, 0, 1));
    std::fill(bg.begin(), bg.end(), Eigen::Vector3d::Zero());
    std::fill(ba.begin(), ba.end(), Eigen::Vector3d::Zero());
    std::fill(v.begin(), v.end(), Eigen::Vector3d::Zero());
    std::fill(p.begin(), p.end(), Eigen::Vector3d::Zero());
    std::fill(feats.begin(), feats.end(), Eigen::Vector3d::Zero());
  }
  // calib values+flags mirrored into the graph's stable-address copy; the
  // solve reads/writes ONLY the copy, and calib itself is never mutated here
  // (the inner holds calibration constant; export frees the COPY's blocks).
  //
  // STABLE ADDRESS is load-bearing, and `cams` is a heap vector: the problem graph registers raw
  // pointers INTO its elements, so a reallocation here would leave every registered calibration
  // block dangling. Copy-assignment only reuses the buffer while the size is unchanged, which holds
  // because the camera count is fixed once at seed time. Assert it rather than trust it -- the
  // failure mode is silent memory corruption inside the solver.
  assert((!G.built || G.calib.cams.size() == calib.cams.size()) && "camera count changed under a built graph");
  G.calib = calib;

  // ---- ACI3 preintegration per interval at the current calibration ----
  // The intrinsic-column layout requires all three groups enabled; subset selection
  // happens through block constancy in the problem, not in the preintegration.
  ImuIntrinsicModel model_all = calib.imu;
  model_all.calib_dw = model_all.calib_da = model_all.calib_RAtoI = true;
  // SESSION-level switch, not the per-stage free flag: the column width must be stable across
  // stages (see SharedCalib::tg_enabled). Stage subsetting happens through block constancy.
  model_all.calib_tg = calib.tg_enabled;
  const auto t_pre0 = std::chrono::steady_clock::now();
  // Preint value key: preintegration is a function of (pi, noise-pi) and the fixed
  // window stream ONLY (see PreintCache.h); the key mirrors live pi into the
  // noise half when the whitener is unfrozen, exactly as integrate() would.
  PreintKey key;
  {
    int o = 0;
    auto put = [&](const double *p, int n) {
      std::memcpy(key.v + o, p, n * sizeof(double));
      o += n;
    };
    put(calib.imu.dw.data(), 6);
    put(calib.imu.da.data(), 6);
    put(calib.imu.q_AtoI.data(), 4);
    const ImuIntrinsicModel &nmk = calib.noise_frozen ? calib.noise_lin : calib.imu;
    put(nmk.dw.data(), 6);
    put(nmk.da.data(), 6);
    put(nmk.q_AtoI.data(), 4);
    // Tg joins the key when its columns exist: two calibrations differing only in Tg (or in
    // whether the 24-column layout is on at all) must never collide in the store. tg_enabled
    // itself is keyed as a slot so a 15-column entry can never serve a 24-column request.
    if (calib.tg_enabled) {
      put(calib.imu.Tg.data(), 9);
      put(nmk.Tg.data(), 9);
      const double on = 1.0;
      put(&on, 1);
    }
    // Noise sigmas feed P15 (the whitener): session-constant today, but an
    // unkeyed dependence is a silent-wrong-reuse landmine -- key them.
    const double sig[4] = {calib.noise.sigma_w, calib.noise.sigma_wb, calib.noise.sigma_a, calib.noise.sigma_ab};
    put(sig, 4);
    key.clone_t0 = win.clone_times.front();
    key.clone_t1 = win.clone_times.back();
    key.imu_t0 = win.imu.empty() ? 0.0 : win.imu.front().timestamp;
    key.imu_t1 = win.imu.empty() ? 0.0 : win.imu.back().timestamp;
    key.imu_n = (std::uint32_t)win.imu.size();
    key.clone_n = (std::uint32_t)win.clone_times.size();
    key.noise_frozen = calib.noise_frozen ? 1 : 0;
  }
  // OV_ZCALIB_PREINT_AUDIT: cross-check cached/chained preintegration against
  // fresh recomputation, abort on any byte mismatch (forensic; not for production).
  static const bool preint_audit = (std::getenv("OV_ZCALIB_PREINT_AUDIT") != nullptr);
  // Forensic bisection knobs (parity investigations only -- no production use):
  // OV_ZCALIB_P3_NOCACHE ignores the store (chain + per-factor whitener);
  // OV_ZCALIB_P3_LOOP uses the legacy per-interval integrate() scan uncached.
  static const bool p3_nocache = (std::getenv("OV_ZCALIB_P3_NOCACHE") != nullptr);
  static const bool p3_loop = (std::getenv("OV_ZCALIB_P3_LOOP") != nullptr);
  if (p3_nocache)
    pc = nullptr;
  std::vector<AciPreintResult> pre_local;
  const std::vector<AciPreintResult> *pre_p = nullptr;
  if (pc && pc->has_means && pc->mean_key == key && (int)pc->pre.size() == N - 1) {
    pre_p = &pc->pre; // bit-identical reuse (preint depends only on the key; audited below)
    rep.preint_hit = true;
    if (preint_audit) {
      std::vector<AciPreintResult> chk;
      if (!AciCalibPreint::integrate_chain(win.imu, win.clone_times, model_all, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                           calib.noise, chk, calib.noise_frozen ? &calib.noise_lin : nullptr))
        std::abort();
      for (int k = 0; k + 1 < N; ++k)
        if (!preint_bitwise_equal(chk[k], pc->pre[k])) {
          std::fprintf(stderr, "PREINT AUDIT FAILURE: cached bytes != recomputation (window uid %u interval %d)\n", win.uid, k);
          std::abort();
        }
    }
  } else if (pc) {
    pc->has_means = pc->has_whit = false; // key moved: the whitener tracks the mean key (never frozen separately)
    if (!AciCalibPreint::integrate_chain(win.imu, win.clone_times, model_all, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                         calib.noise, pc->pre, calib.noise_frozen ? &calib.noise_lin : nullptr))
      return false;
    pc->mean_key = key;
    pc->has_means = true;
    pre_p = &pc->pre;
  } else if (p3_loop) {
    pre_local.resize(N - 1);
    for (int k = 0; k + 1 < N; ++k)
      if (!AciCalibPreint::integrate(win.imu, win.clone_times[k], win.clone_times[k + 1], model_all, Eigen::Vector3d::Zero(),
                                     Eigen::Vector3d::Zero(), calib.noise, pre_local[k],
                                     calib.noise_frozen ? &calib.noise_lin : nullptr))
        return false;
    pre_p = &pre_local;
  } else {
    if (!AciCalibPreint::integrate_chain(win.imu, win.clone_times, model_all, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                         calib.noise, pre_local, calib.noise_frozen ? &calib.noise_lin : nullptr))
      return false;
    pre_p = &pre_local;
  }
  if (preint_audit && !rep.preint_hit) {
    // miss-path cross-check: the chain must equal the per-interval scan on the
    // REAL window (the unit oracle's synthetic battery cannot cover every log)
    for (int k = 0; k + 1 < N; ++k) {
      AciPreintResult ref;
      const bool rok = AciCalibPreint::integrate(win.imu, win.clone_times[k], win.clone_times[k + 1], model_all,
                                                 Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), calib.noise, ref,
                                                 calib.noise_frozen ? &calib.noise_lin : nullptr);
      if (!rok || !preint_bitwise_equal(ref, (*pre_p)[k])) {
        std::fprintf(stderr,
                     "PREINT CHAIN AUDIT FAILURE: window uid %u interval %d [%.9f, %.9f] imu_n %zu (rok %d)\n",
                     win.uid, k, win.clone_times[k], win.clone_times[k + 1], win.imu.size(), rok);
        std::abort();
      }
    }
  }
  const std::vector<AciPreintResult> &pre = *pre_p;
  rep.t_preint = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pre0).count();

  // ---- seeds ----
  Eigen::Vector3d &grav = G.grav;
  if (warm && warm->valid && (int)warm->q.size() == N && warm->feats.size() == win.num_feats) {
    q = warm->q;
    bg = warm->bg;
    v = warm->v;
    ba = warm->ba;
    p = warm->p;
    feats = warm->feats;
    grav = warm->grav;
  } else if (win.has_seeds) {
    // Bootstrap/harvester-provided nuisance seeds (window frame)
    q = win.seed_q;
    v = win.seed_v;
    p = win.seed_p;
    for (int k = 0; k < N; ++k) {
      bg[k] = win.seed_bg;
      ba[k] = win.seed_ba;
    }
    feats = win.seed_feats;
    grav = win.seed_grav;
  } else {
    // Fallback: gravity from the first corrected accel; dead-reckon poses from rest
    Eigen::Vector3d w0, a0;
    model_all.correct(win.imu.front().wm, win.imu.front().am, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), w0, a0);
    grav = calib.grav_mag * a0.normalized();
    for (int k = 0; k + 1 < N; ++k) {
      const double dt = pre[k].dt;
      const Eigen::Matrix3d R_k = ov_core::quat_2_Rot(q[k]);
      q[k + 1] = ov_core::quat_multiply(pre[k].q_KtoK1, q[k]);
      p[k + 1] = p[k] + v[k] * dt - 0.5 * grav * dt * dt + R_k.transpose() * pre[k].alpha;
      v[k + 1] = v[k] - grav * dt + R_k.transpose() * pre[k].beta;
    }
    for (int k = 0; k < N; ++k) {
      for (const CloneObs &o : win.obs[k]) {
        if (feat_seen[o.feat_id])
          continue;
        feat_seen[o.feat_id] = 1;
        // A landmark is anchored through the camera that SAW it -- with no shared field of view,
        // cam 1's points must never be back-projected through cam 0's extrinsic.
        const CamCalib &kc = calib.cams[(size_t)o.cam];
        const Eigen::Matrix3d R_ItoC = ov_core::quat_2_Rot(kc.q_ItoC);
        const double fx = kc.cam(0), fy = kc.cam(1), cx = kc.cam(2), cy = kc.cam(3);
        Eigen::Vector3d bearing_c((o.uv(0) - cx) / fx, (o.uv(1) - cy) / fy, 1.0);
        bearing_c.normalize();
        const Eigen::Matrix3d R_GtoI = ov_core::quat_2_Rot(q[k]);
        const double depth = 4.0;
        feats[o.feat_id] = R_GtoI.transpose() * (R_ItoC.transpose() * (depth * bearing_c - kc.p_IinC)) + p[k];
      }
    }
  }

  // ---- assemble (structure ONCE; constants re-pinned every call -- the
  // previous export freed the calib blocks) ----
  Problem &problem = G.problem;
  if (!G.built) {
    G.s2_local.reset(new GravityS2Parameterization(calib.grav_mag));
    problem.EnableOwnership(false); // factors owned by the graph
    for (int k = 0; k < N; ++k) {
      problem.AddParameterBlock(q[k].data(), 4, &G.quat_local);
      problem.AddParameterBlock(bg[k].data(), 3);
      problem.AddParameterBlock(v[k].data(), 3);
      problem.AddParameterBlock(ba[k].data(), 3);
      problem.AddParameterBlock(p[k].data(), 3);
    }
    problem.AddParameterBlock(grav.data(), 3, G.s2_local.get());
    problem.AddParameterBlock(G.calib.imu.dw.data(), 6);
    problem.AddParameterBlock(G.calib.imu.da.data(), 6);
    problem.AddParameterBlock(G.calib.imu.q_AtoI.data(), 4, &G.quat_local);
    if (G.calib.tg_enabled)
      problem.AddParameterBlock(G.calib.imu.Tg.data(), 9);
    for (CamCalib &kc : G.calib.cams) {
      problem.AddParameterBlock(kc.q_ItoC.data(), 4, &G.quat_local);
      problem.AddParameterBlock(kc.p_IinC.data(), 3);
      problem.AddParameterBlock(kc.cam.data(), 8);
      problem.AddParameterBlock(&kc.td, 1);
    }
    for (size_t f = 0; f < feats.size(); ++f) {
      problem.AddParameterBlock(feats[f].data(), 3);
      problem.SetSchurLandmark(feats[f].data());
    }
  }
  // Calibration blocks are nuisance-constant for the inner solve
  for (double *ptr : {G.calib.imu.dw.data(), G.calib.imu.da.data(), G.calib.imu.q_AtoI.data()})
    problem.SetParameterBlockConstant(ptr);
  if (G.calib.tg_enabled)
    problem.SetParameterBlockConstant(G.calib.imu.Tg.data());
  for (CamCalib &kc : G.calib.cams) {
    for (double *ptr : {kc.q_ItoC.data(), kc.p_IinC.data(), kc.cam.data(), &kc.td})
      problem.SetParameterBlockConstant(ptr);
  }

  // IMU factors. The whitener (two 15x15 factorizations per interval) is the
  // dominant construction cost, timed as t_factor. Cache discipline: a MISS
  // constructs the LEGACY factor and copies its sqrtI/fold members into the
  // cache -- never a re-derivation in another function/TU, whose codegen
  // under -ffast-math drifts 1 ulp (see the parity contract at the legacy
  // ctor). A HIT feeds the copied bytes to the cache-fed ctor, skipping the
  // factorization.
  const auto t_fac0 = std::chrono::steady_clock::now();
  const bool whit_hit = pc && pc->has_whit && pc->whit_key == key && (int)pc->W.size() == N - 1;
  if (pc && !whit_hit) {
    pc->W.resize(N - 1);
    pc->Wfold.resize(N - 1);
  }
  const bool imu_fresh = G.has_factor_key && G.factor_key == key && (int)G.imu_f.size() == N - 1;
  if (!G.built) {
    for (int k = 0; k + 1 < N; ++k) {
      Factor_ImuAci3 *f;
      if (whit_hit) {
        f = new Factor_ImuAci3(pre[k], model_all, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), pc->W[k], pc->Wfold[k]);
      } else {
        f = new Factor_ImuAci3(pre[k], model_all, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        if (pc) {
          pc->W[k] = f->sqrtI;
          pc->Wfold[k] = f->sqrtI_grav_fold;
        }
      }
      G.owned.push_back(f);
      G.imu_f.push_back(f);
      std::vector<double *> fparams = {q[k].data(),     bg[k].data(),     v[k].data(),
                                       ba[k].data(),    p[k].data(),      q[k + 1].data(),
                                       bg[k + 1].data(), v[k + 1].data(), ba[k + 1].data(),
                                       p[k + 1].data(), grav.data(),      G.calib.imu.dw.data(),
                                       G.calib.imu.da.data(), G.calib.imu.q_AtoI.data()};
      if (G.calib.tg_enabled)
        fparams.push_back(G.calib.imu.Tg.data());
      problem.AddResidualBlock(f, nullptr, fparams);
    }
  } else if (!imu_fresh) {
    // preint moved: rewrite the factors' measurement + whitener IN PLACE (the
    // registry binds by pointer; the members are the legacy ctor's exact set)
    for (int k = 0; k + 1 < N; ++k) {
      Factor_ImuAci3 *f = G.imu_f[k];
      f->m = pre[k];
      f->bg_lin.setZero();
      f->ba_lin.setZero();
      f->dw_lin = model_all.dw;
      f->da_lin = model_all.da;
      f->qA_lin = model_all.q_AtoI;
      f->tg_lin = model_all.Tg; // tg_on is construction-stable (session-level tg_enabled)
      if (whit_hit) {
        f->sqrtI = pc->W[k];
        f->sqrtI_grav_fold = pc->Wfold[k];
      } else {
        Factor_ImuAci3 fresh(pre[k], model_all, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        f->sqrtI = fresh.sqrtI;
        f->sqrtI_grav_fold = fresh.sqrtI_grav_fold;
        if (pc) {
          pc->W[k] = f->sqrtI;
          pc->Wfold[k] = f->sqrtI_grav_fold;
        }
      }
    }
  }
  G.factor_key = key;
  G.has_factor_key = true;
  if (pc && !whit_hit) {
    pc->whit_key = key;
    pc->has_whit = true;
  }
  rep.t_factor = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_fac0).count();
  // Reprojection factors (clone kinematics from the preintegration endpoints).
  // Structure is window-fixed; the kinematic transport linearization
  // (w_clone, v_clone) tracks the CURRENT preint/state every call, exactly as
  // the per-call construction did.
  if (!G.built) {
    for (int k = 0; k < N; ++k) {
      const Eigen::Vector3d w_k = (k == 0) ? pre[0].w_end : pre[k - 1].w_end;
      for (const CloneObs &o : win.obs[k]) {
        // Every observation reprojects through ITS OWN camera's blocks. This binding is the only
        // place the rig's multi-camera structure actually enters the geometry: the clones, the
        // landmarks, the gravity and the IMU chain are shared, and the cameras meet nowhere else.
        const size_t c = (size_t)o.cam;
        CamCalib &kc = G.calib.cams[c];
        // Rolling shutter enters HERE and only here: the centered row time u_frac * tr (tr = fixed
        // HAL3 readout, never a parameter) is a known constant of this observation, folded into the
        // factor's dt_ref beside the frame-merge offset.
        auto *f = new Factor_ReprojTd(o.uv, win.pix_sigma, calib.cams[c].fisheye, w_k, v[k], win.td_ref[c],
                                      o.dt_ref + o.u_frac * kc.tr);
        G.owned.push_back(f);
        G.rp_f.push_back(f);
        G.rp_clone.push_back(k);
        problem.AddResidualBlock(f, &G.cauchy,
                                 {q[k].data(), p[k].data(), feats[o.feat_id].data(), kc.q_ItoC.data(), kc.p_IinC.data(),
                                  kc.cam.data(), &kc.td});
      }
    }
  } else {
    for (size_t i = 0; i < G.rp_f.size(); ++i) {
      const int k = G.rp_clone[i];
      G.rp_f[i]->w_clone = (k == 0) ? pre[0].w_end : pre[k - 1].w_end;
      G.rp_f[i]->v_clone = v[k];
    }
  }
  // Gauge priors: first pose + first biases (window-frame gauge; NO calibration
  // priors here). The bias priors are PHYSICAL, not gauge: the per-window
  // gravity <-> accel-bias valley admits a family of (grav, ba, v) splits at
  // near-equal cost, and where a window sits on that floor is set by its seed --
  // the exported (Lambda, g) then carries the seed, not the data, and the
  // fused optimum inherits it. Spec-sheet bias scales collapse the family to a
  // unique split (same role as the seeder's own ba Tikhonov).
  // The bias-prior MEANS must be anchored at the window's (p-independent)
  // seed values, NOT at the current state vector: under VarPro warm starts the
  // state at construction is the PREVIOUS pass's optimum, and a prior centered
  // there chases the estimate -- the gravity<->ba valley collapse this prior
  // exists for silently disengages and da/qA drift along the valley floor.
  const Eigen::Vector3d bg_anchor = win.has_seeds ? win.seed_bg : bg[0];
  const Eigen::Vector3d ba_anchor = win.has_seeds ? win.seed_ba : ba[0];
  const Eigen::Vector4d q_anchor = win.has_seeds ? win.seed_q[0] : q[0];
  const Eigen::Vector3d p_anchor = win.has_seeds ? win.seed_p[0] : p[0];
  if (!G.built) {
    G.pq = new Factor_PriorQuatJPL(q_anchor, Eigen::Vector3d::Constant(1e-4));
    G.pp = new Factor_PriorEuclid(p_anchor, Eigen::Vector3d::Constant(1e-4));
    G.pbg = new Factor_PriorEuclid(bg_anchor, Eigen::Vector3d::Constant(calib.bg_prior_sigma));
    G.pba = new Factor_PriorEuclid(ba_anchor, Eigen::Vector3d::Constant(calib.ba_prior_sigma));
    G.owned.insert(G.owned.end(), {G.pq, G.pp, G.pbg, G.pba});
    problem.AddResidualBlock(G.pq, nullptr, {q[0].data()});
    problem.AddResidualBlock(G.pp, nullptr, {p[0].data()});
    problem.AddResidualBlock(G.pbg, nullptr, {bg[0].data()});
    problem.AddResidualBlock(G.pba, nullptr, {ba[0].data()});
    G.built = true;
  } else {
    G.pq->q0 = q_anchor;
    G.pp->x0 = p_anchor;
    G.pbg->x0 = bg_anchor;
    G.pba->x0 = ba_anchor;
  }

  // ---- export-only state override (export-on-accept re-entry) ----
  // Everything ABOVE ran from the ENTRY state (warm strand or cold seeds),
  // exactly as the evaluation that produced this optimum did: the reprojection
  // transport linearization (w_clone/v_clone) and the gauge anchors are now
  // pinned at the evaluation's values. Only NOW load the kept optimum itself,
  // so the (zero-iteration) export below linearizes at it. Overriding any
  // earlier would move the transport linearization off the evaluation's and
  // break the deferred-export == inline-export byte contract.
  if (state_at) {
    if (!state_at->valid || (int)state_at->q.size() != N || state_at->feats.size() != win.num_feats)
      return false; // a mis-shaped override would silently export a wrong point: fail loudly
    q = state_at->q;
    bg = state_at->bg;
    v = state_at->v;
    ba = state_at->ba;
    p = state_at->p;
    feats = state_at->feats;
    grav = state_at->grav;
  }

  // ---- inner (nuisance) solve ----
  SolverOptions opts;
  opts.max_num_iterations = max_iters;
  // Hang guard ONLY -- must never bind on a healthy solve. A binding wall cap
  // couples machine load into the ITERATE: a time-stopped inner solve returns
  // a different point, duel arbitration flips, and the session diverges
  // run-to-run (measured: 1-ulp committed-YAML drift between quiet and loaded
  // runs of the SAME binary under a 5.0 s cap; split-half falsifier stats
  // shifted). 60 s is ~25-50x the worst measured inner solve; any firing is
  // surfaced as time_stopped and flagged by the evidence table.
  opts.max_solver_time_seconds = 60.0;
  opts.num_threads = 1;
  opts.verbose = verbose;
  // Tight exits: the VarPro outer loop differences window costs across p, so
  // the inner candidate-exit noise IS the outer's merit-function noise floor.
  // Loose (Ceres-default) tolerances leave ~0.5% hysteresis -- larger than a
  // typical outer step's true improvement. Warm starts keep the extra
  // iterations cheap.
  opts.function_tolerance = 1e-10;
  opts.parameter_tolerance = 1e-12;
  // The free-S2 gravity <-> accel-bias valley strands the Ceres-exact lambda
  // schedule partway down from cold (linear-seed) inits: lambda crashes ~5
  // decades in the easy phase, then the climb-back eats the trial budget and
  // the candidate exit fires mid-valley (the documented regime in
  // SolverOptions). Bounding the crash + faster escalation reaches the basin
  // floor; VarPro differences window costs, so stranded inner points shift
  // the fused optimum by far more than these knobs cost in wall clock.
  opts.min_lambda = 1e-7;
  opts.lm_nu_growth = 4.0;
  const auto t_in0 = std::chrono::steady_clock::now();
  if (max_iters > 0) {
    SolverSummary sum = problem.Solve(opts);
    rep.cost_final = sum.final_cost;
    rep.iterations = sum.iterations;
    rep.inner_converged = sum.converged;
    rep.time_stopped = sum.time_stopped;
  } else {
    // export-only re-entry (max_iters=0, export-on-accept): the state was
    // placed at the kept optimum above; a Solve(0) would only pay one wasted
    // nuisance linearization (the export re-linearizes with the calib columns
    // freed anyway). No cost/convergence claim is made -- the consumers of
    // this call read only the export products.
    rep.cost_final = 0.0;
    rep.iterations = 0;
    rep.inner_converged = false;
    rep.time_stopped = false;
  }
  rep.t_inner = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_in0).count();
  if (warm) {
    warm->q = q;
    warm->bg = bg;
    warm->v = v;
    warm->ba = ba;
    warm->p = p;
    warm->feats = feats;
    warm->grav = grav;
    warm->valid = true;
  }

  // ---- export reduced information on the free calibration blocks ----
  // The keep set and the flag flips act on the GRAPH's calib copy (the
  // registered pointers); the caller's calib is never touched here.
  rep.free_dim = G.calib.local_dim(); // layout dim of THIS solve (eval-only reports carry no Lambda)
  rep.ok = true;
  if (export_info) {
    const auto t_ex0 = std::chrono::steady_clock::now();
    std::vector<double *> keep;
    for (auto &b : G.calib.free_blocks()) {
      problem.SetParameterBlockVariable(b.ptr);
      keep.push_back(b.ptr);
    }
    ov_init::zbft_sfm::Problem::ExportStats est;
    rep.ok = problem.ExportReducedInformation(keep, rep.Lambda, rep.gred, opts, &est);
    rep.qn = est.nuis_decrement;
    rep.gn_inf = est.nuis_grad_inf;
    rep.export_min_pivot = est.nuis_min_pivot;
    rep.export_nuis_dim = est.nuis_dim;
    rep.export_clamped = est.clamped_dirs;
    for (auto &b : G.calib.free_blocks())
      problem.SetParameterBlockConstant(b.ptr); // leave calib untouched by this window
    rep.t_export = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ex0).count();
  }
  // (A qn-only stats path without the calib columns was built and measured NOT
  // byte-equal to the full export's q_n -- the smaller H's leading dimension
  // changes SIMD head-peeling under -ffast-math and H_zz drifts 1 ulp. See the
  // note at Problem::ExportReducedInformation. Cert-consuming evaluations must
  // therefore export fully; JointCalib owns that policy.)
  return rep.ok;
}
