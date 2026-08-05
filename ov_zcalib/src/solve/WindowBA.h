/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: per-window targetless micro-BA (nuisance solve) + reduced-information
 * export on the shared calibration block (the VarPro inner stage).
 *
 * Window states: per clone {q_GtoIk, bg_k, v_k, ba_k, p_k} in the window frame
 * {G} = first-clone IMU frame, features p_FinG (Schur landmarks), gravity on S2.
 * Factors: Factor_ImuAci3 between consecutive clones, Factor_ReprojTd per
 * observation, tiny gauge priors on the first pose and first biases. NO
 * calibration priors here — those are applied ONCE at fusion (JointCalib), per
 * the prior-double-counting rule.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_WINDOW_BA_H
#define OV_ZCALIB_WINDOW_BA_H

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "../types/ImuIntrinsicModel.h"
#include "../cpi/AciCalibPreint.h"
#include "../cpi/PreintCache.h"

namespace ov_zcalib {

/**
 * @brief Everything ONE camera owns. A rig carries N of these and exactly one IMU.
 *
 * The cameras of an asynchronous, non-overlapping rig share no landmark and no shutter, so these
 * blocks are genuinely independent of one another: cam 0's extrinsic never appears in a residual
 * that cam 1's extrinsic also appears in. What couples them is the IMU -- the shared inertial chain
 * and the single trajectory it drives through the window. That is the whole reason a joint solve
 * beats two separate ones: every camera's reprojections tighten the SAME trajectory, and the
 * accel-intrinsic directions that one viewing geometry leaves flat are exactly the ones another
 * viewing geometry pins down.
 */
struct CamCalib {
  Eigen::Vector4d q_ItoC = Eigen::Vector4d(0, 0, 0, 1);
  Eigen::Vector3d p_IinC = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 8, 1> cam = (Eigen::Matrix<double, 8, 1>() << 450, 450, 640, 400, 0, 0, 0, 0).finished();
  bool fisheye = false;
  double td = 0.0; ///< camera-IMU time offset (this camera's own; they are NOT equal across cams)
  double tr = 0.0; ///< rolling-shutter readout (0 on a global-shutter camera)
  int img_w = 640, img_h = 480;
  /// Declared frame rate [Hz], 0 = unknown. NOT an estimated quantity -- a hardware fact, sourced
  /// from the estimator chain's cam_N_fps (there is exactly one place a rig declares its cadence,
  /// and a second one would only drift from it). The harvester seeds each camera's period from
  /// this so the first window's drop accounting is right instead of learned.
  double fps = 0.0;
  /// Declared shutter, from the chain's cam_N_shutter. A GLOBAL-shutter camera exposes every row at
  /// the same instant: it has no readout time, so tr is not a quantity that exists to be estimated.
  /// Freeing it anyway would hand the solver a parameter with no signal, which it would happily
  /// fill with noise -- and tr aliases into td, so that noise would land in the time offset. free_tr
  /// is forced off for any camera that is not rolling.
  bool rolling = false;
  bool free_ext = true; ///< extrinsics q_ItoC, p_IinC
  bool free_td = true;  ///< camera-IMU time offset
  bool free_tr = false; ///< rolling-shutter readout
  int cam_mode = 0;     ///< 0 fixed | 1 refine (priors) | 2 full (weak priors)
};

/**
 * @brief The shared calibration state p (one instance across all windows).
 */
struct SharedCalib {
  ImuIntrinsicModel imu; ///< dw / da / q_AtoI / Tg values + subset flags double as free-flags
  /// SESSION-level Tg column switch. When set, every preintegration in the session carries the
  /// 24-column intrinsic layout (dw6|da6|thA3|tg9) and the window graphs are built with the tg
  /// parameter block — the per-STAGE imu.calib_tg flag then only decides whether the block is in
  /// free_blocks() or held constant, exactly the block-constancy doctrine dw/da/thA follow. This
  /// split keeps the column width stable across stages (persistent window graphs cannot change a
  /// factor's parameter list in place) and keeps tg-off sessions BYTE-IDENTICAL to the validated
  /// corpus (n_pi stays 15; under -ffast-math even zero-columns reassociate the reductions).
  bool tg_enabled = false;
  /// The rig's cameras. SIZED ONCE at seed time and never resized afterwards: free_blocks() hands
  /// out raw pointers INTO these elements, and the solver holds them across the whole fusion.
  std::vector<CamCalib> cams = std::vector<CamCalib>(1);
  double grav_mag = 9.81;
  /// Per-window first-bias prior scales (PHYSICAL, not gauge): tight when the
  /// seed bias comes from a still baseline, wider when it is visual-only.
  double bg_prior_sigma = 0.01;
  double ba_prior_sigma = 0.05;
  ImuNoise noise;
  /// Frozen noise-linearization model (set once at fusion entry; see AciCalibPreint)
  ImuIntrinsicModel noise_lin;
  bool noise_frozen = false;

  int n_cams() const { return (int)cams.size(); }

  /// Ordered free-block layout (pointers into THIS object) for export/fusion.
  struct BlockRef {
    double *ptr;
    int gsize, lsize;
    bool is_quat;
    /// The PHYSICAL name ("td", "da", ...), never suffixed with a camera. Every name-keyed policy
    /// map (prior_sigma, step_cap, commit ceilings) and every observability gate is written against
    /// these, so they must stay stable as cameras are added: a per-camera name would silently empty
    /// the Wald gate's subspace, which selects on "da[1]"/"q_AtoI" by string.
    std::string name;
    int cam; ///< -1 = shared (the IMU chain); otherwise the camera this block belongs to
    /// Display/provenance identity: unique across cameras, unlike name.
    std::string label() const { return cam < 0 ? name : name + "@" + std::to_string(cam); }
  };

  std::vector<BlockRef> free_blocks() {
    std::vector<BlockRef> v;
    // The shared IMU chain comes FIRST and stays unsuffixed -- see BlockRef::name.
    if (imu.calib_dw)
      v.push_back({imu.dw.data(), 6, 6, false, "dw", -1});
    if (imu.calib_da)
      v.push_back({imu.da.data(), 6, 6, false, "da", -1});
    if (imu.calib_RAtoI)
      v.push_back({imu.q_AtoI.data(), 4, 3, true, "q_AtoI", -1});
    // Matrix3d STORAGE order (column-major). mixing() enumerates pi in ROW-major order (its
    // legacy shape is parity-frozen); the factor reconciles the two at its boundary — see
    // Factor_ImuAci3::evaluate_tg_. Everything outside the factor (priors, caps, split bands,
    // record, writeback) is per-element symmetric, so storage order is safe here.
    if (imu.calib_tg)
      v.push_back({imu.Tg.data(), 9, 9, false, "tg", -1});
    for (int c = 0; c < (int)cams.size(); ++c) {
      CamCalib &k = cams[(size_t)c];
      if (k.free_ext) {
        v.push_back({k.q_ItoC.data(), 4, 3, true, "q_ItoC", c});
        v.push_back({k.p_IinC.data(), 3, 3, false, "p_IinC", c});
      }
      if (k.free_td)
        v.push_back({&k.td, 1, 1, false, "td", c});
      if (k.free_tr)
        v.push_back({&k.tr, 1, 1, false, "tr", c});
      if (k.cam_mode > 0)
        v.push_back({k.cam.data(), 8, 8, false, "cam", c});
    }
    return v;
  }
  int local_dim() {
    int n = 0;
    for (auto &b : free_blocks())
      n += b.lsize;
    return n;
  }
};

/// One feature observation at a clone
struct CloneObs {
  size_t feat_id = 0;
  Eigen::Vector2d uv = Eigen::Vector2d::Zero();
  double u_frac = 0.5; ///< row fraction (rolling shutter)
  /// Which camera saw it -- i.e. which CamCalib block this residual reprojects through. A track
  /// never crosses cameras (ov_core hands out feature ids from one atomic counter, and the rigs
  /// here have no shared field of view), so this is constant along a track.
  int cam = 0;
  /// Offset [s] from THIS observation's own seed-mapped sampling instant to its clone's timestamp,
  /// at the seed calibration. Zero for the frame that DEFINED the clone -- which is the common case,
  /// because a clone IS a frame. It is nonzero only when two cameras fire close enough together
  /// that their frames were merged into a single clone rather than left as two clones a hair apart
  /// (which would hand the preintegration a ~0 s interval). The reprojection transports across it.
  double dt_ref = 0.0;
  /// Unit bearing in the camera frame, undistorted at the SEED intrinsics.
  /// Consumed only by the linear seeder / bootstrap (the MLE re-projects from
  /// uv through the live cam block); filled by the harvester or sim.
  Eigen::Vector3d bearing = Eigen::Vector3d(0, 0, 1);
};

/// One harvested window (raw IMU + per-clone feature tracks; no images retained)
struct WindowData {
  std::vector<RawImu> imu;
  std::vector<double> clone_times;         ///< frame stamps mapped to IMU clock at the seed td
  std::vector<std::vector<CloneObs>> obs;  ///< per clone
  size_t num_feats = 0;
  double pix_sigma = 1.0;
  /// Session-unique id assigned by the harvester at assembly (monotone from 1;
  /// 0 = directly-built window, never cached). Survives every copy — the
  /// PreintStore entry index (see PreintCache.h).
  std::uint32_t uid = 0;
  /// Temporal reference at which clone_times were laid down (harvest time), PER CAMERA — each
  /// camera has its own time offset, so each has its own linearization point. The reprojection
  /// transport applies Delta = dt_ref + (td[c] - td_ref[c]) + (tr[c] - tr_ref[c])*u — the TOTAL
  /// shift since harvest, NOT the shift since the last outer relinearization (using the current td
  /// as the reference zeroes the value transport forever: the false-stationary-point bug).
  ///
  /// td and tr are NOT symmetric here, and the asymmetry is physical. A clone is stamped at a
  /// per-FRAME instant (t_sof + exposure/2 + td_seed) which already CONTAINS td_seed, so td must be
  /// transported as a DEVIATION from it. That stamp contains no per-ROW term, so the readout must be
  /// transported ABSOLUTELY: tr_ref is structurally ZERO (see WindowHarvester), and Delta carries the
  /// full u*tr. Seeding tr_ref instead made the transport vanish whenever tr was pinned at its seed,
  /// which left the rolling shutter unmodelled and pushed ~tr/2 into td (D10).
  ///
  /// Defaults to ONE camera at a zero reference, so a hand-built single-camera window (the sim
  /// harness, the unit tests) is valid on construction. Indexed by CloneObs::cam, which defaults to
  /// 0 to match — an empty vector here would be an out-of-bounds read on the first residual.
  std::vector<double> td_ref{0.0}, tr_ref{0.0};
  /// Optional nuisance seeds (window frame), provided by the harvester/bootstrap
  /// (in production the Dong-Si linear initializer fills these; the sim harness
  /// fills them from perturbed truth so the gates isolate CALIBRATION recovery).
  bool has_seeds = false;
  std::vector<Eigen::Vector4d> seed_q;
  std::vector<Eigen::Vector3d> seed_v, seed_p;
  Eigen::Vector3d seed_bg = Eigen::Vector3d::Zero(), seed_ba = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> seed_feats;
  Eigen::Vector3d seed_grav = Eigen::Vector3d::Zero();
};

struct WindowSolveReport {
  bool ok = false;
  double cost_final = 0.0;
  int iterations = 0;
  Eigen::MatrixXd Lambda; ///< reduced information on the free calib blocks
  Eigen::VectorXd gred;   ///< matching reduced gradient
  // wall-clock split of this call (solver-time program instrumentation)
  double t_preint = 0.0, t_inner = 0.0, t_export = 0.0;
  // P1 stationarity-certificate evidence (see JointCalib): inner_converged is
  // TRUE only for genuine stationarity exits (ftol/ptol/gradient/max-damping/
  // TR-collapse) — iteration-cap and wall-cap exits are NOT converged; qn is
  // the nuisance Newton decrement of the export linearization.
  bool inner_converged = false;
  /// True iff the wall-clock hang guard (not iterations/tolerances) ended the
  /// inner solve: the iterate is then LOAD-COUPLED — not replay-deterministic —
  /// and the session evidence table flags the whole run as tainted for A/Bs.
  bool time_stopped = false;
  double qn = 0.0;
  double gn_inf = 0.0;
  // P3 preint cache evidence: whether this call reused the window's cached
  // preintegration, and the IMU-factor construction time (whitener build /
  // fetch) — a cost that previously fell UNTIMED between t_preint and t_inner.
  bool preint_hit = false;
  double t_factor = 0.0;
};

/// Per-window nuisance state carried ACROSS outer iterations (VarPro warm
/// start). Cold inner solves from the original seeds make the summed window
/// cost discontinuous in p (inner-solver hysteresis) — the outer loop's
/// monotonicity guard then rejects noise, not steps. Warm starts make the
/// reduced cost continuous and cut inner iterations several-fold.
struct WindowWarmState {
  bool valid = false;
  std::vector<Eigen::Vector4d> q;
  std::vector<Eigen::Vector3d> bg, v, ba, p;
  std::vector<Eigen::Vector3d> feats;
  Eigen::Vector3d grav = Eigen::Vector3d::Zero();
};

class WindowBA {
public:
  /**
   * @brief Solve the window nuisances at the CURRENT calibration (calib blocks held
   *        constant), then (optionally) free them and export the reduced information.
   * @param warm optional in/out nuisance cache (see WindowWarmState)
   * @param pc optional preintegration cache entry for THIS window (PreintCache.h):
   *        value-keyed on the exact (pi, noise-pi) bytes — a hit reuses the stored
   *        preintegration + whitener (bit-identical to recomputation), a miss
   *        refills the entry. nullptr = legacy per-call recomputation.
   */
  static bool solve_and_export(const WindowData &win, SharedCalib &calib, bool export_info, WindowSolveReport &rep,
                               int max_iters = 30, bool verbose = false, WindowWarmState *warm = nullptr,
                               WindowPreint *pc = nullptr);
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_WINDOW_BA_H
