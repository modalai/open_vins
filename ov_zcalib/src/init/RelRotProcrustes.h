/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * ov_zcalib: frame-to-frame relative camera rotation from bearing pairs.
 * Rotation-only Procrustes (Wahba/Markley SVD) under a deterministic 3-point
 * RANSAC: the rng is seeded from the frame timestamp so live and replay runs
 * produce bit-identical hypotheses (the live-vs-replay parity gate depends on
 * it). Translation-induced flow biases a rotation-only fit; the consumers
 * (hand-eye Wahba, xcorr) carry their own Hampel/ratio gates and the joint
 * MLE re-estimates everything downstream — this is a SEED source only.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_REL_ROT_PROCRUSTES_H
#define OV_ZCALIB_REL_ROT_PROCRUSTES_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace ov_zcalib {

/**
 * @brief Rotation-only relative rotation between two bearing sets.
 *
 * Solves R = argmin sum_i || b2_i - R b1_i ||^2 (so R maps frame-1 bearings
 * into frame 2, i.e. {}^{C2}_{C1}R for camera frames) via the SVD of
 * H = sum b1 b2^T with the Markley det-correction, under a deterministic
 * 3-point RANSAC with an inlier refit.
 */
class RelRotProcrustes {
public:
  struct Options {
    int min_pairs = 12;               ///< below this the pair is rejected
    int ransac_iterations = 48;
    double inlier_threshold_rad = 0.008; ///< angular residual gate per bearing
    double min_inlier_ratio = 0.5;
  };

  struct Result {
    bool ok = false;
    Eigen::Matrix3d R_C1toC2 = Eigen::Matrix3d::Identity();
    int inliers = 0;
    double inlier_ratio = 0.0;
    /// Mean inlier angular residual AFTER derotation [rad]: translation flow +
    /// noise. The errors-in-variables weight for hand-eye consumers — pairs
    /// where this rivals |theta| carry a rotation-only bias, not information.
    double mean_resid_rad = 0.0;
  };

  /// Procrustes on all given pairs (no gating). b1/b2 must be unit bearings.
  static Eigen::Matrix3d procrustes(const std::vector<Eigen::Vector3d> &b1, const std::vector<Eigen::Vector3d> &b2) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < b1.size(); ++i)
      H += b1[i] * b2[i].transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    D(2, 2) = ((svd.matrixV() * svd.matrixU().transpose()).determinant() < 0.0) ? -1.0 : 1.0;
    return svd.matrixV() * D * svd.matrixU().transpose();
  }

  /**
   * @brief RANSAC + inlier refit. @param rng_seed pass the frame timestamp in
   *        nanoseconds (or any replay-stable integer) — NEVER a wall clock.
   */
  static Result solve(const std::vector<Eigen::Vector3d> &b1, const std::vector<Eigen::Vector3d> &b2, uint64_t rng_seed) {
    return solve(b1, b2, rng_seed, Options());
  }
  static Result solve(const std::vector<Eigen::Vector3d> &b1, const std::vector<Eigen::Vector3d> &b2, uint64_t rng_seed,
                      const Options &opt) {
    Result out;
    const int N = (int)b1.size();
    if (N < opt.min_pairs)
      return out;
    std::mt19937 rng((unsigned)(rng_seed & 0xFFFFFFFFu));
    const double cos_gate = std::cos(opt.inlier_threshold_rad);

    Eigen::Matrix3d best_R = Eigen::Matrix3d::Identity();
    int best_inl = -1;
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<Eigen::Vector3d> s1(3), s2(3);
    for (int it = 0; it < opt.ransac_iterations; ++it) {
      for (int s = 0; s < 3; ++s) {
        std::uniform_int_distribution<int> dist(s, N - 1);
        std::swap(idx[s], idx[dist(rng)]);
        s1[s] = b1[idx[s]];
        s2[s] = b2[idx[s]];
      }
      const Eigen::Matrix3d R = procrustes(s1, s2);
      int inl = 0;
      for (int i = 0; i < N; ++i)
        if ((R * b1[i]).dot(b2[i]) > cos_gate)
          ++inl;
      if (inl > best_inl) {
        best_inl = inl;
        best_R = R;
      }
      if (best_inl > N * 95 / 100)
        break;
    }
    if (best_inl < std::max(opt.min_pairs, (int)(opt.min_inlier_ratio * N)))
      return out;

    std::vector<Eigen::Vector3d> i1, i2;
    i1.reserve(best_inl);
    i2.reserve(best_inl);
    for (int i = 0; i < N; ++i)
      if ((best_R * b1[i]).dot(b2[i]) > cos_gate) {
        i1.push_back(b1[i]);
        i2.push_back(b2[i]);
      }
    out.R_C1toC2 = procrustes(i1, i2);
    out.inliers = (int)i1.size();
    out.inlier_ratio = (double)out.inliers / (double)N;
    double rsum = 0.0;
    for (size_t i = 0; i < i1.size(); ++i)
      rsum += std::acos(std::min(1.0, std::max(-1.0, (out.R_C1toC2 * i1[i]).dot(i2[i]))));
    out.mean_resid_rad = rsum / std::max<size_t>(i1.size(), 1);
    out.ok = true;
    return out;
  }
};

/**
 * @brief Translation-AWARE relative rotation via essential-matrix decomposition
 * (8-point on unit bearings + cheirality), deterministic RANSAC discipline.
 *
 * Rotation-only Procrustes carries a translation-induced bias ~ (baseline /
 * depth) per pair: negligible at far range, 30-100 mrad per frame pair on
 * close-range handheld data (measured on the kalibr sample at 0.3-0.5 m) —
 * E-decomposed rotations are the primary path for this reason.
 * Falls back to Procrustes when E is degenerate (pure rotation / low inliers),
 * which is exactly the regime where Procrustes is unbiased.
 *
 * Why ESSENTIAL and not FUNDAMENTAL: R is extractable only from E; F would
 * have to be mapped through the intrinsics anyway (E = K2^T F K1), so
 * estimating F here just adds 2 noise-absorbing DOF (7 vs 5) for zero
 * information gain when a calibration seed exists — and under real radtan
 * distortion (k1 ~ -0.28) raw-pixel epipolar lines are CURVES, so the F
 * model is violated unless points are undistorted first, at which point the
 * problem IS the calibrated one. Operating on unit bearings is also the
 * exact analogue of Hartley normalization (conditioning), keeps the RANSAC
 * threshold ANGULAR (resolution/lens-agnostic, fisheye-compatible through
 * the same interface), and the seed-intrinsics error (~1.5 px / 1e-3 k)
 * perturbs bearings by ~0.1% — far below KLT epipolar noise, and the joint
 * MLE re-estimates the intrinsics downstream regardless. F earns its keep
 * only with UNKNOWN intrinsics (self-calibration), which contradicts this
 * module's existing-cal seed contract.
 *
 * PLANAR scenes (a target wall filling the view) make the LINEAR 8-point
 * degenerate for E — and F inherits it worse (the plane-homography ambiguity
 * family). The implemented cure is the third model of the classic family:
 * the calibrated HOMOGRAPHY H = R + t n^T (DLT on bearings, Faugeras/MASKS
 * SVD decomposition, visibility + cheirality disambiguation), which is the
 * EXACT model for planar scenes and subsumes pure rotation (H -> R as t -> 0).
 * Nister 5-point was considered and deliberately skipped: on a true plane the
 * homography is exact while 5-point merely tolerates planarity, and a
 * Groebner-basis polynomial solver is a large delicate surface for a regime
 * H already covers; the selection rule below only trusts E where the scene
 * gives it non-planar support. Selection is Torr/Pollefeys-style:
 *   R-only  (3 dof) when the rotation-only residual sits at the noise floor;
 *   H       (8 dof) when H explains ~= as many inliers as E (planar or
 *                   rotation-dominant: E's R is untrustworthy exactly there);
 *   E       (5 dof) otherwise (general 3D scene with parallax).
 * Thresholds compare like with like: E's point-to-PLANE residual (codim 1)
 * vs H's point TRANSFER residual (codim 2) at sqrt(2) x the angular gate.
 */
class RelRotEssential {
public:
  struct Options {
    int min_pairs = 12;
    int ransac_iterations = 64;
    double inlier_threshold_rad = 0.004; ///< angular distance of b2 to the epipolar plane
    double min_inlier_ratio = 0.5;
    /// planar-degeneracy rule: E's rotation is trustworthy only with enough
    /// OFF-PLANE support — count E-inliers the best homography REJECTS. A raw
    /// inlier-count ratio is too eager on near-planar real scenes (a dominant
    /// plane explains most of E's inliers even when the off-plane background
    /// fully determines E; measured: eager H cost 0.3 deg of hand-eye on the
    /// kalibr sample). Below this support count, the scene is effectively
    /// planar for E and H's rotation is used instead.
    int min_offplane_support = 12;
    bool enable_homography = true;
  };

  enum Model { MODEL_PROCRUSTES = 0, MODEL_ESSENTIAL = 1, MODEL_HOMOGRAPHY = 2 };

  struct Result {
    bool ok = false;
    Eigen::Matrix3d R_C1toC2 = Eigen::Matrix3d::Identity();
    int inliers = 0;
    double inlier_ratio = 0.0;
    double mean_resid_rad = 0.0; ///< mean inlier residual of the SELECTED model
    int model = MODEL_PROCRUSTES;
    bool used_essential = false; ///< kept in sync: model == MODEL_ESSENTIAL
  };

  /// b2^T E b1 residual as an angle: |b2 . n| with n = unit(E b1) (b2 off-plane angle)
  static double epi_ang(const Eigen::Matrix3d &E, const Eigen::Vector3d &b1, const Eigen::Vector3d &b2) {
    const Eigen::Vector3d n = E * b1;
    const double nn = n.norm();
    if (nn < 1e-12)
      return 0.0;
    return std::asin(std::min(1.0, std::abs(b2.dot(n)) / nn));
  }

  /// Linear E fit (|fit set| >= 8) projected onto the essential manifold.
  static bool fit_E(const std::vector<Eigen::Vector3d> &b1, const std::vector<Eigen::Vector3d> &b2, const std::vector<int> &idx,
                    Eigen::Matrix3d &E) {
    if (idx.size() < 8)
      return false;
    Eigen::MatrixXd A((int)idx.size(), 9);
    for (size_t r = 0; r < idx.size(); ++r) {
      const Eigen::Vector3d &p = b1[idx[r]], &q = b2[idx[r]];
      A.row((int)r) << q(0) * p(0), q(0) * p(1), q(0) * p(2), q(1) * p(0), q(1) * p(1), q(1) * p(2), q(2) * p(0), q(2) * p(1),
          q(2) * p(2);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::Matrix<double, 9, 1> e = svd.matrixV().col(8);
    Eigen::Matrix3d E0;
    E0 << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);
    Eigen::JacobiSVD<Eigen::Matrix3d> s2(E0, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d d(1.0, 1.0, 0.0);
    E = s2.matrixU() * d.asDiagonal() * s2.matrixV().transpose();
    return true;
  }

  /// Angular transfer residual of the homography model: angle(b2, H b1).
  static double transfer_ang(const Eigen::Matrix3d &H, const Eigen::Vector3d &b1, const Eigen::Vector3d &b2) {
    const Eigen::Vector3d p = H * b1;
    const double pn = p.norm();
    if (pn < 1e-12)
      return M_PI;
    return std::acos(std::min(1.0, std::max(-1.0, b2.dot(p) / pn)));
  }

  /// Linear calibrated-homography fit on unit bearings: rows of skew(b2) H b1 = 0
  /// (3 rows per pair, rank 2). Sign-fixed so that b2 . (H b1) > 0 on average.
  static bool fit_H(const std::vector<Eigen::Vector3d> &b1, const std::vector<Eigen::Vector3d> &b2, const std::vector<int> &idx,
                    Eigen::Matrix3d &H) {
    if (idx.size() < 4)
      return false;
    Eigen::MatrixXd A(3 * (int)idx.size(), 9);
    for (size_t r = 0; r < idx.size(); ++r) {
      const Eigen::Vector3d &p = b1[idx[r]];
      Eigen::Matrix3d sq;
      sq << 0, -b2[idx[r]](2), b2[idx[r]](1), b2[idx[r]](2), 0, -b2[idx[r]](0), -b2[idx[r]](1), b2[idx[r]](0), 0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) // unknown H row-major: row block i of skew(b2), column j of H acts on p(j)
          for (int k = 0; k < 3; ++k)
            A(3 * (int)r + i, 3 * k + j) = sq(i, k) * p(j);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::Matrix<double, 9, 1> h = svd.matrixV().col(8);
    H << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
    double s = 0.0;
    for (int i : idx)
      s += b2[i].dot(H * b1[i]);
    if (s < 0.0)
      H = -H;
    return H.allFinite();
  }

  /**
   * @brief Faugeras/MASKS decomposition of a calibrated homography H = R + t n^T.
   * Normalizes by the middle singular value, builds the two admissible (R, n, t)
   * pairs from the singular structure of H^T H, resolves the n-sign by plane
   * visibility (n . b1 > 0) and the pair by triangulated-depth cheirality; a
   * pure-rotation H (equal singular values) short-circuits to the polar R.
   */
  static bool decompose_H(const Eigen::Matrix3d &H_in, const std::vector<Eigen::Vector3d> &b1, const std::vector<Eigen::Vector3d> &b2,
                          const std::vector<int> &inl, Eigen::Matrix3d &R_best) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svh(H_in);
    const double s2 = svh.singularValues()(1);
    if (s2 < 1e-12)
      return false;
    const Eigen::Matrix3d H = H_in / s2;
    const Eigen::Matrix3d HtH = H.transpose() * H;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(HtH);
    // eigenvalues ascending: s3^2 <= 1 <= s1^2
    const double l1 = eig.eigenvalues()(2), l3 = eig.eigenvalues()(0);
    const Eigen::Vector3d v1 = eig.eigenvectors().col(2), v2 = eig.eigenvectors().col(1), v3 = eig.eigenvectors().col(0);
    if (l1 - l3 < 1e-9) {
      // pure rotation: H is (numerically) orthogonal — polar projection
      Eigen::JacobiSVD<Eigen::Matrix3d> s(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
      D(2, 2) = ((s.matrixU() * s.matrixV().transpose()).determinant() < 0.0) ? -1.0 : 1.0;
      R_best = s.matrixU() * D * s.matrixV().transpose();
      return true;
    }
    const double a = std::sqrt(std::max(0.0, 1.0 - l3)), b = std::sqrt(std::max(0.0, l1 - 1.0));
    const double den = std::sqrt(std::max(1e-30, l1 - l3));
    const Eigen::Vector3d u1 = (a * v1 + b * v3) / den;
    const Eigen::Vector3d u2 = (a * v1 - b * v3) / den;

    int best_score = -1;
    for (int c = 0; c < 2; ++c) {
      const Eigen::Vector3d &u = (c == 0) ? u1 : u2;
      Eigen::Matrix3d W, Uh;
      W.col(0) = v2;
      W.col(1) = u;
      W.col(2) = v2.cross(u);
      Uh.col(0) = H * v2;
      Uh.col(1) = H * u;
      Uh.col(2) = (H * v2).cross(H * u);
      const Eigen::Matrix3d R = Uh * W.transpose();
      Eigen::Vector3d n = v2.cross(u);
      // plane must be in FRONT of camera 1: n . b1 > 0 for the inliers
      int vis = 0;
      for (size_t s = 0; s < inl.size(); s += std::max<size_t>(1, inl.size() / 16))
        if (n.dot(b1[inl[s]]) > 0.0)
          ++vis;
      if (2 * vis < (int)std::min<size_t>(16, inl.size()))
        n = -n; // flip to the visible side (t flips with it; R unchanged)
      const Eigen::Vector3d t = (H - R) * n;
      // cheirality on triangulated depths (as in the essential path)
      int pos = 0;
      if (t.norm() < 1e-9) {
        pos = (int)inl.size(); // rotation-only member: trivially consistent
      } else {
        for (size_t s = 0; s < inl.size(); s += std::max<size_t>(1, inl.size() / 16)) {
          const Eigen::Vector3d Rb1 = R * b1[inl[s]];
          Eigen::Matrix<double, 3, 2> M;
          M.col(0) = Rb1;
          M.col(1) = -b2[inl[s]];
          const Eigen::Vector2d d = M.colPivHouseholderQr().solve(-t);
          if (d(0) > 0 && d(1) > 0)
            ++pos;
        }
      }
      if (pos > best_score) {
        best_score = pos;
        R_best = R;
      }
    }
    return best_score > 0 && R_best.allFinite();
  }

  /// Decompose E into the cheirality-consistent (R, t). Returns false on tie.
  static bool decompose(const Eigen::Matrix3d &E, const std::vector<Eigen::Vector3d> &b1, const std::vector<Eigen::Vector3d> &b2,
                        const std::vector<int> &inl, Eigen::Matrix3d &R_best) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    if (U.determinant() < 0)
      U.col(2) *= -1.0;
    if (V.determinant() < 0)
      V.col(2) *= -1.0;
    Eigen::Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    const Eigen::Matrix3d Ra = U * W * V.transpose();
    const Eigen::Matrix3d Rb = U * W.transpose() * V.transpose();
    const Eigen::Vector3d t = U.col(2);
    int best_pos = -1;
    for (int c = 0; c < 4; ++c) {
      const Eigen::Matrix3d &R = (c < 2) ? Ra : Rb;
      const Eigen::Vector3d tc = (c % 2 == 0) ? t : Eigen::Vector3d(-t);
      int pos = 0;
      for (size_t s = 0; s < inl.size(); s += std::max<size_t>(1, inl.size() / 16)) {
        // two-view midpoint depths: b2 ~ (R b1 d1 + tc)/d2
        const Eigen::Vector3d Rb1 = R * b1[inl[s]];
        Eigen::Matrix<double, 3, 2> M;
        M.col(0) = Rb1;
        M.col(1) = -b2[inl[s]];
        const Eigen::Vector2d d = M.colPivHouseholderQr().solve(-tc);
        if (d(0) > 0 && d(1) > 0)
          ++pos;
      }
      if (pos > best_pos) {
        best_pos = pos;
        R_best = R;
      } else if (pos == best_pos) {
        // ambiguous cheirality (near-degenerate translation): caller falls back
      }
    }
    return best_pos > 0;
  }

  static Result solve(const std::vector<Eigen::Vector3d> &b1, const std::vector<Eigen::Vector3d> &b2, uint64_t rng_seed) {
    return solve(b1, b2, rng_seed, Options());
  }
  static Result solve(const std::vector<Eigen::Vector3d> &b1, const std::vector<Eigen::Vector3d> &b2, uint64_t rng_seed,
                      const Options &opt) {
    Result out;
    const int N = (int)b1.size();
    if (N < std::max(opt.min_pairs, 9))
      return fallback_(b1, b2, rng_seed, opt);
    // Model selection (GRIC-lite): the rotation-only model is UNBIASED and
    // lower-variance when the per-pair translation is inside the noise floor
    // (far scenes) — there the extra E DOF only absorb noise. Prefer Procrustes
    // unless its residual shows real translation signal for E to model.
    const Result rot = fallback_(b1, b2, rng_seed, opt);
    if (rot.ok && rot.mean_resid_rad < std::max(opt.inlier_threshold_rad, 1.5e-3))
      return rot;
    std::mt19937 rng((unsigned)(rng_seed & 0xFFFFFFFFu));
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    Eigen::Matrix3d best_E = Eigen::Matrix3d::Zero();
    int best_inl = -1;
    std::vector<int> samp(8);
    for (int it = 0; it < opt.ransac_iterations; ++it) {
      for (int s = 0; s < 8; ++s) {
        std::uniform_int_distribution<int> dist(s, N - 1);
        std::swap(idx[s], idx[dist(rng)]);
        samp[s] = idx[s];
      }
      Eigen::Matrix3d E;
      if (!fit_E(b1, b2, samp, E))
        continue;
      int inl = 0;
      for (int i = 0; i < N; ++i)
        if (epi_ang(E, b1[i], b2[i]) < opt.inlier_threshold_rad)
          ++inl;
      if (inl > best_inl) {
        best_inl = inl;
        best_E = E;
      }
      if (best_inl > N * 95 / 100)
        break;
    }
    // ---- homography model (deterministic sibling RANSAC; 4-point samples) ----
    // On a planar scene E's linear system has a solution FAMILY: it still
    // "inlies" everything while its decomposed R is arbitrary — inlier counts,
    // not residuals, expose the degeneracy (Torr/Pollefeys H-vs-F test).
    Eigen::Matrix3d best_H = Eigen::Matrix3d::Zero();
    int best_hinl = -1;
    const double h_gate = opt.inlier_threshold_rad * std::sqrt(2.0); // codim-2 transfer vs codim-1 plane distance
    if (opt.enable_homography) {
      std::mt19937 hrng((unsigned)((rng_seed ^ 0x9E3779B97F4A7C15ull) & 0xFFFFFFFFu));
      std::vector<int> hidx(N);
      std::iota(hidx.begin(), hidx.end(), 0);
      std::vector<int> hsamp(4);
      for (int it = 0; it < opt.ransac_iterations; ++it) {
        for (int s = 0; s < 4; ++s) {
          std::uniform_int_distribution<int> dist(s, N - 1);
          std::swap(hidx[s], hidx[dist(hrng)]);
          hsamp[s] = hidx[s];
        }
        Eigen::Matrix3d H;
        if (!fit_H(b1, b2, hsamp, H))
          continue;
        int inl = 0;
        for (int i = 0; i < N; ++i)
          if (transfer_ang(H, b1[i], b2[i]) < h_gate)
            ++inl;
        if (inl > best_hinl) {
          best_hinl = inl;
          best_H = H;
        }
        if (best_hinl > N * 95 / 100)
          break;
      }
    }

    const bool e_ok = best_inl >= std::max(opt.min_pairs, (int)(opt.min_inlier_ratio * N));
    const bool h_ok = best_hinl >= std::max(opt.min_pairs, (int)(opt.min_inlier_ratio * N));
    int offplane = 0; // E-inliers the homography cannot explain = E's non-planar support
    if (e_ok && h_ok)
      for (int i = 0; i < N; ++i)
        if (epi_ang(best_E, b1[i], b2[i]) < opt.inlier_threshold_rad && transfer_ang(best_H, b1[i], b2[i]) >= h_gate)
          ++offplane;
    const bool planar = h_ok && (!e_ok || offplane < opt.min_offplane_support);

    if (planar) {
      // refit H on its inliers, decompose to R
      std::vector<int> hinl;
      hinl.reserve(best_hinl);
      for (int i = 0; i < N; ++i)
        if (transfer_ang(best_H, b1[i], b2[i]) < h_gate)
          hinl.push_back(i);
      Eigen::Matrix3d H, R;
      if (fit_H(b1, b2, hinl, H) && decompose_H(H, b1, b2, hinl, R)) {
        double rsum = 0.0;
        for (int i : hinl)
          rsum += transfer_ang(H, b1[i], b2[i]);
        out.ok = true;
        out.model = MODEL_HOMOGRAPHY;
        out.used_essential = false;
        out.R_C1toC2 = R;
        out.inliers = (int)hinl.size();
        out.inlier_ratio = (double)hinl.size() / (double)N;
        out.mean_resid_rad = rsum / std::max<size_t>(hinl.size(), 1);
        return out;
      }
      // decomposition failure: fall through to E (if usable) or Procrustes
    }
    if (!e_ok)
      return fallback_(b1, b2, rng_seed, opt);
    std::vector<int> inl;
    inl.reserve(best_inl);
    for (int i = 0; i < N; ++i)
      if (epi_ang(best_E, b1[i], b2[i]) < opt.inlier_threshold_rad)
        inl.push_back(i);
    Eigen::Matrix3d E;
    if (!fit_E(b1, b2, inl, E))
      return fallback_(b1, b2, rng_seed, opt);
    Eigen::Matrix3d R;
    if (!decompose(E, b1, b2, inl, R))
      return fallback_(b1, b2, rng_seed, opt);
    double rsum = 0.0;
    for (int i : inl)
      rsum += epi_ang(E, b1[i], b2[i]);
    out.ok = true;
    out.model = MODEL_ESSENTIAL;
    out.used_essential = true;
    out.R_C1toC2 = R;
    out.inliers = (int)inl.size();
    out.inlier_ratio = (double)inl.size() / (double)N;
    out.mean_resid_rad = rsum / std::max<size_t>(inl.size(), 1);
    return out;
  }

private:
  static Result fallback_(const std::vector<Eigen::Vector3d> &b1, const std::vector<Eigen::Vector3d> &b2, uint64_t rng_seed,
                          const Options &opt) {
    RelRotProcrustes::Options po;
    po.min_pairs = opt.min_pairs;
    const RelRotProcrustes::Result pr = RelRotProcrustes::solve(b1, b2, rng_seed, po);
    Result out;
    out.ok = pr.ok;
    out.R_C1toC2 = pr.R_C1toC2;
    out.inliers = pr.inliers;
    out.inlier_ratio = pr.inlier_ratio;
    out.mean_resid_rad = pr.mean_resid_rad;
    out.model = MODEL_PROCRUSTES;
    out.used_essential = false;
    return out;
  }
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_REL_ROT_PROCRUSTES_H
