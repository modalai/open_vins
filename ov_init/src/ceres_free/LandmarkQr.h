/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef OV_INIT_ZBFT_SFM_LANDMARK_QR_H
#define OV_INIT_ZBFT_SFM_LANDMARK_QR_H

#include "Parallel.h"

#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ov_init {
namespace zbft_sfm {
namespace landmark_qr {

// Floating-point predicates may be optimized away under -ffast-math. Inspect
// IEEE-754 exponent bits instead; these checks are part of the export contract.
inline bool finite_scalar(double value) {
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value), "64-bit IEEE double required");
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

template <typename Derived>
inline bool all_finite(const Eigen::MatrixBase<Derived> &value) {
  for (Eigen::Index c = 0; c < value.cols(); ++c)
    for (Eigen::Index r = 0; r < value.rows(); ++r)
      if (!finite_scalar(value(r, c)))
        return false;
  return true;
}

struct Evidence {
  double cost = 0.0;             // original robust objective, before elimination
  double land_decrement = 0.0;   // twice the quadratic cost decrease from feature elimination
  int clamped_dirs = 0;
};

// B and [A r] contain sqrt(rho')-weighted LOCAL Jacobian/residual rows.
// B has three columns and at least three rows (short tracks are zero-padded).
// The surviving rows of Ar start at `rank`. No B'B or subtractive Schur fold is
// formed. The original information rank rule lambda_i > 1e-8*lambda_max,
// lambda_max > 1e-12 is applied to singular values without squaring them.
inline bool project(const Eigen::MatrixXd &B, Eigen::MatrixXd &Ar, int &rank, double &decrement) {
  if (B.cols() != 3 || B.rows() < 3 || B.rows() != Ar.rows() || Ar.cols() < 1 ||
      !all_finite(B) || !all_finite(Ar))
    return false;
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(B);
  if (!all_finite(qr.matrixQR()) || !all_finite(qr.hCoeffs()))
    return false;
  const Eigen::Matrix3d R = qr.matrixQR().topRows(3).template triangularView<Eigen::Upper>();
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU);
  const Eigen::Vector3d sigma = svd.singularValues();
  if (!all_finite(sigma) || !all_finite(svd.matrixU()))
    return false;
  rank = 0;
  decrement = 0.0;
  if (sigma(0) > 1e-6)
    for (int k = 0; k < 3; ++k)
      if (sigma(k) > 1e-4 * sigma(0))
        ++rank;
  if (rank == 0)
    return true;
  Ar = (qr.householderQ().adjoint() * Ar).eval();
  // For full rank the first three rows all disappear, so rotating them is
  // unnecessary. For deficient rank, rotate the top rows to the retained
  // left-singular directions before dropping only those directions.
  if (rank < 3)
    Ar.topRows(3) = (svd.matrixU().transpose() * Ar.topRows(3)).eval();
  decrement = Ar.col(Ar.cols() - 1).head(rank).squaredNorm();
  return all_finite(Ar) && finite_scalar(decrement);
}

// Build UNDAMPED information directly from projected feature rows. The template
// keeps Problem's protected graph types private and adds no object/ABI state.
// Every residual must touch at most one distinct variable 3-dof landmark;
// unsupported coupled/non-3-dof landmark graphs fail explicitly.
//
// Memory is O(workers*n_nav^2) plus one track/factor workspace per worker:
// O(max_rows*(max_adjacent_local_dim+4) + max_adjacent_local_dim^2).
// Rows from different features never coexist in a projection. Factors without
// variable landmarks retain all their local columns and are accumulated normally.
template <typename Blocks, typename Residuals>
bool assemble(const Blocks &blocks, const Residuals &residuals,
              const std::vector<int> &land_blocks, const std::vector<std::vector<int>> &land_adj,
              int n_nav, ParallelExecutor &exec, Eigen::MatrixXd &Hnav, Eigen::VectorXd &gnav,
              Evidence &evidence) {
  const int nl = (int)land_blocks.size();
  if ((int)land_adj.size() != nl || n_nav < 0)
    return false;
  std::vector<int> land_of_block(blocks.size(), -1);
  for (int li = 0; li < nl; ++li) {
    if (blocks[land_blocks[li]].lsize != 3)
      return false;
    land_of_block[land_blocks[li]] = li;
  }
  std::vector<std::vector<int>> track_residuals(nl);
  std::vector<int> other_residuals;
  for (int ri = 0; ri < (int)residuals.size(); ++ri) {
    int land = -1;
    for (int bi : residuals[ri].blocks) {
      if (blocks[bi].constant || !blocks[bi].landmark)
        continue;
      const int li = land_of_block[bi];
      if (li < 0 || (land >= 0 && land != li))
        return false;
      land = li;
    }
    if (land < 0)
      other_residuals.push_back(ri);
    else
      track_residuals[land].push_back(ri);
  }

  struct Accumulator {
    Eigen::MatrixXd H;
    Eigen::VectorXd g;
    Evidence evidence;
    bool ok = true;
  };
  const int nw = exec.num_workers();
  std::vector<Accumulator> accum(nw);
  for (auto &a : accum) {
    a.H = Eigen::MatrixXd::Zero(n_nav, n_nav);
    a.g = Eigen::VectorXd::Zero(n_nav);
  }
  const auto body = [&](int worker, int begin, int end) {
    auto &out = accum[worker];
    using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    std::vector<const double *> params;
    std::vector<double *> jacptrs;
    std::vector<RowMatrix> Jstore;
    RowMatrix basis;
    Eigen::MatrixXd B, Ar, track_H;
    Eigen::VectorXd rbuf, track_g;
    std::vector<int> adjacent, local_offset(blocks.size(), -1), singleton(1);

    for (int task = begin; task < end && out.ok; ++task) {
      const bool is_track = task < nl;
      if (is_track) {
        adjacent = land_adj[task];
      } else {
        singleton[0] = other_residuals[task - nl];
        adjacent.clear();
        for (int bi : residuals[singleton[0]].blocks)
          if (!blocks[bi].constant &&
              std::find(adjacent.begin(), adjacent.end(), bi) == adjacent.end())
            adjacent.push_back(bi);
      }
      const auto &indices = is_track ? track_residuals[task] : singleton;
      int d = 0, m = 0;
      for (int bi : adjacent) {
        if (blocks[bi].constant || blocks[bi].landmark || blocks[bi].offset < 0 ||
            blocks[bi].offset + blocks[bi].lsize > n_nav) {
          out.ok = false;
          break;
        }
        local_offset[bi] = d;
        d += blocks[bi].lsize;
      }
      if (!out.ok)
        break;
      for (int ri : indices)
        m += residuals[ri].cost->num_residuals();
      const int rows = is_track ? std::max(3, m) : m;
      Ar.setZero(rows, d + 1);
      if (is_track)
        B.setZero(rows, 3);
      int row = 0;
      for (int ri : indices) {
        const auto &res = residuals[ri];
        const int nr = res.cost->num_residuals(), nb = (int)res.blocks.size();
        params.resize(nb);
        jacptrs.assign(nb, nullptr);
        Jstore.resize(nb);
        for (int k = 0; k < nb; ++k) {
          const auto &b = blocks[res.blocks[k]];
          params[k] = b.data;
          if (!b.constant) {
            Jstore[k].resize(nr, b.gsize);
            jacptrs[k] = Jstore[k].data();
          }
        }
        rbuf.resize(nr);
        if (!res.cost->Evaluate(params.data(), rbuf.data(), jacptrs.data())) {
          out.ok = false;
          break;
        }
        if (!all_finite(rbuf)) {
          out.ok = false;
          break;
        }
        for (int k = 0; k < nb; ++k)
          if (jacptrs[k] && !all_finite(Jstore[k]))
            out.ok = false;
        const double s = rbuf.squaredNorm();
        if (!out.ok || !finite_scalar(s)) {
          out.ok = false;
          break;
        }
        double weight = 1.0;
        if (res.loss) {
          double rho[2];
          res.loss->Evaluate(s, rho);
          if (!finite_scalar(rho[0]) || !finite_scalar(rho[1])) {
            out.ok = false;
            break;
          }
          out.evidence.cost += 0.5 * rho[0];
          weight = rho[1] > 0.0 ? rho[1] : 0.0;
        } else {
          out.evidence.cost += 0.5 * s;
        }
        if (weight > 0.0) {
          const double root_weight = std::sqrt(weight);
          Ar.col(d).segment(row, nr) = root_weight * rbuf;
          for (int k = 0; k < nb; ++k) {
            const int bi = res.blocks[k];
            const auto &b = blocks[bi];
            if (b.constant)
              continue;
            // += combines duplicate occurrences of a parameter in one factor.
            // This preserves all their self/cross terms after row assembly.
            if (b.tangent_leading_identity) {
              if (b.landmark)
                B.middleRows(row, nr).noalias() += root_weight * Jstore[k].leftCols(b.lsize);
              else
                Ar.block(row, local_offset[bi], nr, b.lsize).noalias() += root_weight * Jstore[k].leftCols(b.lsize);
            } else {
              basis.resize(b.gsize, b.lsize);
              if (!b.param || !b.param->ComputeJacobian(b.data, basis.data()) || !all_finite(basis)) {
                out.ok = false;
                break;
              }
              if (b.landmark)
                B.middleRows(row, nr).noalias() += root_weight * Jstore[k] * basis;
              else
                Ar.block(row, local_offset[bi], nr, b.lsize).noalias() += root_weight * Jstore[k] * basis;
            }
          }
        }
        row += nr;
        if (!out.ok)
          break;
      }
      if (!out.ok)
        break;
      int rank = 0;
      double decrement = 0.0;
      if (is_track) {
        if (!project(B, Ar, rank, decrement)) {
          out.ok = false;
          break;
        }
        out.evidence.clamped_dirs += 3 - rank;
        out.evidence.land_decrement += decrement;
      }
      const auto A = Ar.bottomRows(rows - rank).leftCols(d);
      const auto r = Ar.col(d).tail(rows - rank);
      track_H.noalias() = A.transpose() * A;
      track_g.noalias() = A.transpose() * r;
      if (!all_finite(track_H) || !all_finite(track_g)) {
        out.ok = false;
        break;
      }
      for (int ba : adjacent) {
        const auto &a = blocks[ba];
        const int ca = local_offset[ba];
        out.g.segment(a.offset, a.lsize) += track_g.segment(ca, a.lsize);
        for (int bb : adjacent) {
          const auto &b = blocks[bb];
          if (a.offset < b.offset)
            continue;
          out.H.block(a.offset, b.offset, a.lsize, b.lsize) +=
              track_H.block(ca, local_offset[bb], a.lsize, b.lsize);
        }
      }
    }
  };
  exec.parallel_ranges(nl + (int)other_residuals.size(), body);
  for (const auto &a : accum)
    if (!a.ok || !finite_scalar(a.evidence.cost) || !finite_scalar(a.evidence.land_decrement))
      return false;
  Hnav = std::move(accum[0].H);
  gnav = std::move(accum[0].g);
  evidence = accum[0].evidence;
  for (int w = 1; w < nw; ++w) {
    for (int j = 0; j < n_nav; ++j)
      Hnav.col(j).tail(n_nav - j) += accum[w].H.col(j).tail(n_nav - j);
    gnav += accum[w].g;
    evidence.cost += accum[w].evidence.cost;
    evidence.land_decrement += accum[w].evidence.land_decrement;
    evidence.clamped_dirs += accum[w].evidence.clamped_dirs;
  }
  Hnav = Eigen::MatrixXd(Hnav.template selfadjointView<Eigen::Lower>());
  return all_finite(Hnav) && all_finite(gnav) && finite_scalar(evidence.cost) && finite_scalar(evidence.land_decrement);
}

} // namespace landmark_qr
} // namespace zbft_sfm
} // namespace ov_init

#endif
