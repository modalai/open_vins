/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Contributor: Joao Leonardo Silva Cotta (@zauberflote1)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "Problem.h"

#include "Parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>

using namespace ov_init::zbft_sfm;

Problem::~Problem() {
  if (!owns_)
    return;
  // Delete each unique pointer once (a loss/param may be shared across residuals/blocks).
  std::set<const void *> seen;
  for (const CostFunction *p : owned_cost_)
    if (p && seen.insert(p).second)
      delete p;
  for (const LossFunction *p : owned_loss_)
    if (p && seen.insert(p).second)
      delete p;
  for (const LocalParameterization *p : owned_param_)
    if (p && seen.insert(p).second)
      delete p;
}

namespace {
inline double seconds_since(const std::chrono::steady_clock::time_point &t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
} // namespace

int Problem::block_index(double *values) const {
  auto it = index_.find(values);
  return (it == index_.end()) ? -1 : it->second;
}

int Problem::AddParameterBlock(double *values, int global_size, const LocalParameterization *param) {
  int existing = block_index(values);
  if (existing >= 0)
    return existing;
  Block b;
  b.data = values;
  b.gsize = global_size;
  b.param = param;
  b.lsize = (param != nullptr) ? param->LocalSize() : global_size;
  b.constant = false;
  b.landmark = false;
  // Resolve once: true for Euclidean (param==nullptr), JPL quat; false for S² gravity.
  b.tangent_leading_identity = (param == nullptr) || param->tangent_is_leading_identity();
  int idx = (int)blocks_.size();
  blocks_.push_back(b);
  index_[values] = idx;
  if (owns_ && param != nullptr)
    owned_param_.push_back(param);
  return idx;
}

void Problem::SetParameterBlockConstant(double *values) {
  int i = block_index(values);
  if (i >= 0)
    blocks_[i].constant = true;
}
void Problem::SetParameterBlockVariable(double *values) {
  int i = block_index(values);
  if (i >= 0)
    blocks_[i].constant = false;
}
void Problem::SetSchurLandmark(double *values) {
  int i = block_index(values);
  if (i >= 0)
    blocks_[i].landmark = true;
}

void Problem::AddResidualBlock(const CostFunction *cost, const LossFunction *loss, const std::vector<double *> &blocks) {
  Residual r;
  r.cost = cost;
  r.loss = loss;
  r.blocks.reserve(blocks.size());
  for (double *b : blocks) {
    int idx = block_index(b);
    // Blocks must be registered with AddParameterBlock first (so we know their sizes/manifolds).
    r.blocks.push_back(idx);
  }
  residuals_.push_back(std::move(r));
  if (owns_) {
    owned_cost_.push_back(cost);
    if (loss != nullptr)
      owned_loss_.push_back(loss);
  }
}

void Problem::assign_ordering() {
  n_nav_ = 0;
  n_land_ = 0;
  land_diag_.clear();
  land_block_idx_.clear();
  // Pass 1: navigation (non-landmark) variable blocks come first.
  for (auto &b : blocks_) {
    if (b.constant || b.landmark) {
      b.offset = -1;
      continue;
    }
    b.offset = n_nav_;
    n_nav_ += b.lsize;
  }
  // Pass 2: landmark variable blocks occupy the tail partition.
  for (int bi = 0; bi < (int)blocks_.size(); ++bi) {
    Block &b = blocks_[bi];
    if (b.constant || !b.landmark)
      continue;
    b.offset = n_nav_ + n_land_;
    land_diag_.emplace_back(n_land_, b.lsize); // offset within the land partition
    land_block_idx_.push_back(bi);
    n_land_ += b.lsize;
  }
  n_total_ = n_nav_ + n_land_;

  // Size the apply_delta Plus() scratch once (largest ambient block) -- no alloc per trial.
  int max_gsize = 0;
  for (const auto &b : blocks_)
    max_gsize = std::max(max_gsize, b.gsize);
  plus_tmp_.resize(max_gsize);

  // Used-lower-triangle column tails (see Problem.h): nav columns run to n_total_
  // (nav-nav lower + landmark W^T strip); a landmark column ends at its own diagonal block.
  col_tail_end_.assign(n_total_, n_total_);
  for (const auto &od : land_diag_) {
    const int b0 = n_nav_ + od.first;
    for (int c = 0; c < od.second; ++c)
      col_tail_end_[b0 + c] = b0 + od.second;
  }

  // Visibility structure: for each landmark, the navigation (pose) blocks that co-occur
  // with it in a residual. Landmarks couple ONLY to these blocks (not v/bg/ba), so the
  // Schur reduction touches only them -- a handful of small updates instead of a dense gemm.
  std::unordered_map<int, int> li_of_block;
  for (int li = 0; li < (int)land_block_idx_.size(); ++li)
    li_of_block[land_block_idx_[li]] = li;
  land_adj_.assign(land_diag_.size(), {});
  for (const Residual &res : residuals_) {
    for (int bidx : res.blocks) {
      auto it = li_of_block.find(bidx);
      if (it == li_of_block.end())
        continue;
      std::vector<int> &adj = land_adj_[it->second];
      for (int nb : res.blocks) {
        const Block &nbk = blocks_[nb];
        if (nbk.constant || nbk.landmark)
          continue;
        if (std::find(adj.begin(), adj.end(), nb) == adj.end())
          adj.push_back(nb);
      }
    }
  }
}

double Problem::evaluate_cost(ParallelExecutor &exec) const {
  // Residual-only trial scoring. This runs once per LM/dogleg TRIAL (accepted or rejected),
  // so it is parallelized on the same fixed-range executor as linearize(), with the same
  // worker-ordered reduction: bit-identical run-to-run for a given worker count, and the
  // W==1 inline path is arithmetically identical to the old serial loop.
  ++n_res_evals_;
  const int W = exec.num_workers();
  if ((int)cw_.size() != W)
    cw_.assign(W, 0.0);
  const auto body = [&](int worker, int begin, int end) {
    double cl = 0.0;
    std::vector<const double *> params;
    Eigen::VectorXd rbuf; // residual scratch, grown once to the max residual count (no per-residual heap alloc)
    for (int ri = begin; ri < end; ++ri) {
      const Residual &res = residuals_[ri];
      const int nres = res.cost->num_residuals();
      params.clear();
      params.reserve(res.blocks.size());
      for (int bidx : res.blocks)
        params.push_back(blocks_[bidx].data);
      if (rbuf.size() < nres)
        rbuf.resize(nres);
      res.cost->Evaluate(params.data(), rbuf.data(), nullptr);
      const double s = rbuf.head(nres).squaredNorm();
      if (res.loss) {
        double rho[2];
        res.loss->Evaluate(s, rho);
        cl += 0.5 * rho[0];
      } else {
        cl += 0.5 * s;
      }
    }
    cw_[worker] = cl;
  };
  exec.parallel_ranges((int)residuals_.size(), body);
  double cost = cw_[0];
  for (int w = 1; w < W; ++w)
    cost += cw_[w];
  return cost;
}

void Problem::linearize(Eigen::MatrixXd &H, Eigen::VectorXd &grad, double &cost, ParallelExecutor &exec) const {
  ++n_jac_evals_;
  const int W = exec.num_workers();
  const int N = n_total_;

  // Direct manifold Jacobians: for the parameterizations used here (Euclidean V=I and
  // the JPL quaternion V=[I3;0], OVINS convention), the local (error-state) Jacobian is
  // EXACTLY the leading `lsize` columns of the ambient Jacobian the factor writes. So we
  // use J.leftCols(lsize) directly -- no ambient->local matmul, no PlusJacobian allocation.

  // Per-worker private accumulators (no shared writes -> race-free), PREALLOCATED and
  // reused across iterations -- only zeroed each call (no heap traffic in the hot loop).
  // Worker 0 accumulates DIRECTLY into the caller's H/grad (they persist across iterations),
  // so slots [1..W) are the only extra buffers and the old H = Hw_[0] full-matrix copy is gone.
  if ((int)Hw_.size() != W) {
    Hw_.assign(W, Eigen::MatrixXd());
    gw_.assign(W, Eigen::VectorXd());
    cw_.assign(W, 0.0);
  }
  // Zero ONLY the used lower-triangle column tails (see col_tail_end_): the upper triangle
  // and the landmark-landmark off-diagonal region are never written nor read.
  const auto zero_used = [&](Eigen::MatrixXd &M) {
    for (int j = 0; j < N; ++j)
      M.col(j).segment(j, col_tail_end_[j] - j).setZero();
  };
  // On (re)allocation, zero the WHOLE matrix once: the structurally-zero region outside the
  // used tails is never written again, so selfadjoint<Lower> consumers (dogleg matvec) can
  // safely read the full lower triangle. Steady state pays only the used-tail zeroing.
  if (H.rows() != N || H.cols() != N) {
    H.resize(N, N);
    H.setZero();
  }
  if (grad.size() != N)
    grad.resize(N);
  zero_used(H);
  grad.setZero();
  for (int w = 1; w < W; ++w) {
    if (Hw_[w].rows() != N) {
      Hw_[w].resize(N, N);
      Hw_[w].setZero();
    }
    if (gw_[w].size() != N)
      gw_[w].resize(N);
    zero_used(Hw_[w]);
    gw_[w].setZero();
  }

  const auto body = [&](int worker, int begin, int end) {
    Eigen::MatrixXd &Hloc = (worker == 0) ? H : Hw_[worker];
    Eigen::VectorXd &gloc = (worker == 0) ? grad : gw_[worker];
    double cl = 0.0; // local accumulator (a shared cw_[worker] target would false-share the cache line)
    std::vector<const double *> params;
    std::vector<double *> jacptrs;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Jstore;
    std::vector<int> vidx; // global block index of each variable block
    std::vector<int> vk;   // its position in res.blocks / Jstore
    Eigen::VectorXd rbuf;  // residual scratch, grown to the max residual count once (no per-residual heap alloc)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 16, 16> Mab; // off-diagonal Hessian block scratch (stack, no heap)

    // Effective local Jacobians for non-identity-tangent blocks (e.g., S² gravity).
    // For tangent_leading_identity=true blocks, we keep using the zero-copy leftCols view.
    // For false blocks (gravity), Jeff[k] = Jstore[k] * V where V is the tangent basis.
    // Sized per-block per-residual, heap-backed but resize-and-reuse (steady-state no alloc).
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Jeff;
    Eigen::Matrix<double, 3, 2> Vbuf; // tangent basis scratch for S² (stack, 3×2)

    for (int ri = begin; ri < end; ++ri) {
      const Residual &res = residuals_[ri];
      const int nb = (int)res.blocks.size();
      const int nres = res.cost->num_residuals();

      params.assign(nb, nullptr);
      jacptrs.assign(nb, nullptr);
      Jstore.resize(nb);
      Jeff.resize(nb);
      for (int k = 0; k < nb; ++k) {
        const Block &b = blocks_[res.blocks[k]];
        params[k] = b.data;
        if (!b.constant) {
          Jstore[k].resize(nres, b.gsize);
          jacptrs[k] = Jstore[k].data();
        }
      }

      if (rbuf.size() < nres)
        rbuf.resize(nres);
      auto r = rbuf.head(nres);
      res.cost->Evaluate(params.data(), rbuf.data(), jacptrs.data());

      // Robustified cost (0.5*rho(s)) and IRLS weight w = rho'(s), s = ||r||^2.
      const double s = r.squaredNorm();
      double w = 1.0;
      if (res.loss) {
        double rho[2];
        res.loss->Evaluate(s, rho);
        cl += 0.5 * rho[0];
        w = (rho[1] > 0.0) ? rho[1] : 0.0;
      } else {
        cl += 0.5 * s;
      }
      if (w == 0.0)
        continue;

      vidx.clear();
      vk.clear();
      for (int k = 0; k < nb; ++k) {
        if (blocks_[res.blocks[k]].constant)
          continue;
        vidx.push_back(res.blocks[k]);
        vk.push_back(k);
      }

      // Pre-compute effective local Jacobians for non-identity-tangent blocks.
      // For identity blocks (JPL quat, Euclidean), Jeff stays unused; we use leftCols directly.
      for (size_t i = 0; i < vidx.size(); ++i) {
        const Block &bi = blocks_[vidx[i]];
        if (!bi.tangent_leading_identity && bi.param != nullptr) {
          // Compute V (gsize × lsize) and J_local = J_ambient * V
          Jeff[vk[i]].resize(nres, bi.lsize);
          bi.param->ComputeJacobian(bi.data, Vbuf.data()); // 3×2 row-major for S²
          Jeff[vk[i]].noalias() = Jstore[vk[i]] * Vbuf.topLeftCorner(bi.gsize, bi.lsize);
        }
      }

      // Accumulate gradient/Hessian into the LOWER triangle only (see col_tail_end_ in
      // Problem.h). For tangent_leading_identity=true, use the zero-copy leftCols view; for
      // false (gravity), use the pre-computed Jeff. Each off-diagonal product J_a^T J_b is
      // formed ONCE and stored in its lower-triangle slot; diagonal blocks accumulate their
      // lower part via a rank update (syrk: half the flops and half the writes of a gemm).
      for (size_t a = 0; a < vidx.size(); ++a) {
        const Block &ba = blocks_[vidx[a]];
        // Effective local Jacobian: leftCols for identity, Jeff for non-identity
        const auto &Ja_ref = ba.tangent_leading_identity ? Jstore[vk[a]].leftCols(ba.lsize)
                                                          : Jeff[vk[a]].leftCols(ba.lsize);
        gloc.segment(ba.offset, ba.lsize).noalias() += w * (Ja_ref.transpose() * r);
        Hloc.block(ba.offset, ba.offset, ba.lsize, ba.lsize).selfadjointView<Eigen::Lower>().rankUpdate(Ja_ref.transpose(), w);
        for (size_t b = a + 1; b < vidx.size(); ++b) {
          const Block &bb = blocks_[vidx[b]];
          const auto &Jb_ref = bb.tangent_leading_identity ? Jstore[vk[b]].leftCols(bb.lsize)
                                                            : Jeff[vk[b]].leftCols(bb.lsize);
          Mab.noalias() = w * (Ja_ref.transpose() * Jb_ref); // lsize_a x lsize_b
          if (ba.offset > bb.offset) {
            Hloc.block(ba.offset, bb.offset, ba.lsize, bb.lsize).noalias() += Mab; // H(a,b) is the lower slot
          } else if (ba.offset < bb.offset) {
            Hloc.block(bb.offset, ba.offset, bb.lsize, ba.lsize).noalias() += Mab.transpose(); // lower slot holds Jb^T Ja
          } else {
            // Duplicate parameter block within one residual (e.g. a shared bias passed as both
            // bg1 and bg2 of an IMU factor): both cross terms land on the SAME diagonal block;
            // add the lower part of (Mab + Mab^T) column by column.
            for (int c = 0; c < ba.lsize; ++c)
              Hloc.col(ba.offset + c).segment(ba.offset + c, ba.lsize - c) +=
                  Mab.col(c).tail(ba.lsize - c) + Mab.row(c).tail(ba.lsize - c).transpose();
          }
        }
      }
    }
    cw_[worker] = cl;
  };

  exec.parallel_ranges((int)residuals_.size(), body);

  // Deterministic reduction in fixed worker order (bit-identical regardless of W;
  // worker 0 already lives in H/grad). Adds ONLY the used lower-triangle column tails.
  cost = cw_[0];
  for (int w = 1; w < W; ++w) {
    for (int j = 0; j < N; ++j)
      H.col(j).segment(j, col_tail_end_[j] - j) += Hw_[w].col(j).segment(j, col_tail_end_[j] - j);
    grad += gw_[w];
    cost += cw_[w];
  }
}

bool Problem::solve_step(const Eigen::MatrixXd &H, const Eigen::VectorXd &grad, double lambda, Eigen::VectorXd &delta) const {
  delta.resize(n_total_);

  // Landmark-free (inertial-only) path: damped dense Cholesky (preallocated buffers).
  // H holds only the used lower triangle (see col_tail_end_), which is exactly what LLT reads.
  if (n_land_ == 0 || n_nav_ == 0) {
    if (Hd_.rows() != n_total_)
      Hd_.resize(n_total_, n_total_);
    if (n_nav_ == 0)
      Hd_.setZero(); // degenerate all-landmark problem: LLT reads the whole lower triangle, so the
                     // structurally-zero landmark-landmark region must be explicit here
    for (int j = 0; j < n_total_; ++j)
      Hd_.col(j).segment(j, col_tail_end_[j] - j) = H.col(j).segment(j, col_tail_end_[j] - j);
    for (int i = 0; i < n_total_; ++i)
      Hd_(i, i) += lambda * std::min(std::max(H(i, i), 1e-6), 1e32); // Ceres min/max_lm_diagonal clamp
    // In-place Cholesky: factors Hd_'s own storage (Hd_ is rebuilt every call), so no
    // n^2 factor-storage copy per trial; the triangular solves run in-place on delta.
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(Hd_);
    if (llt.info() != Eigen::Success)
      return false;
    delta = -grad;
    llt.matrixL().solveInPlace(delta);
    llt.matrixU().solveInPlace(delta);
    return delta.allFinite();
  }

  // VISIBILITY-AWARE arrowhead Schur. Each landmark Hessian block is 3x3 and each landmark
  // couples ONLY to its observing pose blocks (q,p) -- never to v/bg/ba -- so the reduction
  // is a few small 3x3 updates per landmark on those poses, NOT a dense n_nav x n_land x
  // n_nav gemm with ~60% zero rows. This is what makes Ceres' Schur eliminator fast.
  if (Hred_.rows() != n_nav_) {
    Hred_.resize(n_nav_, n_nav_);
    rhs_.resize(n_nav_);
    da_.resize(n_nav_);
  }
  // H holds only the used lower triangle; copy the nav-nav lower column tails (all the LLT reads).
  for (int j = 0; j < n_nav_; ++j)
    Hred_.col(j).segment(j, n_nav_ - j) = H.col(j).segment(j, n_nav_ - j);
  for (int i = 0; i < n_nav_; ++i)
    Hred_(i, i) += lambda * std::min(std::max(H(i, i), 1e-6), 1e32);
  rhs_.noalias() = -grad.head(n_nav_);

  for (size_t li = 0; li < land_diag_.size(); ++li) {
    const int g0 = n_nav_ + land_diag_[li].first;
    Eigen::Matrix3d V = Eigen::Matrix3d(H.block(g0, g0, 3, 3).selfadjointView<Eigen::Lower>());
    for (int d = 0; d < 3; ++d)
      V(d, d) += lambda * std::min(std::max(H(g0 + d, g0 + d), 1e-6), 1e32);
    const Eigen::Matrix3d Vinv = V.inverse();
    const Eigen::Vector3d gl = grad.segment(g0, 3);
    const std::vector<int> &adj = land_adj_[li];
    const int P = (int)adj.size();
    if ((int)schur_off_.size() < P) {
      schur_off_.resize(P);
      schur_W_.resize(P);
      schur_Ma_.resize(P);
    }
    // Precompute, per observing pose block: its nav offset, W_a = H[off_a, landmark] (stored
    // transposed in the landmark row-strip of the lower triangle), and M_a = W_a * V^-1.
    // Also fold the rhs update -ga += M_a * g_l here (one pass over P).
    for (int ia = 0; ia < P; ++ia) {
      const int off = blocks_[adj[ia]].offset;
      schur_off_[ia] = off;
      schur_W_[ia] = H.block(g0, off, 3, 3).transpose();
      schur_Ma_[ia].noalias() = schur_W_[ia] * Vinv;
      rhs_.segment(off, 3).noalias() += schur_Ma_[ia] * gl;
    }
    // Symmetric rank-3 fill-in Hred -= W V^-1 W^T. Only the LOWER triangle is written
    // (Eigen's LLT reads a single triangle) -- halves the O(P^2) 3x3 block updates -- and
    // W_b is reused from the precompute instead of re-fetched/transposed in the inner loop.
    for (int ia = 0; ia < P; ++ia) {
      const int offa = schur_off_[ia];
      const Eigen::Matrix3d &Ma = schur_Ma_[ia];
      for (int ib = 0; ib < P; ++ib) {
        const int offb = schur_off_[ib];
        if (offa < offb)
          continue; // lower triangle (incl. diagonal blocks) only
        Hred_.block(offa, offb, 3, 3).noalias() -= Ma * schur_W_[ib].transpose();
      }
    }
  }

  // In-place Cholesky on Hred_ (rebuilt every call) -- no factor-storage copy per trial.
  Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(Hred_);
  if (llt.info() != Eigen::Success)
    return false;
  da_ = rhs_;
  llt.matrixL().solveInPlace(da_);
  llt.matrixU().solveInPlace(da_);
  if (!da_.allFinite())
    return false;
  delta.head(n_nav_) = da_;

  // Back-substitute landmarks: d_l = -V^-1 (g_l + sum_a W_a^T d_a), with W_a^T read directly
  // from the landmark row-strip of the lower triangle.
  for (size_t li = 0; li < land_diag_.size(); ++li) {
    const int g0 = n_nav_ + land_diag_[li].first;
    Eigen::Matrix3d V = Eigen::Matrix3d(H.block(g0, g0, 3, 3).selfadjointView<Eigen::Lower>());
    for (int d = 0; d < 3; ++d)
      V(d, d) += lambda * std::min(std::max(H(g0 + d, g0 + d), 1e-6), 1e32);
    Eigen::Vector3d acc = grad.segment(g0, 3);
    for (int nb : land_adj_[li]) {
      const Block &b = blocks_[nb];
      acc.noalias() += H.block(g0, b.offset, 3, 3) * delta.segment(b.offset, 3);
    }
    delta.segment(g0, 3).noalias() = V.inverse() * (-acc);
  }
  return delta.allFinite();
}

double Problem::snapshot(std::vector<double> &backup) const {
  // Saves the variable-block ambient values AND returns their 2-norm: Ceres' x_norm_,
  // needed by the candidate-step parameter-tolerance check (relative, not absolute).
  backup.clear();
  double sq = 0.0;
  for (const auto &b : blocks_) {
    if (b.constant)
      continue;
    for (int i = 0; i < b.gsize; ++i) {
      backup.push_back(b.data[i]);
      sq += b.data[i] * b.data[i];
    }
  }
  return std::sqrt(sq);
}

void Problem::restore(const std::vector<double> &backup) {
  size_t p = 0;
  for (auto &b : blocks_) {
    if (b.constant)
      continue;
    for (int i = 0; i < b.gsize; ++i)
      b.data[i] = backup[p++];
  }
}

void Problem::apply_delta(const Eigen::VectorXd &delta) {
  double *tmp = plus_tmp_.data(); // preallocated in assign_ordering; no heap in the trial loop
  for (auto &b : blocks_) {
    if (b.constant)
      continue;
    const double *d = delta.data() + b.offset;
    if (b.param != nullptr) {
      b.param->Plus(b.data, d, tmp);
    } else {
      for (int i = 0; i < b.gsize; ++i)
        tmp[i] = b.data[i] + d[i];
    }
    for (int i = 0; i < b.gsize; ++i)
      b.data[i] = tmp[i];
  }
}

SolverSummary Problem::Solve(const SolverOptions &options) {
  SolverSummary summary;
  const auto t0 = std::chrono::steady_clock::now();

  if (options.pin_eigen_single_thread)
    Eigen::setNbThreads(1); // our ParallelExecutor owns the parallelism; keep Eigen serial

  n_jac_evals_ = 0;
  n_res_evals_ = 0;
  assign_ordering();
  if (n_total_ == 0) {
    summary.message = "no variable parameters";
    summary.converged = true;
    return summary;
  }

  ParallelExecutor exec(options.num_threads, options.worker_init_fn);

  // Wall-clock split (a handful of steady_clock reads per iteration; not in inner loops).
  double t_linearize = 0.0, t_linsolve = 0.0, t_residual = 0.0;

  Eigen::MatrixXd H;
  Eigen::VectorXd grad;
  double cost = 0.0;
  {
    const auto tt = std::chrono::steady_clock::now();
    linearize(H, grad, cost, exec); // residual + Jacobian + cost in one pass
    t_linearize += seconds_since(tt);
  }
  summary.initial_cost = cost;

  std::vector<double> backup;
  bool converged = false;

  if (options.use_dogleg) {
    // ===== Powell's traditional dogleg (faithful to Ceres' DoglegStrategy) =====
    // The Gauss-Newton and Cauchy steps are computed once per linearization; the trust-region
    // trials are then cheap GN/Cauchy blends, so a rejected trial costs NO new factorization.
    // The GN solve is mu-regularized (start 1e-8, retried x10 on a failed factorization) so a
    // near-rank-deficient linearization degrades gracefully instead of diverging.
    const double kMinMu = 1e-8, kMaxMu = 1.0, kMuFactor = 10.0;
    double radius = options.initial_radius;
    double mu = kMinMu;

    for (int iter = 0; iter < options.max_num_iterations; ++iter) {
      if (seconds_since(t0) > options.max_solver_time_seconds) {
        summary.message = "time budget reached";
        break;
      }
      if (grad.lpNorm<Eigen::Infinity>() < options.gradient_tolerance) {
        converged = true;
        summary.message = "gradient tolerance";
        break;
      }

      // Gauss-Newton step, mu-regularized, retried with larger mu on a failed solve.
      bool gn_ok = false;
      {
        const auto ts = std::chrono::steady_clock::now();
        double m = mu;
        for (int t = 0; t < 24; ++t) {
          if (solve_step(H, grad, m, dgn_)) {
            gn_ok = true;
            break;
          }
          m *= kMuFactor;
          if (m > kMaxMu)
            break;
        }
        t_linsolve += seconds_since(ts);
        if (!gn_ok) {
          summary.message = "GN solve failed";
          break;
        }
        mu = std::max(kMinMu, 2.0 * mu / kMuFactor); // relax on success
      }

      // Cauchy point: alpha = ||g||^2 / (g^T H g). (H holds the used lower triangle only.)
      Hg_.noalias() = H.selfadjointView<Eigen::Lower>() * grad;
      const double gg = grad.squaredNorm();
      const double gHg = grad.dot(Hg_);
      const double alpha = (gHg > 0.0) ? (gg / gHg) : 1.0;
      const double grad_norm = std::sqrt(gg);
      const double gn_norm = dgn_.norm();

      bool step_taken = false;
      while (true) {
        if (seconds_since(t0) > options.max_solver_time_seconds) {
          summary.message = "time budget reached";
          break;
        }

        // Traditional dogleg, tracked as step = c1*grad + c2*dgn. Then
        //   H*step = c1*(H*grad) + c2*(H*dgn) ~= c1*Hg_ - c2*grad
        // (H*dgn ~= -grad for the tiny GN regularization), so the predicted reduction needs
        // NO per-trial matvec -- the dominant per-iteration overhead dogleg otherwise has.
        double c1, c2;
        if (gn_norm <= radius) {
          c1 = 0.0;
          c2 = 1.0;
        } else if (grad_norm * alpha >= radius) {
          c1 = -(radius / grad_norm); // truncated Cauchy
          c2 = 0.0;
        } else {
          // interior blend between Cauchy point a = -alpha*g and GN step b = dgn_ (Ceres' beta)
          const double b_dot_a = -alpha * grad.dot(dgn_);
          const double a_sq = alpha * alpha * gg;
          const double bma_sq = a_sq - 2.0 * b_dot_a + gn_norm * gn_norm;
          const double c = b_dot_a - a_sq;
          const double dd = std::sqrt(std::max(0.0, c * c + bma_sq * (radius * radius - a_sq)));
          const double beta = (c <= 0.0) ? (dd - c) / bma_sq : (radius * radius - a_sq) / (dd + c);
          c1 = -alpha * (1.0 - beta);
          c2 = beta;
        }
        Hdl_.noalias() = c1 * grad + c2 * dgn_;
        const double pred = -(grad.dot(Hdl_) + 0.5 * Hdl_.dot(c1 * Hg_ - c2 * grad));

        const double x_norm = snapshot(backup);
        apply_delta(Hdl_);
        double cost_new;
        {
          const auto tr = std::chrono::steady_clock::now();
          cost_new = evaluate_cost(exec);
          t_residual += seconds_since(tr);
        }
        const double actual = cost - cost_new;
        const double rho = (pred > 0.0) ? (actual / pred) : (actual > 0.0 ? 1.0 : -1.0);
        const double step_norm = Hdl_.norm();

        // CERES-ORDER TERMINATION (TrustRegionMinimizer::Minimize, verified against tag 2.2.0):
        // parameter- and function-tolerance are checked on the CANDIDATE step BEFORE the
        // accept/reject decision (ptol additionally requires one prior successful step, per
        // Ceres 2.2.0's atleast_one_successful_step guard). A stall in a flat valley (the
        // free-S2 gravity <-> accel-bias ambiguity) therefore exits after ONE negligible trial
        // instead of shrinking the radius to collapse through repeated rejected trials. One
        // deviation from Ceres (which always discards the terminal candidate): we KEEP it iff
        // it decreased the cost -- never worse than Ceres' iterate, identical trial counts,
        // and identical to the pre-candidate-check terminal behavior.
        if ((summary.successful_steps > 0 &&
             step_norm <= options.parameter_tolerance * (x_norm + options.parameter_tolerance)) ||
            std::abs(actual) <= options.function_tolerance * std::abs(cost)) {
          if (actual > 0.0) {
            cost = cost_new;
            summary.successful_steps++;
          } else {
            restore(backup);
          }
          converged = true;
          summary.message = (step_norm <= options.parameter_tolerance * (x_norm + options.parameter_tolerance))
                                ? "parameter tolerance"
                                : "function tolerance";
          break;
        }

        if (rho > options.min_relative_decrease) {
          const double rel = actual / std::max(1e-12, cost);
          cost = cost_new;
          if (rho < 0.25)
            radius *= 0.5; // Ceres dogleg radius update
          else if (rho > 0.75)
            radius = std::min(options.max_radius, std::max(radius, 3.0 * step_norm));
          summary.successful_steps++;
          step_taken = true;
          if (options.verbose)
            std::fprintf(stderr, "[zbft/dl] it=%2d cost=%.8e rel=%.3e gnorm=%.3e radius=%.2e rho=%.3f\n", iter, cost, rel,
                         grad.lpNorm<Eigen::Infinity>(), radius, rho);
          // Re-linearize at the accepted iterate (the candidate checks above already handled
          // the terminal case, so an accepted step here always continues).
          {
            const auto tt = std::chrono::steady_clock::now();
            double cc = 0.0;
            linearize(H, grad, cc, exec);
            t_linearize += seconds_since(tt);
          }
          break;
        } else {
          restore(backup); // cheap reject: shrink radius and re-blend (NO factorization)
          radius *= 0.5;
          if (radius < 1e-12) {
            // Trust region collapsed with no further decrease => a stationary point (same rationale
            // as the LM max-damping case): CONVERGENCE, not failure. Gates vet the result downstream.
            converged = true;
            summary.message = "trust region collapsed (stationary)";
            break;
          }
        }
      }
      summary.iterations = iter + 1;
      if (!step_taken || converged)
        break;
    }
  } else {
    // ===== Levenberg-Marquardt (Madsen-Nielsen) fallback (use_dogleg=false) =====
    double lambda = options.initial_lambda;
    double nu = 2.0;

  for (int iter = 0; iter < options.max_num_iterations; ++iter) {
    if (seconds_since(t0) > options.max_solver_time_seconds) {
      summary.message = "time budget reached";
      break;
    }
    if (grad.lpNorm<Eigen::Infinity>() < options.gradient_tolerance) {
      converged = true;
      summary.message = "gradient tolerance";
      break;
    }

    // Marquardt damping diagonal: the same D that solve_step adds as lambda*D.
    lm_Dvec_ = H.diagonal().cwiseMax(1e-6).cwiseMin(1e32); // matches solve_step clamp

    bool step_taken = false;
    while (true) {
      if (seconds_since(t0) > options.max_solver_time_seconds) {
        summary.message = "time budget reached";
        break;
      }
      bool solved;
      {
        const auto ts = std::chrono::steady_clock::now();
        solved = solve_step(H, grad, lambda, lm_delta_);
        t_linsolve += seconds_since(ts);
      }
      if (!solved) {
        lambda *= nu;
        nu *= options.lm_nu_growth;
        if (lambda > options.max_lambda) {
          summary.message = "linear solve failed (rank deficient gauge?)";
          break;
        }
        continue;
      }

      // Predicted reduction of the quadratic model: 0.5 * delta^T (lambda*D*delta - grad).
      const double pred = 0.5 * lm_delta_.dot((lambda * lm_Dvec_.cwiseProduct(lm_delta_)) - grad);

      const double x_norm = snapshot(backup);
      apply_delta(lm_delta_);
      // CHEAP TRIAL: evaluate the COST ONLY (residual pass, no Jacobian) to score the step.
      // The expensive linearization is deferred to the accept-and-continue branch below, so a
      // rejected step costs only a residual pass (Ceres' "lazy" evaluation).
      double cost_new;
      {
        const auto tr = std::chrono::steady_clock::now();
        cost_new = evaluate_cost(exec);
        t_residual += seconds_since(tr);
      }
      const double actual = cost - cost_new;
      const double rho = (pred > 0.0) ? (actual / pred) : (actual > 0.0 ? 1.0 : -1.0);
      const double step_norm = lm_delta_.norm();

      // CERES-ORDER TERMINATION (TrustRegionMinimizer::Minimize, verified against tag 2.2.0):
      // parameter- and function-tolerance are checked on the CANDIDATE step BEFORE the
      // accept/reject decision (the ptol check additionally requires one prior successful
      // step, per Ceres 2.2.0's atleast_one_successful_step guard). This is what lets Ceres
      // exit a flat valley (the free-S2 gravity <-> accel-bias ambiguity) after ONE negligible
      // trial; checking only ACCEPTED steps kept rejecting trials while lambda escalated to
      // max_lambda -- ~10 extra Schur solves + residual passes per S2 solve. One deviation
      // from Ceres (which always discards the terminal candidate): we KEEP it iff it decreased
      // the cost -- never worse than Ceres' iterate, identical trial counts, and identical to
      // the pre-candidate-check terminal behavior on accepted steps.
      if ((summary.successful_steps > 0 &&
           step_norm <= options.parameter_tolerance * (x_norm + options.parameter_tolerance)) ||
          std::abs(actual) <= options.function_tolerance * std::abs(cost)) {
        if (actual > 0.0) {
          cost = cost_new;
          summary.successful_steps++;
        } else {
          restore(backup);
        }
        converged = true;
        summary.message = (step_norm <= options.parameter_tolerance * (x_norm + options.parameter_tolerance))
                              ? "parameter tolerance"
                              : "function tolerance";
        break;
      }

      if (rho > options.min_relative_decrease) {
        const double rel = actual / std::max(1e-12, cost);
        cost = cost_new;
        const double f = 2.0 * rho - 1.0;
        lambda *= std::max(1.0 / 3.0, 1.0 - f * f * f);
        lambda = std::max(lambda, options.min_lambda);
        nu = 2.0;
        summary.successful_steps++;
        step_taken = true;
        if (options.verbose)
          std::fprintf(stderr, "[zbft/lm] it=%2d cost=%.8e rel=%.3e gnorm=%.3e lambda=%.2e rho=%.3f\n", iter, cost, rel,
                       grad.lpNorm<Eigen::Infinity>(), lambda, rho);
        // Re-linearize at the accepted iterate (the candidate checks above already handled the
        // terminal case, so an accepted step here always continues to the next iteration).
        {
          const auto tt = std::chrono::steady_clock::now();
          double relin_cost = 0.0;
          linearize(H, grad, relin_cost, exec);
          t_linearize += seconds_since(tt);
        }
        break;
      } else {
        restore(backup);
        summary.rejected_steps++;
        lambda *= nu;
        nu *= options.lm_nu_growth;
        if (lambda > options.max_lambda) {
          // No descent direction even at maximum damping => a (local) stationary point: this is
          // CONVERGENCE, not failure. The free-gravity init's gravity/accel-bias ambiguity makes the
          // reduced Hessian near-singular along the ambiguous direction, so LM legitimately stalls AT
          // the minimum (the iterate is good -- verified consistent by the NEES gold standard). The
          // downstream gravity-direction gate + covariance-PD check still vet the result.
          converged = true;
          summary.message = "no further decrease at max damping (stationary)";
          break;
        }
      }
    }

    summary.iterations = iter + 1;
    if (!step_taken || converged)
      break;
    }
  } // end: if (options.use_dogleg) { ... } else { Levenberg-Marquardt }

  summary.final_cost = cost;
  summary.converged = converged;
  summary.solve_time_seconds = seconds_since(t0);
  summary.jacobian_evals = n_jac_evals_;
  summary.residual_evals = n_res_evals_;
  summary.time_linearize_seconds = t_linearize;
  summary.time_linear_solve_seconds = t_linsolve;
  summary.time_residual_seconds = t_residual;
  return summary;
}

bool Problem::ComputeCovariance(const std::vector<double *> &blocks, Eigen::MatrixXd &covariance, const SolverOptions &options) {
  assign_ordering();
  if (n_total_ == 0)
    return false;

  // Requested blocks must be variable & non-landmark (we return the landmark-marginalized
  // navigation covariance). Compute output ordering/size from the request.
  std::vector<std::pair<int, int>> req; // (nav-offset, lsize)
  int out_dim = 0;
  for (double *ptr : blocks) {
    int i = block_index(ptr);
    if (i < 0 || blocks_[i].constant || blocks_[i].landmark || blocks_[i].offset < 0 || blocks_[i].offset >= n_nav_)
      return false;
    req.emplace_back(blocks_[i].offset, blocks_[i].lsize);
    out_dim += blocks_[i].lsize;
  }

  ParallelExecutor exec(options.num_threads, options.worker_init_fn);
  Eigen::MatrixXd H;
  Eigen::VectorXd grad;
  double cov_cost = 0.0;
  linearize(H, grad, cov_cost, exec); // at the current (solved) iterate, undamped

  // Reduced navigation information with landmarks marginalized (lambda = 0).
  // H holds only the used lower triangle: B^T = H(land-rows, nav-cols) is stored directly,
  // the nav-nav block is materialized from its lower half.
  Eigen::MatrixXd Hred;
  if (n_land_ == 0) {
    Hred = H.topLeftCorner(n_nav_, n_nav_).selfadjointView<Eigen::Lower>();
  } else {
    // Hred = Hnn - (B * D^-1) * B^T with D BLOCK-diagonal: scale B's landmark column-blocks by
    // the small (lsize x lsize) inverses directly -- never materialize the dense
    // n_land x n_land D^-1 (that costs an O(n_land^2) zero-fill plus an O(n_nav*n_land^2)
    // gemm for what is O(n_nav*n_land*lsize) work).
    // Small ridge keeps weakly-observed (near-singular) landmark blocks invertible;
    // it perturbs the marginal only at the ~1e-8 level for well-observed landmarks.
    const auto Bt = H.block(n_nav_, 0, n_land_, n_nav_); // = B^T (landmark row-strip, always written)
    Eigen::MatrixXd BD(n_nav_, n_land_);
    for (const auto &od : land_diag_) {
      const int off = od.first;
      const int ls = od.second;
      Eigen::MatrixXd Dblk =
          Eigen::MatrixXd(H.block(n_nav_ + off, n_nav_ + off, ls, ls).selfadjointView<Eigen::Lower>());
      Dblk.diagonal().array() += 1e-10;
      BD.middleCols(off, ls).noalias() = Bt.middleRows(off, ls).transpose() * Dblk.inverse();
    }
    Hred = Eigen::MatrixXd(H.topLeftCorner(n_nav_, n_nav_).selfadjointView<Eigen::Lower>()) - BD * Bt;
  }

  // Invert (requires the gauge to be anchored -> PD). Marginal covariance = Hred^{-1}.
  Eigen::LDLT<Eigen::MatrixXd> ldlt(Hred);
  if (ldlt.info() != Eigen::Success)
    return false;
  Eigen::MatrixXd Sigma = ldlt.solve(Eigen::MatrixXd::Identity(n_nav_, n_nav_));
  if (!Sigma.allFinite() || (ldlt.vectorD().array() <= 0.0).any())
    return false;

  // Extract requested sub-blocks in the requested order.
  covariance.resize(out_dim, out_dim);
  int ri = 0;
  for (size_t a = 0; a < req.size(); ++a) {
    int rj = 0;
    for (size_t b = 0; b < req.size(); ++b) {
      covariance.block(ri, rj, req[a].second, req[b].second) =
          Sigma.block(req[a].first, req[b].first, req[a].second, req[b].second);
      rj += req[b].second;
    }
    ri += req[a].second;
  }
  covariance = 0.5 * (covariance + covariance.transpose()).eval(); // symmetrize
  return true;
}
