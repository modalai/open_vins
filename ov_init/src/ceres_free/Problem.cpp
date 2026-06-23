/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
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

double Problem::evaluate_cost() const {
  ++n_res_evals_;
  double cost = 0.0;
  std::vector<const double *> params;
  Eigen::VectorXd rbuf; // residual scratch, grown once to the max residual count (no per-residual heap alloc)
  for (const auto &res : residuals_) {
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
      cost += 0.5 * rho[0];
    } else {
      cost += 0.5 * s;
    }
  }
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
  if ((int)Hw_.size() != W) {
    Hw_.assign(W, Eigen::MatrixXd(N, N));
    gw_.assign(W, Eigen::VectorXd(N));
    cw_.assign(W, 0.0);
  }
  for (int w = 0; w < W; ++w) {
    if (Hw_[w].rows() != N)
      Hw_[w].resize(N, N);
    if (gw_[w].size() != N)
      gw_[w].resize(N);
    Hw_[w].setZero();
    gw_[w].setZero();
    cw_[w] = 0.0;
  }

  const auto body = [&](int worker, int begin, int end) {
    Eigen::MatrixXd &Hloc = Hw_[worker];
    Eigen::VectorXd &gloc = gw_[worker];
    double &cl = cw_[worker];
    std::vector<const double *> params;
    std::vector<double *> jacptrs;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Jstore;
    std::vector<int> vidx; // global block index of each variable block
    std::vector<int> vk;   // its position in res.blocks / Jstore
    Eigen::VectorXd rbuf;  // residual scratch, grown to the max residual count once (no per-residual heap alloc)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 16, 16> Mab; // off-diagonal Hessian block scratch (stack, no heap)

    for (int ri = begin; ri < end; ++ri) {
      const Residual &res = residuals_[ri];
      const int nb = (int)res.blocks.size();
      const int nres = res.cost->num_residuals();

      params.assign(nb, nullptr);
      jacptrs.assign(nb, nullptr);
      Jstore.resize(nb);
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

      // Accumulate gradient/Hessian using the manifold Jacobian = leading lsize columns
      // of the ambient Jacobian (no V multiply). Block expressions, no per-residual alloc.
      // H is symmetric, so each off-diagonal product J_a^T J_b is formed ONCE and scattered
      // to both (a,b) and its transpose (b,a) -- ~halving the per-factor Hessian products --
      // while still storing the full H (the Schur reduction reads its nav<->landmark coupling).
      for (size_t a = 0; a < vidx.size(); ++a) {
        const Block &ba = blocks_[vidx[a]];
        const auto Ja = Jstore[vk[a]].leftCols(ba.lsize);
        gloc.segment(ba.offset, ba.lsize).noalias() += w * (Ja.transpose() * r);
        Hloc.block(ba.offset, ba.offset, ba.lsize, ba.lsize).noalias() += w * (Ja.transpose() * Ja);
        for (size_t b = a + 1; b < vidx.size(); ++b) {
          const Block &bb = blocks_[vidx[b]];
          Mab.noalias() = w * (Ja.transpose() * Jstore[vk[b]].leftCols(bb.lsize)); // lsize_a x lsize_b
          Hloc.block(ba.offset, bb.offset, ba.lsize, bb.lsize).noalias() += Mab;
          Hloc.block(bb.offset, ba.offset, bb.lsize, ba.lsize).noalias() += Mab.transpose();
        }
      }
    }
  };

  exec.parallel_ranges((int)residuals_.size(), body);

  // Deterministic reduction in fixed worker order (bit-identical regardless of W).
  H = Hw_[0];
  grad = gw_[0];
  cost = cw_[0];
  for (int w = 1; w < W; ++w) {
    H += Hw_[w];
    grad += gw_[w];
    cost += cw_[w];
  }
}

static Eigen::MatrixXd Problem_build_Dinv(const Eigen::MatrixXd &Hsrc, int n_nav, int n_land,
                                          const std::vector<std::pair<int, int>> &land_diag, double ridge) {
  Eigen::MatrixXd Dinv = Eigen::MatrixXd::Zero(n_land, n_land);
  for (const auto &od : land_diag) {
    const int off = od.first;
    const int ls = od.second;
    Eigen::MatrixXd Dblk = Hsrc.block(n_nav + off, n_nav + off, ls, ls);
    if (ridge > 0.0)
      Dblk.diagonal().array() += ridge;
    Dinv.block(off, off, ls, ls) = Dblk.inverse();
  }
  return Dinv;
}

bool Problem::solve_step(const Eigen::MatrixXd &H, const Eigen::VectorXd &grad, double lambda, Eigen::VectorXd &delta) const {
  delta.resize(n_total_);

  // Landmark-free (inertial-only) path: damped dense LDLT (preallocated buffers).
  if (n_land_ == 0 || n_nav_ == 0) {
    if (Hd_.rows() != n_total_)
      Hd_.resize(n_total_, n_total_);
    Hd_ = H;
    for (int i = 0; i < n_total_; ++i)
      Hd_(i, i) += lambda * std::min(std::max(H(i, i), 1e-6), 1e32); // Ceres min/max_lm_diagonal clamp
    lltRed_.compute(Hd_);
    if (lltRed_.info() != Eigen::Success)
      return false;
    delta.noalias() = lltRed_.solve(-grad);
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
  Hred_ = H.topLeftCorner(n_nav_, n_nav_);
  for (int i = 0; i < n_nav_; ++i)
    Hred_(i, i) += lambda * std::min(std::max(H(i, i), 1e-6), 1e32);
  rhs_.noalias() = -grad.head(n_nav_);

  for (size_t li = 0; li < land_diag_.size(); ++li) {
    const int g0 = n_nav_ + land_diag_[li].first;
    Eigen::Matrix3d V = H.block(g0, g0, 3, 3);
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
    // Precompute, per observing pose block: its nav offset, W_a = H[off_a, landmark],
    // and M_a = W_a * V^-1. Also fold the rhs update -ga += M_a * g_l here (one pass over P).
    for (int ia = 0; ia < P; ++ia) {
      const int off = blocks_[adj[ia]].offset;
      schur_off_[ia] = off;
      schur_W_[ia] = H.block(off, g0, 3, 3);
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

  lltRed_.compute(Hred_);
  if (lltRed_.info() != Eigen::Success)
    return false;
  da_.noalias() = lltRed_.solve(rhs_);
  if (!da_.allFinite())
    return false;
  delta.head(n_nav_) = da_;

  // Back-substitute landmarks: d_l = -V^-1 (g_l + sum_a W_a^T d_a).
  for (size_t li = 0; li < land_diag_.size(); ++li) {
    const int g0 = n_nav_ + land_diag_[li].first;
    Eigen::Matrix3d V = H.block(g0, g0, 3, 3);
    for (int d = 0; d < 3; ++d)
      V(d, d) += lambda * std::min(std::max(H(g0 + d, g0 + d), 1e-6), 1e32);
    Eigen::Vector3d acc = grad.segment(g0, 3);
    for (int nb : land_adj_[li]) {
      const Block &b = blocks_[nb];
      acc.noalias() += H.block(b.offset, g0, 3, 3).transpose() * delta.segment(b.offset, 3);
    }
    delta.segment(g0, 3).noalias() = V.inverse() * (-acc);
  }
  return delta.allFinite();
}

void Problem::snapshot(std::vector<double> &backup) const {
  backup.clear();
  for (const auto &b : blocks_) {
    if (b.constant)
      continue;
    for (int i = 0; i < b.gsize; ++i)
      backup.push_back(b.data[i]);
  }
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
  std::vector<double> tmp;
  for (auto &b : blocks_) {
    if (b.constant)
      continue;
    const double *d = delta.data() + b.offset;
    tmp.assign(b.gsize, 0.0);
    if (b.param != nullptr) {
      b.param->Plus(b.data, d, tmp.data());
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

  Eigen::MatrixXd H;
  Eigen::VectorXd grad;
  double cost = 0.0;
  linearize(H, grad, cost, exec); // residual + Jacobian + cost in one pass
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
        if (!gn_ok) {
          summary.message = "GN solve failed";
          break;
        }
        mu = std::max(kMinMu, 2.0 * mu / kMuFactor); // relax on success
      }

      // Cauchy point: alpha = ||g||^2 / (g^T H g).
      Hg_.noalias() = H * grad;
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

        snapshot(backup);
        apply_delta(Hdl_);
        const double cost_new = evaluate_cost();
        const double actual = cost - cost_new;
        const double rho = (pred > 0.0) ? (actual / pred) : (actual > 0.0 ? 1.0 : -1.0);
        const double step_norm = Hdl_.norm();

        if (rho > options.min_relative_decrease) {
          const double rel = actual / std::max(1e-12, cost);
          cost = cost_new;
          if (rho < 0.25)
            radius *= 0.5; // Ceres dogleg radius update
          else if (rho > 0.75)
            radius = std::min(options.max_radius, std::max(radius, 3.0 * step_norm));
          summary.successful_steps++;
          step_taken = true;
          if (rel >= 0.0 && rel < options.function_tolerance) {
            converged = true;
            summary.message = "function tolerance";
          }
          if (step_norm < options.parameter_tolerance) {
            converged = true;
            summary.message = "parameter tolerance";
          }
          if (options.verbose)
            std::fprintf(stderr, "[zbft/dl] it=%2d cost=%.8e rel=%.3e gnorm=%.3e radius=%.2e rho=%.3f\n", iter, cost, rel,
                         grad.lpNorm<Eigen::Infinity>(), radius, rho);
          // Re-linearize at the accepted iterate ONLY if continuing (the Jacobian is needed
          // solely by the next iteration); skip it on convergence to avoid a wasted linearization.
          if (!converged) {
            double cc = 0.0;
            linearize(H, grad, cc, exec);
          }
          break;
        } else {
          restore(backup); // cheap reject: shrink radius and re-blend (NO factorization)
          radius *= 0.5;
          if (radius < 1e-12) {
            summary.message = "trust region collapsed";
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
      if (!solve_step(H, grad, lambda, lm_delta_)) {
        lambda *= nu;
        nu *= 2.0;
        if (lambda > options.max_lambda) {
          summary.message = "linear solve failed (rank deficient gauge?)";
          break;
        }
        continue;
      }

      // Predicted reduction of the quadratic model: 0.5 * delta^T (lambda*D*delta - grad).
      const double pred = 0.5 * lm_delta_.dot((lambda * lm_Dvec_.cwiseProduct(lm_delta_)) - grad);

      snapshot(backup);
      apply_delta(lm_delta_);
      // CHEAP TRIAL: evaluate the COST ONLY (residual pass, no Jacobian) to score the step.
      // The expensive linearization is deferred to the accept-and-continue branch below, so a
      // rejected step costs only a residual pass (Ceres' "lazy" evaluation).
      const double cost_new = evaluate_cost();
      const double actual = cost - cost_new;
      const double rho = (pred > 0.0) ? (actual / pred) : (actual > 0.0 ? 1.0 : -1.0);

      if (rho > options.min_relative_decrease) {
        const double rel = actual / std::max(1e-12, cost);
        const double step_norm = lm_delta_.norm();
        cost = cost_new;
        const double f = 2.0 * rho - 1.0;
        lambda *= std::max(1.0 / 3.0, 1.0 - f * f * f);
        lambda = std::max(lambda, options.min_lambda);
        nu = 2.0;
        summary.successful_steps++;
        step_taken = true;
        if (rel >= 0.0 && rel < options.function_tolerance) {
          converged = true;
          summary.message = "function tolerance";
        }
        if (step_norm < options.parameter_tolerance) {
          converged = true;
          summary.message = "parameter tolerance";
        }
        if (options.verbose)
          std::fprintf(stderr, "[zbft/lm] it=%2d cost=%.8e rel=%.3e gnorm=%.3e lambda=%.2e rho=%.3f\n", iter, cost, rel,
                       grad.lpNorm<Eigen::Infinity>(), lambda, rho);
        // Re-linearize at the accepted iterate ONLY if we are continuing -- the Hessian/gradient
        // are needed solely by the next iteration. On convergence we skip it (this is the single
        // wasted linearization that previously made the LM path ~1 linearize/solve slower than Ceres).
        if (!converged) {
          double relin_cost = 0.0;
          linearize(H, grad, relin_cost, exec);
        }
        break;
      } else {
        restore(backup);
        summary.rejected_steps++;
        lambda *= nu;
        nu *= 2.0;
        if (lambda > options.max_lambda) {
          summary.message = "no further decrease (lambda max)";
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
  Eigen::MatrixXd Hred;
  if (n_land_ == 0) {
    Hred = H.topLeftCorner(n_nav_, n_nav_);
  } else {
    // Small ridge keeps weakly-observed (near-singular) landmark blocks invertible;
    // it perturbs the marginal only at the ~1e-8 level for well-observed landmarks.
    const Eigen::MatrixXd Dinv = Problem_build_Dinv(H, n_nav_, n_land_, land_diag_, 1e-10);
    const Eigen::MatrixXd B = H.block(0, n_nav_, n_nav_, n_land_);
    Hred = H.topLeftCorner(n_nav_, n_nav_) - (B * Dinv) * B.transpose();
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
