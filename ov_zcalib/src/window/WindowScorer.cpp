/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: window reservoir + D-optimal selection (see WindowScorer.h).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "WindowScorer.h"

#include <algorithm>
#include <cmath>
#include <map>

using namespace ov_zcalib;

ReservoirDecision WindowScorer::consider(const WindowMeta &meta) {
  ReservoirDecision d;
  Slot s;
  s.meta = meta;
  s.nfp = meta.fingerprint.cwiseQuotient(cfg_.fp_scale);
  s.holdout = ((accepted_count_ % cfg_.holdout_every) == cfg_.holdout_every - 1);

  if ((int)slots_.size() < cfg_.capacity) {
    // under capacity: append + O(N) cache update
    const int n = (int)slots_.size();
    for (int j = 0; j < n; ++j) {
      const double sim = sim_(s, slots_[j]);
      if (sim > s.max_sim) {
        s.max_sim = sim;
        s.max_idx = j;
      }
      if (sim > slots_[j].max_sim) {
        slots_[j].max_sim = sim;
        slots_[j].max_idx = n;
      }
    }
    slots_.push_back(s);
    d.accepted = true;
    d.slot = n;
    d.is_holdout = s.holdout;
    ++accepted_count_;
    return d;
  }

  // at capacity: most-redundant NON-holdout slot vs the candidate
  const int n = (int)slots_.size();
  std::vector<double> sim_to_new(n);
  double new_max = -1.0;
  int new_max_idx = -1;
  for (int j = 0; j < n; ++j) {
    sim_to_new[j] = sim_(s, slots_[j]);
    if (sim_to_new[j] > new_max) {
      new_max = sim_to_new[j];
      new_max_idx = j;
    }
  }
  int worst = -1;
  double worst_sim = -1.0;
  for (int j = 0; j < n; ++j) {
    if (slots_[j].holdout)
      continue; // holdout windows are never evicted
    if (slots_[j].max_sim > worst_sim) {
      worst_sim = slots_[j].max_sim;
      worst = j;
    }
  }
  if (worst < 0 || new_max >= worst_sim)
    return d; // candidate adds no diversity

  s.max_sim = new_max;
  s.max_idx = (new_max_idx == worst) ? -1 : new_max_idx;
  slots_[worst] = s;
  // cache repair: full rescan only for slots orphaned by the eviction
  for (int j = 0; j < n; ++j) {
    if (j == worst)
      continue;
    Slot &wj = slots_[j];
    const double sim_repl = sim_to_new[j];
    if (wj.max_idx == worst) {
      if (sim_repl >= wj.max_sim - 1e-12) {
        wj.max_sim = sim_repl;
      } else {
        wj.max_sim = -1.0;
        wj.max_idx = -1;
        for (int k = 0; k < n; ++k) {
          if (k == j)
            continue;
          const double sk = sim_(wj, slots_[k]);
          if (sk > wj.max_sim) {
            wj.max_sim = sk;
            wj.max_idx = k;
          }
        }
      }
    } else if (sim_repl > wj.max_sim) {
      wj.max_sim = sim_repl;
      wj.max_idx = worst;
    }
  }
  d.accepted = true;
  d.slot = worst;
  d.evicted_slot = worst;
  d.is_holdout = s.holdout;
  ++accepted_count_;
  return d;
}

std::vector<int> WindowScorer::thermal_bin() const {
  // Greedy 1-D binning on temp_mean: sort non-holdout slots, sweep a window of
  // width temp_span_gate, return the densest bin.
  std::vector<std::pair<double, int>> ts;
  for (int j = 0; j < (int)slots_.size(); ++j)
    if (!slots_[j].holdout)
      ts.push_back({slots_[j].meta.temp_mean, j});
  std::sort(ts.begin(), ts.end());
  std::vector<int> best;
  for (size_t i = 0; i < ts.size(); ++i) {
    std::vector<int> bin;
    for (size_t j = i; j < ts.size() && ts[j].first - ts[i].first <= cfg_.temp_span_gate; ++j)
      bin.push_back(ts[j].second);
    if (bin.size() > best.size())
      best = bin;
  }
  std::sort(best.begin(), best.end());
  return best;
}

std::vector<int> WindowScorer::select_logdet(const std::vector<Eigen::MatrixXd> &Lw, const std::vector<std::pair<double, double>> &spans,
                                             int K, double overlap_penalty, double *min_eig_out) {
  std::vector<int> sel;
  if (Lw.empty())
    return sel;
  const int np = (int)Lw[0].rows();
  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(np, np); // whitened prior
  auto logdet = [&](const Eigen::MatrixXd &M) {
    Eigen::LLT<Eigen::MatrixXd> llt(M);
    if (llt.info() != Eigen::Success)
      return -1e300;
    return 2.0 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
  };
  auto overlap = [&](int a, int b) {
    const double lo = std::max(spans[a].first, spans[b].first);
    const double hi = std::min(spans[a].second, spans[b].second);
    const double la = spans[a].second - spans[a].first;
    return (hi > lo && la > 0.0) ? (hi - lo) / la : 0.0;
  };
  std::vector<char> used(Lw.size(), 0);
  double base = logdet(A);
  for (int round = 0; round < K; ++round) {
    int best = -1;
    double best_gain = 1e-9; // require strictly positive information gain
    for (size_t c = 0; c < Lw.size(); ++c) {
      if (used[c] || (int)Lw[c].rows() != np)
        continue;
      double gain = logdet(A + Lw[c]) - base;
      for (int s : sel)
        gain -= overlap_penalty * overlap((int)c, s) * std::abs(gain);
      if (gain > best_gain) {
        best_gain = gain;
        best = (int)c;
      }
    }
    if (best < 0)
      break;
    A += Lw[best];
    base = logdet(A);
    used[best] = 1;
    sel.push_back(best);
  }
  if (min_eig_out) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(A);
    *min_eig_out = eig.eigenvalues()(0);
  }
  std::sort(sel.begin(), sel.end());
  return sel;
}
