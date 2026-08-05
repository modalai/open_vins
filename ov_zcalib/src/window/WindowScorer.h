/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: window reservoir (streaming diversity retention) + solve-time
 * D-optimal selection.
 *
 * Retention is a max-min diversity reservoir over the 12-dim excitation
 * fingerprint (generalizing the rotation-axis-only criterion): each slot
 * caches its max similarity to any other slot, so admission/eviction is O(N)
 * amortized. Every holdout_every-th ACCEPTED window is flagged holdout: never
 * evicted, never fused -- the VERIFY stage's held-out data. Selection at solve
 * time is greedy logdet (D-optimal, submodular (1-1/e) on the prior-whitened
 * one-pass information) with a time-overlap penalty against double counting,
 * and the whitened minimum eigenvalue (E-optimal) is reported alongside --
 * min-eig catches near-degenerate selections that logdet smooths over. The
 * degenerate-motion gating this implements follows the observability analysis
 * of Yang et al. (RSS 2020 / T-RO 2023); windows that never excite a subspace
 * leave its posterior at the prior, and the partial-commit rule refuses it.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_WINDOW_SCORER_H
#define OV_ZCALIB_WINDOW_SCORER_H

#include <Eigen/Dense>
#include <vector>

#include "WindowHarvester.h"

namespace ov_zcalib {

struct ScorerConfig {
  int capacity = 64;      ///< retained-window budget (~1 MB each: IMU + tracks)
  int holdout_every = 5;  ///< every k-th accepted window is holdout (never fused/evicted)
  double temp_span_gate = 6.0; ///< deg C: max fused-set temperature span (thermal rule)
  /// fingerprint normalization scales (nominal excitation units per dim):
  /// [ std_w xyz | std_a xyz | grav sweep | flow px/s | row cov | parallax | dur s | temp span ]
  Eigen::Matrix<double, 12, 1> fp_scale =
      (Eigen::Matrix<double, 12, 1>() << 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 150.0, 0.25, 0.3, 4.0, 3.0).finished();
};

struct ReservoirDecision {
  bool accepted = false;
  bool is_holdout = false;
  int slot = -1;         ///< storage slot assigned (stable across the session)
  int evicted_slot = -1; ///< slot whose window must be discarded first (-1 none)
};

class WindowScorer {
public:
  explicit WindowScorer(const ScorerConfig &cfg) : cfg_(cfg) {}

  /// Streaming admission. Slots are [0, capacity): the caller owns the actual
  /// WindowData storage indexed by slot.
  ReservoirDecision consider(const WindowMeta &meta);

  /// Read-only admission probe: would consider() accept this window right now?
  /// Admission is FINGERPRINT-only by design, so the probe is exact -- the
  /// session uses it to pay the nonlinear window solve/export ONLY for windows
  /// that will actually be retained (on long collections every overflow window
  /// otherwise burns a full BA just to be discarded).
  bool peek(const WindowMeta &meta) const {
    if ((int)slots_.size() < cfg_.capacity)
      return true;
    Slot s;
    s.meta = meta;
    s.nfp = meta.fingerprint.cwiseQuotient(cfg_.fp_scale);
    double new_max = -1.0, worst_sim = -1.0;
    bool have_evictable = false;
    for (const Slot &sj : slots_) {
      new_max = std::max(new_max, sim_(s, sj));
      if (!sj.holdout && sj.max_sim > worst_sim) {
        worst_sim = sj.max_sim;
        have_evictable = true;
      }
    }
    return have_evictable && new_max < worst_sim;
  }

  int size() const { return (int)slots_.size(); }
  bool is_holdout(int slot) const { return slots_[slot].holdout; }
  /// Retro-designate a retained slot as holdout. Valid ONLY before the slot has
  /// influenced any solve (windows are display-only until the end-of-session
  /// fusion): the every-k rule leaves 0 holdouts on short reservoirs, and an
  /// unverifiable session must never commit.
  void force_holdout(int slot) { slots_[slot].holdout = true; }
  const WindowMeta &meta(int slot) const { return slots_[slot].meta; }

  /// Fused-set thermal gate: bins the NON-holdout slots by temperature so no
  /// fused subset spans more than temp_span_gate. Returns the slot list of the
  /// LARGEST bin (the fusion set); the rest stay retained for later sessions.
  std::vector<int> thermal_bin() const;

  /**
   * @brief Greedy D-optimal selection over prior-whitened one-pass window
   *        information, with a pairwise time-overlap penalty.
   * @param Lw       whitened Lambda_w per candidate (same np x np size)
   * @param spans    camera-clock [t0,t1] per candidate (overlap penalty)
   * @param K        max windows to select
   * @param overlap_penalty  logdet-gain multiplier per unit overlap fraction
   * @param min_eig_out  whitened min eigenvalue of I + sum(selected) (E-opt report)
   */
  static std::vector<int> select_logdet(const std::vector<Eigen::MatrixXd> &Lw, const std::vector<std::pair<double, double>> &spans,
                                        int K, double overlap_penalty, double *min_eig_out);

private:
  struct Slot {
    WindowMeta meta;
    Eigen::Matrix<double, 12, 1> nfp; ///< normalized fingerprint
    bool holdout = false;
    double max_sim = -1.0; ///< cached max similarity to any other slot
    int max_idx = -1;
  };
  double sim_(const Slot &a, const Slot &b) const {
    const double na = a.nfp.norm(), nb = b.nfp.norm();
    return (na > 1e-12 && nb > 1e-12) ? a.nfp.dot(b.nfp) / (na * nb) : 1.0;
  }

  ScorerConfig cfg_;
  std::vector<Slot> slots_;
  int accepted_count_ = 0;
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_WINDOW_SCORER_H
