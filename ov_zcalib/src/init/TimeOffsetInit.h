/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * ov_zcalib: coarse camera-IMU time-offset seed by normalized cross-correlation
 * of rotation-rate magnitudes (the kalibr timeshift prior, with interpolated
 * lag sampling and sub-step parabolic refinement). Magnitudes are invariant to
 * the unknown R_ItoC, and the gyro bias only perturbs them at the mrad/s level
 * against rad/s-scale excitation, so this runs before any other estimate
 * exists. OpenVINS convention: t_imu = t_cam + td, so the score at lag tau
 * compares the camera rate at t with the IMU rate at t + tau, and the argmax
 * IS the td seed directly.
 *
 * A high peak or agreement between even/odd samples does not establish clock
 * accuracy: adjacent visual pairs can share translation and rolling-shutter
 * bias. Weighted trimming and interleaved splits remain evidence about coarse
 * peak repeatability. A separate unweighted first/last time split detects
 * motion-dependent drift. The runner can geometrically refine that seed. If
 * image support is too sparse, an interior hand-eye result remains a
 * provisional start; it does not acquire a timing certificate from xcorr.
 *
 * The primary scan remains unweighted, with the legacy arithmetic unchanged.
 * Its td is always the full-series refined argmax; diagnostic trimming and
 * splits do not move the raw peak. Geometry exports its own estimate instead
 * of changing the sign or relabeling the correlation curve. Keep the primary
 * loop separate: reassociation under fast-math can affect downstream window
 * selection even when two formulations are mathematically equivalent.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_TIME_OFFSET_INIT_H
#define OV_ZCALIB_TIME_OFFSET_INIT_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "../cpi/AciCalibPreint.h"

namespace ov_zcalib {

/// One camera rotation-rate sample: |log R_C1toC2| / dt stamped at the pair midtime.
struct CamRateSample {
  double t_mid = 0.0;
  double rate = 0.0;   ///< rad/s
  double weight = 1.0; ///< front-end pair quality (the SAME trust the hand-eye pairs carry)
};

struct TimeOffsetResult {
  bool ok = false;
  double td = 0.0;             ///< seconds, t_imu = t_cam + td (full-series refined argmax)
  double peak_corr = 0.0;      ///< unweighted normalized xcorr at the peak (quality; 1 = perfect)
  double peak_sharpness = 0.0; ///< -d2(corr)/dtau2 at the peak (identifiability)
  bool at_bound = false;       ///< peak pinned at the search edge: widen the search
  // ---- robust-certificate evidence (computed only for a usable interior peak) ----
  double peak_trimmed = 0.0;   ///< peak after one Hampel trim at the found lag + full rescan
  double trim_retention = 1.0; ///< weighted fraction of evidence the trim kept (1 = kept all)
  bool trim_consistent = false;///< trimmed argmax within 2 lag steps of the untrimmed one
  double td_split_delta = -1.0;///< |td(even half) - td(odd half)| [s]; negative = halves unusable
  double split_min_peak = 0.0; ///< the WEAKER half's peak (both halves must show a real ridge)
  double temporal_split_delta = -1.0; ///< first/last time halves; adjacent samples are not independent evidence
  double temporal_min_peak = 0.0;
  bool temporally_consistent(double tolerance) const {
    return temporal_split_delta >= 0.0 && temporal_split_delta <= tolerance && temporal_min_peak >= 0.36;
  }
};

/// Optional copy of the full-series correlation scan; does not alter the estimate.
struct XcorrCurve {
  double search = 0.0; ///< lag search half-width [s]; lag_i = -search + i*step
  double step = 0.0;   ///< lag grid step [s]
  /// Unweighted normalized correlation per lag, from the FULL-SERIES scan.
  /// A lag with fewer than 8 overlapping samples is left at the -2.0 sentinel
  /// (plot it as a gap, never as a correlation of -2).
  std::vector<double> corr;
};

class TimeOffsetInit {
public:
  /**
   * @brief Normalized xcorr over a lag grid with linear IMU interpolation.
   * @param cam    camera rotation-rate samples (any spacing; gaps fine; weight 1 = legacy)
   * @param imu    raw IMU (only wm used; bias left in -- see header note)
   * @param search half-width of the lag search [s]
   * @param step   coarse lag step [s] (parabolic refine goes sub-step)
   * @param curve  optional diagnostic export of the full-series correlation
   *               curve; nullptr (the default) allocates nothing and leaves
   *               every computed value bit-identical
   */
  static TimeOffsetResult solve(const std::vector<CamRateSample> &cam, const std::vector<RawImu> &imu, double search = 0.10,
                                double step = 0.002, XcorrCurve *curve = nullptr) {
    TimeOffsetResult out;
    if (cam.size() < 8 || imu.size() < 8 || !(search > 0.0) || !(step > 0.0))
      return out;

    // |w| interpolator over the (sorted) IMU stream
    auto imu_rate_at = [&](double t, double &val) -> bool {
      auto it = std::lower_bound(imu.begin(), imu.end(), t, [](const RawImu &s, double tq) { return s.timestamp < tq; });
      if (it == imu.begin() || it == imu.end())
        return false;
      const RawImu &s1 = *it, &s0 = *std::prev(it);
      const double dt = s1.timestamp - s0.timestamp;
      if (!(dt > 1e-12))
        return false;
      const double lam = (t - s0.timestamp) / dt;
      val = (1.0 - lam) * s0.wm.norm() + lam * s1.wm.norm();
      return true;
    };

    const int n_lags = (int)std::floor(2.0 * search / step) + 1;

    // Subset scans: weighted for trim/interleaved diagnostics, unweighted for
    // the first/last time intervals. The primary scan stays separate below.
    struct ScanOut {
      bool ok = false;
      int imax = 0;
      double td = 0.0, peak = -2.0, sharpness = 0.0;
      bool at_bound = false;
    };
    auto scan = [&](const std::vector<int> &idx, bool use_weights) -> ScanOut {
      ScanOut so;
      if (idx.size() < 8)
        return so;
      // Camera-side (weighted) mean once, over this subset
      double mu_c = 0.0, w_mu = 0.0;
      for (int k : idx) {
        const double w = use_weights ? cam[(size_t)k].weight : 1.0;
        mu_c += w * cam[(size_t)k].rate;
        w_mu += w;
      }
      if (!(w_mu > 0.0))
        return so;
      mu_c /= w_mu;
      std::vector<double> score((size_t)n_lags, -2.0);
      for (int li = 0; li < n_lags; ++li) {
        const double tau = -search + li * step;
        double sc = 0.0, si = 0.0, si2 = 0.0, scc = 0.0, cross = 0.0, W = 0.0;
        int cnt = 0;
        for (int k : idx) {
          const CamRateSample &c = cam[(size_t)k];
          double v;
          if (!imu_rate_at(c.t_mid + tau, v))
            continue;
          const double dc = c.rate - mu_c;
          const double w = use_weights ? c.weight : 1.0;
          sc += w * dc;
          scc += w * dc * dc;
          si += w * v;
          si2 += w * v * v;
          cross += w * dc * v;
          W += w;
          ++cnt;
        }
        if (cnt < 8 || !(W > 0.0))
          continue;
        // rho = cov(c,i)/sqrt(var(c) var(i)) with the camera mean fixed globally
        const double mu_i = si / W;
        const double var_i = si2 / W - mu_i * mu_i;
        const double cov = cross / W - (sc / W) * mu_i;
        const double var_c = scc / W - (sc / W) * (sc / W);
        if (var_i <= 1e-12 || var_c <= 1e-12)
          continue;
        score[(size_t)li] = cov / std::sqrt(var_c * var_i);
      }
      int imax = 0;
      for (int li = 1; li < n_lags; ++li)
        if (score[(size_t)li] > score[(size_t)imax])
          imax = li;
      if (score[(size_t)imax] <= -2.0)
        return so;
      so.imax = imax;
      so.peak = score[(size_t)imax];
      so.td = -search + imax * step;
      so.at_bound = (imax == 0 || imax == n_lags - 1);
      // Sub-step parabolic refinement + sharpness at the peak
      if (imax > 0 && imax < n_lags - 1 && score[(size_t)imax - 1] > -2.0 && score[(size_t)imax + 1] > -2.0) {
        const double y0 = score[(size_t)imax - 1], y1 = score[(size_t)imax], y2 = score[(size_t)imax + 1];
        const double denom = y0 - 2.0 * y1 + y2;
        if (std::abs(denom) > 1e-12)
          so.td += 0.5 * (y0 - y2) / denom * step;
        so.sharpness = -denom / (step * step);
      }
      so.ok = true;
      return so;
    };

    // ---- PRIMARY: the VERBATIM legacy scan (see WEIGHT SCOPE in the header) ----
    ScanOut full;
    {
      // Camera-side mean once
      double mu_c = 0.0;
      for (const auto &c : cam)
        mu_c += c.rate;
      mu_c /= (double)cam.size();

      std::vector<double> score(n_lags, -2.0);
      for (int li = 0; li < n_lags; ++li) {
        const double tau = -search + li * step;
        double sc = 0.0, si = 0.0, si2 = 0.0, scc = 0.0, cross = 0.0;
        int cnt = 0;
        for (const auto &c : cam) {
          double v;
          if (!imu_rate_at(c.t_mid + tau, v))
            continue;
          const double dc = c.rate - mu_c;
          sc += dc;
          scc += dc * dc;
          si += v;
          si2 += v * v;
          cross += dc * v;
          ++cnt;
        }
        if (cnt < 8)
          continue;
        // rho = cov(c,i)/sqrt(var(c) var(i)) with the camera mean fixed globally
        const double mu_i = si / cnt;
        const double var_i = si2 / cnt - mu_i * mu_i;
        const double cov = cross / cnt - (sc / cnt) * mu_i;
        const double var_c = scc / cnt - (sc / cnt) * (sc / cnt);
        if (var_i <= 1e-12 || var_c <= 1e-12)
          continue;
        score[li] = cov / std::sqrt(var_c * var_i);
      }

      // Export before early returns so rejected scans retain their diagnostics.
      if (curve != nullptr) {
        curve->search = search;
        curve->step = step;
        curve->corr.assign(score.begin(), score.end());
      }

      int imax = 0;
      for (int li = 1; li < n_lags; ++li)
        if (score[li] > score[imax])
          imax = li;
      if (score[imax] <= -2.0)
        return out;
      full.imax = imax;
      full.peak = score[imax];
      full.td = -search + imax * step;
      full.at_bound = (imax == 0 || imax == n_lags - 1);
      if (imax > 0 && imax < n_lags - 1 && score[imax - 1] > -2.0 && score[imax + 1] > -2.0) {
        const double y0 = score[imax - 1], y1 = score[imax], y2 = score[imax + 1];
        const double denom = y0 - 2.0 * y1 + y2;
        if (std::abs(denom) > 1e-12)
          full.td += 0.5 * (y0 - y2) / denom * step;
        full.sharpness = -denom / (step * step);
      }
      full.ok = true;
    }
    out.peak_corr = full.peak;
    out.td = full.td;
    out.at_bound = full.at_bound;
    out.peak_sharpness = full.sharpness;
    out.ok = true;
    if (out.at_bound)
      return out; // a bound-pinned peak gets no certificate: widen the search instead

    std::vector<int> all((int)cam.size());
    for (int k = 0; k < (int)cam.size(); ++k)
      all[(size_t)k] = k;

    // ---- (a) one Hampel trim at the found lag, then a full rescan of the kept set ----
    // The regression c ~ a + b*i(t+tau*) at the peak lag exposes fat-tail pairs (blur slips,
    // model-selection misfires) as residual outliers; median +/- 3*1.4826*MAD keeps the family.
    // The trim can only earn authority through the caller's retention + argmax-consistency
    // conditions -- a peak bought by discarding a third of the data must not pass.
    {
      const double tau = -search + full.imax * step;
      std::vector<int> covered;
      std::vector<double> vi;
      covered.reserve(all.size());
      vi.reserve(all.size());
      double sc = 0.0, si = 0.0, si2 = 0.0, cross = 0.0, W = 0.0;
      for (int k : all) {
        const CamRateSample &c = cam[(size_t)k];
        double v;
        if (!imu_rate_at(c.t_mid + tau, v))
          continue;
        covered.push_back(k);
        vi.push_back(v);
        const double w = c.weight;
        sc += w * c.rate;
        si += w * v;
        si2 += w * v * v;
        cross += w * c.rate * v;
        W += w;
      }
      if ((int)covered.size() >= 16 && W > 0.0) {
        const double m_c = sc / W, m_i = si / W;
        const double var_i = si2 / W - m_i * m_i;
        if (var_i > 1e-12) {
          const double b = (cross / W - m_c * m_i) / var_i;
          const double a = m_c - b * m_i;
          std::vector<double> r(covered.size());
          for (size_t j = 0; j < covered.size(); ++j)
            r[j] = cam[(size_t)covered[j]].rate - (a + b * vi[j]);
          std::vector<double> tmp = r;
          std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
          const double med = tmp[tmp.size() / 2];
          for (size_t j = 0; j < tmp.size(); ++j)
            tmp[j] = std::abs(r[j] - med);
          std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
          const double mad = tmp[tmp.size() / 2];
          std::vector<int> kept;
          kept.reserve(covered.size());
          double w_kept = 0.0;
          if (mad > 1e-12) {
            const double gate = 3.0 * 1.4826 * mad;
            for (size_t j = 0; j < covered.size(); ++j)
              if (std::abs(r[j] - med) <= gate) {
                kept.push_back(covered[j]);
                w_kept += cam[(size_t)covered[j]].weight;
              }
          } else {
            kept = covered;
            w_kept = W;
          }
          out.trim_retention = (W > 0.0) ? (w_kept / W) : 1.0;
          const ScanOut trimmed = scan(kept, /*use_weights=*/true);
          if (trimmed.ok) {
            out.peak_trimmed = trimmed.peak;
            out.trim_consistent = !trimmed.at_bound && std::abs(trimmed.imax - full.imax) <= 2;
          }
        }
      }
    }

    // ---- (b) split-half reproducibility (even/odd, so both halves span the session) ----
    {
      std::vector<int> ha, hb;
      ha.reserve(all.size() / 2 + 1);
      hb.reserve(all.size() / 2 + 1);
      for (size_t j = 0; j < all.size(); ++j)
        ((j & 1u) ? hb : ha).push_back(all[j]);
      const ScanOut sa = scan(ha, /*use_weights=*/true), sb = scan(hb, /*use_weights=*/true);
      if (sa.ok && sb.ok && !sa.at_bound && !sb.at_bound) {
        out.td_split_delta = std::abs(sa.td - sb.td);
        out.split_min_peak = std::min(sa.peak, sb.peak);
      }
    }
    // Separate time intervals expose motion-dependent bias that interleaved
    // samples can share. This diagnostic does not change the primary estimate.
    {
      const double cut = 0.5 * (cam.front().t_mid + cam.back().t_mid);
      std::vector<int> early, late;
      for (int k : all)
        (cam[(size_t)k].t_mid < cut ? early : late).push_back(k);
      const ScanOut a = scan(early, false), b = scan(late, false);
      if (a.ok && b.ok && !a.at_bound && !b.at_bound) {
        out.temporal_split_delta = std::abs(a.td - b.td);
        out.temporal_min_peak = std::min(a.peak, b.peak);
      }
    }
    return out;
  }
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_TIME_OFFSET_INIT_H
