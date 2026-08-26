/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_UPDATER_OPTIONS_H
#define OV_MSCKF_UPDATER_OPTIONS_H

#include <cmath>

#include "utils/print.h"

namespace ov_msckf {

/**
 * @brief Struct which stores general updater options
 */
struct UpdaterOptions {

  /// What chi-squared multipler we should apply
  double chi2_multipler = 5;

  /// Noise sigma for our raw pixel measurements
  double sigma_pix = 1;

  /// Covariance for our raw pixel measurements
  double sigma_pix_sq = 1;

  /// Noise sigma for STEREO (multi-camera) feature observations. Cross-camera
  /// ZNCC matches are noisier than same-camera KLT temporal tracks, so a feature
  /// observed in more than one camera is modeled with this (larger) noise. If
  /// left <= 0 it falls back to sigma_pix (no stereo-specific weighting).
  double sigma_pix_stereo = -1;

  /// Covariance for stereo pixel measurements (sigma_pix_stereo^2). Resolved at
  /// config-parse time; equals sigma_pix_sq when no stereo value is configured.
  double sigma_pix_sq_stereo = 1;

  /// Nice print function of what parameters we have loaded
  void print() {
    PRINT_DEBUG("    - chi2_multipler: %.1f\n", chi2_multipler);
    PRINT_DEBUG("    - sigma_pix: %.2f\n", sigma_pix);
    PRINT_DEBUG("    - sigma_pix_stereo: %.2f\n", std::sqrt(sigma_pix_sq_stereo));
  }
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_OPTIONS_H