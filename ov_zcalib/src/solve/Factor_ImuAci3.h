/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * ov_zcalib: ACI3 preintegration factor (Factor_ImuCPIv1 + IMU-intrinsic blocks).
 * Residual structure and the 11 canonical blocks are IDENTICAL to the proven
 * Factor_ImuCPIv1; three calibration blocks are appended and corrected to first
 * order through the ACI3 columns (after Yang et al., MVIS, IJRR 2024):
 *
 *   [0..10] as Factor_ImuCPIv1 (q1,bg1,v1,ba1,p1, q2,bg2,v2,ba2,p2, gravity-S2)
 *   [11] dw6   [12] da6   [13] q_AtoI (4, JPL local)   [14] tg9 (only when the model calibrates Tg)
 *
 * Subset selection (kalibr-style) is done by holding blocks constant in the
 * Problem, not by changing the factor — EXCEPT Tg, whose columns exist only when
 * model_lin.calib_tg is set (n_pi 15 -> 24). That asymmetry is deliberate: the
 * Tg-off arithmetic must stay BYTE-IDENTICAL to the validated corpus, and under
 * -ffast-math even multiplying through nine explicitly-zero columns reassociates
 * the reductions (the measured 1-ulp lesson of 2026-07-13). The legacy 15-wide
 * statements below are therefore kept verbatim in their own branch; the 24-wide
 * path is separate code that only runs when Tg is being estimated. tg is packed
 * in Matrix3d STORAGE order (column-major) to match mixing()'s enumeration.
 * All Jacobians are pinned by the factor-level FD oracle in test_aci3_fd (the
 * J_r-flavor episode is the standing justification).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_FACTOR_IMU_ACI3_H
#define OV_ZCALIB_FACTOR_IMU_ACI3_H

#include <Eigen/Dense>

#include "ceres_free/CostFunction.h"

#include "../cpi/AciCalibPreint.h"

namespace ov_zcalib {

class Factor_ImuAci3 : public ov_init::zbft_sfm::CostFunction {
public:
  // Preintegrated measurement (at the linearization point), CPIv1 conventions
  AciPreintResult m;
  // Linearization points
  Eigen::Vector3d bg_lin, ba_lin;
  Eigen::Matrix<double, 6, 1> dw_lin, da_lin;
  Eigen::Vector4d qA_lin;
  Eigen::Matrix3d tg_lin = Eigen::Matrix3d::Zero(); ///< only consulted when tg_on
  bool tg_on = false;                               ///< preint carried 24 intrinsic columns (dw6|da6|thA3|tg9)
  // sqrt information of m.P15
  Eigen::Matrix<double, 15, 15> sqrtI;
  Eigen::Matrix<double, 15, 3> sqrtI_grav_fold;

  /**
   * @brief Construct from a preintegration result; the model carries the intrinsic
   *        linearization point (dw/da/thA groups enabled: n_pi == 15, or 24 with calib_tg).
   */
  Factor_ImuAci3(const AciPreintResult &meas, const ImuIntrinsicModel &model_lin, const Eigen::Vector3d &bg_lin_,
                 const Eigen::Vector3d &ba_lin_);

  /**
   * @brief Cache-fed overload (P3): identical factor, but the whitener comes
   *        precomputed so construction skips the double 15x15 factorization
   *        entirely — the dominant, previously untimed cost of rebuilding
   *        factors every window evaluation. The caller obtains sqrtI_ext /
   *        fold_ext ONLY by copying them out of a legacy-constructed factor
   *        (WindowBA cache fill): a separate whitener implementation compiled
   *        in a different codegen context measurably drifts 1 ulp under
   *        -ffast-math (see the parity contract in the legacy ctor).
   */
  Factor_ImuAci3(const AciPreintResult &meas, const ImuIntrinsicModel &model_lin, const Eigen::Vector3d &bg_lin_,
                 const Eigen::Vector3d &ba_lin_, const Eigen::Matrix<double, 15, 15> &sqrtI_ext,
                 const Eigen::Matrix<double, 15, 3> &fold_ext);

  virtual ~Factor_ImuAci3() {}

  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

private:
  /// The 24-column (Tg) path; Evaluate dispatches here FIRST so the legacy body keeps its exact
  /// emitted code under -ffast-math (in-body branching alone measurably reflowed the vectorizer).
  bool evaluate_tg_(double const *const *parameters, double *residuals, double **jacobians) const;
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_FACTOR_IMU_ACI3_H
