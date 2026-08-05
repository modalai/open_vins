/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: seed the IMU intrinsics from an OpenVINS/kalibr IMU chain.
 * ------------------------------------------------------------------------
 * The two shipped gauges describe the SAME physics in DIFFERENT body frames,
 * and converting between them is a real conjugation -- not a repacking:
 *
 *   KALIBR gauge: IMU frame == ACCEL frame  (R_ACCtoIMU = I structural)
 *       w_hat = R_GYROtoIMU * Dw_k * (w_m - bg)      Dw_k, Da_k LOWER-tri
 *       a_hat =               Da_k * (a_m - ba)
 *
 *   RPNG gauge (== ov_zcalib's imu2): IMU frame == GYRO frame (R_GtoI = I)
 *       w_hat =         Dw_r * (w_m - bg)            Dw_r, Da_r UPPER-tri
 *       a_hat = R_AtoI * Da_r * (a_m - ba)
 *
 * The two IMU frames differ by the physical gyro-vs-accel die misalignment R
 * (~0.8 deg on ICM-class parts), with w_hat^K = R w_hat^R and a_hat^K = R a_hat^R.
 * Matching the gyro rows gives  R_GYROtoIMU * Dw_k = R * Dw_r  and matching the
 * accel rows gives  R_ACCtoIMU * Da_k = R * R_AtoI * Da_r. Each is an
 * "orthogonal x upper-triangular" factorization of a KNOWN matrix -- i.e. a QR:
 *
 *   M := R_GYROtoIMU * Dw_chain          QR:  M = Q_w U_w   =>  R = Q_w,  Dw_r = U_w
 *   N := Q_w^T * R_ACCtoIMU * Da_chain   QR:  N = Q_a U_a   =>  R_AtoI = Q_a, Da_r = U_a
 *
 * Scale factors are positive by construction, so the QR is made sign-canonical
 * (positive diagonal). An rpng-gauge chain is the identity case of this map
 * (R_GYROtoIMU = I and Dw already upper-tri => Q_w = I), so ONE path serves both
 * models and a same-gauge chain round-trips exactly.
 *
 * WHY SEED AT ALL (operator directive, 2026-07-12): the extrinsics and td are
 * earned BLIND, but the IMU and camera intrinsics are per-unit factory data that
 * the rig already ships. Seeding them (a) starts the accel/gyro chain at the
 * truth instead of identity, so the solve converges far sooner, and (b) makes an
 * unmoved block ship the FACTORY value rather than an uncalibrated identity --
 * the failure that would otherwise silently overwrite a good factory Dw.
 *
 * Tg (g-sensitivity) IS seeded, and the gauge change conjugates it -- but only on
 * one side. In both models the bracket is (w_m - b_g - Tg * a_hat): w_m and b_g are
 * RAW GYRO AXES quantities, identical in either gauge, so the only thing that moves
 * is a_hat's frame. With a_hat^K = R a_hat^R (R = Q_w, the same rotation the QR
 * above already produces),
 *
 *   Tg_k * a_hat^K = Tg_k * Q_w * a_hat^R  ==  Tg_r * a_hat^R   =>   Tg_r = Tg_chain * Q_w
 *
 * An rpng-gauge chain has Q_w = I, so Tg round-trips exactly, like Dw and Da: one
 * path still serves both models.
 *
 * It is seeded FIXED (calib_tg stays false). Estimating it is a different question
 * and remains closed -- Factor_ImuAci3 rejects n_pi != 15. What was NOT closed until
 * now is that zeroing Tg threw away a term the chain actually measured: at ~4e-4
 * (rad/s)/(m/s^2) it is 0.23 deg/s of spurious rate at 1 g, and the session keeps
 * SLOW-TILT windows (the fast ones are seed-gate rejected), where that is a ~1%
 * relative gyro error -- landing squarely on dw, the one block both cameras share.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_IMU_CHAIN_CONVERT_H
#define OV_ZCALIB_IMU_CHAIN_CONVERT_H

#include <Eigen/Dense>

#include "ImuIntrinsicModel.h"
#include "utils/quat_ops.h"

namespace ov_zcalib {

/// Sign-canonical QR: A = Q * U with U upper-triangular and diag(U) > 0.
inline void qr_positive_diag(const Eigen::Matrix3d &A, Eigen::Matrix3d &Q, Eigen::Matrix3d &U) {
  Eigen::HouseholderQR<Eigen::Matrix3d> qr(A);
  Q = qr.householderQ();
  U = qr.matrixQR().triangularView<Eigen::Upper>();
  for (int k = 0; k < 3; ++k) {
    if (U(k, k) < 0.0) { // flip: Q.col(k) * U.row(k) is invariant under a joint sign flip
      U.row(k) *= -1.0;
      Q.col(k) *= -1.0;
    }
  }
}

/// Convert an OpenVINS IMU chain (EITHER gauge) into ov_zcalib's imu2 parameters.
/// @param Dw_chain      full 3x3 Dw = Tw^-1 from the chain
/// @param Da_chain      full 3x3 Da = Ta^-1 from the chain
/// @param R_GYROtoIMU   chain's gyro->IMU rotation (identity in the rpng gauge)
/// @param R_ACCtoIMU    chain's accel->IMU rotation (identity in the kalibr gauge)
/// @param Tg_chain      full 3x3 g-sensitivity from the chain (Zero if the chain has none)
/// @param[out] imu      dw/da (upper-tri packed) + q_AtoI + Tg, ready to seed SharedCalib
inline void imu_chain_to_calib(const Eigen::Matrix3d &Dw_chain, const Eigen::Matrix3d &Da_chain,
                               const Eigen::Matrix3d &R_GYROtoIMU, const Eigen::Matrix3d &R_ACCtoIMU,
                               const Eigen::Matrix3d &Tg_chain, ImuIntrinsicModel &imu) {
  Eigen::Matrix3d Qw, Uw, Qa, Ua;
  qr_positive_diag(R_GYROtoIMU * Dw_chain, Qw, Uw);            // gyro rows -> the frame rotation + upper-tri Dw
  qr_positive_diag(Qw.transpose() * R_ACCtoIMU * Da_chain, Qa, Ua); // accel rows, in the GYRO frame
  imu.dw << Uw(0, 0), Uw(0, 1), Uw(1, 1), Uw(0, 2), Uw(1, 2), Uw(2, 2); // ut() packing [d11,d12,d22,d13,d23,d33]
  imu.da << Ua(0, 0), Ua(0, 1), Ua(1, 1), Ua(0, 2), Ua(1, 2), Ua(2, 2);
  imu.q_AtoI = ov_core::rot_2_quat(Qa);
  imu.Tg = Tg_chain * Qw; // a_hat's frame is the only thing the gauge change moves (see header)
  // The port only SEEDS the value. Whether the session may ESTIMATE tg is session policy
  // (estimate_tg -> SessionConfig::free_tg), decided by the runner at construction and unlocked
  // through the A1b excitation gate -- never by the chain port.
  imu.calib_tg = false;
}

/// Inverse map: ov_zcalib's imu2 parameters -> the corrected-signal matrices a
/// chain consumer needs. Returns the GYRO-frame (rpng) quantities directly --
/// the caller converts to its own gauge if it is kalibr-modelled. Provided so a
/// writeback (and the round-trip test) can close the loop without re-deriving.
inline void calib_to_imu_chain(const ImuIntrinsicModel &imu, Eigen::Matrix3d &Dw_rpng, Eigen::Matrix3d &Da_rpng,
                               Eigen::Matrix3d &R_ACCtoIMU_rpng) {
  Dw_rpng = ImuIntrinsicModel::ut(imu.dw);
  Da_rpng = ImuIntrinsicModel::ut(imu.da);
  R_ACCtoIMU_rpng = ov_core::quat_2_Rot(imu.q_AtoI);
}

} // namespace ov_zcalib

#endif // OV_ZCALIB_IMU_CHAIN_CONVERT_H
