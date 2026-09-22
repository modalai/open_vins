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

#include "UpdaterZeroVelocity.h"

#include <cstdint>
#include <cstring>

#include "UpdaterHelper.h"
#include "ZeroVelocityModel.h"

#include "feat/FeatureDatabase.h"
#include "feat/FeatureHelper.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/colors.h"
#include "utils/innovation.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include "utils/ChronoProf.h"
#include "utils/chi_square/chi_squared_quantile_table_0_95.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

namespace {
bool finite_time(double value) {
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value), "64-bit IEEE double required");
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

template <class Derived> bool finite_sampled_coefficients(const Eigen::MatrixBase<Derived> &value) {
  for (Eigen::Index c = 0; c < value.cols(); ++c)
    for (Eigen::Index r = 0; r < value.rows(); ++r)
      if (!finite_time(value(r, c))) return false;
  return true;
}

const State::SampledImuSlot *sampled_owner(const std::shared_ptr<State> &state, const State::SampledImuRecord &record) {
  for (const auto &slot : state->sampled_imu_slots()) {
    if (slot.active && slot.record.stream_episode == record.stream_episode && slot.record.sequence == record.sequence &&
        initializer_time_bits(slot.record.timestamp) == initializer_time_bits(record.timestamp) &&
        std::memcmp(slot.record.measured.data(), record.measured.data(), 6 * sizeof(double)) == 0 &&
        std::memcmp(slot.record.prior.data(), record.prior.data(), 36 * sizeof(double)) == 0) return &slot;
  }
  return nullptr;
}
} // namespace

UpdaterZeroVelocity::UpdaterZeroVelocity(UpdaterOptions &options, NoiseManager &noises, std::shared_ptr<ov_core::FeatureDatabase> db,
                                         std::shared_ptr<Propagator> prop, double gravity_mag, double zupt_max_velocity,
                                         double zupt_noise_multiplier, double zupt_max_disparity, double prop_window)
    : _options(options), _noises(noises), _db(db), _prop(prop), _zupt_max_velocity(zupt_max_velocity),
      _zupt_noise_multiplier(zupt_noise_multiplier), _zupt_max_disparity(zupt_max_disparity), _prop_window(prop_window) {

  // Gravity
  _gravity << 0.0, 0.0, gravity_mag;

  // Save our raw pixel noise squared
  _noises.sigma_w_2 = std::pow(_noises.sigma_w, 2);
  _noises.sigma_a_2 = std::pow(_noises.sigma_a, 2);
  _noises.sigma_wb_2 = std::pow(_noises.sigma_wb, 2);
  _noises.sigma_ab_2 = std::pow(_noises.sigma_ab, 2);

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  // Chi-squared 0.95 gating thresholds come from the baked table
  // (utils/chi_square/, bit-identical to the boost::math values this used to compute here)
}

bool UpdaterZeroVelocity::linearize_sampled_at_knot(
    std::shared_ptr<State> state, const State::SampledImuRecord &record,
    const Eigen::Matrix<double, 6, 6> &independent_stationarity_covariance, SampledStationaryFactor &out) const {
  if (!state || !state->has_sampled_imu_boundary() || !state->_imu_endpoint_valid || state->_options.do_fej ||
      state->_options.do_calib_imu_intrinsics || state->_options.do_calib_imu_g_sensitivity || state->imu_intrinsic_size() != 0 ||
      (state->_options.integration_method != StateOptions::ANALYTICAL && state->_options.integration_method != StateOptions::DISCRETE) ||
      initializer_time_bits(state->imu_endpoint()) != initializer_time_bits(record.timestamp) ||
      !StateHelper::valid_sampled_imu_record(record) ||
      !StateHelper::valid_initial_covariance(independent_stationarity_covariance, true)) return false;
  const auto *owner = sampled_owner(state, record);
  if (!owner || !owner->noise || !finite_sampled_coefficients(owner->noise->value()) ||
      !finite_sampled_coefficients(state->_imu->value()) || std::abs(state->_imu->quat().squaredNorm() - 1.) > 1e-9) return false;
  const Eigen::Matrix3d A = state->_calib_imu_ACCtoIMU->Rot() * State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  const Eigen::Matrix3d W = state->_calib_imu_GYROtoIMU->Rot() * State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  const Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());
  if (!finite_sampled_coefficients(W)) return false;
  const Eigen::FullPivLU<Eigen::Matrix3d> gyro_factor(W);
  if (!gyro_factor.isInvertible() || !(W.determinant() > 0.) || !(gyro_factor.rcond() > 1e-12)) return false;
  ZeroVelocityModel stationary_model;
  if (!stationary_model.set_calibration(A, Tg)) return false;
  const Eigen::Matrix<double, 6, 1> raw = record.measured - owner->noise->value();
  const Eigen::Vector3d force = state->_imu->Rot() * _gravity;
  ZeroVelocityModel::Jacobian H;
  SampledStationaryFactor staged;
  if (!stationary_model.linearize(raw.tail<3>(), raw.head<3>(), state->_imu->bias_g(), state->_imu->bias_a(),
                                  force, force, 1., 1., H, staged.residual)) return false;
  staged.H.leftCols<9>() = H;
  staged.H.rightCols<6>() = -Eigen::Matrix<double, 6, 6>::Identity();
  staged.independent_R = independent_stationarity_covariance;
  out = staged;
  return true;
}

bool UpdaterZeroVelocity::try_update_sampled_at_knot(
    std::shared_ptr<State> state, const State::SampledImuRecord &record,
    const Eigen::Matrix<double, 6, 6> &independent_stationarity_covariance, SampledStationaryFactor *out) {
  if (!_prop || (sampled_stream_episode && (sampled_stream_episode != record.stream_episode ||
      record.sequence <= sampled_last_sequence)) || !finite_time(_options.chi2_multipler) ||
      !(_options.chi2_multipler > 0.) || !finite_time(_zupt_max_velocity) || _zupt_max_velocity < 0.) return false;
  SampledStationaryFactor factor;
  if (!linearize_sampled_at_knot(state, record, independent_stationarity_covariance, factor)) return false;
  const auto *owner = sampled_owner(state, record);
  const std::vector<std::shared_ptr<Type>> order{state->_imu->q(), state->_imu->bg(), state->_imu->ba(), owner->noise};
  const Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, order);
  Eigen::Matrix<double, 6, 6> innovation = factor.H * P * factor.H.transpose() + factor.independent_R;
  if (!finite_sampled_coefficients(innovation)) return false;
  const Eigen::LLT<Eigen::Matrix<double, 6, 6>> solve(innovation.selfadjointView<Eigen::Upper>());
  if (solve.info() != Eigen::Success) return false;
  const Eigen::Matrix<double, 6, 1> normalized = solve.solve(factor.residual);
  const double chi2 = factor.residual.dot(normalized);
  const double threshold = _options.chi2_multipler * chi_squared_quantile_0_95(6);
  if (!finite_sampled_coefficients(normalized) || !finite_time(chi2) || chi2 < 0. ||
      !finite_time(threshold) || chi2 > threshold || state->_imu->vel().norm() > _zupt_max_velocity) return false;
  // Full raw sample uncertainty/correlation enters through H and P. Bias
  // evolution already occurred in sampled propagation; neither is added to R.
  if (!StateHelper::EKFUpdate(state, order, factor.H, factor.residual, factor.independent_R)) return false;
  sampled_stream_episode = record.stream_episode;
  sampled_last_sequence = record.sequence;
  sampled_last_timestamp_bits = initializer_time_bits(record.timestamp);
  _prop->invalidate_cache();
  if (out) *out = factor;
  return true;
}

bool UpdaterZeroVelocity::try_update(std::shared_ptr<State> state, double timestamp) {
  if (!state || state->has_sampled_imu_boundary()) return false;
  if (state->_timestamp == timestamp)
    return false;
  const double offset = state->cam_imu_dt_ref();
  const bool accepted = try_update_at_imu(state, timestamp + offset, timestamp);
  if (accepted)
    last_prop_time_offset = offset; // Preserve the legacy descriptor arithmetic.
  return accepted;
}

bool UpdaterZeroVelocity::try_update_at_imu(std::shared_ptr<State> state, double target_imu, double reference_timestamp,
                                           const RawCameraKeys &raw_keys) {
  // This legacy window model treats raw sensor noise as independent. A
  // prepared sampled state requires the separate common-noise factor.
  if (!state || state->has_sampled_imu_boundary()) return false;
  const double timestamp = reference_timestamp;
  const double time0 = state->imu_endpoint();
  const double time1 = target_imu;
  const bool physical = state->uses_physical_clones();
  if (!finite_time(time0) || !finite_time(time1) || !finite_time(timestamp) ||
      !finite_time(time1 - timestamp) || !(time1 > time0) || (physical && raw_keys.empty()))
    return false;
  if (physical) {
    for (size_t i = 0; i < raw_keys.size(); ++i) {
      if (raw_keys[i].first >= static_cast<size_t>(state->_options.num_cameras) || !finite_time(raw_keys[i].second))
        return false;
      // At most one current frame per camera in a physical event group.
      for (size_t j = 0; j < i; ++j)
        if (raw_keys[j].first == raw_keys[i].first)
          return false;
    }
  }

  // Select bounding inertial measurements
  std::vector<ov_core::ImuData> imu_recent = Propagator::select_imu_readings(imu_data, time0, time1);

  // Check that we have at least one measurement to propagate with
  if (imu_recent.size() < 2) {
    PRINT_WARNING(RED "[ZUPT]: There are no IMU data to check for zero velocity with!!\n" RESET);
    return false;
  }

  if (physical && camera_history.size() < static_cast<size_t>(state->_options.num_cameras))
    camera_history.resize(static_cast<size_t>(state->_options.num_cameras));

  // If we should integrate the acceleration and say the velocity should be zero
  // Also if we should still inflate the bias based on their random walk noises
  bool integrated_accel_constraint = false; // untested
  bool model_time_varying_bias = true;
  bool override_with_disparity_check = true;

  // Order of our Jacobian
  std::vector<std::shared_ptr<Type>> Hx_order;
  Hx_order.push_back(state->_imu->q());
  Hx_order.push_back(state->_imu->bg());
  Hx_order.push_back(state->_imu->ba());
  if (integrated_accel_constraint) {
    Hx_order.push_back(state->_imu->v());
  }

  // Large final matrices used for update (we will compress these)
  int h_size = (integrated_accel_constraint) ? 12 : 9;
  int m_size = 6 * ((int)imu_recent.size() - 1);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m_size, h_size);
  Eigen::VectorXd res = Eigen::VectorXd::Zero(m_size);

  // IMU intrinsic calibration estimates (static)
  Eigen::Matrix3d Dw = State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  Eigen::Matrix3d Da = State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());
  const Eigen::Matrix3d A = state->_calib_imu_ACCtoIMU->Rot() * Da;
  // Preserve the historical arithmetic path for the default identity model,
  // including its Eigen/FMA evaluation order through compression and update.
  const bool identity_model = (Dw.array() == Eigen::Matrix3d::Identity().array()).all() &&
                              (Da.array() == Eigen::Matrix3d::Identity().array()).all() &&
                              (state->_calib_imu_ACCtoIMU->Rot().array() == Eigen::Matrix3d::Identity().array()).all() &&
                              (state->_calib_imu_GYROtoIMU->Rot().array() == Eigen::Matrix3d::Identity().array()).all() &&
                              (Tg.array() == 0.0).all();
  ZeroVelocityModel stationary_model;
  if (!integrated_accel_constraint && !stationary_model.set_calibration(A, Tg)) {
    PRINT_WARNING(YELLOW "[ZUPT]: invalid fixed accelerometer/Tg model\n" RESET);
    last_zupt_state_timestamp = 0.0;
    last_zupt_count = 0;
    return false;
  }
  const Eigen::Vector3d force = state->_imu->Rot() * _gravity;
  const Eigen::Matrix3d R_GtoI_jacob = state->_options.do_fej ? state->_imu->Rot_fej() : state->_imu->Rot();
  const Eigen::Vector3d force_jacobian = R_GtoI_jacob * _gravity;

  // Loop through all our IMU and construct the residual and Jacobian
  // TODO: should add jacobians here in respect to IMU intrinsics!!
  // State order is: [q_GtoI, bg, ba, v_IinG]
  // Measurement order is: [w_true = 0, a_true = 0 or v_k+1 = 0]
  // w_true = w_m - bw - nw
  // a_true = a_m - ba - R*g - na
  // v_true = v_k - g*dt + R^T*(a_m - ba - na)*dt
  double dt_summed = 0;
  for (size_t i = 0; i < imu_recent.size() - 1; i++) {

    // Precomputed values
    double dt = imu_recent.at(i + 1).timestamp - imu_recent.at(i).timestamp;

    // Measurement noise (convert from continuous to discrete)
    // NOTE: The dt time might be different if we have "cut" any imu measurements
    // NOTE: We are performing "whittening" thus, we will decompose R_meas^-1 = L*L^t
    // NOTE: This is then multiplied to the residual and Jacobian (equivalent to just updating with R_meas)
    // NOTE: See Maybeck Stochastic Models, Estimation, and Control Vol. 1 Equations (7-21a)-(7-21c)
    double w_omega = std::sqrt(dt) / _noises.sigma_w;
    double w_accel = std::sqrt(dt) / _noises.sigma_a;
    double w_accel_v = 1.0 / (std::sqrt(dt) * _noises.sigma_a);

    if (!integrated_accel_constraint && !identity_model) {
      // Use raw stationary constraints and raw white-noise densities. With
      // a_true=R*g, wm-bg-Tg*a_true and am-ba-A^-1*a_true have independent
      // noise. Whitening corrected means with scalar raw sigmas loses both
      // the intrinsic mixing and the gyro/accel correlation induced by Tg.
      // Gyro scale/rotation cancels in the exact zero-angular-rate constraint.
      ZeroVelocityModel::Jacobian H_i;
      ZeroVelocityModel::Residual r_i;
      if (!stationary_model.linearize(imu_recent.at(i).am, imu_recent.at(i).wm, state->_imu->bias_g(), state->_imu->bias_a(),
                                      force, force_jacobian, w_omega, w_accel, H_i, r_i)) {
        last_zupt_state_timestamp = 0.0;
        last_zupt_count = 0;
        return false;
      }
      H.block<6,9>(6*i,0) = H_i;
      res.segment<6>(6*i) = r_i;
      dt_summed += dt;
      continue;
    }

    // Original identity-model arithmetic; the integrated-velocity alternative
    // also remains here, disabled and untested.
    Eigen::Vector3d a_hat = state->_calib_imu_ACCtoIMU->Rot() * Da * (imu_recent.at(i).am - state->_imu->bias_a());
    Eigen::Vector3d w_hat = state->_calib_imu_GYROtoIMU->Rot() * Dw * (imu_recent.at(i).wm - state->_imu->bias_g() - Tg * a_hat);

    // Measurement residual (true value is zero)
    res.block(6 * i + 0, 0, 3, 1) = -w_omega * w_hat;
    if (!integrated_accel_constraint) {
      res.block(6 * i + 3, 0, 3, 1) = -w_accel * (a_hat - state->_imu->Rot() * _gravity);
    } else {
      res.block(6 * i + 3, 0, 3, 1) = -w_accel_v * (state->_imu->vel() - _gravity * dt + state->_imu->Rot().transpose() * a_hat * dt);
    }

    // Measurement Jacobian
    H.block(6 * i + 0, 3, 3, 3) = -w_omega * Eigen::Matrix3d::Identity();
    if (!integrated_accel_constraint) {
      H.block(6 * i + 3, 0, 3, 3) = -w_accel * skew_x(R_GtoI_jacob * _gravity);
      H.block(6 * i + 3, 6, 3, 3) = -w_accel * Eigen::Matrix3d::Identity();
    } else {
      H.block(6 * i + 3, 0, 3, 3) = -w_accel_v * R_GtoI_jacob.transpose() * skew_x(a_hat) * dt;
      H.block(6 * i + 3, 6, 3, 3) = -w_accel_v * R_GtoI_jacob.transpose() * dt;
      H.block(6 * i + 3, 9, 3, 3) = w_accel_v * Eigen::Matrix3d::Identity();
    }
    dt_summed += dt;
  }

  // QR compression preserves the state update, but the orthogonal residual
  // still carries evidence against stationarity. Retain its energy for the
  // full innovation test below (otherwise zero-mean motion can disappear).
  double discarded_residual_squared = 0.0;
  UpdaterHelper::measurement_compress_inplace(H, res, discarded_residual_squared);
  if (H.rows() < 1) {
    return false;
  }

  // Multiply our noise matrix by a fixed amount
  // We typically need to treat the IMU as being "worst" to detect / not become overconfident
  Eigen::MatrixXd R = _zupt_noise_multiplier * Eigen::MatrixXd::Identity(res.rows(), res.rows());

  // Next propagate the biases forward in time
  // NOTE: G*Qd*G^t = dt*Qd*dt = dt*(1/dt*Qc)*dt = dt*Qc
  Eigen::MatrixXd Q_bias = Eigen::MatrixXd::Identity(6, 6);
  Q_bias.block(0, 0, 3, 3) *= dt_summed * _noises.sigma_wb_2;
  Q_bias.block(3, 3, 3, 3) *= dt_summed * _noises.sigma_ab_2;

  // Chi2 distance check
  // NOTE: we also append the propagation we "would do before the update" if this was to be accepted (just the bias evolution)
  // NOTE: we don't propagate first since if we fail the chi2 then we just want to return and do normal logic
  Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
  if (model_time_varying_bias) {
    P_marg.block(3, 3, 6, 6) += Q_bias;
  }
  Eigen::MatrixXd S = H * P_marg * H.transpose() + R;
  double chi2;
  bool valid_innovation = innovation_chi2(S, res, chi2) &&
                          ov_core::numeric::finite(_zupt_noise_multiplier) && _zupt_noise_multiplier > 0.0 &&
                          ov_core::numeric::finite(discarded_residual_squared) && discarded_residual_squared >= 0.0;
  if (valid_innovation) {
    chi2 += discarded_residual_squared / _zupt_noise_multiplier;
    valid_innovation = ov_core::numeric::finite(chi2);
  }

  // This is an innovation test with a state prior, not a nuisance-fit test:
  // all original measurement rows contribute a degree of freedom, including
  // the rows eliminated by QR. Compression only reduces the update's cost.
  double chi2_check = ov_core::chi_squared_quantile_0_95(m_size);

  // Check if the image disparity
  bool disparity_passed = false;
  if (override_with_disparity_check) {

    // Get the disparity statistics from this image to the previous
    double time0_cam = state->_timestamp;
    double time1_cam = timestamp;
    int num_features = 0;
    double disp_avg = 0.0;
    double disp_var = 0.0;
    if (!physical) {
      FeatureHelper::compute_disparity(_db, time0_cam, time1_cam, disp_avg, disp_var, num_features);
    } else {
      // Pool the same per-observation pixel statistic by counts, not by camera.
      double mean = 0.0, centered_sum = 0.0;
      for (const auto &key : raw_keys) {
        const auto &history = camera_history[key.first];
        if (!history.has_previous)
          continue;
        double camera_mean = 0.0, camera_std = 0.0;
        int count = 0;
        FeatureHelper::compute_disparity(_db, history.previous_raw, key.second, camera_mean, camera_std, count,
                                        static_cast<int>(key.first));
        if (count == 0)
          continue;
        const int total = num_features + count;
        const double difference = camera_mean - mean;
        centered_sum += (count - 1) * camera_std * camera_std +
                        difference * difference * (static_cast<double>(num_features) * count / total);
        mean += difference * (static_cast<double>(count) / total);
        num_features = total;
      }
      disp_avg = num_features > 0 ? mean : -1.0;
      disp_var = num_features > 1 ? std::sqrt(centered_sum / (num_features - 1)) : 0.0;
    }

    // Check if this disparity is enough to be classified as moving
    disparity_passed = (disp_avg < _zupt_max_disparity && num_features > 20);
    if (disparity_passed) {
      PRINT_INFO(CYAN "[ZUPT]: passed disparity (%.3f < %.3f, %d features)\n" RESET, disp_avg, _zupt_max_disparity, (int)num_features);
    } else {
      PRINT_DEBUG(YELLOW "[ZUPT]: failed disparity (%.3f > %.3f, %d features)\n" RESET, disp_avg, _zupt_max_disparity, (int)num_features);
    }
  }

  // A covered, evaluated camera event becomes the previous raw observation
  // for that camera even when the IMU/velocity gate rejects stationarity.
  if (physical) {
    for (const auto &key : raw_keys) {
      camera_history[key.first].previous_raw = key.second;
      camera_history[key.first].has_previous = true;
    }
  }

  // Check if we are currently zero velocity
  // We need to pass the chi2 and not be above our velocity threshold
  const double chi2_limit = _options.chi2_multipler * chi2_check;
  const double velocity_norm = state->_imu->vel().norm();
  // A valid low-disparity observation may override the motion test, but cannot
  // make invalid inertial arithmetic safe to apply to the filter.
  if (!valid_innovation || !valid_innovation_limit(chi2_limit) || !ov_core::numeric::finite(velocity_norm) ||
      (!disparity_passed && (chi2 > chi2_limit || velocity_norm > _zupt_max_velocity))) {
    last_zupt_state_timestamp = 0.0;
    last_zupt_count = 0;
    if (physical)
      for (const auto &key : raw_keys)
        camera_history[key.first].accepted_count = 0;
    PRINT_DEBUG(YELLOW "[ZUPT]: rejected |v_IinG| = %.3f (chi2 %.3f > %.3f)\n" RESET, state->_imu->vel().norm(), chi2,
                _options.chi2_multipler * chi2_check);
    return false;
  }
  // The only production branch is the direct IMU constraint. Stage its
  // six-dimensional bias prior so a failed measurement update does not leave
  // an unconsumed random-walk increment behind. The alternative clone branch
  // was hardcoded unreachable and had no rollback contract.
  const std::vector<std::shared_ptr<Type>> bias_order = {state->_imu->bg(), state->_imu->ba()};
  Eigen::MatrixXd bias_prior;
  if (model_time_varying_bias) {
    bias_prior = StateHelper::get_marginal_covariance(state, bias_order);
    if (!StateHelper::EKFPropagation(state, bias_order, bias_order, Eigen::MatrixXd::Identity(6, 6), Q_bias)) return false;
  }
  if (!StateHelper::EKFUpdate(state, Hx_order, H, res, R)) {
    if (model_time_varying_bias)
      StateHelper::set_initial_covariance(state, bias_prior, bias_order);
    last_zupt_state_timestamp = 0.0;
    last_zupt_count = 0;
    if (physical)
      for (const auto &key : raw_keys) camera_history[key.first].accepted_count = 0;
    PRINT_WARNING(YELLOW "[ZUPT]: rejected invalid update; endpoint and measurements retained\n" RESET);
    return false;
  }
  PRINT_INFO(CYAN "[ZUPT]: accepted |v_IinG| = %.3f (chi2 %.3f < %.3f)\n" RESET, state->_imu->vel().norm(), chi2,
             _options.chi2_multipler * chi2_check);

  // Do our update, only do this update if we have previously detected
  // If we have succeeded, then we should remove the current timestamp feature tracks
  // This is because we will not clone at this timestep and instead do our zero velocity update
  // NOTE: We want to keep the tracks from the second time we have called the zv-upt since this won't have a clone
  // NOTE: All future times after the second call to this function will also *not* have a clone, so we can remove those
  if (physical) {
    for (const auto &key : raw_keys) {
      const auto &history = camera_history[key.first];
      if (history.accepted_count >= 2)
        _db->cleanup_measurements_exact_camera(key.first, history.accepted_raw);
    }
  } else if (last_zupt_count >= 2) {
    _db->cleanup_measurements_exact(last_zupt_state_timestamp);
  }

  // Finally return
  state->_timestamp = timestamp;
  state->_imu_endpoint = time1;
  state->_imu_endpoint_valid = true;
  last_prop_time_offset = time1 - timestamp;
  have_last_prop_time_offset = true;
  _prop->invalidate_cache();
  if (physical) {
    for (const auto &key : raw_keys) {
      auto &history = camera_history[key.first];
      history.accepted_raw = key.second;
      if (history.accepted_count < 2)
        ++history.accepted_count; // Only the 0/1/at-least-2 states are needed.
    }
  }
  last_zupt_state_timestamp = timestamp;
  last_zupt_count++;
  return true;
}

void UpdaterZeroVelocity::reset_for_new_state() {
  last_prop_time_offset = 0.0;
  have_last_prop_time_offset = false;
  last_zupt_state_timestamp = 0.0;
  last_zupt_count = 0;
  sampled_stream_episode = sampled_last_sequence = sampled_last_timestamp_bits = 0;
  for (auto &history : camera_history)
    history = CameraHistory();
  _prop->invalidate_cache();
}

UpdaterZeroVelocity::Snapshot UpdaterZeroVelocity::capture() const {
  Snapshot snapshot;
  snapshot.imu_data = imu_data;
  snapshot.camera_history = camera_history;
  snapshot.last_prop_time_offset = last_prop_time_offset;
  snapshot.have_last_prop_time_offset = have_last_prop_time_offset;
  snapshot.last_zupt_state_timestamp = last_zupt_state_timestamp;
  snapshot.last_zupt_count = last_zupt_count;
  snapshot.sampled_stream_episode = sampled_stream_episode;
  snapshot.sampled_last_sequence = sampled_last_sequence;
  snapshot.sampled_last_timestamp_bits = sampled_last_timestamp_bits;
  return snapshot;
}

void UpdaterZeroVelocity::restore(const Snapshot &snapshot) {
  imu_data = snapshot.imu_data;
  camera_history = snapshot.camera_history;
  last_prop_time_offset = snapshot.last_prop_time_offset;
  have_last_prop_time_offset = snapshot.have_last_prop_time_offset;
  last_zupt_state_timestamp = snapshot.last_zupt_state_timestamp;
  last_zupt_count = snapshot.last_zupt_count;
  sampled_stream_episode = snapshot.sampled_stream_episode;
  sampled_last_sequence = snapshot.sampled_last_sequence;
  sampled_last_timestamp_bits = snapshot.sampled_last_timestamp_bits;
  _prop->invalidate_cache();
}

void UpdaterZeroVelocity::feed_imu_batch(const std::vector<ov_core::ImuData>& messages, double oldest_time) {
    if (messages.empty()) return;

    // Insert all measurements at once
    imu_data.reserve(imu_data.size() + messages.size()); // Pre-allocate space
    for (const auto& msg : messages) {
        imu_data.emplace_back(msg);
    }

    // Clean old measurements if needed
    if (oldest_time != -1) {
        clean_old_imu_measurements(oldest_time - _prop_window);
    }
}
