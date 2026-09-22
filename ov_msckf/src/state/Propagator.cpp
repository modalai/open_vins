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

#include "Propagator.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

namespace {
// This translation unit is built with -ffast-math. Representation checks must
// survive finite-math assumptions, especially at the input/coverage boundary.
bool finite_timestamp(double value) {
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value), "64-bit IEEE double required");
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

template <class Derived> bool finite_coefficients(const Eigen::MatrixBase<Derived> &value) {
  for (Eigen::Index c = 0; c < value.cols(); ++c)
    for (Eigen::Index r = 0; r < value.rows(); ++r)
      if (!finite_timestamp(value(r, c))) return false;
  return true;
}

bool sampled_weights(const State::SampledImuRecord &left, const State::SampledImuRecord &right,
                     double time, Eigen::Vector2d &out) {
  const double span = right.timestamp - left.timestamp;
  if (!finite_timestamp(time) || !finite_timestamp(span) || !(span > 0.) ||
      time < left.timestamp || time > right.timestamp) return false;
  if (time == left.timestamp) out << 1., 0.;
  else if (time == right.timestamp) out << 0., 1.;
  else {
    const double alpha = (time - left.timestamp) / span;
    if (!finite_timestamp(alpha) || alpha < 0. || alpha > 1.) return false;
    out << 1. - alpha, alpha;
  }
  return true;
}

// Exact chain-rule tangent of the normalized-quaternion RK4 mean. Only the
// 18 driving coordinates need stage storage: bg, ba, original-left noise6,
// original-right noise6. Initial pose/velocity blocks have closed-form maps.
// Every matrix below has fixed dimensions; the caller owns the 15x27 output.
using SampledRk4Drive = Eigen::Matrix<double, 3, 18>;
using SampledRk4Quaternion = Eigen::Matrix<double, 4, 18>;

bool normalize_sampled_rk4(Eigen::Vector4d &q, SampledRk4Quaternion &derivative) {
  const double norm = q.norm();
  if (!finite_timestamp(norm) || !(norm > 0.)) return false;
  const double scale = (q(3) < 0. ? -1. : 1.) / norm;
  q *= scale;
  const Eigen::Matrix4d normalization = scale * (Eigen::Matrix4d::Identity() - q * q.transpose());
  derivative = (normalization * derivative).eval();
  return finite_coefficients(q) && finite_coefficients(derivative);
}

bool sampled_rk4_mean_tangent(const Eigen::Vector4d &q0, const Eigen::Vector3d &p0,
                              const Eigen::Vector3d &v0, const Eigen::Vector3d &gravity, double dt,
                              const Eigen::Vector3d &w0, const Eigen::Vector3d &a0,
                              const Eigen::Vector3d &w1, const Eigen::Vector3d &a1,
                              const Eigen::Matrix3d &A, const Eigen::Matrix3d &W,
                              const Eigen::Matrix3d &Tg, const Eigen::Vector2d &weights0,
                              const Eigen::Vector2d &weights1, Eigen::Vector4d &new_q,
                              Eigen::Vector3d &new_v, Eigen::Vector3d &new_p,
                              Eigen::Ref<Eigen::MatrixXd> Phi) {
  // A large pair of opposing endpoint rates must not pass the average-rate
  // guard. This local RK4 bound also keeps every incremental normalization
  // away from zero; it does not change ANALYTICAL/DISCRETE acceptance.
  const double angle0 = (dt * w0).norm(), angle1 = (dt * w1).norm();
  if (!finite_timestamp(angle0) || !finite_timestamp(angle1) || angle0 > .5 || angle1 > .5 ||
      !finite_coefficients(a0) || !finite_coefficients(a1)) return false;

  // Legacy RK4 normalizes each product with q0 before using its rotation.
  // Normalize once here so accepted near-unit input means have the same map.
  const Eigen::Vector4d q0_unit = quatnorm(q0);
  const Eigen::Matrix3d R0_transpose = quat_2_Rot(q0_unit).transpose();
  const Eigen::Matrix3d WTgA = W * Tg * A;
  const Eigen::Vector4d identity(0., 0., 0., 1.);
  constexpr double times[4] = {0., .5, .5, 1.};
  constexpr double weights[4] = {1. / 6., 1. / 3., 1. / 3., 1. / 6.};
  Eigen::Vector4d previous_kq = Eigen::Vector4d::Zero(), sum_q = identity;
  Eigen::Vector3d previous_kv = Eigen::Vector3d::Zero();
  SampledRk4Quaternion previous_Dq = SampledRk4Quaternion::Zero(), sum_Dq = SampledRk4Quaternion::Zero();
  SampledRk4Drive previous_Dv = SampledRk4Drive::Zero();
  SampledRk4Drive sum_Dp = SampledRk4Drive::Zero(), sum_Dv = SampledRk4Drive::Zero();
  Eigen::Matrix3d previous_Dv_theta = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d sum_Dp_theta = Eigen::Matrix3d::Zero(), sum_Dv_theta = Eigen::Matrix3d::Zero();
  new_p = p0;
  new_v = v0;

  for (int stage = 0; stage < 4; ++stage) {
    const double c = times[stage], coefficient = weights[stage];
    const Eigen::Vector2d raw_weights = (1. - c) * weights0 + c * weights1;
    const Eigen::Vector3d w = (1. - c) * w0 + c * w1, a = (1. - c) * a0 + c * a1;
    SampledRk4Drive Dw = SampledRk4Drive::Zero(), Da = SampledRk4Drive::Zero();
    Dw.block<3, 3>(0, 0) = -W;
    Dw.block<3, 3>(0, 3) = WTgA;
    Da.block<3, 3>(0, 3) = -A;
    for (int sample = 0; sample < 2; ++sample) {
      const int column = 6 + 6 * sample;
      Dw.block<3, 3>(0, column) = -raw_weights(sample) * W;
      Dw.block<3, 3>(0, column + 3) = raw_weights(sample) * WTgA;
      Da.block<3, 3>(0, column + 3) = -raw_weights(sample) * A;
    }

    Eigen::Vector4d dq = identity + c * previous_kq;
    SampledRk4Quaternion Ddq = c * previous_Dq;
    if (!normalize_sampled_rk4(dq, Ddq)) return false;
    // Position slope is the previous stage's velocity, before overwriting its
    // tangent. Stage0 has c=0 and therefore its exact initial-velocity slope.
    new_p.noalias() += (coefficient * dt) * (v0 + c * previous_kv);
    sum_Dp.noalias() += (coefficient * dt * c) * previous_Dv;
    sum_Dp_theta.noalias() += (coefficient * dt * c) * previous_Dv_theta;

    // Omega(w) dq is bilinear. This 4x3 map differentiates it with respect
    // to w; the normalization above accounts for every intermediate stage.
    Eigen::Matrix<double, 4, 3> angular_map;
    angular_map.topRows<3>() = dq(3) * Eigen::Matrix3d::Identity() + skew_x(dq.head<3>());
    angular_map.bottomRows<1>() = -dq.head<3>().transpose();
    const Eigen::Matrix4d omega = Omega(w);
    previous_kq.noalias() = (.5 * dt) * omega * dq;
    previous_Dq.noalias() = (.5 * dt) * (omega * Ddq + angular_map * Dw);

    const Eigen::Matrix3d R_increment_transpose = quat_2_Rot(dq).transpose();
    const Eigen::Vector3d body_acceleration = R_increment_transpose * a;
    // Derivative of R(dq)^T*a with respect to all four quaternion entries.
    Eigen::Matrix<double, 3, 4> rotation_action;
    const Eigen::Vector3d vector = dq.head<3>();
    rotation_action.leftCols<3>() = -2. * dq(3) * skew_x(a) +
        2. * vector.dot(a) * Eigen::Matrix3d::Identity() + 2. * vector * a.transpose();
    rotation_action.rightCols<1>() = 4. * dq(3) * a + 2. * vector.cross(a);
    previous_kv.noalias() = dt * (R0_transpose * body_acceleration - gravity);
    previous_Dv.noalias() = dt * R0_transpose * (rotation_action * Ddq + R_increment_transpose * Da);
    previous_Dv_theta.noalias() = -dt * R0_transpose * skew_x(body_acceleration);

    sum_q.noalias() += coefficient * previous_kq;
    sum_Dq.noalias() += coefficient * previous_Dq;
    new_v.noalias() += coefficient * previous_kv;
    sum_Dv.noalias() += coefficient * previous_Dv;
    sum_Dv_theta.noalias() += coefficient * previous_Dv_theta;
  }
  if (!normalize_sampled_rk4(sum_q, sum_Dq)) return false;
  new_q = quat_multiply(sum_q, q0_unit);
  Phi.setZero();
  Phi.block<3, 3>(0, 0) = quat_2_Rot(sum_q);
  Phi.block<3, 3>(3, 0) = sum_Dp_theta;
  Phi.block<3, 3>(6, 0) = sum_Dv_theta;
  Phi.block<3, 3>(3, 3).setIdentity();
  Phi.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity();
  Phi.block<3, 3>(6, 6).setIdentity();
  Phi.block<6, 6>(9, 9).setIdentity();
  // Convert the unit-quaternion differential into the final left JPL chart.
  Eigen::Matrix<double, 3, 4> chart;
  chart.leftCols<3>() = 2. * (sum_q(3) * Eigen::Matrix3d::Identity() - skew_x(sum_q.head<3>()));
  chart.rightCols<1>() = -2. * sum_q.head<3>();
  Phi.block<3, 18>(0, 9).noalias() = chart * sum_Dq;
  Phi.block<3, 18>(3, 9) = sum_Dp;
  Phi.block<3, 18>(6, 9) = sum_Dv;
  return finite_coefficients(new_q) && finite_coefficients(new_p) && finite_coefficients(new_v) && finite_coefficients(Phi);
}

// A second owner can request the already accepted physical endpoint. It still
// needs an actual covered raw sample for its clock Jacobian, but no interval Q.
bool select_covered_point(const std::vector<ov_core::ImuData> &data, double time, ov_core::ImuData &out) {
  if (!finite_timestamp(time) || data.empty())
    return false;
  for (size_t i = 0; i < data.size(); ++i) {
    if (!finite_timestamp(data[i].timestamp) || (i > 0 && !(data[i].timestamp > data[i - 1].timestamp)))
      return false;
  }
  if (time < data.front().timestamp || time > data.back().timestamp)
    return false;
  const auto upper = std::lower_bound(data.begin(), data.end(), time,
                                      [](const ov_core::ImuData &sample, double t) { return sample.timestamp < t; });
  const ov_core::ImuData selected = upper->timestamp == time ? *upper : Propagator::interpolate_data(*(upper - 1), *upper, time);
  for (int axis = 0; axis < 3; ++axis) {
    if (!finite_timestamp(selected.wm(axis)) || !finite_timestamp(selected.am(axis)))
      return false;
  }
  out = selected;
  return true;
}

Propagator::EndpointKinematics endpoint_kinematics(const std::shared_ptr<State> &state, const ov_core::ImuData &sample) {
  const Eigen::Matrix3d Dw = State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  const Eigen::Matrix3d Da = State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  const Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());
  const Eigen::Matrix3d Dw_fej = State::Dm(state->_options.imu_model, state->_calib_imu_dw->fej());
  const Eigen::Matrix3d Da_fej = State::Dm(state->_options.imu_model, state->_calib_imu_da->fej());
  const Eigen::Matrix3d Tg_fej = State::Tg(state->_calib_imu_tg->fej());
  const Eigen::Vector3d a = state->_calib_imu_ACCtoIMU->Rot() * Da * (sample.am - state->_imu->bias_a());
  const Eigen::Vector3d a_fej = state->_calib_imu_ACCtoIMU->Rot_fej() * Da_fej * (sample.am - state->_imu->bias_a_fej());
  Propagator::EndpointKinematics out;
  out.omega = state->_calib_imu_GYROtoIMU->Rot() * Dw * (sample.wm - state->_imu->bias_g() - Tg * a);
  out.omega_fej = state->_calib_imu_GYROtoIMU->Rot_fej() * Dw_fej * (sample.wm - state->_imu->bias_g_fej() - Tg_fej * a_fej);
  return out;
}
} // namespace

bool Propagator::propagate_and_clone(std::shared_ptr<State> state, double timestamp) {

  // A clone already exists at this exact time, or the request goes backwards: refuse instead of
  // aborting the process -- callers skip that frame's clone/update and keep running
  if (state->_timestamp == timestamp) {
    PRINT_WARNING(YELLOW "Propagator::propagate_and_clone(): called again at the last update timestep, skipping\n" RESET);
    return false;
  }
  if (state->_timestamp > timestamp) {
    PRINT_WARNING(YELLOW "Propagator::propagate_and_clone(): asked to propagate backwards (%.4f s), skipping\n" RESET,
                  (timestamp - state->_timestamp));
    return false;
  }

  const double t_off_new = state->cam_imu_dt_ref();
  EndpointKinematics endpoint;
  if (!propagate_to_imu(state, timestamp + t_off_new, timestamp, endpoint))
    return false;
  // Keep the legacy snapshot descriptor's arithmetic unchanged. The State's
  // accepted endpoint, not this descriptor, owns subsequent integration starts.
  last_prop_time_offset = t_off_new;
  StateHelper::augment_clone(state, endpoint.omega, endpoint.omega_fej);
  return true;
}

bool Propagator::propagate_to_imu(std::shared_ptr<State> state, double target_imu, double reference_timestamp,
                                 EndpointKinematics &out) {
  // Explicit sampled ownership must never silently receive continuous sensor
  // Q as well. No runtime selects the sampled proof caller yet.
  if (!state || state->has_sampled_imu_boundary()) return false;
  const double time0 = state->imu_endpoint();
  const double time1 = target_imu;
  if (!finite_timestamp(time0) || !finite_timestamp(time1) || !finite_timestamp(reference_timestamp) ||
      !finite_timestamp(time1 - time0) || !finite_timestamp(time1 - reference_timestamp) || time1 < time0)
    return false;

  std::vector<ov_core::ImuData> prop_data;
  ov_core::ImuData endpoint_sample;
  {
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    if (time1 == time0) {
      if (!select_covered_point(imu_data, time1, endpoint_sample))
        return false;
    } else {
      prop_data = Propagator::select_imu_readings(imu_data, time0, time1);
      if (prop_data.size() < 2)
        return false;
      endpoint_sample = prop_data.back();
    }
  }

  if (time1 == time0) {
    const auto endpoint = endpoint_kinematics(state, endpoint_sample);
    if (!finite_coefficients(endpoint.omega) || !finite_coefficients(endpoint.omega_fej)) return false;
    out = endpoint;
    state->_timestamp = reference_timestamp;
    state->_imu_endpoint = time1;
    state->_imu_endpoint_valid = true;
    last_prop_time_offset = time1 - reference_timestamp;
    have_last_prop_time_offset = true;
    invalidate_cache();
    return true;
  }

  // Only the 16 navigation values/FEJ are changed by the mean integrator.
  // Retain these bounded snapshots until the aggregate covariance is accepted.
  const Eigen::Matrix<double,16,1> original_value = state->_imu->value();
  const Eigen::Matrix<double,16,1> original_fej = state->_imu->fej();
  if (!finite_coefficients(original_value) || !finite_coefficients(original_fej)) return false;
  const auto reject_mean = [&]() {
    state->_imu->set_value(original_value); state->_imu->set_fej(original_fej); return false;
  };

  // We are going to sum up all the state transition matrices, so we can do a single large multiplication at the end
  // Phi_summed = Phi_i*Phi_summed
  // Q_summed = Phi_i*Q_summed*Phi_i^T + Q_i
  // After summing we can multiple the total phi to get the updated covariance
  // We will then add the noise to the IMU portion of the state
  Eigen::MatrixXd Phi_summed = Eigen::MatrixXd::Identity(state->imu_intrinsic_size() + 15, state->imu_intrinsic_size() + 15);
  Eigen::MatrixXd Qd_summed = Eigen::MatrixXd::Zero(state->imu_intrinsic_size() + 15, state->imu_intrinsic_size() + 15);
  double dt_summed = 0;

  // Loop through all IMU messages, and use them to move the state forward in time
  // This uses the zero'th order quat, and then constant acceleration discrete
  if (prop_data.size() > 1) {
    for (size_t i = 0; i < prop_data.size() - 1; i++) {

      // Get the next state Jacobian and noise Jacobian for this IMU reading
      Eigen::MatrixXd F, Qdi;
      predict_and_compute(state, prop_data.at(i), prop_data.at(i + 1), F, Qdi);
      if (!finite_coefficients(state->_imu->value()) || !finite_coefficients(state->_imu->fej()) ||
          !finite_coefficients(F) || !finite_coefficients(Qdi)) return reject_mean();

      // Next we should propagate our IMU covariance
      // Pii' = F*Pii*F.transpose() + G*Q*G.transpose()
      // Pci' = F*Pci and Pic' = Pic*F.transpose()
      // NOTE: Here we are summing the state transition F so we can do a single mutiplication later
      // NOTE: Phi_summed = Phi_i*Phi_summed
      // NOTE: Q_summed = Phi_i*Q_summed*Phi_i^T + G*Q_i*G^T
      Phi_summed = F * Phi_summed;
      Qd_summed = F * Qd_summed * F.transpose() + Qdi;
      Qd_summed = 0.5 * (Qd_summed + Qd_summed.transpose());
      dt_summed += prop_data.at(i + 1).timestamp - prop_data.at(i).timestamp;
    }
  }
  assert(std::abs((time1 - time0) - dt_summed) < 1e-4);

  const EndpointKinematics endpoint = endpoint_kinematics(state, endpoint_sample);
  if (!finite_coefficients(endpoint.omega) || !finite_coefficients(endpoint.omega_fej)) return reject_mean();

  // Do the update to the covariance with our "summed" state transition and IMU noise addition...
  std::vector<std::shared_ptr<Type>> Phi_order;
  Phi_order.push_back(state->_imu);
  if (state->_options.do_calib_imu_intrinsics) {
    Phi_order.push_back(state->_calib_imu_dw);
    Phi_order.push_back(state->_calib_imu_da);
    if (state->_options.do_calib_imu_g_sensitivity) {
      Phi_order.push_back(state->_calib_imu_tg);
    }
    if (state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
      Phi_order.push_back(state->_calib_imu_GYROtoIMU);
    } else {
      Phi_order.push_back(state->_calib_imu_ACCtoIMU);
    }
  }
  if (!StateHelper::EKFPropagation(state, Phi_order, Phi_order, Phi_summed, Qd_summed)) return reject_mean();

  // Commit both clocks only after a fully covered propagation. The accepted
  // physical endpoint survives online clock updates, ZUPT and state snapshots.
  state->_timestamp = reference_timestamp;
  state->_imu_endpoint = time1;
  state->_imu_endpoint_valid = true;
  last_prop_time_offset = time1 - reference_timestamp;
  have_last_prop_time_offset = true;
  invalidate_cache();
  out = endpoint;
  return true;
}

bool Propagator::compute_bridge(std::shared_ptr<State> state, double t0_imu, double t1_imu, BridgeData &out) {
  // This legacy deterministic bridge neither subtracts retained raw-noise
  // posterior means nor carries their stochastic outputs. Refuse before even
  // clearing the caller's output when sampled ownership has been prepared.
  if (!state || state->has_sampled_imu_boundary()) return false;
  out = BridgeData();
  if (!(t1_imu > t0_imu)) {
    return false;
  }

  // Collect the (boundary-interpolated) IMU samples spanning the interval
  std::vector<ov_core::ImuData> prop_data;
  {
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    prop_data = Propagator::select_imu_readings(imu_data, t0_imu, t1_imu, false);
  }
  if (prop_data.size() < 2) {
    return false;
  }

  // Static intrinsics and the build-time bias linearization (ACI2 partial-fixed estimates)
  const Eigen::Matrix3d Dw = State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  const Eigen::Matrix3d Da = State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  const Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());
  const Eigen::Matrix3d R_ACCtoIMU = state->_calib_imu_ACCtoIMU->Rot();
  const Eigen::Matrix3d R_GYROtoIMU = state->_calib_imu_GYROtoIMU->Rot();
  const Eigen::Matrix3d A_calib = R_ACCtoIMU * Da;
  const Eigen::Matrix3d W_calib = R_GYROtoIMU * Dw;
  out.bg0 = state->_imu->bias_g();
  out.ba0 = state->_imu->bias_a();

  // Relative accumulators (see BridgeData docs). J blocks: rows th(0:3), alpha(3:6), beta(6:9);
  // cols bg(0:3), ba(3:6). J_th is kept in the CURRENT interval's start frame during each step
  // (that is what the alpha/beta chain rules consume), then transported by the step rotation.
  Eigen::Matrix3d DR = Eigen::Matrix3d::Identity();
  Eigen::Vector3d alpha = Eigen::Vector3d::Zero(), beta = Eigen::Vector3d::Zero();
  Eigen::Matrix3d J_th_g = Eigen::Matrix3d::Zero(), J_th_a = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_a_g = Eigen::Matrix3d::Zero(), J_a_a = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_b_g = Eigen::Matrix3d::Zero(), J_b_a = Eigen::Matrix3d::Zero();

  for (size_t i = 0; i + 1 < prop_data.size(); i++) {
    const double dt = prop_data.at(i + 1).timestamp - prop_data.at(i).timestamp;
    if (!(dt > 0)) {
      continue;
    }

    // Bias/intrinsics-corrected signals (identical chain to predict_and_compute)
    Eigen::Vector3d a_hat1 = R_ACCtoIMU * Da * (prop_data.at(i).am - out.ba0);
    Eigen::Vector3d a_hat2 = R_ACCtoIMU * Da * (prop_data.at(i + 1).am - out.ba0);
    Eigen::Vector3d a_hat = 0.5 * (a_hat1 + a_hat2);
    Eigen::Vector3d w_hat1 = R_GYROtoIMU * Dw * (prop_data.at(i).wm - out.bg0 - Tg * a_hat1);
    Eigen::Vector3d w_hat2 = R_GYROtoIMU * Dw * (prop_data.at(i + 1).wm - out.bg0 - Tg * a_hat2);
    Eigen::Vector3d w_hat = 0.5 * (w_hat1 + w_hat2);
    out.w_end = w_hat2;

    // Analytic integration components for this step (shared closed forms, small-omega safe)
    Eigen::Matrix<double, 3, 18> Xi_sum = Eigen::Matrix<double, 3, 18>::Zero();
    compute_Xi_sum(state, dt, w_hat, a_hat, Xi_sum);
    const Eigen::Matrix3d R_step = Xi_sum.block(0, 0, 3, 3); // exp_so3(-w dt) = R_{I_i -> I_i+1}
    const Eigen::Matrix3d Xi_1 = Xi_sum.block(0, 3, 3, 3);
    const Eigen::Matrix3d Xi_2 = Xi_sum.block(0, 6, 3, 3);
    const Eigen::Matrix3d Jr_step = Xi_sum.block(0, 9, 3, 3); // Jr_so3(-w dt), shared with the stock path
    const Eigen::Matrix3d Xi_3 = Xi_sum.block(0, 12, 3, 3);
    const Eigen::Matrix3d Xi_4 = Xi_sum.block(0, 15, 3, 3);

    const Eigen::Matrix3d A = DR.transpose(); // maps I_i-frame vectors into the I_k frame
    const Eigen::Vector3d X1a = Xi_1 * a_hat;
    const Eigen::Vector3d X2a = Xi_2 * a_hat;

    // ---- bias Jacobians (consume the PRE-transport J_th, then advance everything).
    // Coupling sign matches THIS file's J_th convention (DR(b) = exp_so3(J_th db) DR(b0)),
    // pinned by the finite-difference oracle in test_preint_bridge ----
    // Bias states live in the raw sensor frames. A raw accel-bias change also
    // changes the corrected gyro through Tg, including its accumulated rotation.
    J_a_g += J_b_g * dt + A * (Xi_4 * W_calib + ov_core::skew_x(X2a) * J_th_g);
    J_a_a += J_b_a * dt + A * (-(Xi_2 + Xi_4 * W_calib * Tg) * A_calib + ov_core::skew_x(X2a) * J_th_a);
    J_b_g += A * (Xi_3 * W_calib + ov_core::skew_x(X1a) * J_th_g);
    J_b_a += A * (-(Xi_1 + Xi_3 * W_calib * Tg) * A_calib + ov_core::skew_x(X1a) * J_th_a);
    // Exact increment for the left perturbation DR(b) = exp_so3(J_th db) DR(b0):
    // exp(-w dt + db dt) = R_step exp(Jr(-w dt) db dt) and R_step Jr(-w dt) = Jr(+w dt),
    // so the increment needs the +w flavor (Jr(-w dt) alone errs O(|w| dt) per step).
    // Factored to reuse the Jr(-w dt) block Xi_sum already carries -- no fresh Jr_so3
    // evaluation on the RT path; identical math, pinned by test_preint_bridge.
    J_th_g = R_step * (J_th_g + Jr_step * dt * W_calib);
    J_th_a = R_step * (J_th_a - Jr_step * dt * W_calib * Tg * A_calib);

    // ---- mean (alpha consumes the PRE-update beta) ----
    alpha += beta * dt + A * X2a;
    beta += A * X1a;
    DR = R_step * DR;
  }

  out.dt = t1_imu - t0_imu;
  out.DR = DR;
  out.alpha = alpha;
  out.beta = beta;
  out.p_grav = -0.5 * _gravity * out.dt * out.dt;
  out.v_grav = -_gravity * out.dt;
  out.J_b.block(0, 0, 3, 3) = J_th_g;
  out.J_b.block(0, 3, 3, 3) = J_th_a;
  out.J_b.block(3, 0, 3, 3) = J_a_g;
  out.J_b.block(3, 3, 3, 3) = J_a_a;
  out.J_b.block(6, 0, 3, 3) = J_b_g;
  out.J_b.block(6, 3, 3, 3) = J_b_a;
  out.valid = true;
  return true;
}

bool Propagator::fast_state_propagate(std::shared_ptr<State> state, double timestamp, Eigen::Matrix<double, 13, 1> &state_plus,
                                      Eigen::Matrix<double, 12, 12> &covariance) {
  if (!state || state->has_sampled_imu_boundary()) return false;

  // Check coverage before initializing or advancing the mutable cache. The
  // requested endpoint and the accepted navigation endpoint use the IMU clock.
  const bool had_cache = cache_imu_valid.load();
  const double time0 = had_cache ? cache_state_time + cache_t_off : state->imu_endpoint();
  const double time1 = timestamp;
  std::vector<ov_core::ImuData> prop_data;
  {
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    prop_data = Propagator::select_imu_readings(imu_data, time0, time1, false);
  }
  if (prop_data.size() < 2)
    return false;

  // Predict in fixed navigation-sized scratch. Rejecting a later interval must
  // preserve the last valid cache as well as the caller's published output.
  Eigen::Matrix<double,16,1> next_est;
  Eigen::Matrix<double,15,15> next_cov;
  if (had_cache) { next_est=cache_state_est; next_cov=cache_state_covariance; }
  else { next_est=state->_imu->value(); next_cov=StateHelper::get_marginal_covariance(state, {state->_imu}); }
  Eigen::Matrix<double,13,1> next_output;
  Eigen::Matrix<double,12,12> next_output_cov;
  if (!finite_coefficients(next_est) || !finite_coefficients(next_cov)) return false;

  // Biases
  Eigen::Vector3d bias_g = next_est.block(10, 0, 3, 1);
  Eigen::Vector3d bias_a = next_est.block(13, 0, 3, 1);

  // IMU intrinsic calibration estimates (static)
  Eigen::Matrix3d Dw = State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  Eigen::Matrix3d Da = State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());
  Eigen::Matrix3d R_ACCtoIMU = state->_calib_imu_ACCtoIMU->Rot();
  Eigen::Matrix3d R_GYROtoIMU = state->_calib_imu_GYROtoIMU->Rot();
  const Eigen::Matrix3d A_calib = R_ACCtoIMU * Da;
  const Eigen::Matrix3d W_calib = R_GYROtoIMU * Dw;

  // Loop through all IMU messages, and use them to move the state forward in time
  // This uses the zero'th order quat, and then constant acceleration discrete
  for (size_t i = 0; i < prop_data.size() - 1; i++) {

    // Time elapsed over interval
    auto data_minus = prop_data.at(i);
    auto data_plus = prop_data.at(i + 1);
    double dt = data_plus.timestamp - data_minus.timestamp;

    // Corrected imu acc measurements with our current biases
    Eigen::Vector3d a_hat1 = R_ACCtoIMU * Da * (data_minus.am - bias_a);
    Eigen::Vector3d a_hat2 = R_ACCtoIMU * Da * (data_plus.am - bias_a);
    Eigen::Vector3d a_hat = 0.5 * (a_hat1 + a_hat2);

    // Corrected imu gyro measurements with our current biases
    Eigen::Vector3d w_hat1 = R_GYROtoIMU * Dw * (data_minus.wm - bias_g - Tg * a_hat1);
    Eigen::Vector3d w_hat2 = R_GYROtoIMU * Dw * (data_plus.wm - bias_g - Tg * a_hat2);
    Eigen::Vector3d w_hat = 0.5 * (w_hat1 + w_hat2);

    // Current state estimates
    Eigen::Matrix3d R_Gtoi = quat_2_Rot(next_est.block(0, 0, 4, 1));
    Eigen::Vector3d v_iinG = next_est.block(7, 0, 3, 1);
    Eigen::Vector3d p_iinG = next_est.block(4, 0, 3, 1);

    // State transition and noise matrix
    // TODO: should probably track the correlations with the IMU intrinsics if we are calibrating
    // TODO: currently this just does a quick discrete prediction using only the previous marg IMU uncertainty
    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Zero();
    F.block(0, 0, 3, 3) = exp_so3(-w_hat * dt);
    F.block(0, 9, 3, 3).noalias() = -exp_so3(-w_hat * dt) * Jr_so3(-w_hat * dt) * dt * W_calib;
    F.block(0, 12, 3, 3).noalias() = -F.block(0, 9, 3, 3) * Tg * A_calib;
    F.block(9, 9, 3, 3).setIdentity();
    F.block(6, 0, 3, 3).noalias() = -R_Gtoi.transpose() * skew_x(a_hat * dt);
    F.block(6, 6, 3, 3).setIdentity();
    F.block(6, 12, 3, 3) = -R_Gtoi.transpose() * dt * A_calib;
    F.block(12, 12, 3, 3).setIdentity();
    F.block(3, 0, 3, 3).noalias() = -0.5 * R_Gtoi.transpose() * skew_x(a_hat * dt * dt);
    F.block(3, 6, 3, 3) = Eigen::Matrix3d::Identity() * dt;
    F.block(3, 12, 3, 3) = -0.5 * R_Gtoi.transpose() * dt * dt * A_calib;
    F.block(3, 3, 3, 3).setIdentity();
    Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();
    G.block(0, 0, 3, 3) = F.block(0, 9, 3, 3);
    G.block(0, 3, 3, 3) = F.block(0, 12, 3, 3);
    G.block(6, 3, 3, 3) = F.block(6, 12, 3, 3);
    G.block(3, 3, 3, 3) = F.block(3, 12, 3, 3);
    G.block(9, 6, 3, 3).setIdentity();
    G.block(12, 9, 3, 3).setIdentity();

    // Construct our discrete noise covariance matrix
    // Note that we need to convert our continuous time noises to discrete
    // Equations (129) amd (130) of Trawny tech report
    Eigen::Matrix<double, 15, 15> Qd = Eigen::Matrix<double, 15, 15>::Zero();
    Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
    Qc.block(0, 0, 3, 3) = _noises.sigma_w_2 / dt * Eigen::Matrix3d::Identity();
    Qc.block(3, 3, 3, 3) = _noises.sigma_a_2 / dt * Eigen::Matrix3d::Identity();
    Qc.block(6, 6, 3, 3) = _noises.sigma_wb_2 * dt * Eigen::Matrix3d::Identity();
    Qc.block(9, 9, 3, 3) = _noises.sigma_ab_2 * dt * Eigen::Matrix3d::Identity();
    Qd = G * Qc * G.transpose();
    Qd = 0.5 * (Qd + Qd.transpose());
    next_cov = F * next_cov * F.transpose() + Qd;

    // Propagate the mean forward
    next_est.block(0, 0, 4, 1) = rot_2_quat(exp_so3(-w_hat * dt) * R_Gtoi);
    next_est.block(4, 0, 3, 1) = p_iinG + v_iinG * dt + 0.5 * R_Gtoi.transpose() * a_hat * dt * dt - 0.5 * _gravity * dt * dt;
    next_est.block(7, 0, 3, 1) = v_iinG + R_Gtoi.transpose() * a_hat * dt - _gravity * dt;
    if (!finite_coefficients(next_est) || !finite_coefficients(next_cov)) return false;
  }

  // Now record what the predicted state should be
  Eigen::Vector4d q_Gtoi = next_est.block(0, 0, 4, 1);
  Eigen::Vector3d v_iinG = next_est.block(7, 0, 3, 1);
  Eigen::Vector3d p_iinG = next_est.block(4, 0, 3, 1);
  next_output.setZero();
  next_output.block(0, 0, 4, 1) = q_Gtoi;
  next_output.block(4, 0, 3, 1) = p_iinG;
  next_output.block(7, 0, 3, 1) = quat_2_Rot(q_Gtoi) * v_iinG; // local frame v_iini
  Eigen::Vector3d last_a = R_ACCtoIMU * Da * (prop_data.at(prop_data.size() - 1).am - bias_a);
  Eigen::Vector3d last_w = R_GYROtoIMU * Dw * (prop_data.at(prop_data.size() - 1).wm - bias_g - Tg * last_a);
  next_output.block(10, 0, 3, 1) = last_w;

  // Pull the IMU marginal into the published coordinates. Body velocity depends
  // on attitude as well as global velocity; corrected angular rate depends on
  // both raw biases when Tg is nonzero.
  Eigen::Matrix<double, 12, 15> J = Eigen::Matrix<double, 12, 15>::Zero();
  J.topLeftCorner<6, 6>().setIdentity();
  J.block<3, 3>(6, 0) = skew_x(next_output.segment<3>(7));
  J.block<3, 3>(6, 6) = quat_2_Rot(q_Gtoi);
  J.block<3, 3>(9, 9) = -W_calib;
  J.block<3, 3>(9, 12) = W_calib * Tg * A_calib;

  // Approximate the output measurement noise as independent of the cached state.
  // The mean uses the final sample, while propagation averages/interpolates
  // samples. An exact joint covariance needs their raw-sample noise ownership
  // and its correlation with the starting filter marginal; the interval Qd
  // alone cannot supply a valid endpoint cross-covariance coefficient.
  const double dt = prop_data.back().timestamp - prop_data.at(prop_data.size() - 2).timestamp;
  Eigen::Matrix<double, 6, 6> Qraw = Eigen::Matrix<double, 6, 6>::Zero();
  Qraw.topLeftCorner<3, 3>() = _noises.sigma_w_2 / dt * Eigen::Matrix3d::Identity();
  Qraw.bottomRightCorner<3, 3>() = _noises.sigma_a_2 / dt * Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 12, 6> D = Eigen::Matrix<double, 12, 6>::Zero();
  D.block<3, 3>(9, 0) = -W_calib;
  D.block<3, 3>(9, 3) = W_calib * Tg * A_calib;
  next_output_cov = J * next_cov * J.transpose() + D * Qraw * D.transpose();
  next_output_cov = (0.5 * (next_output_cov + next_output_cov.transpose())).eval();
  if (!finite_coefficients(next_output) || !finite_coefficients(next_output_cov)) return false;
  for (int i=0;i<12;++i) if (next_output_cov(i,i)<0.) return false;
  cache_state_est = next_est; cache_state_covariance = next_cov;
  cache_state_time = time1; cache_t_off = 0.; cache_imu_valid = true;
  state_plus = next_output; covariance = next_output_cov;
  return true;
}

std::vector<ov_core::ImuData> Propagator::select_imu_readings(const std::vector<ov_core::ImuData> &imu_data, double time0, double time1,
                                                              bool warn) {

  // The caller may not propagate any part of an uncovered interval. In
  // particular, never extrapolate the newest pair to a future endpoint.
  auto reject = [warn]() {
    if (warn)
      PRINT_WARNING(YELLOW "Propagator::select_imu_readings(): invalid or unbracketed IMU interval, skipping\n" RESET);
    return std::vector<ov_core::ImuData>();
  };
  if (!finite_timestamp(time0) || !finite_timestamp(time1) || !(time1 > time0) ||
      !finite_timestamp(time1 - time0) || imu_data.size() < 2)
    return reject();

  // Validate ordering once, in linear time. lower_bound requires this contract;
  // duplicate/nonfinite stamps cannot form a positive-duration integration step.
  for (size_t i = 0; i < imu_data.size(); ++i) {
    if (!finite_timestamp(imu_data[i].timestamp) || (i > 0 && !(imu_data[i].timestamp > imu_data[i - 1].timestamp)))
      return reject();
  }
  if (time0 < imu_data.front().timestamp || time1 > imu_data.back().timestamp)
    return reject();

  const auto first = std::lower_bound(imu_data.begin(), imu_data.end(), time0,
                                     [](const ov_core::ImuData &sample, double time) { return sample.timestamp < time; });
  const auto last = std::lower_bound(first, imu_data.end(), time1,
                                    [](const ov_core::ImuData &sample, double time) { return sample.timestamp < time; });
  std::vector<ov_core::ImuData> prop_data;
  prop_data.reserve(static_cast<size_t>(last - first) + 2);
  // Preserve exact sample values at exact endpoints. If both endpoints lie in
  // one sample interval, interpolate each from that same enclosing pair.
  prop_data.push_back(first->timestamp == time0 ? *first : interpolate_data(*(first - 1), *first, time0));
  for (auto sample = first; sample != last; ++sample) {
    if (sample->timestamp > time0)
      prop_data.push_back(*sample);
  }
  prop_data.push_back(last->timestamp == time1 ? *last : interpolate_data(*(last - 1), *last, time1));

  // Reject nonfinite used signals before callers touch means/covariances. This
  // also catches overflow in interpolation without relying on fast-math isfinite.
  for (const auto &sample : prop_data) {
    for (int axis = 0; axis < 3; ++axis) {
      if (!finite_timestamp(sample.wm(axis)) || !finite_timestamp(sample.am(axis)))
        return reject();
    }
  }
  return prop_data;
}

bool Propagator::select_sampled_imu_readings(const std::vector<State::SampledImuRecord> &records,
                                           double time0, double time1, std::vector<SampledImuSegment> &out) {
  if (records.size() < 2 || !finite_timestamp(time0) || !finite_timestamp(time1) ||
      !finite_timestamp(time1 - time0) || !(time1 > time0)) return false;
  for (size_t i = 0; i < records.size(); ++i) {
    if (!StateHelper::valid_sampled_imu_record(records[i]) || records[i].stream_episode != records[0].stream_episode ||
        (i && (!(records[i].timestamp > records[i - 1].timestamp) || !(records[i].sequence > records[i - 1].sequence))))
      return false;
  }
  if (time0 < records.front().timestamp || time1 > records.back().timestamp) return false;
  // Validate/count without touching the caller's vector. A second bounded pass
  // writes only into its already reserved storage; no rollback buffer is needed.
  size_t count = 0;
  for (size_t i = 0; i + 1 < records.size(); ++i) {
    const double begin = std::max(time0, records[i].timestamp), end = std::min(time1, records[i + 1].timestamp);
    if (!(end > begin)) continue;
    Eigen::Vector2d w0, w1;
    if (!sampled_weights(records[i], records[i + 1], begin, w0) ||
        !sampled_weights(records[i], records[i + 1], end, w1)) return false;
    ++count;
  }
  if (!count || count > out.capacity()) return false;
  out.resize(count);
  size_t selected = 0;
  for (size_t i = 0; i + 1 < records.size(); ++i) {
    const double begin = std::max(time0, records[i].timestamp), end = std::min(time1, records[i + 1].timestamp);
    if (!(end > begin)) continue;
    auto &segment = out[selected++];
    segment.records = {records[i], records[i + 1]};
    segment.time0 = begin; segment.time1 = end;
    sampled_weights(records[i], records[i + 1], begin, segment.weights0);
    sampled_weights(records[i], records[i + 1], end, segment.weights1);
  }
  return true;
}

bool Propagator::propagate_sampled_segment(std::shared_ptr<State> state, const SampledImuSegment &segment,
                                          double reference_timestamp, EndpointKinematics &out,
                                          SampledPropagationLinearization *linearization) {
  if (!state || !state->has_sampled_imu_boundary() || !state->_imu_endpoint_valid || state->_options.do_fej ||
      state->_options.do_calib_imu_intrinsics || state->_options.do_calib_imu_g_sensitivity ||
      state->imu_intrinsic_size() != 0 ||
      (state->_options.integration_method != StateOptions::ANALYTICAL && state->_options.integration_method != StateOptions::DISCRETE &&
       state->_options.integration_method != StateOptions::RK4) ||
      ov_core::initializer_time_bits(segment.time0) != ov_core::initializer_time_bits(state->imu_endpoint()) ||
      !finite_timestamp(reference_timestamp) || !finite_timestamp(segment.time1 - reference_timestamp)) return false;
  const double dt = segment.time1 - segment.time0;
  if (!finite_timestamp(dt) || !(dt > 0.)) return false;
  Eigen::Vector2d w0, w1;
  if (!sampled_weights(segment.records[0], segment.records[1], segment.time0, w0) ||
      !sampled_weights(segment.records[0], segment.records[1], segment.time1, w1) ||
      !finite_coefficients(segment.weights0) || !finite_coefficients(segment.weights1) ||
      (w0.array() != segment.weights0.array()).any() || (w1.array() != segment.weights1.array()).any()) return false;
  // Active posterior means live in reusable slots, located by original record
  // identity rather than by slot address or the interpolated endpoint stamp.
  Eigen::Matrix<double, 6, 2> corrected;
  for (int i = 0; i < 2; ++i) {
    if (!StateHelper::valid_sampled_imu_record(segment.records[i])) return false;
    corrected.col(i) = segment.records[i].measured;
    for (const auto &slot : state->sampled_imu_slots())
      if (slot.active && slot.record.stream_episode == segment.records[i].stream_episode &&
          slot.record.sequence == segment.records[i].sequence) corrected.col(i) -= slot.noise->value();
  }
  const Eigen::Matrix<double, 6, 1> raw0 = corrected * w0, raw1 = corrected * w1;
  const Eigen::Matrix3d A = state->_calib_imu_ACCtoIMU->Rot() * State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  const Eigen::Matrix3d W = state->_calib_imu_GYROtoIMU->Rot() * State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  const Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());
  const Eigen::Vector3d a0 = raw0.tail<3>() - state->_imu->bias_a(), a1 = raw1.tail<3>() - state->_imu->bias_a();
  const Eigen::Vector3d a_uncorrected = .5 * (a0 + a1), a = A * a_uncorrected;
  const Eigen::Vector3d w_uncorrected = .5 * ((raw0.head<3>() - state->_imu->bias_g() - Tg * A * a0) +
                                             (raw1.head<3>() - state->_imu->bias_g() - Tg * A * a1));
  const Eigen::Vector3d w = W * w_uncorrected;
  const double angle = (w * dt).norm();
  if (!finite_coefficients(state->_imu->value()) || !finite_coefficients(A) || !finite_coefficients(W) ||
      !finite_coefficients(Tg) || !finite_coefficients(a) || !finite_coefficients(w) ||
      !finite_timestamp(angle) || angle > .5 || std::abs(state->_imu->quat().squaredNorm() - 1.) > 1e-9) return false;
  Eigen::Matrix<double, 3, 18> Xi;
  compute_Xi_sum(state, dt, w, a, Xi);
  Eigen::Vector4d new_q;
  Eigen::Vector3d new_v, new_p;
  SampledPropagationLinearization staged;
  staged.Phi.resize(15, 27);
  if (state->_options.integration_method == StateOptions::RK4) {
    const Eigen::Vector3d calibrated_a0 = A * a0, calibrated_a1 = A * a1;
    const Eigen::Vector3d calibrated_w0 = W * (raw0.head<3>() - state->_imu->bias_g() - Tg * calibrated_a0);
    const Eigen::Vector3d calibrated_w1 = W * (raw1.head<3>() - state->_imu->bias_g() - Tg * calibrated_a1);
    if (!sampled_rk4_mean_tangent(state->_imu->quat(), state->_imu->pos(), state->_imu->vel(), _gravity, dt,
                                 calibrated_w0, calibrated_a0, calibrated_w1, calibrated_a1, A, W, Tg, w0, w1,
                                 new_q, new_v, new_p, staged.Phi)) return false;
  } else {
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15), G = Eigen::MatrixXd::Zero(15, 12);
    if (state->_options.integration_method == StateOptions::ANALYTICAL) {
      predict_mean_analytic(state, dt, w, a, new_q, new_v, new_p, Xi);
      compute_F_and_G_analytic(state, dt, w, a, w_uncorrected, a_uncorrected, new_q, new_v, new_p, Xi, F, G);
    } else {
      predict_mean_discrete(state, dt, w, a, new_q, new_v, new_p);
      compute_F_and_G_discrete(state, dt, w, a, w_uncorrected, a_uncorrected, new_q, new_v, new_p, F, G);
    }
    // Xi1 = dt*Jr(-omega*dt), including its small-angle terms. Differentiate
    // the actual mean without the legacy Jr=I approximation or a principal-log
    // reconstruction of the discrete increment. The ordinary caller is unchanged.
    const Eigen::Matrix3d dR = quat_2_Rot(new_q) * state->_imu->Rot().transpose();
    F.block<3, 3>(0, 9) = -dR * Xi.block<3, 3>(0, 3) * W;
    F.block<3, 3>(0, 12) = -F.block<3, 3>(0, 9) * Tg * A;
    G.block<3, 6>(0, 0) = F.block<3, 6>(0, 9);
    staged.Phi.leftCols<15>() = F;
    const Eigen::Vector2d weights = .5 * (w0 + w1);
    staged.Phi.middleCols<6>(15) = weights(0) * G.leftCols<6>();
    staged.Phi.rightCols<6>() = weights(1) * G.leftCols<6>();
  }
  staged.independent_Q = Eigen::MatrixXd::Zero(15, 15);
  if (state->_options.integration_method == StateOptions::ANALYTICAL || state->_options.integration_method == StateOptions::RK4) {
    // Preserve the declared frozen-average continuous bias-only approximation,
    // with its attitude impulse chart at the actual RK4 endpoint. This Q is
    // separate from the exact RK4 mean tangent above.
    staged.independent_Q = compute_Qd_analytic(state, dt, w, a, new_q, Xi, false);
  } else {
    staged.independent_Q.block<3, 3>(9, 9) = (_noises.sigma_wb_2 * dt) * Eigen::Matrix3d::Identity();
    staged.independent_Q.block<3, 3>(12, 12) = (_noises.sigma_ab_2 * dt) * Eigen::Matrix3d::Identity();
  }
  Eigen::Matrix<double, 16, 1> imu_x = state->_imu->value();
  imu_x.head<4>() = new_q; imu_x.segment<3>(4) = new_p; imu_x.segment<3>(7) = new_v;
  EndpointKinematics endpoint;
  endpoint.omega = W * (raw1.head<3>() - state->_imu->bias_g() - Tg * A * a1);
  endpoint.omega_fej = endpoint.omega; // the proof supports current tangents only
  if (!finite_coefficients(imu_x) || !finite_coefficients(endpoint.omega)) return false;
  if (!StateHelper::EKFPropagationSampled(state, segment.records, staged.Phi, staged.independent_Q)) return false;
  state->_imu->set_value(imu_x); state->_imu->set_fej(imu_x);
  state->_timestamp = reference_timestamp;
  state->_imu_endpoint = segment.time1; state->_imu_endpoint_valid = true;
  // Both records were admitted atomically above. Exact-bit knot equality is
  // the same condition used by retirement; no fallible operation follows it.
  if (ov_core::initializer_time_bits(segment.time1) == ov_core::initializer_time_bits(segment.records[1].timestamp))
    StateHelper::retire_sampled_imu_noise_at_knot(state, segment.records[0].sequence);
  last_prop_time_offset = segment.time1 - reference_timestamp; have_last_prop_time_offset = true;
  invalidate_cache(); out = endpoint;
  if (linearization) {
    linearization->Phi.swap(staged.Phi);
    linearization->independent_Q.swap(staged.independent_Q);
  }
  return true;
}

void Propagator::predict_and_compute(std::shared_ptr<State> state, const ov_core::ImuData &data_minus, const ov_core::ImuData &data_plus,
                                     Eigen::MatrixXd &F, Eigen::MatrixXd &Qd) {

  // Time elapsed over interval
  double dt = data_plus.timestamp - data_minus.timestamp;
  // assert(data_plus.timestamp>data_minus.timestamp);

  // IMU intrinsic calibration estimates (static)
  Eigen::Matrix3d Dw = State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  Eigen::Matrix3d Da = State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());

  // Corrected imu acc measurements with our current biases
  Eigen::Vector3d a_hat1 = data_minus.am - state->_imu->bias_a();
  Eigen::Vector3d a_hat2 = data_plus.am - state->_imu->bias_a();
  Eigen::Vector3d a_hat_avg = .5 * (a_hat1 + a_hat2);

  // Convert "raw" imu to its corrected frame using the IMU intrinsics
  Eigen::Vector3d a_uncorrected = a_hat_avg;
  Eigen::Matrix3d R_ACCtoIMU = state->_calib_imu_ACCtoIMU->Rot();
  a_hat1 = R_ACCtoIMU * Da * a_hat1;
  a_hat2 = R_ACCtoIMU * Da * a_hat2;
  a_hat_avg = R_ACCtoIMU * Da * a_hat_avg;

  // Corrected imu gyro measurements with our current biases and gravity sensitivity
  Eigen::Vector3d w_hat1 = data_minus.wm - state->_imu->bias_g() - Tg * a_hat1;
  Eigen::Vector3d w_hat2 = data_plus.wm - state->_imu->bias_g() - Tg * a_hat2;
  Eigen::Vector3d w_hat_avg = .5 * (w_hat1 + w_hat2);

  // Convert "raw" imu to its corrected frame using the IMU intrinsics
  Eigen::Vector3d w_uncorrected = w_hat_avg;
  Eigen::Matrix3d R_GYROtoIMU = state->_calib_imu_GYROtoIMU->Rot();
  w_hat1 = R_GYROtoIMU * Dw * w_hat1;
  w_hat2 = R_GYROtoIMU * Dw * w_hat2;
  w_hat_avg = R_GYROtoIMU * Dw * w_hat_avg;

  // Pre-compute some analytical values for the mean and covariance integration
  Eigen::Matrix<double, 3, 18> Xi_sum = Eigen::Matrix<double, 3, 18>::Zero(3, 18);
  if (state->_options.integration_method == StateOptions::IntegrationMethod::RK4 ||
      state->_options.integration_method == StateOptions::IntegrationMethod::ANALYTICAL) {
    compute_Xi_sum(state, dt, w_hat_avg, a_hat_avg, Xi_sum);
  }

  // Compute the new state mean value
  Eigen::Vector4d new_q;
  Eigen::Vector3d new_v, new_p;
  if (state->_options.integration_method == StateOptions::IntegrationMethod::ANALYTICAL) {
    predict_mean_analytic(state, dt, w_hat_avg, a_hat_avg, new_q, new_v, new_p, Xi_sum);
  } else if (state->_options.integration_method == StateOptions::IntegrationMethod::RK4) {
    predict_mean_rk4(state, dt, w_hat1, a_hat1, w_hat2, a_hat2, new_q, new_v, new_p);
  } else {
    predict_mean_discrete(state, dt, w_hat_avg, a_hat_avg, new_q, new_v, new_p);
  }

  // Allocate state transition and continuous-time noise Jacobian
  F = Eigen::MatrixXd::Zero(state->imu_intrinsic_size() + 15, state->imu_intrinsic_size() + 15);
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(state->imu_intrinsic_size() + 15, 12);
  if (state->_options.integration_method == StateOptions::IntegrationMethod::RK4 ||
      state->_options.integration_method == StateOptions::IntegrationMethod::ANALYTICAL) {
    compute_F_and_G_analytic(state, dt, w_hat_avg, a_hat_avg, w_uncorrected, a_uncorrected, new_q, new_v, new_p, Xi_sum, F, G);
  } else {
    compute_F_and_G_discrete(state, dt, w_hat_avg, a_hat_avg, w_uncorrected, a_uncorrected, new_q, new_v, new_p, F, G);
  }

  // DISCRETE retains its interval-constant measurement draw and endpoint bias
  // random-walk approximation. ANALYTICAL/RK4 use the continuous error dynamics:
  // G (Qc/dt) G^T only integrates the average draw, losing within-interval noise
  // (even stationary position variance is dt^3/4 instead of dt^3/3). It also
  // misses navigation/bias random-walk cross-covariance. Exposure-time splitting
  // must not introduce that discrepancy into the owned physical-pose model.
  Qd = Eigen::MatrixXd::Zero(state->imu_intrinsic_size() + 15, state->imu_intrinsic_size() + 15);
  if (state->_options.integration_method == StateOptions::IntegrationMethod::DISCRETE) {
    Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
    Qc.block<3, 3>(0, 0) = _noises.sigma_w_2 / dt * Eigen::Matrix3d::Identity();
    Qc.block<3, 3>(3, 3) = _noises.sigma_a_2 / dt * Eigen::Matrix3d::Identity();
    Qc.block<3, 3>(6, 6) = _noises.sigma_wb_2 / dt * Eigen::Matrix3d::Identity();
    Qc.block<3, 3>(9, 9) = _noises.sigma_ab_2 / dt * Eigen::Matrix3d::Identity();
    Qd.noalias() = G * Qc * G.transpose();
    Qd = (0.5 * (Qd + Qd.transpose())).eval();
  } else {
    Qd.topLeftCorner<15, 15>() = compute_Qd_analytic(state, dt, w_hat_avg, a_hat_avg, new_q, Xi_sum);
  }

  // Now replace imu estimate and fej with propagated values
  Eigen::Matrix<double, 16, 1> imu_x = state->_imu->value();
  imu_x.block(0, 0, 4, 1) = new_q;
  imu_x.block(4, 0, 3, 1) = new_p;
  imu_x.block(7, 0, 3, 1) = new_v;
  state->_imu->set_value(imu_x);
  state->_imu->set_fej(imu_x);
}

Eigen::Matrix<double, 15, 15> Propagator::compute_Qd_analytic(
    std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
    const Eigen::Vector4d &new_q, const Eigen::Matrix<double, 3, 18> &Xi_sum) {
  return compute_Qd_analytic(state, dt, w_hat, a_hat, new_q, Xi_sum, true);
}

Eigen::Matrix<double, 15, 15> Propagator::compute_Qd_analytic(
    std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
    const Eigen::Vector4d &new_q, const Eigen::Matrix<double, 3, 18> &Xi_sum, bool include_sensor_noise) {
  const Eigen::Matrix3d A = state->_calib_imu_ACCtoIMU->Rot() * State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  const Eigen::Matrix3d W = state->_calib_imu_GYROtoIMU->Rot() * State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  const Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());
  const Eigen::Matrix3d WTg = W * Tg;
  const Eigen::Matrix3d R_k = state->_options.do_fej ? state->_imu->Rot_fej() : state->_imu->Rot();
  const Eigen::Matrix3d R_end_transpose = R_k.transpose() * Xi_sum.block<3, 3>(0, 0).transpose();
  // Analytic G uses dR = R(new_q) R_k^T, also when RK4 mean integration or
  // separated FEJ/current values differ from exp(-omega dt) R_k. Apply exactly
  // that output attitude chart to every impulse, including bias random walks.
  const Eigen::Matrix3d attitude_chart = quat_2_Rot(new_q) * R_end_transpose;

  // Nodes/weights on [0,1]. Six nodes integrate degree <= 11 exactly; the
  // zero-rate impulse has degree <= 3 even for gyro-bias -> position coupling.
  constexpr double nodes[6] = {0.0337652428984239861, 0.169395306766867743, 0.380690406958401546,
                                0.619309593041598454, 0.830604693233132257, 0.966234757101576014};
  constexpr double weights[6] = {0.0856622461895851725, 0.180380786524069304, 0.233956967286345524,
                                  0.233956967286345524, 0.180380786524069304, 0.0856622461895851725};
  Eigen::Matrix<double, 15, 15> Q = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 12, 1> density;
  density << Eigen::Vector3d::Constant(_noises.sigma_w_2), Eigen::Vector3d::Constant(_noises.sigma_a_2),
      Eigen::Vector3d::Constant(_noises.sigma_wb_2), Eigen::Vector3d::Constant(_noises.sigma_ab_2);
  if (!include_sensor_noise) density.head<6>().setZero();
  for (int node = 0; node < 6; ++node) {
    const double h = dt * nodes[node]; // remaining time after the noise impulse
    Eigen::Matrix<double, 3, 18> xi;
    compute_Xi_sum(state, h, w_hat, a_hat, xi);
    const Eigen::Matrix3d Rh = xi.block<3, 3>(0, 0);
    const Eigen::Matrix3d X1 = xi.block<3, 3>(0, 3), X2 = xi.block<3, 3>(0, 6);
    const Eigen::Matrix3d X3 = xi.block<3, 3>(0, 12), X4 = xi.block<3, 3>(0, 15);
    const Eigen::Matrix3d R_impulse_transpose = R_end_transpose * Rh;
    const Eigen::Matrix3d S1 = skew_x(X1 * a_hat), S2 = skew_x(X2 * a_hat);
    Eigen::Matrix<double, 15, 12> B = Eigen::Matrix<double, 15, 12>::Zero();

    // Instantaneous raw measurement noise, transported to the interval end.
    B.block<3, 3>(0, 0) = -attitude_chart * Rh * W;
    B.block<3, 3>(3, 0) = R_impulse_transpose * S2 * W;
    B.block<3, 3>(6, 0) = R_impulse_transpose * S1 * W;
    B.block<3, 3>(0, 3) = -B.block<3, 3>(0, 0) * Tg * A;
    B.block<3, 3>(3, 3) = -R_impulse_transpose * (h * Eigen::Matrix3d::Identity() + S2 * WTg) * A;
    B.block<3, 3>(6, 3) = -R_impulse_transpose * (Eigen::Matrix3d::Identity() + S1 * WTg) * A;

    // A raw-axis bias impulse persists for h; these are the analytic bias
    // transition columns for that remaining interval, not an endpoint kick.
    // X1 is the integrated rotation, algebraically Jr(-omega*h)*h. Use it
    // directly: the transition's inherited Jr=I tiny-angle approximation
    // would otherwise discard first-order bias-noise cross terms here.
    B.block<3, 3>(0, 6) = -attitude_chart * Rh * X1 * W;
    B.block<3, 3>(3, 6) = R_impulse_transpose * X4 * W;
    B.block<3, 3>(6, 6) = R_impulse_transpose * X3 * W;
    B.block<3, 3>(9, 6).setIdentity();
    B.block<3, 3>(0, 9) = -B.block<3, 3>(0, 6) * Tg * A;
    B.block<3, 3>(3, 9) = -R_impulse_transpose * (X2 + X4 * WTg) * A;
    B.block<3, 3>(6, 9) = -R_impulse_transpose * (X1 + X3 * WTg) * A;
    B.block<3, 3>(12, 9).setIdentity();
    // Explicit fixed-size rank-one products keep the positive accumulation
    // independent of Eigen's general GEMM scratch-allocation policy.
    for (int column = 0; column < 12; ++column)
      Q.noalias() += (dt * weights[node] * density(column)) * B.col(column) * B.col(column).transpose();
  }
  return (0.5 * (Q + Q.transpose())).eval();
}

void Propagator::predict_mean_discrete(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                       Eigen::Vector4d &new_q, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p) {

  // Pre-compute things
  double w_norm = w_hat.norm();
  Eigen::Matrix4d I_4x4 = Eigen::Matrix4d::Identity();
  Eigen::Matrix3d R_Gtoi = state->_imu->Rot();

  // Orientation: Equation (101) and (103) and of Trawny indirect TR
  Eigen::Matrix<double, 4, 4> bigO;
  if (w_norm > 1e-12) {
    bigO = cos(0.5 * w_norm * dt) * I_4x4 + 1 / w_norm * sin(0.5 * w_norm * dt) * Omega(w_hat);
  } else {
    bigO = I_4x4 + 0.5 * dt * Omega(w_hat);
  }
  new_q = quatnorm(bigO * state->_imu->quat());
  // new_q = rot_2_quat(exp_so3(-w_hat*dt)*R_Gtoi);

  // Velocity: just the acceleration in the local frame, minus global gravity
  new_v = state->_imu->vel() + R_Gtoi.transpose() * a_hat * dt - _gravity * dt;

  // Position: just velocity times dt, with the acceleration integrated twice
  new_p = state->_imu->pos() + state->_imu->vel() * dt + 0.5 * R_Gtoi.transpose() * a_hat * dt * dt - 0.5 * _gravity * dt * dt;
}

void Propagator::predict_mean_rk4(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat1, const Eigen::Vector3d &a_hat1,
                                  const Eigen::Vector3d &w_hat2, const Eigen::Vector3d &a_hat2, Eigen::Vector4d &new_q,
                                  Eigen::Vector3d &new_v, Eigen::Vector3d &new_p) {

  // Pre-compute things
  Eigen::Vector3d w_hat = w_hat1;
  Eigen::Vector3d a_hat = a_hat1;
  Eigen::Vector3d w_alpha = (w_hat2 - w_hat1) / dt;
  Eigen::Vector3d a_jerk = (a_hat2 - a_hat1) / dt;

  // y0 ================
  Eigen::Vector4d q_0 = state->_imu->quat();
  Eigen::Vector3d p_0 = state->_imu->pos();
  Eigen::Vector3d v_0 = state->_imu->vel();

  // k1 ================
  Eigen::Vector4d dq_0 = {0, 0, 0, 1};
  Eigen::Vector4d q0_dot = 0.5 * Omega(w_hat) * dq_0;
  Eigen::Vector3d p0_dot = v_0;
  Eigen::Matrix3d R_Gto0 = quat_2_Rot(quat_multiply(dq_0, q_0));
  Eigen::Vector3d v0_dot = R_Gto0.transpose() * a_hat - _gravity;

  Eigen::Vector4d k1_q = q0_dot * dt;
  Eigen::Vector3d k1_p = p0_dot * dt;
  Eigen::Vector3d k1_v = v0_dot * dt;

  // k2 ================
  w_hat += 0.5 * w_alpha * dt;
  a_hat += 0.5 * a_jerk * dt;

  Eigen::Vector4d dq_1 = quatnorm(dq_0 + 0.5 * k1_q);
  // Eigen::Vector3d p_1 = p_0+0.5*k1_p;
  Eigen::Vector3d v_1 = v_0 + 0.5 * k1_v;

  Eigen::Vector4d q1_dot = 0.5 * Omega(w_hat) * dq_1;
  Eigen::Vector3d p1_dot = v_1;
  Eigen::Matrix3d R_Gto1 = quat_2_Rot(quat_multiply(dq_1, q_0));
  Eigen::Vector3d v1_dot = R_Gto1.transpose() * a_hat - _gravity;

  Eigen::Vector4d k2_q = q1_dot * dt;
  Eigen::Vector3d k2_p = p1_dot * dt;
  Eigen::Vector3d k2_v = v1_dot * dt;

  // k3 ================
  Eigen::Vector4d dq_2 = quatnorm(dq_0 + 0.5 * k2_q);
  // Eigen::Vector3d p_2 = p_0+0.5*k2_p;
  Eigen::Vector3d v_2 = v_0 + 0.5 * k2_v;

  Eigen::Vector4d q2_dot = 0.5 * Omega(w_hat) * dq_2;
  Eigen::Vector3d p2_dot = v_2;
  Eigen::Matrix3d R_Gto2 = quat_2_Rot(quat_multiply(dq_2, q_0));
  Eigen::Vector3d v2_dot = R_Gto2.transpose() * a_hat - _gravity;

  Eigen::Vector4d k3_q = q2_dot * dt;
  Eigen::Vector3d k3_p = p2_dot * dt;
  Eigen::Vector3d k3_v = v2_dot * dt;

  // k4 ================
  w_hat += 0.5 * w_alpha * dt;
  a_hat += 0.5 * a_jerk * dt;

  Eigen::Vector4d dq_3 = quatnorm(dq_0 + k3_q);
  // Eigen::Vector3d p_3 = p_0+k3_p;
  Eigen::Vector3d v_3 = v_0 + k3_v;

  Eigen::Vector4d q3_dot = 0.5 * Omega(w_hat) * dq_3;
  Eigen::Vector3d p3_dot = v_3;
  Eigen::Matrix3d R_Gto3 = quat_2_Rot(quat_multiply(dq_3, q_0));
  Eigen::Vector3d v3_dot = R_Gto3.transpose() * a_hat - _gravity;

  Eigen::Vector4d k4_q = q3_dot * dt;
  Eigen::Vector3d k4_p = p3_dot * dt;
  Eigen::Vector3d k4_v = v3_dot * dt;

  // y+dt ================
  Eigen::Vector4d dq = quatnorm(dq_0 + (1.0 / 6.0) * k1_q + (1.0 / 3.0) * k2_q + (1.0 / 3.0) * k3_q + (1.0 / 6.0) * k4_q);
  new_q = quat_multiply(dq, q_0);
  new_p = p_0 + (1.0 / 6.0) * k1_p + (1.0 / 3.0) * k2_p + (1.0 / 3.0) * k3_p + (1.0 / 6.0) * k4_p;
  new_v = v_0 + (1.0 / 6.0) * k1_v + (1.0 / 3.0) * k2_v + (1.0 / 3.0) * k3_v + (1.0 / 6.0) * k4_v;
}

void Propagator::compute_Xi_sum(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                Eigen::Matrix<double, 3, 18> &Xi_sum) {

  // Decompose our angular velocity into a direction and amount
  double w_norm = w_hat.norm();
  double d_th = w_norm * dt;
  Eigen::Vector3d k_hat = Eigen::Vector3d::Zero();
  if (w_norm > 1e-12) {
    k_hat = w_hat / w_norm;
  }

  // Compute useful identities used throughout
  Eigen::Matrix3d I_3x3 = Eigen::Matrix3d::Identity();
  double d_t2 = std::pow(dt, 2);
  double d_t3 = std::pow(dt, 3);
  double w_norm2 = std::pow(w_norm, 2);
  double w_norm3 = std::pow(w_norm, 3);
  double cos_dth = std::cos(d_th);
  double sin_dth = std::sin(d_th);
  double d_th2 = std::pow(d_th, 2);
  double d_th3 = std::pow(d_th, 3);
  Eigen::Matrix3d sK = ov_core::skew_x(k_hat);
  Eigen::Matrix3d sK2 = sK * sK;
  Eigen::Matrix3d sA = ov_core::skew_x(a_hat);

  // Integration components will be used later
  Eigen::Matrix3d R_ktok1, Xi_1, Xi_2, Jr_ktok1, Xi_3, Xi_4;
  // Same Rodrigues/right-Jacobian formulas and small-angle branches as the
  // shared SO(3) helpers, with fixed-size identities. exp_so3's dynamic identity
  // can allocate a MatrixXd temporary; the noise quadrature calls this six
  // times per interval and must not multiply that heap traffic.
  const Eigen::Vector3d increment = -w_hat * dt;
  const Eigen::Matrix3d increment_skew = skew_x(increment);
  const double theta = increment.norm();
  const double sinc = theta < 1e-7 ? 1. : std::sin(theta) / theta;
  const double cosc = theta < 1e-7 ? 0.5 : (1. - std::cos(theta)) / (theta * theta);
  R_ktok1 = I_3x3 + sinc * increment_skew + cosc * increment_skew * increment_skew;
  if (theta < 1e-6) {
    Jr_ktok1 = I_3x3;
  } else {
    const Eigen::Vector3d axis = -increment / theta;
    const double sin_over_theta = std::sin(theta) / theta;
    Jr_ktok1 = sin_over_theta * I_3x3 + (1. - sin_over_theta) * axis * axis.transpose() +
                ((1. - std::cos(theta)) / theta) * skew_x(axis);
  }

  // Now begin the integration of each component
  // Based on the delta theta, let's decide which integration will be used
  const bool small_angle = std::abs(d_th) < 0.1;
  if (!small_angle) {

    // first order rotation integration with constant omega
    Xi_1 = I_3x3 * dt + (1.0 - cos_dth) / w_norm * sK + (dt - sin_dth / w_norm) * sK2;

    // second order rotation integration with constant omega
    Xi_2 = 1.0 / 2 * d_t2 * I_3x3 + (d_th - sin_dth) / w_norm2 * sK + (1.0 / 2 * d_t2 - (1.0 - cos_dth) / w_norm2) * sK2;

    // first order integration with constant omega and constant acc
    Xi_3 = 1.0 / 2 * d_t2 * sA + (sin_dth - d_th) / w_norm2 * sA * sK + (sin_dth - d_th * cos_dth) / w_norm2 * sK * sA +
           (1.0 / 2 * d_t2 - (1.0 - cos_dth) / w_norm2) * sA * sK2 +
           (1.0 / 2 * d_t2 + (1.0 - cos_dth - d_th * sin_dth) / w_norm2) * (sK2 * sA + k_hat.dot(a_hat) * sK) -
           (3 * sin_dth - 2 * d_th - d_th * cos_dth) / w_norm2 * k_hat.dot(a_hat) * sK2;

    // second order integration with constant omega and constant acc
    Xi_4 = 1.0 / 6 * d_t3 * sA + (2 * (1.0 - cos_dth) - d_th2) / (2 * w_norm3) * sA * sK +
           ((2 * (1.0 - cos_dth) - d_th * sin_dth) / w_norm3) * sK * sA + ((sin_dth - d_th) / w_norm3 + d_t3 / 6) * sA * sK2 +
           ((d_th - 2 * sin_dth + 1.0 / 6 * d_th3 + d_th * cos_dth) / w_norm3) * (sK2 * sA + k_hat.dot(a_hat) * sK) +
           (4 * cos_dth - 4 + d_th2 + d_th * sin_dth) / w_norm3 * k_hat.dot(a_hat) * sK2;

  } else {

    // Integrate the exponential series, rather than approximating its integral
    // by the endpoint rotation. Differentiate the SAME series for Xi_3/4 so the
    // zero-rate limit and its bias derivatives agree. The fixed eight terms
    // avoid cancellation in the closed forms at small |omega|*dt, without any
    // dynamic allocation or convergence loop on the propagation path.
    const Eigen::Matrix3d M = ov_core::skew_x(w_hat * dt);
    Eigen::Matrix3d power = I_3x3, derivative = Eigen::Matrix3d::Zero();
    Eigen::Vector3d powered_a = a_hat;
    double c1 = dt, c2 = 0.5 * d_t2;
    Xi_1 = c1 * I_3x3;
    Xi_2 = c2 * I_3x3;
    Xi_3.setZero();
    Xi_4.setZero();
    for (int n = 1; n <= 8; ++n) {
      derivative = (-dt * ov_core::skew_x(powered_a) + M * derivative).eval();
      powered_a = (M * powered_a).eval();
      power = (M * power).eval();
      c1 /= n + 1;
      c2 /= n + 2;
      Xi_1 += c1 * power;
      Xi_2 += c2 * power;
      Xi_3 -= c1 * derivative;
      Xi_4 -= c2 * derivative;
    }
  }

  // Store the integrated parameters
  Xi_sum.setZero();
  Xi_sum.block(0, 0, 3, 3) = R_ktok1;
  Xi_sum.block(0, 3, 3, 3) = Xi_1;
  Xi_sum.block(0, 6, 3, 3) = Xi_2;
  Xi_sum.block(0, 9, 3, 3) = Jr_ktok1;
  Xi_sum.block(0, 12, 3, 3) = Xi_3;
  Xi_sum.block(0, 15, 3, 3) = Xi_4;
}

void Propagator::predict_mean_analytic(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                       Eigen::Vector4d &new_q, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p,
                                       Eigen::Matrix<double, 3, 18> &Xi_sum) {

  // Pre-compute things
  Eigen::Matrix3d R_Gtok = state->_imu->Rot();
  Eigen::Vector4d q_ktok1 = ov_core::rot_2_quat(Xi_sum.block(0, 0, 3, 3));
  Eigen::Matrix3d Xi_1 = Xi_sum.block(0, 3, 3, 3);
  Eigen::Matrix3d Xi_2 = Xi_sum.block(0, 6, 3, 3);

  // Use our integrated Xi's to move the state forward
  new_q = ov_core::quat_multiply(q_ktok1, state->_imu->quat());
  new_v = state->_imu->vel() + R_Gtok.transpose() * Xi_1 * a_hat - _gravity * dt;
  new_p = state->_imu->pos() + state->_imu->vel() * dt + R_Gtok.transpose() * Xi_2 * a_hat - 0.5 * _gravity * dt * dt;
}

void Propagator::compute_F_and_G_analytic(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat,
                                          const Eigen::Vector3d &a_hat, const Eigen::Vector3d &w_uncorrected,
                                          const Eigen::Vector3d &a_uncorrected, const Eigen::Vector4d &new_q, const Eigen::Vector3d &new_v,
                                          const Eigen::Vector3d &new_p, const Eigen::Matrix<double, 3, 18> &Xi_sum, Eigen::MatrixXd &F,
                                          Eigen::MatrixXd &G) {

  // Get the locations of each entry of the imu state
  int local_size = 0;
  int th_id = local_size;
  local_size += state->_imu->q()->size();
  int p_id = local_size;
  local_size += state->_imu->p()->size();
  int v_id = local_size;
  local_size += state->_imu->v()->size();
  int bg_id = local_size;
  local_size += state->_imu->bg()->size();
  int ba_id = local_size;
  local_size += state->_imu->ba()->size();

  // If we are doing calibration, we can define their "local" id in the state transition
  int Dw_id = -1;
  int Da_id = -1;
  int Tg_id = -1;
  int th_atoI_id = -1;
  int th_wtoI_id = -1;
  if (state->_options.do_calib_imu_intrinsics) {
    Dw_id = local_size;
    local_size += state->_calib_imu_dw->size();
    Da_id = local_size;
    local_size += state->_calib_imu_da->size();
    if (state->_options.do_calib_imu_g_sensitivity) {
      Tg_id = local_size;
      local_size += state->_calib_imu_tg->size();
    }
    if (state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
      th_wtoI_id = local_size;
      local_size += state->_calib_imu_GYROtoIMU->size();
    } else {
      th_atoI_id = local_size;
      local_size += state->_calib_imu_ACCtoIMU->size();
    }
  }

  // The change in the orientation from the end of the last prop to the current prop
  // This is needed since we need to include the "k-th" updated orientation information
  Eigen::Matrix3d R_k = state->_imu->Rot();
  Eigen::Vector3d v_k = state->_imu->vel();
  Eigen::Vector3d p_k = state->_imu->pos();
  if (state->_options.do_fej) {
    R_k = state->_imu->Rot_fej();
    v_k = state->_imu->vel_fej();
    p_k = state->_imu->pos_fej();
  }
  Eigen::Matrix3d dR_ktok1 = quat_2_Rot(new_q) * R_k.transpose();

  Eigen::Matrix3d Dw = State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  Eigen::Matrix3d Da = State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());
  Eigen::Matrix3d R_atoI = state->_calib_imu_ACCtoIMU->Rot();
  Eigen::Matrix3d R_wtoI = state->_calib_imu_GYROtoIMU->Rot();
  Eigen::Vector3d a_k = R_atoI * Da * a_uncorrected;
  Eigen::Vector3d w_k = R_wtoI * Dw * w_uncorrected; // contains gravity correction already

  Eigen::Matrix3d Xi_1 = Xi_sum.block(0, 3, 3, 3);
  Eigen::Matrix3d Xi_2 = Xi_sum.block(0, 6, 3, 3);
  Eigen::Matrix3d Jr_ktok1 = Xi_sum.block(0, 9, 3, 3);
  Eigen::Matrix3d Xi_3 = Xi_sum.block(0, 12, 3, 3);
  Eigen::Matrix3d Xi_4 = Xi_sum.block(0, 15, 3, 3);

  // for th
  F.block(th_id, th_id, 3, 3) = dR_ktok1;
  F.block(p_id, th_id, 3, 3) = -skew_x(new_p - p_k - v_k * dt + 0.5 * _gravity * dt * dt) * R_k.transpose();
  F.block(v_id, th_id, 3, 3) = -skew_x(new_v - v_k + _gravity * dt) * R_k.transpose();

  // for p
  F.block(p_id, p_id, 3, 3).setIdentity();

  // for v
  F.block(p_id, v_id, 3, 3) = Eigen::Matrix3d::Identity() * dt;
  F.block(v_id, v_id, 3, 3).setIdentity();

  // for bg
  F.block(th_id, bg_id, 3, 3) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw;
  F.block(p_id, bg_id, 3, 3) = R_k.transpose() * Xi_4 * R_wtoI * Dw;
  F.block(v_id, bg_id, 3, 3) = R_k.transpose() * Xi_3 * R_wtoI * Dw;
  F.block(bg_id, bg_id, 3, 3).setIdentity();

  // for ba
  F.block(th_id, ba_id, 3, 3) = dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * R_atoI * Da;
  F.block(p_id, ba_id, 3, 3) = -R_k.transpose() * (Xi_2 + Xi_4 * R_wtoI * Dw * Tg) * R_atoI * Da;
  F.block(v_id, ba_id, 3, 3) = -R_k.transpose() * (Xi_1 + Xi_3 * R_wtoI * Dw * Tg) * R_atoI * Da;
  F.block(ba_id, ba_id, 3, 3).setIdentity();

  // begin to add the state transition matrix for the omega intrinsics Dw part
  if (Dw_id != -1) {
    Eigen::MatrixXd H_Dw = compute_H_Dw(state, w_uncorrected);
    F.block(th_id, Dw_id, 3, state->_calib_imu_dw->size()) = dR_ktok1 * Jr_ktok1 * dt * R_wtoI * H_Dw;
    F.block(p_id, Dw_id, 3, state->_calib_imu_dw->size()) = -R_k.transpose() * Xi_4 * R_wtoI * H_Dw;
    F.block(v_id, Dw_id, 3, state->_calib_imu_dw->size()) = -R_k.transpose() * Xi_3 * R_wtoI * H_Dw;
    F.block(Dw_id, Dw_id, state->_calib_imu_dw->size(), state->_calib_imu_dw->size()).setIdentity();
  }

  // begin to add the state transition matrix for the acc intrinsics Da part
  if (Da_id != -1) {
    Eigen::MatrixXd H_Da = compute_H_Da(state, a_uncorrected);
    F.block(th_id, Da_id, 3, state->_calib_imu_da->size()) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * R_atoI * H_Da;
    F.block(p_id, Da_id, 3, state->_calib_imu_da->size()) = R_k.transpose() * (Xi_2 + Xi_4 * R_wtoI * Dw * Tg) * R_atoI * H_Da;
    F.block(v_id, Da_id, 3, state->_calib_imu_da->size()) = R_k.transpose() * (Xi_1 + Xi_3 * R_wtoI * Dw * Tg) * R_atoI * H_Da;
    F.block(Da_id, Da_id, state->_calib_imu_da->size(), state->_calib_imu_da->size()).setIdentity();
  }

  // add the state transition matrix of the Tg part
  if (Tg_id != -1) {
    Eigen::MatrixXd H_Tg = compute_H_Tg(state, a_k);
    F.block(th_id, Tg_id, 3, state->_calib_imu_tg->size()) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * H_Tg;
    F.block(p_id, Tg_id, 3, state->_calib_imu_tg->size()) = R_k.transpose() * Xi_4 * R_wtoI * Dw * H_Tg;
    F.block(v_id, Tg_id, 3, state->_calib_imu_tg->size()) = R_k.transpose() * Xi_3 * R_wtoI * Dw * H_Tg;
    F.block(Tg_id, Tg_id, state->_calib_imu_tg->size(), state->_calib_imu_tg->size()).setIdentity();
  }

  // begin to add the state transition matrix for the R_ACCtoIMU part
  if (th_atoI_id != -1) {
    F.block(th_id, th_atoI_id, 3, 3) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * ov_core::skew_x(a_k);
    F.block(p_id, th_atoI_id, 3, 3) = R_k.transpose() * (Xi_2 + Xi_4 * R_wtoI * Dw * Tg) * ov_core::skew_x(a_k);
    F.block(v_id, th_atoI_id, 3, 3) = R_k.transpose() * (Xi_1 + Xi_3 * R_wtoI * Dw * Tg) * ov_core::skew_x(a_k);
    F.block(th_atoI_id, th_atoI_id, 3, 3).setIdentity();
  }

  // begin to add the state transition matrix for the R_GYROtoIMU part
  if (th_wtoI_id != -1) {
    F.block(th_id, th_wtoI_id, 3, 3) = dR_ktok1 * Jr_ktok1 * dt * ov_core::skew_x(w_k);
    F.block(p_id, th_wtoI_id, 3, 3) = -R_k.transpose() * Xi_4 * ov_core::skew_x(w_k);
    F.block(v_id, th_wtoI_id, 3, 3) = -R_k.transpose() * Xi_3 * ov_core::skew_x(w_k);
    F.block(th_wtoI_id, th_wtoI_id, 3, 3).setIdentity();
  }

  // construct the G part
  G.block(th_id, 0, 3, 3) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw;
  G.block(p_id, 0, 3, 3) = R_k.transpose() * Xi_4 * R_wtoI * Dw;
  G.block(v_id, 0, 3, 3) = R_k.transpose() * Xi_3 * R_wtoI * Dw;
  G.block(th_id, 3, 3, 3) = dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * R_atoI * Da;
  G.block(p_id, 3, 3, 3) = -R_k.transpose() * (Xi_2 + Xi_4 * R_wtoI * Dw * Tg) * R_atoI * Da;
  G.block(v_id, 3, 3, 3) = -R_k.transpose() * (Xi_1 + Xi_3 * R_wtoI * Dw * Tg) * R_atoI * Da;
  G.block(bg_id, 6, 3, 3) = dt * Eigen::Matrix3d::Identity();
  G.block(ba_id, 9, 3, 3) = dt * Eigen::Matrix3d::Identity();
}

void Propagator::compute_F_and_G_discrete(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat,
                                          const Eigen::Vector3d &a_hat, const Eigen::Vector3d &w_uncorrected,
                                          const Eigen::Vector3d &a_uncorrected, const Eigen::Vector4d &new_q, const Eigen::Vector3d &new_v,
                                          const Eigen::Vector3d &new_p, Eigen::MatrixXd &F, Eigen::MatrixXd &G) {

  // Get the locations of each entry of the imu state
  int local_size = 0;
  int th_id = local_size;
  local_size += state->_imu->q()->size();
  int p_id = local_size;
  local_size += state->_imu->p()->size();
  int v_id = local_size;
  local_size += state->_imu->v()->size();
  int bg_id = local_size;
  local_size += state->_imu->bg()->size();
  int ba_id = local_size;
  local_size += state->_imu->ba()->size();

  // If we are doing calibration, we can define their "local" id in the state transition
  int Dw_id = -1;
  int Da_id = -1;
  int Tg_id = -1;
  int th_atoI_id = -1;
  int th_wtoI_id = -1;
  if (state->_options.do_calib_imu_intrinsics) {
    Dw_id = local_size;
    local_size += state->_calib_imu_dw->size();
    Da_id = local_size;
    local_size += state->_calib_imu_da->size();
    if (state->_options.do_calib_imu_g_sensitivity) {
      Tg_id = local_size;
      local_size += state->_calib_imu_tg->size();
    }
    if (state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
      th_wtoI_id = local_size;
      local_size += state->_calib_imu_GYROtoIMU->size();
    } else {
      th_atoI_id = local_size;
      local_size += state->_calib_imu_ACCtoIMU->size();
    }
  }

  // This is the change in the orientation from the end of the last prop to the current prop
  // This is needed since we need to include the "k-th" updated orientation information
  Eigen::Matrix3d R_k = state->_imu->Rot();
  Eigen::Vector3d v_k = state->_imu->vel();
  Eigen::Vector3d p_k = state->_imu->pos();
  if (state->_options.do_fej) {
    R_k = state->_imu->Rot_fej();
    v_k = state->_imu->vel_fej();
    p_k = state->_imu->pos_fej();
  }
  Eigen::Matrix3d dR_ktok1 = quat_2_Rot(new_q) * R_k.transpose();

  // This is the change in the orientation from the end of the last prop to the current prop
  // This is needed since we need to include the "k-th" updated orientation information
  Eigen::Matrix3d Dw = State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  Eigen::Matrix3d Da = State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());
  Eigen::Matrix3d R_atoI = state->_calib_imu_ACCtoIMU->Rot();
  Eigen::Matrix3d R_wtoI = state->_calib_imu_GYROtoIMU->Rot();
  Eigen::Vector3d a_k = R_atoI * Da * a_uncorrected;
  Eigen::Vector3d w_k = R_wtoI * Dw * w_uncorrected; // contains gravity correction already
  Eigen::Matrix3d Jr_ktok1 = Jr_so3(log_so3(dR_ktok1));

  // for theta
  F.block(th_id, th_id, 3, 3) = dR_ktok1;
  // F.block(th_id, bg_id, 3, 3) = -dR_ktok1 * Jr_so3(w_hat * dt) * dt * R_wtoI_fej * Dw_fej;
  F.block(th_id, bg_id, 3, 3) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw;
  F.block(th_id, ba_id, 3, 3) = dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * R_atoI * Da;

  // for position
  F.block(p_id, th_id, 3, 3) = -skew_x(new_p - p_k - v_k * dt + 0.5 * _gravity * dt * dt) * R_k.transpose();
  F.block(p_id, p_id, 3, 3).setIdentity();
  F.block(p_id, v_id, 3, 3) = Eigen::Matrix3d::Identity() * dt;
  F.block(p_id, ba_id, 3, 3) = -0.5 * R_k.transpose() * dt * dt * R_atoI * Da;

  // for velocity
  F.block(v_id, th_id, 3, 3) = -skew_x(new_v - v_k + _gravity * dt) * R_k.transpose();
  F.block(v_id, v_id, 3, 3).setIdentity();
  F.block(v_id, ba_id, 3, 3) = -R_k.transpose() * dt * R_atoI * Da;

  // for bg
  F.block(bg_id, bg_id, 3, 3).setIdentity();

  // for ba
  F.block(ba_id, ba_id, 3, 3).setIdentity();

  // begin to add the state transition matrix for the omega intrinsics Dw part
  if (Dw_id != -1) {
    Eigen::MatrixXd H_Dw = compute_H_Dw(state, w_uncorrected);
    F.block(th_id, Dw_id, 3, state->_calib_imu_dw->size()) = dR_ktok1 * Jr_ktok1 * dt * R_wtoI * H_Dw;
    F.block(Dw_id, Dw_id, state->_calib_imu_dw->size(), state->_calib_imu_dw->size()).setIdentity();
  }

  // begin to add the state transition matrix for the acc intrinsics Da part
  if (Da_id != -1) {
    Eigen::MatrixXd H_Da = compute_H_Da(state, a_uncorrected);
    F.block(th_id, Da_id, 3, state->_calib_imu_da->size()) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * R_atoI * H_Da;
    F.block(p_id, Da_id, 3, state->_calib_imu_da->size()) = 0.5 * R_k.transpose() * dt * dt * R_atoI * H_Da;
    F.block(v_id, Da_id, 3, state->_calib_imu_da->size()) = R_k.transpose() * dt * R_atoI * H_Da;
    F.block(Da_id, Da_id, state->_calib_imu_da->size(), state->_calib_imu_da->size()).setIdentity();
  }

  // begin to add the state transition matrix for the gravity sensitivity Tg part
  if (Tg_id != -1) {
    Eigen::MatrixXd H_Tg = compute_H_Tg(state, a_k);
    F.block(th_id, Tg_id, 3, state->_calib_imu_tg->size()) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * H_Tg;
    F.block(Tg_id, Tg_id, state->_calib_imu_tg->size(), state->_calib_imu_tg->size()).setIdentity();
  }

  // begin to add the state transition matrix for the R_ACCtoIMU part
  if (th_atoI_id != -1) {
    F.block(th_id, th_atoI_id, 3, 3) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * ov_core::skew_x(a_k);
    F.block(p_id, th_atoI_id, 3, 3) = 0.5 * R_k.transpose() * dt * dt * ov_core::skew_x(a_k);
    F.block(v_id, th_atoI_id, 3, 3) = R_k.transpose() * dt * ov_core::skew_x(a_k);
    F.block(th_atoI_id, th_atoI_id, 3, 3).setIdentity();
  }

  // begin to add the state transition matrix for the R_GYROtoIMU part
  if (th_wtoI_id != -1) {
    F.block(th_id, th_wtoI_id, 3, 3) = dR_ktok1 * Jr_ktok1 * dt * ov_core::skew_x(w_k);
    F.block(th_wtoI_id, th_wtoI_id, 3, 3).setIdentity();
  }

  // Noise jacobian
  G.block(th_id, 0, 3, 3) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw;
  G.block(th_id, 3, 3, 3) = dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * R_atoI * Da;
  G.block(v_id, 3, 3, 3) = -R_k.transpose() * dt * R_atoI * Da;
  G.block(p_id, 3, 3, 3) = -0.5 * R_k.transpose() * dt * dt * R_atoI * Da;
  G.block(bg_id, 6, 3, 3) = dt * Eigen::Matrix3d::Identity();
  G.block(ba_id, 9, 3, 3) = dt * Eigen::Matrix3d::Identity();
}

Eigen::MatrixXd Propagator::compute_H_Dw(std::shared_ptr<State> state, const Eigen::Vector3d &w_uncorrected) {

  Eigen::Matrix3d I_3x3 = Eigen::MatrixXd::Identity(3, 3);
  Eigen::Vector3d e_1 = I_3x3.block(0, 0, 3, 1);
  Eigen::Vector3d e_2 = I_3x3.block(0, 1, 3, 1);
  Eigen::Vector3d e_3 = I_3x3.block(0, 2, 3, 1);
  double w_1 = w_uncorrected(0);
  double w_2 = w_uncorrected(1);
  double w_3 = w_uncorrected(2);
  assert(state->_options.do_calib_imu_intrinsics);

  Eigen::MatrixXd H_Dw = Eigen::MatrixXd::Zero(3, 6);
  if (state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    H_Dw << w_1 * I_3x3, w_2 * e_2, w_2 * e_3, w_3 * e_3;
  } else {
    H_Dw << w_1 * e_1, w_2 * e_1, w_2 * e_2, w_3 * I_3x3;
  }
  return H_Dw;
}

Eigen::MatrixXd Propagator::compute_H_Da(std::shared_ptr<State> state, const Eigen::Vector3d &a_uncorrected) {

  Eigen::Matrix3d I_3x3 = Eigen::MatrixXd::Identity(3, 3);
  Eigen::Vector3d e_1 = I_3x3.block(0, 0, 3, 1);
  Eigen::Vector3d e_2 = I_3x3.block(0, 1, 3, 1);
  Eigen::Vector3d e_3 = I_3x3.block(0, 2, 3, 1);
  double a_1 = a_uncorrected(0);
  double a_2 = a_uncorrected(1);
  double a_3 = a_uncorrected(2);
  assert(state->_options.do_calib_imu_intrinsics);

  Eigen::MatrixXd H_Da = Eigen::MatrixXd::Zero(3, 6);
  if (state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    H_Da << a_1 * I_3x3, a_2 * e_2, a_2 * e_3, a_3 * e_3;
  } else {
    H_Da << a_1 * e_1, a_2 * e_1, a_2 * e_2, a_3 * I_3x3;
  }
  return H_Da;
}

Eigen::MatrixXd Propagator::compute_H_Tg(std::shared_ptr<State> state, const Eigen::Vector3d &a_inI) {

  Eigen::Matrix3d I_3x3 = Eigen::MatrixXd::Identity(3, 3);
  double a_1 = a_inI(0);
  double a_2 = a_inI(1);
  double a_3 = a_inI(2);
  assert(state->_options.do_calib_imu_intrinsics);
  assert(state->_options.do_calib_imu_g_sensitivity);

  Eigen::MatrixXd H_Tg = Eigen::MatrixXd::Zero(3, 9);
  H_Tg << a_1 * I_3x3, a_2 * I_3x3, a_3 * I_3x3;
  return H_Tg;
}

void Propagator::feed_imu_batch(const std::vector<ov_core::ImuData>& messages, double oldest_time) {
    if (messages.empty()) return;

    // Insert all measurements at once with a single lock
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    imu_data.reserve(imu_data.size() + messages.size()); // Pre-allocate space
    for (const auto& msg : messages) {
        imu_data.emplace_back(msg);
    }

    // Clean old measurements if needed
    if (oldest_time != -1) {
        clean_old_imu_measurements(oldest_time - _prop_window);
    }
}

bool Propagator::sampled_state_at_endpoint(std::shared_ptr<State> state, double imu_time, SampledStateOutput &out,
                                          Eigen::MatrixXd *state_cross) const {
  if (!state || !state->has_sampled_imu_boundary() || !state->_imu_endpoint_valid || !state->_imu ||
      !finite_timestamp(imu_time) || initializer_time_bits(imu_time) != initializer_time_bits(state->imu_endpoint()) ||
      state->_options.do_fej || state->_options.do_calib_imu_intrinsics || state->_options.do_calib_imu_g_sensitivity ||
      state->imu_intrinsic_size() != 0 ||
      (state_cross && (state_cross->rows() != state->max_covariance_size() || state_cross->cols() != 12)) ||
      !finite_coefficients(state->_imu->value()) || std::abs(state->_imu->quat().squaredNorm() - 1.) > 1e-9) return false;

  const auto &slots = state->sampled_imu_slots();
  std::array<int, 2> active{-1, -1}, selected{-1, -1};
  int active_count = 0, selected_count = 0;
  for (size_t i = 0; i < slots.size(); ++i) {
    if (!slots[i].active) continue;
    if (!StateHelper::valid_sampled_imu_record(slots[i].record) || !slots[i].noise ||
        slots[i].noise->value().rows() != 6 || slots[i].noise->value().cols() != 1 ||
        !finite_coefficients(slots[i].noise->value())) return false;
    active[active_count++] = static_cast<int>(i);
  }
  if (!active_count) return false;
  if (active_count == 2) {
    if (slots[active[1]].record.timestamp < slots[active[0]].record.timestamp) std::swap(active[0], active[1]);
    const auto &left = slots[active[0]].record, &right = slots[active[1]].record;
    if (left.stream_episode != right.stream_episode || !(left.timestamp < right.timestamp) || !(left.sequence < right.sequence))
      return false;
  }
  Eigen::Vector2d weights = Eigen::Vector2d::Zero();
  for (int i = 0; i < active_count; ++i) {
    if (initializer_time_bits(slots[active[i]].record.timestamp) == initializer_time_bits(imu_time)) {
      selected[0] = active[i]; selected_count = 1; weights(0) = 1.;
    }
  }
  if (!selected_count) {
    if (active_count != 2 || !(slots[active[0]].record.timestamp < imu_time) || !(imu_time < slots[active[1]].record.timestamp) ||
        !sampled_weights(slots[active[0]].record, slots[active[1]].record, imu_time, weights)) return false;
    selected = active; selected_count = 2;
  }

  const Eigen::Matrix3d A = state->_calib_imu_ACCtoIMU->Rot() * State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  const Eigen::Matrix3d W = state->_calib_imu_GYROtoIMU->Rot() * State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  const Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());
  if (!finite_coefficients(A) || !finite_coefficients(W) || !finite_coefficients(Tg)) return false;
  const Eigen::FullPivLU<Eigen::Matrix3d> accel_factor(A), gyro_factor(W);
  if (!accel_factor.isInvertible() || !(A.determinant() > 0.) || !(accel_factor.rcond() > 1e-12) ||
      !gyro_factor.isInvertible() || !(W.determinant() > 0.) || !(gyro_factor.rcond() > 1e-12)) return false;
  const Eigen::Matrix3d WTA = W * Tg * A;
  Eigen::Matrix<double, 6, 1> raw = Eigen::Matrix<double, 6, 1>::Zero();
  SampledStateOutput staged;
  staged.imu_time = imu_time; staged.support_count = selected_count; staged.weights = weights;
  staged.stream_episode = slots[selected[0]].record.stream_episode;
  Eigen::Matrix<double, 12, 27> H = Eigen::Matrix<double, 12, 27>::Zero();
  for (int i = 0; i < selected_count; ++i) {
    const int slot = selected[i];
    const auto &record = slots[slot].record;
    raw += weights(i) * (record.measured - slots[slot].noise->value());
    staged.support[i].sequence = record.sequence; staged.support[i].timestamp = record.timestamp;
    H.block<3, 3>(9, 15 + 6 * slot) = -weights(i) * W;
    H.block<3, 3>(9, 18 + 6 * slot) = weights(i) * WTA;
  }
  staged.mean.head<4>() = state->_imu->quat();
  staged.mean.segment<3>(4) = state->_imu->pos();
  staged.mean.segment<3>(7) = state->_imu->Rot() * state->_imu->vel();
  staged.mean.tail<3>() = W * (raw.head<3>() - state->_imu->bias_g() - Tg * A * (raw.tail<3>() - state->_imu->bias_a()));
  H.topLeftCorner<6, 6>().setIdentity();
  H.block<3, 3>(6, 0) = skew_x(staged.mean.segment<3>(7));
  H.block<3, 3>(6, 6) = state->_imu->Rot();
  H.block<3, 3>(9, 9) = -W;
  H.block<3, 3>(9, 12) = WTA;
  if (!finite_coefficients(staged.mean) || !StateHelper::project_sampled_imu_output(state, H, staged.covariance, state_cross))
    return false;
  out = staged;
  return true;
}
