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

#include "StateHelper.h"

#include "state/State.h"

#include "cam/CamBase.h"
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "init/PhysicalResetWindow.h"
#include "types/IMU.h"
#include "types/JPLQuat.h"
#include "types/Landmark.h"
#include "types/PoseJPL.h"
#include "types/Vec.h"
#include "utils/colors.h"
#include "utils/innovation.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include "utils/chi_square/chi_squared_quantile_table_0_95.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

std::shared_ptr<State> StateHelper::clone_state(std::shared_ptr<State> state) {

  std::lock_guard<std::mutex> lock(state->_mutex_state);

  // A fresh State builds default IMU/calib objects + a _Cov sized to the active calib set; we
  // overwrite _variables, _Cov and every map below, so those defaults are simply released.
  auto out = std::make_shared<State>(state->_options);

  // 1) Clone every covariance variable, preserving its local id (== its block offset in _Cov).
  //    Type::clone() copies value + fej but NOT the id (its contract), so we re-stamp the id.
  //    Landmark/IMU/PoseJPL/JPLQuat/Vec all override clone() to preserve their dynamic type.
  //    Build an old-pointer -> new-clone map so every state map can be rewired to the clones.
  std::map<ov_type::Type *, std::shared_ptr<ov_type::Type>> ptr_map;
  std::vector<std::shared_ptr<ov_type::Type>> new_vars;
  new_vars.reserve(state->_variables.size());
  for (const auto &var : state->_variables) {
    std::shared_ptr<ov_type::Type> nv = var->clone();
    nv->set_local_id(var->id());
    ptr_map[var.get()] = nv;
    new_vars.push_back(nv);
  }

  // Resolve an old pointer to its clone. Variables in _variables resolve to the exact object now
  // in out->_variables (so the maps and the covariance layout share identity). Pointers NOT in
  // _variables are inactive/fixed calibration (id == -1, not in _Cov): clone once and cache, so
  // any aliasing (e.g. _calib_dt_CAMtoIMU into _calib_dt_CAMtoIMU_map) still shares one object.
  auto resolve = [&](const std::shared_ptr<ov_type::Type> &old) -> std::shared_ptr<ov_type::Type> {
    if (old == nullptr)
      return nullptr;
    auto it = ptr_map.find(old.get());
    if (it != ptr_map.end())
      return it->second;
    std::shared_ptr<ov_type::Type> nv = old->clone();
    nv->set_local_id(old->id());
    ptr_map[old.get()] = nv;
    return nv;
  };

  out->_variables = new_vars;
  out->_Cov = state->_Cov; // ids preserved above, so the block layout is identical -- verbatim copy

  // 2) Scalars + non-covariance metadata (all value types, deep-copy by assignment)
  out->_timestamp = state->_timestamp;
  out->_imu_endpoint = state->_imu_endpoint;
  out->_imu_endpoint_valid = state->_imu_endpoint_valid;
  out->_initialization_episode_id = state->_initialization_episode_id;
  out->_sampled_imu_stream_episode = state->_sampled_imu_stream_episode;
  out->_sampled_imu_last_sequence = state->_sampled_imu_last_sequence;
  out->_sampled_imu_last_timestamp = state->_sampled_imu_last_timestamp;
  out->_sampled_imu_slots = state->_sampled_imu_slots;
  for (auto &slot : out->_sampled_imu_slots)
    slot.noise = std::dynamic_pointer_cast<Vec>(resolve(slot.noise));
  out->_options = state->_options;
  out->_kin_miss_count = state->_kin_miss_count;
  out->_clones_kinematics = state->_clones_kinematics;
  out->_epoch_residuals = state->_epoch_residuals;
  out->_epoch_bridges = state->_epoch_bridges;

  // 3) Rewire every Type-holding map/pointer to the clones
  out->_imu = std::dynamic_pointer_cast<IMU>(resolve(state->_imu));

  out->_clones_IMU.clear();
  for (const auto &kv : state->_clones_IMU)
    out->_clones_IMU[kv.first] = std::dynamic_pointer_cast<PoseJPL>(resolve(kv.second));

  out->_exposure_poses = state->_exposure_poses;
  for (auto &view : out->_exposure_poses)
    view.pose = std::dynamic_pointer_cast<PoseJPL>(resolve(view.pose));

  out->_features_SLAM.clear();
  for (const auto &kv : state->_features_SLAM)
    out->_features_SLAM[kv.first] = std::dynamic_pointer_cast<Landmark>(resolve(kv.second));

  out->_calib_dt_CAMtoIMU_map.clear();
  for (const auto &kv : state->_calib_dt_CAMtoIMU_map)
    out->_calib_dt_CAMtoIMU_map[kv.first] = std::dynamic_pointer_cast<Vec>(resolve(kv.second));
  // alias into the map: resolves to the same clone as the ref camera's map entry (cached above)
  out->_calib_dt_CAMtoIMU = std::dynamic_pointer_cast<Vec>(resolve(state->_calib_dt_CAMtoIMU));

  out->_calib_camera_readout.clear();
  for (const auto &kv : state->_calib_camera_readout)
    out->_calib_camera_readout[kv.first] = std::dynamic_pointer_cast<Vec>(resolve(kv.second));

  out->_calib_IMUtoCAM.clear();
  for (const auto &kv : state->_calib_IMUtoCAM)
    out->_calib_IMUtoCAM[kv.first] = std::dynamic_pointer_cast<PoseJPL>(resolve(kv.second));

  out->_cam_intrinsics.clear();
  for (const auto &kv : state->_cam_intrinsics)
    out->_cam_intrinsics[kv.first] = std::dynamic_pointer_cast<Vec>(resolve(kv.second));

  // Camera intrinsic objects are NOT Types / not in covariance: deep-copy via CamBase::clone()
  // (online intrinsic calib mutates them, so the snapshot must own independent copies)
  out->_cam_intrinsics_cameras.clear();
  for (const auto &kv : state->_cam_intrinsics_cameras)
    out->_cam_intrinsics_cameras[kv.first] = kv.second ? kv.second->clone() : nullptr;

  out->_calib_imu_dw = std::dynamic_pointer_cast<Vec>(resolve(state->_calib_imu_dw));
  out->_calib_imu_da = std::dynamic_pointer_cast<Vec>(resolve(state->_calib_imu_da));
  out->_calib_imu_tg = std::dynamic_pointer_cast<Vec>(resolve(state->_calib_imu_tg));
  out->_calib_imu_GYROtoIMU = std::dynamic_pointer_cast<JPLQuat>(resolve(state->_calib_imu_GYROtoIMU));
  out->_calib_imu_ACCtoIMU = std::dynamic_pointer_cast<JPLQuat>(resolve(state->_calib_imu_ACCtoIMU));

  return out;
}

bool StateHelper::prepare_sampled_imu_boundary(std::shared_ptr<State> state, uint64_t stream_episode) {
  if (!state || !stream_episode)
    return false;
  if (state->has_sampled_imu_boundary())
    return state->_sampled_imu_stream_episode == stream_episode;

  // Stage all allocations before modifying the existing estimator. These two
  // coordinates groups remain allocated through every subsequent raw knot.
  const int n = state->_Cov.rows();
  Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(n + 12, n + 12);
  covariance.topLeftCorner(n, n) = state->_Cov;
  auto variables = state->_variables;
  variables.reserve(variables.size() + 2);
  std::array<State::SampledImuSlot, 2> slots;
  for (size_t i = 0; i < slots.size(); ++i) {
    slots[i].noise = std::make_shared<Vec>(6);
    slots[i].noise->set_local_id(n + 6 * i);
    variables.push_back(slots[i].noise);
  }
  state->_Cov.swap(covariance);
  state->_variables.swap(variables);
  state->_sampled_imu_slots = std::move(slots);
  state->_sampled_imu_stream_episode = stream_episode;
  return true;
}

namespace {
bool sampled_finite(double value) {
  const uint64_t bits = ov_core::initializer_time_bits(value);
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

bool same_sampled_record(const State::SampledImuRecord &a, const State::SampledImuRecord &b) {
  return a.stream_episode == b.stream_episode && a.sequence == b.sequence &&
      ov_core::initializer_time_bits(a.timestamp) == ov_core::initializer_time_bits(b.timestamp) &&
      std::memcmp(a.measured.data(), b.measured.data(), 6 * sizeof(double)) == 0 &&
      std::memcmp(a.prior.data(), b.prior.data(), 36 * sizeof(double)) == 0;
}

bool valid_sampled_record(const State::SampledImuRecord &record) {
  if (!record.stream_episode || !record.sequence || !sampled_finite(record.timestamp))
    return false;
  for (int i = 0; i < 6; ++i) {
    if (!sampled_finite(record.measured(i)) || record.prior(i, i) < 0.) return false;
    for (int j = 0; j < 6; ++j)
      if (!sampled_finite(record.prior(i, j))) return false;
  }
  const double tolerance = 64. * 6 * std::numeric_limits<double>::epsilon() * record.prior.cwiseAbs().maxCoeff();
  if (!sampled_finite(tolerance) || (record.prior - record.prior.transpose()).cwiseAbs().maxCoeff() > tolerance)
    return false;
  // Fixed six-dimensional workspace: no per-record heap allocation, including
  // singular PSD priors. Never add jitter or drop calibrated off-diagonal terms.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> spectrum(record.prior.selfadjointView<Eigen::Upper>(),
                                                                  Eigen::EigenvaluesOnly);
  if (spectrum.info() != Eigen::Success) return false;
  for (int i = 0; i < 6; ++i)
    if (!sampled_finite(spectrum.eigenvalues()(i)) || spectrum.eigenvalues()(i) < -tolerance) return false;
  return true;
}
} // namespace

std::shared_ptr<Vec> StateHelper::admit_sampled_imu_noise(std::shared_ptr<State> state,
                                                        const State::SampledImuRecord &record) {
  if (!state || !state->has_sampled_imu_boundary() || record.stream_episode != state->_sampled_imu_stream_episode)
    return nullptr;
  State::SampledImuSlot *free_slot = nullptr;
  for (auto &slot : state->_sampled_imu_slots) {
    if (slot.active && slot.record.sequence == record.sequence)
      return same_sampled_record(slot.record, record) ? slot.noise : nullptr;
    if (!slot.active) free_slot = &slot;
  }
  if (!free_slot || record.sequence <= state->_sampled_imu_last_sequence ||
      (state->_sampled_imu_last_sequence && !(record.timestamp > state->_sampled_imu_last_timestamp)) ||
      !valid_sampled_record(record))
    return nullptr;

  // Inactive means/FEJ/cross rows were zeroed at preparation or retirement.
  // Preserve the caller's exact prior metadata; use its symmetric upper view
  // for the covariance, matching the covariance boundary's symmetry tolerance.
  const int id = free_slot->noise->id();
  state->_Cov.block<6, 6>(id, id) = record.prior.selfadjointView<Eigen::Upper>();
  free_slot->record = record;
  free_slot->active = true;
  state->_sampled_imu_last_sequence = record.sequence;
  state->_sampled_imu_last_timestamp = record.timestamp;
  return free_slot->noise;
}

bool StateHelper::retire_sampled_imu_noise_at_knot(std::shared_ptr<State> state, uint64_t sequence) {
  if (!state || !state->has_sampled_imu_boundary() || !state->_imu_endpoint_valid)
    return false;
  auto &a = state->_sampled_imu_slots[0];
  auto &b = state->_sampled_imu_slots[1];
  if (!a.active || !b.active) return false;
  auto &older = a.record.sequence < b.record.sequence ? a : b;
  const auto &newer = a.record.sequence < b.record.sequence ? b : a;
  if (sequence != older.record.sequence ||
      ov_core::initializer_time_bits(state->_imu_endpoint) != ov_core::initializer_time_bits(newer.record.timestamp))
    return false;

  // Covariance marginalization leaves the kept principal block untouched.
  // Replace only the retired coordinates with an independent deterministic
  // zero slot; unlike conditioning, this removes no information from P_kept.
  const int id = older.noise->id();
  state->_Cov.middleRows(id, 6).setZero();
  state->_Cov.middleCols(id, 6).setZero();
  older.noise->set_value(older.noise->fej()); // frozen raw-noise origin is zero
  older.record = State::SampledImuRecord();
  older.active = false;
  return true;
}

bool StateHelper::valid_sampled_imu_record(const State::SampledImuRecord &record) { return valid_sampled_record(record); }

bool StateHelper::EKFPropagationSampled(std::shared_ptr<State> state,
                                       const std::array<State::SampledImuRecord, 2> &records,
                                       const Eigen::MatrixXd &Phi, const Eigen::MatrixXd &Q) {
  if (!state || !state->has_sampled_imu_boundary() || state->_imu->id() != 0 || state->imu_intrinsic_size() != 0)
    return false;
  const int d = 15, n = state->_Cov.rows();
  if (Phi.rows() != d || Phi.cols() != d + 12 || Q.rows() != d || Q.cols() != d ||
      !valid_initial_covariance(Q, true) || !(records[1].sequence > records[0].sequence) ||
      !(records[1].timestamp > records[0].timestamp))
    return false;
  auto finite_matrix = [](const Eigen::MatrixXd &m) {
    for (Eigen::Index i = 0; i < m.size(); ++i) if (!sampled_finite(m.data()[i])) return false;
    return true;
  };
  if (!finite_matrix(Phi)) return false;

  // Preview the two admissions using only fixed-size metadata. A bad second
  // record cannot partially admit the first or advance its watermark.
  auto slots = state->_sampled_imu_slots;
  auto last_sequence = state->_sampled_imu_last_sequence;
  auto last_timestamp = state->_sampled_imu_last_timestamp;
  std::array<int, 2> selected;
  for (size_t record_index = 0; record_index < 2; ++record_index) {
    const auto &record = records[record_index];
    if (record.stream_episode != state->_sampled_imu_stream_episode || !valid_sampled_record(record)) return false;
    int active = -1, available = -1;
    for (int i = 0; i < 2; ++i) {
      if (slots[i].active && slots[i].record.sequence == record.sequence) active = i;
      if (!slots[i].active) available = i;
    }
    if (active >= 0) {
      if (!same_sampled_record(slots[active].record, record)) return false;
      selected[record_index] = active;
    } else {
      if (available < 0 || record.sequence <= last_sequence || (last_sequence && !(record.timestamp > last_timestamp)))
        return false;
      selected[record_index] = available;
      slots[available].record = record;
      slots[available].active = true;
      last_sequence = record.sequence;
      last_timestamp = record.timestamp;
    }
  }
  for (const auto &slot : slots)
    if (!slot.noise || slot.noise->id() < d || slot.noise->id() + 6 > n) return false;

  // This is the ordinary EKFPropagation product with a virtual independent
  // prior in each newly occupied zero slot. Scratch is O(n*d), not a new
  // full-State snapshot; no live covariance is written before validation.
  Eigen::MatrixXd cross = state->_Cov.leftCols(d) * Phi.leftCols(d).transpose();
  for (int i = 0; i < 2; ++i) {
    const int index = selected[i], id = slots[index].noise->id();
    cross.noalias() += state->_Cov.middleCols(id, 6) * Phi.middleCols(d + 6 * i, 6).transpose();
    if (!state->_sampled_imu_slots[index].active)
      cross.middleRows(id, 6).noalias() += records[i].prior.selfadjointView<Eigen::Upper>() *
                                          Phi.middleCols(d + 6 * i, 6).transpose();
  }
  Eigen::MatrixXd predicted = Q.selfadjointView<Eigen::Upper>();
  predicted.noalias() += Phi.leftCols(d) * cross.topRows(d);
  for (int i = 0; i < 2; ++i)
    predicted.noalias() += Phi.middleCols(d + 6 * i, 6) * cross.middleRows(slots[selected[i]].noise->id(), 6);
  if (!finite_matrix(cross) || !finite_matrix(predicted) || !valid_initial_covariance(predicted, true)) return false;

  for (int i = 0; i < 2; ++i) {
    if (!state->_sampled_imu_slots[i].active)
      state->_Cov.block<6, 6>(slots[i].noise->id(), slots[i].noise->id()) = slots[i].record.prior.selfadjointView<Eigen::Upper>();
  }
  state->_Cov.topRows(d) = cross.transpose();
  state->_Cov.leftCols(d) = cross;
  state->_Cov.topLeftCorner(d, d) = predicted.selfadjointView<Eigen::Upper>();
  state->_sampled_imu_slots = std::move(slots);
  state->_sampled_imu_last_sequence = last_sequence;
  state->_sampled_imu_last_timestamp = last_timestamp;
  return true;
}

bool StateHelper::EKFPropagation(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &order_NEW,
                                 const std::vector<std::shared_ptr<Type>> &order_OLD, const Eigen::MatrixXd &Phi,
                                 const Eigen::MatrixXd &Q) {

  // A valid prior is the caller's contract. Validate only the owned map and
  // new process noise here, without factoring or copying the full covariance.
  if (!state || order_NEW.empty() || order_OLD.empty() || Phi.rows() <= 0 || Phi.cols() <= 0 ||
      Q.rows() != Phi.rows() || Q.cols() != Phi.rows() ||
      !ov_core::numeric::finite_matrix(Phi) || !ov_core::numeric::finite_matrix(Q)) return false;
  const int n = state->_Cov.rows();
  if (state->_Cov.cols() != n) return false;
  int owned = 0;
  for (const auto &owner : state->_variables) {
    if (!owner || owner->id() != owned || owner->size() <= 0 || owner->size() > n-owned) return false;
    owned += owner->size();
  }
  if (owned != n) return false;
  const auto valid_order = [&](const std::vector<std::shared_ptr<Type>> &order, int size, bool contiguous) {
    int count = 0, previous_end = -1;
    for (const auto &var : order) {
      if (!var || var->id() < 0 || var->size() <= 0 || var->id() > n || var->size() > n-var->id() ||
          var->size() > size-count || (contiguous && previous_end >= 0 && previous_end != var->id())) return false;
      const auto after = std::upper_bound(state->_variables.begin(), state->_variables.end(), var->id(),
          [](int id, const std::shared_ptr<Type> &owner) { return id < owner->id(); });
      if (after == state->_variables.begin()) return false;
      const auto &owner = *std::prev(after);
      if (var->id()+var->size() > owner->id()+owner->size() ||
          (var != owner && owner->check_if_subvariable(var) != var)) return false;
      previous_end = var->id()+var->size(); count += var->size();
    }
    return count == size;
  };
  if (!valid_order(order_NEW, Phi.rows(), true) || !valid_order(order_OLD, Phi.cols(), false)) return false;
  bool diagonal_noise = true;
  for (int c=0;c<Q.cols();++c) {
    if (Q(c,c) < 0.) return false;
    for (int r=0;r<Q.rows();++r) if (r!=c && Q(r,c)!=0.) diagonal_noise=false;
  }
  if (!diagonal_noise && !valid_initial_covariance(Q, true)) return false;

  // Loop through all our old states and get the state transition times it
  // Cov_PhiT = [ Pxx ] [ Phi' ]'
  Eigen::MatrixXd Cov_PhiT = Eigen::MatrixXd::Zero(state->_Cov.rows(), Phi.rows());
  int current_it = 0;
  for (size_t i = 0; i < order_OLD.size(); i++) {
    std::shared_ptr<Type> var = order_OLD.at(i);
    Cov_PhiT.noalias() +=
        state->_Cov.block(0, var->id(), state->_Cov.rows(), var->size()) * Phi.block(0, current_it, Phi.rows(), var->size()).transpose();
    current_it += var->size();
  }

  // Get Phi_NEW*Covariance*Phi_NEW^t + Q
  Eigen::MatrixXd Phi_Cov_PhiT = Q.selfadjointView<Eigen::Upper>();
  current_it = 0;
  for (size_t i = 0; i < order_OLD.size(); i++) {
    std::shared_ptr<Type> var = order_OLD.at(i);
    Phi_Cov_PhiT.noalias() += Phi.block(0, current_it, Phi.rows(), var->size()) * Cov_PhiT.block(var->id(), 0, var->size(), Phi.rows());
    current_it += var->size();
  }

  // Existing O(n*k) and O(k*k) temporaries contain every value to commit.
  // Inspect them before overwriting any row of the live prior. A state-sized
  // rollback copy or eigenvalue factorization is unnecessary for a valid P/Q.
  if (!ov_core::numeric::finite_matrix(Cov_PhiT) || !ov_core::numeric::finite_matrix(Phi_Cov_PhiT)) return false;
  for (int i=0;i<Phi_Cov_PhiT.rows();++i) if (Phi_Cov_PhiT(i,i)<0.) return false;

  // We are good to go!
  int start_id = order_NEW.at(0)->id();
  int phi_size = Phi.rows();
  int total_size = state->_Cov.rows();
  state->_Cov.block(start_id, 0, phi_size, total_size) = Cov_PhiT.transpose();
  state->_Cov.block(0, start_id, total_size, phi_size) = Cov_PhiT;
  state->_Cov.block(start_id, start_id, phi_size, phi_size) = Phi_Cov_PhiT;

  return true;
}

bool StateHelper::EKFUpdate(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &H_order, const Eigen::MatrixXd &H,
                            const Eigen::VectorXd &res, const Eigen::MatrixXd &R) {

  // Validate data and ownership before assembling products. Existing state
  // covariance is a valid prior; this boundary does not refactor all of P.
  if (!state || res.rows() <= 0 || R.rows() != res.rows() || R.cols() != res.rows() ||
      H.rows() != res.rows() || !ov_core::numeric::finite_matrix(H) ||
      !ov_core::numeric::finite_matrix(R) || !ov_core::numeric::finite_matrix(res))
    return false;
  const int n = state->_Cov.rows();
  int ordered_size = 0, value_size = 0, state_size = 0;
  if (state->_Cov.cols() != n) return false;
  for (const auto &var : state->_variables) {
    if (!var || var->id() != state_size || var->size() <= 0 ||
        var->value().cols() != 1 || var->value().rows() <= 0 ||
        !ov_core::numeric::finite_matrix(var->value())) return false;
    state_size += var->size();
    value_size += var->value().rows();
  }
  if (state_size != n) return false;
  for (const auto &var : H_order) {
    if (!var || var->id() < 0 || var->size() <= 0 || var->id() + var->size() > n) return false;
    const auto after = std::upper_bound(state->_variables.begin(), state->_variables.end(), var->id(),
        [](int id, const std::shared_ptr<Type> &owner) { return id < owner->id(); });
    if (after == state->_variables.begin()) return false;
    const auto &owner = *std::prev(after);
    if (var->id() + var->size() > owner->id() + owner->size() ||
        (owner != var && owner->check_if_subvariable(var) != var)) return false;
    ordered_size += var->size();
  }
  if (ordered_size != H.cols()) return false;
  // Most callers whiten to diagonal R. Exact constraints may have zero R;
  // their innovation must still be SPD. General correlated R needs a PSD
  // check, never an implicit assumption that positive diagonals suffice.
  bool diagonal_noise = true;
  for (int c = 0; c < R.cols(); ++c) {
    if (R(c,c) < 0.) return false;
    for (int row = 0; row < R.rows(); ++row)
      if (row != c && R(row,c) != 0.) diagonal_noise = false;
  }
  if (!diagonal_noise && !valid_initial_covariance(R, true)) return false;

  //==========================================================
  //==========================================================
  // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
  assert(res.rows() == R.rows());
  assert(H.rows() == res.rows());
  Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());

  // Get the location in small jacobian for each measuring variable
  int current_it = 0;
  std::vector<int> H_id;
  for (const auto &meas_var : H_order) {
    H_id.push_back(current_it);
    current_it += meas_var->size();
  }

  //==========================================================
  //==========================================================
  // For each active variable find its M = P*H^T
  for (const auto &var : state->_variables) {
    // Sum up effect of each subjacobian = K_i= \sum_m (P_im Hm^T)
    Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
    for (size_t i = 0; i < H_order.size(); i++) {
      std::shared_ptr<Type> meas_var = H_order[i];
      M_i.noalias() += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                       H.block(0, H_id[i], H.rows(), meas_var->size()).transpose();
    }
    M_a.block(var->id(), 0, var->size(), res.rows()) = M_i;
  }

  //==========================================================
  //==========================================================
  // Get covariance of the involved terms
  Eigen::MatrixXd P_small = StateHelper::get_marginal_covariance(state, H_order);

  // Residual covariance S = H*Cov*H' + R
  Eigen::MatrixXd S(R.rows(), R.rows());
  S.triangularView<Eigen::Upper>() = H * P_small * H.transpose();
  S.triangularView<Eigen::Upper>() += R;
  // Eigen::MatrixXd S = H * P_small * H.transpose() + R;

  // Factor the declared upper triangle and solve S*K^T = M^T directly.
  // The transpose is a writable view of K, so neither an explicit inverse
  // nor a second n-by-m gain temporary is needed.
  for (int c = 0; c < S.cols(); ++c)
    for (int row = 0; row <= c; ++row)
      if (!ov_core::numeric::finite(S(row,c))) return false;
  Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> innovation(S.selfadjointView<Eigen::Upper>());
  if (innovation.info() != Eigen::Success) return false;
  Eigen::MatrixXd K = M_a;
  auto K_transpose = K.transpose();
  innovation.solveInPlace(K_transpose);
  if (!ov_core::numeric::finite_matrix(K)) return false;

  // A Schmidt update keeps the full H (and hence nuisance uncertainty in S), but sets the
  // temporal gain rows to zero. With the ordinary optimal K, its Joseph covariance has the
  // ordinary active-active and active-frozen blocks; only frozen-frozen remains its prior.
  // Save that small block before the ordinary update. Gate-off keeps the original arithmetic
  // and does not touch/allocate scratch. IDs are rebuilt because state marginalization can
  // renumber variables; the reference clock is already one entry of the per-camera map.
  const bool freeze_temporal = state->_options.dt_calib_gate && state->dt_calib_degenerate();
  if (freeze_temporal) {
    auto &ids = state->_temporal_schmidt_ids;
    ids.clear();
    ids.reserve(2 * static_cast<size_t>(state->_options.num_cameras));
    for (int cam = 0; cam < state->_options.num_cameras; ++cam) {
      for (const auto &var : {state->_calib_dt_CAMtoIMU_map.at(cam), state->_calib_camera_readout.at(cam)}) {
        if (var->id() >= 0) {
          assert(var->size() == 1);
          ids.push_back(var->id());
        }
      }
    }
    auto &prior = state->_temporal_schmidt_prior;
    prior.resize(ids.size() * ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      for (size_t j = 0; j < ids.size(); ++j) {
        prior[i * ids.size() + j] = state->_Cov(ids[i], ids[j]);
      }
    }
  }

  // O(n) proposed values use the exact Type retractions without cloning the
  // state or individual Types. FEJ and object identities remain untouched.
  Eigen::VectorXd dx = K * res;
  if (freeze_temporal)
    for (int id : state->_temporal_schmidt_ids) dx(id) = 0.;
  if (!ov_core::numeric::finite_matrix(dx)) return false;
  Eigen::VectorXd proposed_values(value_size);
  int value_offset = 0;
  for (const auto &var : state->_variables) {
    var->propose_update(dx.segment(var->id(), var->size()),
                        proposed_values.segment(value_offset, var->value().rows()));
    value_offset += var->value().rows();
  }
  if (!ov_core::numeric::finite_matrix(proposed_values)) return false;
  if (state->_options.do_calib_camera_intrinsics) {
    for (const auto &calib : state->_cam_intrinsics) {
      const auto model = state->_cam_intrinsics_cameras.find(calib.first);
      if (model == state->_cam_intrinsics_cameras.end() || !model->second) return false;
      int offset = 0;
      bool found = false;
      for (const auto &var : state->_variables) {
        if (var == calib.second) {
          if (var->value().rows() != 8 || !(proposed_values(offset) > 0.) || !(proposed_values(offset+1) > 0.)) return false;
          found = true;
          break;
        }
        offset += var->value().rows();
      }
      if (!found) return false;
    }
  }

  // Preserve the original triangular product arithmetic and use P's existing
  // lower triangle as the rollback copy. Only its shared diagonal needs O(n)
  // scratch. State access is serialized by the caller, as for every EKF update;
  // no observer may inspect the in-progress upper triangle.
  const Eigen::VectorXd prior_diagonal = state->_Cov.diagonal();
  state->_Cov.triangularView<Eigen::Upper>() -= K * M_a.transpose();
  if (freeze_temporal) {
    const auto &ids = state->_temporal_schmidt_ids;
    const auto &prior = state->_temporal_schmidt_prior;
    for (size_t i = 0; i < ids.size(); ++i)
      for (size_t j = 0; j < ids.size(); ++j)
        state->_Cov(ids[i], ids[j]) = prior[i * ids.size() + j];
  }
  bool valid_covariance = true;
  for (int c = 0; c < n && valid_covariance; ++c) {
    if (state->_Cov(c,c) < 0.) valid_covariance = false;
    for (int row = 0; row <= c && valid_covariance; ++row)
      valid_covariance = ov_core::numeric::finite(state->_Cov(row,c));
  }
  if (!valid_covariance) {
    state->_Cov.diagonal() = prior_diagonal;
    state->_Cov = state->_Cov.selfadjointView<Eigen::Lower>();
    return false;
  }
  // All numerical proposals passed. Mirror the committed triangle and install
  // the already validated means. Allocation failure is outside this contract.
  state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
  value_offset = 0;
  for (const auto &var : state->_variables) {
    var->set_value(proposed_values.segment(value_offset, var->value().rows()));
    value_offset += var->value().rows();
  }

  // If we are doing online intrinsic calibration we should update our camera objects
  // NOTE: is this the best place to put this update logic??? probably..
  if (state->_options.do_calib_camera_intrinsics) {
    for (auto const &calib : state->_cam_intrinsics) {
      state->_cam_intrinsics_cameras.at(calib.first)->set_value(calib.second->value());
    }
  }
  return true;
}

void StateHelper::set_initial_covariance(std::shared_ptr<State> state, const Eigen::MatrixXd &covariance,
                                         const std::vector<std::shared_ptr<ov_type::Type>> &order) {

  // We need to loop through each element and overwrite the current covariance values
  // For example consider the following:
  // x = [ ori pos ] -> insert into -> x = [ ori bias pos ]
  // P = [ P_oo P_op ] -> P = [ P_oo  0   P_op ]
  //     [ P_po P_pp ]        [  0    P*    0  ]
  //                          [ P_po  0   P_pp ]
  // The key assumption here is that the covariance is block diagonal (cross-terms zero with P* can be dense)
  // This is normally the care on startup (for example between calibration and the initial state

  // For each variable, lets copy over all other variable cross terms
  // Note: this copies over itself to when i_index=k_index
  int i_index = 0;
  for (size_t i = 0; i < order.size(); i++) {
    int k_index = 0;
    for (size_t k = 0; k < order.size(); k++) {
      state->_Cov.block(order[i]->id(), order[k]->id(), order[i]->size(), order[k]->size()) =
          covariance.block(i_index, k_index, order[i]->size(), order[k]->size());
      k_index += order[k]->size();
    }
    i_index += order[i]->size();
  }
  state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
}

bool StateHelper::valid_initial_covariance(const Eigen::Ref<const Eigen::MatrixXd> &covariance) {
  return valid_initial_covariance(covariance, false);
}

bool StateHelper::valid_initial_covariance(const Eigen::Ref<const Eigen::MatrixXd> &covariance, bool allow_zero_diagonal) {
  const auto finite = [](double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
  };
  const int n = covariance.rows();
  if (n == 0 || covariance.cols() != n)
    return false;
  // A marginal block may have a larger outer stride than n. Do not inspect it
  // as a flat contiguous array. Bit checks remain valid under -ffast-math.
  for (int col = 0; col < n; ++col)
    for (int row = 0; row < n; ++row)
      if (!finite(covariance(row, col)))
        return false;
  for (int i = 0; i < n; ++i)
    if (covariance(i, i) < 0. || (!allow_zero_diagonal && covariance(i, i) == 0.))
      return false;
  const double roundoff = (64.0 * std::numeric_limits<double>::epsilon() * n) * covariance.cwiseAbs().maxCoeff();
  if (!finite(roundoff)) return false;
  if ((covariance - covariance.transpose()).cwiseAbs().maxCoeff() > roundoff)
    return false;
  Eigen::LDLT<Eigen::MatrixXd> factor(covariance.selfadjointView<Eigen::Upper>());
  bool psd = factor.info() == Eigen::Success;
  for (int i = 0; psd && i < n; ++i)
    psd = finite(factor.vectorD()(i)) && factor.vectorD()(i) >= -roundoff;
  if (!psd) {
    // A valid newest clone can duplicate the IMU pose exactly. LDLT sometimes
    // reports NumericalIssue at its zero pivot; resolve that ambiguity without
    // modifying the recovered covariance or pretending it is full rank.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(covariance.selfadjointView<Eigen::Upper>(),
                                                          Eigen::EigenvaluesOnly);
    if (spectrum.info() != Eigen::Success)
      return false;
    for (int i = 0; i < n; ++i)
      if (!finite(spectrum.eigenvalues()(i)) || spectrum.eigenvalues()(i) < -roundoff)
        return false;
  }
  return true;
}

bool StateHelper::make_physical_reset_state(std::shared_ptr<State> state, uint64_t snapshot_id,
                                           const std::vector<double> &raw_watermarks, double imu_raw_cutoff,
                                           const Eigen::Matrix<double,6,1> &bias_rw_variance, int cause,
                                           std::shared_ptr<State> &replacement,
                                           std::shared_ptr<const InitPhysicalResetPrior> &prior) {
  using ov_core::numeric::finite;
  using ov_core::numeric::finite_matrix;
  if(!state || !snapshot_id || !state->uses_physical_clones() || state->has_sampled_imu_boundary() ||
      !state->_imu_endpoint_valid || !state->_imu || state->_imu->id()!=0 || state->_imu->value().size()!=16 ||
      !finite_matrix(state->_imu->value()) || !finite(state->imu_endpoint()) || !finite(imu_raw_cutoff) ||
      imu_raw_cutoff<state->imu_endpoint() || (cause!=0 && cause!=1) || !finite_matrix(bias_rw_variance) ||
      (bias_rw_variance.array()<0.).any()) return false;
  const auto &options=state->_options;
  if(options.num_cameras<=0 || raw_watermarks.size()!=size_t(options.num_cameras) ||
      options.do_calib_camera_readout || options.do_calib_imu_intrinsics || options.do_calib_imu_g_sensitivity ||
      state->_Cov.rows()<15 || state->_Cov.rows()!=state->_Cov.cols()) return false;
  for(double raw:raw_watermarks)
    if(!finite(raw) && raw!=-std::numeric_limits<double>::infinity()) return false;
  auto staged=std::make_shared<State>(state->_options);
  auto next=std::make_shared<InitPhysicalResetPrior>();
  next->snapshot_id=snapshot_id;next->imu_endpoint=state->imu_endpoint();next->imu_raw_cutoff=imu_raw_cutoff;
  next->bias_mean=state->_imu->value().block<6,1>(10,0);next->bias_covariance=state->_Cov.block<6,6>(9,9);
  next->bias_rw_variance=bias_rw_variance;next->raw_watermarks=raw_watermarks;next->cause=cause;
  next->reference_camera_id=options.cam_imu_dt_ref_camid;
  const auto copy_value=[&](const std::shared_ptr<Type> &from,const std::shared_ptr<Type> &to,int size) {
    if(!from || !to || from->value().size()!=size || from->fej().size()!=size ||
        !finite_matrix(from->value()) || !finite_matrix(from->fej())) return false;
    to->set_value(from->value());to->set_fej(from->value());return true;
  };
  if(!copy_value(state->_calib_imu_dw,staged->_calib_imu_dw,6) ||
      !copy_value(state->_calib_imu_da,staged->_calib_imu_da,6) ||
      !copy_value(state->_calib_imu_tg,staged->_calib_imu_tg,9) ||
      !copy_value(state->_calib_imu_GYROtoIMU,staged->_calib_imu_GYROtoIMU,4) ||
      !copy_value(state->_calib_imu_ACCtoIMU,staged->_calib_imu_ACCtoIMU,4)) return false;
  next->imu_accel_map=staged->_calib_imu_ACCtoIMU->Rot()*State::Dm(options.imu_model,staged->_calib_imu_da->value());
  next->imu_gyro_map=staged->_calib_imu_GYROtoIMU->Rot()*State::Dm(options.imu_model,staged->_calib_imu_dw->value());
  next->imu_tg=State::Tg(staged->_calib_imu_tg->value());
  for(int camera=0;camera<options.num_cameras;++camera) {
    const auto clock=state->_calib_dt_CAMtoIMU_map.find(camera),readout=state->_calib_camera_readout.find(camera);
    const auto pose=state->_calib_IMUtoCAM.find(camera);const auto intrinsics=state->_cam_intrinsics.find(camera);
    const auto model=state->_cam_intrinsics_cameras.find(camera);
    if(clock==state->_calib_dt_CAMtoIMU_map.end() || readout==state->_calib_camera_readout.end() ||
        pose==state->_calib_IMUtoCAM.end() || intrinsics==state->_cam_intrinsics.end() ||
        model==state->_cam_intrinsics_cameras.end() || !model->second ||
        !copy_value(clock->second,staged->cam_imu_dt_var(camera),1) ||
        !copy_value(readout->second,staged->_calib_camera_readout.at(camera),1) ||
        !copy_value(pose->second,staged->_calib_IMUtoCAM.at(camera),7) ||
        !copy_value(intrinsics->second,staged->_cam_intrinsics.at(camera),8) ||
        readout->second->value()(0)!=0. || std::abs(pose->second->value().block<4,1>(0,0).norm()-1.)>1e-10 ||
        !finite_matrix(model->second->get_value()) || model->second->get_value().size()!=8 ||
        !(model->second->get_value().array()==intrinsics->second->value().array()).all()) return false;
    InitFixedCameraCalibration calibration;
    calibration.camera_id=camera;calibration.clock_mean=clock->second->value()(0);
    calibration.extrinsics=pose->second->value();calibration.intrinsics=intrinsics->second->value();
    calibration.width=model->second->w();calibration.height=model->second->h();
    calibration.fisheye=bool(std::dynamic_pointer_cast<CamEqui>(model->second));
    if(calibration.width<=0 || calibration.height<=0 ||
        (!calibration.fisheye && !std::dynamic_pointer_cast<CamRadtan>(model->second))) return false;
    next->calibration.push_back(std::move(calibration));
    staged->_cam_intrinsics_cameras.emplace(camera,model->second); // frontend identity, no mutation
  }
  InitPhysicalWarmRequest layout;
  if(!make_initial_physical_warm_request(staged,snapshot_id,layout)) return false;
  next->consider=layout.consider;
  const int k=layout.calibration_covariance.rows();
  next->calibration_covariance.resize(k,k);next->bias_calibration_covariance.resize(6,k);
  std::vector<int> source,destination;
  for(const auto &block:next->consider) {
    std::shared_ptr<Type> a,b;
    if(block.kind==InitCameraCalibrationKind::Clock){a=state->cam_imu_dt_var(block.camera_id);b=staged->cam_imu_dt_var(block.camera_id);}
    else if(block.kind==InitCameraCalibrationKind::Extrinsics){a=state->_calib_IMUtoCAM.at(block.camera_id);b=staged->_calib_IMUtoCAM.at(block.camera_id);}
    else {a=state->_cam_intrinsics.at(block.camera_id);b=staged->_cam_intrinsics.at(block.camera_id);}
    if(a->id()<15 || a->id()+a->size()>state->_Cov.rows() || a->size()!=block.local_size() ||
        std::find(state->_variables.begin(),state->_variables.end(),a)==state->_variables.end()) return false;
    for(int i=0;i<a->size();++i) {
      if(std::find(source.begin(),source.end(),a->id()+i)!=source.end()) return false;
      source.push_back(a->id()+i);destination.push_back(b->id()+i);
    }
  }
  for(int r=0;r<k;++r) {
    next->bias_calibration_covariance.col(r)=state->_Cov.block<6,1>(9,source[r]);
    if(!(state->_Cov.block<1,6>(source[r],9).transpose().array()==next->bias_calibration_covariance.col(r).array()).all())return false;
    for(int c=0;c<k;++c) next->calibration_covariance(r,c)=state->_Cov(source[r],source[c]);
  }
  if(!ov_init::conditional_bias::valid_joint(next->bias_covariance,next->bias_calibration_covariance,next->calibration_covariance)) return false;
  auto mean=staged->_imu->value();mean.block<6,1>(10,0)=next->bias_mean;
  staged->_imu->set_value(mean);staged->_imu->set_fej(mean);
  staged->_Cov.block<6,6>(9,9)=next->bias_covariance;
  for(int r=0;r<k;++r) {
    staged->_Cov.block<6,1>(9,destination[r])=next->bias_calibration_covariance.col(r);
    staged->_Cov.block<1,6>(destination[r],9)=next->bias_calibration_covariance.col(r).transpose();
    for(int c=0;c<k;++c)staged->_Cov(destination[r],destination[c])=next->calibration_covariance(r,c);
  }
  if(!make_initial_physical_warm_request(staged,snapshot_id,layout,next)) return false;
  replacement=std::move(staged);prior=std::move(next);return true;
}

bool StateHelper::make_initial_physical_warm_request(std::shared_ptr<State> state, uint64_t episode_id,
                                                    InitPhysicalWarmRequest &request,
                                                    std::shared_ptr<const InitPhysicalResetPrior> reset_prior) {
  if (!state || !episode_id || !state->uses_physical_clones() || !state->_imu || state->_imu->id() != 0 ||
      state->_variables.empty() || state->_variables.front() != state->_imu || state->_timestamp != -1. ||
      state->_initialization_episode_id || state->_imu_endpoint_valid || !state->_exposure_poses.empty() ||
      !state->_clones_IMU.empty() || !state->_features_SLAM.empty() || !state->_clones_kinematics.empty() ||
      !state->_epoch_residuals.empty() || !state->_epoch_bridges.empty()) return false;
  const auto &options = state->_options;
  if (options.num_cameras <= 0 || options.max_pose_clones() <= 0 || options.do_calib_camera_readout ||
      options.do_calib_imu_intrinsics || options.do_calib_imu_g_sensitivity) return false;
  const auto finite = [](const auto &x) {
    for (Eigen::Index c=0;c<x.cols();++c)
      for (Eigen::Index r=0;r<x.rows();++r)
        if ((initializer_time_bits(x(r,c)) & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000)) return false;
    return true;
  };
  InitPhysicalWarmRequest next;
  next.episode_id = episode_id; next.max_retained_owners = options.max_pose_clones();
  std::vector<std::shared_ptr<Type>> variables;
  for (int camera=0;camera<options.num_cameras;++camera) {
    const auto clock = state->_calib_dt_CAMtoIMU_map.find(camera);
    const auto pose = state->_calib_IMUtoCAM.find(camera);
    const auto intrinsics = state->_cam_intrinsics.find(camera);
    const auto readout = state->_calib_camera_readout.find(camera);
    if (clock == state->_calib_dt_CAMtoIMU_map.end() || pose == state->_calib_IMUtoCAM.end() ||
        intrinsics == state->_cam_intrinsics.end() || readout == state->_calib_camera_readout.end() ||
        !clock->second || !pose->second || !intrinsics->second || !readout->second || readout->second->id() >= 0 ||
        !finite(readout->second->value()) || readout->second->value()(0) != 0.) return false;
    const auto append = [&](InitCameraCalibrationKind kind, const std::shared_ptr<Type> &type, bool estimated) {
      if (!type || (estimated ? type->id() < 15 : type->id() >= 0)) return false;
      if (!estimated) return true;
      InitCameraCalibrationBlock block; block.camera_id=camera; block.kind=kind;
      block.mean=type->value(); block.fej=type->fej();
      if (type->size() != block.local_size() || block.mean.size() != block.value_size() ||
          block.fej.size() != block.value_size() || !finite(block.mean) || !finite(block.fej) ||
          !(block.mean.array() == block.fej.array()).all() ||
          std::find(variables.begin(),variables.end(),type) != variables.end()) return false;
      next.consider.push_back(std::move(block)); variables.push_back(type); return true;
    };
    if (!append(InitCameraCalibrationKind::Clock,clock->second,options.do_calib_camera_timeoffset) ||
        !append(InitCameraCalibrationKind::Extrinsics,pose->second,options.do_calib_camera_pose) ||
        !append(InitCameraCalibrationKind::Intrinsics,intrinsics->second,options.do_calib_camera_intrinsics)) return false;
  }
  if (state->_variables.size() != 1+variables.size()) return false;
  int dimension=0;
  for (const auto &type : state->_variables) {
    if (!type || type->id() != dimension || (type != state->_imu &&
        std::find(variables.begin(),variables.end(),type) == variables.end())) return false;
    dimension += type->size();
  }
  const int k=dimension-15;
  if (dimension < 15 || state->_Cov.rows() != dimension || state->_Cov.cols() != dimension ||
      !finite(state->_Cov)) return false;
  if(!reset_prior && (!state->_Cov.topRightCorner(15,k).isZero(0.) ||
      !state->_Cov.bottomLeftCorner(k,15).isZero(0.))) return false;
  next.calibration_covariance.resize(k,k);
  int row=0;
  for (const auto &a : variables) {
    int column=0;
    for (const auto &b : variables) {
      next.calibration_covariance.block(row,column,a->size(),b->size()) = state->_Cov.block(a->id(),b->id(),a->size(),b->size());
      column += b->size();
    }
    row += a->size();
  }
  if (k && !valid_initial_covariance(next.calibration_covariance,true)) return false;
  if(reset_prior) {
    const auto &prior=*reset_prior;
    const auto exact=[](const auto &a,const auto &b) {
      if(a.rows()!=b.rows() || a.cols()!=b.cols())return false;
      for(Eigen::Index r=0;r<a.rows();++r)for(Eigen::Index c=0;c<a.cols();++c)
        if(initializer_time_bits(a(r,c))!=initializer_time_bits(b(r,c)))return false;
      return true;
    };
    if(!ov_init::valid_physical_reset_prior(prior) || prior.consider.size()!=next.consider.size() ||
        !exact(prior.calibration_covariance,next.calibration_covariance) ||
        !exact(prior.bias_mean,state->_imu->value().block<6,1>(10,0)) ||
        !exact(prior.bias_covariance,state->_Cov.block<6,6>(9,9)) ||
        !state->_Cov.block(0,9,9,6).isZero(0.) || !state->_Cov.block(9,0,6,9).isZero(0.) ||
        !state->_Cov.topRightCorner(9,k).isZero(0.) || !state->_Cov.bottomLeftCorner(k,9).isZero(0.) ||
        !ov_init::conditional_bias::valid_joint(prior.bias_covariance,prior.bias_calibration_covariance,prior.calibration_covariance)) return false;
    int column=0;
    for(size_t i=0;i<variables.size();++i) {
      const auto &a=next.consider[i],&b=prior.consider[i];const auto &type=variables[i];
      if(a.camera_id!=b.camera_id || a.kind!=b.kind || !exact(a.mean,b.mean) || !exact(a.fej,b.fej) ||
          !exact(prior.bias_calibration_covariance.middleCols(column,type->size()),state->_Cov.block(9,type->id(),6,type->size())) ||
          !exact(prior.bias_calibration_covariance.middleCols(column,type->size()).transpose(),state->_Cov.block(type->id(),9,type->size(),6))) return false;
      column+=type->size();
    }
    next.reset_prior=std::move(reset_prior);
  }
  request = std::move(next);
  return true;
}

bool StateHelper::set_initial_state_physical_warm(std::shared_ptr<State> state, const InitPhysicalWarmResult &result,
                                                uint64_t expected_episode_id,
                                                std::shared_ptr<const InitPhysicalResetPrior> expected_reset_prior) {
  const auto finite = [](double value) {
    return (initializer_time_bits(value) & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
  };
  const auto finite_matrix = [&](const auto &value) {
    for (Eigen::Index c = 0; c < value.cols(); ++c)
      for (Eigen::Index r = 0; r < value.rows(); ++r)
        if (!finite(value(r,c))) return false;
    return true;
  };
  if (!state || !expected_episode_id || result.episode_id != expected_episode_id ||
      !state->uses_physical_clones() || !state->_imu || state->_imu->id() != 0 ||
      state->_variables.empty() || state->_variables.front() != state->_imu || state->_Cov.rows() < 15 ||
      !state->_exposure_poses.empty() || !state->_clones_IMU.empty() || !state->_features_SLAM.empty() ||
      !state->_clones_kinematics.empty() || !state->_epoch_residuals.empty() || !state->_epoch_bridges.empty() ||
      state->_initialization_episode_id != 0 || state->_imu_endpoint_valid) return false;
  InitPhysicalWarmRequest live;
  if(result.reset_prior!=expected_reset_prior ||
      !make_initial_physical_warm_request(state,expected_episode_id,live,expected_reset_prior) || live.consider.size() != result.consider.size()) return false;
  if(expected_reset_prior && (!finite(result.reset_first_imu_time) || !finite(result.reset_first_imu_support_time) ||
      !(result.reset_first_imu_support_time>expected_reset_prior->imu_raw_cutoff) ||
      result.reset_first_imu_time<result.reset_first_imu_support_time || result.reset_first_imu_time>result.accepted_imu_endpoint ||
      expected_reset_prior->raw_watermarks.size()!=size_t(state->_options.num_cameras))) return false;
  const int calibration_dimension = state->_Cov.rows()-15;
  if (result.calibration_covariance.rows() != calibration_dimension || result.calibration_covariance.cols() != calibration_dimension ||
      !finite_matrix(result.pose_error_scale) || (result.pose_error_scale.array() <= 0.).any()) return false;
  std::vector<bool> seen(live.consider.size(),false);
  std::vector<int> calibration_rows;
  std::map<size_t,int> clock_columns;
  for (const auto &block : result.consider) {
    size_t index=0;
    while (index < live.consider.size() && (live.consider[index].camera_id != block.camera_id || live.consider[index].kind != block.kind)) ++index;
    if (index == live.consider.size() || seen[index] || block.mean.size() != live.consider[index].mean.size() ||
        block.fej.size() != live.consider[index].fej.size() || !finite_matrix(block.mean) || !finite_matrix(block.fej) ||
        !(block.mean.array() == live.consider[index].mean.array()).all() || !(block.fej.array() == live.consider[index].fej.array()).all()) return false;
    seen[index]=true;
    std::shared_ptr<Type> type;
    if (block.kind == InitCameraCalibrationKind::Clock) {
      type=state->_calib_dt_CAMtoIMU_map.at(block.camera_id); clock_columns.emplace(block.camera_id,calibration_rows.size());
    } else if (block.kind == InitCameraCalibrationKind::Extrinsics) type=state->_calib_IMUtoCAM.at(block.camera_id);
    else type=state->_cam_intrinsics.at(block.camera_id);
    for (int j=0;j<type->size();++j) calibration_rows.push_back(type->id()+j);
  }
  if (calibration_rows.size() != size_t(calibration_dimension)) return false;
  for (int r=0;r<calibration_dimension;++r)
    for (int c=0;c<calibration_dimension;++c)
      if (initializer_time_bits(result.calibration_covariance(r,c)) !=
          initializer_time_bits(state->_Cov(calibration_rows[r],calibration_rows[c]))) return false;
  for (const auto &calibration : {std::static_pointer_cast<Type>(state->_calib_imu_da), std::static_pointer_cast<Type>(state->_calib_imu_dw),
                                 std::static_pointer_cast<Type>(state->_calib_imu_tg), std::static_pointer_cast<Type>(state->_calib_imu_ACCtoIMU),
                                 std::static_pointer_cast<Type>(state->_calib_imu_GYROtoIMU)})
    if (!calibration || calibration->id() >= 0) return false;
  const auto &options = state->_options;
  if (options.do_calib_camera_readout || options.do_calib_imu_intrinsics || options.do_calib_imu_g_sensitivity ||
      options.num_cameras <= 0 || options.max_pose_clones() <= 0 ||
      result.owners.empty() || result.owners.size() > size_t(options.max_pose_clones()) || !result.graph_node_count ||
      result.calibration.size() != size_t(options.num_cameras) || result.consumed_observations.empty() ||
      !finite(result.accepted_imu_endpoint) || !finite(result.reference_clock_label) || !finite(result.reference_clock_mean) ||
      result.reference_clock_mean != state->cam_imu_dt_ref() ||
      initializer_time_bits(result.reference_clock_label) != initializer_time_bits(result.accepted_imu_endpoint - result.reference_clock_mean) ||
      !finite_matrix(result.imu_mean) || std::abs(result.imu_mean.head<4>().norm() - 1.) > 1e-10 ||
      !finite_matrix(result.imu_accel_map) || !finite_matrix(result.imu_gyro_map) || !finite_matrix(result.imu_tg)) return false;
  const Eigen::Matrix3d accel = state->_calib_imu_ACCtoIMU->Rot() * State::Dm(options.imu_model, state->_calib_imu_da->value());
  const Eigen::Matrix3d gyro = state->_calib_imu_GYROtoIMU->Rot() * State::Dm(options.imu_model, state->_calib_imu_dw->value());
  if (!finite_matrix(accel) || !finite_matrix(gyro) || !finite_matrix(State::Tg(state->_calib_imu_tg->value())) ||
      !accel.isApprox(result.imu_accel_map, 1e-13) || !gyro.isApprox(result.imu_gyro_map, 1e-13) ||
      (State::Tg(state->_calib_imu_tg->value()) - result.imu_tg).cwiseAbs().maxCoeff() > 1e-13) return false;
  for (size_t camera = 0; camera < result.calibration.size(); ++camera) {
    const auto &calibration = result.calibration[camera];
    const auto model = state->_cam_intrinsics_cameras.find(camera);
    const auto extrinsic = state->_calib_IMUtoCAM.find(camera);
    const auto intrinsic = state->_cam_intrinsics.find(camera);
    const auto readout = state->_calib_camera_readout.find(camera);
    const auto clock = state->_calib_dt_CAMtoIMU_map.find(camera);
    if (calibration.camera_id != camera || !finite(calibration.clock_mean) || calibration.clock_mean != state->cam_imu_dt(camera) ||
        !finite_matrix(calibration.extrinsics) || !finite_matrix(calibration.intrinsics) ||
        model == state->_cam_intrinsics_cameras.end() || !model->second || extrinsic == state->_calib_IMUtoCAM.end() || !extrinsic->second ||
        intrinsic == state->_cam_intrinsics.end() || !intrinsic->second || readout == state->_calib_camera_readout.end() || !readout->second ||
        clock == state->_calib_dt_CAMtoIMU_map.end() || !clock->second ||
        (clock->second->id() >= 0) != options.do_calib_camera_timeoffset ||
        (extrinsic->second->id() >= 0) != options.do_calib_camera_pose ||
        (intrinsic->second->id() >= 0) != options.do_calib_camera_intrinsics || readout->second->id() >= 0 ||
        !finite_matrix(readout->second->value()) || readout->second->value()(0) != 0. ||
        !finite_matrix(extrinsic->second->value()) || !finite_matrix(intrinsic->second->value()) ||
        !(calibration.extrinsics.array() == extrinsic->second->value().array()).all() ||
        !(calibration.intrinsics.array() == intrinsic->second->value().array()).all() ||
        !finite_matrix(model->second->get_value()) || !(calibration.intrinsics.array() == model->second->get_value().array()).all() ||
        calibration.fisheye != bool(std::dynamic_pointer_cast<CamEqui>(model->second)) ||
        (!calibration.fisheye && !std::dynamic_pointer_cast<CamRadtan>(model->second)) ||
        calibration.width != model->second->w() || calibration.height != model->second->h()) return false;
  }
  for (size_t i = 0; i < result.consumed_observations.size(); ++i) {
    const auto &key = result.consumed_observations[i];
    if (key.camera_id >= result.calibration.size() || !finite(key.raw_time) ||
        !finite(key.raw_time + result.calibration[key.camera_id].clock_mean) ||
        key.raw_time + result.calibration[key.camera_id].clock_mean > result.accepted_imu_endpoint ||
        (i && !(result.consumed_observations[i-1] < key))) return false;
    if(expected_reset_prior && (!(key.raw_time>expected_reset_prior->raw_watermarks[key.camera_id]) ||
        !(key.raw_time+result.calibration[key.camera_id].clock_mean>expected_reset_prior->imu_endpoint) ||
        key.raw_time+result.calibration[key.camera_id].clock_mean<result.reset_first_imu_time)) return false;
  }
  const int navigation_dimension = 15+6*int(result.owners.size());
  const int dimension = navigation_dimension+calibration_dimension;
  if (result.joint_covariance.rows() != dimension || result.joint_covariance.cols() != dimension ||
      !valid_initial_covariance(result.joint_covariance, true)) return false;
  for (int r=0;r<calibration_dimension;++r)
    for (int c=0;c<calibration_dimension;++c)
      if (initializer_time_bits(result.joint_covariance(navigation_dimension+r,navigation_dimension+c)) !=
          initializer_time_bits(result.calibration_covariance(r,c))) return false;
  const double roundoff = 64. * std::numeric_limits<double>::epsilon() * dimension * result.joint_covariance.cwiseAbs().maxCoeff();
  const auto direct_clock = [&](const InitExposureOwner &owner) {
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(6,calibration_dimension);
    const auto column = clock_columns.find(owner.camera_id);
    if (column != clock_columns.end()) {
      D.block<3,1>(0,column->second) = result.pose_error_scale.head<3>().asDiagonal()*owner.omega_body;
      D.block<3,1>(3,column->second) = result.pose_error_scale.tail<3>().asDiagonal()*owner.velocity_world;
    }
    return D;
  };
  const auto equal_conditional_rows = [&](int first, int second, const Eigen::MatrixXd &difference) {
    const Eigen::MatrixXd error = result.joint_covariance.middleRows(first,6)-result.joint_covariance.middleRows(second,6)-
        difference*result.joint_covariance.bottomRows(calibration_dimension);
    return finite_matrix(error) && error.cwiseAbs().maxCoeff() <= roundoff*(1.+difference.cwiseAbs().sum());
  };
  for (size_t i = 0; i < result.owners.size(); ++i) {
    const auto &owner = result.owners[i];
    if (owner.camera_id >= result.calibration.size() || !finite(owner.raw_time) || !finite(owner.nominal_imu_time) ||
        owner.nominal_imu_time > result.accepted_imu_endpoint || owner.graph_node >= result.graph_node_count ||
        initializer_time_bits(owner.nominal_imu_time) != initializer_time_bits(owner.raw_time + result.calibration[owner.camera_id].clock_mean) ||
        !finite_matrix(owner.pose_mean) || std::abs(owner.pose_mean.head<4>().norm() - 1.) > 1e-10 ||
        !finite_matrix(owner.velocity_world) || !finite_matrix(owner.omega_body)) return false;
    bool has_receipt = false;
    for (const auto &key : result.consumed_observations)
      has_receipt |= key.camera_id == owner.camera_id && initializer_time_bits(key.raw_time) == initializer_time_bits(owner.raw_time);
    if (!has_receipt) return false;
    if (i) {
      const auto &previous = result.owners[i-1];
      if (owner.nominal_imu_time < previous.nominal_imu_time ||
          (owner.nominal_imu_time == previous.nominal_imu_time &&
           (owner.camera_id < previous.camera_id || (owner.camera_id == previous.camera_id &&
            initializer_time_bits(owner.raw_time) <= initializer_time_bits(previous.raw_time))))) return false;
      if (owner.nominal_imu_time == previous.nominal_imu_time) {
        if (owner.graph_node != previous.graph_node || !(owner.pose_mean.array() == previous.pose_mean.array()).all() ||
            !(owner.velocity_world.array() == previous.velocity_world.array()).all() ||
            !(owner.omega_body.array() == previous.omega_body.array()).all() ||
            !equal_conditional_rows(15+6*i,15+6*(i-1),direct_clock(owner)-direct_clock(previous)))
          return false;
      } else if (owner.graph_node <= previous.graph_node) return false;
    }
    if (owner.nominal_imu_time == result.accepted_imu_endpoint &&
        (!(owner.pose_mean.array() == result.imu_mean.head<7>().array()).all() ||
         !(owner.velocity_world.array() == result.imu_mean.segment<3>(7).array()).all() ||
         !equal_conditional_rows(15+6*i,0,direct_clock(owner)))) return false;
  }

  // Stage all allocations and Type identities before committing anything.
  const int old_dimension = state->_Cov.rows();
  std::vector<int> destination;
  destination.reserve(dimension);
  for (int row=0;row<15;++row) destination.push_back(row);
  for (int row=0;row<navigation_dimension-15;++row) destination.push_back(old_dimension+row);
  destination.insert(destination.end(),calibration_rows.begin(),calibration_rows.end());
  Eigen::MatrixXd covariance(dimension,dimension);
  for (int r=0;r<dimension;++r)
    for (int c=0;c<dimension;++c) covariance(destination[r],destination[c])=result.joint_covariance(r,c);
  auto imu = std::make_shared<IMU>();
  imu->set_local_id(0); imu->set_value(result.imu_mean); imu->set_fej(result.imu_mean);
  std::vector<std::shared_ptr<Type>> variables = state->_variables;
  const size_t owner_capacity = size_t(options.max_pose_clones()) + size_t(options.num_cameras);
  variables.reserve(variables.size()+owner_capacity); variables.front()=imu;
  std::vector<State::ExposurePose> owners;
  owners.reserve(owner_capacity);
  for (const auto &owner : result.owners) {
    State::ExposurePose view;
    view.camera_id = owner.camera_id; view.raw_time = owner.raw_time; view.imu_time = owner.nominal_imu_time;
    view.pose = std::make_shared<PoseJPL>();
    view.pose->set_local_id(old_dimension + 6 * owners.size());
    view.pose->set_value(owner.pose_mean); view.pose->set_fej(owner.pose_mean);
    view.kinematics.vel = view.kinematics.vel_fej = owner.velocity_world;
    view.kinematics.omega = view.kinematics.omega_fej = owner.omega_body;
    variables.push_back(view.pose); owners.push_back(std::move(view));
  }
  state->_Cov.swap(covariance); state->_variables.swap(variables); state->_exposure_poses.swap(owners);
  state->_imu = std::move(imu);
  state->_timestamp = result.reference_clock_label;
  state->_imu_endpoint = result.accepted_imu_endpoint;
  state->_imu_endpoint_valid = true;
  state->_initialization_episode_id = result.episode_id;
  return true;
}

bool StateHelper::set_initial_state_warmstart(std::shared_ptr<State> state, const Eigen::MatrixXd &covariance,
                                              const std::map<double, std::shared_ptr<ov_type::PoseJPL>> &clones_IMU) {

  // Contract: `covariance` is the joint, landmark-marginalized navigation covariance in LOCAL error
  // coordinates, ordered [IMU(15) | clone_0(6) | clone_1(6) | ...] with clones ASCENDING in time, which
  // is exactly clones_IMU's (std::map) iteration order. The IMU block is the active state->_imu, already
  // valued by the initializer; the clones are fresh, valued PoseJPL objects not yet in the covariance.
  const int K = (int)clones_IMU.size();
  const int imu_sz = state->_imu->size(); // 15
  const int expected = imu_sz + 6 * K;

  // Defensive gate -- a wrong joint covariance is worse than a cold start, so verify the contract and
  // let the caller fall back to the legacy IMU-only seed on any violation.
  if (K == 0 || K > state->_options.max_pose_clones() || !state->_clones_IMU.empty() || state->uses_physical_clones() ||
      state->_imu->id() != 0 || covariance.rows() != expected || covariance.cols() != expected) {
    PRINT_ERROR(RED "StateHelper::set_initial_state_warmstart() - contract violated (imu_id=%d, cov=%ldx%ld, expected %d, K=%d)\n" RESET,
                state->_imu->id(), (long)covariance.rows(), (long)covariance.cols(), expected, K);
    return false;
  }
  const auto finite = [](double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
  };
  for (auto it = clones_IMU.begin(); it != clones_IMU.end(); ++it) {
    const auto &cp = *it;
    if (!finite(cp.first) || cp.second == nullptr || cp.second->id() >= 0 ||
        state->_clones_IMU.find(cp.first) != state->_clones_IMU.end()) {
      PRINT_ERROR(RED "StateHelper::set_initial_state_warmstart() - clone @ %.6f null or already in state\n" RESET, cp.first);
      return false;
    }
    for (auto previous = clones_IMU.begin(); previous != it; ++previous)
      if (previous->second == cp.second)
        return false; // two covariance blocks cannot own the same mutable Type
    const auto value = cp.second->value();
    const auto fej = cp.second->fej();
    for (int j = 0; j < value.size(); ++j)
      if (!finite(value(j)) || !finite(fej(j)))
        return false;
  }

  // Reject before mutating ownership. The caller must separately validate the
  // IMU marginal before taking a cold fallback; rejecting this joint alone is
  // not permission to inject a malformed top-left block.
  if (!valid_initial_covariance(covariance))
    return false;

  // Grow the covariance ONCE for all K clones (single reallocation; the new rows/cols are zero-filled,
  // i.e. clones start uncorrelated with the existing calibration blocks -- the same block-diagonal
  // assumption set_initial_covariance() makes for the IMU). Register each clone (id / _variables /
  // _clones_IMU) and build the [imu, clones-ascending] ordering used to place the joint covariance.
  std::vector<std::shared_ptr<Type>> order;
  order.reserve(1 + K);
  order.push_back(state->_imu);
  const int old_size = (int)state->_Cov.rows();
  state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(old_size + 6 * K, old_size + 6 * K));
  int loc = old_size;
  for (auto const &cp : clones_IMU) { // ascending time -> matches the clone-block order in `covariance`
    std::shared_ptr<PoseJPL> pose = cp.second;
    pose->set_local_id(loc);
    state->_variables.push_back(pose);
    state->_clones_IMU[cp.first] = pose;
    order.push_back(pose);
    loc += pose->size(); // 6
  }

  // Per-clone kinematics for the restored window, recovered ON THE MANIFOLD from the injected
  // trajectory (these are state-quantity estimates feeding linearization points -- the analytic-
  // Jacobian policy governs H matrices, which stay closed-form; it does not forbid geodesic rates).
  // JPL convention R_GtoI(t+dt) = exp_so3(-w dt) R_GtoI(t)  ==>  the right-trivialized geodesic rate
  //   w_k = -log_so3(R_{k+1} R_k^T) / dt
  // is EXACT under the piecewise-constant-omega assumption the whole integrator stack already makes.
  // Velocity uses the interval mean (secant) on flat R^3 -- exact mean velocity by the MVT, second-
  // order accurate at the midpoint; the NEWEST clone instead takes the injected IMU velocity, which
  // is exact. FEJ twins equal the values: a (re)init is a fresh linearization point. (Optional later
  // refinement: forward the initializer's per-node MAP velocities and gyro-from-history omegas.)
  if (clones_IMU.size() >= 2) {
    std::vector<double> ts;
    ts.reserve(clones_IMU.size());
    for (auto const &cp : clones_IMU)
      ts.push_back(cp.first);
    for (size_t k = 0; k < ts.size(); k++) {
      const size_t kp = (k + 1 < ts.size()) ? k + 1 : k; // forward neighbor (self at end)
      const size_t km = (k > 0) ? k - 1 : k;             // backward neighbor (self at start)
      const size_t kw = (k + 1 < ts.size()) ? k : k - 1; // omega interval start (forward; one-sided at end)
      State::CloneKinematics kin;
      const double dt_v = ts.at(kp) - ts.at(km);
      if (dt_v > 0) {
        kin.vel = (clones_IMU.at(ts.at(kp))->pos() - clones_IMU.at(ts.at(km))->pos()) / dt_v;
      }
      const double dt_w = ts.at(kw + 1) - ts.at(kw);
      if (dt_w > 0) {
        kin.omega = -ov_core::log_so3(clones_IMU.at(ts.at(kw + 1))->Rot() * clones_IMU.at(ts.at(kw))->Rot().transpose()) / dt_w;
      }
      if (k + 1 == ts.size()) {
        kin.vel = state->_imu->vel(); // newest clone == injected IMU state: exact velocity
      }
      kin.vel_fej = kin.vel;
      kin.omega_fej = kin.omega;
      state->_clones_kinematics[ts.at(k)] = kin;
    }
  } else if (clones_IMU.size() == 1) {
    State::CloneKinematics kin;
    kin.vel = state->_imu->vel();
    kin.vel_fej = kin.vel;
    state->_clones_kinematics[clones_IMU.begin()->first] = kin;
  }

  // Copy the joint covariance (diagonal blocks AND all cross-terms) into the state covariance, mapping
  // contiguous source offsets to each variable's id. Identical block-copy to set_initial_covariance.
  int i_index = 0;
  for (size_t i = 0; i < order.size(); i++) {
    int k_index = 0;
    for (size_t k = 0; k < order.size(); k++) {
      state->_Cov.block(order[i]->id(), order[k]->id(), order[i]->size(), order[k]->size()) =
          covariance.block(i_index, k_index, order[i]->size(), order[k]->size());
      k_index += order[k]->size();
    }
    i_index += order[i]->size();
  }
  state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
  return true;
}

Eigen::MatrixXd StateHelper::get_marginal_covariance(std::shared_ptr<State> state,
                                                     const std::vector<std::shared_ptr<Type>> &small_variables) {

  // Calculate the marginal covariance size we need to make our matrix
  int cov_size = 0;
  for (size_t i = 0; i < small_variables.size(); i++) {
    cov_size += small_variables[i]->size();
  }

  // Construct our return covariance
  Eigen::MatrixXd Small_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

  // For each variable, lets copy over all other variable cross terms
  // Note: this copies over itself to when i_index=k_index
  int i_index = 0;
  for (size_t i = 0; i < small_variables.size(); i++) {
    int k_index = 0;
    for (size_t k = 0; k < small_variables.size(); k++) {
      Small_cov.block(i_index, k_index, small_variables[i]->size(), small_variables[k]->size()) =
          state->_Cov.block(small_variables[i]->id(), small_variables[k]->id(), small_variables[i]->size(), small_variables[k]->size());
      k_index += small_variables[k]->size();
    }
    i_index += small_variables[i]->size();
  }

  // Return the covariance
  // Small_cov = 0.5*(Small_cov+Small_cov.transpose());
  return Small_cov;
}

Eigen::MatrixXd StateHelper::get_full_covariance(std::shared_ptr<State> state) {

  // Size of the covariance is the active
  int cov_size = (int)state->_Cov.rows();

  // Construct our return covariance
  Eigen::MatrixXd full_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

  // Copy in the active state elements
  full_cov.block(0, 0, state->_Cov.rows(), state->_Cov.rows()) = state->_Cov;

  // Return the covariance
  return full_cov;
}

bool StateHelper::project_sampled_imu_output(std::shared_ptr<State> state, const Eigen::Matrix<double, 12, 27> &H,
                                            Eigen::Matrix<double, 12, 12> &covariance, Eigen::MatrixXd *state_cross) {
  if (!state || !state->has_sampled_imu_boundary() || !ov_core::numeric::finite_matrix(H)) return false;
  const int n = state->_Cov.rows();
  if (n < 27 || state->_Cov.cols() != n ||
      (state_cross && (state_cross->rows() != n || state_cross->cols() != 12))) return false;
  const std::array<std::shared_ptr<Type>, 3> variables{state->_imu, state->_sampled_imu_slots[0].noise,
                                                     state->_sampled_imu_slots[1].noise};
  constexpr std::array<int, 3> sizes{15, 6, 6}, columns{0, 15, 21};
  for (size_t i = 0; i < variables.size(); ++i) {
    const auto &var = variables[i];
    if (!var || var->size() != sizes[i] || var->id() < 0 || var->id() > n - sizes[i]) return false;
    const auto found = std::lower_bound(state->_variables.begin(), state->_variables.end(), var->id(),
        [](const std::shared_ptr<Type> &owner, int id) { return owner->id() < id; });
    if (found == state->_variables.end() || *found != var) return false;
    for (size_t j = 0; j < i; ++j) if (variables[j] == var) return false;
    if (i && !state->_sampled_imu_slots[i-1].active && !H.middleCols<6>(columns[i]).isZero(0.)) return false;
  }
  Eigen::Matrix<double, 27, 27> local;
  for (size_t i = 0; i < variables.size(); ++i)
    for (size_t j = 0; j < variables.size(); ++j)
      local.block(columns[i], columns[j], sizes[i], sizes[j]) =
          state->_Cov.block(variables[i]->id(), variables[j]->id(), sizes[i], sizes[j]);
  if (!ov_core::numeric::finite_matrix(local)) return false;
  Eigen::Matrix<double, 12, 12> staged_covariance = H * local * H.transpose();
  staged_covariance = (0.5 * (staged_covariance + staged_covariance.transpose())).eval();
  if (!ov_core::numeric::finite_matrix(staged_covariance)) return false;
  // State already owns a valid prior. This is its linear projection, not a
  // second posterior or an independent output-noise model. Exact constraints
  // can make it singular; preserve roundoff without adding jitter or R.
  Eigen::MatrixXd staged_cross;
  if (state_cross) {
    staged_cross.resize(n, 12);
    staged_cross.noalias() = state->_Cov.middleCols(variables[0]->id(), 15) * H.leftCols<15>().transpose();
    for (size_t i = 1; i < variables.size(); ++i)
      staged_cross.noalias() += state->_Cov.middleCols(variables[i]->id(), 6) * H.middleCols<6>(columns[i]).transpose();
    if (!ov_core::numeric::finite_matrix(staged_cross)) return false;
  }
  covariance = staged_covariance;
  if (state_cross) *state_cross = staged_cross; // existing exact shape, no resize
  return true;
}

void StateHelper::marginalize(std::shared_ptr<State> state, std::shared_ptr<Type> marg) {

  for (const auto &slot : state->_sampled_imu_slots)
    if (slot.noise && slot.noise == marg)
      throw std::invalid_argument("sampled IMU slots must retire at the exact successor raw knot");

  // Check if the current state has the element we want to marginalize
  if (std::find(state->_variables.begin(), state->_variables.end(), marg) == state->_variables.end()) {
    PRINT_ERROR(RED "StateHelper::marginalize() - Called on variable that is not in the state\n" RESET);
    PRINT_ERROR(RED "StateHelper::marginalize() - Marginalization, does NOT work on sub-variables yet...\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Generic covariance has this form for x_1, x_m, x_2. If we want to remove x_m:
  //
  //  P_(x_1,x_1) P(x_1,x_m) P(x_1,x_2)
  //  P_(x_m,x_1) P(x_m,x_m) P(x_m,x_2)
  //  P_(x_2,x_1) P(x_2,x_m) P(x_2,x_2)
  //
  //  to
  //
  //  P_(x_1,x_1) P(x_1,x_2)
  //  P_(x_2,x_1) P(x_2,x_2)
  //
  // i.e. x_1 goes from 0 to marg_id, x_2 goes from marg_id+marg_size to Cov.rows() in the original covariance

  int marg_size = marg->size();
  int marg_id = marg->id();
  int x2_size = (int)state->_Cov.rows() - marg_id - marg_size;

  Eigen::MatrixXd Cov_new(state->_Cov.rows() - marg_size, state->_Cov.rows() - marg_size);

  // P_(x_1,x_1)
  Cov_new.block(0, 0, marg_id, marg_id) = state->_Cov.block(0, 0, marg_id, marg_id);

  // P_(x_1,x_2)
  Cov_new.block(0, marg_id, marg_id, x2_size) = state->_Cov.block(0, marg_id + marg_size, marg_id, x2_size);

  // P_(x_2,x_1)
  Cov_new.block(marg_id, 0, x2_size, marg_id) = Cov_new.block(0, marg_id, marg_id, x2_size).transpose();

  // P(x_2,x_2)
  Cov_new.block(marg_id, marg_id, x2_size, x2_size) = state->_Cov.block(marg_id + marg_size, marg_id + marg_size, x2_size, x2_size);

  // Now set new covariance
  // state->_Cov.resize(Cov_new.rows(),Cov_new.cols());
  state->_Cov = Cov_new;
  // state->Cov() = 0.5*(Cov_new+Cov_new.transpose());
  assert(state->_Cov.rows() == Cov_new.rows());

  // Now we keep the remaining variables and update their ordering
  // Note: DOES NOT SUPPORT MARGINALIZING SUBVARIABLES YET!!!!!!!
  std::vector<std::shared_ptr<Type>> remaining_variables;
  for (size_t i = 0; i < state->_variables.size(); i++) {
    // Only keep non-marginal states
    if (state->_variables.at(i) != marg) {
      if (state->_variables.at(i)->id() > marg_id) {
        // If the variable is "beyond" the marginal one in ordering, need to "move it forward"
        state->_variables.at(i)->set_local_id(state->_variables.at(i)->id() - marg_size);
      }
      remaining_variables.push_back(state->_variables.at(i));
    }
  }

  // Delete the old state variable to free up its memory
  // NOTE: we don't need to do this any more since our variable is a shared ptr
  // NOTE: thus this is automatically managed, but this allows outside references to keep the old variable
  // delete marg;
  marg->set_local_id(-1);

  // Now set variables as the remaining ones
  state->_variables = remaining_variables;
}

std::shared_ptr<Type> StateHelper::clone(std::shared_ptr<State> state, std::shared_ptr<Type> variable_to_clone) {

  // Get total size of new cloned variables, and the old covariance size
  int total_size = variable_to_clone->size();
  int old_size = (int)state->_Cov.rows();
  int new_loc = (int)state->_Cov.rows();

  // Resize both our covariance to the new size
  state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(old_size + total_size, old_size + total_size));

  // What is the new state, and variable we inserted
  const std::vector<std::shared_ptr<Type>> new_variables = state->_variables;
  std::shared_ptr<Type> new_clone = nullptr;

  // Loop through all variables, and find the variable that we are going to clone
  for (size_t k = 0; k < state->_variables.size(); k++) {

    // Skip this if it is not the same
    // First check if the top level variable is the same, then check the sub-variables
    std::shared_ptr<Type> type_check = state->_variables.at(k)->check_if_subvariable(variable_to_clone);
    if (state->_variables.at(k) == variable_to_clone) {
      type_check = state->_variables.at(k);
    } else if (type_check != variable_to_clone) {
      continue;
    }

    // So we will clone this one
    int old_loc = type_check->id();

    // Copy the covariance elements
    state->_Cov.block(new_loc, new_loc, total_size, total_size) = state->_Cov.block(old_loc, old_loc, total_size, total_size);
    state->_Cov.block(0, new_loc, old_size, total_size) = state->_Cov.block(0, old_loc, old_size, total_size);
    state->_Cov.block(new_loc, 0, total_size, old_size) = state->_Cov.block(old_loc, 0, total_size, old_size);

    // Create clone from the type being cloned
    new_clone = type_check->clone();
    new_clone->set_local_id(new_loc);
    break;
  }

  // Check if the current state has this variable
  if (new_clone == nullptr) {
    PRINT_ERROR(RED "StateHelper::clone() - Called on variable is not in the state\n" RESET);
    PRINT_ERROR(RED "StateHelper::clone() - Ensure that the variable specified is a variable, or sub-variable..\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Add to variable list and return
  state->_variables.push_back(new_clone);
  return new_clone;
}

bool StateHelper::valid_initialization_order(const std::shared_ptr<State> &state,
    const std::shared_ptr<Type> &proposal,
    const std::vector<std::shared_ptr<Type>> &order, int columns) {
  if (!state || !proposal || proposal->id() != -1 || proposal->size() <= 0 ||
      proposal->value().cols() != 1 || proposal->value().rows() <= 0 ||
      proposal->fej().rows() != proposal->value().rows() || proposal->fej().cols() != 1 ||
      !ov_core::numeric::finite_matrix(proposal->value()) ||
      !ov_core::numeric::finite_matrix(proposal->fej()) || columns < 0)
    return false;
  const int n = state->_Cov.rows();
  if (state->_Cov.cols() != n) return false;
  int count = 0;
  for (const auto &owner : state->_variables) {
    if (!owner || owner == proposal || owner->id() != count || owner->size() <= 0 || owner->size() > n-count)
      return false;
    count += owner->size();
  }
  if (count != n) return false;
  count = 0;
  for (const auto &var : order) {
    if (!var || var->id() < 0 || var->size() <= 0 || var->id() > n || var->size() > n-var->id() ||
        var->size() > columns-count) return false;
    const auto after = std::upper_bound(state->_variables.begin(), state->_variables.end(), var->id(),
        [](int id, const std::shared_ptr<Type> &owner) { return id < owner->id(); });
    if (after == state->_variables.begin()) return false;
    const auto &owner = *std::prev(after);
    if (var->id()+var->size() > owner->id()+owner->size() ||
        (owner != var && owner->check_if_subvariable(var) != var)) return false;
    count += var->size();
  }
  return count == columns;
}

bool StateHelper::initialize(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                             const std::vector<std::shared_ptr<Type>> &H_order, Eigen::MatrixXd &H_R, Eigen::MatrixXd &H_L,
                             Eigen::MatrixXd &R, Eigen::VectorXd &res, double chi_2_mult) {

  // Reject invalid data before in-place QR or any state augmentation. This
  // also covers exactly determined systems with no residual left to gate.
  if (!valid_initialization_order(state, new_variable, H_order, H_R.cols()) || R.rows() == 0 || R.cols() != R.rows() || res.rows() != R.rows() ||
      H_R.rows() != res.rows() || H_L.rows() != res.rows() ||
      H_L.cols() != new_variable->size() || H_L.rows() < H_L.cols() ||
      !ov_core::numeric::finite_matrix(R) || !(R(0, 0) > 0.0) ||
      !ov_core::numeric::finite_matrix(H_R) || !ov_core::numeric::finite_matrix(H_L) ||
      !ov_core::numeric::finite_matrix(res) || !valid_innovation_limit(chi_2_mult))
    return false;

  // Check that we have isotropic noise (i.e. is diagonal and all the same value)
  // TODO: can we simplify this so it doesn't take as much time?
  assert(R.rows() == R.cols());
  assert(R.rows() > 0);
  for (int r = 0; r < R.rows(); r++) {
    for (int c = 0; c < R.cols(); c++) {
      if (r == c && R(0, 0) != R(r, c)) {
        return false;
      } else if (r != c && R(r, c) != 0.0) {
        return false;
      }
    }
  }

  //==========================================================
  //==========================================================
  // First we perform QR givens to seperate the system
  // The top will be a system that depends on the new state, while the bottom does not
  size_t new_var_size = new_variable->size();
  assert((int)new_var_size == H_L.cols());

  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_L.cols(); ++n) {
    for (int m = (int)H_L.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_L(m - 1, n), H_L(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_L.block(m - 1, n, 2, H_L.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_R.block(m - 1, 0, 2, H_R.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // Separate into initializing and updating portions
  // 1. Invertible initializing system
  Eigen::MatrixXd Hxinit = H_R.block(0, 0, new_var_size, H_R.cols());
  Eigen::MatrixXd H_finit = H_L.block(0, 0, new_var_size, new_var_size);
  Eigen::VectorXd resinit = res.block(0, 0, new_var_size, 1);
  Eigen::MatrixXd Rinit = R.block(0, 0, new_var_size, new_var_size);

  // 2. Nullspace projected updating system
  Eigen::MatrixXd Hup = H_R.block(new_var_size, 0, H_R.rows() - new_var_size, H_R.cols());
  Eigen::VectorXd resup = res.block(new_var_size, 0, res.rows() - new_var_size, 1);
  Eigen::MatrixXd Rup = R.block(new_var_size, new_var_size, R.rows() - new_var_size, R.rows() - new_var_size);

  //==========================================================
  //==========================================================

  // The new variable has no prior: its fitted directions have been removed,
  // leaving only resup.rows() degrees of freedom for a consistency test. The
  // existing-state prior enters S, but does not restore those fitted directions.
  // An exactly determined initialization has no consistency residual to gate.
  if (resup.rows() > 0) {
    Eigen::MatrixXd P_up = get_marginal_covariance(state, H_order);
    assert(Rup.rows() == Hup.rows());
    assert(Hup.cols() == P_up.cols());
    Eigen::MatrixXd S = Hup * P_up * Hup.transpose() + Rup;
    double chi2;
    const bool valid_innovation = innovation_chi2(S, resup, chi2);

    double chi2_check = ov_core::chi_squared_quantile_0_95((int)resup.rows());
    const double chi2_limit = chi_2_mult * chi2_check;
    if (!valid_innovation || !valid_innovation_limit(chi2_limit) || chi2 > chi2_limit) {
      return false;
    }
  }

  //==========================================================
  //==========================================================
  // Finally, initialize it in our state. If the remaining update fails,
  // remove only the just-added block: its old principal covariance was never
  // changed, and EKFUpdate itself rejects before any mutation.
  const int old_size = state->_Cov.rows();
  const Eigen::MatrixXd old_value = new_variable->value();
  if (!StateHelper::initialize_invertible(state, new_variable, H_order, Hxinit, H_finit, Rinit, resinit))
    return false;
  if (Hup.rows() > 0 && !StateHelper::EKFUpdate(state, H_order, Hup, resup, Rup)) {
    state->_Cov.conservativeResize(old_size, old_size);
    state->_variables.pop_back();
    new_variable->set_local_id(-1);
    new_variable->set_value(old_value);
    return false;
  }
  return true;
}

bool StateHelper::initialize_invertible(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                                        const std::vector<std::shared_ptr<Type>> &H_order, const Eigen::MatrixXd &H_R,
                                        const Eigen::MatrixXd &H_L, const Eigen::MatrixXd &R, const Eigen::VectorXd &res) {

  if (!valid_initialization_order(state, new_variable, H_order, H_R.cols()) || R.rows() == 0 || R.cols() != R.rows() || res.rows() != R.rows() ||
      H_R.rows() != res.rows() || H_L.rows() != res.rows() || H_L.cols() != res.rows() ||
      H_L.cols() != new_variable->size() || !ov_core::numeric::finite_matrix(R) || !(R(0,0) > 0.0) ||
      !ov_core::numeric::finite_matrix(H_R) || !ov_core::numeric::finite_matrix(H_L) ||
      !ov_core::numeric::finite_matrix(res))
    return false;
  // Numerical rank uses Eigen's relative machine-precision criterion, not a
  // new feature-quality threshold. A small residual cannot certify an
  // unobservable landmark direction, including an exactly determined system.
  if (!H_L.fullPivLu().isInvertible())
    return false;

  // Check that we have isotropic noise (i.e. is diagonal and all the same value)
  // TODO: can we simplify this so it doesn't take as much time?
  assert(R.rows() == R.cols());
  assert(R.rows() > 0);
  for (int r = 0; r < R.rows(); r++) {
    for (int c = 0; c < R.cols(); c++) {
      if (r == c && R(0, 0) != R(r, c)) {
        return false;
      } else if (r != c && R(r, c) != 0.0) {
        return false;
      }
    }
  }

  //==========================================================
  //==========================================================
  // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
  assert(res.rows() == R.rows());
  assert(H_L.rows() == res.rows());
  assert(H_L.rows() == H_R.rows());
  Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());

  // Get the location in small jacobian for each measuring variable
  int current_it = 0;
  std::vector<int> H_id;
  for (const auto &meas_var : H_order) {
    H_id.push_back(current_it);
    current_it += meas_var->size();
  }

  //==========================================================
  //==========================================================
  // For each active variable find its M = P*H^T
  for (const auto &var : state->_variables) {
    // Sum up effect of each subjacobian= K_i= \sum_m (P_im Hm^T)
    Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
    for (size_t i = 0; i < H_order.size(); i++) {
      std::shared_ptr<Type> meas_var = H_order.at(i);
      M_i += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
             H_R.block(0, H_id[i], H_R.rows(), meas_var->size()).transpose();
    }
    M_a.block(var->id(), 0, var->size(), res.rows()) = M_i;
  }

  //==========================================================
  //==========================================================
  // Get covariance of this small jacobian
  Eigen::MatrixXd P_small = StateHelper::get_marginal_covariance(state, H_order);

  // M = H_R*Cov*H_R' + R
  Eigen::MatrixXd M(H_R.rows(), H_R.rows());
  M.triangularView<Eigen::Upper>() = H_R * P_small * H_R.transpose();
  M.triangularView<Eigen::Upper>() += R;

  // Covariance of the variable/landmark that will be initialized
  assert(H_L.rows() == H_L.cols());
  assert(H_L.rows() == new_variable->size());
  Eigen::MatrixXd H_Linv = H_L.inverse();
  if (!ov_core::numeric::finite_matrix(H_Linv))
    return false;
  Eigen::MatrixXd P_LL = H_Linv * M.selfadjointView<Eigen::Upper>() * H_Linv.transpose();
  Eigen::MatrixXd P_xL = -M_a * H_Linv.transpose();
  const Eigen::VectorXd delta = H_Linv * res;
  if (!valid_initial_covariance(P_LL) || !ov_core::numeric::finite_matrix(P_xL) || !ov_core::numeric::finite_matrix(delta))
    return false;

  // Evaluate the small proposed variable off-state. Invalid nonlinear
  // retraction or an overflowing vector sum must not partially augment the
  // covariance, mutate the caller's variable, or allocate it a live state ID.
  const auto trial_variable = new_variable->clone();
  trial_variable->update(delta);
  if (!ov_core::numeric::finite_matrix(trial_variable->value()))
    return false;

  // Augment the covariance matrix
  size_t oldSize = state->_Cov.rows();
  state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(oldSize + new_variable->size(), oldSize + new_variable->size()));
  state->_Cov.block(0, oldSize, oldSize, new_variable->size()) = P_xL;
  state->_Cov.block(oldSize, 0, new_variable->size(), oldSize) = state->_Cov.block(0, oldSize, oldSize, new_variable->size()).transpose();
  state->_Cov.block(oldSize, oldSize, new_variable->size(), new_variable->size()) = P_LL;

  // Update the variable that will be initialized (invertible systems can only update the new variable).
  // However this update should be almost zero if we already used a conditional Gauss-Newton to solve for the initial estimate
  new_variable->set_value(trial_variable->value());

  // Now collect results, and add it to the state variables
  new_variable->set_local_id(oldSize);
  state->_variables.push_back(new_variable);
  return true;

  // std::stringstream ss;
  // ss << new_variable->id() <<  " init dx = " << (H_Linv * res).transpose() << std::endl;
  // PRINT_DEBUG(ss.str().c_str());
}

void StateHelper::augment_clone(std::shared_ptr<State> state, Eigen::Matrix<double, 3, 1> last_w,
                                Eigen::Matrix<double, 3, 1> last_w_fej) {

  // We can't insert a clone that occured at the same timestamp!
  if (state->_clones_IMU.find(state->_timestamp) != state->_clones_IMU.end()) {
    PRINT_ERROR(RED "TRIED TO INSERT A CLONE AT THE SAME TIME AS AN EXISTING CLONE, EXITING!#!@#!@#\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  const auto pose = StateHelper::augment_pose_view(state, static_cast<size_t>(state->cam_imu_dt_ref_camid()), last_w);

  // Append the new clone to our clone vector
  state->_clones_IMU[state->_timestamp] = pose;

  // Store velocity and angular rate at clone time (value + FEJ linearizations): consumed by the
  // per-camera time-offset / rolling-shutter measurement models
  State::CloneKinematics clone_kin;
  clone_kin.vel = state->_imu->vel();
  clone_kin.omega = last_w;
  clone_kin.vel_fej = state->_imu->vel_fej();
  clone_kin.omega_fej = last_w_fej;
  state->_clones_kinematics[state->_timestamp] = clone_kin;
}

std::shared_ptr<PoseJPL> StateHelper::augment_pose_view(std::shared_ptr<State> state, size_t clock_cam_id,
                                                      const Eigen::Vector3d &omega) {
  // Validate/resolve the owner before adding any covariance variable. Fixed
  // clocks have id == -1; they do not contribute stochastic clock columns.
  const std::shared_ptr<Vec> clock = state->_calib_dt_CAMtoIMU_map.at(clock_cam_id);
  const auto pose = std::dynamic_pointer_cast<PoseJPL>(StateHelper::clone(state, state->_imu->pose()));
  if (pose == nullptr) {
    PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
    std::exit(EXIT_FAILURE);
  }
  if (clock && clock->id() >= 0) {
    Eigen::Matrix<double, 6, 1> dnc_dt = Eigen::MatrixXd::Zero(6, 1);
    dnc_dt.block(0, 0, 3, 1) = omega;
    dnc_dt.block(3, 0, 3, 1) = state->_imu->vel();
    // Sparse congruence by A = E_pose + [omega; velocity] e_owner_clock^T.
    // The second write deliberately reads the updated clock/pose column, so
    // the clock variance and all existing owner-view cross terms appear once.
    state->_Cov.block(0, pose->id(), state->_Cov.rows(), 6) +=
        state->_Cov.block(0, clock->id(), state->_Cov.rows(), 1) * dnc_dt.transpose();
    state->_Cov.block(pose->id(), 0, 6, state->_Cov.rows()) +=
        dnc_dt * state->_Cov.block(clock->id(), 0, 1, state->_Cov.rows());
  }
  return pose;
}

void StateHelper::marginalize_old_clone(std::shared_ptr<State> state) {
  if (state->uses_physical_clones()) {
    if (state->_exposure_poses.size() > static_cast<size_t>(state->_options.max_pose_clones())) {
      std::lock_guard<std::mutex> lock(state->_mutex_state);
      // The visual owner must already have reanchored or removed landmarks
      // using this exposure. Equal-time owners are distinct covariance Types.
      StateHelper::marginalize(state, state->_exposure_poses.front().pose);
      state->_exposure_poses.erase(state->_exposure_poses.begin());
    }
    return;
  }
  if ((int)state->_clones_IMU.size() > state->_options.max_pose_clones()) {
    double marginal_time = state->margtimestep();
    // Lock the mutex to avoid deleting any elements from _clones_IMU while accessing it from other threads
    std::lock_guard<std::mutex> lock(state->_mutex_state);
    assert(marginal_time != INFINITY);
    StateHelper::marginalize(state, state->_clones_IMU.at(marginal_time));
    // Note that the marginalizer should have already deleted the clone
    // Thus we just need to remove the pointer to it from our state
    state->_clones_IMU.erase(marginal_time);
    state->_clones_kinematics.erase(marginal_time);
    state->_epoch_residuals.erase(marginal_time);
    state->_epoch_bridges.erase(marginal_time);
  }
}

void StateHelper::marginalize_slam(std::shared_ptr<State> state) {
  // Remove SLAM features that have their marginalization flag set
  // We also check that we do not remove any aruoctag landmarks
  int ct_marginalized = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((*it0).second->should_marg && (int)(*it0).first > 4 * state->_options.max_aruco_features) {
      StateHelper::marginalize(state, (*it0).second);
      it0 = state->_features_SLAM.erase(it0);
      ct_marginalized++;
    } else {
      it0++;
    }
  }
}
