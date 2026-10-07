/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "SampledCpiStatistics.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <Eigen/Eigenvalues>

namespace ov_init {
namespace {
uint64_t bits(double value) {
  uint64_t result;
  static_assert(sizeof(result) == sizeof(value), "IEEE binary64 required");
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

bool finite(double value) {
  return (bits(value) & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

template<class Derived> bool finite(const Eigen::MatrixBase<Derived> &value) {
  for (int column = 0; column < value.cols(); ++column)
    for (int row = 0; row < value.rows(); ++row)
      if (!finite(value(row, column))) return false;
  return true;
}

// Check in dimensionless correlation coordinates so mixed sensor/state units
// cannot hide an invalid small-variance block. Singular PSD needs no jitter.
template<int N> bool covariance_valid(const Eigen::Matrix<double, N, N> &value) {
  if (!finite(value)) return false;
  Eigen::Matrix<double, N, 1> scale;
  for (int row = 0; row < N; ++row) {
    if (value(row, row) < 0.) return false;
    scale(row) = std::sqrt(value(row, row));
  }
  Eigen::Matrix<double, N, N> normalized = Eigen::Matrix<double, N, N>::Zero();
  const double tolerance = 128. * N * std::numeric_limits<double>::epsilon();
  for (int column = 0; column < N; ++column) {
    for (int row = 0; row < N; ++row) {
      if (scale(row) == 0. || scale(column) == 0.) {
        if (value(row, column) != 0.) return false;
      } else {
        normalized(row, column) = value(row, column) / scale(row) / scale(column);
        if (!finite(normalized(row, column)) || std::abs(normalized(row, column)) > 1. + tolerance) return false;
      }
    }
  }
  if ((normalized - normalized.transpose()).cwiseAbs().maxCoeff() > tolerance) return false;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> spectrum(
      normalized.template selfadjointView<Eigen::Upper>(), Eigen::EigenvaluesOnly);
  return spectrum.info() == Eigen::Success && finite(spectrum.eigenvalues()) &&
         spectrum.eigenvalues().minCoeff() >= -tolerance;
}

using Statistics = SampledCpiStatistics;
using Record = Statistics::Record;
using Output = Statistics::Output;

bool record_valid(const Record &record) {
  return record.stream_episode && record.sequence && finite(record.timestamp) &&
         finite(record.measured) && finite(record.noise_linearization) && covariance_valid(record.prior);
}

bool same_record(const Record &a, const Record &b) {
  return a.stream_episode == b.stream_episode && a.sequence == b.sequence && bits(a.timestamp) == bits(b.timestamp) &&
         std::memcmp(a.measured.data(), b.measured.data(), sizeof(double) * 6) == 0 &&
         std::memcmp(a.prior.data(), b.prior.data(), sizeof(double) * 36) == 0 &&
         std::memcmp(a.noise_linearization.data(), b.noise_linearization.data(), sizeof(double) * 6) == 0;
}

bool weights(const std::array<Record, 2> &records, double time, Eigen::Vector2d &out) {
  const double span = records[1].timestamp - records[0].timestamp;
  if (!finite(time) || !finite(span) || !(span > 0.) || time < records[0].timestamp || time > records[1].timestamp)
    return false;
  if (time == records[0].timestamp) out << 1., 0.;
  else if (time == records[1].timestamp) out << 0., 1.;
  else {
    const double alpha = (time - records[0].timestamp) / span;
    out << 1. - alpha, alpha;
  }
  return finite(out) && (out.array() >= 0.).all() && (out.array() <= 1.).all();
}

int find(const Output &out, uint64_t sequence) {
  for (unsigned i = 0; i < out.owner_count; ++i)
    if (out.owners[i].record.sequence == sequence) return static_cast<int>(i);
  return -1;
}
} // namespace

bool SampledCpiStatistics::append(const Step &step) {
  Eigen::Vector2d w0, w1;
  if (!record_valid(step.records[0]) || !record_valid(step.records[1]) ||
      step.records[0].stream_episode != step.records[1].stream_episode ||
      !(step.records[1].sequence > step.records[0].sequence) ||
      !finite(step.time1 - step.time0) || !(step.time1 > step.time0) ||
      !weights(step.records, step.time0, w0) || !weights(step.records, step.time1, w1) ||
      !finite(step.weights0) || !finite(step.weights1) ||
      (w0.array() != step.weights0.array()).any() || (w1.array() != step.weights1.array()).any() ||
      !finite(step.transition) || !finite(step.noise[0]) || !finite(step.noise[1]) ||
      !covariance_valid(step.independent_covariance) || statistics_.steps == std::numeric_limits<uint64_t>::max())
    return false;

  if (statistics_.steps) {
    if (bits(step.time0) != bits(statistics_.time1)) return false;
    const int right = find(statistics_, right_sequence_);
    if (right < 0) return false;
    if (bits(statistics_.time1) == bits(statistics_.owners[right].record.timestamp)) {
      // Only the next adjacent support is legal after a raw knot. The previous
      // left record cannot be referenced again, even if it remains start-pinned.
      if (!same_record(step.records[0], statistics_.owners[right].record) ||
          !(step.records[1].sequence > right_sequence_)) return false;
    } else {
      const int left = find(statistics_, left_sequence_);
      if (left < 0 || !same_record(step.records[0], statistics_.owners[left].record) ||
          !same_record(step.records[1], statistics_.owners[right].record)) return false;
    }
  }

  Output staged = statistics_;
  if (!staged.steps) staged.time0 = step.time0;
  int current[2];
  for (int record = 0; record < 2; ++record) {
    current[record] = find(staged, step.records[record].sequence);
    if (current[record] < 0) {
      if (staged.owner_count == staged.owners.size()) return false;
      current[record] = static_cast<int>(staged.owner_count++);
      staged.owners[current[record]].record = step.records[record];
    } else if (!same_record(staged.owners[current[record]].record, step.records[record])) {
      return false;
    }
    if (!staged.steps && w0(record) > 0.) staged.owners[current[record]].start_support = true;
  }

  staged.transition = (step.transition * staged.transition).eval();
  staged.offset = (step.transition * staged.offset).eval();
  const Matrix15 independent = step.independent_covariance.selfadjointView<Eigen::Upper>();
  staged.conditional_covariance = (step.transition * staged.conditional_covariance * step.transition.transpose() +
                                 independent).eval();
  for (unsigned i = 0; i < staged.owner_count; ++i) {
    staged.owners[i].derivative = (step.transition * staged.owners[i].derivative).eval();
    staged.owners[i].end_support = false;
  }
  for (int record = 0; record < 2; ++record) {
    staged.owners[current[record]].derivative += step.noise[record];
    staged.owners[current[record]].end_support = w1(record) > 0.;
  }

  for (unsigned i = 0; i < staged.owner_count;) {
    const auto &owner = staged.owners[i];
    if (owner.start_support || owner.end_support) {
      ++i;
      continue;
    }
    // The support chain and positive-duration substep prove that this original
    // left record has reached its successor knot. Contract its accumulated
    // derivative once; never treat repeated substeps as independent readings.
    if (staged.eliminated_records == std::numeric_limits<uint64_t>::max()) return false;
    staged.offset.noalias() -= owner.derivative * owner.record.noise_linearization;
    staged.conditional_covariance.noalias() +=
        owner.derivative * owner.record.prior.selfadjointView<Eigen::Upper>() * owner.derivative.transpose();
    ++staged.eliminated_records;
    for (unsigned j = i + 1; j < staged.owner_count; ++j) staged.owners[j - 1] = staged.owners[j];
    staged.owners[--staged.owner_count] = Owner{};
  }
  staged.conditional_covariance =
      (0.5 * staged.conditional_covariance + 0.5 * staged.conditional_covariance.transpose()).eval();
  if (!finite(staged.transition) || !finite(staged.offset) || !covariance_valid(staged.conditional_covariance)) return false;
  for (unsigned i = 0; i < staged.owner_count; ++i)
    if (!finite(staged.owners[i].derivative)) return false;
  staged.time1 = step.time1;
  ++staged.steps;
  statistics_ = staged;
  left_sequence_ = step.records[0].sequence;
  right_sequence_ = step.records[1].sequence;
  return true;
}

bool SampledCpiStatistics::export_statistics(Output &output) const {
  if (!statistics_.steps) return false;
  output = statistics_;
  return true;
}

} // namespace ov_init
