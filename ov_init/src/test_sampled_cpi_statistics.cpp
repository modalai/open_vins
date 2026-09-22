/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Dense all-record oracle; no nonlinear initializer/runtime claim.
 */
#include "dynamic/SampledCpiStatistics.h"
#include "cpi/CpiV1.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <vector>

#if defined(__GLIBC__) && !defined(OV_TEST_NO_MALLOC_INTERPOSE)
static thread_local bool count_allocations = false;
static thread_local size_t allocations = 0;
extern "C" void *__libc_malloc(size_t);
extern "C" void *__libc_calloc(size_t, size_t);
extern "C" void *__libc_realloc(void *, size_t);
extern "C" void *__libc_memalign(size_t, size_t);
extern "C" void *malloc(size_t n) noexcept { if (count_allocations) ++allocations; return __libc_malloc(n); }
extern "C" void *calloc(size_t n, size_t s) noexcept { if (count_allocations) ++allocations; return __libc_calloc(n, s); }
extern "C" void *realloc(void *p, size_t n) noexcept { if (count_allocations) ++allocations; return __libc_realloc(p, n); }
extern "C" void *aligned_alloc(size_t a, size_t n) noexcept { if (count_allocations) ++allocations; return __libc_memalign(a, n); }
extern "C" int posix_memalign(void **p, size_t a, size_t n) noexcept {
  if (a < sizeof(void *) || (a & (a - 1))) return EINVAL;
  if (count_allocations) ++allocations;
  void *value = __libc_memalign(a, n);
  if (!value) return ENOMEM;
  *p = value; return 0;
}
#endif

namespace {
using Stats = ov_init::SampledCpiStatistics;
using Record = Stats::Record;
using Step = Stats::Step;
using Output = Stats::Output;
using M15 = Stats::Matrix15;
using M6 = Stats::Matrix6;
using J = Stats::Jacobian;
using V15 = Stats::Vector15;
using V6 = Stats::Vector6;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
int checks = 0, failures = 0;
double maximum_covariance_error = 0., maximum_mean_error = 0.;
void check(bool condition, const char *message) {
  ++checks;
  if (!condition) { ++failures; std::printf("FAIL: %s\n", message); }
}
uint64_t bits(double value) { uint64_t b; std::memcpy(&b, &value, sizeof(b)); return b; }
template<class A, class B> bool exact(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
  for (int column = 0; column < a.cols(); ++column)
    for (int row = 0; row < a.rows(); ++row)
      if (bits(a(row, column)) != bits(b(row, column))) return false;
  return true;
}
bool exact(const Record &a, const Record &b) {
  return a.stream_episode == b.stream_episode && a.sequence == b.sequence && bits(a.timestamp) == bits(b.timestamp) &&
         exact(a.measured, b.measured) && exact(a.prior, b.prior) && exact(a.noise_linearization, b.noise_linearization);
}
bool exact(const Output &a, const Output &b) {
  if (bits(a.time0) != bits(b.time0) || bits(a.time1) != bits(b.time1) || a.steps != b.steps ||
      a.eliminated_records != b.eliminated_records || a.owner_count != b.owner_count ||
      !exact(a.transition, b.transition) || !exact(a.offset, b.offset) || !exact(a.conditional_covariance, b.conditional_covariance)) return false;
  for (unsigned i = 0; i < a.owners.size(); ++i)
    if (!exact(a.owners[i].record, b.owners[i].record) || !exact(a.owners[i].derivative, b.owners[i].derivative) ||
        a.owners[i].start_support != b.owners[i].start_support || a.owners[i].end_support != b.owners[i].end_support) return false;
  return true;
}
Mat pattern(int rows, int columns, double phase = 0.) {
  Mat out(rows, columns);
  for (int c = 0; c < columns; ++c)
    for (int r = 0; r < rows; ++r) out(r, c) = std::sin(.43 * (r + 1) + .71 * (c + 1) + phase) + .3 * std::cos(.29 * (r + 1) * (c + 1));
  return out;
}
Record record(uint64_t sequence, double time, bool singular = false) {
  Record r;
  r.stream_episode = 37; r.sequence = sequence; r.timestamp = time;
  M6 L = .025 * pattern(6, 6, .13 * sequence);
  L.diagonal().array() += .2;
  if (singular) L.rightCols<3>().setZero();
  r.prior = L * L.transpose();
  r.measured = pattern(6, 1, .07 * sequence);
  r.noise_linearization = .03 * pattern(6, 1, .17 * sequence);
  return r;
}
Eigen::Vector2d weights(const Record &left, const Record &right, double time) {
  if (time == left.timestamp) return Eigen::Vector2d(1., 0.);
  if (time == right.timestamp) return Eigen::Vector2d(0., 1.);
  const double fraction = (time - left.timestamp) / (right.timestamp - left.timestamp);
  return Eigen::Vector2d(1. - fraction, fraction);
}
Step step(const Record &left, const Record &right, double begin, double end, int phase = 0) {
  Step s;
  s.records = {left, right}; s.time0 = begin; s.time1 = end;
  s.weights0 = weights(left, right, begin); s.weights1 = weights(left, right, end);
  const double dt = end - begin;
  M15 A = .012 * pattern(15, 15, .3 * phase);
  A.block<3, 3>(0, 3) -= Eigen::Matrix3d::Identity();
  A.block<3, 3>(6, 9) -= Eigen::Matrix3d::Identity();
  A.block<3, 3>(12, 6) += Eigen::Matrix3d::Identity();
  s.transition = M15::Identity() + dt * A + .5 * dt * dt * A * A;
  J G0 = .06 * pattern(15, 6, .14 * phase), G1 = .06 * pattern(15, 6, .14 * phase + .1);
  G0.topLeftCorner<3, 3>() -= Eigen::Matrix3d::Identity();
  G0.block<3, 3>(6, 3) -= Eigen::Matrix3d::Identity();
  G1.topLeftCorner<3, 3>() -= Eigen::Matrix3d::Identity();
  G1.block<3, 3>(6, 3) -= Eigen::Matrix3d::Identity();
  for (int i = 0; i < 2; ++i) s.noise[i] = .5 * dt * (s.weights0(i) * G0 + s.weights1(i) * G1);
  J L = .003 * dt * pattern(15, 6, .2 * phase);
  L.block<3, 3>(3, 0) += .02 * Eigen::Matrix3d::Identity();
  L.block<3, 3>(9, 3) += .04 * Eigen::Matrix3d::Identity();
  s.independent_covariance = dt * L * L.transpose();
  return s;
}

struct Model {
  Mat prior;
  Vec mean;
  std::vector<Mat> cut;
  std::vector<Vec> offset;
  std::vector<int> raw_ids; // original zero-based index at each six-column block
  int raw_column(int id) const {
    auto position = std::find(raw_ids.begin(), raw_ids.end(), id);
    return position == raw_ids.end() ? -1 : 30 + 6 * static_cast<int>(position - raw_ids.begin());
  }
};

struct Scene {
  std::vector<Record> records;
  std::vector<Step> steps;
  std::vector<int> cuts;
  std::vector<Output> intervals;
  bool begins_at_knot;
  Scene(bool singular, bool first_knot, bool last_knot) : begins_at_knot(first_knot) {
    for (int i = 0; i < 13; ++i) records.push_back(record(i + 1, 10. + i / 16., singular));
    const std::vector<double> camera{first_knot ? 10. : 10.015625, 10.046875, 10.0625, 10.296875,
                                     10.609375, last_knot ? 10.75 : 10.703125};
    for (size_t c = 0; c + 1 < camera.size(); ++c) {
      Stats stats;
      for (size_t i = 0; i + 1 < records.size(); ++i) {
        const double begin = std::max(camera[c], records[i].timestamp);
        const double end = std::min(camera[c + 1], records[i + 1].timestamp);
        if (!(end > begin)) continue;
        const double mid = .5 * begin + .5 * end;
        for (const auto &range : {std::make_pair(begin, mid), std::make_pair(mid, end)}) {
          steps.push_back(step(records[i], records[i + 1], range.first, range.second, static_cast<int>(steps.size())));
          check(stats.append(steps.back()), "chronological substep accepted");
        }
      }
      Output output;
      check(stats.export_statistics(output), "nonempty interval exports conditional statistics");
      check(output.owner_count <= 4, "at most four original cut owners");
      check(output.time0 == camera[c] && output.time1 == camera[c + 1], "exact physical interval endpoints retained");
      unsigned end_support = 0;
      for (unsigned i = 0; i < output.owner_count; ++i) end_support += output.owners[i].end_support;
      check(end_support == (camera[c + 1] == records[1].timestamp || camera[c + 1] == records.back().timestamp ? 1u : 2u),
            "exact knot retains one end owner; interior cut retains two");
      cuts.push_back(static_cast<int>(steps.size())); intervals.push_back(output);
    }
  }
};

void make_prior(const Scene &scene, Model &model, int dimension) {
  model.prior = Mat::Zero(dimension, dimension); model.mean = Vec::Zero(dimension);
  const Mat L = .02 * pattern(30, 30) + .15 * Mat::Identity(30, 30);
  model.prior.topLeftCorner(30, 30) = L * L.transpose();
  model.mean.head(30) = .04 * pattern(30, 1);
  for (int id : model.raw_ids) {
    const int column = model.raw_column(id);
    model.prior.block<6, 6>(column, column) = scene.records[id].prior;
    if (id <= (scene.begins_at_knot ? 0 : 1)) {
      // A valid incoming joint prior with nonzero navigation/camera/calibration
      // and boundary-noise crosses. This must survive interval compression.
      const Mat J0 = .22 * pattern(30, 6, .6 * id);
      const Mat cross = J0 * scene.records[id].prior;
      model.prior.block(0, column, 30, 6) = cross;
      model.prior.block(column, 0, 6, 30) = cross.transpose();
      model.prior.topLeftCorner(30, 30) += cross * J0.transpose();
      model.mean.segment<6>(column) = scene.records[id].prior * pattern(6, 1, .3 * id);
    }
  }
  // A previous likelihood conditions the incoming navigation and its original
  // support jointly. In particular two starting raw records become correlated;
  // their current covariance differs from the immutable ORIGINAL prior stored
  // in Record. The interval must neither marginalize these pinned coordinates
  // independently nor replace their posterior with those original priors.
  Mat H = Mat::Zero(4, dimension);
  H.leftCols(30) = .4 * pattern(4, 30, .6);
  for (int id = 0; id <= (scene.begins_at_knot ? 0 : 1); ++id)
    H.block<4, 6>(0, model.raw_column(id)) = .5 * pattern(4, 6, .4 * id);
  const Mat cross = model.prior * H.transpose();
  const Eigen::Matrix4d innovation = H * cross + .007 * Eigen::Matrix4d::Identity();
  model.mean += cross * innovation.ldlt().solve(.1 * pattern(4, 1) - H * model.mean);
  model.prior -= cross * innovation.ldlt().solve(cross.transpose());
}

Model dense_model(const Scene &scene) {
  Model m;
  for (int i = 0; i < static_cast<int>(scene.records.size()); ++i) m.raw_ids.push_back(i);
  const int process0 = 30 + 6 * static_cast<int>(m.raw_ids.size());
  const int dimension = process0 + 15 * static_cast<int>(scene.steps.size());
  make_prior(scene, m, dimension);
  for (size_t i = 0; i < scene.steps.size(); ++i)
    m.prior.block<15, 15>(process0 + 15 * i, process0 + 15 * i) = scene.steps[i].independent_covariance;
  for (int count : scene.cuts) {
    // Independent batch construction: explicit products of all later F's for
    // each original record and each process increment; no streaming contraction.
    Mat C = Mat::Zero(15, dimension); V15 offset = V15::Zero();
    M15 initial = M15::Identity();
    for (int i = 0; i < count; ++i) initial = (scene.steps[i].transition * initial).eval();
    C.leftCols<15>() = initial;
    for (int i = 0; i < count; ++i) {
      M15 suffix = M15::Identity();
      for (int j = i + 1; j < count; ++j) suffix = (scene.steps[j].transition * suffix).eval();
      for (int raw = 0; raw < 2; ++raw) {
        const int id = static_cast<int>(scene.steps[i].records[raw].sequence - 1);
        const J T = suffix * scene.steps[i].noise[raw];
        C.block<15, 6>(0, m.raw_column(id)) += T;
        offset -= T * scene.steps[i].records[raw].noise_linearization;
      }
      C.block<15, 15>(0, process0 + 15 * i) = suffix;
    }
    m.cut.push_back(C); m.offset.push_back(offset);
  }
  return m;
}

Model compressed_model(const Scene &scene, int wrong_model = 0) {
  Model m;
  std::set<int> retained;
  for (const auto &interval : scene.intervals)
    for (unsigned i = 0; i < interval.owner_count; ++i) retained.insert(static_cast<int>(interval.owners[i].record.sequence - 1));
  m.raw_ids.assign(retained.begin(), retained.end());
  const int process0 = 30 + 6 * static_cast<int>(m.raw_ids.size());
  const int dimension = process0 + 15 * static_cast<int>(scene.intervals.size());
  make_prior(scene, m, dimension);
  Mat C = Mat::Zero(15, dimension); C.leftCols<15>().setIdentity();
  V15 offset = V15::Zero();
  for (size_t c = 0; c < scene.intervals.size(); ++c) {
    const auto &interval = scene.intervals[c];
    C = (interval.transition * C).eval();
    offset = (interval.transition * offset + interval.offset).eval();
    M15 Q = interval.conditional_covariance;
    for (unsigned i = 0; i < interval.owner_count; ++i) {
      const auto &owner = interval.owners[i];
      if (wrong_model != 2) C.block<15, 6>(0, m.raw_column(static_cast<int>(owner.record.sequence - 1))) += owner.derivative;
      offset -= owner.derivative * owner.record.noise_linearization;
      if (wrong_model) Q += owner.derivative * owner.record.prior * owner.derivative.transpose();
    }
    C.block<15, 15>(0, process0 + 15 * c).setIdentity();
    m.prior.block<15, 15>(process0 + 15 * c, process0 + 15 * c) = Q;
    m.cut.push_back(C); m.offset.push_back(offset);
  }
  return m;
}

struct Marginal { Vec mean; Mat covariance; };
struct View { Mat map; Vec offset; };
View export_view(const Scene &scene, const Model &model) {
  const auto &last = scene.intervals.back();
  int boundary = 0;
  for (unsigned i = 0; i < last.owner_count; ++i) boundary += last.owners[i].end_support;
  View out{Mat::Zero(30 + 6 * boundary, model.prior.rows()), Vec::Zero(30 + 6 * boundary)};
  out.map.topRows(15) = model.cut.back(); out.offset.head(15) = model.offset.back();
  // Two retained image owners are functions of two earlier navigation poses,
  // mixed with the incoming camera extrinsic uncertainty. The three remaining
  // coordinates stand for a camera-consider block in the linear oracle.
  const int cuts[2] = {0, 3};
  for (int i = 0; i < 2; ++i) {
    out.map.middleRows(15 + 6 * i, 3) = model.cut[cuts[i]].topRows(3);
    out.map.middleRows(18 + 6 * i, 3) = model.cut[cuts[i]].bottomRows(3);
    out.offset.segment<3>(15 + 6 * i) = model.offset[cuts[i]].head(3);
    out.offset.segment<3>(18 + 6 * i) = model.offset[cuts[i]].tail(3);
    out.map.block<6, 6>(15 + 6 * i, 15 + 6 * i) += Eigen::Matrix<double, 6, 6>::Identity();
  }
  out.map.block<3, 3>(27, 27).setIdentity();
  int row = 30;
  for (unsigned i = 0; i < last.owner_count; ++i) {
    if (!last.owners[i].end_support) continue;
    out.map.block<6, 6>(row, model.raw_column(static_cast<int>(last.owners[i].record.sequence - 1))).setIdentity(); row += 6;
  }
  return out;
}

Marginal posterior(const Scene &scene, const Model &model, bool duplicate_prior = false) {
  const View out = export_view(scene, model);
  const int visual_rows = 6 * static_cast<int>(model.cut.size());
  const int extra = duplicate_prior ? out.map.rows() - 30 : 0;
  Mat H = Mat::Zero(visual_rows + extra, model.prior.rows());
  Mat R = Mat::Identity(visual_rows + extra, visual_rows + extra) * .0025;
  Vec z = Vec::Zero(visual_rows + extra), constant = Vec::Zero(visual_rows + extra);
  for (size_t i = 0; i < model.cut.size(); ++i) {
    const Mat A = .5 * pattern(6, 15, .2 * i);
    H.middleRows(6 * i, 6) = A * model.cut[i];
    H.block<6, 12>(6 * i, 15) += .12 * pattern(6, 12, .3 * i);
    H.block<6, 3>(6 * i, 27) += .21 * pattern(6, 3, .4 * i);
    constant.segment<6>(6 * i) = A * model.offset[i];
    z.segment<6>(6 * i) = .2 * pattern(6, 1, .5 * i);
  }
  if (extra) {
    H.bottomRows(extra) = out.map.bottomRows(extra);
    const auto &interval = scene.intervals.back();
    int row = visual_rows;
    for (unsigned i = 0; i < interval.owner_count; ++i)
      if (interval.owners[i].end_support) { R.block<6, 6>(row, row) = interval.owners[i].record.prior; row += 6; }
  }
  const Mat innovation = H * model.prior * H.transpose() + R;
  const Mat cross = out.map * model.prior * H.transpose();
  const auto solve = innovation.ldlt();
  return {out.map * model.mean + out.offset + cross * solve.solve(z - constant - H * model.mean),
          out.map * model.prior * out.map.transpose() - cross * solve.solve(cross.transpose())};
}

void compare(const Marginal &a, const Marginal &b, const char *message) {
  const double covariance_error = (a.covariance - b.covariance).norm() / std::max(1., a.covariance.norm());
  const double mean_error = (a.mean - b.mean).norm() / std::max(1., a.mean.norm());
  maximum_covariance_error = std::max(maximum_covariance_error, covariance_error);
  maximum_mean_error = std::max(maximum_mean_error, mean_error);
  check(covariance_error < 2e-12 && mean_error < 2e-12, message);
}

void gaussian_oracle(bool singular, bool start_knot, bool end_knot) {
  const Scene scene(singular, start_knot, end_knot);
  const Model dense = dense_model(scene), compressed = compressed_model(scene);
  check(compressed.raw_ids.size() < dense.raw_ids.size(), "full Gaussian fixture actually eliminates interior original records");
  if (!start_knot)
    check(dense.prior.block<6, 6>(dense.raw_column(0), dense.raw_column(1)).norm() > 1e-4,
          "incoming posterior contains nonzero cross covariance between the two original boundary records");
  const View d = export_view(scene, dense), c = export_view(scene, compressed);
  compare({d.map * dense.mean + d.offset, d.map * dense.prior * d.map.transpose()},
          {c.map * compressed.mean + c.offset, c.map * compressed.prior * c.map.transpose()},
          "compressed full prior equals dense all-record model with incoming navigation/camera/raw crosses");
  const Marginal expected = posterior(scene, dense), actual = posterior(scene, compressed);
  compare(expected, actual, "image-conditioned posterior means and all navigation/owner/calibration/raw cross blocks match dense oracle");
  const int boundary = actual.mean.size() - 30;
  check(actual.mean.tail(boundary).norm() > 1e-3, "visual likelihood changes retained raw-noise posterior means");
  check(actual.covariance.topRightCorner(30, boundary).norm() > 1e-4,
        "retained raw posterior correlates with navigation, image owners and camera block");
  check(actual.covariance.block(27, 30, 3, boundary).norm() > 1e-6, "camera/raw cross covariance is nonzero");
  const Eigen::SelfAdjointEigenSolver<Mat> spectrum(actual.covariance);
  check(spectrum.info() == Eigen::Success && spectrum.eigenvalues().minCoeff() > -2e-12,
        "complete exported joint covariance is PSD including singular raw priors");
  // These deliberately wrong models are controls, not alternate accepted paths.
  const Marginal doubled = posterior(scene, compressed_model(scene, 1));
  check((doubled.covariance - actual.covariance).norm() > 1e-5,
        "adding retained original priors to conditional CPI Q twice is detected");
  const Marginal independent = posterior(scene, compressed_model(scene, 2));
  check((independent.covariance - actual.covariance).norm() > 1e-3,
        "marginalizing shared raw records independently inside each CPI factor is detected");
  if (!singular) {
    const Marginal duplicate = posterior(scene, compressed, true);
    check((duplicate.covariance - actual.covariance).norm() > 1e-3,
          "a second prior likelihood for the same original boundary record is detected");
  }
  Marginal fresh = actual;
  fresh.mean.tail(boundary).setZero();
  fresh.covariance.rightCols(boundary).setZero(); fresh.covariance.bottomRows(boundary).setZero();
  int raw_row = 30;
  for (unsigned i = 0; i < scene.intervals.back().owner_count; ++i) {
    const auto &owner = scene.intervals.back().owners[i];
    if (!owner.end_support) continue;
    fresh.covariance.block<6, 6>(raw_row, raw_row) = owner.record.prior; raw_row += 6;
  }
  const Mat continuation = Mat::Identity(actual.mean.size(), actual.mean.size()) + .08 * pattern(actual.mean.size(), actual.mean.size());
  check((continuation * (actual.mean - fresh.mean)).norm() > 1e-3 &&
        (continuation * (actual.covariance - fresh.covariance) * continuation.transpose()).norm() > 1e-3,
        "fresh independent admission at handoff corrupts continuation mean and covariance");
  Model no_incoming_cross = compressed;
  for (int id : compressed.raw_ids) {
    no_incoming_cross.prior.block(0, compressed.raw_column(id), 30, 6).setZero();
    no_incoming_cross.prior.block(compressed.raw_column(id), 0, 6, 30).setZero();
  }
  check((posterior(scene, no_incoming_cross).covariance - actual.covariance).norm() > 1e-3,
        "discarding incoming navigation/boundary cross covariance is detected");
}

void repeated_pair_and_units() {
  const Record a = record(1, 0.), b = record(2, 1.), c = record(3, 2.), d = record(4, 3.);
  Stats stats;
  Step s = step(a, b, 0., .25);
  s.transition.setIdentity(); s.independent_covariance.setZero();
  s.noise[0].setZero(); s.noise[1].setZero(); s.noise[1].topRows<6>().setIdentity();
  check(stats.append(s), "first substep uses a single original right record");
  s.time0 = .25; s.time1 = .75; s.weights0 = weights(a, b, s.time0); s.weights1 = weights(a, b, s.time1);
  check(stats.append(s), "repeated raw pair accumulates same record derivative");
  s.time0 = .75; s.time1 = 1.; s.weights0 = weights(a, b, s.time0); s.weights1 = weights(a, b, s.time1);
  check(stats.append(s), "third substep reaches exact raw knot");
  Output out; stats.export_statistics(out);
  check(out.conditional_covariance.norm() == 0. && out.eliminated_records == 0,
        "a retained raw owner is not independently contracted on any substep");
  Step next = step(b, c, 1., 2.); next.transition.setIdentity(); next.independent_covariance.setZero();
  next.noise[0].setZero(); next.noise[1].setZero();
  check(stats.append(next), "next pair retires the previous unpinned original");
  stats.export_statistics(out);
  check((out.conditional_covariance.topLeftCorner<6, 6>() - 9. * b.prior).norm() < 1e-13,
        "repeated derivative contracts as (B+B+B) R (B+B+B)^T, not three independent BRB^T");
  check((out.offset.head<6>() + 3. * b.noise_linearization).norm() < 1e-14,
        "elimination preserves the original zero-mean prior at a nonzero linearization point");
  check(out.eliminated_records == 1 && out.owner_count == 2 && out.owners[0].start_support,
        "start-cut record remains pinned after losing direct interpolation support");
  next = step(c, d, 2., 3.); next.transition.setIdentity(); next.independent_covariance.setZero();
  next.noise[0].setZero(); next.noise[1].setZero();
  check(stats.append(next), "interior owner can be eliminated after its successor knot");
  stats.export_statistics(out);
  check(out.eliminated_records == 2 && out.owner_count == 2, "longer interval keeps only pinned start and exact final knot");

  const Scene scene(true, false, false);
  Stats base, scaled;
  const V6 scale = (V6() << 1e-9, 1e6, .01, 1e9, 1e-5, 100.).finished();
  for (int i = scene.cuts[2]; i < scene.cuts[3]; ++i) {
    const Step &original = scene.steps[i]; Step changed = original;
    for (int raw = 0; raw < 2; ++raw) {
      changed.records[raw].measured = scale.asDiagonal() * original.records[raw].measured;
      changed.records[raw].noise_linearization = scale.asDiagonal() * original.records[raw].noise_linearization;
      changed.records[raw].prior = scale.asDiagonal() * original.records[raw].prior * scale.asDiagonal();
      changed.noise[raw] = original.noise[raw] * scale.cwiseInverse().asDiagonal();
    }
    check(base.append(original) && scaled.append(changed), "mixed-unit singular PSD raw prior remains admissible");
  }
  Output original, changed; base.export_statistics(original); scaled.export_statistics(changed);
  check((original.conditional_covariance - changed.conditional_covariance).norm() < 1e-14 &&
        (original.offset - changed.offset).norm() < 1e-14, "unit rescaling preserves conditional covariance and mean offset");

  Stats deterministic;
  Step zero = step(a, b, 0., 1.);
  zero.records[0].prior.setZero(); zero.records[1].prior.setZero(); zero.independent_covariance.setZero();
  check(deterministic.append(zero), "exactly zero original and process covariance is valid PSD");
  deterministic.export_statistics(changed);
  check(changed.conditional_covariance.norm() == 0., "deterministic directions receive no numerical jitter");
}

void rejection_atomicity() {
  const Record a = record(1, 0.), b = record(2, 1.), c = record(3, 2.);
  Stats stats; Output before, after;
  check(!stats.export_statistics(before), "empty accumulator declines export");
  Step first = step(a, b, 0., .5), valid = step(a, b, .5, 1.);
  check(stats.append(first), "atomicity fixture starts"); stats.export_statistics(before);
  auto reject = [&](const Step &bad, const char *message) {
    check(!stats.append(bad), message); check(stats.export_statistics(after) && exact(before, after), "rejection leaves every exported byte unchanged");
  };
  const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
  Step bad = valid; bad.records[0].stream_episode = 0; reject(bad, "zero stream episode rejected");
  bad = valid; bad.records[1].stream_episode += 1; reject(bad, "mixed stream episode rejected");
  bad = valid; bad.records[0].sequence = bad.records[1].sequence; reject(bad, "duplicate original sequence rejected");
  bad = valid; bad.records[0].timestamp = -0.; reject(bad, "same numeric timestamp with changed exact bits rejected");
  bad = valid; bad.records[1].timestamp = std::nextafter(1., 2.); bad.weights0 = weights(bad.records[0], bad.records[1], bad.time0);
  bad.weights1 = weights(bad.records[0], bad.records[1], bad.time1); reject(bad, "changed original raw timestamp rejected");
  bad = valid; bad.records[0].measured(2) += .1; reject(bad, "changed original measurement rejected");
  bad = valid; bad.records[1].noise_linearization(1) += .1; reject(bad, "changed record linearization within interval rejected");
  bad = valid; bad.records[1].prior *= 2.; reject(bad, "changed original covariance rejected");
  bad = valid; bad.time0 = std::nextafter(.5, 1.); bad.weights0 = weights(a, b, bad.time0); reject(bad, "gap between substeps rejected");
  bad = valid; bad.time0 = .4; bad.weights0 = weights(a, b, bad.time0); reject(bad, "overlap between substeps rejected");
  bad = valid; bad.time1 = .5; bad.weights1 = weights(a, b, bad.time1); reject(bad, "zero-duration substep rejected");
  bad = valid; bad.time1 = 1.1; reject(bad, "extrapolated substep rejected");
  bad = valid; bad.weights0(0) += .01; reject(bad, "invented fragment weights rejected");
  for (int kind = 0; kind < 10; ++kind) {
    bad = valid;
    switch (kind) {
      case 0: bad.time0 = nan; break;
      case 1: bad.records[0].timestamp = inf; break;
      case 2: bad.records[0].measured(0) = nan; break;
      case 3: bad.records[0].noise_linearization(0) = inf; break;
      case 4: bad.records[0].prior(0, 0) = nan; break;
      case 5: bad.transition(0, 0) = inf; break;
      case 6: bad.noise[0](1, 0) = nan; break;
      case 7: bad.independent_covariance(1, 1) = inf; break;
      case 8: bad.weights1(0) = nan; break;
      case 9: bad.noise[1](2, 3) = -inf; break;
    }
    reject(bad, "nonfinite input rejected under production fast math");
  }
  bad = valid; bad.independent_covariance.setZero(); bad.independent_covariance(0, 1) = 1e-20;
  reject(bad, "zero-variance row with nonzero covariance rejected");
  bad = valid; bad.independent_covariance.setIdentity(); bad.independent_covariance(0, 1) = .1;
  reject(bad, "asymmetric covariance rejected");
  bad = valid; bad.independent_covariance.setIdentity(); bad.independent_covariance(0, 1) = 1.2; bad.independent_covariance(1, 0) = 1.2;
  reject(bad, "indefinite covariance rejected");
  bad = valid; bad.independent_covariance.setIdentity(); bad.independent_covariance(0, 0) = -1e-30;
  reject(bad, "negative diagonal rejected despite other larger units");
  bad = valid; bad.independent_covariance.setZero(); bad.independent_covariance(0, 0) = 1e-18;
  bad.independent_covariance(1, 1) = 1e18; bad.independent_covariance(0, 1) = bad.independent_covariance(1, 0) = 1.1;
  reject(bad, "mixed-unit indefinite small block cannot hide behind a global PSD tolerance");
  bad = valid; bad.transition.setConstant(std::numeric_limits<double>::max());
  reject(bad, "finite inputs causing overflow reject before committing statistics");
  check(stats.append(valid), "valid retry after all rejected substeps succeeds"); stats.export_statistics(before);
  bad = step(b, c, 1., 1.5); bad.records[0] = a;
  reject(bad, "advancing support must reuse exact previous right original record");
  bad = step(b, c, 1., 1.5); bad.records[1].sequence = 1;
  reject(bad, "retired or older sequence cannot be re-admitted");
  check(stats.append(step(b, c, 1., 1.5)), "valid adjacent support still succeeds after invalid pair attempts");

  for (int kind = 0; kind < 4; ++kind) {
    Stats empty; bad = first;
    bad.records[0].prior.setIdentity();
    if (kind == 0) bad.records[0].prior(0, 0) = -1e-30;
    if (kind == 1) bad.records[0].prior(0, 1) = .1;
    if (kind == 2) bad.records[0].prior(0, 1) = bad.records[0].prior(1, 0) = 1.1;
    if (kind == 3) {
      bad.records[0].prior(0, 0) = 1e-18; bad.records[0].prior(1, 1) = 1e18;
      bad.records[0].prior(0, 1) = bad.records[0].prior(1, 0) = 1.1;
    }
    check(!empty.append(bad) && !empty.export_statistics(after), "invalid original covariance rejected before first admission");
  }
}

void cpi_missing_information() {
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const double dt = .25, eps = 1e-6;
  auto integrate = [&](double first, double second) {
    ov_core::CpiV1 cpi(.2, .03, .4, .05, true);
    cpi.setLinearizationPoints(zero, zero);
    cpi.feed_IMU(0., dt, Eigen::Vector3d(first, 0., 0.), zero, Eigen::Vector3d(second, 0., 0.), zero);
    return cpi;
  };
  const auto nominal = integrate(0., 0.);
  const auto plus0 = integrate(eps, 0.), minus0 = integrate(-eps, 0.);
  const auto plus1 = integrate(0., eps), minus1 = integrate(0., -eps);
  const double B0 = ov_core::log_so3(plus0.R_k2tau * minus0.R_k2tau.transpose())(0) / (2. * eps);
  const double B1 = ov_core::log_so3(plus1.R_k2tau * minus1.R_k2tau.transpose())(0) / (2. * eps);
  check(std::abs(B0 + .5 * dt) < 1e-10 && std::abs(B1 + .5 * dt) < 1e-10,
        "actual CPI mean depends on each original endpoint reading");
  const double Ra0 = .1, Ra1 = .3, Rb0 = .25, Rb1 = .15;
  const double Pa = B0 * B0 * Ra0 + B1 * B1 * Ra1, Pb = B0 * B0 * Rb0 + B1 * B1 * Rb1;
  check(std::abs(Pa - Pb) < 1e-14 && std::abs(B1 * Ra1 - B1 * Rb1) > .01,
        "equal CPI marginal admits different original boundary cross covariance");
  const double neighboring_cross = B1 * .2 * B0;
  check(neighboring_cross > .003 && nominal.P_meas(0, 0) > 0.,
        "neighboring actual CPI means share a raw record although independent CPI factors omit its cross covariance");
  // Do not turn the existing continuous CPI into a sampled factor by simply
  // removing white density. At zero motion, position variance from accel bias
  // RW starts at dt^5, beyond this one-step covariance RK4 polynomial order.
  ov_core::CpiV1 bias_only(0., .2, 0., .3, true);
  bias_only.setLinearizationPoints(zero, zero); bias_only.feed_IMU(0., dt, zero, zero, zero, zero);
  const double determinant = bias_only.P_meas(12, 12) * bias_only.P_meas(9, 9) - std::pow(bias_only.P_meas(12, 9), 2);
  check(determinant < -1e-10 && bias_only.P_meas(12, 12) == 0.,
        "actual CPI zero-white covariance is not a sound PSD bias-only replacement");
  std::printf("CPI_COUNTEREXAMPLE marginal=%.9g boundary_cross_A=%.9g boundary_cross_B=%.9g adjacent_cross=%.9g bias_minor=%.9g\n",
              Pa, B1 * Ra1, B1 * Rb1, neighboring_cross, determinant);
}

void allocation_bound() {
  Stats stats;
  Record left = record(1, 0.), right = record(2, 1.);
  Step s = step(left, right, .25, .5);
  s.transition.setIdentity(); s.independent_covariance.setZero();
  s.noise[0].setZero(); s.noise[1].setZero();
  s.noise[0].topRows<6>() = .001 * M6::Identity(); s.noise[1].topRows<6>() = .002 * M6::Identity();
  Output out;
  bool okay = true; unsigned max_owners = 0;
  const int pairs = 10000;
#if defined(__GLIBC__) && !defined(OV_TEST_NO_MALLOC_INTERPOSE)
  allocations = 0; count_allocations = true;
#endif
  Eigen::internal::set_is_malloc_allowed(false);
  for (int i = 0; i < pairs; ++i) {
    const double first = i == 0 ? .25 : double(i);
    s.records = {left, right}; s.time0 = first; s.time1 = i + .5;
    s.weights0 = weights(left, right, s.time0); s.weights1 = weights(left, right, s.time1);
    okay = stats.append(s) && okay;
    okay = stats.export_statistics(out) && okay; max_owners = std::max(max_owners, out.owner_count);
    s.time0 = i + .5; s.time1 = i + 1.; s.weights0 = weights(left, right, s.time0); s.weights1 = weights(left, right, s.time1);
    okay = stats.append(s) && okay;
    okay = stats.export_statistics(out) && okay; max_owners = std::max(max_owners, out.owner_count);
    left = right; right.sequence += 1; right.timestamp += 1.; right.measured.array() += .001;
  }
  Eigen::internal::set_is_malloc_allowed(true);
#if defined(__GLIBC__) && !defined(OV_TEST_NO_MALLOC_INTERPOSE)
  count_allocations = false;
  check(allocations == 0, "20,000 accepted substeps and exports perform zero intercepted heap allocations");
  std::printf("ALLOCATION_BOUND substeps=%d allocations=%zu sizeof_accumulator=%zu sizeof_step=%zu sizeof_output=%zu max_owners=%u\n",
              2 * pairs, allocations, sizeof(Stats), sizeof(Step), sizeof(Output), max_owners);
#else
  std::printf("ALLOCATION_BOUND libc interception unavailable; Eigen no-allocation assertion active\n");
#endif
  check(okay && out.steps == 2u * pairs && out.eliminated_records == pairs - 2u,
        "long original-record stream preserves counts without historical storage");
  check(max_owners == 4 && out.owner_count == 3, "storage stays at four owners and final exact knot keeps one end owner");
}
} // namespace

int main() {
  for (bool singular : {false, true})
    for (bool start_knot : {false, true})
      for (bool end_knot : {false, true}) gaussian_oracle(singular, start_knot, end_knot);
  repeated_pair_and_units(); rejection_atomicity(); cpi_missing_information(); allocation_bound();
  std::printf("SAMPLED_CPI_STATISTICS checks=%d failures=%d covariance_error=%.3e mean_error=%.3e\n",
              checks, failures, maximum_covariance_error, maximum_mean_error);
  return failures ? 1 : 0;
}
