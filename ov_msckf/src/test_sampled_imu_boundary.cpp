/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/print.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

// Host-only allocation interception includes allocations inside the linked
// library. No replacement estimator arithmetic is supplied by this fixture.
#ifdef __GLIBC__
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

using namespace ov_msckf;
using namespace ov_type;
namespace {
using Mat = Eigen::MatrixXd;
using VecX = Eigen::VectorXd;
using V6 = Eigen::Matrix<double, 6, 1>;
using M6 = Eigen::Matrix<double, 6, 6>;
int checks = 0, failures = 0;
double max_covariance_error = 0., max_mean_error = 0.;
void check(bool ok, const char *what) { ++checks; if (!ok) { ++failures; std::printf("FAIL: %s\n", what); } }
template <class A, class B> bool same(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b) {
  const Mat aa = a, bb = b;
  return aa.rows() == bb.rows() && aa.cols() == bb.cols() && std::memcmp(aa.data(), bb.data(), aa.size() * sizeof(double)) == 0;
}
State::SampledImuRecord record(uint64_t sequence, double time) {
  State::SampledImuRecord r; r.stream_episode = 73; r.sequence = sequence; r.timestamp = time;
  M6 L = M6::Zero();
  for (int i = 0; i < 6; ++i) {
    r.measured(i) = .17 * i + .01 * sequence;
    for (int j = 0; j <= i; ++j) L(i, j) = i == j ? .2 + .03 * i : .01 * (i + j + 1);
  }
  r.prior = L * L.transpose(); return r;
}
std::shared_ptr<State> make_state(bool clocks = true) {
  StateOptions o; o.num_cameras = 2; o.physical_camera_clones = true; o.do_fej = false;
  o.do_calib_camera_timeoffset = clocks; o.max_clone_size = 4;
  check(o.configure_clone_policy(false, false), "fixture configures bounded physical owners");
  return std::make_shared<State>(o);
}
std::vector<std::shared_ptr<Type>> variables(const std::shared_ptr<State> &s) {
  std::vector<std::shared_ptr<Type>> out{s->_imu};
  for (int i = 0; i < s->_options.num_cameras; ++i)
    if (s->cam_imu_dt_var(i)->id() >= 0) out.push_back(s->cam_imu_dt_var(i));
  for (const auto &slot : s->sampled_imu_slots()) if (slot.noise) out.push_back(slot.noise);
  for (const auto &pose : s->_exposure_poses) out.push_back(pose.pose);
  std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a->id() < b->id(); });
  return out;
}
VecX mean(const std::shared_ptr<State> &s) {
  VecX m = VecX::Zero(s->max_covariance_size());
  // This oracle exercises Euclidean means; attitude remains uncorrelated with
  // the measured coordinates. It does not test the nonlinear JPL retraction.
  m.segment<3>(3) = s->_imu->pos(); m.segment<3>(6) = s->_imu->vel();
  m.segment<3>(9) = s->_imu->bias_g(); m.segment<3>(12) = s->_imu->bias_a();
  for (int i = 0; i < s->_options.num_cameras; ++i)
    if (s->cam_imu_dt_var(i)->id() >= 0) m(s->cam_imu_dt_var(i)->id()) = s->cam_imu_dt_var(i)->value()(0);
  for (const auto &slot : s->sampled_imu_slots()) if (slot.noise) m.segment<6>(slot.noise->id()) = slot.noise->value();
  for (const auto &view : s->_exposure_poses) m.segment<3>(view.pose->id() + 3) = view.pose->pos();
  return m;
}
bool same_registry(const std::shared_ptr<State> &a, const std::shared_ptr<State> &b, bool separate) {
  for (size_t i = 0; i < 2; ++i) {
    const auto &x = a->sampled_imu_slots()[i], &y = b->sampled_imu_slots()[i];
    if (x.active != y.active || x.record.sequence != y.record.sequence || x.record.stream_episode != y.record.stream_episode ||
        ov_core::initializer_time_bits(x.record.timestamp) != ov_core::initializer_time_bits(y.record.timestamp) ||
        !same(x.record.measured, y.record.measured) || !same(x.record.prior, y.record.prior) ||
        bool(x.noise) != bool(y.noise)) return false;
    if (x.noise && ((separate && x.noise == y.noise) || x.noise->id() != y.noise->id() ||
                    !same(x.noise->value(), y.noise->value()) || !same(x.noise->fej(), y.noise->fej()))) return false;
  }
  return true;
}

struct BatchOracle {
  Mat root_cov, C, observations, R;
  VecX root_mean, z;
  int initial_dimension;
  BatchOracle(const Mat &P, const VecX &m) : initial_dimension(P.rows()) {
    const int n = initial_dimension + 18;
    root_cov = Mat::Zero(n, n); root_cov.topLeftCorner(P.rows(), P.cols()) = P;
    root_mean = VecX::Zero(n); root_mean.head(m.size()) = m;
    for (int i = 0; i < 3; ++i) root_cov.block<6, 6>(initial_dimension + 6 * i, initial_dimension + 6 * i) = record(i + 1, 0.).prior;
    C = Mat::Zero(initial_dimension + 12, n); C.topLeftCorner(initial_dimension, initial_dimension).setIdentity();
    observations.resize(0, n); R.resize(0, 0); z.resize(0);
  }
  void admit(int id, int sequence) { C.middleRows(id, 6).setZero(); C.block<6, 6>(id, initial_dimension + 6 * (sequence - 1)).setIdentity(); }
  void retire(int id) { C.middleRows(id, 6).setZero(); }
  void transform(const Mat &T) { C = (T * C).eval(); }
  void observe(const Mat &H, const VecX &value, const Mat &noise) {
    int old = z.size(), n = value.size();
    observations.conservativeResize(old + n, Eigen::NoChange); observations.bottomRows(n) = H * C;
    z.conservativeResize(old + n); z.tail(n) = value;
    Mat next = Mat::Zero(old + n, old + n); next.topLeftCorner(old, old) = R; next.bottomRightCorner(n, n) = noise; R.swap(next);
  }
  std::pair<VecX, Mat> posterior() const {
    if (z.size() == 0) return {C * root_mean, C * root_cov * C.transpose()};
    const Mat S = observations * root_cov * observations.transpose() + R;
    const Mat cross = C * root_cov * observations.transpose();
    // Dense all-measurement conditioning, independent of the production
    // triangular block update and all previous recursive posteriors.
    const auto solver = S.ldlt();
    return {C * root_mean + cross * solver.solve(z - observations * root_mean),
            C * root_cov * C.transpose() - cross * solver.solve(cross.transpose())};
  }
};

void compare(const std::shared_ptr<State> &s, const BatchOracle &oracle, const char *what) {
  const auto expected = oracle.posterior();
  const double pe = (StateHelper::get_full_covariance(s) - expected.second).cwiseAbs().maxCoeff();
  const double me = (mean(s) - expected.first).cwiseAbs().maxCoeff();
  max_covariance_error = std::max(max_covariance_error, pe); max_mean_error = std::max(max_mean_error, me);
  check(pe < 2e-11 && me < 2e-11, what);
}

void propagate(const std::shared_ptr<State> &s, BatchOracle &oracle, const std::shared_ptr<Vec> &left,
               const std::shared_ptr<Vec> &right, double dt, double wl, double wr) {
  Mat F = Mat::Identity(15, 15); F.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 3, 6> M;
  M << -.2,.04,.01,-1.,.08,.03, .03,-.18,.02,.02,-.9,.05, -.01,.06,-.16,-.03,.04,-1.1;
  Mat G = Mat::Zero(15, 6); G.block<3, 6>(3, 0) = .5 * dt * dt * M; G.block<3, 6>(6, 0) = dt * M;
  Mat phi = Mat::Zero(15, 27); phi.leftCols(15) = F; phi.middleCols(15, 6) = wl * G; phi.rightCols(6) = wr * G;
  Mat T = Mat::Identity(s->max_covariance_size(), s->max_covariance_size());
  T.topRows(15).setZero(); T.topLeftCorner(15, 15) = F;
  T.block(0, left->id(), 15, 6) = wl * G; T.block(0, right->id(), 15, 6) = wr * G;
  const VecX predicted = T * mean(s);
  auto x = s->_imu->value(); x.block<3, 1>(4, 0) = predicted.segment<3>(3); x.block<3, 1>(7, 0) = predicted.segment<3>(6);
  s->_imu->set_value(x); s->_imu->set_fej(x);
  StateHelper::EKFPropagation(s, {s->_imu}, {s->_imu, left, right}, phi, Mat::Zero(15, 15));
  oracle.transform(T); compare(s, oracle, "actual augmented propagation retains full sample/state mean and covariance");
}

void observe(const std::shared_ptr<State> &s, BatchOracle &oracle, int step, const std::shared_ptr<Vec> &direct = nullptr) {
  std::vector<std::shared_ptr<Type>> order{s->_exposure_poses.front().pose, s->_exposure_poses.back().pose, s->_imu->v()};
  if (direct) order.push_back(direct);
  const int cols = direct ? 21 : 15;
  Mat H = Mat::Zero(3, cols);
  H.block<3, 3>(0, 3) = (.8 + .03 * step) * Eigen::Matrix3d::Identity();
  H.block<3, 3>(0, 9) = -.27 * Eigen::Matrix3d::Identity();
  H.block<3, 3>(0, 12) = .4 * Eigen::Matrix3d::Identity();
  if (direct) { H.block<3, 3>(0, 15) = .3 * Eigen::Matrix3d::Identity(); H.block<3, 3>(0, 18) = -.2 * Eigen::Matrix3d::Identity(); }
  Mat full = Mat::Zero(3, s->max_covariance_size()); int col = 0;
  for (const auto &v : order) { full.middleCols(v->id(), v->size()) += H.middleCols(col, v->size()); col += v->size(); }
  VecX z(3); z << .18 * std::sin(.3 + step), -.13 * std::cos(.4 + step), .09 + .015 * step;
  const Mat R = .018 * Mat::Identity(3, 3);
  const VecX residual = z - full * mean(s);
  StateHelper::EKFUpdate(s, order, H, residual, R);
  oracle.observe(full, z, R); compare(s, oracle, "actual repeated EKF update matches all-measurement dense conditioning");
}

void linked_gaussian() {
  auto s = make_state(); const int initial = s->max_covariance_size();
  Mat P = Mat::Zero(initial, initial); P.topLeftCorner<3, 3>() = .02 * Eigen::Matrix3d::Identity();
  Mat L = Mat::Zero(initial - 3, initial - 3);
  for (int i = 0; i < L.rows(); ++i) for (int j = 0; j <= i; ++j) L(i, j) = i == j ? .21 + .01 * i : .018 * std::sin(i + .3 * j);
  P.bottomRightCorner(initial - 3, initial - 3) = L * L.transpose();
  StateHelper::set_initial_covariance(s, P, variables(s));
  auto x = s->_imu->value(); x.block<3, 1>(7, 0) << .3, -.1, .05; s->_imu->set_value(x); s->_imu->set_fej(x);
  BatchOracle oracle(P, mean(s));
  check(!s->has_sampled_imu_boundary(), "ordinary state does not enable sampled ownership");
  check(StateHelper::prepare_sampled_imu_boundary(s, 73), "explicit foundation allocates exactly two slots");
  auto first = StateHelper::admit_sampled_imu_noise(s, record(1, 0.));
  auto second = StateHelper::admit_sampled_imu_noise(s, record(2, .5));
  check(first && second && first != second, "two distinct raw-axis owners are admitted");
  oracle.admit(first->id(), 1); oracle.admit(second->id(), 2);
  compare(s, oracle, "first-use prior augmentation is full and independent");
  propagate(s, oracle, first, second, .2, .8, .2);

  for (int camera = 0; camera < 2; ++camera) {
    const int old = s->max_covariance_size();
    Mat J = Mat::Zero(6, old); J.leftCols<6>().setIdentity();
    J.block<3, 1>(3, s->cam_imu_dt_var(camera)->id()) = s->_imu->vel();
    Mat T = Mat::Zero(old + 6, old); T.topRows(old).setIdentity(); T.bottomRows(6) = J;
    State::ExposurePose view; view.camera_id = camera; view.raw_time = .2; view.imu_time = .2;
    view.pose = StateHelper::augment_pose_view(s, camera, Eigen::Vector3d::Zero());
    s->_exposure_poses.push_back(view); oracle.transform(T);
  }
  check(s->_exposure_poses[0].pose != s->_exposure_poses[1].pose, "same-time camera owners use separate pose identities");
  compare(s, oracle, "same-time owned pose augmentation copies every raw-noise cross row");
  observe(s, oracle, 0); observe(s, oracle, 1);
  check(first->value().norm() > 1e-5 && second->value().norm() > 1e-5, "ordinary pose observations infer nonzero raw-noise posterior means");
  const Mat before_repeat = StateHelper::get_full_covariance(s); const VecX before_mean = mean(s);
  check(StateHelper::admit_sampled_imu_noise(s, record(1, 0.)) == first && same(before_repeat, StateHelper::get_full_covariance(s)) &&
        same(before_mean, mean(s)), "identical active record reuse preserves posterior mean and covariance");
  propagate(s, oracle, first, second, .3, .3, .7);
  s->_imu_endpoint_valid = true; s->_imu_endpoint = std::nextafter(.5, 0.);
  check(!StateHelper::retire_sampled_imu_noise_at_knot(s, 1), "retirement refuses an interior endpoint one ULP before the raw knot");
  compare(s, oracle, "rejected early retirement leaves the full posterior unchanged");
  s->_imu_endpoint = .5;
  check(!StateHelper::retire_sampled_imu_noise_at_knot(s, 2), "retirement refuses the newer owner");
  const int first_id = first->id();
  check(StateHelper::retire_sampled_imu_noise_at_knot(s, 1), "older raw owner retires at exact accepted successor knot");
  oracle.retire(first_id); compare(s, oracle, "retirement marginalizes without conditioning the retained visual posterior");
  observe(s, oracle, 2);
  check(!StateHelper::admit_sampled_imu_noise(s, record(1, 0.)), "retired raw information cannot be reintroduced as independent");
  auto third = StateHelper::admit_sampled_imu_noise(s, record(3, 1.1));
  check(third == first && third->id() == first_id, "successor admission reuses the same Vec and covariance coordinates");
  oracle.admit(third->id(), 3);
  propagate(s, oracle, second, third, .22, .8, .2);
  observe(s, oracle, 3); observe(s, oracle, 4, third);

  auto snapshot = StateHelper::clone_state(s);
  check(same_registry(s, snapshot, true) && same(StateHelper::get_full_covariance(s), StateHelper::get_full_covariance(snapshot)),
        "snapshot rewires registry Types and preserves exact posterior, FEJ and metadata");
  const Mat saved = StateHelper::get_full_covariance(snapshot); const VecX saved_mean = mean(snapshot);
  observe(s, oracle, 5);
  check(same(saved, StateHelper::get_full_covariance(snapshot)) && same(saved_mean, mean(snapshot)), "snapshot is independent of later source updates");
  check(!StateHelper::admit_sampled_imu_noise(snapshot, record(1, 0.)), "snapshot preserves retired-identity watermark");
  s = StateHelper::clone_state(snapshot);
  check(same_registry(s, snapshot, true) && same(saved, StateHelper::get_full_covariance(s)) && same(saved_mean, mean(s)),
        "restore-style clone preserves nonzero sample means and every covariance cross block");
  bool refused = false;
  try { StateHelper::marginalize(s, s->sampled_imu_slots()[0].noise); } catch (const std::invalid_argument &) { refused = true; }
  check(refused && same(saved, StateHelper::get_full_covariance(s)), "generic marginalization cannot invalidate permanent slot ownership");
}

void invalid_and_singular() {
  auto s = make_state(false);
  check(!StateHelper::prepare_sampled_imu_boundary(s, 0), "zero stream identity is rejected");
  check(StateHelper::prepare_sampled_imu_boundary(s, 73), "singular fixture prepares its permanent slots");
  const Mat baseline = StateHelper::get_full_covariance(s);
  for (int variant = 0; variant < 8; ++variant) {
    auto r = record(1, 1.);
    if (variant == 0) r.timestamp = std::numeric_limits<double>::infinity();
    if (variant == 1) r.measured(2) = std::numeric_limits<double>::quiet_NaN();
    if (variant == 2) r.prior(1, 2) += .1;
    if (variant == 3) r.prior(0, 0) = -.01;
    if (variant == 4) { r.prior.setIdentity(); r.prior(0, 1) = r.prior(1, 0) = 2.; }
    if (variant == 5) r.prior(4, 4) = std::numeric_limits<double>::infinity();
    if (variant == 6) r.sequence = 0;
    if (variant == 7) r.stream_episode = 99;
    check(!StateHelper::admit_sampled_imu_noise(s, r) && same(baseline, StateHelper::get_full_covariance(s)),
          "invalid sample/prior is rejected atomically under production arithmetic flags");
  }
  auto r = record(1, 1.); r.prior.setZero(); r.prior.topLeftCorner<3, 3>().setConstant(.04);
  auto a = StateHelper::admit_sampled_imu_noise(s, r);
  check(bool(a) && same(StateHelper::get_full_covariance(s).block<6, 6>(a->id(), a->id()), r.prior),
        "rank-one PSD prior with zero-variance axes is accepted without jitter");
  const Mat singular = StateHelper::get_full_covariance(s);
  r.measured(0) = std::nextafter(r.measured(0), 1.);
  check(!StateHelper::admit_sampled_imu_noise(s, r) && same(singular, StateHelper::get_full_covariance(s)),
        "changed immutable raw value is rejected for an existing sample identity");
  r = record(2, 1.);
  check(!StateHelper::admit_sampled_imu_noise(s, r), "new sequence cannot reuse an existing raw timestamp");
  r.timestamp = 2.; r.prior.setZero();
  check(bool(StateHelper::admit_sampled_imu_noise(s, r)), "deterministic zero prior is a valid singular sample");
  const Mat full = StateHelper::get_full_covariance(s);
  check(!StateHelper::admit_sampled_imu_noise(s, record(3, 3.)) && same(full, StateHelper::get_full_covariance(s)), "third live raw sample is rejected without growing state");
  check(StateHelper::prepare_sampled_imu_boundary(s, 73) && !StateHelper::prepare_sampled_imu_boundary(s, 74) &&
        same(full, StateHelper::get_full_covariance(s)), "repeat preparation cannot reset posteriors or change stream identity");
}

void reindex_reset_and_inactive() {
  auto s = make_state(false);
  State::ExposurePose owner; owner.camera_id = 0; owner.raw_time = 1.; owner.imu_time = 1.;
  owner.pose = StateHelper::augment_pose_view(s, 0, Eigen::Vector3d::Zero());
  s->_exposure_poses.push_back(owner);
  const int removed_id = owner.pose->id();
  check(StateHelper::prepare_sampled_imu_boundary(s, 73), "slots can follow existing stochastic pose owners");
  const auto a = s->sampled_imu_slots()[0].noise, b = s->sampled_imu_slots()[1].noise;
  const int aid = a->id(), bid = b->id();
  StateHelper::EKFPropagation(s, {s->_imu->bg(), s->_imu->ba()}, {s->_imu->bg(), s->_imu->ba()},
                              M6::Identity(), .001 * M6::Identity());
  Mat P = StateHelper::get_full_covariance(s);
  check(P.middleRows(aid, 6).isZero(0.) && P.middleRows(bid, 6).isZero(0.) &&
        a->value().isZero(0.) && b->value().isZero(0.) && a->fej().isZero(0.) && b->fej().isZero(0.),
        "ordinary bias process noise leaves both inactive slots structurally zero");
  check(StateHelper::valid_initial_covariance(P, true), "complete covariance with inactive zero rows remains valid PSD");
  Mat expected(P.rows() - 6, P.cols() - 6);
  for (int r = 0; r < expected.rows(); ++r) for (int c = 0; c < expected.cols(); ++c)
    expected(r, c) = P(r < removed_id ? r : r + 6, c < removed_id ? c : c + 6);
  StateHelper::marginalize(s, owner.pose); s->_exposure_poses.clear();
  check(a->id() == aid - 6 && b->id() == bid - 6 && same(expected, StateHelper::get_full_covariance(s)),
        "ordinary earlier pose removal reindexes reserved Types and keeps registry covariance coherent");
  auto first = StateHelper::admit_sampled_imu_noise(s, record(1, 1.));
  auto second = StateHelper::admit_sampled_imu_noise(s, record(2, 2.));
  s->_imu_endpoint_valid = true; s->_imu_endpoint = 2.;
  check(first && second && StateHelper::retire_sampled_imu_noise_at_knot(s, 1), "admission and retirement work after ordinary state reindexing");
  const auto snapshot = StateHelper::clone_state(s);
  check(same_registry(s, snapshot, true) && same(StateHelper::get_full_covariance(s), StateHelper::get_full_covariance(snapshot)),
        "snapshot rewires a mixture of active and inactive reserved slots");
  auto fresh = make_state(false);
  check(!fresh->has_sampled_imu_boundary() && !fresh->sampled_imu_slots()[0].noise && !fresh->sampled_imu_slots()[1].noise &&
        fresh->max_covariance_size() == 15, "fresh reset State rebuilds covariance and registry together without old owners");
}

void resource_cycles() {
  auto s = make_state(false); const int initial = s->max_covariance_size();
  StateHelper::prepare_sampled_imu_boundary(s, 73);
  auto a = StateHelper::admit_sampled_imu_noise(s, record(1, 1.));
  auto b = StateHelper::admit_sampled_imu_noise(s, record(2, 2.));
  const int aid = a->id(), bid = b->id();
  auto next = record(3, 3.); s->_imu_endpoint_valid = true;
  bool ok = true;
#ifdef __GLIBC__
  allocations = 0; count_allocations = true;
  const Mat allocation_control = StateHelper::get_full_covariance(s);
  count_allocations = false;
  std::printf("SAMPLED_BOUNDARY_RESOURCE control_allocations=%zu\n", allocations);
  check(allocations > 0 && allocation_control.rows() == initial + 12,
        "allocation interception observes an intentional linked-library covariance copy");
#endif
  for (int count : {200, 20000}) {
#ifdef __GLIBC__
    allocations = 0; count_allocations = true;
#endif
    for (int i = 0; i < count; ++i) {
      s->_imu_endpoint = next.timestamp - 1.;
      ok = StateHelper::retire_sampled_imu_noise_at_knot(s, next.sequence - 2) && ok;
      ok = bool(StateHelper::admit_sampled_imu_noise(s, next)) && ok;
      ++next.sequence; next.timestamp += 1.;
    }
#ifdef __GLIBC__
    count_allocations = false;
    std::printf("SAMPLED_BOUNDARY_RESOURCE swaps=%d allocations=%zu\n", count, allocations);
    check(allocations == 0, "linked admission and raw-knot retirement allocate no heap memory after preparation");
#endif
    check(ok && s->max_covariance_size() == initial + 12 && a->id() == aid && b->id() == bid,
          "repeated swaps preserve the fixed twelve-coordinate budget and Type identities");
  }
}
} // namespace

int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  linked_gaussian(); invalid_and_singular(); reindex_reset_and_inactive(); resource_cycles();
  std::printf("SAMPLED_BOUNDARY %s checks=%d failures=%d covariance_error=%.12g mean_error=%.12g\n",
              failures ? "FAIL" : "PASS", checks, failures, max_covariance_error, max_mean_error);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
