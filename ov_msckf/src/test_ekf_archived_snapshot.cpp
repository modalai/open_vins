/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "cam/CamRadtan.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/print.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>

// Test-only glibc probe. Returned allocations and the target allocator are
// unchanged. A fixed table records only allocations born during the update;
// setup, oracle work and existing State storage are outside this measurement.
#ifdef __GLIBC__
namespace allocation_probe {
struct Entry {
  void *address;
  size_t bytes;
};
static thread_local Entry entries[8192]{};
static thread_local bool enabled = false;
static thread_local size_t calls = 0, total = 0, live = 0, peak = 0,
                           overflows = 0;
static size_t bucket(void *p) {
  return (reinterpret_cast<uintptr_t>(p) >> 4) % 8192;
}
static void admitted(void *p, size_t n) {
  if (!enabled || !p)
    return;
  ++calls;
  total += n;
  live += n;
  peak = std::max(peak, live);
  for (size_t i = bucket(p), j = 0; j < 8192; ++j, i = (i + 1) % 8192)
    if (entries[i].address == nullptr ||
        entries[i].address == reinterpret_cast<void *>(1)) {
      entries[i] = {p, n};
      return;
    }
  ++overflows;
}
static void retired(void *p) {
  if (!enabled || !p)
    return;
  for (size_t i = bucket(p), j = 0; j < 8192; ++j, i = (i + 1) % 8192) {
    if (entries[i].address == nullptr)
      return;
    if (entries[i].address == p) {
      live -= entries[i].bytes;
      entries[i] = {reinterpret_cast<void *>(1), 0};
      return;
    }
  }
}
static void begin() {
  std::memset(entries, 0, sizeof entries);
  calls = total = live = peak = overflows = 0;
  enabled = true;
}
} // namespace allocation_probe
extern "C" void *__libc_malloc(size_t);
extern "C" void *__libc_calloc(size_t, size_t);
extern "C" void *__libc_realloc(void *, size_t);
extern "C" void *__libc_memalign(size_t, size_t);
extern "C" void __libc_free(void *);
extern "C" void *malloc(size_t n) noexcept {
  void *p = __libc_malloc(n);
  allocation_probe::admitted(p, n);
  return p;
}
extern "C" void *calloc(size_t n, size_t s) noexcept {
  void *p = __libc_calloc(n, s);
  allocation_probe::admitted(p, n * s);
  return p;
}
extern "C" void free(void *p) noexcept {
  allocation_probe::retired(p);
  __libc_free(p);
}
extern "C" void *realloc(void *p, size_t n) noexcept {
  // Record successful requests. Allocator-internal realloc workspace and
  // alignment overhead are outside these requested-byte measurements.
  void *q = __libc_realloc(p, n);
  if (q || n == 0) {
    allocation_probe::retired(p);
    allocation_probe::admitted(q, n);
  }
  return q;
}
extern "C" void *aligned_alloc(size_t a, size_t n) noexcept {
  void *p = __libc_memalign(a, n);
  allocation_probe::admitted(p, n);
  return p;
}
extern "C" int posix_memalign(void **p, size_t a, size_t n) noexcept {
  if (a < sizeof(void *) || (a & (a - 1)))
    return EINVAL;
  void *q = __libc_memalign(a, n);
  if (!q)
    return ENOMEM;
  allocation_probe::admitted(q, n);
  *p = q;
  return 0;
}
#endif

using namespace ov_msckf;
using namespace ov_type;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Read-only ownership inventory and covariance replacement are confined to
// restoring this immutable test corpus; all updates call the real StateHelper.
template <class Tag, typename Tag::type Member> struct Access {
  friend typename Tag::type access(Tag) { return Member; }
};
struct CovTag {
  using type = Matrix State::*;
  friend type access(CovTag);
};
struct VariablesTag {
  using type = std::vector<std::shared_ptr<Type>> State::*;
  friend type access(VariablesTag);
};
template struct Access<CovTag, &State::_Cov>;
template struct Access<VariablesTag, &State::_variables>;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char *why) {
  ++checks;
  if (!ok) {
    ++failures;
    std::printf("FAIL: %s\n", why);
  }
}
bool same(const Matrix &a, const Matrix &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         std::memcmp(a.data(), b.data(), 8 * a.size()) == 0;
}
template <class T> T scalar(std::ifstream &in) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof value);
  if (!in)
    std::abort();
  return value;
}
Matrix matrix(std::ifstream &in, int rows, int cols) {
  Matrix a(rows, cols);
  in.read(reinterpret_cast<char *>(a.data()), 8 * a.size());
  if (!in)
    std::abort();
  return a;
}
struct Variable {
  int id, size;
  Vector value, fej;
};
struct Corpus {
  Matrix P, H, R, expected_P;
  Vector residual, expected_dx;
  std::vector<Variable> variables;
  std::vector<int> order;
};
Corpus load(const char *path) {
  std::ifstream in(path, std::ios::binary);
  char magic[16];
  in.read(magic, 16);
  if (!in || std::memcmp(magic, "OV_EKF_SNAPSHOT1", 16) != 0)
    std::abort();
  const auto n = scalar<uint64_t>(in), m = scalar<uint64_t>(in),
             hc = scalar<uint64_t>(in), nv = scalar<uint64_t>(in),
             no = scalar<uint64_t>(in);
  if (n != 183 || m != 86 || hc != 86 || nv != 30 || no != 14)
    std::abort();
  Corpus c;
  c.P = matrix(in, n, n);
  c.H = matrix(in, m, hc);
  c.residual = matrix(in, m, 1);
  c.R = matrix(in, m, m);
  for (size_t i = 0; i < nv; ++i) {
    Variable v;
    v.id = scalar<int32_t>(in);
    v.size = scalar<int32_t>(in);
    const int rows = scalar<int32_t>(in);
    if (rows <= 0 || rows > 16)
      std::abort();
    v.value = matrix(in, rows, 1);
    v.fej = matrix(in, rows, 1);
    c.variables.push_back(v);
  }
  for (size_t i = 0; i < no; ++i)
    c.order.push_back(scalar<int32_t>(in));
  c.expected_P = matrix(in, n, n);
  c.expected_dx = matrix(in, n, 1);
  return c;
}
std::shared_ptr<State> restore(const Corpus &c) {
  StateOptions options;
  options.num_cameras = 2;
  options.do_calib_camera_timeoffset = true;
  options.do_calib_camera_pose = true;
  options.do_calib_camera_intrinsics = true;
  options.dt_calib_gate = false;
  auto state = std::make_shared<State>(options);
  auto &vars = state.get()->*access(VariablesTag{});
  for (size_t i = 0; i < c.variables.size(); ++i) {
    const auto &saved = c.variables[i];
    if (i >= vars.size()) {
      if (saved.size != 6 || saved.value.rows() != 7)
        std::abort();
      auto pose = std::make_shared<PoseJPL>();
      pose->set_local_id(saved.id);
      vars.push_back(pose);
      state->_clones_IMU.emplace(double(i), pose);
    }
    auto &v = vars[i];
    if (v->id() != saved.id || v->size() != saved.size ||
        v->value().rows() != saved.value.rows())
      std::abort();
    v->set_value(saved.value);
    v->set_fej(saved.fej);
  }
  state.get()->*access(CovTag{}) = c.P;
  for (size_t cam = 0; cam < 2; ++cam) {
    auto model = std::make_shared<ov_core::CamRadtan>(640, 480);
    model->set_value(state->_cam_intrinsics.at(cam)->value());
    state->_cam_intrinsics_cameras[cam] = model;
  }
  return state;
}
Vector retract(const Variable &v, const Vector &dx) {
  if (v.value.rows() == dx.rows())
    return v.value + dx;
  Eigen::Vector4d dq;
  dq << .5 * dx.head<3>(), 1.;
  dq.normalize();
  const Eigen::Vector3d a = dq.head<3>(), b = v.value.head<3>();
  const double sa = dq(3), sb = v.value(3);
  Eigen::Vector4d q;
  q.head<3>() = sa * b + sb * a - a.cross(b);
  q(3) = sa * sb - a.dot(b);
  if (q(3) < 0.)
    q = -q;
  q.normalize();
  Vector out = v.value;
  out.head<4>() = q;
  if (dx.rows() > 3)
    out.tail(dx.rows() - 3) += dx.tail(dx.rows() - 3);
  return out;
}
void run(const Corpus &c, const std::array<std::ptrdiff_t, 3> &cache) {
  Eigen::setCpuCacheSizes(cache[0], cache[1], cache[2]);
  auto state = restore(c);
  const auto &vars = state.get()->*access(VariablesTag{});
  std::vector<std::shared_ptr<Type>> order;
  for (int id : c.order) {
    auto it = std::find_if(vars.begin(), vars.end(),
                           [id](const auto &v) { return v->id() == id; });
    if (it == vars.end())
      std::abort();
    order.push_back(*it);
  }
  std::vector<const Type *> identities;
  for (const auto &v : vars)
    identities.push_back(v.get());
#ifdef __GLIBC__
  allocation_probe::begin();
#endif
  const bool accepted =
      StateHelper::EKFUpdate(state, order, c.H, c.residual, c.R);
#ifdef __GLIBC__
  allocation_probe::enabled = false;
  check(allocation_probe::overflows == 0,
        "allocation probe retained every measured allocation");
  std::printf("RESOURCE cache_l1=%td calls=%zu requested_bytes=%zu "
              "peak_extra_live_bytes=%zu remaining_extra_live_bytes=%zu\n",
              cache[0], allocation_probe::calls, allocation_probe::total,
              allocation_probe::peak, allocation_probe::live);
#endif
  check(accepted, "actual StateHelper accepts archived visual update");
  const double covariance_error =
      (StateHelper::get_full_covariance(state) - c.expected_P)
          .cwiseAbs()
          .maxCoeff();
  check(
      covariance_error < 5e-13,
      "full covariance agrees with independent 70-digit dense joint reference");
  double mean_error = 0.;
  for (size_t i = 0; i < vars.size(); ++i) {
    const auto &v = vars[i];
    const auto &old = c.variables[i];
    mean_error = std::max(
        mean_error,
        (v->value() - retract(old, c.expected_dx.segment(old.id, old.size)))
            .cwiseAbs()
            .maxCoeff());
    check(v.get() == identities[i] && v->id() == old.id,
          "update preserves variable identities and covariance ownership");
    check(same(v->fej(), old.fej),
          "update leaves complete FEJ values unchanged");
  }
  check(mean_error < 5e-13, "all IMU, calibration and clone means agree with "
                            "independent retractions");
  for (size_t cam = 0; cam < 2; ++cam)
    check(same(state->_cam_intrinsics.at(cam)->value(),
               state->_cam_intrinsics_cameras.at(cam)->get_value()),
          "actual camera model receives committed intrinsic mean");
  std::printf("NUMERICAL cache_l1=%td covariance_max_error=%.17g "
              "mean_max_error=%.17g\n",
              cache[0], covariance_error, mean_error);
}
} // namespace
int main(int argc, char **argv) {
  ov_core::Printer::setPrintLevel("ERROR");
#ifdef __GLIBC__
  allocation_probe::begin();
  void *(*volatile allocate)(size_t) = malloc;
  void *p = allocate(12345);
  free(p);
  allocation_probe::enabled = false;
  check(allocation_probe::calls == 1 && allocation_probe::total == 12345 &&
            allocation_probe::peak == 12345 && allocation_probe::live == 0,
        "allocation probe self-control");
#endif
  const auto c = load(argc > 1 ? argv[1] : TEST_EKF_SNAPSHOT_PATH);
  run(c, {49152, 2621440, 12582912});
  run(c, {32768, 4194304, 2097152});
  std::printf("EKF_ARCHIVED_SNAPSHOT %s checks=%d failures=%d\n",
              failures ? "FAIL" : "PASS", checks, failures);
  return failures ? 1 : 0;
}
