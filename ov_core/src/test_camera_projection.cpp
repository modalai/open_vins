/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

// Count allocations in both the caller and linked implementation. This probe
// is glibc-specific; the numerical oracle below is portable.
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
  *p = value;
  return 0;
}
#endif

namespace {
using MP = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<80>>;
using Input = std::array<MP, 10>; // normalized x,y then fx,fy,cx,cy,k1..k4
using Output = std::array<MP, 2>;
int checks = 0, failures = 0;
double max_forward_error = 0., max_point_error = 0., max_intrinsic_error = 0.;
double max_forward_relative = 0., max_derivative_relative = 0., old_forward_error = 0.;

void check(bool ok, const char *message) {
  ++checks;
  if (!ok) { ++failures; std::printf("FAIL: %s\n", message); }
}

// Independent high-precision defining equations. No production helper,
// derivative, small-radius approximation, or Jacobian is used here.
Output reference(bool equi, const Input &a) {
  const MP x = a[0], y = a[1], s = x * x + y * y;
  MP xd, yd;
  if (equi) {
    MP scale = 1;
    if (s != 0) {
      const MP radius = sqrt(s), theta = atan(radius);
      MP distorted = theta;
      for (int j = 0; j < 4; ++j) distorted += a[6 + j] * pow(theta, 3 + 2 * j);
      scale = distorted / radius;
    }
    xd = x * scale;
    yd = y * scale;
  } else {
    const MP radial = 1 + a[6] * s + a[7] * s * s;
    xd = x * radial + 2 * a[8] * x * y + a[9] * (s + 2 * x * x);
    yd = y * radial + a[8] * (s + 2 * y * y) + 2 * a[9] * x * y;
  }
  return {a[2] * xd + a[4], a[3] * yd + a[5]};
}

void verify_point(bool equi, ov_core::CamBase &camera, const Eigen::Matrix<double, 8, 1> &calib,
                  const Eigen::Vector2d &point) {
  Input in;
  for (int i = 0; i < 2; ++i) in[i] = MP(point(i));
  for (int i = 0; i < 8; ++i) in[2 + i] = MP(calib(i));
  const auto exact = reference(equi, in);
  const auto actual = camera.distort_d(point);
  const Eigen::Vector2d legacy = camera.distort_f(point.cast<float>()).cast<double>();
  Eigen::MatrixXd Hpoint, Hcalib;
  camera.compute_distort_jacobian(point, Hpoint, Hcalib);
  check(Hpoint.rows() == 2 && Hpoint.cols() == 2 && Hcalib.rows() == 2 && Hcalib.cols() == 8,
        "base-pointer Jacobian dimensions");
  for (int row = 0; row < 2; ++row) {
    const double expected = static_cast<double>(exact[row]);
    const double error = std::abs(actual(row) - expected);
    const double relative = error / (1.0 + std::abs(expected));
    max_forward_error = std::max(max_forward_error, error);
    max_forward_relative = std::max(max_forward_relative, relative);
    old_forward_error = std::max(old_forward_error, std::abs(legacy(row) - expected));
    check(relative < 5e-14, "true-double forward projection agrees with 80-digit defining equation");
  }
  // Symmetric differences at 80 decimal digits leave ample separation between
  // truncation (O(1e-50)) and cancellation (about 1e-55) for these finite inputs.
  const MP base_step("1e-25");
  for (int column = 0; column < 10; ++column) {
    const MP step = base_step * (1 + abs(in[column]));
    auto plus = in, minus = in;
    plus[column] += step;
    minus[column] -= step;
    const auto zp = reference(equi, plus), zm = reference(equi, minus);
    for (int row = 0; row < 2; ++row) {
      const double expected = static_cast<double>((zp[row] - zm[row]) / (2 * step));
      const double actual_derivative = column < 2 ? Hpoint(row, column) : Hcalib(row, column - 2);
      const double error = std::abs(actual_derivative - expected);
      const double relative = error / (1.0 + std::abs(expected));
      (column < 2 ? max_point_error : max_intrinsic_error) =
          std::max(column < 2 ? max_point_error : max_intrinsic_error, error);
      max_derivative_relative = std::max(max_derivative_relative, relative);
      if (relative >= 5e-13)
        std::printf("  model=%s xy=(%.17g,%.17g) derivative(%d,%d) got=%.17g expected=%.17g\n",
                    equi ? "equi" : "radtan", point(0), point(1), row, column, actual_derivative, expected);
      check(relative < 5e-13, "point and intrinsic Jacobians agree with independent high-precision finite differences");
    }
  }
}

void verify_actual_mean_derivatives(ov_core::CamBase &camera, const Eigen::Matrix<double, 8, 1> &calib) {
  // This uses the actual production forward method through CamBase, so a
  // dispatch back to the old float wrapper cannot pass by testing formulas alone.
  const Eigen::Vector2d p(.25, -.375);
  Eigen::MatrixXd Hpoint, Hcalib;
  camera.compute_distort_jacobian(p, Hpoint, Hcalib);
  for (int column = 0; column < 10; ++column) {
    const double h = column < 2 ? 1e-4 : 1e-4 * std::max(1.0, std::abs(calib(column - 2)));
    std::array<Eigen::Vector2d, 4> samples;
    const int shifts[4] = {-2, -1, 1, 2};
    for (int k = 0; k < 4; ++k) {
      auto point = p;
      auto intrinsics = calib;
      if (column < 2) point(column) += shifts[k] * h;
      else intrinsics(column - 2) += shifts[k] * h;
      camera.set_value(intrinsics);
      samples[k] = camera.distort_d(point);
    }
    camera.set_value(calib);
    const Eigen::Vector2d fd = (samples[0] - 8.0 * samples[1] + 8.0 * samples[2] - samples[3]) / (12.0 * h);
    const Eigen::Vector2d analytic = column < 2 ? Eigen::Vector2d(Hpoint.col(column)) : Eigen::Vector2d(Hcalib.col(column - 2));
    check((fd - analytic).cwiseAbs().maxCoeff() < 2e-8, "actual virtual forward mean matches its analytic derivative");
  }
  const Eigen::Vector2d epsilon(1e-9, 0.);
  const Eigen::Vector2d legacy_fd = (camera.distort_f((p + epsilon).cast<float>()).cast<double>() -
                                     camera.distort_f((p - epsilon).cast<float>()).cast<double>()) / 2e-9;
  const double legacy_error = (legacy_fd - Hpoint.col(0)).norm();
  std::printf("CAMERA_PROJECTION_NEGATIVE old_float_wrapper_point_derivative_error=%.12g\n", legacy_error);
  check(legacy_error > 100., "old float-roundtrip negative control fails actual-mean derivative contract");
  const Eigen::Vector2d true_delta = camera.distort_d(p + epsilon) - camera.distort_d(p - epsilon);
  check(true_delta.norm() > 1e-7, "double projection preserves sub-float normalized-coordinate changes");
  const auto copy = camera.clone();
  check((copy->distort_d(p) - camera.distort_d(p)).norm() == 0., "concrete camera clone preserves double dispatch and calibration");
}

void verify_allocations(ov_core::CamBase &camera) {
#ifdef __GLIBC__
  Eigen::MatrixXd Hpoint(2, 2), Hcalib(2, 8);
  const Eigen::Vector2d point(.3125, -.21875);
  camera.compute_distort_jacobian(point, Hpoint, Hcalib);
  allocations = 0;
  count_allocations = true;
  const Eigen::MatrixXd intentional_copy = camera.get_value();
  count_allocations = false;
  const auto positive_control = allocations;
  check(positive_control > 0 && intentional_copy.rows() == 8, "allocation interceptor sees intentional parameter copy");
  allocations = 0;
  double sum = 0.;
  count_allocations = true;
  for (int i = 0; i < 10000; ++i) {
    const Eigen::Vector2d p = point + Eigen::Vector2d(1e-7 * i, -2e-8 * i);
    sum += camera.distort_d(p).sum();
    sum += camera.distort_f(p.cast<float>()).sum();
    camera.compute_distort_jacobian(p, Hpoint, Hcalib);
    sum += Hpoint(0, 0) + Hcalib(1, 1);
  }
  count_allocations = false;
  std::printf("CAMERA_PROJECTION_RESOURCE iterations=10000 allocations=%zu positive_control=%zu checksum=%.12g\n",
              allocations, positive_control, sum);
  check(allocations == 0 && sum != 0., "float/double projection and reused Jacobian buffers allocate no heap memory");
#else
  std::puts("CAMERA_PROJECTION_RESOURCE allocation count unavailable outside glibc");
#endif
}
} // namespace

int main() {
  std::vector<Eigen::Vector2d> points{{0., 0.}, {1e-12, -2e-12}, {1e-9, 0.}, {0., -1e-8}, {1e-7, -2e-7},
                                    {1e-4 * (1. - 1e-9), 0.}, {1e-4, 0.}, {1e-4 * (1. + 1e-9), 0.},
                                    {6e-5, 8e-5 * (1. - 1e-9)}, {6e-5, 8e-5 * (1. + 1e-9)}};
  for (double x : {-2.1, -.75, -.2, 0., .18, .85, 2.2})
    for (double y : {-1.7, -.35, 0., .6, 1.9}) points.emplace_back(x, y);
  std::array<Eigen::Matrix<double, 8, 1>, 3> calibrations;
  calibrations[0] << 450., 460., 640., 400., 0., 0., 0., 0.;
  calibrations[1] << 452.71432460722764, 452.9539497196888, 647.5970442085256, 394.62095013980564,
      .054524609058479147, .035670072247377976, -.02095895969142939, .003522306834791643;
  calibrations[2] << 413.271, 402.52, 639.7, 390.1, -.03, .011, .001, -.0015;
  for (bool equi : {false, true}) {
    std::unique_ptr<ov_core::CamBase> camera;
    if (equi) camera = std::make_unique<ov_core::CamEqui>(1280, 800);
    else camera = std::make_unique<ov_core::CamRadtan>(1280, 800);
    for (const auto &calib : calibrations) {
      camera->set_value(calib);
      for (const auto &point : points) verify_point(equi, *camera, calib, point);
      verify_actual_mean_derivatives(*camera, calib);
    }
    verify_allocations(*camera);
  }
  check(old_forward_error > 1e-5, "old float wrapper has measurable forward error against independent oracle");
  std::printf("CAMERA_PROJECTION_ORACLE forward_abs=%.12g forward_scaled=%.12g point_jac_abs=%.12g intrinsic_jac_abs=%.12g "
              "derivative_scaled=%.12g old_float_forward_abs=%.12g\n",
              max_forward_error, max_forward_relative, max_point_error, max_intrinsic_error, max_derivative_relative, old_forward_error);
  std::printf("CAMERA_PROJECTION_ABI CamBase=%zu CamRadtan=%zu CamEqui=%zu alignment=%zu\n",
              sizeof(ov_core::CamBase), sizeof(ov_core::CamRadtan), sizeof(ov_core::CamEqui), alignof(ov_core::CamBase));
  std::printf("CAMERA_PROJECTION_CHECKS %d/%d\n", checks - failures, checks);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
