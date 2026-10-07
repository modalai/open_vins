/*
 * Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Reprojection derivatives against the legacy model and an independent
 * finite-difference oracle, including the optical-center limit.
 */
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "DistortDouble.h"
#include "Factor_ImageReprojCalib.h"
#include "State_JPLQuatLocal.h"
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"

namespace {
int failures = 0;
void check(double error, double tolerance, const char *name) {
  std::printf("[%s] %s: %.9g (tolerance %.3g)\n", error <= tolerance ? "PASS" : "FAIL", name, error, tolerance);
  if (!(error <= tolerance)) ++failures;
}

using Intrinsics = Eigen::Matrix<double, 8, 1>;
using IntrinsicsJac = Eigen::Matrix<double, 2, 8>;

void check_rotation() {
  std::mt19937 rng(5389);
  std::normal_distribution<double> sample(0, 1);
  double parity = 0;
  for (int i = 0; i < 1000; ++i) {
    const Eigen::Vector4d q = Eigen::Vector4d(sample(rng), sample(rng), sample(rng), sample(rng)).normalized();
    const Eigen::Matrix3d skew = ov_core::skew_x(q.head<3>());
    // Exact historical expression, including its dynamic temporary.
    const Eigen::MatrixXd old = (2*std::pow(q(3),2)-1)*Eigen::MatrixXd::Identity(3,3) - 2*q(3)*skew +
                                2*q.head<3>()*q.head<3>().transpose();
    Eigen::internal::set_is_malloc_allowed(false);
    const Eigen::Matrix3d fixed = ov_core::quat_2_Rot(q);
    Eigen::internal::set_is_malloc_allowed(true);
    parity = std::max(parity, (old-fixed).cwiseAbs().maxCoeff());
  }
  check(parity, 1e-15, "fixed-size JPL rotation historical parity, 1000 quaternions");
}

void check_distortion(bool fisheye) {
  Intrinsics c;
  c << 456.7, 461.2, 640.3, 402.1, 0.065, -0.01, 0.002, -0.001;
  ov_core::CamEqui equi(1280, 800);
  ov_core::CamRadtan radtan(1280, 800);
  ov_core::CamBase &legacy = fisheye ? static_cast<ov_core::CamBase &>(equi) : static_cast<ov_core::CamBase &>(radtan);
  legacy.set_value(c);
  double parity = 0.0, fd = 0.0, optional = 0.0;
  std::mt19937 rng(7214);
  std::uniform_real_distribution<double> sample(-2.0, 2.0);
  for (int i = 0; i < 300; ++i) {
    Eigen::Vector2d uv(sample(rng), sample(rng));
    Eigen::Matrix2d Jn, Jn_only;
    IntrinsicsJac Jc;
    Eigen::MatrixXd old_n, old_c;
    legacy.compute_distort_jacobian(uv, old_n, old_c);
    Eigen::internal::set_is_malloc_allowed(false);
    if (fisheye) {
      ov_init::equidistant_jacobian_double(c, uv, Jn, &Jc);
      ov_init::equidistant_jacobian_double(c, uv, Jn_only);
    } else {
      ov_init::radtan_jacobian_double(c, uv, Jn, &Jc);
      ov_init::radtan_jacobian_double(c, uv, Jn_only);
    }
    Eigen::internal::set_is_malloc_allowed(true);
    parity = std::max(parity, (Jn-old_n).cwiseAbs().maxCoeff());
    parity = std::max(parity, (Jc-old_c).cwiseAbs().maxCoeff());
    optional = std::max(optional, (Jn-Jn_only).cwiseAbs().maxCoeff());
    for (int k = 0; k < 10; ++k) {
      const double eps = k < 2 ? 1e-6 : 1e-5;
      Eigen::Vector2d up = uv, um = uv;
      Intrinsics cp = c, cm = c;
      if (k < 2) { up(k) += eps; um(k) -= eps; }
      else { cp(k-2) += eps; cm(k-2) -= eps; }
      Eigen::Vector2d col = (ov_init::distort_double(cp, up, fisheye) - ov_init::distort_double(cm, um, fisheye))/(2.0*eps);
      fd = std::max(fd, (col - (k < 2 ? Eigen::Vector2d(Jn.col(k)) : Eigen::Vector2d(Jc.col(k-2)))).cwiseAbs().maxCoeff());
    }
  }
  check(parity, 5e-10, fisheye ? "equidistant noncenter legacy parity" : "radtan legacy parity");
  check(fd, 2e-6, fisheye ? "equidistant derivative finite differences" : "radtan derivative finite differences");
  check(optional, 0.0, fisheye ? "equidistant optional intrinsics" : "radtan optional intrinsics");

  if (fisheye) {
    double center_fd = 0.0, core_center_fd = 0.0, continuity = 0.0;
    for (double radius : {0.0, 0.5e-8, 1e-8, 1.01e-8, 1e-7, 0.999e-4, 1.001e-4}) {
      Eigen::Vector2d uv(radius, 0.0);
      Eigen::Matrix2d Jn;
      IntrinsicsJac Jc;
      ov_init::equidistant_jacobian_double(c, uv, Jn, &Jc);
      Eigen::MatrixXd core_n, core_c;
      legacy.compute_distort_jacobian(uv, core_n, core_c);
      for (int k = 0; k < 2; ++k) {
        Eigen::Vector2d up = uv, um = uv;
        up(k) += 1e-6; um(k) -= 1e-6;
        Eigen::Vector2d col = (ov_init::distort_double(c, up, true)-ov_init::distort_double(c, um, true))/2e-6;
        center_fd = std::max(center_fd, (col-Jn.col(k)).cwiseAbs().maxCoeff());
        // CamBase::distort_d returns float-quantized pixels, so use the same
        // physical forward model in double for this finite-difference oracle.
        core_center_fd = std::max(core_center_fd, (col-core_n.col(k)).cwiseAbs().maxCoeff());
      }
      if (radius <= 1.01e-8)
        continuity = std::max(continuity, (Jn-c.head<2>().asDiagonal().toDenseMatrix()).cwiseAbs().maxCoeff());
    }
    check(center_fd, 1e-6, "equidistant optical center / series transition finite differences");
    check(core_center_fd, 1e-6, "filter CamEqui optical center / near-center finite differences");
    check(continuity, 1e-10, "equidistant optical-center derivative continuity");
  }
}

void check_factor(bool fisheye, bool center) {
  using ov_init::zbft_sfm::Factor_ImageReprojCalib;
  using ov_init::zbft_sfm::State_JPLQuatLocal;
  const std::array<int, 6> size{4, 3, 3, 4, 3, 8};
  std::array<std::array<double, 8>, 6> x{};
  x[0] = {0.0, 0.0, 0.0, 1.0}; x[3] = x[0];
  x[2] = center ? std::array<double, 8>{0.0, 0.0, 4.0} : std::array<double, 8>{1.7, -0.6, 3.2};
  x[5] = {459.0, 470.0, 642.0, 402.0, .061, -.012, .005, -.001};
  std::array<const double *, 6> p;
  std::array<std::array<double, 16>, 6> J{};
  std::array<double *, 6> jp;
  for (int i = 0; i < 6; ++i) { p[i] = x[i].data(); jp[i] = J[i].data(); }
  Factor_ImageReprojCalib f(Eigen::Vector2d(637, 403), 1.7, fisheye);
  f.gate = 0.43; // Check whitening and gating along with all parameter blocks.
  Eigen::Vector2d residual;
  Eigen::internal::set_is_malloc_allowed(false);
  f.Evaluate(p.data(), residual.data(), jp.data());
  Eigen::internal::set_is_malloc_allowed(true);
  double worst = 0.0, optional = 0.0;
  State_JPLQuatLocal quat;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < (size[i] == 4 ? 3 : size[i]); ++j) {
      const auto saved = x[i];
      const double eps = 1e-6;
      Eigen::Vector2d rp, rm;
      if (size[i] == 4) {
        Eigen::Vector3d d = Eigen::Vector3d::Zero(); d(j) = eps;
        quat.Plus(saved.data(), d.data(), x[i].data());
        f.Evaluate(p.data(), rp.data(), nullptr);
        d(j) = -eps;
        quat.Plus(saved.data(), d.data(), x[i].data());
      } else {
        x[i][j] += eps;
        f.Evaluate(p.data(), rp.data(), nullptr);
        x[i][j] -= 2.0*eps;
      }
      f.Evaluate(p.data(), rm.data(), nullptr);
      x[i] = saved;
      for (int row = 0; row < 2; ++row)
        worst = std::max(worst, std::abs((rp(row)-rm(row))/(2.0*eps)-J[i][row*size[i]+j]));
    }
  }
  // VINS normally keeps extrinsics and intrinsics constant. Exercise null
  // output blocks as well as full calibration Jacobians and residual-only.
  jp[3] = jp[4] = jp[5] = nullptr;
  Eigen::Vector2d r_fixed, r_only;
  Eigen::internal::set_is_malloc_allowed(false);
  f.Evaluate(p.data(), r_fixed.data(), jp.data());
  f.Evaluate(p.data(), r_only.data(), nullptr);
  Eigen::internal::set_is_malloc_allowed(true);
  optional = std::max((r_fixed-residual).norm(), (r_only-residual).norm());
  check(worst, 2e-6, center ? "full factor at optical center FD" : (fisheye ? "full equidistant factor FD" : "full radtan factor FD"));
  check(optional, 0.0, "fixed-calibration / residual-only parity");
}

void benchmark(int count) {
  Intrinsics c;
  c << 456.7, 461.2, 640.3, 402.1, 0.065, -0.01, 0.002, -0.001;
  ov_core::CamEqui legacy(1280, 800);
  Eigen::MatrixXd old_n, old_c;
  Eigen::Matrix2d Jn;
  volatile double checksum = 0.0;
  // Interleave trials to limit thermal/frequency bias. Print timings only;
  // performance is evidence, never a noisy CI correctness assertion.
  for (int repeat = 0; repeat < 5; ++repeat) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
      Eigen::Vector2d uv(0.5 + (i%17)*0.005, -0.25);
      legacy.set_value(c);
      legacy.compute_distort_jacobian(uv, old_n, old_c);
      checksum += old_n(0, 0);
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
      Eigen::Vector2d uv(0.5 + (i%17)*0.005, -0.25);
      ov_init::equidistant_jacobian_double(c, uv, Jn);
      checksum += Jn(0, 0);
    }
    auto t2 = std::chrono::steady_clock::now();
    std::printf("BENCH equidistant fixed-calibration count=%d legacy_ms=%.6f optimized_ms=%.6f checksum=%.12g\n", count,
                std::chrono::duration<double, std::milli>(t1-t0).count(), std::chrono::duration<double, std::milli>(t2-t1).count(), checksum);
  }
}
} // namespace

int main(int argc, char **argv) {
  check_rotation();
  check_distortion(false); check_distortion(true);
  check_factor(false, false); check_factor(true, false); check_factor(true, true);
  if (argc > 1) benchmark(std::max(1, std::atoi(argv[1])));
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
