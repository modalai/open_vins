/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>
#include "ceres/State_JPLQuatLocal.h"
#include "ceres/Factor_GenericPrior.h"
#include "ceres/Factor_ImageReprojCalib.h"
#include "ceres/Factor_ImuCPIv1.h"
#include "utils/quat_ops.h"

namespace {
using V3 = Eigen::Vector3d;
using V4 = Eigen::Vector4d;
using Mat = Eigen::MatrixXd;
using Jplus = Eigen::Matrix<double, 4, 3, Eigen::RowMajor>;
using Jminus = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
int checks = 0, failures = 0;
double max_plus = 0., max_minus = 0., max_roundtrip = 0., max_local = 0., max_ambient = 0.;
void check(bool ok, const char *message) {
  ++checks;
  if (!ok) { ++failures; std::printf("FAIL: %s\n", message); }
}
Jplus plus_jacobian(const ov_init::State_JPLQuatLocal &manifold, const V4 &q) {
  Jplus J;
#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  manifold.PlusJacobian(q.data(), J.data());
#else
  manifold.ComputeJacobian(q.data(), J.data());
#endif
  return J;
}
void manifold_contract() {
  ov_init::State_JPLQuatLocal manifold;
  for (double angle : {0., .2, 1.7, 3.141592653589793 - 1e-5})
    for (double sign : {-1., 1.}) {
      const V4 q = sign * ov_core::rot_2_quat(ov_core::exp_so3(angle * V3(.3, -.7, .2).normalized()));
      V4 zero; const V3 d0 = V3::Zero(); manifold.Plus(q.data(), d0.data(), zero.data());
      check((zero - q).norm() < 1e-14, "Plus(q,0) preserves either quaternion representative");
      const Jplus J = plus_jacobian(manifold, q); Jplus fd;
      for (int k = 0; k < 3; ++k) {
        V3 d = V3::Zero(); d(k) = 1e-6; V4 p, m;
        manifold.Plus(q.data(), d.data(), p.data()); d(k) = -1e-6;
        manifold.Plus(q.data(), d.data(), m.data()); fd.col(k) = (p - m) / 2e-6;
      }
      const double ep = (J - fd).cwiseAbs().maxCoeff(); max_plus = std::max(max_plus, ep);
      check(ep < 2e-8, "Ceres PlusJacobian differentiates the actual retraction");
#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
      Jminus L, lfd; manifold.MinusJacobian(q.data(), L.data());
      for (int k = 0; k < 4; ++k) {
        V4 p = q, m = q; p(k) += 1e-6; m(k) -= 1e-6; p.normalize(); m.normalize();
        V3 dp, dm; manifold.Minus(p.data(), q.data(), dp.data()); manifold.Minus(m.data(), q.data(), dm.data());
        lfd.col(k) = (dp - dm) / 2e-6;
      }
      const double em = (L - lfd).cwiseAbs().maxCoeff(); max_minus = std::max(max_minus, em);
      check(em < 2e-8, "Ceres MinusJacobian differentiates the unit-quaternion inverse map");
      check((L * J - Eigen::Matrix3d::Identity()).norm() < 2e-14 &&
            (J * L - (Eigen::Matrix4d::Identity() - q * q.transpose())).norm() < 2e-14,
            "plus/minus derivatives are inverse on the tangent and project out the radial gauge");
      for (double scale : {1e-8, .01, .7}) {
        const V3 delta = scale * V3(.3, -.2, .4); V4 y; V3 recovered;
        manifold.Plus(q.data(), delta.data(), y.data()); manifold.Minus(y.data(), q.data(), recovered.data());
        V4 inplace = q; manifold.Plus(inplace.data(), delta.data(), inplace.data());
        check((inplace - y).norm() < 1e-14, "in-place quaternion retraction preserves the nonaliased result");
        const double error = (recovered - delta).norm(); max_roundtrip = std::max(max_roundtrip, error);
        check(error < 2e-12, "Minus(Plus(q,delta),q) returns delta with the JPL sign and scale");
      }
#endif
    }
}
Eigen::VectorXd residual(const ceres::CostFunction &factor, const std::vector<Eigen::VectorXd> &values) {
  std::vector<const double *> p; for (const auto &x : values) p.push_back(x.data());
  Eigen::VectorXd r(factor.num_residuals()); factor.Evaluate(p.data(), r.data(), nullptr); return r;
}
void factor_contract(const ceres::CostFunction &factor, std::vector<Eigen::VectorXd> values,
                     std::initializer_list<int> quaternion_blocks) {
  std::vector<const double *> p; std::vector<std::vector<double>> storage;
  std::vector<double *> j;
  for (const auto &x : values) { p.push_back(x.data()); storage.emplace_back(factor.num_residuals() * x.size()); }
  for (auto &x : storage) j.push_back(x.data());
  Eigen::VectorXd r(factor.num_residuals()); factor.Evaluate(p.data(), r.data(), j.data());
  ov_init::State_JPLQuatLocal manifold;
  for (int block : quaternion_blocks) {
    const V4 q = values[block]; const Jplus D = plus_jacobian(manifold, q);
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>> H(j[block], factor.num_residuals(), 4);
    Mat fd_local(factor.num_residuals(), 3), fd_ambient(factor.num_residuals(), 4);
    for (int k = 0; k < 3; ++k) {
      V3 d = V3::Zero(); d(k) = 1e-6;
      manifold.Plus(q.data(), d.data(), values[block].data()); const auto rp = residual(factor, values);
      d(k) = -1e-6; manifold.Plus(q.data(), d.data(), values[block].data()); const auto rm = residual(factor, values);
      fd_local.col(k) = (rp - rm) / 2e-6;
    }
    for (int k = 0; k < 4; ++k) {
      values[block] = q; values[block](k) += 1e-6; values[block].normalize(); const auto rp = residual(factor, values);
      values[block] = q; values[block](k) -= 1e-6; values[block].normalize(); const auto rm = residual(factor, values);
      fd_ambient.col(k) = (rp - rm) / 2e-6;
    }
    values[block] = q;
    const double el = (H * D - fd_local).cwiseAbs().maxCoeff(); max_local = std::max(max_local, el);
    const double ea = (H - fd_ambient).cwiseAbs().maxCoeff(); max_ambient = std::max(max_ambient, ea);
    check(el < 2e-5, "assembled Ceres factor/manifold Jacobian matches physical tangent finite differences");
    check(ea < 2e-5, "Ceres factor ambient Jacobian matches a normalized quaternion perturbation");
  }
}
void factors() {
  const V4 q = ov_core::rot_2_quat(ov_core::exp_so3(V3(.3, -.2, .4)));
  const V4 reference = ov_core::rot_2_quat(ov_core::exp_so3(V3(.1, .2, -.1)));
  for (const std::string kind : {"quat", "quat_yaw"}) {
    const int n = kind == "quat" ? 3 : 1;
    ov_init::Factor_GenericPrior prior(reference, {kind}, 2. * Mat::Identity(n, n), Mat::Zero(n, 1));
    factor_contract(prior, {q}, {0});
    factor_contract(prior, {-q}, {0});
  }
  for (bool fisheye : {false, true}) {
    ov_init::Factor_ImageReprojCalib image(Eigen::Vector2d(315., 205.), 1.2, fisheye);
    Eigen::VectorXd intr(8); intr << 300, 310, 320, 240, .02, -.003, .001, -.0002;
    const V4 qc = ov_core::rot_2_quat(ov_core::exp_so3(V3(-.2, .1, .03)));
    factor_contract(image, {q, V3(.1, -.2, .3), V3(.4, .6, 4.), qc, V3(.03, -.02, .01), intr}, {0, 3});
  }
  V3 gravity(0, 0, 9.81), alpha(.01, -.02, .05), beta(.03, .01, .2), ba(.02, -.03, .04), bg(.004, -.003, .002);
  V4 dq = ov_core::rot_2_quat(ov_core::exp_so3(V3(.03, -.04, .01)));
  Eigen::Matrix3d Jq = -.1 * Eigen::Matrix3d::Identity(), Jb = .01 * Eigen::Matrix3d::Identity(),
                  Ja = .001 * Eigen::Matrix3d::Identity(), Hb = .1 * Eigen::Matrix3d::Identity(),
                  Ha = .005 * Eigen::Matrix3d::Identity(), Hq = .002 * Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 15, 15> P = .01 * Eigen::Matrix<double, 15, 15>::Identity();
  ov_init::Factor_ImuCPIv1 imu(.1, gravity, alpha, beta, dq, ba, bg, Jq, Jb, Ja, Hb, Ha, P, Hq);
  factor_contract(imu, {q, bg + V3(.02, -.01, .03), V3(.2, -.1, .3), ba + V3(.01, .02, -.01),
                       V3(.1, .2, .3), reference, bg, V3(.3, .2, .1), ba, V3(.2, .3, .5)}, {0, 5});
}
}
int main() {
  manifold_contract(); factors();
  std::printf("CERES_MANIFOLD %s checks=%d failures=%d plusFD=%.3e minusFD=%.3e roundtrip=%.3e localFD=%.3e ambientFD=%.3e\n",
      failures ? "FAIL" : "PASS", checks, failures, max_plus, max_minus, max_roundtrip, max_local, max_ambient);
  return failures ? 1 : 0;
}
