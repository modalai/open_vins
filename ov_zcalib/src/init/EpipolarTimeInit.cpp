/* OpenVINS ov_zcalib — geometric camera/IMU bootstrap. GPL-3.0-or-later. */
#include "EpipolarTimeInit.h"
#include "../window/CamUndistort.h"
#include "utils/quat_ops.h"
#include <algorithm>
#include <chrono>
#include <limits>

namespace ov_zcalib {
namespace {
using V4 = Eigen::Vector4d;
using M4 = Eigen::Matrix4d;
constexpr double time_scale = 0.01;
constexpr double huber = 0.002; // angular Sampson residual, radians

struct Pair {
  double t0, t1;
  std::vector<Eigen::Vector3d> a, b;
  std::vector<double> row0, row1;
};

struct Problem {
  const std::vector<RawImu> &imu;
  std::vector<Eigen::Quaterniond> orientation;
  std::vector<Pair> pairs;
  Eigen::Matrix3d seed_R;
  double search;

  Problem(const std::vector<RawImu> &samples, const HandEyeResult &seed, double bound)
      : imu(samples), seed_R(ov_core::quat_2_Rot(seed.q_ItoC)), search(bound) {
    orientation.reserve(imu.size());
    orientation.emplace_back(Eigen::Quaterniond::Identity());
    for (size_t i = 1; i < imu.size(); ++i) {
      const double dt = imu[i].timestamp - imu[i - 1].timestamp;
      const Eigen::Vector3d w = 0.5 * (imu[i - 1].wm + imu[i].wm) - seed.bg;
      orientation.emplace_back((Eigen::Quaterniond(ov_core::exp_so3(-w * dt)) * orientation.back()).normalized());
    }
  }

  Eigen::Matrix3d at(double t) const {
    auto it = std::lower_bound(imu.begin(), imu.end(), t, [](const RawImu &s, double t) { return s.timestamp < t; });
    const size_t i = std::max<size_t>(1, std::min<size_t>(it - imu.begin(), imu.size() - 1));
    const double f = (t - imu[i - 1].timestamp) / (imu[i].timestamp - imu[i - 1].timestamp);
    return orientation[i - 1].slerp(f, orientation[i]).toRotationMatrix();
  }

  bool in_bounds(const V4 &x) const {
    return x.head<3>().squaredNorm() < 0.25 * 0.25 && std::abs(x(3) * time_scale) < search;
  }

  // The same observations contribute at every trial. No trial-dependent
  // inlier selection, endpoint clipping, or pose/translation priors.
  double cost(const V4 &x, int half = -1) const {
    if (!in_bounds(x))
      return 1e20;
    const Eigen::Matrix3d R = ov_core::exp_so3(x.head<3>()) * seed_R;
    const double td = time_scale * x(3);
    const double cut = 0.5 * (pairs[pairs.size() / 2 - 1].t1 + pairs[pairs.size() / 2].t0);
    double total = 0.0;
    int count = 0;
    std::vector<Eigen::Vector3d> u, v, cross;
    std::vector<double> weights;
    for (const Pair &p : pairs) {
      if ((half == 0 && p.t1 >= cut) || (half == 1 && p.t0 <= cut))
        continue; // the halves share no frame interval
      const size_t n = p.a.size();
      u.resize(n);
      v.resize(n);
      cross.resize(n);
      weights.assign(n, 1.0);
      for (size_t i = 0; i < n; ++i) {
        // Rotate each bearing to its center-row time before eliminating the
        // translation direction. This also works for global-shutter cameras.
        // The common center-frame rotation cancels in the epipolar distance;
        // express both bearings directly in the gyro trajectory's reference.
        u[i] = at(p.t0 + td + p.row0[i]).transpose() * R.transpose() * p.a[i];
        v[i] = at(p.t1 + td + p.row1[i]).transpose() * R.transpose() * p.b[i];
        cross[i] = u[i].cross(v[i]);
      }
      Eigen::Vector3d normal = Eigen::Vector3d::Zero();
      for (int iter = 0; iter < 3; ++iter) {
        Eigen::Matrix3d N = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < n; ++i)
          N.noalias() += weights[i] * cross[i] * cross[i].transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(N);
        if (eig.info() != Eigen::Success)
          return 1e20;
        normal = eig.eigenvectors().col(0);
        for (size_t i = 0; i < n; ++i) {
          const double e = normal.dot(cross[i]);
          const double d =
              std::max(0.01, normal.cross(u[i]).squaredNorm() + normal.cross(v[i]).squaredNorm() - 2 * e * e);
          const double r = std::abs(e) / std::sqrt(d);
          weights[i] = std::min(1.0, huber / std::max(r, 1e-12)) / d;
        }
      }
      for (size_t i = 0; i < n; ++i) {
        const double e = normal.dot(cross[i]);
        const double d =
            std::max(0.01, normal.cross(u[i]).squaredNorm() + normal.cross(v[i]).squaredNorm() - 2 * e * e);
        const double r = std::abs(e) / std::sqrt(d);
        total += r <= huber ? r * r : 2 * huber * r - huber * huber;
        ++count;
      }
    }
    return count >= 200 ? 1e6 * total / count : 1e20;
  }

  V4 gradient(const V4 &x, int half) const {
    V4 g;
    for (int k = 0; k < 4; ++k) {
      V4 a = x, b = x;
      const double eps = k == 3 ? 1e-4 : 1e-5;
      a(k) += eps;
      b(k) -= eps;
      g(k) = (cost(a, half) - cost(b, half)) / (2 * eps);
    }
    return g;
  }

  // Four-dimensional BFGS with bounded, monotone line search. The numerical
  // gradient differentiates through the eliminated translation direction,
  // including its robust weights, rather than freezing a nuisance optimum.
  bool fit(V4 &x, int half) const {
    M4 inv = M4::Identity();
    double f = cost(x, half);
    if (!(f < 1e10))
      return false;
    V4 g = gradient(x, half);
    for (int iter = 0; iter < 35; ++iter) {
      if (g.cwiseAbs().maxCoeff() < 1e-4)
        return true;
      V4 dir = -inv * g;
      if (!(dir.dot(g) < 0.0)) {
        inv.setIdentity();
        dir = -g;
      }
      double step = std::min(
          1.0, std::min(0.08 / std::max(dir.head<3>().norm(), 1e-12), 0.8 / std::max(std::abs(dir(3)), 1e-12)));
      V4 next = x;
      double nf = f;
      bool accepted = false;
      for (int ls = 0; ls < 18; ++ls, step *= 0.5) {
        next = x + step * dir;
        nf = cost(next, half);
        if (nf <= f + 1e-4 * step * dir.dot(g)) {
          accepted = true;
          break;
        }
      }
      if (!accepted)
        return g.cwiseAbs().maxCoeff() < 0.02;
      const V4 ng = gradient(next, half), dx = next - x, dg = ng - g;
      const double curvature = dx.dot(dg);
      if (curvature > 1e-10 * dx.norm() * dg.norm()) {
        const M4 A = M4::Identity() - dx * dg.transpose() / curvature;
        inv = A * inv * A.transpose() + dx * dx.transpose() / curvature;
      }
      x = next;
      g = ng;
      f = nf;
    }
    return g.cwiseAbs().maxCoeff() < 0.02;
  }

  // Marginalize rotation in the profile curvature. Use frame pairs, not
  // individual pixels, for the effective evidence count: tracks in a pair
  // are correlated. Time-split agreement remains a separate gate.
  double sigma(const V4 &x, int half) const {
    const double eps[4] = {2e-4, 2e-4, 2e-4, 0.005};
    const double f = cost(x, half);
    M4 H;
    for (int i = 0; i < 4; ++i) {
      V4 a = x, b = x;
      a(i) += eps[i];
      b(i) -= eps[i];
      // A bound penalty is not measurement curvature and cannot certify td.
      if (!in_bounds(a) || !in_bounds(b))
        return -1.0;
      H(i, i) = (cost(a, half) - 2 * f + cost(b, half)) / (eps[i] * eps[i]);
      for (int j = 0; j < i; ++j) {
        V4 aa = x, ab = x, ba = x, bb = x;
        aa(i) += eps[i];
        aa(j) += eps[j];
        ab(i) += eps[i];
        ab(j) -= eps[j];
        ba(i) -= eps[i];
        ba(j) += eps[j];
        bb(i) -= eps[i];
        bb(j) -= eps[j];
        if (!in_bounds(aa) || !in_bounds(ab) || !in_bounds(ba) || !in_bounds(bb))
          return -1.0;
        H(i, j) = H(j, i) = (cost(aa, half) - cost(ab, half) - cost(ba, half) + cost(bb, half)) / (4 * eps[i] * eps[j]);
      }
    }
    Eigen::SelfAdjointEigenSolver<M4> e(H);
    if (e.info() != Eigen::Success || !(e.eigenvalues()(0) > 1e-8))
      return -1.0;
    const double variance = H.ldlt().solve(V4::Unit(3))(3);
    const double n = half < 0 ? pairs.size() : pairs.size() / 2 - 1;
    return time_scale * std::sqrt(2 * std::max(f, 1e-4) * variance / std::max(n - 4, 1.0));
  }
};
} // namespace

EpipolarTimeResult EpipolarTimeInit::solve(const std::deque<FrameObs> &frames, const std::vector<RawImu> &imu,
                                           const CamCalib &camera, int cam_id, const HandEyeResult &seed,
                                           double search_s, double split_tol_s) {
  EpipolarTimeResult out;
  out.attempted = true;
  out.seed_td = out.td = seed.td;
  out.q_ItoC = seed.q_ItoC;
  const auto start = std::chrono::steady_clock::now();
  auto done = [&]() {
    out.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return out;
  };
  if (imu.size() < 8 || camera.img_h <= 0 || !(search_s > 0 && split_tol_s > 0))
    return done();
  for (size_t i = 1; i < imu.size(); ++i)
    if (!(imu[i].timestamp > imu[i - 1].timestamp) || imu[i].timestamp - imu[i - 1].timestamp > 0.02)
      return done();
  Problem p(imu, seed, search_s);
  std::vector<const FrameObs *> fs;
  for (const auto &f : frames)
    if ((int)f.cam == cam_id && (fs.empty() || f.timestamp > fs.back()->timestamp))
      fs.push_back(&f);
  const double readout = camera.rolling ? camera.tr : 0.0;
  double last = -std::numeric_limits<double>::infinity();
  size_t first = 0;
  for (size_t i = 1; i < fs.size(); ++i) {
    const FrameObs &b = *fs[i];
    if (b.timestamp - last < 0.049)
      continue;
    while (first + 1 < i && fs[first + 1]->timestamp <= b.timestamp - 1.0 / 15.0)
      ++first;
    const FrameObs &a = *fs[first];
    const double dt = b.timestamp - a.timestamp;
    if (dt < 0.04 || dt > 0.12)
      continue;
    if (a.timestamp - search_s - readout / 2 <= imu.front().timestamp ||
        b.timestamp + search_s + readout / 2 >= imu.back().timestamp)
      continue;
    const double rate =
        ov_core::log_so3(p.at(b.timestamp + seed.td) * p.at(a.timestamp + seed.td).transpose()).norm() / dt;
    if (rate < 0.15 || rate > 6.0)
      continue;
    Pair pair;
    pair.t0 = a.timestamp;
    pair.t1 = b.timestamp;
    for (const auto &v : b.pts)
      for (const auto &u : a.pts)
        if (u.id == v.id) {
          pair.a.push_back(CamUndistort::bearing(Eigen::Vector2d(u.u, u.v), camera.cam, camera.fisheye));
          pair.b.push_back(CamUndistort::bearing(Eigen::Vector2d(v.u, v.v), camera.cam, camera.fisheye));
          pair.row0.push_back((u.v / camera.img_h - 0.5) * readout);
          pair.row1.push_back((v.v / camera.img_h - 0.5) * readout);
          break;
        }
    // Translation elimination needs denser support than rotation-only initialization.
    if (pair.a.size() < 25)
      continue;
    last = b.timestamp;
    p.pairs.push_back(std::move(pair));
  }
  // Bound cost without favoring the beginning or end of the motion span.
  if (p.pairs.size() > 200) {
    std::vector<Pair> reduced;
    for (size_t k = 0; k < 200; ++k)
      reduced.push_back(std::move(p.pairs[k * (p.pairs.size() - 1) / 199]));
    p.pairs = std::move(reduced);
  }
  out.pairs = (int)p.pairs.size();
  if (!out.has_support())
    return done();
  V4 x(0, 0, 0, seed.td / time_scale);
  out.cost_before = p.cost(x);
  if (!p.fit(x, -1))
    return done();
  out.td = time_scale * x(3);
  out.q_ItoC = ov_core::rot_2_quat(ov_core::exp_so3(x.head<3>()) * p.seed_R);
  out.cost_after = p.cost(x);
  out.sigma_td = p.sigma(x, -1);
  if (!(out.sigma_td > 0 && out.sigma_td <= split_tol_s) || std::abs(out.td) >= search_s - 0.002)
    return done();
  V4 a = x, b = x;
  if (!p.fit(a, 0) || !p.fit(b, 1))
    return done();
  const double sa = p.sigma(a, 0), sb = p.sigma(b, 1);
  out.split_delta = time_scale * std::abs(a(3) - b(3));
  out.ok = sa > 0 && sb > 0 && sa <= split_tol_s && sb <= split_tol_s && out.split_delta <= split_tol_s &&
           time_scale * std::max(std::abs(a(3)), std::abs(b(3))) < search_s - 0.002 &&
           time_scale * std::max(std::abs(a(3) - x(3)), std::abs(b(3) - x(3))) <= split_tol_s;
  return done();
}
} // namespace ov_zcalib
