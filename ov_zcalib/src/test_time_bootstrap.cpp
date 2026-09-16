/* Independent motion/clock oracles for the geometric time bootstrap. */
#include "init/EpipolarTimeInit.h"
#include "init/TimeOffsetInit.h"
#include "ceres_free/DistortDouble.h"
#include "utils/quat_ops.h"
#include <cstdio>
#include <random>

using namespace ov_zcalib;
static int failures = 0;
#define CHECK(x, msg)                                                                                                  \
  do {                                                                                                                 \
    if (!(x)) {                                                                                                        \
      std::printf("FAIL: %s\n", msg);                                                                                  \
      ++failures;                                                                                                      \
    }                                                                                                                  \
  } while (0)

static Eigen::Matrix3d pose(double t) {
  return (Eigen::AngleAxisd(0.3 * std::sin(1.2 * t), Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(0.4 * std::sin(1.8 * t + 0.5), Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(0.25 * std::sin(2.3 * t), Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

// Analytic -vee(Rdot R^T), independent of the initializer's gyro integration.
static Eigen::Vector3d gyro(double t) {
  const Eigen::Matrix3d z = Eigen::AngleAxisd(0.3 * std::sin(1.2 * t), Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Matrix3d y =
      Eigen::AngleAxisd(0.4 * std::sin(1.8 * t + 0.5), Eigen::Vector3d::UnitY()).toRotationMatrix();
  return -(Eigen::Vector3d::UnitZ() * (0.36 * std::cos(1.2 * t)) +
           z * Eigen::Vector3d::UnitY() * (0.72 * std::cos(1.8 * t + 0.5)) +
           z * y * Eigen::Vector3d::UnitX() * (0.575 * std::cos(2.3 * t)));
}

static Eigen::Vector3d position(double t) {
  return Eigen::Vector3d(0.25 * std::sin(0.9 * t), 0.18 * std::sin(1.1 * t + 0.2), 0.1 * std::cos(1.4 * t));
}

int main() {
  // Both signs and a drifting visual-rate bias hidden by interleaved splits.
  for (double truth : {-0.011, 0.007}) {
    auto wave = [](double t) { return 2.0 + 0.4 * std::sin(6 * t) + 0.2 * std::sin(15.7 * t); };
    std::vector<RawImu> imu;
    std::vector<CamRateSample> cam;
    for (int i = 0; i < 13000; ++i) {
      RawImu s;
      s.timestamp = 0.001 * i;
      s.wm = Eigen::Vector3d(wave(s.timestamp), 0, 0);
      imu.push_back(s);
    }
    for (int i = 0; i < 720; ++i) {
      CamRateSample c;
      c.t_mid = 0.3 + i / 60.0;
      c.rate = wave(c.t_mid + truth);
      cam.push_back(c);
    }
    const auto a = TimeOffsetInit::solve(cam, imu, 0.08, 0.002);
    CHECK(a.ok && std::abs(a.td - truth) < 1e-4, "xcorr sign/sub-grid accuracy");
    CHECK(a.temporally_consistent(0.004), "stable offset refused");
    for (size_t i = 0; i < cam.size(); ++i)
      cam[i].rate = wave(cam[i].t_mid + truth + (i < cam.size() / 2 ? -0.012 : 0.012));
    const auto b = TimeOffsetInit::solve(cam, imu, 0.08, 0.002);
    CHECK(b.peak_corr > 0.9 && b.td_split_delta < 0.004, "biased fixture must fool the old confidence test");
    CHECK(!b.temporally_consistent(0.004) && b.temporal_split_delta > 0.02, "time-varying bias accepted");
  }

  for (bool fisheye : {false, true}) {
    const double truth = fisheye ? -0.011 : 0.007;
    CamCalib cam;
    cam.fisheye = fisheye;
    cam.img_w = 1280;
    cam.img_h = 720;
    cam.cam << 510, 505, 640, 360, -0.04, 0.008, 0.0005, -0.0003;
    cam.rolling = true;
    cam.tr = 0.008;
    const Eigen::Matrix3d R = ov_core::exp_so3(Eigen::Vector3d(0.1, -0.2, 0.05));
    const Eigen::Vector3d bg(0.002, -0.003, 0.001);
    std::vector<RawImu> imu;
    for (int i = 0; i <= 13000; ++i) {
      RawImu s;
      s.timestamp = 0.001 * i;
      s.wm = gyro(s.timestamp) + bg;
      imu.push_back(s);
    }
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> uni(-1, 1);
    std::vector<Eigen::Vector3d> points;
    for (int j = 0; j < 90; ++j)
      points.emplace_back(1.8 * uni(rng), 1.0 * uni(rng), 3.0 + uni(rng));
    std::deque<FrameObs> frames;
    for (int i = 0; i < 660; ++i) {
      FrameObs f;
      f.timestamp = 0.5 + i / 60.0;
      f.cam = 0;
      f.seq = i;
      f.exposure_s = 0.010; // already centered; must not change the offset
      for (size_t j = 0; j < points.size(); ++j) {
        double row = 0;
        Eigen::Vector2d uv;
        for (int it = 0; it < 6; ++it) {
          const double t = f.timestamp + truth + row;
          const Eigen::Vector3d pc = R * pose(t) * (points[j] - position(t));
          uv = ov_init::distort_double(cam.cam, pc.head<2>() / pc(2), fisheye);
          row = (uv(1) / cam.img_h - 0.5) * cam.tr;
        }
        if (uv(0) < 5 || uv(0) > 1275 || uv(1) < 5 || uv(1) > 715)
          continue;
        FrameObsPoint pt;
        pt.id = (uint32_t)j;
        pt.u = uv(0);
        pt.v = uv(1);
        f.pts.push_back(pt);
      }
      frames.push_back(f);
    }
    HandEyeResult seed;
    seed.ok = true;
    seed.bg = bg;
    seed.td = truth - 0.014;
    seed.q_ItoC = ov_core::rot_2_quat(ov_core::exp_so3(Eigen::Vector3d(0.018, -0.025, 0.012)) * R);
    const auto a = EpipolarTimeInit::solve(frames, imu, cam, 0, seed, 0.08, 0.004);
    std::printf("geometry %s: ok=%d td=%+.6f truth=%+.6f ms split=%.4f sigma=%.4f ms cost %.4g -> %.4g\n",
                fisheye ? "equi" : "radtan", a.ok, a.td * 1e3, truth * 1e3, a.split_delta * 1e3, a.sigma_td * 1e3,
                a.cost_before, a.cost_after);
    CHECK(a.ok && std::abs(a.td - truth) < 0.0005,
          "geometry recovery with translation, rolling shutter, and poor seed");
    for (auto &f : frames) {
      f.timestamp += 0.019;
      f.exposure_s = 0.020;
    }
    seed.td -= 0.019;
    const auto b = EpipolarTimeInit::solve(frames, imu, cam, 0, seed, 0.08, 0.004);
    CHECK(b.ok && std::abs(b.td - (truth - 0.019)) < 0.0005, "camera-clock shift/exposure invariance");
    // Put the same physical observations near the search edge. A truncated
    // search must not masquerade as a precise, interior timing estimate.
    for (auto &f : frames)
      f.timestamp -= 0.079 - (truth - 0.019);
    seed.td = 0.075;
    const auto edge = EpipolarTimeInit::solve(frames, imu, cam, 0, seed, 0.08, 0.004);
    CHECK(edge.pairs >= 40 && !edge.ok, "search-edge timing must not be certified");
    for (auto &s : imu)
      s.wm = bg;
    const auto quiet = EpipolarTimeInit::solve(frames, imu, cam, 0, seed, 0.08, 0.004);
    CHECK(!quiet.ok, "no angular excitation must not certify timing");
  }
  std::printf("%s time bootstrap\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
