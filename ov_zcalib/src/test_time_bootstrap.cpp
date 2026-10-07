/* Independent motion/clock oracles for the geometric time bootstrap. */
#include "init/EpipolarTimeInit.h"
#include "init/TimeOffsetInit.h"
#include "ceres_free/DistortDouble.h"
#include "core/CalibSessionRunner.h"
#include "sim/SynthWorld.h"
#include "utils/quat_ops.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
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

static void handeye_bias_status_regression() {
  // Independent analytic rotations/gyro distinguish a fitted small bias from
  // a physically excessive fit that the unchanged sanity guard rejects.
  const Eigen::Matrix3d R = ov_core::exp_so3(Eigen::Vector3d(0.12, -0.18, 0.08));
  const Eigen::Vector3d seed(0.003, -0.002, 0.001);
  std::vector<HandEyePair> pairs;
  for (int i = 0; i < 150; ++i) {
    HandEyePair p;
    p.t0 = 0.2 + 0.05 * i;
    p.t1 = p.t0 + 0.04;
    p.theta_C = R * ov_core::log_so3(pose(p.t1) * pose(p.t0).transpose());
    pairs.push_back(p);
  }
  for (bool excessive : {false, true}) {
    const Eigen::Vector3d actual = excessive ? Eigen::Vector3d(0.12, -0.10, 0.09) : Eigen::Vector3d(0.010, -0.012, 0.009);
    std::vector<RawImu> imu;
    for (int i = 0; i <= 8000; ++i) {
      RawImu s;
      s.timestamp = 0.001 * i;
      s.wm = gyro(s.timestamp) + actual;
      imu.push_back(s);
    }
    HandEyeConfig cfg;
    cfg.td_fine_range = 0.0; // Known clock; isolate the bias decision.
    HandEyeResult fit, fixed;
    CHECK(HandEyeWahba::solve(imu, pairs, 0.0, seed, cfg, fit), "bias-status fixture must solve");
    cfg.estimate_bg = false;
    CHECK(HandEyeWahba::solve(imu, pairs, 0.0, seed, cfg, fixed), "fixed-bias fixture must solve");
    CHECK(fixed.bg_status == HandEyeBiasStatus::FIXED_SEED && (fixed.bg - seed).norm() == 0.0,
          "fixed input bias must not be reported as estimated");
    CHECK(fixed.bg_initial_delta_norm == 0.0 && fixed.bg_sanity_limit == 0.0,
          "fixed bias must not invent an attempted fit");
    CHECK(fit.bg_sanity_limit == cfg.max_bg_sane, "bias diagnostic must preserve the actual sanity limit");
    if (excessive) {
      CHECK(fit.bg_status == HandEyeBiasStatus::SANITY_FALLBACK && fit.bg_initial_delta_norm > cfg.max_bg_sane,
            "rejected bias update must be reported as a sanity fallback");
      CHECK((fit.bg - seed).norm() == 0.0 && (fit.q_ItoC - fixed.q_ItoC).norm() == 0.0 && fit.td == fixed.td &&
                fit.rmse_rad == fixed.rmse_rad && fit.pairs_used == fixed.pairs_used,
            "fallback diagnostics must preserve the fixed-seed solve exactly");
    } else {
      CHECK(fit.bg_status == HandEyeBiasStatus::ESTIMATED && fit.bg_initial_delta_norm < cfg.max_bg_sane,
            "accepted bias fit must be reported as estimated");
      CHECK((fit.bg - actual).norm() < 1e-5, "accepted bias fixture must recover its independent truth");
    }
    std::printf("hand-eye bias %s: status=%s initial delta=%.6g limit=%.6g\n", excessive ? "excessive" : "small",
                handeye_bias_status_name(fit.bg_status), fit.bg_initial_delta_norm, fit.bg_sanity_limit);
    HandEyeResult failed;
    CHECK(!HandEyeWahba::solve(imu, {}, 0.0, seed, cfg, failed) && failed.bg_status == HandEyeBiasStatus::NOT_SOLVED,
          "failed hand-eye must not claim a solved bias");
  }
}

static void seeded_intrinsics_bootstrap(bool still_baseline) {
  // The sensor measurements are generated by INVERTING the physical intrinsic
  // model. Bootstrap must earn extrinsics/time while respecting that known seed,
  // and must return bias in the raw sensor frame used by downstream integration.
  synth::Truth truth = synth::make_truth();
  truth.imu.dw << 1.16, 0.04, 0.88, -0.025, 0.015, 1.10;
  truth.imu.da << 0.94, -0.02, 1.08, 0.03, -0.015, 1.04;
  truth.imu.q_AtoI = ov_core::rot_2_quat(ov_core::exp_so3(Eigen::Vector3d(0.07, -0.09, 0.05)));
  truth.imu.Tg << 0.0010, -0.0005, 0.0015,
                 -0.0008, 0.0012, -0.0010,
                 0.0004, 0.0007, 0.0009;
  truth.bg << 0.025, -0.020, 0.015;
  truth.ba.setZero(); // Isolate gyro bias/gauge; bootstrap has no accel-bias estimate.
  const Eigen::Vector3d mapped_bg = ImuIntrinsicModel::ut(truth.imu.dw) * truth.bg;
  CHECK((mapped_bg - truth.bg).norm() > 0.004, "bias fixture must distinguish raw and corrected frames");

  synth::Trajectory trajectory;
  trajectory.excite_t0 = still_baseline ? 3.0 : -10.0;
  trajectory.excite_t1 = 100.0;
  synth::StreamOptions options;
  options.dur = 34.0;
  options.fps = 60.0;
  options.pix_noise = options.w_noise = options.a_noise = 0.0;
  std::vector<RawImu> imu;
  std::vector<FrameObs> frames;
  std::vector<synth::Truth> cameras(2, truth);
  cameras[0].td = -0.007;
  cameras[1].td = 0.009;
  cameras[1].q_ItoC = ov_core::rot_2_quat(ov_core::exp_so3(Eigen::Vector3d(-0.16, 0.20, -0.24)));
  for (auto &camera : cameras)
    camera.p_IinC.setZero();
  SessionSeed seed;
  seed.calib.imu = truth.imu;
  seed.calib.cams.resize(cameras.size());
  for (size_t c = 0; c < cameras.size(); ++c) {
    std::vector<RawImu> camera_imu;
    std::vector<FrameObs> camera_frames;
    synth::make_streams(cameras[c], trajectory, options, 170 + c, camera_imu, camera_frames);
    if (c == 0)
      imu = std::move(camera_imu);
    // A rig rotating about its IMU center isolates the intrinsic/bias contract
    // from translation-contaminated visual relative-rotation approximations.
    const auto cloud = synth::make_cloud(180, 170 + c);
    const auto R_ItoC = ov_core::quat_2_Rot(cameras[c].q_ItoC);
    for (auto &frame : camera_frames) {
      // Exactly still IMU prefix below has no camera observations. This avoids
      // treating SynthWorld's deliberate quiet jitter as a nonzero bias truth.
      if (still_baseline && frame.timestamp < 3.0)
        continue;
      frame.cam = (int)c;
      frame.pts.clear();
      for (size_t j = 0; j < cloud.size(); ++j) {
        const Eigen::Vector3d pc = R_ItoC * trajectory.R_of(frame.timestamp + cameras[c].td) * cloud[j];
        if (pc.z() <= 0.4)
          continue;
        const Eigen::Vector2d uv(cameras[c].cam(0) * pc.x() / pc.z() + cameras[c].cam(2),
                                 cameras[c].cam(1) * pc.y() / pc.z() + cameras[c].cam(3));
        if (uv.x() < 10 || uv.x() > truth.img_w - 10 || uv.y() < 10 || uv.y() > truth.img_h - 10)
          continue;
        FrameObsPoint point;
        point.id = j + 10000 * c; // no cross-camera feature correspondence
        point.u = uv.x();
        point.v = uv.y();
        frame.pts.push_back(point);
      }
      frames.push_back(std::move(frame));
    }
    auto &camera_seed = seed.calib.cams[c];
    camera_seed.cam = cameras[c].cam;
    camera_seed.img_w = cameras[c].img_w;
    camera_seed.img_h = cameras[c].img_h;
    camera_seed.fps = options.fps;
    camera_seed.q_ItoC << 0, 0, 0, 1; // both extrinsics and td are blind
    camera_seed.p_IinC.setZero();
    camera_seed.td = 0.0;
  }
  const Eigen::Matrix3d Dw_inverse = ImuIntrinsicModel::ut(truth.imu.dw).inverse();
  const Eigen::Matrix3d accel_inverse = ImuIntrinsicModel::ut(truth.imu.da).inverse() *
      ov_core::quat_2_Rot(truth.imu.q_AtoI).transpose();
  for (auto &sample : imu) {
    const bool stationary = still_baseline && sample.timestamp < 3.0;
    const Eigen::Vector3d acceleration = stationary ? truth.g_W : trajectory.R_of(sample.timestamp) * truth.g_W;
    const Eigen::Vector3d omega = stationary ? Eigen::Vector3d::Zero().eval() : trajectory.omega_I(sample.timestamp);
    sample.wm = Dw_inverse * omega + truth.bg + truth.imu.Tg * acceleration;
    sample.am = accel_inverse * acceleration;
  }
  std::sort(frames.begin(), frames.end(), [](const FrameObs &a, const FrameObs &b) {
    return a.timestamp == b.timestamp ? a.cam < b.cam : a.timestamp < b.timestamp;
  });
  SessionConfig cfg;
  cfg.settle_timeout_s = still_baseline ? 5.0 : 0.1;
  cfg.retro_harvest = false; // Stop at COLLECT; no window or joint optimization is needed.
  cfg.verbose = true;
  cfg.out_yaml.clear();
  CalibSessionRunner runner(cfg, seed);
  size_t ii = 0;
  for (const auto &frame : frames) {
    while (ii < imu.size() && imu[ii].timestamp <= frame.timestamp + 0.06)
      runner.feed_imu(imu[ii++]);
    runner.feed_frame(frame);
    if (runner.state() == RunnerState::COLLECT || runner.state() == RunnerState::ABORT)
      break;
  }
  std::printf("seeded intrinsic bootstrap %s: state=%d\n", still_baseline ? "still" : "moving", (int)runner.state());
  CHECK(runner.state() == RunnerState::COLLECT, "seeded two-camera bootstrap must reach collection");
  const auto &report = runner.report();
  CHECK(report.handeye.size() == cameras.size(), "bootstrap must return both camera estimates");
  for (size_t c = 0; c < report.handeye.size() && c < cameras.size(); ++c) {
    const auto &result = report.handeye[c];
    const double rotation_error = ov_core::log_so3(ov_core::quat_2_Rot(result.q_ItoC) *
        ov_core::quat_2_Rot(cameras[c].q_ItoC).transpose()).norm();
    const double bias_error = (result.bg - truth.bg).norm();
    std::printf("  cam%zu: dR=%.4g rad, dtd=%.4g ms, raw-bg error=%.4g, corrected-bg distance=%.4g\n", c,
                rotation_error, 1e3 * (result.td - cameras[c].td), bias_error, (result.bg - mapped_bg).norm());
    CHECK(result.ok && rotation_error < 0.002, "known intrinsics must not leak into blind camera rotation");
    CHECK(std::abs(result.td - cameras[c].td) < 0.0006, "each camera must recover its own clock offset");
    CHECK(bias_error < (still_baseline ? 1e-9 : 0.001), "bootstrap must recover raw bias, compensating seeded Tg");
    CHECK((result.bg - mapped_bg).norm() > 0.003, "corrected-frame bias must not escape into raw-frame consumers");
    if (c > 0)
      CHECK((result.bg - report.handeye[0].bg).norm() < 1e-12, "both cameras must share one raw IMU bias");
  }
  CHECK(report.t_solve_s == 0.0 && report.evidence.empty(), "bootstrap regression must not run joint refinement");
}

int main(int argc, char **argv) {
  if (argc > 1 && std::strcmp(argv[1], "--handeye-bias-status") == 0) {
    handeye_bias_status_regression();
    std::printf("%s hand-eye bias status\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
  }
  if (argc > 1 && std::strcmp(argv[1], "--runner-intrinsics") == 0) {
    seeded_intrinsics_bootstrap(true);
    seeded_intrinsics_bootstrap(false);
    std::printf("%s seeded runner bootstrap\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
  }
  handeye_bias_status_regression();
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
  seeded_intrinsics_bootstrap(true);
  seeded_intrinsics_bootstrap(false);
  std::printf("%s time bootstrap\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
