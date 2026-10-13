/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
// Stereo geometry sanity checks, standalone (nothing in the estimator calls this). On synthetic geometry -- a
// D0008-like parallel pair (68.9 mm) translating sideways past points at 2-7 m -- checks that stereo observations are
// consistent with the stereo calibration and with depth from motion:
//   epipolar   a right observation lies on the epipolar curve of its left partner under the calibration; a vertical
//              offset reads as its size, a shift along the baseline (a disparity error) does not
//   depth      the cam0-only (motion parallax) inverse depth predicts the right observation: exact data reads 0 px,
//              a 3 px disparity error reads ~3 px at every depth, a track without parallax is not judged, and clean
//              noisy tracks stay within noise
// Raw pixels go through frozen copies of the calibrated camera models; the features' normalized coordinates carry a
// 2 px principal-point drift (online intrinsic calibration) that the checks must ignore.
#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "cam/CamBase.h"
#include "cam/CamRadtan.h"
#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "utils/quat_ops.h"

using ov_core::Feature;
using ov_core::FeatureInitializer;

namespace {

struct StereoCheckOptions {
    double depth_max_pred_sigma_px = 0.5;  // judge a track only if its motion parallax predicts the disparity this well
                                           // (per px of cam0 noise; ~ stereo baseline / temporal parallax baseline)
    int depth_min_obs = 3;                 // cam0 observations needed for the motion-parallax depth
};

/// Fixed cam0 -> cam1 geometry the stereo data is checked against: P_1 = R_lr * P_0 + t_lr, and frozen copies of the
/// configured camera models. Raw pixels are normalized through those copies, never through the estimator's cameras:
/// with online intrinsic calibration a principal-point drift of either camera shifts every disparity, so checking the
/// tracker's normalized coordinates would let the filter bend the reference it is checked against.
struct StereoReference {
    bool valid = false;
    size_t cam_left = 0, cam_right = 1;
    Eigen::Matrix3d R_lr = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_lr = Eigen::Vector3d::Zero();
    double f_right = 1.0;  // px per radian of the right camera (focal length)
    std::shared_ptr<ov_core::CamBase> model_left,
        model_right;  // frozen; null = use the feature's normalized coordinates

    double baseline() const { return t_lr.norm(); }

    /// Normalized coordinates of observation i of `feat` in camera `cam`, through the frozen model
    Eigen::Vector2d normalized(const ov_core::Feature& feat, size_t cam, size_t i) const {
        const std::shared_ptr<ov_core::CamBase> model =
            cam == cam_left ? model_left : (cam == cam_right ? model_right : nullptr);
        if (model) return model->undistort_d(feat.uvs.at(cam).at(i).head<2>().cast<double>());
        return feat.uvs_norm.at(cam).at(i).head<2>().cast<double>();
    }

    /// Angular distance (as right-camera px) of the right bearing from the epipolar plane of the left bearing.
    /// Bearings are normalized image coordinates (x/z, y/z); the angular form is valid for fisheye lenses too.
    double epipolar_distance_px(const Eigen::Vector2d& n_left, const Eigen::Vector2d& n_right) const {
        const Eigen::Vector3d b0 = R_lr * Eigen::Vector3d(n_left(0), n_left(1), 1.0).normalized();
        const Eigen::Vector3d normal = t_lr.cross(b0);
        const double nn = normal.norm();
        if (nn < 1e-12) return 0.0;
        const Eigen::Vector3d b1 = Eigen::Vector3d(n_right(0), n_right(1), 1.0).normalized();
        return f_right * std::asin(std::min(1.0, std::abs(normal.dot(b1)) / nn));
    }
};

using ClonePoses = std::unordered_map<size_t, std::unordered_map<double, ov_core::FeatureInitializer::ClonePose>>;

/// Mean signed disparity residual (px) of a feature's right observations against its cam0-only (temporal parallax)
/// inverse depth, predicted through the cam0 poses and the reference extrinsic. The inverse depth along the latest
/// paired cam0 bearing is a scalar fit with no depth cutoffs: its error is symmetric, so selecting features cannot
/// bias the mean (a depth-bounded triangulation such as FeatureInitializer's keeps the ones that came out near). False
/// if the feature cannot be judged: too few cam0 observations, or a predicted disparity sigma above
/// depth_max_pred_sigma_px.
inline bool stereo_depth_residual(const ov_core::Feature& feat, ClonePoses& clones_cam, const StereoCheckOptions& opt,
                                  const StereoReference& ref, double& resid) {
    const size_t L = ref.cam_left, R = ref.cam_right;
    if (!ref.valid || feat.timestamps.find(L) == feat.timestamps.end() ||
        feat.timestamps.find(R) == feat.timestamps.end() || clones_cam.find(L) == clones_cam.end())
        return false;
    const std::vector<double>&tl = feat.timestamps.at(L), &tr = feat.timestamps.at(R);
    auto& poses = clones_cam.at(L);
    if ((int)tl.size() < opt.depth_min_obs) return false;

    // anchor: the latest cam0 observation that has a right observation at the same instant
    int anchor = -1;
    for (size_t i = 0; i < tl.size(); i++) {
        if (poses.find(tl.at(i)) == poses.end() || (anchor >= 0 && tl.at(i) <= tl.at(anchor))) continue;
        for (double t : tr)
            if (t == tl.at(i)) anchor = (int)i;
    }
    if (anchor < 0) return false;
    const Eigen::Matrix3d R_GtoA = poses.at(tl.at(anchor)).Rot();
    const Eigen::Vector3d p_AinG = poses.at(tl.at(anchor)).pos();
    const Eigen::Vector2d n_a = ref.normalized(feat, L, anchor);
    const Eigen::Vector3d b_a(n_a(0), n_a(1), 1.0);

    // a point rho-scaled into camera k: h_k = R_AtoK * b_a + rho * t_k, t_k = R_GtoK * (p_AinG - p_KinG)
    auto relative = [&](double t, Eigen::Vector3d& u, Eigen::Vector3d& tk) {
        auto it = poses.find(t);
        if (it == poses.end()) return false;
        u = it->second.Rot() * R_GtoA.transpose() * b_a;
        tk = it->second.Rot() * (p_AinG - it->second.pos());
        return true;
    };

    // inverse depth from every other cam0 observation: the linear fit (rho * (t_x - n_x t_z) = -(u_x - n_x u_z)) seeds
    // Gauss-Newton on the reprojection error (the linear fit's regressors carry the measurement noise, which can
    // attenuate it towards rho = 0 at low parallax)
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> rel;  // (u, t_k) per observation
    std::vector<Eigen::Vector2d> obs;
    double saa = 0.0, sac = 0.0;
    for (size_t k = 0; k < tl.size(); k++) {
        Eigen::Vector3d u, tk;
        if ((int)k == anchor || !relative(tl.at(k), u, tk)) continue;
        const Eigen::Vector2d n = ref.normalized(feat, L, k);
        rel.emplace_back(u, tk);
        obs.push_back(n);
        for (int r = 0; r < 2; r++) {
            const double a = tk(r) - n(r) * tk(2), c = u(r) - n(r) * u(2);
            saa += a * a;
            sac += a * c;
        }
    }
    if (saa <= 0.0) return false;
    double rho = -sac / saa, jtj = 0.0;
    for (int it = 0; it < 5; it++) {
        double jtr = 0.0;
        jtj = 0.0;
        for (size_t k = 0; k < rel.size(); k++) {
            const Eigen::Vector3d h = rel.at(k).first + rho * rel.at(k).second;
            if (h(2) <= 1e-6) return false;
            const Eigen::Vector3d& t = rel.at(k).second;
            const Eigen::Vector2d J = (t.head<2>() * h(2) - h.head<2>() * t(2)) / (h(2) * h(2));
            jtj += J.squaredNorm();
            jtr += J.dot(obs.at(k) - h.head<2>() / h(2));
        }
        if (jtj <= 0.0) return false;
        rho += jtr / jtj;
    }
    // predicted disparity sigma: f * baseline * sigma_rho, sigma_rho = (1 px / f) / sqrt(J^T J)
    if (!std::isfinite(rho) || ref.baseline() / std::sqrt(jtj) > opt.depth_max_pred_sigma_px) return false;

    double sum = 0.0;
    int n = 0;
    for (size_t j = 0; j < tr.size(); j++) {
        Eigen::Vector3d u, tk;
        if (!relative(tr.at(j), u, tk)) continue;
        // predict through the cam0 pose and the REFERENCE extrinsic: the filter's own cam1 extrinsic can absorb a
        // uniform disparity bias as a small cam1 yaw (soft lock), which would hide it from this check. The point is
        // rho-scaled into cam1 at this instant; disparity varies only the stereo term (depth along the cam0 ray of this
        // instant): s = rho is the prediction, s = 0 the point at infinity, larger s nearer
        const Eigen::Vector3d ray = ref.R_lr * (u + rho * tk);
        auto predict = [&](double s) -> Eigen::Vector3d { return ray + s * ref.t_lr; };
        const Eigen::Vector3d h = predict(rho), h_near = predict(rho + 1e-4);
        if (h(2) <= 1e-6 || h_near(2) <= 1e-6) continue;
        const Eigen::Vector2d pred = h.head<2>() / h(2);
        Eigen::Vector2d dir = h_near.head<2>() / h_near(2) - pred;  // towards larger disparity (nearer)
        if (dir.norm() < 1e-12) continue;
        dir.normalize();
        // normalized-plane distance -> px: scale by f at the bearing's angle (equidistant fisheye: dθ = dn * cos²θ)
        const double px = ref.f_right / (1.0 + pred.squaredNorm());
        const Eigen::Vector2d meas = ref.normalized(feat, R, j);
        const double r = px * (meas - pred).dot(dir);
        if (std::isfinite(r)) sum += r, n++;
    }
    if (n == 0) return false;
    resid = sum / n;
    return true;
}

int checks = 0, failures = 0;
void check(bool ok, const char* why) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", why);
    }
}

StereoReference make_ref() {
    StereoReference ref;
    ref.valid = true;
    ref.R_lr = ov_core::exp_so3(Eigen::Vector3d(0.03, -0.017, -0.03));  // ~2.5 deg, like the D0008 calibration
    ref.t_lr = Eigen::Vector3d(-0.0689, 0.0016, -0.001);
    ref.f_right = 465.0;
    for (auto* model : {&ref.model_left, &ref.model_right}) {
        *model = std::make_shared<ov_core::CamRadtan>(1280, 800);
        Eigen::Matrix<double, 8, 1> calib;
        calib << 465.0, 465.0, 640.0, 400.0, 0.0, 0.0, 0.0, 0.0;
        (*model)->set_value(calib);
    }
    return ref;
}

// cam0 poses at n clone times moving along cam0 x at speed v (m per clone); clones_cam[0] (and a copy for cam 1)
ClonePoses make_poses(int n, double step) {
    ClonePoses clones;
    for (int k = 0; k < n; k++) {
        const double t = 0.1 * k;
        const Eigen::Matrix3d R_GtoC = ov_core::exp_so3(Eigen::Vector3d(0.0, 0.002 * k, 0.0));
        const Eigen::Vector3d p_CinG(step * k, 0.0, 0.0);
        clones[0][t] = FeatureInitializer::ClonePose(R_GtoC, p_CinG);
    }
    return clones;
}

// observe p_FinG from every clone in cam0 and cam1 (cam1 pixel shift `bias_px` along x, noise sigma px)
Feature observe(const Eigen::Vector3d& p_FinG, ClonePoses& clones, const StereoReference& ref, double bias_px,
                double sigma, std::mt19937& rng) {
    std::normal_distribution<double> noise(0.0, sigma);
    Feature f;
    f.featid = 1;
    for (auto& c : clones.at(0)) {
        const Eigen::Vector3d p0 = c.second.Rot() * (p_FinG - c.second.pos());
        const Eigen::Vector3d p1 = ref.R_lr * p0 + ref.t_lr;
        // raw pixels through the reference models (as the tracker reports them); uvs_norm deliberately left wrong by a
        // principal-point drift, as the estimator's online-calibrated cameras would produce: the checks must not use it
        Eigen::VectorXf n0(2), n1(2), u0(2), u1(2);
        n0 << (float)(p0(0) / p0(2)), (float)(p0(1) / p0(2));
        n1 << (float)(p1(0) / p1(2)), (float)(p1(1) / p1(2));
        u0 = ref.model_left->distort_f(n0) + Eigen::Vector2f((float)noise(rng), (float)noise(rng));
        u1 = ref.model_right->distort_f(n1) + Eigen::Vector2f((float)(bias_px + noise(rng)), (float)noise(rng));
        const Eigen::Vector2f drift(2.0f / (float)ref.f_right, 0.0f);
        f.timestamps[0].push_back(c.first), f.uvs[0].push_back(u0), f.uvs_norm[0].push_back(n0 + drift);
        f.timestamps[1].push_back(c.first), f.uvs[1].push_back(u1), f.uvs_norm[1].push_back(n1);
    }
    return f;
}
}  // namespace

int main() {
    const StereoReference ref = make_ref();
    StereoCheckOptions opt;
    std::mt19937 rng(7);

    {
        // epipolar distance: zero on the curve, the perpendicular offset off it, unchanged by a shift along it
        const Eigen::Vector3d p0(0.4, -0.3, 3.0), p1 = ref.R_lr * p0 + ref.t_lr;
        const Eigen::Vector2d n0 = p0.head<2>() / p0(2), n1 = p1.head<2>() / p1(2);
        check(ref.epipolar_distance_px(n0, n1) < 1e-6, "a true correspondence is on the epipolar curve");
        check(std::abs(ref.epipolar_distance_px(n0, n1 + Eigen::Vector2d(0, 4.0 / ref.f_right)) - 4.0) < 0.3,
              "a 4 px vertical offset is ~4 px off the (near-horizontal) epipolar curve");
        check(ref.epipolar_distance_px(n0, n1 + Eigen::Vector2d(6.0 / ref.f_right, 0)) < 0.4,
              "a 6 px shift along the baseline stays on the epipolar curve (a disparity error, not an epipolar one)");
    }

    ClonePoses poses = make_poses(11, 0.03);  // 30 cm temporal baseline over the window
    const Eigen::Vector3d pts[] = {{0.3, 0.2, 2.0}, {-0.5, 0.1, 4.0}, {1.0, -0.4, 7.0}};
    {
        // exact data: zero residual
        for (const auto& p : pts) {
            Feature f = observe(p, poses, ref, 0.0, 0.0, rng);
            double r = 1e9;
            check(stereo_depth_residual(f, poses, opt, ref, r), "a well-observed track is judged");
            check(std::abs(r) < 1e-3, "exact stereo agrees with the temporal depth");
        }
    }
    {
        // a +3 px right-image shift is a disparity error of the same size (sign depends on the rig side)
        for (const auto& p : pts) {
            Feature f = observe(p, poses, ref, 3.0, 0.0, rng);
            double r = 0;
            check(stereo_depth_residual(f, poses, opt, ref, r) && std::abs(std::abs(r) - 3.0) < 0.3,
                  "a 3 px shift reads as ~3 px of disparity error");
            std::printf("  z=%.1f m  residual for +3 px: %.2f px\n", p(2), r);
        }
    }
    {
        // no temporal parallax (hover): not judged
        ClonePoses still = make_poses(11, 0.0);
        Feature f = observe(pts[0], still, ref, 0.0, 0.0, rng);
        double r;
        check(!stereo_depth_residual(f, still, opt, ref, r), "a track without temporal parallax is not judged");
    }
    {
        // clean tracks at 1 px noise: every judged disparity is within a few px of the motion-predicted one, and the
        // right observations sit on the epipolar curve to within noise (two 1 px observations -> ~1.4 px perpendicular)
        std::uniform_real_distribution<double> ux(-2.0, 2.0), uz(2.0, 7.0);
        int judged = 0, pairs = 0, off_band = 0;
        double worst = 0.0;
        for (int i = 0; i < 300; i++) {
            const Eigen::Vector3d p(ux(rng), 0.5 * ux(rng), uz(rng));
            Feature f = observe(p, poses, ref, 0.0, 1.0, rng);
            double r;
            if (stereo_depth_residual(f, poses, opt, ref, r)) {
                judged++;
                worst = std::max(worst, std::abs(r));
            }
            for (size_t j = 0; j < f.timestamps.at(1).size(); j++, pairs++)
                off_band += ref.epipolar_distance_px(ref.normalized(f, 0, j), ref.normalized(f, 1, j)) > 3.0;
        }
        check(judged > 250 && worst < 4.0,
              "clean stereo depth agrees with motion depth to within noise on every judged track");
        check(off_band < 0.08 * pairs,
              "clean right observations lie within 3 px of the epipolar curve (noise tail only)");
        std::printf(
            "  clean: %d/300 judged, worst |disparity error| %.2f px; %d/%d pairs > 3 px off the epipolar curve\n",
            judged, worst, off_band, pairs);
    }
    std::printf("STEREO_CONSISTENCY %s checks=%d failures=%d\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
