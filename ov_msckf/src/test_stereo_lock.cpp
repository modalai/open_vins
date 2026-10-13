/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
// Soft stereo lock: T_cam1 = T_01 * T_cam0 with T_01 a 6-DoF calibration state (tight prior, no process noise).
// Checks the cam1 -> (cam0, T_01) Jacobian fold against finite differences of the composition (stacked and
// block-list forms, linearized at the current estimate rather than the FEJ), the state
// layout and prior, that every EKF update leaves cam1 exactly derived from cam0 and T_01 (value and FEJ),
// clone isolation, re-capture idempotence (retained-state soft reset) and the default-on resolution.
#include <unistd.h>

#include <cstdio>
#include <fstream>

#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterHelper.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_msckf;
using namespace ov_type;
namespace {
using Mat = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using M3 = Eigen::Matrix3d;
using V3 = Eigen::Vector3d;
int checks = 0, failures = 0;
void check(bool ok, const char* why) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", why);
    }
}
double rot_err(const M3& a, const M3& b) { return log_so3(a * b.transpose()).norm(); }
Eigen::Matrix<double, 7, 1> pose_value(const M3& R, const V3& p) {
    Eigen::Matrix<double, 7, 1> v;
    v.head<4>() = rot_2_quat(R);
    v.tail<3>() = p;
    return v;
}
void set_pose(const std::shared_ptr<PoseJPL>& pose, const M3& R, const V3& p) {
    pose->set_value(pose_value(R, p));
    pose->set_fej(pose_value(R, p));
}
// Calibrated-looking rig: cam0 a generic cam-IMU pose, T_01 a 68 mm baseline with a ~1 deg relative rotation
const M3 R0 = exp_so3(V3(.31, -1.2, .45));
const V3 p0(.012, -.034, .021);
const M3 R01 = exp_so3(V3(.004, -.015, .009));
const V3 t01(-.0681, .0005, -.0008);
const double sig_rot = .0044, sig_pos = .002;

std::shared_ptr<State> make_rig() {
    StateOptions o;
    o.num_cameras = 2;
    o.do_fej = true;
    o.do_calib_camera_pose = true;
    o.calib_cam_stereo_lock_soft = true;
    o.calib_cam_stereo_lock_sigma_rot = sig_rot;
    o.calib_cam_stereo_lock_sigma_pos = sig_pos;
    auto s = std::make_shared<State>(o);
    set_pose(s->_calib_IMUtoCAM.at(0), R0, p0);
    set_pose(s->_calib_IMUtoCAM.at(1), R01 * R0, R01 * p0 + t01);
    s->capture_stereo_lock();
    StateHelper::set_initial_covariance(s, .01 * Mat::Identity(15, 15), {s->_imu});
    return s;
}
// cam1 must equal T_01 * cam0 for the value, and T_01 * FEJ(cam0) for the FEJ
bool cam1_derived(const std::shared_ptr<State>& s, double tol) {
    const auto &c0 = s->_calib_IMUtoCAM.at(0), &c1 = s->_calib_IMUtoCAM.at(1);
    const M3 Rr = s->stereo_R01();
    const V3 tr = s->stereo_t01();
    return rot_err(c1->Rot(), Rr * c0->Rot()) < tol && (c1->pos() - (Rr * c0->pos() + tr)).norm() < tol &&
           rot_err(c1->Rot_fej(), Rr * c0->Rot_fej()) < tol && (c1->pos_fej() - (Rr * c0->pos_fej() + tr)).norm() < tol;
}
Mat block(const std::shared_ptr<State>& s, const std::shared_ptr<Type>& v) {
    return StateHelper::get_full_covariance(s).block(v->id(), v->id(), v->size(), v->size());
}

void layout_and_prior() {
    auto s = make_rig();
    const auto &c0 = s->_calib_IMUtoCAM.at(0), &c1 = s->_calib_IMUtoCAM.at(1);
    check(s->_stereo_rel != nullptr && s->_stereo_rel->id() >= 0, "soft lock adds T_01 as a state variable");
    check(c0->id() >= 0 && c1->id() < 0, "cam0 extrinsic is estimated, cam1 is derived (no state columns)");
    check(!s->_options.calib_camera_pose_estimated(1) && s->_options.calib_camera_pose_estimated(0),
          "calib_camera_pose_estimated excludes only cam1");
    check(s->stereo_locked_camera(1) && !s->stereo_locked_camera(0), "lock reports cam1 as derived after capture");
    check(s->max_covariance_size() == 15 + 6 + 6, "state = IMU + cam0 extrinsic + T_01");
    Mat P = Mat::Zero(6, 6);
    P.diagonal() << sig_rot * sig_rot, sig_rot * sig_rot, sig_rot * sig_rot, sig_pos * sig_pos, sig_pos * sig_pos,
        sig_pos * sig_pos;
    check((block(s, s->_stereo_rel) - P).norm() < 1e-18, "T_01 prior is diag(sigma_rot^2 I, sigma_pos^2 I)");
    check(StateHelper::get_full_covariance(s).block(s->_stereo_rel->id(), c0->id(), 6, 6).norm() == 0.,
          "T_01 starts uncorrelated with cam0");
    check(rot_err(s->stereo_R01(), R01) < 1e-12 && (s->stereo_t01() - t01).norm() < 1e-12,
          "capture recovers T_01 = T_cam1 * T_cam0^-1 from the seeded extrinsics");
    check(cam1_derived(s, 1e-12), "cam1 equals T_01 * cam0 after capture");
}

// Numeric map from (dth0, dp0, dth01, dt01) to (dth1, dp1) through the composition, JPL error convention
// R = exp(-[dth]x) R_hat, p = p_hat + dp (matches PoseJPL::update)
Mat numeric_maps(const M3& Ra, const V3& pa, const M3& Rr, const V3& tr) {
    const double h = 1e-6;
    const M3 R1 = Rr * Ra;
    const V3 p1 = Rr * pa + tr;
    Mat J(6, 12);
    for (int k = 0; k < 12; ++k) {
        Vector out[2];
        for (int sgn = 0; sgn < 2; ++sgn) {
            Vector d = Vector::Zero(12);
            d(k) = sgn ? -h : h;
            const M3 Ra_p = exp_so3(-V3(d.segment<3>(0))) * Ra, Rr_p = exp_so3(-V3(d.segment<3>(6))) * Rr;
            const V3 pa_p = pa + d.segment<3>(3), tr_p = tr + d.segment<3>(9);
            const M3 R1_p = Rr_p * Ra_p;
            const V3 p1_p = Rr_p * pa_p + tr_p;
            out[sgn] = Vector(6);
            out[sgn].head<3>() = -log_so3(R1_p * R1.transpose());
            out[sgn].tail<3>() = p1_p - p1;
        }
        J.col(k) = (out[0] - out[1]) / (2 * h);
    }
    return J;
}

void jacobian_fold() {
    auto s = make_rig();
    const auto c0 = s->_calib_IMUtoCAM.at(0), c1 = s->_calib_IMUtoCAM.at(1);
    const Mat J = numeric_maps(c0->Rot(), c0->pos(), s->stereo_R01(), s->stereo_t01());

    // cam1 alone: folds onto [cam0 | T_01] as [M0 | Mrel]
    Mat H = Mat::Identity(6, 6);
    std::vector<std::shared_ptr<Type>> order = {c1};
    UpdaterHelper::fold_stereo_lock(s, H, order);
    check(order.size() == 2 && order[0] == c0 && order[1] == s->_stereo_rel, "fold replaces cam1 with cam0 then T_01");
    check(H.rows() == 6 && H.cols() == 12 && (H - J).norm() < 1e-7, "folded cam1 Jacobian matches finite differences");

    // cam0 already present: cam1's contribution accumulates into cam0's columns, other columns untouched
    Mat A(6, 6), B(6, 6), C(6, 15);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            A(i, j) = std::sin(.3 + i + .7 * j);
            B(i, j) = std::cos(.2 * i - .9 * j);
        }
        for (int j = 0; j < 15; ++j) C(i, j) = std::sin(.11 * i * j + .5);
    }
    Mat Hx(6, 27);
    Hx << A, B, C;
    order = {c0, c1, s->_imu};
    UpdaterHelper::fold_stereo_lock(s, Hx, order);
    check(order.size() == 3 && order[0] == c0 && order[1] == s->_imu && order[2] == s->_stereo_rel,
          "stacked fold keeps cam0 and the IMU in place and appends T_01");
    check(Hx.cols() == 27 && (Hx.leftCols(6) - (A + B * J.leftCols(6))).norm() < 1e-7 && Hx.middleCols(6, 15) == C &&
              (Hx.rightCols(6) - B * J.rightCols(6)).norm() < 1e-7,
          "stacked fold accumulates H_cam1 * M0 into cam0 and H_cam1 * Mrel into T_01");

    // block-list form (SLAM anchor change)
    std::vector<Mat> Hb = {B};
    std::vector<std::shared_ptr<Type>> ob = {c1};
    UpdaterHelper::fold_stereo_lock(s, Hb, ob);
    check(ob.size() == 2 && ob[0] == c0 && ob[1] == s->_stereo_rel && (Hb[0] - B * J.leftCols(6)).norm() < 1e-7 &&
              (Hb[1] - B * J.rightCols(6)).norm() < 1e-7,
          "block-list fold matches the stacked maps");

    // block-list with cam0 already present: cam1's block becomes a second cam0 entry, which the anchor-change
    // consumer (UpdaterSLAM) accumulates by type, so the per-type sums must equal the stacked fold
    std::vector<Mat> Hd = {A, B};
    std::vector<std::shared_ptr<Type>> od = {c0, c1};
    UpdaterHelper::fold_stereo_lock(s, Hd, od);
    Mat sum0 = Mat::Zero(6, 6), sum_rel = Mat::Zero(6, 6);
    int n_rel = 0, n_other = 0;
    for (size_t k = 0; k < od.size(); ++k) {
        if (od[k] == c0)
            sum0 += Hd[k];
        else if (od[k] == s->_stereo_rel)
            sum_rel += Hd[k], ++n_rel;
        else
            ++n_other;
    }
    check(od.size() == Hd.size() && n_rel == 1 && n_other == 0,
          "block-list fold with cam0 present drops cam1 and adds T_01 once");
    check((sum0 - (A + B * J.leftCols(6))).norm() < 1e-7 && (sum_rel - B * J.rightCols(6)).norm() < 1e-7,
          "block-list fold with cam0 present sums to the stacked fold");

    // no lock: fold is a no-op
    StateOptions o;
    o.num_cameras = 2;
    o.do_calib_camera_pose = true;
    auto plain = std::make_shared<State>(o);
    Mat Hp = Mat::Identity(6, 6);
    std::vector<std::shared_ptr<Type>> op = {plain->_calib_IMUtoCAM.at(1)};
    UpdaterHelper::fold_stereo_lock(plain, Hp, op);
    check(op.size() == 1 && op[0] == plain->_calib_IMUtoCAM.at(1) && Hp == Mat::Identity(6, 6),
          "fold is a no-op without the lock");
}

// Calibration Jacobians are evaluated at the current extrinsic estimate, not its FEJ (UpdaterHelper keeps
// calibration->Rot_fej() disabled), so the fold maps must linearize there too even when FEJ and value differ
void fold_linearization_point() {
    auto s = make_rig();
    const auto c0 = s->_calib_IMUtoCAM.at(0), c1 = s->_calib_IMUtoCAM.at(1);
    c0->set_fej(pose_value(exp_so3(V3(.02, -.01, .03)) * c0->Rot(), c0->pos() + V3(.03, -.02, .015)));
    s->_stereo_rel->set_fej(
        pose_value(exp_so3(V3(-.01, .02, .005)) * s->stereo_R01(), s->stereo_t01() + V3(.004, .002, -.003)));
    s->refresh_stereo_lock();
    check(cam1_derived(s, 1e-12), "cam1 FEJ follows cam0's FEJ through T_01 when it differs from the value");
    const Mat J_val = numeric_maps(c0->Rot(), c0->pos(), s->stereo_R01(), s->stereo_t01());
    const Mat J_fej = numeric_maps(c0->Rot_fej(), c0->pos_fej(), s->stereo_R01(), s->stereo_t01());
    check((J_val - J_fej).norm() > 1e-3, "FEJ offset changes the composition Jacobian (check is sensitive)");
    Mat H = Mat::Identity(6, 6);
    std::vector<std::shared_ptr<Type>> order = {c1};
    UpdaterHelper::fold_stereo_lock(s, H, order);
    check((H - J_val).norm() < 1e-7, "fold linearizes at the current cam0 estimate, like the calibration Jacobians");
}

// one EKF update on [cam0, T_01]; returns false if it was rejected
bool update(const std::shared_ptr<State>& s, double scale, double phase) {
    const int m = 5;
    Mat H(m, 12);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < 12; ++j) H(i, j) = std::sin(phase + .9 * i + .37 * j) + (i == j ? 1. : 0.);
    Vector r(m);
    for (int i = 0; i < m; ++i) r(i) = scale * std::cos(phase + .5 * i);
    return StateHelper::EKFUpdate(s, {s->_calib_IMUtoCAM.at(0), s->_stereo_rel}, H, r, 1e-4 * Mat::Identity(m, m));
}

void update_invariant() {
    auto s = make_rig();
    const auto c0 = s->_calib_IMUtoCAM.at(0);
    const Mat P_rel = block(s, s->_stereo_rel), P_c0 = block(s, c0);
    const M3 Rc0 = c0->Rot(), Rr = s->stereo_R01();
    check(update(s, 2e-3, 0.), "update on cam0 and T_01 is accepted");
    check(rot_err(c0->Rot(), Rc0) > 1e-9 && rot_err(s->stereo_R01(), Rr) > 1e-9, "update moves both cam0 and T_01");
    check(cam1_derived(s, 1e-12), "after an update cam1 is re-derived from cam0 and T_01 (value and FEJ)");
    bool all_accepted = true;
    for (int k = 1; k < 20; ++k) all_accepted = update(s, 2e-3, .7 * k) && all_accepted;
    check(all_accepted, "repeated updates are accepted");
    check(cam1_derived(s, 1e-12), "cam1 stays derived across repeated updates");
    const Eigen::SelfAdjointEigenSolver<Mat> shrink_rel(P_rel - block(s, s->_stereo_rel)),
        shrink_c0(P_c0 - block(s, c0));
    check(shrink_rel.eigenvalues().minCoeff() > -1e-15 && shrink_c0.eigenvalues().minCoeff() > -1e-15,
          "EKF updates only shrink the T_01 and cam0 covariances");
}

void clone_isolation() {
    auto s = make_rig();
    update(s, 2e-3, .3);
    auto c = StateHelper::clone_state(s);
    check(c->_stereo_rel && c->_stereo_rel != s->_stereo_rel, "clone owns its own T_01 variable");
    check(c->_stereo_rel->id() == s->_stereo_rel->id() && c->_stereo_rel->value() == s->_stereo_rel->value() &&
              c->stereo_locked_camera(1) && c->_stereo_lock_R01 == s->_stereo_lock_R01 &&
              c->_stereo_lock_t01 == s->_stereo_lock_t01,
          "clone copies T_01 and the lock fields");
    const Mat rel = s->_stereo_rel->value(), cam1 = s->_calib_IMUtoCAM.at(1)->value();
    check(update(c, 3e-3, 1.1), "update on the clone is accepted");
    check(s->_stereo_rel->value() == rel && s->_calib_IMUtoCAM.at(1)->value() == cam1,
          "updating the clone leaves the original untouched");
    check(cam1_derived(c, 1e-12) && cam1_derived(s, 1e-12), "both clone and original keep cam1 derived");
}

// A retained-state soft reset re-captures the lock from the current values: T_01 and its covariance must not move
void recapture_idempotent() {
    auto s = make_rig();
    bool all_accepted = true;
    for (int k = 0; k < 5; ++k) all_accepted = update(s, 2e-3, .4 * k) && all_accepted;
    check(all_accepted, "updates before re-capture are accepted");
    const Mat rel = s->_stereo_rel->value(), P = StateHelper::get_full_covariance(s);
    s->capture_stereo_lock();
    check((s->_stereo_rel->value() - rel).norm() < 1e-12, "re-capture keeps the refined T_01");
    check(StateHelper::get_full_covariance(s) == P, "re-capture does not touch the covariance");
    check(cam1_derived(s, 1e-12), "cam1 still derived after re-capture");
}

void default_resolution() {
    for (int cams : {1, 2, 3})
        for (bool stereo : {false, true})
            for (bool calib : {false, true})
                for (int expl : {-1, 0, 1}) {
                    VioManagerOptions o;
                    o.state_options.num_cameras = cams;
                    o.state_options.do_calib_camera_pose = calib;
                    o.use_stereo = stereo;
                    o.state_options.calib_cam_stereo_lock_soft_explicit = expl >= 0;
                    o.state_options.calib_cam_stereo_lock_soft = expl == 1;
                    o.resolve_stereo_lock();
                    const bool expected = expl >= 0 ? expl == 1 : stereo && cams == 2 && calib;
                    check(o.state_options.calib_cam_stereo_lock_soft == expected,
                          "soft lock defaults on only for 2-camera stereo with extrinsic calibration");
                }
    // the owner overriding use_stereo after parsing is honored by resolving afterwards
    VioManagerOptions o;
    o.state_options.num_cameras = 2;
    o.state_options.do_calib_camera_pose = true;
    o.use_stereo = false;
    o.resolve_stereo_lock();
    check(!o.state_options.calib_cam_stereo_lock_soft, "mono association leaves the lock off");
    o.use_stereo = true;
    o.resolve_stereo_lock();
    check(o.state_options.calib_cam_stereo_lock_soft, "re-resolving after a use_stereo override turns the lock on");

    // YAML presence detection: absent key -> default resolution, present key -> explicit value wins
    char path[] = "/tmp/ov-stereo-lock-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        check(false, "temp yaml");
        return;
    }
    close(fd);
    for (int key : {-1, 0, 1}) {
        {
            std::ofstream yaml(path);
            yaml << "%YAML:1.0\n---\nfixture: 1\n";
            if (key >= 0) yaml << "calib_cam_stereo_lock_soft: " << (key ? "true" : "false") << "\n";
        }
        auto parser = std::make_shared<YamlParser>(path);
        StateOptions so;
        so.parse_stereo_lock(parser);
        check(so.calib_cam_stereo_lock_soft_explicit == (key >= 0) &&
                  (key < 0 || so.calib_cam_stereo_lock_soft == (key == 1)),
              "calib_cam_stereo_lock_soft is explicit only when present in the YAML");
        check(parser->successful(), "stereo lock keys are optional");
    }
    unlink(path);
}
}  // namespace

int main() {
    Printer::setPrintLevel("SILENT");
    layout_and_prior();
    jacobian_fold();
    fold_linearization_point();
    update_invariant();
    clone_isolation();
    recapture_idempotent();
    default_resolution();
    std::printf("STEREO_LOCK %s checks=%d failures=%d\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
