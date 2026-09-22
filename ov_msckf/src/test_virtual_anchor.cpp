/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "cam/CamRadtan.h"
#include "feat/Feature.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "utils/quat_ops.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

namespace {
using namespace ov_msckf;
using namespace ov_core;
using namespace ov_type;
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
using Mat = Eigen::MatrixXd;
using Rep = LandmarkRepresentation::Representation;
using HF = UpdaterHelper::UpdaterHelperFeature;
int failures = 0;
void check(bool ok, const char *message) {
  if (!ok) {
    ++failures;
    std::printf("FAIL: %s\n", message);
  }
}
const std::vector<Rep> reps{
    LandmarkRepresentation::ANCHORED_3D,
    LandmarkRepresentation::ANCHORED_FULL_INVERSE_DEPTH,
    LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH,
    LandmarkRepresentation::ANCHORED_INVERSE_DEPTH_SINGLE};
int dimension(Rep r) {
  return r == LandmarkRepresentation::ANCHORED_INVERSE_DEPTH_SINGLE ? 1 : 3;
}
V3 world(const std::shared_ptr<State> &s, int camera, double time,
         const V3 &point, bool fej = false) {
  const auto &p = s->_clones_IMU.at(time);
  const auto &c = s->_calib_IMUtoCAM.at(camera);
  const M3 R = fej ? p->Rot_fej() : p->Rot();
  const V3 pos = fej ? p->pos_fej() : p->pos();
  return R.transpose() * c->Rot().transpose() * (point - c->pos()) + pos;
}
V3 local(const std::shared_ptr<State> &s, int camera, double time,
         const V3 &point, bool fej = false) {
  const auto &p = s->_clones_IMU.at(time);
  const auto &c = s->_calib_IMUtoCAM.at(camera);
  const M3 R = fej ? p->Rot_fej() : p->Rot();
  const V3 pos = fej ? p->pos_fej() : p->pos();
  return c->Rot() * R * (point - pos) + c->pos();
}
std::vector<std::shared_ptr<Type>>
state_order(const std::shared_ptr<State> &s) {
  std::vector<std::shared_ptr<Type>> out{s->_imu};
  for (int c = 0; c < s->_options.num_cameras; ++c) {
    for (auto v :
         {std::static_pointer_cast<Type>(s->cam_imu_dt_var(c)),
          std::static_pointer_cast<Type>(s->_calib_IMUtoCAM.at(c)),
          std::static_pointer_cast<Type>(s->_cam_intrinsics.at(c)),
          std::static_pointer_cast<Type>(s->_calib_camera_readout.at(c))})
      if (v->id() >= 0)
        out.push_back(v);
  }
  for (auto &p : s->_clones_IMU)
    out.push_back(p.second);
  for (auto &p : s->_features_SLAM)
    out.push_back(p.second);
  std::sort(out.begin(), out.end(),
            [](const std::shared_ptr<Type> &a, const std::shared_ptr<Type> &b) {
              return a->id() < b->id();
            });
  return out;
}
struct Fixture {
  std::shared_ptr<State> s;
  V3 point = V3(1.35, .25, 4.8);
  std::vector<double> times;
  int mode;
  Fixture(int timing, bool fej = false, bool online = false) : mode(timing) {
    StateOptions o;
    o.num_cameras = 2;
    o.max_aruco_features = 0;
    o.do_fej = fej;
    o.imu_model = StateOptions::RPNG;
    o.do_calib_camera_pose = online;
    o.do_calib_camera_timeoffset = online;
    o.do_calib_camera_readout = online;
    s = std::make_shared<State>(o);
    Eigen::Matrix<double, 8, 1> intr;
    intr << 400, 405, 320, 240, 0, 0, 0, 0;
    for (int c = 0; c < 2; ++c) {
      auto camera = std::make_shared<CamRadtan>(640, 480);
      camera->set_value(intr);
      s->_cam_intrinsics_cameras[c] = camera;
      s->_cam_intrinsics[c]->set_value(intr);
      s->_cam_intrinsics[c]->set_fej(intr);
      Eigen::Matrix<double, 7, 1> extr;
      extr << rot_2_quat(
          exp_so3(c ? V3(.035, -.018, .012) : V3(-.01, .015, .005))),
          .08 * c, .015 * c, 0;
      s->_calib_IMUtoCAM[c]->set_value(extr);
      s->_calib_IMUtoCAM[c]->set_fej(extr);
      Eigen::VectorXd td(1);
      td << ((timing && c == 1) ? -.006 : 0);
      s->cam_imu_dt_var(c)->set_value(td);
      s->cam_imu_dt_var(c)->set_fej(td);
      Eigen::VectorXd tr(1);
      tr << ((timing == 2 && c == 1) ? .012 : 0);
      s->_calib_camera_readout[c]->set_value(tr);
      s->_calib_camera_readout[c]->set_fej(tr);
    }
    for (int k = 0; k < 6; ++k) {
      const double t = 10. + .1 * k;
      times.push_back(t);
      s->_timestamp = t;
      Eigen::Matrix<double, 16, 1> x = s->_imu->value();
      x.head<4>() = rot_2_quat(exp_so3(V3(.025 * k, -.015 * k, .018 * k)));
      x.segment<3>(4) = V3(.24 * k, .015 * k * k, .025 * k);
      x.segment<3>(7) = V3(.35 + .025 * k, -.06 + .02 * k, .04);
      s->_imu->set_value(x);
      s->_imu->set_fej(x);
      const V3 omega(.55 + .025 * k, -.3 + .01 * k, .35 - .015 * k);
      StateHelper::augment_clone(s, omega, omega);
      if (fej) {
        auto pose = s->_clones_IMU.at(t);
        Eigen::Matrix<double, 7, 1> v = pose->value();
        v.head<4>() = rot_2_quat(exp_so3(V3(.012, -.009, .006)) * pose->Rot());
        v.tail<3>() += V3(.018, -.012, .007);
        pose->set_fej(v);
        s->_clones_kinematics[t].vel_fej += V3(.03, -.02, .01);
        s->_clones_kinematics[t].omega_fej += V3(.01, .005, -.007);
      }
      if (timing == 2) {
        PreintBridgeData b;
        b.valid = true;
        b.dt = .018;
        b.DR = exp_so3(-omega * b.dt);
        b.w_end = omega;
        b.alpha = V3(.0006, -.0003, .0012);
        b.beta = V3(.04, -.015, .08);
        b.p_grav = V3(0, 0, -.5 * 9.81 * b.dt * b.dt);
        b.v_grav = V3(0, 0, -9.81 * b.dt);
        b.J_b.topLeftCorner<3, 3>() = b.dt * M3::Identity();
        b.J_b.block<3, 3>(3, 3) = -.5 * b.dt * b.dt * M3::Identity();
        s->_epoch_bridges[t][1] = b;
        s->_epoch_residuals[t][1] = b.dt;
      }
    }
    const int n = s->max_covariance_size();
    Mat P = .001 * Mat::Identity(n, n);
    for (int c = 0; c < 2; ++c) {
      if (s->cam_imu_dt_var(c)->id() >= 0)
        P(s->cam_imu_dt_var(c)->id(), s->cam_imu_dt_var(c)->id()) = 1e-6;
      if (s->_calib_camera_readout[c]->id() >= 0)
        P(s->_calib_camera_readout[c]->id(),
          s->_calib_camera_readout[c]->id()) = 1e-6;
    }
    StateHelper::set_initial_covariance(s, P, state_order(s));
  }
  V3 exposure_point(int camera, double t, double row) const {
    const auto &clone = s->_clones_IMU.at(t);
    const auto &cal = s->_calib_IMUtoCAM.at(camera);
    const auto kin = s->_clones_kinematics.at(t);
    M3 R = clone->Rot();
    V3 p = clone->pos(), v = kin.vel, w = kin.omega;
    const auto b = s->epoch_bridge(camera, t);
    double tau = s->cam_imu_dt_delta(camera) +
                 (row / 480. - s->_options.rs_row_anchor) *
                     s->_calib_camera_readout.at(camera)->value()(0);
    if (b) {
      R = b->DR * R;
      p += kin.vel * b->dt + b->p_grav + clone->Rot().transpose() * b->alpha;
      v += b->v_grav + clone->Rot().transpose() * b->beta;
      w = b->w_end;
    } else
      tau += s->epoch_residual(camera, t);
    R = exp_so3(-w * tau) * R;
    p += v * tau;
    return cal->Rot() * R * (point - p) + cal->pos();
  }
  Eigen::Vector2f observation(int camera, double t) const {
    double row = 240.;
    Eigen::Vector2d uv;
    for (int k = 0; k < 20; ++k) {
      const V3 p = exposure_point(camera, t, row);
      uv << 400 * p.x() / p.z() + 320, 405 * p.y() / p.z() + 240;
      row = uv.y();
    }
    return uv.cast<float>();
  }
  std::shared_ptr<Feature> track(bool noise = false, bool stereo = true) const {
    auto f = std::make_shared<Feature>();
    f->featid = 123;
    f->quality = 1.;
    for (int camera : stereo ? std::vector<int>{0, 1} : std::vector<int>{1})
      for (size_t k = 0; k < times.size(); ++k) {
        Eigen::Vector2f uv = observation(camera, times[k]);
        if (noise) {
          uv.x() += .04 * std::sin(.7 * k + camera);
          uv.y() += .03 * std::cos(.5 * k - camera);
        }
        Eigen::Vector2f un;
        un << (uv.x() - 320) / 400, (uv.y() - 240) / 405;
        f->timestamps[camera].push_back(times[k]);
        f->uvs[camera].push_back(uv);
        f->uvs_norm[camera].push_back(un);
      }
    return f;
  }
  HF helper(Rep rep) const {
    HF f;
    auto obs = track();
    f.featid = obs->featid;
    f.timestamps = obs->timestamps;
    f.uvs = obs->uvs;
    f.uvs_norm = obs->uvs_norm;
    f.feat_representation = rep;
    f.anchor_cam_id = 1;
    f.anchor_clone_timestamp = times.front();
    f.p_FinG = f.p_FinG_fej = point;
    f.p_FinA = local(s, 1, times.front(), point);
    f.p_FinA_fej = f.p_FinA;
    return f;
  }
};
struct Linear {
  Mat Hf, Hx;
  Eigen::VectorXd r;
  std::vector<std::shared_ptr<Type>> order;
};
Linear linear(const std::shared_ptr<State> &s, HF f) {
  Linear o;
  UpdaterHelper::get_feature_jacobian_full(s, f, o.Hf, o.Hx, o.r, o.order);
  return o;
}
Mat align(const std::shared_ptr<State> &s, const Linear &a) {
  Mat H = Mat::Zero(a.Hx.rows(), s->max_covariance_size());
  int k = 0;
  for (const auto &v : a.order) {
    H.middleCols(v->id(), v->size()) = a.Hx.middleCols(k, v->size());
    k += v->size();
  }
  return H;
}
void getters() {
  for (auto rep : reps) {
    Landmark p(dimension(rep));
    p._feat_representation = rep;
    const V3 current(.4, -.3, 4.), frozen(-.2, .6, 6.);
    p.set_from_xyz(current, false);
    p.set_from_xyz(frozen, true);
    check((p.get_xyz(false) - current).norm() < 2e-14 &&
              (p.get_xyz(true) - frozen).norm() < 2e-14,
          "landmark current/FEJ depth and bearing pairs are independent");
    auto copy = std::dynamic_pointer_cast<Landmark>(p.clone());
    check((copy->get_xyz(true) - frozen).norm() < 2e-14,
          "landmark clone preserves frozen bearing/depth");
  }
}
void representation_checks() {
  double maxfd = 0;
  for (auto rep : reps)
    for (bool fej : {false, true}) {
      Fixture a(2, fej, true);
      auto f = a.helper(rep);
      Mat hf;
      std::vector<Mat> hx;
      std::vector<std::shared_ptr<Type>> order;
      UpdaterHelper::get_feature_jacobian_representation(a.s, f, hf, hx, order);
      check(order.size() == 2 &&
                order[0] == a.s->_clones_IMU.at(f.anchor_clone_timestamp) &&
                order[1] == a.s->_calib_IMUtoCAM.at(1),
            "virtual representation exports only anchor pose/extrinsic, no "
            "timing/readout/bias columns");
      const V3 best = world(a.s, 1, f.anchor_clone_timestamp, f.p_FinA);
      auto frozen = StateHelper::clone_state(a.s);
      if (fej)
        for (auto &p : frozen->_clones_IMU)
          p.second->set_value(p.second->fej());
      frozen->_options.do_fej = false;
      const V3 pA = local(frozen, 1, f.anchor_clone_timestamp, best);
      const double eps = 1e-6;
      for (int block = 0; block < 2; ++block)
        for (int c = 0; c < 6; ++c) {
          auto p = StateHelper::clone_state(frozen),
               m = StateHelper::clone_state(frozen);
          auto vp =
              block ? std::static_pointer_cast<Type>(p->_calib_IMUtoCAM.at(1))
                    : std::static_pointer_cast<Type>(
                          p->_clones_IMU.at(f.anchor_clone_timestamp));
          auto vm =
              block ? std::static_pointer_cast<Type>(m->_calib_IMUtoCAM.at(1))
                    : std::static_pointer_cast<Type>(
                          m->_clones_IMU.at(f.anchor_clone_timestamp));
          Eigen::VectorXd dx = Eigen::VectorXd::Zero(6);
          dx(c) = eps;
          vp->update(dx);
          vm->update(-dx);
          const V3 fd = (world(p, 1, f.anchor_clone_timestamp, pA) -
                         world(m, 1, f.anchor_clone_timestamp, pA)) /
                        (2 * eps);
          maxfd =
              std::max(maxfd, (fd - hx[block].col(c)).cwiseAbs().maxCoeff());
        }
      for (int c = 0; c < dimension(rep); ++c) {
        Landmark p(dimension(rep)), m(dimension(rep));
        p._feat_representation = m._feat_representation = rep;
        p.set_from_xyz(pA, false);
        m.set_from_xyz(pA, false);
        Eigen::VectorXd dx = Eigen::VectorXd::Zero(dimension(rep));
        dx(c) = eps;
        p.update(dx);
        m.update(-dx);
        const V3 fd =
            (world(frozen, 1, f.anchor_clone_timestamp, p.get_xyz(false)) -
             world(frozen, 1, f.anchor_clone_timestamp, m.get_xyz(false))) /
            (2 * eps);
        maxfd = std::max(maxfd, (fd - hf.col(c)).cwiseAbs().maxCoeff());
      }
      // Temporal metadata changes must not redefine an already-stored landmark.
      a.s->_epoch_bridges.clear();
      a.s->_epoch_residuals.clear();
      a.s->_clones_kinematics.clear();
      for (int cam = 0; cam < 2; ++cam) {
        Eigen::VectorXd value(1);
        value << .08 + .01 * cam;
        a.s->cam_imu_dt_var(cam)->set_value(value);
        a.s->_calib_camera_readout[cam]->set_value(value);
      }
      Mat hf2;
      std::vector<Mat> hx2;
      std::vector<std::shared_ptr<Type>> o2;
      UpdaterHelper::get_feature_jacobian_representation(a.s, f, hf2, hx2, o2);
      check(hf == hf2 && hx.size() == hx2.size() && hx[0] == hx2[0] &&
                hx[1] == hx2[1],
            "anchor representation is unchanged by td/RS/bridge/kinematic "
            "metadata");
      const V3 gravity(0, 0, 1);
      const auto clone = frozen->_clones_IMU.at(f.anchor_clone_timestamp);
      Eigen::Matrix<double, 6, 1> N;
      N << clone->Rot() * gravity, -skew_x(clone->pos()) * gravity;
      check((hx[0] * N + skew_x(best) * gravity).norm() < 1e-11,
            "virtual anchor representation preserves global-yaw covariance "
            "direction at its stated FEJ point");
    }
  std::printf("VIRTUAL_REPRESENTATION maxFD=%.3e\n", maxfd);
  check(maxfd < 2e-7, "all virtual anchor representation derivatives match "
                      "exact double geometry");
}
void system_equivalence() {
  for (int mode : {0, 1, 2})
    for (bool fej : {false, true}) {
      Fixture a(mode, fej, true);
      auto g = linear(a.s, a.helper(LandmarkRepresentation::GLOBAL_3D));
      const Mat Hg = align(a.s, g);
      Mat gf = g.Hf, gx = Hg;
      auto gr = g.r;
      UpdaterHelper::nullspace_project_inplace(gf, gx, gr);
      for (auto rep : reps) {
        if (dimension(rep) != 3)
          continue;
        auto f = a.helper(rep);
        auto b = linear(a.s, f);
        check((g.r - b.r).cwiseAbs().maxCoeff() < 1e-9,
              "global and virtual anchored means agree for physical "
              "td/RS/bridge observations");
        Mat bf = b.Hf, bx = align(a.s, b);
        auto br = b.r;
        UpdaterHelper::nullspace_project_inplace(bf, bx, br);
        const double info = (gx.transpose() * gx - bx.transpose() * bx).norm() /
                            std::max(1., (gx.transpose() * gx).norm());
        const double score = (gx.transpose() * gr - bx.transpose() * br).norm();
        std::printf("VIRTUAL_SYSTEM mode=%d fej=%d rep=%d "
                    "relative_information=%.3e score=%.3e\n",
                    mode, fej, rep, info, score);
        check(info < 2e-10 && score < 2e-6,
              "feature-eliminated information and score are invariant to "
              "virtual 3D anchor coordinates");
      }
      // The own-anchor image is generated from its exposure pose. Its virtual
      // pFinA is not the measured exposure bearing when timing transport is
      // active.
      auto anchored = a.helper(LandmarkRepresentation::ANCHORED_3D);
      anchored.timestamps = {{1, {a.times.front()}}};
      anchored.uvs = {{1, {a.observation(1, a.times.front())}}};
      anchored.uvs_norm = {{1, {Eigen::Vector2f::Zero()}}};
      auto one = linear(a.s, anchored);
      check(one.r.norm() < 5e-5, "own-anchor exposure projects the world point "
                                 "through observer transport correctly");
    }
}
struct SlamProbe : UpdaterSLAM {
  using UpdaterSLAM::perform_anchor_change;
  using UpdaterSLAM::UpdaterSLAM;
  size_t scratch_capacity() const { return _virtual_anchor_scratch.capacity(); }
  size_t scratch_size() const { return _virtual_anchor_scratch.size(); }
  const void *scratch_data() const { return _virtual_anchor_scratch.data(); }
  size_t scratch_record_bytes() const { return sizeof(VirtualAnchorPose); }
#ifndef OV_TEST_LEGACY_RS_RETRY
  size_t row_camera_capacity() const { return _row_camera_scratch.capacity(); }
  size_t row_motion_capacity() const { return _row_motion_scratch.capacity(); }
  const void *row_camera_data() const { return _row_camera_scratch.data(); }
  const void *row_motion_data() const { return _row_motion_scratch.data(); }
  size_t row_camera_bytes() const { return sizeof(RowCameraSnapshot); }
  size_t row_motion_bytes() const { return sizeof(RowMotionSnapshot); }
#endif
};
// Independent coordinate-map oracle. FEJ in the production helper holds the
// best current world point and moves the clone linearization to its frozen
// pose. It does not use the independently stored historical landmark FEJ point.
Mat reanchor_transition(const std::shared_ptr<State> &state, size_t id,
                        double new_time, int new_camera) {
  auto s = StateHelper::clone_state(state);
  auto l = s->_features_SLAM.at(id);
  const int old_camera = l->_anchor_cam_id;
  const double old_time = l->_anchor_clone_timestamp;
  const V3 best = world(s, old_camera, old_time, l->get_xyz(false));
  if (s->_options.do_fej) {
    for (auto &p : s->_clones_IMU)
      p.second->set_value(p.second->fej());
    l->set_from_xyz(local(s, old_camera, old_time, best), false);
  }
  s->_options.do_fej = false;
  auto evaluate = [&](const std::shared_ptr<State> &x) {
    auto p = x->_features_SLAM.at(id);
    if (new_camera < 0)
      return world(x, old_camera, old_time, p->get_xyz(false)).eval();
    Landmark out(3);
    out._feat_representation = p->_feat_representation;
    out.set_from_xyz(local(x, new_camera, new_time,
                           world(x, old_camera, old_time, p->get_xyz(false))),
                     false);
    return V3(out.value());
  };
  const int n = s->max_covariance_size();
  Mat T = Mat::Identity(n, n);
  T.middleRows(l->id(), 3).setZero();
  std::set<int> relevant{s->_clones_IMU.at(old_time)->id(), l->id(),
                         s->_calib_IMUtoCAM.at(old_camera)->id()};
  if (new_camera >= 0) {
    relevant.insert(s->_clones_IMU.at(new_time)->id());
    relevant.insert(s->_calib_IMUtoCAM.at(new_camera)->id());
  }
  const double eps = 1e-6;
  for (const auto &v : state_order(s)) {
    if (!relevant.count(v->id()))
      continue;
    for (int c = 0; c < v->size(); ++c) {
      auto plus = StateHelper::clone_state(s),
           minus = StateHelper::clone_state(s);
      const auto plus_order = state_order(plus),
                 minus_order = state_order(minus);
      Eigen::VectorXd dx = Eigen::VectorXd::Zero(v->size());
      dx(c) = eps;
      for (const auto &p : plus_order)
        if (p->id() == v->id())
          p->update(dx);
      for (const auto &p : minus_order)
        if (p->id() == v->id())
          p->update(-dx);
      T.block(l->id(), v->id() + c, 3, 1) =
          (evaluate(plus) - evaluate(minus)) / (2 * eps);
    }
  }
  return T;
}
void reanchor_checks() {
  UpdaterOptions op, aruco;
  FeatureInitializerOptions fi;
  SlamProbe update(op, aruco, fi);
  double max_covariance_error = 0;
  for (auto rep : reps)
    for (bool fej : {false, true}) {
      Fixture a(2, fej, true);
      auto landmark = std::make_shared<Landmark>(dimension(rep));
      landmark->_featid = 123;
      landmark->_feat_representation = rep;
      landmark->_anchor_cam_id = 1;
      landmark->_anchor_clone_timestamp = a.times.front();
      const V3 frozen_world = a.point + V3(.2, -.15, .3);
      landmark->set_from_xyz(local(a.s, 1, a.times.front(), a.point), false);
      landmark->set_from_xyz(local(a.s, 1, a.times.front(), frozen_world, true),
                             true);
      StateHelper::initialize_invertible(
          a.s, landmark, {a.s->_imu}, Mat::Zero(dimension(rep), 15),
          Mat::Identity(dimension(rep), dimension(rep)),
          .01 * Mat::Identity(dimension(rep), dimension(rep)),
          Eigen::VectorXd::Zero(dimension(rep)));
      a.s->_features_SLAM[123] = landmark;
      const int n = a.s->max_covariance_size();
      Mat L = Mat::Identity(n, n);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
          L(i, j) = .015 * std::sin(.3 * i + .7 * j);
      StateHelper::set_initial_covariance(a.s, .001 * L * L.transpose(),
                                          state_order(a.s));
      int step = 0;
      for (double t :
           {a.times[2], a.times[5], a.times[1], a.times[4], a.times[0]}) {
        const int camera = (step++ % 2) ? 0 : 1;
        Mat predicted;
        if (dimension(rep) == 3) {
          const Mat T = reanchor_transition(a.s, 123, t, camera);
          predicted = T * StateHelper::get_full_covariance(a.s) * T.transpose();
        }
        update.perform_anchor_change(a.s, landmark, t, camera);
        check(
            (world(a.s, camera, t, landmark->get_xyz(false)) - a.point).norm() <
                2e-12,
            "repeated reanchor preserves current world point under virtual "
            "semantics");
        check((world(a.s, camera, t, landmark->get_xyz(true), true) -
               frozen_world)
                      .norm() < 2e-12,
              "repeated reanchor preserves independently frozen world point "
              "and bearing");
        check(StateHelper::get_full_covariance(a.s).allFinite(),
              "reanchor covariance remains finite for every supported anchored "
              "representation");
        if (dimension(rep) == 3) {
          const double error =
              (predicted - StateHelper::get_full_covariance(a.s))
                  .cwiseAbs()
                  .maxCoeff();
          max_covariance_error = std::max(max_covariance_error, error);
          check(error < 2e-9, "repeated 3D reanchor transforms all state cross "
                              "covariance according to geometric FD");
        }
      }
    }
  std::printf("VIRTUAL_REANCHOR maxCovarianceFD=%.3e (3D representations; "
              "single-depth keeps its existing rank-one projection)\n",
              max_covariance_error);
}
void public_handoffs() {
  UpdaterOptions options, aruco;
  FeatureInitializerOptions fi;
  for (int mode : {0, 1, 2})
    for (bool fej : {false, true}) {
      Fixture base(mode, fej, false);
      auto global = StateHelper::clone_state(base.s),
           anchored = StateHelper::clone_state(base.s);
      global->_options.feat_rep_msckf = LandmarkRepresentation::GLOBAL_3D;
      anchored->_options.feat_rep_msckf =
          LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
      auto fg = base.track(true), fa = base.track(true);
      std::vector<std::shared_ptr<Feature>> vg{fg}, va{fa};
      UpdaterMSCKF ug(options, fi), ua(options, fi);
      ug.update(global, vg);
      ua.update(anchored, va);
      check(vg.size() == 1 && va.size() == 1,
            "public MSCKF update accepts matched global and virtual anchored "
            "tracks with existing gates");
      const double pc = (StateHelper::get_full_covariance(global) -
                         StateHelper::get_full_covariance(anchored))
                            .cwiseAbs()
                            .maxCoeff();
      double mean = (global->_imu->value() - anchored->_imu->value())
                        .cwiseAbs()
                        .maxCoeff();
      for (auto &p : global->_clones_IMU)
        mean = std::max(mean, (p.second->value() -
                               anchored->_clones_IMU.at(p.first)->value())
                                  .cwiseAbs()
                                  .maxCoeff());
      std::printf(
          "VIRTUAL_MSCKF_HANDOFF mode=%d fej=%d maxCov=%.3e maxMean=%.3e\n",
          mode, fej, pc, mean);
      check(
          pc < 2e-10 && mean < 2e-9,
          "actual MSCKF triangulation/handoff/update is coordinate invariant");
      for (auto rep : reps) {
        Fixture a(mode, fej, false);
        a.s->_options.feat_rep_slam = rep;
        auto track = a.track(false, false);
        std::vector<std::shared_ptr<Feature>> tracks{track};
        UpdaterSLAM us(options, aruco, fi);
        us.delayed_init(a.s, tracks);
        check(a.s->_features_SLAM.count(123) == 1,
              "public delayed SLAM initialization succeeds for anchored "
              "representation with unchanged gates");
        if (!a.s->_features_SLAM.count(123))
          continue;
        auto l = a.s->_features_SLAM.at(123);
        const V3 reconstructed =
            world(a.s, l->_anchor_cam_id, l->_anchor_clone_timestamp,
                  l->get_xyz(false));
        const double err = (reconstructed - track->p_FinG).norm();
        std::printf(
            "VIRTUAL_SLAM_HANDOFF mode=%d fej=%d rep=%d world_error=%.3e\n",
            mode, fej, rep, err);
        check(err < 3e-5 && (reconstructed - a.point).norm() < 3e-5,
              "stored SLAM virtual coordinates reconstruct triangulated world "
              "point");
        if (mode != 0)
          check((track->p_FinA - l->get_xyz(false)).norm() > 1e-4,
                "frontend exposure coordinates remain distinct from stored "
                "virtual anchor coordinates");
      }
    }
}
// Persistent SLAM keeps landmark columns, unlike the eliminated MSCKF system.
// Compare a public update in two equivalent local coordinate charts, explicitly
// matching the global feature's FEJ point to the anchored helper's best current
// world point. This does not assert equivalence of different FEJ histories.
void persistent_slam_checks() {
  UpdaterOptions options, aruco;
  FeatureInitializerOptions fi;
  double max_covariance = 0, max_navigation = 0;
  for (int mode : {0, 1, 2})
    for (bool fej : {false, true})
      for (auto rep : reps) {
        if (dimension(rep) != 3)
          continue;
        Fixture a(mode, fej, true);
        auto l = std::make_shared<Landmark>(3);
        l->_featid = 123;
        l->_feat_representation = rep;
        l->_anchor_cam_id = 1;
        l->_anchor_clone_timestamp = a.times.front();
        l->set_from_xyz(local(a.s, 1, a.times.front(), a.point), false);
        l->set_from_xyz(local(a.s, 1, a.times.front(), a.point, true), true);
        StateHelper::initialize_invertible(
            a.s, l, {a.s->_imu}, Mat::Zero(3, 15), Mat::Identity(3, 3),
            .01 * Mat::Identity(3, 3), Eigen::Vector3d::Zero());
        a.s->_features_SLAM[123] = l;
        const int n = a.s->max_covariance_size();
        Mat L = Mat::Identity(n, n);
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < i; ++j)
            L(i, j) = .015 * std::sin(.3 * i + .7 * j);
        const Mat Pa = .001 * L * L.transpose();
        StateHelper::set_initial_covariance(a.s, Pa, state_order(a.s));
        auto global = StateHelper::clone_state(a.s);
        auto g = global->_features_SLAM.at(123);
        g->_feat_representation = LandmarkRepresentation::GLOBAL_3D;
        g->set_from_xyz(a.point, false);
        g->set_from_xyz(a.point, true);
        const Mat T = reanchor_transition(a.s, 123, 0, -1);
        StateHelper::set_initial_covariance(global, T * Pa * T.transpose(),
                                            state_order(global));
        auto ta = a.track(true), tg = a.track(true);
        std::vector<std::shared_ptr<Feature>> va{ta}, vg{tg};
        UpdaterSLAM ua(options, aruco, fi), ug(options, aruco, fi);
        ua.update(a.s, va);
        ug.update(global, vg);
        check(va.size() == 1 && vg.size() == 1,
              "equivalent persistent SLAM updates pass the same existing gate");
        const double covariance =
            (T * StateHelper::get_full_covariance(a.s) * T.transpose() -
             StateHelper::get_full_covariance(global))
                .cwiseAbs()
                .maxCoeff();
        double navigation = 0;
        const auto av = state_order(a.s), gv = state_order(global);
        for (size_t i = 0; i < av.size(); ++i) {
          if (av[i]->id() == l->id())
            continue;
          navigation =
              std::max(navigation,
                       (av[i]->value() - gv[i]->value()).cwiseAbs().maxCoeff());
        }
        max_covariance = std::max(max_covariance, covariance);
        max_navigation = std::max(max_navigation, navigation);
        check(covariance < 2e-8 && navigation < 2e-9,
              "persistent SLAM navigation/calibration correction and joint "
              "covariance agree under the geometric coordinate transform");
      }
  std::printf("VIRTUAL_PERSISTENT_SLAM maxCovarianceFD=%.3e "
              "maxNavigationDifference=%.3e\n",
              max_covariance, max_navigation);
}
void batched_initialization_checks() {
  UpdaterOptions options, aruco;
  FeatureInitializerOptions fi;
  double max_seed_error = 0;
  double smallest_live_pose_change = 1;
  for (int mode : {0, 1, 2})
    for (bool fej : {false, true})
      for (auto rep : reps) {
        Fixture a(mode, fej, true);
        a.s->_options.feat_rep_slam = rep;
        auto batch = StateHelper::clone_state(a.s);
        auto first = StateHelper::clone_state(a.s);
        std::vector<std::shared_ptr<Feature>> tracks;
        const V3 original_point = a.point;
        for (int k = 0; k < 3; ++k) {
          a.point = original_point + V3(.13 * k, -.09 * k, .35 * k);
          auto track = a.track(true, false);
          track->featid += k;
          tracks.push_back(track);
        }
        std::vector<std::shared_ptr<Feature>> only_first{
            std::make_shared<Feature>(*tracks.front())};
        UpdaterSLAM first_update(options, aruco, fi),
            batch_update(options, aruco, fi);
        first_update.delayed_init(first, only_first);
        batch_update.delayed_init(batch, tracks);
        check(only_first.size() == 1 && tracks.size() == 3 &&
                  batch->_features_SLAM.size() == 3,
              "three-feature delayed initialization passes existing gates");
        for (const auto &track : tracks) {
          const auto l = batch->_features_SLAM.at(track->featid);
          // The stored FEJ value is the seed handed to StateHelper::initialize;
          // its following EKF correction changes only the current value. This
          // exposes snapshot ownership without test hooks in production code.
          const V3 expected =
              local(a.s, track->anchor_cam_id, track->anchor_clone_timestamp,
                    track->p_FinG);
          const double error = (l->get_xyz(true) - expected).norm();
          max_seed_error = std::max(max_seed_error, error);
          check(error < 2e-12,
                "every batched virtual anchor seed uses its triangulation "
                "snapshot despite earlier sequential EKF updates");
          if (mode == 0)
            check((expected - track->p_FinA).norm() < 2e-12,
                  "zero-transport batch preserves frontend anchor seed "
                  "coordinate convention");
          if (track->featid == 124) {
            const double changed =
                (local(first, track->anchor_cam_id,
                       track->anchor_clone_timestamp, track->p_FinG) -
                 expected)
                    .norm();
            smallest_live_pose_change =
                std::min(smallest_live_pose_change, changed);
            check(
                changed > 1e-8,
                "first initialization measurably moves the next feature's "
                "live anchor/extrinsics, making the batch test discriminating");
          }
        }
      }
  std::printf("VIRTUAL_BATCH maxSeedError=%.3e "
              "minChangedLiveAnchorSeed=%.3e\n",
              max_seed_error, smallest_live_pose_change);
}
void scratch_reuse_checks() {
  UpdaterOptions options, aruco;
  FeatureInitializerOptions fi;
  Fixture a(1, false, true);
  SlamProbe updater(options, aruco, fi), global_updater(options, aruco, fi);
  auto run = [&](SlamProbe &u, bool global, bool larger) {
    auto state = StateHelper::clone_state(a.s);
    state->_options.feat_rep_slam =
        global ? LandmarkRepresentation::GLOBAL_3D
               : LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    if (larger)
      state->_options.max_clone_size += 7;
    std::vector<std::shared_ptr<Feature>> tracks{a.track(true, false)};
    u.delayed_init(state, tracks);
    check(tracks.size() == 1, "scratch reuse fixture initializes its feature");
    return static_cast<size_t>(state->_options.num_cameras) *
           (static_cast<size_t>(state->_options.max_pose_clones()) + 1);
  };
  run(global_updater, true, false);
  check(global_updater.scratch_capacity() == 0 &&
            global_updater.scratch_size() == 0,
        "global-only initialization does not allocate virtual anchor scratch");
  const size_t bound = run(updater, false, false);
  const size_t capacity = updater.scratch_capacity();
  const void *storage = updater.scratch_data();
  check(capacity >= bound,
        "virtual scratch reserves the configured camera/window bound");
  for (int k = 0; k < 4; ++k) {
    run(updater, false, false);
    check(updater.scratch_capacity() == capacity &&
              updater.scratch_data() == storage,
          "repeated delayed-init calls reuse virtual scratch allocation");
  }
  const size_t larger_bound = run(updater, false, true);
  check(updater.scratch_capacity() >= larger_bound &&
            updater.scratch_capacity() > capacity,
        "virtual scratch grows for a larger configured window bound");
  const size_t grown = updater.scratch_capacity();
  const size_t grown_size = updater.scratch_size();
  storage = updater.scratch_data();
  run(updater, true, false);
  check(updater.scratch_capacity() == grown &&
            updater.scratch_size() == grown_size &&
            updater.scratch_data() == storage,
        "global-only path leaves retained scratch storage untouched");
  run(updater, false, false);
  check(updater.scratch_capacity() == grown &&
            updater.scratch_data() == storage,
        "virtual scratch does not shrink when the configured window decreases");
  std::printf("VIRTUAL_SCRATCH record_bytes=%zu initial_capacity=%zu "
              "grown_capacity=%zu reused_calls=4\n",
              updater.scratch_record_bytes(), capacity, grown);
}

// A corrupted secondary-camera correspondence must fail the unchanged stereo
// chi-square gate. Its valid anchor-camera track is then retriangulated by the
// actual delayed_init mono retry. Compare the stored FEJ seed to that same mono
// track initialized through the normal RS path at the pre-batch snapshot.
void rolling_shutter_retry_checks() {
  UpdaterOptions options, aruco;
  FeatureInitializerOptions fi;
  double max_seed_difference = 0.0, max_truth_error = 0.0;
  double min_live_calibration_change = 1.0, min_live_endpoint_change = 1.0;
  int retries = 0;
  for (int mode : {1, 2})
    for (bool fej : {false, true})
      for (auto rep : reps)
        for (double row_anchor : {0.0, 0.5, 1.0}) {
          Fixture a(mode, fej, true);
          a.s->_options.feat_rep_slam = rep;
          a.s->_options.rs_row_anchor = row_anchor;
          Eigen::VectorXd tr(1); tr << .03;
          a.s->_calib_camera_readout.at(1)->set_value(tr);
          a.s->_calib_camera_readout.at(1)->set_fej(tr);
          for (auto &entry : a.s->_epoch_bridges)
            entry.second.at(1).w_end += V3(.12, -.07, .1);
          auto first_track = a.track(true, false);
          a.point += V3(.2, .5, .6);
          auto reference = a.track(false, false);
          reference->featid = 124;
          auto stereo = a.track(false, true);
          stereo->featid = 124;
          // Camera 1 has the most observations, so the retry owns its good
          // temporal track. Pixel/normalized-coordinate arrays stay aligned.
          stereo->timestamps.at(0).pop_back();
          stereo->uvs.at(0).pop_back();
          stereo->uvs_norm.at(0).pop_back();
          for (size_t k = 0; k < stereo->uvs.at(0).size(); ++k) {
            auto &uv = stereo->uvs.at(0)[k];
            uv.x() += 16.0 * (k % 2 ? 1.0 : -1.0);
            uv.y() += 16.0 * .7 * std::sin(1.7 * k + .3);
            stereo->uvs_norm.at(0)[k] = a.s->_cam_intrinsics_cameras.at(0)->undistort_f(uv);
          }
          auto mono_state = StateHelper::clone_state(a.s);
          auto first_state = StateHelper::clone_state(a.s);
          auto batch_state = StateHelper::clone_state(a.s);
          const int initial_dimension = a.s->max_covariance_size();
          std::vector<std::shared_ptr<Feature>> mono_tracks{reference};
          std::vector<std::shared_ptr<Feature>> first_tracks{std::make_shared<Feature>(*first_track)};
          std::vector<std::shared_ptr<Feature>> batch_tracks{first_track, stereo};
          UpdaterSLAM mono_update(options, aruco, fi), first_update(options, aruco, fi), batch_update(options, aruco, fi);
          mono_update.delayed_init(mono_state, mono_tracks);
          first_update.delayed_init(first_state, first_tracks);
          batch_update.delayed_init(batch_state, batch_tracks);
          check(mono_state->_features_SLAM.count(124) == 1 && first_state->_features_SLAM.count(123) == 1 &&
                    batch_state->_features_SLAM.count(123) == 1 && batch_state->_features_SLAM.count(124) == 1,
                "actual RS delayed initialization accepts the earlier feature and the mono retry");
          if (!mono_state->_features_SLAM.count(124) || !batch_state->_features_SLAM.count(124)) continue;
          const auto expected = mono_state->_features_SLAM.at(124);
          const auto retried = batch_state->_features_SLAM.at(124);
          check(retried->demoted_to_mono && retried->_anchor_cam_id == 1,
                "corrupted stereo view fails the unchanged gate and reaches the actual anchor-camera retry");
          retries += retried->demoted_to_mono;
          const double error = (expected->get_xyz(true) - retried->get_xyz(true)).norm();
          max_seed_difference = std::max(max_seed_difference, error);
          check(error < 3e-12,
                "mono retry retains the same per-row triangulation and virtual-anchor seed as the pre-batch normal path");
          const V3 reconstructed = world(a.s, expected->_anchor_cam_id, expected->_anchor_clone_timestamp, expected->get_xyz(true));
          const double truth_error = (reconstructed - a.point).norm();
          max_truth_error = std::max(max_truth_error, truth_error);
          check(truth_error < 3e-5, "normal row-time triangulation agrees with independent analytic point truth");
          check(batch_state->max_covariance_size() == initial_dimension + 2 * dimension(rep),
                "retry metadata adds no estimator dimensions beyond the two intended landmarks");
          const double calibration_change =
              (first_state->_calib_IMUtoCAM.at(1)->value() - a.s->_calib_IMUtoCAM.at(1)->value()).norm() +
              (first_state->_calib_camera_readout.at(1)->value() - a.s->_calib_camera_readout.at(1)->value()).norm();
          min_live_calibration_change = std::min(min_live_calibration_change, calibration_change);
          check(calibration_change > 1e-9, "earlier insertion moves the live RS/extrinsic calibration in the retry fixture");
          if (mode == 2) {
            double endpoint_change = 0.0;
            for (const auto &entry : a.s->_epoch_bridges) {
              const auto &beta = entry.second.at(1).beta;
              endpoint_change = std::max(endpoint_change,
                  ((first_state->_clones_IMU.at(entry.first)->Rot().transpose() -
                    a.s->_clones_IMU.at(entry.first)->Rot().transpose()) * beta).norm());
            }
            min_live_endpoint_change = std::min(min_live_endpoint_change, endpoint_change);
            check(endpoint_change > 1e-9, "earlier insertion changes bridge endpoint velocity if recomputed from the live pose");
          }
        }
  check(retries == 48, "all endpoint/clone-rate, FEJ, representation and row-convention cases exercised mono retry");
  std::printf("RS_RETRY cases=%d maxSeedDifference=%.3e maxTruthError=%.3e minLiveCalibrationChange=%.3e minLiveEndpointChange=%.3e\n",
              retries, max_seed_difference, max_truth_error, min_live_calibration_change, min_live_endpoint_change);
}

#ifndef OV_TEST_LEGACY_RS_RETRY
void rolling_shutter_scratch_checks() {
  UpdaterOptions options, aruco;
  FeatureInitializerOptions fi;
  SlamProbe updater(options, aruco, fi);
  auto run = [&](int mode, bool larger) {
    Fixture a(mode, false, true);
    a.s->_options.feat_rep_slam = LandmarkRepresentation::ANCHORED_3D;
    if (larger) a.s->_options.max_clone_size += 7;
    std::vector<std::shared_ptr<Feature>> tracks{a.track(true, false)};
    updater.delayed_init(a.s, tracks);
    check(tracks.size() == 1, "RS metadata reuse fixture initializes normally");
    return static_cast<size_t>(a.s->_options.num_cameras) *
           (static_cast<size_t>(a.s->_options.max_pose_clones()) + 1);
  };
  run(1, false);
  check(updater.row_camera_capacity() == 0 && updater.row_motion_capacity() == 0,
        "global-shutter calls allocate no additional row-time metadata");
  const size_t bound = run(2, false);
  const size_t camera_capacity = updater.row_camera_capacity(), motion_capacity = updater.row_motion_capacity();
  const void *cameras = updater.row_camera_data(), *motion = updater.row_motion_data();
  check(camera_capacity >= 2 && motion_capacity >= bound, "RS metadata reserves the configured camera/window bound");
  for (int i = 0; i < 4; ++i) {
    run(2, false);
    check(updater.row_camera_capacity() == camera_capacity && updater.row_motion_capacity() == motion_capacity &&
              updater.row_camera_data() == cameras && updater.row_motion_data() == motion,
          "repeated RS calls reuse small contiguous camera/motion metadata");
  }
  run(1, false);
  check(updater.row_camera_data() == cameras && updater.row_motion_data() == motion &&
            updater.row_motion_capacity() == motion_capacity,
        "subsequent global-shutter calls leave RS storage untouched");
  const size_t larger_bound = run(2, true);
  check(updater.row_motion_capacity() >= larger_bound && updater.row_motion_capacity() > motion_capacity,
        "RS motion metadata grows only when the configured window bound grows");
  std::printf("RS_SCRATCH camera_record_bytes=%zu motion_record_bytes=%zu camera_capacity=%zu motion_capacity=%zu\n",
              updater.row_camera_bytes(), updater.row_motion_bytes(), camera_capacity, motion_capacity);
}
#endif
} // namespace
int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  getters();
  representation_checks();
  system_equivalence();
  reanchor_checks();
  public_handoffs();
  persistent_slam_checks();
  batched_initialization_checks();
  scratch_reuse_checks();
  rolling_shutter_retry_checks();
#ifndef OV_TEST_LEGACY_RS_RETRY
  rolling_shutter_scratch_checks();
#endif
  std::printf("VIRTUAL_ANCHOR %s failures=%d\n", failures ? "FAIL" : "PASS",
              failures);
  return failures ? 1 : 0;
}
