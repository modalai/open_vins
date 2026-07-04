/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 * @author Joao Leonardo Silva Cotta (@zauberflote1)
 *
 * Async dual-camera A/B harness (ROS-free).
 *
 * Runs the simulator + VioManager with per-camera truth injection (frame phase offsets, per-camera
 * cam-IMU time offsets, rolling-shutter readout) and reports RMSE / NEES against ground truth plus
 * scheduling counters (per-camera updates, out-of-order drops, clone-window span). Thresholds are
 * asserted from the command line so CTest can gate stages against the frozen synced baseline.
 *
 * Usage:
 *   test_async_dual <config.yaml> [--traj <traj.txt>] [--mono] [--phase1 <s>] [--dt0 <s>] [--dt1 <s>]
 *                   [--readout1 <s>] [--jitter] [--seed <n>] [--csv <out.csv>] [--name <label>]
 *                   [--assert-pos-rmse <m>] [--assert-ori-rmse <deg>] [--assert-nees-max <v>]
 */

#include <cinttypes>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "sim/Simulator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"

using namespace ov_msckf;

// One buffered camera event (mirrors the run_simulation one-event delay so IMU always leads)
struct CamEvent {
  double time_cam = -1;
  std::vector<int> camids;
  std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> feats;
};

int main(int argc, char **argv) {

  // -------------------- argument parsing --------------------
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <config.yaml> [options]\n", argv[0]);
    return EXIT_FAILURE;
  }
  std::string config_path = argv[1];
  std::string traj_path, csv_path, name = "run";
  double phase1 = 0.0, readout1 = 0.0;
  bool has_dt0 = false, has_dt1 = false, mono = false, jitter = false;
  double dt0 = 0.0, dt1 = 0.0;
  int seed = -1;
  double assert_pos_rmse = -1, assert_ori_rmse = -1, assert_nees_max = -1;
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    auto next = [&](double &val) { val = std::atof(argv[++i]); };
    if (arg == "--traj")
      traj_path = argv[++i];
    else if (arg == "--csv")
      csv_path = argv[++i];
    else if (arg == "--name")
      name = argv[++i];
    else if (arg == "--phase1")
      next(phase1);
    else if (arg == "--dt0") {
      next(dt0);
      has_dt0 = true;
    } else if (arg == "--dt1") {
      next(dt1);
      has_dt1 = true;
    } else if (arg == "--readout1")
      next(readout1);
    else if (arg == "--mono")
      mono = true;
    else if (arg == "--jitter")
      jitter = true;
    else if (arg == "--seed")
      seed = std::atoi(argv[++i]);
    else if (arg == "--assert-pos-rmse")
      next(assert_pos_rmse);
    else if (arg == "--assert-ori-rmse")
      next(assert_ori_rmse);
    else if (arg == "--assert-nees-max")
      next(assert_nees_max);
    else {
      std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
      return EXIT_FAILURE;
    }
  }

  // -------------------- configuration --------------------
  ov_core::Printer::setPrintLevel("WARNING");
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  VioManagerOptions params;
  params.print_and_load(parser);
  params.print_and_load_simulation(parser);
  params.num_opencv_threads = 0; // repeatability
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;
  if (!traj_path.empty())
    params.sim_traj_path = traj_path;
  if (mono)
    params.use_stereo = false;
  if (seed >= 0) {
    params.sim_seed_measurements = seed;
    params.sim_seed_state_init = seed + 1;
    params.sim_seed_preturb = seed + 2;
  }

  // Per-camera truth injection (defaults were sized by print_and_load_simulation)
  int num_cams = std::max(1, params.state_options.num_cameras);
  if (has_dt0)
    params.sim_camimu_dts.at(0) = dt0;
  if (num_cams > 1) {
    params.sim_cam_phase_offsets.at(1) = phase1;
    if (has_dt1)
      params.sim_camimu_dts.at(1) = dt1;
    params.sim_cam_readouts.at(1) = readout1;
  }
  if (!parser->successful()) {
    std::fprintf(stderr, "unable to parse all parameters, please fix\n");
    return EXIT_FAILURE;
  }

  auto sim = std::make_shared<Simulator>(params);
  auto sys = std::make_shared<VioManager>(params);
  const VioManagerOptions truth = sim->get_true_parameters();

  // -------------------- ground-truth initialization --------------------
  // State clock runs in the (reference) camera clock: subtract cam0's TRUE offset
  double next_imu_time = sim->current_timestamp() + 1.0 / params.sim_freq_imu;
  Eigen::Matrix<double, 17, 1> imustate;
  if (!sim->get_state(next_imu_time, imustate)) {
    std::fprintf(stderr, "[SIM]: could not initialize the filter to the first state\n");
    return EXIT_FAILURE;
  }
  imustate(0, 0) -= truth.sim_camimu_dts.at(0);
  sys->initialize_with_gt(imustate);

  // -------------------- metrics state --------------------
  const double settle_s = 5.0;
  double t_start_camclock = imustate(0, 0);
  double sum_pos_sq = 0.0, sum_ori_sq = 0.0, sum_nees = 0.0;
  size_t n_err = 0, n_nees = 0;
  std::vector<size_t> updates_per_cam(num_cams, 0);
  size_t drops_out_of_order = 0, events_fed = 0;
  double sum_clones = 0.0, sum_window = 0.0;
  size_t n_statesamples = 0;
  std::ofstream csv;
  if (!csv_path.empty()) {
    csv.open(csv_path);
    csv << "t,cam,pos_err,ori_err_deg,nees,clones,window_s\n";
  }

  // Evaluate estimate vs truth after an update triggered by camera `camid` at stamp `t_state`
  auto evaluate = [&](int camid, double t_state) {
    auto state = sys->get_state();
    if (state == nullptr || !sys->initialized())
      return;
    if (t_state - t_start_camclock < settle_s)
      return;
    // Truth at the IMU-clock instant of this camera stamp
    Eigen::Matrix<double, 17, 1> gt;
    if (!sim->get_state(t_state + truth.sim_camimu_dts.at(camid), gt))
      return;
    // Errors in the OpenVINS/JPL error convention: q_true = dq(+dtheta/2) x q_est ; dp = p_true - p_est
    Eigen::Vector4d q_est = state->_imu->quat();
    Eigen::Vector3d p_est = state->_imu->pos();
    Eigen::Vector4d q_true = gt.block(1, 0, 4, 1);
    Eigen::Vector4d dq = ov_core::quat_multiply(q_true, ov_core::Inv(q_est));
    Eigen::Vector3d dtheta = 2.0 * dq.block(0, 0, 3, 1);
    if (dq(3) < 0)
      dtheta *= -1.0;
    Eigen::Vector3d dp = gt.block(5, 0, 3, 1) - p_est;
    double pos_err = dp.norm();
    double ori_err_deg = 180.0 / M_PI * dtheta.norm();
    sum_pos_sq += pos_err * pos_err;
    sum_ori_sq += ori_err_deg * ori_err_deg;
    n_err++;
    // 6-dof pose NEES from the marginal covariance (order: [dtheta; dp])
    std::vector<std::shared_ptr<ov_type::Type>> vars{state->_imu->pose()};
    Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, vars);
    Eigen::Matrix<double, 6, 1> e;
    e << dtheta, dp;
    Eigen::Matrix<double, 6, 6> Psym = 0.5 * (P + P.transpose());
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(Psym);
    if (ldlt.info() == Eigen::Success) {
      double nees = e.dot(ldlt.solve(e));
      if (std::isfinite(nees)) {
        sum_nees += nees;
        n_nees++;
      }
    }
    // Window statistics
    double clones = (double)state->_clones_IMU.size();
    double window = 0.0;
    if (state->_clones_IMU.size() >= 2)
      window = state->_clones_IMU.rbegin()->first - state->_clones_IMU.begin()->first;
    sum_clones += clones;
    sum_window += window;
    n_statesamples++;
    if (csv.is_open())
      csv << t_state << "," << camid << "," << pos_err << "," << ori_err_deg << "," << (n_nees ? sum_nees / n_nees : -1) << "," << clones
          << "," << window << "\n";
  };

  // Feed one buffered event into the estimator (counting prospective out-of-order drops)
  auto feed_event = [&](CamEvent &ev) {
    if (ev.time_cam < 0)
      return;
    auto state = sys->get_state();
    if (state != nullptr && sys->initialized() && state->_timestamp > ev.time_cam)
      drops_out_of_order++;
    sys->feed_measurement_simulation(ev.time_cam, ev.camids, ev.feats);
    events_fed++;
    for (int cid : ev.camids)
      if (cid >= 0 && cid < num_cams)
        updates_per_cam.at(cid)++;
    if (!ev.camids.empty())
      evaluate(ev.camids.front(), ev.time_cam);
    ev = CamEvent();
  };

  // -------------------- main loop (mirrors run_simulation, one-event buffer) --------------------
  signal(SIGINT, SIG_DFL);
  CamEvent buffered, held; // `held` implements the deterministic jitter swap (B3 reproduction)
  size_t pair_counter = 0;
  while (sim->ok()) {
    ov_core::ImuData m_imu;
    if (sim->get_next_imu(m_imu.timestamp, m_imu.wm, m_imu.am)) {
      sys->feed_measurement_imu(m_imu);
    }
    CamEvent ev;
    if (sim->get_next_cam(ev.time_cam, ev.camids, ev.feats)) {
      if (jitter) {
        // Swap delivery order of every 8th adjacent event pair
        if (held.time_cam < 0) {
          if ((pair_counter++ % 8) == 0) {
            held = ev; // hold this one; the NEXT event will be fed first (out of order)
            continue;
          }
        } else {
          feed_event(buffered);
          feed_event(ev);   // newer event first
          feed_event(held); // held (older) event second -> dropped by the estimator gate
          continue;
        }
      }
      feed_event(buffered);
      buffered = ev;
    }
  }
  feed_event(held);
  feed_event(buffered);

  // -------------------- summary --------------------
  double pos_rmse = (n_err > 0) ? std::sqrt(sum_pos_sq / n_err) : -1;
  double ori_rmse = (n_err > 0) ? std::sqrt(sum_ori_sq / n_err) : -1;
  double nees_avg = (n_nees > 0) ? sum_nees / n_nees : -1;
  double clones_avg = (n_statesamples > 0) ? sum_clones / n_statesamples : -1;
  double window_avg = (n_statesamples > 0) ? sum_window / n_statesamples : -1;
  auto st = sys->get_state();
  double est_dt0 = (st != nullptr) ? st->cam_imu_dt(0) : 0.0;
  double est_dt1 = (st != nullptr && num_cams > 1) ? st->cam_imu_dt(1) : 0.0;
  double est_rd1 = (st != nullptr && num_cams > 1) ? st->_calib_camera_readout.at(1)->value()(0) : 0.0;
  std::printf("[RESULT] name=%s pos_rmse=%.4f ori_rmse=%.4f nees_avg=%.2f updates_cam0=%zu updates_cam1=%zu drops=%zu "
              "clones_avg=%.1f window_avg=%.3f events=%zu samples=%zu est_dt0=%.6f est_dt1=%.6f est_rd1=%.6f kin_miss=%" PRIu64 "\n",
              name.c_str(), pos_rmse, ori_rmse, nees_avg, updates_per_cam.at(0), (num_cams > 1 ? updates_per_cam.at(1) : 0),
              drops_out_of_order, clones_avg, window_avg, events_fed, n_err, est_dt0, est_dt1, est_rd1,
              (st != nullptr) ? st->_kin_miss_count : 0);

  // -------------------- assertions --------------------
  bool ok = true;
  bool any_assert = (assert_pos_rmse > 0 || assert_ori_rmse > 0 || assert_nees_max > 0);
  if (n_err == 0) {
    std::printf("[FAIL] no error samples collected (estimator never initialized or diverged early)\n");
    ok = false;
  }
  if (assert_pos_rmse > 0 && !(pos_rmse >= 0 && pos_rmse <= assert_pos_rmse)) {
    std::printf("[FAIL] pos_rmse %.4f > %.4f\n", pos_rmse, assert_pos_rmse);
    ok = false;
  }
  if (assert_ori_rmse > 0 && !(ori_rmse >= 0 && ori_rmse <= assert_ori_rmse)) {
    std::printf("[FAIL] ori_rmse %.4f > %.4f\n", ori_rmse, assert_ori_rmse);
    ok = false;
  }
  if (assert_nees_max > 0 && !(nees_avg >= 0 && nees_avg <= assert_nees_max)) {
    std::printf("[FAIL] nees_avg %.2f > %.2f\n", nees_avg, assert_nees_max);
    ok = false;
  }
  // A run without thresholds proves nothing -- never print PASS for it
  std::printf("%s\n", !ok ? "[FAILED]" : (any_assert ? "[PASS]" : "[DONE unverified]"));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
