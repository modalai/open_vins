/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
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

#include "VioManager.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#if OV_HAVE_MODAL_FLOW
#include "track/TrackOCL/TrackOCL.h"
#endif
#ifndef DISABLE_TRACK_KLT
#include "track/TrackKLT.h"
#endif
#include "track/TrackSIM.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

VioManager::VioManager(VioManagerOptions &params_) : thread_init_running(false), thread_init_success(false) {

  // Nice startup message
  PRINT_DEBUG("=======================================\n");
  PRINT_DEBUG("OPENVINS ON-MANIFOLD EKF IS STARTING\n");
  PRINT_DEBUG("=======================================\n");

  // Nice debug
  this->params = params_;
  params.print_and_load_estimator();
  params.print_and_load_noise();
  params.print_and_load_state();
  params.print_and_load_trackers();

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(params.num_opencv_threads);
  cv::setRNGSeed(0);

  // Forward manager-level knobs consumed inside the state/updaters
  params.state_options.epoch_bridge_bias_cols = params.epoch_bridge_bias_cols;

  // Shutter declarations decide which cameras carry an ESTIMATED readout state under
  // calib_cam_readout (undeclared legacy rigs infer from a nonzero readout value)
  for (int i = 0; i < params.state_options.num_cameras; i++) {
    const bool rolling = params.camera_shutter_rolling.count((size_t)i)
                             ? params.camera_shutter_rolling.at((size_t)i)
                             : (params.camera_readout_time.count((size_t)i) && params.camera_readout_time.at((size_t)i) != 0.0);
    params.state_options.camera_estimate_readout[(size_t)i] = rolling;
  }

  // Config-drift guard: an unsynced multi-camera rig running WITHOUT epoch-anchored cloning
  // clones at every camera's frame time, so the shared window covers only window/N seconds per
  // camera -- no track can reach max-track length (SLAM starves at zero features forever) and
  // near-hover MSCKF triangulation collapses with the baseline. This is exactly what a STALE
  // deployed estimator_config.yaml (missing the async block, epoch_mode defaulting to false)
  // looks like, and it dead-reckons into a health auto-reset loop on hardware.
  if (params.state_options.num_cameras >= 2 && !params.use_stereo && !params.epoch_mode) {
    PRINT_WARNING(RED "=======================================================================\n" RESET);
    PRINT_WARNING(RED "VioManager(): %d UNSYNCED cameras with epoch_mode DISABLED!\n" RESET, params.state_options.num_cameras);
    PRINT_WARNING(RED "Per-frame cloning divides the clone window across cameras: SLAM features\n" RESET);
    PRINT_WARNING(RED "cannot reach max-track length and MSCKF triangulation loses its baseline.\n" RESET);
    PRINT_WARNING(RED "Set epoch_mode: true (async rigs) or use_stereo: true (synced pairs).\n" RESET);
    PRINT_WARNING(RED "If you DID set it, the DEPLOYED estimator_config.yaml is an older file\n" RESET);
    PRINT_WARNING(RED "that lacks the async block -- regenerate the on-target config.\n" RESET);
    PRINT_WARNING(RED "=======================================================================\n" RESET);
  }

  // Create the state!!
  state = std::make_shared<State>(params.state_options);

  // Set the IMU intrinsics
  state->_calib_imu_dw->set_value(params.vec_dw);
  state->_calib_imu_dw->set_fej(params.vec_dw);
  state->_calib_imu_da->set_value(params.vec_da);
  state->_calib_imu_da->set_fej(params.vec_da);
  state->_calib_imu_tg->set_value(params.vec_tg);
  state->_calib_imu_tg->set_fej(params.vec_tg);
  state->_calib_imu_GYROtoIMU->set_value(params.q_GYROtoIMU);
  state->_calib_imu_GYROtoIMU->set_fej(params.q_GYROtoIMU);
  state->_calib_imu_ACCtoIMU->set_value(params.q_ACCtoIMU);
  state->_calib_imu_ACCtoIMU->set_fej(params.q_ACCtoIMU);

  // Loop through and load each of the cameras
  state->_cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < state->_options.num_cameras; i++) {

    // Timeoffset from this camera to IMU (t_imu = t_cam_i + t_off_i); ref cam fills the legacy alias
    Eigen::VectorXd dt_val(1);
    dt_val(0) = params.camera_imu_dt.count(i) ? params.camera_imu_dt.at(i) : params.calib_camimu_dt;
    state->cam_imu_dt_var((size_t)i)->set_value(dt_val);
    state->cam_imu_dt_var((size_t)i)->set_fej(dt_val);

    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));

    // Rolling shutter readout time (0 = global shutter)
    Eigen::VectorXd readout_val(1);
    readout_val(0) = params.camera_readout_time.count(i) ? params.camera_readout_time.at(i) : 0.0;
    state->_calib_camera_readout.at(i)->set_value(readout_val);
    state->_calib_camera_readout.at(i)->set_fej(readout_val);
  }

  // The initializer runs in the reference camera's clock
  params.init_options.calib_camimu_dt = state->cam_imu_dt_ref();

  // Declared nominal frame rates seed the rate estimators that otherwise need frames to settle:
  // the epoch binding horizon runs at design width from the FIRST reference frame (fresh start
  // and every hard reset), instead of a 50ms bootstrap guess.
  {
    const int ref_id = params.state_options.cam_imu_dt_ref_camid;
    if (params.camera_fps.count((size_t)ref_id) && params.camera_fps.at((size_t)ref_id) > 0.0) {
      ref_period_ema = 1.0 / params.camera_fps.at((size_t)ref_id);
    }

    // Epoch-anchoring constraint: the reference must be the FASTEST declared camera. A slower
    // reference makes a faster camera deliver >1 frame per epoch; the extras hit the
    // one-frame-per-(camera,epoch) rule and fall back to their own clones, fragmenting the
    // window into mixed epoch/fallback times -- the fast camera's tracks split across them,
    // max-track/SLAM graduation starves, and the few surviving MSCKF features drag the
    // calibration (observed on hardware with ref=30Hz vs 42Hz: calib random-walk, divergence).
    if (params.epoch_mode) {
      for (auto const &fps : params.camera_fps) {
        if (params.camera_fps.count((size_t)ref_id) && fps.second > params.camera_fps.at((size_t)ref_id) + 1e-6) {
          PRINT_WARNING(RED "VioManager(): epoch reference cam%d (%.1f fps) is SLOWER than cam%zu (%.1f fps)!\n" RESET, ref_id,
                        params.camera_fps.at((size_t)ref_id), fps.first, fps.second);
          PRINT_WARNING(RED "\tExpect epoch-fallback churn and starved updates; set cam_imu_dt_ref_camid to the fastest camera.\n" RESET);
        }
      }
    }
  }

  // Lock-free async multi-camera ingest: one SPSC ring per stream, ordered release from the IMU
  // feed. Dropped frames flow through the processed-callback with processed=false so the owner can
  // release external image handles exactly once.
  AsyncCameraBuffer::Options buf_opts;
  buf_opts.ring_capacity = (size_t)std::max(2, params.async_ring_size);
  buf_opts.guard = params.async_guard;
  buf_opts.stale_factor = params.async_stale_factor;
  buf_opts.initial_periods.resize((size_t)state->_options.num_cameras, 0.0);
  for (int i = 0; i < state->_options.num_cameras; i++) {
    if (params.camera_fps.count((size_t)i) && params.camera_fps.at((size_t)i) > 0.0) {
      buf_opts.initial_periods[(size_t)i] = 1.0 / params.camera_fps.at((size_t)i);
    }
  }
  camera_buffer = std::make_shared<AsyncCameraBuffer>(state->_options.num_cameras, buf_opts, [this](const ov_core::CameraData &msg) {
    if (camera_processed_cb) {
      camera_processed_cb(msg, false);
    }
  });

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // If we are recording statistics, then open our file
  if (params.record_timing_information) {
    // If the file exists, then delete it
    if (boost::filesystem::exists(params.record_timing_filepath)) {
      boost::filesystem::remove(params.record_timing_filepath);
      PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    // Create the directory that we will open the file in
    boost::filesystem::path p(params.record_timing_filepath);
    boost::filesystem::create_directories(p.parent_path());
    // Open our statistics file!
    of_statistics.open(params.record_timing_filepath, std::ofstream::out | std::ofstream::app);
    // Write the header information into it
    of_statistics << "# timestamp (sec),tracking,propagation,msckf update,";
    if (state->_options.max_slam_features > 0) {
      of_statistics << "slam update,slam delayed,";
    }
    of_statistics << "re-tri & marg,total" << std::endl;
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Let's make a feature extractor
  // NOTE: after we initialize we will increase the total number of feature tracks
  // NOTE: we will split the total number of features over all cameras uniformly
  int init_max_features = std::floor((double)params.init_options.init_max_features / (double)params.state_options.num_cameras);
  if (params.use_klt) {
#if !OV_HAVE_MODAL_FLOW
    // Host/test builds without the VOXL sysroot have no TrackOCL: fall back to the CPU tracker
    if (params.use_gpu) {
      fprintf(stderr, "[VioManager] use_gpu requested but modal_flow/OpenCL is unavailable in this build; using CPU tracker\n");
    }
    if (false) {
#else
    if (params.use_gpu) {
      auto track_ocl = std::make_shared<TrackOCL>(state->_cam_intrinsics_cameras, init_max_features,
                                                  state->_options.max_aruco_features, params.use_stereo,
                                                  params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist);
      // Bind the ZNCC epipolar-band stereo matcher whenever stereo is on and
      // the calibration has been packed by VoxlConfigure. It is the only
      // stereo-projection path in TrackOCL; without it, stereo features stay
      // mono-only.
      PRINT_DEBUG("[VioManager] stereo: use_stereo=%d  calib_valid=%d\n",
                  (int)params.use_stereo, (int)params.stereo_calib_valid);
      if (params.use_stereo && params.stereo_calib_valid) {
        track_ocl->enable_zncc_stereo_matcher(params.stereo_calib,
                                              params.stereo_z_min,
                                              params.stereo_z_max);
      } else if (params.use_stereo) {
        fprintf(stderr, "[VioManager] stereo requested but stereo_calib not "
                        "populated; stereo features will stay mono-only\n");
      }
      trackFEATS = std::static_pointer_cast<TrackBase>(track_ocl);
#endif
    } else {
#ifndef DISABLE_TRACK_KLT
      trackFEATS = std::shared_ptr<TrackBase>(new TrackKLT(state->_cam_intrinsics_cameras, init_max_features,
                                                          state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
                                                          params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist));
#else
      PRINT_ERROR(RED "[VioManager]: TrackKLT is disabled at compile time. Use GPU mode or TrackDescriptor.\n" RESET);
      std::exit(EXIT_FAILURE);
#endif
    }
  } else {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackDescriptor(
        state->_cam_intrinsics_cameras, init_max_features, state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
        params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio));
  }

  // Initialize our aruco tag extractor
  if (params.use_aruco) {
    trackARUCO = std::shared_ptr<TrackBase>(new TrackAruco(state->_cam_intrinsics_cameras, state->_options.max_aruco_features,
                                                           params.use_stereo, params.histogram_method, params.downsize_aruco));
  }

  // Initialize our state propagator
  propagator = std::make_shared<Propagator>(params.imu_noises, params.gravity_mag, params.prop_window);

  // Our state initialize
  initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());

  // Make the updater!
  updaterMSCKF = std::make_shared<UpdaterMSCKF>(params.msckf_options, params.featinit_options);
  updaterSLAM = std::make_shared<UpdaterSLAM>(params.slam_options, params.aruco_options, params.featinit_options);

  // If we are using zero velocity updates, then create the updater
  if (params.try_zupt) {
    updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                        propagator, params.gravity_mag, params.zupt_max_velocity,
                                                        params.zupt_noise_multiplier, params.zupt_max_disparity, params.zupt_prop_window);
  }
}

void VioManager::soft_reset(SoftResetCause cause) {

  // SNAPSHOT the live bias state FIRST (before the EKF is torn down below): on a mid-ops reset the
  // filter's converged bg/ba (+ their marginal sigmas from _Cov) are the best available bias
  // knowledge, and the dynamic initializer consumes them as a GATED prior (CPI linearization
  // points, MLE seed, tightened first-pose bias prior -- see init_dyn_reset_prior_*). The caller
  // has already quiesced sensor callbacks, so reading the state here is race-free. An invalid
  // snapshot (never-initialized filter) still arms the context, which clears any stale prior.
  ov_init::ResetBiasPrior bias_prior;
  if (is_initialized_vio && timelastupdate != -1 && state != nullptr) {
    bias_prior.bg = state->_imu->bias_g();
    bias_prior.ba = state->_imu->bias_a();
    std::vector<std::shared_ptr<ov_type::Type>> bias_vars = {state->_imu->bg(), state->_imu->ba()};
    Eigen::MatrixXd Pbb = StateHelper::get_marginal_covariance(state, bias_vars); // 6x6 [bg, ba]
    bias_prior.sigma_bg = Pbb.block(0, 0, 3, 3).diagonal().cwiseMax(0.0).cwiseSqrt();
    bias_prior.sigma_ba = Pbb.block(3, 3, 3, 3).diagonal().cwiseMax(0.0).cwiseSqrt();
    bias_prior.t_snapshot = state->_timestamp;
    bias_prior.cause = (int)cause;
    bias_prior.valid = true;
    PRINT_INFO("[soft-reset]: bias prior snapshot |bg|=%.4f |ba|=%.4f (max sig %.4f/%.4f, cause=%d)\n", bias_prior.bg.norm(),
               bias_prior.ba.norm(), bias_prior.sigma_bg.maxCoeff(), bias_prior.sigma_ba.maxCoeff(), (int)cause);
  }
  if (initializer != nullptr)
    initializer->set_reset_prior(bias_prior);

  // Stop/clear the async initialization machinery (the caller has already quiesced sensor callbacks).
  {
    std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
    camera_queue_init.clear();
  }
  thread_init_running.store(false);
  thread_init_success.store(false);
  is_initialized_vio = false;
  timelastupdate = -1;
  startup_time = -1;

  // This is a mid-ops re-init with a warm front-end (feature DB + IMU history preserved below), so the
  // next successful initialization may warm-start. First boot / hard reset never set this -> cold-start.
  warmstart_next_init.store(true);

  // Fresh navigation EKF with the configured calibration (mirrors the constructor). The feature
  // tracker, inertial initializer (IMU history) and propagator are VioManager members and are
  // intentionally PRESERVED, so re-initialization reuses the already-buffered recent measurements.
  state = std::make_shared<State>(params.state_options);
  state->_calib_imu_dw->set_value(params.vec_dw);
  state->_calib_imu_dw->set_fej(params.vec_dw);
  state->_calib_imu_da->set_value(params.vec_da);
  state->_calib_imu_da->set_fej(params.vec_da);
  state->_calib_imu_tg->set_value(params.vec_tg);
  state->_calib_imu_tg->set_fej(params.vec_tg);
  state->_calib_imu_GYROtoIMU->set_value(params.q_GYROtoIMU);
  state->_calib_imu_GYROtoIMU->set_fej(params.q_GYROtoIMU);
  state->_calib_imu_ACCtoIMU->set_value(params.q_ACCtoIMU);
  state->_calib_imu_ACCtoIMU->set_fej(params.q_ACCtoIMU);
  state->_cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < state->_options.num_cameras; i++) {
    Eigen::VectorXd dt_val(1);
    dt_val(0) = params.camera_imu_dt.count(i) ? params.camera_imu_dt.at(i) : params.calib_camimu_dt;
    state->cam_imu_dt_var((size_t)i)->set_value(dt_val);
    state->cam_imu_dt_var((size_t)i)->set_fej(dt_val);
    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
    Eigen::VectorXd readout_val(1);
    readout_val(0) = params.camera_readout_time.count(i) ? params.camera_readout_time.at(i) : 0.0;
    state->_calib_camera_readout.at(i)->set_value(readout_val);
    state->_calib_camera_readout.at(i)->set_fej(readout_val);
  }

  // A soft reset starts a NEW estimation episode on a continuous sensor clock: purge everything
  // keyed by pre-reset update times or scoped to the old episode, or downstream consumers
  // (quality metric, extended packet) would present stale pre-reset evidence as current. The
  // margtimestep-based map cleanup cannot do this -- a warm-started clone window reaches ~2 s
  // back past the reset instant, so pre-reset entries would sit inside the new window.
  used_features_map.clear();
  // ZUPT episode flags are only written while initialized, so they would otherwise stay frozen
  // at their pre-reset values: a stale did_zupt_update=true lets the publisher report CEP
  // quality (fresh covariance => ~100) with state OK throughout the whole re-init, and a stale
  // has_moved_since_zupt=true disables ZUPT for the entire new episode when
  // zupt_only_at_beginning is set.
  did_zupt_update = false;
  has_moved_since_zupt = false;

  PRINT_INFO("[soft-reset]: EKF reset; feature DB + IMU history preserved for fast re-init\n");
}

void VioManager::feed_measurement_batch_imu(const std::vector<ov_core::ImuData>& messages, double target_freq_hz) {
    if (messages.empty()) return;

    // Calculate oldest time needed once for the whole batch
    double oldest_time = state->margtimestep();
    if (oldest_time > state->_timestamp) {
        oldest_time = -1;
    }
    if (!is_initialized_vio) {
        oldest_time = messages.back().timestamp - params.init_options.init_window_time +
                     state->cam_imu_dt_min() - params.zupt_prop_window;
    }

    // Downsample if requested
    std::vector<ov_core::ImuData> processed_messages;
    if (target_freq_hz > 0.0) {
        double dt = 1.0 / target_freq_hz;
        double next_time = messages.front().timestamp;
        
        for (const auto& msg : messages) {
            if (msg.timestamp >= next_time) {
                processed_messages.push_back(msg);
                next_time += dt;
            }
        }
    } else {
        processed_messages = messages;
    }

    // Use batch methods for each component
    propagator->feed_imu_batch(processed_messages, oldest_time);

    if (!is_initialized_vio) {
        initializer->feed_imu_batch(processed_messages, oldest_time);
    }

    if (is_initialized_vio && updaterZUPT != nullptr &&
        (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
        updaterZUPT->feed_imu_batch(processed_messages, oldest_time);
    }

    // Release every buffered camera frame whose global ordering this batch has decided
    newest_imu_time = std::max(newest_imu_time, messages.back().timestamp);
    drain_camera_buffer();
}

void VioManager::feed_measurement_camera(const ov_core::CameraData &message) {
  if (camera_buffer != nullptr) {
    camera_buffer->push(message);
  }
}

std::shared_ptr<VioManager::Snapshot> VioManager::snapshot() {
  auto snap = std::make_shared<Snapshot>();

  // EKF state: full independent deep clone (values + fej + covariance + clone/SLAM/calib maps +
  // epoch metadata). This is the non-reproducible core -- GPU tracking feeds updates
  // non-deterministically, so a past state cannot be re-derived by replaying, only captured.
  snap->state = StateHelper::clone_state(state);

  // Propagator IMU history + prop-time offset (fast-prop cache excluded; rebuilt on restore)
  if (propagator != nullptr)
    snap->prop = propagator->capture();

  // Tracker CPU frontend (pts_last / ids_last / currid / per-cam IMU-seed state) + feature DB
  if (trackFEATS != nullptr) {
    snap->track_front = trackFEATS->capture_frontend();
    if (auto db = trackFEATS->get_feature_database())
      snap->feature_db = db->clone_features();
  }

  // Manager scalars
  snap->is_initialized_vio = is_initialized_vio;
  snap->timelastupdate = timelastupdate;
  snap->startup_time = startup_time;
  snap->distance = distance;
  snap->newest_imu_time = newest_imu_time;
  snap->last_ref_frame_time = last_ref_frame_time;
  snap->ref_period_ema = ref_period_ema;
  snap->epoch_marg_pending = epoch_marg_pending;
  snap->used_features_map = used_features_map;

  return snap;
}

void VioManager::restore(const std::shared_ptr<Snapshot> &snap, const std::vector<ov_core::CameraData> &prime_frames) {
  if (snap == nullptr)
    return;

  // 1) Install a FRESH clone of the snapshot state, so the snapshot node stays pristine and can
  //    be restored again to spawn a second branch. Swapping the shared_ptr is safe: no sub-object
  //    (updater/propagator/initializer) caches the State -- all take it as a per-call argument.
  state = StateHelper::clone_state(snap->state);

  // 2) Propagator + scalars restored in place (Propagator object identity preserved -- ZUPT
  //    caches this exact pointer).
  if (propagator != nullptr)
    propagator->restore(snap->prop);

  is_initialized_vio = snap->is_initialized_vio;
  timelastupdate = snap->timelastupdate;
  startup_time = snap->startup_time;
  distance = snap->distance;
  newest_imu_time = snap->newest_imu_time;
  last_ref_frame_time = snap->last_ref_frame_time;
  ref_period_ema = snap->ref_period_ema;
  epoch_marg_pending = snap->epoch_marg_pending;
  used_features_map = snap->used_features_map;

  // Discard whatever the live ring holds now. The in-flight (pushed-but-not-drained) frames that
  // were pending at snapshot time are NOT re-injected here -- instead the caller rewinds its feed
  // cursor to the last PROCESSED frame, so those frames re-push from the log and re-drain
  // naturally as feeding resumes. (Restoring an old snapshot after the filter ran forward also
  // makes clearing correct: the live ring holds unrelated future frames.)
  clear_camera_buffers();

  // 3) Rebuild the GPU previous-frame pyramid. It is not host-visible, so we re-feed the snapshot
  //    frame's image(s) through the tracker: feed_new_camera uploads the frame into img_buf_next_
  //    (the next real feed then swaps it into img_buf_prev_, exactly as if we had never rewound).
  //    We first clear the CPU frontend so this priming feed DETECTS (it must not try to KLT-track
  //    from the stale post-snapshot pyramid); its detection output is overwritten in step 4.
  if (trackFEATS != nullptr && !prime_frames.empty()) {
    trackFEATS->restore_frontend(std::make_shared<ov_core::TrackBase::FrontendState>());
    for (const auto &f : prime_frames)
      trackFEATS->feed_new_camera(f);
  }

  // 4) Force-restore the tracker CPU frontend and feature database to the snapshot. This overwrites
  //    whatever the priming feed produced, so the NEXT real feed KLT-tracks the snapshot's features
  //    (in pts_last) from the just-rebuilt pyramid -- continuation is faithful to the snapshot.
  if (trackFEATS != nullptr) {
    trackFEATS->restore_frontend(snap->track_front);
    if (auto db = trackFEATS->get_feature_database())
      db->restore_features(snap->feature_db);
  }
}

bool VioManager::apply_epoch_snap(double &timestamp, const std::vector<int> &sensor_ids) {
  if (!params.epoch_mode || !is_initialized_vio) {
    return false;
  }
  const int ref_id = state->cam_imu_dt_ref_camid();
  const bool has_ref = std::find(sensor_ids.begin(), sensor_ids.end(), ref_id) != sensor_ids.end();
  if (has_ref) {
    if (last_ref_frame_time > 0 && timestamp > last_ref_frame_time) {
      const double period = timestamp - last_ref_frame_time;
      ref_period_ema = (ref_period_ema > 0) ? (0.9 * ref_period_ema + 0.1 * period) : period;
    }
    last_ref_frame_time = timestamp; // this frame IS the new epoch
    return false;
  }
  const double t_raw = timestamp;
  const double horizon = ((ref_period_ema > 0) ? ref_period_ema : 0.05) * params.epoch_bind_factor;
  bool can_bind = last_ref_frame_time > 0 && t_raw >= last_ref_frame_time && (t_raw - last_ref_frame_time) <= horizon &&
                  state->_clones_IMU.find(last_ref_frame_time) != state->_clones_IMU.end();
  if (can_bind) {
    // One frame per (camera, epoch): a rate-beat collision falls back to its own clone
    auto res_it = state->_epoch_residuals.find(last_ref_frame_time);
    for (int cid : sensor_ids) {
      if (res_it != state->_epoch_residuals.end() && res_it->second.count((size_t)cid) > 0) {
        can_bind = false;
      }
    }
  }
  if (!can_bind) {
    epoch_fallbacks++;
    return false;
  }
  auto &residuals = state->_epoch_residuals[last_ref_frame_time];
  for (int cid : sensor_ids) {
    residuals[(size_t)cid] = t_raw - last_ref_frame_time;
  }

  // Build the exact ACI2 bridge over the KNOWN residual, at the current bias estimates (IMU
  // coverage is guaranteed by the ingest release gate). If it cannot be built the updaters
  // degrade to the first-order model for this frame -- still snapped, still consistent.
  Propagator::BridgeData bd;
  const double t0_imu = last_ref_frame_time + state->cam_imu_dt_ref();
  if (propagator->compute_bridge(state, t0_imu, t0_imu + (t_raw - last_ref_frame_time), bd)) {
    auto &bmap = state->_epoch_bridges[last_ref_frame_time];
    for (int cid : sensor_ids) {
      bmap[(size_t)cid] = bd;
    }
  }

  timestamp = last_ref_frame_time;
  epoch_snapped++;
  return true;
}

void VioManager::drain_camera_buffer() {
  if (camera_buffer == nullptr) {
    return;
  }
  camera_buffer->drain(
      newest_imu_time, [this](const std::vector<int> &sensor_ids) { return state->cam_imu_dt_max_for_ids(sensor_ids); },
      [this](ov_core::CameraData &&msg) {
        track_image_and_update(msg);
        return (camera_processed_cb == nullptr) || camera_processed_cb(msg, true);
      });
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {

  // The oldest time we need IMU with is the last clone
  // We shouldn't really need the whole window, but if we go backwards in time we will
  double oldest_time = state->margtimestep();
  if (oldest_time > state->_timestamp) {
    oldest_time = -1;
  }
  if (!is_initialized_vio) {
    oldest_time = message.timestamp - params.init_options.init_window_time + state->cam_imu_dt_min() - params.zupt_prop_window;
  }
  propagator->feed_imu(message, oldest_time);

  // Push back to our initializer
  if (!is_initialized_vio) {
    initializer->feed_imu(message, oldest_time);
  }

  // Push back to the zero velocity updater if it is enabled
  // No need to push back if we are just doing the zv-update at the begining and we have moved
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    updaterZUPT->feed_imu(message, oldest_time);
  }

  // Release every buffered camera frame whose global ordering this sample has decided
  newest_imu_time = std::max(newest_imu_time, message.timestamp);
  drain_camera_buffer();
}

void VioManager::feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                             const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Check if we actually have a simulated tracker
  // If not, recreate and re-cast the tracker to our simulation tracker
  std::shared_ptr<TrackSIM> trackSIM = std::dynamic_pointer_cast<TrackSIM>(trackFEATS);
  if (trackSIM == nullptr) {
    // Replace with the simulated tracker
    trackSIM = std::make_shared<TrackSIM>(state->_cam_intrinsics_cameras, state->_options.max_aruco_features);
    trackFEATS = trackSIM;
    // Need to also replace it in init and zv-upt since it points to the trackFEATS db pointer
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());
    if (params.try_zupt) {
      updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                          propagator, params.gravity_mag, params.zupt_max_velocity,
                                                          params.zupt_noise_multiplier, params.zupt_max_disparity, params.zupt_prop_window);
    }
    PRINT_WARNING(RED "[SIM]: casting our tracker to a TrackSIM object!\n" RESET);
  }

  // Epoch-anchored cloning applies to the simulation path too (obs must land at clone times)
  apply_epoch_snap(timestamp, camids);

  // Feed our simulation tracker
  trackSIM->feed_measurement_simulation(timestamp, camids, feats);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == timestamp);
      // Clean against the SMALLEST per-cam offset: strictly conservative (keeps every sample any
      // camera's pending frame could still need; the per-cam dt spread is << prop_window)
      propagator->clean_old_imu_measurements(timestamp + state->cam_imu_dt_min() - params.prop_window);
      updaterZUPT->clean_old_imu_measurements(timestamp + state->cam_imu_dt_min() - params.prop_window);
      propagator->invalidate_cache();
      // timelastupdate = timestamp;
      return;
    }
  }

  // If we do not have VIO initialization, then return an error
  if (!is_initialized_vio) {
    PRINT_ERROR(RED "[SIM]: your vio system should already be initialized before simulating features!!!\n" RESET);
    PRINT_ERROR(RED "[SIM]: initialize your system first before calling feed_measurement_simulation()!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our propagate and update function
  // Simulation is either all sync, or single camera...
  ov_core::CameraData message;
  message.timestamp = timestamp;
  for (auto const &camid : camids) {
    int width = state->_cam_intrinsics_cameras.at(camid)->w();
    int height = state->_cam_intrinsics_cameras.at(camid)->h();
    message.sensor_ids.push_back(camid);
    message.images.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
  }
  do_feature_propagate_update(message);
}

void VioManager::track_image_and_update(const ov_core::CameraData &message_const) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Assert we have valid measurement data and ids
  assert(!message_const.sensor_ids.empty());
  assert(message_const.sensor_ids.size() == message_const.images.size());
  for (size_t i = 0; i < message_const.sensor_ids.size() - 1; i++) {
    assert(message_const.sensor_ids.at(i) != message_const.sensor_ids.at(i + 1));
  }

  // Downsample if we are downsampling
  ov_core::CameraData message = message_const;
  for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras; i++) {
    cv::Mat img = message.images.at(i);
    cv::Mat mask = message.masks.at(i);
    cv::Mat img_temp, mask_temp;
    cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
    message.images.at(i) = img_temp;
    cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
    message.masks.at(i) = mask_temp;
  }

  // Epoch-anchored cloning: only REFERENCE-camera frames define clone times; a non-reference
  // frame arriving within the binding horizon of the newest epoch is SNAPPED onto it -- its
  // timestamp becomes the epoch time (bit-exact, so every obs time equals a clone time across
  // the whole pipeline) and its KNOWN residual t_raw - t_epoch enters the measurement model's
  // dt_total. This keeps the clone rate at the reference rate and the window baseline intact
  // for unsynced rigs. Frames with no bindable epoch clone fall back to cloning (counted).
  apply_epoch_snap(message.timestamp, message.sensor_ids);

  // Perform our feature tracking!
  trackFEATS->feed_new_camera(message);

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    trackARUCO->feed_new_camera(message);
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != message.timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, message.timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == message.timestamp);
      // Conservative clean horizon (see feed_measurement_simulation note)
      propagator->clean_old_imu_measurements(message.timestamp + state->cam_imu_dt_min() - params.prop_window);
      updaterZUPT->clean_old_imu_measurements(message.timestamp + state->cam_imu_dt_min() - params.prop_window);
      propagator->invalidate_cache();
      // timelastupdate = message.timestamp;
      return;
    }
  }

  // If we do not have VIO initialization, then try to initialize
  // TODO: Or if we are trying to reset the system, then do that here!
  if (!is_initialized_vio) {
    is_initialized_vio = try_to_initialize(message);
    if (!is_initialized_vio) {
      double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
      PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
      return;
    }
  }

  // Call on our propagate and update function
  do_feature_propagate_update(message);
}

void VioManager::do_feature_propagate_update(const ov_core::CameraData &message) {

  //===================================================================================
  // State propagation, and clone augmentation
  //===================================================================================

  // Return if the camera measurement is out of order
  if (state->_timestamp > message.timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
                  (message.timestamp - state->_timestamp));
    return;
  }

  // Epoch mode defers the old-clone marginalization until the epoch is COMPLETE (a message with
  // a NEW time arrives): every camera's own update call must still see the full window, otherwise
  // non-reference tracks can never reach max-track length and never graduate to SLAM features
  // (they would be consumed as short MSCKF scraps at the reference camera's call instead).
  if (epoch_marg_pending && state->_timestamp != message.timestamp) {
    StateHelper::marginalize_old_clone(state);
    epoch_marg_pending = false;
  }

  // Propagate the state forward to the current update time
  // Also augment it with a new clone!
  // NOTE: if the state is already at the given time (can happen in sim)
  // NOTE: then no need to prop since we already are at the desired timestep
  if (state->_timestamp != message.timestamp) {
    if (!propagator->propagate_and_clone(state, message.timestamp)) {
      return; // no clone for this frame (duplicate/backward request): skip its update
    }
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // If we have not reached max clones, we should just return...
  // This isn't super ideal, but it keeps the logic after this easier...
  // We can start processing things when we have at least 5 clones since we can start triangulating things...
  if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, 5)) {
    PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(),
                std::min(state->_options.max_clone_size, 5));
    return;
  }

  // Return if we where unable to propagate
  if (state->_timestamp != message.timestamp) {
    PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state->_timestamp);
    return;
  }
  has_moved_since_zupt = true;

  //===================================================================================
  // MSCKF features and KLT tracks that are SLAM features
  //===================================================================================

  // Now, lets get all features that should be used for an update that are lost in the newest frame
  // We explicitly request features that have not been deleted (used) in another update step
  std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
  feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(state->_timestamp, false, true);

  // Don't need to get the oldest features until we reach our max number of clones
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5) {
    feats_marg = trackFEATS->get_feature_database()->features_containing(state->margtimestep(), false, true);
    if (trackARUCO != nullptr && message.timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(state->margtimestep(), false, true);
    }
  }

  // Remove any lost features that were from other image streams
  // E.g: if we are cam1 and cam0 has not processed yet, we don't want to try to use those in the update yet
  // E.g: thus we wait until cam0 process its newest image to remove features which were seen from that camera
  auto it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    bool found_current_message_camid = false;
    for (const auto &camuvpair : (*it1)->uvs) {
      if (std::find(message.sensor_ids.begin(), message.sensor_ids.end(), camuvpair.first) != message.sensor_ids.end()) {
        found_current_message_camid = true;
        break;
      }
    }
    if (found_current_message_camid) {
      it1++;
    } else {
      it1 = feats_lost.erase(it1);
    }
  }

  // We also need to make sure that the max tracks does not contain any lost features
  // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
  it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {
      // PRINT_WARNING(YELLOW "FOUND FEATURE THAT WAS IN BOTH feats_lost and feats_marg!!!!!!\n" RESET);
      it1 = feats_lost.erase(it1);
    } else {
      it1++;
    }
  }

  // Find tracks that have reached max length, these can be made into SLAM features
  std::vector<std::shared_ptr<Feature>> feats_maxtracks;
  auto it2 = feats_marg.begin();
  while (it2 != feats_marg.end()) {
    // See if any of our camera's reached max track
    bool reached_max = false;
    for (const auto &cams : (*it2)->timestamps) {
      if ((int)cams.second.size() > state->_options.max_clone_size) {
        reached_max = true;
        break;
      }
    }
    // If max track, then add it to our possible slam feature list
    if (reached_max) {
      feats_maxtracks.push_back(*it2);
      it2 = feats_marg.erase(it2);
    } else {
      it2++;
    }
  }

  // Count how many aruco tags we have in our state
  int curr_aruco_tags = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((int)(*it0).second->_featid <= 4 * state->_options.max_aruco_features)
      curr_aruco_tags++;
    it0++;
  }

  // Append a new SLAM feature if we have the room to do so
  // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
  if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay &&
      (int)state->_features_SLAM.size() < state->_options.max_slam_features + curr_aruco_tags) {
    // Get the total amount to add, then the max amount that we can add given our marginalize feature array
    int amount_to_add = (state->_options.max_slam_features + curr_aruco_tags) - (int)state->_features_SLAM.size();
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
    // If we have at least 1 that we can add, lets add it!
    // Note: we remove them from the feat_marg array since we don't want to reuse information...
    if (valid_amount > 0) {
      feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
      feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
    }
  }

  // Loop through current SLAM features, we have tracks of them, grab them for this update!
  // NOTE: if we have a slam feature that has lost tracking, then we should marginalize it out
  // NOTE: we only enforce this if the current camera message is where the feature was seen from
  // NOTE: if you do not use FEJ, these types of slam features *degrade* the estimator performance....
  // NOTE: we will also marginalize SLAM features if they have failed their update a couple times in a row
  for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : state->_features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat1 = trackARUCO->get_feature_database()->get_feature(landmark.second->_featid);
      if (feat1 != nullptr)
        feats_slam.push_back(feat1);
    }
    std::shared_ptr<Feature> feat2 = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
    if (feat2 != nullptr)
      feats_slam.push_back(feat2);
    assert(landmark.second->_unique_camera_id != -1);
    bool current_unique_cam =
        std::find(message.sensor_ids.begin(), message.sensor_ids.end(), landmark.second->_unique_camera_id) != message.sensor_ids.end();
    if (feat2 == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
    if (landmark.second->update_fail_count > 1)
      landmark.second->should_marg = true;
  }

  // Lets marginalize out all old SLAM features here
  // These are ones that where not successfully tracked into the current frame
  // We do *NOT* marginalize out our aruco tags landmarks
  StateHelper::marginalize_slam(state);

  // Separate our SLAM features into new ones, and old ones
  std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
  for (size_t i = 0; i < feats_slam.size(); i++) {
    if (state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_features_SLAM.end()) {
      feats_slam_UPDATE.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    } else {
      feats_slam_DELAYED.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: new feature ready %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    }
  }

  // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF = feats_lost;
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

  //===================================================================================
  // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
  //===================================================================================

  // Sort based on track length
  // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
  // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
  auto compare_feat = [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
    size_t asize = 0;
    size_t bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

  // Pass them to our MSCKF updater
  // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
  // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
  if ((int)featsup_MSCKF.size() > state->_options.max_msckf_in_update)
    featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - state->_options.max_msckf_in_update);
  updaterMSCKF->update(state, featsup_MSCKF);
  propagator->invalidate_cache();
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Perform SLAM delay init and update
  // NOTE: that we provide the option here to do a *sequential* update
  // NOTE: this will be a lot faster but won't be as accurate.
  std::vector<std::shared_ptr<Feature>> feats_slam_UPDATE_TEMP;
  while (!feats_slam_UPDATE.empty()) {
    // Get sub vector of the features we will update with
    std::vector<std::shared_ptr<Feature>> featsup_TEMP;
    featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATE.begin(),
                        feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    feats_slam_UPDATE.erase(feats_slam_UPDATE.begin(),
                            feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    // Do the update
    updaterSLAM->update(state, featsup_TEMP);
    feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
    propagator->invalidate_cache();
  }
  feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
  rT5 = boost::posix_time::microsec_clock::local_time();
  updaterSLAM->delayed_init(state, feats_slam_DELAYED);
  rT6 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Update our visualization feature set, and clean up the old features
  //===================================================================================

  // Re-triangulate all current tracks in the current frame
  if (message.sensor_ids.at(0) == 0) {

    // Re-triangulate features
    retriangulate_active_tracks(message);

    // Clear the MSCKF features only on the base camera
    // Thus we should be able to visualize the other unique camera stream
    // MSCKF features as they will also be appended to the vector
    good_features_MSCKF.clear();
  }

  // Save all the MSCKF features used in the update
  for (auto const &feat : featsup_MSCKF) {
    good_features_MSCKF.push_back(feat->p_FinG);
    feat->to_delete = true;
  }

  // Append used features to the timestamp-based map
  std::vector<std::shared_ptr<ov_core::Feature>> all_used_features;
  all_used_features.insert(all_used_features.end(), featsup_MSCKF.begin(), featsup_MSCKF.end());
  all_used_features.insert(all_used_features.end(), feats_slam_UPDATE.begin(), feats_slam_UPDATE.end());
  all_used_features.insert(all_used_features.end(), feats_slam_DELAYED.begin(), feats_slam_DELAYED.end());
  
  if (!all_used_features.empty()) {
    if (used_features_map.find(state->_timestamp) != used_features_map.end()) {
      used_features_map[state->_timestamp].insert(used_features_map[state->_timestamp].end(), 
                                                  all_used_features.begin(), all_used_features.end());
    } else {
      used_features_map[state->_timestamp] = all_used_features;
    }
  }

  //===================================================================================
  // Cleanup, marginalize out what we don't need any more...
  //===================================================================================

  // Remove features that where used for the update from our extractors at the last timestep
  // This allows for measurements to be used in the future if they failed to be used this time
  // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
  trackFEATS->get_feature_database()->cleanup();
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup();
  }

  // First do anchor change if we are about to lose an anchor pose
  updaterSLAM->change_anchors(state);

  // Cleanup any features older than the marginalization time
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    trackFEATS->get_feature_database()->cleanup_measurements(state->margtimestep());
    if (trackARUCO != nullptr) {
      trackARUCO->get_feature_database()->cleanup_measurements(state->margtimestep());
    }
    
    // Cleanup old entries from used_features_map (steady-state pruning only; the dynamic-reset
    // case is handled in soft_reset(), which purges the whole map -- margtimestep alone cannot,
    // since a warm-started window reaches back past the reset instant)
    auto it = used_features_map.begin();
    while (it != used_features_map.end()) {
      if (it->first < state->margtimestep()) {
        it = used_features_map.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Finally marginalize the oldest clone if needed
  if (params.epoch_mode) {
    // Defer: the epoch's remaining (snapped) camera calls must still see the full window so their
    // tracks can reach max-track length and graduate to SLAM; executed when the next NEW-time
    // message arrives (see the top of this function)
    epoch_marg_pending = ((int)state->_clones_IMU.size() > state->_options.max_clone_size);
  } else {
    StateHelper::marginalize_old_clone(state);
  }
  rT7 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
  double time_prop = (rT3 - rT2).total_microseconds() * 1e-6;
  double time_msckf = (rT4 - rT3).total_microseconds() * 1e-6;
  double time_slam_update = (rT5 - rT4).total_microseconds() * 1e-6;
  double time_slam_delay = (rT6 - rT5).total_microseconds() * 1e-6;
  double time_marg = (rT7 - rT6).total_microseconds() * 1e-6;
  double time_total = (rT7 - rT1).total_microseconds() * 1e-6;

  // Timing information
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation\n" RESET, time_prop);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats)\n" RESET, time_msckf, (int)featsup_MSCKF.size());
  if (state->_options.max_slam_features > 0) {
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM update (%d feats)\n" RESET, time_slam_update, (int)state->_features_SLAM.size());
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM delayed init (%d feats)\n" RESET, time_slam_delay, (int)feats_slam_DELAYED.size());
  }
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state)\n" RESET, time_marg, (int)state->_clones_IMU.size());

  // Epoch/ingest health: fallbacks should stay ~0 on a well-configured rig (fastest cam = ref).
  // A climbing fallback count = fragmented clone window = starved updates (see the ctor guard).
  if (params.epoch_mode && camera_buffer != nullptr) {
    PRINT_DEBUG(BLUE "[EPOCH]: %llu snapped, %llu fallbacks | ingest: %llu late, %llu full, %llu bogus drops\n" RESET,
                (unsigned long long)epoch_snapped, (unsigned long long)epoch_fallbacks,
                (unsigned long long)camera_buffer->count_drop_late(), (unsigned long long)camera_buffer->count_drop_full(),
                (unsigned long long)camera_buffer->count_drop_bogus());
  }

  std::stringstream ss;
  ss << "[TIME]: " << std::setprecision(4) << time_total << " seconds for total (camera";
  for (const auto &id : message.sensor_ids) {
    ss << " " << id;
  }
  ss << ")" << std::endl;
  PRINT_DEBUG(BLUE "%s" RESET, ss.str().c_str());

  // Finally if we are saving stats to file, lets save it to file
  if (params.record_timing_information && of_statistics.is_open()) {
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double t_ItoC = state->cam_imu_dt_ref();
    double timestamp_inI = state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_track << ","
                  << time_prop << "," << time_msckf << ",";
    if (state->_options.max_slam_features > 0) {
      of_statistics << time_slam_update << "," << time_slam_delay << ",";
    }
    of_statistics << time_marg << "," << time_total << std::endl;
    of_statistics.flush();
  }

  // Update our distance traveled
  if (timelastupdate != -1 && state->_clones_IMU.find(timelastupdate) != state->_clones_IMU.end()) {
    Eigen::Matrix<double, 3, 1> dx = state->_imu->pos() - state->_clones_IMU.at(timelastupdate)->pos();
    distance += dx.norm();
  }
  timelastupdate = message.timestamp;

  // Debug, print our current state
  PRINT_INFO("q_GtoI = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f | dist = %.2f (meters)\n", state->_imu->quat()(0),
             state->_imu->quat()(1), state->_imu->quat()(2), state->_imu->quat()(3), state->_imu->pos()(0), state->_imu->pos()(1),
             state->_imu->pos()(2), distance);
  PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2),
             state->_imu->bias_a()(0), state->_imu->bias_a()(1), state->_imu->bias_a()(2));

  // Debug for camera imu offset
  if (state->_options.do_calib_camera_timeoffset) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      PRINT_INFO("cam%d-imu timeoffset = %.5f\n", i, state->cam_imu_dt((size_t)i));
    }
  }

  // Debug for camera intrinsics
  if (state->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = state->_cam_intrinsics.at(i);
      PRINT_INFO("cam%d intrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f\n", (int)i, calib->value()(0), calib->value()(1),
                 calib->value()(2), calib->value()(3), calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Debug for camera extrinsics
  if (state->_options.do_calib_camera_pose) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = state->_calib_IMUtoCAM.at(i);
      PRINT_INFO("cam%d extrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f\n", (int)i, calib->quat()(0), calib->quat()(1), calib->quat()(2),
                 calib->quat()(3), calib->pos()(0), calib->pos()(1), calib->pos()(2));
    }
  }

  // Debug for imu intrinsics
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("q_GYROtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_GYROtoIMU->value()(0), state->_calib_imu_GYROtoIMU->value()(1),
               state->_calib_imu_GYROtoIMU->value()(2), state->_calib_imu_GYROtoIMU->value()(3));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
    PRINT_INFO("q_ACCtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_ACCtoIMU->value()(0), state->_calib_imu_ACCtoIMU->value()(1),
               state->_calib_imu_ACCtoIMU->value()(2), state->_calib_imu_ACCtoIMU->value()(3));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("Dw = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
               state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
               state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
               state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
               state->_calib_imu_da->value()(5));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
    PRINT_INFO("Dw = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
               state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
               state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
               state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
               state->_calib_imu_da->value()(5));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.do_calib_imu_g_sensitivity) {
    PRINT_INFO("Tg = | %.4f,%.4f,%.4f |  %.4f,%.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_tg->value()(0),
               state->_calib_imu_tg->value()(1), state->_calib_imu_tg->value()(2), state->_calib_imu_tg->value()(3),
               state->_calib_imu_tg->value()(4), state->_calib_imu_tg->value()(5), state->_calib_imu_tg->value()(6),
               state->_calib_imu_tg->value()(7), state->_calib_imu_tg->value()(8));
  }
}
