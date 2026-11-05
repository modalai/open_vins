# MODALAI - OpenVINS

[![ROS 1 Workflow](https://github.com/rpng/open_vins/actions/workflows/build_ros1.yml/badge.svg)](https://github.com/rpng/open_vins/actions/workflows/build_ros1.yml)
[![ROS 2 Workflow](https://github.com/rpng/open_vins/actions/workflows/build_ros2.yml/badge.svg)](https://github.com/rpng/open_vins/actions/workflows/build_ros2.yml)
[![ROS Free Workflow](https://github.com/rpng/open_vins/actions/workflows/build.yml/badge.svg)](https://github.com/rpng/open_vins/actions/workflows/build.yml)

Welcome to the OpenVINS project!
The OpenVINS project houses some core computer vision code along with a state-of-the art filter-based visual-inertial
estimator. The core filter is an [Extended Kalman filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter) which
fuses inertial information with sparse visual feature tracks. These visual feature tracks are fused leveraging
the [Multi-State Constraint Kalman Filter (MSCKF)](https://ieeexplore.ieee.org/document/4209642) sliding window
formulation which allows for 3D features to update the state estimate without directly estimating the feature states in
the filter. Inspired by graph-based optimization systems, the included filter has modularity allowing for convenient
covariance management with a proper type-based state system. Please take a look at the feature list below for full
details on what the system supports.

* Github project page - https://github.com/rpng/open_vins
* Documentation - https://docs.openvins.com/
* Getting started guide - https://docs.openvins.com/getting-started.html
* Publication reference - https://pgeneva.com/downloads/papers/Geneva2020ICRA.pdf

## News / Events

* **May 11, 2023** - Inertial intrinsic support released as part of v2.7 along with a few bug fixes and improvements to stereo KLT tracking. Please check out the [release page](https://github.com/rpng/open_vins/releases/tag/v2.7) for details.
* **April 15, 2023** - Minor update to v2.6.3 to support incremental feature triangulation of active features for downstream applications, faster zero-velocity update, small bug fixes, some example realsense configurations, and cached fast state prediction. Please check out the [release page](https://github.com/rpng/open_vins/releases/tag/v2.6.3) for details.
* **April 3, 2023** - We have released a monocular plane-aided VINS, termed [ov_plane](https://github.com/rpng/ov_plane), which leverages the OpenVINS project. Both now support the released [Indoor AR Table](https://github.com/rpng/ar_table_dataset) dataset.
* **July 14, 2022** - Improved feature extraction logic for >100hz tracking, some bug fixes and updated scripts. See v2.6.1 [PR#259](https://github.com/rpng/open_vins/pull/259) and v2.6.2 [PR#264](https://github.com/rpng/open_vins/pull/264).
* **March 14, 2022** - Initial dynamic initialization open sourcing, asynchronous subscription to inertial readings and publishing of odometry, support for lower frequency feature tracking. See v2.6 [PR#232](https://github.com/rpng/open_vins/pull/232) for details.
* **December 13, 2021** - New YAML configuration system, ROS2 support, Docker images, robust static initialization based on disparity, internal logging system to reduce verbosity, image transport publishers, dynamic number of features support, and other small fixes. See
  v2.5 [PR#209](https://github.com/rpng/open_vins/pull/209) for details.
* **July 19, 2021** - Camera classes, masking support, alignment utility, and other small fixes. See
  v2.4 [PR#117](https://github.com/rpng/open_vins/pull/186) for details.
* **December 1, 2020** - Released improved memory management, active feature pointcloud publishing, limiting number of
  features in update to bound compute, and other small fixes. See
  v2.3 [PR#117](https://github.com/rpng/open_vins/pull/117) for details.
* **November 18, 2020** - Released groundtruth generation utility package, [vicon2gt](https://github.com/rpng/vicon2gt)
  to enable creation of groundtruth trajectories in a motion capture room for evaulating VIO methods.
* **July 7, 2020** - Released zero velocity update for vehicle applications and direct initialization when standing
  still. See [PR#79](https://github.com/rpng/open_vins/pull/79) for details.
* **May 18, 2020** - Released secondary pose graph example
  repository [ov_secondary](https://github.com/rpng/ov_secondary) based
  on [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion). OpenVINS now publishes marginalized feature
  track, feature 3d position, and first camera intrinsics and extrinsics.
  See [PR#66](https://github.com/rpng/open_vins/pull/66) for details and discussion.
* **April 3, 2020** - Released [v2.0](https://github.com/rpng/open_vins/releases/tag/v2.0) update to the codebase with
  some key refactoring, ros-free building, improved dataset support, and single inverse depth feature representation.
  Please check out the [release page](https://github.com/rpng/open_vins/releases/tag/v2.0) for details.
* **January 21, 2020** - Our paper has been accepted for presentation in [ICRA 2020](https://www.icra2020.org/). We look
  forward to seeing everybody there! We have also added links to a few videos of the system running on different
  datasets.
* **October 23, 2019** - OpenVINS placed first in the [IROS 2019 FPV Drone Racing VIO Competition
  ](http://rpg.ifi.uzh.ch/uzh-fpv.html). We will be giving a short presentation at
  the [workshop](https://wp.nyu.edu/workshopiros2019mav/) at 12:45pm in Macau on November 8th.
* **October 1, 2019** - We will be presenting at the [Visual-Inertial Navigation: Challenges and Applications
  ](http://udel.edu/~ghuang/iros19-vins-workshop/index.html) workshop at [IROS 2019](https://www.iros2019.org/). The
  submitted workshop paper can be found at [this](http://udel.edu/~ghuang/iros19-vins-workshop/papers/06.pdf) link.
* **August 21, 2019** - Open sourced [ov_maplab](https://github.com/rpng/ov_maplab) for interfacing OpenVINS with
  the [maplab](https://github.com/ethz-asl/maplab) library.
* **August 15, 2019** - Initial release of OpenVINS repository and documentation website!

## Project Features

* Sliding window visual-inertial MSCKF
* Modular covariance type system
* Comprehensive documentation and derivations
* Extendable visual-inertial simulator
    * On manifold SE(3) b-spline
    * Arbitrary number of cameras
    * Arbitrary sensor rate
    * Automatic feature generation
* Five different feature representations
    1. Global XYZ
    2. Global inverse depth
    3. Anchored XYZ
    4. Anchored inverse depth
    5. Anchored MSCKF inverse depth
    6. Anchored single inverse depth
* Calibration of sensor intrinsics and extrinsics
    * Camera to IMU transform
    * Camera to IMU time offset
    * Camera intrinsics
    * Inertial intrinsics (including g-sensitivity)
* Environmental SLAM feature
    * OpenCV ARUCO tag SLAM features
    * Sparse feature SLAM features
* Visual tracking support
    * Monocular camera
    * Stereo camera (synchronized)
    * Binocular cameras (synchronized)
    * KLT or descriptor based
    * Masked tracking
* Static and dynamic state initialization
* Zero velocity detection and updates
* Out of the box evaluation on EuRocMav, TUM-VI, UZH-FPV, KAIST Urban and other VIO datasets
* Extensive evaluation suite (ATE, RPE, NEES, RMSE, etc..)

## Extensions by Joao Leonardo Silva Cotta and ModalAI

This repository extends [OpenVINS](https://github.com/rpng/open_vins) with work by
**Joao Leonardo Silva Cotta, PhD** at [ModalAI](https://www.modalai.com/), from May 2025 (`159cd51`) onward.
The extensions target estimators that run on real vehicles: unsynchronized multi-camera rigs, rolling-shutter sensors, imperfect calibration and limited compute.
Throughout, the filter stays consistent and every change is testable.
The OpenCL tracker was developed together with Kyle T.
Each item below cites its commits.

**Highlights**

* **Consistent asynchronous multi-camera VIO.**
  Every camera frame owns a stochastic pose at its own physical IMU-clock instant, with per-camera time offsets and rolling-shutter readout.
  On the 6-DoF Gaussian NEES check the filter reaches an ANEES of 6.00 (expected value 6), against 241.5 when the exposure noise is omitted.
* **Ceres-free initialization.**
  A purpose-built Levenberg–Marquardt / Schur solver with S²-gravity dynamic initialization, gravity-direction gates and a warm-start covariance.
  Ceres is no longer a dependency.
* **A hardened filter core.**
  Propagation and updates are atomic with rollback, inputs are validated instead of exiting the process, process noise is continuous-time, and numerics stay IEEE-safe under `-ffast-math`.
* **Targetless self-calibration** with [ov_zcalib](ov_zcalib/README.md), and **[zstudio](zstudio/README.md)**, a complete inspection tool for VINS.
* **GPU feature tracking** (TrackOCL). It builds wherever libmodal-flow is available, including on desktop OpenCL through an in-tree harness.
* **Reproducible validation.** Host CI runs about 116 regression tests on every pull request, plus extended VINS and calibration cases.

### Initialization (ov_init)

* **Ceres-free solver, the default** (`2017800`, `91fbfa5`, `23a6744`, `cd62ed3`, `3f8975e`, `8517eeb`).
  `ov_init::zbft_sfm` provides Levenberg–Marquardt with Schur elimination, a lock-free thread pool and covariance recovery.
  `-DOV_INIT_CERES_FREE=OFF` restores the Ceres MLE.
  See [ov_init/src/ceres_free/README.md](ov_init/src/ceres_free/README.md).
* **S²-gravity dynamic initialization** (`91fbfa5`, `8fd30c4`, `64484da`).
  Gravity is estimated on S² with a gauge prior, checked by a post-solve gravity gate with conditional re-alignment, and given a covariance recovered at the optimum.
  This work also corrects the equidistant optical-center derivative in the filter's camera model.
* **Gravity-direction gates** (`bf87f58`, `466aa02`).
  They prevent upside-down or strongly tilted initialization, measured against the IMU's −Z.
  The static initializer has its own tight gate.
* **Warm start** (`91fbfa5`, `144a7cc`).
  The initializer's joint IMU + clone covariance is injected into the filter.
  Observations the initializer already used are removed so they are not counted twice.
* **Soft-reset bias prior** (`ac4c4c3`).
  `VioManager::soft_reset` snapshots the live biases and their marginal sigmas.
  The next dynamic initialization uses them as a gated prior.
* **Asynchronous initialization** (`144a7cc`).
  One owned worker runs on a private IMU copy.
  Its result is imported only if it is still current and passes finiteness, quaternion and PSD-covariance checks.
* **Other additions:**
  * batch IMU feeding (`159cd51`);
  * solver extensions for batch calibration (`b5a1a09`);
  * spectral landmark elimination (`eac8fc5`).

### Multi-camera timing and rolling shutter (ov_msckf)

* **Asynchronous multi-camera ingest inside the library** (`a12a902`, `144a7cc`, `7d1e2db`).
  `AsyncCameraBuffer` gives each camera its own ring buffer and releases frames in physical IMU-clock order, gated on IMU coverage.
  Late frames are dropped, never retimed, and a stalled camera never blocks the others.
  Every front end gets the same behaviour: server, ROS, replay and tests.
* **Per-camera camera–IMU time offsets and rolling-shutter readout** (`2ddd7ed`, `e642930`, `aafa3a7`).
  These are per-camera states with analytic measurement Jacobians.
  The acquisition declarations `camN_shutter`, `camN_readout_time_s` and `camN_fps` are validated at boot.
  The readout is the declared hardware value; estimating it is optional (`calib_cam_readout`).
* **One timestamp convention** (`b8de2f0`, `aa3eeb3`).
  A frame stamp is the center-row mid-exposure instant, applied once by the producer.
  Row *v* is sampled at stamp + (v/h − a)·t_r, where `rs_convention` (`top` | `center` | `bottom`) sets *a*.
  Rolling-shutter and time-offset pose shifts compose exactly on SO(3).
* **Epoch mode: stochastic exposure ownership** (`70cb061`, which supersedes the earlier epoch snapping and preintegration bridge of `31482d9` and `15634ce`).
  Each camera frame gets its own stochastic pose at its physical IMU endpoint: 6-DoF for global shutter, 9-DoF pose + velocity for rolling shutter.
  Raw stamps are never rewritten, and the full process-noise and calibration cross-covariance is kept.
  It is the default for unsynchronized multi-camera rigs.
  See [ov_msckf/README_ASYNC.md](ov_msckf/README_ASYNC.md).
* **Other camera-timing policies** (`144a7cc`):
  * `force_camera_sync` groups hardware-synchronized views;
  * `async_frame_clones` clones each view at its raw frame time;
  * `physical_camera_clones` is for global shutter only.

  `max_clones` now means the per-view track length; the pose capacity is resolved separately.
* **Temporal-calibration gate** (`e642930`, `144a7cc`).
  Under motion too weak to observe them, the time offset and readout are held as Schmidt (consider) states (`dt_calib_gate`).

### Filter consistency and robustness (ov_msckf, ov_core)

* **Propagation** (`144a7cc`, `714e9b3`, `a058ba8`):
  * It targets the IMU-clock endpoint, so a later time-offset update cannot relabel a published pose.
  * It is atomic, rolling back state and FEJ on failure, and never extrapolates IMU data.
  * Continuous-time process noise is integrated by Gauss–Legendre quadrature.
  * The IMU-intrinsics Jacobians and the default D_w/D_a layout (previously singular) are corrected, as are the signs of the right/left Jacobians.
  * The propagation windows are adjustable: `prop_window`, `zupt_prop_window`.
* **Updates** (`144a7cc`):
  * Inputs are validated, and update and propagation return failure instead of exiting the process.
  * The update uses an LLT solve and rolls back the covariance on a non-PSD posterior.
  * Innovation χ² checks are guarded, and delayed initialization uses the correct degrees of freedom plus a rank check.
  * Virtual anchors keep time-offset and rolling-shutter shifts off the anchor pose.
* **ZUPT** (`144a7cc`, `2454f2b`):
  * per-camera disparity;
  * a raw-axis model for non-identity IMU intrinsics;
  * the discarded residual energy is counted in χ²;
  * `get_did_zupt_update()`.
* **Snapshot / restore for replay rewind** (with Kyle T.: `655fa68`, `683777e`, `5e68e6e`, `b5ae2bf`).
* **ov_core numerics** (`144a7cc`):
  * double-precision distortion, and a CamEqui Jacobian series near r = 0;
  * `Type::propose_update` for validated retractions;
  * IEEE bit-level finiteness checks that survive `-ffast-math`;
  * camera-scoped `FeatureDatabase` queries;
  * an Eigen alignment ABI exported to ROS 1/2 consumers.
* **Feature quality and diagnostics** (`cc0ff56`, `cf64b98`, `05a05c3`, `6fdf6fd`).
  Each feature carries a reprojection-inlier quality into SLAM landmarks.
  A `used_features_map` with accessors records the features used in each update.
* **Simulation and validation harnesses** (`a146131`, `be235b7`).
  The simulator models per-camera phase offsets, time offsets and readouts.
  ROS-free RMSE/NEES harnesses, and observation replay through the same ingest path as images.

### Feature tracking: OpenCL tracker (TrackOCL)

* **TrackOCL** — a GPU KLT tracker on libmodal-flow, enabled with `use_gpu: true`.
  Without libmodal-flow, `use_gpu` falls back to TrackKLT. `DISABLE_TRACK_KLT` (`71f7c00`) removes the OpenCV video dependency.
* **Per-camera masks** (`d165fb8`). Masks apply through detection, temporal tracking and stereo. This fixes the right camera previously using the left camera's mask.
* **Builds anywhere.** TrackOCL builds automatically whenever libmodal-flow is available: as an installed library (VOXL), or from its source tree through an in-tree host harness, on any architecture and any OpenCL ≥ 2.0 platform. Set `OV_MODAL_FLOW=AUTO|ON|OFF`; see [docs/opencl-host-harness.md](docs/opencl-host-harness.md).
* **Stereo front end** (`e1fa8f2`, `649a990`, `f20e3ca`, `cd8e5dc`, `a208801`, `739f5fc`):
  * ZNCC epipolar-band matching;
  * stereo-to-mono demotion;
  * IMU-aided KLT seeding, off by default.

  It is compiled in, but stereo tracking is currently disabled ("Stereo Tracking is under R&D", `5e9d267`).

### Build, dependencies and CI

* **Lighter dependencies.** Ceres-free by default, and Boost retired from every built target (`aa3eeb3`).
* **Toolchain.** CMake ≥ 3.5 (`41a14b8`; `0bf82ea`) and a ROS-free superbuild at the top level.
* **Host CI** (`2fd809a`, `e0decdd`, `cbb7e15`).
  It is containerized and cached: about 116 regression tests run on every pull request, and the extended VINS and calibration cases run on master pushes.
  Workflows not relevant to VOXL were removed (`b0703ee`).

### New and changed configuration keys

| Key | Default | Notes |
|---|---|---|
| `prop_window`, `zupt_prop_window` | 0.10 | required |
| `use_gpu` | false | required; selects TrackOCL |
| `async_ring_size`, `async_guard`, `async_stale_factor` | 16, 0.002, 1.5 | asynchronous ingest |
| `epoch_mode` | on for unsynchronized multi-camera rigs | stochastic exposure ownership |
| `max_epoch_clones` | 64 | epoch window bound (sized from `camN_fps`) |
| `force_camera_sync`, `async_frame_clones`, `physical_camera_clones` | false | other camera-timing policies |
| `camN_shutter`, `camN_readout_time_s`, `camN_fps` | unset | per-camera acquisition declarations |
| `timeshift_cam_imu`, `t_readout` (Kalibr imucam) | calib dt, 0 | per-camera time offset and readout |
| `cam_imu_dt_ref_camid` | 0 | reference camera for time offsets |
| `calib_cam_readout`, `calib_cam_readout_init_sigma` | off, 0.001 | optional readout estimation |
| `rs_convention` | center | rolling-shutter row anchor |
| `dt_calib_gate`, `dt_calib_gate_min_omega`, `dt_calib_gate_min_vel_spread` | false, 0.10, 0.10 | temporal-calibration gate |
| `init_gravity_max_angle`, `init_static_gravity_max_angle` | 45, 5 (deg) | gravity-direction gates (vs raw −Z) |
| `init_dyn_grav_prior_sigma` | 0.5 | S² gravity prior |
| `init_warmstart_inject` | false | warm-start covariance injection |

Removed during development: `cam_sync`, `epoch_bind_factor`, `epoch_bridge_bias_cols`.

### Tools

* **[zstudio](zstudio/README.md)** — a complete inspection tool for VINS. It covers:
  * voxl-logger logs and ROS 1 / ROS 2 bags, with no ROS environment needed;
  * data-loss and configuration checks, and camera orientation fixes;
  * real-time multi-camera playback;
  * ROS-free OpenVINS replay with a QGC-style parameter editor;
  * calibration.

  `ov_log_inspector` is its command-line counterpart.
* **[ov_zcalib](ov_zcalib/README.md)** — targetless in-flight visual-inertial self-calibration.
* **`ov_msckf_replay`** — OpenVINS over a ROS-free sensor stream (`ov_core/src/utils/sensor_stream.h`), used by zstudio.

## Building with all dependencies

```bash
# System / C++ dependencies (ROS workspaces): rosdep resolves every package.xml
rosdep install --from-paths . --ignore-src -y

# ROS-free superbuild with zstudio; install also resolves zstudio's Python requirements
cmake -S . -B build -DENABLE_ROS=OFF -DOV_BUILD_CALIB=ON \
  -DOV_BUILD_ZSTUDIO=ON -DZSTUDIO_STANDALONE=ON -DOV_ZCALIB_BUILD_TESTS=OFF
cmake --build build -j"$(nproc)"
cmake --install build --prefix "$HOME/.local"
zstudio                                   # or: zstudio --open /path/log0005
```

* **Python** — [`zstudio/requirements.txt`](zstudio/requirements.txt)
  is resolved during install (`cmake --install`, `colcon build`,
  `catkin_make install`): requirements already importable are kept, missing ones
  are pip-installed into a private directory next to zstudio, never into the
  system Python. `-DZSTUDIO_INSTALL_PYTHON_DEPS=OFF` disables it.
* **OpenCL / libmodal-flow** — `OV_MODAL_FLOW=AUTO` (default) detects it,
  `ON` requires it, `OFF` skips it; `OV_MODAL_FLOW_SOURCE_DIR` names a
  specific source tree.

## Codebase Extensions

* **[ov_plane](https://github.com/rpng/ov_plane)** - A real-time monocular visual-inertial odometry (VIO) system which leverages
  environmental planes. At the core it presents an efficient robust monocular-based plane detection algorithm which does
  not require additional sensing modalities such as a stereo, depth camera or neural network. The plane detection and tracking
  algorithm enables real-time regularization of point features to environmental planes which are either maintained in the state
  vector as long-lived planes, or marginalized for efficiency. Planar regularities are applied to both in-state SLAM and
  out-of-state MSCKF point features, enabling long-term point-to-plane loop-closures due to the large spacial volume of planes.

* **[vicon2gt](https://github.com/rpng/vicon2gt)** - This utility was created to generate groundtruth trajectories using
  a motion capture system (e.g. Vicon or OptiTrack) for use in evaluating visual-inertial estimation systems.
  Specifically we calculate the inertial IMU state (full 15 dof) at camera frequency rate and generate a groundtruth
  trajectory similar to those provided by the EurocMav datasets. Performs fusion of inertial and motion capture
  information and estimates all unknown spacial-temporal calibrations between the two sensors.

* **[ov_maplab](https://github.com/rpng/ov_maplab)** - This codebase contains the interface wrapper for exporting
  visual-inertial runs from [OpenVINS](https://github.com/rpng/open_vins) into the ViMap structure taken
  by [maplab](https://github.com/ethz-asl/maplab). The state estimates and raw images are appended to the ViMap as
  OpenVINS runs through a dataset. After completion of the dataset, features are re-extract and triangulate with
  maplab's feature system. This can be used to merge multi-session maps, or to perform a batch optimization after first
  running the data through OpenVINS. Some example have been provided along with a helper script to export trajectories
  into the standard groundtruth format.

* **[ov_secondary](https://github.com/rpng/ov_secondary)** - This is an example secondary thread which provides loop
  closure in a loosely coupled manner for [OpenVINS](https://github.com/rpng/open_vins). This is a modification of the
  code originally developed by the HKUST aerial robotics group and can be found in
  their [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) repository. Here we stress that this is a
  loosely coupled method, thus no information is returned to the estimator to improve the underlying OpenVINS odometry.
  This codebase has been modified in a few key areas including: exposing more loop closure parameters, subscribing to
  camera intrinsics, simplifying configuration such that only topics need to be supplied, and some tweaks to the loop
  closure detection to improve frequency.


## Demo Videos

<a href="http://www.youtube.com/watch?v=KCX51GvYGss">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/KCX51GvYGss.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=Lc7VQHngSuQ">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/Lc7VQHngSuQ.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=vaia7iPaRW8">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/vaia7iPaRW8.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=MCzTF9ye2zw">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/MCzTF9ye2zw.jpg"  width="120" height="90"/>
</a>
<a href="http://www.youtube.com/watch?v=eSQLWcNrx_I">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/eSQLWcNrx_I.jpg" width="120" height="90" />
</a>
<br/>

<a href="http://www.youtube.com/watch?v=187AXuuGNNw">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/187AXuuGNNw.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=oUoLlrFryk0">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/oUoLlrFryk0.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=ExPIGwORm4E">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/ExPIGwORm4E.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=lXHl-qgLGl8">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/lXHl-qgLGl8.jpg" width="120" height="90" />
</a>



## Credit / Licensing

This code was written by the [Robot Perception and Navigation Group (RPNG)](https://sites.udel.edu/robot/) at the
University of Delaware. If you have any issues with the code please open an issue on our github page with relevant
implementation details and references. For researchers that have leveraged or compared to this work, please cite the
following:

```txt
@Conference{Geneva2020ICRA,
  Title      = {{OpenVINS}: A Research Platform for Visual-Inertial Estimation},
  Author     = {Patrick Geneva and Kevin Eckenhoff and Woosik Lee and Yulin Yang and Guoquan Huang},
  Booktitle  = {Proc. of the IEEE International Conference on Robotics and Automation},
  Year       = {2020},
  Address    = {Paris, France},
  Url        = {\url{https://github.com/rpng/open_vins}}
}
```

The extensions described in
[Extensions by Joao Leonardo Silva Cotta and ModalAI](#extensions-by-joao-leonardo-silva-cotta-and-modalai) — the
estimator work, ov_zcalib and zstudio — are by **Joao Leonardo Silva Cotta, PhD**, developed at [ModalAI](https://www.modalai.com/).
With thanks to Alex K., Donald H., Reginald W., James S. and Kyle T.

The codebase and documentation is licensed under the [GNU General Public License v3 (GPL-3)](https://www.gnu.org/licenses/gpl-3.0.txt).
You must preserve the copyright and license notices in your derivative work and make available the complete source code with modifications under the same license ([see this](https://choosealicense.com/licenses/gpl-3.0/); this is not legal advice).


