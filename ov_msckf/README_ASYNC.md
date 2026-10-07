# Stochastic epoch multi-camera MSCKF

Epoch mode retains a camera-owned stochastic exposure view at each physical camera
endpoint. It replaces reference-frame snapping and the bridge transport noise
omission described in the historical Jabba document. There is one epoch policy;
the snapping path, bridge caches, binding options, and deferred epoch retirement
have been removed.

## Exposure and clock ownership

OpenVINS uses JPL rotations mapping global coordinates into the IMU frame. For
camera $i$, a frame with immutable raw key $t_{f,i}$ has nominal stamp-row endpoint
$t_i=t_{f,i}+t_{d,i}$. A pixel on row $m$ of an image of height $M$ uses

```math
t_{i,m}=t_i+t_{r,i}(m/M-a),
```

where `rs_convention` selects $a=0$, $0.5$, or $1$ for top, center, or bottom.
The producer and estimator must use the same timestamp convention. A global
shutter has $t_{r,i}=0$. Declared global-shutter cameras have no estimated readout
coordinate.

`epoch_mode` selects the stochastic exposure policy after option resolution.
Asynchronous multi-camera configurations select it automatically unless explicitly
overridden or another camera policy is selected. `force_camera_sync` remains the
separate hardware synchronization policy. Stereo tracking is forced off, with
`Stereo Tracking is under R&D, release coming soon` when requested.

The ingest merge releases frames in physical IMU order using each camera's clock.
Equal nominal endpoints share one navigation propagation but retain separate
camera views because their clock errors can differ. Raw observation keys never
snap or change after insertion. Once propagation succeeds, `State::imu_endpoint()`
is the authoritative navigation time; a later clock correction cannot relabel it.
The reference camera supplies only the compatibility timestamp label.

## Covariance contract

Ordinary IMU propagation retains the complete navigation process covariance and
all existing calibration, historical exposure, and navigation cross blocks:

```math
P^- = F P F^\top + Q.
```

New process noise enters the propagated navigation block once. The camera view
is then augmented with the owner camera's clock derivative:

```math
A_i=E_{\theta,p}+[\omega_i;v_i]e_{t_{d,i}}^\top \quad\text{(GS)},
\qquad
A_i=E_{\theta,p,v}+[\omega_i;v_i;a_i]e_{t_{d,i}}^\top \quad\text{(RS)},
```

```math
P_{\mathrm{aug}}=
\begin{bmatrix}
P^- & P^-A_i^\top \\
A_iP^- & A_iP^-A_i^\top
\end{bmatrix}.
```

A fixed clock contributes no stochastic clock column. RS velocity is an ordinary
covariance-owned `Vec` with an immutable FEJ value; cached velocity metadata is
not its stochastic substitute. RS pose and velocity are augmented together with
one covariance growth. Every feature observing that frame reads the same owner.
The joint-state innovation $S=H P_{\mathrm{aug}} H^\top+R_{\mathrm{pixel}}$
therefore retains the shared transport error. Visual factors do not add duplicate
clock or bridge-bias columns: their correlation enters through the owned exposure
and propagated covariance.

The former 0.33% single-pixel noise bound did not bound joint information. For 200
observations with the same transport direction, that contribution can be about
66% relative to independent pixel noise in the common direction. Diagonal pixel
inflation would discard the cross-feature correlation. Covariance ownership
handles it without constructing a dense cross-feature noise matrix.

Owners survive until their actual window retirement, including delayed factors.
Landmarks are reanchored and camera-specific observations are cleaned before
retirement. Removing pose and RS velocity takes the kept principal covariance
submatrix; it adds no noise and conditions on no removed coordinate. Snapshots
rewire both types so branch updates cannot modify the live state.

## Rolling shutter model and mathematical scope

Between the stamp row and an observed row, the existing mean model uses constant
corrected angular rate and world velocity:

```math
R_{i,m}=\operatorname{Exp}(-\omega_i\tau_m)R_i,\qquad
p_{i,m}=p_i+v_i\tau_m,\qquad \tau_m=t_{r,i}(m/M-a).
```

Projection, triangulation, and delayed initialization use the same row geometry.
The observing pose derivative transports the clone's perturbation through the
SO(3) row rotation. The velocity derivative is $-\tau_m J_\pi R_{I\to C}R_{i,m}$.
FEJ reads the frozen exposure pose and velocity, preserving global translation
and gravity-yaw gauge directions. A delayed initialization batch captures row
motion once; earlier accepted landmarks must not change a later retry's seed.

This covariance contract fixes inter-frame transport within the existing
continuous-noise IMU and first-order EKF model. Constant-kinematics RS rows remain
an approximation. Complete raw-sample noise ownership, angular-rate uncertainty,
and conditional process noise within the readout are not enabled here. Neither
this repair nor a passing time-average NEES gate certifies an exact nonlinear RS
estimator. Calibrated readout and stamp-row conventions remain required inputs.

## Bounded resources

`max_clones` keeps its configured feature graduation length $C$. With all nominal
camera rates declared, the stochastic owner capacity is

```math
K=\left\lceil C\frac{\sum_i f_i}{\min_i f_i}\right\rceil.
```

For 60 Hz RS and 30 Hz GS with $C=11$, this is 33 pose owners, with velocity only
for RS owners. Undeclared rates use $K=C N_{\mathrm{cameras}}$. Both policies fail
closed above `max_epoch_clones` (default 64); invalid rates are rejected. Covariance
storage remains quadratic in the bounded state dimension. There is no unbounded
per-feature transport registry.

Epoch exposure lookup costs $O(\log K)$ rather than a window scan per observation.
RS triangulation copies its pose map once per update batch and resets only each
track's observed keys: $O(K+\sum\mathrm{observations})$ map preparation instead of
$O(\mathrm{features}\times K)$ full-map copies. Camera rings, IMU coverage guards,
and stream liveness bounds retain their existing contracts. These bounds and host
timings do not establish QRB/QCS deadline performance.

`epoch_bind_factor` and `epoch_bridge_bias_cols` are obsolete. Current epoch mode
has no binding horizon or omission escape hatch. `dt_calib_gate` uses Schmidt
mean-gain freezing while retaining temporal Jacobians and cross-covariance.

## Reproducible validation

Build host tests with ROS disabled and `OV_MSCKF_BUILD_TESTS=ON`.

`.github/workflows/host-math.yml` runs the fast host tier on pull requests to
`master`: core, initialization, estimator and calibration numerical tests plus
one complete 60 Hz RS / 30 Hz GS trajectory with its original accuracy and NEES
gates. Full calibration session E2E, calibration Monte Carlo and the remaining
VINS trajectories carry the CTest `extended` label. They run in a separate job
after pushes to `master`, or through manual dispatch with `extended` enabled.
The complete suite and its acceptance thresholds are retained.

The dependency environment is a cached Docker image keyed by its Dockerfile,
with a pinned Ubuntu base. Package installation runs only when that image is
absent. A separate ccache stores compiled objects across source revisions;
its key includes the dependency image and native CPU fingerprint. Both tiers
use production Release math flags, CPU trackers, the Ceres-free initializer
and all available runner CPUs. Full calibration sessions run one at a time
because each already owns a parallel solver pool. PR commits trigger one run
instead of duplicate branch-push and PR runs. Failures stop the job; JUnit and
CTest diagnostics are retained. All execution stays on the host.

The same host build can be run locally:

```sh
cmake -S . -B build-host -DCMAKE_BUILD_TYPE=Release -DENABLE_ROS=OFF \
  -DOV_INIT_CERES_FREE=ON -DBUILD_OV_EVAL=OFF -DDISABLE_MATPLOTLIB=ON \
  -DOV_CORE_BUILD_TESTS=ON -DOV_INIT_BUILD_TESTS=ON -DOV_MSCKF_BUILD_TESTS=ON \
  -DOV_BUILD_CALIB=ON -DOV_ZCALIB_BUILD_TESTS=ON
cmake --build build-host --target ov_host_tests --parallel "$(nproc)"
ctest --test-dir build-host --output-on-failure --parallel "$(nproc)"
```

For just the PR tier, build `ov_host_tests_fast` and pass `-LE extended` to
CTest. To run the extended tier separately, build `ov_host_tests` and pass
`-L extended`. A plain CTest invocation still runs every registered test.

- `test_epoch_exposure_jacobian`: independent double-precision projection finite
  differences, pose/velocity/readout columns, and FEJ translation/yaw nullspaces.
- `test_epoch_velocity_owner`: independent dense covariance oracle covering
  augmentation, propagation, late updates, snapshots, and true marginalization;
  5,000 independent six-dimensional Gaussian NEES draws; omitted shared transport
  noise as a negative control.
- `test_async_dual_epoch_mixed_rs_gs_{6,30,40}`: 60 Hz RS / 30 Hz GS, 12 ms
  calibrated readout, center-row stamps, independent feature noise, and the
  original 0.6 m / 1 degree / 25 time-average NEES diagnostic gates.
- Existing initialization, warm/reset prior, physical ownership, rolling-shutter
  retry, numeric rejection, ingest, synchronization, and global-shutter tests.

The mixed fixture generates independent 60 Hz streams and retains every second
GS frame. Simulator noise draws and retained pixels stay identical across camera
policies. `test_async_dual --mixed-rs-gs` exposes the fixture without external
source patches. It scores navigation at the accepted IMU endpoint. Its truth
initialization and lack of image tracking limit what it validates. The explicitly
named unmodeled-timing negative control must fail its accuracy gates.

Independent linear-model NEES checks establish that covariance ownership matches
that Gaussian model. Small-seed, correlated time-average VINS NEES is diagnostic;
full nonlinear Monte Carlo coverage needs a separate study. Recorded host replay
without ground-truth poses checks initialization and numerical/resource behavior,
not RMSE or NEES. `ov_zcalib` uses `ov_core` and `ov_init` directly; epoch changes
introduce no calibration solver, likelihood, or Jacobian change.
