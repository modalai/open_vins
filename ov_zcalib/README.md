# ov_zcalib — targetless in-flight visual-inertial calibrator

On-demand, targetless calibration for VOXL platforms: camera-IMU extrinsics
(`R_ItoC`, `p_IinC`), time offset `td`, IMU intrinsics (imu2 model:
upper-triangular inverse `Dw`/`Da` + `R_AtoI`, and g-sensitivity `Tg` behind
its own falsifier), and staged camera-intrinsic refinement. Ceres-free
(built entirely on the `ov_init::zbft_sfm` solver stack), zero `ov_msckf`
coupling, never co-resident with the filter: the calibrator is a separate
session whose committed YAML the filter consumes on its next start (or
hot-start).

Two front doors, one engine:

- **On device**: `voxl-open-vins-server --calibrate` feeds this module
  in-process from the server's own MPA drivers with the TrackOCL front-end
  (the only tracker on target), and writes the session record mirror at
  consumption (`record_session` profile key, ON by default; an unrecorded
  session cannot be replayed).
- **On host**: the `ov_zcalibrate` CLI replays recorded sessions
  (`--replay`, the CI and forensics path), converts voxl-logger logs
  (`--voxl`), and runs a writeback smoke test (`--selftest`).

## Layout

| dir | content |
|---|---|
| `src/core/` | session runner (SETTLE -> BOOTSTRAP -> COLLECT -> SOLVE -> VERIFY -> COMMIT), profiles, config schema |
| `src/cpi/` | ACI^2 calibration preintegration (mean + 15x15 covariance + intrinsic columns) + value-keyed cache |
| `src/init/` | bootstrap: relative-rotation model family, xcorr td seed, Markley-SVD hand-eye |
| `src/solve/` | per-window micro-BA + reduced-information export, cross-window VarPro fusion (JointCalib), factors |
| `src/types/` | shared calibration state, imu2 intrinsic model, kalibr chain conversion |
| `src/utils/` | session record (kFormat 5), lock-free SPSC rings, voxl-logger feeder, atomic YAML writeback |
| `src/window/` | excitation-gated harvester, diversity reservoir + D-optimal selection, arrowhead linear seeder |
| `src/sim/` | synthetic targetless world shared by the tests |

## Conventions (the firewall)

Everything is OpenVINS/JPL: `{}^I_G R` global-to-IMU, `R_ItoC` maps
IMU-frame vectors into the camera frame, left-quaternion error, gravity
`+Z`, inverse intrinsics applied to raw-frame-biased measurements
(`a_hat = R_AtoI Da (a_m - b_a)`, `w_hat = Dw (w_m - b_g - Tg a_hat)`), and
`t_imu = t_cam + td`. On top of that, the system-wide timestamp convention:

- Every frame stamp is the **center-row mid-exposure** instant,
  `t = SOF + (readout + exposure)/2`, applied once by the producer (server
  ingest, log feeder, sim generators) and never re-corrected downstream.
- The rolling-shutter model is **centered**: row `v` samples at
  `stamp + (v/h - 0.5) * readout`.
- Every RS/dt kinematic pose shift composes exactly on SO(3) (`exp_so3`);
  Jacobians stay first-order by the documented contract.
- `tr` (readout) is **never estimated**: it is the HAL3
  `ANDROID_SENSOR_ROLLING_SHUTTER_SKEW` of the streamed sensor mode, a fixed
  transport constant folded into each observation at factor construction.
  `--tr <s>` on the CLI *is* that value; 0 means global shutter.
- `td` is defined against center stamps and consumed verbatim by the chain,
  VIO, and the hot-start path.
- Session records are **kFormat 5**; older-format records refuse loudly —
  re-record rather than guess at stamp semantics.

## Statistical honesty (the gates)

- Blind protocol on real rigs: per-unit facts (camera intrinsics, IMU noise)
  are seeded; extrinsics and `td` are earned — the platform chain is a
  printed reference, never a seed. The calibration weighting is the raw
  Allan fit of the part, never the filter's inflated chain values.
- Split-half falsifier with per-dof bands for the accel chain; a Wald gate
  with per-session df-corrected thresholds (MC-calibrated,
  `zcalib_test_wald_mc`; junk-injection power study behind `--h1`).
- Tg recovery arbiter: split-tg falsifier with retry arbitration and a
  commit-walk revert arm — a refused `Tg` can never ship its solved value
  silently (`zcalib_test_tg_e2e`).
- Commit rules: per-block 3-sigma with ceilings, atomic block pairing,
  committed-mixture re-verify, holdout VERIFY floor, camera-center
  quadrant-coverage gate. A block that cannot be earned ships its seed and
  the report says so.
- Export-on-accept: rejected-pass and duel-loser exports are dead state;
  the export runs once at the accepted optimum (ON == OFF byte parity,
  `zcalib_test_export_parity`). Export-failure doctrine: candidate-point
  failure VETOES the candidate (the fused set never silently shrinks);
  entry-point failure marks the window dead. Spectral landmark elimination
  (relative rank cutoff) keeps degenerate landmark directions from
  poisoning the reduced nuisance Hessian.

## Determinism doctrine

No wall-clock condition inside an iterate (wall budgets stop between passes
at accepted points; the inner hang guard is the only timer and self-reports
through the evidence table). Fixed worker ranges keep serial == parallel
bit-identical. The invariant is within-binary replay determinism —
cross-binary byte equality is not attainable under `-ffast-math`.

## Usage (host)

```
ov_zcalibrate --replay session.bin --out cal.yaml \
              --tr 0.007777778 --solve-budget-s 60      # faithful replay
ov_zcalibrate --voxl /path/log0001 --flight             # voxl-logger log
ov_zcalibrate --selftest                                # writeback smoke
```

- A device session's record mirror replays bit-identically on the same
  binary; cross-platform replays gate on value tolerances plus identical
  gate/commit decisions.
- Records do not carry `--tr` or the solve budget: pass the device values
  (`--tr <readout>` for rolling cameras, `--solve-budget-s <profile value>`)
  or the replay truncates where the live session did not.
- `--flight` applies the flight profile; `--threads N` changes wall clock
  only (the result YAML is byte-identical); the seeding-study knobs
  (`--imu-seed`, `--seed-rot/-pos/-td`, `--tg`, `--noise`) let one recorded
  session be scored across the whole seeding matrix.

## Tests (ctest, `OV_ZCALIB_BUILD_TESTS=ON`)

| target | pins |
|---|---|
| `zcalib_test_aci3_fd` | every preintegration mean/factor column vs finite differences |
| `zcalib_test_preint_chain` | chronological chain sweep + cache == recompute |
| `zcalib_test_convention_ports` | sign/frame/ordering firewall gates |
| `zcalib_test_frontend` | seeding, harvest, reservoir, selection + record round-trip |
| `zcalib_test_calib_e2e` | truth-seeded recovery + Fisher-spectrum falsifier |
| `zcalib_test_session_e2e` | no-truth production path; world blocks selectable by argv; byte-identical replays |
| `zcalib_test_tg_e2e` | Tg earn/refuse/recovery oracle |
| `zcalib_test_wald_mc` | MC calibration of the Wald thresholds; `--h1` power study |
| `zcalib_test_export_parity` | export-on-accept ON == OFF byte parity + veto-path fault injection |

On device builds (`DISABLE_TRACK_KLT`), the library and `ov_zcalibrate`
(`--replay` mode) still build; `--voxl` log ingest and the tests are
host-only.
