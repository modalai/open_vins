"""Generate an OpenVINS configuration (estimator + Kalibr chains) for a log.

A template preset (one of the repository's ``config/<name>/`` directories) is
copied into a fresh output directory and specialised for the selected log
channels. The estimator and IMU files are edited in place with ``params.py``,
so the template's comments survive and the Studio's parameter editor shows
every derived value as "modified" relative to the template. The camera chain
is device-specific, so it is rebuilt from the log.

Template search order (first hit per preset name wins):

1. ``$OV_CONFIG_DIR`` (a directory of presets),
2. the source checkout: the nearest ``config/`` above this file that contains
   ``*/estimator_config.yaml``,
3. installed packages: ``<prefix>/share/ov_msckf/config`` for every prefix in
   ``AMENT_PREFIX_PATH`` / ``CMAKE_PREFIX_PATH`` (ov_msckf installs its configs
   there for ROS 2), and ``config/`` under ``ROS_PACKAGE_PATH`` entries,
4. the studio's own install: ``<prefix>/share/zstudio/openvins_config``.

Default template: ``voxl_sim`` for voxl-logger logs -- it carries the shipped
VOXL production estimator set (AR0144 fisheye, 1 kHz IMU) -- and ``euroc_mav``
(stereo global-shutter VIO, the canonical OpenVINS tuning) for everything else.

Extrinsics are written as ``T_imu_cam`` (camera frame -> IMU frame,
p_I = R p_C + t). That is the key ``VioManagerOptions`` requests; OpenVINS only
falls back to (and inverts) ``T_cam_imu`` when it is missing, so writing
``T_imu_cam`` avoids the inversion path entirely. Template camera blocks are
discarded rather than translated, because they describe another device.

Stamps streamed from logs are already center-row mid-exposure (see
``ov_core::CameraData``), so ``timeshift_cam_imu`` starts at 0.
"""

from datetime import datetime
import math
import os
from pathlib import Path

import numpy as np

from .geometry import angle_between_deg, find_rest
from .params import ParamFile, ParamSet

ESTIMATOR = "estimator_config.yaml"
GLOBAL_TEMPLATE = "euroc_mav"
VOXL_TEMPLATE = "voxl_sim"
_OUTPUT_KEYS = ("record_timing_filepath", "filepath_est", "filepath_std", "filepath_gt")
_IMU_INTRINSICS = {
    "imu0.Tw": np.eye(3),
    "imu0.Ta": np.eye(3),
    "imu0.R_IMUtoGYRO": np.eye(3),
    "imu0.R_IMUtoACC": np.eye(3),
    "imu0.Tg": np.zeros((3, 3)),
}


# Options this OpenVINS build parses strictly (like ros1_serial_msckf) that older presets
# predate. Values are the code defaults (VioManagerOptions.h / InertialInitializerOptions.h).
REQUIRED_KEYS = (
    (
        "init_gravity_max_angle",
        45.0,
        "max angle [deg] between measured gravity and the IMU +z axis to allow initialization",
        "init_imu_thresh",
    ),
    ("prop_window", 0.10, "IMU propagation window [s]", "integration"),
    (
        "zupt_prop_window",
        0.10,
        "IMU window [s] used by the zero-velocity update",
        "zupt_max_velocity",
    ),
    (
        "use_gpu",
        False,
        "TrackOCL (libmodal-flow) when this build has it; CPU TrackKLT otherwise",
        "use_klt",
    ),
)


def config_roots():
    """Directories holding preset subdirectories, in search order."""
    roots = []
    env = os.environ.get("OV_CONFIG_DIR")
    if env:
        roots.append(Path(env).expanduser())
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config"
        if candidate.is_dir() and any(candidate.glob(f"*/{ESTIMATOR}")):
            roots.append(candidate)
            break
    for variable in ("AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
        for prefix in os.environ.get(variable, "").split(os.pathsep):
            if prefix:
                roots.append(Path(prefix) / "share" / "ov_msckf" / "config")
                roots.append(
                    Path(prefix) / "share" / "zstudio" / "openvins_config"
                )
    for entry in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep):
        if entry:
            roots.append(Path(entry) / "config")
    # The studio installs the presets beside itself: <prefix>/share/zstudio/openvins_config,
    # with the module under <prefix>/lib/zstudio/ (standalone) or <prefix>/lib/pythonX/site-packages/.
    for prefix in list(Path(__file__).resolve().parents)[2:5]:
        roots.append(prefix / "share" / "zstudio" / "openvins_config")
    unique = []
    for root in roots:
        if root.is_dir() and root.resolve() not in [u.resolve() for u in unique]:
            unique.append(root)
    return unique


def _templates():
    found = {}
    for root in config_roots():
        for estimator in sorted(root.glob(f"*/{ESTIMATOR}")):
            found.setdefault(estimator.parent.name, estimator.parent)
    return found


def list_templates():
    """Available preset names (e.g. euroc_mav, rs_d455, tum_vi, voxl_sim)."""
    return sorted(_templates())


def template_path(name):
    """Directory of a preset name, or of an explicit directory path."""
    path = Path(str(name)).expanduser()
    if path.is_dir() and (path / ESTIMATOR).is_file():
        return path
    templates = _templates()
    if name not in templates:
        raise ValueError(
            f"unknown OpenVINS config template {name!r}; available: {', '.join(sorted(templates)) or 'none found'}"
        )
    return templates[name]


def default_template(log=None):
    names = list_templates()
    preferred = VOXL_TEMPLATE if getattr(log, "kind", "") == "voxl" else GLOBAL_TEMPLATE
    for name in (preferred, GLOBAL_TEMPLATE):
        if name in names:
            return name
    if not names:
        raise ValueError(
            "no OpenVINS config templates found; set OV_CONFIG_DIR to a directory of presets"
        )
    return names[0]


def _rate(times):
    dt = np.diff(np.asarray(times, dtype=float))
    dt = dt[dt > 0]
    return 1.0 / float(np.median(dt)) if len(dt) else 0.0


def _skew(a, b):
    """Median |stamp difference| [s] between nearest frames of two cameras."""
    if len(a.times) < 2 or len(b.times) < 2:
        return math.inf
    j = np.clip(np.searchsorted(b.times, a.times), 1, len(b.times) - 1)
    nearest = np.where(
        np.abs(b.times[j - 1] - a.times) < np.abs(b.times[j] - a.times), j - 1, j
    )
    return float(np.median(np.abs(b.stamps_ns[nearest] - a.stamps_ns))) * 1e-9


def _f(value):
    value = float(value)
    return (
        "0.0" if abs(value) < 1e-12 else repr(value)
    )  # no 4.3e-17 trigonometric residue


def _row(values):
    return "[" + ", ".join(_f(v) for v in values) + "]"


def _camera_block(index, cam, T, overlaps):
    intr = cam.intrinsics
    d = [float(v) for v in intr["d"]][:4]
    d += [0.0] * (4 - len(d))
    model = "equidistant" if intr["model"] == "equidistant" else "radtan"
    lines = [f"cam{index}:", "  T_imu_cam:"]
    lines += [f"    - {_row(T[r])}" for r in range(4)]
    lines += [
        f"  cam_overlaps: [{', '.join(str(o) for o in overlaps)}]",
        "  camera_model: pinhole",
        f"  distortion_coeffs: {_row(d)}",
        f"  distortion_model: {model}",
        f"  intrinsics: {_row(intr['k'])}",
        f"  resolution: [{int(cam.width)}, {int(cam.height)}]",
        f"  rostopic: {cam.name}",
        "  timeshift_cam_imu: 0.0",
    ]
    return lines


def generate(
    log,
    cameras,
    imu,
    out_dir,
    template=None,
    overrides=None,
    overwrite=False,
    template_cameras=False,
):
    """Write estimator_config.yaml + Kalibr chains for ``cameras``/``imu`` of ``log``.

    Returns ``(Path to estimator_config.yaml, notes)``; notes explain every
    derived value and everything the user should still check. ``overrides``
    maps ParamSet display keys (``max_clones``, ``imu0.update_rate``,
    ``cam0.intrinsics`` ...) to values applied last.

    ``template_cameras=True`` keeps the template's camera + IMU calibration
    (intrinsics, T_imu_cam, IMU intrinsics) for logs recorded by the device the
    preset describes -- e.g. a EuRoC bag with ``euroc_mav`` -- or logs without
    CameraInfo. Only topics are rewritten; resolutions are checked.
    """
    cameras = list(cameras)
    if not cameras:
        raise ValueError("select at least one camera")
    if imu is None or len(imu) < 2:
        raise ValueError("select an IMU channel with data")
    for cam in cameras:
        if template_cameras:
            continue
        if not cam.intrinsics:
            source = (
                cam.intrinsics_expected or "a CameraInfo topic in the image namespace"
            )
            why = f" ({cam.intrinsics_problem})" if cam.intrinsics_problem else ""
            raise ValueError(f"{cam.name} has no intrinsics{why}; expected {source}")
        if not cam.width or not cam.height:
            raise ValueError(f"{cam.name}: image size is unknown")
        iw, ih = cam.intrinsics.get("width") or 0, cam.intrinsics.get("height") or 0
        if iw and ih and (iw, ih) != (cam.width, cam.height):
            raise ValueError(
                f"{cam.name}: intrinsics are for {iw}x{ih} but frames are {cam.width}x{cam.height}"
            )
    name = template or default_template(log)
    source = template_path(name)
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if (out / ESTIMATOR).exists() and not overwrite:
        raise FileExistsError(f"{out / ESTIMATOR} exists; choose a new directory")
    notes = [f"template: {name} ({source})"]

    template_est = ParamFile.load(source / ESTIMATOR)
    imu_name = template_est.value("relative_config_imu") or "kalibr_imu_chain.yaml"
    cam_name = (
        template_est.value("relative_config_imucam") or "kalibr_imucam_chain.yaml"
    )
    for key, target in (
        ("relative_config_imu", imu_name),
        ("relative_config_imucam", cam_name),
    ):
        if Path(target).name != target:
            raise ValueError(
                f"template {key} must name a file beside the estimator config, got {target!r}"
            )
    (out / ESTIMATOR).write_bytes((source / ESTIMATOR).read_bytes())
    if not (source / imu_name).is_file():
        raise ValueError(f"template {name} lacks {imu_name}")
    (out / imu_name).write_bytes((source / imu_name).read_bytes())

    # -- estimator --------------------------------------------------------
    est = ParamFile.load(out / ESTIMATOR, reference=template_est)
    n = len(cameras)
    est.set("max_cameras", n)
    stereo = False
    if n == 2:
        skew = _skew(cameras[0], cameras[1])
        stereo = skew < 1e-3
        notes.append(
            f"use_stereo: {'true' if stereo else 'false'} ({cameras[0].name}/{cameras[1].name} median stamp skew {skew * 1e3:.3f} ms"
            + (
                "; synchronized pair)"
                if stereo
                else "; not synchronized, cameras are tracked independently)"
            )
        )
    if "use_stereo" in est:
        est.set("use_stereo", stereo)
    rates = [_rate(c.times) for c in cameras]
    if "track_frequency" in est and max(rates) > 0:
        current = float(est.value("track_frequency"))
        if current < max(rates) * 1.02:
            wanted = round(max(rates) * 1.05, 1)
            est.set("track_frequency", wanted)
            notes.append(
                f"track_frequency: {current:g} -> {wanted:g} Hz so no {max(rates):.2f} Hz frame is dropped before tracking"
            )
    if est.value("use_mask") is True:
        est.set("use_mask", False)
        notes.append(
            "use_mask: disabled; the template's masks belong to another camera"
        )
    moved = []
    for key in _OUTPUT_KEYS:
        if key in est and isinstance(est.value(key), str):
            est.set(key, str(out / Path(est.value(key)).name))
            moved.append(key)
    if moved:
        notes.append(f"output files ({', '.join(moved)}) now go to {out}")
    if name == VOXL_TEMPLATE and est.value("zupt_max_disparity") == 0:
        # voxl_sim documents this as a simulation-only deviation; real logs boot static.
        est.set("zupt_max_disparity", 8.0)
        notes.append(
            "zupt_max_disparity: 0 -> 8 (voxl_sim's simulation deviation reverted to the production value)"
        )
    for key, value, comment, after in REQUIRED_KEYS:
        if key not in est:
            est.add(
                key,
                value,
                comment + " (added, required by this OpenVINS build)",
                after=after if after in est else None,
            )
            notes.append(
                f"{key}: missing from the template, added with the code default {value}"
            )
    # Both initializers gate on the raw accelerometer direction at rest versus raw -Z
    # (VOXL's z-down IMU; StaticInitializer / DynamicInitializer). Other mountings need
    # wider gates, which also weakens their upside-down protection -- say so loudly.
    rest = find_rest(imu.data)
    gravity, where = (
        (rest[2], f"at rest ({rest[0]:.2f} s)") if rest is not None else (None, "")
    )
    if gravity is None:
        first = imu.data[imu.times <= imu.times[0] + 2.0, 4:7].mean(axis=0)
        if abs(np.linalg.norm(first) - 9.81) < 1.5:
            gravity, where = first, "averaged over the first 2 s (no stationary period)"
    if gravity is not None and "init_gravity_max_angle" in est:
        tilt = angle_between_deg(gravity, [0.0, 0.0, -1.0])
        gate = float(est.value("init_gravity_max_angle"))
        if tilt + 10.0 > gate:
            wanted = float(min(179.0, math.ceil(tilt + 15.0)))
            est.set("init_gravity_max_angle", wanted)
            notes.append(
                f"init_gravity_max_angle: {gate:g} -> {wanted:g} deg; the accelerometer {where} points {tilt:.1f} deg "
                "from the raw -Z the initializers expect (VOXL z-down IMU). This IMU is mounted differently, so the "
                "upside-down protection is weakened accordingly"
            )
        static_gate = (
            float(est.value("init_static_gravity_max_angle"))
            if "init_static_gravity_max_angle" in est
            else 5.0
        )
        if rest is not None and tilt > static_gate - 1.0 and static_gate >= 0:
            wanted = float(min(179.0, math.ceil(tilt + 5.0)))
            if "init_static_gravity_max_angle" in est:
                est.set("init_static_gravity_max_angle", wanted)
            else:
                est.add(
                    "init_static_gravity_max_angle",
                    wanted,
                    "static-init gravity gate [deg] versus raw -Z (generated for this IMU mounting)",
                    after="init_gravity_max_angle",
                )
            notes.append(
                f"init_static_gravity_max_angle: {static_gate:g} -> {wanted:g} deg so the stationary start can initialize "
                f"with this mounting ({tilt:.1f} deg from raw -Z)"
            )
    early = imu.data[imu.times <= imu.times[0] + 3.0]
    if find_rest(early) is None and est.value("init_dyn_use") is False:
        est.set("init_dyn_use", True)
        notes.append(
            "init_dyn_use: true; the log is already moving in its first 3 s, so static initialization cannot start"
        )
    anchor = "max_cameras"
    for i, cam in enumerate(cameras):
        if cam.global_shutter:
            est.add(
                f"cam{i}_shutter",
                "global",
                f"{cam.sensor}: global shutter (generated)",
                after=anchor,
            )
            anchor = f"cam{i}_shutter"
        elif cam.sensor and not cam.thermal:
            notes.append(
                f"cam{i} ({cam.name}, {cam.sensor}): shutter type unknown; for rolling shutter set cam{i}_shutter: rolling "
                f"and cam{i}_readout_time_s"
            )
    est.save()

    # -- IMU chain ----------------------------------------------------------
    imu_file = ParamFile.load(out / imu_name, reference=source / imu_name)
    imu_rate = _rate(imu.times)
    if "imu0.update_rate" in imu_file and imu_rate > 0:
        imu_file.set("imu0.update_rate", round(imu_rate, 3))
        notes.append(f"imu0.update_rate: {imu_rate:.3f} Hz measured from {imu.name}")
    if "imu0.rostopic" in imu_file:
        imu_file.set("imu0.rostopic", imu.name)
    reset = []
    for key, matrix in ({} if template_cameras else _IMU_INTRINSICS).items():
        if key in imu_file and not np.allclose(
            np.array(imu_file.value(key), dtype=float), matrix
        ):
            imu_file.set(key, matrix.tolist())
            reset.append(key.split(".", 1)[1])
    if reset:
        notes.append(
            f"IMU intrinsics {', '.join(reset)} reset to identity/zero: the template's values belong to another unit"
        )
    noise = {
        k: imu_file.value(f"imu0.{k}")
        for k in (
            "gyroscope_noise_density",
            "gyroscope_random_walk",
            "accelerometer_noise_density",
            "accelerometer_random_walk",
        )
    }
    notes.append(
        "CHECK IMU noise densities (kept from the template, not measured): "
        + ", ".join(
            f"{k} {v:g}" for k, v in noise.items() if isinstance(v, (int, float))
        )
    )
    imu_file.save()

    if template_cameras:
        _template_cameras(source / cam_name, out / cam_name, cameras, stereo, notes)
        return _finish(out, source, overrides, notes)

    # -- camera chain (rebuilt) ---------------------------------------------
    rig = log.rig
    header = [
        "%YAML:1.0",
        f"# Generated by ov_zcalib Studio on {datetime.now().isoformat(timespec='seconds')}",
        f"# from {log.path} ({log.format_label}); IMU {imu.name}.",
        "# T_imu_cam maps camera-frame points into the IMU frame: p_I = R * p_C + t.",
        "# Camera stamps are streamed as center-row mid-exposure, so timeshift_cam_imu starts at 0.",
        "",
    ]
    body = []
    for i, cam in enumerate(cameras):
        transform = None
        if rig.edges and imu.frame and cam.frame:
            transform = rig.transform(imu.frame, cam.frame)
        if transform is None:
            T = np.eye(4)
            notes.append(
                f"WARNING cam{i} ({cam.name}): no {rig.source or 'extrinsics'} path from {imu.frame or imu.name} to "
                f"{cam.frame or cam.name}; T_imu_cam is IDENTITY and VIO will not work until it is set"
            )
        else:
            R, t = transform
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            notes.append(
                f"cam{i} ({cam.name}): T_imu_cam from {rig.source} ({imu.frame} -> {cam.frame})"
            )
        intr = cam.intrinsics
        extra = [float(v) for v in intr["d"][4:]]
        if intr["model"] != "equidistant" and any(abs(v) > 0 for v in extra):
            notes.append(
                f"cam{i} ({cam.name}): distortion terms beyond k1 k2 p1 p2 were dropped ({extra}); radtan uses four"
            )
        overlaps = [1 - i] if stereo else []
        body += _camera_block(i, cam, T, overlaps) + [""]
    (out / cam_name).write_text("\n".join(header + body), encoding="utf-8")

    return _finish(out, source, overrides, notes)


def _finish(out, source, overrides, notes):
    params = ParamSet.from_estimator_config(
        out / ESTIMATOR, reference=source / ESTIMATOR
    )
    if overrides:
        for key, value in overrides.items():
            params.set(key, value)
        params.save()
        notes.append(f"overrides: {', '.join(sorted(overrides))}")
    return out / ESTIMATOR, notes


def _template_cameras(source, target, cameras, stereo, notes):
    """Keep the template's camera chain; point its topics at the selected channels."""
    target.write_bytes(source.read_bytes())
    chain = ParamFile.load(target, reference=source)
    for i, cam in enumerate(cameras):
        if f"cam{i}.intrinsics" not in chain:
            raise ValueError(
                f"the template calibrates fewer than {len(cameras)} cameras (no cam{i})"
            )
        if f"cam{i}.rostopic" in chain:
            chain.set(f"cam{i}.rostopic", cam.name)
        resolution = chain.value(f"cam{i}.resolution")
        if resolution and cam.width and list(resolution) != [cam.width, cam.height]:
            raise ValueError(
                f"cam{i}: the template calibrates {resolution[0]}x{resolution[1]} but {cam.name} is {cam.width}x{cam.height}"
            )
        notes.append(
            f"cam{i} ({cam.name}): calibration (intrinsics, T_imu_cam) kept from the template"
        )
    if f"cam{len(cameras)}.intrinsics" in chain:
        notes.append(
            f"the template calibrates more cameras than selected; max_cameras={len(cameras)} uses the first {len(cameras)}"
        )
    chain.save()
