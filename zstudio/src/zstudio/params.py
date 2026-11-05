"""Format-preserving parameter files: OpenVINS/OpenCV YAML, Kalibr chains and VOXL JSON confs.

Every editable leaf becomes a ``Param`` that remembers the exact character
span of each scalar it was parsed from. Saving rewrites only the spans of
values that changed, so comments, blank lines, ordering, indentation, number
spelling (``1e-12``, ``11``, ``4.5e+02``), quote style and a VOXL ``/* */``
header survive byte-for-byte. A file saved without edits is identical to the
input.

Supported shapes (what OpenVINS, Kalibr and voxl-* configs actually use):

* YAML: ``%YAML`` directives, ``key: value # comment``, nested maps by
  indentation (``cam0:`` / ``imu0:``), flow lists ``[a, b]`` / ``[[..], [..]]``
  and block sequences (``- [a, b, c, d]`` matrix rows or ``- scalar`` items).
* JSON: objects, arrays (of scalars, of numeric rows, of objects), strings,
  numbers, booleans, null; ``/* */`` and ``//`` comments are treated as
  whitespace.

``ParamSet`` views an OpenVINS ``estimator_config.yaml`` together with the
files named by its ``relative_config_imu`` / ``relative_config_imucam`` keys.
"""

from dataclasses import dataclass, field
import json
import math
import re
from pathlib import Path

TYPES = ("bool", "int", "float", "string", "enum", "list", "matrix")


# ---------------------------------------------------------------------------
# Known OpenVINS metadata (descriptions where the files are silent, enums, types)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Meta:
    description: str = ""
    type: str = ""  # forces bool/int/float/enum/string when the text is ambiguous ("1" for a double)
    choices: tuple = ()
    minimum: float = None
    maximum: float = None
    unit: str = ""


_FEAT_REPS = (
    "GLOBAL_3D",
    "GLOBAL_FULL_INVERSE_DEPTH",
    "ANCHORED_3D",
    "ANCHORED_FULL_INVERSE_DEPTH",
    "ANCHORED_MSCKF_INVERSE_DEPTH",
    "ANCHORED_INVERSE_DEPTH_SINGLE",
)

KNOWN = {
    # general / estimator
    "verbosity": Meta(
        "Console log level of the estimator",
        "enum",
        ("ALL", "DEBUG", "INFO", "WARNING", "ERROR", "SILENT"),
    ),
    "use_fej": Meta(
        "Use first-estimate Jacobians (keeps the filter consistent)", "bool"
    ),
    "integration": Meta(
        "IMU propagation integrator; rk4/analytical also use analytical covariance propagation",
        "enum",
        ("discrete", "rk4", "analytical"),
    ),
    "use_stereo": Meta(
        "With 2+ cameras, track stereo constraints between synchronized pairs", "bool"
    ),
    "max_cameras": Meta(
        "Number of cameras: 1 mono, 2 stereo, >2 binocular (independent mono tracking)",
        "int",
        minimum=1,
    ),
    "gravity_mag": Meta("Local gravity magnitude", "float", minimum=0, unit="m/s^2"),
    "prop_window": Meta(
        "IMU propagation window kept around the newest camera time",
        "float",
        minimum=0,
        unit="s",
    ),
    "zupt_prop_window": Meta(
        "IMU window used by the zero-velocity update", "float", minimum=0, unit="s"
    ),
    "num_opencv_threads": Meta(
        "OpenCV threads: -1 auto, 0-1 serial, >1 count", "int", minimum=-1
    ),
    "multi_threading_pubs": Meta(
        "Publish visualization from a separate thread", "bool"
    ),
    "multi_threading_subs": Meta(
        "Process camera callbacks in a separate thread", "bool"
    ),
    # online calibration
    "calib_cam_extrinsics": Meta(
        "Estimate camera-IMU extrinsics online (R_ItoC, p_CinI)", "bool"
    ),
    "calib_cam_intrinsics": Meta(
        "Estimate camera intrinsics online (focal, center, distortion)", "bool"
    ),
    "calib_cam_timeoffset": Meta("Estimate the camera-IMU time offset online", "bool"),
    "calib_cam_readout": Meta(
        "Estimate the rolling-shutter readout time online (needs a positive seed)",
        "bool",
    ),
    "calib_cam_readout_init_sigma": Meta(
        "Initial 1-sigma of the readout time state", "float", minimum=0, unit="s"
    ),
    "calib_imu_intrinsics": Meta(
        "Estimate IMU intrinsics online (rotation and skew-scale)", "bool"
    ),
    "calib_imu_g_sensitivity": Meta(
        "Estimate the gyroscope gravity sensitivity Tg online", "bool"
    ),
    "cam_imu_dt_ref_camid": Meta(
        "Camera whose time offset is the shared reference", "int", minimum=0
    ),
    "rs_convention": Meta(
        "Rolling-shutter stamp row anchor", "enum", ("top", "center", "bottom")
    ),
    "dt_calib_gate": Meta(
        "Only update the time offset under sufficient excitation", "bool"
    ),
    "dt_calib_gate_min_omega": Meta(
        "Minimum angular rate for time-offset updates", "float", minimum=0, unit="rad/s"
    ),
    "dt_calib_gate_min_vel_spread": Meta(
        "Minimum velocity spread for time-offset updates",
        "float",
        minimum=0,
        unit="m/s",
    ),
    # sliding window
    "max_clones": Meta("Clones (poses) in the sliding window", "int", minimum=2),
    "max_epoch_clones": Meta(
        "Total pose capacity when epoch clones are enabled", "int", minimum=1
    ),
    "max_slam": Meta("SLAM features kept in the state vector", "int", minimum=0),
    "max_slam_in_update": Meta(
        "SLAM features per sequential update batch", "int", minimum=1
    ),
    "max_msckf_in_update": Meta("MSCKF features used per update", "int", minimum=0),
    "dt_slam_delay": Meta(
        "Delay after initialization before SLAM features are added",
        "float",
        minimum=0,
        unit="s",
    ),
    "feat_rep_msckf": Meta(
        "Feature representation for MSCKF features", "enum", _FEAT_REPS
    ),
    "feat_rep_slam": Meta(
        "Feature representation for SLAM features", "enum", _FEAT_REPS
    ),
    "feat_rep_aruco": Meta(
        "Feature representation for ArUco features", "enum", _FEAT_REPS
    ),
    # zero velocity
    "try_zupt": Meta("Enable zero-velocity updates", "bool"),
    "zupt_chi2_multipler": Meta(
        "ZUPT chi-squared multiplier; 0 uses disparity detection only",
        "float",
        minimum=0,
    ),
    "zupt_max_velocity": Meta(
        "Maximum velocity considered stationary", "float", minimum=0, unit="m/s"
    ),
    "zupt_noise_multiplier": Meta(
        "Noise multiplier applied in the ZUPT update", "float", minimum=0
    ),
    "zupt_max_disparity": Meta(
        "Maximum feature disparity considered stationary; 0 uses the IMU test only",
        "float",
        minimum=0,
        unit="px",
    ),
    "zupt_only_at_beginning": Meta(
        "Only apply ZUPT before the platform first moves", "bool"
    ),
    # initialization
    "init_window_time": Meta(
        "Data window collected for initialization", "float", minimum=0, unit="s"
    ),
    "init_imu_thresh": Meta(
        "Accelerometer variance threshold that detects a motion jerk",
        "float",
        minimum=0,
    ),
    "init_max_disparity": Meta(
        "Maximum disparity to consider the platform stationary",
        "float",
        minimum=0,
        unit="px",
    ),
    "init_max_features": Meta(
        "Features tracked during initialization", "int", minimum=1
    ),
    "init_dyn_use": Meta("Allow dynamic (moving) initialization", "bool"),
    "init_dyn_mle_opt_calib": Meta(
        "Optimize calibration during the initialization MLE (not recommended)", "bool"
    ),
    "init_dyn_mle_max_iter": Meta(
        "MLE refinement iterations (0 skips the MLE)", "int", minimum=0
    ),
    "init_dyn_mle_max_time": Meta("MLE time budget", "float", minimum=0, unit="s"),
    "init_dyn_mle_max_threads": Meta("Threads used by the MLE", "int", minimum=1),
    "init_dyn_num_pose": Meta(
        "Poses in the initialization window (evenly spaced)", "int", minimum=3
    ),
    "init_dyn_min_deg": Meta(
        "Orientation change required before trying to initialize",
        "float",
        minimum=0,
        unit="deg",
    ),
    "init_dyn_inflation_ori": Meta(
        "Inflation of the recovered orientation covariance", "float", minimum=0
    ),
    "init_dyn_inflation_pos": Meta(
        "Inflation of the recovered position covariance", "float", minimum=0
    ),
    "init_dyn_inflation_vel": Meta(
        "Inflation of the recovered velocity covariance", "float", minimum=0
    ),
    "init_dyn_inflation_bg": Meta(
        "Inflation of the recovered gyroscope-bias covariance", "float", minimum=0
    ),
    "init_dyn_inflation_ba": Meta(
        "Inflation of the recovered accelerometer-bias covariance", "float", minimum=0
    ),
    "init_dyn_min_rec_cond": Meta(
        "Reciprocal condition threshold for information inversion", "float", minimum=0
    ),
    "init_dyn_bias_g": Meta("Initial gyroscope bias guess", unit="rad/s"),
    "init_dyn_bias_a": Meta("Initial accelerometer bias guess", unit="m/s^2"),
    "init_gravity_max_angle": Meta(
        "Maximum tilt accepted by the dynamic-init gravity gate",
        "float",
        minimum=0,
        unit="deg",
    ),
    "init_static_gravity_max_angle": Meta(
        "Static-init gravity gate (<0 uses init_gravity_max_angle)", "float", unit="deg"
    ),
    "init_dyn_grav_prior_sigma": Meta(
        "Gravity prior sigma in the dynamic MLE (<=0 disables)", "float"
    ),
    "init_dyn_grav_gate_deg": Meta(
        "Post-solve gravity flip-reject gate", "float", minimum=0, unit="deg"
    ),
    "init_dyn_mle_lm_min_lambda": Meta(
        "Levenberg-Marquardt lambda floor", "float", minimum=0
    ),
    "init_dyn_mle_lm_nu_growth": Meta(
        "Levenberg-Marquardt reject escalation growth", "float", minimum=1
    ),
    "init_dyn_mle_lm_initial_lambda": Meta(
        "Levenberg-Marquardt initial damping", "float", minimum=0
    ),
    "init_dyn_mle_ftol": Meta("MLE relative cost tolerance", "float", minimum=0),
    "init_warmstart_inject": Meta(
        "Inject the warm-start state into the initializer", "bool"
    ),
    "init_dyn_reset_prior_use": Meta(
        "Use the previous run's biases as a prior after a soft reset", "bool"
    ),
    "init_dyn_reset_prior_max_bg": Meta(
        "Reject the reset prior if |bg| exceeds this", "float", minimum=0, unit="rad/s"
    ),
    "init_dyn_reset_prior_max_ba": Meta(
        "Reject the reset prior if |ba| exceeds this", "float", minimum=0, unit="m/s^2"
    ),
    "init_dyn_reset_prior_max_sigma_bg": Meta(
        "Reject if any age-inflated bg sigma exceeds this", "float", minimum=0
    ),
    "init_dyn_reset_prior_max_sigma_ba": Meta(
        "Reject if any age-inflated ba sigma exceeds this", "float", minimum=0
    ),
    "init_dyn_reset_prior_sigma_floor_bg": Meta(
        "Floor on the tightened bg prior sigma", "float", minimum=0
    ),
    "init_dyn_reset_prior_sigma_floor_ba": Meta(
        "Floor on the tightened ba prior sigma", "float", minimum=0
    ),
    "init_dyn_reset_prior_divergence_infl": Meta(
        "Prior sigma inflation after a divergence-triggered reset", "float", minimum=0
    ),
    "init_dyn_fix_ba_on_reset": Meta(
        "Hold the accelerometer bias fixed after a reset", "bool"
    ),
    # camera timing
    "force_camera_sync": Meta(
        "Group cameras by hardware trigger and process them as one synchronized frame",
        "bool",
    ),
    "epoch_mode": Meta("Clone at every independent camera's raw frame time", "bool"),
    "async_frame_clones": Meta(
        "Asynchronous per-frame clones for unsynchronized cameras", "bool"
    ),
    "physical_camera_clones": Meta("One clone per physical camera exposure", "bool"),
    "async_ring_size": Meta("Asynchronous camera ring-buffer size", "int", minimum=1),
    "async_guard": Meta(
        "Asynchronous buffer guard interval", "float", minimum=0, unit="s"
    ),
    "async_stale_factor": Meta(
        "Frames older than this many periods are stale", "float", minimum=0
    ),
    # tracking
    "use_klt": Meta(
        "KLT optical-flow tracking (otherwise ORB descriptors + matching)", "bool"
    ),
    "use_gpu": Meta("Use the TrackOCL GPU tracker (VOXL only)", "bool"),
    "num_pts": Meta("Features extracted and tracked per camera", "int", minimum=1),
    "fast_threshold": Meta(
        "FAST corner threshold (lower is more expensive)", "int", minimum=1
    ),
    "grid_x": Meta("Extraction grid cells, horizontal", "int", minimum=1),
    "grid_y": Meta("Extraction grid cells, vertical", "int", minimum=1),
    "min_px_dist": Meta(
        "Minimum distance between features", "int", minimum=0, unit="px"
    ),
    "knn_ratio": Meta(
        "Descriptor KNN ratio test threshold", "float", minimum=0, maximum=1
    ),
    "track_frequency": Meta(
        "Frames faster than this rate are dropped before tracking",
        "float",
        minimum=0,
        unit="Hz",
    ),
    "downsample_cameras": Meta("Halve image resolution before tracking", "bool"),
    "histogram_method": Meta(
        "Image equalization before tracking", "enum", ("NONE", "HISTOGRAM", "CLAHE")
    ),
    "use_mask": Meta("Apply the mask images named by maskN", "bool"),
    "use_aruco": Meta("Track ArUco tags", "bool"),
    "num_aruco": Meta("Number of ArUco tag ids", "int", minimum=0),
    "downsize_aruco": Meta("Detect ArUco tags on a half-resolution image", "bool"),
    # triangulation
    "fi_triangulate_1d": Meta(
        "Triangulate features with the 1D (depth only) method", "bool"
    ),
    "fi_refine_features": Meta("Gauss-Newton refine triangulated features", "bool"),
    "fi_max_runs": Meta("Maximum feature refinement iterations", "int", minimum=0),
    "fi_init_lamda": Meta("Initial Levenberg-Marquardt lambda", "float", minimum=0),
    "fi_max_lamda": Meta("Maximum Levenberg-Marquardt lambda", "float", minimum=0),
    "fi_min_dx": Meta("Convergence: minimum state change", "float", minimum=0),
    "fi_min_dcost": Meta("Convergence: minimum cost change", "float", minimum=0),
    "fi_lam_mult": Meta("Lambda multiplier", "float", minimum=0),
    "fi_min_dist": Meta("Minimum feature depth", "float", minimum=0, unit="m"),
    "fi_max_dist": Meta("Maximum feature depth", "float", minimum=0, unit="m"),
    "fi_max_baseline": Meta("Maximum baseline ratio", "float", minimum=0),
    "fi_max_cond_number": Meta(
        "Maximum condition number of the triangulation", "float", minimum=0
    ),
    # update noise
    "up_msckf_sigma_px": Meta("MSCKF pixel noise", "float", minimum=0, unit="px"),
    "up_msckf_chi2_multipler": Meta(
        "MSCKF chi-squared gate multiplier", "float", minimum=0
    ),
    "up_slam_sigma_px": Meta("SLAM pixel noise", "float", minimum=0, unit="px"),
    "up_slam_chi2_multipler": Meta(
        "SLAM chi-squared gate multiplier", "float", minimum=0
    ),
    "up_aruco_sigma_px": Meta("ArUco pixel noise", "float", minimum=0, unit="px"),
    "up_aruco_chi2_multipler": Meta(
        "ArUco chi-squared gate multiplier", "float", minimum=0
    ),
    # files / output
    "relative_config_imu": Meta(
        "IMU chain (noise, intrinsics) relative to this file", "string"
    ),
    "relative_config_imucam": Meta(
        "Camera chain (intrinsics, extrinsics) relative to this file", "string"
    ),
    "record_timing_information": Meta("Record per-stage timing", "bool"),
    "record_timing_filepath": Meta("Timing output file", "string"),
    "save_total_state": Meta("Save the estimate and its diagonal covariance", "bool"),
    "filepath_est": Meta("Estimate output file", "string"),
    "filepath_std": Meta("Standard deviation output file", "string"),
    "filepath_gt": Meta("Ground-truth output file", "string"),
    # kalibr chains (leaf names)
    "T_imu_cam": Meta("Camera to IMU transform: p_I = T_imu_cam * p_C"),
    "T_cam_imu": Meta("IMU to camera transform (OpenVINS inverts it into T_imu_cam)"),
    "T_i_b": Meta("Body to IMU transform"),
    "intrinsics": Meta("Pinhole intrinsics [fx, fy, cx, cy]", unit="px"),
    "distortion_coeffs": Meta(
        "Distortion coefficients (radtan k1 k2 p1 p2 / equidistant k1..k4)"
    ),
    "distortion_model": Meta(
        "Lens distortion model", "enum", ("radtan", "equidistant")
    ),
    "camera_model": Meta("Projection model", "enum", ("pinhole",)),
    "resolution": Meta("Image size [width, height]", unit="px"),
    "timeshift_cam_imu": Meta(
        "Camera-IMU time offset t_imu = t_cam + timeshift", "float", unit="s"
    ),
    "t_readout": Meta(
        "Rolling-shutter readout time of the full frame", "float", minimum=0, unit="s"
    ),
    "cam_overlaps": Meta("Indices of cameras with overlapping fields of view"),
    "rostopic": Meta("Source topic or VOXL pipe (informational for replay)", "string"),
    "accelerometer_noise_density": Meta(
        "Accelerometer white noise density", "float", minimum=0, unit="m/s^2/sqrt(Hz)"
    ),
    "accelerometer_random_walk": Meta(
        "Accelerometer bias random walk", "float", minimum=0, unit="m/s^3/sqrt(Hz)"
    ),
    "gyroscope_noise_density": Meta(
        "Gyroscope white noise density", "float", minimum=0, unit="rad/s/sqrt(Hz)"
    ),
    "gyroscope_random_walk": Meta(
        "Gyroscope bias random walk", "float", minimum=0, unit="rad/s^2/sqrt(Hz)"
    ),
    "update_rate": Meta("IMU sample rate", "float", minimum=0, unit="Hz"),
    "time_offset": Meta("IMU time offset", "float", unit="s"),
    "model": Meta("IMU intrinsic model", "enum", ("kalibr", "calibrated", "rpng")),
    "Tw": Meta("Gyroscope scale/misalignment matrix"),
    "Ta": Meta("Accelerometer scale/misalignment matrix"),
    "R_IMUtoGYRO": Meta("IMU to gyroscope frame rotation"),
    "R_IMUtoACC": Meta("IMU to accelerometer frame rotation"),
    "Tg": Meta("Gyroscope gravity sensitivity"),
}
_CAM_KEY = re.compile(r"cam(\d+)_(readout_time_s|shutter|fps)\Z")
_CAM_META = {
    "readout_time_s": Meta(
        "Rolling-shutter readout time of the full frame (wins over the kalibr t_readout)",
        "float",
        minimum=0,
        unit="s",
    ),
    "shutter": Meta(
        "Shutter type: global zeroes any readout, rolling requires a positive readout",
        "enum",
        ("global", "rolling"),
    ),
    "fps": Meta("Nominal camera frame rate", "float", minimum=0, unit="Hz"),
}

# Display groups for flat OpenVINS estimator keys (QGC-style prefix grouping).
_GROUPS = [
    (r"verbosity\Z", "General"),
    (
        r"(use_fej|integration|use_stereo|max_cameras|gravity_mag|prop_window|zupt_prop_window|multi_threading_\w+|num_opencv_threads)\Z",
        "Estimator",
    ),
    (
        r"(calib_|dt_calib_gate|rs_convention\Z|cam_imu_dt_ref_camid\Z)",
        "Online calibration",
    ),
    (
        r"(max_clones|max_epoch_clones|max_slam|max_slam_in_update|max_msckf_in_update|dt_slam_delay)\Z",
        "Sliding window",
    ),
    (r"feat_rep_", "Feature representation"),
    (r"(try_zupt|zupt_)", "Zero-velocity update"),
    (
        r"(init_dyn_reset_prior_|init_dyn_fix_ba_on_reset)",
        "Initialization: reset prior",
    ),
    (r"init_dyn_", "Initialization: dynamic"),
    (r"init_", "Initialization"),
    (r"(force_camera_sync|epoch_mode|async_|physical_camera_clones)", "Camera timing"),
    (r"cam\d+_", "Cameras"),
    (
        r"(use_klt|use_gpu|num_pts|fast_threshold|grid_x|grid_y|min_px_dist|knn_ratio|track_frequency|downsample_cameras|histogram_method|use_mask|mask\d+)\Z",
        "Feature tracking",
    ),
    (r"fi_", "Feature triangulation"),
    (r"(use_aruco|num_aruco|downsize_aruco)\Z", "ArUco"),
    (r"up_", "Update noise"),
    (r"relative_config_", "Linked files"),
    (r"(record_timing_|save_total_state|filepath_)", "Output"),
    (r"sim_", "Simulation"),
]


def meta_for(key):
    leaf = key.rsplit(".", 1)[-1]
    if leaf in KNOWN:
        return KNOWN[leaf]
    m = _CAM_KEY.match(leaf)
    return _CAM_META[m.group(2)] if m else Meta()


# ---------------------------------------------------------------------------
# scalar lexing / rendering
# ---------------------------------------------------------------------------
_INT = re.compile(r"[-+]?\d+\Z")
_FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?\Z")


@dataclass
class _Elem:
    """One scalar occurrence: absolute [start, end) span in the file text."""

    start: int
    end: int
    text: str
    kind: str  # bool | int | float | string | null
    quote: str = ""  # '"' or "'" for quoted strings


def _scalar_value(text, quote=""):
    if quote:
        return text, "string"
    low = text.lower()
    if low in ("true", "false"):
        return low == "true", "bool"
    if low == "null":
        return None, "null"
    if _INT.match(text):
        return int(text), "int"
    if _FLOAT.match(text):
        return float(text), "float"
    return text, "string"


def _format_float(value, like=""):
    value = float(value)
    like = (like or "").strip()
    if like and _INT.match(like) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))  # "1" stays "1" for a double parameter
    if value == 0.0:
        return "0.0"
    text = repr(value)
    if "e" in like.lower() and "e" not in text:
        for digits in range(17):
            candidate = f"{value:.{digits}e}"
            if float(candidate) == value:
                return candidate
    return text


def _render_scalar(value, like, fmt):
    """Render ``value`` in the style of ``like`` (an _Elem or None)."""
    kind = like.kind if like is not None else None
    if isinstance(value, bool):
        token = "true" if value else "false"
        if like is not None and like.kind == "bool":
            if like.text.isupper():
                token = token.upper()
            elif like.text[:1].isupper():
                token = token.capitalize()
        return token
    if value is None:
        return "null"
    if isinstance(value, int) and kind != "float":
        return str(value)
    if isinstance(value, (int, float)):
        return _format_float(value, like.text if like is not None else "")
    text = str(value)
    if fmt == "json":
        return json.dumps(text, ensure_ascii=False)
    quote = like.quote if like is not None else ""
    if not quote and (
        text == ""
        or text != text.strip()
        or re.search(r"(:\s|\s#|[\[\]{},]|^[-?!&*|>'\"%@`])", text)
        or text.lower() in ("true", "false", "null")
        or _FLOAT.match(text)
    ):
        quote = '"'
    if quote == "'":
        return "'" + text.replace("'", "''") + "'"
    if quote == '"':
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _same(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or (
            isinstance(a, float)
            and isinstance(b, float)
            and math.isnan(a)
            and math.isnan(b)
        )
    return a == b


def value_text(value):
    """Compact human text for a parameter value (lists as [a, b], matrices as [[..], [..]])."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(value_text(v) for v in value) + "]"
    if value is None:
        return "null"
    return str(value)


# ---------------------------------------------------------------------------
# flow-sequence lexer shared by YAML and user input: "[1, 2, [3, 4]]"
# ---------------------------------------------------------------------------
def _flow(text, base, pos):
    """Parse a flow list starting at text[pos] == '['. Returns (values, elems, end)."""
    assert text[pos] == "["
    values, elems = [], []
    i = pos + 1
    n = len(text)
    while True:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            raise ValueError("unterminated '['")
        if text[i] == "]":
            return values, elems, i + 1
        if text[i] == "[":
            v, e, i = _flow(text, base, i)
            values.append(v)
            elems.append(e)
        elif text[i] in "\"'":
            q = text[i]
            j = i + 1
            buf = []
            while j < n:
                if text[j] == "\\" and q == '"' and j + 1 < n:
                    buf.append({"n": "\n", "t": "\t"}.get(text[j + 1], text[j + 1]))
                    j += 2
                    continue
                if text[j] == q:
                    if q == "'" and j + 1 < n and text[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    break
                buf.append(text[j])
                j += 1
            if j >= n:
                raise ValueError("unterminated quote")
            values.append("".join(buf))
            elems.append(_Elem(base + i, base + j + 1, text[i : j + 1], "string", q))
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in ",]":
                j += 1
            token = text[i:j].rstrip()
            if not token:
                raise ValueError("empty list element")
            v, kind = _scalar_value(token)
            values.append(v)
            elems.append(_Elem(base + i, base + i + len(token), token, kind))
            i = j
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i < n and text[i] == ",":
            i += 1
        elif i < n and text[i] == "]":
            continue
        else:
            raise ValueError("expected ',' or ']' in list")


def parse_user_value(text):
    """Values typed by a user: lists/matrices in flow syntax, else a bare scalar string."""
    text = text.strip()
    if text.startswith("["):
        values, _, end = _flow(text, 0, 0)
        if text[end:].strip():
            raise ValueError(f"unexpected text after list: {text[end:].strip()!r}")
        return values
    return text


# ---------------------------------------------------------------------------
# Param / ParamFile
# ---------------------------------------------------------------------------
@dataclass
class Param:
    key: str
    value: object
    type: str
    default: object = None
    description: str = ""
    notes: str = ""
    group: str = "General"
    choices: tuple = ()
    minimum: float = None
    maximum: float = None
    unit: str = ""
    line: int = 0
    readonly: bool = False
    file: "ParamFile" = field(default=None, repr=False, compare=False)
    saved: object = field(default=None, repr=False)
    _elems: object = field(default=None, repr=False)  # _Elem | [_Elem] | [[_Elem]]
    _container: tuple = field(
        default=None, repr=False
    )  # (start, end) of a flow list / JSON array
    _block: bool = field(
        default=False, repr=False
    )  # YAML block sequence ("- ..." rows)

    @property
    def leaf(self):
        return self.key.rsplit(".", 1)[-1]

    @property
    def modified(self):
        return not _same(self.value, self.default)

    @property
    def dirty(self):
        return not _same(self.value, self.saved)

    @property
    def shape(self):
        if self.type == "matrix":
            return (len(self.value), len(self.value[0]) if self.value else 0)
        if self.type == "list":
            return (len(self.value),)
        return ()

    def text(self, value=None):
        return value_text(self.value if value is None else value)

    # -- validation ------------------------------------------------------
    def coerce(self, value):
        """Validate ``value`` for this parameter and return it in the stored type."""
        if self.readonly:
            raise ValueError(f"{self.key} is read-only")
        kind = self.type
        if isinstance(value, str) and kind in ("list", "matrix"):
            value = parse_user_value(value)
        if kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str) and value.strip().lower() in (
                "true",
                "false",
                "1",
                "0",
                "yes",
                "no",
                "on",
                "off",
            ):
                return value.strip().lower() in ("true", "1", "yes", "on")
            raise ValueError(f"{self.key} expects true or false, got {value!r}")
        if kind == "int":
            number = self._number(value)
            if not float(number).is_integer():
                raise ValueError(f"{self.key} expects an integer, got {value!r}")
            return self._bounded(int(number))
        if kind == "float":
            return self._bounded(float(self._number(value)))
        if kind == "enum":
            text = str(value).strip()
            for choice in self.choices:
                if text.lower() == choice.lower():
                    return choice
            raise ValueError(
                f"{self.key} must be one of {', '.join(self.choices)}; got {text!r}"
            )
        if kind == "string":
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                raise ValueError(f"{self.key} expects text, got {value!r}")
            return str(value)
        if kind == "list":
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{self.key} expects a list like [a, b, c]")
            items = [self._like_item(v, i) for i, v in enumerate(value)]
            if self._block and len(items) != len(self.value):
                raise ValueError(f"{self.key} has a fixed length of {len(self.value)}")
            return items
        if kind == "matrix":
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(r, (list, tuple)) for r in value
            ):
                raise ValueError(f"{self.key} expects rows like [[a, b], [c, d]]")
            rows = [[self._number(v) for v in r] for r in value]
            if (len(rows), len(rows[0]) if rows else 0) != self.shape or any(
                len(r) != self.shape[1] for r in rows
            ):
                raise ValueError(
                    f"{self.key} must stay {self.shape[0]}x{self.shape[1]}"
                )
            return rows
        raise ValueError(f"{self.key} cannot be edited")

    def _number(self, value):
        if isinstance(value, bool):
            raise ValueError(f"{self.key} expects a number, got {value!r}")
        if isinstance(value, (int, float)):
            number = value
        else:
            text = str(value).strip()
            if _INT.match(text):
                number = int(text)
            elif _FLOAT.match(text) or re.fullmatch(r"[-+]?(inf|nan)", text.lower()):
                number = float(text)
            else:
                raise ValueError(f"{self.key} expects a number, got {value!r}")
        if not math.isfinite(float(number)):
            raise ValueError(f"{self.key} must be finite")
        return number

    def _bounded(self, number):
        if self.minimum is not None and number < self.minimum:
            raise ValueError(f"{self.key} must be >= {value_text(self.minimum)}")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"{self.key} must be <= {value_text(self.maximum)}")
        return number

    def _like_item(self, value, index):
        template = self.value[min(index, len(self.value) - 1)] if self.value else None
        if isinstance(template, (int, float)) and not isinstance(template, bool):
            number = self._number(value)
            return (
                float(number)
                if isinstance(template, float) and not isinstance(number, float)
                else number
            )
        if isinstance(template, bool):
            if isinstance(value, bool):
                return value
            raise ValueError(f"{self.key} expects true/false items")
        return (
            str(value)
            if not isinstance(value, (int, float)) or isinstance(value, bool)
            else value
        )


class ParamFile:
    """One parameter file. ``params`` are in file order; keys are dotted paths."""

    def __init__(self, path, text, fmt, params, reference=None):
        self.path = Path(path) if path else None
        self._text_changed = False  # lines were added in memory (add()); disk differs
        self.text = text
        self.format = fmt
        self.params = params
        self.by_key = {p.key: p for p in params}
        for p in params:
            p.file = self
            p.saved = p.value
            if p.default is None:
                p.default = p.value
        if reference is not None:
            self.apply_reference(reference)

    # -- construction ----------------------------------------------------
    @classmethod
    def load(cls, path, reference=None):
        path = Path(path)
        text = path.read_bytes().decode("utf-8")
        return cls.parse(text, path, reference=reference)

    @classmethod
    def parse(cls, text, path=None, fmt=None, reference=None):
        fmt = fmt or detect_format(text, path)
        try:
            params = _parse_json(text) if fmt == "json" else _parse_yaml(text)
        except (IndexError, ValueError) as error:
            where = Path(path).name if path else "text"
            raise ValueError(
                f"{where}: cannot parse {fmt.upper()}: {error or 'unexpected end of file'}"
            ) from error
        return cls(path, text, fmt, params, reference)

    def apply_reference(self, reference):
        """Use another file's values (same keys) as defaults, e.g. the template a config came from."""
        if not isinstance(reference, ParamFile):
            reference = ParamFile.load(reference)
        for p in self.params:
            ref = reference.by_key.get(p.key)
            if ref is not None and ref.type == p.type and ref.shape == p.shape:
                p.default = ref.value

    @property
    def name(self):
        return self.path.name if self.path else "(memory)"

    # -- access ------------------------------------------------------------
    def __contains__(self, key):
        return key in self.by_key

    def __iter__(self):
        return iter(self.params)

    def get(self, key):
        try:
            return self.by_key[key]
        except KeyError:
            raise KeyError(f"{self.name} has no parameter {key!r}") from None

    def value(self, key, default=None):
        p = self.by_key.get(key)
        return p.value if p is not None else default

    def set(self, key, value):
        p = self.get(key)
        p.value = p.coerce(value)
        return p

    def reset(self, key):
        p = self.get(key)
        p.value = p.default
        return p

    def reset_all(self):
        for p in self.params:
            if not p.readonly:
                p.value = p.default

    @property
    def modified(self):
        return [p for p in self.params if p.modified]

    @property
    def dirty(self):
        return [p for p in self.params if p.dirty]

    def has_unsaved(self):
        return self._text_changed or any(p.dirty for p in self.params)

    # -- persistence -------------------------------------------------------
    def render(self):
        """File text with every dirty value rewritten in place."""
        edits = []
        for p in self.params:
            if p.dirty:
                edits.extend(_edits_for(p, self.format))
        text = self.text
        for start, end, new in sorted(edits, key=lambda e: e[0], reverse=True):
            text = text[:start] + new + text[end:]
        return text

    def save(self, path=None):
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path to save to")
        text = self.render()
        target.write_bytes(text.encode("utf-8"))
        self._reload(text, target)
        self._text_changed = False
        return target

    def revert(self):
        """Discard unsaved edits by re-reading the file from disk (defaults are kept)."""
        if self.path is None:
            for p in self.params:
                p.value = p.saved
            return
        self._reload(self.path.read_bytes().decode("utf-8"), self.path)
        self._text_changed = False

    def add(self, key, value, comment="", after=None):
        """Append a new top-level ``key: value`` line (YAML) after ``after`` or at the end."""
        if self.format != "yaml":
            raise ValueError("adding parameters is only supported for YAML files")
        if "." in key or not re.fullmatch(r"[A-Za-z_]\w*", key):
            raise ValueError(f"invalid top-level key {key!r}")
        if key in self.by_key:
            return self.set(key, value)
        if isinstance(value, str):
            rendered = '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        elif isinstance(value, (list, tuple)):
            rendered = value_text(value)
        else:
            rendered = _render_scalar(value, None, "yaml")
        text = self.render()
        line = f"{key}: {rendered}"
        if comment:
            # OpenCV's YAML reader (OpenVINS) splits on ':' even inside a trailing
            # comment and then reads the value as a map; never write one.
            line += " # " + comment.replace(":", " -")
        if after is not None and after in self.by_key:
            anchor = self.by_key[after]
            lines = text.splitlines(keepends=True)
            index = anchor.line + (len(anchor.value) if anchor._block else 0)
            lines.insert(index + 1, line + "\n")
            text = "".join(lines)
        else:
            if text and not text.endswith("\n"):
                text += "\n"
            text += line + "\n"
        self._reload(text, self.path)
        self._text_changed = True
        return self.by_key[key]

    def _reload(self, text, path):
        defaults = {p.key: (p.default, p.type, p.shape) for p in self.params}
        fresh = ParamFile.parse(text, path, fmt=self.format)
        self.text, self.path = text, Path(path) if path else None
        self.params, self.by_key = fresh.params, fresh.by_key
        for p in self.params:
            p.file = self
            if p.key in defaults and defaults[p.key][1:] == (p.type, p.shape):
                p.default = defaults[p.key][0]


def detect_format(text, path=None):
    suffix = Path(path).suffix.lower() if path else ""
    stripped = re.sub(r"/\*.*?\*/", "", text, flags=re.S).lstrip()
    if (
        suffix in (".json", ".conf")
        or stripped.startswith("{")
        or stripped.startswith("[")
    ):
        if stripped.startswith(("{", "[")):
            return "json"
    return "yaml"


def _edits_for(p, fmt):
    """[(start, end, replacement)] turning p.saved's text into p.value's."""
    if p.type in ("list", "matrix") and isinstance(p._elems, list):
        new, old = p.value, p.saved
        if p.type == "matrix":
            if len(new) == len(p._elems) and all(
                len(r) == len(e) for r, e in zip(new, p._elems)
            ):
                out = []
                for row_new, row_old, row_elems in zip(new, old, p._elems):
                    for v, o, e in zip(row_new, row_old, row_elems):
                        if not _same(v, o):
                            out.append((e.start, e.end, _render_scalar(v, e, fmt)))
                return out
        elif len(new) == len(p._elems):
            return [
                (e.start, e.end, _render_scalar(v, e, fmt))
                for v, o, e in zip(new, old, p._elems)
                if not _same(v, o)
            ]
        if p._container is None:
            raise ValueError(f"{p.key}: block sequences keep their length")
        start, end = p._container
        original = p.file.text[start:end]
        inner_space = " " if original.startswith("[ ") else ""

        def render(values, elems):
            like = elems[0] if elems and isinstance(elems[0], _Elem) else None
            parts = []
            for v in values:
                if isinstance(v, (list, tuple)):
                    sub = elems[0] if elems and isinstance(elems[0], list) else []
                    parts.append(render(v, sub))
                else:
                    parts.append(_render_scalar(v, like, fmt))
            return "[" + inner_space + ", ".join(parts) + inner_space + "]"

        return [(start, end, render(new, p._elems))]
    elem = p._elems
    return [(elem.start, elem.end, _render_scalar(p.value, elem, fmt))]


# ---------------------------------------------------------------------------
# YAML (OpenCV / OpenVINS / Kalibr subset)
# ---------------------------------------------------------------------------
_KEY = re.compile(r"^([ \t]*)([A-Za-z_][\w\-./]*)[ \t]*:(?=\s|$)")
_BANNER = re.compile(r"^[=\-#*~_+ ]{4,}$")


def _split_comment(text):
    """(value_end, comment) for the remainder of a line after 'key:'."""
    quote = None
    for i, c in enumerate(text):
        if quote:
            if c == quote:
                quote = None
            continue
        if c in "\"'" and (i == 0 or text[i - 1] in " \t[,"):
            quote = c
        elif c == "#" and (i == 0 or text[i - 1] in " \t"):
            return len(text[:i].rstrip()), text[i + 1 :].strip()
    return len(text.rstrip("\r\n").rstrip()), ""


def _top_group(key, heading):
    for pattern, name in _GROUPS:
        if re.match(pattern, key):
            return name
    if heading:
        return heading[:1].upper() + heading[1:].rstrip(".:")
    return "General"


def _typed(key, value, kind, elem):
    """(value, type, meta, choices) after applying the metadata table."""
    meta = meta_for(key)
    ptype, choices = kind, ()
    if kind == "null":
        ptype = "string"
    if meta.type == "float" and kind == "int":
        ptype, value = "float", float(value)
    elif meta.type == "enum" and kind == "string":
        ptype, choices = "enum", meta.choices
        if not any(value.lower() == c.lower() for c in choices):
            choices = choices + (
                value,
            )  # keep values this fork accepts but the table lacks
    elif meta.type == "string" and kind in ("int", "float", "bool"):
        ptype, value = "string", elem.text
    return value, ptype, meta, choices


def _parse_yaml(text):
    params = []
    lines = text.splitlines(keepends=True)
    offsets = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    stack = []  # (indent, key)
    pending, heading, fresh_paragraph = [], "", True
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            pending, heading, fresh_paragraph = [], "", True
            i += 1
            continue
        if stripped.startswith("%"):
            i += 1
            continue
        if stripped.startswith("#"):
            comment = stripped.lstrip("#").strip()
            if not comment or _BANNER.match(comment):
                pending, heading, fresh_paragraph = [], "", True
            else:
                if fresh_paragraph and not pending:
                    heading = comment
                pending.append(comment)
            i += 1
            continue
        m = _KEY.match(raw)
        if not m:
            pending = []
            fresh_paragraph = False
            i += 1
            continue
        fresh_paragraph = False
        indent = len(m.group(1).expandtabs())
        key = m.group(2)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([k for _, k in stack] + [key])
        rest_start = m.end()
        rest = raw[rest_start:]
        lead = len(rest) - len(rest.lstrip(" \t"))
        value_end, comment = _split_comment(rest[lead:])
        value_text_ = rest[lead : lead + value_end]
        base = offsets[i] + rest_start + lead
        notes = "\n".join(pending)
        pending = []
        group = stack[0][1] if stack else _top_group(key, heading)
        if not value_text_:
            # nested map or block sequence: look at the next meaningful line
            j = i + 1
            while j < len(lines) and (
                not lines[j].strip() or lines[j].strip().startswith("#")
            ):
                j += 1
            next_indent = (
                len(lines[j]) - len(lines[j].lstrip(" \t")) if j < len(lines) else -1
            )
            if (
                j < len(lines)
                and lines[j].lstrip().startswith("- ")
                and next_indent >= indent
            ):
                rows, elems, k = [], [], j
                while k < len(lines):
                    s = lines[k].strip()
                    if (
                        not s.startswith("- ")
                        or len(lines[k]) - len(lines[k].lstrip(" \t")) != next_indent
                    ):
                        break
                    dash = lines[k].index("-")
                    item_start = (
                        dash
                        + 1
                        + (
                            len(lines[k][dash + 1 :])
                            - len(lines[k][dash + 1 :].lstrip(" \t"))
                        )
                    )
                    item = lines[k][item_start:]
                    if item.startswith("["):
                        v, e, _ = _flow(lines[k], offsets[k], item_start)
                    else:
                        end, _ = _split_comment(item)
                        token = item[:end]
                        quote = (
                            token[0]
                            if token[:1] in "\"'"
                            and token[-1:] == token[:1]
                            and len(token) > 1
                            else ""
                        )
                        v, kind = _scalar_value(token[1:-1] if quote else token, quote)
                        e = _Elem(
                            offsets[k] + item_start,
                            offsets[k] + item_start + len(token),
                            token,
                            kind,
                            quote,
                        )
                    rows.append(v)
                    elems.append(e)
                    k += 1
                matrix = rows and all(isinstance(r, list) for r in rows)
                meta = meta_for(path)
                if comment and meta.description and comment != meta.description:
                    notes = "\n".join(filter(None, [meta.description, notes]))
                params.append(
                    Param(
                        path,
                        rows,
                        "matrix" if matrix else "list",
                        description=comment or meta.description,
                        notes=notes,
                        group=group if stack else path,
                        unit=meta.unit,
                        line=i,
                        _elems=elems,
                        _block=True,
                    )
                )
                i = k
                continue
            if j < len(lines) and next_indent > indent:
                stack.append((indent, key))
                i += 1
                continue
            # empty scalar: make it an editable empty string anchored after the colon
            elem = _Elem(base, base, "", "string", "")
            meta = meta_for(path)
            params.append(
                Param(
                    path,
                    "",
                    "string",
                    description=meta.description or comment,
                    notes=notes,
                    group=group,
                    line=i,
                    _elems=elem,
                )
            )
            i += 1
            continue
        meta = meta_for(path)
        description = comment or meta.description
        if comment and meta.description and comment != meta.description:
            notes = "\n".join(
                filter(None, [meta.description, notes])
            )  # the curated text stays visible
        if value_text_.startswith("["):
            try:
                values, elems, end = _flow(raw, offsets[i], rest_start + lead)
            except ValueError:
                values = None
            if values is not None:
                matrix = values and all(isinstance(v, list) for v in values)
                params.append(
                    Param(
                        path,
                        values,
                        "matrix" if matrix else "list",
                        description=description,
                        notes=notes,
                        group=group,
                        unit=meta.unit,
                        line=i,
                        _elems=elems,
                        _container=(base, offsets[i] + end),
                    )
                )
                i += 1
                continue
        quote = (
            value_text_[0]
            if value_text_[:1] in "\"'"
            and value_text_[-1:] == value_text_[:1]
            and len(value_text_) > 1
            else ""
        )
        token = value_text_[1:-1] if quote else value_text_
        if quote == '"':
            token = token.replace('\\"', '"').replace("\\\\", "\\")
        elif quote == "'":
            token = token.replace("''", "'")
        value, kind = _scalar_value(token, quote)
        elem = _Elem(base, base + len(value_text_), value_text_, kind, quote)
        value, ptype, meta, choices = _typed(path, value, kind, elem)
        params.append(
            Param(
                path,
                value,
                ptype,
                description=description,
                notes=notes,
                group=group,
                choices=choices,
                minimum=meta.minimum,
                maximum=meta.maximum,
                unit=meta.unit,
                line=i,
                readonly=kind == "null",
                _elems=elem,
            )
        )
        i += 1
    return params


# ---------------------------------------------------------------------------
# JSON (cJSON-style VOXL confs, comments tolerated)
# ---------------------------------------------------------------------------
class _Json:
    def __init__(self, text):
        self.t = text
        self.i = 0
        self.params = []

    def line_of(self, pos):
        return self.t.count("\n", 0, pos)

    def ws(self):
        t, n = self.t, len(self.t)
        while self.i < n:
            c = t[self.i]
            if c in " \t\r\n":
                self.i += 1
            elif t.startswith("/*", self.i):
                end = t.find("*/", self.i + 2)
                if end < 0:
                    raise ValueError("unterminated /* comment")
                self.i = end + 2
            elif t.startswith("//", self.i):
                end = t.find("\n", self.i)
                self.i = n if end < 0 else end + 1
            else:
                break

    def expect(self, c):
        self.ws()
        if self.i >= len(self.t) or self.t[self.i] != c:
            raise ValueError(f"expected '{c}' at line {self.line_of(self.i) + 1}")
        self.i += 1

    def string(self):
        start = self.i
        self.i += 1
        t = self.t
        while self.i < len(t):
            if t[self.i] == "\\":
                self.i += 2
                continue
            if t[self.i] == '"':
                self.i += 1
                raw = t[start : self.i]
                return json.loads(raw), _Elem(start, self.i, raw, "string", '"')
            self.i += 1
        raise ValueError("unterminated string")

    def literal(self):
        m = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|true|false|null").match(
            self.t, self.i
        )
        if not m:
            raise ValueError(f"invalid JSON value at line {self.line_of(self.i) + 1}")
        self.i = m.end()
        value, kind = _scalar_value(m.group(0))
        return value, _Elem(m.start(), m.end(), m.group(0), kind)

    def value(self):
        """Returns (python value, elem structure | None for objects/mixed)."""
        self.ws()
        c = self.t[self.i] if self.i < len(self.t) else ""
        if c == "{":
            return self.obj(None), None
        if c == "[":
            return self.array(None)
        if c == '"':
            return self.string()
        return self.literal()

    def obj(self, path):
        self.expect("{")
        members = []
        self.ws()
        if self.t[self.i] == "}":
            self.i += 1
            return {}
        while True:
            self.ws()
            key, _ = self.string()
            self.expect(":")
            self.ws()
            members.append((key, self.i))
            child = f"{path}.{key}" if path else key
            self.member(child, path)
            self.ws()
            if self.t[self.i] == ",":
                self.i += 1
                continue
            self.expect("}")
            return {}

    def array(self, path):
        start = self.i
        self.expect("[")
        values, elems = [], []
        self.ws()
        if self.t[self.i] == "]":
            self.i += 1
            return [], ((start, self.i), [])
        index = 0
        while True:
            self.ws()
            c = self.t[self.i]
            if c == "{":
                self.obj(f"{path}[{index}]" if path is not None else f"[{index}]")
                values.append(None)
                elems.append(None)
            elif c == "[":
                v, e = self.array(f"{path}[{index}]" if path is not None else None)
                values.append(v)
                elems.append(e[1] if e is not None else None)
            else:
                v, e = self.string() if c == '"' else self.literal()
                values.append(v)
                elems.append(e)
            index += 1
            self.ws()
            if self.t[self.i] == ",":
                self.i += 1
                continue
            self.expect("]")
            return values, ((start, self.i), elems)

    def member(self, path, parent):
        c = self.t[self.i]
        line = self.line_of(self.i)
        group = _json_group(path, parent)
        meta = meta_for(path)
        if c == "{":
            self.obj(path)
            return
        if c == "[":
            values, (span, elems) = self.array(path)
            scalars = all(isinstance(e, _Elem) for e in elems)
            rows = elems and all(
                isinstance(e, list) and e and all(isinstance(x, _Elem) for x in e)
                for e in elems
            )
            if scalars and all(
                e.kind in ("int", "float", "string", "bool") for e in elems
            ):
                self.params.append(
                    Param(
                        path,
                        values,
                        "list",
                        description=meta.description,
                        group=group,
                        unit=meta.unit,
                        line=line,
                        _elems=elems,
                        _container=span,
                    )
                )
            elif rows and all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for r in values
                for v in r
            ):
                self.params.append(
                    Param(
                        path,
                        values,
                        "matrix",
                        description=meta.description,
                        group=group,
                        unit=meta.unit,
                        line=line,
                        _elems=elems,
                        _container=span,
                    )
                )
            return
        value, elem = self.string() if c == '"' else self.literal()
        value, ptype, meta, choices = _typed(path, value, elem.kind, elem)
        self.params.append(
            Param(
                path,
                value,
                ptype,
                description=meta.description,
                group=group,
                choices=choices,
                minimum=meta.minimum,
                maximum=meta.maximum,
                unit=meta.unit,
                line=line,
                readonly=elem.kind == "null",
                _elems=elem,
            )
        )


def _json_group(path, parent):
    if parent:
        return parent
    m = re.match(r"((?:aux_)?[A-Za-z]+\d+)_", path)
    return m.group(1) if m else "General"


def _parse_json(text):
    parser = _Json(text)
    parser.ws()
    if parser.i < len(text) and text[parser.i] == "[":
        parser.array("")
    else:
        parser.obj(None)
    parser.ws()
    if parser.i != len(text):
        raise ValueError(
            f"unexpected text after JSON document at line {parser.line_of(parser.i) + 1}"
        )
    # Friendlier group labels for arrays of named objects: cameras[0] -> "cameras[0] tracking_left".
    names = {
        p.key.rsplit(".", 1)[0]: p.value
        for p in parser.params
        if p.leaf == "name" and isinstance(p.value, str)
    }
    for p in parser.params:
        root = p.group
        if root in names:
            p.group = f"{root} ({names[root]})"
    return parser.params


# ---------------------------------------------------------------------------
# ParamSet: estimator_config.yaml + linked kalibr chains
# ---------------------------------------------------------------------------
LINK_KEYS = ("relative_config_imu", "relative_config_imucam")


class ParamSet:
    """Several ParamFiles edited as one. Display keys are unique across files."""

    def __init__(self, files):
        self.files = list(files)
        self._index()

    def _index(self):
        counts = {}
        for f in self.files:
            for p in f.params:
                counts[p.key] = counts.get(p.key, 0) + 1
        self._keys = {}
        self._params = []
        for f in self.files:
            for p in f.params:
                display = p.key if counts[p.key] == 1 else f"{f.name}:{p.key}"
                self._keys[display] = p
                self._params.append((display, p))

    @classmethod
    def load(cls, paths, reference=None):
        refs = list(reference) if reference else [None] * len(paths)
        return cls([ParamFile.load(p, reference=r) for p, r in zip(paths, refs)])

    @classmethod
    def from_estimator_config(cls, path, reference=None):
        """The estimator config plus its relative_config_* files.

        ``reference``: another estimator_config.yaml (e.g. the template a
        generated config came from) whose values become the defaults.
        """
        path = Path(path)
        main = ParamFile.load(path)
        ref_main = ParamFile.load(reference) if reference else None
        if ref_main is not None:
            main.apply_reference(ref_main)
        files = [main]
        for key in LINK_KEYS:
            name = main.value(key)
            if not isinstance(name, str) or not name:
                continue
            linked = (path.parent / name).resolve()
            if not linked.is_file() or any(
                f.path and f.path.resolve() == linked for f in files
            ):
                continue
            ref = None
            if ref_main is not None:
                ref_name = ref_main.value(key)
                candidate = (
                    Path(reference).parent / ref_name
                    if isinstance(ref_name, str)
                    else None
                )
                ref = (
                    candidate if candidate is not None and candidate.is_file() else None
                )
            files.append(ParamFile.load(linked, reference=ref))
        return cls(files)

    @property
    def params(self):
        return [p for _, p in self._params]

    def items(self):
        return list(self._params)

    def key_of(self, param):
        for display, p in self._params:
            if p is param:
                return display
        raise KeyError(param.key)

    def get(self, key):
        if key in self._keys:
            return self._keys[key]
        raise KeyError(f"no parameter {key!r}")

    def value(self, key, default=None):
        return self._keys[key].value if key in self._keys else default

    def set(self, key, value):
        p = self.get(key) if isinstance(key, str) else key
        p.value = p.coerce(value)
        return p

    def reset(self, key):
        p = self.get(key) if isinstance(key, str) else key
        p.value = p.default
        return p

    def reset_all(self):
        for f in self.files:
            f.reset_all()

    @property
    def modified(self):
        return [p for p in self.params if p.modified]

    @property
    def dirty(self):
        return [p for p in self.params if p.dirty]

    def has_unsaved(self):
        return any(f.has_unsaved() for f in self.files)

    def save(self):
        saved = [f.save() for f in self.files if f.has_unsaved()]
        self._index()
        return saved

    def revert(self):
        for f in self.files:
            f.revert()
        self._index()

    def file_for(self, path):
        path = Path(path).resolve()
        return next(
            (f for f in self.files if f.path and f.path.resolve() == path), None
        )

    def reindex(self):
        self._index()
