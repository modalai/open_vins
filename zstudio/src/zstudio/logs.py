"""ROS-free readers for voxl-logger directories and ROS 1 / ROS 2 bags.

``probe_log`` lists a log's channels (VOXL pipes from info.json, ROS topics
from bag metadata) without reading sensor data. ``open_log`` returns a
``LogSource`` whose channels load on demand: only included channels are read,
and each loaded channel is cached so including/excluding is cheap.

Times are float seconds relative to the log epoch (info.json
``start_time_monotonic_ns`` / the bag's first log time); integer nanosecond
stamps are kept for exact comparisons. Image pixels are never held in memory;
``CameraStream.picture(i)`` decodes one frame.

Bags are read with the pure-Python ``rosbags`` package (no sourced ROS needed);
it is imported only when a bag is opened.
"""

from dataclasses import dataclass, field
import fnmatch
import json
from pathlib import Path
import os
import re
import struct
import threading

import numpy as np

from .geometry import RigConfig, RigEdge, matrix_rpy, quat_xyzw_matrix, rpy_matrix
from .media import (
    BAYER_PATTERNS,
    decode_bytes,
    decode_file,
    ros_image_to_picture,
    rotate_intrinsics,
    rotate_picture,
)

STUDIO_FILE = "ov_studio.json"

ROSBAGS_HINT = "Reading bags requires the 'rosbags' Python package: pip install rosbags (rosdep key python3-rosbags-pip)."
IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
IMU_TYPE = "sensor_msgs/msg/Imu"
INFO_TYPE = "sensor_msgs/msg/CameraInfo"
TF_TYPE = "tf2_msgs/msg/TFMessage"
THERMAL_SENSORS = ("nyctis", "lepton", "boson", "flir")
GLOBAL_SHUTTER_SENSORS = ("ar0144", "ar0234", "ov7251", "ov9282", "ov9782")
# Stream suffixes voxl-camera-server appends to a camera's base name.
VOXL_PIPE_SUFFIXES = (
    "_misp_norm_ion",
    "_misp_norm",
    "_misp_grey",
    "_misp_color",
    "_small_encoded",
    "_large_encoded",
    "_encoded",
    "_snapshot",
    "_bayer",
    "_color",
    "_grey",
    "_ion",
)
_IMAGE_FILE = re.compile(r"^(\d+)\.([A-Za-z0-9]+)$")


class LogError(RuntimeError):
    pass


@dataclass
class ChannelInfo:
    """A pipe/topic as listed by the log, before any data is read."""

    name: str
    kind: str  # imu | camera | camera_info | tf | other
    type: str
    count: int = 0
    supported: bool = True
    reason: str = ""

    @property
    def default(self):
        return self.supported


@dataclass
class ImuStream:
    name: str
    data: np.ndarray  # (N, 8): t [s], wx wy wz [rad/s], ax ay az [m/s^2], temperature [C] (NaN if unavailable)
    stamps_ns: np.ndarray
    rows: np.ndarray = None  # VOXL CSV 'i'
    batch: np.ndarray = None  # VOXL batch_id
    seq: np.ndarray = None  # ROS 1 header.seq
    receive: np.ndarray = None  # bag log time [s]
    declared: int = None
    configured_rate: float = None
    frame: str = ""  # extrinsics child / ROS header.frame_id
    covariance_unavailable: bool = False
    kind: str = "imu"

    @property
    def times(self):
        return self.data[:, 0]

    @property
    def temperature_available(self):
        return bool(len(self.data)) and bool(np.isfinite(self.data[:, 7]).any())

    def __len__(self):
        return len(self.data)


@dataclass
class CameraStream:
    name: str
    times: np.ndarray
    stamps_ns: np.ndarray
    width: int = 0
    height: int = 0
    encoding: str = ""
    stamp_reference: str = (
        "header"  # start: start-of-exposure (VOXL MPA) | header: as published
    )
    frame_ids: np.ndarray = None  # VOXL frame_id or ROS 1 header.seq
    exposure: np.ndarray = None  # [s]
    gain: np.ndarray = None
    sensor: str = ""
    intrinsics: dict = None
    intrinsics_expected: str = ""
    intrinsics_problem: str = ""
    receive: np.ndarray = None
    declared: int = None
    skip: int = 0
    dropped_flag: bool = False
    encoded: bool = False
    encoded_file: str = ""
    frame: str = ""
    directory: str = ""
    files: list = None
    file_sizes: list = None
    orphans: list = field(default_factory=list)
    pattern: str = "RGGB"
    info_topic: str = ""
    info_size: tuple = None
    kind: str = "camera"
    camera: str = (
        ""  # physical camera this pipe/topic comes from (VOXL: voxl-camera-server name)
    )
    rotation: int = 0  # clockwise degrees applied to frames, size and intrinsics
    _load: object = field(default=None, repr=False)
    _payload: object = field(default=None, repr=False)
    _raw: dict = field(default=None, repr=False)

    def __len__(self):
        return len(self.times)

    @property
    def label(self):
        """'hires (hires_bayer)' when the pipe/topic name differs from the camera."""
        return (
            f"{self.camera} ({self.name})"
            if self.camera and self.camera != self.name
            else self.name
        )

    @property
    def raw_size(self):
        """Frame size as recorded, before any studio rotation."""
        raw = self._raw or {}
        return raw.get("width", self.width), raw.get("height", self.height)

    def set_rotation(self, degrees):
        """Rotate frames clockwise (0/90/180/270) like VOXL's en_rotate.

        Width/height and intrinsics follow, so tracking, conversion and VIO see a
        consistent virtual camera; the camera frame turns about its optical axis
        (see geometry.image_rotation), which the caller applies to the extrinsics.
        """
        degrees = int(degrees) % 360
        if degrees % 90:
            raise ValueError("camera rotation must be a multiple of 90 degrees")
        if self._raw is None:
            self._raw = {
                "width": self.width,
                "height": self.height,
                "intrinsics": self.intrinsics,
            }
        raw = self._raw
        self.rotation = degrees
        intrinsics, width, height = rotate_intrinsics(
            raw["intrinsics"], raw["width"], raw["height"], degrees
        )
        if raw["intrinsics"] is None:
            width, height = (
                (raw["height"], raw["width"])
                if degrees in (90, 270)
                else (raw["width"], raw["height"])
            )
        self.intrinsics, self.width, self.height = intrinsics, width, height

    def orient(self, picture):
        """Apply the studio rotation to a decoded frame of this camera."""
        return rotate_picture(picture, self.rotation) if self.rotation else picture

    @property
    def thermal(self):
        return self.sensor.lower().startswith(THERMAL_SENSORS)

    @property
    def global_shutter(self):
        return self.sensor.lower().startswith(GLOBAL_SHUTTER_SENSORS)

    @property
    def has_pictures(self):
        return self._load is not None and not self.encoded

    def picture(self, index):
        if not self.has_pictures:
            raise LogError(f"{self.name}: frames are not stored as individual images")
        return self.orient(self._load(int(index)))

    def preview(self, index, width):
        """A frame for display at about ``width`` pixels (cheaper decode where the format allows)."""
        if not self.has_pictures:
            raise LogError(f"{self.name}: frames are not stored as individual images")
        try:
            raw_width = self.raw_size[0 if self.rotation in (0, 180) else 1]
            return self.orient(
                self._load(int(index), preview_width=int(width) if width else 0)
                if width and width < raw_width
                else self._load(int(index))
            )
        except TypeError:  # loaders without a preview path (bags)
            return self.orient(self._load(int(index)))

    def payload(self, index):
        """Raw pixel payload for identity comparisons (no decoding)."""
        return self._payload(int(index)) if self._payload else None


def is_log_path(path):
    path = Path(path).expanduser()
    if path.is_dir():
        return (path / "info.json").is_file() or (path / "metadata.yaml").is_file()
    return (
        path.suffix.lower() in (".bag", ".db3", ".mcap") or path.name == "metadata.yaml"
    )


def _bag_root(path):
    path = Path(path).expanduser().resolve()
    if path.name == "metadata.yaml" or (
        path.suffix.lower() in (".db3", ".mcap")
        and (path.parent / "metadata.yaml").is_file()
    ):
        return path.parent
    return path


def select(channels, only="", exclude=""):
    """Default selection (supported channels) filtered by comma-separated globs."""
    patterns = [p.strip() for p in (only or "").split(",") if p.strip()]
    skips = [p.strip() for p in (exclude or "").split(",") if p.strip()]

    def match(name, globs):
        short = name.strip("/").split("/")[-1]
        return any(
            fnmatch.fnmatchcase(name, g)
            or fnmatch.fnmatchcase(short, g)
            or fnmatch.fnmatchcase(name.strip("/"), g.strip("/"))
            for g in globs
        )

    chosen = []
    for channel in channels:
        if not channel.supported:
            continue
        if patterns and not match(channel.name, patterns):
            continue
        if skips and match(channel.name, skips):
            continue
        chosen.append(channel.name)
    return chosen


def probe_log(path):
    """List channels without reading sensor data: (kind, name, [ChannelInfo])."""
    path = Path(path).expanduser()
    if path.is_dir() and (path / "info.json").is_file():
        return VoxlLog.probe(path)
    if is_log_path(path):
        return BagLog.probe(_bag_root(path))
    raise LogError(f"{path} is not a voxl-logger directory or a ROS bag")


def open_log(path, include=None, progress=None, cancelled=None):
    """Open a log; ``include`` lists channel names to load (None: every supported channel)."""
    path = Path(path).expanduser()
    if path.is_dir() and (path / "info.json").is_file():
        log = VoxlLog(path)
    elif is_log_path(path):
        log = BagLog(_bag_root(path))
    else:
        raise LogError(f"{path} is not a voxl-logger directory or a ROS bag")
    try:
        log.load(include, progress=progress, cancelled=cancelled)
    except BaseException:
        log.close()
        raise
    return log


class LogSource:
    kind = ""

    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name
        self.channels = []
        self.included = []
        self.meta = {}
        self.rig = RigConfig()
        self.epoch_ns = 0
        self.errors = {}  # channel name -> why it could not be read
        self.rotations = {}  # camera name -> clockwise degrees (studio edit)
        self.rig_overrides = {}  # child frame -> RigEdge replacing the configured one
        self._saved = ({}, {})
        self._cache = {}

    # -- channel selection -------------------------------------------------
    def channel(self, name):
        return next((c for c in self.channels if c.name == name), None)

    def load(self, include=None, progress=None, cancelled=None):
        """Load (or reuse) the named channels; unsupported names are ignored."""
        names = (
            [c.name for c in self.channels if c.supported]
            if include is None
            else list(include)
        )
        wanted = [c for c in self.channels if c.name in names and c.supported]
        missing = [
            c for c in wanted if c.name not in self._cache and c.name not in self.errors
        ]
        if missing:
            self._read(missing, progress, cancelled)
        self.included = [c.name for c in wanted]
        self._post_load()
        return self

    def _read(self, channels, progress, cancelled):
        raise NotImplementedError

    def _post_load(self):
        pass

    def camera_of(self, channel):
        """Physical camera of a pipe/topic name (itself when unknown)."""
        stream = self._cache.get(channel)
        if isinstance(stream, CameraStream) and stream.camera:
            return stream.camera
        return channel

    # -- studio orientation edits (camera rotation + extrinsics) -------------
    @property
    def overrides_path(self):
        """Sidecar with studio edits: inside a voxl-logger log, next to a bag."""
        if self.kind == "voxl":
            return self.path / STUDIO_FILE
        return self.path.parent / f"{self.path.name}.{STUDIO_FILE}"

    def _load_overrides(self):
        path = self.overrides_path
        self.rotations, self.rig_overrides = {}, {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # Keys are cameras; older sidecars keyed by pipe/topic are mapped to their camera.
                self.rotations = {
                    self.camera_of(str(k)): int(v) % 360
                    for k, v in data.get("rotations", {}).items()
                }
                for child, e in data.get("extrinsics", {}).items():
                    self.rig_overrides[str(child)] = RigEdge(
                        str(e["parent"]),
                        str(child),
                        rpy_matrix(*[float(v) for v in e["RPY_parent_to_child"]]),
                        np.array([float(v) for v in e["T_child_wrt_parent"]]),
                    )
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.errors[STUDIO_FILE] = f"ignored {path.name}: {error}"
        self._mark_saved()

    def _mark_saved(self):
        self._saved = (
            dict(self.rotations),
            {
                k: (v.parent, v.R.copy(), v.t.copy())
                for k, v in self.rig_overrides.items()
            },
        )

    def _compose(self, base):
        """rig = configured transforms + studio overrides; rotations onto cameras."""
        rig = RigConfig(
            base.source,
            list(base.edges),
            base.body,
            base.body_convention,
            list(base.notes),
        )
        for child, edge in self.rig_overrides.items():
            rig.edges = [e for e in rig.edges if e.child != child] + [edge]
        if self.rig_overrides:
            rig.source = (
                f"{base.source} + studio edits" if base.source else "studio edits"
            )
            if self.kind == "voxl" and not rig.body:
                rig.body, rig.body_convention = "body", "FRD"
        self.rig = rig if rig.edges else RigConfig()
        for cam in self.cameras:
            cam.set_rotation(self.rotations.get(cam.camera or cam.name, 0))
            frame = self.camera_frame(cam)
            if frame in self.rig.frames:
                cam.frame = frame

    def camera_frame(self, cam):
        """Rig frame that holds a camera's extrinsics (existing or the name to create)."""
        if cam.frame and cam.frame in self.rig.frames:
            return cam.frame
        if self.kind == "voxl":
            return (
                self.rig.match(cam.camera)
                or self.rig.match(cam.name)
                or cam.camera
                or voxl_base(cam.name)
            )
        return cam.frame or cam.name.strip("/").replace("/", "_")

    def set_orientation(self, cam, rotation, edge=None):
        """Studio edit: rotate a camera's frames and/or replace its extrinsic edge."""
        if rotation % 360:
            self.rotations[cam.camera or cam.name] = int(rotation) % 360
        else:
            self.rotations.pop(cam.camera or cam.name, None)
        if edge is not None:
            self.rig_overrides[edge.child] = edge
            cam.frame = edge.child
        self._post_load()

    def orientation_changes(self):
        """Unsaved studio edits: ({camera: (saved, now)}, {child: RigEdge})."""
        saved_rot, saved_edges = self._saved
        rotations = {
            name: (saved_rot.get(name, 0), self.rotations.get(name, 0))
            for name in set(saved_rot) | set(self.rotations)
            if saved_rot.get(name, 0) != self.rotations.get(name, 0)
        }
        edges = {}
        for child, edge in self.rig_overrides.items():
            old = saved_edges.get(child)
            if (
                old is None
                or old[0] != edge.parent
                or not np.allclose(old[1], edge.R)
                or not np.allclose(old[2], edge.t)
            ):
                edges[child] = edge
        return rotations, edges

    def revert_orientation(self):
        saved_rot, saved_edges = self._saved
        self.rotations = dict(saved_rot)
        self.rig_overrides = {
            k: RigEdge(p, k, R.copy(), t.copy()) for k, (p, R, t) in saved_edges.items()
        }
        self._post_load()

    def save_orientation(self):
        """Persist studio edits. Returns [(path, backup or None)] of written files."""
        written = []
        _, edges = self.orientation_changes()
        if self.kind == "voxl" and edges:
            written.append(self._write_voxl_extrinsics(list(edges.values())))
            for child in edges:
                self.rig_overrides.pop(child, None)
            self._config()
        data = {
            "about": "ov_zcalib Studio edits: camera rotations (clockwise degrees, applied to frames, size and "
            "intrinsics) and extrinsics overrides (VOXL convention).",
            "rotations": {k: v for k, v in sorted(self.rotations.items()) if v},
            "extrinsics": {
                child: {
                    "parent": e.parent,
                    "RPY_parent_to_child": [_tidy(v) for v in matrix_rpy(e.R)],
                    "T_child_wrt_parent": [_tidy(v) for v in e.t],
                }
                for child, e in sorted(self.rig_overrides.items())
            },
        }
        path = self.overrides_path
        if data["rotations"] or data["extrinsics"]:
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            written.append((path, None))
        elif path.is_file():
            path.unlink()
            written.append((path, None))
        self._mark_saved()
        self._post_load()
        return written

    @property
    def imus(self):
        return [
            self._cache[n]
            for n in self.included
            if isinstance(self._cache.get(n), ImuStream)
        ]

    @property
    def cameras(self):
        return [
            self._cache[n]
            for n in self.included
            if isinstance(self._cache.get(n), CameraStream)
        ]

    @property
    def streams(self):
        return self.imus + self.cameras

    @property
    def start(self):
        starts = [float(s.times[0]) for s in self.streams if len(s.times)]
        return min(starts, default=0.0)

    @property
    def end(self):
        ends = [float(s.times[-1]) for s in self.streams if len(s.times)]
        return max(ends, default=self.start)

    @property
    def duration(self):
        return max(0.0, self.end - self.start)

    @property
    def format_label(self):
        return {"voxl": "voxl-logger", "ros1": "ROS 1 bag", "ros2": "ROS 2 bag"}.get(
            self.kind, self.kind
        )

    def close(self):
        pass


# ---------------------------------------------------------------------------
# voxl-logger
# ---------------------------------------------------------------------------
def _read_json(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        # VOXL configs open with a /* */ header; hand edits add // comments and
        # trailing commas. '//' is only a comment at line start or after
        # whitespace, so URLs inside strings survive.
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r"(?m)(^|\s)//.*$", r"\1", text)
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        return json.loads(text)


def _read_csv(path):
    """(lower-case stripped header, float64 array, int64 first timestamp column)."""
    with open(path, encoding="utf-8", errors="replace") as stream:
        header = [c.strip() for c in stream.readline().split(",")]
        header = [c for c in header if c]
        rows = [line for line in stream if line.strip()]
    if not rows:
        return header, np.empty((0, len(header))), np.empty(0, np.int64)
    width = len(header)
    try:
        values = np.loadtxt(
            rows, delimiter=",", usecols=range(width), ndmin=2, dtype=np.float64
        )
    except ValueError:
        values = np.array(
            [
                [
                    float(x) if x.strip() else np.nan
                    for x in (r.rstrip("\n").split(",") + [""] * width)[:width]
                ]
                for r in rows
            ]
        )
    tcol = _col(header, "timestamp")
    stamps = np.empty(0, np.int64)
    if tcol is not None:
        stamps = np.array([int(r.split(",")[tcol]) for r in rows], dtype=np.int64)
    return header, values, stamps


def _col(header, prefix):
    for i, name in enumerate(header):
        if name.lower().startswith(prefix.lower()):
            return i
    return None


def voxl_base(pipe):
    for suffix in VOXL_PIPE_SUFFIXES:
        if pipe.endswith(suffix) and len(pipe) > len(suffix):
            return pipe[: -len(suffix)]
    return pipe


def read_opencv_yaml(path):
    """Minimal reader for OpenCV FileStorage YAML camera calibrations."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out = {}
    for m in re.finditer(
        r"(?m)^(\w+):\s*!!opencv-matrix\s*\n\s*rows:\s*(\d+)\s*\n\s*cols:\s*(\d+)\s*\n\s*dt:\s*\w+\s*\n\s*data:\s*\[([^\]]*)\]",
        text,
    ):
        values = [
            float(v) for v in m.group(4).replace("\n", " ").split(",") if v.strip()
        ]
        out[m.group(1)] = np.array(values).reshape(int(m.group(2)), int(m.group(3)))
    for m in re.finditer(r"(?m)^(\w+):\s*([^!\n][^\n]*)$", text):
        if m.group(1) in out:
            continue
        value = m.group(2).strip().strip('"')
        try:
            out[m.group(1)] = (
                float(value) if re.fullmatch(r"[-+0-9.eE]+", value) else value
            )
        except ValueError:
            out[m.group(1)] = value
    return out


def voxl_intrinsics(path):
    data = read_opencv_yaml(path)
    M, D = data.get("M"), data.get("D")
    if M is None or D is None or M.shape != (3, 3) or D.size < 4:
        raise LogError(f"malformed intrinsics {Path(path).name}")
    model = str(data.get("distortion_model", "")).lower()
    return {
        "k": [float(M[0, 0]), float(M[1, 1]), float(M[0, 2]), float(M[1, 2])],
        "d": [float(v) for v in D.ravel()],
        "model": "equidistant" if model in ("fisheye", "equidistant") else "plumb_bob",
        "width": int(data.get("width", 0) or 0),
        "height": int(data.get("height", 0) or 0),
        "reprojection_error": data.get("reprojection_error"),
        "source": str(path),
    }


class VoxlLog(LogSource):
    kind = "voxl"

    @staticmethod
    def probe(path):
        info = _read_json(Path(path) / "info.json")
        channels = []
        for ch in info.get("channels", []):
            name = (
                Path(str(ch.get("pipe_path", "")).rstrip("/")).name
                or f"channel{ch.get('channel')}"
            )
            typ = str(ch.get("type_string", ""))
            kind = {"imu": "imu", "cam": "camera"}.get(typ, "other")
            channels.append(
                ChannelInfo(
                    name,
                    kind,
                    typ,
                    int(ch.get("n_samples", 0) or 0),
                    kind != "other",
                    "" if kind != "other" else f"'{typ}' channels are not inspected",
                )
            )
        return "voxl", Path(path).name, channels

    def __init__(self, path):
        super().__init__(path)
        info_path = self.path / "info.json"
        try:
            self.info = _read_json(info_path)
        except (OSError, ValueError) as error:
            raise LogError(f"cannot read {info_path}: {error}") from error
        _, _, self.channels = self.probe(self.path)
        self._raw = {
            Path(str(c.get("pipe_path", "")).rstrip("/")).name: c
            for c in self.info.get("channels", [])
        }
        self.epoch_ns = int(self.info.get("start_time_monotonic_ns", 0) or 0)
        self.meta = {
            "note": self.info.get("note", ""),
            "declared_duration": float(self.info.get("duration_s", 0) or 0),
            "start_date": self.info.get("start_time_date", ""),
            "format_version": self.info.get("log_format_version"),
        }
        self._config()
        self._load_overrides()

    # -- configuration snapshot -------------------------------------------
    def _config(self):
        etc = self.path / "etc" / "modalai"
        self.config_files = (
            sorted(p.name for p in etc.glob("*.conf")) if etc.is_dir() else []
        )
        self.config_errors = []
        self.sensor_types = {}
        self.vio_cams = []
        self.imu_rates = {}

        def load(name):
            path = etc / name
            if not path.is_file():
                return None
            try:
                return _read_json(path)
            except (OSError, ValueError) as error:
                self.config_errors.append(f"{name}: {error}")
                return None

        cams = load("voxl-camera-server.conf") or {}
        for cam in cams.get("cameras", []) if isinstance(cams, dict) else []:
            if isinstance(cam, dict) and cam.get("name"):
                self.sensor_types[str(cam["name"])] = str(cam.get("type", ""))
        vio = load("vio_cams.conf") or {}
        self.vio_cams = (
            [c for c in vio.get("cams", []) if isinstance(c, dict)]
            if isinstance(vio, dict)
            else []
        )
        imu = load("voxl-imu-server.conf") or {}
        if isinstance(imu, dict):
            for key, value in imu.items():
                m = re.fullmatch(r"(?:aux_)?imu(\d+)_sample_rate_hz", key)
                if m and isinstance(value, (int, float)):
                    self.imu_rates[int(m.group(1))] = float(value)
        ext = load("extrinsics.conf")
        rig = RigConfig(source="extrinsics.conf", body="body", body_convention="FRD")
        if isinstance(ext, dict):
            rig.notes.append(str(ext.get("name", "")))
            for e in ext.get("extrinsics", []):
                try:
                    rig.edges.append(
                        RigEdge(
                            str(e["parent"]),
                            str(e["child"]),
                            rpy_matrix(*[float(v) for v in e["RPY_parent_to_child"]]),
                            np.array([float(v) for v in e["T_child_wrt_parent"]]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    self.config_errors.append(
                        f"extrinsics.conf: malformed entry {e!r:.80}"
                    )
        self.base_rig = rig
        self.rig = rig if rig.edges else RigConfig()

    def _post_load(self):
        self._compose(self.base_rig)

    def _write_voxl_extrinsics(self, edges):
        """Update/append extrinsics.conf entries in place (format-preserving), keeping a backup."""
        from datetime import datetime

        from .params import ParamFile

        etc = self.path / "etc" / "modalai"
        path = etc / "extrinsics.conf"
        backup = None
        entries = [
            {
                "parent": e.parent,
                "child": e.child,
                "T_child_wrt_parent": [_tidy(v) for v in e.t],
                "RPY_parent_to_child": [_tidy(v) for v in matrix_rpy(e.R)],
            }
            for e in edges
        ]
        if path.is_file():
            backup = path.with_name(
                f"extrinsics.conf.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            )
            backup.write_bytes(path.read_bytes())
            conf = ParamFile.load(path)
            index = {
                conf.value(p.key): int(p.key.split("[")[1].split("]")[0])
                for p in conf.params
                if p.key.endswith("].child")
            }
            new = []
            for entry in entries:
                i = index.get(entry["child"])
                if i is None:
                    new.append(entry)
                    continue
                conf.set(f"extrinsics[{i}].parent", entry["parent"])
                conf.set(
                    f"extrinsics[{i}].T_child_wrt_parent", entry["T_child_wrt_parent"]
                )
                conf.set(
                    f"extrinsics[{i}].RPY_parent_to_child", entry["RPY_parent_to_child"]
                )
            conf.save()
            if new:
                text = path.read_text(encoding="utf-8")
                header = re.match(r"\s*/\*.*?\*/\s*", text, re.S)
                data = _read_json(path)
                data.setdefault("extrinsics", []).extend(new)
                path.write_text(
                    (header.group(0) if header else "")
                    + json.dumps(data, indent="\t")
                    + "\n",
                    encoding="utf-8",
                )
        else:
            etc.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"name": "ov_zcalib Studio", "extrinsics": entries}, indent="\t"
                )
                + "\n",
                encoding="utf-8",
            )
        return path, backup

    def camera_of(self, channel):
        """voxl-camera-server camera publishing a pipe (hires_bayer -> hires)."""
        best = ""
        for name in self.sensor_types:
            if channel.startswith(name) and len(name) > len(best):
                best = name
        if best:
            return best
        return (
            voxl_base(channel)
            if any(c.name == channel for c in self.channels)
            else channel
        )

    def sensor_for(self, pipe):
        best = ""
        for name in self.sensor_types:
            if pipe.startswith(name) and len(name) > len(best):
                best = name
        return self.sensor_types.get(best, "")

    def intrinsics_path(self, pipe):
        """Expected per-unit calibration path for a pipe (vio_cams.conf cal_file wins)."""
        base = voxl_base(pipe)
        for cam in self.vio_cams:
            pipes = {
                str(cam.get(k, ""))
                for k in ("name", "pipe_for_preview", "pipe_for_tracking")
            }
            if pipe in pipes or base in pipes:
                cal = str(cam.get("cal_file", "")).strip()
                if cal:
                    return self.path / "data" / "modalai" / cal
        return self.path / "data" / "modalai" / f"opencv_{base}_intrinsics.yml"

    def is_vio_camera(self, pipe):
        base = voxl_base(pipe)
        for cam in self.vio_cams:
            if cam.get("enable", True) is False:
                continue
            pipes = {
                str(cam.get(k, ""))
                for k in ("name", "pipe_for_preview", "pipe_for_tracking")
            }
            if pipe in pipes or base in pipes:
                return True
        return False

    def configured_imu_rate(self, name):
        m = re.fullmatch(r"imu(\d+)", name)
        index = int(m.group(1)) if m else (0 if name == "imu_apps" else None)
        return self.imu_rates.get(index) if index is not None else None

    # -- channel data ---------------------------------------------------
    def _read(self, channels, progress, cancelled):
        for n, channel in enumerate(channels):
            if cancelled and cancelled():
                raise InterruptedError("log loading cancelled")
            raw = self._raw.get(channel.name, {})
            directory = self.path / str(raw.get("pipe_path", channel.name)).strip("/")
            try:
                if not directory.is_dir():
                    raise LogError(f"listed in info.json but {directory} is missing")
                if channel.kind == "imu":
                    self._cache[channel.name] = self._imu(channel, raw, directory)
                else:
                    self._cache[channel.name] = self._camera(channel, raw, directory)
            except (LogError, OSError, ValueError) as error:
                self.errors[channel.name] = str(error)
            if progress:
                progress((n + 1) / len(channels), channel.name)

    def _seconds(self, stamps):
        return (stamps - self.epoch_ns).astype(np.float64) * 1e-9

    def _imu(self, channel, raw, directory):
        csv = directory / "data.csv"
        if not csv.is_file():
            raise LogError(f"{channel.name}: data.csv missing in {directory}")
        header, values, stamps = _read_csv(csv)
        cols = [_col(header, k) for k in ("gx", "gy", "gz", "ax", "ay", "az")]
        if any(c is None for c in cols) or not len(stamps):
            raise LogError(
                f"{channel.name}: IMU CSV lacks timestamp/gyro/accel columns"
            )
        tcol = (
            _col(header, "t(")
            if _col(header, "t(") is not None
            else _col(header, "temp")
        )
        data = np.empty((len(values), 8))
        data[:, 0] = self._seconds(stamps)
        data[:, 1:7] = values[:, cols]
        data[:, 7] = values[:, tcol] if tcol is not None else np.nan
        icol, bcol = _col(header, "i"), _col(header, "batch_id")
        return ImuStream(
            channel.name,
            data,
            stamps,
            rows=values[:, icol].astype(np.int64)
            if icol == 0 and header[0].lower() == "i"
            else None,
            batch=values[:, bcol].astype(np.int64) if bcol is not None else None,
            declared=int(raw["n_samples"]) if "n_samples" in raw else None,
            configured_rate=self.configured_imu_rate(channel.name),
            frame=self.rig.match(channel.name) or channel.name,
        )

    def _camera(self, channel, raw, directory):
        csv = directory / "data.csv"
        if not csv.is_file():
            raise LogError(f"{channel.name}: data.csv missing in {directory}")
        header, values, stamps = _read_csv(csv)

        def column(prefix, dtype=np.float64):
            c = _col(header, prefix)
            return values[:, c].astype(dtype) if c is not None else None

        sensor = self.sensor_for(channel.name) or self.sensor_for(
            voxl_base(channel.name)
        )
        encoded = any(h.lower().startswith("offset") for h in header)
        rows = (
            column("i", np.int64)
            if header and header[0].lower() == "i"
            else np.arange(len(values))
        )
        exposure = column("exposure")
        width = column("width", np.int64)
        height = column("height", np.int64)
        cam = CameraStream(
            channel.name,
            self._seconds(stamps),
            stamps,
            width=int(width[0]) if width is not None and len(width) else 0,
            height=int(height[0]) if height is not None and len(height) else 0,
            stamp_reference="start",
            camera=self.camera_of(channel.name),
            frame_ids=column("frame_id", np.int64),
            exposure=exposure * 1e-9 if exposure is not None else None,
            gain=column("gain"),
            sensor=sensor,
            declared=int(raw["n_samples"]) if "n_samples" in raw else None,
            skip=int(raw.get("n_to_skip", 0) or 0),
            dropped_flag=bool(raw.get("dropped_data", False)),
            encoded=encoded,
            directory=str(directory),
            frame=self.rig.match(channel.name)
            or self.rig.match(voxl_base(channel.name))
            or "",
        )
        fmt = column("format", np.int64)
        cam.encoding = f"format {int(fmt[0])}" if fmt is not None and len(fmt) else ""
        if encoded:
            videos = sorted(p.name for p in directory.glob("video.*"))
            cam.encoded_file = videos[0] if videos else ""
            cam.encoding = (
                f"encoded {Path(cam.encoded_file).suffix.lstrip('.') or 'stream'}"
            )
            return cam
        listing = {}
        with os.scandir(directory) as entries:
            for entry in entries:
                m = _IMAGE_FILE.match(entry.name)
                if m and entry.is_file():
                    listing[int(m.group(1))] = (entry.name, entry.stat().st_size)
        cam.files = [listing.get(int(r), (None, 0))[0] for r in rows]
        cam.file_sizes = [listing.get(int(r), (None, 0))[1] for r in rows]
        referenced = set(int(r) for r in rows)
        cam.orphans = sorted(k for k in listing if k not in referenced)
        exts = {Path(f).suffix.lower() for f in cam.files if f}
        cam.encoding = ", ".join(sorted(e.lstrip(".") for e in exts)) + (
            f" · {cam.encoding}" if cam.encoding else ""
        )
        expected = self.intrinsics_path(channel.name)
        cam.intrinsics_expected = str(expected)
        if expected.is_file():
            try:
                cam.intrinsics = voxl_intrinsics(expected)
            except (OSError, LogError) as error:
                cam.intrinsics_problem = str(error)

        def load(index, preview_width=0, cam=cam):
            name = cam.files[index]
            if not name:
                raise LogError(f"{cam.name}: row {index} has no image file")
            width, height = (
                cam.raw_size
            )  # files are stored as recorded, before any rotation
            return decode_file(
                directory / name,
                width,
                height,
                cam.pattern,
                preview_width=preview_width,
            )

        def payload(index, cam=cam):
            name = cam.files[index]
            return (directory / name).read_bytes() if name else None

        cam._load, cam._payload = load, payload
        return cam


# ---------------------------------------------------------------------------
# ROS bags (rosbags)
# ---------------------------------------------------------------------------
def _rosbags():
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise LogError(ROSBAGS_HINT) from error
    return AnyReader, Stores, get_typestore


def _header_from_raw(raw, ros1):
    """(stamp_ns, seq) from a serialized std_msgs/Header-first message."""
    if ros1:
        seq, sec, nsec = struct.unpack_from("<III", raw, 0)
        return sec * 1_000_000_000 + nsec, seq
    endian = "<" if raw[1] == 1 else ">"
    sec, nsec = struct.unpack_from(endian + "iI", raw, 4)
    return sec * 1_000_000_000 + nsec, None


def _stamp_ns(stamp):
    if hasattr(stamp, "sec"):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)


class BagLog(LogSource):
    @staticmethod
    def _reader(path):
        AnyReader, Stores, get_typestore = _rosbags()
        try:
            reader = AnyReader(
                [Path(path)], default_typestore=get_typestore(Stores.ROS2_JAZZY)
            )
            reader.open()
        except Exception as error:  # rosbags raises several unrelated types
            raise LogError(f"cannot open bag {path}: {error}") from error
        return reader

    @staticmethod
    def _channels(reader):
        topics = {}
        for conn in reader.connections:
            entry = topics.setdefault(conn.topic, [conn.msgtype, 0])
            entry[1] += int(conn.msgcount or 0)
        channels = []
        for topic, (msgtype, count) in sorted(topics.items()):
            if msgtype == IMU_TYPE:
                kind, ok, why = "imu", True, ""
            elif msgtype in IMAGE_TYPES:
                kind, ok, why = "camera", True, ""
            elif msgtype == INFO_TYPE:
                kind, ok, why = "camera_info", True, ""
            elif msgtype == TF_TYPE and topic.endswith("tf_static"):
                kind, ok, why = "tf", True, ""
            else:
                kind, ok, why = "other", False, f"{msgtype} is not inspected"
            channels.append(ChannelInfo(topic, kind, msgtype, count, ok, why))
        return channels

    @classmethod
    def probe(cls, path):
        reader = cls._reader(path)
        try:
            kind = "ros1" if getattr(reader, "is2", None) is False else "ros2"
            return kind, Path(path).name, cls._channels(reader)
        finally:
            reader.close()

    def __init__(self, path):
        super().__init__(path)
        self.reader = self._reader(self.path)
        self.lock = threading.RLock()
        self.kind = "ros1" if getattr(self.reader, "is2", None) is False else "ros2"
        self.channels = self._channels(self.reader)
        self.epoch_ns = int(self.reader.start_time)
        self._load_overrides()
        self.meta = {
            "declared_duration": (
                int(self.reader.end_time) - int(self.reader.start_time)
            )
            * 1e-9
        }
        self._infos = {}

    def close(self):
        with self.lock:
            if self.reader is not None:
                try:
                    self.reader.close()
                finally:
                    self.reader = None

    def _connections(self, topic):
        return [c for c in self.reader.connections if c.topic == topic]

    def _seconds(self, stamps):
        return (np.asarray(stamps, dtype=np.int64) - self.epoch_ns).astype(
            np.float64
        ) * 1e-9

    def _read(self, channels, progress, cancelled):
        ros1 = self.kind == "ros1"
        order = {"tf": 0, "camera_info": 1, "imu": 2, "camera": 3}
        channels = sorted(channels, key=lambda c: order.get(c.kind, 4))
        total = sum(max(1, c.count) for c in channels)
        done = 0
        for channel in channels:
            conns = self._connections(channel.name)
            state = {"stamps": [], "seq": [], "log": [], "first": None}
            imu_rows, imu_flags = [], []
            with self.lock:
                for n, (conn, logtime, raw) in enumerate(
                    self.reader.messages(connections=conns)
                ):
                    if cancelled and n % 256 == 0 and cancelled():
                        raise InterruptedError("log loading cancelled")
                    if channel.kind == "camera":
                        stamp, seq = _header_from_raw(raw, ros1)
                        if state["first"] is None:
                            state["first"] = self.reader.deserialize(raw, conn.msgtype)
                    else:
                        msg = self.reader.deserialize(raw, conn.msgtype)
                        if channel.kind in ("camera_info", "tf"):
                            state.setdefault("msgs", []).append(msg)
                            if channel.kind == "camera_info" and len(state["msgs"]) > 1:
                                state["msgs"] = state["msgs"][:1]
                        stamp = (
                            _stamp_ns(msg.header.stamp)
                            if hasattr(msg, "header")
                            else logtime
                        )
                        seq = (
                            getattr(msg.header, "seq", None)
                            if hasattr(msg, "header")
                            else None
                        )
                        if channel.kind == "imu":
                            if state["first"] is None:
                                state["first"] = msg
                            a, w = msg.linear_acceleration, msg.angular_velocity
                            imu_rows.append((w.x, w.y, w.z, a.x, a.y, a.z))
                            imu_flags.append(
                                msg.angular_velocity_covariance[0] == -1
                                or msg.linear_acceleration_covariance[0] == -1
                            )
                    state["stamps"].append(stamp)
                    state["seq"].append(seq if seq is not None else -1)
                    state["log"].append(logtime)
                    if progress and n % 512 == 0:
                        progress(min(1.0, (done + n) / total), channel.name)
            done += max(1, channel.count)
            stamps = np.array(state["stamps"], dtype=np.int64)
            receive = self._seconds(state["log"])
            seqs = np.array(state["seq"], dtype=np.int64) if ros1 else None
            if channel.kind == "imu":
                data = np.empty((len(stamps), 8))
                data[:, 0] = self._seconds(stamps)
                data[:, 1:7] = np.array(imu_rows, dtype=np.float64).reshape(-1, 6)
                data[:, 7] = np.nan  # sensor_msgs/Imu carries no temperature
                first = state["first"]
                self._cache[channel.name] = ImuStream(
                    channel.name,
                    data,
                    stamps,
                    seq=seqs,
                    receive=receive,
                    declared=channel.count,
                    frame=str(first.header.frame_id) if first is not None else "",
                    covariance_unavailable=bool(imu_flags) and bool(np.any(imu_flags)),
                )
            elif channel.kind == "camera":
                self._cache[channel.name] = self._camera(
                    channel,
                    conns,
                    stamps,
                    seqs,
                    receive,
                    np.array(state["log"], np.int64),
                    state["first"],
                )
            elif channel.kind == "camera_info":
                self._infos[channel.name] = state.get("msgs", [None])[0]
                self._cache[channel.name] = {
                    "kind": "camera_info",
                    "count": len(stamps),
                }
            elif channel.kind == "tf":
                self._cache[channel.name] = {
                    "kind": "tf",
                    "msgs": state.get("msgs", []),
                }
            if progress:
                progress(min(1.0, done / total), channel.name)

    def _camera(self, channel, conns, stamps, seqs, receive, logtimes, first):
        compressed = channel.type.endswith("CompressedImage")
        cam = CameraStream(
            channel.name,
            self._seconds(stamps),
            stamps,
            stamp_reference="header",
            camera=(str(first.header.frame_id).strip("/") if first is not None else "")
            or channel.name.strip("/").split("/")[0],
            frame_ids=seqs,
            receive=receive,
            declared=channel.count,
            frame=str(first.header.frame_id) if first is not None else "",
        )
        if first is not None:
            if compressed:
                cam.encoding = f"compressed {first.format}".strip()
                try:
                    picture = decode_bytes(first.data)
                    cam.width, cam.height = picture.width, picture.height
                except (
                    Exception
                ):  # dimensions stay unknown; viewer reports per-frame errors
                    pass
            else:
                cam.encoding = str(first.encoding)
                cam.width, cam.height = int(first.width), int(first.height)
                if (
                    cam.encoding.lower().startswith("bayer_")
                    and cam.encoding[6:10].upper() in BAYER_PATTERNS
                ):
                    cam.pattern = cam.encoding[6:10].upper()
        # Occurrence index disambiguates messages sharing one log time.
        occurrence = np.zeros(len(logtimes), dtype=np.int64)
        if len(logtimes) > 1:
            same = np.concatenate(([False], logtimes[1:] == logtimes[:-1]))
            run = 0
            for i, s in enumerate(same):
                run = run + 1 if s else 0
                occurrence[i] = run

        def message(index):
            with self.lock:
                if self.reader is None:
                    raise LogError("bag is closed")
                t = int(logtimes[index])
                seen = 0
                for conn, _, raw in self.reader.messages(
                    connections=conns, start=t, stop=t + 1
                ):
                    if seen == occurrence[index]:
                        return self.reader.deserialize(raw, conn.msgtype)
                    seen += 1
            raise LogError(f"{channel.name}: message {index} not found")

        def load(index):
            msg = message(index)
            return decode_bytes(msg.data) if compressed else ros_image_to_picture(msg)

        def payload(index):
            return bytes(message(index).data)

        cam._load, cam._payload = load, payload
        cam._logtimes, cam._conns = logtimes, conns
        return cam

    def iter_messages(self, topics):
        """Sequential (topic, logtime_ns, message) over topics in log-time order."""
        conns = [c for c in self.reader.connections if c.topic in set(topics)]
        with self.lock:
            for conn, logtime, raw in self.reader.messages(connections=conns):
                yield conn.topic, logtime, self.reader.deserialize(raw, conn.msgtype)

    def _post_load(self):
        # CameraInfo pairing: the CameraInfo topic sharing the image's namespace.
        infos = {
            name: msg for name, msg in self._infos.items() if name in self.included
        }
        for cam in self.cameras:
            cam.info_topic, cam.intrinsics, cam.intrinsics_problem, cam.info_size = (
                "",
                None,
                "",
                None,
            )
            namespace = cam.name.rsplit("/", 1)[0]
            candidates = [t for t in infos if t.rsplit("/", 1)[0] == namespace]
            if not candidates:
                parent = namespace.rsplit("/", 1)[0]
                candidates = [
                    t for t in infos if t.rsplit("/", 1)[0] in (parent, namespace)
                ]
            if not candidates:
                continue
            topic = sorted(
                candidates, key=lambda t: (not t.endswith("camera_info"), t)
            )[0]
            msg = infos[topic]
            cam.info_topic = topic
            if msg is None:
                continue
            # ROS 2 spells the arrays k/d; ROS 1 (and rosbags' ROS 1 types) K/D.
            K = list(msg.k if hasattr(msg, "k") else msg.K)
            D = [float(v) for v in (msg.d if hasattr(msg, "d") else msg.D)]
            cam.info_size = (int(msg.width), int(msg.height))
            model = str(msg.distortion_model)
            if K[0] <= 0 or K[4] <= 0:
                cam.intrinsics_problem = f"{topic} is uncalibrated (K is zero)"
            elif model not in ("plumb_bob", "equidistant", "rational_polynomial"):
                cam.intrinsics_problem = (
                    f"{topic}: distortion model '{model}' is not supported"
                )
            else:
                cam.intrinsics = {
                    "k": [float(K[0]), float(K[4]), float(K[2]), float(K[5])],
                    "d": D,
                    "model": "equidistant" if model == "equidistant" else "plumb_bob",
                    "ros_model": model,
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "source": topic,
                }
        rig = RigConfig(source="/tf_static")
        for name in self.included:
            entry = self._cache.get(name)
            if isinstance(entry, dict) and entry.get("kind") == "tf":
                for msg in entry["msgs"]:
                    for tf in msg.transforms:
                        tr, q = tf.transform.translation, tf.transform.rotation
                        rig.edges.append(
                            RigEdge(
                                str(tf.header.frame_id).lstrip("/"),
                                str(tf.child_frame_id).lstrip("/"),
                                quat_xyzw_matrix([q.x, q.y, q.z, q.w]),
                                np.array([tr.x, tr.y, tr.z]),
                            )
                        )
        if "base_link" in rig.frames:
            rig.body, rig.body_convention = "base_link", "FLU"
        for stream in self.streams:
            stream.frame = stream.frame.lstrip("/")
        self._compose(rig)


def _tidy(value):
    """Config-friendly number: integers stay integers, others keep 6 decimals."""
    value = float(value)
    rounded = round(value, 6)
    return int(round(rounded)) if abs(rounded - round(rounded)) < 1e-9 else rounded
