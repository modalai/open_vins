"""ROS-independent session reader. Matches SessionRecord.cpp's packed LE contract."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import struct

import numpy as np

POINT_DTYPE = np.dtype([("id", "<u4"), ("u", "<f4"), ("v", "<f4")])
CAMERA_STRUCT = struct.Struct("<8dB4d3d2dIIdB")
IMU_STRUCT = struct.Struct("<8d")
FRAME_STRUCT = struct.Struct("<dffIII")


@dataclass
class Camera:
    width: int = 640
    height: int = 480
    k: list = field(default_factory=lambda: [450.0, 450.0, 320.0, 240.0])
    d: list = field(default_factory=lambda: [0.0] * 4)
    q: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    p: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    td: float = 0.0
    tr: float = 0.0
    fps: float = 0.0
    fisheye: bool = False
    rolling: bool = False
    sigma: float = 0.0


@dataclass
class Frame:
    timestamp: float
    points: np.ndarray
    cam: int = 0
    seq: int = 0
    exposure: float = 0.0
    temperature: float = 0.0
    image: object = None


@dataclass
class Session:
    name: str
    cameras: list
    imu: np.ndarray
    frames: list
    path: str = ""
    profile: int = 0
    cam_mode: int = -1
    format: int = 7
    complete: bool = True
    demo: bool = False
    log: object = None  # LogSource when the session views a voxl-logger log or a bag

    def __post_init__(self):
        if len(self.imu) and np.any(np.diff(self.imu[:, 0]) < 0):
            self.imu = self.imu[np.argsort(self.imu[:, 0], kind="stable")]
        self.frames.sort(key=lambda frame: frame.timestamp)
        self.by_camera = [
            [f for f in self.frames if f.cam == i] for i in range(len(self.cameras))
        ]
        self.frame_times = [
            np.array([f.timestamp for f in frames]) for frames in self.by_camera
        ]
        self.camera_rates = [rate(times) for times in self.frame_times]
        self.imu_rate = rate(self.imu[:, 0]) if len(self.imu) else 0.0
        starts = ([float(self.imu[0, 0])] if len(self.imu) else []) + (
            [self.frames[0].timestamp] if self.frames else []
        )
        ends = ([float(self.imu[-1, 0])] if len(self.imu) else []) + (
            [self.frames[-1].timestamp] if self.frames else []
        )
        self.start = min(starts, default=0.0)
        self.duration = max(ends, default=self.start) - self.start

    def frame_at(self, offset, cam=0):
        frames, times = self.by_camera[cam], self.frame_times[cam]
        index = int(np.searchsorted(times, self.start + offset, side="right")) - 1
        return frames[index] if index >= 0 else None

    def imu_window(self, offset, span=8.0):
        if not len(self.imu):
            return np.empty((0, 8))
        end = self.start + offset
        lo, hi = np.searchsorted(self.imu[:, 0], [end - span, end], side="right")
        return envelope_rows(self.imu[lo:hi], 400)


def envelope_rows(data, bins):
    """Reduce rows to a min/max pair per bin and column, keeping spikes visible.

    Stride decimation aliases vibration and drops short spikes; this keeps each
    bin's extremes (per column) at the bin's first/last timestamps.
    """
    if len(data) <= 2 * bins:
        return data
    starts = np.linspace(0, len(data), bins + 1).astype(int)
    out = np.empty((2 * bins, data.shape[1]))
    out[0::2] = np.minimum.reduceat(data, starts[:-1], axis=0)
    out[1::2] = np.maximum.reduceat(data, starts[:-1], axis=0)
    out[0::2, 0] = data[starts[:-1], 0]
    out[1::2, 0] = data[starts[1:] - 1, 0]
    return out


def _unpack(stream, fmt, label):
    size = struct.calcsize(fmt)
    blob = stream.read(size)
    if len(blob) != size:
        raise ValueError(f"Truncated session: {label}.")
    return struct.unpack(fmt, blob)


def read_record(path, interrupted=lambda: False):
    path = Path(path).expanduser().resolve()
    # Bound viewer memory; the engine can still calibrate larger logs directly.
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError(
            "This session exceeds the viewer's 512 MiB limit. Run ov_zcalibrate directly for this recording."
        )
    with path.open("rb") as stream:
        magic, version, count = _unpack(stream, "<III", "header")
        if magic != 0x5343564F or version not in (5, 6, 7):
            raise ValueError("Expected an ov_zcalib session (format 5, 6, or 7).")
        if not 1 <= count <= 16:
            raise ValueError("Invalid camera count in session.")
        imu_seed = _unpack(stream, "<30d", "IMU seed")
        if not all(math.isfinite(v) for v in imu_seed):
            raise ValueError("Session contains a non-finite IMU seed.")
        cameras = []
        for _ in range(count):
            values = _unpack(stream, CAMERA_STRUCT.format, "camera seed")
            k, d = list(values[:4]), list(values[4:8])
            cam = Camera(
                k=k,
                d=d,
                fisheye=bool(values[8]),
                q=list(values[9:13]),
                p=list(values[13:16]),
                td=values[16],
                tr=values[17],
                width=values[18],
                height=values[19],
                fps=values[20],
                rolling=bool(values[21]),
            )
            if not (
                0 < cam.width <= 65536
                and 0 < cam.height <= 65536
                and k[0] > 0
                and k[1] > 0
            ):
                raise ValueError("Invalid camera dimensions or focal length.")
            if not all(
                math.isfinite(v)
                for v in k + d + cam.q + cam.p + [cam.td, cam.tr, cam.fps]
            ):
                raise ValueError("Session contains non-finite camera values.")
            cameras.append(cam)
        profile, cam_mode = _unpack(stream, "<Bi", "profile")
        if version >= 6:
            sigmas = _unpack(stream, f"<{count}d", "camera noise")
            for cam, sigma in zip(cameras, sigmas):
                if not math.isfinite(sigma) or sigma < 0:
                    raise ValueError("Invalid camera measurement noise.")
                cam.sigma = sigma
        imu_blob, frames, complete = bytearray(), [], False
        while True:
            if interrupted():
                raise InterruptedError("Session loading cancelled.")
            tag = stream.read(1)
            if not tag:
                if version == 7:
                    raise ValueError(
                        "Incomplete session: completion footer is missing."
                    )
                break
            if tag == b"\x00":
                blob = stream.read(IMU_STRUCT.size)
                if len(blob) != IMU_STRUCT.size:
                    raise ValueError("Truncated session: IMU sample.")
                sample = IMU_STRUCT.unpack(blob)
                if not all(math.isfinite(v) for v in sample):
                    raise ValueError("Non-finite IMU sample.")
                imu_blob.extend(blob)
            elif tag == b"\x01":
                stamp, exposure, temp, seq, cam, n = _unpack(
                    stream, FRAME_STRUCT.format, "frame"
                )
                if (
                    cam >= count
                    or n > 100000
                    or not all(math.isfinite(v) for v in (stamp, exposure, temp))
                ):
                    raise ValueError("Invalid camera frame in session.")
                blob = stream.read(n * POINT_DTYPE.itemsize)
                if len(blob) != n * POINT_DTYPE.itemsize:
                    raise ValueError("Truncated feature observations.")
                points = np.frombuffer(blob, dtype=POINT_DTYPE).copy()
                if (
                    not np.isfinite(points["u"]).all()
                    or not np.isfinite(points["v"]).all()
                ):
                    raise ValueError("Non-finite feature observation.")
                frames.append(Frame(stamp, points, cam, seq, exposure, temp))
            elif tag == b"\x02" and version == 7:
                end, ni, nf = _unpack(stream, "<IQQ", "completion footer")
                if (
                    end != 0x4543564F
                    or ni != len(imu_blob) // IMU_STRUCT.size
                    or nf != len(frames)
                    or stream.read(1)
                ):
                    raise ValueError("Invalid session footer or sample counts.")
                complete = True
                break
            else:
                raise ValueError("Unknown record tag in session.")
    return Session(
        path.name,
        cameras,
        np.frombuffer(imu_blob, dtype="<f8").reshape(-1, 8),
        frames,
        path=str(path),
        profile=profile,
        cam_mode=cam_mode,
        format=version,
        complete=complete,
    )


def read_report(path):
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        report = json.load(stream)
    if (
        not isinstance(report, dict)
        or report.get("schema") != 1
        or not isinstance(report.get("verdict"), dict)
    ):
        raise ValueError("Expected an ov_zcalib report with schema 1.")
    return report


def coverage(points, width, height):
    """Occupied cells, not a statistical observability or calibration certificate."""
    grid = np.zeros((6, 8), dtype=int)
    if len(points):
        x, y = points["u"], points["v"]
        keep = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        np.add.at(
            grid,
            (
                np.minimum(5, (y[keep] / height * 6).astype(int)),
                np.minimum(7, (x[keep] / width * 8).astype(int)),
            ),
            1,
        )
    return grid


def rate(times):
    differences = np.diff(times)
    differences = differences[differences > 0]
    # Include gaps in the displayed delivery rate; median period can conceal
    # substantial frame loss while still showing the nominal sensor rate.
    return len(differences) / float(np.sum(differences)) if len(differences) else 0.0


def demo_session():
    """A deterministic, clearly labelled viewer example. Never submitted to the solver."""
    rng = np.random.default_rng(14)
    t = np.arange(0, 42, 0.005)
    gyro = np.column_stack(
        (0.35 * np.sin(t * 0.9), 0.23 * np.cos(t * 0.7), 0.3 * np.sin(t * 0.55))
    )
    accel = np.column_stack(
        (1.5 * np.sin(t * 0.8), 0.9 * np.cos(t * 0.6), 9.81 + 0.4 * np.sin(t * 1.2))
    )
    imu = np.column_stack((t, gyro, accel, np.full_like(t, 32.0)))
    u, v = rng.uniform(30, 930, 180), rng.uniform(30, 570, 180)
    frames = []
    for seq, stamp in enumerate(np.arange(0, 42, 0.05)):
        pts = np.zeros(180, dtype=POINT_DTYPE)
        pts["id"] = np.arange(180)
        pts["u"] = u + 13 * np.sin(stamp * 0.8 + v / 180) + 5 * np.cos(stamp * 0.4)
        pts["v"] = v + 9 * np.cos(stamp * 0.7 + u / 270)
        frames.append(Frame(float(stamp), pts, seq=seq, exposure=0.004))
    return Session(
        "Explore the studio",
        [Camera(width=960, height=600, k=[620.0, 620.0, 480.0, 300.0])],
        imu,
        frames,
        demo=True,
    )
