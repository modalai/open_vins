"""ROS-free sensor streaming from logs into OpenVINS consumers, and their results.

Two little-endian protocols, specified in ``ov_core/src/utils/sensor_stream.h``:

Sensor stream v1 (this module -> ``ov_zcalib_track`` / ``ov_msckf_replay`` stdin)
    header ``b"OVSS"`` + u32 version 1, then records (u8 tag + payload):
    ``C`` camera  u32 cam, i32 w, i32 h, f64 fx fy cx cy, u32 nd, f64[nd], u8 model
                  (0 radtan, 1 equidistant), f64 readout_s, u32 len, name
    ``I`` imu     f64 t, f64 wx wy wz, f64 ax ay az, f64 temperature_c (NaN unavailable)
    ``F`` frame   u32 cam, f64 t (center-row mid-exposure), f32 exposure_s, i64 sync_ns,
                  i32 w, i32 h, u8[w*h] mono8
    ``E`` end

VIO result stream v1 (``ov_msckf_replay`` stdout -> ``VioRun``)
    header ``b"OVVR"`` + u32 version 1, then ``S`` state, ``P`` points, ``T`` tracks,
    ``M`` message and ``E`` end records (see the header for field layouts).

Time base: frames and IMU samples keep the LogSource time base (seconds since the log
epoch). OpenVINS treats non-positive stamps as "unset", so the stream adds an integer
``stream_offset`` (first sample >= 1 s) and ``ov_msckf_replay --time-offset`` subtracts it
from every reported time: ``VioRun`` records, ``VioRun.frame_times`` and the TUM file are all
in log time. A converted session stores ``log_time + stream_offset`` (recorded in its
``.capture.json`` as ``time_offset_s``).
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import threading
import time

import numpy as np

from .engine import find_executable
from .logs import BagLog, LogError
from .media import black_level_for, decode_bytes, ros_image_to_picture, to_gray8

SENSOR_MAGIC = b"OVSS"
RESULT_MAGIC = b"OVVR"
VERSION = 1
MAX_CAMERAS = 16
MAX_PIXELS = 40_000_000
MAX_DISTORTION = 16
REPLAY_BINARY = "ov_msckf_replay"
TRACK_BINARY = "ov_zcalib_track"
BUILD_DIRS = ("build", "build-studio", "build-host", "build_native", "build-ocl-host")

_CAMERA_HEAD = struct.Struct("<Iii4dI")
_IMU = struct.Struct("<8d")
_FRAME_HEAD = struct.Struct("<Idfqii")
_STATE_HEAD = struct.Struct("<d4d3d3d3d3d3d3dB")
_CALIB = struct.Struct("<4d3dd")
_STATE_TAIL = struct.Struct("<dBII")
_POINT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("kind", "u1")])
_TRACK = np.dtype([("id", "<u4"), ("u", "<f4"), ("v", "<f4")])
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_FATAL = re.compile(r"could not|unable|error|failed|invalid|not found", re.I)
_MISSING = re.compile(r"the node (\S+) of type \[[^\]]*\] was not found\.\.\.$")


# ---------------------------------------------------------------------------
# Sensor stream writer / reader
# ---------------------------------------------------------------------------
class SensorStreamWriter:
    """Writes sensor stream v1 to a binary file object (pipe, file, BytesIO)."""

    def __init__(self, fileobj):
        self._f = fileobj
        self.bytes_written = 0
        self._write(SENSOR_MAGIC + struct.pack("<I", VERSION))

    def _write(self, data):
        self._f.write(data)
        self.bytes_written += len(data)

    def camera(self, cam, width, height, intrinsics, readout_s, name):
        if not 0 <= cam < MAX_CAMERAS:
            raise ValueError(f"camera id {cam} out of range")
        if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
            raise ValueError(f"invalid camera size {width}x{height}")
        intr = intrinsics or {}
        k = list(intr.get("k", [0.0, 0.0, 0.0, 0.0]))
        dist = [float(v) for v in intr.get("d", [])]
        if len(dist) > MAX_DISTORTION:
            raise ValueError("too many distortion coefficients")
        model = 1 if intr.get("model") == "equidistant" else 0
        label = str(name).encode("utf-8")[:4096]
        self._write(
            b"C"
            + _CAMERA_HEAD.pack(cam, width, height, *[float(v) for v in k], len(dist))
            + struct.pack(f"<{len(dist)}d", *dist)
            + struct.pack("<BdI", model, float(readout_s), len(label))
            + label
        )

    def imu(self, t, gyro, accel, temperature=math.nan):
        self._write(
            b"I"
            + _IMU.pack(
                float(t), *map(float, gyro), *map(float, accel), float(temperature)
            )
        )

    def frame(self, cam, t_center, exposure_s, gray, sync_ns=-1):
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        if gray.ndim != 2:
            raise ValueError("frames must be 2-D mono8")
        height, width = gray.shape
        if width * height > MAX_PIXELS:
            raise ValueError("frame too large")
        self._write(
            b"F"
            + _FRAME_HEAD.pack(
                cam, float(t_center), float(exposure_s), int(sync_ns), width, height
            )
        )
        self._f.write(memoryview(gray).cast("B"))
        self.bytes_written += gray.nbytes

    def end(self):
        self._write(b"E")
        self._f.flush()


def _read_exact(fileobj, size):
    data = fileobj.read(size)
    if data is None or len(data) != size:
        raise EOFError("truncated stream")
    return data


def read_sensor_stream(fileobj):
    """Pure-Python sensor stream parser (tests, debugging). Yields (kind, payload)."""
    magic = _read_exact(fileobj, 8)
    if magic[:4] != SENSOR_MAGIC or struct.unpack("<I", magic[4:])[0] != VERSION:
        raise ValueError("not a v1 sensor stream")
    while True:
        tag = fileobj.read(1)
        if not tag:
            raise EOFError("truncated stream (no end record)")
        if tag == b"C":
            cam, w, h, fx, fy, cx, cy, nd = _CAMERA_HEAD.unpack(
                _read_exact(fileobj, _CAMERA_HEAD.size)
            )
            dist = list(struct.unpack(f"<{nd}d", _read_exact(fileobj, 8 * nd)))
            model, readout, n = struct.unpack("<BdI", _read_exact(fileobj, 13))
            name = _read_exact(fileobj, n).decode("utf-8")
            yield (
                "camera",
                {
                    "cam": cam,
                    "width": w,
                    "height": h,
                    "k": [fx, fy, cx, cy],
                    "d": dist,
                    "model": model,
                    "readout_s": readout,
                    "name": name,
                },
            )
        elif tag == b"I":
            yield "imu", _IMU.unpack(_read_exact(fileobj, _IMU.size))
        elif tag == b"F":
            cam, t, exposure, sync, w, h = _FRAME_HEAD.unpack(
                _read_exact(fileobj, _FRAME_HEAD.size)
            )
            pixels = np.frombuffer(_read_exact(fileobj, w * h), np.uint8).reshape(h, w)
            yield (
                "frame",
                {
                    "cam": cam,
                    "t": t,
                    "exposure_s": exposure,
                    "sync_ns": sync,
                    "pixels": pixels,
                },
            )
        elif tag == b"E":
            yield "end", None
            return
        else:
            raise ValueError(f"unknown sensor record {tag!r}")


# ---------------------------------------------------------------------------
# Log -> sensor stream
# ---------------------------------------------------------------------------
@dataclass
class CameraSettings:
    """How a camera's stamps map to center-row mid-exposure.

    reference: "center" (stamps used verbatim) or "start" (start-of-exposure: the stream
    adds (readout + exposure) / 2). With ``per_frame_exposure`` the log's per-frame exposure
    is used when it has one; ``exposure_s`` is the constant otherwise.
    """

    reference: str = "center"
    readout_s: float = 0.0
    exposure_s: float = 0.0
    per_frame_exposure: bool = True


def default_settings(cam):
    """VOXL MPA stamps are start-of-exposure with per-frame exposure; bags are taken as published.

    Readout is 0 (global shutter) unless known; rolling-shutter VOXL sensors need the
    HAL3 readout entered by the user.
    """
    if cam.stamp_reference == "start":
        return CameraSettings("start", 0.0, 0.0, True)
    return CameraSettings("center", 0.0, 0.0, True)


def _settings_for(cameras, settings):
    if settings is None:
        return [default_settings(c) for c in cameras]
    if isinstance(settings, dict):
        return [settings.get(c.name) or default_settings(c) for c in cameras]
    out = list(settings)
    if len(out) != len(cameras):
        raise ValueError("one CameraSettings per camera is required")
    return [s or default_settings(c) for s, c in zip(out, cameras)]


@dataclass
class StreamPlan:
    """Which samples will be streamed, at which (log-time) instants."""

    cameras: list
    imu: object
    settings: list
    frame_times: list  # per camera: centered log time per row, NaN where not streamed
    exposures: list  # per camera: exposure [s] per row
    imu_rows: np.ndarray
    dropped: dict
    stream_offset: float
    start: float
    end: float

    @property
    def frame_rows(self):
        return [np.where(np.isfinite(t))[0] for t in self.frame_times]


def plan_stream(log, cameras, imu, settings=None, start=None, end=None):
    """Center, crop and de-duplicate every stream (cheap: metadata arrays only)."""
    if not cameras:
        raise ValueError("choose at least one camera")
    if len(cameras) > MAX_CAMERAS:
        raise ValueError("at most 16 cameras can be streamed")
    if imu is None or not len(imu):
        raise ValueError("choose an IMU channel with samples")
    settings = _settings_for(cameras, settings)
    lo = -math.inf if start is None else float(start)
    hi = math.inf if end is None else float(end)
    dropped, frame_times, exposures = {}, [], []
    for cam, s in zip(cameras, settings):
        if s.reference not in ("center", "start"):
            raise ValueError(f"{cam.name}: timestamp reference must be center or start")
        if not (
            math.isfinite(s.readout_s)
            and s.readout_s >= 0
            and math.isfinite(s.exposure_s)
            and s.exposure_s >= 0
        ):
            raise ValueError(
                f"{cam.name}: readout and exposure must be finite and nonnegative"
            )
        exp = np.full(len(cam), float(s.exposure_s))
        if (
            s.per_frame_exposure
            and cam.exposure is not None
            and len(cam.exposure) == len(cam)
        ):
            logged = np.asarray(cam.exposure, dtype=np.float64)
            exp = np.where(np.isfinite(logged) & (logged > 0), logged, exp)
        if cam.encoded or not cam.has_pictures:
            raise ValueError(
                f"{cam.name}: frames are not stored as images and cannot be streamed"
            )
        if not cam.width or not cam.height:
            raise ValueError(f"{cam.name}: image size is unknown")
        center = cam.times + (
            0.5 * (s.readout_s + exp) if s.reference == "start" else 0.0
        )
        keep = (center >= lo) & (center <= hi)
        if cam.files is not None:
            keep &= np.array([bool(f) for f in cam.files], dtype=bool)
        times = np.full(len(cam), np.nan)
        last, last_stamp, bad = -math.inf, None, 0
        stamps = cam.stamps_ns
        for i in np.where(keep)[0]:
            # A repeated raw stamp is a re-sent frame even when its exposure (and so its
            # centered time) differs; the centered sequence must also strictly increase.
            if center[i] > last and (last_stamp is None or stamps[i] > last_stamp):
                times[i] = center[i]
                last, last_stamp = center[i], stamps[i]
            else:
                bad += 1
        dropped[cam.name] = bad
        frame_times.append(times)
        exposures.append(exp)
    t = imu.data[:, 0]
    streamed = [ft[np.isfinite(ft)] for ft in frame_times]
    if not any(len(ft) for ft in streamed):
        raise ValueError("no camera frames fall inside the selected time range")
    first = min(float(ft[0]) for ft in streamed if len(ft))
    last_frame = max(float(ft[-1]) for ft in streamed if len(ft))
    # IMU slightly beyond the frames so the first/last views are covered.
    in_range = (t >= max(lo, first - 1.0)) & (t <= min(hi, last_frame) + 0.5)
    rows, last, bad = [], -math.inf, 0
    finite = np.isfinite(imu.data[:, :7]).all(axis=1)
    for i in np.where(in_range)[0]:
        if finite[i] and t[i] > last:
            rows.append(i)
            last = t[i]
        else:
            bad += 1
    dropped[imu.name] = bad
    imu_rows = np.array(rows, dtype=np.int64)
    if not len(imu_rows):
        raise ValueError("no IMU samples fall inside the selected time range")
    t0 = min(first, float(t[imu_rows[0]]))
    offset = float(max(0, math.ceil(1.0 - t0)))
    return StreamPlan(
        cameras,
        imu,
        settings,
        frame_times,
        exposures,
        imu_rows,
        dropped,
        offset,
        t0,
        max(last_frame, float(t[imu_rows[-1]])),
    )


class _BagFrames:
    """Sequential decode of the chosen image topics with a private reader (no shared lock)."""

    def __init__(self, log, plan, black):
        self.reader = BagLog._reader(log.path)
        self.plan, self.black = plan, black
        self.index = {cam.name: i for i, cam in enumerate(plan.cameras)}
        self.counters = [0] * len(plan.cameras)
        self.compressed = [
            log.channel(cam.name).type.endswith("CompressedImage")
            for cam in plan.cameras
        ]
        conns = [c for c in self.reader.connections if c.topic in self.index]
        start = None
        rows = plan.frame_rows
        logtimes = [getattr(cam, "_logtimes", None) for cam in plan.cameras]
        if all(lt is not None and len(r) for lt, r in zip(logtimes, rows)):
            start = int(min(lt[r[0]] for lt, r in zip(logtimes, rows)))
            # Rows before the first selected one still advance the per-topic counters.
            for i, (lt, r) in enumerate(zip(logtimes, rows)):
                self.counters[i] = int(np.searchsorted(lt, start, side="left"))
        self.messages = self.reader.messages(connections=conns, start=start)
        self.buffer = {}

    def get(self, cam, row):
        key = (cam, row)
        while key not in self.buffer:
            try:
                conn, _, raw = next(self.messages)
            except StopIteration:
                raise LogError(
                    f"{self.plan.cameras[cam].name}: bag ended before row {row}"
                ) from None
            i = self.index[conn.topic]
            r = self.counters[i]
            self.counters[i] += 1
            wanted = r < len(self.plan.frame_times[i]) and math.isfinite(
                self.plan.frame_times[i][r]
            )
            if not wanted:
                continue  # cropped or de-duplicated row: skip without decoding
            msg = self.reader.deserialize(raw, conn.msgtype)
            picture = (
                decode_bytes(msg.data)
                if self.compressed[i]
                else ros_image_to_picture(msg)
            )
            # Same studio rotation as CameraStream.picture() (frames, size, intrinsics agree).
            picture = self.plan.cameras[i].orient(picture)
            self.buffer[(i, r)] = to_gray8(picture, self.black[i])
        return self.buffer.pop(key)

    def close(self):
        self.reader.close()


def stream_log(
    log,
    writer,
    cameras,
    imu,
    settings=None,
    start=None,
    end=None,
    progress=None,
    cancelled=None,
    realtime=0.0,
    plan=None,
    black_level="",
    workers=4,
):
    """Stream IMU + frames merged by center time. Returns stats (counts, drops, frame_times).

    Times written are log times + ``plan.stream_offset``. Raises InterruptedError on cancel
    (no end record is written, so consumers reject the partial stream).
    """
    plan = plan or plan_stream(log, cameras, imu, settings, start, end)
    offset = plan.stream_offset
    black = [black_level_for(c.name, c.sensor, black_level) for c in plan.cameras]
    for i, cam in enumerate(plan.cameras):
        writer.camera(
            i,
            cam.width,
            cam.height,
            cam.intrinsics,
            plan.settings[i].readout_s,
            cam.name,
        )
    # Merge: IMU first at equal times so a frame's instant is always covered.
    t_imu = imu.data[plan.imu_rows, 0]
    kinds = [np.zeros(len(t_imu), np.int64)]
    times = [t_imu]
    owners = [plan.imu_rows]
    for i, rows in enumerate(plan.frame_rows):
        times.append(plan.frame_times[i][rows])
        kinds.append(np.full(len(rows), i + 1, np.int64))
        owners.append(rows)
    times, kinds, owners = (
        np.concatenate(times),
        np.concatenate(kinds),
        np.concatenate(owners),
    )
    order = np.lexsort((kinds, times))
    total = len(order)
    frames_total = int(sum(len(r) for r in plan.frame_rows))
    stats = {
        "imu": 0,
        "frames": {c.name: 0 for c in plan.cameras},
        "decode_errors": {c.name: 0 for c in plan.cameras},
        "dropped": dict(plan.dropped),
        "stream_offset": offset,
        "start": plan.start,
        "end": plan.end,
    }
    bag = isinstance(log, BagLog)
    source = _BagFrames(log, plan, black) if bag else None
    pool = ThreadPoolExecutor(max_workers=max(1, workers)) if not bag else None
    pending = deque()
    frame_events = [(kinds[j] - 1, owners[j]) for j in order if kinds[j]]
    frame_iter = iter(frame_events)

    def load(cam, row):
        return to_gray8(plan.cameras[cam].picture(row), black[cam])

    def prefetch():
        while pool is not None and len(pending) < 4 * max(1, workers):
            nxt = next(frame_iter, None)
            if nxt is None:
                return
            pending.append((nxt, pool.submit(load, *nxt)))

    wall0, t_first = time.monotonic(), float(times[order[0]]) if total else 0.0
    imu_data = imu.data
    try:
        prefetch()
        for n, j in enumerate(order):
            if cancelled is not None and n % 64 == 0 and cancelled():
                raise InterruptedError("streaming cancelled")
            t = float(times[j])
            if realtime and realtime > 0:
                ahead = (t - t_first) / realtime - (time.monotonic() - wall0)
                if ahead > 0.002:
                    time.sleep(ahead)
            if kinds[j] == 0:
                row = imu_data[owners[j]]
                writer.imu(t + offset, row[1:4], row[4:7], row[7])
                stats["imu"] += 1
            else:
                cam, row = int(kinds[j] - 1), int(owners[j])
                try:
                    if pool is not None:
                        (pc, pr), future = pending.popleft()
                        prefetch()
                        if (pc, pr) != (cam, row):
                            raise AssertionError(
                                "frame prefetch order diverged from the merge order"
                            )
                        gray = future.result()
                    else:
                        gray = source.get(cam, row)
                except (LogError, OSError, ValueError, RuntimeError) as error:
                    if isinstance(error, RuntimeError) and "OpenCV" in str(error):
                        raise
                    stats["decode_errors"][plan.cameras[cam].name] += 1
                    plan.frame_times[cam][row] = np.nan
                    continue
                c = plan.cameras[cam]
                if gray.shape != (c.height, c.width):
                    raise ValueError(
                        f"{c.name}: frame {row} is {gray.shape[1]}x{gray.shape[0]}, expected {c.width}x{c.height}"
                    )
                writer.frame(cam, t + offset, float(plan.exposures[cam][row]), gray)
                stats["frames"][c.name] += 1
                if progress is not None and stats["frames"][c.name] % 10 == 0:
                    done = sum(stats["frames"].values())
                    progress(
                        done / max(1, frames_total), f"{done}/{frames_total} frames"
                    )
        writer.end()
    finally:
        if pool is not None:
            for _, future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
        if source is not None:
            source.close()
    stats["frame_times"] = plan.frame_times
    if progress is not None:
        progress(1.0, "stream complete")
    return stats


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------
def _drain_lines(stream, sink, tail):
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        tail.append(line)
        if sink is not None:
            try:
                sink(line)
            except Exception:  # a UI callback must never kill the pipe reader
                pass
    stream.close()


def _stop_process(process, timeout=3.0):
    if process is None or process.poll() is not None:
        return
    try:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    except (OSError, ValueError):
        pass
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _noise_args(noise):
    if noise is None:
        return []
    values = asdict(noise) if hasattr(noise, "__dataclass_fields__") else dict(noise)
    args = []
    for key, flag in (
        ("sigma_w", "--sigma-w"),
        ("sigma_wb", "--sigma-wb"),
        ("sigma_a", "--sigma-a"),
        ("sigma_ab", "--sigma-ab"),
    ):
        if values.get(key) is not None:
            args += [flag, repr(float(values[key]))]
    return args


def convert_to_session(
    log,
    out_path,
    cameras,
    imu,
    binary=None,
    features=200,
    noise=None,
    settings=None,
    progress=None,
    cancelled=None,
    on_line=None,
    start=None,
    end=None,
    black_level="",
):
    """Track a log's frames with TrackKLT into a format-7 session (``ov_zcalib_track``).

    Writes ``<out>.capture.json`` provenance. Raises on failure; the partial record is
    removed by the converter. Returns a summary dict.
    """
    out = Path(out_path).expanduser()
    if out.exists() or Path(str(out) + ".capture.json").exists():
        raise FileExistsError(
            f"{out} already exists; existing sessions are never overwritten"
        )
    missing = [c.name for c in cameras if not c.intrinsics]
    if missing:
        raise ValueError(
            "intrinsics are required for tracking; missing for " + ", ".join(missing)
        )
    binary = binary or find_executable(TRACK_BINARY)
    if not binary or not os.access(binary, os.X_OK):
        raise FileNotFoundError(
            f"{TRACK_BINARY} was not found; build ov_zcalib (OV_BUILD_CALIB=ON) with TrackKLT"
        )
    plan = plan_stream(log, cameras, imu, settings, start, end)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [binary, str(out), "--features", str(int(features))] + _noise_args(noise)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1 << 20,
    )
    tail = deque(maxlen=200)
    reader = threading.Thread(
        target=_drain_lines, args=(process.stdout, on_line, tail), daemon=True
    )
    reader.start()
    stats, failure = None, None
    try:
        stats = stream_log(
            log,
            SensorStreamWriter(process.stdin),
            cameras,
            imu,
            plan=plan,
            progress=progress,
            cancelled=cancelled,
            black_level=black_level,
        )
        process.stdin.close()
        process.wait()
    except InterruptedError:
        _stop_process(process)
        reader.join(timeout=5)
        raise
    except (BrokenPipeError, OSError) as error:
        failure = error
        _stop_process(process)
    except BaseException:
        _stop_process(process)
        raise
    reader.join(timeout=10)
    if failure is not None or process.returncode != 0 or not out.is_file():
        detail = "\n".join(list(tail)[-12:])
        raise RuntimeError(
            f"{TRACK_BINARY} failed (exit {process.returncode}): {detail or failure}"
        )
    temperature = bool(imu.temperature_available)
    sidecar = Path(str(out) + ".capture.json")
    meta = {
        "tracker": "TrackKLT",
        "converter": TRACK_BINARY,
        "source": {
            "kind": log.kind,
            "path": str(log.path),
            "cameras": [c.name for c in cameras],
            "imu": imu.name,
        },
        "temperature_available": temperature,
        "thermal_checks_evaluated": False,
        "temperature_encoding": "celsius" if temperature else "legacy_zero_placeholder",
        "timestamp_reference": {
            c.name: (
                "start-of-exposure, centered by (readout + exposure) / 2"
                if s.reference == "start"
                else "center (as recorded)"
            )
            for c, s in zip(plan.cameras, plan.settings)
        },
        "readout_s": {c.name: s.readout_s for c, s in zip(plan.cameras, plan.settings)},
        "exposure": {
            c.name: (
                "per-frame (log)"
                if s.per_frame_exposure and c.exposure is not None
                else f"constant {s.exposure_s:g} s"
            )
            for c, s in zip(plan.cameras, plan.settings)
        },
        "time_offset_s": plan.stream_offset,
        "time_base": "session time = log time + time_offset_s",
        "frames": stats["frames"],
        "imu_samples": stats["imu"],
        "dropped": stats["dropped"],
        "decode_errors": stats["decode_errors"],
        "features": int(features),
    }
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)
    return {
        "path": str(out),
        "sidecar": str(sidecar),
        "stats": {k: v for k, v in stats.items() if k != "frame_times"},
        "output": list(tail),
    }


# ---------------------------------------------------------------------------
# VIO results
# ---------------------------------------------------------------------------
@dataclass
class VioCalib:
    q_ItoC: np.ndarray  # JPL x, y, z, w
    p_IinC: np.ndarray
    td: float  # camera -> IMU offset: t_imu = t_cam + td


@dataclass
class VioState:
    t: float
    q_GtoI: np.ndarray  # JPL x, y, z, w (numerically Hamilton q_ItoG)
    p_IinG: np.ndarray
    v_IinG: np.ndarray
    bg: np.ndarray
    ba: np.ndarray
    sigma_ori: np.ndarray
    sigma_pos: np.ndarray
    calib: list
    proc_ms: float
    initialized: bool
    zupt: bool
    n_slam: int
    n_msckf: int


@dataclass
class VioPoints:
    t: float
    xyz: np.ndarray  # (N, 3) float32, global frame
    kind: np.ndarray  # (N,) uint8: 0 MSCKF, 1 SLAM, 2 ARUCO


@dataclass
class VioTracks:
    t: float
    cam: int
    ids: np.ndarray  # (N,) uint32
    uv: np.ndarray  # (N, 2) float32, distorted pixels


@dataclass
class VioMessage:
    level: int  # 0 info, 1 warning, 2 error
    text: str


@dataclass
class VioEnd:
    ok: bool


def read_results(fileobj):
    """Parse a VIO result stream; yields Vio* records until the end record or EOF."""
    head = fileobj.read(8)
    if not head:
        return
    if (
        len(head) != 8
        or head[:4] != RESULT_MAGIC
        or struct.unpack("<I", head[4:])[0] != VERSION
    ):
        raise ValueError("not a v1 VIO result stream")
    while True:
        tag = fileobj.read(1)
        if not tag:
            return
        if tag == b"S":
            v = _STATE_HEAD.unpack(_read_exact(fileobj, _STATE_HEAD.size))
            arr = np.array(v[:-1])
            ncam = v[-1]
            calib = []
            for _ in range(ncam):
                c = _CALIB.unpack(_read_exact(fileobj, _CALIB.size))
                calib.append(VioCalib(np.array(c[:4]), np.array(c[4:7]), c[7]))
            proc, flags, n_slam, n_msckf = _STATE_TAIL.unpack(
                _read_exact(fileobj, _STATE_TAIL.size)
            )
            yield VioState(
                float(arr[0]),
                arr[1:5],
                arr[5:8],
                arr[8:11],
                arr[11:14],
                arr[14:17],
                arr[17:20],
                arr[20:23],
                calib,
                proc,
                bool(flags & 1),
                bool(flags & 2),
                n_slam,
                n_msckf,
            )
        elif tag == b"P":
            t, n = struct.unpack("<dI", _read_exact(fileobj, 12))
            pts = np.frombuffer(_read_exact(fileobj, n * _POINT.itemsize), _POINT)
            xyz = np.column_stack((pts["x"], pts["y"], pts["z"])).astype(np.float32)
            yield VioPoints(t, xyz, pts["kind"].copy())
        elif tag == b"T":
            t, cam, n = struct.unpack("<dII", _read_exact(fileobj, 16))
            tr = np.frombuffer(_read_exact(fileobj, n * _TRACK.itemsize), _TRACK)
            yield VioTracks(
                t,
                cam,
                tr["id"].copy(),
                np.column_stack((tr["u"], tr["v"])).astype(np.float32),
            )
        elif tag == b"M":
            level, n = struct.unpack("<BI", _read_exact(fileobj, 5))
            yield VioMessage(
                level, _read_exact(fileobj, n).decode("utf-8", errors="replace")
            )
        elif tag == b"E":
            yield VioEnd(bool(_read_exact(fileobj, 1)[0]))
            return
        else:
            raise ValueError(f"unknown VIO result record {tag!r}")


# ---------------------------------------------------------------------------
# Replay binaries
# ---------------------------------------------------------------------------
_CAPABILITIES = {}


def replay_capabilities(binary=None):
    """``ov_msckf_replay --capabilities`` as a dict (cached per path); {} on failure."""
    binary = binary or find_replay_binary()
    if not binary:
        return {}
    key = os.path.realpath(binary)
    try:
        stamp = os.stat(key).st_mtime_ns
    except OSError:
        return {}
    cached = _CAPABILITIES.get(key)
    if cached and cached[0] == stamp:
        return dict(cached[1])
    try:
        done = subprocess.run(
            [key, "--capabilities"], capture_output=True, timeout=10, check=False
        )
        caps = (
            json.loads(done.stdout.decode("utf-8").strip().splitlines()[0])
            if done.returncode == 0
            else {}
        )
        if not isinstance(caps, dict):
            caps = {}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        caps = {}
    _CAPABILITIES[key] = (stamp, caps)
    return dict(caps)


def find_replay_binaries():
    """Every ov_msckf_replay found (PATH, ROS/CMake prefixes, repo build dirs) with capabilities."""
    candidates = []
    found = shutil.which(REPLAY_BINARY)
    if found:
        candidates.append(found)
    for variable in ("AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
        for prefix in os.environ.get(variable, "").split(os.pathsep):
            if prefix:
                candidates += [
                    Path(prefix) / "lib" / "ov_msckf" / REPLAY_BINARY,
                    Path(prefix) / "bin" / REPLAY_BINARY,
                ]
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "bin" / REPLAY_BINARY)
        if (parent / "ov_msckf").is_dir():
            candidates += [
                parent / build / "ov_msckf" / REPLAY_BINARY for build in BUILD_DIRS
            ]
            break
    out, seen = [], set()
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        out.append((str(path), replay_capabilities(str(path))))
    return out


def find_replay_binary(tracker=None):
    """First replay binary, optionally one offering ``tracker`` ("klt" or "ocl")."""
    for path, caps in find_replay_binaries():
        if tracker is None or tracker in caps.get("trackers", []):
            return path
    return ""


class VioRun:
    """One ov_msckf_replay process fed from a LogSource; never blocks the caller.

    ``poll()`` returns parsed records (log time base). ``frame_times[i]`` holds the centered
    log time of every row of the i-th streamed camera (NaN: not streamed), so a VioTracks
    record maps back to its CameraStream row with ``np.searchsorted``.
    """

    def __init__(self):
        self.process = None
        self.frame_times = []
        self.cameras = []
        self.stream_offset = 0.0
        self.summary = {}
        self.error = ""
        self.tracker = ""
        self.lines = deque(maxlen=2000)
        self.on_line = None
        self._records = deque()
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._threads = []
        self.stream_stats = None
        self.ended = None
        self.missing_keys = []  # required configuration keys the estimator reported missing

    def start(
        self,
        config_path,
        log,
        cameras,
        imu,
        settings=None,
        binary=None,
        tum_path=None,
        realtime=0.0,
        start=None,
        end=None,
        points_every=1,
        on_line=None,
        progress=None,
        black_level="",
    ):
        if self.running:
            raise RuntimeError("a VIO run is already active")
        binary = binary or find_replay_binary()
        if not binary or not os.access(binary, os.X_OK):
            raise FileNotFoundError(
                f"{REPLAY_BINARY} was not found; build ov_msckf (its target ov_msckf_replay)"
            )
        if not Path(config_path).is_file():
            raise FileNotFoundError(f"configuration {config_path} does not exist")
        plan = plan_stream(log, cameras, imu, settings, start, end)
        self.cameras = list(cameras)
        self.frame_times = plan.frame_times
        self.stream_offset = plan.stream_offset
        self.on_line = on_line
        command = [
            binary,
            str(config_path),
            "--points-every",
            str(max(1, int(points_every))),
            "--time-offset",
            repr(plan.stream_offset),
        ]
        if tum_path:
            command += ["--tum", str(tum_path)]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        stdin = io.BufferedWriter(self.process.stdin, buffer_size=1 << 20)

        def feed():
            try:
                self.stream_stats = stream_log(
                    log,
                    SensorStreamWriter(stdin),
                    cameras,
                    imu,
                    plan=plan,
                    cancelled=self._cancel.is_set,
                    realtime=realtime,
                    progress=progress,
                    black_level=black_level,
                )
            except InterruptedError:
                pass
            except (BrokenPipeError, ValueError, OSError) as error:
                if not self._cancel.is_set() and self.process.poll() is None:
                    self.error = f"streaming failed: {error}"
                elif not self._cancel.is_set() and not self.error:
                    self.error = f"estimator stopped while streaming ({error})"
            except Exception as error:  # surfaced to the caller via .error
                self.error = f"streaming failed: {error}"
            finally:
                try:
                    stdin.close()
                except (OSError, ValueError):
                    pass

        def results():
            reader = io.BufferedReader(self.process.stdout, buffer_size=1 << 16)
            try:
                for record in read_results(reader):
                    if isinstance(record, VioMessage) and record.text.startswith(
                        "tracker: "
                    ):
                        self.tracker = record.text[len("tracker: ") :]
                    if isinstance(record, VioEnd):
                        self.ended = record.ok
                    if isinstance(record, VioMessage) and record.text.startswith(
                        "finished: "
                    ):
                        self.summary["finished"] = record.text
                    if (
                        isinstance(record, VioMessage)
                        and record.level == 2
                        and not self._cancel.is_set()
                    ):
                        record = VioMessage(2, self._explain(record.text))
                        self.error = self.error or record.text
                    with self._lock:
                        self._records.append(record)
            except (EOFError, ValueError, OSError) as error:
                if not self._cancel.is_set():
                    self.error = (
                        self.error or f"result stream ended abnormally: {error}"
                    )
            finally:
                reader.close()

        def errors():
            _drain_lines(self.process.stderr, self._line, self.lines)

        for target in (feed, results, errors):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def _explain(self, text):
        """Append what the console said: missing configuration keys and the fatal reason."""
        if len(self._threads) > 2:
            self._threads[2].join(timeout=3)  # stderr carries the parser's diagnostics
        parts = [text]
        if self.missing_keys:
            parts.append("missing keys: " + ", ".join(self.missing_keys))
        for line in reversed(self.lines):
            clean = _ANSI.sub("", line).strip()
            if (
                clean
                and not _MISSING.search(clean)
                and _FATAL.search(clean)
                and clean not in text
            ):
                parts.append(clean)
                break
        return "; ".join(parts)

    def _line(self, line):
        clean = _ANSI.sub("", line).strip()
        match = _MISSING.search(clean)
        if match and match.group(1) not in self.missing_keys:
            self.missing_keys.append(match.group(1))
        if self.on_line is not None:
            self.on_line(line)

    def poll(self, limit=None):
        with self._lock:
            if limit is None or limit >= len(self._records):
                out = list(self._records)
                self._records.clear()
            else:
                out = [self._records.popleft() for _ in range(limit)]
        return out

    @property
    def cancelled(self):
        """True after stop(); the estimator's truncated-stream exit is then not an error."""
        return self._cancel.is_set()

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    @property
    def returncode(self):
        return None if self.process is None else self.process.poll()

    def wait(self, timeout=None):
        if self.process is None:
            return None
        try:
            code = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        for thread in self._threads:
            thread.join(timeout=5)
        return code

    def stop(self, timeout=3.0):
        """Cancel streaming, let the estimator exit, then terminate/kill if needed."""
        self._cancel.set()
        _stop_process(self.process, timeout)
        for thread in self._threads:
            thread.join(timeout=timeout)
