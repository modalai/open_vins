"""Sensor stream / VIO result protocols, log streaming, and the C++ consumers end to end."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import os
import shutil
import struct
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PyQt5.QtGui import QImage

from zstudio import stream as S
from zstudio.logs import open_log
from zstudio.model import read_record

SCRATCH_ROSBAGS = Path(
    "/tmp/claude-1000/-home-zbft-community-open-vins/d14c4c6c-0380-424c-a31d-35e8c666e009/scratchpad/pylib"
)
try:
    import rosbags  # noqa: F401
except ImportError:
    if SCRATCH_ROSBAGS.is_dir():
        sys.path.append(str(SCRATCH_ROSBAGS))
try:
    from rosbags.rosbag2 import Writer as Bag2Writer
    from rosbags.typesys import Stores, get_typestore

    HAVE_ROSBAGS = True
except ImportError:
    HAVE_ROSBAGS = False

REPO = Path(__file__).resolve().parents[2]
IMU_APRIL = Path(
    "/home/zbft/calib_eval/data/calibration_datasets/imu_april/imu_april.bag"
)
TRACK = S.find_executable(S.TRACK_BINARY)
REPLAY = S.find_replay_binary()

INTRINSICS = """%YAML:1.0
---
M: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 200., 0., 160., 0., 200., 120., 0., 0., 1. ]
D: !!opencv-matrix
   rows: 4
   cols: 1
   dt: d
   data: [ 0., 0., 0., 0. ]
width: 320
height: 240
distortion_model: "plumb_bob"
"""


def texture(width=900, height=700, seed=3):
    """Blob texture with strong, well-spread corners for KLT."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, size=(height // 10 + 2, width // 10 + 2)).astype(
        np.float32
    )
    image = np.kron(coarse, np.ones((10, 10), np.float32))[:height, :width]
    for _ in range(2):
        image = (
            image
            + np.roll(image, 1, 0)
            + np.roll(image, 1, 1)
            + np.roll(image, -1, 0)
            + np.roll(image, -1, 1)
        ) / 5
    return image.astype(np.uint8)


def write_png(path, gray):
    gray = np.ascontiguousarray(gray)
    image = QImage(
        gray.data, gray.shape[1], gray.shape[0], gray.shape[1], QImage.Format_Grayscale8
    )
    if not image.save(str(path), "PNG"):
        raise OSError(f"cannot write {path}")


def make_voxl_log(
    root,
    frames=60,
    fps=20.0,
    imu_rate=200.0,
    duplicate_frame=None,
    imu_reversal=None,
    epoch=5_000_000_000,
    exposure_ns=4_000_000,
):
    """A voxl-logger directory with one tracking camera and imu_apps (start-of-exposure stamps)."""
    root = Path(root)
    imu_dir, cam_dir = root / "run/mpa/imu_apps", root / "run/mpa/tracking_front"
    imu_dir.mkdir(parents=True)
    cam_dir.mkdir(parents=True)
    (root / "data/modalai").mkdir(parents=True)
    (root / "data/modalai/opencv_tracking_front_intrinsics.yml").write_text(INTRINSICS)
    duration = frames / fps
    n_imu = int(duration * imu_rate) + int(imu_rate)
    t_imu = epoch - 200_000_000 + (np.arange(n_imu) * (1e9 / imu_rate)).astype(np.int64)
    if imu_reversal is not None:
        t_imu[imu_reversal] = t_imu[imu_reversal - 2]
    lines = [
        "i,timestamp(ns),batch_id,AX(m/s2),AY(m/s2),AZ(m/s2),GX(rad/s),GY(rad/s),GZ(rad/s),T(C)"
    ]
    for i, t in enumerate(t_imu):
        s = (t - epoch) * 1e-9
        lines.append(
            f"{i},{t},{i // 10},{0.1 * math.sin(s)},0.0,9.81,{0.2 * math.cos(s)},0.05,0.0,{40 + 0.01 * i}"
        )
    (imu_dir / "data.csv").write_text("\n".join(lines) + "\n")
    world = texture()
    t_cam = epoch + (np.arange(frames) * (1e9 / fps)).astype(np.int64)
    if duplicate_frame is not None:
        t_cam[duplicate_frame] = t_cam[duplicate_frame - 1]
    lines = ["i,timestamp(ns),gain,exposure(ns),format,height,width,frame_id,reserved"]
    for i, t in enumerate(t_cam):
        x, y = 40 + 3 * i, 30 + 2 * i
        write_png(cam_dir / f"{i:05d}.png", world[y : y + 240, x : x + 320])
        lines.append(f"{i},{t},100,{exposure_ns + 1000 * i},0,240,320,{100 + i},0")
    (cam_dir / "data.csv").write_text("\n".join(lines) + "\n")
    info = {
        "log_format_version": 1,
        "note": "synthetic",
        "n_channels": 2,
        "start_time_monotonic_ns": epoch,
        "duration_s": duration,
        "channels": [
            {
                "channel": 0,
                "n_samples": n_imu,
                "n_to_skip": 0,
                "type": 1,
                "type_string": "imu",
                "pipe_path": "/run/mpa/imu_apps/",
            },
            {
                "channel": 1,
                "n_samples": frames,
                "n_to_skip": 0,
                "dropped_data": False,
                "type": 2,
                "type_string": "cam",
                "pipe_path": "/run/mpa/tracking_front/",
            },
        ],
    }
    (root / "info.json").write_text(json.dumps(info))
    return root


def state_record(t, ncam=1, flags=1):
    blob = b"S" + struct.pack(
        "<d4d3d3d3d3d3d3dB",
        t,
        0,
        0,
        0,
        1,
        1,
        2,
        3,
        0.1,
        0.2,
        0.3,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
        0.01,
        0.01,
        0.02,
        0.02,
        0.02,
        ncam,
    )
    for c in range(ncam):
        blob += struct.pack("<4d3dd", 0, 0, 0, 1, 0.01 * c, 0, 0, 0.001)
    return blob + struct.pack("<dBII", 4.5, flags, 7, 11)


class SensorProtocolTest(unittest.TestCase):
    def test_round_trip(self):
        buffer = BytesIO()
        w = S.SensorStreamWriter(buffer)
        w.camera(
            0,
            4,
            3,
            {"k": [1, 2, 3, 4], "d": [0.1, 0.2, 0.3, 0.4], "model": "equidistant"},
            0.012,
            "cam/ü",
        )
        w.camera(1, 4, 3, None, 0.0, "uncalibrated")
        w.imu(1.5, (1, 2, 3), (4, 5, 6))
        pixels = np.arange(12, dtype=np.uint8).reshape(3, 4)
        w.frame(1, 2.25, 0.004, pixels, sync_ns=77)
        w.end()
        self.assertEqual(w.bytes_written, len(buffer.getvalue()))
        buffer.seek(0)
        records = list(S.read_sensor_stream(buffer))
        kinds = [k for k, _ in records]
        self.assertEqual(kinds, ["camera", "camera", "imu", "frame", "end"])
        cam = records[0][1]
        self.assertEqual(
            (cam["cam"], cam["width"], cam["height"], cam["model"], cam["name"]),
            (0, 4, 3, 1, "cam/ü"),
        )
        self.assertEqual(cam["k"], [1, 2, 3, 4])
        self.assertAlmostEqual(cam["readout_s"], 0.012)
        self.assertEqual(records[1][1]["k"], [0, 0, 0, 0])
        imu = records[2][1]
        self.assertEqual(imu[:7], (1.5, 1, 2, 3, 4, 5, 6))
        self.assertTrue(math.isnan(imu[7]))
        frame = records[3][1]
        self.assertEqual((frame["cam"], frame["t"], frame["sync_ns"]), (1, 2.25, 77))
        np.testing.assert_array_equal(frame["pixels"], pixels)

    def test_truncated_and_bounds(self):
        buffer = BytesIO()
        w = S.SensorStreamWriter(buffer)
        w.imu(1.0, (0, 0, 0), (0, 0, 9.8))
        buffer.seek(0)
        with self.assertRaises(EOFError):
            list(S.read_sensor_stream(buffer))
        with self.assertRaises(ValueError):
            w.camera(16, 4, 4, None, 0, "x")
        with self.assertRaises(ValueError):
            w.camera(0, 10_000, 10_000, None, 0, "x")
        with self.assertRaises(ValueError):
            w.frame(0, 1.0, 0.0, np.zeros((2, 2, 3), np.uint8))


class ResultProtocolTest(unittest.TestCase):
    def test_parse_all_records(self):
        blob = b"OVVR" + struct.pack("<I", 1) + state_record(3.5, ncam=2, flags=3)
        blob += (
            b"P"
            + struct.pack("<dI", 3.5, 2)
            + struct.pack("<fffB", 1, 2, 3, 0)
            + struct.pack("<fffB", 4, 5, 6, 1)
        )
        blob += (
            b"T" + struct.pack("<dII", 3.4, 1, 1) + struct.pack("<Iff", 42, 10.5, 20.25)
        )
        text = "initialized".encode()
        blob += b"M" + struct.pack("<BI", 0, len(text)) + text + b"E\x01"
        records = list(S.read_results(BytesIO(blob)))
        state, points, tracks, message, end = records
        self.assertIsInstance(state, S.VioState)
        self.assertEqual(state.t, 3.5)
        np.testing.assert_array_equal(state.q_GtoI, [0, 0, 0, 1])
        np.testing.assert_array_equal(state.p_IinG, [1, 2, 3])
        np.testing.assert_array_equal(state.sigma_pos, [0.02, 0.02, 0.02])
        self.assertEqual(len(state.calib), 2)
        self.assertAlmostEqual(state.calib[1].p_IinC[0], 0.01)
        self.assertAlmostEqual(state.calib[0].td, 0.001)
        self.assertTrue(state.initialized and state.zupt)
        self.assertEqual((state.n_slam, state.n_msckf, state.proc_ms), (7, 11, 4.5))
        np.testing.assert_array_equal(points.kind, [0, 1])
        np.testing.assert_allclose(points.xyz[1], [4, 5, 6])
        self.assertEqual((tracks.cam, int(tracks.ids[0])), (1, 42))
        np.testing.assert_allclose(tracks.uv[0], [10.5, 20.25])
        self.assertEqual(message.text, "initialized")
        self.assertTrue(end.ok)

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            list(S.read_results(BytesIO(b"XXXX\x01\x00\x00\x00")))
        with self.assertRaises(EOFError):
            list(S.read_results(BytesIO(b"OVVR\x01\x00\x00\x00S\x00")))


class StreamLogTest(unittest.TestCase):
    def test_centering_merge_and_drops(self):
        with TemporaryDirectory() as tmp:
            log = open_log(
                make_voxl_log(tmp, frames=20, duplicate_frame=5, imu_reversal=50)
            )
            cam, imu = log.cameras[0], log.imus[0]
            buffer = BytesIO()
            stats = S.stream_log(log, S.SensorStreamWriter(buffer), [cam], imu)
            self.assertEqual(stats["dropped"], {"tracking_front": 1, "imu_apps": 1})
            self.assertEqual(stats["frames"]["tracking_front"], 19)
            offset = stats["stream_offset"]
            self.assertEqual(offset, float(int(offset)))
            buffer.seek(0)
            records = list(S.read_sensor_stream(buffer))
            times = [
                r["t"] if k == "frame" else r[0]
                for k, r in records
                if k in ("frame", "imu")
            ]
            self.assertEqual(times, sorted(times))
            self.assertGreaterEqual(min(times), 1.0)
            frames = [r for k, r in records if k == "frame"]
            # start-of-exposure stamps are centered by exposure / 2 (global shutter, readout 0)
            row = 7
            expected = cam.times[row] + 0.5 * cam.exposure[row] + offset
            got = [f for f in frames if abs(f["t"] - expected) < 1e-9]
            self.assertEqual(len(got), 1)
            self.assertAlmostEqual(got[0]["exposure_s"], cam.exposure[row], places=9)
            np.testing.assert_array_equal(got[0]["pixels"], cam.picture(row).pixels)
            self.assertTrue(np.isnan(stats["frame_times"][0][5]))
            self.assertAlmostEqual(stats["frame_times"][0][row], expected - offset)
            # an IMU sample shares a frame's instant -> it is written first
            self.assertTrue(all(np.isfinite(r[7]) for k, r in records if k == "imu"))

    def test_crop_and_settings(self):
        with TemporaryDirectory() as tmp:
            log = open_log(make_voxl_log(tmp, frames=40))
            cam, imu = log.cameras[0], log.imus[0]
            settings = {cam.name: S.CameraSettings("start", 0.010, 0.0, False)}
            plan = S.plan_stream(log, [cam], imu, settings, start=0.5, end=1.0)
            rows = plan.frame_rows[0]
            self.assertTrue(len(rows) > 0)
            np.testing.assert_allclose(
                plan.frame_times[0][rows], cam.times[rows] + 0.005
            )
            self.assertTrue(
                (
                    (plan.frame_times[0][rows] >= 0.5)
                    & (plan.frame_times[0][rows] <= 1.0)
                ).all()
            )
            self.assertEqual(S.default_settings(cam).reference, "start")
            with self.assertRaises(ValueError):
                S.plan_stream(log, [cam], imu, start=100.0)

    def test_cancel_writes_no_end_record(self):
        with TemporaryDirectory() as tmp:
            log = open_log(make_voxl_log(tmp, frames=10))
            buffer = BytesIO()
            with self.assertRaises(InterruptedError):
                S.stream_log(
                    log,
                    S.SensorStreamWriter(buffer),
                    log.cameras,
                    log.imus[0],
                    cancelled=lambda: True,
                )
            buffer.seek(0)
            with self.assertRaises(EOFError):
                list(S.read_sensor_stream(buffer))


@unittest.skipUnless(HAVE_ROSBAGS, "rosbags is not installed")
class BagStreamTest(unittest.TestCase):
    def test_sequential_bag_frames_match_random_access(self):
        typestore = get_typestore(Stores.ROS2_JAZZY)
        Image = typestore.types["sensor_msgs/msg/Image"]
        Imu = typestore.types["sensor_msgs/msg/Imu"]
        Header = typestore.types["std_msgs/msg/Header"]
        Time = typestore.types["builtin_interfaces/msg/Time"]
        Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
        Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
        rng = np.random.default_rng(1)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bag"
            base = 1_700_000_000_000_000_000
            with Bag2Writer(path, version=9) as writer:
                cams = [
                    writer.add_connection(
                        f"/cam{i}/image_raw", Image.__msgtype__, typestore=typestore
                    )
                    for i in range(2)
                ]
                imu_conn = writer.add_connection(
                    "/imu0", Imu.__msgtype__, typestore=typestore
                )
                events = []
                for k in range(400):
                    events.append((base + k * 5_000_000, "imu", k))
                for k in range(40):
                    for i in range(
                        2
                    ):  # cam1 arrives 1 ms later in log time (reordering stress)
                        events.append(
                            (
                                base + 10_000_000 + k * 50_000_000 + i * 1_000_000,
                                f"cam{i}",
                                k,
                            )
                        )
                for logtime, kind, k in sorted(events):
                    stamp = Time(
                        sec=(logtime - 2_000_000) // 1_000_000_000,
                        nanosec=(logtime - 2_000_000) % 1_000_000_000,
                    )
                    if kind == "imu":
                        msg = Imu(
                            header=Header(stamp=stamp, frame_id="imu"),
                            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                            orientation_covariance=np.zeros(9),
                            angular_velocity=Vector3(x=0.1, y=0.0, z=0.0),
                            angular_velocity_covariance=np.zeros(9),
                            linear_acceleration=Vector3(x=0.0, y=0.0, z=9.81),
                            linear_acceleration_covariance=np.zeros(9),
                        )
                        writer.write(
                            imu_conn,
                            logtime,
                            typestore.serialize_cdr(msg, Imu.__msgtype__),
                        )
                    else:
                        data = rng.integers(0, 255, 64 * 48, dtype=np.uint8)
                        msg = Image(
                            header=Header(stamp=stamp, frame_id=kind),
                            height=48,
                            width=64,
                            encoding="mono8",
                            is_bigendian=0,
                            step=64,
                            data=data,
                        )
                        writer.write(
                            cams[int(kind[-1])],
                            logtime,
                            typestore.serialize_cdr(msg, Image.__msgtype__),
                        )
            log = open_log(path)
            try:
                cameras = sorted(log.cameras, key=lambda c: c.name)
                buffer = BytesIO()
                stats = S.stream_log(
                    log, S.SensorStreamWriter(buffer), cameras, log.imus[0], start=0.5
                )
                buffer.seek(0)
                frames = [r for k, r in S.read_sensor_stream(buffer) if k == "frame"]
                self.assertEqual(len(frames), sum(stats["frames"].values()))
                self.assertLess(len(frames), 80)
                offset = stats["stream_offset"]
                for frame in frames:
                    cam = cameras[frame["cam"]]
                    row = int(
                        np.nanargmin(
                            np.abs(
                                stats["frame_times"][frame["cam"]]
                                - (frame["t"] - offset)
                            )
                        )
                    )
                    np.testing.assert_array_equal(
                        frame["pixels"], cam.picture(row).pixels
                    )
            finally:
                log.close()


@unittest.skipUnless(TRACK, "ov_zcalib_track is not built")
class ConvertTest(unittest.TestCase):
    def test_voxl_log_to_session(self):
        with TemporaryDirectory() as tmp:
            log = open_log(make_voxl_log(Path(tmp) / "log0001", frames=50))
            out = Path(tmp) / "session.bin"
            lines = []
            summary = S.convert_to_session(
                log, out, log.cameras, log.imus[0], features=120, on_line=lines.append
            )
            session = read_record(out)
            self.assertTrue(session.complete)
            self.assertEqual(session.format, 7)
            self.assertEqual(len(session.frames), 50)
            # IMU is streamed from 1 s before the first frame to 0.5 s after the last
            self.assertEqual(len(session.imu), summary["stats"]["imu"])
            self.assertGreater(summary["stats"]["imu"], 0.8 * len(log.imus[0]))
            self.assertGreater(np.mean([len(f.points) for f in session.frames[5:]]), 40)
            np.testing.assert_allclose(
                [f.exposure for f in session.frames], log.cameras[0].exposure, rtol=1e-6
            )
            meta = json.loads(Path(summary["sidecar"]).read_text())
            self.assertTrue(meta["temperature_available"])
            self.assertEqual(meta["source"]["cameras"], ["tracking_front"])
            offset = meta["time_offset_s"]
            first = log.cameras[0].times[0] + 0.5 * log.cameras[0].exposure[0] + offset
            self.assertAlmostEqual(session.frames[0].timestamp, first, places=9)
            self.assertAlmostEqual(session.imu[0, 7], log.imus[0].data[0, 7])
            with self.assertRaises(FileExistsError):
                S.convert_to_session(log, out, log.cameras, log.imus[0])

    def test_failure_removes_output(self):
        with TemporaryDirectory() as tmp:
            log = open_log(make_voxl_log(Path(tmp) / "log0001", frames=12))
            out = Path(tmp) / "session.bin"
            with self.assertRaises(InterruptedError):
                calls = iter(range(10**6))
                S.convert_to_session(
                    log,
                    out,
                    log.cameras,
                    log.imus[0],
                    cancelled=lambda: next(calls) > 2,
                )
            self.assertFalse(out.exists())
            cam = log.cameras[0]
            cam.intrinsics = None
            with self.assertRaises(ValueError):
                S.convert_to_session(log, out, [cam], log.imus[0])


@unittest.skipUnless(REPLAY, "ov_msckf_replay is not built")
class ReplayTest(unittest.TestCase):
    def test_capabilities(self):
        caps = S.replay_capabilities(REPLAY)
        self.assertEqual(caps.get("sensor_stream"), 1)
        self.assertEqual(caps.get("vio_results"), 1)
        self.assertIn("klt", caps.get("trackers", []))
        self.assertIn(caps.get("modal_flow"), (True, False))
        self.assertEqual(S.replay_capabilities(REPLAY), caps)
        found = dict(S.find_replay_binaries())
        self.assertTrue(
            any(os.path.realpath(p) == os.path.realpath(REPLAY) for p in found)
        )
        self.assertEqual(S.replay_capabilities("/nonexistent/ov_msckf_replay"), {})

    def test_incomplete_config_names_missing_keys(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "estimator_config.yaml"
            config.write_text('%YAML:1.0\nverbosity: "INFO"\nmax_cameras: 1\n')
            log = open_log(make_voxl_log(Path(tmp) / "log0001", frames=10))
            run = S.VioRun().start(config, log, log.cameras, log.imus[0])
            self.assertIsNotNone(run.wait(60))
            run.poll()
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("missing keys:", run.error)
            self.assertIn("max_slam", run.error)
            self.assertIn("relative_config_imu", run.error)

    @unittest.skipUnless(
        HAVE_ROSBAGS and IMU_APRIL.is_file(), "imu_april.bag or rosbags is unavailable"
    )
    def test_imu_april_initializes(self):
        """VI-sensor stereo bag; euroc_mav completed with this fork's newer keys and dynamic init."""
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "euroc_mav"
            shutil.copytree(REPO / "config" / "euroc_mav", config_dir)
            config = config_dir / "estimator_config.yaml"
            text = config.read_text().replace(
                "init_dyn_use: false", "init_dyn_use: true"
            )
            # The VI-sensor IMU has +x up: widen this fork's dynamic-init tilt gate.
            extra = {
                "prop_window": "0.10",
                "zupt_prop_window": "0.10",
                "init_gravity_max_angle": "85.0",
                "use_gpu": "false",
            }
            for key, value in extra.items():
                if f"\n{key}:" not in text:
                    text += f"\n{key}: {value}"
            config.write_text(text + "\n")
            log = open_log(IMU_APRIL)
            try:
                cameras = sorted(log.cameras, key=lambda c: c.name)
                tum = Path(tmp) / "traj.txt"
                run = S.VioRun().start(
                    config,
                    log,
                    cameras,
                    log.imus[0],
                    start=log.start,
                    end=log.start + 20.0,
                    tum_path=tum,
                )
                self.assertIsNotNone(run.wait(600))
                records = run.poll()
                self.assertEqual(
                    run.returncode,
                    0,
                    run.error + "\n" + "\n".join(list(run.lines)[-10:]),
                )
                states = [r for r in records if isinstance(r, S.VioState)]
                initialized = [s for s in states if s.initialized]
                self.assertGreater(len(initialized), 100)
                self.assertLess(initialized[0].t - log.start, 8.0)
                self.assertTrue(
                    any(isinstance(r, S.VioPoints) and len(r.xyz) for r in records)
                )
                tracks = [r for r in records if isinstance(r, S.VioTracks)]
                self.assertTrue(tracks)
                # Result times are log times: tracks map exactly onto streamed rows.
                for tr in tracks[:50]:
                    times = run.frame_times[tr.cam]
                    self.assertLess(np.nanmin(np.abs(times - tr.t)), 1e-6)
                self.assertIn("TrackKLT", run.tracker)
                last = initialized[-1]
                self.assertTrue(np.isfinite(last.p_IinG).all())
                self.assertLess(np.linalg.norm(last.p_IinG), 50.0)
                self.assertLess(float(last.sigma_pos.max()), 1.0)
                rows = [
                    line.split()
                    for line in tum.read_text().splitlines()
                    if not line.startswith("#")
                ]
                self.assertEqual(len(rows), len(initialized))
                self.assertAlmostEqual(float(rows[0][0]), initialized[0].t, places=6)
            finally:
                log.close()


if __name__ == "__main__":
    unittest.main()
