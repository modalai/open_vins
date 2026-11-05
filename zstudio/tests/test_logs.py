"""Log inspector: readers, checks, CLI, reports, geometry, and the studio's Inspect/VIO pages."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logfixtures import make_ros2_bag, make_voxl_log  # noqa: E402

from zstudio.checks import inspect_log  # noqa: E402
from zstudio.geometry import (  # noqa: E402
    RigConfig,
    RigEdge,
    find_rest,
    jpl_from_matrix,
    matrix_quat_xyzw,
    quat_xyzw_matrix,
    rotation_angle_deg,
    rpy_matrix,
)
from zstudio.logcli import main as cli  # noqa: E402
from zstudio.logreport import text_lines, write_html  # noqa: E402
from zstudio.logs import (
    LogError,
    open_log,
    probe_log,
    read_opencv_yaml,
    select,
    voxl_base,
)  # noqa: E402
from zstudio.logviews import envelope  # noqa: E402
from zstudio.media import (
    Picture,
    Tonemapper,
    demosaic,
    to_gray8,
    unpack_mipi10,
    unpack_mipi12,
)  # noqa: E402
from zstudio.model import envelope_rows  # noqa: E402

HAVE_ROSBAGS = importlib.util.find_spec("rosbags") is not None


def codes(inspection, severity=None):
    return {
        f.code
        for f in inspection.findings
        if severity is None or f.severity == severity
    }


def finding(inspection, code):
    return next(f for f in inspection.findings if f.code == code)


class VoxlChecks(unittest.TestCase):
    def inspect(self, **kwargs):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = make_voxl_log(Path(self.tmp.name) / "log0001", **kwargs)
        log = open_log(root)
        self.addCleanup(log.close)
        return log, inspect_log(log)

    def test_clean_log_passes_with_measured_context(self):
        log, ins = self.inspect()
        self.assertEqual(ins.verdict, "pass", [f.message for f in ins.findings])
        self.assertEqual(ins.exit_code(), 0)
        self.assertEqual(
            [c.name for c in log.cameras], ["tracking_left", "tracking_right"]
        )
        imu = log.imus[0]
        self.assertTrue(imu.temperature_available)
        self.assertAlmostEqual(1.0 / np.median(np.diff(imu.times)), 1000.0, places=3)
        self.assertEqual(log.cameras[0].stamp_reference, "start")
        self.assertIsNotNone(log.cameras[0].intrinsics)
        self.assertEqual(log.cameras[0].intrinsics["model"], "equidistant")
        # pitch-180 body->IMU with +Z specific force at rest is consistent with gravity
        self.assertEqual(finding(ins, "gravity.orientation").severity, "info")
        self.assertIn("hardware-synchronized", finding(ins, "stereo.sync").message)

    def test_imu_gap_duplicate_and_rate(self):
        _, ins = self.inspect(
            imu_gaps=range(500, 520), imu_duplicates=(900,), configured_rate=800
        )
        gap = finding(ins, "timing.gap")
        self.assertEqual(gap.severity, "problem")
        # The removed run (20 samples) plus the 2 ms hole the duplicate leaves behind.
        self.assertIn("2 timing gaps", gap.message)
        self.assertIn("~20 samples missing", gap.events[0].text)
        self.assertAlmostEqual(gap.events[0].time, 0.010 + 0.499, places=3)
        self.assertIn("timing.duplicate", codes(ins, "problem"))
        self.assertEqual(finding(ins, "rate.configured").severity, "warning")
        self.assertEqual(ins.exit_code(), 1)

    def test_frame_counter_files_and_orphans(self):
        _, ins = self.inspect(
            frame_gaps={"tracking_left": [10]},
            missing_files={"tracking_left": [5]},
            empty_files={"tracking_right": [7]},
            orphans={"tracking_right": [999]},
        )
        gap = finding(ins, "sequence.gap")
        self.assertEqual((gap.channel, gap.events[0].row), ("tracking_left", 9))
        self.assertIn("3 frames skipped", gap.message)
        self.assertEqual(finding(ins, "files.missing").events[0].row, 5)
        self.assertEqual(finding(ins, "files.empty").channel, "tracking_right")
        self.assertEqual(finding(ins, "files.orphan").severity, "warning")

    def test_thermal_shutter_frames_are_expected_others_are_problems(self):
        _, ins = self.inspect(
            cameras=(
                ("nyctis_bayer", "nyctis640", 30.0),
                ("tracking_left", "ar0144", 30.0),
            ),
            repeats={"nyctis_bayer": [10, 40], "tracking_left": [20]},
        )
        shutter = finding(ins, "thermal.shutter")
        self.assertEqual(
            (shutter.severity, shutter.channel, len(shutter.events)),
            ("expected", "nyctis_bayer", 2),
        )
        self.assertIn("every ~1.0 s", shutter.message)
        resent = finding(ins, "sequence.resent")
        self.assertEqual(
            (resent.severity, resent.channel), ("problem", "tracking_left")
        )
        self.assertEqual(ins.exit_code(), 1)

    def test_gravity_contradiction_is_a_problem(self):
        _, ins = self.inspect(body_imu_rpy=(0, 0, 180))
        gravity = finding(ins, "gravity.orientation")
        self.assertEqual(gravity.severity, "problem")
        self.assertIn("deg apart", gravity.message)

    def test_missing_pipe_and_intrinsics(self):
        _, ins = self.inspect(missing_dirs=("tracking_right",), intrinsics=False)
        self.assertEqual(finding(ins, "channel.unreadable").channel, "tracking_right")
        self.assertEqual(finding(ins, "intrinsics.missing").severity, "warning")

    def test_channel_selection_and_reload(self):
        log, ins = self.inspect()
        self.assertEqual(
            select(log.channels, only="tracking_*"), ["tracking_left", "tracking_right"]
        )
        self.assertEqual(
            select(log.channels, exclude="tracking_right"),
            ["imu_apps", "tracking_left"],
        )
        log.load(["imu_apps", "tracking_left"])
        self.assertEqual([c.name for c in log.cameras], ["tracking_left"])
        again = inspect_log(log)
        self.assertNotIn("stereo.sync", codes(again))
        log.load(["imu_apps", "tracking_left", "tracking_right"])
        self.assertEqual(len(log.cameras), 2)

    def test_picture_decode_and_payload(self):
        log, _ = self.inspect()
        picture = log.cameras[0].picture(3)
        self.assertEqual(
            (picture.kind, picture.width, picture.height), ("mono8", 64, 48)
        )
        self.assertEqual(
            log.cameras[0].payload(3),
            (Path(log.cameras[0].directory) / "00003.png").read_bytes(),
        )

    def test_config_parsing_helpers(self):
        log, _ = self.inspect(header_comment=True)
        self.assertEqual(log.sensor_for("tracking_left"), "ar0144")
        self.assertEqual(len(log.rig.edges), 3)
        self.assertEqual(voxl_base("tracking_front_misp_grey"), "tracking_front")
        self.assertEqual(voxl_base("hires_bayer"), "hires")
        data = read_opencv_yaml(log.cameras[0].intrinsics["source"])
        self.assertEqual(data["M"].shape, (3, 3))


class CliAndReports(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = make_voxl_log(
            Path(self.tmp.name) / "log0002", imu_gaps=range(100, 110)
        )

    def run_cli(self, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli([str(self.root), *args])
        return code, out.getvalue()

    def test_exit_codes_json_and_selection(self):
        code, text = self.run_cli(
            "-o", str(Path(self.tmp.name) / "out"), "--no-plots", "--json"
        )
        self.assertEqual(code, 1)
        self.assertIn("RESULT: PROBLEMS FOUND", text)
        data = json.loads((Path(self.tmp.name) / "out" / "report.json").read_text())
        self.assertEqual(data["verdict"], "problems")
        code, _ = self.run_cli(
            "--only",
            "tracking_*",
            "--no-plots",
            "-q",
            "-o",
            str(Path(self.tmp.name) / "o2"),
        )
        self.assertEqual(code, 0)
        code, text = self.run_cli("--list-channels")
        self.assertEqual(code, 0)
        self.assertIn("imu_apps", text)
        self.assertEqual(cli([str(Path(self.tmp.name) / "nope")]), 2)

    def test_json_stdout_is_clean(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli([str(self.root), "--json", "-", "--no-plots"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out.getvalue())["counts"]["problem"], 1)

    def test_html_and_text(self):
        log = open_log(self.root)
        self.addCleanup(log.close)
        ins = inspect_log(log)
        lines = text_lines(ins)
        self.assertTrue(lines[-1].startswith("RESULT: PROBLEMS FOUND"))
        path = write_html(ins, Path(self.tmp.name) / "r.html")
        html = path.read_text()
        self.assertIn('class="pill problem"', html)
        self.assertIn("tracking_left", html)
        self.assertIn("prefers-color-scheme: dark", html)


class Media(unittest.TestCase):
    def test_raw10_and_raw12_unpack(self):
        values = np.array([[0, 1023, 512, 3, 7, 1000, 2, 9]], dtype=np.uint16)
        packed = []
        for group in values[0].reshape(-1, 4):
            packed += [int(v) >> 2 for v in group] + [
                sum((int(v) & 3) << (2 * k) for k, v in enumerate(group))
            ]
        np.testing.assert_array_equal(
            unpack_mipi10(np.array(packed, np.uint8), 8, 1), values
        )
        twelve = np.array([[4095, 1, 2048, 77]], dtype=np.uint16)
        packed = []
        for a, b in twelve[0].reshape(-1, 2):
            packed += [int(a) >> 4, int(b) >> 4, (int(a) & 15) | ((int(b) & 15) << 4)]
        np.testing.assert_array_equal(
            unpack_mipi12(np.array(packed, np.uint8), 4, 1), twelve
        )

    def test_demosaic_tonemap_and_gray(self):
        bayer = np.zeros((8, 8), np.uint16)
        bayer[0::2, 0::2] = 1000  # R of an RGGB mosaic
        rgb = demosaic(bayer, "RGGB", full=False)
        self.assertEqual(rgb.shape, (4, 4, 3))
        self.assertTrue(np.all(rgb[..., 0] == 1000) and np.all(rgb[..., 2] == 0))
        out = Tonemapper(0, 0)(Picture("bayer", bayer, 10, "RGGB"))
        self.assertEqual(out.dtype, np.uint8)
        gray = to_gray8(
            Picture("mono16", np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000, 16)
        )
        self.assertEqual((gray.dtype, gray.min(), gray.max()), (np.uint8, 0, 255))

    def test_envelopes_keep_spikes(self):
        t = np.arange(20000) / 1000.0
        y = np.sin(t)
        y[12345] = 50.0
        tt, yy = envelope(t, y, 0.0, 20.0, 300)
        self.assertEqual(yy.max(), 50.0)
        self.assertLessEqual(len(tt), 600)
        rows = np.column_stack([t] + [y] * 7)
        self.assertEqual(envelope_rows(rows, 100)[:, 3].max(), 50.0)


class Geometry(unittest.TestCase):
    def test_conventions(self):
        R = rpy_matrix(10, -45, -90)
        a, b, c = np.radians([10, -45, -90])
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
        )
        Ry = np.array(
            [[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]
        )
        Rz = np.array(
            [[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]]
        )
        np.testing.assert_allclose(R, Rx @ Ry @ Rz, atol=1e-12)
        np.testing.assert_allclose(quat_xyzw_matrix(matrix_quat_xyzw(R)), R, atol=1e-12)
        from zstudio.widgets import jpl_rotation

        np.testing.assert_allclose(jpl_rotation(jpl_from_matrix(R)), R, atol=1e-12)

    def test_rig_tree(self):
        rig = RigConfig(
            "test",
            [
                RigEdge("body", "imu", rpy_matrix(0, 180, 0), np.array([1.0, 0, 0])),
                RigEdge("imu", "cam", rpy_matrix(0, 0, 90), np.array([0, 1.0, 0])),
            ],
            "body",
            "FRD",
        )
        R, t = rig.transform("body", "cam")
        np.testing.assert_allclose(t, [1.0, 1.0, 0.0], atol=1e-12)
        Rinv, tinv = rig.transform("cam", "body")
        np.testing.assert_allclose(Rinv, R.T, atol=1e-12)
        np.testing.assert_allclose(Rinv @ t + tinv, 0, atol=1e-12)
        self.assertEqual(rig.match("cam_misp_grey"), "cam")
        self.assertAlmostEqual(rotation_angle_deg(rpy_matrix(0, 0, 30)), 30.0)

    def test_find_rest(self):
        t = np.arange(4000) / 1000.0
        imu = np.zeros((len(t), 8))
        imu[:, 0] = t
        imu[:, 6] = 9.81
        imu[2000:, 1:4] = 1.0
        start, end, accel = find_rest(imu)
        self.assertLess(end, 2.0)
        self.assertAlmostEqual(accel[2], 9.81)


@unittest.skipUnless(HAVE_ROSBAGS, "rosbags is not installed")
class Bags(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_ros2_bag(self):
        for storage in ("sqlite3", "mcap"):
            with self.subTest(storage=storage):
                path = make_ros2_bag(
                    Path(self.tmp.name) / f"bag_{storage}",
                    zero_stamps=(3,),
                    storage=storage,
                )
                kind, _, channels = probe_log(path)
                self.assertEqual(kind, "ros2")
                self.assertEqual(
                    {c.kind for c in channels}, {"imu", "camera", "camera_info", "tf"}
                )
                log = open_log(path)
                self.addCleanup(log.close)
                cam = log.cameras[0]
                self.assertEqual(
                    (cam.width, cam.height, cam.encoding, cam.info_topic),
                    (32, 24, "mono8", "/cam0/camera_info"),
                )
                self.assertEqual(cam.intrinsics["k"], [40.0, 40.0, 16.0, 12.0])
                self.assertEqual(log.cameras[0].picture(1).pixels.shape, (24, 32))
                self.assertEqual(log.rig.body_convention, "FLU")
                ins = inspect_log(log)
                self.assertEqual(finding(ins, "stamp.unset").events[0].row, 3)
                latency = next(f for f in ins.channels if f.name == "/imu").facts
                self.assertTrue(
                    any("median 2.000 ms" in fact for fact in latency), latency
                )
                self.assertEqual(finding(ins, "gravity.orientation").severity, "info")

    def test_ros1_bag_sequence(self):
        path = make_ros2_bag(Path(self.tmp.name) / "a.bag", ros1=True, seq_gaps=(4,))
        log = open_log(path)
        self.addCleanup(log.close)
        self.assertEqual(log.kind, "ros1")
        ins = inspect_log(log)
        gap = finding(ins, "sequence.gap")
        self.assertEqual(gap.channel, "/cam0/image_raw")
        self.assertIn("4 frames skipped", gap.message)
        self.assertEqual(
            log.cameras[0].intrinsics["k"], [40.0, 40.0, 16.0, 12.0]
        )  # ROS 1 CameraInfo K/D

    def test_rosbags_missing_message(self):
        with self.assertRaises(LogError):
            probe_log(Path(self.tmp.name) / "nothing.bag")


def _project(intr, p):
    x, y = p[0] / p[2], p[1] / p[2]
    fx, fy, cx, cy = intr["k"]
    d = intr["d"]
    if intr["model"] == "equidistant":
        r = np.hypot(x, y)
        th = np.arctan(r)
        thd = th * (1 + d[0] * th**2 + d[1] * th**4 + d[2] * th**6 + d[3] * th**8)
        scale = thd / r if r > 1e-12 else 1.0
        xd, yd = x * scale, y * scale
    else:
        k1, k2, p1, p2 = d[:4]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.array([fx * xd + cx, fy * yd + cy])


class Orientation(unittest.TestCase):
    def test_rotated_camera_projects_to_rotated_pixels(self):
        from zstudio.geometry import image_rotation
        from zstudio.media import rotate_intrinsics

        rng = np.random.default_rng(0)
        W, H = 640, 480
        for model, d in (
            ("plumb_bob", [0.1, -0.05, 0.003, -0.002]),
            ("equidistant", [0.02, -0.01, 0.003, -0.001]),
        ):
            intr = {
                "k": [450.0, 455.0, 330.0, 240.0],
                "d": d,
                "model": model,
                "width": W,
                "height": H,
            }
            for deg in (90, 180, 270):
                rotated, w, h = rotate_intrinsics(intr, W, H, deg)
                self.assertEqual((w, h), (H, W) if deg % 180 else (W, H))
                R = image_rotation(deg)
                for _ in range(50):
                    p = np.array(
                        [
                            rng.uniform(-1, 1),
                            rng.uniform(-0.8, 0.8),
                            rng.uniform(1.5, 4),
                        ]
                    )
                    u, v = _project(intr, p)
                    expected = {
                        90: (H - 1 - v, u),
                        180: (W - 1 - u, H - 1 - v),
                        270: (v, W - 1 - u),
                    }[deg]
                    np.testing.assert_allclose(
                        _project(rotated, R.T @ p), expected, atol=1e-9
                    )

    def test_pixels_bayer_and_rpy(self):
        from zstudio.geometry import matrix_rpy
        from zstudio.media import rotate_pixels, rotated_pattern

        a = np.zeros((48, 64))
        a[10, 20] = 1
        for deg, expected in ((90, (20, 37)), (180, (37, 43)), (270, (43, 10))):
            self.assertEqual(tuple(np.argwhere(rotate_pixels(a, deg))[0]), expected)
        mosaic = np.array(
            [["RGGB"[(r % 2) * 2 + c % 2] for c in range(6)] for r in range(5)]
        )
        for deg in (90, 180, 270):
            turned = rotate_pixels(mosaic, deg)
            self.assertEqual(
                "".join(turned[i, j] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))),
                rotated_pattern("RGGB", 6, 5, deg),
            )
        for rpy in ([0, -90, 90], [12.5, -33, 170], [0, 180, 0], [30, 90, -40]):
            np.testing.assert_allclose(
                rpy_matrix(*matrix_rpy(rpy_matrix(*rpy))), rpy_matrix(*rpy), atol=1e-9
            )
        self.assertEqual(matrix_rpy(rpy_matrix(0, 180, 0)), [0.0, 180.0, 0.0])

    def test_camera_rotation_save_reopen_revert(self):
        from zstudio.geometry import RigEdge, image_rotation

        with TemporaryDirectory() as tmp:
            root = make_voxl_log(Path(tmp) / "log0005", size=(64, 48))
            conf = root / "etc/modalai/extrinsics.conf"
            log = open_log(root)
            try:
                cam = log.cameras[0]
                before = cam.picture(2).pixels.copy()
                old_k = list(cam.intrinsics["k"])
                old_edge = next(e for e in log.rig.edges if e.child == cam.name)
                edge = RigEdge(
                    old_edge.parent,
                    old_edge.child,
                    old_edge.R @ image_rotation(180),
                    old_edge.t.copy(),
                )
                log.set_orientation(cam, 180, edge)
                self.assertEqual((cam.width, cam.height), (64, 48))
                np.testing.assert_array_equal(cam.picture(2).pixels, before[::-1, ::-1])
                self.assertAlmostEqual(cam.intrinsics["k"][2], 64 - 1 - old_k[2])
                rotations, edges = log.orientation_changes()
                self.assertEqual(rotations, {cam.name: (0, 180)})
                self.assertIn(cam.name, edges)
                log.set_orientation(log.cameras[1], 90)
                self.assertEqual(
                    (log.cameras[1].width, log.cameras[1].height), (48, 64)
                )
                log.revert_orientation()
                self.assertEqual((cam.rotation, log.cameras[1].rotation), (0, 0))
                self.assertEqual(log.orientation_changes(), ({}, {}))
                log.set_orientation(cam, 180, edge)
                written = log.save_orientation()
                self.assertEqual(log.orientation_changes(), ({}, {}))
                self.assertTrue(
                    any(
                        str(p).endswith("extrinsics.conf")
                        and b is not None
                        and b.is_file()
                        for p, b in written
                    )
                )
                self.assertTrue(conf.read_text().startswith("/**"))
                self.assertEqual(
                    json.loads((root / "ov_studio.json").read_text())["rotations"],
                    {cam.name: 180},
                )
            finally:
                log.close()
            again = open_log(root)
            try:
                cam = again.cameras[0]
                self.assertEqual(cam.rotation, 180)
                saved = next(e for e in again.rig.edges if e.child == cam.name)
                np.testing.assert_allclose(
                    saved.R, old_edge.R @ image_rotation(180), atol=1e-6
                )
                np.testing.assert_array_equal(cam.picture(2).pixels, before[::-1, ::-1])
                self.assertEqual(again.orientation_changes(), ({}, {}))
            finally:
                again.close()

    def test_rotation_belongs_to_the_camera_not_the_pipe(self):
        with TemporaryDirectory() as tmp:
            root = make_voxl_log(
                Path(tmp) / "log0010",
                cameras=(
                    ("hires_bayer", "imx664", 30.0),
                    ("hires_grey", "imx664", 30.0),
                ),
                camera_names={"hires_bayer": "hires", "hires_grey": "hires"},
                intrinsics=False,
            )
            log = open_log(root)
            try:
                bayer, grey = log.cameras
                self.assertEqual((bayer.camera, grey.camera), ("hires", "hires"))
                self.assertEqual(bayer.label, "hires (hires_bayer)")
                self.assertEqual(log.camera_frame(bayer), "hires")
                log.set_orientation(bayer, 180)
                self.assertEqual((bayer.rotation, grey.rotation), (180, 180))
                self.assertEqual(log.orientation_changes()[0], {"hires": (0, 180)})
                log.save_orientation()
                self.assertEqual(
                    json.loads((root / "ov_studio.json").read_text())["rotations"],
                    {"hires": 180},
                )
            finally:
                log.close()
            (root / "ov_studio.json").write_text(
                json.dumps({"rotations": {"hires_bayer": 90}})
            )
            again = open_log(root)
            try:  # older sidecars keyed by pipe still load, as the camera's rotation
                self.assertEqual([c.rotation for c in again.cameras], [90, 90])
                self.assertEqual(again.rotations, {"hires": 90})
            finally:
                again.close()

    @unittest.skipUnless(HAVE_ROSBAGS, "rosbags is not installed")
    def test_bag_orientation_sidecar(self):
        from zstudio.geometry import RigEdge

        with TemporaryDirectory() as tmp:
            path = make_ros2_bag(Path(tmp) / "flight")
            log = open_log(path)
            try:
                cam = log.cameras[0]
                log.set_orientation(
                    cam,
                    270,
                    RigEdge("imu_link", cam.frame, rpy_matrix(0, 0, 90), np.zeros(3)),
                )
                log.save_orientation()
            finally:
                log.close()
            sidecar = Path(tmp) / "flight.ov_studio.json"
            self.assertTrue(sidecar.is_file())
            again = open_log(path)
            try:
                cam = again.cameras[0]
                self.assertEqual((cam.rotation, cam.width, cam.height), (270, 24, 32))
                self.assertEqual(cam.picture(0).pixels.shape, (32, 24))
                self.assertIn("studio edits", again.rig.source)
                self.assertEqual(
                    inspect_log(again).verdict in ("pass", "warnings", "problems"), True
                )
                self.assertFalse(
                    any(
                        f.code == "intrinsics.size" for f in inspect_log(again).findings
                    )
                )
            finally:
                again.close()


class StudioPages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def pump(self, seconds, until=lambda: False):
        end = time.time() + seconds
        while time.time() < end and not until():
            self.app.processEvents()
            time.sleep(0.005)

    def test_vio_coverage_grid_follows_estimator_config(self):
        from zstudio.vio_page import OPENVINS_GRID, VioPage

        page = VioPage(lambda: None)
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "estimator_config.yaml"
            config.write_text("%YAML:1.0\nuse_klt: true\ngrid_x: 8 # cols\ngrid_y: 4\n")
            page.config_path = str(config)
            self.assertEqual(page._config_grid(), (8, 4))
            config.write_text("%YAML:1.0\nuse_klt: true\n")
            self.assertEqual(page._config_grid(), OPENVINS_GRID)
        page.deleteLater()

    def test_overview_shows_vio_feature_coverage(self):
        """After a VIO run, the Overview shows the tracked features and their coverage."""
        from types import SimpleNamespace

        from zstudio.window import PAGE_OVERVIEW, StudioWindow

        with TemporaryDirectory() as tmp:
            root = make_voxl_log(Path(tmp) / "log0007")
            window = StudioWindow()
            try:
                window.playing = False
                window.open_path(str(root), ask=False)
                self.pump(30, lambda: window.log_loader is None and window.inspection is not None)
                window.navigate(PAGE_OVERVIEW)
                window._refresh_view()
                self.assertFalse(window.quality_card.isVisibleTo(window))
                self.assertIn("run VIO", window.metric_notes[0].text())

                # A run that tracked tracking_left only: features on its left half.
                log = window.session.log
                left = next(i for i, c in enumerate(log.cameras) if c.name == "tracking_left")
                cam = log.cameras[left]
                vio = window.vio_page
                vio.run_cameras = [cam.name]
                vio.run_grid = (5, 5)  # grid_x / grid_y of the run's estimator config
                uv = np.column_stack(
                    (np.linspace(1, cam.width / 2 - 1, 30), np.linspace(1, cam.height - 1, 30))
                )
                for t in cam.times:
                    vio.results.add_tracks(
                        SimpleNamespace(cam=0, t=float(t), ids=np.arange(30), uv=uv)
                    )
                window.offset = window.session.duration / 2  # cameras start after the IMU
                window._refresh_view()
                self.assertTrue(window.quality_card.isVisibleTo(window))
                self.assertIn("VIO", window.metric_notes[0].text())
                self.assertEqual(window.view_tabs.currentIndex(), 1)
                tiles = window.multi_views
                self.assertEqual(len(tiles[left].frame.points), 30)
                self.assertTrue(tiles[left].show_grid)
                other = 1 - left
                self.assertEqual(len(tiles[other].frame.points), 0)
                grid = window.coverage_view.grid  # only the tracked camera is summarised
                self.assertEqual(len(window.coverage_view.grids), 1)
                self.assertEqual(grid.shape, (5, 5))  # the estimator's extraction grid
                self.assertEqual(tiles[left].coverage_grid, (5, 5))
                self.assertGreater(np.count_nonzero(grid[:, :3]), 0)
                self.assertEqual(np.count_nonzero(grid[:, 3:]), 0)
                self.assertIn("/25 cells", window.coverage_text.text())
                self.assertEqual(window.metric_values[0].text(), "30")
            finally:
                window.close()
                self.app.processEvents()

    def test_channel_and_convert_dialogs(self):
        from zstudio.logdialogs import ChannelDialog, ConvertDialog
        from zstudio.logs import ChannelInfo

        channels = [
            ChannelInfo("/imu", "imu", "sensor_msgs/msg/Imu", 100),
            ChannelInfo("/cam/image_raw", "camera", "sensor_msgs/msg/Image", 10),
            ChannelInfo(
                "/clock", "other", "rosgraph_msgs/msg/Clock", 5, False, "not inspected"
            ),
        ]
        dialog = ChannelDialog("Open", "test", channels)
        self.assertEqual(dialog.selected(), ["/imu", "/cam/image_raw"])
        dialog._check(lambda c: c.kind == "imu")
        self.assertEqual(dialog.selected(), ["/imu"])
        dialog.filter.setText("cam")
        self.assertTrue(dialog.table.isRowHidden(0))
        with TemporaryDirectory() as tmp:
            root = make_voxl_log(Path(tmp) / "log0004", intrinsics=True)
            log = open_log(root)
            try:
                convert = ConvertDialog(log, tmp)
                result = convert.result_settings()
                self.assertEqual(
                    [c.name for c in result["cameras"]],
                    ["tracking_left", "tracking_right"],
                )
                self.assertEqual(result["imu"].name, "imu_apps")
                self.assertEqual(result["settings"]["tracking_left"].reference, "start")
                self.assertTrue(str(result["out"]).endswith(".bin"))
            finally:
                log.close()

    def test_quick_rotation_unsaved_bar_and_save(self):
        from unittest import mock

        from PyQt5.QtWidgets import QMessageBox

        from zstudio.window import StudioWindow

        with TemporaryDirectory() as tmp:
            root = make_voxl_log(Path(tmp) / "log0006")
            window = StudioWindow()
            try:
                window.open_path(str(root), ask=False)
                self.pump(
                    30,
                    lambda: window.log_loader is None and window.inspection is not None,
                )
                page = window.inspect_page
                page.select_channel("tracking_left")
                page.action.emit("rotate-cw")
                page.action.emit("rotate-cw")
                self.pump(30, lambda: window.log_loader is None)
                cam = window.log_source.cameras[0]
                self.assertEqual(cam.rotation, 180)
                self.assertTrue(page.unsaved.isVisibleTo(page))
                self.assertIn("tracking_left rotated 180", page.unsaved_label.text())
                page.action.emit("orientation-revert")
                self.pump(30, lambda: window.log_loader is None)
                self.assertEqual(cam.rotation, 0)
                self.assertFalse(page.unsaved.isVisibleTo(page))
                page.action.emit("rotate-ccw")
                self.pump(30, lambda: window.log_loader is None)
                with mock.patch.object(
                    QMessageBox, "question", return_value=QMessageBox.Save
                ):
                    page.action.emit("orientation-save")
                self.pump(30, lambda: window.log_loader is None)
                self.assertFalse(page.unsaved.isVisibleTo(page))
                self.assertEqual(
                    json.loads((root / "ov_studio.json").read_text())["rotations"],
                    {"tracking_left": 270},
                )
                self.assertEqual(
                    (window.session.cameras[0].width, window.session.cameras[0].height),
                    (48, 64),
                )
            finally:
                window.close()
                self.pump(1)

    def test_orientation_dialog(self):
        from zstudio.geometry import image_rotation
        from zstudio.logdialogs import CameraOrientationDialog

        with TemporaryDirectory() as tmp:
            root = make_voxl_log(Path(tmp) / "log0007")
            log = open_log(root)
            try:
                cam = log.cameras[0]
                old = next(e for e in log.rig.edges if e.child == cam.name)
                dialog = CameraOrientationDialog(log, cam, 3)
                self.assertTrue(dialog.had_edge)
                self.assertIsNone(dialog.edge())  # nothing changed yet
                # The fixture mounts this camera with image "down" pointing up.
                self.assertIn("upside down", dialog.check.text())
                dialog.rotation.setCurrentIndex(2)  # 180 deg, extrinsics follow
                edge = dialog.edge()
                np.testing.assert_allclose(
                    edge.R, old.R @ image_rotation(180), atol=1e-6
                )
                self.assertIn("upright", dialog.check.text())
                # Gravity maps into a non-IMU parent with the transpose: expressing the
                # same camera relative to body (pitched 180 deg) flips the verdict.
                dialog.parent_box.setCurrentText("body")
                self.assertIn("upside down", dialog.check.text())
                self.assertIn("en_rotate", dialog.snippet.toPlainText())
                self.assertEqual(dialog.rotation_value, 180)
            finally:
                log.close()

    def test_orientation_dialog_without_extrinsics(self):
        from zstudio.logdialogs import CameraOrientationDialog

        with TemporaryDirectory() as tmp:
            root = make_voxl_log(Path(tmp) / "log0008")
            conf = root / "etc/modalai/extrinsics.conf"
            header, body = conf.read_text().split("*/", 1)
            data = json.loads(body)
            data["extrinsics"] = [
                e for e in data["extrinsics"] if e["child"] != "tracking_right"
            ]
            conf.write_text(header + "*/" + json.dumps(data))
            log = open_log(root)
            try:
                cam = log.cameras[1]
                dialog = CameraOrientationDialog(log, cam, 0)
                self.assertFalse(dialog.had_edge)
                self.assertFalse(dialog.follow.isEnabled())
                dialog.rotation.setCurrentIndex(2)
                self.assertIsNone(
                    dialog.edge()
                )  # rotating alone writes no placeholder entry
                self.assertIn("no extrinsics", dialog.check.text())
                self.assertIn("No extrinsics entry", dialog.snippet.toPlainText())
                dialog.t[0].setValue(0.05)
                self.assertIsNotNone(dialog.edge())
            finally:
                log.close()

    def test_rapid_rotations_end_consistent(self):
        from zstudio.window import StudioWindow

        with TemporaryDirectory() as tmp:
            root = make_voxl_log(Path(tmp) / "log0009")
            window = StudioWindow()
            try:
                window.open_path(str(root), ask=False)
                self.pump(
                    30,
                    lambda: window.log_loader is None and window.inspection is not None,
                )
                page = window.inspect_page
                page.select_channel("tracking_left")
                for _ in range(3):
                    page.action.emit("rotate-cw")
                self.pump(
                    30,
                    lambda: (
                        window.log_loader is None
                        and not getattr(window, "_reinspect_queued", None)
                    ),
                )
                self.pump(0.3)
                cam = window.log_source.cameras[0]
                self.assertEqual(cam.rotation, 270)
                self.assertEqual(
                    (window.session.cameras[0].width, window.session.cameras[0].height),
                    (48, 64),
                )
                self.assertIn("rotated 270", page.unsaved_label.text())
            finally:
                window.close()
                self.pump(1)

    def test_open_inspect_seek_and_toggle(self):
        from zstudio.window import PAGE_INSPECT, PAGE_OVERVIEW, StudioWindow

        with TemporaryDirectory() as tmp:
            root = make_voxl_log(Path(tmp) / "log0003", repeats={"tracking_left": [12]})
            window = StudioWindow()
            try:
                window.open_path(str(root), ask=False)
                self.pump(
                    30,
                    lambda: window.log_loader is None and window.inspection is not None,
                )
                self.assertEqual(window.pages.currentIndex(), PAGE_INSPECT)
                self.assertEqual(window.inspection.verdict, "problems")
                page = window.inspect_page
                self.assertEqual(page.table.rowCount(), 3)
                event = finding(window.inspection, "sequence.resent").events[0]
                page.seek.emit(event.time)
                self.assertAlmostEqual(
                    window.session.start + window.offset, event.time, places=6
                )
                # The visible page's view decodes; hidden views do not.
                preview = page.preview
                self.pump(
                    5,
                    lambda: (
                        preview.frame is not None and preview.frame.image is not None
                    ),
                )
                self.assertIsNotNone(preview.frame.image)
                hidden = (
                    window.feature_view.frame
                )  # Overview is hidden: no log image decoded for it
                self.assertTrue(hidden is None or hidden.image is None)
                window.navigate(PAGE_OVERVIEW)
                # Several cameras open on the grid; every tile decodes.
                self.assertEqual(window.view_tabs.currentIndex(), 1)
                tiles = window.multi_views
                self.pump(
                    5,
                    lambda: all(
                        v.frame is not None and v.frame.image is not None for v in tiles
                    ),
                )
                self.assertTrue(all(v.frame.image is not None for v in tiles))
                window.view_switch.set_value(0, emit=True)
                main = window.feature_view
                self.pump(
                    5, lambda: main.frame is not None and main.frame.image is not None
                )
                self.assertIsNotNone(main.frame.image)
                # Playback from the Inspect page's own transport bar keeps images moving.
                window.navigate(PAGE_INSPECT)
                window.seek_time(window.session.start)
                start = window.offset
                shown = set()
                page.transport.play.emit()
                self.assertTrue(window.playing)
                end = time.time() + 1.5
                while time.time() < end:
                    self.app.processEvents()
                    time.sleep(0.005)
                    if preview.frame is not None and preview.frame.image is not None:
                        shown.add(preview.frame.seq)
                page.transport.play.emit()
                self.assertFalse(window.playing)
                self.assertGreater(window.offset - start, 0.5)
                self.assertGreater(len(shown), 5)
                page.transport.step.emit(1)
                self.assertGreater(window.offset, 0.0)
                window._channels_changed(["imu_apps", "tracking_left"])
                self.pump(
                    30,
                    lambda: (
                        window.log_loader is None
                        and len(window.log_source.cameras) == 1
                    ),
                )
                self.assertEqual(
                    [c.name for c in window.log_source.cameras], ["tracking_left"]
                )
                self.assertEqual(len(window.session.cameras), 1)
                self.assertEqual(
                    [b.cam.name for b in window.calibration_page.camera_boxes],
                    ["tracking_left"],
                )
            finally:
                window.close()
                self.pump(1)


if __name__ == "__main__":
    unittest.main()
