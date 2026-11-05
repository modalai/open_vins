"""Persisted-format, image transport, process, and desktop regression checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import os
import struct
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication
from zstudio.engine import calibration_command, capture_command, stage_from_log
from zstudio.model import (
    CAMERA_STRUCT,
    FRAME_STRUCT,
    IMU_STRUCT,
    coverage,
    read_record,
    read_report,
)
from zstudio.sources import SourceSettings, image_from_message


def record_bytes(version=7):
    seed = (
        [1, 0, 1, 0, 0, 1] * 2
        + [0, 0, 0, 1]
        + [0] * 9
        + [9.81, 1.0e-4, 1.0e-5, 2.0e-3, 3.0e-3]
    )
    blob = struct.pack("<III", 0x5343564F, version, 2) + struct.pack("<30d", *seed)
    for index in range(2):
        blob += CAMERA_STRUCT.pack(
            400,
            401,
            320,
            240,
            0,
            0,
            0,
            0,
            index,
            0,
            0,
            0,
            1,
            0.01 * index,
            0,
            0,
            0.002,
            0.007,
            640,
            480,
            30,
            1,
        )
    blob += struct.pack("<Bi", 0, 1)
    if version >= 6:
        blob += struct.pack("<2d", 0.7, 1.2)
    blob += b"\x00" + IMU_STRUCT.pack(100.0, 0.1, 0.2, 0.3, 0, 0, 9.81, 32.0)
    blob += (
        b"\x01"
        + FRAME_STRUCT.pack(100.01, 0.004, 32.0, 42, 1, 1)
        + struct.pack("<Iff", 0xFFFFFFFF, 125.5, 240.25)
    )
    if version == 7:
        blob += b"\x02" + struct.pack("<IQQ", 0x4543564F, 1, 1)
    return blob


class FormatContracts(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = Path(self.temp.name) / "session.bin"

    def tearDown(self):
        self.temp.cleanup()

    def test_legacy_and_current_records_preserve_pixels_and_hardware(self):
        for version in (5, 6, 7):
            with self.subTest(version=version):
                self.path.write_bytes(record_bytes(version))
                session = read_record(self.path)
                self.assertEqual(session.complete, version == 7)
                self.assertEqual(len(session.cameras), 2)
                self.assertEqual(session.frames[0].points["id"][0], 0xFFFFFFFF)
                self.assertEqual(session.frames[0].cam, 1)
                self.assertAlmostEqual(session.cameras[1].tr, 0.007)
                self.assertAlmostEqual(
                    session.cameras[0].sigma, 0 if version == 5 else 0.7
                )
                self.assertAlmostEqual(session.imu[0, 0], 100.0)
                self.assertIsNone(session.frame_at(0, 1))
                self.assertEqual(session.frame_at(0.02, 1).seq, 42)

    def test_partial_records_and_incorrect_completion_cannot_appear_complete(self):
        good = record_bytes()
        for bad in (
            good[:-21],
            good[:-1],
            good + b"junk",
            good[:-16] + struct.pack("<QQ", 999, 1),
            good[:15],
        ):
            with self.subTest(size=len(bad)):
                self.path.write_bytes(bad)
                with self.assertRaises(ValueError):
                    read_record(self.path)

    def test_unsupported_format_and_invalid_camera_are_rejected(self):
        for version in (4, 8):
            blob = bytearray(record_bytes())
            struct.pack_into("<I", blob, 4, version)
            self.path.write_bytes(blob)
            with self.assertRaises(ValueError):
                read_record(self.path)
        blob = bytearray(record_bytes())
        struct.pack_into("<d", blob, 12 + 240, float("nan"))
        self.path.write_bytes(blob)
        with self.assertRaises(ValueError):
            read_record(self.path)

    def test_loading_can_be_cancelled(self):
        self.path.write_bytes(record_bytes())
        with self.assertRaises(InterruptedError):
            read_record(self.path, lambda: True)

    def test_coverage_ignores_outside_pixels(self):
        self.path.write_bytes(record_bytes())
        session = read_record(self.path)
        points = session.frames[0].points.copy()
        self.assertEqual(coverage(points, 640, 480).sum(), 1)
        points["u"] = -1
        self.assertEqual(coverage(points, 640, 480).sum(), 0)

    def test_report_schema_and_null_precision(self):
        path = self.path.with_suffix(".json")
        report = {
            "schema": 1,
            "verdict": {"state": "ABORT"},
            "posterior": {"sigma": [None]},
        }
        path.write_text(json.dumps(report))
        self.assertIsNone(read_report(path)["posterior"]["sigma"][0])
        path.write_text("[]")
        with self.assertRaises(ValueError):
            read_report(path)


class ImageContracts(unittest.TestCase):
    def test_bgr_stride_and_detached_lifetime(self):
        data = bytearray(
            [0, 0, 255, 0, 255, 0, 99, 99, 255, 0, 0, 255, 255, 255, 99, 99]
        )
        msg = SimpleNamespace(
            encoding="bgr8", width=2, height=2, step=8, data=data, is_bigendian=False
        )
        image = image_from_message(msg)
        data[:] = bytes(len(data))
        self.assertEqual(image.pixelColor(0, 0).getRgb()[:3], (255, 0, 0))
        self.assertEqual(image.pixelColor(0, 1).getRgb()[:3], (0, 0, 255))

    def test_big_endian_mono16(self):
        msg = SimpleNamespace(
            encoding="mono16",
            width=2,
            height=1,
            step=4,
            data=bytes([255, 0, 128, 0]),
            is_bigendian=True,
        )
        image = image_from_message(msg)
        self.assertEqual(image.pixelColor(0, 0).red(), 255)
        self.assertEqual(image.pixelColor(1, 0).red(), 128)

    def test_short_stride_and_unknown_encoding(self):
        msg = SimpleNamespace(
            encoding="bgr8",
            width=2,
            height=1,
            step=4,
            data=bytes(4),
            is_bigendian=False,
        )
        with self.assertRaises(ValueError):
            image_from_message(msg)
        msg.encoding = "bayer_rggb8"
        with self.assertRaises(ValueError):
            image_from_message(msg)


class ProcessContracts(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_paths_are_literal_and_record_readout_is_authoritative(self):
        source = self.root / "input $(echo unsafe).bin"
        source.touch()
        program, args = calibration_command(
            "/bin/true",
            source,
            self.root / "out.yaml",
            self.root / "report.json",
            readout_s=0.123,
        )
        self.assertEqual(program, "/bin/true")
        self.assertEqual(args[1], str(source))
        self.assertNotIn("--tr", args)
        _, args = calibration_command(
            "/bin/true",
            self.root,
            self.root / "out.yaml",
            self.root / "report.json",
            voxl=True,
            readout_s=0.007,
        )
        self.assertIn("--tr", args)
        self.assertIn("0.007", args)

    def test_existing_outputs_are_preserved(self):
        source = self.root / "session.bin"
        source.touch()
        output = self.root / "result.yaml"
        output.write_text("existing")
        with self.assertRaises(ValueError):
            calibration_command("/bin/true", source, output, self.root / "report.json")
        self.assertEqual(output.read_text(), "existing")

    def test_ros_capture_without_temperature_keeps_that_provenance(self):
        source = self.root / "session.bin"
        source.touch()
        Path(str(source) + ".capture.json").write_text(
            json.dumps({"temperature_available": False})
        )
        _, args = calibration_command(
            "/bin/true", source, self.root / "result.yaml", self.root / "report.json"
        )
        self.assertIn("--temperature-unavailable", args)

    def test_capture_quotes_ros2_string_parameters_and_remaps_tracks(self):
        settings = SourceSettings(tracks_topic="/custom/tracks")
        _, args = capture_command(
            "/bin/true", settings, self.root / "record with spaces.bin"
        )
        value = next(a for a in args if a.startswith("record_path:="))[
            len("record_path:=") :
        ]
        self.assertEqual(json.loads(value), str(self.root / "record with spaces.bin"))
        self.assertIn("/ov_zcalib/tracks:=/custom/tracks", args)

    def test_aborts_take_priority_over_earlier_stage_messages(self):
        self.assertEqual(
            stage_from_log("bootstrap\n[session] COMMIT:\nstate: ABORT"), "ABORT"
        )
        self.assertIsNone(stage_from_log("some unrelated progress"))


class DesktopContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, cls.temp.name)
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_replay_navigation_and_sparse_reports(self):
        from zstudio.widgets import STYLE
        from zstudio.window import StudioWindow

        self.app.setStyleSheet(STYLE)
        window = StudioWindow()
        window.playing = False
        window.show()
        self.app.processEvents()
        before = window.offset
        window.step_frame(1)
        self.assertGreater(window.offset, before)
        window.step_frame(-1)
        self.assertLessEqual(window.offset, before + 0.001)
        window.start_calibration()
        self.assertFalse(window.job_active)
        self.assertIn("Open a session.bin", window.notice.text())
        report = {
            "schema": 1,
            "verdict": {"state": "ABORT", "abort_reason": "insufficient motion"},
            "blocks": [
                {"name": "td", "cam": 0, "committed": False, "worst_sigma": None}
            ],
            "posterior": {"labels": ["td"], "sigma": [None], "prior_sigma": [0.01]},
        }
        window._show_report(report, "/tmp/test.report.json")
        self.assertEqual(window.blocks_table.item(0, 2).text(), "SEED RETAINED")
        self.assertEqual(window.posterior_table.item(0, 1).text(), "—")
        for index in range(4):
            window.navigate(index)
            self.app.processEvents()
            self.assertFalse(window.grab().isNull())
        window.close()
        self.app.processEvents()


    def test_coverage_shown_for_every_camera(self):
        """Feature coverage stays visible in the multi-camera view, per camera."""
        import numpy as np
        from zstudio.model import POINT_DTYPE, Camera, Frame, Session
        from zstudio.window import PAGE_OVERVIEW, StudioWindow

        def points(u0, u1):
            pts = np.zeros(40, dtype=POINT_DTYPE)
            pts["id"] = np.arange(40)
            pts["u"] = np.linspace(u0, u1, 40)
            pts["v"] = np.linspace(20, 460, 40)
            return pts

        frames = []
        for seq, stamp in enumerate(np.arange(0.0, 2.0, 0.05)):
            frames.append(Frame(float(stamp), points(10, 300), cam=0, seq=seq))
            frames.append(Frame(float(stamp), points(330, 630), cam=1, seq=seq))
        t = np.arange(0.0, 2.0, 0.005)
        imu = np.column_stack((t, np.zeros((len(t), 6)), np.full(len(t), 25.0)))
        session = Session("two cameras", [Camera(), Camera()], imu, frames)
        window = StudioWindow()
        try:
            window.playing = False
            window._session_loaded(session)
            window.show()
            self.app.processEvents()
            self.assertEqual(window.pages.currentIndex(), PAGE_OVERVIEW)
            self.assertEqual(window.view_tabs.currentIndex(), 1)  # All cameras
            self.assertTrue(window.grid_check.isVisibleTo(window))
            self.assertTrue(window.quality_card.isVisibleTo(window))
            self.assertTrue(all(view.show_grid for view in window.multi_views))
            window._refresh_view()
            grids = window.coverage_view.grids
            self.assertEqual(len(grids), 2)
            self.assertGreater(np.count_nonzero(grids[0][1][:, :4]), 0)  # camera 0: left half
            self.assertEqual(np.count_nonzero(grids[1][1][:, :4]), 0)
            self.assertGreater(np.count_nonzero(grids[1][1][:, 4:]), 0)
            window.grid_check.setChecked(False)
            self.assertFalse(any(view.show_grid for view in window.multi_views))
            window.view_switch.set_value(0, emit=True)
            window._refresh_view()
            self.assertEqual(len(window.coverage_view.grids), 1)
        finally:
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
