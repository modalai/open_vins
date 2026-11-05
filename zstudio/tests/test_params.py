"""Format-preserving parameter files, the QGC-style editor, and OpenVINS config generation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import shutil
import struct
import unittest
import zlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from zstudio.geometry import rpy_matrix
from zstudio.logs import open_log
from zstudio.params import ParamFile, ParamSet
from zstudio import vioconfig

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"
VOXL_CONF = """/**
 * Test configuration: comments must survive every save.
 * "string with // and : inside the header"
 */
{
\t"version":\t0.1,
\t"imu0_enable":\ttrue,
\t"imu0_sample_rate_hz":\t1000,
\t"imu0_lp_cutoff_freq_hz":\t92,
\t"mbps":\t1.2,
\t"note":\t"keep \\"quotes\\" and \\u00b0",
\t"offsets":\t[0.5, -1e-05, 3],
\t"rows":\t[[1, 0], [0, 1]],
\t"cameras":\t[{
\t\t\t"name":\t"tracking_left",
\t\t\t"type":\t"ar0144",
\t\t\t"fps":\t30,
\t\t\t"roi":\t{
\t\t\t\t"center_x":\t0.5
\t\t\t}
\t\t}, {
\t\t\t"name":\t"hires",
\t\t\t"fps":\t60
\t\t}]
}
"""


def _png(path, width=8, height=8):
    """A tiny valid 8-bit grayscale PNG without Qt or OpenCV."""

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(
        b"\x00" + bytes((x * 31 + y * 7) & 255 for x in range(width))
        for y in range(height)
    )
    blob = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    )
    blob += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    Path(path).write_bytes(blob)


def make_voxl_log(
    root, intrinsics=("tracking_left", "tracking_right"), synchronized=True
):
    """A small, self-consistent voxl-logger directory: 1 kHz IMU and two 30 Hz AR0144 cameras."""
    root = Path(root)
    epoch = 1_000_000_000
    duration = 2.0
    imu_t = epoch + (np.arange(int(duration * 1000)) * 1_000_000).astype(np.int64)
    cam_t = (
        epoch
        + 5_000_000
        + (np.arange(int(duration * 30)) * 33_333_333).astype(np.int64)
    )
    channels = [
        {
            "channel": 0,
            "n_samples": len(imu_t),
            "type": 1,
            "type_string": "imu",
            "pipe_path": "/run/mpa/imu_apps/",
        },
    ]
    imu_dir = root / "run/mpa/imu_apps"
    imu_dir.mkdir(parents=True)
    with open(imu_dir / "data.csv", "w") as stream:
        stream.write(
            "i,timestamp(ns),batch_id,AX(m/s2),AY(m/s2),AZ(m/s2),GX(rad/s),GY(rad/s),GZ(rad/s),T(C)\n"
        )
        for i, t in enumerate(imu_t):
            stream.write(f"{i},{t},{i // 30},0.01,-0.02,9.81,0.001,0.0,-0.001,40.0\n")
    for c, name in enumerate(("tracking_left", "tracking_right"), 1):
        directory = root / "run/mpa" / name
        directory.mkdir(parents=True)
        stamps = cam_t if synchronized or c == 1 else cam_t + 7_000_000
        with open(directory / "data.csv", "w") as stream:
            stream.write(
                "i,timestamp(ns),gain,exposure(ns),format,height,width,frame_id,reserved\n"
            )
            for i, t in enumerate(stamps):
                stream.write(f"{i},{t},100,5000000,0,40,64,{i + 10},0\n")
                _png(directory / f"{i:05d}.png", 64, 40)
        channels.append(
            {
                "channel": c,
                "n_samples": len(stamps),
                "type": 2,
                "type_string": "cam",
                "pipe_path": f"/run/mpa/{name}/",
            }
        )
    info = {
        "log_format_version": 1,
        "note": "bench",
        "n_channels": len(channels),
        "start_time_monotonic_ns": epoch,
        "duration_s": duration,
        "channels": channels,
    }
    (root / "info.json").write_text(json.dumps(info, indent="\t"))
    etc = root / "etc/modalai"
    etc.mkdir(parents=True)
    (etc / "extrinsics.conf").write_text(
        "/**\n * Extrinsic Configuration File\n */\n"
        + json.dumps(
            {
                "name": "test_rig",
                "extrinsics": [
                    {
                        "parent": "body",
                        "child": "imu_apps",
                        "T_child_wrt_parent": [0.01, 0.0, 0.0],
                        "RPY_parent_to_child": [0, 180, 0],
                    },
                    {
                        "parent": "imu_apps",
                        "child": "tracking_left",
                        "T_child_wrt_parent": [0.07, 0.02, 0.01],
                        "RPY_parent_to_child": [0, -90, 90],
                    },
                    {
                        "parent": "imu_apps",
                        "child": "tracking_right",
                        "T_child_wrt_parent": [0.07, -0.02, 0.01],
                        "RPY_parent_to_child": [0, -90, 90],
                    },
                ],
            },
            indent="\t",
        )
    )
    (etc / "voxl-camera-server.conf").write_text(
        json.dumps(
            {
                "cameras": [
                    {"type": "ar0144", "name": "tracking_left"},
                    {"type": "ar0144", "name": "tracking_right"},
                ]
            }
        )
    )
    data = root / "data/modalai"
    data.mkdir(parents=True)
    for k, name in enumerate(intrinsics):
        (data / f"opencv_{name}_intrinsics.yml").write_text(
            "%YAML:1.0\n---\nM: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n"
            f"   data: [ {30.5 + k}, 0., 32.25, 0.,\n       {30.75 + k}, 20.125, 0., 0., 1. ]\n"
            "D: !!opencv-matrix\n   rows: 4\n   cols: 1\n   dt: d\n   data: [ 0.05, 0.01, -0.002,\n       0.0004 ]\n"
            "reprojection_error: 0.25\nwidth: 64\nheight: 40\ndistortion_model: fisheye\n"
        )
    return root


class RoundTrip(unittest.TestCase):
    def test_every_shipped_config_round_trips_byte_identical(self):
        files = sorted(CONFIG.glob("*/*.yaml"))
        self.assertGreater(len(files), 30)
        with TemporaryDirectory() as tmp:
            for path in files:
                with self.subTest(path=str(path.relative_to(REPO))):
                    pf = ParamFile.load(path)
                    self.assertTrue(pf.params)
                    self.assertEqual(pf.render(), pf.text)
                    target = Path(tmp) / path.name
                    pf.save(target)
                    self.assertEqual(target.read_bytes(), path.read_bytes())

    def test_voxl_json_round_trip_and_structure(self):
        pf = ParamFile.parse(VOXL_CONF, "voxl-imu-server.conf")
        self.assertEqual(pf.format, "json")
        self.assertEqual(pf.render(), VOXL_CONF)
        self.assertEqual(pf.value("imu0_sample_rate_hz"), 1000)
        self.assertEqual(pf.get("imu0_sample_rate_hz").group, "imu0")
        self.assertEqual(pf.value("note"), 'keep "quotes" and °')
        self.assertEqual(pf.get("offsets").type, "list")
        self.assertEqual(pf.get("rows").type, "matrix")
        self.assertEqual(pf.value("cameras[0].roi.center_x"), 0.5)
        self.assertEqual(pf.get("cameras[1].fps").group, "cameras[1] (hires)")

    def test_real_voxl_confs_round_trip_when_available(self):
        confs = sorted((Path.home() / "Desktop/log0005/etc/modalai").glob("*.conf"))
        if not confs:
            self.skipTest("no local VOXL log configs")
        for path in confs:
            with self.subTest(path=path.name):
                pf = ParamFile.load(path)
                self.assertEqual(pf.render(), pf.text)


class Edits(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for name in (
            "estimator_config.yaml",
            "kalibr_imu_chain.yaml",
            "kalibr_imucam_chain.yaml",
        ):
            shutil.copy(CONFIG / "euroc_mav" / name, self.dir / name)
        self.path = self.dir / "estimator_config.yaml"
        self.original = self.path.read_text().splitlines()

    def tearDown(self):
        self.tmp.cleanup()

    def changed_lines(self, path, original):
        after = Path(path).read_text().splitlines()
        self.assertEqual(len(after), len(original))
        return {i: (a, b) for i, (a, b) in enumerate(zip(original, after)) if a != b}

    def line(self, key):
        return next(
            line
            for line in self.path.read_text().splitlines()
            if line.startswith(key + ":")
        )

    def test_typed_edits_change_only_their_values(self):
        pf = ParamFile.load(self.path)
        pf.set("max_clones", 15)
        pf.set("init_dyn_min_rec_cond", 2e-12)
        pf.set("up_msckf_sigma_px", 1.5)
        pf.set("up_slam_sigma_px", "2")
        pf.set("use_fej", "false")
        pf.set("integration", "ANALYTICAL")
        pf.set("verbosity", "WARNING")
        pf.set("init_dyn_bias_g", [0.1, 0, -2e-3])
        pf.save()
        changes = self.changed_lines(self.path, self.original)
        self.assertEqual(len(changes), 8)
        self.assertEqual(
            self.line("max_clones"),
            "max_clones: 15 # how many clones in the sliding window",
        )
        self.assertTrue(
            self.line("init_dyn_min_rec_cond").startswith(
                "init_dyn_min_rec_cond: 2e-12 #"
            )
        )
        self.assertEqual(self.line("up_msckf_sigma_px"), "up_msckf_sigma_px: 1.5")
        self.assertEqual(self.line("up_slam_sigma_px"), "up_slam_sigma_px: 2")
        self.assertTrue(
            self.line("use_fej").startswith("use_fej: false # if first-estimate")
        )
        self.assertTrue(
            self.line("integration").startswith('integration: "analytical" # discrete')
        )
        self.assertTrue(self.line("verbosity").startswith('verbosity: "WARNING" #'))
        self.assertTrue(
            self.line("init_dyn_bias_g").startswith(
                "init_dyn_bias_g: [ 0.1, 0.0, -0.002 ] #"
            )
        )
        reloaded = ParamFile.load(self.path)
        self.assertEqual(reloaded.value("init_dyn_bias_g"), [0.1, 0.0, -0.002])
        self.assertEqual(reloaded.get("up_msckf_sigma_px").type, "float")

    def test_invalid_values_are_rejected_with_reasons(self):
        pf = ParamFile.load(self.path)
        cases = {
            "max_clones": (1.5, "integer"),
            "max_cameras": (0, ">= 1"),
            "integration": ("euler", "one of"),
            "use_fej": ("maybe", "true or false"),
            "gravity_mag": ("nine", "number"),
            "init_dyn_bias_g": ("[1, x]", "number"),
            "knn_ratio": (1.5, "<= 1"),
        }
        for key, (value, reason) in cases.items():
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as caught:
                    pf.set(key, value)
                self.assertIn(reason, str(caught.exception))
        self.assertFalse(pf.has_unsaved())

    def test_matrix_rows_keep_layout_and_untouched_numbers(self):
        path = self.dir / "kalibr_imucam_chain.yaml"
        original = path.read_text().splitlines()
        pf = ParamFile.load(path)
        p = pf.get("cam0.T_imu_cam")
        self.assertEqual(p.type, "matrix")
        self.assertEqual(p.shape, (4, 4))
        rows = [list(r) for r in p.value]
        rows[0][3] = -0.05
        pf.set("cam0.T_imu_cam", rows)
        with self.assertRaises(ValueError):
            pf.set("cam0.T_imu_cam", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pf.save()
        changes = self.changed_lines(path, original)
        self.assertEqual(len(changes), 1)
        ((before, after),) = changes.values()
        self.assertEqual(before.rsplit(",", 1)[0], after.rsplit(",", 1)[0])
        self.assertTrue(after.rstrip().endswith("-0.05]"))
        self.assertEqual(ParamFile.load(path).value("cam0.T_imu_cam")[0][3], -0.05)

    def test_json_edits_keep_header_and_style(self):
        pf = ParamFile.parse(VOXL_CONF, "x.conf")
        pf.set("imu0_sample_rate_hz", 800)
        pf.set("cameras[0].fps", 15)
        pf.set("note", 'new "value"')
        pf.set("offsets", [1, 2])
        pf.set("imu0_enable", False)
        text = pf.render()
        self.assertTrue(text.startswith(VOXL_CONF.split("{", 1)[0]))
        self.assertIn('\t"imu0_sample_rate_hz":\t800,\n', text)
        self.assertIn('\t\t\t"fps":\t15,\n', text)
        self.assertIn('"note":\t"new \\"value\\"",', text)
        self.assertIn('"offsets":\t[1.0, 2.0],', text)
        self.assertIn('"imu0_enable":\tfalse,', text)
        reparsed = ParamFile.parse(text, "x.conf")
        self.assertEqual(reparsed.value("cameras[0].fps"), 15)
        self.assertEqual(reparsed.value("note"), 'new "value"')

    def test_param_set_follows_relative_configs(self):
        ps = ParamSet.from_estimator_config(self.path)
        self.assertEqual(
            [f.name for f in ps.files],
            [
                "estimator_config.yaml",
                "kalibr_imu_chain.yaml",
                "kalibr_imucam_chain.yaml",
            ],
        )
        self.assertEqual(ps.get("imu0.update_rate").file.name, "kalibr_imu_chain.yaml")
        keys = [k for k, _ in ps.items()]
        self.assertEqual(len(keys), len(set(keys)))
        ps.set("imu0.update_rate", 250)
        self.assertTrue(ps.has_unsaved())
        saved = ps.save()
        self.assertEqual([Path(p).name for p in saved], ["kalibr_imu_chain.yaml"])
        self.assertEqual(self.path.read_text().splitlines(), self.original)
        self.assertFalse(ps.has_unsaved())

    def test_defaults_from_reference_mark_modifications(self):
        pf = ParamFile.load(self.path)
        pf.set("max_slam", 7)
        pf.save()
        again = ParamFile.load(
            self.path, reference=CONFIG / "euroc_mav" / "estimator_config.yaml"
        )
        self.assertEqual([p.key for p in again.modified], ["max_slam"])
        self.assertFalse(again.has_unsaved())
        again.reset("max_slam")
        self.assertTrue(again.has_unsaved())

    def test_add_revert_and_save(self):
        pf = ParamFile.load(self.path)
        pf.add("cam0_shutter", "global", "generated", after="max_cameras")
        self.assertEqual(pf.value("cam0_shutter"), "global")
        self.assertEqual(pf.get("cam0_shutter").type, "enum")
        self.assertTrue(pf.has_unsaved())
        pf.revert()
        self.assertNotIn("cam0_shutter", pf)
        pf.add("cam0_shutter", "global", "generated", after="max_cameras")
        pf.save()
        lines = self.path.read_text().splitlines()
        index = next(
            i for i, line in enumerate(lines) if line.startswith("max_cameras:")
        )
        self.assertEqual(lines[index + 1], 'cam0_shutter: "global" # generated')


class Generate(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_templates_are_discovered(self):
        names = vioconfig.list_templates()
        self.assertIn("euroc_mav", names)
        self.assertIn("voxl_sim", names)
        self.assertEqual(vioconfig.template_path("euroc_mav"), CONFIG / "euroc_mav")
        with self.assertRaises(ValueError):
            vioconfig.template_path("no_such_preset")

    def test_generate_from_synthetic_voxl_log(self):
        log = open_log(make_voxl_log(self.root / "log0001"))
        try:
            self.assertEqual(vioconfig.default_template(log), "voxl_sim")
            out = self.root / "run"
            path, notes = vioconfig.generate(
                log, log.cameras, log.imus[0], out, template="euroc_mav"
            )
        finally:
            log.close()
        ps = ParamSet.from_estimator_config(
            path, reference=CONFIG / "euroc_mav" / "estimator_config.yaml"
        )
        self.assertEqual(ps.value("max_cameras"), 2)
        self.assertIs(ps.value("use_stereo"), True)
        self.assertEqual(ps.value("cam0_shutter"), "global")
        self.assertEqual(ps.value("cam1_shutter"), "global")
        self.assertAlmostEqual(ps.value("imu0.update_rate"), 1000.0, places=3)
        self.assertEqual(ps.value("imu0.rostopic"), "imu_apps")
        expected = np.eye(4)
        expected[:3, :3] = rpy_matrix(0, -90, 90)
        expected[:3, 3] = [0.07, 0.02, 0.01]
        np.testing.assert_allclose(
            np.array(ps.value("cam0.T_imu_cam")), expected, atol=1e-12
        )
        self.assertEqual(ps.value("cam1.T_imu_cam")[1][3], -0.02)
        self.assertEqual(ps.value("cam0.intrinsics"), [30.5, 30.75, 32.25, 20.125])
        self.assertEqual(ps.value("cam1.intrinsics")[0], 31.5)
        self.assertEqual(ps.value("cam0.distortion_model"), "equidistant")
        self.assertEqual(
            ps.value("cam0.distortion_coeffs"), [0.05, 0.01, -0.002, 0.0004]
        )
        self.assertEqual(ps.value("cam0.resolution"), [64, 40])
        self.assertEqual(ps.value("cam0.cam_overlaps"), [1])
        self.assertEqual(ps.value("cam0.timeshift_cam_imu"), 0.0)
        self.assertGreaterEqual(ps.value("track_frequency"), 30.0)
        self.assertTrue(str(ps.value("filepath_est")).startswith(str(out)))
        self.assertTrue(any("CHECK IMU noise" in n for n in notes))
        self.assertFalse(ps.has_unsaved())
        # Template files are never modified, and a run directory is never overwritten.
        self.assertEqual(
            ParamFile.load(CONFIG / "euroc_mav" / "estimator_config.yaml").value(
                "max_cameras"
            ),
            2,
        )
        with self.assertRaises(FileExistsError):
            log = open_log(self.root / "log0001")
            try:
                vioconfig.generate(
                    log, log.cameras, log.imus[0], out, template="euroc_mav"
                )
            finally:
                log.close()

    def test_voxl_default_template_and_unsynchronized_pair(self):
        log = open_log(make_voxl_log(self.root / "log0002", synchronized=False))
        try:
            path, notes = vioconfig.generate(
                log,
                log.cameras,
                log.imus[0],
                self.root / "run2",
                overrides={"max_clones": 9},
            )
        finally:
            log.close()
        ps = ParamSet.from_estimator_config(path)
        self.assertIs(ps.value("use_stereo"), False)
        self.assertEqual(ps.value("cam0.cam_overlaps"), [])
        self.assertEqual(ps.value("max_clones"), 9)
        self.assertEqual(ps.value("zupt_max_disparity"), 8.0)
        np.testing.assert_allclose(np.array(ps.value("imu0.Tw")), np.eye(3))
        self.assertTrue(any("not synchronized" in n for n in notes))
        self.assertTrue(notes[0].startswith("template: voxl_sim"))

    def test_missing_intrinsics_name_the_camera(self):
        log = open_log(
            make_voxl_log(self.root / "log0003", intrinsics=("tracking_left",))
        )
        try:
            with self.assertRaises(ValueError) as caught:
                vioconfig.generate(log, log.cameras, log.imus[0], self.root / "run3")
        finally:
            log.close()
        self.assertIn("tracking_right", str(caught.exception))
        self.assertIn("opencv_tracking_right_intrinsics.yml", str(caught.exception))
        self.assertFalse((self.root / "run3").exists())


class EditorSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_search_edit_highlight_and_save(self):
        from zstudio.param_editor import ParamEditor
        from zstudio.widgets import AMBER

        with TemporaryDirectory() as tmp:
            for name in (
                "estimator_config.yaml",
                "kalibr_imu_chain.yaml",
                "kalibr_imucam_chain.yaml",
            ):
                shutil.copy(CONFIG / "euroc_mav" / name, Path(tmp) / name)
            editor = ParamEditor()
            editor.resize(1200, 700)
            editor.show()
            editor.load_estimator_config(Path(tmp) / "estimator_config.yaml")
            model = editor.model
            total = model.rowCount()
            self.assertEqual(total, len(editor.param_set.params))
            self.assertFalse(editor.save_button.isEnabled())

            editor.search.setText("zupt")
            editor._refresh()
            self.assertTrue(model.rows)
            self.assertTrue(
                all(
                    "zupt" in (k + p.description + p.notes).lower()
                    for k, p in model.rows
                )
            )

            editor.search.setText("max_clones")
            editor._refresh()
            row = next(i for i, (k, _) in enumerate(model.rows) if k == "max_clones")
            value = model.index(row, 1)
            self.assertTrue(model.flags(value) & Qt.ItemIsEditable)
            self.assertFalse(model.setData(value, "eleven"))
            self.assertIn("expects a number", editor.status.text())
            self.assertTrue(model.setData(value, 14))
            self.assertEqual(
                model.data(model.index(row, 0), Qt.ForegroundRole).name(), AMBER.lower()
            )
            self.assertTrue(model.data(model.index(row, 0), Qt.FontRole).bold())
            self.assertTrue(editor.has_unsaved())
            self.assertTrue(editor.save_button.isEnabled())

            editor.search.setText("use_fej")
            editor._refresh()
            row = next(i for i, (k, _) in enumerate(model.rows) if k == "use_fej")
            self.assertTrue(
                model.setData(model.index(row, 1), Qt.Unchecked, Qt.CheckStateRole)
            )

            editor.search.clear()
            editor.modified_only.setChecked(True)
            self.assertEqual(
                sorted(k for k, _ in model.rows), ["max_clones", "use_fej"]
            )
            saved = []
            editor.saved.connect(saved.append)
            editor.save()
            self.assertEqual([Path(p).name for p in saved], ["estimator_config.yaml"])
            self.assertFalse(editor.has_unsaved())
            text = (Path(tmp) / "estimator_config.yaml").read_text()
            self.assertIn("max_clones: 14 # how many clones", text)
            self.assertIn("use_fej: false #", text)

            editor.param_set.set("max_slam", 3)
            editor.revert()
            self.assertFalse(editor.has_unsaved())
            editor.reset_all(confirm=False)
            self.assertEqual(editor.param_set.value("max_clones"), 11)
            editor.close()

    def test_editor_shortcuts_win_over_window_shortcuts(self):
        from PyQt5.QtGui import QKeySequence
        from PyQt5.QtTest import QTest
        from PyQt5.QtWidgets import QMainWindow, QShortcut
        from zstudio.param_editor import ParamEditor

        with TemporaryDirectory() as tmp:
            for name in (
                "estimator_config.yaml",
                "kalibr_imu_chain.yaml",
                "kalibr_imucam_chain.yaml",
            ):
                shutil.copy(CONFIG / "euroc_mav" / name, Path(tmp) / name)
            window, hits = QMainWindow(), []
            QShortcut(QKeySequence("Ctrl+S"), window).activated.connect(
                lambda: hits.append("window")
            )
            editor = ParamEditor()
            window.setCentralWidget(editor)
            window.show()
            window.activateWindow()
            QApplication.processEvents()
            try:
                editor.load_estimator_config(Path(tmp) / "estimator_config.yaml")
                editor.param_set.set("max_clones", 12)
                editor.table.setFocus()
                QApplication.processEvents()
                QTest.keyClick(editor.table, Qt.Key_S, Qt.ControlModifier)
                self.assertFalse(editor.has_unsaved())
                self.assertEqual(hits, [])
                QTest.keyClick(editor.table, Qt.Key_F, Qt.ControlModifier)
                QApplication.processEvents()
                self.assertIs(QApplication.focusWidget(), editor.search)
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
