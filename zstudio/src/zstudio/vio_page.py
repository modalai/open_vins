"""The studio's VIO page: run OpenVINS on a log and explore the estimate.

Setup (cameras / IMU / template / tracker) -> generated OpenVINS configs edited
in the parameter editor -> ``ov_msckf_replay`` fed by the ROS-free sensor stream.
Results are kept for the whole run and tied to the log timeline: the 3D view,
camera tracks and plots all follow the shared playhead, so the filter's history
can be scrubbed (live runs follow the newest estimate until you seek).
"""

import bisect
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QMenu,
    QStackedWidget,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .geometry import rotation_angle_deg
from .logviews import RED, SERIES_COLORS, LogPlot, TimeRange
from .model import POINT_DTYPE, Frame
from .vioview import VioView, camera_poses
from .widgets import (
    CameraGrid,
    Segmented,
    TransportBar,
    AMBER,
    BLUE,
    MUTED,
    TEAL,
    TEXT,
    Card,
    FeatureView,
    button,
    jpl_rotation,
    label,
    pill,
    row,
    set_pill_color,
)

# A hand-held / drone log moving faster than this is a diverged estimate, not motion.
DIVERGED_SPEED = 30.0

SPEEDS = (
    ("As fast as possible", 0.0),
    ("Real time (1×)", 1.0),
    ("2×", 2.0),
    ("4×", 4.0),
    ("0.5×", 0.5),
)


def _get(record, *names, default=None):
    for name in names:
        if hasattr(record, name):
            return getattr(record, name)
    return default


# ov_msckf VioManagerOptions defaults for grid_x / grid_y.
OPENVINS_GRID = (5, 5)


class VioResults:
    """Every estimate of one run, columnar for plotting and scrubbing."""

    def __init__(self):
        self.t, self.p, self.v, self.bg, self.ba = [], [], [], [], []
        self.R, self.sig_p, self.sig_q, self.proc, self.flags = [], [], [], [], []
        self.n_slam, self.n_msckf, self.calib, self.cams = [], [], [], []
        self.points = []  # (t, xyz, kind)
        self.point_times = []
        self.track_times = {}  # cam -> [t]
        self.tracks = {}  # cam -> [(t, ids, uv)]
        self.messages = []
        self.map = np.empty((0, 3))
        self.initialized_at = None

    def add_state(self, s):
        q = np.asarray(_get(s, "q_GtoI", "q"), float)
        p = np.asarray(_get(s, "p_IinG", "p"), float)
        R_ItoG = jpl_rotation(q).T
        calib = [
            (np.asarray(c.q_ItoC, float), np.asarray(c.p_IinC, float), float(c.td))
            if hasattr(c, "q_ItoC")
            else tuple(c)
            for c in (_get(s, "calib", "cams", default=[]) or [])
        ]
        self.t.append(float(s.t))
        self.p.append(p)
        self.v.append(np.asarray(_get(s, "v_IinG", "v"), float))
        self.bg.append(np.asarray(_get(s, "bg", "bias_g"), float))
        self.ba.append(np.asarray(_get(s, "ba", "bias_a"), float))
        self.sig_p.append(np.asarray(_get(s, "sigma_pos", default=[np.nan] * 3), float))
        self.sig_q.append(np.asarray(_get(s, "sigma_ori", default=[np.nan] * 3), float))
        self.proc.append(float(_get(s, "proc_ms", default=np.nan)))
        flags = int(_get(s, "flags", default=0) or 0)
        if _get(s, "initialized", default=None) is not None:
            flags = (flags & ~1) | int(bool(s.initialized))
        self.flags.append(flags)
        self.n_slam.append(int(_get(s, "n_slam", default=0) or 0))
        self.n_msckf.append(int(_get(s, "n_msckf", default=0) or 0))
        self.R.append(R_ItoG)
        self.calib.append(calib)
        self.cams.append(camera_poses(R_ItoG, p, calib))
        if self.initialized_at is None and flags & 1:
            self.initialized_at = float(s.t)

    def add_points(self, rec):
        xyz = np.asarray(_get(rec, "xyz", "points"), float).reshape(-1, 3)
        kind = np.asarray(_get(rec, "kind", "kinds"), np.uint8).reshape(-1)
        self.points.append((float(rec.t), xyz, kind))
        self.point_times.append(float(rec.t))
        slam = xyz[kind == 1]
        if len(slam):
            self.map = np.vstack((self.map, slam))[-30000:]

    def add_tracks(self, rec):
        ids = np.asarray(_get(rec, "ids"), np.uint32).reshape(-1)
        uv = np.asarray(_get(rec, "uv"), np.float32).reshape(-1, 2)
        self.tracks.setdefault(int(rec.cam), []).append((float(rec.t), ids, uv))
        self.track_times.setdefault(int(rec.cam), []).append(float(rec.t))

    def arrays(self):
        return np.asarray(self.t), np.asarray(self.p).reshape(-1, 3)

    def index_at(self, t):
        if not self.t:
            return -1
        return max(0, bisect.bisect_right(self.t, t) - 1)

    def points_at(self, t):
        if not self.points:
            return None
        i = max(0, bisect.bisect_right(self.point_times, t) - 1)
        return self.points[i][1], self.points[i][2]

    def tracks_at(self, cam, t, tolerance=0.03):
        times = self.track_times.get(cam)
        if not times:
            return None
        j = bisect.bisect_left(times, t)
        candidates = [k for k in (j - 1, j) if 0 <= k < len(times)]
        i = min(candidates, key=lambda k: abs(times[k] - t))
        if abs(times[i] - t) > tolerance:
            return None
        _, ids, uv = self.tracks[cam][i]
        points = np.zeros(len(ids), dtype=POINT_DTYPE)
        points["id"], points["u"], points["v"] = ids, uv[:, 0], uv[:, 1]
        return points


class VioPage(QWidget):
    seek = pyqtSignal(float)
    notify = pyqtSignal(str, bool)

    def __init__(self, provider_getter, parent=None):
        super().__init__(parent)
        self.provider_getter = provider_getter
        self.log = None
        self.run = None
        self.results = VioResults()
        self.run_grid = OPENVINS_GRID
        self.config_path = None
        self.run_dir = None
        self.live = True
        self.range = TimeRange(self)
        self.range.seek.connect(self._user_seek)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        head = Card(margins=14)
        self.status = pill("NO LOG", MUTED)
        self.title = label("Open a log to run OpenVINS on it", 13, bold=True)
        self.subtitle = label(
            "ROS-free replay through ov_msckf_replay. CPU TrackKLT by default; no libmodal-flow required.",
            9,
            MUTED,
        )
        self.subtitle.setWordWrap(True)
        self.run_button = button("Run VIO", self.start, "play", True)
        self.stop_button = button("Stop", self.stop, "stop")
        self.stop_button.setEnabled(False)
        self.save_button = button("Save trajectory…", self.save_trajectory, "download")
        self.folder_button = button("Run folder", self.open_folder, "folder")
        titles = QVBoxLayout()
        titles.setSpacing(3)
        titles.addLayout(row(self.status, self.title, None))
        titles.addWidget(self.subtitle)
        more = button("More", None)
        menu = QMenu(more)
        menu.addAction("Save trajectory (TUM)…", self.save_trajectory)
        menu.addAction("Open run folder", self.open_folder)
        menu.addAction("Open an existing config…", self.open_config)
        more.setMenu(menu)
        self.save_button.hide()
        self.folder_button.hide()
        head.box.addLayout(row(titles, None, more, self.stop_button, self.run_button))
        layout.addWidget(head)
        self.transport = TransportBar()
        self.playback_rate = (
            0.0  # playback speed factor while the window plays (0 when paused)
        )
        transport = Card(margins=10)
        transport.box.addLayout(
            row(
                self.transport,
                label(
                    "playing replays the estimate; seeking leaves live follow", 8, MUTED
                ),
            )
        )
        layout.addWidget(transport)
        body = QSplitter(Qt.Horizontal)
        body.setHandleWidth(12)
        body.addWidget(self._setup_panel())
        center = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        self.view = VioView()
        view_page = QWidget()
        vbox = QVBoxLayout(view_page)
        vbox.setContentsMargins(6, 6, 6, 6)
        self.follow = QCheckBox("Follow")
        self.follow.setChecked(True)
        self.follow.toggled.connect(
            lambda on: (setattr(self.view, "follow", on), self.view.update())
        )
        self.map_check = QCheckBox("Map")
        self.map_check.setChecked(True)
        self.map_check.toggled.connect(
            lambda on: (setattr(self.view, "show_map", on), self.view.update())
        )
        self.ghost = QCheckBox("Future path")
        self.ghost.setChecked(True)
        self.ghost.toggled.connect(
            lambda on: (setattr(self.view, "show_ghost", on), self.view.update())
        )
        self.live_check = QCheckBox("Follow live estimate")
        self.live_check.setChecked(True)
        self.live_check.toggled.connect(self._set_live)
        vbox.addLayout(
            row(
                self.follow,
                self.map_check,
                self.ghost,
                self.live_check,
                None,
                button("Top", self.view.top_view),
                button("Fit", self.view.fit),
                button("Reset", self.view.reset_view),
            )
        )
        vbox.addWidget(self.view, 1)
        self.tabs.addTab(view_page, "Trajectory")
        self.editor_host = QWidget()
        self.editor_layout = QVBoxLayout(self.editor_host)
        self.editor_layout.setContentsMargins(0, 0, 0, 0)
        self.editor = None
        self.editor_placeholder = label(
            "Generate or open an OpenVINS configuration to edit its parameters.",
            10,
            MUTED,
        )
        self.editor_placeholder.setAlignment(Qt.AlignCenter)
        self.editor_layout.addWidget(self.editor_placeholder)
        self.tabs.addTab(self.editor_host, "Parameters")
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(20000)
        self.tabs.addTab(self.console, "Estimator log")
        center.addWidget(self.tabs)
        plots = Card(margins=8)
        self.plot_box = QComboBox()
        for name in self._plot_names():
            self.plot_box.addItem(name)
        self.plot_box.currentIndexChanged.connect(lambda *_: self._refresh_plot())
        plots.box.addLayout(
            row(
                label("Plot", 9, MUTED),
                self.plot_box,
                None,
                label("wheel: zoom · drag: pan · click: seek", 8, MUTED),
                button("", self.range.reset, "reset"),
            )
        )
        self.plot = LogPlot(self.range)
        self.plot.setMinimumHeight(170)
        plots.box.addWidget(self.plot, 1)
        center.addWidget(plots)
        center.setSizes([560, 230])
        body.addWidget(center)
        side = QWidget()
        sbox = QVBoxLayout(side)
        sbox.setContentsMargins(0, 0, 0, 0)
        sbox.setSpacing(12)
        cam_card = Card(margins=8)
        self.cam_box = QComboBox()
        self.cam_box.currentIndexChanged.connect(lambda *_: self.refresh())
        self.track_switch = Segmented(
            (("Selected", "selected"), ("All cameras", "all"))
        )
        self.track_switch.changed.connect(self._track_mode_changed)
        cam_card.box.addLayout(row(label("Tracks", 9, MUTED), None, self.track_switch))
        cam_card.box.addWidget(self.cam_box)
        self.track_stack = QStackedWidget()
        self.camera_view = FeatureView()
        self.camera_view.show_grid = False
        self.camera_view.captions = ("Features the filter tracked in this frame", "vio")
        self.camera_view.setMinimumSize(300, 200)
        self.track_stack.addWidget(self.camera_view)
        self.track_grid = CameraGrid()
        self.track_stack.addWidget(self.track_grid)
        cam_card.box.addWidget(self.track_stack, 1)
        sbox.addWidget(cam_card, 1)
        stats = Card(margins=14)
        self.stats = label("—", 9, TEXT)
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        stats.box.addWidget(label("Run", 9, MUTED))
        stats.box.addWidget(self.stats)
        sbox.addWidget(stats)
        body.addWidget(side)
        body.setSizes([310, 820, 360])
        layout.addWidget(body, 1)
        self.set_log(None)

    # -- setup panel -------------------------------------------------------------
    def _setup_panel(self):
        card = Card(margins=14)
        card.box.addWidget(label("Setup", 12, bold=True))
        form = QFormLayout()
        form.setSpacing(10)
        self.cameras = QListWidget()
        self.cameras.setMinimumHeight(110)
        self.cameras.itemChanged.connect(lambda *_: self._setup_changed())
        form.addRow("Cameras", self.cameras)
        self.imu_box = QComboBox()
        form.addRow("IMU", self.imu_box)
        self.template = QComboBox()
        form.addRow("Template", self.template)
        self.template_cameras = QCheckBox("Template calibration")
        self.template_cameras.setToolTip(
            "For logs recorded by the device the preset describes (e.g. a EuRoC bag with euroc_mav) or logs without "
            "CameraInfo: keep the template's intrinsics, T_imu_cam and IMU intrinsics; only topics are rewritten."
        )
        self.template_cameras.toggled.connect(lambda *_: self._refresh_camera_items())
        form.addRow("", self.template_cameras)
        self.tracker = QComboBox()
        self.tracker.setToolTip(
            "CPU TrackKLT is the default. OpenCL TrackOCL appears when an ov_msckf_replay built with the libmodal-flow host harness is found."
        )
        self.tracker.currentIndexChanged.connect(lambda *_: self._tracker_changed())
        form.addRow("Tracker", self.tracker)
        self.start_time = QDoubleSpinBox()
        self.end_time = QDoubleSpinBox()
        for spin in (self.start_time, self.end_time):
            spin.setDecimals(2)
            spin.setSuffix(" s")
            spin.setRange(0, 1e7)
        form.addRow("Start", self.start_time)
        form.addRow("End", self.end_time)
        self.speed = QComboBox()
        for text, value in SPEEDS:
            self.speed.addItem(text, value)
        form.addRow("Playback", self.speed)
        card.box.addLayout(form)
        self.generate_button = button("Generate config", self.generate, "tune", True)
        self.open_config_button = button("Open config…", self.open_config, "folder")
        card.box.addWidget(self.generate_button)
        self.open_config_button.hide()  # also in More
        self.config_label = label("No configuration yet", 8, MUTED)
        self.config_label.setWordWrap(True)
        self.config_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        card.box.addWidget(self.config_label)
        self.notes = label("", 8, AMBER)
        self.notes.setWordWrap(True)
        card.box.addWidget(self.notes)
        card.box.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(card)
        scroll.setMinimumWidth(320)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    def _binaries(self):
        try:
            from .stream import find_replay_binaries

            return find_replay_binaries()
        except Exception:
            return []

    def _populate_trackers(self):
        self.tracker.blockSignals(True)
        self.tracker.clear()
        binaries = self._binaries()
        klt = next(
            (
                p
                for p, caps in binaries
                if "klt" in (caps or {}).get("trackers", ["klt"])
            ),
            None,
        )
        ocl = next(
            (p for p, caps in binaries if "ocl" in (caps or {}).get("trackers", [])),
            None,
        )
        self.tracker.addItem("CPU · TrackKLT (default)", ("klt", klt))
        if ocl:
            version = next(
                (caps.get("modal_flow_version") for p, caps in binaries if p == ocl),
                None,
            )
            self.tracker.addItem(
                f"OpenCL · TrackOCL (modal-flow {version or 'host harness'})",
                ("ocl", ocl),
            )
        if not binaries:
            self.tracker.setItemText(0, "CPU · TrackKLT (ov_msckf_replay not found)")
        self.tracker.blockSignals(False)

    def _templates(self):
        try:
            from .vioconfig import list_templates

            return list_templates()
        except Exception:
            return []

    # -- log -----------------------------------------------------------------------
    def set_log(self, log):
        if self.run is not None and log is not self.log:
            self.stop()
        self.log = log
        self.results = VioResults()
        self.run_grid = OPENVINS_GRID
        self.config_path = None
        self.run_dir = None
        self.cameras.blockSignals(True)
        self.cameras.clear()
        self.imu_box.clear()
        self.cam_box.clear()
        self.view.set_run([], [], [], [])
        self.view.status = ""
        self.plot.clear("Run VIO to plot the estimate")
        self.camera_view.set_frame(None, None)
        self.config_label.setText("No configuration yet")
        self.notes.setText("")
        enabled = log is not None and bool(log.imus) and bool(log.cameras)
        for widget in (self.run_button, self.generate_button, self.open_config_button):
            widget.setEnabled(enabled)
        if log is None:
            set_pill_color(self.status, MUTED)
            self.status.setText("NO LOG")
            self.title.setText("Open a log to run OpenVINS on it")
            self.cameras.blockSignals(False)
            self._set_editor(None)
            return
        self.title.setText(f"{log.name} · {log.format_label}")
        self.status.setText("READY" if enabled else "NEEDS IMU + CAMERA")
        set_pill_color(self.status, TEAL if enabled else AMBER)
        with_intrinsics = sum(1 for cam in log.cameras if cam.intrinsics is not None)
        self.template_cameras.setChecked(with_intrinsics == 0 and bool(log.cameras))
        checked = 0
        for cam in log.cameras:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, cam.name)
            check = (
                cam.has_pictures
                and checked < 2
                and (
                    cam.global_shutter
                    or log.kind != "voxl"
                    or cam.name.startswith("tracking")
                )
            )
            check = check and (cam.intrinsics is not None or with_intrinsics == 0)
            item.setCheckState(Qt.Checked if check else Qt.Unchecked)
            checked += int(check)
            self.cameras.addItem(item)
            self.cam_box.addItem(cam.name)
        self.cameras.blockSignals(False)
        self.track_grid.set_cameras(
            [c.camera or c.name for c in log.cameras], ("", "vio")
        )
        self._refresh_camera_items()
        for imu in log.imus:
            self.imu_box.addItem(
                f"{imu.name} · {1.0 / np.median(np.diff(imu.times)):.0f} Hz"
                if len(imu) > 1
                else imu.name,
                imu.name,
            )
        self.template.clear()
        for name in self._templates():
            self.template.addItem(str(name))
        try:
            from .vioconfig import default_template

            self.template.setCurrentText(default_template(log))
        except Exception:
            pass
        self._rig_warnings()
        self._populate_trackers()
        self.start_time.setRange(log.start, log.end)
        self.end_time.setRange(log.start, log.end)
        self.start_time.setValue(log.start)
        self.end_time.setValue(log.end)
        self.range.set_bounds(log.start, max(log.end, log.start + 1e-3))
        self._set_editor(None)
        self._refresh_stats()

    def set_inspection(self, inspection):
        self.inspection = inspection
        self._rig_warnings()

    def _rig_warnings(self):
        """Generated configs trust the log's rig; surface inspection findings that say it is wrong."""
        inspection = getattr(self, "inspection", None)
        if self.log is None or inspection is None or inspection.log is not self.log:
            return
        codes = (
            "gravity.orientation",
            "stereo.geometry",
            "intrinsics.size",
            "intrinsics.invalid",
            "rig.camera",
            "stereo.sync",
        )
        found = [
            f
            for f in inspection.findings
            if f.code in codes and f.severity in ("problem", "warning")
        ]
        if found:
            self.notes.setText(
                "Check before running — the generated T_imu_cam comes from the log's rig:\n"
                + "\n".join(f"• {f.channel or 'log'}: {f.message}" for f in found)
            )

    def _refresh_camera_items(self):
        """Cameras need log intrinsics unless the template's calibration is used."""
        if self.log is None:
            return
        keep = self.template_cameras.isChecked()
        self.cameras.blockSignals(True)
        for i in range(self.cameras.count()):
            item = self.cameras.item(i)
            cam = next(c for c in self.log.cameras if c.name == item.data(Qt.UserRole))
            usable = cam.has_pictures and (keep or cam.intrinsics is not None)
            item.setText(
                f"{cam.label}  ·  {cam.width}×{cam.height}"
                + ("" if cam.intrinsics is not None or keep else "  (no intrinsics)")
            )
            item.setFlags(
                Qt.ItemIsUserCheckable
                | (Qt.ItemIsEnabled if usable else Qt.NoItemFlags)
            )
            if not usable:
                item.setCheckState(Qt.Unchecked)
            item.setToolTip(
                ""
                if usable
                else cam.intrinsics_problem
                or f"missing intrinsics: {cam.intrinsics_expected or 'no CameraInfo'}; use the template's calibration or Open config…"
            )
        self.cameras.blockSignals(False)

    def chosen_cameras(self):
        names = [
            self.cameras.item(i).data(Qt.UserRole)
            for i in range(self.cameras.count())
            if self.cameras.item(i).checkState() == Qt.Checked
        ]
        return [c for c in self.log.cameras if c.name in names] if self.log else []

    def chosen_imu(self):
        if not self.log:
            return None
        return next(
            (i for i in self.log.imus if i.name == self.imu_box.currentData()),
            self.log.imus[0] if self.log.imus else None,
        )

    def _setup_changed(self):
        if self.config_path is not None:
            self.notes.setText(
                "Camera selection changed: generate the configuration again before running."
            )

    # -- configuration -------------------------------------------------------------
    def _set_editor(self, path, reference=None):
        if self.editor is not None:
            self.editor.setParent(None)
            self.editor.deleteLater()
            self.editor = None
        self.editor_placeholder.setVisible(path is None)
        if path is None:
            return
        try:
            from .param_editor import ParamEditor

            self.editor = ParamEditor()
            # With the template as reference, every generated value shows as modified.
            self.editor.load_estimator_config(
                str(path), reference=str(reference) if reference else None
            )
            self.editor_layout.addWidget(self.editor)
        except Exception as error:
            self.editor = None
            self.editor_placeholder.setVisible(True)
            self.notify.emit(f"Parameter editor unavailable: {error}", True)

    def _new_run_dir(self):
        base = Path.home() / "ov_zcalib_sessions" / "vio"
        safe = self.log.name.replace("/", "_").strip("_") or "log"
        path = base / f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def generate(self):
        if self.log is None:
            return
        cameras, imu = self.chosen_cameras(), self.chosen_imu()
        if not cameras or imu is None:
            self.notify.emit(
                "Select at least one camera with intrinsics and an IMU.", True
            )
            return
        try:
            from .vioconfig import generate

            self.run_dir = self._new_run_dir()
            path, notes = generate(
                self.log,
                cameras,
                imu,
                self.run_dir / "config",
                template=self.template.currentText() or None,
                template_cameras=self.template_cameras.isChecked(),
            )
        except Exception as error:
            self.notify.emit(
                f"Could not generate the OpenVINS configuration: {error}", True
            )
            return
        self.config_path = Path(path)
        self.config_label.setText(str(self.config_path))
        self.notes.setText("\n".join(f"• {n}" for n in notes))
        self._set_editor(self.config_path, self._template_reference())
        self._tracker_changed()
        self.tabs.setCurrentIndex(1)
        self.notify.emit(
            f"Generated OpenVINS configuration for {len(cameras)} camera{'s' if len(cameras) > 1 else ''} in {self.run_dir}",
            False,
        )

    def _template_reference(self):
        try:
            from .vioconfig import ESTIMATOR, template_path

            return template_path(self.template.currentText()) / ESTIMATOR
        except Exception:
            return None

    def open_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "OpenVINS estimator config",
            str(Path.home()),
            "OpenVINS config (estimator_config.yaml *.yaml)",
        )
        if path:
            self.config_path = Path(path)
            self.run_dir = self.config_path.parent
            self.config_label.setText(str(self.config_path))
            self.notes.setText(
                "Using an existing configuration: its cameras must match the selected channels in order."
            )
            self._set_editor(self.config_path)

    def _tracker_changed(self):
        """Write use_gpu explicitly: some templates (voxl_sim) default it to true."""
        kind = (self.tracker.currentData() or ("klt", None))[0]
        if self.editor is None or self.editor.param_set is None:
            return
        pset = self.editor.param_set
        wanted = kind == "ocl"
        try:
            if pset.get("use_gpu") is not None:
                if pset.value("use_gpu") != wanted:
                    pset.set("use_gpu", wanted)
            else:
                estimator = pset.file_for(self.config_path)
                after = "use_klt" if estimator.get("use_klt") is not None else None
                estimator.add(
                    "use_gpu",
                    wanted,
                    "TrackOCL (libmodal-flow) when built with it; CPU TrackKLT otherwise",
                    after=after,
                )
                pset.reindex()
            self.editor.set_param_set(pset)
        except Exception as error:
            self.notify.emit(f"Could not set use_gpu: {error}", True)

    # -- run -------------------------------------------------------------------------
    def start(self):
        if self.log is None or self.run is not None:
            return
        if self.config_path is None:
            self.generate()
            if self.config_path is None:
                return
        if self.editor is not None and self.editor.has_unsaved():
            self.editor.save()
        cameras, imu = self.chosen_cameras(), self.chosen_imu()
        kind, binary = self.tracker.currentData() or ("klt", None)
        try:
            from .stream import VioRun

            self.results = VioResults()
            self.console.clear()
            self.run = VioRun()
            self.run.start(
                str(self.config_path),
                self.log,
                cameras,
                imu,
                binary=binary,
                tum_path=str(
                    (self.run_dir or self.config_path.parent) / "trajectory_tum.txt"
                ),
                realtime=float(self.speed.currentData() or 0.0),
                start=self.start_time.value(),
                end=self.end_time.value(),
                on_line=self._line,
            )
        except Exception as error:
            self.run = None
            self.notify.emit(f"VIO could not start: {error}", True)
            return
        self.run_cameras = [c.name for c in cameras]
        self.run_grid = self._config_grid()
        if cameras and self.cam_box.findText(cameras[0].name) >= 0:
            self.cam_box.setCurrentText(cameras[0].name)
        self.diverged = False
        self.live = True
        self.live_check.setChecked(True)
        self.status.setText("INITIALIZING")
        set_pill_color(self.status, AMBER)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.view.status = "Waiting for the filter to initialize…"
        self.tabs.setCurrentIndex(0)
        self.timer.start(50)

    def stop(self):
        if self.run is not None:
            self.run.stop()
            self._poll()

    def _line(self, text):
        self._pending_lines = getattr(self, "_pending_lines", [])
        self._pending_lines.append(text)
        if "OpenCL device:" in text:
            self.device = text.split("OpenCL device:", 1)[1].strip()

    def _track_mode_changed(self, mode):
        self.track_stack.setCurrentIndex(1 if mode == "all" else 0)
        self.cam_box.setVisible(mode != "all")
        self.refresh()

    def _poll(self):
        run = self.run
        if run is None:
            self.timer.stop()
            return
        for text in getattr(self, "_pending_lines", [])[:]:
            self.console.appendPlainText(text.rstrip("\n"))
        self._pending_lines = []
        changed = False
        for record in run.poll():
            kind = type(record).__name__
            if kind == "VioState":
                self.results.add_state(record)
                changed = True
            elif kind == "VioPoints":
                self.results.add_points(record)
            elif kind == "VioTracks":
                self.results.add_tracks(record)
            elif kind == "VioMessage":
                self.results.messages.append(record)
                self.console.appendPlainText(f"[vio] {record.text}")
                if record.text.lower().startswith("tracker:"):
                    device = getattr(self, "device", "")
                    self.subtitle.setText(
                        f"ov_msckf_replay · {record.text}"
                        + (f" · {device}" if device else "")
                    )
        if changed:
            t, p = self.results.arrays()
            self.view.set_run(t, p, self.results.R, self.results.cams)
            if self.live:
                self.seek.emit(float(t[-1]))
            speed = float(np.linalg.norm(self.results.v[-1])) if self.results.v else 0.0
            if speed > DIVERGED_SPEED and not getattr(self, "diverged", False):
                self.diverged = True
                self.notes.setText(
                    f"The estimate diverged (speed {speed:.0f} m/s at t={t[-1]:.2f} s). Check T_imu_cam and intrinsics "
                    "(run Inspect's stereo check / the gravity finding), the IMU noise densities, and the time offset."
                )
            if getattr(self, "diverged", False):
                self.status.setText("DIVERGED")
                set_pill_color(self.status, RED)
            elif self.results.initialized_at is not None:
                self.status.setText("TRACKING")
                set_pill_color(self.status, TEAL)
        if not run.running:
            self.timer.stop()
            code = run.returncode
            ok = code == 0
            if getattr(self, "diverged", False):
                self.status.setText("DIVERGED")
                set_pill_color(self.status, RED)
            elif ok and self.results.initialized_at is None:
                self.status.setText("NEVER INITIALIZED")
                set_pill_color(self.status, AMBER)
            else:
                self.status.setText(
                    "FINISHED" if ok else "STOPPED" if code in (-15, -9) else "FAILED"
                )
                set_pill_color(
                    self.status, TEAL if ok else AMBER if code in (-15, -9) else RED
                )
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.last_run = run
            summary = getattr(run, "summary", None)
            if summary:
                self.console.appendPlainText(f"[vio] summary: {summary}")
            missing = getattr(run, "missing_keys", None)
            error = getattr(run, "error", None)
            if missing:
                self.notes.setText(
                    "The configuration lacks parameters this OpenVINS build requires: "
                    + ", ".join(missing)
                    + ". Generate a new configuration or add them in Parameters."
                )
            elif error and not ok:
                self.notes.setText(f"Estimator stopped: {error}")
            self.run = None
            if not self.results.t:
                self.view.status = (
                    "The filter produced no estimate; see the estimator log."
                )
                self.tabs.setCurrentIndex(2)
            self.notify.emit(
                f"VIO {'finished' if ok else 'ended'}: {len(self.results.t)} estimates",
                not ok,
            )
        self._refresh_plot()
        self._refresh_stats()
        self.refresh()

    def _set_live(self, on):
        self.live = on
        if on and self.results.t:
            self.seek.emit(float(self.results.t[-1]))

    def _user_seek(self, t):
        self.live_check.setChecked(False)
        self.seek.emit(t)

    # -- playhead ------------------------------------------------------------------------
    def set_playhead(self, t):
        self.range.set_playhead(t)
        self.refresh()

    def refresh(self):
        t = self.range.playhead
        index = self.results.index_at(t)
        self.view.set_index(index)
        points = self.results.points_at(t)
        if points is not None:
            self.view.set_points(
                points, self.results.map if self.view.show_map else None
            )
        self._refresh_camera(t)

    def _refresh_camera(self, t):
        log = self.log
        if log is None or not log.cameras:
            return
        if self.track_switch.value() == "all":
            for index, view in enumerate(self.track_grid.views[: len(log.cameras)]):
                self._show_tracks(view, index, t)
        elif self.cam_box.currentIndex() >= 0:
            self._show_tracks(self.camera_view, self.cam_box.currentIndex(), t)

    def _config_grid(self):
        """(grid_x, grid_y) of the estimator config this run uses (OpenVINS default 5 x 5)."""
        values = {}
        pset = self.editor.param_set if self.editor is not None else None
        for key in ("grid_x", "grid_y"):
            value = pset.value(key) if pset is not None else None
            if value is None and self.config_path:
                try:
                    text = Path(self.config_path).read_text(encoding="utf-8")
                    match = re.search(rf"^\s*{key}\s*:\s*(\d+)", text, re.M)
                    value = match.group(1) if match else None
                except OSError:
                    value = None
            try:
                values[key] = max(1, int(value))
            except (TypeError, ValueError):
                values[key] = None
        default_x, default_y = OPENVINS_GRID
        return (values["grid_x"] or default_x, values["grid_y"] or default_y)

    def has_tracks(self):
        """True once the latest run has delivered feature tracks for some camera."""
        return bool(getattr(self, "run_cameras", None)) and bool(self.results.track_times)

    def tracked_points(self, cam_index, row):
        """Features the latest VIO run tracked in log camera ``cam_index`` at frame ``row``.

        None when that camera was not part of the run or has no tracks at that frame.
        """
        if self.log is None or not 0 <= cam_index < len(self.log.cameras):
            return None
        cam = self.log.cameras[cam_index]
        run_cameras = getattr(self, "run_cameras", [])
        if cam.name not in run_cameras or not 0 <= row < len(cam):
            return None
        period = float(np.median(np.diff(cam.times))) if len(cam) > 1 else 0.05
        return self.results.tracks_at(
            run_cameras.index(cam.name),
            float(cam.times[row]),
            tolerance=max(0.6 * period, 0.012),
        )

    def tracked_cameras(self):
        """Indices of log cameras that the latest run tracked."""
        if self.log is None:
            return []
        names = set(getattr(self, "run_cameras", []))
        return [i for i, cam in enumerate(self.log.cameras) if cam.name in names]

    def _show_tracks(self, view, cam_index, t):
        cam = self.log.cameras[cam_index]
        if not len(cam):
            return
        row = max(0, int(np.searchsorted(cam.times, t, side="right")) - 1)
        provider = self.provider_getter()
        fps = 1.0 / float(np.median(np.diff(cam.times))) if len(cam) > 1 else 0.0
        image = None
        if provider is not None:
            provider.set_width(cam_index, view.width() * view.devicePixelRatioF())
            image = provider.image(cam_index, row, rate=self.playback_rate * fps)
        points = self.tracked_points(cam_index, row)
        view.coverage_grid = self.run_grid
        from .model import Camera

        intr = cam.intrinsics or {}
        camera = Camera(
            width=cam.width or 640,
            height=cam.height or 480,
            k=intr.get("k", [400, 400, 320, 240]),
        )
        last = getattr(self, "_last_images", {})
        frame = Frame(
            float(cam.times[row]),
            points if points is not None else np.zeros(0, dtype=POINT_DTYPE),
            cam=cam_index,
            image=image if image is not None else last.get(cam_index),
        )
        if image is not None:
            last[cam_index] = image
            self._last_images = last
        view.set_frame(frame, camera)

    # -- plots / stats --------------------------------------------------------------------
    def _plot_names(self):
        return [
            "Position",
            "Velocity",
            "Speed",
            "Gyroscope bias",
            "Accelerometer bias",
            "Position σ",
            "Orientation σ",
            "Extrinsic rotation change",
            "Extrinsic translation change",
            "Camera-IMU time offset",
            "Features in update",
            "Update time",
        ]

    def _refresh_plot(self):
        r = self.results
        if not r.t:
            self.plot.clear("Run VIO to plot the estimate")
            return
        t = np.asarray(r.t)
        name = self.plot_box.currentText()
        xyz = ("x", "y", "z")

        def three(values, units, title=None):
            data = np.asarray(values).reshape(-1, 3)
            return (
                title or name,
                units,
                [
                    (a, c, t, data[:, k])
                    for k, (a, c) in enumerate(zip(xyz, SERIES_COLORS))
                ],
            )

        if name == "Position":
            spec = three(r.p, "m")
        elif name == "Velocity":
            spec = three(r.v, "m/s")
        elif name == "Speed":
            spec = (
                name,
                "m/s",
                [
                    (
                        "|v|",
                        TEAL,
                        t,
                        np.linalg.norm(np.asarray(r.v).reshape(-1, 3), axis=1),
                    )
                ],
            )
        elif name == "Gyroscope bias":
            spec = three(r.bg, "rad/s")
        elif name == "Accelerometer bias":
            spec = three(r.ba, "m/s²")
        elif name == "Position σ":
            spec = three(r.sig_p, "m")
        elif name == "Orientation σ":
            spec = three(np.degrees(np.asarray(r.sig_q)), "deg")
        elif name in (
            "Extrinsic rotation change",
            "Extrinsic translation change",
            "Camera-IMU time offset",
        ):
            series = []
            names = getattr(self, "run_cameras", [])
            ncam = min((len(c) for c in r.calib), default=0)
            for k in range(ncam):
                first = r.calib[0][k]
                if name == "Extrinsic rotation change":
                    R0 = jpl_rotation(first[0])
                    values = [
                        rotation_angle_deg(jpl_rotation(c[k][0]) @ R0.T)
                        for c in r.calib
                    ]
                    units = "deg"
                elif name == "Extrinsic translation change":
                    p0 = np.asarray(first[1], float)
                    values = [
                        1e3 * np.linalg.norm(np.asarray(c[k][1], float) - p0)
                        for c in r.calib
                    ]
                    units = "mm"
                else:
                    values = [
                        1e3 * float(c[k][2]) if len(c[k]) > 2 else np.nan
                        for c in r.calib
                    ]
                    units = "ms"
                series.append(
                    (
                        names[k] if k < len(names) else f"cam{k}",
                        SERIES_COLORS[k % len(SERIES_COLORS)],
                        t,
                        np.asarray(values, float),
                    )
                )
            spec = (name, units if ncam else "", series)
        elif name == "Features in update":
            spec = (
                name,
                "",
                [
                    ("SLAM", AMBER, t, np.asarray(r.n_slam, float)),
                    ("MSCKF", BLUE, t, np.asarray(r.n_msckf, float)),
                ],
            )
        else:
            spec = (name, "ms", [("update", TEAL, t, np.asarray(r.proc, float))])
        markers = []
        if r.initialized_at is not None:
            markers.append((r.initialized_at, "expected"))
        self.plot.set_series(spec[0], spec[1], spec[2], markers)

    def _refresh_stats(self):
        r = self.results
        if not r.t:
            self.stats.setText("No estimate yet." if self.log else "—")
            return
        p = np.asarray(r.p).reshape(-1, 3)
        distance = (
            float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
            if len(p) > 1
            else 0.0
        )
        proc = np.asarray(r.proc, float)
        proc = proc[np.isfinite(proc)]
        speed = np.linalg.norm(np.asarray(r.v).reshape(-1, 3)[-1])
        lines = [
            f"estimates {len(r.t):,} over {r.t[-1] - r.t[0]:.1f} s",
            f"initialized at {r.initialized_at:.2f} s"
            if r.initialized_at is not None
            else "not initialized yet",
            f"path length {distance:.2f} m · speed {speed:.2f} m/s",
            f"end position [{p[-1, 0]:+.2f} {p[-1, 1]:+.2f} {p[-1, 2]:+.2f}] m",
            f"update {np.mean(proc):.1f} ms mean · {np.percentile(proc, 95):.1f} ms p95"
            if len(proc)
            else "",
            f"features SLAM {r.n_slam[-1]} · MSCKF {r.n_msckf[-1]}",
        ]
        self.stats.setText("\n".join(line for line in lines if line))

    # -- outputs ----------------------------------------------------------------------------
    def save_trajectory(self):
        if not self.results.t:
            self.notify.emit("Run VIO first.", True)
            return
        default = str((self.run_dir or Path.home()) / "trajectory_tum.txt")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save trajectory (TUM)", default, "TUM trajectory (*.txt)"
        )
        if not path:
            return
        from .geometry import matrix_quat_xyzw

        with open(path, "w", encoding="utf-8") as stream:
            stream.write(
                "# timestamp(s) tx ty tz qx qy qz qw  (IMU in global, Hamilton)\n"
            )
            for t, p, R in zip(self.results.t, self.results.p, self.results.R):
                q = matrix_quat_xyzw(R)
                stream.write(
                    f"{t:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
                )
        self.notify.emit(f"Saved {path}", False)

    def open_folder(self):
        folder = self.run_dir or (Path.home() / "ov_zcalib_sessions" / "vio")
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def shutdown(self):
        if self.run is not None:
            self.run.stop()
            self.run = None
        self.timer.stop()
