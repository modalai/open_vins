"""Native calibration workspace; transport, replay and solving have separate lifetimes."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import shlex
import time

import numpy as np
from PyQt5.QtCore import QProcess, QSettings, QThread, QTimer, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QShortcut,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from . import APP_NAME, TAGLINE
from .checks import inspect_log, stereo_findings
from .calibration_page import CalibrationPage
from .dialogs import SourceDialog
from .help_page import HelpPage
from .inspect_page import InspectPage
from .geometry import RigEdge, image_rotation, matrix_rpy
from .logdialogs import CameraOrientationDialog, ChannelDialog
from .logs import LogError, is_log_path, open_log, probe_log
from .logsession import FrameProvider, log_session
from .vio_page import VioPage
from .engine import (
    calibration_command,
    capture_command,
    find_executable,
    stage_from_log,
)
from .model import (
    Camera,
    Frame,
    POINT_DTYPE,
    coverage,
    demo_session,
    rate,
    read_record,
    read_report,
)
from .sources import RosSource, SourceSettings
from .widgets import (
    ACCENT,
    TEXT,
    Segmented,
    Toast,
    TransportBar,
    help_button,
    AMBER,
    MUTED,
    TEAL,
    Card,
    CoverageView,
    FeatureView,
    RigView,
    TelemetryPlot,
    button,
    font,
    icon,
    label,
    pill,
    row,
)


def table(headers):
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setAlternatingRowColors(True)
    widget.verticalHeader().hide()
    widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
    widget.setSelectionBehavior(QAbstractItemView.SelectRows)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    return widget


def clock_text(seconds):
    minutes, seconds = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes):02d}:{seconds:06.3f}"


def number(value, digits=4):
    return (
        f"{value:.{digits}g}"
        if isinstance(value, (int, float)) and np.isfinite(value)
        else "—"
    )


(
    PAGE_OVERVIEW,
    PAGE_INSPECT,
    PAGE_VIO,
    PAGE_CALIBRATION,
    PAGE_RESULTS,
    PAGE_CONSOLE,
    PAGE_HELP,
) = range(7)
PAGE_TITLES = ("Overview", "Inspect", "VIO", "Calibrate", "Results", "Console", "Help")
PAGE_HELP_ANCHORS = (
    "overview",
    "inspect",
    "vio",
    "calibrate",
    "results",
    "cli",
    "flow",
)
NAV_ITEMS = (
    ("Overview", "workspace", PAGE_OVERVIEW),
    ("Inspect", "inspect", PAGE_INSPECT),
    ("VIO", "trajectory", PAGE_VIO),
    ("Calibrate", "calibrate", PAGE_CALIBRATION),
    ("Results", "results", PAGE_RESULTS),
)


class LogLoader(QThread):
    """Opens (or re-selects channels of) a log and inspects it off the GUI thread."""

    loaded = pyqtSignal(object, object)
    problem = pyqtSignal(str)
    progress = pyqtSignal(float, str)

    def __init__(self, parent, path=None, include=None, log=None):
        super().__init__(parent)
        self.path, self.include, self.log = path, include, log

    def run(self):
        created = None
        try:
            report = lambda fraction, name: self.progress.emit(
                float(fraction), str(name)
            )
            if self.log is None:
                created = open_log(
                    self.path,
                    self.include,
                    progress=report,
                    cancelled=self.isInterruptionRequested,
                )
                log = created
            else:
                log = self.log.load(
                    self.include,
                    progress=report,
                    cancelled=self.isInterruptionRequested,
                )
            self.loaded.emit(log, inspect_log(log))
        except InterruptedError:
            if created is not None:
                created.close()
        except Exception as error:
            if created is not None:
                created.close()
            self.problem.emit(str(error))


class Worker(QThread):
    """Runs one callable off the GUI thread; emits its result or error text."""

    done = pyqtSignal(object)
    problem = pyqtSignal(str)
    progress = pyqtSignal(float, str)

    def __init__(self, parent, function):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.done.emit(self.function(self))
        except InterruptedError:
            self.problem.emit("Cancelled.")
        except Exception as error:
            self.problem.emit(
                f"{type(error).__name__}: {error}" if not str(error) else str(error)
            )


class SessionLoader(QThread):
    loaded = pyqtSignal(object)
    problem = pyqtSignal(str)

    def __init__(self, path, parent):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            self.loaded.emit(read_record(self.path, self.isInterruptionRequested))
        except InterruptedError:
            pass
        except Exception as error:
            self.problem.emit(str(error))


class StudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} · {TAGLINE}")
        self.resize(1510, 960)
        self.setMinimumSize(1180, 780)
        self.preferences = QSettings("OpenVINS", "CalibrationStudio")
        try:
            self.source_settings = SourceSettings(
                **json.loads(self.preferences.value("source", "{}"))
            )
        except (TypeError, ValueError):
            self.source_settings = SourceSettings()
        self.session = demo_session()
        self.source = None
        self.loader = None
        self.log_loader = None
        self.log_source = None
        self.inspection = None
        self.provider = None
        self.workers = []
        self._last_images = {}
        self.offset = 8.0
        self.playing = True
        self.last_tick = time.monotonic()
        self.report = None
        self.report_path = ""
        self.record_path = ""
        self.pending_report = ""
        self.job_active = False
        self.capture_active = False
        self.capture_finishing = False
        self.closing = False
        self.stage = "READY"
        self._last_live_frame = None
        self.solver = QProcess(self)
        self.solver.setProcessChannelMode(QProcess.MergedChannels)
        self.solver.readyReadStandardOutput.connect(
            lambda: self._read_process(self.solver, "solver")
        )
        self.solver.finished.connect(self._solver_finished)
        self.solver.errorOccurred.connect(self._solver_error)
        self.recorder = QProcess(self)
        self.recorder.setProcessChannelMode(QProcess.MergedChannels)
        self.recorder.readyReadStandardOutput.connect(
            lambda: self._read_process(self.recorder, "capture")
        )
        self.recorder.finished.connect(self._capture_finished)
        self.recorder.errorOccurred.connect(self._capture_error)
        self._build_ui()
        self._set_session(self.session)
        self._shortcuts = []
        for key, callback in (
            ("Ctrl+O", self.open_dialog),
            ("Ctrl+S", self.save_view),
            ("Space", self.toggle_play),
            ("Left", lambda: self.step_frame(-1)),
            ("Right", lambda: self.step_frame(1)),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self.log(
            "Studio ready. Open a recorded session or connect sensor topics. The preview is synthetic.\n"
        )

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(208)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(12, 18, 12, 14)
        side.setSpacing(2)
        mark = QLabel()
        mark.setPixmap(icon("calibrate", "#ffffff", 22).pixmap(22, 22))
        mark.setAlignment(Qt.AlignCenter)
        mark.setFixedSize(34, 34)
        mark.setStyleSheet(f"background:{ACCENT}; border-radius:9px;")
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(label(APP_NAME, 10.5, bold=True))
        titles.addWidget(label("Inspection for VINS", 8.5, MUTED))
        side.addLayout(row(mark, titles))
        side.addSpacing(22)
        side.addWidget(label("Workflow", 8.5, MUTED, True))
        side.addSpacing(4)
        self.nav = {}
        for name, symbol, page in NAV_ITEMS:
            self.nav[page] = self._nav_button(name, symbol, page)
            side.addWidget(self.nav[page])
        side.addStretch()
        for name, symbol, page in (
            ("Console", "console", PAGE_CONSOLE),
            ("Help", "info", PAGE_HELP),
        ):
            self.nav[page] = self._nav_button(name, symbol, page)
            side.addWidget(self.nav[page])
        self.nav[PAGE_OVERVIEW].setChecked(True)
        root.addWidget(sidebar)
        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(26, 18, 26, 14)
        body.setSpacing(14)
        heading = QVBoxLayout()
        heading.setSpacing(2)
        self.page_title = label("Overview", 21, bold=True)
        heading.addWidget(self.page_title)
        self.page_subtitle = label(
            "Open a log, a bag or a session to begin.", 10, MUTED
        )
        heading.addWidget(self.page_subtitle)
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addLayout(heading)
        header.addStretch()
        self.source_badge = pill("PREVIEW", MUTED)
        header.addWidget(self.source_badge, 0, Qt.AlignVCenter)
        self.help_button = help_button(
            lambda: self.show_help(PAGE_HELP_ANCHORS[self.pages.currentIndex()])
        )
        header.addWidget(self.help_button, 0, Qt.AlignVCenter)
        self.connect_button = button("Disconnect", self.connect_dialog, "connect")
        self.connect_button.hide()
        header.addWidget(self.connect_button, 0, Qt.AlignVCenter)
        self.open_button = button("Open", self.open_dialog, "folder", True)
        menu = QMenu(self.open_button)
        menu.addAction(
            "Open file…   (log info.json, bag, session, report)", self.open_dialog
        )
        menu.addAction(
            "Open log folder…   (voxl-logger, ROS 2 bag)", self.open_log_folder_dialog
        )
        menu.addSeparator()
        menu.addAction("Connect sensors…   (live ROS 2 / ROS 1)", self.connect_dialog)
        self.open_button.setMenu(menu)
        header.addWidget(self.open_button, 0, Qt.AlignVCenter)
        body.addLayout(header)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._workspace())
        self.inspect_page = InspectPage()
        self.inspect_page.seek.connect(self.seek_time)
        self.inspect_page.channels_changed.connect(self._channels_changed)
        self.inspect_page.action.connect(self._inspect_action)
        self.pages.addWidget(self.inspect_page)
        self.vio_page = VioPage(lambda: self.provider)
        self.vio_page.seek.connect(self.seek_time)
        self.vio_page.notify.connect(self.notify)
        self.pages.addWidget(self.vio_page)
        self.calibration_page = CalibrationPage(self.preferences)
        self.calibration_page.run_requested.connect(self.start_calibration)
        self.calibration_page.cancel_requested.connect(self.cancel_calibration)
        self.calibration_page.show_results.connect(lambda: self.navigate(PAGE_RESULTS))
        self.calibration_page.open_folder.connect(self.open_output_folder)
        self.calibration_page.help_requested.connect(self.show_help)
        # Historical attribute names (scripts and integration tests drive these).
        self.binary = self.calibration_page.binary
        self.output_dir = self.calibration_page.output_dir
        self.budget = self.calibration_page.budget
        self.threads = self.calibration_page.threads
        self.pages.addWidget(self.calibration_page)
        self.pages.addWidget(self._results_page())
        self.pages.addWidget(self._console_page())
        self.help_page = HelpPage()
        self.pages.addWidget(self.help_page)
        for bar in (
            self.transport,
            self.inspect_page.transport,
            self.vio_page.transport,
        ):
            bar.play.connect(self.toggle_play)
            bar.rewind.connect(self.rewind)
            bar.step.connect(self.step_frame)
            bar.speed_changed.connect(self.speed.setCurrentIndex)
            bar.loop_changed.connect(self.loop.setChecked)
        body.addWidget(self.pages, 1)
        root.addWidget(content, 1)
        self.toast = Toast(central)
        self.notice = (
            self.toast.text
        )  # the last notification's text (scripts and tests read it)
        self.statusBar().hide()

    def _nav_button(self, name, symbol, page):
        item = button(name, lambda checked=False, i=page: self.navigate(i), symbol)
        item.setObjectName("nav")
        item.setCheckable(True)
        item.setFixedHeight(32)
        return item

    def show_help(self, anchor="flow"):
        self.navigate(PAGE_HELP)
        self.help_page.show_section(anchor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "toast") and self.toast.isVisible():
            self.toast.place()

    def _workspace(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        self.metric_values = []
        self.metric_notes = []
        for title, note in (
            ("Tracked features", "Stable feature IDs"),
            ("Camera rate", "Image timestamps"),
            ("IMU rate", "Raw sensor samples"),
            ("Span", "Recorded sensor time"),
        ):
            card = Card(margins=12)
            card.box.setSpacing(1)
            card.box.addWidget(label(title, 8.5, MUTED))
            value = label("—", 17, bold=True)
            card.box.addWidget(value)
            self.metric_values.append(value)
            detail = label(note, 8, MUTED)
            card.box.addWidget(detail)
            self.metric_notes.append(detail)
            metrics.addWidget(card)
        layout.addLayout(metrics)
        middle = QSplitter(Qt.Horizontal)
        middle.setHandleWidth(12)
        viewer = Card(margins=10)
        self.view_switch = Segmented(
            (("All cameras", 1), ("Selected camera", 0), ("Rig", 2))
        )
        self.view_switch.changed.connect(
            lambda value: self.view_tabs.setCurrentIndex(value)
        )
        self.camera_box = QComboBox()
        self.camera_box.setMinimumWidth(150)
        self.camera_box.currentIndexChanged.connect(self._camera_changed)
        self.grid_check = QCheckBox("Coverage")
        self.grid_check.setChecked(True)
        self.trails_check = QCheckBox("Trails")
        self.trails_check.setChecked(True)
        self.grid_check.toggled.connect(
            lambda value: self._toggle_overlay("show_grid", value)
        )
        self.trails_check.toggled.connect(
            lambda value: self._toggle_overlay("show_trails", value)
        )
        reset = button("", self._reset_views, "reset")
        reset.setToolTip("Reset zoom and orbit")
        reset.setFixedWidth(34)
        viewer.box.addLayout(
            row(
                self.view_switch,
                None,
                self.camera_box,
                self.grid_check,
                self.trails_check,
                reset,
            )
        )
        self.view_tabs = QTabWidget()
        self.view_tabs.tabBar().hide()
        self.feature_view = FeatureView()
        self.rig_view = RigView()
        self.view_tabs.addTab(self.feature_view, "Selected camera")
        self.multi_page = QWidget()
        self.multi_layout = QGridLayout(self.multi_page)
        self.multi_layout.setContentsMargins(0, 0, 0, 0)
        self.multi_layout.setSpacing(8)
        self.multi_views = []
        self.multi_tab = QScrollArea()
        self.multi_tab.setWidgetResizable(True)
        self.multi_tab.setWidget(self.multi_page)
        self.view_tabs.addTab(self.multi_tab, "All cameras")
        self.view_tabs.addTab(self.rig_view, "Rig")
        self.view_tabs.currentChanged.connect(self._view_changed)
        viewer.box.addWidget(self.view_tabs, 1)
        left = QWidget()
        leftbox = QVBoxLayout(left)
        leftbox.setContentsMargins(0, 0, 0, 0)
        leftbox.setSpacing(12)
        leftbox.addWidget(viewer, 1)
        telemetry = QHBoxLayout()
        telemetry.setSpacing(12)
        self.gyro_plot = TelemetryPlot("Angular velocity", "rad/s", [1, 2, 3])
        self.accel_plot = TelemetryPlot("Linear acceleration", "m/s²", [4, 5, 6])
        for plot in (self.gyro_plot, self.accel_plot):
            card = Card(margins=0)
            card.box.addWidget(plot)
            card.setMaximumHeight(190)
            telemetry.addWidget(card)
        leftbox.addLayout(telemetry)
        middle.addWidget(left)
        inspector = QWidget()
        right = QVBoxLayout(inspector)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)
        source = Card(margins=14)
        source.box.setSpacing(6)
        source.box.addWidget(label("Source", 8.5, MUTED))
        self.session_name = label("", 12, bold=True)
        self.session_name.setWordWrap(True)
        source.box.addWidget(self.session_name)
        self.session_meta = label("", 9, MUTED)
        self.session_meta.setWordWrap(True)
        source.box.addWidget(self.session_meta)
        self.capture_button = button("Record session", self.toggle_capture, "record")
        self.capture_button.setToolTip(
            "Record ROS camera + IMU with the default TrackKLT frontend"
        )
        source.box.addWidget(self.capture_button)
        right.addWidget(source)
        self.quality_card = Card(margins=14)
        self.quality_card.box.setSpacing(6)
        self.quality_card.box.addWidget(label("Image-space coverage", 8.5, MUTED))
        self.coverage_view = CoverageView()
        self.coverage_view.setFixedHeight(68)
        self.quality_card.box.addWidget(self.coverage_view)
        self.coverage_text = label("", 9, TEXT)
        self.quality_card.box.addWidget(self.coverage_text)
        hint = label(
            "Start still. Rotate about every axis, then vary translation and speed.",
            9,
            MUTED,
        )
        hint.setWordWrap(True)
        self.quality_card.box.addWidget(hint)
        right.addWidget(self.quality_card)
        right.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inspector)
        scroll.setMinimumWidth(250)
        middle.addWidget(scroll)
        middle.setSizes([900, 270])
        middle.setStretchFactor(0, 1)
        middle.setStretchFactor(1, 0)
        layout.addWidget(middle, 1)
        playback = Card(margins=10)
        self.transport = TransportBar()
        # Historical names of the playback controls.
        self.play_button, self.rewind_button = (
            self.transport.play_button,
            self.transport.rewind_button,
        )
        self.time_label, self.duration_label = (
            self.transport.time_label,
            self.transport.duration_label,
        )
        self.speed, self.loop = self.transport.speed, self.transport.loop
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 10000)
        self.slider.sliderMoved.connect(self._seek)
        self.slider.sliderPressed.connect(self._scrub_start)
        self.slider.sliderReleased.connect(self._scrub_end)
        playback.box.addWidget(self.slider)
        playback.box.addWidget(self.transport)
        layout.addWidget(playback)
        return page

    def _results_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        self.result_summary = label(
            "Your calibration evidence will appear here.", 18, bold=True
        )
        layout.addWidget(self.result_summary)
        self.result_detail = label(
            "Open a report.json or finish a calibration to inspect its verdict and committed values.",
            10,
            MUTED,
        )
        self.result_detail.setWordWrap(True)
        layout.addWidget(self.result_detail)
        layout.addLayout(
            row(
                button("Open report", self.open_report_dialog, "folder"),
                button("Export report", self.export_report, "download"),
                button("Output folder", self.open_output_folder, "folder"),
                None,
            )
        )
        tabs = QTabWidget()
        self.blocks_table = table(
            ["Block", "Camera", "Decision", "σ / prior", "Worst local σ", "Gate detail"]
        )
        tabs.addTab(self.blocks_table, "Commit decisions")
        values = QSplitter(Qt.Horizontal)
        self.values_tree = QTreeWidget()
        self.values_tree.setHeaderLabels(["Committed value", "Value"])
        self.values_tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.values_tree.setAlternatingRowColors(True)
        values.addWidget(self.values_tree)
        self.result_rig = RigView()
        values.addWidget(self.result_rig)
        values.setSizes([600, 500])
        tabs.addTab(values, "Values + sensor rig")
        self.posterior_table = table(["Parameter", "Local 1σ", "Prior 1σ", "σ / prior"])
        tabs.addTab(self.posterior_table, "Posterior precision")
        self.raw_report = QPlainTextEdit()
        self.raw_report.setReadOnly(True)
        self.raw_report.setFont(font(9, mono=True))
        tabs.addTab(self.raw_report, "Full evidence")
        layout.addWidget(tabs, 1)
        caption = label(
            "Posterior σ describes local curvature precision. Accuracy coverage is uncalibrated; frozen blocks retain their seeds.",
            9,
            MUTED,
        )
        caption.setWordWrap(True)
        layout.addWidget(caption)
        return page

    def _console_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        layout.addLayout(
            row(
                label("Session console", 18, bold=True),
                None,
                button("Save log", self.save_log, "download"),
            )
        )
        layout.addWidget(
            label(
                "Capture messages, calibration stages, and storage errors from the actual processes.",
                10,
                MUTED,
            )
        )
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(font(9, mono=True))
        self.console.document().setMaximumBlockCount(8000)
        layout.addWidget(self.console, 1)
        return page

    def navigate(self, index):
        self.pages.setCurrentIndex(index)
        for page, item in self.nav.items():
            item.setChecked(page == index)
        self.page_title.setText(PAGE_TITLES[index])
        self._refresh_header()
        if self.session.log is not None and self.source is None:
            self._refresh_view()

    def _refresh_header(self):
        """One line under the title: what is open, and what this page does with it."""
        index = self.pages.currentIndex() if hasattr(self, "pages") else 0
        session, log = self.session, self.session.log
        if self.source is not None:
            data = f"Live {self.source_settings.transport}"
        elif log is not None:
            data = f"{log.name} · {log.format_label} · {log.duration:.1f} s"
        elif session.demo:
            data = "Synthetic preview — open a log, a bag or a session"
        else:
            data = f"{session.name} · session · {session.duration:.1f} s"
        purpose = (
            "",
            "Health of every pipe and topic",
            "OpenVINS on this data",
            "Targetless camera–IMU calibration",
            "Verdict and committed values",
            "Process output",
            "How the studio works",
        )[index]
        self.page_subtitle.setText(
            data
            if index == PAGE_OVERVIEW
            else f"{purpose} · {data}"
            if index not in (PAGE_HELP, PAGE_CONSOLE)
            else purpose
        )
        live = self.source is not None
        self.connect_button.setVisible(live)
        self.source_badge.setVisible(live)
        tracked = live or (not session.demo and log is None)
        self.capture_button.setVisible(live)
        self.quality_card.setVisible(tracked or session.demo)

    def log(self, text):
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def notify(self, text, error=False):
        self.toast.show_message(text, "error" if error else "info")
        self.log(("[error] " if error else "[studio] ") + text + "\n")

    def _toggle_overlay(self, key, value):
        for view in [self.feature_view, *self.multi_views]:
            setattr(view, key, value)
            view.update()

    def _reset_views(self):
        self.feature_view.reset_view()
        for view in self.multi_views:
            view.reset_view()
        self.rig_view.reset_view()

    def _camera_changed(self, index):
        if index >= 0 and self.session.log is not None:
            self.feature_view.title = self.camera_box.itemText(index)
        if index >= 0:
            self.feature_view.history.clear()
            self._refresh_view()

    def _set_session(self, session):
        if session.log is None:
            self._close_log()
        self.session = session
        if session.cam_mode in (0, 1, 2):
            self.calibration_page.intrinsics.set_value(session.cam_mode)
        if not self.job_active:
            self._set_stage(
                "READY",
                "Ready for your data"
                if session.demo
                else "Choose cameras, then Calibrate"
                if session.log is not None
                else "Recording ready to solve",
            )
        self.offset = min(8.0, session.duration) if session.demo else 0.0
        self.playing = session.demo
        self.last_tick = time.monotonic()
        self._last_live_frame = None
        self.feature_view.history.clear()
        self.feature_view.reset_view()
        names = (
            [cam.name for cam in session.log.cameras]
            if session.log is not None
            else [f"Camera {i:02d}" for i in range(len(session.cameras))]
        )
        self.camera_box.blockSignals(True)
        self.camera_box.clear()
        self.camera_box.addItems(names)
        self.camera_box.blockSignals(False)
        self._setup_multi_views(names)
        # Several cameras open on the grid, one camera on its large view.
        self.view_tabs.setCurrentIndex(1 if len(names) > 1 else 0)
        self._view_changed(self.view_tabs.currentIndex())
        captions = ("", "log") if session.log is not None else None
        for index, view in enumerate([self.feature_view, *self.multi_views]):
            view.captions = captions
            view.title = (
                (names[index - 1] if index else "") if session.log is not None else ""
            )
        if session.log is not None and names:
            self.feature_view.title = names[0]
        if session.log is not None:
            known = sum(getattr(session, "extrinsics_known", []))
            caption = (
                f"{session.log.rig.source}: {known} of {len(session.cameras)} cameras"
                if known
                else "No camera extrinsics in this log"
            )
            self.rig_view.set_cameras(
                [c for c, k in zip(session.cameras, session.extrinsics_known) if k],
                caption,
            )
        else:
            self.rig_view.set_cameras(
                session.cameras,
                "Recording seed" if not session.demo else "Preview seed",
            )
        self.session_name.setText(session.name)
        if session.log is not None:
            log = session.log
            self.session_meta.setText(
                f"{len(log.cameras)} camera{'s' if len(log.cameras) != 1 else ''} · {len(log.imus)} IMU · {len(session.frames):,} frames\n"
                f"{log.format_label} · images from the log · no tracks"
            )
        integrity = (
            "Complete recording"
            if session.complete
            else "Legacy record · completeness unknown"
        )
        if session.log is None:
            self.session_meta.setText(
                f"{len(session.cameras)} camera{'s' if len(session.cameras) > 1 else ''} · 1 IMU · {len(session.frames):,} frames\n"
                f"{'Synthetic viewer example' if session.demo else integrity}"
            )
        self.source_badge.setText(
            "SYNTHETIC PREVIEW"
            if session.demo
            else (session.log.format_label.upper() + " REPLAY")
            if session.log is not None
            else "SESSION REPLAY"
        )
        self.capture_button.setEnabled(False)
        if session.log is None and session.path and not session.demo:
            self.calibration_page.set_session(session.path)
        for control in (
            self.play_button,
            self.rewind_button,
            self.slider,
            self.speed,
            self.loop,
        ):
            control.setEnabled(True)
        self.duration_label.setText(clock_text(session.duration))
        self.metric_notes[0].setText(
            "Synthetic observations"
            if session.demo
            else "Images only · no tracks"
            if session.log is not None
            else "Recorded observations"
        )
        self.metric_notes[2].setText("Raw sensor samples")
        if session.path:
            sidecar = Path(session.path + ".capture.json")
            try:
                if (
                    json.loads(sidecar.read_text(encoding="utf-8")).get(
                        "temperature_available"
                    )
                    is False
                ):
                    self.metric_notes[2].setText("Temperature unavailable")
            except (OSError, ValueError, AttributeError):
                pass
        self._refresh_view()

    def _refresh_view(self):
        if self.source is not None:
            return
        cam_index = max(0, self.camera_box.currentIndex())
        points = np.zeros(0, dtype=POINT_DTYPE)
        if self.session.cameras:
            cam_index = min(cam_index, len(self.session.cameras) - 1)
            cam = self.session.cameras[cam_index]
            frame = self.session.frame_at(self.offset, cam_index)
            # Logs decode images on demand: only the visible views request them.
            log = self.session.log is not None
            overview = self.pages.currentIndex() == PAGE_OVERVIEW
            tab = self.view_tabs.currentIndex()
            if not log or (overview and tab == 0):
                frame = self._with_image(frame, cam_index, self.feature_view)
                self.feature_view.set_frame(frame, cam, self.session.demo)
            if overview and (not log or tab == 1):
                for index, view in enumerate(self.multi_views):
                    view.set_frame(
                        self._with_image(
                            self.session.frame_at(self.offset, index), index, view
                        ),
                        self.session.cameras[index],
                        self.session.demo,
                    )
            if frame is not None:
                points = frame.points
            self._coverage(points, cam)
            self.metric_values[1].setText(
                f"{self.session.camera_rates[cam_index]:.1f} Hz"
            )
        else:
            self.feature_view.set_frame(None, None)
            self.metric_values[1].setText("—")
        self._refresh_playheads()
        data = self.session.imu_window(self.offset)
        self.gyro_plot.set_data(data)
        self.accel_plot.set_data(data)
        self.metric_values[0].setText(f"{len(points):,}")
        self.metric_values[2].setText(
            f"{self.session.imu_rate:.0f} Hz" if len(self.session.imu) else "—"
        )
        self.metric_values[3].setText(f"{self.session.duration:.1f} s")
        self.time_label.setText(clock_text(self.offset))
        if not self.slider.isSliderDown():
            self.slider.setValue(
                round(self.offset / max(self.session.duration, 0.001) * 10000)
            )
        self.play_button.setIcon(icon("pause" if self.playing else "play"))

    # -- logs ---------------------------------------------------------------------
    def _with_image(self, frame, cam_index, view=None):
        """Attach the decoded log image to a replay frame (asynchronous, cached)."""
        if frame is None or self.session.log is None or self.provider is None:
            return frame
        if view is not None:
            self.provider.set_width(cam_index, view.width() * view.devicePixelRatioF())
        image = self.provider.image(
            cam_index, frame.seq, rate=self._playback_rate(cam_index)
        )
        if image is None:
            image = self._last_images.get(cam_index)
        else:
            self._last_images[cam_index] = image
        return Frame(
            frame.timestamp,
            frame.points,
            cam=frame.cam,
            seq=frame.seq,
            exposure=frame.exposure,
            image=image,
        )

    def _speed_factor(self):
        return (0.25, 0.5, 1.0, 2.0, 4.0)[self.speed.currentIndex()]

    def _playback_rate(self, cam_index):
        """Frames per second of wall time for a camera while playing (0 when paused)."""
        if not self.playing or cam_index >= len(self.session.camera_rates):
            return 0.0
        return self.session.camera_rates[cam_index] * self._speed_factor()

    def _refresh_playheads(self):
        t = self.session.start + self.offset
        page = self.pages.currentIndex()
        is_log = self.session.log is not None
        for bar in (self.inspect_page.transport, self.vio_page.transport):
            bar.set_state(
                self.playing,
                clock_text(self.offset),
                clock_text(self.session.duration),
                self.speed.currentIndex(),
                self.loop.isChecked(),
                enabled=is_log and self.source is None,
            )
        if is_log:
            self.inspect_page.set_playhead(t)
            if page == PAGE_INSPECT:
                self._refresh_preview()
            if page == PAGE_VIO:
                self.vio_page.playback_rate = (
                    self._speed_factor() if self.playing else 0.0
                )
                self.vio_page.set_playhead(t)

    def _refresh_preview(self):
        page, log = self.inspect_page, self.session.log
        if log is None:
            page.preview.set_frame(None, None)
            return
        if page.preview_mode() == "all":
            for index, view in enumerate(
                page.preview_grid.views[: len(self.session.cameras)]
            ):
                frame = self._with_image(
                    self.session.frame_at(self.offset, index), index, view
                )
                view.set_frame(frame, self.session.cameras[index])
            return
        cam = page.selected_camera()
        if cam is None or cam not in log.cameras:
            page.preview.set_frame(None, None)
            return
        index = log.cameras.index(cam)
        frame = self._with_image(
            self.session.frame_at(self.offset, index), index, page.preview
        )
        page.preview.set_frame(frame, self.session.cameras[index])

    def seek_time(self, t):
        """Seek every view to a stream time (seconds in the session/log time base)."""
        self.offset = float(np.clip(t - self.session.start, 0.0, self.session.duration))
        self.playing = False
        self._refresh_view()

    def open_log_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open a voxl-logger log or a ROS 2 bag folder"
        )
        if path:
            self.open_path(path)

    def _close_log(self):
        if self.provider is not None:
            self.provider.stop()
            self.provider.deleteLater()
            self.provider = None
        if self.log_source is not None:
            self.vio_page.set_log(None)
            self.inspect_page.clear()
            self.calibration_page.set_log(None)
            self.log_source.close()
            self.log_source = None
            self.inspection = None
        self._last_images = {}

    def open_log(self, path, ask=True):
        if self.capture_active:
            self.notify("Stop recording before opening another input.", True)
            return
        if self.log_loader is not None and self.log_loader.isRunning():
            self.notify("A log is already loading.")
            return
        try:
            kind, name, channels = probe_log(path)
        except (LogError, OSError, ValueError) as error:
            self.notify(str(error), True)
            return
        include = None
        if ask:
            label_text = {
                "voxl": "voxl-logger pipes from info.json",
                "ros1": "ROS 1 bag topics",
                "ros2": "ROS 2 bag topics",
            }.get(kind, kind)
            dialog = ChannelDialog(f"Open {name}", label_text, channels, parent=self)
            if dialog.exec_() != QDialog.Accepted:
                return
            include = dialog.selected()
        if not self._disconnect_source():
            return
        self._start_log_loader(LogLoader(self, path=path, include=include))

    def _start_log_loader(self, loader):
        self.playing = False
        self.open_button.setEnabled(False)
        self.statusBar().showMessage("Reading log…")
        self.log_loader = loader
        loader.loaded.connect(self._log_loaded)
        loader.problem.connect(lambda text: self.notify(text, True))
        loader.progress.connect(
            lambda f, name: self.statusBar().showMessage(f"Reading {name} · {f:.0%}")
        )
        loader.finished.connect(self._log_loader_finished)
        loader.start()

    def _log_loader_finished(self):
        self.open_button.setEnabled(True)
        if self.log_loader is not None:
            self.log_loader.deleteLater()
            self.log_loader = None
        queued = getattr(self, "_reinspect_queued", None)
        if queued is not None:
            self._reinspect_queued = None
            self._reinspect(queued)

    def _log_loaded(self, log, inspection):
        same = log is self.log_source
        offset = self.offset
        if not same:
            self._close_log()
        elif self.provider is not None:
            self.provider.stop()
            self.provider.deleteLater()
            self.provider = None
        self.log_source, self.inspection = log, inspection
        self._last_images = {}
        session = log_session(log)
        self.provider = FrameProvider(log.cameras, parent=self)
        self.provider.ready.connect(self._provider_ready)
        self._set_session(session)
        self.offset = min(offset, session.duration) if same else 0.0
        self.playing = False
        self.inspect_page.set_inspection(log, inspection)
        self.inspect_page.set_orientation_state(log.orientation_changes())
        self.vio_page.set_log(log)
        self.vio_page.set_inspection(inspection)
        if not same:
            self.navigate(PAGE_INSPECT)
        self.calibration_page.set_log(log, inspection)
        self._refresh_view()
        notice = getattr(self, "_pending_notice", "")
        self._pending_notice = ""
        if same and notice:
            self.notify(notice)
            return
        problems, warnings = inspection.count("problem"), inspection.count("warning")
        self.notify(
            f"Opened {log.name} · {log.format_label} · {log.duration:.1f} s · "
            + (
                "no problems"
                if not problems
                else f"{problems} problem{'s' if problems != 1 else ''}"
            )
            + (
                f", {warnings} warning{'s' if warnings != 1 else ''}"
                if warnings
                else ""
            ),
            problems > 0,
        )

    def _provider_ready(self):
        if self.session.log is not None and not self.playing:
            self._refresh_view()

    def _channels_changed(self, names):
        if self.log_source is None:
            return
        if self.log_loader is not None and self.log_loader.isRunning():
            self.notify("The log is still loading.")
            return
        if self.vio_page.run is not None:
            self.notify("Stop the VIO run before changing channels.", True)
            self.inspect_page.set_inspection(self.log_source, self.inspection)
            return
        self._start_log_loader(LogLoader(self, include=names, log=self.log_source))

    def _inspect_action(self, action):
        log = self.log_source
        if action == "open":
            self.open_dialog()
        elif action == "preview":
            self._refresh_preview()
        elif log is None:
            return
        elif action.startswith(("rotate-", "orientation")):
            self._orientation_action(action)
        elif action == "channels":
            dialog = ChannelDialog(
                f"Channels of {log.name}",
                "Pipes / topics of this log",
                log.channels,
                log.included,
                self,
                "Apply",
            )
            if dialog.exec_() == QDialog.Accepted:
                self._channels_changed(dialog.selected())
        elif action == "export":
            self._export_inspection()
        elif action == "convert":
            self.navigate(PAGE_CALIBRATION)
        elif action == "vio":
            self.navigate(PAGE_VIO)
        elif action == "stereo":
            self._stereo_check()

    # -- camera orientation (rotation + extrinsics) -------------------------------
    def _rotated_edge(self, cam, delta):
        """The camera's current extrinsic edge turned with a clockwise image rotation, or None."""
        log = self.log_source
        frame = log.camera_frame(cam)
        edge = next((e for e in log.rig.edges if e.child == frame), None)
        if edge is None:
            return None
        return RigEdge(
            edge.parent, edge.child, edge.R @ image_rotation(delta), edge.t.copy()
        )

    def _orientation_action(self, action):
        log, cam = self.log_source, self.inspect_page.selected_camera()
        if self.vio_page.run is not None:
            self.notify(
                "Stop the VIO run before changing a camera's orientation.", True
            )
            return
        if action in ("rotate-cw", "rotate-ccw"):
            if cam is None:
                return
            delta = 90 if action == "rotate-cw" else 270
            edge = self._rotated_edge(cam, delta)
            log.set_orientation(cam, (cam.rotation + delta) % 360, edge)
            extra = "" if edge is not None else " (it has no extrinsics to rotate)"
            self._reinspect(
                f"Camera {cam.camera or cam.name} rotated to {cam.rotation}°{extra} · unsaved: Save… or Revert below the preview"
            )
        elif action == "orientation":
            if cam is None:
                return
            row = max(
                0,
                int(
                    np.searchsorted(
                        cam.times, self.session.start + self.offset, side="right"
                    )
                )
                - 1,
            )
            dialog = CameraOrientationDialog(log, cam, row, self)
            if dialog.exec_() != QDialog.Accepted:
                return
            log.set_orientation(cam, dialog.rotation_value, dialog.edge())
            if dialog.choice == "save":
                self._save_orientation()
            else:
                self._reinspect(
                    f"Camera {cam.camera or cam.name} orientation applied for this session · unsaved"
                )
        elif action == "orientation-save":
            self._save_orientation()
        elif action == "orientation-revert":
            log.revert_orientation()
            self._reinspect("Orientation edits reverted")

    def _save_orientation(self):
        log = self.log_source
        rotations, edges = log.orientation_changes()
        if not rotations and not edges:
            self.notify("No orientation changes to save.")
            self._reinspect()
            return
        lines = [
            f"• camera {name}: image rotation {old}° → {new}°"
            + (
                f" (pipes {', '.join(pipes)})"
                if (
                    pipes := [
                        c.name
                        for c in log.cameras
                        if c.camera == name and c.name != name
                    ]
                )
                else ""
            )
            for name, (old, new) in sorted(rotations.items())
        ]
        for child, edge in sorted(edges.items()):
            rpy = ", ".join(f"{v:g}" for v in matrix_rpy(edge.R))
            t = ", ".join(f"{v:g}" for v in edge.t)
            lines.append(f"• {child}: parent {edge.parent}, RPY [{rpy}] deg, T [{t}] m")
        files = []
        if log.kind == "voxl" and edges:
            files.append(
                f"{log.path / 'etc' / 'modalai' / 'extrinsics.conf'}  (a timestamped .bak copy is kept)"
            )
        files.append(
            f"{log.overrides_path}  (studio camera rotations{'' if log.kind == 'voxl' else ' and extrinsics'})"
        )
        answer = QMessageBox.question(
            self,
            "Save orientation?",
            "Save these changes?\n\n"
            + "\n".join(lines)
            + "\n\nFiles written:\n"
            + "\n".join(files)
            + "\n\nLogged frames are never modified; the studio rotates them when it reads the log.",
            QMessageBox.Save | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if answer != QMessageBox.Save:
            self._reinspect("Not saved · the edits stay active for this session")
            return
        try:
            written = log.save_orientation()
        except Exception as error:
            self.notify(f"Could not save the orientation: {error}", True)
            return
        for path, backup in written:
            self.log(
                f"[orientation] wrote {path}"
                + (f" (backup {backup})" if backup else "")
                + "\n"
            )
        self._reinspect("Saved " + ", ".join(Path(path).name for path, _ in written))

    def _reinspect(self, message=""):
        """Re-run the inspection after a configuration edit (cached channels; fast)."""
        if self.log_source is None:
            return
        if self.log_loader is not None:
            # A re-inspection is running: run once more afterwards with the newest state.
            self._reinspect_queued = (
                message or getattr(self, "_reinspect_queued", None) or ""
            )
            return
        self._pending_notice = message
        self._start_log_loader(
            LogLoader(self, include=list(self.log_source.included), log=self.log_source)
        )

    def _run_worker(self, function, done, busy_text):
        worker = Worker(self, function)
        self.workers.append(worker)
        worker.done.connect(done)
        worker.problem.connect(lambda text: self.notify(text, True))
        worker.progress.connect(
            lambda f, name: self.statusBar().showMessage(
                f"{busy_text} · {name} {f:.0%}"
            )
        )
        worker.finished.connect(
            lambda: (
                self.workers.remove(worker) if worker in self.workers else None,
                worker.deleteLater(),
            )
        )
        self.statusBar().showMessage(busy_text)
        worker.start()
        return worker

    def _export_inspection(self):
        default = str(
            Path(self.calibration_page.output_dir.text()).expanduser()
            / f"{self.log_source.name}_inspection.html"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Export inspection report", default, "HTML report (*.html)"
        )
        if not path:
            return
        inspection = self.inspection

        def export(worker):
            import tempfile

            from .logreport import plot_figures, write_html

            with tempfile.TemporaryDirectory() as folder:
                plots = plot_figures(inspection, folder)
                return write_html(inspection, path, plots)

        self._run_worker(
            export, lambda result: self.notify(f"Saved {result}"), "Exporting report"
        )

    def _stereo_check(self):
        from .checks import _stereo_pairs

        log = self.log_source
        pairs = [
            (a, b)
            for a, b in _stereo_pairs(log.cameras)
            if a.intrinsics and b.intrinsics
        ]
        if not pairs and len(log.cameras) >= 2:
            usable = [c for c in log.cameras if c.intrinsics and c.has_pictures]
            pairs = [(usable[0], usable[1])] if len(usable) >= 2 else []
        if not pairs:
            self.notify("The stereo check needs two cameras with intrinsics.", True)
            return
        inspection = self.inspection

        def check(worker):
            from .geometry import stereo_geometry

            findings = []
            for a, b in pairs:
                result = stereo_geometry(
                    a,
                    b,
                    a.intrinsics,
                    b.intrinsics,
                    progress=lambda f, a=a: worker.progress.emit(f, a.name),
                    cancelled=worker.isInterruptionRequested,
                )
                findings += stereo_findings(log, a, b, result)
            return findings

        def done(findings):
            if self.inspection is not inspection:
                return
            for finding in findings:
                inspection.add(finding)
            self.inspect_page.set_inspection(log, inspection)
            self.vio_page.set_inspection(inspection)
            self.notify(
                findings[0].message if findings else "Stereo check finished.",
                any(f.severity != "info" for f in findings),
            )

        self._run_worker(check, done, "Stereo geometry check")

    def _coverage(self, points, cam):
        grid = coverage(points, cam.width, cam.height)
        occupied = int(np.count_nonzero(grid))
        self.coverage_view.set_grid(grid)
        self.coverage_text.setText(
            f"{occupied}/48 cells occupied  ·  {occupied / 48:.0%}"
        )

    def _tick(self):
        now = time.monotonic()
        elapsed = min(0.25, now - self.last_tick)
        self.last_tick = now
        if self.source is not None:
            self._refresh_live()
        elif self.playing and not self.slider.isSliderDown():
            speed = (0.25, 0.5, 1.0, 2.0, 4.0)[self.speed.currentIndex()]
            self.offset += elapsed * speed
            if self.offset > self.session.duration:
                if self.loop.isChecked() and self.session.duration > 0:
                    self.offset %= self.session.duration
                    self.feature_view.history.clear()
                else:
                    self.offset = self.session.duration
                    self.playing = False
            self._refresh_view()
        if self.closing:
            self._try_close()

    def toggle_play(self):
        if self.source is not None or self.pages.currentIndex() not in (
            PAGE_OVERVIEW,
            PAGE_INSPECT,
            PAGE_VIO,
        ):
            return
        if isinstance(
            QApplication.focusWidget(),
            (QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox),
        ):
            return
        self.playing = not self.playing
        if self.playing and self.pages.currentIndex() == PAGE_VIO:
            self.vio_page.live_check.setChecked(
                False
            )  # replaying the estimate, not following the run
        self._refresh_view()

    def rewind(self):
        if self.source is None:
            self.offset = 0.0
            self.feature_view.history.clear()
            self._refresh_view()

    def _seek(self, value):
        if self.source is None:
            self.offset = value / 10000 * self.session.duration
            self._refresh_view()

    def _scrub_start(self):
        self._was_playing = self.playing
        self.playing = False

    def _scrub_end(self):
        self.playing = getattr(self, "_was_playing", False)
        self._seek(self.slider.value())

    def step_frame(self, direction):
        if self.source is not None or self.pages.currentIndex() not in (
            PAGE_OVERVIEW,
            PAGE_INSPECT,
            PAGE_VIO,
        ):
            return
        times = self.session.frame_times[max(0, self.camera_box.currentIndex())]
        if not len(times):
            return
        stamp = self.session.start + self.offset
        index = int(
            np.searchsorted(
                times,
                stamp + (1.0e-7 if direction > 0 else -1.0e-7),
                side="right" if direction > 0 else "left",
            )
        )
        if direction < 0:
            index -= 1
        index = max(0, min(len(times) - 1, index))
        self.offset = float(times[index] - self.session.start)
        self.playing = False
        self._refresh_view()

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open calibration data",
            "",
            "Studio data (*.bin *.json *.bag *.db3 *.mcap metadata.yaml info.json);;Session records (*.bin);;Reports (*.json);;"
            "ROS bags (*.bag *.db3 *.mcap metadata.yaml);;voxl-logger logs (info.json)",
        )
        if path:
            self.open_path(path)

    def open_report_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open calibration report", "", "ov_zcalib report (*.json)"
        )
        if path:
            self.open_path(path)

    def open_path(self, path, ask=True):
        if Path(path).name == "info.json":
            path = str(Path(path).parent)
        if is_log_path(path):
            self.open_log(path, ask)
            return
        if self.capture_active:
            self.notify("Stop recording before opening another input.", True)
            return
        if self.loader is not None and self.loader.isRunning():
            self.notify("A session is already loading.")
            return
        if str(path).lower().endswith(".json"):
            try:
                self._show_report(read_report(path), str(path))
                self.navigate(PAGE_RESULTS)
            except Exception as error:
                self.notify(str(error), True)
            return
        if not self._disconnect_source():
            return
        self.playing = False
        self.open_button.setEnabled(False)
        self.statusBar().showMessage("Loading and validating session…")
        self.loader = SessionLoader(path, self)
        self.loader.loaded.connect(self._session_loaded)
        self.loader.problem.connect(lambda text: self.notify(text, True))
        self.loader.finished.connect(self._load_finished)
        self.loader.start()

    def _session_loaded(self, session):
        self._set_session(session)
        self.navigate(PAGE_OVERVIEW)
        self.notify(
            f"Opened {session.name} · {session.duration:.1f} s · format {session.format}."
        )

    def _load_finished(self):
        self.open_button.setEnabled(True)
        if self.loader is not None:
            self.loader.deleteLater()
            self.loader = None

    def connect_dialog(self):
        if self.source is not None:
            if self.capture_active:
                self.notify("Stop recording before disconnecting sensors.", True)
                return
            if self._disconnect_source():
                self._set_session(self.session)
            return
        dialog = SourceDialog(self.source_settings, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        settings = dialog.settings()
        if not all(
            (
                settings.image_topic,
                settings.info_topic,
                settings.imu_topic,
                settings.tracks_topic,
            )
        ):
            self.notify(
                "Supply image, CameraInfo, raw IMU, and overlay topic names.", True
            )
            return
        self.start_source(settings)

    def start_source(self, settings):
        if self.capture_active or not self._disconnect_source():
            return
        try:
            cameras = settings.camera_sources()
        except (TypeError, ValueError) as error:
            self.notify(str(error), True)
            return
        self.source_settings = settings
        self.preferences.setValue("source", json.dumps(asdict(settings)))
        self.playing = False
        self.feature_view.history.clear()
        worker = RosSource(settings, self)
        self.source = worker
        worker.state.connect(lambda text: self._source_state(worker, text))
        worker.problem.connect(
            lambda text: self.notify(text, True) if self.source is worker else None
        )
        worker.finished.connect(lambda: self._source_finished(worker))
        self.camera_box.blockSignals(True)
        self.camera_box.clear()
        self.camera_box.addItems([cam["camera_name"] for cam in cameras])
        self.camera_box.blockSignals(False)
        self._setup_multi_views([cam["camera_name"] for cam in cameras])
        if len(cameras) > 1:
            self.view_tabs.setCurrentWidget(self.multi_tab)
        for control in (
            self.play_button,
            self.rewind_button,
            self.slider,
            self.speed,
            self.loop,
        ):
            control.setEnabled(False)
        self.source_badge.setText(settings.transport.upper() + " · CONNECTING")
        self.session_name.setText(settings.image_topic)
        self.session_meta.setText("Waiting for image, CameraInfo, and IMU…")
        self.connect_button.show()
        self.capture_button.setEnabled(False)
        self.navigate(PAGE_OVERVIEW)
        self._live_start = None
        self._last_live_frame = None
        worker.start()

    def _source_state(self, worker, text):
        if self.source is worker:
            self.statusBar().showMessage(text, 6000)
            self.log(f"[{worker.settings.transport}] {text}\n")

    def _source_finished(self, worker):
        if self.source is worker:
            self.source = None
            self._set_session(self.session)
        worker.deleteLater()

    def _disconnect_source(self):
        if self.source is None:
            return True
        worker = self.source
        worker.requestInterruption()
        if not worker.wait(1800):
            self.notify(
                "Sensor transport is still shutting down; try again when it disconnects.",
                True,
            )
            return False
        self.source = None
        return True

    def _refresh_live(self):
        camera_index = max(0, self.camera_box.currentIndex())
        snapshot = self.source.snapshot(camera_index)
        camera_settings = self.source.camera_settings[camera_index]
        cam = snapshot["camera"]
        image = snapshot["image"]
        frame = snapshot["frame"]
        stamp = snapshot["image_stamp"]
        if stamp is not None:
            if self._live_start is None:
                self._live_start = stamp
            # Only associate tracks with a nearby image; stale overlays disappear.
            if frame is None or abs(frame.timestamp - stamp) > 0.15:
                frame = Frame(stamp, np.zeros(0, dtype=POINT_DTYPE))
            frame = Frame(
                frame.timestamp,
                frame.points,
                cam=camera_index,
                exposure=camera_settings["exposure_s"],
                image=image,
            )
            if not snapshot["info"]:
                cam = Camera(width=image.width(), height=image.height())
        points = frame.points if frame is not None else np.zeros(0, dtype=POINT_DTYPE)
        self.feature_view.set_frame(frame, cam)
        self.rig_view.set_cameras(
            [entry["camera"] for entry in snapshot["cameras"]],
            "Live seed · extrinsics unknown",
        )
        for index, entry in enumerate(snapshot["cameras"]):
            live_frame = entry["frame"]
            live_stamp = entry["image_stamp"]
            live_cam = entry["camera"]
            if live_stamp is not None:
                if live_frame is None or abs(live_frame.timestamp - live_stamp) > 0.15:
                    live_frame = Frame(
                        live_stamp, np.zeros(0, dtype=POINT_DTYPE), cam=index
                    )
                live_frame = Frame(
                    live_frame.timestamp,
                    live_frame.points,
                    cam=index,
                    image=entry["image"],
                )
                if not entry["info"]:
                    live_cam = Camera(
                        width=entry["image"].width(), height=entry["image"].height()
                    )
            self.multi_views[index].set_frame(live_frame, live_cam)
        data = snapshot["imu"]
        if len(data):
            data = data[data[:, 0] > data[-1, 0] - 8.0][:: max(1, len(data) // 1000)]
        self.gyro_plot.set_data(data)
        self.accel_plot.set_data(data)
        self._coverage(points, cam)
        ready_cameras = sum(
            entry["image_age"] < 2.0 and entry["info"] for entry in snapshot["cameras"]
        )
        healthy = (
            ready_cameras == len(snapshot["cameras"]) and snapshot["imu_age"] < 2.0
        )
        self.source_badge.setText(
            self.source_settings.transport.upper()
            + (" · LIVE" if healthy else " · WAITING")
        )
        self.capture_button.setEnabled(
            (healthy and snapshot["info"] and not self.job_active)
            or self.capture_active
        )
        if self.capture_finishing:
            self.capture_button.setEnabled(False)
        self.metric_values[0].setText(str(len(points)))
        self.metric_notes[0].setText(
            "TrackKLT capture" if self.capture_active else "Record to track features"
        )
        self.metric_notes[2].setText("Temperature unavailable")
        self.metric_values[1].setText(
            f"{rate(snapshot['image_times']):.1f} Hz"
            if snapshot["image_age"] < 2
            else "—"
        )
        imu = snapshot["imu"]
        self.metric_values[2].setText(
            f"{rate(imu[:, 0]):.0f} Hz" if len(imu) and snapshot["imu_age"] < 2 else "—"
        )
        span = max(0.0, (stamp - self._live_start)) if stamp is not None else 0.0
        self.metric_values[3].setText(f"{span:.1f} s")
        self.time_label.setText(clock_text(span))
        self.duration_label.setText("LIVE")
        self.session_meta.setText(
            f"{ready_cameras}/{len(snapshot['cameras'])} cameras ready · {'IMU streaming' if snapshot['imu_age'] < 2 else 'Waiting for IMU'}\n"
            f"{cam.width} × {cam.height} · {camera_settings['timestamp_reference']} stamps\n"
            f"{'Recording with TrackKLT' if self.capture_active else 'Live viewer · ready to capture' if healthy else 'Waiting for sensor messages'}"
        )

    def _view_changed(self, index):
        self.view_switch.set_value(index)
        self.camera_box.setVisible(index == 0)
        self.grid_check.setVisible(index == 0)
        self.trails_check.setVisible(index == 0)
        self._refresh_view()

    def _setup_multi_views(self, names):
        while self.multi_layout.count():
            item = self.multi_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.multi_views = []
        count = len(names)
        columns = 1 if count <= 1 else 2 if count <= 4 else 3
        for index, name in enumerate(names):
            view = FeatureView()
            view.setMinimumSize(200, 140)
            view.title = name
            view.show_grid = False
            view.show_trails = self.trails_check.isChecked()
            self.multi_views.append(view)
            self.multi_layout.addWidget(view, index // columns, index % columns)
        for c in range(3):
            self.multi_layout.setColumnStretch(c, 1 if c < columns else 0)

    def _set_stage(self, state, description=""):
        self.stage = state
        self.calibration_page.set_stage(
            state, description or state.replace("_", " ").title()
        )

    def start_calibration(self):
        if self.job_active:
            return
        if self.capture_active:
            self.notify("Stop the recording before running calibration.", True)
            return
        try:
            cfg = self.calibration_page.settings()
        except ValueError as error:
            self.notify(
                f"{error} Open a session.bin or a log to calibrate first."
                if self.session.demo
                else str(error),
                True,
            )
            return
        directory = Path(cfg["output_dir"]).expanduser().resolve()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = directory / f"calibration_{stamp}.yaml"
        report = output.with_suffix(".report.json")
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.notify(f"Cannot create {directory}: {error}", True)
            return
        self.calibration_page.remember()
        self.pending_report, self.pending_yaml = str(report), str(output)
        self._cancelled = False
        self.job_active = True
        self.calibration_page.set_running(True)
        self.capture_button.setEnabled(False)
        self.navigate(PAGE_CALIBRATION)
        if cfg["source"] == "session":
            self._set_stage("SETTLE", "Replaying the session")
            self._launch_solver(cfg, cfg["session_path"], output, report)
        elif cfg["voxl_ingest"]:
            self._set_stage("SETTLE", "Reading the VOXL log")
            self._launch_solver(
                cfg,
                self.log_source.path,
                output,
                report,
                voxl=True,
                cam_pipe=cfg["cameras"][0].name,
            )
        else:
            log = self.log_source
            session_path = (
                directory
                / f"{log.name.replace('/', '_').strip('_') or 'log'}_{stamp}.bin"
            )
            names = ", ".join(c.label for c in cfg["cameras"])
            self._set_stage("PREPARE", f"Tracking {names} with TrackKLT")

            def convert(worker):
                from .stream import convert_to_session

                return convert_to_session(
                    log,
                    session_path,
                    cfg["cameras"],
                    cfg["imu"],
                    features=cfg["features"],
                    settings=cfg["settings"],
                    progress=lambda f, *rest: worker.progress.emit(
                        float(f), str(rest[0]) if rest else ""
                    ),
                    cancelled=worker.isInterruptionRequested,
                    on_line=lambda text: None,
                )

            def converted(summary):
                self.log(f"[convert] {summary}\n")
                if self._cancelled:
                    self._calibration_done(False, "Cancelled before solving")
                    return
                self.calibration_page.set_session(str(session_path))
                self._set_stage("SETTLE", "Replaying the tracked session")
                self._launch_solver(cfg, session_path, output, report)

            worker = self._run_worker(convert, converted, "Tracking frames")
            self._convert_worker = worker
            worker.problem.connect(
                lambda text: self._calibration_done(False, f"Tracking failed: {text}")
            )
            worker.progress.connect(
                lambda f, name: self.calibration_page.set_stage(
                    "PREPARE", f"Tracking {names} · {f:.0%}"
                )
            )

    def _launch_solver(self, cfg, source, output, report, voxl=False, cam_pipe=""):
        try:
            program, args = calibration_command(
                cfg["binary"],
                source,
                output,
                report,
                cfg["cam_mode"],
                cfg["budget"],
                cfg["threads"],
                cfg["profile"],
                voxl,
                cfg.get("readout_s", 0.0),
                cam_pipe,
            )
        except Exception as error:
            self._calibration_done(False, str(error))
            self.notify(str(error), True)
            return
        self.log("\n[solver] " + shlex.join([program] + args) + "\n")
        self.calibration_page.append_log("$ " + shlex.join([Path(program).name] + args))
        self.solver.start(program, args)

    def _calibration_done(self, ok, text):
        self.job_active = False
        self.calibration_page.finished(ok, text)
        self.capture_button.setEnabled(self.source is not None)
        if not ok:
            self._set_stage("ABORT", text)

    def _read_process(self, process, kind):
        output = bytes(process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if output:
            self.log(output)
            if kind == "solver":
                self.calibration_page.append_log(output)
                state = stage_from_log(output)
                if state:
                    self._set_stage(state)
            elif "[capture] READY" in output:
                self.session_meta.setText("Recording · TrackKLT")

    def cancel_calibration(self):
        if not self.job_active:
            return
        self._cancelled = True
        self.calibration_page.cancelling()
        worker = getattr(self, "_convert_worker", None)
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
        if self.solver.state() == QProcess.NotRunning:
            return
        self.solver.terminate()
        pid = self.solver.processId()
        QTimer.singleShot(
            3000,
            lambda: (
                self.solver.kill()
                if self.solver.state() != QProcess.NotRunning
                and self.solver.processId() == pid
                else None
            ),
        )

    def _solver_error(self, error):
        if error == QProcess.FailedToStart:
            self._calibration_done(False, "Could not start the engine")
            self.notify(self.solver.errorString(), True)

    def _solver_finished(self, code, status):
        self._read_process(self.solver, "solver")
        if self.closing:
            self.job_active = False
            return
        if self._cancelled:
            self._calibration_done(False, "Cancelled")
            self.notify("Calibration cancelled. No successful commit is reported.")
            return
        if not Path(self.pending_report).is_file():
            self._calibration_done(False, "No report written")
            self.notify(
                f"The engine exited with code {code} without a readable report. See Console.",
                True,
            )
            return
        try:
            report = read_report(self.pending_report)
        except Exception as error:
            self._calibration_done(False, "Invalid report")
            self.notify(str(error), True)
            return
        self._show_report(report, self.pending_report)
        verdict = report["verdict"].get("state", "UNKNOWN")
        ok = code == 0 and verdict == "DONE"
        self._set_stage("DONE" if ok else "ABORT", verdict.title())
        self._calibration_done(
            ok, f"{verdict.title()} · report {Path(self.pending_report).name}"
        )
        self.navigate(PAGE_RESULTS)
        if not ok:
            self.notify(
                f"Calibration ended with {verdict}. The results explain why.", True
            )

    def toggle_capture(self):
        if self.capture_active:
            self.stop_capture()
            return
        if self.source is None or self.job_active:
            return
        folder = Path(self.calibration_page.output_dir.text()).expanduser()
        default = folder / (
            "capture_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".bin"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Record camera + IMU session",
            str(default),
            "ov_zcalib session (*.bin)",
        )
        if path:
            self.start_capture(path)

    def start_capture(self, path):
        if self.source is None or self.capture_active or self.job_active:
            return
        try:
            program, args = capture_command(
                find_executable("ov_zcalib_capture"), self.source_settings, path
            )
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            self.notify(str(error), True)
            return
        self.record_path = str(path)
        self.capture_active = True
        self.capture_button.setText("Stop recording")
        self.capture_button.setIcon(icon("stop", AMBER))
        self.connect_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.log("\n[capture] " + shlex.join([program] + args) + "\n")
        self.recorder.start(program, args)

    def stop_capture(self):
        if not self.capture_active or self.capture_finishing:
            return
        self.capture_finishing = True
        self.capture_button.setText("Finishing recording…")
        self.capture_button.setEnabled(False)
        self.recorder.terminate()
        pid = self.recorder.processId()
        QTimer.singleShot(
            5000,
            lambda: (
                self.recorder.kill()
                if self.recorder.state() != QProcess.NotRunning
                and self.recorder.processId() == pid
                else None
            ),
        )

    def _capture_reset(self):
        self.capture_active = False
        self.capture_finishing = False
        self.capture_button.setText("Record session")
        self.capture_button.setIcon(icon("record"))
        self.connect_button.setEnabled(True)
        self.run_button.setEnabled(not self.job_active)
        self.open_button.setEnabled(True)

    def _capture_error(self, error):
        if error == QProcess.FailedToStart:
            self._capture_reset()
            self.notify(self.recorder.errorString(), True)

    def _capture_finished(self, code, status):
        self._read_process(self.recorder, "capture")
        self._capture_reset()
        if self.closing:
            return
        if (
            code == 0
            and status == QProcess.NormalExit
            and Path(self.record_path).is_file()
        ):
            try:
                # Companion provenance keeps topic and exposure assumptions inspectable.
                sidecar = Path(self.record_path + ".capture.json")
                with sidecar.open("x", encoding="utf-8") as stream:
                    json.dump(
                        {
                            "tracker": "TrackKLT",
                            "source": asdict(self.source_settings),
                            "temperature_available": False,
                            "thermal_checks_evaluated": False,
                            "temperature_encoding": "legacy_zero_placeholder",
                        },
                        stream,
                        indent=2,
                    )
            except OSError as error:
                self.notify(
                    f"Recording saved; could not save capture metadata: {error}", True
                )
            self.open_path(self.record_path)
        else:
            self.notify(
                f"Capture exited with code {code}. The recording is not marked successful; inspect Console.",
                True,
            )

    def _show_report(self, report, path):
        self.report, self.report_path = report, path
        verdict = report.get("verdict", {})
        blocks = report.get("blocks", [])
        committed = sum(bool(block.get("committed")) for block in blocks)
        self.result_summary.setText(
            f"{verdict.get('state', 'UNKNOWN')}  /  "
            + (
                f"{committed} of {len(blocks)} blocks committed"
                if blocks
                else "No blocks evaluated"
            )
        )
        verify = report.get("verify", {})
        improve = verify.get("mixture_improve")
        improvement = (
            f"{improve * 100:.1f}%" if isinstance(improve, (int, float)) else "—"
        )
        if not verify.get("windows_used", 0):
            improvement = "not evaluated"
        detail = f"Held-out mixture improvement: {improvement}   ·   {verify.get('windows_used', 0)} verification windows   ·   {Path(path).name}"
        if verdict.get("abort_reason"):
            detail += "\n" + str(verdict["abort_reason"])
        if report.get("session", {}).get("temperature_available") is False:
            detail += "\nTemperature unavailable; thermal consistency was not assessed."
        if report.get("tainted"):
            detail += "\nEngine reports truncated inner solves; inspect full evidence."
        self.result_detail.setText(detail)
        self.blocks_table.setRowCount(len(blocks))
        for index, block in enumerate(blocks):
            decision = "COMMITTED" if block.get("committed") else "SEED RETAINED"
            reason = (
                "Not estimated"
                if block.get("not_estimated")
                else "Atomic pairing reverted"
                if block.get("atomic_reverted")
                else "Precision ceiling missed"
                if not block.get("ceiling_ok", True)
                else "Engine gate decision"
            )
            values = [
                block.get("label", block.get("name", "")),
                str(block.get("cam", -1)) if block.get("cam", -1) >= 0 else "IMU",
                decision,
                number(block.get("worst_ratio")),
                number(block.get("worst_sigma")),
                reason,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 2:
                    item.setForeground(
                        QColor(TEAL if block.get("committed") else AMBER)
                    )
                self.blocks_table.setItem(index, column, item)
        posterior = report.get("posterior", {})
        labels = posterior.get("labels", [])
        sigmas = posterior.get("sigma", [])
        priors = posterior.get("prior_sigma", [])
        self.posterior_table.setRowCount(len(labels))
        for index, name in enumerate(labels):
            sigma = sigmas[index] if index < len(sigmas) else None
            prior = priors[index] if index < len(priors) else None
            ratio = (
                sigma / prior
                if isinstance(sigma, (int, float))
                and isinstance(prior, (int, float))
                and prior > 0
                else None
            )
            for column, value in enumerate(
                (name, number(sigma), number(prior), number(ratio))
            ):
                self.posterior_table.setItem(index, column, QTableWidgetItem(value))
        values = report.get("values", {}).get("committed", {})
        self.values_tree.clear()
        self._tree(self.values_tree.invisibleRootItem(), values)
        self.values_tree.expandToDepth(1)
        cameras = []
        for data in values.get("cams", []):
            cameras.append(
                Camera(
                    width=data.get("img_w", 640),
                    height=data.get("img_h", 480),
                    q=data.get("q_ItoC", [0, 0, 0, 1]),
                    p=data.get("p_IinC", [0, 0, 0]),
                    td=data.get("timeshift_cam_imu", 0),
                    tr=data.get("t_readout", 0),
                    k=data.get("cam_k", [450, 450, 320, 240]),
                    d=data.get("cam_d", [0] * 4),
                    fisheye=data.get("fisheye", False),
                )
            )
        self.result_rig.set_cameras(cameras, "Committed geometry")
        self.raw_report.setPlainText(json.dumps(report, indent=2, ensure_ascii=False))
        self.log(f"[report] Loaded {path}\n")

    def _tree(self, parent, values):
        entries = values.items() if isinstance(values, dict) else enumerate(values)
        for key, value in entries:
            if isinstance(value, dict) or (
                isinstance(value, list) and value and isinstance(value[0], dict)
            ):
                child = QTreeWidgetItem(parent, [str(key), ""])
                self._tree(child, value)
            else:
                display = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, bool))
                    else number(value, 8)
                    if isinstance(value, (int, float))
                    else str(value)
                    if value is not None
                    else "Unavailable"
                )
                child = QTreeWidgetItem(parent, [str(key), display])
                child.setToolTip(1, display)

    def export_report(self):
        if self.report is None:
            self.notify("Open or generate a report first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export calibration evidence",
            "calibration.report.json",
            "JSON report (*.json)",
        )
        if path:
            try:
                Path(path).write_text(
                    json.dumps(self.report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                self.notify(f"Exported {path}.")
            except OSError as error:
                self.notify(str(error), True)

    def open_output_folder(self):
        path = (
            Path(self.report_path).parent
            if self.report_path
            else Path(self.calibration_page.output_dir.text()).expanduser()
        )
        if path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            self.notify("The output folder has not been created yet.")

    def save_view(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save sensor view", "sensor-view.png", "PNG image (*.png)"
        )
        if path:
            view = (
                self.feature_view
                if self.view_tabs.currentIndex() == 0
                else self.rig_view
            )
            if view.grab().save(path):
                self.notify(f"Saved {path}.")
            else:
                self.notify("Could not save this view.", True)

    def save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session log", "calibration.log", "Log files (*.log)"
        )
        if path:
            try:
                Path(path).write_text(self.console.toPlainText(), encoding="utf-8")
            except OSError as error:
                self.notify(str(error), True)

    def closeEvent(self, event):
        active = (
            self.job_active
            or self.capture_active
            or (self.source is not None and self.source.isRunning())
            or (self.loader is not None and self.loader.isRunning())
            or (self.log_loader is not None and self.log_loader.isRunning())
            or any(worker.isRunning() for worker in self.workers)
        )
        self.vio_page.shutdown()
        if active:
            self.closing = True
            if self.job_active:
                self.cancel_calibration()
            if self.capture_active:
                self.stop_capture()
            if self.source is not None:
                self.source.requestInterruption()
            if self.loader is not None:
                self.loader.requestInterruption()
            if self.log_loader is not None:
                self.log_loader.requestInterruption()
            for worker in self.workers:
                worker.requestInterruption()
            event.ignore()
        else:
            self.timer.stop()
            self._close_log()
            self.preferences.sync()
            event.accept()

    def _try_close(self):
        if (
            self.solver.state() != QProcess.NotRunning
            or self.recorder.state() != QProcess.NotRunning
        ):
            return
        if self.source is not None and self.source.isRunning():
            return
        if self.loader is not None and self.loader.isRunning():
            return
        if self.log_loader is not None and self.log_loader.isRunning():
            return
        if any(worker.isRunning() for worker in self.workers):
            return
        self.job_active = self.capture_active = False
        self.close()
