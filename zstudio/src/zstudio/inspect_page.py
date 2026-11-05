"""The studio's Inspect page: channel health, whole-log timeline, plots and findings."""

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QMenu,
    QStackedWidget,
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .checks import LOG_CHANNEL
from .logreport import count_text
from .logviews import SERIES_COLORS, SEVERITY_COLORS, LogPlot, TimelineView, TimeRange
from .widgets import (
    CameraGrid,
    EmptyState,
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
    label,
    pill,
    row,
    set_pill_color,
)

STATUS_TEXT = {
    "ok": "OK",
    "problem": "Problem",
    "warning": "Warning",
    "info": "Info",
    "expected": "Expected",
}
VERDICT = {
    "pass": ("PASS", TEAL),
    "warnings": ("WARNINGS", AMBER),
    "problems": ("PROBLEMS", SEVERITY_COLORS["problem"]),
}


def plot_options(log, inspection):
    """[(label, channel, builder)] where builder() -> (title, units, series, markers)."""
    options = []

    def markers(name):
        return [
            (e.time, f.severity)
            for f in inspection.for_channel(name)
            for e in f.events[:2000]
        ]

    for imu in log.imus:
        t, d = imu.times, imu.data

        def gyro(imu=imu, t=t, d=d):
            return (
                f"{imu.name} · angular velocity",
                "rad/s",
                [
                    (a, c, t, d[:, k])
                    for a, c, k in zip(("x", "y", "z"), SERIES_COLORS, (1, 2, 3))
                ],
                markers(imu.name),
            )

        def accel(imu=imu, t=t, d=d):
            series = [
                (a, c, t, d[:, k])
                for a, c, k in zip(("x", "y", "z"), SERIES_COLORS, (4, 5, 6))
            ]
            series.append(("|a|", TEXT, t, np.linalg.norm(d[:, 4:7], axis=1)))
            return (
                f"{imu.name} · linear acceleration",
                "m/s²",
                series,
                markers(imu.name),
            )

        def interval(imu=imu, t=t):
            return (
                f"{imu.name} · sample interval",
                "ms",
                [("dt", TEAL, t[1:], np.diff(t) * 1e3)],
                markers(imu.name),
            )

        options += [
            (f"{imu.name} · angular velocity", imu.name, gyro),
            (f"{imu.name} · linear acceleration", imu.name, accel),
            (f"{imu.name} · sample interval", imu.name, interval),
        ]
        if imu.temperature_available:
            options.append(
                (
                    f"{imu.name} · temperature",
                    imu.name,
                    lambda imu=imu: (
                        f"{imu.name} · temperature",
                        "°C",
                        [("T", AMBER, imu.times, imu.data[:, 7])],
                        markers(imu.name),
                    ),
                )
            )
        if imu.receive is not None:
            options.append(
                (
                    f"{imu.name} · bag latency",
                    imu.name,
                    lambda imu=imu: (
                        f"{imu.name} · bag time − header stamp",
                        "ms",
                        [("latency", BLUE, imu.times, (imu.receive - imu.times) * 1e3)],
                        markers(imu.name),
                    ),
                )
            )
    for cam in log.cameras:
        t = cam.times
        options.append(
            (
                f"{cam.name} · frame interval",
                cam.name,
                lambda cam=cam, t=t: (
                    f"{cam.name} · frame interval",
                    "ms",
                    [("dt", TEAL, t[1:], np.diff(t) * 1e3)],
                    markers(cam.name),
                ),
            )
        )
        if (
            cam.exposure is not None
            and len(cam.exposure)
            and np.nanmax(cam.exposure) > 0
        ):
            options.append(
                (
                    f"{cam.name} · exposure",
                    cam.name,
                    lambda cam=cam: (
                        f"{cam.name} · exposure",
                        "ms",
                        [("exposure", AMBER, cam.times, cam.exposure * 1e3)],
                        markers(cam.name),
                    ),
                )
            )
        if cam.gain is not None and len(cam.gain) and np.nanmax(cam.gain) > 0:
            options.append(
                (
                    f"{cam.name} · gain",
                    cam.name,
                    lambda cam=cam: (
                        f"{cam.name} · gain",
                        "",
                        [("gain", BLUE, cam.times, cam.gain)],
                        markers(cam.name),
                    ),
                )
            )
        if (
            cam.frame_ids is not None
            and len(cam.frame_ids) > 1
            and len(np.unique(cam.frame_ids)) > 1
        ):
            options.append(
                (
                    f"{cam.name} · id step",
                    cam.name,
                    lambda cam=cam: (
                        f"{cam.name} · frame counter step",
                        "",
                        [
                            (
                                "Δid",
                                BLUE,
                                cam.times[1:],
                                np.diff(cam.frame_ids).astype(float),
                            )
                        ],
                        markers(cam.name),
                    ),
                )
            )
        if cam.receive is not None:
            options.append(
                (
                    f"{cam.name} · bag latency",
                    cam.name,
                    lambda cam=cam: (
                        f"{cam.name} · bag time − header stamp",
                        "ms",
                        [("latency", BLUE, cam.times, (cam.receive - cam.times) * 1e3)],
                        markers(cam.name),
                    ),
                )
            )
    return options


class InspectPage(QWidget):
    seek = pyqtSignal(float)  # stream time [s]
    channels_changed = pyqtSignal(list)  # included channel names
    action = pyqtSignal(
        str
    )  # export | convert | vio | stereo | calibrate | channels | open

    def __init__(self, parent=None):
        super().__init__(parent)
        self.log = None
        self.inspection = None
        self.options = []
        self.range = TimeRange(self)
        self.range.seek.connect(self.seek)
        self._updating = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        outer.addWidget(self.stack)
        self.empty = EmptyState(
            "inspect",
            "No log open",
            "Open a voxl-logger folder or a ROS 1 / ROS 2 bag. Every pipe and topic is checked for data loss, "
            "timing and configuration problems.",
            lambda: self.action.emit("open"),
            "Open…",
        )
        self.stack.addWidget(self.empty)
        content = QWidget()
        self.stack.addWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        head = Card(margins=14)
        self.verdict = pill("NO LOG", MUTED)
        self.title = label("Open a voxl-logger directory or a ROS bag", 13, bold=True)
        self.summary = label(
            "Inspection checks skipped samples, timing, files, intrinsics and the rig configuration.",
            9,
            MUTED,
        )
        self.summary.setWordWrap(True)
        more = button("More", None, None)
        menu = QMenu(more)
        stereo = menu.addAction("Stereo check")
        stereo.setToolTip(
            "Recover the stereo pair's rotation and baseline direction from the images and compare with the rig"
        )
        stereo.triggered.connect(lambda: self.action.emit("stereo"))
        export = menu.addAction("Export report…")
        export.triggered.connect(lambda: self.action.emit("export"))
        menu.addSeparator()
        calibrate = menu.addAction("Calibrate these cameras…")
        calibrate.triggered.connect(lambda: self.action.emit("convert"))
        more.setMenu(menu)
        self.buttons = {
            "channels": button(
                "Channels…", lambda: self.action.emit("channels"), "tune"
            ),
            "more": more,
            "vio": button(
                "Run VIO", lambda: self.action.emit("vio"), "trajectory", True
            ),
            "stereo": stereo,
            "export": export,
            "convert": calibrate,
        }
        titles = QVBoxLayout()
        titles.setSpacing(3)
        titles.addLayout(row(self.verdict, self.title, None))
        titles.addWidget(self.summary)
        head.box.addLayout(
            row(
                titles,
                None,
                self.buttons["channels"],
                self.buttons["more"],
                self.buttons["vio"],
            )
        )
        layout.addWidget(head)
        split = QSplitter(Qt.Vertical)
        split.setHandleWidth(10)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "Inspect",
                "Channel",
                "Kind",
                "Rate",
                "Count",
                "Status",
                "First problem or warning",
            ]
        )
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.table.horizontalHeader()
        for column in range(6):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        self.table.itemChanged.connect(self._item_changed)
        self.table.itemSelectionChanged.connect(self._row_selected)
        split.addWidget(self.table)
        self.transport = TransportBar()
        transport = Card(margins=10)
        transport.box.addLayout(
            row(
                self.transport,
                label("click the timeline or a finding to seek", 8, MUTED),
            )
        )
        layout.insertWidget(1, transport)
        self.timeline = TimelineView(self.range)
        self.timeline.lane_clicked.connect(self.select_channel)
        lanes = QScrollArea()
        lanes.setWidgetResizable(True)
        lanes.setWidget(self.timeline)
        lanes.setMinimumHeight(110)
        split.addWidget(lanes)
        bottom = QSplitter(Qt.Horizontal)
        bottom.setHandleWidth(10)
        plots = Card(margins=8)
        self.plot_box = QComboBox()
        self.plot_box.currentIndexChanged.connect(self._plot_changed)
        reset = button("", self.range.reset, "reset")
        reset.setToolTip("Show the whole log (double-click a plot also resets)")
        reset.setFixedWidth(37)
        plots.box.addLayout(
            row(
                label("Plot", 9, MUTED),
                self.plot_box,
                None,
                label("wheel: zoom · drag: pan · click: seek", 8, MUTED),
                reset,
            )
        )
        self.plot = LogPlot(self.range)
        plots.box.addWidget(self.plot, 1)
        bottom.addWidget(plots)
        side = QSplitter(Qt.Vertical)
        preview = Card(margins=8)
        self.preview_label = label("Preview", 9, MUTED)
        self.rotate_left = button("⟲", lambda: self.action.emit("rotate-ccw"))
        self.rotate_left.setToolTip(
            "Rotate this camera 90° counter-clockwise (frames, intrinsics and extrinsics)"
        )
        self.rotate_right = button("⟳", lambda: self.action.emit("rotate-cw"))
        self.rotate_right.setToolTip(
            "Rotate this camera 90° clockwise (frames, intrinsics and extrinsics)"
        )
        for widget in (self.rotate_left, self.rotate_right):
            widget.setFixedWidth(38)
        self.orient_button = button(
            "Orientation…", lambda: self.action.emit("orientation"), "camera"
        )
        self.orient_button.setToolTip(
            "Rotate the image and edit this camera's extrinsics, then choose whether to save"
        )
        preview.box.addLayout(
            row(
                self.preview_label,
                None,
                self.rotate_left,
                self.rotate_right,
                self.orient_button,
            )
        )
        self.preview_switch = Segmented(
            (("Selected", "selected"), ("All cameras", "all"))
        )
        self.preview_switch.changed.connect(self._preview_mode_changed)
        preview.box.addWidget(self.preview_switch, 0, Qt.AlignLeft)
        self.preview_stack = QStackedWidget()
        self.preview = FeatureView()
        self.preview.show_grid = False
        self.preview.captions = ("Log image at the playhead", "log")
        self.preview.setMinimumSize(260, 170)
        self.preview_stack.addWidget(self.preview)
        self.preview_grid = CameraGrid()
        self.preview_stack.addWidget(self.preview_grid)
        preview.box.addWidget(self.preview_stack, 1)
        self.unsaved = QWidget()
        self.unsaved_label = label("", 9, AMBER)
        self.unsaved_label.setWordWrap(True)
        self.save_orientation = button(
            "Save…", lambda: self.action.emit("orientation-save"), "download", True
        )
        self.revert_orientation = button(
            "Revert", lambda: self.action.emit("orientation-revert"), "reset"
        )
        unsaved = row(
            self.unsaved_label, None, self.revert_orientation, self.save_orientation
        )
        unsaved.setContentsMargins(0, 0, 0, 0)
        self.unsaved.setLayout(unsaved)
        self.unsaved.hide()
        preview.box.addWidget(self.unsaved)
        side.addWidget(preview)
        self.findings = QTreeWidget()
        self.findings.setHeaderLabels(["Severity", "Channel", "Finding", "t [s]"])
        self.findings.setAlternatingRowColors(True)
        self.findings.setUniformRowHeights(True)
        self.findings.itemClicked.connect(self._finding_clicked)
        fh = self.findings.header()
        fh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        fh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        fh.setSectionResizeMode(2, QHeaderView.Stretch)
        fh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        side.addWidget(self.findings)
        side.setSizes([260, 300])
        bottom.addWidget(side)
        bottom.setSizes([780, 420])
        split.addWidget(bottom)
        split.setSizes([170, 160, 420])
        layout.addWidget(split, 1)
        self.preview_camera = 0
        self.stack.setCurrentIndex(0)
        self.clear()

    # -- state ---------------------------------------------------------------
    def clear(self, message="Open a voxl-logger directory or a ROS bag"):
        self.log, self.inspection, self.options = None, None, []
        self._updating = True
        self.table.setRowCount(0)
        self._updating = False
        self.findings.clear()
        self.timeline.set_lanes([])
        self.plot_box.clear()
        self.plot.clear("No log open")
        self.preview.set_frame(None, None)
        self.verdict.setText("NO LOG")
        set_pill_color(self.verdict, MUTED)
        self.title.setText(message)
        self.summary.setText(
            "Inspection checks skipped samples, timing, files, intrinsics and the rig configuration."
        )
        self.stack.setCurrentIndex(0)
        for widget in self.buttons.values():
            widget.setEnabled(False)
        for widget in (self.rotate_left, self.rotate_right, self.orient_button):
            widget.setEnabled(False)
        self.unsaved.hide()

    def set_inspection(self, log, inspection):
        self.log, self.inspection = log, inspection
        text, color = VERDICT[inspection.verdict]
        self.verdict.setText(text)
        set_pill_color(self.verdict, color)
        self.title.setText(f"{log.name} · {log.format_label}")
        counts = count_text(inspection) or "no findings"
        self.summary.setText(
            f"{log.duration:.2f} s · {len(log.imus)} IMU · {len(log.cameras)} camera{'s' if len(log.cameras) != 1 else ''} · {len(log.included)} of {len(log.channels)} channels inspected · {counts}\n{log.path}"
        )
        for key, widget in self.buttons.items():
            widget.setEnabled(True)
        self.buttons["stereo"].setEnabled(len(log.cameras) >= 2)
        for widget in (self.rotate_left, self.rotate_right, self.orient_button):
            widget.setEnabled(bool(log.cameras))
        self.buttons["convert"].setEnabled(bool(log.imus and log.cameras))
        self.stack.setCurrentIndex(1)
        self.preview_grid.set_cameras(
            [c.camera or c.name for c in log.cameras], ("", "log")
        )
        self.buttons["vio"].setEnabled(bool(log.imus and log.cameras))
        self._fill_table()
        self._fill_findings()
        self._fill_timeline()
        self.options = plot_options(log, inspection)
        self.plot_box.blockSignals(True)
        self.plot_box.clear()
        for text, *_ in self.options:
            self.plot_box.addItem(text)
        self.plot_box.blockSignals(False)
        self.range.set_bounds(log.start, max(log.end, log.start + 1e-3))
        self._plot_changed(0)
        if log.cameras:
            self.preview_camera = min(self.preview_camera, len(log.cameras) - 1)
            cam = log.cameras[self.preview_camera]
            self.preview.title = cam.camera or cam.name
            self.preview_label.setText(f"Preview · {cam.label}")

    def _fill_table(self):
        log, inspection = self.log, self.inspection
        reports = {r.name: r for r in inspection.channels}
        self._updating = True
        self.table.setRowCount(len(log.channels))
        for i, channel in enumerate(log.channels):
            included = channel.name in log.included
            check = QTableWidgetItem()
            check.setFlags(
                Qt.ItemIsUserCheckable
                | Qt.ItemIsSelectable
                | (Qt.ItemIsEnabled if channel.supported else Qt.NoItemFlags)
            )
            check.setCheckState(Qt.Checked if included else Qt.Unchecked)
            check.setData(Qt.UserRole, channel.name)
            if not channel.supported:
                check.setToolTip(channel.reason)
            report = reports.get(channel.name)
            status = inspection.status(channel.name) if included else "excluded"
            found = inspection.for_channel(channel.name)
            first = next(
                (f.message for f in found if f.severity in ("problem", "warning")), ""
            )
            values = [
                channel.name,
                {
                    "imu": "IMU",
                    "camera": "Camera",
                    "camera_info": "CameraInfo",
                    "tf": "Static TF",
                }.get(channel.kind, channel.type),
                f"{report.rate:.2f} Hz" if report and report.rate else "",
                f"{report.count:,}" if report else f"{channel.count:,}",
                STATUS_TEXT.get(
                    status, "Excluded" if channel.supported else "Not inspected"
                ),
                first or (channel.reason if not channel.supported else ""),
            ]
            self.table.setItem(i, 0, check)
            for column, value in enumerate(values, 1):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, channel.name)
                if column == 5:
                    item.setForeground(QColor(SEVERITY_COLORS.get(status, MUTED)))
                elif not included:
                    item.setForeground(QColor(MUTED))
                if column == 6 and first:
                    item.setToolTip(first)
                self.table.setItem(i, column, item)
        self._updating = False

    def _fill_findings(self):
        self.findings.clear()
        for finding in self.inspection.findings:
            color = QColor(SEVERITY_COLORS.get(finding.severity, MUTED))
            top = QTreeWidgetItem(
                [
                    finding.severity,
                    finding.channel or "log",
                    finding.message,
                    "" if finding.time is None else f"{finding.time:.3f}",
                ]
            )
            top.setForeground(0, color)
            top.setToolTip(2, finding.message)
            top.setData(0, Qt.UserRole, (finding.channel, finding.time))
            for event in finding.events[:1000]:
                child = QTreeWidgetItem(
                    [
                        "",
                        "",
                        event.text
                        + (f" (row {event.row})" if event.row is not None else ""),
                        f"{event.time:.3f}",
                    ]
                )
                child.setData(0, Qt.UserRole, (finding.channel, event.time))
                top.addChild(child)
            if len(finding.events) > 1000:
                top.addChild(
                    QTreeWidgetItem(
                        ["", "", f"… {len(finding.events) - 1000} more events", ""]
                    )
                )
            self.findings.addTopLevelItem(top)

    def _fill_timeline(self):
        log, inspection = self.log, self.inspection
        lanes = []
        for stream in log.streams:
            events = [
                (e.time, f.severity, f.message if len(f.events) == 1 else e.text)
                for f in inspection.for_channel(stream.name)
                for e in f.events
            ]
            lanes.append(
                {
                    "name": stream.name,
                    "label": f"{'IMU' if stream.kind == 'imu' else 'CAM'} {stream.name}",
                    "start": float(stream.times[0]) if len(stream.times) else None,
                    "end": float(stream.times[-1]) if len(stream.times) else None,
                    "status": inspection.status(stream.name),
                    "events": events,
                    "tooltip": "\n".join(
                        f.message for f in inspection.for_channel(stream.name)
                    )
                    or "no findings",
                }
            )
        config = [
            (e.time, f.severity, f.message)
            for f in inspection.for_channel(LOG_CHANNEL)
            for e in f.events
        ]
        if inspection.for_channel(LOG_CHANNEL):
            lanes.append(
                {
                    "name": LOG_CHANNEL,
                    "label": "LOG configuration",
                    "start": None,
                    "end": None,
                    "status": inspection.status(LOG_CHANNEL),
                    "events": config,
                    "tooltip": "\n".join(
                        f.message for f in inspection.for_channel(LOG_CHANNEL)
                    ),
                }
            )
        self.timeline.set_lanes(lanes)

    # -- interaction -----------------------------------------------------------
    def set_playhead(self, t):
        self.range.set_playhead(t)

    def select_channel(self, name):
        self.timeline.select(name)
        for i, (text, channel, _) in enumerate(self.options):
            if channel == name:
                if (
                    self.plot_box.currentIndex() != i
                    and not self.options[self.plot_box.currentIndex()][1] == name
                ):
                    self.plot_box.setCurrentIndex(i)
                break
        if self.log is not None:
            names = [c.name for c in self.log.cameras]
            if name in names:
                self.preview_camera = names.index(name)
                cam = self.log.cameras[self.preview_camera]
                self.preview_label.setText(f"Preview · {cam.label}")
                self.preview.title = cam.camera or cam.name
                self.action.emit("preview")
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 1)
            if (
                item is not None
                and item.data(Qt.UserRole) == name
                and self.table.currentRow() != r
            ):
                self.table.blockSignals(True)
                self.table.selectRow(r)
                self.table.blockSignals(False)

    def preview_mode(self):
        return self.preview_switch.value()

    def _preview_mode_changed(self, mode):
        self.preview_stack.setCurrentIndex(1 if mode == "all" else 0)
        self.action.emit("preview")

    def set_orientation_state(self, changes):
        """Show the unsaved-orientation bar for (rotations, edges) from LogSource.orientation_changes()."""
        rotations, edges = changes
        parts = [
            f"camera {name} rotated {now}°"
            if now
            else f"camera {name} rotation removed"
            for name, (_, now) in sorted(rotations.items())
        ]
        parts += [f"extrinsics of {child}" for child in sorted(edges)]
        self.unsaved_label.setText("Unsaved: " + ", ".join(parts) if parts else "")
        self.unsaved.setVisible(bool(parts))

    def selected_camera(self):
        if self.log is None or not self.log.cameras:
            return None
        return self.log.cameras[min(self.preview_camera, len(self.log.cameras) - 1)]

    def _row_selected(self):
        items = self.table.selectedItems()
        if items:
            name = items[0].data(Qt.UserRole)
            if name and self.log is not None and name in self.log.included:
                self.select_channel(name)

    def _item_changed(self, item):
        if self._updating or item.column() != 0 or self.log is None:
            return
        names = [
            self.table.item(r, 0).data(Qt.UserRole)
            for r in range(self.table.rowCount())
            if self.table.item(r, 0).checkState() == Qt.Checked
        ]
        if not names:
            self._updating = True
            item.setCheckState(Qt.Checked)
            self._updating = False
            return
        self.channels_changed.emit(names)

    def _plot_changed(self, index):
        if not self.options or index < 0:
            self.plot.clear("No plottable channel included")
            return
        title, units, series, markers = self.options[index][2]()
        self.plot.set_series(title, units, series, markers)

    def _finding_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        channel, t = data
        if channel:
            self.select_channel(channel)
        if t is not None:
            span = self.range.t1 - self.range.t0
            if not self.range.t0 <= t <= self.range.t1:
                self.range.set_view(t - span / 2, t + span / 2)
            self.seek.emit(float(t))
