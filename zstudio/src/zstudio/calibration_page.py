"""The Calibrate page: what to calibrate, how to solve it, where to write it.

One primary action. The source is the open log (its checked cameras; frames
are tracked with TrackKLT into a session first, or the engine's VOXL ingest is
used for a single VOXL camera) or a recorded session file. A readiness list
shows what is missing before anything runs; the run panel follows the engine's
own stages and output.
"""

import os
import time
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .engine import find_executable
from .widgets import (
    AMBER,
    GREEN,
    MUTED,
    RED,
    TEXT,
    Card,
    Segmented,
    StageView,
    button,
    font,
    label,
    plain_button,
    row,
)

PROFILES = (
    ("Inherit", "record"),
    ("Library", "library"),
    ("VOXL bench", "voxl"),
    ("VOXL flight", "flight"),
)
INTRINSICS = (("Fixed", 0), ("Refine", 1), ("Full", 2))


def _check(ok, text, hint=False):
    mark, color = ("✓", GREEN) if ok else (("•", AMBER) if hint else ("✕", RED))
    widget = label(f"{mark}  {text}", 9.5, TEXT if ok else color)
    widget.setWordWrap(True)
    return widget


class CalibrationPage(QWidget):
    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    show_results = pyqtSignal()
    open_folder = pyqtSignal()
    help_requested = pyqtSignal(str)

    def __init__(self, preferences, parent=None):
        super().__init__(parent)
        self.preferences = preferences
        self.log = None
        self.session_path = ""
        self.camera_boxes = []
        self.readouts = {}
        self.started = None
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(16)
        form_host = QWidget()
        form = QVBoxLayout(form_host)
        form.setContentsMargins(0, 0, 4, 0)
        form.setSpacing(14)

        # 1 · Data -----------------------------------------------------------
        data = Card()
        data.box.addLayout(
            row(
                label("Data", 13, bold=True),
                None,
            )
        )
        self.source = Segmented((("Open log", "log"), ("Session file", "session")))
        self.source.changed.connect(lambda *_: self._refresh())
        data.box.addWidget(self.source)
        self.log_box = QWidget()
        log_layout = QVBoxLayout(self.log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(6)
        self.log_caption = label("", 9, MUTED)
        self.log_caption.setWordWrap(True)
        log_layout.addWidget(self.log_caption)
        self.camera_list = QVBoxLayout()
        self.camera_list.setSpacing(4)
        log_layout.addLayout(self.camera_list)
        imu_row = QFormLayout()
        self.imu = QComboBox()
        self.imu.currentIndexChanged.connect(lambda *_: self._refresh())
        imu_row.addRow("IMU", self.imu)
        log_layout.addLayout(imu_row)
        data.box.addWidget(self.log_box)
        self.session_box = QWidget()
        session_layout = QVBoxLayout(self.session_box)
        session_layout.setContentsMargins(0, 0, 0, 0)
        self.session_edit = QLineEdit()
        self.session_edit.setPlaceholderText(
            "session.bin recorded by the studio, ov_zcalib_track or the device"
        )
        self.session_edit.textChanged.connect(lambda *_: self._refresh())
        session_layout.addLayout(
            row(self.session_edit, button("Choose…", self._choose_session))
        )
        data.box.addWidget(self.session_box)
        form.addWidget(data)

        # 2 · Solve ------------------------------------------------------------
        solve = Card()
        solve.box.addWidget(label("Solve", 13, bold=True))
        grid = QFormLayout()
        grid.setSpacing(10)
        self.profile = Segmented(PROFILES)
        grid.addRow("Profile", self.profile)
        self.intrinsics = Segmented(INTRINSICS)
        self.intrinsics.set_value(1)
        self.intrinsics.setToolTip(
            "Fixed: keep the camera calibration · Refine: with its priors · Full: estimate from scratch"
        )
        grid.addRow("Intrinsics", self.intrinsics)
        self.budget = QDoubleSpinBox()
        self.budget.setRange(0, 7200)
        self.budget.setDecimals(0)
        self.budget.setSuffix(" s")
        self.budget.setValue(60.0)
        self.budget.setSpecialValueText("Unlimited")
        self.budget.setToolTip(
            "Time allowed for refinement; held-out verification is timed separately"
        )
        grid.addRow("Solve budget", self.budget)
        solve.box.addLayout(grid)
        self.advanced_toggle = plain_button(
            "Show advanced options", self._toggle_advanced
        )
        solve.box.addWidget(self.advanced_toggle, 0, Qt.AlignLeft)
        self.advanced = QWidget()
        adv = QFormLayout(self.advanced)
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setSpacing(10)
        self.voxl_ingest = QCheckBox(
            "Use the engine's VOXL ingest (ov_zcalibrate --voxl)"
        )
        self.voxl_ingest.setToolTip(
            "Single VOXL camera only: the engine reads the log itself, exactly like the device tools"
        )
        self.voxl_ingest.toggled.connect(lambda *_: self._refresh())
        adv.addRow("", self.voxl_ingest)
        self.features = QSpinBox()
        self.features.setRange(20, 2000)
        self.features.setValue(200)
        adv.addRow("Features per camera", self.features)
        self.readout_box = QFormLayout()
        adv.addRow("Readout (rolling shutter)", self._wrap(self.readout_box))
        self.threads = QSpinBox()
        self.threads.setRange(1, max(1, os.cpu_count() or 1))
        self.threads.setValue(min(2, self.threads.maximum()))
        adv.addRow("Worker threads", self.threads)
        self.binary = QLineEdit(
            preferences.value("binary", find_executable("ov_zcalibrate"))
        )
        self.binary.setPlaceholderText("ov_zcalibrate")
        adv.addRow("Engine", row(self.binary, button("Choose…", self._choose_binary)))
        self.advanced.hide()
        solve.box.addWidget(self.advanced)
        form.addWidget(solve)

        # 3 · Output -----------------------------------------------------------
        output = Card()
        output.box.addWidget(label("Output", 13, bold=True))
        self.output_dir = QLineEdit(
            preferences.value("output_dir", str(Path.home() / "ov_zcalib_sessions"))
        )
        output.box.addLayout(
            row(self.output_dir, button("Choose…", self._choose_output))
        )
        note = label(
            "Every run writes a new YAML and report; existing files are never overwritten.",
            9,
            MUTED,
        )
        note.setWordWrap(True)
        output.box.addWidget(note)
        form.addWidget(output)
        form.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_host)
        layout.addWidget(scroll, 0, 0)

        # Readiness + run --------------------------------------------------------
        side = Card()
        side.setFixedWidth(360)
        side.box.addWidget(label("Ready to calibrate", 13, bold=True))
        self.checks = QVBoxLayout()
        self.checks.setSpacing(5)
        side.box.addLayout(self.checks)
        self.run_button = button("Calibrate", self.run_requested.emit, "play", True)
        self.cancel_button = button("Cancel", self.cancel_requested.emit, "stop")
        self.cancel_button.hide()
        side.box.addLayout(row(self.run_button, self.cancel_button, None))
        side.box.addSpacing(6)
        self.status = label("Not started", 10, MUTED)
        self.status.setWordWrap(True)
        side.box.addWidget(self.status)
        self.stages = StageView()
        side.box.addWidget(self.stages)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(font(8, mono=True))
        self.console.setMaximumBlockCount(200)
        self.console.setMinimumHeight(110)
        side.box.addWidget(self.console, 1)
        self.results_button = button("View results", self.show_results.emit, "results")
        self.folder_button = plain_button("Open output folder", self.open_folder.emit)
        self.results_button.hide()
        side.box.addLayout(row(self.results_button, None, self.folder_button))
        layout.addWidget(side, 0, 1)
        self.clock = QTimer(self)
        self.clock.timeout.connect(self._tick)
        self._refresh()

    @staticmethod
    def _wrap(layout):
        host = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        host.setLayout(layout)
        return host

    def _toggle_advanced(self):
        visible = not self.advanced.isVisible()
        self.advanced.setVisible(visible)
        self.advanced_toggle.setText(
            "Hide advanced options" if visible else "Show advanced options"
        )

    # -- data ---------------------------------------------------------------------
    def set_log(self, log, inspection=None):
        self.log = log
        for box in self.camera_boxes:
            box.setParent(None)
            box.deleteLater()
        self.camera_boxes = []
        while self.readout_box.rowCount():
            self.readout_box.removeRow(0)
        self.readouts = {}
        self.imu.blockSignals(True)
        self.imu.clear()
        if log is not None:
            self.log_caption.setText(
                f"{log.name} · {log.format_label} · {log.duration:.1f} s. Checked cameras are tracked with TrackKLT, then calibrated."
            )
            usable = [c for c in log.cameras if c.has_pictures]
            preferred = [
                c
                for c in usable
                if c.intrinsics is not None and (c.global_shutter or log.kind != "voxl")
            ]
            for cam in usable:
                box = QCheckBox(cam.label)
                box.setChecked(
                    cam in preferred[:1]
                    if log.kind == "voxl"
                    else cam.intrinsics is not None
                )
                box.setEnabled(cam.intrinsics is not None)
                box.setToolTip(
                    ""
                    if cam.intrinsics
                    else (
                        cam.intrinsics_problem
                        or f"intrinsics missing: {cam.intrinsics_expected or 'no CameraInfo'}"
                    )
                )
                box.cam = cam
                box.toggled.connect(lambda *_: self._refresh())
                self.camera_list.addWidget(box)
                self.camera_boxes.append(box)
                if not cam.global_shutter and not cam.thermal:
                    spin = QDoubleSpinBox()
                    spin.setRange(0, 100)
                    spin.setDecimals(3)
                    spin.setSuffix(" ms")
                    self.readout_box.addRow(cam.label, spin)
                    self.readouts[cam.name] = spin
            for imu in log.imus:
                self.imu.addItem(imu.name, imu.name)
            self.profile.set_value(
                "flight"
                if log.kind == "voxl"
                and str(log.meta.get("note", "")).lower() == "armed"
                else "voxl"
                if log.kind == "voxl"
                else "library"
            )
            self.source.set_value("log")
        else:
            self.log_caption.setText(
                "Open a voxl-logger log or a bag to calibrate its cameras, or choose a session file."
            )
            self.source.set_value("session")
        self.imu.blockSignals(False)
        self.source.set_enabled_value("log", log is not None)
        self._refresh()

    def set_session(self, path):
        self.session_path = path or ""
        if path:
            self.session_edit.setText(path)
            if self.log is None:
                self.source.set_value("session")
        self._refresh()

    def chosen_cameras(self):
        return [b.cam for b in self.camera_boxes if b.isChecked() and b.isEnabled()]

    def chosen_imu(self):
        if self.log is None:
            return None
        return next(
            (i for i in self.log.imus if i.name == self.imu.currentData()), None
        )

    def _refresh(self):
        source = self.source.value()
        self.log_box.setVisible(source == "log")
        self.session_box.setVisible(source == "session")
        self.profile.set_enabled_value("record", source == "session")
        if source == "log" and self.profile.value() == "record":
            self.profile.set_value(
                "voxl"
                if self.log is not None and self.log.kind == "voxl"
                else "library"
            )
        cams = self.chosen_cameras()
        single_voxl = (
            self.log is not None and self.log.kind == "voxl" and len(cams) == 1
        )
        self.voxl_ingest.setEnabled(source == "log" and single_voxl)
        if not single_voxl:
            self.voxl_ingest.setChecked(False)
        while self.checks.count():
            item = self.checks.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        ready = True
        if source == "log":
            if self.log is None:
                self.checks.addWidget(_check(False, "No log is open"))
                ready = False
            else:
                ready &= bool(cams)
                self.checks.addWidget(
                    _check(
                        bool(cams),
                        f"{len(cams)} camera{'s' if len(cams) != 1 else ''} selected"
                        if cams
                        else "Select at least one camera with intrinsics",
                    )
                )
                missing = [b.cam for b in self.camera_boxes if not b.isEnabled()]
                if missing:
                    self.checks.addWidget(
                        _check(
                            False,
                            "No intrinsics: "
                            + ", ".join(c.label for c in missing)
                            + " (copy them from the vehicle's /data/modalai)",
                            hint=True,
                        )
                    )
                imu = self.chosen_imu()
                ready &= imu is not None
                self.checks.addWidget(
                    _check(
                        imu is not None,
                        f"IMU {imu.name} · {len(imu):,} samples"
                        if imu
                        else "The log has no IMU channel",
                    )
                )
                if imu is not None and len(imu) > 100:
                    for text, ok in self._excitation(imu):
                        self.checks.addWidget(_check(ok, text, hint=True))
                tracker = find_executable("ov_zcalib_track")
                if not self.voxl_ingest.isChecked():
                    ready &= bool(tracker)
                    self.checks.addWidget(
                        _check(
                            bool(tracker),
                            "Frame tracker ov_zcalib_track found"
                            if tracker
                            else "ov_zcalib_track not found: build ov_zcalib",
                        )
                    )
        else:
            path = self.session_edit.text().strip()
            ok = bool(path) and Path(path).expanduser().is_file()
            ready &= ok
            self.checks.addWidget(
                _check(ok, Path(path).name if ok else "Choose a session file")
            )
        engine = self.binary.text().strip() or find_executable("ov_zcalibrate")
        engine_ok = bool(engine) and Path(engine).is_file()
        ready &= engine_ok
        self.checks.addWidget(
            _check(
                engine_ok,
                "Engine ov_zcalibrate found"
                if engine_ok
                else "ov_zcalibrate not found (Advanced → Engine)",
            )
        )
        if self.cancel_button.isHidden():
            self.run_button.setEnabled(ready)

    @staticmethod
    def _excitation(imu):
        """Motion hints: zcalib needs rotation about every axis and some translation."""
        gyro = imu.data[:, 1:4] - np.median(imu.data[:, 1:4], axis=0)
        rms = np.sqrt(np.mean(gyro**2, axis=0))
        weak = [axis for axis, value in zip("xyz", rms) if value < 0.15]
        accel = np.linalg.norm(imu.data[:, 4:7], axis=1)
        dynamic = float(np.std(accel))
        text = (
            "Rotation RMS "
            + " · ".join(f"{a} {v:.2f}" for a, v in zip("xyz", rms))
            + " rad/s"
        )
        hints = [
            (text + (f" — weak about {', '.join(weak)}" if weak else ""), not weak)
        ]
        hints.append(
            (
                f"Acceleration variation {dynamic:.2f} m/s²"
                + (" — add translation" if dynamic < 0.5 else ""),
                dynamic >= 0.5,
            )
        )
        return hints

    # -- settings -----------------------------------------------------------------
    def settings(self):
        from .stream import CameraSettings, default_settings

        source = self.source.value()
        out = {
            "source": source,
            "profile": self.profile.value(),
            "cam_mode": int(self.intrinsics.value()),
            "budget": float(self.budget.value()),
            "threads": int(self.threads.value()),
            "binary": self.binary.text().strip() or find_executable("ov_zcalibrate"),
            "output_dir": Path(self.output_dir.text()).expanduser(),
            "features": int(self.features.value()),
            "voxl_ingest": bool(self.voxl_ingest.isChecked()),
        }
        if source == "session":
            path = Path(self.session_edit.text().strip()).expanduser()
            if not path.is_file():
                raise ValueError("Choose a session file to calibrate.")
            out["session_path"] = path
            return out
        cams, imu = self.chosen_cameras(), self.chosen_imu()
        if self.log is None or not cams:
            raise ValueError("Select at least one camera with intrinsics.")
        if imu is None:
            raise ValueError("The log has no IMU channel.")
        settings = {}
        for cam in cams:
            base = default_settings(cam)
            readout = self.readouts.get(cam.name)
            settings[cam.name] = CameraSettings(
                base.reference,
                readout.value() * 1e-3 if readout else base.readout_s,
                base.exposure_s,
                base.per_frame_exposure,
            )
        out.update(
            cameras=cams,
            imu=imu,
            settings=settings,
            readout_s=settings[cams[0].name].readout_s,
        )
        return out

    def remember(self):
        self.preferences.setValue("binary", self.binary.text().strip())
        self.preferences.setValue("output_dir", self.output_dir.text().strip())

    # -- run state ------------------------------------------------------------------
    def set_running(self, running):
        self.run_button.setVisible(not running)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(running)
        for widget in (
            self.source,
            self.log_box,
            self.session_box,
            self.advanced,
            self.profile,
            self.intrinsics,
            self.budget,
            self.output_dir,
        ):
            widget.setEnabled(not running)
        if running:
            self.console.clear()
            self.results_button.hide()
            self.started = time.monotonic()
            self.clock.start(500)
        else:
            self.clock.stop()
            self._refresh()

    def set_stage(self, state, text=""):
        self.stages.set_state(state)
        if text:
            self.status.setText(text)
            self.status.setStyleSheet(
                f"color:{RED if state == 'ABORT' else GREEN if state == 'DONE' else TEXT}; background:transparent;"
            )

    def append_log(self, text):
        for line in text.splitlines():
            if line.strip():
                self.console.appendPlainText(line)

    def finished(self, ok, text):
        self.set_running(False)
        self.status.setText(text)
        self.status.setStyleSheet(
            f"color:{GREEN if ok else RED}; background:transparent;"
        )
        self.results_button.setVisible(True)

    def cancelling(self):
        self.cancel_button.setEnabled(False)
        self.status.setText("Cancelling…")

    def _tick(self):
        if self.started is not None and self.cancel_button.isVisible():
            elapsed = time.monotonic() - self.started
            current = self.status.text().split(" · ")[0]
            self.status.setText(f"{current} · {elapsed:.0f} s")

    # -- choosers -------------------------------------------------------------------
    def _choose_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a session",
            self.session_edit.text(),
            "ov_zcalib session (*.bin)",
        )
        if path:
            self.session_edit.setText(path)

    def _choose_binary(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose ov_zcalibrate", self.binary.text(), "All files (*)"
        )
        if path:
            self.binary.setText(path)
            self._refresh()

    def _choose_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Output folder", self.output_dir.text()
        )
        if path:
            self.output_dir.setText(path)
