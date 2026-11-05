"""Dialogs for logs: channel selection (pipes/topics) and conversion to a session."""

from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QLabel,
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .widgets import AMBER, MUTED, button, label, row

KIND_LABEL = {
    "imu": "IMU",
    "camera": "Camera",
    "camera_info": "CameraInfo",
    "tf": "Static TF",
    "other": "Other",
}


class ChannelDialog(QDialog):
    """Checkable pipe/topic list from info.json or bag metadata; supported channels start checked."""

    def __init__(
        self,
        title,
        source_label,
        channels,
        selected=None,
        parent=None,
        accept_text="Open",
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(760, 520)
        self.channels = list(channels)
        chosen = (
            set(selected)
            if selected is not None
            else {c.name for c in self.channels if c.supported}
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 18)
        layout.setSpacing(12)
        layout.addWidget(label(title, 17, bold=True))
        note = label(
            f"{source_label}. Uncheck pipes or topics to skip them; heavy image topics load faster when skipped.",
            9,
            MUTED,
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter channels…")
        self.filter.textChanged.connect(self._filter)
        layout.addLayout(
            row(
                self.filter,
                button("All", lambda: self._check(lambda c: True)),
                button("None", lambda: self._check(lambda c: False)),
                button(
                    "Cameras",
                    lambda: self._check(
                        lambda c: c.kind in ("camera", "camera_info", "tf")
                    ),
                ),
                button("IMU", lambda: self._check(lambda c: c.kind == "imu")),
            )
        )
        self.table = QTableWidget(len(self.channels), 4)
        self.table.setHorizontalHeaderLabels(["Channel", "Kind", "Type", "Messages"])
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        for i, channel in enumerate(self.channels):
            item = QTableWidgetItem(channel.name)
            flags = Qt.ItemIsUserCheckable | (
                Qt.ItemIsEnabled if channel.supported else Qt.NoItemFlags
            )
            item.setFlags(flags)
            item.setCheckState(
                Qt.Checked
                if channel.name in chosen and channel.supported
                else Qt.Unchecked
            )
            if not channel.supported:
                item.setToolTip(channel.reason)
            self.table.setItem(i, 0, item)
            for column, text in enumerate(
                (
                    KIND_LABEL.get(channel.kind, channel.kind),
                    channel.type,
                    f"{channel.count:,}",
                ),
                1,
            ):
                cell = QTableWidgetItem(text)
                cell.setFlags(Qt.ItemIsEnabled if channel.supported else Qt.NoItemFlags)
                if column == 3:
                    cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if not channel.supported:
                    cell.setToolTip(channel.reason)
                self.table.setItem(i, column, cell)
        self.table.itemChanged.connect(lambda *_: self._update())
        layout.addWidget(self.table, 1)
        self.summary = label("", 9, MUTED)
        layout.addWidget(self.summary)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(accept_text)
        buttons.button(QDialogButtonBox.Ok).setObjectName("primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok = buttons.button(QDialogButtonBox.Ok)
        layout.addWidget(buttons)
        self._update()

    def _check(self, predicate):
        for i, channel in enumerate(self.channels):
            if channel.supported and not self.table.isRowHidden(i):
                self.table.item(i, 0).setCheckState(
                    Qt.Checked if predicate(channel) else Qt.Unchecked
                )

    def _filter(self, text):
        text = text.strip().lower()
        for i, channel in enumerate(self.channels):
            hay = f"{channel.name} {channel.kind} {channel.type}".lower()
            self.table.setRowHidden(i, bool(text) and text not in hay)

    def _update(self):
        chosen = self.selected()
        sensors = [
            c for c in self.channels if c.name in chosen and c.kind in ("imu", "camera")
        ]
        self.summary.setText(
            f"{len(chosen)} of {len(self.channels)} channels selected · {len(sensors)} sensor streams"
        )
        self.ok.setEnabled(bool(chosen))

    def selected(self):
        return [
            self.channels[i].name
            for i in range(len(self.channels))
            if self.table.item(i, 0).checkState() == Qt.Checked
        ]


class ConvertDialog(QDialog):
    """Choose cameras/IMU and timing, then convert a log to a session.bin with TrackKLT."""

    def __init__(self, log, output_dir, parent=None):
        super().__init__(parent)
        from .stream import default_settings

        self.log = log
        self.setWindowTitle("Convert log to session")
        self.setMinimumSize(780, 520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 18)
        layout.setSpacing(12)
        layout.addWidget(label("Convert to a calibration session", 17, bold=True))
        note = label(
            "Frames are tracked with the default OpenVINS TrackKLT frontend at the camera intrinsics and written with "
            "the IMU stream to a completion-checked session.bin. Stamps are converted to center-row mid-exposure.",
            9,
            MUTED,
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.cameras = list(log.cameras)
        self.table = QTableWidget(len(self.cameras), 5)
        self.table.setHorizontalHeaderLabels(
            ["Camera", "Stamps", "Readout ms", "Exposure ms", "Intrinsics"]
        )
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.widgets = []
        for i, cam in enumerate(self.cameras):
            settings = default_settings(cam)
            item = QTableWidgetItem(cam.name)
            usable = cam.intrinsics is not None and cam.has_pictures
            item.setFlags(
                Qt.ItemIsUserCheckable
                | (Qt.ItemIsEnabled if usable else Qt.NoItemFlags)
            )
            item.setCheckState(
                Qt.Checked
                if usable and (i == 0 or cam.global_shutter or log.kind != "voxl")
                else Qt.Unchecked
            )
            self.table.setItem(i, 0, item)
            reference = QComboBox()
            if cam.stamp_reference == "start":
                reference.addItem("start of exposure · per-frame exposure", "start")
                reference.setEnabled(False)
            else:
                reference.addItem("center of exposure", "center")
                reference.addItem("start of exposure", "start")
                reference.setCurrentIndex(0 if settings.reference == "center" else 1)
            readout = QDoubleSpinBox()
            readout.setRange(0, 1000)
            readout.setDecimals(4)
            readout.setValue(settings.readout_s * 1e3)
            exposure = QDoubleSpinBox()
            exposure.setRange(0, 1000)
            exposure.setDecimals(4)
            exposure.setValue(settings.exposure_s * 1e3)
            exposure.setEnabled(cam.stamp_reference != "start")
            if cam.stamp_reference == "start":
                exposure.setToolTip("Per-frame exposure is read from the log")
            self.table.setCellWidget(i, 1, reference)
            self.table.setCellWidget(i, 2, readout)
            self.table.setCellWidget(i, 3, exposure)
            source = (
                (cam.intrinsics or {}).get("source")
                or cam.intrinsics_problem
                or "missing — " + (cam.intrinsics_expected or "no CameraInfo")
            )
            info = QTableWidgetItem(Path(source).name if cam.intrinsics else source)
            info.setToolTip(source)
            info.setFlags(Qt.ItemIsEnabled)
            if not usable:
                info.setForeground(Qt.darkYellow)
            self.table.setItem(i, 4, info)
            self.widgets.append((reference, readout, exposure))
        self.table.resizeRowsToContents()
        layout.addWidget(self.table, 1)
        form = QFormLayout()
        form.setSpacing(10)
        self.imu = QComboBox()
        for imu in log.imus:
            self.imu.addItem(f"{imu.name} · {len(imu):,} samples", imu.name)
        form.addRow("IMU", self.imu)
        self.features = QSpinBox()
        self.features.setRange(20, 2000)
        self.features.setValue(200)
        form.addRow("Features per camera", self.features)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = log.name.replace("/", "_").strip("_") or "log"
        self.output = QLineEdit(
            str(Path(output_dir).expanduser() / f"{safe}_{stamp}.bin")
        )
        form.addRow(
            "Session file", row(self.output, button("Browse", self._browse, "folder"))
        )
        layout.addLayout(form)
        self.warning = label("", 9, AMBER)
        self.warning.setWordWrap(True)
        layout.addWidget(self.warning)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Convert")
        buttons.button(QDialogButtonBox.Ok).setObjectName("primary")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if not log.imus:
            self.warning.setText("Include an IMU channel to convert this log.")
            buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Session file", self.output.text(), "ov_zcalib session (*.bin)"
        )
        if path:
            self.output.setText(path)

    def _accept(self):
        if not self.chosen_cameras():
            self.warning.setText("Select at least one camera with intrinsics.")
            return
        if Path(self.output.text()).expanduser().exists():
            self.warning.setText(
                "That session file already exists; existing recordings are never overwritten."
            )
            return
        self.accept()

    def chosen_cameras(self):
        return [
            cam
            for i, cam in enumerate(self.cameras)
            if self.table.item(i, 0).checkState() == Qt.Checked
        ]

    def result_settings(self):
        from .stream import CameraSettings

        settings = {}
        for i, cam in enumerate(self.cameras):
            reference, readout, exposure = self.widgets[i]
            settings[cam.name] = CameraSettings(
                reference.currentData(), readout.value() * 1e-3, exposure.value() * 1e-3
            )
        imu = next((i for i in self.log.imus if i.name == self.imu.currentData()), None)
        return {
            "cameras": self.chosen_cameras(),
            "imu": imu,
            "settings": settings,
            "out": Path(self.output.text()).expanduser(),
            "features": self.features.value(),
        }


ROTATIONS = (
    ("0°", 0),
    ("90° clockwise", 90),
    ("180°", 180),
    ("90° counter-clockwise", 270),
)


class CameraOrientationDialog(QDialog):
    """Rotate a camera's frames and edit its extrinsic edge (VOXL convention).

    Returns via ``choice`` ("save" | "apply") with ``rotation`` and ``edge``.
    Rotating the image turns the camera frame about its optical axis; with
    "Rotate the extrinsics with the image" the edge follows so frames,
    intrinsics and extrinsics stay physically consistent.
    """

    def __init__(self, log, cam, frame_row=0, parent=None):
        super().__init__(parent)
        import json as _json

        import numpy as np
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtWidgets import (
            QCheckBox,
            QGridLayout,
            QPlainTextEdit,
        )

        from .geometry import (
            angle_between_deg,
            find_rest,
            image_rotation,
            matrix_rpy,
            rpy_matrix,
        )
        from .media import Tonemapper, array_to_qimage, black_level_for, rotate_picture

        self._np, self._json, self._QPixmap = np, _json, QPixmap
        self._rpy_matrix, self._matrix_rpy, self._image_rotation = (
            rpy_matrix,
            matrix_rpy,
            image_rotation,
        )
        self._angle, self._array_to_qimage, self._rotate_picture = (
            angle_between_deg,
            array_to_qimage,
            rotate_picture,
        )
        self.log, self.cam, self.choice = log, cam, None
        self.setWindowTitle(f"Camera orientation · {cam.camera or cam.name}")
        self.setMinimumSize(860, 640)
        self.start_rotation = cam.rotation
        self.frame_name = log.camera_frame(cam)
        rig = log.rig
        edge = next((e for e in rig.edges if e.child == self.frame_name), None)
        imu = log.imus[0] if log.imus else None
        self.imu_frame = (
            imu.frame
            if imu and imu.frame in rig.frames
            else rig.match(imu.name)
            if imu
            else None
        ) or (imu.name if imu else "")
        if edge is not None:
            self.base_parent, self.base_R, self.base_t = (
                edge.parent,
                edge.R.copy(),
                edge.t.copy(),
            )
        else:
            self.base_parent = self.imu_frame or rig.body or "imu"
            self.base_R, self.base_t = np.eye(3), np.zeros(3)
        self.had_edge = edge is not None
        rest = find_rest(imu.data) if imu is not None else None
        self.down_in_imu = (
            -rest[2] / np.linalg.norm(rest[2]) if rest is not None else None
        )
        try:
            raw = cam._load(int(frame_row)) if cam.has_pictures else None
        except Exception:
            raw = None
        self.raw_picture = raw
        self.mapper = Tonemapper(
            black_level_for(cam.name, cam.sensor), 720, smooth=False
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 18)
        layout.setSpacing(12)
        layout.addWidget(label(f"Orientation of camera {cam.label}", 17, bold=True))
        note = label(
            "Rotating the image turns the camera frame about its optical axis (like VOXL's en_rotate): frames, size, "
            "intrinsics and — with the option below — the extrinsics change together. Edits apply to this session; "
            "Save asks before writing anything.",
            9,
            MUTED,
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        self.preview = QLabel()
        self.preview.setMinimumSize(380, 260)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(
            "background:#0e1822; border:1px solid #283440; border-radius:6px;"
        )
        grid.addWidget(self.preview, 0, 0, 3, 1)
        form = QFormLayout()
        form.setSpacing(9)
        self.rotation = QComboBox()
        for text, value in ROTATIONS:
            self.rotation.addItem(text, value)
        self.rotation.setCurrentIndex([v for _, v in ROTATIONS].index(cam.rotation))
        form.addRow("Image rotation", self.rotation)
        self.follow = QCheckBox("Rotate the extrinsics with the image")
        self.follow.setChecked(self.had_edge)
        self.follow.setEnabled(self.had_edge)
        self.follow.setToolTip(
            "Keeps the physical camera pose: the camera frame turns about z by the image rotation."
            if self.had_edge
            else f"{cam.name} has no extrinsics yet: enter them below to add an entry."
        )
        form.addRow("", self.follow)
        self.parent_box = QComboBox()
        self.parent_box.setEditable(True)
        frames = sorted(f for f in rig.frames if f != self.frame_name) or [
            self.base_parent
        ]
        for f in [self.base_parent] + [f for f in frames if f != self.base_parent]:
            self.parent_box.addItem(f)
        form.addRow("Parent frame", self.parent_box)
        self.child = QLineEdit(self.frame_name)
        self.child.setReadOnly(self.had_edge)
        form.addRow("Camera frame", self.child)
        self.rpy, self.t = [], []
        for name, values, store, rng, decimals in (
            ("RPY parent→child (deg)", matrix_rpy(self.base_R), self.rpy, 360.0, 3),
            ("T child in parent (m)", self.base_t, self.t, 100.0, 4),
        ):
            spins = []
            for v in values:
                spin = QDoubleSpinBox()
                spin.setRange(-rng, rng)
                spin.setDecimals(decimals)
                spin.setValue(float(v))
                spin.valueChanged.connect(self._update)
                spins.append(spin)
                store.append(spin)
            form.addRow(name, row(*spins))
        grid.addLayout(form, 0, 1)
        self.check = label("", 9, MUTED)
        self.check.setWordWrap(True)
        grid.addWidget(self.check, 1, 1)
        layout.addLayout(grid)
        layout.addWidget(label("For the vehicle (/etc/modalai)", 9, MUTED, True))
        self.snippet = QPlainTextEdit()
        self.snippet.setReadOnly(True)
        self.snippet.setMaximumHeight(130)
        layout.addWidget(self.snippet)
        buttons = QDialogButtonBox()
        self.save_button = buttons.addButton("Save…", QDialogButtonBox.AcceptRole)
        self.save_button.setObjectName("primary")
        apply = buttons.addButton("Apply for this session", QDialogButtonBox.ApplyRole)
        buttons.addButton(QDialogButtonBox.Cancel)
        self.save_button.clicked.connect(lambda: self._finish("save"))
        apply.clicked.connect(lambda: self._finish("apply"))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._last_rotation = cam.rotation
        self.rotation.currentIndexChanged.connect(self._rotation_changed)
        self.parent_box.currentTextChanged.connect(self._update)
        self._update()

    # -- state --------------------------------------------------------------
    def _rotation_changed(self):
        new = self.rotation.currentData()
        if self.follow.isChecked():
            delta = (new - self._last_rotation) % 360
            R = self._rpy_matrix(*[s.value() for s in self.rpy]) @ self._image_rotation(
                delta
            )
            for spin, value in zip(self.rpy, self._matrix_rpy(R)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self._last_rotation = new
        self._update()

    @property
    def rotation_value(self):
        return int(self.rotation.currentData())

    def edge(self):
        from .geometry import RigEdge

        np = self._np
        R = self._rpy_matrix(*[s.value() for s in self.rpy])
        t = np.array([s.value() for s in self.t])
        unchanged = (
            self.had_edge
            and self.parent_box.currentText() == self.base_parent
            and np.allclose(R, self.base_R, atol=1e-9)
            and np.allclose(t, self.base_t, atol=1e-9)
        )
        # Without a configured edge, identity/zero are placeholders, not entered values;
        # the image rotation is carried separately (rotation_value).
        if unchanged or (
            not self.had_edge and np.allclose(R, np.eye(3)) and not t.any()
        ):
            return None
        return RigEdge(
            self.parent_box.currentText().strip(),
            self.child.text().strip() or self.frame_name,
            R,
            t,
        )

    def _update(self, *_):
        if self.raw_picture is not None:
            pixels = self.mapper(
                self._rotate_picture(self.raw_picture, self.rotation_value)
            )
            pixmap = self._QPixmap.fromImage(self._array_to_qimage(pixels))
            self.preview.setPixmap(
                pixmap.scaled(
                    self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        else:
            self.preview.setText("no frame to preview")
        R = self._rpy_matrix(*[s.value() for s in self.rpy])
        parent = self.parent_box.currentText().strip()
        entered = self.had_edge or self.edge() is not None
        lines = [
            f"optical axis (z) in {parent}: [{R[0, 2]:+.2f} {R[1, 2]:+.2f} {R[2, 2]:+.2f}]"
        ]
        transform = (
            self.log.rig.transform(self.imu_frame, parent) if self.imu_frame else None
        )
        if not entered:
            lines = [
                f"{self.frame_name} has no extrinsics in {self.log.rig.source or 'the log'} yet. Rotating the image "
                "needs none; to add them, enter RPY/T relative to a parent frame and the gravity check will tell "
                "whether the image is upright."
            ]
        elif self.down_in_imu is not None and transform is not None:
            # transform = parent axes in the IMU frame; IMU-frame vectors map with its transpose.
            down_in_parent = transform[0].T @ self.down_in_imu
            angle = self._angle(R[:, 1], down_in_parent)
            verdict = (
                "upright"
                if angle < 30
                else "upside down"
                if angle > 150
                else "on its side"
            )
            lines.append(
                f"image 'down' is {angle:.0f}° from gravity at rest: the image is {verdict}"
            )
        elif self.down_in_imu is None:
            lines.append("no stationary IMU period: gravity check unavailable")
        self.check.setText("\n".join(lines))
        rpy = [round(s.value(), 3) for s in self.rpy]
        t = [round(s.value(), 4) for s in self.t]
        entry = {
            "parent": parent,
            "child": self.child.text().strip() or self.frame_name,
            "T_child_wrt_parent": t,
            "RPY_parent_to_child": rpy,
        }
        fields = ",\n".join(
            f"\t{self._json.dumps(key)}:\t{self._json.dumps(value)}"
            for key, value in entry.items()
        )
        text = (
            "extrinsics.conf entry:\n{\n" + fields + "\n}"
            if entered
            else "No extrinsics entry (none entered)."
        )
        if self.rotation_value:
            camera = self.cam.camera or self.cam.name
            if self.log.kind == "voxl" and self.rotation_value == 180:
                text += f'\n\nvoxl-camera-server.conf, camera "{camera}": set "en_rotate": true'
            else:
                text += f"\n\nRotate camera {camera} by {self.rotation_value}° clockwise in the camera pipeline"
            text += " (frames recorded before that change stay as logged; the studio rotates them)."
        self.snippet.setPlainText(text)

    def _finish(self, choice):
        self.choice = choice
        self.accept()
