"""Sensor setup kept separate from calibration and transport-independent viewing."""

import os

from PyQt5.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from .sources import SourceSettings, TopicDiscovery
from .widgets import MUTED, label


class SourceDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect sensors")
        self.setMinimumWidth(590)
        self.fields = {}
        self.discovery = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(18)
        layout.addWidget(label("Your sensors. One workspace.", 19, bold=True))
        subtitle = label(
            "View topics live, then record with the default OpenVINS TrackKLT frontend.",
            10,
            MUTED,
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(18, 20, 18, 20)
        form.setSpacing(14)
        self.transport = QComboBox()
        self.transport.addItems(["ROS 2", "ROS 1"])
        self.transport.setCurrentText(settings.transport)
        form.addRow("Transport", self.transport)
        self.camera_name = QLineEdit(settings.camera_name)
        form.addRow("Camera name", self.camera_name)
        self.reliable_images = QCheckBox("Request reliable image delivery")
        self.reliable_images.setChecked(settings.reliable_images)
        self.reliable_images.setToolTip(
            "ROS 2 only. Enable when the image publisher offers reliable QoS, including the D455 bridge."
        )
        form.addRow("Image delivery", self.reliable_images)
        for key, title in (
            ("image_topic", "Camera image"),
            ("info_topic", "Camera info"),
            ("imu_topic", "Raw IMU"),
            ("tracks_topic", "Feature overlay"),
        ):
            field = QComboBox()
            field.setEditable(True)
            field.addItem(getattr(settings, key))
            field.setMinimumWidth(340)
            self.fields[key] = field
            form.addRow(title, field)
        self.discover_button = QPushButton("Discover published topics")
        self.discover_button.clicked.connect(self._discover)
        form.addRow("ROS graph", self.discover_button)
        self.discovery_status = label("Select a topic or type its name.", 9, MUTED)
        self.discovery_status.setWordWrap(True)
        form.addRow(self.discovery_status)
        note = label(
            f"ROS_DOMAIN_ID: {os.environ.get('ROS_DOMAIN_ID', '0')}   ·   Sensor-data QoS\n"
            "Use a raw Image topic and its matching CameraInfo.\n"
            "Feature overlays are published by ov_zcalib_capture while recording.",
            9,
            MUTED,
        )
        note.setWordWrap(True)
        form.addRow(note)
        tabs.addTab(page, "Topics")
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(18, 20, 18, 20)
        form.setSpacing(14)
        self.reference = QComboBox()
        self.reference.addItem("Center-row mid-exposure (use verbatim)", "center")
        self.reference.addItem("Start of exposure (convert once)", "start")
        self.reference.setCurrentIndex(
            0 if settings.timestamp_reference == "center" else 1
        )
        form.addRow("Image timestamp", self.reference)
        for key, title in (
            ("readout_s", "Hardware readout · ms"),
            ("exposure_s", "Exposure · ms"),
        ):
            field = QDoubleSpinBox()
            field.setRange(0, 1000)
            field.setDecimals(6)
            field.setValue(getattr(settings, key) * 1000)
            self.fields[key] = field
            form.addRow(title, field)
        features = QSpinBox()
        features.setRange(20, 2000)
        features.setValue(settings.max_features)
        self.fields["max_features"] = features
        form.addRow("Maximum features", features)
        note = label(
            "Readout is a measured sensor property and stays fixed.\n"
            "Start-of-exposure stamps gain (readout + exposure) / 2.\n"
            "Use center stamps for publishers that already apply this correction.\n"
            "The exposure setting assumes constant exposure during capture.",
            9,
            MUTED,
        )
        note.setWordWrap(True)
        form.addRow(note)
        tabs.addTab(page, "Timing + tracker")
        page = QWidget()
        extra_layout = QVBoxLayout(page)
        extra_layout.setContentsMargins(18, 20, 18, 20)
        note = label(
            "Record every camera with the shared IMU. Each camera keeps its own geometry and exposure timeline.",
            10,
            MUTED,
        )
        note.setWordWrap(True)
        extra_layout.addWidget(note)
        self.extra_table = QTableWidget(0, 6)
        self.extra_table.setHorizontalHeaderLabels(
            [
                "Name",
                "Image topic",
                "CameraInfo topic",
                "Readout ms",
                "Exposure ms",
                "Stamp",
            ]
        )
        self.extra_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.extra_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.extra_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        extra_layout.addWidget(self.extra_table)
        controls = QHBoxLayout()
        add = QPushButton("Add camera")
        add.clicked.connect(lambda: self._add_camera())
        remove = QPushButton("Remove selected")
        remove.clicked.connect(
            lambda: self.extra_table.removeRow(self.extra_table.currentRow())
        )
        preset = QPushButton("D455 · stereo + RGB")
        preset.clicked.connect(self._d455)
        for control in (add, remove, preset):
            controls.addWidget(control)
        extra_layout.addLayout(controls)
        note = label(
            "D455 preset uses exposure-centered topics from ov_zcalib_realsense_bridge. Left/right IR are already rectified by D4; the bridge rectifies RGB with its full CameraInfo model.",
            9,
            MUTED,
        )
        note.setWordWrap(True)
        extra_layout.addWidget(note)
        tabs.addTab(page, "Additional cameras")
        for camera in settings.camera_sources()[1:]:
            self._add_camera(camera)
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(18, 20, 18, 20)
        form.setSpacing(14)
        for key, title in (
            ("sigma_w", "Gyro white noise"),
            ("sigma_wb", "Gyro bias random walk"),
            ("sigma_a", "Accel white noise"),
            ("sigma_ab", "Accel bias random walk"),
        ):
            field = QDoubleSpinBox()
            field.setDecimals(9)
            field.setRange(0.000000001, 100.0)
            field.setSingleStep(0.00001)
            field.setValue(getattr(settings, key))
            self.fields[key] = field
            form.addRow(title, field)
        note = label(
            "Continuous-time densities in Kalibr-chain units.\n"
            "Set these from your IMU's Allan characterization. Values are\n"
            "recorded with the session and used for calibration weighting.",
            9,
            MUTED,
        )
        note.setWordWrap(True)
        form.addRow(note)
        tabs.addTab(page, "IMU weighting")
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.button(QDialogButtonBox.Ok).setText("Connect sensors")
        buttons.button(QDialogButtonBox.Ok).setObjectName("primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self):
        values = {
            key: field.currentText().strip()
            if isinstance(field, QComboBox)
            else field.value()
            for key, field in self.fields.items()
        }
        values["readout_s"] /= 1000
        values["exposure_s"] /= 1000
        return SourceSettings(
            transport=self.transport.currentText(),
            timestamp_reference=self.reference.currentData(),
            camera_name=self.camera_name.text().strip() or "Camera 00",
            extra_cameras=self._extra_settings(),
            reliable_images=self.reliable_images.isChecked(),
            **values,
        )

    def _add_camera(self, camera=None):
        if self.extra_table.rowCount() >= 15:
            return
        camera = camera or {}
        row = self.extra_table.rowCount()
        self.extra_table.insertRow(row)
        self.extra_table.setItem(
            row, 0, QTableWidgetItem(camera.get("camera_name", f"Camera {row + 1:02d}"))
        )
        for column, key in ((1, "image_topic"), (2, "info_topic")):
            topic = QComboBox()
            topic.setEditable(True)
            topic.addItem(camera.get(key, ""))
            self.extra_table.setCellWidget(row, column, topic)
        for column, key in ((3, "readout_s"), (4, "exposure_s")):
            value = QDoubleSpinBox()
            value.setRange(0, 1000)
            value.setDecimals(6)
            value.setValue(camera.get(key, 0.0) * 1000)
            self.extra_table.setCellWidget(row, column, value)
        reference = QComboBox()
        reference.addItems(["center", "start"])
        reference.setCurrentText(camera.get("timestamp_reference", "center"))
        self.extra_table.setCellWidget(row, 5, reference)

    def _extra_settings(self):
        result = []
        for row in range(self.extra_table.rowCount()):
            result.append(
                dict(
                    camera_name=self.extra_table.item(row, 0).text().strip(),
                    image_topic=self.extra_table.cellWidget(row, 1)
                    .currentText()
                    .strip(),
                    info_topic=self.extra_table.cellWidget(row, 2)
                    .currentText()
                    .strip(),
                    readout_s=self.extra_table.cellWidget(row, 3).value() / 1000,
                    exposure_s=self.extra_table.cellWidget(row, 4).value() / 1000,
                    timestamp_reference=self.extra_table.cellWidget(
                        row, 5
                    ).currentText(),
                )
            )
        return result

    def _d455(self):
        self.reliable_images.setChecked(True)
        self.camera_name.setText("Left IR · D4 rectified")
        self.fields["image_topic"].setCurrentText("/zcalib/d455/infra1/image")
        self.fields["info_topic"].setCurrentText("/zcalib/d455/infra1/camera_info")
        self.fields["imu_topic"].setCurrentText("/camera/camera/imu")
        self.fields["readout_s"].setValue(0)
        self.fields["exposure_s"].setValue(10)
        self.reference.setCurrentIndex(0)
        self.extra_table.setRowCount(0)
        for stream, name in (
            ("infra2", "Right IR · D4 rectified"),
            ("color", "RGB · rectified"),
        ):
            self._add_camera(
                dict(
                    camera_name=name,
                    image_topic=f"/zcalib/d455/{stream}/image",
                    info_topic=f"/zcalib/d455/{stream}/camera_info",
                    exposure_s=0.01,
                )
            )

    def accept(self):
        try:
            self.settings().camera_sources()
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Sensor settings", str(error))
            return
        super().accept()

    def _discover(self):
        if self.discovery is not None and self.discovery.isRunning():
            return
        self.discover_button.setEnabled(False)
        self.discovery_status.setText("Discovering sensor topics…")
        self.discovery = TopicDiscovery(self.transport.currentText(), self)
        self.discovery.found.connect(self._topics)
        self.discovery.problem.connect(
            lambda text: self.discovery_status.setText("Discovery unavailable: " + text)
        )
        self.discovery.finished.connect(lambda: self.discover_button.setEnabled(True))
        self.discovery.start()

    def _topics(self, topics):
        suffixes = {
            "image_topic": "Image",
            "info_topic": "CameraInfo",
            "imu_topic": "Imu",
            "tracks_topic": "PointCloud",
        }
        matches = 0
        for key, suffix in suffixes.items():
            field = self.fields[key]
            current = field.currentText()
            names = sorted(
                name
                for name, types in topics.items()
                if any(typ.split("/")[-1] == suffix for typ in types)
            )
            matches += len(names)
            field.clear()
            field.addItems(list(dict.fromkeys([current] + names)))
            field.setCurrentText(current)
        for row in range(self.extra_table.rowCount()):
            for column, suffix in ((1, "Image"), (2, "CameraInfo")):
                field = self.extra_table.cellWidget(row, column)
                current = field.currentText()
                names = sorted(
                    name
                    for name, types in topics.items()
                    if any(typ.split("/")[-1] == suffix for typ in types)
                )
                field.clear()
                field.addItems(list(dict.fromkeys([current] + names)))
                field.setCurrentText(current)
        self.discovery_status.setText(
            f"Found {matches} matching sensor topics. Choose from the lists above."
        )

    def done(self, result):
        if self.discovery is not None and self.discovery.isRunning():
            self.discovery.requestInterruption()
            if not self.discovery.wait(1500):
                return
        super().done(result)
