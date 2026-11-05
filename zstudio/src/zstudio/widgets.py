"""Qt-painted scientific views. No OpenGL context or plotting dependency required."""

from collections import deque
import math

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from .model import coverage

# Design tokens: Apple-style dark appearance (system grays, one accent, semantic colors).
BG = "#1c1c1e"  # window
SIDEBAR = "#232325"
PANEL = "#2c2c2e"  # grouped content
RAISED = "#3a3a3c"  # controls
LINE = "#38383a"  # separators
TEXT = "#f5f5f7"
MUTED = "#98989d"  # secondary label
FAINT = "#636366"  # tertiary label
ACCENT = "#0a84ff"
TEAL = ACCENT  # historical name of the accent color
BLUE = "#64d2ff"  # secondary / informational
GREEN = "#30d158"
AMBER = "#ff9f0a"
RED = "#ff453a"
PURPLE = "#bf5af2"
CANVAS = "#111113"  # image / 3D canvases
AXES = (RED, GREEN, ACCENT)  # x, y, z

_UI_FAMILIES = [
    "Inter",
    "SF Pro Text",
    "Helvetica Neue",
    "Noto Sans",
    "Ubuntu",
    "DejaVu Sans",
]
_MONO_FAMILIES = ["SF Mono", "JetBrains Mono", "DejaVu Sans Mono"]


def font(size=10, bold=False, mono=False):
    result = QFont()
    result.setFamilies(_MONO_FAMILIES if mono else _UI_FAMILIES)
    result.setPointSizeF(size)
    result.setWeight(QFont.DemiBold if bold else QFont.Normal)
    return result


def icon(name, color=MUTED, size=20):
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(size / 24, size / 24)
    painter.setPen(QPen(QColor(color), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    if name == "workspace":
        for x, y, w, h in ((3, 3, 8, 8), (15, 3, 6, 8), (3, 15, 8, 6), (15, 15, 6, 6)):
            painter.drawRoundedRect(QRectF(x, y, w, h), 1.5, 1.5)
    elif name == "calibrate":
        for y, x in ((5, 8), (12, 16), (19, 10)):
            painter.drawLine(3, y, 21, y)
            painter.setBrush(QColor(PANEL))
            painter.drawEllipse(QPointF(x, y), 2.3, 2.3)
    elif name == "results":
        painter.drawRoundedRect(QRectF(5, 3, 14, 18), 2, 2)
        painter.drawLine(8, 8, 16, 8)
        painter.drawLine(8, 12, 16, 12)
        painter.drawLine(8, 16, 13, 16)
    elif name == "console":
        painter.drawRoundedRect(QRectF(2, 4, 20, 16), 2, 2)
        painter.drawPolyline(QPolygonF([QPointF(6, 9), QPointF(9, 12), QPointF(6, 15)]))
        painter.drawLine(12, 15, 17, 15)
    elif name == "folder":
        painter.drawPolyline(
            QPolygonF(
                [
                    QPointF(2, 19),
                    QPointF(2, 5),
                    QPointF(9, 5),
                    QPointF(11, 8),
                    QPointF(21, 8),
                    QPointF(21, 19),
                    QPointF(2, 19),
                ]
            )
        )
    elif name == "connect":
        painter.drawRoundedRect(QRectF(7, 8, 10, 8), 2, 2)
        painter.drawLine(9, 3, 9, 8)
        painter.drawLine(15, 3, 15, 8)
        painter.drawLine(12, 16, 12, 21)
    elif name == "play":
        painter.setBrush(QColor(color))
        painter.drawPolygon(QPolygonF([QPointF(8, 5), QPointF(19, 12), QPointF(8, 19)]))
    elif name == "pause":
        painter.drawLine(8, 5, 8, 19)
        painter.drawLine(16, 5, 16, 19)
    elif name == "record":
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(12, 12), 5.5, 5.5)
    elif name == "stop":
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(QRectF(6, 6, 12, 12), 2, 2)
    elif name == "download":
        painter.drawLine(12, 3, 12, 15)
        painter.drawPolyline(
            QPolygonF([QPointF(7, 10), QPointF(12, 15), QPointF(17, 10)])
        )
        painter.drawPolyline(
            QPolygonF(
                [QPointF(4, 16), QPointF(4, 21), QPointF(20, 21), QPointF(20, 16)]
            )
        )
    elif name == "reset":
        painter.drawArc(QRectF(5, 5, 15, 15), 30 * 16, 285 * 16)
        painter.drawPolyline(QPolygonF([QPointF(4, 3), QPointF(4, 9), QPointF(10, 9)]))
    elif name == "camera":
        painter.drawRoundedRect(QRectF(2, 6, 20, 14), 2, 2)
        painter.drawEllipse(QPointF(12, 13), 4, 4)
        painter.drawLine(8, 3, 16, 3)
    elif name == "inspect":
        painter.drawEllipse(QPointF(10.5, 10.5), 6.5, 6.5)
        painter.drawLine(QPointF(15.5, 15.5), QPointF(21, 21))
        painter.drawLine(QPointF(7.5, 10.5), QPointF(13.5, 10.5))
    elif name == "trajectory":
        path = QPainterPath(QPointF(3, 19))
        path.cubicTo(QPointF(8, 6), QPointF(13, 22), QPointF(20, 6))
        painter.drawPath(path)
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(20, 6), 2.2, 2.2)
    elif name == "tune":
        for x, y in ((6, 9), (12, 15), (18, 7)):
            painter.drawLine(x, 3, x, 21)
            painter.setBrush(QColor(PANEL))
            painter.drawEllipse(QPointF(x, y), 2.3, 2.3)
    elif name == "cube":
        polygon = [
            QPointF(12, 2),
            QPointF(21, 7),
            QPointF(21, 17),
            QPointF(12, 22),
            QPointF(3, 17),
            QPointF(3, 7),
            QPointF(12, 2),
        ]
        painter.drawPolyline(QPolygonF(polygon))
        painter.drawLine(QPointF(3, 7), QPointF(12, 12))
        painter.drawLine(QPointF(21, 7), QPointF(12, 12))
        painter.drawLine(12, 12, 12, 22)
    else:
        painter.drawEllipse(QPointF(12, 12), 8, 8)
        painter.drawLine(12, 8, 12, 13)
        painter.drawPoint(12, 17)
    painter.end()
    return QIcon(pix)


class Card(QFrame):
    def __init__(self, parent=None, margins=16):
        super().__init__(parent)
        self.setObjectName("card")
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(margins, margins, margins, margins)
        self.box.setSpacing(10)


def label(text, size=10, color=TEXT, bold=False):
    widget = QLabel(text)
    widget.setFont(font(size, bold))
    widget.setStyleSheet(f"color:{color}; background:transparent; border:none;")
    return widget


def pill(text, color=TEAL):
    widget = label(text, 8.5, color, True)
    set_pill_color(widget, color)
    return widget


def set_pill_color(widget, color):
    """Capsule tag: tinted fill, colored text (macOS tag style)."""
    c = QColor(color)
    widget.setStyleSheet(
        f"color:{color}; background:rgba({c.red()},{c.green()},{c.blue()},46); "
        f"border:none; border-radius:9px; padding:3px 10px;"
    )


class FeatureView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 240)
        self.setMouseTracking(True)
        self.frame = None
        self.camera = None
        self.demo = False
        self.show_grid = True
        self.show_trails = True
        self.history = deque(maxlen=14)
        self.title = ""  # camera name shown in the heading
        self.empty_text = "No image"

        self.captions = None  # (wide caption, short word) for non-live image sources
        self.zoom = 1.0
        self.pan = QPointF()
        self.drag = None
        self.bounds = QRectF()

    def set_frame(self, frame, camera, demo=False):
        previous = self.frame
        if frame is not None and (
            previous is None
            or frame.timestamp != previous.timestamp
            or frame.cam != previous.cam
        ):
            if previous is not None and (
                frame.timestamp < previous.timestamp
                or frame.cam != previous.cam
                or frame.timestamp - previous.timestamp > 0.5
            ):
                self.history.clear()
            self.history.append(
                {int(p["id"]): (float(p["u"]), float(p["v"])) for p in frame.points}
            )
        self.frame, self.camera, self.demo = frame, camera, demo
        self.update()

    def reset_view(self):
        self.zoom = 1.0
        self.pan = QPointF()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(CANVAS))
        if self.camera is None:
            painter.setFont(font(11))
            painter.setPen(QColor(FAINT))
            painter.drawText(self.rect(), Qt.AlignCenter, self.empty_text)
            return
        cam, frame = self.camera, self.frame
        has_caption = self.demo or self.captions is None or bool(self.captions[0])
        top, bottom = (26, 22) if self.height() > 220 and has_caption else (24, 4)
        scale = (
            min(
                (self.width() - 16) / cam.width,
                (self.height() - top - bottom) / cam.height,
            )
            * self.zoom
        )
        width, height = cam.width * scale, cam.height * scale
        self.bounds = QRectF(
            (self.width() - width) / 2 + self.pan.x(),
            top + (self.height() - top - bottom - height) / 2 + self.pan.y(),
            width,
            height,
        )
        painter.save()
        painter.setClipRect(QRectF(0, top, self.width(), self.height() - top - bottom))
        image = frame.image if frame is not None else None
        if image is not None:
            # Frames arrive decoded near view size; smooth scaling is then cheap.
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawImage(self.bounds, image)
        else:
            painter.fillRect(self.bounds, QColor("#0b0b0c"))
        painter.translate(self.bounds.topLeft())
        painter.scale(scale, scale)
        area = QRectF(0, 0, cam.width, cam.height)
        painter.setClipRect(area, Qt.IntersectClip)
        if image is None and self.demo:
            self._demo_background(painter, cam, frame.timestamp if frame else 0.0)
        if self.show_grid and frame is not None and len(frame.points):
            grid = coverage(frame.points, cam.width, cam.height)
            for y in range(6):
                for x in range(8):
                    if grid[y, x]:
                        cell = QRectF(
                            x * cam.width / 8,
                            y * cam.height / 6,
                            cam.width / 8,
                            cam.height / 6,
                        )
                        painter.fillRect(
                            cell, QColor(10, 132, 255, min(40, int(grid[y, x]) * 5))
                        )
            painter.setPen(QPen(QColor(255, 255, 255, 18), 0.6 / scale))
            for x in range(9):
                painter.drawLine(
                    QPointF(x * cam.width / 8, 0),
                    QPointF(x * cam.width / 8, cam.height),
                )
            for y in range(7):
                painter.drawLine(
                    QPointF(0, y * cam.height / 6),
                    QPointF(cam.width, y * cam.height / 6),
                )
        if frame is not None and len(frame.points):
            painter.setRenderHint(QPainter.Antialiasing, True)
            if self.show_trails and len(self.history) > 1 and len(frame.points) <= 600:
                painter.setPen(QPen(QColor(10, 132, 255, 110), 0.9 / scale))
                history = list(self.history)
                for feature in frame.points:
                    fid = int(feature["id"])
                    path = QPainterPath()
                    drawing = False
                    for sample in history:
                        if fid in sample:
                            point = QPointF(*sample[fid])
                            if drawing:
                                path.lineTo(point)
                            else:
                                path.moveTo(point)
                                drawing = True
                        else:
                            drawing = False
                    painter.drawPath(path)
            from .paint import polygon

            pen = QPen(QColor("#30d158"), 4.0 / scale)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawPoints(polygon(frame.points["u"], frame.points["v"]))
        painter.restore()
        name = self.title or f"Camera {frame.cam if frame else 0:02d}"
        painter.setFont(font(8, True))
        painter.setPen(QColor(TEXT))
        room = self.width() - 16
        name_width = painter.fontMetrics().width(name)
        painter.drawText(
            QRectF(8, 2, room, top - 4), Qt.AlignLeft | Qt.AlignVCenter, name
        )
        painter.setFont(font(8))
        painter.setPen(QColor(MUTED))
        parts = [f"{cam.width}×{cam.height}"]
        if frame is not None and len(frame.points):
            parts.insert(0, f"{len(frame.points)} features")
        if self.zoom != 1.0:
            parts.append(f"{self.zoom:.1f}×")
        metrics = painter.fontMetrics()
        while parts and name_width + 14 + metrics.width(" · ".join(parts)) > room:
            parts.pop()  # drop details rather than overlap the camera name
        if parts:
            painter.drawText(
                QRectF(8, 2, room, top - 4),
                Qt.AlignRight | Qt.AlignVCenter,
                " · ".join(parts),
            )
        if bottom > 10:
            if self.demo:
                caption = "Synthetic preview · open a recording or connect your sensors"
            elif self.captions is not None:
                caption = self.captions[0]
            elif image is not None:
                caption = "Live image"
            else:
                caption = (
                    "Feature replay · session records contain observations, not images"
                )
            painter.setPen(QColor(FAINT))
            painter.drawText(
                QRectF(8, self.height() - bottom, self.width() - 16, bottom),
                Qt.AlignLeft | Qt.AlignVCenter,
                caption,
            )

    def _demo_background(self, painter, camera, t):
        # A code-native spatial backdrop, labelled synthetic in the viewport.
        w, h = camera.width, camera.height
        vanishing = QPointF(w * 0.52 + 10 * math.sin(t * 0.4), h * 0.42)
        painter.fillRect(QRectF(0, 0, w, h), QColor(CANVAS))
        painter.setPen(QPen(QColor(RAISED), 1.0))
        for x in np.linspace(-w, w * 2, 17):
            painter.drawLine(vanishing, QPointF(float(x), h))
        for y in (0.51, 0.59, 0.7, 0.85):
            painter.drawLine(QPointF(0, h * y), QPointF(w, h * y))
        painter.setPen(QPen(QColor("#48484a"), 1.2))
        for x, y, bw, bh in (
            (0.07, 0.16, 0.2, 0.37),
            (0.32, 0.21, 0.2, 0.28),
            (0.68, 0.11, 0.24, 0.41),
        ):
            box = QRectF(w * x, h * y, w * bw, h * bh)
            painter.fillRect(box, QColor("#242426"))
            painter.drawRect(box)
            for a in (0.25, 0.5, 0.75):
                painter.drawLine(
                    QPointF(box.left(), box.top() + box.height() * a),
                    QPointF(box.right(), box.top() + box.height() * a),
                )
            painter.drawLine(
                QPointF(box.center().x(), box.top()),
                QPointF(box.center().x(), box.bottom()),
            )

    def wheelEvent(self, event):
        self.zoom = min(
            8.0,
            max(0.5, self.zoom * (1.12 if event.angleDelta().y() > 0 else 1 / 1.12)),
        )
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag = event.pos()

    def mouseReleaseEvent(self, event):
        self.drag = None

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            self.pan += QPointF(event.pos() - self.drag)
            self.drag = event.pos()
            self.update()
        elif (
            self.frame is not None
            and len(self.frame.points)
            and self.bounds.width() > 0
        ):
            x = (event.x() - self.bounds.x()) / self.bounds.width() * self.camera.width
            y = (
                (event.y() - self.bounds.y())
                / self.bounds.height()
                * self.camera.height
            )
            distances = (self.frame.points["u"] - x) ** 2 + (
                self.frame.points["v"] - y
            ) ** 2
            index = int(np.argmin(distances))
            if distances[index] < (10 / self.bounds.width() * self.camera.width) ** 2:
                point = self.frame.points[index]
                QToolTip.showText(
                    event.globalPos(),
                    f"Feature {point['id']}  ·  ({point['u']:.1f}, {point['v']:.1f}) px",
                    self,
                )

    def mouseDoubleClickEvent(self, event):
        self.reset_view()


class TelemetryPlot(QWidget):
    def __init__(self, title, units, columns, parent=None):
        super().__init__(parent)
        self.title, self.units, self.columns = title, units, columns
        self.data = np.empty((0, 8))
        self.setMinimumHeight(165)

    def set_data(self, data):
        self.data = data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(PANEL))
        painter.setFont(font(10, True))
        painter.setPen(QColor(TEXT))
        painter.drawText(17, 24, self.title)
        painter.setFont(font(8))
        painter.setPen(QColor(MUTED))
        painter.drawText(17, 42, self.units)
        area = QRectF(48, 57, self.width() - 64, self.height() - 84)
        values = self.data[:, self.columns] if len(self.data) else np.empty((0, 3))
        lo, hi = (
            (float(values.min()), float(values.max())) if len(values) else (-1.0, 1.0)
        )
        padding = max((hi - lo) * 0.2, 0.1)
        lo -= padding
        hi += padding
        painter.setFont(font(7, mono=True))
        for ratio in (0.0, 0.5, 1.0):
            y = area.bottom() - ratio * area.height()
            painter.setPen(QPen(QColor(LINE), 1))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(QColor(MUTED))
            painter.drawText(
                QRectF(0, y - 8, 40, 16),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{lo + ratio * (hi - lo):.1f}",
            )
        for index, color in enumerate(AXES):
            x = self.width() - 195 + index * 62
            painter.setPen(QColor(color))
            painter.drawText(x, 24, "XYZ"[index])
            painter.setFont(font(8, mono=True))
            painter.drawText(
                x + 13, 24, f"{values[-1, index]:+.2f}" if len(values) else "—"
            )
        if len(self.data) > 1:
            from .paint import polygon

            times = self.data[:, 0]
            end = float(times[-1])
            start = end - 8.0
            painter.save()
            painter.setClipRect(area)
            painter.setRenderHint(
                QPainter.Antialiasing, False
            )  # dense data: crisp and ~5x cheaper
            xs = area.left() + (times - start) / 8.0 * area.width()
            for index, color in enumerate(AXES):
                ys = area.bottom() - (values[:, index] - lo) / (hi - lo) * area.height()
                painter.setPen(QPen(QColor(color), 1.2))
                painter.drawPolyline(polygon(xs, ys))
            painter.restore()
        else:
            painter.setPen(QColor(MUTED))
            painter.drawText(area, Qt.AlignCenter, "Waiting for IMU data")
        painter.setFont(font(7, mono=True))
        painter.setPen(QColor(MUTED))
        painter.drawText(48, self.height() - 10, "−8 s")
        painter.drawText(self.width() - 42, self.height() - 10, "now")


def button(text, callback, symbol=None, primary=False):
    widget = QPushButton(text)
    if symbol:
        widget.setIcon(icon(symbol, "#ffffff" if primary else MUTED))
    if primary:
        widget.setObjectName("primary")
    if callback is not None:
        widget.clicked.connect(callback)
    return widget


def row(*widgets):
    layout = QHBoxLayout()
    layout.setSpacing(10)
    for widget in widgets:
        if widget is None:
            layout.addStretch()
        elif isinstance(widget, QWidget):
            layout.addWidget(widget)
        else:
            layout.addLayout(widget)
    return layout


SPEEDS = ("0.25×", "0.5×", "1×", "2×", "4×")


class TransportBar(QWidget):
    """Play / step / speed / loop controls; the window owns the playback state.

    Every page with a playhead gets one; all of them drive the same replay.
    """

    play = pyqtSignal()
    rewind = pyqtSignal()
    step = pyqtSignal(int)
    speed_changed = pyqtSignal(int)
    loop_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.rewind_button = button("", self.rewind.emit, "reset")
        self.rewind_button.setToolTip("Back to the start")
        self.back_button = button("‹", lambda: self.step.emit(-1))
        self.back_button.setToolTip("Previous frame (Left)")
        self.play_button = button("", self.play.emit, "play", True)
        self.play_button.setToolTip("Play / pause (Space)")
        self.forward_button = button("›", lambda: self.step.emit(1))
        self.forward_button.setToolTip("Next frame (Right)")
        for widget in (
            self.rewind_button,
            self.back_button,
            self.play_button,
            self.forward_button,
        ):
            widget.setFixedWidth(40)
            layout.addWidget(widget)
        self.time_label = label("00:00.000", 9, TEAL)
        self.time_label.setFont(font(9, mono=True))
        layout.addSpacing(6)
        layout.addWidget(self.time_label)
        self.duration_label = label("/ 00:00.000", 9, MUTED)
        self.duration_label.setFont(font(9, mono=True))
        layout.addWidget(self.duration_label)
        layout.addStretch()
        self.speed = QComboBox()
        self.speed.addItems(SPEEDS)
        self.speed.setCurrentIndex(2)
        self.speed.setFixedWidth(78)
        self.speed.currentIndexChanged.connect(self.speed_changed.emit)
        layout.addWidget(self.speed)
        self.loop = QCheckBox("Loop")
        self.loop.setChecked(True)
        self.loop.toggled.connect(self.loop_changed.emit)
        layout.addWidget(self.loop)
        self._playing = None

    def set_state(
        self, playing, time_text, duration_text, speed_index, loop, enabled=True
    ):
        if playing != self._playing:
            self._playing = playing
            self.play_button.setIcon(icon("pause" if playing else "play", "#ffffff"))
        self.time_label.setText(time_text)
        self.duration_label.setText(f"/ {duration_text}")
        self.speed.blockSignals(True)
        self.speed.setCurrentIndex(speed_index)
        self.speed.blockSignals(False)
        self.loop.blockSignals(True)
        self.loop.setChecked(loop)
        self.loop.blockSignals(False)
        for widget in (
            self.rewind_button,
            self.back_button,
            self.play_button,
            self.forward_button,
            self.speed,
            self.loop,
        ):
            widget.setEnabled(enabled)


def jpl_rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    q = q / np.linalg.norm(q)
    v, w = q[:3], q[3]
    x, y, z = v
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (2 * w * w - 1) * np.eye(3) - 2 * w * skew + 2 * np.outer(v, v)


class RigView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 240)
        self.cameras = []
        self.yaw, self.pitch, self.zoom = -0.65, 0.55, 1.0
        self.drag = None
        self.caption = "Seed geometry"

    def set_cameras(self, cameras, caption="Seed geometry"):
        self.cameras = []
        for camera in cameras:
            try:
                q = np.asarray(camera.q, dtype=float)
                p = np.asarray(camera.p, dtype=float)
                if (
                    q.shape == (4,)
                    and p.shape == (3,)
                    and np.isfinite(q).all()
                    and np.isfinite(p).all()
                    and np.linalg.norm(q) > 1.0e-9
                ):
                    self.cameras.append(camera)
            except (ValueError, TypeError):
                pass
        self.caption = caption
        self.update()

    def reset_view(self):
        self.yaw, self.pitch, self.zoom = -0.65, 0.55, 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(CANVAS))
        gradient.setColorAt(1, QColor(CANVAS))
        painter.fillRect(self.rect(), gradient)
        cy, sy, cp, sp = (
            math.cos(self.yaw),
            math.sin(self.yaw),
            math.cos(self.pitch),
            math.sin(self.pitch),
        )
        view = np.array([[cy, -sy, 0], [sy * sp, cy * sp, -cp], [sy * cp, cy * cp, sp]])
        extent = max(
            [0.14] + [float(np.linalg.norm(cam.p)) + 0.07 for cam in self.cameras]
        )
        scale = min(self.width(), self.height()) / (extent * 2.8) * self.zoom

        def project(v):
            q = view @ np.asarray(v, dtype=float)
            return QPointF(
                self.width() / 2 + q[0] * scale, self.height() * 0.55 + q[1] * scale
            )

        def line(a, b, color, width=1.0):
            painter.setPen(QPen(QColor(color), width))
            painter.drawLine(project(a), project(b))

        for t in np.linspace(-extent, extent, 17):
            line([t, -extent, 0], [t, extent, 0], RAISED)
            line([-extent, t, 0], [extent, t, 0], RAISED)
        for axis, color, name in zip(np.eye(3) * extent * 0.65, AXES, "XYZ"):
            line([0, 0, 0], axis, color, 1.8)
            painter.setPen(QColor(color))
            painter.setFont(font(9, True))
            painter.drawText(project(axis) + QPointF(6, -4), name)
        corners = (
            np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ]
            )
            * 0.012
        )
        for face in (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ):
            painter.setBrush(QColor(80, 122, 133, 130))
            painter.setPen(QPen(QColor(TEAL), 1.0))
            painter.drawPolygon(QPolygonF([project(corners[i]) for i in face]))
        painter.setFont(font(9, True))
        painter.setPen(QColor(TEAL))
        painter.drawText(project([0, 0, 0]) + QPointF(-55, 30), "IMU")
        for index, cam in enumerate(self.cameras):
            if not np.isfinite(cam.q).all() or np.linalg.norm(cam.q) < 1.0e-9:
                continue
            rotation = jpl_rotation(cam.q)
            position = -rotation.T @ np.asarray(cam.p)
            local = np.array(
                [
                    [-0.025, -0.018, 0.045],
                    [0.025, -0.018, 0.045],
                    [0.025, 0.018, 0.045],
                    [-0.025, 0.018, 0.045],
                ]
            )
            points = [position + rotation.T @ point for point in local]
            color = (BLUE, AMBER, TEAL)[index % 3]
            for point in points:
                line(position, point, color, 1.3)
            painter.setBrush(QColor(142, 171, 245, 25))
            painter.setPen(QPen(QColor(color), 1.4))
            painter.drawPolygon(QPolygonF([project(point) for point in points]))
            line([0, 0, 0], position, FAINT, 1.0)
            painter.setPen(QColor(color))
            painter.drawText(project(points[1]) + QPointF(8, -9), f"CAM {index:02d}")
        painter.setFont(font(8, mono=True))
        painter.setPen(QColor(MUTED))
        painter.drawText(18, 23, self.caption.upper() + "   /   IMU FRAME")
        painter.drawText(
            18,
            self.height() - 14,
            "drag to orbit  ·  scroll to zoom  ·  double-click to reset",
        )
        painter.drawText(self.width() - 105, 23, f"grid {extent / 8 * 100:.1f} cm")

    def mousePressEvent(self, event):
        self.drag = event.pos()

    def mouseReleaseEvent(self, event):
        self.drag = None

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            change = event.pos() - self.drag
            self.drag = event.pos()
            self.yaw += change.x() * 0.008
            self.pitch = min(1.5, max(-1.5, self.pitch + change.y() * 0.008))
            self.update()

    def wheelEvent(self, event):
        self.zoom = min(
            5.0, max(0.3, self.zoom * (1.1 if event.angleDelta().y() > 0 else 1 / 1.1))
        )
        self.update()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()


class CoverageView(QWidget):
    """Occupied 8x6 image cells: one camera, or one titled grid per camera side by side."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grids = [("", np.zeros((6, 8), dtype=int))]
        self.setMinimumHeight(96)

    @property
    def grid(self):
        return self.grids[0][1]

    def set_grid(self, grid):
        self.set_grids([("", grid)])

    def set_grids(self, grids):
        self.grids = list(grids) or [("", np.zeros((6, 8), dtype=int))]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        count = len(self.grids)
        gap = 10 if count > 1 else 0
        width = (self.width() - gap * (count - 1)) / count
        titled = any(name for name, _ in self.grids)
        top = 14 if titled else 0
        cell_w, cell_h = width / 8, (self.height() - top) / 6
        painter.setFont(font(7.5))
        for index, (name, grid) in enumerate(self.grids):
            left = index * (width + gap)
            if name:
                painter.setPen(QColor(MUTED))
                painter.drawText(
                    QRectF(left, 0, width, top - 2),
                    Qt.AlignLeft | Qt.AlignVCenter,
                    painter.fontMetrics().elidedText(name, Qt.ElideRight, int(width)),
                )
            painter.setPen(Qt.NoPen)
            for y in range(6):
                for x in range(8):
                    rect = QRectF(
                        left + x * cell_w + 1.5,
                        top + y * cell_h + 1.5,
                        cell_w - 3,
                        cell_h - 3,
                    )
                    amount = int(grid[y, x])
                    painter.setBrush(
                        QColor(10, 132, 255, min(200, 45 + amount * 18))
                        if amount
                        else QColor("#2f2f31")
                    )
                    painter.drawRoundedRect(rect, 2.0, 2.0)


class StageView(QWidget):
    stages = ("PREPARE", "SETTLE", "BOOTSTRAP", "COLLECT", "SOLVE", "VERIFY", "COMMIT")
    names = (
        "Track frames",
        "Still baseline",
        "Time + rotation bootstrap",
        "Collect motion windows",
        "Joint refinement",
        "Held-out verification",
        "Write accepted blocks",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = "READY"
        self.setMinimumHeight(178)

    def set_state(self, state):
        self.state = state
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        index = (
            self.stages.index(self.state)
            if self.state in self.stages
            else (len(self.stages) if self.state == "DONE" else -1)
        )
        row = self.height() / len(self.stages)
        for i, name in enumerate(self.names):
            y = (i + 0.5) * row
            color = TEAL if 0 <= i <= index else MUTED
            if i < len(self.stages) - 1:
                painter.setPen(QPen(QColor(TEAL if i < index else LINE), 1.0))
                painter.drawLine(QPointF(9, y + 8), QPointF(9, y + row - 8))
            painter.setPen(QPen(QColor(color), 1.3))
            painter.setBrush(QColor(TEAL) if 0 <= i < index else QColor(PANEL))
            painter.drawEllipse(QPointF(9, y), 4, 4)
            if i == index:
                painter.setBrush(QColor(TEAL))
                painter.drawEllipse(QPointF(9, y), 1.6, 1.6)
            painter.setFont(font(9, i == index))
            painter.setPen(QColor(TEXT if i == index else color))
            painter.drawText(QPointF(26, y + 4), name)


STYLE = f"""
QMainWindow, QDialog {{ background:{BG}; color:{TEXT}; }}
QWidget {{ color:{TEXT}; }}
QFrame#card {{ background:{PANEL}; border:none; border-radius:12px; }}
QFrame#sidebar {{ background:{SIDEBAR}; border-right:1px solid {LINE}; }}
QFrame#toast {{ background:#2c2c2e; border:1px solid #48484a; border-radius:12px; }}
QPushButton {{ background:{RAISED}; color:{TEXT}; border:none; border-radius:7px; padding:6px 12px; }}
QPushButton:hover {{ background:#48484a; }}
QPushButton:pressed {{ background:#545458; }}
QPushButton:disabled {{ color:{FAINT}; background:#2a2a2c; }}
QPushButton:checked {{ background:#545458; }}
QPushButton#primary {{ background:{ACCENT}; color:#ffffff; font-weight:600; }}
QPushButton#primary:hover {{ background:#2b95ff; }}
QPushButton#primary:pressed {{ background:#0071e3; }}
QPushButton#primary:disabled {{ background:#1d3a5f; color:#7d97b8; }}
QPushButton#danger {{ color:{RED}; }}
QPushButton#plain {{ background:transparent; color:{ACCENT}; padding:4px 6px; }}
QPushButton#plain:hover {{ background:rgba(255,255,255,0.06); }}
QPushButton#nav {{ background:transparent; color:{TEXT}; border-radius:7px; padding:7px 10px; text-align:left; }}
QPushButton#nav:hover {{ background:rgba(255,255,255,0.06); }}
QPushButton#nav:checked {{ background:rgba(10,132,255,0.24); color:#ffffff; }}
QPushButton#help {{ background:{RAISED}; color:{MUTED}; border-radius:11px; padding:0; min-width:22px; max-width:22px; min-height:22px; max-height:22px; font-weight:600; }}
QPushButton#help:hover {{ color:{TEXT}; background:#48484a; }}
QPushButton#segment {{ background:transparent; color:{TEXT}; border-radius:6px; padding:4px 12px; }}
QPushButton#segment:checked {{ background:#636366; }}
QPushButton#segment:hover:!checked {{ background:rgba(255,255,255,0.06); }}
QFrame#segmented {{ background:{RAISED}; border-radius:8px; }}
QPushButton::menu-indicator {{ width:0; image:none; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background:#1f1f21; border:1px solid {LINE}; border-radius:7px; padding:5px 8px; selection-background-color:{ACCENT}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border:1px solid {ACCENT}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color:{FAINT}; }}
QComboBox::drop-down {{ border:none; width:20px; }}
QComboBox QAbstractItemView {{ background:{PANEL}; border:1px solid #48484a; selection-background-color:{ACCENT}; outline:0; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width:0; border:none; }}
QCheckBox, QRadioButton {{ color:{TEXT}; spacing:8px; }}
QCheckBox::indicator {{ width:15px; height:15px; border:1px solid #545458; border-radius:4px; background:#1f1f21; }}
QCheckBox::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT}; }}
QCheckBox::indicator:disabled {{ background:#2a2a2c; border-color:#3a3a3c; }}
QSlider::groove:horizontal {{ height:4px; background:#48484a; border-radius:2px; }}
QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:2px; }}
QSlider::handle:horizontal {{ width:14px; height:14px; margin:-5px 0; background:#f5f5f7; border-radius:7px; }}
QTabWidget::pane {{ border:none; background:transparent; }}
QTabBar {{ qproperty-drawBase:0; }}
QTabBar::tab {{ background:transparent; color:{MUTED}; padding:5px 14px; margin:2px; border-radius:6px; }}
QTabBar::tab:selected {{ background:#48484a; color:{TEXT}; }}
QTabBar::tab:hover:!selected {{ color:{TEXT}; }}
QScrollArea {{ border:none; background:transparent; }}
QScrollArea > QWidget > QWidget {{ background:transparent; }}
QScrollBar:vertical {{ background:transparent; width:9px; margin:2px; }}
QScrollBar::handle:vertical {{ background:rgba(255,255,255,0.18); min-height:28px; border-radius:4px; }}
QScrollBar::handle:vertical:hover {{ background:rgba(255,255,255,0.30); }}
QScrollBar::add-line, QScrollBar::sub-line {{ height:0; width:0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background:transparent; }}
QScrollBar:horizontal {{ background:transparent; height:9px; margin:2px; }}
QScrollBar::handle:horizontal {{ background:rgba(255,255,255,0.18); min-width:28px; border-radius:4px; }}
QTreeWidget, QTreeView, QTableWidget, QTableView, QListWidget, QPlainTextEdit, QTextBrowser {{ background:{PANEL}; alternate-background-color:#303033; border:none; border-radius:10px; gridline-color:transparent; selection-background-color:rgba(10,132,255,0.38); selection-color:{TEXT}; outline:0; }}
QTreeWidget::item, QTableWidget::item, QListWidget::item {{ padding:3px 4px; }}
QHeaderView {{ background:transparent; }}
QHeaderView::section {{ background:transparent; color:{MUTED}; padding:6px 8px; border:none; border-bottom:1px solid {LINE}; font-weight:600; }}
QToolTip {{ background:{RAISED}; color:{TEXT}; border:1px solid #48484a; border-radius:6px; padding:6px 8px; }}
QStatusBar {{ background:{BG}; color:{MUTED}; border-top:1px solid {LINE}; }}
QStatusBar::item {{ border:none; }}
QMenu {{ background:#2c2c2e; border:1px solid #48484a; border-radius:8px; padding:4px; }}
QMenu::item {{ padding:6px 22px 6px 12px; border-radius:5px; }}
QMenu::item:selected {{ background:{ACCENT}; color:#ffffff; }}
QMenu::separator {{ height:1px; background:{LINE}; margin:4px 6px; }}
QSplitter::handle {{ background:transparent; }}
QDialogButtonBox QPushButton {{ min-width:72px; }}
"""


class Segmented(QFrame):
    """macOS-style segmented control. ``changed`` carries the selected value."""

    changed = pyqtSignal(object)

    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.setObjectName("segmented")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.buttons = []
        for text, value in items:
            b = QPushButton(text)
            b.setObjectName("segment")
            b.setCheckable(True)
            b.clicked.connect(
                lambda checked=False, v=value: self.set_value(v, emit=True)
            )
            b.value = value
            layout.addWidget(b)
            self.buttons.append(b)
        if self.buttons:
            self.buttons[0].setChecked(True)

    def value(self):
        return next((b.value for b in self.buttons if b.isChecked()), None)

    def set_value(self, value, emit=False):
        for b in self.buttons:
            b.setChecked(b.value == value)
        if emit:
            self.changed.emit(value)

    def set_enabled_value(self, value, enabled):
        for b in self.buttons:
            if b.value == value:
                b.setEnabled(enabled)


def help_button(callback, tip="Help for this page"):
    widget = QPushButton("?")
    widget.setObjectName("help")
    widget.setToolTip(tip)
    widget.clicked.connect(callback)
    return widget


def plain_button(text, callback):
    widget = QPushButton(text)
    widget.setObjectName("plain")
    widget.clicked.connect(callback)
    return widget


class Toast(QFrame):
    """Transient notification at the bottom of its parent (errors stay until dismissed)."""

    def __init__(self, parent):
        super().__init__(parent)
        from PyQt5.QtCore import QTimer

        self.setObjectName("toast")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 9, 10, 9)
        layout.setSpacing(10)
        self.dot = QLabel()
        self.dot.setFixedSize(8, 8)
        layout.addWidget(self.dot)
        self.text = label("", 10)
        self.text.setWordWrap(True)
        layout.addWidget(self.text, 1)
        close = QPushButton("✕")
        close.setObjectName("plain")
        close.setStyleSheet(f"color:{MUTED}; padding:0 4px;")
        close.clicked.connect(self.hide)
        layout.addWidget(close)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)
        self.hide()

    def show_message(self, text, kind="info"):
        color = {"error": RED, "warning": AMBER, "success": GREEN}.get(kind, ACCENT)
        self.dot.setStyleSheet(f"background:{color}; border-radius:4px;")
        self.text.setText(text)
        self.place()
        self.show()
        self.raise_()
        self.timer.start(9000 if kind == "error" else 4500)

    def place(self):
        parent = self.parentWidget()
        width = min(720, max(360, parent.width() - 260))
        self.setFixedWidth(width)
        self.adjustSize()
        self.move(
            (parent.width() - width) // 2 + 90, parent.height() - self.height() - 44
        )


class EmptyState(QWidget):
    """Centered title, explanation and one call to action."""

    def __init__(self, symbol, title, text, action=None, action_text="", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addStretch()
        glyph = QLabel()
        glyph.setPixmap(icon(symbol, FAINT, 44).pixmap(44, 44))
        glyph.setAlignment(Qt.AlignCenter)
        layout.addWidget(glyph)
        heading = label(title, 15, TEXT, True)
        heading.setAlignment(Qt.AlignCenter)
        layout.addWidget(heading)
        body = label(text, 10, MUTED)
        body.setAlignment(Qt.AlignCenter)
        body.setWordWrap(True)
        body.setMaximumWidth(460)
        layout.addWidget(body, 0, Qt.AlignHCenter)
        if action is not None:
            go = button(action_text, action, None, True)
            layout.addSpacing(6)
            layout.addWidget(go, 0, Qt.AlignHCenter)
        layout.addStretch()


class CameraGrid(QWidget):
    """Every camera at once: a responsive grid of FeatureViews with titles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from PyQt5.QtWidgets import QGridLayout

        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        self.views = []
        self.names = []

    def set_cameras(self, names, captions=None):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.views, self.names = [], list(names)
        count = len(self.names)
        columns = 1 if count <= 1 else 2 if count <= 4 else 3
        for index, name in enumerate(self.names):
            view = FeatureView()
            view.setMinimumSize(160, 110)
            view.title = name
            view.captions = captions
            self.views.append(view)
            self.grid.addWidget(view, index // columns, index % columns)
        for c in range(columns):
            self.grid.setColumnStretch(c, 1)
