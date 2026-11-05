"""3D VIO view (QPainter): trajectory, pose, camera frusta and feature map.

Compared with the classic OpenVINS RViz layout this view is time-aware: the
estimate is drawn up to the shared playhead (the rest of the run stays as a
faint ghost), so seeking the log timeline replays the filter's history.
"""

import math

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QLinearGradient, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget

from .logviews import RED, nice_ticks
from .paint import points as qpoints
from .widgets import AMBER, BLUE, LINE, MUTED, TEAL, TEXT, font, jpl_rotation

AXIS_COLORS = ("#f2877e", "#80e0c1", "#8eabf5")


class VioView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 320)
        self.setMouseTracking(True)
        self.yaw, self.pitch, self.scale = -0.6, 0.75, 40.0
        self.target = np.zeros(3)
        self.follow = True
        self.show_map = True
        self.show_ghost = True
        self.times = np.empty(0)
        self.positions = np.empty((0, 3))
        self.rotations = []  # R_ItoG per state
        self.cameras = []  # per state: [(R_CtoG, p_CinG)]
        self.points = None  # (xyz, kind) at the playhead
        self.map_points = np.empty((0, 3))
        self.index = -1
        self.drag = None
        self.status = ""

    # -- data ---------------------------------------------------------------
    def set_run(self, times, positions, rotations, cameras):
        self.times = np.asarray(times, float)
        self.positions = np.asarray(positions, float).reshape(-1, 3)
        self.rotations, self.cameras = rotations, cameras
        self.update()

    def set_points(self, points, map_points=None):
        self.points = points
        if map_points is not None:
            self.map_points = map_points
        self.update()

    def set_index(self, index):
        self.index = index
        if self.follow and 0 <= index < len(self.positions):
            self.target = self.positions[index].copy()
        self.update()

    def fit(self):
        if not len(self.positions):
            return
        lo, hi = self.positions.min(axis=0), self.positions.max(axis=0)
        self.target = (lo + hi) / 2
        extent = max(float(np.max(hi - lo)), 0.5)
        self.scale = 0.75 * min(self.width(), self.height()) / extent
        self.follow = False
        self.update()

    def top_view(self):
        self.pitch, self.yaw = math.pi / 2 - 1e-3, -math.pi / 2
        self.update()

    def reset_view(self):
        self.yaw, self.pitch = -0.6, 0.75
        self.fit()

    # -- projection -------------------------------------------------------------
    def project(self, points):
        rel = np.atleast_2d(points) - self.target
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x1 = cy * rel[:, 0] + sy * rel[:, 1]
        y1 = -sy * rel[:, 0] + cy * rel[:, 1]
        z1 = rel[:, 2]
        sx = x1
        syv = z1 * cp + y1 * sp
        center = QPointF(self.width() / 2, self.height() / 2 + 10)
        return np.column_stack(
            (center.x() + self.scale * sx, center.y() - self.scale * syv)
        )

    # -- painting ---------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#101b25"))
        gradient.setColorAt(1, QColor("#0a1118"))
        painter.fillRect(self.rect(), gradient)
        self._grid(painter)
        if not len(self.positions):
            painter.setPen(QColor(MUTED))
            painter.setFont(font(11))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                self.status or "Run VIO to see the trajectory",
            )
            return
        index = (
            self.index
            if 0 <= self.index < len(self.positions)
            else len(self.positions) - 1
        )
        if self.show_map and len(self.map_points):
            self._dots(painter, self.map_points, QColor(142, 171, 245, 55), 1.4)
        if self.points is not None:
            xyz, kind = self.points
            if len(xyz):
                self._dots(painter, xyz[kind == 0], QColor(142, 171, 245, 170), 2.0)
                self._dots(painter, xyz[kind == 1], QColor(AMBER), 3.2)
                aruco = xyz[kind == 2]
                if len(aruco):
                    self._dots(painter, aruco, QColor(RED), 4.0)
        screen = self.project(self.positions)
        if self.show_ghost and index < len(screen) - 1:
            painter.setPen(QPen(QColor(255, 255, 255, 28), 1.2))
            painter.drawPolyline(qpoints(screen[index:]))
        if index > 0:
            segments = max(1, min(64, index))
            edges = np.linspace(0, index, segments + 1).astype(int)
            start, end = QColor(BLUE), QColor(TEAL)
            for k in range(segments):
                a, b = edges[k], edges[k + 1]
                if b <= a:
                    continue
                f = k / max(1, segments - 1)
                color = QColor(
                    int(start.red() + f * (end.red() - start.red())),
                    int(start.green() + f * (end.green() - start.green())),
                    int(start.blue() + f * (end.blue() - start.blue())),
                )
                painter.setPen(
                    QPen(color, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                )
                painter.drawPolyline(qpoints(screen[a : b + 1]))
        if index < len(self.rotations):
            self._pose(painter, self.positions[index], self.rotations[index])
        if index < len(self.cameras):
            for R_CtoG, p_CinG in self.cameras[index]:
                self._frustum(painter, R_CtoG, p_CinG)
        self._legend(painter, index)

    def _dots(self, painter, xyz, color, size):
        if not len(xyz):
            return
        pts = self.project(xyz)
        keep = (
            (pts[:, 0] > -10)
            & (pts[:, 0] < self.width() + 10)
            & (pts[:, 1] > -10)
            & (pts[:, 1] < self.height() + 10)
        )
        painter.setPen(QPen(color, size, Qt.SolidLine, Qt.RoundCap))
        painter.drawPoints(qpoints(pts[keep]))

    def _grid(self, painter):
        span = max(self.width(), self.height()) / max(self.scale, 1e-6)
        ticks = nice_ticks(-span, span, 12)
        step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
        z = float(self.positions[:, 2].min()) - 0.05 if len(self.positions) else 0.0
        cx, cy = self.target[0], self.target[1]
        base = np.array([math.floor(cx / step) * step, math.floor(cy / step) * step])
        n = int(min(40, span / step)) + 1
        k = np.arange(-n, n + 1) * step
        far = n * step
        # Every grid line's two endpoints, projected in one call.
        ends = np.concatenate(
            [
                np.column_stack(
                    (base[0] + k, np.full_like(k, base[1] - far), np.full_like(k, z))
                ),
                np.column_stack(
                    (base[0] + k, np.full_like(k, base[1] + far), np.full_like(k, z))
                ),
                np.column_stack(
                    (np.full_like(k, base[0] - far), base[1] + k, np.full_like(k, z))
                ),
                np.column_stack(
                    (np.full_like(k, base[0] + far), base[1] + k, np.full_like(k, z))
                ),
            ]
        )
        screen = self.project(ends).reshape(4, len(k), 2)
        lines = []
        for a, b in ((0, 1), (2, 3)):
            for p0, p1 in zip(screen[a], screen[b]):
                lines += [QPointF(*p0), QPointF(*p1)]
        painter.setPen(QPen(QColor(LINE), 0.8))
        painter.drawLines(lines)
        origin = self.project(np.array([[0, 0, 0.0]]))[0]
        for axis, color in enumerate(AXIS_COLORS):
            tip = np.zeros(3)
            tip[axis] = step
            end = self.project(tip[None, :])[0]
            painter.setPen(QPen(QColor(color), 1.2))
            painter.drawLine(QPointF(*origin), QPointF(*end))
        self._grid_step = step

    def _pose(self, painter, p, R_ItoG):
        length = 45.0 / max(self.scale, 1e-6)
        origin = self.project(p[None, :])[0]
        for axis, color in enumerate(AXIS_COLORS):
            end = self.project((p + R_ItoG[:, axis] * length)[None, :])[0]
            painter.setPen(QPen(QColor(color), 2.4, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(*origin), QPointF(*end))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(TEXT))
        painter.drawEllipse(QPointF(*origin), 3, 3)

    def _frustum(self, painter, R_CtoG, p_CinG):
        depth = 32.0 / max(self.scale, 1e-6)
        corners = [
            np.array([x, y, 1.0]) * depth
            for x, y in ((-0.6, -0.4), (0.6, -0.4), (0.6, 0.4), (-0.6, 0.4))
        ]
        world = [p_CinG + R_CtoG @ c for c in corners]
        apex = self.project(p_CinG[None, :])[0]
        pts = [self.project(w[None, :])[0] for w in world]
        painter.setPen(QPen(QColor(TEAL), 1.1))
        painter.setBrush(Qt.NoBrush)
        for q in pts:
            painter.drawLine(QPointF(*apex), QPointF(*q))
        painter.drawPolygon(QPolygonF([QPointF(*q) for q in pts]))
        up = p_CinG + R_CtoG @ (np.array([0, -0.55, 1.0]) * depth)
        painter.drawLine(QPointF(*pts[0]), QPointF(*self.project(up[None, :])[0]))
        painter.drawLine(QPointF(*pts[1]), QPointF(*self.project(up[None, :])[0]))

    def _legend(self, painter, index):
        painter.setFont(font(8))
        painter.setPen(QColor(MUTED))
        t = self.times[index] if index < len(self.times) else 0.0
        distance = (
            float(
                np.sum(
                    np.linalg.norm(np.diff(self.positions[: index + 1], axis=0), axis=1)
                )
            )
            if index > 0
            else 0.0
        )
        p = self.positions[index]
        painter.drawText(
            QRectF(14, 10, 520, 18),
            Qt.AlignLeft,
            f"t {t:8.3f} s   p [{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f}] m   path {distance:.2f} m   grid {getattr(self, '_grid_step', 1):g} m",
        )
        painter.setPen(QColor(MUTED))
        items = (("trajectory", TEAL), ("SLAM", AMBER), ("MSCKF", BLUE), ("ArUco", RED))
        x = 14
        y = self.height() - 14
        for text, color in items:
            painter.setPen(QPen(QColor(color), 3))
            painter.drawPoint(QPointF(x, y - 4))
            painter.setPen(QColor(MUTED))
            painter.drawText(QPointF(x + 7, y), text)
            x += painter.fontMetrics().width(text) + 22
        painter.drawText(
            QRectF(self.width() - 330, self.height() - 22, 320, 18),
            Qt.AlignRight,
            "drag: orbit · shift/right-drag: pan · wheel: zoom · double-click: fit",
        )

    # -- interaction ------------------------------------------------------------
    def mousePressEvent(self, event):
        pan = (
            event.button() in (Qt.RightButton, Qt.MiddleButton)
            or event.modifiers() & Qt.ShiftModifier
        )
        self.drag = (event.pos(), self.yaw, self.pitch, self.target.copy(), bool(pan))

    def mouseMoveEvent(self, event):
        if not self.drag:
            return
        start, yaw, pitch, target, pan = self.drag
        delta = event.pos() - start
        if pan:
            self.follow = False
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            dx, dy = -delta.x() / self.scale, delta.y() / self.scale
            # Screen x is the yawed x axis; screen y moves along the yawed y axis (ground plane pan).
            self.target = target + np.array([cy * dx - sy * dy, sy * dx + cy * dy, 0.0])
        else:
            self.yaw = yaw - delta.x() * 0.01
            self.pitch = float(
                np.clip(pitch + delta.y() * 0.01, -0.2, math.pi / 2 - 1e-3)
            )
        self.update()

    def mouseReleaseEvent(self, event):
        self.drag = None

    def wheelEvent(self, event):
        self.scale = float(
            np.clip(self.scale * 1.15 ** (event.angleDelta().y() / 120.0), 0.05, 50000)
        )
        self.update()

    def mouseDoubleClickEvent(self, event):
        self.fit()


def camera_poses(R_ItoG, p_IinG, calib):
    """[(R_CtoG, p_CinG)] from JPL q_ItoC / p_IinC per camera."""
    poses = []
    for q_ItoC, p_IinC, *_ in calib:
        R_ItoC = jpl_rotation(q_ItoC)
        p_CinI = -R_ItoC.T @ np.asarray(p_IinC, float)
        poses.append((R_ItoG @ R_ItoC.T, p_IinG + R_ItoG @ p_CinI))
    return poses
