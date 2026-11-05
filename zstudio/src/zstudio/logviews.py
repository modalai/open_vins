"""Whole-log views: a shared time range, an event timeline, and envelope plots.

QPainter-only (no plotting dependency). Performance contract: the static layer
(axes, lanes, data) is rendered into a cached pixmap and rebuilt only when the
visible range, size or data change; moving the playhead only composites a line.
Plots decimate to a min/max pair per pixel column, so spikes stay visible at
every zoom level, and draw dense data without antialiasing.
"""

import math

import numpy as np
from PyQt5.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QToolTip, QWidget

from .paint import LayerCache, polygon
from .widgets import (
    ACCENT,
    AMBER,
    BLUE,
    FAINT,
    GREEN,
    LINE,
    MUTED,
    PANEL,
    PURPLE,
    RED,
    TEXT,
    font,
)

SEVERITY_COLORS = {
    "problem": RED,
    "warning": AMBER,
    "info": BLUE,
    "expected": GREEN,
    "ok": GREEN,
    "excluded": FAINT,
}
SERIES_COLORS = (ACCENT, GREEN, AMBER, PURPLE, BLUE, "#ffd60a", RED, "#ac8e68")


def envelope(t, y, t0, t1, bins):
    """Visible samples of (t, y) reduced to a min/max pair per bin; returns (tt, yy)."""
    t = np.asarray(t)
    lo, hi = np.searchsorted(t, [t0, t1])
    lo, hi = max(0, lo - 1), min(len(t), hi + 1)
    t, y = t[lo:hi], np.asarray(y, dtype=float)[lo:hi]
    finite = np.isfinite(y)
    t, y = t[finite], y[finite]
    if len(t) <= 2 * bins or bins < 1:
        return t, y
    edges = np.linspace(0, len(t), bins + 1).astype(int)
    starts, stops = edges[:-1], edges[1:]
    mins = np.minimum.reduceat(y, starts)
    maxs = np.maximum.reduceat(y, starts)
    # Keep the temporal order of each bin's extremes (vectorised first argmin/argmax per bin).
    segment = np.repeat(np.arange(bins), stops - starts)
    index = np.arange(len(y))
    first_min = np.full(bins, len(y))
    first_max = np.full(bins, len(y))
    is_min = y == mins[segment]
    is_max = y == maxs[segment]
    np.minimum.at(first_min, segment[is_min], index[is_min])
    np.minimum.at(first_max, segment[is_max], index[is_max])
    min_first = first_min <= first_max
    tt = np.empty(2 * bins)
    yy = np.empty(2 * bins)
    tt[0::2], tt[1::2] = t[starts], t[stops - 1]
    yy[0::2] = np.where(min_first, mins, maxs)
    yy[1::2] = np.where(min_first, maxs, mins)
    return tt, yy


def nice_ticks(lo, hi, target=6):
    span = hi - lo
    if span <= 0 or not math.isfinite(span):
        return []
    raw = span / max(1, target)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step) + 1)]


def time_label(t, step):
    digits = 0 if step >= 1 else min(3, max(0, -int(math.floor(math.log10(step)))))
    return f"{t:.{digits}f}s"


class TimeRange(QObject):
    """Shared visible window (``changed``) and playhead (``moved``), stream time [s]."""

    changed = pyqtSignal()
    moved = pyqtSignal()
    seek = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lo, self.hi = 0.0, 1.0
        self.t0, self.t1 = 0.0, 1.0
        self.playhead = 0.0

    def set_bounds(self, lo, hi):
        self.lo, self.hi = float(lo), float(max(hi, lo + 1e-3))
        self.t0, self.t1 = self.lo, self.hi
        self.changed.emit()

    def set_view(self, t0, t1):
        span = min(max(t1 - t0, 1e-3), self.hi - self.lo)
        t0 = min(max(t0, self.lo), self.hi - span)
        if (t0, t0 + span) != (self.t0, self.t1):
            self.t0, self.t1 = t0, t0 + span
            self.changed.emit()

    def zoom(self, center, factor):
        self.set_view(
            center - (center - self.t0) * factor, center + (self.t1 - center) * factor
        )

    def pan(self, dt):
        self.set_view(self.t0 + dt, self.t1 + dt)

    def reset(self):
        self.set_view(self.lo, self.hi)

    def set_playhead(self, t):
        t = float(t)
        if t != self.playhead:
            self.playhead = t
            self.moved.emit()


class _TimeWidget(QWidget):
    """Shared x mapping and interaction: wheel zoom, drag pan, click seek, double-click reset."""

    gutter = 64

    def __init__(self, time_range, parent=None):
        super().__init__(parent)
        self.range = time_range
        self.layer = LayerCache(self, self.render_static)
        self.range.changed.connect(self.invalidate)
        self.range.moved.connect(self._playhead_moved)
        self.setMouseTracking(True)
        self._press = None
        self._drawn_playhead = None

    def invalidate(self):
        self.layer.invalidate()
        self.update()

    def resizeEvent(self, event):
        self.layer.invalidate()
        super().resizeEvent(event)

    def _playhead_moved(self):
        # Repaint only narrow strips under the old and the new playhead.
        for t in (self._drawn_playhead, self.range.playhead):
            if t is not None:
                self.update(int(self.x_of(t)) - 3, 0, 7, self.height())

    def area(self):
        return QRectF(
            self.gutter,
            8,
            max(10, self.width() - self.gutter - 12),
            max(10, self.height() - 30),
        )

    def x_of(self, t):
        area = self.area()
        return (
            area.left()
            + (t - self.range.t0)
            / max(self.range.t1 - self.range.t0, 1e-9)
            * area.width()
        )

    def t_of(self, x):
        area = self.area()
        return self.range.t0 + (x - area.left()) / max(area.width(), 1) * (
            self.range.t1 - self.range.t0
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        self.layer.draw(painter)
        area = self.area()
        x = self.x_of(self.range.playhead)
        if area.left() <= x <= area.right():
            painter.setPen(QPen(QColor(ACCENT), 1.5))
            painter.drawLine(QPointF(x, area.top() - 4), QPointF(x, area.bottom()))
        self._drawn_playhead = self.range.playhead

    def render_static(self, painter):
        raise NotImplementedError

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.range.zoom(self.t_of(event.pos().x()), 0.8**steps)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press = (event.pos(), self.range.t0, False)

    def mouseMoveEvent(self, event):
        if self._press and event.buttons() & Qt.LeftButton:
            start, t0, moved = self._press
            dx = event.pos().x() - start.x()
            if moved or abs(dx) > 3:
                self._press = (start, t0, True)
                span = self.range.t1 - self.range.t0
                shift = dx / max(self.area().width(), 1) * span
                self.range.set_view(t0 - shift, t0 - shift + span)
        else:
            self.hover(event)

    def mouseReleaseEvent(self, event):
        if self._press and event.button() == Qt.LeftButton and not self._press[2]:
            if self.area().left() <= event.pos().x() <= self.area().right():
                self.range.seek.emit(self.t_of(event.pos().x()))
        self._press = None

    def mouseDoubleClickEvent(self, event):
        self.range.reset()

    def hover(self, event):
        pass

    def draw_axis(self, painter, area, bottom=True):
        ticks = nice_ticks(
            self.range.t0, self.range.t1, max(2, int(area.width() / 110))
        )
        step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
        painter.setFont(font(7.5))
        for t in ticks:
            x = self.x_of(t)
            painter.setPen(QPen(QColor(LINE), 1))
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            painter.setPen(QColor(MUTED))
            y = area.bottom() + 14 if bottom else area.top() - 4
            painter.drawText(
                QRectF(x - 40, y - 10, 80, 14), Qt.AlignCenter, time_label(t, step)
            )


def _vertical_lines(xs, top, bottom):
    return [p for x in xs for p in (QPointF(float(x), top), QPointF(float(x), bottom))]


class TimelineView(_TimeWidget):
    """One lane per channel: data span plus findings' events colored by severity."""

    lane_clicked = pyqtSignal(str)
    gutter = 190
    lane_height = 24

    def __init__(self, time_range, parent=None):
        super().__init__(time_range, parent)
        self.lanes = []
        self.selected = ""
        self.setMinimumHeight(80)

    def set_lanes(self, lanes):
        """lanes: [{name, label, start, end, status, events: [(t, severity, text)]}]"""
        self.lanes = lanes
        for lane in lanes:
            events = lane.get("events", [])
            lane["_t"] = np.array([e[0] for e in events], dtype=float)
            lane["_sev"] = np.array([e[1] for e in events], dtype=object)
        self.setMinimumHeight(30 + self.lane_height * max(1, len(lanes)))
        self.invalidate()

    def select(self, name):
        if name != self.selected:
            self.selected = name
            self.invalidate()

    def area(self):
        return QRectF(
            self.gutter,
            22,
            max(10, self.width() - self.gutter - 12),
            self.lane_height * max(1, len(self.lanes)),
        )

    def render_static(self, painter):
        painter.fillRect(QRectF(0, 0, self.width(), self.height()), QColor(PANEL))
        area = self.area()
        self.draw_axis(painter, area, bottom=False)
        span = max(self.range.t1 - self.range.t0, 1e-9)
        for i, lane in enumerate(self.lanes):
            top = area.top() + i * self.lane_height
            painter.setRenderHint(QPainter.Antialiasing, True)
            if lane["name"] == self.selected:
                painter.fillRect(
                    QRectF(0, top, self.width(), self.lane_height),
                    QColor(255, 255, 255, 14),
                )
            painter.setPen(Qt.NoPen)
            painter.setBrush(
                QColor(SEVERITY_COLORS.get(lane.get("status", "ok"), MUTED))
            )
            painter.drawEllipse(QPointF(12, top + self.lane_height / 2), 3.5, 3.5)
            painter.setPen(QColor(TEXT if lane.get("included", True) else MUTED))
            painter.setFont(font(8.5))
            text = painter.fontMetrics().elidedText(
                lane.get("label", lane["name"]), Qt.ElideMiddle, self.gutter - 30
            )
            painter.drawText(
                QRectF(22, top, self.gutter - 26, self.lane_height),
                Qt.AlignVCenter | Qt.AlignLeft,
                text,
            )
            if lane.get("start") is not None and lane.get("end") is not None:
                x0 = max(area.left(), self.x_of(lane["start"]))
                x1 = min(area.right(), self.x_of(lane["end"]))
                if x1 > x0:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor("#3a3a3c"))
                    painter.drawRoundedRect(
                        QRectF(x0, top + 8, x1 - x0, self.lane_height - 16), 4, 4
                    )
            if not len(lane["_t"]):
                continue
            painter.setRenderHint(QPainter.Antialiasing, False)
            xs = (lane["_t"] - self.range.t0) / span * area.width() + area.left()
            for severity in ("expected", "info", "warning", "problem"):
                mask = lane["_sev"] == severity
                if not mask.any():
                    continue
                cols = np.unique(np.round(xs[mask]).astype(int))
                cols = cols[(cols >= area.left()) & (cols <= area.right())]
                if len(cols):
                    width = 2 if severity in ("problem", "warning") else 1.5
                    painter.setPen(QPen(QColor(SEVERITY_COLORS[severity]), width))
                    painter.drawLines(
                        _vertical_lines(cols, top + 4, top + self.lane_height - 4)
                    )

    def _lane_at(self, y):
        i = int((y - self.area().top()) // self.lane_height)
        return self.lanes[i] if 0 <= i < len(self.lanes) else None

    def hover(self, event):
        lane = self._lane_at(event.pos().y())
        if lane is None:
            QToolTip.hideText()
            return
        if event.pos().x() < self.area().left():
            QToolTip.showText(
                event.globalPos(), lane.get("tooltip", lane["name"]), self
            )
            return
        events = lane.get("events", [])
        if not events:
            QToolTip.hideText()
            return
        area = self.area()
        span = max(self.range.t1 - self.range.t0, 1e-9)
        xs = (lane["_t"] - self.range.t0) / span * area.width() + area.left()
        i = int(np.argmin(np.abs(xs - event.pos().x())))
        if abs(xs[i] - event.pos().x()) <= 5:
            t, severity, text = events[i][:3]
            QToolTip.showText(
                event.globalPos(), f"{severity} · t={t:.3f}s\n{text}", self
            )
        else:
            QToolTip.hideText()

    def mouseReleaseEvent(self, event):
        lane = self._lane_at(event.pos().y())
        if lane is not None and self._press and not self._press[2]:
            self.select(lane["name"])
            self.lane_clicked.emit(lane["name"])
        super().mouseReleaseEvent(event)


class LogPlot(_TimeWidget):
    """Multi-series time plot over the shared range with envelope decimation."""

    def __init__(self, time_range, parent=None):
        super().__init__(time_range, parent)
        self.series = []
        self.markers = []
        self.title = ""
        self.units = ""
        self.message = "Select a channel to plot"
        self.setMinimumHeight(180)

    def set_series(self, title, units, series, markers=()):
        """series: [(label, color, t, y)], markers: [(t, severity)]"""
        self.title, self.units = title, units
        self.series = [
            (label, color, np.asarray(t, float), np.asarray(y, float))
            for label, color, t, y in series
        ]
        self.markers = list(markers)
        self.invalidate()

    def clear(self, message="Select a channel to plot"):
        self.series, self.markers, self.message, self.title = [], [], message, ""
        self.invalidate()

    def area(self):
        return QRectF(
            self.gutter,
            30,
            max(10, self.width() - self.gutter - 14),
            max(10, self.height() - 56),
        )

    def render_static(self, painter):
        painter.fillRect(QRectF(0, 0, self.width(), self.height()), QColor(PANEL))
        painter.setRenderHint(QPainter.Antialiasing, True)
        area = self.area()
        painter.setFont(font(9.5, True))
        painter.setPen(QColor(TEXT))
        painter.drawText(QPointF(14, 20), self.title)
        if self.units:
            offset = painter.fontMetrics().width(self.title)
            painter.setFont(font(8.5))
            painter.setPen(QColor(MUTED))
            painter.drawText(QPointF(14 + offset + 8, 20), self.units)
        if not self.series:
            painter.setPen(QColor(MUTED))
            painter.setFont(font(10))
            painter.drawText(area, Qt.AlignCenter, self.message)
            return
        bins = max(16, int(area.width()))
        reduced = [
            (label, color, *envelope(t, y, self.range.t0, self.range.t1, bins))
            for label, color, t, y in self.series
        ]
        values = [yy for *_, yy in reduced if len(yy)]
        if values:
            values = np.concatenate(values)
            lo, hi = float(values.min()), float(values.max())
        else:
            lo, hi = -1.0, 1.0
        pad = max((hi - lo) * 0.08, 1e-9 if hi != lo else max(abs(hi) * 0.05, 1e-3))
        lo, hi = lo - pad, hi + pad
        self.draw_axis(painter, area)
        painter.setFont(font(7.5))
        for v in nice_ticks(lo, hi, max(2, int(area.height() / 38))):
            y = area.bottom() - (v - lo) / (hi - lo) * area.height()
            painter.setPen(QPen(QColor(LINE), 1))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(QColor(MUTED))
            painter.drawText(
                QRectF(0, y - 8, self.gutter - 6, 16),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{v:.4g}",
            )
        painter.setRenderHint(QPainter.Antialiasing, False)
        span = max(self.range.t1 - self.range.t0, 1e-9)
        for severity in {s for _, s in self.markers}:
            ts = np.array([t for t, s in self.markers if s == severity])
            xs = np.unique(
                np.round(
                    area.left() + (ts - self.range.t0) / span * area.width()
                ).astype(int)
            )
            xs = xs[(xs >= area.left()) & (xs <= area.right())]
            if len(xs):
                color = QColor(SEVERITY_COLORS.get(severity, MUTED))
                color.setAlpha(110)
                painter.setPen(QPen(color, 1))
                painter.drawLines(_vertical_lines(xs, area.top(), area.bottom()))
        painter.save()
        painter.setClipRect(area)
        scale_x = area.width() / span
        scale_y = area.height() / (hi - lo)
        for label, color, tt, yy in reduced:
            if not len(tt):
                continue
            painter.setPen(QPen(QColor(color), 1.3))
            xs = area.left() + (tt - self.range.t0) * scale_x
            ys = area.bottom() - (yy - lo) * scale_y
            if len(tt) == 1:
                painter.drawEllipse(QPointF(xs[0], ys[0]), 2, 2)
            else:
                painter.drawPolyline(polygon(xs, ys))
        painter.restore()
        painter.setRenderHint(QPainter.Antialiasing, True)
        x = self.width() - 14
        painter.setFont(font(8.5))
        for label, color, *_ in reversed(self.series):
            width = painter.fontMetrics().width(label)
            x -= width + 22
            painter.setPen(QPen(QColor(color), 2.5))
            painter.drawLine(QPointF(x, 16), QPointF(x + 12, 16))
            painter.setPen(QColor(TEXT))
            painter.drawText(QPointF(x + 16, 20), label)

    def hover(self, event):
        if not self.series or not self.area().contains(QPointF(event.pos())):
            QToolTip.hideText()
            return
        t = self.t_of(event.pos().x())
        parts = [f"t={t:.4f}s"]
        for label, _, ts, ys in self.series:
            if len(ts):
                i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
                parts.append(f"{label}: {ys[i]:.6g}")
        QToolTip.showText(event.globalPos(), "\n".join(parts), self)
