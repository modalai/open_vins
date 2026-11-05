"""Fast QPainter helpers: zero-copy numpy -> QPolygonF and cached layers.

Python-level loops over points dominate QPainter cost in PyQt; these helpers
fill Qt's own buffers from numpy, and ``LayerCache`` keeps static content in a
pixmap so per-frame updates (a moving playhead) only composite.
"""

import numpy as np
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QPainter, QPixmap, QPolygonF


def polygon(x, y):
    """QPolygonF from coordinate arrays without a Python loop."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    poly = QPolygonF(n)
    if n:
        pointer = poly.data()
        pointer.setsize(16 * n)
        buffer = np.frombuffer(pointer, dtype=np.float64).reshape(n, 2)
        buffer[:, 0] = x
        buffer[:, 1] = y
    return poly


def points(xy):
    """QPolygonF from an (N, 2) array, for drawPoints/drawPolyline."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return polygon(xy[:, 0], xy[:, 1])


class LayerCache:
    """A HiDPI-aware pixmap of a widget's static layer, rebuilt only when invalidated."""

    def __init__(self, widget, render):
        self.widget = widget
        self.render = (
            render  # render(painter) draws the static layer in widget coordinates
        )
        self.pixmap = None
        self.key = None

    def invalidate(self):
        self.pixmap = None

    def draw(self, painter):
        w = self.widget
        ratio = w.devicePixelRatioF()
        key = (w.width(), w.height(), ratio)
        if self.pixmap is None or self.key != key:
            pixmap = QPixmap(int(w.width() * ratio), int(w.height() * ratio))
            pixmap.setDevicePixelRatio(ratio)
            pixmap.fill(Qt.transparent)
            layer = QPainter(pixmap)
            self.render(layer)
            layer.end()
            self.pixmap, self.key = pixmap, key
        painter.drawPixmap(QPointF(0, 0), self.pixmap)
