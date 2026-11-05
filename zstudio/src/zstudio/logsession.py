"""Adapters between LogSource and the studio viewer.

``log_session`` turns a log into a Session so the Overview's replay, multi-camera
grid and rig view work unchanged (frames carry no tracks; images come from the
log). ``FrameProvider`` decodes frames on a worker thread into a byte-bounded
LRU cache of QImages; the GUI thread never decodes.
"""

from collections import OrderedDict, deque
import math
import threading
import time

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from .geometry import jpl_from_matrix
from .media import Tonemapper, array_to_qimage, black_level_for
from .model import POINT_DTYPE, Camera, Frame, Session

_NO_POINTS = np.zeros(0, dtype=POINT_DTYPE)


def camera_extrinsics(log, imu, cam):
    """(q_ItoC JPL, p_IinC) from the log's rig config, or None."""
    rig = log.rig
    if imu is None or not rig.edges or not cam.frame:
        return None
    imu_frame = imu.frame if imu.frame in rig.frames else rig.match(imu.name)
    transform = rig.transform(imu_frame, cam.frame) if imu_frame else None
    if transform is None:
        return None
    R_IC, p_CinI = transform  # camera axes / origin expressed in the IMU frame
    R_ItoC = R_IC.T
    return list(jpl_from_matrix(R_ItoC)), list(-R_ItoC @ p_CinI)


def log_session(log):
    imu = log.imus[0] if log.imus else None
    cameras, frames = [], []
    for index, cam in enumerate(log.cameras):
        intr = cam.intrinsics or {}
        width, height = (
            cam.width or intr.get("width") or 640,
            cam.height or intr.get("height") or 480,
        )
        k = intr.get("k") or [0.8 * width, 0.8 * width, width / 2, height / 2]
        d = (list(intr.get("d") or []) + [0.0] * 4)[:4]
        ext = camera_extrinsics(log, imu, cam)
        q, p = ext if ext else ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        fps = (
            1.0 / float(np.median(np.diff(cam.times)))
            if len(cam) > 1 and np.median(np.diff(cam.times)) > 0
            else 0.0
        )
        cameras.append(
            Camera(
                width=int(width),
                height=int(height),
                k=list(k),
                d=d,
                q=q,
                p=p,
                fps=fps,
                fisheye=intr.get("model") == "equidistant",
            )
        )
        exposure = cam.exposure if cam.exposure is not None else np.zeros(len(cam))
        for row, t in enumerate(cam.times):
            frames.append(
                Frame(
                    float(t),
                    _NO_POINTS,
                    cam=index,
                    seq=row,
                    exposure=float(exposure[row])
                    if np.isfinite(exposure[row])
                    else 0.0,
                )
            )
    data = imu.data if imu is not None else np.empty((0, 8))
    session = Session(
        log.name, cameras, data, frames, path=str(log.path), format=0, complete=True
    )
    session.log = log
    session.extrinsics_known = [
        camera_extrinsics(log, imu, cam) is not None for cam in log.cameras
    ]
    return session


class FrameProvider(QObject):
    """Asynchronous frame decoding for log replay.

    ``image(camera, row, rate)`` never blocks. An exact cache hit is returned;
    otherwise the newest decoded frame at or before ``row`` is shown (so views
    keep moving even when decoding is slower than playback) and a decode is
    scheduled. During playback (``rate`` = frames per second of wall time) the
    request aims where the playhead will be when the decode finishes, using the
    measured per-camera decode time, and a decode already in flight that will
    still be useful is not preempted.
    """

    ready = pyqtSignal()
    LOOKBACK = 120  # rows searched backwards for a stand-in frame

    def __init__(self, cameras, max_width=1600, budget_mb=384, workers=2, parent=None):
        super().__init__(parent)
        self.cameras = list(cameras)
        self.mappers = [
            Tonemapper(black_level_for(c.name, c.sensor), max_width, smooth=False)
            for c in self.cameras
        ]
        self.cache = OrderedDict()
        self.bytes = 0
        self.budget = budget_mb * 1024 * 1024
        self.errors = {}
        self.pending = OrderedDict()  # camera -> row (latest request wins)
        self.prefetch = deque(maxlen=64)
        self.inflight = set()  # (camera, row) keys being decoded
        self.cost = {}  # camera -> EMA decode seconds
        self.active = {}  # camera -> monotonic time of the last request
        self.latest = {}  # camera -> (row, QImage) most recently decoded
        self.max_width = max_width
        self.widths = {}  # camera -> pixel width of its largest visible view
        self.cond = threading.Condition()
        self.stopped = False
        # Decoding (OpenCV, numpy) releases the GIL; two workers keep a camera grid moving.
        self.threads = [
            threading.Thread(target=self._run, name=f"frame-provider-{i}", daemon=True)
            for i in range(max(1, workers))
        ]
        for thread in self.threads:
            thread.start()

    def set_width(self, camera, pixels):
        """Decode a camera at about the size it is shown (smaller views decode much faster)."""
        pixels = int(max(160, min(self.max_width, pixels)))
        if camera < len(self.mappers) and abs(self.widths.get(camera, 0) - pixels) > 32:
            self.widths[camera] = pixels
            self.mappers[camera].out_width = pixels

    def _sharp_enough(self, camera, image):
        wanted = min(
            self.widths.get(camera, self.max_width),
            self.cameras[camera].width or self.max_width,
        )
        return image.width() >= 0.8 * wanted

    def image(self, camera, row, rate=0.0):
        """QImage for (camera, row): exact, else the nearest earlier decoded frame, else None."""
        if camera >= len(self.cameras):
            return None
        cam = self.cameras[camera]
        now = time.monotonic()
        with self.cond:
            self.active[camera] = now
            key = (camera, row)
            image = self.cache.get(key)
            if image is not None and self._sharp_enough(camera, image):
                self.cache.move_to_end(key)
                if rate > 0:
                    self._schedule(camera, row, rate, now)
                return image
            stale = image  # a smaller decode of this very frame: show it while a sharper one decodes
            image = None
            for back in range(1, self.LOOKBACK):
                stand_in = self.cache.get((camera, row - back))
                if stand_in is not None:
                    image = stand_in
                    break
            if stale is not None:
                image = stale
            if image is None and camera in self.latest:
                image = self.latest[camera][1]
            if cam.has_pictures:
                if rate > 0:
                    self._schedule(camera, row, rate, now)
                else:
                    self.pending[camera] = row  # an update keeps its place in line
                    for nxt in (row + 1, row - 1):
                        if 0 <= nxt < len(cam) and (camera, nxt) not in self.cache:
                            self.prefetch.append((camera, nxt))
                self.cond.notify()
        return image

    def _schedule(self, camera, row, rate, now):
        """Request the row the playhead reaches when this decode would finish."""
        busy = [c for c, t in self.active.items() if now - t < 0.5] or [camera]
        wait = sum(self.cost.get(c, 0.02) for c in busy)
        lead = max(1, int(math.ceil(rate * wait)))
        target = min(len(self.cameras[camera]) - 1, row + lead)
        if any(c == camera and row <= r <= target for c, r in self.inflight):
            return  # a decode in flight will be shown in time
        cached = self.cache.get((camera, target))
        if cached is not None and self._sharp_enough(camera, cached):
            return
        self.pending[camera] = target  # an update keeps its place in line (round robin)

    def stop(self):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        for thread in self.threads:
            thread.join(timeout=2.0)

    def _run(self):
        while True:
            with self.cond:
                while not self.stopped and not self.pending and not self.prefetch:
                    self.cond.wait()
                if self.stopped:
                    return
                if self.pending:
                    key = self.pending.popitem(last=False)
                else:
                    key = self.prefetch.popleft()
                cached = self.cache.get(key)
                if (
                    cached is not None and self._sharp_enough(key[0], cached)
                ) or key in self.inflight:
                    continue
                self.inflight.add(key)
            camera, row = key
            started = time.perf_counter()
            try:
                cam = self.cameras[camera]
                width = self.mappers[camera].out_width
                image = array_to_qimage(
                    self.mappers[camera](cam.preview(row, width)), display=True
                )
                error = None
            except Exception as exc:  # report per camera; keep serving the others
                image, error = None, f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            with self.cond:
                self.inflight.discard(key)
                if self.stopped:
                    return
                previous = self.cost.get(camera)
                self.cost[camera] = (
                    elapsed if previous is None else 0.7 * previous + 0.3 * elapsed
                )
                if image is not None:
                    old = self.cache.pop(key, None)
                    if old is not None:
                        self.bytes -= old.byteCount()
                    self.cache[key] = image
                    self.bytes += image.byteCount()
                    while self.bytes > self.budget and len(self.cache) > 1:
                        _, old = self.cache.popitem(last=False)
                        self.bytes -= old.byteCount()
                    self.latest[camera] = (row, image)
                    self.errors.pop(camera, None)
                else:
                    self.errors[camera] = error
            self.ready.emit()
