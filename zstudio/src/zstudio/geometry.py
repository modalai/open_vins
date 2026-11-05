"""Rig conventions and geometry checks for logs (numpy; OpenCV only for the stereo check).

Conventions
-----------
* An edge parent -> child stores ``R`` with the child's axes expressed in the
  parent frame and ``t`` = the child's origin in the parent frame. VOXL
  ``extrinsics.conf`` (``RPY_parent_to_child`` intrinsic X-Y-Z in degrees,
  ``T_child_wrt_parent``) and ROS ``/tf_static`` (parent=header.frame_id,
  child=child_frame_id) both have exactly these semantics.
* ``ov_core`` quaternions are JPL: ``quat_2_Rot(q_ItoC) = R_ItoC``. A JPL
  quaternion of ``R`` equals the Hamilton quaternion of ``R^T``.
"""

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

GRAVITY = 9.80665


def rpy_matrix(roll_deg, pitch_deg, yaw_deg):
    """VOXL RPY (intrinsic X-Y-Z, degrees): Rx(roll) @ Ry(pitch) @ Rz(yaw)."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    rx = np.array(
        [[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]]
    )
    ry = np.array(
        [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]]
    )
    rz = np.array(
        [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]]
    )
    return rx @ ry @ rz


def matrix_rpy(R):
    """Inverse of ``rpy_matrix``: intrinsic X-Y-Z angles in degrees (yaw 0 at gimbal lock)."""
    R = np.asarray(R, dtype=float)
    pitch = math.asin(max(-1.0, min(1.0, R[0, 2])))
    if abs(R[0, 2]) < 1.0 - 1e-9:
        roll = math.atan2(-R[1, 2], R[2, 2])
        yaw = math.atan2(-R[0, 1], R[0, 0])
    else:  # pitch = +-90 deg: roll and yaw share one axis; keep it in roll
        # With yaw = 0, R = Rx(roll) Ry(pitch): R[2,1] = sin(roll), R[1,1] = cos(roll).
        roll = math.atan2(R[2, 1], R[1, 1])
        yaw = 0.0

    # Every rotation also equals (roll + 180, 180 - pitch, yaw + 180); show the simpler one.
    def wrap(a):
        a = (a + 180.0) % 360.0 - 180.0
        return 180.0 if abs(a + 180.0) < 1e-9 else a

    first = [wrap(v) for v in np.degrees([roll, pitch, yaw])]
    second = [wrap(first[0] + 180.0), wrap(180.0 - first[1]), wrap(first[2] + 180.0)]
    best = min(
        (first, second), key=lambda c: (round(sum(abs(v) for v in c), 6), -sum(c))
    )
    return [0.0 if abs(v) < 1e-9 else round(v, 9) for v in best]


def image_rotation(degrees_cw):
    """Axes of a camera whose image is rotated clockwise, expressed in the original camera.

    Rotating the image (like VOXL's en_rotate) rotates the camera frame about its
    optical axis: new pixels (u', v') of a 90 deg clockwise turn see x' = -y,
    y' = x, so the new axes are Rz(-90) in the old frame. Compose it on the
    right of a parent->camera rotation: R_parent_new = R_parent_old @ this.
    """
    angle = -math.radians(int(degrees_cw) % 360)
    c, s = round(math.cos(angle)), round(math.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_xyzw_matrix(q):
    """Hamilton (x, y, z, w) quaternion -> rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_quat_xyzw(R):
    """Rotation matrix -> Hamilton (x, y, z, w), w >= 0."""
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
            0.25 * s,
        ]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1e-300, 1.0 + R[i, i] - R[j, j] - R[k, k]))
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def jpl_from_matrix(R):
    """JPL quaternion (x, y, z, w) whose ov_core quat_2_Rot equals ``R``."""
    return matrix_quat_xyzw(np.asarray(R).T)


def rotation_angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


def rotation_axis(R):
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(w)
    if n > 1e-9:
        return w / n
    values, vectors = np.linalg.eigh(
        (R + R.T) / 2
    )  # 180 degrees: axis is the +1 eigenvector
    return vectors[:, int(np.argmax(values))]


def angle_between_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@dataclass
class RigEdge:
    parent: str
    child: str
    R: np.ndarray  # child axes expressed in parent
    t: np.ndarray  # child origin in parent


@dataclass
class RigConfig:
    """A static frame tree from extrinsics.conf or /tf_static."""

    source: str = ""
    edges: list = field(default_factory=list)
    body: str = ""
    body_convention: str = ""  # FRD (VOXL body) | FLU (REP-103 base_link)
    notes: list = field(default_factory=list)

    @property
    def frames(self):
        names = set()
        for edge in self.edges:
            names.update((edge.parent, edge.child))
        return names

    def transform(self, parent, child):
        """(R, t): child axes/origin expressed in ``parent``; None when unconnected."""
        if parent == child:
            return np.eye(3), np.zeros(3)
        graph = {}
        for e in self.edges:
            graph.setdefault(e.parent, []).append((e.child, e.R, e.t))
            Rinv = e.R.T
            graph.setdefault(e.child, []).append((e.parent, Rinv, -Rinv @ e.t))
        seen, queue = {parent}, deque([(parent, np.eye(3), np.zeros(3))])
        while queue:
            node, R, t = queue.popleft()
            for nxt, Re, te in graph.get(node, ()):
                if nxt in seen:
                    continue
                R2, t2 = R @ Re, R @ te + t
                if nxt == child:
                    return R2, t2
                seen.add(nxt)
                queue.append((nxt, R2, t2))
        return None

    def match(self, name):
        """Frame for a VOXL pipe / ROS frame name: exact, else longest frame that prefixes it."""
        if name in self.frames:
            return name
        best = ""
        for frame in self.frames:
            if name.startswith(frame) and len(frame) > len(best):
                best = frame
        return best or None


def gravity_in_body(convention):
    """Specific force measured by an accelerometer at rest, in the body frame."""
    if convention == "FRD":
        return np.array([0.0, 0.0, -GRAVITY])
    if convention == "FLU":
        return np.array([0.0, 0.0, GRAVITY])
    raise ValueError(convention)


def find_rest(imu, window_s=0.5, max_gyro=0.05):
    """Most stationary window of an (N, 8) IMU array: (t_start, t_end, mean specific force) or None."""
    if len(imu) < 20:
        return None
    t = imu[:, 0]
    rate = (len(t) - 1) / max(t[-1] - t[0], 1e-9)
    n = max(10, int(window_s * rate))
    if len(imu) < n:
        return None
    gyro = np.linalg.norm(imu[:, 1:4], axis=1)
    accel = imu[:, 4:7]
    norm = np.linalg.norm(accel, axis=1)
    kernel = np.ones(n) / n
    gyro_mean = np.convolve(gyro, kernel, mode="valid")
    accel_std = np.sqrt(
        np.maximum(
            np.convolve(norm**2, kernel, "valid")
            - np.convolve(norm, kernel, "valid") ** 2,
            0,
        )
    )
    score = gyro_mean + 0.05 * accel_std
    i = int(np.argmin(score))
    window = slice(i, i + n)
    mean_norm = float(norm[window].mean())
    if gyro_mean[i] > max_gyro or abs(mean_norm - GRAVITY) > 0.6:
        return None
    return float(t[i]), float(t[i + n - 1]), accel[window].mean(axis=0)


def stereo_geometry(
    cam_a, cam_b, intr_a, intr_b, frames=60, progress=None, cancelled=None, seed=0
):
    """Relative pose of two synchronized cameras from their images (needs OpenCV).

    Returns dict(rotation_deg, axis, R_ab, baseline_dir (unit, b's origin in a's
    frame), baseline_spread_deg, inliers, matches, pure_rotation_inliers, pairs).
    Scale is unobservable from images alone.
    """
    import cv2
    from .media import to_gray8

    def undistort(points, intr):
        K = np.array(
            [
                [intr["k"][0], 0, intr["k"][2]],
                [0, intr["k"][1], intr["k"][3]],
                [0, 0, 1],
            ]
        )
        D = np.array(intr["d"][:4], dtype=float)
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        if intr["model"] == "equidistant":
            return cv2.fisheye.undistortPoints(pts, K, D).reshape(-1, 2)
        return cv2.undistortPoints(pts, K, np.array(intr["d"], dtype=float)).reshape(
            -1, 2
        )

    tb = cam_b.times
    pairs = []
    for i, t in enumerate(cam_a.times):
        j = int(np.searchsorted(tb, t))
        best = min(
            (k for k in (j - 1, j) if 0 <= k < len(tb)),
            key=lambda k: abs(tb[k] - t),
            default=None,
        )
        if best is not None and abs(tb[best] - t) < 0.002:
            pairs.append((i, best))
    if len(pairs) < 3:
        raise ValueError("fewer than three time-synchronized image pairs")
    chosen = [
        pairs[int(k)]
        for k in np.linspace(0, len(pairs) - 1, min(frames, len(pairs))).round()
    ]
    sift = cv2.SIFT_create(3000)
    matcher = cv2.BFMatcher()
    ua, ub, groups = [], [], []
    for n, (i, j) in enumerate(chosen):
        if cancelled and cancelled():
            raise InterruptedError
        ga, gb = to_gray8(cam_a.picture(i)), to_gray8(cam_b.picture(j))
        ka, da = sift.detectAndCompute(ga, None)
        kb, db = sift.detectAndCompute(gb, None)
        if da is not None and db is not None and len(ka) > 8 and len(kb) > 8:
            good = [
                m
                for m, s in (x for x in matcher.knnMatch(da, db, k=2) if len(x) == 2)
                if m.distance < 0.7 * s.distance
            ]
            if len(good) >= 12:
                pa = np.float32([ka[m.queryIdx].pt for m in good])
                pb = np.float32([kb[m.trainIdx].pt for m in good])
                ua.append(undistort(pa, intr_a))
                ub.append(undistort(pb, intr_b))
                groups += [n] * len(good)
        if progress:
            progress((n + 1) / len(chosen))
    if not ua:
        raise ValueError("no reliable feature matches between the two cameras")
    ua, ub, groups = np.concatenate(ua), np.concatenate(ub), np.asarray(groups)
    focal = 0.5 * (intr_a["k"][0] + intr_a["k"][1])
    threshold = 1.0 / focal

    def solve(a, b):
        E, mask = cv2.findEssentialMat(
            a, b, np.eye(3), method=cv2.RANSAC, prob=0.99999, threshold=threshold
        )
        if E is None or E.shape != (3, 3):
            raise ValueError("essential matrix estimation failed")
        count, R, t, mask = cv2.recoverPose(E, a, b, np.eye(3), mask=mask)
        return R, t.ravel(), int((mask.ravel() > 0).sum())

    R, t, inliers = solve(ua, ub)
    _, homography_mask = cv2.findHomography(ua, ub, cv2.RANSAC, threshold)
    pure = int(homography_mask.sum()) if homography_mask is not None else 0
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    directions = []
    for _ in range(12 if len(unique) >= 6 else 0):
        keep = np.isin(
            groups, rng.choice(unique, max(3, len(unique) // 2), replace=False)
        )
        try:
            R2, t2, _ = solve(ua[keep], ub[keep])
            directions.append(-R2.T @ t2)
        except (ValueError, cv2.error):
            pass
    # x_b = R x_a + t  =>  R_ab (b axes in a) = R^T ; b's origin in a = -R^T t
    baseline = -R.T @ t
    baseline /= np.linalg.norm(baseline)
    spread = max(
        (angle_between_deg(baseline, d) for d in directions), default=float("nan")
    )
    R_ab = R.T
    return {
        "rotation_deg": rotation_angle_deg(R_ab),
        "axis": rotation_axis(R_ab),
        "R_ab": R_ab,
        "baseline_dir": baseline,
        "baseline_spread_deg": spread,
        "inliers": inliers,
        "matches": int(len(ua)),
        "pure_rotation_inliers": pure,
        "pairs": len(chosen),
    }
