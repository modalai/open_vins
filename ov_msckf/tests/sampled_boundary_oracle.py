#!/usr/bin/env python3
"""Independent Gaussian oracle for a proposed sampled-IMU boundary contract.

No production estimator code is imported. A bounded recursive filter is checked
against conditioning all primitive random variables in one dense batch. This is
a model/contract test, not a test or implementation of the VINS runtime.
"""

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import block_diag, expm


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


D = 18  # IMU15, one calibration error, two independent camera clocks.
RAW_TIMES = np.array([0., .5, 1.1, 1.8])
EVENTS = [.2, .4, .5, .72, 1., 1.1, 1.35, 1.8]
CAPACITY = 3
SEED = 20260922
RNG = np.random.default_rng(SEED)
W = np.array([[1.08, .07, -.02], [.03, .92, .04], [-.05, .02, 1.03]])
ACC = np.array([[.96, .03, .08], [-.04, 1.07, .01], [.02, -.06, .99]])
TG = np.array([[.06, -.03, .02], [.01, -.04, .05], [-.02, .03, .07]])
A = np.zeros((D, D))
A[:3, :3] = -skew([.12, -.09, .18])
A[:3, 9:12] = -W
A[:3, 12:15] = W @ TG @ ACC
A[3:6, 6:9] = np.eye(3)
A[6:9, :3] = -skew([.3, -.2, .7])
A[6:9, 12:15] = -ACC
A[:3, 15] = [-.2, .1, .06]
A[6:9, 15] = [.3, -.1, .2]
B = np.zeros((D, 6))
B[:3, :3] = -W
B[:3, 3:] = W @ TG @ ACC
B[6:9, 3:] = -ACC

raw_covariances = []
for k in range(len(RAW_TIMES)):
    L = np.tril(RNG.normal(0., .035, (6, 6)))
    L[np.diag_indices(6)] = np.linspace(.23, .42, 6) * (1. + .07 * k)
    raw_covariances.append(L @ L.T)
L0 = np.tril(RNG.normal(0., .035, (D, D)))
L0[np.diag_indices(D)] = np.linspace(.12, .3, D)
P0 = L0 @ L0.T
M0 = np.linspace(-.08, .11, D)


def interval(t0, t1):
    k = int(np.searchsorted(RAW_TIMES, t0 + 1e-12, side="right") - 1)
    assert 0 <= k < len(RAW_TIMES) - 1 and t1 <= RAW_TIMES[k + 1] + 1e-12
    return k, (t0 - RAW_TIMES[k]) / (RAW_TIMES[k + 1] - RAW_TIMES[k])


def recursive_transition(t0, t1):
    """Quadrature of each raw sample's interpolant coefficient."""
    k, lam0 = interval(t0, t1)
    h, gap = t1 - t0, RAW_TIMES[k + 1] - RAW_TIMES[k]
    left, right = np.zeros((D, 6)), np.zeros((D, 6))
    nodes, weights = np.polynomial.legendre.leggauss(10)
    for node, weight in zip(nodes, weights):
        s = .5 * h * (node + 1.)
        kernel = .5 * h * weight * expm(A * (h - s)) @ B
        lam = lam0 + s / gap
        left += kernel * (1. - lam)
        right += kernel * lam
    return expm(A * h), left, right


def batch_transition(t0, t1):
    """Independent augmented ODE: x'=Ax+Bu, u'=slope, slope'=0."""
    k, lam0 = interval(t0, t1)
    gap = RAW_TIMES[k + 1] - RAW_TIMES[k]
    generator = np.zeros((D + 12, D + 12))
    generator[:D, :D] = A
    generator[:D, D:D + 6] = B
    generator[D:D + 6, D + 6:] = np.eye(6)
    E = expm(generator * (t1 - t0))
    J0, J1 = E[:D, D:D + 6], E[:D, D + 6:]
    return E[:D, :D], (1. - lam0) * J0 - J1 / gap, lam0 * J0 + J1 / gap


def independent_kick(h):
    # Explicit independent terminal bias kick. This fixture does not claim to
    # validate a continuous bias-walk discretizer or the nonlinear IMU ODE.
    Q = np.zeros((D, D))
    Q[9:15, 9:15] = np.diag(np.linspace(.0004, .0011, 6)) * h
    return Q


class DenseBatch:
    def __init__(self):
        segments = list(zip([0.] + EVENTS[:-1], EVENTS))
        self.raw_start = [D + 6 * k for k in range(len(RAW_TIMES))]
        self.kick_start = [D + 6 * len(RAW_TIMES) + D * k for k in range(len(segments))]
        self.P = block_diag(P0, *raw_covariances, *[independent_kick(b - a) for a, b in segments])
        self.m = np.zeros(self.P.shape[0]); self.m[:D] = M0
        self.C = np.zeros((D, self.P.shape[0])); self.C[:, :D] = np.eye(D)
        self.rows, self.measurements, self.noises = [], [], []
        self.labels = [("x", i) for i in range(D)]
        self.chart = np.eye(D)

    def append_raw(self, k):
        T = np.zeros((6, self.P.shape[0]))
        T[:, self.raw_start[k]:self.raw_start[k] + 6] = np.eye(6)
        self.C = np.vstack([self.C, T]); self.labels += [(f"n{k}", i) for i in range(6)]

    def propagate(self, t0, t1, step):
        k, _ = interval(t0, t1)
        F, Gl, Gr = batch_transition(t0, t1)
        F = self.chart @ F @ np.linalg.inv(self.chart)
        Gl, Gr = self.chart @ Gl, self.chart @ Gr
        self.C[:D] = F @ self.C[:D]
        self.C[:D, self.raw_start[k]:self.raw_start[k] + 6] += Gl
        self.C[:D, self.raw_start[k + 1]:self.raw_start[k + 1] + 6] += Gr
        self.C[:D, self.kick_start[step]:self.kick_start[step] + D] += self.chart

    def condition(self, H, z, R):
        self.rows.append(H @ self.C); self.measurements.append(z); self.noises.append(R)

    def posterior(self):
        if not self.rows:
            return self.C @ self.m, self.C @ self.P @ self.C.T
        O, z, R = np.vstack(self.rows), np.concatenate(self.measurements), block_diag(*self.noises)
        S = O @ self.P @ O.T + R
        CPO = self.C @ self.P @ O.T
        m = self.C @ self.m + CPO @ np.linalg.solve(S, z - O @ self.m)
        P = self.C @ self.P @ self.C.T - CPO @ np.linalg.solve(S, CPO.T)
        return m, .5 * (P + P.T)

    def transform(self, T, labels):
        self.C = T @ self.C; self.labels = list(labels)


class BoundedFilter:
    def __init__(self, mode):
        self.mode = mode
        self.m, self.P = M0.copy(), P0.copy()
        self.labels = [("x", i) for i in range(D)]
        self.max_raw_dimension = 0
        self.chart = np.eye(D)

    def append_raw(self, k):
        self.m = np.concatenate([self.m, np.zeros(6)])
        self.P = block_diag(self.P, raw_covariances[k])
        self.labels += [(f"n{k}", i) for i in range(6)]
        self.max_raw_dimension = max(self.max_raw_dimension, len(self.raw_indices()))

    def raw_indices(self):
        return [i for i, label in enumerate(self.labels) if label[0].startswith("n")]

    def propagate(self, t0, t1, _step):
        k, _ = interval(t0, t1)
        F, Gl, Gr = recursive_transition(t0, t1)
        F = self.chart @ F @ np.linalg.inv(self.chart)
        Gl, Gr = self.chart @ Gl, self.chart @ Gr
        raw = self.raw_indices()
        if self.mode == "independent_segments":
            # Even gives every segment its correct sampled marginal; it loses
            # only reuse across segments, unlike production's different Q rule.
            self.m[raw] = 0.
            self.P[raw, :] = 0.; self.P[:, raw] = 0.
            for sample in [k, k + 1]:
                ids = [self.labels.index((f"n{sample}", j)) for j in range(6)]
                self.P[np.ix_(ids, ids)] = raw_covariances[sample]
        T = np.eye(len(self.m)); T[:D] = 0.; T[:D, :D] = F
        for sample, G in [(k, Gl), (k + 1, Gr)]:
            ids = [self.labels.index((f"n{sample}", j)) for j in range(6)]
            T[np.ix_(range(D), ids)] += G
        self.m = T @ self.m
        self.P = T @ self.P @ T.T
        self.P[:D, :D] += self.chart @ independent_kick(t1 - t0) @ self.chart.T

    def condition(self, H, z, R):
        H, R = H.copy(), R.copy()
        if self.mode == "noise_to_R":
            raw = self.raw_indices()
            L = H[:, raw].copy()
            if np.any(L):
                z = z - L @ self.m[raw]
                R += L @ self.P[np.ix_(raw, raw)] @ L.T
                H[:, raw] = 0.
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.m += K @ (z - H @ self.m)
        J = np.eye(len(self.m)) - K @ H
        self.P = J @ self.P @ J.T + K @ R @ K.T
        self.P = .5 * (self.P + self.P.T)
        if self.mode == "covariance_only":
            self.m[self.raw_indices()] = 0.

    def transform(self, T, labels):
        self.m, self.P = T @ self.m, T @ self.P @ T.T
        self.labels = list(labels)


def select_rows(obj, keep):
    T = np.eye(len(obj.labels))[keep]
    obj.transform(T, [obj.labels[i] for i in keep])


def add_clone(obj, number, camera):
    n = len(obj.labels)
    J = np.zeros((6, n)); J[:, :6] = np.eye(6)
    # Two same-time owners have independent clock columns, like physical views.
    J[:, 16 + camera] = np.array([.2, -.1, .3, .7, -.2, .4])
    obj.transform(np.vstack([np.eye(n), J]), obj.labels + [(f"c{number}", i) for i in range(6)])


def change_linear_chart(obj):
    # Generic invertible linear chart, not a nonlinear attitude/gauge test.
    # A full-state congruence must also transform cross blocks with raw noise.
    R = expm(skew([.13, -.08, .11]))
    T = np.eye(len(obj.labels))
    for base in [0, 3, 6]:
        T[base:base + 3, base:base + 3] = R
    for name in sorted({name for name, _ in obj.labels if name.startswith("c")}):
        ids = [obj.labels.index((name, j)) for j in range(6)]
        T[np.ix_(ids, ids)] = block_diag(R, R)
    obj.transform(T, obj.labels)
    obj.chart = T[:D, :D] @ obj.chart


def run(mode):
    f, ref = BoundedFilter(mode), DenseBatch()
    max_mean, max_cov, checkpoints = 0., 0., []
    max_nav_mean, max_nav_cov = 0., 0.
    prior_time, clone_number = 0., 0
    snapshot = None
    snapshot_restore_error = None
    handoff = None
    for step, time in enumerate(EVENTS):
        k, lam0 = interval(prior_time, time)
        for sample in [k, k + 1]:
            if (f"n{sample}", 0) not in f.labels:
                f.append_raw(sample); ref.append_raw(sample)
        f.propagate(prior_time, time, step); ref.propagate(prior_time, time, step)
        for camera in ([0, 1] if step == 1 else [step % 2]):
            for obj in [f, ref]:
                add_clone(obj, clone_number, camera)
                clones = sorted({name for name, _ in obj.labels if name.startswith("c")}, key=lambda s: int(s[1:]))
                if len(clones) > CAPACITY:
                    select_rows(obj, [i for i, label in enumerate(obj.labels) if label[0] != clones[0]])
            clone_number += 1

        for measurement in range(2):
            H = np.zeros((3, len(f.m)))
            H[:, :3] = np.diag([.2, -.3, .25])
            H[:, 3:6] = np.eye(3)
            H[:, 6:9] = np.diag([.4, .25, -.2])
            H[:, 15] = [.2, -.15, .1]
            oldest = min(int(name[1:]) for name, _ in f.labels if name.startswith("c"))
            for axis in range(3):
                H[axis, f.labels.index((f"c{oldest}", axis + 3))] -= .65
            R = np.diag([.018, .023, .021]) * (1. + .1 * measurement)
            z = np.sin(np.arange(3) + .6 * step + .2 * measurement) * .27
            # One raw-dependent observation exposes why merely adding LQL' to
            # R omits C terms and fails to update the retained raw noise mean.
            if step == 3 and measurement == 1:
                lam = (time - RAW_TIMES[k]) / (RAW_TIMES[k + 1] - RAW_TIMES[k])
                L = np.hstack([np.diag([.6, -.4, .5]), TG])
                for sample, weight in [(k, 1. - lam), (k + 1, lam)]:
                    ids = [f.labels.index((f"n{sample}", j)) for j in range(6)]
                    H[:, ids] = weight * L
            f.condition(H, z, R); ref.condition(H, z, R)
            rm, rP = ref.posterior()
            mean_error = float(np.max(np.abs(f.m - rm)))
            cov_error = float(np.max(np.abs(f.P - rP)))
            max_mean, max_cov = max(max_mean, mean_error), max(max_cov, cov_error)
            max_nav_mean = max(max_nav_mean, float(np.max(abs(f.m[:15] - rm[:15]))))
            max_nav_cov = max(max_nav_cov, float(np.max(abs(f.P[:15, :15] - rP[:15, :15]))))
            checkpoints.append({"time": time, "measurement": measurement, "mean_error": mean_error,
                                "covariance_error": cov_error, "raw_mean_norm": float(np.linalg.norm(rm[f.raw_indices()]))})
        if step == 1:
            snapshot = copy.deepcopy(f)
            saved_m, saved_P = f.m.copy(), f.P.copy()
            f.m += 7.; f.P *= 3.  # Deliberately dirty both means and covariance.
            f = copy.deepcopy(snapshot)
            snapshot_restore_error = max(float(np.max(abs(f.m - saved_m))), float(np.max(abs(f.P - saved_P))))
        if step == 4:
            # A new estimator can retain the current joint posterior while
            # dropping the old visual window. Raw means and cross blocks are
            # still required to continue through this same enclosing pair.
            raw = f.raw_indices()
            handoff = {"raw_mean_norm": float(np.linalg.norm(f.m[raw])),
                       "state_raw_cross_norm": float(np.linalg.norm(f.P[np.ix_(range(D), raw)]))}
            for obj in [f, ref]:
                select_rows(obj, [i for i, label in enumerate(obj.labels) if not label[0].startswith("c")])
            for obj in [f, ref]:
                change_linear_chart(obj)
            rm, rP = ref.posterior()
            max_mean = max(max_mean, float(np.max(abs(f.m - rm))))
            max_cov = max(max_cov, float(np.max(abs(f.P - rP))))
        # At the right raw knot the left sample has no future direct use.
        # All correlations with old visual clones remain in the principal block.
        if time == RAW_TIMES[k + 1]:
            for obj in [f, ref]:
                select_rows(obj, [i for i, label in enumerate(obj.labels) if label[0] != f"n{k}"])
        prior_time = time
    rm, rP = ref.posterior()
    return {"mode": mode, "max_mean_error": max_mean, "max_covariance_error": max_cov,
            "max_navigation_mean_error": max_nav_mean, "max_navigation_covariance_error": max_nav_cov,
            "final_mean_error": float(np.max(abs(f.m - rm))), "final_covariance_error": float(np.max(abs(f.P - rP))),
            "final_navigation_mean_error": float(np.max(abs(f.m[:15] - rm[:15]))),
            "max_raw_dimension": f.max_raw_dimension, "final_raw_dimension": len(f.raw_indices()),
            "final_state_dimension": len(f.m), "snapshot_restore_error": snapshot_restore_error,
            "joint_handoff_inside_raw_interval": handoff,
            "final_current_mean": f.m[:D].tolist(), "dense_current_mean": rm[:D].tolist(),
            "checkpoints": checkpoints}


def scalar_counterexample():
    # One raw interval [0,1]. Two half-interval velocity errors integrate the
    # linear interpolant. This also supplies easily inspectable numeric values.
    a, b = np.array([.375, .125]), np.array([.125, .375])
    P = np.diag([.4, 1., 1.]); m = np.zeros(3)
    T = np.eye(3); T[0, 1:] = a
    P = T @ P @ T.T
    H = np.array([[1., 0., 0.]])
    K = P @ H.T / (P[0, 0] + .1)
    m += K[:, 0] * .2
    P -= K @ H @ P
    after_first_noise_mean = m[1:].copy()
    state_raw_cross = P[0, 1:].copy()
    T[0, 1:] = b
    m = T @ m; P = T @ P @ T.T
    exact_prior_m, exact_prior_P = float(m[0]), float(P[0, 0])
    K = P @ H.T / (P[0, 0] + .12)
    m += K[:, 0] * (-.1 - m[0]); P -= K @ H @ P
    # Forget the boundary after the first measurement, but use the exact
    # second segment marginal. This is already better than guessing sigma/dt.
    oldP = .4 + a @ a
    oldm = oldP / (oldP + .1) * .2
    oldP = oldP * .1 / (oldP + .1) + b @ b
    oldprior = (float(oldm), float(oldP))
    oldm += oldP / (oldP + .12) * (-.1 - oldm)
    oldP = oldP * .12 / (oldP + .12)
    # Independently condition the primitive [initial state, eta0, eta1].
    rootP = np.diag([.4, 1., 1.])
    O = np.array([[1., *a], [1., *(a + b)]])
    out = np.array([1., *(a + b)])
    S = O @ rootP @ O.T + np.diag([.1, .12])
    C = out @ rootP @ O.T
    batch_m = C @ np.linalg.solve(S, np.array([.2, -.1]))
    batch_P = out @ rootP @ out - C @ np.linalg.solve(S, C)
    assert abs(m[0] - batch_m) < 1e-14 and abs(P[0, 0] - batch_P) < 1e-14
    return {"first_weights": a.tolist(), "second_weights": b.tolist(),
            "cross_interval_covariance": float(a @ b), "unsplit_sampled_variance": float((a + b) @ (a + b)),
            "independent_segment_variance": float(a @ a + b @ b), "continuous_density_one_variance": 1.,
            "posterior_raw_mean_after_first_update": after_first_noise_mean.tolist(),
            "state_raw_cross_after_first_update": state_raw_cross.tolist(),
            "before_second_update_exact": [exact_prior_m, exact_prior_P],
            "before_second_update_independent": list(oldprior),
            "after_second_update_exact": [float(m[0]), float(P[0, 0])],
            "after_second_update_independent": [float(oldm), float(oldP)]}


def boundary_rank():
    # Two future horizons after an interior boundary need two independent raw
    # coefficients per axis. A single interpolated 6-vector is insufficient.
    rows = []
    for end in [.7, 1.]:
        start = .4
        rows.append([end - start - .5 * (end ** 2 - start ** 2), .5 * (end ** 2 - start ** 2)])
    return {"scalar_future_weight_matrix": rows, "rank_per_axis": int(np.linalg.matrix_rank(rows)),
            "six_axis_rank": int(np.linalg.matrix_rank(np.kron(rows, np.eye(6))))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    modes = {mode: run(mode) for mode in ["full_joint", "independent_segments", "covariance_only", "noise_to_R"]}
    correct = modes["full_joint"]
    assert correct["max_mean_error"] < 2e-11 and correct["max_covariance_error"] < 2e-11
    assert correct["snapshot_restore_error"] == 0.
    assert correct["max_raw_dimension"] == 12 and correct["final_raw_dimension"] == 6
    assert modes["independent_segments"]["max_mean_error"] > 1e-3
    assert modes["independent_segments"]["max_covariance_error"] > 1e-3
    assert modes["independent_segments"]["final_navigation_mean_error"] > 1e-3
    assert modes["covariance_only"]["max_covariance_error"] < 2e-11
    assert modes["covariance_only"]["final_mean_error"] > 1e-3
    assert modes["covariance_only"]["final_navigation_mean_error"] > 1e-3
    assert modes["noise_to_R"]["max_mean_error"] > 1e-3
    assert modes["noise_to_R"]["max_covariance_error"] > 1e-3
    assert modes["noise_to_R"]["final_navigation_mean_error"] > 1e-3
    rank = boundary_rank()
    assert rank["rank_per_axis"] == 2 and rank["six_axis_rank"] == 12
    transition_error = 0.
    for t0, t1 in zip([0.] + EVENTS[:-1], EVENTS):
        for a, b in zip(recursive_transition(t0, t1), batch_transition(t0, t1)):
            transition_error = max(transition_error, float(np.max(abs(a - b))))
    assert transition_error < 2e-13
    receipt = {"result": "PASS", "scope": "independent linear Gaussian model/contract oracle; no production runtime linked",
               "seed": SEED, "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "raw_times": RAW_TIMES.tolist(), "events": EVENTS, "nav_calibration_clock_dimension": D,
               "clone_capacity": CAPACITY, "transition_crosscheck_error": transition_error,
               "scalar_counterexample": scalar_counterexample(), "minimum_boundary_rank": rank, "modes": modes}
    encoded = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(json.dumps({"result": "PASS", "transition_error": transition_error,
                      "modes": {name: {key: value[key] for key in ["max_mean_error", "max_covariance_error", "final_mean_error"]}
                                for name, value in modes.items()}}, indent=2))


if __name__ == "__main__":
    main()
