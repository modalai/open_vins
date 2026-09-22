#!/usr/bin/env python3
"""Independent RS ownership/row-chain contract oracle; no runtime RS enablement.

Requires NumPy and SciPy. Dense covariance and QR work here are test code only.
The continuous process, observed-row pixel-noise convention, and completed
linear-factor chain are deliberately tested as distinct contracts.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm, block_diag, solve_triangular


CHECKS = {}


def check(name, value, limit=None, greater=False):
    passed = bool(value) if limit is None else bool(value > limit if greater else value < limit)
    CHECKS[name] = {"passed": passed, "value": float(value), "limit": limit, "greater": greater}
    if not passed:
        raise AssertionError((name, value, limit))


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def normalized_error(actual, expected):
    scale = 1. / np.sqrt(np.maximum(np.diag(expected), 1e-30))
    return np.max(np.abs(scale[:, None] * (actual - expected) * scale[None, :]))


def condition(P, H, residual, R, mean=None):
    mean = np.zeros(P.shape[0]) if mean is None else mean
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T
    J = np.eye(P.shape[0]) - K @ H
    posterior = J @ P @ J.T + K @ R @ K.T
    return mean + K @ (residual - H @ mean), posterior, S


class ContinuousRows:
    """Constant corrected body signals, raw-axis calibration and three clocks."""
    def __init__(self):
        self.w = np.array([.8, -.6, .3])
        self.a = np.array([1.5, -.8, 9.6])
        self.R0 = expm(skew([.2, -.1, .05]))
        self.p0 = np.array([.3, -.2, .1])
        self.v0 = np.array([.7, -.1, .05])
        self.C = expm(skew([-.03, .02, .01]))
        self.extrinsic = np.array([.08, .015, -.01])
        self.W = np.array([[1.04, .02, -.01], [0., .97, .015], [0., 0., 1.02]])
        self.Acc = expm(skew([.1, -.07, .03])) @ np.diag([.98, 1.03, 1.01])
        self.Tg = np.array([[.008, -.004, .003], [.002, .007, -.005], [-.006, .001, .004]])
        self.A = np.zeros((15, 15)); self.L = np.zeros((15, 12))
        for i in (0, 3, 6):
            self.A[i:i+3, i:i+3] = -skew(self.w)
        self.A[0:3, 9:12] = -self.W
        self.A[0:3, 12:15] = self.W @ self.Tg @ self.Acc
        self.A[3:6, 6:9] = np.eye(3)
        self.A[6:9, 0:3] = -skew(self.a)
        self.A[6:9, 12:15] = -self.Acc
        self.L[0:3, 0:3] = -self.W
        self.L[0:3, 3:6] = self.W @ self.Tg @ self.Acc
        self.L[6:9, 3:6] = -self.Acc
        self.L[9:12, 6:9] = self.L[12:15, 9:12] = np.eye(3)
        densities = np.repeat(np.square([.06, .4, .02, .05]), 3)
        self.N = (self.L * densities) @ self.L.T
        rng = np.random.default_rng(37)
        root = np.eye(18) + .05 * np.tril(rng.normal(size=(18, 18)), -1)
        sigmas = np.r_[np.repeat([.015, .07, .08, .01, .025], 3), .0012, .0018, .0015]
        root = sigmas[:, None] * root
        self.P0 = root @ root.T
        self.nodes, self.weights = leggauss(24)

    def mean(self, time):
        R = expm(-skew(self.w) * time) @ self.R0
        velocity = self.v0 - np.array([0., 0., 9.81]) * time
        position = self.p0 + self.v0 * time - np.array([0., 0., 9.81]) * (.5 * time*time)
        for node, weight in zip(self.nodes, self.weights):
            u = .5 * time * (node + 1.)
            accel = self.R0.T @ expm(skew(self.w) * u) @ self.a
            velocity += .5 * time * weight * accel
            position += .5 * time * weight * (time-u) * accel
        return R, position, velocity

    def pose_map(self, time):
        R, _, velocity = self.mean(time)
        S = np.zeros((6, 15)); S[:3, :3] = np.eye(3); S[3:, 3:6] = R.T
        return S, np.r_[self.w, velocity]

    def process_cross_integral(self, left, right):
        # Direct source integral: no Van Loan block, process recursion or row
        # augmentation is used by this dense reference.
        end = min(left, right); out = np.zeros((15, 15))
        for node, weight in zip(self.nodes, self.weights):
            u = .5 * end * (node + 1.)
            out += .5 * end * weight * expm(self.A * (left-u)) @ self.N @ expm(self.A.T * (right-u))
        return out

    def interval(self, dt):
        # Independent sequential construction uses a Van Loan exponential.
        M = np.zeros((30, 30)); M[:15, :15] = self.A
        M[:15, 15:] = self.N; M[15:, 15:] = -self.A.T
        E = expm(dt * M); F = E[:15, :15]
        return F, E[:15, 15:] @ F.T

    def dense(self, rows, endpoint=.05, independent_process=False):
        outputs = [(endpoint, np.c_[np.eye(15), np.zeros((15, 3))])]
        clocks = np.c_[np.zeros((3, 15)), np.eye(3)]
        outputs.append((0., clocks))
        for row in rows:
            S, g = self.pose_map(row["time"])
            outputs.append((row["time"], np.c_[S, np.outer(g, row["clock"])]))
        widths = [o[1].shape[0] for o in outputs]
        P = np.zeros((sum(widths), sum(widths)))
        start = np.r_[0, np.cumsum(widths)]
        for i, (ti, Ji) in enumerate(outputs):
            Ti = block_diag(expm(self.A * ti), np.eye(3))
            for j, (tj, Jj) in enumerate(outputs):
                Tj = block_diag(expm(self.A * tj), np.eye(3))
                block = Ti @ self.P0 @ Tj.T
                if not independent_process or i == j:
                    block[:15, :15] += self.process_cross_integral(ti, tj)
                P[start[i]:start[i+1], start[j]:start[j+1]] = Ji @ block @ Jj.T
        return .5 * (P + P.T)

    def sequential(self, rows, endpoint=.05):
        P = self.P0.copy(); last = 0.
        for row in rows:
            F, Q = self.interval(row["time"] - last)
            transition = np.eye(P.shape[0]); transition[:15, :15] = F
            P = transition @ P @ transition.T; P[:15, :15] += Q
            S, g = self.pose_map(row["time"])
            J = np.zeros((6, P.shape[0])); J[:, :15] = S; J[:, 15:18] = np.outer(g, row["clock"])
            augment = np.r_[np.eye(P.shape[0]), J]
            P = augment @ P @ augment.T; last = row["time"]
        F, Q = self.interval(endpoint-last)
        T = np.eye(P.shape[0]); T[:15, :15] = F
        P = T @ P @ T.T; P[:15, :15] += Q
        return .5 * (P + P.T)

    def projection(self, time, point):
        R, p, _ = self.mean(time)
        q = self.C @ R @ (point-p) + self.extrinsic
        return np.array([400.*q[0]/q[2] + 320., 405.*q[1]/q[2] + 240.])

    def geometry(self, time, point):
        R, p, _ = self.mean(time); local = R @ (point-p)
        q = self.C @ local + self.extrinsic
        D = np.array([[400./q[2], 0., -400.*q[0]/q[2]**2], [0., 405./q[2], -405.*q[1]/q[2]**2]])
        return D @ self.C @ np.c_[skew(local), -R], D @ self.C @ R

    def point_on_row(self, time, row, x=345., depth=4.8):
        R, p, _ = self.mean(time)
        q = depth * np.array([(x-320.)/400., (row-240.)/405., 1.])
        return p + R.T @ self.C.T @ (q-self.extrinsic)


def row_ownership():
    model = ContinuousRows(); raw = .02; td = -.002; readout = .016
    rows = [{"time": raw+td+(v/480.-.5)*readout, "clock": np.array([0., 1., v/480.-.5]), "pixel": v}
            for v in (40., 240., 420.)]
    # Same nominal instant as the bottom RS row, with independent GS clock.
    rows.append({"time": rows[-1]["time"], "clock": np.array([1., 0., 0.]), "pixel": 300.})
    rows.sort(key=lambda row: row["time"])
    expected = model.dense(rows); got = model.sequential(rows)
    check("row_owned_covariance_matches_dense_sources", normalized_error(got, expected), 3e-11)
    check("same_time_different_clock_rows_remain_distinct", np.linalg.norm(expected[-12:-6, -12:-6]-expected[-6:, -6:]), 1e-7, True)
    H = np.zeros((4*len(rows), got.shape[0])); points = []
    for i, row in enumerate(rows):
        for j in range(2):
            point = model.point_on_row(row["time"], row["pixel"], 330.+20*j, 4.+j)
            Hr, _ = model.geometry(row["time"], point)
            H[4*i+2*j:4*i+2*j+2, 18+6*i:24+6*i] = Hr
            points.append(point)
    residual = .12 * np.sin(np.arange(H.shape[0])+.2); R = .36*np.eye(H.shape[0])
    mu, post, S = condition(expected, H, residual, R)
    ma, pa, sa = condition(got, H, residual, R)
    check("joint_row_update_mean", np.max(np.abs(mu-ma)), 1e-11)
    check("joint_row_update_covariance", normalized_error(pa, post), 3e-10)
    check("joint_row_update_innovation", np.max(np.abs(S-sa)), 1e-10)
    check("clock_readout_updated_through_owned_rows", np.linalg.norm(post[15:18, 15:18]-expected[15:18, 15:18]), 1e-10, True)
    wrong = model.dense(rows, independent_process=True)
    _, wrong_post, _ = condition(wrong, H, residual, R)
    check("negative_fresh_independent_row_noise", normalized_error(wrong_post, post), 1e-4, True)
    midpoint = [dict(row, time=raw+td) if row["clock"][1] else row for row in rows]
    wrong = model.dense(midpoint)
    check("negative_shared_midpoint_pose_covariance", normalized_error(wrong, expected), 1e-3, True)
    shift = np.linalg.norm(model.projection(rows[0]["time"], points[0])-model.projection(raw+td, points[0]))
    check("negative_shared_midpoint_projection_pixels", shift, .1, True)
    # A delayed subset conditions every retained row, including rows that it
    # does not observe. The second subset must use those updated means/blocks.
    cut = 6
    m1, p1, _ = condition(expected, H[:cut], residual[:cut], R[:cut, :cut])
    m2, p2, _ = condition(p1, H[cut:], residual[cut:], R[cut:, cut:], m1)
    check("delayed_row_batches_retain_all_means", np.max(np.abs(m2-mu)), 1e-10)
    check("delayed_row_batches_retain_all_cross_blocks", normalized_error(p2, post), 3e-10)
    reset = p1.copy(); reset[18:, :] = 0.; reset[:, 18:] = 0.; reset[18:, 18:] = expected[18:, 18:]
    reset_mean = m1.copy(); reset_mean[18:] = 0.
    _, reset_post, _ = condition(reset, H[cut:], residual[cut:], R[cut:, cut:], reset_mean)
    check("negative_recreate_historical_row_posterior", normalized_error(reset_post[:18, :18], post[:18, :18]), 1e-4, True)
    return model, rows


def geometry_and_row_noise(model, rows):
    for i, row in enumerate(rows[:3]):
        time = row["time"]; beta = row["clock"][2]
        point = model.point_on_row(time, row["pixel"])
        Hpose, Hfeature = model.geometry(time, point); _, g = model.pose_map(time)
        eps = 1e-7
        fd = (model.projection(time+eps, point)-model.projection(time-eps, point))/(2*eps)
        check("row_time_jacobian_%d" % i, np.max(np.abs(fd-Hpose@g)), 2e-5)
        fd_readout = (model.projection(time+beta*eps, point)-model.projection(time-beta*eps, point))/(2*eps)
        check("row_readout_jacobian_%d" % i, np.max(np.abs(fd_readout-beta*Hpose@g)), 2e-5)
        if beta:
            check("negative_duplicate_direct_readout_column_%d" % i, np.linalg.norm(2*beta*Hpose@g-fd_readout), 1., True)
        R, p, _ = model.mean(time)
        N = np.zeros((9, 4)); N[3:6, :3] = N[6:, :3] = np.eye(3)
        N[:3, 3] = R @ np.array([0., 0., 1.])
        N[3:6, 3] = skew([0., 0., 1.]) @ p
        N[6:, 3] = skew([0., 0., 1.]) @ point
        check("row_translation_yaw_geometry_%d" % i, np.max(np.abs(np.c_[Hpose, Hfeature]@N)), 1e-11)
    # If v_observed itself supplies row time, the residual is implicit in v.
    # This local pixel covariance is distinct from shared IMU/row process noise.
    row = rows[0]; point = model.point_on_row(row["time"], row["pixel"])
    z = model.projection(row["time"], point); readout = .016
    Hpose, _ = model.geometry(row["time"], point); _, g = model.pose_map(row["time"])
    M = np.eye(2)-np.outer(Hpose@g, [0., readout/480.])
    def residual(observed):
        t = .02-.002+(observed[1]/480.-.5)*readout
        return observed-model.projection(t, point)
    numeric = np.column_stack([(residual(z+np.eye(2)[i]*1e-5)-residual(z-np.eye(2)[i]*1e-5))/(2e-5) for i in range(2)])
    check("observed_row_pixel_noise_jacobian", np.max(np.abs(numeric-M)), 1e-8)
    pixel = np.diag([.7**2, 1.1**2]); transformed = M@pixel@M.T
    check("negative_untransformed_observed_row_pixel_noise", normalized_error(pixel, transformed), 1e-4, True)


def conditional_row_rank():
    t = np.linspace(.05, .95, 17)
    K = np.minimum.outer(t, t)-np.outer(t, t)
    values = np.linalg.eigvalsh(K)
    check("continuous_interior_row_kernel_rank", np.linalg.matrix_rank(K) == 17)
    check("continuous_interior_row_smallest_eigenvalue", values[0], 1e-3, True)
    check("negative_fixed_six_dimensional_frame_latent", np.linalg.norm(values[:-6])/np.linalg.norm(values), .02, True)
    P = np.eye(1); H = np.ones((len(t), 1)); pixel = .02*np.eye(len(t))
    _, exact, _ = condition(P, H, np.zeros(len(t)), K+pixel)
    _, independent, _ = condition(P, H, np.zeros(len(t)), np.diag(np.diag(K))+pixel)
    check("negative_diagonal_row_noise_information", normalized_error(independent, exact), .2, True)


def chain_factors():
    # A small linear Markov chain makes the separator and its fill-in inspectable.
    # Navigation is [angle, position, velocity]; td/readout and three unfinished
    # scalar feature coordinates remain alive across several row times.
    d, c, f, count = 3, 2, 3, 7
    dt = .017; F = np.eye(d); F[1, 2] = dt
    Q = np.array([[.0005*dt, 0., 0.], [0., .04*dt**3/3, .04*dt**2/2], [0., .04*dt**2/2, .04*dt]])
    root = np.eye(d+c); root[2, 0] = .2; root[3, 1] = -.1; root[4, 0] = .05
    root = np.array([.15, .3, .12, .025, .03])[:, None]*root
    initial = root@root.T
    schedule = [(0, (0, 2)), (1, (1,)), (2, (0, 1)), (3, (1, 2)), (4, (2,)), (5, (0, 1)), (6, (0, 2))]
    measurements = []
    for row, features in schedule:
        entries = []
        for feature in features:
            h = np.array([.7+.1*feature, 1., .03*row])
            clocks = np.array([.8+.03*row, (.14*row-.42)*(.8+.03*row)])
            hf = np.zeros(f); hf[feature] = -(1.+.03*row)
            entries.append((np.r_[h, clocks, hf]/.08, (.04*np.sin(2*row+feature)+.03)/.08))
        measurements.append(entries)
    return d, c, f, count, F, Q, initial, measurements


def row_chain_elimination():
    d, c, f, count, F, Q, initial, measurements = chain_factors()
    sep = d+c+f; size = d*count+c+f
    W0 = solve_triangular(np.linalg.cholesky(initial), np.eye(d+c), lower=True)
    Wq = solve_triangular(np.linalg.cholesky(Q), np.eye(d), lower=True)
    initial_rows = np.zeros((d+c, size)); initial_rows[:, :d] = W0[:, :d]; initial_rows[:, d*count:d*count+c] = W0[:, d:]
    all_A = [initial_rows]; all_b = [np.zeros(d+c)]
    for row in range(count):
        if row:
            block = np.zeros((d, size)); block[:, d*(row-1):d*row] = -Wq@F; block[:, d*row:d*(row+1)] = Wq
            all_A.append(block); all_b.append(np.zeros(d))
        for h, residual in measurements[row]:
            block = np.zeros((1, size)); block[0, d*row:d*(row+1)] = h[:d]; block[0, d*count:] = h[d:]
            all_A.append(block); all_b.append(np.array([residual]))
    A = np.concatenate(all_A); b = np.concatenate(all_b)
    full_mean = np.linalg.lstsq(A, b, rcond=None)[0]
    full_covariance = np.linalg.inv(A.T@A)
    keep = np.r_[np.arange(d*(count-1), d*count), np.arange(d*count, size)]
    expected_mean = full_mean[keep]; expected_covariance = full_covariance[np.ix_(keep, keep)]

    def eliminate(reset_feature_cross=False):
        C = np.zeros((d+c, sep)); C[:, :d+c] = W0; rhs = np.zeros(d+c)
        logdet_eliminated = 0.; tail = 0.
        for row in range(count):
            if row:
                old = np.zeros((C.shape[0], sep+d)); old[:, :d] = C[:, :d]; old[:, 2*d:] = C[:, d:]
                transition = np.zeros((d, sep+d)); transition[:, :d] = -Wq@F; transition[:, d:2*d] = Wq
                joined = np.r_[old, transition]; joined_rhs = np.r_[rhs, np.zeros(d)]
                U, R = np.linalg.qr(joined[:, :d], mode="complete")
                logdet_eliminated += 2*np.log(np.abs(np.diag(R[:d]))).sum()
                C = (U.T@joined)[d:, d:]; rhs = (U.T@joined_rhs)[d:]
            for h, residual in measurements[row]:
                C = np.r_[C, h[None, :]]; rhs = np.r_[rhs, residual]
            if C.shape[0] > sep:
                U, R = np.linalg.qr(C, mode="complete"); transformed = U.T@rhs
                tail += np.dot(transformed[sep:], transformed[sep:]); C = R[:sep]; rhs = transformed[:sep]
            if reset_feature_cross and row == 3:
                mean = np.linalg.lstsq(C, rhs, rcond=None)[0]; P = np.linalg.inv(C.T@C)
                P[:d+c, d+c:] = 0.; P[d+c:, :d+c] = 0.; P[d+c:, d+c:] = np.diag(np.diag(P[d+c:, d+c:]))
                C = solve_triangular(np.linalg.cholesky(P), np.eye(sep), lower=True); rhs = C@mean
        mean = np.linalg.lstsq(C, rhs, rcond=None)[0]; covariance = np.linalg.inv(C.T@C)
        objective = np.linalg.norm(C@mean-rhs)**2+tail
        logdet = np.linalg.slogdet(C.T@C)[1]+logdet_eliminated
        return mean, covariance, objective, logdet

    mean, covariance, objective, logdet = eliminate()
    check("completed_row_chain_separator_mean", np.max(np.abs(mean-expected_mean)), 1e-9)
    check("completed_row_chain_separator_covariance", normalized_error(covariance, expected_covariance), 1e-8)
    check("completed_row_chain_profile_cost", abs(objective-np.linalg.norm(A@full_mean-b)**2), 1e-9)
    check("completed_row_chain_information_determinant", abs(logdet-np.linalg.slogdet(A.T@A)[1]), 1e-7)
    _, wrong, _, _ = eliminate(True)
    check("negative_discard_open_feature_cross_blocks", normalized_error(wrong, expected_covariance), .01, True)
    # A new factor on an eliminated old row requires its conditional law. The
    # current separator marginal alone does not specify that law.
    old = np.r_[np.arange(d, 2*d), np.arange(4*d, 5*d)]
    Pss = expected_covariance; Pos = full_covariance[np.ix_(old, keep)]
    K = np.linalg.solve(Pss, Pos.T).T
    conditional = full_covariance[np.ix_(old, old)]-K@Pos.T
    H_old = np.array([[.8, 1., .1, 0., 0., 0.], [0., 0., 0., 1., -.7, .05]])
    H_full = np.zeros((2, size)); H_full[:, old] = H_old
    late = np.array([.08, -.03]); pixel = .001*np.eye(2)
    dense_mean, dense_covariance, _ = condition(full_covariance, H_full, late, pixel, full_mean)
    offset = H_old@(full_mean[old]-K@expected_mean)
    local_mean, local_covariance, _ = condition(Pss, H_old@K, late-offset, pixel+H_old@conditional@H_old.T, expected_mean)
    check("late_rows_require_conditional_mean", np.max(np.abs(local_mean-dense_mean[keep])), 1e-9)
    check("late_rows_require_conditional_covariance", normalized_error(local_covariance, dense_covariance[np.ix_(keep, keep)]), 1e-8)
    _, naive, _ = condition(Pss, H_old@K, late-offset, pixel, expected_mean)
    check("negative_discard_eliminated_row_conditional_noise", normalized_error(naive, local_covariance), 1e-4, True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model, rows = row_ownership(); geometry_and_row_noise(model, rows)
    conditional_row_rank(); row_chain_elimination()
    result = {"checks": len(CHECKS), "failures": sum(not v["passed"] for v in CHECKS.values()),
              "runtime_rolling_shutter_enabled": False, "measurements": CHECKS,
              "scope": "continuous row ownership, implicit-row geometry, complete frozen linear row-chain factors; not production support"}
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text+"\n")
    print(text)


if __name__ == "__main__":
    main()
