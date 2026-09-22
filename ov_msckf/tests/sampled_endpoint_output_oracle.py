#!/usr/bin/env python3
"""Independent dense common-raw-noise output oracle, with no runtime import."""
import json
import numpy as np
from scipy.linalg import block_diag, expm


def skew(x):
    return np.array([[0., -x[2], x[1]], [x[2], 0., -x[0]], [-x[1], x[0], 0.]])


def check(ok, message):
    global checks
    checks += 1
    if not ok:
        raise AssertionError(message)


def mapping(nominal, weights, dx=None, omit_mean=False):
    if dx is None:
        dx = np.zeros(len(nominal))
    r = expm(-skew(dx[:3])) @ rotation
    x = nominal + dx
    mu = np.zeros(12) if omit_mean else x[15:27]
    raw = weights[0] * (records[0] - mu[:6]) + weights[1] * (records[1] - mu[6:])
    angular = gyro @ (raw[:3] - x[9:12] - tg @ accel @ (raw[3:] - x[12:15]))
    # First three entries are the output attitude error about the current R.
    return np.r_[dx[:3], x[3:6], r @ x[6:9], angular]


def analytic(nominal, weights):
    h = np.zeros((12, len(nominal)))
    h[:6, :6] = np.eye(6)
    h[6:9, :3] = skew(rotation @ nominal[6:9])
    h[6:9, 6:9] = rotation
    h[9:12, 9:12] = -gyro
    h[9:12, 12:15] = gyro @ tg @ accel
    sensor = np.c_[-gyro, gyro @ tg @ accel]
    for i in range(2):
        h[9:12, 15+6*i:21+6*i] = weights[i] * sensor
    return h


def finite_difference(nominal, weights, step):
    h = np.empty((12, len(nominal)))
    for col in range(len(nominal)):
        dx = np.zeros(len(nominal)); dx[col] = step
        h[:, col] = (mapping(nominal, weights, dx) - mapping(nominal, weights, -dx)) / (2*step)
    return h


def condition(p, mean, h, residual, independent_r):
    k = np.linalg.solve(h @ p @ h.T + independent_r, h @ p).T
    t = np.eye(len(mean)) - k @ h
    return t @ p @ t.T + k @ independent_r @ k.T, mean + k @ residual


checks = 0
rng = np.random.default_rng(730619)
rotation = expm(skew(np.array([.31, -.22, .18])))
accel = expm(skew(np.array([.16, -.11, .07]))) @ np.array([[.97, 0., 0.], [-.024, 1.055, 0.], [.016, -.019, 1.02]])
gyro = expm(skew(np.array([-.09, .04, .12]))) @ np.array([[1.07, .018, -.012], [0., .94, .021], [0., 0., 1.025]])
tg = np.array([[.028, -.014, .013], [.012, .027, -.015], [-.016, .011, .024]])
records = [np.array([.34, -.21, .17, .62, -.38, 9.7]), np.array([.29, -.18, .22, .51, -.44, 9.8])]
n = 41  # IMU15, original raw noises12, historical poses12 and two clocks.
mean = np.zeros(n); mean[3:15] = [.7, -.4, .2, .3, -.16, .09, .013, -.021, .009, .035, -.026, .019]
prior0 = np.diag([.02, .03, .04, .06, .08, .07])
prior1 = np.diag([.03, .04, .02, .08, .07, .06])
p = block_diag(np.eye(15)*.02, prior0, prior1, np.eye(14)*.03)
# Transport common raw draws into current navigation and retained history.
transport = np.eye(n)
transport[:15, 15:27] = rng.normal(size=(15, 12))*.11
transport[27:39, 15:27] = rng.normal(size=(12, 12))*.07
p = transport @ p @ transport.T
observation = rng.normal(size=(7, n))*.4
p, mean = condition(p, mean, observation, np.linspace(-.13, .17, 7), np.eye(7)*.001)
check(np.linalg.norm(mean[15:27]) > .01, "conditioning infers original raw-noise means")
check(np.linalg.norm(p[15:21, 21:27]) > .001, "conditioning correlates original raw endpoints")

errors = []
controls = []
for weights in [np.array([1., 0.]), np.array([.63, .37]), np.array([0., 1.])]:
    h = analytic(mean, weights)
    coarse = finite_difference(mean, weights, 4e-4)
    middle = finite_difference(mean, weights, 2e-4)
    fine = finite_difference(mean, weights, 1e-4)
    ratio = np.linalg.norm(coarse-middle)/np.linalg.norm(middle-fine)
    fd = (4*fine-middle)/3
    error = np.max(np.abs(h-fd)); errors.append(error)
    check(3.9 < ratio < 4.1, "nonlinear finite differences converge quadratically")
    check(error < 2e-9, "output Jacobian agrees with independent nonlinear map")
    covariance, cross = h @ p @ h.T, p @ h.T
    check(np.max(np.abs(covariance - fd @ p @ fd.T)) < 2e-10, "full output covariance agrees with finite-difference projection")
    check(np.max(np.abs(cross-p @ fd.T)) < 2e-10, "all historical/state cross terms survive output projection")
    check(np.linalg.eigvalsh(covariance)[0] > 0., "unconstrained dense output is SPD")
    missing_mean = np.linalg.norm(mapping(mean, weights, omit_mean=True) - mapping(mean, weights))
    no_cross = p.copy()
    other = list(range(15)) + list(range(27, n))
    no_cross[np.ix_(other, range(15, 27))] = 0.
    no_cross[np.ix_(range(15, 27), other)] = 0.
    discarded_cross = np.linalg.norm(h @ no_cross @ h.T-covariance)
    independent_h = h.copy(); independent_h[:, 15:27] = 0.
    sensor = np.c_[-gyro, gyro @ tg @ accel]
    independent_cov = independent_h @ p @ independent_h.T
    independent_cov[9:12, 9:12] += sensor @ (weights[0]**2*prior0 + weights[1]**2*prior1) @ sensor.T
    independent_error = np.linalg.norm(independent_cov-covariance)
    check(missing_mean > .001, "negative omitted posterior raw mean changes output")
    check(discarded_cross > .001, "negative discarded state/raw cross changes covariance")
    check(independent_error > .001, "negative independent angular R loses common-noise conditioning")
    controls.append(dict(weights=weights.tolist(), omitted_mean=missing_mean, omitted_cross=discarded_cross,
                         independent_R=independent_error, fd_ratio=ratio))

# At the successor knot, the old raw owner has zero direct weight. Marginalizing
# it preserves the current output and its cross with every retained variable.
h = analytic(mean, np.array([0., 1.]))
keep = list(range(15)) + list(range(21, n))
check(np.array_equal(h[:, 15:21], np.zeros((12, 6))), "old raw owner has no direct successor-knot output dependence")
check(np.max(np.abs(h @ p @ h.T-h[:, keep] @ p[np.ix_(keep, keep)] @ h[:, keep].T)) < 1e-15,
      "retirement preserves output marginal without conditioning the old raw draw")
check(np.max(np.abs((p @ h.T)[keep]-p[np.ix_(keep, keep)] @ h[:, keep].T)) < 1e-15,
      "retirement preserves every retained state/output cross")
reused = mean.copy(); reused[15:21] = [.05, -.06, .04, -.03, .02, -.01]
check(np.linalg.norm(mapping(mean, np.array([1., 0.]))-mapping(reused, np.array([1., 0.]))) > .01,
      "negative reused coordinate changes a historical output despite identical slot address")

# An exact observation of angular rate annihilates its output uncertainty. The
# projection supports this singular limit; no independent R restores noise.
angular_h = h[9:12]
constrained_p, constrained_mean = condition(p, mean, angular_h, np.array([.001, -.002, .003]), np.zeros((3, 3)))
check(np.max(np.abs(angular_h @ constrained_p @ angular_h.T)) < 1e-16, "exact constraint yields zero angular covariance to roundoff")
check(np.max(np.abs(constrained_p @ angular_h.T)) < 1e-16, "exact constraint also removes angular cross covariance")

print(json.dumps(dict(checks=checks, failures=0, max_jacobian_fd=max(errors), controls=controls,
                      exact_constraint_covariance=float(np.max(np.abs(angular_h @ constrained_p @ angular_h.T)))), indent=2))
