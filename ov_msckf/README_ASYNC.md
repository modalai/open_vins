# Asynchronous Multi-Camera MSCKF: Epoch-Anchored Cloning, ACI² Bridging, and Per-Camera Temporal Calibration

> **OpenVINS: An Open Platform for Visual-Inertial Research**
> Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
> Copyright (C) 2018-2023 Patrick Geneva
> Copyright (C) 2018-2023 Guoquan Huang
> Copyright (C) 2018-2023 OpenVINS Contributors
>
> Distributed under the GNU General Public License v3, under the same terms as the rest of OpenVINS.

This document specifies the extensions in `ov_msckf` that make the filter correct and consistent on
**unsynchronized multi-camera rigs** (dual-mono without a hardware sync line, and mixed
global-shutter / rolling-shutter rigs such as a hires RS camera paired with a GS tracking camera).
It covers the lock-free ingest, the per-camera temporal states, the epoch-anchored cloning scheme
with its ACI² preintegration bridge, the rolling-shutter measurement model, the deferred
marginalization rule, the reset semantics, the deliberate omission of the bridge transport noise
(the `Q_tp` question), and every configuration key added by this work. The synchronized code path
is preserved bit-identical: FNV-1a hash parity `6d38d0ae8d4fc3ae` of the simulated estimator
stream against the pre-branch baseline, measured with the out-of-tree jig driven by
`scripts/sim_ab.sh parity` (the jig itself is not committed; see §11).

---

## 1. Notation

We follow OpenVINS conventions [1]: JPL quaternions, with ${}^{I}_{G}R$ the rotation taking a
vector from the global frame $\{G\}$ into the IMU frame $\{I\}$. The core state is

```math
x = \big(\; {}^{I}_{G}\bar q,\; {}^{G}p_I,\; {}^{G}v_I,\; b_g,\; b_a
\;\mid\; x_{\text{calib}} \;\mid\; \{ {}^{I_{e}}_{G}\bar q,\ {}^{G}p_{I_{e}} \}_{e=1..C} \;\mid\; x_{\text{feat}} \big)
```

with $C$ stochastic clones. Camera $i$ has extrinsics $({}^{C_i}_{I}R,\ {}^{C_i}p_I)$, intrinsics
$\zeta_i$, a **cam-to-IMU time offset** $t_{d,i}$, and a **rolling-shutter readout time** $t_{r,i}$
(seconds from first to last row). $\lfloor \cdot \rfloor$ is the skew operator,
$\mathrm{Exp}(\cdot)$ the $SO(3)$ exponential. Gravity ${}^{G}g = [0\ 0\ 9.81]^\top$.

**Time model.** A pixel on row $m$ (of $M$) of a frame stamped $t_{f,i}$ in camera clock was truly
imaged, in IMU clock, at

```math
t(m) \;=\; \underbrace{t_{f,i} + t_{d,i}}_{\text{frame start, IMU clock}} \;+\; t_{r,i}\,\tfrac{m}{M},
\qquad\qquad t_{r,i} = 0 \iff \text{global shutter.}
```

Every effect in this document is a consequence of taking this equation seriously for **every**
camera independently.

---

## 2. The unsynchronized multi-camera problem

Stock multi-camera MSCKF [1,3] assumes camera frames share timestamps, so one clone serves every
camera's measurements at that instant. Without hardware sync the frame times $\{t_{f,i}\}$ never
coincide, and the two naive treatments both fail:

1. **Pretend they are synced** (snap timestamps): injects a systematic time error up to half a
   frame period. At $\|\omega\| = 1\,\mathrm{rad/s}$ and $f = 450\,\mathrm{px}$, a $7\,\mathrm{ms}$ phase
   error is a $\sim 3\,\mathrm{px}$ *systematic* reprojection bias against a $1.2\,\mathrm{px}$ noise model
   — the filter digests it as calibration and bias corrections and diverges (observed: $41.9\,\mathrm{m}$
   ATE, NEES $5041$ on the frozen sim baseline).

2. **Clone at every camera's frame time**: with $N$ cameras at similar rates, a window of $C$
   clones spans only $\approx C/N$ frames *per camera*. Feature tracks hit the marginalization
   horizon early, never reach the max-track length that graduates them to SLAM anchors, and
   near-hover MSCKF triangulation collapses with the shortened baseline. This starves the update
   (observed on hardware: 1–5 MSCKF features per update, calibration random-walk, divergence).

The resolution is **epoch-anchored cloning** (§5): clone only at one reference camera's frame
times, and bind every other camera's frames onto those clones **exactly**, by integrating the IMU
over the known time residual (§6), rather than by first-order pose interpolation.

---

## 3. Lock-free asynchronous ingest (`core/AsyncCameraBuffer`)

One SPSC ring per camera (capacity `async_ring_size`), producers are the per-camera driver
threads, the single consumer is the IMU feed thread — the estimator therefore runs strictly
single-threaded with respect to state. Frames are *released* to the estimator by a $k$-way merge
that maintains three invariants:

- **I1 (release-order FIFO):** each ring preserves arrival order; a frame not newer than the
  merge's last released timestamp (a single cursor across all streams — release order is global)
  is dropped as `late`.
- **I2 (IMU coverage):** the merge head at time $t_h$ is released only once IMU data covers
  $t_h + g$, with guard $g =$ `async_guard`. Propagation and bridging (§6) therefore never
  extrapolate inertial data.
- **I3 (global ordering with liveness):** the head (minimum staged timestamp across streams) is
  released only when every other *live* stream has either staged a frame at $\ge t_h$ or has been
  silent longer than `async_stale_factor` $\times$ its EMA frame period — a dead or throttled
  camera bounds the added latency instead of blocking the pipeline. The period EMA is seeded from
  the declared `camN_fps` (§10) so the gate runs at design width from the very first frame, on
  cold start *and* after every hard reset.

Same-timestamp frames across cameras are bundled into one `CameraData` (this is how synchronized
rigs flow through the identical code path). Frames stamped further than `bogus_future` into the
IMU future are broken timestamps and are dropped. Every discarded frame — ring-full, late, bogus —
flows through the **disposer callback exactly once** with `processed=false`, so externally owned
image handles (e.g. GPU `cl_mem`) are released deterministically; drop counters
(`count_drop_late/full/bogus`) are exported and printed per update in the `[EPOCH]` debug
telemetry line. On a well-configured rig all three counters stay pinned at zero.

---

## 4. Per-camera temporal calibration states

The single scalar $t_d$ of stock OpenVINS [4] becomes a map: each camera owns
$t_{d,i}$ (`State::_calib_dt_CAMtoIMU_map`, accessors `cam_imu_dt(i)`, `cam_imu_dt_delta(i)`), all
read from the kalibr chain [9] (`timeshift_cam_imu` **per camera**). One camera
(`cam_imu_dt_ref_camid`, $r$) is the **clock reference**: clones are stamped in camera-$r$-aligned
IMU time, so the quantity that shifts camera $i$'s measurements relative to the clones is the
*delta* $\Delta t_{d,i} = t_{d,i} - t_{d,r}$. Consequently the measurement Jacobian carries
**paired $\pm$ dt columns** — a column on $t_{d,i}$ and the negated coupling on $t_{d,r}$ — which
is exactly the structure that keeps the common clock mode unobservable (as it must be: shifting
*all* clocks together is a gauge freedom) while the *relative* offsets remain observable.

Rolling-shutter readout times $t_{r,i}$ are additional 1-DOF states, registered **only for
cameras declared `rolling`** (§10): a global-shutter camera's readout is physically zero and
estimating it anyway would hand the filter a spurious DOF that absorbs residual junk on that
stream. The initial prior is $\sigma_{t_r,0} =$ `calib_cam_readout_init_sigma`; the legacy
$1\,\mathrm{ms}$ is correct for *refining a measured readout*, while recovering from a datasheet-grade
guess needs $\approx 5\,\mathrm{ms}$ — a $12\,\mathrm{ms}$ seed error under a $1\,\mathrm{ms}$ prior is a 12-σ fight the
filter effectively never wins under FEJ.

**Degenerate-motion gating.** Temporal states are observable only when the window's kinematics
vary: under constant ${}^{I}\omega$ and constant ${}^{G}v$ (hover, straight cruise) the dt/readout
columns become collinear with clone and feature errors and the states drift on noise [7]. The gate
tests the clone window's excitation: its **peak** $\|{}^{I}\omega\|$ against
`dt_calib_gate_min_omega` and its velocity **spread** (max deviation from the window mean, an
acceleration proxy) against `dt_calib_gate_min_vel_spread`; when both fall below threshold the
**columns are skipped while the current values remain applied** (`dt_calib_gate`) — the model
stays correct, only its refinement pauses.

---

## 5. Epoch-anchored cloning (`epoch_mode`)

Clones are created **only at reference-camera frame times** $t_e$ (the *epochs*). A non-reference
frame from camera $i$, stamped $t_{f,i}$ (same camera clock convention as $t_e$), defines the
**epoch residual**

```math
\delta_i \;=\; \underbrace{\big(t_{f,i} - t_e\big)}_{\text{raw, stored}} \;+\; \Delta t_{d,i} ,
```

whose raw part is stored per (clone, camera) in `State::_epoch_residuals` while the calibration
part $\Delta t_{d,i} = t_{d,i} - t_{d,r}$ is added at update time via `cam_imu_dt_delta(i)` — so
the transport interval stays differentiable in the live dt states, and $\delta_i$ equals the true
IMU-clock separation $(t_{f,i} + t_{d,i}) - (t_e + t_{d,r})$. The frame **binds** to the epoch
clone if $0 \le t_{f,i} - t_e \le \beta\, T_r$ (the ordered release of §3 makes the raw residual
non-negative), with $T_r$ the EMA-tracked reference frame period and $\beta =$
`epoch_bind_factor`; otherwise it falls back to its own clone (`epoch_fallbacks` counts these; the
one-frame-per-(camera, epoch) rule prevents double binding). Binding is *exact*, not
interpolated: the measurement model transports the state across the raw residual with a
preintegration bridge (§6), while the $\Delta t_{d,i}$ remainder — an estimated state, so it must
stay differentiable — rides the first-order dt transport of §7.

**The reference must be the fastest camera.** If some camera has rate $f_i \gt f_r$, pigeonholing
forces at least $f_i - f_r$ extra frames per second into already-occupied epochs; each one falls
back to its own clone. The window fragments into mixed epoch/fallback times, camera $i$'s tracks
split across both clone families, max-track graduation dies, and updates starve — precisely the
hardware failure observed with `ref = 30 Hz` against a 60 Hz camera (1–5 MSCKF features/update →
calibration random-walk). The constructor warns loudly if a declared `camN_fps` exceeds the
reference's.

**Deferred epoch marginalization.** Stock OpenVINS marginalizes the oldest clone after *every*
update. In epoch mode an epoch's measurements arrive in **installments** — the reference frame
first, bound frames up to one binding horizon later. Marginalizing between installments silently
re-shrinks the effective window of every late-arriving camera, reproducing failure mode (2) of §2
through the back door: non-reference tracks never span `max_clones`, never reach max-track length,
never graduate to SLAM. The fix is one bit of state (`epoch_marg_pending`): the oldest clone is
marginalized **only when its epoch completes**, i.e. when the next new-time message opens a new
epoch. An A/B bisection isolated this as *the* decisive mechanism: split-vs-bundled updates on
identical data go from $5.3\,\mathrm{m}$ to $0.44\,\mathrm{m}$ ATE, independent of camera phase.

---

## 6. The ACI² preintegration bridge (`state/PreintegrationBridge.h`, `Propagator::compute_bridge`)

The bridge relates the IMU pose at the true imaging time $t_f = t_e + \delta$ to the epoch clone,
using the **actual IMU samples** over $[t_e,\, t_e + \delta]$ with the analytic closed forms of
ACI² [2,10] (implemented in this module's `Propagator::compute_Xi_sum`, small-$\omega$ safe).
Define the relative terms, all expressed in the epoch frame $\{I_e\}$:

```math
\Delta R = {}^{I_f}_{I_e}R, \qquad
\alpha = \iint_{t_e}^{t_f} {}^{I_e}_{I(s)}R\; \hat a(s)\, ds, \qquad
\beta = \int_{t_e}^{t_f} {}^{I_e}_{I(s)}R\; \hat a(s)\, ds,
```

so that

```math
{}^{I_f}_{G}R = \Delta R \; {}^{I_e}_{G}R, \qquad
{}^{G}p_{I_f} = {}^{G}p_{I_e} + {}^{G}v_{I_e}\,\delta - \tfrac{1}{2}\,{}^{G}g\,\delta^2 + {}^{I_e}_{G}R^\top \alpha, \qquad
{}^{G}v_{I_f} = {}^{G}v_{I_e} - {}^{G}g\,\delta + {}^{I_e}_{G}R^\top \beta .
```

Per IMU step of length $\Delta t$ (bias/intrinsics-corrected midpoint signals $\hat\omega, \hat a$;
$R_{step} = \mathrm{Exp}(-\hat\omega \Delta t)$; $A = \Delta R^\top$; $\Xi_1 \ldots \Xi_4$ the
analytic integrals of [10]):

```math
\alpha \leftarrow \alpha + \beta\,\Delta t + A\,\Xi_2\,\hat a, \qquad
\beta \leftarrow \beta + A\,\Xi_1\,\hat a, \qquad
\Delta R \leftarrow R_{step}\,\Delta R .
```

**Bias Jacobians.** The bridge is built once at bias linearization $\bar b = (\bar b_g, \bar b_a)$
(the ACI² partial-fixed convention [10]) and corrected to first order at update time, never
re-integrated:

```math
\Delta R(\hat b) = \mathrm{Exp}\!\big(J_{\theta g}\,\delta b_g\big)\,\Delta R(\bar b), \qquad
\alpha(\hat b) = \alpha + J_{\alpha g}\,\delta b_g + J_{\alpha a}\,\delta b_a, \qquad
\beta(\hat b) = \beta + J_{\beta g}\,\delta b_g + J_{\beta a}\,\delta b_a,
```

with $\delta b = \hat b - \bar b$ and the recursions (consuming the pre-transport $J_{\theta g}$,
then advancing it):

```math
J_{\alpha g} \leftarrow J_{\alpha g} + J_{\beta g} \Delta t + A\big(\Xi_4 + \lfloor \Xi_2 \hat a \rfloor J_{\theta g}\big), \qquad
J_{\alpha a} \leftarrow J_{\alpha a} + J_{\beta a} \Delta t - A\,\Xi_2,
```

```math
J_{\beta g} \leftarrow J_{\beta g} + A\big(\Xi_3 + \lfloor \Xi_1 \hat a \rfloor J_{\theta g}\big), \qquad
J_{\beta a} \leftarrow J_{\beta a} - A\,\Xi_1, \qquad
J_{\theta g} \leftarrow R_{step}\, J_{\theta g} + J_r(+\hat\omega \Delta t)\, \Delta t .
```

Note the **positive** argument of the right Jacobian in the $J_{\theta g}$ increment: from
$\mathrm{Exp}(-\hat\omega\Delta t + \delta b_g\,\Delta t) = R_{step}\,\mathrm{Exp}\big(J_r(-\hat\omega\Delta t)\,\delta b_g\,\Delta t\big)$,
transporting the new increment through $R_{step}$ gives
$R_{step}\, J_r(-\hat\omega\Delta t) = J_l(-\hat\omega\Delta t) = J_r(+\hat\omega\Delta t)$.
Writing $J_r(-\hat\omega\Delta t)$ here (the flavor stored by `compute_Xi_sum` for the stock
propagator's noise Jacobian) is wrong by $O(\|\hat\omega\|\Delta t)$ per step — measured
$1.0\times10^{-3}$ relative at 800 Hz, *below* the original $2\times10^{-3}$ oracle tolerance,
which is how it initially slipped through.

These give **analytic bias columns** in the measurement Jacobian of every epoch-snapped
observation (`epoch_bridge_bias_cols`): camera residuals correct $b_g, b_a$ *through the bridge*,
which is what makes the bound measurement a first-class EKF update rather than a fixed-lag hack.
All coupling signs are pinned by a finite-difference oracle (`test_preint_bridge`), not by
convention arguments — two sign errors in the literature-derived forms were caught exactly this
way, and the $J_r$-flavor subtlety above was later caught by review and is now pinned too: the
oracle runs at two IMU rates (800 and 200 Hz) under a $10^{-6}$ relative tolerance — three orders
below the wrong flavor's error, and comfortably above the exact form's measured residual
($\approx 4\times10^{-9}$). Caveat that earns its own knob: if timing or RS is grossly mismodeled, these same columns
are the conduit by which systematic image error drives the IMU biases (a hardware bring-up run
walked $b_a$ to $0.45\,\mathrm{m/s^2}$ this way); `epoch_bridge_bias_cols: false` is the
bring-up escape hatch.

Jacobians with respect to the epoch clone and $\delta$ itself use FEJ values throughout, so the
bridge does not disturb the observability structure of the standard FEJ-EKF [8].

---

## 7. Rolling-shutter measurement model

With the whole-frame residual $\delta_i$ handled *exactly* by the bridge, what remains is the
**row-dependent** part of the time model of §1. For a measurement on row $m$, the additional time
beyond the (bridged or plain) clone instant $t$ is

```math
\Delta(m) \;=\; \Delta t_{d,i} \;+\; t_{r,i}\,\tfrac{m}{M} \;\equiv\; \Delta t_{d,i} + t_{r,i}\, u,
\qquad u \in [0,1] \ \text{the row fraction},
```

and since $|\Delta| \le T_i$ (one frame period) it is linearized with clone kinematics
$({}^{I}\omega,\ {}^{G}v_I)$ — the standard RS treatment [5,6], but with the crucial difference
that here it models **only sub-frame effects**, never the inter-camera phase (that is the
bridge's job, done without linearization error).

First order in $\Delta$:

```math
{}^{I(t+\Delta)}_{G}R \approx \mathrm{Exp}\!\big({-{}^{I}\omega\,\Delta}\big)\; {}^{I(t)}_{G}R,
\qquad
{}^{G}p_{I(t+\Delta)} \approx {}^{G}p_{I(t)} + {}^{G}v_I\,\Delta,
```

```math
z = \pi\!\left( {}^{C_i}_{I}R\; {}^{I(t+\Delta)}_{G}R \big( {}^{G}p_f - {}^{G}p_{I(t+\Delta)} \big) + {}^{C_i}p_I \right) + n .
```

Differentiating through $\Delta$ gives the two readout columns implemented in
`UpdaterHelper::get_feature_jacobian_full` / `get_feature_jacobian_representation`:

**Clone side** (the observing pose moved during readout):

```math
\frac{\partial\, {}^{I_i}p_f}{\partial t_{r,i}}
= -\Big( \big\lfloor {}^{I}\omega \big\rfloor\, {}^{I_i}p_f \;+\; {}^{I_i}_{G}R\; {}^{G}v_I \Big)\, u ,
\qquad
H_{t_r} = J_\pi\; {}^{C}_{I}R\; \frac{\partial\, {}^{I_i}p_f}{\partial t_{r,i}} .
```

**Anchor side** (for `ANCHORED_*` representations the anchor pose was itself captured
row-shifted, so the *global* feature position inherits a readout sensitivity):

```math
\frac{\partial\, {}^{G}p_f}{\partial t_{r,a}}
= \Big( {}^{I_a}_{G}R^\top \big\lfloor \omega_a \big\rfloor\, {}^{I_a}p_f \;+\; {}^{G}v_{I_a} \Big)\, u_a ,
\qquad {}^{I_a}p_f = {}^{I}_{C}R \big( {}^{C_a}p_f - {}^{C}p_I \big).
```

The same $\Delta$-transport generates the $\pm$ paired $\Delta t_{d}$ columns of §4 (set $u = 1$,
differentiate w.r.t. $t_{d,i}$ and $-t_{d,r}$).

**Value/column separation — the load-bearing rule.** The RS *correction* (the value transport
above) is applied whenever $t_{r,i} \ne 0$ at value **or** linearization point; the RS *column*
(estimation) is added only when that camera's readout is a registered state
(declared-`rolling` **and** `calib_cam_readout: true`), the motion is not degenerate-gated (§4),
and the clone carries stored kinematics. Linearization uses FEJ readout iff the state is
estimated. This separation is why a **wrong readout is worse than none**: the row-mean of the
readout error, $t_r^{err}/2$ on average over rows, aliases directly into that camera's $t_{d,i}$
(observed: a $15\,\mathrm{ms}$ guess aliased $-4.3\,\mathrm{ms}$ into dt), while the row-*dependent* part poisons
the calibration states through $H_{t_r}$-shaped residual structure. Declaring a camera `rolling`
with a zero readout is therefore permitted — loudly — as a deliberate bring-up state (RS
unmodeled, error absorbed by an inflated $\chi^2$ gate), whereas *estimating* from a zero seed is
refused at boot: FEJ pins the linearization at $0$ and the prior fights every step of the
recovery.

Magnitude honesty: unmodeled RS costs up to $f \|\omega\| t_r$ pixels at the last row
($450 \cdot 1 \cdot 0.003\text{--}0.006 \approx 1.4\text{--}2.7\,\mathrm{px}$ per rad/s for a 3–6 ms readout) — this,
not the bridge noise of §8, is the dominant unmodeled term during bring-up, and it is what the
relaxed `up_msckf_chi2_multipler` in the bring-up configs is absorbing.

---

## 8. Bridge transport noise: why $Q_{tp}$ is measured, bounded, and *not* used

The bridge of §6 transports the state across $\delta$ using noisy IMU samples, so the bound
measurement's honest covariance is

```math
R_{\text{eff}} \;=\; \sigma_{px}^2 I \;+\; H_\delta\, Q_{tp}\, H_\delta^\top,
\qquad
Q_{tp} = \mathrm{cov}\big(\theta_\delta,\ \alpha_\delta\big) \in \mathbb{R}^{6\times 6},
```

where $Q_{tp}$ obeys the standard discrete recursion over the bridge interval (state order
$(\theta, \alpha, \beta)$, $Q_d = \mathrm{diag}(\sigma_g^2/\Delta t\, I_3,\ \sigma_a^2/\Delta t\, I_3)$):

```math
Q \leftarrow \Phi\, Q\, \Phi^\top + G\, Q_d\, G^\top,
\qquad
\Phi = \begin{bmatrix} R_{step} & 0 & 0 \\ A\lfloor \Xi_2 \hat a \rfloor & I & \Delta t I \\ A\lfloor \Xi_1 \hat a \rfloor & 0 & I \end{bmatrix},
\qquad
G = \begin{bmatrix} J_r(\hat\omega\Delta t)\, \Delta t & 0 \\ A\,\Xi_4 & -A\,\Xi_2 \\ A\,\Xi_3 & -A\,\Xi_1 \end{bmatrix},
```

with the noise columns of $G$ matching the bias columns of §6 entry for entry — as they must,
since $n_g, n_a$ enter $\hat\omega = \omega_m - b_g - n_g$, $\hat a = a_m - b_a - n_a$ exactly
like the biases (the $\theta$-row therefore carries the same $J_r(+\hat\omega\Delta t)$ increment
as $J_{\theta g}$, and the same sign).

This recursion was implemented, measured, and then **deliberately removed** from
`compute_bridge`; it is preserved in full above, should it ever be needed. The engineering
argument, in decreasing order of importance:

1. **Magnitude.** The bridge never exceeds the binding horizon $\delta_{max} = \beta\, T_r$.
   The deployed reference rig (60 Hz reference camera, `epoch_bind_factor: 1.5` in its hardware
   config — not the in-repo defaults) gives $\delta_{max} = 1.5/60 = 25\,\mathrm{ms}$; at the in-repo
   defaults ($\beta = 1.2$, 30 Hz `voxl_sim`) it is $40\,\mathrm{ms}$, which scales $e_\theta$ below by
   $1.26\times$ and $e_p$ by $2\times$ without changing the conclusion. Worst-case 1-σ pixel
   impact at $\delta_{max} = 25\,\mathrm{ms}$, using the
   *kalibr-inflated* sheet noises ($\sigma_g = 9.7\!\times\!10^{-4}\,\mathrm{rad/s/\sqrt{Hz}}$,
   $\sigma_a = 2.3\!\times\!10^{-2}\,\mathrm{m/s^2/\sqrt{Hz}}$, $f = 450\,\mathrm{px}$, depth $d$):

   ```math
   e_\theta \le f\,\sigma_g \sqrt{\delta_{max}} \approx 0.069\,\text{px},
   \qquad
   e_p \le \frac{f\,\sigma_a\, \delta_{max}^{3/2}}{\sqrt{3}\, d} \approx \frac{0.024}{d[\text{m}]}\,\text{px}.
   ```

   Against $\sigma_{px} = 1.2$ this is a $\le 0.33\%$ variance perturbation — three orders below
   the unmodeled-RS term of §7 that the same update is already digesting.

2. **The part that matters is already modeled.** Bridge error decomposes into a
   *bias-correlated systematic* component and a *zero-mean white* component. The systematic part —
   being wrong about $b_g, b_a$ at the bridge's linearization — **is** in the filter, to first
   order, via the analytic $J_b$ columns of §6. Only the white part is neglected, and item 1
   bounds it.

3. **Structure.** The white transport error is *common-mode across every feature of the snapped
   frame*: correct treatment is a per-epoch 6-DOF error state $\eta_\delta \sim \mathcal N(0, Q_{tp})$
   marginalized at epoch close, or equivalently dense cross-feature blocks in $R$. Naive diagonal
   inflation — the only cheap option — mis-models the correlation and breaks the per-feature
   nullspace projection / QR compression pipeline's optimality anyway. The honest choices are
   "expensive and correct" or "neglect with a bound"; item 1 licenses the latter.

4. **Consistency evidence.** The NEES gold-standard suite passes with the term absent: unsynced
   dual-mono $0.385\,\mathrm{m}$ / NEES $11.2$ against the synced baseline's $0.436\,\mathrm{m}$ / NEES $32$;
   GS+RS async $0.352\,\mathrm{m}$ / NEES $7.7$ with $94\%$ readout recovery. No optimism signature.

**Revisit criterion.** Re-land the recursion (as a per-epoch error state, not R-inflation) when
$f\,\sigma_g\sqrt{\delta_{max}} \gtrsim \sigma_{px}/3$. No single factor crosses that line at the
reference values — a slow reference camera ($\delta_{max} \gtrsim 100\,\mathrm{ms}$) alone reaches
$0.14\,\mathrm{px}$ against the $0.4\,\mathrm{px}$ threshold — but pairings do: a narrow-FOV lens
($f \gtrsim 2000\,\mathrm{px}$) with a sub-$0.5\,\mathrm{px}$ front end, a slow reference with a narrow-FOV
lens, or any of these compounded by MEMS-grade-noise regressions. Also revisit if a soak shows
epoch-snapped residuals running systematically hotter $\chi^2$ than same-camera unsnapped ones.

---

## 9. Reset semantics

Two reset classes exist, with one shared contract: **the sensor clock is continuous across
resets**, so timestamps alone can never distinguish estimation episodes — every consumer keyed by
update times must be purged explicitly.

**Soft reset** (`VioManager::soft_reset(SoftResetCause)`, causes `CLIENT` and `DIVERGENCE`)
resets the EKF while preserving the expensive front-end state: feature database, active tracks,
and IMU history survive, so re-initialization is warm rather than a cold $\sim 2\,\mathrm{s}$ window
rebuild. Specifically it:

- rebuilds state and covariance, re-pins the value and FEJ of every $t_{d,i}$ and $t_{r,i}$ at
  the **configured** calibration (the config's calibration knowledge survives the reset; online
  refinements and their transient history are deliberately discarded — a reset returns to the
  trusted prior, not to estimates the divergence may have poisoned);
- snapshots the live $b_g, b_a$ and their (random-walk age-inflated) sigmas as a **reset bias
  prior** for the initializer, gated by `init_dyn_reset_prior_max_{bg,ba}` /
  `..._max_sigma_{bg,ba}` with sigma floors and a `DIVERGENCE`-cause inflation — the main
  conditioner of the gravity/accel-bias valley on re-init;
- **purges the episode**: `used_features_map.clear()` (steady-state pruning by `margtimestep()`
  cannot do this — a warm-started window legitimately reaches back *past* the reset instant, so
  pre-reset entries would sit inside the new window and downstream consumers would present stale
  evidence as current), and resets the ZUPT episode flags `did_zupt_update` /
  `has_moved_since_zupt` (they are written only while initialized; left stale, the first lets the
  publisher report fresh-covariance CEP as quality throughout re-init, the second disables ZUPT
  for the entire new episode under `zupt_only_at_beginning`).

**Hard reset** destroys and reconstructs the `VioManager`. The dying manager's ingest rings
dispose every still-queued frame through the processed-callback (`processed=false` — the
exactly-once handle-release contract of §3). Teardown housekeeping is not a health event: the
transport error codes it raises are cleared after the swap, and the supervising health layer
auto-resets only on *estimator-poisoning* codes — a dropped frame is data the async ingest
legitimately disposes (ordering, staleness, teardown), not evidence the filter is corrupt.
Declared `camN_fps` re-seeds the ingest gates and the epoch horizon so the fresh manager runs at
design width from its first frame (§3).

---

## 10. Configuration reference (all keys added by this work)

### Per-camera acquisition declarations (estimator config, `camN_*`)

Declare what each physical camera **is**; the estimator validates the declaration against the
rest of the configuration at boot and fails fast on contradictions.

| Key | Type | Semantics |
|---|---|---|
| `camN_shutter` | `"global"` / `"rolling"` | Authoritative shutter declaration. `global`: readout pinned to 0 forever, **never** estimated (no spurious DOF); any configured nonzero readout is zeroed with a warning. `rolling`: RS model of §7 active whenever the readout value is nonzero. Missing on a multi-camera rig: legacy inference from the readout value, with a warning (that is what a stale deployed config looks like). Invalid string: fatal. |
| `camN_readout_time_s` | float ≥ 0 | Readout (first→last row, seconds). Priority: this key **wins over** kalibr-chain `t_readout` (survives recalibration); absent both → 0. `rolling` + 0 is allowed as the declared bring-up state "RS deliberately unmodeled" (loud warning; see §7 for why a wrong value is worse than none). `rolling` + 0 + `calib_cam_readout: true` is **fatal**: estimating from a zero seed is the documented FEJ trap. Must satisfy $t_r \le 1/\text{fps}$ (fatal otherwise). |
| `camN_fps` | float ≥ 0, finite | Nominal rate of the **streamed sensor mode**; `0` (or absent) means *undeclared* — no EMA seeding, no readout bound. When positive: seeds the ingest staleness EMA and the epoch binding horizon from the first frame (cold start and every hard reset), and sanity-bounds the readout. Negative or non-finite is fatal. Warns if any camera is declared faster than the epoch reference (§5). |

### Asynchronous multi-camera (estimator config)

| Key | Default | Semantics |
|---|---|---|
| `epoch_mode` | `false` | Master switch for epoch-anchored cloning (§5). `false` preserves the stock synced behavior bit-identically. A rig with ≥2 cameras, `use_stereo: false`, and `epoch_mode: false` triggers the **config-drift guard** — a loud boot warning that this exact combination is what a stale deployed config looks like, and that it dead-reckons into an auto-reset loop on hardware. |
| `cam_imu_dt_ref_camid` | `0` | Clock-reference camera $r$ (§4). **Must be the fastest camera** (§5). |
| `epoch_bind_factor` | `1.2` | Binding horizon in reference periods: bind iff $0 \le t_{f,i} - t_e \le \beta\, T_r$ (§5). Larger admits slower cameras' phase excursions; it also bounds $\delta_{max}$ in the $Q_{tp}$ analysis (§8). |
| `epoch_bridge_bias_cols` | `true` | Analytic $J_b$ bias columns on bridged measurements (§6). Disable only as a bring-up escape hatch when gross timing/RS mismodeling is suspected of driving the biases. |
| `async_ring_size` | `16` | Per-camera SPSC ring capacity (frames). |
| `async_guard` | `0.002` | IMU coverage guard $g$ (s) — invariant **I2** of §3. |
| `async_stale_factor` | `1.5` | Liveness bound — invariant **I3**: a stream silent longer than this multiple of its EMA period cannot block the ordered release. |

### Rolling-shutter calibration (estimator config)

| Key | Default | Semantics |
|---|---|---|
| `calib_cam_readout` | `false` | Online refinement of $t_{r,i}$ — **only** for declared-`rolling` cameras; the readout *value* is applied regardless (value/column separation, §7). |
| `calib_cam_readout_init_sigma` | `0.001` | Initial 1-σ prior on estimated readouts (s). $1\,\mathrm{ms}$ suits a measured seed; $\approx 5\,\mathrm{ms}$ a datasheet guess (§4). Must be positive (fatal otherwise). |
| `dt_calib_gate` | `false` | Opt-in: freeze dt/readout **columns** under degenerate window motion (§4); values stay applied. |
| `dt_calib_gate_min_omega`, `dt_calib_gate_min_vel_spread` | `0.10` | Window excitation thresholds — peak $\|\omega\|$ ($\mathrm{rad/s}$) and velocity spread ($\mathrm{m/s}$); the gate engages when **both** fall below threshold (§4). |

### Kalibr chain (per camera)

`timeshift_cam_imu` is now read **per camera** (it seeds each $t_{d,i}$); `t_readout` remains the
fallback readout source, overridden by `camN_readout_time_s`.

---

## 11. Validation

- **Simulation A/B** (frozen sim baseline, unsynced dual-mono, 7.3 ms phase, 5.2 ms initial dt
  error): $0.385\,\mathrm{m}$ / $0.60^\circ$ / NEES $11.2$ vs synced baseline $0.436\,\mathrm{m}$ / $0.40^\circ$ / NEES $32$;
  pre-fix $41.9\,\mathrm{m}$ / NEES $5041$. Mixed GS+RS: $0.352\,\mathrm{m}$ / NEES $7.7$, readout recovered to
  $94\%$ of truth.
- **Synced-path parity**: FNV-1a hash `6d38d0ae8d4fc3ae` of the simulated estimator stream,
  identical to the pre-branch baseline with `epoch_mode: false` — the legacy path is untouched by
  construction, not by hope. Measured by `scripts/sim_ab.sh parity`, which compiles an
  out-of-tree hash jig against both trees; the jig source is a development artifact and is not
  committed, so reproducing the check requires rebuilding it per the S0 stage notes.
- **Finite-difference oracles**: `test_preint_bridge` pins every $J_b$ coupling sign *and*
  Jacobian flavor in §6 (FD at two IMU rates, $10^{-6}$ relative tolerance), plus mean exactness
  against dense integration and the bias-correction convention; `test_async_buffer` exercises the
  ordering/liveness/disposal invariants of §3; `test_async_dual` runs the full async dual-mono
  estimator against truth.
- **Runtime telemetry**: per-update `[EPOCH]` line (snapped/fallback counts, late/full/bogus
  drops) — a healthy rig holds fallbacks and all drop counters at zero (§3, §5).

---

## 12. References

[1] P. Geneva, K. Eckenhoff, W. Lee, Y. Yang, G. Huang, "OpenVINS: A Research Platform for
Visual-Inertial Estimation," ICRA 2020.

[2] K. Eckenhoff, P. Geneva, G. Huang, "Closed-form Preintegration Methods for Graph-based
Visual-Inertial Navigation," IJRR 2019.

[3] A. I. Mourikis, S. I. Roumeliotis, "A Multi-State Constraint Kalman Filter for Vision-aided
Inertial Navigation," ICRA 2007.

[4] M. Li, A. I. Mourikis, "Online Temporal Calibration for Camera-IMU Systems: Theory and
Algorithms," IJRR 2014.

[5] M. Li, A. I. Mourikis, "Vision-aided Inertial Navigation with Rolling-Shutter Cameras," IJRR
2014.

[6] T. Qin, S. Shen, "Online Temporal Calibration for Monocular Visual-Inertial Systems," IROS
2018.

[7] Y. Yang, P. Geneva, X. Zuo, G. Huang, "Online Self-Calibration for Visual-Inertial Navigation:
Models, Analysis, and Degeneracy," IEEE T-RO 2023.

[8] J. A. Hesch, D. G. Kottas, S. L. Bowman, S. I. Roumeliotis, "Consistency Analysis and
Improvement of Vision-aided Inertial Navigation," IEEE T-RO 2014.

[9] P. Furgale, J. Rehder, R. Siegwart, "Unified Temporal and Spatial Calibration for Multi-Sensor
Systems," IROS 2013.

[10] Y. Yang, B. P. W. Babu, C. Chen, G. Huang, L. Ren, "Analytic Combined IMU Integration (ACI²)
for Visual-Inertial Navigation," ICRA 2020.
