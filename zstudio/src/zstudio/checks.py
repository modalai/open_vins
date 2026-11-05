"""Log health checks shared by ``ov_log_inspector`` and the studio's Inspect page.

``inspect_log`` turns a LogSource into structured findings. Severity:

* problem  -- data loss or corruption (skipped samples, missing files, bad stamps)
* warning  -- likely to break calibration/VIO or worth a look (config mismatches)
* info     -- measurements and context
* expected -- known-benign sensor behaviour, e.g. thermal shutter (FFC) frames

Only problems fail the CLI (warnings too with ``--strict``).
"""

from dataclasses import dataclass, field
import math

import numpy as np

from .geometry import angle_between_deg, find_rest, gravity_in_body, rotation_angle_deg

SEVERITIES = ("problem", "warning", "info", "expected")
_RANK = {s: i for i, s in enumerate(SEVERITIES)}
LOG_CHANNEL = ""


@dataclass
class Event:
    time: float  # seconds since the log epoch (stream time base)
    text: str
    row: int = None


@dataclass
class Finding:
    severity: str
    channel: str
    code: str
    message: str
    events: list = field(default_factory=list)

    @property
    def time(self):
        return self.events[0].time if self.events else None


@dataclass
class ChannelReport:
    name: str
    kind: str
    count: int
    rate: float = None
    nominal: float = None  # median period [s]
    start: float = None
    end: float = None
    facts: list = field(default_factory=list)


@dataclass
class Inspection:
    log: object
    channels: list
    findings: list
    options: dict = field(default_factory=dict)

    def for_channel(self, name):
        return [f for f in self.findings if f.channel == name]

    def status(self, name):
        found = self.for_channel(name)
        return min((f.severity for f in found), key=_RANK.get, default="ok")

    def count(self, severity):
        return sum(1 for f in self.findings if f.severity == severity)

    @property
    def verdict(self):
        if self.count("problem"):
            return "problems"
        return "warnings" if self.count("warning") else "pass"

    def exit_code(self, strict=False):
        return 1 if self.count("problem") or (strict and self.count("warning")) else 0

    def add(self, finding):
        self.findings.append(finding)
        self.findings.sort(key=lambda f: (_RANK[f.severity], f.channel, f.code))

    def to_dict(self):
        log = self.log
        return {
            "schema": 1,
            "log": {
                "path": str(log.path),
                "format": log.kind,
                "name": log.name,
                "duration_s": log.duration,
                "meta": {
                    k: v
                    for k, v in log.meta.items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                },
                "included": list(log.included),
                "excluded": [
                    c.name for c in log.channels if c.name not in log.included
                ],
            },
            "verdict": self.verdict,
            "counts": {s: self.count(s) for s in SEVERITIES},
            "channels": [
                {
                    "name": c.name,
                    "kind": c.kind,
                    "count": c.count,
                    "rate_hz": c.rate,
                    "status": self.status(c.name),
                    "facts": c.facts,
                }
                for c in self.channels
            ],
            "findings": [
                {
                    "severity": f.severity,
                    "channel": f.channel,
                    "code": f.code,
                    "message": f.message,
                    "events": [
                        {
                            "t": e.time,
                            "text": e.text,
                            **({"row": e.row} if e.row is not None else {}),
                        }
                        for e in f.events
                    ],
                }
                for f in self.findings
            ],
            "options": self.options,
        }


def fmt_ms(seconds):
    return f"{seconds * 1e3:.3f} ms"


def _timing(name, times, gap_factor, unit, out, report):
    """Gaps, duplicates and reversals of a stamp sequence (seconds)."""
    if len(times) < 2:
        return None
    dt = np.diff(times)
    positive = dt[dt > 0]
    nominal = float(np.median(positive)) if len(positive) else 0.0
    report.nominal = nominal or None
    if nominal > 0:
        report.rate = 1.0 / nominal
        report.facts.append(
            f"nominal rate {1.0 / nominal:.3f} Hz (period {fmt_ms(nominal)}); "
            f"dt min/median/max {fmt_ms(dt.min())} / {fmt_ms(nominal)} / {fmt_ms(dt.max())}, std {fmt_ms(dt.std())}"
        )
        span = times[-1] - times[0]
        expected = int(round(span / nominal)) + 1
        report.facts.append(
            f"duration {span:.3f} s; {expected} {unit} expected at the nominal rate (deficit {expected - len(times)})"
        )
        gaps = np.where(dt > gap_factor * nominal)[0]
        if len(gaps):
            missed = [max(1, int(round(dt[i] / nominal)) - 1) for i in gaps]
            out.append(
                Finding(
                    "problem",
                    name,
                    "timing.gap",
                    f"{len(gaps)} timing gaps (dt > {gap_factor:g}x nominal), ~{sum(missed)} {unit} skipped",
                    [
                        Event(
                            float(times[i]),
                            f"dt {fmt_ms(dt[i])}, ~{m} {unit} missing",
                            int(i),
                        )
                        for i, m in zip(gaps, missed)
                    ],
                )
            )
    back = np.where(dt < 0)[0]
    if len(back):
        out.append(
            Finding(
                "problem",
                name,
                "timing.nonmonotonic",
                f"{len(back)} timestamps go backwards",
                [
                    Event(
                        float(times[i + 1]),
                        f"stamp moves back {fmt_ms(-dt[i])}",
                        int(i + 1),
                    )
                    for i in back
                ],
            )
        )
    return dt


def _declared(name, stream, unit, out):
    if stream.declared is not None and stream.declared != len(stream):
        out.append(
            Finding(
                "problem",
                name,
                "count.declared",
                f"log metadata declares {stream.declared} {unit}, {len(stream)} were read",
            )
        )


def _bag_stamps(name, stream, out, report):
    if stream.receive is None or not len(stream.receive):
        return
    zero = np.where(stream.stamps_ns <= 0)[0]
    if len(zero):
        out.append(
            Finding(
                "problem",
                name,
                "stamp.unset",
                f"{len(zero)} messages have an unset (zero) header stamp",
                [
                    Event(float(stream.receive[i]), "header.stamp = 0", int(i))
                    for i in zero[:200]
                ],
            )
        )
    valid = stream.stamps_ns > 0
    if not valid.any():
        return
    latency = stream.receive[valid] - stream.times[valid]
    report.facts.append(
        "bag time - header stamp: median {}, p99 {}, max {}".format(
            fmt_ms(float(np.median(latency))),
            fmt_ms(float(np.percentile(latency, 99))),
            fmt_ms(float(latency.max())),
        )
    )
    ahead = np.where(latency < -0.001)[0]
    if len(ahead):
        out.append(
            Finding(
                "warning",
                name,
                "stamp.future",
                f"{len(ahead)} header stamps are later than the bag's receive time (up to {fmt_ms(-float(latency.min()))}); "
                "publisher and recorder clocks disagree",
            )
        )


def _sequence(
    name, ids, times, skip, out, payload=None, thermal=False, label="frame_id"
):
    """Counter continuity; repeated ids are classified by comparing payloads."""
    keep = ids >= 0
    idx = np.where(keep)[0]
    ids, times = ids[keep], times[keep]
    if len(ids) < 2:
        return
    step = skip + 1
    d = np.diff(ids)
    jumps = np.where(d > step)[0]
    if len(jumps):
        missed = int(np.sum(d[jumps] - step))
        out.append(
            Finding(
                "problem",
                name,
                "sequence.gap",
                f"{len(jumps)} {label} gaps, {missed} frames skipped"
                + (f" beyond n_to_skip={skip}" if skip else ""),
                [
                    Event(
                        float(times[i]),
                        f"{label} {ids[i]} -> {ids[i + 1]} ({d[i] - step} skipped)",
                        int(idx[i]),
                    )
                    for i in jumps
                ],
            )
        )
    shutter, resent, changed, unknown = [], [], [], []
    for i in np.where(d == 0)[0]:
        a, b = int(idx[i]), int(idx[i + 1])
        same = None
        if payload is not None:
            try:
                pa, pb = payload(a), payload(b)
                same = None if pa is None or pb is None else pa == pb
            except Exception:
                same = None
        event = Event(
            float(times[i + 1]),
            f"{label} {ids[i]} repeated, dt {fmt_ms(times[i + 1] - times[i])}",
            b,
        )
        if same is None:
            unknown.append(event)
        elif not same:
            event.text += ", different image"
            changed.append(event)
        elif thermal:
            event.text += ", identical image (shutter frame)"
            shutter.append(event)
        else:
            event.text += ", identical image"
            resent.append(event)
    if shutter:
        period = (
            np.median(np.diff([e.time for e in shutter]))
            if len(shutter) > 1
            else float("nan")
        )
        every = f", every ~{period:.1f} s" if math.isfinite(period) else ""
        out.append(
            Finding(
                "expected",
                name,
                "thermal.shutter",
                f"{len(shutter)} thermal shutter (FFC) frames re-sent with a new timestamp{every}",
                shutter,
            )
        )
    if resent:
        out.append(
            Finding(
                "problem",
                name,
                "sequence.resent",
                f"{len(resent)} frames re-sent with the same {label} and a new timestamp",
                resent,
            )
        )
    if changed:
        out.append(
            Finding(
                "problem",
                name,
                "sequence.repeat",
                f"{len(changed)} repeated {label}s carry a different image",
                changed,
            )
        )
    if unknown:
        out.append(
            Finding(
                "problem",
                name,
                "sequence.repeat",
                f"{len(unknown)} repeated {label}s",
                unknown,
            )
        )
    back = np.where(d < 0)[0]
    if len(back):
        out.append(
            Finding(
                "problem",
                name,
                "sequence.backwards",
                f"{label} goes backwards at {len(back)} places",
                [
                    Event(
                        float(times[i + 1]),
                        f"{label} {ids[i]} -> {ids[i + 1]}",
                        int(idx[i + 1]),
                    )
                    for i in back
                ],
            )
        )


def _counter_used(ids, report, label):
    """A counter that never changes (e.g. header.seq left 0 by bag converters) carries no continuity."""
    valid = ids[ids >= 0]
    if len(valid) > 1 and np.all(valid == valid[0]):
        report.facts.append(
            f"{label} is constant ({valid[0]}); not populated by the producer, continuity not checked"
        )
        return False
    return len(valid) > 1


def _imu(log, imu, gap_factor, temp_drift_warn, rate_tolerance, out):
    report = ChannelReport(imu.name, "imu", len(imu))
    if len(imu):
        report.start, report.end = float(imu.times[0]), float(imu.times[-1])
    report.facts.append(f"samples: {len(imu)}")
    _declared(imu.name, imu, "samples", out)
    _timing(imu.name, imu.times, gap_factor, "samples", out, report)
    dups = np.where(np.diff(imu.stamps_ns) == 0)[0] if len(imu) > 1 else []
    if len(dups):
        out.append(
            Finding(
                "problem",
                imu.name,
                "timing.duplicate",
                f"{len(dups)} duplicate timestamps",
                [
                    Event(float(imu.times[i + 1]), "duplicate stamp", int(i + 1))
                    for i in dups
                ],
            )
        )
    if imu.rows is not None and len(imu.rows) > 1:
        bad = np.where(np.diff(imu.rows) != 1)[0]
        if len(bad):
            out.append(
                Finding(
                    "problem",
                    imu.name,
                    "rows.gap",
                    f"CSV index column is not contiguous at {len(bad)} places",
                    [
                        Event(
                            float(imu.times[i + 1]),
                            f"row index {imu.rows[i]} -> {imu.rows[i + 1]}",
                            int(i + 1),
                        )
                        for i in bad
                    ],
                )
            )
    if imu.batch is not None and len(imu.batch):
        ids, counts = np.unique(imu.batch, return_counts=True)
        jumps = np.where(np.diff(ids) > 1)[0]
        report.facts.append(
            f"batches: {len(ids)} (ids {ids[0]}..{ids[-1]}), samples/batch min/median/max {counts.min()}/{int(np.median(counts))}/{counts.max()}"
        )
        if len(jumps):
            # voxl-logger increments batch_id on every pipe read, including empty reads.
            report.facts.append(
                f"{len(jumps)} batch_id jumps ({int(np.sum(np.diff(ids)[jumps] - 1))} ids unused) - empty pipe reads, not data loss"
            )
        if np.any(np.diff(imu.batch) < 0):
            out.append(
                Finding(
                    "problem", imu.name, "batch.backwards", "batch_id goes backwards"
                )
            )
    if imu.seq is not None:
        if _counter_used(imu.seq, report, "header.seq"):
            _sequence(imu.name, imu.seq, imu.times, 0, out, label="header.seq")
    if imu.configured_rate and report.rate:
        error = (report.rate - imu.configured_rate) / imu.configured_rate
        if abs(error) > rate_tolerance:
            out.append(
                Finding(
                    "warning",
                    imu.name,
                    "rate.configured",
                    f"measured {report.rate:.2f} Hz but voxl-imu-server.conf configures {imu.configured_rate:g} Hz ({error:+.1%})",
                )
            )
        else:
            report.facts.append(f"rate matches configured {imu.configured_rate:g} Hz")
    if imu.temperature_available:
        temp = imu.data[:, 7]
        finite = temp[np.isfinite(temp)]
        drift = float(finite[-1] - finite[0])
        report.facts.append(
            f"temperature {finite[0]:.1f} -> {finite[-1]:.1f} C (min {finite.min():.1f}, max {finite.max():.1f}, {drift:+.1f} C)"
        )
        if abs(float(finite.max() - finite.min())) > temp_drift_warn:
            out.append(
                Finding(
                    "warning",
                    imu.name,
                    "temperature.drift",
                    f"IMU temperature spans {finite.max() - finite.min():.1f} C ({finite[0]:.1f} -> {finite[-1]:.1f} C); biases may drift during the log",
                )
            )
    elif imu.receive is not None:
        report.facts.append(
            "temperature unavailable (sensor_msgs/Imu has no temperature)"
        )
    if imu.covariance_unavailable:
        out.append(
            Finding(
                "problem",
                imu.name,
                "imu.unavailable",
                "messages set angular_velocity/linear_acceleration covariance[0] = -1 (data unavailable); capture refuses this topic",
            )
        )
    _bag_stamps(imu.name, imu, out, report)
    return report


def _camera(log, cam, gap_factor, out):
    report = ChannelReport(cam.name, "camera", len(cam))
    if len(cam):
        report.start, report.end = float(cam.times[0]), float(cam.times[-1])
    report.facts.append(
        f"frames: {len(cam)}"
        + (f"; {cam.width}x{cam.height}" if cam.width else "")
        + (f"; {cam.encoding}" if cam.encoding else "")
    )
    if cam.sensor:
        shutter = (
            "thermal"
            if cam.thermal
            else "global shutter"
            if cam.global_shutter
            else "rolling shutter or unknown"
        )
        report.facts.append(f"sensor: {cam.sensor} ({shutter})")
    if cam.stamp_reference == "start":
        report.facts.append(
            "stamps: start of exposure; center-row mid-exposure = stamp + (readout + exposure) / 2"
        )
    if cam.skip:
        report.facts.append(f"n_to_skip={cam.skip}: intentional decimation")
    _declared(cam.name, cam, "frames", out)
    if cam.dropped_flag:
        out.append(
            Finding(
                "problem",
                cam.name,
                "logger.dropped",
                "info.json reports dropped_data = true (frame_id gap or a flushed, backed-up pipe)",
            )
        )
    label = "frame_id" if cam.stamp_reference == "start" else "header.seq"
    if cam.frame_ids is not None and _counter_used(cam.frame_ids, report, label):
        _sequence(
            cam.name,
            cam.frame_ids,
            cam.times,
            cam.skip,
            out,
            cam.payload if cam.has_pictures else None,
            cam.thermal,
            label,
        )
        ids = cam.frame_ids[cam.frame_ids >= 0]
        if len(ids):
            report.facts.append(
                f"{label} {ids[0]}..{ids[-1]} (span implies {ids[-1] - ids[0] + 1} frames)"
            )
    _timing(cam.name, cam.times, gap_factor, "frames", out, report)
    if len(cam) > 1:
        same = np.where(np.diff(cam.stamps_ns) == 0)[0]
        if len(same):
            out.append(
                Finding(
                    "problem",
                    cam.name,
                    "timing.duplicate",
                    f"{len(same)} duplicate timestamps",
                    [
                        Event(float(cam.times[i + 1]), "duplicate stamp", int(i + 1))
                        for i in same
                    ],
                )
            )
    if cam.exposure is not None and len(cam.exposure) and np.nanmax(cam.exposure) > 0:
        e = cam.exposure * 1e3
        report.facts.append(
            f"exposure min/median/max {np.nanmin(e):.3f} / {np.nanmedian(e):.3f} / {np.nanmax(e):.3f} ms"
        )
    if cam.gain is not None and len(cam.gain) and np.nanmax(cam.gain) > 0:
        report.facts.append(
            f"gain min/median/max {np.nanmin(cam.gain):g} / {np.nanmedian(cam.gain):g} / {np.nanmax(cam.gain):g}"
        )
    if cam.encoded:
        if cam.encoded_file:
            report.facts.append(f"encoded stream file: {cam.encoded_file}")
        else:
            out.append(
                Finding(
                    "problem",
                    cam.name,
                    "files.video",
                    "encoded channel but no video.h264/h265 file",
                )
            )
    elif cam.files is not None:
        missing = [i for i, f in enumerate(cam.files) if not f]
        if missing:
            out.append(
                Finding(
                    "problem",
                    cam.name,
                    "files.missing",
                    f"{len(missing)} CSV rows have no image file",
                    [
                        Event(float(cam.times[i]), "image file missing", i)
                        for i in missing
                    ],
                )
            )
        if cam.orphans:
            out.append(
                Finding(
                    "warning",
                    cam.name,
                    "files.orphan",
                    f"{len(cam.orphans)} image files are not listed in data.csv (first: {cam.orphans[:10]})",
                )
            )
        zero = [i for i, s in enumerate(cam.file_sizes) if cam.files[i] and s == 0]
        if zero:
            out.append(
                Finding(
                    "problem",
                    cam.name,
                    "files.empty",
                    f"{len(zero)} zero-byte image files",
                    [Event(float(cam.times[i]), "zero-byte file", i) for i in zero],
                )
            )
        raw = [
            i
            for i, f in enumerate(cam.files)
            if f
            and f.lower().endswith(
                (".raw8", ".raw10", ".raw12", ".nv12", ".nv21", ".yuv")
            )
        ]
        if raw:
            sizes = np.array([cam.file_sizes[i] for i in raw])
            values, counts = np.unique(sizes, return_counts=True)
            mode = values[np.argmax(counts)]
            odd = [i for i, s in zip(raw, sizes) if s != mode]
            if odd:
                out.append(
                    Finding(
                        "problem",
                        cam.name,
                        "files.size",
                        f"{len(odd)} raw frames differ from the typical size {mode} B (truncated?)",
                        [
                            Event(float(cam.times[i]), f"{cam.file_sizes[i]} B", i)
                            for i in odd
                        ],
                    )
                )
        if not missing and not cam.orphans:
            report.facts.append("every CSV row has an image file")
    # Intrinsics: required to track/calibrate; only expected for VIO cameras on VOXL.
    if cam.intrinsics is not None:
        k = cam.intrinsics["k"]
        report.facts.append(
            f"intrinsics ({cam.intrinsics['model']}): fx {k[0]:.2f} fy {k[1]:.2f} cx {k[2]:.2f} cy {k[3]:.2f} from {cam.intrinsics['source']}"
        )
        iw, ih = cam.intrinsics.get("width"), cam.intrinsics.get("height")
        if iw and ih and cam.width and (iw, ih) != (cam.width, cam.height):
            out.append(
                Finding(
                    "problem",
                    cam.name,
                    "intrinsics.size",
                    f"intrinsics are for {iw}x{ih} but frames are {cam.width}x{cam.height}",
                )
            )
    elif cam.intrinsics_problem:
        out.append(
            Finding("warning", cam.name, "intrinsics.invalid", cam.intrinsics_problem)
        )
    elif log.kind == "voxl":
        vio = log.is_vio_camera(cam.name) or cam.name.startswith("tracking")
        if vio and not cam.encoded:
            out.append(
                Finding(
                    "warning",
                    cam.name,
                    "intrinsics.missing",
                    f"per-unit intrinsics not in the log ({cam.intrinsics_expected}); tracking and calibration need them",
                )
            )
    else:
        out.append(
            Finding(
                "warning",
                cam.name,
                "intrinsics.missing",
                "no CameraInfo topic found in this image's namespace; tracking and calibration need one",
            )
        )
    if cam.info_size and cam.width and cam.info_size != cam.raw_size:
        out.append(
            Finding(
                "problem",
                cam.name,
                "intrinsics.size",
                f"CameraInfo is {cam.info_size[0]}x{cam.info_size[1]} but images are {cam.width}x{cam.height}",
            )
        )
    _bag_stamps(cam.name, cam, out, report)
    return report


def _stereo_pairs(cameras):
    names = {c.name: c for c in cameras}
    pairs = []
    for name, cam in names.items():
        for a, b in (("left", "right"), ("infra1", "infra2"), ("cam0", "cam1")):
            if a in name:
                other = name.replace(a, b)
                if other in names:
                    pairs.append((cam, names[other]))
    return pairs


def _cross(log, out, log_report):
    imus = log.imus
    for cam in log.cameras:
        if not imus or not len(cam):
            continue
        t0 = min(float(i.times[0]) for i in imus if len(i))
        t1 = max(float(i.times[-1]) for i in imus if len(i))
        outside = np.where((cam.times < t0 - 0.05) | (cam.times > t1 + 0.05))[0]
        if len(outside):
            out.append(
                Finding(
                    "warning",
                    cam.name,
                    "imu.coverage",
                    f"{len(outside)} frames fall outside the IMU's time span ({t0:.3f}..{t1:.3f} s)",
                    [
                        Event(float(cam.times[i]), "no IMU coverage", int(i))
                        for i in outside[:200]
                    ],
                )
            )
    for a, b in _stereo_pairs(log.cameras):
        if len(a) < 2 or len(b) < 2:
            continue
        j = np.clip(np.searchsorted(b.times, a.times), 1, len(b) - 1)
        nearest = np.where(
            np.abs(b.times[j - 1] - a.times) < np.abs(b.times[j] - a.times), j - 1, j
        )
        skew = np.abs(b.stamps_ns[nearest] - a.stamps_ns) * 1e-9
        median, worst = float(np.median(skew)), float(skew.max())
        text = f"stereo pair {a.name} / {b.name}: median skew {fmt_ms(median)}, max {fmt_ms(worst)} over {len(a)} frames"
        if median < 50e-6:
            text += " (hardware-synchronized stamps)"
        sev = "warning" if median > 0.001 else "info"
        out.append(
            Finding(
                sev,
                a.name,
                "stereo.sync",
                text if sev == "info" else text + "; not synchronized",
            )
        )
    rig = log.rig
    if rig.edges and rig.body:
        for imu in imus:
            frame = imu.frame if imu.frame in rig.frames else rig.match(imu.name)
            if not frame:
                out.append(
                    Finding(
                        "info",
                        imu.name,
                        "rig.imu",
                        f"{rig.source} has no frame for this IMU; gravity check skipped",
                    )
                )
                continue
            transform = rig.transform(rig.body, frame)
            if transform is None:
                out.append(
                    Finding(
                        "info",
                        imu.name,
                        "rig.imu",
                        f"{frame} is not connected to {rig.body} in {rig.source}",
                    )
                )
                continue
            rest = find_rest(imu.data)
            if rest is None:
                out.append(
                    Finding(
                        "info",
                        imu.name,
                        "gravity.rest",
                        "no stationary period found; gravity check skipped",
                    )
                )
                continue
            t_a, t_b, measured = rest
            predicted = transform[0].T @ gravity_in_body(rig.body_convention)
            angle = angle_between_deg(measured, predicted)
            detail = (
                f"at rest ({t_a:.2f}-{t_b:.2f} s) the accelerometer reads [{measured[0]:+.2f} {measured[1]:+.2f} {measured[2]:+.2f}] m/s^2; "
                f"{rig.source} ({rig.body} {rig.body_convention} -> {frame}) predicts [{predicted[0]:+.2f} {predicted[1]:+.2f} {predicted[2]:+.2f}] "
                f"for a level vehicle: {angle:.1f} deg apart"
            )
            if angle > 60:
                out.append(
                    Finding(
                        "problem",
                        imu.name,
                        "gravity.orientation",
                        f"{rig.body} -> {frame} orientation contradicts gravity; "
                        + detail,
                        [Event(t_a, "rest window")],
                    )
                )
            elif angle > 25:
                out.append(
                    Finding(
                        "warning",
                        imu.name,
                        "gravity.orientation",
                        f"{rig.body} -> {frame} orientation disagrees with gravity (or the vehicle rested tilted); "
                        + detail,
                        [Event(t_a, "rest window")],
                    )
                )
            else:
                out.append(
                    Finding(
                        "info",
                        imu.name,
                        "gravity.orientation",
                        f"{rig.body} -> {frame} is consistent with gravity; " + detail,
                    )
                )
    if rig.edges:
        for cam in log.cameras:
            frame = cam.frame if cam.frame in rig.frames else None
            if not frame:
                out.append(
                    Finding(
                        "info",
                        cam.name,
                        "rig.camera",
                        f"camera {cam.camera or cam.name} has no entry in {rig.source}",
                    )
                )
    for error in getattr(log, "config_errors", []):
        out.append(Finding("warning", LOG_CHANNEL, "config.parse", error))


def inspect_log(
    log,
    gap_factor=1.5,
    temp_drift_warn=10.0,
    rate_tolerance=0.05,
    progress=None,
    cancelled=None,
):
    findings, channels = [], []
    streams = log.streams
    for n, stream in enumerate(streams):
        if cancelled and cancelled():
            raise InterruptedError("inspection cancelled")
        if stream.kind == "imu":
            channels.append(
                _imu(log, stream, gap_factor, temp_drift_warn, rate_tolerance, findings)
            )
        else:
            channels.append(_camera(log, stream, gap_factor, findings))
        if progress:
            progress((n + 1) / max(1, len(streams)), stream.name)
    for name, error in log.errors.items():
        if name in log.included:
            channel = log.channel(name)
            channels.append(
                ChannelReport(name, channel.kind if channel else "other", 0)
            )
            findings.append(
                Finding(
                    "problem",
                    name,
                    "channel.unreadable",
                    f"channel could not be read: {error}",
                )
            )
    summary = ChannelReport(LOG_CHANNEL, "log", len(streams))
    summary.facts.append(f"{log.format_label}: {log.path}")
    for key, label in (
        ("note", "note"),
        ("start_date", "started"),
        ("declared_duration", "declared duration"),
    ):
        value = log.meta.get(key)
        if value not in (None, ""):
            summary.facts.append(
                f"{label}: {value:.3f} s"
                if isinstance(value, float)
                else f"{label}: {value}"
            )
    excluded = [c for c in log.channels if c.name not in log.included]
    if excluded:
        summary.facts.append(
            "not inspected: "
            + ", ".join(f"{c.name} ({c.reason or 'excluded'})" for c in excluded)
        )
    if log.rig.edges:
        summary.facts.append(
            f"rig: {len(log.rig.edges)} transforms from {log.rig.source}"
            + (
                f", body frame {log.rig.body} ({log.rig.body_convention})"
                if log.rig.body
                else ""
            )
        )
    _cross(log, findings, summary)
    channels.insert(0, summary)
    findings.sort(key=lambda f: (_RANK[f.severity], f.channel, f.code))
    return Inspection(
        log,
        channels,
        findings,
        {
            "gap_factor": gap_factor,
            "temp_drift_warn": temp_drift_warn,
            "rate_tolerance": rate_tolerance,
        },
    )


def stereo_findings(log, a, b, result):
    """Findings for a geometry.stereo_geometry result, compared with the rig config."""
    out = []
    axis = result["axis"]
    base = result["baseline_dir"]
    text = (
        f"stereo geometry {a.name} -> {b.name} from {result['pairs']} image pairs ({result['inliers']} inliers): "
        f"relative rotation {result['rotation_deg']:.2f} deg about [{axis[0]:+.2f} {axis[1]:+.2f} {axis[2]:+.2f}] ({a.name} frame); "
        f"{b.name} sits along [{base[0]:+.2f} {base[1]:+.2f} {base[2]:+.2f}] (x right, y down, z forward), "
        f"bootstrap spread {result['baseline_spread_deg']:.1f} deg; baseline length is not observable from images"
    )
    if result["pure_rotation_inliers"] >= 0.8 * result["inliers"]:
        out.append(
            Finding(
                "warning",
                a.name,
                "stereo.geometry",
                text + "; parallax is too small for a reliable baseline direction",
            )
        )
        return out
    rig = log.rig
    if rig.edges and a.frame and b.frame:
        transform = rig.transform(a.frame, b.frame)
        if transform is not None:
            R_cfg, t_cfg = transform
            d_rot = rotation_angle_deg(result["R_ab"].T @ R_cfg)
            d_dir = (
                angle_between_deg(base, t_cfg)
                if np.linalg.norm(t_cfg) > 1e-6
                else float("nan")
            )
            cfg = t_cfg / max(np.linalg.norm(t_cfg), 1e-12)
            text += (
                f". {rig.source} says {b.frame} is at [{cfg[0]:+.2f} {cfg[1]:+.2f} {cfg[2]:+.2f}] in {a.frame} "
                f"({np.linalg.norm(t_cfg) * 1e3:.1f} mm) and rotated {rotation_angle_deg(R_cfg):.2f} deg; "
                f"disagreement: rotation {d_rot:.1f} deg, baseline direction {d_dir:.1f} deg"
            )
            if d_rot > 5 or (math.isfinite(d_dir) and d_dir > 30):
                out.append(Finding("warning", a.name, "stereo.geometry", text))
                return out
    out.append(Finding("info", a.name, "stereo.geometry", text))
    return out
