"""Text, plot, HTML and video outputs for a log Inspection (CLI and studio export)."""

import base64
import html as H
import os
import subprocess
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .checks import SEVERITIES
from .media import Tonemapper, array_to_qimage, black_level_for, cv2, per_channel

MARK = {"problem": "[!]", "warning": "[w]", "info": "[i]", "expected": "[~]"}


def title(report):
    if not report.name:
        return "Log"
    return f"{'IMU' if report.kind == 'imu' else 'CAM' if report.kind == 'camera' else report.kind.upper()}: {report.name}"


def count_text(inspection):
    parts = []
    for severity in SEVERITIES:
        n = inspection.count(severity)
        if n:
            plural = "s" if n != 1 and severity in ("problem", "warning") else ""
            parts.append(f"{n} {severity}{plural}")
    return ", ".join(parts)


def text_lines(inspection, max_list=20):
    lines = []
    for report in inspection.channels:
        lines.append("")
        lines.append(f"=== {title(report)}")
        lines.extend(f"  {fact}" for fact in report.facts)
        found = inspection.for_channel(report.name)
        if report.name and not any(f.severity == "problem" for f in found):
            lines.append("  [ok] no data loss detected")
        for finding in found:
            lines.append(f"  {MARK[finding.severity]} {finding.message}")
            for k, event in enumerate(finding.events):
                if k >= max_list:
                    lines.append(f"      ... {len(finding.events) - max_list} more")
                    break
                row = f" row {event.row:6d}" if event.row is not None else ""
                lines.append(f"     {row}  t={event.time:9.4f}s  {event.text}")
    counts = count_text(inspection)
    lines.append("")
    lines.append("=" * 60)
    verdict = {
        "pass": "PASS",
        "warnings": "PASS with warnings",
        "problems": "PROBLEMS FOUND",
    }[inspection.verdict]
    lines.append(f"RESULT: {verdict}" + (f" ({counts})" if counts else ""))
    return lines


# ---------------------------------------------------------------------------
# matplotlib figures (optional dependency of the CLI and HTML export)
# ---------------------------------------------------------------------------
def plot_figures(inspection, out_dir, show=False):
    """PNG figures; returns their paths ([] when matplotlib is unavailable)."""
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    log, out_dir = inspection.log, Path(out_dir)
    gap_factor = inspection.options.get("gap_factor", 1.5)
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    files = []

    def marks(ax, name):
        for finding in inspection.for_channel(name):
            if finding.severity in ("problem", "expected") and finding.events:
                color = "red" if finding.severity == "problem" else "teal"
                for event in finding.events[:400]:
                    ax.axvline(event.time, color=color, alpha=0.35, lw=0.8)

    def save(fig, name):
        fig.tight_layout()
        path = out_dir / f"{name.strip('/').replace('/', '_')}.png"
        fig.savefig(path, dpi=110)
        files.append(str(path))
        if not show:
            plt.close(fig)

    for imu in log.imus:
        t = imu.times
        for kind, cols, unit, letters in (
            ("gyro", (1, 2, 3), "rad/s", "GX GY GZ"),
            ("accel", (4, 5, 6), "m/s²", "AX AY AZ"),
        ):
            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            for ax, col, color, letter in zip(axes[:3], cols, colors, letters.split()):
                ax.plot(t, imu.data[:, col], lw=0.5, color=color)
                ax.set_ylabel(f"{letter} [{unit}]")
                ax.grid(alpha=0.3)
                marks(ax, imu.name)
            ax = axes[3]
            for col, color, letter in zip(cols, colors, letters.split()):
                ax.plot(t, imu.data[:, col], lw=0.5, color=color, label=letter)
            if kind == "accel":
                ax.plot(
                    t,
                    np.linalg.norm(imu.data[:, 4:7], axis=1),
                    lw=0.5,
                    color="k",
                    label="|a|",
                )
            legend = ax.legend(loc="upper right", ncol=4, fontsize=8)
            for line in legend.get_lines():
                line.set_linewidth(2)
            ax.set_ylabel(f"all [{unit}]")
            ax.set_xlabel("time since log start [s]")
            ax.grid(alpha=0.3)
            marks(ax, imu.name)
            fig.suptitle(f"{imu.name} - {kind}")
            save(fig, f"{imu.name}_{kind}")
        if len(t) > 1:
            fig, ax = plt.subplots(figsize=(14, 4))
            dt = np.diff(t) * 1e3
            ax.plot(t[1:], dt, ".", ms=1.5)
            nominal = np.median(dt[dt > 0]) if np.any(dt > 0) else 0
            if nominal:
                ax.axhline(
                    gap_factor * nominal,
                    color="red",
                    ls="--",
                    lw=0.8,
                    label=f"gap threshold ({gap_factor:g}x)",
                )
                ax.legend(loc="upper right")
            ax.set_ylabel("dt [ms]")
            ax.set_xlabel("time since log start [s]")
            ax.set_title(f"{imu.name} - sample interval")
            ax.grid(alpha=0.3)
            save(fig, f"{imu.name}_dt")
        if imu.temperature_available:
            fig, ax = plt.subplots(figsize=(14, 3))
            ax.plot(t, imu.data[:, 7], lw=0.8, color="#d62728")
            ax.set_ylabel("temperature [°C]")
            ax.set_xlabel("time since log start [s]")
            ax.set_title(f"{imu.name} - temperature")
            ax.grid(alpha=0.3)
            save(fig, f"{imu.name}_temperature")
    cams = [c for c in log.cameras if len(c) > 1]
    if cams:
        fig, axes = plt.subplots(
            len(cams), 1, figsize=(14, 2.6 * len(cams)), sharex=True, squeeze=False
        )
        for ax, cam in zip(axes[:, 0], cams):
            dt = np.diff(cam.times) * 1e3
            ax.plot(cam.times[1:], dt, ".", ms=2)
            nominal = np.median(dt[dt > 0]) if np.any(dt > 0) else 0
            if nominal:
                ax.axhline(gap_factor * nominal, color="red", ls="--", lw=0.8)
            marks(ax, cam.name)
            ax.set_ylabel("dt [ms]")
            ax.set_title(cam.name, fontsize=9)
            ax.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("time since log start [s]")
        fig.suptitle(
            "camera frame intervals (red: gap threshold / problems, teal: expected events)"
        )
        save(fig, "camera_frame_intervals")
        exposed = [
            c for c in cams if c.exposure is not None and np.nanmax(c.exposure) > 0
        ]
        if exposed:
            fig, axes = plt.subplots(
                len(exposed),
                1,
                figsize=(14, 2.4 * len(exposed)),
                sharex=True,
                squeeze=False,
            )
            for ax, cam in zip(axes[:, 0], exposed):
                ax.plot(cam.times, cam.exposure * 1e3, lw=0.8, label="exposure [ms]")
                if cam.gain is not None and np.nanmax(cam.gain) > 0:
                    twin = ax.twinx()
                    twin.plot(cam.times, cam.gain, lw=0.8, color="#ff7f0e")
                    twin.set_ylabel("gain", color="#ff7f0e")
                ax.set_ylabel("exposure [ms]")
                ax.set_title(cam.name, fontsize=9)
                ax.grid(alpha=0.3)
            axes[-1, 0].set_xlabel("time since log start [s]")
            fig.suptitle("camera exposure and gain")
            save(fig, "camera_exposure_gain")
    return files


# ---------------------------------------------------------------------------
# camera clips
# ---------------------------------------------------------------------------
def _rotate(frame, degrees):
    k = {"0": 0, "90": -1, "180": 2, "270": 1}.get(str(int(degrees) % 360))
    if k is None:
        raise ValueError(f"rotation {degrees} is not one of 0/90/180/270")
    return np.ascontiguousarray(np.rot90(frame, k)) if k else frame


def _label(frame, lines):
    if cv2 is None:
        return frame
    frame = frame if frame.ndim == 3 else np.dstack([frame] * 3)
    frame = np.ascontiguousarray(frame)
    y = frame.shape[0] - 10 - 22 * (len(lines) - 1)
    for text in lines:
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return frame


def make_camera_video(
    cam,
    out_dir,
    clip_seconds=10.0,
    fps=30.0,
    width=960,
    crf=26,
    rotate="",
    black_level="",
    bayer_pattern="",
    jobs=4,
    log=print,
):
    """Sped-up H.264 clip of evenly subsampled frames. Returns {name, path, info} or None."""
    if not cam.has_pictures or not len(cam):
        log(f"video: {cam.name}: no per-frame images, skipped")
        return None
    pattern = per_channel(bayer_pattern, cam.name, cam.pattern or "RGGB").upper()
    cam.pattern = pattern
    rows = np.unique(
        np.linspace(
            0, len(cam) - 1, min(len(cam), max(2, int(round(clip_seconds * fps))))
        )
        .round()
        .astype(int)
    )
    span = float(cam.times[rows[-1]] - cam.times[rows[0]])
    fps = min(fps, len(rows) / clip_seconds)
    speed = span / (len(rows) / fps) if len(rows) > 1 else 1.0
    mapper = Tonemapper(
        black_level_for(cam.name, cam.sensor, black_level), width, smooth=True
    )
    degrees = per_channel(rotate, cam.name, "0")
    out_path = Path(out_dir) / f"{cam.name.strip('/').replace('/', '_')}.mp4"
    ffmpeg, written, failures, started = None, 0, 0, time.perf_counter()

    def load(row):
        try:
            return cam.picture(row)
        except Exception as error:  # a single unreadable frame must not abort the clip
            return error

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        queue, rows_iter = deque(), iter(rows)
        for row in rows_iter:
            queue.append((row, pool.submit(load, row)))
            if len(queue) >= 2 * jobs:
                break
        while queue:
            row, future = queue.popleft()
            nxt = next(rows_iter, None)
            if nxt is not None:
                queue.append((nxt, pool.submit(load, nxt)))
            picture = future.result()
            if isinstance(picture, Exception):
                failures += 1
                continue
            frame = _rotate(mapper(picture), degrees)
            label = f"{cam.name}  t={cam.times[row]:7.2f}s"
            if cam.frame_ids is not None:
                label += f"  id {cam.frame_ids[row]}"
            frame = _label(frame, [label, f"x{speed:.1f} speed"])
            if frame.ndim == 2:
                frame = np.dstack([frame] * 3)
            h, w = frame.shape[:2]
            frame = np.ascontiguousarray(frame[: h // 2 * 2, : w // 2 * 2])
            if ffmpeg is None:
                try:
                    ffmpeg = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-loglevel",
                            "error",
                            "-y",
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "-s",
                            f"{frame.shape[1]}x{frame.shape[0]}",
                            "-r",
                            f"{fps:.4f}",
                            "-i",
                            "-",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "veryfast",
                            "-crf",
                            str(crf),
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            str(out_path),
                        ],
                        stdin=subprocess.PIPE,
                    )
                except FileNotFoundError:
                    log("video: ffmpeg not found; install ffmpeg to export clips")
                    return None
            ffmpeg.stdin.write(frame.tobytes())
            written += 1
    if ffmpeg is None:
        return None
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        log(f"video: {cam.name}: ffmpeg failed")
        return None
    kind = {
        "bayer": f"RAW Bayer {pattern}, debayered",
        "mono16": "16-bit mono, auto-contrast",
        "mono8": "8-bit mono",
        "rgb": "color",
    }.get(picture.kind if not isinstance(picture, Exception) else "", "")
    info = (
        f"{written} of {len(cam)} frames (every ~{len(cam) / max(written, 1):.1f}), {fps:.0f} fps, x{speed:.1f} speed"
        + (f", {kind}" if kind else "")
        + (f", {failures} unreadable" if failures else "")
    )
    log(
        f"video: {out_path}  [{info}; {1e3 * (time.perf_counter() - started) / max(written, 1):.1f} ms/frame]"
    )
    return {"name": cam.name, "path": str(out_path), "info": info}


def snapshot_png(cam, row, path, black_level=""):
    """Write one display-mapped frame as PNG (uses Qt; no OpenCV needed)."""
    picture = cam.picture(row)
    image = array_to_qimage(
        Tonemapper(black_level_for(cam.name, cam.sensor, black_level), 0, smooth=False)(
            picture
        )
    )
    if not image.save(str(path)):
        raise OSError(f"cannot write {path}")
    return path


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
HTML_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--muted:#5d6676;--line:#dde1e7;
--problem:#c62828;--problem-bg:#fdecea;--warning:#9a5b00;--warning-bg:#fff4e0;
--expected:#0f766e;--expected-bg:#e0f5f2;--info:#3b5bab;--info-bg:#e8eefc;--ok:#1a7f37;--ok-bg:#e6f4ea}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
--bg:#14171c;--card:#1c2027;--fg:#e4e7ec;--muted:#9aa3b2;--line:#2c323c;
--problem:#ff6b6b;--problem-bg:#3a1c1c;--warning:#f0b35a;--warning-bg:#3a2c16;
--expected:#5fd3c3;--expected-bg:#16322f;--info:#9db4f5;--info-bg:#1e2740;--ok:#4cc26b;--ok-bg:#183222}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:32px 0 12px}
.sub{color:var(--muted);font-size:13px;word-break:break-all}
.banner{margin:18px 0;padding:14px 16px;border-radius:8px;font-weight:600}
.banner.pass{background:var(--ok-bg);color:var(--ok)}
.banner.warnings{background:var(--warning-bg);color:var(--warning)}
.banner.problems{background:var(--problem-bg);color:var(--problem)}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;font-size:14px}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase}
.tbl{overflow-x:auto}
.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.pill.ok{background:var(--ok-bg);color:var(--ok)}
.pill.problem{background:var(--problem-bg);color:var(--problem)}
.pill.warning{background:var(--warning-bg);color:var(--warning)}
.pill.info{background:var(--info-bg);color:var(--info)}
.pill.expected{background:var(--expected-bg);color:var(--expected)}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:10px 0}
summary{cursor:pointer;padding:10px 14px;font-weight:600}
pre{margin:0;padding:4px 14px 14px;overflow-x:auto;font-size:12.5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre .problem{color:var(--problem);font-weight:600} pre .warning{color:var(--warning)} pre .ok{color:var(--ok)}
pre .info{color:var(--info)} pre .expected{color:var(--expected)}
figure{margin:0 0 20px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
figure img{width:100%;height:auto;display:block;background:#fff;border-radius:4px}
figcaption{color:var(--muted);font-size:13px;margin-top:6px}
.vgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,460px),1fr));gap:16px}
.vgrid figure{margin:0}
video{width:100%;height:auto;display:block;background:#000;border-radius:4px}
"""


def write_html(inspection, path, plots=(), videos=(), embed_video=True):
    log = inspection.log
    path = Path(path)
    status_text = {
        "ok": "OK",
        "problem": "problem",
        "warning": "warning",
        "info": "info",
        "expected": "expected",
    }

    def pill(status, count=0):
        label = status_text.get(status, status)
        if count > 1 and status in ("problem", "warning"):
            label = f"{count} {label}s"
        return f'<span class="pill {status}">{H.escape(label)}</span>'

    rows = []
    for report in inspection.channels:
        found = inspection.for_channel(report.name)
        status = inspection.status(report.name)
        worst = [f for f in found if f.severity == status]
        rate = f"{report.rate:.2f} Hz" if report.rate else ""
        first = next(
            (f.message for f in found if f.severity in ("problem", "warning")), ""
        )
        rows.append(
            f"<tr><td>{H.escape(report.kind)}</td><td><b>{H.escape(report.name or 'log')}</b></td><td>{rate}</td>"
            f"<td>{report.count if report.name else ''}</td><td>{pill(status, len(worst))}</td><td>{H.escape(first)}</td></tr>"
        )
    findings = "".join(
        f"<tr><td>{pill(f.severity)}</td><td>{H.escape(f.channel or 'log')}</td><td>{H.escape(f.message)}</td>"
        f"<td>{'' if f.time is None else f'{f.time:.3f} s'}</td><td>{len(f.events) or ''}</td></tr>"
        for f in inspection.findings
    )

    def line(text):
        stripped = text.strip()
        for severity, mark in MARK.items():
            if stripped.startswith(mark):
                return f'<span class="{severity}">{H.escape(text)}</span>'
        if stripped.startswith("[ok]"):
            return f'<span class="ok">{H.escape(text)}</span>'
        return H.escape(text)

    blocks, current = [], None
    for text in text_lines(inspection, 50):
        if text.startswith("=== "):
            current = [text[4:], []]
            blocks.append(current)
        elif text.startswith("====="):
            current = None
        elif current is not None and text:
            current[1].append(text)
    details = "".join(
        f"<details{' open' if any(MARK['problem'] in l or MARK['warning'] in l for l in body) else ''}><summary>{H.escape(head)}</summary><pre>{chr(10).join(line(l) for l in body)}</pre></details>"
        for head, body in blocks
    )
    figures = []
    for plot in plots:
        data = base64.b64encode(Path(plot).read_bytes()).decode()
        caption = Path(plot).stem.replace("_", " ")
        figures.append(
            f'<figure><img alt="{H.escape(caption)}" loading="lazy" src="data:image/png;base64,{data}"><figcaption>{H.escape(caption)}</figcaption></figure>'
        )
    clips = []
    for video in videos:
        if embed_video:
            src = (
                "data:video/mp4;base64,"
                + base64.b64encode(Path(video["path"]).read_bytes()).decode()
            )
        else:
            src = H.escape(os.path.relpath(video["path"], path.parent))
        clips.append(
            f'<figure><video controls loop muted playsinline preload="metadata" src="{src}"></video><figcaption><b>{H.escape(video["name"])}</b> &middot; {H.escape(video["info"])}</figcaption></figure>'
        )
    verdict = inspection.verdict
    counts = count_text(inspection) or "no findings"
    banner = {
        "pass": "PASS: no skipped frames or samples, no other problems found",
        "warnings": "PASS with warnings: no data loss, but review the warnings below",
        "problems": f"{inspection.count('problem')} problem{'s' if inspection.count('problem') != 1 else ''} found",
    }[verdict]
    name = H.escape(log.name)
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Log inspection {name}</title><style>{HTML_CSS}</style></head>
<body><main>
<h1>Log inspection: {name}</h1>
<div class="sub">{H.escape(str(log.path))}<br>{H.escape(log.format_label)} &middot; {log.duration:.3f} s &middot; {len(log.included)} of {len(log.channels)} channels inspected{(" &middot; note: " + H.escape(str(log.meta.get("note")))) if log.meta.get("note") else ""}</div>
<div class="banner {verdict}">{H.escape(banner)} <span style="font-weight:400">({H.escape(counts)})</span></div>
<h2>Channels</h2>
<div class="tbl"><table><thead><tr><th>Type</th><th>Channel</th><th>Rate</th><th>Count</th><th>Status</th><th>First problem or warning</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<h2>Findings</h2>
<div class="tbl"><table><thead><tr><th>Severity</th><th>Channel</th><th>Finding</th><th>First at</th><th>Events</th></tr></thead><tbody>{findings or '<tr><td colspan="5">No findings.</td></tr>'}</tbody></table></div>
<h2>Details</h2>
{details}
{f'<h2>Camera clips</h2><div class="vgrid">{"".join(clips)}</div>' if clips else ""}
<h2>Plots</h2>
{"".join(figures) if figures else '<p class="sub">Plots unavailable (disabled, or matplotlib not installed).</p>'}
</main></body></html>
"""
    path.write_text(doc, encoding="utf-8")
    return path
