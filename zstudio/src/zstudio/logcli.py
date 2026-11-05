"""ov_log_inspector - sanity-check a voxl-logger directory or a ROS 1 / ROS 2 bag.

Checks (per channel):
  * IMU    : sample-interval gaps, duplicate / non-monotonic stamps, CSV index
             and batch continuity, ROS header.seq continuity, rate vs the
             configured rate, temperature drift, header-stamp vs bag-time latency
  * Camera : frame_id / header.seq gaps (beyond n_to_skip), repeated ids with
             identical images (thermal shutter frames are expected), timestamp
             gaps, missing / extra / empty / truncated image files, sample
             counts vs the log metadata, intrinsics / CameraInfo availability
  * Rig    : IMU covers every camera, stereo pairs are synchronized, the
             configured body->IMU orientation agrees with gravity at rest
             (extrinsics.conf or /tf_static); --stereo-check also compares the
             stereo pair geometry recovered from the images with the config
Outputs: report.txt, optional JSON, plots and a self-contained report.html,
and sped-up per-camera clips (--video; RAW10/RAW12 Bayer is debayered).

Severity: problem (data loss/corruption), warning, info, expected (known-benign,
e.g. thermal FFC shutter frames). Exit code 1 if any problem (or any warning
with --strict), 2 if the log cannot be opened.

Examples:
  ov_log_inspector log0005 --html --video --rotate hires_bayer=180
  ov_log_inspector flight.bag --only '/cam*,/imu0' --json -
  ov_log_inspector log0005 --list-channels
"""

import argparse
import json
import os
import sys
from pathlib import Path

from .checks import SEVERITIES, inspect_log, stereo_findings
from .logs import LogError, open_log, probe_log, select
from .logreport import make_camera_video, plot_figures, text_lines, write_html
from .media import use_opencl

GAP_FACTOR_DEFAULT = 1.5


def default_out(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir() and (path / "info.json").is_file():
        return path / "log_inspector"
    if path.name == "metadata.yaml":
        path = path.parent
    return path.parent / f"{path.stem if path.is_file() else path.name}_inspector"


def build_parser():
    ap = argparse.ArgumentParser(
        prog="ov_log_inspector",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "log",
        help="voxl-logger directory (info.json), ROS 1 .bag, or ROS 2 bag directory/.db3/.mcap",
    )
    ap.add_argument(
        "-o",
        "--out",
        help="output directory (default: <log>/log_inspector, or <bag>_inspector next to a bag)",
    )
    sel = ap.add_argument_group(
        "channel selection (pipes / topics come from info.json or the bag metadata)"
    )
    sel.add_argument(
        "--list-channels", action="store_true", help="list pipes/topics and exit"
    )
    sel.add_argument(
        "--only",
        default="",
        help="comma-separated globs of channels to inspect (default: all supported)",
    )
    sel.add_argument(
        "--exclude", default="", help="comma-separated globs of channels to skip"
    )
    chk = ap.add_argument_group("checks")
    chk.add_argument(
        "--gap-factor",
        type=float,
        default=GAP_FACTOR_DEFAULT,
        help=f"dt > FACTOR x nominal period is a gap (default {GAP_FACTOR_DEFAULT})",
    )
    chk.add_argument(
        "--temp-drift-warn",
        type=float,
        default=10.0,
        help="warn when IMU temperature spans more than this [C] (default 10)",
    )
    chk.add_argument(
        "--rate-tolerance",
        type=float,
        default=0.05,
        help="warn when the IMU rate differs from its configured rate by more than this fraction (default 0.05)",
    )
    chk.add_argument(
        "--stereo-check",
        action="store_true",
        help="recover stereo pair geometry from images and compare with the config (needs OpenCV)",
    )
    chk.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    out = ap.add_argument_group("output")
    out.add_argument(
        "--max-list",
        type=int,
        default=20,
        help="events listed per finding in report.txt (default 20)",
    )
    out.add_argument(
        "--json",
        nargs="?",
        const="report.json",
        metavar="PATH",
        help="write the inspection as JSON (PATH or '-' for stdout; default report.json in the output dir)",
    )
    out.add_argument("--no-plots", action="store_true")
    out.add_argument(
        "--html", action="store_true", help="write a self-contained report.html"
    )
    out.add_argument("--show", action="store_true", help="show plots interactively")
    out.add_argument(
        "-q", "--quiet", action="store_true", help="only print the result line"
    )
    vid = ap.add_argument_group("camera clips")
    vid.add_argument(
        "--video",
        action="store_true",
        help="make a sped-up mp4 clip per camera (needs ffmpeg)",
    )
    vid.add_argument(
        "--clip-seconds",
        type=float,
        default=10.0,
        help="clip length (default 10); frames are skipped to fit",
    )
    vid.add_argument(
        "--video-fps", type=float, default=30.0, help="output fps (default 30)"
    )
    vid.add_argument(
        "--video-width",
        type=int,
        default=960,
        help="max output width in px (default 960)",
    )
    vid.add_argument(
        "--video-crf",
        type=int,
        default=26,
        help="x264 CRF quality, lower = better/bigger (default 26)",
    )
    vid.add_argument(
        "--bayer-pattern",
        default="",
        help="CFA pattern, e.g. RGGB or 'RGGB,hires_bayer=GRBG' (default: from the encoding, else RGGB)",
    )
    vid.add_argument(
        "--rotate",
        default="",
        help="rotate clips clockwise by 90/180/270: '180' or 'hires_bayer=180,tracking_left=90'",
    )
    vid.add_argument(
        "--black-level",
        default="",
        help="Bayer black level in 10-bit units: '50' or 'hires_bayer=64' (default: by sensor, else 50)",
    )
    vid.add_argument(
        "--gpu",
        choices=("auto", "on", "off"),
        default="auto",
        help="debayer with OpenCL via OpenCV when available (default auto)",
    )
    vid.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="threads reading/unpacking frames",
    )
    vid.add_argument(
        "--video-link",
        action="store_true",
        help="reference the mp4 files from report.html instead of embedding them",
    )
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    # With --json - stdout carries only JSON; human-readable text goes to stderr.
    stream = sys.stderr if args.json == "-" else sys.stdout

    def say(*parts):
        if not args.quiet:
            print(*parts, file=stream)

    try:
        kind, name, channels = probe_log(args.log)
    except LogError as error:
        print(f"ov_log_inspector: {error}", file=sys.stderr)
        return 2
    if args.list_channels:
        width = max((len(c.name) for c in channels), default=10)
        for c in channels:
            state = "yes" if c.supported else f"no ({c.reason})"
            print(
                f"{c.name:<{width}}  {c.kind:<11} {c.count:>8}  {c.type:<32} inspect: {state}"
            )
        return 0
    chosen = select(channels, args.only, args.exclude)
    if not chosen:
        print(
            "ov_log_inspector: no channels selected (see --list-channels)",
            file=sys.stderr,
        )
        return 2
    try:
        log = open_log(args.log, chosen)
    except (LogError, OSError) as error:
        print(f"ov_log_inspector: {error}", file=sys.stderr)
        return 2
    try:
        inspection = inspect_log(
            log, args.gap_factor, args.temp_drift_warn, args.rate_tolerance
        )
        if args.stereo_check:
            from .checks import _stereo_pairs
            from .geometry import stereo_geometry

            for a, b in _stereo_pairs(log.cameras):
                if a.intrinsics is None or b.intrinsics is None:
                    say(f"stereo-check: {a.name}/{b.name} skipped (intrinsics missing)")
                    continue
                try:
                    result = stereo_geometry(a, b, a.intrinsics, b.intrinsics)
                except Exception as error:
                    say(f"stereo-check: {a.name}/{b.name}: {error}")
                    continue
                for finding in stereo_findings(log, a, b, result):
                    inspection.add(finding)
        out_dir = Path(args.out).expanduser() if args.out else default_out(args.log)
        lines = text_lines(inspection, args.max_list)
        say("\n".join([f"log: {log.path}"] + lines[:-1]))
        print(lines[-1], file=stream)
        if args.json == "-":
            json.dump(inspection.to_dict(), sys.stdout, indent=1, default=float)
            sys.stdout.write("\n")
        needs_dir = not (
            args.json == "-" and args.no_plots and not args.html and not args.video
        ) or args.json not in (None, "-")
        if needs_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "report.txt").write_text(
                "\n".join([f"log: {log.path}"] + lines) + "\n", encoding="utf-8"
            )
            say(f"report: {out_dir / 'report.txt'}")
            if args.json not in (None, "-"):
                target = (
                    Path(args.json)
                    if os.path.isabs(args.json) or os.sep in args.json
                    else out_dir / args.json
                )
                target.write_text(
                    json.dumps(inspection.to_dict(), indent=1, default=float),
                    encoding="utf-8",
                )
                say(f"json:   {target}")
        plots = [] if args.no_plots else plot_figures(inspection, out_dir, args.show)
        if not args.no_plots and not plots:
            say("plots: matplotlib not available, skipped")
        videos = []
        if args.video:
            say(f"video: debayer backend: {use_opencl(args.gpu)}")
            for cam in log.cameras:
                clip = make_camera_video(
                    cam,
                    out_dir,
                    args.clip_seconds,
                    args.video_fps,
                    args.video_width,
                    args.video_crf,
                    args.rotate,
                    args.black_level,
                    args.bayer_pattern,
                    args.jobs,
                    say,
                )
                if clip:
                    videos.append(clip)
        if args.html:
            html = write_html(
                inspection,
                out_dir / "report.html",
                plots,
                videos,
                embed_video=not args.video_link,
            )
            say(f"html:   {html}")
        for plot in plots:
            say(f"plot:   {plot}")
        if args.show and plots:
            import matplotlib.pyplot as plt

            plt.show()
        return inspection.exit_code(args.strict)
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "build_parser", "SEVERITIES"]
