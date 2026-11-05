"""Argument-vector process integration; no shell or solver work on the GUI thread."""

from pathlib import Path
import json
import os
import re
import shutil


def find_executable(name):
    found = shutil.which(name)
    if found:
        return found
    for variable in ("AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
        for prefix in os.environ.get(variable, "").split(os.pathsep):
            if prefix:
                candidate = Path(prefix) / "lib" / "ov_zcalib" / name
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate)
    # Source checkout convenience for a normal ROS-free CMake build.
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        if (parent / "ov_zcalib").is_dir():
            for build in ("build", "build-studio", "build-host", "build_native"):
                candidate = parent / build / "ov_zcalib" / name
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate)
            break
    return ""


def calibration_command(
    binary,
    source,
    output,
    report,
    cam_mode=1,
    budget=60.0,
    threads=2,
    profile="record",
    voxl=False,
    readout_s=0.0,
    cam_pipe="tracking_front",
):
    if not binary or not Path(binary).is_file() or not os.access(binary, os.X_OK):
        raise ValueError(
            "Choose the built ov_zcalibrate executable in Calibration settings."
        )
    if not Path(source).exists():
        raise ValueError("Calibration input does not exist.")
    if Path(output).exists() or Path(report).exists():
        raise ValueError(
            "Choose new output filenames; existing calibration artifacts are preserved."
        )
    args = [
        "--voxl" if voxl else "--replay",
        str(source),
        "--out",
        str(output),
        "--report-json",
        str(report),
        "--cam-mode",
        str(cam_mode),
        "--solve-budget-s",
        str(float(budget)),
        "--threads",
        str(threads),
    ]
    if profile != "record":
        if voxl:
            if profile == "flight":
                args.append("--flight")
        else:
            args.extend(["--profile", profile])
    if voxl:
        args.extend(["--tr", str(readout_s), "--cam-pipe", cam_pipe])
    else:
        sidecar = Path(str(source) + ".capture.json")
        if sidecar.is_file():
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                if metadata.get("temperature_available") is False:
                    args.append("--temperature-unavailable")
            except (OSError, ValueError, AttributeError):
                pass  # Optional provenance is unknown; it cannot manufacture temperature.
    return binary, args


def capture_command(binary, settings, record):
    if not binary or not Path(binary).is_file() or not os.access(binary, os.X_OK):
        raise ValueError(
            "ov_zcalib_capture was not found. Build ov_zcalib with the default tracker and source the workspace."
        )
    if Path(record).exists():
        raise ValueError("This recording already exists; choose a new filename.")
    parameters = settings.capture_parameters(record)
    if settings.transport == "ROS 2":
        args = ["--ros-args"]
        for key, value in parameters.items():
            # rclcpp parameter syntax is YAML. Quote all strings including paths
            # with spaces, colons, or YAML-looking topic names.
            encoded = json.dumps(value)
            args.extend(["-p", f"{key}:={encoded}"])
        args.extend(["-r", f"/ov_zcalib/tracks:={settings.tracks_topic}"])
    else:
        args = [
            f"_{key}:={json.dumps(value) if isinstance(value, list) else value}"
            for key, value in parameters.items()
        ]
        args.append(f"/ov_zcalib/tracks:={settings.tracks_topic}")
    return binary, args


def stage_from_log(text):
    """Only explicit engine evidence changes the displayed stage."""
    lower = text.lower()
    if "state: abort" in lower or "replay failed" in lower:
        return "ABORT"
    if "state: done" in lower:
        return "DONE"
    states = re.findall(r"\[session\] -> ([A-Z_]+)", text)
    if states:
        state = states[-1]
        return {"SOLVE_REFINE": "SOLVE", "THERMAL_HOLD": "COLLECT"}.get(state, state)
    if "[session] commit:" in lower:
        return "COMMIT"
    if "[session] verify" in lower:
        return "VERIFY"
    if "[session] fusing" in lower or "[session] stage-select" in lower:
        return "SOLVE"
    if "-> collect" in lower or "enter collect" in lower:
        return "COLLECT"
    if "bootstrap" in lower or "hand-eye" in lower:
        return "BOOTSTRAP"
    return None
