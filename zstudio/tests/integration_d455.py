"""Opt-in real stereo/RGB/IMU capture check; requires the D455 ingress bridge.

Run in the camera's sourced ROS 2 environment with a NEW --output directory.
No calibration certificate is inferred from this short acquisition check.
"""

from pathlib import Path
import argparse
import json
import time

from PyQt5.QtWidgets import QApplication
from zstudio.app import configure_application
from zstudio.model import read_record
from zstudio.sources import SourceSettings
from zstudio.window import StudioWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--leave-open", action="store_true")
    args = parser.parse_args()
    if not 3 <= args.seconds <= 60:
        parser.error("Choose a capture duration between 3 and 60 seconds")
    settings = SourceSettings(**json.loads(args.profile.read_text()))
    cameras = settings.camera_sources()
    args.output.mkdir(parents=True, exist_ok=False)
    app = QApplication([])
    configure_application(app)
    window = StudioWindow()
    window.show()

    def wait_for(condition, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.processEvents()
            if condition():
                return True
            time.sleep(0.01)
        return False

    leave_open = False
    try:
        window.start_source(settings)
        assert wait_for(lambda: window.capture_button.isEnabled(), 20), (
            window.console.toPlainText()
        )
        record = args.output / "session.bin"
        window.start_capture(str(record))
        wait_for(lambda: False, args.seconds)
        assert window.capture_active, window.console.toPlainText()
        snapshot = window.source.snapshot()
        assert all(
            camera["frame"] is not None and len(camera["frame"].points) > 10
            for camera in snapshot["cameras"]
        ), "TrackKLT overlays missing from a camera"
        window.grab().save(str(args.output / "live.png"))
        window.stop_capture()
        assert wait_for(
            lambda: not window.capture_active and window.loader is None, 10
        ), window.console.toPlainText()
        session = read_record(record)
        counts = [
            sum(frame.cam == camera for frame in session.frames)
            for camera in range(len(cameras))
        ]
        assert (
            session.complete
            and len(session.cameras) == len(cameras)
            and min(counts) > 30
            and len(session.imu) > 300
        )
        assert Path(window.session.path).resolve() == record.resolve(), (
            "GUI did not reopen the completed recording"
        )
        result = dict(
            schema=1,
            complete=session.complete,
            frames_per_camera=counts,
            imu_samples=len(session.imu),
            camera_rates=list(session.camera_rates),
            imu_rate=session.imu_rate,
            tracker="TrackKLT",
            temperature_available=False,
            calibration_performed=False,
            record=str(record.resolve()),
        )
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output / "console.txt").write_text(window.console.toPlainText())
        print(json.dumps(result), flush=True)
        if args.leave_open:
            window.start_source(settings)
            leave_open = True
            app.exec_()
    finally:
        if not leave_open:
            window.close()
            wait_for(lambda: not window.isVisible(), 8)


if __name__ == "__main__":
    main()
