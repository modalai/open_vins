"""Opt-in ROS 2 graph → default TrackKLT → record → GUI replay → engine report.

Run in a sourced ROS 2/colcon workspace. Uses an isolated, localhost-only DDS
domain. Synthetic inconsistent image/IMU motion must not pass calibration.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["ROS_DOMAIN_ID"] = "186"
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication
from zstudio.engine import find_executable
from zstudio.model import read_record
from zstudio.sources import SourceSettings, TopicDiscovery
from zstudio.app import configure_application
from zstudio.window import PAGE_RESULTS, StudioWindow


def main():
    with TemporaryDirectory(prefix="zcalib_ros2_") as directory:
        root = Path(directory)
        QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(root))
        app = QApplication([])
        configure_application(app)
        window = StudioWindow()
        window.show()
        rclpy.init(args=[])
        node = Node("zcalib_test_publisher")
        image_pubs = [
            node.create_publisher(
                Image, f"/zcalib_test/cam{i}/image", qos_profile_sensor_data
            )
            for i in range(3)
        ]
        info_pubs = [
            node.create_publisher(
                CameraInfo, f"/zcalib_test/cam{i}/info", qos_profile_sensor_data
            )
            for i in range(3)
        ]
        imu_pub = node.create_publisher(
            Imu, "/zcalib_test/imu", qos_profile_sensor_data
        )
        settings = SourceSettings(
            image_topic="/zcalib_test/cam0/image",
            info_topic="/zcalib_test/cam0/info",
            imu_topic="/zcalib_test/imu",
            extra_cameras=[
                dict(
                    camera_name="Right IR",
                    image_topic="/zcalib_test/cam1/image",
                    info_topic="/zcalib_test/cam1/info",
                    timestamp_reference="start",
                    exposure_s=0.002,
                ),
                dict(
                    camera_name="RGB",
                    image_topic="/zcalib_test/cam2/image",
                    info_topic="/zcalib_test/cam2/info",
                    readout_s=0.007,
                ),
            ],
        )
        tick = 0
        previous_image = -1
        rng = np.random.default_rng(8)
        pattern = rng.integers(20, 235, size=(480, 640), dtype=np.uint8)

        def publish():
            nonlocal tick, previous_image
            stamp = 10000.0 + tick * 0.005
            imu = Imu()
            imu.header.stamp.sec = int(stamp)
            imu.header.stamp.nanosec = int((stamp % 1) * 1.0e9)
            imu.angular_velocity.x = 0.2 * np.sin(stamp)
            imu.angular_velocity.y = 0.1 * np.cos(stamp * 0.7)
            imu.linear_acceleration.z = 9.81
            imu.linear_acceleration.x = 0.15 * np.sin(stamp)
            imu_pub.publish(imu)
            frame = tick // 10
            if frame != previous_image:
                previous_image = frame
                for camera in range(3):
                    info = CameraInfo()
                    image_stamp = stamp + camera * 0.002
                    info.header.stamp.sec = int(image_stamp)
                    info.header.stamp.nanosec = round((image_stamp % 1) * 1e9)
                    info.header.frame_id = f"cam{camera}_optical"
                    info.width, info.height = 640, 480
                    info.k = [
                        400.0 + camera * 5,
                        0.0,
                        320.0,
                        0.0,
                        400.0,
                        240.0,
                        0.0,
                        0.0,
                        1.0,
                    ]
                    info.d = [0.0] * 5
                    info.distortion_model = "plumb_bob"
                    info_pubs[camera].publish(info)
                    image = Image()
                    image.header = info.header
                    image.width, image.height = 640, 480
                    image.encoding = "mono8" if camera < 2 else "rgb8"
                    image.step = 640 if camera < 2 else 1920
                    pixels = np.roll(pattern, (frame + camera * 7) % 80, axis=1)
                    if camera == 2:
                        pixels = np.repeat(pixels[:, :, None], 3, axis=2)
                    image.data = pixels.tobytes()
                    image_pubs[camera].publish(image)
            tick += 1

        def pump(seconds, publishing=True):
            until = time.monotonic() + seconds
            while time.monotonic() < until:
                if publishing:
                    publish()
                rclpy.spin_once(node, timeout_sec=0)
                app.processEvents()
                time.sleep(0.005)

        try:
            discovery = TopicDiscovery("ROS 2")
            discovered = []
            problems = []
            discovery.found.connect(discovered.append)
            discovery.problem.connect(problems.append)
            discovery.start()
            pump(5.3)
            discovery.wait(2000)
            app.processEvents()
            assert discovered and "/zcalib_test/cam2/image" in discovered[0], (
                f"topic discovery: {discovered} {problems}"
            )
            window.start_source(settings)
            pump(2.0)
            snapshot = window.source.snapshot()
            assert (
                all(
                    camera["image"] is not None and camera["info"]
                    for camera in snapshot["cameras"]
                )
                and len(snapshot["imu"]) > 50
            ), "viewer topic reception"
            record = root / "capture with spaces.bin"
            window.start_capture(str(record))
            pump(5.0)
            assert window.capture_active, window.console.toPlainText()
            snapshot = window.source.snapshot()
            assert all(
                camera["frame"] is not None and len(camera["frame"].points) > 30
                for camera in snapshot["cameras"]
            ), (
                f"default tracker overlays: {None if snapshot['frame'] is None else len(snapshot['frame'].points)}\n{window.console.toPlainText()}"
            )
            window.grab().save("/tmp/ov_zcalib-live.png")
            window.stop_capture()
            deadline = time.monotonic() + 8
            while (
                window.capture_active or window.loader is not None
            ) and time.monotonic() < deadline:
                pump(0.05, False)
            assert not window.capture_active and window.loader is None, (
                window.console.toPlainText()
            )
            session = read_record(record)
            assert (
                session.complete and len(session.frames) > 30 and len(session.imu) > 300
            ), "completed record footer"
            assert len(session.cameras) == 3 and {
                frame.cam for frame in session.frames
            } == {0, 1, 2}, "all three cameras recorded"
            assert (
                session.cameras[2].rolling
                and abs(session.cameras[2].tr - 0.007) < 1e-12
            ), "per-camera hardware readout preserved"
            feature_ids = [
                {
                    int(point["id"])
                    for frame in session.frames
                    if frame.cam == camera
                    for point in frame.points
                }
                for camera in range(3)
            ]
            assert all(
                not feature_ids[i].intersection(feature_ids[j])
                for i in range(3)
                for j in range(i)
            ), "independent cameras have globally unique track IDs"
            a, b = [
                next(frame for frame in session.frames if frame.cam == camera)
                for camera in (0, 1)
            ]
            assert abs((b.timestamp - a.timestamp) % 0.05 - 0.003) < 1e-7, (
                "start stamp converted once per camera"
            )
            assert window.session.path == str(record), "GUI automatic replay"
            assert Path(str(record) + ".capture.json").exists(), "capture provenance"
            assert not json.loads(Path(str(record) + ".capture.json").read_text())[
                "temperature_available"
            ]
            assert np.all(session.imu[:, 7] == 0), (
                "legacy unavailable-temperature sentinel"
            )
            window.binary.setText(find_executable("ov_zcalibrate"))
            window.output_dir.setText(str(root / "results"))
            window.budget.setValue(0.1)
            window.start_calibration()
            deadline = time.monotonic() + 20
            while window.job_active and time.monotonic() < deadline:
                pump(0.05, False)
            assert not window.job_active and window.report is not None, (
                window.console.toPlainText()
            )
            assert window.report["verdict"]["state"] == "ABORT", (
                "inconsistent synthetic motion must not certify"
            )
            assert window.report["session"]["temperature_available"] is False
            assert window.report["session"]["thermal_checks_evaluated"] is False
            assert (
                window.pages.currentIndex() == PAGE_RESULTS
                and window.result_summary.text().startswith("ABORT")
            ), "honest report UI"
            window.grab().save("/tmp/ov_zcalib-results.png")
            print(
                json.dumps(
                    {
                        "frames": len(session.frames),
                        "cameras": len(session.cameras),
                        "imu": len(session.imu),
                        "features": len(session.frames[-1].points),
                        "format": session.format,
                        "complete": session.complete,
                        "verdict": window.report["verdict"]["state"],
                        "tracker": "TrackKLT",
                        "topic_discovery": True,
                        "gui_replay": True,
                    }
                )
            )
        finally:
            if window.source is not None:
                window.source.requestInterruption()
            window.close()
            for _ in range(200):
                app.processEvents()
                if not window.isVisible():
                    break
                time.sleep(0.01)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
