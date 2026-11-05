"""ROS 2 D455 ingress: preserve IR geometry and use measured exposure centers.

Raw RealSense image/CameraInfo/Metadata triplets must have their original,
individual frame stamps (enable_sync=false, or the patched driver's
preserve_frame_timestamps=true; align_depth disabled).
Missing metadata never silently falls back to a host arrival timestamp.
"""

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import argparse
import json
import math
import time


def exposure_offset_ns(metadata, scale=1):
    """Signed difference of the D400's wrapping 32-bit microsecond counters."""
    values = []
    for key in ("sensor_timestamp", "hw_timestamp"):
        value = metadata[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value != int(value)
        ):
            raise ValueError(f"Invalid {key} in RealSense metadata")
        # The SDK subtracts exposure/2 from an unsigned UVC counter and may
        # expose a negative SENSOR_TIMESTAMP close to the 32-bit wrap.
        if not (-(2**31) if key == "sensor_timestamp" else 0) <= value < 2**32:
            raise ValueError(f"{key} exceeds the D400 32-bit clock range")
        values.append(int(value))
    delta = (values[0] - values[1] + 2**31) % 2**32 - 2**31
    if scale not in (1, 100):
        raise ValueError("Exposure-offset scale must be 1 or 100")
    delta *= scale
    if abs(delta) > 200_000:
        raise ValueError(
            "Exposure center is more than 200 ms from the hardware frame timestamp"
        )
    return delta * 1000


def stamp_ns(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def validated_exposure_offset_ns(metadata, scale=1):
    """Keep the D455 workaround within its measured operating conditions."""
    if scale == 100:
        if metadata.get("auto_exposure") != 0 or "actual_exposure" not in metadata:
            raise ValueError(
                "The verified 100-us RGB workaround requires manual exposure and per-frame actual_exposure"
            )
        exposure = metadata["actual_exposure"]
        if (
            isinstance(exposure, bool)
            or not isinstance(exposure, (int, float))
            or not math.isfinite(exposure)
            or exposure <= 0
        ):
            raise ValueError("Invalid RGB actual_exposure metadata")
        raw_offset = -exposure_offset_ns(metadata) / 1000
        if abs(2 * raw_offset - exposure) > 1:
            raise ValueError(
                "RGB exposure metadata does not match the verified 100-us workaround"
            )
    return exposure_offset_ns(metadata, scale)


def validate_original_stamp(header_ns, metadata):
    """Detect stock frameset stamp replacement before it can certify fake sync."""
    if metadata.get("clock_domain") == "global_time":
        value = metadata["frame_timestamp"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Invalid SDK frame timestamp")
        if abs(header_ns - round(value * 1_000_000)) > 10_000:
            raise ValueError(
                "Driver replaced the individual frame timestamp; grouped mode requires preserve_frame_timestamps=true"
            )


class Triplets:
    """Bounded, exact-stamp joins; late incomplete messages cannot reorder output."""

    def __init__(self, capacity=24):
        self.pending = OrderedDict()
        self.capacity = capacity
        self.dropped = 0
        self.last = -1

    def add(self, kind, message):
        key = stamp_ns(message)
        if key <= self.last:
            self.dropped += 1
            return []
        self.pending.setdefault(key, {})[kind] = message
        ready = []
        for stamp in sorted(self.pending):
            entry = self.pending[stamp]
            if len(entry) != 3:
                continue
            if stamp > self.last:
                ready.append(entry)
                self.last = stamp
        for stamp in list(self.pending):
            if stamp <= self.last:
                if len(self.pending[stamp]) != 3:
                    self.dropped += 1
                del self.pending[stamp]
        while len(self.pending) > self.capacity:
            self.pending.popitem(last=False)
            self.dropped += 1
        return ready


class ColorRectifier:
    def __init__(self):
        self.key = None
        self.maps = None

    def apply(self, image, info, cv_bridge):
        import cv2
        import numpy as np

        if self.key is None:
            cv2.setNumThreads(1)

        key = (
            info.width,
            info.height,
            tuple(info.k),
            tuple(info.d),
            info.distortion_model,
        )
        if key != self.key:
            k = np.asarray(info.k, dtype=float).reshape(3, 3)
            d = np.asarray(info.d, dtype=float)
            if (
                not np.isfinite(k).all()
                or not np.isfinite(d).all()
                or min(k[0, 0], k[1, 1]) <= 0
            ):
                raise ValueError("Invalid RGB camera calibration")
            size = (int(info.width), int(info.height))
            if info.distortion_model in ("plumb_bob", "rational_polynomial", ""):
                self.maps = cv2.initUndistortRectifyMap(
                    k, d, np.eye(3), k, size, cv2.CV_32FC1
                )
            elif info.distortion_model == "equidistant" and len(d) == 4:
                self.maps = cv2.fisheye.initUndistortRectifyMap(
                    k, d, np.eye(3), k, size, cv2.CV_32FC1
                )
            else:
                raise ValueError(
                    f"Unsupported RGB rectification model: {info.distortion_model}"
                )
            self.key = key
        pixels = cv_bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        pixels = cv2.remap(pixels, *self.maps, interpolation=cv2.INTER_LINEAR)
        output = cv_bridge.cv2_to_imgmsg(pixels, encoding=image.encoding)
        output.header = deepcopy(image.header)
        calibration = deepcopy(info)
        calibration.distortion_model = "plumb_bob"
        calibration.d = [0.0] * 5
        calibration.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        k = list(info.k)
        calibration.p = [
            k[0],
            k[1],
            k[2],
            0.0,
            k[3],
            k[4],
            k[5],
            0.0,
            k[6],
            k[7],
            k[8],
            0.0,
        ]
        return output, calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-prefix", default="/camera/camera")
    parser.add_argument("--output-prefix", default="/zcalib/d455")
    parser.add_argument("--no-rectify-color", action="store_true")
    parser.add_argument(
        "--best-effort-input",
        action="store_true",
        help="Use when raw Image/CameraInfo/Metadata publishers offer best-effort QoS",
    )
    parser.add_argument(
        "--color-offset-scale",
        type=int,
        choices=(1, 100),
        default=1,
        help="100 only for the verified D455 FW 5.17.3.10 UVC exposure-unit quirk; requires manual RGB exposure",
    )
    parser.add_argument(
        "--summary", type=Path, help="Write final counts and timing provenance as JSON"
    )
    args, ros_args = parser.parse_known_args(argv)
    if args.summary and args.summary.exists():
        parser.error("Summary output already exists")
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
        from rclpy.executors import ExternalShutdownException
        from realsense2_camera_msgs.msg import Metadata
        from sensor_msgs.msg import Image, CameraInfo
        from cv_bridge import CvBridge
    except ImportError as error:
        parser.error(
            f"Source ROS 2 and realsense2_camera; install python3-opencv: {error}"
        )

    rclpy.init(args=ros_args)

    class Bridge(Node):
        def __init__(self):
            super().__init__("ov_zcalib_realsense_bridge")
            self.cv = CvBridge()
            self.image_qos = QoSProfile(
                depth=10, reliability=ReliabilityPolicy.RELIABLE
            )
            self.rectifier = ColorRectifier()
            self.buffers = {name: Triplets() for name in ("infra1", "infra2", "color")}
            self.counts = {name: 0 for name in self.buffers}
            self.errors = {name: 0 for name in self.buffers}
            self.last = {}
            self.last_center = {}
            self.warning_time = {}
            self.outputs = {}
            self.subscriptions_ = []
            for name in self.buffers:
                output = args.output_prefix.rstrip("/") + "/" + name
                self.outputs[name] = (
                    self.create_publisher(Image, output + "/image", self.image_qos),
                    self.create_publisher(
                        CameraInfo, output + "/camera_info", qos_profile_sensor_data
                    ),
                    self.create_publisher(
                        Metadata, output + "/metadata", qos_profile_sensor_data
                    ),
                )
                for kind, typ, suffix in (
                    (
                        "image",
                        Image,
                        "image_raw" if name == "color" else "image_rect_raw",
                    ),
                    ("info", CameraInfo, "camera_info"),
                    ("metadata", Metadata, "metadata"),
                ):
                    topic = args.input_prefix.rstrip("/") + "/" + name + "/" + suffix
                    self.subscriptions_.append(
                        self.create_subscription(
                            typ,
                            topic,
                            lambda message, n=name, k=kind: self.receive(n, k, message),
                            self.image_qos
                            if not args.best_effort_input
                            else qos_profile_sensor_data,
                        )
                    )
            self.get_logger().info(
                "Waiting for image + CameraInfo + exposure metadata. Grouped delivery requires preserve_frame_timestamps=true."
            )

        def receive(self, name, kind, message):
            for entry in self.buffers[name].add(kind, message):
                try:
                    self.publish(name, entry)
                except (ValueError, KeyError, TypeError, RuntimeError) as error:
                    self.errors[name] += 1
                    if time.monotonic() - self.warning_time.get(name, 0) > 5:
                        self.get_logger().warning(f"{name}: {error}; frame omitted")
                        self.warning_time[name] = time.monotonic()

        def publish(self, name, entry):
            image, info = entry["image"], entry["info"]
            if image.width != info.width or image.height != info.height:
                raise ValueError("Image and CameraInfo dimensions differ")
            metadata = json.loads(entry["metadata"].json_data)
            if metadata.get("clock_domain") not in ("hardware_clock", "global_time"):
                raise ValueError("Hardware-clock metadata is required")
            validate_original_stamp(stamp_ns(image), metadata)
            scale = args.color_offset_scale if name == "color" else 1
            offset = validated_exposure_offset_ns(metadata, scale)
            center = stamp_ns(image) + offset
            if center <= self.last_center.get(name, -1):
                raise ValueError("Non-monotonic exposure center")
            if name == "color" and not args.no_rectify_color:
                image, info = self.rectifier.apply(image, info, self.cv)
            else:
                if name != "color" and any(float(value) != 0 for value in info.d):
                    raise ValueError("Expected D4-rectified IR with zero distortion")
                image, info = deepcopy(image), deepcopy(info)
            image.header.stamp.sec, image.header.stamp.nanosec = divmod(
                center, 1_000_000_000
            )
            info.header = deepcopy(image.header)
            provenance = deepcopy(entry["metadata"])
            provenance.header = deepcopy(image.header)
            metadata.update(
                original_ros_stamp_ns=stamp_ns(entry["image"]),
                exposure_center_ros_stamp_ns=center,
                center_correction_ns=offset,
                exposure_offset_scale=scale,
                exposure_duration_us=metadata["actual_exposure"] * scale
                if "actual_exposure" in metadata
                else None,
                rgb_rectified=(name == "color" and not args.no_rectify_color),
            )
            provenance.json_data = json.dumps(metadata)
            for publisher, value in zip(self.outputs[name], (image, info, provenance)):
                publisher.publish(value)
            self.counts[name] += 1
            self.last_center[name] = center
            self.last[name] = metadata
            if self.counts[name] == 1:
                self.get_logger().info(
                    f"{name}: publishing exposure centers, correction {offset / 1e6:.3f} ms"
                )

        def summary(self):
            return dict(
                schema=1,
                counts=self.counts,
                errors=self.errors,
                dropped={name: buffer.dropped for name, buffer in self.buffers.items()},
                last=self.last,
                timestamp_reference="center",
                temperature_available=False,
            )

    node = Bridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(node.summary(), indent=2) + "\n")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
