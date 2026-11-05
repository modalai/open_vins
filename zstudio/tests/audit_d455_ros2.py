"""Measure D455 individual exposure-center timestamps from ROS 2 metadata.

Requires individual frame stamps, including with the patched combined driver.
The 100-us RGB workaround is explicit and checked
against manual exposure metadata, as in the ingress bridge. No frameset stamps
or host callback arrival times are substituted for exposure timestamps.
"""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from realsense2_camera_msgs.msg import Metadata
from zstudio.realsense_bridge import (
    validated_exposure_offset_ns,
    validate_original_stamp,
    stamp_ns,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--color-offset-scale", type=int, choices=(1, 100), default=1)
    args = parser.parse_args()
    if args.output.exists() or not 2 <= args.seconds <= 60:
        parser.error("Choose a new output path and 2-60 seconds")
    rclpy.init()
    node = Node("zcalib_d455_metadata_audit")
    frames = {name: [] for name in ("infra1", "infra2", "color")}
    subscriptions = []
    errors = []

    def receive(name, message):
        try:
            metadata = json.loads(message.json_data)
            validate_original_stamp(stamp_ns(message), metadata)
            scale = args.color_offset_scale if name == "color" else 1
            offset = validated_exposure_offset_ns(metadata, scale)
            frames[name].append(
                dict(
                    frame=metadata["frame_number"],
                    ros_stamp_ns=stamp_ns(message),
                    center_ns=stamp_ns(message) + offset,
                    offset_ns=offset,
                    actual_exposure_us=metadata.get("actual_exposure", 0) * scale,
                    hw_timestamp=metadata["hw_timestamp"],
                    sensor_timestamp=metadata["sensor_timestamp"],
                    arrival_ns=time.time_ns(),
                    arrival_monotonic_ns=time.monotonic_ns(),
                )
            )
        except (KeyError, ValueError) as error:
            errors.append(str(error))

    for name in frames:
        subscriptions.append(
            node.create_subscription(
                Metadata,
                f"/camera/camera/{name}/metadata",
                lambda message, n=name: receive(n, message),
                qos_profile_sensor_data,
            )
        )
    deadline = time.monotonic() + args.seconds + 1
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
    node.destroy_node()
    rclpy.shutdown()
    if not all(len(values) > 20 for values in frames.values()):
        raise RuntimeError(
            f"Missing camera metadata: {[len(values) for values in frames.values()]} {errors[:3]}"
        )
    start = max(values[0]["center_ns"] for values in frames.values()) + 1_000_000_000
    end = min(values[-1]["center_ns"] for values in frames.values())
    centers = {
        name: np.array(
            [
                value["center_ns"] - start
                for value in values
                if start <= value["center_ns"] <= end
            ],
            dtype=np.int64,
        )
        for name, values in frames.items()
    }
    skew = {}
    for name in ("infra2", "color"):
        left, other = centers["infra1"], centers[name]
        selected = np.searchsorted(other, left)
        distance = np.minimum(
            abs(left - other[np.clip(selected - 1, 0, len(other) - 1)]),
            abs(left - other[np.clip(selected, 0, len(other) - 1)]),
        )
        distance = distance[(left >= other[0]) & (left <= other[-1])] / 1000
        skew[name] = dict(
            samples=len(distance),
            median_us=float(np.median(distance)),
            p95_us=float(np.quantile(distance, 0.95)),
            maximum_us=float(np.max(distance)),
        )
    result = dict(
        schema=1,
        color_offset_scale=args.color_offset_scale,
        errors=errors,
        nearest_center_skew=skew,
        metadata_hz={
            name: (len(values) - 1) * 1e9 / (values[-1] - values[0])
            for name, values in centers.items()
        },
        clock_accuracy_verified=False,
        external_exposure_measurement=False,
        frames=frames,
    )
    result["delivery_age_ms"] = {}
    result["hardware_frame_period_us"] = {}
    for name, values in frames.items():
        age = (
            np.array([value["arrival_ns"] - value["center_ns"] for value in values])
            / 1e6
        )
        result["delivery_age_ms"][name] = dict(
            median=float(np.median(age)),
            p95=float(np.quantile(age, 0.95)),
            maximum=float(np.max(age)),
        )
        hw = np.array([value["hw_timestamp"] for value in values], dtype=np.int64)
        counters = np.array([value["frame"] for value in values], dtype=np.int64)
        ticks = (np.diff(hw) + 2**31) % 2**32 - 2**31
        steps = np.diff(counters)
        periods = ticks[steps > 0] / steps[steps > 0]
        result["hardware_frame_period_us"][name] = dict(
            median=float(np.median(periods)), p95=float(np.quantile(periods, 0.95))
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in result.items() if key != "frames"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
