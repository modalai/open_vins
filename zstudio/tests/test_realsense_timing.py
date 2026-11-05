"""Clock-wrap, message association, and multi-camera transport contracts."""

from types import SimpleNamespace
import json
import unittest

from zstudio.realsense_bridge import (
    exposure_offset_ns,
    validated_exposure_offset_ns,
    validate_original_stamp,
    Triplets,
)
from zstudio.sources import SourceSettings
from zstudio.engine import capture_command
from zstudio.d455_launch import cpu_prefix


def message(ns):
    sec, nanosec = divmod(ns, 1_000_000_000)
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))
    )


class RealSenseTiming(unittest.TestCase):
    def test_scheduler_affinity_is_optional_and_restricted_to_available_cpus(self):
        self.assertIsNone(cpu_prefix("", allowed={2, 3}))
        self.assertEqual(cpu_prefix("2-3", allowed={2, 3}), "taskset --cpu-list 2-3")
        for invalid in ("2; echo bad", "3-2", "2-4099", "0,2", "2,$(id)"):
            with self.subTest(cpus=invalid), self.assertRaises(ValueError):
                cpu_prefix(invalid, allowed={2, 3})

    def test_combined_delivery_cannot_hide_replaced_frame_timestamps(self):
        metadata = dict(clock_domain="global_time", frame_timestamp=1792025739500.0)
        original = 1792025739500_000000
        validate_original_stamp(original + 500, metadata)
        with self.assertRaisesRegex(ValueError, "replaced"):
            validate_original_stamp(original + 5_000_000, metadata)

    def test_rgb_workaround_requires_the_verified_manual_exposure_conditions(self):
        measured = dict(
            hw_timestamp=1000,
            sensor_timestamp=950,
            actual_exposure=100,
            auto_exposure=0,
        )
        self.assertEqual(validated_exposure_offset_ns(measured, 100), -5_000_000)
        for invalid in (
            {**measured, "auto_exposure": 1},
            {key: value for key, value in measured.items() if key != "actual_exposure"},
            {**measured, "actual_exposure": 200},
            {**measured, "actual_exposure": float("nan")},
        ):
            with self.subTest(metadata=invalid), self.assertRaises(ValueError):
                validated_exposure_offset_ns(invalid, 100)

    def test_exposure_offset_wraps_in_both_directions(self):
        self.assertEqual(
            exposure_offset_ns(dict(hw_timestamp=100, sensor_timestamp=2**32 - 400)),
            -500_000,
        )
        self.assertEqual(
            exposure_offset_ns(dict(hw_timestamp=2**32 - 400, sensor_timestamp=100)),
            500_000,
        )
        self.assertEqual(
            exposure_offset_ns(dict(hw_timestamp=70000, sensor_timestamp=60000)),
            -10_000_000,
        )
        self.assertEqual(
            exposure_offset_ns(dict(hw_timestamp=100, sensor_timestamp=-400)), -500_000
        )

    def test_missing_or_invalid_metadata_cannot_invent_a_timestamp(self):
        self.assertEqual(
            exposure_offset_ns(dict(hw_timestamp=1000, sensor_timestamp=950), 100),
            -5_000_000,
        )
        for data in (
            {},
            dict(hw_timestamp=1, sensor_timestamp=1.5),
            dict(hw_timestamp=1, sensor_timestamp=float("nan")),
            dict(hw_timestamp=-1, sensor_timestamp=1),
            dict(hw_timestamp=1, sensor_timestamp=500000),
        ):
            with self.subTest(data=data), self.assertRaises((ValueError, KeyError)):
                exposure_offset_ns(data)

    def test_triplets_accept_out_of_order_delivery_without_reordering_frames(self):
        buffer = Triplets(capacity=2)
        self.assertFalse(buffer.add("metadata", message(10)))
        self.assertFalse(buffer.add("image", message(10)))
        self.assertFalse(buffer.add("image", message(11)))
        self.assertEqual(len(buffer.add("info", message(10))), 1)
        self.assertFalse(buffer.add("info", message(11)))
        self.assertEqual(len(buffer.add("metadata", message(11))), 1)
        self.assertFalse(buffer.add("image", message(10)))
        for ns in (12, 13, 14, 15):
            buffer.add("image", message(ns))
        self.assertEqual(len(buffer.pending), 2)
        self.assertGreater(buffer.dropped, 0)

    def test_camera_parameters_are_yaml_arrays_and_timing_is_independent(self):
        settings = SourceSettings(
            extra_cameras=[
                dict(image_topic="/right", info_topic="/right/info", readout_s=0.007)
            ]
        )
        _, args = capture_command("/bin/true", settings, "/tmp/unused-zcalib-test.bin")
        values = dict(arg.split(":=", 1) for arg in args if ":=" in arg)
        self.assertEqual(json.loads(values["readout_times"]), [0.0, 0.007])
        self.assertEqual(
            json.loads(values["image_topics"]), ["/camera/image_raw", "/right"]
        )
        settings.extra_cameras[0]["image_topic"] = settings.image_topic
        with self.assertRaises(ValueError):
            settings.camera_sources()


if __name__ == "__main__":
    unittest.main()
