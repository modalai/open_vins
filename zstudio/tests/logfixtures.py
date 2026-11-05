"""Synthetic voxl-logger logs and ROS bags for the log inspector / studio tests."""

import json
import os
import shutil
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

INTRINSICS_YML = """%YAML:1.0
---
M: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ {fx}, 0., {cx}, 0.,
       {fy}, {cy}, 0., 0., 1. ]
D: !!opencv-matrix
   rows: 4
   cols: 1
   dt: d
   data: [ 0.01, -0.002,
       0.0003, -0.0001 ]
reprojection_error: 2.5e-01
width: {w}
height: {h}
distortion_model: fisheye
calibration_time: "2026-01-01 00:00:00"
"""


def _png(path, pixels):
    from PyQt5.QtGui import QImage

    h, w = pixels.shape
    data = np.ascontiguousarray(pixels.astype(np.uint8))
    if not QImage(data.data, w, h, w, QImage.Format_Grayscale8).copy().save(str(path)):
        raise OSError(f"cannot write {path}")


def make_voxl_log(
    root,
    *,
    duration=2.0,
    imu_rate=1000.0,
    configured_rate=1000,
    cameras=(("tracking_left", "ar0144", 30.0), ("tracking_right", "ar0144", 30.0)),
    size=(64, 48),
    body_imu_rpy=(0, 180, 0),
    accel=(0.0, 0.0, 9.81),
    imu_gaps=(),
    imu_duplicates=(),
    frame_gaps=None,
    repeats=None,
    missing_files=None,
    empty_files=None,
    orphans=None,
    intrinsics=True,
    missing_dirs=(),
    header_comment=True,
    epoch_ns=5_000_000_000,
    textured=False,
    camera_names=None,
):
    """Write a voxl-logger directory; returns its Path.

    imu_gaps: sample indices removed; imu_duplicates: indices whose stamp repeats
    the previous one. frame_gaps / repeats / missing_files / empty_files /
    orphans: {camera: [rows]}. repeats re-send the previous frame (same
    frame_id, identical file, new stamp). textured=True draws a moving texture
    that feature trackers can follow.
    """
    root = Path(root)
    frame_gaps, repeats = frame_gaps or {}, repeats or {}
    missing_files, empty_files, orphans = (
        missing_files or {},
        empty_files or {},
        orphans or {},
    )
    rng = np.random.default_rng(7)
    channels = [
        {
            "channel": 0,
            "type": 1,
            "type_string": "imu",
            "pipe_path": "/run/mpa/imu_apps/",
            "n_to_skip": 0,
        }
    ]
    # IMU
    n = int(duration * imu_rate)
    t_ns = epoch_ns + 10_000_000 + (np.arange(n) * (1e9 / imu_rate)).astype(np.int64)
    keep = np.ones(n, bool)
    keep[list(imu_gaps)] = False
    for i in imu_duplicates:
        t_ns[i] = t_ns[i - 1]
    imu_dir = root / "run/mpa/imu_apps"
    imu_dir.mkdir(parents=True)
    rows = []
    for k, i in enumerate(np.where(keep)[0]):
        noise = rng.normal(0, 0.01, 6)
        rows.append(
            f"{k},{t_ns[i]},{1 + k // 30},{accel[0] + noise[0]:.6f},{accel[1] + noise[1]:.6f},{accel[2] + noise[2]:.6f},"
            f"{noise[3] * 0.1:.6f},{noise[4] * 0.1:.6f},{noise[5] * 0.1:.6f},{40 + 5 * k / n:.4f}"
        )
    (imu_dir / "data.csv").write_text(
        "i,timestamp(ns),batch_id,AX(m/s2),AY(m/s2),AZ(m/s2),GX(rad/s),GY(rad/s),GZ(rad/s),T(C)\n"
        + "\n".join(rows)
        + "\n"
    )
    channels[0]["n_samples"] = int(keep.sum())
    w, h = size
    texture = rng.integers(0, 255, (h * 3, w * 3)).astype(np.uint8)
    if textured:
        k = np.ones(5) / 5
        texture = np.apply_along_axis(
            lambda r: np.convolve(r, k, "same"), 0, texture.astype(float)
        )
        texture = np.apply_along_axis(
            lambda r: np.convolve(r, k, "same"), 1, texture
        ).astype(np.uint8)
    for c, (name, _sensor, fps) in enumerate(cameras, 1):
        directory = root / "run/mpa" / name
        channels.append(
            {
                "channel": c,
                "type": 2,
                "type_string": "cam",
                "pipe_path": f"/run/mpa/{name}/",
                "n_to_skip": 0,
                "dropped_data": False,
            }
        )
        if name in missing_dirs:
            channels[-1]["n_samples"] = 10
            continue
        directory.mkdir(parents=True)
        count = int(duration * fps)
        lines, frame_id, previous = [], 100, None
        for row in range(count):
            stamp = epoch_ns + 20_000_000 + int(row * 1e9 / fps)
            if row in frame_gaps.get(name, ()):
                frame_id += 3
            repeat = row in repeats.get(name, ()) and previous is not None
            if not repeat:
                frame_id += 1
            lines.append(f"{row},{stamp},500,{8_000_000},0,{h},{w},{frame_id},0")
            path = directory / f"{row:05d}.png"
            if row in missing_files.get(name, ()):
                continue
            if row in empty_files.get(name, ()):
                path.write_bytes(b"")
                continue
            if repeat:
                shutil.copyfile(previous, path)
            else:
                if textured:
                    pixels = texture[
                        h + int(4 * np.sin(row / 6)) : 2 * h + int(4 * np.sin(row / 6)),
                        w + row % w // 4 : 2 * w + row % w // 4,
                    ]
                else:
                    pixels = rng.integers(0, 255, (h, w))
                _png(path, pixels)
            previous = path
        for extra in orphans.get(name, ()):
            _png(directory / f"{extra:05d}.png", rng.integers(0, 255, (h, w)))
        (directory / "data.csv").write_text(
            "i,timestamp(ns),gain,exposure(ns),format,height,width,frame_id,reserved\n"
            + "\n".join(lines)
            + "\n"
        )
        channels[-1]["n_samples"] = count
    info = {
        "log_format_version": 1,
        "note": "armed",
        "n_channels": len(channels),
        "start_time_monotonic_ns": epoch_ns,
        "start_time_date": "1970-01-01 00:00:05",
        "duration_s": duration,
        "channels": channels,
    }
    (root / "info.json").write_text(json.dumps(info, indent="\t"))
    etc = root / "etc/modalai"
    etc.mkdir(parents=True)
    header = (
        "/**\n * test header, as written by voxl-configure\n */\n"
        if header_comment
        else ""
    )
    (etc / "voxl-camera-server.conf").write_text(
        header
        + json.dumps(
            {
                "cameras": [
                    {"type": sensor, "name": camera}
                    for camera, sensor in dict(
                        ((camera_names or {}).get(name, name), sensor)
                        for name, sensor, _ in cameras
                    ).items()
                ]
            },
            indent="\t",
        )
    )
    (etc / "voxl-imu-server.conf").write_text(
        header + json.dumps({"imu0_sample_rate_hz": configured_rate}, indent="\t")
    )
    (etc / "vio_cams.conf").write_text(
        json.dumps(
            {
                "cams": [
                    {
                        "enable": True,
                        "name": name,
                        "pipe_for_tracking": name + "_misp_norm_ion",
                        "imu": "imu_apps",
                        "cal_file": f"opencv_{name}_intrinsics.yml",
                    }
                    for name, sensor, _ in cameras
                    if sensor == "ar0144"
                ]
            },
            indent="\t",
        )
    )
    extrinsics = [
        {
            "parent": "body",
            "child": "imu_apps",
            "T_child_wrt_parent": [0.0, 0.0, 0.0],
            "RPY_parent_to_child": list(body_imu_rpy),
        }
    ]
    children = list(
        dict.fromkeys((camera_names or {}).get(n, n) for n, _, _ in cameras)
    )
    for k, name in enumerate(children):
        extrinsics.append(
            {
                "parent": "imu_apps",
                "child": name,
                "T_child_wrt_parent": [0.08, 0.04 - 0.08 * k, 0.02],
                "RPY_parent_to_child": [0, 90, 90],
            }
        )
    (etc / "extrinsics.conf").write_text(
        header + json.dumps({"name": "TEST", "extrinsics": extrinsics}, indent="\t")
    )
    data = root / "data/modalai"
    data.mkdir(parents=True)
    if intrinsics:
        for name, sensor, _ in cameras:
            if sensor == "ar0144":
                (data / f"opencv_{name}_intrinsics.yml").write_text(
                    INTRINSICS_YML.format(
                        fx=50.0, fy=50.0, cx=w / 2, cy=h / 2, w=w, h=h
                    )
                )
    return root


def make_ros2_bag(
    path,
    *,
    duration=1.0,
    imu_rate=200.0,
    fps=10.0,
    size=(32, 24),
    camera_info=True,
    tf_static=True,
    zero_stamps=(),
    latency_ns=2_000_000,
    storage="sqlite3",
    ros1=False,
    seq_gaps=(),
):
    """Write a small ROS 2 (or ROS 1) bag with Imu, Image, CameraInfo and /tf_static."""
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_JAZZY)
    Header = store.types["std_msgs/msg/Header"]
    Time = store.types["builtin_interfaces/msg/Time"]
    Imu = store.types["sensor_msgs/msg/Imu"]
    Image = store.types["sensor_msgs/msg/Image"]
    CameraInfo = store.types["sensor_msgs/msg/CameraInfo"]
    Vector3 = store.types["geometry_msgs/msg/Vector3"]
    Quaternion = store.types["geometry_msgs/msg/Quaternion"]
    Transform = store.types["geometry_msgs/msg/Transform"]
    TransformStamped = store.types["geometry_msgs/msg/TransformStamped"]
    TFMessage = store.types.get("tf2_msgs/msg/TFMessage")
    RegionOfInterest = store.types["sensor_msgs/msg/RegionOfInterest"]

    def header(stamp_ns, frame, seq=0):
        sec, nsec = divmod(int(stamp_ns), 1_000_000_000)
        if ros1:
            return Header(seq=seq, stamp=Time(sec=sec, nanosec=nsec), frame_id=frame)
        return Header(stamp=Time(sec=sec, nanosec=nsec), frame_id=frame)

    base = 1_700_000_000_000_000_000
    w, h = size
    rng = np.random.default_rng(3)
    messages = []
    for i in range(int(duration * imu_rate)):
        t = base + int(i * 1e9 / imu_rate)
        cov = np.zeros(9)
        msg = Imu(
            header=header(t, "imu_link", seq=i),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            orientation_covariance=np.full(9, -1.0)
            if not ros1
            else np.array([-1.0] + [0.0] * 8),
            angular_velocity=Vector3(x=0.01, y=0.0, z=0.0),
            angular_velocity_covariance=cov,
            linear_acceleration=Vector3(x=0.0, y=0.0, z=9.81),
            linear_acceleration_covariance=cov,
        )
        messages.append(("/imu", Imu.__msgtype__, t + latency_ns, msg))
    seq = 0
    for i in range(int(duration * fps)):
        t = base + 5_000_000 + int(i * 1e9 / fps)
        if i in seq_gaps:
            seq += 4
        stamp = 0 if i in zero_stamps else t
        msg = Image(
            header=header(stamp, "cam0_optical", seq=seq),
            height=h,
            width=w,
            encoding="mono8",
            is_bigendian=0,
            step=w,
            data=rng.integers(0, 255, w * h).astype(np.uint8),
        )
        seq += 1
        messages.append(("/cam0/image_raw", Image.__msgtype__, t + latency_ns, msg))
        if camera_info:
            arrays = {
                "d": np.array([0.01, -0.01, 0.0, 0.0, 0.0]),
                "k": np.array([40.0, 0, w / 2, 0, 40.0, h / 2, 0, 0, 1]),
                "r": np.eye(3).ravel(),
                "p": np.array([40.0, 0, w / 2, 0, 0, 40.0, h / 2, 0, 0, 0, 1, 0]),
            }
            if ros1:  # ROS 1 spells the arrays D, K, R, P
                arrays = {key.upper(): value for key, value in arrays.items()}
            info = CameraInfo(
                header=header(t, "cam0_optical"),
                height=h,
                width=w,
                distortion_model="plumb_bob",
                **arrays,
                binning_x=0,
                binning_y=0,
                roi=RegionOfInterest(
                    x_offset=0, y_offset=0, height=0, width=0, do_rectify=False
                ),
            )
            messages.append(
                ("/cam0/camera_info", CameraInfo.__msgtype__, t + latency_ns, info)
            )
    if tf_static and not ros1:  # rosbags' ROS 1 store has no tf2_msgs
        transforms = [
            TransformStamped(
                header=header(base, "base_link"),
                child_frame_id="imu_link",
                transform=Transform(
                    translation=Vector3(x=0.0, y=0.0, z=0.0),
                    rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            ),
            TransformStamped(
                header=header(base, "imu_link"),
                child_frame_id="cam0_optical",
                transform=Transform(
                    translation=Vector3(x=0.05, y=0.0, z=0.0),
                    rotation=Quaternion(x=-0.5, y=0.5, z=-0.5, w=0.5),
                ),
            ),
        ]
        messages.append(
            (
                "/tf_static",
                TFMessage.__msgtype__,
                base,
                TFMessage(transforms=transforms),
            )
        )
    messages.sort(key=lambda m: m[2])
    if ros1:
        from rosbags.rosbag1 import Writer

        with Writer(Path(path)) as writer:
            conns = {}
            for topic, msgtype, t, msg in messages:
                if topic not in conns:
                    conns[topic] = writer.add_connection(
                        topic, msgtype, typestore=store
                    )
                writer.write(conns[topic], t, store.serialize_ros1(msg, msgtype))
    else:
        from rosbags.rosbag2 import StoragePlugin, Writer

        plugin = StoragePlugin.MCAP if storage == "mcap" else StoragePlugin.SQLITE3
        with Writer(Path(path), version=9, storage_plugin=plugin) as writer:
            conns = {}
            for topic, msgtype, t, msg in messages:
                if topic not in conns:
                    conns[topic] = writer.add_connection(
                        topic, msgtype, typestore=store
                    )
                writer.write(conns[topic], t, store.serialize_cdr(msg, msgtype))
    return Path(path)
