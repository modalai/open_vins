"""Shared D455 launch profiles; ROS imports stay out of standalone GUI startup."""

import os
from pathlib import Path
import re


def cpu_prefix(value, allowed=None):
    """Optional affinity for our processes, with no shell evaluation."""
    if not value:
        return None
    if not re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", value):
        raise ValueError("CPU lists must contain numbers or ranges, such as 2,3 or 2-3")
    selected = set()
    for item in value.split(","):
        limits = list(map(int, item.split("-")))
        first, last = limits[0], limits[-1]
        if last < first or last - first > 4096:
            raise ValueError("Invalid CPU range")
        selected.update(range(first, last + 1))
    if not selected <= set(os.sched_getaffinity(0) if allowed is None else allowed):
        raise ValueError("CPU affinity includes unavailable CPUs")
    return f"taskset --cpu-list {value}"


def description(combined=False):
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, OpaqueFunction
    from launch.conditions import IfCondition
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node

    def nodes(context):
        def value(name):
            return LaunchConfiguration(name).perform(context)

        profile = (
            Path(get_package_share_directory("zstudio")) / "config" / "d455.json"
        )
        parameters = {
            "initial_reset": value("initial_reset") == "true",
            "device_type": "d455",
            "enable_depth": True,
            "enable_infra1": True,
            "enable_infra2": True,
            "enable_color": True,
            "enable_gyro": True,
            "enable_accel": True,
            "gyro_fps": int(value("gyro_fps")),
            "accel_fps": int(value("accel_fps")),
            "unite_imu_method": 2,
            "enable_sync": combined,
            "hold_back_imu_for_frames": False,
            "align_depth.enable": False,
            "depth_module.depth_profile": "640,480,30",
            "depth_module.infra_profile": "640,480,30",
            "rgb_camera.color_profile": "640,480,30",
            "depth_module.global_time_enabled": True,
            "rgb_camera.global_time_enabled": True,
            "motion_module.global_time_enabled": True,
            "depth_module.enable_auto_exposure": False,
            "depth_module.exposure": 10000,
            "rgb_camera.enable_auto_exposure": False,
            "rgb_camera.exposure": 100,
            "rgb_camera.auto_exposure_priority": False,
            "depth_module.inter_cam_sync_mode": int(value("hw_sync_mode")),
        }
        if combined:
            parameters["preserve_frame_timestamps"] = True
        return [
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                namespace="camera",
                name="camera",
                output="screen",
                parameters=[parameters],
                prefix=cpu_prefix(value("driver_cpus")),
            ),
            Node(
                package="zstudio",
                executable="ov_zcalib_realsense_bridge",
                output="screen",
                arguments=["--color-offset-scale", "100"],
                prefix=cpu_prefix(value("bridge_cpus")),
            ),
            Node(
                package="zstudio",
                executable="zstudio",
                output="screen",
                arguments=["--connect", str(profile)],
                prefix=cpu_prefix(value("viewer_cpus")),
                condition=IfCondition(LaunchConfiguration("gui")),
            ),
        ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument(
                "gyro_fps",
                default_value="400",
                choices=["200", "400"],
                description="D455 gyro sampling rate in Hz",
            ),
            DeclareLaunchArgument(
                "accel_fps",
                default_value="400",
                choices=["100", "200", "400"],
                description="D455 accelerometer sampling rate in Hz",
            ),
            DeclareLaunchArgument(
                "initial_reset", default_value="false", choices=["false", "true"]
            ),
            DeclareLaunchArgument(
                "hw_sync_mode",
                default_value="1" if combined else "0",
                choices=["0", "1"],
                description="0: free-running; 1: SDK master mode, verified for combined D455 capture",
            ),
            DeclareLaunchArgument(
                "driver_cpus",
                default_value="",
                description="Optional capture-driver CPU list",
            ),
            DeclareLaunchArgument(
                "bridge_cpus",
                default_value="",
                description="Optional rectification/ingress CPU list",
            ),
            DeclareLaunchArgument(
                "viewer_cpus",
                default_value="",
                description="Optional GUI and child recorder CPU list",
            ),
            OpaqueFunction(function=nodes),
        ]
    )
