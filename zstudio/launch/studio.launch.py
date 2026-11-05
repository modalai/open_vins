from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "session",
                default_value="",
                description="Optional session.bin or report.json",
            ),
            Node(
                package="zstudio",
                executable="zstudio",
                name="zstudio",
                arguments=["--open", LaunchConfiguration("session")],
                output="screen",
            ),
        ]
    )
