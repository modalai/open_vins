"""Verified manual-exposure D455 profile for the stock RealSense ROS driver."""

from zstudio.d455_launch import description


def generate_launch_description():
    return description(combined=False)
