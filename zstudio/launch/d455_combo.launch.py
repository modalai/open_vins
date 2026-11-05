"""Grouped D455 acquisition with preserved frame stamps and optional CPU affinity.

Requires a realsense2_camera build that supports `preserve_frame_timestamps`
(individual frame stamps while the SDK delivers grouped frames). The bridge
validates timestamp preservation and the manual RGB workaround.
"""

from zstudio.d455_launch import description


def generate_launch_description():
    return description(combined=True)
