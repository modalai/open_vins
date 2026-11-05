"""Optional ROS transports. The desktop and record viewer never import ROS at startup."""

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
import os
import time

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from .model import Camera, Frame, POINT_DTYPE


@dataclass
class SourceSettings:
    transport: str = "ROS 2"
    image_topic: str = "/camera/image_raw"
    info_topic: str = "/camera/camera_info"
    imu_topic: str = "/imu/data_raw"
    tracks_topic: str = "/ov_zcalib/tracks"
    readout_s: float = 0.0
    exposure_s: float = 0.0
    timestamp_reference: str = "center"
    max_features: int = 200
    sigma_w: float = 1.6968e-4
    sigma_wb: float = 1.9393e-5
    sigma_a: float = 2.0e-3
    sigma_ab: float = 3.0e-3
    camera_name: str = "Camera 00"
    extra_cameras: list = field(default_factory=list)
    reliable_images: bool = False

    def camera_sources(self):
        keys = (
            "camera_name",
            "image_topic",
            "info_topic",
            "readout_s",
            "exposure_s",
            "timestamp_reference",
        )
        primary = {key: getattr(self, key) for key in keys}
        cameras = [primary]
        for index, extra in enumerate(self.extra_cameras, 1):
            if not isinstance(extra, dict) or set(extra) - set(keys):
                raise ValueError("Additional cameras contain unsupported settings.")
            cameras.append({**primary, "camera_name": f"Camera {index:02d}", **extra})
        if len(cameras) > 16:
            raise ValueError("A recording supports at most 16 cameras.")
        if len({cam["image_topic"] for cam in cameras}) != len(cameras):
            raise ValueError("Choose a different image topic for each camera.")
        for cam in cameras:
            if not cam["image_topic"] or not cam["info_topic"]:
                raise ValueError("Every camera needs an image and CameraInfo topic.")
            if cam["timestamp_reference"] not in ("center", "start"):
                raise ValueError(
                    "Image timestamps must reference center or start of exposure."
                )
            if (
                not np.isfinite([cam["readout_s"], cam["exposure_s"]]).all()
                or min(cam["readout_s"], cam["exposure_s"]) < 0
            ):
                raise ValueError(
                    "Camera readout and exposure must be finite and nonnegative."
                )
        return cameras

    def capture_parameters(self, path):
        keys = (
            "image_topic",
            "info_topic",
            "imu_topic",
            "readout_s",
            "exposure_s",
            "timestamp_reference",
            "max_features",
            "sigma_w",
            "sigma_wb",
            "sigma_a",
            "sigma_ab",
            "reliable_images",
        )
        parameters = {
            "record_path": str(path),
            **{key: getattr(self, key) for key in keys},
        }
        cameras = self.camera_sources()
        if len(cameras) > 1:
            for parameter, key in (
                ("image_topics", "image_topic"),
                ("info_topics", "info_topic"),
                ("readout_times", "readout_s"),
                ("exposure_times", "exposure_s"),
                ("timestamp_references", "timestamp_reference"),
            ):
                parameters[parameter] = [cam[key] for cam in cameras]
        return parameters


def message_time(message):
    stamp = message.header.stamp
    if hasattr(stamp, "sec"):
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
    return float(stamp.secs) + float(stamp.nsecs) * 1.0e-9


def image_from_message(message):
    """Respect row stride, endian, and channel order; detach from the ROS message."""
    enc = message.encoding.lower()
    formats = {
        "mono8": (1, QImage.Format_Grayscale8),
        "8uc1": (1, QImage.Format_Grayscale8),
        "rgb8": (3, QImage.Format_RGB888),
        "bgr8": (3, QImage.Format_RGB888),
        "rgba8": (4, QImage.Format_RGBA8888),
        "bgra8": (4, QImage.Format_RGBA8888),
    }
    width, height, step = int(message.width), int(message.height), int(message.step)
    if (
        width <= 0
        or height <= 0
        or width * height > 40_000_000
        or step * height > len(message.data)
    ):
        raise ValueError("Invalid ROS image dimensions or row stride.")
    rows = np.frombuffer(message.data, dtype=np.uint8, count=step * height).reshape(
        height, step
    )
    if enc in ("mono16", "16uc1"):
        if step < width * 2:
            raise ValueError("Image row stride is too small.")
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        pixels = rows[:, : width * 2].copy().view(dtype).reshape(height, width)
        gray = (pixels >> 8).astype(np.uint8)
        return QImage(gray.data, width, height, width, QImage.Format_Grayscale8).copy()
    if enc not in formats:
        raise ValueError(
            f"Viewer does not support {message.encoding}; publish mono8, rgb8, or bgr8."
        )
    channels, fmt = formats[enc]
    if step < width * channels:
        raise ValueError("Image row stride is too small.")
    pixels = rows[:, : width * channels].reshape(height, width, channels)
    if enc == "bgr8":
        pixels = pixels[:, :, ::-1]
    elif enc == "bgra8":
        pixels = pixels[:, :, [2, 1, 0, 3]]
    pixels = pixels.copy()
    return QImage(pixels.data, width, height, width * channels, fmt).copy()


class RosSource(QThread):
    state = pyqtSignal(str)
    problem = pyqtSignal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.lock = Lock()
        self.imu = deque(maxlen=16000)
        self.camera_settings = settings.camera_sources()
        self.cameras = [
            dict(
                camera=Camera(),
                image=None,
                image_stamp=None,
                frame=None,
                info=False,
                image_times=deque(maxlen=100),
                received_at=0.0,
            )
            for _ in self.camera_settings
        ]
        self.imu_received_at = 0.0
        self._last_error = ""

    def snapshot(self, camera_index=0):
        with self.lock:
            cameras = [
                {
                    **cam,
                    "image_times": np.array(cam["image_times"]),
                    "image_age": time.monotonic() - cam["received_at"],
                }
                for cam in self.cameras
            ]
            return dict(
                **cameras[camera_index],
                cameras=cameras,
                imu=np.array(self.imu, dtype=float).reshape(-1, 8),
                imu_age=time.monotonic() - self.imu_received_at,
            )

    def _protect(self, callback):
        def wrapped(message):
            try:
                callback(message)
            except Exception as error:
                text = str(error)
                if text != self._last_error:
                    self._last_error = text
                    self.problem.emit(text)

        return wrapped

    def _image(self, message, camera_index=0):
        image = image_from_message(message)
        stamp = message_time(message)
        with self.lock:
            cam = self.cameras[camera_index]
            cam.update(
                image=image,
                image_stamp=stamp,
                received_at=time.monotonic(),
                frame_id=message.header.frame_id,
            )
            cam["image_times"].append(stamp)

    def _imu(self, message):
        w, a = message.angular_velocity, message.linear_acceleration
        sample = (message_time(message), w.x, w.y, w.z, a.x, a.y, a.z, 0.0)
        if not np.isfinite(sample).all():
            raise ValueError("Non-finite IMU message.")
        with self.lock:
            self.imu.append(sample)
            self.imu_received_at = time.monotonic()

    def _info(self, message, camera_index=0):
        k = message.k if hasattr(message, "k") else message.K
        d = message.d if hasattr(message, "d") else message.D
        cam = Camera(
            width=message.width,
            height=message.height,
            k=[k[0], k[4], k[2], k[5]],
            d=(list(d) + [0.0] * 4)[:4],
            fisheye=message.distortion_model == "equidistant",
            tr=self.camera_settings[camera_index]["readout_s"],
            rolling=self.camera_settings[camera_index]["readout_s"] > 0.0,
        )
        with self.lock:
            self.cameras[camera_index].update(camera=cam, info=True)

    def _tracks(self, message):
        points = np.zeros(len(message.points), dtype=POINT_DTYPE)
        ids = next(
            (channel.values for channel in message.channels if channel.name == "id"), []
        )
        if len(ids) != len(points):
            raise ValueError("Track cloud is missing feature IDs.")
        points["id"] = ids
        points["u"] = [p.x for p in message.points]
        points["v"] = [p.y for p in message.points]
        indices = next(
            (
                channel.values
                for channel in message.channels
                if channel.name == "camera"
            ),
            [0],
        )
        camera_index = (
            int(indices[0])
            if len(indices)
            else next(
                (
                    index
                    for index, cam in enumerate(self.cameras)
                    if cam.get("frame_id") == message.header.frame_id
                ),
                -1,
            )
        )
        if not len(points) and camera_index == -1:
            return
        if camera_index < 0 or camera_index >= len(self.cameras):
            raise ValueError("Track cloud references an unknown camera.")
        with self.lock:
            self.cameras[camera_index]["frame"] = Frame(
                message_time(message), points, cam=camera_index
            )

    def run(self):
        try:
            if self.settings.transport == "ROS 2":
                self._ros2()
            else:
                self._ros1()
        except ImportError as error:
            self.problem.emit(
                f"{self.settings.transport} is unavailable in this Python environment: {error}. "
                "Source the ROS workspace before starting the studio."
            )
        except Exception as error:
            self.problem.emit(str(error))
        finally:
            self.state.emit("Disconnected")

    def _ros2(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud

        context = Context()
        node = executor = None
        try:
            rclpy.init(
                args=[], context=context, signal_handler_options=SignalHandlerOptions.NO
            )
            node = Node(f"ov_zcalib_viewer_{os.getpid()}", context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            callbacks = [
                (Imu, self.settings.imu_topic, self._imu),
                (PointCloud, self.settings.tracks_topic, self._tracks),
            ]
            for index, cam in enumerate(self.camera_settings):
                callbacks.extend(
                    [
                        (
                            Image,
                            cam["image_topic"],
                            lambda msg, i=index: self._image(msg, i),
                        ),
                        (
                            CameraInfo,
                            cam["info_topic"],
                            lambda msg, i=index: self._info(msg, i),
                        ),
                    ]
                )
            for typ, topic, callback in callbacks:
                qos = (
                    QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
                    if typ is Image and self.settings.reliable_images
                    else qos_profile_sensor_data
                )
                node.create_subscription(typ, topic, self._protect(callback), qos)
            self.state.emit("Connected · waiting for topics")
            while not self.isInterruptionRequested() and context.ok():
                executor.spin_once(timeout_sec=0.05)
        finally:
            if executor:
                executor.shutdown(timeout_sec=1.0)
            if node:
                node.destroy_node()
            if context.ok():
                context.shutdown()

    def _ros1(self):
        import rospy
        from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud

        if not rospy.core.is_initialized():
            rospy.init_node(
                "ov_zcalib_viewer", anonymous=True, disable_signals=True, argv=[]
            )
        subscribers = []
        try:
            callbacks = [
                (Imu, self.settings.imu_topic, self._imu),
                (PointCloud, self.settings.tracks_topic, self._tracks),
            ]
            for index, cam in enumerate(self.camera_settings):
                callbacks.extend(
                    [
                        (
                            Image,
                            cam["image_topic"],
                            lambda msg, i=index: self._image(msg, i),
                        ),
                        (
                            CameraInfo,
                            cam["info_topic"],
                            lambda msg, i=index: self._info(msg, i),
                        ),
                    ]
                )
            for typ, topic, callback in callbacks:
                subscribers.append(
                    rospy.Subscriber(
                        topic,
                        typ,
                        self._protect(callback),
                        queue_size=10,
                        buff_size=32 * 1024 * 1024,
                    )
                )
            self.state.emit("Connected · waiting for topics")
            while not self.isInterruptionRequested() and not rospy.is_shutdown():
                self.msleep(50)
        finally:
            for subscriber in subscribers:
                subscriber.unregister()


class TopicDiscovery(QThread):
    found = pyqtSignal(object)
    problem = pyqtSignal(str)

    def __init__(self, transport, parent=None):
        super().__init__(parent)
        self.transport = transport

    def run(self):
        try:
            if self.transport == "ROS 2":
                import rclpy
                from rclpy.context import Context
                from rclpy.node import Node
                from rclpy.executors import SingleThreadedExecutor
                from rclpy.signals import SignalHandlerOptions

                context = Context()
                node = executor = None
                try:
                    rclpy.init(
                        args=[],
                        context=context,
                        signal_handler_options=SignalHandlerOptions.NO,
                    )
                    node = Node(f"zcalib_discovery_{os.getpid()}", context=context)
                    executor = SingleThreadedExecutor(context=context)
                    executor.add_node(node)
                    started = time.monotonic()
                    deadline = started + 5.0
                    while time.monotonic() < deadline:
                        if self.isInterruptionRequested():
                            return
                        executor.spin_once(timeout_sec=0.05)
                        topics = node.get_topic_names_and_types()
                        if time.monotonic() - started >= 1.0 and any(
                            typ.split("/")[-1] in ("Image", "CameraInfo", "Imu")
                            for _, types in topics
                            for typ in types
                        ):
                            break
                    topics = node.get_topic_names_and_types()
                finally:
                    if executor:
                        executor.shutdown(timeout_sec=1.0)
                    if node:
                        node.destroy_node()
                    if context.ok():
                        context.shutdown()
            else:
                import xmlrpc.client

                class Transport(xmlrpc.client.Transport):
                    def make_connection(self, host):
                        connection = super().make_connection(host)
                        connection.timeout = 1.0
                        return connection

                with xmlrpc.client.ServerProxy(
                    os.environ.get("ROS_MASTER_URI", "http://localhost:11311"),
                    transport=Transport(),
                ) as master:
                    code, message, topics = master.getPublishedTopics(
                        "/zstudio", ""
                    )
                    if code != 1:
                        raise RuntimeError(message)
                    topics = [(name, [typ]) for name, typ in topics]
            self.found.emit(dict(topics))
        except Exception as error:
            self.problem.emit(str(error))
