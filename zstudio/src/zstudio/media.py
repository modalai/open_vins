"""Sensor image decoding shared by the inspector, the studio viewer, and conversion.

numpy is the only requirement. OpenCV is used when importable (full-resolution
demosaic, area resize, JPEG/PNG decode); otherwise Qt decodes PNG/JPEG and Bayer
mosaics are demosaiced by 2x2 binning, which is adequate for viewing and tracking.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # Optional accelerator; never required.
    import cv2
except ImportError:  # pragma: no cover - exercised on hosts without OpenCV
    cv2 = None

BAYER_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")
# Top-left 2x2 CFA cell -> (row, col) of R and B; greens fill the remainder.
_BAYER_RB = {
    "RGGB": ((0, 0), (1, 1)),
    "BGGR": ((1, 1), (0, 0)),
    "GRBG": ((0, 1), (1, 0)),
    "GBRG": ((1, 0), (0, 1)),
}
# OpenCV names Bayer codes after the second row/column of the mosaic, so the
# sensor's top-left pattern maps to the "opposite" code (RGGB -> BayerBG).
_BAYER_CV_RGB = {
    "RGGB": "COLOR_BayerBG2RGB",
    "BGGR": "COLOR_BayerRG2RGB",
    "GRBG": "COLOR_BayerGB2RGB",
    "GBRG": "COLOR_BayerGR2RGB",
}
RAW_BITS = {".raw10": 10, ".raw12": 12}
_RAW10_SHIFTS = np.array([0, 2, 4, 6], dtype=np.uint8)
# Display gamma 2.2 as a lookup table over 12-bit linear intensity.
_GAMMA_LUT = (np.power(np.linspace(0.0, 1.0, 4096), 1 / 2.2) * 255 + 0.5).astype(
    np.uint8
)
_USE_UMAT = False


def use_opencl(mode="auto"):
    """Demosaic through OpenCL (OpenCV UMat) when available. Returns a backend description."""
    global _USE_UMAT
    _USE_UMAT = False
    if cv2 is None:
        return "numpy (OpenCV not installed)"
    if mode == "off" or not cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(False)
        return "CPU (OpenCL disabled)" if mode == "off" else "CPU (no OpenCL device)"
    cv2.ocl.setUseOpenCL(True)
    _USE_UMAT = True
    try:
        return f"OpenCL: {cv2.ocl.Device.getDefault().name()}"
    except Exception:
        return "OpenCL"


# Black levels in 10-bit units, only for sensors whose value has been measured.
SENSOR_BLACK_LEVEL10 = {"imx664": 50}
DEFAULT_BLACK_LEVEL10 = 50


@dataclass
class Picture:
    """A decoded sensor frame before display mapping.

    kind: mono8 | mono16 | bayer | rgb. ``bits`` is the significant bit depth of
    mono16/bayer data; ``pattern`` is the Bayer CFA for kind == "bayer".
    """

    kind: str
    pixels: np.ndarray
    bits: int = 8
    pattern: str = ""

    @property
    def width(self):
        return int(self.pixels.shape[1])

    @property
    def height(self):
        return int(self.pixels.shape[0])


def unpack_mipi10(raw, width, height, stride=None):
    """MIPI RAW10 (4 px / 5 bytes, LSBs in the 5th byte) -> uint16 (h, w)."""
    raw = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    stride = stride or len(raw) // height
    if stride * height > len(raw) or stride < (width + 3) // 4 * 5:
        raise ValueError("RAW10 buffer is smaller than its declared geometry.")
    rows = raw[: height * stride].reshape(height, stride)
    groups = (width + 3) // 4
    p = rows[:, : groups * 5].reshape(height, groups, 5)
    # One broadcast: pixel k = MSB byte k << 2 | bits 2k..2k+1 of the 5th byte.
    px = p[:, :, :4].astype(np.uint16) << 2
    px |= (p[:, :, 4:5] >> _RAW10_SHIFTS) & 0x3
    return px.reshape(height, groups * 4)[:, :width]


def unpack_mipi12(raw, width, height, stride=None):
    """MIPI RAW12 (2 px / 3 bytes, nibbles in the 3rd byte) -> uint16 (h, w)."""
    raw = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    stride = stride or len(raw) // height
    if stride * height > len(raw) or stride < (width + 1) // 2 * 3:
        raise ValueError("RAW12 buffer is smaller than its declared geometry.")
    rows = raw[: height * stride].reshape(height, stride)
    pairs = (width + 1) // 2
    p = rows[:, : pairs * 3].reshape(height, pairs, 3).astype(np.uint16)
    px = np.stack(
        [
            (p[:, :, 0] << 4) | (p[:, :, 2] & 0x0F),
            (p[:, :, 1] << 4) | (p[:, :, 2] >> 4),
        ],
        axis=2,
    )
    return px.reshape(height, pairs * 2)[:, :width]


def _qimage_to_array(image):
    from PyQt5.QtGui import QImage

    if image.isNull():
        raise ValueError("image could not be decoded")
    if image.format() == QImage.Format_Grayscale16:
        fmt, channels, dtype = QImage.Format_Grayscale16, 1, np.uint16
    elif image.isGrayscale():
        fmt, channels, dtype = QImage.Format_Grayscale8, 1, np.uint8
    else:
        fmt, channels, dtype = QImage.Format_RGB888, 3, np.uint8
    image = image.convertToFormat(fmt)
    width, height, line = image.width(), image.height(), image.bytesPerLine()
    pointer = image.constBits()
    pointer.setsize(line * height)
    rows = np.frombuffer(pointer, dtype=np.uint8).reshape(height, line)
    item = np.dtype(dtype).itemsize
    pixels = rows[:, : width * channels * item].copy().view(dtype)
    return (
        pixels.reshape(height, width, channels)
        if channels == 3
        else pixels.reshape(height, width)
    )


def decode_bytes(blob):
    """Decode an encoded PNG/JPEG buffer to a Picture (8- or 16-bit)."""
    if cv2 is not None:
        image = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("image could not be decoded")
        return _from_cv(image)
    from PyQt5.QtGui import QImage

    return _from_array(_qimage_to_array(QImage.fromData(bytes(blob))))


def _from_cv(image):
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB
        image = cv2.cvtColor(image, code)
    return _from_array(image)


def _from_array(pixels):
    if pixels.ndim == 3:
        rgb = pixels[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = (rgb >> 8).astype(np.uint8)
        return Picture("rgb", np.ascontiguousarray(rgb))
    if pixels.dtype == np.uint16:
        return Picture("mono16", pixels, 16)
    return Picture("mono8", pixels.astype(np.uint8, copy=False))


def msb_mosaic(raw, width, height, bits, half=False):
    """8 most significant bits of a MIPI RAW10/RAW12 mosaic, without unpacking (preview).

    ``half`` keeps every other 2x2 CFA cell (rows and columns), giving a
    half-size mosaic with the same Bayer pattern from a quarter of the data.
    """
    raw = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    stride = len(raw) // height
    group, keep = (5, 4) if bits == 10 else (3, 2)
    groups = (width + keep - 1) // keep
    if stride < groups * group:
        raise ValueError("RAW buffer is smaller than its declared geometry.")
    rows = raw[: height * stride].reshape(height, stride)[:, : groups * group]
    cells = rows.reshape(height, groups, group)
    if half and bits == 10 and height % 4 == 0:
        # Columns 4g+0, 4g+1 are one 2x2-cell column of every 4-pixel group; rows 4r+0, 4r+1 likewise.
        quarter = cells.reshape(height // 4, 4, groups, group)[:, :2, :, :2]
        return np.ascontiguousarray(
            quarter.reshape(height // 2, groups * 2)[:, : width // 2]
        )
    msb = cells[:, :, :keep].reshape(height, groups * keep)
    return np.ascontiguousarray(msb[:, :width])


def decode_file(path, width=0, height=0, pattern="RGGB", fmt=None, preview_width=0):
    """Decode a logged frame file. Raw Bayer files need the CSV width/height.

    ``preview_width`` (display only) at most half the sensor width decodes
    RAW10/12 from the 8 most significant bits: no unpacking, and the >=2x
    downscale that follows restores the dynamic range a viewer needs.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in RAW_BITS and preview_width and width and preview_width * 2 <= width:
        data = np.fromfile(path, dtype=np.uint8)
        half = preview_width * 4 <= width
        return Picture(
            "bayer", msb_mosaic(data, width, height, RAW_BITS[ext], half), 8, pattern
        )
    if ext in RAW_BITS:
        bits = RAW_BITS[ext]
        data = np.fromfile(path, dtype=np.uint8)
        if not width or not height:
            raise ValueError(f"{path.name}: raw frames need width and height")
        unpack = unpack_mipi10 if bits == 10 else unpack_mipi12
        return Picture("bayer", unpack(data, width, height), bits, pattern)
    if ext == ".raw8":
        data = np.fromfile(path, dtype=np.uint8)
        if not width or not height or len(data) < width * height:
            raise ValueError(f"{path.name}: raw8 size does not match {width}x{height}")
        return Picture("mono8", data[: width * height].reshape(height, width))
    if ext in (".nv12", ".nv21", ".yuv"):
        data = np.fromfile(path, dtype=np.uint8)
        if not width or not height or len(data) < width * height:
            raise ValueError(f"{path.name}: YUV size does not match {width}x{height}")
        # The luma plane is the leading width*height bytes for NV12/NV21/I420.
        return Picture("mono8", data[: width * height].reshape(height, width))
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"cannot decode {path.name}")
        return _from_cv(image)
    from PyQt5.QtGui import QImage

    return _from_array(_qimage_to_array(QImage(str(path))))


def demosaic(bayer, pattern, full=True, dtype=np.float32):
    """Bayer mosaic -> RGB in sensor units (``dtype``; None keeps the input type).

    Full resolution with OpenCV; 2x2 binning (half resolution) without it.
    """
    pattern = pattern.upper()
    if pattern not in BAYER_PATTERNS:
        raise ValueError(f"unknown Bayer pattern {pattern}")
    if cv2 is not None and full:
        code = getattr(cv2, _BAYER_CV_RGB[pattern])
        if _USE_UMAT:
            rgb = cv2.cvtColor(cv2.UMat(bayer), code).get()
        else:
            rgb = cv2.cvtColor(bayer, code)
        return rgb if dtype is None else rgb.astype(dtype)
    h, w = bayer.shape[0] // 2 * 2, bayer.shape[1] // 2 * 2
    cells = bayer[:h, :w].astype(np.float32)
    (rr, rc), (br, bc) = _BAYER_RB[pattern]
    red = cells[rr::2, rc::2]
    blue = cells[br::2, bc::2]
    green = (cells[rr::2, 1 - rc :: 2] + cells[1 - rr :: 2, rc::2]) * 0.5
    rgb = np.dstack((red, green, blue))
    return rgb.astype(bayer.dtype if dtype is None else dtype)


def resize_to_width(pixels, width):
    """Downscale (never upscale) to at most ``width`` columns, keeping aspect."""
    if not width or pixels.shape[1] <= width:
        return pixels
    scale = width / pixels.shape[1]
    size = (
        max(2, int(pixels.shape[1] * scale) // 2 * 2),
        max(2, int(pixels.shape[0] * scale) // 2 * 2),
    )
    if cv2 is not None:
        # Exact 2x halvings take OpenCV's fast path; one area resize finishes.
        while pixels.shape[1] >= 2 * size[0] and pixels.shape[0] >= 2 * size[1]:
            pixels = cv2.resize(
                pixels,
                (pixels.shape[1] // 2, pixels.shape[0] // 2),
                interpolation=cv2.INTER_AREA,
            )
        if pixels.shape[1] == size[0]:
            return pixels
        return cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
    factor = int(np.ceil(pixels.shape[1] / width))
    h, w = pixels.shape[0] // factor * factor, pixels.shape[1] // factor * factor
    shape = (h // factor, factor, w // factor, factor) + pixels.shape[2:]
    return pixels[:h, :w].reshape(shape).mean(axis=(1, 3)).astype(pixels.dtype)


class Tonemapper:
    """Maps decoded frames to 8-bit display/video images (RGB or gray).

    bayer : black level -> demosaic -> resize -> gray-world AWB (fixed from the
            first frame) -> auto exposure (EMA when ``smooth``) -> gamma 2.2
    mono16: percentile AGC (EMA when ``smooth``), e.g. thermal / HDR
    mono8 / rgb: resized only
    """

    def __init__(self, black_level10=DEFAULT_BLACK_LEVEL10, out_width=0, smooth=True):
        self.black10 = int(black_level10)
        self.out_width = int(out_width)
        self.smooth = smooth
        self.white = None
        self.exposure = None
        self.agc = None

    def __call__(self, picture):
        kind, pixels = picture.kind, picture.pixels
        if kind == "bayer":
            black = int(round(self.black10 * 2.0 ** (picture.bits - 10)))
            peak = max(1.0, float((1 << picture.bits) - 1 - black))
            # Saturating black subtraction, demosaic and resize stay in integers;
            # only the display-size image becomes float, and white balance,
            # exposure and the 12-bit gamma index fold into one gain.
            dtype = np.uint8 if pixels.dtype == np.uint8 else np.uint16
            if cv2 is not None:
                mosaic = cv2.subtract(np.ascontiguousarray(pixels, dtype=dtype), black)
            else:
                mosaic = np.maximum(pixels, black).astype(dtype, copy=False) - dtype(
                    black
                )
            rgb = resize_to_width(
                demosaic(mosaic, picture.pattern or "RGGB", dtype=None), self.out_width
            ).astype(np.float32)
            sample = rgb[::4, ::4].reshape(-1, 3)
            if self.white is None:
                mean = sample.mean(axis=0) + 1e-6
                self.white = (mean[1] / mean).astype(np.float32)
            level = float((sample * self.white).mean()) / peak
            target = float(np.clip(0.18 / max(level, 1e-4), 0.5, 64.0))
            if self.smooth and self.exposure is not None:
                target = 0.85 * self.exposure + 0.15 * target
            self.exposure = target
            rgb *= (self.white * (target * 4095.0 / peak)).astype(np.float32)
            np.clip(rgb, 0.0, 4095.0, out=rgb)
            return _GAMMA_LUT[rgb.astype(np.uint16)]
        if kind == "mono16":
            small = resize_to_width(pixels, self.out_width)
            lo, hi = np.percentile(small[::2, ::2], (1, 99))
            if self.smooth and self.agc is not None:
                lo, hi = 0.8 * self.agc[0] + 0.2 * lo, 0.8 * self.agc[1] + 0.2 * hi
            self.agc = (lo, hi)
            scaled = (small.astype(np.float32) - lo) / max(hi - lo, 1.0)
            return (np.clip(scaled, 0, 1) * 255 + 0.5).astype(np.uint8)
        return np.ascontiguousarray(resize_to_width(pixels, self.out_width))


def to_gray8(picture, black_level10=DEFAULT_BLACK_LEVEL10):
    """Full-resolution 8-bit luminance for feature tracking (stateless).

    Bayer frames keep their native resolution (OpenCV demosaic) so pixel
    coordinates match the camera calibration; without OpenCV they are rejected
    rather than silently halved.
    """
    kind, pixels = picture.kind, picture.pixels
    if kind == "mono8":
        return np.ascontiguousarray(pixels)
    if kind == "rgb":
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return (pixels.astype(np.float32) @ weights + 0.5).astype(np.uint8)
    if kind == "mono16":
        lo, hi = np.percentile(pixels[::4, ::4], (0.5, 99.5))
        scaled = (pixels.astype(np.float32) - lo) / max(float(hi - lo), 1.0)
        return (np.clip(scaled, 0, 1) * 255 + 0.5).astype(np.uint8)
    if kind == "bayer":
        if cv2 is None:
            raise RuntimeError(
                "Tracking Bayer frames at full resolution requires OpenCV (python3-opencv)."
            )
        black = black_level10 << max(0, picture.bits - 10)
        mosaic = np.clip(pixels.astype(np.int32) - black, 0, None).astype(np.uint16)
        rgb = demosaic(mosaic, picture.pattern or "RGGB")
        gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        peak = max(1.0, float(np.percentile(gray[::4, ::4], 99.5)))
        return (np.clip(gray / peak, 0, 1) * 255 + 0.5).astype(np.uint8)
    raise ValueError(f"unsupported picture kind {kind}")


def _quarter_turns(degrees_cw):
    degrees = int(degrees_cw) % 360
    if degrees % 90:
        raise ValueError(
            f"image rotation must be a multiple of 90 degrees, got {degrees_cw}"
        )
    return degrees // 90


def rotate_pixels(pixels, degrees_cw):
    """Rotate an image array clockwise by 0/90/180/270 degrees."""
    turns = _quarter_turns(degrees_cw)
    return np.ascontiguousarray(np.rot90(pixels, -turns)) if turns else pixels


def rotated_pattern(pattern, width, height, degrees_cw):
    """Bayer CFA of a mosaic after a clockwise rotation (depends on the image size parity)."""
    turns = _quarter_turns(degrees_cw)
    pattern = (pattern or "RGGB").upper()

    def letter(r, c):
        return pattern[(r % 2) * 2 + (c % 2)]

    if turns == 0:
        return pattern
    if turns == 1:  # new[i, j] = old[H - 1 - j, i]
        source = lambda i, j: (height - 1 - j, i)  # noqa: E731
    elif turns == 2:  # new[i, j] = old[H - 1 - i, W - 1 - j]
        source = lambda i, j: (height - 1 - i, width - 1 - j)  # noqa: E731
    else:  # new[i, j] = old[j, W - 1 - i]
        source = lambda i, j: (j, width - 1 - i)  # noqa: E731
    return "".join(letter(*source(i, j)) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))


def rotate_picture(picture, degrees_cw):
    """Rotate a decoded frame; Bayer mosaics keep a correct CFA pattern."""
    turns = _quarter_turns(degrees_cw)
    if not turns:
        return picture
    pattern = picture.pattern
    if picture.kind == "bayer":
        pattern = rotated_pattern(pattern, picture.width, picture.height, degrees_cw)
    return Picture(
        picture.kind, rotate_pixels(picture.pixels, degrees_cw), picture.bits, pattern
    )


def rotate_intrinsics(intrinsics, width, height, degrees_cw):
    """Pinhole + distortion of the same camera after rotating its image clockwise.

    Returns (intrinsics, width, height). Radial terms (radtan k, equidistant k)
    are invariant to a rotation about the optical axis; radtan's tangential
    (p1, p2) rotate with it. Pixel centers follow u' = W-1-u style mappings.
    """
    turns = _quarter_turns(degrees_cw)
    if intrinsics is None or not turns:
        return intrinsics, width, height
    fx, fy, cx, cy = intrinsics["k"]
    d = list(intrinsics["d"])
    tangential = intrinsics["model"] != "equidistant" and len(d) >= 4
    p1, p2 = (d[2], d[3]) if tangential else (0.0, 0.0)
    if turns == 1:  # u' = H-1-v, v' = u  (x' = -y, y' = x)
        k, size, p = [fy, fx, height - 1 - cy, cx], (height, width), (p2, -p1)
    elif turns == 2:  # u' = W-1-u, v' = H-1-v
        k, size, p = (
            [fx, fy, width - 1 - cx, height - 1 - cy],
            (width, height),
            (-p1, -p2),
        )
    else:  # u' = v, v' = W-1-u  (x' = y, y' = -x)
        k, size, p = [fy, fx, cy, width - 1 - cx], (height, width), (-p2, p1)
    if tangential:
        d[2], d[3] = p
    out = dict(intrinsics, k=[float(v) for v in k], d=[float(v) for v in d])
    if out.get("width") and out.get("height"):
        out["width"], out["height"] = size
    out["rotation"] = (intrinsics.get("rotation", 0) + 90 * turns) % 360
    return out, size[0], size[1]


def array_to_qimage(pixels, display=False):
    """uint8 gray (h, w) or RGB (h, w, 3) -> detached QImage.

    ``display=True`` returns Format_RGB32, which QPainter draws without a
    per-paint conversion (convert on a worker thread, not the GUI thread).
    """
    from PyQt5.QtGui import QImage

    pixels = np.ascontiguousarray(pixels)
    height, width = pixels.shape[:2]
    if pixels.ndim == 2:
        image = QImage(pixels.data, width, height, width, QImage.Format_Grayscale8)
    else:
        image = QImage(pixels.data, width, height, width * 3, QImage.Format_RGB888)
    return image.convertToFormat(QImage.Format_RGB32) if display else image.copy()


def ros_image_to_picture(message):
    """sensor_msgs/Image (any ROS distro, rosbags or rclpy) -> Picture."""
    enc = str(message.encoding).lower()
    width, height, step = int(message.width), int(message.height), int(message.step)
    data = np.frombuffer(
        bytes(message.data)
        if not isinstance(message.data, np.ndarray)
        else message.data,
        dtype=np.uint8,
    )
    if (
        width <= 0
        or height <= 0
        or width * height > 40_000_000
        or step * height > len(data)
    ):
        raise ValueError("Invalid ROS image dimensions or row stride.")
    rows = data[: step * height].reshape(height, step)
    big = bool(message.is_bigendian)
    if enc in ("mono8", "8uc1"):
        return Picture("mono8", rows[:, :width].copy())
    if enc in ("mono16", "16uc1", "16sc1"):
        pixels = (
            rows[:, : width * 2]
            .copy()
            .view(">u2" if big else "<u2")
            .reshape(height, width)
        )
        return Picture("mono16", pixels.astype(np.uint16), 16)
    if enc == "32fc1":
        pixels = (
            rows[:, : width * 4]
            .copy()
            .view(">f4" if big else "<f4")
            .reshape(height, width)
        )
        finite = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
        top = max(float(np.percentile(finite, 99.5)), 1e-6)
        return Picture(
            "mono16", (np.clip(finite / top, 0, 1) * 65535).astype(np.uint16), 16
        )
    if enc in ("rgb8", "bgr8", "rgba8", "bgra8", "8uc3"):
        channels = 4 if enc.endswith("a8") else 3
        pixels = rows[:, : width * channels].reshape(height, width, channels)
        order = [2, 1, 0] if enc.startswith("bgr") or enc == "8uc3" else [0, 1, 2]
        return Picture("rgb", np.ascontiguousarray(pixels[:, :, order]))
    if enc.startswith("bayer_"):
        pattern = enc[6:10].upper()
        if enc.endswith("16"):
            pixels = (
                rows[:, : width * 2]
                .copy()
                .view(">u2" if big else "<u2")
                .reshape(height, width)
            )
            return Picture("bayer", pixels.astype(np.uint16), 16, pattern)
        return Picture("bayer", rows[:, :width].astype(np.uint16), 8, pattern)
    if enc in ("yuv422", "uyvy", "yuv422_yuy2", "yuyv"):
        pairs = rows[:, : width * 2].reshape(height, width, 2)
        luma = pairs[:, :, 1] if enc in ("yuv422", "uyvy") else pairs[:, :, 0]
        return Picture("mono8", np.ascontiguousarray(luma))
    raise ValueError(f"Unsupported image encoding {message.encoding}.")


def black_level_for(name, sensor, spec=""):
    """Per-channel override ('50' or 'hires_bayer=50,...'), then sensor table, then default."""
    default, per = None, {}
    for part in (spec or "").split(","):
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            per[key.strip()] = int(value)
        elif part:
            default = int(part)
    if name in per:
        return per[name]
    if default is not None:
        return default
    sensor = (sensor or "").lower()
    for key, value in SENSOR_BLACK_LEVEL10.items():
        if sensor.startswith(key):
            return value
    return DEFAULT_BLACK_LEVEL10


def per_channel(spec, name, default):
    """Parse 'VALUE' or 'name=VALUE,other=VALUE' channel specs."""
    fallback, per = default, {}
    for part in (spec or "").split(","):
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            per[key.strip()] = value.strip()
        elif part:
            fallback = part
    return per.get(name, fallback)
