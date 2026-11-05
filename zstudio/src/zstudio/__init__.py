"""zstudio, a complete inspection tool for VINS. GPL-3.0-or-later.

Log inspection (voxl-logger, ROS 1 / ROS 2 bags), ROS-free VIO replay and
targetless camera-IMU calibration (ov_zcalib) for OpenVINS.
"""

import sys
from pathlib import Path

__version__ = "0.1.0"
APP_NAME = "zstudio"
TAGLINE = "A complete inspection tool for VINS"
AUTHOR = "Joao Leonardo Silva Cotta, PhD"
ORGANIZATION = "ModalAI"
ACKNOWLEDGEMENTS = ("Alex K.", "Donald H.", "Reginald W.", "James S.", "Kyle T.")


def _python_deps():
    """Private directories that the install step fills from requirements.txt.

    Appended (not prepended) to sys.path so the interpreter's own packages always
    win; see scripts/install_requirements.py.
    """
    package = Path(__file__).resolve().parent
    candidates = (
        package.parent / "python-deps",  # standalone: <prefix>/lib/zstudio/
        package.parents[2] / "zstudio" / "python-deps",  # catkin / ament site-packages
        package.parents[1] / ".python-deps",  # source checkout: zstudio/.python-deps
    )
    for path in candidates:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))


_python_deps()
