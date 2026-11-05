from setuptools import find_packages, setup

# Python requirements live in requirements.txt and are resolved by the CMake
# install step (scripts/install_requirements.py), not by setup.py: catkin runs
# `setup.py install`, which must not fetch packages on its own.
setup(
    name="zstudio",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    description="zstudio, a complete inspection tool for VINS (OpenVINS log inspection, VIO replay, ov_zcalib calibration)",
    author="Joao Leonardo Silva Cotta, PhD",
    author_email="joao.cotta@modalai.com",
    license="GPL-3.0-or-later",
    entry_points={
        "console_scripts": [
            "zstudio=zstudio.app:main",
            "ov_zcalib_realsense_bridge=zstudio.realsense_bridge:main",
            "ov_log_inspector=zstudio.logcli:main",
        ]
    },
)
