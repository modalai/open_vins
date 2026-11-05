"""Exercise real desktop signal shutdown without a camera or ROS installation."""

import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest


class DesktopShutdown(unittest.TestCase):
    def test_signals_close_rendered_desktop_without_interrupting_paint(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum), TemporaryDirectory() as temporary:
                root = Path(temporary)
                preview = root / "ready.png"
                environment = os.environ.copy()
                source = Path(__file__).parents[1] / "src"
                environment.update(
                    QT_QPA_PLATFORM="offscreen",
                    XDG_CONFIG_HOME=str(root / "config"),
                    XDG_CACHE_HOME=str(root / "cache"),
                    PYTHONPATH=str(source)
                    + os.pathsep
                    + environment.get("PYTHONPATH", ""),
                )
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "zstudio.app",
                        "--screenshot",
                        str(preview),
                        "--quit-after",
                        "15",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=environment,
                )
                try:
                    deadline = time.monotonic() + 8
                    while (
                        not preview.exists()
                        and child.poll() is None
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.05)
                    self.assertTrue(
                        preview.exists(), "Desktop did not render before shutdown test"
                    )
                    child.send_signal(signum)
                    stdout, stderr = child.communicate(timeout=4)
                    self.assertEqual(child.returncode, 0, stdout + stderr)
                    self.assertNotIn("KeyboardInterrupt", stderr)
                    self.assertNotIn("QPainter::", stderr)
                    self.assertNotIn("active painter", stderr)
                    self.assertNotIn("Traceback", stderr)
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.communicate()


if __name__ == "__main__":
    unittest.main()
