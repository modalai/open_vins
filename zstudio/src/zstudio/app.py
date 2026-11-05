"""Entry point shared by ros2 run, rosrun and direct desktop launch."""

import argparse
import json
from pathlib import Path
import signal
import sys

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication
from . import APP_NAME, AUTHOR, ORGANIZATION, TAGLINE
from .widgets import BG, PANEL, TEXT, STYLE, font, icon
from .window import StudioWindow
from .sources import SourceSettings


def install_shutdown_handlers(app, window):
    # A Python KeyboardInterrupt can otherwise unwind paintEvent while a
    # QPainter is active. Signal handlers only set a flag; Qt initiates the
    # existing asynchronous worker/process shutdown on its next timer event.
    requested = False

    def request_shutdown(signum, frame):
        nonlocal requested
        requested = True

    def deliver_shutdown():
        nonlocal requested
        if requested:
            requested = False
            window.close()

    timer = QTimer(app)
    timer.timeout.connect(deliver_shutdown)
    previous = {
        signum: signal.signal(signum, request_shutdown)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    timer.start(100)

    def cleanup():
        timer.stop()
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    return cleanup


def configure_application(app):
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("OpenVINS")
    app.setStyle("Fusion")
    app.setFont(font(10))
    app.setStyleSheet(STYLE)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.Base, QColor(PANEL))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    app.setPalette(palette)
    app.setWindowIcon(icon("calibrate", size=64))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description=f"{APP_NAME} · {TAGLINE} · ROS 2, ROS 1, and standalone",
        epilog=f"Created by {AUTHOR} at {ORGANIZATION}.",
    )
    parser.add_argument(
        "--open",
        default="",
        metavar="SESSION_OR_REPORT",
        help="Open session.bin or report.json",
    )
    parser.add_argument(
        "--screenshot", metavar="PNG", help="Save a workspace preview PNG"
    )
    parser.add_argument(
        "--connect",
        metavar="PROFILE_JSON",
        help="Connect using saved sensor topic and timing settings",
    )
    parser.add_argument(
        "--quit-after", type=float, default=0, metavar="SECONDS", help=argparse.SUPPRESS
    )
    # Launch systems append ROS remapping arguments; transport nodes have their
    # own contexts and the desktop intentionally ignores those arguments.
    args, _ = parser.parse_known_args(argv)
    settings = None
    if args.connect:
        try:
            settings = SourceSettings(**json.loads(Path(args.connect).read_text()))
            settings.camera_sources()
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication([sys.argv[0]])
    configure_application(app)
    window = StudioWindow()
    restore_signal_handlers = install_shutdown_handlers(app, window)
    window.show()
    if args.open:
        QTimer.singleShot(50, lambda: window.open_path(args.open, ask=False))
    elif settings:
        QTimer.singleShot(50, lambda: window.start_source(settings))
    if args.screenshot:

        def screenshot():
            Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(args.screenshot)

        QTimer.singleShot(1200, screenshot)
    if args.quit_after > 0:
        QTimer.singleShot(int(args.quit_after * 1000), window.close)
    try:
        return app.exec_()
    finally:
        restore_signal_handlers()


if __name__ == "__main__":
    sys.exit(main())
