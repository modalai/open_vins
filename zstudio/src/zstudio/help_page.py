"""In-app help: the workflow and every page, opened at a section by each page's ? button."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QTextBrowser,
    QWidget,
)

from . import ACKNOWLEDGEMENTS, APP_NAME, AUTHOR, ORGANIZATION, TAGLINE, __version__
from .widgets import ACCENT, BLUE, GREEN, LINE, MUTED, PANEL, RED, AMBER, TEXT

SECTIONS = (
    ("flow", "The workflow"),
    ("open", "Opening data"),
    ("overview", "Overview"),
    ("inspect", "Inspect"),
    ("orientation", "Camera orientation"),
    ("vio", "VIO"),
    ("calibrate", "Calibrate"),
    ("results", "Results"),
    ("cli", "Command line"),
    ("shortcuts", "Keyboard shortcuts"),
    ("troubleshooting", "Troubleshooting"),
    ("about", "About zstudio"),
)

CSS = f"""
body {{ color:{TEXT}; font-size:10.5pt; }}
h1 {{ font-size:19pt; font-weight:600; margin:18px 0 6px 0; }}
h2 {{ font-size:13pt; font-weight:600; margin:16px 0 4px 0; }}
p, li {{ color:{TEXT}; margin:4px 0; }}
.muted {{ color:{MUTED}; }}
code {{ font-family:'SF Mono','JetBrains Mono','DejaVu Sans Mono'; background:#1f1f21; color:{BLUE}; }}
pre {{ font-family:'SF Mono','JetBrains Mono','DejaVu Sans Mono'; background:#1f1f21; color:{TEXT}; padding:8px; }}
table {{ border-collapse:collapse; margin:6px 0; }}
td, th {{ padding:5px 10px; border-bottom:1px solid {LINE}; vertical-align:top; }}
th {{ color:{MUTED}; text-align:left; font-weight:600; }}
.step {{ color:{ACCENT}; font-weight:600; }}
.problem {{ color:{RED}; font-weight:600; }}
.warning {{ color:{AMBER}; font-weight:600; }}
.info {{ color:{BLUE}; font-weight:600; }}
.expected {{ color:{GREEN}; font-weight:600; }}
"""

HTML = """
<p class="muted">{app} {version} · {tagline}</p>
<a name="flow"></a><h1>The workflow</h1>
<p class="muted">From a recording to a calibration you can trust, in six steps. Each page has a <b>?</b> button that opens its section here.</p>
<table>
<tr><td class="step">1&nbsp;&nbsp;Open</td><td>A voxl-logger log, a ROS 1 / ROS 2 bag, a session or a report — or connect live ROS sensors.</td></tr>
<tr><td class="step">2&nbsp;&nbsp;Inspect</td><td>Check every pipe or topic for data loss and configuration mistakes before trusting it.</td></tr>
<tr><td class="step">3&nbsp;&nbsp;Fix orientation</td><td>If a camera is upside down or its extrinsics are wrong, fix it from Inspect and decide whether to save.</td></tr>
<tr><td class="step">4&nbsp;&nbsp;VIO</td><td>Run OpenVINS on the log: a sane trajectory means the calibration and timing are usable.</td></tr>
<tr><td class="step">5&nbsp;&nbsp;Calibrate</td><td>Run the targetless engine on the cameras you choose.</td></tr>
<tr><td class="step">6&nbsp;&nbsp;Results</td><td>Read the verdict, what was committed and how precisely.</td></tr>
</table>

<a name="open"></a><h1>Opening data</h1>
<p><b>Open</b> (top right) accepts <code>.bag</code>, <code>.db3</code>, <code>.mcap</code>, a bag's <code>metadata.yaml</code>, a log's <code>info.json</code>, <code>session.bin</code> and <code>report.json</code>; <b>Log folder…</b> opens a voxl-logger directory or a ROS 2 bag folder. <b>Connect sensors…</b> subscribes to live ROS 2 / ROS 1 topics.</p>
<p>Before a log loads you choose its channels: pipes come from <code>info.json</code>, topics from the bag metadata. Everything supported starts checked; uncheck heavy image topics you don't need and large bags load faster. You can change the selection later from Inspect → <b>Channels…</b>.</p>
<p class="muted">Bags are read with the pure-Python <code>rosbags</code> package — no sourced ROS needed (<code>pip install rosbags</code>).</p>

<a name="overview"></a><h1>Overview</h1>
<p>Every camera plays at once in <b>All cameras</b>; <b>Selected camera</b> shows one camera large with its feature tracks; <b>Rig</b> draws the configured extrinsics. The IMU plots show the last 8 seconds at the playhead.</p>
<p>Use the playback bar or <b>Space</b> / <b>←</b> <b>→</b>. Images decode in the background at the size they are shown, so every camera keeps up with playback. With live sensors, <b>Record</b> tracks the cameras with TrackKLT into a session.</p>

<a name="inspect"></a><h1>Inspect</h1>
<p>The table lists each pipe/topic with its rate, count and status; the timeline shows the whole log, one lane per channel, with every finding as a mark you can click to seek. Below: plots of any channel (IMU axes, sample interval, temperature, frame interval, exposure, gain, bag latency) and a preview of the cameras at the playhead — <b>Selected</b> or <b>All</b>.</p>
<table>
<tr><th>Severity</th><th>Meaning</th></tr>
<tr><td class="problem">Problem</td><td>Data loss or corruption: timing gaps, duplicate or backwards stamps, skipped frame counters, frames re-sent with a new stamp, missing or truncated files, metadata counts that disagree, a body→IMU rotation that contradicts gravity.</td></tr>
<tr><td class="warning">Warning</td><td>Likely to hurt VIO or calibration: IMU rate differs from its configuration, large temperature drift, missing intrinsics or CameraInfo, unsynchronized stereo.</td></tr>
<tr><td class="info">Info</td><td>Measurements and context: stereo skew, gravity consistency, bag latency.</td></tr>
<tr><td class="expected">Expected</td><td>Known-benign sensor behaviour, such as thermal shutter (FFC) frames.</td></tr>
</table>
<p><b>Stereo check</b> recovers a stereo pair's relative rotation and baseline direction from the images and compares them with the configuration. <b>Export report…</b> writes a self-contained HTML report.</p>

<a name="orientation"></a><h1>Camera orientation</h1>
<p>In Inspect's preview, <b>⟲</b> and <b>⟳</b> rotate the selected camera by 90°. A rotation belongs to the <i>camera</i> (for example <code>hires</code>), so all of its pipes (<code>hires_bayer</code>, …) turn together. Frames, image size, intrinsics and the Bayer pattern rotate with it, and the camera's extrinsics turn about its optical axis so VIO and calibration stay consistent — exactly like VOXL's <code>en_rotate</code>.</p>
<p><b>Orientation…</b> edits the rotation and the extrinsics (parent frame, roll/pitch/yaw, translation) with a live preview and a gravity check that tells whether the image is upright. Changes apply immediately for the session; the bar under the preview offers <b>Save…</b> or <b>Revert</b>. Saving asks first and lists what it writes: the log's <code>extrinsics.conf</code> (a <code>.bak</code> copy is kept) and <code>ov_studio.json</code> for rotations; bags get a sidecar next to the bag. Logged frames are never modified.</p>

<a name="vio"></a><h1>VIO</h1>
<p>Check the cameras and IMU, pick a template (<code>voxl_sim</code> for VOXL logs, <code>euroc_mav</code> otherwise) and <b>Generate config</b>: intrinsics come from the log, <code>T_imu_cam</code> from its rig, plus the measured IMU rate and gates that match the IMU's mounting. Every derived value is highlighted in <b>Parameters</b>, a searchable editor for the YAML. For public datasets whose preset describes the same device, or bags without CameraInfo, use <b>Use the template's camera calibration</b>.</p>
<p><b>Run VIO</b> replays the log through <code>ov_msckf_replay</code> without ROS. The trajectory, the filter's tracks on each camera and plots of position, velocity, biases, uncertainty, online extrinsic and time-offset calibration and update time all follow the playhead, so a finished run can be scrubbed. Diverged or never-initialized runs are flagged. <b>Tracker</b> offers CPU TrackKLT (default) and, when a build with the OpenCL host harness is present, TrackOCL on the GPU.</p>

<a name="calibrate"></a><h1>Calibrate</h1>
<p><b>Data</b>: the open log's checked cameras (tracked with TrackKLT into a session first) or a session file. A single VOXL camera can also use the engine's own VOXL ingest (Advanced). The <b>Ready to calibrate</b> list shows what is missing — intrinsics, IMU, engine — and hints about motion.</p>
<p><b>Solve</b>: the profile (Library, VOXL bench, VOXL flight; Inherit for sessions), how much to trust the intrinsics (Fixed, Refine with priors, Full) and the time budget. The run panel follows the engine's stages: track frames, still baseline, bootstrap, collect motion windows, joint refinement, held-out verification, commit.</p>
<p class="muted">A good calibration recording starts still, then rotates slowly about all three axes with some translation and varied speed. A flight log often lacks that excitation; the engine then refuses to commit rather than report a guess.</p>

<a name="results"></a><h1>Results</h1>
<p>The verdict, every calibration block with its decision (committed or seed retained) and why, committed values with the rig, posterior precision relative to the prior, and the full evidence. Posterior σ is local curvature precision, not accuracy.</p>

<a name="cli"></a><h1>Command line</h1>
<pre>ov_log_inspector LOG --html            # the Inspect checks, headless (exit 1 on problems)
ov_log_inspector flight.bag --only '/cam*,/imu*' --json -
ov_zcalibrate --replay session.bin --out cal.yaml --report-json cal.report.json
ov_zcalib_track OUT.bin &lt; stream      # frames + IMU -> session (used by the studio)
ov_msckf_replay estimator_config.yaml   # OpenVINS on a sensor stream (used by VIO)</pre>

<a name="shortcuts"></a><h1>Keyboard shortcuts</h1>
<table>
<tr><td><b>Space</b></td><td>Play / pause</td></tr>
<tr><td><b>← →</b></td><td>Previous / next frame</td></tr>
<tr><td><b>Ctrl+O</b></td><td>Open</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save the current camera view (in Parameters: save the configuration)</td></tr>
<tr><td><b>Ctrl+F</b></td><td>Search parameters</td></tr>
<tr><td>Wheel / drag / double-click</td><td>Zoom / pan or orbit / reset, in images, plots, the timeline and 3D views</td></tr>
</table>

<a name="troubleshooting"></a><h1>Troubleshooting</h1>
<table>
<tr><th>You see</th><th>Do this</th></tr>
<tr><td>body → IMU orientation contradicts gravity</td><td>The configured mounting is wrong (often a 180° roll/pitch/yaw mix-up). Fix the body→IMU entry; VIO is unaffected but anything using the body frame is.</td></tr>
<tr><td>Intrinsics missing</td><td>Copy <code>/data/modalai/opencv_*_intrinsics.yml</code> from the vehicle into the log's <code>data/modalai</code>, or publish CameraInfo.</td></tr>
<tr><td>IMU rate differs from configured</td><td>The sensor runs at a different rate than <code>voxl-imu-server.conf</code> says; check the driver configuration.</td></tr>
<tr><td>VIO never initialized</td><td>Check the gravity finding and the Parameters notes; dynamic initialization needs motion, static needs a still start.</td></tr>
<tr><td>VIO diverged</td><td>Usually extrinsics: check the camera rotation (Orientation… shows the gravity check) and the time offset.</td></tr>
<tr><td>Calibration ABORT</td><td>Not enough excitation or inconsistent data. Record the motion described under Calibrate.</td></tr>
<tr><td>Images look upside down</td><td>Rotate the camera in Inspect (⟳ twice) and save if the vehicle should do the same (<code>en_rotate</code>).</td></tr>
</table>

<a name="about"></a><h1>About {app}</h1>
<p><b>{app}</b> {version} — {tagline}: log inspection for voxl-logger logs and ROS 1 / ROS 2 bags, ROS-free OpenVINS replay (CPU, or OpenCL through libmodal-flow when it was found at build time) and targetless camera–IMU calibration with ov_zcalib.</p>
<h2>Credits</h2>
<p>Created by <b>{author}</b> at {organization}.</p>
<p>With thanks to {thanks}</p>
<p class="muted">Built on OpenVINS by the Robot Perception and Navigation Group (RPNG), University of Delaware. GPL-3.0-or-later.</p>
"""


def thanks(names):
    """'A, B and C'."""
    names = list(names)
    text = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return text if text.endswith(".") else text + "."


class HelpPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        self.toc = QListWidget()
        self.toc.setFixedWidth(220)
        for anchor, title in SECTIONS:
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, anchor)
            self.toc.addItem(item)
        self.toc.currentItemChanged.connect(
            lambda item, _: (
                item and self.show_section(item.data(Qt.UserRole), sync=False)
            )
        )
        layout.addWidget(self.toc)
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        self.text.document().setDefaultStyleSheet(CSS)
        self.text.setHtml(
            HTML.replace("{app}", APP_NAME)
            .replace("{version}", __version__)
            .replace("{tagline}", TAGLINE)
            .replace("{author}", AUTHOR)
            .replace("{organization}", ORGANIZATION)
            .replace("{thanks}", thanks(ACKNOWLEDGEMENTS))
        )
        self.text.document().setDocumentMargin(22)
        self.text.setStyleSheet(
            f"QTextBrowser {{ background:{PANEL}; border-radius:12px; }}"
        )
        layout.addWidget(self.text, 1)

    def show_section(self, anchor, sync=True):
        self.text.scrollToAnchor(anchor)
        if sync:
            for i in range(self.toc.count()):
                if self.toc.item(i).data(Qt.UserRole) == anchor:
                    self.toc.blockSignals(True)
                    self.toc.setCurrentRow(i)
                    self.toc.blockSignals(False)
