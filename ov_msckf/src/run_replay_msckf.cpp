/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 * ov_msckf_replay: ROS-free VIO over a recorded sensor stream.
 *
 *   <producer> | ov_msckf_replay <estimator_config.yaml> [--tum OUT.txt] [--points-every N]
 *                                [--time-offset S]
 *   ov_msckf_replay --capabilities
 *
 * --time-offset S is subtracted from every reported time (results and TUM). Producers shift a
 * log's time base by an integer S so the estimator never sees non-positive stamps (OpenVINS uses
 * negative times as "unset"); reporting in the log's own time keeps viewers aligned 1:1.
 *
 * stdin carries the ov_core sensor stream (utils/sensor_stream.h); the original stdout carries
 * the VIO result stream defined in the same header. Every OpenVINS PRINT_* line is redirected
 * to stderr so the binary result channel is never corrupted.
 *
 * Input follows the live ingest contract: each view is pushed as its own CameraData with its own
 * center-row mid-exposure stamp (bit-equal stamps are bundled by AsyncCameraBuffer). When the
 * configuration selects forced camera sync (stereo or force_camera_sync) and the producer gave no
 * hardware trigger key, views within 0.5 ms share a key, so the buffer assigns the reference
 * camera's time to the group exactly like a forced-sync driver would.
 */

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "track/TrackBase.h"
#include "track/TrackDescriptor.h"
#include "types/IMU.h"
#include "types/Landmark.h"
#include "types/PoseJPL.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_stream.h"
#ifndef DISABLE_TRACK_KLT
#include "track/TrackKLT.h"
#endif
#if OV_HAVE_MODAL_FLOW
#include "track/TrackOCL/TrackOCL.h"
#if defined(__has_include)
#if __has_include(<modal_flow/Version.hpp>)
#include <modal_flow/Version.hpp>
#define OV_REPLAY_MODAL_FLOW_VERSION MODAL_FLOW_VERSION_STR
#endif
#endif
#endif

#include <dlfcn.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace ov_core::sensor_stream;
using Clock = std::chrono::steady_clock;

namespace {

constexpr double kGroupTolerance = 0.0005; ///< near-coincident views share a forced-sync key
constexpr size_t kMaxMsckfPoints = 4000;

std::string json_escape(const std::string &text) {
  std::string out;
  for (char c : text) {
    if (c == '"' || c == '\\')
      out += '\\';
    out += c;
  }
  return out;
}

/// Machine-readable build capabilities (one JSON line on stdout).
int print_capabilities() {
  std::string trackers = "\"klt\"";
#ifdef DISABLE_TRACK_KLT
  trackers = "";
#endif
#if OV_HAVE_MODAL_FLOW
  trackers += std::string(trackers.empty() ? "" : ",") + "\"ocl\"";
#endif
  trackers += std::string(trackers.empty() ? "" : ",") + "\"descriptor\"";
  std::string version = "null", runtime = "null";
#ifdef OV_REPLAY_MODAL_FLOW_VERSION
  version = "\"" + json_escape(OV_REPLAY_MODAL_FLOW_VERSION) + "\"";
#endif
#if OV_HAVE_MODAL_FLOW
  // Resolved at run time so this binary never needs a direct link to libmodal_flow; pre-1.0.9
  // libraries have no stamp and report null.
  using VersionFn = const char *(*)();
  if (auto fn = reinterpret_cast<VersionFn>(dlsym(RTLD_DEFAULT, "modal_flow_runtime_version")))
    runtime = "\"" + json_escape(fn()) + "\"";
#endif
  std::printf("{\"sensor_stream\":%u,\"vio_results\":%u,\"trackers\":[%s],\"modal_flow\":%s,"
              "\"modal_flow_version\":%s,\"modal_flow_runtime_version\":%s}\n",
              kSensorVersion, kResultVersion, trackers.c_str(), OV_HAVE_MODAL_FLOW ? "true" : "false", version.c_str(),
              runtime.c_str());
  std::fflush(stdout);
  return 0;
}

/// Owns the duplicated stdout result channel. Records are flushed per update so a viewer sees
/// them immediately; a closed reader turns into an exception, not SIGPIPE.
class Results {
public:
  explicit Results(std::FILE *out) : file_(out), out_(out) { out_.magic(kResultMagic, kResultVersion); }
  ~Results() {
    if (file_)
      std::fclose(file_);
  }

  void message(int level, const std::string &text) {
    out_.tag(kTagMessage);
    out_.put<uint8_t>(static_cast<uint8_t>(level));
    out_.put<uint32_t>(static_cast<uint32_t>(text.size()));
    out_.raw(text.data(), text.size());
    out_.flush();
  }

  void end(bool ok) {
    if (ended_)
      return;
    ended_ = true;
    out_.tag(kTagEnd);
    out_.put<uint8_t>(ok ? 1 : 0);
    out_.flush();
  }
  bool ended() const { return ended_; }

  ov_core::sensor_stream::BinaryWriter &w() { return out_; }

private:
  std::FILE *file_;
  BinaryWriter out_;
  bool ended_ = false;
};

/// Set once results exist so an std::exit inside option parsing still terminates the stream.
Results *g_results = nullptr;
void end_on_exit() {
  if (g_results && !g_results->ended()) {
    try {
      g_results->message(2, "estimator exited during setup (see the console for the reason)");
      g_results->end(false);
    } catch (...) {
    }
  }
}

std::string tracker_name(const std::shared_ptr<ov_core::TrackBase> &tracker) {
  if (!tracker)
    return "none";
#if OV_HAVE_MODAL_FLOW
  if (std::dynamic_pointer_cast<ov_core::TrackOCL>(tracker))
    return "TrackOCL (OpenCL, libmodal-flow)";
#endif
#ifndef DISABLE_TRACK_KLT
  if (std::dynamic_pointer_cast<ov_core::TrackKLT>(tracker))
    return "TrackKLT (CPU KLT)";
#endif
  if (std::dynamic_pointer_cast<ov_core::TrackDescriptor>(tracker))
    return "TrackDescriptor (CPU ORB descriptors)";
  return "unknown tracker";
}

struct Options {
  std::string config;
  std::string tum;
  int points_every = 1;
  double time_offset = 0.0;
};

Options parse_args(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tum" && i + 1 < argc)
      o.tum = argv[++i];
    else if (arg == "--points-every" && i + 1 < argc)
      o.points_every = std::max(1, std::atoi(argv[++i]));
    else if (arg == "--time-offset" && i + 1 < argc) {
      char *end = nullptr;
      o.time_offset = std::strtod(argv[++i], &end);
      if (end == argv[i] || *end != '\0' || !std::isfinite(o.time_offset))
        throw std::runtime_error("--time-offset expects seconds");
    } else if (!arg.empty() && arg[0] == '-')
      throw std::runtime_error("unknown option " + arg);
    else if (o.config.empty())
      o.config = arg;
    else
      throw std::runtime_error("only one configuration file may be given");
  }
  if (o.config.empty())
    throw std::runtime_error("usage: <producer> | ov_msckf_replay <estimator_config.yaml> [--tum OUT.txt] "
                             "[--points-every N] [--time-offset S]\n       ov_msckf_replay --capabilities");
  return o;
}

class Replay {
public:
  Replay(const Options &options, Results &results) : opt_(options), results_(results) {}

  int run() {
    if (FILE *probe = std::fopen(opt_.config.c_str(), "r"))
      std::fclose(probe);
    else
      throw std::runtime_error("cannot open configuration " + opt_.config);

    auto parser = std::make_shared<ov_core::YamlParser>(opt_.config);
    std::string verbosity = "INFO";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);
    params_.print_and_load(parser);
    params_.use_multi_threading_subs = false;
    if (!parser->successful())
      throw std::runtime_error("unable to parse all parameters in " + opt_.config + " (see the console)");
    num_cameras_ = params_.state_options.num_cameras;
    forced_sync_ = params_.synchronize_camera_timestamps();

    sys_ = std::make_shared<ov_msckf::VioManager>(params_);
    sys_->prepare_camera_replay();
    sys_->set_camera_processed_callback(
        [this](const ov_core::CameraData &msg, bool processed) { return on_processed(msg, processed); });
    results_.message(0, "tracker: " + tracker_name(sys_->get_track_feats()) +
                            (params_.use_gpu && tracker_name(sys_->get_track_feats()).rfind("TrackOCL", 0) != 0
                                 ? " (use_gpu requested, OpenCL unavailable in this build)"
                                 : ""));
    results_.message(0, "cameras: " + std::to_string(num_cameras_) +
                            (forced_sync_ ? ", forced sync" : ", asynchronous") +
                            (params_.use_stereo ? ", stereo" : ""));
    if (!opt_.tum.empty()) {
      tum_.open(opt_.tum);
      if (!tum_)
        throw std::runtime_error("cannot write " + opt_.tum);
      tum_ << "# timestamp tx ty tz qx qy qz qw (OpenVINS q_GtoI JPL == Hamilton q_ItoG)\n";
      tum_.precision(9);
    }

    std::setvbuf(stdin, nullptr, _IOFBF, 1 << 22);
    SensorStreamReader reader(stdin);
    reader.read_header();
    StreamRecord record = reader.next();
    std::vector<bool> declared(kMaxCameras, false);
    while (record == StreamRecord::Camera) {
      declare_camera(reader.camera(), declared);
      record = reader.next();
    }
    for (int i = 0; i < num_cameras_; ++i)
      if (!declared[i])
        throw std::runtime_error("the configuration uses " + std::to_string(num_cameras_) +
                                 " cameras but the stream did not declare camera " + std::to_string(i));

    for (; record != StreamRecord::End; record = reader.next()) {
      if (record == StreamRecord::Camera)
        throw std::runtime_error("camera records must precede sensor data");
      if (record == StreamRecord::Imu) {
        const StreamImu &s = reader.imu();
        if (pending_.size() && s.t > pending_t_ + kGroupTolerance)
          flush_group();
        if (!std::isfinite(s.t) || s.t <= last_imu_) {
          ++dropped_imu_;
          continue;
        }
        ov_core::ImuData imu;
        imu.timestamp = s.t;
        imu.wm << s.w[0], s.w[1], s.w[2];
        imu.am << s.a[0], s.a[1], s.a[2];
        last_imu_ = s.t;
        ++imu_count_;
        feed_start_ = Clock::now();
        sys_->feed_measurement_imu(imu);
      } else {
        StreamFrame &f = reader.frame();
        if ((int)f.cam >= num_cameras_)
          continue; // extra streamed cameras the configuration does not use
        const bool joins = !pending_.empty() && std::abs(f.t - pending_t_) <= kGroupTolerance &&
                           std::none_of(pending_.begin(), pending_.end(),
                                        [&](const ov_core::CameraData &m) { return m.sensor_ids[0] == (int)f.cam; });
        if (!pending_.empty() && !joins)
          flush_group();
        add_frame(f);
      }
    }
    flush_group();
    feed_start_ = Clock::now();
    sys_->finish_camera_replay();

    const double wall = std::chrono::duration<double>(Clock::now() - start_).count();
    char summary[512];
    std::snprintf(summary, sizeof(summary),
                  "finished: %llu IMU, %llu frames pushed, %llu updates, %llu dropped by the ingest, %llu by "
                  "track_frequency, %llu non-monotonic IMU; mean update %.2f ms; %.1f s wall%s",
                  (unsigned long long)imu_count_, (unsigned long long)frames_pushed_, (unsigned long long)updates_,
                  (unsigned long long)ingest_dropped_, (unsigned long long)rate_dropped_,
                  (unsigned long long)dropped_imu_, updates_ ? total_ms_ / updates_ : 0.0, wall,
                  sys_->initialized() ? "" : "; never initialized");
    results_.message(sys_->initialized() ? 0 : 1, summary);
    results_.end(true);
    return 0;
  }

private:
  void declare_camera(const StreamCamera &c, std::vector<bool> &declared) {
    declared[c.cam] = true;
    if ((int)c.cam >= num_cameras_) {
      results_.message(1, "stream camera " + std::to_string(c.cam) + " (" + c.name +
                              ") is not used by the configuration and will be ignored");
      return;
    }
    auto it = params_.camera_intrinsics.find(c.cam);
    if (it != params_.camera_intrinsics.end() && (it->second->w() != c.width || it->second->h() != c.height)) {
      results_.message(1, "camera " + std::to_string(c.cam) + " (" + c.name + ") streams " + std::to_string(c.width) +
                              "x" + std::to_string(c.height) + " but the configuration is calibrated for " +
                              std::to_string(it->second->w()) + "x" + std::to_string(it->second->h()));
    }
    names_.resize(std::max<size_t>(names_.size(), c.cam + 1));
    names_[c.cam] = c.name;
  }

  void add_frame(StreamFrame &f) {
    ov_core::CameraData m;
    m.timestamp = f.t;
    m.sync_timestamp_ns = f.sync_ns;
    m.sensor_ids = {(int)f.cam};
    cv::Mat image(f.height, f.width, CV_8UC1);
    std::memcpy(image.data, f.pixels.data(), f.pixels.size());
    m.images = {image};
    if (params_.use_mask && params_.masks.count(f.cam))
      m.masks = {params_.masks.at(f.cam)};
    else
      m.masks = {cv::Mat::zeros(f.height, f.width, CV_8UC1)};
    m.exposures = {f.exposure_s};
    if (pending_.empty())
      pending_t_ = f.t;
    pending_.push_back(std::move(m));
  }

  /// Push one near-coincident group, applying track_frequency like the ROS callbacks.
  void flush_group() {
    if (pending_.empty())
      return;
    const double period = params_.track_frequency > 0 ? 1.0 / params_.track_frequency : 0.0;
    if (forced_sync_) {
      // A forced-sync group is kept or dropped as a whole; partial groups would only expire.
      const bool keep = pending_t_ >= last_group_ + period - 1e-9;
      if (keep) {
        last_group_ = pending_t_;
        int64_t key = -1;
        for (const auto &m : pending_)
          if (m.sync_timestamp_ns >= 0)
            key = m.sync_timestamp_ns;
        if (key < 0)
          key = static_cast<int64_t>(std::llround(pending_t_ * 1e9));
        for (auto &m : pending_) {
          if (m.sync_timestamp_ns < 0)
            m.sync_timestamp_ns = key;
          push(m);
        }
      } else {
        rate_dropped_ += pending_.size();
      }
    } else {
      for (auto &m : pending_) {
        const int cam = m.sensor_ids[0];
        if (last_frame_.size() <= (size_t)cam)
          last_frame_.resize(cam + 1, -1e300);
        if (m.timestamp < last_frame_[cam] + period - 1e-9) {
          ++rate_dropped_;
          continue;
        }
        last_frame_[cam] = m.timestamp;
        push(m);
      }
    }
    pending_.clear();
  }

  void push(const ov_core::CameraData &m) {
    ++frames_pushed_;
    sys_->feed_measurement_camera(m);
  }

  bool on_processed(const ov_core::CameraData &msg, bool processed) {
    if (!processed) {
      ingest_dropped_ += msg.sensor_ids.size();
      return true;
    }
    const auto now = Clock::now();
    const auto since = std::max(feed_start_, last_done_);
    const double proc_ms = std::chrono::duration<double, std::milli>(now - since).count();
    last_done_ = now;
    ++updates_;
    total_ms_ += proc_ms;
    auto &w = results_.w();
    auto state = sys_->get_state();
    const bool initialized = sys_->initialized();

    // Tracker observations (raw distorted pixels) for every view in this update
    if (auto tracker = sys_->get_track_feats()) {
      auto obs = tracker->get_last_obs();
      auto ids = tracker->get_last_ids();
      for (int cam : msg.sensor_ids) {
        const auto &kps = obs[(size_t)cam];
        const auto &kid = ids[(size_t)cam];
        const uint32_t n = static_cast<uint32_t>(std::min(kps.size(), kid.size()));
        w.tag(kTagTracks);
        w.put<double>(msg.timestamp - opt_.time_offset);
        w.put<uint32_t>(static_cast<uint32_t>(cam));
        w.put<uint32_t>(n);
        for (uint32_t i = 0; i < n; ++i) {
          w.put<uint32_t>(static_cast<uint32_t>(kid[i]));
          w.put<float>(kps[i].pt.x);
          w.put<float>(kps[i].pt.y);
        }
      }
    }

    // Reported in the producer's time base (see --time-offset).
    const double t = (initialized ? state->imu_endpoint() : msg.timestamp) - opt_.time_offset;
    const Eigen::Vector4d q = state->_imu->quat();
    const Eigen::Vector3d p = state->_imu->pos(), v = state->_imu->vel();
    const Eigen::Vector3d bg = state->_imu->bias_g(), ba = state->_imu->bias_a();
    Eigen::Matrix<double, 6, 1> sigma = Eigen::Matrix<double, 6, 1>::Zero();
    if (initialized) {
      std::vector<std::shared_ptr<ov_type::Type>> vars = {state->_imu->q(), state->_imu->p()};
      const Eigen::MatrixXd cov = ov_msckf::StateHelper::get_marginal_covariance(state, vars);
      for (int i = 0; i < 6; ++i)
        sigma(i) = std::sqrt(std::max(0.0, cov(i, i)));
    }
    const std::vector<Eigen::Vector3d> msckf = sys_->get_good_features_MSCKF();
    w.tag(kTagState);
    w.put<double>(t);
    for (int i = 0; i < 4; ++i)
      w.put<double>(q(i));
    for (const auto *vec : {&p, &v, &bg, &ba})
      for (int i = 0; i < 3; ++i)
        w.put<double>((*vec)(i));
    for (int i = 0; i < 6; ++i)
      w.put<double>(sigma(i));
    w.put<uint8_t>(static_cast<uint8_t>(num_cameras_));
    for (int c = 0; c < num_cameras_; ++c) {
      auto calib = state->_calib_IMUtoCAM.at((size_t)c);
      const Eigen::Vector4d qc = calib->quat();
      const Eigen::Vector3d pc = calib->pos();
      for (int i = 0; i < 4; ++i)
        w.put<double>(qc(i));
      for (int i = 0; i < 3; ++i)
        w.put<double>(pc(i));
      w.put<double>(state->cam_imu_dt((size_t)c));
    }
    w.put<double>(proc_ms);
    w.put<uint8_t>(static_cast<uint8_t>((initialized ? 1 : 0) | (sys_->get_did_zupt_update() ? 2 : 0)));
    w.put<uint32_t>(static_cast<uint32_t>(state->_features_SLAM.size()));
    w.put<uint32_t>(static_cast<uint32_t>(msckf.size()));

    if (initialized && (updates_ % (uint64_t)opt_.points_every) == 0) {
      const std::vector<Eigen::Vector3d> slam = sys_->get_features_SLAM();
      const std::vector<Eigen::Vector3d> aruco = sys_->get_features_ARUCO();
      const size_t step = msckf.size() > kMaxMsckfPoints ? (msckf.size() + kMaxMsckfPoints - 1) / kMaxMsckfPoints : 1;
      const uint32_t n = static_cast<uint32_t>((msckf.size() + step - 1) / step + slam.size() + aruco.size());
      w.tag(kTagPoints);
      w.put<double>(t);
      w.put<uint32_t>(n);
      auto put_point = [&](const Eigen::Vector3d &x, uint8_t kind) {
        w.put<float>(static_cast<float>(x(0)));
        w.put<float>(static_cast<float>(x(1)));
        w.put<float>(static_cast<float>(x(2)));
        w.put<uint8_t>(kind);
      };
      for (size_t i = 0; i < msckf.size(); i += step)
        put_point(msckf[i], 0);
      for (const auto &x : slam)
        put_point(x, 1);
      for (const auto &x : aruco)
        put_point(x, 2);
    }
    w.flush();

    if (initialized && !was_initialized_) {
      char text[160];
      std::snprintf(text, sizeof(text), "initialized at t=%.3f s after %llu updates", t, (unsigned long long)updates_);
      results_.message(0, text);
    } else if (!initialized && was_initialized_) {
      results_.message(1, "estimator reset; reinitializing");
    }
    was_initialized_ = initialized;
    if (initialized && tum_) {
      tum_ << std::fixed << t << ' ' << p(0) << ' ' << p(1) << ' ' << p(2) << ' ' << q(0) << ' ' << q(1) << ' ' << q(2)
           << ' ' << q(3) << '\n';
    }
    return true;
  }

  Options opt_;
  Results &results_;
  ov_msckf::VioManagerOptions params_;
  std::shared_ptr<ov_msckf::VioManager> sys_;
  int num_cameras_ = 0;
  bool forced_sync_ = false;
  std::vector<std::string> names_;
  std::vector<ov_core::CameraData> pending_;
  double pending_t_ = 0, last_imu_ = -1e300, last_group_ = -1e300;
  std::vector<double> last_frame_;
  std::ofstream tum_;
  bool was_initialized_ = false;
  uint64_t imu_count_ = 0, frames_pushed_ = 0, updates_ = 0, ingest_dropped_ = 0, rate_dropped_ = 0, dropped_imu_ = 0;
  double total_ms_ = 0;
  Clock::time_point start_ = Clock::now(), feed_start_ = Clock::now(), last_done_ = Clock::now();
};

} // namespace

int main(int argc, char **argv) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--capabilities") == 0)
      return print_capabilities();

  // Keep the original stdout for binary results; everything printed by OpenVINS goes to stderr.
  std::fflush(stdout);
  const int result_fd = dup(STDOUT_FILENO);
  std::FILE *result_file = result_fd >= 0 ? fdopen(result_fd, "wb") : nullptr;
  if (result_file == nullptr || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
    std::fprintf(stderr, "[replay] cannot set up the result channel\n");
    return 1;
  }
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::setvbuf(result_file, nullptr, _IOFBF, 1 << 16);
  signal(SIGPIPE, SIG_IGN);

  Results results(result_file);
  g_results = &results;
  std::atexit(end_on_exit);
  int code = 1;
  try {
    const Options options = parse_args(argc, argv);
    Replay replay(options, results);
    code = replay.run();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[replay] error: %s\n", error.what());
    try {
      results.message(2, error.what());
      results.end(false);
    } catch (...) {
    }
    code = 1;
  }
  g_results = nullptr;
  return code;
}
