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

#ifndef OV_CORE_SENSOR_STREAM_H
#define OV_CORE_SENSOR_STREAM_H

/**
 * @file sensor_stream.h
 * @brief ROS-free binary sensor stream (producer -> consumer pipe) and the VIO result stream.
 *
 * Any log reader (voxl-logger directory, ROS 1/2 bag, live driver) can feed an OpenVINS
 * consumer through a pipe without ROS, rosbag, or image transport. The producer decodes,
 * converts to 8-bit luminance and applies its timestamp convention; consumers only check
 * bounds and ordering. All values are little-endian. Every record is one tag byte followed
 * by its payload; there is no padding.
 *
 * Sensor stream v1 (producer -> consumer, e.g. ov_zcalib_track / ov_msckf_replay stdin)
 * ------------------------------------------------------------------------------------
 *   header : char[4] "OVSS", u32 version (= 1)
 *   'C'    : camera  u32 cam, i32 width, i32 height, f64 fx, f64 fy, f64 cx, f64 cy,
 *                    u32 nd, f64 distortion[nd], u8 model (0 radtan/plumb_bob, 1 equidistant),
 *                    f64 readout_s, u32 name_len, char name[name_len]
 *   'I'    : imu     f64 t, f64 wx, wy, wz [rad/s], f64 ax, ay, az [m/s^2],
 *                    f64 temperature_c (NaN = unavailable)
 *   'F'    : frame   u32 cam, f64 t, f32 exposure_s, i64 sync_ns (-1 = none), i32 width, i32 height,
 *                    u8 pixels[width*height] (mono8, row stride == width)
 *   'E'    : end of stream (no payload). EOF before 'E' is a truncated stream.
 *
 * Times are seconds relative to an epoch chosen by the producer. A frame's t is its
 * center-row mid-exposure instant: the producer has already applied
 * start-of-exposure + (readout + exposure) / 2 where its source requires it, and consumers
 * never shift it again (see ov_core::CameraData::exposures). Camera records precede all
 * data; camera ids are 0..N-1. Bounds: N <= 16 cameras, width*height <= 40e6,
 * nd <= 16, name_len <= 4096.
 *
 * VIO result stream v1 (ov_msckf_replay stdout -> viewer)
 * -------------------------------------------------------
 *   header : char[4] "OVVR", u32 version (= 1)
 *   'S'    : state   f64 t, f64 q_GtoI[4] (JPL x,y,z,w), f64 p_IinG[3], f64 v_IinG[3],
 *                    f64 bg[3], f64 ba[3], f64 sigma_ori[3], f64 sigma_pos[3],
 *                    u8 ncam, ncam * (f64 q_ItoC[4] (JPL), f64 p_IinC[3], f64 td),
 *                    f64 proc_ms, u8 flags (bit0 initialized, bit1 zero-velocity update),
 *                    u32 n_slam, u32 n_msckf
 *   'P'    : points  f64 t, u32 n, n * (f32 x, f32 y, f32 z, u8 kind 0 MSCKF / 1 SLAM / 2 ARUCO)
 *   'T'    : tracks  f64 t, u32 cam, u32 n, n * (u32 id, f32 u, f32 v)   (distorted pixels)
 *   'M'    : message u8 level (0 info, 1 warning, 2 error), u32 len, char text[len]
 *   'E'    : end     u8 ok
 *
 * td is the camera-to-IMU time offset t_imu = t_cam + td. Sigmas are square roots of the
 * marginal covariance diagonals (orientation in the IMU frame [rad], position in global [m]).
 * The Python counterpart is zstudio/src/zstudio/stream.py.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov_core {
namespace sensor_stream {

constexpr char kSensorMagic[4] = {'O', 'V', 'S', 'S'};
constexpr uint32_t kSensorVersion = 1;
constexpr char kResultMagic[4] = {'O', 'V', 'V', 'R'};
constexpr uint32_t kResultVersion = 1;
constexpr uint32_t kMaxCameras = 16;
constexpr uint64_t kMaxPixels = 40000000ull;
constexpr uint32_t kMaxDistortion = 16;
constexpr uint32_t kMaxName = 4096;

enum : uint8_t {
  kTagCamera = 'C',
  kTagImu = 'I',
  kTagFrame = 'F',
  kTagEnd = 'E',
  kTagState = 'S',
  kTagPoints = 'P',
  kTagTracks = 'T',
  kTagMessage = 'M',
};

/// Little-endian encode/decode independent of host byte order.
template <class T> inline void store_le(uint8_t *out, T value) {
  uint8_t bytes[sizeof(T)];
  std::memcpy(bytes, &value, sizeof(T));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  for (size_t i = 0; i < sizeof(T); ++i)
    out[i] = bytes[sizeof(T) - 1 - i];
#else
  std::memcpy(out, bytes, sizeof(T));
#endif
}

template <class T> inline T load_le(const uint8_t *in) {
  uint8_t bytes[sizeof(T)];
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  for (size_t i = 0; i < sizeof(T); ++i)
    bytes[i] = in[sizeof(T) - 1 - i];
#else
  std::memcpy(bytes, in, sizeof(T));
#endif
  T value;
  std::memcpy(&value, bytes, sizeof(T));
  return value;
}

/// Buffered little-endian writer over a stdio stream (no allocation per field).
class BinaryWriter {
public:
  explicit BinaryWriter(std::FILE *out) : out_(out) {
    if (out_ == nullptr)
      throw std::invalid_argument("sensor_stream: null output stream");
  }
  template <class T> void put(T value) {
    uint8_t bytes[sizeof(T)];
    store_le(bytes, value);
    bytes_(bytes, sizeof(T));
  }
  void tag(uint8_t value) { bytes_(&value, 1); }
  void raw(const void *data, size_t size) { bytes_(data, size); }
  void magic(const char (&magic)[4], uint32_t version) {
    raw(magic, 4);
    put<uint32_t>(version);
  }
  void flush() {
    if (std::fflush(out_) != 0)
      throw std::runtime_error("sensor_stream: write failed (consumer closed?)");
  }

private:
  void bytes_(const void *data, size_t size) {
    if (size && std::fwrite(data, 1, size, out_) != size)
      throw std::runtime_error("sensor_stream: write failed (consumer closed?)");
  }
  std::FILE *out_;
};

/// Buffered little-endian reader over a stdio stream; every read is exact or throws.
class BinaryReader {
public:
  explicit BinaryReader(std::FILE *in) : in_(in) {
    if (in_ == nullptr)
      throw std::invalid_argument("sensor_stream: null input stream");
  }
  template <class T> T get() {
    uint8_t bytes[sizeof(T)];
    raw(bytes, sizeof(T));
    return load_le<T>(bytes);
  }
  void raw(void *data, size_t size) {
    if (size && std::fread(data, 1, size, in_) != size)
      throw std::runtime_error(std::feof(in_) ? "sensor_stream: truncated stream (EOF before end record)"
                                              : "sensor_stream: read error");
  }
  /// Tag byte, or -1 on a clean EOF exactly at a record boundary.
  int tag() {
    const int c = std::fgetc(in_);
    if (c == EOF && std::ferror(in_))
      throw std::runtime_error("sensor_stream: read error");
    return c;
  }
  void expect_magic(const char (&magic)[4], uint32_t version) {
    char got[4];
    raw(got, 4);
    if (std::memcmp(got, magic, 4) != 0)
      throw std::runtime_error(std::string("sensor_stream: bad magic (expected ") + std::string(magic, 4) + ")");
    const uint32_t v = get<uint32_t>();
    if (v != version)
      throw std::runtime_error("sensor_stream: unsupported version " + std::to_string(v) + " (expected " +
                               std::to_string(version) + ")");
  }

private:
  std::FILE *in_;
};

/// Distortion model codes carried by camera records.
enum class StreamModel : uint8_t { Radtan = 0, Equidistant = 1 };

struct StreamCamera {
  uint32_t cam = 0;
  int32_t width = 0, height = 0;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  std::vector<double> distortion;
  StreamModel model = StreamModel::Radtan;
  double readout_s = 0;
  std::string name;
};

struct StreamImu {
  double t = 0;
  double w[3] = {0, 0, 0};
  double a[3] = {0, 0, 0};
  double temperature_c = std::numeric_limits<double>::quiet_NaN();
};

struct StreamFrame {
  uint32_t cam = 0;
  double t = 0;
  float exposure_s = 0;
  int64_t sync_ns = -1;
  int32_t width = 0, height = 0;
  std::vector<uint8_t> pixels; ///< row-major mono8, stride == width
};

enum class StreamRecord { Camera, Imu, Frame, End };

/// Pull parser for the sensor stream. Validates every bound before allocating.
class SensorStreamReader {
public:
  explicit SensorStreamReader(std::FILE *in) : in_(in) {}

  void read_header() { in_.expect_magic(kSensorMagic, kSensorVersion); }

  StreamRecord next() {
    const int tag = in_.tag();
    if (tag < 0)
      throw std::runtime_error("sensor_stream: truncated stream (EOF before end record)");
    switch (tag) {
    case kTagCamera:
      read_camera_();
      return StreamRecord::Camera;
    case kTagImu:
      imu_.t = in_.get<double>();
      for (double &v : imu_.w)
        v = in_.get<double>();
      for (double &v : imu_.a)
        v = in_.get<double>();
      imu_.temperature_c = in_.get<double>();
      return StreamRecord::Imu;
    case kTagFrame:
      read_frame_();
      return StreamRecord::Frame;
    case kTagEnd:
      return StreamRecord::End;
    default:
      throw std::runtime_error("sensor_stream: unknown record tag " + std::to_string(tag));
    }
  }

  const StreamCamera &camera() const { return camera_; }
  const StreamImu &imu() const { return imu_; }
  StreamFrame &frame() { return frame_; }

private:
  void read_camera_() {
    camera_.cam = in_.get<uint32_t>();
    camera_.width = in_.get<int32_t>();
    camera_.height = in_.get<int32_t>();
    camera_.fx = in_.get<double>();
    camera_.fy = in_.get<double>();
    camera_.cx = in_.get<double>();
    camera_.cy = in_.get<double>();
    const uint32_t nd = in_.get<uint32_t>();
    if (nd > kMaxDistortion)
      throw std::runtime_error("sensor_stream: camera record has " + std::to_string(nd) + " distortion terms (max 16)");
    camera_.distortion.resize(nd);
    for (double &v : camera_.distortion)
      v = in_.get<double>();
    const uint8_t model = in_.get<uint8_t>();
    if (model > 1)
      throw std::runtime_error("sensor_stream: unknown distortion model code " + std::to_string(model));
    camera_.model = static_cast<StreamModel>(model);
    camera_.readout_s = in_.get<double>();
    const uint32_t len = in_.get<uint32_t>();
    if (len > kMaxName)
      throw std::runtime_error("sensor_stream: camera name too long");
    camera_.name.assign(len, '\0');
    in_.raw(&camera_.name[0], len);
    if (camera_.cam >= kMaxCameras)
      throw std::runtime_error("sensor_stream: camera id " + std::to_string(camera_.cam) + " exceeds 15");
    check_size_(camera_.width, camera_.height);
  }

  void read_frame_() {
    frame_.cam = in_.get<uint32_t>();
    frame_.t = in_.get<double>();
    frame_.exposure_s = in_.get<float>();
    frame_.sync_ns = in_.get<int64_t>();
    frame_.width = in_.get<int32_t>();
    frame_.height = in_.get<int32_t>();
    if (frame_.cam >= kMaxCameras)
      throw std::runtime_error("sensor_stream: frame camera id " + std::to_string(frame_.cam) + " exceeds 15");
    check_size_(frame_.width, frame_.height);
    frame_.pixels.resize(static_cast<size_t>(frame_.width) * static_cast<size_t>(frame_.height));
    in_.raw(frame_.pixels.data(), frame_.pixels.size());
  }

  static void check_size_(int32_t width, int32_t height) {
    if (width <= 0 || height <= 0 || static_cast<uint64_t>(width) * static_cast<uint64_t>(height) > kMaxPixels)
      throw std::runtime_error("sensor_stream: invalid image size " + std::to_string(width) + "x" +
                               std::to_string(height));
  }

  BinaryReader in_;
  StreamCamera camera_;
  StreamImu imu_;
  StreamFrame frame_;
};

/// Producer side of the sensor stream (C++ producers and tests; Python has its own writer).
class SensorStreamWriter {
public:
  explicit SensorStreamWriter(std::FILE *out) : out_(out) { out_.magic(kSensorMagic, kSensorVersion); }

  void camera(const StreamCamera &c) {
    if (c.distortion.size() > kMaxDistortion || c.name.size() > kMaxName)
      throw std::invalid_argument("sensor_stream: camera record exceeds bounds");
    out_.tag(kTagCamera);
    out_.put<uint32_t>(c.cam);
    out_.put<int32_t>(c.width);
    out_.put<int32_t>(c.height);
    for (double v : {c.fx, c.fy, c.cx, c.cy})
      out_.put<double>(v);
    out_.put<uint32_t>(static_cast<uint32_t>(c.distortion.size()));
    for (double v : c.distortion)
      out_.put<double>(v);
    out_.put<uint8_t>(static_cast<uint8_t>(c.model));
    out_.put<double>(c.readout_s);
    out_.put<uint32_t>(static_cast<uint32_t>(c.name.size()));
    out_.raw(c.name.data(), c.name.size());
  }

  void imu(const StreamImu &s) {
    out_.tag(kTagImu);
    out_.put<double>(s.t);
    for (double v : s.w)
      out_.put<double>(v);
    for (double v : s.a)
      out_.put<double>(v);
    out_.put<double>(s.temperature_c);
  }

  void frame(const StreamFrame &f) {
    if (f.pixels.size() != static_cast<size_t>(f.width) * static_cast<size_t>(f.height))
      throw std::invalid_argument("sensor_stream: frame pixel count does not match its size");
    out_.tag(kTagFrame);
    out_.put<uint32_t>(f.cam);
    out_.put<double>(f.t);
    out_.put<float>(f.exposure_s);
    out_.put<int64_t>(f.sync_ns);
    out_.put<int32_t>(f.width);
    out_.put<int32_t>(f.height);
    out_.raw(f.pixels.data(), f.pixels.size());
  }

  void end() {
    out_.tag(kTagEnd);
    out_.flush();
  }

private:
  BinaryWriter out_;
};

} // namespace sensor_stream
} // namespace ov_core

#endif // OV_CORE_SENSOR_STREAM_H
