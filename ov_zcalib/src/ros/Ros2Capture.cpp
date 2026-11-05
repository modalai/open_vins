// SPDX-License-Identifier: GPL-3.0-or-later
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include "CaptureCore.h"

class CaptureNode : public rclcpp::Node {
public:
  CaptureNode() : Node("ov_zcalib_capture") {
    const std::string path = declare_parameter<std::string>("record_path", "");
    const double readout = declare_parameter<double>("readout_s", 0.0);
    const double exposure = declare_parameter<double>("exposure_s", 0.0);
    const std::string stamp =
        declare_parameter<std::string>("timestamp_reference", "center");
    if (stamp != "center" && stamp != "start")
      throw std::runtime_error("timestamp_reference must be center or start");
    const int features = declare_parameter<int>("max_features", 200);
    auto image_qos = rclcpp::SensorDataQoS().keep_last(10);
    if (declare_parameter<bool>("reliable_images", false))
      image_qos.reliable();
    const auto image_topics = declare_parameter<std::vector<std::string>>(
        "image_topics",
        {declare_parameter<std::string>("image_topic", "/camera/image_raw")});
    const auto info_topics = declare_parameter<std::vector<std::string>>(
        "info_topics",
        {declare_parameter<std::string>("info_topic", "/camera/camera_info")});
    const auto readouts = declare_parameter<std::vector<double>>(
        "readout_times", std::vector<double>(image_topics.size(), readout));
    const auto exposures = declare_parameter<std::vector<double>>(
        "exposure_times", std::vector<double>(image_topics.size(), exposure));
    const auto references = declare_parameter<std::vector<std::string>>(
        "timestamp_references",
        std::vector<std::string>(image_topics.size(), stamp));
    if (info_topics.size() != image_topics.size() ||
        readouts.size() != image_topics.size() ||
        exposures.size() != image_topics.size() ||
        references.size() != image_topics.size())
      throw std::runtime_error(
          "camera topic and timing arrays must have the same length");
    std::vector<ov_zcalib::CaptureCore::CameraTiming> timing;
    for (size_t i = 0; i < image_topics.size(); ++i) {
      if (references[i] != "center" && references[i] != "start")
        throw std::runtime_error(
            "timestamp_references must contain center or start");
      if (image_topics[i].empty() || info_topics[i].empty())
        throw std::runtime_error("camera topic names must not be empty");
      timing.push_back({readouts[i], exposures[i], references[i] == "start"});
    }
    ov_zcalib::ImuNoise noise;
    noise.sigma_w = declare_parameter<double>("sigma_w", noise.sigma_w);
    noise.sigma_wb = declare_parameter<double>("sigma_wb", noise.sigma_wb);
    noise.sigma_a = declare_parameter<double>("sigma_a", noise.sigma_a);
    noise.sigma_ab = declare_parameter<double>("sigma_ab", noise.sigma_ab);
    core_ =
        std::make_unique<ov_zcalib::CaptureCore>(path, timing, features, noise);
    tracks_ = create_publisher<sensor_msgs::msg::PointCloud>(
        "/ov_zcalib/tracks", rclcpp::SensorDataQoS());
    for (size_t camera = 0; camera < image_topics.size(); ++camera) {
      infos_.push_back(create_subscription<sensor_msgs::msg::CameraInfo>(
          info_topics[camera], rclcpp::SensorDataQoS(),
          [this, camera](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
            guard([&] {
              core_->configure(msg->width, msg->height,
                               {msg->k[0], msg->k[4], msg->k[2], msg->k[5]},
                               msg->d, msg->distortion_model, camera);
            });
          }));
    }
    imu_ = create_subscription<sensor_msgs::msg::Imu>(
        declare_parameter<std::string>("imu_topic", "/imu/data_raw"),
        rclcpp::SensorDataQoS().keep_last(2000),
        [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) {
          guard([&] {
            if (msg->angular_velocity_covariance[0] == -1 ||
                msg->linear_acceleration_covariance[0] == -1)
              throw std::runtime_error("IMU message declares gyro or "
                                       "accelerometer data unavailable");
            ov_zcalib::RawImu sample;
            // sensor_msgs/Imu has no temperature. Keep the record's legacy
            // unavailable sentinel (0), never infer a physical 0 degrees C.
            sample.temp_c = 0.0;
            sample.timestamp = rclcpp::Time(msg->header.stamp).seconds();
            sample.wm << msg->angular_velocity.x, msg->angular_velocity.y,
                msg->angular_velocity.z;
            sample.am << msg->linear_acceleration.x, msg->linear_acceleration.y,
                msg->linear_acceleration.z;
            core_->imu(sample);
          });
        });
    for (size_t camera = 0; camera < image_topics.size(); ++camera) {
      images_.push_back(create_subscription<sensor_msgs::msg::Image>(
          image_topics[camera], image_qos,
          [this, camera](sensor_msgs::msg::Image::ConstSharedPtr msg) {
            guard([&] {
              if (!core_->ready())
                return;
              auto image = cv_bridge::toCvCopy(msg, "mono8");
              const auto frame = core_->image(
                  image->image, rclcpp::Time(msg->header.stamp).seconds(),
                  camera);
              sensor_msgs::msg::PointCloud cloud;
              cloud.header = msg->header; // original image stamp, for the
                                          // viewer's overlay association
              cloud.channels.resize(2);
              cloud.channels[0].name = "id";
              cloud.channels[1].name = "camera";
              for (const auto &point : frame.pts) {
                geometry_msgs::msg::Point32 pixel;
                pixel.x = point.u;
                pixel.y = point.v;
                pixel.z = 0;
                cloud.points.push_back(pixel);
                cloud.channels[0].values.push_back(float(point.id));
                cloud.channels[1].values.push_back(float(camera));
              }
              tracks_->publish(cloud);
            });
          }));
    }
    RCLCPP_INFO(get_logger(),
                "Waiting for CameraInfo. Frontend: default TrackKLT (CPU). "
                "Temperature unavailable; not required.");
  }
  bool close() { return core_->close() && !failed_; }

private:
  template <class F> void guard(F &&function) {
    if (failed_)
      return;
    try {
      function();
    } catch (const std::exception &error) {
      failed_ = true;
      RCLCPP_ERROR(get_logger(), "%s", error.what());
      rclcpp::shutdown();
    }
  }
  bool failed_ = false;
  std::unique_ptr<ov_zcalib::CaptureCore> core_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr>
      infos_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> images_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr tracks_;
};

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  rclcpp::init(argc, argv);
  int result = 1;
  try {
    auto node = std::make_shared<CaptureNode>();
    rclcpp::spin(node);
    result = node->close() ? 0 : 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[capture] %s\n", error.what());
  }
  if (rclcpp::ok())
    rclcpp::shutdown();
  return result;
}
