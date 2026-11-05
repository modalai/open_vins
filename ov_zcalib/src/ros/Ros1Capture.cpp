// SPDX-License-Identifier: GPL-3.0-or-later
#include "CaptureCore.h"
#include <csignal>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>

namespace {
volatile std::sig_atomic_t stop_requested = 0;
void stop(int) { stop_requested = 1; }
} // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  ros::init(argc, argv, "ov_zcalib_capture",
            ros::init_options::NoSigintHandler);
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);
  ros::NodeHandle node, params("~");
  int result = 1;
  try {
    std::string path, image_topic, info_topic, imu_topic, stamp;
    double readout, exposure;
    int features;
    params.param<std::string>("record_path", path, "");
    params.param<std::string>("image_topic", image_topic, "/camera/image_raw");
    params.param<std::string>("info_topic", info_topic, "/camera/camera_info");
    params.param<std::string>("imu_topic", imu_topic, "/imu/data_raw");
    params.param<std::string>("timestamp_reference", stamp, "center");
    params.param("readout_s", readout, 0.0);
    params.param("exposure_s", exposure, 0.0);
    params.param("max_features", features, 200);
    if (stamp != "center" && stamp != "start")
      throw std::runtime_error("timestamp_reference must be center or start");
    std::vector<std::string> image_topics{image_topic}, info_topics{info_topic},
        references;
    std::vector<double> readouts, exposures;
    params.getParam("image_topics", image_topics);
    params.getParam("info_topics", info_topics);
    readouts.assign(image_topics.size(), readout);
    exposures.assign(image_topics.size(), exposure);
    references.assign(image_topics.size(), stamp);
    params.getParam("readout_times", readouts);
    params.getParam("exposure_times", exposures);
    params.getParam("timestamp_references", references);
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
    params.param("sigma_w", noise.sigma_w, noise.sigma_w);
    params.param("sigma_wb", noise.sigma_wb, noise.sigma_wb);
    params.param("sigma_a", noise.sigma_a, noise.sigma_a);
    params.param("sigma_ab", noise.sigma_ab, noise.sigma_ab);
    ov_zcalib::CaptureCore core(path, timing, features, noise);
    bool failed = false;
    auto guard = [&](auto &&function) {
      if (failed)
        return;
      try {
        function();
      } catch (const std::exception &error) {
        failed = true;
        stop_requested = 1;
        ROS_ERROR("%s", error.what());
      }
    };
    auto tracks =
        node.advertise<sensor_msgs::PointCloud>("/ov_zcalib/tracks", 5);
    std::vector<ros::Subscriber> infos, images;
    for (size_t camera = 0; camera < image_topics.size(); ++camera) {
      infos.push_back(node.subscribe<sensor_msgs::CameraInfo>(
          info_topics[camera], 5,
          [&, camera](const sensor_msgs::CameraInfo::ConstPtr &msg) {
            guard([&] {
              core.configure(msg->width, msg->height,
                             {msg->K[0], msg->K[4], msg->K[2], msg->K[5]},
                             msg->D, msg->distortion_model, camera);
            });
          }));
    }
    auto imu = node.subscribe<sensor_msgs::Imu>(
        imu_topic, 2000,
        [&](const sensor_msgs::Imu::ConstPtr &msg) {
          guard([&] {
            if (msg->angular_velocity_covariance[0] == -1 ||
                msg->linear_acceleration_covariance[0] == -1)
              throw std::runtime_error("IMU message declares gyro or "
                                       "accelerometer data unavailable");
            ov_zcalib::RawImu sample;
            // Standard Imu messages have no temperature; zero is the existing
            // record-format placeholder for unavailable telemetry, not 0
            // degrees C.
            sample.temp_c = 0.0;
            sample.timestamp = msg->header.stamp.toSec();
            sample.wm << msg->angular_velocity.x, msg->angular_velocity.y,
                msg->angular_velocity.z;
            sample.am << msg->linear_acceleration.x, msg->linear_acceleration.y,
                msg->linear_acceleration.z;
            core.imu(sample);
          });
        },
        ros::TransportHints().tcpNoDelay());
    for (size_t camera = 0; camera < image_topics.size(); ++camera) {
      images.push_back(node.subscribe<sensor_msgs::Image>(
          image_topics[camera], 10,
          [&, camera](const sensor_msgs::Image::ConstPtr &msg) {
            guard([&] {
              if (!core.ready())
                return;
              const auto frame =
                  core.image(cv_bridge::toCvCopy(msg, "mono8")->image,
                             msg->header.stamp.toSec(), camera);
              sensor_msgs::PointCloud cloud;
              cloud.header = msg->header;
              cloud.channels.resize(2);
              cloud.channels[0].name = "id";
              cloud.channels[1].name = "camera";
              for (const auto &point : frame.pts) {
                geometry_msgs::Point32 pixel;
                pixel.x = point.u;
                pixel.y = point.v;
                cloud.points.push_back(pixel);
                cloud.channels[0].values.push_back(float(point.id));
                cloud.channels[1].values.push_back(float(camera));
              }
              tracks.publish(cloud);
            });
          }));
    }
    ROS_INFO("Waiting for CameraInfo. Frontend: default TrackKLT (CPU).");
    ros::WallRate rate(500);
    while (ros::ok() && !stop_requested) {
      ros::spinOnce();
      rate.sleep();
    }
    result = core.close() && !failed ? 0 : 1;
  } catch (const std::exception &error) {
    ROS_ERROR("%s", error.what());
  }
  ros::shutdown();
  return result;
}
