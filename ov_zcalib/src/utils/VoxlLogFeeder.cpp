/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: voxl-logger feeder implementation (see VoxlLogFeeder.h).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "VoxlLogFeeder.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "track/TrackKLT.h"

using namespace ov_zcalib;

namespace {
struct CsvImu {
  double t;
  Eigen::Vector3d w, a;
  double temp;
};
struct CsvFrame {
  double t;
  double exposure_s;
  int idx;
};
} // namespace

bool VoxlLogFeeder::load_seed_intrinsics(const std::string &log_dir, const std::string &cam_pipe, SessionSeed &seed) {
  // pipe name -> per-unit cal file base: strip the stream suffixes voxl-camera-server appends
  std::string base = cam_pipe;
  for (const char *suf : {"_misp_grey", "_misp_norm_ion", "_misp_norm", "_grey", "_ion"}) {
    const size_t n = base.size(), m = std::string(suf).size();
    if (n > m && base.compare(n - m, m, suf) == 0) {
      base = base.substr(0, n - m);
      break;
    }
  }
  const std::string yml = log_dir + "/data/modalai/opencv_" + base + "_intrinsics.yml";
  cv::FileStorage fs(yml, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::printf("[voxl] cannot open per-unit intrinsics %s\n", yml.c_str());
    return false;
  }
  cv::Mat M, D;
  fs["M"] >> M;
  fs["D"] >> D;
  std::string model;
  fs["distortion_model"] >> model;
  int w = 0, h = 0;
  fs["width"] >> w;
  fs["height"] >> h;
  if (M.rows != 3 || D.total() < 4 || w <= 0 || h <= 0) {
    std::printf("[voxl] malformed intrinsics yml %s\n", yml.c_str());
    return false;
  }
  // A voxl log carries ONE camera stream, so this feeder seeds camera 0 and nothing else.
  CamCalib &kc = seed.calib.cams[0];
  kc.cam << M.at<double>(0, 0), M.at<double>(1, 1), M.at<double>(0, 2), M.at<double>(1, 2), D.at<double>(0), D.at<double>(1),
      D.at<double>(2), D.at<double>(3);
  kc.fisheye = (model == "fisheye" || model == "equidistant");
  kc.img_w = w;
  kc.img_h = h;
  double reproj = -1.0;
  fs["reprojection_error"] >> reproj;
  std::printf("[voxl] per-unit intrinsics %s: fx=%.2f fy=%.2f cx=%.2f cy=%.2f %s (cal reproj %.3f px)\n", base.c_str(), kc.cam(0),
              kc.cam(1), kc.cam(2), kc.cam(3), kc.fisheye ? "fisheye" : "radtan", reproj);
  return true;
}

bool VoxlLogFeeder::convert(const std::string &log_dir, const std::string &cam_pipe, const SessionSeed &seed,
                            const std::string &record_out, double max_seconds, int num_feats, double *mean_exposure_s) {
  // ---- IMU csv: i, timestamp(ns), batch_id, AX AY AZ (m/s2), GX GY GZ (rad/s), T(C) ----
  std::vector<CsvImu> imu;
  {
    std::ifstream f(log_dir + "/run/mpa/imu_apps/data.csv");
    if (!f.is_open()) {
      std::printf("[voxl] cannot open %s/run/mpa/imu_apps/data.csv\n", log_dir.c_str());
      return false;
    }
    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
      if (header) {
        header = false;
        continue;
      }
      std::stringstream ss(line);
      std::string tok;
      double v[10];
      int i = 0;
      while (std::getline(ss, tok, ',') && i < 10)
        v[i++] = std::atof(tok.c_str());
      if (i < 10)
        continue;
      CsvImu s;
      s.t = v[1] * 1e-9;
      s.a = Eigen::Vector3d(v[3], v[4], v[5]);
      s.w = Eigen::Vector3d(v[6], v[7], v[8]);
      s.temp = v[9];
      imu.push_back(s);
    }
  }
  // ---- camera csv: i, timestamp(ns), gain, exposure(ns), format, height, width, frame_id, reserved ----
  std::vector<CsvFrame> cams;
  {
    std::ifstream f(log_dir + "/run/mpa/" + cam_pipe + "/data.csv");
    if (!f.is_open()) {
      std::printf("[voxl] cannot open %s/run/mpa/%s/data.csv\n", log_dir.c_str(), cam_pipe.c_str());
      return false;
    }
    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
      if (header) {
        header = false;
        continue;
      }
      std::stringstream ss(line);
      std::string tok;
      double v[9];
      int i = 0;
      while (std::getline(ss, tok, ',') && i < 9)
        v[i++] = std::atof(tok.c_str());
      if (i < 4)
        continue;
      CsvFrame c;
      c.idx = (int)v[0];
      c.t = v[1] * 1e-9; // start-of-exposure stamp (MPA camera convention)
      c.exposure_s = v[3] * 1e-9;
      cams.push_back(c);
    }
  }
  if (imu.size() < 100 || cams.size() < 10) {
    std::printf("[voxl] too little data (%zu imu, %zu frames)\n", imu.size(), cams.size());
    return false;
  }
  const double t0 = imu.front().t;
  const double t_end = (max_seconds > 0.0) ? t0 + max_seconds : 1e300;

  // ---- KLT at the seed intrinsics (fisheye-aware; the estimator-grade models) ----
  const CamCalib &kc0 = seed.calib.cams[0];
  Eigen::MatrixXd camv(8, 1);
  for (int i = 0; i < 8; ++i)
    camv(i) = kc0.cam(i);
  std::shared_ptr<ov_core::CamBase> cam;
  if (kc0.fisheye)
    cam = std::make_shared<ov_core::CamEqui>(kc0.img_w, kc0.img_h);
  else
    cam = std::make_shared<ov_core::CamRadtan>(kc0.img_w, kc0.img_h);
  cam->set_value(camv);
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cam_map;
  cam_map[0] = cam;
  ov_core::TrackKLT klt(cam_map, num_feats, 0, false, ov_core::TrackBase::HistogramMethod::HISTOGRAM, 15, 8, 6, 10);

  SessionRecordWriter wr;
  if (!wr.open(record_out, seed))
    return false;
  size_t ii = 0;
  uint32_t seq = 0;
  int written = 0;
  double exp_sum = 0.0;
  char png[64];
  // PRODUCER-SIDE stamp convention (identical to the server's camera ingest): the logged MPA
  // stamp is HAL3 start-of-exposure of the FIRST row; the system convention anchors every frame
  // at its center-row mid-exposure instant, t = SOF + (readout + exposure)/2. readout is the
  // fixed HAL3 value carried in the seed (--tr on the CLI; 0 = global shutter).
  const double tr_hw = seed.calib.cams[0].tr;
  for (const auto &cf : cams) {
    if (cf.t > t_end)
      break;
    while (ii < imu.size() && imu[ii].t <= cf.t + 0.02) {
      RawImu s;
      s.timestamp = imu[ii].t - t0;
      s.wm = imu[ii].w;
      s.am = imu[ii].a;
      s.temp_c = imu[ii].temp;
      wr.write_imu(s);
      ++ii;
    }
    std::snprintf(png, sizeof(png), "/%05d.png", cf.idx);
    cv::Mat img = cv::imread(log_dir + "/run/mpa/" + cam_pipe + png, cv::IMREAD_GRAYSCALE);
    if (img.empty())
      continue;
    const double t_center = cf.t - t0 + 0.5 * (tr_hw + cf.exposure_s);
    ov_core::CameraData cd;
    cd.timestamp = t_center;
    cd.sensor_ids.push_back(0);
    cd.images.push_back(img);
    cd.masks.push_back(cv::Mat::zeros(img.rows, img.cols, CV_8UC1));
    klt.feed_new_camera(cd);
    FrameObs fo;
    fo.timestamp = t_center;
    fo.exposure_s = (float)cf.exposure_s;
    fo.temp_c = 0.f;
    fo.seq = seq++;
    auto obs = klt.get_last_obs();
    auto ids = klt.get_last_ids();
    const auto &kps = obs[0];
    const auto &kid = ids[0];
    for (size_t i = 0; i < kps.size() && i < kid.size(); ++i) {
      FrameObsPoint p;
      p.id = (uint32_t)kid[i];
      p.u = kps[i].pt.x;
      p.v = kps[i].pt.y;
      fo.pts.push_back(p);
    }
    wr.write_frame(fo);
    exp_sum += cf.exposure_s;
    ++written;
  }
  wr.close();
  if (mean_exposure_s)
    *mean_exposure_s = (written > 0) ? exp_sum / written : 0.0;
  std::printf("[voxl] wrote %d frames / %.1f s to %s (mean exposure %.3f ms)\n", written, (cams.back().t - t0), record_out.c_str(),
              (written > 0) ? 1e3 * exp_sum / written : 0.0);
  return written > 10;
}

bool VoxlLogFeeder::load_ref_extrinsics(const std::string &log_dir, const std::string &cam_pipe, RefExtrinsics &out) {
  // voxl-logger snapshots /etc/modalai into the log; the conf is modal_json
  // (JSON + C comments). The entries are flat objects, so a comment-strip +
  // per-object key scan is exact enough -- this feeds the evaluation REFERENCE
  // print only, never the estimator.
  std::ifstream f(log_dir + "/etc/modalai/extrinsics.conf");
  if (!f.is_open())
    return false;
  std::stringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();
  // strip /* ... */ and // ... comments
  for (size_t p = s.find("/*"); p != std::string::npos; p = s.find("/*")) {
    const size_t e = s.find("*/", p + 2);
    s.erase(p, (e == std::string::npos) ? std::string::npos : e + 2 - p);
  }
  for (size_t p = s.find("//"); p != std::string::npos; p = s.find("//", p)) {
    const size_t e = s.find('\n', p);
    s.erase(p, (e == std::string::npos) ? std::string::npos : e - p);
  }
  auto get_str = [](const std::string &blk, const char *key) {
    const size_t k = blk.find("\"" + std::string(key) + "\"");
    if (k == std::string::npos)
      return std::string();
    const size_t q0 = blk.find('"', blk.find(':', k) + 1);
    const size_t q1 = (q0 == std::string::npos) ? std::string::npos : blk.find('"', q0 + 1);
    return (q1 == std::string::npos) ? std::string() : blk.substr(q0 + 1, q1 - q0 - 1);
  };
  auto get_vec3 = [](const std::string &blk, const char *key, double v[3]) {
    const size_t k = blk.find("\"" + std::string(key) + "\"");
    if (k == std::string::npos)
      return false;
    const size_t b0 = blk.find('[', k);
    const size_t b1 = (b0 == std::string::npos) ? std::string::npos : blk.find(']', b0);
    if (b1 == std::string::npos)
      return false;
    std::string body = blk.substr(b0 + 1, b1 - b0 - 1);
    std::replace(body.begin(), body.end(), ',', ' ');
    std::istringstream is(body);
    return bool(is >> v[0] >> v[1] >> v[2]);
  };
  size_t best_child = 0; // longest conf child that prefixes cam_pipe wins
  for (size_t p = s.find('{'); p != std::string::npos; p = s.find('{', p + 1)) {
    const size_t e = s.find('}', p);
    if (e == std::string::npos)
      break;
    const std::string blk = s.substr(p, e - p);
    const std::string parent = get_str(blk, "parent"), child = get_str(blk, "child");
    if (parent.empty() || child.empty())
      continue;
    if (parent == "imu_apps" && child.size() > best_child && cam_pipe.rfind(child, 0) == 0) {
      if (get_vec3(blk, "RPY_parent_to_child", out.cam_rpy) && get_vec3(blk, "T_child_wrt_parent", out.cam_t)) {
        out.have_cam = true;
        best_child = child.size();
      }
    } else if (parent == "body" && child == "imu_apps") {
      double t_unused[3];
      (void)t_unused;
      if (get_vec3(blk, "RPY_parent_to_child", out.body_rpy))
        out.have_body = true;
    }
  }
  return out.have_cam || out.have_body;
}
