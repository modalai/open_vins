/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstdio>
#include <fstream>
#include <unistd.h>
#include "core/VioManagerOptions.h"

int main() {
  ov_core::Printer::setPrintLevel("ERROR");
  int failures = 0;
  {
    ov_msckf::VioManagerOptions options;
    if (options.use_stereo) ++failures;
    options.use_stereo = options.init_options.use_stereo = true;
    ov_core::Printer::setPrintLevel("WARNING");
    options.print_and_load_trackers();
    ov_core::Printer::setPrintLevel("ERROR");
    if (options.use_stereo || options.init_options.use_stereo) ++failures;
  }
  char path[] = "/tmp/ov-camera-epoch-XXXXXX";
  const int fd = mkstemp(path);
  if (fd < 0) return 1;
  close(fd);
  for (int force : {-1, 0, 1}) for (int declaration : {-1, 0, 1}) {
    {
      std::ofstream yaml(path);
      yaml << "%YAML:1.0\n---\nfixture: 1\n";
      if (declaration >= 0) yaml << "epoch_mode: " << (declaration ? "true" : "false") << "\n";
      if (force >= 0) yaml << "force_camera_sync: " << (force ? "true" : "false") << "\n";
    }
    auto parser = std::make_shared<ov_core::YamlParser>(path);
    for (int cameras : {1, 2, 3}) for (bool stereo : {false, true}) for (bool initial : {false, true}) {
      ov_msckf::VioManagerOptions options;
      options.state_options.num_cameras = cameras;
      options.use_stereo = stereo;
      options.epoch_mode = initial;
      options.resolve_camera_epoch_mode(parser);
      const bool synchronized = cameras > 1 && force == 1;
      const bool expected = declaration >= 0 ? declaration != 0 : cameras > 1 && !synchronized;
      if (options.epoch_mode != expected || !parser->successful()) ++failures;
      if (options.use_stereo || options.init_options.use_stereo || options.force_camera_sync != (force == 1) ||
          options.synchronize_camera_timestamps() != synchronized ||
          options.use_epoch_clones() != (expected && !synchronized)) ++failures;
      options.async_frame_clones = true;
      if (options.use_async_frame_clones() != (cameras > 1 && !synchronized)) ++failures;
      if (!options.state_options.configure_clone_policy(true, synchronized) ||
          options.state_options.max_pose_clones() != options.state_options.max_clone_size *
              (cameras > 1 && !synchronized ? cameras : 1)) ++failures;
      // The server may try to enable stereo after parsing the file; timing stays mono.
      options.use_stereo = !stereo;
      options.resolve_camera_epoch_mode(parser);
      const bool override_expected = declaration >= 0 ? declaration != 0 : cameras > 1 && force != 1;
      if (options.epoch_mode != override_expected || !parser->successful()) ++failures;
      if (options.use_stereo || options.init_options.use_stereo || options.force_camera_sync != (force == 1)) ++failures;
    }
  }
  {
    std::ofstream yaml(path);
    yaml << "%YAML:1.0\n---\nuse_stereo: true\n"
         << "use_klt: true\nuse_gpu: false\nuse_aruco: false\n"
         << "downsize_aruco: true\ndownsample_cameras: false\nnum_opencv_threads: 1\n"
         << "num_pts: 100\nfast_threshold: 20\ngrid_x: 5\ngrid_y: 5\nmin_px_dist: 10\n"
         << "histogram_method: HISTOGRAM\nknn_ratio: 0.85\ntrack_frequency: 20.0\n";
  }
  {
    auto parser = std::make_shared<ov_core::YamlParser>(path);
    ov_msckf::VioManagerOptions options;
    options.state_options.num_cameras = 2;
    options.print_and_load_trackers(parser);
    options.resolve_camera_epoch_mode(parser);
    if (!parser->successful() || options.use_stereo || options.init_options.use_stereo ||
        options.synchronize_camera_timestamps() || !options.use_epoch_clones()) ++failures;
  }
  unlink(path);
  for (bool initial : {false, true}) {
    ov_msckf::VioManagerOptions options;
    options.state_options.num_cameras = 2;
    options.use_stereo = false;
    options.epoch_mode = initial;
    options.resolve_camera_epoch_mode();
    if (options.epoch_mode != initial) ++failures;
  }
  std::printf("CAMERA_EPOCH_OPTIONS %s failures=%d\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
