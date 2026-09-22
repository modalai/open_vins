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
      const bool synchronized = cameras > 1 && (stereo || force == 1);
      const bool expected = declaration >= 0 ? declaration != 0 : cameras > 1 && !synchronized;
      if (options.epoch_mode != expected || !parser->successful()) ++failures;
      if (options.use_stereo != stereo || options.force_camera_sync != (force == 1) ||
          options.synchronize_camera_timestamps() != synchronized ||
          options.use_epoch_clones() != (expected && !synchronized)) ++failures;
      options.async_frame_clones = true;
      if (options.use_async_frame_clones() != (cameras > 1 && !synchronized)) ++failures;
      if (!options.state_options.configure_clone_policy(true, synchronized) ||
          options.state_options.max_pose_clones() != options.state_options.max_clone_size *
              (cameras > 1 && !synchronized ? cameras : 1)) ++failures;
      // The server may override stereo association after parsing the file.
      options.use_stereo = !stereo;
      options.resolve_camera_epoch_mode(parser);
      const bool override_expected = declaration >= 0 ? declaration != 0 : cameras > 1 && stereo && force != 1;
      if (options.epoch_mode != override_expected || !parser->successful()) ++failures;
      if (options.use_stereo != !stereo || options.force_camera_sync != (force == 1)) ++failures;
    }
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
