/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_core: the three filesystem operations the built targets actually used from
 * boost::filesystem (exists / remove / create parent directories), as thin POSIX wrappers.
 * Not std::filesystem: the qrb5165 1.x cross toolchain (gcc 7 era, 18.04 sysroot) predates
 * a complete <filesystem>, and these paths are debug/record plumbing -- three syscalls do
 * not justify a library dependency on the flight target.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_CORE_FS_LITE_H
#define OV_CORE_FS_LITE_H

#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

namespace ov_core {

/// true when the path names an existing file or directory
inline bool fs_exists(const std::string &path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}

/// remove a regular file (best-effort; false when it did not exist or unlink failed)
inline bool fs_remove(const std::string &path) { return ::remove(path.c_str()) == 0; }

/// mkdir -p of the DIRECTORY PART of a file path (everything before the last '/'):
/// the boost::filesystem::create_directories(p.parent_path()) idiom of the record paths
inline void fs_create_parent_dirs(const std::string &file_path) {
  size_t last = file_path.find_last_of('/');
  if (last == std::string::npos)
    return; // bare filename in CWD: nothing to create
  std::string dir = file_path.substr(0, last);
  std::string partial;
  size_t pos = 0;
  while (pos != std::string::npos) {
    pos = dir.find('/', pos + 1);
    partial = dir.substr(0, pos);
    if (!partial.empty())
      ::mkdir(partial.c_str(), 0777); // EEXIST is fine; deeper failures surface at fopen
  }
}

} // namespace ov_core

#endif // OV_CORE_FS_LITE_H
