#ifndef OV_ZCALIB_ATOMIC_FILE_H
#define OV_ZCALIB_ATOMIC_FILE_H

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

namespace ov_zcalib {

/// Publish only a completely written, flushed, synced and closed temporary
/// file. All failures before rename preserve the current destination; errno
/// reports the first failure even when cleanup itself fails. An optional
/// rollback hard link preserves the old inode without moving the live path.
inline bool finish_atomic_file(FILE *file, const std::string &temporary_path, const std::string &destination,
                               const std::string &rollback_path = {}) {
  int error = std::ferror(file) ? (errno ? errno : EIO) : 0;
  if (std::fflush(file) != 0 && error == 0)
    error = errno ? errno : EIO;
  if (std::ferror(file) && error == 0)
    error = errno ? errno : EIO;
  if (error == 0 && ::fsync(::fileno(file)) != 0)
    error = errno ? errno : EIO;
  if (std::fclose(file) != 0 && error == 0)
    error = errno ? errno : EIO;

  if (error == 0 && !rollback_path.empty()) {
    const std::string backup_tmp = rollback_path + ".tmp";
    if (::unlink(backup_tmp.c_str()) != 0 && errno != ENOENT)
      error = errno;
    if (error == 0) {
      if (::link(destination.c_str(), backup_tmp.c_str()) == 0) {
        if (std::rename(backup_tmp.c_str(), rollback_path.c_str()) != 0) {
          error = errno;
          ::unlink(backup_tmp.c_str());
        }
      } else if (errno != ENOENT) {
        error = errno;
      }
    }
  }
  if (error == 0 && std::rename(temporary_path.c_str(), destination.c_str()) == 0)
    return true;
  if (error == 0)
    error = errno ? errno : EIO;
  ::unlink(temporary_path.c_str());
  errno = error;
  return false;
}

/// Withdraw a candidate whose required metadata could not be published.
/// had_previous must be captured BEFORE publication: a stale rollback file
/// must not resurrect a result the operator deliberately removed.
inline bool restore_atomic_file_backup(const std::string &destination, bool had_previous, std::string *error = nullptr) {
  if (error)
    error->clear();
  const std::string backup = destination + ".rollback";
  const int result = had_previous ? std::rename(backup.c_str(), destination.c_str()) : ::unlink(destination.c_str());
  if (result == 0 || (!had_previous && errno == ENOENT))
    return true;
  const int saved_errno = errno ? errno : EIO;
  if (error)
    *error = (had_previous ? "could not restore previous calibration from " + backup
                          : "could not remove incomplete calibration " + destination) + ": " + std::strerror(saved_errno);
  errno = saved_errno;
  return false;
}

} // namespace ov_zcalib

#endif // OV_ZCALIB_ATOMIC_FILE_H
