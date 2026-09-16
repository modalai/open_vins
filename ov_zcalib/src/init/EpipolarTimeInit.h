/* OpenVINS ov_zcalib — geometric camera/IMU bootstrap. GPL-3.0-or-later. */
#ifndef OV_ZCALIB_EPIPOLAR_TIME_INIT_H
#define OV_ZCALIB_EPIPOLAR_TIME_INIT_H

#include <deque>
#include "HandEyeWahba.h"
#include "../utils/SessionRecord.h"

namespace ov_zcalib {

struct EpipolarTimeResult {
  bool attempted = false, ok = false;
  double td = 0.0, seed_td = 0.0;
  double split_delta = -1.0, sigma_td = -1.0;
  double cost_before = 0.0, cost_after = 0.0, wall_s = 0.0;
  int pairs = 0;
  bool has_support() const { return pairs >= 40; }
  Eigen::Vector4d q_ItoC = Eigen::Vector4d(0, 0, 0, 1);
};

// Refine an unreliable angular-speed correlation using matched bearings.
// A gyro orientation prefix permits continuous timestamp queries without
// re-integrating the IMU for every trial. Per-pair translation directions are
// eliminated by 3x3 eigensolves; only R_ItoC and td are optimized. Bearings are
// rotated from their rolling-shutter row times to their frame centers.
// This is a bootstrap, not a replacement for the final visual-inertial solve.
// Local constant-translation-velocity and fixed gyro-bias assumptions are
// checked by fitting two non-overlapping time intervals independently.
class EpipolarTimeInit {
public:
  static EpipolarTimeResult solve(const std::deque<FrameObs> &frames, const std::vector<RawImu> &imu,
                                  const CamCalib &camera, int cam_id, const HandEyeResult &seed, double search_s,
                                  double split_tol_s);
};

} // namespace ov_zcalib
#endif
