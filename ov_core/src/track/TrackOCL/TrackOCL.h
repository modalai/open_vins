/**
 * @file TrackOCL.h
 * @brief GPU (OpenCL) FAST + pyramidal-KLT feature tracker for VOXL OpenVINS: stereo-gated
 *        detection, ZNCC epipolar stereo matching, and IMU-aided KLT seeding.
 * @author kyletyni
 */
#ifndef OV_CORE_TRACK_OCL_H
#define OV_CORE_TRACK_OCL_H

#include "../TrackBase.h"
#include "cam/CamBase.h"
#include "cam/CamEqui.h"
#include <modal_flow_ocl_manager.h>
#include <modal_flow/ocl/OclDevice.hpp>
#include <modal_flow/ocl/ManagerCL.hpp>
#include <modal_flow/StereoMatcher.hpp>
#include <modal_flow/Types.hpp>
#include <modal_flow/Tracker.hpp>
#include "ImuRotationIntegrator.h"
#include <unordered_map>

namespace ov_core
{

  /**
   * @brief Leveraging OpenCL + GPU to perform KLT tracking of features.
   */
  class TrackOCL : public TrackBase
  {

  public:
    /**
     * @brief Public constructor with configuration variables
     * @param cameras camera calibration object which has all camera intrinsics in it
     * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
     * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
     * @param stereo if we should do stereo feature tracking or binocular
     * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
     * @param fast_threshold FAST detection threshold
     * @param gridx size of grid in the x-direction / u-direction
     * @param gridy size of grid in the y-direction / v-direction
     * @param minpxdist features need to be at least this number pixels away from each other
     */
    explicit TrackOCL(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                      int fast_threshold, int gridx, int gridy, int minpxdist)
        : TrackBase(cameras, numfeats, numaruco, stereo, NONE),
          threshold(fast_threshold),
          grid_x(gridx),
          grid_y(gridy),
          min_px_dist(minpxdist),
          dev_(modal_flow::ocl::OclDevice::Instance()),
          mgr_(dev_)
    {
      if (cameras.empty() || !cameras.at(0))
      {
        throw std::runtime_error("Invalid camera data");
      }

      // Create and set detector
      auto det = std::make_unique<modal_flow::ocl::DetectorCL>(dev_, 3);
      mgr_.set_detector(std::move(det));

      // create and set tracker
      auto trk = std::make_unique<modal_flow::ocl::TrackerCL>(dev_);
      mgr_.set_tracker(std::move(trk));
      int num_bufs = 4;

      for (auto const &[camId, camPtr] : cameras)
      {
        //  assumes track input frames will be uint8_t grayscale
        modal_flow::Camera cam{.id = camId, .width = camPtr->w(), .height = camPtr->h(), .format = modal_flow::PixelFormat::R8};
        // Carry intrinsics so the tracker can do IMU-predicted nextPts seeding. Equidistant
        // fisheye (CamEqui) -> Fisheye4; other models leave intrinsics invalid (identity seed).
        if (auto eq = std::dynamic_pointer_cast<CamEqui>(camPtr))
        {
          Eigen::MatrixXd v = eq->get_value(); // [fx fy cx cy k1 k2 k3 k4]
          cam.intrinsics.valid = true;
          cam.intrinsics.model = modal_flow::DistortionModel::Fisheye4;
          cam.intrinsics.fx = (float)v(0); cam.intrinsics.fy = (float)v(1);
          cam.intrinsics.cx = (float)v(2); cam.intrinsics.cy = (float)v(3);
          cam.intrinsics.d0 = (float)v(4); cam.intrinsics.d1 = (float)v(5);
          cam.intrinsics.d2 = (float)v(6); cam.intrinsics.d3 = (float)v(7);
        }
        mgr_.add_camera(cam, num_bufs);
      }

      // Retrieve width and height
      int width = cameras.at(0)->w();
      int height = cameras.at(0)->h();
    }

    /**
     * @brief Process a new image
     * @param message Contains our timestamp, images, and camera ids
     */
    void feed_new_camera(const CameraData &message) override;

    /**
     * @brief set pyramid levels
     */
    void set_pyramid_levels(int levels) { pyr_levels = levels; };

    // --- IMU-aided KLT seeding (rotation prediction) -------------------------------------------
    // All optional and safe-by-default: until feed_imu() AND set_cam_imu_rotation() are supplied,
    // the tracker uses identity seeding (nextPts == prevPts, original behavior).
    //
    // feed_imu: one raw gyro sample (rad/s, IMU/body frame) at time t (SECONDS, same clock as the
    //   image timestamps). Fed from the server's IMU callback via the TrackBase interface.
    void feed_imu(double t, double gx, double gy, double gz) override { imu_rot_.add_gyro(t, gx, gy, gz); }
    // Slowly-varying gyro bias (rad/s), e.g. state->_imu->bias_g(). Optional; sub-pixel effect.
    void set_gyro_bias(double bx, double by, double bz) override { imu_rot_.set_bias(bx, by, bz); }
    // R_ItoC: rotation mapping an IMU-frame vector into camera `cam_id`'s frame
    // (OpenVINS state->_calib_IMUtoCAM[cam_id]->Rot()). Required for delta_q to be non-identity.
    void set_cam_imu_rotation(size_t cam_id, const Eigen::Matrix3d &R_ItoC) override { R_ItoC_[cam_id] = R_ItoC; }

    /**
     * @brief Enable the stereo matcher used at perform_detection_stereo time. 
     *        Must be called BEFORE the first feed_new_camera() invocation -- 
     *        the calibration is uploaded to the GPU once at this point and 
     *        reused for every frame thereafter.
     *
     *        z_min / z_max bound the depth-sweep for the epipolar search.
     *        Defaults (0.10 m / 100 m) cover the disparity range of a 68 mm
     *        baseline stereo rig from near-touch out to effectively infinity;
     *        tighten z_min if you know the scene is farther.
     */
    void enable_zncc_stereo_matcher(const modal_flow::StereoCalib &calib,
                                    float z_min = 0.10f, float z_max = 100.0f);

    /**
     * @brief Per-feature stereo-match confidence stashed when the ZNCC matcher
     *        accepts a stereo pair. Populated only when the matcher is in use;
     *        consumed by Phase-5 EKF measurement-noise weighting. StereoConfidence
     *        is defined on TrackBase so consumers reach it via the base interface.
     */
    const std::unordered_map<size_t, StereoConfidence>& stereo_confidence_map() const override {
        return stereo_confidence_;
    }

  protected:
    /**
     * @brief Process a new monocular image
     * @param message Contains our timestamp, images, and camera ids
     * @param msg_id the camera index in message data vector
     */
    void feed_monocular(const CameraData &message, size_t msg_id);
      
    /**
     * @brief Process new stereo pair of images
     * @param message Contains our timestamp, images, and camera ids
     * @param msg_id_left first image index in message data vector
     * @param msg_id_right second image index in message data vector
     */
    void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);


    void perform_detection_monocular(modal_flow::BufferId& buf_id, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                     std::vector<size_t> &ids0, int id);

    void perform_detection_stereo(modal_flow::BufferId buf_id0, modal_flow::BufferId buf_id1, 
                                  const cv::Mat &mask0, const cv::Mat &mask1,
                                  size_t cam_id_left, size_t cam_id_right,
                                  std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1,
                                  std::vector<size_t> &ids0, std::vector<size_t> &ids1);

    /**
     * @brief KLT track between two images, and do RANSAC afterwards
     * @param img0pyr starting image pyramid
     * @param img1pyr image pyramid we want to track too
     * @param pts0 starting points
     * @param pts1 points we have tracked
     * @param id0 id of the first camera
     * @param id1 id of the second camera
     * @param mask_out what points had valid tracks
     *
     * This will track features from the first image into the second image.
     * The two point vectors will be of equal size, but the mask_out variable will specify which points are good or bad.
     * If the second vector is non-empty, it will be used as an initial guess of where the keypoints are in the second image.
     */
    void perform_matching(modal_flow::BufferId buf0, modal_flow::BufferId buf1,
                          std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1,
                          size_t id0, size_t id1, std::vector<uchar> &mask_out,
                          const modal_flow::RotationQuat &delta_q = {});

    // Camera-frame relative rotation for `cam_id` over [t_prev, t_curr], from the gyro buffer
    // conjugated by R_ItoC. Returns identity (== no prediction) if IMU/extrinsic aren't available.
    modal_flow::RotationQuat delta_q_for_cam(size_t cam_id, double t_prev, double t_curr);

    // IMU-aided seeding state (see feed_imu / set_cam_imu_rotation). Safe defaults => identity.
    mf_imu::ImuRotationIntegrator imu_rot_{4096};
    std::unordered_map<size_t, Eigen::Matrix3d> R_ItoC_;      // IMU->camera rotation per cam
    std::unordered_map<size_t, double> last_cam_time_;        // previous frame time per cam (s)

    // Parameters for our FAST grid detector
    int threshold;
    int grid_x;
    int grid_y;

    // Minimum pixel distance to be "far away enough" to be a different extracted feature
    int min_px_dist;

    // How many pyramid levels to track
    int pyr_levels = 5;
    cv::Size win_size = cv::Size(15, 15);

    // Last set of image pyramids
    std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;
    std::map<size_t, cv::Mat> img_curr;
    std::map<size_t, std::vector<cv::Mat>> img_pyramid_curr;

  private:

    modal_flow::ocl::OclDevice &dev_;
    modal_flow::ocl::ManagerCL mgr_;

    std::map<size_t, modal_flow::BufferId> img_buf_prev_;
    std::map<size_t, modal_flow::BufferId> img_buf_next_;

    // Cam IDs the ZNCC stereo matcher is bound to. Filled in by
    // enable_zncc_stereo_matcher; perform_detection_stereo gates on incoming
    // cam_id_left/right matching these (a safety check, no-ops if the matcher
    // was never enabled since both default to 0).
    size_t                                     stereo_cam_id_left_  = 0;
    size_t                                     stereo_cam_id_right_ = 0;

    // Static left camera model used to undistort left features into bearings 
    // for the ZNCC matcher. Built once in enable_zncc_stereo_matcher from the
    // the static /etc/modalai conf + opencv_tracking_left_intrinsics.yml seed
    // so that the stereo correspondence search never drifts with online 
    // intrinsic calibration -- the matcher's epipolar projection is likewise 
    // static, so both sides of the match stay on seed calib.
    // NOTE: deliberately NOT camera_calib.at(left), whose values get 
    // overwritten by the EKF when do_calib_camera_intrinsics is enabled.
    std::shared_ptr<CamBase>                 stereo_static_cam_left_;

    // Right-camera model + left->right extrinsics, mirrored from the StereoCalib
    // in enable_zncc_stereo_matcher. Used ONLY by the per-feature epipolar dump
    // (dump_stereo_epipolar_) to replicate the matcher kernel's right projection
    // on the host so we can log where the epipolar curve endpoints (z_max/z_min)
    // land and whether they fall out of bounds. CamEqui::distort_f is byte-for-
    // byte the same equidistant model the kernel uses.
    std::shared_ptr<CamBase>                 stereo_static_cam_right_;
    Eigen::Matrix3f                          stereo_R_lr_ = Eigen::Matrix3f::Identity();
    Eigen::Vector3f                          stereo_t_lr_ = Eigen::Vector3f::Zero();
    float                                    stereo_z_min_ = 0.4f;
    float                                    stereo_z_max_ = 100.0f;

    // Per-feature stereo-match confidence (peak_zncc, margin, lr_err) keyed by
    // feature id.
    std::unordered_map<size_t, StereoConfidence> stereo_confidence_;

    // DRIFT-CORRECTION state. Per-feature inverse-depth (1/Z) prior, keyed by feature id, seeded
    // when a pair is first matched (detection) and refreshed on each successful correction.
    // feed_stereo KLT-tracks BOTH cameras (so stereo pairs persist), then runs a narrow ZNCC
    // re-match [prior +/- stereo_rho_half_width_] + L-R check as a CORRECTOR: a confident match that
    // disagrees with the KLT right point snaps it to the matcher (kills drift); a rejected match
    // does nothing (KLT keeps the pair alive). The match is never a per-frame survival gate -- that
    // starves stereo, since the strict gates only need to be cleared once, at detection.
    std::unordered_map<size_t, float> stereo_inv_depth_prior_;
    // Inverse-depth half-window for the narrow re-match, = kRematchBandPx / (fx_left * baseline).
    float stereo_rho_half_width_ = 0.f;

    // DIAGNOSTIC: count of mono-left -> stereo upgrades accepted in the most
    // recent perform_detection_stereo call (ZNCC "promote" pass). Read by
    // feed_stereo's per-frame track-stats log.
    int last_n_promoted_ = 0;

    // ---- Stereo match reject diagnostics ----------------------------------
    // Attributes every submitted stereo candidate to the gate that rejected it
    // (weak peak / low margin / L-R / reverse / runner ceiling) plus near-miss
    // counts (would the match pass if that one gate were loosened?). Pure
    // accounting -- no effect on matching behavior. Flip stereo_diag_on_ to
    // false to silence; stereo_diag_period_ is the print cadence in detect calls.
    struct StereoRejectStats {
      // index 0 = top-off (fresh corners, L->R), 1 = mono-left->stereo promote (L->R),
      // 2 = mono-right->stereo promote (REVERSE, R->L). For pass 2 the "right_oob"
      // slot counts TARGET (left-image) out-of-bounds matches.
      long submitted[3]{}, accepted[3]{};
      long oob[3]{}, weak_peak[3]{}, low_margin[3]{}, lr_fail[3]{},
           reverse_fail[3]{}, runner_ceiling[3]{}, right_oob[3]{};
      long nm_peak[3]{}, nm_margin[3]{}, nm_lr[3]{}, nm_runner[3]{}; // near-miss
      long ndist[3]{};
      double peak_sum[3]{}, marg_sum[3]{}, lr_sum[3]{};
      float peak_min[3]{1e9f, 1e9f, 1e9f},  peak_max[3]{-1e9f, -1e9f, -1e9f};
      float marg_min[3]{1e9f, 1e9f, 1e9f},  marg_max[3]{-1e9f, -1e9f, -1e9f};
      float lr_min[3]{1e9f, 1e9f, 1e9f},    lr_max[3]{-1e9f, -1e9f, -1e9f};
    };
    bool  stereo_diag_on_     = false; // OFF by default for now; set true to write the /run diag logs
    int   stereo_diag_period_ = 30;   // print every N perform_detection_stereo calls (~1/s @30Hz)
    int   stereo_diag_calls_  = 0;
    StereoRejectStats stereo_reject_stats_{};
    // Threshold mirrors -- set in enable_zncc_stereo_matcher to the SAME values
    // pushed to the matcher, so the diagnostic attribution never drifts from the
    // live gates.
    float stereo_zncc_min_   = 0.70f;
    float stereo_margin_min_ = 0.30f;
    float stereo_lr_thresh_  = 3.0f;

    void accumulate_stereo_reject_(int pass, float peak, float margin, float lr,
                                   bool matcher_status, bool right_oob);
    void maybe_print_stereo_diag_();

    // Drift-correction diagnostics (feed_stereo). Rolling counts over stereo_diag_period_ frames:
    // pairs submitted to the corrector, matcher-accepted, and actually snapped (KLT/matcher
    // disagreement > kCorrectSnapPx), plus snap-distance stats. Confirms the corrector is live.
    long   corr_sub_ = 0, corr_acc_ = 0, corr_snap_ = 0;
    double corr_snap_sum_ = 0.0;
    float  corr_snap_max_ = 0.f;
    int    corr_diag_frames_ = 0;

    // IMU-aided seeding diagnostic (feed_stereo). Every stereo_diag_period_ frames prints, for the
    // left camera: gyro-buffer liveness, the predicted rotation magnitude, and how much of the
    // inter-frame feature motion the IMU seed removed (reduction% ~ 0 => not helping/identity;
    // high under rotation => working). Written to /run/voxl-open-vins-imu-seed.log (+ stdout).
    int    imu_diag_frames_ = 0;

    // Per-feature epipolar dump: for a sampled subset of features, log the left
    // point, the right projection at z_max (far) and z_min (near), in/out-of-bounds
    // flags, and the matcher's best match + peak/margin/lr. Distinguishes FOV/
    // calibration OOB (off-image even at z_max) from z-range OOB (off only near).
    bool stereo_dump_on_         = false;   // per-feature [STEREO DUMP] epipolar log (noisy; off)
    int  stereo_dump_per_window_ = 8;   // features sampled per dump
    void dump_stereo_epipolar_(const modal_flow::StereoMatchInput &in,
                               const modal_flow::StereoMatchResult &res,
                               int img_w1, int img_h1);
  };

} // namespace ov_core

#endif /* OV_CORE_TRACK_KLT_H */
