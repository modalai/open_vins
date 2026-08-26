/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 * Copyright (C) 2025-2026 Kyle Tyni
 * Copyright (C) 2018-2022 Patrick Geneva
 * Copyright (C) 2018-2022 Guoquan Huang
 * Copyright (C) 2018-2022 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
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
 // TrackOCL.cpp -- GPU (OpenCL) FAST + pyramidal-KLT tracker: stereo-gated detection,
 // ZNCC epipolar stereo matching, and IMU-aided KLT seeding.

#include "TrackOCL.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include "../Grider_FAST.h"
#include "../Grider_GRID.h"
#include "Grider_OCL.h"
#include "cam/CamBase.h"
#include "cam/CamEqui.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"
#include <modal_flow/ocl/StereoMatcherCL.hpp>
#include "utils/print.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unistd.h>

using namespace ov_core;

int test_feed_all = 0;

// ============================================================================
// A/B TOGGLE: IMU-aided KLT seeding. Flip to false + rebuild for the baseline
// (prev-position seeding). The IMU rotation is still integrated and the
// [IMU-SEED] diagnostic still logs the predicted rotation + COUNTERFACTUAL
// reduction% either way -- only whether the prediction is applied to the KLT
// seed changes. So: OFF build = baseline tracking with "what prediction would
// do" logged; ON build = prediction applied with realized reduction logged.
// ============================================================================
static constexpr bool kImuAidedSeeding = false;

// ============================================================================
// A/B TOGGLE: stereo front-end scheme. Flip + rebuild to compare empirically.
//   false (Approach A, current): independent temporal KLT in BOTH cameras; the
//     right partner rides its own KLT and a narrow ZNCC only SNAPS it when a
//     confident match disagrees by >kCorrectSnapPx. A lost right-KLT permanently
//     degrades the pair to mono-left ("fails-silent" drift/decay).
//   true (Approach B, re-derivation): KLT the LEFT as the temporal anchor; the
//     right partner is the fresh ZNCC epipolar re-derivation EVERY frame (right-
//     KLT ignored for paired features). On reject the pair yields mono-left this
//     frame (fails-safe) and can RE-ACQUIRE the partner on a later frame even
//     after a right-KLT loss, since the partner never depended on right-KLT.
// Both builds emit the same STEREO EPI / track-stats diagnostics for comparison.
// ============================================================================
static constexpr bool kStereoRederive = false;

namespace {
// DIAGNOSTIC: per-frame feature-track breakdown handed to the feature DB (what the
// VIO sees). Lets us see stereo-vs-mono counts, when tracking thins out / goes
// IMU-only, and how many L->R (stereo) matches actually exist. Single per-process
// file; correlate meas_ts with the ov/data.csv published feat count.
//   STEREO rows: cam=-1, full breakdown (n_stereo = L&R paired, mono_left/right).
//   MONO rows:   cam=<id>, n_total = that camera's tracks (stereo fields 0).
void note_track_stats(double meas_ts, const char *mode, int cam,
                      int n_total, int n_stereo, int n_mono_left, int n_mono_right, int n_promoted)
{
    static std::mutex mtx;
    static FILE *fp = nullptr;
    static bool tried = false;
    std::lock_guard<std::mutex> lk(mtx);
    if (!tried) {
        tried = true;
        fp = fopen("/run/voxl-open-vins-track-stats.log", "w"); // truncate once per process
        if (fp == nullptr) fp = fopen("/tmp/voxl-open-vins-track-stats.log", "w");
        if (fp != nullptr) {
            fprintf(fp, "# voxl-open-vins-server per-frame feature-track stats (pid=%d)\n", (int)getpid());
            fprintf(fp, "# meas_ts_s, mode, cam, n_total, n_stereo, n_mono_left, n_mono_right, n_promoted\n");
            fflush(fp);
        }
    }
    if (fp == nullptr) return;
    fprintf(fp, "%.6f, %s, %d, %d, %d, %d, %d, %d\n",
            meas_ts, mode, cam, n_total, n_stereo, n_mono_left, n_mono_right, n_promoted);
    fflush(fp);
}

// Shared per-process log file for the stereo match diagnostics (rolling reject
// summary + per-feature epipolar dump). Opened once, truncated. Returns nullptr
// if neither /run nor /tmp is writable, in which case the diagnostic is dropped.
FILE *stereo_diag_log()
{
    static std::mutex mtx;
    static FILE      *fp    = nullptr;
    static bool       tried = false;
    std::lock_guard<std::mutex> lk(mtx);
    if (!tried) {
        tried = true;
        fp = fopen("/run/voxl-open-vins-stereo-diag.log", "w");
        if (fp == nullptr) fp = fopen("/tmp/voxl-open-vins-stereo-diag.log", "w");
        if (fp != nullptr)
            fprintf(fp, "# voxl-open-vins-server stereo match diagnostics (pid=%d)\n", (int)getpid());
    }
    return fp;
}

// Dedicated per-process log for the IMU-aided-seeding diagnostic ([IMU-SEED] lines). Kept separate
// from the stereo match log so it can be tailed on its own alongside the other OV diagnostic files
// in /run. Opened once, truncated; nullptr if neither /run nor /tmp is writable (diagnostic dropped).
FILE *imu_seed_log()
{
    static std::mutex mtx;
    static FILE      *fp    = nullptr;
    static bool       tried = false;
    std::lock_guard<std::mutex> lk(mtx);
    if (!tried) {
        tried = true;
        fp = fopen("/run/voxl-open-vins-imu-seed.log", "w");
        if (fp == nullptr) fp = fopen("/tmp/voxl-open-vins-imu-seed.log", "w");
        if (fp != nullptr)
            fprintf(fp, "# voxl-open-vins-server IMU-aided KLT seeding diagnostics (pid=%d)\n", (int)getpid());
    }
    return fp;
}
} // namespace


// Compile-time toggle for the per-frame feature-track stats log
// (/run/voxl-open-vins-track-stats.log via note_track_stats). On by default; flip
// to false to dead-strip the logging (and its per-frame stereo/mono breakdown
// computation) for clean production runs.
static constexpr bool kEnableTrackStatsDiag = false;

// Stereo match uniqueness ceiling. The matcher's margin gate (best - runner >=
// margin_min) only checks RELATIVE dominance: on repetitive texture two strong
// periodic peaks (e.g. best 0.90, runner 0.70) pass even though the runner-up is
// itself a perfectly good candidate -> ambiguous -> the period-aliased "ghost".
// runner = peak - margin (exact). Reject the match when its runner-up is itself
// >= a valid-match level, i.e. there is "another good option" on the epipolar
// line. Rejected matches fall through to being kept as mono-left features.
// Tunable: raise toward 0.70 if this rejects too many legit matches.
static constexpr float kStereoRunnerMax = 0.60f;
// Helper: does this match pass the uniqueness ceiling (runner-up not too strong)?
static inline bool stereo_runner_ok(float peak, float margin) {
    return (peak - margin) < kStereoRunnerMax;
}


void TrackOCL::enable_zncc_stereo_matcher(const modal_flow::StereoCalib &calib_in,
                                          float z_min, float z_max)
{
    auto stm = std::make_unique<modal_flow::ocl::StereoMatcherCL>(dev_);
    // Matcher gates. margin_min (forward peak minus runner-up) is the uniqueness /
    // anti-aliasing gate: on repetitive texture an aliased match has a strong
    // runner-up (the other period of the pattern) -> small margin. The default 0.10
    // was too loose and let near-field FALSE matches through; they triangulate at
    // large disparity and pile at the z_min floor (~0.4 m), corrupting local scale.
    // Tightened to 0.20 to reject ambiguous matches (the central large-disparity
    // "ghost" aliases that still survived at 0.15). lr_thresh tightened 5.0 -> 3.0
    // (L-R round-trip px) for the same reason.
    // 2026-07: ReinitEvent diagnostic (peak_zncc/margin logged per delayed_init) showed
    // ~76% of bad (too-close) single-instant stereo promotions had peak_zncc in
    // [0.60,0.70) and ~50% had margin in [0.20,0.25) -- i.e. barely clearing the old
    // floors, not confidently-wrong. Tightened zncc_min 0.60->0.70 and margin_min
    // 0.20->0.30 to reject that population; re-measure both the degenerate-reinit rate
    // and overall stereo yield after this change (stereo has previously been found
    // starved -- this trades some yield for fewer false near-field promotions).
    // Mirror the host-validated ZNCC params (stereo_exploration): high peak floor + LOOSE
    // uniqueness margin + 2px L-R. The prior 0.70/0.30 over-rejected good-but-ambiguous matches
    // (per-feature [STEREO DUMP] showed peak~0.99 killed by margin<0.30 on repetitive texture).
    // A high peak floor (0.80) still rejects the bad near-field promotions (those sat at 0.60-0.70),
    // while a loose margin (0.10) admits the correct-but-non-unique matches the host accepted.
    stereo_zncc_min_   = 0.80f;
    stereo_margin_min_ = 0.10f;
    stereo_lr_thresh_  = 2.0f;
    stm->set_zncc_min(stereo_zncc_min_);
    stm->set_margin_min(stereo_margin_min_);
    stm->set_lr_thresh(stereo_lr_thresh_);
    stm->set_band(stereo_band_px_);   // widen perpendicular search to tolerate residual R_lr tilt
    mgr_.set_stereo_matcher(std::move(stm));

    if (stereo_diag_on_)
        printf("[TrackOCL] stereo reject diagnostics ON (summary every %d detect calls%s -> /run/voxl-open-vins-stereo-diag.log)\n",
               stereo_diag_period_, stereo_dump_on_ ? " + per-feature epipolar dump" : "");

    modal_flow::StereoCalib c = calib_in;
    c.z_min = z_min;
    c.z_max = z_max;
    mgr_.set_stereo_calibration(c);

    stereo_cam_id_left_  = (size_t)c.left;
    stereo_cam_id_right_ = (size_t)c.right;

    // Build the static left camera model used to undistort left features into
    // bearings for the matcher. Seeded from the StereoCalib K_left/D_left (the
    // static conf, NOT the online-calibrated camera_calib) so the ZNCC search is
    // fully decoupled from EKF intrinsic calibration. The matcher kernel is
    // equidistant-only, so a CamEqui is exact here. undistort_f only consumes
    // K/D, but we pass the real dims from camera_calib for completeness.
    {
        int w = 0, h = 0;
        auto it = camera_calib.find(stereo_cam_id_left_);
        if (it != camera_calib.end() && it->second) {
            w = it->second->w();
            h = it->second->h();
        }
        auto cam = std::make_shared<CamEqui>(w, h);
        Eigen::MatrixXd calib(8, 1);
        calib << c.K_left[0], c.K_left[1], c.K_left[2], c.K_left[3],
                 c.D_left[0], c.D_left[1], c.D_left[2], c.D_left[3];
        cam->set_value(calib);
        stereo_static_cam_left_ = cam;
    }

    // Right-camera model + extrinsics for the per-feature epipolar dump (host
    // replica of the kernel projection). Same static seed calib as the matcher.
    {
        int w = 0, h = 0;
        auto it = camera_calib.find(stereo_cam_id_right_);
        if (it != camera_calib.end() && it->second) {
            w = it->second->w();
            h = it->second->h();
        }
        auto camR = std::make_shared<CamEqui>(w, h);
        Eigen::MatrixXd calibR(8, 1);
        calibR << c.K_right[0], c.K_right[1], c.K_right[2], c.K_right[3],
                  c.D_right[0], c.D_right[1], c.D_right[2], c.D_right[3];
        camR->set_value(calibR);
        stereo_static_cam_right_ = camR;
    }
    for (int r = 0; r < 3; r++)
        for (int cc = 0; cc < 3; cc++)
            stereo_R_lr_(r, cc) = c.R_lr[r * 3 + cc]; // R_lr is row-major
    stereo_t_lr_ << c.t_lr[0], c.t_lr[1], c.t_lr[2];
    stereo_z_min_ = z_min;
    stereo_z_max_ = z_max;

    // Narrow-search half-window (inverse depth) for the continuous per-frame re-match. Disparity
    // is ~linear in inverse depth (d ~= fx*baseline*rho), so a +/-kRematchBandPx disparity window
    // maps to +/-(kRematchBandPx/(fx*baseline)) in rho. A tracked feature's depth changes slowly,
    // so this bounds the search cheaply AND stops re-matches from jumping to a far epipolar period.
    {
        const float kRematchBandPx = 12.0f;   // level-0 px; ~generous vs typical per-frame disparity change
        const float baseline = stereo_t_lr_.norm();
        const float fx = c.K_left[0];
        stereo_rho_half_width_ = (fx * baseline > 1e-6f) ? kRematchBandPx / (fx * baseline) : 0.f;
    }

    printf("[TrackOCL] ZNCC-band stereo matcher enabled (src cam %zu -> dst cam %zu, z=[%.2f,%.1f]m)\n",
           stereo_cam_id_left_, stereo_cam_id_right_, (double)z_min, (double)z_max);
    printf("[TrackOCL]   static left bearing calib: fxy=(%.1f,%.1f) c=(%.1f,%.1f) D=(%.4f,%.4f,%.4f,%.4f)\n",
           (double)c.K_left[0], (double)c.K_left[1], (double)c.K_left[2], (double)c.K_left[3],
           (double)c.D_left[0], (double)c.D_left[1], (double)c.D_left[2], (double)c.D_left[3]);
}

// Attribute one submitted stereo candidate to the gate that rejected it (or to
// accepted). Reason priority mirrors how the gates are actually applied: the
// matcher enforces peak -> margin -> L-R -> reverse, then TrackOCL adds the
// runner-up ceiling and the right-image bounds check. "reverse_fail" is inferred
// by elimination: peak+margin+L-R all pass but the matcher still set status=0,
// which can only be the reverse-pass peak gate (rev_peak is not surfaced).
void TrackOCL::accumulate_stereo_reject_(int pass, float peak, float margin, float lr,
                                         bool matcher_status, bool right_oob)
{
    auto &s = stereo_reject_stats_;
    s.submitted[pass]++;
    if (peak <= -1.5f) { s.oob[pass]++; return; } // no in-bounds epipolar candidate

    s.ndist[pass]++;
    s.peak_sum[pass] += peak; s.marg_sum[pass] += margin; s.lr_sum[pass] += lr;
    s.peak_min[pass] = std::min(s.peak_min[pass], peak); s.peak_max[pass] = std::max(s.peak_max[pass], peak);
    s.marg_min[pass] = std::min(s.marg_min[pass], margin); s.marg_max[pass] = std::max(s.marg_max[pass], margin);
    s.lr_min[pass]   = std::min(s.lr_min[pass], lr);       s.lr_max[pass]   = std::max(s.lr_max[pass], lr);

    const bool peak_ok   = peak   >= stereo_zncc_min_;
    const bool marg_ok   = margin >= stereo_margin_min_;
    const bool lr_ok     = lr     <  stereo_lr_thresh_;
    const bool runner_ok = (peak - margin) < kStereoRunnerMax;

    if (matcher_status && runner_ok && !right_oob) { s.accepted[pass]++; return; }

    if      (!peak_ok)         { s.weak_peak[pass]++;      if (peak   >= stereo_zncc_min_   - 0.10f) s.nm_peak[pass]++; }
    else if (!marg_ok)         { s.low_margin[pass]++;     if (margin >= stereo_margin_min_ - 0.05f) s.nm_margin[pass]++; }
    else if (!lr_ok)           { s.lr_fail[pass]++;        if (lr     <  stereo_lr_thresh_  + 2.00f) s.nm_lr[pass]++; }
    else if (!matcher_status)  { s.reverse_fail[pass]++; } // peak+margin+L-R ok, matcher said no => reverse
    else if (!runner_ok)       { s.runner_ceiling[pass]++; if (peak - margin < kStereoRunnerMax + 0.10f) s.nm_runner[pass]++; }
    else if (right_oob)        { s.right_oob[pass]++; }
}

void TrackOCL::maybe_print_stereo_diag_()
{
    if (!stereo_diag_on_) return;
    if (++stereo_diag_calls_ < stereo_diag_period_) return;

    // Write to the shared per-process log file only (stdout diag prints removed).
    FILE *fp = stereo_diag_log();

    auto       &s        = stereo_reject_stats_;
    const char *names[3] = {"top-off", "promote", "r-promote"};
    FILE       *outs[1]  = {fp};
    const int   nouts    = (fp != nullptr) ? 1 : 0;
    for (int o = 0; o < nouts; o++) {
        FILE *f = outs[o];
        fprintf(f, "[STEREO DIAG] over %d detect calls  gates: zncc>=%.2f margin>=%.2f lr<%.1fpx runner<%.2f\n",
                stereo_diag_calls_, stereo_zncc_min_, stereo_margin_min_, stereo_lr_thresh_, kStereoRunnerMax);
        for (int p = 0; p < 3; p++) {
            const long sub = s.submitted[p];
            if (sub == 0) { fprintf(f, "  %-7s: (none)\n", names[p]); continue; }
            fprintf(f, "  %-7s: sub=%ld acc=%ld (%.1f%%) | oob=%ld weak_peak=%ld low_margin=%ld lr=%ld reverse=%ld runner=%ld rOOB=%ld\n",
                    names[p], sub, s.accepted[p], 100.0 * (double)s.accepted[p] / (double)sub,
                    s.oob[p], s.weak_peak[p], s.low_margin[p], s.lr_fail[p],
                    s.reverse_fail[p], s.runner_ceiling[p], s.right_oob[p]);
            fprintf(f, "           near-miss(loosen 1 gate->pass): peak=%ld margin=%ld lr=%ld runner=%ld",
                    s.nm_peak[p], s.nm_margin[p], s.nm_lr[p], s.nm_runner[p]);
            if (s.ndist[p] > 0) {
                const double inv = 1.0 / (double)s.ndist[p];
                fprintf(f, "  | dist peak[%.2f/%.2f/%.2f] margin[%.2f/%.2f/%.2f] lr[%.1f/%.1f/%.1f]\n",
                        s.peak_min[p], s.peak_sum[p] * inv, s.peak_max[p],
                        s.marg_min[p], s.marg_sum[p] * inv, s.marg_max[p],
                        s.lr_min[p],   s.lr_sum[p]   * inv, s.lr_max[p]);
            } else {
                fprintf(f, "\n");
            }
        }
        fflush(f);
    }
    stereo_reject_stats_ = StereoRejectStats{};
    stereo_diag_calls_   = 0;
}

void TrackOCL::dump_stereo_epipolar_(const modal_flow::StereoMatchInput &in,
                                     const modal_flow::StereoMatchResult &res,
                                     int img_w1, int img_h1)
{
    const int N = (int)in.source_points.size();
    if (N == 0 || !stereo_static_cam_right_) return;

    // Mirror the kernel's BORDER guard (level-0 px). At kStereoPyrLevel>0 the
    // kernel scales this; the dump runs the projection at full res for clarity.
    const int   BORDER = 12;
    const float zf = stereo_z_max_, zn = stereo_z_min_; // far / near sweep ends

    // Replicate the kernel forward projection: left bearing -> 3D at depth Z ->
    // R_lr/t_lr -> normalize -> right-cam equidistant distort -> right pixel.
    auto project_right = [&](float bx, float by, float Z, bool &in_bounds) -> cv::Point2f {
        Eigen::Vector3f P1 = stereo_R_lr_ * Eigen::Vector3f(bx * Z, by * Z, Z) + stereo_t_lr_;
        if (P1.z() <= 1e-6f) { in_bounds = false; return {-1.f, -1.f}; }
        Eigen::Vector2f uv = stereo_static_cam_right_->distort_f(
            Eigen::Vector2f(P1.x() / P1.z(), P1.y() / P1.z()));
        in_bounds = (uv.x() >= BORDER && uv.x() < img_w1 - BORDER &&
                     uv.y() >= BORDER && uv.y() < img_h1 - BORDER);
        return {uv.x(), uv.y()};
    };

    FILE *fp     = stereo_diag_log();
    FILE *outs[1] = {fp};                 // file only (stdout diag prints removed)
    const int nouts = (fp != nullptr) ? 1 : 0;

    const int step = std::max(1, N / stereo_dump_per_window_);
    for (int o = 0; o < nouts; o++) {
        FILE *f = outs[o];
        fprintf(f, "[STEREO DUMP] z=[%.2f,%.1f]m border=%d img=%dx%d  (sample %d of %d)\n",
                zn, zf, BORDER, img_w1, img_h1, stereo_dump_per_window_, N);
        for (int i = 0, shown = 0; i < N && shown < stereo_dump_per_window_; i += step, shown++) {
            bool far_in = false, near_in = false;
            cv::Point2f pf = project_right(in.source_bearings[i].x, in.source_bearings[i].y, zf, far_in);
            cv::Point2f pn = project_right(in.source_bearings[i].x, in.source_bearings[i].y, zn, near_in);
            const float peak = res.peak_zncc[i], marg = res.margin[i], lr = res.lr_err[i];
            const bool  st   = res.status[i] != 0;
            const char *why  = (peak <= -1.5f)             ? "oob"
                             : (peak < stereo_zncc_min_)   ? "weak_peak"
                             : (marg < stereo_margin_min_) ? "low_margin"
                             : (lr  >= stereo_lr_thresh_)  ? "lr_fail"
                             : (!st)                       ? "reverse_fail"
                                                           : "accept";
            fprintf(f, "  L(%.1f,%.1f) far(%.1f,%.1f)%s near(%.1f,%.1f)%s best(%.1f,%.1f) "
                       "peak=%.2f marg=%.2f lr=%.1f st=%d -> %s\n",
                    in.source_points[i].x, in.source_points[i].y,
                    pf.x, pf.y, far_in ? "IN " : "OOB",
                    pn.x, pn.y, near_in ? "IN " : "OOB",
                    res.target_points[i].x, res.target_points[i].y,
                    peak, marg, lr, st ? 1 : 0, why);
        }
        fflush(f);
    }
}

void TrackOCL::feed_new_camera(const CameraData &message)
{

    // Error check that we have all the data
    if (message.sensor_ids.empty() ||
        (message.sensor_ids.size() != message.images.size()) ||
        (message.images.size() != message.masks.size()))
    {
        PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
        PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
        PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
        PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
        std::exit(EXIT_FAILURE);
    }

    // Preprocessing steps that we do not parallelize
    // NOTE: DO NOT PARALLELIZE THESE!
    // NOTE: These seem to be much slower if you parallelize them...
    rT1 = prof_now();

    size_t num_images = message.images.size();

    // Stability guard: drop single-cam "MONO-fallback" messages when configured
    // for stereo. CameraQueueFusion emits them at startup before both cam frames
    // have landed; routing them to feed_monocular has triggered cv::resize() on
    // empty masks and OclPyramidRing ABA throws. Stereo VIO can't use single-cam
    // frames anyway. One-shot warning so we know if they fire at steady state.
    if (use_stereo && num_images < 2)
    {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr,
                    "[TrackOCL] dropping MONO-fallback CameraData (num_images=%zu) "
                    "because TrackOCL is in stereo mode -- this should only happen at "
                    "startup before both cams have landed a frame. Subsequent drops "
                    "are silent.\n", num_images);
            warned = true;
        }
        return;
    }

    // Guard the "map::at" throw: a bundle carrying a cam_id not in our configured maps
    // (mtx_feeds/camera_calib) throws std::out_of_range downstream. Skip+log instead of throwing,
    // and surface the offending id so the pairing bug is traceable rather than a bare "map::at".
    for (size_t k = 0; k < message.sensor_ids.size(); k++) {
        size_t sid = message.sensor_ids[k];
        if (sid >= mtx_feeds.size() || camera_calib.find(sid) == camera_calib.end()) {
            static int s_badid = 0;
            if (s_badid++ < 50)
                fprintf(stderr, "[TrackOCL] SKIP malformed frame: sensor_id=%zu not configured "
                        "(n_img=%zu, n_cams=%zu)\n", sid, num_images, mtx_feeds.size());
            return;
        }
    }

    for (size_t msg_id = 0; msg_id < num_images; msg_id++)
    {
        // Lock this data feed for this camera
        size_t cam_id = message.sensor_ids.at(msg_id);
        std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

        modal_flow::Frame frame = message.img_frames[msg_id];

        // upload image to flow manager
        if (img_buf_next_[cam_id]) {
            if (img_buf_prev_[cam_id]) {
                mgr_.release_pyramid((modal_flow::CameraId)cam_id, img_buf_prev_[cam_id]);
            }
            img_buf_prev_[cam_id] = img_buf_next_[cam_id];
        }
        img_buf_next_[cam_id] = mgr_.acquire_pyramid_buf((modal_flow::CameraId)cam_id);
        mgr_.upload_frame_to_buf(frame, img_buf_next_[cam_id]);
    }

    // Either call our stereo or monocular version
    // If we are doing binocular tracking, then we should parallize our tracking
    if (num_images == 1)
    {
        feed_monocular(message, 0);
    }
    else if (num_images == 2 && use_stereo)
    {
        feed_stereo(message, 0, 1);
    }
    else if (!use_stereo)
    {
        // NOTE: opencv::parallel_for() seems to be less efficient than direct loop
        for (int i = 0; i < (int)num_images; i++)
        {
            feed_monocular(message, i);
        }
    }
    else
    {
        PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, we only support mono or stereo tracking", num_images);
        std::exit(EXIT_FAILURE);
    }
}

static int64_t _apps_time_monotonic_ns()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts))
    {
        fprintf(stderr, "ERROR calling clock_gettime\n");
        return -1;
    }
    return (int64_t)ts.tv_sec * 1000000000 + (int64_t)ts.tv_nsec;
}

void TrackOCL::feed_monocular(const CameraData &message, size_t msg_id)
{
    // Lock this data feed for this camera
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Get our image objects for this image
    cv::Mat mask = message.masks.at(msg_id);
    
    std::pair<int, int> dims = mgr_.get_cam_dim(cam_id);
    int cam_width  = std::get<0>(dims);
    int cam_height = std::get<1>(dims);
    
    rT2 = prof_now();
    int64_t t2 = _apps_time_monotonic_ns();
    // If we didn't have any successful tracks last time, just extract this time
    // This also handles, the tracking initalization on the first call to this extractor
    if (pts_last[cam_id].empty())
    {
        // Detect new features
        std::vector<cv::KeyPoint> good_left;
        std::vector<size_t> good_ids_left;
        perform_detection_monocular(img_buf_next_[cam_id], mask, good_left, good_ids_left, cam_id);

        // Save the current image and pyramid
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_mask_last[cam_id] = mask;
        pts_last[cam_id] = good_left;
        ids_last[cam_id] = good_ids_left;

        return;
    }

    // First we should make that the last images have enough features so we can do KLT
    // This will "top-off" our number of tracks so always have a constant number
    auto pts_left_old = pts_last[cam_id];
    auto ids_left_old = ids_last[cam_id];

    perform_detection_monocular(img_buf_prev_[cam_id], img_mask_last[cam_id], pts_left_old, ids_left_old, cam_id);
    rT3 = prof_now();
    int64_t t3 = _apps_time_monotonic_ns();

    // Our return success masks, and predicted new features
    std::vector<uchar> mask_ll;
    std::vector<cv::KeyPoint> pts_left_new = pts_left_old;

    // Lets track temporally (IMU-predicted seeding when enabled; identity otherwise -- A/B toggle)
    const modal_flow::RotationQuat dq_pred = delta_q_for_cam(cam_id, last_cam_time_[cam_id], message.timestamp);
    last_cam_time_[cam_id] = message.timestamp;
    const modal_flow::RotationQuat dq = kImuAidedSeeding ? dq_pred : modal_flow::RotationQuat{};
    perform_matching(img_buf_prev_[cam_id], img_buf_next_[cam_id], pts_left_old, pts_left_new, cam_id, cam_id, mask_ll, dq);
    assert(pts_left_new.size() == ids_left_old.size());
    int64_t t4 = _apps_time_monotonic_ns();
    rT4 = prof_now();

    // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
    if (mask_ll.empty())
    {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_mask_last[cam_id] = mask;
        pts_last[cam_id].clear();
        ids_last[cam_id].clear();
        PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
        return;
    }

    // Get our "good tracks"
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;
    // printf("cam_id: %d, pts tracked: %zu\n", cam_id, pts_left_new.size());

    // Loop through all left points
    for (size_t i = 0; i < pts_left_new.size(); i++)
    {
        // Ensure we do not have any bad KLT tracks (i.e., points are negative)
        if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= cam_width ||
            (int)pts_left_new.at(i).pt.y >= cam_height)
            continue;
        // Check if it is in the mask
        // NOTE: mask has max value of 255 (white) if it should be
        if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
            continue;
        // If it is a good track, and also tracked from left to right
        if (mask_ll[i])
        {
            good_left.push_back(pts_left_new[i]);
            good_ids_left.push_back(ids_left_old[i]);
        }
    }
    // printf("cam_id: %d, good tracks: %zu\n", cam_id, good_left.size());

    // Update our feature database, with theses new observations
    for (size_t i = 0; i < good_left.size(); i++)
    {
        cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
        database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                                 npt_l.y);
    }

    // Move forward in time
    {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_mask_last[cam_id] = mask;
        pts_last[cam_id] = good_left;
        ids_last[cam_id] = good_ids_left;
    }
    int64_t t5 = _apps_time_monotonic_ns();
    rT5 = prof_now();

    // DIAGNOSTIC: per-frame mono track count for this camera (stereo fields 0).
    if (kEnableTrackStatsDiag) {
        note_track_stats(message.timestamp, "MONO", (int)cam_id, (int)good_left.size(), 0, 0, 0, 0);
    }

    // Timing prints in milliseconds
    auto dt = [](int64_t a, int64_t b){ return double(b - a) / 1e6; };

    // printf("[TIME-KLT]: %.3f ms for pyramid\n", dt(t1, t2));
    // printf("[TIME-KLT]: %.3f ms for detection\n", dt(t2, t3));
    // printf("[TIME-KLT]: %.3f ms for temporal klt\n", dt(t3, t4));
    // printf("[TIME-KLT]: %.3f ms for feature DB update (%d features)\n",
    //        dt(t4, t5), (int)good_left.size());
    // printf("[TIME-KLT]: %.3f ms total\n", dt(t1, t5));
}

void TrackOCL::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right)
{
    // Lock this data feed for this camera
    size_t cam_id_left = message.sensor_ids.at(msg_id_left);
    size_t cam_id_right = message.sensor_ids.at(msg_id_right);
    std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
    std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

    // Get our mask objects for this image.
    // NOTE: the OCL path keeps images/pyramids on the GPU (img_buf_*_), so the CPU
    // img_curr / img_pyramid_curr maps are never populated here (unlike CPU TrackKLT).
    cv::Mat mask_left = message.masks.at(msg_id_left);
    cv::Mat mask_right = message.masks.at(msg_id_right);

    std::pair<int, int> dims_left = mgr_.get_cam_dim(cam_id_left);
    int cam_width_left   = std::get<0>(dims_left);
    int cam_height_left = std::get<1>(dims_left);

    std::pair<int, int> dims_right = mgr_.get_cam_dim(cam_id_right);
    int cam_width_right  = std::get<0>(dims_right);
    int cam_height_right = std::get<1>(dims_right);

    // ---- ONE-SHOT DIAGNOSTIC: log the L/R ORDER the matcher receives (first few frames) ----
    // The pipeline assumes images[0]=LEFT, images[1]=RIGHT (feed_new_camera dispatches (0,1)); a
    // swapped bundle silently inverts the epipolar geometry -> uniform weak ZNCC. Order confirmed
    // correct during bring-up; kept as a cheap startup assertion.
    {
        static int s_pairdump = 0;
        if (s_pairdump < 4) {
            printf("[STEREO PAIRDUMP] #%d ts=%.6f  sensor_ids=[%zu,%zu]  idx0->cam_left=%zu  idx1->cam_right=%zu\n",
                   s_pairdump, message.timestamp,
                   message.sensor_ids.size() > 0 ? message.sensor_ids[0] : (size_t)999,
                   message.sensor_ids.size() > 1 ? message.sensor_ids[1] : (size_t)999,
                   cam_id_left, cam_id_right);
            fflush(stdout);
            s_pairdump++;
        }
    }

    // NOTE: do NOT re-upload frames here. feed_new_camera() already swapped prev<-next
    // and uploaded the CURRENT frame for BOTH cameras before dispatching to feed_stereo().
    // Re-uploading collapses prev==next (both = current frame), so temporal KLT matches a
    // frame against itself -> 0.000 disparity (stuck in ZUPT). feed_monocular() relies on
    // the same feed_new_camera() upload; mirror that behavior here.
    
    rT2 = prof_now();
    int64_t t2 = _apps_time_monotonic_ns();

    if (pts_last[cam_id_left].empty() && pts_last[cam_id_right].empty()) {
        // Track into the new image
        std::vector<cv::KeyPoint> good_left, good_right;
        std::vector<size_t> good_ids_left, good_ids_right;
        perform_detection_stereo(img_buf_next_[cam_id_left], img_buf_next_[cam_id_right], mask_left, mask_right,
                                 cam_id_left, cam_id_right, good_left, good_right, good_ids_left, good_ids_right);
        // Save the current image and pyramid
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        // img_last / img_pyramid_last are unused by the OCL tracker (images stay on the GPU)
        img_mask_last[cam_id_left] = mask_left;
        img_mask_last[cam_id_right] = mask_right;
        pts_last[cam_id_left] = good_left;
        pts_last[cam_id_right] = good_right;
        ids_last[cam_id_left] = good_ids_left;
        ids_last[cam_id_right] = good_ids_right;
        return;
    }

    // First we should make that the last images have enough features so we can do KLT
    // This will "top-off" our number of tracks so always have a constant number
    int pts_before_detect = (int)pts_last[cam_id_left].size();
    auto pts_left_old = pts_last[cam_id_left];
    auto pts_right_old = pts_last[cam_id_right];
    auto ids_left_old = ids_last[cam_id_left];
    auto ids_right_old = ids_last[cam_id_right];
    perform_detection_stereo(img_buf_prev_[cam_id_left], img_buf_prev_[cam_id_right], 
                             img_mask_last[cam_id_left], img_mask_last[cam_id_right],
                             cam_id_left, cam_id_right, pts_left_old, pts_right_old, ids_left_old, ids_right_old);
    rT3 = prof_now();

    // Temporal KLT on BOTH cameras. KLT is what makes stereo pairs PERSIST: a pair only has to
    // pass the (strict) stereo-match gate ONCE, at detection, and KLT then carries it for its
    // lifetime -- that persistence is what accumulates a healthy stereo population. The narrow
    // re-match below is a drift CORRECTION on top of KLT, NOT a per-frame gate (an earlier
    // continuous-matching design that re-gated every frame starved stereo -- pairs were gained
    // then immediately lost).
    std::vector<uchar> mask_ll, mask_rr;
    std::vector<cv::KeyPoint> pts_left_new  = pts_left_old;
    std::vector<cv::KeyPoint> pts_right_new = pts_right_old;
    // Per-camera IMU-predicted rotation over the inter-frame interval (each camera's own R_ItoC).
    // Identity until feed_imu()/set_cam_imu_rotation() are wired -> original behavior.
    const modal_flow::RotationQuat dq_l_pred = delta_q_for_cam(cam_id_left,  last_cam_time_[cam_id_left],  message.timestamp);
    const modal_flow::RotationQuat dq_r_pred = delta_q_for_cam(cam_id_right, last_cam_time_[cam_id_right], message.timestamp);
    last_cam_time_[cam_id_left]  = message.timestamp;
    last_cam_time_[cam_id_right] = message.timestamp;
    // A/B TOGGLE (kImuAidedSeeding): apply the prediction to the KLT seed only when enabled. The
    // [IMU-SEED] diagnostic below always measures dq_*_pred, so the OFF build logs the counterfactual.
    const modal_flow::RotationQuat dq_l = kImuAidedSeeding ? dq_l_pred : modal_flow::RotationQuat{};
    const modal_flow::RotationQuat dq_r = kImuAidedSeeding ? dq_r_pred : modal_flow::RotationQuat{};
    perform_matching(img_buf_prev_[cam_id_left],  img_buf_next_[cam_id_left],  pts_left_old,  pts_left_new,  cam_id_left,  cam_id_left,  mask_ll, dq_l);
    perform_matching(img_buf_prev_[cam_id_right], img_buf_next_[cam_id_right], pts_right_old, pts_right_new, cam_id_right, cam_id_right, mask_rr, dq_r);

    // ---- IMU-aided seeding diagnostic (left camera; periodic) ----
    // reduction% = how much of the per-frame feature motion the IMU-predicted seed removed:
    //   naive = |prev - klt_result| (total motion),  pred = |imu_seed - klt_result| (residual).
    // High reduction under a non-zero predicted rotation => prediction is working; ~0 => identity
    // seed (IMU/extrinsics not wired) or a wrong ΔR. Also reports gyro-buffer liveness.
    if (stereo_diag_on_ && ++imu_diag_frames_ % stereo_diag_period_ == 0 && !mask_ll.empty()) {
        auto cam_it = camera_calib.find(cam_id_left);
        // predicted seed = project(dq^-1 * unproject(prev)); mirrors the tracker's internal seed.
        auto predict_px = [&](const cv::Point2f &prev) -> cv::Point2f {
            if (cam_it == camera_calib.end()) return prev;
            Eigen::Vector2f nb = cam_it->second->undistort_f(Eigen::Vector2f(prev.x, prev.y));
            // Mirror predict_next EXACTLY: dq_l_pred is JPL (v=(x,y,z), scalar w); transport the
            // bearing by R{q}^T:  (2w^2-1) b + 2w (v x b) + 2 v (v . b).
            const float vx = dq_l_pred.x, vy = dq_l_pred.y, vz = dq_l_pred.z, w = dq_l_pred.w;
            const float bx = nb(0), by = nb(1), bz = 1.f;
            const float vdotb = vx*bx + vy*by + vz*bz, s = 2.f*w*w - 1.f;
            const float rx = s*bx + 2.f*w*(vy*bz - vz*by) + 2.f*vx*vdotb;
            const float ry = s*by + 2.f*w*(vz*bx - vx*bz) + 2.f*vy*vdotb;
            const float rz = s*bz + 2.f*w*(vx*by - vy*bx) + 2.f*vz*vdotb;
            if (rz <= 1e-6f) return prev;
            Eigen::Vector2f uv = cam_it->second->distort_f(Eigen::Vector2f(rx/rz, ry/rz));
            return cv::Point2f(uv(0), uv(1));
        };
        std::vector<float> naive, pred;
        for (size_t i = 0; i < mask_ll.size() && i < pts_left_new.size(); i++) {
            if (!mask_ll[i]) continue;
            const cv::Point2f &pv = pts_left_old[i].pt, &nw = pts_left_new[i].pt;
            cv::Point2f sd = predict_px(pv);
            naive.push_back(std::hypot(nw.x - pv.x, nw.y - pv.y));
            pred.push_back(std::hypot(nw.x - sd.x, nw.y - sd.y));
        }
        auto med = [](std::vector<float> &v) -> float {
            if (v.empty()) return 0.f; std::sort(v.begin(), v.end()); return v[v.size()/2];
        };
        const float mn = med(naive), mp = med(pred);
        const float ang_deg = 2.f * std::acos(std::min(1.f, std::fabs(dq_l_pred.w))) * 57.29578f;
        const double span = imu_rot_.size() ? (imu_rot_.t_newest() - imu_rot_.t_oldest()) : 0.0;
        const double reduction = (mn > 1e-3f) ? 100.0 * (1.0 - mp / mn) : 0.0;
        FILE *fp = imu_seed_log();   // file only (stdout diag prints removed)
        if (fp) {
            fprintf(fp, "[IMU-SEED] mode=%s cam%zu  gyro_buf=%zu span=%.2fs  predRot=%.2fdeg  "
                    "medMotion=%.2fpx medResidual=%.2fpx  reduction=%.0f%%  (n=%zu)\n",
                    kImuAidedSeeding ? "ON" : "OFF",
                    cam_id_left, imu_rot_.size(), span, ang_deg, mn, mp, reduction, naive.size());
            fflush(fp);
        }
    }

    rT4 = prof_now();

    // left to right matching
    // TODO: we should probably still do this to reject outliers
    // TODO: maybe we should collect all tracks that are in both frames and make they pass this?
    // std::vector<uchar> mask_lr;
    // perform_matching(imgpyr_left, imgpyr_right, pts_left_new, pts_right_new, cam_id_left, cam_id_right, mask_lr);
    rT5 = prof_now();

    // If any of our masks are empty, that means we didn't have enough to do ransac, so just return
    if (mask_ll.empty() && mask_rr.empty()) {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_mask_last[cam_id_left] = mask_left;
        img_mask_last[cam_id_right] = mask_right;
        pts_last[cam_id_left].clear();
        pts_last[cam_id_right].clear();
        ids_last[cam_id_left].clear();
        ids_last[cam_id_right].clear();
        PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
        return;
    }

    // ---- DRIFT CORRECTION (continuous re-match as a CORRECTOR, not a gate) ----
    // For each KLT-surviving stereo pair, re-derive the right point via a narrow ZNCC search
    // around its inverse-depth prior + L-R check. If the matcher CONFIDENTLY accepts and lands
    // more than kCorrectSnapPx from the KLT right point, snap to the matcher's point (removes the
    // independent-KLT drift). If the matcher rejects, we do NOTHING -- KLT keeps the pair alive,
    // so the strict match gate can no longer kill a pair, only refine it. Pairs whose right KLT
    // was lost fall through to mono-left in the assembly below (as before).
    const bool stereo_bound = (cam_id_left  == stereo_cam_id_left_ &&
                               cam_id_right == stereo_cam_id_right_);
    // Approach B only: id -> right pixel re-derived from the left this frame (empty in Approach A).
    std::unordered_map<size_t, cv::Point2f> rederive_right;
    if (stereo_bound && !kStereoRederive) {
        std::unordered_map<size_t, int> right_idx;   // id -> index of KLT-surviving right track
        for (size_t n = 0; n < ids_right_old.size(); n++)
            if (n < mask_rr.size() && mask_rr[n]) right_idx[ids_right_old[n]] = (int)n;

        modal_flow::StereoMatchInput cin{};
        cin.source_cam_id   = (modal_flow::CameraId)cam_id_left;
        cin.target_cam_id  = (modal_flow::CameraId)cam_id_right;
        cin.source_img_buf  = img_buf_next_[cam_id_left];    // CURRENT frame, both cameras
        cin.target_img_buf = img_buf_next_[cam_id_right];
        cin.rho_half_width = stereo_rho_half_width_;
        std::vector<int> corr_li, corr_ri;                 // parallel: left idx, right idx
        for (size_t i = 0; i < pts_left_new.size(); i++) {
            if (!mask_ll[i]) continue;
            auto pit = stereo_inv_depth_prior_.find(ids_left_old.at(i));
            if (pit == stereo_inv_depth_prior_.end()) continue;   // not a stereo pair
            auto rit = right_idx.find(ids_left_old.at(i));
            if (rit == right_idx.end()) continue;                 // right KLT lost -> becomes mono below
            const cv::Point2f &p = pts_left_new.at(i).pt;
            if (p.x < 0 || p.y < 0 || (int)p.x > cam_width_left || (int)p.y > cam_height_left) continue;
            Eigen::Vector2f nb = stereo_static_cam_left_->undistort_f(Eigen::Vector2f(p.x, p.y));
            cin.source_points  .push_back({p.x, p.y, 0.f});
            cin.source_bearings.push_back({nb(0), nb(1), 0.f});
            cin.inv_depth_prior.push_back(pit->second);
            corr_li.push_back((int)i);
            corr_ri.push_back(rit->second);
        }
        if (!cin.source_points.empty()) {
            const float kCorrectSnapPx = 1.5f;   // only override KLT when they disagree by more than this
            modal_flow::StereoMatchResult c = mgr_.match_stereo(cin);
            corr_sub_ += (long)cin.source_points.size();
            for (size_t k = 0; k < cin.source_points.size(); k++) {
                if (!(c.status[k] && stereo_runner_ok(c.peak_zncc[k], c.margin[k]))) continue;  // reject -> keep KLT
                cv::Point2f mpt(c.target_points[k].x, c.target_points[k].y);
                if ((int)mpt.x < 0 || (int)mpt.x >= cam_width_right ||
                    (int)mpt.y < 0 || (int)mpt.y >= cam_height_right) continue;
                corr_acc_++;
                // Phase-0 R_lr observable: accumulate the perp-residual tilt fit over accepted,
                // in-bounds matches. Weight by margin (confident, unique matches vote harder).
                // Gated on the diag flag so the running sums reset with the print block below.
                if (stereo_diag_on_) {
                    const double xc = (double)cin.source_points[k].x - 0.5 * (double)cam_width_left;
                    const double r  = (double)c.perp_resid[k];
                    const double w  = (double)std::max(c.margin[k], 0.f);
                    epi_sw_ += w; epi_swx_ += w * xc; epi_swxx_ += w * xc * xc;
                    epi_swr_ += w * r; epi_swrx_ += w * r * xc; epi_swrr_ += w * r * r;
                    epi_n_++;
                }
                cv::Point2f &kpt = pts_right_new.at(corr_ri[k]).pt;
                float dpx = std::hypot(mpt.x - kpt.x, mpt.y - kpt.y);
                if (dpx > kCorrectSnapPx) {                        // snap out drift
                    kpt = mpt;
                    corr_snap_++;
                    corr_snap_sum_ += dpx;
                    corr_snap_max_ = std::max(corr_snap_max_, dpx);
                }
                const size_t id = ids_left_old.at(corr_li[k]);
                stereo_inv_depth_prior_[id] = c.inv_depth[k];      // refresh depth prior
                stereo_confidence_[id] = StereoConfidence{c.peak_zncc[k], c.margin[k], c.lr_err[k]};
            }
        }
    } else if (stereo_bound && kStereoRederive) {
        // ===== Approach B: re-derive the right partner from the left EVERY frame =====
        // Submit ALL left-KLT survivors that carry a stereo depth prior -- independent of
        // right-KLT state, so a pair can re-acquire its partner after a right-KLT loss. The
        // ZNCC match (narrow window around the per-frame refreshed inverse-depth prior) becomes
        // the right observation outright; reject => no right this frame (mono-left, fails-safe).
        std::unordered_map<size_t, int> right_klt;   // id -> surviving right-KLT idx (snap-px stat only)
        for (size_t n = 0; n < ids_right_old.size(); n++)
            if (n < mask_rr.size() && mask_rr[n]) right_klt[ids_right_old[n]] = (int)n;

        modal_flow::StereoMatchInput cin{};
        cin.source_cam_id  = (modal_flow::CameraId)cam_id_left;
        cin.target_cam_id  = (modal_flow::CameraId)cam_id_right;
        cin.source_img_buf = img_buf_next_[cam_id_left];    // CURRENT frame, both cameras
        cin.target_img_buf = img_buf_next_[cam_id_right];
        cin.rho_half_width = stereo_rho_half_width_;
        std::vector<size_t> cand_id;
        for (size_t i = 0; i < pts_left_new.size(); i++) {
            if (!mask_ll[i]) continue;
            auto pit = stereo_inv_depth_prior_.find(ids_left_old.at(i));
            if (pit == stereo_inv_depth_prior_.end()) continue;   // not a stereo pair
            const cv::Point2f &p = pts_left_new.at(i).pt;
            if (p.x < 0 || p.y < 0 || (int)p.x > cam_width_left || (int)p.y > cam_height_left) continue;
            Eigen::Vector2f nb = stereo_static_cam_left_->undistort_f(Eigen::Vector2f(p.x, p.y));
            cin.source_points  .push_back({p.x, p.y, 0.f});
            cin.source_bearings.push_back({nb(0), nb(1), 0.f});
            cin.inv_depth_prior.push_back(pit->second);
            cand_id.push_back(ids_left_old.at(i));
        }
        if (!cin.source_points.empty()) {
            modal_flow::StereoMatchResult c = mgr_.match_stereo(cin);
            corr_sub_ += (long)cin.source_points.size();
            for (size_t k = 0; k < cin.source_points.size(); k++) {
                if (!(c.status[k] && stereo_runner_ok(c.peak_zncc[k], c.margin[k]))) continue;  // reject -> mono-left
                cv::Point2f mpt(c.target_points[k].x, c.target_points[k].y);
                if ((int)mpt.x < 0 || (int)mpt.x >= cam_width_right ||
                    (int)mpt.y < 0 || (int)mpt.y >= cam_height_right) continue;
                corr_acc_++;
                const size_t id = cand_id[k];
                // stat: how far the re-derivation sits from the drift-prone right-KLT, when present.
                auto rk = right_klt.find(id);
                if (rk != right_klt.end()) {
                    float dpx = std::hypot(mpt.x - pts_right_new.at(rk->second).pt.x,
                                           mpt.y - pts_right_new.at(rk->second).pt.y);
                    corr_snap_++; corr_snap_sum_ += dpx; corr_snap_max_ = std::max(corr_snap_max_, dpx);
                }
                if (stereo_diag_on_) {
                    const double xc = (double)cin.source_points[k].x - 0.5 * (double)cam_width_left;
                    const double r  = (double)c.perp_resid[k];
                    const double w  = (double)std::max(c.margin[k], 0.f);
                    epi_sw_ += w; epi_swx_ += w * xc; epi_swxx_ += w * xc * xc;
                    epi_swr_ += w * r; epi_swrx_ += w * r * xc; epi_swrr_ += w * r * r;
                    epi_n_++;
                }
                rederive_right[id] = mpt;                          // this frame's right observation
                stereo_inv_depth_prior_[id] = c.inv_depth[k];      // refresh depth prior
                stereo_confidence_[id] = StereoConfidence{c.peak_zncc[k], c.margin[k], c.lr_err[k]};
            }
        }
    }
    // Rolling correction diagnostic: confirms the corrector is live and how often it snaps drift.
    // Mirrored to the shared stereo diag file (/run/voxl-open-vins-stereo-diag.log) like STEREO DIAG.
    if (stereo_diag_on_ && ++corr_diag_frames_ % stereo_diag_period_ == 0) {
        double mean = corr_snap_ ? corr_snap_sum_ / (double)corr_snap_ : 0.0;
        FILE *fp = stereo_diag_log();   // file only (stdout diag prints removed)
        if (fp) {
            fprintf(fp, "[STEREO CORRECTION] over %d frames: submitted=%ld accepted=%ld snapped=%ld (%.0f%% of acc)  snap px mean/max=%.2f/%.2f\n",
                    stereo_diag_period_, corr_sub_, corr_acc_, corr_snap_,
                    corr_acc_ ? 100.0 * (double)corr_snap_ / (double)corr_acc_ : 0.0, mean, (double)corr_snap_max_);
            // R_lr observable: solve the weighted fit perp = a + b*x_center and report the tilt.
            const double det = epi_sw_ * epi_swxx_ - epi_swx_ * epi_swx_;
            if (epi_n_ > 20 && std::fabs(det) > 1e-9) {
                const double a  = (epi_swr_ * epi_swxx_ - epi_swx_ * epi_swrx_) / det;   // centre residual (~pitch)
                const double b  = (epi_sw_  * epi_swrx_ - epi_swx_ * epi_swr_) / det;    // slope (px per px)
                const double e2e  = b * (double)cam_width_left;                          // edge-to-edge tilt (~roll)
                const double mean_r = epi_swr_ / epi_sw_;
                const double rms  = std::sqrt(std::max(0.0, epi_swrr_ / epi_sw_));        // raw perp RMS (band-clipped)
                char epi_line[256];
                snprintf(epi_line, sizeof(epi_line),
                         "[STEREO EPI] over %d frames n=%ld  perp mean=%.2f rms=%.2f px  fit: centre a=%.2f  edge-to-edge b*w=%.2f px  (band=%d clip)\n",
                         stereo_diag_period_, epi_n_, mean_r, rms, a, e2e, stereo_band_px_);
                fputs(epi_line, fp);        // /run diag file (full stereo diagnostics)
                fputs(epi_line, stdout);    // ALSO echo to terminal for Phase-0 bring-up watch
                fflush(stdout);
            }
            fflush(fp);
        }
        corr_sub_ = corr_acc_ = corr_snap_ = 0; corr_snap_sum_ = 0.0; corr_snap_max_ = 0.f;
        epi_sw_ = epi_swx_ = epi_swxx_ = epi_swr_ = epi_swrx_ = epi_swrr_ = 0.0; epi_n_ = 0;
    }
    rT5 = prof_now();

    // Get our "good tracks" (original assembly; pts_right_new may have been drift-corrected above).
    std::vector<cv::KeyPoint> good_left, good_right;
    std::vector<size_t> good_ids_left, good_ids_right;

    if (!(kStereoRederive && stereo_bound)) {
    // Loop through all left points
    for (size_t i = 0; i < pts_left_new.size(); i++) {
        if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x > cam_width_left ||
            (int)pts_left_new.at(i).pt.y > cam_height_left)
        continue;
        bool found_right = false;
        size_t index_right = 0;
        for (size_t n = 0; n < ids_right_old.size(); n++) {
        if (ids_left_old.at(i) == ids_right_old.at(n)) { found_right = true; index_right = n; break; }
        }
        if (mask_ll[i] && found_right && mask_rr[index_right]) {
        if (pts_right_new.at(index_right).pt.x < 0 || pts_right_new.at(index_right).pt.y < 0 ||
            (int)pts_right_new.at(index_right).pt.x >= cam_width_right || (int)pts_right_new.at(index_right).pt.y >= cam_height_right)
            continue;
        good_left.push_back(pts_left_new.at(i));
        good_right.push_back(pts_right_new.at(index_right));
        good_ids_left.push_back(ids_left_old.at(i));
        good_ids_right.push_back(ids_right_old.at(index_right));
        } else if (mask_ll[i]) {
        good_left.push_back(pts_left_new.at(i));
        good_ids_left.push_back(ids_left_old.at(i));
        }
    }

    // Loop through all right points (mono-right features -- independent right-camera tracks)
    for (size_t i = 0; i < pts_right_new.size(); i++) {
        if (pts_right_new.at(i).pt.x < 0 || pts_right_new.at(i).pt.y < 0 || (int)pts_right_new.at(i).pt.x >= cam_width_right ||
            (int)pts_right_new.at(i).pt.y >= cam_height_right)
        continue;
        bool added_already = (std::find(good_ids_right.begin(), good_ids_right.end(), ids_right_old.at(i)) != good_ids_right.end());
        if (mask_rr[i] && !added_already) {
        good_right.push_back(pts_right_new.at(i));
        good_ids_right.push_back(ids_right_old.at(i));
        }
    }
    } else {
    // ===== Approach B assembly: the right stereo obs is the re-derivation ONLY =====
    // left survivors -> good_left; right partner attached iff re-derived this frame. Paired
    // rights are NEVER taken from right-KLT (that is the drift we are avoiding); a paired right
    // that failed re-derivation is intentionally dropped (mono-left). Unpaired right-KLT tracks
    // still flow through as mono-right.
    std::unordered_set<size_t> left_ids(ids_left_old.begin(), ids_left_old.end());
    for (size_t i = 0; i < pts_left_new.size(); i++) {
        if (!mask_ll[i]) continue;
        const cv::Point2f &lp = pts_left_new.at(i).pt;
        if (lp.x < 0 || lp.y < 0 || (int)lp.x > cam_width_left || (int)lp.y > cam_height_left) continue;
        good_left.push_back(pts_left_new.at(i));
        good_ids_left.push_back(ids_left_old.at(i));
        auto rd = rederive_right.find(ids_left_old.at(i));
        if (rd != rederive_right.end()) {
            good_right.push_back(cv::KeyPoint(rd->second, pts_left_new.at(i).size));
            good_ids_right.push_back(ids_left_old.at(i));
        }
    }
    for (size_t i = 0; i < pts_right_new.size(); i++) {
        if (!mask_rr[i]) continue;
        const cv::Point2f &rp = pts_right_new.at(i).pt;
        if (rp.x < 0 || rp.y < 0 || (int)rp.x >= cam_width_right || (int)rp.y >= cam_height_right) continue;
        if (left_ids.count(ids_right_old.at(i))) continue;   // paired -> re-derivation-only
        good_right.push_back(pts_right_new.at(i));
        good_ids_right.push_back(ids_right_old.at(i));
    }
    }

    // Update our feature database, with theses new observations
    for (size_t i = 0; i < good_left.size(); i++) {
        cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
        database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                                npt_l.y);
    }
    for (size_t i = 0; i < good_right.size(); i++) {
        cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
        database->update_feature(good_ids_right.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                                npt_r.y);
    }

    // Move forward in time
    {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        // img_last / img_pyramid_last are unused by the OCL tracker (images stay on the GPU)
        img_mask_last[cam_id_left] = mask_left;
        img_mask_last[cam_id_right] = mask_right;
        pts_last[cam_id_left] = good_left;
        pts_last[cam_id_right] = good_right;
        ids_last[cam_id_left] = good_ids_left;
        ids_last[cam_id_right] = good_ids_right;
    }
    rT6 = prof_now();

    // DIAGNOSTIC: per-frame track breakdown. n_stereo = features observed in BOTH
    // cams (shared id between good_ids_left/right); the rest are mono on one side.
    // last_n_promoted_ = this frame's mono-left -> stereo upgrades (promote pass).
    if (kEnableTrackStatsDiag) {
        std::unordered_set<size_t> right_set(good_ids_right.begin(), good_ids_right.end());
        int n_stereo = 0;
        for (size_t id : good_ids_left)
            if (right_set.count(id)) n_stereo++;
        int n_mono_left  = (int)good_ids_left.size()  - n_stereo;
        int n_mono_right = (int)good_ids_right.size() - n_stereo;
        note_track_stats(message.timestamp, "STEREO", -1,
                         n_stereo + n_mono_left + n_mono_right,
                         n_stereo, n_mono_left, n_mono_right, last_n_promoted_);
    }

    //  // Timing information
    PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", prof_s(rT1, rT2));
    PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%d detected)\n", prof_s(rT2, rT3),
                (int)pts_last[cam_id_left].size() - pts_before_detect);
    PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", prof_s(rT3, rT4));
    PRINT_ALL("[TIME-KLT]: %.4f seconds for stereo klt\n", prof_s(rT4, rT5));
    PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", prof_s(rT5, rT6),
                (int)good_left.size());
    PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", prof_s(rT1, rT6));
}

void TrackOCL::perform_detection_monocular(modal_flow::BufferId& buf_id, const cv::Mat &mask0,
                                           std::vector<cv::KeyPoint> &pts0,
                                           std::vector<size_t> &ids0, int cam_id)
{

    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less than grid_px_size points away then existing features

    int64_t rT1 = _apps_time_monotonic_ns();
    
    int img_width  = mask0.cols;
    int img_height = mask0.rows;

    cv::Size size_close((int)((float)img_width  / (float)min_px_dist),
                        (int)((float)img_height / (float)min_px_dist)); // width x height
    cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
    float size_x = (float)img_width  / (float)grid_x;
    float size_y = (float)img_height / (float)grid_y;
    cv::Size size_grid(grid_x, grid_y); // width x height
    cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
    cv::Mat mask0_updated = mask0.clone();
    auto it0 = pts0.begin();
    auto it1 = ids0.begin();
    while (it0 != pts0.end())
    {
        // Get current left keypoint, check that it is in bounds
        cv::KeyPoint kpt = *it0;
        int x = (int)kpt.pt.x;
        int y = (int)kpt.pt.y;
        int edge = 10;
        if (x < edge || x >= img_width - edge || y < edge || y >= img_height - edge)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate mask coordinates for close points
        int x_close = (int)(kpt.pt.x / (float)min_px_dist);
        int y_close = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate what grid cell this feature is in
        int x_grid = std::floor(kpt.pt.x / size_x);
        int y_grid = std::floor(kpt.pt.y / size_y);
        if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Check if this keypoint is near another point
        if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Now check if it is in a mask area or not
        // NOTE: mask has max value of 255 (white) if it should be
        if (mask0.at<uint8_t>(y, x) > 127)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Else we are good, move forward to the next point
        grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
        if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255)
        {
            grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
        }
        // Append this to the local mask of the image
        if (x - min_px_dist >= 0 && x + min_px_dist < img_width && y - min_px_dist >= 0 && y + min_px_dist < img_height)
        {
            cv::Point pt1(x - min_px_dist, y - min_px_dist);
            cv::Point pt2(x + min_px_dist, y + min_px_dist);
            cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255));
        }
        it0++;
        it1++;
    }

    int64_t rT2 = _apps_time_monotonic_ns();

    // First compute how many more features we need to extract from this image
    // If we don't need any features, just return
    double min_feat_percent = 0.50;
    int num_featsneeded = num_features - (int)pts0.size();
    if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
        return;

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask0_grid;
    cv::resize(mask0, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid.cols; x++)
    {
        for (int y = 0; y < grid_2d_grid.rows; y++)
        {
            if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255)
            {
                valid_locs.emplace_back(x, y);
            }
        }
    }
    std::vector<cv::KeyPoint> pts0_ext;
    Grider_OCL::perform_griding_use_flow(mgr_, cam_id, buf_id, mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);
    
    int64_t rT3 = _apps_time_monotonic_ns();

    // Now, reject features that are close to a current feature
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext)
    {
        // Check that it is in bounds
        int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
        int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
            continue;
        // See if there is a point at this location
        if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
            continue;
        // Else lets add it!
        kpts0_new.push_back(kpt);
        pts0_new.push_back(kpt.pt);
        grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
    }

    // Loop through and record only ones that are valid
    // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
    // NOTE: this is due to the fact that we select update features based on feat id
    // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
    // NOTE: not sure how to remove... maybe a better way?
    for (size_t i = 0; i < pts0_new.size(); i++)
    {
        // update the uv coordinates
        kpts0_new.at(i).pt = pts0_new.at(i);
        // append the new uv coordinate
        pts0.push_back(kpts0_new.at(i));
        // move id foward and append this new point
        size_t temp = ++currid;
        ids0.push_back(temp);
    }
    int64_t rT4 = _apps_time_monotonic_ns();
    // printf("[TIME-DTCT]: %.4f seconds for grid creation\n", (rT2 - rT1) * 1e-6);
    // printf("[TIME-DTCT]: %.4f seconds for grid detection\n", (rT3 - rT2) * 1e-6);
    // printf("[TIME-DTCT]: %.4f seconds for feature rejection\n", (rT4 - rT3) * 1e-6);
}

void TrackOCL::perform_detection_stereo(modal_flow::BufferId buf_id_left, modal_flow::BufferId buf_id_right,
                                        const cv::Mat &mask0, const cv::Mat &mask1,
                                        size_t cam_id_left, size_t cam_id_right,
                                        std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1,
                                        std::vector<size_t> &ids0, std::vector<size_t> &ids1)
{
    last_n_promoted_ = 0; // reset per-call promote counter (diagnostic)
    int img_width0  = mask0.cols;
    int img_height0 = mask0.rows;
    int img_width1  = mask1.cols;
    int img_height1 = mask1.rows;

    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less then grid_px_size points away then existing features
    cv::Size size_close0((int)((float)img_width0 / (float)min_px_dist),
                        (int)((float)img_height0 / (float)min_px_dist)); // width x height
    cv::Mat grid_2d_close0 = cv::Mat::zeros(size_close0, CV_8UC1);
    float size_x0 = (float)img_width0 / (float)grid_x;
    float size_y0 = (float)img_height0 / (float)grid_y;
    cv::Size size_grid0(grid_x, grid_y); // width x height
    cv::Mat grid_2d_grid0 = cv::Mat::zeros(size_grid0, CV_8UC1);
    cv::Mat mask0_updated = mask0.clone();
    auto it0 = pts0.begin();
    auto it1 = ids0.begin();
    while (it0 != pts0.end()) {
        // Get current left keypoint, check that it is in bounds
        cv::KeyPoint kpt = *it0;
        int x = (int)kpt.pt.x;
        int y = (int)kpt.pt.y;
        int edge = 10;
        if (x < edge || x >= img_width0 - edge || y < edge || y >= img_height0 - edge) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate mask coordinates for close points
        int x_close = (int)(kpt.pt.x / (float)min_px_dist);
        int y_close = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_close < 0 || x_close >= size_close0.width || y_close < 0 || y_close >= size_close0.height) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate what grid cell this feature is in
        int x_grid = std::floor(kpt.pt.x / size_x0);
        int y_grid = std::floor(kpt.pt.y / size_y0);
        if (x_grid < 0 || x_grid >= size_grid0.width || y_grid < 0 || y_grid >= size_grid0.height) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Check if this keypoint is near another point
        if (grid_2d_close0.at<uint8_t>(y_close, x_close) > 127) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Now check if it is in a mask area or not
        // NOTE: mask has max value of 255 (white) if it should be
        if (mask0.at<uint8_t>(y, x) > 127) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Else we are good, move forward to the next point
        grid_2d_close0.at<uint8_t>(y_close, x_close) = 255;
        if (grid_2d_grid0.at<uint8_t>(y_grid, x_grid) < 255) {
            grid_2d_grid0.at<uint8_t>(y_grid, x_grid) += 1;
        }
        // Append this to the local mask of the image
        if (x - min_px_dist >= 0 && x + min_px_dist < img_width0 && y - min_px_dist >= 0 && y + min_px_dist < img_height0) {
            cv::Point pt1(x - min_px_dist, y - min_px_dist);
            cv::Point pt2(x + min_px_dist, y + min_px_dist);
            cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
        }
        it0++;
        it1++;
    }

    // First compute how many more features we need to extract from this image
    double min_feat_percent = 0.50;
    int num_featsneeded_0 = num_features - (int)pts0.size();

    // LEFT: if we need features we should extract them in the current frame
    // LEFT: we will also try to track them from this frame over to the right frame
    // LEFT: in the case that we have two features that are the same, then we should merge them
    if (num_featsneeded_0 > std::min(20, (int)(min_feat_percent * num_features))) {
        // We also check a downsampled mask such that we don't extract in areas where it is all masked!
        cv::Mat mask0_grid;
        cv::resize(mask0, mask0_grid, size_grid0, 0.0, 0.0, cv::INTER_NEAREST);

        // Create grids we need to extract from and then extract our features (use fast with griding)
        int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
        int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
        std::vector<std::pair<int, int>> valid_locs;
        for (int x = 0; x < grid_2d_grid0.cols; x++) {
            for (int y = 0; y < grid_2d_grid0.rows; y++) {
                if ((int)grid_2d_grid0.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
                    valid_locs.emplace_back(x, y);
                }
            }
        }
        std::vector<cv::KeyPoint> pts0_ext;
        Grider_OCL::perform_griding_use_flow(mgr_, cam_id_left, buf_id_left, mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

        // Now, reject features that are close a current feature
        std::vector<cv::KeyPoint> kpts0_new;
        std::vector<cv::Point2f> pts0_new;
        for (auto &kpt : pts0_ext) {
            // Check that it is in bounds
            int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
            int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
            if (x_grid < 0 || x_grid >= size_close0.width || y_grid < 0 || y_grid >= size_close0.height)
                continue;
            // See if there is a point at this location
            if (grid_2d_close0.at<uint8_t>(y_grid, x_grid) > 127)
                continue;
            // Else lets add it!
            grid_2d_close0.at<uint8_t>(y_grid, x_grid) = 255;
            kpts0_new.push_back(kpt);
            pts0_new.push_back(kpt.pt);
        }


        // Project the new left features into the right image via the ZNCC
        // epipolar-band matcher (set up once at startup by
        // enable_zncc_stereo_matcher). The matcher does forward+reverse ZNCC
        // along the calibrated epipolar curve and gates on peak / margin /
        // L-R round-trip consistency; per-feature confidence is stashed in
        // stereo_confidence_ for downstream EKF measurement weighting.
        std::vector<cv::KeyPoint> kpts1_new;
        std::vector<cv::Point2f>  pts1_new;
        kpts1_new = kpts0_new;
        pts1_new  = pts0_new;

        if (pts0_new.empty()) {
            // nothing to project -- fall through to the right-image dedupe loop below
        } else if (cam_id_left  == stereo_cam_id_left_ &&
                   cam_id_right == stereo_cam_id_right_) {
            // Pre-compute normalized cam0 bearings for every new left feature.
            // camera_calib already exists in the base class and exposes the
            // ov_core fisheye/radtan undistort -- exact match for what the
            // GPU matcher expects (it does no iterative undistort itself).
            modal_flow::StereoMatchInput in{};
            in.source_cam_id   = (modal_flow::CameraId)cam_id_left;
            in.target_cam_id  = (modal_flow::CameraId)cam_id_right;
            // Match against the buffer this function was called for, NOT
            // img_buf_next_. pts0_new and buf_id_* refer to the same frame
            // (prev on subsequent-frame calls). Using img_buf_next_ would
            // align positions to prev but images to next -- lr_err then
            // measures inter-frame motion (~8-60 px) instead of the round-trip
            // residual. perform_matching below carries accepted prev-frame
            // pairs forward to current via temporal KLT.
            in.source_img_buf  = buf_id_left;
            in.target_img_buf = buf_id_right;
            in.source_points  .reserve(pts0_new.size());
            in.source_bearings.reserve(pts0_new.size());
            for (const auto &p : pts0_new) {
                // Static seed calib (see stereo_static_cam_left_), NOT the
                // online-calibrated camera_calib, so the ZNCC search stays on the
                // same fixed calibration as the matcher's epipolar projection.
                Eigen::Vector2f n = stereo_static_cam_left_->undistort_f(Eigen::Vector2f(p.x, p.y));
                in.source_points  .push_back({p.x, p.y, 0.f});
                in.source_bearings.push_back({n(0), n(1), 0.f});
            }

            modal_flow::StereoMatchResult res = mgr_.match_stereo(in);

            // Reject diagnostics (runtime, rolling): attribute every candidate
            // to the gate that rejected it. Pure accounting -- the accept loop
            // below is the real logic and is unchanged.
            if (stereo_diag_on_) {
                for (size_t i = 0; i < pts0_new.size(); i++) {
                    cv::Point2f rpt(res.target_points[i].x, res.target_points[i].y);
                    bool right_oob = ((int)rpt.x < 0 || (int)rpt.x >= img_width1 ||
                                      (int)rpt.y < 0 || (int)rpt.y >= img_height1);
                    accumulate_stereo_reject_(/*pass=*/0, res.peak_zncc[i], res.margin[i],
                                              res.lr_err[i], res.status[i], right_oob);
                }
            }

            for (size_t i = 0; i < pts0_new.size(); i++) {
                bool oob_left = ((int)pts0_new.at(i).x < 0 || (int)pts0_new.at(i).x >= img_width0 ||
                                 (int)pts0_new.at(i).y < 0 || (int)pts0_new.at(i).y >= img_height0);
                if (oob_left) continue;

                if (res.status[i] && stereo_runner_ok(res.peak_zncc[i], res.margin[i])) {
                    cv::Point2f rpt(res.target_points[i].x, res.target_points[i].y);
                    bool oob_right = ((int)rpt.x < 0 || (int)rpt.x >= img_width1 ||
                                      (int)rpt.y < 0 || (int)rpt.y >= img_height1);
                    if (!oob_right) {
                        kpts0_new.at(i).pt = pts0_new.at(i);
                        kpts1_new.at(i).pt = rpt;
                        pts0.push_back(kpts0_new.at(i));
                        pts1.push_back(kpts1_new.at(i));
                        size_t temp = ++currid;
                        ids0.push_back(temp);
                        ids1.push_back(temp);
                        // Stash confidence for the eventual EKF measurement-noise weighting.
                        stereo_confidence_[temp] = StereoConfidence{
                            res.peak_zncc[i], res.margin[i], res.lr_err[i]};
                        stereo_inv_depth_prior_[temp] = res.inv_depth[i];  // seed continuous-match prior
                        continue;
                    }
                }
                // Match rejected (or right oob) -> still record as a mono left feature.
                kpts0_new.at(i).pt = pts0_new.at(i);
                pts0.push_back(kpts0_new.at(i));
                ids0.push_back(++currid);
            }
        } else {
            // ZNCC matcher not bound to this cam pair -- unreachable in normal
            // operation since enable_zncc_stereo_matcher is called at startup
            // whenever use_stereo is true. As a defensive fallback, record the
            // FAST corners as mono-left features so they're temporally tracked;
            // the mono->stereo promote pass below will upgrade them once the
            // matcher is bound.
            for (size_t i = 0; i < pts0_new.size(); i++) {
                bool oob_left = ((int)pts0_new.at(i).x < 0 || (int)pts0_new.at(i).x >= img_width0 ||
                                 (int)pts0_new.at(i).y < 0 || (int)pts0_new.at(i).y >= img_height0);
                if (oob_left) continue;
                kpts0_new.at(i).pt = pts0_new.at(i);
                pts0.push_back(kpts0_new.at(i));
                ids0.push_back(++currid);
            }
        }
    }

    // Mono->stereo promote pass: re-run match_stereo on every existing mono-left
    // track (ids0[i] not in ids1) and upgrade successful matches to full stereo
    // pairs by appending the right point to pts1/ids1 with the same id. The
    // right-image dedup loop below preserves features whose id is in ids0 via
    // its is_stereo branch, so newly-promoted pairs survive cleanup.
    if (cam_id_left  == stereo_cam_id_left_ &&
        cam_id_right == stereo_cam_id_right_ &&
        !pts0.empty())
    {
        std::unordered_set<size_t> right_ids(ids1.begin(), ids1.end());

        modal_flow::StereoMatchInput in{};
        in.source_cam_id   = (modal_flow::CameraId)cam_id_left;
        in.target_cam_id  = (modal_flow::CameraId)cam_id_right;
        // Use this function's buf_id parameters; see the longer comment on the
        // top-off match call above for why img_buf_next_ would mis-align.
        in.source_img_buf  = buf_id_left;
        in.target_img_buf = buf_id_right;

        // Map kept index in `in` -> index in pts0/ids0 so we can write back.
        std::vector<size_t> src_idx;
        src_idx.reserve(pts0.size());
        in.source_points  .reserve(pts0.size());
        in.source_bearings.reserve(pts0.size());
        for (size_t i = 0; i < pts0.size(); i++) {
            if (right_ids.count(ids0[i])) continue;            // already stereo
            const cv::Point2f &p = pts0[i].pt;
            // Skip features too close to the border for the matcher's 11x11 patch.
            // The matcher runs on pyramid level kStereoPyrLevel, so the left patch
            // spans (PATCH_HALF << level) level-0 px around the point; keep the
            // point that far from the edge (plus slack) so the whole patch stays in
            // bounds. The kernel's BORDER guard handles the projected right point.
            // Out-of-bounds features yield the OOB sentinel anyway -- skipping them
            // up-front saves a kernel slot and keeps the accept-rate print honest.
            const int edge_margin = (5 << modal_flow::ocl::kStereoPyrLevel) + 5; // 15 @ L1, 25 @ L2
            if (p.x < edge_margin || p.x >= img_width0 - edge_margin ||
                p.y < edge_margin || p.y >= img_height0 - edge_margin) continue;
            // Static seed calib (see stereo_static_cam_left_), NOT camera_calib.
            Eigen::Vector2f n = stereo_static_cam_left_->undistort_f(Eigen::Vector2f(p.x, p.y));
            in.source_points  .push_back({p.x, p.y, 0.f});
            in.source_bearings.push_back({n(0), n(1), 0.f});
            src_idx.push_back(i);
        }

        if (!in.source_points.empty()) {
            modal_flow::StereoMatchResult res = mgr_.match_stereo(in);
            for (size_t k = 0; k < in.source_points.size(); k++) {
                if (!res.status[k] || !stereo_runner_ok(res.peak_zncc[k], res.margin[k])) continue;
                cv::Point2f rpt(res.target_points[k].x, res.target_points[k].y);
                if ((int)rpt.x < 0 || (int)rpt.x >= img_width1 ||
                    (int)rpt.y < 0 || (int)rpt.y >= img_height1) continue;
                size_t i = src_idx[k];
                cv::KeyPoint rkpt = pts0[i]; // copy keypoint attributes (size, octave, etc.)
                rkpt.pt = rpt;
                pts1.push_back(rkpt);
                ids1.push_back(ids0[i]);     // SAME id == becomes a stereo pair
                last_n_promoted_++;          // diagnostic: count mono->stereo upgrades
                stereo_confidence_[ids0[i]] = StereoConfidence{
                    res.peak_zncc[k], res.margin[k], res.lr_err[k]};
                stereo_inv_depth_prior_[ids0[i]] = res.inv_depth[k];  // seed continuous-match prior
            }
            // Reject diagnostics (runtime, rolling) -- pure accounting.
            if (stereo_diag_on_) {
                for (size_t k = 0; k < in.source_points.size(); k++) {
                    cv::Point2f rpt(res.target_points[k].x, res.target_points[k].y);
                    bool right_oob = ((int)rpt.x < 0 || (int)rpt.x >= img_width1 ||
                                      (int)rpt.y < 0 || (int)rpt.y >= img_height1);
                    accumulate_stereo_reject_(/*pass=*/1, res.peak_zncc[k], res.margin[k],
                                              res.lr_err[k], res.status[k], right_oob);
                }
                // Per-feature epipolar dump once per summary window (first call of
                // the window: stereo_diag_calls_ resets to 0 after each print).
                if (stereo_dump_on_ && stereo_diag_calls_ == 0)
                    dump_stereo_epipolar_(in, res, img_width1, img_height1);
            }
        }
    }
    maybe_print_stereo_diag_();

    // RIGHT: Now summarise the number of tracks in the right image
    // RIGHT: We will try to extract some monocular features if we have the room
    // RIGHT: This will also remove features if there are multiple in the same location
    cv::Size size_close1((int)((float)img_width1 / (float)min_px_dist), (int)((float)img_height1 / (float)min_px_dist));
    cv::Mat grid_2d_close1 = cv::Mat::zeros(size_close1, CV_8UC1);
    float size_x1 = (float)img_width1  / (float)grid_x;
    float size_y1 = (float)img_height1 / (float)grid_y;
    cv::Size size_grid1(grid_x, grid_y); // width x height
    cv::Mat grid_2d_grid1 = cv::Mat::zeros(size_grid1, CV_8UC1);
    cv::Mat mask1_updated = mask0.clone();
    it0 = pts1.begin();
    it1 = ids1.begin();
    
    while (it0 != pts1.end()) {
        // Get current left keypoint, check that it is in bounds
        cv::KeyPoint kpt = *it0;
        int x = (int)kpt.pt.x;
        int y = (int)kpt.pt.y;
        int edge = 10;
        if (x < edge || x >= img_width1 - edge || y < edge || y >= img_height1 - edge) {
            it0 = pts1.erase(it0);
            it1 = ids1.erase(it1);
            continue;
        }
        // Calculate mask coordinates for close points
        int x_close = (int)(kpt.pt.x / (float)min_px_dist);
        int y_close = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_close < 0 || x_close >= size_close1.width || y_close < 0 || y_close >= size_close1.height) {
            it0 = pts1.erase(it0);
            it1 = ids1.erase(it1);
            continue;
        }
        // Calculate what grid cell this feature is in
        int x_grid = std::floor(kpt.pt.x / size_x1);
        int y_grid = std::floor(kpt.pt.y / size_y1);
        if (x_grid < 0 || x_grid >= size_grid1.width || y_grid < 0 || y_grid >= size_grid1.height) {
            it0 = pts1.erase(it0);
            it1 = ids1.erase(it1);
            continue;
        }
        // Check if this keypoint is near another point
        // NOTE: if it is *not* a stereo point, then we will not delete the feature
        // NOTE: this means we might have a mono and stereo feature near each other, but that is ok
        bool is_stereo = (std::find(ids0.begin(), ids0.end(), *it1) != ids0.end());
        if (grid_2d_close1.at<uint8_t>(y_close, x_close) > 127 && !is_stereo) {
            it0 = pts1.erase(it0);
            it1 = ids1.erase(it1);
            continue;
        }
        // Now check if it is in a mask area or not
        // NOTE: mask has max value of 255 (white) if it should be
        if (mask1.at<uint8_t>(y, x) > 127) {
            it0 = pts1.erase(it0);
            it1 = ids1.erase(it1);
            continue;
        }
        // Else we are good, move forward to the next point
        grid_2d_close1.at<uint8_t>(y_close, x_close) = 255;
        if (grid_2d_grid1.at<uint8_t>(y_grid, x_grid) < 255) {
            grid_2d_grid1.at<uint8_t>(y_grid, x_grid) += 1;
        }

        // Append this to the local mask of the image
        if (x - min_px_dist >= 0 && x + min_px_dist < img_width1 && y - min_px_dist >= 0 && y + min_px_dist < img_height1) {
            cv::Point pt1(x - min_px_dist, y - min_px_dist);
            cv::Point pt2(x + min_px_dist, y + min_px_dist);
            cv::rectangle(mask1_updated, pt1, pt2, cv::Scalar(255), -1);
        }
        it0++;
        it1++;
    }

    // RIGHT: if we need features we should extract them in the current frame
    // RIGHT: note that we don't track them to the left as we already did left->right tracking above
    int num_featsneeded_1 = num_features - (int)pts1.size();
    if (num_featsneeded_1 > std::min(20, (int)(min_feat_percent * num_features))) {

        // We also check a downsampled mask such that we don't extract in areas where it is all masked!
        cv::Mat mask1_grid;
        cv::resize(mask1, mask1_grid, size_grid1, 0.0, 0.0, cv::INTER_NEAREST);

        // Create grids we need to extract from and then extract our features (use fast with griding)
        int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
        int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
        std::vector<std::pair<int, int>> valid_locs;
        for (int x = 0; x < grid_2d_grid1.cols; x++) {
            for (int y = 0; y < grid_2d_grid1.rows; y++) {
                if ((int)grid_2d_grid1.at<uint8_t>(y, x) < num_features_grid_req && (int)mask1_grid.at<uint8_t>(y, x) != 255) {
                valid_locs.emplace_back(x, y);
                }
            }
        }
        std::vector<cv::KeyPoint> pts1_ext;
        Grider_OCL::perform_griding_use_flow(mgr_, cam_id_right, buf_id_right, mask1_updated, valid_locs, pts1_ext, num_features, grid_x, grid_y, threshold, true);

        // Now, reject features that are close a current feature
        for (auto &kpt : pts1_ext) {
            // Check that it is in bounds
            int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
            int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
            if (x_grid < 0 || x_grid >= size_close1.width || y_grid < 0 || y_grid >= size_close1.height)
                continue;
            // See if there is a point at this location
            if (grid_2d_close1.at<uint8_t>(y_grid, x_grid) > 127)
                continue;
            // Else lets add it!
            pts1.push_back(kpt);
            size_t temp = ++currid;
            ids1.push_back(temp);
            grid_2d_close1.at<uint8_t>(y_grid, x_grid) = 255;
        }
    }

    // Mono-right -> stereo promote pass (REVERSE direction: the right image is
    // the match SOURCE, the left image the TARGET). This mirrors the mono-left
    // promote pass above and gives every right-only track a second, independent
    // chance to become a stereo pair -- roughly doubling stereo yield in scenes
    // the left detector seeded poorly. Placed after the right extraction so the
    // freshly-detected mono-right corners are also eligible.
    //
    // A right feature whose id is NOT already in ids0 (i.e. not already half of a
    // stereo pair) is reverse-matched into the left image via the generalized
    // source/target matcher API; on success we append the matched LEFT point to
    // pts0/ids0 under the SAME id, which makes it a stereo pair. The feed_stereo
    // assembly keys pairs purely on shared id and does not care which camera
    // anchored them, so a right-anchored pair is a first-class stereo feature.
    if (cam_id_left  == stereo_cam_id_left_ &&
        cam_id_right == stereo_cam_id_right_ &&
        stereo_static_cam_right_ && !pts1.empty())
    {
        std::unordered_set<size_t> left_ids(ids0.begin(), ids0.end());

        modal_flow::StereoMatchInput in{};
        in.source_cam_id  = (modal_flow::CameraId)cam_id_right;  // REVERSE: right is source
        in.target_cam_id  = (modal_flow::CameraId)cam_id_left;   //          left is target
        in.source_img_buf = buf_id_right;
        in.target_img_buf = buf_id_left;

        // Map kept index in `in` -> index in pts1/ids1 so we can write the pair back.
        std::vector<size_t> src_idx;
        src_idx.reserve(pts1.size());
        in.source_points  .reserve(pts1.size());
        in.source_bearings.reserve(pts1.size());
        for (size_t i = 0; i < pts1.size(); i++) {
            if (left_ids.count(ids1[i])) continue;             // already stereo
            const cv::Point2f &p = pts1[i].pt;
            // Same border margin as the forward promote pass (matcher patch @ level).
            const int edge_margin = (5 << modal_flow::ocl::kStereoPyrLevel) + 5;
            if (p.x < edge_margin || p.x >= img_width1 - edge_margin ||
                p.y < edge_margin || p.y >= img_height1 - edge_margin) continue;
            // Source bearings use the RIGHT (cam1) static model -- the reverse of
            // the forward passes, which undistort with stereo_static_cam_left_.
            Eigen::Vector2f n = stereo_static_cam_right_->undistort_f(Eigen::Vector2f(p.x, p.y));
            in.source_points  .push_back({p.x, p.y, 0.f});
            in.source_bearings.push_back({n(0), n(1), 0.f});
            src_idx.push_back(i);
        }

        if (!in.source_points.empty()) {
            modal_flow::StereoMatchResult res = mgr_.match_stereo(in);
            for (size_t k = 0; k < in.source_points.size(); k++) {
                if (!res.status[k] || !stereo_runner_ok(res.peak_zncc[k], res.margin[k])) continue;
                cv::Point2f lpt(res.target_points[k].x, res.target_points[k].y);  // matched LEFT point
                if ((int)lpt.x < 0 || (int)lpt.x >= img_width0 ||
                    (int)lpt.y < 0 || (int)lpt.y >= img_height0) continue;
                // Dedup guard: skip if the matched left point coincides with an
                // existing left feature -- otherwise a mono-right feature sitting
                // near an existing L->R pair would spawn a second landmark on ~one
                // 3D point. Reuse the left min_px_dist occupancy grid, and mark it
                // so reverse-promoted points also stay min_px_dist from each other.
                int xc = (int)(lpt.x / (float)min_px_dist);
                int yc = (int)(lpt.y / (float)min_px_dist);
                if (xc < 0 || xc >= size_close0.width || yc < 0 || yc >= size_close0.height) continue;
                if (grid_2d_close0.at<uint8_t>(yc, xc) > 127) continue;
                grid_2d_close0.at<uint8_t>(yc, xc) = 255;

                size_t i = src_idx[k];
                cv::KeyPoint lkpt = pts1[i];   // copy keypoint attributes (size, octave, etc.)
                lkpt.pt = lpt;
                pts0.push_back(lkpt);
                ids0.push_back(ids1[i]);       // SAME id == becomes a stereo pair
                last_n_promoted_++;            // diagnostic: count promotions (both directions)
                // inv_depth here is 1/Z in the RIGHT (source) frame. The forward
                // drift-corrector reads stereo_inv_depth_prior_ as a LEFT-frame
                // prior, so we deliberately do NOT seed it here (a mismatched frame
                // would corrupt the narrow re-match). The pair is carried by KLT on
                // both sides; the corrector re-acquires a left-frame prior on its
                // next successful forward match. Leaving the prior unset just means
                // "cold" for one cycle -- exactly how a brand-new pair behaves.
                stereo_confidence_[ids1[i]] = StereoConfidence{
                    res.peak_zncc[k], res.margin[k], res.lr_err[k]};
            }
            // Reject diagnostics (pass=2 -> reverse promote); pure accounting. The
            // "right_oob" slot here counts TARGET (left-image) OOB matches.
            if (stereo_diag_on_) {
                for (size_t k = 0; k < in.source_points.size(); k++) {
                    cv::Point2f lpt(res.target_points[k].x, res.target_points[k].y);
                    bool left_oob = ((int)lpt.x < 0 || (int)lpt.x >= img_width0 ||
                                     (int)lpt.y < 0 || (int)lpt.y >= img_height0);
                    accumulate_stereo_reject_(/*pass=*/2, res.peak_zncc[k], res.margin[k],
                                              res.lr_err[k], res.status[k], left_oob);
                }
            }
        }
    }

    return;
}

// Camera-frame relative rotation for cam_id over [t_prev, t_curr]: integrate the gyro buffer on
// SO(3), then conjugate the IMU-frame delta into the camera frame with R_ItoC. Returns identity
// (== no prediction) whenever the extrinsic isn't set, the interval is degenerate, or the IMU
// buffer doesn't span it -- so this is always a safe no-op until IMU is wired from VioManager.
modal_flow::RotationQuat TrackOCL::delta_q_for_cam(size_t cam_id, double t_prev, double t_curr)
{
    modal_flow::RotationQuat out{};  // identity
    auto rit = R_ItoC_.find(cam_id);
    if (rit == R_ItoC_.end() || t_prev <= 0.0 || !(t_curr > t_prev))
        return out;
    
    bool ok = false;
    
    Eigen::Vector4d dq_jpl = imu_rot_.delta_rotation(t_prev, t_curr, &ok);   // JPL quat
    
    if (!ok) return out;
    
    Eigen::Matrix3d R_imu = ov_core::quat_2_Rot(dq_jpl);
    Eigen::Matrix3d R_cam = rit->second * R_imu * rit->second.transpose();
    // Hand libmodal a JPL quaternion (vector = q(0:3), scalar = q(3)) -- its RotationQuat is JPL
    // too, so no Hamilton conversion at the boundary; the whole chain is one convention.
    Eigen::Vector4d q_cam = ov_core::rot_2_quat(R_cam);
    out.x = (float)q_cam(0);
    out.y = (float)q_cam(1);
    out.z = (float)q_cam(2);
    out.w = (float)q_cam(3);
    return out;
}

void TrackOCL::perform_matching(modal_flow::BufferId buf0, modal_flow::BufferId buf1, std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out, const modal_flow::RotationQuat &delta_q)
{

    // We must have equal vectors
    assert(kpts0.size() == kpts1.size());

    // Return if we don't have any points
    if (kpts0.empty() || kpts1.empty())
        return;

    // Convert keypoints into points (stupid opencv stuff)
    std::vector<cv::Point2f> pts0, pts1;
    std::vector<modal_flow::Keypoint> pts_in;
    std::vector<float> pts_out;
    for (size_t i = 0; i < kpts0.size(); i++)
    {
        pts0.push_back(kpts0.at(i).pt);
        pts1.push_back(kpts1.at(i).pt);

        modal_flow::Keypoint kp;
        kp.x = kpts0.at(i).pt.x;
        kp.y = kpts0.at(i).pt.y;
        kp.score = 0.f;
        pts_in.push_back(kp);

        // for gpu run
        pts_out.push_back(kpts0.at(i).pt.x);
        pts_out.push_back(kpts0.at(i).pt.y);
    }

    // If we don't have enough points for ransac just return empty
    // We set the mask to be all zeros since all points failed RANSAC
    if (pts0.size() < 10)
    {
        for (size_t i = 0; i < pts0.size(); i++)
            mask_out.push_back((uchar)0);
        return;
    }

    std::vector<uchar> mask_klt;
    int64_t t0 = _apps_time_monotonic_ns();

    modal_flow::TrackingBatch track_batch;
    modal_flow::TrackOptions topt;

    size_t cam_id = id0;
    std::pair<int, int> dims = mgr_.get_cam_dim(cam_id);

    std::vector<modal_flow::TrackInput> track_in(1);
    track_in[0].prev_cam_id = id0;
    track_in[0].next_cam_id = id1;
    track_in[0].prev_img_buf = buf0;
    track_in[0].next_img_buf = buf1;
    track_in[0].prev_points = pts_in;
    track_in[0].delta_q = delta_q;   // IMU-predicted seeding (identity => original behavior)

    auto res = mgr_.track_many(track_in);

    int64_t t1 = _apps_time_monotonic_ns();
    // printf("[TIME-KLT-INTERNAL]: run_track = %.3f ms\n", (t1 - t0) / 1e6);

    int n_points = res[0].next_points.size();
    mask_klt.resize(n_points);

    for (int i = 0; i < n_points; i++)
    {
        modal_flow::Keypoint point = res[0].next_points[i];
        pts1[i] = (cv::Point2f){point.x, point.y};
        mask_klt[i] = res[0].status[i];
    }

    std::vector<cv::Point2f> pts0_keep, pts1_keep;
    std::vector<int>         keep_idx;        // map back to original i

    pts0_keep.reserve(pts0.size());
    pts1_keep.reserve(pts1.size());
    keep_idx.reserve(pts0.size());

    for (size_t i = 0; i < pts0.size(); ++i) {
        if (mask_klt[i]) {                    // only keep successfully tracked points
            pts0_keep.push_back(pts0[i]);
            pts1_keep.push_back(pts1[i]);
            keep_idx.push_back((int)i);
        }
    }

    // Normalize these points, so we can then do ransac
    // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
    std::vector<cv::Point2f> pts0_n, pts1_n;
    for (size_t i = 0; i < pts0_keep.size(); i++)
    {
        pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0_keep.at(i)));
        pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1_keep.at(i)));
    }
    int64_t t2 = _apps_time_monotonic_ns();
    // printf("[TIME-KLT-INTERNAL]: undistort = %.3f ms\n", (t2 - t1) / 1e6);

    // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
    std::vector<uchar> mask_rsc;
    double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
    double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
    double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
    cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.95, 50, mask_rsc);
    // cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc);

    // int inliers = 0, outliers = 0;
    // for (auto m : mask_rsc) {
    //     if (m) inliers++;
    //     else   outliers++;
    // }

    // printf("[RANSAC] Inliers = %d, Outliers = %d, Total = %zu\n",
    //     inliers, outliers, mask_rsc.size());

    // printf("[RANSAC] Dumping %zu point pairs:\n", pts0_n.size());
    // for (size_t i = 0; i < pts0_n.size(); i++) {
    //     printf("  [%zu] (%.3f, %.3f) -> (%.3f, %.3f), status=%d\n",
    //         i, pts0_n[i].x, pts0_n[i].y,
    //         pts1_n[i].x, pts1_n[i].y,
    //         mask_rsc[i]);
    // }

    int64_t t3 = _apps_time_monotonic_ns();
    // printf("[TIME-KLT-INTERNAL]: perform_matching total = %6.3f ms,  RANSAC = %6.3f ms\n", (t3 - t0) / 1e6, (t3 - t2) / 1e6);

    // Loop through and record only ones that are valid
    // Expand compact RANSAC mask back to original indexing
    mask_out.assign(pts0.size(), (uchar)0);

    size_t M = std::min(mask_rsc.size(), keep_idx.size());
    for (size_t j = 0; j < M; ++j) {
        if (mask_rsc[j]) {
            int i_orig = keep_idx[j];
            // mask_klt[i_orig] is already true for kept points; AND for clarity
            mask_out[i_orig] = (uchar)1;
        }
    }

    // Copy back the updated positions
    for (size_t i = 0; i < pts0.size(); i++)
    {
        kpts0.at(i).pt = pts0.at(i);
        kpts1.at(i).pt = pts1.at(i);
    }
}
