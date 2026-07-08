/*
    ImuRotationIntegrator.h -- SO(3) gyro integration for feature-tracking prediction.
    
    Produces the relative rotation delta_R the camera underwent between two image timestamps, from the
    raw gyro stream, so the KLT tracker can seed nextPts with an IMU-predicted guess (vital under
    dynamic rotation). Mean rotation only -- no covariance/Jacobians; not preintegration, we just
    re-integrate the interval each frame with the current bias.
    
    Composed MULTIPLICATIVELY on SO(3): delta_R = prod exp_so3(w_i dt_i), RIGHT-multiplied since the
    gyro is body-frame (R_curr = R_prev * exp(w dt)); summing rotation vectors would drop the BCH
    commutator terms and be wrong when the axis turns. Trapezoidal rate + endpoint interpolation cover
    exactly [t0,t1]; bias subtracted before the exp map. See the per-step comments in the code below.
    JPL convention (ov_core::quat_ops): delta_rotation() returns a JPL quaternion [q1 q2 q3 q4]
    (scalar LAST, identity [0,0,0,1]). Timestamps are seconds on one clock (caller applies any td /
    exposure-midpoint offset); gyro is rad/s in the body/IMU frame.
    
    @author kyletyni
*/
#pragma once

#include <cstddef>
#include <mutex>
#include <vector>

#include <Eigen/Eigen>
#include "utils/quat_ops.h"  // ov_core::exp_so3, rot_2_quat, quat_2_Rot

namespace mf_imu {

// JPL identity quaternion (scalar last): zero vector part, unit scalar.
inline Eigen::Vector4d quat_identity_jpl() { return Eigen::Vector4d(0.0, 0.0, 0.0, 1.0); }

class ImuRotationIntegrator {
public:
    // capacity = ring-buffer depth in samples. 4096 @ ~1 kHz IMU is ~4 s of history -- far more
    // than the one inter-frame interval we ever integrate, with margin for jitter/late frames.
    explicit ImuRotationIntegrator(std::size_t capacity = 4096)
        : cap_(capacity ? capacity : 1) { buf_.resize(cap_); }

    // Slowly-varying gyro bias (rad/s), subtracted from every sample during integration. Optional:
    // a stale/absent value only costs a sub-pixel prediction error over one frame.
    void set_bias(double bx, double by, double bz) {
        std::lock_guard<std::mutex> lk(mtx_);
        bias_ << bx, by, bz;
    }

    // Append one gyro sample (timestamps must be non-decreasing). Overwrites the oldest when full.
    // Thread-safe against the reader: add_gyro runs on the IMU thread while delta_rotation runs on
    // the tracking thread.
    void add_gyro(double t, double gx, double gy, double gz) {
        std::lock_guard<std::mutex> lk(mtx_);
        buf_[head_] = Sample{t, gx, gy, gz};
        head_ = (head_ + 1) % cap_;   // advance write cursor (wraps)
        if (n_ < cap_) ++n_;          // grow until the buffer is full, then stay saturated
    }

    // Relative rotation over [t0, t1] as a JPL quaternion. Thin wrapper: integrate to a matrix, then
    // convert once with rot_2_quat. ok=false (identity) if the buffer does not fully span [t0, t1].
    Eigen::Vector4d delta_rotation(double t0, double t1, bool* ok = nullptr) const {
        return ov_core::rot_2_quat(delta_rotation_matrix(t0, t1, ok));
    }

    // The actual integration, returned as a rotation matrix (skips the quat round-trip when the
    // caller wants the matrix directly, e.g. for the R_ItoC conjugation).
    Eigen::Matrix3d delta_rotation_matrix(double t0, double t1, bool* ok = nullptr) const {
        std::lock_guard<std::mutex> lk(mtx_);
        auto set_ok = [&](bool v) { if (ok) *ok = v; };

        // Accumulator starts at the identity rotation (no motion yet).
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

        // Degenerate interval: need >=2 samples to form a sub-interval; t1==t0 is a valid no-op.
        if (n_ < 2 || !(t1 > t0)) { set_ok(t1 == t0 && n_ >= 1); return R; }

        // Coverage guard: if the buffer doesn't bracket the whole [t0, t1] (IMU gap / not caught up
        // yet), refuse rather than integrate a partial interval -- caller falls back to identity
        // ("no prediction"), which is always safe.
        if (t0 < at(0).t || t1 > at(n_ - 1).t) { set_ok(false); return R; }

        // Walk consecutive sample pairs and multiply in each sub-interval's rotation.
        for (std::size_t k = 0; k + 1 < n_; ++k) {
            const Sample& s0 = at(k);       // earlier sample
            const Sample& s1 = at(k + 1);   // later sample

            // Clamp this sample pair's span [s0.t, s1.t] to the requested window [t0, t1]. The
            // overlap [a, b] is what actually contributes; the first/last pairs are partial.
            const double a = s0.t > t0 ? s0.t : t0;   // max(s0.t, t0)
            const double b = s1.t < t1 ? s1.t : t1;   // min(s1.t, t1)
            if (!(b > a)) {                            // no overlap yet (or zero-width)
                if (s1.t >= t1) break;                 // ...and we've passed t1 -> done
                else continue;                         // ...still before t0 -> skip ahead
            }

            // Angular rate at the exact overlap endpoints, by linear interpolation of the two
            // bracketing samples (valid because rate is a Lie-algebra vector, not a rotation).
            Eigen::Vector3d wa, wb;
            interp(s0, s1, a, wa);
            interp(s0, s1, b, wb);

            // Rotation VECTOR for this sub-interval: (trapezoidal mean rate - bias) * elapsed time.
            // phi in so(3): its direction is the rotation axis, its magnitude the rotation angle.
            const Eigen::Vector3d phi = (0.5 * (wa + wb) - bias_) * (b - a);

            // exp_so3(phi) is the incremental rotation MATRIX; right-multiply because omega is in
            // the body frame (R_curr = R_prev * exp(w dt)). This is the multiplicative composition
            // that preserves the non-commutativity a vector sum would drop.
            R = R * ov_core::exp_so3(phi);

            if (s1.t >= t1) break;   // consumed up to t1
        }
        set_ok(true);
        return R;
    }

    // --- liveness/diagnostics (thread-safe reads) ---
    std::size_t size() const { std::lock_guard<std::mutex> lk(mtx_); return n_; }
    double t_oldest() const { std::lock_guard<std::mutex> lk(mtx_); return n_ ? at(0).t : 0.0; }
    double t_newest() const { std::lock_guard<std::mutex> lk(mtx_); return n_ ? at(n_ - 1).t : 0.0; }

private:
    struct Sample { double t, gx, gy, gz; };  // timestamp (s) + gyro (rad/s)

    // Chronological access: at(0) is the oldest valid sample, at(n_-1) the newest. Before the ring
    // wraps, samples sit at [0, n_); after it saturates, the oldest is at `head_` (the next write
    // slot), so we offset by head_ and take mod cap_.
    const Sample& at(std::size_t i) const {
        const std::size_t start = (n_ < cap_) ? 0 : head_;
        return buf_[(start + i) % cap_];
    }

    // Linear interpolation of the angular RATE at time t within [s0.t, s1.t]. f is the fractional
    // position of t in the interval; g = (1-f) w0 + f w1, per axis. (dt<=0 guard avoids /0.)
    static void interp(const Sample& s0, const Sample& s1, double t, Eigen::Vector3d& g) {
        const double dt = s1.t - s0.t;
        const double f = (dt > 1e-12) ? (t - s0.t) / dt : 0.0;
        g(0) = s0.gx + (s1.gx - s0.gx) * f;
        g(1) = s0.gy + (s1.gy - s0.gy) * f;
        g(2) = s0.gz + (s1.gz - s0.gz) * f;
    }

    std::vector<Sample> buf_;   // ring buffer of recent samples
    std::size_t cap_;           // ring capacity
    std::size_t head_ = 0;      // next write index (also the oldest sample once saturated)
    std::size_t n_ = 0;         // number of valid samples (<= cap_)
    Eigen::Vector3d bias_ = Eigen::Vector3d::Zero();
    mutable std::mutex mtx_;    // guards buf_/head_/n_/bias against concurrent add_gyro vs reads
};

}  // namespace mf_imu
