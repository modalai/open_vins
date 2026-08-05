/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: BITWISE oracle for the P3 chronological preintegration sweep.
 *
 * integrate_chain()'s reuse license is byte-exactness: the preint cache keys
 * on exact values and serves stored bytes, so the sweep must reproduce the
 * per-interval integrate() scan BIT-FOR-BIT — including boundary interpolants
 * at interior clone times (shared straddling pair), clones landing exactly ON
 * sample stamps, intervals with no interior samples, duplicate-stamp (dt = 0)
 * pairs, and streams ending before the last clone. A randomized battery over
 * those regimes asserts memcmp equality on EVERY emitted field; a mismatch of
 * one ulp anywhere is a FAIL. The whitener cache is pinned the same way: the
 * cache-fed factor constructor must carry the exact bytes of the legacy
 * self-factorizing constructor.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cpi/AciCalibPreint.h"
#include "cpi/PreintCache.h"
#include "solve/Factor_ImuAci3.h"

using namespace ov_zcalib;

static int g_fail = 0;
#define CHECK(cond, ...)                                                                                                             \
  do {                                                                                                                               \
    if (!(cond)) {                                                                                                                   \
      std::printf("FAIL: " __VA_ARGS__);                                                                                             \
      std::printf("\n");                                                                                                             \
      g_fail++;                                                                                                                      \
    }                                                                                                                                \
  } while (0)

namespace {

ImuIntrinsicModel random_model(std::mt19937 &rng) {
  std::normal_distribution<double> n01(0.0, 1.0);
  ImuIntrinsicModel m;
  m.calib_dw = m.calib_da = m.calib_RAtoI = true;
  m.calib_tg = false;
  m.dw << 1 + 0.02 * n01(rng), 0.01 * n01(rng), 1 + 0.02 * n01(rng), 0.01 * n01(rng), 0.01 * n01(rng), 1 + 0.02 * n01(rng);
  m.da << 1 + 0.02 * n01(rng), 0.01 * n01(rng), 1 + 0.02 * n01(rng), 0.01 * n01(rng), 0.01 * n01(rng), 1 + 0.02 * n01(rng);
  Eigen::Vector3d th(0.02 * n01(rng), 0.02 * n01(rng), 0.02 * n01(rng));
  m.q_AtoI << 0.5 * th, 1.0;
  m.q_AtoI /= m.q_AtoI.norm();
  return m;
}

std::vector<RawImu> random_stream(std::mt19937 &rng, double t0, double t1, double rate_hz, bool jitter, bool dups) {
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::normal_distribution<double> n01(0.0, 1.0);
  std::vector<RawImu> imu;
  const double dt = 1.0 / rate_hz;
  for (double t = t0; t <= t1; ) {
    RawImu s;
    s.timestamp = t;
    s.wm = Eigen::Vector3d(0.6 * std::sin(2.1 * t), -0.4 * std::cos(1.3 * t), 0.5 * std::sin(3.7 * t + 0.4)) +
           0.02 * Eigen::Vector3d(n01(rng), n01(rng), n01(rng));
    s.am = Eigen::Vector3d(0.8 * std::sin(1.7 * t + 1.0), 9.6 + 0.5 * std::cos(2.3 * t), 1.1 * std::sin(0.9 * t)) +
           0.05 * Eigen::Vector3d(n01(rng), n01(rng), n01(rng));
    imu.push_back(s);
    if (dups && u01(rng) < 0.03) {
      RawImu d = s; // duplicate stamp, different values: a dt=0 pair the step loop must skip
      d.wm += 0.01 * Eigen::Vector3d(n01(rng), n01(rng), n01(rng));
      imu.push_back(d);
    }
    t += jitter ? dt * (0.6 + 0.8 * u01(rng)) : dt;
  }
  return imu;
}

bool run_case(std::mt19937 &rng, int n_clones, bool on_samples, bool jitter, bool dups, bool tight, const char *tag) {
  const double T0 = 10.0, T1 = 16.0;
  std::vector<RawImu> imu = random_stream(rng, T0, T1, 500.0, jitter, dups);
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  // clone grid inside the padded stream (pad both ends so boundaries clamp)
  std::vector<double> ct;
  if (on_samples) {
    // clones exactly ON sample stamps (plus one interpolated in the middle)
    std::uniform_int_distribution<int> pick(5, (int)imu.size() - 6);
    std::vector<int> idx;
    while ((int)idx.size() < n_clones) {
      int i = pick(rng);
      bool used = false;
      for (int j : idx)
        used |= (j == i);
      if (!used)
        idx.push_back(i);
    }
    std::sort(idx.begin(), idx.end());
    for (int i : idx)
      ct.push_back(imu[i].timestamp);
    ct[ct.size() / 2] += 1e-4; // one boundary forced OFF-sample
    std::sort(ct.begin(), ct.end());
  } else if (tight) {
    // sub-sample-gap intervals: consecutive clones inside ONE sample gap
    const double base = T0 + 2.0 + 0.5 * u01(rng);
    ct = {base, base + 2.2e-4, base + 5.1e-4, base + 0.3, base + 0.3 + 3.3e-4, base + 1.1};
  } else {
    double t = T0 + 0.5 + 0.2 * u01(rng);
    for (int k = 0; k < n_clones; ++k) {
      ct.push_back(t);
      t += 0.05 + 0.4 * u01(rng);
    }
  }
  const int N = (int)ct.size();
  ImuIntrinsicModel model = random_model(rng);
  ImuIntrinsicModel noise_lin = random_model(rng); // frozen whitener at DIFFERENT values
  ImuNoise noise;
  const Eigen::Vector3d bg(0.002, -0.001, 0.0015), ba(0.03, -0.02, 0.01);

  for (int frozen = 0; frozen < 2; ++frozen) {
    const ImuIntrinsicModel *nm = frozen ? &noise_lin : nullptr;
    std::vector<AciPreintResult> ref(N - 1);
    bool ref_ok = true;
    for (int k = 0; k + 1 < N; ++k)
      ref_ok = ref_ok && AciCalibPreint::integrate(imu, ct[k], ct[k + 1], model, bg, ba, noise, ref[k], nm);
    std::vector<AciPreintResult> chain;
    const bool chain_ok = AciCalibPreint::integrate_chain(imu, ct, model, bg, ba, noise, chain, nm);
    CHECK(ref_ok == chain_ok, "%s frozen=%d: ok mismatch (ref %d chain %d)", tag, frozen, ref_ok, chain_ok);
    if (!ref_ok || !chain_ok)
      continue;
    CHECK((int)chain.size() == N - 1, "%s frozen=%d: size", tag, frozen);
    for (int k = 0; k + 1 < N; ++k)
      CHECK(preint_bitwise_equal(ref[k], chain[k]), "%s frozen=%d: interval %d NOT bit-identical", tag, frozen, k);

    // skip_cov: mean/Jacobian fields bit-identical, P15 untouched (default)
    std::vector<AciPreintResult> nocov;
    CHECK(AciCalibPreint::integrate_chain(imu, ct, model, bg, ba, noise, nocov, nm, true), "%s frozen=%d: skip_cov ok", tag, frozen);
    for (int k = 0; k + 1 < N; ++k) {
      AciPreintResult a = chain[k], b = nocov[k];
      a.P15.setIdentity();
      b.P15.setIdentity(); // exclude covariance from the comparison
      CHECK(preint_bitwise_equal(a, b), "%s frozen=%d: skip_cov interval %d mean fields differ", tag, frozen, k);
      const Eigen::Matrix<double, 15, 15> eye = Eigen::Matrix<double, 15, 15>::Identity();
      CHECK(std::memcmp(nocov[k].P15.data(), eye.data(), 225 * sizeof(double)) == 0, "%s frozen=%d: skip_cov touched P15", tag, frozen);
    }
  }
  return true;
}

} // namespace

int main() {
  std::mt19937 rng(20260711);

  // ---- randomized sweep battery over the boundary regimes ----
  for (int rep = 0; rep < 6; ++rep) {
    run_case(rng, 24, false, true, true, false, "interior-jitter-dups");
    run_case(rng, 24, false, false, false, false, "interior-uniform");
    run_case(rng, 16, true, true, false, false, "clones-on-samples");
    run_case(rng, 0, false, true, true, true, "sub-gap-intervals");
  }

  // ---- failure parity: clones beyond the stream must fail in BOTH paths ----
  {
    std::vector<RawImu> imu = random_stream(rng, 0.0, 1.0, 400.0, true, false);
    std::vector<double> ct = {0.2, 0.6, 1.4}; // last interval leaves the stream
    ImuIntrinsicModel model = random_model(rng);
    ImuNoise noise;
    AciPreintResult r0, r1;
    const bool ok0 = AciCalibPreint::integrate(imu, ct[0], ct[1], model, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise, r0);
    const bool ok1 = AciCalibPreint::integrate(imu, ct[1], ct[2], model, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise, r1);
    std::vector<AciPreintResult> chain;
    const bool okc = AciCalibPreint::integrate_chain(imu, ct, model, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise, chain);
    // per-interval integrate: [0.2,0.6] fine, [0.6,1.4]: stream ends at 1.0 —
    // integrate() still succeeds on the spanned part; the CHAIN must agree
    // with the AND of the two, and when it succeeds, bytes must match.
    CHECK(okc == (ok0 && ok1), "failure parity: chain %d vs per-interval %d&%d", okc, ok0, ok1);
    if (okc) {
      CHECK(preint_bitwise_equal(r0, chain[0]), "spanned interval 0 bytes");
      CHECK(preint_bitwise_equal(r1, chain[1]), "spanned interval 1 bytes");
    }
    // degenerate clone order
    std::vector<double> bad = {0.5, 0.5};
    CHECK(!AciCalibPreint::integrate_chain(imu, bad, model, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise, chain),
          "degenerate interval must fail");
  }

  // ---- whitener: the cache-fill discipline is COPY-from-a-legacy-factor
  // (WindowBA miss path), never a re-derivation — a separate implementation in
  // a different codegen context drifted 1 ulp under -ffast-math (2026-07-11).
  // This pins the full round trip: legacy ctor -> copied bytes -> cache-fed
  // ctor == legacy ctor, plus the whitener's defining identity.
  {
    std::vector<RawImu> imu = random_stream(rng, 5.0, 8.0, 500.0, true, false);
    std::vector<double> ct = {5.3, 5.9, 6.6, 7.4};
    ImuIntrinsicModel model = random_model(rng);
    ImuNoise noise;
    std::vector<AciPreintResult> pre;
    CHECK(AciCalibPreint::integrate_chain(imu, ct, model, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise, pre),
          "whitener battery: chain failed");
    for (size_t k = 0; k < pre.size(); ++k) {
      Factor_ImuAci3 legacy(pre[k], model, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
      // the WindowBA miss path: harvest the constructed factor's bytes
      const Eigen::Matrix<double, 15, 15> W = legacy.sqrtI;
      const Eigen::Matrix<double, 15, 3> fold = legacy.sqrtI_grav_fold;
      Factor_ImuAci3 cached(pre[k], model, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), W, fold);
      CHECK(std::memcmp(legacy.sqrtI.data(), cached.sqrtI.data(), 225 * sizeof(double)) == 0, "sqrtI bytes interval %zu", k);
      CHECK(std::memcmp(legacy.sqrtI_grav_fold.data(), cached.sqrtI_grav_fold.data(), 45 * sizeof(double)) == 0,
            "grav fold bytes interval %zu", k);
      // invariance sanity: ||W r||^2 == r' P^-1 r (the whitener's defining identity)
      Eigen::Matrix<double, 15, 1> r;
      std::normal_distribution<double> n01(0.0, 1.0);
      for (int i = 0; i < 15; ++i)
        r(i) = n01(rng);
      const double lhs = (W * r).squaredNorm();
      const double rhs = r.dot(pre[k].P15.llt().solve(r));
      CHECK(std::abs(lhs - rhs) <= 1e-9 * std::max(1.0, std::abs(rhs)), "whitener identity interval %zu: %.3e vs %.3e", k, lhs, rhs);
    }
  }

  if (g_fail == 0)
    std::printf("test_preint_chain: ALL PASS (sweep bit-parity + whitener bytes pinned)\n");
  else
    std::printf("test_preint_chain: %d FAILURES\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
