/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: the CALIBRATOR profile (ov_zcalib.yaml) -- deliberately a separate file
 * from estimator_config.yaml, because the two answer different questions:
 *
 *   estimator_config.yaml  what the FILTER believes at boot and refines ONLINE.
 *   ov_zcalib.yaml          what the CALIBRATOR is GIVEN versus what it must EARN.
 *
 * The seeding defaults retain a historical single-session comparison on one
 * reference rig, with all other settings held fixed:
 *
 *   seed cam intrinsics   VERIFY 62.2%
 *   + seed imu intrinsics VERIFY 62.2 -> 73.3%
 *   + seed R_ItoC         byte-identical result after hand-eye bootstrap
 *   + seed p_IinC         VERIFY 73.3 -> 48.6%
 *   + seed td             VERIFY      -> 28.3%
 *
 * VERIFY measures held-out reprojection improvement, not calibration ground truth.
 * An active chain is an initializer, not proof that its per-unit values are correct.
 * To start the full IMU chain blind, disable BOTH seed_imu_intrinsics and seed_tg;
 * their estimation policies remain separate. Defaults are unchanged.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_CALIB_CONFIG_YAML_H
#define OV_ZCALIB_CALIB_CONFIG_YAML_H

#include <cstdint>
#include <cstring>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include "utils/print.h"
#include "utils/NumericChecks.h"

#include "CalibSessionRunner.h"
#include "SessionProfiles.h"

namespace ov_zcalib {

/// What the session is GIVEN. Everything false here must be EARNED from the data.
struct SeedPolicy {
  bool cam_intrinsics = true;    ///< camera intrinsic initializer from the active chain
  bool imu_intrinsics = true;    ///< Dw/Da/R_AtoI, gauge-ported from the chain (kalibr <-> imu2)
  bool extrinsic_rotation = false; ///< hand-eye bootstrap re-solves the rotation
  bool extrinsic_position = false; ///< zero unless the chain's lever arm is requested
  bool time_offset = false;        ///< zero unless the chain's clock offset is requested
  // NOTE: no readout knob. tr is a hardware fact (HAL3 sensor mode) that ALWAYS seeds from the
  // estimator config's camN_readout_time_s and is never estimated -- there is no policy to set.
  /// Tg initialization is independent of imu_intrinsics: true uses the gauge-ported
  /// chain, false starts at zero. Neither choice disables estimate_tg.
  bool tg = true;
};

/// Apply independent seed choices AFTER converting the chain into the imu2 gauge.
/// This changes parameter values only; raw sensor samples are never transformed here.
inline ImuIntrinsicModel make_imu_seed(const ImuIntrinsicModel &ported_chain, const SeedPolicy &policy) {
  ImuIntrinsicModel seed = ported_chain;
  if (!policy.imu_intrinsics) {
    const ImuIntrinsicModel identity;
    seed.dw = identity.dw;
    seed.da = identity.da;
    seed.q_AtoI = identity.q_AtoI;
  }
  if (!policy.tg)
    seed.Tg.setZero();
  return seed;
}

/// What the session may ESTIMATE and COMMIT. A block that is seeded but NOT estimated
/// ships its seed and is reported as such (see the YAML's committed_blocks/seed_blocks).
struct EstimatePolicy {
  bool imu_intrinsics = true; ///< false => seed AND FREEZE: skips A1a/A1b entirely (~30% faster solve)
  int cam_intrinsics = 0;     ///< 0 fixed | 1 refine (tight priors) | 2 full (weak priors, gated)
  /// Estimate Tg (gyro g-sensitivity). EARNED per unit: the rig's own kalibr sessions scatter
  /// beyond |Tg| between runs, so no chain value deserves seed authority.
  /// Unlocks only through the A1b excitation gate + split-half/Wald falsifier; requires an
  /// estimable IMU chain (a frozen factory chain freezes tg with it).
  bool tg = true;
};

/// Estimation permission is independent of where parameter values originated.
inline void apply_imu_estimate_policy(ImuIntrinsicModel &imu, const EstimatePolicy &policy) {
  if (!policy.imu_intrinsics)
    imu.calib_dw = imu.calib_da = imu.calib_RAtoI = imu.calib_tg = false;
}

/// The whole calibrator profile: window shape + gates (SessionConfig), what is given
/// (SeedPolicy), what is earned (EstimatePolicy), and the session's own budgets.
struct CalibProfile {
  SessionConfig session;
  SeedPolicy seed;
  EstimatePolicy estimate;
  /// Calibration-WEIGHTING IMU densities -- deliberately not the filter's. sigma here is not a
  /// safety margin, it is the lever arm between the IMU and camera residuals: inflate the IMU and
  /// the gyro loses every argument with a camera, including the one about dw. See
  /// apply_voxl_noise_defaults().
  ImuNoise noise;
  /// Calibration reprojection residual 1-sigma, in pixels per image axis.
  /// Independent of MSCKF/SLAM tuning and of lens-calibration reprojection RMS.
  double camera_pixel_sigma = 1.0;
  std::map<std::string, double> camera_pixel_sigma_by_name;
  /// Fixed magnitude for calibration only (m/s^2). Zero uses the loaded estimator
  /// value without modifying it. Gravity direction remains an estimated S2 state,
  /// as in Kalibr; a free norm would trade against the accelerometer scale.
  double gravity_mag = 0.0;
  /// Frames per second handed to the TRACKER, per camera (0 = the sensor's own rate). The calibrator
  /// clones far less than this -- the harvester subsamples each window to max_clones/n_cams -- so
  /// everything above what KLT needs is tracked at full cost and then discarded. On a multi-camera
  /// rig that waste is what saturates the (serialized, inline) tracker and sheds frames, and shed
  /// frames come back as destroyed windows. VIO itself tracks at 20-30 Hz on these rigs.
  double track_rate_hz = 30.0;
  std::string base = "flight"; ///< "flight" (thin, <=60 s class) | "bench" (full shape, kalibr-grade)
  double collect_budget_s = 40.0;    ///< live cutover ceiling (a live stream has no EOF)
  double collect_min_eig = 5.0;      ///< early cutover once the WEAKEST direction is this well determined
  bool cpu_performance_mode = true;  ///< measured 3.1x on the solve; restored on exit
  /// Write the session record mirror (<out>.session.bin). ON by default: the record is the
  /// replay/forensics contract (a session replays bit-identically on the same binary).
  /// Disable only where storage forbids it -- an unrecorded session cannot be replayed,
  /// cross-checked across platforms, or examined after the fact.
  bool record_session = true;
};

inline bool valid_camera_pixel_sigma(double sigma) {
  uint64_t bits;
  static_assert(sizeof(bits) == sizeof(sigma), "IEEE-754 double required");
  std::memcpy(&bits, &sigma, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000) && sigma > 0.0;
}

/// Strict parsing is intentional: a misspelled camera weight must not silently
/// turn into the default. OpenCV keeps duplicate map keys when iterating nodes.
inline bool parse_camera_pixel_noise(const cv::FileNode &root, CalibProfile &out, std::string &error) {
  double fallback = 1.0;
  std::map<std::string, double> overrides;
  std::set<std::string> seen;
  for (const auto &node : root) {
    const std::string key = node.name();
    if (key != "camera_pixel_sigma" && key != "camera_pixel_sigma_by_name")
      continue;
    if (!seen.insert(key).second) { error = "duplicate key '" + key + "'"; return false; }
    if (key == "camera_pixel_sigma") {
      if (!node.isInt() && !node.isReal()) { error = key + " must be a numeric scalar"; return false; }
      fallback = (double)node;
      if (!valid_camera_pixel_sigma(fallback)) { error = key + " must be finite and greater than zero"; return false; }
    } else {
      if (!node.isMap()) { error = key + " must be a map from canonical camera names to pixel sigmas"; return false; }
      for (const auto &entry : node) {
        const std::string name = entry.name();
        if (name.empty() || (!entry.isInt() && !entry.isReal())) {
          error = key + ": each camera name must have a numeric scalar"; return false;
        }
        const double sigma = (double)entry;
        if (!valid_camera_pixel_sigma(sigma)) {
          error = key + "." + name + " must be finite and greater than zero"; return false;
        }
        if (!overrides.emplace(name, sigma).second) { error = key + ": duplicate camera '" + name + "'"; return false; }
      }
    }
  }
  out.camera_pixel_sigma = fallback;
  out.camera_pixel_sigma_by_name = std::move(overrides);
  return true;
}

/// Resolve by identity, never by the order selected. The caller supplies the
/// declared-camera registry so configured but unselected cameras can be reported.
inline bool resolve_camera_pixel_noise(const CalibProfile &profile,
    const std::vector<std::pair<size_t, std::string>> &selected,
    const std::set<std::string> &known_names, std::map<size_t, double> &resolved,
    std::vector<std::string> &unused, std::string &error) {
  resolved.clear();
  unused.clear();
  if (!valid_camera_pixel_sigma(profile.camera_pixel_sigma)) {
    error = "camera_pixel_sigma must be finite and greater than zero"; return false;
  }
  std::set<std::string> selected_names;
  for (const auto &camera : selected) {
    if (camera.second.empty() || !selected_names.insert(camera.second).second || resolved.count(camera.first)) {
      error = "selected cameras must have unique, nonempty canonical names and unique IDs"; return false;
    }
    const auto override = profile.camera_pixel_sigma_by_name.find(camera.second);
    resolved.emplace(camera.first, override == profile.camera_pixel_sigma_by_name.end() ? profile.camera_pixel_sigma : override->second);
  }
  for (const auto &override : profile.camera_pixel_sigma_by_name) {
    if (!valid_camera_pixel_sigma(override.second)) {
      error = "camera_pixel_sigma_by_name." + override.first + " must be finite and greater than zero"; return false;
    }
    if (selected_names.count(override.first))
      continue;
    if (!known_names.count(override.first)) {
      error = "camera_pixel_sigma_by_name: unknown camera '" + override.first + "' (use the canonical camera name, not cam0/cam1 or a declared camera's pipe alias)";
      return false;
    }
    unused.push_back(override.first);
  }
  return true;
}

// Read policy values without silent defaults or lossy coercions for malformed
// values. Missing keys retain the selected profile.
inline bool calib_profile_node(const cv::FileNode &root, const char *key, cv::FileNode &value, std::string &error) {
  value = cv::FileNode();
  bool found = false;
  for (const auto &node : root) {
    if (node.name() != key)
      continue;
    if (found) { error = std::string("duplicate key '") + key + "'"; return false; }
    found = true;
    value = node;
  }
  return true;
}

inline bool read_calib_profile_number(const cv::FileNode &root, const char *key, double &value, std::string &error) {
  cv::FileNode node;
  if (!calib_profile_node(root, key, node, error))
    return false;
  if (node.isNone())
    return true;
  if (!node.isInt() && !node.isReal()) { error = std::string(key) + " must be a numeric scalar"; return false; }
  const double parsed = (double)node;
  if (!finite_scalar(parsed)) { error = std::string(key) + " must be finite"; return false; }
  value = parsed;
  return true;
}

inline bool read_calib_profile_bool(const cv::FileNode &root, const char *key, bool &value, std::string &error) {
  cv::FileNode node;
  if (!calib_profile_node(root, key, node, error))
    return false;
  if (node.isNone())
    return true;
  if (node.isInt()) {
    const int parsed = (int)node;
    if (parsed == 0 || parsed == 1) { value = parsed == 1; return true; }
  } else if (node.isString()) {
    // OpenCV retains ordinary trailing comments in unquoted string nodes.
    std::string parsed = (std::string)node;
    parsed = parsed.substr(0, parsed.find('#'));
    while (!parsed.empty() && std::isspace(static_cast<unsigned char>(parsed.back()))) parsed.pop_back();
    if (parsed == "true" || parsed == "True" || parsed == "TRUE" || parsed == "1") { value = true; return true; }
    if (parsed == "false" || parsed == "False" || parsed == "FALSE" || parsed == "0") { value = false; return true; }
  }
  error = std::string(key) + " must be true/false (also True/False or TRUE/FALSE) or 0/1";
  return false;
}

inline bool read_calib_profile_count(const cv::FileNode &root, const char *key, int &value, std::string &error) {
  double parsed = value;
  if (!read_calib_profile_number(root, key, parsed, error))
    return false;
  // Check range BEFORE the cast; out-of-range/NaN-to-int conversion is unsafe.
  if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
    error = std::string(key) + " must be an integer in the supported range";
    return false;
  }
  const int integral = (int)parsed;
  if (parsed != (double)integral) { error = std::string(key) + " must be an integer"; return false; }
  value = integral;
  return true;
}

// OpenCV stores YAML integer literals in int32 before FileNode can expose them:
// 4294967300 becomes 4. Check their original spelling for the numeric keys we
// actually consume. Grammar/type checks remain OpenCV's job; this only prevents
// lossy integer conversion. Decimal, octal, hex, !!int and flow maps retain their
// existing meanings. A real/scientific literal is validated as a double later.
inline bool calib_profile_integer_literals(const std::string &text, const std::set<std::string> &keys, std::string &error) {
  std::size_t line_start = 0, root_indent = std::string::npos;
  int braces = 0, brackets = 0;
  bool root_flow = false;
  const auto space = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
  for (std::size_t i = 0; i < text.size();) {
    const char c = text[i];
    if (c == '\n') { line_start = ++i; continue; }
    if (space(c)) { ++i; continue; }
    if (c == '#' || c == '%') {
      while (i < text.size() && text[i] != '\n') ++i;
      continue;
    }
    if (root_indent == std::string::npos && text.compare(i, 3, "---") == 0) { i += 3; continue; }
    if (root_indent == std::string::npos) {
      root_indent = i - line_start;
      root_flow = c == '{';
    }
    if (c == '{') { ++braces; ++i; continue; }
    if (c == '}') { --braces; ++i; continue; }
    if (c == '[') { ++brackets; ++i; continue; }
    if (c == ']') { --brackets; ++i; continue; }
    const std::size_t start = i;
    std::string token;
    if (c == '\'' || c == '"') {
      const char quote = c;
      ++i;
      while (i < text.size()) {
        if (text[i] == '\\' && quote == '"' && i + 1 < text.size()) { i += 2; continue; }
        if (text[i] == quote) {
          if (quote == '\'' && i + 1 < text.size() && text[i + 1] == quote) { i += 2; continue; }
          break;
        }
        ++i;
      }
      token = text.substr(start + 1, i - start - 1);
      if (i < text.size()) ++i;
    } else if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
      while (i < text.size() && (std::isalnum(static_cast<unsigned char>(text[i])) || text[i] == '_')) ++i;
      token = text.substr(start, i - start);
    } else { ++i; continue; }
    const bool top = brackets == 0 && (root_flow ? braces == 1 : braces == 0 && start - line_start == root_indent);
    if (!top || keys.count(token) == 0)
      continue;
    std::size_t value = i;
    while (value < text.size() && space(text[value])) ++value;
    if (value == text.size() || text[value++] != ':')
      continue;
    while (value < text.size() && space(text[value])) ++value;
    if (text.compare(value, 2, "!!") == 0) {
      while (value < text.size() && !space(text[value])) ++value;
      while (value < text.size() && space(text[value])) ++value;
    }
    std::size_t end = value;
    while (end < text.size() && !space(text[end]) && text[end] != ',' && text[end] != '}' && text[end] != ']' && text[end] != '#') ++end;
    const std::string literal = text.substr(value, end - value);
    if (literal.empty())
      continue;
    char *after = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(literal.c_str(), &after, 0);
    if (after == literal.c_str() + literal.size() &&
        (errno == ERANGE || parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max())) {
      error = token + " integer literal is outside OpenCV's supported range; use a representable value";
      return false;
    }
  }
  return true;
}

inline bool parse_calib_profile_bools(const cv::FileNode &root, const std::string &source, CalibProfile &p, std::string &error) {
  const std::pair<const char *, bool *> flags[] = {
      {"seed_cam_intrinsics", &p.seed.cam_intrinsics}, {"seed_imu_intrinsics", &p.seed.imu_intrinsics},
      {"seed_extrinsic_rotation", &p.seed.extrinsic_rotation}, {"seed_extrinsic_position", &p.seed.extrinsic_position},
      {"seed_time_offset", &p.seed.time_offset}, {"seed_tg", &p.seed.tg},
      {"estimate_imu_intrinsics", &p.estimate.imu_intrinsics}, {"estimate_tg", &p.estimate.tg},
      {"cpu_performance_mode", &p.cpu_performance_mode}, {"record_session", &p.record_session},
      {"stage_select", &p.session.stage_select}, {"tg_precision_screen", &p.session.tg_precision_screen},
      {"bootstrap_epipolar", &p.session.bootstrap_epipolar}};
  std::set<std::string> keys;
  for (const auto &entry : flags) keys.insert(entry.first);
  if (!calib_profile_integer_literals(source, keys, error))
    return false;
  for (const auto &entry : flags)
    if (!read_calib_profile_bool(root, entry.first, *entry.second, error))
      return false;
  p.session.free_tg = p.estimate.tg;
  return true;
}

/// Validate the effective profile AFTER its base overlay and explicit overrides.
/// Zero densities are permitted (zero diffusion), as are the documented zero/off
/// settings. These are input domains, not new calibration acceptance thresholds.
inline bool validate_calib_profile(const CalibProfile &p, std::string &error) {
  if (p.base != "flight" && p.base != "bench") { error = "profile must be exactly 'flight' or 'bench'"; return false; }
  const std::pair<const char *, double> nonnegative[] = {
      {"gyroscope_noise_density", p.noise.sigma_w}, {"gyroscope_random_walk", p.noise.sigma_wb},
      {"accelerometer_noise_density", p.noise.sigma_a}, {"accelerometer_random_walk", p.noise.sigma_ab},
      {"solve_budget_s", p.session.solve_budget_s}, {"collect_min_eig", p.collect_min_eig},
      {"track_rate_hz", p.track_rate_hz}};
  for (const auto &entry : nonnegative)
    if (!finite_scalar(entry.second) || entry.second < 0.0) {
      error = std::string(entry.first) + " must be finite and nonnegative"; return false;
    }
  const std::pair<const char *, double> positive[] = {
      {"collect_budget_s", p.collect_budget_s}, {"commit_sigma_factor", p.session.commit_sigma_factor},
      {"radtan_tangent_refine_sigma", p.session.radtan_tangent_refine_sigma},
      {"radtan_tangent_full_sigma", p.session.radtan_tangent_full_sigma}};
  for (const auto &entry : positive)
    if (!finite_scalar(entry.second) || !(entry.second > 0.0)) {
      error = std::string(entry.first) + " must be finite and greater than zero"; return false;
    }
  if (!finite_scalar(p.gravity_mag) || !(p.gravity_mag == 0.0 || (p.gravity_mag >= 9.0 && p.gravity_mag <= 10.0))) {
    error = "gravity_mag must be 0 (inherit) or between 9 and 10 m/s^2"; return false;
  }
  if (!finite_scalar(p.session.verify_min_improve) || p.session.verify_min_improve < 0.0 || p.session.verify_min_improve > 1.0) {
    error = "verify_min_improve must be finite and between 0 and 1"; return false;
  }
  if (p.estimate.cam_intrinsics < 0 || p.estimate.cam_intrinsics > 2) {
    error = "estimate_cam_intrinsics must be 0, 1 or 2"; return false;
  }
  if (p.session.joint.num_threads < 0) { error = "num_threads must be nonnegative (0 or 1 runs inline)"; return false; }
  if (p.session.harvester.max_clones < p.session.harvester.min_clones) {
    error = "max_clones must be at least min_clones (" + std::to_string(p.session.harvester.min_clones) + ")"; return false;
  }
  if (p.session.harvester.max_track_len < p.session.harvester.min_track_len) {
    error = "max_track_len must be at least min_track_len (" + std::to_string(p.session.harvester.min_track_len) + ")"; return false;
  }
  if (p.session.select_K < 1) { error = "select_k must be positive"; return false; }
  if (p.session.min_holdout < 0) { error = "min_holdout must be nonnegative"; return false; }
  if (p.session.stage_select && (p.session.select_K_a0 < 0 || p.session.select_K_a1 < 0 || p.session.select_K_b < 0)) {
    error = "select_k_a0/select_k_a1/select_k_b must be nonnegative (0 inherits select_k)"; return false;
  }
  return true;
}

inline bool parse_calib_profile_numbers(const cv::FileNode &root, const std::string &source, CalibProfile &p, std::string &error) {
  const std::pair<const char *, double *> numbers[] = {
      {"gyroscope_noise_density", &p.noise.sigma_w}, {"gyroscope_random_walk", &p.noise.sigma_wb},
      {"accelerometer_noise_density", &p.noise.sigma_a}, {"accelerometer_random_walk", &p.noise.sigma_ab},
      {"gravity_mag", &p.gravity_mag}, {"track_rate_hz", &p.track_rate_hz},
      {"collect_budget_s", &p.collect_budget_s}, {"collect_min_eig", &p.collect_min_eig},
      {"solve_budget_s", &p.session.solve_budget_s}, {"commit_sigma_factor", &p.session.commit_sigma_factor},
      {"verify_min_improve", &p.session.verify_min_improve},
      {"radtan_tangent_refine_sigma", &p.session.radtan_tangent_refine_sigma},
      {"radtan_tangent_full_sigma", &p.session.radtan_tangent_full_sigma}};
  const std::pair<const char *, int *> counts[] = {
      {"estimate_cam_intrinsics", &p.estimate.cam_intrinsics}, {"num_threads", &p.session.joint.num_threads},
      {"max_clones", &p.session.harvester.max_clones}, {"max_track_len", &p.session.harvester.max_track_len},
      {"select_k", &p.session.select_K}, {"min_holdout", &p.session.min_holdout}};
  std::set<std::string> numeric_keys;
  for (const auto &entry : numbers) numeric_keys.insert(entry.first);
  for (const auto &entry : counts) numeric_keys.insert(entry.first);
  if (p.session.stage_select)
    for (const char *key : {"select_k_a0", "select_k_a1", "select_k_b"}) numeric_keys.insert(key);
  if (!calib_profile_integer_literals(source, numeric_keys, error))
    return false;
  for (const auto &entry : numbers)
    if (!read_calib_profile_number(root, entry.first, *entry.second, error))
      return false;
  for (const auto &entry : counts)
    if (!read_calib_profile_count(root, entry.first, *entry.second, error))
      return false;
  // Disabled stage selection does not consume these overrides. Do not parse a
  // dormant NaN/fractional count through OpenCV's integer conversion either.
  if (p.session.stage_select) {
    const std::pair<const char *, int *> stages[] = {{"select_k_a0", &p.session.select_K_a0},
        {"select_k_a1", &p.session.select_K_a1}, {"select_k_b", &p.session.select_K_b}};
    for (const auto &entry : stages)
      if (!read_calib_profile_count(root, entry.first, *entry.second, error))
        return false;
  }
  p.session.cam_mode = p.estimate.cam_intrinsics;
  return validate_calib_profile(p, error);
}

/// Load a calibrator profile. A MISSING FILE IS NOT AN ERROR: the built-in defaults are the
/// measured-good configuration, and the YAML exists to override them. Returns false only if
/// the file exists but cannot be parsed, so a typo is loud rather than silently ignored.
inline bool load_calib_profile(const std::string &path, CalibProfile &out) {
  // ---- built-in defaults first (the measured-good configuration) ----
  out = CalibProfile();
  apply_voxl_profile(out.session);
  apply_flight_overlay(out.session, /*set_cam_mode0=*/true);
  apply_voxl_noise_defaults(out.noise);

  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) {
    PRINT_INFO("[ov_zcalib] no profile at %s -- using built-in defaults (seed intrinsics, earn ext/td)\n", path.c_str());
    return true;
  }
  std::string source;
  char buffer[4096];
  std::size_t nread = 0;
  while ((nread = std::fread(buffer, 1, sizeof(buffer), f)) != 0)
    source.append(buffer, nread);
  const bool read_ok = std::ferror(f) == 0;
  std::fclose(f);
  if (!read_ok) {
    PRINT_ERROR("[ov_zcalib] cannot read profile %s\n", path.c_str());
    return false;
  }

  cv::FileStorage profile_yaml;
  std::string error;
  try {
    profile_yaml.open(source, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    if (!profile_yaml.isOpened() || !parse_camera_pixel_noise(profile_yaml.root(), out, error)) {
      PRINT_ERROR("[ov_zcalib] invalid camera weighting in %s: %s\n", path.c_str(), error.c_str());
      return false;
    }
  } catch (const cv::Exception &e) {
    PRINT_ERROR("[ov_zcalib] cannot parse profile %s: %s\n", path.c_str(), e.what());
    return false;
  }

  // ---- base shape ----
  cv::FileNode base;
  if (!calib_profile_node(profile_yaml.root(), "profile", base, error) ||
      (!base.isNone() && !base.isString())) {
    PRINT_ERROR("[ov_zcalib] invalid profile in %s: %s\n", path.c_str(), error.empty() ? "profile must be a string" : error.c_str());
    return false;
  }
  if (!base.isNone())
    out.base = (std::string)base;
  if (out.base != "flight" && out.base != "bench") {
    PRINT_ERROR("[ov_zcalib] invalid profile '%s' in %s: expected exactly 'flight' or 'bench'\n", out.base.c_str(), path.c_str());
    return false;
  }
  out.session = SessionConfig();
  apply_voxl_profile(out.session);
  if (out.base == "flight")
    apply_flight_overlay(out.session, /*set_cam_mode0=*/true);

  // All options use the same file snapshot. Present malformed booleans are
  // errors even when optional; a typo must not silently preserve a true seed.
  if (!parse_calib_profile_bools(profile_yaml.root(), source, out, error)) {
    PRINT_ERROR("[ov_zcalib] invalid boolean profile in %s: %s\n", path.c_str(), error.c_str());
    return false;
  }
  if (!parse_calib_profile_numbers(profile_yaml.root(), source, out, error)) {
    PRINT_ERROR("[ov_zcalib] invalid numeric profile in %s: %s\n", path.c_str(), error.c_str());
    return false;
  }
  if (!out.record_session)
    PRINT_WARNING("[ov_zcalib] record_session=false -- the session record mirror is DISABLED (session will not be replayable)\n");

  PRINT_INFO("[ov_zcalib] profile '%s' from %s\n", out.base.c_str(), path.c_str());
  PRINT_INFO("[ov_zcalib]   SEED   cam_intr=%d imu_intr=%d ext_R=%d ext_p=%d td=%d tg=%d (tr: HAL3 hardware, always)\n",
             (int)out.seed.cam_intrinsics, (int)out.seed.imu_intrinsics, (int)out.seed.extrinsic_rotation,
             (int)out.seed.extrinsic_position, (int)out.seed.time_offset, (int)out.seed.tg);
  PRINT_INFO("[ov_zcalib]   EARN   imu_intr=%d cam_mode=%d tg=%d   (everything not seeded is blind; tr never earned)\n",
             (int)out.estimate.imu_intrinsics, out.estimate.cam_intrinsics, (int)out.estimate.tg);
  PRINT_INFO("[ov_zcalib]   WEIGHT gyro %.4e/%.4e  accel %.4e/%.4e  (nd/rw -- calibration weighting, NOT the filter's)\n",
             out.noise.sigma_w, out.noise.sigma_wb, out.noise.sigma_a, out.noise.sigma_ab);
  PRINT_INFO("[ov_zcalib]   CAMERA WEIGHT default %.6g px, %zu named overrides (calibration only; independent of MSCKF/SLAM)\n",
             out.camera_pixel_sigma, out.camera_pixel_sigma_by_name.size());
  return true;
}

} // namespace ov_zcalib

#endif // OV_ZCALIB_CALIB_CONFIG_YAML_H
