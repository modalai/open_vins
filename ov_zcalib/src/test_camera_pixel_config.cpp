#include "core/CalibConfigYaml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unistd.h>

using namespace ov_zcalib;

int main() {
  int failures = 0;
  auto check = [&](bool ok, const char *why) { if (!ok) { ++failures; std::printf("FAIL: %s\n", why); } };
  auto parse = [](const std::string &body, CalibProfile &profile, std::string &error) {
    try {
      cv::FileStorage yaml("%YAML:1.0\n---\n" + body, cv::FileStorage::READ | cv::FileStorage::MEMORY);
      return parse_camera_pixel_noise(yaml.root(), profile, error);
    } catch (const cv::Exception &e) { error = e.what(); return false; }
  };
  CalibProfile p;
  std::string error;
  check(parse("profile: flight\n", p, error) && p.camera_pixel_sigma == 1.0 && p.camera_pixel_sigma_by_name.empty(),
        "existing profiles must retain the independent 1px batch default");
  check(parse("camera_pixel_sigma: 1.25\ncamera_pixel_sigma_by_name:\n  tracking_front: 0.75\n  tracking_down: 1.5\n", p, error),
        "valid named camera weights rejected");
  check(p.camera_pixel_sigma == 1.25 && p.camera_pixel_sigma_by_name.at("tracking_front") == .75 &&
            p.camera_pixel_sigma_by_name.at("tracking_down") == 1.5, "pixel sigmas parsed incorrectly");
  for (const std::string &bad : {"0", "-1", ".Nan", ".Inf", "-.Inf", "\"1.0\"", "[1.0]", "{ sigma: 1.0 }"}) {
    CalibProfile invalid;
    check(!parse("camera_pixel_sigma: " + bad + "\n", invalid, error), "invalid fallback silently accepted");
    check(!parse("camera_pixel_sigma_by_name:\n  tracking_front: " + bad + "\n", invalid, error),
          "invalid named override silently accepted");
  }
  for (const std::string &bad : {
      "camera_pixel_sigma: 1.0\ncamera_pixel_sigma: 2.0\n",
      "camera_pixel_sigma_by_name:\n  tracking_front: 1.0\n  tracking_front: 2.0\n",
      "camera_pixel_sigma_by_name: [1.0, 2.0]\n",
      "camera_pixel_sigma_by_name: 1.0\n"}) {
    CalibProfile invalid;
    check(!parse(bad, invalid, error), "duplicate/malformed camera mapping silently accepted");
  }
  check(!valid_camera_pixel_sigma(std::numeric_limits<double>::quiet_NaN()) &&
            !valid_camera_pixel_sigma(std::numeric_limits<double>::infinity()),
        "nonfinite sigmas accepted under fast-math");

  const std::set<std::string> registry = {"tracking_front", "tracking_down", "disabled_declared"};
  std::map<size_t, double> resolved;
  std::vector<std::string> unused;
  check(resolve_camera_pixel_noise(p, {{0, "tracking_front"}, {1, "tracking_down"}}, registry, resolved, unused, error) &&
            resolved.at(0) == .75 && resolved.at(1) == 1.5 && unused.empty(), "canonical names did not bind to IDs");
  check(resolve_camera_pixel_noise(p, {{0, "tracking_down"}, {1, "tracking_front"}}, registry, resolved, unused, error) &&
            resolved.at(0) == 1.5 && resolved.at(1) == .75, "reversed camera selection swapped physical camera weights");
  check(resolve_camera_pixel_noise(p, {{5, "tracking_down"}, {2, "tracking_front"}}, registry, resolved, unused, error) &&
            resolved.at(5) == 1.5 && resolved.at(2) == .75, "sparse sensor IDs were treated as selection positions");
  p.camera_pixel_sigma_by_name["disabled_declared"] = 2.0;
  check(resolve_camera_pixel_noise(p, {{2, "tracking_front"}}, registry, resolved, unused, error) &&
            resolved.at(2) == .75 && unused.size() == 2, "known unselected/disabled cameras must be explicitly reported");
  check(!resolve_camera_pixel_noise(p, {{0, "tracking_front"}, {1, "tracking_front"}}, registry, resolved, unused, error),
        "ambiguous duplicate selected camera name accepted");
  check(!resolve_camera_pixel_noise(p, {{0, "tracking_front"}, {0, "tracking_down"}}, registry, resolved, unused, error),
        "duplicate selected camera ID accepted");
  p.camera_pixel_sigma_by_name["tracking_fron"] = .5;
  check(!resolve_camera_pixel_noise(p, {{0, "tracking_front"}}, registry, resolved, unused, error) &&
            error.find("tracking_fron") != std::string::npos, "misspelled override fell back silently");
  p.camera_pixel_sigma_by_name.erase("tracking_fron");
  p.camera_pixel_sigma_by_name["tracking_front_misp_norm_ion"] = .5;
  check(!resolve_camera_pixel_noise(p, {{0, "tracking_front"}}, registry, resolved, unused, error),
        "declared pipe alias mistaken for canonical camera name");
  p.camera_pixel_sigma_by_name.erase("tracking_front_misp_norm_ion");
  p.camera_pixel_sigma_by_name["/run/mpa/custom_mono"] = .9;
  check(resolve_camera_pixel_noise(p, {{3, "/run/mpa/custom_mono"}, {7, "named_adhoc"}}, registry, resolved, unused, error) &&
            resolved.at(3) == .9 && resolved.at(7) == 1.25,
        "ad-hoc pipe identity or fallback must work without a vio_cams.conf declaration");

  // Exercise the real file loader, including profile overlays and the same
  // OpenCV representation that supplied NaN/Inf to production under fast-math.
  auto load = [](const std::string &body, CalibProfile &profile) {
    char path[] = "/tmp/zcalib-numeric-policy-XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0)
      return false;
    FILE *file = ::fdopen(fd, "w");
    if (!file) { ::close(fd); std::remove(path); return false; }
    const std::string yaml = "%YAML:1.0\n---\n" + body;
    const bool written = std::fwrite(yaml.data(), 1, yaml.size(), file) == yaml.size();
    const bool saved = std::fclose(file) == 0 && written;
    bool loaded = false;
    try { loaded = saved && load_calib_profile(path, profile); }
    catch (const cv::Exception &) { loaded = false; }
    std::remove(path);
    return loaded;
  };
  auto rejected = [&](const std::string &key, const std::string &value, const std::string &prefix = "") {
    CalibProfile invalid;
    check(!load(prefix + key + ": " + value + "\n", invalid), ("invalid " + key + "=" + value + " accepted").c_str());
  };
  CalibProfile flight, bench, valid;
  for (const char *key : {"gyroscope_noise_density", "gyroscope_random_walk", "accelerometer_noise_density",
                          "accelerometer_random_walk", "solve_budget_s", "collect_min_eig", "track_rate_hz"}) {
    for (const char *bad : {"-1", ".Nan", ".Inf", "-.Inf", "\"1.0\"", "[1.0]", "[]", "{ value: 1.0 }"})
      rejected(key, bad);
    CalibProfile zero;
    check(load(std::string(key) + ": 0\n", zero), (std::string(key) + " valid zero rejected").c_str());
  }
  for (const char *key : {"collect_budget_s", "commit_sigma_factor", "radtan_tangent_refine_sigma", "radtan_tangent_full_sigma"})
    for (const char *bad : {"0", "-1", ".Nan", ".Inf", "-.Inf", "\"1.0\"", "[1.0]"})
      rejected(key, bad);
  for (const char *key : {"num_threads", "max_clones", "max_track_len", "select_k", "min_holdout", "estimate_cam_intrinsics"})
    for (const char *bad : {"-1", "3.5", ".Nan", ".Inf", "-.Inf", "1e40", "4294967300", "-4294967292",
                            "0x100000004", "040000000004", "18446744073709551620", "\"4\"", "[4]"})
      rejected(key, bad);
  rejected("max_clones", "11");
  rejected("max_track_len", "2");
  rejected("select_k", "0");
  rejected("estimate_cam_intrinsics", "3");
  rejected("gyroscope_noise_density", "-4294967292");
  rejected("num_threads", "!!int 4294967300");
  check(!load("{ profile: flight, num_threads: 4294967300 }\n", valid),
        "flow-map integer overflow accepted");
  for (const char *bad : {"-1", "8.9", "10.1", ".Nan", ".Inf", "-.Inf", "\"9.81\""})
    rejected("gravity_mag", bad);
  for (const char *bad : {"-0.1", "1.01", ".Nan", ".Inf", "-.Inf", "\"0.05\""})
    rejected("verify_min_improve", bad);
  for (const char *bad : {"Flight", "benhc", "0", "[flight]", "{ value: flight }"})
    rejected("profile", bad);
  for (const char *key : {"profile", "num_threads", "gyroscope_noise_density"}) {
    const std::string value = std::string(key) == "profile" ? "flight" : "4";
    rejected(key, value, std::string(key) + ": " + value + "\n");
  }

  // Every boolean must reject malformed present values instead of keeping its
  // default. Read the actual resulting member so both default-true and
  // default-false switches exercise the same parser and profile precedence.
  const std::pair<const char *, bool *> flag_values[] = {
      {"seed_cam_intrinsics", &valid.seed.cam_intrinsics}, {"seed_imu_intrinsics", &valid.seed.imu_intrinsics},
      {"seed_extrinsic_rotation", &valid.seed.extrinsic_rotation}, {"seed_extrinsic_position", &valid.seed.extrinsic_position},
      {"seed_time_offset", &valid.seed.time_offset}, {"seed_tg", &valid.seed.tg},
      {"estimate_imu_intrinsics", &valid.estimate.imu_intrinsics}, {"estimate_tg", &valid.estimate.tg},
      {"cpu_performance_mode", &valid.cpu_performance_mode}, {"record_session", &valid.record_session},
      {"stage_select", &valid.session.stage_select}, {"tg_precision_screen", &valid.session.tg_precision_screen},
      {"bootstrap_epipolar", &valid.session.bootstrap_epipolar}};
  const std::pair<const char *, bool> boolean_spellings[] = {
      {"true", true}, {"True", true}, {"TRUE", true}, {"1", true}, {"\"1\"", true}, {"\"true\"", true},
      {"false", false}, {"False", false}, {"FALSE", false}, {"0", false}, {"\"0\"", false}, {"\"false\"", false}};
  for (const auto &flag : flag_values) {
    for (const char *bad : {"flase", "yes", "2", "-1", "0.0", "1.0", ".Nan", ".Inf", "[]", "[false]",
                            "{ value: false }", "\"\"", "\"false extra\"", "4294967296", "4294967297",
                            "-4294967295", "0x100000001", "040000000001", "!!int 4294967297"})
      rejected(flag.first, bad);
    rejected(flag.first, "false", std::string(flag.first) + ": true\n");
    for (const auto &spelling : boolean_spellings)
      check(load(std::string(flag.first) + ": " + spelling.first + "\n", valid) && *flag.second == spelling.second,
            (std::string(flag.first) + " valid boolean spelling changed: " + spelling.first).c_str());
  }
  check(!load("{ profile: flight, seed_tg: 4294967297 }\n", valid), "flow-map boolean overflow accepted");
  check(!load("seed_tg: false # note: move this comment above the key\n", valid),
        "OpenCV map produced by colon-containing boolean comment must fail loudly");
  check(load("seed_tg: false # a normal comment\nstage_select: true # enabled\nselect_k_a0: 0\n", valid) &&
            !valid.seed.tg && valid.session.stage_select, "ordinary trailing boolean comments rejected");

  check(load("profile: flight\n", flight) && flight.session.harvester.max_clones == 32 &&
            flight.session.harvester.max_track_len == 10 && flight.session.select_K == 10 &&
            flight.session.a_gate_mode == 2 && flight.session.solve_budget_s == 20.0 && flight.session.retro_harvest,
        "valid flight profile changed");
  check(load("profile: bench\n", bench) && bench.session.harvester.max_clones == 70 &&
            bench.session.harvester.max_track_len == 40 && bench.session.select_K == 18 &&
            bench.session.a_gate_mode == 1 && bench.session.solve_budget_s == 0.0 && !bench.session.retro_harvest,
        "valid bench profile changed");
  check(load("profile: flight\nmax_clones: 70\nmax_track_len: 40\nselect_k: 18\nnum_threads: 4.0\n"
             "solve_budget_s: 120\nestimate_cam_intrinsics: 2\n", valid) &&
            valid.session.harvester.max_clones == 70 && valid.session.harvester.max_track_len == 40 &&
            valid.session.select_K == 18 && valid.session.joint.num_threads == 4 &&
            valid.session.solve_budget_s == 120 && valid.session.cam_mode == 2,
        "explicit numeric overrides must win over the flight overlay");
  for (const char *number : {"4", "+4", "04", "0x4", "4.0", "4e0", "!!int 4"})
    check(load(std::string("num_threads: ") + number + "\n", valid) && valid.session.joint.num_threads == 4,
          "valid OpenCV integer representation changed");
  check(load("{ profile: flight, num_threads: 4, stage_select: true, select_k_a0: 0 }\n", valid) &&
            valid.session.joint.num_threads == 4 && valid.session.select_K_a0 == 0,
        "valid flow-map representation changed");
  check(load("# num_threads: 4294967300\nprofile: flight\nnum_threads: 4 # 4294967300\n"
             "camera_pixel_sigma_by_name: { num_threads: 4294967300.0 }\n", valid),
        "integer range precheck must not consume comments or nested camera names");
  check(load("num_threads: 0\nmin_holdout: 0\ngravity_mag: 0\nverify_min_improve: 0\n"
             "max_clones: 12\nmax_track_len: 3\nselect_k: 1\n", valid) &&
            valid.session.joint.num_threads == 0 && valid.session.min_holdout == 0 && valid.gravity_mag == 0,
        "valid zero/off and structural boundary values rejected");
  for (const char *gravity : {"9", "9.81", "10"})
    check(load(std::string("gravity_mag: ") + gravity + "\nverify_min_improve: 1\n", valid),
          "valid gravity / verification-fraction endpoints rejected");
  for (const char *key : {"select_k_a0", "select_k_a1", "select_k_b"}) {
    for (const char *bad : {"-1", "2.5", ".Nan", ".Inf", "1e40"}) {
      rejected(key, bad, "stage_select: true\n");
      check(load(std::string("stage_select: false\n") + key + ": " + bad + "\n", valid) &&
                valid.session.select_K_a0 == 0 && valid.session.select_K_a1 == 0 && valid.session.select_K_b == 0,
            "disabled stage counts must remain unused without integer coercion");
    }
  }
  check(load("stage_select: true\nselect_k_a0: 0\nselect_k_a1: 20\nselect_k_b: 0\n", valid) &&
            valid.session.select_K_a0 == 0 && valid.session.select_K_a1 == 20 && valid.session.select_K_b == 0,
        "active stage count zero must preserve inheritance");
  check(load("profile: bench\nseed_imu_intrinsics: false\nseed_tg: false\nestimate_imu_intrinsics: false\n"
             "estimate_tg: false\nestimate_cam_intrinsics: 0\ntg_precision_screen: false\n", valid) &&
            !valid.seed.imu_intrinsics && !valid.seed.tg && !valid.estimate.imu_intrinsics &&
            !valid.estimate.tg && !valid.session.free_tg && !valid.session.tg_precision_screen && valid.session.cam_mode == 0,
        "valid independent seed / estimate / screen switches changed");
  char absent[] = "/tmp/zcalib-profile-absent-XXXXXX";
  const int absent_fd = ::mkstemp(absent);
  check(absent_fd >= 0, "could not reserve a missing-profile test name");
  if (absent_fd >= 0) {
    ::close(absent_fd);
    std::remove(absent);
    check(load_calib_profile(absent, valid) && valid.base == "flight" && valid.session.harvester.max_clones == 32 &&
              valid.session.solve_budget_s == 20 && valid.noise.sigma_w == 1.3990944749616306e-4,
          "documented missing-file fallback changed");
  }

  // Exercise the same parsed policy and helper used by the live seed builder.
  // Distinct values expose accidental Tg coupling to the Dw/Da/q_AtoI switch.
  ImuIntrinsicModel ported, identity;
  ported.dw << 1.02, -.013, .98, .021, -.008, 1.01;
  ported.da << .97, .017, 1.03, -.009, .006, 1.04;
  ported.q_AtoI << .02, -.03, .01, .99;
  ported.q_AtoI.normalize();
  ported.Tg << 1e-4, 2e-4, -3e-4, -4e-4, 5e-4, 6e-4, 7e-4, -8e-4, 9e-4;
  // Seeding must not silently alter the separate estimation policy.
  ported.calib_dw = false;
  ported.calib_RAtoI = false;
  ported.calib_tg = true;
  auto exact = [](const auto &a, const auto &b) {
    return std::memcmp(a.data(), b.data(), sizeof(double) * a.size()) == 0;
  };
  const ImuIntrinsicModel default_seed = make_imu_seed(ported, SeedPolicy());
  check(exact(default_seed.dw, ported.dw) && exact(default_seed.da, ported.da) &&
            exact(default_seed.q_AtoI, ported.q_AtoI) && exact(default_seed.Tg, ported.Tg),
        "default seed policy must preserve every ported chain value exactly");
  for (bool seed_imu : {false, true}) {
    for (bool seed_tg : {false, true}) {
      char path[] = "/tmp/zcalib-seed-policy-XXXXXX";
      const int fd = ::mkstemp(path);
      check(fd >= 0, "temporary seed policy file could not be created");
      if (fd < 0)
        continue;
      FILE *file = ::fdopen(fd, "w");
      check(file != nullptr, "temporary seed policy file could not be opened");
      if (!file) {
        ::close(fd);
        std::remove(path);
        continue;
      }
      const int written = std::fprintf(file, "%%YAML:1.0\n---\nseed_imu_intrinsics: %s\nseed_tg: %s\n",
                                       seed_imu ? "true" : "false", seed_tg ? "true" : "false");
      const bool saved = std::fclose(file) == 0 && written > 0;
      CalibProfile profile;
      const bool loaded = saved && load_calib_profile(path, profile);
      std::remove(path);
      check(loaded, "seed policy YAML failed to load");
      if (!loaded)
        continue;
      check(profile.seed.imu_intrinsics == seed_imu && profile.seed.tg == seed_tg,
            "seed_imu_intrinsics and seed_tg must parse independently");
      const ImuIntrinsicModel seeded = make_imu_seed(ported, profile.seed);
      const ImuIntrinsicModel &expected_chain = seed_imu ? ported : identity;
      check(exact(seeded.dw, expected_chain.dw) && exact(seeded.da, expected_chain.da) &&
                exact(seeded.q_AtoI, expected_chain.q_AtoI),
            "Dw/Da/q_AtoI initializer must depend only on seed_imu_intrinsics");
      check(exact(seeded.Tg, seed_tg ? ported.Tg : identity.Tg),
            "Tg initializer must depend only on seed_tg");
      check(seeded.calib_dw == ported.calib_dw && seeded.calib_da == ported.calib_da &&
                seeded.calib_RAtoI == ported.calib_RAtoI && seeded.calib_tg == ported.calib_tg &&
                profile.estimate.imu_intrinsics && profile.estimate.tg && profile.session.free_tg,
            "seed choices must not change estimation switches");
      for (bool estimate_imu : {false, true}) {
        ImuIntrinsicModel effective = seeded;
        EstimatePolicy estimation;
        estimation.imu_intrinsics = estimate_imu;
        apply_imu_estimate_policy(effective, estimation);
        check(effective.calib_dw == (estimate_imu && seeded.calib_dw) &&
                  effective.calib_da == (estimate_imu && seeded.calib_da) &&
                  effective.calib_RAtoI == (estimate_imu && seeded.calib_RAtoI) &&
                  effective.calib_tg == (estimate_imu && seeded.calib_tg),
              "disabled IMU estimation must freeze even an identity seed");
        check(exact(effective.dw, seeded.dw) && exact(effective.da, seeded.da) &&
                  exact(effective.q_AtoI, seeded.q_AtoI) && exact(effective.Tg, seeded.Tg),
              "estimation switches must not replace selected seed values");
      }
      std::printf("seed policy imu_intrinsics=%d tg=%d: chain=%s Tg=%s\n", (int)seed_imu, (int)seed_tg,
                  seed_imu ? "ported" : "identity", seed_tg ? "ported" : "zero");
    }
  }
  std::printf("%s camera pixel config: defaults, strict values, identity binding, reversed selection, unused and ad-hoc cameras\n",
              failures ? "FAIL" : "PASS");
  std::printf("%s IMU seed policy: parsed four-way independence, exact defaults, estimation switches preserved\n",
              failures ? "FAIL" : "PASS");
  std::printf("%s numeric profile: finite domains, exact profile names, count types, overlays, dormant stages and zero semantics\n",
              failures ? "FAIL" : "PASS");
  std::printf("%s boolean profile: all switches, strict scalar types, duplicates, integer overflow and valid spellings\n",
              failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
