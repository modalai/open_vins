#!/bin/bash
# sim_ab.sh -- fast iteration driver for the async dual-camera sim harness (test_async_dual).
# Wraps the host x86 test build, single runs, config variants, CTest, the HEAD bit-parity jig,
# and the voxl-cross container production build behind stable subcommands so iterations don't
# depend on interactive-shell quirks.
#
# Usage:
#   sim_ab.sh build                       # host x86 cmake+make (tests ON), honest exit code
#   sim_ab.sh run <name> [harness args]   # run voxl_sim baseline; prints the [RESULT] line
#   sim_ab.sh variant <name> <sed-expr> [harness args]
#                                         # copy voxl_sim cfg, apply sed to estimator_config.yaml, run
#   sim_ab.sh ctest                       # run the registered CTest gates
#   sim_ab.sh parity                      # HEAD-vs-branch simulator measurement-stream hash compare
#   sim_ab.sh container                   # ./build.sh qrb5165 inside the voxl-cross container
#
# Author: Joao Leonardo Silva Cotta (@zauberflote1)
set -u

REPO="${REPO:-/home/zbft/Documents/SFM_LIFE/voxl-open-vins-server}"
OV="$REPO/external/open_vins"
SCRATCH="${SCRATCH:-/tmp/claude-1000/-home-zbft-Documents-SFM-LIFE/254fb043-6360-41b2-bd29-f28d2a003679/scratchpad}"
BUILD="$SCRATCH/ovbuild"
LOGS="$SCRATCH/simlogs"
TRAJ="${TRAJ:-$OV/ov_data/sim/udel_gore.txt}"
CFG_DIR="$OV/config/voxl_sim"
BIN="$BUILD/ov_msckf/test_async_dual"
CONTAINER="${CONTAINER:-pensive_dubinsky}"
mkdir -p "$LOGS"

die() { echo "[sim_ab] ERROR: $*" >&2; exit 1; }

cmd_build() {
  if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    cmake -S "$OV" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DENABLE_ROS=OFF -DENABLE_ARUCO_TAGS=OFF \
      -DOV_MSCKF_BUILD_TESTS=ON -DBUILD_OV_EVAL=OFF > "$LOGS/cmake.log" 2>&1 || { tail -20 "$LOGS/cmake.log"; die "cmake configure failed"; }
  else
    cmake -S "$OV" -B "$BUILD" > /dev/null 2>&1 || true # refresh generated files (add_test changes)
  fi
  make -C "$BUILD" -j"$(nproc)" > "$LOGS/build.log" 2>&1
  ec=$?
  if [ $ec -ne 0 ]; then grep -E "error:|Error " "$LOGS/build.log" | head -15; die "build failed (full log: $LOGS/build.log)"; fi
  echo "[sim_ab] build OK"
}

run_with_cfg() { # cfg_yaml name args...
  local cfg="$1" name="$2"; shift 2
  [ -x "$BIN" ] || die "harness not built ($BIN); run: sim_ab.sh build"
  "$BIN" "$cfg" --traj "$TRAJ" --name "$name" "$@" > "$LOGS/$name.log" 2>&1
  local ec=$?
  grep -aE "\[RESULT\]|\[FAIL|\[PASS" "$LOGS/$name.log" | sed 's/\x1b\[[0-9;]*m//g'
  grep -a "unable to parse" "$LOGS/$name.log" | head -3
  return $ec
}

cmd_run() { local name="$1"; shift; run_with_cfg "$CFG_DIR/estimator_config.yaml" "$name" "$@"; }

cmd_variant() { # name sed-expr args...
  local name="$1" expr="$2"; shift 2
  local vdir="$SCRATCH/cfg_$name"
  rm -rf "$vdir" && cp -r "$CFG_DIR" "$vdir" || die "cfg copy failed"
  sed -i -e "$expr" "$vdir/estimator_config.yaml" || die "sed failed: $expr"
  run_with_cfg "$vdir/estimator_config.yaml" "$name" "$@"
}

cmd_ctest() { ctest --test-dir "$BUILD/ov_msckf" --output-on-failure 2>&1 | tail -20; }

cmd_parity() {
  [ -x "$SCRATCH/sim_hash_branch" ] && [ -x "$SCRATCH/sim_hash_head" ] || die "parity jigs missing (see S0 notes)"
  local b h
  b=$("$SCRATCH/sim_hash_branch" "$OV/config/rpng_sim/estimator_config.yaml" "$TRAJ" 2>/dev/null | tail -1)
  h=$("$SCRATCH/sim_hash_head" "$SCRATCH/ov_head/config/rpng_sim/estimator_config.yaml" "$TRAJ" 2>/dev/null | tail -1)
  echo "branch: $b"
  echo "head:   $h"
  [ "${b%% *}" = "${h%% *}" ] && echo "[sim_ab] PARITY OK" || die "HASH MISMATCH"
}

cmd_container() {
  docker exec "$CONTAINER" bash -lc "cd /home/root/voxl-open-vins-server && ./build.sh qrb5165 > /tmp/build.log 2>&1; ec=\$?; tail -3 /tmp/build.log; exit \$ec" \
    && echo "[sim_ab] container build OK" || die "container build failed"
}

case "${1:-}" in
  build) cmd_build ;;
  run) shift; cmd_run "$@" ;;
  variant) shift; cmd_variant "$@" ;;
  ctest) cmd_ctest ;;
  parity) cmd_parity ;;
  container) cmd_container ;;
  *) grep '^#   sim_ab.sh' "$0" | sed 's/^# *//'; exit 1 ;;
esac
