#!/usr/bin/env bash
set -euo pipefail

mkdir -p .ci-cache/ccache
exec docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$PWD,target=/work" ov-host-math:ci "$@"
