#!/usr/bin/env bash
set -euo pipefail

archive=.ci-cache/host-environment.tar
if [[ -f "$archive" ]]; then
  docker load --input "$archive"
else
  docker build --tag ov-host-math:ci --file .github/ci/host-math.Dockerfile .github/ci
  mkdir -p .ci-cache
  docker save --output "$archive" ov-host-math:ci
fi
