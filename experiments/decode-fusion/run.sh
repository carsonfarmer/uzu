#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
mkdir -p work
xcrun metal -O2 -std=metal4.0 -mmacosx-version-min=26.0 \
  -I crates/uzu-engine/src/backends/metal/kernel \
  -I crates/uzu-engine/src/backends/metal/kernel/generated \
  -c experiments/decode-fusion/paired.metal -o work/paired.air
xcrun metallib work/paired.air -o work/paired.metallib
xcrun swiftc -O experiments/decode-fusion/run.swift -o work/paired-bench
# Optional MODEL_TENSORS points to extract.py output; capture path is optional.
work/paired-bench work/paired.metallib "$@"
