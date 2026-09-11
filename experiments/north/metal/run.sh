#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
mkdir -p work/north-metal
xcrun metal -O2 -std=metal4.0 -mmacosx-version-min=26.4 -c experiments/north/metal/branch.metal -o work/north-metal/branch.air
xcrun metallib work/north-metal/branch.air -o work/north-metal/branch.metallib
xcrun swiftc -O experiments/north/metal/run.swift -o work/north-metal/bench
work/north-metal/bench work/north-metal/branch.metallib "$@"
