#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
mkdir -p work/task-graph
xcrun metal -O2 -std=metal4.0 -mmacosx-version-min=26.4 -c experiments/task-graph/graph.metal -o work/task-graph/graph.air
xcrun metallib work/task-graph/graph.air -o work/task-graph/graph.metallib
xcrun swiftc -O experiments/task-graph/run.swift -o work/task-graph/bench
work/task-graph/bench work/task-graph/graph.metallib "$@"
