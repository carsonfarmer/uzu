# Metal task/dependency execution prototype

This tests an initial part of Cohere's megakernel design: a persistent worker
reads descriptors and executes dependent operations without a CPU return or
dispatch between those operations. It is a synthetic scheduler experiment,
not a transformer block or an Apple port of Cohere's serving engine.

## Graph and controls

Each independent graph has 64 scalar lanes and six tasks:
`a=2*x; b=a+1; c=a*a; d=b+c; e=d/2; f=e+c`.
This includes fan-out from a and c, and fan-in at d and f.

Task IDs 0...5 identify separate output slots. The host emits 32-byte descriptors:
ID, opcode, input slots, dependency bitmask, three reserved fields.
The interpreter scans descriptors for ready tasks and keeps a private completed
bitmask. Reverse order takes several scans. The six-scan bound follows from the
six-node DAG; the host validates IDs, input dependencies and acyclicity.

The controls use identical arithmetic, device scratch layout, output slots and
task-visit instrumentation:

- `phased`: six compute dispatches/encoders in one command buffer, one per
  operation, with ordinary Metal ordering.
- `fixed_fusion`: one threadgroup per graph, fixed topological task loop,
  one dispatch.
- `static_tasks`: a configurable worker pool; workers traverse strided graph
  IDs and interpret each graph's descriptor list, one dispatch.
- `dynamic_tasks`: workers atomically claim independent graphs, interpreting
  each before claiming another, one dispatch.

The dynamic queue distributes whole graphs, not individual ready DAG nodes.
Dependencies and activation visibility are **within one owning threadgroup**.
A `threadgroup_barrier(mem_device)` orders device scratch accesses by that
group. Relaxed atomic fetch-add only allocates prepublished graph IDs and counts
visits; it does not publish activations or signal cross-group dependencies.
No worker waits for another worker. Every queue claim advances toward a finite
end. Worker counts are experimental parameters, not physical core assignments.

This is the threadgroup-local subset of the task/dependency milestone.
Cross-threadgroup fan-in, a general ready-node queue, prefetch, transformer
weights and full-model decoding remain unimplemented.

## Run

From the repository root on this Apple Silicon/Xcode setup:

```sh
experiments/task-graph/run.sh > work/task-graph-run.json
python3 experiments/task-graph/summarize.py work/task-graph-run.json
```

The standalone build uses Metal 4.0, minimum macOS 26.4, `-O2`, and Swift `-O`.
No engine code or extra package installation is required.

Separate validation run (exclude from performance comparisons):

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  work/task-graph/bench work/task-graph/graph.metallib --checks-only \
  > work/task-graph-validation.json 2> work/task-graph-validation.log
```

## Checks

Each process checks 144 executions across 1, 19, and 257 graphs; worker
counts 1, 7, 20, 64 and 256; topological, reverse, and shuffled descriptor
orders; and 24 repeated steps with changing inputs and orders. The fixed-order
controls only use topological descriptors. All six intermediate arrays must
match the CPU calculation bit-for-bit, every task count must be exactly one,
and every completion mask must contain all six tasks. Scratch is poisoned
before every submission to expose missing writes. Every timed sample is
also checked after GPU completion.

These checks establish correctness for this bounded graph and tested device,
not a general proof of Metal inter-threadgroup progress.

## Timing

4,096 graphs, 64 lanes, 6 tasks; 15 samples per configuration after 5 warmups.
Configurations rotate and reverse order across repetitions. Static/dynamic
worker counts range from 4 to 256. GPU command-buffer time and CPU wall time
include GPU counter reset, scratch poison and all compute dispatches. They
exclude compilation, host descriptor/input preparation and post-run checking.

The full-capacity scratch poison and diagnostic counters intentionally remain
in every control; these costs mean the timings are not pure interpreter cost.
Phased control uses separate encoders, so its difference also includes encoder
boundaries. There is no operation-specific bandwidth attribution.

Raw runs 1–3 and separate validation are retained in results/. They ran on
the M4 Pro, AC power, macOS 27.0, with nominal thermal samples. The North Mini
Code subagent was doing source/network research concurrently, so this is not
a fully quiescent machine measurement. No model server or compiler ran during
the repeated measured submissions.

Do not interpret a synthetic phased-to-fused improvement as a decode speedup.
The useful outcome is a working, validated execution harness plus an explicit
control showing whether persistence helps beyond fixed fusion.
