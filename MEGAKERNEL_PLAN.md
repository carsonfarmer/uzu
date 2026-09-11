# Apple megakernel research plan

## Objective and correction

Test whether the scheduling and dataflow ideas in Cohere's decode megakernel
can improve small-batch inference on Apple GPUs. The working M4 Pro/Uzu/Qwen
baseline is the test bed, not the research question. A negative up/gate fusion
microbenchmark does not answer this question.

The first experiment tested an operator epilogue. Subsequent work established
correct packed-W4 kernels, cross-threadgroup dependency publication, a
persistent ready-work scheduler and controlled weight-staging ablations on real
North inputs. Opt-in fused and persistent branch replacements now run in all 48
MoE layers and have been measured during full-model generation. The remaining
goal is to extend persistent execution across the rest of the decode graph and
test Cohere's cross-layer QKV/router overlap against the strongest fused control.

## Preserved real-model checkpoint

NORTH_RESULTS.md records the current checkpoint. The corrected fused Metal
branch improves paired full-model decode throughput by 6.6–11.7% over original
MLX, and the optimized persistent scheduler improves it by 6.0–6.6% in the main
run. An independent counterbalanced run confirms 7.0–9.0% persistent gains.
Five corrected variants pass 1,424 full-vocabulary raw-logit byte comparisons
each, for 7,120 comparisons plus prefill checks. A chart, reproducible packet
and tweet draft are in experiments/north/share/.

This checkpoint validates useful pieces of Cohere's approach: mixing independent
attention-output and MoE work, removing the false dependency between different
experts, and running a persistent ready-work scheduler. It does not yet validate
their full decode megakernel. QKV, attention, routing, normalization, layer
transitions and the output head still run through MLX, and next-layer QKV/router
weight overlap has not been implemented.

## Earlier implemented checkpoint

experiments/task-graph now provides phased, fixed-fused, static-interpreter and
dynamic-graph-queue controls. It executes dependent tasks within an owning
threadgroup, with shuffled descriptors and exactly-once checks. Three runs
and separate Metal validation passed. This covers stage 1 and the local-chain
subset of stage 2; cross-threadgroup fan-in remains open. See its README and
results for the scope and synthetic timing limits.

NORTH_MINI_CODE.md assesses the requested real model. Next use a North-shaped
parallel branch graph, then a quantized MLX reference and real dense/MoE layers.
That assessment predates the completed local baseline and decode experiment.

## Reference implementation inspected

Reference: https://github.com/cohere-ai/cohere-megakernel
Pinned local checkout: work/cohere-megakernel
Commit: 67d0b9ca22ea3652796b715d1d1863459e0e2c3c

- src/decode/megakernel.cuh: task fields/opcodes, interpreter, cross-SM
  synchronization, shared GEMM pipeline and dynamic work claims.
- src/decode/schedule.py: host-built task streams; round-robin waves,
  dependency-aware placement, context-dependent schedules.
- src/decode/launch.cuh: launches num_sms blocks, with a fixed worker shape.
- src/decode/abi.h: host/device launch parameters, task streams and barrier
  storage shared with the serving runtime.

The release targets H100/CUDA and North Mini Code. Its instruction sequences,
block residency assumptions and scheduling choices are not a drop-in Metal
implementation. The existing Qwen test model also has a different dependency
graph; we cannot assume North Mini Code's parallel attention/FFN branches.

Design explanation: https://cohere.com/blog/megakernels

## Experiments, with explicit controls

| Stage | Question | Experiment/control |
|---|---|---|
| 1. Task execution | What does a persistent Metal worker/interpreter cost? | Same tiled math and outputs through ordinary dispatches, static descriptor lists, and a dynamic ready-work queue. Sweep worker counts; do not equate threadgroups with physical GPU cores. |
| 2. Cross-operation scheduling | Can workers move directly into useful downstream work? | A small producer/consumer graph with fan-out and fan-in. Compare operation-wide phase boundaries with dependency-local execution. Keep arithmetic, data layout, and work totals fixed. |
| 3. Real block | Does the mechanism survive real weight traffic and reductions? | Up/gate, activation/input preparation, and down projection from a complete MLP block; include reduction cost and all intermediates. Compare against the unchanged Uzu block. |
| 4. Overlap | Can immutable weight loading overlap preceding work? | Add and remove prefetch/pipelining while holding the schedule fixed; measure bandwidth/latency and resource-pressure effects. |
| 5. Decode graph | Does this improve something the user can run? | Extend the viable scheduler to normalization, QKV, attention or DeltaNet, projections and layer transitions, then an opt-in repeated-decoding route with the original fallback. |

Stage 1 alone is not a megakernel demonstration. The first architectural
milestone is stage 2: a Metal worker actually executes dependent tasks from
different operations without returning to the CPU between each task.
Stage 3 is the first real-model block milestone. Stage 5 is the serving goal.
Each stage must report what mechanism it exercised, not just a throughput number.

## Metal synchronization investigation

Cohere's counter protocol includes device memory publication and cross-block
waiting. Replacing its CUDA fences with relaxed Metal atomics is not a
correctness argument.

Before adopting a shared dependency queue, establish the supported memory-order
and visibility protocol from the selected Metal language version and compile
small producer/consumer checks. Establish forward progress independently of
memory visibility. Historical progress tests found scheduling differences on
Apple hardware; they are a warning against inheriting CUDA assumptions, not a
measurement of this M4 Pro:
https://arxiv.org/abs/2109.06132

Start with independent ready tasks and threadgroup-local dependent chains.
For dependencies across threadgroups, evaluate a nonblocking scheduler or
bounded epochs with unresolved work resumed in a later dispatch. Do not run
an unbounded spin-wait grid on an assumed one-worker-per-core residency model.
A phased implementation must be labelled as such; it is not proof that a
single-dispatch full-model megakernel works.

Required scheduler checks: every task executes exactly once, no task consumes
unpublished input, fan-in counters match actual producers, buffers/counters
reset across steps, and results remain correct under changed worker counts
and task ordering. Oversubscription must not introduce a dependency deadlock.

## Measurement and promotion

Use the same model tensors, math and context for comparisons. Preserve
unfused, fusion-only, persistent-worker, dependency-scheduling and prefetch
variants so benefits can be attributed. GPU command-buffer time and CPU wall
time must remain separate. Include queue setup/reset, reductions and any extra
dispatches; do not hide scheduler costs.

Real-block outputs and then full-model logits/generated tokens need comparison
with Uzu. Repeat multi-token decoding and coding prompts at multiple contexts.
Record power/thermal state and retain negative results. A single benchmark win
does not establish compatibility across Apple GPU families.

Keep the existing isolated fusion result as one ablation. It is neither a
reason to abandon the larger research program nor evidence that the larger
program works.
