# North Mini Code megakernel results on Apple M4 Pro

## Additional decode target — September 13, 2026

The additional 10% decode-throughput target is now met on the three tested
prompts. A prepared async path improves over the unchanged synchronous exact
fused control by **11.88% Python, 12.00% Rust, and 12.03% at 1,672 prompt
tokens** in twelve balanced AB/BA pairs per prompt. All 127 full-logit positions
match stock byte-for-byte, and all measured generations have identical tokens.
Paired confidence intervals have lower bounds above +10% on all three prompts.

The gain comes mainly from overlapping host submission, with about 1% more
from exact preparation fusion. It is not an incremental complete-megakernel
speedup. The [report and raw evidence](experiments/north/additional/README.md)
document the controls, noisy earlier runs, fixed-length scope, and attribution.

## Full-megakernel status correction

The complete-megakernel performance objective remains unfinished. The full queue implementation
below is a research checkpoint, and its completion is not evidence of a
megakernel speedup. “Safe” in historical variant names means the retained checks
completed; it is not a proof of scheduler liveness on arbitrary workloads.

A new [branch-mixing ablation](experiments/north/branch_mixing/README.md)
separates the contribution of combining attention/expert work from the other
fused calculations. The additional decode result above uses fresh matched
measurements against that fused control.

## The retained full-queue result

We built a Metal path that executes all 49 North Mini Code transformer
layers, every K/V update, final RMSNorm, and the entire 262,144-entry output
head in one custom GPU dispatch for each generated token.

It is bitwise equal to stock MLX on two independent 127-step coding
continuations. Six measured runs produce 53.423 tok/s, while six stock MLX runs
produce 53.951 tok/s. The safe path is 0.98% slower, or within 1% on this
desktop test. The best exact fused control remains faster at 59.952 tok/s.

## Why the earlier +6.75% result is not the headline

The first complete kernel used a global barrier: every threadgroup had to reach
each phase boundary before any could continue. That version completed long
exact runs and measured 57.472 tok/s, 6.75% above its adjacent stock result.

When the exact test was rerun adversarially, the kernel produced two correct
steps and then hung. A residency-related global-barrier deadlock is a suspected
cause; we did not conclusively diagnose it. The failure belongs to this
implementation and does not establish a general limitation of Metal. Correct
output on completed steps does not fix a liveness failure.

We retained the benchmark and partial stalled artifact under
[whole-pass history](experiments/north/whole_pass/history/STATIC_SCHEDULER_FAILURE.md),
but removed the speedup from every validated claim.

## The replacement

The safe version uses a dynamic ready-task queue. Each resident threadgroup:

1. reads the current phase;
2. atomically claims one ready job;
3. runs that job;
4. increments the completion count;
5. advances the phase if it finished the final job.

No group waits for a particular group that might not be resident. The full
297-phase decode completes and returns matching claim/completion cursors at 1,
20, 32, and 36 groups. Thirty-six groups are used for the measured path.

## Performance evidence

Apple M4 Pro with 20 GPU cores and 48 GB; MLX 0.32.2; batch one; 136-token
prompt; 127 timed decode steps. Model load, packing, tokenization, prefill, and
warmups are excluded. Embedding lookup, the transformer/logit dispatch, K/V
carry and growth, synchronization, and argmax are included.

| Path | Median tok/s | Difference from stock |
|---|---:|---:|
| Stock MLX, 6 runs | 53.951 | — |
| Exact fused control, 6 runs | 59.952 | +11.12% |
| **Safe one-dispatch queue, 6 runs** | **53.423** | **-0.98%** |

The two adjacent safe-versus-stock process-pair estimates are -1.26% and
-0.81%. Both pairs put the safe path within 1.3% of stock.

## Scheduling and prefetch ablations

Four queue configurations share the packed weights in one process and rotate
after every decode step. This makes the variants experience nearly the same
clock and system conditions.

| Configuration | Median tok/s | Difference from tuned |
|---|---:|---:|
| Tuned 64/64/8-row jobs | 50.518 | — |
| Coarse 128/128/16-row jobs | 48.898 | -3.21% |
| Tuned plus one next-weight cache-warming stage | 48.098 | -4.79% |
| Tuned plus ten cache-warming stages | 37.964 | -24.85% |

The tuned configuration was 3.31% faster than the coarse configuration. That
result is consistent with a task-placement benefit, but does not isolate
partial-wave utilization from other effects of the tile-size change. The cache-warming experiment
does not reproduce H100 TMA: it performs extra reads of next-layer QKV/router
weights to warm cache. Those reads cost bandwidth on this path.

## Correctness evidence

- 18 primitive raw-byte comparisons across six representative layers and three
  positions.
- Four complete one-step comparisons at 1, 20, 32, and 36 groups, covering the
  final hidden state, all used cache bytes, all logits, and queue completion.
- 127 exact full-logit decode steps for a Python prompt and 127 for a Rust
  prompt.
- Identical generated token sequences across all 30 measured benchmark rows.
- Source SHA-256 provenance checked by the summary script.

Changed logits or tokens fail the gate. No numeric tolerance is used.

## How this maps to Cohere's work

[Cohere's article](https://cohere.com/blog/megakernels) and
[pinned source](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c)
inspired the full-model task engine, resident worker pool, fine task sizing,
dependency queue, and next-layer weight-overlap test.

The fused control is related research, not the megakernel result. It fuses
selected parallel operations inside each layer and leaves the rest of the model
under MLX. It establishes a strong Apple baseline and shows where specialized
fusion still beats the general queue.

## Why the fused control is hard to beat

These are ranked hypotheses, not conclusions. Each has a direct experiment that
can prove or reject it.

| Rank | Hypothesis | Why it fits the evidence | Decisive next test |
|---:|---|---|---|
| 1 | The safe queue pays too much synchronization overhead. | The full path advances 297 phases with atomic claims, completions, device fences, and polling. The unsafe static schedule was faster when it completed. | Add per-phase GPU timestamps, then merge only the most expensive adjacent phases while retaining ready-task progress. |
| 2 | Copying the active K/V prefix consumes the launch savings. | MLX custom-kernel outputs are new buffers, so the one-dispatch path copies every used cache entry on every token. The fused path updates native caches without this whole-prefix copy. | Implement alias-safe in-place cache updates through a lower-level Metal command path and compare identical queue math at several context lengths. |
| 3 | The fused path keeps MLX's best specialized kernels. | It fuses the profitable attention-output/MoE branch while leaving QKV, normalization, routing, attention, and the huge output head to tuned MLX kernels. The megakernel replaces all of them with one general program. | Benchmark identical packed inputs operation by operation and compare bytes moved, GPU time, and arithmetic throughput with the MLX kernel for each stage. |
| 4 | The giant kernel reduces GPU occupancy or compiler quality. | One Metal function contains dense and MoE paths, attention, routing, cache logic, queue logic, and the output head. Register pressure, instruction-cache pressure, or conservative compilation can offset fewer launches. | Compare one dispatch with carefully chosen two-, four-, and eight-dispatch cuts; collect Metal occupancy, register, and GPU-counter data. |
| 5 | The strongest Cohere overlap mechanism is missing. | H100 TMA can move future weights asynchronously. The available cache-warming substitute adds reads and loses 4.79% to 24.85%. | Build a native Metal prototype with explicit staged transfers and double-buffered threadgroup memory, then measure overlap rather than cache warming. |
| 6 | North's affine W4 decode has a different bottleneck from Cohere's BF16 H100 path. | Weight decoding, scales/biases, unified memory, and a 262k head make this Apple workload strongly bandwidth-sensitive. Launch count may be a smaller fraction of total time. | Produce a bytes-per-token roofline and repeat on M4 Max plus BF16 or another quantization with the same scheduler. |

The first two tests are the highest priority. Removing the cache copy attacks
work the fused path simply does not perform, while phase timing will show whether
the queue itself or the math kernels dominate the remaining 10.89% gap.

## Scope

These tests establish observed correctness and throughput for this implementation
on a 4-bit model, batch-one generation, short context, and one M4 Pro. They do
not establish a performance advantage for the complete megakernel, nor validate
Cohere's full serving system, continuous batching, paged attention, ragged
sequences, 256K contexts, or a true asynchronous Metal weight-transfer
pipeline.

Use [the full walkthrough](experiments/north/whole_pass/README.md) for the
implementation and reproduction commands. The
[machine-readable summary](experiments/north/whole_pass/summary-v1.json)
contains every retained sample and artifact hash.
