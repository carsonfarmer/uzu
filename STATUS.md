# Apple North research status — September 14, 2026

Fresh matched tests confirm the useful combined result: targeted fusion plus
async decode is **22.4% faster on Python, 22.7% on Rust and 19.4% on the longer
prompt** than the original synchronous research runner. Optional preparation
fusion raises those gains to 24.4%, 24.5% and 22.0%. These were measured together;
no percentages from historical runs were multiplied to obtain them.

Core fusion retains 7.1–9.8% over the stock layers with the same async loop;
including preparation gives 9.5–11.4%. All 1,905 full-logit comparisons match
byte-for-byte, and all 150 measured continuations have identical token IDs.
The longer-prompt interval is wider after a late speed shift; all runs remain
in the data. A fresh Python installation also passed the correctness smoke run.

The minimal implementation and evidence are committed and pushed as 2ae12a85
on the clean [cf/north-fused-async branch](https://github.com/carsonfarmer/uzu/tree/cf/north-fused-async/experiments/north_fused_async), starting directly from
upstream 7096cf32. It adds a standalone MLX experiment and leaves Uzu's engine
code untouched. See the [matched results](https://github.com/carsonfarmer/uzu/tree/cf/north-fused-async/experiments/north_fused_async/RESULTS.md) and
[reproduction instructions](https://github.com/carsonfarmer/uzu/tree/cf/north-fused-async/experiments/north_fused_async/README.md). The older research branches
remain preserved.

The full megakernel investigation is active separately on
[cf/north-megakernel-fullest](https://github.com/carsonfarmer/uzu/tree/cf/north-megakernel-fullest).
Its new goal targets at least another 10% over the strongest exact
fusion+preparation+async control through faithful Cohere scheduling and real
future-weight overlap, or evidence exhausting the plausible mechanisms.
The clean baseline has released its GPU reservation for the task's numerical
and performance tests. No megakernel performance success is claimed yet.

## September 13 checkpoint and earlier ablations

The additional 10% research-runner target is verified on the original short
Python workload (+10.70%), corroborated by Rust (+10.92%). Both use the actual
prior fused host-token loop as a fresh control; ten balanced rounds per prompt
clear 10% with both bootstrap and independent log-ratio t intervals. Full logits
match byte-for-byte. Longer-context correctness passes, but its throughput
estimate remains uncertain after substantial timing stalls.

The implementation and complete evidence are preserved separately on
[`cf/north-additional-ten-percent`](https://github.com/carsonfarmer/uzu/tree/cf/north-additional-ten-percent/experiments/north/additional).
Most of the runner gain comes from adopting the standard async pattern already
present in pinned MLX-VLM generation. New preparation fusion adds roughly
1.1–1.2% over the unchanged fused async path. An earlier 12% confirmation used
an extra wait in its control and was withdrawn; all of that evidence is retained.

The full megakernel performance objective remains unfinished. This runner result
does not demonstrate a benefit from the complete persistent scheduler or a new
async generation technique. The verified fused control and prior full-queue
experiments are preserved. The additional work remains in its separate worktree;
its implementation has not been merged into this branch.

The new [branch-mixing ablation](experiments/north/branch_mixing/README.md)
finds that nearly all of the fused gain survives with separate attention-output
and expert-front launches. In five uninterrupted-generation runs per prompt,
the mixed/separate ratio of medians is +0.094% for Python and +0.568% for Rust;
separate launches retain about 12.6% and 12.7% over the original MLX reference.
The particular alternating task-ID schedule brings no consistent benefit.

Those small incremental differences are not a reliable speedup claim. A large
Python timing outlier reverses the paired geometric estimate, and the
per-token alternating protocol gives different estimates. All samples are
retained. Full logits match byte-for-byte for all four custom variants on
127 positions per prompt in both final experiment processes.

MLX already permits independent Metal dispatches to overlap. The new explicit
serial-dependency control finds no repeatable gain from removing that dependency
across both prompts and timing protocols. Physical overlap was not measured.
We therefore cannot credit the earlier overall 11% improvement to this
particular Cohere-inspired scheduling mechanism.

Historical complete-queue checkpoint: 53.423 tok/s versus stock at 53.951 and
exact fused at 59.952. The queue is 10.89% behind fused. Its tested outputs match,
but that is not a full-model performance success or a general liveness proof.
An earlier static scheduler hung; a residency/barrier deadlock is suspected,
not conclusively diagnosed. See [NORTH_RESULTS.md](NORTH_RESULTS.md).

Current [five-variant source-checked summary](experiments/north/branch_mixing/summary-v2.json)
and [earlier four-variant evidence](experiments/north/branch_mixing/summary-v1.json)
include every measured run. No results have been posted or sent to Cohere.
