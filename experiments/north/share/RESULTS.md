# North Mini Code: one safe Metal decode dispatch on M4 Pro

The complete 49-layer North Mini Code 4-bit transformer, all K/V updates, final
RMSNorm, and the 262,144-entry output head run in one custom Metal dispatch per
batch-one decode step.

| Path | Median decode tok/s | Change vs stock |
|---|---:|---:|
| Stock MLX | 53.951 | — |
| Exact fused control | 59.952 | +11.12% |
| **Safe one-dispatch queue** | **53.423** | **-0.98%** |

The headline combines six 127-step samples from two processes per path after a
warmup in each. The two adjacent safe-versus-stock point estimates are -1.26%
and -0.81%; both put the safe path within 1.3% of stock.

Correctness uses raw-byte comparison. Python and Rust prompts each pass 127
consecutive full-vocabulary logit comparisons. Complete state, every used cache
byte, all logits, and task completion pass at 1, 20, 32, and 36 threadgroups.

A step-interleaved ablation gives two useful mechanism results:

- smaller queue jobs are 3.31% faster than coarse jobs;
- next-QKV/router cache-warming reads lose 4.79% with one stage and 24.85% with
  ten stages.

The first static scheduler reached 57.472 tok/s, but later deadlocked after two
exact steps. That faster result is preserved and explicitly rejected. The safe
queue never waits for a particular threadgroup that Metal may not have
scheduled.

This validates the core complete-decode megakernel idea for this model, batch
one, short context, and one M4 Pro. The prefetch test is a cache-warming
approximation, not Cohere's H100 TMA pipeline. Continuous batching, paged
attention, ragged sequences, 256K context, and other Apple GPUs remain open.

Read [the full report](../whole_pass/README.md) and
[machine-readable evidence](../whole_pass/summary-v1.json).
