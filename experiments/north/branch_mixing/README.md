# What does mixing attention and expert work contribute?

September 13, 2026. This experiment tests the contribution of cross-branch
fusion inside the existing exact control. It is not a new optimized decoder.

## Finding

**Most of the fused improvement survives when attention-output and expert-front
work use separate launches. We cannot credit the historical 11% gain to our
particular branch-mixing schedule.** Alternating the task IDs brought no
consistent benefit. Combining the branches inside one kernel had a small,
protocol-dependent effect that did not establish a reliable extra gain in
uninterrupted generation.

The final five-variant uninterrupted-generation test produced these medians
over five measured continuations per variant and prompt:

| Variant | Python tok/s | Rust tok/s |
|---|---:|---:|
| Original MLX | 54.639 | 54.391 |
| Existing mixed fused control | 61.580 | 61.648 |
| Same fused control, blocked task IDs | 61.777 | 62.171 |
| Separate independent front launches | 61.522 | 61.300 |
| Separate front launches with a forced dependency | 60.892 | 61.131 |

The ratios of the medians put mixed versus separate at **+0.094% Python**
and **+0.568% Rust**. Separate still exceeds original by about **12.6% and
12.7%** respectively. These percentages describe the new matched experiment;
they are not replacements for the historical benchmark's 11.12% estimate.
The interleaved task-ID order is slightly slower than blocked IDs in these
medians. This does not establish that blocked placement is generally better.

There is material timing uncertainty. One Python mixed run fell to 26.763
tok/s, with a 304 ms step near the cache-growth boundary; its cause was not
established. With that run retained, the geometric mean of paired mixed/separate
ratios is **-13.914% Python**, versus **+0.389% Rust**. The median is not evidence
that this slow run can be ignored. We report both summaries and do not claim
a reliable sub-percent improvement.

The token-alternating protocol gave different paired geometric estimates:

| Comparison | Python | Rust |
|---|---:|---:|
| Mixed / separate | +2.492% | +2.913% |
| Mixed / blocked IDs | -0.265% | -0.741% |
| Separate / forced dependency | -0.544% | +0.032% |

The Rust mixed/separate estimate includes a +12.14% pair during a large system
slowdown; its other four pairs range from -0.17% to +2.38%. These results do not
justify promoting “mixing adds 3%” as a standalone decode claim. Likewise,
allowing the two separate front launches to overlap did not provide a
repeatable gain across prompts and protocols. No GPU timeline was captured,
so the experiment does not establish how much physical overlap occurred.

The specific Cohere connection remains a design motivation: its article
highlights filling idle capacity with independent branch work. This experiment
does not show that this mechanism caused our earlier overall speedup. Most of
the practical gain persists in the retained within-expert fusion and final
down/weighted-sum/residual fusion. We have not individually apportioned the
benefit among those remaining changes.

All four custom variants passed 127 full-vocabulary, byte-for-byte logit checks
against native MLX on each of two prompts in each of the two v2 processes:
2,032 variant/step comparisons, plus identical token IDs in all 100 measured
v2 continuations. These are repeated checks on two continuations, not 2,032
consecutive positions or a general correctness proof.

The separate throughput task is now pursuing an additional 10% over the
strongest fused control, with matched measurements and a hypothesis ledger.
This ablation does not meet that future goal.

Canonical data: [five-variant summary](summary-v2.json),
[token-alternating runs](step-v2a.jsonl),
[uninterrupted-generation runs](generation-v2b.jsonl).
The [initial four-variant summary](summary-v1.json) retains all earlier data,
including the unstable generation run. No measured run was discarded.

## The comparison

All custom variants use the same expert matrix/activation functions, attention
output matrix function, BF16 rounding, and final expert-down/weighted-sum/residual
kernel. The front has 96 expert tasks and 64 attention-output tasks, each with
256 threads. The existing control uses 160 groups, so every group receives one
task. The down kernel uses 16 output rows per group. The rest of the model,
weights, prefill, prompts, and generated tokens are unchanged.

| Variant | Front execution | Total launches in the replaced branch |
|---|---|---:|
| `original` | Pinned native MLX model | Native operations |
| `mixed` | Existing exact control; alternating expert and attention task IDs within one launch | 2 |
| `blocked` | Same launch, with all expert task IDs before all attention task IDs | 2 |
| `separate` | Expert front and attention output in separate independent launches | 3 |
| `serial` | Same two front launches, with attention required to wait for the expert front | 3 |

`blocked` changes placement; it does not prevent both branches from executing
at the same time. Nor does `separate` force serial execution: MLX 0.32.2 uses
a concurrent Metal command encoder. This was an important correction to the
initial proposed experiment. See [the pinned encoder implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp#L576).

`serial` passes the produced expert-hidden buffer as an additional declared
input to the attention kernel. No arithmetic reads it. MLX registers every
custom-kernel input with its resource hazard tracker, so this dependency causes
a GPU barrier or a corresponding encoder-fence dependency before attention.
There is no added CPU wait, dummy computation, or spin loop. See
[custom-kernel input registration](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/custom_kernel.cpp#L63)
and [hazard tracking and barriers](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp#L345).

## What each comparison can establish

- `mixed / blocked`: contribution of our particular task-ID ordering, holding
  launch count and math fixed.
- `mixed / separate`: contribution of putting both branches in one custom
  kernel, including launch count, grid scheduling, and compiler resource use.
  It does not isolate those causes from each other.
- `separate / serial`: contribution of allowing independent front launches,
  compared with imposing an unnecessary dependency. A concurrency-capable
  encoder does not by itself prove physical hardware overlap.
- `mixed / original` and `separate / original`: how much improvement remains
  when the other fused calculations are retained with separate front launches.

The serial control only orders these two front computations. It does not
serialize the complete attention and expert branches throughout the model.
These tests do not reproduce Cohere's full persistent task interpreter or
asynchronous next-layer weight pipeline.

## Measurement and correctness

One shared copy of the pinned North Mini Code 4-bit model runs on the M4 Pro
with 48 GB memory and MLX 0.32.2. Python and Rust prompts contain 136 and 140
tokens respectively. Each continuation generates 128 tokens: the first is
obtained from prefill, followed by 127 timed decode steps. Prefill, warmup,
correctness checks, loading, and tokenization are outside decode timing. Decode
includes model evaluation, cache updates, and argmax with CPU synchronization.

In `step` runs, each variant has its own KV cache and generates the same next
token before moving on. Variant order rotates after every token and repetition.
This exposes all variants to similar short-term system conditions. In
`generation` runs, each full continuation runs without switching variants;
order rotates between repetitions. These protocols are summarized separately.

Before timing, every custom variant is compared byte-for-byte with all 262,144
native MLX logits at every one of the 127 decode positions, on both prompts.
All timed token IDs must also match the previously retained exact reference.
Sources and raw outputs have SHA-256 provenance. The original fused source
files are imported unchanged; no patch to the previous result is needed.

The initial four-variant design is retained in `ablate.py`, `step-v1.jsonl`,
and `generation-v1.jsonl`. The five-variant design including the explicit
dependency is `ablate_v2.py`. `generation-v1.jsonl` contains large system-time
swings, including matched-pair ratios with opposite signs. Every sample is
retained. A host snapshot is in `host-during-generation-v1.json`; it establishes
concurrent system activity, not the cause of any particular slowdown.

## Reproduction

Use the existing pinned environment described in
[the quantized experiment setup](../quantized/README.md). The v2 provenance also
requires the matching MLX source checkout at `work/mlx-src`, tag `v0.32.2`,
commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85`. If absent, clone
`https://github.com/ml-explore/mlx.git` at that tag into that directory; the
source is inspected and hashed, not built or substituted for installed MLX.
GPU runs acquire an
exclusive lock on `/tmp/north-metal-research-gpu.lock`; other local GPU research
must use the same lock. Each output path must be new, to avoid overwriting data.

```bash
work/north-venv/bin/python experiments/north/branch_mixing/ablate_v2.py \
  --timing step --runs 5 \
  --output experiments/north/branch_mixing/step-new.jsonl

work/north-venv/bin/python experiments/north/branch_mixing/ablate_v2.py \
  --timing generation --runs 5 --order-offset 2 \
  --output experiments/north/branch_mixing/generation-new.jsonl

work/north-venv/bin/python experiments/north/branch_mixing/summarize.py \
  experiments/north/branch_mixing/step-new.jsonl \
  experiments/north/branch_mixing/generation-new.jsonl \
  --output experiments/north/branch_mixing/summary-new.json
```

## Attribution

[Cohere's article](https://cohere.com/blog/megakernels) explicitly motivates
mixing North's independent attention/expert work to use otherwise idle GPU
capacity. That motivated this research and matches the structure of the fused
control. The arithmetic primitives are adapted from Apple's MLX, as recorded
in [NOTICE.md](../quantized/NOTICE.md). Fusion is established practice; these
experiments assess one implementation and do not establish a novel algorithm.
