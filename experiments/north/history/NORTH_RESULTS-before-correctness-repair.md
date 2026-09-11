> **Reopened 2026-09-11: work remains incomplete.** The previous completion/approval framing was premature. The custom decoder fails correctness and omits key Cohere mechanisms; its slower timings are not evidence against the complete approach. Numerical debugging and the implementation work are active. Do not use the earlier tweet draft as a completed research conclusion.

# North Mini Code megakernel ideas on Apple M4 Pro

**We reached a runnable full-model experiment. The custom Metal path is slower
than MLX and changes generated tokens. A repeatable 6–10% reduction in an
isolated BF16 branch's GPU time did not transfer to faster, equivalent decoding.**

The useful result is the measured gap between a scheduling experiment and an
inference optimization, plus a concrete BF16 arithmetic issue found along the
way. This is an Apple implementation of selected scheduling ideas, not a port
of Cohere's complete CUDA megakernel.

![Measured results](experiments/north/share/north-m4-pro-results.png)

## What the earlier result meant

North has parallel attention and expert-MLP branches. Our first real-data
prototype lets a GPU worker finish a small up/gate tile and immediately compute
its down-projection contribution, with independent attention projection jobs
in the same work pool. It avoids a whole-grid wait between those local stages.

With identical BF16 math, changing the schedule reduced GPU time by 6–10% on
layer 1 and about 8% on layer 7, across three runs each. That measured only part
of one layer. It omitted the router, QKV, attention/KV processing, normalization,
and the other layers. The prototype also expanded quantized weights into BF16.
An improvement there could not establish better token throughput.

Cohere's full design adds a persistent task interpreter, fine-grained dependency
signaling and weight-load overlap across a much larger decode graph. Its advice
to begin with competitive individual operations is directly relevant here.
[Original design article](https://cohere.com/blog/megakernels),
[pinned CUDA implementation](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c).

## What now runs

The new kernel reads the checkpoint's original packed affine 4-bit/group64
experts directly. It computes eight selected experts and the attention output
projection, then performs the residual join in a second dispatch. A phased
control executes the same arithmetic in four dispatches. A tile/worker sweep
selected 64-feature chunks and 128 workers with interleaved ready jobs.

An explicit local patch now runs that branch in all **48 MoE layers** during
single-token decode. The first dense layer and multi-token prefill use the
original implementation. Routing, attention/KV, normalization and the model's
other operations remain MLX operations. This is not one kernel for the whole
model and does not implement cross-threadgroup dependency waits or prefetch.

We compare three routes using the same loaded model: unmodified MLX, a wrapper
with compiled native MLX branch operations, and the compiled custom branch.
The compiled control checks whether any benefit comes from compiler/Python
changes. The wrapper is reversible; no Uzu engine or installed runtime source
was modified.

## Full-model generation results

M4 Pro / 20 GPU cores / 48 GB, macOS 27.0 (26A5425a), MLX 0.32.2,
Python 3.12.10. Fresh KV per request, batch one, greedy decoding, reasoning
disabled, checkpoint chat template, 256-token prefill chunks. One warmup and
three measured requests per variant/prompt; variant order rotates. All runs
here generated 256 tokens and hit the cap. The first token is counted in TTFT;
throughput measures the subsequent 255 decode steps.

| Prompt | Input tokens | Original MLX | Compiled native control | Custom Metal |
|---|---:|---:|---:|---:|
| Short Python | 136 | 53.77 tok/s | 53.96 tok/s | 38.04 tok/s |
| Long Python | 1672 | 48.65 tok/s | 50.65 tok/s | 42.84 tok/s |
| Rust | 140 | 52.40 tok/s | 49.20 tok/s | 38.26 tok/s |

These are medians, not peak rates. The chart includes each cell's minimum and
maximum across the three runs. For example, the Rust compiled-control runs
ranged from 38.24 to 51.72 tok/s. Desktop timing varied; GPU clocks were not
locked and the machine was not exclusive. No general compiler speedup is
claimed. All raw samples, including slow runs, remain in the packet.

The custom continuations differ, so these are practical free-generation rates
at the same token cap, not matched-token timing or an equivalent-output speedup.
The original and compiled-native continuations match each other in every run.
Each route repeats its own token sequence across its three runs.

An earlier, separate 512-token baseline measured 55.45 / 49.02 / 54.13 tok/s,
with peak MLX allocations of 18.79 / 19.19 / 18.80 GB. Those are not the controls
for the new 256-token comparison. Separate complete short Python and Rust
answers passed 1,225 Python input cases and five Rust tests; no coding-quality
claim is made for the custom kernel's capped outputs.

## Correctness: what passed and what failed

All custom scheduling variants match their same-tile-size phased control
bit-for-bit, including hidden values, down-projection partials, attention
projection and final outputs. The frozen quantized comparison passed these
checks at both layers over six runs and 1,116 timed samples. Separate Metal API
and GPU validation also completed successfully; its timings are excluded.

Native MLX preserves BF16 rounding inside input-sum and sigmoid arithmetic.
Our earlier float32 intermediates changed results despite computing the same
real-number formula. Reproducing those steps made layer 1's captured branch
output exact. Layer 7's relative L2 error fell to 0.00000900 (0.0009%), with
maximum absolute difference 0.00012207. This remaining difference is not proof
of full-model equivalence.

For the model test, every route received the **same reference continuation**
for 64 decode steps per prompt. This compares full-vocabulary logits without
letting different generated inputs explain the difference:

| Prompt | Compiled-native greedy agreement | Custom greedy agreement | Custom median logit relative L2 | First free-generation difference, 1-based |
|---|---:|---:|---:|---:|
| Short Python | 64/64 | 64/64 | 4.55% | 147 |
| Long Python | 64/64 | 63/64 | 3.61% | 34 |
| Rust | 64/64 | 63/64 | 2.71% | 63 |

The compiled-native logits are bitwise exact on all 192 checks. The custom
route matches 190/192 greedy choices, but none of its full logit vectors are
exact. Its worst logit relative L2 is 59.76%; maximum KL(reference || custom)
is 0.06872 nats. All outputs are finite. Small branch discrepancies can propagate
through the model, so token agreement on short prefixes is insufficient.
The custom route fails the numerical-equivalence gate and is not promoted.

## Repeated quantized branch comparison

Both paths execute in MLX with explicit dynamic array inputs. Eight warmups,
31 measured samples per configuration, rotated/reversed order; compilation,
loading and checks excluded. Timing is host wall time, including graph creation,
evaluation and waiting. It must not be equated with the earlier GPU-only study.

The frozen mixed candidate takes **16–21% more wall time** than compiled-native
MLX by the median within-repetition ratio in each of six runs. These are
exploratory desktop measurements, not confidence intervals. Layer-1 run 2 has
substantial timing drift: dividing its separate medians reverses the apparent
ranking. We retain it and report the paired statistic, all raw samples, and
the drift rather than selecting a favorable subset.

The original BF16 schedule gain also does not survive unchanged: static 128 is
not consistently better than phased execution with the packed-weight kernels.
Twenty workers are much slower in both studies; the 20 physical GPU cores do
not prescribe 20 software workgroups for this geometry. No universal occupancy
rule or claim about all Apple devices follows from this one Mac.

## What would justify continuing

The next useful experiment is to make the quantized branch numerically match
MLX across many layer/token captures while improving the standalone projection
kernels. Profile the down-projection tiling, memory traffic and worker geometry,
then re-run the existing full-model gates. A general dependency scheduler or
weight prefetch remains an untested extension; today's measurements neither
establish nor rule out the full megakernel approach on Apple hardware.

## Reproduce and share

- [Runnable instructions](experiments/north/quantized/README.md)
- [Tweet drafts and chart](experiments/north/share/SHARE.md)
- [Machine-readable summary](experiments/north/share/summary.json)
- [Full decode evidence](experiments/north/quantized/results/decode.jsonl)
- [Earlier BF16 study](experiments/north/share/BF16_STAGE.md)

Checkpoint: mlx-community/North-Mini-Code-1.0-4bit,
revision `dfbe084dfa26e241345af99ca32848f38fd865f9`.
Runtime: unmodified mlx-vlm text-model code at
`cdc745ad8a32d162f6d8e9d08be256910d663ac2`.
The public inputs were re-verified against file hashes. Requirements, raw runs,
source snapshots and Metal validation logs are included. `prepare.py` can
fetch/verify the same inputs over public HTTPS; the packet contains no weights.
No results have been posted or sent to Cohere.
