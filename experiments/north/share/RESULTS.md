# Correct, faster North Mini Code decoding on M4 Pro

The original prototype was incomplete and contained accumulation-order errors.
Those errors are fixed. Both the fused branch and the optimized persistent
scheduler now have repeatable full-model comparisons with identical generated
tokens. The persistent path also passes 1,424 complete-logit byte comparisons,
plus prefill; the fused path passes the same gate at the preceding checkpoint
with unchanged kernel arithmetic.

## Full-model results

Decode tokens per second, median of three measured runs per prompt:

| Prompt | Original MLX | Compiled MLX | Fused Metal | Persistent Metal |
|---|---:|---:|---:|---:|
| short | 51.92 | 52.00 | 58.48 | 56.06 |
| long | 47.54 | 48.69 | 52.99 | 50.83 |
| rust | 52.11 | 50.52 | 55.32 | 54.88 |

Median paired throughput changes (pair by prompt and repetition):

| Prompt | Fused / original | Persistent / original | Fused / compiled | Persistent / compiled |
|---|---:|---:|---:|---:|
| short | +11.7% | +6.6% | +12.5% | +7.3% |
| long | +11.7% | +6.0% | +8.8% | +3.9% |
| rust | +6.6% | +6.3% | +11.2% | +8.5% |

The ratio of medians and median paired ratio are different statistics; the
percentages above use paired ratios. All 36 measured requests use identical
token sequences across variants. Output cap is 256; the first token belongs
to prefill, and decode throughput measures the subsequent 255 forward passes.
The inputs contain 136, 1,672 and 140 tokens. Model load, tokenization and
warmup compilation are excluded. These are decode timings within complete
model generation, not end-to-end serving speedups. TTFT and full wall times
are retained in the raw JSONL. Desktop clocks are unlocked; ranges and every
paired observation are available in `summary.json` and the raw runs.

An additional confirmation uses only original MLX and the persistent path,
with two measured repetitions per prompt: one in each order. All six pairs
again improve, by 7.0–9.0%, with matching tokens. This separate confirmation is
not pooled into the headline table. Its sources match the full-logit gate.

## What was wrong

The old attention projection accumulated products in a different per-lane
order and used a different reduction tree. The old down projection split K
and reassociated partial sums. Those changes altered BF16 rounding and then
logits and greedy tokens. Component isolation found exact up/gate/activation
values but mismatches in down and attention. Preserving the native MLX
accumulation order, reduction tree and rounding boundaries removed the errors.
No tolerance was loosened and the model weights were not changed.

The selected persistent scheduler further reduces overhead by having one
thread acquire each dependency, publishing visibility through a device-memory
threadgroup barrier, staging the hidden vector once per threadgroup, and
caching readiness that cannot change back. Full-model timing omits diagnostic
visit counters; correctness stress runs enable them and require every task
to execute exactly once. Error/progress checks remain in the timed route.

## What the Cohere ideas showed here

The implementation follows [Cohere’s article](https://cohere.com/blog/megakernels)
and [pinned code](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c).
Independent MoE and attention-output tiles share a grid. A persistent ready
queue starts each expert’s down projection when its own hidden tiles are ready.
Counters directly record starts before all experts finish. Separate variants
load the same down weights into threadgroup memory before or after readiness.

The first implementation of those mechanisms had substantial overhead. Its
complete, correctness-passing full-model comparison is retained separately:

| Prompt | Original | Compiled | Fused | First scheduler | Prefetch | Stage after ready |
|---|---:|---:|---:|---:|---:|---:|
| short | 54.74 | 54.96 | 61.00 | 47.09 | 43.39 | 45.68 |
| long | 48.44 | 50.32 | 54.13 | 41.21 | 41.12 | 41.13 |
| rust | 52.36 | 54.27 | 58.94 | 46.18 | 42.52 | 42.99 |

These are a separate run session; do not subtract its rates from the final
table as if they were paired measurements. Fetching an expert's down-projection
weights early did not beat the best fused branch in our implementation. Cohere
also fetches the next layer's attention and routing weights while the current
layer is finishing. We have not implemented that cross-layer optimization on
Apple yet, so this experiment says nothing about whether it will help here.

Controlled branch ablations use 29 timed pairs after two warmups on each of
two captured-layer fixtures, varying activations and expert IDs. These numbers
are changes in latency, so negative is faster:

| Layer fixture | Per-expert readiness vs all experts | Prefetch vs load after ready |
|---|---:|---:|
| 1 | -5.2% | +2.9% |
| 7 | -2.9% | +3.5% |

Both sides include workspace initialization, the final join and diagnostic
counters. Each intermediate byte matches and every task runs exactly once.
For the fine-grained scheduler, the first down job starts with only 28–52 of
96 hidden tiles complete across these fixtures; the control waits for all 96.
For the prefetch variant, counters confirm weights are staged while their
activation dependencies are still unready. This distinguishes a measured
mechanism from merely reducing the number of launches.

The target-process GPU traces are retained locally, with aggregate summaries
in `correctness/results/profile-*.json`. The final original/fast traces record
roughly 75 versus 51 Compute-active command buffers per generated token and
17.50 versus 16.41 ms of Compute activity per step. Instrumentation substantially
increases CPU/driver overhead, so those trace wall times are not throughput
results. These traces do not provide per-shader timings or bandwidth counters;
the uninstrumented paired runs above are the performance evidence.

## Correctness, scope and reproduction

Five corrected variants each pass 1,424 decode steps, totaling 7,120
full-vocabulary byte comparisons, plus prefill checks. The three continuations
contain 512, 512 and 403 generated tokens; Rust stops naturally, and the two
Python requests reach the cap. This establishes equivalence on the checked
workloads, not broad coding-quality evaluation. Additional scheduler stress
varies activations, expert IDs and worker counts, including one worker and
oversubscription. Source hashes and snapshots accompany the gates and timings.
Metal API and GPU shader validation also pass for the dependency and staging
variants. The arbitrary-prompt runner passes raw-logit verification with both
the fused and persistent routes on an additional Python request.

This is a branch integration into all 48 MoE layers, not one kernel for the
entire forward pass. Original MLX still handles layer 0, prefill, normalization,
QKV, attention, routing and the output head. The checkpoint is the community
affine-W4/group64 conversion, not Cohere’s H100 BF16 setup. The only tested
device is this M4 Pro (20 GPU cores, 48 GB), macOS 27.0 / 26A5425a, MLX 0.32.2.

Use `experiments/north/quantized/README.md` for installation and commands,
`experiments/north/correctness/COHERE_REFERENCE.md` for the mechanism mapping,
and `experiments/north/generate.py --mode exact --verify --prompt '...'` to
try your own prompt. Use `--mode fast` for the persistent scheduler.

Model: `mlx-community/North-Mini-Code-1.0-4bit`, revision
`dfbe084dfa26e241345af99ca32848f38fd865f9`. Reference source: MLX-VLM revision
`cdc745ad8a32d162f6d8e9d08be256910d663ac2`. Model files are downloaded and
verified separately; the result packet contains no weights. Nothing has been
posted, pushed or sent to Cohere.
