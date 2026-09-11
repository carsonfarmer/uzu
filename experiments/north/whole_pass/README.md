# North Mini Code in one safe Apple Metal decode dispatch

This experiment runs the complete 49-layer North Mini Code 4-bit transformer,
updates all K/V caches, applies final normalization, and computes all 262,144
logits in one custom Metal dispatch for each batch-one decode step.

On the tested M4 Pro, the safe path is byte-for-byte equal to the pinned MLX
reference across two 127-step coding continuations. Its measured throughput is
53.423 tok/s versus 53.951 tok/s for stock MLX: a -0.98% point estimate, which
puts the safe path within 1% of stock at the resolution of this test. The strongest
exact fused control reaches 59.952 tok/s, 10.89% faster than the safe
megakernel.

The important result is that the whole transformer can stay inside one exact,
live Metal dispatch without giving up stock performance. The first faster
version did not meet that standard: its global barrier could deadlock, so its
57.472 tok/s result is preserved as an unsafe historical result and excluded
from the headline.

This tests ideas from [Cohere's megakernel article](https://cohere.com/blog/megakernels)
and [their implementation at the pinned commit](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c).
It is a Metal implementation specialized for the community 4-bit checkpoint,
not a line-by-line CUDA port.

## What “one dispatch” means

Normal MLX inference launches many GPU kernels for each token. Each launch
performs one operation or a small fused group of operations, then returns
control to the framework. This experiment launches one custom transformer
kernel. Thirty-six Metal threadgroups stay in that kernel and pull ready jobs
from a shared queue until the logits are finished.

The host still performs three small pieces of work outside that dispatch:

- looking up the current token's embedding;
- choosing the next token with argmax;
- processing the prompt, or prefill.

The claim is therefore one Metal dispatch for the transformer and full output
head per generated token. It is not one command for the entire application.

## One decode step, in plain language

The safe kernel moves through 297 phases:

1. **Prepare state.** Copy the used part of the 49 K/V caches into the output
   buffers required by MLX's custom-kernel API and copy the embedded token.
2. **Run dense layer 0.** Normalize the hidden state; compute Q, K, and V; apply
   RoPE; insert the new K/V values; run attention; run the dense SwiGLU MLP;
   project the results; and add the residual. This takes six phases.
3. **Prepare each MoE layer.** Normalize once, then split the Q, K, V, and
   128-way router projections into small row jobs. Any free threadgroup can
   claim the next job.
4. **Update attention state.** Apply RoPE and write the new K/V values. Select
   the same top eight experts and routing scores as MLX.
5. **Fill the GPU with independent work.** Attention heads and selected-expert
   up/gate jobs are independent, so the queue mixes them. A group that finishes
   one kind of work can take the other kind instead of waiting.
6. **Finish the layer.** Mix expert down-projection jobs with attention output
   jobs, combine the expert scores, then add attention, MoE, and the residual.
   Steps 3–6 use six phases per MoE layer, repeated for all 48 layers.
7. **Produce logits.** Apply final RMSNorm, split the 262,144-row quantized
   output projection into jobs, and write the complete BF16 logit vector.

The queue stores the current phase, the next unclaimed job, and the number of
finished jobs. A resident group atomically claims work that is ready. The group
that finishes the last job advances the phase. Progress never depends on a
particular threadgroup being scheduled.

## Why the scheduler changed

The first complete implementation assigned work to every dispatched group and
used a global spin barrier between phases. It passed long exactness tests and
produced a stable 57.472 tok/s benchmark. An adversarial rerun then completed
two exact steps and hung on the third. A 20-group rerun hung after one step.

That is possible because Metal does not promise that every dispatched
threadgroup is resident at once. A resident group can wait at the barrier for a
group the GPU has not scheduled, while the waiting group occupies the resources
needed to schedule it.

The dynamic queue removes that dependency. It has passed complete output,
cache, logit, and completion-counter checks at 1, 20, 32, and 36 groups. The
canonical geometry uses 36 groups. This test does not claim arbitrary
oversubscription: an 80-group stress case became unacceptably slow and was not
adopted.

[The static-scheduler incident record](history/STATIC_SCHEDULER_FAILURE.md)
links the completed fast run and the partial stalled artifact.

## Performance

Hardware: Apple M4 Pro, 20 GPU cores, 48 GB unified memory. Software: MLX
0.32.2. Workload: batch one, a 136-token coding prompt, 127 timed decode steps,
and cache growth from 256 to 512 entries.

The headline rates combine six measured runs from two processes per path after
one warmup in each process. The processes ran in the sequence safe, stock,
safe, stock. The two adjacent safe-versus-stock pair estimates were -1.26% and
-0.81%; their combined medians differ by -0.98%.

| Path | Median decode tok/s | Change vs stock |
|---|---:|---:|
| Stock MLX | 53.951 | — |
| Exact fused control | 59.952 | +11.12% |
| **Safe complete one-dispatch path** | **53.423** | **-0.98%** |

The safe path is 0.98% slower and both adjacent pairs put it within 1.3% of
stock. The safe one-dispatch path is 10.89% behind
the exact fused control.

The scheduling ablations use the same packed weights in one process and rotate
the four configurations after every decode step. That keeps each comparison
only milliseconds apart.

| Queue configuration | Median decode tok/s | Change vs tuned |
|---|---:|---:|
| Tuned 64/64/8-row tiles | 50.518 | — |
| Coarse 128/128/16-row tiles | 48.898 | -3.21% |
| Tuned + one cache-warming stage | 48.098 | -4.79% |
| Tuned + ten cache-warming stages | 37.964 | -24.85% |

The tuned queue is 3.31% faster than the coarse queue. This supports Cohere's
wave-quantization idea on this workload: smaller jobs leave fewer GPU groups
idle near the end of an operation.

## What the prefetch experiment says

Cohere overlaps a future layer's QKV/router weight transfer with current work
using H100 TMA and shared memory. MLX custom Metal kernels do not expose an
equivalent asynchronous pipeline.

The safe approximation here rereads 64-column pieces of the next layer's
QKV/router weights during the current layer's final phase. That warms cache but
also consumes memory bandwidth. One stage loses 4.79%; ten stages lose 24.85%.

This is evidence against extra cache-warming reads in this implementation. It
does not predict how a lower-level Metal asynchronous staging pipeline would
perform.

## What the fused control is

The exact fused control combines the attention-output and MoE branches inside
specialized per-layer kernels while leaving the model as a sequence of MLX
operations. It was developed as an incremental control while exploring the
same Cohere article: it tests whether targeted fusion can beat a general
persistent scheduler.

It is not Cohere's full-model megakernel architecture. The Cohere-derived
experiment is the one-dispatch task queue, including the tile and prefetch
ablations. The control winning by 10.89% tells us that fewer launches alone do
not guarantee the fastest Apple implementation; the work inside each launch
still matters.

## Correctness

Changed tokens are a hard failure. The retained gates compare raw bytes:

- 18 primitive cases compare RoPE Q/K, V insertion, top-eight expert IDs, and
  routing scores across six layers and three positions.
- A complete one-step gate compares the final hidden state, every used K/V
  cache byte, all 262,144 logits, the final phase, and claimed/completed task
  counts at 1, 20, 32, and 36 groups.
- Python and Rust coding prompts each pass 127 consecutive full-vocabulary
  logit comparisons: 254 exact decode steps.
- All 30 measured benchmark rows produce the same 128 token IDs as the exact
  reference.
- Every canonical artifact records SHA-256 hashes for all source files that can
  affect its result. The summarizer refuses stale artifacts.

The Python continuation crosses cache growth from 256 to 512. Earlier testing
also found a dense-down stride bug that passed 15 steps and changed one BF16
byte at step 28; all retained results were regenerated after the fix.

## What this validates

For North Mini Code 4-bit, batch one, and short contiguous contexts on this M4
Pro, the experiment validates:

- a complete transformer and full-logit Metal megakernel;
- bitwise equality with MLX on the retained workloads;
- a liveness-safe ready-task queue at 1, 20, 32, and 36 groups;
- throughput within 1% of stock MLX in the combined result;
- a 3.31% benefit from finer task sizing inside the safe queue;
- negative results for the tested cache-warming prefetch approximation.

The experiment does not yet cover continuous batching, paged attention, ragged
sequences, sampling, an API server, 256K context, other Apple GPU generations,
or a true TMA-like Metal pipeline.

## Reproduce

Prepare the pinned environment and model using
[the quantized experiment instructions](../quantized/README.md), then run from
the repository root:

~~~sh
# Verify every retained source hash, exactness record, token sequence, and ratio.
work/north-venv/bin/python experiments/north/whole_pass/summarize.py

# Full state/cache/logit queue gate at four group counts.
work/north-venv/bin/python experiments/north/whole_pass/check_whole_pass.py \
  --layers 48 --workers 1 20 32 36 --prefix --head \
  --output work/check-safe-megakernel.json

# A 127-step full-logit gate. Safe complete settings are now the defaults.
work/north-venv/bin/python experiments/north/whole_pass/verify_decode.py \
  --tokens 128 --output work/full-safe-megakernel.jsonl

# Standalone safe-path timing.
work/north-venv/bin/python experiments/north/whole_pass/benchmark_deployed.py \
  --tokens 128 --runs 3 --profiles tuned \
  --output work/benchmark-safe-megakernel.jsonl

# Step-interleaved scheduling and prefetch ablations.
work/north-venv/bin/python experiments/north/whole_pass/benchmark_deployed.py \
  --tokens 128 --runs 3 --profiles tuned coarse prefetch1 prefetch10 \
  --output work/benchmark-safe-profiles.jsonl
~~~

[The machine-readable summary](summary-v1.json) lists the canonical artifacts,
raw samples, source hashes, exactness counts, and rejected static result.
