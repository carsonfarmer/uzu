# Cohere reference and Apple experiment contract

The Apple work is checked against [Cohere's megakernel article](https://cohere.com/blog/megakernels)
and [repository commit 67d0b9ca22ea3652796b715d1d1863459e0e2c3c](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c).
The local reference checkout is pinned at that commit.

Other pinned ground truth:

- [Apple's Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
  downloaded 2026-09-11, especially its atomics, memory fences, and barriers;
- [MLX 0.32.2 quantized kernels](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/quantized.h);
- MLX-VLM's North text implementation at
  cdc745ad8a32d162f6d8e9d08be256910d663ac2;
- [North Mini Code 4-bit](https://huggingface.co/mlx-community/North-Mini-Code-1.0-4bit/tree/dfbe084dfa26e241345af99ca32848f38fd865f9)
  at dfbe084dfa26e241345af99ca32848f38fd865f9.

## Mechanism mapping

| Cohere mechanism | Apple test | Result |
|---|---|---|
| Keep the decode forward pass in one persistent GPU launch | The complete 49-layer transformer, all cache updates, final norm, and 262,144 logits run in one Metal dispatch. | Bitwise exact on 254 repeated decode steps; 53.423 tok/s versus stock MLX at 53.951, or 0.98% slower. |
| Use a worker pool that consumes small tasks | Thirty-six 256-thread groups claim ready jobs from a device queue across 297 phases. | Complete phase and task counters match at 1, 20, 32, and 36 groups. |
| Reduce partial-wave waste with smaller jobs | Tuned 64/64/8-row jobs are compared with 128/128/16-row jobs, rotating variants every decode step. | Tuned is 3.31% faster. |
| Publish results before signalling dependencies | The queue uses device-coherent memory, device-scope fences, and atomic claim/completion cursors. | Hidden state, every used K/V byte, and all logits match MLX in the retained gates. |
| Overlap next-layer QKV/router weight movement | The closest safe MLX custom-kernel test rereads pieces of those weights in the current layer's final phase to warm cache. | Exact, but one stage is 4.79% slower and ten stages are 24.85% slower. This is not an H100 TMA reproduction. |
| Let GPU blocks wait on cross-SM dependencies | The first Metal version used static assignments and all-grid spin barriers. | Rejected: it reached 57.472 tok/s when it completed, then deadlocked in repeated exact testing. |
| Avoid dependence on an unscheduled block | The replacement queue gives resident groups only ready jobs; the last completion advances the phase. | Safe at the four tested group counts. An 80-group stress case was too slow, so arbitrary oversubscription is not claimed. |
| Continuous batching, paged attention, and ragged sequences | Outside the current harness. | No Apple result yet. |

## What is a direct test and what is an approximation

The one-dispatch task engine, dynamic worker queue, operation tiling, mixed
attention/MoE work, and full-model traversal are direct Metal versions of the
central scheduling ideas in Cohere's article.

The prefetch experiment is narrower. Cohere uses H100 TMA to move weights
asynchronously into shared memory while other work continues. MLX custom Metal
kernels do not expose that mechanism. Extra cache-warming reads answer whether
that substitute helps; they do not establish the result of a lower-level Metal
asynchronous pipeline.

The exact fused control is also separate. It combines selected attention-output
and MoE work inside specialized per-layer kernels, then returns to MLX for the
rest of the graph. It was inspired by the same opportunity to combine
independent work, but it is not the article's full-model persistent
megakernel. Its 59.952 tok/s rate is a strong control for the Apple experiment.

## Correctness contract

The gate is raw-byte equality with the pinned MLX reference. Equal top-one
tokens alone are insufficient.

- 18 primitive cases cover RoPE Q/K, cache insertion, expert IDs, and routing
  scores across six layers and three positions.
- Complete hidden-state, K/V-cache, 262,144-logit, and queue-completion checks
  pass at 1, 20, 32, and 36 groups.
- Python and Rust coding prompts contribute 254 consecutive full-logit-vector
  comparisons, including one cache growth event.
- All 30 retained benchmark rows generate the same 128 token IDs.

The summary tool checks every recorded source hash before calculating results.
See [the verifier](../whole_pass/summarize.py) and
[machine-readable summary](../whole_pass/summary-v1.json).

## Timing contract

Headline rates count 127 single-token decode iterations and combine six samples
from two processes per path. Model loading, tokenization, weight packing,
prefill, and warmup are excluded. Embedding lookup, the custom dispatch, K/V
carry and growth, synchronization, and argmax are included.

The stock/fused and packed paths use separate processes because retaining both
17 GB weight layouts creates about 34 GB of active allocations and severe
residency stalls. The run sequence is safe, stock, safe, stock. The two adjacent
safe-versus-stock point estimates are -1.26% and -0.81%.

The task-size and prefetch variants share one packed process and rotate after
every decode step. Their three measured samples therefore compare work performed
only milliseconds apart.
