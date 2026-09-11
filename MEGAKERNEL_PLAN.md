# Apple North megakernel: objective and result

## Objective

Test whether the central ideas in Cohere's North Mini Code decode megakernel can
work on an Apple GPU. The success gate requires a complete batch-one transformer
and output head in one Metal dispatch, bitwise-correct logits during repeated
generation, a scheduler that completes reliably at the tested geometry, and
fair comparisons with stock MLX and the strongest exact fused path.

References:

- [Cohere's megakernel article](https://cohere.com/blog/megakernels)
- [Pinned Cohere source](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c)
- [Pinned North Mini Code 4-bit checkpoint](https://huggingface.co/mlx-community/North-Mini-Code-1.0-4bit/tree/dfbe084dfa26e241345af99ca32848f38fd865f9)

## Result

The gate is met for the tested M4 Pro workload.

| Question | Evidence |
|---|---|
| Can the complete model stay in one Metal dispatch? | Yes: 49 layers, all K/V updates, final norm, and the 262,144-row head. |
| Does it preserve model output? | Yes on the retained workloads: 254 consecutive full-logit vectors and all audited cache bytes are bitwise equal to MLX. |
| Does the scheduler complete reliably? | Yes at 1, 20, 32, and 36 groups using the dynamic ready-task queue. |
| Does it match stock speed? | Yes within measurement resolution: 53.423 versus 53.951 tok/s, a -0.98% point estimate. |
| Does it beat the exact fused control? | No: the control reaches 59.952 tok/s, leaving the safe queue 10.89% behind. |
| Do finer task sizes help? | Yes: +3.31% versus coarse jobs in the step-interleaved ablation. |
| Does the tested next-weight cache warming help? | No: one stage is -4.79%; ten stages are -24.85%. |
| Can a CUDA-style all-grid spin barrier be assumed safe? | No: the static Metal scheduler deadlocked after previously completing exact fast runs. |

## Work completed

1. Matched MLX's affine-W4 arithmetic, reduction order, and BF16 rounding.
2. Fused the per-layer parallel attention and MoE branches as an exact control.
3. Added QKV/router projection, RoPE, cache writes, top-eight routing, attention,
   MoE, and residual joins for all 49 layers.
4. Packed the weights and caches so one Metal kernel can traverse the model
   within Metal's buffer-binding limits.
5. Added final RMSNorm and the full 262,144-logit output projection.
6. Found a dense-down stride error at decode step 28 and regenerated every
   retained result after fixing it.
7. Rejected the faster static scheduler after a liveness failure and replaced
   it with a ready-task queue whose progress does not require unscheduled groups.
8. Added source hashes, two 127-step exactness gates, queue completion checks,
   adjacent-process headline timing, and step-interleaved ablations.

## What came from Cohere

The one-dispatch task engine, resident worker pool, small independent jobs,
dependency-controlled progress, wave-filling experiment, and next-layer
QKV/router overlap experiment directly test mechanisms described by Cohere.

The exact fused path is a control developed while exploring the same work. It
combines selected operations inside specialized per-layer kernels, but it is not
a full-model persistent megakernel. Its win shows that a general scheduler can
lose to efficient targeted fusion even after launch overhead falls.

## Next engineering questions

The validation covers one model, batch one, and short contiguous contexts. The
next useful work is:

1. replace per-step cache-prefix copying with in-place or alias-safe cache
   updates;
2. prototype true asynchronous weight staging with lower-level Metal APIs;
3. measure longer contexts and additional Apple GPU generations;
4. add continuous batching, paged caches, ragged attention, and sampling.

These are extensions to the validated experiment. The current evidence and its
limits are recorded in
[the whole-pass report](experiments/north/whole_pass/README.md).
