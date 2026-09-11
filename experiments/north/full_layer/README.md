# Full-layer persistent experiment

**Superseded milestone:** `../whole_pass/` now integrates all 49 transformer
layers, K/V updates, and the output head in one safe Metal dispatch. It is
bitwise exact on the retained repeated-decode gates and reaches stock-MLX
within 1% of stock MLX. The results below remain the one-dispatch-per-layer
checkpoint that led to it.

This experiment extends the verified persistent MoE/attention-output branch
to include one-pass attention, the complete MoE branch, residual join, and the
next layer's RMSNorm, Q/K/V projections, and router logits in one Metal
dispatch. It uses the real 4-bit North Mini Code weights on an M4 Pro.

`static_full_attention.py` is the selected schedule. Thirty-two persistent
256-thread groups follow fixed, strided task lists with three explicit device
dependency boundaries. The 256 threads simulate MLX attention's 32 logical
SIMD groups without changing its reduction order. This schedule was faster
than the dynamic ready-task claimer in `full_attention.py`.

## Current verified checkpoint

- `check-full-static-layer1-v1.json`: 9/9 submissions match raw bytes for
  attention, hidden activations, attention projection, all routed expert
  outputs, final output, next RMSNorm, next Q/K/V, and next router. Worker
  counts 1, 20, and 32 all execute every audited task exactly once.
- `check-full-static-layer47-v1.json`: the same gate passes 4/4 submissions at
  the late-model boundary.
- `full-static-short64-v1.jsonl`: original MLX and the full-layer path produce
  identical full-vocabulary logit bytes for prefill and 63 consecutive decode
  steps.
- `benchmark-full-static-short-w32-v1.jsonl`: over three rotated 127-step
  measurements, median throughput is 55.782 tok/s for original MLX, 61.893
  tok/s for the strongest exact fused baseline, and 59.482 tok/s for the
  full-layer path. The median paired ratios are +6.97% versus MLX and -3.87%
  versus exact fused.

The checkpoint is still one persistent dispatch per transformer layer. MLX
applies RoPE, updates each KV cache, and selects router experts between those
dispatches. Cohere's implementation keeps the complete multi-layer decode
task graph inside one CUDA launch. A whole-pass Metal experiment therefore
needs packed per-layer weights, in-kernel RoPE/KV updates, and in-kernel top-k;
that is the next gate before claiming a complete Apple megakernel validation.

## Reproduction

Run from the repository root with the pinned environment in `work/north-venv`:

```sh
work/north-venv/bin/python experiments/north/full_layer/check_full_attention.py \
  --layer 1 --steps 3 --workers 1 20 32 --static

work/north-venv/bin/python experiments/north/full_layer/verify_decode.py \
  --tokens 64 --workers 32 --schedule full_static --prompts short

work/north-venv/bin/python experiments/north/full_layer/benchmark_decode.py \
  --tokens 128 --runs 3 --workers 32 --modes exact full_static --prompts short
```
