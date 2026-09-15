# North Mini Code: real baseline and Metal branch experiment

**Scope correction:** this is a custom MLX runner, not Uzu engine integration.
Its timings do not establish an improvement over the normal generation path of
Uzu or MLX-VLM. See [the baseline correction](../../BASELINE_CORRECTION.md).

This directory establishes a real quantized North reference and now includes a
complete batch-one decode graph in one Metal dispatch. The safe M4 Pro path
runs all 49 transformer layers, K/V updates, final normalization, and the full
language-model head; it matches MLX logits byte-for-byte on two repeated coding
continuations and reaches within 1% of stock MLX. Start with
[whole_pass/README.md](whole_pass/README.md). See quantized/README.md for model
setup and correctness/COHERE_REFERENCE.md for the mapping to Cohere's code and
article. The bounded branch, BF16 materialization, and Swift graph described
below record earlier milestones.

## Reference model

- Model: mlx-community/North-Mini-Code-1.0-4bit,
  revision dfbe084dfa26e241345af99ca32848f38fd865f9.
- Runtime source: mlx-vlm, revision cdc745ad8a32d162f6d8e9d08be256910d663ac2.
- Python packages: results/python-environment.txt.
- Cached weights: work/models/North-Mini-Code-1.0-4bit.
- Every downloaded LFS file is checked against the pinned SHA-256 and size.
  See results/model-provenance.json.

reference.py imports the pinned, unmodified text-model implementation directly.
It bypasses the package's multimedia frontend imports, not model math. The
language-model prefix is removed from checkpoint names, the source sanitize
method is applied, modules are quantized according to actual scale tensors,
and all weights load strictly. This is a minimal local text harness, not an
installed mlx-vlm serving application.

Run from the repository root:

```sh
work/north-venv/bin/python experiments/north/baseline.py --runs 3 --tokens 512
```

The driver uses the checkpoint chat template, reasoning=false, greedy argmax,
fresh KV caches, and 256-token prefill chunks. It records first-token latency,
subsequent decode steps and time, total wall time, MLX memory and full token IDs.
The first repetition per prompt is warmup. Tokenization/model loading are
outside request timings. The first output token belongs to prefill; decode
throughput counts only subsequent one-token forward passes.

The 512-token cap is part of the timing protocol. A separate --runs 0 request
with a larger --tokens value can collect complete code without changing the
timed workload. --prompt selects short, long or rust; --output selects a new
results file.

## Real layer extraction

```sh
work/north-venv/bin/python experiments/north/export_layer.py
```

This captures layer 1 during an actual single-token decode following the
prompt. It saves the normalized shared input, residual, attention output
projection input, actual top-eight expert IDs/scores, and reference outputs.
Only the selected expert weights are materialized to BF16. That changes the
weight representation compared with the native quantized reference, and must
not be presented as a pure scheduler change. A second, independent MLX
materialized-weight calculation measures the effect of that distinction.

The reference branch timing includes selected quantized experts, fixed routing
scores, attention output projection and residual. It excludes routing and
earlier attention stages. It measures CPU plus GPU wall time, not GPU-only time.

## Metal graph

The dimensions are real North dimensions: hidden 2048, expert intermediate 768,
eight selected experts, attention output projection 4096 to 2048. Each expert
is split into 24 chunks of 32 intermediate values. A local chain computes
up/gate, BF16 SiLU multiplication, then the chunk's contribution to all 2048
down-projection outputs. Attention output tiles are independent ready work.
An ordered final dispatch reduces partials, applies routing scores and joins
both branches with the residual.

All controls use exactly the same BF16 weights and rounding, FP32 split-K
partials, output layout and reduction order:

| Control | Compute dispatches | Scheduling |
|---|---:|---|
| phased | 4 | up/gate; down partials; attention O-proj; join |
| local_fusion | 3 | local expert chains; attention O-proj; join |
| mixed_grid | 2 | expert chains and attention tiles in one grid; join |
| mixed_interleaved | 2 | same grid, interleaved task placement; join |
| static | 2 | strided persistent workers; join |
| static_interleaved | 2 | same workers, interleaved placement; join |
| dynamic | 2 | atomic work claims with interleaved placement; join |

Dependencies inside an expert chunk use threadgroup memory/barriers. Global
fan-in uses an ordinary dispatch boundary. No workgroup spins waiting for
another; the queue counter only allocates already-ready tasks. This tests
branch placement and local producer/consumer chains, not Cohere's full
cross-SM counter protocol or a complete decode megakernel.

```sh
# Synthetic tensors (build and run):
experiments/north/metal/run.sh > work/north-synthetic.json
# Captured real tensors (after export_layer.py):
NORTH_TENSORS=work/north-layer NORTH_OUTPUT=work/north-metal/output.bin \
  work/north-metal/bench work/north-metal/branch.metallib > work/north-real.json
work/north-venv/bin/python experiments/north/compare_output.py
python3 experiments/north/summarize.py work/north-real.json
```

## Checks and limits

Every scheduling control must produce bitwise-identical hidden values, FP32
partials, attention projection and final output to the phased control. Checks
poison buffers before execution and verify exactly-once task visits for mixed
and persistent schedules. Separate Metal API/GPU validation runs are excluded
from timing. Each timed sample is checked after completion.

Five warmups and fifteen measured repetitions per configuration; configuration
order rotates and reverses. GPU command-buffer time includes counter resets,
partial reduction and every dispatch. CPU wall time includes encoding and
waiting. Compilation, allocation/loading and verification are excluded.
The synthetic runs overlapped model downloads; real runs must be kept separate.

The real experiment includes selected expert matvecs, activation, O-proj and
the residual join. It excludes RMSNorm, router/top-k, QKV, RoPE, KV reads,
softmax/attention, and other layers. It streams materialized BF16 weights;
the full reference model streams affine quantized experts. End-to-end decoder
improvement cannot be inferred from a branch timing.
