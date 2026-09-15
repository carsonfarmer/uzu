# North Mini Code: targeted Metal fusion + asynchronous decode

## Scope correction — standard-engine benefit unproven

**This branch has not demonstrated an improvement to Uzu or the normal MLX-VLM
generation path.** It contains a standalone MLX runner and custom kernels.
The combined 19–24% measurements partly restore async behavior already present
in standard MLX-VLM. Smaller gains versus stock layers with async are also
internal to this runner; they need validation through an existing engine's
ordinary entry points before being reported as a practical engine improvement.

The code and raw measurements are preserved. Earlier shareable-result framing
is withdrawn, and further optimization against this substitute baseline has
been stopped. See the [baseline and scope correction](https://github.com/carsonfarmer/uzu/blob/cf/decode-fusion/BASELINE_CORRECTION.md).

A small, opt-in MLX experiment for the public North Mini Code 4-bit checkpoint
on Apple Silicon. This branch preserves the useful implementation from the
larger megakernel investigation. It starts directly from Uzu `7096cf32` and adds
this directory plus a local-cache ignore rule. It does not modify Uzu's Rust
inference engine.

## What the changes do

1. **Targeted fusion:** one Metal kernel combines each selected expert's up and
   gate projections with its activation, alongside attention output projection.
   A second kernel combines the expert down projections, routing weights and
   residual addition. Keeping these operations together removes intermediate
   work while retaining MLX's arithmetic order and BF16 rounding boundaries.
2. **Asynchronous decode:** submit the next dependent GPU calculation before
   reading the previous output token on the CPU. This allows CPU preparation and
   GPU execution to overlap. Every token still comes from the model's complete
   logits; no tokens are guessed or skipped.
3. **Optional preparation fusion:** combine normalization and Q/K/V/router
   projections. This is the `prepared_async` variant; `fused_async` is the
   smaller core result and does not use this preparation kernel.

The native attention implementation, KV cache, first dense layer, final head,
and multi-token prefill remain in use. The specialized paths operate on the
48 MoE layers for batch-one decode. The validated settings are 160 front
threadgroups, 16 down rows per threadgroup, and optional preparation at
64 QKV rows / 8 router rows with normalization.

## Result

See [RESULTS.md](RESULTS.md) for the fresh matched comparison, uncertainty,
correctness evidence and raw measurements. Both stock synchronous and stock
asynchronous controls are included, so the combined improvement and the
improvement over an already asynchronous runner can be distinguished.

## Reproduce

Validated on an M4 Pro with 20 GPU cores and 48 GB memory, macOS 27.0 build
26A428, Python 3.12.10 and MLX 0.32.2. The checkpoint is affine W4, group size
64, with BF16 activations and scales. Other models, precision formats and
devices require their own correctness and performance validation.

From the repository root, with Python 3.12 installed:

```sh
python3.12 -m venv work/north-venv
work/north-venv/bin/python -m pip install -r experiments/north_fused_async/requirements.txt
work/north-venv/bin/python experiments/north_fused_async/prepare.py --download
work/north-venv/bin/python experiments/north_fused_async/benchmark.py --output experiments/north_fused_async/results/local.jsonl
work/north-venv/bin/python experiments/north_fused_async/summarize.py experiments/north_fused_async/results/local.jsonl --output experiments/north_fused_async/results/local-summary.json
```

Preparation downloads approximately 18.5 GB of public weights plus the pinned
runtime source and verifies their hashes. Existing caches can be verified with
`prepare.py` without `--download`. Source and model revision/hash manifests are
in `provenance/`. No files from the other research branches are needed.
Leave `MLX_MAX_MB_PER_BUFFER` and `MLX_MAX_OPS_PER_BUFFER` unset to reproduce
the recorded default batching behavior. The benchmark serializes cooperating
research jobs through `/tmp/north-metal-research-gpu.lock`.

The default benchmark measures ten counterbalanced rounds of five variants
over Python, Rust and a longer Python prompt. Every variant occupies every
position equally, with each pair's order reversed equally often. Each run
generates 128 tokens: the first comes from untimed prefill, followed by 127
timed full-model decode steps. The timer includes the full vocabulary head,
argmax, token readbacks and final GPU drain. All measured sequences must match
the stock reference. Before timing, every variant's full logits are compared
byte for byte against that reference at all 127 steps.

For a quick correctness smoke check:

```sh
work/north-venv/bin/python experiments/north_fused_async/benchmark.py --runs 0 --tokens 8 --prompts short --output experiments/north_fused_async/results/local-smoke.jsonl
```

Output files are created exclusively; choose a new name for each run. Run
Python normally, without `-O`, so the validation assertions remain enabled.

## Scope and attribution

This is a fixed-length greedy decode runner. It rejects early EOS, excludes
prefill from throughput, and does not claim streaming frontend or serving
throughput. The longest measured prompt is 1,672 tokens; this does not validate
crossing the 4,096-token rotating-cache boundary. Tests of full logits establish
exactness for these runs, not a mathematical guarantee for every prompt.

[Cohere's article](https://cohere.com/blog/megakernels) motivated the
investigation and the focus on avoiding idle GPU time. Fusion and async
submission are established techniques. The async pattern is already present
in the pinned [MLX-VLM generation loop](https://github.com/Blaizzy/mlx-vlm/blob/cdc745ad8a32d162f6d8e9d08be256910d663ac2/mlx_vlm/generate/ar.py#L546).
Our arithmetic follows Apple's MLX kernels; see [NOTICE.md](NOTICE.md).

This result does not establish the benefit of Cohere's persistent task
scheduler or its future-layer weight prefetch. Those mechanisms remain the
subject of the separate `cf/north-megakernel-fullest` research branch. The
preserved research checkpoints are `cf/decode-fusion` at `6301f36e` and
`cf/north-additional-ten-percent` at `1ebab5ff`; they are not dependencies of
this clean branch.

## File map

| Files | Purpose |
| --- | --- |
| `layers.py`, `inputs.py`, `fusion.py`, `kernel.h` | Exact targeted branch fusion and reversible layer selection |
| `decode.py` | Common synchronous/asynchronous fixed-length generation loop |
| `prep.py`, `prep.h` | Optional preparation fusion |
| `reference.py`, `prepare.py`, `requirements.txt`, `provenance/` | Pinned text-model loading and reproducible setup |
| `benchmark.py`, `summarize.py`, `results/` | Full-logit validation, balanced measurement and raw evidence |
