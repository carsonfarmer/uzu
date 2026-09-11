# Correct North Mini Code decoding on Metal

**Latest result:** [the whole-pass experiment](../whole_pass/README.md) extends
this exact branch work to one safe Metal dispatch containing all 49 transformer
layers, K/V updates, final normalization, and full logits. It reaches 53.423
decode tok/s on the tested M4 Pro versus 53.951 for stock MLX, with byte-exact
logits on two 127-step coding continuations. The earlier 57.472 tok/s static
result was rejected after that scheduler deadlocked on a repeated run.

The original packed-W4 prototype contained accumulation-order bugs. Its old
results are retained as history; use the corrected entry points below.

The current experiment replaces the MoE and attention-output branch in all 48
MoE layers during single-token decode. It uses the original affine W4/group64
expert weights and BF16 attention weights. Prefill, layer 0, QKV, attention,
routing, normalization, and the output head use the pinned MLX reference.

## Reproduce

Tested on an M4 Pro with 20 GPU cores and 48 GB, macOS 27.0 / 26A5425a,
Python 3.12.10 and MLX 0.32.2. Run from this repository's root. Model files
occupy about 18.5 GB; allow additional runtime and working space.

```sh
uv venv --python 3.12.10 work/north-venv
uv pip install --python work/north-venv/bin/python -r experiments/north/quantized/requirements.txt
work/north-venv/bin/python experiments/north/prepare.py --download

# Try your own prompt and compare every logit byte with original MLX.
work/north-venv/bin/python experiments/north/generate.py --mode exact --verify \
  --prompt 'Return only Python code for merging two sorted lists.'

# Every logit byte must match, with independent caches and identical inputs.
work/north-venv/bin/python experiments/north/correctness/verify_decode.py \
  --modes exact scheduled prefetch staged --tokens 512 \
  --output work/reproduced-correctness.jsonl

# Matched-token, full-model generation. One warmup; rotated variant order.
work/north-venv/bin/python experiments/north/correctness/benchmark_decode.py \
  --modes native_compiled exact scheduled prefetch staged --tokens 256 --runs 3 \
  --output work/reproduced-throughput.jsonl
```

The preparation script uses public HTTPS, checks pinned runtime source files,
and verifies every checkpoint file's SHA-256 and size. The checkpoint is
`mlx-community/North-Mini-Code-1.0-4bit` at
`dfbe084dfa26e241345af99ca32848f38fd865f9`; the text-model source is MLX-VLM at
`cdc745ad8a32d162f6d8e9d08be256910d663ac2`. It imports that model's unmodified
text implementation directly, bypassing multimedia frontend imports. This is a
local text harness, not a benchmark of MLX-VLM's serving frontend.

## Variants and controls

| Variant | What executes inside the replacement branch |
|---|---|
| `original` | Original pinned MLX model, automatically included by the harness. |
| `native_compiled` | Same wrapper and compiled boundary as the candidates, using native MLX operations. |
| `exact` | Mixed up/gate/activation and attention-projection tasks, followed by complete-K down dots and residual join. Two compute dispatches. |
| `scheduled` | Persistent ready-work queue. Down tasks depend on their own expert's hidden tiles. Workspace initialization and final join remain outside that dispatch. |
| `staged` | Claims down tasks, helps execute producers while inputs are unready, and loads down weights into threadgroup memory after readiness. |
| `prefetch` | Same task allocator and staged GEMV as `staged`, but loads immutable down weights before waiting/helping producers. |
| `fast` | Ready-work scheduler with one acquisition per consumer threadgroup, cached readiness and shared activation loads. Diagnostic visit counters are disabled in full-model timing. |

The geometry is specialized: batch one, hidden 2048, intermediate 768, eight
routed experts, 128 available experts, attention output 4096→2048. The `exact`
variant uses 160 front threadgroups and 16 down rows per threadgroup; persistent variants
use 64 workers. The patch is explicit and restored when the harness exits.
Nothing modifies Uzu's engine or the installed reference model source.

These are branch integrations into a complete decoder, not an entire forward
pass in one kernel. Next-layer QKV/router prefetch and the full Cohere opcode
stream remain outside scope. See [the source mapping](../correctness/COHERE_REFERENCE.md).

## Correctness and timing

The corrected kernels preserve the pinned MLX per-lane accumulation order,
reduction tree, BF16 rounding boundaries, affine input sums, and routing-product
rounding. The earlier split-K down reduction and attention summation order did
not satisfy that contract. `correctness/isolate.py` compares components on
identical native-layer inputs so errors cannot hide behind accumulated drift.
See [NOTICE.md](NOTICE.md) for MLX attribution.

The full-model correctness driver compares raw logit bytes, including prefill,
with independent fresh caches. Equal logits imply equal greedy decisions; the
measured generation runs also check all token IDs against the saved reference.
The 512-token gate includes both length-capped continuations and Rust's natural
stop at 403 tokens. It is equivalence testing on these prompts, not a broad
quality benchmark.

Throughput includes host graph construction, execution, synchronization and
argmax. The first token is counted in time to first token, and subsequent
one-token forward passes form decode throughput. Model load, tokenization and
JIT warmup are excluded from reported decode speed. Prefill is measured
separately. This is a desktop measurement with unlocked clocks; inspect raw
paired runs as well as summaries. Instrumented GPU traces are separate from
uninstrumented throughput.

## Dependency and prefetch ablations

First capture actual layer inputs:

```sh
work/north-venv/bin/python experiments/north/export_layer.py
work/north-venv/bin/python experiments/north/export_layer.py --layer 7 \
  --output work/north-layer7 --prompt 'Write a Rust function that finds the first duplicate in a slice.'

# Compare per-expert readiness with waiting for every expert.
work/north-venv/bin/python experiments/north/persistent/check.py \
  --mode ready --workers 64 --repetitions 31 --output work/ready-ablation.json

# Move the same staged weight loads before or after the dependency.
work/north-venv/bin/python experiments/north/persistent/check.py \
  --mode staged --workers 64 --repetitions 31 --output work/prefetch-ablation.json
```

The checks vary activations and expert IDs on repeated submissions, check every
intermediate byte, require exactly-once task visits, and report publication or
progress failures. The wider stress sweep includes one worker and many more
workers than GPU cores. Timing includes zero-initializing the workspace and
joining outputs. These small branch runs do not replace full-model timing.

Current evidence is under `../correctness/results/` and `../persistent/`.
The old `results/decode.jsonl`, its `decode-source/` snapshot, and the legacy
`custom` route document the failed prototype; do not pool them with corrected
runs or use the old `report.py` to summarize this checkpoint.
