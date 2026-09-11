# North's packed 4-bit branch on Metal

This is a runnable research experiment, including an explicit full-model decode
patch. The custom route is slower than native MLX and changes some generated
tokens. It is not a production optimization or a complete Cohere megakernel.

## Reproduce on the tested Mac

Tested on M4 Pro (20 GPU cores), 48 GB, macOS 27.0 / 26A5425a, Python 3.12.10,
MLX 0.32.2. Other Apple chips and OS/compiler releases are untested. The model
uses about 18.5 GB of MLX allocations; allow space for weights and runtime.
Run from this repository's root. `uv` and `curl` must be installed.

```sh
uv venv --python 3.12.10 work/north-venv
uv pip install --python work/north-venv/bin/python -r experiments/north/quantized/requirements.txt
work/north-venv/bin/python experiments/north/prepare.py --download
```

The preparation script downloads the pinned public runtime source and checkpoint,
then checks source-file hashes and every model file's SHA-256 and size. It uses
public HTTPS without credential helpers. Without `--download`, it only verifies
existing inputs. The checkpoint is the community affine W4/group64 conversion,
not Cohere's BF16 or NVFP4 package.

```sh
# Full-model original, compiled-native control, and experimental custom path.
# One warmup + three measured requests per variant/prompt; 256-token cap.
work/north-venv/bin/python experiments/north/quantized/decode.py

# Regenerate real branch captures, then the frozen candidate's controls.
work/north-venv/bin/python experiments/north/export_layer.py
work/north-venv/bin/python experiments/north/export_layer.py --layer 7 --output work/north-layer7 --prompt 'Write a Rust function that finds the first duplicate in a slice.'
work/north-venv/bin/python experiments/north/quantized/bench.py --selected --output work/q-layer1.json
work/north-venv/bin/python experiments/north/quantized/bench.py --selected --layer 7 --capture work/north-layer7 --output work/q-layer7.json

# The original 512-token baseline is separately reproducible.
work/north-venv/bin/python experiments/north/baseline.py --runs 3 --tokens 512
```

The prompts and routing are recorded in the parent `results/layer*-provenance.json`
files. Captures are ignored work files and regenerated; large weights are never
included in the result packet.

## What is changed

`kernel.h` reads original uint32-packed 4-bit expert weights and BF16 affine
scales/biases. Eight routed experts each contain 768 intermediate features.
A job computes up/gate, the BF16 activation, and its chunk's contribution to
the down projection. Independent jobs compute the attention output projection.
An ordinary second dispatch reduces partials and joins the residual.

`kernels.py` compares this with four phased dispatches using identical math.
The sweep varies chunks of 32/64/128, worker counts and interleaving. The frozen
full-model candidate is chunk 64, 128 workers, interleaving: 96 expert-chain jobs
and 64 attention jobs. This is a static assignment of ready jobs; it does not
implement cross-threadgroup waits, general dependency counters, weight prefetch,
continuous batching, or a whole-forward-pass megakernel.

`patch.py` replaces only single-token branches in layers 1–48. Layer 0 and
multi-token prefill use the original implementation. The original attention
module is shallow-copied with its output projection removed; QKV, rotary/KV,
and attention math are unchanged. Router scores/IDs are computed normally.
The native compiled control follows the same wrapper but uses MLX operations.
The harness restores the original layer list even if a check fails. No Uzu
engine files or installed model-source files are modified.

## Timing and numerical controls

Branch timing uses eager and compiled MLX functions in one process, with explicit
array arguments and a changed-input check against stale replay. Eight warmups,
31 samples, rotated/reversed order, model load/JIT/checks excluded. Wall time
includes graph construction/evaluation and host waiting; it is not GPU-only.
All intermediate and final outputs must match the phased control bitwise for
the same chunk size; every timed output is checked. Selected routing is fixed
within one capture. The full-model test uses actual changing routing.

Full-model generation follows the baseline's chat template, fresh caches,
256-token prefill chunks, greedy decoding, and reasoning disabled. Decode
throughput counts subsequent one-token steps; the first token is in TTFT.
One warmup and three runs per prompt rotate the variant order. Outputs have a
256-token cap, distinct from the earlier 512-token baseline. These free-running
paths can generate different continuations, so this is a practical throughput
comparison, not a matched-token speedup claim. All token IDs and step times
are retained. It is a desktop measurement, without locked GPU clocks or an
exclusive-machine guarantee; inspect individual runs as well as medians.

Teacher-forced checks feed all variants the same saved original continuation
for 64 steps per prompt and compare full-vocabulary logits and greedy choices.
This catches accumulated errors that a single-layer check cannot. The compiled
native control is exact on these checks. The custom path is not exact and must
not be advertised as interchangeable with the reference.

`diagnose.py` isolates projections and activation rounding. Matching MLX requires
preserving its BF16 input-sum and sigmoid arithmetic, not merely the same
real-number formula. This reduced the layer-1 branch error to zero and the
layer-7 branch error to about 9e-6 relative L2, but did not establish full-model
equivalence. Remaining reduction/projection differences can propagate through
decoding. See NOTICE.md for Apple MLX attribution.

## Evidence versions

- `results/pilot.json` and `pilot-source/`: initial packed-weight prototype.
- `sweep.json`: exploratory tile sweep before arithmetic alignment.
- `activation-before-alignment.json`, `activation-diagnostic.json`: diagnosis.
- `aligned-sweep.json`, `aligned-layer7.json`: corrected exploratory comparisons.
- `final-layer{1,7}-{1,2,3}.json`: frozen candidate and controls, repeated runs.
- `decode.jsonl`: all teacher checks and full-model generation runs.
- `decode-source/`: exact source snapshot hashed in the decode run. Current
  kernels.py adds shape/dtype assertions only; kernel arithmetic is unchanged.
- Parent `results/`: prior BF16 branch studies, baseline and pinned input hashes.

Historical exploratory files are retained, not pooled into final estimates.

To regenerate the chart and numerical summary from the included raw runs:

```sh
work/north-venv/bin/python experiments/north/quantized/report.py
```

For new repeated branch runs, use `--output` with a distinct file for each
layer/repetition. The report's six input filenames are listed above. Preserve
the bundled raw runs if comparing a new machine or kernel revision.
