> Historical stage report. The [current full-model results](../../../NORTH_RESULTS.md) supersede its next-step statements.

# North Mini Code on the M4 Pro: first real results

**North Mini Code runs locally, and a persistent task schedule shows a repeatable
GPU-time improvement in our real-data branch prototype. The full-model decoder
has not been accelerated.**

The successful baseline uses the public MLX affine 4-bit checkpoint, pinned at
dfbe084dfa26e241345af99ca32848f38fd865f9, with unmodified text-model code from
mlx-vlm cdc745ad8a32d162f6d8e9d08be256910d663ac2 and MLX 0.32.2. Every LFS
download passed its SHA-256 and byte-count check. No Uzu engine source changed.

## Real model baseline

M4 Pro / 48 GB, AC power, batch one, greedy decoding, thinking disabled,
checkpoint chat template, fresh KV per request, three measured runs after
one warmup. Prefill uses 256-token chunks; first-token latency includes
prefill and first-token selection. Decode speed counts subsequent token steps.

| Prompt | Input tokens | Output tokens | Median decode | Median first token | Peak MLX allocation |
|---|---:|---:|---:|---:|---:|
| short | 136 | 512 | 55.45 tok/s | 0.285 s | 18.79 GB |
| long | 1672 | 512 | 49.02 tok/s | 2.825 s | 19.19 GB |
| rust | 140 | 403 | 54.13 tok/s | 0.290 s | 18.80 GB |

All measured token sequences were identical within each prompt. Short and long
timing runs hit their 512-token cap. Rust terminated naturally at 403 tokens.
A separate 1,024-token request completed the short Python answer at 519 tokens:
its assertions and 1,225 additional sorted-list input cases passed without
input mutation. The Rust answer compiled and all five generated tests passed.
The long answer still hit the larger cap; no completed long-code test is claimed.

After loading, MLX reported 18.495 GB of active allocations. Peak values above
are MLX allocation metrics, not whole-process RSS or total system memory.
There were no concurrent model downloads, model servers or other experiment
benchmarks during these timing runs. The harness is sequential and does not
implement a pipelined serving engine; these are its measured rates.

## What the Metal experiment actually executes

Captured a real single-token decode at layer 1, then repeated the investigation
at layer 7 using a different prompt and routed expert set. Each case includes
the normalized shared input, actual routing indices/scores, attention output
projection input, residual and reference layer output.

The graph covers eight selected expert up/gate projections, activation,
down-projection partials, attention output projection and the residual join.
It omits routing itself, normalization, QKV, attention/KV processing and other
layers. Expert weights are materialized from affine 4-bit into BF16 for this
first scheduling experiment. The native model continues to use quantized
expert matmuls.

Fourteen controls vary fusion, mixed placement, static/interleaved assignment,
dynamic claiming, and worker counts. All use the same arithmetic and FP32
split-K partial/reduction scheme. Local dependencies execute inside an owning
threadgroup; the global join remains a second dispatch. No unbounded
cross-threadgroup spin-wait or core-residency assumption is used.

The phased control takes four compute dispatches. Simple local fusion takes
three. Mixed/persistent schedules take two. Merely removing dispatches did not
produce the best result: static assignment with 128 workers did.

## GPU timing: the selected schedule

Five warmups per configuration, fifteen measured samples, rotated/reversed
configuration order. Values include GPU queue-counter reset, all dispatches
and the partial reduction. The change column is the median matched-pair
change, which can differ from dividing the two separate medians.

| Captured layer | Run | Phased GPU | Static 128 GPU | Paired change |
|---|---:|---:|---:|---:|
| 1 | 1 | 589.00 us | 530.92 us | -9.67% |
| 1 | 2 | 548.71 us | 512.46 us | -6.37% |
| 1 | 3 | 536.54 us | 500.08 us | -6.96% |
| 7 | 1 | 582.58 us | 544.67 us | -8.56% |
| 7 | 2 | 547.54 us | 505.25 us | -8.04% |
| 7 | 3 | 554.67 us | 509.92 us | -7.97% |

The 128-worker static candidate was selected from the layer-1 sweep, then checked
on layer 7. Both source and complete sweeps remain available; no slow cases were
discarded. These are exploratory repeated measurements, not a confidence
interval or evidence across all layers/context lengths.

A 20-worker pool was roughly 2.5 times slower than phased execution on these
real tensors. Treating the M4 Pro's 20 GPU cores as a prescription for 20
persistent threadgroups would therefore be a poor choice for this kernel.
This is a measurement of the tested geometry, not a general residency model.

Dynamic queueing and interleaving did not beat static 128 in this experiment.
The interleaving controls help distinguish queue overhead from placement.
Absolute first-run times drifted; matched ordering reduces but does not erase
clock/power variation. Every saved thermal sample in the real sweeps was nominal.

CPU wall improvements were smaller and noisier than GPU improvements. The
standalone host encodes and waits for each sample; it is not an integrated
decoder. No end-to-end token/sec gain is claimed.

## Numerical evidence

Every scheduling variant matched the phased control bit-for-bit for all hidden
values, FP32 partials, attention projection and final output. Each real-tensor
run includes 42 poisoned-buffer correctness submissions plus checks after all
210 timed samples. Exactly-once task visits are checked for mixed/persistent
schedules. Separate Metal API and GPU validation passed for layer 1.

Against an independent MLX calculation with the same materialized weights:
layer 1's output was bitwise exact; layer 7 had relative L2 error 0.0000403
(max absolute error 0.0004883). All outputs were finite.

Against the original quantized model layer output: relative L2 errors were
0.00317 and 0.00349, respectively (about 0.32–0.35%). Those comparisons include
weight materialization, arithmetic/activation and reduction differences.
Do not attribute the discrepancy solely to scheduling or promote this as
a numerically interchangeable full-model replacement.

## The practical limit and next experiment

The native MLX selected-branch control still wins wall time: medians were about
399 us at layer 1 and 364 us at layer 7, versus roughly 696–762 us for the
128-worker BF16 prototype. These runs use different representations and are
not paired compiler/engine comparisons; they show why the prototype should
not replace native inference yet.

The evidence justifies carrying the successful schedule into a competitive
affine-quantized kernel, preserving quantized weight streaming and the native
numerical contract. Then compare whole-block latency and only afterwards
test an opt-in full-model route. Weight prefetch and general cross-threadgroup
dependency execution remain separate untested mechanisms.

## Artifacts and rerun instructions

- experiments/north/README.md: exact scope, scripts and commands.
- experiments/north/results/baseline.jsonl and baseline-summary.json: real
  generated tokens, timing and memory; complete-code.jsonl: separate code checks.
- branch-real-*.json and branch-layer7-*.json: all scheduling measurements.
- layer-provenance.json and layer7-provenance.json: prompt, token, shapes and
  actual expert indices.
- real-layer-numerics.json and layer7-numerics.json: reference comparisons.
- model-provenance.json and python-environment.txt: hashes and package versions.
- Ignored work/ holds the model, pinned source checkout, venv, captures and
  extracted tensors. No commits, pushes, publication or maintainer contact.
