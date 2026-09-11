> **Reopened 2026-09-11: work remains incomplete.** The previous completion/approval framing was premature. The custom decoder fails correctness and omits key Cohere mechanisms; its slower timings are not evidence against the complete approach. Numerical debugging and the implementation work are active. Do not use the earlier tweet draft as a completed research conclusion.

# Apple megakernel experiments — status

**The requested full-model test and shareable result packet are complete.**

North Mini Code 4-bit runs locally on the M4 Pro. The isolated BF16 branch
schedule saves 6–10% GPU time, but the packed-W4 custom branch does not beat
MLX. An explicit all-48-MoE-layer decode patch measured 38–43 tok/s versus
49–54 tok/s for the original MLX path on three 256-token prompts.

The compiled-native control preserves logits/tokens. The custom path matches
190/192 teacher-forced greedy choices but changes full logits and free-running
continuations. It is not a numerically interchangeable optimization.

Current explanation: NORTH_RESULTS.md. Chart, tweet drafts and raw summary:
experiments/north/share/. Reproduction: experiments/north/quantized/README.md.
Metal API/GPU validation and pinned input integrity checks passed. No Uzu engine
source changed. Nothing was committed, pushed, posted or sent to Cohere.
All benchmark and validation jobs have finished.

## Historical checkpoints (superseded by NORTH_RESULTS.md)

The default-feature release CLI builds and serves the official Qwen3.5-4B-M
toolchain 0.15.0 package locally. Proper nightly gate tests pass: 12 passed,
0 failed, 1 upstream-ignored as flaky. Both experimental fusion variants pass
bitwise BF16 checks. Neither has earned production integration: sequential
pairing is inconsistent and concurrent pairing regresses. Engine code remains
unchanged; local changes are the isolated experiment, evidence, and documentation.

This is a negative experiment, not completion of the proposed optimized-engine
milestone. No production opt-in route, optimized-model logit comparison, or
baseline-versus-optimized end-to-end benchmark has been produced.

## Latest checkpoint: task/dependency prototype

The new experiments/task-graph harness executes a six-task diamond graph in
Metal. Controls are six phased dispatches, a fixed fused graph, static persistent
workers, and a dynamic queue of independent graphs. Each worker interprets
dependency descriptors within its owned graph; it does not wait on other
threadgroups. This is a working threadgroup-local dependency prototype, not
yet a cross-threadgroup ready-node scheduler or real transformer block.

Three repeated runs each passed 144 correctness cases and 270 timed sample
checks. Every intermediate is bitwise equal to a CPU reference; every task
executes exactly once. Reversed/shuffled task order, worker counts above job
counts, and repeated resets pass. A separate run passed Metal API and GPU
validation. Raw evidence and commands are in experiments/task-graph/.

Synthetic GPU medians (us), including reset/poison and diagnostic counters:

| Run | Phased | Fixed fusion | Static, 256 workers | Dynamic, 256 workers |
|---|---:|---:|---:|---:|
| 1 | 140.12 | 70.29 | 93.33 | 97.83 |
| 2 | 143.17 | 73.71 | 95.63 | 98.67 |
| 3 | 113.54 | 57.88 | 69.46 | 72.92 |

Fixed fusion is fastest in this workload. Small persistent worker pools are
much slower; larger pools narrow the gap. This does not establish model
performance. The benchmark is synthetic with graph-local dependencies and
does not implement Cohere's complete scheduling mechanism.

The requested subagent assessment is in NORTH_MINI_CODE.md. Public 4-bit MLX
weights total 17.23 GiB and plausibly fit this 48 GB Mac. The existing Uzu
checkout needs architecture/routing/quantized-expert work; a pinned mlx-vlm
baseline avoids coupling scheduler research to that port. No North weights
were downloaded and no North runtime result is claimed.

Next: a North-shaped parallel-branch workload, followed by real dense/MoE
layer comparisons against an independent quantized North reference.

## Scope correction after user clarification

The objective is to test Cohere-style megakernel execution on Apple devices.
The prior experiment covered only up/gate epilogue fusion. It has not tested
persistent task execution, scheduling across operations, dependency-local
synchronization, or weight prefetch. The broader objective remains open;
see MEGAKERNEL_PLAN.md for the source-mapped experiments and milestones.
A better per-kernel profile will help evaluation but should not block building
the scheduler feasibility tests.

## Reproducibility

- Project: `/Users/carsonfarmer/Developer/Personal/uzu-metal-lab`
- Local branch: `cf/decode-fusion`; no push or maintainer contact.
- Upstream: `https://github.com/trymirai/uzu`, commit
  `7096cf32e70400c377b7341f75949d52c36f8f4d` (full history cloned into verified empty directory).
- No applicable AGENTS.md found in the project or ancestor directories. Rust skill read.
- Machine: MacBook Pro Mac16,8, Apple M4 Pro, 20 GPU cores, 48 GB unified memory.
- OS: macOS 27.0 build 26A5425a. Initial available disk: 112 GiB;
  initial system memory free percentage: 65%.
- Initial power: battery; 75% observed after initial measurements. Thermal state was
  nominal in recorded measurements. Background model/Cargo downloads were active;
  do not treat these microbenchmarks as controlled full-model performance results.
- Rust default: 1.97.1. Builds explicitly use installed Rust 1.98.0
  (`88d9e12ae`, LLVM 22.1.8), meeting upstream's rust-version. Upstream's
  rust-toolchain.toml selects nightly; no global toolchain setting changed.
- Xcode: `/Applications/Xcode-beta.app/Contents/Developer`.
- Metal compiler: 32023.917; `-O2 -std=metal4.0 -mmacosx-version-min=26.0`.
- Swift: 6.4 (`swiftlang-6.4.0.23.5`), host harness built with `-O`.
- Model: `trymirai/Qwen3.5-4B-M`, revision
  `4a434ddcac0c7e129da78bdde9b669937ab51ef2`.
- Entire model downloaded locally to `work/models/Qwen3.5-4B-M`.
  Weights: 2,441,821,800 bytes; SHA-256
  `86b726111af2b17b921221f899e21e837b7c680ec12bf86f841de0806a5e4c08`.
  Model/tokenizer SHA-256 hashes match Hugging Face LFS metadata. Full file hashes
  are in `experiments/decode-fusion/results/provenance.json`.
- Checkpoint layer 0 MLP: combined up projection [18432,2560], down [2560,9216];
  asymmetric W4/group32, BF16 scales, packed zero points, input/output RHT signs.

## Implemented and checked

See `experiments/decode-fusion/README.md` for exact runnable commands.
`paired.metal` includes the existing Uzu GEMV accumulation, reduction, epilogue,
and activation helpers. The baseline instantiates the original GEMV geometry
(32 output rows, eight SIMD groups), then runs a separate gate/RHT kernel.
The fused candidate pairs matching value/gate tiles within one threadgroup,
preserves BF16 rounding boundaries, and writes the prepared down input directly.
Down projection remains outside the experiment. A second option evaluates the
paired halves concurrently with four SIMD groups per half.

Both kernels are explicitly selected only by the standalone harness. Fixed host
shape/type guards and tensor metadata checks exclude unsupported configurations.
There is no production dispatch or backend API extension yet.

- All 9,216 BF16 outputs match bit-for-bit for input magnitudes 0, 0.01, 1, and 8,
  for deterministic synthetic weights and for actual layer 0 checkpoint weights.
- Input activations are synthetic, not captured model activations.
- Repeated dispatch output equality is checked in the final harness.
- Both scheduling variants ran with Metal API and shader validation enabled;
  no validation errors reported. These validation runs are excluded from timing claims.
- Standalone capture saved at `work/paired.gputrace`; upstream's capture parser
  confirms baseline_projection, baseline_gate, and paired_projection_gate.
  This is a prototype capture. A separate full-model decode capture is now in work/model-decode.gputrace.
- Shell syntax and Python compilation checks pass.
- Stable Rust formatting check was attempted; it cannot enforce the two nightly
  import-format options. No production Rust source has been edited.

## Measurements and interpretation

Compilation excluded; ten warmup pairs, fifteen alternating-order baseline/fused
pairs per run, three runs, one or 32 operations per command buffer. Raw GPU
command-buffer and CPU encode/submit/wait measurements are saved as JSON
under `experiments/decode-fusion/results/`. Thermal state is recorded per sample.

Single-submission GPU medians, microseconds (baseline / fused):

| Variant | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Sequential paired tiles | 128.33 / 130.92 | 118.12 / 120.50 | 120.25 / 120.00 |
| Concurrent paired halves | 135.88 / 161.29 | 123.00 / 136.00 | 121.13 / 134.92 |

Sequential pairing has no consistent improvement. Concurrent pairing regresses;
matched-pair median single-submission GPU changes are +17.05%, +17.13%, +13.40%.
Repeated-submission sequential results reverse direction across runs. Do not
cherry-pick those results. Repeated operations reuse the same weight buffers;
**they are not different model context lengths** and omit full-model memory traffic.

The eliminated combined-up intermediate is only 36,864 bytes (73,728 bytes for
one write plus one read), compared with about 27 MB of weights/scales/zero points
for this projection. This arithmetic suggests limited memory-traffic savings;
tile scheduling and occupancy can outweigh the dispatch saving. This is an
interpretation, not a measured bottleneck diagnosis.


## Compatible model and real decoding baseline

The original pinned Hugging Face config predates this engine's attention schema.
Use the official catalog package (not a local config/weight conversion):
trymirai/Qwen3.5-4B-M revision ad86386e34fb3b82371b875f145eea01f2c92131,
catalog ID alibaba:qwen3.5:4b:mirai:mirai-m:4, toolchain 0.15.0.
It is cached at work/models/Qwen3.5-4B-M-toolchain015. All three published
CRC32C checks pass; SHA-256 hashes and the catalog record are saved in
experiments/decode-fusion/results/current-model-provenance.json.
All five layer 0 MLP tensor files are byte-identical to the original checkpoint.

The release CLI was built with Rust 1.98.0 and task-local CMake 4.4.3.
The focused tests were subsequently run without bootstrap on
Rust 1.100.0-nightly (a36d05efa, 2026-09-09):

~~~sh
CMAKE="$PWD/work/build-tools/cmake/data/bin/cmake" cargo +nightly test   --offline --locked -p uzu-engine --no-default-features   --features cpu,metal --lib gated_act_mul -- --nocapture
~~~

The final test result is retained in results/nightly-tests.log. Initial network
and stable-custom-test-harness failures were resolved; no pending build remains.
The temporary task-local network tunnel is stopped.

Serving benchmark: AC power (62% charging observed), no concurrent build/model
download, prefix cache disabled, greedy decoding, thinking disabled, one warmup
then five measured repetitions per prompt, 512-token cap. HTTP wall time includes
prefill and serving overhead; the server's decode rate is a separate CPU-observed
metric, not GPU-only time.

| Prompt | Input tokens | Output tokens | Median HTTP wall | Median server decode |
|---|---:|---:|---:|---:|
| Python merge | 26 | 345 | 4.426 s | 80.2 tok/s |
| Same request after 128 helper functions | 1,854 | 354 | 7.300 s | 74.3 tok/s |
| Rust first duplicate | 39 | 512 (capped) | 6.513 s | 80.6 tok/s |

All five measured texts are identical within each prompt. Both Python outputs
terminated naturally and each passed 1,225 combinations of sorted inputs,
including empties and duplicates, while preserving inputs. A separate Rust
request with a 1,024-token cap terminated naturally at 762 tokens; its complete
code compiled and all six generated tests passed. This is a coding smoke check,
not broad model-quality evaluation. Raw response text, timing logs, summaries,
and checks are under experiments/decode-fusion/results/.

A first-decode Metal capture is saved at work/model-decode.gputrace. Resource and
command-stream strings show BF16 ScaleZeroPointDequant W4/group32 GEMV
with tile parameters 1,32,32,2,8 and BF16 GatedActMul. This agrees with
DenseMlp::encode: combined up projection, gate plus input preparation, separate
down projection. The upstream resource parser misses embedded-library mangled
symbols; raw matching strings are retained in model-capture-kernel-strings.json.
String presence verifies the specialization exists in the capture, but does not
measure duration or dispatch counts. INT8 activation preparation is outside
this prototype.

The standalone compiler target is macOS 26.0, whereas upstream targets 26.4;
both execute on this macOS 27.0 machine. Do not claim identical compiler settings.

## Follow-up measurements on AC power

After stopping the server and finishing trace export, both unchanged kernels
were rerun three times, alternating sequential and concurrent experiments.
No build, model download, or profiler was running. Every check remained bitwise
exact and every thermal-state sample was nominal. Raw results and GPU/host
medians are in results/microbenchmark-ac-*.json and microbenchmark-ac-summary.txt.

| Variant | Run 1 GPU us, baseline/fused | Run 2 | Run 3 |
|---|---:|---:|---:|
| Sequential | 293.92 / 316.38 | 125.71 / 132.00 | 115.38 / 114.79 |
| Concurrent | 123.75 / 140.88 | 122.63 / 143.67 | 115.00 / 127.25 |

Matched single-dispatch changes: sequential +5.11%, +2.75%, -0.59%;
concurrent +13.67%, +15.90%, +11.53%. The first sequential run has much higher
absolute latency despite ten warmups; clock/power settling is a possible cause,
not a measured diagnosis. Retain it rather than silently removing it.
At 32 reused operations per submission, sequential paired changes are -2.41%,
-4.53%, +0.24%; these still do not establish a model decode gain.

## System GPU trace

work/model-system.trace records a separate 15.702-second attached trace.
The upstream parser attributes 6.060 seconds of GPU-active interval union,
502 command buffers and 8,627 encoders to the server. It spans the tail of the
separate Rust request and the short profiling request; it is not a per-token
or per-request GPU timing measurement. The trace exports no shader dispatch
intervals, so per-kernel duration and the gate's share are unavailable.
The decode capture does confirm the expected kernel specialization, but
bottleneck attribution remains incomplete. The aggregate result and limitations
are saved in results/model-trace-summary.json; the larger trace/export stay
under ignored work/. The direct xctrace command and recorder compatibility
workaround are documented in the experiment README.

## Decision and remaining work

Keep this as an isolated, explicitly selected experiment. No engine dispatch,
backend API, or serving behavior changed. No push, publication, or maintainer
contact; no commit has been made.

Any later production integration needs a new measured win, an opt-in guard and
unsupported-shape fallback, actual model activation/logit comparisons, repeated
decoding, and matched baseline/optimized runs at multiple context lengths.
Do not infer an end-to-end gain from eliminated dispatches or the most favorable
microbenchmark run. The threadgroup-local prototype is now implemented. Next steps are the
North-shaped workload and real-layer reference comparisons described above.
