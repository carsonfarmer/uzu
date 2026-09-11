# Paired up/gate projection experiment

This is an isolated, opt-in Metal microbenchmark, not a production Uzu route.
The unmodified Uzu engine and OpenAI-compatible server remain the fallback.
There is **no established decoding speedup**. See STATUS.md for milestone progress.

The candidate fuses the combined MLP up/gate projection with output Hadamard,
SiLU multiplication, and the down projection's input Hadamard. It leaves down
projection separate. Each threadgroup computes matching 32-row blocks from the
value and gate halves, then combines them after a threadgroup barrier. An alternative (`PARALLEL_PAIRS=1`) gives each half four SIMD groups in the same
threadgroup, so they run concurrently with eight rows per lane. There
are no cross-threadgroup waits. GEMV accumulation/reduction and numerical helper
functions come directly from the checked-out upstream headers.

Supported experiment configuration: batch 1; BF16 input/output; unsigned W4,
asymmetric zero points, group size 32; K=2560 and hidden=9216; output/input
Hadamard signs; SiLU alpha=1; no biases or clipping. The host fixes and checks
these dimensions. Other formats, INT8 activation preparation, clipping, and
other activations have no experimental route. No production selection is changed.
The intermediate BF16 conversions before/after output RHT, activation, and
multiplication are retained. The input buffer represents already-prepared
projection activations; its initial RHT is outside both measured variants.

Run from the repository root on Apple Silicon with Xcode's Metal toolchain:

```sh
experiments/decode-fusion/run.sh > work/synthetic.json
python3 experiments/decode-fusion/extract.py \
  work/models/Qwen3.5-4B-M/model.safetensors work/layer0
MODEL_TENSORS=work/layer0 experiments/decode-fusion/run.sh > work/real-weights.json
python3 experiments/decode-fusion/summarize.py work/real-weights.json
```

Compilation is outside the timed section. The harness performs four numerical
checks, ten warmup pairs, and fifteen alternating-order matched pairs for each
of one and 32 operations per command buffer. Output includes command-buffer GPU
time, host encode/submit/wait time, and macOS thermal state (0=nominal). Input
activations are deterministic synthetic values even when using checkpoint weights.
Repeated operations reuse the same weights and are not model context lengths.

Capture separately; the target path must not already exist:

```sh
MTL_CAPTURE_ENABLED=1 MODEL_TENSORS=work/layer0 \
  experiments/decode-fusion/run.sh work/paired-new.gputrace > work/capture-run.json
```

The capture is taken after the timed loops; do not compare timings from a run
with capture instrumentation enabled to ordinary timing runs. Open the gputrace
in Xcode to inspect the two baseline dispatches and one fused dispatch.

The raw first three real-weight repetitions of each scheduling variant are in results/.
The parallel variant was slower in all three runs; the sequential variant was inconsistent. They were collected
while dependency/model downloads were active, with nominal reported thermal state, on battery power (75% observed after the runs).
Their variability is a reason to reject a speed claim, not select the best run.
There is no full-model traffic, attention/KV cache, down projection, CPU sampling,
or network serving in these measurements. The baseline wrapper instantiates the
upstream GEMV headers directly; it is not a trace-verified Uzu model baseline.

For the measured full-model baseline (requires built CLI and the catalog toolchain 0.15.0 package; see results/current-model-provenance.json):

```sh
LOCAL_PATH="$PWD/work/models" target/release/cli server --model "$PWD/work/models/Qwen3.5-4B-M-toolchain015" \
  --host 127.0.0.1 --port 18000 --no-prefix-cache
# In another terminal:
python3 experiments/decode-fusion/http-bench.py --runs 5 --tokens 512 > work/http-baseline.jsonl
```

The HTTP driver uses greedy decoding, a warmup per prompt, five measured runs,
and a short/long code-context pair plus a Rust prompt. Preserve actual server
input/output token counts; HTTP wall time includes prefill. Use a separate server
run with `UZU_CAPTURE_FIRST_DECODE=1` to capture its first decode, and exclude it
from performance timing. Upstream writes this capture under `/tmp/uzu-capture-metal-decode.gputrace`.

The original Hugging Face revision's config predates the current engine schema.
The serving baseline uses the official converted package at catalog revision
ad86386e34fb3b82371b875f145eea01f2c92131, with published CRC32C checks verified.
All five extracted layer 0 MLP tensor files are byte-identical across packages.
The decode capture contains the expected BF16 W4/group32 GEMV specialization
and BF16 GatedActMul. The upstream capture-info parser misses mangled symbols
from embedded libraries; raw resource/capture string evidence is retained in
results/model-capture-kernel-strings.json. String presence alone does not
measure dispatch count or GPU duration.

The standalone deployment target is macOS 26.0; upstream targets 26.4. Both
ran on the same macOS 27.0 host. Do not claim every compiler setting is identical.

For an attached Metal System Trace, invoke xctrace directly. The upstream
recorder passes --target-stdout with --attach, which this Xcode rejects:

~~~sh
xcrun xctrace record --template 'Metal System Trace'   --output work/model-system-new.trace --no-prompt --time-limit 15s --attach SERVER_PID
# Send a separate request while recording, then analyze:
PYTHONPATH="$PWD/work/python-tools" python3 -m tools.gpu_trace analyze   work/model-system-new.trace --json work/model-trace-new.json
~~~

Use --prompt short, --prompt long, or --prompt rust to select one HTTP prompt.
--runs 0 makes one warmup-labelled request, useful for a separate capture or
coding smoke check; it produces no measured benchmark samples.
