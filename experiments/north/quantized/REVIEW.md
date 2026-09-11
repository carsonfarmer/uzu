# Verification record — corrected North experiment

The earlier approval/completion framing was withdrawn. The original custom
route changed logits and tokens because its attention and down-projection
reductions did not match the reference. It is retained as historical evidence,
not as the final optimization.

The corrected result is supported by:

- Component isolation on identical native-layer inputs: 384 layer/token cases.
- Raw-byte equality of every full-vocabulary logit for 1,424 decode steps per
  corrected variant, plus prefill; five variants total 7,120 comparisons.
- Matched free-running token IDs in every measured final request: three prompts,
  three runs and four paths, 36 measured requests plus 12 warmups.
- A compiled-native control, rotated run order, fresh caches and recorded step
  times, TTFT, full wall time and memory. Headline gains are median paired ratios.
- Source hashes matching the final fast-path gate and benchmark. The fused
  kernel files match the preceding complete raw-byte gate; the intervening
  wrapper edits only add other opt-in modes. Snapshots are included.
- Dependency stress with one through 256 workers, changed activations and
  expert IDs, exact intermediate bytes and exactly-once task visits.
- Controlled per-expert versus all-expert readiness and before/after readiness
  weight-staging comparisons on two fixtures; 29 timed pairs per comparison.
- Metal API and GPU validation for dependency and staging variants, no errors.
- An additional Python prompt through both user-facing routes with raw-logit
  verification; both stop naturally with the same 19 generated tokens.
- GPU traces of full decoding, kept separate from uninstrumented throughput.

The persistent protocol uses Metal 3.2 coherent buffers and device-scoped
sequential fences. Producers complete a threadgroup device-memory barrier
before publishing. The optimized consumer acquires through thread 0, carries
visibility through a device-memory threadgroup barrier, then cooperatively
loads the immutable hidden vector. Consumers claim ready work and producers
can progress without all worker groups being resident.

The final gain is a full-model decode gain from replacing branch work in all
48 MoE layers. It is not a single kernel for the whole forward pass, an
end-to-end server benchmark, or a replication of Cohere's BF16 H100 results.
QKV/router prefetch across layer boundaries remains separate future work.
The staging prototype did not beat the simpler fused branch. Desktop clocks
were not locked; only three prompts and one Apple device were tested. Trace
startup and instrumentation overhead prevent treating trace wall time as
steady-state throughput, and per-shader/bandwidth counters were not available.

Use NORTH_RESULTS.md and share/RESULTS.md for the current numbers. The old
quantized/results/decode.jsonl is the failed prototype, and the six-variant
correctness/results/benchmark-all-v3.jsonl precedes the optimized scheduler.
Neither is pooled into the final four-variant estimates.
