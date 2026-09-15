**Stopped on user direction, September 14, 2026.** No further GPU experiments are authorized in this custom-runner investigation. The goal is incomplete and the mechanism matrix is not exhausted. These measurements do not establish an improvement to Uzu or stock MLX-VLM generation. See [engine integration inventory](ENGINE_INTEGRATION.md).

Historical investigation scope follows.

# Fullest megakernel investigation

Active research, with no new performance result yet. The preserved minimal
runner branch is `cf/north-additional-ten-percent` at `1ebab5ff`. This branch
investigates additional gains over its exact fused + preparation + async path.
All baseline and candidate generation uses matched async submission. Target:
at least 10% additional end-to-end throughput, or a concrete exhausted mechanism
matrix. A losing whole dispatch, host async adoption, or cache warming alone
cannot complete the goal.

GPU work must wait for `/tmp/north-minimal-baseline-20260914.done`. Every GPU
process runs through this directory's `gpu_run.py`, which acquires the shared
`/tmp/north-metal-research-gpu.lock` before importing MLX. Older markers do not
release this investigation. CPU source review and implementation can proceed.

See [coverage ledger](COVERAGE.md). All failures and outliers are retained.
Correctness gates require bitwise complete logits and identical greedy tokens;
intermediate checks and scheduler visit/progress checks localize failures.
Wall latency, GPU duration, and physical overlap are separate measurements.
