# Actual-engine megakernel research: progress, not a completed speedup claim

The preserved actual MLX-VLM improvement remains **10.25–11.62% over the untouched
published package**, documented in [MLX_VLM_INTEGRATION.md](MLX_VLM_INTEGRATION.md).
The active goal is another 10% beyond that optimized `prepared` control.

Research now lives in a separate actual MLX-VLM worktree:
`/Users/carsonfarmer/Developer/Personal/mlx-vlm-megakernel`, branch
`cf/north-megakernel-engine`. Normal MLX-VLM `load/generate`, async submission,
sampling, stopping and cache ownership remain in use. This work does not use
Uzu's inference engine or the withdrawn custom decode runner.

The first two integrated scheduling candidates are correct but slower:

| Actual-engine experiment | Short | Rust | Long |
|---|---:|---:|---:|
| Boundary scheduler, 32 workers, relative to matched prepared control | -7.24% | -7.00% | -5.70% |
| Attention/expert overlap scheduler, 48 workers, relative to matched prepared control | -7.25% | -7.24% | -24.74% |

Each pilot retained four measured runs per mode and prompt plus two warmups.
Both compared two candidate worker counts with the prepared control, so each
pilot contains 36 measured and 18 warmup generations. Every output matched.
The overlap candidate also passed all five full-model correctness cases,
including cache rotation and natural EOS: 566 byte-exact full-vocabulary logit
arrays per mode, 1,698 comparisons including the control.

The boundary scheduler uses completion counters to release each expert's down
projection independently, then computes the next layer's normalization and
QKV/router projections. The overlap scheduler additionally runs attention tasks
alongside expert work. Native cache updates occur exactly once per layer, and
fallback uses the already prepared native views. Neither is a whole-model
megakernel, and neither yet loads next-layer weights before their activation
dependency is ready.

An isolated access-pattern experiment identified a concrete problem in the
first attention port: arbitrary inner strides caused much of its long-context
primitive slowdown. At length 1,672, a guarded unit-stride version reduced
diagnostic wall time from about 259 to 193 microseconds, against 184 for native
MLX. All 25 numerical comparisons passed. Removing device-coherent qualifiers
had little effect in this test. The integrated stride fix is the next measured
experiment; it has not yet established an engine speedup.

Additional preserved controls include producer-first task ordering, a simpler
queue-free attention/expert backfill attempt, native-equivalent attention tasks
through 65,537 KV tokens, and a weight-byte/host-cost analysis. These are tests
and measurements, not reasons to claim the 10% goal complete or options exhausted.

Reference: [Cohere's article](https://cohere.com/blog/megakernels) and source
`cohere-ai/cohere-megakernel@67d0b9ca22ea3652796b715d1d1863459e0e2c3c`.
The experiments are explicitly testing its smaller task dependencies, attention
backfilling and future-weight loading ideas. Retained prefetch remains open.

The research branch and all committed raw results through
`878c0101d87a52255c8de400b77ff65e35fc5807` are backed up in
`experiments/north_mlx_vlm/mlx-vlm-megakernel.bundle`. The bundle was verified and
requires the public MLX-VLM base `1ecf1ecdd28af102eded679be0daa5c76ab2a068`.
From an MLX-VLM clone containing that base, restore it with:

```bash
git fetch /path/to/uzu-metal-lab/experiments/north_mlx_vlm/mlx-vlm-megakernel.bundle refs/heads/cf/north-megakernel-engine:refs/heads/cf/north-megakernel-engine
```
