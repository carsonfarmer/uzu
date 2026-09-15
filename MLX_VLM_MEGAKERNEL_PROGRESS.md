# Actual-engine megakernel research: progress, not a completed speedup claim

The preserved actual MLX-VLM improvement remains **10.25–11.62% over the untouched
published package**, documented in [MLX_VLM_INTEGRATION.md](MLX_VLM_INTEGRATION.md).
The active goal is another 10% beyond that optimized `prepared` control.

Research now lives in a separate actual MLX-VLM worktree:
`/Users/carsonfarmer/Developer/Personal/mlx-vlm-megakernel`, branch
`cf/north-megakernel-engine`. Normal MLX-VLM `load/generate`, async submission,
sampling, stopping and cache ownership remain in use. This work does not use
Uzu's inference engine or the withdrawn custom decode runner.

The integrated scheduling candidates matched stock in their full-generation
test cases, but are slower. Later stress tests exposed synchronization defects
in the experimental schedulers; the hardened versions subsequently passed all
15 generation cases and 1,698 full-logit comparisons against stock. The
preserved prepared control is unaffected. The scheduler timing rows below are
the historical pilots before that hardening.

| Actual-engine experiment | Short | Rust | Long |
|---|---:|---:|---:|
| Boundary scheduler, 32 workers, relative to matched prepared control | -7.24% | -7.00% | -5.70% |
| Attention/expert overlap scheduler, 48 workers, relative to matched prepared control | -7.25% | -7.24% | -24.74% |
| Guarded unit-stride overlap, 48 workers | -7.95% | -7.79% | -19.74% |
| Retained 640-column prefix, early copy on ordinary stream | -10.40% | -10.23% | -10.71% |
| Native MLX attention with expert-first dispatch, first adapter pilot | -1.52% | -1.55% | -1.80% |

The first three scheduler pilots retained four measured runs per mode and
prompt plus two warmups. Each compared two candidate worker counts with the
prepared control, yielding 36 measured and 18 warmup generations. Every output matched.
The overlap candidate also passed all five full-model correctness cases,
including cache rotation and natural EOS: 566 byte-exact full-vocabulary logit
arrays per mode, 1,698 comparisons including the control.

The boundary scheduler uses completion counters to release each expert's down
projection independently, then computes the next layer's normalization and
QKV/router projections. The overlap scheduler additionally runs attention tasks
alongside expert work. Native cache updates occur exactly once per layer, and
fallback uses the already prepared native views. Neither is a whole-model
megakernel. The later retained-prefix experiment tests future-weight loading
separately.

An isolated access-pattern experiment identified a concrete problem in the
first attention port: arbitrary inner strides caused much of its long-context
primitive slowdown. At length 1,672, a guarded unit-stride version reduced
diagnostic wall time from about 259 to 193 microseconds, against 184 for native
MLX. All 25 numerical comparisons passed. Removing device-coherent qualifiers
had little effect in this test. The integrated stride fix reduced the long
prompt regression, but still lost 19.74% to the matched prepared control.

The retained-prefix experiment copies the first 640 columns across every
Q/K/V/router row of the next layer: 6,717,440 bytes, or 31.25% of those weights.
The consumer actually uses the retained values; an isolated check zeroes the
original prefix after copying to prove that. Both early and late copies passed
1,698 stock full-logit comparisons, and all 54 timing generations were exact.
Both ordinary-stream variants lost roughly 10–11%, with every copy included.
This uses a global GPU buffer, unlike Cohere's retained shared-memory tiles.

A dedicated MLX stream also passed all 1,698 full-logit comparisons. Its matched
short-prompt screen was much slower: prepared 68.801 tokens/s, early copy 50.875
(-26.05%), late copy 34.321 (-50.12%). All 24 generations were exact. This large
loss did not warrant expanding that screen to the other prompts. Neither copy
implementation establishes useful physical prefetch overlap.

Wider scheduler grids were rejected before performance promotion. At 96 workers,
the original primitive produced different output bytes even though every task
was counted complete. Repeated checks located expert-down corruption and
frequent idle-limit failures at larger grids. Review found shared task/finalizer
slot races and incomplete publication handoffs. The hardened code passed all
192 repeated primitive checks at 32, 48 and 64 workers. At 80 workers, 63 of 64
passed and one hit the idle limit; the public cap remains 64. Those are
constituent checks, and oversized-grid behavior is not considered resolved.

A separate direct-Metal diagnostic tested whether narrowing a barrier to one
resource removes its execution wait. All 240 output checks were exact. At low
occupancy, both the broad and resource-specific barrier took about 50.7 ms;
a legally reordered control took about 25.4 ms. That control demonstrates
overlap for this synthetic workload, while changing barrier resource scope
alone did not unlock it. This does not establish model throughput. It motivated
the native command-ordering experiment below.

A native MLX extension now reuses MLX's shipped SDPA metallib unchanged and
places expert-up work first, followed by an exact combined output/down tail.
Its real engine adapter passed 15 full-generation cases, 1,698 byte-exact
full-logit comparisons and 54,336 observed candidate branch calls, including
rotation and EOS. The first balanced pilot retained 54 measured generations
and 18 warmups, all exact. Prepared/native-up-first medians were
68.898/67.854 tokens/s (short), 68.823/67.754 (Rust), and 61.843/60.730 (long).
The constituent layer gain did not translate into an engine gain.

Subsequent host diagnostics found that re-enabling the adapter recreated its
compiled function. A bounded cache of pure compiled functions now preserves
that lifecycle across mode changes while all weights remain explicit live
inputs. Nine contract tests and another 1,698 full-logit comparisons passed.
The repeated throughput pilot is not included in this snapshot; the lifecycle
fix is not credited with a performance gain.

Five further constituent screens tested actual retained future-weight values
inside workers, without a global staging copy. Shared-memory tiles retain
128 or 192 columns; per-thread storage variants extend through 640 columns.
A productive version resumes current tasks while holding its future tile.
All 1,928 sampled output triples, 100 initial gates and 72 retention gates were
exact. In the retention gates the ordinary consumer prefix is zeroed, proving
that the staged values are used. No variant beat prepared. Per-thread storage
does not by itself prove physical register residency, and completion counters
are not a measured GPU overlap timeline. Sources, raw runs and detailed limits
are preserved under the retained-prefix result folders in the research bundle.

A reusable indirect compute-command test passed 336 exact dependent-chain
runs. Reuse lowered CPU encoding cost but increased GPU and drained wall time
in every synthetic case. It establishes API feasibility on M4 Pro, not a model
speedup. Actual integration would also need ICB-compatible pipelines and
supported indirect-resource/hazard APIs in MLX; its existing ordinary pipeline
objects reported no ICB support.

Additional preserved controls include producer-first task ordering, a simpler
queue-free attention/expert backfill attempt, native-equivalent attention tasks
through 65,537 KV tokens, and a weight-byte/host-cost analysis. These are tests
and measurements, not reasons to claim the 10% goal complete or options exhausted.

Reference: [Cohere's article](https://cohere.com/blog/megakernels) and source
`cohere-ai/cohere-megakernel@67d0b9ca22ea3652796b715d1d1863459e0e2c3c`.
The experiments are explicitly testing its smaller task dependencies, attention
backfilling and future-weight loading ideas. The additional 10% goal and
whole-model megakernel validation remain open.

The research branch and all committed raw results through
`57e0bb6d8fd597a82eb5947da5a8636f5d649ff6` are backed up in
`experiments/north_mlx_vlm/mlx-vlm-megakernel.bundle`. The bundle was verified and
requires the public MLX-VLM base `1ecf1ecdd28af102eded679be0daa5c76ab2a068`.
From an MLX-VLM clone containing that base, restore it with:

```bash
git fetch /path/to/uzu-metal-lab/experiments/north_mlx_vlm/mlx-vlm-megakernel.bundle refs/heads/cf/north-megakernel-engine:refs/heads/cf/north-megakernel-engine
```
