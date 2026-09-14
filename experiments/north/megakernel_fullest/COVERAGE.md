# Mechanism coverage ledger

Status on opening this branch: **unexhausted**. No new GPU runs yet.
Source anchor: Cohere `67d0b9ca22ea3652796b715d1d1863459e0e2c3c`, inspected locally
at its clean pinned checkout; article https://cohere.com/blog/megakernels.
The numerical control is unchanged `additional/layers.py` prepared async at
`1ebab5ff`, not the historical host runner. Timing must match submission.

| ID | Mechanism and discriminating experiment | Existing evidence / gap | Next action |
|---|---|---|---|
| M1 | Competitive primitives under identical inputs: QKV/router, attention, expert front/down, norm/join, head | Prior exactness gates do not establish primitive speed within giant kernel | Measure isolated native/custom duration and fused combinations, retain wall and GPU distinction |
| M2 | Persistent scheduling without whole-operation false dependencies | Whole-pass queue uses six global phases/layer; old per-expert branch queue exists | Generalize ready DAG and benchmark phase versus ready scheduling with identical math |
| M3 | Host task lists, round-robin cursor across waves, dependency affinity | Cohere schedule.py:1244/1287 implements placement, unlike generic phase claiming | Implement lists with ready fallback/stealing; compare equal task geometry and quotas |
| M4 | Real future QKV/router weight load retained until dependent computation | Whole-pass cache warming is a surrogate. Older tail_prep.py DOES retain first192 Q rows (3.66% of5248 rows), consuming staged weights after norm; router/K/V absent | Generalize prefix staging to Q/K/V/router with same-load late control and audited early bytes |
| M5 | Producer/consumer buffering across tasks | Cohere megakernel.cuh:975–1045 loads W into ring before input wait, then X; Metal implementation absent | Explicit buffer ownership/epochs, threadgroup or register stages, depth/size sweep; compare late load and single buffer |
| M6 | Fine expert dependencies and attention group readiness | Existing branch waits per expert. Whole-pass waits all front tasks. O-proj reference reduction spans4096 elements | Per-expert down early; for O-proj preserve exact sum order while feeding ready K/V groups, no reordered partial sums |
| M7 | Cache carry and output initialization cost | whole_pass.run init_value=0 clears all outputs, then phase0 copies used cache; no isolated cost attribution | Measure clear/copy bytes and standalone costs first; native alias-safe cache path only if material |
| M8 | Monolithic compiler/resource cost | Whole kernel includes head, dense prefix and all stages; no resource attribution | Compare1/2/4/8 dispatch cuts with same math; inspect compiler/counters where available |
| M9 | Global barrier failure mechanism and progress | Retained static run hangs after2 exact steps; root cause not established. Historical failure doc overstates cause | Bounded polling records phase/arrivals/worker visits; simulate adverse residency; no unbounded rerun |
| M10 | End-to-end composition | Strongest prep+async control ~68t/s historical, not valid current paired estimate | Fresh balanced short/Rust/long matched-async runs, full logits/tokens, raw pairs and regressions |

## Scheduling and staging design obligations

- A worker must never hold a future task and spin while it could execute that
  task's unfinished producers. Ready fallback must remain available.
- Staged reservations must be dynamically claimable, not require every grid
  group to become resident. Late workers can claim unstaged work.
- Loads counted as overlap must populate bytes later consumed by the exact
  arithmetic, not checksum reads or a second global read of the same prefix.
- Early-load counters establish ordering, not physical simultaneous memory and
  arithmetic execution. GPU traces or controlled latency comparisons are needed
  for an overlap claim.
- Preserve MLX dense/router reduction trees, BF16 cast points, expert order,
  and cache semantics. Partial dot products cannot be combined differently.
- Instrumented runs are correctness/diagnostic evidence. Timing controls carry
  equal diagnostics or both disable them; no hidden baseline extra eval.

## Initial source findings

The whole-pass queue's task-counter stages enforce global operation readiness,
although workers themselves dynamically claim jobs. This differs from Cohere's
per-producer counters and placement. `tail_prep.py` already contains real small
Q staging and should not be conflated with whole-pass cache warming. Its fixed
48 prefetch-owner groups and exit test need residency/progress scrutiny before
reuse. It also reserves16KiB threadgroup memory while one SIMD consumes each
four-row tile; primitive competitiveness must be measured.

The rejected static scheduler waits for all WORKERS at every barrier. This is
a plausible residency deadlock route, not a demonstrated cause of the recorded
hang. Stale visibility, publication ordering, and initialization remain audit
questions. A successful bounded stress test would only support that tested
configuration, not establish universal Metal progress.

## CPU checkpoint 1

`staged_prep.py` retains a128/256-element K prefix for every Q/K/V/router task
and consumes it in the original arithmetic order. Early/late variants move the
identical load before/after local RMSNorm. All six generated Metal variants
compile offline (`results/staged-cpu-compile-v1.json`). Runtime exactness and
competitiveness are **unverified**. This primitive does not include prior-layer
work and cannot establish cross-layer overlap.

`schedule.py` specifies per-head Q/RoPE/KV readiness, per-expert down readiness,
row joins, and next norm. Continuous round-robin/affinity lists use a ready-task
fallback so a blocked preferred task cannot pin a worker. Thirty CPU stress
trials with partial residency and random completion execute each task exactly
once (`results/schedule-cpu-v1.json`). This checks the abstract graph and claim
policy, not Metal memory visibility, residency, or arithmetic. It deliberately
keeps O-proj dependent on all heads to preserve its4096-element reduction.
