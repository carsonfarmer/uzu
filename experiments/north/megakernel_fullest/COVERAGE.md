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

## CPU checkpoint 2

`ready_tail.py` extends the real branch arithmetic to dynamically reserve future
Q/K/V/router tiles. Each worker holds at most one consumed prefix and continues
ready producer work; completion counts prep tasks rather than fixed worker IDs.
This is actual cross-layer staging implementation, **not yet runtime verified**.
The late-load control performs the same prefix copy and computation after norm.
No speed or physical-overlap claim follows from the implementation.

Six variants compile with the pinned MLX Metal headers. Failed CPU compilation
attempts are retained: missing imported constant, missing source augmentation,
coherent pointer qualification, and include-root setup were fixed. No GPU was
used. `check_tail.py` prepares a bitwise/intermediate/visit/progress gate at1,
20,32,64 workers. `bench_staged.py` prepares balanced primitive screens.

`native_runner.swift` compiles on CPU. After release, it can measure exported
primitive command-buffer GPU duration plus workspace fill, and a fill-only
control, with pipeline threadgroup memory/resource metadata. Its measurements
will be explicitly distinct from Python wall time and end-to-end decode.

## CPU checkpoint 3

Added a direct/no-staging control to the same ready engine, plus same-copy late
and early stages. All eight Metal compile configurations pass. `native_tail.py`
exports real weights and synthetic inputs for balanced native command-buffer
measurements of direct/late/early/fill-only. `cross_layer.py` carries actual
preparation into the next layer; `bench_decode.py` compares all paths with the
frozen prepared baseline under identical async submission. An AST audit matches
the prior async body exactly. All numerical and timing gates remain pending.

Source-derived cache accounting (not measured traffic) finds53–104MB extra
clear/copy traffic per short-context step, ~1.7–3.2% of prior3.206GB active-weight
accounting. At the long workload it rises to541–567MB, ~17%. This does not justify
assuming cache copies dominate the short-context deficit. Measure before native
cache ownership redesign. The GPU wrapper was exercised before release and
correctly refused to enter the target script.

## Additional primary-source constraint

Apple's [Metal Performance Primitives Programming Guide, section2.2](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
explains that its GEMM kernels can obtain memory/compute overlap through
occupancy and direct device-memory access, without explicit threadgroup staging.
This is guidance, not a measured conclusion about this exact W4 GEMV workload.
It strengthens the need for the direct/no-staging control and resource metadata;
a CUDA-style staging design is not presumed to win. Software buffering and
future immutable-weight reservations remain testable mechanisms.

Retained older partial-Q staging screens also warn against assuming benefit:
`full_layer/bench-prefetch-q-layer1-v2.json` has median wall451.71us prefetched
versus451.50us unstaged cross-tail and425.38us exact+MLX preparation; layer7
has528.33us versus520.67us and469.37us. These historical constituent timings
are not the new matched-async control and are not newly measured GPU durations.

## CPU checkpoint 4

Added `register_tail.py`: the same dynamically reserved future prefix is held
by its eventual consumer thread, avoiding the threadgroup buffer and testing a
different resource tradeoff. Eight configurations compile. The CPU address audit
checks every prefix read against its expected dense/router weight address and
checks per-thread slot bounds. Native, intermediate, and decode harnesses accept
`--storage register`. Register allocation/spilling, numerical exactness, and GPU
performance remain unverified. This is a single retained prefix per worker;
producer/consumer double buffering and full-model fine scheduling remain open.

## GPU checkpoint 1: real staging exact, initial route loses

Parent released `/tmp/north-minimal-baseline-20260914.done`; every run below used
the locked wrapper. `tail-gate-v1` failed before the gate due a relative source
path in provenance; v2 fixes it and passes all six intermediate arrays at1/20/32/64
workers, early/late, layer7. Register gate v1 passes the same checks on three
inputs at layers1/7/47 (72 cases), with exactly-once tasks and no progress errors.

Native v1 defaulted to standalone fast math and failed exactness. It is rejected
for performance inference and retained. MLX custom kernels default to Safe math;
matching safe math and this system's Metal4.1 language makes all native direct,
late, and early outputs bitwise exact. Native threadgroup v2 and register v1
show large timing stalls and broad paired intervals; they do not establish an
early-staging gain. Typical clear-only median is3–4us versus350–470us for the
tail, so workspace clearing is a small constituent cost here. This is GPU command
duration, not Python wall time, physical overlap proof, or decode throughput.

`staged-primitive-v1` passes all byte gates. All six explicit threadgroup-stage
variants lose in median wall latency against prepared across layers1/7/47:
prepared195/206/215us; staged208–251/229–250/227–256us. This standalone screen
stages before local norm only; cross-layer findings come from the next record.

`ready-register-decode-pilot-v1` passes124 full-logit array comparisons and all32
measured32-token generations. Fresh prepared+async denominator; eight balanced
rounds. Paired geometric ratios: direct0.9207 (95%0.9145–0.9256), late0.9093
(0.9054–0.9127), early0.9037 (0.9008–0.9072). This actual cross-layer consumed-prefix
implementation loses; no promotion. It does not exhaust task placement, ready
scanning reductions, double buffering, or whole-model fine dependency scheduling.

## GPU checkpoint 2: readiness scanning

`tuned_tail.py` claims available producers before scanning dependency bits and
avoids atomic increments on exhausted front/prep queues. It passes the layer7
intermediate/progress gate at1/20/32/64 workers. The tuned register32-token decode
pilot remains slower than its fresh prepared+async control; all full logits and
tokens match. See retained summary for paired intervals. No speedup promotion.

A source review found a coverage limit in the initial reservation order: resident
workers mostly reserve Q tiles first, so K/V/router support in the helper did
not imply early loads of all four operations. `interleaved_tail.py` adds a
bijective round-robin Q/K/V/router reservation order and per-operation audit
counters. Its CPU permutation check passes; GPU early-operation counts and
performance are pending. This is explicitly still a partial-layer task engine,
not the final full-model persistent scheduler.

## GPU checkpoint 3: whole-model fine DAG

`fine_whole.py` replaces the six MoE phase boundaries with440 ready nodes per
layer: Q-head/KV readiness, expert-specific down dependencies, and row joins.
Prefix/cache carry/head retain the phase queue. Per-layer states are never reset
within a decode, avoiding stale-worker reuse. Bounded idle diagnostics are checked.
Two-layer gates pass at1/20/32/64 workers. The complete49-layer + head gate passes
at20/32 workers, including every used cache byte and all logits/task completions.

`shared_pack.py` lets baseline modules view packed weights rather than retaining
two complete weight copies. `fine-whole-decode-pilot-v1` then passes93 full-logit
arrays, all per-layer task-state checks, and18 measured32-token generations.
The ready DAG is substantially slower (~23t/s) than both the phase queue (~56t/s)
and fresh prepared+async (~61t/s). Scanning costs are a hypothesis, not isolated
proof. `affinity_whole.py` adds continuous RR preferred lists with ready stealing;
that new scheduler is not yet GPU tested. Long-context SDPA algorithm compatibility
remains explicitly unimplemented, and the harness rejects that unsupported range.
