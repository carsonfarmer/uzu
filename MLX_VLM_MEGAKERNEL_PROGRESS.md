# Actual-engine megakernel research: progress, not a completed speedup claim

The preserved actual MLX-VLM improvement remains **10.25–11.62% over the untouched
published package**, documented in [MLX_VLM_INTEGRATION.md](MLX_VLM_INTEGRATION.md).
The active goal is another 10% beyond that optimized `prepared` control.
The first router-ready scheduling pilot now improves that control by
**3.11–3.48% in actual generation**, with exact stock logits. The goal remains open.

Research now lives in a separate actual MLX-VLM worktree:
`/Users/carsonfarmer/Developer/Personal/mlx-vlm-megakernel`, branch
`cf/north-megakernel-engine`. Normal MLX-VLM `load/generate`, async submission,
sampling, stopping and cache ownership remain in use. This work does not use
Uzu's inference engine or the withdrawn custom decode runner.

Earlier integrated scheduling candidates matched stock in their full-generation
test cases, but were slower. Later stress tests exposed synchronization defects
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
| Native dispatch after compiled-function lifetime fix | -1.62% | -1.49% | -1.78% |
| Exact router fusion added to prepared, six measured runs per mode/prompt | +1.09% | +1.21% | +0.38% |
| Router-ready expert release, matched actual-engine pilot | +3.32% | +3.48% | +3.11% |

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
The repeated throughput pilot retained 72 exact generations and remained
slower on all prompts: prepared/native-up-first medians were 68.848/67.730,
68.407/67.386, and 61.660/60.564 tokens/s. The lifecycle fix did not recover a
performance gain. Moving dynamic attention outside the compiled function also
failed to recover it in a separate exact short-prompt screen.

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

The native router is now an exact reusable task. Exhaustive BF16 checks found
that default JIT float32 exponential differs from the shipped native MLX unary
pipeline on 628 finite inputs. A local `metal::precise::exp` fixes every BF16
input pattern while preserving MLX's stable expert selection after sigmoid.
This precision change is confined to routing; previously exact expert math is
unchanged. Primary-source compiler details and rejected variants are retained.

The opt-in `routed` fusion mode replaces only sigmoid/top-eight/score gathering
inside the real prepared decoder. It passed 1,132 complete-vocabulary logit
comparisons across ten generation cases, including cache rotation and EOS, plus
48 exact timing generations. It improved native decode throughput by only
0.38–1.21% beyond prepared. This is a useful control for the next experiment,
not the additional 10% goal or a whole-model megakernel.

A router-ready scheduler now allows expert-up tasks to start while QKV work
remains unfinished. Its expanded constituent screen passed all 7,056 sampled
output sets, 84 initial gates and six incomplete-work poison gates. Across
three real layers and two prompts, early release reduced this section's median
time by 3.6–5.4% against the same scheduler holding routing until QKV completion.
All late-control runs started zero expert tasks before QKV completion; the
expert-priority early variant started all 96. These counters show program
ordering, not physical GPU overlap. Absolute timing drift and the initial
attempt's mixed source provenance are explicitly documented.

Early release is now integrated into ordinary MLX-VLM generation. The balanced
four-mode pilot retained 72 measured generations and 24 warmups, all stock-exact.
Prepared/early medians were 68.276/70.544 tokens/s (short), 68.283/70.659 (Rust),
and 61.788/63.709 (long). Early release beat the identical late scheduler by
5.57%, 5.70% and 5.03%; all 18 paired comparisons improved against both prepared
and late controls. Twenty full-generation correctness cases matched all 2,264
complete-vocabulary arrays, including cache rotation and EOS. Observed branch
calls rule out silently falling back for the entire run.

This establishes a useful earlier-dependency result in the actual engine.
It does not establish physical GPU overlap or a full-model megakernel: native
attention still starts after the combined preparation/expert primitive.
The all-router-tiles-first Apple queue adapts Cohere's independent router
dependency; it does not copy their QKV-then-router wave placement verbatim.

A 6,084-sample router tile-size ablation found no consistent improvement over
the original eight-row choice; all numerical and failure gates passed and the
runtime geometry remains unchanged. Removing two diagnostic counters produced
no measurable additional gain in 96 exact timing generations; another 3,396
full-logit comparisons passed. The early-release gain over matched prepared
remained 2.72–3.11% in that repeat.

A bridge from the current layer's final operations into the next early-release
front passed its numerical and incomplete-work gates. The expanded six-case
screen retained 2,916 exact samples, 54 gates and 18 poison checks. Keeping 32
workers was near parity with separate kernels, while starting 64 and retiring
half after the tail made it slower. No reliable improvement was promoted, and
this bridge has not been timed in the actual generation engine.

An attention prerequisite now reads the ordinary old cache while substituting
one newly computed K/V slot. All 84 cases, 252 initial gates and 1,008 sampled
outputs matched native SDPA exactly, through length 8,191 and including strided
views, append and physical-slot replacement. An initial empty-input compiler
failure is preserved. This gate establishes arithmetic, not model cache
lifecycle or throughput. Integration is now testing one full-attention MoE
layer with the ordinary native cache update API before considering expansion.

That opt-in combined layer now passes full-checkpoint diagnosis: 15 ordinary
generation calls and 1,698 complete logit arrays matched stock. The candidate
executed 308 times across short/Rust/EOS; long and rotating-cache cases correctly
fell back. The other eligible layers retain the established early-release path.
The first expanded unit gate nevertheless had one changed-weight comparison
failure. Its values were not captured, and no cause or fix has been identified.
Subsequent varied-weight, retained-intermediate and 32 fresh-process runs passed;
the latter ran the original 14-test invocation in 32 processes, 448 tests total. These
passes do not resolve the initial observation. Performance promotion remains
withheld, and the raw failing log is preserved alongside the passing evidence.

A per-KV-group attention readiness/placement ablation passed 2,040 samples,
60 gates and 54 incomplete-work poison checks. Grouping the dependent Q/K/V
tiles and prioritizing them removed some combined-kernel overhead, but the best
variant was only near parity with separate early release across six real-input
layer cases. It is not an additional engine speedup claim. All variants include
native cache updates; captured old-buffer aliases mean their component timing
cannot measure live generation's cache donation behavior.

Native single-token RoPE also has a tested arithmetic helper: three candidates
matched all 152 cases, including all finite BF16 patterns at two placements and
19 offsets through 131,071. Other multiplication/contraction orders changed
thousands of output values and are explicitly rejected. Its Q/K projection
epilogue passed 1,836 component samples, 54 gates and 48 poison checks on six
sliding-layer input cases. Component latency improved by 0.57–3.36% against
separate early release, but this did not predict a real-generation improvement.

The expanded `virtual_grouped` model path combines preparation, routing,
expert-up, per-KV-group attention and the output/down tail for all 48 eligible
MoE layers below context length 1,024, including native-rounded RoPE. It passed
2,264 complete logit comparisons in 20 ordinary generation cases and ran
14,784 times. Long and rotating-cache cases use the established fallback.
The diagnostic pilot retained all 72 stock-exact generations: grouped versus
early release was 65.308/70.785 tokens/s (short), 65.029/70.514 (Rust), and
63.701/63.662 (long fallback). The combined path is about 7.8% slower, and is
not promoted. The earlier unexplained changed-weight unit failure remains a
separate reason to withhold promotion.

A matched host profile found less process CPU time for grouped generation
(0.824 versus 0.857 seconds), despite higher API wall time (2.245 versus 2.097).
The instrumented native cache wrappers added only about 3.6 milliseconds.
That does not establish cache copying or occupancy as the cause; it directs
the next ablations toward GPU execution and graph dependencies. Changes to
compiler threadgroup bounds produced no clear gain across 780 exact component
samples. Actual generation remains the performance acceptance test.

The long-context combined layer also remains slower. Specializing append loads
reduced its best median from 630.73 to 577.69 microseconds, but separate early
release took 496.58. All 260 samples, ten gates and nine poison checks passed.
This removes some overhead introduced by the attention port; it does not
improve the actual engine and has not been integrated for long context.

Actual short-generation ablations now identify two costs introduced by the
combined path. Joining native cache dependencies once at the model output
raised grouped decode from 65.396 to 67.400 tokens/s; simplifying append loads
raised it to 66.647; together they reached 68.722. Early release was 70.876.
All 35 generations and 1,698 full-logit comparisons matched stock. The changes
recover part of the regression, and remain benchmark-only. A subsequent
worker-count screen kept all 42 generations: the best combined configuration,
40 workers, reached 68.923 versus early release at 70.816. No new winner.

A 64-worker warmup in an earlier screen exceeded 90 seconds and was terminated.
Its numerical/audit micro-gates had passed, but actual generation did not finish;
this is recorded as a timeout, not a performance measurement or proven deadlock.
A fresh native GPU recovery check passed. Subsequent generation screens have an
explicit per-call deadline. The failed screen and stack evidence are preserved.

Static task assignment did not improve early release across 3,120 exact
component samples. Per-expert down readiness in the newer combined graph also
failed to produce a consistent gain at 128/256/512 rows per task. The latter
extends an earlier boundary prototype; it is not the first per-expert test.
One task-number substitution corrupted a workspace offset in its initial
256-row source. That rejected gate and source are retained; the corrected
six-case run passed all 780 samples. This identified/fixed bug is separate from
the older unresolved changed-weight unit observation.

A follow-up 64-worker replay also returned the wrong first token during ordinary
generation. An instrumented run found an idle-limit failure and NaNs at layer42,
but it forced per-layer evaluation and changed array lifetimes. Final counters
cannot establish producer readiness at the moment of the first timeout. Saved
real inputs then passed18 isolated replays. The actual-generation failure is
preserved and unresolved;64 workers are rejected for further promotion.

Splitting the combined tail into separate output projection, per-expert down
projection and ordered join did not help actual short generation. Late down
release reached66.725 tokens/s and early down63.982, versus70.909 for early
router release. All24 generations were stock-exact, and10 correctness generations
matched1,132 full-logit arrays with29,568 observed candidate layer calls. The
slower component was not expanded into another three-prompt performance claim.

Per-KV-group output projection also lost in a separate component screen. It
carried each lane's FP32 sum through four ordered groups to preserve the native
reduction order. All780 samples,30 output/cache gates and24 poison checks
passed. Counters recorded128 or192 output tasks starting before all attention
finished, versus zero in the matched late control. Early release still lost to
the separate early decoder in all six input cases. This establishes an executed
dependency change, not useful physical overlap or an engine speedup.

The larger multi-layer experiment now has an actual backend path. An isolated
MLX0.32.2 worktree adds indirect input/output resource registration while keeping
the existing hazard, size, encoder and lifetime rules. A native extension uses
MTLArgumentEncoder to reference live MLX arrays. It passed72 resource cases up
to931 inputs and an80-call lazy native/indirect chain. The original published
metallib is unchanged; the rebuilt prepared and early-release engines separately
matched1,132 full-logit arrays across all five test prompts. The additive patch,
build manifests and resource sources are preserved. The indirect kernel now executes all 48 MoE layers in one dispatch during
normal MLX-VLM generation. Dense layer 0, final normalization, the vocabulary
projection and native cache commits remain outside that dispatch. All 566
full-vocabulary arrays match stock over the five-case gate; counters confirm
308 executed spans (14,784 layer calls) in short/Rust/EOS cases, while long and
rotating-cache cases use the established fallback.

A balanced six-mode short pilot recorded 36 measured and 12 warmup generations,
all stock-exact. Prepared reached 68.938 tokens/s, early router release 70.696,
four-layer persistence 70.716, eight-layer 70.696, sixteen-layer 70.138 and
48-layer 67.786. The full span is 4.12% slower than early release and 1.67% slower
than prepared. Four/eight-layer differences are effectively zero. Reducing
launches alone has not produced an additional gain.

A separate long-context component gate covers 4/8/16/48 persistent layers
using native 128-partition attention arithmetic. All 608 returned arrays and
eight abort-poison checks passed. The actual generation adapter then matched
566 full-vocabulary arrays across five prompts, executing 129 long-context spans.
Its balanced 24-generation pilot was strongly negative: 43.408 tokens/s versus
64.267 for early release and 62.457 for prepared. This path remains experimental.

Moving the final cache-dependency join from normalized hidden states to public
logits also passed 1,132 full-logit arrays and 24 timed generations. It reached
67.784 tokens/s versus67.805 for the matched scoped-hidden control and70.892
for early release. It provided no useful gain.

Retained future QKV/router weights now run inside the multi-layer dispatch.
Four tile sizes (8/12/16/24 KiB) and separate compiled retained/non-retained
preparation paths passed the checkpoint screens. Deliberately corrupting the
retained copy proves that later layers consume it; the normal variants remain
exact. Early loading sometimes beats the matched late control, but no screened
variant consistently beats original persistence on both short and Rust inputs.

The selected 12 KiB specialized early and late paths each passed another 832
native-array checks, eight abort checks and four consumption proofs. The native
generation gate then matched 1,132 full-vocabulary arrays. The balanced five-mode
comparison preserved all 105 generations: 75 measured and 30 warmups across
three prompts. Early loading improved short/Rust throughput by 1.98%/1.70% over
matched late loading, but remained 1.42%/1.47% below persistence without prefetch
and 5.45%/5.61% below early expert release. The long case uses the shared fallback.
Both policies use the same storage and reservation rules, but timing can change
reserved tile counts; this does not isolate pure physical memory overlap.
At most 32 tiles are retained per transition:
384 KiB, about 1.83% of the next QKV/router matrices. This is a much smaller
prefetch window than Cohere's Hopper implementation. No new winning mode was
established.

Final normalization and the full vocabulary head have also been included in
the persistent dispatch in checkpoint component tests. Dynamic and static
head scheduling each passed 800 native-array comparisons and four intentional
abort checks. Neither improved the balanced component timing; full generation
has not been claimed for these head variants.

An 18-generation CPU profile measured 148 ms more main-thread work for the
persistent stack than early release, alongside an 81 ms wall penalty. Live-input
guards and graph construction were substantial in a separate instrumented run.
A cached MLX compiled body therefore received lifecycle tests (live weights,
cache growth, strides and output-only completion), all five stock correctness
cases with 566 full-vocabulary arrays, and a 72-generation comparison. It did
not improve throughput: 67.830 versus 67.899 tokens/s for uncompiled persistence
on short, and effectively tied on Rust. Both remain about 4.2% behind early
release. A second 18-generation profile measured about 50 ms less instrumented
graph-construction CPU, but variable whole-run CPU and unchanged throughput.
Saved CPU need not reduce wall time when it overlaps GPU execution. MLX
compilation is an integration technique, not a new contribution from Cohere.

A long-attention component variant groups all native partial calculations and
their reduction for one head under one worker, reducing scheduling from 801 to
289 tasks. A second variant reuses partial scratch after all preceding layer
tasks finish, reducing 48-layer workspace from 53.9 MB to 3.1 MB. Each passed
424 native-array comparisons and five abort checks; the two screens retained
90 exact timing samples. In the matched three-mode screen, the 48-layer
component measured 21.051 ms for the old schedule, 17.837 ms for grouped heads
and 17.417 ms with shared scratch. These changes improve a slow prototype;
they are not gains over native generation. The numeric 128-partition attention
and reduction order remain unchanged. Small-span raw times varied substantially
and are retained without exclusions.

A guard-collection experiment also removed repeated host-side validation work
without reusing stale weight or cache references. Its 219 guard/lifecycle records,
566 stock-logit arrays and 72 generation comparisons were exact. It did not
improve throughput over the cached compiled adapter.

The shared-scratch attention path now supports native rotating-cache positions.
A component gate passed 1,696 arrays and 20 abort checks; actual generation passed
566 complete stock-logit arrays, with 129 persistent 48-layer spans each for long
and rotation. Native cache methods still own all updates. The 48-generation pilot
measured 51.807 tokens/s on long, versus 64.111 for early release, and 30.951 on
rotation, versus 57.884. This closes a correctness/integration gap, while remaining
slower than the real engine control.

A matched follow-up applies ordinary native cache loads to partitions that cannot
contain the replaced slot. Only its owning partition uses virtual replacement;
append behavior and arithmetic order remain unchanged. Another 1,696 component
arrays, 53 lifecycle records, 566 stock-logit arrays and 48 actual generations
passed. Long remained effectively unchanged (51.723 versus 51.763 tokens/s).
Rotation improved from 30.809 to 37.095, but still trailed early release at 57.872
by 35.90%. The 20.40% increase is relative to the slow ring prototype and must not
be reported as a gain over MLX-VLM.

A separately rebuilt diagnostic backend then observed native cache-buffer reuse
inside nine normal, fresh-process generations. Four calibration cases established
that it distinguishes native donation from copying while preserving independent
old views. Every model generation matched stock tokens, text and stopping; source
and binary hashes stayed fixed. Prepared and early-release controls reused all
12,642 observed buffers on every prompt. Persistent mode instead scheduled copies
of 4.609 GB of logical cache contents on short, 24.333 GB on long and 54.453 GB on
rotation during each 128-token generation. These are logical copy payloads in
instrumented runs, not measured HBM traffic or an attribution of the timing gap.

The MLX source provides a concrete explanation to test: GPU evaluation holds its
input buffers until completion, while native SliceUpdate requires exclusive
ownership to reuse a buffer. Our persistent span reads old caches before the
native update, creating a lifetime/ownership cost that the ordinary model's
update-before-attention order avoids. A separate candidate is now testing safe
reuse when the update itself depends on that completed read. It must preserve
external aliases, unrelated readers, stream ordering and every GPU lifetime hold.
The first reuse candidate subsequently passed 30 synthetic cases (352 cache
updates), 160 native lifecycle checks and 20 actual-engine generations with
2,264 byte-exact full-vocabulary arrays. It nevertheless avoided only 120,
139 and 413 additional copies on short, long and rotation respectively. Logical
copy payload fell only 0.90%, 1.10% and 3.26%; no timing win is claimed.

An optional rejection diagnostic found the dominant reason: the cache had a
unique descriptor and three Data owners, but only one GPU producer hold was
credited. A second diagnostic recorded 129 native `Depends` evaluations each
retaining 96 cache dependencies without GPU reads. The native implementation
only aliases returned outputs, while `gpu::eval` retains every dependency input
until completion. This explains an additional lifetime hold that the initial
policy conservatively refused to discount. A narrower follow-up now accounts
for the exact native `Depends` type, excluding forwarded cache outputs and
requiring a matching real producer. It keeps all original lifetime references.
The follow-up passed 52 synthetic cases (690 cache updates), 160 lifecycle
checks and 20 full-model generations with 2,264 exact full-vocabulary arrays.
Six matched model audits then reduced logical copied cache payload by 50.14%
on short, 49.24% on long and 70.75% on rotation. Every output was exact.

An uninstrumented causal screen ran 12 balanced fresh processes: 72 actual
generations, 36 measured, two observations per setting/mode/prompt. Enabled reuse
improved the persistent prototype by 1.83%, 4.14% and 10.71%, respectively. It
still trailed early release with reuse disabled by 2.69%, 15.74% and 28.98%.
Control on/off differences were below 0.24%; this small sample does not establish
zero tracking overhead. All generation outputs and source/binary hashes matched.
These are gains within the slower prototype, not new wins over the real engine.

This screen attributes part of the integration regression to added cache copies,
while leaving a substantial gap. It does not establish an inherent Apple GPU
limitation. Detailed observations and timings are in `results/cache-copy-audit-v1`,
`results/cache-reuse-model-audit-v2` and `results/cache-reuse-v2-screen`.

Removing the adapter's extra output dependency now passes full native-engine
checks. Twenty-five public generations match all 2,830 preserved stock-logit
arrays. Five native reference runs record 490 final cache arrays; the other
20 runs independently match 1,960 drained cache arrays, their metadata and
capacity shapes. Separate lifecycle checks cover continuation without a prior
cache drain, EOS and early stream close. Pending graphs remain bounded in the
recorded repeated-forward tests and clear after explicit evaluation.

Nine fresh diagnostic generations include all final cache writes and record
12,642 updates each. Fully drained cache contents and history match prepared.
Removing the dependency avoids 192 copies per generation, reducing logical
copy payload only 1.66% on short, 1.56% on long and 1.53% on rotation. Prepared
reuses every observed cache buffer. This changes the integration contract
safely in the tested cases but leaves most added copies in place.

A balanced short screen preserves 24 exact generations, including 16 measured.
Prepared/early/joined/lazy medians are 68.914/70.823/67.804/69.285 tokens/s.
The lazy adapter improves on joined persistence, but remains 2.17% behind early.
Its final cache drain adds a median 4.20 ms, versus 0.56 ms for joined and
0.23–0.25 ms for the native controls. Including that work, lazy's fixed-work
rate is only 0.33% above prepared and 2.11% below early. No additional goal win
is established, and this small screen was not expanded into a headline claim.
Raw results are in `results/lazy-cache-copy-audit-v1` and
`results/indirect-lazy-short-pilot-v1`.

A separate descriptor-history policy now carries a proven read-before-write
relationship across successive native cache updates. It retains weak identity
for the exact successful output descriptor, requires the same Data and stream,
counts each holder once, and preserves all original GPU lifetime references.
External views, branches and unproven readers retain native copy-on-write.
This is an MLX integration change motivated by the difference from Cohere's
in-place K/V writes, rather than a technique claimed in their article.

The initial 52-case gate and its repeat on the final build each pass 690 cache
updates. Twenty-five additional observations pass retained-alias, double-credit,
branch, view, growth, stream, weak-identity and injected-failure checks. Their
2,089 audit decisions include one deliberately failed write attempt; that total
must not be called completed writes. In both 512-update lazy chains, reuse
increases from 256 to all 512 buffers. Allocation failure loses a reuse
opportunity without changing results, and a pre-write exception recovers exactly.

Nine fresh actual-engine copy audits then match all final cache contents,
metadata and history, with 12,642 completed updates per generation. On the same
binary, enabling history over the previous reuse policy reduces logical copied
bytes from 2.298 GB to 0.070 GB on short, 12.188 GB to 0.466 GB on long and
15.698 GB to 1.189 GB on rotation: reductions of 96.97%, 96.18% and 92.43%.
History reuses 12,390 buffers per case; prepared reuses all 12,642. These are
observed logical copy payloads, with every final write included, not physical
traffic or throughput gains. The remaining 252 copies have not been attributed.

The history-enabled engine also passes the existing 19-test lifecycle suite
and 20 public generations with 2,264 complete stock-logit arrays exact across
short, Rust, long, rotation and EOS. All captured sources and binaries remain
unchanged. The three-policy, counterbalanced screen has now finished: 108 exact
public generations, 54 measured, with two observations per setting/mode/prompt.
History-enabled persistence reaches 69.858/56.300/42.930 tokens/s on
short/long/rotation. That improves its native-ownership prototype by
4.27%/9.03%/15.82%, but remains 1.09%/11.46%/25.70% behind early release with
native ownership. Against prepared with native ownership it is
+1.68%/-8.82%/-23.85%. Control policy shifts range roughly -0.17% to +0.46%;
this small sample does not establish zero overhead. Source and binary hashes
remain stable. Raw runs are in `results/cache-history-v1-screen`.

The copy fix explains another part of our integration cost, but does not meet
the additional 10% goal. A bounded follow-up tests compiled, collected and lazy
host adapters with the fix enabled. Separately, source assessment identifies
96 cache-slot writes per decode step that our 48-layer span still leaves to
native SliceUpdates. Cohere places these writes inside its QKV tasks. A pure
cache-output integration is the next implementation experiment; no correctness
or performance result is claimed for it yet.

The host-adapter composition follow-up is now measured. All compiled, collected
and lazy variants preserve history reuse and pass full logits/continuation/close.
The balanced short screen records36 exact generations,24 measured. Plain,
compiled,collected,lazy and early medians are69.772,69.674,69.768,69.723 and
70.736 tokens/s. Including final cache completion, the three compositions are
0.13%,0.05% and0.30% below plain. They do not justify a larger gain campaign.

A new functional cache-output extension moves the slot stores into QKV tasks.
It first passed848 native component output comparisons,424 full-cache checks
and424 retained-input checks across1/4/48-layer spans on short and Rust.
An initial capture without physical cache capacity failed safely; its cache-only
checker could pass an empty reference update, so the repeat explicitly requires
an existing slot and successful tasks. Both runs are preserved.

The first actual-engine integration passes four short public generations with
516 complete stock-logit arrays and exact final caches, metadata and capacities.
Native cache methods handle growth and advance metadata exactly once; an exact
SliceUpdate graph matcher then substitutes the equivalent functional outputs.
The candidate executes129 spans/6,192 MoE layers. Observers confirm elimination
of96 separate slot writes per eligible decode step. It still copies every cache
output in this run; no speed claim is made. The full set now passes 20 public generations and 2,264 stock-logit arrays,
plus 40 continuation generations and four early-close streams. Long and rotation
use the existing fallback; the new kernel currently handles short, Rust and EOS.
Safe output reuse is in progress. Initial v1 component/fixture manifests lacked a dyld check;
the actual-engine wrapper asserts the loaded library and captures preamble inputs.

A source-inventory review also found that older copy audits captured their
manifest before several shader builders were lazily imported. Their observer
records, output checks and library hashes remain valid, but those manifests
cannot independently prove the omitted source files stayed unchanged during
each process. The original records are preserved with that limitation in
`docs/north-cache-audit-provenance-scope.md`; no hashes were added retroactively.
The new lazy audits and repeated correctness gates explicitly capture those
helpers before generation. The earlier full timing harness already captured
them in its separate indirect-source manifest.

Reference: [Cohere's article](https://cohere.com/blog/megakernels) and source
`cohere-ai/cohere-megakernel@67d0b9ca22ea3652796b715d1d1863459e0e2c3c`.
The experiments are explicitly testing its smaller task dependencies, attention
backfilling and future-weight loading ideas. The additional 10% goal and
whole-model megakernel validation remain open.

The research branch and all committed raw results through
`3929d8244f00658506c5a3fc2368d5b0f4e57cab` are backed up in
`experiments/north_mlx_vlm/mlx-vlm-megakernel.bundle`. The bundle was verified and
requires the public MLX-VLM base `1ecf1ecdd28af102eded679be0daa5c76ab2a068`.
From an MLX-VLM clone containing that base, restore it with:

```bash
git fetch /path/to/uzu-metal-lab/experiments/north_mlx_vlm/mlx-vlm-megakernel.bundle refs/heads/cf/north-megakernel-engine:refs/heads/cf/north-megakernel-engine
```
