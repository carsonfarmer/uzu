# Additional ten-percent research

> **Completion claim withdrawn after control audit.** The earlier final control
> inserted `mx.eval(logits)` before reading argmax. The strongest historical
> host-token loop directly reads `mx.argmax(logits[0, -1]).item()` and has no
> such extra evaluation. Statements below calling that loop unchanged, or
> declaring the target achieved, are superseded. The old data are retained as
> measurements against the extra-evaluation control. Corrected balanced
> experiments are in progress; the throughput goal is active again.

Base: `008d0282ad46cddc4dbf614a554301b7586dcb54`. Control: exact branch,
160 front workers and 16 down rows, unchanged. Goal is candidate/control >=1.10
in fresh matched end-to-end measurements. Historical 59.952 tok/s is context,
not a current denominator. The final prepared-async confirmation meets the
target; the evidence and limits are in README.md. Remaining avenues were not
exhausted.

## Evidence rules

Every GPU process must use `gpu_run.py`, which refuses to start before the
parent's release file and holds an exclusive `fcntl.flock` for its lifetime.
Use the original pinned Python executable with `-B`; model and reference source
are read-only symlinks under this worktree's ignored `work/`. All artifacts go
in this worktree. Never write into the original runtime or weight directories.

Constituent tests screen candidates. Promotion requires full-vocabulary raw-byte
equality against pinned stock MLX for every checked step, independent greedy
token identity, and repeated counterbalanced 128-token timing on Python, Rust,
and a longer context. Record all regressions. Do not substitute tolerances,
launch counts, instrumented timing, or matching argmax for these gates.

## Hypothesis ledger

| ID | Hypothesis and causal experiment | State / outcome |
|---|---|---|
| H1 | QKV/router launch boundaries and unequal task sizes waste time. Compare the native constituent with independent fused tasks at several geometries. Keep the router's distinct reduction order. | Implemented, not run; awaiting GPU release. |
| H2 | Repeating the 2048-element RMSNorm locally in each preparation group costs less than a separate dispatch and dependency. Compare H1 with and without local RMSNorm. | Implemented, not run. Extra work may outweigh saved launch. |
| H3 | More independent attention/MoE mixing can improve utilization. | Parent is measuring the existing branch-mixing ablation; await evidence before duplicating it. |
| H4 | Down projection task sizing and expert placement dominate the branch. Profile the unchanged exact front/down against MLX primitives before changing arithmetic or scheduling. | Pending. |
| H5 | Removing cache prefix copying improves the whole-pass queue. Compare alias-safe native updates across context lengths. | Pending; helps the queue, but cannot itself beat the fused control which already avoids that copy. |
| H6 | Phase synchronization and giant-kernel compiler resource costs outweigh saved launches. Compare constituent costs, then selective phase cuts. | Pending; existing static scheduler hang prohibits using it as a candidate. |
| H7 | True weight staging can overlap current arithmetic. Prototype register/threadgroup double buffering preserving reduction order after identifying a memory-bound constituent. | Pending. Prior cache warming lost 4.79–24.85%; this does not exhaust real staging. |
| H8 | Output head and CPU dispatch limit total speedup. Measure head constituent and decode CPU/GPU timing before choosing additional fusion. | Pending. Old traces are instrumented observations, not a current causal breakdown. |
| H9 | Unused custom-kernel bindings and MLX command-buffer byte accounting induce submission overhead. Compare unchanged exact math with unused down-weight bindings removed; independently sweep batching settings across fresh matched processes. | Lean-binding candidate implemented, not run. All up/gate expert banks still exceed the default byte threshold, so trimming alone may not change commit count. |

## CPU-only findings

Safetensors header accounting gives 3,206,881,280 active weight bytes per token
under a one-read assumption: 1,849,688,064 attention projection bytes,
1,019,215,872 active expert bytes, 301,989,888 output-head bytes, and the
remaining router/prefix/norm weights. Thus attention projections account for
about 58% of these bytes. Quantized experts do not imply quantized QKV/O here;
those projections are BF16. See `weight-bytes.json`. This is a lower-level work
inventory, not measured memory traffic or a hardware bandwidth claim.

Parent provisional branch ablation: mixed-versus-blocked task-ID ordering was
approximately neutral, while mixed-versus-two-separate-front-launches improved
about 3%. Confirmation is pending. With 160 workers and 160 front jobs,
INTERLEAVE changes placement, not whether branches share a dispatch. The split
contrast combines launch overhead and possible overlap; it does not isolate
either mechanism.

Parent follow-up: MLX `device.cpp` uses a concurrent Metal compute encoder;
independent separate launches can overlap too. A dependency-forced serialized
control is being added in the parent. Consequently even calling the split
front "serial" would be unjustified. Parent uninterrupted runs also showed
substantial system variation (all samples retained); no final ablation estimate
has been imported here yet. Local MLX source is clean at
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`.

`CommandEncoder::set_input_array` increments `buffer_sizes_` once per distinct
input buffer; `needs_commit()` compares MiB with the device threshold. The measured runtime architecture is `applegpu_g16s`; local
source switches on the suffix `s`, selecting 50 ops / 50 MiB, and reads `MLX_MAX_MB_PER_BUFFER` and
`MLX_MAX_OPS_PER_BUFFER` once into static values. Thus runtime environment
changes require fresh processes. The decode harness records both variables.
`lean.py` removes only the unused down/scales/bias inputs from the front;
all arithmetic and launch geometry remain those of the exact control.
Changing batching for both control and candidate cannot be reported as a
candidate-only kernel gain; compare configuration effects separately.

CPU checks: all new Python files parse; `git diff --check` passes. The GPU guard
refused an invocation while the release signal was absent, before MLX imports.
These checks do not validate Metal compilation or arithmetic.

## Reproduction after GPU release

Run from this worktree using the original runtime read-only:

```sh
/Users/carsonfarmer/Developer/Personal/uzu-metal-lab/work/north-venv/bin/python -B experiments/north/additional/gpu_run.py experiments/north/additional/bench_prep.py --output experiments/north/additional/results/prep-v1.jsonl
/Users/carsonfarmer/Developer/Personal/uzu-metal-lab/work/north-venv/bin/python -B experiments/north/additional/gpu_run.py experiments/north/additional/bench_parts.py --output experiments/north/additional/results/parts-v1.jsonl
```

Select preparation geometry only from passing screening results. Run a short
full-logit pilot first, then the full independent prompt checks and six-pair
timing using `decode.py`. Its default geometry is a starting point, not a
measured winner. Keep every failed and successful output under `results/`.

## Cohere grounding and limits

Revisited https://cohere.com/blog/megakernels and local source pinned at
`67d0b9ca22ea3652796b715d1d1863459e0e2c3c`, including
`src/decode/schedule.py`. The relevant design is scheduling small ready tasks,
retaining competitive standalone kernels, removing false dependencies, and
moving immutable weights early. Hopper-specific TMA and warp specialization do
not transfer automatically to Apple Metal. H1/H2 isolate a small subset without
claiming to implement Cohere's whole serving engine.

`prep.h` extracts the existing MLX-derived QKV/router arithmetic from
`whole_pass/kernel.py`; RMSNorm follows the existing exact helper. Attribution
and license remain in `../quantized/NOTICE.md`. The prototype has no global
barriers: each group writes disjoint output rows, and the optional normalized
output is written only by group zero. This is a source-level progress argument,
not a claim of completed validation.

The checkpoint's NORTH_RESULTS.md calls the queue safe and claims broad
validation; the measured artifacts support only their recorded configurations.
No finite test establishes arbitrary scheduler liveness or all-context exactness.

## First GPU screening outcomes

The parent release signal arrived and all runs below used the exclusive lock.
These are screening results, not completed goal validation.

- H1/H2: `prep-v1.jsonl`, layer1, 3 random BF16 inputs, all24 custom
  configurations matched all Q/K/V/router/norm bytes. Best isolated hot wall
  latency was220.625us (64 rows,8 router rows, local RMS) versus249.750us
  compiled native. Full-model `prep-pilot-v1.jsonl` checked16 exact full-logit
  steps, then2 paired63-step runs: geometric ratio1.002261. Local latency
  did not translate into an end-to-end10% gain.
- H4/H8: `parts-v1.jsonl` uses real layer1 weights and synthetic inputs.
  Median hot wall times (us): front native292.334/exact268.083; down
  native212.959/exact16-row190.125; exact8-row194.500 and32-row193.916;
  whole branch native354.042/exact314.167; native head1306.709. All
  constituent output bytes matched. These host-inclusive times are not
  additive and cannot establish attention's share of decode.
- H9: `lean-pilot-v1.jsonl` passes16 full-logit steps and3 paired127-step
  generations. Binding-only change is approximately neutral. Setting512MiB
  lowers control throughput to about58tok/s;128MiB gives about61tok/s
  control but58–59tok/s lean. These are process-level screens, not a
  counterbalanced configuration proof. No settings change is promoted.
- H10 (new): overlap host submission of dependent token graphs using MLX
  async evaluation, preserving exact math/full logits. `pipeline.py` compares
  exact synchronous control, exact async, and stock async. No model-token
  speculation; fixed-length runs reject early EOS rather than hide extra work.
  Pilot running. Any gain belongs to host submission, not GPU fusion.

Parent ablation commit55caa0ef was cherry-picked as c7674ab0, retaining
its raw results and corrected interpretation in this worktree.

## Host-submission evidence and confirmation plan

`pipeline-full-v1.jsonl` completed 127 full-logit comparisons per variant on
136-token Python, 140-token Rust, and 1672-token long prompts. All bytes and
free-running generated tokens matched stock. Six whole-generation pairs per
prompt give exact-async / exact-sync geometric ratios 1.12799, 1.11329, and
1.11181. The paired bootstrap lower bounds are 1.10179, 1.10787, and 1.10024.
All samples, including large stalls, are retained. The first and third margins
are narrow; confirmation is required before closing the goal.

`pipeline_prepared.py` adds the 64-row / 8-router-row / local-RMS preparation
kernel. Its 63-step Python pilot passes full logits and adds about 1% over the
unchanged exact async path. `pipeline_ablation.py` then compares synchronous
separate evaluation, synchronous joint logits/argmax evaluation, exact async,
and prepared async, with 127-step gates and eight runs per prompt. This is an
exploratory four-way ordering protocol; it is not balanced for every pair.

The final `pipeline_confirmation.py` uses exactly two paths and alternates
AB/BA. Twelve pairs per prompt provide six of each order; its offline auditor
checks the counts explicitly. This separates the primary candidate comparison
from the exploratory four-way scheduling protocol. The selected candidate is
prepared async; the denominator remains the unchanged original exact fused
kernel path and established synchronous loop. Both use the same process,
weights, prompt, token count, default batching settings, and GPU lock.

This result concerns host submission and a small preparation fusion, not a
complete megakernel. The single-evaluation synchronous ablation is needed to
separate redundant evaluation-boundary cost from asynchronous host overlap.
Fixed-length tests reject early EOS; they do not validate arbitrary streaming
stop handling or sliding-cache wrap beyond the recorded contexts.

The initial pipeline pilot failed while recording a relative source path,
before decode. Its error log is retained as `pipeline-pilot-v1.log`; v2 fixes
that harness path. It is not counted as an arithmetic or performance result.

## Final outcome

`pipeline-confirmation-v1.jsonl` completed successfully: twelve AB/BA pairs per
prompt, six of each order, 127 full-logit byte checks per path per prompt, and
identical independently generated token sequences. Paired gains are 11.884%,
12.005%, and 12.027% for Python, Rust, and longer Python. Bootstrap lower bounds
are 11.122%, 11.794%, and 11.686%; separate log-ratio t intervals also clear 10%.
No final prompt regresses. One Python pair is only +9.35% and remains included.
Source hashes and unchanged control files were audited after the run.

The selected path combines H10 with the modest H1/H2 preparation change.
H4's tested down geometry remains unchanged, H9 settings were rejected, and
H3's parent ablation was retained. H5/H6/H7 and broader attention work remain
open: the task ends because the measured throughput target is reached, not
because those ideas were systematically exhausted. No complete-megakernel
speedup or true asynchronous Metal weight-transfer result is claimed.
