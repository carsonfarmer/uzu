# Additional ten-percent research

Base: `008d0282ad46cddc4dbf614a554301b7586dcb54`. Control: exact branch,
160 front workers and 16 down rows, unchanged. Goal is candidate/control >=1.10
in fresh matched end-to-end measurements. Historical 59.952 tok/s is context,
not a current denominator. No success or exhaustion has been established.

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
