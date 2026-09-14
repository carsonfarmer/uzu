# Corrected verification status

The earlier self-audit failed: it verified unchanged kernel files and wrongly
inferred that the denominator's host loop was unchanged. The independent parent
review identified the extra `mx.eval(logits)`. The
[withdrawn audit](history/EXTRA_EVALUATION_AUDIT_WITHDRAWN.md) is preserved.

Current verified claims:

- `audit_control_source.py` matches the historical model and direct-argmax
  expressions by AST and checks the surrounding host branch for extra evals.
- `summarize_corrected.py` rehashes recorded sources, validates all expected
  correctness positions and token lists, rejects duplicate rows and inconsistent
  warmup labels, and checks all pair orders and positions.
- The corrected process completed 127-step gates for all five paths on all
  three prompts: 1,905 complete-array unequal-byte counts are zero. All 150
  measured generations match the independently generated stock continuation.
- Timed work includes dependent forwards, logits, argmax, readback, cache
  updates, loop administration, and the common final device drain.
- Python and Rust intervals clear +10% against both actual historical host and
  joint controls. The longer-context intervals do not.

The goal is still active. Async submission is standard pinned MLX-VLM behavior;
new preparation is only about 1% over unchanged fused async. General frontend
performance, early stopping, wider contexts, and GPU-side utilization remain
unverified. All old data, including the mismatched-control runs, are retained.
