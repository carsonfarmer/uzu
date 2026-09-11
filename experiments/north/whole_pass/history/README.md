# Exploratory whole-pass results

These files preserve intermediate measurements and correctness checks made while
building the complete Metal decode path. They are excluded from the safe
headline result. Some point to earlier source hashes, and several cover only
layers 1–48 or leave the output head outside the dispatch.

`STATIC_SCHEDULER_FAILURE.md` records the crucial rejected result. The original
all-grid spin-barrier scheduler produced exact 127-step runs and reached 57.472
tok/s, then deadlocked after two exact steps when the same 32-group workload was
rerun. The successful and stalled raw files are retained here. This disproves a
safe deployment claim for that scheduler even though completed outputs were
exact.

Files whose names begin with `benchmark-invalid-` were rejected because paging
or clock drift made the samples unstable. They are retained so the exclusion is
auditable.

The safe, source-matched artifacts live one directory above. Run
`../summarize.py` from the repository root to verify their source hashes,
correctness assertions, token identity, and reported performance ratios.
