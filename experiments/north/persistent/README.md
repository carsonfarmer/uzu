# Persistent branch experiment

`branch.py` implements the first ready-work scheduler. `fast.py` keeps the same
work graph and changes dependency acquisition, readiness caching and activation
staging; this is the selected full-model persistent route. `prefetch.py` tests
staging down weights before or after the activation dependency. All operate on
the original packed affine-W4 weights and preserve the reference arithmetic.

Use `check.py` for intermediate-byte, exactly-once and progress checks. The
current ablation evidence is `ablation-*-v5.json`: each has two warmups, 29 timed
pairs and alternating order within adjacent pairs. The `v4` ablation files are
superseded because combining rotation and reversal accidentally fixed the
order when only two variants were selected. They are not used in the report.

`check-fast-v4.json` includes the one-to-256-worker correctness stress.
`validation-*-v4.json` and `.log` record Metal API and GPU validation and are
excluded from timing estimates. Full-model gates and measurements live in
`../correctness/results/`; source snapshots there identify their exact versions.

The timed `fast` path disables diagnostic visit counters but retains error and
progress checks. The branch ablations enable the same diagnostics on both
variants. Workspace initialization and final joining remain separate dispatches
and are included in timing. The entire decoder is not inside this kernel.

See `../correctness/COHERE_REFERENCE.md` for the source mapping, memory protocol
and the distinction between this MoE-down staging test and Cohere's next-layer
QKV/router prefetch.
