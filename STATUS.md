# Apple North research status — September 13, 2026

The additional runner goal is complete within its original short-workload scope:
corrected balanced measurements show +10.70% Python and +10.92% Rust against
the actual historical fused runner, with 95% paired intervals above +10%.
Full logits and greedy tokens match. Long-context correctness passes, but
performance precision is not established; all stalls and regressions are retained.
The earlier approximately 12% claim used an extra baseline logits evaluation
and remains explicitly withdrawn. Independent parent review confirms the
corrected result. This does not complete the full-megakernel objective.

The work targets the actual historical host-token loop with direct
`argmax.item()`, includes joint and fused-async controls, and separately
compares stock async. Async submission is standard pinned MLX-VLM behavior.
The new preparation fusion contributes about 1% under equal async submission.
See [the current report](experiments/north/additional/README.md).

The new [branch-mixing ablation](experiments/north/branch_mixing/README.md)
finds that nearly all of the fused gain survives with separate attention-output
and expert-front launches. In five uninterrupted-generation runs per prompt,
the mixed/separate ratio of medians is +0.094% for Python and +0.568% for Rust;
separate launches retain about 12.6% and 12.7% over the original MLX reference.
The particular alternating task-ID schedule brings no consistent benefit.

Those small incremental differences are not a reliable speedup claim. A large
Python timing outlier reverses the paired geometric estimate, and the
per-token alternating protocol gives different estimates. All samples are
retained. Full logits match byte-for-byte for all four custom variants on
127 positions per prompt in both final experiment processes.

MLX already permits independent Metal dispatches to overlap. The new explicit
serial-dependency control finds no repeatable gain from removing that dependency
across both prompts and timing protocols. Physical overlap was not measured.
We therefore cannot credit the earlier overall 11% improvement to this
particular Cohere-inspired scheduling mechanism.

Historical complete-queue checkpoint: 53.423 tok/s versus stock at 53.951 and
exact fused at 59.952. The queue is 10.89% behind fused. Its tested outputs match,
but that is not a full-model performance success or a general liveness proof.
An earlier static scheduler hung; a residency/barrier deadlock is suspected,
not conclusively diagnosed. See [NORTH_RESULTS.md](NORTH_RESULTS.md).

Current [five-variant source-checked summary](experiments/north/branch_mixing/summary-v2.json)
and [earlier four-variant evidence](experiments/north/branch_mixing/summary-v1.json)
include every measured run. No results have been posted or sent to Cohere.
