# Apple North research status — September 13, 2026

> **Completion claim withdrawn after control audit.** The earlier final control
> inserted `mx.eval(logits)` before reading argmax. The strongest historical
> host-token loop directly reads `mx.argmax(logits[0, -1]).item()` and has no
> such extra evaluation. Statements below calling that loop unchanged, or
> declaring the target achieved, are superseded. The old data are retained as
> measurements against the extra-evaluation control. Corrected balanced
> experiments are in progress; the throughput goal is active again.

The additional 10% decode-throughput target is confirmed on
`cf/north-additional-ten-percent`: **+11.88% Python, +12.00% Rust, and +12.03%
at 1,672 prompt tokens**, versus fresh matched measurements of the unchanged
synchronous exact fused control. Twelve AB/BA pairs per prompt, full-logit
bytewise checks for all 127 decode steps, and identical free-running tokens
support the result. The gain mainly comes from host submission overlap, plus
about 1% from preparation fusion. See the
[report, raw runs, and verification](experiments/north/additional/README.md).

The complete megakernel remains slower than the fused control. Its historical
results are preserved, and the new host-loop gain must not be attributed to
that scheduler. The measured candidate is a fixed-length research path; general
streaming-stop behavior and wider contexts remain outside its validation.
All GPU experiments used the shared exclusive lock.

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
