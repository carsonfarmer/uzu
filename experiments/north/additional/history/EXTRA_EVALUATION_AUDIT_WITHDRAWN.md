# Adversarial verification

> **Completion claim withdrawn after control audit.** The earlier final control
> inserted `mx.eval(logits)` before reading argmax. The strongest historical
> host-token loop directly reads `mx.argmax(logits[0, -1]).item()` and has no
> such extra evaluation. Statements below calling that loop unchanged, or
> declaring the target achieved, are superseded. The old data are retained as
> measurements against the extra-evaluation control. Corrected balanced
> experiments are in progress; the throughput goal is active again.

Status: failed control audit; previous pass withdrawn. This is a self-audit using the
adversarial-verifier skill, including a separate computation from raw seconds;
it is not an independent reviewer or an additional GPU experiment.

The prior target-completion claim is unsupported because its denominator added a wait.
No claim of a complete-megakernel gain, arbitrary-context correctness, or a
guaranteed gain on every run is supported.

| Question | Direct investigation | Finding |
|---|---|---|
| Was the denominator weakened or replaced by history? | Diffed exact.py, patch.py, kernel.h, and reference.py against 008d0282; traced control construction and timing. | Kernel files have no diffs, but that does not validate the host loop. The final control added a logits eval absent from the historical host-token loop. This row previously conflated unchanged kernels with an unchanged denominator; that conclusion was false. |
| Can async omit unfinished GPU work? | Traced pending-token collection and final mx.synchronize before elapsed time is read. | All 127 dependent forwards and the final argmax complete inside timing. No extra forward is hidden outside it. |
| Is the result only matching argmax? | Traced reference generation, retained logits, uint8 views, and assertion sites; counted final raw rows. | 127 complete logit-array byte checks for both paths on each of three prompts; all 762 unequal-byte counts are zero. |
| Are measured generations teacher-forced? | Traced each generate call and token dependencies. | Each timing generation feeds its own previous token. All 72 measured token lists match the independently generated reference through assertions. |
| Were samples discarded or orders unbalanced? | Counted raw non-warmup rows and their sequence, rather than trusting summary labels. | Twelve pairs per prompt, six AB and six BA. One Python pair is +9.35% and is retained. Earlier noisy ablation rows remain too. |
| Does the summary process ratios correctly? | Recomputed as control_seconds/candidate_seconds, independently of recorded tok/s; computed log-ratio t intervals and order-stratified means. | Same geometric gains; all alternative lower bounds exceed +10%; both order strata exceed +10%. See independent-audit.json. |
| Are sources reproducible? | Rehashed every recorded source; checked clean pinned reference and MLX source tag v0.32.2. | Recorded hashes match. Large model weights were reused from the existing pinned assets, not independently rehashed in full. |
| Does small fusion explain the whole gain? | Inspected four-way controls and all raw results. | No. Stable cases attribute about 1% to preparation and most of the remaining gain to the changed host submission path. Rust exploratory intervals are wide. |
| Could deadlocking global waits explain successful tests? | Read prep.h/prep.py task ownership and barriers. | New preparation tasks have disjoint output rows and only local barriers. No global wait or persistent scheduler was added. This is not an audit of the old whole-pass scheduler. |

Concerns retained in the report:

- The primary workload is fixed length. Early EOS, arbitrary streaming stop
  handling, sliding-window wrap, and broader contexts were not validated.
- Bootstrap and t intervals summarize these process runs. They do not eliminate
  desktop interference or prove universal performance.
- The four-way exploratory ordering was not balanced for every pair. Primary
  promotion uses the later, explicitly balanced two-path protocol.
- Source/configuration provenance relies on the preexisting pinned model assets;
  no claim of a fresh full-weight cryptographic audit is made.
- This is primarily a host-loop optimization. Treating its full gain as a
  megakernel scheduling or GPU-fusion gain would be false attribution.
