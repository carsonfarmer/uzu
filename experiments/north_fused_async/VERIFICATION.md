# Verification of the clean extraction

## Scope correction — standard-engine benefit unproven

**This branch has not demonstrated an improvement to Uzu or the normal MLX-VLM
generation path.** It contains a standalone MLX runner and custom kernels.
The combined 19–24% measurements partly restore async behavior already present
in standard MLX-VLM. Smaller gains versus stock layers with async are also
internal to this runner; they need validation through an existing engine's
ordinary entry points before being reported as a practical engine improvement.

The code and raw measurements are preserved. Earlier shareable-result framing
is withdrawn, and further optimization against this substitute baseline has
been stopped. See the [baseline and scope correction](https://github.com/carsonfarmer/uzu/blob/cf/decode-fusion/BASELINE_CORRECTION.md).

**Passed for the documented workload and pinned runtime.** This was a
self-review using the code-reviewer and adversarial-verifier workflows, plus
the executable correctness gates and a separate calculation from raw durations.

| Check | Evidence |
| --- | --- |
| Arithmetic preserved during extraction | Front/down Metal construction, `run_exact`, input validation and both layer forward methods have unchanged ASTs; only the unused split-K helper was removed from `kernel.h`. Preparation source differs only in its NOTICE link. See `provenance/extraction.json`. |
| Both stock controls are meaningful | Stock host uses the original layers and direct `argmax.item()`; stock async uses those same layers with the identical async loop used by fusion. No extra forced logits evaluation weakens the host control. |
| Same work is timed | 127 model calls per timed generation; complete logits and argmax; no speculative tokens; all readbacks and final `mx.synchronize()` occur before the timer stops. |
| Exact full-model output | 1,905 arrays of 262,144 BF16 logits matched byte for byte. All 150 measured continuations matched their prompt's stock reference. |
| Balanced order | All ten rounds are complete. Each variant occupies each position twice; an independent check found every pair runs in each order exactly five times. |
| Statistics use actual durations | Rates were recomputed as `127 / decode_seconds`; paired gains were independently recomputed as the geometric mean of baseline seconds divided by candidate seconds. The reported intervals use every round. |
| No source changed during measurement | Every source hash recorded in `matched-v1.jsonl` matches the final implementation. The nine actually imported MLX-VLM source files match the pinned manifest. |
| Inputs pinned | All model shards, tokenizer/config files and the reference dependency files passed SHA-256 verification with `prepare.py`. |
| Fresh installation works | A new Python 3.12 environment installed `requirements.txt`, passed `pip check`, and passed a 35-array/five-generation smoke test. No historical experiment modules are required. |
| Repository scope | New standalone MLX directory on upstream `7096cf32`, plus a work-cache ignore rule. Prior research branches are preserved separately. |

Two inherited blank lines at EOF in `inputs.py` and `prep.h` are retained to
preserve the measured source fingerprints. The default whitespace check flags
those formatting-only lines; the check with `blank-at-eof` disabled passes.

The front kernel assigns all 96 expert and 64 attention jobs exactly once with
160 threadgroups. The down kernel covers 2,048 output rows using 128 groups of
16 rows. Each output has one owner; the retained implementation uses no global
barrier or inter-threadgroup spin wait. BF16 rounding points, per-lane sums and
expert accumulation order are retained. Numerical equality is tested against
the original model, not merely against the previous custom implementation.

The async loop enqueues the next dependent graph before reading the previous
pending token. Its first token is the same untimed prefill result, its final
pending token is collected, and the GPU is drained before timing ends. This
avoids dropping the last decode step from the performance measurement.

Material limits: one device, one measured process, three prompts, fixed output
length, default MLX batching, and an interactive desktop. The last long-context
round contains a large rate shift; it is retained and reflected in uncertainty.
The optional preparation increment is not established for the longer prompt.
No claim is made about general model accuracy, crossing the rotating-cache
boundary, serving frontend throughput, or successful persistent megakernel
scheduling. The combined ~20% claim names the original synchronous research
runner as its baseline; the stock async comparison is reported alongside it.
