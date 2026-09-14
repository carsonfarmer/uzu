# Additional exact decode throughput: confirmed on M4 Pro

The selected path exceeds the additional 10% end-to-end decode target on all
three tested prompts. It overlaps host submission using MLX async evaluation
and adds a small exact RMSNorm/QKV/router preparation kernel. The established
two-dispatch fused branch remains unchanged. Most of the gain comes from the
submission loop; preparation fusion contributes about another 1%.

The complete whole-model megakernel remains a separate, slower checkpoint.
These results establish an improved exact decode path, not a new speedup from
the complete megakernel or a reproduction of Cohere's H100 serving results.

## Final matched result

Apple M4 Pro, 48 GB, MLX 0.32.2, pinned affine 4-bit North Mini Code. Each prompt
has twelve control/candidate pairs, six AB and six BA, in one fresh process.
Each generation contains 128 tokens, with 127 decode steps timed. All pairs
are retained, including one Python pair below +10%.

| Prompt | Input tokens | Exact fused control, tok/s | Prepared async, tok/s | Paired gain | 95% paired bootstrap interval |
|---|---:|---:|---:|---:|---:|
| Python | 136 | 60.142 | 67.143 | **+11.88%** | +11.12% to +12.71% |
| Rust | 140 | 60.746 | 68.055 | **+12.00%** | +11.79% to +12.21% |
| Longer Python context | 1,672 | 55.210 | 61.891 | **+12.03%** | +11.69% to +12.38% |

Throughput columns are medians; gains are geometric means of the twelve paired
ratios. The historical 59.952 tok/s is not used as a denominator. The confidence
interval resamples complete run pairs, not individual correlated token steps.
A separate raw-seconds recomputation and log-ratio t interval also put every
lower bound above +10%. Both order strata exceed +10% on every prompt.

[Final raw runs](results/pipeline-confirmation-v1.jsonl),
[source-checked summary](results/pipeline-confirmation-v1-summary.json), and
[separate numerical audit](results/independent-audit.json) retain the evidence.
The final process exited successfully. No observations were removed or
classified as warmups after measurement.

## What was verified and timed

Before timing each prompt, the pinned stock MLX path independently generated
the reference continuation. Both the unchanged fused control and prepared
async path then generated their own continuation. All 262,144 logits at every
one of 127 decode positions were compared as raw bytes against stock. Both
paths passed on all three prompts: 762 complete-array comparisons in the final
process. The subsequent 72 measured generations produced identical token IDs.
Earlier full runs and the four-way ablation also retain their exactness gates.

Timed work includes model execution, embeddings, all layers, native cache
updates/growth, complete logits, argmax, host token readback, and a final device
synchronization. Model loading, tokenization, prefill, correctness comparisons,
and one warmup per path are excluded equally. No future model token is guessed.
Each next graph consumes the previous GPU argmax; the host reads the previous
token after submitting the next dependent graph. The final pending token and
GPU work are collected before stopping the timer.

All GPU processes used `gpu_run.py`: no MLX import or model load occurred before
the parent's release signal, and an exclusive `fcntl.flock` remained held for
each complete experiment. Other desktop activity is outside that research lock.
The runtime, model, and reference checkout were reused without modifying them.
The measured source hashes match, the pinned reference checkout is clean at
`cdc745ad8a32d162f6d8e9d08be256910d663ac2`, and the original `exact.py`, `patch.py`,
`kernel.h`, and `reference.py` have no diff from checkpoint `008d0282`.

## Where the gain comes from

The [four-way ablation](results/pipeline-ablation-full-v1-summary.json) separates
the established synchronous loop, synchronous joint evaluation of logits and
argmax, unchanged exact async execution, and prepared async execution.

On the stable Python and longer-context portions, joint evaluation contributes
about 0.8% and 0.4%; exact async improves over that joint-evaluation control by
about 9.8% and 9.9%. Preparation adds 1.24% and 1.03% over exact async. These are
wall-time contrasts, not GPU-side duration, occupancy, or bandwidth measurements.
The exploratory four-way order was not balanced for every pair; the final
two-path AB/BA confirmation supplies the primary candidate estimate.

Rust's exploratory ablation contained substantial stalls and has a wide
prepared/control interval of +6.87% to +13.78%. It does not independently
establish the threshold. Those observations remain in the raw data. The later
predeclared twelve-pair balanced confirmation passes with all samples retained.
The [earlier full async run](results/pipeline-full-v1-summary.json) also passed
all prompt gates and showed +12.80%, +11.33%, and +11.18% paired gains, but its
Python and long-context confidence margins were narrow. It is supporting
evidence for the host-loop change, not a replacement for the final confirmation.

The model already uses MLX's concurrent Metal encoder. The parent's
[branch-mixing ablation](../branch_mixing/README.md) found no consistent gain
from the particular alternating task-ID order. We therefore did not build
another large scheduler on that assumption.

## Other experiments and regressions

- Independent preparation fusion preserved all tested bytes and reduced hot
  constituent wall latency from 249.750 to 220.625 microseconds, but its
  synchronous decode pilot improved only 0.23%. Its benefit appears after
  changing submission, and remains small.
- The branch constituent screen favored the existing 16-row exact down kernel.
  Native head wall latency was about 1.307 ms in the isolated test. These
  host-inclusive measurements are not additive stage times.
- Removing unused down-weight bindings from the front was approximately neutral
  at default settings. Raising the batching threshold to 512 MiB reduced
  control throughput to about 58 tok/s. At 128 MiB, the lean variant was about
  58–59 tok/s against its approximately 61 tok/s control. No batching change is
  part of the selected result.
- The actual reported architecture is `applegpu_g16s`; the inspected MLX v0.32.2
  source selects 50 operations / 50 MiB from that suffix. Inferring 40 MiB from
  the product name alone was incorrect. Final runs leave both batching
  environment settings unset and record that fact.

The [experiment ledger](LEDGER.md) preserves hypotheses, failed approaches,
scope, and remaining options. Weight staging, larger scheduler changes, and
native cache aliasing remain open research avenues. This checkpoint closes the
throughput target; it does not claim those avenues were exhausted.

## Cohere and arithmetic attribution

[Cohere's megakernel article](https://cohere.com/blog/megakernels) guided the
search for waiting between ready work and the decision to compare competitive
constituents before extending fusion. The pinned checkout
`67d0b9ca22ea3652796b715d1d1863459e0e2c3c` was inspected, including its schedule
builder and producer weight-loading path. This Apple result uses host submission
and small independent Metal tasks; it does not implement Hopper TMA or the
complete Cohere serving scheduler.

The numerical RMSNorm, QKV, router, and affine projection ordering derives from
Apple MLX, with its MIT attribution retained in
[NOTICE.md](../quantized/NOTICE.md). The preparation kernel retains the router's
different reduction order and uses only local threadgroup barriers. No tolerant
comparison, changed token, or reassociated approximate arithmetic counts here.

## Limits and reproduction

These are batch-one, fixed-128-token research runs on one M4 Pro. They reject
early EOS rather than benchmark an unimplemented general streaming-stop path.
They cover the listed contexts and native cache growth, but not sliding-window
wrap at 4,096, arbitrary prompts, continuous batching, or general scheduler
liveness. Desktop stalls show that a gain cannot be promised for every run;
the final Python minimum pair was +9.35%, and it remains included.

From this worktree, with the pinned runtime and asset symlinks in place:

```sh
/Users/carsonfarmer/Developer/Personal/uzu-metal-lab/work/north-venv/bin/python -B experiments/north/additional/gpu_run.py experiments/north/additional/pipeline_confirmation.py --output work/reproduce-confirmation.jsonl --prompts short rust long --tokens 128 --runs 12
python3 experiments/north/additional/summarize_confirmation.py work/reproduce-confirmation.jsonl
```

`pipeline_confirmation.py` is the selected executable research path;
`layers.py`, `prep.py`, and `prep.h` contain its preparation change. The other
pipeline scripts preserve the distinct experimental protocols and their source
hashes. See [the verification audit](VERIFICATION.md) for claim-level checks.
