# Exact decode follow-up: corrected control, goal active

The previous completion claim was wrong. Its `exact_sync` loop added
`mx.eval(logits)` before reading argmax, while the strongest historical runner
directly calls `argmax.item()`. Kernel files were unchanged, but the host loop
was not. The [withdrawn report](history/EXTRA_EVALUATION_REPORT_WITHDRAWN.md),
[failed audit](history/EXTRA_EVALUATION_AUDIT_WITHDRAWN.md), and all old raw data
are preserved. Their approximately 12% gains cannot establish the requested
target against the actual strongest prior runner.

## Corrected evidence

`corrected_control.py` uses the actual historical expressions:

```python
logits = model(mx.array([[token]]), cache=cache).logits
token = int(mx.argmax(logits[0, -1]).item())
```

The [source audit](results/corrected-control-source-audit-v2.json) checks these
expressions against `full_layer/benchmark_decode.py` and the surrounding host
path for an extra eval. Every path includes a common final drain and total
decode wall time, including loop administration. The historical source summed
per-step intervals; that timing-wrapper distinction is explicit here.

A fresh process compares historical fused host, fused joint evaluation,
unchanged fused async, prepared async, and stock layers in the same async
research loop. Ten forward/reverse rotation rounds balance every pair five
times in each order, and every variant twice in every position. All 150 measured
generations contain 128 tokens, with 127 decode steps timed. All five paths
match the stock reference's full logit bytes at every checked step on each
prompt, and all generated token sequences match.

| Prompt | Input tokens | Prepared / actual historical host | 95% paired bootstrap interval |
|---|---:|---:|---:|
| Python | 136 | +10.70% | +10.43% to +11.01% |
| Rust | 140 | +10.92% | +10.55% to +11.37% |
| Longer Python | 1,672 | +10.67% | +9.84% to +11.56% |

The goal remains active: longer-context intervals cross +10%, including against
the additional joint-evaluation control. Every pair is retained; the interval
resamples whole generation pairs, never individual token steps.

[Raw runs](results/corrected-control-v1.jsonl),
[source-checked summary](results/corrected-control-v1-summary.json), and
[run log](results/corrected-control-v1.log) contain the complete record.

## Attribution and external relevance

Async submission is standard in the pinned MLX-VLM frontend:
`work/mlx-vlm/mlx_vlm/generate/ar.py:521` enqueues evaluation, and lines 546–552
enqueue the next dependent step before `y.item()`. The clean source checkout is
`cdc745ad8a32d162f6d8e9d08be256910d663ac2`. This improves our stripped research
runner; it does not establish a gain over the normal complete frontend or invent
async generation. Stock async here uses stock layers inside our research loop.

Prepared async improves over unchanged fused async by **1.13% Python, 1.20%
Rust, and 1.04% longer context** in the corrected run. That isolates the small
new preparation change under equal submission. The approximately 11% difference
from stock async also includes the previously existing fused branch work.

[Cohere's article](https://cohere.com/blog/megakernels) and pinned source
`67d0b9ca22ea3652796b715d1d1863459e0e2c3c` informed the constituent-first search
and investigation of waiting. These measurements do not reproduce its complete
H100 scheduler, TMA staging, or serving engine. No GPU-side timing, occupancy,
bandwidth, or physical overlap was measured. MLX-derived arithmetic and MIT
attribution remain in [NOTICE.md](../quantized/NOTICE.md).

## Work continuing and scope

The next bounded test removes the unused `routed` diagnostic output from the
exact down kernel, preserving arithmetic and geometry. Generation never consumes
this output. `compact_pilot.py` compares it with prepared async and the actual
historical host loop on the longer context. No result is claimed yet.
The [ledger](LEDGER.md) retains neutral/regressing tests and open hypotheses.

All GPU experiments use `gpu_run.py` and the exclusive research lock. Runtime,
reference, and model assets are reused read-only. Validation covers batch one,
128 fixed generated tokens, the listed contexts, and one M4 Pro with MLX 0.32.2.
Early EOS is rejected. General streaming stops, sliding-window wrap beyond the
recorded contexts, and other hardware remain outside scope. Desktop activity
is not excluded by the research lock.

```sh
/Users/carsonfarmer/Developer/Personal/uzu-metal-lab/work/north-venv/bin/python -B experiments/north/additional/gpu_run.py experiments/north/additional/corrected_control.py --output work/corrected-reproduction.jsonl --prompts short rust long --tokens 128 --runs 10
python3 experiments/north/additional/summarize_corrected.py work/corrected-reproduction.jsonl
python3 experiments/north/additional/audit_control_source.py
```
