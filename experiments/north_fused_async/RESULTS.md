# Matched fusion + async results — 2026-09-14

**The minimal targeted fusion + async path improved decode throughput by 22.4% on Python, 22.7% on Rust and 19.4% on the longer prompt, compared with our original synchronous research runner.** Adding the optional preparation fusion gave 24.4%, 24.5% and 22.0%, respectively. These are fresh, paired measurements; no historical gains were multiplied together.

Against the stock layers using the same async submission, core fusion improved throughput by 8.9%, 9.8% and 7.1%. Including preparation fusion raised those gains to 10.7%, 11.4% and 9.5%. The combined ~20% result is useful, and its baseline must be named when reporting it.

## Throughput

M4 Pro, 20 GPU cores, 48 GB, affine-W4 North Mini Code with BF16 activations, MLX 0.32.2. Values below are median decode tokens/second over ten measured runs per variant and prompt.

| Prompt | Input tokens | Stock sync | Stock async | Fusion sync | Fusion async | + preparation fusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python | 136 | 45.70 | 51.44 | 51.10 | 56.21 | 56.86 |
| Rust | 140 | 45.50 | 50.88 | 50.45 | 55.81 | 56.78 |
| Long Python | 1672 | 41.65 | 46.37 | 45.01 | 50.22 | 51.10 |

Paired geometric mean gains and 95% bootstrap intervals below use the ten complete rounds. These ratios need not equal ratios of the medians above.

| Prompt | Core vs stock sync | Core vs stock async | + prep vs stock sync | + prep vs stock async |
| --- | ---: | ---: | ---: | ---: |
| Python | +22.39% [21.60, 23.07] | +8.89% [8.20, 9.58] | +24.37% [23.86, 24.87] | +10.65% [10.20, 11.18] |
| Rust | +22.71% [22.16, 23.19] | +9.83% [9.10, 10.50] | +24.47% [23.87, 25.02] | +11.40% [10.50, 12.30] |
| Long Python | +19.37% [16.12, 21.45] | +7.12% [4.39, 8.76] | +22.02% [20.58, 23.10] | +9.50% [7.65, 10.81] |

## Correctness and method

- **1,905 full-logit array comparisons, zero differing bytes.** Each array has shape `(1, 1, 262144)` and BF16 dtype. Every one of the five variants is checked at all 127 decode steps for each of the three prompts.
- **150 measured generations with identical token sequences** to the stock reference within each prompt. Fifteen warmup generations are recorded separately and excluded from throughput aggregation.
- Ten counterbalanced rounds: every variant occupies each position twice, and every pair runs in each order five times. All measured samples are retained.
- Each generation emits 128 tokens. Untimed prefill produces the first token; 127 full-model decode steps are timed, including the vocabulary head, argmax, all token readbacks and final GPU drain. Early EOS causes failure.
- The synchronous stock control directly reads `argmax.item()` without a separate logits evaluation. Both async variants and the stock async control use the same dependent-graph submission loop.
- A separate clean Python environment installed only the declared requirements, passed `pip check`, and passed another 35 full-logit comparisons plus five matching smoke generations. It reused verified model/source files, with no imports from historical experiment modules.

## Uncertainty

This was one process on an interactive desktop with other CPU activity, including setup and CPU-only research. The cooperating megakernel task did not run GPU experiments concurrently. Rates increased sharply in the last three variants of the final long-context round; a slower preparation run also occurred earlier. Both remain in the data. The long-prompt interval is consequently wider. We have not established the cause of the rate changes.

The optional preparation increment over core fusion+async was +1.62% on Python and +1.44% on Rust. Its long-prompt estimate was +2.22%, with an interval spanning -0.19% to +5.56%; that incremental long-context benefit remains uncertain.

The intervals describe these rounds, not a guarantee across machines or workloads. This is fixed-length batch-one decode, excluding prefill; it is not an end-to-end benchmark of the MLX-VLM serving frontend. The longest prompt plus generation does not cross the 4,096-token rotating-cache boundary. Exactness is established for the tested runs.

## What is preserved

The branch contains the useful fusion kernels, their layer wrappers, the async runner, optional preparation fusion, and a self-contained reproduction harness. It starts at Uzu `7096cf32`, with no megakernel research commits in its history. The only addition outside this directory is the local work-cache ignore rule. The optimized kernels retain their previously validated arithmetic; extraction checks are recorded in [provenance/extraction.json](provenance/extraction.json).

[Cohere’s article](https://cohere.com/blog/megakernels) motivated the investigation. These gains come from targeted fusion and a standard async submission pattern; [the pinned MLX-VLM loop already uses that pattern](https://github.com/Blaizzy/mlx-vlm/blob/cdc745ad8a32d162f6d8e9d08be256910d663ac2/mlx_vlm/generate/ar.py#L546). They do not establish the benefit of a persistent task scheduler or future-layer weight prefetch. The full megakernel investigation continues separately on `cf/north-megakernel-fullest`, using the strongest exact fusion+prep+async path as its control.

## Reproduction and evidence

- [Setup and commands](README.md)
- [All raw rounds and correctness checks](results/matched-v1.jsonl)
- [Computed summary, intervals and every paired gain](results/matched-v1-summary.json)
- [Clean-install smoke run and package versions](results/clean-install-smoke.jsonl)
- [Verified source and model files](results/inputs-verification.txt)
- [Host conditions](results/host.json)
- [Review and validation](VERIFICATION.md)

## Short shareable finding

> North Mini Code W4 on an M4 Pro: targeted Metal fusion + async submission made our decode loop ~20–23% faster than its original synchronous version, with bit-identical logits in our tests. Core fusion still adds ~7–10% over the async baseline. Code and raw runs included.
