# Evidence classification

Raw artifacts are immutable records of the indicated protocols. A passing
numeric field in an older summary does not override a withdrawn interpretation.

| Artifacts | Meaning and status |
|---|---|
| `corrected-control-v1.*` | Actual historical direct-argmax host loop, joint control, fused async, prepared async, stock async. Ten balanced rounds on all three prompts. Python/Rust conservative gates pass; long-context interval crosses +10%. |
| `precision-long-v1.*` | Predeclared 24-round long-context follow-up with unchanged prepared path and both corrected host controls. Completed; all exactness checks pass, with large timing stalls. All pairs pooled in `pooled-long-v1.json`; no precise long-context +10% claim. |
| `corrected-control-source-audit*.json`, `precision-long-source-audit.json` | AST/source checks for the actual historical host expressions. |
| `scoped-goal-v1.json`, `parent-corrected-control-review.json` | Passing original short-workload gate and independent parent recomputation/review. Rust corroborates; long uncertainty explicit. |
| `async-batching512-short-v1.*` | Nondefault batching screen. Host controls slowed; ineligible as target evidence. |
| `compact-long-pilot-v1.*` | Different candidate omitting an unused diagnostic output. Exactness passes; timing is highly variable. Not promoted or pooled with unchanged prepared results. |
| `pipeline-confirmation-v1.*`, `pipeline-full-v1.*`, `pipeline-ablation-full-v1.*`, `pipeline-prepared-pilot-v1.*`, `pipeline-pilot-v2.*`, `independent-audit.json` | Earlier extra-evaluation host controls. Their bitwise checks remain observations, but their completion claim and unchanged-denominator audit were withdrawn. They cannot supply the requested historical-control performance denominator. |
| `prep-pilot-v1.*`, `lean-pilot-v1.*`, `batching*-pilot-v1.*` | Early screening with the extra-evaluation control. No final target claim. |
| `prep-v1.*`, `parts-v1.*` | Synthetic-input constituent screens with real weights; wall latency, not GPU duration or end-to-end throughput. |
| `pipeline-pilot-v1.log`, empty `pipeline-pilot-v1.jsonl` | Harness provenance-path failure before decode; no performance result. |
| `pipeline-host-*.txt` | Host observations during earlier work; not a causal explanation of timing stalls. |

See `../history/` for withdrawn narratives and `../README.md` for current status.
