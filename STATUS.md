# Apple North megakernel status

The safe validation target is complete on this Apple M4 Pro:

- North Mini Code 4-bit runs all 49 transformer layers, K/V updates, final
  RMSNorm, and all 262,144 logits in one custom Metal dispatch per decode step.
- Two coding prompts pass 254 consecutive full-logit raw-byte comparisons with
  MLX. Complete state, cache, logit, and queue-completion checks pass with 1,
  20, 32, and 36 threadgroups.
- Six safe-path runs have a 53.423 tok/s median. Six stock MLX runs have a
  53.951 tok/s median. The safe path is 0.98% slower, or within 1% of stock.
- The strongest exact fused control reaches 59.952 tok/s, 10.89% above the safe
  megakernel.
- In a step-interleaved test, tuned task sizes beat coarse task sizes by 3.31%.
  Cache-warming approximations lose 4.79% with one stage and 24.85% with ten.

The original static all-grid scheduler reached 57.472 tok/s when it completed,
then deadlocked during an adversarial repeated-decode rerun. That result is
preserved as an unsafe finding and excluded from the headline.

Read [the full walkthrough](experiments/north/whole_pass/README.md) and verify
the [machine-readable summary](experiments/north/whole_pass/summary-v1.json).
Nothing has been posted or sent to Cohere.
