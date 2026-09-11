# Apple megakernel experiments — preserved checkpoint, full validation active

North Mini Code 4-bit now has correct, faster full-model decode on this M4 Pro.
The fused Metal branch improves throughput by 6.6–11.7% and the persistent
scheduler by 6.0–6.6% versus original MLX in the final paired comparison.
Both beat the compiled-native control. Each passes 1,424 complete-logit checks;
the final checks compare raw bytes. Tokens and stopping behavior match.

Completed checkpoint: numerical repairs, full-model branch integration,
dependency and weight-prefetch ablations, GPU traces, Metal validation,
arbitrary-prompt verification, repeated timing, and a shareable reproduction
packet. This is not a complete single-kernel forward pass: QKV, attention,
routing, normalization, layer transitions and the output head still use the
pinned MLX implementation.

Active goal: extend persistent execution through the remaining decode graph,
including the next-layer QKV/router weight overlap emphasized by Cohere, then
compare it with original MLX and the strongest fused baseline under the same
bitwise-correctness gate. The goal is incomplete until that comparison exists.

Read [NORTH_RESULTS.md](NORTH_RESULTS.md), use
[the reproduction guide](experiments/north/quantized/README.md), or open
[the share packet](experiments/north/share/SHARE.md). The
[Cohere reference mapping](experiments/north/correctness/COHERE_REFERENCE.md)
tracks the mechanisms tested and what remains for a larger port.

No Uzu engine source changed. Earlier work is preserved under
experiments/north/history/; its failed-prototype completion claim is superseded
by this verified checkpoint. Nothing has been published or sent to Cohere.
