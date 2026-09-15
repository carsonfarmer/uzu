# CPU-only engine integration inventory — September 14, 2026

The user rejected further optimization of the standalone custom MLX runner. GPU research stopped after the already-running dispatch-cut pilot finished exit0. No further GPU process was launched. This inventory is not a verified integration or an engine speedup.

## Stock MLX-VLM

Pinned source in `work/mlx-vlm` already contains `cohere2_moe`, including its outer Model, input embedding interface, LanguageModelOutput, and ordinary KV/rotating cache creation. The actual autoregressive loop is `mlx_vlm/generate/ar.py:181` (generate_step), reached by public streaming dispatch in `mlx_vlm/generate/dispatch.py:800`. It already calls mx.async_eval at lines521 and546 and queues the next step before yielding tokens. Async adoption in a custom loop is not a new stock-engine optimization.

The least invasive integration would preserve public model loading, outer Model, generation, sampling/log-probabilities, stopping, tokenizer/processor, and caches. Replace only eligible single-token computations in `mlx_vlm/models/cohere2_moe/language.py:CohereMoEDecoderLayer.__call__`, with stock fallback for prefill and unsupported shapes, quantization, cache or mask modes. Existing experimental layer wrappers are potential kernel sources, not evidence this public path was integrated or measured. Cross-layer preparation needs lifecycle handling on every forward. The whole-pass prototype owns packed caches and has short-context limits, requiring substantially more adaptation.

Acceptance needs the actual public generation path on both sides with identical settings and existing async behavior; full-logit and token gates; prefill, rotating-cache boundaries and fallback checks. No new GPU validation occurred after the stop.

## Uzu

Source inspection does not establish out-of-box North support. There are concrete architectural gaps: `crates/uzu-engine/src/encodable_block/transformer_layer.rs:200` encodes mixer then pre-MLP normalization and MLP in sequence. North's `cohere2_moe/language.py:175` computes both branches from the same normalized input and returns attention + MLP + input. Uzu transformer config has no corresponding parallel-branch option. Its routing enum in `crates/uzu-engine/src/config/mlp/routing_function/mod.rs` exposes SoftmaxRouting only; the pinned North config specifies sigmoid expert selection and norm_topk_prob=false.

A native Uzu implementation needs architecture/config and weight conversion, parallel residual branch encoding, and sigmoid top-k without normalization. It must also audit affine4-bit group64 weights, mixed full/sliding attention, RoPE, prefix dense layer, tied readout and tokenizer handling. This is an audit list, not a claim that all these facilities are absent. Metal kernels need Uzu buffer, pipeline, encoder and resource-lifetime integration. A Rust host for the MLX experiment is not a Uzu engine benchmark.

Existing results are preserved as kernel research. No Uzu or stock-generation speedup can be inferred from them.

## Corrected acceptance and hold

Read and accepted `BASELINE_CORRECTION.md` from commit16da4041 on cf/decode-fusion. The prior custom-runner objective is superseded, not achieved. Uzu is the existing project context; stock MLX-VLM is a correctness/reference capability, not a substitute for native integration. Further implementation and performance work is on hold pending the corrected engine plan. Native support must be verified before a same-engine enabled/disabled optimization comparison through normal loading, caches, generation and sampling. The correction also identifies Uzu MoE bias requirements and model-data-type expert loading as concrete conversion/quantized-expert prerequisites to address. No new GPU work or performance claim is authorized by this inventory.
