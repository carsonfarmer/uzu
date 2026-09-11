# North Mini Code on Apple Silicon: feasibility assessment

**Follow-up:** the model has since been downloaded, hash-verified and run. See NORTH_RESULTS.md for measured baseline, coding checks and real-layer Metal experiments. The assessment below records the initial source-based findings.

Assessed 2026-09-10 for this M4 Pro / 48 GB machine. **Yes: it is a strong eventual model for the megakernel experiment, and a quantized Apple baseline is available. It is not currently loadable as a correct North Mini Code model in this Uzu checkout.** No model weights were downloaded and no engine files were changed during this assessment.

## What is available

Cohere's own megakernel is specifically built for North Mini Code. The local research checkout is pinned to `67d0b9ca22ea3652796b715d1d1863459e0e2c3c`; it targets H100 / CUDA, not Metal. Its task graph exploits independent attention and MoE branches. This makes North Mini Code more directly relevant to that experiment than the sequential Qwen MLP alone. [Pinned Cohere README](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/README.md)

The official release is text-only, Apache-2.0, approximately 30B total / 3B active parameters. Its advertised context is 256K with 64K maximum output; the config's larger `max_position_embeddings` should not be treated as a validated context promise. [Official model card](https://huggingface.co/CohereLabs/North-Mini-Code-1.0)

Exact revisions and sums of published safetensors file sizes, read from the Hugging Face API (not measured resident memory):

| Repository | Revision | Weight file bytes / GiB |
| --- | --- | --- |
| [CohereLabs/North-Mini-Code-1.0-fp8](https://huggingface.co/CohereLabs/North-Mini-Code-1.0-fp8/tree/736dde3c255d7726551e6e12af59967f08a20eb6) | `736dde3c255d7726551e6e12af59967f08a20eb6` | 32,026,285,128 / 29.83 GiB |
| [CohereLabs/North-Mini-Code-1.0-w4a16](https://huggingface.co/CohereLabs/North-Mini-Code-1.0-w4a16/tree/1e55f4aa327aba4c0b7a1da0d0f24626d3af5c90) | `1e55f4aa327aba4c0b7a1da0d0f24626d3af5c90` | 19,347,587,704 / 18.02 GiB |
| [CohereLabs/North-Mini-Code-1.0](https://huggingface.co/CohereLabs/North-Mini-Code-1.0/tree/d11e61a842617a22dc328552fa5bb86231ee4f37) | `d11e61a842617a22dc328552fa5bb86231ee4f37` | 60,970,901,384 / 56.78 GiB |
| [mlx-community/North-Mini-Code-1.0-4bit](https://huggingface.co/mlx-community/North-Mini-Code-1.0-4bit/tree/dfbe084dfa26e241345af99ca32848f38fd865f9) | `dfbe084dfa26e241345af99ca32848f38fd865f9` | 18,495,219,875 / 17.23 GiB |
| [mlx-community/North-Mini-Code-1.0-5bit](https://huggingface.co/mlx-community/North-Mini-Code-1.0-5bit/tree/f7338bfdf56a1242b07d0a1770864161375e1422) | `f7338bfdf56a1242b07d0a1770864161375e1422` | 22,188,566,715 / 20.66 GiB |

Official FP8 is compressed-tensors float8 with selected tensors left BF16. Official `w4a16` is **NVFP4**, `nvfp4-pack-quantized`, group size 16 and float8 scales; it is not an interchangeable integer-W4 checkpoint. The MLX community 4-bit package uses affine group-64 weights, and its card identifies mlx-vlm 0.6.2 as its converter. These are distinct quantizations and require separate numerical baselines. [FP8 config](https://huggingface.co/CohereLabs/North-Mini-Code-1.0-fp8/blob/736dde3c255d7726551e6e12af59967f08a20eb6/config.json), [W4A16 config](https://huggingface.co/CohereLabs/North-Mini-Code-1.0-w4a16/blob/1e55f4aa327aba4c0b7a1da0d0f24626d3af5c90/config.json), [MLX config](https://huggingface.co/mlx-community/North-Mini-Code-1.0-4bit/blob/dfbe084dfa26e241345af99ca32848f38fd865f9/config.json)

## Architecture and the Uzu gaps

The pinned official config specifies `Cohere2MoeForCausalLM` / `cohere2_moe`: 49 layers, hidden size 2048, 32 query heads, 4 KV heads, head dimension 128, 128 experts with top-8 selection, and no shared experts. The first layer is dense with intermediate width 3072; expert width is 768. Its routing is sigmoid with `norm_topk_prob=false`. Attention consists of 13 global layers and 36 sliding layers, with window 4096. Parallel attention/MLP residual blocks are enabled. [Pinned architecture config](https://huggingface.co/CohereLabs/North-Mini-Code-1.0/blob/d11e61a842617a22dc328552fa5bb86231ee4f37/config.json)

Against Uzu `7096cf32e70400c377b7341f75949d52c36f8f4d`:

- `crates/uzu-engine/src/encodable_block/transformer_layer.rs:194` feeds attention output through residual/pre-MLP normalization into the MLP. North requires both branches to consume the same normalized layer input before their outputs join the residual. A new block execution path is needed; conversion alone cannot preserve this graph.
- `config/mlp/routing_function/mod.rs` exposes only SoftmaxRouting. North needs sigmoid scores, top-8 selection, and the exact unnormalized selected-score semantics. Matching expert indices alone is insufficient.
- `encodable_block/mlp/moe/mod.rs:109` requires router/up/down biases, whereas North's checkpoint does not supply these. Zero-bias materialization could bridge that specific issue, but should not hide the remaining gaps.
- The same MoE loader reads full-precision expert arrays (`w13`, `w2`) in model data type. Uzu's general dense MLX/int weight formats do not imply quantized MoE support. Expanding all experts to BF16 defeats the 48 GB memory target.
- Uzu already exposes per-layer sliding-window and optional RoPE configuration, RMS normalization, SwiGLU, and sufficient router dimension limits (2048/128/8 are within bounds). These are reusable pieces, not evidence that the whole architecture is supported.
- HF sharded tensor names, expert packing, tied embedding/output behavior, and Cohere chat/thinking/tool formatting must also be converted or implemented and checked against reference outputs. The existing Qwen prompt wrapper is not a North template.

## Practical Apple baseline

Use the pinned MLX community 4-bit package with **mlx-vlm**, initially at 2K–8K context and batch one. The inspected implementation at `cdc745ad8a32d162f6d8e9d08be256910d663ac2` has an explicit `cohere2_moe` model, sigmoid routing, parallel branches, and rotating KV caches for sliding layers. The model rejects image inputs: the conversion card's image example and image-text tag are generic boilerplate, not model capability. Do not blindly copy that example. [MLX language implementation](https://github.com/Blaizzy/mlx-vlm/blob/cdc745ad8a32d162f6d8e9d08be256910d663ac2/mlx_vlm/models/cohere2_moe/language.py), [text-only wrapper](https://github.com/Blaizzy/mlx-vlm/blob/cdc745ad8a32d162f6d8e9d08be256910d663ac2/mlx_vlm/models/cohere2_moe/cohere2_moe.py)

A source-verified alternative is llama.cpp's Cohere2MoE implementation, including Metal as that project's Apple backend. That can supply an independent GGUF baseline, but adds another quantization and runtime; start with one baseline. Neither runtime has been installed or executed as part of this assessment. [llama.cpp model implementation](https://github.com/ggml-org/llama.cpp/blob/c61b98b875eaa5e654a3f5c73b34c310d2c6ab4c/src/models/cohere2moe.cpp)

## Memory: facts versus estimates

The file-size table is factual metadata. The following are planning estimates, not measurements:

- BF16 weights alone are 56.78 GiB, exceeding physical memory; not a useful all-resident performance baseline on this 48 GB machine.
- MLX 4-bit weights are 17.23 GiB on disk. This plausibly leaves room for model runtime, KV cache, scratch space and the OS at short contexts; actual Metal working-set and peak loading memory must be measured. All experts must be stored even though only eight execute per token.
- For batch one with BF16 KV, per cached token per layer is `2 × 4 × 128 × 2 = 2048` bytes. With 13 global layers and a correctly bounded 4096-token cache for each of 36 sliding layers, estimated KV is `2048 × (13 × T + 36 × min(T,4096))` bytes. This gives about **0.38 GiB at 4K**, **0.48 GiB at 8K**, **1.09 GiB at 32K**, and **6.78 GiB at 256K**. The formula excludes allocator overhead, alignment, temporary attention storage and prefill scratch. A runtime that stores full-length KV for every layer would use 24.5 GiB at 256K instead.
- Official FP8 files at 29.83 GiB might fit with careful runtime support, but that does not establish efficient native execution or a drop-in Uzu loader. Official NVFP4 files at 18.02 GiB fit as storage but need a distinct decoder/kernel format. Avoid a large conversion effort before the scheduler experiments establish merit.

## Recommended sequence

1. Keep the current Metal dependency/task prototype as the first executable milestone. Add a North-shaped synthetic workload: hidden 2048, expert width 768, top-8 experts, parallel attention-like and MoE branches followed by a residual join. This directly tests the scheduling opportunity without requiring a model port.
2. Establish an independent real North baseline using the pinned MLX 4-bit package. Pin runtime revision, preserve the model chat template, verify plain text and thinking/tool parsing, and record time-to-first-token, decode time, GPU/peak memory, and actual prompt/generated token counts at several short contexts.
3. Extract one dense layer and one routed MoE layer, with real inputs, router results and reference outputs, from that baseline. Compare ordinary Metal dispatch and dependency scheduling at identical arithmetic/quantization. Include the separate quantization error versus BF16 where feasible; never label a quantization change as a scheduler speedup.
4. Only then choose between adding North block/routing/quantized-MoE support to Uzu and integrating a Metal custom-kernel experiment in an MLX harness. The latter avoids coupling scheduler validation to a full Uzu architecture port; the former is larger but gives an eventual single-engine comparison.

**Assessment outcome:** feasible on this Apple device in a quantized reference runtime, highly relevant to Cohere's scheduling ideas, and suitable as the next real-model target after the task scheduler works. Full North-in-Uzu inference is a separate implementation milestone. No local North generation, throughput, correctness, or memory result is claimed yet.

Evidence snapshots (metadata/configs only) are in ignored `work/north-mini-assessment/`.
