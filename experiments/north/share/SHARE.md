# Shareable result

Attach `north-m4-pro-results.png`. The source, raw evidence and reproduction
commands are in `north-m4-pro-research.zip`. Read `RESULTS.md` for the complete
scope and statistics. This packet supersedes the earlier failed-prototype
packet; the new decoders preserve the checked logits and generated tokens.

Draft post (262 characters):

> Testing Cohere’s megakernel ideas on M4 Pro: North Mini Code 4-bit, 48GB. Persistent Metal scheduling: ~6% faster full-model decode vs MLX. Fused branch: up to ~12%. All 1,424 checked decode steps matched logits bit-for-bit. 3 prompts × 3 runs. Repro + raw data.

Suggested follow-up:

> Scope: branch work in all 48 MoE layers; QKV, attention and routing still use MLX. Expert-level dependencies helped. Down-weight prefetch added cost here. Next-layer QKV/router prefetch—the area emphasized in Cohere’s batch-1 code—remains future work.

Source references:

- [Cohere article](https://cohere.com/blog/megakernels)
- [Pinned Cohere implementation](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c)
- [Pinned community 4-bit checkpoint](https://huggingface.co/mlx-community/North-Mini-Code-1.0-4bit/tree/dfbe084dfa26e241345af99ca32848f38fd865f9)

The bundle is local; no public repository link has been created and no post
has been sent. It contains no model weights or raw system traces.
