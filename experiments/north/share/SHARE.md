# Shareable Apple megakernel result

Attach north-m4-pro-megakernel-results.png. The reproducible source, canonical
raw evidence, and report are in the project bundle. No model weights are
included.

## Suggested post

> Tested @Cohere’s megakernel idea on Apple: North Mini Code 4-bit, all 49 layers + KV + 262k logits in ONE Metal dispatch. Byte-exact for 254 decode steps. Safe queue: 53.42 tok/s vs MLX 53.95 (within 1%). Static hit 57.47—then deadlocked. H100 residency assumptions matter.

## Suggested follow-up

> Apple result: finer ready-task tiles made the safe queue 3.31% faster. A QKV/router cache-warming approximation lost 4.79% with 1 stage and 24.85% with 10. Best exact fused control: 59.95 tok/s, 10.89% ahead of the safe megakernel. Repro + raw samples in the fork.

## Prefetch caveat

> The prefetch test performs extra cache-warming reads inside an MLX custom Metal kernel. It does not reproduce Cohere’s asynchronous H100 TMA/shared-memory pipeline; a lower-level Metal implementation remains an open experiment.

## Links

- [Cohere article](https://cohere.com/blog/megakernels)
- [Pinned Cohere implementation](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c)
- [Pinned community 4-bit checkpoint](https://huggingface.co/mlx-community/North-Mini-Code-1.0-4bit/tree/dfbe084dfa26e241345af99ca32848f38fd865f9)

Nothing has been posted or sent to Cohere.
