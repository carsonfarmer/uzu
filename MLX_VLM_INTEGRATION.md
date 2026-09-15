# Actual MLX-VLM integration — September 14, 2026

The targeted North Metal kernels now run inside MLX-VLM 0.7.1 through its standard
`load`, `apply_chat_template`, and `generate` APIs. The user authorized MLX-VLM as
the engine target after the Uzu support gap and custom-runner scope error were
made explicit.

Against the untouched published package, the committed `prepared` mode improved
median decode throughput by **10.25–11.62%** across three coding prompts on the
M4 Pro. Complete generation latency fell by 4.74–9.24%; peak memory was essentially
unchanged. Eight measured generations per engine/prompt were spread across four
alternating process pairs. All 48 measured outputs matched stock exactly.

Full-checkpoint correctness also passed for five cases, including sliding-cache
rollover and natural EOS: 566 full-vocabulary logit arrays per mode matched stock
at every byte. Both optimized modes and the disabled control were checked, for
1,698 array comparisons. Six lifecycle/fallback integration tests and the existing
Cohere model test passed.

The source is in the separate repository
`/Users/carsonfarmer/Developer/Personal/mlx-vlm-north`, source-only branch
`cf/north-metal-fusion`, commit `12ed9fb5a6027ec3b271d27c31e2f1b6445b7b42`.
Full methodology, code, and raw results are in that repository's
`benchmarks/north_decode/README.md`, preserved on `cf/north-metal-fusion-results`.

This establishes an actual existing-engine improvement. It does not revive the
old 19–24% custom-runner headline, establish a Uzu-engine speedup, or complete the
Cohere whole-model megakernel goal. The next experiments must improve the preserved
MLX-VLM `prepared` control through the same normal generation API.

Both MLX-VLM branches, their exact commits, and all raw results are also backed up
in `experiments/north_mlx_vlm/mlx-vlm-integration.bundle` in this research repository.
The bundle requires the public MLX-VLM 0.7.1 base. From a clone of MLX-VLM containing
that release, restore both branches with:

```bash
git fetch /path/to/uzu-metal-lab/experiments/north_mlx_vlm/mlx-vlm-integration.bundle 'refs/heads/cf/*:refs/heads/cf/*'
```

The results branch is commit `c19d9b8`; the minimal library branch remains
`12ed9fb`. The library repository itself has not been pushed to a new GitHub fork.
