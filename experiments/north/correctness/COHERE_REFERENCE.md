# Implementation reference and experiment contract

Active research checkpoint, 2026-09-11. The earlier failed decoder was an
unfinished experiment. Its timing does not test the complete Cohere design.

## Pinned references

- [Cohere article](https://cohere.com/blog/megakernels), particularly “Where the speedup comes from” and the porting recipe.
- [Cohere source](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c), commit `67d0b9ca22ea3652796b715d1d1863459e0e2c3c`.
- [Apple Metal language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), downloaded 2026-09-11, document dated 2026-06-04, sections 4.8 and 6.16.
- [MLX affine quantized kernel](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/quantized.h) and [GEMV dispatch](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/matmul.cpp), version 0.32.2.

## Mechanism mapping

| Cohere mechanism | Concrete source | Apple implementation / remaining work |
|---|---|---|
| Competitive underlying matrix kernels | Article’s porting recipe; tiled operations in `megakernel.cuh` | Packed W4 up/gate and complete-K down dot preserve MLX’s accumulation and BF16 rounding; attention projection uses the native per-lane order and shuffle tree. |
| Combine ready independent work | `schedule.py::_round_robin_waves` (1244 onward), persistent per-SM instruction lists | Correct two-dispatch anchor mixes MoE up/gate and attention projection tasks. Full QKV, attention and router remain outside this branch. |
| Down projection depends only on its expert | Article’s “Drop false dependencies”; dynamic MoE drain tasks and `wait_input_bars` | Experimental ready-work scheduler publishes each hidden tile and starts down tasks once that expert’s 12 tiles are ready. Compare against waiting for all experts. |
| Memory publication, not merely atomic counters | `megakernel.cuh::wait_cross_sm` / `arrive_cross_sm` around 561–602 | Metal 3.2 `coherent(device)` buffers, device-scoped sequential fences, and a producer threadgroup barrier. Relaxed atomics alone are insufficient. |
| Placement aware of producers | `schedule.py::_dependency_affinity_round_robin_waves` (1287 onward) | Current prototype uses ready-job selection. It does not yet implement Cohere’s complete host affinity schedule. |
| Load immutable weights before activation dependency | `megakernel.cuh` around 975–1068: `load_weight`, then dependency wait, then `load_activation` | `persistent/prefetch.py` stages down weights before the expert is ready and executes available producer tasks while waiting. The control stages the same weights afterward. Next-layer QKV/router overlap remains unimplemented. |
| Prefetch depth chosen by workload | `schedule.py::default_nmc_tiling_for_bs` around 594–620 | Batch-1 MoE uses zero explicit prefetch stages in the pinned code; QKV/router use ten. Deep MoE prefetch should be a measured experiment, not assumed beneficial. |
| Entire decode forward pass resident | Article; complete opcode stream in Cohere source | Still outside current branch scope. Layer transitions, normalization, QKV, attention, routing, dense layer and LM head need coverage before claiming a complete port. |

## Correctness gate

Bitwise equality to the pinned MLX reference is the current gate. The original
prototype changed accumulation order in the attention and down projections.
Local component checks identified those errors; restoring the native order
removed them. The exact-v1 checkpoint matches all checked intermediate values
on 384 layer/token cases. The subsequent `full-bitwise-v3.jsonl` gate checks
raw bytes of every full-vocabulary logit: 1,424 steps for each of four corrected
variants, or 5,696 complete-vector comparisons, plus prefill. All pass. Generated
tokens and stopping behavior also match. Source snapshots and hashes accompany
the results. The optimized `fast` scheduler subsequently passes its own 1,424
raw-byte comparisons in `full-fast-v4.jsonl`, plus prefill. Its final paired
comparison is `benchmark-final-v4.jsonl`; the source hashes match that gate.

## Scheduling progress and timing rules

Apple does not expose CUDA’s one-block-per-SM placement contract here. The ready
queue therefore claims only runnable consumers; it does not occupy workers
waiting for one particular producer. Stress includes one worker and many more
workers than GPU cores, changed input activations, and changed expert IDs.
Each submission checks every task ran exactly once and intermediate tensors
match. A bounded idle guard reports failure through NaN output and diagnostics.

Count workspace initialization and final join in persistent-branch timings.
Measure full-model decode with fresh caches, identical token sequences, warmup,
rotated variant order, and a compiled-native control. Separate GPU trace
measurements from uninstrumented throughput. Desktop timing variation remains
visible in the raw runs; no exclusive clocks or hardware isolation are claimed.
