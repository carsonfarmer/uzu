# Why the current MLX-VLM control is hard to beat

These are working explanations for the measured results on this M4 Pro, with
North Mini Code in affine 4-bit form. They are not established hardware causes.
The accepted control and all comparisons use actual MLX-VLM generation.

1. **The control has already removed several useful boundaries.** Prepared mode
   combines normalization, QKV/router projection, and the attention-output/MoE
   work inside the real model. Early router release adds another roughly 3% over
   that prepared control. A larger kernel must find savings beyond these changes.
   The four/eight-layer persistent pilot was effectively tied with early release;
   48-layer persistence was 4.12% slower. Fewer launches alone did not help.

2. **Our retained-weight window is much smaller than Cohere's.** The selected
   12 KiB tile can retain at most 384 KiB across 32 workers, around 1.83% of the
   next layer's 20.5 MiB QKV/router matrices. We proved those retained values are
   consumed and recorded loads finishing before the previous layer completed.
   This does not provide the same coverage as Cohere's larger Hopper pipeline.
   In 105 actual generations, early loading improved short/Rust throughput by
   1.98%/1.70% over the matched late-loading policy, but remained 1.42%/1.47%
   below the original span and 5.45%/5.61% below early router release. The
   policies share storage and reservation rules; executed tile counts can
   differ with scheduling, so this is not a pure measurement of memory overlap.

3. **Each task must remain competitive inside the larger kernel.** The long
   attention path is exact, yet the 48-layer long-context generation pilot fell
   to 43.408 tokens/s from 64.267 for early release. Register allocation, task
   scheduling, partial-result storage and cache integration could contribute.
   We have not isolated their shares, so the result cannot identify one cause.

4. **Host work increases, but removing it may not shorten the critical path.**
   Normal-generation profiling measured 148 ms more main-thread CPU for a
   128-token persistent generation than early release, alongside an 81 ms wall
   penalty. The separately instrumented profile found live-input guards and
   graph construction were substantial. A compiled body then passed lifecycle
   tests and 566 stock-logit comparisons, but its 72-generation comparison
   showed no throughput gain. The compiler preserves live weights and cache
   specialization. A follow-up instrumented profile measured about 50 ms less
   graph-construction CPU, but whole-run main-thread CPU varied and throughput
   remained unchanged. CPU and GPU execution overlap, so saved CPU is not
   automatically saved generation time.

5. **The Apple and Hopper workloads differ.** Cohere reports BF16 on H100; this
   experiment uses W4/G64 expert and embedding weights on an M4 Pro, with BF16
   attention/router weights. Quantization changes bytes moved and adds unpacking
   arithmetic. Different threadgroup storage and scheduling constraints affect
   which overlap is useful. An H100 speedup therefore does not predict the
   magnitude of an Apple speedup.

Extending the persistent span through final normalization and the vocabulary
head also passed full-array component gates. Neither a dynamic head queue nor
static head assignment improved the balanced component timing. These are not
full-generation results. Grouping native attention partitions into larger
worker tasks reduced the 48-layer component from 21.05 to 17.84 ms; reusing
partial storage after each completed layer reduced it to 17.42 ms. Native
numerical partitions and order remain exact. This identifies useful changes
within a slow prototype; the actual engine must still establish any benefit
over its control.

6. **The persistent integration adds native cache copies.** A calibrated backend
   observer found that prepared and early release reused every cache buffer on
   all three prompts. Persistent generation scheduled 4.609 GB, 24.333 GB and
   54.453 GB of logical cache copies on short, long and rotating contexts,
   respectively, over 128 generated tokens. All nine normal generations matched
   stock. This is a measured integration cost; its share of the throughput loss
   remains a hypothesis. It does not show an inherent Apple megakernel limit.
   MLX retains input buffers for GPU completion and requires unique ownership
   for native updates. Reading caches before updating them can therefore force
   copies. The first dependency-aware reuse candidate passed synthetic and
   real-engine correctness gates but removed only 0.90–3.26% of copy payload.
   A diagnostic identified an additional native `Depends` lifetime hold: it
   retains cache dependencies despite performing no GPU read of them. A narrow
   follow-up accounts for those holds while preserving all buffer lifetimes;
   no throughput benefit from that candidate has been established yet.

The rotation attention-load ablation separately improved its prototype from
30.809 to 37.095 tokens/s, while early release reached 57.872. Only the partition
containing the replaced slot needs conditional loads; other partitions now use
ordinary native loads. The 20.40% relative improvement remains 35.90% behind the
proper engine control. This result isolates a useful implementation change,
not the hardware cause of every remaining cost.

The additional 10% goal remains open; these hypotheses do not establish that it
is impossible.

References: [Cohere's article](https://cohere.com/blog/megakernels),
[pinned Cohere scheduler](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/schedule.py),
and the preserved raw runs described in [the progress record](MLX_VLM_MEGAKERNEL_PROGRESS.md).
