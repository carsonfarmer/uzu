> **Update:** An actual MLX-VLM 0.7.1 integration has now been validated against the untouched published package: 10.25–11.62% median decode improvement. See [the new engine report](MLX_VLM_INTEGRATION.md). The earlier custom-runner headline remains withdrawn.

# Baseline and scope correction — September 14, 2026

**The requested existing-engine improvement has not been demonstrated.** The
North results were produced by a custom Python generation runner using MLX
and MLX-VLM model components, stored in the Uzu repository. They did not run
through Uzu's inference engine or the normal MLX-VLM generation entry point.

The combined 19–24% measurements compare with that runner's initial synchronous
loop. Part of the gain restores async submission already present in standard
MLX-VLM. Even the smaller custom-kernel gains against stock layers with async
remain measurements inside the custom runner. Neither establishes a speedup of
Uzu or the normal MLX-VLM application. The raw measurements are retained, but
their presentation as completion of the requested goal is withdrawn.

## How the scope went wrong

The experiment used real MLX libraries and real model weights. The failure was
choosing a substitute execution path and control, then treating internal
improvements to that path as the requested existing-engine result. Writing
custom kernels and validating their output did not remove the need to integrate
and measure them through the real engine's normal generation path.

The original [North feasibility assessment](NORTH_MINI_CODE.md) already identified
that this Uzu checkout could not correctly run North without additional support.
Those were prerequisites to address, not grounds to silently substitute a
standalone runner for an engine integration.

Source checks repeated on September 14 confirm the relevant gaps:

- Uzu's `crates/uzu-engine/src/encodable_block/transformer_layer.rs` feeds the
  attention result into the MLP path. North's reference computes attention and
  MLP from the same normalized input before joining their outputs.
- `crates/uzu-engine/src/config/mlp/routing_function/mod.rs` exposes softmax
  routing; North requires sigmoid scores with its selected-score semantics.
- `crates/uzu-engine/src/encodable_block/mlp/moe/mod.rs` requires biases and
  loads expert weights in the model data type. Correct North support also
  needs the checkpoint's bias and quantized-expert handling.
- MLX-VLM has an existing `cohere2_moe` model and an existing async generation
  loop. Its ordinary loading and generation entry points must be exercised
  directly for a stock MLX-VLM comparison.

## Required acceptance criteria

1. Establish an unmodified existing-engine baseline through its normal loading,
   cache, generation and sampling path. Record actual support gaps before
   choosing or changing the engine under test.
2. For Uzu, implement and verify missing model support before reporting a
   North-in-Uzu result. The user subsequently authorized MLX-VLM as an engine
   target; the new MLX-VLM result is reported explicitly as such.
3. Put the proposed optimization behind a switch in that same engine path.
   Compare enabled and disabled with the same model, precision, prompts,
   cache behavior, sampling and standard async submission. Do not rewrite
   generation merely to simplify the control.
4. Verify model outputs and repeated timings. Distinguish decode-only throughput
   from complete generation cost. Count only gains beyond functionality the
   standard engine already provides.
5. Test the actual Cohere scheduling and weight-overlap mechanisms against the
   competitive engine control before describing the megakernel goal as met.

Further GPU optimization against the old custom-runner target has been stopped.
The separate task is preserving its work and performing a CPU-only integration
inventory. No new engine-level performance claim is supported by the existing
measurements. Code, original artifacts and branches remain available for audit.
