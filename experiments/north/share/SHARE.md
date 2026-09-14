# Current sharing status

The previous suggested post is withdrawn. It overstated full-megakernel
validation and attributed a static scheduler failure to hardware residency
without a conclusive diagnosis. Its original text remains in Git history.

Use the [September 13 branch-mixing report](../branch_mixing/README.md) and
[current research status](../../../STATUS.md) for the findings. The specific
alternating task-ID schedule did not help. Nearly all of the fused gain remained
with separate front launches, and the small incremental mixed-kernel result was
not robust to timing protocol and outliers. This does not substantiate a claim
that Cohere's scheduling mechanism produced the historical 11% gain.

The full megakernel performance objective remains unfinished. A separate
[worktree report](https://github.com/carsonfarmer/uzu/tree/cf/north-additional-ten-percent/experiments/north/additional)
now verifies +10.70% Python and +10.92% Rust over the actual prior fused runner.
Most of that gain adopts standard MLX async submission; new preparation fusion
adds about 1.1–1.2% over fused async. Longer-context throughput is uncertain.
These are scoped research-runner results, not a new megakernel speedup claim.
No post or message has been sent to Cohere.
