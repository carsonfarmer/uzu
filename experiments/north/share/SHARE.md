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
worktree is pursuing another 10% over the strongest exact fused control.
No post or message has been sent to Cohere.
