# Rejected static-scheduler result

The first complete one-dispatch implementation assigned work statically to all
Metal threadgroups and used global spin barriers between phases. When a run
completed, it was bitwise exact and fast. It was not live under repetition.

## Successful evidence

- `benchmark-static-success-short128-v1.jsonl` contains three measured 127-step
  runs at 57.472, 57.271, and 57.588 tok/s; median 57.472 tok/s.
- `full-megakernel-short128-v1.jsonl` contains a completed 127-step exact-logit
  gate.
- SHA-256: `fb9c1d1b9b4f82b35ef4e6b408f9fdc4fe3aa672067d373d90d5ce938f09b807`
  for the benchmark and
  `84cbf049571c16db954b6fa7e564722c7fffbcb6ed6e3fcc6d48643bbc4f0510`
  for the completed exact gate.

## Failure evidence

On an adversarial rerun with the same 32-group static schedule, the verifier
wrote two bitwise-exact steps and then stopped making progress inside the Metal
dispatch. `full-static-python-aborted-after-deadlock-short128-v1.jsonl` is the
partial three-line artifact: provenance followed by steps 1 and 2, with no
completion record. Its SHA-256 is
`82753ebfd35f4f003af573357d5c19e97916f5b30462b54a53723d88ddc591f8`.
The stalled process was terminated manually. A 20-group rerun also stopped
after one exact step; that partial file was not retained.

The cause was not conclusively diagnosed. One plausible failure mechanism is
that a resident threadgroup waits for a group that has not been scheduled.
The retained artifact does not distinguish residency from publication/visibility
or other implementation errors. This failure does not prove a Metal limitation. The successful timing therefore shows
performance potential only. It cannot support a deployable-result claim.

The replacement queue gives resident groups only ready work. The group that
completes the last task advances the phase, so progress never depends on a
specific unscheduled group. The canonical queue checks live one directory
above; the static path remains in source solely for reproducing this finding.
