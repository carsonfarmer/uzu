"""Raw-byte fixture gate for the persistent layer-tail/next-layer preparation."""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quantized"))

from reference import ROOT, load
from kernels import inputs
from tail_prep import EARLY_ROUNDS, ERROR, PREFETCH_DONE, PREFETCH_GROUPS, TOTAL_TASKS, VISITS, run_tail_prep
from full_layer.static_tail import (
    ERROR as STATIC_ERROR,
    TOTAL_TASKS as STATIC_TOTAL_TASKS,
    VISITS as STATIC_VISITS,
    run_static_tail,
)


p = argparse.ArgumentParser()
p.add_argument("--layer", type=int, default=1)
p.add_argument("--mode", choices=["fine", "static"], default="fine")
p.add_argument("--prefetch", action="store_true")
p.add_argument("--workers", type=int, nargs="+", default=[48, 64, 128, 256])
p.add_argument("--repetitions", type=int, default=3)
p.add_argument("--output", default="experiments/north/full_layer/check-v1.json")
a = p.parse_args()

model, _ = load()
layer = model.layers[a.layer]
next_layer = model.layers[a.layer + 1]
capture = ROOT / ("work/north-layer" if a.layer == 1 else "work/north-layer7")
meta = json.loads((capture / "metadata.json").read_text())


def bf(name, shape):
    return mx.array(np.fromfile(capture / f"{name}.bin", dtype=np.uint16)).view(mx.bfloat16).reshape(shape)


x = bf("x", (1, 1, 2048))
ax = bf("ax", (1, 1, 4096))
residual = bf("residual", (1, 1, 2048))
ids = mx.array(meta["experts"], dtype=mx.uint32).reshape(1, 1, 8)
scores = mx.array(np.fromfile(capture / "scores.bin", dtype=np.float32)).reshape(1, 1, 8)
data = inputs(layer.mlp, x, ax, ids, layer.self_attn.o_proj.weight)
mx.eval(*data, scores, residual)

checks = []
for repetition in range(a.repetitions):
    xx = (x * (1 + repetition * 0.013)).astype(mx.bfloat16)
    ii = ((ids + repetition * 7) % 128).astype(mx.uint32)
    current_data = [xx, ax, ii, *data[3:]]

    # Build the reference from the exact branch result, then use the unchanged
    # next-layer MLX modules for normalization and projections.
    from exact import run_exact

    out_ref = run_exact(current_data, scores, residual, workers=160, rows=16)
    norm_ref = next_layer.input_layernorm(out_ref)
    refs = (
        out_ref,
        norm_ref,
        next_layer.self_attn.q_proj(norm_ref),
        next_layer.self_attn.k_proj(norm_ref),
        next_layer.self_attn.v_proj(norm_ref),
        next_layer.mlp.gate(norm_ref),
    )
    mx.eval(*refs)

    for workers in a.workers:
        if a.mode == "static":
            got = run_static_tail(current_data, scores, residual, next_layer, workers=workers, return_state=True)
            error_index, visits_index, total_tasks = STATIC_ERROR, STATIC_VISITS, STATIC_TOTAL_TASKS
        else:
            got = run_tail_prep(
                current_data,
                scores,
                residual,
                next_layer,
                workers=workers,
                prefetch=a.prefetch,
                return_state=True,
            )
            error_index, visits_index, total_tasks = ERROR, VISITS, TOTAL_TASKS
        mx.eval(*got)
        unequal = [
            int(mx.sum(value.view(mx.uint8) != reference.view(mx.uint8)).item())
            for value, reference in zip(got[:6], refs)
        ]
        state = np.array(got[6])
        visits = state[visits_index : visits_index + total_tasks]
        row = {
            "repetition": repetition,
            "workers": workers,
            "unequal_bytes": unequal,
            "errors": int(state[error_index]),
            "task_visits_min": int(visits.min()),
            "task_visits_max": int(visits.max()),
            "prefetched_groups": int(state[PREFETCH_DONE]) if a.mode == "fine" else 0,
            "early_prefetch_rounds": int(state[EARLY_ROUNDS]) if a.mode == "fine" else 0,
        }
        checks.append(row)
        print(row, flush=True)
        assert unequal == [0] * 6, row
        assert row["errors"] == 0 and np.all(visits == 1), row
        if a.mode == "fine":
            assert row["prefetched_groups"] == PREFETCH_GROUPS, row
            assert (row["early_prefetch_rounds"] > 0) == a.prefetch, row

result = {"device": mx.device_info(), "args": vars(a), "checks": checks}
(ROOT / a.output).write_text(json.dumps(result, indent=2) + "\n")
print("PASS", len(checks), "cross-layer checks", flush=True)
