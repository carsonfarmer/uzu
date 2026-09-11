"""Paired branch-tail and next-layer preparation latency benchmark."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quantized"))

from reference import ROOT, load
from kernels import inputs
from exact import run_exact
from persistent.fast import run_fast
from full_layer.tail_prep import run_tail_prep
from full_layer.static_tail import run_static_tail


p = argparse.ArgumentParser()
p.add_argument("--layer", type=int, default=1)
p.add_argument("--workers", type=int, nargs="+", default=[20, 32, 48, 64, 80, 128])
p.add_argument("--repetitions", type=int, default=31)
p.add_argument("--output", default="experiments/north/full_layer/bench-tail-prep-v1.json")
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


def native_prep(out):
    norm = next_layer.input_layernorm(out)
    return (
        out,
        norm,
        next_layer.self_attn.q_proj(norm),
        next_layer.self_attn.k_proj(norm),
        next_layer.self_attn.v_proj(norm),
        next_layer.mlp.gate(norm),
    )


def fast_tail(h, attention_input, expert_ids, expert_scores, residual_input):
    current = inputs(layer.mlp, h, attention_input, expert_ids, layer.self_attn.o_proj.weight)
    return native_prep(run_fast(current, expert_scores, residual_input, workers=64))


def exact_tail(h, attention_input, expert_ids, expert_scores, residual_input):
    current = inputs(layer.mlp, h, attention_input, expert_ids, layer.self_attn.o_proj.weight)
    return native_prep(run_exact(current, expert_scores, residual_input, workers=160, rows=16))


controls = {
    "exact_plus_mlx_prep": mx.compile(exact_tail),
    "fast_plus_mlx_prep": mx.compile(fast_tail),
}
for workers in a.workers:
    def candidate(h, attention_input, expert_ids, expert_scores, residual_input, workers=workers):
        current = inputs(layer.mlp, h, attention_input, expert_ids, layer.self_attn.o_proj.weight)
        return run_tail_prep(current, expert_scores, residual_input, next_layer, workers=workers)

    controls[f"cross_w{workers}"] = mx.compile(candidate)

    def prefetch_candidate(h, attention_input, expert_ids, expert_scores, residual_input, workers=workers):
        current = inputs(layer.mlp, h, attention_input, expert_ids, layer.self_attn.o_proj.weight)
        return run_tail_prep(
            current,
            expert_scores,
            residual_input,
            next_layer,
            workers=workers,
            prefetch=True,
        )

    controls[f"prefetch_w{workers}"] = mx.compile(prefetch_candidate)

    def static_candidate(h, attention_input, expert_ids, expert_scores, residual_input, workers=workers):
        current = inputs(layer.mlp, h, attention_input, expert_ids, layer.self_attn.o_proj.weight)
        return run_static_tail(current, expert_scores, residual_input, next_layer, workers=workers)

    controls[f"static_w{workers}"] = mx.compile(static_candidate)

samples = []
for repetition in range(a.repetitions):
    xx = (x * (1 + repetition * 0.013)).astype(mx.bfloat16)
    ii = ((ids + repetition * 7) % 128).astype(mx.uint32)
    names = list(controls)
    shift = (repetition // 2) % len(names)
    names = names[shift:] + names[:shift]
    if repetition % 2:
        names.reverse()
    for name in names:
        started = time.perf_counter()
        values = controls[name](xx, ax, ii, scores, residual)
        mx.eval(*values)
        elapsed = (time.perf_counter() - started) * 1e6
        if repetition >= 2:
            samples.append({"repetition": repetition, "variant": name, "wall_us": elapsed})
        print(repetition, name, round(elapsed, 2), "us", flush=True)

result = {"device": mx.device_info(), "args": vars(a), "samples": samples}
(ROOT / a.output).write_text(json.dumps(result, indent=2) + "\n")
for name in controls:
    values = [sample["wall_us"] for sample in samples if sample["variant"] == name]
    print(name, round(statistics.median(values), 2), "us", flush=True)
