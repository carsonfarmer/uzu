"""Paired latency comparison of the exact custom and MLX attention kernels."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import ROOT
from full_layer.attention import run_attention


p = argparse.ArgumentParser()
p.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 1000])
p.add_argument("--repetitions", type=int, default=31)
p.add_argument("--output", default="experiments/north/full_layer/bench-attention-v1.json")
a = p.parse_args()
mx.random.seed(7)
samples = []

for context in a.contexts:
    q = mx.random.normal((1, 32, 1, 128)).astype(mx.bfloat16)
    k = mx.random.normal((1, 4, context, 128)).astype(mx.bfloat16)
    v = mx.random.normal((1, 4, context, 128)).astype(mx.bfloat16)

    def native(q, k, v):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=128**-0.5)

    def custom(q, k, v):
        return run_attention(q, k, v, context)

    variants = {"mlx": mx.compile(native), "custom": mx.compile(custom)}
    for repetition in range(a.repetitions):
        order = list(variants)
        if repetition % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter()
            out = variants[name](q, k, v)
            mx.eval(out)
            elapsed = (time.perf_counter() - started) * 1e6
            if repetition >= 2:
                samples.append({"context": context, "repetition": repetition, "variant": name, "wall_us": elapsed})

result = {"device": mx.device_info(), "args": vars(a), "samples": samples}
(ROOT / a.output).write_text(json.dumps(result, indent=2) + "\n")
for context in a.contexts:
    table = {
        name: [row["wall_us"] for row in samples if row["context"] == context and row["variant"] == name]
        for name in ("mlx", "custom")
    }
    paired = [custom / native - 1 for custom, native in zip(table["custom"], table["mlx"])]
    print(
        context,
        {name: round(statistics.median(values), 2) for name, values in table.items()},
        "custom/MLX paired",
        round(statistics.median(paired) * 100, 2),
        "%",
        flush=True,
    )
