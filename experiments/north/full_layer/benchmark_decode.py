"""Paired full-model throughput benchmark for the cross-layer decode path."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quantized"))

from reference import ROOT, load
from patch import variants as branch_variants
from full_layer.patch import cross_layer_path


p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=256)
p.add_argument("--runs", type=int, default=3)
p.add_argument("--workers", type=int, default=64)
p.add_argument("--rows", type=int, default=16)
p.add_argument("--modes", nargs="+", choices=["exact", "fast", "cross", "cross_static", "cross_prefetch", "full", "full_front", "full_native", "full_static", "full_static_native"], default=["exact", "cross"])
p.add_argument("--prompts", nargs="+", default=["short", "long", "rust"])
p.add_argument("--output", default="experiments/north/full_layer/benchmark-cross-v1.jsonl")
a = p.parse_args()

model, config = load()
original = list(model.layers)
paths = {"original": original}
branch_modes = tuple(mode for mode in a.modes if mode in ("exact", "fast"))
if branch_modes:
    controls = branch_variants(model, workers=160, rows=a.rows, modes=branch_modes, schedule_workers=a.workers)
    paths.update({mode: controls[mode] for mode in branch_modes})
if "cross" in a.modes:
    _, paths["cross"] = cross_layer_path(model, workers=a.workers)
if "cross_static" in a.modes:
    _, paths["cross_static"] = cross_layer_path(model, workers=a.workers, schedule="static")
if "cross_prefetch" in a.modes:
    _, paths["cross_prefetch"] = cross_layer_path(model, workers=a.workers, schedule="prefetch")
if "full" in a.modes:
    _, paths["full"] = cross_layer_path(model, workers=a.workers, schedule="full")
if "full_front" in a.modes:
    _, paths["full_front"] = cross_layer_path(model, workers=a.workers, schedule="full_front")
if "full_native" in a.modes:
    _, paths["full_native"] = cross_layer_path(model, workers=a.workers, schedule="full_native")
if "full_static" in a.modes:
    _, paths["full_static"] = cross_layer_path(model, workers=a.workers, schedule="full_static")
if "full_static_native" in a.modes:
    _, paths["full_static_native"] = cross_layer_path(model, workers=a.workers, schedule="full_static_native")

model_path = ROOT / "work/models/North-Mini-Code-1.0-4bit"
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
tokenizer.chat_template = (model_path / "chat_template.jinja").read_text()
eos = config["eos_token_id"]
eos = {eos} if isinstance(eos, int) else set(eos)
prompts = {
    "short": "Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.",
    "long": "\n".join(f"def helper_{i}(x): return x + {i}" for i in range(128))
    + "\nReturn only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.",
    "rust": "Return only Rust code for finding the first duplicate in a u32 slice, with tests for empty input and duplicates.",
}
prior = {
    row["prompt"]: row["token_ids"]
    for row in map(
        json.loads,
        (ROOT / "experiments/north/correctness/results/full-exact-v1.jsonl").read_text().splitlines(),
    )
    if row["kind"] == "completion"
}


def prefill(ids, mode):
    model.model.layers = paths[mode]
    cache = model.make_cache()
    for position in range(0, len(ids), 256):
        logits = model(mx.array([ids[position : position + 256]]), cache=cache).logits
        mx.eval(logits)
    return cache, logits


def measure(ids, mode):
    mx.reset_peak_memory()
    started = time.perf_counter()
    cache, logits = prefill(ids, mode)
    token = int(mx.argmax(logits[0, -1]).item())
    ttft = time.perf_counter() - started
    generated = [token]
    steps = []
    while len(generated) < a.tokens and token not in eos:
        step_started = time.perf_counter()
        logits = model(mx.array([[token]]), cache=cache).logits
        token = int(mx.argmax(logits[0, -1]).item())
        steps.append(time.perf_counter() - step_started)
        generated.append(token)
    row = {
        "ttft_seconds": ttft,
        "decode_steps": len(steps),
        "decode_seconds": sum(steps),
        "decode_tokens_per_second": len(steps) / sum(steps),
        "step_seconds": steps,
        "wall_seconds": time.perf_counter() - started,
        "token_ids": generated,
        "generated_tokens": len(generated),
        "finish_reason": "stop" if token in eos else "length",
        "peak_memory_bytes": mx.get_peak_memory(),
        "text": tokenizer.decode(generated, skip_special_tokens=True),
    }
    del cache, logits
    gc.collect()
    return row


output = ROOT / a.output
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w") as stream:
    def save(row):
        stream.write(json.dumps(row) + "\n")
        stream.flush()

    sources = [
        Path(__file__),
        Path(__file__).with_name("patch.py"),
        Path(__file__).with_name("tail_prep.py"),
        Path(__file__).with_name("attention.py"),
        Path(__file__).with_name("full_attention.py"),
        Path(__file__).with_name("static_full_attention.py"),
    ]
    save({
        "kind": "provenance",
        "device": mx.device_info(),
        "mlx": mx.__version__,
        "args": vars(a),
        "mlx_environment": {key: value for key, value in os.environ.items() if key.startswith("MLX_")},
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "note": "One warmup per variant and prompt; order rotates each repetition. Prefill remains native.",
    })
    for label in a.prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[label]}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            reasoning=False,
            skip_thinking=True,
        )
        for repetition in range(-1, a.runs):
            order = list(paths)
            shift = (repetition + 1) % len(order)
            order = order[shift:] + order[:shift]
            for mode in order:
                model.model.layers = paths[mode]
                row = measure(ids, mode)
                assert row["token_ids"] == prior[label][: len(row["token_ids"])], (label, mode)
                row.update({
                    "kind": "generation",
                    "prompt": label,
                    "prompt_tokens": len(ids),
                    "variant": mode,
                    "repetition": repetition,
                    "warmup": repetition < 0,
                })
                save(row)
                print(label, mode, repetition, round(row["decode_tokens_per_second"], 2), "tok/s", flush=True)
    model.model.layers = original

rows = [json.loads(line) for line in output.read_text().splitlines()]
for label in a.prompts:
    values = {
        mode: [row["decode_tokens_per_second"] for row in rows if row.get("prompt") == label and row.get("variant") == mode and not row.get("warmup")]
        for mode in paths
    }
    medians = {
        mode: round(statistics.median(samples), 3)
        for mode, samples in values.items()
        if samples
    }
    if medians:
        print(label, medians, flush=True)
