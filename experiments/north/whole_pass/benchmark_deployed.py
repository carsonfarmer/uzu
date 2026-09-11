"""Benchmark the packed path after releasing duplicate per-layer model weights."""

import argparse
import copy
import gc
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quantized"))

from reference import ROOT, load
from mlx_vlm.models.base import create_attention_mask
from whole_pass.kernel import run_whole_pass
from whole_pass.pack import grow_caches, pack_caches, pack_weights


p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=64)
p.add_argument("--runs", type=int, default=3)
p.add_argument("--workers", type=int, nargs="+", default=[36])
p.add_argument("--schedule", choices=("static", "fine", "queue"), default="queue")
p.add_argument("--prefetch-stages", type=int, default=0)
p.add_argument("--prep-rows", type=int, choices=(32, 64, 128), default=64)
p.add_argument("--oproj-rows", type=int, choices=(32, 64, 128), default=64)
p.add_argument("--router-rows", type=int, choices=(4, 8, 16, 32), default=8)
p.add_argument("--lm-rows", type=int, choices=(32, 64, 128, 256, 512, 1024), default=512)
p.add_argument("--head", action=argparse.BooleanOptionalAction, default=True)
p.add_argument("--prefix", action=argparse.BooleanOptionalAction, default=True)
p.add_argument(
    "--profiles",
    nargs="+",
    choices=("custom", "tuned", "coarse", "prefetch1", "prefetch10"),
    default=["custom"],
    help="Cycle named queue configurations in one packed process for fair ablations.",
)
p.add_argument(
    "--interleave-steps",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Rotate profiles after every decode step to reduce clock-drift bias.",
)
p.add_argument("--output", default="experiments/north/whole_pass/benchmark-safe-megakernel-v1.jsonl")
a = p.parse_args()

model, config = load()
print("packing 48 layers", flush=True)
weights = pack_weights(model, start=1, count=48)
print("weights packed", flush=True)

model_path = ROOT / "work/models/North-Mini-Code-1.0-4bit"
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
tokenizer.chat_template = (model_path / "chat_template.jinja").read_text()
prompt = "Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values."
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True,
    reasoning=False,
    skip_thinking=True,
)
max_capacity = 256
while len(prompt_ids) + a.tokens > max_capacity:
    max_capacity *= 2
assert max_capacity <= 1024

# Prefill with the ordinary model, then convert the immutable decode state once.
caches = model.make_cache()
logits = model(mx.array([prompt_ids]), cache=caches).logits
mx.eval(logits)
initial_token = int(mx.argmax(logits[0, -1]).item())
initial_capacity = max(256, caches[1].keys.shape[2])
cache_start, cache_count = (0, 49) if a.prefix else (1, 48)
initial_keys, initial_values = pack_caches(caches, start=cache_start, count=cache_count, capacity=initial_capacity)
initial_position = caches[cache_start].offset
initial_cache0 = caches[0]
layer0 = model.layers[0]

# A deployed packed engine does not retain a second 48-layer weight layout.
model.model.layers = [] if a.prefix else [layer0]
del caches
gc.collect()
mx.clear_cache()
active_after_release = mx.get_active_memory()
print("active packed memory", round(active_after_release / 2**30, 3), "GiB", flush=True)


profiles = {
    "custom": {
        "prefetch_stages": a.prefetch_stages,
        "prep_rows": a.prep_rows,
        "oproj_rows": a.oproj_rows,
        "router_rows": a.router_rows,
        "lm_rows": a.lm_rows,
    },
    "tuned": {
        "prefetch_stages": 0,
        "prep_rows": 64,
        "oproj_rows": 64,
        "router_rows": 8,
        "lm_rows": 512,
    },
    "coarse": {
        "prefetch_stages": 0,
        "prep_rows": 128,
        "oproj_rows": 128,
        "router_rows": 16,
        "lm_rows": 512,
    },
    "prefetch1": {
        "prefetch_stages": 1,
        "prep_rows": 64,
        "oproj_rows": 64,
        "router_rows": 8,
        "lm_rows": 512,
    },
    "prefetch10": {
        "prefetch_stages": 10,
        "prep_rows": 64,
        "oproj_rows": 64,
        "router_rows": 8,
        "lm_rows": 512,
    },
}
assert len(set(a.profiles)) == len(a.profiles)


def fresh_state():
    return {
        "cache0": copy.copy(initial_cache0),
        "key_cache": initial_keys,
        "value_cache": initial_values,
        "capacity": initial_capacity,
        "position": initial_position,
        "token": initial_token,
        "generated": [initial_token],
        "steps": [],
    }


def advance(state, workers, profile):
    profile_args = profiles[profile]
    started = time.perf_counter()
    if state["position"] >= state["capacity"]:
        state["capacity"] *= 2
        state["key_cache"], state["value_cache"] = grow_caches(
            state["key_cache"], state["value_cache"], state["capacity"]
        )
    h = model.model.embed_tokens(mx.array([[state["token"]]]))
    if not a.prefix:
        mask = create_attention_mask(h, state["cache0"])
        h = layer0(h, mask, state["cache0"])
    got = run_whole_pass(
        weights,
        h,
        state["key_cache"],
        state["value_cache"],
        state["position"],
        workers=workers,
        schedule=a.schedule,
        prefetch_stages=profile_args["prefetch_stages"],
        prep_rows=profile_args["prep_rows"],
        oproj_rows=profile_args["oproj_rows"],
        router_rows=profile_args["router_rows"],
        do_head=a.head,
        lm_rows=profile_args["lm_rows"],
        do_prefix=a.prefix,
    )
    h, state["key_cache"], state["value_cache"] = got[:3]
    if a.head:
        logits = got[4]
    else:
        logits = (
            model.model.embed_tokens.as_linear(model.model.norm(h))
            * model.model.args.logit_scale
        )
    mx.eval(logits, state["key_cache"], state["value_cache"])
    state["token"] = int(mx.argmax(logits[0, -1]).item())
    state["steps"].append(time.perf_counter() - started)
    state["generated"].append(state["token"])
    state["position"] += 1


def measure(workers, profile):
    state = fresh_state()
    while len(state["generated"]) < a.tokens:
        advance(state, workers, profile)
    return state["generated"], state["steps"]


def measure_interleaved(configurations, repetition):
    states = {configuration: fresh_state() for configuration in configurations}
    for step in range(a.tokens - 1):
        shift = (repetition + 1 + step) % len(configurations)
        order = configurations[shift:] + configurations[:shift]
        for profile, workers in order:
            advance(states[(profile, workers)], workers, profile)
    return {
        configuration: (state["generated"], state["steps"])
        for configuration, state in states.items()
    }


output = ROOT / a.output
output.parent.mkdir(parents=True, exist_ok=True)
sources = [
    Path(__file__),
    Path(__file__).resolve().parents[1] / "reference.py",
    Path(__file__).with_name("kernel.py"),
    Path(__file__).with_name("pack.py"),
    Path(__file__).with_name("primitives.py"),
    Path(__file__).resolve().parents[1] / "full_layer/full_attention.py",
    Path(__file__).resolve().parents[1] / "quantized/kernels.py",
    Path(__file__).resolve().parents[1] / "quantized/kernel.h",
    Path(__file__).resolve().parents[1] / "persistent/down.h",
]
with output.open("w") as stream:
    def save(row):
        stream.write(json.dumps(row) + "\n")
        stream.flush()

    save({
        "kind": "provenance",
        "device": mx.device_info(),
        "mlx": mx.__version__,
        "args": vars(a),
        "prompt_tokens": len(prompt_ids),
        "initial_capacity": initial_capacity,
        "max_capacity": max_capacity,
        "active_memory_after_original_release": active_after_release,
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "note": "Ordinary layers 1-48 are released after prefill. One warmup precedes measured runs; packing and prefill are excluded.",
    })
    expected = None
    for repetition in range(-1, a.runs):
        order = [(profile, workers) for profile in a.profiles for workers in a.workers]
        shift = (repetition + 1) % len(order)
        order = order[shift:] + order[:shift]
        mx.reset_peak_memory()
        if a.interleave_steps and len(order) > 1:
            measured = measure_interleaved(order, repetition)
        else:
            measured = {
                (profile, workers): measure(workers, profile)
                for profile, workers in order
            }
        peak_memory = mx.get_peak_memory()
        for profile, workers in order:
            generated, steps = measured[(profile, workers)]
            if expected is None:
                expected = generated
            assert generated == expected
            row = {
                "kind": "generation",
                "profile": profile,
                "profile_args": profiles[profile],
                "workers": workers,
                "repetition": repetition,
                "warmup": repetition < 0,
                "decode_steps": len(steps),
                "decode_seconds": sum(steps),
                "decode_tokens_per_second": len(steps) / sum(steps),
                "first_step_seconds": steps[0],
                "median_step_seconds": statistics.median(steps),
                "step_seconds": steps,
                "token_ids": generated,
                "peak_memory_bytes": peak_memory,
            }
            save(row)
            print(profile, workers, repetition, round(row["decode_tokens_per_second"], 3), "tok/s", flush=True)

rows = [json.loads(line) for line in output.read_text().splitlines()]
for profile in a.profiles:
    for workers in a.workers:
        values = [
            row["decode_tokens_per_second"]
            for row in rows
            if row.get("kind") == "generation"
            and not row["warmup"]
            and row["profile"] == profile
            and row["workers"] == workers
        ]
        print(profile, workers, "median", round(statistics.median(values), 3), "tok/s", flush=True)
