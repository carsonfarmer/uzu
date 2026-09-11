"""Paired decode benchmark for the 48-layer single-dispatch Metal path."""

import argparse
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

from patch import variants as branch_variants
from reference import ROOT, load
from mlx_vlm.models.base import create_attention_mask
from whole_pass.kernel import run_whole_pass
from whole_pass.pack import grow_caches, pack_caches, pack_weights


p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=128)
p.add_argument("--runs", type=int, default=3)
p.add_argument("--workers", type=int, default=36)
p.add_argument("--whole-schedules", nargs="+", choices=("static", "fine", "queue"), default=["queue"])
p.add_argument("--prefetch-stages", type=int, nargs="+", default=[0])
p.add_argument("--prep-rows", type=int, nargs="+", choices=(32, 64, 128), default=[64])
p.add_argument("--oproj-rows", type=int, nargs="+", choices=(32, 64, 128), default=[64])
p.add_argument("--router-rows", type=int, nargs="+", choices=(4, 8, 16, 32), default=[8])
p.add_argument("--lm-rows", type=int, nargs="+", choices=(32, 64, 128, 256, 512, 1024), default=[512])
p.add_argument("--head-modes", nargs="+", choices=("external", "integrated"), default=["integrated"])
p.add_argument("--prefix-modes", nargs="+", choices=("external", "integrated"), default=["integrated"])
p.add_argument("--modes", nargs="+", default=["original", "exact", "whole"])
p.add_argument("--output", default="experiments/north/whole_pass/benchmark-whole-short-v1.jsonl")
a = p.parse_args()

model, config = load()
original = list(model.layers)
exact = branch_variants(model, workers=160, rows=16, modes=("exact",))["exact"]
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
eos = config["eos_token_id"]
eos = {eos} if isinstance(eos, int) else set(eos)


def prefill(layers):
    model.model.layers = layers
    caches = model.make_cache()
    logits = model(mx.array([prompt_ids]), cache=caches).logits
    mx.eval(logits)
    return caches, logits


def standard_measure(layers):
    caches, logits = prefill(layers)
    token = int(mx.argmax(logits[0, -1]).item())
    generated = [token]
    steps = []
    while len(generated) < a.tokens and token not in eos:
        started = time.perf_counter()
        logits = model(mx.array([[token]]), cache=caches).logits
        mx.eval(logits)
        token = int(mx.argmax(logits[0, -1]).item())
        steps.append(time.perf_counter() - started)
        generated.append(token)
    return generated, steps


def whole_measure(schedule, prefetch_stages, prep_rows, oproj_rows, router_rows, lm_rows, head_mode, prefix_mode):
    caches, logits = prefill(original)
    capacity = max(256, caches[1].keys.shape[2])
    cache_start, cache_count = (0, 49) if prefix_mode == "integrated" else (1, 48)
    key_cache, value_cache = pack_caches(caches, start=cache_start, count=cache_count, capacity=capacity)
    position = caches[cache_start].offset
    token = int(mx.argmax(logits[0, -1]).item())
    generated = [token]
    steps = []
    while len(generated) < a.tokens and token not in eos:
        started = time.perf_counter()
        if position >= capacity:
            capacity *= 2
            key_cache, value_cache = grow_caches(key_cache, value_cache, capacity)
        h = model.model.embed_tokens(mx.array([[token]]))
        if prefix_mode == "external":
            layer0 = original[0]
            mask = create_attention_mask(h, caches[0])
            h = layer0(h, mask, caches[0])
        got = run_whole_pass(
            weights,
            h,
            key_cache,
            value_cache,
            position,
            workers=a.workers,
            schedule=schedule,
            prefetch_stages=prefetch_stages,
            prep_rows=prep_rows,
            oproj_rows=oproj_rows,
            router_rows=router_rows,
            do_head=head_mode == "integrated",
            lm_rows=lm_rows,
            do_prefix=prefix_mode == "integrated",
        )
        h, key_cache, value_cache = got[:3]
        if head_mode == "integrated":
            logits = got[4]
        else:
            logits = model.model.embed_tokens.as_linear(model.model.norm(h)) * model.model.args.logit_scale
        mx.eval(logits, key_cache, value_cache)
        token = int(mx.argmax(logits[0, -1]).item())
        steps.append(time.perf_counter() - started)
        generated.append(token)
        position += 1
    return generated, steps


controls = {
    "original": lambda: standard_measure(original),
    "exact": lambda: standard_measure(exact),
}
for schedule in a.whole_schedules:
    for prefetch_stages in a.prefetch_stages:
        for prep_rows in a.prep_rows:
            for oproj_rows in a.oproj_rows:
                for router_rows in a.router_rows:
                    for lm_rows in a.lm_rows:
                        for head_mode in a.head_modes:
                            for prefix_mode in a.prefix_modes:
                                suffix = "" if prep_rows == 32 else f"_r{prep_rows}"
                                suffix += "" if oproj_rows == 32 else f"_o{oproj_rows}"
                                suffix += "" if router_rows == 4 else f"_t{router_rows}"
                                suffix += "" if lm_rows == 128 else f"_l{lm_rows}"
                                suffix += "" if prefetch_stages == 0 else f"_p{prefetch_stages}"
                                suffix += "" if prefix_mode == "external" else "_prefix"
                                suffix += "" if head_mode == "external" else "_head"
                                name = "whole" if schedule == "queue" else f"whole_{schedule}_unsafe"
                                name += suffix
                                controls[name] = lambda schedule=schedule, prefetch_stages=prefetch_stages, prep_rows=prep_rows, oproj_rows=oproj_rows, router_rows=router_rows, lm_rows=lm_rows, head_mode=head_mode, prefix_mode=prefix_mode: whole_measure(
                                    schedule, prefetch_stages, prep_rows, oproj_rows, router_rows, lm_rows, head_mode, prefix_mode
                                )
unknown_modes = set(a.modes) - set(controls)
assert not unknown_modes, f"unknown modes for requested schedules/stages: {sorted(unknown_modes)}"
controls = {name: controls[name] for name in a.modes}
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
        "initial_capacity": 256,
        "max_capacity": max_capacity,
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "note": "One warmup per variant. Variant order rotates. Weight packing is one-time setup and excluded; cache carry-forward is included.",
    })
    expected = None
    for repetition in range(-1, a.runs):
        names = list(controls)
        shift = (repetition + 1) % len(names)
        names = names[shift:] + names[:shift]
        for name in names:
            mx.reset_peak_memory()
            generated, steps = controls[name]()
            if expected is None or name == "original":
                expected = generated
            assert generated == expected, (repetition, name)
            row = {
                "kind": "generation",
                "variant": name,
                "repetition": repetition,
                "warmup": repetition < 0,
                "decode_steps": len(steps),
                "decode_seconds": sum(steps),
                "decode_tokens_per_second": len(steps) / sum(steps),
                "step_seconds": steps,
                "token_ids": generated,
                "peak_memory_bytes": mx.get_peak_memory(),
            }
            save(row)
            print(name, repetition, round(row["decode_tokens_per_second"], 2), "tok/s", flush=True)
            gc.collect()

model.model.layers = original
rows = [json.loads(line) for line in output.read_text().splitlines()]
medians = {}
for name in controls:
    values = [row["decode_tokens_per_second"] for row in rows if row.get("variant") == name and not row.get("warmup")]
    medians[name] = statistics.median(values)
print({name: round(value, 3) for name, value in medians.items()}, flush=True)
