"""Repeated full-logit gate for the 48-layer single-dispatch Metal path."""

import argparse
import hashlib
import json
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
p.add_argument("--workers", type=int, default=36)
p.add_argument("--schedule", choices=("static", "fine", "queue"), default="queue")
p.add_argument("--prefetch-stages", type=int, default=0)
p.add_argument("--prep-rows", type=int, choices=(32, 64, 128), default=64)
p.add_argument("--oproj-rows", type=int, choices=(32, 64, 128), default=64)
p.add_argument("--router-rows", type=int, choices=(4, 8, 16, 32), default=8)
p.add_argument("--lm-rows", type=int, choices=(32, 64, 128, 256, 512, 1024), default=512)
p.add_argument("--head", action=argparse.BooleanOptionalAction, default=True)
p.add_argument("--prefix", action=argparse.BooleanOptionalAction, default=True)
p.add_argument(
    "--prompt",
    default="Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.",
)
p.add_argument("--output", default="experiments/north/whole_pass/full-safe-megakernel-short64-v1.jsonl")
a = p.parse_args()

model, config = load()
print("packing 48 layers", flush=True)
weights = pack_weights(model, start=1, count=48)
print("weights packed", flush=True)
model_path = ROOT / "work/models/North-Mini-Code-1.0-4bit"
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
tokenizer.chat_template = (model_path / "chat_template.jinja").read_text()
ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": a.prompt}],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True,
    reasoning=False,
    skip_thinking=True,
)
capacity = 256
assert len(ids) + a.tokens <= 1024
ref_caches = model.make_cache()
cand_caches = model.make_cache()
logits = {}
for name, caches in (("reference", ref_caches), ("whole", cand_caches)):
    logits[name] = model(mx.array([ids]), cache=caches).logits
    mx.eval(logits[name])
assert bool(mx.all(logits["reference"].view(mx.uint8) == logits["whole"].view(mx.uint8)).item())
cache_start, cache_count = (0, 49) if a.prefix else (1, 48)
key_cache, value_cache = pack_caches(cand_caches, start=cache_start, count=cache_count, capacity=capacity)
position = cand_caches[cache_start].offset
token = int(mx.argmax(logits["reference"][0, -1]).item())
generated = [token]

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
        "prompt_tokens": len(ids),
        "capacity": capacity,
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    })
    started = time.perf_counter()
    while len(generated) < a.tokens:
        logits["reference"] = model(mx.array([[token]]), cache=ref_caches).logits

        if position >= capacity:
            capacity *= 2
            key_cache, value_cache = grow_caches(key_cache, value_cache, capacity)

        h = model.model.embed_tokens(mx.array([[token]]))
        if not a.prefix:
            layer0 = model.layers[0]
            mask = create_attention_mask(h, cand_caches[0])
            h = layer0(h, mask, cand_caches[0])
        got = run_whole_pass(
            weights,
            h,
            key_cache,
            value_cache,
            position,
            workers=a.workers,
            schedule=a.schedule,
            prefetch_stages=a.prefetch_stages,
            prep_rows=a.prep_rows,
            oproj_rows=a.oproj_rows,
            router_rows=a.router_rows,
            do_head=a.head,
            lm_rows=a.lm_rows,
            do_prefix=a.prefix,
        )
        h, key_cache, value_cache, state = got[:4]
        if a.head:
            logits["whole"] = got[4]
        else:
            logits["whole"] = model.model.embed_tokens.as_linear(model.model.norm(h)) * model.model.args.logit_scale
        mx.eval(logits["reference"], logits["whole"], key_cache, value_cache, state)
        different = logits["reference"].view(mx.uint8) != logits["whole"].view(mx.uint8)
        if bool(mx.any(different).item()):
            delta = logits["whole"].astype(mx.float32) - logits["reference"].astype(mx.float32)
            cache_differences = []
            for local, layer_index in enumerate(range(cache_start, cache_start + cache_count)):
                reference_cache = ref_caches[layer_index]
                for packed, name in ((key_cache, "keys"), (value_cache, "values")):
                    reference_value = getattr(reference_cache, name)[0, :, : position + 1]
                    packed_value = packed[local, :, : position + 1]
                    unequal = int(
                        mx.sum(
                            packed_value.view(mx.uint8).reshape(-1)
                            != reference_value.view(mx.uint8).reshape(-1)
                        ).item()
                    )
                    if unequal:
                        cache_differences.append(
                            {"layer": layer_index, "kind": name, "unequal_bytes": unequal}
                        )
            failure = {
                "kind": "failure",
                "step": len(generated),
                "position": position,
                "unequal_bytes": int(mx.sum(different).item()),
                "max_abs": float(mx.max(mx.abs(delta)).item()),
                "cache_differences": cache_differences,
            }
            save(failure)
            raise AssertionError(failure)
        token = int(mx.argmax(logits["reference"][0, -1]).item())
        generated.append(token)
        position += 1
        save({"kind": "step", "step": len(generated) - 1, "position": position - 1, "bitwise": True, "token": token})
        if (len(generated) - 1) % 8 == 0:
            print(len(generated) - 1, "exact whole-pass decode steps", flush=True)

    save({
        "kind": "completion",
        "prompt_tokens": len(ids),
        "generated_tokens": len(generated),
        "exact_decode_steps": len(generated) - 1,
        "token_ids": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "wall_seconds": time.perf_counter() - started,
    })
print("PASS", len(generated) - 1, "whole-pass decode steps", flush=True)
