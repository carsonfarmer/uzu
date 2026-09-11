"""Full-model raw-logit gate for the cross-layer persistent decode path."""

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
from full_layer.patch import cross_layer_path


p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=512)
p.add_argument("--workers", type=int, default=64)
p.add_argument("--schedule", choices=["fine", "static", "prefetch", "full", "full_front", "full_native", "full_static", "full_static_native"], default="fine")
p.add_argument("--prompts", nargs="+", default=["short", "long", "rust"])
p.add_argument("--output", default="experiments/north/full_layer/full-cross-v1.jsonl")
a = p.parse_args()

model, config = load()
original, cross = cross_layer_path(model, workers=a.workers, schedule=a.schedule)
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

output = ROOT / a.output
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w") as stream:
    def save(row):
        stream.write(json.dumps(row) + "\n")
        stream.flush()

    files = [
        Path(__file__),
        Path(__file__).with_name("patch.py"),
        Path(__file__).with_name("tail_prep.py"),
        Path(__file__).with_name("attention.py"),
        Path(__file__).with_name("full_attention.py"),
    ]
    save({
        "kind": "provenance",
        "mlx": mx.__version__,
        "device": mx.device_info(),
        "args": vars(a),
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    })
    try:
        for label in a.prompts:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompts[label]}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                reasoning=False,
                skip_thinking=True,
            )
            caches = {"original": model.make_cache(), "cross": model.make_cache()}
            logits = {}
            for mode, layers in (("original", original), ("cross", cross)):
                model.model.layers = layers
                for offset in range(0, len(ids), 256):
                    logits[mode] = model(mx.array([ids[offset : offset + 256]]), cache=caches[mode]).logits
                    mx.eval(logits[mode])
            assert bool(mx.all(logits["original"].view(mx.uint8) == logits["cross"].view(mx.uint8)).item()), "prefill"

            token = int(mx.argmax(logits["original"][0, -1]).item())
            generated = [token]
            started = time.perf_counter()
            while len(generated) < a.tokens and token not in eos:
                for mode, layers in (("original", original), ("cross", cross)):
                    model.model.layers = layers
                    logits[mode] = model(mx.array([[token]]), cache=caches[mode]).logits
                    mx.eval(logits[mode])
                different = logits["original"].view(mx.uint8) != logits["cross"].view(mx.uint8)
                if bool(mx.any(different).item()):
                    delta = logits["cross"].astype(mx.float32) - logits["original"].astype(mx.float32)
                    failure = {
                        "kind": "failure",
                        "prompt": label,
                        "step": len(generated),
                        "unequal_bytes": int(mx.sum(different).item()),
                        "max_abs": float(mx.max(mx.abs(delta)).item()),
                    }
                    save(failure)
                    raise AssertionError(failure)
                token = int(mx.argmax(logits["original"][0, -1]).item())
                generated.append(token)
                save({"kind": "step", "prompt": label, "step": len(generated) - 1, "bitwise": True, "token": token})
                if (len(generated) - 1) % 64 == 0:
                    print(label, len(generated) - 1, "exact decode steps", flush=True)
            save({
                "kind": "completion",
                "prompt": label,
                "prompt_tokens": len(ids),
                "generated_tokens": len(generated),
                "exact_decode_steps": len(generated) - 1,
                "token_ids": generated,
                "text": tokenizer.decode(generated, skip_special_tokens=True),
                "finish_reason": "stop" if token in eos else "length",
                "check_wall_seconds": time.perf_counter() - started,
            })
            print(label, "PASS", len(generated) - 1, "exact decode steps", flush=True)
    finally:
        model.model.layers = original
