"""Compare whole-pass boundary primitives with pinned MLX raw bytes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference import ROOT, load
from whole_pass.primitives import run_primitives


p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=3)
p.add_argument("--layers", type=int, nargs="+", default=[1, 4, 7, 31, 47, 48])
p.add_argument("--output", default="experiments/north/whole_pass/check-primitives-v1.json")
a = p.parse_args()

model, _ = load()
model_path = ROOT / "work/models/North-Mini-Code-1.0-4bit"
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
tokenizer.chat_template = (model_path / "chat_template.jinja").read_text()
ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Return only Python code for merging two sorted lists."}],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True,
    reasoning=False,
    skip_thinking=True,
)
caches = model.make_cache()
logits = model(mx.array([ids]), cache=caches).logits
mx.eval(logits)
token = int(mx.argmax(logits[0, -1]).item())
checks = []

for step in range(a.steps):
    h = model.model.embed_tokens(mx.array([[token]]))
    for layer_index, (layer, cache) in enumerate(zip(model.layers, caches)):
        residual = h
        normalized = layer.input_layernorm(h)
        attn = layer.self_attn
        q_raw = attn.q_proj(normalized)
        k_raw = attn.k_proj(normalized)
        v_raw = attn.v_proj(normalized)
        q = q_raw.reshape(1, 1, attn.n_heads, -1).transpose(0, 2, 1, 3)
        k = k_raw.reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v_raw.reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        position = cache.offset
        use_rope = attn.use_sliding_window or attn.force_rope
        if use_rope:
            q_ref = attn.rope(q, offset=position)
            k_ref = attn.rope(k, offset=position)
        else:
            q_ref, k_ref = q, k
        router = layer.mlp.gate(normalized) if layer_index else mx.zeros((1, 1, 128), mx.bfloat16)
        if layer_index:
            gates = layer.mlp.gate_act(router.astype(mx.float32))
            ids_ref = mx.stop_gradient(mx.argpartition(-gates, kth=7, axis=-1)[..., :8])
            scores_ref = mx.take_along_axis(gates, ids_ref, axis=-1)
        else:
            ids_ref = mx.zeros((1, 1, 8), mx.uint32)
            scores_ref = mx.zeros((1, 1, 8), mx.float32)

        if layer_index in a.layers:
            got = run_primitives(q_raw, k_raw, v_raw, router, position, use_rope)
            refs = (q_ref, k_ref, v, ids_ref, scores_ref)
            mx.eval(*got, *refs)
            unequal = [
                int(
                    mx.sum(
                        value.view(mx.uint8).reshape(-1)
                        != reference.view(mx.uint8).reshape(-1)
                    ).item()
                )
                for value, reference in zip(got, refs)
            ]
            row = {
                "step": step,
                "layer": layer_index,
                "position": position,
                "use_rope": use_rope,
                "unequal_bytes": unequal,
            }
            checks.append(row)
            print(row, flush=True)
            if unequal[-1]:
                print(
                    {
                        "ids": got[3].tolist(),
                        "scores_got": got[4].tolist(),
                        "scores_ref": scores_ref.tolist(),
                    },
                    flush=True,
                )
            assert unequal == [0] * 5, row

        keys, values = cache.update_and_fetch(k_ref, v)
        attention = mx.fast.scaled_dot_product_attention(q_ref, keys, values, scale=attn.scale)
        ax = attention.transpose(0, 2, 1, 3).reshape(1, 1, -1)
        if layer_index:
            routed = layer.mlp.switch_mlp(normalized, ids_ref)
            moe = (routed * scores_ref[..., None]).sum(-2).astype(routed.dtype)
            h = attn.o_proj(ax) + moe + residual
        else:
            h = attn.o_proj(ax) + layer.mlp(normalized) + residual
        mx.eval(h)
    logits = model.model.embed_tokens.as_linear(model.model.norm(h)) * model.model.args.logit_scale
    token = int(mx.argmax(logits[0, -1]).item())

sources = [
    Path(__file__),
    Path(__file__).resolve().parents[1] / "reference.py",
    Path(__file__).with_name("primitives.py"),
]
result = {
    "device": mx.device_info(),
    "mlx": mx.__version__,
    "args": vars(a),
    "sources": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    },
    "checks": checks,
}
(ROOT / a.output).parent.mkdir(parents=True, exist_ok=True)
(ROOT / a.output).write_text(json.dumps(result, indent=2) + "\n")
print("PASS", len(checks), "whole-pass primitive comparisons", flush=True)
