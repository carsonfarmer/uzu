"""Compare the custom one-pass attention task with the pinned MLX primitive."""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quantized"))

from reference import ROOT, load
from mlx_vlm.models.base import create_attention_mask
from full_layer.attention import run_attention


p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=16)
p.add_argument("--layers", type=int, nargs="+", default=[1, 4, 17, 32, 48])
p.add_argument("--output", default="experiments/north/full_layer/check-attention-v1.json")
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
        mask = create_attention_mask(
            h,
            cache,
            window_size=model.model.window_size if layer.self_attn.use_sliding_window else None,
        )
        normalized = layer.input_layernorm(h)
        attn = layer.self_attn
        q, k, v = attn.q_proj(normalized), attn.k_proj(normalized), attn.v_proj(normalized)
        q = q.reshape(1, 1, attn.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if attn.use_sliding_window or attn.force_rope:
            q = attn.rope(q, offset=cache.offset)
            k = attn.rope(k, offset=cache.offset)
        keys, values = cache.update_and_fetch(k, v)
        reference = mx.fast.scaled_dot_product_attention(q, keys, values, scale=attn.scale, mask=mask)
        if layer_index in a.layers:
            candidate = run_attention(q, cache.keys, cache.values, cache.size())
            mx.eval(reference, candidate)
            unequal = int(mx.sum(reference.view(mx.uint8) != candidate.view(mx.uint8)).item())
            row = {"step": step, "layer": layer_index, "context": keys.shape[2], "unequal_bytes": unequal}
            checks.append(row)
            print(row, flush=True)
            assert unequal == 0, row
        ax = reference.transpose(0, 2, 1, 3).reshape(1, 1, -1)
        if layer_index:
            gates = layer.mlp.gate_act(layer.mlp.gate(normalized).astype(mx.float32))
            expert_ids = mx.stop_gradient(mx.argpartition(-gates, kth=7, axis=-1)[..., :8])
            scores = mx.take_along_axis(gates, expert_ids, axis=-1)
            routed = layer.mlp.switch_mlp(normalized, expert_ids)
            moe = (routed * scores[..., None]).sum(-2).astype(routed.dtype)
            h = attn.o_proj(ax) + moe + h
        else:
            h = attn.o_proj(ax) + layer.mlp(normalized) + h
        mx.eval(h)
    logits = model.model.embed_tokens.as_linear(model.model.norm(h)) * model.model.args.logit_scale
    token = int(mx.argmax(logits[0, -1]).item())

result = {"device": mx.device_info(), "args": vars(a), "checks": checks}
(ROOT / a.output).write_text(json.dumps(result, indent=2) + "\n")
print("PASS", len(checks), "attention comparisons", flush=True)
