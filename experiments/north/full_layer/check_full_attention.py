"""Raw-byte and scheduler gate for the persistent attention-plus-MoE layer."""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quantized"))

from exact import run_exact
from full_layer.full_attention import ERROR, TOTAL_TASKS, VISITS, run_full_attention
from full_layer.static_full_attention import run_static_full_attention
from kernels import inputs
from reference import ROOT, load
from mlx_vlm.models.base import create_attention_mask


p = argparse.ArgumentParser()
p.add_argument("--layer", type=int, default=1)
p.add_argument("--steps", type=int, default=2)
p.add_argument("--workers", type=int, nargs="+", default=[1, 20, 32])
p.add_argument("--debug-zero-query", action="store_true")
p.add_argument("--static", action="store_true")
p.add_argument("--output", default="experiments/north/full_layer/check-full-attention-v1.json")
a = p.parse_args()

assert 1 <= a.layer < 48
model, _ = load()
model_path = ROOT / "work/models/North-Mini-Code-1.0-4bit"
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
tokenizer.chat_template = (model_path / "chat_template.jinja").read_text()
prompt = "Return only Python code for merging two sorted lists."
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True,
    reasoning=False,
    skip_thinking=True,
)
caches = model.make_cache()
logits = model(mx.array([prompt_ids]), cache=caches).logits
mx.eval(logits)
token = int(mx.argmax(logits[0, -1]).item())
checks = []

for step in range(a.steps):
    h = model.model.embed_tokens(mx.array([[token]]))
    candidate_fixture = None
    for layer_index, (layer, cache) in enumerate(zip(model.layers, caches)):
        mask = create_attention_mask(
            h,
            cache,
            window_size=model.model.window_size if layer.self_attn.use_sliding_window else None,
        )
        residual = h
        normalized = layer.input_layernorm(h)
        attn = layer.self_attn
        q = attn.q_proj(normalized).reshape(1, 1, attn.n_heads, -1).transpose(0, 2, 1, 3)
        k = attn.k_proj(normalized).reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = attn.v_proj(normalized).reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if attn.use_sliding_window or attn.force_rope:
            q = attn.rope(q, offset=cache.offset)
            k = attn.rope(k, offset=cache.offset)
        if a.debug_zero_query and layer_index == a.layer:
            q = mx.zeros_like(q)
        keys, values = cache.update_and_fetch(k, v)
        attention = mx.fast.scaled_dot_product_attention(q, keys, values, scale=attn.scale, mask=mask)
        ax = attention.transpose(0, 2, 1, 3).reshape(1, 1, -1)

        if layer_index:
            router = layer.mlp.gate(normalized)
            gates = layer.mlp.gate_act(router.astype(mx.float32))
            expert_ids = mx.stop_gradient(mx.argpartition(-gates, kth=7, axis=-1)[..., :8])
            scores = mx.take_along_axis(gates, expert_ids, axis=-1)
            data = inputs(layer.mlp, normalized, ax, expert_ids, attn.o_proj.weight)
            out_ref, hidden_ref, routed_ref, projected_ref = run_exact(
                data, scores, residual, workers=160, rows=16, return_parts=True
            )

            if layer_index == a.layer:
                next_layer = model.layers[layer_index + 1]
                norm_ref = next_layer.input_layernorm(out_ref)
                refs = (
                    out_ref,
                    norm_ref,
                    next_layer.self_attn.q_proj(norm_ref),
                    next_layer.self_attn.k_proj(norm_ref),
                    next_layer.self_attn.v_proj(norm_ref),
                    next_layer.mlp.gate(norm_ref),
                )
                head_stride = cache.keys.shape[2] * 128
                params = mx.array([cache.size(), head_stride, head_stride], dtype=mx.uint32)
                candidate_fixture = (
                    layer,
                    normalized,
                    q,
                    cache.keys,
                    cache.values,
                    params,
                    expert_ids,
                    scores,
                    residual,
                    next_layer,
                    refs,
                    attention,
                    (hidden_ref, projected_ref, routed_ref),
                )
            h = out_ref
        else:
            h = attn.o_proj(ax) + layer.mlp(normalized) + residual
        mx.eval(h)

    assert candidate_fixture is not None
    (
        layer,
        normalized,
        q,
        kc,
        vc,
        params,
        expert_ids,
        scores,
        residual,
        next_layer,
        refs,
        attention_ref,
        branch_refs,
    ) = candidate_fixture
    mx.eval(normalized, q, kc, vc, params, expert_ids, scores, residual, *refs)
    for workers in a.workers:
        runner = run_static_full_attention if a.static else run_full_attention
        got = runner(
            layer.mlp,
            normalized,
            q,
            kc,
            vc,
            params,
            expert_ids,
            layer.self_attn.o_proj.weight,
            scores,
            residual,
            next_layer,
            workers=workers,
            return_state=True,
        )
        mx.eval(*got)
        unequal = [
            int(mx.sum(value.view(mx.uint8) != reference.view(mx.uint8)).item())
            for value, reference in zip(got[:6], refs)
        ]
        attention_unequal = int(
            mx.sum(got[7].view(mx.uint8) != attention_ref.view(mx.uint8)).item()
        )
        branch_unequal = [
            int(mx.sum(value.view(mx.uint8) != reference.view(mx.uint8)).item())
            for value, reference in zip(got[8:11], branch_refs)
        ]
        state = np.array(got[6])
        visits = state[VISITS : VISITS + TOTAL_TASKS]
        row = {
            "step": step,
            "layer": a.layer,
            "context": int(params[0].item()),
            "workers": workers,
            "unequal_bytes": unequal,
            "attention_unequal_bytes": attention_unequal,
            "branch_unequal_bytes": branch_unequal,
            "errors": int(state[ERROR]),
            "task_visits_min": int(visits.min()),
            "task_visits_max": int(visits.max()),
        }
        checks.append(row)
        print(row, flush=True)
        if attention_unequal:
            candidate_values = np.array(got[7].astype(mx.float32)).reshape(-1)
            reference_values = np.array(attention_ref.astype(mx.float32)).reshape(-1)
            uniform_values = np.array(
                mx.mean(vc[:, :, : int(params[0].item()), :].astype(mx.float32), axis=2)
            ).reshape(-1)
            print(
                {
                    "attention_candidate_head": candidate_values[:16].tolist(),
                    "attention_reference_head": reference_values[:16].tolist(),
                    "attention_uniform_head": uniform_values[:16].tolist(),
                    "attention_candidate_minmax": [float(candidate_values.min()), float(candidate_values.max())],
                    "attention_reference_minmax": [float(reference_values.min()), float(reference_values.max())],
                },
                flush=True,
            )
        assert unequal == [0] * 6, row
        assert row["errors"] == 0 and np.all(visits == 1), row

    logits = model.model.embed_tokens.as_linear(model.model.norm(h)) * model.model.args.logit_scale
    token = int(mx.argmax(logits[0, -1]).item())

result = {"device": mx.device_info(), "args": vars(a), "checks": checks}
(ROOT / a.output).write_text(json.dumps(result, indent=2) + "\n")
print("PASS", len(checks), "full-attention layer comparisons", flush=True)
