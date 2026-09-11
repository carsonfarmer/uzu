"""Opt-in decode path carrying QKV/router preparation across layer boundaries."""

import copy

import mlx.core as mx
import mlx.nn as nn

from kernels import inputs
from persistent.fast import run_fast
from mlx_vlm.models.base import scaled_dot_product_attention

from full_layer.tail_prep import run_tail_prep
from full_layer.static_tail import run_static_tail
from full_layer.attention import run_attention
from full_layer.full_attention import run_full_attention
from full_layer.static_full_attention import run_static_full_attention


class PreparedState:
    def __init__(self):
        self.layer = None
        self.values = None

    def put(self, layer, values):
        self.layer = layer
        self.values = values

    def take(self, layer):
        assert self.layer == layer and self.values is not None, (self.layer, layer)
        values = self.values
        self.layer = None
        self.values = None
        return values


def prepare_attention(attn, queries, keys, values, cache):
    batch, length, _ = queries.shape
    queries = queries.reshape(batch, length, attn.n_heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(batch, length, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(batch, length, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

    if attn.use_sliding_window or attn.force_rope:
        if cache is None:
            queries = attn.rope(queries)
            keys = attn.rope(keys)
        else:
            queries = attn.rope(queries, offset=cache.offset)
            keys = attn.rope(keys, offset=cache.offset)
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)
    return queries, keys, values


def attention_from_prepared(attn, queries, keys, values, mask, cache):
    batch = queries.shape[0]
    length = queries.shape[2]

    sdpa_type = mx.float32 if queries.dtype == mx.float16 else queries.dtype
    output = scaled_dot_product_attention(
        queries.astype(sdpa_type),
        keys,
        values,
        cache=cache,
        scale=attn.scale,
        mask=mask,
    ).astype(queries.dtype)
    return output.transpose(0, 2, 1, 3).reshape(batch, length, -1)


def attention_from_qkv(attn, queries, keys, values, mask, cache):
    queries, keys, values = prepare_attention(attn, queries, keys, values, cache)
    return attention_from_prepared(attn, queries, keys, values, mask, cache)


class CrossLayerDecode(nn.Module):
    def __init__(self, original, layer_index, carrier, next_layer, workers=64, schedule="fine"):
        super().__init__()
        self.original = original
        self.layer_index = layer_index
        self.carrier = carrier
        self.next_layer = next_layer
        self.schedule = schedule
        self.workers = workers
        self.self_attn = original.self_attn
        self.mlp = original.mlp
        assert self.mlp.shared_experts is None
        assert self.mlp.top_k == 8 and not self.mlp.norm_topk_prob

        # Validate the quantized layout while weights are still easy to identify.
        inputs(
            self.mlp,
            mx.zeros((1, 1, 2048), mx.bfloat16),
            mx.zeros((1, 1, 4096), mx.bfloat16),
            mx.zeros((1, 1, 8), mx.uint32),
            self.self_attn.o_proj.weight,
        )

        if next_layer is not None:
            def branch(h, ax, ids, scores, residual):
                data = inputs(self.mlp, h, ax, ids, self.self_attn.o_proj.weight)
                if schedule == "static":
                    return run_static_tail(data, scores, residual, next_layer, workers=workers)
                return run_tail_prep(
                    data,
                    scores,
                    residual,
                    next_layer,
                    workers=max(workers, 48) if schedule.startswith("full") else workers,
                    prefetch=schedule == "prefetch",
                )
        else:
            def branch(h, ax, ids, scores, residual):
                data = inputs(self.mlp, h, ax, ids, self.self_attn.o_proj.weight)
                return (run_fast(data, scores, residual, workers=workers),)
        self.branch = mx.compile(branch)

        if schedule.startswith("full") and next_layer is not None:
            def full_branch(h, q, key_cache, value_cache, params, ids, scores, residual):
                full_runner = run_static_full_attention if schedule.startswith("full_static") else run_full_attention
                kwargs = {
                    "workers": workers,
                    "do_prep": not schedule.endswith("native"),
                }
                if full_runner is run_full_attention:
                    kwargs["front_first"] = schedule in ("full_front", "full_native")
                result = full_runner(
                    self.mlp,
                    h,
                    q,
                    key_cache,
                    value_cache,
                    params,
                    ids,
                    self.self_attn.o_proj.weight,
                    scores,
                    residual,
                    next_layer,
                    **kwargs,
                )
                if not schedule.endswith("native"):
                    return result
                out = result
                norm = next_layer.input_layernorm(out)
                return (
                    out,
                    norm,
                    next_layer.self_attn.q_proj(norm),
                    next_layer.self_attn.k_proj(norm),
                    next_layer.self_attn.v_proj(norm),
                    next_layer.mlp.gate(norm),
                )

            self.full_branch = mx.compile(full_branch)
        else:
            self.full_branch = None

    def __call__(self, x, mask=None, cache=None):
        if x.shape != (1, 1, 2048):
            return self.original(x, mask, cache)

        if self.layer_index == 1:
            h = self.original.input_layernorm(x)
            q = self.self_attn.q_proj(h)
            k = self.self_attn.k_proj(h)
            v = self.self_attn.v_proj(h)
            router = self.mlp.gate(h)
        else:
            h, q, k, v, router = self.carrier.take(self.layer_index)

        queries, keys, values = prepare_attention(self.self_attn, q, k, v, cache)
        gates = self.mlp.gate_act(router.astype(mx.float32))
        ids = mx.stop_gradient(mx.argpartition(-gates, kth=7, axis=-1)[..., :8])
        scores = mx.take_along_axis(gates, ids, axis=-1)

        can_run_one_pass = (
            self.schedule.startswith("full")
            and cache is not None
            and mask is None
            and 0 < cache.size() < 1024
            and hasattr(cache, "keys")
            and hasattr(cache, "values")
        )
        if can_run_one_pass and self.full_branch is not None:
            head_stride = cache.keys.shape[2] * 128
            params = mx.array([cache.size(), head_stride, head_stride], dtype=mx.uint32)
            values = self.full_branch(h, queries, cache.keys, cache.values, params, ids, scores, x)
        else:
            if can_run_one_pass:
                attention = run_attention(queries, cache.keys, cache.values, cache.size())
                ax = attention.transpose(0, 2, 1, 3).reshape(1, 1, -1)
            else:
                ax = attention_from_prepared(self.self_attn, queries, keys, values, mask, cache)
            values = self.branch(h, ax, ids, scores, x)
        if self.next_layer is not None:
            out, norm, next_q, next_k, next_v, next_router = values
            self.carrier.put(
                self.layer_index + 1,
                (norm, next_q, next_k, next_v, next_router),
            )
            return out
        return values[0]


def cross_layer_path(model, workers=64, schedule="fine"):
    assert schedule in (
        "fine", "static", "prefetch", "full", "full_front", "full_native",
        "full_static", "full_static_native",
    )
    original = list(model.layers)
    carrier = PreparedState()
    cross = [original[0]]
    for layer_index in range(1, len(original)):
        next_layer = original[layer_index + 1] if layer_index + 1 < len(original) else None
        cross.append(CrossLayerDecode(original[layer_index], layer_index, carrier, next_layer, workers, schedule))
    return original, cross
