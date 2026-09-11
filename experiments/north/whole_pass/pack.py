"""Pack North's per-layer arrays for pointer arithmetic inside one Metal kernel."""

import mlx.core as mx


WEIGHT_NAMES = (
    "norm_w",
    "qw",
    "kw",
    "vw",
    "rw",
    "aw",
    "up",
    "us",
    "ub",
    "gate",
    "gs",
    "gb",
    "down",
    "ds",
    "db",
)

HEAD_NAMES = ("final_norm_w", "lm_w", "lm_s", "lm_b")
PREFIX_NAMES = ("prefix_dense", "prefix_q", "prefix_meta")


def pack_weights(model, start=1, count=2):
    layers = model.layers[start : start + count]
    assert len(layers) == count and all(hasattr(layer.mlp, "switch_mlp") for layer in layers)

    arrays = {
        "norm_w": [layer.input_layernorm.weight for layer in layers],
        "qw": [layer.self_attn.q_proj.weight for layer in layers],
        "kw": [layer.self_attn.k_proj.weight for layer in layers],
        "vw": [layer.self_attn.v_proj.weight for layer in layers],
        "rw": [layer.mlp.gate.weight for layer in layers],
        "aw": [layer.self_attn.o_proj.weight for layer in layers],
    }
    for short, module_name, field in (
        ("up", "up_proj", "weight"),
        ("us", "up_proj", "scales"),
        ("ub", "up_proj", "biases"),
        ("gate", "gate_proj", "weight"),
        ("gs", "gate_proj", "scales"),
        ("gb", "gate_proj", "biases"),
        ("down", "down_proj", "weight"),
        ("ds", "down_proj", "scales"),
        ("db", "down_proj", "biases"),
    ):
        arrays[short] = [
            getattr(getattr(layer.mlp.switch_mlp, module_name), field) for layer in layers
        ]

    packed = {}
    for name in WEIGHT_NAMES:
        packed[name] = mx.stack(arrays[name])
        mx.eval(packed[name])
    head = model.model.embed_tokens
    assert (head.bits, head.group_size, head.mode) == (4, 64, "affine")
    packed.update(
        final_norm_w=model.model.norm.weight,
        lm_w=head.weight,
        lm_s=head.scales,
        lm_b=head.biases,
    )
    prefix = model.layers[0]
    dense = prefix.mlp
    packed.update(
        prefix_dense=mx.concatenate(
            [
                value.reshape(-1)
                for value in (
                    prefix.input_layernorm.weight,
                    prefix.self_attn.q_proj.weight,
                    prefix.self_attn.k_proj.weight,
                    prefix.self_attn.v_proj.weight,
                    prefix.self_attn.o_proj.weight,
                )
            ]
        ),
        prefix_q=mx.concatenate(
            [
                value.reshape(-1)
                for value in (
                    dense.up_proj.weight,
                    dense.gate_proj.weight,
                    dense.down_proj.weight,
                )
            ]
        ),
        prefix_meta=mx.concatenate(
            [
                value.reshape(-1)
                for value in (
                    dense.up_proj.scales,
                    dense.up_proj.biases,
                    dense.gate_proj.scales,
                    dense.gate_proj.biases,
                    dense.down_proj.scales,
                    dense.down_proj.biases,
                )
            ]
        ),
    )
    mx.eval(*[packed[name] for name in HEAD_NAMES + PREFIX_NAMES])
    return packed


def pack_caches(caches, start=1, count=2, capacity=256):
    selected = caches[start : start + count]
    assert len(selected) == count

    def pad(value):
        value = value[0]
        used_capacity = value.shape[1]
        assert used_capacity <= capacity
        if used_capacity == capacity:
            return value
        zeros = mx.zeros((4, capacity - used_capacity, 128), dtype=value.dtype)
        return mx.concatenate([value, zeros], axis=1)

    keys = mx.stack([pad(cache.keys) for cache in selected])
    values = mx.stack([pad(cache.values) for cache in selected])
    mx.eval(keys, values)
    return keys, values


def grow_caches(keys, values, capacity):
    assert keys.shape == values.shape and keys.ndim == 4
    current = keys.shape[2]
    assert capacity > current
    shape = (keys.shape[0], keys.shape[1], capacity - current, keys.shape[3])
    keys = mx.concatenate([keys, mx.zeros(shape, dtype=keys.dtype)], axis=2)
    values = mx.concatenate([values, mx.zeros(shape, dtype=values.dtype)], axis=2)
    mx.eval(keys, values)
    return keys, values
