"""Isolate the dense prefix layer at the first repeated-decode failure."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quantized"))

from reference import ROOT, load
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.activations import swiglu
from whole_pass.kernel import run_whole_pass
from whole_pass.pack import pack_caches, pack_weights


model, _ = load()
weights = pack_weights(model, start=1, count=48)
model_path = ROOT / "work/models/North-Mini-Code-1.0-4bit"
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
tokenizer.chat_template = (model_path / "chat_template.jinja").read_text()
prompt = "Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values."
ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True,
    reasoning=False,
    skip_thinking=True,
)
ref_caches = model.make_cache()
cand_caches = model.make_cache()
ref_logits = model(mx.array([ids]), cache=ref_caches).logits
cand_logits = model(mx.array([ids]), cache=cand_caches).logits
mx.eval(ref_logits, cand_logits)
token = int(mx.argmax(ref_logits[0, -1]).item())

# Advance through the last known-good step so the next input is the failing token.
for _ in range(27):
    ref_logits = model(mx.array([[token]]), cache=ref_caches).logits
    cand_logits = model(mx.array([[token]]), cache=cand_caches).logits
    mx.eval(ref_logits, cand_logits)
    assert bool(mx.all(ref_logits.view(mx.uint8) == cand_logits.view(mx.uint8)).item())
    token = int(mx.argmax(ref_logits[0, -1]).item())

position = cand_caches[0].offset
assert position == 163 and token == 32092, (position, token)
embedded = model.model.embed_tokens(mx.array([[token]]))
layer0 = model.layers[0]
normalized = layer0.input_layernorm(embedded)
mask = create_attention_mask(normalized, ref_caches[0])
attention_ref = layer0.self_attn(normalized, mask, ref_caches[0])
mlp_ref = layer0.mlp(normalized)
hidden_ref = swiglu(layer0.mlp.gate_proj(normalized), layer0.mlp.up_proj(normalized))
prefix_ref = attention_ref + mlp_ref + embedded
mx.eval(attention_ref, mlp_ref, prefix_ref)

capacity = max(256, max(cache.keys.shape[2] for cache in cand_caches))
keys, values = pack_caches(cand_caches, start=0, count=49, capacity=capacity)
got = run_whole_pass(
    weights,
    embedded,
    keys,
    values,
    position,
    workers=32,
    do_prefix=True,
    return_prefix=True,
)
mx.eval(*got)
prefix, attention, mlp, hidden = got[-4:]


def compare(name, candidate, reference):
    unequal = candidate.view(mx.uint8).reshape(-1) != reference.view(mx.uint8).reshape(-1)
    delta = candidate.astype(mx.float32) - reference.astype(mx.float32)
    candidate_np = np.array(candidate.astype(mx.float32)).reshape(-1)
    reference_np = np.array(reference.astype(mx.float32)).reshape(-1)
    indices = np.flatnonzero(candidate_np != reference_np)[:8]
    return {
        "name": name,
        "unequal_bytes": int(mx.sum(unequal).item()),
        "max_abs": float(mx.max(mx.abs(delta)).item()),
        "first_values": [
            {
                "index": int(index),
                "candidate": float(candidate_np[index]),
                "reference": float(reference_np[index]),
            }
            for index in indices
        ],
    }


result = {
    "position": position,
    "token": token,
    "comparisons": [
        compare("attention", attention, attention_ref),
        compare("mlp", mlp, mlp_ref),
        compare("hidden", hidden, hidden_ref),
        compare("prefix", prefix, prefix_ref),
    ],
}
output = ROOT / "work/check-prefix-step163.json"
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
