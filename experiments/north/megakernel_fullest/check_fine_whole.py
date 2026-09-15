"""Raw-byte gate for packed multi-layer persistent decode."""

import argparse
import hashlib
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
from fine_whole import run as fine_run
from whole_pass.kernel import SIZE
def run_whole_pass(weights,x,k,v,position,**kwargs):
    return fine_run(weights,x,k,v,position,workers=kwargs["workers"],do_head=kwargs["do_head"],do_prefix=kwargs["do_prefix"])
from whole_pass.pack import pack_caches, pack_weights


p = argparse.ArgumentParser()
p.add_argument("--layers", type=int, default=1)
p.add_argument("--workers", type=int, nargs="+", default=[1, 20, 32, 36])
p.add_argument("--schedule", choices=("static", "fine", "queue"), default="queue")
p.add_argument("--prefetch-stages", type=int, default=0)
p.add_argument("--prep-rows", type=int, choices=(32, 64, 128), default=64)
p.add_argument("--oproj-rows", type=int, choices=(32, 64, 128), default=64)
p.add_argument("--router-rows", type=int, choices=(4, 8, 16, 32), default=8)
p.add_argument("--head", action="store_true")
p.add_argument("--lm-rows", type=int, choices=(32, 64, 128, 256, 512, 1024), default=512)
p.add_argument("--prefix", action="store_true")
p.add_argument("--output", default="experiments/north/whole_pass/check-whole-pass-v1.json")
a = p.parse_args()
assert 1 <= a.layers <= 48

model, _ = load()
print("packing", a.layers, "layers", flush=True)
weights = pack_weights(model, start=1, count=a.layers)
print("weights packed", flush=True)

model_path = ROOT / "work/models/North-Mini-Code-1.0-4bit"
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
tokenizer.chat_template = (model_path / "chat_template.jinja").read_text()
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Return only Python code for merging two sorted lists."}],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True,
    reasoning=False,
    skip_thinking=True,
)
ref_caches = model.make_cache()
cand_caches = model.make_cache()
for caches in (ref_caches, cand_caches):
    logits = model(mx.array([prompt_ids]), cache=caches).logits
    mx.eval(logits)
token = int(mx.argmax(logits[0, -1]).item())


def run_layer(layer, h, cache):
    mask = create_attention_mask(
        h,
        cache,
        window_size=model.model.window_size if layer.self_attn.use_sliding_window else None,
    )
    return layer(h, mask, cache)


embedded = model.model.embed_tokens(mx.array([[token]]))
if a.prefix:
    assert a.layers == 48
    ref = embedded
    cand = embedded
    cache_start, cache_count = 0, 49
    position = cand_caches[0].offset
    first_reference_layer = 0
else:
    ref = run_layer(model.layers[0], embedded, ref_caches[0])
    cand = run_layer(model.layers[0], embedded, cand_caches[0])
    mx.eval(ref, cand)
    assert bool(mx.all(ref.view(mx.uint8) == cand.view(mx.uint8)).item())
    cache_start, cache_count = 1, a.layers
    position = cand_caches[1].offset
    first_reference_layer = 1

capacity = max(256, cand_caches[1].keys.shape[2])
packed_keys, packed_values = pack_caches(cand_caches, start=cache_start, count=cache_count, capacity=capacity)
for layer_index in range(first_reference_layer, a.layers + 1):
    ref = run_layer(model.layers[layer_index], ref, ref_caches[layer_index])
mx.eval(ref)

checks = []
for workers in a.workers:
    got = run_whole_pass(
        weights,
        cand,
        packed_keys,
        packed_values,
        position,
        workers=workers,
        schedule=a.schedule,
        prefetch_stages=a.prefetch_stages,
        prep_rows=a.prep_rows,
        oproj_rows=a.oproj_rows,
        router_rows=a.router_rows,
        do_head=a.head,
        lm_rows=a.lm_rows,
        do_prefix=a.prefix,
    )
    out, keys, values, state = got[:4]
    logits = got[4] if a.head else None
    mx.eval(*got)
    out_unequal = int(
        mx.sum(out.view(mx.uint8).reshape(-1) != ref.view(mx.uint8).reshape(-1)).item()
    )
    cache_unequal = []
    for packed, name in ((keys, "keys"), (values, "values")):
        unequal = 0
        for local, layer_index in enumerate(range(cache_start, cache_start + cache_count)):
            reference = getattr(ref_caches[layer_index], name)
            unequal += int(
                mx.sum(
                    packed[local, :, : position + 1].view(mx.uint8).reshape(-1)
                    != reference[0, :, : position + 1].view(mx.uint8).reshape(-1)
                ).item()
            )
        cache_unequal.append(unequal)
    logits_unequal = None
    if a.head:
        assert a.layers == 48
        logits_ref = model.model.embed_tokens.as_linear(model.model.norm(ref)) * model.model.args.logit_scale
        mx.eval(logits_ref)
        logits_unequal = int(
            mx.sum(logits.view(mx.uint8).reshape(-1) != logits_ref.view(mx.uint8).reshape(-1)).item()
        )
    counters = np.array(state)
    row = {
        "layers": a.layers,
        "position": position,
        "workers": workers,
        "schedule": a.schedule,
        "prefetch_stages": a.prefetch_stages,
        "prep_rows": a.prep_rows,
        "oproj_rows": a.oproj_rows,
        "router_rows": a.router_rows,
        "output_unequal_bytes": out_unequal,
        "cache_unequal_bytes": cache_unequal,
        "logits_unequal_bytes": logits_unequal,
        "state": counters[:4].tolist(),
        "task_states_min": int(counters[SIZE:].reshape(a.layers,512)[:,:440].min()),
        "task_states_max": int(counters[SIZE:].reshape(a.layers,512)[:,:440].max()),
        "completed_per_layer": counters[SIZE:].reshape(a.layers,512)[:,511].tolist(),
    }
    checks.append(row)
    print(row, flush=True)
    assert out_unequal == 0 and cache_unequal == [0, 0], row
    assert logits_unequal in (None, 0), row
    if a.schedule == "queue":
        expected_stages = 1 + 6 * a.layers + 6 * int(a.prefix) + 2 * int(a.head)
        assert counters[0] == expected_stages, row
        assert counters[3]==0 and row["task_states_min"]==row["task_states_max"]==2 and row["completed_per_layer"]==[440]*a.layers,row
    else:
        assert np.all(counters == workers), row

sources = [Path(__file__).resolve(),Path(__file__).with_name('fine_whole.py').resolve()]+[p for folder in ('whole_pass','full_layer','quantized','persistent') for p in Path(__file__).resolve().parents[1].joinpath(folder).iterdir() if p.suffix in ('.py','.h')]
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
print("PASS", len(checks), "packed whole-pass comparisons", flush=True)
