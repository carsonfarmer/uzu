"""Isolate serialization, task ordering, and cross-branch fusion in the existing exact control.

Run from the repository root with work/north-venv/bin/python. Existing production
and reference sources are imported unchanged. GPU experiments take a shared lock.
"""

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

NORTH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NORTH))
sys.path.insert(0, str(NORTH / "quantized"))
from reference import ROOT, load
from kernels import HEADER, NAMES, inputs
from exact import down, run_exact
from patch import DecodeBranch


# Identical math functions, tile shapes, and final group barrier to exact.front.
# With WORKERS=160, that front has exactly one task per group: 96 expert tasks
# and 64 attention-output tasks. Splitting moves those same tasks to two grids.
SPLIT_SOURCE = """
uint group=threadgroup_position_in_grid.x;
uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
threadgroup float tile[64];
BODY
threadgroup_barrier(mem_flags::mem_threadgroup);
"""
split_hidden = mx.fast.metal_kernel(
    name="north_ablation_split_hidden", input_names=NAMES,
    output_names=["hidden"], header="#define NCHUNK 64\n" + HEADER,
    source=SPLIT_SOURCE.replace(
        "BODY", "north_hidden(group,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);"),
)
split_attention = mx.fast.metal_kernel(
    name="north_ablation_split_attention", input_names=NAMES,
    output_names=["attention"], header="#define NCHUNK 64\n" + HEADER,
    source=SPLIT_SOURCE.replace(
        "BODY", "north_attention(group,sg,lane,ax,aw,attention);"),
)


serial_attention = mx.fast.metal_kernel(
    name="north_ablation_serial_attention", input_names=NAMES + ["hidden_dependency"],
    output_names=["attention"], header="#define NCHUNK 64\n" + HEADER,
    source=SPLIT_SOURCE.replace(
        "BODY", "north_attention(group,sg,lane,ax,aw,attention);"),
)


def run_branch(data, scores, residual, mode):
    if mode in ("mixed", "blocked"):
        return run_exact(data, scores, residual, workers=160, rows=16,
                         interleave=mode == "mixed")
    assert mode in ("separate", "serial")
    hidden = split_hidden(
        inputs=data, grid=(96 * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(8, 768)], output_dtypes=[mx.bfloat16],
    )[0]
    attention_kernel = serial_attention if mode == "serial" else split_attention
    attention = attention_kernel(
        inputs=data + [hidden] if mode == "serial" else data, grid=(64 * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1, 1, 2048)], output_dtypes=[mx.bfloat16],
    )[0]
    return down(
        inputs=[hidden, data[2], *data[9:12], scores, attention, residual],
        template=[("ROWS", 16)], grid=(16384, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(1, 1, 2048), (8, 2048)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )[0]


def make_path(original, mode):
    if mode == "original":
        return original
    result = [original[0]]
    for layer in original[1:]:
        wrapped = DecodeBranch(layer, "exact", workers=160, rows=16)

        def branch(h, ax, residual, ids, scores, layer=layer):
            data = inputs(layer.mlp, h, ax, ids, layer.self_attn.o_proj.weight)
            return run_branch(data, scores, residual, mode)

        wrapped.branch = mx.compile(branch)
        result.append(wrapped)
    return result


PROMPTS = {
    "short": "Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.",
    "rust": "Return only Rust code for finding the first duplicate in a u32 slice, with tests for empty input and duplicates.",
}
MODES = ["original", "mixed", "blocked", "separate", "serial"]


def byte_digest(value):
    return np.array(value.view(mx.uint8)).tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--order-offset", type=int, default=0)
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=list(PROMPTS))
    parser.add_argument("--timing", choices=["step", "generation"], default="step")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 2 <= args.tokens <= 128 and args.runs > 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects retained data from accidental replacement.
    stream = args.output.open("x")
    gpu_lock = open("/tmp/north-metal-research-gpu.lock", "a+")
    print("Waiting for exclusive GPU experiment lock", flush=True)
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)

    def save(row):
        stream.write(json.dumps(row) + "\n")
        stream.flush()

    sources = [Path(__file__), NORTH / "reference.py"]
    sources += [NORTH / "quantized" / name for name in
                ("exact.py", "kernels.py", "kernel.h", "patch.py")]
    sources += sorted((ROOT / "work/mlx-vlm/mlx_vlm/models/cohere2_moe").glob("*.py"))
    prior_path = NORTH / "correctness/results/full-exact-v1.jsonl"
    sources.append(prior_path)
    sources += [ROOT / "work/mlx-src/mlx/backend/metal" / name
                for name in ("device.cpp", "custom_kernel.cpp")]
    # Source inspection establishes that all custom-kernel inputs are registered
    # with the hazard tracker, even when unused by the MSL arithmetic. The extra
    # hidden_dependency input therefore forces a GPU memory barrier after hidden
    # production and before attention dispatch, without a CPU synchronization.
    # No extra arithmetic or deliberate spin/delay is added.
    save({
        "kind": "provenance", "args": {**vars(args), "output": str(args.output)},
        "device": mx.device_info(), "mlx": mx.__version__,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "environment": {k: v for k, v in os.environ.items() if k.startswith("MLX_")},
        "design": {
            "mixed": "Existing exact control: interleaved expert/attention task IDs in one front launch, then the unchanged down/join launch.",
            "blocked": "Same two launches and all math; expert task IDs first, attention task IDs second within the front launch. Does NOT enforce branch serialization.",
            "separate": "Same expert/activation and attention tile functions in separate front launches, then the unchanged down/join launch. Three launches total.",
            "serial": "Same three launches as separate; attention has an extra hidden input dependency. MLX hazard tracking enforces expert-front completion before attention. No extra arithmetic or CPU evaluation barrier.",
            "workers": 160, "down_rows": 16,
            "comparison_limit": "Separate permits native MLX cross-dispatch concurrency; serial prevents it through a declared dependency. Mixed-vs-separate changes launch count and grid/resource scheduling together. Separate-vs-serial tests removal of a forced dependency, not measured hardware overlap.",
        },
    })
    print("Loading one shared model", flush=True)
    model, config = load()
    original = list(model.layers)
    paths = {mode: make_path(original, mode) for mode in MODES}
    model_path = ROOT / "work/models/North-Mini-Code-1.0-4bit"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.chat_template = (model_path / "chat_template.jinja").read_text()
    eos = config["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)
    prior = {r["prompt"]: r["token_ids"] for r in
             map(json.loads, prior_path.read_text().splitlines()) if r["kind"] == "completion"}

    def prefill(ids, mode):
        model.model.layers = paths[mode]
        cache = model.make_cache()
        for pos in range(0, len(ids), 256):
            logits = model(mx.array([ids[pos:pos + 256]]), cache=cache).logits
            mx.eval(logits)
        return {"cache": cache, "token": int(mx.argmax(logits[0, -1]).item()),
                "tokens": [int(mx.argmax(logits[0, -1]).item())], "steps": []}

    def advance(state, mode):
        model.model.layers = paths[mode]
        assert state["token"] not in eos, "Unexpected early EOS changes measured workload"
        start = time.perf_counter()
        logits = model(mx.array([[state["token"]]]), cache=state["cache"]).logits
        token = int(mx.argmax(logits[0, -1]).item())
        duration = time.perf_counter() - start
        state["steps"].append(duration)
        state["tokens"].append(token)
        state["token"] = token
        return logits

    def order(index):
        shift = (index + args.order_offset) % len(MODES)
        return MODES[shift:] + MODES[:shift]

    for label in args.prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPTS[label]}], tokenize=True,
            return_dict=False, add_generation_prompt=True, reasoning=False, skip_thinking=True,
        )
        # Full-vocabulary byte equality is checked outside all performance runs.
        states = {mode: prefill(ids, mode) for mode in MODES}
        assert all(s["tokens"] == prior[label][:1] for s in states.values())
        for step in range(args.tokens - 1):
            logits = {mode: advance(states[mode], mode) for mode in order(step)}
            reference = byte_digest(logits["original"])
            for mode in MODES[1:]:
                assert logits[mode].shape == logits["original"].shape
                assert logits[mode].dtype == logits["original"].dtype
                assert byte_digest(logits[mode]) == reference, (label, step, mode, "logits differ")
            if step % 32 == 0:
                print(label, "exact logit step", step, flush=True)
        for mode in MODES:
            assert states[mode]["tokens"] == prior[label][:args.tokens], (label, mode, "tokens differ")
        save({"kind": "correctness", "prompt": label, "prompt_tokens": len(ids),
              "decode_steps_per_variant": args.tokens - 1, "variants": MODES[1:],
              "full_logits_bitwise_equal": True, "logit_shape": list(logits["original"].shape),
              "logit_dtype": str(logits["original"].dtype), "token_ids": states["original"]["tokens"]})
        del states, logits
        gc.collect()

        for rep in range(-1, args.runs):
            records = {}
            if args.timing == "step":
                states = {mode: prefill(ids, mode) for mode in order(rep + 1)}
                for step in range(args.tokens - 1):
                    for mode in order(step + rep + 1):
                        advance(states[mode], mode)
                records = states
            else:
                for mode in order(rep + 1):
                    state = prefill(ids, mode)
                    for _ in range(args.tokens - 1):
                        advance(state, mode)
                    records[mode] = {k: v for k, v in state.items() if k != "cache"}
                    del state
                    gc.collect()
            for mode, state in records.items():
                assert state["tokens"] == prior[label][:args.tokens], (label, rep, mode)
                samples = state["steps"]
                row = {"kind": "generation", "prompt": label, "prompt_tokens": len(ids),
                       "variant": mode, "repetition": rep, "warmup": rep < 0,
                       "decode_steps": len(samples), "decode_seconds": sum(samples),
                       "decode_tokens_per_second": len(samples) / sum(samples),
                       "step_seconds": samples, "token_ids": state["tokens"]}
                save(row)
                print(label, mode, rep, round(row["decode_tokens_per_second"], 3), "tok/s", flush=True)
            del records, state
            if args.timing == "step":
                del states
            gc.collect()
    model.model.layers = original
    stream.close()
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    for label in args.prompts:
        print(label, {mode: statistics.median(r["decode_tokens_per_second"] for r in rows
              if r.get("kind") == "generation" and r["prompt"] == label
              and r["variant"] == mode and not r["warmup"]) for mode in MODES}, flush=True)
    # Keep the lock until process teardown releases the loaded GPU model.


if __name__ == "__main__":
    main()
