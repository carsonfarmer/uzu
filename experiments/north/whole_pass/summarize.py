"""Verify the safe canonical artifacts and emit the publishable result summary."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "summary-v1.json"

BASELINES = (
    "benchmark-safe-baselines-short128-v1.jsonl",
    "benchmark-safe-baselines-short128-v2.jsonl",
)
SAFE_RUNS = (
    "benchmark-safe-megakernel-short128-v1.jsonl",
    "benchmark-safe-megakernel-short128-v2.jsonl",
)
ABLATIONS = "benchmark-safe-profiles-short128-v1.jsonl"
CHECKS = (
    "check-primitives-v1.json",
    "check-safe-megakernel-v1.json",
    "check-safe-prefetch10-v1.json",
)
SEQUENCES = (
    "full-safe-megakernel-python-short128-v1.jsonl",
    "full-safe-megakernel-rust-short128-v1.jsonl",
)
REJECTED_BENCHMARK = "history/benchmark-static-success-short128-v1.jsonl"
REJECTED_BASELINE = "history/benchmark-static-era-baselines-short128-v1.jsonl"
STALLED_STATIC_RUN = "history/full-static-python-aborted-after-deadlock-short128-v1.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / name).read_text().splitlines()]


def provenance(name: str) -> dict:
    path = HERE / name
    data = json.loads(path.read_text()) if path.suffix == ".json" else jsonl(name)[0]
    for relative, expected in data["sources"].items():
        actual = sha256(ROOT / relative)
        assert actual == expected, (name, relative, expected, actual)
    return data


def measured(
    name: str, variant: str | None = None, profile: str | None = None
) -> list[dict]:
    rows = [
        row
        for row in jsonl(name)
        if row.get("kind") == "generation" and not row["warmup"]
    ]
    if variant is not None:
        rows = [row for row in rows if row["variant"] == variant]
    if profile is not None:
        rows = [row for row in rows if row["profile"] == profile]
    assert len(rows) == 3, (name, variant, profile, len(rows))
    return rows


def rate_stats(rows: list[dict]) -> dict:
    values = [row["decode_tokens_per_second"] for row in rows]
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "samples": values,
    }


canonical = (*BASELINES, *SAFE_RUNS, ABLATIONS, *CHECKS, *SEQUENCES)
records = {name: provenance(name) for name in canonical}

baseline_rows = {
    variant: [row for name in BASELINES for row in measured(name, variant=variant)]
    for variant in ("original", "exact")
}
safe_rows = [row for name in SAFE_RUNS for row in measured(name, profile="tuned")]
ablation_rows = {
    profile: measured(ABLATIONS, profile=profile)
    for profile in ("tuned", "coarse", "prefetch1", "prefetch10")
}

python_completion = next(
    row for row in jsonl(SEQUENCES[0]) if row.get("kind") == "completion"
)
reference_tokens = tuple(python_completion["token_ids"])
all_benchmark_rows = [
    *baseline_rows["original"],
    *baseline_rows["exact"],
    *safe_rows,
    *(row for rows in ablation_rows.values() for row in rows),
]
assert len(all_benchmark_rows) == 30
assert all(tuple(row["token_ids"]) == reference_tokens for row in all_benchmark_rows)

sequence_checks = {}
for name in SEQUENCES:
    rows = jsonl(name)
    steps = [row for row in rows if row.get("kind") == "step"]
    completion = next(row for row in rows if row.get("kind") == "completion")
    assert len(steps) == completion["exact_decode_steps"] == 127
    assert all(row["bitwise"] for row in steps)
    sequence_checks[name] = {
        "prompt_tokens": completion["prompt_tokens"],
        "exact_decode_steps": len(steps),
        "first_position": steps[0]["position"],
        "last_position": steps[-1]["position"],
        "token_ids_sha256": hashlib.sha256(
            json.dumps(completion["token_ids"], separators=(",", ":")).encode()
        ).hexdigest(),
    }

primitive = records[CHECKS[0]]
assert len(primitive["checks"]) == 18
assert all(
    row["unequal_bytes"] == [0, 0, 0, 0, 0] for row in primitive["checks"]
)

whole_checks = {}
for name in CHECKS[1:]:
    rows = records[name]["checks"]
    for row in rows:
        assert row["schedule"] == "queue"
        assert row["output_unequal_bytes"] == 0
        assert row["cache_unequal_bytes"] == [0, 0]
        assert row["logits_unequal_bytes"] == 0
        assert row["state"][0] == 297
        assert row["state"][1] == row["state"][2]
    whole_checks[name] = {
        "workers": [row["workers"] for row in rows],
        "comparisons": len(rows),
        "final_phase": 297,
        "claimed_tasks_equal_completed_tasks": True,
        "all_output_cache_and_logit_bytes_exact": True,
    }

stats = {
    "original_mlx": rate_stats(baseline_rows["original"]),
    "exact_fused": rate_stats(baseline_rows["exact"]),
    "safe_megakernel": rate_stats(safe_rows),
    "ablation_tuned": rate_stats(ablation_rows["tuned"]),
    "coarse_tiles": rate_stats(ablation_rows["coarse"]),
    "prefetch_1": rate_stats(ablation_rows["prefetch1"]),
    "prefetch_10": rate_stats(ablation_rows["prefetch10"]),
}
rates = {name: values["median"] for name, values in stats.items()}
tuned = rates["safe_megakernel"]
pairwise = []
for index, (baseline_name, safe_name) in enumerate(zip(BASELINES, SAFE_RUNS), 1):
    baseline_rate = statistics.median(
        row["decode_tokens_per_second"]
        for row in measured(baseline_name, variant="original")
    )
    safe_rate = statistics.median(
        row["decode_tokens_per_second"]
        for row in measured(safe_name, profile="tuned")
    )
    pairwise.append(
        {
            "pair": index,
            "safe_megakernel": safe_rate,
            "original_mlx": baseline_rate,
            "safe_vs_original_percent": 100 * (safe_rate / baseline_rate - 1),
        }
    )

rejected_benchmark_rows = measured(REJECTED_BENCHMARK)
rejected_baseline_rows = measured(REJECTED_BASELINE, variant="original")
rejected_rate = statistics.median(
    row["decode_tokens_per_second"] for row in rejected_benchmark_rows
)
rejected_stock_rate = statistics.median(
    row["decode_tokens_per_second"] for row in rejected_baseline_rows
)
stalled_rows = jsonl(STALLED_STATIC_RUN)
assert stalled_rows[0]["args"]["schedule"] == "static"
assert [row["step"] for row in stalled_rows[1:]] == [1, 2]
assert all(row["bitwise"] for row in stalled_rows[1:])
assert not any(row.get("kind") == "completion" for row in stalled_rows)

summary = {
    "schema": "north-apple-safe-megakernel-summary-v1",
    "device": records[BASELINES[0]]["device"],
    "mlx": records[BASELINES[0]]["mlx"],
    "model": {
        "id": "mlx-community/North-Mini-Code-1.0-4bit",
        "revision": "dfbe084dfa26e241345af99ca32848f38fd865f9",
    },
    "cohere_reference_commit": "67d0b9ca22ea3652796b715d1d1863459e0e2c3c",
    "scope": {
        "batch_size": 1,
        "transformer_layers_in_dispatch": 49,
        "moe_layers_in_dispatch": 48,
        "metal_dispatches_for_transformer_and_logits_per_decode_step": 1,
        "includes": [
            "RMSNorm",
            "QKV and router projections",
            "RoPE",
            "KV updates",
            "attention",
            "top-8 routing",
            "MoE",
            "residual joins",
            "final RMSNorm",
            "262144-vocabulary logits",
        ],
        "outside_dispatch": ["token embedding lookup", "argmax", "prefill"],
    },
    "scheduler": {
        "kind": "dynamic ready-task queue",
        "phases": 297,
        "threadgroups": 36,
        "threadgroup_width": 256,
        "tested_threadgroup_counts": whole_checks[CHECKS[1]]["workers"],
        "static_global_barrier_scheduler": "rejected after a repeated-decode deadlock",
    },
    "correctness": {
        "primitive_cases": 18,
        "whole_model_worker_counts": whole_checks[CHECKS[1]]["workers"],
        "consecutive_full_logit_comparisons": sum(
            item["exact_decode_steps"] for item in sequence_checks.values()
        ),
        "sequence_checks": sequence_checks,
        "measured_benchmark_runs_with_identical_tokens": len(all_benchmark_rows),
        "unique_measured_benchmark_token_sequences": len(
            {tuple(row["token_ids"]) for row in all_benchmark_rows}
        ),
        "all_checked_logits_and_caches_bitwise_exact": True,
    },
    "performance": {
        "unit": "decode_tokens_per_second",
        "statistic": "median of 6 headline runs across 2 processes; median of 3 step-interleaved ablation runs; each process has 1 warmup",
        "rates": rates,
        "sample_statistics": stats,
        "adjacent_process_pairs": pairwise,
        "safe_megakernel_vs_original_percent": 100
        * (tuned / rates["original_mlx"] - 1),
        "safe_megakernel_vs_exact_fused_percent": 100
        * (tuned / rates["exact_fused"] - 1),
        "tuned_vs_coarse_percent": 100
        * (rates["ablation_tuned"] / rates["coarse_tiles"] - 1),
        "prefetch_1_vs_tuned_percent": 100
        * (rates["prefetch_1"] / rates["ablation_tuned"] - 1),
        "prefetch_10_vs_tuned_percent": 100
        * (rates["prefetch_10"] / rates["ablation_tuned"] - 1),
    },
    "method": {
        "prompt_tokens": records[SAFE_RUNS[0]]["prompt_tokens"],
        "generated_tokens_per_run": 128,
        "decode_steps_per_run": 127,
        "baseline_and_megakernel_processes": "two separate runs of each path, sequenced safe/baseline/safe/baseline",
        "reason": "Keeping both weight layouts live creates memory-residency stalls at about 34 GB.",
        "excluded": ["model load", "weight packing", "tokenization", "prefill", "warmup"],
        "included": ["embedding lookup", "kernel dispatch", "KV carry and growth", "argmax"],
        "ablation_control": "Tuned, coarse, prefetch-1, and prefetch-10 paths rotate after every decode step in one packed process.",
        "interpretation": "The safe path is 0.98% slower in the combined median and stays within 1.3% in both adjacent process pairs.",
    },
    "rejected_result": {
        "scheduler": "static all-grid spin barrier",
        "successful_run_rate": rejected_rate,
        "successful_run_stock_rate": rejected_stock_rate,
        "successful_run_change_percent": 100
        * (rejected_rate / rejected_stock_rate - 1),
        "reason_rejected": "The same scheduler later deadlocked after two exact steps with 32 groups and after one exact step with 20 groups.",
        "status": "preserved as an unsafe historical result; excluded from canonical performance claims",
        "evidence": {
            "successful_benchmark": REJECTED_BENCHMARK,
            "successful_baseline": REJECTED_BASELINE,
            "stalled_exact_run": STALLED_STATIC_RUN,
            "incident_record": "history/STATIC_SCHEDULER_FAILURE.md",
            "sha256": {
                REJECTED_BENCHMARK: sha256(HERE / REJECTED_BENCHMARK),
                REJECTED_BASELINE: sha256(HERE / REJECTED_BASELINE),
                STALLED_STATIC_RUN: sha256(HERE / STALLED_STATIC_RUN),
            },
        },
    },
    "checks": whole_checks,
    "artifacts": {name: sha256(HERE / name) for name in canonical},
}

OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary["performance"], indent=2))
print("PASS: safe artifacts, source hashes, correctness, queue completion, and token identity")
