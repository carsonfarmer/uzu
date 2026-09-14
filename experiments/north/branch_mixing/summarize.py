"""Validate retained ablation records and summarize matched comparisons."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
MODES = ["original", "mixed", "blocked", "separate"]
CONTRASTS = [("mixed", "original"), ("separate", "original"),
             ("mixed", "separate"), ("mixed", "blocked")]


def summarize(paths):
    groups = {}
    files = []
    for path in paths:
        path = path.resolve()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        provenance = rows[0]
        assert provenance["kind"] == "provenance"
        args = provenance["args"]
        for name, expected in provenance["sources"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        checks = {r["prompt"]: r for r in rows if r["kind"] == "correctness"}
        assert set(checks) == set(args["prompts"])
        modes = ["original", *checks[args["prompts"][0]]["variants"]]
        assert modes in (MODES, MODES + ["serial"])
        measured = [r for r in rows if r["kind"] == "generation"]
        assert len(measured) == len(args["prompts"]) * len(modes) * (args["runs"] + 1)
        for label in args["prompts"]:
            check = checks[label]
            assert check["full_logits_bitwise_equal"] is True
            assert check["decode_steps_per_variant"] == args["tokens"] - 1
            assert check["variants"] == modes[1:]
            assert check["logit_shape"] == [1, 1, 262144]
            key = (args["timing"], label)
            group = groups.setdefault(key, {mode: [] for mode in modes})
            assert list(group) == modes, "Summarize different designs separately"
            for rep in range(-1, args["runs"]):
                block = [r for r in measured if r["prompt"] == label and r["repetition"] == rep]
                assert len(block) == len(modes) and {r["variant"] for r in block} == set(modes)
                for row in sorted(block, key=lambda r: modes.index(r["variant"])):
                    assert row["warmup"] == (rep < 0)
                    assert row["token_ids"] == check["token_ids"]
                    assert len(row["step_seconds"]) == row["decode_steps"] == args["tokens"] - 1
                    assert all(math.isfinite(t) and t > 0 for t in row["step_seconds"])
                    assert math.isclose(sum(row["step_seconds"]), row["decode_seconds"], rel_tol=1e-12)
                    rate = row["decode_steps"] / row["decode_seconds"]
                    assert math.isclose(rate, row["decode_tokens_per_second"], rel_tol=1e-12)
                    if rep >= 0:
                        group[row["variant"]].append({"file": path.name, "repetition": rep, "rate": rate})
        files.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "timing": args["timing"], "runs_per_prompt_variant": args["runs"],
                      "correctness": checks})
    result = {"files": files, "comparisons": {}}
    for (timing, label), group in groups.items():
        entry = {"median_tokens_per_second": {m: statistics.median(s["rate"] for s in v) for m, v in group.items()},
                 "samples_per_variant": len(group["mixed"]), "contrasts": {}}
        contrasts = CONTRASTS + ([("mixed", "serial"), ("separate", "serial")] if "serial" in group else [])
        for numerator, denominator in contrasts:
            a, b = group[numerator], group[denominator]
            assert [(r["file"], r["repetition"]) for r in a] == [(r["file"], r["repetition"]) for r in b]
            ratios = [x["rate"] / y["rate"] for x, y in zip(a, b)]
            entry["contrasts"][f"{numerator}_over_{denominator}"] = {
                "ratio_of_median_rates_percent": 100 * (entry["median_tokens_per_second"][numerator] / entry["median_tokens_per_second"][denominator] - 1),
                "paired_geomean_percent": 100 * (math.exp(statistics.mean(math.log(r) for r in ratios)) - 1),
                "paired_percent": [100 * (r - 1) for r in ratios],
                "per_process_paired_geomean_percent": {
                    file: 100 * (math.exp(statistics.mean(math.log(x["rate"] / y["rate"])
                            for x, y in zip(a, b) if x["file"] == file)) - 1)
                    for file in dict.fromkeys(x["file"] for x in a)
                },
            }
        result["comparisons"][f"{timing}/{label}"] = entry
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.inputs)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    for key, value in result["comparisons"].items():
        print(key, {k: round(v, 3) for k, v in value["median_tokens_per_second"].items()})
        print({k: round(v["paired_geomean_percent"], 3) for k, v in value["contrasts"].items()})
