"""Render the shareable whole-pass chart from verified summary data."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
data = json.loads((HERE.parent / "whole_pass" / "summary-v1.json").read_text())
rates = data["performance"]["rates"]
rejected = data["rejected_result"]

rows = [
    ("Exact fused control", rates["exact_fused"], "#6D45D8", ""),
    (
        "Static schedule — rejected: deadlocked",
        rejected["successful_run_rate"],
        "#F2B5AE",
        "///",
    ),
    ("Stock MLX", rates["original_mlx"], "#5E6673", ""),
    ("Safe one-dispatch queue", rates["safe_megakernel"], "#0096C7", ""),
    ("Safe queue + coarse jobs", rates["coarse_tiles"], "#91C9DD", ""),
    ("Safe queue + 1 cache-warm stage", rates["prefetch_1"], "#B6DCE8", ""),
    ("Safe queue + 10 cache-warm stages", rates["prefetch_10"], "#CAD1D9", ""),
]

fig, ax = plt.subplots(figsize=(12.5, 8), dpi=160)
fig.patch.set_facecolor("#F7F8FA")
ax.set_facecolor("#F7F8FA")

plot_rows = list(reversed(rows))
labels = [row[0] for row in plot_rows]
values = [row[1] for row in plot_rows]
colors = [row[2] for row in plot_rows]
bars = ax.barh(labels, values, color=colors, height=0.68)
for bar, row in zip(bars, plot_rows):
    if row[3]:
        bar.set_hatch(row[3])
        bar.set_edgecolor("#B64B42")
        bar.set_linewidth(1.2)

stock = rates["original_mlx"]
ax.axvline(stock, color="#5E6673", linewidth=1.3, linestyle=(0, (3, 3)), alpha=0.8)
for bar, row in zip(bars, plot_rows):
    label, value, _, _ = row
    if label.startswith("Static"):
        note = f"{value:.2f} tok/s  (unsafe)"
    else:
        delta = 100 * (value / stock - 1)
        note = f"{value:.2f} tok/s  ({delta:+.2f}%)"
    ax.text(
        value + 0.45,
        bar.get_y() + bar.get_height() / 2,
        note,
        va="center",
        ha="left",
        fontsize=10.3,
        color="#22252A",
        fontweight="bold" if label == "Safe one-dispatch queue" else "normal",
    )

ax.set_xlim(0, 67)
ax.set_xlabel("Decode tokens per second", fontsize=11, color="#3B4048", labelpad=10)
ax.tick_params(axis="y", labelsize=10.5, colors="#272B31", length=0)
ax.tick_params(axis="x", labelsize=9, colors="#626A76")
ax.xaxis.grid(True, color="#DDE1E7", linewidth=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

fig.suptitle(
    "North Mini Code 4-bit decode on Apple M4 Pro",
    x=0.075,
    y=0.965,
    ha="left",
    fontsize=21,
    fontweight="bold",
    color="#17191D",
)
ax.set_title(
    "All 49 transformer layers + K/V updates + 262k logits in one Metal dispatch\n"
    "Safe queue: within 1% of stock MLX; full logits byte-exact over 254 steps",
    loc="left",
    fontsize=12,
    color="#454C56",
    pad=22,
    linespacing=1.5,
)
fig.text(
    0.075,
    0.035,
    "Headline: 6 runs/path · ablations: step-interleaved, 3 runs · 127 decode steps/run · MLX 0.32.2",
    fontsize=9.3,
    color="#626A76",
)
fig.text(
    0.075,
    0.014,
    "The faster static result is historical evidence only: its all-grid barrier later deadlocked.",
    fontsize=9.3,
    color="#9B3B34",
)
fig.text(
    0.99,
    0.025,
    "Cohere megakernel ideas, adapted to Metal",
    fontsize=9.3,
    color="#626A76",
    ha="right",
)
plt.subplots_adjust(left=0.34, right=0.92, top=0.80, bottom=0.13)

for suffix in ("png", "svg"):
    output = HERE / f"north-m4-pro-megakernel-results.{suffix}"
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    if suffix == "svg":
        output.write_text(
            "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
        )
    print(output)
