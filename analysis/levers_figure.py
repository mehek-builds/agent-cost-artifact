"""Compression break-even figure from results/levers.json (run analysis/levers.py first).

python3 analysis/levers_figure.py  -> results/fig-compression.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
ROOT = Path(__file__).resolve().parent.parent
L = json.load(open(ROOT / "results" / "levers.json"))
fig, ax = plt.subplots(figsize=(3.45, 2.3))
rs = [i / 100 for i in range(0, 101)]
for label, col in (("small+cache (5-call cap)", "#F58518"), ("large+cache", "#4C78A8")):
    d = L[label]["dollars"]
    base = d["input"] + d["cache_write"] + d["cache_read"] + d["output"]
    name = "Small" if label.startswith("small") else "Large"
    ax.plot(rs, [((1 - r) * d["prompt_at_fresh_price"] + d["output"]) / base for r in rs], color=col, lw=1.4,
            label=f"{name}: recompress every step")
    ax.plot(rs, [((1 - r) * (d["cache_read"] + d["cache_write"]) + d["input"] + d["output"]) / base for r in rs],
            color=col, lw=1.4, ls="--", label=f"{name}: compress cached prefix once")
    rb = L[label]["compression_break_even_r"]
    ax.plot([rb], [1], "o", color=col, ms=3.5)
    ax.annotate(f"{rb:.0%}", (rb, 1), textcoords="offset points", xytext=(-30, -12) if name == "Small" else (6, 4), fontsize=7, color=col)
ax.axhline(1, color="#888", lw=0.7)
ax.set_xlabel("fraction of prompt tokens removed", fontsize=7.5)
ax.set_ylabel("cost / cached baseline", fontsize=7.5)
ax.set_ylim(0, 7)
ax.tick_params(labelsize=7)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=6.3, frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(ROOT / "results" / "fig-compression.pdf")
