#!/usr/bin/env python3
"""Fig. 2: what each of the 50 tasks ended as, by policy and fixed budget (cached runs, mean over orders)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "results" / "fixed_spend.json").read_text())
# the paced policy is drawn with causal estimates (out-of-sample priors), the rest need none
rows = ([r for r in data["cached"]["rows"] if r["policy"] != "hybrid"]
        + [r for r in data["causal_cached"]["rows"] if r["policy"] == "hybrid"])
POL = [("large", "Large"), ("stop", "Stop"), ("route", "Route"), ("hybrid", "Paced+SD")]
FR = [0.2, 0.4, 1.0]
CAT = [("solved", "Solved", "#1f77b4"), ("model", "Model failed", "#9ecae1"), ("cap", "Stopped by policy", "#c0392b"),
       ("pool", "Pool ran out mid-task", "#e08e0b"), ("unstarted", "Never started", "#bbbbbb")]
fig, ax = plt.subplots(figsize=(3.5, 3.2))
x, labels = 0, []
xs = []
for f in FR:
    for p, name in POL:
        r = next(r for r in rows if r["policy"] == p and abs(r["frac"] - f) < 1e-9)
        vals = {"solved": r["succ_mean"], **r["reasons"]}
        bottom = 0
        for k, lab, c in CAT:
            ax.bar(x, vals[k], bottom=bottom, color=c, width=0.8, label=lab if x == 0 else None)
            bottom += vals[k]
        xs.append(x); labels.append(name)
        x += 1
    x += 0.6
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
for i, f in enumerate(FR):
    ax.text(xs[i * 4] + 1.5, 52.5, f"{int(f * 100)}% budget", ha="center", fontsize=7)
ax.set_ylim(0, 57)
ax.set_ylabel("Tasks (of 50)", fontsize=8)
ax.tick_params(axis="y", labelsize=7)
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, frameon=False, handlelength=1.2, columnspacing=0.8)
fig.tight_layout()
out = HERE / "results" / "fig-causes.pdf"
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
print(out)
