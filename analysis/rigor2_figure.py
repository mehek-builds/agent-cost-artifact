"""Trigger frontier figure from results/rigor2.json (run analysis/rigor2.py first).

python3 analysis/rigor2_figure.py  -> results/fig-trigger-frontier.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
ROOT = Path(__file__).resolve().parent.parent
R = json.load(open(ROOT / "results" / "rigor2.json"))
P = R["frontier"]["policies"]
fig, ax = plt.subplots(figsize=(3.45, 2.5))
first = True
for name, v in P.items():
    if name in ("small", "large"):
        continue
    ax.plot(v["success"] * 100, v["cost_per_success"], "o", ms=3.5, color="#9DB4D0", zorder=2,
            label="other causal triggers" if first else None)
    first = False
fe = P["('err', 1)"]
ax.plot(fe["success"] * 100, fe["cost_per_success"], "o", ms=6, color="#4C78A8", zorder=3, label="first tool error")
for name, col, lab in (("small", "#F58518", "small only"), ("large", "#E45756", "large only")):
    ax.plot(P[name]["success"] * 100, P[name]["cost_per_success"], "s", ms=5.5, color=col, label=lab, zorder=3)
mix = R["frontier"]["matched_success_random_mix"]
xs = [58 + i * 0.5 for i in range(57)]
ys = []
for x in xs:
    w = (x / 100 - 0.58) / 0.28
    ys.append(((1 - w) * P["small"]["cost_per_task"] + w * P["large"]["cost_per_task"]) / (x / 100))
ax.plot(xs, ys, color="#555", lw=0.9, ls="--", label="random mix of tiers")
ax.plot(mix["success"] * 100, mix["cost_per_success"], "x", color="#555", ms=5)
ip = R["in_run_predictor"]
ax.plot([v["success"] * 100 for v in ip.values()], [v["cost_per_success"] for v in ip.values()], "^", ms=4.5,
        color="#54A24B", label="learned in-run predictor", zorder=3)
ax.set_xlabel("success (%)", fontsize=7.5)
ax.set_ylabel("USD per successful task", fontsize=7.5)
ax.tick_params(labelsize=7)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=6.2, frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(ROOT / "results" / "fig-trigger-frontier.pdf")
