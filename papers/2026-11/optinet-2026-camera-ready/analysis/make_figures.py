#!/usr/bin/env python3
"""Regenerate Fig. 1 and Fig. 2 for the camera-ready (revision round 1).

Fig. 1: tasks solved against the budget as a share of Large's need (same unit as the tables), with
Small as a horizontal line and the Retry + reserve (checker) cascade added. Curves are means over 500
arrival orders from the artifact's optinet/results/fixed_spend.json; Retry + reserve is simulated here
with analysis/paired_and_retry.py (same pool, estimator and 500 orders). Legend names equal Table I.
Fig. 2: how each task ended, with Table I names and no label/legend overlap.

    python3 analysis/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paired_and_retry as pr  # noqa: E402  (also puts the artifact on sys.path)
import fixed_spend as fs  # noqa: E402

ART = pr.ART
R = json.loads((ART / "optinet/results/fixed_spend.json").read_text())
OUT = HERE.parent / "figs"

# (policy key, Table I name, color, linestyle, marker, width)
STYLE = [("small", "Small", "#7f7f7f", "-", None, 1.4),
         ("large", "Large", "#222222", "-", None, 1.3),
         ("stop-fallback", "Stop + fallback", "#c0392b", "--", None, 1.3),
         ("route", "Route only", "#e08e0b", (0, (8, 2)), None, 1.3),
         ("admit", "Admission control", "#8c564b", ":", None, 1.5),
         ("retry-paced", "Retry + reserve", "#2ca02c", (0, (5, 1, 1, 1)), None, 1.4),
         ("hybrid", "Paced + spend-down", "#1f77b4", "-", "o", 2.2)]


def rows_for(name):
    causal = {r["policy"] for r in R[f"causal_{name}"]["rows"]} - {"large"}
    rows = ([r for r in R[name]["rows"] if r["policy"] not in causal]
            + [r for r in R[f"causal_{name}"]["rows"] if r["policy"] in causal])
    small, large, ids, est, rate = pr.setup(name)
    full = sum(fs.task_cost(large[i]) for i in ids)
    sw, _ = pr.sweep(name, ["retry-paced"], fs.FRACS)
    for f in fs.FRACS:
        r = sw[f"{f}|retry-paced"]
        rows.append({"policy": "retry-paced", "frac": f, "succ_mean": r["succ_mean"], "succ_p05": r["p05"], "succ_p95": r["p95"]})
    return rows


fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.15), sharey=True)
for ax, name in zip(axes, ["cached", "uncached"]):
    rows = rows_for(name)
    for p, label, color, ls, mk, lw in STYLE:
        r = sorted((x for x in rows if x["policy"] == p), key=lambda x: x["frac"])
        xs = [100 * x["frac"] for x in r]
        ax.plot(xs, [x["succ_mean"] for x in r], linestyle=ls, color=color, lw=lw, label=label,
                marker=mk, markersize=2.5, zorder=3 if p == "hybrid" else 2)
        if p == "hybrid":
            ax.fill_between(xs, [x["succ_p05"] for x in r], [x["succ_p95"] for x in r], color=color, alpha=0.3, lw=0)
    ax.set_xlabel("Budget (% of Large's need)", fontsize=7.5)
    ax.set_title(f"{name.capitalize()} runs", fontsize=8)
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 50)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7.5)
axes[0].set_ylabel("Tasks solved (of 50)", fontsize=7.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, loc="lower center", ncol=4, frameon=False, columnspacing=0.8, handlelength=1.8)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(OUT / "fig-budget-curve.pdf")
fig.savefig(OUT / "fig-budget-curve.png", dpi=150)

# ---- Fig. 2
rows = ([r for r in R["cached"]["rows"] if r["policy"] != "hybrid"]
        + [r for r in R["causal_cached"]["rows"] if r["policy"] == "hybrid"])
POL = [("large", "Large"), ("stop", "Stop only"), ("route", "Route only"), ("hybrid", "Paced + SD")]
FR = [0.2, 0.4, 1.0]
CAT = [("solved", "Solved", "#1f77b4", ""), ("model", "Model failed", "#9ecae1", "////"), ("cap", "Stopped by policy", "#e8a0a0", "\\\\"),
       ("pool", "Pool ran out mid-task", "#111111", ""), ("unstarted", "Never started", "#dddddd", "xxxx")]
fig, ax = plt.subplots(figsize=(3.5, 2.0))
x, xs, labels = 0, [], []
for f in FR:
    for p, name in POL:
        r = next(r for r in rows if r["policy"] == p and abs(r["frac"] - f) < 1e-9)
        vals = {"solved": r["succ_mean"], **r["reasons"]}
        bottom = 0
        for k, lab, c, hz in CAT:
            ax.bar(x, vals[k], bottom=bottom, color=c, width=0.8, hatch=hz, edgecolor="#444444" if hz else c, linewidth=0.3, label=lab if x == 0 else None)
            bottom += vals[k]
        xs.append(x); labels.append(name)
        x += 1
    x += 0.6
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=90, fontsize=9)
for i, f in enumerate(FR):
    ax.text(xs[i * 4] + 1.5, 62, f"{int(f * 100)}% budget", ha="center", fontsize=8)
ax.set_ylim(0, 60)
ax.set_ylabel("Tasks (of 50)", fontsize=9)
ax.tick_params(axis="y", labelsize=8)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, handlelength=1.2, columnspacing=0.8)
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(OUT / "fig-causes.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig-causes.png", dpi=150)
print("wrote", OUT / "fig-budget-curve.pdf", OUT / "fig-causes.pdf")
