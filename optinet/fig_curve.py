#!/usr/bin/env python3
"""Fig. 1: successes against the fixed batch budget (cached runs), with 5th-95th percentile bands over task orders."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "results" / "fixed_spend.json").read_text())
STYLE = {"large": ("Large, no policy", "#444444", "-"), "stop": ("Stop only (paced cap)", "#c0392b", "--"),
         "stop-fallback": ("Stop + small fallback", "#e377c2", ":"),
         "route": ("Route only (first error)", "#e08e0b", "-."), "random-paced": ("Random escalation, paced", "#2ca02c", "--"),
         "oracle-paced": ("Oracle escalation, paced", "#000000", (0, (1, 1))),
         "admit": ("Admission control (large first)", "#8c564b", "-."),
         "paced+cap": ("Paced (ours)", "#1f77b4", "-"), "hybrid": ("Paced + spend-down (ours)", "#17becf", "--")}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, name in zip(axes, ["cached", "uncached"]):
    # policies that use cost estimates are drawn with causal estimates (out-of-sample priors)
    causal = {r["policy"] for r in data[f"causal_{name}"]["rows"]} - {"large"}
    rows = ([r for r in data[name]["rows"] if r["policy"] not in causal]
            + [r for r in data[f"causal_{name}"]["rows"] if r["policy"] in causal])
    n = 50
    for p, (label, color, ls) in STYLE.items():
        r = [x for x in rows if x["policy"] == p]
        xs = [x["budget"] / n for x in r]
        ax.plot(xs, [100 * x["succ_mean"] / n for x in r], linestyle=ls, color=color, lw=1.8 if p in ("paced+cap", "hybrid") else 1.1, label=label)
        ax.fill_between(xs, [100 * x["succ_p05"] / n for x in r], [100 * x["succ_p95"] / n for x in r], color=color, alpha=0.12, lw=0)
    ax.set_xlabel("Batch budget per task (USD)")
    ax.set_title(f"{name.capitalize()} runs", fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("Tasks solved (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=6.5, loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.13, 1, 1))
out = HERE / "results" / "fig-budget-curve.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=150)
print(out)
