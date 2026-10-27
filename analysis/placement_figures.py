"""Figures for the small-tier-first placement analysis, from results/placement.json.

python3 analysis/placement_figures.py  (run analysis/placement.py first)
Writes results/fig-frontier.pdf, results/fig-footprint.pdf and results/fig-energy-ratio.pdf.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
P = json.load(open(RES / "placement.json"))
POL = [("small", "Small", "#F58518"), ("large", "Large", "#4C78A8"),
       ("first_error_restart", "First-error", "#54A24B"), ("oracle", "Oracle", "#B279A2")]
# 95% bootstrap intervals for USD per success (escalation.py, 10,000 samples); none for the oracle.
CI = {"cached": {"small": (0.0023, 0.0041), "large": (0.0240, 0.0332), "first_error_restart": (0.0094, 0.0196)},
      "uncached": {"small": (0.0090, 0.0163), "large": (0.1287, 0.1834), "first_error_restart": (0.0469, 0.1027)}}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


def frontier():
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
    for ax, mode in zip(axes, ["cached", "uncached"]):
        d = P[f"{mode}_alpha1_beta1"]
        for key, name, col in POL:
            x, y = d[key]["successes"] / 50, d[key]["usd_per_success"]
            lo, hi = CI[mode].get(key, (y, y))
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", color=col, ms=4, capsize=2, label=name)
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=6)
        ax.set_yscale("log")
        ax.set_xlabel("success rate", fontsize=7)
        ax.set_title("with caching" if mode == "cached" else "without caching", fontsize=7)
        style(ax)
    axes[0].set_ylabel("USD per successful task (log)", fontsize=7)
    plt.tight_layout()
    plt.savefig(RES / "fig-frontier.pdf")


def footprint():
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    rows = [(key, name) for key, name, _ in POL]
    for i, (key, name) in enumerate(rows):
        d = P["cached_alpha1_beta1"][key]
        s, l = d["footprint_small_per_task"] / 1000, d["footprint_large_per_task"] / 1000
        ax.barh(i, s, color="#F58518", height=0.6, label="small tier" if i == 0 else None)
        ax.barh(i, l, left=s, color="#4C78A8", height=0.6, label="large tier" if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for _, n in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("footprint per task (thousand tokens, cached)", fontsize=6)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    style(ax)
    plt.tight_layout()
    plt.savefig(RES / "fig-footprint.pdf")


def energy_ratio():
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    d = P["cached_alpha0.1_beta1"]
    L = d["large"]
    rhos = [1 + i * 0.05 for i in range(81)]
    for key, name, col in POL:
        if key == "large":
            continue
        p = d[key]
        rel = [((p["footprint_small_per_task"] + r * p["footprint_large_per_task"]) / p["successes"])
               / (r * L["footprint_large_per_task"] / L["successes"]) for r in rhos]
        ax.plot(rhos, rel, color=col, label=name)
    ax.axhline(1, color="#4C78A8", lw=0.8, ls="--", label="Large (reference)")
    ax.set_xlabel("large-tier energy per token / small-tier energy per token", fontsize=7)
    ax.set_ylabel("energy per success\nrelative to Large", fontsize=7)
    ax.set_ylim(0, 2)
    ax.legend(fontsize=6, frameon=False)
    style(ax)
    plt.tight_layout()
    plt.savefig(RES / "fig-energy-ratio.pdf")


if __name__ == "__main__":
    frontier()
    footprint()
    energy_ratio()
