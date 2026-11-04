"""Figures for the two AAAI-27 workshop papers (no API calls).

python3 analysis/workshop_figures.py OUTDIR
  fig-stop-landing.pdf  reactive vs transactional stops across per-task budgets (results/transactional_stop.json)
  fig-loop-limits.pdf   paired change in cost per success for each hard stop (results/loop_limits_paired.json)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
ORDER = ["small", "small+cache", "large", "large+cache"]
NAMES = {"small": "Small", "small+cache": "Small, cached", "large": "Large", "large+cache": "Large, cached"}
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})


def stop_landing():
    d = json.loads((ROOT / "results" / "transactional_stop.json").read_text())
    qs = [0.5, 0.75, 0.9]
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.55))
    for cfg, mk in zip(ORDER, "os^D"):
        for rule, ls in (("reactive", "-"), ("transactional", "--")):
            half = [d[cfg][f"q={q} {rule}"]["half_applied_turns"] for q in qs]
            succ = [100 * d[cfg][f"q={q} {rule}"]["success"] for q in qs]
            col = "#c0392b" if rule == "reactive" else "#2c6fb0"
            axes[0].plot(qs, half, ls, marker=mk, ms=3, lw=0.8, color=col)
            axes[1].plot(qs, succ, ls, marker=mk, ms=3, lw=0.8, color=col)
    axes[0].set_ylabel("Tasks with a half-applied turn")
    axes[1].set_ylabel("Success (%)")
    for ax in axes:
        ax.set_xlabel("Budget percentile")
        ax.set_xticks(qs)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].plot([], [], "-", color="#c0392b", label="Reactive")
    axes[0].plot([], [], "--", color="#2c6fb0", label="Transactional")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig-stop-landing.pdf")


def loop_limits():
    """Paired 95% interval of the change in cost per success, each hard stop against no limit (loop_limits_paired.json)."""
    d = json.loads((ROOT / "results" / "loop_limits_paired.json").read_text())
    names = ["Steps per turn 3", "Steps per turn 4", "Steps per turn 6", "Errors 1", "Errors 2", "Deadline p90", "Deadline p75"]
    cols = ["#2c6fb0"] * 3 + ["#c0392b"] * 2 + ["#7d8c2c"] * 2
    mks = ["o"] * 3 + ["s"] * 2 + ["^"] * 2
    xmax = 100
    with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 9}):
        fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.8), sharey=True, sharex=True)
        for ax, cfg in zip(axes, ORDER):
            rows = [r for r in d[cfg] if not r["setting"].startswith("budget")]
            for i, (r, col, mk) in enumerate(zip(rows, cols, mks)):
                y = len(rows) - 1 - i
                lo, hi = r["ci95_pct"]
                est = r["d_usd_per_success_pct"]
                ax.plot([lo, min(hi, xmax)], [y, y], color=col, lw=1.1, solid_capstyle="butt", zorder=2)
                if hi > xmax:
                    ax.annotate("", xy=(xmax + 4, y), xytext=(xmax - 6, y),
                                arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, mutation_scale=6), annotation_clip=False)
                    ax.text(xmax - 6, y - 0.2, f"{hi:.0f}", fontsize=8.5, ha="right", va="top", color=col)
                ax.scatter(est, y, marker=mk, s=30, zorder=3, facecolor=col if r["excludes_zero"] else "white",
                           edgecolor=col, linewidth=0.9)
            ax.axvline(0, color="0.45", lw=0.7, zorder=1)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(names[::-1])
            ax.set_xlim(-20, xmax + 6)
            ax.set_xticks([0, 50, 100])
            ax.set_ylim(-0.6, len(rows) - 0.4)
            ax.set_title(NAMES[cfg], fontsize=9, pad=2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2, pad=1.5)
        fig.supxlabel("Change in cost per successful task against no limit (%), paired 95% interval", fontsize=9, y=0.0)
        fig.tight_layout(pad=0.3, w_pad=0.5)
        fig.savefig(OUT / "fig-loop-limits.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    stop_landing()
    loop_limits()
