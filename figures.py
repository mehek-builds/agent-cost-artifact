#!/usr/bin/env python3
"""Figures for the paper, from results/*.json and the logs.  python3 figures.py
Writes results/fig-cost-success.pdf and results/fig-task39.pdf."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from escalation import escalate  # noqa: E402
from replay import ROOT, cache_fault, load, step_cost  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a85", "#1f1f1e", "#6b6b66"
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.5,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "axes.labelcolor": INK, "pdf.fonttype": 42, "ps.fonttype": 42})


def cost_success() -> None:
    E = json.loads((ROOT / "results" / "escalation_numbers.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.85), sharey=True)
    for ax, pair in zip(axes, ("cached", "uncached")):
        P = E["pairs"][pair]
        pts = [("small", P["static_small"]["cost_per_task"], P["static_small"]["success"], None, BLUE, "o", True),
               ("large", P["static_large"]["cost_per_task"], P["static_large"]["success"], None, BLUE, "o", True),
               ("oracle", P["oracle"]["cost_per_task"], P["oracle"]["success"], None, GRAY, "s", False)]
        r = P["policies"]["first_error/restart"]
        pts.append(("restart", r["cost_per_task"], r["success_lo"], None, AQUA, "^", True))
        f = P["policies"]["first_error/full"]
        pts.append(("full", f["cost_per_task"], sum(f["success_q"]) / 2, (f["success_lo"], f["success_hi"]), YELLOW, "D", True))
        t = P["tally_priced_choice"]
        pts.append(("Tally", t["cost_per_task"], sum(t["success_q"]) / 2, (t["success_lo"], t["success_hi"]), ORANGE, "*", True))
        offs = {"small": (4, -2), "large": (-4, -8), "oracle": (-6, 3), "restart": (5, 4), "full": (5, 0), "Tally": (-5, -2)}
        if pair == "uncached":
            offs.update({"restart": (4, -3), "full": (4, -9), "Tally": (-4, 5), "oracle": (-4, 4)})
        for name, x, y, iv, col, mk, filled in pts:
            if iv:
                ax.plot([x, x], [100 * iv[0], 100 * iv[1]], color=col, lw=1.2, solid_capstyle="round", zorder=2)
            hollow = not filled or name == "restart"
            ax.scatter([x], [100 * y], s=36 if mk == "*" else (26 if name == "restart" else 16), marker=mk,
                       facecolor="none" if hollow else col, edgecolor=col if hollow else "white",
                       linewidth=1.0 if hollow else 0.8, zorder=4 if name == "restart" else 3)
            ha = "left" if offs[name][0] > 0 else "right"
            ax.annotate(name, (x, 100 * y), xytext=offs[name], textcoords="offset points", ha=ha, va="center",
                        fontsize=6.5, color=INK)
        ax.set_xscale("log")
        ax.set_title("with caching" if pair == "cached" else "without caching", fontsize=7, color=INK)
        ax.set_xlabel("cost per task (USD, log)")
        ax.grid(True, which="major", color="#e6e6e3", lw=0.4)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("success (%)")
    axes[0].set_ylim(50, 92)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(ROOT / "results" / "fig-cost-success.pdf")


def task39() -> None:
    sc, lc = load("small_cache_budget"), load("frontier_cache")
    i = "multi_turn_base_39"

    def cum(xs):
        out, c = [0.0], 0.0
        for x in xs:
            c += x
            out.append(c)
        return out

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.7))
    ax = axes[0]
    small = cum(step_cost(s) for s in sc[i]["steps"])
    large = cum(step_cost(s) for s in lc[i]["steps"])
    rst = cum(escalate(sc[i], lc[i], "first_error", "restart", True)["costs"])
    full = cum(escalate(sc[i], lc[i], "first_error", "full", True)["costs"])
    for ys, col, ls, lab in ((large, BLUE, "-", "large"), (small, GRAY, "-", "small (fails)"),
                             (full, YELLOW, "--", "full handoff"), (rst, ORANGE, "-", "Tally: restart")):
        ax.plot(range(len(ys)), [1000 * y for y in ys], color=col, ls=ls, lw=1.2, label=lab)
    ax.legend(loc="upper left", fontsize=5.5, frameon=False, handlelength=1.6, labelspacing=0.3, borderaxespad=0.2)
    ax.axvline(4, color=MUTED, lw=0.5, ls=":")
    ax.annotate("1st error", (4, 10), xytext=(-2, 0), textcoords="offset points", fontsize=5.5, color=MUTED, ha="right")
    ax.set_title("(a) escalation", fontsize=7, color=INK)
    ax.set_ylabel("cost so far (USD $\\times 10^{-3}$)")
    ax.set_xlabel("step")
    ax = axes[1]
    healthy = cum(step_cost(s) for s in lc[i]["steps"])
    nomon = cum(step_cost(s, "uncached") for s in lc[i]["steps"])
    task = lc[i]
    mon, misses, rep = [], 0, None
    for idx, s in enumerate(task["steps"]):
        if rep is None:
            mon.append(step_cost(s, "uncached"))
            if idx > 0:
                rep = idx + 1
        elif idx == rep:
            mon.append(step_cost(s, "cold"))
        else:
            mon.append(step_cost(s))
    assert abs(sum(mon) - cache_fault(task, "ignored", 1)[0]) < 1e-12
    mon = cum(mon)
    for ys, col, ls, lab in ((nomon, GRAY, "-", "fault, no monitor"), (mon, ORANGE, "-", "fault, Tally repairs"),
                             (healthy, BLUE, "-", "healthy cache")):
        ax.plot(range(len(ys)), [1000 * y for y in ys], color=col, ls=ls, lw=1.2)
        ax.annotate(lab, (len(ys) - 1, 1000 * ys[-1]), xytext=(-2, 4), textcoords="offset points", ha="right",
                    fontsize=6, color=INK)
    ax.set_title("(b) silent cache fault", fontsize=7, color=INK)
    ax.set_xlabel("step")
    for ax in axes:
        ax.grid(True, color="#e6e6e3", lw=0.4)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(ROOT / "results" / "fig-task39.pdf")


if __name__ == "__main__":
    cost_success()
    task39()
