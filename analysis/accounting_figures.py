"""Figures for the token and dollar accounting, from results/accounting.json.

python3 analysis/accounting_figures.py  (run analysis/accounting.py first)
Writes results/fig-tokens-cps.pdf, results/fig-prompt-growth.pdf and results/fig-spend-mix.pdf.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
A = json.load(open(RES / "accounting.json"))
ORDER = [("Small", "small"), ("Small+cache", "small+cache"), ("Large", "large"), ("Large+cache", "large+cache")]
KINDS = ["input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens"]
COLS = dict(zip(KINDS, ["#4C78A8", "#F58518", "#72B7B2", "#54A24B"]))
NAMES = dict(zip(KINDS, ["fresh input", "cache write", "cache read", "output"]))
# 95% bootstrap intervals for USD per success, from escalation.py's static baselines (10,000 samples).
CPS_CI = {"small": (0.0090, 0.0163), "small+cache": (0.0023, 0.0041), "large": (0.1287, 0.1834), "large+cache": (0.0240, 0.0332)}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


def tokens_cps():
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    y = range(len(ORDER))
    left = [0.0] * len(ORDER)
    for k in KINDS:
        v = [A[c]["tokens_per_task"] * (A[c]["input_share_of_tokens"] * A[c]["input_mix"][k] if k != "output_tokens"
             else 1 - A[c]["input_share_of_tokens"]) / 1000 for _, c in ORDER]
        ax[0].barh(list(y), v, left=left, color=COLS[k], label=NAMES[k], height=0.6)
        left = [a + b for a, b in zip(left, v)]
    ax[0].set_yticks(list(y))
    ax[0].set_yticklabels([n for n, _ in ORDER])
    ax[0].invert_yaxis()
    ax[0].set_xlabel("thousand tokens per task")
    ax[0].legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, handlelength=1)
    cps = [A[c]["usd_per_success"] for _, c in ORDER]
    lo = [CPS_CI[c][0] for _, c in ORDER]
    hi = [CPS_CI[c][1] for _, c in ORDER]
    ax[1].barh(list(y), cps, color="#888", height=0.6,
               xerr=[[c - l for c, l in zip(cps, lo)], [h - c for c, h in zip(cps, hi)]], capsize=2)
    ax[1].set_xscale("log")
    ax[1].set_yticks(list(y))
    ax[1].set_yticklabels([])
    ax[1].invert_yaxis()
    ax[1].set_xlabel("USD per successful task (log)")
    for i, c in enumerate(cps):
        ax[1].text(c * 1.35, i, f"${c:.4f}", va="center", fontsize=6)
    for a in ax:
        style(a)
    plt.tight_layout()
    plt.savefig(RES / "fig-tokens-cps.pdf")


def prompt_growth():
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for c, lab, col in [("large+cache", "Large+cache", "#4C78A8"), ("small+cache", "Small+cache", "#F58518")]:
        pts = sorted((int(i), v) for i, v in A[c]["prompt_by_step"].items() if v["n"] >= 10)
        xs = [i for i, _ in pts]
        ax.plot(xs, [v["mean_prompt"] / 1000 for _, v in pts], marker="o", ms=2.5, color=col, label=lab + " (prompt)")
        ax.plot(xs, [v["mean_first_prompt"] / 1000 for _, v in pts], ls="--", lw=0.8, color=col, label=lab + " (first-step part)")
    ax.set_xlabel("step index within task", fontsize=7)
    ax.set_ylabel("k tokens per request", fontsize=7)
    ax.set_ylim(0, 9)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    style(ax)
    plt.tight_layout()
    plt.savefig(RES / "fig-prompt-growth.pdf")


def spend_mix():
    # The production row is the deployed agent's 90-day dollar split by billing category, from the
    # provider's daily cost records (2026-07-30 to 2026-10-27); it is an aggregate, not part of these logs.
    production = {"input_tokens": 0.24, "cache_write_tokens": 0.18, "cache_read_tokens": 0.01, "output_tokens": 0.57}
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    rows = [("Benchmark, large+cache", "large+cache"), ("Benchmark, small+cache", "small+cache")]
    shares = [A[c]["dollar_share_by_kind"] for _, c in rows] + [production]
    for i, share in enumerate(shares):
        left = 0.0
        for k in KINDS:
            ax.barh(i, share[k], left=left, color=COLS[k], height=0.6, label=NAMES[k] if i == 0 else None)
            left += share[k]
    ax.set_yticks(range(3))
    ax.set_yticklabels([r for r, _ in rows] + ["Production agent, 90 days"], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of token spend", fontsize=6)
    ax.legend(fontsize=5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.3, -0.45), handlelength=1, columnspacing=0.8)
    style(ax)
    plt.tight_layout()
    plt.savefig(RES / "fig-spend-mix.pdf")


if __name__ == "__main__":
    tokens_cps()
    prompt_growth()
    spend_mix()
