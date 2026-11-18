"""Fig. 3: Litos daily provider token cost, Sep 28 to Oct 27, split by token category (the bill's view),
with the operation-attributed window marked.
python3 fig_daily.py -> ../fig-daily.pdf; prints the plotted daily values."""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DAILY = Path(os.environ.get("AGENT_AGGREGATES_DIR") or Path(os.environ.get("AGENT_ARTIFACT_ROOT") or Path(__file__).resolve().parents[4]) / "papers" / "2026-11" / "production-aggregates") / "litos-console-daily-2026-07-30_2026-10-27.csv"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})
rows = [r for r in csv.DictReader(open(DAILY)) if r["date"] >= "2026-09-28"]
cats = [("usd_in", "uncached input", "#4C72B0"), ("usd_cache_write", "cache write", "#55A868"),
        ("usd_cache_read", "cache read", "#8C8C8C"), ("usd_out", "output", "#DD8452")]
fig, ax = plt.subplots(figsize=(3.45, 1.45))
bottom = [0.0] * len(rows)
for key, lab, col in cats:
    v = [float(r[key]) for r in rows]
    ax.bar(range(len(rows)), v, bottom=bottom, color=col, width=0.8, label=lab)
    bottom = [b + x for b, x in zip(bottom, v)]
for r, b in zip(rows, bottom):
    print(r["date"], round(b, 2))
dates = [r["date"] for r in rows]
i0 = dates.index("2026-10-23")
ax.axvspan(i0 - 0.5, len(rows) - 0.5, color="#EEEEEE", zorder=0)
ax.text(i0 + 0.3, max(bottom) * 0.92, "attributed\nwindow", fontsize=7.5, va="top")
ticks = [i for i, d in enumerate(dates) if d[-2:] in ("01", "08", "15", "22")]
ax.set_xticks(ticks)
ax.set_xticklabels([dates[i][5:].replace("-", "/") for i in ticks])
ax.set_ylabel("USD per day")
ax.legend(fontsize=7.5, frameon=False, loc="upper left", handlelength=1, labelspacing=0.2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.2)
fig.savefig(HERE.parent / "fig-daily.pdf")
