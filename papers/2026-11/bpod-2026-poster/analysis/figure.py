"""Fig. 1 of the P270 camera-ready. Left: share of spend by billing category.
Bars are recomputed from the multi-turn step logs (recompute.json), every configuration with
complete step logs ((a'), whose per-step records are incomplete, is omitted).
Right: multi-turn cost per task vs success with 95% Wilson intervals (recompute.json).
Text is at least 7 pt, serif face matches the Times body, and each billing category has its
own hatch so categories survive grayscale printing."""
import json, os
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(H, "recompute.json")))
plt.rcParams.update({"font.size": 7, "font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "hatch.linewidth": 0.5})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 1.8), gridspec_kw={"width_ratios": [1.05, 1]})
# ---- left panel
MT = {"(a) Sonnet": "frontier", "(d) Sonnet+C": "frontier_cache", "(b) Haiku": "small", "(c) Haiku+C+B": "small_cache_budget", "(b') Haiku, low": "small_low"}
rows = []
for k, c in MT.items():
    rows.append((k, {"in": R[f"mt.{c}.share_in_pct"], "cw": R[f"mt.{c}.share_cw_pct"],
                                  "cr": R[f"mt.{c}.share_cr_pct"], "out": R[f"mt.{c}.share_out_pct"]}))
cols = {"in": ("#6f9fd0", "", "Uncached input"),
        "cw": ("#fdd0a2", "////", "Cache write"), "cr": ("#c7e9c0", "\\\\\\\\", "Cache read"),
        "out": ("#222222", "", "Output + reasoning")}
seen = set()
for i, (lab, d) in enumerate(rows[::-1]):
    left = 0
    for key in ["in", "cw", "cr", "out"]:
        if key in d and d[key] > 0:
            a1.barh(i, d[key], left=left, color=cols[key][0], hatch=cols[key][1], edgecolor="black", linewidth=0.4,
                    label=None if key in seen else cols[key][2], height=0.72)
            if key in ("in", "out") and d[key] >= 6:
                fmt = "%.1f"
                a1.text(left + d[key]/2, i, fmt % d[key], ha="center", va="center", fontsize=6.5,
                        color="white" if key in ("out", "in") else "black")
            elif key in ("cw", "cr") and d[key] >= 15:
                a1.text(left + d[key]/2, i, "%.1f" % d[key], ha="center", va="center", fontsize=6.5, color="black",
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
            seen.add(key); left += d[key]
a1.set_yticks(range(len(rows))); a1.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7)
a1.set_xlim(0, 100); a1.set_xlabel("Share of spend (%)", fontsize=7)
a1.tick_params(labelsize=7)
a1.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False,
          handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
# ---- right panel
names = {"frontier": "(a)", "frontier_cache": "(d) selected", "small": "(b)", "small_cache_budget": "(c)",
         "frontier_low": "(a')", "small_low": "(b')"}
off = {"frontier": (-16, 5), "frontier_cache": (-9, -3), "small": (6, -14), "small_cache_budget": (5, -9),
       "frontier_low": (5, -17), "small_low": (-24, -3)}
for c, (cost, s, (lo, hi)) in R["fig.frontier_points"].items():
    low = c.endswith("_low"); x = cost
    star = c == "frontier_cache"
    a2.errorbar(x, s, yerr=[[s - lo], [hi - s]], fmt="*" if star else ("s" if low else "o"),
                mfc="white" if low else "k", mec="k", ecolor="0.45", ms=8 if star else (5.2 if low else (7.5 if c == "frontier" else 3.2)), capsize=1.5, lw=0.7, zorder=(2 if c == "frontier" else 3))
    a2.annotate(names[c], (x, s), textcoords="offset points", xytext=off[c], fontsize=7,
                fontweight="bold" if star else "normal", ha="right" if star else "left")
a2.axhline(R["mt.quality_bar_pct"], ls=":", c="0.35", lw=0.8)
a2.text(0.00105, R["mt.quality_bar_pct"] + 1.2, "quality bar (90% of best)", fontsize=7, color="0.2")
a2.set_xscale("log"); a2.set_xlim(0.001, 0.3); a2.set_ylim(40, 100)
a2.set_xlabel("Mean cost per multi-turn task (USD, log)", fontsize=7); a2.set_ylabel("Task success (%)", fontsize=7)
a2.tick_params(labelsize=7)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(H, "..", "fig-mix-frontier.pdf"))
print("wrote fig-mix-frontier.pdf")
