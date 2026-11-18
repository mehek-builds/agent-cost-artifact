"""Fig. 1: trace waterfall with a cost column for one recorded run (frontier, multi_turn_base_62),
and Fig. 2: the most expensive accepted Litos package operation as cumulative cost by call.

python3 -I fig_waterfall.py -> ../fig-waterfall.pdf, ../fig-packet.pdf, prints the plotted values
Bar start = sum of the recorded latencies of earlier steps (the harness runs steps one after another,
tools are local simulators, so the gaps between model calls are not recorded and are drawn as zero).
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LOGS = Path(os.environ.get("AGENT_ARTIFACT_ROOT") or Path(__file__).resolve().parents[4]) / "logs"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})

CFG, TASK = "frontier", "multi_turn_base_62"
steps = sorted((json.loads(l) for l in open(LOGS / f"steps_{CFG}.jsonl")), key=lambda s: s["step"])
steps = [s for s in steps if s["task_id"] == TASK]
task = [json.loads(l) for l in open(LOGS / f"tasks_{CFG}.jsonl") if json.loads(l)["task_id"] == TASK][0]
print("task", task)

COL = {"act": "#4C72B0", "final": "#8C8C8C", "failed": "#C44E52", "repair": "#DD8452"}
fig, ax = plt.subplots(figsize=(3.45, 2.15))
t = 0.0
cum = 0.0
prev_failed = False
labels = []
n = len(steps)
for i, s in enumerate(steps):
    kind = "failed" if s["kind"] == "failed" else ("repair" if prev_failed else s["kind"])
    prev_failed = s["kind"] == "failed"
    y = n - 1 - i
    ax.barh(y, s["latency_s"], left=t, color=COL[kind], height=0.7)
    cum += s["cost_usd"]
    u = s["usage"]
    prompt = u["input_tokens"] + u["cache_read_tokens"] + u["cache_write_tokens"]
    print(f"turn {s['turn']} step {s['step']} {kind:7s} calls={s['n_calls']} err={s['n_errors']} "
          f"start={t:.1f}s lat={s['latency_s']:.2f}s prompt={prompt} out={u['output_tokens']} usd={s['cost_usd']:.4f} cum={cum:.4f}")
    labels.append(f"T{s['turn']} {'errored' if kind == 'failed' else kind}" + (f" ({s['n_calls']})" if s["n_calls"] else ""))
    ax.text(1.02, y, f"{s['cost_usd']:.4f}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5)
    ax.text(1.27, y, f"{cum:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5)
    t += s["latency_s"]
# turn boundaries and the checker's flag
turn_end = {}
for i, s in enumerate(steps):
    turn_end[s["turn"]] = n - 1 - i
for tr, yy in turn_end.items():
    if tr < max(turn_end):
        ax.axhline(yy - 0.5, color="#BBBBBB", lw=0.5, ls="--")
ax.text(6.5, turn_end[0] - 0.5 + 0.12, "checker flags T0", ha="left", va="bottom", fontsize=7.5, color="#C44E52")
ax.set_yticks(range(n))
ax.set_yticklabels(list(reversed(labels)), fontsize=7.5)
ax.set_xlabel("seconds since task start (model latency)")
ax.text(1.02, n - 0.2, "USD", transform=ax.get_yaxis_transform(), fontsize=7.5, fontweight="bold")
ax.text(1.27, n - 0.2, "cum.", transform=ax.get_yaxis_transform(), fontsize=7.5, fontweight="bold")
ax.set_ylim(-0.6, n + 0.4)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.24, right=0.72, top=0.97, bottom=0.2)
fig.savefig(HERE.parent / "fig-waterfall.pdf")
print("total", round(cum, 6), "after turn0 share", round(sum(task["per_turn_cost_usd"][1:]) / task["cost_usd"], 3))

# ---------- Fig. 2 ----------
w = json.load(open(HERE / "out" / "prod.json"))["worst_accepted_packet"]
RC = {"spec": "#55A868", "rewrite": "#4C72B0", "judge": "#8C8C8C", "retry": "#DD8452", "research (web search)": "#8172B2", "other": "#CCB974"}
fig, ax = plt.subplots(figsize=(3.45, 1.5))
cum = 0.0
seen = set()
for i, x in enumerate(w["trace"]):
    lab = x["role"] if x["role"] not in seen else None
    seen.add(x["role"])
    ax.bar(i, x["usd"], bottom=cum, color=RC[x["role"]], width=0.85, label=lab)
    cum += x["usd"]
ax.set_xlabel("provider call, in order of completion")
ax.set_ylabel("cum. USD")
ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2, handlelength=1, labelspacing=0.2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.25)
fig.savefig(HERE.parent / "fig-packet.pdf")
print("packet total", round(cum, 4), "calls", len(w["trace"]))
