"""Replication of the hard-stop replay on the third recorded run of each model (no API calls).

python3 analysis/loop_limits_low.py  -> prints a table, writes results/loop_limits_low.json

The main replay (loop_limits.py, loop_limits_paired.py) uses four recorded runs of the same 50 tasks. Each model
also has a third recorded run at low reasoning effort. The small model's (logs/tasks_small_low.jsonl) is complete;
the large model's (tasks_frontier_low.jsonl) lost its per-step records for 42 of 50 tasks and is not replayed.
This script applies the same seven hard stops and three dollar budgets to the small model's low-effort run, with
thresholds set on the run itself as in the main replay, and reports success, change in cost per task and per success against no
limit, the paired 95% interval (10,000 task resamples, seed 42) and McNemar's exact test on lost successes.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from replay import load, step_cost, SEED, BOOT, mcnemar_exact  # noqa: E402
from loop_limits import run, pct  # noqa: E402
from loop_limits_paired import paired  # noqa: E402
from transactional_stop import reactive  # noqa: E402

# The large model's low-effort run lost its per-step records when the API credit limit stopped it (8 of 50 tasks
# have steps), so it cannot be replayed; only the small model's low-effort run is replicated.
LOW = {"small, low effort": "small_low"}


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    for label, log in LOW.items():
        T = load(log)
        tasks = list(T.values())
        n = len(tasks)
        idx = rng.integers(0, n, size=(BOOT, n))
        ok_secs = [sum(s["latency_s"] for s in t["steps"]) for t in tasks if t["success"]]
        full = [sum(step_cost(s) for s in t["steps"]) for t in tasks]
        succ0 = [int(t["success"]) for t in tasks]
        sc = sorted(c for c, s in zip(full, succ0) if s)
        settings = [("turn_steps", k, f"turn_steps={k}") for k in (3, 4, 6)] + \
                   [("errors", k, f"errors={k}") for k in (1, 2)] + \
                   [("deadline", pct(ok_secs, q), f"deadline=p{int(q * 100)}") for q in (0.9, 0.75)]
        rows = []
        for rule, arg, name in settings:
            res = [run(t, rule, arg) for t in tasks]
            rows.append((name, [r[0] for r in res], [int(r[2]) for r in res]))
        for q in (0.5, 0.75, 0.9):
            b = sc[min(len(sc) - 1, int(round(q * len(sc))) - 1)]
            res = [reactive(t, b) for t in tasks]
            rows.append((f"budget=p{int(q * 100)}", [r[0] for r in res], [int(r[1]) for r in res]))
        k0 = sum(succ0)
        rec = {"n_tasks": n, "success_unlimited": k0 / n, "usd_per_success_unlimited": sum(full) / k0, "settings": []}
        for name, cost, succ in rows:
            k1 = sum(succ)
            d = 100 * ((sum(cost) / k1) / (sum(full) / k0) - 1)
            dt = 100 * (sum(cost) / sum(full) - 1)
            lo, hi = paired(full, succ0, cost, succ, idx)
            lost = sum(1 for a, b in zip(succ0, succ) if a and not b)
            gained = sum(1 for a, b in zip(succ0, succ) if b and not a)
            row = {"setting": name, "success_points": 100 * (k1 - k0) / n, "d_usd_per_task_pct": dt,
                   "d_usd_per_success_pct": d, "ci95_pct": [lo, hi], "excludes_zero": lo > 0 or hi < 0,
                   "lost": lost, "gained": gained, "mcnemar_p": mcnemar_exact(lost, gained)}
            rec["settings"].append(row)
            print(f"{label:18s} {name:14s} dS {row['success_points']:+6.1f} d$/task {dt:+6.1f}% d$/succ {d:+7.1f}% "
                  f"CI [{lo:+7.1f},{hi:+7.1f}] lost {lost} gained {gained} p={row['mcnemar_p']:.4f}")
        out[label] = rec
    hs = [r for v in out.values() for r in v["settings"] if not r["setting"].startswith("budget")]
    out["summary_hard_stops"] = {
        "settings": len(hs),
        "lowered_cost_per_task": sum(r["d_usd_per_task_pct"] < 0 for r in hs),
        "raised_cost_per_success": sum(r["d_usd_per_success_pct"] > 0 for r in hs),
        "lowered_cost_per_success": [(r["setting"], round(r["d_usd_per_success_pct"], 1)) for r in hs
                                     if r["d_usd_per_success_pct"] <= 0],
        "ci_excludes_zero": sum(r["excludes_zero"] for r in hs),
        "ci_excludes_zero_below": sum(r["ci95_pct"][1] < 0 for r in hs),
        "max_d_usd_per_task_pct": max(r["d_usd_per_task_pct"] for r in hs),
        "min_d_usd_per_task_pct": min(r["d_usd_per_task_pct"] for r in hs)}
    print(json.dumps(out["summary_hard_stops"], indent=1))
    (ROOT / "results" / "loop_limits_low.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
