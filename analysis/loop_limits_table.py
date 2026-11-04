"""Numbers for the merged hard-stop and dollar-budget table, and the per-task cost interval (no API calls).

python3 analysis/loop_limits_table.py  -> prints the table, writes results/loop_limits_table.json

For the four configurations of loop_limits.py and the ten settings (seven hard stops, three dollar budgets of
loop_limits_extra.py) it gives success, cost per successful task, tasks the stop ended that the recorded run had
solved, the 90th-percentile task wall time (a dollar budget stops a task after the step that takes its spend to
the budget, so its time is the time up to and including that step), the change in cost per task against no
limit with a paired 95% interval (10,000 task resamples, seed 42), and the p90 time as a share of the unlimited
p90.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from replay import CONFIGS, load, step_cost, SEED, BOOT  # noqa: E402
from loop_limits import run, pct  # noqa: E402
from transactional_stop import turns  # noqa: E402


def reactive_secs(task, budget):
    """Same stop as transactional_stop.reactive, also returning the wall time spent."""
    spent = secs = 0.0
    tt = turns(task)
    for ti, steps in enumerate(tt):
        for si, s in enumerate(steps):
            spent += step_cost(s)
            secs += s["latency_s"]
            last = ti == len(tt) - 1 and si == len(steps) - 1
            if spent >= budget and not last:
                return spent, secs, False, True
    return spent, secs, bool(task["success"]), False


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    for label, (log, _) in CONFIGS.items():
        tasks = list(load(log).values())
        n = len(tasks)
        idx = rng.integers(0, n, size=(BOOT, n))
        ok_secs = [sum(s["latency_s"] for s in t["steps"]) for t in tasks if t["success"]]
        full = np.array([sum(step_cost(s) for s in t["steps"]) for t in tasks])
        full_secs = [sum(s["latency_s"] for s in t["steps"]) for t in tasks]
        succ0 = [int(t["success"]) for t in tasks]
        k0 = sum(succ0)
        sc = sorted(c for c, s in zip(full, succ0) if s)
        settings = [("turn_steps", k, f"turn_steps={k}") for k in (3, 4, 6)] + \
                   [("errors", k, f"errors={k}") for k in (1, 2)] + \
                   [("deadline", pct(ok_secs, q), f"deadline=p{int(q * 100)}") for q in (0.9, 0.75)]
        rows = [("none", full, full_secs, succ0, 0)]
        for rule, arg, name in settings:
            res = [run(t, rule, arg) for t in tasks]
            rows.append((name, np.array([r[0] for r in res]), [r[1] for r in res], [int(r[2]) for r in res],
                         sum(r[3] for r in res)))
        for q in (0.5, 0.75, 0.9):
            b = sc[min(len(sc) - 1, int(round(q * len(sc))) - 1)]
            res = [reactive_secs(t, b) for t in tasks]
            rows.append((f"budget=p{int(q * 100)}", np.array([r[0] for r in res]), [r[1] for r in res],
                         [int(r[2]) for r in res], sum(r[3] for r in res)))
        base_p90 = pct(full_secs, 0.9)
        rec = []
        for name, cost, secs, succ, stopped in rows:
            k1 = sum(succ)
            lost = sum(1 for a, b in zip(succ0, succ) if a and not b)
            ratio = (cost[idx].sum(1) / full[idx].sum(1) - 1) * 100
            p90 = pct(secs, 0.9)
            row = {"setting": name, "success_pct": 100 * k1 / n, "usd_per_success": float(cost.sum() / k1),
                   "lost": lost, "stopped": stopped, "p90_s": p90, "p90_cut_s": base_p90 - p90,
                   "p90_cut_share_pct": 100 * (base_p90 - p90) / base_p90,
                   "d_usd_per_task_pct": float(100 * (cost.sum() / full.sum() - 1)),
                   "d_usd_per_task_ci95": [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))]}
            rec.append(row)
            print(f"{label:12s} {name:14s} S {row['success_pct']:4.0f} $/succ {row['usd_per_success']:.4f} lost {lost:2d} "
                  f"p90 {p90:5.1f} cut {row['p90_cut_s']:4.1f}s ({row['p90_cut_share_pct']:4.1f}%) "
                  f"d$/task {row['d_usd_per_task_pct']:+6.1f}% [{row['d_usd_per_task_ci95'][0]:+6.1f},{row['d_usd_per_task_ci95'][1]:+6.1f}]")
        out[label] = rec
    (ROOT / "results" / "loop_limits_table.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
