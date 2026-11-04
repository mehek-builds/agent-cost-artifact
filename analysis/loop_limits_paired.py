"""Paired uncertainty for the hard-stop replay (no API calls).

python3 analysis/loop_limits_paired.py  -> prints tables, writes results/loop_limits_paired.json

loop_limits.py reports each setting's own interval. Every limited arm and its unlimited arm are the same 50
tasks, so the quantity that matters is the paired difference. For each configuration this script resamples the
50 tasks once (10,000 resamples, seed 42), evaluates the unlimited run and every limited setting on the same
resample, and takes the percentile interval of

    d_usd_per_success_pct = 100 * (cost per success, limited / cost per success, unlimited - 1)

It also gives McNemar's exact test on success (a stop can only turn a success into a failure, so the
discordant pairs are the stopped tasks that the recorded run had solved), and for the deadline rules how far a
stopped task ran past its threshold (the rule is checked after each model step).

Settings: the seven hard stops of loop_limits.py and the three reactive dollar budgets of
loop_limits_extra.py, for the four configurations.
"""
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from replay import CONFIGS, load, step_cost, SEED, BOOT, mcnemar_exact  # noqa: E402
from loop_limits import run, pct  # noqa: E402
from transactional_stop import reactive  # noqa: E402


def paired(cost0, succ0, cost1, succ1, idx):
    """Percentile interval of the relative change in cost per success, limited vs unlimited, on shared resamples."""
    c0, s0 = np.asarray(cost0), np.asarray(succ0)
    c1, s1 = np.asarray(cost1), np.asarray(succ1)
    k0, k1 = s0[idx].sum(1), s1[idx].sum(1)
    ok = (k0 > 0) & (k1 > 0)
    r = 100 * ((c1[idx].sum(1)[ok] / k1[ok]) / (c0[idx].sum(1)[ok] / k0[ok]) - 1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    rng = np.random.default_rng(SEED)
    out, flat = {}, []
    for label, (log, _) in CONFIGS.items():
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
            rows.append(dict(name=name, family=rule, arg=arg, cost=[r[0] for r in res], succ=[int(r[2]) for r in res],
                             stopped=[r[3] for r in res], secs=[r[1] for r in res]))
        for q in (0.5, 0.75, 0.9):
            b = sc[min(len(sc) - 1, int(round(q * len(sc))) - 1)]
            res = [reactive(t, b) for t in tasks]
            rows.append(dict(name=f"budget=p{int(q * 100)}", family="budget", arg=b, cost=[r[0] for r in res],
                             succ=[int(r[1]) for r in res], stopped=[r[2] for r in res], secs=None))
        rec = []
        for r in rows:
            k0, k1 = sum(succ0), sum(r["succ"])
            d = 100 * ((sum(r["cost"]) / k1) / (sum(full) / k0) - 1)
            lo, hi = paired(full, succ0, r["cost"], r["succ"], idx)
            lost = sum(1 for a, b in zip(succ0, r["succ"]) if a and not b)
            gained = sum(1 for a, b in zip(succ0, r["succ"]) if b and not a)
            row = {"setting": r["name"], "success_points": 100 * (k1 - k0) / n, "d_usd_per_success_pct": d,
                   "ci95_pct": [lo, hi], "excludes_zero": lo > 0 or hi < 0, "mcnemar_lost": lost,
                   "mcnemar_gained": gained, "mcnemar_p": mcnemar_exact(lost, gained)}
            if r["family"] == "deadline":
                over = [s - r["arg"] for s, st in zip(r["secs"], r["stopped"]) if st]
                row["deadline_s"] = r["arg"]
                row["stopped"] = len(over)
                row["overrun_s_median"] = statistics.median(over) if over else 0.0
                row["overrun_s_max"] = max(over) if over else 0.0
            rec.append(row)
            flat.append((r["family"], row))
            print(f"{label:12s} {row['setting']:16s} dS {row['success_points']:+6.1f}  d$/succ {d:+7.1f}%  "
                  f"CI [{lo:+7.1f}, {hi:+7.1f}]  McNemar {lost}-{gained} p={row['mcnemar_p']:.4f}")
        out[label] = rec
    stops = [r for f, r in flat if f != "budget"]
    buds = [r for f, r in flat if f == "budget"]

    def summ(rs):
        return {"settings": len(rs),
                "ci_excludes_zero": sum(r["excludes_zero"] for r in rs),
                "ci_excludes_zero_above": sum(r["ci95_pct"][0] > 0 for r in rs),
                "ci_excludes_zero_below": sum(r["ci95_pct"][1] < 0 for r in rs),
                "point_estimate_raised": sum(r["d_usd_per_success_pct"] > 0 for r in rs),
                "mcnemar_p_below_0.05": sum(r["mcnemar_p"] < 0.05 for r in rs)}
    out["summary_hard_stops"] = summ(stops)
    out["summary_dollar_budgets"] = summ(buds)
    # where the 28 intervals do not exclude zero
    out["hard_stops_not_resolved"] = [f"{k} {r['setting']}" for k, v in out.items() if k in CONFIGS for r in v
                                      if r["setting"].split("=")[0] != "budget" and not r["excludes_zero"]]
    print(json.dumps({k: out[k] for k in ("summary_hard_stops", "summary_dollar_budgets", "hard_stops_not_resolved")}, indent=1))
    (ROOT / "results" / "loop_limits_paired.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
