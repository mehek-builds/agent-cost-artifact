"""Overrun of stops and a held-out check of the in-sample thresholds (no API calls).

python3 analysis/loop_limits_heldout.py  -> prints tables, writes results/loop_limits_heldout.json

1. overrun: limits are checked after each model step, so a stopped task can run past its threshold. For the deadline
   rules (p90, p75 of successful-task wall time) the overrun is the seconds spent past the threshold; for the dollar
   budgets (p50, p75, p90 of successful-task cost, as in loop_limits_extra.py) it is the dollars past the budget and
   that excess as a share of the budget. Median over stopped tasks and maximum, per configuration and setting.
2. heldout: loop_limits.py and loop_limits_extra.py set the deadline and budget thresholds from the same 50 tasks
   they are replayed on. Here the 50 tasks are split at random into two halves; the threshold is set from the
   successful tasks of one half and applied to the other, and the two held-out halves are pooled, so every task is
   judged by a threshold it did not help set. Repeated for 200 random splits (seed 1). Reported per setting: success
   and cost per success on the pooled held-out tasks (mean and 5th to 95th percentile over splits) next to the
   in-sample values, and the change in cost per success against no limit on the same tasks.
"""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from replay import CONFIGS, load, step_cost  # noqa: E402
from loop_limits import run, pct  # noqa: E402
from transactional_stop import reactive  # noqa: E402

SPLITS = 200


def secs(t):
    return sum(s["latency_s"] for s in t["steps"])


def full_cost(t):
    return sum(step_cost(s) for s in t["steps"])


def budget_of(ok_costs, q):
    sc = sorted(ok_costs)
    return sc[min(len(sc) - 1, int(round(q * len(sc))) - 1)]


def eval_set(tasks, kind, arg):
    """-> (n_success, total_usd, n_stopped) of a rule on a task list."""
    if kind == "deadline":
        res = [run(t, "deadline", arg) for t in tasks]
        return sum(r[2] for r in res), sum(r[0] for r in res), sum(r[3] for r in res)
    res = [reactive(t, arg) for t in tasks]
    return sum(r[1] for r in res), sum(r[0] for r in res), sum(r[2] for r in res)


def main():
    out = {}
    for label, (log, _) in CONFIGS.items():
        tasks = list(load(log).values())
        n = len(tasks)
        ok = [t for t in tasks if t["success"]]
        ok_secs = [secs(t) for t in ok]
        ok_cost = [full_cost(t) for t in ok]
        base_s = len(ok)
        base_c = sum(full_cost(t) for t in tasks)
        settings = [("deadline", "p90", pct(ok_secs, .9)), ("deadline", "p75", pct(ok_secs, .75))] + \
                   [("budget", f"p{int(q * 100)}", budget_of(ok_cost, q)) for q in (.5, .75, .9)]
        rec = {"overrun": {}, "heldout": {}}
        for kind, name, thr in settings:
            key = f"{kind}_{name}"
            # overrun, in sample
            over, over_share = [], []
            stopped_n = 0
            for t in tasks:
                if kind == "deadline":
                    spent, sec, _, stopped = run(t, "deadline", thr)
                    if stopped:
                        over.append(sec - thr)
                else:
                    spent, _, stopped, _ = reactive(t, thr)
                    if stopped:
                        over.append(spent - thr)
                        over_share.append((spent - thr) / thr)
            rec["overrun"][key] = {"threshold": thr, "stopped": len(over), "median_excess": statistics.median(over) if over else None,
                                   "max_excess": max(over) if over else None,
                                   "median_excess_share_of_budget": statistics.median(over_share) if over_share else None,
                                   "max_excess_share_of_budget": max(over_share) if over_share else None}
            # in-sample values
            s_in, c_in, _ = eval_set(tasks, kind, thr)
            ins = {"success": s_in / n, "usd_per_success": c_in / s_in if s_in else None,
                   "d_usd_per_success_pct": 100 * ((c_in / s_in) / (base_c / base_s) - 1) if s_in else None}
            # held out
            rng = random.Random(1)
            succ, dps, dsucc = [], [], []
            for _ in range(SPLITS):
                idx = list(range(n))
                rng.shuffle(idx)
                halves = (idx[: n // 2], idx[n // 2:])
                tot_s = tot_c = 0.0
                for tr, te in (halves, halves[::-1]):
                    trk = [tasks[i] for i in tr if tasks[i]["success"]]
                    if not trk:
                        continue
                    if kind == "deadline":
                        thr_h = pct([secs(t) for t in trk], .9 if name == "p90" else .75)
                    else:
                        thr_h = budget_of([full_cost(t) for t in trk], float(name[1:]) / 100)
                    s, c, _ = eval_set([tasks[i] for i in te], kind, thr_h)
                    tot_s += s
                    tot_c += c
                succ.append(tot_s / n)
                if tot_s:
                    dps.append(100 * ((tot_c / tot_s) / (base_c / base_s) - 1))
                dsucc.append(100 * (tot_s - base_s) / n)

            def band(xs):
                xs = sorted(xs)
                return {"mean": statistics.mean(xs), "p05": xs[int(.05 * len(xs))], "p95": xs[int(.95 * len(xs)) - 1]}
            rec["heldout"][key] = {"in_sample": ins, "heldout_success": band(succ), "heldout_d_success_points": band(dsucc),
                                   "heldout_d_usd_per_success_pct": band(dps),
                                   "in_sample_d_success_points": 100 * (s_in - base_s) / n}
            h, i = rec["heldout"][key], ins
            print(f"{label:12s} {key:14s} thr {thr:8.4f} | overrun med {rec['overrun'][key]['median_excess']} max {rec['overrun'][key]['max_excess']} "
                  f"share med {rec['overrun'][key]['median_excess_share_of_budget']} | S in {100 * i['success']:.0f} held {100 * h['heldout_success']['mean']:.1f} "
                  f"({100 * h['heldout_success']['p05']:.0f}-{100 * h['heldout_success']['p95']:.0f}) | d$/succ in {i['d_usd_per_success_pct']:+.1f} held {h['heldout_d_usd_per_success_pct']['mean']:+.1f}")
        out[label] = rec
    (ROOT / "results" / "loop_limits_heldout.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
