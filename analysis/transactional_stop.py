"""Where a budget stop lands relative to tool-call turns, replayed on the recorded runs (no API calls).

python3 analysis/transactional_stop.py  -> prints a table, writes results/transactional_stop.json

A user turn is the unit of external effect: its steps call tools that change state (files, orders,
messages), and the turn is complete when the agent answers. A stop that lands after some of a turn's tool
calls ran and before the turn finished leaves those effects half-applied. Two stop rules are compared at a
per-task budget set at a percentile of the cost of successful tasks:

reactive       stop after the step whose cost crosses the budget (the replay.py guard)
transactional  before each turn, reserve the expected cost of a turn (the mean cost of this task's
               completed turns, or for the first turn the median first-turn cost of the other tasks); start the
               turn only if spend plus the reservation fits the budget, and never stop inside a turn
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from replay import CONFIGS, load, step_cost  # noqa: E402


def turns(task):
    out = {}
    for s in task["steps"]:
        out.setdefault(s["turn"], []).append(s)
    return [out[k] for k in sorted(out)]


def reactive(task, budget):
    spent = 0.0
    tt = turns(task)
    for ti, steps in enumerate(tt):
        called = False
        for si, s in enumerate(steps):
            spent += step_cost(s)
            called = called or s["n_calls"] > 0
            last = ti == len(tt) - 1 and si == len(steps) - 1
            if spent >= budget and not last:
                half = called and si < len(steps) - 1
                return spent, False, True, half
    return spent, bool(task["success"]), False, False


def transactional(task, budget, first_turn_reserve):
    spent, done = 0.0, []
    tt = turns(task)
    for ti, steps in enumerate(tt):
        reserve = statistics.mean(done) if done else first_turn_reserve
        if spent + reserve > budget and ti > 0:
            return spent, False, True, False
        if spent + reserve > budget and ti == 0:
            return 0.0, False, True, False
        c = sum(step_cost(s) for s in steps)
        spent += c
        done.append(c)
    return spent, bool(task["success"]), False, False


def main():
    res = {}
    for label in ("large+cache", "small+cache", "large", "small"):
        T = load(CONFIGS[label][0])
        ids = sorted(T)
        full = {i: sum(step_cost(s) for s in T[i]["steps"]) for i in ids}
        sc = sorted(full[i] for i in ids if T[i]["success"])
        first_costs = {i: sum(step_cost(s) for s in turns(T[i])[0]) for i in ids}
        rows = {}
        for q in (0.5, 0.75, 0.9):
            b = sc[min(len(sc) - 1, int(round(q * len(sc))) - 1)]
            for name in ("reactive", "transactional"):
                out = []
                for i in ids:
                    if name == "reactive":
                        out.append(reactive(T[i], b))
                    else:
                        others = [first_costs[j] for j in ids if j != i]
                        out.append(transactional(T[i], b, statistics.median(others)))
                over = [o[0] - b for o in out if o[0] > b]
                rows[f"q={q} {name}"] = {"budget": b, "success": sum(o[1] for o in out) / len(ids),
                                         "stopped": sum(o[2] for o in out), "half_applied_turns": sum(o[3] for o in out),
                                         "tasks_over_budget": len(over), "max_overshoot_usd": max(over) if over else 0.0,
                                         "median_overshoot_share": statistics.median([v / b for v in over]) if over else 0.0,
                                         "saving": 1 - sum(o[0] for o in out) / sum(full.values())}
        res[label] = rows
    (ROOT / "results" / "transactional_stop.json").write_text(json.dumps(res, indent=1))
    for label, rows in res.items():
        for k, v in rows.items():
            print(f"{label:12s} {k:22s} succ {v['success']:.2f} stopped {v['stopped']:2d} half {v['half_applied_turns']:2d} "
                  f"over {v['tasks_over_budget']:2d} max_over ${v['max_overshoot_usd']:.4f} med_over {v['median_overshoot_share']:.2f} save {v['saving']:.2f}")


if __name__ == "__main__":
    main()
