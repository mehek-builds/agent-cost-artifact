"""Extra replay numbers for the hard-stop analysis (no API calls).

python3 analysis/loop_limits_extra.py  -> prints tables, writes results/loop_limits_extra.json

Adds to loop_limits.py, from the same recorded runs and the same stop rules:
  steps_per_turn   percentiles of model steps per user turn, over turns of successful tasks and over all turns
  stopped          for every limit setting: tasks stopped, and how many of those the recorded run had solved
  deltas           change of cost per task and cost per success against no limit, per setting
  task_max_steps  the most steps any turn of a task took, over successful tasks (what a per-turn cap must clear,
                   because a task survives only if every one of its turns does)
  aggregate        stopped tasks that the recorded run had solved, summed over the seven settings, for the two large
                   and the two small configurations
  failed_vs_ok     median cost of a failed run and of a successful run
  dollar_budgets   a per-task dollar budget set at the 50th, 75th and 90th percentile of the cost of successful
                   tasks, applied as a reactive stop (stop after the step whose cost crosses the budget):
                   success, tasks stopped, change in cost per task and in cost per success
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from replay import CONFIGS, load, step_cost  # noqa: E402
from loop_limits import run, pct  # noqa: E402
from transactional_stop import reactive  # noqa: E402


def per_turn_steps(task):
    n = {}
    for s in task["steps"]:
        n[s["turn"]] = n.get(s["turn"], 0) + 1
    return list(n.values())


def main():
    out = {}
    for label, (log, _) in CONFIGS.items():
        T = load(log)
        tasks = list(T.values())
        ok = [t for t in tasks if t["success"]]
        ok_secs = [sum(s["latency_s"] for s in t["steps"]) for t in ok]
        ok_turns = [n for t in ok for n in per_turn_steps(t)]
        all_turns = [n for t in tasks for n in per_turn_steps(t)]
        ok_max = [max(per_turn_steps(t)) for t in ok]
        rec = {"task_max_steps": {"tasks": len(ok_max), "p50": pct(ok_max, .5), "p75": pct(ok_max, .75),
                                  "p90": pct(ok_max, .9), "max": max(ok_max),
                                  "tasks_over_3": sum(n > 3 for n in ok_max), "tasks_over_4": sum(n > 4 for n in ok_max),
                                  "tasks_over_5": sum(n > 5 for n in ok_max), "tasks_over_6": sum(n > 6 for n in ok_max)},
               "steps_per_turn": {
            "successful_tasks": {"turns": len(ok_turns), "p50": pct(ok_turns, .5), "p75": pct(ok_turns, .75),
                                 "p90": pct(ok_turns, .9), "max": max(ok_turns),
                                 "turns_over_3": sum(n > 3 for n in ok_turns), "turns_over_4": sum(n > 4 for n in ok_turns),
                                 "turns_over_6": sum(n > 6 for n in ok_turns)},
            "all_tasks": {"turns": len(all_turns), "p90": pct(all_turns, .9), "max": max(all_turns),
                          "turns_over_6": sum(n > 6 for n in all_turns)}}}
        rules = [("none", None)] + [("turn_steps", k) for k in (3, 4, 6)] + [("errors", k) for k in (1, 2)] + \
                [("deadline", pct(ok_secs, q)) for q in (0.9, 0.75)]
        rows, base = [], None
        for rule, arg in rules:
            res = [run(t, rule, arg) for t in tasks]
            cost = [r[0] for r in res]
            k = sum(r[2] for r in res)
            stopped = [t for t, r in zip(tasks, res) if r[3]]
            row = {"rule": rule if arg is None else (f"{rule}={arg:.0f}" if rule != "deadline" else f"deadline={arg:.0f}s"),
                   "deadline_exact_s": arg if rule == "deadline" else None,
                   "stopped": len(stopped), "stopped_that_succeeded": sum(1 for t in stopped if t["success"]),
                   "success_n": k, "usd_total": sum(cost), "usd_per_task": sum(cost) / len(tasks),
                   "usd_per_success": sum(cost) / k}
            if base is None:
                base = row
            row["d_usd_per_task_pct"] = 100 * (row["usd_per_task"] / base["usd_per_task"] - 1)
            row["d_usd_per_success_pct"] = 100 * (row["usd_per_success"] / base["usd_per_success"] - 1)
            row["d_success_points"] = 100 * (k - base["success_n"]) / len(tasks)
            rows.append(row)
        rec["stopped"] = rows
        full = {i: sum(step_cost(s) for s in t["steps"]) for i, t in T.items()}
        rec["failed_vs_ok"] = {
            "median_failed_run_usd": statistics.median(full[i] for i, t in T.items() if not t["success"]),
            "median_successful_run_usd": statistics.median(full[i] for i, t in T.items() if t["success"])}
        sc = sorted(full[i] for i, t in T.items() if t["success"])
        db = {}
        for q in (0.5, 0.75, 0.9):
            b = sc[min(len(sc) - 1, int(round(q * len(sc))) - 1)]
            res = [reactive(t, b) for t in tasks]  # (spent, success, stopped, half_applied)
            cost = [r[0] for r in res]
            k = sum(r[1] for r in res)
            db[f"q={q}"] = {"budget_usd": b, "success": k / len(tasks), "stopped": sum(r[2] for r in res),
                            "stopped_that_succeeded": sum(1 for t, r in zip(tasks, res) if r[2] and t["success"]),
                            "d_usd_per_task_pct": 100 * (sum(cost) / sum(full.values()) - 1),
                            "d_usd_per_success_pct": 100 * ((sum(cost) / k) / (sum(full.values()) / len(ok)) - 1)}
        rec["dollar_budgets"] = db
        out[label] = rec
        sp = rec["steps_per_turn"]["successful_tasks"]
        print(f"\n{label}: steps/turn (successful tasks) p50 {sp['p50']} p75 {sp['p75']} p90 {sp['p90']} max {sp['max']} "
              f"turns>3 {sp['turns_over_3']} >4 {sp['turns_over_4']} >6 {sp['turns_over_6']} of {sp['turns']}; "
              f"all-task p90 {rec['steps_per_turn']['all_tasks']['p90']}")
        tm = rec["task_max_steps"]
        print(f"  per-task max steps (successful tasks) p50 {tm['p50']} p75 {tm['p75']} p90 {tm['p90']} max {tm['max']}; "
              f"tasks over 3/4/5/6: {tm['tasks_over_3']}/{tm['tasks_over_4']}/{tm['tasks_over_5']}/{tm['tasks_over_6']} of {tm['tasks']}")
        print("  median run cost failed %.4f ok %.4f" % (rec["failed_vs_ok"]["median_failed_run_usd"], rec["failed_vs_ok"]["median_successful_run_usd"]))
        for r in rows:
            print(f"  {r['rule']:15s} stopped {r['stopped']:2d} of which solved {r['stopped_that_succeeded']:2d} "
                  f"d$/task {r['d_usd_per_task_pct']:+6.1f}% d$/succ {r['d_usd_per_success_pct']:+6.1f}% dS {r['d_success_points']:+5.1f}")
        for q, v in db.items():
            print(f"  budget {q} ${v['budget_usd']:.4f} S {100*v['success']:.0f} stopped {v['stopped']} solved {v['stopped_that_succeeded']} "
                  f"d$/task {v['d_usd_per_task_pct']:+.1f}% d$/succ {v['d_usd_per_success_pct']:+.1f}%")
    agg = {}
    for name, cfgs in (("large", ("large", "large+cache")), ("small", ("small", "small+cache"))):
        st = sum(r["stopped"] for c in cfgs for r in out[c]["stopped"] if r["rule"] != "none")
        sv = sum(r["stopped_that_succeeded"] for c in cfgs for r in out[c]["stopped"] if r["rule"] != "none")
        agg[name] = {"settings": 7 * len(cfgs), "stopped": st, "stopped_that_succeeded": sv, "share": sv / st}
        print(f"aggregate {name}: stopped {st}, recorded run solved {sv} ({100 * sv / st:.0f}%)")
    out["aggregate"] = agg
    (ROOT / "results" / "loop_limits_extra.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
