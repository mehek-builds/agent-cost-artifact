"""Hard stops on runaway loops, priced on three axes: success, money and latency (no API calls).

python3 analysis/loop_limits.py  -> prints a table, writes results/loop_limits.json

A deployed agent usually carries limits that are not dollar budgets: a cap on model steps inside one user
turn (the loop guard), a cap on tool errors per task (the retry limit) and a wall-clock deadline. Each rule
below stops the recorded trace at the first step that breaks it; a stopped task counts as a failure, and its
cost and latency are what it spent up to and including that step. Replay is exact for cost and latency
because a stop only truncates a trace.

none          the recorded run
turn_steps=k  stop when a turn reaches its k-th model step without a final answer
errors=k      stop at the k-th failed tool step in the task
deadline=q    stop once wall time passes the q-th percentile of successful tasks' wall time
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from replay import CONFIGS, load, step_cost, wilson, boot_ratio_ci  # noqa: E402


def run(task, rule, arg):
    spent = secs = 0.0
    errs = 0
    in_turn = {}
    steps = task["steps"]
    for i, s in enumerate(steps):
        spent += step_cost(s)
        secs += s["latency_s"]
        errs += s["n_errors"] > 0
        in_turn[s["turn"]] = in_turn.get(s["turn"], 0) + 1
        last = i == len(steps) - 1
        if last:
            break
        if rule == "turn_steps" and in_turn[s["turn"]] >= arg and s["kind"] != "final":
            return spent, secs, False, True
        if rule == "errors" and errs >= arg:
            return spent, secs, False, True
        if rule == "deadline" and secs >= arg:
            return spent, secs, False, True
    return spent, secs, bool(task["success"]), False


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    out = {}
    for label, (log, _) in CONFIGS.items():
        tasks = list(load(log).values())
        ok_secs = [sum(s["latency_s"] for s in t["steps"]) for t in tasks if t["success"]]
        steps_per_turn = [sum(1 for s in t["steps"] if s["turn"] == k)
                          for t in tasks for k in {s["turn"] for s in t["steps"]}]
        rules = [("none", None)] + [("turn_steps", k) for k in (3, 4, 6)] + [("errors", k) for k in (1, 2)] + \
                [("deadline", pct(ok_secs, q)) for q in (0.9, 0.75)]
        rows = []
        for rule, arg in rules:
            res = [run(t, rule, arg) for t in tasks]
            cost = [r[0] for r in res]
            succ = [int(r[2]) for r in res]
            k = sum(succ)
            fail_spend = sum(c for c, s in zip(cost, succ) if not s)
            row = {
                "rule": rule if arg is None else f"{rule}={arg:.0f}" if rule != "deadline" else f"deadline={arg:.0f}s",
                "success": k / len(tasks), "success_ci": wilson(k, len(tasks)),
                "stopped": sum(r[3] for r in res),
                "usd_per_task": sum(cost) / len(tasks),
                "usd_per_success": sum(cost) / k if k else None,
                "usd_per_success_ci": boot_ratio_ci(cost, succ) if k else None,
                "failed_share_of_spend": fail_spend / sum(cost),
                "secs_p50": statistics.median(r[1] for r in res),
                "secs_p90": pct([r[1] for r in res], 0.9),
            }
            rows.append(row)
            print(f"{label:12s} {row['rule']:16s} succ {row['success']:.2f} stop {row['stopped']:2d} "
                  f"$/task {row['usd_per_task']:.4f} $/succ {row['usd_per_success'] or 0:.4f} "
                  f"fail$ {row['failed_share_of_spend']:.2f} p50 {row['secs_p50']:.0f}s p90 {row['secs_p90']:.0f}s")
        out[label] = {"rows": rows, "max_steps_in_turn": max(steps_per_turn),
                      "turns_with_6plus_steps": sum(n >= 6 for n in steps_per_turn), "turns": len(steps_per_turn)}
        print(f"{label:12s} turns {len(steps_per_turn)}, max steps in a turn {max(steps_per_turn)}, "
              f">=6 steps {out[label]['turns_with_6plus_steps']}")
    (ROOT / "results" / "loop_limits.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
