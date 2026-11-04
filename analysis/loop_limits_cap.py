"""How much the cached small model's five-call cap can matter for the step-limit rows (no API calls).

python3 analysis/loop_limits_cap.py  -> prints, writes results/loop_limits_cap.json

The cached small run capped tool calls at five per turn (n_budget_denied counts the denied calls). This script
counts the tasks and turns where the cap denied a call, replays the three step-per-turn limits on the cached
small run twice (all 50 tasks; the tasks where no call was denied), and counts how many of the 28 hard-stop
settings raised cost per successful task with and without that run's three step-limit settings.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from replay import load, step_cost  # noqa: E402
from loop_limits import run  # noqa: E402
from transactional_stop import reactive  # noqa: E402


def cps(tasks, rule, arg):
    res = [run(t, rule, arg) if rule else (sum(step_cost(s) for s in t["steps"]), 0, bool(t["success"]), False)
           for t in tasks]
    k = sum(int(r[2]) for r in res)
    return sum(r[0] for r in res) / k, k, sum(r[3] for r in res), [int(r[2]) for r in res]


def main():
    T = list(load("small_cache_budget").values())
    capped = [t for t in T if any(s["n_budget_denied"] > 0 for s in t["steps"])]
    free = [t for t in T if t not in capped]
    out = {"tasks_with_denied_call": len(capped),
           "turns_with_denied_call": len({(t["task_id"], s["turn"]) for t in capped for s in t["steps"] if s["n_budget_denied"] > 0}),
           "denied_calls": sum(s["n_budget_denied"] for t in capped for s in t["steps"]),
           "turns": len({(t["task_id"], s["turn"]) for t in T for s in t["steps"]}),
           "capped_tasks_solved": sum(int(t["success"]) for t in capped), "rows": {}}
    base_all, base_free = cps(T, None, None), cps(free, None, None)
    for k in (3, 4, 6):
        a, f = cps(T, "turn_steps", k), cps(free, "turn_steps", k)
        stopped_capped = sum(run(t, "turn_steps", k)[3] for t in capped)
        out["rows"][f"turn_steps={k}"] = {
            "all50": {"success": a[1], "stopped": a[2], "d_usd_per_success_pct": 100 * (a[0] / base_all[0] - 1)},
            "uncapped40": {"success": f[1], "of": len(free), "stopped": f[2], "unlimited_success": base_free[1],
                           "d_usd_per_success_pct": 100 * (f[0] / base_free[0] - 1)},
            "stopped_among_capped10": stopped_capped}
    full = [sum(step_cost(x) for x in t["steps"]) for t in T]
    sc = sorted(c for c, t in zip(full, T) if t["success"])
    stopped = {}
    for q in (0.75, 0.9):
        b = sc[min(len(sc) - 1, int(round(q * len(sc))) - 1)]
        stopped[q] = (b, {t["task_id"] for t in T if reactive(t, b)[2]})
    out["budget_p75_p90_same_tasks"] = {"p75_usd": stopped[0.75][0], "p90_usd": stopped[0.9][0],
                                        "stopped_p75": len(stopped[0.75][1]), "stopped_p90": len(stopped[0.9][1]),
                                        "same_set": stopped[0.75][1] == stopped[0.9][1]}
    pj = json.load(open(ROOT / "results" / "loop_limits_paired.json"))
    stops = [(c, r) for c in ("large", "large+cache", "small", "small+cache") for r in pj[c]
             if not r["setting"].startswith("budget")]
    out["raised_all28"] = sum(r["d_usd_per_success_pct"] > 0 for c, r in stops)
    rest = [(c, r) for c, r in stops if not (c == "small+cache" and r["setting"].startswith("turn_steps"))]
    out["settings_without_cached_small_step_limits"] = len(rest)
    out["raised_without_cached_small_step_limits"] = sum(r["d_usd_per_success_pct"] > 0 for c, r in rest)
    print(json.dumps(out, indent=1))
    (ROOT / "results" / "loop_limits_cap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
