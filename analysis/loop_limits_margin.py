"""Margin between twice the 90th percentile and the largest successful task (no API calls).

python3 analysis/loop_limits_margin.py  -> prints a table, writes results/loop_limits_margin.json

For each of the four configurations it takes the successful tasks of the recorded run and reports, for the most
steps any turn took, the task wall time and the task cost: the 90th percentile (same rule as loop_limits.pct),
the maximum, and whether twice the 90th percentile is at or above the maximum. It supports the rule that a limit
set at twice the 90th percentile of successful work would have stopped no successful task.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from replay import CONFIGS, load, step_cost  # noqa: E402
from loop_limits import pct  # noqa: E402


def main():
    out = {}
    for label, (log, _) in CONFIGS.items():
        tasks = [t for t in load(log).values() if t["success"]]
        quantities = {
            "max_turn_steps": [max(sum(1 for s in t["steps"] if s["turn"] == k) for k in {s["turn"] for s in t["steps"]})
                               for t in tasks],
            "wall_s": [sum(s["latency_s"] for s in t["steps"]) for t in tasks],
            "cost_usd": [sum(step_cost(s) for s in t["steps"]) for t in tasks],
        }
        rec = {"successful_tasks": len(tasks)}
        for name, xs in quantities.items():
            p90, mx = pct(xs, 0.9), max(xs)
            rec[name] = {"p90": p90, "twice_p90": 2 * p90, "max": mx, "max_over_p90": mx / p90,
                         "twice_p90_clears_max": 2 * p90 >= mx}
            print(f"{label:12s} {name:15s} p90 {p90:9.4f} 2xp90 {2 * p90:9.4f} max {mx:9.4f} "
                  f"max/p90 {mx / p90:4.2f} clears {2 * p90 >= mx}")
        out[label] = rec
    (ROOT / "results" / "loop_limits_margin.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
