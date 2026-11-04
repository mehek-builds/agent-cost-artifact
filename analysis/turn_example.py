"""One recorded user turn as a worked example of the stop-placement exposure (no API calls).

python3 analysis/turn_example.py  -> prints the turn, writes results/turn_example.json

Selection rule, fixed before looking at any stop policy: among the turns of the large-model, no-cache
configuration in which no step errored, take the one with the most tool calls (ties: first by task id, then
turn index). The output lists the model steps of that turn, the calls each made, and how many calls a stop
placed after each non-final step would leave applied. It does not apply any budget rule.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from replay import CONFIGS, load  # noqa: E402


def main():
    label = "large"
    log, _ = CONFIGS[label]
    cands = []
    for tid, t in sorted(load(log).items()):
        turns = {}
        for s in t["steps"]:
            turns.setdefault(s["turn"], []).append(s)
        for k, steps in sorted(turns.items()):
            if any(s["n_errors"] for s in steps):
                continue
            cands.append((-sum(s["n_calls"] for s in steps), tid, k, steps))
    cands.sort(key=lambda c: c[:3])
    neg, tid, k, steps = cands[0]
    calls = [s["n_calls"] for s in steps]
    out = {
        "config": label, "task_id": tid, "turn": k, "model_steps": len(steps),
        "step_kinds": [s["kind"] for s in steps], "calls_per_step": calls, "total_calls": -neg,
        "turn_cost_usd": round(sum(s["cost_usd"] for s in steps), 6),
        "calls_left_applied_by_stop_after_step": {str(i): sum(calls[: i + 1]) for i in range(len(steps) - 1)},
        "error_free_turns_considered": len(cands),
    }
    print(json.dumps(out, indent=1))
    (ROOT / "results" / "turn_example.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
