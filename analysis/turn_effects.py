"""Tool calls per user turn in the recorded runs: how many actions a mid-turn stop can cut through (no API calls).

python3 analysis/turn_effects.py  -> prints a table, writes results/turn_effects.json

A user turn is the unit of external effect: its tool calls change the state of the task's simulated services, and
the turn is complete when the agent answers. A stop inside a turn that has already run k >= 1 calls and has more to
make leaves k actions applied for a request that is never finished. This counts calls per turn and how many calls
precede the turn's last step, which is what a stop on an arbitrary step could leave behind.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from replay import CONFIGS, load  # noqa: E402


def main():
    out = {}
    for label, (log, _) in CONFIGS.items():
        per_turn, exposed = [], []
        for t in load(log).values():
            turns = {}
            for s in t["steps"]:
                turns.setdefault(s["turn"], []).append(s)
            for steps in turns.values():
                calls = [s["n_calls"] for s in steps]
                per_turn.append(sum(calls))
                # a stop before the turn's last step can leave behind the calls made so far
                exposed.append(max((sum(calls[:i + 1]) for i in range(len(calls) - 1)), default=0))
        n = len(per_turn)
        row = {
            "turns": n,
            "calls_per_turn_mean": statistics.mean(per_turn),
            "calls_per_turn_max": max(per_turn),
            "share_turns_2plus_calls": sum(c >= 2 for c in per_turn) / n,
            "share_turns_exposed": sum(e >= 1 for e in exposed) / n,
            "exposed_calls_max": max(exposed),
        }
        out[label] = row
        print(f"{label:12s} turns {n} calls/turn mean {row['calls_per_turn_mean']:.2f} max {row['calls_per_turn_max']} "
              f">=2 calls {row['share_turns_2plus_calls']:.2f} exposed {row['share_turns_exposed']:.2f} "
              f"max exposed {row['exposed_calls_max']}")
    (ROOT / "results" / "turn_effects.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
