"""Input split (first read / carried / new) for step 7 of multi_turn_base_62 (config frontier).
python3 -I span_split.py"""
import json
from pathlib import Path
LOGS = Path(__file__).resolve().parents[4] / "logs"  # repository logs/
PRICES = json.load(open(LOGS.parent / "prices.json"))["models"]
S = sorted([json.loads(l) for l in open(LOGS / "steps_frontier.jsonl")
            if json.loads(l)["task_id"] == "multi_turn_base_62"], key=lambda s: s["step"])
prevP = None
for s in S:
    u = s["usage"]; R, I, W = u["cache_read_tokens"], u["input_tokens"], u["cache_write_tokens"]
    P = I + R + W; p = PRICES[s["model"]]
    if "tier_low" in p: p = p["tier_low"] if P <= 100000 else p["tier_high"]
    in_usd = I * p["input_per_mtok"] / 1e6 + W * p["cache_write_5min_per_mtok"] / 1e6 + R * p["cache_read_per_mtok"] / 1e6
    if s["step"] == 7:
        carried = min(prevP, P)
        print("step", s["step"], "turn", s["turn"], "P", P, "prevP", prevP, "cost_usd", s["cost_usd"],
              "in_usd %.6f" % in_usd, "carried_usd %.6f" % (in_usd * carried / P), "new_input_usd %.6f" % (in_usd * (P - carried) / P),
              "billed_carried_usd (no cache => equal) %.6f" % (in_usd * carried / P))
    prevP = P
