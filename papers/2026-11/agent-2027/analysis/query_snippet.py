"""Runs the group-by shown in Section II of the paper over the recorded Workload 1 logs and prints its result.
python3 -I query_snippet.py   (needs pandas)"""
import json
import os
from pathlib import Path
import pandas as pd
LOGS = Path(os.environ.get("AGENT_ARTIFACT_ROOT") or Path(__file__).resolve().parents[4]) / "logs"
CFG = ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"]
spans = pd.concat([pd.read_json(LOGS / f"steps_{c}.jsonl", lines=True) for c in CFG])
units = pd.concat([pd.read_json(LOGS / f"tasks_{c}.jsonl", lines=True) for c in CFG])
units["unit_id"] = units.config + "/" + units.task_id
spans["unit_id"] = spans.config + "/" + spans.task_id
units["flag_turn"] = pd.to_numeric(units.grade_error.str.extract(r"@turn(\d+)")[0])
# ---- the snippet printed in the paper ----
s = spans.merge(units[["unit_id", "success", "flag_turn"]], on="unit_id")
tot = s.cost_usd.sum()
fail = s[~s.success]
fail_share = fail.cost_usd.sum() / tot
fl = fail[fail.flag_turn.notna()]
late = fl[fl.turn > fl.flag_turn]
after = late.cost_usd.sum() / fl.cost_usd.sum()
# ---- end of snippet ----
print("failure-cost share (pooled five step-logged configs):", round(fail_share, 3))
print("share of failed-run spend after the flag turn:", round(after, 3))
print("per config failure share:", {c: round(g[~g.success].cost_usd.sum() / g.cost_usd.sum(), 3) for c, g in s.groupby("config")})
