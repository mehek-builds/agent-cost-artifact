"""Numbers quoted in the text that combine outputs of bfcl.py, prod.py and fig_waterfall.py.
python3 -I derived.py  (run after bfcl.py and prod.py)"""
import json
from pathlib import Path

O = Path(__file__).resolve().parent / "out"
b = json.load(open(O / "bfcl.json"))
p = json.load(open(O / "prod.json"))
LOGS = Path(__file__).resolve().parents[4] / "logs"  # repository logs/

steps = {k: v["steps"] for k, v in b["checks"].items()}
print("step spans in the five complete step logs:", sum(steps.values()), steps)
print("frontier_low step records:", sum(1 for _ in open(LOGS / "steps_frontier_low.jsonl")))
print("max |sum(step) - task| USD:", max(v["max_abs_step_sum_minus_task_cost"] for v in b["checks"].values()))
print("max |reprice - logged| USD:", max(v["max_abs_reprice_minus_logged"] for v in b["checks"].values()))

for cfg, v in b["table_b"].items():
    ks = v["kind_share"]
    print(cfg, "error+repair share:", round(ks["error_step"] + ks["repair_step"], 3),
          "reread+first_read:", round(v["role_share"]["reread"] + v["role_share"]["first_read"], 3))

# task 62 in Fig. 1
t62 = [json.loads(l) for l in open(LOGS / "tasks_frontier.jsonl") if json.loads(l)["task_id"] == "multi_turn_base_62"][0]
med = b["table_a"]["frontier"]["succ_run_median"]
print("task62 cost", t62["cost_usd"], "vs median successful large run", med, "ratio", round(t62["cost_usd"] / med, 2))
print("task62 share after turn 0:", round(sum(t62["per_turn_cost_usd"][1:]) / t62["cost_usd"], 3))
print("task62 errored step cost", t62["failed_step_cost_usd"], "share", round(t62["failed_step_cost_usd"] / t62["cost_usd"], 3))

# runs and failures
a = b["table_a"]
print("successes large/small (uncached):", a["frontier"]["successes"], a["small"]["successes"])
print("cache cut in USD per success, large:", round(1 - a["frontier_cache"]["usd_per_success"] / a["frontier"]["usd_per_success"], 3),
      "small (cache+cap):", round(1 - a["small_cache_budget"]["usd_per_success"] / a["small"]["usd_per_success"], 3))

# production
tb = p["table_b_packet"]["roles"]
print("packet spend share retry+judge+rewrite:", round(tb["retry"]["share_of_spend"] + tb["judge"]["share_of_spend"] + tb["rewrite"]["share_of_spend"], 3))
print("packet calls share judge:", tb["judge"]["share_of_calls"], "spend share judge:", tb["judge"]["share_of_spend"])
print("search share of packet spend:", round(p["table_b_packet"]["search_usd_in_packets"] / p["table_b_packet"]["total_usd"], 3))
s = p["surge_days"]
print("surge Oct18-22 total:", round(sum(s[d]["usd"] for d in ["2026-10-18", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]), 2))
print("Oct 20 vs Oct1-14 median daily:", round(s["2026-10-20"]["usd"] / p["round1"]["oct1_14_median_daily_usd_unrounded"], 1))
print("surge share of 90-day console spend:", round(sum(s[d]["usd"] for d in ["2026-10-18", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]) / p["console_total_usd"], 3))
w = p["worst_accepted_packet"]
print("worst packet usd", w["usd"], "calls", w["calls"], "elapsed", w["elapsed_s"], "ratio", w["ratio_to_accepted_median"], w["role_calls"])
c = p["table_c"]
print("retry vs no-retry accepted medians:", c["accepted_packet_with_retry"]["median_usd"], c["accepted_packet_without_retry_with_calls"]["median_usd"],
      "ratio", round(c["accepted_packet_with_retry"]["median_usd"] / c["accepted_packet_without_retry_with_calls"]["median_usd"], 2))
print("unattributed share of window spend:", round(p["usd_without_operation_key"] / p["total_usd_with_search"], 3))
print("application_run share of window:", p["window_share_by_kind"]["application_run"])

from datetime import datetime
w0 = datetime.fromisoformat(p["window"]["first_event"][:19]); w1 = datetime.fromisoformat(p["window"]["last_event"][:19])
print("window days:", round((w1 - w0).total_seconds() / 86400, 2))
for cfg, v in b["table_a"].items():
    print(cfg, "succ p90/median:", round(v["succ_run_p90"] / v["succ_run_median"], 2))
ta = p["table_a"]["application_packet"]
print("packet p90/median:", round(ta["accepted_p90"] / ta["accepted_median"], 2))
print("total recorded runs W1:", sum(v["runs"] for v in b["table_a"].values()))
fails = {k: v["runs"] - v["successes"] for k, v in b["table_a"].items()}
print("failures per config:", fails)
import collections
kinds = collections.Counter()
for cfg in ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"]:
    for l in open(LOGS / f"steps_{cfg}.jsonl"):
        kinds[json.loads(l)["kind"]] += 1
print("step kinds in five step logs (a 'limit' kind marks the per-turn step limit):", dict(kinds))
print("small cached+cap vs large cached USD per success cut:", round(1 - a["small_cache_budget"]["usd_per_success"] / a["frontier_cache"]["usd_per_success"], 3))
turns = [json.loads(l)["turns"] for cfg in ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"] for l in open(LOGS / f"tasks_{cfg}.jsonl")]
print("turns per task min/max:", min(turns), max(turns))

# Sonnet 5 share of the backend key's token cost, monthly by-model report (no per-day model column exists)
import csv
rows = [r for r in csv.DictReader(open(Path(__file__).resolve().parents[2] / "production-aggregates" / "console-cost-by-key-model.csv"))
        if r["api_key_name"] == "first-1k-apps" and r["window"] == "2026-09-29~2026-10-27" and r["cost_type"] == "tokens"]
tot = sum(float(r["usd"]) for r in rows)
son = sum(float(r["usd"]) for r in rows if r["model"] == "claude-sonnet-5")
print("Sonnet 5 share of first-1k-apps token cost, 2026-09-29~2026-10-27:", round(100 * son / tot, 1), "% of", round(tot, 4))
