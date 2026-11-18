"""Task-level bootstrap (seed 42, 10,000 resamples) for the Workload 1 failure-mode rows of the failure-economics table:
median cost ratio (run cost / median successful run of the same configuration, recomputed per resample) and
median after-flag share per failure mode. Same task resampling as bootstrap_bfcl.py (50 task ids shared across the five step-logged configurations).
python3 -I bootstrap_modes.py -> out/bootstrap_modes.json"""
import json, math, random, statistics as st
from pathlib import Path
LOGS = Path(__file__).resolve().parents[4] / "logs"  # repository logs/
CFG = ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"]
OUT = Path(__file__).resolve().parent / "out"
T = {c: {t["task_id"]: t for t in map(json.loads, open(LOGS / f"tasks_{c}.jsonl"))} for c in CFG}
ids = sorted(set.intersection(*[set(T[c]) for c in CFG]))
MODES = ["fail_state_mismatch", "fail_response_mismatch", "hit_tool_call_cap", "success_after_tool_error", "success_clean"]

def labels(t):
    L = []
    if not t["success"]: L.append("fail_" + (t["grade_error"] or "unknown").split("@")[0])
    else: L.append("success_clean" if not t["failed_steps"] else "success_after_tool_error")
    if t["budget_denied_calls"]: L.append("hit_tool_call_cap")
    return L

def stat(sample):
    ratios = {m: [] for m in MODES}; after = {m: [] for m in MODES}
    for c in CFG:
        runs = [T[c][i] for i in sample]
        ms = st.median(t["cost_usd"] for t in runs if t["success"])
        for t in runs:
            for l in labels(t):
                if l in ratios:
                    ratios[l].append(t["cost_usd"] / ms)
                    if not t["success"] and t["grade_error"] and "@turn" in t["grade_error"]:
                        d = int(t["grade_error"].split("@turn")[1]); p = t["per_turn_cost_usd"]
                        after[l].append(sum(p[d + 1:]) / sum(p))
    return ({m: (st.median(v) if v else float("nan")) for m, v in ratios.items()},
            {m: (st.median(v) if v else float("nan")) for m, v in after.items()})

pt = stat(ids)
random.seed(42); res = [stat([random.choice(ids) for _ in ids]) for _ in range(10000)]
def ci(k, m):
    v = sorted(r[k][m] for r in res if not math.isnan(r[k][m]))
    return [round(v[int(0.025 * len(v))], 2), round(v[int(0.975 * len(v)) - 1], 2)]
out = {"resamples": 10000, "n_tasks": len(ids), "modes": {m: {
    "ratio": {"point": round(pt[0][m], 2), "ci95": ci(0, m)},
    "after_flag_median": ({"point": round(pt[1][m], 2), "ci95": ci(1, m)} if not math.isnan(pt[1][m]) else None)} for m in MODES}}
json.dump(out, open(OUT / "bootstrap_modes.json", "w"), indent=1); print(json.dumps(out, indent=1))
