"""Task-level bootstrap (seed 42, 10,000 resamples) for Workload 1:
after-flag share of failed-run spend (pooled, five step-logged configs), alert precision and recall.
Resamples the 50 task ids with replacement, shared across configurations (cluster bootstrap);
the p90 threshold is recomputed in every resample (in-sample alert, as in the point estimate).
python3 -I bootstrap_bfcl.py"""
import json, math, random
from pathlib import Path
LOGS = Path(__file__).resolve().parents[4] / "logs"  # repository logs/
CFG = ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"]
OUT = Path(__file__).resolve().parent / "out"
T = {c: {t["task_id"]: t for t in map(json.loads, open(LOGS / f"tasks_{c}.jsonl"))} for c in CFG}
ids = sorted(set.intersection(*[set(T[c]) for c in CFG]))

def pct(xs, q):
    s = sorted(xs); return s[max(1, math.ceil(q * len(s))) - 1]

GROUPS = {"large": ["frontier", "frontier_cache"], "small": ["small", "small_cache_budget", "small_low"]}

def after_flag(sample, cfgs):
    num = den = 0.0
    for c in cfgs:
        for t in (T[c][i] for i in sample):
            if not t["success"] and t["grade_error"] and "@turn" in t["grade_error"]:
                d = int(t["grade_error"].split("@turn")[1]); p = t["per_turn_cost_usd"]
                num += sum(p[d + 1:]); den += sum(p)
    return num / den if den else float("nan")

def stat(sample):
    num = den = 0.0; fl = ff = nf = 0
    for c in CFG:
        runs = [T[c][i] for i in sample]
        for t in runs:
            if not t["success"]:
                nf += 1
                if t["grade_error"] and "@turn" in t["grade_error"]:
                    d = int(t["grade_error"].split("@turn")[1]); p = t["per_turn_cost_usd"]
                    num += sum(p[d + 1:]); den += sum(p)
        thr = pct([t["cost_usd"] for t in runs if t["success"]], 0.9)
        f = [t for t in runs if t["cost_usd"] > thr]
        fl += len(f); ff += sum(1 for t in f if not t["success"])
    return num / den, (ff / fl if fl else float("nan")), (ff / nf if nf else float("nan")), ff / fl - nf / (len(sample) * len(CFG)) if fl else float("nan"), after_flag(sample, GROUPS["large"]), after_flag(sample, GROUPS["small"])

pt = stat(ids)
random.seed(42)
B = 10000; res = []
for _ in range(B):
    res.append(stat([random.choice(ids) for _ in ids]))
def ci(k):
    v = sorted(r[k] for r in res if not math.isnan(r[k]))
    return round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v)) - 1], 3), len(v)
out = {"n_tasks": len(ids), "resamples": B,
       "after_flag_share": {"point": round(pt[0], 3), "ci95": ci(0)},
       "alert_precision": {"point": round(pt[1], 3), "ci95": ci(1)},
       "alert_recall": {"point": round(pt[2], 3), "ci95": ci(2)},
       "precision_minus_base_rate": {"point": round(pt[3], 3), "ci95": ci(3)},
       "after_flag_share_large_two_step_logged_configs": {"point": round(pt[4], 3), "ci95": ci(4)},
       "after_flag_share_small_three_configs": {"point": round(pt[5], 3), "ci95": ci(5)}}
json.dump(out, open(OUT / "bootstrap_bfcl.json", "w"), indent=1)
print(json.dumps(out, indent=1))
