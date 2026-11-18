"""Comparison of the cost alert with a trivial non-cost alert on Workload 1 (five step-logged configurations).
Alerts, per run: (a) cost: spend above the configuration's successful-run p90 (in-sample, as in bootstrap_bfcl.py);
(b) error: the run contains at least one errored tool call (tasks_*.jsonl failed_steps > 0); (c) union: either.
Pooled precision (share of flagged runs that failed) and recall (share of failed runs flagged), with the same
task-level cluster bootstrap (seed 42, 10,000 resamples, threshold recomputed in every resample).
python3 -I bootstrap_error_alert.py -> out/bootstrap_error_alert.json"""
import json, math, random
from pathlib import Path
LOGS = Path(__file__).resolve().parents[4] / "logs"  # repository logs/
CFG = ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"]
OUT = Path(__file__).resolve().parent / "out"
T = {c: {t["task_id"]: t for t in map(json.loads, open(LOGS / f"tasks_{c}.jsonl"))} for c in CFG}
ids = sorted(set.intersection(*[set(T[c]) for c in CFG]))

def pct(xs, q):
    s = sorted(xs); return s[max(1, math.ceil(q * len(s))) - 1]

def stat(sample):
    cnt = {k: [0, 0] for k in ("cost", "error", "union")}  # flagged, flagged and failed
    nf = nr = 0
    for c in CFG:
        runs = [T[c][i] for i in sample]
        thr = pct([t["cost_usd"] for t in runs if t["success"]], 0.9)
        for t in runs:
            nr += 1; nf += (not t["success"])
            a = t["cost_usd"] > thr; b = t["failed_steps"] > 0
            for k, f in (("cost", a), ("error", b), ("union", a or b)):
                if f:
                    cnt[k][0] += 1; cnt[k][1] += (not t["success"])
    res = {}
    for k, (fl, ff) in cnt.items():
        res[k + "_flagged"] = fl
        res[k + "_precision"] = ff / fl if fl else float("nan")
        res[k + "_recall"] = ff / nf
    res["base_rate"] = nf / nr
    return res

pt = stat(ids)
random.seed(42)
B = 10000; res = [stat([random.choice(ids) for _ in ids]) for _ in range(B)]
def ci(k):
    v = sorted(r[k] for r in res if not math.isnan(r[k]))
    return [round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v)) - 1], 3)]
out = {"n_tasks": len(ids), "resamples": B, "base_failure_rate": round(pt["base_rate"], 3)}
for k in ("cost", "error", "union"):
    out[k] = {"flagged": pt[k + "_flagged"], "precision": round(pt[k + "_precision"], 3), "precision_ci95": ci(k + "_precision"),
              "recall": round(pt[k + "_recall"], 3), "recall_ci95": ci(k + "_recall")}
json.dump(out, open(OUT / "bootstrap_error_alert.json", "w"), indent=1)
print(json.dumps(out, indent=1))
