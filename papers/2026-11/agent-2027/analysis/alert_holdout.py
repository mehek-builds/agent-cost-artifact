"""Out-of-sample alert check for Workload 1 (five step-logged configurations).
Each of 10,000 random splits (seed 42) puts 25 of the 50 task ids in a training half and 25 in a held-out half
(same split for every configuration). Per configuration, the alert threshold is the nearest-rank p90 of
successful-run cost on the training half; runs of the held-out half above it are flagged. Precision and recall
are pooled over the five configurations on the held-out half. The interval is the 2.5th to 97.5th percentile
of the 10,000 splits (split-to-split spread), not a sampling interval.
python3 -I alert_holdout.py -> out/alert_holdout.json"""
import json, math, random
import os
from pathlib import Path
LOGS = Path(os.environ.get("AGENT_ARTIFACT_ROOT") or Path(__file__).resolve().parents[4]) / "logs"
CFG = ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"]
OUT = Path(__file__).resolve().parent / "out"
T = {c: {t["task_id"]: t for t in map(json.loads, open(LOGS / f"tasks_{c}.jsonl"))} for c in CFG}
ids = sorted(set.intersection(*[set(T[c]) for c in CFG]))

def pct(xs, q):
    s = sorted(xs); return s[max(1, math.ceil(q * len(s))) - 1]

random.seed(42)
P, R, N, BASE = [], [], [], []
for _ in range(10000):
    tr = set(random.sample(ids, len(ids) // 2)); te = [i for i in ids if i not in tr]
    fl = ff = nf = nr = 0
    for c in CFG:
        thr = pct([T[c][i]["cost_usd"] for i in tr if T[c][i]["success"]], 0.9)
        for i in te:
            t = T[c][i]; nr += 1; nf += (not t["success"])
            if t["cost_usd"] > thr:
                fl += 1; ff += (not t["success"])
    if fl:
        P.append(ff / fl)
    R.append(ff / nf); N.append(fl / nr); BASE.append(nf / nr)
def q(v, a):
    s = sorted(v); return s[min(len(s) - 1, int(a * len(s)))]
out = {"splits": 10000, "train_tasks": len(ids) // 2, "test_tasks": len(ids) - len(ids) // 2,
       "precision": {"median": round(q(P, .5), 3), "mean": round(sum(P) / len(P), 3), "p2.5": round(q(P, .025), 3), "p97.5": round(q(P, .975), 3), "n_defined": len(P)},
       "recall": {"median": round(q(R, .5), 3), "mean": round(sum(R) / len(R), 3), "p2.5": round(q(R, .025), 3), "p97.5": round(q(R, .975), 3)},
       "flag_rate_mean": round(sum(N) / len(N), 3), "base_failure_rate_mean": round(sum(BASE) / len(BASE), 3)}
json.dump(out, open(OUT / "alert_holdout.json", "w"), indent=1); print(json.dumps(out, indent=1))
