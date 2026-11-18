"""Task-level bootstrap (seed 42, 10,000 resamples) of the carried-context share of uncached
spend (proportional split and billed-rate split) for the five step-logged Workload 1 configs.
Resamples the 50 task ids with replacement, shared across configurations (as bootstrap_bfcl.py).
python3 -I bootstrap_carried.py -> out/bootstrap_carried.json"""
import json, math, random
from collections import defaultdict
from pathlib import Path
LOGS = Path(__file__).resolve().parents[4] / "logs"  # repository logs/
PRICES = json.load(open(LOGS.parent / "prices.json"))["models"]
CFG = ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"]
OUT = Path(__file__).resolve().parent / "out"

def rates(model, P):
    p = PRICES[model]
    if "tier_low" in p:
        p = p["tier_low"] if P <= 100_000 else p["tier_high"]
    return p

def per_task(cfg):
    S = [json.loads(l) for l in open(LOGS / f"steps_{cfg}.jsonl")]
    by = defaultdict(list)
    for s in S: by[s["task_id"]].append(s)
    res = {}
    for tid, steps in by.items():
        steps.sort(key=lambda s: s["step"])
        tot = rp = rb = 0.0; prevP = None
        for s in steps:
            u = s["usage"]; R, I, W = u["cache_read_tokens"], u["input_tokens"], u["cache_write_tokens"]
            P = I + R + W; r = rates(s["model"], P)
            parts = {"in": I * r["input_per_mtok"] / 1e6, "cw": W * r["cache_write_5min_per_mtok"] / 1e6,
                     "cr": R * r["cache_read_per_mtok"] / 1e6, "out": u["output_tokens"] * r["output_per_mtok"] / 1e6}
            in_usd = parts["in"] + parts["cw"] + parts["cr"]
            tot += sum(parts.values())
            if prevP is not None:
                carried = min(prevP, P)
                rp += in_usd * carried / P if P else 0
                cr_ = min(R, carried); rem = carried - cr_
                ci_ = min(I, rem); rem -= ci_
                cw_ = min(W, rem)
                rb += (cr_ * parts["cr"] / R if R else 0) + (ci_ * parts["in"] / I if I else 0) + (cw_ * parts["cw"] / W if W else 0)
            prevP = P
        res[tid] = (tot, rp, rb)
    return res
D = {c: per_task(c) for c in CFG}
ids = sorted(set.intersection(*[set(D[c]) for c in CFG]))
def share(c, sample):
    t = sum(D[c][i][0] for i in sample)
    return sum(D[c][i][1] for i in sample) / t, sum(D[c][i][2] for i in sample) / t
out = {"n_tasks": len(ids), "resamples": 10000, "configs": {}}
random.seed(42)
samples = [[random.choice(ids) for _ in ids] for _ in range(10000)]
for c in CFG:
    pt = share(c, ids)
    rs = [share(c, s) for s in samples]
    def ci(k):
        v = sorted(r[k] for r in rs)
        return [round(100 * v[int(0.025 * len(v))], 1), round(100 * v[int(0.975 * len(v)) - 1], 1)]
    out["configs"][c] = {"proportional_pct": round(100 * pt[0], 1), "proportional_ci95": ci(0),
                         "billed_pct": round(100 * pt[1], 1), "billed_ci95": ci(1)}
json.dump(out, open(OUT / "bootstrap_carried.json", "w"), indent=1)
print(json.dumps(out, indent=1))
