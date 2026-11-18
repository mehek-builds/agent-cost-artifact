"""Bootstrap intervals for the Workload 1 errored-plus-repair share and top-10% task share (five step-logged
configurations, 50 task ids resampled jointly, seed 42), the Litos top-10% package share (packages, seed 2027),
and the Table III median and p90 intervals for all six configurations (successful runs, seed 42 per config; Litos
packages seed 2027). 10,000 resamples, percentile 95% intervals.
python3 -I bootstrap_shares.py -> out/bootstrap_shares.json  (the Litos part needs the private event stream)"""
import json, math, random, statistics as st
from collections import defaultdict
from pathlib import Path
import os, sys
EVENTS = Path(os.environ.get("LITOS_EVENTS", "ops-events-2026-10-23_27.ndjson"))  # private raw input
if not EVENTS.exists():
    sys.exit(f"private input {EVENTS} not found: the Litos event stream is not published; set LITOS_EVENTS to its path, "
             "or start from the published outputs in out/ (see papers/2026-11/README.md)")
LOGS = Path(__file__).resolve().parents[4] / "logs"  # repository logs/
OUT = Path(__file__).resolve().parent / "out"
CFG = ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"]
ALL = ["frontier", "frontier_cache", "frontier_low", "small", "small_cache_budget", "small_low"]
B = 10000
def pct(xs, q):
    s = sorted(xs); return s[max(1, math.ceil(q * len(s))) - 1]
def q(v, a): return sorted(v)[min(len(v) - 1, int(a * len(v)))]
def ci(v, nd=1): return [round(100 * q(v, .025), nd), round(100 * q(v, .975), nd)]

# ---- Workload 1 per-task spend and errored+repair spend
D = {}
for c in CFG:
    S = [json.loads(l) for l in open(LOGS / f"steps_{c}.jsonl")]
    by = defaultdict(list)
    for s in S: by[s["task_id"]].append(s)
    d = {}
    for tid, steps in by.items():
        steps.sort(key=lambda s: s["step"]); tot = er = 0.0; prev = False
        for s in steps:
            tot += s["cost_usd"]
            if s["kind"] == "failed" or prev: er += s["cost_usd"]
            prev = s["kind"] == "failed"
        d[tid] = (tot, er)
    D[c] = d
ids = sorted(set.intersection(*[set(D[c]) for c in CFG]))
def top(costs): 
    s = sorted(costs, reverse=True); k = max(1, len(s) // 10); return sum(s[:k]) / sum(s)
res = {"resamples": B, "n_tasks": len(ids), "w1": {}}
random.seed(42)
samples = [[random.choice(ids) for _ in ids] for _ in range(B)]
for c in CFG:
    pt_er = sum(D[c][i][1] for i in ids) / sum(D[c][i][0] for i in ids)
    pt_top = top([D[c][i][0] for i in ids])
    er = [sum(D[c][i][1] for i in s) / sum(D[c][i][0] for i in s) for s in samples]
    tp = [top([D[c][i][0] for i in s]) for s in samples]
    res["w1"][c] = {"err_repair_pct": round(100 * pt_er, 1), "err_repair_ci95": ci(er),
                    "top10_pct": round(100 * pt_top, 1), "top10_ci95": ci(tp)}
res["w1_envelope"] = {"err_repair_lo": min(v["err_repair_ci95"][0] for v in res["w1"].values()),
                      "err_repair_hi": max(v["err_repair_ci95"][1] for v in res["w1"].values()),
                      "top10_lo": min(v["top10_ci95"][0] for v in res["w1"].values()),
                      "top10_hi": max(v["top10_ci95"][1] for v in res["w1"].values())}

# ---- Table III intervals, all six configurations (task-level, seed 42 each)
def boot(vals, seed):
    rnd = random.Random(seed); m, p = [], []
    for _ in range(B):
        s = [vals[rnd.randrange(len(vals))] for _ in range(len(vals))]
        m.append(st.median(s)); p.append(pct(s, 0.9))
    return {"n": len(vals), "median": round(st.median(vals), 4), "median_lo": round(q(m, .025), 4), "median_hi": round(q(m, .975), 4),
            "p90": round(pct(vals, .9), 4), "p90_lo": round(q(p, .025), 4), "p90_hi": round(q(p, .975), 4)}
t3 = {}
for c in ALL:
    T = [json.loads(l) for l in open(LOGS / f"tasks_{c}.jsonl")]
    t3[c] = boot([t["cost_usd"] for t in T if t["success"]], 42)

# ---- Litos package concentration (packages, seed 2027), same pricing as litos_holdout.py
PR = {"claude-sonnet-5": (2.0, 2.5, 0.20, 10.0), "claude-haiku-4-5-20251001": (1.0, 1.25, 0.10, 5.0)}
SEARCH = 10.0 / 1000
def tok(r):
    p = PR.get(r.get("model"))
    if not p: return None
    return (r.get("inputTokens", 0) * p[0] + (r.get("cacheWrite5mTokens", 0) or 0) * p[1]
            + (r.get("cacheReadInputTokens", 0) or 0) * p[2] + r.get("outputTokens", 0) * p[3]) / 1e6
rows, seen = [], set()
for line in open(EVENTS):
    r = json.loads(line); k = r.get("eventId") or json.dumps(r, sort_keys=True)
    if (r.get("occurredAt") or r["timestamp"])[:16] < "2026-10-23T14:58": continue  # window: from operation roll-out onward, as in prod.py
    if k in seen: continue
    seen.add(k); rows.append(r)
kind = {r["operationId"]: r["resultKind"] for r in rows if r["event"] == "litos_result_started"}
outcome = {r["operationId"]: r["outcome"] for r in rows if r["event"] == "litos_result_completed"}
usd = defaultdict(float)
for r in rows:
    if r["event"] != "anthropic_response_usage": continue
    t = tok(r)
    if t is None: continue
    usd[r.get("operationId") or "unattributed"] += t + (r.get("webSearchRequests") or 0) * SEARCH
pk = [usd[o] for o, k in kind.items() if k == "application_packet"]
rnd = random.Random(2027)
tp = [top([pk[rnd.randrange(len(pk))] for _ in pk]) for _ in range(B)]
res["litos_top10"] = {"packages": len(pk), "top10_pct": round(100 * top(pk), 1), "ci95": ci(tp)}
# resume-parse accepted operations, same procedure as the package row (operations resampled, seed 2027)
rp = [usd[o] for o, k in kind.items() if k == "resume_parse" and outcome.get(o) == "accepted"]
t3["litos_resume_parse"] = boot(rp, 2027)
json.dump({**res, "table_iii_all": t3}, open(OUT / "bootstrap_shares.json", "w"), indent=1)
print(json.dumps({**res, "table_iii_all": t3}, indent=1))
