"""Time-ordered held-out alert check for Litos application packages, plus bootstrap intervals for the
Table III median and p90 of successful runs (Litos packages, seed 2027; Workload 1 task ids, seed 42).
Packages are ordered by the time of their litos_result_started event. The first half (by count) sets the
nearest-rank p90 of accepted-package cost; packages of the second half above it are flagged.
Reads only recorded files; python3 -I litos_holdout.py -> out/litos_holdout.json"""
import json, math, random, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
import os, sys
EVENTS = Path(os.environ.get("LITOS_EVENTS", "ops-events-2026-10-23_27.ndjson"))  # private raw input
if not EVENTS.exists():
    sys.exit(f"private input {EVENTS} not found: the Litos event stream is not published; set LITOS_EVENTS to its path, "
             "or start from the published outputs in out/ (see papers/2026-11/README.md)")
LOGS = Path(__file__).resolve().parents[4] / "logs"  # repository logs/
OUT = Path(__file__).resolve().parent / "out"
PR = {"claude-sonnet-5": (2.0, 2.5, 0.20, 10.0), "claude-haiku-4-5-20251001": (1.0, 1.25, 0.10, 5.0)}
SEARCH = 10.0 / 1000

def pct(xs, q):
    s = sorted(xs); return s[max(1, math.ceil(q * len(s))) - 1]
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
kind, outcome, start = {}, {}, {}
for r in rows:
    if r["event"] == "litos_result_started":
        kind[r["operationId"]] = r["resultKind"]; start[r["operationId"]] = r["occurredAt"]
    if r["event"] == "litos_result_completed":
        outcome[r["operationId"]] = r["outcome"]
usd = defaultdict(float); calls = Counter(); retry = defaultdict(float)
for r in rows:
    if r["event"] != "anthropic_response_usage": continue
    t = tok(r)
    if t is None: continue
    c = t + (r.get("webSearchRequests") or 0) * SEARCH
    o = r.get("operationId") or "unattributed"
    usd[o] += c; calls[o] += 1
    if r.get("workload") == "bullet_litos_bullet_retry": retry[o] += c
pk = sorted([o for o, k in kind.items() if k == "application_packet"], key=lambda o: start[o])
half = len(pk) // 2
tr, te = pk[:half], pk[half:]
acc = lambda L: [o for o in L if outcome.get(o) == "accepted"]
thr = pct([usd[o] for o in acc(tr)], 0.9)
fl = [o for o in te if usd[o] > thr]
res = {"packages": len(pk), "train_packages": len(tr), "test_packages": len(te),
       "train_first_start": start[tr[0]], "train_last_start": start[tr[-1]], "test_first_start": start[te[0]], "test_last_start": start[te[-1]],
       "train_accepted": len(acc(tr)), "p90_threshold_usd": round(thr, 4),
       "test_accepted": len(acc(te)), "test_not_accepted": len(te) - len(acc(te)),
       "flagged": len(fl), "flagged_not_accepted": sum(1 for o in fl if outcome.get(o) != "accepted"),
       "flagged_with_retry": sum(1 for o in fl if retry[o] > 0),
       "test_accepted_with_retry": sum(1 for o in acc(te) if retry[o] > 0),
       "test_failures_caught_share": None}
nf = res["test_not_accepted"]
res["test_failures_caught_share"] = (res["flagged_not_accepted"] / nf) if nf else None
res["test_failures_with_zero_calls"] = sum(1 for o in te if outcome.get(o) != "accepted" and calls[o] == 0)

# Table III intervals
def boot(vals, seed, n=10000):
    rnd = random.Random(seed); m, p = [], []
    for _ in range(n):
        s = [vals[rnd.randrange(len(vals))] for _ in range(len(vals))]
        m.append(st.median(s)); p.append(pct(s, 0.9))
    q = lambda v, a: sorted(v)[min(len(v) - 1, int(a * len(v)))]
    return {"n": len(vals), "median": round(st.median(vals), 4), "median_lo": round(q(m, .025), 4), "median_hi": round(q(m, .975), 4),
            "p90": round(pct(vals, .9), 4), "p90_lo": round(q(p, .025), 4), "p90_hi": round(q(p, .975), 4)}
t3 = {"litos_package": boot([usd[o] for o in acc(pk)], 2027)}
for c in ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"]:
    T = [json.loads(l) for l in open(LOGS / f"tasks_{c}.jsonl")]
    t3[c] = boot([t["cost_usd"] for t in T if t["success"]], 42)
res["table_iii_intervals"] = t3
json.dump(res, open(OUT / "litos_holdout.json", "w"), indent=1); print(json.dumps(res, indent=1))
