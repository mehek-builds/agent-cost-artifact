"""Package-level bootstrap intervals (10,000 resamples, seed 2027, percentile 95%) for the Litos values that Tables
IV (Panel 2) and V and the text print: role spend shares, the rewrite+judge+retry loop share, the retry share of
package spend, the retry share within retry packages, and the Table V ratios of the retry and no-retry groups to the
accepted median. Also task-level intervals (seed 42) for the Table III fail-share column of Workload 1.
python3 -I bootstrap_litos_shares.py -> out/bootstrap_litos_shares.json  (the Litos part needs the private event stream)"""
import io, json, random, statistics as st, contextlib, sys
from collections import defaultdict
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
with contextlib.redirect_stdout(io.StringIO()):
    import prod
B = 10000
def q(v, a): return sorted(v)[min(len(v) - 1, int(a * len(v)))]
def ci(v, nd=1, m=100): return [round(m * q(v, .025), nd), round(m * q(v, .975), nd)]
pk, acc = [o for o, k in prod.kind.items() if k == "application_packet"], prod.pk_acc
U, R, RC = prod.op_usd, prod.op_role_usd, prod.op_role_calls
roles = ["research (web search)", "rewrite", "retry", "judge", "spec", "other"]
rnd = random.Random(2027)
res = {"resamples": B, "packages": len(pk), "seed": 2027}
samples = [[pk[rnd.randrange(len(pk))] for _ in pk] for _ in range(B)]
def share(s, rs):
    t = sum(U[o] for o in s); return sum(R[o][r] for o in s for r in rs) / t
pts = {}
for ro in roles:
    pts[ro] = {"pct": round(100 * share(pk, [ro]), 1), "ci95": ci([share(s, [ro]) for s in samples])}
res["role_share_of_package_spend"] = pts
loop = ["rewrite", "judge", "retry"]
res["loop_share"] = {"pct": round(100 * share(pk, loop), 1), "ci95": ci([share(s, loop) for s in samples])}
# accepted-package ratios: resample accepted packages; group labels as in prod.py
nor = {o for o in acc if not RC[o]["retry"] and prod.op_calls[o]}
ret = {o for o in acc if RC[o]["retry"]}
rnd = random.Random(2027)
rr, rn, rw = [], [], []
for _ in range(B):
    s = [acc[rnd.randrange(len(acc))] for _ in acc]
    m = st.median(U[o] for o in s)
    a = [U[o] for o in s if o in ret]; b = [U[o] for o in s if o in nor]
    rr.append(st.median(a) / m); rn.append(st.median(b) / m)
    rw.append(sum(R[o]["retry"] for o in s if o in ret) / sum(U[o] for o in s if o in ret))
med = st.median(U[o] for o in acc)
res["table_v"] = {"retry_ratio": round(st.median(U[o] for o in ret) / med, 2), "retry_ratio_ci95": ci(rr, 2, 1),
                  "no_retry_ratio": round(st.median(U[o] for o in nor) / med, 2), "no_retry_ratio_ci95": ci(rn, 2, 1),
                  "retry_share_within_retry_packages_pct": round(100 * sum(R[o]["retry"] for o in ret) / sum(U[o] for o in ret), 1),
                  "retry_share_within_retry_packages_ci95": ci(rw)}
# Workload 1 fail share per configuration (task ids resampled, seed 42)
LOGS = Path(os.environ.get("AGENT_ARTIFACT_ROOT") or Path(__file__).resolve().parents[4]) / "logs"
ALL = ["frontier", "frontier_cache", "frontier_low", "small", "small_cache_budget", "small_low"]
fs = {}
for c in ALL:
    T = [json.loads(l) for l in open(LOGS / f"tasks_{c}.jsonl")]
    r = random.Random(42)
    pt = sum(t["cost_usd"] for t in T if not t["success"]) / sum(t["cost_usd"] for t in T)
    v = []
    for _ in range(B):
        s = [T[r.randrange(len(T))] for _ in T]
        v.append(sum(t["cost_usd"] for t in s if not t["success"]) / sum(t["cost_usd"] for t in s))
    fs[c] = {"fail_share_pct": round(100 * pt, 1), "ci95": ci(v)}
res["w1_fail_share"] = fs
json.dump(res, open(HERE / "out" / "bootstrap_litos_shares.json", "w"), indent=1)
print(json.dumps(res, indent=1))
