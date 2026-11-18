"""Recompute the Litos application-package rows of Tables III to V and the time-ordered hold-out from the released
per-package aggregate only (no private stream). Point estimates and the like-for-like interval (seed 2027).
python3 -I recompute_from_aggregate.py -> out/recompute_from_aggregate_stdout.txt"""
import csv, math, random, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
P = list(csv.DictReader(open(HERE / "out" / "litos_package_aggregate.csv")))
N = ["research", "rewrite", "retry", "judge", "spec", "other"]
for r in P:
    r["usd"] = float(r["total_usd"]); r["acc"] = r["outcome"] == "accepted"
    r["calls"] = int(r["calls_total"])
    for n in N:
        r["u_" + n] = float(r["usd_" + n]); r["c_" + n] = int(r["calls_" + n])
def pct(xs, q):
    s = sorted(xs); return s[max(1, math.ceil(q * len(s))) - 1]
tot = sum(r["usd"] for r in P); acc = [r for r in P if r["acc"]]
print("packages", len(P), "accepted", len(acc), "spend", round(tot, 2))
print("accepted median", round(st.median(r["usd"] for r in acc), 3), "p90", round(pct([r["usd"] for r in acc], .9), 3),
      "fail share %", round(100 * sum(r["usd"] for r in P if not r["acc"]) / tot, 1))
print("cost per accepted package", round(tot / len(acc), 3), " top-10% share %", round(100 * sum(sorted((r["usd"] for r in P), reverse=True)[:len(P) // 10]) / tot, 1))
allc = sum(r["calls"] for r in P)
for n in N:
    used = [r for r in acc if r["c_" + n]]
    print(n, "calls %", round(100 * sum(r["c_" + n] for r in P) / allc, 1), "used by", len(used),
          "spend %", round(100 * sum(r["u_" + n] for r in P) / tot, 1),
          "median usd if used", round(st.median(r["u_" + n] for r in used), 3) if used else None)
loop = sum(r["u_rewrite"] + r["u_judge"] + r["u_retry"] for r in P) / tot
print("rewrite+judge+retry share %", round(100 * loop, 1), " searches", sum(int(r["web_searches"]) for r in P))
ret = [r for r in acc if r["c_retry"]]; nor = [r for r in acc if not r["c_retry"] and r["calls"]]
print("accepted with retry", len(ret), "no retry with calls", len(nor), "zero-call accepted", sum(1 for r in acc if not r["calls"]))
print("retry share of accepted-with-retry spend %", round(100 * sum(r["u_retry"] for r in ret) / sum(r["usd"] for r in ret), 1))
mr, mn = st.median(r["usd"] for r in ret), st.median(r["usd"] for r in nor)
print("medians retry/no-retry", round(mr, 3), round(mn, 3), "ratio", round(mr / mn, 2))
nj = [r for r in nor if r["c_judge"]]
random.seed(2027)
xa = [r["usd"] for r in ret if r["c_judge"]]; xb = [r["usd"] for r in nj]
rs = sorted(st.median(random.choices(xa, k=len(xa))) / st.median(random.choices(xb, k=len(xb))) for _ in range(10000))
print("judge-loop no-retry n", len(nj), "median", round(st.median(xb), 3), "ratio", round(st.median(xa) / st.median(xb), 2), "ci", round(rs[250], 2), round(rs[9749], 2))
# in-sample and time-ordered alert
thr = pct([r["usd"] for r in acc], .9); fl = [r for r in P if r["usd"] > thr]
print("in-sample alert thr", round(thr, 3), "flagged", len(fl), "not accepted", sum(1 for r in fl if not r["acc"]),
      "with retry", sum(1 for r in fl if r["c_retry"]), "with research", sum(1 for r in fl if r["c_research"]),
      "accepted with research", sum(1 for r in acc if r["c_research"]))
P.sort(key=lambda r: int(r["package_rank_by_start"])); h = len(P) // 2; tr, te = P[:h], P[h:]
t = pct([r["usd"] for r in tr if r["acc"]], .9); f = [r for r in te if r["usd"] > t]
print("hold-out: first half accepted", sum(r["acc"] for r in tr), "thr", round(t, 4), "flagged", len(f), "not accepted", sum(1 for r in f if not r["acc"]),
      "with retry", sum(1 for r in f if r["c_retry"]), "accepted-with-retry in later half", sum(1 for r in te if r["acc"] and r["c_retry"]),
      "of accepted", sum(r["acc"] for r in te), "later-half failures with zero calls", sum(1 for r in te if not r["acc"] and not r["calls"]))
