#!/usr/bin/env python3
"""Task-bootstrap intervals for Table III (re-priced pairs) and the single-turn workload, at a fifth of the budget.

Reads the recorded traces through the artifact's own code (optinet/extras.py helpers, BudgetPacer replay). No API calls.
Protocol as in paired_and_retry.py: 2,000 resamples of the tasks with replacement, one random arrival order per
resample, both policies on the same resampled tasks and order, seed 42; 2.5th and 97.5th percentiles.
Single-turn: each resample also draws one small and one large recorded rollout at random.

    python3 -I analysis/table3_intervals.py > analysis/table3_intervals_stdout.txt
"""
import json, random, statistics, sys
from pathlib import Path
ART = Path(__file__).resolve().parents[4]  # repository root
sys.path.insert(0, str(ART / "optinet")); sys.path.insert(0, str(ART))
import extras as E  # noqa: E402
F = E.F
BOOT, SEED, FRAC = 2000, 42, 0.2
EXTRA = (0.4, 0.8, 1.0)  # extra budgets for the Large comparison (printed after the 20% lines)


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


small, large, ids, _, _ = F.setup("small_cache_budget", "frontier_cache")
for lc, sc in E.PAIRS:
    cost = E.pricer(lc, sc)
    est, full, full_small = E.loo(small, large, ids, cost)
    rng, d = random.Random(SEED), []
    for _ in range(BOOT):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        B = FRAC * full
        p = E.run_batch(sample, small, large, B, est, cost)["succ"]
        l = E.large_only(sample, large, B, cost)
        d.append(p - l)
    print(f"{lc} / {sc} (small batch cost {full_small / full:.0%}) at 20%: paced+SD minus Large paired mean {statistics.mean(d):+.1f} "
          f"95% [{pct(d, 0.025)}, {pct(d, 0.975)}]; resamples with difference > 0: {sum(x > 0 for x in d)} of {BOOT}")
    for fr in EXTRA:
        rng, d = random.Random(SEED), []
        for _ in range(BOOT):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            B = fr * full
            p = E.run_batch(sample, small, large, B, est, cost)["succ"]
            l = E.large_only(sample, large, B, cost)
            d.append(p - l)
        print(f"{lc} / {sc} at {fr:.0%}: paced+SD minus Large paired mean {statistics.mean(d):+.1f} "
              f"95% [{pct(d, 0.025)}, {pct(d, 0.975)}]")

# single turn
dd = ART / "logs" / "single_turn"
load = lambda p: {t["task_id"]: t for t in map(json.loads, open(p))}  # noqa: E731
smalls = [load(p) for p in sorted(dd.glob("tasks_small*.jsonl"))]
larges = [load(p) for p in sorted(dd.glob("tasks_frontier*.jsonl"))]
sid = sorted(set.intersection(*(set(r) for r in smalls + larges)))
rng, d, sm = random.Random(SEED), [], []
for _ in range(BOOT):
    S, L = smalls[rng.randrange(len(smalls))], larges[rng.randrange(len(larges))]
    sample = [sid[rng.randrange(len(sid))] for _ in sid]
    cs = {i: S[i]["cost_usd"] for i in sid}; cl = {i: L[i]["cost_usd"] for i in sid}
    n, ts, tl = len(sid), sum(cs.values()), sum(cl.values())
    order = list(sample); rng.shuffle(order)
    pool, succ = FRAC * tl, 0
    for pos, tid in enumerate(order):
        if pool <= 0:
            continue
        k = n - pos - 1
        el = (tl - cl[tid]) / (n - 1)
        t = (L if pool >= (k + 1) * el else S)[tid]
        pool -= t["cost_usd"]; succ += int(t["success"])
    small_succ = sum(int(S[t]["success"]) for t in order)
    d.append(succ - small_succ); sm.append(small_succ)
print(f"single-turn, {len(sid)} tasks, 20%: paced+SD minus Small paired mean {statistics.mean(d):+.1f} "
      f"95% [{pct(d, 0.025)}, {pct(d, 0.975)}]; Small mean {statistics.mean(sm):.1f}")
