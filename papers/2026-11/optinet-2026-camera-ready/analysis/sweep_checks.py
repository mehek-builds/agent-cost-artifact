#!/usr/bin/env python3
"""Cached sweep over every budget fraction: first budget where Retry + reserve reaches 43 tasks, and the largest
mean gap of Retry + reserve over Paced + spend-down (500-order means).

    python3 analysis/sweep_checks.py > analysis/sweep_checks_stdout.txt
"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired_and_retry as pr
fs = pr.fs
rows, full = pr.sweep("cached", ["retry-paced", "hybrid"], fs.FRACS)
first = None
best = None
for f in fs.FRACS:
    a = rows[f"{f}|retry-paced"]["succ_mean"]
    b = rows[f"{f}|hybrid"]["succ_mean"]
    print(f"{f:.0%}: retry+reserve {a:.2f} paced+spend-down {b:.2f} gap {a - b:+.2f}")
    if first is None and a >= 43 - 1e-9:
        first = f
    if best is None or a - b > best[1]:
        best = (f, a - b)
print(f"first budget where Retry + reserve reaches 43: {first:.0%}; budgets run {len(fs.FRACS)}")
print(f"largest mean gap Retry + reserve minus Paced + spend-down: {best[1]:+.2f} at {best[0]:.0%}")
print("leads at every budget from 20% to 110%:", all(rows[f"{f}|retry-paced"]["succ_mean"] > rows[f"{f}|hybrid"]["succ_mean"] for f in fs.FRACS if 0.2 <= f <= 1.1))

# Paired task-bootstrap, Paced + spend-down minus Retry + reserve (same protocol and seeds as paired_and_retry.bootstrap),
# at every cached budget from 20% to 110%.
small, large, ids, est, rate = pr.setup("cached")
full_need = sum(fs.task_cost(large[i]) for i in ids)
all_touch = True
for f in [x for x in fs.FRACS if 0.2 <= x <= 1.1]:
    rng = random.Random(fs.SEED)
    d = []
    for _ in range(fs.BOOT):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        seed = rng.randrange(1 << 30)
        a = pr.simulate("hybrid", sample, small, large, f * full_need, est, random.Random(seed), rate)["succ"]
        b = pr.simulate("retry-paced", sample, small, large, f * full_need, est, random.Random(seed), rate)["succ"]
        d.append(a - b)
    lo, hi = fs.pct(d, 0.025), fs.pct(d, 0.975)
    touch = lo <= 0 <= hi or hi >= 0
    all_touch = all_touch and hi >= 0
    print(f"paired {f:.0%}: Paced + SD minus Retry + reserve mean {sum(d)/len(d):+.2f} 95% [{lo}, {hi}] resamples with Paced + SD ahead {sum(x > 0 for x in d)} of {len(d)}; interval upper end >= 0: {hi >= 0}")
print("Retry + reserve never significantly ahead (every interval upper end >= 0, 20% to 110%):", all_touch)
