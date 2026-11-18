#!/usr/bin/env python3
"""Paired task-bootstrap interval for Paced + spend-down minus Small at a 5% pool on the uncached pairing
(the paper's one negative result). Same protocol as paired_and_retry.bootstrap (2,000 resamples, same seeds),
run at fraction 0.05 only. Also prints the 500-order means as a check against revision_numbers_stdout.txt.

    python3 analysis/tight_pool_paired.py > analysis/tight_pool_paired_stdout.txt
"""
import random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired_and_retry as pr
fs = pr.fs

for pair in ("uncached",):
    for f in (0.05, 0.08, 0.10):
        rows, full = pr.sweep(pair, ["small", "hybrid"], [f])
        print(f"{pair} {f:.0%}: 500-order means hybrid {rows[f'{f}|hybrid']['succ_mean']:.2f} small {rows[f'{f}|small']['succ_mean']:.2f}")
        small, large, ids, est, rate = pr.setup(pair)
        rng, d = random.Random(fs.SEED), []
        for _ in range(fs.BOOT):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            seed = rng.randrange(1 << 30)
            h = pr.simulate("hybrid", sample, small, large, f * full, est, random.Random(seed), rate)["succ"]
            s = pr.simulate("small", sample, small, large, f * full, est, random.Random(seed), rate)["succ"]
            d.append(h - s)
        print(f"  paired hybrid - small at {f:.0%}: mean {statistics.mean(d):+.2f} 95% [{fs.pct(d,0.025)}, {fs.pct(d,0.975)}] share<=0 {sum(x<=0 for x in d)/len(d):.3f}; resamples with difference > 0: {sum(x>0 for x in d)} of {len(d)}")
