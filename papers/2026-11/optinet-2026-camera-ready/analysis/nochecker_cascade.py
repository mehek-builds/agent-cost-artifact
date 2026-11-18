#!/usr/bin/env python3
"""No-checker small-then-retry-large cascade (revision round 1, C-EVIDENCE).

Same recorded runs, pool, 500 orders and bootstrap as paired_and_retry.py. Difference: the
retry signal is not the benchmark checker. A finished small run is retried on the large model
iff any of its steps had a tool error (a signal the logs already contain), whether or not the
task would have passed. Both attempts are charged to the pool. "retry-err" retries whenever
the pool is positive; "retry-err-paced" retries only if the reserve allows.

    python3 analysis/nochecker_cascade.py > analysis/nochecker_cascade_stdout.txt
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired_and_retry as pr  # noqa: E402
fs = pr.fs

NEW = ("retry-err", "retry-err-paced")


def simulate(policy, order, small, large, B, est, rng=None, rate=0.38):
    if policy not in NEW:
        return pr.simulate_orig(policy, order, small, large, B, est, rng, rate)
    pool, succ = B, 0
    fs.TOK["small"] = fs.TOK["large"] = 0
    if isinstance(est, fs.OnlineEst):
        est = est.fresh()
    reasons = {"pool": 0, "cap": 0, "model": 0, "unstarted": 0}
    esc, n = 0, len(order)
    for pos, tid in enumerate(order):
        k_after = n - pos - 1
        if pool <= 0:
            reasons["unstarted"] += 1
            continue
        S, L = small[tid], large[tid]
        e_s, e_l = est[tid]
        sp, _, done, why = fs.run_steps(S["steps"], pool)
        ok, fm, att = done and S["success"], "small", None
        had_error = any(s["n_errors"] > 0 for s in S["steps"])
        if done and had_error:
            room = pool - sp - k_after * e_s
            go = (pool - sp > 0) if policy == "retry-err" else (room >= e_l)
            if go:
                esc += 1
                cap = None if policy == "retry-err" else max(room, 0.0)
                sp2, _, done, why = fs.run_steps(L["steps"], pool - sp, cap)
                ok, fm, att, sp = done and L["success"], "large", sp2, sp + sp2
        if isinstance(est, fs.OnlineEst) and why is None:
            est.observe(fm, sp if att is None else att)
        pool -= sp
        succ += int(bool(ok))
        if not ok:
            reasons[why if why else "model"] += 1
    return {"succ": succ, "spent": B - pool, "reasons": reasons, "esc": esc}


pr.simulate_orig = pr.simulate
pr.simulate = simulate

FRACS = (0.2, 0.4, 0.8, 1.0)
pols = ["small", "hybrid", "retry", "retry-paced", "retry-err", "retry-err-paced"]
pairs = [("retry-err-paced", "retry-err"), ("retry-err-paced", "hybrid"), ("retry-err-paced", "small"),
         ("retry-paced", "retry-err-paced")]
for pair in ("cached",):
    small, large, ids, est, rate = pr.setup(pair)
    nerr = sum(any(s["n_errors"] > 0 for s in small[i]["steps"]) for i in ids)
    nerr_fail = sum(any(s["n_errors"] > 0 for s in small[i]["steps"]) and not small[i]["success"] for i in ids)
    nfail = sum(not small[i]["success"] for i in ids)
    print(f"[{pair}] small-run tasks with any tool error: {nerr} of {len(ids)}; "
          f"of the {nfail} small failures, {nerr_fail} had a tool error")
    rows, full = pr.sweep(pair, pols, FRACS)
    boot = pr.bootstrap(pair, pols, pairs)
    for f in FRACS:
        print(f"--- budget {f:.0%}")
        for p in pols:
            r = rows[f"{f}|{p}"]
            print(f"{p:>16} mean500 {r['succ_mean']:5.2f} boot95 {boot[str(f)]['ci95'][p]} "
                  f"spent ${r['spent_mean']:.4f} esc {r['esc_mean']:.2f} unstarted {r['unstarted_mean']:.2f}")
        for k, v in boot[str(f)]["diff"].items():
            print(f"  paired {k:>34}: mean {v['mean']:+.2f} 95% [{v['ci95'][0]}, {v['ci95'][1]}]")
