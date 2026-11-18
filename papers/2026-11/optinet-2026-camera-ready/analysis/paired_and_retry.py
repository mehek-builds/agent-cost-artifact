#!/usr/bin/env python3
"""Camera-ready analyses for OPTINET 2026 S38206 (reviewer fix points 2 and 3).

Reads the recorded BFCL v4 multi-turn traces in the public artifact
(this repository, logs/) through its own loaders and policy
simulator (optinet/fixed_spend.py). Nothing in the artifact is modified.

1. Paired differences against the Small baseline. Uses the artifact's bootstrap protocol
   exactly (2,000 resamples of the 50 tasks with replacement, one random arrival order per
   resample, seed 42) and first checks that it reproduces the recorded Table I brackets.
   The brackets are 2.5th and 97.5th percentiles of solved counts over resamples
   (task-bootstrap 95% percentile intervals), not ranges over orders.
2. A Small-then-retry-Large baseline: every task runs to its end on the small model; if
   the BFCL checker marks it failed, the task is rerun from its first turn on the large
   model (the large model's recorded run). Two variants: unpaced (retry whenever the pool
   is positive) and paced (retry only if the reserve allows, with the paced cap).
   The retry needs a task-level failure signal; the checker supplies it here, which a
   deployment would not have for free, so this baseline is optimistic.

    python3 analysis/paired_and_retry.py  > analysis/paired_and_retry_stdout.txt
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

ART = Path(__file__).resolve().parents[4]  # repository root
sys.path.insert(0, str(ART / "optinet"))
sys.path.insert(0, str(ART))
import fixed_spend as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
RETRY = ("retry", "retry-paced")


def simulate(policy, order, small, large, B, est, rng=None, rate=0.38):
    """fixed_spend.simulate plus the two retry baselines (same pool, stop and estimator rules)."""
    if policy not in RETRY:
        return fs.simulate(policy, order, small, large, B, est, rng, rate)
    pool, succ = B, 0
    fs.TOK["small"] = fs.TOK["large"] = 0
    if isinstance(est, fs.OnlineEst):
        est = est.fresh()
    reasons = {"pool": 0, "cap": 0, "model": 0, "unstarted": 0}
    esc, spent_small_failed, n = 0, 0.0, len(order)
    for pos, tid in enumerate(order):
        k_after = n - pos - 1
        if pool <= 0:
            reasons["unstarted"] += 1
            continue
        S, L = small[tid], large[tid]
        e_s, e_l = est[tid]
        sp, _, done, why = fs.run_steps(S["steps"], pool)
        ok, fm, att = done and S["success"], "small", None
        if done and not S["success"]:
            spent_small_failed += sp
            room = pool - sp - k_after * e_s
            go = (pool - sp > 0) if policy == "retry" else (room >= e_l)
            if go:
                esc += 1
                cap = None if policy == "retry" else max(room, 0.0)
                sp2, _, done, why = fs.run_steps(L["steps"], pool - sp, cap)
                ok, fm, att, sp = done and L["success"], "large", sp2, sp + sp2
        if isinstance(est, fs.OnlineEst) and why is None:  # final attempt only, as CostEstimator
            est.observe(fm, sp if att is None else att)
        pool -= sp
        succ += int(bool(ok))
        if not ok:
            reasons[why if why else "model"] += 1
    return {"succ": succ, "spent": B - pool, "reasons": reasons, "esc": esc,
            "small_spend_on_retried": spent_small_failed, "overdraw": max(0.0, -pool)}


def setup(pair):
    small, large, ids, loo, rate = fs.setup(*fs.PAIRS[pair])
    prior_runs = {"cached": ("small", "frontier"), "uncached": ("small_cache_budget", "frontier_cache")}
    return small, large, ids, fs.out_of_sample_prior(prior_runs[pair]), rate


def sweep(pair, policies, fracs):
    """Mean over 500 orders, same orders and seeds as fixed_spend.sweep."""
    small, large, ids, est, rate = setup(pair)
    full = sum(fs.task_cost(large[i]) for i in ids)
    rng = random.Random(fs.SEED)
    orders = [rng.sample(ids, len(ids)) for _ in range(fs.ORDERS)]
    rows = {}
    for f in fracs:
        for p in policies:
            prng = random.Random(fs.SEED + 1)
            sims = [simulate(p, o, small, large, f * full, est, prng, rate) for o in orders]
            s = [x["succ"] for x in sims]
            rows[f"{f}|{p}"] = {"succ_mean": statistics.mean(s), "p05": fs.pct(s, 0.05), "p95": fs.pct(s, 0.95),
                                "spent_mean": statistics.mean(x["spent"] for x in sims),
                                "esc_mean": statistics.mean(x["esc"] for x in sims),
                                "unstarted_mean": statistics.mean(x["reasons"]["unstarted"] for x in sims),
                                "per_order": s}
    return rows, full


def bootstrap(pair, policies, pairs_to_diff):
    """fixed_spend.bootstrap's protocol, for any policy set (policies do not touch the
    outer RNG, so the resamples and per-resample seeds are identical to the recorded run)."""
    small, large, ids, est, rate = setup(pair)
    full = sum(fs.task_cost(large[i]) for i in ids)
    out = {}
    for f in fs.TABLE_FRACS:
        rng, draws = random.Random(fs.SEED), {p: [] for p in policies}
        for _ in range(fs.BOOT):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            seed = rng.randrange(1 << 30)
            for p in policies:
                draws[p].append(simulate(p, sample, small, large, f * full, est, random.Random(seed), rate)["succ"])
        ci = {p: (fs.pct(v, 0.025), fs.pct(v, 0.975)) for p, v in draws.items()}
        diff = {}
        for a, b in pairs_to_diff:
            d = [x - y for x, y in zip(draws[a], draws[b])]
            diff[f"{a} - {b}"] = {"mean": statistics.mean(d), "ci95": (fs.pct(d, 0.025), fs.pct(d, 0.975)),
                                  "share_le0": sum(x <= 0 for x in d) / len(d)}
        out[str(f)] = {"ci95": ci, "diff": diff}
    return out


def main():
    recorded = json.loads((ART / "optinet/results/fixed_spend.json").read_text())
    pols = ["small", "large", "hybrid", "paced+cap", "admit", "retry", "retry-paced"]
    pairs = [("hybrid", "small"), ("paced+cap", "small"), ("retry", "small"), ("retry-paced", "small"),
             ("hybrid", "retry"), ("hybrid", "retry-paced"), ("admit", "small"), ("retry-paced", "retry")]
    res = {}
    for pair in ("cached", "uncached"):
        boot = bootstrap(pair, pols, pairs)
        rows, full = sweep(pair, ["small", "hybrid", "retry", "retry-paced"], fs.TABLE_FRACS)
        # consistency with the recorded artifact run
        rec_ci = recorded[f"bootstrap_causal_{pair}"]
        for f in fs.TABLE_FRACS:
            for p in ("small", "large", "hybrid", "paced+cap", "admit"):
                assert tuple(boot[str(f)]["ci95"][p]) == tuple(rec_ci[str(f)]["ci95"][p]), (pair, f, p)
        rec_rows = {f"{r['frac']}|{r['policy']}": r for r in recorded[f"causal_{pair}"]["rows"]}
        for f in fs.TABLE_FRACS:
            assert abs(rows[f"{f}|hybrid"]["succ_mean"] - rec_rows[f"{f}|hybrid"]["succ_mean"]) < 1e-9
        print(f"===== {pair}: recorded Table I brackets and means reproduced (assertions passed)")
        print(f"full large-model need: ${full:.4f}")
        # order-level paired difference (same 500 orders): paced+spend-down minus small
        order_diff = {}
        for f in fs.TABLE_FRACS:
            d = [a - b for a, b in zip(rows[f"{f}|hybrid"]["per_order"], rows[f"{f}|small"]["per_order"])]
            order_diff[str(f)] = {"mean": statistics.mean(d), "p05": fs.pct(d, 0.05), "p95": fs.pct(d, 0.95),
                                  "min": min(d), "share_le0": sum(x <= 0 for x in d) / len(d)}
        for f in fs.TABLE_FRACS:
            b = boot[str(f)]
            print(f"--- budget {f:.0%}")
            for p in pols:
                m = rows.get(f"{f}|{p}", {}).get("succ_mean")
                ms = f"{m:5.2f}" if m is not None else "  -  "
                extra = ""
                if f"{f}|{p}" in rows:
                    r = rows[f"{f}|{p}"]
                    extra = f" spent ${r['spent_mean']:.4f} esc {r['esc_mean']:.2f} unstarted {r['unstarted_mean']:.2f}"
                print(f"{p:>12} mean500 {ms} boot95 {b['ci95'][p]}{extra}")
            for k, v in b["diff"].items():
                print(f"  paired {k:>22}: mean {v['mean']:+.2f} 95% [{v['ci95'][0]}, {v['ci95'][1]}] share<=0 {v['share_le0']:.3f}")
            od = order_diff[str(f)]
            print(f"  order-level hybrid - small over 500 orders: mean {od['mean']:+.2f} p05..p95 [{od['p05']}, {od['p95']}] "
                  f"min {od['min']} share<=0 {od['share_le0']:.3f}")
        for r in rows.values():
            r.pop("per_order")
        res[pair] = {"bootstrap": boot, "orders": rows, "order_diff_hybrid_small": order_diff, "full_large": full}
    (HERE / "paired_and_retry.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
