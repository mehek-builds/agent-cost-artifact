#!/usr/bin/env python3
"""Robustness analyses for the OPTINET 2026 paper, run through the shipped BudgetPacer.

1. estimate error   the pacer's cost estimates scaled by 0.25x to 4x, held fixed or learned
                    online from finished tasks (the deployed agent's cost per request moved
                    0.24x to 1.9x from one month to the next)
2. price pairs      the recorded multi-turn traces re-priced from their token counts with
                    other model pairs from the same provider's list prices (verified
                    2026-10-29); outcomes stay those of the recorded models
3. single turn      150 single-turn BFCL tasks with 8 small-model and 5 large-model
                    rollouts (task records only, so pool checks happen at task start)

    python3 optinet/extras.py   -> optinet/results/extras.json
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import fixed_spend as F  # noqa: E402
from budget_pacer import PriceCard, cost_of  # noqa: E402
from pacer_replay import run_batch  # noqa: E402
from replay import step_cost  # noqa: E402

SEED, ORDERS = 42, 200
FRACS = [0.2, 0.4, 0.8, 1.0]

# USD per million tokens (input, 5-minute cache write, cache read, output), from
# https://platform.claude.com/docs/en/about-claude/pricing, read 2026-10-29.
CARDS = {
    "sonnet-5.5": PriceCard(2.0, 2.5, 0.10, 10.0),
    "opus-5.5": PriceCard(4.0, 5.0, 0.20, 20.0),
    "sonnet-5": PriceCard(2.0, 2.5, 0.20, 10.0),
    "opus-5": PriceCard(5.0, 6.25, 0.50, 25.0),
    "sonnet-4.6": PriceCard(3.0, 3.75, 0.30, 15.0),
    "haiku-4.5": PriceCard(1.0, 1.25, 0.10, 5.0),
    "haiku-5.5": PriceCard(0.10, 0.125, 0.01, 0.50),      # prompts up to 100k tokens
    "haiku-5.5-long": PriceCard(0.50, 0.625, 0.05, 2.50),  # prompts over 100k tokens
}
PAIRS = [("sonnet-5.5", "haiku-5.5"), ("opus-5.5", "haiku-5.5"), ("sonnet-5", "haiku-4.5"),
         ("sonnet-4.6", "haiku-4.5"), ("opus-5", "haiku-4.5")]


def mean(xs):
    return statistics.mean(xs)


def pricer(large_card, small_card):
    def cost(s):
        if "sonnet" in s["model"]:
            card = CARDS[large_card]
        else:
            card = CARDS[small_card]
            if small_card == "haiku-5.5" and s["prompt"] > 100_000:
                card = CARDS["haiku-5.5-long"]
        u = s["usage"]
        return cost_of({"input_tokens": u["input_tokens"], "cache_write_tokens": u["cache_write_tokens"],
                        "cache_read_tokens": u["cache_read_tokens"], "output_tokens": u["output_tokens"]}, card)
    return cost


def loo(small, large, ids, cost):
    cs = {i: sum(cost(s) for s in small[i]["steps"]) for i in ids}
    cl = {i: sum(cost(s) for s in large[i]["steps"]) for i in ids}
    n, ts, tl = len(ids), sum(cs.values()), sum(cl.values())
    return {i: ((ts - cs[i]) / (n - 1), (tl - cl[i]) / (n - 1)) for i in ids}, tl, ts


def large_only(order, large, pool, cost):
    succ = 0
    for tid in order:
        if pool <= 0:
            continue
        spent, done = 0.0, True
        for s in large[tid]["steps"]:
            if pool - spent <= 0:
                done = False
                break
            spent += cost(s)
        pool -= spent
        succ += int(done and large[tid]["success"])
    return succ


def estimate_error():
    small, large, ids, est, _ = F.setup("small_cache_budget", "frontier_cache")
    full = sum(F.task_cost(large[i]) for i in ids)
    orders = [random.Random(SEED + k).sample(ids, len(ids)) for k in range(ORDERS)]
    rows = []
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        for online in (False, True):
            for f in FRACS:
                r = [run_batch(o, small, large, f * full, est, step_cost, online=online, prior_scale=scale)
                     for o in orders]
                rows.append({"scale": scale, "online": online, "frac": f, "succ": mean(x["succ"] for x in r),
                             "unstarted": mean(x["reasons"]["unstarted"] for x in r),
                             "overdraw_max": max(x["overdraw"] for x in r)})
    return rows


def price_pairs():
    small, large, ids, _, _ = F.setup("small_cache_budget", "frontier_cache")
    orders = [random.Random(SEED + k).sample(ids, len(ids)) for k in range(ORDERS)]
    out = []
    for lc, sc in PAIRS:
        cost = pricer(lc, sc)
        est, full, full_small = loo(small, large, ids, cost)
        for f in FRACS:
            B = f * full
            row = {"large": lc, "small": sc, "frac": f, "batch_large_usd": full, "batch_small_usd": full_small,
                   "large_only": mean(large_only(o, large, B, cost) for o in orders)}
            for name, kw in (("paced", {"spend_down": False}), ("paced_spend_down", {})):
                row[name] = mean(run_batch(o, small, large, B, est, cost, **kw)["succ"] for o in orders)
            out.append(row)
    return out


def single_turn():
    d = HERE.parent / "logs" / "single_turn"

    def load(path):
        return {t["task_id"]: t for t in map(json.loads, open(path))}
    smalls = [load(p) for p in sorted(d.glob("tasks_small*.jsonl"))]
    larges = [load(p) for p in sorted(d.glob("tasks_frontier*.jsonl"))]
    ids = sorted(set.intersection(*(set(r) for r in smalls + larges)))
    rows = []
    for f in FRACS:
        per = {k: [] for k in ("large", "small", "paced", "paced_spend_down", "admission")}
        spent = {k: [] for k in per}
        for S in smalls:
            for L in larges:
                cs = {i: S[i]["cost_usd"] for i in ids}
                cl = {i: L[i]["cost_usd"] for i in ids}
                n, ts, tl = len(ids), sum(cs.values()), sum(cl.values())
                B = f * tl
                rng = random.Random(SEED)
                for _ in range(20):
                    order = rng.sample(ids, n)
                    for pol in per:
                        pool, succ = B, 0
                        for pos, tid in enumerate(order):
                            if pool <= 0:
                                continue
                            k = n - pos - 1
                            es, el = (ts - cs[tid]) / (n - 1), (tl - cl[tid]) / (n - 1)
                            use_large = (pol == "large"
                                         or (pol == "paced_spend_down" and pool >= (k + 1) * el)
                                         or (pol == "admission" and pool - k * es >= el))
                            # single-turn tasks execute no tools, so the first-error trigger never
                            # fires and "paced" keeps every task on the small model
                            t = (L if use_large else S)[tid]
                            pool -= t["cost_usd"]
                            succ += int(t["success"])
                        per[pol].append(succ)
                        spent[pol].append(B - pool)
        rows.append({"frac": f, **{k: {"mean": mean(v), "min": min(v), "max": max(v),
                                        "spent": mean(spent[k])} for k, v in per.items()}})
    meta = {"tasks": len(ids), "small_rollouts": len(smalls), "large_rollouts": len(larges),
            "small_success": [sum(r[i]["success"] for i in ids) for r in smalls],
            "large_success": [sum(r[i]["success"] for i in ids) for r in larges],
            "small_usd": [sum(r[i]["cost_usd"] for i in ids) for r in smalls],
            "large_usd": [sum(r[i]["cost_usd"] for i in ids) for r in larges]}
    return {"meta": meta, "rows": rows}


def main():
    small, large, ids, _, _ = F.setup("small_cache_budget", "frontier_cache")
    check = pricer("sonnet-5.5", "haiku-5.5")
    gap = max(abs(check(s) - step_cost(s)) for t in list(small.values()) + list(large.values()) for s in t["steps"])
    out = {"repricing_check_max_abs_gap_usd": gap, "estimate_error": estimate_error(),
           "price_pairs": price_pairs(), "single_turn": single_turn()}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "extras.json").write_text(json.dumps(out, indent=1))
    print("re-pricing check, max per-step gap vs recorded cost:", gap)
    for r in out["estimate_error"]:
        print("est", r)
    for r in out["price_pairs"]:
        print("price", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print(json.dumps(out["single_turn"], indent=1))


if __name__ == "__main__":
    main()
