#!/usr/bin/env python3
"""Fixed-spend replay for the OPTINET 2026 paper.

A batch of 50 tasks shares one spend pool B. Tasks arrive in a random order; each policy
decides, step by step, whether to keep running, stop, or move a task from the small to
the large model. Reuses the loaders and recorded traces in this repository's logs/ (all three rollouts
per model).
No API calls. Writes results/fixed_spend.json and prints the tables used in the paper.

    python3 optinet/fixed_spend.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import replay  # noqa: E402
from replay import step_cost  # noqa: E402
from escalation import trigger_step  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
ORDERS, SEED, BOOT = 500, 42, 2000
FRACS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.85, 0.88, 0.90, 0.95, 1.00, 1.10]
TABLE_FRACS = [0.20, 0.40, 0.80, 1.00]
PAIRS = {"cached": ("small_cache_budget", "frontier_cache"), "uncached": ("small", "frontier")}
# frontier_low is excluded: its step log was cut short (74 of about 500 steps; see the artifact README).
SMALL_RUNS, LARGE_RUNS = ["small", "small_cache_budget", "small_low"], ["frontier", "frontier_cache"]
POLICIES = ["small", "large", "stop", "stop-fallback", "admit", "route", "random-paced", "paced", "paced-abandon", "paced+cap", "hybrid", "oracle-paced"]
# Processed tokens (prompt incl. cache reads and writes, plus output) of the steps that ran, by model.
TOK = {"small": 0, "large": 0}


def run_steps(steps, pool, cap=None):
    """Run steps against the pool. Pool rule (provider credit limit): a request is sent
    while the pool is positive, so the last one may overdraw. Optional per-task cap:
    predictive, forecasting each step as the previous step's cost.
    Returns (spent, latency, finished, reason)."""
    spent, lat, prev = 0.0, 0.0, None
    for s in steps:
        if pool - spent <= 0:
            return spent, lat, False, "pool"
        if cap is not None and prev is not None and spent + prev > cap:
            return spent, lat, False, "cap"
        c = step_cost(s)
        TOK["large" if "sonnet" in s["model"] else "small"] += s["prompt"] + s["usage"]["output_tokens"]
        spent += c
        lat += s["latency_s"]
        prev = c
    return spent, lat, True, None


def task_cost(t):
    return sum(step_cost(s) for s in t["steps"])


def loo_estimates(small, large, ids):
    """Expected cost per task on each model, leave-one-out: for task i, the mean recorded
    cost of the other tasks (in deployment this would come from the operator's history)."""
    cs = {i: task_cost(small[i]) for i in ids}
    cl = {i: task_cost(large[i]) for i in ids}
    ts, tl, n = sum(cs.values()), sum(cl.values()), len(ids)
    return {i: ((ts - cs[i]) / (n - 1), (tl - cl[i]) / (n - 1)) for i in ids}


class OnlineEst:
    """Causal cost estimates, as BudgetPacer's CostEstimator keeps them: start from priors
    and use the running mean cost of tasks already finished in this batch on each model."""

    def __init__(self, prior_small, prior_large):
        self.prior = {"small": prior_small, "large": prior_large}
        self.sum, self.n = {"small": 0.0, "large": 0.0}, {"small": 0, "large": 0}

    def fresh(self):
        return OnlineEst(self.prior["small"], self.prior["large"])

    def mean(self, m):
        return self.sum[m] / self.n[m] if self.n[m] else self.prior[m]

    def __getitem__(self, tid):
        return self.mean("small"), self.mean("large")

    def observe(self, m, cost):
        self.sum[m] += cost
        self.n[m] += 1


def fires(S):
    j = trigger_step(S, "first_error")
    return j if j is not None and j != len(S["steps"]) - 1 else None


def simulate(policy, order, small, large, B, est, rng=None, rate=0.38):
    pool, succ, lat = B, 0, 0.0
    TOK["small"] = TOK["large"] = 0
    if isinstance(est, OnlineEst):
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
        fm, att = "small", None
        if policy == "small":
            sp, la, done, why = run_steps(S["steps"], pool)
            ok = done and S["success"]
        elif policy == "large":
            sp, la, done, why = run_steps(L["steps"], pool)
            ok, fm = done and L["success"], "large"
        elif policy == "stop":  # early stopping only: paced per-task cap on the large model
            sp, la, done, why = run_steps(L["steps"], pool, pool / (k_after + 1))
            ok, fm = done and L["success"], "large"
        elif policy == "stop-fallback":  # the same cap, but a capped task reruns on the small model
            sp, la, done, why = run_steps(L["steps"], pool, pool / (k_after + 1))
            ok, fm = done and L["success"], "large"
            if why == "cap" and pool - sp > 0:
                sp2, la2, done, why = run_steps(S["steps"], pool - sp)
                ok, fm, att, sp, la = done and S["success"], "small", sp2, sp + sp2, la + la2
        elif policy in ("admit", "random-paced", "oracle-paced"):  # pre-run assignment, same pacing gate
            # admit: admission control, large model whenever the pool can afford it after
            # reserving the small model for every remaining task, small model otherwise
            want = (True if policy == "admit" else
                    rng.random() < rate if policy == "random-paced" else not S["success"])
            room = pool - k_after * e_s
            if want and room >= e_l:
                esc += 1
                sp, la, done, why = run_steps(L["steps"], pool, room)
                ok, fm = done and L["success"], "large"
            else:
                sp, la, done, why = run_steps(S["steps"], pool)
                ok = done and S["success"]
        elif policy == "hybrid" and pool >= (k_after + 1) * e_l:  # pool covers large for every task left
            esc += 1
            sp, la, done, why = run_steps(L["steps"], pool, pool - k_after * e_s)
            ok, fm = done and L["success"], "large"
        else:  # routing policies start on the small model (hybrid falls through to paced + cap)
            j = fires(S)
            if j is None:
                sp, la, done, why = run_steps(S["steps"], pool)
                ok = done and S["success"]
            else:
                pre, la1, done1, why1 = run_steps(S["steps"][: j + 1], pool)
                if not done1:
                    sp, la, done, why, ok = pre, la1, False, why1, False
                else:
                    room = pool - pre - k_after * e_s
                    if policy == "route" or room >= e_l:
                        esc += 1
                        cap = None if policy in ("route", "paced") else max(room, 0.0)  # paced+cap, hybrid
                        sp2, la2, done, why = run_steps(L["steps"], pool - pre, cap)
                        ok, fm = done and L["success"], "large"
                    elif policy == "paced-abandon":  # refused escalation: stop the task here
                        sp2, la2, done, why, ok = 0.0, 0.0, False, "cap", False
                    else:  # anti-thrash: no escalation, finish on the small model
                        sp2, la2, done, why = run_steps(S["steps"][j + 1:], pool - pre)
                        ok = done and S["success"]
                    sp, la = pre + sp2, la1 + la2
                    att = sp2 if fm == "large" else sp
        if isinstance(est, OnlineEst) and why is None:  # cost of the attempt that finished
            est.observe(fm, sp if att is None else att)
        pool -= sp
        lat += la
        succ += int(bool(ok))
        if not ok:
            reasons[why if why else "model"] += 1
    return {"succ": succ, "spent": B - pool, "lat": lat / n, "reasons": reasons, "esc": esc,
            "overdraw": max(0.0, -pool), "tok_small": TOK["small"], "tok_large": TOK["large"]}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def setup(sl, ll):
    small, large = replay.load(sl), replay.load(ll)
    ids = sorted(set(small) & set(large))
    rate = sum(fires(small[i]) is not None for i in ids) / len(ids)
    return small, large, ids, loo_estimates(small, large, ids), rate


def out_of_sample_prior(prior_runs):
    """Causal estimator whose priors are the mean task costs of a different pair of
    recorded runs (other rollouts, other billing), so no cost of the batch under test is
    known in advance."""
    ps, pl = (replay.load(r) for r in prior_runs)
    ms = statistics.mean(task_cost(t) for t in ps.values())
    ml = statistics.mean(task_cost(t) for t in pl.values())
    return OnlineEst(ms, ml)


def sweep(sl, ll, fracs, orders_n, policies, est_override=None):
    small, large, ids, est, rate = setup(sl, ll)
    if est_override is not None:
        est = est_override
    full_large = sum(task_cost(large[i]) for i in ids)
    rng = random.Random(SEED)
    orders = [rng.sample(ids, len(ids)) for _ in range(orders_n)]
    rows = []
    for f in fracs:
        B = f * full_large
        for p in policies:
            prng = random.Random(SEED + 1)
            sims = [simulate(p, o, small, large, B, est, prng, rate) for o in orders]
            s = [x["succ"] for x in sims]
            rows.append({"frac": f, "budget": B, "policy": p,
                         "succ_mean": statistics.mean(s), "succ_p05": pct(s, 0.05), "succ_p95": pct(s, 0.95),
                         "spent_mean": statistics.mean(x["spent"] for x in sims),
                         "lat_mean": statistics.mean(x["lat"] for x in sims),
                         "esc_mean": statistics.mean(x["esc"] for x in sims),
                         "overdraw_max": max(x["overdraw"] for x in sims),
                         "tok_per_solved": sum(x["tok_small"] + x["tok_large"] for x in sims) / max(1, sum(s)),
                         "tok_large_share": sum(x["tok_large"] for x in sims) / max(1, sum(x["tok_small"] + x["tok_large"] for x in sims)),
                         "reasons": {k: statistics.mean(x["reasons"][k] for x in sims) for k in sims[0]["reasons"]}})
    meta = {"full_large": full_large, "full_small": sum(task_cost(small[i]) for i in ids), "orders": orders_n,
            "esc_rate": rate, "large_succ": sum(large[i]["success"] for i in ids),
            "small_succ": sum(small[i]["success"] for i in ids)}
    return {"meta": meta, "rows": rows}


def bootstrap(sl, ll, est_override=None):
    """Resample tasks with replacement (one random order per resample) to put task-level
    95% intervals on solved tasks, and on paired differences against the baselines."""
    small, large, ids, est, rate = setup(sl, ll)
    if est_override is not None:
        est = est_override
    full_large = sum(task_cost(large[i]) for i in ids)
    out = {}
    for f in TABLE_FRACS:
        B = f * full_large
        rng, draws = random.Random(SEED), {p: [] for p in POLICIES}
        for _ in range(BOOT):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            prng_seed = rng.randrange(1 << 30)
            for p in POLICIES:
                draws[p].append(simulate(p, sample, small, large, B, est, random.Random(prng_seed), rate)["succ"])
        ci = {p: (pct(v, 0.025), pct(v, 0.975)) for p, v in draws.items()}
        diff = {}
        for ours in ["paced+cap", "hybrid"]:
            for base in ["large", "admit", "route", "random-paced", "stop", "stop-fallback", "paced-abandon", "paced+cap"]:
                if base == ours:
                    continue
                d = [a - b for a, b in zip(draws[ours], draws[base])]
                diff[f"{ours} - {base}"] = {"mean": statistics.mean(d), "ci95": (pct(d, 0.025), pct(d, 0.975)),
                                            "share_le0": sum(x <= 0 for x in d) / len(d)}
        out[str(f)] = {"ci95": ci, "diff": diff}
    return out


def pairings():
    """All six pairings of a small-model rollout with a large-model rollout; budgets are a
    share of that pairing's own large-model need, costs as recorded."""
    out = []
    for sl in SMALL_RUNS:
        for ll in LARGE_RUNS:
            res = sweep(sl, ll, TABLE_FRACS, 200, ["large", "stop", "admit", "route", "paced+cap", "hybrid"])
            out.append({"small": sl, "large": ll, "meta": res["meta"],
                        "rows": [{k: r[k] for k in ("frac", "policy", "succ_mean")} for r in res["rows"]]})
    return out


def main():
    out = {}
    for name, (sl, ll) in PAIRS.items():
        out[name] = sweep(sl, ll, FRACS, ORDERS, POLICIES)
        print(f"===== {name}", json.dumps(out[name]["meta"]))
        print(f"{'frac':>5} {'B':>7} {'policy':>13} {'succ':>6} {'p05':>4} {'p95':>4} {'spent':>7} {'lat':>6} {'esc':>5} {'odraw':>7}  reasons")
        for r in out[name]["rows"]:
            rs = " ".join(f"{k}={v:.1f}" for k, v in r["reasons"].items())
            print(f"{r['frac']:>5.2f} {r['budget']:>7.4f} {r['policy']:>13} {r['succ_mean']:>6.2f} {r['succ_p05']:>4} {r['succ_p95']:>4} "
                  f"{r['spent_mean']:>7.4f} {r['lat_mean']:>6.1f} {r['esc_mean']:>5.1f} {r['overdraw_max']:>7.4f} tok/solved {r['tok_per_solved'] / 1e3:6.1f}k L{r['tok_large_share']:.2f}  {rs}")
    out["bootstrap_cached"] = bootstrap(*PAIRS["cached"])
    print("===== bootstrap (cached)")
    print(json.dumps(out["bootstrap_cached"], indent=1))
    # Causal estimates with out-of-sample priors: the cached batch starts from the uncached
    # runs' mean task costs and vice versa, then learns only from tasks already finished.
    prior_runs = {"cached": ("small", "frontier"), "uncached": ("small_cache_budget", "frontier_cache")}
    causal_pols = ["large", "admit", "random-paced", "paced+cap", "hybrid", "oracle-paced", "paced", "paced-abandon"]
    for name, (sl, ll) in PAIRS.items():
        prior = out_of_sample_prior(prior_runs[name])
        out[f"causal_{name}"] = sweep(sl, ll, FRACS, ORDERS, causal_pols, est_override=prior)
        out[f"causal_{name}"]["prior"] = prior.prior
        print(f"===== causal {name}, prior {prior.prior}")
        for r in out[f"causal_{name}"]["rows"]:
            if r["frac"] in TABLE_FRACS:
                print(f"{r['frac']:>5.2f} {r['policy']:>13} {r['succ_mean']:6.2f} [{r['succ_p05']}, {r['succ_p95']}] "
                      f"unstarted {r['reasons']['unstarted']:.2f} overdraw {r['overdraw_max']:.4f}")
    for name in PAIRS:
        out[f"bootstrap_causal_{name}"] = bootstrap(*PAIRS[name], est_override=out_of_sample_prior(prior_runs[name]))
        print(f"===== bootstrap (causal, {name})")
        print(json.dumps(out[f"bootstrap_causal_{name}"], indent=1))
    out["pairings"] = pairings()
    print("===== six pairings: solved (mean over 200 orders)")
    for f in TABLE_FRACS:
        for p in ["large", "stop", "admit", "route", "paced+cap", "hybrid"]:
            v = [r["succ_mean"] for x in out["pairings"] for r in x["rows"] if r["frac"] == f and r["policy"] == p]
            print(f"{f:.2f} {p:>10} min {min(v):.1f} max {max(v):.1f} mean {statistics.mean(v):.1f}")
    v = [(x["small"], x["large"], [r["succ_mean"] for r in x["rows"] if r["frac"] == f and r["policy"] == "hybrid"][0]
          - max(r["succ_mean"] for r in x["rows"] if r["frac"] == f and r["policy"] in ("admit", "paced+cap")))
         for x in out["pairings"] for f in TABLE_FRACS]
    print("hybrid minus best of admit and paced+cap, per pairing and budget:", [round(d, 1) for *_, d in v], "min:", min(d for *_, d in v))
    print("full-budget paced+cap vs large per pairing:",
          [(x["small"], x["large"], [r["succ_mean"] for r in x["rows"] if r["frac"] == 1.0 and r["policy"] == p][0])
           for x in out["pairings"] for p in ("paced+cap", "large")])
    OUT.mkdir(exist_ok=True)
    (OUT / "fixed_spend.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
