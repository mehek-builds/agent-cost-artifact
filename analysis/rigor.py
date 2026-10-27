"""Robustness of the escalation results, from the recorded runs only (no API calls).

python3 analysis/rigor.py  -> prints a summary, writes results/rigor.json

1. Baselines from prior work, replayed on the same 50 tasks and the same recorded runs:
   random routing at the escalation policy's rate, a learned pre-run router (RouteLLM-style,
   logistic regression on task features known before the run, leave-one-out), and a cascade
   that runs the small model to the end and reruns on the large model when a run-end check
   flags it (FrugalGPT-style; check = "the run raised a tool error", and a perfect check as
   an upper bound).
2. Held-out trigger selection: the trigger is chosen on training folds from a family of
   causal triggers and evaluated on the held-out fold (5-fold, 200 random splits).
3. Run-to-run robustness: each task has two independent small-model runs (cached and
   uncached) and two large-model runs; the policy is replayed on all four pairings.
4. Tool-call cap: whether the five-call cap of the cached small-model run bound, and on
   which tasks.
5. Energy with cloud prefill: the break-even cloud energy per output token when the cloud's
   prompt processing is charged with the relative weights measured on the edge device.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from escalation import escalate, trigger_step  # noqa: E402
from replay import CONFIGS, boot_ratio_ci, load, step_cost, wilson  # noqa: E402

RUNS = {label: load(log) for label, (log, _) in CONFIGS.items()}
IDS = sorted(RUNS["small"], key=lambda i: int(i.rsplit("_", 1)[1]))
N = len(IDS)
SMALL, LARGE = RUNS["small+cache"], RUNS["large+cache"]
cost = lambda t: sum(step_cost(s) for s in t["steps"])  # noqa: E731


def row(costs, oks):
    c, k = sum(costs), sum(oks)
    return {"success": k / N, "successes": k, "wilson95": wilson(k, N), "cost_per_task": c / N,
            "cost_per_success": c / k if k else None, "cost_per_success_ci95": boot_ratio_ci(costs, oks)}


# ---------------------------------------------------------------- triggers

def fires(task: dict, trig: tuple) -> int | None:
    """Causal triggers: ('err', n) the n-th error step; ('err_by', k) an error within the first k
    steps; ('steps', k) the run reaches step k; ('err_or_steps', k) first error or step k."""
    kind, k = trig
    steps = task["steps"]
    if kind == "err":
        return trigger_step(task, {1: "first_error", 2: "second_error"}[k])
    first_err = next((i for i, s in enumerate(steps) if s["n_errors"] > 0), None)
    if kind == "err_by":
        return first_err if first_err is not None and first_err < k else None
    if kind == "steps":
        return k - 1 if len(steps) > k else None
    if kind == "err_or_steps":
        cands = [x for x in (first_err, k - 1 if len(steps) > k else None) if x is not None]
        return min(cands) if cands else None
    raise ValueError(kind)


TRIGGERS = [("err", 1), ("err", 2)] + [("err_by", k) for k in (2, 4, 6, 8)] + \
           [("steps", k) for k in (6, 8, 10, 12, 16)] + [("err_or_steps", k) for k in (8, 12, 16)]


def replay_trigger(trig, ids, small=SMALL, large=LARGE):
    out = []
    for i in ids:
        s, l = small[i], large[i]
        j = fires(s, trig)
        if j is None or j == len(s["steps"]) - 1:
            out.append((cost(s), int(s["success"]), False))
        else:
            out.append((sum(step_cost(x) for x in s["steps"][: j + 1]) + cost(l), int(l["success"]), True))
    return out


def score(res):
    c, k = sum(r[0] for r in res), sum(r[1] for r in res)
    return c, k


def select(train, bar_pts=0.10):
    """Choose the trigger with the lowest cost per success on the training tasks among those within
    bar_pts of always-large success on the same tasks; fall back to the most successful trigger."""
    large_rate = sum(int(LARGE[i]["success"]) for i in train) / len(train)
    best, fallback = None, None
    for t in TRIGGERS:
        c, k = score(replay_trigger(t, train))
        rate = k / len(train)
        cps = c / k if k else float("inf")
        if fallback is None or rate > fallback[1]:
            fallback = (t, rate)
        if rate >= large_rate - bar_pts and (best is None or cps < best[1]):
            best = (t, cps)
    return (best or fallback)[0]


def heldout(splits=200, folds=5, seed=7):
    rng = random.Random(seed)
    succ, cps, picks = [], [], {}
    for _ in range(splits):
        ids = IDS[:]
        rng.shuffle(ids)
        tot_c = tot_k = 0
        for f in range(folds):
            test = ids[f::folds]
            train = [i for i in ids if i not in test]
            t = select(train)
            picks[str(t)] = picks.get(str(t), 0) + 1
            c, k = score(replay_trigger(t, test))
            tot_c, tot_k = tot_c + c, tot_k + k
        succ.append(tot_k / N)
        cps.append(tot_c / tot_k)
    q = lambda xs: [float(np.percentile(xs, p)) for p in (2.5, 50, 97.5)]  # noqa: E731
    return {"success_p2.5_50_97.5": q(succ), "cost_per_success_p2.5_50_97.5": q(cps),
            "mean_success": statistics.mean(succ), "mean_cost_per_success": statistics.mean(cps),
            "trigger_picks": dict(sorted(picks.items(), key=lambda kv: -kv[1])), "splits": splits, "folds": folds}


# ---------------------------------------------------------------- baselines

def features():
    spec = {t["id"]: t for t in map(json.loads, open(ROOT / "edge" / "harness" / "bfcl_v4_multi_turn.jsonl"))}
    X = []
    for i in IDS:
        t = spec[i]
        text = " ".join(m["content"] for turn in t["turns"] for m in turn if isinstance(m.get("content"), str))
        X.append([len(t["turns"]), len(t.get("involved_classes", [])), len(text) / 1000,
                  sum(1 for turn in t["turns"] if not turn), len(json.dumps(t.get("initial_config", {}))) / 1000])
    return np.array(X, dtype=float)


def learned_router():
    """Leave-one-out logistic regression predicting small-model success from pre-run features;
    route to the large model when predicted success is below a threshold chosen on the 49 training
    tasks with the same rule as the trigger selection."""
    X, y = features(), np.array([int(SMALL[i]["success"]) for i in IDS])
    out = []
    for n, i in enumerate(IDS):
        tr = [m for m in range(N) if m != n]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(C=1.0).fit((X[tr] - mu) / sd, y[tr])
        p_tr = clf.predict_proba((X[tr] - mu) / sd)[:, 1]
        large_rate = statistics.mean(int(LARGE[IDS[m]]["success"]) for m in tr)
        best = None
        for tau in np.linspace(0, 1, 41):
            cs = [cost(LARGE[IDS[m]]) if p < tau else cost(SMALL[IDS[m]]) for m, p in zip(tr, p_tr)]
            ks = [int(LARGE[IDS[m]]["success"]) if p < tau else int(SMALL[IDS[m]]["success"]) for m, p in zip(tr, p_tr)]
            rate = sum(ks) / len(tr)
            if rate >= large_rate - 0.10 and sum(ks) and (best is None or sum(cs) / sum(ks) < best[1]):
                best = (tau, sum(cs) / sum(ks))
        tau = best[0] if best else 1.0
        p = clf.predict_proba(((X[n] - mu) / sd)[None])[0, 1]
        t = LARGE[i] if p < tau else SMALL[i]
        out.append((cost(t), int(t["success"]), p < tau))
    auc_p = []
    return out


def baselines(rate):
    s_c = [cost(SMALL[i]) for i in IDS]
    l_c = [cost(LARGE[i]) for i in IDS]
    s_k = [int(SMALL[i]["success"]) for i in IDS]
    l_k = [int(LARGE[i]["success"]) for i in IDS]
    res = {"small": row(s_c, s_k), "large": row(l_c, l_k)}
    # random routing at the policy's escalation rate (expectation over assignments)
    ec = [(1 - rate) * a + rate * b for a, b in zip(s_c, l_c)]
    ek = (1 - rate) * sum(s_k) + rate * sum(l_k)
    res["random_router"] = {"rate": rate, "success": ek / N, "cost_per_task": sum(ec) / N, "cost_per_success": sum(ec) / ek}
    lr = learned_router()
    res["learned_router_loo"] = {**row([r[0] for r in lr], [r[1] for r in lr]), "routed_large": sum(r[2] for r in lr)}
    # cascades: small runs to the end; a run-end check decides whether to rerun on large
    flag = [any(s["n_errors"] > 0 for s in SMALL[i]["steps"]) for i in IDS]
    cc = [a + (b if f else 0) for a, b, f in zip(s_c, l_c, flag)]
    ck = [lk if f else sk for sk, lk, f in zip(s_k, l_k, flag)]
    res["cascade_error_check"] = {**row(cc, ck), "reruns": sum(flag)}
    pc = [a + (0 if sk else b) for a, b, sk in zip(s_c, l_c, s_k)]
    pk = [max(sk, lk) for sk, lk in zip(s_k, l_k)]
    res["cascade_perfect_check"] = {**row(pc, pk), "reruns": N - sum(s_k)}
    fe = replay_trigger(("err", 1), IDS)
    res["first_error"] = {**row([r[0] for r in fe], [r[1] for r in fe]), "escalated": sum(r[2] for r in fe)}
    return res


# ---------------------------------------------------------------- run-to-run, tool cap, energy

def pairings():
    out = {}
    for sl in ("small", "small+cache"):
        for ll in ("large", "large+cache"):
            res = replay_trigger(("err", 1), IDS, RUNS[sl], RUNS[ll])
            k = sum(r[1] for r in res)
            out[f"{sl} x {ll}"] = {"successes": k, "escalated": sum(r[2] for r in res)}
    stat = {"large_runs_agree": sum(RUNS["large"][i]["success"] == RUNS["large+cache"][i]["success"] for i in IDS),
            "small_runs_agree": sum(RUNS["small"][i]["success"] == RUNS["small+cache"][i]["success"] for i in IDS)}
    trig_agree = sum((fires(RUNS["small"][i], ("err", 1)) is None) == (fires(RUNS["small+cache"][i], ("err", 1)) is None) for i in IDS)
    return {"first_error_by_pairing": out, **stat, "trigger_decision_agree_across_small_runs": trig_agree}


def tool_cap():
    capped = [i for i in IDS if any(s["n_budget_denied"] > 0 for s in RUNS["small+cache"][i]["steps"])]
    return {"tasks_where_cap_bound": len(capped),
            "capped_success_cached": sum(int(RUNS["small+cache"][i]["success"]) for i in capped),
            "capped_success_uncapped_run": sum(int(RUNS["small"][i]["success"]) for i in capped),
            "first_error_fires_on_capped": sum(fires(SMALL[i], ("err", 1)) is not None for i in capped)}


def energy_with_prefill():
    """Break-even cloud J per output token for first-error vs always-large, when cloud prompt tokens
    are charged at the edge-measured relative weight (fresh prompt token = w_fresh output tokens;
    cache write as fresh; cache read at alpha times fresh)."""
    run = ROOT / "edge" / "runs" / "full"
    es = json.load(open(run / "energy_summary.json"))
    pe = json.load(open(run / "placement_edge.json"))
    w_fresh = es["fit_dynamic_j_per_fresh_prompt_token"] / es["fit_dynamic_j_per_output_token"]
    L = {}
    for l in open(ROOT / "logs" / "steps_frontier_cache.jsonl"):
        st = json.loads(l)
        L.setdefault(st["task_id"], []).append(st)
    etasks = {t["task_id"]: t for t in map(json.loads, open(run / "tasks_edge.jsonl"))}
    esteps = {}
    for l in open(run / "steps_edge.jsonl"):
        st = json.loads(l)
        esteps.setdefault(st["task_id"], []).append(st)
    escal = [tid for tid in etasks if any(x["n_errors"] > 0 for x in esteps.get(tid, []))]
    out = {"w_fresh_prompt_per_output_token": w_fresh}
    for alpha in (0.0, 0.1, 1.0):
        def W(tids):
            w = 0.0
            for tid in tids:
                for s in L[tid]:
                    u = s["usage"]
                    w += u["output_tokens"] + w_fresh * (u["input_tokens"] + u["cache_write_tokens"] + alpha * u["cache_read_tokens"])
            return w
        wl, wf = W(list(etasks)), W(escal)
        sl, sf = pe["large"]["successes"], pe["first_error"]["successes"]
        jf = pe["first_error"]["edge_joules"]
        out[f"alpha={alpha}"] = {"break_even_j_per_output_token": (jf / sf) / (wl / sl - wf / sf),
                                 "cloud_weight_per_task_large": wl / len(etasks)}
    wl0 = sum(s["usage"]["output_tokens"] for tid in etasks for s in L[tid])
    wf0 = sum(s["usage"]["output_tokens"] for tid in escal for s in L[tid])
    out["output_only"] = {"break_even_j_per_output_token": (pe["first_error"]["edge_joules"] / pe["first_error"]["successes"]) /
                          (wl0 / pe["large"]["successes"] - wf0 / pe["first_error"]["successes"])}
    return out


def main():
    b = baselines(rate=19 / N)
    res = {"baselines": b, "heldout_trigger": heldout(), "pairings": pairings(), "tool_cap": tool_cap(),
           "energy_with_prefill": energy_with_prefill()}
    (ROOT / "results" / "rigor.json").write_text(json.dumps(res, indent=1, default=str))
    for k, v in b.items():
        cps = v.get("cost_per_success")
        print(f"{k:24s} success {v['success']:.2f}  $/success {cps:.4f}" + (f"  CI {v['cost_per_success_ci95'][0]:.4f}-{v['cost_per_success_ci95'][1]:.4f}" if "cost_per_success_ci95" in v else ""))
    for k in ("heldout_trigger", "pairings", "tool_cap", "energy_with_prefill"):
        print(k, json.dumps(res[k], default=str)[:900])


if __name__ == "__main__":
    main()
