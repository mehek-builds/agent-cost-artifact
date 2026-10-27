"""Further analyses of the recorded runs (no API calls).

python3 analysis/rigor2.py  -> prints a summary, writes results/rigor2.json

1. Matched-success comparison and the trigger frontier: every causal trigger of analysis/rigor.py is
   placed on the success / cost-per-success plane, the cost to reach target success rates is read off,
   and First-error is compared with a random mix of the two tiers that reaches the same success.
2. A learned in-run failure predictor: after the k-th small-tier step, a logistic regression on features
   observed so far (errors, calls, steps, prompt growth, output) predicts small-tier failure; leave-one-out,
   with the escalation threshold chosen on the training tasks by the same rule as the trigger selection.
3. Why escalation beats the run-to-completion cascade: the cascade additionally pays for every small-tier
   step after the trigger; the gap is reported against the share of the small run that remains.
4. Uncertainty of the edge energy fit: bootstrap over requests of the per-token coefficients.
5. A held-out budget for scenario S1: the budget quantile is chosen on training folds and scored on the
   held-out fold.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "edge"))
import rigor as G  # noqa: E402
from replay import guard, step_cost  # noqa: E402

IDS, N, S, L, cost = G.IDS, G.N, G.SMALL, G.LARGE, G.cost


def frontier():
    pts = {"small": (sum(int(S[i]["success"]) for i in IDS), sum(cost(S[i]) for i in IDS)),
           "large": (sum(int(L[i]["success"]) for i in IDS), sum(cost(L[i]) for i in IDS))}
    for t in G.TRIGGERS:
        c, k = G.score(G.replay_trigger(t, IDS))
        pts[str(t)] = (k, c)
    rows = {name: {"success": k / N, "cost_per_success": c / k, "cost_per_task": c / N} for name, (k, c) in pts.items()}
    targets = {}
    for tgt in (0.70, 0.76, 0.80, 0.86):
        ok = {n: r for n, r in rows.items() if r["success"] >= tgt - 1e-9}
        best = min(ok.items(), key=lambda kv: kv[1]["cost_per_success"])
        targets[str(tgt)] = {"policy": best[0], **best[1]}
    # random mix of the two tiers reaching First-error's success
    fe = rows["('err', 1)"]
    s0, s1 = rows["small"], rows["large"]
    w = (fe["success"] - s0["success"]) / (s1["success"] - s0["success"])
    mix_task = (1 - w) * s0["cost_per_task"] + w * s1["cost_per_task"]
    return {"policies": rows, "cheapest_at_target": targets,
            "matched_success_random_mix": {"large_share": w, "success": fe["success"],
                                           "cost_per_success": mix_task / fe["success"]}}


def in_run_predictor(k_steps=(2, 3, 4)):
    out = {}
    y = np.array([1 - int(S[i]["success"]) for i in IDS])  # 1 = small tier fails

    def feats(t, k):
        st = t["steps"][:k]
        return [sum(s["n_errors"] > 0 for s in st), sum(s["n_calls"] for s in st), len(st),
                (st[-1]["prompt"] - st[0]["prompt"]) / 1000 if st else 0.0,
                sum(s["usage"]["output_tokens"] for s in st) / 1000, int(len(t["steps"]) > k)]

    for k in k_steps:
        X = np.array([feats(S[i], k) for i in IDS], dtype=float)
        res = []
        for n in range(N):
            tr = [m for m in range(N) if m != n]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            clf = LogisticRegression(C=1.0).fit((X[tr] - mu) / sd, y[tr])
            p_tr = clf.predict_proba((X[tr] - mu) / sd)[:, 1]
            large_rate = statistics.mean(int(L[IDS[m]]["success"]) for m in tr)

            def run(i, esc):
                t = S[i]
                if not esc or len(t["steps"]) <= k:
                    return cost(t), int(t["success"]), False
                return sum(step_cost(s) for s in t["steps"][:k]) + cost(L[i]), int(L[i]["success"]), True

            best = None
            for tau in np.linspace(0.05, 0.95, 19):
                rr = [run(IDS[m], p > tau) for m, p in zip(tr, p_tr)]
                c, kk = sum(r[0] for r in rr), sum(r[1] for r in rr)
                if kk / len(tr) >= large_rate - 0.10 and (best is None or c / kk < best[1]):
                    best = (tau, c / kk)
            tau = best[0] if best else 0.5
            p = clf.predict_proba(((X[n] - mu) / sd)[None])[0, 1]
            res.append(run(IDS[n], p > tau))
        c, kk = sum(r[0] for r in res), sum(r[1] for r in res)
        out[f"after_step_{k}"] = {"success": kk / N, "cost_per_success": c / kk, "escalated": sum(r[2] for r in res)}
    return out


def cascade_gap():
    rows = []
    for i in IDS:
        j = G.fires(S[i], ("err", 1))
        if j is None or j == len(S[i]["steps"]) - 1:
            continue
        after = S[i]["steps"][j + 1:]
        rows.append({"task": i, "steps_total": len(S[i]["steps"]), "steps_after_trigger": len(after),
                     "remaining_share": len(after) / len(S[i]["steps"]),
                     "extra_cascade_usd": sum(step_cost(s) for s in after),
                     "large_usd": cost(L[i])})
    extra = sum(r["extra_cascade_usd"] for r in rows)
    large = sum(r["large_usd"] for r in rows)
    rem = [r["remaining_share"] for r in rows]
    # the cascade's extra spend scales with the small tier's per-step price relative to the large tier's;
    # report what the gap becomes if the small tier cost the same per step as the large (an expensive edge)
    ratio = statistics.mean(cost(L[i]) / max(len(L[i]["steps"]), 1) for i in IDS) / \
        statistics.mean(cost(S[i]) / max(len(S[i]["steps"]), 1) for i in IDS)
    return {"escalated_tasks": len(rows), "median_remaining_share": statistics.median(rem),
            "extra_cascade_usd_total": extra, "extra_as_share_of_large_usd_on_those_tasks": extra / large,
            "large_to_small_cost_per_step": ratio}


def edge_fit_ci(reps=2000, seed=11):
    from energy import energy, lstsq, samples
    run = ROOT / "edge" / "runs" / "full"
    s = samples(run / "power.txt")
    b0, b1 = (float((run / f).read_text()) for f in ("baseline_start", "baseline_end"))
    idle_s = [w for t0, t1, w in s if t0 >= b0 + 5 and t1 <= b1 - 5]
    idle = sum(idle_s) / len(idle_s)
    timing = [json.loads(l) for l in open(run / "timing_edge.jsonl")]
    prev, rows, ys = {}, [], []
    for t in timing:
        p, o = t["prompt_eval_count"], t["eval_count"]
        reused = prev.get(t["task_id"], 0)
        fresh = p if not reused else max(p - reused, 0)
        prev[t["task_id"]] = p + o
        rows.append([fresh, o, 1.0])
        ys.append(energy(s, t["t_start"], t["t_end"], idle))
    rng = random.Random(seed)
    co = []
    for _ in range(reps):
        idx = [rng.randrange(len(rows)) for _ in rows]
        co.append(lstsq([rows[i] for i in idx], [ys[i] for i in idx]))
    q = lambda xs: [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]  # noqa: E731
    return {"j_per_fresh_prompt_token_ci95": q([c[0] for c in co]), "j_per_output_token_ci95": q([c[1] for c in co]),
            "j_per_request_ci95": q([c[2] for c in co]), "idle_w": idle,
            "idle_w_sample_range": [min(idle_s), max(idle_s)]}


def heldout_budget(splits=200, folds=5, seed=5):
    base_ok = {i: int(L[i]["success"]) for i in IDS}
    full = {i: cost(L[i]) for i in IDS}

    def budget_for(train, q):
        sc = sorted(full[i] for i in train if base_ok[i])
        return sc[min(len(sc) - 1, math.ceil(q * len(sc)) - 1)]

    rng, res = random.Random(seed), []
    for _ in range(splits):
        ids = IDS[:]
        rng.shuffle(ids)
        k = c = cb = over = 0
        for f in range(folds):
            test = ids[f::folds]
            train = [i for i in ids if i not in test]
            base_rate = sum(base_ok[i] for i in train) / len(train)
            pick = None
            for q in (0.5, 0.75, 0.9, 0.95, 1.0):  # smallest budget keeping success within 5 points on train
                b = budget_for(train, q)
                rate = sum(int(guard(L[i], b, "reactive")[1]) for i in train) / len(train)
                if rate >= base_rate - 0.05:
                    pick = b
                    break
            pick = pick or budget_for(train, 1.0)
            for i in test:
                sp, ok, v = guard(L[i], pick, "reactive")
                k, c, cb, over = k + int(ok), c + sp, cb + full[i], over + int(v)
        res.append({"success": k / N, "saving": 1 - c / cb, "overshoot_tasks": over})
    q = lambda key: [float(np.percentile([r[key] for r in res], p)) for p in (2.5, 50, 97.5)]  # noqa: E731
    return {"success_p2.5_50_97.5": q("success"), "saving_p2.5_50_97.5": q("saving"),
            "overshoot_tasks_p2.5_50_97.5": q("overshoot_tasks"), "baseline_success": sum(base_ok.values()) / N}


def main():
    res = {"frontier": frontier(), "in_run_predictor": in_run_predictor(), "cascade_gap": cascade_gap(),
           "edge_fit_ci": edge_fit_ci(), "heldout_budget_S1": heldout_budget()}
    (ROOT / "results" / "rigor2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
