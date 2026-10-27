"""The recorded runs as repeated rollouts of the same tasks (no API calls).

python3 analysis/repeats.py  -> prints a summary, writes results/repeats.json

Multi-turn (50 BFCL v4 multi_turn_base tasks): each model ran every task three times, as three
configurations recorded on the same day: uncached, cached, and uncached at low reasoning effort. Caching
changes billing, not the model's inputs, and low effort did not change success measurably
(analysis/levers.py), so for success the three runs are treated as three rollouts per task. The cached
small-model run also capped tool calls at five per turn; results are also reported without it.

Single-turn (150 BFCL v4 tasks): the large model ran each task five times (three medium-effort runs, two
low-effort runs) and the small model eight times (three runs, three capped cached runs, two low-effort runs).

Reported: pooled success with task-clustered bootstrap intervals, run-to-run agreement, the per-task
distribution of successes, the paired model gap, and the first-error escalation policy replayed on all
nine pairings of a small-model rollout with a large-model rollout (restart mode: the small run up to its
first tool error, then the large run's full recorded cost and outcome).
"""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from replay import load, mcnemar_exact, step_cost  # noqa: E402

LOGS = ROOT / "logs"
B = 10000


def tasks_only(log):
    return {t["task_id"]: t for t in map(json.loads, open(LOGS / f"tasks_{log}.jsonl"))}


def clustered_ci(mat, seed=1):
    """mat: list of per-task lists of 0/1 outcomes. 95% CI of the pooled rate, resampling tasks."""
    rng, n = random.Random(seed), len(mat)
    vals = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        num = sum(sum(mat[i]) for i in idx)
        den = sum(len(mat[i]) for i in idx)
        vals.append(num / den)
    vals.sort()
    return vals[int(0.025 * B)], vals[int(0.975 * B) - 1]


def summarize(runs, ids):
    mat = [[int(r[i]["success"]) for r in runs] for i in ids]
    k = len(runs)
    dist = {str(j): sum(1 for m in mat if sum(m) == j) for j in range(k + 1)}
    return {"rollouts": k, "pooled_success": sum(map(sum, mat)) / (k * len(ids)),
            "ci95_task_clustered": clustered_ci(mat), "per_run_success": [sum(m[j] for m in mat) / len(ids) for j in range(k)],
            "tasks_same_outcome_every_run": dist["0"] + dist[str(k)], "tasks_by_successes": dist,
            "mean_task_success_probability": statistics.mean(sum(m) / k for m in mat)}


def paired_gap(a_runs, b_runs, ids, seed=2):
    """Per-task difference in success rate (b minus a), bootstrap over tasks, and a sign test."""
    d = [statistics.mean(int(r[i]["success"]) for r in b_runs) - statistics.mean(int(r[i]["success"]) for r in a_runs) for i in ids]
    rng = random.Random(seed)
    vals = sorted(statistics.mean(d[rng.randrange(len(d))] for _ in d) for _ in range(B))
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return {"mean_gap": statistics.mean(d), "ci95": [vals[int(0.025 * B)], vals[int(0.975 * B) - 1]],
            "tasks_b_better": pos, "tasks_a_better": neg, "sign_test_p": mcnemar_exact(neg, pos)}


def multi_turn():
    S = {"small": load("small"), "small+cache (capped)": load("small_cache_budget"), "small low effort": load("small_low")}
    L = {"large": tasks_only("frontier"), "large+cache": tasks_only("frontier_cache"), "large low effort": tasks_only("frontier_low")}
    ids = sorted(S["small"])
    out = {"small": summarize(list(S.values()), ids), "large": summarize(list(L.values()), ids),
           "small_without_capped_run": summarize([S["small"], S["small low effort"]], ids),
           "gap_large_minus_small": paired_gap(list(S.values()), list(L.values()), ids)}
    # first-error escalation on all nine pairings
    pair, fe_mat, large_mat = {}, {i: [] for i in ids}, {i: [] for i in ids}
    for sn, sr in S.items():
        for ln, lr in L.items():
            k = c = 0
            for i in ids:
                t = sr[i]
                j = next((n for n, s in enumerate(t["steps"]) if s["n_errors"] > 0), None)
                if j is None or j == len(t["steps"]) - 1:
                    ok, cost = int(t["success"]), sum(step_cost(s) for s in t["steps"])
                else:
                    ok, cost = int(lr[i]["success"]), sum(step_cost(s) for s in t["steps"][: j + 1]) + lr[i]["cost_usd"]
                k += ok
                c += cost
                fe_mat[i].append(ok)
            pair[f"{sn} x {ln}"] = {"successes": k, "cost_per_success": c / k,
                                    "same_billing": ("cache" in sn) == ("cache" in ln)}
    for i in ids:
        large_mat[i] = [int(r[i]["success"]) for r in L.values()]
    ks = [v["successes"] for v in pair.values()]
    out["first_error_nine_pairings"] = {"pairings": pair, "successes_min_median_max": [min(ks), statistics.median(ks), max(ks)],
                                         "pooled_success": sum(ks) / (9 * len(ids)),
                                         "ci95_task_clustered": clustered_ci([fe_mat[i] for i in ids])}
    # pooled first-error vs pooled large, paired by task
    d = [statistics.mean(fe_mat[i]) - statistics.mean(large_mat[i]) for i in ids]
    rng = random.Random(3)
    vals = sorted(statistics.mean(d[rng.randrange(len(d))] for _ in d) for _ in range(B))
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    out["first_error_minus_large"] = {"mean_gap": statistics.mean(d), "ci95": [vals[int(0.025 * B)], vals[int(0.975 * B) - 1]],
                                      "tasks_large_better": neg, "tasks_policy_better": pos, "sign_test_p": mcnemar_exact(neg, pos)}
    return out


def single_turn():
    st = LOGS / "single_turn"
    runs = {}
    for f in sorted(st.glob("tasks_*.jsonl")):
        cfg = f.stem[len("tasks_"):].split("__")[0]
        runs.setdefault("large" if cfg.startswith("frontier") else "small", []).append(
            {t["task_id"]: t for t in map(json.loads, open(f))})
    ids = sorted(runs["large"][0])
    out = {m: summarize(r, ids) for m, r in runs.items()}
    out["gap_large_minus_small"] = paired_gap(runs["small"], runs["large"], ids)
    for m, r in runs.items():
        costs = [sum(t[i]["cost_usd"] for i in ids) for t in r]
        out[m]["run_cost_cv"] = statistics.pstdev(costs) / statistics.mean(costs)
    return out


def main():
    res = {"multi_turn": multi_turn(), "single_turn": single_turn()}
    (ROOT / "results" / "repeats.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
