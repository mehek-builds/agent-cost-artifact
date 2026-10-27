#!/usr/bin/env python3
"""Trace replay of the escalation tactic (RQ2, RQ3) and of the managing system's own
overhead (RQ5).

No API calls.  python3 escalation.py  -> prints tables, writes results/escalation_numbers.json

Each policy starts a task on the small model and may hand it to the large model once.
Two handoff formats are replayed:

restart  The large model reruns the task from its first turn with a fresh context. Its
         run is the recorded large-model run of the same task, so cost and outcome are
         taken from the log (exact up to sampling: one rollout per task).
full     The large model continues mid-run with the small model's history. This is a
         counterfactual. Cost = small prefix + one cold cache write of the history on the
         large model + the large model's recorded steps from the trigger turn on, with the
         first of those steps re-billed as a cold write instead of a cache read. Success
         is reported as an interval [small outcome, large outcome] and as a calibrated
         estimate that recovers a fraction q of the gap, q from The Handoff Tax (0.36 to 0.47).
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from replay import (CONFIGS, ROOT, boot_ci, boot_ratio_ci, card, guard, load, mcnemar_exact,
                    step_cost, wilson)

PAIRS = {"cached": ("small+cache", "large+cache"), "uncached": ("small", "large")}
Q_HANDOFF = (0.36, 0.47)


# ---------------------------------------------------------------- triggers (Analyze)

def trigger_step(task: dict, rule: str) -> int | None:
    """Index of the step after which the trigger fires, or None.

    first_error   the first step whose tool calls returned an error
    second_error  the second such step (errors need not be consecutive)
    fixed_half    after half of the run's steps, regardless of signals (the fixed-point
                  design of prior work; it needs the run length, so it is not causal)
    """
    steps = task["steps"]
    if rule == "fixed_half":
        return max(0, len(steps) // 2 - 1) if len(steps) > 1 else None
    need = {"first_error": 1, "second_error": 2}[rule]
    seen = 0
    for idx, s in enumerate(steps):
        if s["n_errors"] > 0:
            seen += 1
            if seen == need:
                return idx
    return None


# ---------------------------------------------------------------- handoff (Plan, Execute)

def handoff_write(small_step: dict, large_model: str, cached: bool) -> float:
    """Adaptation cost of a full-history handoff: the history the large model must
    take in fresh. With caching it is written to a cold cache; without caching the
    large model would read it as fresh input anyway, so there is no extra charge."""
    if not cached:
        return 0.0
    h = small_step["prompt"] + small_step["usage"]["output_tokens"]
    return h * card(large_model, h)["cache_write_5min_per_mtok"] / 1e6


def escalate(small: dict, large: dict, rule: str, fmt: str, cached: bool) -> dict:
    """Replay one task. Returns the step-cost sequence and the success bounds."""
    j = trigger_step(small, rule)
    s_ok, l_ok = int(small["success"]), int(large["success"])
    if j is None or j == len(small["steps"]) - 1:
        costs = [step_cost(s) for s in small["steps"]]
        return {"costs": costs, "fired": False, "lo": s_ok, "hi": s_ok, "handoff": 0.0}
    prefix = [step_cost(s) for s in small["steps"][: j + 1]]
    if fmt == "restart":
        rest = [step_cost(s) for s in large["steps"]]
        return {"costs": prefix + rest, "fired": True, "lo": l_ok, "hi": l_ok, "handoff": 0.0,
                "prefix": sum(prefix)}
    turn = small["steps"][j]["turn"]
    tail = [s for s in large["steps"] if s["turn"] >= turn]
    if not tail:  # large model finished this turn's work in fewer turns; nothing left to run
        tail = large["steps"][-1:]
    model = tail[0]["model"]
    kappa = handoff_write(small["steps"][j], model, cached)
    first = tail[0]
    reread = first["usage"]["cache_read_tokens"] * card(model, first["prompt"])["cache_read_per_mtok"] / 1e6
    rest = [step_cost(s) for s in tail]
    rest[0] += kappa - (reread if cached else 0.0)
    return {"costs": prefix + rest, "fired": True, "lo": s_ok, "hi": max(s_ok, l_ok), "handoff": kappa,
            "prefix": sum(prefix)}


# ---------------------------------------------------------------- summaries

def policy_row(results: list[dict], ids: list[str], large_ok: list[int]) -> dict:
    cost = [sum(r["costs"]) for r in results]
    lo, hi = [r["lo"] for r in results], [r["hi"] for r in results]
    n = len(ids)
    row = {"cost_per_task": statistics.mean(cost), "cost_per_task_ci95": boot_ci(cost),
           "success_lo": sum(lo) / n, "success_hi": sum(hi) / n,
           "success_q": [(sum(lo) + q * (sum(hi) - sum(lo))) / n for q in Q_HANDOFF],
           "cost_per_success_hi": sum(cost) / sum(hi) if sum(hi) else None,
           "cost_per_success_lo": sum(cost) / sum(lo) if sum(lo) else None,
           "fired": sum(r["fired"] for r in results),
           "handoff_usd_mean_when_fired": statistics.mean([r["handoff"] for r in results if r["fired"]]) if any(r["fired"] for r in results) else 0.0}
    if row["success_lo"] == row["success_hi"]:  # exact replay: report CI and paired test vs large
        row["wilson95"] = wilson(sum(hi), n)
        row["cost_per_success_ci95"] = boot_ratio_ci(cost, hi)
        b = sum(1 for a, z in zip(large_ok, hi) if a and not z)
        c = sum(1 for a, z in zip(large_ok, hi) if z and not a)
        row["mcnemar_vs_large"] = {"large_only": b, "policy_only": c, "p": mcnemar_exact(b, c)}
    return row


def static_row(tasks: dict, ids: list[str]) -> dict:
    cost = [sum(step_cost(s) for s in tasks[i]["steps"]) for i in ids]
    ok = [int(tasks[i]["success"]) for i in ids]
    return {"cost_per_task": statistics.mean(cost), "cost_per_task_ci95": boot_ci(cost),
            "success": sum(ok) / len(ids), "wilson95": wilson(sum(ok), len(ids)),
            "cost_per_success": sum(cost) / sum(ok), "cost_per_success_ci95": boot_ratio_ci(cost, ok)}


def trigger_quality(small: dict, ids: list[str], rule: str) -> dict:
    fired = {i for i in ids if trigger_step(small[i], rule) not in (None, len(small[i]["steps"]) - 1)}
    fails = {i for i in ids if not small[i]["success"]}
    return {"fired": len(fired), "small_failures": len(fails), "true_alarms": len(fired & fails),
            "recall": len(fired & fails) / len(fails), "precision": len(fired & fails) / len(fired) if fired else None,
            "fired_on_small_successes": len(fired - fails)}


def handoff_vs_position(small: dict, large: dict, ids: list[str], cached: bool) -> list[dict]:
    """RQ3: cold-write cost of a full handoff at every error step, against the cost of
    restarting the task on the large model."""
    rows = []
    for i in ids:
        n = len(small[i]["steps"])
        large_cost = sum(step_cost(s) for s in large[i]["steps"])
        for idx, s in enumerate(small[i]["steps"]):
            if s["n_errors"] > 0:
                k = handoff_write(s, large[i]["steps"][0]["model"], cached)
                rows.append({"task": i, "position": idx / max(1, n - 1), "history_tokens": s["prompt"] + s["usage"]["output_tokens"],
                             "handoff_usd": k, "share_of_large_task": k / large_cost})
    return rows


# ---------------------------------------------------------------- RQ5: overhead of the managing system

def overhead(small: dict, ids: list[str], reps: int = 200) -> dict:
    """Wall-clock cost of one Monitor-Analyze-Plan pass per step, measured by replaying
    every step through the same arithmetic the loop runs, against the recorded latency of
    the model call it guards. The loop makes no model calls, so it spends no tokens."""
    steps = [s for i in ids for s in small[i]["steps"]]
    t0 = time.perf_counter()
    for _ in range(reps):
        spent, errors, misses, prev = 0.0, 0, 0, 0.0
        for idx, s in enumerate(steps):
            c = step_cost(s, "recorded" if idx % 2 else "uncached")  # Monitor: price the step
            spent += c                                               # Knowledge: budget ledger
            errors += s["n_errors"] > 0                              # Analyze: trouble signal
            misses = misses + 1 if s["usage"]["cache_read_tokens"] == 0 else 0  # cache health
            _ = spent + prev > 0.05 or errors >= 1 or misses >= 1   # Plan: tactic guards
            prev = c
    per_step = (time.perf_counter() - t0) / (reps * len(steps))
    lat = statistics.median(s["latency_s"] for s in steps)
    return {"loop_seconds_per_step": per_step, "median_model_call_latency_s": lat,
            "overhead_share_of_call": per_step / lat, "llm_tokens_spent_by_loop": 0,
            "steps_timed": len(steps) * reps}


# ---------------------------------------------------------------- main

def main() -> None:
    runs = {label: load(log) for label, (log, _) in CONFIGS.items()}
    ids = sorted(set.intersection(*(set(t) for t in runs.values())))
    N: dict = {"n_tasks": len(ids), "q_handoff": Q_HANDOFF, "pairs": {}}
    for pair, (sl, ll) in PAIRS.items():
        small, large, cached = runs[sl], runs[ll], CONFIGS[sl][1]
        large_ok = [int(large[i]["success"]) for i in ids]
        P = {"static_small": static_row(small, ids), "static_large": static_row(large, ids)}
        # Oracle router: knows in advance whether the small model will succeed.
        oc = [sum(step_cost(s) for s in (small[i] if small[i]["success"] else large[i])["steps"]) for i in ids]
        ok = [int(small[i]["success"] or large[i]["success"]) for i in ids]
        P["oracle"] = {"cost_per_task": statistics.mean(oc), "success": sum(ok) / len(ids), "cost_per_success": sum(oc) / sum(ok)}
        P["policies"] = {}
        for rule in ("first_error", "second_error", "fixed_half"):
            for fmt in ("restart", "full"):
                res = [escalate(small[i], large[i], rule, fmt, cached) for i in ids]
                P["policies"][f"{rule}/{fmt}"] = policy_row(res, ids, large_ok)
        # Composite: escalation by restart on the first error, under a budget guard set at
        # the 90th percentile of the large model's successful-task cost.
        succ = sorted(sum(step_cost(s) for s in large[i]["steps"]) for i in ids if large[i]["success"])
        B = succ[int(0.9 * len(succ)) - 1]
        comp = []
        for i in ids:
            r = escalate(small[i], large[i], "first_error", "restart", cached)
            t = {"steps": [{"cost_usd": c} for c in r["costs"]], "success": bool(r["hi"])}
            c, s_ok, _ = guard(t, B, "predictive")
            comp.append({"costs": [c], "fired": r["fired"], "lo": int(s_ok), "hi": int(s_ok), "handoff": 0.0})
        P["composite_restart_guard_p90"] = {"budget_usd": B, **policy_row(comp, ids, large_ok)}
        # Tally's planner: on the first error, price both handoff formats with Knowledge
        # estimated leave-one-out from the other 49 tasks, and pick the lower expected cost
        # per success. The full-history format is credited with q = mean of Q_HANDOFF.
        tally, choices = [], {"restart": 0, "full": 0}
        q = sum(Q_HANDOFF) / 2
        for i in ids:
            others = [x for x in ids if x != i]
            C_l = statistics.mean(sum(step_cost(s) for s in large[x]["steps"]) for x in others)
            p_l = statistics.mean(large[x]["success"] for x in others)
            p_s = statistics.mean(small[x]["success"] for x in others)
            T = statistics.mean(large[x]["turns"] for x in others)
            j = trigger_step(small[i], "first_error")
            if j is None or j == len(small[i]["steps"]) - 1:
                tally.append(escalate(small[i], large[i], "first_error", "restart", cached))
                continue
            s = small[i]["steps"][j]
            kappa = handoff_write(s, large[i]["steps"][0]["model"], cached)
            est_full = (kappa + C_l * max(0.0, 1 - s["turn"] / T)) / (p_s + q * (p_l - p_s))
            est_restart = C_l / p_l
            fmt = "full" if est_full < est_restart else "restart"
            choices[fmt] += 1
            tally.append(escalate(small[i], large[i], "first_error", fmt, cached))
        P["tally_priced_choice"] = {"choices": choices, **policy_row(tally, ids, large_ok)}
        P["trigger_quality"] = {rule: trigger_quality(small, ids, rule) for rule in ("first_error", "second_error")}
        hv = handoff_vs_position(small, large, ids, cached)
        P["handoff_vs_position"] = {
            "n_error_steps": len(hv),
            "median_share_of_large_task": statistics.median(r["share_of_large_task"] for r in hv) if hv else 0.0,
            "max_share_of_large_task": max(r["share_of_large_task"] for r in hv) if hv else 0.0,
            "median_history_tokens": statistics.median(r["history_tokens"] for r in hv) if hv else 0,
            "points": hv}
        N["pairs"][pair] = P
    N["rq5_overhead"] = overhead(runs["small+cache"], ids)
    (ROOT / "results" / "escalation_numbers.json").write_text(json.dumps(N, indent=2) + "\n")

    for pair, P in N["pairs"].items():
        print(f"\n=== {pair} ===")
        for k in ("static_small", "static_large"):
            r = P[k]
            print(f"  {k:<28} succ {r['success']:.0%} [{r['wilson95'][0]:.0%},{r['wilson95'][1]:.0%}]  ${r['cost_per_task']:.4f}/task  ${r['cost_per_success']:.4f}/succ"
                  f" CI [{r['cost_per_success_ci95'][0]:.4f},{r['cost_per_success_ci95'][1]:.4f}]")
        o = P["oracle"]
        print(f"  {'oracle router':<28} succ {o['success']:.0%}  ${o['cost_per_task']:.4f}/task  ${o['cost_per_success']:.4f}/succ")
        for name, r in list(P["policies"].items()) + [("composite", P["composite_restart_guard_p90"]), ("tally " + str(P["tally_priced_choice"]["choices"]), P["tally_priced_choice"])]:
            succ = f"{r['success_lo']:.0%}" if r["success_lo"] == r["success_hi"] else \
                f"[{r['success_lo']:.0%},{r['success_hi']:.0%}] q:{r['success_q'][0]:.0%}-{r['success_q'][1]:.0%}"
            extra = f" CI [{r['cost_per_success_ci95'][0]:.4f},{r['cost_per_success_ci95'][1]:.4f}] McNemar {r['mcnemar_vs_large']}" if "mcnemar_vs_large" in r else ""
            print(f"  {name:<28} succ {succ}  ${r['cost_per_task']:.4f}/task  ${r['cost_per_success_hi']:.4f}/succ(best)"
                  f"  fired {r['fired']}  handoff ${r['handoff_usd_mean_when_fired']:.4f}{extra}")
        print("  triggers:", {k: {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in t.items()} for k, t in P["trigger_quality"].items()})
        h = P["handoff_vs_position"]
        print(f"  handoff at error steps: n={h['n_error_steps']} median {h['median_share_of_large_task']:.0%} max {h['max_share_of_large_task']:.0%} of a large task, median history {h['median_history_tokens']} tokens")
    print("\nRQ5 overhead:", N["rq5_overhead"])


if __name__ == "__main__":
    main()
