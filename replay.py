#!/usr/bin/env python3
"""Trace replay of the managing system's budget guard (RQ1) and cache-health
monitor (RQ4) over recorded multi-turn agent runs.

No API calls. Every number is recomputed from logs/ and prices.json.
    python3 replay.py            # prints tables, writes results/replay_numbers.json

Fidelity: both tactics only stop a trace or change how its tokens are billed, so
the recorded model outputs stay valid and the replay is exact for cost. A task the
guard stops is counted as a failure.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
PRICES = json.loads((ROOT / "prices.json").read_text())["models"]
CONFIGS = {  # label -> (log name, cached)
    "large": ("frontier", False),
    "large+cache": ("frontier_cache", True),
    "small": ("small", False),
    "small+cache": ("small_cache_budget", True),
}
SEED, BOOT = 42, 10_000


# ---------------------------------------------------------------- loading and prices

def card(model: str, prompt_tokens: int) -> dict:
    m = PRICES[model]
    if "tier_low" in m:
        return m["tier_low"] if prompt_tokens <= 100_000 else m["tier_high"]
    return m


def load(log: str) -> dict:
    tasks = {t["task_id"]: {**t, "steps": []} for t in map(json.loads, open(LOGS / f"tasks_{log}.jsonl"))}
    for s in map(json.loads, open(LOGS / f"steps_{log}.jsonl")):
        u = s["usage"]
        s["prompt"] = u["input_tokens"] + u["cache_read_tokens"] + u["cache_write_tokens"]
        tasks[s["task_id"]]["steps"].append(s)
    for t in tasks.values():
        t["steps"].sort(key=lambda s: s["step"])
    return tasks


def step_cost(s: dict, billing: str = "recorded") -> float:
    """Cost of one step. billing: recorded | uncached | thrash (written, never read) | cold (one full write)."""
    if billing == "recorded":
        return s["cost_usd"]
    c, out = card(s["model"], s["prompt"]), s["usage"]["output_tokens"]
    rate = {"uncached": c["input_per_mtok"], "thrash": c["cache_write_5min_per_mtok"],
            "cold": c["cache_write_5min_per_mtok"]}[billing]
    return (s["prompt"] * rate + out * c["output_per_mtok"]) / 1e6


# ---------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_ci(xs: list[float], stat=statistics.mean) -> tuple[float, float]:
    rng = random.Random(SEED)
    vals = sorted(stat([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(BOOT))
    return vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT) - 1]


def boot_ratio_ci(cost: list[float], succ: list[int]) -> tuple[float, float]:
    """95% CI of cost per successful task, resampling tasks."""
    rng, n, vals = random.Random(SEED), len(cost), []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        k = sum(succ[i] for i in idx)
        if k:
            vals.append(sum(cost[i] for i in idx) / k)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def mcnemar_exact(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# ---------------------------------------------------------------- RQ1: budget guard

def guard(task: dict, budget: float, mode: str) -> tuple[float, bool, bool]:
    """Replay one task under a per-task budget.

    reactive:   Monitor reads the bill after each step; stop once spend >= budget.
    predictive: before each step, forecast its cost as the previous step's cost and
                stop if spend + forecast would exceed the budget.
    Returns (cost, success, violated) where violated means final spend > budget.
    """
    spent, prev = 0.0, None
    for s in task["steps"]:
        if mode == "predictive" and prev is not None and spent + prev > budget:
            return spent, False, False
        c = step_cost(s)
        spent += c
        prev = c
        if mode == "reactive" and spent >= budget and s is not task["steps"][-1]:
            return spent, False, spent > budget
    return spent, task["success"], spent > budget


def summarize(costs: list[float], succ: list[int]) -> dict:
    n, k = len(costs), sum(succ)
    return {"success": k / n, "successes": k, "wilson95": wilson(k, n),
            "cost_per_task": statistics.mean(costs), "cost_per_task_ci95": boot_ci(costs),
            "cost_per_success": sum(costs) / k if k else None,
            "cost_per_success_ci95": boot_ratio_ci(costs, succ) if k else None}


def rq1(runs: dict, ids: list[str]) -> dict:
    out = {}
    for label, tasks in runs.items():
        base_c = [sum(step_cost(s) for s in tasks[i]["steps"]) for i in ids]
        base_s = [int(tasks[i]["success"]) for i in ids]
        succ_costs = sorted(c for c, s in zip(base_c, base_s) if s)
        fail_costs = [c for c, s in zip(base_c, base_s) if not s]
        row = {"baseline": summarize(base_c, base_s),
               "median_cost_success": statistics.median(succ_costs),
               "median_cost_failure": statistics.median(fail_costs) if fail_costs else None,
               "budgets": []}
        # Budgets at percentiles of the cost of successful tasks: an operator who knows
        # what a normal successful task costs sets the budget from that distribution.
        for q in (0.50, 0.75, 0.90, 1.00):
            b = succ_costs[min(len(succ_costs) - 1, math.ceil(q * len(succ_costs)) - 1)]
            for mode in ("reactive", "predictive"):
                res = [guard(runs[label][i], b, mode) for i in ids]
                c, s, v = [r[0] for r in res], [int(r[1]) for r in res], [r[2] for r in res]
                lost = sum(1 for a, z in zip(base_s, s) if a and not z)
                over = [r[0] - b for r in res if r[2]]
                row["budgets"].append({
                    "quantile_of_success_cost": q, "budget_usd": b, "mode": mode,
                    **summarize(c, s),
                    "violations": sum(v), "violation_rate": sum(v) / len(ids),
                    "median_overshoot_usd": statistics.median(over) if over else 0.0,
                    "max_overshoot_usd": max(over) if over else 0.0,
                    "successes_lost": lost, "mcnemar_p_vs_baseline": mcnemar_exact(lost, 0),
                    "stopped_tasks": sum(1 for r, full in zip(res, base_c) if r[0] < full - 1e-12),
                    "saving_vs_baseline": 1 - statistics.mean(c) / statistics.mean(base_c),
                })
        out[label] = row
    return out


# ---------------------------------------------------------------- RQ4: cache-health monitor

def cache_fault(task: dict, fault: str, k: int | None) -> tuple[float, int | None]:
    """Replay one cached task with a cache fault present from its first step.

    fault "ignored": the cache marker is silently ignored (e.g. placed on a prefix below
        the provider's minimum cacheable length), so every prompt token bills as fresh input.
    fault "thrash": the cached prefix changes on every call, so every prompt is written
        to the cache and never read.
    k: the monitor raises an alarm after k consecutive post-warm-up steps that read
        nothing from the cache; None means no monitor. After the alarm the Execute phase
        repairs the marker: the next step pays one cold write, later steps bill as recorded.
    Returns (cost, step index of the alarm or None).
    """
    spent, misses, alarm, repaired_at = 0.0, 0, None, None
    for idx, s in enumerate(task["steps"]):
        if repaired_at is None:
            spent += step_cost(s, "uncached" if fault == "ignored" else "thrash")
            if idx > 0:                      # step 0 is warm-up: it may legitimately miss
                misses += 1
                if k is not None and misses >= k:
                    alarm, repaired_at = idx, idx + 1
        elif idx == repaired_at:
            spent += step_cost(s, "cold")
        else:
            spent += step_cost(s)
    return spent, alarm


def false_alarms(tasks: dict, k: int) -> int:
    """Alarms the monitor would raise on the healthy recorded traces."""
    n = 0
    for t in tasks.values():
        run = 0
        for idx, s in enumerate(t["steps"]):
            run = run + 1 if idx > 0 and s["usage"]["cache_read_tokens"] == 0 else 0
            if run >= k:
                n += 1
                break
    return n


def rq4(runs: dict, ids: list[str]) -> dict:
    out = {}
    for label in ("large+cache", "small+cache"):
        tasks = runs[label]
        healthy = [sum(step_cost(s) for s in tasks[i]["steps"]) for i in ids]
        row = {"healthy_cost_per_task": statistics.mean(healthy), "faults": {}}
        for fault in ("ignored", "thrash"):
            faulty = [cache_fault(tasks[i], fault, None)[0] for i in ids]
            f = {"no_monitor_cost_per_task": statistics.mean(faulty),
                 "no_monitor_x_healthy": statistics.mean(faulty) / statistics.mean(healthy),
                 "monitored": []}
            for k in (1, 2, 3):
                res = [cache_fault(tasks[i], fault, k) for i in ids]
                c = [r[0] for r in res]
                alarms = [r[1] for r in res if r[1] is not None]
                excess = statistics.mean(faulty) - statistics.mean(healthy)
                f["monitored"].append({
                    "k": k, "cost_per_task": statistics.mean(c),
                    "cost_per_task_ci95": boot_ci(c),
                    "x_healthy": statistics.mean(c) / statistics.mean(healthy),
                    "excess_recovered": (statistics.mean(faulty) - statistics.mean(c)) / excess,
                    "tasks_alarmed": len(alarms), "median_alarm_step": statistics.median(alarms) if alarms else None,
                    "false_alarms_on_healthy_traces": false_alarms(tasks, k),
                    # Fleet mode: the repair fixes the agent's configuration, so only the
                    # first task that meets the fault pays for it. Averaged over which task
                    # comes first.
                    "fleet_cost_per_task": statistics.mean(healthy)
                    + statistics.mean(a - h for a, h in zip(c, healthy)) / len(ids),
                })
            row["faults"][fault] = f
        out[label] = row
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    runs = {label: load(log) for label, (log, _) in CONFIGS.items()}
    ids = sorted(set.intersection(*(set(t) for t in runs.values())))
    check = {label: max(abs(sum(s["cost_usd"] for s in t[i]["steps"]) - t[i]["cost_usd"]) for i in ids)
             for label, t in runs.items()}
    rp = {label: max(abs(sum(step_cost(s, "uncached") for s in t[i]["steps"]) - t[i]["cost_usd"]) for i in ids)
          for label, t in runs.items() if not CONFIGS[label][1]}
    N = {"n_tasks": len(ids), "check_steps_sum_to_task_cost_max_abs_err": check,
         "check_uncached_reprice_max_abs_err": rp, "rq1_budget_guard": rq1(runs, ids), "rq4_cache_monitor": rq4(runs, ids)}
    (ROOT / "results" / "replay_numbers.json").write_text(json.dumps(N, indent=2) + "\n")

    print(f"tasks: {len(ids)}   step sums vs task cost (max abs err): {check}")
    print(f"uncached re-pricing vs recorded (max abs err): {rp}\n")
    print("RQ1 budget guard")
    for label, row in N["rq1_budget_guard"].items():
        b = row["baseline"]
        print(f"  {label:<12} baseline  succ {b['success']:.0%}  ${b['cost_per_task']:.4f}/task  ${b['cost_per_success']:.4f}/succ"
              f"  median cost success ${row['median_cost_success']:.4f} failure ${row['median_cost_failure'] or 0:.4f}")
        for r in row["budgets"]:
            print(f"    B=p{int(r['quantile_of_success_cost']*100):<3} ${r['budget_usd']:.4f} {r['mode']:<10} succ {r['success']:.0%}"
                  f" (-{r['successes_lost']}, p={r['mcnemar_p_vs_baseline']:.3f})  ${r['cost_per_task']:.4f}/task"
                  f" ({r['saving_vs_baseline']:+.0%})  ${r['cost_per_success']:.4f}/succ  violations {r['violations']}"
                  f" max over ${r['max_overshoot_usd']:.4f}")
    print("\nRQ4 cache-health monitor")
    for label, row in N["rq4_cache_monitor"].items():
        print(f"  {label:<12} healthy ${row['healthy_cost_per_task']:.4f}/task")
        for fault, f in row["faults"].items():
            print(f"    {fault:<8} no monitor ${f['no_monitor_cost_per_task']:.4f} ({f['no_monitor_x_healthy']:.1f}x)")
            for m in f["monitored"]:
                print(f"      k={m['k']} ${m['cost_per_task']:.4f} ({m['x_healthy']:.2f}x) recovered {m['excess_recovered']:.0%}"
                      f"  alarm step {m['median_alarm_step']}  false alarms {m['false_alarms_on_healthy_traces']}")


if __name__ == "__main__":
    main()
