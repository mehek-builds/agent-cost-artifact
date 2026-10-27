"""Per-step token and dollar accounting of the four recorded configurations.

No API calls.  python3 analysis/accounting.py  -> prints tables, writes results/accounting.json

Covers: tokens by billing category, input share, re-sent share of input, cache-read share,
cost per task and per success, spend on failed tasks, latency per success, Wilson intervals,
prompt growth by step, dollar share by billing category, and per-task cost spread.
"""
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
PRICES = json.load(open(ROOT / "prices.json"))
CONFIGS = {"small": "small", "small+cache": "small_cache_budget", "large": "frontier", "large+cache": "frontier_cache"}
KINDS = ["input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens"]


def load(log):
    steps = [json.loads(l) for l in open(LOGS / f"steps_{log}.jsonl")]
    tasks = [json.loads(l) for l in open(LOGS / f"tasks_{log}.jsonl")]
    return steps, tasks


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def prompt(u):
    return u["input_tokens"] + u["cache_write_tokens"] + u["cache_read_tokens"]


def dollars_by_kind(steps):
    """Re-price each step by billing category, using the per-request tier for tiered models."""
    out = dict.fromkeys(KINDS, 0.0)
    for s in steps:
        m, u = PRICES["models"][s["model"]], s["usage"]
        if "tier_low" in m:
            m = m["tier_low"] if prompt(u) <= 100_000 else m["tier_high"]
        out["input_tokens"] += u["input_tokens"] * m["input_per_mtok"] / 1e6
        out["cache_write_tokens"] += u["cache_write_tokens"] * m["cache_write_5min_per_mtok"] / 1e6
        out["cache_read_tokens"] += u["cache_read_tokens"] * m["cache_read_per_mtok"] / 1e6
        out["output_tokens"] += u["output_tokens"] * m["output_per_mtok"] / 1e6
    return out


def summarize(steps, tasks):
    n = len(tasks)
    succ = sum(t["success"] for t in tasks)
    tok = {k: sum(s["usage"][k] for s in steps) for k in KINDS}
    total = sum(tok.values())
    inp = total - tok["output_tokens"]
    first = {}
    for s in steps:
        if s["step"] == 0:
            first[s["task_id"]] = prompt(s["usage"])
    lat = [s["latency_s"] for s in steps]
    cost = sum(t["cost_usd"] for t in tasks)
    by_step = {}
    for s in steps:
        by_step.setdefault(s["step"], []).append((prompt(s["usage"]), first[s["task_id"]]))
    dk = dollars_by_kind(steps)
    succ_costs = [t["cost_usd"] for t in tasks if t["success"]]
    all_costs = [t["cost_usd"] for t in tasks]
    return {
        "tasks": n,
        "successes": succ,
        "success_wilson95": wilson(succ, n),
        "requests": len(steps),
        "requests_per_task": len(steps) / n,
        "tokens_per_task": total / n,
        "tokens_per_success": total / succ,
        "output_tokens_per_task": tok["output_tokens"] / n,
        "input_share_of_tokens": inp / total,
        "input_mix": {k: tok[k] / inp for k in KINDS[:3]},
        "first_request_share_of_input": sum(first.values()) / inp,
        "resent_share_of_input": 1 - sum(first.values()) / inp,
        "median_step_latency_s": st.median(lat),
        "latency_per_success_s": sum(lat) / succ,
        "usd_per_task": cost / n,
        "usd_per_success": cost / succ,
        "failed_step_cost_share": sum(t["failed_step_cost_usd"] for t in tasks) / cost,
        "failed_task_cost_share": sum(t["cost_usd"] for t in tasks if not t["success"]) / cost,
        "dollar_share_by_kind": {k: v / sum(dk.values()) for k, v in dk.items()},
        "max_over_median_success_cost": max(succ_costs) / st.median(succ_costs),
        "max_over_min_task_cost": max(all_costs) / min(all_costs),
        "median_cost_success": st.median(succ_costs),
        "median_cost_failure": st.median([t["cost_usd"] for t in tasks if not t["success"]]),
        "prompt_by_step": {
            i: {"n": len(v), "mean_prompt": sum(a for a, _ in v) / len(v), "mean_first_prompt": sum(b for _, b in v) / len(v)}
            for i, v in sorted(by_step.items())
        },
    }


def main():
    res = {name: summarize(*load(log)) for name, log in CONFIGS.items()}
    for name, r in res.items():
        lo, hi = r["success_wilson95"]
        print(f"{name:12s} succ {r['successes']}/50 [{lo:.2f},{hi:.2f}]  tokens/task {r['tokens_per_task']:,.0f}  "
              f"tokens/succ {r['tokens_per_success']:,.0f}  input {r['input_share_of_tokens']:.3f}  "
              f"re-sent {r['resent_share_of_input']:.3f}  cache-read {r['input_mix']['cache_read_tokens']:.3f}")
        print(f"{'':12s} $/task {r['usd_per_task']:.4f}  $/succ {r['usd_per_success']:.4f}  latency/succ {r['latency_per_success_s']:.1f}s  "
              f"failed-task share {r['failed_task_cost_share']:.2f}  $-share {({k.replace('_tokens', ''): round(v, 3) for k, v in r['dollar_share_by_kind'].items()})}")
    for a, b in [("small", "small+cache"), ("large", "large+cache")]:
        print(f"caching saves {1 - res[b]['usd_per_task'] / res[a]['usd_per_task']:.0%} per task ({a})")
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results" / "accounting.json", "w"), indent=1)


if __name__ == "__main__":
    main()
