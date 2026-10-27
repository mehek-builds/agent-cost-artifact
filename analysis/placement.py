"""Small-tier-first placement policies, token footprint by tier, and the break-even energy ratio.

No API calls.  python3 analysis/placement.py  -> prints tables, writes results/placement.json

Footprint of a step = fresh input + cache writes + alpha * cache reads + beta * output.
rho* is the ratio of large-tier to small-tier energy per footprint token above which a policy
uses less energy per successful task than always using the large tier.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
MODES = {"cached": ("small_cache_budget", "frontier_cache"), "uncached": ("small", "frontier")}
WEIGHTS = [(1, 1), (0.1, 1), (0.1, 5)]
POLICIES = ["small", "large", "first_error_restart", "oracle"]


def load(log):
    steps, tasks = {}, {}
    for s in map(json.loads, open(LOGS / f"steps_{log}.jsonl")):
        steps.setdefault(s["task_id"], []).append(s)
    for t in map(json.loads, open(LOGS / f"tasks_{log}.jsonl")):
        tasks[t["task_id"]] = t
    for v in steps.values():
        v.sort(key=lambda s: s["step"])
    return steps, tasks


def footprint(steps, a, b):
    return sum(s["usage"]["input_tokens"] + s["usage"]["cache_write_tokens"]
               + a * s["usage"]["cache_read_tokens"] + b * s["usage"]["output_tokens"] for s in steps)


def place(policy, ss, ls, small_ok, large_ok):
    """Return (small-tier steps, large-tier steps, success, escalated)."""
    if policy == "small":
        return ss, [], small_ok, False
    if policy == "large":
        return [], ls, large_ok, False
    if policy == "oracle":
        return (ss, [], True, False) if small_ok else ([], ls, large_ok, True)
    # first_error_restart: run small until its first tool error, then restart the task on the large tier
    k = next((i for i, s in enumerate(ss) if s["n_errors"] > 0), None)
    if k is None:
        return ss, [], small_ok, False
    return ss[: k + 1], ls, large_ok, True


def run(mode, a, b):
    (Ss, St), (Ls, Lt) = (load(x) for x in MODES[mode])
    out = {}
    for pol in POLICIES:
        fs = fl = cost = lat = 0.0
        succ = esc = 0
        for tid in St:
            sp, lp, ok, e = place(pol, Ss[tid], Ls[tid], St[tid]["success"], Lt[tid]["success"])
            fs += footprint(sp, a, b)
            fl += footprint(lp, a, b)
            cost += sum(s["cost_usd"] for s in sp + lp)
            lat += sum(s["latency_s"] for s in sp + lp)
            succ += ok
            esc += e
        out[pol] = {"successes": succ, "escalations": esc, "usd_per_task": cost / len(St), "usd_per_success": cost / succ,
                    "latency_per_success_s": lat / succ, "footprint_small_per_task": fs / len(St),
                    "footprint_large_per_task": fl / len(St), "large_share": fl / (fs + fl)}
    L = out["large"]
    ref = L["footprint_large_per_task"] / L["successes"]
    for pol in ["small", "first_error_restart", "oracle"]:
        p = out[pol]
        d = ref - p["footprint_large_per_task"] / p["successes"]
        p["break_even_rho"] = (p["footprint_small_per_task"] / p["successes"]) / d if d > 0 else None
    return out


def main():
    res = {}
    for mode in MODES:
        for a, b in WEIGHTS:
            key = f"{mode}_alpha{a}_beta{b}"
            res[key] = r = run(mode, a, b)
            print(f"== {key}")
            for pol, p in r.items():
                rho = p.get("break_even_rho")
                print(f"  {pol:20s} succ {p['successes']:2d} esc {p['escalations']:2d} $/succ {p['usd_per_success']:.4f} "
                      f"lat/succ {p['latency_per_success_s']:.1f}s  large share {p['large_share']:.2f}"
                      + (f"  rho* {rho:.2f}" if rho else ""))
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results" / "placement.json", "w"), indent=1)


if __name__ == "__main__":
    main()
