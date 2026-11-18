"""Workload 1: recorded BFCL v4 multi-turn runs. Fresh cuts for the AGENT 2027 paper.

python3 -I bfcl.py  -> prints every number and writes out/bfcl.json
Reads only the logs/ folder at the repository root (no API calls).

Table A: cost per successful task (all runs in the numerator), median and p90 cost of a
         successful run, share of spend on failed runs.
Table B: spend by step kind (tool-calling step, final-answer step, step whose tool call
         errored, repair step right after an errored step) and by token role (first read of
         the task prompt, carried context re-read, new input, output).
Table C: failure economics: failure mode, runs, median cost relative to the same
         configuration's median successful run, share of total spend, and share of a failed
         run's spend that lands on or after the turn where the checker first saw a divergence.
Spans:   per-step span checks (step costs sum to task cost; re-pricing reproduces cost).
"""
import json
import statistics as st
from collections import defaultdict
import os
from pathlib import Path

LOGS = Path(os.environ.get("AGENT_ARTIFACT_ROOT") or Path(__file__).resolve().parents[4]) / "logs"
PRICES = json.load(open(LOGS.parent / "prices.json"))["models"]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
STEP_CONFIGS = ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"]
ALL_CONFIGS = ["frontier", "frontier_cache", "frontier_low", "small", "small_cache_budget", "small_low"]


def load(kind, cfg):
    return [json.loads(l) for l in open(LOGS / f"{kind}_{cfg}.jsonl")]


def pct(xs, q):
    """Nearest-rank percentile (q in 0..1)."""
    s = sorted(xs)
    import math
    k = max(1, math.ceil(q * len(s)))
    return s[k - 1]


def rates(model, prompt_len):
    p = PRICES[model]
    if "tier_low" in p:
        p = p["tier_low"] if prompt_len <= 100_000 else p["tier_high"]
    return p


def reprice(s):
    u = s["usage"]
    P = u["input_tokens"] + u["cache_read_tokens"] + u["cache_write_tokens"]
    r = rates(s["model"], P)
    parts = {
        "in": u["input_tokens"] * r["input_per_mtok"] / 1e6,
        "cw": u["cache_write_tokens"] * r["cache_write_5min_per_mtok"] / 1e6,
        "cr": u["cache_read_tokens"] * r["cache_read_per_mtok"] / 1e6,
        "out": u["output_tokens"] * r["output_per_mtok"] / 1e6,
    }
    return parts, P


out = {"table_a": {}, "table_b": {}, "table_c": {}, "checks": {}, "concentration": {}}

# ---------- Table A ----------
for cfg in ALL_CONFIGS:
    T = load("tasks", cfg)
    succ = [t for t in T if t["success"]]
    total = sum(t["cost_usd"] for t in T)
    fail_usd = sum(t["cost_usd"] for t in T if not t["success"])
    sc = [t["cost_usd"] for t in succ]
    fc = [t["cost_usd"] for t in T if not t["success"]]
    out["table_a"][cfg] = {
        "runs": len(T), "successes": len(succ), "total_usd": round(total, 4),
        "usd_per_success": round(total / len(succ), 5),
        "succ_run_median": round(st.median(sc), 5), "succ_run_p90": round(pct(sc, 0.9), 5),
        "failed_run_median": round(st.median(fc), 5) if fc else None,
        "failed_share_of_spend": round(fail_usd / total, 3),
        "task_records_from_console_log": sum(1 for t in T if t.get("source", "").startswith("console_log")),
    }

# ---------- span checks and Table B ----------
for cfg in STEP_CONFIGS:
    S = load("steps", cfg)
    T = {t["task_id"]: t for t in load("tasks", cfg)}
    by_task = defaultdict(list)
    for s in S:
        by_task[s["task_id"]].append(s)
    max_sum_err, max_reprice_err = 0.0, 0.0
    kind_usd = defaultdict(float)
    role_usd = defaultdict(float)
    role_usd_billed = defaultdict(float)
    turn_usd = defaultdict(float)
    steps_n = defaultdict(int)
    total = 0.0
    task_cost = {}
    for tid, steps in by_task.items():
        steps.sort(key=lambda s: s["step"])
        max_sum_err = max(max_sum_err, abs(sum(s["cost_usd"] for s in steps) - T[tid]["cost_usd"]))
        task_cost[tid] = sum(s["cost_usd"] for s in steps)
        prevP = None
        prev_failed = False
        for s in steps:
            parts, P = reprice(s)
            c = sum(parts.values())
            max_reprice_err = max(max_reprice_err, abs(c - s["cost_usd"]))
            total += c
            # step kind (overlapping flags folded into one exclusive label)
            if s["kind"] == "failed":
                k = "error_step"
            elif prev_failed:
                k = "repair_step"
            elif s["kind"] == "act":
                k = "tool_call_step"
            else:
                k = "final_step"
            kind_usd[k] += c
            steps_n[k] += 1
            prev_failed = s["kind"] == "failed"
            # token roles: carried tokens are the previous step's prompt, which the current
            # prompt repeats; input dollars are split in proportion to token counts.
            in_usd = parts["in"] + parts["cw"] + parts["cr"]
            if prevP is None:
                role_usd["first_read"] += in_usd
            else:
                carried = min(prevP, P)
                role_usd["reread"] += in_usd * carried / P if P else 0
                role_usd["new_input"] += in_usd * (P - carried) / P if P else 0
            role_usd["output"] += parts["out"]
            # billed-rate split: carried tokens are attributed to cache-read tokens first, then
            # uncached input, then cache-write tokens; each at its own billed rate.
            u = s["usage"]
            R, I, Wt = u["cache_read_tokens"], u["input_tokens"], u["cache_write_tokens"]
            if prevP is None:
                role_usd_billed["first_read"] += in_usd
            else:
                carried = min(prevP, P)
                cr_ = min(R, carried); rem = carried - cr_
                ci_ = min(I, rem); rem -= ci_
                cw_ = min(Wt, rem)
                car_usd = (cr_ * parts["cr"] / R if R else 0) + (ci_ * parts["in"] / I if I else 0) + (cw_ * parts["cw"] / Wt if Wt else 0)
                role_usd_billed["reread"] += car_usd
                role_usd_billed["new_input"] += in_usd - car_usd
            role_usd_billed["output"] += parts["out"]
            prevP = P
            turn_usd[min(s["turn"], 4)] += c
    costs = sorted(task_cost.values(), reverse=True)
    top = costs[: max(1, len(costs) // 10)]
    out["checks"][cfg] = {"steps": len(S), "tasks_with_steps": len(by_task),
                          "max_abs_step_sum_minus_task_cost": max_sum_err,
                          "max_abs_reprice_minus_logged": max_reprice_err}
    out["table_b"][cfg] = {
        "total_usd": round(total, 4),
        "kind_share": {k: round(v / total, 3) for k, v in kind_usd.items()},
        "kind_steps": dict(steps_n),
        "role_share": {k: round(v / total, 3) for k, v in role_usd.items()},
        "role_share_billed_rate": {k: round(v / total, 3) for k, v in role_usd_billed.items()},
        "turn_share": {f"turn{k}{'+' if k == 4 else ''}": round(v / total, 3) for k, v in sorted(turn_usd.items())},
    }
    out["concentration"][cfg] = {"top10pct_tasks_share": round(sum(top) / sum(costs), 3), "n_top": len(top)}

# ---------- Table C ----------
FAM = {"frontier": "large", "frontier_cache": "large", "small": "small", "small_cache_budget": "small", "small_low": "small"}
modes = defaultdict(lambda: {"runs": 0, "ratios": [], "usd": defaultdict(float), "at_or_after": [], "after": [], "configs": set()})
fam_total = defaultdict(float)
for cfg in STEP_CONFIGS:
    T = load("tasks", cfg)
    med_s = st.median(t["cost_usd"] for t in T if t["success"])
    fam_total[FAM[cfg]] += sum(t["cost_usd"] for t in T)
    for t in T:
        labels = []
        if not t["success"]:
            ge = t["grade_error"] or "unknown"
            labels.append("fail_" + ge.split("@")[0])
        else:
            labels.append("success_clean" if not t["failed_steps"] else "success_after_tool_error")
        if t["budget_denied_calls"]:
            labels.append("hit_tool_call_cap")
        for lab in labels:
            m = modes[lab]
            m["runs"] += 1
            m["ratios"].append(t["cost_usd"] / med_s)
            m["usd"][FAM[cfg]] += t["cost_usd"]
            m["configs"].add(cfg)
            if not t["success"] and t["grade_error"] and "@turn" in t["grade_error"]:
                d = int(t["grade_error"].split("@turn")[1])
                ptc = t["per_turn_cost_usd"]
                m["at_or_after"].append(sum(ptc[d:]) / sum(ptc))
                m["after"].append(sum(ptc[d + 1:]) / sum(ptc))
allfail_after, allfail_usd_after, allfail_usd = [], 0.0, 0.0
for cfg in STEP_CONFIGS:
    for t in load("tasks", cfg):
        if not t["success"] and t["grade_error"] and "@turn" in t["grade_error"]:
            d = int(t["grade_error"].split("@turn")[1])
            ptc = t["per_turn_cost_usd"]
            allfail_usd_after += sum(ptc[d + 1:])
            allfail_usd += sum(ptc)
for lab, m in sorted(modes.items()):
    out["table_c"][lab] = {
        "runs": m["runs"], "configs": sorted(m["configs"]),
        "median_cost_vs_median_success": round(st.median(m["ratios"]), 2),
        "share_of_large_family_spend": round(m["usd"]["large"] / fam_total["large"], 3),
        "share_of_small_family_spend": round(m["usd"]["small"] / fam_total["small"], 3),
        "median_share_at_or_after_flagged_turn": round(st.median(m["at_or_after"]), 2) if m["at_or_after"] else None,
        "median_share_after_flagged_turn": round(st.median(m["after"]), 2) if m["after"] else None,
        "n_with_flagged_turn": len(m["after"]),
    }
out["table_c_family_total_usd"] = {k: round(v, 4) for k, v in fam_total.items()}
out["failed_runs_usd_after_flagged_turn_share"] = round(allfail_usd_after / allfail_usd, 3)

json.dump(out, open(OUT / "bfcl.json", "w"), indent=1, default=list)
print(json.dumps(out, indent=1, default=list))

# ---------- alert check: runs above the configuration's successful-run p90 (in-sample) ----------
alerts = {}
tot_flag = tot_flag_fail = tot_runs = tot_fail = 0
for cfg in STEP_CONFIGS:
    T = load("tasks", cfg)
    thr = pct([t["cost_usd"] for t in T if t["success"]], 0.9)
    flagged = [t for t in T if t["cost_usd"] > thr]
    nf = sum(1 for t in flagged if not t["success"])
    alerts[cfg] = {"threshold_usd": round(thr, 5), "flagged": len(flagged), "flagged_failed": nf,
                   "base_failed": sum(1 for t in T if not t["success"]), "runs": len(T),
                   "flagged_with_error_step": sum(1 for t in flagged if t["failed_steps"])}
    tot_flag += len(flagged); tot_flag_fail += nf; tot_runs += len(T); tot_fail += alerts[cfg]["base_failed"]
alerts["pooled"] = {"flagged": tot_flag, "flagged_failed": tot_flag_fail, "precision": round(tot_flag_fail / tot_flag, 3),
                    "base_failure_rate": round(tot_fail / tot_runs, 3), "runs": tot_runs, "failed": tot_fail,
                    "recall": round(tot_flag_fail / tot_fail, 3),
                    "flagged_with_error_step": sum(a["flagged_with_error_step"] for a in alerts.values())}
out["alerts"] = alerts
json.dump(out, open(OUT / "bfcl.json", "w"), indent=1, default=list)
print(json.dumps(alerts, indent=1))
