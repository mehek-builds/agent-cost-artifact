"""Recompute every number used in the BPOD P270 camera-ready poster paper from the
recorded BFCL v4 runs in this repository (logs/, prices.json).
Reads only; makes no API calls. Writes analysis/recompute.json and prints each number
with a tag that evidence-ledger.md cites."""
import json, math, glob, os, statistics
from collections import defaultdict
A = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))  # repository root
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recompute.json")
P = json.load(open(f"{A}/prices.json"))["models"]
res = {}

def say(tag, val):
    res[tag] = val
    print(f"{tag} = {val}")

def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-h), 100*(c+h))

def load(p):
    return [json.loads(l) for l in open(p)]

# per-category dollars for one step
def cat_cost(s):
    u = s["usage"]; m = s["model"]
    if m == "claude-haiku-5-5":
        prompt = u["input_tokens"] + u["cache_read_tokens"] + u["cache_write_tokens"]
        r = P[m]["tier_low"] if prompt <= 100000 else P[m]["tier_high"]
    else:
        r = P[m]
    return {"in": u["input_tokens"]*r["input_per_mtok"]/1e6,
            "cw": u["cache_write_tokens"]*r["cache_write_5min_per_mtok"]/1e6,
            "cr": u["cache_read_tokens"]*r["cache_read_per_mtok"]/1e6,
            "out": u["output_tokens"]*r["output_per_mtok"]/1e6}

# ---------------- single-turn ----------------
st = defaultdict(list)
for f in sorted(glob.glob(f"{A}/logs/single_turn/tasks_*.jsonl")):
    cfg = os.path.basename(f)[6:].split("__")[0]
    st[cfg].append(load(f))
for cfg, runs in st.items():
    rows = [r for run in runs for r in run]
    k = sum(r["success"] for r in rows); n = len(rows)
    tot = sum(r["cost_usd"] for r in rows)
    lo, hi = wilson(k, n)
    per_run = [statistics.mean(r["cost_usd"] for r in run) for run in runs]
    cv = statistics.pstdev(per_run)/statistics.mean(per_run)*100 if len(per_run) > 1 else 0.0
    say(f"st.{cfg}.runs", len(runs))
    say(f"st.{cfg}.success_pct", round(100*k/n, 1))
    say(f"st.{cfg}.wilson", (round(lo), round(hi)))
    say(f"st.{cfg}.cost_per_task", round(tot/n, 5))
    say(f"st.{cfg}.cost_per_success", round(tot/k, 5))
    say(f"st.{cfg}.cv_pct", round(cv, 2))
    res[f"_st.{cfg}.cpt_raw"] = tot/n
g = lambda t: res[f"_st.{t}.cpt_raw"]
say("st.small_vs_frontier_cost_reduction_pct", round(100*(1-g("small")/g("frontier")), 1))
say("st.frontier_over_small_cost_per_success_x", round(res["st.frontier.cost_per_success"]/res["st.small.cost_per_success"], 1))
say("st.cache_vs_small_cost_change_pct", round(100*(g("small_cache_budget")/g("small")-1), 1))
say("st.frontier_low_vs_frontier_cost_change_pct", round(100*(g("frontier_low")/g("frontier")-1), 1))
say("st.small_low_vs_small_cost_change_pct", round(100*(g("small_low")/g("small")-1), 1))

# ---------------- multi-turn ----------------
CFG = ["frontier", "frontier_cache", "small", "small_cache_budget", "frontier_low", "small_low"]
T = {c: load(f"{A}/logs/tasks_{c}.jsonl") for c in CFG}
S = {c: load(f"{A}/logs/steps_{c}.jsonl") for c in CFG}
say("mt.turns_min", min(r["turns"] for c in CFG for r in T[c])); say("mt.turns_max", max(r["turns"] for c in CFG for r in T[c]))
mt = {}
for c in CFG:
    rows = T[c]; n = len(rows); k = sum(r["success"] for r in rows)
    tot = sum(r["cost_usd"] for r in rows)
    lo, hi = wilson(k, n)
    steps = statistics.mean(r["llm_steps"] for r in rows)
    fsteps = sum(r["failed_steps"] for r in rows); allsteps = sum(r["llm_steps"] for r in rows)
    say(f"mt.{c}.n", n); say(f"mt.{c}.successes", k)
    say(f"mt.{c}.success_pct", round(100*k/n, 1))
    say(f"mt.{c}.wilson", (round(lo), round(hi)))
    say(f"mt.{c}.cost_per_task", round(tot/n, 5))
    say(f"mt.{c}.cost_per_success", round(tot/k, 5))
    say(f"mt.{c}.mean_steps", round(steps, 1))
    say(f"mt.{c}.f", round(fsteps/allsteps, 3))
    if c != "frontier_low":
        say(f"mt.{c}.retry_share_pct", round(100*sum(r["failed_step_cost_usd"] for r in rows)/tot, 1))
    say(f"mt.{c}.tasks_with_failed_step_pct", round(100*sum(r["failed_steps"] > 0 for r in rows)/n))
    fc = [r["cost_usd"] for r in rows if not r["success"]]; sc = [r["cost_usd"] for r in rows if r["success"]]
    say(f"mt.{c}.fail_over_success_cost_x", round(statistics.mean(fc)/statistics.mean(sc), 2))
    for key, lab in (("llm_steps", "steps"), ("turns", "turns")):
        say(f"mt.{c}.mean_{lab}_failed", round(statistics.mean(r[key] for r in rows if not r["success"]), 1))
        say(f"mt.{c}.mean_{lab}_success", round(statistics.mean(r[key] for r in rows if r["success"]), 1))
    costs = [r["cost_usd"] for r in rows]
    say(f"mt.{c}.max_over_min_task_cost_x", round(max(costs)/min(costs), 1))
    mt[c] = tot/n
src = {}
for r in T["frontier_low"]:
    src[r.get("source") is None] = src.get(r.get("source") is None, 0) + 1
say("mt.frontier_low.records_from_console_log", src.get(False, 0))
say("mt.frontier_low.steps_logged", len(S["frontier_low"]))

say("mt.cache_saving_sonnet_pct", round(100*(1-mt["frontier_cache"]/mt["frontier"]), 1))
say("mt.cache_saving_haiku_pct", round(100*(1-mt["small_cache_budget"]/mt["small"]), 1))
say("mt.effort_change_sonnet_pct", round(100*(mt["frontier_low"]/mt["frontier"]-1), 2))
say("mt.effort_change_haiku_pct", round(100*(mt["small_low"]/mt["small"]-1), 1))
say("mt.sonnet_multi_over_single_cost_x", round(mt["frontier"]/g("frontier")))
say("mt.small_over_frontier_cost_per_task_x", round(mt["frontier"]/mt["small"], 1))
say("mt.budget_denied_calls_small_cache_budget", sum(r["budget_denied_calls"] for r in T["small_cache_budget"]))

# billing-category shares (complete step logs only), and final-reply share
for c in ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"]:
    agg = defaultdict(float); fin = 0.0; tot = 0.0; recost = 0.0
    for s in S[c]:
        cc = cat_cost(s)
        for k2, v in cc.items(): agg[k2] += v
        tot += s["cost_usd"]; recost += sum(cc.values())
        if s["kind"] == "final": fin += s["cost_usd"]
    say(f"mt.{c}.reprice_matches_logged_pct", round(100*recost/tot, 2))
    for k2 in ["in", "cw", "cr", "out"]:
        say(f"mt.{c}.share_{k2}_pct", round(100*agg[k2]/recost, 1))
    say(f"mt.{c}.share_all_input_pct", round(100*(agg["in"]+agg["cw"]+agg["cr"])/recost, 1))
    say(f"mt.{c}.final_reply_share_pct", round(100*fin/tot, 1))

# paired comparison Sonnet vs Haiku (exact McNemar)
a = {r["task_id"]: r["success"] for r in T["frontier"]}; b = {r["task_id"]: r["success"] for r in T["small"]}
s_only = sum(a[t] and not b[t] for t in a); h_only = sum(b[t] and not a[t] for t in a)
nn = s_only + h_only; kk = min(s_only, h_only)
p = min(1.0, 2*sum(math.comb(nn, i) for i in range(kk+1))/2**nn)
say("mt.sonnet_only_solved", s_only); say("mt.haiku_only_solved", h_only); say("mt.mcnemar_exact_p", round(p, 4))

# trace example: multi_turn_base_11 under Sonnet (uncached)
tr = [s for s in S["frontier"] if s["task_id"] == "multi_turn_base_11"]
trow = [r for r in T["frontier"] if r["task_id"] == "multi_turn_base_11"][0]
rows_out = []
for s in tr:
    rows_out.append((s["turn"]+1, s["step"]+1, s["kind"], s["usage"]["input_tokens"], s["usage"]["output_tokens"], round(s["cost_usd"], 4)))
say("trace.rows", rows_out)
say("trace.task_cost", round(trow["cost_usd"], 4))
say("trace.success", trow["success"])
failed = sum(s["cost_usd"] for s in tr if s["kind"] == "failed")
say("trace.failed_step_share_pct", round(100*failed/trow["cost_usd"]))
fin = sum(s["cost_usd"] for s in tr if s["kind"] == "final")
say("trace.reply_steps_share_pct", round(100*fin/trow["cost_usd"]))
say("trace.f", f"{sum(s['kind']=='failed' for s in tr)}/{len(tr)}")

# frontier figure points (multi-turn)
say("fig.frontier_points", {c: (round(mt[c], 5), res[f"mt.{c}.success_pct"], res[f"mt.{c}.wilson"]) for c in CFG})

# quality bar: 90% of best success
best = max(res[f"mt.{c}.success_pct"] for c in CFG)
bar = 0.9*best
say("mt.quality_bar_pct", round(bar, 1))
ok = [c for c in CFG if res[f"mt.{c}.success_pct"] >= bar]
say("mt.configs_meeting_bar", ok)
say("mt.selected_cheapest_meeting_bar", min(ok, key=lambda c: mt[c]))
say("mt.single_turn_pick_gap_pts", round(res["mt.frontier_cache.success_pct"]-res["mt.small_low.success_pct"], 1))

# single-turn selection under the same rule (cheapest cost per success among configs within 90% of best success)
stc = ["frontier", "frontier_low", "small", "small_cache_budget", "small_low"]
sbest = max(res[f"st.{c}.success_pct"] for c in stc)
sok = [c for c in stc if res[f"st.{c}.success_pct"] >= 0.9*sbest]
say("st.quality_bar_pct", round(0.9*sbest, 1))
say("st.configs_meeting_bar", sok)
say("st.selected", min(sok, key=lambda c: res[f"_st.{c}.cpt_raw"]))  # cheapest cost per task, as stated in the text
say("st.selected_by_cost_per_success", min(sok, key=lambda c: res[f"_st.{c}.cpt_raw"] / (res[f"st.{c}.success_pct"]/100)))
say("mt.cache_saving_haiku_raw_pct", round(100*(1-mt["small_cache_budget"]/mt["small"]), 3))
# sample (n-1) coefficient of variation of per-run mean single-turn cost
for cfg, runs in st.items():
    pr = [statistics.mean(r["cost_usd"] for r in run) for run in runs]
    say(f"st.{cfg}.cv_sample_pct", round(statistics.stdev(pr)/statistics.mean(pr)*100, 2))


# ---------- single-turn intervals clustered by task (bootstrap over the 150 tasks) ----------
import random
for cfg, runs in st.items():
    per_task = defaultdict(list)
    for run in runs:
        for r in run: per_task[r["task_id"]].append(1.0 if r["success"] else 0.0)
    means = [statistics.mean(v) for v in per_task.values()]
    rng = random.Random(42)
    nT = len(means)
    boots = sorted(statistics.mean(rng.choices(means, k=nT)) for _ in range(10000))
    say(f"st.{cfg}.tasks", nT)
    say(f"st.{cfg}.boot_task_clustered_ci", (round(100*boots[249]), round(100*boots[9749])))
    k150 = round(sum(means))  # success count on the 150 tasks, run-averaged
    say(f"st.{cfg}.wilson_n150", tuple(round(x) for x in wilson(sum(means), nT)))

# ---------- Table I: per-step share of task cost for the trace ----------
say("trace.step_share_pct", [round(100*s["cost_usd"]/trow["cost_usd"]) for s in tr])

# ---------- Haiku rate tier: largest prompt in any logged Haiku request ----------
for c in ["small", "small_cache_budget", "small_low"]:
    mp = max(s["usage"]["input_tokens"]+s["usage"]["cache_read_tokens"]+s["usage"]["cache_write_tokens"] for s in S[c])
    say(f"mt.{c}.max_prompt_tokens", mp)
    say(f"mt.{c}.requests_over_100k", sum((s["usage"]["input_tokens"]+s["usage"]["cache_read_tokens"]+s["usage"]["cache_write_tokens"]) > 100000 for s in S[c]))

# ---------- single-turn: one request per task, and the costliest single-turn Haiku task ----------
say("st.all_llm_calls_equal_1", all(r.get("llm_calls") == 1 for c in st for run in st[c] for r in run))
for c in ["small", "small_cache_budget", "small_low"]:
    say(f"st.{c}.max_task_cost_usd", max(r["cost_usd"] for run in st[c] for r in run))

# ---------- steps: requests per task, and model requests per task by config ----------
say("mt.mean_steps_all", {c: round(statistics.mean(r["llm_steps"] for r in T[c]), 2) for c in CFG})

# ---------- failure grading categories (counts over the six multi-turn configs) ----------
from collections import Counter
gc = Counter((r["grade_error"] or "ok_or_empty").split("@")[0] for c in CFG for r in T[c] if not r["success"])
say("mt.failed_task_grade_kinds_all_configs", dict(gc))

# ---------- single-turn category split (first run file of config 'frontier') ----------
_f = sorted(glob.glob(f"{A}/logs/single_turn/tasks_frontier__*.jsonl"))[0]
say("st.category_counts", dict(Counter(json.loads(l)["category"] for l in open(_f))))

# ---------- round 3: paired tests and reply-step counts ----------
def mcnemar(x, y):
    a_ = {r["task_id"]: r["success"] for r in T[x]}; b_ = {r["task_id"]: r["success"] for r in T[y]}
    o = sum(a_[t] and not b_[t] for t in a_); q = sum(b_[t] and not a_[t] for t in a_)
    n_ = o + q; k_ = min(o, q)
    pp = 1.0 if n_ == 0 else min(1.0, 2*sum(math.comb(n_, i) for i in range(k_+1))/2**n_)
    return o, q, round(pp, 4)
say("mt.mcnemar_cache_vs_uncached_sonnet(d_only,a_only,p)", mcnemar("frontier_cache", "frontier"))
say("mt.mcnemar_c_vs_b(c_only,b_only,p)", mcnemar("small_cache_budget", "small"))
_pt = {}
for cfg in ("frontier", "small"):
    pt = defaultdict(list)
    for run in st[cfg]:
        for r in run: pt[r["task_id"]].append(1.0 if r["success"] else 0.0)
    _pt[cfg] = {t: statistics.mean(v) for t, v in pt.items()}
_ids = sorted(_pt["frontier"])
_d = [_pt["small"][t] - _pt["frontier"][t] for t in _ids]
_rng = random.Random(42)
_bs = sorted(statistics.mean(_rng.choices(_d, k=len(_d))) for _ in range(10000))
say("st.paired_diff_haiku_minus_sonnet_pts", round(100*statistics.mean(_d), 2))
say("st.paired_diff_boot_ci_pts", (round(100*_bs[249], 2), round(100*_bs[9749], 2)))
_fs = [s for s in S["frontier"]]
say("mt.frontier.final_steps", sum(s["kind"] == "final" for s in _fs))
say("mt.frontier.total_steps", len(_fs))
say("mt.frontier.final_step_share_pct", round(100*sum(s["kind"] == "final" for s in _fs)/len(_fs), 1))
say("trace.reply_steps_cost_usd", round(sum(s["cost_usd"] for s in tr if s["kind"] == "final"), 4))
say("trace.reply_steps_cost_usd_unrounded", sum(s["cost_usd"] for s in tr if s["kind"] == "final"))
say("fig.frontier_true_costs", {c: round(mt[c], 5) for c in CFG})

# ---------- round 5: task-bootstrap intervals for cost contrasts (10,000 resamples, seed 42) ----------
def ratio_boot(c):
    rows = T[c]; rng = random.Random(42); out = []
    for _ in range(10000):
        smp = rng.choices(rows, k=len(rows))
        fc = [r["cost_usd"] for r in smp if not r["success"]]; sc = [r["cost_usd"] for r in smp if r["success"]]
        if fc and sc: out.append(statistics.mean(fc)/statistics.mean(sc))
    out.sort(); m = len(out)
    return round(out[int(0.025*m)], 2), round(out[int(0.975*m)-1], 2)
for c in ["frontier", "small"]:
    say(f"mt.{c}.failed_tasks", sum(not r["success"] for r in T[c]))
    say(f"mt.{c}.fail_over_success_boot_ci", ratio_boot(c))
for c in ["frontier_cache", "small_cache_budget", "small_low", "frontier_low"]:
    say(f"mt.{c}.failed_tasks", sum(not r["success"] for r in T[c]))
    say(f"mt.{c}.fail_over_success_boot_ci", ratio_boot(c))
def saving_boot(base, alt):
    b = {r["task_id"]: r["cost_usd"] for r in T[base]}; a_ = {r["task_id"]: r["cost_usd"] for r in T[alt]}
    ids = sorted(b); rng = random.Random(42); out = []
    for _ in range(10000):
        smp = rng.choices(ids, k=len(ids))
        out.append(100*(1 - sum(a_[t] for t in smp)/sum(b[t] for t in smp)))
    out.sort()
    return round(out[249], 1), round(out[9749], 1)
say("mt.cache_saving_sonnet_boot_ci_pct", saving_boot("frontier", "frontier_cache"))
say("mt.cache_saving_haiku_boot_ci_pct", saving_boot("small", "small_cache_budget"))

# ---------- round 1 (2026-11-17 revision): task-bootstrap intervals for multi-turn cost per task and cost per success ----------
def cost_boot(c):
    rows = T[c]; rng = random.Random(42); cpt = []; cps = []
    for _ in range(10000):
        smp = rng.choices(rows, k=len(rows))
        tot = sum(r["cost_usd"] for r in smp); k_ = sum(1 for r in smp if r["success"])
        cpt.append(tot/len(smp))
        if k_: cps.append(tot/k_)
    cpt.sort(); cps.sort()
    q = lambda v, x: v[int(x*len(v))] if x < 1 else v[-1]
    return (round(q(cpt, 0.025), 5), round(q(cpt, 0.975), 5)), (round(q(cps, 0.025), 5), round(q(cps, 0.975), 5))
for c in CFG:
    a_, b_ = cost_boot(c)
    say(f"mt.{c}.cost_per_task_boot_ci", a_)
    say(f"mt.{c}.cost_per_success_boot_ci", b_)
def change_boot(base, alt):
    b = {r["task_id"]: r["cost_usd"] for r in T[base]}; a_ = {r["task_id"]: r["cost_usd"] for r in T[alt]}
    ids = sorted(b); rng = random.Random(42); out = []
    for _ in range(10000):
        smp = rng.choices(ids, k=len(ids))
        out.append(100*(sum(a_[t] for t in smp)/sum(b[t] for t in smp) - 1))
    out.sort()
    return round(out[249], 1), round(out[9749], 1)
say("mt.effort_change_sonnet_boot_ci_pct", change_boot("frontier", "frontier_low"))
say("mt.effort_change_haiku_boot_ci_pct", change_boot("small", "small_low"))

json.dump({k: v for k, v in res.items() if not k.startswith("_")}, open(OUT, "w"), indent=1, default=str)
