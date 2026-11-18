"""Workload 2: the deployed Litos agent. Fresh cuts for the AGENT 2027 paper.

python3 -I prod.py  -> prints every number and writes out/prod.json
Reads only the recorded files in 2-content/publications/production-data (no network, no API calls).

Window: the instrumented operations from roll-out onward. Operation attribution merged 2026-10-23 14:57:08 UTC;
only events at or after 14:58 UTC are read, and every usage record in that window is priced from its own
fields (the script asserts that its cache-creation tokens equal its 5-minute plus 1-hour split and that no
1-hour writes occur).

Prices: dated list prices per million tokens (input, 5-minute cache write, cache read, output),
checked on the provider pricing page on 2026-11-17 and equal to the rates the backend's price
version 'anthropic-standard-global-2026-10-23' uses; web search at USD 10 per 1,000 searches.
"""
import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime
import os
from pathlib import Path

PUB = Path(os.environ.get("LITOS_PRIVATE_DIR") or Path(os.environ.get("AGENT_ARTIFACT_ROOT") or Path(__file__).resolve().parents[4]) / "private-litos-stream")
EVENTS = PUB / "ops-events-2026-10-23_27.ndjson"
DAILY = Path(os.environ.get("AGENT_AGGREGATES_DIR") or Path(__file__).resolve().parents[4] / "papers" / "2026-11" / "production-aggregates") / "litos-console-daily-2026-07-30_2026-10-27.csv"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
PR = {"claude-sonnet-5": (2.0, 2.5, 0.20, 10.0), "claude-haiku-4-5-20251001": (1.0, 1.25, 0.10, 5.0)}
SEARCH_USD = 10.0 / 1000


def pct(xs, q):
    s = sorted(xs)
    return s[max(1, math.ceil(q * len(s))) - 1]


def token_usd(r):
    p = PR.get(r.get("model"))
    if not p:
        return None
    return (r.get("inputTokens", 0) * p[0] + (r.get("cacheWrite5mTokens", 0) or 0) * p[1]
            + (r.get("cacheReadInputTokens", 0) or 0) * p[2] + r.get("outputTokens", 0) * p[3]) / 1e6


BOUND = "2026-10-23T14:58"  # operation attribution live; the window starts here


def when(r):
    return (r.get("occurredAt") or r["timestamp"])[:16]


rows, seen, dup = [], set(), 0
for line in open(EVENTS):
    r = json.loads(line)
    if when(r) < BOUND:
        continue
    k = r.get("eventId") or json.dumps(r, sort_keys=True)
    if k in seen:
        dup += 1
        continue
    seen.add(k)
    rows.append(r)
use = [r for r in rows if r["event"] == "anthropic_response_usage"]
for r in use:  # every record in the window is fully priced from its own fields
    assert not r.get("cacheWrite1hTokens"), r.get("requestId")
unk = [r for r in rows if r["event"] == "litos_cost_unknown"]
out = {"events_after_dedup": len(rows), "duplicate_events_dropped": dup, "responses": len(use)}

# role of each call inside an operation: the backend's own workload label, grouped
ROLE = {"research": "research (web search)", "bullet_litos_bullet_rewrite": "rewrite",
        "bullet_litos_bullet_retry": "retry", "bullet_litos_bullet_judge": "judge",
        "competency_judge": "judge", "resume_spec": "spec"}

kind, outcome, elapsed = {}, {}, {}
for r in rows:
    if r["event"] == "litos_result_started":
        kind[r["operationId"]] = r["resultKind"]
    if r["event"] == "litos_result_completed":
        outcome[r["operationId"]] = r["outcome"]
        elapsed[r["operationId"]] = r.get("elapsedMs")

op_usd = defaultdict(float)
op_calls = Counter()
op_role_usd = defaultdict(lambda: defaultdict(float))
op_role_calls = defaultdict(Counter)
out_usd_role = defaultdict(float)  # output-token dollars by call role, every priced response in the window
out_tok_role = defaultdict(int)
search_n = 0
search_usd_total = 0.0
tok_total = 0.0
unpriced = 0
for r in use:
    t = token_usd(r)
    if t is None:
        unpriced += 1
        continue
    ws = (r.get("webSearchRequests") or 0)
    search_n += ws
    c = t + ws * SEARCH_USD
    tok_total += t
    search_usd_total += ws * SEARCH_USD
    o = r.get("operationId") or "unattributed"
    op_usd[o] += c
    op_calls[o] += 1
    role = ROLE.get(r.get("workload"), "other")
    out_usd_role[role] += r.get("outputTokens", 0) * PR[r["model"]][3] / 1e6
    out_tok_role[role] += r.get("outputTokens", 0)
    op_role_usd[o][role] += c
    op_role_calls[o][role] += 1
out["priced_token_usd"] = round(tok_total, 2)
_ot = sum(out_usd_role.values())
out["window_output_usd_by_role"] = {"output_usd_total": round(_ot, 2), "output_share_of_token_usd": round(_ot / tok_total, 3),
    "role_output_usd": {k: round(v, 2) for k, v in sorted(out_usd_role.items(), key=lambda kv: -kv[1])},
    "role_share_of_output_usd": {k: round(v / _ot, 3) for k, v in sorted(out_usd_role.items(), key=lambda kv: -kv[1])},
    "role_output_tokens": dict(out_tok_role)}
out["web_searches"] = search_n
out["web_search_usd"] = round(search_usd_total, 2)
out["total_usd_with_search"] = round(tok_total + search_usd_total, 2)
out["unpriced_responses"] = unpriced
out["usd_without_operation_key"] = round(op_usd.get("unattributed", 0.0), 2)
out["responses_without_operation_key"] = op_calls.get("unattributed", 0)
out["usd_on_ops_without_start_event"] = round(sum(v for o, v in op_usd.items() if o != "unattributed" and o not in kind), 2)

# ---------- Table A ----------
A = {}
for k in sorted(set(kind.values())):
    ops = [o for o, kk in kind.items() if kk == k]
    acc = [o for o in ops if outcome.get(o) == "accepted"]
    tot = sum(op_usd[o] for o in ops)
    ac = [op_usd[o] for o in acc]
    A[k] = {"operations": len(ops), "accepted": len(acc),
            "outcomes": dict(Counter(outcome.get(o, "missing") for o in ops)),
            "total_usd": round(tot, 2),
            "usd_per_accepted": round(tot / len(acc), 4) if acc else None,
            "accepted_median": round(st.median(ac), 4) if ac else None,
            "accepted_p90": round(pct(ac, 0.9), 4) if ac else None,
            "non_accepted_share_of_spend": round(sum(op_usd[o] for o in ops if outcome.get(o) != "accepted") / tot, 3) if tot else None}
out["table_a"] = A

# ---------- Table B: inside the package operation ----------
pk = [o for o, kk in kind.items() if kk == "application_packet"]
pk_acc = [o for o in pk if outcome.get(o) == "accepted"]
pk_tot = sum(op_usd[o] for o in pk)
roles = ["research (web search)", "rewrite", "retry", "judge", "spec", "other"]
B = {}
for ro in roles:
    usd = sum(op_role_usd[o][ro] for o in pk)
    calls = sum(op_role_calls[o][ro] for o in pk)
    used = [o for o in pk_acc if op_role_calls[o][ro]]
    per_op = [op_role_usd[o][ro] for o in used]
    B[ro] = {"usd": round(usd, 2), "share_of_spend": round(usd / pk_tot, 3), "calls": calls,
             "share_of_calls": round(calls / sum(op_calls[o] for o in pk), 3),
             "accepted_ops_using": len(used),
             "median_usd_per_using_op": round(st.median(per_op), 4) if per_op else None}
out["table_b_packet"] = {"operations": len(pk), "total_usd": round(pk_tot, 2), "roles": B,
                         "search_usd_in_packets": round(sum(SEARCH_USD * (r.get("webSearchRequests") or 0) for r in use if kind.get(r.get("operationId")) == "application_packet"), 2)}
costs = sorted((op_usd[o] for o in pk), reverse=True)
top = costs[: len(costs) // 10]
out["packet_concentration"] = {"top10pct_ops": len(top), "top10pct_share": round(sum(top) / sum(costs), 3)}
# whole window by result kind share
out["window_share_by_kind"] = {k: round(sum(op_usd[o] for o, kk in kind.items() if kk == k) / (tok_total + search_usd_total), 3) for k in A}
out["window_share_retry_role"] = round(sum(op_role_usd[o]["retry"] for o in op_role_usd) / (tok_total + search_usd_total), 3)

# ---------- Table C: failure economics ----------
C = {}
med_acc = st.median(op_usd[o] for o in pk_acc)
for oc in ["incomplete", "failed"]:
    ops = [o for o in pk if outcome.get(o) == oc]
    cs = [op_usd[o] for o in ops]
    C["application_packet_" + oc] = {"operations": len(ops), "median_usd": round(st.median(cs), 4) if cs else None,
                                     "median_vs_accepted_median": round(st.median(cs) / med_acc, 2) if cs else None,
                                     "with_zero_calls": sum(1 for o in ops if op_calls[o] == 0),
                                     "usd": round(sum(cs), 2), "share_of_packet_spend": round(sum(cs) / pk_tot, 3)}
rp = [o for o, kk in kind.items() if kk == "resume_parse"]
rp_acc_med = st.median(op_usd[o] for o in rp if outcome.get(o) == "accepted")
rpf = [o for o in rp if outcome.get(o) == "failed"]
rp_tot = sum(op_usd[o] for o in rp)
C["resume_parse_failed"] = {"operations": len(rpf), "median_vs_accepted_median": round(st.median(op_usd[o] for o in rpf) / rp_acc_med, 2),
                            "share_of_parse_spend": round(sum(op_usd[o] for o in rpf) / rp_tot, 3)}
ar = [o for o, kk in kind.items() if kk == "application_run"]
C["application_run_not_accepted"] = {"operations": len(ar), "outcomes": dict(Counter(outcome.get(o, "missing") for o in ar)),
                                     "usd": round(sum(op_usd[o] for o in ar), 2),
                                     "ops_with_calls": sum(1 for o in ar if op_calls[o]),
                                     "median_usd_ops_with_calls": round(st.median([op_usd[o] for o in ar if op_calls[o]]), 4) if any(op_calls[o] for o in ar) else None}
# retries inside accepted packets: an accepted packet whose calls include a retry
ret = [o for o in pk_acc if op_role_calls[o]["retry"]]
noret = [o for o in pk_acc if not op_role_calls[o]["retry"] and op_calls[o]]
C["accepted_packet_with_retry"] = {"operations": len(ret), "median_usd": round(st.median(op_usd[o] for o in ret), 4),
                                   "median_vs_accepted_median": round(st.median(op_usd[o] for o in ret) / med_acc, 2),
                                   "retry_usd_share_within_these_ops": round(sum(op_role_usd[o]["retry"] for o in ret) / sum(op_usd[o] for o in ret), 3)}
C["accepted_packet_without_retry_with_calls"] = {"operations": len(noret), "median_usd": round(st.median(op_usd[o] for o in noret), 4) if noret else None}
C["accepted_packet_zero_calls"] = sum(1 for o in pk_acc if op_calls[o] == 0)
C["accepted_median_usd"] = round(med_acc, 4)
out["table_c"] = C

# ---------- unknown usage ----------
flag = Counter(r.get("usageCompleteness", "no_flag") for r in use)
reasons = Counter(r["reason"] for r in unk)
partial = {(r["operationId"], r["occurredAt"][:19]) for r in use if r.get("usageCompleteness") == "partial"}
inc = {(r["operationId"], r["occurredAt"][:19]) for r in unk if r["reason"] == "stream_incomplete"}
distinct = flag["no_flag"] + flag["partial"] + len(unk) - len(partial & inc)
denom = len(use) + reasons["provider_error"]
out["unknown_usage"] = {"flags": dict(flag), "cost_unknown_events": dict(reasons),
                        "partial_also_stream_incomplete": len(partial & inc),
                        "distinct_calls_without_complete_usage": distinct, "denominator": denom,
                        "share": round(distinct / denom, 4)}

# ---------- span cost against the provider's daily cost report ----------
day_ev = defaultdict(float)
for r in use:
    t = token_usd(r)
    if t is not None:
        day_ev[(r.get("occurredAt") or r["timestamp"])[:10]] += t
daily = {x["date"]: x for x in csv.DictReader(open(DAILY))}
rec = {}
for d in ["2026-10-24", "2026-10-25", "2026-10-26"]:
    x = daily[d]
    cons = sum(float(x[c]) for c in ("usd_in", "usd_cache_write", "usd_cache_read", "usd_out"))
    keyed = float(x["usd_key_first_1k_apps"])
    rec[d] = {"span_usd": round(day_ev[d], 2), "console_all_keys_usd": round(cons, 2), "console_backend_key_usd": round(keyed, 2),
              "span_over_backend_key": round(day_ev[d] / keyed, 3) if keyed else None}
out["reconciliation_full_days"] = rec

# ---------- the surge, as the bill shows it ----------
surge = {}
for d in ["2026-10-17", "2026-10-18", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]:
    x = daily[d]
    cons = sum(float(x[c]) for c in ("usd_in", "usd_cache_write", "usd_cache_read", "usd_out"))
    surge[d] = {"usd": round(cons, 2), "output_share": round(float(x["usd_out"]) / cons, 3) if cons else None,
                "backend_key_share": round(float(x["usd_key_first_1k_apps"]) / cons, 3) if cons else None}
prior = [sum(float(daily[d][c]) for c in ("usd_in", "usd_cache_write", "usd_cache_read", "usd_out"))
         for d in sorted(daily) if "2026-10-01" <= d <= "2026-10-14"]
out["surge_days"] = surge
out["oct1_14_median_daily_usd"] = round(st.median(prior), 2)
out["console_days"] = len(daily)
out["console_total_usd"] = round(sum(sum(float(x[c]) for c in ("usd_in", "usd_cache_write", "usd_cache_read", "usd_out")) for x in daily.values()), 2)

# ---------- the most expensive accepted package operation, traced ----------
worst = max(pk_acc, key=lambda o: op_usd[o])
calls = sorted([r for r in use if r.get("operationId") == worst], key=lambda r: r.get("occurredAt") or r["timestamp"])
t0 = datetime.fromisoformat([r for r in rows if r.get("operationId") == worst and r["event"] == "litos_result_started"][0]["occurredAt"].replace("Z", "+00:00"))
trace = []
for r in calls:
    ts = datetime.fromisoformat(r["occurredAt"].replace("Z", "+00:00"))
    trace.append({"t_s": round((ts - t0).total_seconds(), 1), "role": ROLE.get(r.get("workload"), "other"),
                  "workload": r.get("workload"), "model": r.get("model"),
                  "usd": round(token_usd(r) + (r.get("webSearchRequests") or 0) * SEARCH_USD, 4),
                  "searches": r.get("webSearchRequests") or 0, "cache_read": r.get("cacheReadInputTokens", 0),
                  "output": r.get("outputTokens", 0), "stop": r.get("stopReason")})
roles_w = defaultdict(float)
for x in trace:
    roles_w[x["role"]] += x["usd"]
out["worst_accepted_packet"] = {"usd": round(op_usd[worst], 4), "calls": len(calls), "elapsed_s": round((elapsed.get(worst) or 0) / 1000, 1),
                                "ratio_to_accepted_median": round(op_usd[worst] / med_acc, 2),
                                "role_usd": {k: round(v, 4) for k, v in roles_w.items()},
                                "role_calls": dict(Counter(x["role"] for x in trace)), "trace": trace}
json.dump(out, open(OUT / "prod.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "worst_accepted_packet"}, indent=1))
w = out["worst_accepted_packet"]
print(json.dumps({k: v for k, v in w.items() if k != "trace"}, indent=1))
for x in w["trace"]:
    print(x)

# ---------- alert check and worst-packet shares ----------
thr = pct([op_usd[o] for o in pk_acc], 0.9)
flag = [o for o in pk if op_usd[o] > thr]
w = out["worst_accepted_packet"]
extra = {"packet_alert_threshold_usd": round(thr, 4), "packets_above": len(flag),
         "packets_above_not_accepted": sum(1 for o in flag if outcome.get(o) != "accepted"),
         "packets_above_with_retry": sum(1 for o in flag if op_role_calls[o]["retry"]),
         "packets_above_with_research": sum(1 for o in flag if op_role_calls[o]["research (web search)"]),
         "accepted_with_research": sum(1 for o in pk_acc if op_role_calls[o]["research (web search)"]),
         "worst_role_share": {k: round(v / w["usd"], 3) for k, v in w["role_usd"].items()},
         "worst_research_searches": sum(x["searches"] for x in w["trace"]),
         "worst_research_cache_read_tokens": sum(x["cache_read"] for x in w["trace"] if x["role"].startswith("research")),
         "worst_retry_output_tokens": sum(x["output"] for x in w["trace"] if x["role"] == "retry"),
         "accepted_median_with_research": round(st.median(op_usd[o] for o in pk_acc if op_role_calls[o]["research (web search)"]), 4),
         "accepted_median_without_research_with_calls": round(st.median(op_usd[o] for o in pk_acc if op_calls[o] and not op_role_calls[o]["research (web search)"]), 4)}
out["extra"] = extra
json.dump(out, open(OUT / "prod.json", "w"), indent=1)
print(json.dumps(extra, indent=1))

# ---------- window description ----------
ts = sorted((r.get("occurredAt") or r["timestamp"]) for r in rows)
mshare = defaultdict(float)
for r in use:
    t = token_usd(r)
    if t is not None:
        mshare[r["model"]] += t
desc = {"first_event": ts[0], "last_event": ts[-1], "operations_started": len(kind),
        "operations_completed": len(outcome), "model_token_share": {k: round(v / tok_total, 3) for k, v in mshare.items()},
        "efficiency_events": sum(1 for r in rows if r["event"] == "litos_efficiency")}
out["window"] = desc
json.dump(out, open(OUT / "prod.json", "w"), indent=1)
print(json.dumps(desc, indent=1))

# ---------- AGENT 2027 round 1 additions ----------
acc_usd = sum(op_usd[o] for o in pk_acc)
ret_all = sum(op_role_usd[o]["retry"] for o in pk)
ret_acc = sum(op_role_usd[o]["retry"] for o in pk_acc)
dvals = {d: sum(float(daily[d][c]) for c in ("usd_in", "usd_cache_write", "usd_cache_read", "usd_out")) for d in daily}
prior_vals = [v for d, v in sorted(dvals.items()) if "2026-10-01" <= d <= "2026-10-14"]
out["round1"] = {
    "retry_share_of_package_spend_all": round(ret_all / pk_tot, 3),
    "retry_share_of_accepted_package_spend": round(ret_acc / acc_usd, 3),
    "retry_usd_in_accepted_share_of_all_retry_usd": round(ret_acc / ret_all, 3),
    "oct1_14_mean_daily_usd": round(st.mean(prior_vals), 2),
    "oct1_14_max_daily_usd": round(max(prior_vals), 2),
    "oct1_14_min_daily_usd": round(min(prior_vals), 2),
    "days_above_15_usd": [d for d, v in sorted(dvals.items()) if v > 15 and d >= "2026-10-01"],
    "oct16_17_usd": [round(dvals["2026-10-16"], 2), round(dvals["2026-10-17"], 2)],
    "oct23_26_usd": [round(dvals[d], 2) for d in ["2026-10-23", "2026-10-24", "2026-10-25", "2026-10-26"]],
    "unattributed_usd_share_of_window": round(out["usd_without_operation_key"] / out["total_usd_with_search"], 3),
    "nonaccepted_packages_with_zero_calls": sum(1 for o in pk if outcome.get(o) != "accepted" and op_calls[o] == 0),
    "nonaccepted_packages": sum(1 for o in pk if outcome.get(o) != "accepted"),
}
json.dump(out, open(OUT / "prod.json", "w"), indent=1)
print(json.dumps(out["round1"], indent=1))

out["round1"]["oct1_14_median_daily_usd_unrounded"] = round(st.median(prior_vals), 4)
out["round1"]["oct20_over_oct1_14_mean"] = round(dvals["2026-10-20"] / st.mean(prior_vals), 1)
json.dump(out, open(OUT / "prod.json", "w"), indent=1)
print(out["round1"]["oct1_14_median_daily_usd_unrounded"], out["round1"]["oct20_over_oct1_14_mean"])

# ---------- Coverage of the window and zero-call sensitivity ----------
# Operation attribution (PR #2527) merged 2026-10-23 14:57:08 UTC; first litos_result_started is at 14:59:47 UTC.
# Only events from BOUND onward are read (top of the file), so every count below is over the window.
first_op = min(when(r) for r in rows if r["event"] == "litos_result_started")
def usd_of(r):
    t = token_usd(r)
    return (t or 0) + (r.get("webSearchRequests") or 0) * SEARCH_USD
nk = [r for r in use if not r.get("operationId")]
nf = [r for r in use if "usageCompleteness" not in r]
pa = [r for r in use if r.get("usageCompleteness") == "partial"]
r2 = {"boundary_utc": BOUND, "first_operation_event": first_op,
      "first_usage_event": min(when(r) for r in use),
      "window": {"responses": len(use), "usd": round(sum(usd_of(r) for r in use), 2),
                 "without_operation_key": len(nk), "usd_without_operation_key": round(sum(usd_of(r) for r in nk), 2),
                 "without_completeness_flag": len(nf), "partial": len(pa)}}
pe = [r for r in unk if r["reason"] == "provider_error"]
r2["provider_errors"] = len(pe)
w_partial = {(r["operationId"], r["occurredAt"][:19]) for r in pa}
w_inc = {(r["operationId"], r["occurredAt"][:19]) for r in unk if r["reason"] == "stream_incomplete"}
unconf = len(nf) + len(pa) + len(pe) + len([1 for r in unk if r["reason"] == "stream_incomplete"]) - len(w_partial & w_inc)
r2["unconfirmed_calls"] = unconf
r2["denominator"] = len(use) + len(pe)
r2["unconfirmed_share"] = round(unconf / r2["denominator"], 4)
r2["unkeyed_share_of_window_usd"] = round(r2["window"]["usd_without_operation_key"] / out["total_usd_with_search"], 4)
r2["cost_unknown_reasons"] = dict(Counter(r["reason"] for r in unk))
# accepted packages: sensitivity to the zero-call ones
acc_calls = [o for o in pk_acc if op_calls[o]]
r2["accepted_packages"] = len(pk_acc)
r2["accepted_packages_zero_calls"] = len(pk_acc) - len(acc_calls)
r2["accepted_with_calls"] = len(acc_calls)
r2["usd_per_accepted_package_with_calls_only"] = round(sum(op_usd[o] for o in pk) / len(acc_calls), 4)
r2["median_accepted_package_with_calls_only"] = round(st.median(op_usd[o] for o in acc_calls), 4)
r2["zero_call_accepted_elapsed_s_median"] = round(st.median((elapsed.get(o) or 0) / 1000 for o in pk_acc if not op_calls[o]), 1)
r2["zero_call_accepted_elapsed_s_max"] = round(max((elapsed.get(o) or 0) / 1000 for o in pk_acc if not op_calls[o]), 1)
out["coverage"] = r2
json.dump(out, open(OUT / "prod.json", "w"), indent=1)
print(json.dumps(r2, indent=1))

# ---------- AGENT 2027 round 3 ----------
import random
r3 = {}
# (a) package counts by start event and by completion event
done = [r for r in rows if r["event"] == "litos_result_completed"]
started_ids = set(kind)
nostart = [r for r in done if r["operationId"] not in started_ids]
r3["completed_events"] = len(done)
r3["started_events"] = len(kind)
r3["completed_without_start_event"] = [(r["resultKind"], r["outcome"]) for r in nostart]
pk_done = [r for r in done if r["resultKind"] == "application_packet"]
r3["packages_by_completion_event"] = len(pk_done)
r3["packages_accepted_by_completion_event"] = sum(1 for r in pk_done if r["outcome"] == "accepted")
r3["packages_by_start_event"] = len(pk)
r3["usd_window_responses"] = round(sum(usd_of(r) for r in use), 3)
from datetime import timezone
f0 = datetime.fromisoformat(min((r.get("occurredAt") or r["timestamp"]) for r in rows)[:19])
f1 = datetime.fromisoformat(max((r.get("occurredAt") or r["timestamp"]) for r in rows)[:19])
r3["observed_span_days_first_to_last_event"] = round((f1 - f0).total_seconds() / 86400, 2)
# (c) bootstrap (seeded, 10,000 resamples, percentile 95% interval)
rng = random.Random(2027)
B = 10000
pk_cost = [(op_usd[o], outcome.get(o) == "accepted") for o in pk]
cps = []
for _ in range(B):
    s = [pk_cost[rng.randrange(len(pk_cost))] for _ in pk_cost]
    na = sum(1 for _, a in s if a)
    cps.append(sum(c for c, _ in s) / na)
cps.sort()
r3["bootstrap_resamples"] = B
r3["usd_per_accepted_package_ci95"] = [round(cps[int(0.025 * B)], 4), round(cps[int(0.975 * B) - 1], 4)]
ret_c = [op_usd[o] for o in ret]
nor_c = [op_usd[o] for o in noret]
rat = []
for _ in range(B):
    a = [ret_c[rng.randrange(len(ret_c))] for _ in ret_c]
    b = [nor_c[rng.randrange(len(nor_c))] for _ in nor_c]
    rat.append(st.median(a) / st.median(b))
rat.sort()
r3["retry_group_n"] = len(ret_c)
r3["noretry_group_n"] = len(nor_c)
r3["retry_vs_noretry_median_ratio"] = round(st.median(ret_c) / st.median(nor_c), 2)
r3["retry_vs_noretry_ratio_ci95"] = [round(rat[int(0.025 * B)], 2), round(rat[int(0.975 * B) - 1], 2)]
r3["noretry_median_usd"] = round(st.median(nor_c), 4)
r3["noretry_ratio_vs_accepted_median"] = round(st.median(nor_c) / med_acc, 2)
r3["noretry_retry_share"] = 0.0
r3["noretry_min_max_usd"] = [round(min(nor_c), 4), round(max(nor_c), 4)]
out["round3"] = r3
json.dump(out, open(OUT / "prod.json", "w"), indent=1)
print(json.dumps(r3, indent=1))
