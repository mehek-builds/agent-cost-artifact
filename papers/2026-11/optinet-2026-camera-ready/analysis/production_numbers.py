#!/usr/bin/env python3
"""Production figures quoted in Section VI (deployment stories) of the OPTINET camera-ready.

Reads only aggregate exports of the author's deployed agent (no prompts, no user content):
  production-aggregates/litos-console-daily-2026-07-30_2026-10-27.csv        daily tokens and dollars, Litos keys
  production-aggregates/deep_history.json                                     monthly cost per 1k requests
  production-aggregates/console-requests-by-key-daily-2026-06-02_2026-10-27.csv  daily requests, four product keys
  production-aggregates/console-cost-by-key-model.csv                         monthly dollars by key and model
  production-aggregates/ops_models_tagged_window.json                         models seen in the tagged window

    python3 analysis/production_numbers.py > analysis/production_numbers_stdout.txt
"""
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "production-aggregates"  # published aggregates

rows = list(csv.DictReader(open(P / "litos-console-daily-2026-07-30_2026-10-27.csv")))
tok = sum(int(r[k]) for r in rows for k in ("tok_in", "tok_cache_write", "tok_cache_read", "tok_out"))
usd = sum(float(r[k]) for r in rows for k in ("usd_in", "usd_cache_write", "usd_cache_read", "usd_out"))
print(f"console daily: {len(rows)} days {rows[0]['date']}..{rows[-1]['date']}, tokens {tok:,} ({tok / 1e6:.1f} M), "
      f"token usd {usd:.2f}")

# Revision (R7): the four-day event-log counts (usage completeness, window spend, timer stops, skipped restarts)
# moved to analysis/ops_tagged_window.py, which keeps only the operation-tagged window from the roll-out onward.

dh = json.load(open(P / "deep_history.json"))["timeline"]
months = list(dh)
per = [dh[m]["usd_per_1k_requests"] for m in months]
ratios = [per[i + 1] / per[i] for i in range(len(per) - 1)]
print("(five keys incl. testing key; superseded by the four-key lines) usd per 1k requests by period:", {m: round(v, 2) for m, v in zip(months, per)})
print("month-to-month ratios:", [round(r, 2) for r in ratios], "min", round(min(ratios), 2), "max", round(max(ratios), 2))
print(f"highest over lowest month: {max(per):.2f} / {min(per):.2f} = {max(per) / min(per):.2f}")

# ---- round 2: cost per request by 30-day period, the four Litos product keys only (the operator's testing key and
# the research keys are excluded). Periods and cost windows are those of deep_history.json.
KEYS4 = {"first-1k-apps", "litos-first-100", "litos-trial-subs", "student-outreach-extension"}
urows = [r for r in csv.DictReader(open(P / "console-requests-by-key-daily-2026-06-02_2026-10-27.csv"))
         if r["kind"] == "U" and r["api_key_name"] in KEYS4]
crows = [r for r in csv.DictReader(open(P / "console-cost-by-key-model.csv")) if r["api_key_name"] in KEYS4]
PER = [("P1", "2026-06-01", "2026-06-30", "2026-06-02~2026-06-30"), ("P2", "2026-07-01", "2026-07-29", "2026-07-01~2026-07-30"),
       ("P3", "2026-07-30", "2026-08-28", "2026-07-31~2026-08-29"), ("P4", "2026-08-29", "2026-09-27", "2026-08-30~2026-09-28"),
       ("P5", "2026-09-28", "2026-10-27", "2026-09-29~2026-10-27")]
per4 = []
for name, a, b, w in PER:
    req = sum(float(r["requests"]) for r in urows if a <= r["date_or_window"] <= b)
    usd4 = sum(float(r["usd"]) for r in crows if r["window"] == w)
    per4.append(1000 * usd4 / req)
    print(f"four keys {name} {a}..{b}: requests {req:.0f}, usd {usd4:.2f}, usd per 1k requests {per4[-1]:.2f}")
r4 = [per4[i + 1] / per4[i] for i in range(len(per4) - 1)]
print("four-key period-to-period ratios:", [round(x, 2) for x in r4], "min", round(min(r4), 2), "max", round(max(r4), 2))
print(f"four-key highest over lowest 30-day period: {max(per4):.2f} / {min(per4):.2f} = {max(per4) / min(per4):.2f}")

ops = json.load(open(P / "ops_models_tagged_window.json"))
print("models in logged window:", sorted(ops["models"]))

# ---- revision round 1 additions
# (a) model share of spend, the four Litos product keys only (explicit include set), Jun 2 to Oct 27.
# Revision round 4: a personal key of the operator that is not a Litos product key was previously in
# this include set; it is now excluded (the script never names it, so no key name outside the set prints).
import collections
mrows = list(csv.DictReader(open(P / "console-cost-by-key-model.csv")))
INCL = {"first-1k-apps", "litos-first-100", "litos-trial-subs", "student-outreach-extension"}
share = collections.Counter()
left_out_usd = 0.0
left_out_n = set()
for r in mrows:
    if r["api_key_name"] in INCL:
        share[r["model"]] += float(r["usd"])
    else:
        left_out_usd += float(r["usd"])
        left_out_n.add(r["api_key_name"])
tot_share = sum(share.values())
wins = sorted({r["window"] for r in mrows})
top3 = share.most_common(3)
print(f"model share of spend by key-model export, four Litos product keys {sorted(INCL)}, windows {wins[0].split('~')[0]}..{wins[-1].split('~')[1]}, total ${tot_share:.2f}: "
      + ", ".join(f"{m} {v / tot_share:.2%}" for m, v in top3)
      + f"; other models {(tot_share - sum(v for _, v in top3)) / tot_share:.2%}")
print(f"keys left out of the share (research, testing and non-Litos keys): {len(left_out_n)} keys, ${left_out_usd:.2f}")

# (b) which keys the $389 covers
LITOS_COLS = ["usd_key_litos_trial_subs", "usd_key_student_outreach_extension", "usd_key_litos_first_100", "usd_key_first_1k_apps"]
keycols = [k for k in rows[0] if k.startswith("usd_key_")]
kt = {k: sum(float(r[k]) for r in rows) for k in LITOS_COLS}
print("per-key usd columns (four Litos product keys), Jul 30..Oct 27:", {k[8:]: round(v, 2) for k, v in kt.items()})
other_cols = [k for k in keycols if k not in LITOS_COLS]
print(f"other key columns in the daily file (research and non-Litos keys): {len(other_cols)}, "
      f"${sum(float(r[k]) for r in rows for k in other_cols):.2f}")
study_cols = [k for k in other_cols if "research" in k]
print("study key (benchmark runs), 90-day spend in the daily file: "
      + ", ".join(f"${sum(float(r[k]) for r in rows):.2f}" for k in study_cols))
nb = sum(kt.values())
# (difference line printed after the per-day reconciliation below)

# ---- revision round 2, corrected in round 4
# (d) the token-usd total equals the four Litos product key columns, day by day
gaps = []
for r in rows:
    d = sum(float(r[k]) for k in ("usd_in", "usd_cache_write", "usd_cache_read", "usd_out"))
    k4 = sum(float(r[k]) for k in LITOS_COLS)
    gaps.append(abs(d - k4))
signed = sum(sum(float(r[k]) for k in LITOS_COLS)
             - sum(float(r[k]) for k in ("usd_in", "usd_cache_write", "usd_cache_read", "usd_out")) for r in rows)
print(f"per-day |token usd - four Litos key columns|: total {sum(gaps):.2f}, max {max(gaps):.2f}; "
      f"four-key sum {nb:.2f} vs token usd {usd:.2f}, signed difference {signed:.2f}")
