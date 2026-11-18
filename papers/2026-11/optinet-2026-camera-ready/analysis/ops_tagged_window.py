#!/usr/bin/env python3
"""Deployment counts for Stories B and C, restricted to the operation-tagged window (no API calls, no network).

The deployed agent began tagging every provider call with its operation and a usage-completeness flag at a roll-out
on 2026-10-23. This script keeps only events from the first operation event onward, checks that every usage record
kept there is an operation-tagged record, and counts (B) calls without complete usage and (C) timeout restarts skipped. Records before the roll-out are not used.
Events are de-duplicated on eventId. Only counts are printed. The event stream holds per-user operation records
and is not published; set LITOS_EVENTS to its path. The saved output is ops_tagged_window_stdout.txt.

    python3 -I analysis/ops_tagged_window.py > analysis/ops_tagged_window_stdout.txt
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

SRC = Path(os.environ.get("LITOS_EVENTS", "ops-events-2026-10-23_27.ndjson"))  # private raw input
if not SRC.exists():
    sys.exit(f"private input {SRC} not found: the Litos event stream is not published; set LITOS_EVENTS to its path, "
             "or use the published ops_tagged_window_stdout.txt (see papers/2026-11/README.md)")

rows, seen = [], set()
for line in open(SRC):
    r = json.loads(line)
    k = r.get("eventId") or json.dumps(r, sort_keys=True)
    if k in seen:
        continue
    seen.add(k)
    rows.append(r)


def when(r):
    return r.get("occurredAt") or r["timestamp"]


start = min(when(r) for r in rows if r["event"].endswith("_result_started"))
end = max(when(r) for r in rows)
kept = [r for r in rows if when(r) >= start]
use = [r for r in kept if r["event"] == "anthropic_response_usage"]
unk = [r for r in kept if r["event"] == "litos_cost_unknown"]
print(f"operation-tagged window: {start[:16]} .. {end[:16]} UTC")
print(f"usage records kept: {len(use)}; without operation key: {sum(1 for r in use if not r.get('operationId'))}")
flag = Counter(r.get("usageCompleteness", "no_flag") for r in use)
reasons = Counter(r["reason"] for r in unk)
print("completeness flags:", dict(flag), "cost-unknown events:", dict(reasons))
partial = {(r.get("operationId"), when(r)[:19]) for r in use if r.get("usageCompleteness") == "partial"}
inc = {(r.get("operationId"), when(r)[:19]) for r in unk if r["reason"] == "stream_incomplete"}
counted = flag["no_flag"] + flag["partial"] + len(unk)
distinct = counted - len(partial & inc)
den = len(use) + reasons["provider_error"]
print(f"(B) calls without complete usage: {distinct} of {den} = {distinct / den:.4f}")
outc = {r["operationId"]: r["outcome"] for r in kept if r["event"].endswith("_result_completed")}
sk = [r for r in kept if r["event"].endswith("_efficiency") and r.get("reason") == "timeout_restart_skipped"]
print(f"(C) timeout restarts skipped: {sum(int(r.get('count', 1) or 1) for r in sk)}; "
      f"outcomes of their operations: {[outc.get(o) for o in {r['operationId'] for r in sk}]}")
