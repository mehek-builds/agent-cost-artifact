"""De-identified per-package aggregate of the Litos window (the released input for recompute_from_aggregate.py).
One row per application package (rows in the order the events were recorded): its rank by start time (no timestamps, no identifiers), outcome, total USD, USD and
call counts by call role, and web-search count. No text, ids or user data. Needs the private event stream.
python3 -I export_package_aggregate.py -> out/litos_package_aggregate.csv"""
import contextlib, csv, io, sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
with contextlib.redirect_stdout(io.StringIO()):
    import prod
ROLES = ["research (web search)", "rewrite", "retry", "judge", "spec", "other"]
NAMES = ["research", "rewrite", "retry", "judge", "spec", "other"]
start = {r["operationId"]: r["occurredAt"] for r in prod.rows if r["event"] == "litos_result_started"}
searches = defaultdict(int)
for r in prod.use:
    if r.get("operationId") in start:
        searches[r["operationId"]] += r.get("webSearchRequests") or 0
pk = [o for o, k in prod.kind.items() if k == "application_packet"]  # event-file order, as in prod.py (the seeded bootstrap depends on it)
rank = {o: i for i, o in enumerate(sorted(pk, key=lambda o: start[o]), 1)}
with open(HERE / "out" / "litos_package_aggregate.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["package_rank_by_start", "outcome", "total_usd"] + [f"usd_{n}" for n in NAMES]
               + [f"calls_{n}" for n in NAMES] + ["calls_total", "web_searches"])
    for o in pk:
        w.writerow([rank[o], prod.outcome.get(o, "missing"), f"{prod.op_usd[o]:.6f}"]
                   + [f"{prod.op_role_usd[o][r]:.6f}" for r in ROLES]
                   + [prod.op_role_calls[o][r] for r in ROLES] + [prod.op_calls[o], searches[o]])
print("packages:", len(pk))
