"""Compare logged task costs (sum over every task record in the six multi-turn configs and the
five single-turn configs) with the provider bill for the study's dedicated API key,
bpod-research-2026, in papers/2026-11/production-aggregates/console-cost-by-key-model.csv. Read-only."""
import json, glob, csv, os
A = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))  # repository root
AGG = os.path.join(A, "papers", "2026-11", "production-aggregates")
C = os.path.join(AGG, "console-cost-by-key-model.csv")
logged = {"claude-sonnet-5-5": 0.0, "claude-haiku-5-5": 0.0}
def model(cfg): return "claude-sonnet-5-5" if cfg.startswith("frontier") else "claude-haiku-5-5"
for f in glob.glob(f"{A}/logs/tasks_*.jsonl"):
    for l in open(f):
        r = json.loads(l); logged[model(r["config"])] += r["cost_usd"]
for f in glob.glob(f"{A}/logs/single_turn/tasks_*.jsonl"):
    for l in open(f):
        r = json.loads(l); logged[model(r["config"])] += r["cost_usd"]
billed = {}
for row in csv.DictReader(open(C)):
    if row["api_key_name"] == "bpod-research-2026" and row["cost_type"] == "tokens":
        billed[row["model"]] = billed.get(row["model"], 0.0) + float(row["usd"])
tl, tb = sum(logged.values()), sum(billed.values())
for m in logged:
    print(f"{m}: logged {logged[m]:.4f} billed {billed[m]:.4f} share {100*logged[m]/billed[m]:.1f}%")
print(f"total: logged {tl:.4f} billed {tb:.4f} share {100*tl/tb:.1f}%")

# ---- request-count reconciliation (added round 5) ----
U = os.path.join(AGG, "console-usage-by-key-monthly-2026-06-02_2026-10-27.csv")  # monthly rows only
lreq = {"claude-sonnet-5-5": 0, "claude-haiku-5-5": 0}
for f in glob.glob(f"{A}/logs/tasks_*.jsonl"):
    for l in open(f):
        r = json.loads(l); lreq[model(r["config"])] += r["llm_steps"]
for f in glob.glob(f"{A}/logs/single_turn/tasks_*.jsonl"):
    for l in open(f):
        r = json.loads(l); lreq[model(r["config"])] += r["llm_calls"]
breq = {}
for row in csv.DictReader(open(U)):
    if row["api_key_name"] == "bpod-research-2026" and row["kind"] == "M":
        breq[row["model"]] = breq.get(row["model"], 0) + int(row["requests"])
for m in lreq:
    print(f"requests {m}: logged {lreq[m]} billed {breq[m]} unlogged {breq[m]-lreq[m]} ({100*(breq[m]-lreq[m])/breq[m]:.1f}% of billed)")
tl2, tb2 = sum(lreq.values()), sum(breq.values())
print(f"requests total: logged {tl2} billed {tb2} unlogged {tb2-tl2} ({100*(tb2-tl2)/tb2:.1f}% of billed)")
