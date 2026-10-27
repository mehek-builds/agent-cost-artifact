"""Costs of lightweight operation per successful task, from the measured edge run and the recorded large tier.

python3 edge/sustainability.py edge/runs/full  -> prints a table, writes sustainability.json in the run folder

Inputs are measured (edge energy and time per request) or recorded (large-tier output tokens and dollars);
everything else is a stated assumption swept over a range:
  cloud energy per output token  3 to 4 J (Samsi et al., HPEC 2023; GPU only)
  cloud PUE                      1.1 to 1.6 (applied to the cloud tier only; the edge figure is wall-side SoC power)
  grid carbon intensity          50, 400 and 700 gCO2e/kWh (applied to both tiers)
  edge device                    USD 1,600 amortized over 3 years at 10% or 50% utilization
"""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from energy import energy, samples  # noqa: E402

s = samples(run / "power.txt")
timing = [json.loads(l) for l in open(run / "timing_edge.jsonl")]
steps = [json.loads(l) for l in open(run / "steps_edge.jsonl")]
etasks = {t["task_id"]: t for t in map(json.loads, open(run / "tasks_edge.jsonl"))}
for st, tm in zip(steps, timing):
    st["joules"] = energy(s, tm["t_start"], tm["t_end"])
    st["seconds"] = tm["t_end"] - tm["t_start"]
E = {}
for st in steps:
    E.setdefault(st["task_id"], []).append(st)
L = {}
for l in open(ROOT / "logs" / "steps_frontier_cache.jsonl"):
    st = json.loads(l)
    L.setdefault(st["task_id"], []).append(st)
Lt = {t["task_id"]: t for t in map(json.loads, open(ROOT / "logs" / "tasks_frontier_cache.jsonl"))}

YEAR_S = 3 * 365 * 86400
pol = {}
for name in ("edge", "large", "first_error"):
    ej = es = lout = usd = lsec = 0.0
    succ = 0
    for tid, et in etasks.items():
        e, l = E.get(tid, []), L[tid]
        if name == "edge":
            ue, ul, ok = e, [], et["success"]
        elif name == "large":
            ue, ul, ok = [], l, Lt[tid]["success"]
        else:
            k = next((i for i, x in enumerate(e) if x["n_errors"] > 0), None)
            ue, ul, ok = (e, [], et["success"]) if k is None else (e[: k + 1], l, Lt[tid]["success"])
        ej += sum(x["joules"] for x in ue)
        es += sum(x["seconds"] for x in ue)
        lout += sum(x["usage"]["output_tokens"] for x in ul)
        lsec += sum(x["latency_s"] for x in ul)
        usd += sum(x["cost_usd"] for x in ul)
        succ += ok
    pol[name] = {"successes": succ, "edge_j": ej, "edge_s": es, "large_out": lout, "large_s": lsec, "api_usd": usd}

res = {}
for name, p in pol.items():
    k = p["successes"]
    rows = {}
    for jpt in (3.0, 4.0):
        for pue in (1.1, 1.6):
            j = p["edge_j"] + p["large_out"] * jpt * pue
            rows[f"J={jpt},PUE={pue}"] = {"j_per_success": j / k,
                                          **{f"gco2_per_success@{g}": j / k / 3.6e6 * g for g in (50, 400, 700)}}
    amort = {f"util={u}": 1600 * p["edge_s"] / (YEAR_S * u) / k for u in (0.1, 0.5)}
    res[name] = {"successes": k, "seconds_per_success": (p["edge_s"] + p["large_s"]) / k,
                 "api_usd_per_success": p["api_usd"] / k, "device_usd_per_success": amort, "energy": rows}
(run / "sustainability.json").write_text(json.dumps(res, indent=1))
for name, r in res.items():
    lo = min(v["j_per_success"] for v in r["energy"].values())
    hi = max(v["j_per_success"] for v in r["energy"].values())
    print(f"{name:12s} succ {r['successes']}  s/success {r['seconds_per_success']:.1f}  API $/succ {r['api_usd_per_success']:.4f}  "
          f"device $/succ {r['device_usd_per_success']['util=0.1']:.4f} (10%) {r['device_usd_per_success']['util=0.5']:.4f} (50%)  "
          f"J/succ {lo:.0f}-{hi:.0f}  gCO2e/succ @400 {lo/3.6e6*400:.3f}-{hi/3.6e6*400:.3f}")
