"""Placement policies with the measured edge model as the small tier and the recorded cached large model.

python3 edge/placement_edge.py edge/runs/full  -> prints the table, writes placement_edge.json in the run folder

Edge energy per request is measured (edge/energy.py method). Large-tier energy is not measurable on a hosted
API; it uses a published estimate of 3 to 4 J per output token, the range Samsi et al. (HPEC 2023) measured for
LLaMA 65B on 8 to 32 V100 GPUs (GPU power only), applied to the large model's recorded output tokens.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from energy import energy, samples  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
run = Path(sys.argv[1])
s = samples(run / "power.txt")
timing = [json.loads(l) for l in open(run / "timing_edge.jsonl")]
steps = [json.loads(l) for l in open(run / "steps_edge.jsonl")]
etasks = {t["task_id"]: t for t in map(json.loads, open(run / "tasks_edge.jsonl"))}
for st, tm in zip(steps, timing):
    assert st["task_id"] == tm["task_id"]
    st["joules"] = energy(s, tm["t_start"], tm["t_end"])
E = {}
for st in steps:
    E.setdefault(st["task_id"], []).append(st)
L = {}
for l in open(ROOT / "logs" / "steps_frontier_cache.jsonl"):
    st = json.loads(l)
    L.setdefault(st["task_id"], []).append(st)
Lt = {t["task_id"]: t for t in map(json.loads, open(ROOT / "logs" / "tasks_frontier_cache.jsonl"))}

res = {}
for pol in ["edge", "large", "first_error", "oracle"]:
    j_edge = usd = out_large = 0.0
    succ = esc = 0
    for tid, et in etasks.items():
        es, ls = E.get(tid, []), L[tid]
        if pol == "edge":
            use_e, use_l, ok = es, [], et["success"]
        elif pol == "large":
            use_e, use_l, ok = [], ls, Lt[tid]["success"]
        elif pol == "oracle":
            use_e, use_l, ok = (es, [], True) if et["success"] else ([], ls, Lt[tid]["success"])
            esc += not et["success"]
        else:
            k = next((i for i, x in enumerate(es) if x["n_errors"] > 0), None)
            use_e, use_l, ok = (es, [], et["success"]) if k is None else (es[: k + 1], ls, Lt[tid]["success"])
            esc += k is not None
        j_edge += sum(x["joules"] for x in use_e)
        usd += sum(x["cost_usd"] for x in use_l)
        out_large += sum(x["usage"]["output_tokens"] for x in use_l)
        succ += ok
    res[pol] = {"successes": succ, "escalations": esc, "edge_joules": j_edge, "large_usd": usd,
                "large_output_tokens": out_large,
                "joules_per_success_low": (j_edge + 3 * out_large) / succ, "joules_per_success_high": (j_edge + 4 * out_large) / succ,
                "usd_per_success": usd / succ}
    r = res[pol]
    print(f"{pol:12s} succ {succ:2d}/50 esc {esc:2d}  edge {j_edge/1000:6.2f} kJ  large out-tok {out_large:6.0f}  "
          f"J/success {r['joules_per_success_low']:7.0f}-{r['joules_per_success_high']:7.0f}  USD/success {r['usd_per_success']:.4f}")
trig = [tid for tid in etasks if any(x["n_errors"] > 0 for x in E.get(tid, []))]
fails = [tid for tid in etasks if not etasks[tid]["success"]]
tp = len(set(trig) & set(fails))
print(f"first-error trigger: fired {len(trig)}, edge failures {len(fails)}, recall {tp/len(fails):.2f}, precision {tp/max(len(trig),1):.2f}")
(run / "placement_edge.json").write_text(json.dumps(res, indent=1))
