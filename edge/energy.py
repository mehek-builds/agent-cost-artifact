"""Attribute measured SoC power to each model request of an edge run.

python3 edge/energy.py edge/runs/full  -> prints a summary, writes energy_summary.json in the run folder

Input: power.txt from `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500`, the run's
timing_edge.jsonl (wall-clock start and end of every request) and tasks_edge.jsonl (BFCL verdicts), plus
baseline_start/baseline_end (an idle window recorded just before the run).

Method: each sample's "Combined Power (CPU + GPU + ANE)" is placed on a time axis anchored at its
header timestamp. Energy of a request is the integral of power over its window; dynamic energy subtracts
the mean idle power. Fresh prompt tokens are those not in the previous request of the same task (Ollama
reuses the cached prefix); a least-squares fit splits dynamic energy into joules per fresh prompt token,
per output token, and a per-request constant.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

HDR = re.compile(r"\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\)")
PWR = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW")


def samples(path):
    out, t = [], None
    for line in open(path, errors="ignore"):
        m = HDR.search(line)
        if m:
            end = datetime.strptime(m.group(1), "%a %b %d %H:%M:%S %Y %z").timestamp()
            t = (end - float(m.group(2)) / 1000, end)
            continue
        m = PWR.search(line)
        if m and t:
            out.append((t[0], t[1], int(m.group(1)) / 1000))
            t = None
    return out


def energy(s, a, b, idle=0.0):
    """Joules between times a and b, from piecewise-constant samples, minus idle power."""
    e = 0.0
    for t0, t1, w in s:
        ov = min(t1, b) - max(t0, a)
        if ov > 0:
            e += ov * (w - idle)
    return e


def lstsq(rows, ys):
    """Ordinary least squares for 3 coefficients via the normal equations."""
    n = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    aty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(n)]
    for i in range(n):  # Gauss-Jordan
        p = ata[i][i]
        ata[i] = [v / p for v in ata[i]]
        aty[i] /= p
        for k in range(n):
            if k != i:
                f = ata[k][i]
                ata[k] = [vk - f * vi for vk, vi in zip(ata[k], ata[i])]
                aty[k] -= f * aty[i]
    return aty


def main(run):
    run = Path(run)
    s = samples(run / "power.txt")
    b0, b1 = (float((run / f).read_text()) for f in ("baseline_start", "baseline_end"))
    idle_samples = [w for t0, t1, w in s if t0 >= b0 + 5 and t1 <= b1 - 5]
    idle = sum(idle_samples) / len(idle_samples)
    timing = [json.loads(l) for l in open(run / "timing_edge.jsonl")]
    tasks = [json.loads(l) for l in open(run / "tasks_edge.jsonl")]
    prev = {}
    rows, ys, total_e, total_dyn, total_t = [], [], 0.0, 0.0, 0.0
    fresh_sum = out_sum = prompt_sum = 0
    per_task = {}
    for t in timing:
        p, o = t["prompt_eval_count"], t["eval_count"]
        reused = prev.get(t["task_id"], 0)
        fresh = p if not reused else max(p - reused, 0)
        prev[t["task_id"]] = p + o
        e = energy(s, t["t_start"], t["t_end"])
        dyn = energy(s, t["t_start"], t["t_end"], idle)
        total_e += e
        total_dyn += dyn
        total_t += t["t_end"] - t["t_start"]
        fresh_sum += fresh
        out_sum += o
        prompt_sum += p
        per_task[t["task_id"]] = per_task.get(t["task_id"], 0.0) + e
        rows.append([fresh, o, 1.0])
        ys.append(dyn)
    j_fresh, j_out, j_req = lstsq(rows, ys)
    succ = sum(1 for r in tasks if r["success"])
    res = {
        "requests": len(timing), "tasks": len(tasks), "successes": succ,
        "idle_w": idle, "mean_active_w": total_e / total_t, "active_seconds": total_t,
        "energy_j": total_e, "dynamic_energy_j": total_dyn,
        "energy_wh": total_e / 3600,
        "j_per_request": total_e / len(timing), "j_per_task": total_e / len(tasks),
        "j_per_success": total_e / succ if succ else None,
        "dynamic_j_per_success": total_dyn / succ if succ else None,
        "prompt_tokens": prompt_sum, "fresh_prompt_tokens": fresh_sum, "output_tokens": out_sum,
        "fit_dynamic_j_per_fresh_prompt_token": j_fresh, "fit_dynamic_j_per_output_token": j_out,
        "fit_dynamic_j_per_request": j_req,
        "seconds_per_request_median": sorted(t["t_end"] - t["t_start"] for t in timing)[len(timing) // 2],
        "samples": len(s), "idle_samples": len(idle_samples),
    }
    (run / "energy_summary.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k:40s} {v:.4f}" if isinstance(v, float) else f"{k:40s} {v}")


if __name__ == "__main__":
    main(sys.argv[1])
