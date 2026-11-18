#!/usr/bin/env python3
"""Every number the OPTINET camera-ready keeps from the recorded-run analyses.

Reads the result files the public artifact's scripts wrote (optinet/results/fixed_spend.json,
optinet/results/extras.json, optinet/results/extras_stdout.txt, results/escalation_stdout.txt)
and the recorded logs (logs/*.jsonl) through the artifact's loaders. Prints one labeled line
per number so evidence-ledger.md can point at it.

    python3 analysis/artifact_numbers.py > analysis/artifact_numbers_stdout.txt
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ART = Path(__file__).resolve().parents[4]  # repository root
sys.path.insert(0, str(ART / "optinet"))
sys.path.insert(0, str(ART))
import fixed_spend as fs  # noqa: E402
import replay  # noqa: E402

R = json.loads((ART / "optinet/results/fixed_spend.json").read_text())
X = json.loads((ART / "optinet/results/extras.json").read_text())


def row(block, frac, pol):
    return next(r for r in R[block]["rows"] if abs(r["frac"] - frac) < 1e-9 and r["policy"] == pol)


def ci(block, frac, pol):
    return R[block][str(frac)]["ci95"][pol]


def diff(block, frac, key):
    d = R[block][str(frac)]["diff"][key]
    return f"{d['mean']:+.2f} [{d['ci95'][0]}, {d['ci95'][1]}]"


m = R["cached"]["meta"]
print(f"[workload] cached batch cost large ${m['full_large']:.4f}, small ${m['full_small']:.4f}; "
      f"small/large = {m['full_small'] / m['full_large']:.3f}; large solves {m['large_succ']}, small {m['small_succ']}; "
      f"trigger rate {m['esc_rate']}")
mu = R["uncached"]["meta"]
print(f"[workload] uncached batch cost large ${mu['full_large']:.4f} = {mu['full_large'] / m['full_large']:.2f}x cached; "
      f"large solves {mu['large_succ']}")

small, large = replay.load("small_cache_budget"), replay.load("frontier_cache")
ids = sorted(set(small) & set(large))
turns = sum(small[i].get("turns", 0) for i in ids)
fired = [i for i in ids if fs.fires(small[i]) is not None]
print(f"[workload] tasks {len(ids)}; trigger fires on {len(fired)}; small solves {sum(small[i]['success'] for i in fired)} "
      f"of them, large solves {sum(large[i]['success'] for i in fired)}")
tasks_file = [json.loads(l) for l in open(ART / "logs/tasks_small_cache_budget.jsonl")]
print(f"[workload] user turns over the 50 tasks: {sum(t['turns'] for t in tasks_file)}")

for name in ("cached", "uncached"):
    p = R[f"causal_{name}"]["prior"]
    mm = R[name]["meta"]
    print(f"[priors] {name}: prior/true small {p['small'] / (mm['full_small'] / 50):.2f}, large {p['large'] / (mm['full_large'] / 50):.2f}")

print("[Table I] mean over 500 orders [task-bootstrap 95% percentile interval]")
spec = [("small", "cached", "bootstrap_cached"), ("large", "cached", "bootstrap_cached"),
        ("stop", "cached", "bootstrap_cached"), ("stop-fallback", "cached", "bootstrap_cached"),
        ("route", "cached", "bootstrap_cached"), ("admit", "causal_cached", "bootstrap_causal_cached"),
        ("random-paced", "causal_cached", "bootstrap_causal_cached"), ("hybrid", "causal_cached", "bootstrap_causal_cached"),
        ("oracle-paced", "causal_cached", "bootstrap_causal_cached")]
for pol, blk, bblk in spec:
    cells = [f"{row(blk, f, pol)['succ_mean']:.1f} {ci(bblk, f, pol)}" for f in fs.TABLE_FRACS]
    print(f"  {pol:>13}: " + " | ".join(cells))

print("[paired, causal cached]")
for f in fs.TABLE_FRACS:
    print(f"  {f:.0%}: hybrid-large {diff('bootstrap_causal_cached', f, 'hybrid - large')}; "
          f"hybrid-route {diff('bootstrap_causal_cached', f, 'hybrid - route')}; "
          f"hybrid-stop-fallback {diff('bootstrap_causal_cached', f, 'hybrid - stop-fallback')}; "
          f"hybrid-paced-abandon {diff('bootstrap_causal_cached', f, 'hybrid - paced-abandon')}; "
          f"hybrid-random {diff('bootstrap_causal_cached', f, 'hybrid - random-paced')}; "
          f"hybrid-admit {diff('bootstrap_causal_cached', f, 'hybrid - admit')}")
print("[paired, causal uncached]")
for f in fs.TABLE_FRACS:
    print(f"  {f:.0%}: hybrid-admit {diff('bootstrap_causal_uncached', f, 'hybrid - admit')}; "
          f"hybrid {row('causal_uncached', f, 'hybrid')['succ_mean']:.2f}; admit {row('causal_uncached', f, 'admit')['succ_mean']:.2f} "
          f"unstarted {row('causal_uncached', f, 'admit')['reasons']['unstarted']:.2f}")

r = row("causal_cached", 0.2, "hybrid")
lg = row("causal_cached", 0.2, "large")
tl_h, tl_l = r["tok_per_solved"] * r["tok_large_share"], lg["tok_per_solved"] * lg["tok_large_share"]
print(f"[work] 20% large-model tokens per solved: paced {tl_h / 1e3:.1f}k, Large {tl_l / 1e3:.1f}k, "
      f"reduction {1 - tl_h / tl_l:.3f}; total tokens per solved paced {r['tok_per_solved'] / 1e3:.1f}k vs Large {lg['tok_per_solved'] / 1e3:.1f}k")
print(f"[starvation] Large at 20%: unstarted {lg['reasons']['unstarted']:.1f}, model {lg['reasons']['model']:.1f}, pool {lg['reasons']['pool']:.1f}")
_costs = [fs.task_cost(large[i]) for i in ids]
print(f"[cap] cheapest cached large-model task ${min(_costs):.5f}; per-task share pool/50 at 20% ${0.2 * m['full_large'] / 50:.5f}, at 30% ${0.3 * m['full_large'] / 50:.5f}")
ro = row("cached", 0.2, "route")
print(f"[starvation] Route only at 20%: {ro['succ_mean']:.2f} solved, unstarted {ro['reasons']['unstarted']:.1f}")
st = row("cached", 0.2, "stop")
print(f"[cap] Stop only at 20% solved {st['succ_mean']}, at 30% {row('cached', 0.3, 'stop')['succ_mean']}; "
      f"at 100% capped {row('cached', 1.0, 'stop')['reasons']['cap']:.1f} of 50")
pc = row("cached", 0.45, "paced")
print(f"[spend-down] paced w/o spend-down at 45%: spent ${pc['spent_mean']:.4f} = {pc['spent_mean'] / m['full_large']:.3f} of need, "
      f"solved {pc['succ_mean']:.2f}; at 100% {row('cached', 1.0, 'paced')['succ_mean']:.2f}")
txt = (ART / "optinet/results/fixed_spend_stdout.txt").read_text()
_ho = re.search(r"hybrid minus best of admit and paced\+cap, per pairing and budget: \[(.*?)\] min: (\S+)", txt)
_vals = [float(v) for v in _ho.group(1).split(",")]
print(f"[spend-down held-out] hybrid minus best of admit and paced+cap over {len(_vals)} pairing-budget cells: {_ho.group(1)}")
print(f"[spend-down held-out] unrounded min over cells {float(_ho.group(2)):.4f} (cells above are rounded to 0.1); worst shortfall {-float(_ho.group(2)):.3f} tasks")
print("[pairings]", "; ".join(l for l in txt.splitlines() if re.match(r"0\.[24]0\s+(large|hybrid) ", l)))

ex = (ART / "optinet/results/extras_stdout.txt").read_text()
print("[repricing]", ex.splitlines()[0])
for line in ex.splitlines():
    if line.startswith("est {'scale': 0.25, 'online': False, 'frac': 0.2"):
        print("[wrong estimates]", line)
print("[Table II] large/small batch usd, Large / Paced+SD")
for line in ex.splitlines():
    if line.startswith("price "):
        d = eval(line[6:])  # noqa: S307  (our own printed dict)
        print(f"  {d['large']}/{d['small']} small share {d['batch_small_usd'] / d['batch_large_usd']:.3f} "
              f"frac {d['frac']}: {d['large_only']:.1f} / {d['paced_spend_down']:.1f}")
st1 = X.get("single_turn") or X
print("[single-turn] keys:", list(st1)[:6])
esc = (ART / "results/escalation_stdout.txt").read_text()
print("[restart vs handoff]", re.search(r"tally \{'restart': \d+, 'full': \d+\}.*?fired \d+", esc).group(0)[:80])
readme = (ART / "README.md").read_text()
print("[billing]", re.search(r"account for [0-9.]+% of the billed amount", readme).group(0))
print("[low-effort log]", re.search(r"cut short \(74 of about 500 steps\)", readme).group(0))

_R = json.loads((ART / "optinet/results/fixed_spend.json").read_text())
_held = []
for _x in _R["pairings"]:
    if (_x["small"], _x["large"]) == ("small_cache_budget", "frontier_cache"):
        continue
    for _f in (0.2, 0.4, 0.8, 1.0):
        _rr = [r for r in _x["rows"] if abs(r["frac"] - _f) < 1e-9]
        _h = next(r["succ_mean"] for r in _rr if r["policy"] == "hybrid")
        _held.append(_h - max(r["succ_mean"] for r in _rr if r["policy"] in ("admit", "paced+cap")))
print(f"[spend-down held-out five] hybrid minus best of admit and paced+cap over {len(_held)} cells of the five pairings other than (small_cache_budget, frontier_cache): unrounded min {min(_held):.4f}; worst shortfall {-min(_held):.3f} tasks")

# ---- round 3: large-model tokens per solved task across the headline budgets ----
for _f in (0.2, 0.4, 0.8, 1.0):
    _r, _l = row("causal_cached", _f, "hybrid"), row("causal_cached", _f, "large")
    _h, _g = _r["tok_per_solved"] * _r["tok_large_share"], _l["tok_per_solved"] * _l["tok_large_share"]
    print(f"[work-all] {_f:.0%} large-model tokens per solved: paced {_h / 1e3:.1f}k, Large {_g / 1e3:.1f}k, "
          f"reduction {1 - _h / _g:.3f}; total per solved paced {_r['tok_per_solved'] / 1e3:.1f}k vs Large {_l['tok_per_solved'] / 1e3:.1f}k")
