#!/usr/bin/env python3
"""Numbers added in the camera-ready revision round 1 (all read from recorded results, no new simulation).

  - where the paced policy trails running everything small (tight pools, causal estimates)
  - full-task-set mean of Paced+SD against Large at a full pool (vs the bootstrap difference)
  - what the checker-driven cascade spends, and its uncached rows
  - the second workload (150 single-turn tasks) from optinet/results/extras.json
  - the share of bootstrap resamples with zero or negative gain over Small at 20%

    python3 analysis/revision_numbers.py > analysis/revision_numbers_stdout.txt
"""
import json
from pathlib import Path

ART = Path(__file__).resolve().parents[4]  # repository root
HERE = Path(__file__).resolve().parent
R = json.loads((ART / "optinet/results/fixed_spend.json").read_text())
X = json.loads((ART / "optinet/results/extras.json").read_text())
PR = json.loads((HERE / "paired_and_retry.json").read_text())


def row(block, frac, pol):
    return next(r for r in R[block]["rows"] if abs(r["frac"] - frac) < 1e-9 and r["policy"] == pol)


print("[tight pool] Paced+SD (causal estimates) minus Small, mean over 500 orders")
for name in ("uncached", "cached"):
    for f in (0.05, 0.08, 0.10, 0.15, 0.20):
        h, s = row(f"causal_{name}", f, "hybrid")["succ_mean"], row(name, f, "small")["succ_mean"]
        print(f"  {name} {f:.0%}: paced {h:.2f} small {s:.2f} difference {h - s:+.3f}")
print("[full pool] Paced+SD vs Large, mean over 500 orders on the full task set (cached, causal)")
for f in (0.8, 1.0):
    h, l = row("causal_cached", f, "hybrid")["succ_mean"], row("cached", f, "large")["succ_mean"]
    print(f"  {f:.0%}: paced {h:.2f} large {l:.2f} difference {h - l:+.2f}")
print("[bootstrap] 100% Paced+SD - Large:",
      json.dumps(R["bootstrap_causal_cached"]["1.0"]["diff"]["hybrid - large"]))

full = PR["cached"]["full_large"]
for f in ("0.8", "1.0"):
    sp = PR["cached"]["orders"][f"{f}|retry"]["spent_mean"]
    print(f"[cascade cost] cached {float(f):.0%}: retry spent ${sp:.4f} = {sp / full:.3f} of Large's need ${full:.4f}")
for f in ("0.2", "0.4", "0.8", "1.0"):
    for p in ("retry", "retry-paced"):
        print(f"[retry uncached] {float(f):.0%} {p}: mean500 {PR['uncached']['orders'][f'{f}|{p}']['succ_mean']:.2f} "
              f"boot95 {PR['uncached']['bootstrap'][f]['ci95'][p]}")
print("[bootstrap share] cached 20% hybrid - small: share of resamples with difference <= 0 =",
      f"{PR['cached']['bootstrap']['0.2']['diff']['hybrid - small']['share_le0']:.3f}")
print("[order share] cached 20% hybrid - small: share of orders with difference <= 0 =",
      f"{PR['cached']['order_diff_hybrid_small']['0.2']['share_le0']:.3f}")

st = X["single_turn"]
m = st["meta"]
print(f"[single-turn] tasks {m['tasks']}, small rollouts {m['small_rollouts']}, large rollouts {m['large_rollouts']}; "
      f"first-error trigger cannot fire (extras.py: single-turn tasks execute no tools)")
for r in st["rows"]:
    print(f"  {r['frac']:.0%}: small {r['small']['mean']} (spent {r['small']['spent']:.4f}), "
          f"large {r['large']['mean']}, paced {r['paced']['mean']} (spent {r['paced']['spent']:.4f}), "
          f"paced+SD {r['paced_spend_down']['mean']}, admission {r['admission']['mean']} min {r['admission']['min']}")
