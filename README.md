# Agent cost artifact: recorded multi-turn tool-calling runs and their cost analyses

This package recomputes every benchmark number and figure in the accompanying papers from recorded agent runs. It makes no API calls and needs only Python 3.10+ and matplotlib.

```bash
pip install -r requirements.txt
python3 analysis/accounting.py          # token and dollar accounting -> results/accounting.json
python3 analysis/accounting_figures.py  # token mix, cost per success, prompt growth, spend mix -> results/*.pdf
python3 analysis/placement.py           # small-first placement, footprint by tier, break-even energy ratio -> results/placement.json
python3 replay.py                       # budget guard and cache-health monitor -> results/replay_numbers.json (about 3 min)
python3 escalation.py                   # escalation triggers, handoff pricing, overhead -> results/escalation_numbers.json
python3 figures.py                      # escalation figures -> results/fig-cost-success.pdf, results/fig-task39.pdf
```

Results from a reference run are included in `results/`. Rerunning reproduces them exactly, except the measured per-step overhead in `escalation.py`, which depends on the machine.

## What the data is

- `logs/`: 50 tasks from the Berkeley Function Calling Leaderboard (BFCL) v4, `multi_turn_base` category, at a pinned repository commit, sampled with seed 42. Each task was run once by a small model (Claude Haiku 5.5) and once by a large model (Claude Sonnet 5.5), each with and without prompt caching: 200 runs, 2,029 model requests.
  - `steps_<config>.jsonl`: one record per model request: task, turn, step, model, the four provider-reported token counters (fresh input, cache write, cache read, output), cost, tool calls, tool errors, latency.
  - `tasks_<config>.jsonl`: one record per task: BFCL checker verdict, failure point, turns, steps, cost.
  - Configs: `small`, `small_cache_budget` (cached small model; this run also capped tool calls at five per turn), `frontier`, `frontier_cache`.
- `prices.json`: list prices per million tokens, verified on the run date (2026-10-14), including the small model's prompt-length price tiers.

The logs contain task identifiers and usage counters only, not BFCL task text or model outputs. Costs are token counts times list prices; they are not reconciled against invoices. Tools in BFCL are simulated, so tool charges are zero.

## Checks

`replay.py` checks the logs before use: per-step costs must sum to each task's recorded cost, and re-pricing the uncached runs from their token counters must reproduce the recorded bills.

## Replay fidelity

| Analysis | What it changes | Fidelity |
|---|---|---|
| Accounting | nothing; reads the records | exact |
| Budget guard | truncates a run; a stopped run fails | exact |
| Cache repair | billing of later steps only | exact for cost |
| Escalation, restart | small-model prefix, then the large model's recorded run | exact up to sampling |
| Escalation, full-history handoff | small prefix, cold cache write, large model's later turns | bounded counterfactual |
| Energy break-even | token footprint by tier under stated weights | parametric; no energy was measured |

## Limitations

One benchmark category, one provider family, 50 tasks, one run per task. Production observations reported in the papers are aggregates from provider usage records and are not part of this package, because the underlying records contain user data.

## License

MIT (see `LICENSE`).
