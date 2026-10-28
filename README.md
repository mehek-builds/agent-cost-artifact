# Agent cost artifact: recorded multi-turn tool-calling runs and their cost analyses

This package recomputes every benchmark number and figure reported from these recorded agent runs. It makes no API calls and needs only Python 3.10+ and matplotlib.

```bash
pip install -r requirements.txt
python3 analysis/accounting.py          # token and dollar accounting -> results/accounting.json
python3 analysis/accounting_figures.py  # token mix, cost per success, prompt growth, spend mix -> results/*.pdf
python3 analysis/placement.py           # small-first placement, footprint by tier, break-even energy ratio -> results/placement.json
python3 replay.py                       # budget guard and cache-health monitor -> results/replay_numbers.json (about 3 min)
python3 escalation.py                   # escalation triggers, handoff pricing, overhead -> results/escalation_numbers.json
python3 figures.py                      # escalation figures -> results/fig-cost-success.pdf, results/fig-task39.pdf
python3 analysis/rigor.py               # routing baselines, held-out trigger selection, run pairings, tool cap, energy with prefill -> results/rigor.json
python3 analysis/rigor2.py              # matched-success frontier, in-run failure predictor, cascade gap, edge-fit intervals, held-out budget -> results/rigor2.json
python3 analysis/rigor2_figure.py       # trigger frontier figure -> results/fig-trigger-frontier.pdf
python3 analysis/transactional_stop.py  # where budget stops land relative to tool-calling turns -> results/transactional_stop.json
python3 analysis/repeats.py             # the runs as repeated rollouts: pooled success, agreement, nine-pairing escalation -> results/repeats.json
python3 analysis/levers.py              # efficiency levers per success: model size, reasoning effort, caching, compression break-even -> results/levers.json
python3 analysis/levers_figure.py       # compression break-even figure -> results/fig-compression.pdf
```

Results from a reference run are included in `results/`. Rerunning reproduces them exactly, except the measured per-step overhead in `escalation.py`, which depends on the machine.

## Budget pacing under a fixed spend (OPTINET 2026)

`optinet/` holds a small library for agents that share one spend pool, and the analyses of
the OPTINET 2026 paper "What Breaks First Under a Fixed Spend? Budget-Paced Routing and
Early Stopping for LLM Agent Inference". No API calls; Python 3.10+ and matplotlib.

- `optinet/budget_pacer.py`: `BudgetPacer`, the policy as an agent would use it (reserve,
  first-error routing, cap and pool stops, spend-down, one-way escalation; thread-safe
  ledger, online cost estimates). `optinet/example_agent_loop.py` wires it into a
  tool-calling loop and runs offline with a scripted model.
- `optinet/test_budget_pacer.py`: unit tests for every rule, plus a test that replaying the
  recorded traces through `BudgetPacer` reproduces `fixed_spend.py` exactly.

```bash
python3 -m unittest discover -s optinet -p 'test_*.py'   # 11 tests, under a second
python3 optinet/fixed_spend.py   # budget sweep, baselines, bootstrap, rollout pairings -> optinet/results/fixed_spend.json (about 10 min)
python3 optinet/extras.py        # estimate error, other model price pairs, single-turn rollouts -> optinet/results/extras.json
python3 optinet/fig_curve.py     # Fig. 1 -> optinet/results/fig-budget-curve.pdf
python3 optinet/fig_causes.py    # Fig. 2 -> optinet/results/fig-causes.pdf
python3 optinet/example_agent_loop.py
```

`frontier_low` is excluded from step-level replay because its step log is incomplete.
Re-priced runs keep the recorded models' outcomes; only prices change. The paper's main
setting uses causal cost estimates (`OnlineEst` in `fixed_spend.py`, `CostEstimator` in the
library) whose priors come from a different pair of runs and which learn only from tasks
already finished; leave-one-out estimates are reported where the paper says so.

## Measured edge run

`edge/` replays the same 50 tasks with an open-weight model on a local device and measures its energy.

```bash
ollama pull qwen3:8b
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -o edge/runs/NEW/power.txt   # in another terminal
python3 edge/run_edge.py --model qwen3:8b --max-tasks 50 --run-id NEW
python3 edge/energy.py edge/runs/NEW           # joules per request, token and success
python3 edge/placement_edge.py edge/runs/NEW   # placement policies with the measured edge tier
python3 edge/sustainability.py edge/runs/NEW   # energy with cloud PUE, carbon, device amortization, latency per success
```

- `edge/harness/`: the agent loop, BFCL simulators, tools and checker used for every run (BFCL parts under Apache 2.0, see `NOTICE`).
- `edge/runs/full/`: the reported run: Qwen3 8B (4-bit, Ollama, temperature 0, thinking off, 16k context) on an Apple M5 with 34 GB, on AC power, low-power mode off. `power.txt` holds the combined CPU, GPU and neural-engine power samples (0.5 s) for the run window; `baseline_start`/`baseline_end` mark a 2-minute idle window; `timing_edge.jsonl` holds each request's wall-clock window and token counts.
- Large-tier energy is not measured: `placement_edge.py` uses a published estimate of 3 to 4 J per output token (Samsi et al., HPEC 2023, LLaMA 65B on 8 to 32 V100 GPUs, GPU power only), not a bound for the hosted model.

## What the data is

- `logs/`: 50 tasks from the Berkeley Function Calling Leaderboard (BFCL) v4, `multi_turn_base` category, at a pinned repository commit, sampled with seed 42. Each task was run once by a small model (Claude Haiku 5.5) and once by a large model (Claude Sonnet 5.5), each with and without prompt caching: 200 runs, 2,029 model requests.
  - `steps_<config>.jsonl`: one record per model request: task, turn, step, model, the four provider-reported token counters (fresh input, cache write, cache read, output), cost, tool calls, tool errors, latency.
  - `tasks_<config>.jsonl`: one record per task: BFCL checker verdict, failure point, turns, steps, cost.
  - Configs: `small`, `small_cache_budget` (cached small model; this run also capped tool calls at five per turn), `frontier`, `frontier_cache`. `small_low` and `frontier_low` repeat the uncached runs at low reasoning effort; the `frontier_low` step log was cut short (74 of about 500 steps), so only its task-level records are complete.
- `prices.json`: list prices per million tokens, verified on the run date (2026-10-14), including the small model's prompt-length price tiers.

The logs contain task identifiers and usage counters only, not BFCL task text or model outputs. Costs are token counts times list prices. Checked against the provider's billing console for the study's dedicated API key, all logged runs (including runs not in this package) account for 95.8% of the billed amount, with cache-write and cache-read dollars within 0.4%; the remainder is one run whose step log was cut short and one stopped by a credit limit. Tools in BFCL are simulated, so tool charges are zero.

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
| Energy break-even | token footprint by tier under stated weights | parametric (`placement.py`); `edge/` replaces the edge tier with measured energy, the cloud tier stays a published estimate |

## Limitations

One benchmark category, one provider family, 50 multi-turn tasks with three rollouts per model (uncached, cached, low effort; `analysis/repeats.py` pools them), and 150 single-turn tasks with five (large) and eight (small) rollouts in `logs/single_turn/` (task records only, model outputs removed). Production observations reported in the papers are aggregates from provider usage records and are not part of this package, because the underlying records contain user data.

## License

MIT (see `LICENSE`).
