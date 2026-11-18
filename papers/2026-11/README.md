# Analysis code for the November 2026 papers

Each folder below holds the analysis scripts of one paper, the outputs those scripts wrote for the paper, and the figures. Every script reads the recorded runs in this repository (`logs/`, `prices.json`, `optinet/`, `results/`) or the published production aggregates in `production-aggregates/`. A few scripts also read a private event stream; they are marked below, and their saved outputs are published. No script makes API calls.

Run every command from the repository root. Python 3.10+; the figure scripts need matplotlib and `query_snippet.py` needs pandas. Scripts with `-I` in the command use only the standard library; run the figure scripts and `query_snippet.py` without `-I` so that user-installed packages are found. Saved stdout files are the script's standard output redirected to the file named in the Outputs column.

| Folder | Paper |
|---|---|
| `agent-2027/` | AGENT 2027 paper on per-step cost spans for agents (Workload 1: recorded multi-turn runs; Workload 2: the deployed Litos agent) |
| `optinet-2026-camera-ready/` | OPTINET 2026 camera-ready: budget pacing under a fixed spend (extends `optinet/` at the repository root) |
| `bpod-2026-poster/` | BPOD 2026 poster camera-ready: cost accounting of the recorded single-turn and multi-turn runs |
| `production-aggregates/` | Aggregate exports of the Litos deployment shared by the scripts above (see the last section) |

## Private inputs

The Litos event stream (`ops-events-2026-10-23_27.ndjson`, per-operation records of the deployed agent) holds per-user operation records and is not published. Scripts that need it read its path from the `LITOS_EVENTS` environment variable, for example `LITOS_EVENTS=/path/to/ops-events-2026-10-23_27.ndjson python3 -I papers/2026-11/agent-2027/analysis/prod.py`. Without it they stop with a message that names the published output to start from. Every Litos analysis covers the operation-tagged window, events at or after 2026-10-23 14:58 UTC.

## agent-2027

In the commands, `A=papers/2026-11/agent-2027/analysis`. Outputs are in `$A/out/` unless noted.

| Script | Command | Outputs | Inputs |
|---|---|---|---|
| `bfcl.py` | `python3 -I $A/bfcl.py` | `bfcl.json`, `bfcl_stdout.txt` | `logs/`, `prices.json` |
| `prod.py` | `LITOS_EVENTS=<private file> python3 -I $A/prod.py` | `prod.json`, `prod_stdout.txt` | **private** event stream; `production-aggregates/litos-console-daily-2026-07-30_2026-10-27.csv` |
| `bootstrap_bfcl.py` | `python3 -I $A/bootstrap_bfcl.py` | `bootstrap_bfcl.json`, `bootstrap_bfcl_stdout.txt` | `logs/` |
| `bootstrap_carried.py` | `python3 -I $A/bootstrap_carried.py` | `bootstrap_carried.json`, `bootstrap_carried_stdout.txt` | `logs/`, `prices.json` |
| `bootstrap_modes.py` | `python3 -I $A/bootstrap_modes.py` | `bootstrap_modes.json`, `bootstrap_modes_stdout.txt` | `logs/` |
| `bootstrap_litos_shares.py` | `LITOS_EVENTS=<private file> python3 -I $A/bootstrap_litos_shares.py` | `bootstrap_litos_shares.json`, `bootstrap_litos_shares_stdout.txt` | **private** event stream (through `prod.py`), `logs/` |
| `bootstrap_shares.py` | `LITOS_EVENTS=<private file> python3 -I $A/bootstrap_shares.py` | `bootstrap_shares.json`, `bootstrap_shares_stdout.txt` | **private** event stream, `logs/` |
| `bootstrap_error_alert.py` | `python3 -I $A/bootstrap_error_alert.py` | `bootstrap_error_alert.json`, `bootstrap_error_alert_stdout.txt` | `logs/` |
| `alert_holdout.py` | `python3 -I $A/alert_holdout.py` | `alert_holdout.json`, `alert_holdout_stdout.txt` | `logs/` |
| `litos_holdout.py` | `LITOS_EVENTS=<private file> python3 -I $A/litos_holdout.py` | `litos_holdout.json`, `litos_holdout_stdout.txt` | **private** event stream, `logs/` |
| `retry_like_for_like.py` | `LITOS_EVENTS=<private file> python3 -I $A/retry_like_for_like.py` | `retry_like_for_like.json`, `retry_like_for_like_stdout.txt` | **private** event stream (through `prod.py`) |
| `export_package_aggregate.py` | `LITOS_EVENTS=<private file> python3 -I $A/export_package_aggregate.py` | `litos_package_aggregate.csv` | **private** event stream (through `prod.py`) |
| `recompute_from_aggregate.py` | `python3 -I $A/recompute_from_aggregate.py` | `recompute_from_aggregate_stdout.txt` | `out/litos_package_aggregate.csv` |
| `export_spans.py` | `python3 -I $A/export_spans.py` | `export_span_example.json`, `export_spans_stdout.txt` | `logs/`, `prices.json` |
| `span_split.py` | `python3 -I $A/span_split.py` | `span_split_stdout.txt` | `logs/`, `prices.json` |
| `query_snippet.py` | `python3 $A/query_snippet.py` | `query_snippet_stdout.txt` | `logs/` (needs pandas) |
| `derived.py` | `python3 -I $A/derived.py` | `derived_stdout.txt` | `out/bfcl.json`, `out/prod.json`, `logs/`, `production-aggregates/console-cost-by-key-model.csv` |
| `fig_waterfall.py` | `python3 $A/fig_waterfall.py` | `../fig-waterfall.pdf`, `../fig-packet.pdf`, `fig_stdout.txt` | `logs/`, `out/prod.json` |
| `fig_daily.py` | `python3 $A/fig_daily.py` | `../fig-daily.pdf`, `fig_daily_stdout.txt` | `production-aggregates/litos-console-daily-2026-07-30_2026-10-27.csv` |

Run order: `bfcl.py; prod.py; bootstrap_bfcl.py; bootstrap_carried.py; bootstrap_modes.py; bootstrap_litos_shares.py; bootstrap_shares.py; bootstrap_error_alert.py; alert_holdout.py; litos_holdout.py; retry_like_for_like.py; export_package_aggregate.py; recompute_from_aggregate.py; export_spans.py; span_split.py; query_snippet.py; derived.py; fig_waterfall.py; fig_daily.py`. Without the private stream, skip the six scripts marked private: `out/prod.json` (counts, dollars, timings and workload labels only) is the published starting point for `derived.py` and `fig_waterfall.py`, and `out/litos_package_aggregate.csv` (258 application packages: rank by start time, outcome, dollars and call counts by call role; no identifiers or timestamps) is the published input of `recompute_from_aggregate.py`, which recomputes the Litos application-package rows of Tables III to V and the time-ordered hold-out. `$A/README.md` describes each script.

## optinet-2026-camera-ready

In the commands, `O=papers/2026-11/optinet-2026-camera-ready/analysis`. Saved stdout files sit next to the scripts in `analysis/`.

| Script | Command | Outputs | Inputs |
|---|---|---|---|
| `paired_and_retry.py` | `python3 -I $O/paired_and_retry.py` | `paired_and_retry.json`, `paired_and_retry_stdout.txt` | `logs/`, `optinet/fixed_spend.py`, `optinet/results/fixed_spend.json` |
| `artifact_numbers.py` | `python3 -I $O/artifact_numbers.py` | `artifact_numbers_stdout.txt` | `optinet/results/`, `results/escalation_stdout.txt`, `logs/`, root `README.md` |
| `revision_numbers.py` | `python3 -I $O/revision_numbers.py` | `revision_numbers_stdout.txt` | `optinet/results/`, `analysis/paired_and_retry.json` |
| `sweep_checks.py` | `python3 -I $O/sweep_checks.py` | `sweep_checks_stdout.txt` | `logs/`, `optinet/` (through `paired_and_retry.py`) |
| `tight_pool_paired.py` | `python3 -I $O/tight_pool_paired.py` | `tight_pool_paired_stdout.txt` | `logs/`, `optinet/` (through `paired_and_retry.py`) |
| `nochecker_cascade.py` | `python3 -I $O/nochecker_cascade.py` | `nochecker_cascade_stdout.txt` | `logs/`, `optinet/` (through `paired_and_retry.py`) |
| `table3_intervals.py` | `python3 -I $O/table3_intervals.py` | `table3_intervals_stdout.txt` | `logs/`, `logs/single_turn/`, `optinet/extras.py` |
| `make_figures.py` | `python3 $O/make_figures.py` | `figs/fig-budget-curve.{pdf,png}`, `figs/fig-causes.{pdf,png}` | `optinet/results/fixed_spend.json`, `logs/` |
| `production_numbers.py` | `python3 -I $O/production_numbers.py` | `production_numbers_stdout.txt` | `production-aggregates/`: `litos-console-daily-2026-07-30_2026-10-27.csv`, `deep_history.json`, `console-requests-by-key-daily-2026-06-02_2026-10-27.csv`, `console-cost-by-key-model.csv`, `ops_models_tagged_window.json` |
| `billing_reconcile.py` | `python3 -I $O/billing_reconcile.py` | `billing_reconcile_stdout.txt` | `logs/`, `logs/single_turn/`, `production-aggregates/console-cost-by-key-model.csv`, `production-aggregates/console-usage-by-key-monthly-2026-06-02_2026-10-27.csv` |
| `ops_tagged_window.py` | `LITOS_EVENTS=<private file> python3 -I $O/ops_tagged_window.py` | `ops_tagged_window_stdout.txt` | **private** event stream |

Run `paired_and_retry.py` before `revision_numbers.py`. `test_runs.txt` records the unit-test run of `optinet/` (`python3 -I -m unittest discover -s optinet`) and a mutation check of the reserve rule. `billing_reconcile.py` is the same script as in `bpod-2026-poster/`.

## bpod-2026-poster

In the commands, `B=papers/2026-11/bpod-2026-poster/analysis`.

| Script | Command | Outputs | Inputs |
|---|---|---|---|
| `recompute.py` | `python3 -I $B/recompute.py` | `analysis/recompute.json`, `recompute_stdout.txt` | `logs/`, `logs/single_turn/`, `prices.json` |
| `figure.py` | `python3 $B/figure.py` | `fig-mix-frontier.pdf` | `analysis/recompute.json` |
| `billing_reconcile.py` | `python3 -I $B/billing_reconcile.py` | `billing_reconcile_stdout.txt` | `logs/`, `logs/single_turn/`, `production-aggregates/console-cost-by-key-model.csv`, `production-aggregates/console-usage-by-key-monthly-2026-06-02_2026-10-27.csv` |

## production-aggregates

The Litos raw records (the per-operation event stream and per-user product records) are private and are not in this repository. The files below are the aggregate exports the papers report from. They contain only dates, period labels, API key names, model names, token counts, request counts, dollar amounts and derived ratios. They contain no emails, names, user ids, operation ids, job or company names of end users, URLs or keys. One API key that is not a Litos product key appears under the neutral name `personal-agent-key` (column `usd_key_personal_agent_key` in the daily file); every Litos figure excludes it.

| File | Content | Used by |
|---|---|---|
| `litos-console-daily-2026-07-30_2026-10-27.csv` | provider daily cost report: dollars and tokens by token category, dollars per API key, one row per date | `agent-2027/prod.py`, `fig_daily.py`; `optinet-2026-camera-ready/production_numbers.py` |
| `console-cost-by-key-model.csv` | provider cost by API key name and model, per 30-day period | `agent-2027/derived.py`; `bpod-2026-poster/billing_reconcile.py`; `optinet-2026-camera-ready/production_numbers.py`, `billing_reconcile.py` |
| `console-usage-by-key-monthly-2026-06-02_2026-10-27.csv` | provider token and request counts by API key name and model, per 30-day period | `billing_reconcile.py` (both folders) |
| `console-requests-by-key-daily-2026-06-02_2026-10-27.csv` | provider request counts per day for the four Litos product keys | `optinet-2026-camera-ready/production_numbers.py` |
| `deep_history.json` | request counts, tokens per request, cache shares and dollars per 1,000 requests per 30-day period | `optinet-2026-camera-ready/production_numbers.py` |
| `ops_models_tagged_window.json` | model names seen in the provider usage records of the operation-tagged window (no counts or dollars) | `optinet-2026-camera-ready/production_numbers.py` |

`agent-2027/analysis/out/prod.json` and `agent-2027/analysis/out/litos_package_aggregate.csv` are the aggregate outputs of the private-input scripts (see the agent-2027 section).

## Reproduction check

Every script above was run from the repository root and its outputs compared with the outputs saved for the paper. Figures were compared as rendered pages. The private-input scripts were rerun on the author's machine with `LITOS_EVENTS` set.

| Folder | Result |
|---|---|
| `agent-2027` | All JSON, CSV and stdout outputs identical; the three figures render identically. `query_snippet_stdout.txt` was saved with an older NumPy; NumPy 2 prints the same values wrapped as `np.float64(...)`. |
| `optinet-2026-camera-ready` | All JSON and stdout outputs identical; both figures render identically (PDF) and the PNG files are byte-identical. |
| `bpod-2026-poster` | All JSON and stdout outputs identical; the figure renders identically. |
