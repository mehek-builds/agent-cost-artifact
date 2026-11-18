# AGENT 2027 analysis scripts

Run from this folder (Python 3, matplotlib for the figures). Paths resolve from the repository root (this folder is `papers/2026-11/agent-2027/analysis`, so the root is four levels up); Workload 1 logs and `prices.json` are there. Override with the environment variables `AGENT_ARTIFACT_ROOT` (repository root), `AGENT_AGGREGATES_DIR` (folder with the released console CSVs, default `papers/2026-11/production-aggregates`) and `LITOS_PRIVATE_DIR` (folder with the private event stream, not released).

1. `python3 -I bfcl.py` : Tables of Workload 1 (spend, step kinds, failure modes, alert check) -> `out/bfcl.json`
2. `python3 -I prod.py` : Litos aggregates -> `out/prod.json` (needs the private event stream; the released aggregates are in `papers/2026-11/production-aggregates`)
3. `python3 -I bootstrap_litos_shares.py` (Litos package intervals, private stream), `bootstrap_bfcl.py` (pooled and per-group after-flag share), `bootstrap_carried.py`, `bootstrap_modes.py`, `bootstrap_shares.py`, `alert_holdout.py`, `litos_holdout.py` : intervals, the out-of-sample alert check and the time-ordered Litos hold-out (the last needs the private event stream; its output is released)
3a. `python3 -I bootstrap_error_alert.py` : cost alert versus an errored-tool-call alert and their union, task-level bootstrap (reads the released Workload 1 logs)
3b. `python3 -I retry_like_for_like.py` : the like-for-like retry contrast (needs the private event stream; its output is released)
3c. `python3 -I export_package_aggregate.py` (needs the private stream) writes the de-identified per-package aggregate `out/litos_package_aggregate.csv`; `python3 -I recompute_from_aggregate.py` recomputes the Litos application-package rows of Tables III to V and the time-ordered hold-out from that file alone
3d. `python3 -I export_spans.py` : reference exporter, Workload 1 step records to the Table I span attributes
4. `python3 -I span_split.py`, `query_snippet.py`, `derived.py` : span example, query snippet, derived counts
5. `python3 fig_waterfall.py`, `fig_daily.py` : the figures

Scripts that read the private Litos event stream (`prod.py`, `litos_holdout.py`, `bootstrap_litos_shares.py`, `bootstrap_shares.py`, `retry_like_for_like.py`, `export_package_aggregate.py`) are released with their saved outputs in `out/`; the stream itself is not. The Litos application-package rows of Tables III to V and the time-ordered hold-out recompute from `out/litos_package_aggregate.csv` with `recompute_from_aggregate.py`; the other Litos values (resume parse and run rows, coverage, the incident figures, Figs. 2 and 5) need the private stream, and only the daily token cost can be checked against the released console cost reports. The Litos scripts read the instrumented operations from roll-out onward (events at or after 2026-10-23 14:58 UTC), in which every usage record is priced from its own fields.
