# AGENT 2027 analysis scripts

Run every command from the repository root (Python 3, matplotlib for the figures, pandas for `query_snippet.py`). Workload 1 logs and `prices.json` are at the repository root; the released Litos aggregates are in `papers/2026-11/production-aggregates/`. Commands below use `A=papers/2026-11/agent-2027/analysis`.

1. `python3 -I $A/bfcl.py` : Workload 1 tables (spend, step kinds, failure modes, alert check) -> `out/bfcl.json`
2. `LITOS_EVENTS=<private file> python3 -I $A/prod.py` : Litos aggregates -> `out/prod.json` (needs the private event stream)
3. `python3 -I $A/bootstrap_bfcl.py` (pooled and per-group after-flag share), `bootstrap_carried.py`, `bootstrap_modes.py`, `alert_holdout.py` : Workload 1 intervals and the out-of-sample alert check. `bootstrap_litos_shares.py`, `bootstrap_shares.py` and `litos_holdout.py` add the Litos package intervals and the time-ordered Litos hold-out; they need the private event stream, and their outputs are released
3a. `python3 -I $A/bootstrap_error_alert.py` : cost alert versus an errored-tool-call alert and their union, task-level bootstrap (reads the released Workload 1 logs)
3b. `python3 -I $A/retry_like_for_like.py` : the like-for-like retry contrast (needs the private event stream; its output is released)
3c. `python3 -I $A/export_package_aggregate.py` (needs the private stream) writes the de-identified per-package aggregate `out/litos_package_aggregate.csv`; `python3 -I $A/recompute_from_aggregate.py` recomputes the Litos application-package rows of Tables III to V and the time-ordered hold-out from that file alone
3d. `python3 -I $A/export_spans.py` : reference exporter, Workload 1 step records to the Table I span attributes
4. `python3 -I $A/span_split.py`, `python3 $A/query_snippet.py`, `python3 -I $A/derived.py` : span example, query snippet, derived counts
5. `python3 $A/fig_waterfall.py`, `python3 $A/fig_daily.py` : the figures

Scripts that read the private Litos event stream (`prod.py`, `litos_holdout.py`, `bootstrap_litos_shares.py`, `bootstrap_shares.py`, `retry_like_for_like.py`, `export_package_aggregate.py`) read its path from the `LITOS_EVENTS` environment variable and stop with a message when it is absent. They are released with their saved outputs in `out/`; the stream itself is not. The Litos application-package rows of Tables III to V and the time-ordered hold-out recompute from `out/litos_package_aggregate.csv` with `recompute_from_aggregate.py`; the other Litos values (resume parse and run rows, coverage, the incident figures, Figs. 2 and 5) need the private stream, and only the daily token cost can be checked against the released console cost reports. The Litos scripts read the instrumented operations from roll-out onward (events at or after 2026-10-23 14:58 UTC), in which every usage record is priced from its own fields.
