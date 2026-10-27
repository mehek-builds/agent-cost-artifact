"""Model- and context-efficiency levers priced per successful task, from the recorded runs.

python3 analysis/levers.py  -> prints a table, writes results/levers.json

Levers: a smaller model, lower reasoning effort, prompt caching, a tool-call budget, and
context compression. Compression was not run; its break-even is computed from the recorded
token counts: a method that compresses the whole prompt anew at every step (so nothing can be
served from cache) must remove a fraction r* of prompt tokens to match the cached bill, and a
method that compresses only the stable prefix once (cache-preserving) can save at most the
compressed share of cache-read and cache-write dollars.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from replay import card, mcnemar_exact, wilson  # noqa: E402

LOGS = ROOT / "logs"
CONFIGS = {"small": "small", "small, low effort": "small_low", "small+cache (5-call cap)": "small_cache_budget",
           "large": "frontier", "large, low effort": "frontier_low", "large+cache": "frontier_cache"}


def tasks(log):
    return {t["task_id"]: t for t in map(json.loads, open(LOGS / f"tasks_{log}.jsonl"))}


def steps(log):
    return [json.loads(l) for l in open(LOGS / f"steps_{log}.jsonl")]


def dollars_by_kind(st):
    out = {"input": 0.0, "cache_write": 0.0, "cache_read": 0.0, "output": 0.0, "prompt_at_fresh_price": 0.0}
    for s in st:
        u = s["usage"]
        p = u["input_tokens"] + u["cache_read_tokens"] + u["cache_write_tokens"]
        c = card(s["model"], p)
        out["input"] += u["input_tokens"] * c["input_per_mtok"] / 1e6
        out["cache_write"] += u["cache_write_tokens"] * c["cache_write_5min_per_mtok"] / 1e6
        out["cache_read"] += u["cache_read_tokens"] * c["cache_read_per_mtok"] / 1e6
        out["output"] += u["output_tokens"] * c["output_per_mtok"] / 1e6
        out["prompt_at_fresh_price"] += p * c["input_per_mtok"] / 1e6
    return out


def main():
    res = {}
    for label, log in CONFIGS.items():
        t = tasks(log)
        k = sum(int(x["success"]) for x in t.values())
        cost = sum(x["cost_usd"] for x in t.values())
        r = {"successes": k, "wilson95": wilson(k, len(t)), "cost_per_task": cost / len(t), "cost_per_success": cost / k}
        st = steps(log)
        if len(st) >= 400:  # the large low-effort step log is truncated; task-level costs above are complete
            d = dollars_by_kind(st)
            tot = d["input"] + d["cache_write"] + d["cache_read"] + d["output"]
            r["dollars"] = d
            r["dollar_share"] = {kk: d[kk] / tot for kk in ("input", "cache_write", "cache_read", "output")}
            if d["cache_read"] > 0:
                r["compression_break_even_r"] = 1 - (d["input"] + d["cache_write"] + d["cache_read"]) / d["prompt_at_fresh_price"]
                r["cache_preserving_max_saving_at_50pct"] = 0.5 * (d["cache_read"] + d["cache_write"]) / tot
        res[label] = r
    for a, b in (("small", "small, low effort"), ("large", "large, low effort")):
        ta, tb = tasks(CONFIGS[a]), tasks(CONFIGS[b])
        n10 = sum(1 for i in ta if ta[i]["success"] and not tb[i]["success"])
        n01 = sum(1 for i in ta if tb[i]["success"] and not ta[i]["success"])
        res[f"effort {a}"] = {"medium_only": n10, "low_only": n01, "mcnemar_p": mcnemar_exact(n10, n01),
                              "cost_change": res[b]["cost_per_task"] / res[a]["cost_per_task"] - 1}
    (ROOT / "results" / "levers.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
