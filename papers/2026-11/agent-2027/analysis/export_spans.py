"""Reference exporter for the span schema of Table I, applied to Workload 1.

Turns one logged BFCL step record into the span attributes of Table I, including the inferred retry link
(retry.of_span_id: the step right after a step whose tool call errored repairs that step) and cost.usage_status.
`span_attributes` is a plain function that an OpenTelemetry SpanProcessor could call on span end; here it is applied
offline to the recorded step logs.

python3 -I export_spans.py -> out/export_spans_stdout.txt, out/export_span_example.json
Reads only the Workload 1 logs and prices.json at the repository root (no API calls).
"""
import json
from collections import defaultdict
import os
from pathlib import Path

LOGS = Path(os.environ.get("AGENT_ARTIFACT_ROOT") or Path(__file__).resolve().parents[4]) / "logs"
PRICES = json.load(open(LOGS.parent / "prices.json"))
MODELS = PRICES["models"]
PRICE_VERSION = "list-" + PRICES["_meta"]["verified_date"]
OUT = Path(__file__).resolve().parent / "out"
CONFIGS = ["frontier", "frontier_cache", "small", "small_cache_budget", "small_low"]


def rates(model, prompt_len):
    p = MODELS.get(model)
    if p is None:
        return None
    if "tier_low" in p:
        p = p["tier_low"] if prompt_len <= 100_000 else p["tier_high"]
    return p


def span_attributes(step, prev_step=None):
    """One step record (and the previous step of the same task, if any) -> a dict of Table I attributes."""
    u = step.get("usage")
    sid = f'{step["task_id"]}/t{step["turn"]}/s{step["step"]}'
    a = {"cost.unit_id": step["task_id"], "span_id": sid,
         "parent_span_id": f'{step["task_id"]}/t{step["turn"]}',
         "step.kind": "errored_tool_call" if step["kind"] == "failed" else step["kind"],
         "tool.error": step["kind"] == "failed",
         "gen_ai.request.model": step["model"], "cost.price_version": PRICE_VERSION,
         "latency_s": round(step["latency_s"], 3)}
    if prev_step is not None and prev_step["kind"] == "failed":
        a["retry.of_span_id"] = f'{prev_step["task_id"]}/t{prev_step["turn"]}/s{prev_step["step"]}'
    r = rates(step["model"], (u or {}).get("input_tokens", 0) + (u or {}).get("cache_read_tokens", 0) + (u or {}).get("cache_write_tokens", 0))
    if not u or r is None:
        # never zero: an unknown span has a null cost and a reason
        a.update({"cost.usd": None, "cost.usage_status": "unknown",
                  "cost.unknown_reason": "no_usage" if not u else "model_not_in_price_version"})
        return a
    inp, cr, cw, out = u["input_tokens"], u["cache_read_tokens"], u["cache_write_tokens"], u["output_tokens"]
    # OpenTelemetry convention: input tokens include cache reads and writes; the logs hold uncached input separately.
    a["gen_ai.usage.input_tokens"] = inp + cr + cw
    a["gen_ai.usage.cache_creation.input_tokens"] = cw
    a["gen_ai.usage.cache_read.input_tokens"] = cr
    a["gen_ai.usage.output_tokens"] = out
    a["cost.usd"] = (inp * r["input_per_mtok"] + cw * r["cache_write_5min_per_mtok"]
                     + cr * r["cache_read_per_mtok"] + out * r["output_per_mtok"]) / 1e6
    a["cost.usage_status"] = "reported"
    return a


if __name__ == "__main__":
    n = links = 0
    worst = 0.0
    example = None
    for cfg in CONFIGS:
        by_task = defaultdict(list)
        for l in open(LOGS / f"steps_{cfg}.jsonl"):
            s = json.loads(l)
            by_task[s["task_id"]].append(s)
        for tid, steps in by_task.items():
            steps.sort(key=lambda s: s["step"])
            prev = None
            for s in steps:
                a = span_attributes(s, prev)
                n += 1
                links += "retry.of_span_id" in a
                worst = max(worst, abs(a["cost.usd"] - s["cost_usd"]))
                if cfg == "frontier" and tid == "multi_turn_base_62" and s["step"] == 8:
                    example = a
                prev = s
    json.dump(example, open(OUT / "export_span_example.json", "w"), indent=1)
    print("spans exported:", n)
    print("spans with an inferred retry link:", links)
    print("max |exported cost.usd - logged cost_usd| (USD):", worst)
    print("example (frontier, multi_turn_base_62, step 8, the repair of the errored step 7):")
    print(json.dumps(example, indent=1))
