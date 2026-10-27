"""Run the BFCL multi-turn harness against a local open-weight model served by Ollama.

The agent loop, tools, simulators and BFCL checks are the bundled harness in edge/harness, unchanged
from the hosted-model runs; only the model client differs. Each request's wall-clock start and end are logged so that
power samples from macOS powermetrics can be attributed to it (see edge/energy.py).

    ollama pull qwen3:8b
    python3 edge/run_edge.py --model qwen3:8b --max-tasks 50 --out edge/runs
"""
import argparse
import json
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

OLLAMA = "http://localhost:11434/api/chat"


class OllamaClient:
    """Translates the harness's Anthropic-style messages and tools to Ollama's chat API."""

    def __init__(self, model: str, num_ctx: int, timing: list):
        self.model, self.num_ctx, self.timing = model, num_ctx, timing
        self.n = 0

    @staticmethod
    def _tools(tools):
        return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["input_schema"]}} for t in tools]

    @staticmethod
    def _messages(system, messages):
        out = [{"role": "system", "content": system}]
        names = {}
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
                continue
            if m["role"] == "assistant":
                text = "".join(b.get("text", "") for b in content if b["type"] == "text")
                calls = [b for b in content if b["type"] == "tool_use"]
                for b in calls:
                    names[b["id"]] = b["name"]
                msg = {"role": "assistant", "content": text}
                if calls:
                    msg["tool_calls"] = [{"function": {"name": b["name"], "arguments": b["input"]}} for b in calls]
                out.append(msg)
            else:
                for b in content:
                    if b["type"] == "tool_result":
                        out.append({"role": "tool", "tool_name": names.get(b["tool_use_id"], ""),
                                    "content": b["content"]})
                    elif b["type"] == "text":
                        out.append({"role": "user", "content": b["text"]})
        return out

    def call(self, system, messages, tools, ctx):
        body = {"model": self.model, "messages": self._messages(system, messages), "tools": self._tools(tools),
                "stream": False, "think": False,
                "options": {"num_ctx": self.num_ctx, "temperature": 0, "num_predict": 2048}}
        t0 = time.time()
        req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.loads(r.read())
        t1 = time.time()
        msg = resp.get("message", {})
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": fn.get("name", ""), "arguments": args, "id": f"call_{self.n}_{i}"})
        self.n += 1
        text = msg.get("content", "") or ""
        content = [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]} for c in calls]
        if text and not calls:
            content = [{"type": "text", "text": text}]
        processed, generated = resp.get("prompt_eval_count", 0), resp.get("eval_count", 0)
        self.timing.append({"task_id": ctx["task"]["id"], "turn": ctx["turn"], "step_in_turn": ctx["step_in_turn"],
                            "t_start": t0, "t_end": t1, "prompt_eval_count": processed, "eval_count": generated,
                            "prompt_eval_s": resp.get("prompt_eval_duration", 0) / 1e9,
                            "eval_s": resp.get("eval_duration", 0) / 1e9,
                            "load_s": resp.get("load_duration", 0) / 1e9})
        usage = {"input_tokens": processed, "output_tokens": generated, "cache_read_tokens": 0, "cache_write_tokens": 0}
        return {"tool_calls": calls, "text": text, "stop_reason": "tool_use" if calls else "end_turn",
                "content": content}, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default=str(Path(__file__).parent / "harness"))
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-tasks", type=int, default=50)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "runs"))
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    harness = Path(args.harness).expanduser()
    sys.path.insert(0, str(harness))
    import run_multiturn as H  # noqa: E402

    tasks = [json.loads(l) for l in (harness / "bfcl_v4_multi_turn.jsonl").read_text().splitlines() if l.strip()]
    tasks = tasks[args.skip: args.skip + args.max_tasks]
    run_dir = Path(args.out) / (args.run_id or time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"model": args.model, "provider": "ollama", "prompt_caching": False, "max_tool_calls": 12,
           "pricing_key": "local-edge"}
    pricing = {"models": {"local-edge": {"input_per_mtok": 0.0, "output_per_mtok": 0.0}}}
    done = set()
    tpath = run_dir / "tasks_edge.jsonl"
    if tpath.exists():
        done = {json.loads(l)["task_id"] for l in tpath.read_text().splitlines() if l.strip()}
    (run_dir / "run_meta.json").write_text(json.dumps({"model": args.model, "num_ctx": args.num_ctx, "temperature": 0,
                                                       "max_steps_per_turn": H.MAX_STEPS_PER_TURN, "max_tool_calls": 12},
                                                      indent=2))
    run_args = argparse.Namespace(max_spend_usd=float("inf"))
    for i, task in enumerate(tasks):
        if task["id"] in done:
            continue
        timing, step_logs = [], []
        client = OllamaClient(args.model, args.num_ctx, timing)
        row = H.run_task(task, cfg, "edge", pricing, client, run_args, step_logs, {"total": 0.0})
        with open(tpath, "a") as f:
            f.write(json.dumps(row) + "\n")
        with open(run_dir / "steps_edge.jsonl", "a") as f:
            f.writelines(json.dumps(asdict(s)) + "\n" for s in step_logs)
        with open(run_dir / "timing_edge.jsonl", "a") as f:
            f.writelines(json.dumps(t) + "\n" for t in timing)
        print(f"{i + 1}/{len(tasks)} {task['id']} success={row['success']} steps={row['llm_steps']} "
              f"failed={row['failed_steps']} {sum(t['t_end'] - t['t_start'] for t in timing):.0f}s", flush=True)


if __name__ == "__main__":
    main()
