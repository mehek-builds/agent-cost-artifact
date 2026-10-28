#!/usr/bin/env python3
"""Wiring BudgetPacer into a tool-calling agent loop.

`run_batch` below is the whole integration: the agent asks the pacer which model to use
before each task, checks it before each step, reports each step's cost and tool errors
after it, and restarts the task when the pacer escalates. `call_model` is any function that
sends one request to the named model and returns the provider's usage counters, whether a
tool call failed and whether the task finished; swap in your own client.

The demo at the bottom runs offline with a scripted fake model, so it needs no API key:

    python3 optinet/example_agent_loop.py
"""
from __future__ import annotations

import random

from budget_pacer import LARGE, SMALL, Action, BudgetPacer, CostEstimator, PriceCard, cost_of

PRICES = {  # USD per million tokens (input, cache write, cache read, output); set yours
    SMALL: PriceCard(0.10, 0.125, 0.01, 0.50),
    LARGE: PriceCard(2.00, 2.50, 0.10, 10.0),
}


def run_batch(tasks, pool_usd, call_model, grade, estimator):
    pacer = BudgetPacer(pool_usd, estimator)
    results = []
    for i, task in enumerate(tasks):
        if pacer.balance <= 0:
            results.append((task, False, "unstarted"))
            continue
        run = pacer.start_task(tasks_after=len(tasks) - i - 1)
        history, done = [], False
        while run.before_step() is Action.CONTINUE:
            usage, tool_error, done = call_model(run.model, task, history)
            action = run.after_step(cost_of(usage, PRICES[run.model]), tool_error=tool_error, final=done)
            if action is Action.ESCALATE:
                history, done = [], False      # restart from the first turn on the large model
                continue
            history.append(usage)
            if done:
                break
        ok = done and grade(task, run.model)
        run.finish(success=ok)
        results.append((task, ok, run.stop_reason or ("solved" if ok else "model")))
    return results, pacer.balance


if __name__ == "__main__":
    rng = random.Random(0)

    def fake_model(model, task, history):
        """Small model: cheap, errs on hard tasks. Large model: 20x the price, rarely errs."""
        usage = {"input_tokens": 3000 + 800 * len(history), "output_tokens": 120}
        hard = task % 3 == 0
        tool_error = model == SMALL and hard and rng.random() < 0.6
        return usage, tool_error, len(history) >= 4

    def grade(task, model):
        return model == LARGE or task % 3 != 0

    tasks = list(range(30))
    for pool in (0.05, 0.15, 0.60):
        res, left = run_batch(tasks, pool, fake_model, grade, CostEstimator(small=0.002, large=0.04))
        solved = sum(ok for _, ok, _ in res)
        print(f"pool ${pool:.2f}: solved {solved}/{len(tasks)}, left ${left:.4f}")
