"""Drive recorded agent traces through `BudgetPacer`, the same class an agent would use.

`run_batch` replays one batch (a task order) under one pool and returns what each task
ended as. `fixed_spend.py` implements the same policies directly; `test_budget_pacer.py`
checks that both give identical results, so the evaluated policy is the shipped one.
"""
from __future__ import annotations

from budget_pacer import LARGE, SMALL, Action, BudgetPacer, CostEstimator


def run_batch(order, small, large, pool, est, step_cost, *, spend_down=True, route=True, cap=True,
              online=False, prior_scale=1.0):
    """est: task id -> (expected small cost, expected large cost), used as given unless
    `online`, in which case one estimator starts from the batch-mean prior times
    `prior_scale` and learns from finished tasks."""
    n = len(order)
    pacer = BudgetPacer(pool, CostEstimator(0.0, 0.0), route=route, spend_down=spend_down, cap=cap)
    if online:
        ps = sum(e[0] for e in est.values()) / len(est) * prior_scale
        pl = sum(e[1] for e in est.values()) / len(est) * prior_scale
        pacer.estimator = CostEstimator(ps, pl)
    out = {"succ": 0, "reasons": {"pool": 0, "cap": 0, "model": 0, "unstarted": 0}, "esc": 0}
    for pos, tid in enumerate(order):
        if pacer.balance <= 0:
            out["reasons"]["unstarted"] += 1
            continue
        if not online:
            pacer.estimator = CostEstimator(est[tid][0] * prior_scale, est[tid][1] * prior_scale)
        run = pacer.start_task(tasks_after=n - pos - 1)
        if run.model == LARGE:
            out["esc"] += 1
        trace = (large if run.model == LARGE else small)[tid]
        i, done = 0, False
        while run.before_step() is Action.CONTINUE:
            s = trace["steps"][i]
            final = i == len(trace["steps"]) - 1
            act = run.after_step(step_cost(s), tool_error=s["n_errors"] > 0, final=final)
            if act is Action.ESCALATE:
                out["esc"] += 1
                trace, i = large[tid], 0
                continue
            i += 1
            if final:
                done = True
                break
        ok = done and trace["success"]
        run.finish(success=ok)
        out["succ"] += int(bool(ok))
        if not ok:
            out["reasons"][run.stop_reason or "model"] += 1
    out["overdraw"] = max(0.0, -pacer.balance)
    out["spent"] = pool - pacer.balance
    return out
