"""Budget pacing for LLM agents that share a fixed spend pool.

A batch of agent tasks draws on one pool of money (an API key's credit, a monthly
allocation). `BudgetPacer` decides, for each task and each step, which model runs the step
and whether the task stops, so that the pool is never spent on one task at the expense of
the tasks still waiting. It has four rules and one guard:

reserve     Commit money to the large model only if what is left, after setting aside the
            expected small-model cost of every task still to run, covers a whole
            large-model task.
route       Start on the small model; at the first step whose tool calls fail, restart the
            task on the large model if the reserve allows.
stop        Stop a task when the pool is empty, or when the next step (forecast as costing
            what the previous one did) would cross the task's cap.
spend down  Start on the large model when the pool covers it for this task and every task
            still to run.
guard       Escalation is decided once per task and is one-way.

The pacer only does arithmetic on the cost of steps that already ran, so it adds no model
calls. Usage, with any model client:

    pacer = BudgetPacer(pool_usd=25.0, estimator=CostEstimator(small=0.002, large=0.024))
    for i, task in enumerate(tasks):
        run = pacer.start_task(tasks_after=len(tasks) - i - 1)
        while run.before_step() is Action.CONTINUE:
            step = call_model(run.model, task)          # your agent step
            action = run.after_step(cost_of(step.usage, PRICES[run.model]),
                                    tool_error=step.tool_error, final=step.done)
            if action is Action.ESCALATE:
                task.restart()                          # rerun from the first turn on run.model
            elif step.done:
                break
        run.finish(success=grade(task))
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum

SMALL, LARGE = "small", "large"


class Action(Enum):
    CONTINUE = "continue"
    ESCALATE = "escalate"
    STOP = "stop"


@dataclass(frozen=True)
class PriceCard:
    """USD per million tokens, by billing category."""
    input: float
    cache_write: float
    cache_read: float
    output: float


def cost_of(usage: dict, card: PriceCard) -> float:
    """Dollar cost of one request from the provider's usage counters."""
    return (usage.get("input_tokens", 0) * card.input
            + usage.get("cache_write_tokens", 0) * card.cache_write
            + usage.get("cache_read_tokens", 0) * card.cache_read
            + usage.get("output_tokens", 0) * card.output) / 1e6


@dataclass
class CostEstimator:
    """Expected cost of a whole task on each model. Starts from priors and, once tasks
    finish, uses the running mean of completed tasks (failed ones included)."""
    small: float
    large: float
    _sum: dict = field(default_factory=lambda: {SMALL: 0.0, LARGE: 0.0})
    _n: dict = field(default_factory=lambda: {SMALL: 0, LARGE: 0})

    def expected(self, model: str) -> float:
        prior = self.small if model == SMALL else self.large
        return self._sum[model] / self._n[model] if self._n[model] else prior

    def observe(self, model: str, task_cost: float) -> None:
        self._sum[model] += task_cost
        self._n[model] += 1


class BudgetPacer:
    """Shared pool plus the pacing rules. Thread-safe: concurrent tasks charge the same
    ledger, and with `reserve_in_flight` the reserve also covers tasks already running."""

    def __init__(self, pool_usd: float, estimator: CostEstimator, *, route: bool = True,
                 spend_down: bool = True, cap: bool = True, reserve_in_flight: bool = False):
        self.balance = pool_usd
        self.estimator = estimator
        self.route, self.spend_down, self.cap = route, spend_down, cap
        self.reserve_in_flight = reserve_in_flight
        self.in_flight = 0
        self._lock = threading.Lock()

    def _reserve(self, tasks_after: int) -> float:
        waiting = tasks_after + (max(self.in_flight - 1, 0) if self.reserve_in_flight else 0)
        return waiting * self.estimator.expected(SMALL)

    def start_task(self, tasks_after: int) -> "TaskRun":
        with self._lock:
            self.in_flight += 1
            run = TaskRun(self, tasks_after)
            c_large = self.estimator.expected(LARGE)
            if self.spend_down and self.balance >= (tasks_after + 1) * c_large:
                run.model, run.decided = LARGE, True
                run.cap = self.balance - self._reserve(tasks_after) if self.cap else None
            return run

    def _charge(self, usd: float) -> None:
        with self._lock:
            self.balance -= usd


@dataclass
class TaskRun:
    pacer: BudgetPacer
    tasks_after: int
    model: str = SMALL
    decided: bool = False
    cap: float | None = None
    spent: float = 0.0           # spend of the current attempt (resets on escalation)
    total: float = 0.0           # spend of the whole task
    stop_reason: str | None = None
    _prev: float | None = None

    def before_step(self) -> Action:
        if self.stop_reason:
            return Action.STOP
        with self.pacer._lock:
            balance = self.pacer.balance
        if balance <= 0:
            self.stop_reason = "pool"
        elif self.cap is not None and self._prev is not None and self.spent + self._prev > self.cap:
            self.stop_reason = "cap"
        return Action.STOP if self.stop_reason else Action.CONTINUE

    def after_step(self, cost_usd: float, *, tool_error: bool = False, final: bool = False) -> Action:
        self.pacer._charge(cost_usd)
        self.spent += cost_usd
        self.total += cost_usd
        self._prev = cost_usd
        # escalation needs a tool error on the small model, before the task's final step
        if self.model == SMALL and tool_error and not self.decided and not final and self.pacer.route:
            self.decided = True
            p = self.pacer
            with p._lock:
                room = p.balance - p._reserve(self.tasks_after)
                if room >= p.estimator.expected(LARGE):
                    self.model, self.spent, self._prev = LARGE, 0.0, None
                    self.cap = room if p.cap else None
                    return Action.ESCALATE
        return Action.CONTINUE

    def finish(self, success: bool | None = None) -> None:
        """Release the task. A task that ran to its end (solved or not) updates the cost
        estimates with the cost of its final attempt; a stopped task does not."""
        with self.pacer._lock:
            self.pacer.in_flight -= 1
            if self.stop_reason is None:
                self.pacer.estimator.observe(self.model, self.spent)
