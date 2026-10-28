"""Tests for budget_pacer. Run: python3 -m unittest discover -s optinet -p 'test_*.py'"""
from __future__ import annotations

import random
import sys
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from budget_pacer import LARGE, SMALL, Action, BudgetPacer, CostEstimator, PriceCard, cost_of  # noqa: E402


def pacer(pool, small=1.0, large=10.0, **kw):
    return BudgetPacer(pool, CostEstimator(small, large), **kw)


class Rules(unittest.TestCase):
    def test_cost_of_prices_every_category(self):
        card = PriceCard(input=2, cache_write=2.5, cache_read=0.1, output=10)
        u = {"input_tokens": 1_000_000, "cache_write_tokens": 1_000_000,
             "cache_read_tokens": 1_000_000, "output_tokens": 1_000_000}
        self.assertAlmostEqual(cost_of(u, card), 14.6)

    def test_spend_down_only_when_pool_covers_every_remaining_task(self):
        self.assertEqual(pacer(30.0).start_task(tasks_after=2).model, LARGE)   # 30 >= 3 * 10
        self.assertEqual(pacer(29.9).start_task(tasks_after=2).model, SMALL)

    def test_reserve_refuses_escalation_that_would_starve_later_tasks(self):
        run = pacer(12.0, spend_down=False).start_task(tasks_after=3)  # reserve 3 * 1
        self.assertIs(run.after_step(0.5, tool_error=True), Action.CONTINUE)  # room 8.5 < 10
        self.assertEqual(run.model, SMALL)
        run2 = pacer(14.0, spend_down=False).start_task(tasks_after=3)
        self.assertIs(run2.after_step(0.5, tool_error=True), Action.ESCALATE)  # room 10.5
        self.assertEqual((run2.model, run2.cap, run2.spent), (LARGE, 10.5, 0.0))

    def test_escalation_is_decided_once(self):
        run = pacer(12.0, spend_down=False).start_task(tasks_after=3)
        run.after_step(0.5, tool_error=True)                      # refused
        run.pacer.balance += 100                                  # money arrives later
        self.assertIs(run.after_step(0.5, tool_error=True), Action.CONTINUE)

    def test_no_escalation_on_the_final_step(self):
        run = pacer(100.0, spend_down=False).start_task(tasks_after=0)
        self.assertIs(run.after_step(0.5, tool_error=True, final=True), Action.CONTINUE)

    def test_cap_forecasts_the_next_step_from_the_previous_one(self):
        run = pacer(14.0, spend_down=False).start_task(tasks_after=3)
        run.after_step(0.5, tool_error=True)                      # escalate, cap 10.5
        run.after_step(6.0)
        self.assertIs(run.before_step(), Action.STOP)             # 6 + 6 > 10.5
        self.assertEqual(run.stop_reason, "cap")

    def test_empty_pool_stops(self):
        run = pacer(1.0).start_task(tasks_after=0)
        run.after_step(1.5)
        self.assertIs(run.before_step(), Action.STOP)
        self.assertEqual(run.stop_reason, "pool")

    def test_estimator_learns_from_finished_tasks(self):
        p = pacer(1000.0, small=1.0, large=10.0, spend_down=False)
        run = p.start_task(tasks_after=5)
        run.after_step(3.0)
        run.finish(True)
        self.assertEqual(p.estimator.expected(SMALL), 3.0)
        self.assertEqual(p.estimator.expected(LARGE), 10.0)       # prior until observed

    def test_concurrent_tasks_share_one_ledger(self):
        p = pacer(1000.0, reserve_in_flight=True)
        runs = [p.start_task(tasks_after=0) for _ in range(8)]

        def work(r):
            for _ in range(100):
                r.after_step(0.01)
        ts = [threading.Thread(target=work, args=(r,)) for r in runs]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertAlmostEqual(p.balance, 1000.0 - 8.0)


class ReplayEquivalence(unittest.TestCase):
    """The library and the direct implementation in fixed_spend.py agree exactly."""

    def test_library_reproduces_the_replay(self):
        import fixed_spend as F
        from pacer_replay import run_batch
        from replay import step_cost
        small, large, ids, est, rate = F.setup("small_cache_budget", "frontier_cache")
        full = sum(F.task_cost(large[i]) for i in ids)
        rng = random.Random(7)
        for frac in (0.1, 0.2, 0.4, 0.8, 1.0):
            for _ in range(20):
                order = rng.sample(ids, len(ids))
                for policy, kw in (("hybrid", {}), ("paced+cap", {"spend_down": False})):
                    a = F.simulate(policy, order, small, large, frac * full, est)
                    b = run_batch(order, small, large, frac * full, est, step_cost, **kw)
                    self.assertEqual(a["succ"], b["succ"], (frac, policy))
                    self.assertEqual(a["reasons"], b["reasons"], (frac, policy))
                    self.assertAlmostEqual(a["spent"], b["spent"], places=9)

    def test_library_reproduces_the_replay_with_causal_estimates(self):
        import fixed_spend as F
        from pacer_replay import run_batch
        from replay import step_cost
        small, large, ids, est, rate = F.setup("small_cache_budget", "frontier_cache")
        full = sum(F.task_cost(large[i]) for i in ids)
        ms = sum(e[0] for e in est.values()) / len(est)
        ml = sum(e[1] for e in est.values()) / len(est)
        rng = random.Random(11)
        for scale in (0.25, 1.0, 4.0):
            online = F.OnlineEst(ms * scale, ml * scale)
            for frac in (0.2, 0.4, 1.0):
                for _ in range(10):
                    order = rng.sample(ids, len(ids))
                    a = F.simulate("hybrid", order, small, large, frac * full, online)
                    b = run_batch(order, small, large, frac * full, est, step_cost,
                                  online=True, prior_scale=scale)
                    self.assertEqual((a["succ"], a["reasons"]), (b["succ"], b["reasons"]), (scale, frac))


if __name__ == "__main__":
    unittest.main()
