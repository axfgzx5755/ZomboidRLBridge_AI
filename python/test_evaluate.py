"""Evaluation outcome and cleanup tests; no live keyboard input."""

import unittest

from evaluate import run_episode
from telemetry import TelemetryError


class FakeEnv:
    target_x, target_y, goal_radius, max_steps = 2, 0, 0.75, 3

    def __init__(self, *, health=1, dx=1, error=None):
        self.x, self.health, self.dx, self.error = 0, health, dx, error
        self.closed = False

    def info(self):
        return {"telemetry": dict(x=self.x, y=0, z=0, health=self.health)}

    def reset(self):
        return None, self.info()

    def step(self, action):
        if self.error:
            raise self.error
        self.x += self.dx
        return None, 1, False, False, self.info()

    def close(self):
        self.closed = True


class EvaluationTests(unittest.TestCase):
    def test_outcomes_and_cleanup(self):
        for kwargs, status, steps in [
            ({}, "success", 2),
            ({"health": 0}, "dead", 0),
            ({"dx": 0}, "max_steps", 3),
            ({"error": TelemetryError("stale")}, "error", 0),
            ({"error": KeyboardInterrupt()}, "interrupted", 0),
        ]:
            with self.subTest(status=status):
                environment = FakeEnv(**kwargs)
                result = run_episode(environment, focus=lambda: None)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["steps"], steps)
                self.assertTrue(environment.closed)

    def test_already_at_goal(self):
        environment = FakeEnv()
        environment.target_x = 0
        result = run_episode(environment, focus=lambda: self.fail("unexpected movement"))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["steps"], 0)


if __name__ == "__main__":
    unittest.main()
