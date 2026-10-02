"""Training contracts and bridge protocol, without touching the live game."""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from gymnasium.utils.env_checker import check_env
from bridge import TrainingBridge, BridgeError, atomic_text
from telemetry import TelemetryReader, TelemetryError
from training_env import Arena, TrainingEnv


class TrainingTests(unittest.TestCase):
    def test_gym_contract_and_seed(self):
        environment = TrainingEnv()
        check_env(environment, skip_render_check=True)
        first, _ = environment.reset(seed=31)
        environment.step(4)
        second, _ = environment.reset(seed=31)
        self.assertTrue((first == second).all())
        self.assertEqual(environment._sample.x, 0)

    def test_episode_endings(self):
        environment = TrainingEnv(Arena(max_steps=1))
        environment.reset()
        _, _, terminated, truncated, info = environment.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["terminal_reason"], "max_steps")
        with self.assertRaises(RuntimeError):
            environment.step(0)
        environment.reset()
        environment.target_x, environment.target_y = 0.6, 0
        _, _, terminated, truncated, info = environment.step(4)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])

    def test_arena_boundary(self):
        environment = TrainingEnv(Arena(radius=3, min_distance=1, max_distance=2))
        environment.reset()
        environment.target_x, environment.target_y = -2, 0
        for _ in range(6):
            _, _, terminated, _, info = environment.step(4)
        self.assertTrue(terminated)
        self.assertEqual(info["terminal_reason"], "out_of_bounds")

    def test_obstacle_targets_retry_and_bound(self):
        from training_env import SyntheticBackend
        backend = SyntheticBackend()
        calls = []
        def free(arena, x, y):
            calls.append((x, y))
            return len(calls) == 3
        backend.target_is_free = free
        env = TrainingEnv(Arena(allow_obstacles=True), live=True, backend=backend)
        _, info = env.reset(seed=42)
        self.assertEqual(len(calls), 3)
        self.assertEqual(info["target"], calls[-1])
        self.assertTrue(info["allow_obstacles"])
        backend.target_is_free = lambda *args: False
        with self.assertRaisesRegex(ValueError, "32 attempts"):
            env.reset(seed=42)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_invalid_arena(self):
        for kwargs in ({"radius": float("nan")}, {"max_distance": 30}, {"z": 1}, {"max_steps": 1.5}):
            with self.assertRaises(ValueError):
                Arena(**kwargs)

    def test_missing_player_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps(dict(x=0, y=0, z=0, health=1, missingPlayer=True)))
            with self.assertRaises(TelemetryError):
                TelemetryReader(path, retries=0).read()

    def test_ack_id_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = TrainingBridge(directory, timeout=0.2)
            (Path(directory) / "pz_training_ack.txt").write_text('{"id":"old","status":"ok"}')
            with self.assertRaises(BridgeError):
                bridge.request("reset", Arena())
            old_id = (Path(directory) / "pz_training_command.txt").read_text().split()[0]

            def respond():
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        command = (Path(directory) / "pz_training_command.txt").read_text().split()
                    except OSError:
                        time.sleep(0.005)
                        continue
                    if command[0] != old_id:
                        atomic_text(Path(directory) / "pz_training_ack.txt",
                                    json.dumps({"id": command[0], "status": "ok"}))
                        return
                    time.sleep(0.005)

            thread = threading.Thread(target=respond)
            bridge.timeout = 1
            thread.start()
            try:
                self.assertEqual(bridge.request("reset", Arena())["status"], "ok")
            finally:
                thread.join()

    def test_exclusive_lock_and_release(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = TrainingBridge(directory), TrainingBridge(directory)
            first.acquire()
            try:
                first.heartbeat()
                with self.assertRaises(BridgeError):
                    second.acquire()
            finally:
                first.close()
            self.assertEqual(first.lease.read_text().split()[1], "0")
            second.acquire()
            second.close()

    def test_expired_lease_is_not_silently_renewed(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = TrainingBridge(directory)
            bridge.heartbeat()
            bridge._last_heartbeat -= 31
            with self.assertRaises(BridgeError):
                bridge.heartbeat()

    def test_reset_verifies_position(self):
        from training_env import SyntheticBackend
        bridge = TrainingBridge()
        bridge.request = lambda *args: {"status": "ok"}
        with self.assertRaises(TelemetryError):
            bridge.reset(Arena(x=20), SyntheticBackend())


if __name__ == "__main__":
    unittest.main()
