"""Persistent input regressions; all OS keyboard events are mocked."""
import time
import unittest
from unittest.mock import patch, Mock

import actions
from actions import HeldMovement, Movement, ActionError
from training_env import TrainingEnv, Arena, SyntheticBackend


class HeldMovementTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.focus = patch.object(actions, 'is_project_zomboid_foreground', return_value=True).start()
        patch.object(actions, '_key_event', side_effect=lambda key, key_up: self.events.append((key, key_up))).start()
        self.controller = HeldMovement(timeout=0.15)

    def tearDown(self):
        self.controller.close()
        patch.stopall()

    def test_same_direction_has_no_release_gap(self):
        self.controller.move(Movement.NORTH, 0)
        self.controller.move(Movement.NORTH, 0)
        self.assertEqual(self.events, [(0x57, False)])
        self.controller.move(Movement.NORTH_EAST, 0)
        self.assertEqual(self.events[-1], (0x44, False))
        self.controller.move(Movement.EAST, 0)
        self.assertEqual(self.events[-1], (0x57, True))
        self.controller.move(Movement.NONE, 0)
        self.assertEqual(self.events[-1], (0x44, True))

    def wait_for_release(self):
        deadline = time.monotonic() + 2
        while self.controller._keys and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.controller._keys)
        self.assertIn((0x57, True), self.events)
        with self.assertRaises(ActionError):
            self.controller.move(Movement.NORTH, 0)

    def test_focus_loss_releases_between_steps(self):
        self.controller.move(Movement.NORTH, 0)
        self.focus.return_value = False
        self.wait_for_release()

    def test_stalled_policy_releases_between_steps(self):
        self.controller.move(Movement.NORTH, 0)
        self.wait_for_release()

    def test_invalid_input_and_close_release(self):
        self.controller.move(Movement.NORTH, 0)
        with self.assertRaises(ValueError):
            self.controller.move(Movement.EAST, float('nan'))
        self.assertFalse(self.controller._keys)
        self.controller.move(Movement.NORTH, 0)
        self.controller.close()
        self.assertFalse(self.controller._thread.is_alive())
        self.assertFalse(self.controller._keys)

    def test_keydown_failure_releases_partial_input(self):
        def send(key, key_up):
            self.events.append((key, key_up))
            if not key_up:
                raise OSError('input failed')
        with patch.object(actions, '_key_event', side_effect=send):
            with self.assertRaises(OSError):
                self.controller.move(Movement.NORTH, 0)
        self.assertEqual(self.events, [(0x57, False), (0x57, True)])

    def test_episode_end_and_error_stop(self):
        backend = SyntheticBackend()
        backend.stop = Mock()
        env = TrainingEnv(Arena(max_steps=1), backend=backend)
        env.reset()
        backend.stop.reset_mock()
        env.step(0)
        backend.stop.assert_called_once()
        env.reset()
        env.target_x, env.target_y = 0.6, 0
        backend.stop.reset_mock()
        self.assertTrue(env.step(4)[2])
        backend.stop.assert_called_once()
        env.reset()
        backend.stop.reset_mock()
        with self.assertRaises(ValueError):
            env.step(999)
        backend.stop.assert_called_once()

if __name__ == '__main__':
    unittest.main()
