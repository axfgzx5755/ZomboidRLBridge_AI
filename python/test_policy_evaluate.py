"""Offline regression checks for policy evaluation and live arena capture."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import rl
from telemetry import TelemetryError
from bridge import BridgeError
from training_env import Arena


class PolicyEvaluationTests(unittest.TestCase):
    def test_capture_integer_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'arena.json'
            reader = Mock()
            reader.wait_for_update.return_value = SimpleNamespace(x=2094.497559, y=6057.311035, z=0)
            args = SimpleNamespace(output=output, radius=8, min_distance=1, max_distance=2, allow_obstacles=True)
            with patch.object(rl, 'TelemetryReader', return_value=reader):
                rl.capture(args)
            arena = json.loads(output.read_text())
            self.assertEqual((arena['x'], arena['y']), (2094, 6057))
            self.assertTrue(arena['allow_obstacles'])

    def test_episode_records_and_cleanup(self):
        cases = [('reset', BridgeError('reset failed')), ('step', TelemetryError('paused')),
                 ('step', KeyboardInterrupt()), ('success', None)]
        for stage, error in cases:
            with self.subTest(stage=stage, error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'evaluation.jsonl'
                args = SimpleNamespace(model=Path('mock.zip'), output=output, live=False, episodes=3, seed=42)
                env = Mock()
                env.reset.return_value = ([0]*5, {'target': [1, 2]})
                good_step = ([0]*5, 2.0, True, False, {'steps': 1, 'is_success': True, 'terminal_reason': 'success'})
                env.step.return_value = good_step
                if stage == 'reset':
                    env.reset.side_effect = error
                elif stage == 'step':
                    env.step.side_effect = [([0]*5, 1.0, False, False, {'steps': 1}), error]
                model = Mock()
                model.predict.return_value = (4, None)
                with patch.object(rl, 'load_configuration', return_value=Arena()), \
                     patch.object(rl, 'TrainingEnv', return_value=env), \
                     patch('stable_baselines3.PPO.load', return_value=model):
                    if error is not None:
                        with self.assertRaises(type(error)):
                            rl.evaluate(args)
                    else:
                        self.assertEqual(rl.evaluate(args), 0)
                env.close.assert_called_once()
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(len(rows), 3 if error is None else 1)
                row = rows[-1]
                expected = 'success' if error is None else 'interrupted' if isinstance(error, KeyboardInterrupt) else 'error'
                self.assertEqual(row['terminal_reason'], expected)
                if stage == 'step':
                    self.assertEqual(row['steps'], 1)
                    self.assertEqual(row['reward'], 1.0)
                    self.assertEqual(row['target'], [1, 2])
                if error is not None:
                    self.assertFalse(row['is_success'])
                    self.assertEqual(row['error_type'], type(error).__name__)

if __name__ == '__main__':
    unittest.main()
