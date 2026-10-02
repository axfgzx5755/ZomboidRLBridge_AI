import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from door_env import crosses_opening
from navigation import DIRECTIONS, Grid, NavigationEnv
from training_quality import (ACTION_TRANSPOSE, GeometryAugmentation, model_health,
                              ppo_overrides, summarize_episodes, transpose_observation)


class QualityTests(unittest.TestCase):
    def test_transpose_roundtrip_and_semantics(self):
        obs = np.zeros(142, dtype=np.float32)
        obs[:9] = [.2, -.3, .4, 1, .1, -.4, .7, 0, .6]
        obs[134:] = [.3, -.7, 1, 0, 1, 1, 0, 1]
        grid = obs[9:134].reshape(5, 5, 5)
        grid[1, 3, 1] = 1  # north wall at x=+1,y=-1
        changed = transpose_observation(obs)
        self.assertEqual(changed[9:134].reshape(5, 5, 5)[3, 1, 4], 1)
        self.assertEqual(changed[139], 0)
        np.testing.assert_array_equal(changed[136:139], obs[136:139])
        np.testing.assert_array_equal(changed[140:], obs[140:])
        np.testing.assert_array_equal(transpose_observation(changed), obs)
        for action, (dx, dy) in enumerate(DIRECTIONS):
            self.assertEqual(DIRECTIONS[ACTION_TRANSPOSE[action]], (dy, dx))
            self.assertEqual(ACTION_TRANSPOSE[ACTION_TRANSPOSE[action]], action)
        self.assertEqual(ACTION_TRANSPOSE[9], 9)

    def test_transposed_physics_preserves_rewards_and_terminals(self):
        # Real synthetic collision/observation code, including exact integer starts.
        for seed in range(5):
            original = NavigationEnv(stage=2)
            original.reset(seed=seed)
            mirrored = copy.deepcopy(original)
            mirrored.grid = Grid(original.grid.y, original.grid.x, original.grid.radius,
                                 original.grid.cells.transpose(1, 0, 2)[:, :, [0, 4, 3, 2, 1]])
            mirrored.position = original.position[::-1].copy()
            mirrored.target = original.target[::-1].copy()
            for action in np.random.default_rng(seed).integers(0, 9, size=20):
                first = original.step(int(action))
                second = mirrored.step(ACTION_TRANSPOSE[int(action)])
                np.testing.assert_allclose(transpose_observation(first[0]), second[0], atol=1e-6)
                self.assertAlmostEqual(first[1], second[1])
                self.assertEqual(first[2:4], second[2:4])
                if first[2] or first[3]:
                    break

    def test_door_crossing_preserved_both_sides(self):
        door = dict(x=10, y=20, north=True)
        mirrored = dict(x=20, y=10, north=False)
        for side in (-1, 1):
            before, after = [10.5, 20+side*.2], [10.5, 20-side*.2]
            self.assertTrue(crosses_opening(before, after, door, side))
            self.assertTrue(crosses_opening(before[::-1], after[::-1], mirrored, side))

    def test_wrapper_seed_and_action_mapping(self):
        raw = NavigationEnv(stage=2)
        env = GeometryAugmentation(raw, probability=1)
        obs, _ = env.reset(seed=4)
        np.testing.assert_array_equal(obs, transpose_observation(raw._observation()))
        before = raw.position.copy()
        env.step(5)  # (-1,0) in policy frame -> (0,-1) world frame
        self.assertAlmostEqual(raw.position[0], before[0])
        self.assertLess(raw.position[1], before[1])
        first, _ = env.reset(seed=4)
        second, _ = env.reset(seed=4)
        np.testing.assert_array_equal(first, second)
        env.close()

    def test_metrics_intervals_groups_and_failure_counts(self):
        rows = [dict(is_success=i<3, reward=i, steps=i+1, terminal_reason='success' if i<3 else 'stalled',
                     scenario_index=i%2, opened_by_agent=True, crossed_door=i<3) for i in range(4)]
        report = summarize_episodes(rows)
        self.assertEqual(report['success_rate'], .75)
        self.assertEqual(report['terminal_reasons']['stalled'], 1)
        self.assertEqual(report['by_scenario']['0']['success_rate'], 1)
        self.assertEqual(report['by_scenario']['1']['success_rate'], .5)
        self.assertLess(report['success_rate_wilson95'][0], .75)
        self.assertGreater(report['success_rate_wilson95'][1], .75)
        perfect = summarize_episodes([rows[0]]*30)
        self.assertAlmostEqual(perfect['success_rate_wilson95'][0], .8865, places=3)
        json.dumps(report, allow_nan=False)

    def test_door_evaluation_balances_scenarios(self):
        from navigation_train import evaluate
        info = dict(steps=1, is_success=True, terminal_reason='success')
        env = SimpleNamespace(stage=0, scenarios=[{}, {}, {}], stop=Mock(),
                              reset=Mock(return_value=(np.zeros(142), info)),
                              step=Mock(return_value=(np.zeros(142), 1, True, False, info)))
        model = Mock()
        model.predict.return_value = (9, None)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(evaluate(model, env, 6, 42, Path(directory)/'eval.jsonl', 0), 1)
        self.assertEqual([call.kwargs['options']['scenario_index'] for call in env.reset.call_args_list], [0, 1, 2, 0, 1, 2])

    def test_ppo_augmented_training_save_reload_and_override(self):
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        env = GeometryAugmentation(NavigationEnv(stage=2))
        model = PPO('MlpPolicy', env, n_steps=32, batch_size=16, n_epochs=1, seed=7)
        model.learn(64)
        report = model_health(model)
        self.assertTrue(report['parameters_finite'] and report['optimizer_finite'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.zip'
            model.save(path)
            loaded = PPO.load(path, env=env, learning_rate=1e-4, target_kl=.015, n_epochs=5)
            self.assertEqual(loaded.lr_schedule(1), 1e-4)
            obs, _ = env.reset(seed=3)
            self.assertEqual(int(model.predict(obs, deterministic=True)[0]), int(loaded.predict(obs, deterministic=True)[0]))
            loaded.learn(32)
            self.assertEqual(loaded.policy.optimizer.param_groups[0]['lr'], 1e-4)
        env.close()


if __name__ == '__main__':
    unittest.main()
