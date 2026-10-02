import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from grid_cnn import GridCNN
from navigation import NavigationEnv


class GridCNNTests(unittest.TestCase):
    def test_channels_and_extra_scalar_order(self):
        for size in (134, 142):
            obs = torch.arange(size, dtype=torch.float32).repeat(2, 1)
            grid, scalars = GridCNN.split(obs)
            self.assertEqual(tuple(grid.shape), (2, 5, 5, 5))
            # Verify every channel and location, not just output dimensions.
            for y in range(5):
                for x in range(5):
                    for channel in range(5):
                        self.assertEqual(grid[0, channel, y, x].item(), 9+(y*5+x)*5+channel)
            torch.testing.assert_close(scalars, torch.cat((obs[:, :9], obs[:, 134:]), 1))
            encoder = GridCNN(spaces.Box(-1, 1, (size,), dtype=np.float32))
            features = encoder(obs)
            self.assertEqual(tuple(features.shape), (2, 48+size-125))
            torch.testing.assert_close(features[:,48:], scalars)

    def test_cnn_weights_update_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        env = NavigationEnv(stage=2)
        model = PPO('MlpPolicy', env, n_steps=32, batch_size=16, n_epochs=2, seed=4,
                    policy_kwargs=dict(features_extractor_class=GridCNN))
        conv = model.policy.features_extractor.grid_encoder[0]
        before = conv.weight.detach().clone()
        model.learn(64)
        self.assertFalse(torch.equal(before, conv.weight))
        self.assertTrue(torch.isfinite(conv.weight.grad).all())
        obs, _ = env.reset(seed=17)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.zip'
            model.save(path)
            loaded = PPO.load(path)
            self.assertIsInstance(loaded.policy.features_extractor, GridCNN)
            obs_tensor = torch.tensor(obs[None])
            with torch.no_grad():
                torch.testing.assert_close(model.policy.get_distribution(obs_tensor).distribution.probs,
                                           loaded.policy.get_distribution(obs_tensor).distribution.probs)
        env.close()

    def test_paired_statistics_do_not_pool_shared_maps_as_independent(self):
        from cnn_experiment import compare
        results = []
        for seed in (71, 72, 73):
            for arch, wins in (('mlp', [True, False, False, False]),
                               ('cnn', [True, True, False, False])):
                results.append(dict(seed=seed, architecture=arch, successes=wins,
                                    evaluation=dict(success_rate=sum(wins)/4, mean_steps=10),
                                    training_seconds=2 if arch == 'cnn' else 1, parameter_count=100))
        result = compare(results)
        self.assertEqual(result['mean_success_difference_pp'], 25)
        self.assertEqual(result['training_seed_t95_difference_pp'], [25, 25])
        self.assertEqual(result['cnn_training_time_ratio'], 2)
        self.assertEqual(len(result['paired_training_seeds']), 3)
        self.assertEqual(result['paired_training_seeds'][0]['cnn_only'], 1)


if __name__ == '__main__':
    unittest.main()
