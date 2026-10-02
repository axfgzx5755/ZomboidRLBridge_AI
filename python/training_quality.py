"""Geometry-preserving augmentation, PPO settings and auditable diagnostics."""
from collections import Counter
import math

import gymnasium as gym
import numpy as np

from navigation import DIRECTIONS

# Reflection about x=y: no sign flips, so exact tile boundaries stay exact.
ACTION_TRANSPOSE = tuple(DIRECTIONS.index((dy, dx)) for dx, dy in DIRECTIONS) + (9,)


def transpose_observation(observation):
    source = np.asarray(observation, dtype=np.float32)
    if source.shape not in ((134,), (142,)):
        raise ValueError('augmentation requires navigation-v2 or door-v1 observations')
    result = source.copy()
    for x, y in ((0, 1), (5, 6), (7, 8)):
        result[x], result[y] = source[y], source[x]
    # Row/column transpose and N,E,S,W -> W,S,E,N channel mapping.
    grid = source[9:134].reshape(5, 5, 5)
    result[9:134] = grid.transpose(1, 0, 2)[:, :, [0, 4, 3, 2, 1]].reshape(-1)
    if len(source) == 142:
        result[134:136] = source[134:136][::-1]
        result[139] = 1 - source[139]  # north door becomes west door
    return result


class GeometryAugmentation(gym.Wrapper):
    """Choose one coordinate frame per rollout episode; map actions back to world.

    PPO stores observations/actions/log probabilities in the same frame. We never
    duplicate old transitions or reuse a log probability for a different action.
    Raw evaluation and deployment use the original coordinate frame.
    """
    def __init__(self, env, probability=0.5, seed=42):
        super().__init__(env)
        if env.observation_space.shape not in ((134,), (142,)):
            raise ValueError('unsupported observation schema')
        if not 0 <= probability <= 1:
            raise ValueError('probability must be in [0, 1]')
        self.probability = probability
        self.rng = np.random.default_rng(seed)
        self.transposed = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self.transposed = self.rng.random() < self.probability
        return self._obs(obs), dict(info, augmentation='transpose' if self.transposed else 'identity')

    def _obs(self, obs):
        return transpose_observation(obs) if self.transposed else obs

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('invalid augmented action')
        world_action = ACTION_TRANSPOSE[int(action)] if self.transposed else int(action)
        obs, reward, terminated, truncated, info = self.env.step(world_action)
        return self._obs(obs), reward, terminated, truncated, dict(
            info, augmentation='transpose' if self.transposed else 'identity')


def add_quality_arguments(parser):
    parser.add_argument('--augment-geometry', action='store_true',
                        help='train with identity/x-y transpose frames; raw evaluation unchanged')
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--target-kl', type=float)
    parser.add_argument('--entropy-coef', type=float)


def ppo_overrides(args):
    settings = {}
    for flag, key in (('learning_rate', 'learning_rate'), ('epochs', 'n_epochs'),
                      ('target_kl', 'target_kl'), ('entropy_coef', 'ent_coef')):
        value = getattr(args, flag, None)
        if value is not None:
            if not math.isfinite(value) or (value < 0 if key == 'ent_coef' else value <= 0):
                raise ValueError(f'invalid {flag}')
            settings[key] = value
    return settings


def model_health(model):
    import torch
    parameters = list(model.policy.named_parameters())
    bad = [name for name, value in parameters if not torch.isfinite(value).all()]
    optimizer_finite = all(torch.isfinite(value).all().item()
                           for state in model.policy.optimizer.state.values()
                           for value in state.values() if torch.is_tensor(value))
    gradients = [value.grad.detach() for _, value in parameters if value.grad is not None]
    gradients_finite = all(torch.isfinite(value).all().item() for value in gradients)
    return dict(
        timesteps=model.num_timesteps, observation_shape=list(model.observation_space.shape),
        actions=int(model.action_space.n), architecture=str(model.policy),
        parameter_count=sum(value.numel() for _, value in parameters),
        parameters_finite=not bad, nonfinite_parameters=bad,
        optimizer_finite=optimizer_finite,
        gradients_available=bool(gradients), gradients_finite=gradients_finite if gradients else None,
        last_clipped_gradient_l2=float(torch.sqrt(sum((v.double() ** 2).sum() for v in gradients))) if gradients and gradients_finite else None,
        parameter_l2=float(torch.sqrt(sum((v.detach().double() ** 2).sum() for _, v in parameters))) if not bad else None,
        optimizer=type(model.policy.optimizer).__name__,
        settings=dict(learning_rate=float(model.lr_schedule(1)), n_steps=model.n_steps,
                      batch_size=model.batch_size, n_epochs=model.n_epochs,
                      gamma=model.gamma, gae_lambda=model.gae_lambda,
                      clip_range=float(model.clip_range(1)), target_kl=model.target_kl,
                      ent_coef=model.ent_coef, vf_coef=model.vf_coef,
                      max_grad_norm=model.max_grad_norm))


def require_healthy(model):
    report = model_health(model)
    if not report['parameters_finite'] or not report['optimizer_finite'] or report['gradients_finite'] is False:
        raise ValueError('nonfinite model/optimizer; inspect checkpoint before training')
    return report


def summarize_episodes(records):
    if not records:
        raise ValueError('no evaluation records')
    n = len(records)
    successes = sum(bool(row['is_success']) for row in records)
    p = successes / n
    z = 1.959963984540054
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    result = dict(episodes=n, successes=successes, success_rate=p,
                  success_rate_wilson95=[max(0, center-margin), min(1, center+margin)],
                  terminal_reasons=dict(Counter(row['terminal_reason'] for row in records)),
                  mean_reward=float(np.mean([row['reward'] for row in records])),
                  mean_steps=float(np.mean([row['steps'] for row in records])),
                  p95_steps=float(np.percentile([row['steps'] for row in records], 95)))
    for flag in ('opened_by_agent', 'crossed_door'):
        if all(flag in row for row in records):
            result[flag + '_rate'] = sum(bool(row[flag]) for row in records) / n
    groups = sorted({row['scenario_index'] for row in records if 'scenario_index' in row})
    if groups:
        result['by_scenario'] = {
            str(group): summarize_episodes([{k: v for k, v in row.items() if k != 'scenario_index'}
                                            for row in records if row.get('scenario_index') == group])
            for group in groups}
        result['macro_success_rate'] = float(np.mean([g['success_rate'] for g in result['by_scenario'].values()]))
    return result
