"""Reward functions kept independent from input and Gymnasium code."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    progress_scale: float = 1.0
    time_penalty: float = 0.01
    stationary_penalty: float = 0.02
    stationary_epsilon: float = 0.01
    goal_reward: float = 10.0


@dataclass(frozen=True)
class RewardResult:
    reward: float
    progress: float
    travelled: float
    reached_goal: bool


def distance_to_target(x: float, y: float, target_x: float, target_y: float) -> float:
    return math.hypot(target_x - x, target_y - y)


def navigation_reward(
    *,
    previous_x: float,
    previous_y: float,
    current_x: float,
    current_y: float,
    target_x: float,
    target_y: float,
    goal_radius: float,
    config: RewardConfig = RewardConfig(),
) -> RewardResult:
    """Reward progress toward a target, not raw distance travelled."""
    previous_distance = distance_to_target(previous_x, previous_y, target_x, target_y)
    current_distance = distance_to_target(current_x, current_y, target_x, target_y)
    progress = previous_distance - current_distance
    travelled = math.hypot(current_x - previous_x, current_y - previous_y)
    reached_goal = current_distance <= goal_radius

    reward = progress * config.progress_scale - config.time_penalty
    if travelled <= config.stationary_epsilon:
        reward -= config.stationary_penalty
    if reached_goal and previous_distance > goal_radius:
        reward += config.goal_reward

    return RewardResult(reward, progress, travelled, reached_goal)

