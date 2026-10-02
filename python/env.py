"""Minimal state-only Gymnasium environment for navigation experiments."""

from __future__ import annotations

from typing import Any

try:
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
except ImportError as error:
    raise ImportError(
        "env.py requires numpy and gymnasium; install requirements.txt first"
    ) from error

from actions import Movement, move, stop
from reward import RewardConfig, distance_to_target, navigation_reward
from telemetry import TelemetryReader, TelemetrySample


class ZomboidNavigationEnv(gym.Env):
    """A live-game environment; reset does not yet reset Project Zomboid."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        target: tuple[float, float],
        action_duration: float = 0.25,
        telemetry_timeout: float = 1.5,
        goal_radius: float = 0.75,
        observation_radius: float = 100.0,
        max_steps: int = 500,
        reward_config: RewardConfig = RewardConfig(),
        reader: TelemetryReader | None = None,
        move_fn=None,
        stop_fn=None,
    ) -> None:
        super().__init__()
        self.target_x, self.target_y = map(float, target)
        self.action_duration = action_duration
        self.telemetry_timeout = telemetry_timeout
        self.goal_radius = goal_radius
        if observation_radius <= 0:
            raise ValueError("observation_radius must be positive")
        self.observation_radius = observation_radius
        self.max_steps = max_steps
        self.reward_config = reward_config
        self.reader = reader or TelemetryReader()
        self._move = move_fn or move
        self._stop = stop_fn or stop
        self.action_space = spaces.Discrete(len(Movement))
        # dx, dy, z, health, distance-to-target
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._sample: TelemetrySample | None = None
        self._steps = 0

    def _observation(self, sample: TelemetrySample) -> np.ndarray:
        dx = self.target_x - sample.x
        dy = self.target_y - sample.y
        distance = distance_to_target(sample.x, sample.y, self.target_x, self.target_y)
        radius = self.observation_radius
        return np.array(
            [
                np.clip(dx / radius, -1.0, 1.0),
                np.clip(dy / radius, -1.0, 1.0),
                np.clip(sample.z / 32.0, -1.0, 1.0),
                sample.health,
                np.clip(distance / radius, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._stop()
        if options and "target" in options:
            self.target_x, self.target_y = map(float, options["target"])
        self._sample = self.reader.read()
        self._steps = 0
        return self._observation(self._sample), {
            "telemetry": self._sample.as_dict(),
            "target": (self.target_x, self.target_y),
            "game_was_reset": False,
        }

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._sample is None:
            raise RuntimeError("reset() must be called before step()")
        previous = self._sample
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        self._move(Movement(int(action)), self.action_duration)
        # Discard snapshots captured while the movement key was still held.
        barrier = self.reader.read()
        current = self.reader.wait_for_update(barrier, timeout=self.telemetry_timeout)
        self._sample = current
        self._steps += 1

        result = navigation_reward(
            previous_x=previous.x,
            previous_y=previous.y,
            current_x=current.x,
            current_y=current.y,
            target_x=self.target_x,
            target_y=self.target_y,
            goal_radius=self.goal_radius,
            config=self.reward_config,
        )
        terminated = result.reached_goal or current.health <= 0.0
        truncated = self._steps >= self.max_steps
        info = {
            "telemetry": current.as_dict(),
            "target": (self.target_x, self.target_y),
            "progress": result.progress,
            "travelled": result.travelled,
            "steps": self._steps,
        }
        return self._observation(current), result.reward, terminated, truncated, info

    def close(self) -> None:
        self._stop()
        super().close()
