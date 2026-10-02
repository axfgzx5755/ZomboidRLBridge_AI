"""Repeatable bounded navigation episodes, with live and synthetic backends."""

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

from actions import HeldMovement, is_project_zomboid_foreground, ActionError
from bridge import TrainingBridge
from env import ZomboidNavigationEnv
from telemetry import TelemetryReader, TelemetrySample


@dataclass(frozen=True)
class Arena:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    radius: float = 8.0
    min_distance: float = 2.0
    max_distance: float = 5.0
    max_steps: int = 100
    action_duration: float = 0.15
    goal_radius: float = 0.75
    allow_obstacles: bool = False

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("arena values must be finite")
        if not (2 <= self.radius <= 20 and self.goal_radius < self.min_distance <= self.max_distance < self.radius):
            raise ValueError("require 2 <= radius <= 20 and goal_radius < min_distance <= max_distance < radius")
        if self.goal_radius <= 0 or self.action_duration <= 0 or self.max_steps < 1 or int(self.max_steps) != self.max_steps:
            raise ValueError("invalid goal radius, duration, or max_steps")
        if self.z != 0:
            raise ValueError("the first navigation arena requires ground level (z=0)")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class SyntheticBackend:
    """Point-mass test double, not a simulator of Zomboid movement or survival."""

    def __init__(self):
        self.sample = TelemetrySample(0, 0, 0, 1, 0)

    def reset(self, arena):
        self.sample = TelemetrySample(arena.x, arena.y, arena.z, 1, self.sample.modified_ns + 1)
        return self.sample

    def read(self):
        return self.sample

    def wait_for_update(self, previous, **kwargs):
        self.sample = TelemetrySample(self.sample.x, self.sample.y, self.sample.z,
                                      self.sample.health, previous.modified_ns + 1)
        return self.sample

    def move(self, action, duration):
        dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
                  (-1, -1), (1, -1), (-1, 1), (1, 1)][int(action)]
        scale = duration * 4 / max(1, math.hypot(dx, dy))
        p = self.sample
        self.sample = TelemetrySample(p.x + dx * scale, p.y + dy * scale, p.z, 1, p.modified_ns + 1)

    def stop(self):
        pass

    def close(self):
        pass


class LiveBackend:
    def __init__(self):
        self.reader = TelemetryReader()
        self.bridge = TrainingBridge()
        self.bridge.acquire()
        self.movement = HeldMovement()

    def reset(self, arena):
        if not is_project_zomboid_foreground():
            raise ActionError("focus Project Zomboid before resetting the training arena")
        self.reader.wait_for_update(self.reader.read())
        self.stop()
        return self.bridge.reset(arena, self.reader)

    def target_is_free(self, arena, x, y):
        result = self.bridge.request("target", replace(arena, x=x, y=y))
        return result.get("error") == ""

    def read(self):
        self.movement.check()
        return self.reader.read()

    def wait_for_update(self, previous, **kwargs):
        sample = self.reader.wait_for_update(previous, **kwargs)
        self.movement.check()
        return sample

    def move(self, action, duration):
        if not is_project_zomboid_foreground():
            raise ActionError("game lost focus; stopping training")
        self.bridge.heartbeat()
        self.movement.move(action, duration)

    def stop(self):
        self.movement.stop()

    def close(self):
        try:
            self.movement.close()
        finally:
            self.bridge.close()


class TrainingEnv(ZomboidNavigationEnv):
    def __init__(self, arena=Arena(), *, live=False, backend=None):
        self.arena = arena
        self.live = live
        self.backend = backend or (LiveBackend() if live else SyntheticBackend())
        super().__init__(target=(arena.x, arena.y), action_duration=arena.action_duration,
                         max_steps=arena.max_steps, goal_radius=arena.goal_radius,
                         observation_radius=arena.radius * 2, reader=self.backend,
                         move_fn=self.backend.move, stop_fn=self.backend.stop)
        self._done = True

    def reset(self, *, seed=None, options=None):
        self._sample = None
        self._done = True
        self.backend.stop()
        self.backend.reset(self.arena)
        _, info = super().reset(seed=seed)
        for _ in range(32):
            angle = self.np_random.uniform(0, 2 * math.pi)
            distance = self.np_random.uniform(self.arena.min_distance, self.arena.max_distance)
            self.target_x = self.arena.x + math.cos(angle) * distance
            self.target_y = self.arena.y + math.sin(angle) * distance
            if not (self.live and self.arena.allow_obstacles):
                break
            if self.backend.target_is_free(self.arena, self.target_x, self.target_y):
                break
        else:
            raise ValueError("no free target found in 32 attempts; move or reduce target distances")
        info.update(target=(self.target_x, self.target_y), game_was_reset=self.live,
                    backend="live" if self.live else "synthetic",
                    allow_obstacles=self.arena.allow_obstacles)
        self._done = False
        return self._observation(self._sample), info

    def step(self, action):
        if self._done:
            raise RuntimeError("reset() is required before starting a new episode")
        try:
            observation, reward, terminated, truncated, info = super().step(action)
        except BaseException:
            self._done = True
            self.backend.stop()
            raise
        state = self._sample
        outside = math.hypot(state.x - self.arena.x, state.y - self.arena.y) > self.arena.radius or abs(state.z - self.arena.z) > 0.1
        dead = state.health <= 0
        success = math.hypot(state.x - self.target_x, state.y - self.target_y) <= self.goal_radius and not dead and not outside
        terminated = success or dead or outside
        truncated = truncated and not terminated
        if outside or dead:
            reward -= 10.0
        self._done = terminated or truncated
        if self._done:
            self.backend.stop()
        info.update(allow_obstacles=self.arena.allow_obstacles, is_success=success, terminal_reason="success" if success else "dead" if dead else
                    "out_of_bounds" if outside else "max_steps" if truncated else "running")
        return observation, reward, terminated, truncated, info

    def close(self):
        self.backend.close()
