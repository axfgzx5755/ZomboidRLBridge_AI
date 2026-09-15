"""Gymnasium contract test that never sends input to the live game."""

from __future__ import annotations

from gymnasium.utils.env_checker import check_env

import env as env_module
from actions import Movement
from telemetry import TelemetrySample


class FakeReader:
    def __init__(self) -> None:
        self.sample = TelemetrySample(0.0, 0.0, 0.0, 1.0, 1)

    def read(self) -> TelemetrySample:
        return self.sample

    def wait_for_update(self, previous, *, timeout=1.5, poll_interval=0.02):
        dx, dy = {
            Movement.NONE: (0.0, 0.0),
            Movement.NORTH: (0.0, -1.0),
            Movement.SOUTH: (0.0, 1.0),
            Movement.WEST: (-1.0, 0.0),
            Movement.EAST: (1.0, 0.0),
            Movement.NORTH_WEST: (-1.0, -1.0),
            Movement.NORTH_EAST: (1.0, -1.0),
            Movement.SOUTH_WEST: (-1.0, 1.0),
            Movement.SOUTH_EAST: (1.0, 1.0),
        }[last_action]
        self.sample = TelemetrySample(
            previous.x + dx,
            previous.y + dy,
            previous.z,
            previous.health,
            previous.modified_ns + 1,
        )
        return self.sample


last_action = Movement.NONE


def fake_move(action, duration):
    global last_action
    last_action = Movement(action)


def main() -> None:
    env_module.move = fake_move
    env_module.stop = lambda: None
    environment = env_module.ZomboidNavigationEnv(
        target=(10.0, 0.0), reader=FakeReader(), max_steps=20
    )
    check_env(environment, skip_render_check=True)

    environment.reset()
    _, reward, terminated, truncated, info = environment.step(Movement.EAST)
    assert reward > 0
    assert info["progress"] > 0
    assert not terminated and not truncated

    _, reward, _, _, info = environment.step(Movement.WEST)
    assert reward < 0
    assert info["progress"] < 0
    print("Gymnasium environment tests passed")


if __name__ == "__main__":
    main()

