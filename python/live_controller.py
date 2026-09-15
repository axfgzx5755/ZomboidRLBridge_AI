"""Continuously control Project Zomboid from live telemetry.

This is the pre-PPO baseline controller: it greedily selects one of eight
movement actions that reduces distance to a configured target.  Launch it once,
focus Project Zomboid during the countdown, and it handles subsequent actions.
"""

from __future__ import annotations

import argparse
import math
import time

from actions import (
    ActionError,
    Movement,
    foreground_window_title,
    is_project_zomboid_foreground,
    stop,
)
from env import ZomboidNavigationEnv
from telemetry import TelemetryError, TelemetryReader, TelemetrySample


def choose_movement(
    sample: TelemetrySample,
    target_x: float,
    target_y: float,
    axis_tolerance: float,
) -> Movement:
    """Choose the 8-way action pointing from the player to the target."""
    dx = target_x - sample.x
    dy = target_y - sample.y
    horizontal = 1 if dx > axis_tolerance else -1 if dx < -axis_tolerance else 0
    vertical = 1 if dy > axis_tolerance else -1 if dy < -axis_tolerance else 0

    return {
        (0, 0): Movement.NONE,
        (0, -1): Movement.NORTH,
        (0, 1): Movement.SOUTH,
        (-1, 0): Movement.WEST,
        (1, 0): Movement.EAST,
        (-1, -1): Movement.NORTH_WEST,
        (1, -1): Movement.NORTH_EAST,
        (-1, 1): Movement.SOUTH_WEST,
        (1, 1): Movement.SOUTH_EAST,
    }[(horizontal, vertical)]


def wait_for_game_focus() -> None:
    announced = False
    while not is_project_zomboid_foreground():
        if not announced:
            title = foreground_window_title() or "<untitled>"
            print(f"Paused: focus Project Zomboid (foreground={title!r})", flush=True)
            announced = True
        stop()
        time.sleep(0.25)
    if announced:
        print("Project Zomboid focused; resuming", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic telemetry -> action -> telemetry controller"
    )
    target = parser.add_argument_group("target")
    target.add_argument("--target-x", type=float)
    target.add_argument("--target-y", type=float)
    target.add_argument("--relative-x", type=float, default=10.0)
    target.add_argument("--relative-y", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.15)
    parser.add_argument("--goal-radius", type=float, default=0.75)
    parser.add_argument("--axis-tolerance", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()
    if (args.target_x is None) != (args.target_y is None):
        parser.error("--target-x and --target-y must be supplied together")
    if args.duration <= 0 or args.goal_radius <= 0 or args.max_steps <= 0:
        parser.error("duration, goal-radius, and max-steps must be positive")
    return args


def main() -> int:
    args = parse_args()
    reader = TelemetryReader()
    initial = reader.read()

    if args.target_x is None:
        target_x = initial.x + args.relative_x
        target_y = initial.y + args.relative_y
    else:
        target_x, target_y = args.target_x, args.target_y

    initial_distance = math.hypot(target_x - initial.x, target_y - initial.y)
    print(f"start=({initial.x:.3f}, {initial.y:.3f})")
    print(f"target=({target_x:.3f}, {target_y:.3f}), distance={initial_distance:.3f}")
    print("Ctrl+C stops the controller and releases movement keys.")
    for remaining in range(args.countdown, 0, -1):
        print(f"Focus Project Zomboid: automatic control begins in {remaining}...", flush=True)
        time.sleep(1)
    wait_for_game_focus()

    environment = ZomboidNavigationEnv(
        target=(target_x, target_y),
        action_duration=args.duration,
        goal_radius=args.goal_radius,
        max_steps=args.max_steps,
        reader=reader,
    )
    _, info = environment.reset()
    total_reward = 0.0

    try:
        while True:
            wait_for_game_focus()
            state = info["telemetry"]
            sample = TelemetrySample(
                x=float(state["x"]),
                y=float(state["y"]),
                z=float(state["z"]),
                health=float(state["health"]),
                modified_ns=0,
            )
            action = choose_movement(
                sample, target_x, target_y, args.axis_tolerance
            )
            try:
                _, reward, terminated, truncated, info = environment.step(action)
            except ActionError:
                continue
            except TelemetryError as error:
                stop()
                print(f"Telemetry paused: {error}", flush=True)
                time.sleep(0.25)
                continue

            total_reward += reward
            if info["steps"] % args.log_every == 0 or terminated or truncated:
                state = info["telemetry"]
                distance = math.hypot(target_x - state["x"], target_y - state["y"])
                print(
                    f"step={info['steps']} action={action.name} "
                    f"pos=({state['x']:.3f},{state['y']:.3f}) "
                    f"distance={distance:.3f} progress={info['progress']:+.3f} "
                    f"reward={reward:+.3f} total={total_reward:+.3f}",
                    flush=True,
                )

            if terminated:
                if state["health"] <= 0:
                    print("Stopped: player is dead")
                    return 2
                print("SUCCESS: target reached")
                return 0
            if truncated:
                print("Stopped: maximum step count reached")
                return 1
    except KeyboardInterrupt:
        print("\nStopped by user")
        return 130
    finally:
        environment.close()


if __name__ == "__main__":
    raise SystemExit(main())

