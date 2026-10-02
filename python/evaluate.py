"""Evaluate sequential navigation targets and append durable JSONL results."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from actions import ActionError
from env import ZomboidNavigationEnv
from live_controller import choose_movement, wait_for_game_focus
from telemetry import TelemetryError, TelemetryReader, TelemetrySample


def run_episode(environment, *, axis_tolerance=0.15, focus=wait_for_game_focus):
    started = time.monotonic()
    result = {"status": "error", "steps": 0, "total_reward": 0.0,
              "target": [environment.target_x, environment.target_y],
              "game_was_reset": False}
    try:
        _, info = environment.reset()
        result["start"] = info["telemetry"]
        while True:
            state = info["telemetry"]
            distance = math.hypot(environment.target_x - state["x"],
                                  environment.target_y - state["y"])
            result.update(final=state, final_distance=distance)
            if state["health"] <= 0:
                result["status"] = "dead"
                break
            if distance <= environment.goal_radius:
                result["status"] = "success"
                break
            if result["steps"] >= environment.max_steps:
                result["status"] = "max_steps"
                break
            focus()
            sample = TelemetrySample(**state, modified_ns=0)
            action = choose_movement(sample, environment.target_x,
                                     environment.target_y, axis_tolerance)
            _, reward, _, _, info = environment.step(action)
            result["steps"] += 1
            result["total_reward"] += float(reward)
    except (TelemetryError, ActionError) as error:
        result["error"] = str(error)
    except KeyboardInterrupt:
        result["status"] = "interrupted"
    finally:
        environment.close()
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset", nargs=2, type=float, action="append",
                        metavar=("DX", "DY"), help="Repeat for each target, relative to the run start")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.jsonl"))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--duration", type=float, default=0.15)
    parser.add_argument("--countdown", type=int, default=5)
    args = parser.parse_args()
    offsets = args.offset or [(10, 0), (0, 0)]
    if args.max_steps <= 0 or not math.isfinite(args.duration) or args.duration <= 0 or args.countdown < 0:
        parser.error("max-steps and duration must be positive; countdown must be nonnegative")
    if not all(math.isfinite(value) for offset in offsets for value in offset):
        parser.error("offsets must be finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("a", encoding="utf-8") as output:
            for remaining in range(args.countdown, 0, -1):
                print(f"Focus Project Zomboid: starting in {remaining}...", flush=True)
                time.sleep(1)
            wait_for_game_focus()
            reader = TelemetryReader()
            # Require live telemetry before choosing the origin or sending input.
            origin = reader.wait_for_update(reader.read())
            failed = False
            for index, (dx, dy) in enumerate(offsets, 1):
                environment = ZomboidNavigationEnv(
                    target=(origin.x + dx, origin.y + dy), reader=reader,
                    max_steps=args.max_steps, action_duration=args.duration)
                result = run_episode(environment)
                result.update(episode=index, recorded_at=time.time())
                output.write(json.dumps(result, allow_nan=False) + "\n")
                output.flush()
                print(json.dumps(result), flush=True)
                failed |= result["status"] != "success"
                if result["status"] in {"dead", "error", "interrupted"}:
                    return 130 if result["status"] == "interrupted" else 2
            return int(failed)
    except (TelemetryError, ActionError, OSError) as error:
        print(f"Evaluation stopped: {error}")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
