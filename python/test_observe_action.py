"""Manual Phase 5 smoke test: observation -> movement -> observation."""

from __future__ import annotations

import argparse
import time

from actions import Movement, foreground_window_title, move
from telemetry_reader import STATE_PATH, read_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "direction",
        nargs="?",
        choices=[direction.name.lower() for direction in Movement if direction],
        default="north",
    )
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--settle", type=float, default=0.75)
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="seconds allowed to focus Project Zomboid before input",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually send input; without this flag the test is read-only",
    )
    args = parser.parse_args()

    before = read_state()
    if before is None:
        print(f"No telemetry available at {STATE_PATH}")
        return 2

    direction = Movement[args.direction.upper()]
    print(f"telemetry: {STATE_PATH}")
    print(f"foreground: {foreground_window_title()!r}")
    print(f"before: {before}")

    if not args.execute:
        print("dry run: add --execute while Project Zomboid is focused")
        return 0

    if args.countdown < 0:
        parser.error("--countdown must be non-negative")
    for remaining in range(args.countdown, 0, -1):
        print(f"Focus Project Zomboid: action begins in {remaining}...", flush=True)
        time.sleep(1)

    print(f"action: {direction.name} for {args.duration:.3f}s")
    move(direction, args.duration)
    time.sleep(args.settle)

    after = read_state()
    if after is None:
        print("Telemetry disappeared after action")
        return 3

    dx = float(after["x"]) - float(before["x"])
    dy = float(after["y"]) - float(before["y"])
    dz = float(after["z"]) - float(before["z"])
    print(f"after:  {after}")
    print(f"delta:  x={dx:+.6f} y={dy:+.6f} z={dz:+.6f}")

    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        print("FAIL: telemetry position did not change")
        return 1

    print("PASS: observation -> action -> changed observation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
