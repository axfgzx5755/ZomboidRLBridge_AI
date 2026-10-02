"""Capture an arena, train/resume PPO, or evaluate a saved navigation policy."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from bridge import BridgeError, atomic_text
from telemetry import TelemetryReader, TelemetryError
from actions import ActionError
from training_env import Arena, TrainingEnv


def countdown(seconds):
    for remaining in range(seconds, 0, -1):
        print(f"Focus and unpause Project Zomboid: {remaining}...", flush=True)
        time.sleep(1)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def capture(args):
    reader = TelemetryReader()
    sample = reader.wait_for_update(reader.read(), timeout=3)
    # Build 42 teleportTo settles at integer tile coordinates in live testing.
    arena = Arena(x=math.floor(sample.x), y=math.floor(sample.y), z=sample.z, radius=args.radius,
                  min_distance=args.min_distance, max_distance=args.max_distance,
                  allow_obstacles=args.allow_obstacles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError(f"arena file already exists: {args.output}; choose a new --output")
    write_json(args.output, asdict(arena))
    print(f"Arena origin captured: {args.output}. Lua will validate open ground at reset.")
    return 0


def load_configuration(args):
    model_path = args.resume if args.command == "train" else args.model
    metadata = None
    if model_path:
        metadata = json.loads(Path(str(model_path).removesuffix(".zip") + ".json").read_text())
    arena = Arena.load(args.arena) if args.arena else Arena(**metadata["arena"]) if metadata else Arena()
    if args.live and args.arena is None and not (metadata and metadata["backend"] == "live"):
        raise ValueError("live control requires an arena captured from the game (--arena)")
    if metadata and args.command == "train" and asdict(arena) != asdict(Arena(**metadata["arena"])):
        raise ValueError("resumed training must use the saved arena configuration")
    if metadata and arena.radius != metadata["arena"]["radius"]:
        raise ValueError("arena radius must match the model's observation scaling")
    return arena


def save_model(model, path, arena, live):
    model.save(path)
    write_json(Path(str(path) + ".json"), {
        "schema": 1, "backend": "live" if live else "synthetic", "arena": asdict(arena),
        "timesteps": model.num_timesteps, "observation": "dx,dy,z,health,distance",
    })


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure

    torch.set_num_threads(1)
    arena = load_configuration(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", {"arena": asdict(arena), "live": args.live,
               "seed": args.seed, "requested_timesteps": args.timesteps, "resume": str(args.resume) if args.resume else None})
    if args.live:
        countdown(args.countdown)
    environment = TrainingEnv(arena, live=args.live)
    model = None

    class Checkpoint(BaseCallback):
        def _on_rollout_end(self):
            # PPO optimization can take longer than the input watchdog lease.
            environment.unwrapped.backend.stop()

        def _on_step(self):
            if self.n_calls % args.checkpoint_every == 0:
                save_model(self.model, args.output / f"checkpoint_{self.num_timesteps}", arena, args.live)
            return True

    try:
        environment = Monitor(environment, str(args.output / "episodes"),
                              info_keywords=("is_success", "terminal_reason"))
        if args.resume:
            model = PPO.load(args.resume, env=environment, device="cpu")
        else:
            model = PPO("MlpPolicy", environment, n_steps=args.rollout_steps, batch_size=args.batch_size,
                        n_epochs=10, learning_rate=3e-4, ent_coef=0.01, seed=args.seed, device="cpu", verbose=1)
        model.set_logger(configure(str(args.output), ["stdout", "csv"]))
        print(f"Backend: {'LIVE GAME' if args.live else 'SYNTHETIC TEST ONLY'}", flush=True)
        model.learn(total_timesteps=args.timesteps, callback=Checkpoint(), reset_num_timesteps=not bool(args.resume))
        save_model(model, args.output / "final", arena, args.live)
        print(f"Saved {args.output / 'final.zip'}")
        return 0
    except BaseException:
        if model is not None:
            save_model(model, args.output / "interrupted", arena, args.live)
        raise
    finally:
        environment.close()


def evaluate(args):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    arena = load_configuration(args)
    model = PPO.load(args.model, device="cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.live:
        countdown(args.countdown)
    environment = TrainingEnv(arena, live=args.live)
    successes = 0
    try:
        with args.output.open("a", encoding="utf-8") as output:
            for episode in range(args.episodes):
                info = {"steps": 0, "allow_obstacles": arena.allow_obstacles}
                total_reward = 0.0
                started = time.monotonic()
                failure = None
                try:
                    observation, reset_info = environment.reset(seed=args.seed + episode)
                    info.update(reset_info)
                    while True:
                        action, _ = model.predict(observation, deterministic=True)
                        observation, reward, terminated, truncated, step_info = environment.step(int(action))
                        info.update(step_info)
                        total_reward += reward
                        if terminated or truncated:
                            break
                except (Exception, KeyboardInterrupt) as error:
                    failure = error
                    info.update(is_success=False,
                                terminal_reason="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                                error_type=type(error).__name__, error=str(error))
                result = {"episode": episode + 1, "seed": args.seed + episode,
                          "backend": "live" if args.live else "synthetic", "model": str(args.model),
                          "reward": total_reward, "elapsed_seconds": time.monotonic() - started, **info}
                output.write(json.dumps(result, allow_nan=False) + "\n")
                output.flush()
                if failure is not None:
                    raise failure
                successes += int(info["is_success"])
                print(f"episode={episode + 1} status={info['terminal_reason']} reward={total_reward:.3f}", flush=True)
        print(f"Success: {successes}/{args.episodes} ({successes / args.episodes:.1%}); results: {args.output}")
        return 0
    finally:
        environment.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("capture", help="capture the live position as arena origin")
    setup.add_argument("--output", type=Path, default=Path("arena.json"))
    setup.add_argument("--radius", type=float, default=8)
    setup.add_argument("--min-distance", type=float, default=2)
    setup.add_argument("--max-distance", type=float, default=5)
    setup.add_argument("--allow-obstacles", action="store_true",
                       help="validate reset/goal tiles only; does not provide obstacle avoidance")
    training = commands.add_parser("train", help="train or resume PPO (synthetic unless --live)")
    training.add_argument("--timesteps", type=int, default=10000)
    training.add_argument("--rollout-steps", type=int, default=256)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--checkpoint-every", type=int, default=1024)
    training.add_argument("--resume", type=Path)
    training.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("ppo-%Y%m%d-%H%M%S"))
    evaluation = commands.add_parser("evaluate", help="run a saved model and record success rate")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--episodes", type=int, default=10)
    evaluation.add_argument("--output", type=Path, default=Path("runs/policy-evaluation.jsonl"))
    for command in (training, evaluation):
        command.add_argument("--live", action="store_true", help="control the live single-player game")
        command.add_argument("--arena", type=Path)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--countdown", type=int, default=5)
    args = parser.parse_args()
    for name in ("timesteps", "rollout_steps", "batch_size", "checkpoint_every", "episodes"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if getattr(args, "countdown", 0) < 0 or getattr(args, "seed", 0) < 0:
        parser.error("countdown and seed must be nonnegative")
    if args.command == "train" and (args.batch_size < 2 or args.rollout_steps < 2 or args.rollout_steps % args.batch_size):
        parser.error("rollout-steps must be divisible by batch-size; both must be >= 2")
    return args


def main():
    args = parse_args()
    try:
        return {"capture": capture, "train": train, "evaluate": evaluate}[args.command](args)
    except KeyboardInterrupt:
        print("Stopped by user; training checkpoint saved if a model was initialized.")
        return 130
    except (BridgeError, TelemetryError, ActionError, OSError, ValueError) as error:
        print(f"Stopped: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
