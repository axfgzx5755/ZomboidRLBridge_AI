"""Compare saved navigation policies on identical synthetic evaluation seeds."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from navigation import NavigationEnv
from navigation_train import evaluate, load_metadata
from rl import write_json
from training_env import Arena


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=200000)
    parser.add_argument('--stage', type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args()
    if args.episodes <= 0 or args.seed < 0:
        parser.error('episodes must be positive and seed nonnegative')

    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    metadata = [load_metadata(path) for path in args.models]
    if any(item['live'] for item in metadata):
        parser.error('this benchmark accepts synthetic checkpoints only')
    arena = Arena(**metadata[0]['arena'])
    if any(item['arena'] != asdict(arena) for item in metadata):
        parser.error('models must have identical arena settings for comparison')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'config.json', dict(
        models=[str(path) for path in args.models], arena=asdict(arena),
        stage=args.stage, seed=args.seed, episodes=args.episodes, backend='synthetic'))
    results = []
    for index, path in enumerate(args.models):
        env = NavigationEnv(arena, live=False, stage=args.stage)
        records_path = args.output / f'model_{index}_episodes.jsonl'
        try:
            model = PPO.load(path, device='cpu')
            score = evaluate(model, env, args.episodes, args.seed, records_path, 0)
        finally:
            env.close()
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        reasons = {}
        for record in records:
            reason = record['terminal_reason']
            reasons[reason] = reasons.get(reason, 0) + 1
        result = dict(model=str(path), success_rate=score, terminal_reasons=reasons,
                      mean_reward=sum(r['reward'] for r in records) / len(records),
                      mean_steps=sum(r['steps'] for r in records) / len(records))
        results.append(result)
        print(json.dumps(result), flush=True)
        write_json(args.output / 'summary.json', results)


if __name__ == '__main__':
    main()
