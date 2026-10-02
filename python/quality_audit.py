"""Inspect trusted local PPO checkpoints and existing episode logs without a game."""
import argparse
import csv
import json
import math
from pathlib import Path

from rl import write_json
from training_quality import model_health, summarize_episodes


def audit_model(path):
    from stable_baselines3 import PPO
    model = PPO.load(path, device='cpu')
    report = dict(path=str(path.resolve()), **model_health(model))
    report['metadata'] = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    progress = path.parent / 'progress.csv'
    if progress.exists():
        last = {}
        with progress.open(encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                for key, value in row.items():
                    if key and key.startswith('train/') and value:
                        number = float(value)
                        last[key] = number if math.isfinite(number) else None
        report['last_logged_training_metrics'] = last
        report['metrics_note'] = 'Older SB3 logs may omit the final optimizer update; weights and logs differ in time.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, nargs='+', required=True)
    parser.add_argument('--evaluations', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    models = [audit_model(path) for path in args.models]
    evaluations = []
    for path in args.evaluations:
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        evaluations.append(dict(path=str(path.resolve()), **summarize_episodes(records)))
    write_json(args.output/'audit.json', dict(models=models, evaluations=evaluations))
    for model in models:
        print(f"{model['path']}: parameters={model['parameter_count']}, finite={model['parameters_finite']}, steps={model['timesteps']}")


if __name__ == '__main__':
    main()
