"""Equal-budget synthetic navigation control/augmentation pilot; never deploys."""
import argparse
import json
from pathlib import Path

from navigation import NavigationEnv
from navigation_train import evaluate, load_metadata, save
from rl import write_json
from training_env import Arena
from training_quality import GeometryAugmentation, require_healthy, summarize_episodes


def score(model, arena, episodes, seed, directory, name):
    env = NavigationEnv(arena, stage=2)
    path = directory / (name + '.jsonl')
    try:
        evaluate(model, env, episodes, seed, path, 0)
    finally:
        env.close()
    return summarize_episodes([json.loads(line) for line in path.read_text().splitlines()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--seeds', type=int, nargs='+', default=[43, 44, 45])
    parser.add_argument('--validation-episodes', type=int, default=100)
    parser.add_argument('--test-episodes', type=int, default=300)
    parser.add_argument('--validation-seed', type=int, default=1100000)
    parser.add_argument('--test-seed', type=int, default=1200000)
    args = parser.parse_args()
    if min(args.steps, args.validation_episodes, args.test_episodes) <= 0 or min(args.seeds) < 0:
        parser.error('positive counts and nonnegative seeds required')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('training seeds must be distinct')
    if min(args.validation_seed, args.test_seed) < 0 or max(args.validation_seed, args.test_seed) < min(
            args.validation_seed + args.validation_episodes, args.test_seed + args.test_episodes):
        parser.error('validation/test seed ranges must be nonnegative and disjoint')
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    import torch
    torch.set_num_threads(1)
    metadata = load_metadata(args.model)
    if metadata['live']:
        parser.error('only synthetic navigation checkpoints are supported')
    arena = Arena(**metadata['arena'])
    args.output.mkdir(parents=True, exist_ok=False)
    settings = dict(learning_rate=1e-4, n_epochs=5, target_kl=0.015, ent_coef=0.01)
    write_json(args.output/'config.json', dict(model=str(args.model.resolve()), steps=args.steps,
               seeds=args.seeds, validation_seed=args.validation_seed, test_seed=args.test_seed,
               validation_episodes=args.validation_episodes, test_episodes=args.test_episodes,
               settings=settings, backend='synthetic', stage=2))
    baseline = PPO.load(args.model, device='cpu')
    write_json(args.output/'baseline_health.json', require_healthy(baseline))
    results = [dict(name='baseline', model=str(args.model.resolve()),
                    validation=score(baseline, arena, args.validation_episodes,
                                     args.validation_seed, args.output, 'baseline_validation'))]
    for seed in args.seeds:
        for augment in (False, True):
            name = f'{"augmented" if augment else "control"}_{seed}'
            directory = args.output/name
            directory.mkdir()
            raw = NavigationEnv(arena, stage=2)
            env = Monitor(GeometryAugmentation(raw, seed=seed) if augment else raw,
                          str(directory/'episodes'), info_keywords=('is_success','terminal_reason','stage'))
            model = None
            try:
                model = PPO.load(args.model, env=env, device='cpu', seed=seed, **settings)
                write_json(directory/'health_before.json', require_healthy(model))
                model.set_logger(configure(str(directory), ['csv']))
                model.learn(total_timesteps=args.steps, reset_num_timesteps=False)
                model.logger.dump(step=model.num_timesteps)
                write_json(directory/'health_after.json', require_healthy(model))
                model.training_quality = dict(augment_geometry=augment, ppo_overrides=settings)
                save(model, directory/'final', arena, False, 2, seed, metadata['cycle']+1)
                result = dict(name=name, model=str((directory/'final.zip').resolve()),
                              actual_steps=model.num_timesteps-baseline.num_timesteps,
                              validation=score(model, arena, args.validation_episodes,
                                               args.validation_seed, directory, 'validation'))
                results.append(result)
                print(f"{name}: validation={result['validation']['success_rate']:.1%}", flush=True)
            finally:
                env.close()
                if model is not None and hasattr(model, '_logger'):
                    model.logger.close()
            write_json(args.output/'validation.json', results)
    # Freeze selection before reading the held-out test. Ties retain baseline.
    selected = max(results, key=lambda row: row['validation']['success_rate'])['name']
    write_json(args.output/'selection.json', dict(selected=selected, criterion='validation success rate; ties keep earlier candidate'))
    for result in results:
        model = PPO.load(result['model'], device='cpu')
        result['test'] = score(model, arena, args.test_episodes, args.test_seed,
                               args.output, result['name']+'_test')
        print(f"{result['name']}: test={result['test']['success_rate']:.1%}", flush=True)
    write_json(args.output/'summary.json', dict(selected_on_validation=selected, results=results,
               deployment='unchanged; synthetic pilot only; inspect paired controls and live regression before adoption'))
    write_json(args.output/'status.json', dict(status='complete'))


if __name__ == '__main__':
    main()
