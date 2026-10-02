"""Fixed-budget, from-scratch MLP vs grid CNN PPO comparison (synthetic only)."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import t, binomtest
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from grid_cnn import GridCNN
from navigation import NavigationEnv, SCHEMA
from navigation_train import evaluate
from rl import write_json
from training_env import Arena
from training_quality import require_healthy, summarize_episodes


def compare(results):
    seeds = sorted({row['seed'] for row in results})
    pairs = []
    for seed in seeds:
        mlp, cnn = [next(row for row in results if row['seed'] == seed and row['architecture'] == arch)
                    for arch in ('mlp', 'cnn')]
        m, c = [np.array(row['successes'], dtype=bool) for row in (mlp, cnn)]
        cnn_only, mlp_only = int((c & ~m).sum()), int((m & ~c).sum())
        pairs.append(dict(seed=seed, success_difference_pp=100*float(c.mean()-m.mean()),
                          cnn_only=cnn_only, mlp_only=mlp_only,
                          paired_map_binomial_p=float(binomtest(cnn_only, cnn_only+mlp_only).pvalue)
                          if cnn_only+mlp_only else 1.0))
    delta = np.array([row['success_difference_pp'] for row in pairs])
    margin = float(t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None
    aggregate = {}
    for arch in ('mlp', 'cnn'):
        rows = [row for row in results if row['architecture'] == arch]
        aggregate[arch] = dict(
            mean_success_rate=float(np.mean([r['evaluation']['success_rate'] for r in rows])),
            mean_steps=float(np.mean([r['evaluation']['mean_steps'] for r in rows])),
            mean_training_seconds=float(np.mean([r['training_seconds'] for r in rows])),
            parameter_count=rows[0]['parameter_count'])
    return dict(by_architecture=aggregate, paired_training_seeds=pairs,
                mean_success_difference_pp=float(delta.mean()),
                training_seed_t95_difference_pp=[float(delta.mean()-margin), float(delta.mean()+margin)] if margin is not None else None,
                cnn_training_time_ratio=aggregate['cnn']['mean_training_seconds']/aggregate['mlp']['mean_training_seconds'],
                note='Paired training-seed interval conditional on the fixed test map set; only 3 seeds by default. Shared maps are not independent across models. Per-seed p-values are descriptive, without multiple-comparison correction.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-steps', type=int, nargs=3, default=[8192, 8192, 16384])
    parser.add_argument('--seeds', type=int, nargs='+', default=[71, 72, 73])
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--test-seed', type=int, default=2100000)
    args = parser.parse_args()
    if min(args.stage_steps) <= 0 or any(n % 256 for n in args.stage_steps):
        parser.error('stage steps must be positive multiples of 256')
    if args.episodes <= 0 or args.test_seed < 0 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('invalid counts/seeds')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    arena = Arena(allow_obstacles=True)
    settings = dict(n_steps=256, batch_size=64, learning_rate=3e-4, n_epochs=10,
                    ent_coef=.01, gamma=.99, gae_lambda=.95, clip_range=.2,
                    vf_coef=.5, max_grad_norm=.5, target_kl=None)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'config.json', dict(
        stage_steps=args.stage_steps, seeds=args.seeds, test_seed=args.test_seed, episodes=args.episodes,
        arena=asdict(arena), ppo=settings, initialization='from scratch for both architectures',
        augmentation=False, selection='fixed final checkpoints, no test-based tuning',
        torch=torch.__version__, threads=1))
    results = []
    for seed in args.seeds:
        # Alternate execution order to reduce consistent timing order effects.
        order = ('mlp', 'cnn') if seed % 2 else ('cnn', 'mlp')
        for arch in order:
            directory = args.output/f'{arch}_{seed}'
            directory.mkdir()
            raw = NavigationEnv(arena, stage=0)
            env = Monitor(raw, str(directory/'episodes'), info_keywords=('is_success','terminal_reason','stage'))
            model = None
            try:
                kwargs = dict(net_arch=dict(pi=[64,64], vf=[64,64]))
                if arch == 'cnn':
                    kwargs['features_extractor_class'] = GridCNN
                model = PPO('MlpPolicy', env, policy_kwargs=kwargs, seed=seed, device='cpu', **settings)
                model.set_logger(configure(str(directory), ['csv']))
                write_json(directory/'health_before.json', require_healthy(model))
                elapsed = 0.0
                for stage, steps in enumerate(args.stage_steps):
                    raw.stage = stage
                    model.set_env(env, force_reset=True)
                    started = time.perf_counter()
                    model.learn(total_timesteps=steps, reset_num_timesteps=False)
                    elapsed += time.perf_counter()-started
                    model.logger.dump(step=model.num_timesteps)
                    print(f'{arch}_{seed}: stage={stage} steps={model.num_timesteps} train_seconds={elapsed:.1f}', flush=True)
                health = require_healthy(model)
                write_json(directory/'health_after.json', health)
                model.save(directory/'final')
                write_json(directory/'final.json', dict(schema=SCHEMA, architecture=arch, arena=asdict(arena),
                           live=False, stage=2, seed=seed, cycle=3, timesteps=model.num_timesteps))
                # Evaluate the saved artifact, also checking custom CNN deserialization.
                model.logger.close()
                model = PPO.load(directory/'final.zip', device='cpu')
                test = NavigationEnv(arena, stage=2)
                try:
                    evaluate(model, test, args.episodes, args.test_seed, directory/'evaluation.jsonl', 0)
                finally:
                    test.close()
                rows = [json.loads(line) for line in (directory/'evaluation.jsonl').read_text().splitlines()]
                result = dict(architecture=arch, seed=seed, parameter_count=health['parameter_count'],
                              timesteps=model.num_timesteps, training_seconds=elapsed,
                              evaluation=summarize_episodes(rows), successes=[r['is_success'] for r in rows])
                results.append(result)
                write_json(directory/'summary.json', {k:v for k,v in result.items() if k != 'successes'})
                write_json(args.output/'results.json', results)
                print(f'{arch}_{seed}: success={result["evaluation"]["success_rate"]:.1%}', flush=True)
            finally:
                env.close()
                # Original logger remains attached to env-independent training model.
                # Loaded models do not have a configured logger.
                if model is not None and hasattr(model, '_logger'):
                    model.logger.close()
    summary = compare(results)
    write_json(args.output/'summary.json', summary)
    write_json(args.output/'status.json', dict(status='complete', deployment='unchanged'))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
