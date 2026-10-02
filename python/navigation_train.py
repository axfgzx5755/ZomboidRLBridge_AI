"""Repeatable PPO train/save/evaluate cycles for obstacle navigation v2."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from navigation import NavigationEnv, SCHEMA
from training_env import Arena
from rl import countdown, write_json
from training_quality import (GeometryAugmentation, add_quality_arguments,
                              ppo_overrides, require_healthy, summarize_episodes)


def append(path, value):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n')


def save(model, base, arena, live, stage, seed, cycle):
    model.save(base)
    write_json(Path(str(base)+'.json'),dict(schema=SCHEMA,arena=asdict(arena),live=live,
               stage=stage,seed=seed,cycle=cycle,timesteps=model.num_timesteps,
               training_quality=getattr(model, 'training_quality', {})))


def load_metadata(path):
    data=json.loads(Path(path).with_suffix('.json').read_text(encoding='utf-8'))
    if data.get('schema')!=SCHEMA:
        raise ValueError('model is not navigation v2; old coordinate-only models cannot be resumed')
    return data


def evaluate(model, env, episodes, seed, output, cycle):
    wins=0
    for episode in range(episodes):
        started=time.monotonic(); total=0.0; info={'steps':0,'stage':env.stage}
        try:
            if isinstance(getattr(env, 'scenarios', None), list):
                obs,info=env.reset(seed=seed+episode, options={'scenario_index': episode % len(env.scenarios)})
            else:
                obs,info=env.reset(seed=seed+episode)
            while True:
                action,_=model.predict(obs,deterministic=True)
                obs,reward,terminated,truncated,info=env.step(int(action))
                total+=reward
                if terminated or truncated: break
        except (Exception,KeyboardInterrupt) as error:
            env.stop()
            append(output,dict(cycle=cycle,episode=episode,seed=seed+episode,reward=total,
                   **{**info,'is_success':False,'terminal_reason':'interrupted' if isinstance(error,KeyboardInterrupt) else 'error'},
                   error_type=type(error).__name__,error=str(error)))
            raise
        finally:
            env.stop()
        wins+=int(info['is_success'])
        append(output,dict(cycle=cycle,episode=episode,seed=seed+episode,reward=total,
                          elapsed_seconds=time.monotonic()-started,**info))
    return wins/episodes


def run(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    torch.set_num_threads(1)
    args.evaluate_only = getattr(args, 'evaluate_only', False)
    overrides = ppo_overrides(args)
    augment = getattr(args, 'augment_geometry', False)
    saved=load_metadata(args.resume) if args.resume else None
    arena=Arena.load(args.arena) if args.arena else Arena(**saved['arena']) if saved else Arena(allow_obstacles=True)
    if args.live and not args.arena and not (saved and saved['live']):
        raise ValueError('live sessions require a captured --arena')
    if saved and args.live==saved['live'] and asdict(arena)!=saved['arena'] and not args.transfer:
        raise ValueError('resume arena mismatch; use --transfer to explicitly adapt to a new arena')
    if saved and args.live!=saved['live'] and not args.transfer:
        raise ValueError('backend change requires --transfer and a new output directory')
    if args.transfer and not saved: raise ValueError('--transfer requires --resume')
    stage=saved['stage'] if saved else args.stage
    seed=saved['seed'] if saved else args.seed
    cycle=saved['cycle'] if saved else 0
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'config.json',dict(schema=SCHEMA,mode='evaluate' if args.evaluate_only else 'train',arena=asdict(arena),live=args.live,
               seed=seed,stage=stage,cycles=args.cycles,steps_per_cycle=args.steps,eval_episodes=args.episodes,
               resume=str(args.resume) if args.resume else None,transfer=args.transfer,
               augment_geometry=augment,ppo_overrides=overrides))
    model=None; raw=None; env=None
    try:
        if args.live: countdown(args.countdown)
        raw=NavigationEnv(arena,live=args.live,stage=stage)
        training = GeometryAugmentation(raw, seed=seed) if augment and not args.evaluate_only else raw
        env=raw if args.evaluate_only else Monitor(training,str(args.output/'episodes'),info_keywords=('is_success','terminal_reason','stage'))
        if saved:
            model=PPO.load(args.resume,env=env,device='cpu',**overrides)
        else:
            settings = dict(n_steps=256,batch_size=64,learning_rate=3e-4,ent_coef=0.01)
            settings.update(overrides)
            model=PPO('MlpPolicy',env,seed=seed,device='cpu',verbose=0,**settings)
        write_json(args.output/'health_before.json', require_healthy(model))
        if not args.evaluate_only:
            model.training_quality = dict(augment_geometry=augment, ppo_overrides=overrides)
        if not args.evaluate_only:
            model.set_logger(configure(str(args.output),['csv']))

        if args.evaluate_only:
            score=evaluate(model,raw,args.episodes,args.seed,args.output/'evaluation.jsonl',cycle)
            records=[json.loads(line) for line in (args.output/'evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
            reasons={}
            for record in records:
                reason=record['terminal_reason']
                reasons[reason]=reasons.get(reason,0)+1
            write_json(args.output/'summary.json',summarize_episodes(records))
            write_json(args.output/'status.json',dict(status='complete',mode='evaluate',episodes=len(records)))
            print(f'evaluation={score:.1%} ({len(records)} episodes)',flush=True)
            return 0

        class Checkpoint(BaseCallback):
            def _on_rollout_end(self): raw.stop()
            def _on_step(self):
                if self.n_calls%args.checkpoint_every==0:
                    raw.stop()
                    save(self.model,args.output/f'checkpoint_{self.num_timesteps}',arena,args.live,raw.stage,seed,cycle)
                return True

        best={}
        for index in range(args.cycles):
            cycle+=1
            print(f'cycle={cycle} stage={raw.stage} training {args.steps} steps; backend={"LIVE" if args.live else "SYNTHETIC"}',flush=True)
            # Reset after separate evaluation. Do not store eval transitions in PPO rollout.
            model.set_env(env,force_reset=True)
            model.learn(total_timesteps=args.steps,reset_num_timesteps=False,callback=Checkpoint())
            raw.stop()
            model.logger.dump(step=model.num_timesteps)
            write_json(args.output/'health_after.json', require_healthy(model))
            base=args.output/f'cycle_{cycle:04d}'
            save(model,base,arena,args.live,raw.stage,seed,cycle)
            score=evaluate(model,raw,args.episodes,seed+100000,args.output/'evaluation.jsonl',cycle)
            records = [json.loads(line) for line in (args.output/'evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
            summary = summarize_episodes([row for row in records if row['cycle'] == cycle])
            write_json(args.output/f'cycle_{cycle:04d}_evaluation.json', summary)
            append(args.output/'cycles.jsonl',dict(cycle=cycle,stage=raw.stage,success_rate=score,timesteps=model.num_timesteps))
            print(f'cycle={cycle} stage={raw.stage} evaluation={score:.1%}',flush=True)
            if score>best.get(raw.stage,-1):
                best[raw.stage]=score
                save(model,args.output/f'best_stage_{raw.stage}',arena,args.live,raw.stage,seed,cycle)
            # Live map geometry is not changed by curriculum; promotion is synthetic only.
            if not args.live and score>=args.promote_at and raw.stage<2:
                raw.stage+=1
            save(model,args.output/'latest',arena,args.live,raw.stage,seed,cycle)
        write_json(args.output/'status.json',dict(status='complete',cycle=cycle,timesteps=model.num_timesteps))
        return 0
    except (Exception,KeyboardInterrupt) as error:
        if raw: raw.stop()
        if model is not None and not args.evaluate_only:
            save(model,args.output/'interrupted',arena,args.live,raw.stage,seed,cycle)
        write_json(args.output/'status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',
                   cycle=cycle,error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        try:
            if env: env.close()
            elif raw: raw.close()
        finally:
            if model is not None and hasattr(model, "_logger"):
                model.logger.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--live',action='store_true')
    p.add_argument('--arena',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--transfer',action='store_true',help='explicitly allow a new arena or synthetic/live transfer; re-evaluate after transfer')
    p.add_argument('--evaluate-only',action='store_true',help='evaluate a saved policy without training')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=10)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--stage',type=int,choices=(0,1,2),default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--checkpoint-every',type=int,default=2048)
    p.add_argument('--promote-at',type=float,default=0.8)
    p.add_argument('--countdown',type=int,default=15)
    add_quality_arguments(p)
    args=p.parse_args()
    if min(args.cycles,args.steps,args.episodes,args.checkpoint_every)<=0 or args.seed<0 or args.countdown<0 or not 0<args.promote_at<=1:
        p.error('invalid count, seed, countdown or promotion threshold')
    if args.evaluate_only and not args.resume:
        p.error('--evaluate-only requires --resume MODEL.zip')
    if args.evaluate_only and (args.augment_geometry or ppo_overrides(args)):
        p.error('augmentation and PPO overrides are training options')
    try: return run(args)
    except KeyboardInterrupt:
        print('Interrupted; see status.json and interrupted.zip'); return 130
    except Exception as error:
        print(f'Stopped: {type(error).__name__}: {error}'); return 2

if __name__=='__main__':
    raise SystemExit(main())
