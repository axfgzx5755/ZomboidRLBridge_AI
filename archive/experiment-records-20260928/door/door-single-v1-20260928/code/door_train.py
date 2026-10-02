"""User-run live door capture, interaction check, PPO training and evaluation."""
import argparse
import json
import math
from pathlib import Path

from door_env import DoorEnv, INTERACT, SCHEMA, door_request, load_scenario, scenario_from_sample
from navigation_train import evaluate
from rl import countdown, write_json
from training_env import LiveBackend


def capture(args):
    if args.output.exists():
        raise ValueError('choose a new output filename')
    countdown(args.countdown)
    backend = LiveBackend()
    try:
        sample = backend.wait_for_update(backend.read(),timeout=3)
        if sample.z != 0 or sample.health <= 0:
            raise ValueError('living ground-floor character required')
        door = door_request(backend,'door_scan',dict(x=math.floor(sample.x),y=math.floor(sample.y),z=0))
        scenario = scenario_from_sample(sample,door)
        write_json(args.output,scenario)
        print(f"Captured {args.output}; door={scenario['door']}; target={scenario['target']}")
    finally:
        backend.close()


def metadata(path):
    data = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA:
        raise ValueError('requires a door-v1 model; navigation models have incompatible actions/observations')
    return data


def run(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    import torch

    torch.set_num_threads(1)
    scenario = load_scenario(args.scenario)
    saved = metadata(args.model) if args.model else None
    if saved and saved['scenario'] != scenario:
        raise ValueError('model/scenario mismatch; use the original captured scenario')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'config.json',dict(schema=SCHEMA,scenario=scenario,mode=args.mode,
               seed=args.seed,steps=args.steps,episodes=args.episodes,model=str(args.model) if args.model else None))
    raw = env = model = None

    def save(name):
        model.save(args.output/name)
        write_json(args.output/(name+'.json'),dict(schema=SCHEMA,scenario=scenario,timesteps=model.num_timesteps))

    try:
        countdown(args.countdown)
        raw = DoorEnv(scenario)
        if args.mode == 'check':
            _,initial = raw.reset(seed=args.seed)
            _,reward,_,_,opened = raw.step(INTERACT)
            if not opened['opened_by_agent'] or not opened['door_open']:
                raise RuntimeError('interaction did not open the door; inspect position, obstruction and console.txt')
            _,reset = raw.reset(seed=args.seed)
            if reset['door_open'] or reset['opened_by_agent']:
                raise RuntimeError('second reset failed')
            write_json(args.output/'check.json',dict(initial=initial,opened=opened,reward=reward,reset=reset))
            print('PASS: closed reset -> opened by interaction -> closed reset. Crossing and learning are not tested.')
        else:
            env = Monitor(raw,str(args.output/'episodes'),info_keywords=('is_success','terminal_reason','stage'))
            model = PPO.load(args.model,env=env,device='cpu') if saved else PPO(
                'MlpPolicy',env,n_steps=256,batch_size=64,learning_rate=3e-4,ent_coef=0.02,
                seed=args.seed,device='cpu',verbose=0)
            model.set_logger(configure(str(args.output),['csv']))
            if args.mode == 'train':
                class Checkpoint(BaseCallback):
                    def _on_rollout_end(self): raw.stop()
                    def _on_step(self):
                        if self.n_calls % 2048 == 0:
                            raw.stop()
                            save(f'checkpoint_{self.num_timesteps}')
                        return True
                model.learn(total_timesteps=args.steps,reset_num_timesteps=False,callback=Checkpoint())
                raw.stop()
                save('final')
            score = evaluate(model,raw,args.episodes,args.seed+100000,args.output/'evaluation.jsonl',0)
            print(f'Door crossing success: {score:.1%} ({args.episodes} episodes)')
            write_json(args.output/'summary.json',dict(success_rate=score,episodes=args.episodes))
        write_json(args.output/'status.json',dict(status='complete'))
    except (Exception,KeyboardInterrupt) as error:
        if raw: raw.stop()
        if model is not None and args.mode=='train': save('interrupted')
        write_json(args.output/'status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',
                   error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        try:
            if env: env.close()
            elif raw: raw.close()
        finally:
            if model is not None and hasattr(model,'_logger'): model.logger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('capture','check','train','evaluate'))
    parser.add_argument('--scenario',type=Path)
    parser.add_argument('--model',type=Path,help='door-v1 model for resume or evaluation')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--episodes',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--countdown',type=int,default=5)
    args = parser.parse_args()
    if min(args.steps,args.episodes)<=0 or args.seed<0 or args.countdown<0:
        parser.error('counts must be positive; seed/countdown must be nonnegative')
    if args.mode!='capture' and not args.scenario: parser.error('--scenario required')
    if args.mode=='evaluate' and not args.model: parser.error('--model required for evaluate')
    if args.mode in ('capture','check') and args.model: parser.error('--model is only for train/evaluate')
    try:
        if args.mode=='capture': capture(args)
        else: run(args)
        return 0
    except KeyboardInterrupt:
        print('Interrupted; keys released. See the output directory if training started.')
        return 130
    except Exception as error:
        print(f'Stopped: {type(error).__name__}: {error}')
        return 2


if __name__=='__main__':
    raise SystemExit(main())
