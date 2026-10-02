"""No-API free-play exploration. No training reset, teleport, or stat changes."""
import argparse
import json
import math
from pathlib import Path
import time
import uuid

from actions import HeldMovement, is_project_zomboid_foreground, ActionError
from bridge import TrainingBridge, BridgeError, atomic_text
from local_explorer import Explorer, Memory
from navigation import Grid


class ObservationTimeout(RuntimeError): pass


def validate_snapshot(data):
    if data.get('schema')!=1: raise ValueError('unsupported action bridge schema')
    p=data['player']
    for key in ('x','y','z','health'):
        if type(p[key]) not in (float,int) or not math.isfinite(p[key]):
            raise ValueError('invalid player observation')
    if type(p['dead']) is not bool or type(p['vehicle']) is not bool:
        raise ValueError('invalid player flags')
    if p['dead'] or p['health']<=0: raise RuntimeError('player is dead; load a living character manually')
    if p['vehicle'] or abs(p['z'])>0.1: raise RuntimeError('ground-floor character outside vehicle required')
    grid=Grid.parse(data['map'],radius=4)
    if (grid.x,grid.y)!=(math.floor(p['x']),math.floor(p['y'])):
        raise ValueError('map/player mismatch')
    if not isinstance(data['doors'],list): raise ValueError('invalid doors')
    zombies=data.get('zombies',[])
    if not isinstance(zombies,list): raise ValueError('invalid zombie observations')
    for zombie in zombies:
        if not isinstance(zombie,dict) or any(type(zombie.get(k)) not in (int,float) or not math.isfinite(zombie[k]) for k in ('x','y')):
            raise ValueError('invalid zombie position')
        if abs(zombie['x']-p['x'])>8.1 or abs(zombie['y']-p['y'])>8.1:
            raise ValueError('zombie outside observation radius')
    places=data.get('places')
    if not isinstance(places,dict) or places.get('schema')!=1 or not isinstance(places.get('tiles'),list):
        raise ValueError('room observations unavailable; reload the mod with Places.lua')
    seen=set()
    for tile in places['tiles']:
        point=(tile['x'],tile['y'])
        if any(type(v) is not int for v in point) or point in seen or abs(point[0]-grid.x)>4 or abs(point[1]-grid.y)>4:
            raise ValueError('invalid room tile coordinates')
        if any(type(tile[k]) is not str for k in ('building','room','name')):
            raise ValueError('invalid room labels')
        seen.add(point)
    if (grid.x,grid.y) not in seen: raise ValueError('current tile has no place observation')
    for d in data['doors']:
        if any(type(d[k]) is not int for k in ('x','y','z')) or d['z']!=0:
            raise ValueError('invalid door coordinates')
        if abs(d['x']-grid.x)>4 or abs(d['y']-grid.y)>4: raise ValueError('door outside observation')
        if any(type(d[k]) is not bool for k in ('north','open','locked','barricaded','supported')):
            raise ValueError('invalid door flags')
    return data


class FreePlayBridge:
    def __init__(self):
        self.owner=TrainingBridge()
        self.owner.acquire()  # Same lock prevents PPO and free play controlling together.

    def request(self,operation='observe',door=None):
        if not is_project_zomboid_foreground(): raise ActionError('game lost focus')
        self.owner.heartbeat()
        request_id=uuid.uuid4().hex
        d=door or dict(x=0,y=0,z=0,north=False)
        text=f"{request_id} {self.owner.session} {time.time()+4:.3f} {operation} {d['x']} {d['y']} {d['z']} {int(d['north'])}\n"
        atomic_text(self.owner.directory/'pz_action_command.txt',text)
        deadline=time.monotonic()+4
        while time.monotonic()<deadline:
            if not is_project_zomboid_foreground(): raise ActionError('game lost focus')
            try:
                data=json.loads((self.owner.directory/'pz_action_ack.txt').read_text(encoding='utf-8'))
            except (OSError,ValueError):
                data={}
            if data.get('id')==request_id:
                if data.get('status')!='ok': raise BridgeError(data.get('error','Lua action failed'))
                return validate_snapshot(data)
            time.sleep(0.05)
        raise ObservationTimeout('no fresh action reply; unpause or reload the updated mod')

    def close(self): self.owner.close()


def movement_towards(player,tile):
    dx,dy=tile[0]+0.5-player['x'],tile[1]+0.5-player['y']
    if abs(dx)>=abs(dy):
        action=8 if dx>0 else 5  # Default isometric WASD: +X=S+D, -X=W+A.
        distance=abs(dx)
    else:
        action=7 if dy>0 else 6
        distance=abs(dy)
    return action,max(0.025,min(0.15,distance/4))


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    atomic_text(args.output/'config.json',json.dumps(vars(args),default=str,indent=2))
    memory=bridge=movement=None
    status='starting'; steps=0; error=None
    try:
        memory=Memory(args.memory,args.world)
        planner=Explorer(memory)
        bridge=FreePlayBridge()
        movement=HeldMovement()
        for remaining in range(args.countdown,0,-1):
            print(f'Focus and unpause Project Zomboid: {remaining}',flush=True)
            bridge.owner.heartbeat(); time.sleep(1)
        snapshot=None; paused=False; failures=0
        deadline=time.monotonic()+args.minutes*60 if args.minutes else math.inf
        print('Local exploration started. No API. Ctrl+C stops. Losing focus pauses input.',flush=True)
        while time.monotonic()<deadline:
            bridge.owner.heartbeat()
            if not is_project_zomboid_foreground():
                movement.stop(); snapshot=None
                if not paused: print('Paused: game not focused.',flush=True)
                paused=True; time.sleep(0.25); continue
            if paused:
                # Visible pause on return gives the user time to unpause or leave the game again.
                print('Game focused: resuming in 3 seconds.',flush=True)
                for _ in range(3):
                    bridge.owner.heartbeat(); time.sleep(1)
                paused=False; snapshot=None
                if not is_project_zomboid_foreground(): continue
            try:
                snapshot=snapshot or bridge.request()
                p=snapshot['player']
                if p['health']<args.min_health:
                    status='low_health'; break
                action=planner.plan(snapshot)
                if action['kind']=='open':
                    result=bridge.request('open',action['door'])
                elif action['kind'] in ('move','flee'):
                    direction,duration=movement_towards(p,action['tile'])
                    try: movement.move(direction,duration)
                    finally: movement.stop()
                    result=bridge.request()
                else:
                    time.sleep(0.5); result=bridge.request()
                planner.feedback(action,p,result['player'],result['outcome'])
                snapshot=result; steps+=1; failures=0
                if steps%10==0 or action['kind'] in ('open','flee'):
                    print(f"step={steps} action={action['kind']} outcome={result['outcome']} position=({result['player']['x']:.2f},{result['player']['y']:.2f})",flush=True)
                if steps%50==0:
                    atomic_text(args.output/'exploration.json',json.dumps(memory.report(),ensure_ascii=False,indent=2))
                    atomic_text(args.output/'status.json',json.dumps(dict(status='running',steps=steps,player=result['player'])))
            except ActionError:
                movement.stop(); snapshot=None
                if is_project_zomboid_foreground(): raise
                paused=True
            except ObservationTimeout as exc:
                movement.stop(); snapshot=None; failures+=1
                print(f'Waiting for game ({failures}/15): {exc}',flush=True)
                if failures>=15: raise
                time.sleep(0.5)
        else:
            status='time_limit'
        return 0 if status=='time_limit' else 1
    except KeyboardInterrupt:
        status='interrupted'; return 130
    except Exception as exc:
        status='error'; error=f'{type(exc).__name__}: {exc}'
        print(error,flush=True); return 2
    finally:
        try:
            if movement: movement.close()
        finally:
            try:
                if bridge: bridge.close()
            finally:
                if memory:
                    try:
                        atomic_text(args.output/'exploration.json',json.dumps(memory.report(),ensure_ascii=False,indent=2))
                    finally: memory.close()
                atomic_text(args.output/'status.json',json.dumps(dict(status=status,steps=steps,error=error),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world',required=True,help='unique save+character label; reuse to retain exploration memory')
    parser.add_argument('--minutes',type=float,default=5,help='0 runs until interrupted/error/death/low health')
    parser.add_argument('--countdown',type=int,default=15)
    parser.add_argument('--min-health',type=float,default=0.8)
    parser.add_argument('--memory',type=Path,default=Path('runs/local-memory.sqlite3'))
    parser.add_argument('--output',type=Path,required=True,help='new run directory')
    args=parser.parse_args()
    if not args.world.strip() or not math.isfinite(args.minutes) or args.minutes<0 or args.countdown<0 or not 0<args.min_health<=1:
        parser.error('invalid world, duration, countdown or health threshold')
    try: return run(args)
    except Exception as exc:
        print(f'Stopped: {exc}'); return 2


if __name__=='__main__': raise SystemExit(main())
