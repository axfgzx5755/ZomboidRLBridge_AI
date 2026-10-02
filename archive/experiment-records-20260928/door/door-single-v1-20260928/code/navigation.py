"""Version 2 obstacle navigation: identical local geometry encoding for sim/live."""
from collections import deque
from dataclasses import replace
import json
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from telemetry import TelemetryError
from training_env import Arena, LiveBackend

SCHEMA = 'navigation-v2-134'
# Action indices match actions.Movement; vectors describe default isometric WASD.
DIRECTIONS = ((0,0),(-1,-1),(1,1),(-1,1),(1,-1),(-1,0),(0,-1),(0,1),(1,0))
CARDINAL = ((0,-1),(1,0),(0,1),(-1,0))


class Grid:
    """Row-major channels: occupied, blocked N/E/S/W. One means blocked."""
    def __init__(self, x, y, radius, cells):
        self.x, self.y, self.radius = int(x), int(y), int(radius)
        size = 2*self.radius+1
        self.cells = np.asarray(cells, dtype=np.float32).reshape(size, size, 5)

    @classmethod
    def parse(cls, data, *, radius=None):
        if not isinstance(data, dict) or data.get('schema') != 2:
            raise TelemetryError('navigation v2 telemetry unavailable; reload the updated mod')
        try:
            x, y, z, r = (data[k] for k in ('x','y','z','radius'))
            if any(type(v) not in (int,float) or not math.isfinite(v) or int(v)!=v for v in (x,y,z,r)):
                raise ValueError('noninteger geometry')
            if z != 0 or not 1 <= r <= 20 or (radius is not None and r != radius):
                raise ValueError('unexpected geometry dimensions')
            cells = data['cells']
            if len(cells) != (2*int(r)+1)**2*5 or any(v not in (0,1) for v in cells):
                raise ValueError('invalid geometry cells')
            return cls(x,y,r,cells)
        except (KeyError, TypeError, ValueError) as error:
            raise TelemetryError(f'invalid navigation map: {error}') from error

    def at(self, x, y):
        ix, iy = int(x)-self.x+self.radius, int(y)-self.y+self.radius
        if 0 <= ix < len(self.cells) and 0 <= iy < len(self.cells):
            return self.cells[iy,ix]
        return np.ones(5, dtype=np.float32)

    def edge(self, x, y, dx, dy):
        a, b = self.at(x,y), self.at(x+dx,y+dy)
        i = CARDINAL.index((dx,dy))
        return not (a[0] or b[0] or a[i+1] or b[(i+2)%4+1])

    def transition(self, x, y, nx, ny):
        dx, dy = nx-x, ny-y
        if self.at(nx,ny)[0]: return False
        if not dx and not dy: return True
        if abs(dx)>1 or abs(dy)>1: return False
        if dx and dy:
            # No cutting through a blocked corner.
            return (self.edge(x,y,dx,0) and self.edge(x,y,0,dy)
                    and self.edge(x+dx,y,0,dy) and self.edge(x,y+dy,dx,0))
        return self.edge(x,y,dx,dy)

    def reachable(self, x, y, boundary=None):
        start=(math.floor(x),math.floor(y))
        if self.at(*start)[0]: return set()
        seen={start}; queue=deque([start])
        while queue:
            x,y=queue.popleft()
            for dx,dy in CARDINAL:
                point=(x+dx,y+dy)
                inside = boundary is None or math.hypot(point[0]+0.5-boundary.x, point[1]+0.5-boundary.y) < boundary.radius
                if inside and point not in seen and self.edge(x,y,dx,dy):
                    seen.add(point); queue.append(point)
        return seen

    def local(self, x, y):
        x,y=math.floor(x),math.floor(y)
        return np.concatenate([self.at(sx,sy) for sy in range(y-2,y+3) for sx in range(x-2,x+3)])


def synthetic_grid(arena, rng, stage):
    r=int(arena.radius); n=2*r+1
    occupied = rng.random((n,n)) < (0,0.08,0.18)[stage]
    occupied[0,:]=occupied[-1,:]=True
    occupied[:,0]=occupied[:,-1]=True
    occupied[r-1:r+2,r-1:r+2]=False
    cells=np.zeros((n,n,5),dtype=np.float32)
    cells[:,:,0]=occupied
    for iy in range(n):
        for ix in range(n):
            for k,(dx,dy) in enumerate(CARDINAL):
                j,i=iy+dy,ix+dx
                cells[iy,ix,k+1]=occupied[iy,ix] or not (0<=i<n and 0<=j<n) or occupied[j,i]
    # Advanced stage also has tile-edge walls, leaving cells themselves free.
    if stage == 2:
        for iy in range(1,n-1):
            for ix in range(1,n-2):
                if rng.random()<0.08 and (ix,iy)!=(r,r):
                    cells[iy,ix,2]=cells[iy,ix+1,4]=1
    return Grid(arena.x,arena.y,r,cells)


class NavigationEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, arena=None, *, live=False, stage=0, stall_steps=30):
        self.arena=arena or Arena(allow_obstacles=True)
        if stage not in (0,1,2) or stall_steps<2: raise ValueError('invalid stage/stall limit')
        if any(v != int(v) for v in (self.arena.x,self.arena.y,self.arena.radius)):
            raise ValueError('v2 requires integer arena origin and radius; recapture the arena')
        self.live,self.stage,self.stall_steps=live,stage,stall_steps
        self.action_space=spaces.Discrete(9)
        self.observation_space=spaces.Box(-1,1,shape=(134,),dtype=np.float32)
        self.backend=LiveBackend() if live else None
        self.done=True
        self.velocity=np.zeros(2)

    def stop(self):
        if self.backend: self.backend.stop()

    def _local(self):
        if not self.live: return self.grid.local(*self.position)
        grid=Grid.parse(self.sample.navigation,radius=2)
        if (grid.x,grid.y)!=(math.floor(self.sample.x),math.floor(self.sample.y)):
            raise TelemetryError('navigation geometry and player coordinates disagree')
        return grid.cells.flatten()

    def _observation(self):
        delta=(self.target-self.position)/(self.arena.radius*2)
        base=[*delta, np.linalg.norm(delta), self.health,
              min(self.stalled/self.stall_steps,1), *np.clip(self.velocity,-1,1),
              *(self.position-np.floor(self.position))]
        return np.concatenate([np.clip(base,-1,1),self._local()]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.stop(); self.done=True
        self.steps=self.stalled=0
        self.velocity=np.zeros(2)
        self.health=1.0
        if self.live:
            try:
                arena=replace(self.arena,allow_obstacles=True)
                self.sample=self.backend.reset(arena)
                ack=self.backend.bridge.request('map',arena)
                data=json.loads((self.backend.bridge.directory/'pz_navigation_map.txt').read_text())
                if data.get('id') != ack['id']: raise TelemetryError('stale navigation map reply')
                self.grid=Grid.parse(data['map'],radius=int(arena.radius))
                if (self.grid.x,self.grid.y)!=(arena.x,arena.y): raise TelemetryError('map origin mismatch')
                self.position=np.array([self.sample.x,self.sample.y])
                self.health=self.sample.health
                self._local()  # Fail before movement if the mod lacks v2 observations.
            except BaseException:
                self.stop(); raise
        else:
            self.grid=synthetic_grid(self.arena,self.np_random,self.stage)
            self.position=np.array([self.arena.x,self.arena.y],dtype=float)
        points=sorted(self.grid.reachable(*self.position,boundary=self.arena))
        maximum=min(self.arena.max_distance,(max(3,self.arena.min_distance+0.5),max(4,self.arena.min_distance+0.5),self.arena.max_distance)[self.stage])
        candidates=[(x+0.5,y+0.5) for x,y in points
                    if self.arena.min_distance <= math.hypot(x+0.5-self.arena.x,y+0.5-self.arena.y) <= maximum]
        if not candidates: raise ValueError('no reachable target in configured distance range; recapture or adjust distances')
        self.target=np.array(candidates[int(self.np_random.integers(len(candidates)))])
        self.best_distance=float(np.linalg.norm(self.target-self.position))
        self.done=False
        return self._observation(),self._info('running')

    def _info(self, reason):
        return dict(steps=self.steps,is_success=reason=='success',terminal_reason=reason,
                    stage=self.stage,backend='live' if self.live else 'synthetic',
                    target=self.target.tolist(),position=self.position.tolist(),health=self.health,
                    distance=float(np.linalg.norm(self.target-self.position)))

    def step(self, action):
        if self.done: raise RuntimeError('reset required')
        try:
            if not self.action_space.contains(action): raise ValueError('invalid action')
            previous=self.position.copy()
            before=float(np.linalg.norm(self.target-previous))
            if self.live:
                self.backend.move(int(action),self.arena.action_duration)
                barrier=self.backend.read()
                self.sample=self.backend.wait_for_update(barrier,timeout=1.5)
                self.position=np.array([self.sample.x,self.sample.y]); self.health=self.sample.health
            else:
                dx,dy=DIRECTIONS[int(action)]
                delta=np.array([dx,dy])*self.arena.action_duration*4/max(1,math.hypot(dx,dy))
                # Swept substeps prevent crossing walls on long actions.
                count=max(1,math.ceil(np.linalg.norm(delta)/0.1))
                for _ in range(count):
                    nxt=self.position+delta/count
                    if not self.grid.transition(*map(math.floor,self.position),*map(math.floor,nxt)): break
                    self.position=nxt
            self.steps+=1
            self.velocity=self.position-previous
            distance=float(np.linalg.norm(self.target-self.position))
            improved=distance<self.best_distance-0.05
            self.stalled=0 if improved else self.stalled+1
            if improved: self.best_distance=distance
            outside=np.linalg.norm(self.position-[self.arena.x,self.arena.y])>self.arena.radius
            if self.live: outside=outside or abs(self.sample.z-self.arena.z)>0.1
            reason=('dead' if self.health<=0 else 'out_of_bounds' if outside else
                    'success' if distance<=self.arena.goal_radius else
                    'stalled' if self.stalled>=self.stall_steps else
                    'max_steps' if self.steps>=self.arena.max_steps else 'running')
            reward=before-distance-0.02
            if np.linalg.norm(self.velocity)<0.01: reward-=0.05
            if reason=='success': reward+=10
            if reason in ('dead','out_of_bounds','stalled'): reward-=3
            self.done=reason!='running'
            if self.done: self.stop()
            observation=self._observation()
            return observation,float(reward),reason in ('success','dead','out_of_bounds'),reason in ('stalled','max_steps'),self._info(reason)
        except BaseException:
            self.done=True; self.stop(); raise

    def close(self):
        self.done=True
        if self.backend: self.backend.close()
