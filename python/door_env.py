"""Live single-door curriculum: approach, explicitly open, cross, reach target."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from actions import ActionError, is_project_zomboid_foreground
from navigation import NavigationEnv
from telemetry import TelemetryError
from training_env import Arena

SCHEMA = 'door-v1-142'
INTERACT = 9


def door_request(backend, operation, door):
    if not is_project_zomboid_foreground():
        raise ActionError('focus and unpause Project Zomboid')
    backend.stop()
    request = SimpleNamespace(x=door['x'], y=door['y'], z=door['z'],
                              radius=int(door.get('north', False)))
    ack = backend.bridge.request(operation, request)
    data = json.loads((backend.bridge.directory/'pz_door_state.txt').read_text(encoding='utf-8'))
    if data.get('id') != ack['id'] or data.get('schema') != 1:
        raise TelemetryError('stale or unsupported door reply; reload updated mod')
    for key in ('north', 'open', 'locked', 'adjacent'):
        if type(data.get(key)) is not bool:
            raise TelemetryError(f'invalid door flag: {key}')
    for key in ('x', 'y', 'z'):
        value = data.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value != int(value):
            raise TelemetryError(f'invalid door coordinate: {key}')
    if operation != 'door_scan' and any(data[k] != door[k] for k in ('x','y','z','north')):
        raise TelemetryError('door identity changed')
    return data


def scenario_from_sample(sample, door):
    x, y = math.floor(sample.x), math.floor(sample.y)
    axis = 1 if door['north'] else 0
    edge = door['y'] if axis else door['x']
    side = 1 if (y if axis else x) >= edge else -1
    center = np.array([door['x'] + (0.5 if axis else 0),
                       door['y'] + (0 if axis else 0.5)])
    target = center.copy()
    target[axis] -= side * 1.5
    arena = Arena(x=x, y=y, z=0, radius=4, min_distance=1, max_distance=3,
                  goal_radius=0.35, max_steps=100, allow_obstacles=True)
    return dict(schema=SCHEMA, arena=asdict(arena),
                door={k: door[k] for k in ('x','y','z','north')},
                start_side=side, target=target.tolist())


def load_scenario(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA:
        raise ValueError('capture a door-v1 scenario first')
    arena = Arena(**data['arena'])
    door = data['door']
    if type(door.get('north')) is not bool or data.get('start_side') not in (-1, 1):
        raise ValueError('invalid door orientation or start side')
    if any(type(door.get(k)) is not int for k in ('x','y','z')) or door['z'] != 0:
        raise ValueError('door coordinates must be integer ground-floor tiles')
    expected = scenario_from_sample(SimpleNamespace(x=arena.x, y=arena.y), door)
    sides = [(door['x'],door['y']),
             (door['x']-(not door['north']),door['y']-door['north'])]
    if (arena.x,arena.y) not in sides or data['start_side'] != expected['start_side']:
        raise ValueError('start must be on one of the two door tiles; recapture')
    if data['target'] != expected['target'] or arena.goal_radius > 0.5 or not arena.allow_obstacles:
        raise ValueError('invalid crossing target or arena; recapture')
    return data


def crosses_opening(previous, current, door, start_side):
    """A segment must cross the actual one-tile door edge in the exit direction."""
    axis = 1 if door['north'] else 0
    edge = door['y'] if axis else door['x']
    a, b = previous[axis]-edge, current[axis]-edge
    if a*start_side < 0 or b*start_side >= 0 or a == b:
        return False
    along = previous[1-axis] + (-a/(b-a))*(current[1-axis]-previous[1-axis])
    low = door['x'] if axis else door['y']
    return low <= along < low+1


class DoorEnv(NavigationEnv):
    def __init__(self, scenarios):
        self.scenarios = scenarios if isinstance(scenarios, list) else [scenarios]
        if not self.scenarios:
            raise ValueError('at least one door scenario is required')
        self.scenario = self.scenarios[0]
        self.door = self.scenario['door']
        super().__init__(Arena(**self.scenario['arena']), live=True, stall_steps=40)
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(-1, 1, shape=(142,), dtype=np.float32)

    def _observation(self):
        base = super()._observation()
        center = np.array([self.door['x']+(0.5 if self.door['north'] else 0),
                           self.door['y']+(0 if self.door['north'] else 0.5)])
        extra = [*((center-self.position)/(2*self.arena.radius)),
                 self.door_state['open'], self.door_state['locked'],
                 self.door_state['adjacent'], self.door['north'], self.opened, self.crossed]
        return np.concatenate([base, np.clip(extra,-1,1)]).astype(np.float32)

    def _info(self, reason):
        return dict(super()._info(reason), door_open=self.door_state['open'],
                    opened_by_agent=self.opened, crossed_door=self.crossed,
                    scenario_index=self.scenarios.index(self.scenario), door=dict(self.door))

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        index = (options or {}).get('scenario_index')
        if index is None:
            index = int(self.np_random.integers(len(self.scenarios)))
        if type(index) is not int or not 0 <= index < len(self.scenarios):
            raise ValueError('invalid scenario_index')
        self.scenario = self.scenarios[index]
        self.door = self.scenario['door']
        self.arena = Arena(**self.scenario['arena'])
        self.done = True
        try:
            # Captured coordinates identify a tile. Reset at its center so
            # collision correction cannot easily push us across its boundary.
            reset_arena = replace(self.arena, x=math.floor(self.arena.x)+0.5,
                                  y=math.floor(self.arena.y)+0.5)
            self.sample = self.backend.reset(reset_arena)
            self.door_state = door_request(self.backend,'door_close',self.door)
            if self.door_state['open']:
                raise TelemetryError('door reset did not close the door')
            self.sample = self.backend.wait_for_update(self.backend.read(),timeout=1.5)
            self.position = np.array([self.sample.x,self.sample.y])
            self.health = self.sample.health
            self.target = np.array(self.scenario['target'],dtype=float)
            if not self.backend.target_is_free(self.arena,*self.target):
                raise ValueError('clear the target tile 1.5 tiles beyond the door and recapture')
            self.steps = self.stalled = 0
            self.velocity = np.zeros(2)
            self.opened = self.crossed = False
            self.best_distance = float(np.linalg.norm(self.target-self.position))
            obs = self._observation()
            self.done = False
            return obs,self._info('running')
        except BaseException:
            self.stop()
            raise

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required')
        try:
            if not self.action_space.contains(action):
                raise ValueError('invalid door action')
            previous = self.position.copy()
            before = float(np.linalg.norm(self.target-previous))
            was_open = self.door_state['open']
            if int(action) == INTERACT:
                self.door_state = door_request(self.backend,'door_open',self.door)
            else:
                self.backend.move(int(action),self.arena.action_duration)
            self.sample = self.backend.wait_for_update(self.backend.read(),timeout=1.5)
            self.stop()
            if int(action) != INTERACT:
                self.door_state = door_request(self.backend,'door_state',self.door)
            self.position = np.array([self.sample.x,self.sample.y])
            self.health = self.sample.health
            self.velocity = self.position-previous
            self.steps += 1
            newly_opened = int(action)==INTERACT and not was_open and self.door_state['open'] and not self.opened
            self.opened = self.opened or newly_opened
            if self.opened and self.door_state['open'] and crosses_opening(previous,self.position,self.door,self.scenario['start_side']):
                self.crossed = True
            distance = float(np.linalg.norm(self.target-self.position))
            improved = distance < self.best_distance-0.05
            self.stalled = 0 if improved or newly_opened else self.stalled+1
            if improved:
                self.best_distance = distance
            outside = np.linalg.norm(self.position-[self.arena.x,self.arena.y])>self.arena.radius or abs(self.sample.z)>0.1
            reason = ('dead' if self.health<=0 else 'out_of_bounds' if outside else
                      'success' if self.opened and self.crossed and distance<=self.arena.goal_radius else
                      'stalled' if self.stalled>=self.stall_steps else
                      'max_steps' if self.steps>=self.arena.max_steps else 'running')
            reward = before-distance-0.02 + (2 if newly_opened else 0)
            if int(action)==INTERACT and not newly_opened:
                reward -= 0.1
            if reason=='success': reward += 10
            if reason in ('dead','out_of_bounds','stalled'): reward -= 3
            self.done = reason!='running'
            return (self._observation(),float(reward),reason in ('success','dead','out_of_bounds'),
                    reason in ('stalled','max_steps'),self._info(reason))
        except BaseException:
            self.done = True
            self.stop()
            raise
