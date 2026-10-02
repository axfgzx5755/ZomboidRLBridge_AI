"""Run saved navigation and door policies in one live, non-resetting route."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import time

import numpy as np

from actions import ActionError, HeldMovement, is_project_zomboid_foreground
from door_env import SCHEMA as DOOR_SCHEMA, crosses_opening, load_scenario
from local_play import FreePlayBridge
from navigation import Grid, SCHEMA as NAV_SCHEMA
from rl import countdown, write_json
from training_env import Arena
from telemetry import TelemetryError


def model_metadata(path, expected_schema):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'model not found: {path}')
    meta_path = path.with_suffix('.json')
    if not meta_path.is_file():
        raise ValueError(f'model metadata not found: {meta_path}')
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    if metadata.get('schema') != expected_schema:
        raise ValueError(f'{path.name} must use schema {expected_schema}')
    return metadata


def load_policies(args):
    nav_meta = model_metadata(args.navigation_model, NAV_SCHEMA)
    door_meta = model_metadata(args.door_model, DOOR_SCHEMA)
    scenario = load_scenario(args.door_scenario)
    trained_scenarios = door_meta.get('scenarios', [door_meta.get('scenario')])
    if scenario not in trained_scenarios:
        raise ValueError('door scenario must exactly match one of the model training scenarios')
    arena = Arena.load(args.arena)
    if int(arena.radius) != int(nav_meta['arena']['radius']):
        raise ValueError('captured arena radius must match navigation model observation scale')
    if not math.isclose(arena.action_duration, nav_meta['arena']['action_duration'], abs_tol=1e-9):
        raise ValueError('captured arena action duration must match navigation model')
    approach = np.asarray([scenario['arena']['x']+0.5, scenario['arena']['y']+0.5])
    if np.linalg.norm(approach-np.array([arena.x, arena.y])) >= arena.radius:
        raise ValueError('door approach tile must be within the captured navigation arena')
    goal = np.asarray(args.goal, dtype=np.float32)
    if np.linalg.norm(goal-np.array([arena.x, arena.y])) >= arena.radius:
        raise ValueError('final goal must be inside the captured arena')
    axis = 1 if scenario['door']['north'] else 0
    target_axis = scenario['target'][axis]
    if (goal[axis]-target_axis)*(-scenario['start_side']) <= 0.75:
        raise ValueError('choose a final goal at least 0.75 tiles beyond the door target')

    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    nav_model = PPO.load(args.navigation_model, device='cpu')
    door_model = PPO.load(args.door_model, device='cpu')
    if tuple(nav_model.observation_space.shape) != (134,) or nav_model.action_space.n != 9:
        raise ValueError('navigation model must accept 134 values and use 9 actions')
    if tuple(door_model.observation_space.shape) != (142,) or door_model.action_space.n != 10:
        raise ValueError('door model must accept 142 values and use 10 actions')
    return (nav_model, door_model, nav_meta, door_meta,
            scenario, arena, goal)


def position(snapshot):
    p = snapshot['player']
    return np.asarray([p['x'], p['y']], dtype=np.float32)


def nav_observation(snapshot, target, radius, velocity, stalled, stall_limit):
    p = snapshot['player']
    grid = Grid.parse(snapshot['map'], radius=4)
    pos = position(snapshot)
    if (grid.x, grid.y) != (math.floor(pos[0]), math.floor(pos[1])):
        raise TelemetryError('navigation map origin does not match player position')
    delta = (target-pos)/(2*radius)
    base = [*delta, np.linalg.norm(delta), p['health'], min(stalled/stall_limit, 1),
            *np.clip(velocity, -1, 1), *(pos-np.floor(pos))]
    return np.concatenate([np.clip(base, -1, 1), grid.local(*pos)]).astype(np.float32)


def find_door(snapshot, scenario):
    door = scenario['door']
    matches = [d for d in snapshot['doors']
               if (d['x'], d['y'], d['z'], d['north']) ==
                  (door['x'], door['y'], door['z'], door['north'])]
    if len(matches) != 1:
        raise TelemetryError(f'expected one visible target door; found {len(matches)}')
    found = matches[0]
    if found['locked'] or found['barricaded'] or not found['supported']:
        raise TelemetryError('target door is locked, barricaded, or unsupported')
    return found


def on_trained_door_side(snapshot, scenario):
    """Door interaction is valid only from the exact adjacent tile used in training."""
    door = scenario['door']
    p = snapshot['player']
    first = (door['x'], door['y'])
    second = (door['x']-(not door['north']), door['y']-door['north'])
    expected = first if scenario['start_side'] == 1 else second
    return abs(p['z']) < 0.1 and (math.floor(p['x']), math.floor(p['y'])) == expected


def trained_door_tile(scenario):
    door = scenario['door']
    first = (door['x'], door['y'])
    second = (door['x']-(not door['north']), door['y']-door['north'])
    return first if scenario['start_side'] == 1 else second


def grid_path_action(snapshot, target):
    """Return the first cardinal movement action toward a nearby free tile."""
    p = snapshot['player']
    grid = Grid.parse(snapshot['map'], radius=4)
    start = (math.floor(p['x']), math.floor(p['y']))
    if start == target:
        return 0
    queue = deque([start])
    previous = {start: None}
    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
    while queue and target not in previous:
        x, y = queue.popleft()
        for dx, dy in directions:
            neighbor = (x+dx, y+dy)
            if neighbor not in previous and grid.edge(x, y, dx, dy):
                previous[neighbor] = (x, y)
                queue.append(neighbor)
    if target not in previous:
        return None
    step = target
    while previous[step] != start:
        step = previous[step]
    delta = (step[0]-start[0], step[1]-start[1])
    return {(-1, 0): 5, (0, -1): 6, (0, 1): 7, (1, 0): 8}[delta]


def door_observation(snapshot, scenario, velocity, stalled, opened, crossed):
    p = snapshot['player']
    pos = position(snapshot)
    grid = Grid.parse(snapshot['map'], radius=4)
    door_state = find_door(snapshot, scenario)
    arena = scenario['arena']
    target = np.asarray(scenario['target'], dtype=np.float32)
    radius = arena['radius']
    delta = (target-pos)/(2*radius)
    base = [*delta, np.linalg.norm(delta), p['health'], min(stalled/40, 1),
            *np.clip(velocity, -1, 1), *(pos-np.floor(pos))]
    door = scenario['door']
    center = np.asarray([door['x']+(0.5 if door['north'] else 0),
                         door['y']+(0 if door['north'] else 0.5)])
    sides = [(door['x'], door['y']),
             (door['x']-(not door['north']), door['y']-door['north'])]
    tile = (math.floor(pos[0]), math.floor(pos[1]))
    adjacent = abs(p['z']) < 0.1 and tile in sides
    extra = [*((center-pos)/(2*radius)), door_state['open'], door_state['locked'],
             adjacent, door['north'], opened, crossed]
    obs = np.concatenate([np.clip(base, -1, 1), grid.local(*pos), np.clip(extra, -1, 1)])
    if obs.shape != (142,):
        raise TelemetryError(f'door observation shape mismatch: {obs.shape}')
    return obs.astype(np.float32), door_state


def safe_snapshot(snapshot, minimum_health):
    p = snapshot['player']
    if p['health'] < minimum_health:
        raise RuntimeError(f'health fell below safety limit ({p["health"]:.2f})')
    near = [z for z in snapshot.get('zombies', [])
            if math.hypot(z['x']-p['x'], z['y']-p['y']) < 4.5]
    if near:
        raise RuntimeError('zombie entered the 4.5-tile safety radius; stopped before combat')


def act(model, obs, movement, bridge, duration):
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    if action == 0:
        movement.stop()
        time.sleep(duration)
    else:
        movement.move(action, duration)
        movement.stop()
    return action, bridge.request()


def run(args):
    nav_model, door_model, nav_meta, door_meta, scenario, arena, goal = load_policies(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'config.json', dict(
        navigation_model=str(args.navigation_model), door_model=str(args.door_model),
        navigation_schema=nav_meta['schema'], door_schema=door_meta['schema'],
        arena=arena.__dict__, scenario=scenario, goal=goal.tolist(),
        max_steps_per_phase=args.max_steps, minimum_health=args.min_health))

    bridge = movement = None
    stage = 'starting'
    steps = 0
    error = None
    records = args.output/'trajectory.jsonl'
    try:
        bridge = FreePlayBridge()
        movement = HeldMovement()
        countdown(args.countdown)
        snapshot = bridge.request()
        safe_snapshot(snapshot, args.min_health)
        current = position(snapshot)
        stage = 'approach_door'
        approach = np.asarray([scenario['arena']['x']+0.5,
                               scenario['arena']['y']+0.5], dtype=np.float32)
        nav_radius = float(nav_meta['arena']['radius'])
        nav_stalled = 0
        nav_best = float(np.linalg.norm(approach-current))
        nav_velocity = np.zeros(2, dtype=np.float32)
        stage_steps = 0
        opened = crossed = False
        door_stalled = 0
        door_best = math.inf
        door_velocity = np.zeros(2, dtype=np.float32)
        door_steps = 0

        while stage != 'complete':
            safe_snapshot(snapshot, args.min_health)
            stage_before_action = stage
            p = snapshot['player']
            current = position(snapshot)
            if math.hypot(current[0]-arena.x, current[1]-arena.y) > arena.radius:
                raise RuntimeError('player left the captured execution arena')

            if stage == 'approach_door':
                if on_trained_door_side(snapshot, scenario):
                    state = find_door(snapshot, scenario)
                    if state['open']:
                        raise RuntimeError('close the target door manually, then start the route again')
                    stage = 'cross_door'
                    door_best = float(np.linalg.norm(np.asarray(scenario['target'])-current))
                    continue
                if np.linalg.norm(approach-current) <= 2.5:
                    action = grid_path_action(snapshot, trained_door_tile(scenario))
                    if action is None:
                        raise RuntimeError('no walkable local route to the trained door-side tile; reposition and recapture')
                    movement.move(action, arena.action_duration)
                    movement.stop()
                    next_snapshot = bridge.request()
                else:
                    obs = nav_observation(snapshot, approach, nav_radius,
                                          nav_velocity, nav_stalled, 30)
                    action, next_snapshot = act(nav_model, obs, movement, bridge, arena.action_duration)
                new_pos = position(next_snapshot)
                new_distance = float(np.linalg.norm(approach-new_pos))
                improved = new_distance < nav_best-0.05
                nav_stalled = 0 if improved else nav_stalled+1
                if improved: nav_best = new_distance
                nav_velocity = new_pos-current
                stage_steps += 1
                snapshot = next_snapshot
                if on_trained_door_side(snapshot, scenario):
                    state = find_door(snapshot, scenario)
                    if state['open']:
                        raise RuntimeError('close the target door manually, then start the route again')
                    stage = 'cross_door'
                    door_best = float(np.linalg.norm(np.asarray(scenario['target'])-new_pos))
                if stage == 'approach_door' and (nav_stalled >= 30 or stage_steps >= args.max_steps):
                    raise RuntimeError('navigation policy stalled or exceeded its step limit before the door')

            elif stage == 'cross_door':
                obs, state = door_observation(snapshot, scenario, door_velocity,
                                              door_stalled, opened, crossed)
                action, next_snapshot = door_model.predict(obs, deterministic=True)
                action = int(action)
                was_open = state['open']
                if action == 9:
                    next_snapshot = bridge.request('open', scenario['door'])
                    outcome = next_snapshot['outcome']
                    if outcome == 'opened':
                        opened = True
                    elif outcome == 'already_open' and not opened:
                        raise RuntimeError('door was opened outside the policy; refusing false success')
                    elif outcome != 'already_open':
                        raise RuntimeError(f'door interaction failed: {outcome}')
                elif action == 0:
                    movement.stop()
                    time.sleep(scenario['arena']['action_duration'])
                    next_snapshot = bridge.request()
                else:
                    movement.move(action, scenario['arena']['action_duration'])
                    movement.stop()
                    next_snapshot = bridge.request()

                new_pos = position(next_snapshot)
                new_state = find_door(next_snapshot, scenario)
                if opened and new_state['open'] and crosses_opening(
                        current, new_pos, scenario['door'], scenario['start_side']):
                    crossed = True
                new_distance = float(np.linalg.norm(np.asarray(scenario['target'])-new_pos))
                improved = new_distance < door_best-0.05
                newly_opened = action == 9 and not was_open and new_state['open'] and opened
                door_stalled = 0 if improved or newly_opened else door_stalled+1
                if improved: door_best = new_distance
                door_velocity = new_pos-current
                door_steps += 1
                snapshot = next_snapshot
                if opened and crossed and new_distance <= scenario['arena']['goal_radius']:
                    stage = 'navigate_goal'
                    stage_steps = nav_stalled = 0
                    nav_best = float(np.linalg.norm(goal-new_pos))
                    nav_velocity = np.zeros(2, dtype=np.float32)
                elif door_stalled >= 40 or door_steps >= args.max_steps:
                    raise RuntimeError('door policy stalled or exceeded its step limit')

            elif stage == 'navigate_goal':
                distance = float(np.linalg.norm(goal-current))
                if distance <= arena.goal_radius:
                    stage = 'complete'
                    continue
                obs = nav_observation(snapshot, goal, nav_radius, nav_velocity, nav_stalled, 30)
                action, next_snapshot = act(nav_model, obs, movement, bridge, arena.action_duration)
                new_pos = position(next_snapshot)
                new_distance = float(np.linalg.norm(goal-new_pos))
                improved = new_distance < nav_best-0.05
                nav_stalled = 0 if improved else nav_stalled+1
                if improved: nav_best = new_distance
                nav_velocity = new_pos-current
                stage_steps += 1
                snapshot = next_snapshot
                if nav_stalled >= 30 or stage_steps >= args.max_steps:
                    raise RuntimeError('navigation policy stalled or exceeded its step limit before the final goal')

            current = position(snapshot)
            steps += 1
            record = dict(step=steps, stage=stage_before_action, next_stage=stage,
                          position=current.tolist(), action=action,
                          health=snapshot['player']['health'], opened_by_policy=opened,
                          crossed_door=crossed)
            with records.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            if steps % 10 == 0 or stage_before_action == 'cross_door':
                print(f"step={steps} stage={stage_before_action} next={stage} action={action} position=({current[0]:.2f},{current[1]:.2f})", flush=True)

        write_json(args.output/'summary.json', dict(status='success', steps=steps,
                   opened_by_policy=opened, crossed_door=crossed,
                   final_position=current.tolist(), goal=goal.tolist()))
        return 0
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(f'Stopped: {error}', flush=True)
        write_json(args.output/'summary.json', dict(status='error', stage=stage,
                   steps=steps, error=error))
        return 2
    finally:
        if movement:
            try: movement.close()
            except Exception: pass
        if bridge:
            try: bridge.close()
            except Exception: pass
        write_json(args.output/'status.json', dict(status='complete' if not error and stage=='complete' else 'error',
                   stage=stage, steps=steps, error=error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--navigation-model', type=Path, required=True)
    parser.add_argument('--door-model', type=Path, required=True)
    parser.add_argument('--door-scenario', type=Path, required=True)
    parser.add_argument('--arena', type=Path, required=True)
    parser.add_argument('--goal', type=float, nargs=2, required=True, metavar=('X','Y'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--min-health', type=float, default=0.8)
    parser.add_argument('--countdown', type=int, default=5)
    args = parser.parse_args()
    if args.max_steps < 1 or args.countdown < 0 or not 0 < args.min_health <= 1:
        parser.error('invalid step, countdown or health limit')
    try:
        return run(args)
    except (OSError, ValueError, TelemetryError, ActionError) as exc:
        print(f'Stopped before control: {type(exc).__name__}: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
