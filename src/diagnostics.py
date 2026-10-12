"""Reproducible full-episode diagnostics; not an RL performance comparator.

The rule uses current observations plus road/lane connectivity information.
The optional safety wrapper uses the normal delayed packet stream.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .merging import ANTWERP_ENV_VERSION, Merging, maxSteps, parseAction, traci
from .safety_controller import target_lane_line
from .wrapper import DelayWrapper, SafetyWrapper


FIXED_LANES = {'E75': 1, 'E76': 2, 'E77': 3, 'E78': 2, 'E82': 3}


def lane_gaps(observation, line):
    """Return nearest observed front/back distances and closing speeds."""
    result = [float('inf'), float('inf'), 0.0, 0.0]
    if line is None:
        return result
    for x, y, relative_speed in observation[1:]:
        if not np.any((x, y, relative_speed)):
            continue
        if abs(y - (line[0] + line[1] * x)) > 1.6:
            continue
        if x > 0 and x < result[0]:
            result[0], result[2] = float(x), max(-float(relative_speed), 0.0)
        elif x < 0 and -x < result[1]:
            result[1], result[3] = float(-x), max(float(relative_speed), 0.0)
    return result


def rule_action(base):
    if not base.ego_on_road:
        return np.zeros(2, np.float32), {'target_lane': -1}
    edge = traci.vehicle.getRoadID(base.ego_id)
    lane = traci.vehicle.getLaneIndex(base.ego_id)
    speed = float(base.observation[0, 2])
    target = FIXED_LANES.get(edge, lane)
    direction = int(np.sign(target - lane))
    gaps = lane_gaps(base.observation, target_lane_line(base.ego_id, direction))
    required_front = max(5 + 1.8 * speed, 5 + gaps[2] * 2)
    required_back = max(5 + 1.8 * max(speed + gaps[3], 0), 5 + gaps[3] * 2)
    if gaps[0] < required_front or gaps[1] < required_back:
        direction = 0
    current_gaps = lane_gaps(base.observation, target_lane_line(base.ego_id, 0))
    acceleration = float(np.clip(0.8 * (13.89 - speed), -4.5, 2.6))
    if current_gaps[0] < max(5 + 1.2 * speed, 5 + current_gaps[2] * 2):
        acceleration = -4.5
    return np.asarray([acceleration, 5 * direction], np.float32), {
        'target_lane': target, 'front_gap': gaps[0], 'back_gap': gaps[1],
    }


def physical_state(base):
    state = {'edge': '', 'lane': '', 'lane_index': -1, 'speed': 0.0,
             'distance': float(base.observation[0, 0]), 'leader': '',
             'leader_gap': '', 'leader_speed': ''}
    if base.ego_on_road:
        state.update(edge=traci.vehicle.getRoadID(base.ego_id),
                     lane=traci.vehicle.getLaneID(base.ego_id),
                     lane_index=traci.vehicle.getLaneIndex(base.ego_id),
                     speed=traci.vehicle.getSpeed(base.ego_id),
                     distance=traci.vehicle.getDistance(base.ego_id))
        leader = traci.vehicle.getLeader(base.ego_id, 150)
        if leader:
            state.update(leader=leader[0], leader_gap=leader[1],
                         leader_speed=traci.vehicle.getSpeed(leader[0]))
    return state


def run_episode(scenario, seed, safety, output_dir):
    case = f'{scenario}-{seed}-safety-{int(safety)}'
    case_dir = Path(output_dir) / case
    case_dir.mkdir(parents=True, exist_ok=False)
    base = Merging(seed=seed, scenario=scenario)
    delay = DelayWrapper(base, max_delay=20, mode='all', delay_mode='uniform', seed=seed)
    env = SafetyWrapper(delay) if safety else delay
    road_steps, stopped_steps = Counter(), Counter()
    requests, executed_requests, lane_changes, interventions = 0, 0, 0, 0
    teleports = []
    episode_return = 0.0
    started = time.monotonic()
    fields = ['step', 'time', 'edge', 'lane', 'lane_index', 'speed', 'distance',
              'leader', 'leader_gap', 'leader_speed', 'target_lane', 'front_gap',
              'back_gap', 'acceleration', 'requested_lane_change', 'applied_lane_change',
              'after_edge', 'after_lane', 'after_speed', 'after_distance',
              'reward', 'delay', 'teleport_started', 'teleport_ended', 'ego_teleported', 'outcome']
    try:
        observation, reset_info = env.reset(seed=seed)
        with (case_dir / 'trace.csv').open('x', newline='') as trace:
            writer = csv.DictWriter(trace, fieldnames=fields)
            writer.writeheader()
            for step in range(1, maxSteps + 1):
                before = physical_state(base)
                sim_time = traci.simulation.getTime()
                road_steps[before['lane']] += 1
                if before['speed'] < 1 and before['lane']:
                    stopped_steps[before['lane']] += 1
                action, details = rule_action(base)
                decision_delay = delay.delay_of_received_observation
                requested = int(parseAction(action)[1])
                observation, reward, terminated, truncated, info = env.step(action)
                if observation.shape != (134,) or not np.isfinite(observation).all() or not np.isfinite(reward):
                    raise RuntimeError(f'Invalid transition at step {step}')
                if bool(info['is_success']) != (info['message'] == 'Arrived' and not info['ego_teleported']):
                    raise RuntimeError('Inconsistent success accounting')
                applied = int(base.parsed_action[1])
                requests += int(requested != 0)
                executed_requests += int(applied != 0)
                interventions += int(not np.allclose(action, base.action))
                after = physical_state(base) if not (terminated or truncated) else {
                    'edge': base.lane_name, 'lane': base.lane_id,
                    'speed': float(base.observation[0, 2]),
                    'distance': float(base.observation[0, 0]),
                }
                lane_changes += int(before['edge'] == after['edge'] and bool(before['lane'])
                                    and before['lane'] != after['lane'] and not info['teleport_started']
                                    and not info['teleport_ended'])
                if info['teleport_started'] or info['teleport_ended']:
                    teleports.append({'step': step, 'time': round(sim_time + 0.1, 1),
                                      'from_lane': before['lane'], 'to_lane': after['lane'],
                                      'started': info['teleport_started'], 'ended': info['teleport_ended']})
                episode_return += reward
                writer.writerow({'step': step, 'time': round(sim_time, 1), **before, **details,
                                 'acceleration': float(action[0]), 'requested_lane_change': requested,
                                 'applied_lane_change': applied, 'after_edge': after['edge'],
                                 'after_lane': after['lane'], 'after_speed': after['speed'],
                                 'after_distance': after['distance'], 'reward': reward,
                                 'delay': decision_delay,
                                 'teleport_started': info['teleport_started'],
                                 'teleport_ended': info['teleport_ended'],
                                 'ego_teleported': info['ego_teleported'], 'outcome': info['message']})
                if terminated or truncated:
                    break
            else:
                raise RuntimeError('Environment did not end at its configured limit')
        result = {
            'scenario': scenario, 'seed': seed, 'safety': safety,
            'controller': 'fixed-current-observation-rule', 'env_version': ANTWERP_ENV_VERSION,
            'delay_mode': 'uniform', 'max_steps': maxSteps, 'outcome': info['message'],
            'success': bool(info['is_success']), 'ego_teleported': info['ego_teleported'],
            'merged': info['merged'], 'steps': step, 'seconds': round(step * 0.1, 1),
            'return': episode_return, 'final_lane': after['lane'], 'final_distance': after['distance'],
            'lane_requests': requests, 'applied_lane_requests': executed_requests,
            'within_edge_lane_changes': lane_changes, 'safety_interventions': interventions,
            'lane_seconds': {key: round(value * 0.1, 1) for key, value in road_steps.items()},
            'stopped_seconds': {key: round(value * 0.1, 1) for key, value in stopped_steps.items()},
            'teleports': teleports, 'reset_info': reset_info,
            'wall_seconds': time.monotonic() - started,
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
        }
        (case_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: result[key] for key in ('scenario', 'seed', 'safety', 'outcome',
              'success', 'ego_teleported', 'seconds', 'final_lane', 'within_edge_lane_changes')}), flush=True)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', choices=('easy', 'medium', 'hard'), default=['hard'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[579071, 579074])
    parser.add_argument('--safety-modes', nargs='+', choices=('off', 'on'), default=['off', 'on'])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    for scenario in args.scenarios:
        for seed in args.seeds:
            for safety in args.safety_modes:
                run_episode(scenario, seed, safety == 'on', args.output_dir)


if __name__ == '__main__':
    main()
