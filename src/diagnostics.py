"""Reproducible full-episode diagnostics; not an RL performance comparator.

The rule uses current observations plus road/lane connectivity information.
The optional safety wrapper uses the normal delayed packet stream.
"""

import argparse
from collections import Counter
import csv
import json
from functools import lru_cache
from pathlib import Path
import subprocess
import time

import numpy as np
import sumolib

from .merging import ANTWERP_DIR, ANTWERP_ENV_VERSION, Merging, maxSteps, parseAction, traci
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


@lru_cache(maxsize=1)
def d1_reachable_lanes():
    """Lane paths that reach D1 without requiring another within-edge change."""
    network = sumolib.net.readNet(str(ANTWERP_DIR / 'Antwerp.net.xml'))
    predecessors = {}
    lanes = [lane for edge in network.getEdges() for lane in edge.getLanes()]
    for lane in lanes:
        for connection in lane.getOutgoing():
            predecessors.setdefault(connection.getToLane().getID(), []).append(lane.getID())
    reachable = {lane.getID() for lane in network.getEdge('D1').getLanes()}
    pending = list(reachable)
    while pending:
        for lane_id in predecessors.get(pending.pop(), []):
            if lane_id not in reachable:
                reachable.add(lane_id)
                pending.append(lane_id)
    return {edge.getID(): [lane.getIndex() for lane in edge.getLanes()
                           if lane.getID() in reachable] for edge in network.getEdges()}


def rule_action(base, lane_strategy='fixed', longitudinal='rule'):
    if not base.ego_on_road:
        return np.zeros(2, np.float32), {'target_lane': -1}
    edge = traci.vehicle.getRoadID(base.ego_id)
    lane = traci.vehicle.getLaneIndex(base.ego_id)
    speed = float(base.observation[0, 2])
    target = FIXED_LANES.get(edge, lane)
    if lane_strategy == 'reachable':
        candidates = d1_reachable_lanes().get(edge, [])
        if candidates:
            target = min(candidates, key=lambda index: abs(index - lane))
            # Among adjacent D1-reachable lanes, leave a stopped queue only
            # when the currently observed alternative offers more forward room.
            if lane in candidates and speed < 1:
                current_gap = lane_gaps(base.observation, target_lane_line(base.ego_id, 0))[0]
                for candidate in candidates:
                    if abs(candidate - lane) == 1:
                        gap = lane_gaps(base.observation, target_lane_line(base.ego_id, candidate - lane))[0]
                        if gap > current_gap + 10:
                            target, current_gap = candidate, gap
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
    if longitudinal == 'sumo':
        # speedMode=31 retains SUMO's longitudinal safety/physical constraints.
        acceleration = 2.6
    return np.asarray([acceleration, 5 * direction], np.float32), {
        'target_lane': target, 'front_gap': gaps[0], 'back_gap': gaps[1],
    }


def physical_state(base):
    state = {'edge': '', 'lane': '', 'lane_index': -1, 'speed': 0.0,
             'distance': float(base.observation[0, 0]), 'leader': '',
             'leader_gap': '', 'leader_speed': '', 'lateral_position': '', 'angle': ''}
    if base.ego_on_road:
        state.update(edge=traci.vehicle.getRoadID(base.ego_id),
                     lane=traci.vehicle.getLaneID(base.ego_id),
                     lane_index=traci.vehicle.getLaneIndex(base.ego_id),
                     speed=traci.vehicle.getSpeed(base.ego_id),
                     distance=traci.vehicle.getDistance(base.ego_id),
                     lateral_position=traci.vehicle.getLateralLanePosition(base.ego_id),
                     angle=traci.vehicle.getAngle(base.ego_id))
        leader = traci.vehicle.getLeader(base.ego_id, 150)
        if leader:
            state.update(leader=leader[0], leader_gap=leader[1],
                         leader_speed=traci.vehicle.getSpeed(leader[0]))
    return state


def run_episode(scenario, seed, safety, output_dir, lane_strategy='fixed', longitudinal='rule'):
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
              'leader', 'leader_gap', 'leader_speed', 'lateral_position', 'angle', 'target_lane', 'front_gap',
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
                action, details = rule_action(base, lane_strategy, longitudinal)
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
            'controller': 'current-observation-rule', 'lane_strategy': lane_strategy,
            'longitudinal': longitudinal, 'env_version': ANTWERP_ENV_VERSION,
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


def run_native_episode(scenario, seed, output_dir):
    """Passive SUMO reference with the same mapped seed and 600 s budget."""
    case_dir = Path(output_dir) / f'{scenario}-{seed}-native'
    case_dir.mkdir(parents=True, exist_ok=False)
    base = Merging(seed=seed, scenario=scenario)
    teleported = False
    lanes, stops = Counter(), Counter()
    changes = 0
    try:
        base.reset(seed=seed)
        traci.vehicle.setLaneChangeMode('ego', 1621)
        with (case_dir / 'trace.csv').open('x', newline='') as trace:
            writer = csv.DictWriter(trace, fieldnames=['step', 'time', *physical_state(base), 'teleport_started', 'teleport_ended'])
            writer.writeheader()
            for step in range(1, maxSteps + 1):
                before = physical_state(base)
                sim_time = traci.simulation.getTime()
                lanes[before['lane']] += 1
                if before['speed'] < 1 and before['lane']:
                    stops[before['lane']] += 1
                traci.simulationStep()
                started = 'ego' in traci.simulation.getStartingTeleportIDList()
                ended = 'ego' in traci.simulation.getEndingTeleportIDList()
                teleported |= started or ended
                base.ego_on_road = 'ego' in traci.vehicle.getIDList()
                writer.writerow({'step': step, 'time': sim_time, **before,
                                 'teleport_started': started, 'teleport_ended': ended})
                if base.ego_on_road:
                    base.observation[0] = [traci.vehicle.getDistance('ego'),
                                           traci.vehicle.getLaneIndex('ego'), traci.vehicle.getSpeed('ego')]
                    after = physical_state(base)
                    changes += int(before['edge'] == after['edge'] and before['lane'] != after['lane']
                                   and not started and not ended)
                if 'ego' in traci.simulation.getCollidingVehiclesIDList():
                    outcome = 'Collision'
                    break
                if 'ego' in traci.simulation.getArrivedIDList():
                    outcome = 'Arrived'
                    break
                if 'ego' in traci.simulation.getEmergencyStoppingVehiclesIDList():
                    outcome = 'Emergency Stopping'
                    break
            else:
                outcome = 'Max Steps'
        result = {'scenario': scenario, 'seed': seed, 'controller': 'native-sumo',
                  'outcome': outcome, 'success': outcome == 'Arrived' and not teleported,
                  'ego_teleported': teleported, 'seconds': round(step * 0.1, 1),
                  'final_lane': before['lane'], 'within_edge_lane_changes': changes,
                  'lane_seconds': {k: round(v * 0.1, 1) for k, v in lanes.items()},
                  'stopped_seconds': {k: round(v * 0.1, 1) for k, v in stops.items()}}
        (case_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return result
    finally:
        base.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', choices=('easy', 'medium', 'hard'), default=['hard'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[579071, 579074])
    parser.add_argument('--safety-modes', nargs='+', choices=('off', 'on'), default=['off', 'on'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--controller', choices=('rule', 'native'), default='rule')
    parser.add_argument('--lane-strategy', choices=('fixed', 'reachable'), default='fixed')
    parser.add_argument('--longitudinal', choices=('rule', 'sumo'), default='rule')
    args = parser.parse_args()
    for scenario in args.scenarios:
        for seed in args.seeds:
            if args.controller == 'native':
                run_native_episode(scenario, seed, args.output_dir)
                continue
            for safety in args.safety_modes:
                run_episode(scenario, seed, safety == 'on', args.output_dir,
                            args.lane_strategy, args.longitudinal)


if __name__ == '__main__':
    main()
