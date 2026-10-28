"""Frozen E09 interface checks and paired origin evaluation (no training)."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from evaluate_sac_delay import make_env, validate_checkpoint_setup
from src import exit_e76_d3_config as cfg
from src.merging import parseAction


LANES = {'E76': 5, 'E77': 6, 'O1': 3}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def cases(count, origin):
    if origin not in LANES or count not in (10, 20):
        raise ValueError('Invalid E09 case definition')
    # The same seed identifies traffic within every controller/model pairing.
    return [{'seed': 909000 + i, 'lane': i % LANES[origin]} for i in range(count)]


def evaluate_case(model, origin, pending_mode, case, route_dir):
    setup = validate_checkpoint_setup(model, 'full', False, cfg.TASK_ID, pending_mode)
    env = make_env(case['seed'], 'easy', 'uniform', False, 'full', cfg.TASK_ID,
                   setup['potential_gamma'], False, setup['exit_reward_profile'], pending_mode)
    raw = env.unwrapped
    raw.e09_evaluation_origin, raw.e09_evaluation_lane = origin, case['lane']
    raw.e09_evaluation_route_dir = route_dir
    counter = Counter()
    direction_history = []
    command_log = []
    original = None
    try:
        obs, reset = env.reset(seed=case['seed'])
        if reset.get('ego_start_lane') != case['lane']:
            raise ValueError('SUMO did not use the requested starting lane')
        original = raw.traci.vehicle.changeLane

        def change_lane(ego, target, duration):
            result = original(ego, target, duration)
            if ego == raw.ego_id:
                command_log.append({'target': int(target), 'duration': float(duration)})
            return result

        raw.traci.vehicle.changeLane = change_lane
        total = 0.0
        motion = False
        origin_lane = None
        origin_edge = None
        for step in range(1, 6001):
            if np.asarray(obs).shape != (134,) or not np.isfinite(obs).all():
                raise ValueError('Non-finite or misaligned E09 observation')
            action, _ = model.predict(obs, deterministic=True)
            if not np.isfinite(action).all():
                raise ValueError('Non-finite actor action')
            active = raw.ego_on_road and not raw.ego_teleporting
            road = raw.traci.vehicle.getRoadID(raw.ego_id) if active else None
            lane = int(raw.traci.vehicle.getLaneIndex(raw.ego_id)) if active else None
            direction = int(parseAction(action)[1])
            pending = raw.exit_maneuver is not None
            lane_count = (raw.traci.edge.getLaneNumber(road)
                          if active and road in ('E76', 'E77') else None)
            before_commands = len(command_log)
            obs, reward, terminated, truncated, info = env.step(action)
            if not math.isfinite(float(reward)) or info.get('exit_reward_profile') != cfg.URGENCY_REWARD_PROFILE:
                raise ValueError('Invalid reward or reward-profile drift')
            total += float(reward)
            if abs(float(info.get('potential_reward', 0))) > 1e-9:
                raise ValueError('Unexpected potential-based reward')
            sent = len(command_log) - before_commands
            if direction and road in ('E76', 'E77'):
                counter['requests'] += 1
                counter['toward_requests' if direction < 0 else 'away_requests'] += 1
                if not sent:
                    counter['blocked_requests'] += 1
                    if pending and pending_mode == 'on':
                        counter['pending_blocked'] += 1
                    elif not 0 <= lane + direction < lane_count:
                        counter['illegal_lane_blocked'] += 1
                    else:
                        counter['other_blocked'] += 1
            counter['commands'] += sent
            if not (terminated or truncated) and raw.ego_on_road and not raw.ego_teleporting:
                now_road = raw.traci.vehicle.getRoadID(raw.ego_id)
                now_lane = int(raw.traci.vehicle.getLaneIndex(raw.ego_id))
                lat_speed = abs(float(raw.traci.vehicle.getLateralSpeed(raw.ego_id)))
                if not motion and lat_speed > cfg.LATERAL_SETTLED_SPEED_MPS:
                    motion, origin_edge, origin_lane = True, road, lane
                    counter['motion_started'] += 1
                if motion and lat_speed <= cfg.LATERAL_SETTLED_SPEED_MPS:
                    if now_road == origin_edge and origin_lane is not None and now_lane != origin_lane:
                        move = 1 if now_lane > origin_lane else -1
                        counter['physical_completed_same_edge'] += 1
                        if direction_history and move != direction_history[-1]:
                            counter['physical_direction_reversals'] += 1
                        direction_history.append(move)
                    elif now_road != origin_edge:
                        counter['motion_cross_edge_unclassified'] += 1
                    else:
                        counter['motion_stopped_without_lane_change'] += 1
                    motion = False
            if terminated or truncated:
                return {'seed': case['seed'], 'origin': origin, 'start_lane': case['lane'],
                        'steps': step, 'return': total, 'outcome': info.get('message'),
                        'success': bool(info.get('is_success')),
                        'teleport': bool(info.get('ego_teleported')),
                        'counts': dict(counter), 'commands': command_log}
        raise RuntimeError('Episode exceeds 6000 steps')
    finally:
        if original is not None:
            raw.traci.vehicle.changeLane = original
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pending-mode', choices=('on', 'off'), required=True)
    parser.add_argument('--origin', choices=tuple(LANES), required=True)
    parser.add_argument('--count', type=int, choices=(10, 20), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--route-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite existing E09 report')
    torch.set_num_threads(1)
    model = SAC.load(args.checkpoint, device=args.device)
    if (getattr(model, 'num_timesteps', None) not in (100000, 250000)
            or getattr(model, 'seed', None) != 43):
        raise ValueError('E09 requires seed43 100k parent or 250k continuation')
    records = []
    for case in cases(args.count, args.origin):
        row = evaluate_case(model, args.origin, args.pending_mode, case, args.route_dir)
        records.append(row)
        print(json.dumps({'complete': len(records), 'origin': args.origin,
                          'mode': args.pending_mode, 'seed': case['seed'],
                          'outcome': row['outcome']}), flush=True)
    counts = Counter()
    for row in records:
        counts.update(row['counts'])
    result = {'protocol': 'e09-frozen-eval-v1', 'checkpoint_sha256': digest(args.checkpoint),
              'trained_pending_mode': getattr(model, 'e09_pending_mode', None),
              'pending_mode': args.pending_mode, 'origin': args.origin,
              'seed43_model_steps': model.num_timesteps, 'episodes': records,
              'summary': {'n': len(records), 'success': sum(r['success'] for r in records),
                          'outcomes': dict(Counter(r['outcome'] for r in records)),
                          'teleports': sum(r['teleport'] for r in records),
                          'counts': dict(counts)},
              'limitations': ['Physical completed changes exclude junction-crossing maneuvers; '
                              'reversals count consecutive observed same-edge completed changes.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()
