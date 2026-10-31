"""Replay frozen B3 policies with read-only route diagnostics."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from stable_baselines3 import SAC

from evaluate_sac_delay import make_env, validate_checkpoint_setup
from .merging import ANTWERP_DIR, LANE_CHANGE_EDGES
from .route_urgency import load_route_network, observe_route_urgency


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(base, network):
    result = observe_route_urgency(base, network, LANE_CHANGE_EDGES)
    result['n'] = result.get('minimum_lane_changes')
    if not base.ego_on_road:
        return result
    vehicle = base.traci.vehicle
    result.update(lane_position_m=vehicle.getLanePosition(base.ego_id),
                  lateral_position_m=vehicle.getLateralLanePosition(base.ego_id),
                  lateral_speed_mps=vehicle.getLateralSpeed(base.ego_id),
                  vehicle_length_m=vehicle.getLength(base.ego_id))
    raw, usable = [], []
    for i, part in enumerate(result.get('plan', [])):
        if part['edge'] not in LANE_CHANGE_EDGES:
            continue
        lanes = network.lanes[part['edge']]
        indices = range(min(part['entry_lane'], part['exit_lane']),
                        max(part['entry_lane'], part['exit_lane']) + 1)
        position = (result['lane_position_m'] if i == 0 and
                    result.get('road') == part['edge'] else 0)
        raw.append(max(0.0, min(lanes[k].length for k in indices) - position))
        usable.append(part['usable_distance_m'])
    # Distances follow the estimator's selected lane plan, not a motion constraint.
    result['remaining_changeable_distance_m'] = sum(raw) if result.get('plan') else None
    result['estimated_usable_distance_m'] = sum(usable) if result.get('plan') else None
    if result.get('status') == 'at_destination':
        result['remaining_changeable_distance_m'] = 0.0
        result['estimated_usable_distance_m'] = 0.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=Path(
        '/home/cyl/project/DAROM-TPR-a1-components/source/models_experiments/a1_component_ablation'))
    parser.add_argument('--output', type=Path, default=Path('tmp/ai/b3-route-diagnostics'))
    parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)  # Same as the original evaluation entry.
    network = load_route_network(ANTWERP_DIR / 'Antwerp.net.xml')
    manifest = json.loads((args.models_dir / 'experiment_manifest.json').read_text())
    models = {}
    for seed in (42, 43):
        path = args.models_dir / f'B3_seed{seed}' / f'B3_seed{seed}.zip'
        expected = json.loads((path.parent / 'evaluation.json').read_text())
        training = next(row for row in manifest['runs']
                        if row['profile'] == 'B3' and row['seed'] == seed)
        if digest(path) != expected['checkpoint_sha256'] or digest(path) != training['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
        policy = SAC.load(path, device=args.device)
        validate_checkpoint_setup(policy, 'full', False, 'exit_e76_d3', 'B3')
        episodes = expected['runs'][0]['episode_results']
        if policy.seed != seed or [row['seed'] for row in episodes] != list(range(200001, 200051)):
            raise ValueError('Training or evaluation seed mismatch')
        models[seed] = (policy, expected, path)
    args.output.mkdir(parents=True)
    config = {'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'training_steps': 0, 'evaluation_episodes': 100, 'device': args.device,
              'lane_change_seconds': 4.0, 'speed_floor_fraction': 0.7,
              'end_buffer_m': 5.0, 'min_time_s': 0.1,
              'network_sha256': digest(ANTWERP_DIR / 'Antwerp.net.xml'),
              'checkpoint_sha256': {str(k): digest(v[2]) for k, v in models.items()}}
    (args.output / 'manifest.json').write_text(json.dumps(config, indent=2))
    summaries = []
    for seed, (policy, expected, _) in models.items():
        env = make_env(seed, 'easy', 'uniform', False, 'full',
                       task='exit_e76_d3', reward_profile='B3')
        try:
            with gzip.open(args.output / f'B3_seed{seed}_steps.jsonl.gz', 'xt') as trace:
                for reference in expected['runs'][0]['episode_results']:
                    obs, reset = env.reset(seed=reference['seed'])
                    for field in ('scene_files', 'sumo_version', 'sumo_seed'):
                        if reset[field] != reference['reset_info'][field]:
                            raise ValueError(f'Reset provenance mismatch: {field}')
                    total, components, steps = 0.0, {}, 0
                    while True:
                        before = snapshot(env.unwrapped, network)
                        action, _ = policy.predict(obs, deterministic=True)
                        obs, reward, term, trunc, info = env.step(action)
                        steps += 1
                        total += reward
                        for key, value in info['reward_components'].items():
                            components[key] = components.get(key, 0.0) + value
                        record = {'training_seed': seed, 'evaluation_seed': reference['seed'],
                                  'step': steps, 'before_action': before,
                                  'action': np.asarray(action).tolist(),
                                  'after': {key: info.get(key) for key in
                                            ('road', 'lane', 'lane_index', 'lane_change_completed',
                                             'lane_change_delta', 'command_direction', 'ego_distance',
                                             'ego_speed', 'ego_acceleration', 'teleport_started',
                                             'teleport_ended', 'ego_teleported', 'stage_reached')},
                                  'reward': reward, 'reward_components': info['reward_components'],
                                  'outcome': info['message'], 'terminated': term, 'truncated': trunc}
                        trace.write(json.dumps(record, allow_nan=False) + '\n')
                        if term or trunc:
                            matched = (steps == reference['steps'] and
                                       info['message'] == reference['outcome'] and
                                       info['is_success'] == reference['success'] and
                                       np.isclose(total, reference['return'], rtol=0, atol=1e-4) and
                                       all(np.isclose(value, reference['reward_components'][key],
                                                      rtol=0, atol=1e-4)
                                           for key, value in components.items()))
                            row = {'training_seed': seed, 'evaluation_seed': reference['seed'],
                                   'steps': steps, 'outcome': info['message'], 'return': total,
                                   'matches_original': bool(matched)}
                            summaries.append(row)
                            with (args.output / 'episodes.jsonl').open('a') as output:
                                output.write(json.dumps(row) + '\n')
                            print(row, flush=True)
                            if not matched:
                                raise ValueError('Replay differs from original evaluation; stopped')
                            break
        finally:
            env.close()
    (args.output / 'complete.json').write_text(json.dumps({
        'episodes': len(summaries), 'all_match': all(row['matches_original'] for row in summaries)}))


if __name__ == '__main__':
    main()
