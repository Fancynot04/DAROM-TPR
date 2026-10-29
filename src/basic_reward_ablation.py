"""Run the fixed E76-D3 A0-A3 matrix serially and summarize final checkpoints."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from stable_baselines3 import SAC

from . import exit_e76_d3_config as cfg


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'models_experiments' / 'basic_reward_ablation'
PROFILES = tuple(cfg.ABLATION_PROFILES)
SEEDS = (42, 43)
EVAL_SEEDS = list(range(200001, 200051))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def run_dir(profile, seed):
    return OUTPUT / f'{profile}_seed{seed}'


def model_path(profile, seed):
    return run_dir(profile, seed) / f'{profile}_seed{seed}.zip'


def verified_model(profile, seed):
    path = model_path(profile, seed)
    model = SAC.load(path, device='cpu')
    metadata = getattr(model, 'exit_training_config', None)
    if (model.num_timesteps != 200_000 or model.seed != seed or
            model.exit_reward_profile != profile or
            not isinstance(metadata, dict) or metadata.get('reward_profile') != profile or
            metadata.get('total_timesteps') != 200_000 or metadata.get('seed') != seed):
        raise ValueError(f'Final checkpoint metadata invalid: {path}')
    milestones = {}
    for step in (50_000, 100_000, 200_000):
        stem = run_dir(profile, seed) / f'{profile}_seed{seed}' / f'{profile}_seed{seed}_{step}_steps'
        milestone = Path(str(stem) + '.zip')
        replay = Path(str(stem) + '_replay_buffer.pkl')
        if not milestone.is_file() or not replay.is_file():
            raise FileNotFoundError(f'Missing milestone model or Replay Buffer: {stem}')
        milestones[str(step)] = {'checkpoint': str(milestone),
                                 'checkpoint_sha256': sha256(milestone),
                                 'replay_buffer': str(replay),
                                 'replay_sha256': sha256(replay)}
    return {'profile': profile, 'seed': seed, 'num_timesteps': model.num_timesteps,
            'checkpoint': str(path), 'checkpoint_sha256': sha256(path),
            'milestones': milestones, 'training_config': metadata,
            'git_sha': metadata['git_sha']}


def write_manifest(records):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'experiment_manifest.json'
    with path.open('w') as stream:
        json.dump({'source_branch_sha': '8c2eb705ef12f97edd3d46c58d54518b42eac4ce',
                   'experiment_sha': source_sha(), 'runs': records}, stream, indent=2)
        stream.write('\n')
    rows = [{'profile': item['profile'], 'seed': item['seed'],
             'num_timesteps': item['num_timesteps'],
             'checkpoint': item['checkpoint'],
             'checkpoint_sha256': item['checkpoint_sha256'],
             'git_sha': item['git_sha']} for item in records]
    path = OUTPUT / 'train_summary.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_all(device):
    records = []
    for profile in PROFILES:
        for seed in SEEDS:
            path = model_path(profile, seed)
            if path.exists():
                records.append(verified_model(profile, seed))
                write_manifest(records)
                continue
            directory = run_dir(profile, seed)
            if directory.exists():
                raise FileExistsError(f'Incomplete run exists; inspect its checkpoint before resuming: {directory}')
            directory.mkdir(parents=True)
            command = [sys.executable, '-B', 'train_sac_delay.py', '--exit_phase',
                       cfg.ABLATION_PHASE, '--reward_profile', profile,
                       '--seed', str(seed), '--wandb_mode', 'disabled', '--device', device]
            with (directory / 'train.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True)
            records.append(verified_model(profile, seed))
            write_manifest(records)
    return records


def evaluate_all(device):
    for profile in PROFILES:
        for seed in SEEDS:
            verified_model(profile, seed)
            directory = run_dir(profile, seed)
            report = directory / 'evaluation.json'
            if report.exists():
                with report.open() as stream:
                    data = json.load(stream)
                if (data.get('checkpoint_sha256') != sha256(model_path(profile, seed)) or
                        data.get('evaluation_seeds') != [200001] or
                        data.get('total_episodes') != 50):
                    raise ValueError(f'Existing evaluation does not match protocol: {report}')
                continue
            command = [sys.executable, '-B', 'evaluate_sac_delay.py',
                       str(model_path(profile, seed)), '--task', cfg.TASK_ID,
                       '--reward-profile', profile, '--scenario', 'easy',
                       '--delay-mode', 'uniform', '--no-safety', '--episodes', '50',
                       '--seed', '200001', '--device', device, '--output', str(report)]
            with (directory / 'evaluation.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    rows, lane_rows, reward_rows = [], [], []
    for profile in PROFILES:
        for seed in SEEDS:
            with (run_dir(profile, seed) / 'evaluation.json').open() as stream:
                report = json.load(stream)
            episodes = report['runs'][0]['episode_results']
            if len(episodes) != 50 or [item['seed'] for item in episodes] != EVAL_SEEDS:
                raise ValueError(f'Incorrect evaluation episodes for {profile} seed {seed}')
            rates = {key: 100 * sum(item['outcome'] == outcome for item in episodes) / 50
                     for key, outcome in (('success_rate', 'Arrived'),
                                          ('missed_exit_rate', 'Missed Exit'),
                                          ('collision_rate', 'Collision'),
                                          ('ego_teleport_rate', 'Ego Teleported'),
                                          ('emergency_rate', 'Emergency Stopping'),
                                          ('timeout_rate', 'Max Steps'))}
            rates['success_rate'] = 100 * sum(item['success'] for item in episodes) / 50
            rows.append({'profile': profile, 'seed': seed, **rates,
                         'mean_return': statistics.mean(item['return'] for item in episodes)})
            lane_rows.append({'profile': profile, 'seed': seed,
                              **{key: statistics.mean(item[key] for item in episodes)
                                 for key in ('lane_commands', 'toward_exit_requests',
                                             'completed_lane_changes', 'reverse_lane_changes',
                                             'hard_brakes', 'steps')},
                              **{key: statistics.mean(item[key] for item in episodes
                                                      if item[key] is not None)
                                 if any(item[key] is not None for item in episodes) else ''
                                 for key in ('first_toward_position', 'entry_e77_lane',
                                             'first_exit_lane_remaining_distance', 'mean_speed',
                                             'mean_acceleration', 'mean_action_change')}})
            components = sorted({key for item in episodes for key in item['reward_components']})
            reward_rows.append({'profile': profile, 'seed': seed,
                                **{key: statistics.mean(item['reward_components'].get(key, 0)
                                                       for item in episodes)
                                   for key in components}})
    write_csv(OUTPUT / 'evaluation_summary.csv', rows)
    write_csv(OUTPUT / 'lane_behavior_summary.csv', lane_rows)
    write_csv(OUTPUT / 'reward_components.csv', reward_rows)
    group_rows = []
    for profile in PROFILES:
        selected = [row for row in rows if row['profile'] == profile]
        group_rows.append({'profile': profile,
                           **{f'{key}_{stat}': (statistics.mean(values) if stat == 'mean'
                                                else statistics.stdev(values))
                              for key in rows[0] if key not in ('profile', 'seed')
                              for stat in ('mean', 'std')
                              for values in [[row[key] for row in selected]]}})
    write_csv(OUTPUT / 'group_summary.csv', group_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('train', 'evaluate', 'summarize', 'all'),
                        default='all')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.stage in ('train', 'all'):
        train_all(args.device)
    if args.stage in ('evaluate', 'all'):
        evaluate_all(args.device)
    if args.stage in ('summarize', 'all'):
        summarize()


if __name__ == '__main__':
    main()
