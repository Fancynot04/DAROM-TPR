"""Evaluate an Antwerp checkpoint in a process separate from training."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from src.merging import ANTWERP_ENV_VERSION, Merging
from src.wrapper import ABLATION_MODES, DelayWrapper, SafetyWrapper


OUTCOMES = ('Arrived', 'Collision', 'Max Steps', 'Emergency Stopping')
TIMEOUT_DETAILS = ('No Merge Timeout', 'Merged Timeout')
ARRIVAL_DETAILS = ('Uninterrupted Arrived', 'Teleported Arrived', 'Ego Teleported')
REPORT_METRICS = ('success_rate', 'raw_arrival_rate', 'collision_rate',
                  'no_merge_rate', 'emergency_stopping_rate', 'ego_teleport_rate',
                  'mean_return')


def make_env(seed, scenario, delay_mode, safety, ablation):
    env = DelayWrapper(Merging(seed=seed, scenario=scenario),
                       max_delay=20, mode='all', delay_mode=delay_mode,
                       seed=seed, ablation=ablation)
    return SafetyWrapper(env) if safety else env


def validate_checkpoint_setup(model, ablation, safety):
    """Validate policy compatibility; evaluation safety is an intervention.

    Training safety remains checkpoint provenance, not a restriction on which
    evaluation wrapper can be used. Never relabel or mutate the checkpoint.
    """
    if getattr(model, 'antwerp_env_version', None) != ANTWERP_ENV_VERSION:
        raise ValueError('Checkpoint uses an older Antwerp observation/reward contract; retrain it')
    if hasattr(model, 'ablation') and model.ablation != ablation:
        raise ValueError(f'Checkpoint ablation={model.ablation!r}, requested {ablation!r}')
    return {'training_safety': getattr(model, 'training_safety', None),
            'evaluation_safety': bool(safety)}


def evaluate(model, env, episodes, seed):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    counts = Counter()
    returns = []
    records = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        steps = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            steps += 1
            if terminated or truncated:
                outcome = info.get('message', 'Unknown')
                if outcome not in OUTCOMES:
                    raise ValueError(f'Unexpected episode outcome: {outcome!r}')
                counts[outcome] += 1
                teleported = bool(info['ego_teleported'])
                counts['Ego Teleported'] += int(teleported)
                if outcome == 'Arrived':
                    counts['Teleported Arrived' if teleported else 'Uninterrupted Arrived'] += 1
                if outcome == 'Max Steps':
                    counts['Merged Timeout' if info.get('merged') else 'No Merge Timeout'] += 1
                returns.append(episode_return)
                records.append({'seed': seed + episode, 'steps': steps, 'outcome': outcome,
                                'merged': bool(info.get('merged')), 'ego_teleported': teleported,
                                'success': outcome == 'Arrived' and not teleported,
                                'return': episode_return})
                break
    return {**{outcome: counts[outcome] for outcome in OUTCOMES},
            **{outcome: counts[outcome] for outcome in TIMEOUT_DETAILS},
            **{outcome: counts[outcome] for outcome in ARRIVAL_DETAILS},
            'success_rate': 100 * counts['Uninterrupted Arrived'] / episodes,
            'raw_arrival_rate': 100 * counts['Arrived'] / episodes,
            'ego_teleport_rate': 100 * counts['Ego Teleported'] / episodes,
            'collision_rate': 100 * counts['Collision'] / episodes,
            'no_merge_rate': 100 * counts['Max Steps'] / episodes,
            'emergency_stopping_rate': 100 * counts['Emergency Stopping'] / episodes,
            'mean_return': float(np.mean(returns)), 'episode_results': records}


def validate_seed_ranges(episodes, seeds):
    if episodes < 1 or not seeds or any(seed < 0 for seed in seeds):
        raise ValueError('Use positive episodes and non-negative evaluation seeds')
    starts = sorted(seeds)
    if any(right < left + episodes for left, right in zip(starts, starts[1:])):
        raise ValueError('Evaluation seed ranges overlap; space run seeds by at least --episodes')


def summarize_runs(runs):
    """Keep sample spread separate from the upstream normal-approximation CI."""
    if not runs:
        raise ValueError('At least one evaluation run is required')
    summary = {}
    for key in REPORT_METRICS:
        values = [run[key] for run in runs]
        std = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary[key] = {'mean': float(np.mean(values)), 'std': std,
                        'ci95_half_width': 1.96 * std / np.sqrt(len(values)) if std is not None else None}
    return summary


def evaluate_runs(model, scenario, delay_mode, safety, ablation, episodes, seeds):
    """Repeat evaluation of ONE checkpoint; this is not training-seed variance."""
    validate_seed_ranges(episodes, seeds)
    setup = validate_checkpoint_setup(model, ablation, safety)
    runs = []
    for seed in seeds:
        env = make_env(seed, scenario, delay_mode, safety, ablation)
        try:
            runs.append({'seed': seed, 'episodes': episodes,
                         **evaluate(model, env, episodes, seed)})
        finally:
            env.close()
    return {'protocol': 'antwerp-v2-evaluation-v1', 'scenario': scenario,
            'delay_mode': delay_mode, 'ablation': ablation, **setup,
            'antwerp_env_version': ANTWERP_ENV_VERSION,
            'training_seed': getattr(model, 'seed', None),
            'training_timesteps': getattr(model, 'num_timesteps', None),
            'deterministic': True, 'evaluation_seeds': list(seeds),
            'episodes_per_seed': episodes, 'total_episodes': episodes * len(seeds),
            'std_ddof': 1, 'ci_method': '1.96 * sample_std / sqrt(evaluation_runs)',
            'aggregation_unit': 'evaluation_run_of_one_checkpoint',
            'runs': runs, 'aggregate': summarize_runs(runs)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Antwerp SAC checkpoint (.zip)')
    parser.add_argument('--scenario', choices=('easy', 'medium', 'hard'), default='medium')
    parser.add_argument('--delay-mode', choices=('uniform', 'exponential', 'triangular', 'bursty', 'bimodal'), default='uniform')
    parser.add_argument('--episodes', type=int, default=51)
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument('--seed', type=int, default=42)
    seed_options.add_argument('--seeds', nargs='+', type=int,
                              help='Start seed of each run; episode seeds increment within a run')
    safety_options = parser.add_mutually_exclusive_group()
    safety_options.add_argument('--safety', action='store_true', dest='safety',
                                help='Enable evaluation safety, independently of training')
    safety_options.add_argument('--no-safety', action='store_false', dest='safety')
    parser.set_defaults(safety=False)
    parser.add_argument('--ablation', choices=ABLATION_MODES, default='full')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output', type=Path, help='Save a new JSON report; never overwrite')
    args = parser.parse_args(argv)
    args.seeds = args.seeds if args.seeds is not None else [args.seed]
    try:
        validate_seed_ranges(args.episodes, args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if args.torch_threads < 1:
        parser.error('--torch-threads must be positive')
    if args.output and args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        checkpoint = Path(str(checkpoint) + '.zip')
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = SAC.load(checkpoint, device=args.device)
    report = evaluate_runs(model, args.scenario, args.delay_mode, args.safety,
                           args.ablation, args.episodes, args.seeds)
    report.update(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                  device=args.device, torch_threads=args.torch_threads)
    print(f"{args.scenario} | {args.delay_mode} | ablation={args.ablation} | "
          f"training_safety={report['training_safety']} | evaluation_safety={args.safety}")
    for run in report['runs']:
        print(f"Seed {run['seed']} | episodes={args.episodes}")
        for outcome in (*OUTCOMES, *TIMEOUT_DETAILS, *ARRIVAL_DETAILS):
            print(f"  {outcome}: {run[outcome]}/{args.episodes}")
    print('Across evaluation runs of this checkpoint (std and upstream-style CI are separate):')
    for key, values in report['aggregate'].items():
        if values['std'] is None:
            print(f"  {key}: mean={values['mean']:.4f}; std/CI unavailable with one run")
        else:
            print(f"  {key}: mean={values['mean']:.4f}, std={values['std']:.4f}, "
                  f"CI95 half-width={values['ci95_half_width']:.4f}")
    print('success_rate excludes ego teleports; raw_arrival_rate includes them. '
          'Legacy no_merge_rate counts all Max Steps; emergency stops stay separate.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
        print(f'Saved evaluation report to {args.output}')


if __name__ == '__main__':
    main()
