"""Evaluate an Antwerp checkpoint in a process separate from training."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from src.delay_encoder import DelayAwareGRUEncoder
from src.merging import ANTWERP_ENV_VERSION, ANTWERP_TASKS, Merging
from src import exit_e76_d3_config as exit_config
from src.wrapper import ABLATION_MODES, DelayWrapper, SafetyWrapper


OUTCOMES = ('Arrived', 'Collision', 'Max Steps', 'Emergency Stopping')
TIMEOUT_DETAILS = ('No Merge Timeout', 'Merged Timeout')
EXIT_TIMEOUT_DETAILS = ('No Exit Timeout', 'Exit Reached Timeout')
ARRIVAL_DETAILS = ('Uninterrupted Arrived', 'Teleported Arrived', 'Ego Teleported')
REPORT_METRICS = ('success_rate', 'raw_arrival_rate', 'collision_rate', 'timeout_rate',
                  'no_merge_rate', 'emergency_stopping_rate', 'ego_teleport_rate',
                  'mean_return')


def make_env(seed, scenario, delay_mode, safety, ablation, task='legacy_o3_d1',
             potential_gamma=None):
    env = DelayWrapper(Merging(seed=seed, scenario=scenario, task=task,
                               potential_gamma=potential_gamma),
                       max_delay=20, mode='all', delay_mode=delay_mode,
                       seed=seed, ablation=ablation)
    return SafetyWrapper(env) if safety else env


def validate_checkpoint_setup(model, ablation, safety, task='legacy_o3_d1'):
    """Validate policy compatibility; evaluation safety is an intervention.

    Training safety remains checkpoint provenance, not a restriction on which
    evaluation wrapper can be used. Never relabel or mutate the checkpoint.
    """
    if getattr(model, 'antwerp_env_version', None) != ANTWERP_ENV_VERSION:
        raise ValueError('Checkpoint uses an older Antwerp observation/reward contract; retrain it')
    if hasattr(model, 'ablation') and model.ablation != ablation:
        raise ValueError(f'Checkpoint ablation={model.ablation!r}, requested {ablation!r}')
    if task not in ANTWERP_TASKS:
        raise ValueError(f'Unknown Antwerp task: {task}')
    checkpoint_task = getattr(model, 'antwerp_task', 'legacy_o3_d1')
    if checkpoint_task != task:
        raise ValueError(f'Checkpoint task={checkpoint_task!r}, requested {task!r}')
    if task == exit_config.TASK_ID and getattr(model, 'exit_potential_beta', None) != exit_config.POTENTIAL_BETA:
        raise ValueError('Checkpoint E76-to-D3 potential configuration does not match')
    if task == exit_config.TASK_ID and getattr(model, 'exit_reward_profile', None) != exit_config.REWARD_PROFILE:
        raise ValueError('Checkpoint E76-to-D3 reward profile does not match')
    potential_gamma = None
    if task == exit_config.TASK_ID:
        potential_gamma = float(getattr(model, 'exit_potential_gamma', model.gamma))
        if not np.isclose(potential_gamma, float(model.gamma), rtol=0.0, atol=1e-12):
            raise ValueError('Checkpoint SAC and E76-to-D3 potential gamma do not match')
    setup = {'training_safety': getattr(model, 'training_safety', None),
             'evaluation_safety': bool(safety)}
    if potential_gamma is not None:
        setup['potential_gamma'] = potential_gamma
    return setup


def checkpoint_gru_input_mode(model):
    """Recover the input contract of old checkpoints without stored mode metadata."""
    kwargs = getattr(model, 'policy_kwargs', {}) or {}
    if kwargs.get('features_extractor_class') is not DelayAwareGRUEncoder:
        return None
    return kwargs.get('features_extractor_kwargs', {}).get(
        'input_mode', 'antwerp_scaled_reversed')


def evaluate(model, env, episodes, seed, task='legacy_o3_d1'):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    counts = Counter()
    returns = []
    records = []
    for episode in range(episodes):
        observation, reset_info = env.reset(seed=seed + episode)
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
                    if ANTWERP_TASKS[task][2] == 'exit':
                        counts['Exit Reached Timeout' if info.get('stage_reached')
                               else 'No Exit Timeout'] += 1
                    else:
                        counts['Merged Timeout' if info.get('merged')
                               else 'No Merge Timeout'] += 1
                returns.append(episode_return)
                records.append({'seed': seed + episode, 'steps': steps, 'outcome': outcome,
                                'merged': bool(info.get('merged')), 'ego_teleported': teleported,
                                'stage_reached': bool(info.get('stage_reached', info.get('merged'))),
                                'stage_reward_paid': bool(info.get('stage_reward_paid', False)),
                                'success': outcome == 'Arrived' and not teleported,
                                'return': episode_return, 'reset_info': reset_info})
                break
    timeout_details = EXIT_TIMEOUT_DETAILS if ANTWERP_TASKS[task][2] == 'exit' else TIMEOUT_DETAILS
    result = {**{outcome: counts[outcome] for outcome in OUTCOMES},
            **{outcome: counts[outcome] for outcome in timeout_details},
            **{outcome: counts[outcome] for outcome in ARRIVAL_DETAILS},
            'success_rate': 100 * counts['Uninterrupted Arrived'] / episodes,
            'raw_arrival_rate': 100 * counts['Arrived'] / episodes,
            'ego_teleport_rate': 100 * counts['Ego Teleported'] / episodes,
            'collision_rate': 100 * counts['Collision'] / episodes,
            'timeout_rate': 100 * counts['Max Steps'] / episodes,
            'emergency_stopping_rate': 100 * counts['Emergency Stopping'] / episodes,
            'mean_return': float(np.mean(returns)), 'episode_results': records}
    if ANTWERP_TASKS[task][2] != 'exit':
        result['no_merge_rate'] = result['timeout_rate']  # historical report field
    return result


def validate_seed_ranges(episodes, seeds):
    if episodes < 1 or not seeds or any(seed < 0 for seed in seeds):
        raise ValueError('Use positive episodes and non-negative evaluation seeds')
    starts = sorted(seeds)
    if any(right < left + episodes for left, right in zip(starts, starts[1:])):
        raise ValueError('Evaluation seed ranges overlap; space run seeds by at least --episodes')


def summarize_runs(runs, metrics=REPORT_METRICS):
    """Keep sample spread separate from the upstream normal-approximation CI."""
    if not runs:
        raise ValueError('At least one evaluation run is required')
    summary = {}
    for key in metrics:
        values = [run[key] for run in runs]
        std = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary[key] = {'mean': float(np.mean(values)), 'std': std,
                        'ci95_half_width': 1.96 * std / np.sqrt(len(values)) if std is not None else None}
    return summary


def evaluate_runs(model, scenario, delay_mode, safety, ablation, episodes, seeds,
                  task='legacy_o3_d1'):
    """Repeat evaluation of ONE checkpoint; this is not training-seed variance."""
    validate_seed_ranges(episodes, seeds)
    setup = validate_checkpoint_setup(model, ablation, safety, task)
    runs = []
    for seed in seeds:
        env = make_env(seed, scenario, delay_mode, safety, ablation, task=task,
                       potential_gamma=setup.get('potential_gamma'))
        try:
            runs.append({'seed': seed, 'episodes': episodes,
                         **evaluate(model, env, episodes, seed, task=task)})
        finally:
            env.close()
    return {'protocol': 'antwerp-v2-evaluation-v1', 'scenario': scenario,
            'task': task, 'stage_name': ANTWERP_TASKS[task][2],
            'delay_mode': delay_mode, 'ablation': ablation, **setup,
            'gru_input_mode': checkpoint_gru_input_mode(model),
            'antwerp_env_version': ANTWERP_ENV_VERSION,
            'training_seed': getattr(model, 'seed', None),
            'training_timesteps': getattr(model, 'num_timesteps', None),
            'deterministic': True, 'evaluation_seeds': list(seeds),
            'episodes_per_seed': episodes, 'total_episodes': episodes * len(seeds),
            'std_ddof': 1, 'ci_method': '1.96 * sample_std / sqrt(evaluation_runs)',
            'aggregation_unit': 'evaluation_run_of_one_checkpoint',
            'runs': runs, 'aggregate': summarize_runs(
                runs, tuple(key for key in REPORT_METRICS
                            if ANTWERP_TASKS[task][2] != 'exit' or key != 'no_merge_rate'))}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Antwerp SAC checkpoint (.zip)')
    parser.add_argument('--scenario', choices=('easy', 'medium', 'hard'), default='medium')
    parser.add_argument('--task', choices=tuple(ANTWERP_TASKS), default='legacy_o3_d1')
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
                           args.ablation, args.episodes, args.seeds, task=args.task)
    report.update(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                  device=args.device, torch_threads=args.torch_threads)
    print(f"{args.scenario} | {args.delay_mode} | task={args.task} | ablation={args.ablation} | "
          f"training_safety={report['training_safety']} | evaluation_safety={args.safety}")
    for run in report['runs']:
        print(f"Seed {run['seed']} | episodes={args.episodes}")
        timeout_details = (EXIT_TIMEOUT_DETAILS if ANTWERP_TASKS[args.task][2] == 'exit'
                           else TIMEOUT_DETAILS)
        for outcome in (*OUTCOMES, *timeout_details, *ARRIVAL_DETAILS):
            print(f"  {outcome}: {run[outcome]}/{args.episodes}")
    print('Across evaluation runs of this checkpoint (std and upstream-style CI are separate):')
    for key, values in report['aggregate'].items():
        if values['std'] is None:
            print(f"  {key}: mean={values['mean']:.4f}; std/CI unavailable with one run")
        else:
            print(f"  {key}: mean={values['mean']:.4f}, std={values['std']:.4f}, "
                  f"CI95 half-width={values['ci95_half_width']:.4f}")
    print('success_rate excludes ego teleports; raw_arrival_rate includes them. '
          'timeout_rate counts all Max Steps; the legacy/entry no_merge_rate '
          'is a historical alias. Emergency stops stay separate.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
        print(f'Saved evaluation report to {args.output}')


if __name__ == '__main__':
    main()
