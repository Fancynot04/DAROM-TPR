"""Evaluate an Antwerp checkpoint in a process separate from training."""

import argparse
from collections import Counter
import hashlib
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from src.delay_encoder import DelayAwareGRUEncoder
from src.merging import ANTWERP_ENV_VERSION, ANTWERP_TASKS, Merging
from src.c2_navigation_reward import C2Merging, navigation_contract
from src import exit_e76_d3_config as exit_config
from src.wrapper import ABLATION_MODES, DelayWrapper, SafetyWrapper


OUTCOMES = ('Arrived', 'Collision', 'Max Steps', 'Emergency Stopping',
            'Ego Teleported', 'Missed Exit')
TIMEOUT_DETAILS = ('No Merge Timeout', 'Merged Timeout')
EXIT_TIMEOUT_DETAILS = ('No Exit Timeout', 'Exit Reached Timeout')
ARRIVAL_DETAILS = ('Uninterrupted Arrived', 'Teleported Arrived', 'Ego Teleported')
REPORT_METRICS = ('success_rate', 'raw_arrival_rate', 'collision_rate', 'timeout_rate',
                  'no_merge_rate', 'emergency_stopping_rate', 'ego_teleport_rate',
                  'missed_exit_rate', 'mean_return')


def make_env(seed, scenario, delay_mode, safety, ablation, task='legacy_o3_d1',
             reward_profile=None, c2_nav_reward=False):
    env_type = C2Merging if c2_nav_reward else Merging
    env = DelayWrapper(env_type(seed=seed, scenario=scenario, task=task,
                                reward_profile=reward_profile),
                       max_delay=20, mode='all', delay_mode=delay_mode,
                       seed=seed, ablation=ablation)
    return SafetyWrapper(env) if safety else env


def validate_checkpoint_setup(model, ablation, safety, task='legacy_o3_d1',
                              reward_profile=None, c2_nav_reward=False):
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
    expected_beta = 0.0 if reward_profile else exit_config.POTENTIAL_BETA
    expected_profile = reward_profile or exit_config.REWARD_PROFILE
    if task == exit_config.TASK_ID and getattr(model, 'exit_potential_beta', None) != expected_beta:
        raise ValueError('Checkpoint E76-to-D3 potential configuration does not match')
    if task == exit_config.TASK_ID and getattr(model, 'exit_reward_profile', None) != expected_profile:
        raise ValueError('Checkpoint E76-to-D3 reward profile does not match')
    if reward_profile:
        if reward_profile not in exit_config.ALL_REWARD_PROFILES:
            raise ValueError('Unknown reward profile')
        cfg = getattr(model, 'exit_training_config', None)
        expected = {'seed': (42, 43), 'total_timesteps': 200_000,
                    'task': exit_config.TASK_ID, 'scenario': 'easy',
                    'delay_mode': 'uniform', 'max_delay': 20,
                    'use_safety': False, 'encoder_mode': 'GRU',
                    'gru_input_mode': 'paper_raw', 'ablation': 'full',
                    'reward_profile': reward_profile, 'potential_beta': 0.0}
        if not isinstance(cfg, dict) or any(
                cfg.get(key) not in value if isinstance(value, tuple) else cfg.get(key) != value
                for key, value in expected.items()):
            raise ValueError('Checkpoint basic reward training configuration does not match')
        if getattr(model, 'num_timesteps', None) != 200_000:
            raise ValueError('Checkpoint has not completed 200k steps')
        sac = cfg.get('sac', {})
        required_sac = {'learning_rate': 3e-5, 'batch_size': 512,
                        'buffer_size': 1_000_000, 'learning_starts': 100,
                        'gamma': 0.99}
        if any(sac.get(key) != value for key, value in required_sac.items()):
            raise ValueError('Checkpoint SAC configuration does not match')
        if getattr(model, 'observation_space', None) is None or model.observation_space.shape != (134,):
            raise ValueError('Checkpoint observation contract must be 134 dimensions')
    if c2_nav_reward:
        if task != exit_config.TASK_ID or reward_profile != 'B3':
            raise ValueError('C2 evaluation requires the B3 E76-to-D3 task')
        if (getattr(model, 'c2_navigation_contract', None) != navigation_contract()
                or not getattr(model, 'exit_training_config', {}).get('c2_nav_reward')
                or (getattr(model, 'policy_kwargs', {}) or {}).get('features_extractor_class')
                is not DelayAwareGRUEncoder):
            raise ValueError('C2 checkpoint navigation-reward contract does not match')
    elif isinstance(getattr(model, 'c2_navigation_contract', None), dict):
        raise ValueError('C2 checkpoint requires --c2-nav-reward')
    return {'training_safety': getattr(model, 'training_safety', None),
            'evaluation_safety': bool(safety)}


def checkpoint_gru_input_mode(model):
    """Recover the input contract of old checkpoints without stored mode metadata."""
    kwargs = getattr(model, 'policy_kwargs', {}) or {}
    if kwargs.get('features_extractor_class') is not DelayAwareGRUEncoder:
        return None
    return kwargs.get('features_extractor_kwargs', {}).get(
        'input_mode', 'antwerp_scaled_reversed')


def evaluate(model, env, episodes, seed, task='legacy_o3_d1',
             c2_nav_reward=False, trace_sink=None):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    counts = Counter()
    returns = []
    records = []
    for episode in range(episodes):
        observation, reset_info = env.reset(seed=seed + episode)
        episode_return = 0.0
        steps = 0
        reward_totals = Counter()
        lane_commands = toward_exit = completed_changes = reverse_changes = hard_brakes = 0
        first_toward_position = first_exit_lane_distance = entry_e77_lane = None
        entry_e77_n = entry_e77_valid = first_stable_e77_n = None
        last_required_change_distance = None
        last_stable_n = reset_info.get('navigation_n') if c2_nav_reward else None
        changed_since_positive = False
        speeds, accelerations, action_changes = [], [], []
        previous_command_acceleration = 0.0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            steps += 1
            for key, value in info.get('reward_components', {}).items():
                reward_totals[key] += float(value)
            direction = int(info.get('command_direction', 0))
            lane_commands += int(direction != 0)
            toward_exit += int(direction < 0)
            reverse_changes += int(info.get('lane_change_delta', 0) > 0)
            completed_changes += int(info.get('lane_change_completed', False))
            if direction < 0 and first_toward_position is None:
                first_toward_position = info.get('ego_distance')
            if info.get('road') == 'E77' and entry_e77_lane is None:
                entry_e77_lane = info.get('lane_index')
                if c2_nav_reward:
                    entry_e77_n = info['navigation_n']
                    entry_e77_valid = info['navigation_valid']
            if c2_nav_reward:
                if (info.get('road') == 'E77' and info['navigation_valid']
                        and first_stable_e77_n is None):
                    first_stable_e77_n = info['navigation_n']
                if info.get('lane_change_completed') and last_stable_n is not None and last_stable_n > 0:
                    changed_since_positive = True
                if info['navigation_valid']:
                    n = info['navigation_n']
                    if (n == 0 and info['navigation_source'] == 'settled'
                            and info.get('road') in ('E76', 'E77')
                            and last_stable_n is not None and last_stable_n > 0):
                        if changed_since_positive:
                            last_required_change_distance = info.get('distance_to_d3')
                        changed_since_positive = False
                    last_stable_n = n
                if trace_sink is not None:
                    trace_sink.write(json.dumps({
                        'evaluation_seed': seed + episode, 'step': steps,
                        'road': info.get('road'), 'lane_index': info.get('lane_index'),
                        'navigation_n': info['navigation_n'],
                        'navigation_valid': info['navigation_valid'],
                        'navigation_source': info['navigation_source'],
                        'navigation_reward': info['reward_components']['navigation'],
                        'navigation_reward_total': info['navigation_reward_total'],
                        'lane_change_completed': info.get('lane_change_completed'),
                        'lane_change_delta': info.get('lane_change_delta'),
                        'distance_to_d3': info.get('distance_to_d3'),
                        'ego_speed': info.get('ego_speed'),
                        'terminated': terminated, 'truncated': truncated,
                        'outcome': info.get('message'),
                    }, allow_nan=False) + '\n')
            if (info.get('road') in ('E76', 'E77') and info.get('lane_index') == 0
                    and first_exit_lane_distance is None):
                first_exit_lane_distance = info.get('distance_to_d3')
            if info.get('ego_speed') is not None:
                speeds.append(float(info['ego_speed']))
            if info.get('ego_acceleration') is not None:
                acceleration = float(info['ego_acceleration'])
                accelerations.append(acceleration)
                hard_brakes += int(acceleration < -4.5)
            parsed_acceleration = float(np.clip(action[0], -5, 5))
            action_changes.append(abs(parsed_acceleration - previous_command_acceleration))
            previous_command_acceleration = parsed_acceleration
            if terminated or truncated:
                outcome = info.get('message', 'Unknown')
                if outcome not in OUTCOMES:
                    raise ValueError(f'Unexpected episode outcome: {outcome!r}')
                counts[outcome] += 1
                teleported = bool(info['ego_teleported'])
                if outcome != 'Ego Teleported':
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
                                'return': episode_return, 'reset_info': reset_info,
                                'lane_commands': lane_commands,
                                'toward_exit_requests': toward_exit,
                                'completed_lane_changes': completed_changes,
                                'reverse_lane_changes': reverse_changes,
                                'first_toward_position': first_toward_position,
                                'entry_e77_lane': entry_e77_lane,
                                **({'entry_e77_n': entry_e77_n,
                                    'entry_e77_n_valid': entry_e77_valid,
                                    'first_stable_e77_n': first_stable_e77_n,
                                    'last_required_change_distance': last_required_change_distance,
                                    'navigation_reward_total': info['navigation_reward_total']}
                                   if c2_nav_reward else {}),
                                'first_exit_lane_remaining_distance': first_exit_lane_distance,
                                'mean_speed': float(np.mean(speeds)) if speeds else None,
                                'mean_acceleration': float(np.mean(accelerations)) if accelerations else None,
                                'mean_action_change': float(np.mean(action_changes)),
                                'hard_brakes': hard_brakes,
                                'reward_components': dict(reward_totals)})
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
            'missed_exit_rate': 100 * counts['Missed Exit'] / episodes,
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
                  task='legacy_o3_d1', reward_profile=None,
                  c2_nav_reward=False, trace_sink=None):
    """Repeat evaluation of ONE checkpoint; this is not training-seed variance."""
    validate_seed_ranges(episodes, seeds)
    setup = validate_checkpoint_setup(model, ablation, safety, task, reward_profile,
                                      c2_nav_reward)
    if reward_profile and (scenario != 'easy' or delay_mode != 'uniform' or safety or
                           episodes != 50 or seeds != [200001]):
        raise ValueError('Basic reward evaluation requires Easy, uniform, safety off and seeds 200001-200050')
    runs = []
    for seed in seeds:
        env = make_env(seed, scenario, delay_mode, safety, ablation, task=task,
                       reward_profile=reward_profile, c2_nav_reward=c2_nav_reward)
        try:
            runs.append({'seed': seed, 'episodes': episodes,
                         **evaluate(model, env, episodes, seed, task=task,
                                    c2_nav_reward=c2_nav_reward,
                                    trace_sink=trace_sink)})
        finally:
            env.close()
    return {'protocol': 'antwerp-v2-evaluation-v1', 'scenario': scenario,
            'task': task, 'stage_name': ANTWERP_TASKS[task][2],
            'delay_mode': delay_mode, 'ablation': ablation, **setup,
            'gru_input_mode': checkpoint_gru_input_mode(model),
            'antwerp_env_version': ANTWERP_ENV_VERSION,
            'training_seed': getattr(model, 'seed', None),
            'training_timesteps': getattr(model, 'num_timesteps', None),
            **({'c2_navigation_contract': model.c2_navigation_contract,
                'last_required_change_distance_definition':
                    'Distance to D3 at last E76/E77 settled n>0 to n=0 transition '
                    'with an actual completed lane change since the positive settled n; '
                    'zero only means a route exists'}
               if c2_nav_reward else {}),
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
    parser.add_argument('--reward-profile', choices=tuple(exit_config.ALL_REWARD_PROFILES),
                        default=None)
    parser.add_argument('--c2-nav-reward', action='store_true',
                        help='Evaluate a C2 checkpoint with its route-topology reward')
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
    parser.add_argument('--trace-output', type=Path,
                        help='Save a new C2 stepwise gzip JSONL trace; never overwrite')
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
    if args.c2_nav_reward and (args.task != exit_config.TASK_ID
                               or args.reward_profile != 'B3'
                               or args.output is None or args.trace_output is None):
        parser.error('C2 evaluation requires B3 E76-to-D3 and both output paths')
    if args.trace_output and (not args.c2_nav_reward or args.trace_output.exists()):
        parser.error('--trace-output requires C2 and a new path')
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        checkpoint = Path(str(checkpoint) + '.zip')
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = SAC.load(checkpoint, device=args.device)
    validate_checkpoint_setup(model, args.ablation, args.safety, args.task,
                              args.reward_profile, args.c2_nav_reward)
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    trace_sink = gzip.open(args.trace_output, 'xt') if args.trace_output else None
    try:
        report = evaluate_runs(model, args.scenario, args.delay_mode, args.safety,
                               args.ablation, args.episodes, args.seeds, task=args.task,
                               reward_profile=args.reward_profile,
                               c2_nav_reward=args.c2_nav_reward,
                               trace_sink=trace_sink)
    finally:
        if trace_sink is not None:
            trace_sink.close()
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
