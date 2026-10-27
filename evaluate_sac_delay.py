"""Evaluate an Antwerp checkpoint in a process separate from training."""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from src.delay_encoder import DelayAwareGRUEncoder
from src.merging import (ANTWERP_DIR, ANTWERP_ENV_VERSION, ANTWERP_TASKS,
                         LANE_CHANGE_EDGES, Merging, parseAction)
from src import exit_e76_d3_config as exit_config
from src.route_urgency import load_route_network, observe_route_urgency
from src.wrapper import ABLATION_MODES, DelayWrapper, SafetyWrapper


OUTCOMES = ('Arrived', 'Collision', 'Max Steps', 'Emergency Stopping', 'Missed Exit')
TIMEOUT_DETAILS = ('No Merge Timeout', 'Merged Timeout')
EXIT_TIMEOUT_DETAILS = ('No Exit Timeout', 'Exit Reached Timeout')
ARRIVAL_DETAILS = ('Uninterrupted Arrived', 'Teleported Arrived', 'Ego Teleported')
REPORT_METRICS = ('success_rate', 'raw_arrival_rate', 'collision_rate', 'timeout_rate',
                  'no_merge_rate', 'emergency_stopping_rate', 'ego_teleport_rate',
                  'missed_exit_rate', 'mean_return')


def make_env(seed, scenario, delay_mode, safety, ablation, task='legacy_o3_d1',
             potential_gamma=None, nav_features=False, exit_reward_profile=None):
    env = DelayWrapper(Merging(seed=seed, scenario=scenario, task=task,
                               potential_gamma=potential_gamma,
                               exit_reward_profile=exit_reward_profile),
                       max_delay=20, mode='all', delay_mode=delay_mode,
                       seed=seed, ablation=ablation, nav_features=nav_features)
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
    reward_profile = getattr(model, 'exit_reward_profile', None)
    if task == exit_config.TASK_ID:
        try:
            expected_beta = exit_config.potential_beta_for_profile(reward_profile)
        except ValueError as error:
            raise ValueError('Checkpoint E76-to-D3 reward profile does not match') from error
        if getattr(model, 'exit_potential_beta', None) != expected_beta:
            raise ValueError('Checkpoint E76-to-D3 potential configuration does not match')
    if task == exit_config.TASK_ID and getattr(model, 'exit_control_profile', None) != exit_config.CONTROL_PROFILE:
        raise ValueError('Checkpoint E76-to-D3 control profile does not match')
    potential_gamma = None
    nav_features = False
    if task == exit_config.TASK_ID:
        profile = getattr(model, 'exit_observation_profile', None)
        if profile not in (None, exit_config.NAV_OBSERVATION_PROFILE):
            raise ValueError(f'Unknown E76-to-D3 observation profile: {profile!r}')
        nav_features = profile == exit_config.NAV_OBSERVATION_PROFILE
        expected_dim = 136 if nav_features else 134
        if model.observation_space.shape != (expected_dim,):
            raise ValueError('Checkpoint E76-to-D3 observation profile and shape disagree')
        encoder_nav_dim = (getattr(model, 'policy_kwargs', {}) or {}).get(
            'features_extractor_kwargs', {}).get('nav_dim', 0)
        if encoder_nav_dim != (2 if nav_features else 0):
            raise ValueError('Checkpoint E76-to-D3 GRU navigation split disagrees')
        if reward_profile == exit_config.URGENCY_REWARD_PROFILE and nav_features:
            raise ValueError('Urgency reward v1 requires the 134-dimensional observation')
        potential_gamma = float(getattr(model, 'exit_potential_gamma', model.gamma))
        if not np.isclose(potential_gamma, float(model.gamma), rtol=0.0, atol=1e-12):
            raise ValueError('Checkpoint SAC and E76-to-D3 potential gamma do not match')
    setup = {'training_safety': getattr(model, 'training_safety', None),
             'evaluation_safety': bool(safety)}
    if potential_gamma is not None:
        setup['potential_gamma'] = potential_gamma
        setup['exit_observation_profile'] = getattr(model, 'exit_observation_profile', None)
        setup['nav_features'] = nav_features
        setup['exit_reward_profile'] = reward_profile
    return setup


def checkpoint_gru_input_mode(model):
    """Recover the input contract of old checkpoints without stored mode metadata."""
    kwargs = getattr(model, 'policy_kwargs', {}) or {}
    if kwargs.get('features_extractor_class') is not DelayAwareGRUEncoder:
        return None
    return kwargs.get('features_extractor_kwargs', {}).get(
        'input_mode', 'antwerp_scaled_reversed')


def e77_lane_index(env):
    """Read realized E77 lane for diagnostics, never from the policy input."""
    base = getattr(env, 'unwrapped', None)
    lane = getattr(base, 'lane_idx', None)
    if (getattr(base, 'ego_on_road', False)
            and getattr(base, 'lane_name', None) == 'E77'
            and isinstance(lane, Integral)):
        return int(lane)
    return None


def diagnostic_reward_without_potential(reward, info, task):
    """Remove shaping only from the reported diagnostic, not the environment."""
    potential = float(info.get('potential_reward', 0.0))
    if task == exit_config.TASK_ID:
        components = info.get('reward_components', {})
        if ('potential' not in components
                or not math.isclose(potential, float(components['potential']),
                                    rel_tol=0.0, abs_tol=1e-6)):
            raise ValueError('Missing or inconsistent E76-to-D3 potential component')
    unshaped = float(reward) - potential
    if not math.isfinite(unshaped):
        raise ValueError('Nonfinite diagnostic reward without potential')
    return potential, unshaped


def evaluate(model, env, episodes, seed, task='legacy_o3_d1',
             urgency_trace=None, route_network=None):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    counts = Counter()
    returns = []
    records = []
    for episode in range(episodes):
        observation, reset_info = env.reset(seed=seed + episode)
        episode_return = 0.0
        unshaped_episode_return = 0.0
        steps = 0
        toward_requests = away_requests = 0
        previous_e77_lane = None
        e77_zero_departures = 0
        while True:
            urgency_record = None
            if urgency_trace is not None:
                urgency_record = observe_route_urgency(
                    env.unwrapped, route_network, LANE_CHANGE_EDGES,
                    lane_change_seconds=exit_config.LANE_CHANGE_TIME_SECONDS)
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            if urgency_record is not None:
                potential, unshaped_reward = diagnostic_reward_without_potential(
                    reward, info, task)
                unshaped_episode_return += unshaped_reward
                urgency_record.update(
                    seed=seed + episode, step=steps, task=task,
                    actor_lateral_request=int(parseAction(action)[1]),
                    reward=float(reward), potential_reward=potential,
                    reward_without_potential=unshaped_reward,
                    cumulative_reward_without_potential=unshaped_episode_return,
                    after_lane=info.get('lane'),
                    outcome=info.get('message') if terminated or truncated else None)
                urgency_trace.write(json.dumps(urgency_record, allow_nan=False) + '\n')
            toward_requests += int(info.get('exit_actor_request_direction') == -1)
            away_requests += int(info.get('exit_actor_request_direction') == 1)
            if task == exit_config.TASK_ID:
                current_e77_lane = e77_lane_index(env)
                if previous_e77_lane == 0 and current_e77_lane is not None and current_e77_lane > 0:
                    e77_zero_departures += 1
                previous_e77_lane = current_e77_lane
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
                record = {'seed': seed + episode, 'steps': steps, 'outcome': outcome,
                                'merged': bool(info.get('merged')), 'ego_teleported': teleported,
                                'stage_reached': bool(info.get('stage_reached', info.get('merged'))),
                                'stage_reward_paid': bool(info.get('stage_reward_paid', False)),
                                'success': outcome == 'Arrived' and not teleported,
                                'toward_requests': toward_requests,
                                'away_requests': away_requests,
                                'return': episode_return, 'reset_info': reset_info}
                if task == exit_config.TASK_ID:
                    record['e77_zero_departures'] = e77_zero_departures
                    record['left_e77_zero_after_entry'] = e77_zero_departures > 0
                if urgency_trace is not None:
                    record['return_without_potential'] = unshaped_episode_return
                records.append(record)
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
    else:
        result['toward_requests'] = sum(item['toward_requests'] for item in records)
        result['away_requests'] = sum(item['away_requests'] for item in records)
        if task == exit_config.TASK_ID:
            result['e77_zero_redeparture_episodes'] = sum(
                item['left_e77_zero_after_entry'] for item in records)
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
                  task='legacy_o3_d1', urgency_trace_path=None):
    """Repeat evaluation of ONE checkpoint; this is not training-seed variance."""
    validate_seed_ranges(episodes, seeds)
    setup = validate_checkpoint_setup(model, ablation, safety, task)
    if urgency_trace_path is not None and ANTWERP_TASKS[task][2] != 'exit':
        raise ValueError('Route urgency diagnostics require an exit task')
    network = (load_route_network(ANTWERP_DIR / 'Antwerp.net.xml')
               if urgency_trace_path is not None else None)
    runs = []
    with ExitStack() as stack:
        urgency_trace = None
        if urgency_trace_path is not None:
            urgency_trace_path = Path(urgency_trace_path)
            urgency_trace_path.parent.mkdir(parents=True, exist_ok=True)
            urgency_trace = stack.enter_context(urgency_trace_path.open('x'))
        for seed in seeds:
            env = make_env(seed, scenario, delay_mode, safety, ablation, task=task,
                           potential_gamma=setup.get('potential_gamma'),
                           nav_features=setup.get('nav_features', False),
                           exit_reward_profile=setup.get('exit_reward_profile'))
            try:
                runs.append({'seed': seed, 'episodes': episodes,
                             **evaluate(model, env, episodes, seed, task=task,
                                        urgency_trace=urgency_trace, route_network=network)})
            finally:
                env.close()
    report = {'protocol': 'antwerp-v2-evaluation-v1', 'scenario': scenario,
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
    if task == exit_config.TASK_ID:
        report['e77_zero_redeparture_episodes'] = sum(
            run['e77_zero_redeparture_episodes'] for run in runs)
    if urgency_trace_path is not None:
        report['route_urgency_trace'] = str(urgency_trace_path)
        report['route_urgency_profile'] = 'lane_graph_minimax_v1'
        report['diagnostic_reward_profile'] = 'environment_reward_minus_potential_v1'
    return report


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
    parser.add_argument('--urgency-trace', type=Path,
                        help='Write a new read-only per-step route-urgency JSONL trace')
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
    if args.urgency_trace and args.urgency_trace.exists():
        parser.error(f'Urgency trace already exists: {args.urgency_trace}')
    if args.urgency_trace and ANTWERP_TASKS[args.task][2] != 'exit':
        parser.error('--urgency-trace requires an exit task')
    if (args.output and args.urgency_trace
            and args.output.resolve() == args.urgency_trace.resolve()):
        parser.error('--output and --urgency-trace must be different files')
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
                           args.ablation, args.episodes, args.seeds, task=args.task,
                           urgency_trace_path=args.urgency_trace)
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
