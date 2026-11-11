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
from src import exit_progress_reward
from src.route_urgency import load_route_network, observe_route_urgency
from src.trajectory_recorder import TrajectoryRecorder
from src.wrapper import ABLATION_MODES, DelayWrapper, SafetyWrapper


OUTCOMES = ('Arrived', 'Collision', 'Timeout', 'Emergency Stopping',
            'Missed Exit', 'Ego Teleported')
TIMEOUT_DETAILS = ('No Merge Timeout', 'Merged Timeout')
EXIT_TIMEOUT_DETAILS = ('No Exit Timeout', 'Exit Reached Timeout')
ARRIVAL_DETAILS = ('Uninterrupted Arrived', 'Teleported Arrived', 'Ego Teleported')
REPORT_METRICS = ('success_rate', 'raw_arrival_rate', 'collision_rate', 'timeout_rate',
                  'no_merge_rate', 'emergency_stopping_rate', 'ego_teleport_rate',
                  'missed_exit_rate', 'mean_policy_reward_return',
                  'mean_b0_reference_return', 'hard_braking_episode_rate')
B0_REFERENCE_REWARDS = {
    'Arrived': 50.0,
    'Collision': -100.0,
    'Ego Teleported': -100.0,
    'Missed Exit': -100.0,
    'Timeout': -100.0,
}


def make_env(seed, scenario, delay_mode, safety, ablation, task=exit_config.TASK_ID,
             potential_gamma=None, nav_features=False, exit_reward_profile=None,
             progress_experiment=None, reward_config=None,
             reward_gamma=exit_config.DEFAULT_GAMMA, observation_mode='legacy'):
    env = DelayWrapper(Merging(seed=seed, scenario=scenario, task=task,
                               potential_gamma=potential_gamma,
                               exit_reward_profile=exit_reward_profile,
                               progress_experiment=progress_experiment,
                               reward_config=reward_config,
                               reward_gamma=reward_gamma),
                       max_delay=20, mode='all', delay_mode=delay_mode,
                       seed=seed, ablation=ablation, nav_features=nav_features,
                       observation_mode=observation_mode)
    return SafetyWrapper(env) if safety else env


def b0_reference_reward(outcome, ego_teleported=False):
    """Compute the common B0 terminal-event score without affecting execution."""
    if outcome in (None, '', 'Emergency Stopping'):
        return 0.0
    if outcome == 'Arrived' and ego_teleported:
        return -100.0
    if outcome not in B0_REFERENCE_REWARDS:
        raise ValueError(f'Unknown B0 terminal event: {outcome!r}')
    return B0_REFERENCE_REWARDS[outcome]


def checkpoint_reward_setup(model):
    """Load and authenticate the complete reward contract embedded in a checkpoint."""
    required = ('exit_reward_profile', 'exit_reward_config',
                'exit_reward_config_sha256')
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError('Checkpoint is missing required reward provenance: '
                         + ', '.join(missing))
    reward_config = exit_progress_reward.load_experiment(model.exit_reward_config)
    reward_hash = exit_progress_reward.reward_identity(reward_config)
    if model.exit_reward_config_sha256 != reward_hash:
        raise ValueError('Checkpoint reward configuration hash mismatch')
    if model.exit_reward_profile != reward_config['name']:
        raise ValueError('Checkpoint reward profile and configuration disagree')
    return model.exit_reward_profile, reward_config, reward_hash


def validate_checkpoint_setup(model, ablation, safety, task=exit_config.TASK_ID):
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
    checkpoint_task = getattr(model, 'antwerp_task', exit_config.TASK_ID)
    if checkpoint_task != task:
        raise ValueError(f'Checkpoint task={checkpoint_task!r}, requested {task!r}')
    reward_profile, reward_config, reward_config_hash = checkpoint_reward_setup(model)
    if reward_profile not in ('events_only_v1', 'e76_d3_dense_v1'):
        raise ValueError(f'Checkpoint has unsupported E76-to-D3 reward profile: {reward_profile!r}')
    progress_experiment = None
    potential_gamma = None
    nav_features = False
    if task == exit_config.TASK_ID:
        profile = getattr(model, 'exit_observation_profile', None)
        nav_features = False
        observation_mode = getattr(model, 'observation_mode', 'legacy')
        if observation_mode not in ('legacy', 'ego_lateral_v1', 'ego_spatial_v2'):
            raise ValueError('Unknown checkpoint observation mode')
        expected_dim = {'legacy': 134, 'ego_lateral_v1': 136, 'ego_spatial_v2': 137}[observation_mode]
        encoder_mode = (getattr(model, 'policy_kwargs', {}) or {}).get(
            'features_extractor_kwargs', {}).get('observation_mode', 'legacy')
        if encoder_mode != observation_mode:
            raise ValueError('Checkpoint encoder and observation modes disagree')
        if model.observation_space.shape != (expected_dim,):
            raise ValueError('Checkpoint E76-to-D3 observation profile and shape disagree')
        encoder_nav_dim = (getattr(model, 'policy_kwargs', {}) or {}).get(
            'features_extractor_kwargs', {}).get('nav_dim', 0)
        if encoder_nav_dim != (2 if nav_features else 0):
            raise ValueError('Checkpoint E76-to-D3 GRU navigation split disagrees')
        potential_gamma = None
    setup = {'observation_mode': observation_mode, 'training_safety': getattr(model, 'training_safety', None),
             'evaluation_safety': bool(safety),
             'exit_reward_profile': reward_profile,
             'reward_config': reward_config,
             'reward_config_sha256': reward_config_hash,
             'reward_config_source': 'checkpoint',
             'reward_gamma': float(model.gamma)}
    if potential_gamma is not None:
        setup['potential_gamma'] = potential_gamma
        setup['exit_observation_profile'] = getattr(model, 'exit_observation_profile', None)
        setup['nav_features'] = nav_features
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
    return 0.0, float(reward)


def evaluate(model, env, episodes, seed, task=exit_config.TASK_ID,
             urgency_trace=None, route_network=None, recorder=None):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    counts = Counter()
    policy_returns = []
    b0_reference_returns = []
    records = []
    for episode in range(episodes):
        observation, reset_info = env.reset(seed=seed + episode)
        base = env.unwrapped
        if recorder is not None:
            recorder.begin_episode(run_seed=seed, episode_index=episode,
                                   eval_seed=seed + episode, reset_info=reset_info)
        secrm = False
        directional = False
        boundary_counts = Counter()
        boundary_max_area = 0.0
        reward_totals = Counter()
        reward_term_totals = Counter()
        lane_events = Counter()
        first_direct_exit_lane_remaining = None
        previous_lane_direction = None
        reversals = 0
        discounted_return = 0.0
        episode_return = 0.0
        unshaped_episode_return = 0.0
        steps = 0
        toward_requests = away_requests = 0
        previous_e77_lane = None
        e77_zero_departures = 0
        lane_event_details = []
        first_toward_request_distance = None
        first_e77_zero_distance = None
        stopped_steps = 0
        stopped_streak = 0
        max_stopped_streak = 0
        e77_zero_traffic_samples = []
        speed_samples = []
        commands_sent = Counter()
        previous_step_request = 0
        same_direction_repeat_requests = 0
        current_request_run = 0
        max_same_direction_request_run = 0
        last_nonzero_request = None
        short_horizon_request_reversals = 0
        while True:
            if (getattr(base, 'ego_on_road', False)
                    and not getattr(base, 'ego_teleporting', False)
                    and hasattr(base, 'traci')):
                measured_speed = float(base.traci.vehicle.getSpeed(base.ego_id))
                if not math.isfinite(measured_speed):
                    raise ValueError('Non-finite evaluation speed')
                speed_samples.append(measured_speed)
            if secrm and base.ego_on_road:
                vehicle = base.traci.vehicle
                speed = float(vehicle.getSpeed(base.ego_id))
                acceleration = float(vehicle.getAcceleration(base.ego_id))
                stopped_streak = stopped_streak + 1 if speed < 0.5 else 0
                max_stopped_streak = max(max_stopped_streak, stopped_streak)
                stopped_steps += int(speed < 0.5)
                if (base.lane_name == 'E77' and base.lane_idx == 0
                        and (speed < 0.5 or acceleration
                             <= exit_config.HARD_BRAKING_THRESHOLD_MPS2)):
                    leader = vehicle.getLeader(base.ego_id, 100.0)
                    gap = (float(leader[1]) + float(vehicle.getMinGap(base.ego_id))
                           if leader and leader[0] else None)
                    e77_zero_traffic_samples.append({
                        'step': steps, 'speed_mps': speed, 'acceleration_mps2': acceleration,
                        'leader_id': leader[0] if leader else None, 'leader_physical_gap_m': gap,
                        'leader_speed_mps': float(vehicle.getSpeed(leader[0])) if leader and leader[0] else None,
                        'distance_to_d3_start_m': base._remaining_to_d3_start(),
                    })
                if (first_e77_zero_distance is None and base.lane_name == 'E77'
                        and base.lane_idx == 0 and base.exit_maneuver is None
                        and base._exit_laterally_settled()):
                    first_e77_zero_distance = base._remaining_to_d3_start()
            if directional and first_direct_exit_lane_remaining is None and base.ego_on_road:
                road = base.traci.vehicle.getRoadID(base.ego_id)
                lane = int(base.traci.vehicle.getLaneIndex(base.ego_id))
                route_zero = (base._secrm_snapshot()['route'].get('required_changes') == 0
                              if secrm else bool(base.route_network.connections.get(
                                  (road, exit_config.DESTINATION, lane))))
                if (route_zero and base._exit_laterally_settled() and base.exit_maneuver is None):
                    first_direct_exit_lane_remaining = base._remaining_to_d3_end()
            urgency_record = None
            if urgency_trace is not None:
                urgency_record = observe_route_urgency(
                    env.unwrapped, route_network, LANE_CHANGE_EDGES,
                    lane_change_seconds=exit_config.LANE_CHANGE_TIME_SECONDS)
            action, _ = model.predict(observation, deterministic=True)
            trajectory_pre = (recorder.capture_pre(base, observation, action)
                              if recorder is not None else None)
            observation, reward, terminated, truncated, info = env.step(action)
            if recorder is not None:
                recorder.record_step(trajectory_pre, base, observation, reward,
                                     terminated, truncated, info)
            for term in info.get('reward_terms', ()):
                reward_term_totals[term['name']] += float(term['weighted'])
            if secrm:
                boundary = info.get('road_boundary') or {'status': 'unknown'}
                boundary_counts[boundary['status']] += 1
                boundary_max_area = max(boundary_max_area, boundary.get('uncovered_area_m2', 0.0))
            if directional:
                discounted_return += model.gamma ** steps * float(reward)
                for name, value in info['reward_components'].items():
                    reward_totals[name] += float(value)
                event = info.get('directional_lane_change_event')
                if event is not None:
                    lane_event_details.append(dict(event))
                    if (event['route_direction'] == 'toward'
                            and first_toward_request_distance is None):
                        first_toward_request_distance = event.get('request_distance_to_d3_start_m')
                    lane_events[event['route_direction']] += 1
                    if (previous_lane_direction is not None
                            and previous_lane_direction != event['actual_direction']):
                        reversals += 1
                    previous_lane_direction = event['actual_direction']
                    before, after = event['before'], event['after']
                    left_favorable = (before['minimum_lane_changes'] == 0 and after['minimum_lane_changes'] > 0
                                      if secrm else bool(base.route_network.connections.get(
                                          (before['road'], exit_config.DESTINATION, before['lane'])))
                                      and not base.route_network.connections.get(
                                          (after['road'], exit_config.DESTINATION, after['lane'])))
                    if (left_favorable and event['request_direction'] == event['actual_direction']):
                        lane_events['left_direct_exit_lane'] += 1
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
            request_direction = int(info.get('exit_actor_request_direction', 0))
            if request_direction:
                if request_direction == previous_step_request:
                    same_direction_repeat_requests += 1
                    current_request_run += 1
                else:
                    current_request_run = 1
                max_same_direction_request_run = max(
                    max_same_direction_request_run, current_request_run)
                if (last_nonzero_request is not None
                        and request_direction != last_nonzero_request[0]
                        and steps - last_nonzero_request[1] <= 10):
                    short_horizon_request_reversals += 1
                last_nonzero_request = (request_direction, steps)
            else:
                current_request_run = 0
            previous_step_request = request_direction
            applied_direction = int(info.get('exit_applied_lane_change_direction', 0))
            if applied_direction:
                commands_sent['total'] += 1
                commands_sent['toward' if applied_direction < 0 else 'away'] += 1
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
                if recorder is not None:
                    recorder.end_episode(outcome, steps)
                counts[outcome] += 1
                sumo_emergency_steps = int(info['sumo_emergency_stopping_steps'])
                sumo_emergency_episode = bool(info['sumo_emergency_stopping_episode'])
                hard_braking_steps = int(info['hard_braking_steps'])
                hard_braking_episode = bool(info['hard_braking_episode'])
                counts['Emergency Stopping'] += int(sumo_emergency_episode)
                teleported = bool(info['ego_teleported'])
                if outcome != 'Ego Teleported':
                    counts['Ego Teleported'] += int(teleported)
                if outcome == 'Arrived':
                    counts['Teleported Arrived' if teleported else 'Uninterrupted Arrived'] += 1
                if outcome == 'Timeout':
                    if ANTWERP_TASKS[task][2] == 'exit':
                        counts['Exit Reached Timeout' if info.get('stage_reached')
                               else 'No Exit Timeout'] += 1
                    else:
                        counts['Merged Timeout' if info.get('merged')
                               else 'No Merge Timeout'] += 1
                b0_return = b0_reference_reward(outcome, teleported)
                policy_returns.append(episode_return)
                b0_reference_returns.append(b0_return)
                record = {'seed': seed + episode,
                                'evaluation_seed': seed + episode,
                                'steps': steps, 'outcome': outcome,
                                'merged': bool(info.get('merged')), 'ego_teleported': teleported,
                                'stage_reached': bool(info.get('stage_reached', info.get('merged'))),
                                'stage_reward_paid': bool(info.get('stage_reward_paid', False)),
                                'success': outcome == 'Arrived' and not teleported,
                                'toward_requests': toward_requests,
                                'away_requests': away_requests,
                                'lane_change_commands_sent': commands_sent['total'],
                                'toward_commands_sent': commands_sent['toward'],
                                'away_commands_sent': commands_sent['away'],
                                'same_direction_repeat_requests': same_direction_repeat_requests,
                                'max_same_direction_request_run': max_same_direction_request_run,
                                'short_horizon_request_reversals_1s': short_horizon_request_reversals,
                                'request_frequency_hz': ((toward_requests + away_requests)
                                                         / (steps * 0.1)),
                                'mean_speed_mps': (float(np.mean(speed_samples))
                                                   if speed_samples else None),
                                'policy_reward_return': episode_return,
                                'b0_reference_return': b0_return,
                                'reward_term_totals': dict(reward_term_totals),
                                'sumo_emergency_stopping_steps': sumo_emergency_steps,
                                'sumo_emergency_stopping_episode': sumo_emergency_episode,
                                'hard_braking_threshold_mps2': float(
                                    info['hard_braking_threshold_mps2']),
                                'hard_braking_steps': hard_braking_steps,
                                'hard_braking_episode': hard_braking_episode,
                                'return': episode_return, 'reset_info': reset_info}
                if secrm:
                    record['road_boundary'] = {'sample_counts': dict(boundary_counts),
                        'max_uncovered_area_m2': boundary_max_area,
                        'full_body_exit_status': ('failed' if boundary_counts['outside'] or not record['success'] else 'unknown'),
                        'reason': ('observed_road_boundary_violation' if boundary_counts['outside']
                                   else 'no_route_completion' if not record['success']
                                   else 'terminal_vehicle_removed_or_no_arrival_geometry'),
                        'observed_body_safe_arrival': bool(record['success'] and not boundary_counts['outside']
                                                         and not boundary_counts['unknown'])}
                if task == exit_config.TASK_ID:
                    record['e77_zero_departures'] = e77_zero_departures
                    record['left_e77_zero_after_entry'] = e77_zero_departures > 0
                if directional:
                    record.update(
                        first_direct_exit_lane_remaining_m=first_direct_exit_lane_remaining,
                        lane_change_counts=dict(lane_events),
                        lane_change_direction_reversals=reversals,
                        reward_totals=dict(reward_totals),
                        event_reward_total=sum(reward_totals[name] for name in exit_progress_reward.EVENTS.values()),
                        discounted_return=discounted_return)
                    if secrm:
                        record.update(
                            first_toward_request_distance_to_d3_start_m=first_toward_request_distance,
                            first_stable_e77_zero_distance_to_d3_start_m=first_e77_zero_distance,
                            lane_change_events=lane_event_details,
                            completed_changes_by_edge=dict(Counter(
                                x['before']['road'] for x in lane_event_details)),
                            stopped_seconds_below_0_5_mps=stopped_steps * 0.1,
                            max_stopped_streak_seconds=max_stopped_streak * 0.1,
                            hard_braking_steps_le_minus_4_5_mps2=hard_braking_steps,
                            e77_zero_slow_or_braking_samples=e77_zero_traffic_samples)
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
            'timeout_rate': 100 * counts['Timeout'] / episodes,
            'emergency_stopping_rate': 100 * counts['Emergency Stopping'] / episodes,
            'sumo_emergency_stopping_steps': sum(
                row['sumo_emergency_stopping_steps'] for row in records),
            'sumo_emergency_stopping_episodes': counts['Emergency Stopping'],
            'sumo_emergency_stopping_episode_rate': (
                100 * counts['Emergency Stopping'] / episodes),
            'hard_braking_threshold_mps2': exit_config.HARD_BRAKING_THRESHOLD_MPS2,
            'hard_braking_steps': sum(row['hard_braking_steps'] for row in records),
            'hard_braking_episodes': sum(row['hard_braking_episode'] for row in records),
            'hard_braking_episode_rate': (
                100 * sum(row['hard_braking_episode'] for row in records) / episodes),
            'missed_exit_rate': 100 * counts['Missed Exit'] / episodes,
            'mean_policy_reward_return': float(np.mean(policy_returns)),
            'mean_b0_reference_return': float(np.mean(b0_reference_returns)),
            # Compatibility alias; new comparisons must use the explicit fields above.
            'mean_return': float(np.mean(policy_returns)),
            'mean_return_alias': 'mean_policy_reward_return',
            'episode_results': records}
    if ANTWERP_TASKS[task][2] != 'exit':
        result['no_merge_rate'] = result['timeout_rate']  # historical report field
    else:
        result['toward_requests'] = sum(item['toward_requests'] for item in records)
        result['away_requests'] = sum(item['away_requests'] for item in records)
        result['lane_change_commands_sent'] = sum(
            item['lane_change_commands_sent'] for item in records)
        result['toward_commands_sent'] = sum(item['toward_commands_sent'] for item in records)
        result['away_commands_sent'] = sum(item['away_commands_sent'] for item in records)
        result['same_direction_repeat_requests'] = sum(
            item['same_direction_repeat_requests'] for item in records)
        result['short_horizon_request_reversals_1s'] = sum(
            item['short_horizon_request_reversals_1s'] for item in records)
        result['max_same_direction_request_run'] = max(
            (item['max_same_direction_request_run'] for item in records), default=0)
        result['mean_request_frequency_hz'] = float(np.mean([
            item['request_frequency_hz'] for item in records]))
        valid_speeds = [item['mean_speed_mps'] for item in records
                        if item['mean_speed_mps'] is not None]
        result['mean_speed_mps'] = float(np.mean(valid_speeds)) if valid_speeds else None
        if task == exit_config.TASK_ID:
            result['e77_zero_redeparture_episodes'] = sum(
                item['left_e77_zero_after_entry'] for item in records)
    if directional:
        first_distances = [row['first_direct_exit_lane_remaining_m'] for row in records
                           if row['first_direct_exit_lane_remaining_m'] is not None]
        departures = sum(row['lane_change_counts'].get('left_direct_exit_lane', 0) > 0
                         for row in records)
        result['directional_behavior'] = {
            'first_direct_exit_lane_n': len(first_distances),
            'first_direct_exit_lane_remaining_mean_m': (
                float(np.mean(first_distances)) if first_distances else None),
            'left_direct_exit_lane_episodes': departures,
            'left_direct_exit_lane_episode_rate': 100 * departures / episodes,
            'toward_completed_changes': sum(
                row['lane_change_counts'].get('toward', 0) for row in records),
            'away_completed_changes': sum(
                row['lane_change_counts'].get('away', 0) for row in records),
            'direction_reversals': sum(row['lane_change_direction_reversals'] for row in records),
        }
    if secrm:
        result['boundary_diagnostics'] = {
            'profile': 'sampled_footprint_road_union_v1',
            'outside_episodes': sum(row['road_boundary']['sample_counts'].get('outside', 0) > 0 for row in records),
            'unknown_sample_episodes': sum(row['road_boundary']['sample_counts'].get('unknown', 0) > 0 for row in records),
            'observed_body_safe_arrivals': sum(row['road_boundary']['observed_body_safe_arrival'] for row in records),
            'full_body_exit_qualified': 0,
            'full_body_exit_unknown': sum(row['road_boundary']['full_body_exit_status'] == 'unknown' for row in records),
            'full_body_exit_rate': None,
            'limitation': 'SUMO removes arrived vehicles before a terminal footprint can be observed; sampled containment is not terminal certification.'}
        result['directional_behavior']['first_entry_definition'] = 'settled_full_route_minimum_changes_zero'
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
                  task=exit_config.TASK_ID, urgency_trace_path=None,
                  trajectory_dir=None, trajectory_context=None):
    """Repeat evaluation of ONE checkpoint; this is not training-seed variance."""
    validate_seed_ranges(episodes, seeds)
    setup = validate_checkpoint_setup(model, ablation, safety, task)
    if urgency_trace_path is not None and ANTWERP_TASKS[task][2] != 'exit':
        raise ValueError('Route urgency diagnostics require an exit task')
    network = (load_route_network(ANTWERP_DIR / 'Antwerp.net.xml')
               if urgency_trace_path is not None or trajectory_dir is not None else None)
    recorder = None
    if trajectory_dir is not None:
        context = dict(trajectory_context or {})
        recorder = TrajectoryRecorder(
            trajectory_dir, model_label=context.get('model_label'),
            checkpoint=context.get('checkpoint'),
            checkpoint_sha256=context.get('checkpoint_sha256'),
            training_seed=getattr(model, 'seed', None), scenario=scenario,
            reward_config=setup['reward_config'], route_network=network,
            lane_change_edges=LANE_CHANGE_EDGES, parse_action=parseAction,
            destination=exit_config.DESTINATION,
            lane_change_seconds=exit_config.LANE_CHANGE_TIME_SECONDS,
            step_length=exit_config.SUMO_STEP_LENGTH,
            hard_braking_threshold=exit_config.HARD_BRAKING_THRESHOLD_MPS2,
            run_label=context.get('run_label'),
            config={**context.get('evaluation_config', {}), 'delay_mode': delay_mode,
                    'evaluation_safety': bool(safety), 'ablation': ablation,
                    'episodes_per_seed': episodes, 'evaluation_seed_starts': list(seeds),
                    'deterministic': True, 'task': task,
                    'training_safety': setup.get('training_safety'),
                    'observation_mode': setup['observation_mode']})
    status = 'error'
    runs = []
    with ExitStack() as stack:
        if recorder is not None:
            stack.callback(lambda: recorder.close(status))  # also runs on exceptions
        urgency_trace = None
        if urgency_trace_path is not None:
            urgency_trace_path = Path(urgency_trace_path)
            urgency_trace_path.parent.mkdir(parents=True, exist_ok=True)
            urgency_trace = stack.enter_context(urgency_trace_path.open('x'))
        for seed in seeds:
            env = make_env(seed, scenario, delay_mode, safety, ablation, task=task,
                           potential_gamma=setup.get('potential_gamma'),
                           nav_features=setup.get('nav_features', False),
                           exit_reward_profile=setup.get('exit_reward_profile'),
                           progress_experiment=setup.get('progress_experiment'),
                           reward_config=setup['reward_config'],
                           reward_gamma=setup['reward_gamma'],
                           observation_mode=setup['observation_mode'])
            try:
                runs.append({'seed': seed, 'evaluation_seed_start': seed,
                             'episodes': episodes,
                             **evaluate(model, env, episodes, seed, task=task,
                                        urgency_trace=urgency_trace, route_network=network,
                                        recorder=recorder)})
            finally:
                env.close()
        status = 'ok'
    report = {'protocol': 'antwerp-v2-evaluation-v1', 'scenario': scenario,
            'task': task, 'stage_name': ANTWERP_TASKS[task][2],
            'delay_mode': delay_mode, 'ablation': ablation, **setup,
            'gru_input_mode': checkpoint_gru_input_mode(model),
            'antwerp_env_version': ANTWERP_ENV_VERSION,
            'training_seed': getattr(model, 'seed', None),
            'secrm_formula_version': getattr(model, 'secrm_formula_version', None),
            'secrm_effective_vehicle_parameters': getattr(model, 'secrm_effective_vehicle_parameters', None),
            'training_timesteps': getattr(model, 'num_timesteps', None),
            'deterministic': True,
            'evaluation_seed_starts': list(seeds),
            'evaluation_seeds': list(seeds),  # compatibility alias
            'episodes_per_seed': episodes, 'total_episodes': episodes * len(seeds),
            'std_ddof': 1, 'ci_method': '1.96 * sample_std / sqrt(evaluation_runs)',
            'aggregation_unit': 'evaluation_run_of_one_checkpoint',
            'return_definitions': {
                'policy_reward_return': ('Undiscounted environment return computed with the '
                                         'checkpoint-authenticated reward configuration.'),
                'b0_reference_return': ('Common terminal-event reference: Arrived without '
                                        'collision/teleport +50; Collision, Ego Teleported, '
                                        'Missed Exit, or Timeout -100; all other steps 0.'),
            },
            'runs': runs, 'aggregate': summarize_runs(
                runs, tuple(key for key in REPORT_METRICS
                            if ANTWERP_TASKS[task][2] != 'exit' or key != 'no_merge_rate'))}
    if task == exit_config.TASK_ID:
        report['e77_zero_redeparture_episodes'] = sum(
            run['e77_zero_redeparture_episodes'] for run in runs)
    if trajectory_dir is not None:
        report['trajectory_dir'] = str(trajectory_dir)
    if urgency_trace_path is not None:
        report['route_urgency_trace'] = str(urgency_trace_path)
        report['route_urgency_profile'] = 'lane_graph_minimax_v1'
        report['diagnostic_reward_profile'] = 'environment_reward_minus_potential_v1'
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Antwerp SAC checkpoint (.zip)')
    parser.add_argument('--scenario', choices=('easy', 'medium', 'hard'),
                        default=exit_config.DEFAULT_SCENARIO)
    parser.add_argument('--task', choices=(exit_config.TASK_ID,), default=exit_config.TASK_ID)
    parser.add_argument('--delay-mode', choices=('uniform', 'exponential', 'triangular', 'bursty', 'bimodal'), default='uniform')
    parser.add_argument('--episodes', type=int, default=50)
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument('--seed', type=int, default=200001,
                              help='First evaluation-episode seed; distinct from training seed')
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
    parser.add_argument('--trajectory-dir', type=Path,
                        help='Enable full per-step CSV trajectory recording into this NEW directory')
    parser.add_argument('--model-label', help='Label stored in trajectory files (default: checkpoint stem)')
    parser.add_argument('--run-label', help='Experiment id stored in the trajectory manifest')
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
    if args.trajectory_dir and args.trajectory_dir.exists():
        parser.error(f'Trajectory directory already exists: {args.trajectory_dir}')
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
                           urgency_trace_path=args.urgency_trace,
                           trajectory_dir=args.trajectory_dir,
                           trajectory_context={
                               'model_label': args.model_label or checkpoint.stem,
                               'checkpoint': str(checkpoint),
                               'checkpoint_sha256': checkpoint_hash,
                               'run_label': args.run_label,
                               'evaluation_config': {
                                   'argv_scenario': args.scenario, 'device': args.device,
                                   'torch_threads': args.torch_threads,
                                   'seeds': list(args.seeds), 'episodes': args.episodes}})
    report.update(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                  device=args.device, torch_threads=args.torch_threads)
    print(f"{args.scenario} | {args.delay_mode} | task={args.task} | ablation={args.ablation} | "
          f"training_safety={report['training_safety']} | evaluation_safety={args.safety}")
    for run in report['runs']:
        print(f"Seed {run['seed']} | episodes={args.episodes}")
        timeout_details = (EXIT_TIMEOUT_DETAILS if ANTWERP_TASKS[args.task][2] == 'exit'
                           else TIMEOUT_DETAILS)
        for outcome in dict.fromkeys((*OUTCOMES, *timeout_details, *ARRIVAL_DETAILS)):
            print(f"  {outcome}: {run[outcome]}/{args.episodes}")
    print('Across evaluation runs of this checkpoint (std and upstream-style CI are separate):')
    for key, values in report['aggregate'].items():
        if values['std'] is None:
            print(f"  {key}: mean={values['mean']:.4f}; std/CI unavailable with one run")
        else:
            print(f"  {key}: mean={values['mean']:.4f}, std={values['std']:.4f}, "
                  f"CI95 half-width={values['ci95_half_width']:.4f}")
    print('success_rate is SUMO route completion; timeout_rate counts terminal '
          'Timeout events. Road-boundary and emergency-stop values remain diagnostics.')
    print('policy_reward_return uses the checkpoint-authenticated reward configuration; '
          'b0_reference_return uses the common B0 terminal-event reference only.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
        print(f'Saved evaluation report to {args.output}')


if __name__ == '__main__':
    main()
