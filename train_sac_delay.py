# from delay_encoder import DelayAwareGRUEncoder
import os
import sys
import time
import argparse
import json
import hashlib
import subprocess
from pathlib import Path
from collections import deque
import torch
from typing import Optional

import numpy as np

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from src.delay_encoder import DelayAwareEncoder, DelayAwareGRUEncoder, TinyTransformerEncoder

import wandb
from wandb.integration.sb3 import WandbCallback

from src.merging import ANTWERP_ENV_VERSION, Merging
from src import exit_e76_d3_config as exit_config
from src import exit_progress_reward
from src.rewards.config import load_hydra_config, load_reward_file
from src.wrapper import ABLATION_MODES, DelayWrapper, ExitStartLaneCurriculum

class IntermediateCheckpointCallback(CheckpointCallback):
    """Skip a periodic save at the terminal step, saved once after learn()."""
    def __init__(self, final_step, **kwargs):
        super().__init__(**kwargs)
        self.final_step = final_step

    def _on_step(self):
        if self.num_timesteps >= self.final_step:
            return True
        return super()._on_step()


class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0, log_path=None):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_potential_reward = 0.0
        self.current_urgency_reward = 0.0
        self.current_progress_reward = 0.0
        self.current_event_reward = 0.0
        self.current_lane_change_reward = 0.0
        self.current_efficiency_reward = 0.0
        self.current_route_reward = 0.0
        self.lane_episodes = [0] * 5
        self.lane_successes = [0] * 5
        self.recent_outcomes = deque(maxlen=20)
        self.toward_requests = 0
        self.away_requests = 0
        self.completed_episodes = 0
        self.sumo_emergency_stopping_steps_total = 0
        self.sumo_emergency_stopping_episodes = 0
        self.hard_braking_steps_total = 0
        self.hard_braking_episodes = 0
        self.started = time.monotonic()
        self.starting_timesteps = 0
        self.last_logged_env_steps = -1
        self.log_file = None
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self.log_file = open(log_path, 'x')

    def _write(self, record):
        if self.log_file:
            self.log_file.write(json.dumps(record) + '\n')
            self.log_file.flush()

    def _log_metrics(self, metrics):
        """Log every W&B series against the same monotonic environment-step axis."""
        metrics = dict(metrics)
        step = int(metrics.setdefault('env_steps', self.num_timesteps))
        if step < self.last_logged_env_steps:
            raise RuntimeError('env_steps must be globally monotonic')
        self.last_logged_env_steps = step
        wandb.log(metrics)

    def _on_training_end(self):
        if self.log_file:
            self.log_file.close()

    def _on_step(self) -> bool:
        step_info = self.locals['infos'][0]
        direction = step_info.get('exit_actor_request_direction', 0)
        self.toward_requests += int(direction == -1)
        self.away_requests += int(direction == 1)
        if False:
            components = step_info['reward_components']
            self.current_efficiency_reward += components.get('efficiency', 0.0)
            self.current_route_reward += components.get('route_following', 0.0)
            if step_info.get('secrm_diagnostic') is not None:
                target = step_info['secrm_diagnostic']['target']
                effective = {key: target[key] for key in ('ego_decel', 'leader_decel',
                    'effective_decel', 'stopped_gap', 'reaction_seconds')}
                effective['ego_braking'] = target['ego_braking']
                effective['leader_braking'] = target['leader_selection'].get('braking')
                seen = getattr(self.model, 'secrm_effective_vehicle_parameters', [])
                if effective not in seen:
                    self.model.secrm_effective_vehicle_parameters = seen + [effective]
                self._write({'event': 'secrm', 'steps': self.num_timesteps,
                             'diagnostic': step_info['secrm_diagnostic'],
                             'reward_components': components,
                             'road_boundary': step_info.get('road_boundary')})
            self.current_progress_reward += components['progress']
            self.current_lane_change_reward += components['lane_change']
            self.current_event_reward += sum(
                components[name] for name in exit_progress_reward.EVENTS.values())
            event = step_info.get('directional_lane_change_event')
            if event is not None:
                self._write({'event': 'directional_lane_change',
                             'steps': self.num_timesteps, 'transition': event,
                             'reward_components': components})
        # Track episode reward and length
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_potential_reward += self.locals['infos'][0].get('potential_reward', 0.0)
        self.current_urgency_reward += self.locals['infos'][0].get('urgency_reward', 0.0)
        self.current_episode_length += 1

        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            info = self.locals['infos'][0]
            outcome = info.get('message')
            merged = bool(info.get('merged'))
            stage_reached = bool(info.get('stage_reached', merged))
            exit_task = info.get('stage_name') == 'exit'
            self.completed_episodes += 1
            sumo_emergency_steps = int(info.get('sumo_emergency_stopping_steps', 0))
            sumo_emergency_episode = bool(info.get('sumo_emergency_stopping_episode', False))
            hard_braking_steps = int(info.get('hard_braking_steps', 0))
            hard_braking_episode = bool(info.get('hard_braking_episode', False))
            self.sumo_emergency_stopping_steps_total += sumo_emergency_steps
            self.sumo_emergency_stopping_episodes += int(sumo_emergency_episode)
            self.hard_braking_steps_total += hard_braking_steps
            self.hard_braking_episodes += int(hard_braking_episode)
            if exit_task:
                self.recent_outcomes.append((int(bool(info['is_success'])),
                                             int(outcome == 'Missed Exit'),
                                             int(outcome == 'Collision')))
            
            # Log to wandb
            metrics = {
                'train/episode_reward': self.current_episode_reward,
                'train/episode_length': self.current_episode_length,
                'train/mean_episode_reward': float(np.mean(self.episode_rewards)),
                'train/mean_episode_length': float(np.mean(self.episode_lengths)),
                'train/arrival': int(outcome == 'Arrived'),
                'train/success': int(info['is_success']),
                'train/ego_teleported': int(info['ego_teleported']),
                'train/teleported_arrival': int(outcome == 'Arrived' and info['ego_teleported']),
                'train/collision': int(outcome == 'Collision'),
                'train/emergency_stopping': int(sumo_emergency_episode),
                'train/sumo_emergency_stopping_steps_episode': sumo_emergency_steps,
                'train/sumo_emergency_stopping_steps_total': (
                    self.sumo_emergency_stopping_steps_total),
                'train/sumo_emergency_stopping_episodes_total': (
                    self.sumo_emergency_stopping_episodes),
                'train/sumo_emergency_stopping_episode_rate': (
                    self.sumo_emergency_stopping_episodes / self.completed_episodes),
                'train/hard_braking_threshold_mps2': float(
                    info.get('hard_braking_threshold_mps2',
                             exit_config.HARD_BRAKING_THRESHOLD_MPS2)),
                'train/hard_braking_steps_episode': hard_braking_steps,
                'train/hard_braking_steps_total': self.hard_braking_steps_total,
                'train/hard_braking_episodes_total': self.hard_braking_episodes,
                'train/hard_braking_episode_rate': (
                    self.hard_braking_episodes / self.completed_episodes),
                'train/missed_exit': int(outcome == 'Missed Exit'),
                'train/teleport_failure': int(outcome == 'Ego Teleported'),
                'train/no_stage_timeout': int(outcome == 'Timeout' and not stage_reached),
                'train/stage_reached_timeout': int(outcome == 'Timeout' and stage_reached),
                'train/stage_reached': int(stage_reached),
                'train/stage_reward_paid': int(info.get('stage_reward_paid', False)),
                'env_steps': self.num_timesteps,
            }
            if not exit_task:
                metrics['train/no_merge_timeout'] = int(outcome == 'Timeout' and not merged)
                metrics['train/merged_timeout'] = int(outcome == 'Timeout' and merged)
            else:
                recent = np.asarray(self.recent_outcomes, dtype=np.float32)
                metrics.update({
                    'train/recent20_episodes': len(self.recent_outcomes),
                    'train/recent20_strict_success_rate': float(recent[:, 0].mean()),
                    'train/recent20_missed_exit_rate': float(recent[:, 1].mean()),
                    'train/recent20_collision_rate': float(recent[:, 2].mean()),
                })
            if info.get('ego_start_lane') is not None:
                start_lane = int(info['ego_start_lane'])
                metrics['train/ego_start_lane'] = start_lane
                metrics['train/initial_potential'] = float(info['initial_potential'])
                metrics['train/episode_potential_reward'] = float(self.current_potential_reward)
                if False:
                    metrics['train/episode_urgency_reward'] = float(self.current_urgency_reward)
                if False:
                    metrics['train/episode_efficiency_reward'] = float(self.current_efficiency_reward)
                    metrics['train/episode_route_reward'] = float(self.current_route_reward)
                    metrics['train/episode_progress_reward'] = float(self.current_progress_reward)
                    metrics['train/episode_event_reward'] = float(self.current_event_reward)
                    metrics['train/episode_lane_change_reward'] = float(self.current_lane_change_reward)
                if 0 <= start_lane < 5:
                    self.lane_episodes[start_lane] += 1
                    self.lane_successes[start_lane] += int(info['is_success'])
                    metrics[f'train/lane_{start_lane}_episodes'] = self.lane_episodes[start_lane]
                    metrics[f'train/lane_{start_lane}_success_rate'] = (
                        self.lane_successes[start_lane] / self.lane_episodes[start_lane])
            if info.get('curriculum_stage') is not None:
                metrics['train/curriculum_stage'] = int(info['curriculum_stage'])
                metrics['train/curriculum_reset_step'] = int(info['curriculum_reset_step'])
            ledger = info.get('reward_episode_summary') or {}
            for term, stats in ledger.items():
                if term == '_meta':
                    metrics['reward/total/weighted_sum'] = stats['weighted_total']
                    continue
                metric_names = {
                    'raw_sum': 'raw', 'weighted_sum': 'wsum',
                    'discounted_sum': 'disc', 'active_ratio': 'active',
                    'abs_p95': 'p95', 'abs_p99': 'p99',
                    'nonzero_abs_p95': 'nz_p95',
                    'nonzero_abs_p99': 'nz_p99', 'max_abs': 'max',
                    'clipped_ratio': 'clip',
                    # Keep SB3's 36-character key truncation from merging p95/p99.
                    'effective_weight_p95': 'eff_p95',
                    'effective_weight_p99': 'eff_p99',
                    'configured_weight': 'cfg_w',
                }
                for name, metric_name in metric_names.items():
                    metric = f'reward/episode/{term}/{metric_name}'
                    value = float(stats[name])
                    metrics[metric] = value
                    self.logger.record(metric, value)
            if '_meta' in ledger:
                self.logger.record('reward/total/weighted_sum',
                                   float(ledger['_meta']['weighted_total']))
                total_discounted = float(sum(stats['discounted_sum'] for name, stats in ledger.items()
                                             if name != '_meta'))
                metrics['reward/total/discounted_sum'] = total_discounted
                self.logger.record('reward/total/discounted_sum', total_discounted)
            self._log_metrics(metrics)
            self._write({'event': 'episode', 'steps': self.num_timesteps,
                         'outcome': outcome, **{k: float(v) for k, v in metrics.items()}})
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_potential_reward = 0.0
            self.current_urgency_reward = 0.0
            self.current_efficiency_reward = 0.0
            self.current_route_reward = 0.0
            self.current_progress_reward = 0.0
            self.current_event_reward = 0.0
            self.current_lane_change_reward = 0.0

        if self.num_timesteps % 100 == 0 and self.locals['infos'][0].get('ego_start_lane') is not None:
            latest = self.model.logger.name_to_value
            scalar_names = ('train/actor_loss', 'train/critic_loss',
                            'train/ent_coef', 'train/ent_coef_loss')
            scalars = {name: float(latest[name]) for name in scalar_names
                       if name in latest and np.isfinite(latest[name])}
            if scalars:
                record = {'env_steps': self.num_timesteps, **scalars}
                self._log_metrics(record)
                self._write({'event': 'scalars', 'steps': self.num_timesteps, **record})

        if self.num_timesteps % 1000 == 0:
            if self.locals['infos'][0].get('stage_name') == 'exit':
                total_requests = self.toward_requests + self.away_requests
                request_metrics = {
                    'env_steps': self.num_timesteps,
                    'train/toward_request_count_1k': self.toward_requests,
                    'train/away_request_count_1k': self.away_requests,
                }
                if total_requests:
                    request_metrics['train/away_request_fraction_1k'] = (
                        self.away_requests / total_requests)
                self._log_metrics(request_metrics)
                self._write({'event': 'request_window', 'steps': self.num_timesteps,
                             **request_metrics})
                self.toward_requests = self.away_requests = 0
            elapsed = time.monotonic() - self.started
            collected_steps = self.num_timesteps - self.starting_timesteps
            self._write({'event': 'progress', 'steps': self.num_timesteps,
                         'updates': int(self.model._n_updates), 'elapsed_seconds': elapsed,
                         'fps': collected_steps / max(elapsed, 1e-6)})
            print(f'Training step {self.num_timesteps}, {collected_steps / max(elapsed, 1e-6):.1f} steps/s', flush=True)
        return True

class StepLoggingCallback(EpisodeRewardCallback):
    def __init__(self):
        super().__init__()
        self.step_num = 0

    def _on_step(self) -> bool:
        # infos = self.locals.get("infos", [])
        # observations = self.locals.get("new_obs") if "new_obs" in self.locals else self.locals.get("obs")
        # rewards = self.locals.get("rewards")
        # dones = self.locals.get("dones")
        # Print observations, rewards, dones, info per step to console ONLY (no wandb.log)
        # if observations is not None and rewards is not None:
            # batch_size = len(rewards) if hasattr(rewards, '__len__') and not isinstance(rewards, str) else 1
            # for i in range(batch_size):
            #     obs = observations[i] if hasattr(observations, '__getitem__') else observations
            #     rew = rewards[i] if hasattr(rewards, '__getitem__') else rewards
            #     dn = dones[i] if hasattr(dones, '__getitem__') else dones
            #     info = infos[i] if isinstance(infos, (list, tuple)) and len(infos) > 0 else infos
                # print("Step:", self.step_num)
                # print("  Obs:", obs)
                # print("  Reward:", float(rew))
                # print("  Done:", bool(dn))
                # if info is not None and isinstance(info, dict) and len(info) > 0:
                #     print("  Info:")
                #     for k, v in info.items():
                #         print(f"    {k}: {v}")
                # self.step_num += 1
        pass
        # return super()._on_step()

def save_model(model: SAC, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    return path


def read_git_provenance():
    """Return commit and porcelain state without reading file contents."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(
            ['git', 'status', '--porcelain=v1', '--untracked-files=all'],
            text=True).rstrip('\n')
    except (OSError, subprocess.CalledProcessError):
        return {'git_commit': None, 'git_dirty': None,
                'git_status_porcelain': None}
    return {'git_commit': commit, 'git_dirty': bool(status),
            'git_status_porcelain': status}

def make_env(seed, scenario, delay_mode, max_delay, use_safety, render_mode=None,
             ablation='full', task=exit_config.TASK_ID, potential_gamma=None,
             nav_features=False, exit_reward_profile=None,
             curriculum_stages=None, curriculum_route_dir=None,
             progress_experiment=None, reward_config=None, reward_gamma=0.99):
    if use_safety:
        raise ValueError('E76-to-D3 training requires Safety off; SafetyWrapper is forbidden')
    env = DelayWrapper(
        Merging(seed=seed, scenario=scenario, render_mode=render_mode, task=task,
                potential_gamma=potential_gamma, exit_reward_profile=exit_reward_profile,
                progress_experiment=progress_experiment,
                reward_config=reward_config, reward_gamma=reward_gamma),
        max_delay=max_delay, mode='all', delay_mode=delay_mode, seed=seed,
        ablation=ablation, nav_features=nav_features,
    )
    if curriculum_stages is not None:
        env = ExitStartLaneCurriculum(env, curriculum_stages, seed,
                                      curriculum_route_dir)
    return env


def configure_wandb_metrics(run):
    """Bind all manually logged training series to the environment-step axis."""
    run.define_metric('env_steps')
    for namespace in ('train/*', 'reward/*', 'diagnostic/*'):
        run.define_metric(namespace, step_metric='env_steps')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, choices=['must', 'allow', 'never'])
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--radius', type=int, default=50)
    parser.add_argument('--total_timesteps', type=int, default=exit_config.DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument('--checkpoint_freq', type=int, default=exit_config.DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument('--save_replay_buffer', action='store_true',
                        help='Save replay buffers alongside periodic and final models')
    parser.add_argument('--resume-model', default=None,
                        help='SAC checkpoint to resume with its matching Replay Buffer')
    parser.add_argument('--resume-replay-buffer', default=None,
                        help='Replay Buffer paired with --resume-model')
    parser.add_argument('--target-total-timesteps', type=int, default=None,
                        help='Absolute target when resuming; training continues from the loaded model step')
    parser.add_argument('--torch_threads', type=int, default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--log_dir', default='runs')
    parser.add_argument('--episode_log', default=None)
    parser.add_argument('--render_mode', type=str, default=None)
    parser.add_argument('--seed', type=int, default=exit_config.DEFAULT_SEED)
    parser.add_argument('--gamma', type=float, default=exit_config.DEFAULT_GAMMA)
    parser.add_argument('--scenario', choices=['easy', 'medium', 'hard'], default=exit_config.DEFAULT_SCENARIO)
    parser.add_argument('--task', choices=(exit_config.TASK_ID,), default=exit_config.TASK_ID,
                        help='The single supported task: E76 to D3')
    parser.add_argument('--reward-config', default=None,
                        help='Explicit YAML/JSON reward file; otherwise use Hydra conf/')
    parser.add_argument('--config-path', default='conf')
    parser.add_argument('--config-name', default='config')
    parser.add_argument('--hydra-override', action='append', default=[],
                        help='Hydra override, e.g. reward.terms.collision.enabled=false')
    parser.add_argument('--safety', dest='use_safety', action='store_true')
    parser.add_argument('--save_dir', type=str, default='models_experiments')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--encoder_mode', type=str, default='GRU', choices=['Transformer', 'GRU', 'MLP', 'None'])
    parser.add_argument('--gru_input_mode', choices=DelayAwareGRUEncoder.INPUT_MODES,
                        default='paper_raw',
                        help='New GRU runs use unscaled wrapper inputs; the Antwerp legacy mode is for old checkpoints')
    parser.add_argument('--ablation', choices=ABLATION_MODES, default='full')
    parser.add_argument('--delay_mode', type=str, default='uniform', choices=['uniform', 'exponential', 'triangular', 'bursty', 'bimodal'])
    args = parser.parse_args()
    if args.use_safety:
        parser.error('Current E76-to-D3 experiments require Safety off; --safety is forbidden')
    try:
        if args.reward_config:
            args.reward_config, args.reward_config_hash, args.reward_config_yaml = load_reward_file(
                args.reward_config)
        else:
            composed, args.reward_config, args.reward_config_hash, args.reward_config_yaml = load_hydra_config(
                args.config_path, args.config_name, args.hydra_override)
    except (OSError, ValueError) as error:
        parser.error(f'Invalid Hydra reward configuration: {error}')
    args.nav_features = False
    args.exit_reward_profile = args.reward_config['name']
    args.progress_experiment = None
    args.exit_phase = None
    args.project = args.project or exit_config.WANDB_PROJECT
    args.entity = args.entity or exit_config.WANDB_ENTITY
    if args.ablation != 'full' and args.encoder_mode != 'GRU':
        parser.error('Table IV ablations require --encoder_mode GRU')
    if args.encoder_mode != 'GRU' and args.gru_input_mode != 'paper_raw':
        parser.error('--gru_input_mode only applies to --encoder_mode GRU')
    if args.checkpoint_freq < 1 or (args.torch_threads is not None and args.torch_threads < 1):
        parser.error('checkpoint_freq and torch_threads must be positive')
    resuming = bool(args.resume_model)
    if resuming != bool(args.resume_replay_buffer):
        parser.error('--resume-model and --resume-replay-buffer must be supplied together')
    if args.target_total_timesteps is not None and not resuming:
        parser.error('--target-total-timesteps requires --resume-model and --resume-replay-buffer')
    if resuming and (args.target_total_timesteps is None or args.target_total_timesteps < 1):
        parser.error('A positive --target-total-timesteps is required when resuming')
    return args

def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    args = parse_args()
    if args.run_name is None:
        args.run_name = f"SAC-merging-delay-wrapper-{time.strftime('%Y%m%d-%H%M%S')}"

    model_path = os.path.join(args.save_dir, f"{args.run_name}.zip")
    checkpoint_dir = os.path.join(args.save_dir, args.run_name)
    tensorboard_dir = os.path.join(args.log_dir, args.run_name)
    for path in (model_path, checkpoint_dir, tensorboard_dir):
        if os.path.exists(path):
            raise FileExistsError(f'Run output already exists: {path}; choose a new --run_name')
    if args.episode_log and (os.path.exists(args.episode_log)
                             or os.path.exists(args.episode_log + '.config.json')):
        raise FileExistsError('Run JSONL or configuration manifest already exists')

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    ensure_dirs(args.log_dir)

    max_delay = 20
    git_provenance = read_git_provenance()
    sac_parameters = dict(
        learning_rate=3e-5, batch_size=512, buffer_size=1_000_000,
        learning_starts=100, gamma=args.gamma, tau=0.005, train_freq=1,
        gradient_steps=1, target_update_interval=1, ent_coef='auto',
        target_entropy='auto', use_sde=False,
    )
    config = dict(
        algo='SAC',
        **git_provenance,
        radius=args.radius,
        seed=args.seed,
        training_seed=args.seed,
        total_timesteps=args.total_timesteps,
        policy='MlpPolicy',
        encoder_mode=args.encoder_mode,
        gru_input_mode=args.gru_input_mode if args.encoder_mode == 'GRU' else None,
        max_delay=max_delay,
        gru_hidden=64 if args.encoder_mode == 'GRU' else None,
        features_dim=256 if args.encoder_mode == 'GRU' else None,
        net_arch=[256, 256],
        sac=sac_parameters,
        ablation=args.ablation,
        delay_mode=args.delay_mode,
        scenario=args.scenario,
        task=args.task,
        exit_phase=args.exit_phase,
        ego_depart_lane=exit_config.DEPART_LANE,
        reward_config=args.reward_config,
        reward_config_sha256=args.reward_config_hash,
        reward_config_yaml=args.reward_config_yaml,
        checkpoint_freq=args.checkpoint_freq,
        gamma=args.gamma,
        ego_depart_time=exit_config.EGO_DEPART_TIME,
        timeout_steps=exit_config.TIMEOUT_STEPS,
        road_speed_limit_mps=exit_config.ROAD_SPEED_LIMIT_MPS,
        reward_profile=args.exit_reward_profile,
        progress_experiment=None,
        exit_control_profile=None,
        exit_observation_profile=None,
        observation_dim=(136 if args.nav_features else 134),
        nav_distance_scale_meters=None,
        metric_schema='env_steps_v3',
        wandb_project=args.project,
        wandb_entity=args.entity,
        use_safety=False,
        training_safety=False,
        antwerp_env_version=ANTWERP_ENV_VERSION,
        device=args.device,
        torch_threads=args.torch_threads,
        save_replay_buffer=args.save_replay_buffer,
        resume_model=args.resume_model,
        resume_replay_buffer=args.resume_replay_buffer,
        target_total_timesteps=args.target_total_timesteps,
        curriculum_stages=None,
    )
    config['network_sha256'] = hashlib.sha256(
        Path('sumo_new/Antwerp.net.xml').read_bytes()).hexdigest()
    config['experiment_identity'] = args.run_name
    config_bytes = json.dumps(config, sort_keys=True, separators=(',', ':')).encode('utf-8')
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if args.episode_log:
        ensure_dirs(os.path.dirname(os.path.abspath(args.episode_log)))
        with open(args.episode_log + '.config.json', 'x') as manifest:
            json.dump({'sha256': config_sha256, 'config': config}, manifest,
                      sort_keys=True, indent=2)
            manifest.write('\n')
        with open(args.episode_log + '.reward.yaml', 'x') as reward_manifest:
            reward_manifest.write(args.reward_config_yaml)
            reward_manifest.write('\n')
    print(json.dumps(config, sort_keys=True, indent=2), flush=True)
    print(f'Experiment config SHA-256: {config_sha256}', flush=True)
    # Create env
    def _fn():
        return make_env(args.seed, args.scenario, args.delay_mode, max_delay,
                        args.use_safety, args.render_mode, args.ablation, args.task,
                        args.gamma if args.exit_phase else None, args.nav_features,
                        args.exit_reward_profile,
                        config['curriculum_stages'],
                        (os.path.join('tmp', 'ai', 'e76-d3', args.exit_phase,
                                      args.run_name, 'routes')
                         if config['curriculum_stages'] is not None else None),
                        args.progress_experiment, args.reward_config, args.gamma)

    # SB3 expects the Gym API; our env already implements gymnasium API. SB3 v2 works with gymnasium.
    vec_env = DummyVecEnv([_fn])

    wrapped_env = vec_env.envs[0]
    if config['curriculum_stages'] is not None:
        wrapped_env = wrapped_env.env
    delay_env = wrapped_env
    entity_seq_len, entity_feat_dim = delay_env.orig_shape
    act_dim = delay_env.action_space.shape[0]
    encoder_mode = args.encoder_mode
    if encoder_mode == 'Transformer':
        policy_kwargs = dict(
        features_extractor_class=TinyTransformerEncoder,
        features_extractor_kwargs=dict(
            entity_seq_len=entity_seq_len,
            entity_feat_dim=entity_feat_dim,
            act_hist_len=delay_env.max_delay,
            act_dim=act_dim,
            delay_dim=1,
            d_model=32,
            n_heads=1,
            ffn_hidden=64,
            n_layers=1,
            features_dim=128,
        ),
        net_arch=[256, 256],      # actor/critic MLP after transformer
    )
    elif encoder_mode == 'GRU':
        policy_kwargs = dict(features_extractor_class=DelayAwareGRUEncoder, features_extractor_kwargs=dict(act_dim=act_dim, hist_len=delay_env.max_delay, delay_dim=1, features_dim=256, gru_hidden=64, input_mode=args.gru_input_mode, nav_dim=delay_env.nav_dim), net_arch=[256, 256])
    elif encoder_mode == 'MLP':
        policy_kwargs = dict(
        features_extractor_class=DelayAwareEncoder,
        features_extractor_kwargs=dict(act_hist_dim=max_delay * act_dim, delay_dim=1)
        )
    else:
        policy_kwargs = dict()

    if args.resume_model:
        model = SAC.load(args.resume_model, env=vec_env, device=args.device)
        if getattr(model, 'exit_reward_config_sha256', None) != args.reward_config_hash:
            raise ValueError('Resume checkpoint reward configuration hash mismatch')
        if getattr(model, 'antwerp_task', None) != args.task:
            raise ValueError('Resume checkpoint task mismatch')
        model.load_replay_buffer(args.resume_replay_buffer)
        resumed_from_timesteps = int(model.num_timesteps)
        if args.target_total_timesteps <= resumed_from_timesteps:
            raise ValueError('target-total-timesteps must exceed the resumed model step count')
        learn_timesteps = args.target_total_timesteps - resumed_from_timesteps
        model.continuation_source_model = os.path.abspath(args.resume_model)
        model.continuation_source_replay_buffer = os.path.abspath(args.resume_replay_buffer)
        model.continuation_start_timesteps = resumed_from_timesteps
        model.continuation_target_timesteps = int(args.target_total_timesteps)
    else:
        model = SAC('MlpPolicy', vec_env, seed=args.seed, verbose=1,
                    tensorboard_log=os.path.join(args.log_dir, args.run_name),
                    device=args.device, policy_kwargs=policy_kwargs, **sac_parameters)
        resumed_from_timesteps = 0
        learn_timesteps = args.total_timesteps

    model.ablation = args.ablation
    model.antwerp_env_version = ANTWERP_ENV_VERSION
    model.antwerp_task = args.task
    model.training_safety = False
    model.gru_input_mode = args.gru_input_mode if args.encoder_mode == 'GRU' else None
    model.experiment_id = args.run_name
    model.experiment_config_sha256 = config_sha256
    model.exit_reward_profile = args.exit_reward_profile
    model.exit_reward_config = args.reward_config
    model.exit_reward_config_sha256 = config['reward_config_sha256']
    model.exit_potential_beta = 0.0
    model.exit_potential_gamma = args.gamma
    if args.exit_phase:
        model.exit_observation_profile = (exit_config.NAV_OBSERVATION_PROFILE
                                          if args.nav_features else None)
        model.exit_potential_beta = exit_config.potential_beta_for_profile(args.exit_reward_profile)
        model.exit_potential_gamma = args.gamma
        model.exit_reward_profile = args.exit_reward_profile
        if False:
            model.secrm_formula_version = exit_config.secrm_reward.FORMULA_VERSION
            exit_config.secrm_reward.stamp_replay(model.replay_buffer, args.progress_experiment)
            if not hasattr(model, 'secrm_effective_vehicle_parameters'):
                model.secrm_effective_vehicle_parameters = []
        if args.progress_experiment is not None:
            model.exit_progress_experiment = args.progress_experiment
            model.exit_progress_reward_identity = exit_progress_reward.reward_identity(
                args.progress_experiment['reward'])
            model.exit_progress_experiment_identity = exit_progress_reward.experiment_identity(
                args.progress_experiment)
        model.exit_control_profile = exit_config.CONTROL_PROFILE

    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name, config=config,
                     mode=args.wandb_mode, dir=args.log_dir,
                     sync_tensorboard=False,
                     save_code=not args.exit_phase, resume=args.resume, id=args.run_id,
                     group=(f'e76-d3-{args.exit_phase}-{args.exit_reward_profile}'
                            f'-{exit_config.CONTROL_PROFILE}')
                     if args.exit_phase else None,
                     job_type=args.exit_phase,
                     tags=['E76-D3', 'single-ego', 'metric-schema-v3',
                           args.exit_reward_profile, exit_config.CONTROL_PROFILE]
                     if args.exit_phase else None)
    configure_wandb_metrics(run)
    if not args.exit_phase:
        run.log_code('src')

    ensure_dirs(args.save_dir)

    # Create callbacks
    episode_callback = EpisodeRewardCallback(log_path=args.episode_log)
    episode_callback.starting_timesteps = resumed_from_timesteps
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq, save_path=checkpoint_dir,
        name_prefix=args.run_name, verbose=2,
        save_replay_buffer=args.save_replay_buffer,
    )
    if (args.resume_model or args.exit_phase in ('secrm_efficiency_directional',
                           'secrm_efficiency_directional_l1',
                           'secrm_efficiency_directional_l2')):
        checkpoint_callback = IntermediateCheckpointCallback(
            final_step=(args.target_total_timesteps if args.resume_model
                        else args.total_timesteps), save_freq=args.checkpoint_freq,
            save_path=checkpoint_dir, name_prefix=args.run_name, verbose=2,
            save_replay_buffer=args.save_replay_buffer)
    wandb_callback = WandbCallback(
        # Gradient watches use W&B's implicit step and would diverge from env_steps.
        gradient_save_freq=0,
        verbose=2,
    )

    # Combine callbacks
    callbacks = [episode_callback, checkpoint_callback, wandb_callback]

    # Train and save before closing the single active TraCI connection.
    try:
        model.learn(total_timesteps=learn_timesteps, callback=callbacks,
                    reset_num_timesteps=not bool(args.resume_model))
        save_model(model, model_path)
        if args.save_replay_buffer:
            replay_path = os.path.join(args.save_dir, f'{args.run_name}_replay_buffer.pkl')
            model.save_replay_buffer(replay_path)
            print(f'Saved final replay buffer to {replay_path}', flush=True)
    finally:
        vec_env.close()

    print(f"Saved latest model to {model_path}")
    print(f"Periodic checkpoints are in: {checkpoint_dir}")

if __name__ == '__main__':
    main()
