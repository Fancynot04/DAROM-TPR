# from delay_encoder import DelayAwareGRUEncoder
import os
import time
import argparse
import json
import torch
from typing import Optional
from src.wrapper import SafetyWrapper

import numpy as np

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from src.delay_encoder import DelayAwareEncoder, DelayAwareGRUEncoder, TinyTransformerEncoder

import wandb
from wandb.integration.sb3 import WandbCallback

from src.merging import ANTWERP_ENV_VERSION, ANTWERP_TASKS, Merging
from src import exit_e76_d3_config as exit_config
from src.wrapper import ABLATION_MODES, DelayWrapper

class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0, log_path=None):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_potential_reward = 0.0
        self.lane_episodes = [0] * 5
        self.lane_successes = [0] * 5
        self.started = time.monotonic()
        self.log_file = None
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self.log_file = open(log_path, 'x')

    def _write(self, record):
        if self.log_file:
            self.log_file.write(json.dumps(record) + '\n')
            self.log_file.flush()

    def _on_training_end(self):
        if self.log_file:
            self.log_file.close()

    def _on_step(self) -> bool:
        # Track episode reward and length
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_potential_reward += self.locals['infos'][0].get('potential_reward', 0.0)
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
                'train/emergency_stopping': int(outcome == 'Emergency Stopping'),
                'train/no_stage_timeout': int(outcome == 'Max Steps' and not stage_reached),
                'train/stage_reached_timeout': int(outcome == 'Max Steps' and stage_reached),
                'train/stage_reached': int(stage_reached),
                'train/stage_reward_paid': int(info.get('stage_reward_paid', False)),
            }
            if not exit_task:
                metrics['train/no_merge_timeout'] = int(outcome == 'Max Steps' and not merged)
                metrics['train/merged_timeout'] = int(outcome == 'Max Steps' and merged)
            if info.get('ego_start_lane') is not None:
                start_lane = int(info['ego_start_lane'])
                metrics['env_steps'] = self.num_timesteps
                metrics['train/ego_start_lane'] = start_lane
                metrics['train/initial_potential'] = float(info['initial_potential'])
                metrics['train/episode_potential_reward'] = float(self.current_potential_reward)
                if 0 <= start_lane < 5:
                    self.lane_episodes[start_lane] += 1
                    self.lane_successes[start_lane] += int(info['is_success'])
                    metrics[f'train/lane_{start_lane}_episodes'] = self.lane_episodes[start_lane]
                    metrics[f'train/lane_{start_lane}_success_rate'] = (
                        self.lane_successes[start_lane] / self.lane_episodes[start_lane])
            wandb.log(metrics)
            self._write({'event': 'episode', 'steps': self.num_timesteps,
                         'outcome': outcome, **{k: float(v) for k, v in metrics.items()}})
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_potential_reward = 0.0

        if self.num_timesteps % 100 == 0 and self.locals['infos'][0].get('ego_start_lane') is not None:
            latest = self.model.logger.name_to_value
            scalar_names = ('train/actor_loss', 'train/critic_loss',
                            'train/ent_coef', 'train/ent_coef_loss')
            scalars = {name: float(latest[name]) for name in scalar_names
                       if name in latest and np.isfinite(latest[name])}
            if scalars:
                record = {'env_steps': self.num_timesteps, **scalars}
                wandb.log(record)
                self._write({'event': 'scalars', 'steps': self.num_timesteps, **record})

        if self.num_timesteps % 1000 == 0:
            elapsed = time.monotonic() - self.started
            self._write({'event': 'progress', 'steps': self.num_timesteps,
                         'updates': int(self.model._n_updates), 'elapsed_seconds': elapsed,
                         'fps': self.num_timesteps / max(elapsed, 1e-6)})
            print(f'Training step {self.num_timesteps}, {self.num_timesteps / max(elapsed, 1e-6):.1f} steps/s', flush=True)
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

def load_model(path: str, env: gym.Env) -> SAC:
    return SAC.load(path, env=env)

def make_env(seed, scenario, delay_mode, max_delay, use_safety, render_mode=None,
             ablation='full', task='legacy_o3_d1'):
    env = DelayWrapper(
        Merging(seed=seed, scenario=scenario, render_mode=render_mode, task=task),
        max_delay=max_delay, mode='all', delay_mode=delay_mode, seed=seed,
        ablation=ablation,
    )
    return SafetyWrapper(env) if use_safety else env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, choices=['must', 'allow', 'never'])
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--radius', type=int, default=50)
    parser.add_argument('--total_timesteps', type=int, default=7_000_000,
                        help='Training steps; defaults to the paper v5 budget')
    parser.add_argument('--checkpoint_freq', type=int, default=2_000_000,
                        help='Model milestone interval; 2M follows the D long-run schedule')
    parser.add_argument('--save_replay_buffer', action='store_true',
                        help='Save replay buffers alongside periodic and final models')
    parser.add_argument('--torch_threads', type=int, default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--log_dir', default='runs')
    parser.add_argument('--episode_log', default=None)
    parser.add_argument('--render_mode', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scenario', choices=['easy', 'medium', 'hard'], default='medium')
    parser.add_argument('--task', choices=tuple(ANTWERP_TASKS), default='legacy_o3_d1',
                        help='Single-ego task; legacy default preserves existing O3-to-D1 runs')
    parser.add_argument('--exit_phase', choices=tuple(exit_config.PHASES), default=None,
                        help='Dedicated E76-to-D3 Easy pilot or Easy 1M configuration')
    parser.add_argument('--safety', dest='use_safety', action='store_true')
    parser.add_argument('--save_dir', type=str, default='models_experiments')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--replay_buffer_path', type=str, default=None,
                        help='Required with --load_path for E76-to-D3 continuation')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--encoder_mode', type=str, default='GRU', choices=['Transformer', 'GRU', 'MLP', 'None'])
    parser.add_argument('--gru_input_mode', choices=DelayAwareGRUEncoder.INPUT_MODES,
                        default='paper_raw',
                        help='New GRU runs use unscaled wrapper inputs; the Antwerp legacy mode is for old checkpoints')
    parser.add_argument('--ablation', choices=ABLATION_MODES, default='full')
    parser.add_argument('--delay_mode', type=str, default='uniform', choices=['uniform', 'exponential', 'triangular', 'bursty', 'bimodal'])
    args = parser.parse_args()
    if args.exit_phase:
        if args.task not in ('legacy_o3_d1', exit_config.TASK_ID):
            parser.error('--exit_phase cannot be combined with another task')
        if (args.seed != 42 or args.radius != 50 or args.use_safety
                or args.encoder_mode != 'GRU' or args.gru_input_mode != 'paper_raw'
                or args.ablation != 'full' or args.delay_mode != 'uniform'):
            parser.error('E76-to-D3 phase requires the D-group seed, radius, '
                         'safety-off, full paper_raw GRU and uniform delay settings')
        phase = exit_config.PHASES[args.exit_phase]
        args.task = exit_config.TASK_ID
        args.scenario = phase['scenario']
        args.total_timesteps = phase['total_timesteps']
        args.checkpoint_freq = phase['checkpoint_freq']
        args.save_replay_buffer = True
        if args.project not in (None, exit_config.WANDB_PROJECT):
            parser.error('E76-to-D3 profile uses the designated DAROM-OUT Forge project')
        if args.entity not in (None, exit_config.WANDB_ENTITY):
            parser.error('E76-to-D3 profile uses the designated Forge entity')
        args.project = exit_config.WANDB_PROJECT
        args.entity = exit_config.WANDB_ENTITY
        if args.device == 'auto':
            args.device = 'cuda'
        if args.torch_threads is None:
            args.torch_threads = 2
        if args.log_dir == 'runs':
            args.log_dir = f'tmp/ai/e76-d3/{args.exit_phase}/tensorboard'
        if args.save_dir == 'models_experiments':
            args.save_dir = 'models_experiments/e76-d3'
        if args.run_name is None:
            args.run_name = f'e76-d3-{args.exit_phase}-{time.strftime("%Y%m%d-%H%M%S")}'
        if args.episode_log is None:
            args.episode_log = f'tmp/ai/e76-d3/{args.exit_phase}/{args.run_name}.jsonl'
    elif args.task == exit_config.TASK_ID:
        parser.error('Use --exit_phase for the E76-to-D3 task')
    else:
        args.project = args.project or 'merging-SAC_experiments'
    if args.exit_phase and bool(args.load_path) != bool(args.replay_buffer_path):
        parser.error('E76-to-D3 continuation requires both model and replay-buffer paths')
    if args.ablation != 'full' and args.encoder_mode != 'GRU':
        parser.error('Table IV ablations require --encoder_mode GRU')
    if args.encoder_mode != 'GRU' and args.gru_input_mode != 'paper_raw':
        parser.error('--gru_input_mode only applies to --encoder_mode GRU')
    if args.checkpoint_freq < 1 or (args.torch_threads is not None and args.torch_threads < 1):
        parser.error('checkpoint_freq and torch_threads must be positive')
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

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    ensure_dirs(args.log_dir)

    max_delay = 20
    sac_parameters = dict(
        learning_rate=3e-5, batch_size=512, buffer_size=1_000_000,
        learning_starts=100, gamma=0.99, tau=0.005, train_freq=1,
        gradient_steps=1, target_update_interval=1, ent_coef='auto',
        target_entropy='auto', use_sde=False,
    )
    config = dict(
        algo='SAC',
        radius=args.radius,
        seed=args.seed,
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
        ego_depart_lane=(exit_config.DEPART_LANE if args.exit_phase else None),
        potential_beta=(exit_config.POTENTIAL_BETA if args.exit_phase else None),
        potential_gamma=(exit_config.POTENTIAL_GAMMA if args.exit_phase else None),
        reward_profile=(exit_config.REWARD_PROFILE if args.exit_phase else None),
        metric_schema=('env_steps_v2' if args.exit_phase else None),
        wandb_project=args.project,
        wandb_entity=args.entity,
        use_safety=args.use_safety,
        training_safety=args.use_safety,
        antwerp_env_version=ANTWERP_ENV_VERSION,
        device=args.device,
        torch_threads=args.torch_threads,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        replay_buffer_path=args.replay_buffer_path,
    )
    # Create env
    def _fn():
        return make_env(args.seed, args.scenario, args.delay_mode, max_delay,
                        args.use_safety, args.render_mode, args.ablation, args.task)

    # SB3 expects the Gym API; our env already implements gymnasium API. SB3 v2 works with gymnasium.
    vec_env = DummyVecEnv([_fn])

    wrapped_env = vec_env.envs[0]
    delay_env = wrapped_env.env if args.use_safety else wrapped_env
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
        policy_kwargs = dict(features_extractor_class=DelayAwareGRUEncoder, features_extractor_kwargs=dict(act_dim=act_dim, hist_len=delay_env.max_delay, delay_dim=1, features_dim=256, gru_hidden=64, input_mode=args.gru_input_mode), net_arch=[256, 256])
    elif encoder_mode == 'MLP':
        policy_kwargs = dict(
        features_extractor_class=DelayAwareEncoder,
        features_extractor_kwargs=dict(act_hist_dim=max_delay * act_dim, delay_dim=1)
        )
    else:
        policy_kwargs = dict()

    model = SAC('MlpPolicy', vec_env, seed=args.seed, verbose=1, tensorboard_log=os.path.join(args.log_dir, args.run_name), device=args.device, policy_kwargs=policy_kwargs, **sac_parameters)
    
    
    

    # Optionally load existing weights
    if args.load_path:
        if not os.path.exists(args.load_path):
            raise FileNotFoundError(f'Checkpoint does not exist: {args.load_path}')
        try:
            loaded_model = SAC.load(args.load_path, env=vec_env)
        except Exception as e:
            raise RuntimeError(f'Failed to load checkpoint at {args.load_path}') from e
        else:
            if getattr(loaded_model, 'antwerp_env_version', None) != ANTWERP_ENV_VERSION:
                raise ValueError('Loaded model uses an older Antwerp observation/reward contract; retrain it')
            if (hasattr(loaded_model, 'ablation') and
                    loaded_model.ablation != args.ablation):
                raise ValueError('Loaded model ablation does not match --ablation')
            if (hasattr(loaded_model, 'training_safety') and
                    loaded_model.training_safety != args.use_safety):
                raise ValueError('Loaded model safety does not match --safety')
            loaded_task = getattr(loaded_model, 'antwerp_task', 'legacy_o3_d1')
            if loaded_task != args.task:
                raise ValueError(f'Loaded model task is {loaded_task!r}; requested {args.task!r}')
            if args.encoder_mode == 'GRU':
                if loaded_model.policy_kwargs.get('features_extractor_class') is not DelayAwareGRUEncoder:
                    raise ValueError('Loaded model does not use the DAROM-GRU encoder')
                loaded_mode = loaded_model.policy_kwargs.get('features_extractor_kwargs', {}).get(
                    'input_mode', 'antwerp_scaled_reversed')
                if loaded_mode != args.gru_input_mode:
                    raise ValueError(f'Loaded model GRU input mode is {loaded_mode!r}; '
                                     f'requested {args.gru_input_mode!r}')
            if args.exit_phase:
                if (getattr(loaded_model, 'exit_potential_beta', None)
                        != exit_config.POTENTIAL_BETA):
                    raise ValueError('Loaded model has a different E76-to-D3 potential profile')
                if (getattr(loaded_model, 'exit_reward_profile', None)
                        != exit_config.REWARD_PROFILE):
                    raise ValueError('Loaded model has a different E76-to-D3 reward profile')
                loaded_model.load_replay_buffer(args.replay_buffer_path)
                if loaded_model.num_timesteps >= args.total_timesteps:
                    raise ValueError('Checkpoint already reached the configured total step budget')
                loaded_model.tensorboard_log = os.path.join(args.log_dir, args.run_name)
            model = loaded_model
            print(f"Loaded model from {args.load_path}")

    model.ablation = args.ablation
    model.antwerp_env_version = ANTWERP_ENV_VERSION
    model.antwerp_task = args.task
    model.training_safety = args.use_safety
    model.gru_input_mode = args.gru_input_mode if args.encoder_mode == 'GRU' else None
    model.experiment_id = args.run_name
    if args.exit_phase:
        model.exit_potential_beta = exit_config.POTENTIAL_BETA
        model.exit_reward_profile = exit_config.REWARD_PROFILE

    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name, config=config,
                     mode=args.wandb_mode, dir=args.log_dir,
                     sync_tensorboard=not args.exit_phase,
                     save_code=not args.exit_phase, resume=args.resume, id=args.run_id,
                     group=f'e76-d3-{args.exit_phase}-{exit_config.REWARD_PROFILE}'
                     if args.exit_phase else None,
                     job_type=args.exit_phase,
                     tags=['E76-D3', 'single-ego', 'potential-beta10', 'metric-schema-v2',
                           exit_config.REWARD_PROFILE]
                     if args.exit_phase else None)
    if not args.exit_phase:
        run.log_code('src')
    else:
        run.define_metric('env_steps')
        run.define_metric('train/*', step_metric='env_steps')

    ensure_dirs(args.save_dir)

    # Create callbacks
    episode_callback = EpisodeRewardCallback(log_path=args.episode_log)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq, save_path=checkpoint_dir,
        name_prefix=args.run_name, verbose=2,
        save_replay_buffer=args.save_replay_buffer,
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=0 if args.exit_phase else 1000,
        verbose=2,
    )

    # Combine callbacks
    callbacks = [episode_callback, checkpoint_callback, wandb_callback]

    # Train and save before closing the single active TraCI connection.
    try:
        continuing_exit = bool(args.exit_phase and args.load_path)
        remaining_steps = (args.total_timesteps - model.num_timesteps
                           if continuing_exit else args.total_timesteps)
        model.learn(total_timesteps=remaining_steps, callback=callbacks,
                    reset_num_timesteps=not continuing_exit)
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
