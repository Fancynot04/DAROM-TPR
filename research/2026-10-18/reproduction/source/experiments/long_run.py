"""Single D/seed42 long run, with verified resumable checkpoints and separate log segments."""

import argparse
import copy
import fcntl
import json
import math
import numbers
import os
import pickle
import signal
import subprocess
import sys
import time
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from experiments.run import ROOT, configure_torch, evaluate, make_env, preflight, write_json

DEFAULT_CONFIG = ROOT / 'experiments' / 'd_long.json'


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg['groups'] != {'D': {'hist_len': 20, 'features_dim': 256, 'expected_state_dim': 93}}:
        raise ValueError('Long training is restricted to the existing D architecture.')
    if cfg['training_seeds'] != [42] or cfg['runtime']['parallel_runs'] != 1:
        raise ValueError('This is a single D/seed42 experiment.')
    if cfg['environment'] != {'scenario': 'hard', 'radius': 50, 'max_delay': 20,
                               'delay_mode': 'uniform', 'training_safety': False}:
        raise ValueError('Keep the audited D traffic, reward, safety and delay settings.')
    if cfg['sac']['train_freq'] != 1 or cfg['sac']['gradient_steps'] != 1 or cfg['sac']['use_sde']:
        raise ValueError('Checkpoint boundaries require the audited one-step SAC configuration.')
    if cfg['tracking']['upload_code'] or cfg['tracking']['upload_models']:
        raise ValueError('Remote code/model uploads are disabled.')
    if cfg['tracking']['base_url'] != 'https://api.wandb.ai':
        raise ValueError('Forge dashboard is not the API endpoint.')
    if not cfg.get('verification_only') and cfg['total_timesteps'] != 7000000:
        raise ValueError('The formal budget is 7,000,000 total environment steps.')
    interval = cfg['checkpointing']['interval_steps']
    if type(interval) is not int or interval <= 0:
        raise ValueError('Checkpoint interval must be a positive integer.')
    if any(not 0 < s <= cfg['total_timesteps'] for s in cfg['evaluation']['steps']):
        raise ValueError('Evaluation steps must lie within the training budget.')
    if cfg['evaluation']['steps'] != sorted(set(cfg['evaluation']['steps'])):
        raise ValueError('Evaluation steps must be unique and increasing.')
    return cfg


class Metrics:
    def __init__(self, directory, cfg, lineage, parent):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(directory / 'tensorboard'))
        self.stream = (directory / 'metrics.jsonl').open('w')
        self.run = None
        if cfg['tracking']['mode'] == 'disabled':
            return
        import wandb
        start = parent['env_steps'] if parent else 0
        try:
            self.run = wandb.init(
                entity=cfg['tracking']['entity'], project=cfg['tracking']['project'],
                name=f"{'verification-' if cfg.get('verification_only') else ''}D-seed42-from{start}-{directory.name[-8:]}",
                group=lineage, job_type='verification' if cfg.get('verification_only') else 'long-training',
                tags=['group-D', 'seed-42', 'verification-only' if cfg.get('verification_only') else 'long-7m'],
                config={**cfg, 'lineage': lineage, 'parent_checkpoint': parent},
                mode=cfg['tracking']['mode'], dir=str(directory), save_code=False,
                settings=wandb.Settings(base_url=cfg['tracking']['base_url'], disable_git=True,
                    disable_code=True, console='off', x_disable_meta=True, x_disable_stats=True,
                    x_disable_machine_info=True, x_save_requirements=False, init_timeout=60,
                    x_file_stream_transmit_interval=cfg['tracking']['flush_interval_seconds']))
            self.run.define_metric('env_steps')
            self.run.define_metric('*', step_metric='env_steps')
            url = cfg['tracking']['dashboard_url'] + '/runs/' + self.run.id
            write_json(directory / 'tracking.json', {'id': self.run.id, 'url': url, 'lineage': lineage})
            print(f'Live curves: {url}', flush=True)
        except BaseException:
            self.stream.close()
            self.writer.close()
            raise

    def emit(self, values, step):
        values = {str(k): float(v) for k, v in values.items() if isinstance(v, numbers.Real)}
        if not all(math.isfinite(v) for v in values.values()):
            raise RuntimeError(f'Nonfinite metric at step {step}: {values}')
        self.stream.write(json.dumps({'env_steps': int(step), **values}, allow_nan=False) + '\n')
        self.stream.flush()
        for k, v in values.items():
            self.writer.add_scalar(k, v, step)
        if self.run:
            self.run.log({'env_steps': int(step), **values})

    def close(self, failed):
        self.stream.close()
        self.writer.close()
        if self.run:
            self.run.finish(exit_code=int(failed))


class Paused(Exception):
    pass


def train(cfg, output, resume=None, stop_at=None):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import KVWriter, Logger
    from src.delay_encoder import DelayAwareGRUEncoder
    from experiments.checkpoints import (EpisodeReplay, atomic_json, save_checkpoint,
        validate_checkpoint, restore_env, set_rng, get_rng, source_hashes)
    torch = configure_torch(cfg)
    environment = preflight(cfg)
    device = cfg['runtime']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback.')
    environment.update(device=device, gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                       torch_threads=cfg['runtime']['torch_threads'],
                       torch_interop_threads=cfg['runtime']['torch_interop_threads'])
    parent = validate_checkpoint(resume, environment) if resume else None
    starting_step = parent['env_steps'] if parent else 0
    if starting_step >= cfg['total_timesteps']:
        raise ValueError('Checkpoint has already reached the total training budget.')
    if stop_at is not None and not starting_step < stop_at <= cfg['total_timesteps']:
        raise ValueError('--stop-at must be greater than checkpoint step and within total budget.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = output / f"D-seed42-from{starting_step}-{stamp}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    lineage = parent['lineage'] if parent else 'D-long-' + uuid4().hex[:12]
    parent_record = {'path': str(resume), 'env_steps': starting_step} if resume else None
    write_json(directory / 'config.json', cfg)
    write_json(directory / 'environment.json', environment)
    write_json(directory / 'parent.json', parent_record)
    hashes = source_hashes()
    write_json(directory / 'source-hashes.json', hashes)
    for name in hashes:
        destination = directory / 'source-snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    print(f'Run directory: {directory}', flush=True)
    metrics = None
    env = None
    model = None
    failed = True
    requested = []
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, lambda number, frame: requested.append(number))
    started = time.monotonic()

    class Sink(KVWriter):
        def write(self, key_values, key_excluded, step=0):
            metrics.emit(key_values, step)
        def close(self):
            pass

    class Callback(BaseCallback):
        def __init__(self, saved=None):
            super().__init__()
            self.state = saved or {'rewards': [], 'episode_reward': 0.0, 'episode_length': 0,
                'completed_evals': [], 'evaluation_seconds': 0.0, 'active_seconds': 0.0,
                'best': None, 'last_processed_step': -1}
            self.base_seconds = self.state['active_seconds']
            self.latest = None

        def _on_step(self):
            self.state['episode_reward'] += float(self.locals['rewards'][0])
            self.state['episode_length'] += 1
            if self.locals['dones'][0]:
                self.state['rewards'] = (self.state['rewards'] + [self.state['episode_reward']])[-100:]
                metrics.emit({'train/episode_reward': self.state['episode_reward'],
                    'train/episode_length': self.state['episode_length'],
                    'train/reward_mean_100': sum(self.state['rewards']) / len(self.state['rewards'])}, self.num_timesteps)
                self.state['episode_reward'] = 0.0
                self.state['episode_length'] = 0
            return True

        def status(self, status='running'):
            atomic_json(directory / 'status.json', {'status': status, 'pid': os.getpid(),
                'steps': self.model.num_timesteps, 'gradient_updates': self.model._n_updates,
                'total_timesteps': cfg['total_timesteps'], 'lineage': lineage,
                'latest_checkpoint': str(self.latest) if self.latest else str(resume) if resume else None,
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'verification_only': cfg.get('verification_only', False)})

        def boundary(self):
            step = self.model.num_timesteps
            pause = step > 0 and (bool(requested) or stop_at == step)
            if self.state['last_processed_step'] == step and not pause:
                return
            if step and step % cfg['runtime']['scalar_interval'] == 0:
                self.logger.record('train/completed_updates', self.model._n_updates)
                self.logger.record('time/active_seconds', self.base_seconds + time.monotonic() - started)
                self.logger.record('time/evaluation_seconds', self.state['evaluation_seconds'])
                self.logger.dump(step)
                self.status()
            if step in cfg['evaluation']['steps'] and step not in self.state['completed_evals']:
                # Evaluation uses a separate process, so parent SUMO stays untouched.
                before = time.monotonic()
                checkpoint = directory / f'model-{step}.zip'
                self.model.save(checkpoint)
                evaluation = directory / f'eval-{step}'
                evaluation.mkdir()
                with (evaluation / 'process.log').open('w') as log:
                    subprocess.run([sys.executable, '-B', '-m', 'experiments.long_run', '_evaluate',
                        '--config', str(directory / 'config.json'), '--checkpoint', str(checkpoint),
                        '--output', str(evaluation)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        check=True, timeout=cfg['evaluation']['timeout_seconds'])
                values = json.loads((evaluation / 'results.json').read_text())['metrics']
                metrics.emit(values, step)
                score = [values['eval/original_sc/success_rate'], values['eval/original_sc/mean_reward']]
                if self.state['best'] is None or score > self.state['best']['score']:
                    self.state['best'] = {'score': score, 'env_steps': step, 'model': str(checkpoint)}
                    atomic_json(directory / 'best.json', self.state['best'])
                self.state['completed_evals'].append(step)
                self.state['evaluation_seconds'] += time.monotonic() - before
            self.state['last_processed_step'] = step
            due = step > 0 and (step % cfg['checkpointing']['interval_steps'] == 0 or
                               step == cfg['total_timesteps'] or pause)
            if due:
                self.state['active_seconds'] = self.base_seconds + time.monotonic() - started
                self.latest = save_checkpoint(self.model, env, copy.deepcopy(self.state),
                                              directory, cfg, environment, lineage, hashes)
                self.status()
            if pause:
                raise Paused('Graceful pause after a completed optimizer update.')

        def _on_rollout_start(self):
            self.boundary()

        def _on_training_end(self):
            self.boundary()

    callback = None
    try:
        metrics = Metrics(directory, cfg, lineage, parent_record)
        with (directory / 'simulation.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            env = EpisodeReplay(make_env(cfg, 42, directory))
            if resume:
                model = SAC.load(Path(resume) / 'model.zip', env=env, device=device, force_reset=False)
                model.load_replay_buffer(Path(resume) / 'replay-buffer.pkl', truncate_last_traj=False)
                with (Path(resume) / 'state.pkl').open('rb') as stream:
                    saved = pickle.load(stream)
                restore_env(model, env, saved['environment'])
                if model.num_timesteps != starting_step or model._n_updates != parent['gradient_updates']:
                    raise RuntimeError('Saved SAC counters disagree with manifest.')
                callback = Callback(saved['callback'])
            else:
                kwargs = {'features_extractor_class': DelayAwareGRUEncoder,
                    'features_extractor_kwargs': {'act_dim': 2, 'hist_len': 20, 'delay_dim': 1,
                                                  'features_dim': 256, 'gru_hidden': 64},
                    'net_arch': cfg['network']['net_arch']}
                model = SAC('MlpPolicy', env, seed=42, device=device, verbose=0,
                            policy_kwargs=kwargs, **cfg['sac'])
                callback = Callback()
            model.set_logger(Logger(str(directory), [Sink()]))
            if resume:
                set_rng(saved['rng'])
            # Remaining budget, not another 7M steps; preserve the original step counter.
            remaining = cfg['total_timesteps'] - model.num_timesteps
            model.learn(total_timesteps=remaining, callback=callback, log_interval=None,
                        reset_num_timesteps=not bool(resume))
            callback.status('completed')
            failed = False
    except Paused:
        callback.status('paused')
        failed = False
    except BaseException as error:
        atomic_json(directory / 'status.json', {'status': 'failed', 'error': str(error),
                    'steps': model.num_timesteps if model else starting_step,
                    'latest_checkpoint': str(callback.latest) if callback and callback.latest else str(resume) if resume else None})
        raise
    finally:
        if env:
            env.close()
        if metrics:
            metrics.close(failed)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    print(f'Segment finished: {directory}', flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'train', '_evaluate'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume', type=Path, help='Trusted complete checkpoint DIRECTORY, not model.zip')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stop-at', type=int, help='Save and pause at this absolute environment step')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--verification', action='store_true', help='512 steps, small buffer, CPU, local logs')
    parser.add_argument('--verification-device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.resume and (args.config or args.verification or args.command != 'train'):
        parser.error('--resume uses its saved config; do not pass --config or --verification')
    if args.resume:
        args.resume = args.resume.resolve()
    cfg = load_config(args.resume / 'config.json' if args.resume else args.config or DEFAULT_CONFIG)
    if args.verification:
        if args.command != 'train' or not args.output:
            parser.error('--verification requires train and an explicit temporary --output')
        cfg.update(total_timesteps=512, verification_only=True)
        # Small enough to exercise replay-buffer wraparound during the restore test.
        cfg['sac']['buffer_size'] = 256
        cfg['checkpointing']['interval_steps'] = 128
        cfg['evaluation'].update(steps=[256, 512], episodes_per_mode=2)
        cfg['runtime'].update(device=args.verification_device, scalar_interval=32)
        cfg['tracking']['mode'] = 'disabled'
    output = (args.output or ROOT / cfg['runtime']['output_root']).resolve()
    if cfg.get('verification_only') and not (output.is_relative_to('/tmp') or output.is_relative_to(ROOT / 'tmp/ai')):
        parser.error('Verification artifacts must stay in /tmp or ignored tmp/ai.')
    if args.command == 'check':
        print(json.dumps({'config': cfg, 'environment': preflight(cfg)}, indent=2))
        return
    if args.command == '_evaluate':
        if args.checkpoint is None or args.output is None:
            parser.error('_evaluate requires --checkpoint and --output')
        evaluate(cfg, args.checkpoint, output)
        return
    lock_path = ROOT / 'runs' / '.D-long.lock'
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another D long-training process is already running.')
        train(cfg, output, args.resume, args.stop_at)


if __name__ == '__main__':
    main()
