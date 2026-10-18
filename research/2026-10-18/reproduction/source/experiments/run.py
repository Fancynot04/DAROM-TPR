"""Run the controlled study. Listing/checking configuration never starts training."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import numbers
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'experiments' / 'first_round.json'


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    if set(cfg['groups']) != set('ABCD'):
        raise ValueError('The first-round study must contain exactly A/B/C/D.')
    if cfg['environment'] != {
        'scenario': 'hard', 'radius': 50, 'max_delay': 20,
        'delay_mode': 'uniform', 'training_safety': False,
    }:
        raise ValueError('First-round environment settings must remain fixed.')
    expected = {'A': (21, 128), 'B': (20, 128), 'C': (21, 256), 'D': (20, 256)}
    for name, group in cfg['groups'].items():
        if (group['hist_len'], group['features_dim']) != expected[name]:
            raise ValueError(f'{name}: unexpected factorial configuration')
        if 134 - 2 * group['hist_len'] - 1 != group['expected_state_dim']:
            raise ValueError(f'{name}: inconsistent input dimensions')
    if type(cfg['runtime']['parallel_runs']) is not int or not 1 <= cfg['runtime']['parallel_runs'] <= 2:
        raise ValueError('This shared-server study permits one or two concurrent runs.')
    if cfg['evaluation']['modes'] != ['no_sc', 'original_sc']:
        raise ValueError('Both paired evaluation modes are required.')
    if cfg['tracking']['upload_code'] or cfg['tracking']['upload_models']:
        raise ValueError('Code/model uploads are outside this study configuration.')
    if cfg['tracking']['mode'] not in ('disabled', 'offline', 'online'):
        raise ValueError('Invalid tracking mode')
    if cfg['tracking'].get('base_url', 'https://api.wandb.ai') != 'https://api.wandb.ai':
        raise ValueError('This study uses the W&B public API; the Forge workspace URL is not an API endpoint.')
    if cfg['total_timesteps'] <= 0 or cfg['evaluation']['episodes_per_mode'] <= 0:
        raise ValueError('Training/evaluation budgets must be positive.')
    if sorted(set(cfg['evaluation']['steps'])) != cfg['evaluation']['steps']:
        raise ValueError('Evaluation steps must be unique and increasing.')
    if any(step <= 0 for step in cfg['evaluation']['steps']):
        raise ValueError('Evaluation steps must be positive.')
    return cfg


def package_status(cfg):
    result = {}
    for name, target in cfg['target_packages'].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        result[name] = {'target': target, 'installed': actual,
                        'matches': actual is not None and actual.split('+')[0] == target}
    return result


def preflight(cfg, strict=True):
    status = package_status(cfg)
    binary = shutil.which('sumo')
    if not binary:
        raise RuntimeError('SUMO is not on PATH; no environment changes are performed automatically.')
    version = subprocess.check_output([binary, '--version'], text=True).splitlines()[0]
    # The active sumocfg references mergingP.rou.xml; require the audited Hard file.
    routes = ROOT / 'sumo_files'
    if (routes / 'mergingP.rou.xml').read_bytes() != (routes / 'mergingP_hard.rou.xml').read_bytes():
        raise RuntimeError('The active route is no longer identical to the audited Hard route.')
    if strict and not all(row['matches'] for row in status.values()):
        raise RuntimeError('Target dependencies are not ready. Run the check command; installation requires approval.')
    if strict and 'Version 1.18.0' not in version:
        raise RuntimeError('The first round fixes SUMO at 1.18.0; review the version change before training.')
    return {'packages': status, 'sumo': version, 'python': sys.version,
            'executable': sys.executable}


def configure_torch(cfg):
    import torch
    torch.set_num_threads(cfg['runtime']['torch_threads'])
    torch.set_num_interop_threads(cfg['runtime']['torch_interop_threads'])
    return torch


def make_env(cfg, seed, directory):
    from src.merging import Merging
    from src.wrapper import DelayWrapper
    settings = cfg['environment']
    env = Merging(seed=seed, radius=settings['radius'], render_mode='none',
                  reproducible=True, tripinfo_path=Path(directory) / 'tripinfo.xml')
    return DelayWrapper(env, max_delay=settings['max_delay'], mode='all',
                        delay_mode=settings['delay_mode'], delay_seed=seed)


def source_manifest():
    paths = sorted(set(ROOT.glob('src/*.py')) | set(ROOT.glob('experiments/*.py')) |
                   set(ROOT.glob('experiments/*.json')) | set(ROOT.glob('sumo_files/*')))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def evaluate(cfg, checkpoint, directory):
    """Executed only in a new process, with its own TraCI connection and RNGs."""
    import numpy as np
    from stable_baselines3 import SAC
    from src.safety_controller import safetyCheck
    configure_torch(cfg)
    settings = cfg['evaluation']
    model = SAC.load(str(checkpoint), device=settings['device'])
    rows = []
    for mode in settings['modes']:
        mode_dir = directory / mode
        mode_dir.mkdir()
        env = make_env(cfg, settings['episode_seed_start'], mode_dir)
        try:
            for index in range(settings['episodes_per_mode']):
                episode_seed = settings['episode_seed_start'] + index
                obs, _ = env.reset(seed=episode_seed)
                total, length, interventions = 0.0, 0, 0
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    if mode == 'original_sc':
                        action, safety_info = safetyCheck(action, obs, True, 'all')
                        interventions += int(not safety_info['safe'])
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += float(reward)
                    length += 1
                    if terminated or truncated:
                        break
                    if length > 2100:
                        raise RuntimeError('Episode exceeded the audited 200-second simulation limit.')
                rows.append({'mode': mode, 'seed': episode_seed, 'reward': total,
                             'length': length, 'outcome': info['message'],
                             'merged': bool(env.unwrapped.merged),
                             'sc_interventions': interventions})
        finally:
            env.close()
    metrics = {}
    outcomes = {'success': 'Arrived', 'collision': 'Collision',
                'timeout': 'Max Steps', 'emergency_stop': 'Emergency Stopping'}
    for mode in settings['modes']:
        selected = [row for row in rows if row['mode'] == mode]
        if any(row['outcome'] not in outcomes.values() for row in selected):
            raise RuntimeError('Unexpected episode outcome; refusing to silently drop failures.')
        for name, message in outcomes.items():
            metrics[f'eval/{mode}/{name}_rate'] = sum(r['outcome'] == message for r in selected) / len(selected)
        metrics[f'eval/{mode}/mean_reward'] = float(np.mean([r['reward'] for r in selected]))
        metrics[f'eval/{mode}/mean_length'] = float(np.mean([r['length'] for r in selected]))
        metrics[f'eval/{mode}/sc_intervention_rate'] = sum(r['sc_interventions'] for r in selected) / sum(r['length'] for r in selected)
    write_json(directory / 'results.json', {'episodes': rows, 'metrics': metrics})


def launch_evaluation(cfg, checkpoint, directory):
    directory.mkdir()
    write_json(directory / 'config.json', cfg)
    command = [sys.executable, '-B', '-m', 'experiments.run', '_evaluate',
               '--config', str(directory / 'config.json'), '--checkpoint', str(checkpoint),
               '--output', str(directory)]
    with (directory / 'process.log').open('w') as stream:
        subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       timeout=cfg['evaluation']['timeout_seconds'])
    return json.loads((directory / 'results.json').read_text())['metrics']


class Recorder:
    def __init__(self, directory, cfg, run_name):
        from torch.utils.tensorboard import SummaryWriter
        self.stream = (directory / 'metrics.jsonl').open('w')
        self.writer = SummaryWriter(str(directory / 'tensorboard'))
        self.run = None
        if cfg['tracking']['mode'] != 'disabled':
            import wandb
            verification = bool(cfg.get('verification_only', False))
            try:
                self.run = wandb.init(
                    project=cfg['tracking']['project'], entity=cfg['tracking']['entity'],
                    name=f'verification-{run_name}' if verification else run_name,
                    group=cfg['study'] + ('-verification' if verification else ''),
                    job_type='verification' if verification else 'training',
                    tags=[f"group-{cfg.get('selected_group', 'unknown')}",
                          f"seed-{cfg.get('training_seed', 'unknown')}",
                          'verification-only' if verification else 'first-round'],
                    mode=cfg['tracking']['mode'], dir=str(directory), config=cfg,
                    save_code=False, sync_tensorboard=False,
                    settings=wandb.Settings(
                        base_url=cfg['tracking'].get('base_url', 'https://api.wandb.ai'),
                        disable_git=True, disable_code=True, console='off',
                        x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True,
                        x_save_requirements=False, init_timeout=60,
                        x_file_stream_transmit_interval=cfg['tracking'].get('flush_interval_seconds', 2.0),
                    ),
                )
            except BaseException:
                self.writer.close()
                self.stream.close()
                raise
            self.run.define_metric('env_steps')
            self.run.define_metric('*', step_metric='env_steps')
            dashboard = cfg['tracking'].get('dashboard_url')
            tracking = {'id': self.run.id, 'mode': cfg['tracking']['mode'],
                        'path': f"{cfg['tracking']['entity']}/{cfg['tracking']['project']}/{self.run.id}",
                        'url': f'{dashboard}/runs/{self.run.id}' if dashboard else self.run.url}
            write_json(directory / 'tracking.json', tracking)
            if cfg['tracking']['mode'] == 'online':
                print(f"Live curves: {tracking['url']}", flush=True)

    def emit(self, values, step):
        scalars = {str(k): float(v) for k, v in values.items() if isinstance(v, numbers.Real)}
        if not all(math.isfinite(value) for value in scalars.values()):
            raise RuntimeError(f'Nonfinite metric at environment step {step}: {scalars}')
        self.stream.write(json.dumps({'env_steps': int(step), **scalars}, allow_nan=False) + '\n')
        self.stream.flush()
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, step)
        if self.run is not None:
            self.run.log({'env_steps': int(step), **scalars})

    def close(self, exit_code=0):
        self.stream.close()
        self.writer.close()
        if self.run is not None:
            self.run.finish(exit_code=exit_code)


def train(cfg, group, seed, output_root=None, smoke=False):
    environment = preflight(cfg, strict=not smoke)
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import KVWriter, Logger
    from src.delay_encoder import DelayAwareGRUEncoder
    torch = configure_torch(cfg)
    if cfg['runtime']['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; no silent CPU fallback.')
    cfg = copy.deepcopy(cfg)
    cfg['selected_group'], cfg['training_seed'], cfg['verification_only'] = group, seed, smoke
    root = Path(output_root or cfg['runtime']['output_root'])
    if not root.is_absolute():
        root = ROOT / root
    name = f"hard-uniform-{group}-seed{seed}"
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = root / f"{'smoke-' if smoke else ''}{name}-{stamp}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'config.json', cfg)
    write_json(directory / 'environment.json', environment)
    write_json(directory / 'source-hashes.json', source_manifest())
    tracked_diff = subprocess.run(['git', 'diff', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True)
    (directory / 'working-tree.diff').write_text(tracked_diff.stdout)
    # Include new, not-yet-committed experiment sources as well as existing code.
    for relative in source_manifest():
        destination = directory / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    recorder = Recorder(directory, cfg, name)
    started = time.monotonic()
    env = None
    failed = True

    class MetricSink(KVWriter):
        def write(self, key_values, key_excluded, step=0):
            recorder.emit(key_values, step)

        def close(self):
            pass

    class StudyCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rewards = deque(maxlen=100)
            self.episode_reward = 0.0
            self.episode_length = 0
            self.eval_seconds = 0.0
            self.completed_evals = set()
            self.best = None

        def _on_step(self):
            self.episode_reward += float(self.locals['rewards'][0])
            self.episode_length += 1
            if self.locals['dones'][0]:
                self.rewards.append(self.episode_reward)
                recorder.emit({'train/episode_reward': self.episode_reward,
                               'train/episode_length': self.episode_length,
                               'train/reward_mean_100': sum(self.rewards) / len(self.rewards)}, self.num_timesteps)
                self.episode_reward, self.episode_length = 0.0, 0
            return True

        def report(self):
            elapsed = time.monotonic() - started
            self.logger.record('time/elapsed_seconds', elapsed)
            self.logger.record('time/evaluation_seconds', self.eval_seconds)
            self.logger.record('time/effective_steps_per_second', self.num_timesteps / max(elapsed, 1e-9))
            self.logger.record('train/completed_updates', self.model._n_updates)
            self.logger.dump(self.num_timesteps)

        def evaluate_if_due(self):
            step = self.num_timesteps
            if step not in cfg['evaluation']['steps'] or step in self.completed_evals:
                return
            checkpoint = directory / f'model-{step}.zip'
            self.model.save(str(checkpoint))
            before = time.monotonic()
            metrics = launch_evaluation(cfg, checkpoint, directory / f'eval-{step}')
            self.eval_seconds += time.monotonic() - before
            recorder.emit(metrics, step)
            # A deterministic tie-break, common to all four groups; keep final separately.
            score = (metrics['eval/original_sc/success_rate'], metrics['eval/original_sc/mean_reward'])
            if self.best is None or score > self.best:
                self.best = score
                write_json(directory / 'best.json', {'checkpoint': checkpoint.name,
                                                     'env_steps': step, 'selection_score': score})
            self.completed_evals.add(step)

        def _on_rollout_start(self):
            # Previous transition and optimizer update are complete at this boundary.
            if self.num_timesteps and self.num_timesteps % cfg['runtime']['scalar_interval'] == 0:
                self.report()
            self.evaluate_if_due()

        def _on_training_end(self):
            self.evaluate_if_due()
            self.report()

    try:
        with (directory / 'simulation.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            env = make_env(cfg, seed, directory)
            settings = cfg['groups'][group]
            policy_kwargs = {
                'features_extractor_class': DelayAwareGRUEncoder,
                'features_extractor_kwargs': {
                    'act_dim': 2, 'hist_len': settings['hist_len'], 'delay_dim': 1,
                    'features_dim': settings['features_dim'], 'gru_hidden': cfg['network']['gru_hidden'],
                },
                'net_arch': cfg['network']['net_arch'],
            }
            model = SAC('MlpPolicy', env, seed=seed, device=cfg['runtime']['device'],
                        verbose=0, policy_kwargs=policy_kwargs, **cfg['sac'])
            model.set_logger(Logger(str(directory), [MetricSink()]))
            callback = StudyCallback()
            model.learn(total_timesteps=cfg['total_timesteps'], callback=callback, log_interval=None)
            model.save(str(directory / 'final.zip'))
            if not smoke:
                model.save_replay_buffer(str(directory / 'replay-buffer.pkl'))
            # Diagnostic RNG snapshots do not constitute an exact simulator resume.
            import random
            import numpy as np
            torch.save({'python': random.getstate(), 'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                       directory / 'rng-state.pt')
            write_json(directory / 'status.json', {'status': 'completed', 'steps': model.num_timesteps,
                       'gradient_updates': model._n_updates, 'wall_seconds': time.monotonic() - started,
                       'evaluation_seconds': callback.eval_seconds, 'verification_only': smoke,
                       'exact_resume_supported': False})
            failed = False
    except BaseException as error:
        write_json(directory / 'status.json', {'status': 'failed', 'error': str(error),
                                             'verification_only': smoke})
        raise
    finally:
        if env is not None:
            env.close()
        recorder.close(exit_code=1 if failed else 0)
    print(f'Completed: {directory}')
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'check', 'train', 'suite', '_evaluate'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--group', choices=list('ABCD'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--tracking', choices=['disabled', 'offline', 'online'])
    parser.add_argument('--smoke', action='store_true', help='128 CPU steps, two eval episodes per mode; never a study result')
    parser.add_argument('--smoke-steps', type=int, default=128,
                        help='Short verification budget (128..10000); requires --smoke')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.tracking:
        cfg['tracking']['mode'] = args.tracking
    if args.command == 'list':
        for group, settings in cfg['groups'].items():
            for seed in cfg['training_seeds']:
                print(json.dumps({'group': group, 'seed': seed, 'steps': cfg['total_timesteps'], **settings}))
        return
    if args.command == 'check':
        print(json.dumps(package_status(cfg), indent=2))
        try:
            print(json.dumps(preflight(cfg), indent=2))
        except RuntimeError as error:
            parser.exit(1, f'Preflight failed: {error}\n')
        return
    if args.command == '_evaluate':
        if args.checkpoint is None or args.output is None:
            parser.error('_evaluate requires --checkpoint and --output')
        evaluate(cfg, args.checkpoint, args.output)
        return
    if args.smoke:
        if not 128 <= args.smoke_steps <= 10000:
            parser.error('--smoke-steps must be between 128 and 10000')
        cfg['total_timesteps'] = args.smoke_steps
        cfg['sac']['buffer_size'] = 1024
        cfg['evaluation']['steps'] = [args.smoke_steps // 2, args.smoke_steps]
        cfg['evaluation']['episodes_per_mode'] = 2
        cfg['runtime']['scalar_interval'] = 32
        cfg['runtime']['device'] = 'cpu'
        # Default verification stays local; online verification requires an explicit flag.
        cfg['tracking']['mode'] = args.tracking or 'disabled'
        if args.output is None or not args.output.is_absolute():
            parser.error('--smoke requires an absolute --output in /tmp or an ignored temporary directory')
        args.output = args.output.resolve()
        if not (args.output.is_relative_to(Path('/tmp')) or args.output.is_relative_to(ROOT / 'tmp' / 'ai')):
            parser.error('Smoke artifacts must stay in /tmp or this repository\'s ignored tmp/ai directory')
    elif args.smoke_steps != 128:
        parser.error('--smoke-steps requires --smoke')
    if args.command == 'suite':
        from experiments.suite import run_suite
        run_suite(cfg, args.output, args.smoke, args.smoke_steps)
        return
    if args.group is None or args.seed not in cfg['training_seeds']:
        parser.error('train requires --group and a seed listed in the configuration')
    train(cfg, args.group, args.seed, args.output, args.smoke)


if __name__ == '__main__':
    main()
