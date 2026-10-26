"""Mirror E76-to-D3 JSONL training metrics to one W&B/Forge run.

The synchronizer is deliberately separate from the learner. It reads only the
durable JSONL metric stream and never uploads source, models, replay buffers or
TensorBoard files. A connection failure delays cloud visibility but does not
interrupt training; rerunning with the same state directory resumes the run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from src import exit_e76_d3_config as exit_config


PAIRED_PHASES = {
    'control_134_200k': {'observation_dim': 134, 'nav_features': False},
    'nav_200k': {'observation_dim': 136, 'nav_features': True},
}


def expected_run_name(phase, seed):
    label = 'CONTROL134' if phase == 'control_134_200k' else 'NAV136'
    return f'E76D3-{label}-GAMMA099-S{seed}-200K'


def validate_run(phase, seed, run_name):
    if phase not in PAIRED_PHASES:
        raise ValueError('Sync accepts only the paired 200k E76-to-D3 phases')
    if seed not in exit_config.PHASES[phase].get('allowed_seeds', (42,)):
        raise ValueError(f'Seed {seed} is not approved for {phase}')
    expected = expected_run_name(phase, seed)
    if run_name != expected:
        raise ValueError(f'Expected run name {expected}, got {run_name}')
    return {
        'task': exit_config.TASK_ID,
        'exit_phase': phase,
        'seed': seed,
        'scenario': 'easy',
        'total_timesteps': 200_000,
        'gamma': exit_config.POTENTIAL_GAMMA,
        'observation_dim': PAIRED_PHASES[phase]['observation_dim'],
        'nav_features': PAIRED_PHASES[phase]['nav_features'],
        'training_safety': False,
        'delay_mode': 'uniform',
        'delay_steps': [0, 20],
        'gru_input_mode': 'paper_raw',
        'reward_profile': exit_config.REWARD_PROFILE,
        'control_profile': exit_config.CONTROL_PROFILE,
        'metric_schema': 'env_steps_v2',
    }


def metric_from_record(record):
    if record.get('event') not in ('episode', 'scalars', 'request_window'):
        return None
    if 'env_steps' not in record:
        raise ValueError('Metric record is missing env_steps')
    metric = {'env_steps': record['env_steps']}
    metric.update({key: value for key, value in record.items()
                   if key.startswith('train/')})
    return metric


def read_complete_records(path, consumed_lines):
    records = []
    next_line = consumed_lines
    if not path.exists():
        return records, next_line
    with path.open() as stream:
        for index, line in enumerate(stream):
            if index < consumed_lines:
                continue
            if not line.endswith('\n'):
                break
            records.append(json.loads(line))
            next_line = index + 1
    return records, next_line


def process_alive(pid_file):
    if not pid_file or not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=tuple(PAIRED_PHASES), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--jsonl', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--training-pid-file', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=15.0)
    parser.add_argument('--base-url', default='https://api.wandb.ai')
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error('--poll-seconds must be positive')
    try:
        args.formal_config = validate_run(args.phase, args.seed, args.run_name)
    except ValueError as error:
        parser.error(str(error))
    return args


def load_state(path, phase, seed, run_name):
    run_id = hashlib.sha256(f'e76-d3:{run_name}'.encode()).hexdigest()[:8]
    expected = {'phase': phase, 'seed': seed, 'run_name': run_name, 'run_id': run_id}
    if not path.exists():
        return {**expected, 'consumed_lines': 0, 'latest_env_steps': 0}
    state = json.loads(path.read_text())
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError('Existing sync state belongs to a different formal run')
    return state


def store_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, sort_keys=True) + '\n')
    temporary.replace(path)


def sync_session(args, state, state_path):
    import wandb

    os.environ['WANDB_BASE_URL'] = args.base_url
    settings = wandb.Settings(disable_git=True, disable_code=True,
                              x_disable_stats=True, x_disable_machine_info=True)
    with wandb.init(entity=exit_config.WANDB_ENTITY,
                    project=exit_config.WANDB_PROJECT,
                    id=state['run_id'], resume='allow', name=args.run_name,
                    group='e76-d3-seed43-paired-ablation',
                    job_type=args.phase, mode='online',
                    dir=str(args.state_dir), save_code=False,
                    config=args.formal_config, settings=settings) as run:
        run.define_metric('env_steps')
        run.define_metric('train/*', step_metric='env_steps')
        print(json.dumps({'event': 'connected', 'run_id': run.id,
                          'run_url': run.url}), flush=True)
        idle_after_exit = 0
        while True:
            records, _ = read_complete_records(args.jsonl, state['consumed_lines'])
            logged = 0
            for record in records:
                metric = metric_from_record(record)
                if metric:
                    run.log(metric)
                    state['latest_env_steps'] = max(
                        state['latest_env_steps'], int(metric['env_steps']))
                    logged += 1
                # Advance only after this individual record was handled. If a
                # log call raises, retry starts at that record instead of
                # silently skipping the remainder of the batch.
                state['consumed_lines'] += 1
                store_state(state_path, state)
            alive = process_alive(args.training_pid_file)
            if not alive and not records:
                idle_after_exit += 1
            else:
                idle_after_exit = 0
            if (not alive and idle_after_exit >= 2
                    and state['latest_env_steps'] >= 200_000):
                run.summary['synced_env_steps'] = state['latest_env_steps']
                run.summary['jsonl_lines'] = state['consumed_lines']
                print(json.dumps({'event': 'complete', 'logged': logged,
                                  'env_steps': state['latest_env_steps']}), flush=True)
                return
            time.sleep(args.poll_seconds)


def main(argv=None):
    args = parse_args(argv)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dir / 'state.json'
    state = load_state(state_path, args.phase, args.seed, args.run_name)
    retry = 0
    while True:
        try:
            sync_session(args, state, state_path)
            return
        except KeyboardInterrupt:
            raise
        except Exception as error:
            retry += 1
            print(json.dumps({'event': 'sync_retry', 'attempt': retry,
                              'error_type': type(error).__name__,
                              'error': str(error)[:300]}), file=sys.stderr, flush=True)
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
