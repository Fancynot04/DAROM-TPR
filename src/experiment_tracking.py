"""Serialization and observational training instrumentation (no RNG use)."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid
from collections import deque

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_metadata():
    root = Path(__file__).resolve().parents[1]
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)))


def runtime_versions():
    import sys
    import torch
    return dict(python=sys.version.split()[0], torch=str(torch.__version__), numpy=np.__version__)


def evaluation_fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / 'src').rglob('*.py')) + [root / 'evaluate_masac.py', root / 'train_masac.py']
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with temporary.open('wb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, data):
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode()
    atomic_write(path, lambda stream: stream.write(payload))


def checkpoint_due(step, total, interval):
    if interval <= 0:
        raise ValueError('checkpoint_every must be positive')
    return step % interval == 0 or step == total


class TrainingLog:
    def __init__(self, directory, tensorboard=True):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Never silently mix two runs in one experiment directory.
        self.stream = (directory / 'metrics.jsonl').open('x', encoding='utf-8')
        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(directory / 'tb'))

    def write(self, event, global_step, **values):
        record = dict(event=event, global_step=int(global_step), **values)
        payload = json.dumps(record, allow_nan=False, ensure_ascii=False)
        self.stream.write(payload + '\n')
        self.stream.flush()
        if self.writer:
            for key, value in values.items():
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    self.writer.add_scalar(f'exploration_training/{event}/{key}', value, global_step)
                elif isinstance(value, bool):
                    self.writer.add_scalar(f'exploration_training/{event}/{key}', int(value), global_step)

    def close(self):
        self.stream.close()
        if self.writer:
            self.writer.close()


class ActionWindow:
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self.clear()

    def clear(self):
        self.steps = 0
        self.values = {key: [] for key in self.agent_ids}
        self.started = time.monotonic()

    def observe(self, actions, mask):
        self.steps += 1
        for i, key in enumerate(self.agent_ids):
            if mask[i]:
                self.values[key].append(np.asarray(actions[i], dtype=float).copy())

    def result(self):
        from src.envs.control import parse_action
        from src.evaluation_metrics import wilson
        result = dict(window_global_steps=self.steps,
                      global_steps_per_second=self.steps / max(time.monotonic() - self.started, 1e-9))
        for key, values in self.values.items():
            result[key + '_action_samples'] = len(values)
            result[key + '_valid_action_fraction'] = len(values) / self.steps if self.steps else None
            result[key + '_valid_action_fraction_stats'] = wilson(len(values), self.steps, scale=1)
            result[key + '_lane_request_fraction_stats'] = wilson(sum(parse_action(v)[1] != 0 for v in values), len(values), scale=1)
            for name, value in (('accel_mean', np.mean([v[0] for v in values]) if values else None),
                                ('accel_std', np.std([v[0] for v in values]) if values else None),
                                ('lateral_mean', np.mean([v[1] for v in values]) if values else None),
                                ('lane_request_fraction', np.mean([parse_action(v)[1] != 0 for v in values]) if values else None)):
                result[key + '_' + name] = float(value) if value is not None else None
        return result


class EpisodeWindow:
    def __init__(self):
        self.rows = deque(maxlen=100)

    def add(self, agents):
        from src.evaluation_metrics import wilson
        self.rows.append(agents)
        result = {'rolling_episodes': len(self.rows), 'policy': 'exploration_training', 'rate_unit': 'percent'}
        for key in agents:
            rows = [row[key] for row in self.rows]
            for name, flags in (
                ('success', [r['success'] for r in rows]),
                ('collision', [r['terminal_reason'] == 'Collision' for r in rows]),
                ('missed_exit', [r['terminal_reason'] == 'Missed Exit' for r in rows]),
                ('timeout', [r['terminal_reason'] == 'Timeout' for r in rows]),
                ('stalled', [r['terminal_reason'] == 'Stalled' for r in rows]),
            ):
                result[key + '_' + name + '_rate'] = 100 * sum(flags) / len(rows)
                result[key + '_' + name + '_rate_stats'] = wilson(sum(flags), len(rows))
            result[key + '_mean_reward'] = float(np.mean([r['mean_policy_reward_return'] for r in rows]))
        result['joint_success_rate'] = 100 * sum(all(r['success'] for r in row.values()) for row in self.rows) / len(self.rows)
        result['joint_success_rate_stats'] = wilson(sum(all(r['success'] for r in row.values()) for row in self.rows), len(self.rows))
        joint_stalls = sum(any(r['terminal_reason'] == 'Stalled' for r in row.values()) for row in self.rows)
        result['joint_stalled_rate'] = 100 * joint_stalls / len(self.rows)
        result['joint_stalled_rate_stats'] = wilson(joint_stalls, len(self.rows))
        result['mean_reward'] = float(np.mean([sum(r['mean_policy_reward_return'] for r in row.values()) for row in self.rows]))
        return result
