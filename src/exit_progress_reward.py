"""Configurable E76-to-D3 route-end progress and terminal-event reward."""

import hashlib
import json
import math
from pathlib import Path


PROFILE = 'route_end_progress_events_v1'
DEFAULT_EXPERIMENT = {
    'seed': 42,
    'scenario': 'easy',
    'total_timesteps': 200_000,
    'checkpoint_freq': 50_000,
    'timeout_seconds': 400.0,
    'reward': {
        'progress_per_meter': 0.15,
        'proximity_gain': 1.0,
        'arrival': 50.0,
        'collision': -100.0,
        'timeout': -100.0,
        'missed_exit': -100.0,
        'teleport': -100.0,
    },
}
EVENTS = {
    'Arrived': 'arrival',
    'Collision': 'collision',
    'Max Steps': 'timeout',
    'Missed Exit': 'missed_exit',
    'Ego Teleported': 'teleport',
}


def validate_experiment(config):
    if not isinstance(config, dict) or set(config) != set(DEFAULT_EXPERIMENT):
        raise ValueError('Progress experiment must contain exactly the documented run and reward fields')
    if (type(config['seed']) is not int or config['seed'] < 0
            or config['scenario'] not in ('easy', 'medium', 'hard')
            or type(config['total_timesteps']) is not int or config['total_timesteps'] <= 0
            or type(config['checkpoint_freq']) is not int or config['checkpoint_freq'] <= 0):
        raise ValueError('Invalid progress experiment seed, scenario, budget or checkpoint interval')
    if not isinstance(config['reward'], dict) or set(config['reward']) != set(DEFAULT_EXPERIMENT['reward']):
        raise ValueError('Progress reward requires exactly seven configured coefficients')
    for name, value in {**config['reward'], 'timeout_seconds': config['timeout_seconds']}.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
    if (config['reward']['progress_per_meter'] < 0 or config['reward']['proximity_gain'] < 0
            or config['reward']['arrival'] < 0
            or any(config['reward'][name] > 0 for name in ('collision', 'timeout', 'missed_exit', 'teleport'))
            or config['timeout_seconds'] <= 0):
        raise ValueError('Progress, completion, failure and timeout signs must match the task')
    return json.loads(json.dumps(config))


def load_experiment(path=None):
    return validate_experiment(DEFAULT_EXPERIMENT if path is None else json.loads(Path(path).read_text()))


def reward_identity(reward):
    payload = json.dumps({'profile': PROFILE, 'reward': reward}, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def experiment_identity(config):
    payload = json.dumps({'profile': PROFILE, 'experiment': validate_experiment(config)},
                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def progress_reward(before_remaining, after_remaining, before_odometer,
                    after_odometer, initial_remaining, reward):
    values = (before_remaining, after_remaining, before_odometer,
              after_odometer, initial_remaining)
    if any(value is None or not math.isfinite(value) for value in values):
        return 0.0
    if before_remaining < 0 or after_remaining < 0 or initial_remaining <= 0:
        return 0.0
    meters = min(max(0.0, before_remaining - after_remaining),
                 max(0.0, after_odometer - before_odometer))
    closeness = 1.0 - min(max(before_remaining / initial_remaining, 0.0), 1.0)
    return reward['progress_per_meter'] * meters * (1.0 + reward['proximity_gain'] * closeness)


def event_reward(outcome, reward):
    key = EVENTS.get(outcome)
    return (key, float(reward[key])) if key is not None else (None, 0.0)
