"""Compatibility-named loader for the independent B0 reward JSON."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'rewards' / 'events_only_v1.json'
REQUIRED_EVENTS = ('success', 'collision', 'missed_exit', 'teleport', 'timeout')

def validate_experiment(config):
    if not isinstance(config, dict) or set(config) != {'version', 'events'}:
        raise ValueError('Reward configuration must contain exactly version and events')
    if config['version'] != 'events_only_v1' or not isinstance(config['events'], dict):
        raise ValueError('Unknown reward configuration version')
    if set(config['events']) != set(REQUIRED_EVENTS):
        raise ValueError('Reward configuration contains unknown or missing events')
    values = config['events']
    if any(type(value) not in (int, float) for value in values.values()):
        raise ValueError('Reward event values must be numeric')
    if values['success'] != 50 or any(values[name] != -100 for name in REQUIRED_EVENTS[1:]):
        raise ValueError('events_only_v1 has fixed B0 event values')
    return json.loads(json.dumps(config))

def load_experiment(path=None, **_ignored):
    if isinstance(path, dict):
        return validate_experiment(path)
    chosen = DEFAULT_PATH if path is None else Path(path)
    try:
        payload = json.loads(chosen.read_text())
    except FileNotFoundError as error:
        raise ValueError(f'Unknown reward configuration: {chosen}') from error
    return validate_experiment(payload)

def reward_identity(config):
    payload = json.dumps(validate_experiment(config), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()

def experiment_identity(config):
    return reward_identity(config)

def reward_profile(config):
    return validate_experiment(config)['version']
