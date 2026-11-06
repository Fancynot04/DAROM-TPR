"""Configurable E76-to-D3 route-end progress and terminal-event reward."""

import hashlib
import json
import math
from . import secrm_reward
from pathlib import Path


PROFILE = 'route_end_progress_events_v1'
DIRECTIONAL_PROFILE = 'route_end_progress_events_directional_lc_v1'
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
DIRECTIONAL_EXPERIMENT = {
    **DEFAULT_EXPERIMENT,
    'total_timesteps': 1_000_000,
    'checkpoint_freq': 200_000,
    'reward': {**DEFAULT_EXPERIMENT['reward'], 'alpha': 3.0, 'beta': 6.0},
}
SECRM_DEFAULTS = {
    name: {**DEFAULT_EXPERIMENT, 'reward': {
        **{key: DEFAULT_EXPERIMENT['reward'][key] for key in
           ('arrival', 'collision', 'timeout', 'missed_exit', 'teleport')},
        'efficiency_weight': 1.0, 'speed_epsilon': 2.0,
        **({'route_weight': 1.0} if name == 'secrm_efficiency_route' else {})}}
    for name in ('secrm_efficiency', 'secrm_efficiency_route')
}
SECRM_DEFAULTS['secrm_efficiency_directional'] = {
    **SECRM_DEFAULTS['secrm_efficiency'],
    'total_timesteps': 500_000, 'checkpoint_freq': 250_000,
    'reward': {**SECRM_DEFAULTS['secrm_efficiency']['reward'], 'alpha': 3.0, 'beta': 6.0},
}
for mode in ('l1', 'l2'):
    SECRM_DEFAULTS[f'secrm_efficiency_directional_{mode}'] = {
        **SECRM_DEFAULTS['secrm_efficiency_directional'],
        'reward': {**SECRM_DEFAULTS['secrm_efficiency']['reward'],
                   'lane_reward_mode': mode},
    }
SECRM_DEFAULTS['secrm_efficiency_directional_l1_speed120_500k'] = {
    **SECRM_DEFAULTS['secrm_efficiency_directional_l1'],
    'total_timesteps': 500_000,
    'checkpoint_freq': 250_000,
}
SECRM_DEFAULTS['secrm_efficiency_directional_l1_speed120_1m'] = {
    **SECRM_DEFAULTS['secrm_efficiency_directional_l1_speed120_500k'],
    'total_timesteps': 1_000_000,
}
SECRM_DEFAULTS['secrm_efficiency_directional_l1_speed120_2m'] = {
    **SECRM_DEFAULTS['secrm_efficiency_directional_l1_speed120_1m'],
    'total_timesteps': 2_000_000,
}
# C shares B's formula profile; only event weights change, hence a new identity.
SECRM_DEFAULTS['secrm_efficiency_route_no_events'] = {
    **SECRM_DEFAULTS['secrm_efficiency_route'],
    'reward': {**SECRM_DEFAULTS['secrm_efficiency_route']['reward'],
               **{key: 0.0 for key in ('arrival', 'collision', 'timeout', 'missed_exit', 'teleport')}}}
CONFIGURABLE_PROFILES = (PROFILE, DIRECTIONAL_PROFILE, *secrm_reward.PROFILES)

from .rewards.events import EVENTS, event_reward
from .rewards.progress import progress_reward
from .rewards.directional_lane import directional_lane_reward


def reward_profile(config):
    keys = set(config['reward'])
    base = set(DEFAULT_EXPERIMENT['reward'])
    if keys == base:
        return PROFILE
    if keys == base | {'alpha', 'beta'}:
        return DIRECTIONAL_PROFILE
    if keys == set(SECRM_DEFAULTS['secrm_efficiency_directional_l1']['reward']):
        return secrm_reward.ADAPTIVE_PROFILE
    for name, config in SECRM_DEFAULTS.items():
        if keys == set(config['reward']):
            return (secrm_reward.DIRECTIONAL_PROFILE if name == 'secrm_efficiency_directional'
                    else secrm_reward.ROUTE_PROFILE if name.endswith('_route')
                    else secrm_reward.EFFICIENCY_PROFILE)
    raise ValueError('Progress reward fields do not match a known version')


def validate_experiment(config):
    if not isinstance(config, dict) or set(config) != set(DEFAULT_EXPERIMENT):
        raise ValueError('Progress experiment must contain exactly the documented run and reward fields')
    if (type(config['seed']) is not int or config['seed'] < 0
            or config['scenario'] not in ('easy', 'medium', 'hard')
            or type(config['total_timesteps']) is not int or config['total_timesteps'] <= 0
            or type(config['checkpoint_freq']) is not int or config['checkpoint_freq'] <= 0):
        raise ValueError('Invalid progress experiment seed, scenario, budget or checkpoint interval')
    if not isinstance(config['reward'], dict):
        raise ValueError('Progress reward must be an object')
    profile = reward_profile(config)
    for name, value in {**config['reward'], 'timeout_seconds': config['timeout_seconds']}.items():
        if name == 'lane_reward_mode':
            if value not in ('l1', 'l2'):
                raise ValueError('lane_reward_mode must be l1 or l2')
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
    if (config['reward'].get('progress_per_meter', 0) < 0 or config['reward'].get('proximity_gain', 0) < 0
            or config['reward']['arrival'] < 0
            or any(config['reward'][name] > 0 for name in ('collision', 'timeout', 'missed_exit', 'teleport'))
            or (profile in (DIRECTIONAL_PROFILE, secrm_reward.DIRECTIONAL_PROFILE) and
                (config['reward']['alpha'] < 0 or config['reward']['beta'] < 0))
            or config['reward'].get('efficiency_weight', 0) < 0
            or config['reward'].get('route_weight', 0) < 0
            or config['reward'].get('speed_epsilon', 0.1) <= 0
            or config['timeout_seconds'] <= 0):
        raise ValueError('Progress, completion, failure and timeout signs must match the task')
    return json.loads(json.dumps(config))


def load_experiment(path=None, directional=False, phase=None):
    default = (SECRM_DEFAULTS[phase] if phase in SECRM_DEFAULTS else
               DIRECTIONAL_EXPERIMENT if directional else DEFAULT_EXPERIMENT)
    config = validate_experiment(default if path is None else json.loads(Path(path).read_text()))
    expected = reward_profile(default)
    if reward_profile(config) != expected:
        raise ValueError(f'Configuration reward version must be {expected}')
    if (phase in ('secrm_efficiency_directional_l1', 'secrm_efficiency_directional_l2',
                  'secrm_efficiency_directional_l1_speed120_500k',
                  'secrm_efficiency_directional_l1_speed120_1m',
                  'secrm_efficiency_directional_l1_speed120_2m')
            and config['reward']['lane_reward_mode'] != default['reward']['lane_reward_mode']):
        raise ValueError('Configuration lane reward mode does not match the phase')
    return config


def reward_identity(reward):
    profile = reward_profile({'reward': reward})
    payload = json.dumps({'profile': profile, 'reward': reward}, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def experiment_identity(config):
    profile = reward_profile(config)
    payload = json.dumps({'profile': profile, 'experiment': validate_experiment(config)},
                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()
