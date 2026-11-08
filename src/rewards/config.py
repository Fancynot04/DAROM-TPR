"""Hydra/OmegaConf reward configuration loading and validation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

TERM_NAMES = ('success', 'collision', 'missed_exit', 'teleport', 'timeout')
DENSE_TERM_NAMES = ('efficiency', 'route')


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value


def validate_reward_config(config):
    config = _plain(config)
    if not isinstance(config, dict) or set(config) != {'name', 'terms', 'nonterminal_dense'}:
        raise ValueError('Reward config must contain name, terms and nonterminal_dense')
    if config['name'] not in ('events_only_v1', 'e76_d3_dense_v1'):
        raise ValueError(f'Unknown E76-D3 reward profile: {config["name"]!r}')
    dense = config['name'] == 'e76_d3_dense_v1'
    if type(config['nonterminal_dense']) is not bool or config['nonterminal_dense'] != dense:
        raise ValueError('nonterminal_dense must agree with the reward profile')
    expected_names = set(TERM_NAMES + (DENSE_TERM_NAMES if dense else ()))
    if not isinstance(config['terms'], dict) or set(config['terms']) != expected_names:
        raise ValueError('Reward config terms do not match its E76-D3 profile')
    for name in TERM_NAMES:
        term = config['terms'][name]
        if not isinstance(term, dict) or set(term) != {'enabled', 'weight', 'value'}:
            raise ValueError(f'Invalid reward term fields: {name}')
        if type(term['enabled']) is not bool:
            raise ValueError(f'{name}.enabled must be boolean')
        for key in ('weight', 'value'):
            if type(term[key]) not in (int, float) or not math.isfinite(float(term[key])):
                raise ValueError(f'{name}.{key} must be finite numeric')
    expected = {'success': 50.0, 'collision': -100.0, 'missed_exit': -100.0,
                'teleport': -100.0, 'timeout': -100.0}
    if {name: float(config['terms'][name]['value']) for name in TERM_NAMES} != expected:
        raise ValueError('events_only_v1 reward values must remain at the B0 baseline')
    if dense:
        for name in DENSE_TERM_NAMES:
            term = config['terms'][name]
            expected_fields = {'enabled', 'weight', 'clip_min', 'clip_max'}
            if name == 'efficiency':
                expected_fields.add('epsilon')
            else:
                expected_fields.add('distance_scale_m')
            if not isinstance(term, dict) or set(term) != expected_fields:
                raise ValueError(f'Invalid dense reward term fields: {name}')
            if type(term['enabled']) is not bool:
                raise ValueError(f'{name}.enabled must be boolean')
            for key, value in term.items():
                if key != 'enabled' and (type(value) not in (int, float) or not math.isfinite(float(value))):
                    raise ValueError(f'{name}.{key} must be finite numeric')
            if not term['clip_min'] <= term['clip_max'] <= 0:
                raise ValueError(f'{name} clip range must end at a non-positive value')
            if name == 'efficiency' and term['epsilon'] <= 0:
                raise ValueError('efficiency.epsilon must be positive')
            if name == 'route' and term['distance_scale_m'] <= 0:
                raise ValueError('route.distance_scale_m must be positive')
    return config


def reward_hash(config) -> str:
    payload = json.dumps(validate_reward_config(config), sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def load_hydra_config(config_dir=None, config_name='config', overrides=()):
    directory = Path(config_dir or Path(__file__).resolve().parents[2] / 'conf').resolve()
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=config_name, overrides=list(overrides))
    reward = validate_reward_config(composed.reward)
    # Persist only the resolved reward contract here. Runtime parameters remain
    # CLI-owned and are recorded from their final argparse values in the manifest.
    reward_yaml = OmegaConf.to_yaml(OmegaConf.create(reward), resolve=True)
    return composed, reward, reward_hash(reward), reward_yaml


def load_reward_file(path):
    path = Path(path)
    if path.suffix in ('.yaml', '.yml'):
        config = OmegaConf.load(path)
        return validate_reward_config(config), reward_hash(config), OmegaConf.to_yaml(config, resolve=True)
    # Retain explicit compatibility with the baseline JSON, but do not use it
    # as the training default.
    data = json.loads(path.read_text())
    if set(data) == {'version', 'events'}:
        terms = {name: {'enabled': True, 'weight': 1.0, 'value': data['events'][name]}
                 for name in TERM_NAMES}
        config = {'name': data['version'], 'terms': terms, 'nonterminal_dense': False}
        return validate_reward_config(config), reward_hash(config), OmegaConf.to_yaml(config, resolve=True)
    raise ValueError(f'Unknown reward configuration format: {path}')
