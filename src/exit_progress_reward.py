"""Compatibility facade for the modular E76-D3 reward configuration."""
from __future__ import annotations

from pathlib import Path

from .rewards.config import (load_hydra_config, load_reward_file, reward_hash,
                             validate_reward_config)

DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'rewards' / 'events_only_v1.json'


def validate_experiment(config):
    if isinstance(config, dict) and set(config) == {'version', 'events'}:
        terms = {name: {'enabled': True, 'weight': 1.0, 'value': value}
                 for name, value in config['events'].items()}
        config = {'name': config['version'], 'terms': terms, 'nonterminal_dense': False}
    return validate_reward_config(config)


def load_experiment(path=None, **kwargs):
    if isinstance(path, dict):
        return validate_experiment(path)
    if path is None:
        return load_hydra_config()[1]
    return load_reward_file(path)[0]


def reward_identity(config):
    return reward_hash(validate_experiment(config))


def experiment_identity(config):
    return reward_identity(config)


def reward_profile(config):
    return validate_experiment(config)['name']
