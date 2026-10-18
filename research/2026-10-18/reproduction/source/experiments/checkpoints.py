"""Full SAC checkpoints with deterministic reconstruction of the current SUMO episode.

No native SUMO snapshot is assumed to capture all TraCI/lane-change state. Replay
the bounded current episode from its independent seed, verify its trace, then
restore Python state and all training RNGs before collecting another transition.
Only load trusted, locally generated pickle/torch checkpoints.
"""

import copy
import hashlib
import json
import os
import pickle
import random
from pathlib import Path
from uuid import uuid4

import gymnasium as gym
import numpy as np
import torch

from experiments.run import ROOT, write_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    paths = [*ROOT.glob('src/*.py'), *ROOT.glob('sumo_files/*')]
    paths += [ROOT / 'experiments' / name for name in ('run.py', 'long_run.py', 'checkpoints.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths) if p.is_file()}


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    write_json(pending, value)
    with pending.open('rb') as stream:
        os.fsync(stream.fileno())
    pending.replace(path)


def get_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available():
            raise RuntimeError('Checkpoint has CUDA RNG state but CUDA is unavailable.')
        torch.cuda.set_rng_state_all(state['cuda'])


def plain_state(obj, excluded):
    return copy.deepcopy({k: v for k, v in vars(obj).items() if k not in excluded})


class EpisodeReplay(gym.Wrapper):
    """Records actions actually delivered to SUMO, without changing them."""

    def __init__(self, env):
        super().__init__(env)
        self.actions = []
        self.trace = hashlib.sha256()

    def record(self, observation, reward=0.0, terminated=False, truncated=False, message=''):
        self.trace.update(np.asarray(observation, dtype=np.float32).tobytes())
        self.trace.update(np.asarray([reward], dtype=np.float64).tobytes())
        self.trace.update(bytes([bool(terminated), bool(truncated)]))
        self.trace.update(message.encode())

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.actions = []
        self.trace = hashlib.sha256()
        self.record(observation)
        return observation, info

    def step(self, action):
        result = self.env.step(action)
        self.actions.append(np.asarray(action).copy())
        self.record(*result[:4], result[4].get('message', ''))
        return result

    def capture(self):
        import traci
        excluded = {'env', 'observation_space', 'action_space', '_action_space', '_observation_space'}
        return copy.deepcopy({'actions': self.actions, 'trace': self.trace.hexdigest(),
                'delay': {k: v for k, v in vars(self.env).items() if k not in excluded},
                'merging': {k: v for k, v in vars(self.env.unwrapped).items()
                            if k not in excluded | {'tripinfo_path'}},
                'space_rng': copy.deepcopy(self.action_space.np_random.bit_generator.state),
                'simulation_time': traci.simulation.getTime(),
                'vehicles': self.vehicle_state()})

    @staticmethod
    def vehicle_state():
        import traci
        return {v: (traci.vehicle.getPosition(v), traci.vehicle.getSpeed(v),
                    traci.vehicle.getLaneID(v)) for v in traci.vehicle.getIDList()}

    def restore(self, state):
        import traci
        delay, merging = self.env, self.env.unwrapped
        delay._delay_seed = state['delay']['_delay_seed']
        delay._delay_episode = state['delay']['_delay_episode'] - 1
        merging._root_seed = state['merging']['_root_seed']
        merging._episode_index = state['merging']['_episode_index'] - 1
        if min(delay._delay_episode, merging._episode_index) < 0:
            raise RuntimeError('Checkpoint was taken before the environment was reset.')
        observation, _ = self.reset()
        for action in state['actions']:
            observation, _, terminated, truncated, _ = self.step(action)
            if terminated or truncated:
                raise RuntimeError('Reconstruction unexpectedly terminated the saved episode.')
        if (self.trace.hexdigest() != state['trace'] or
                self.vehicle_state() != state['vehicles'] or
                traci.simulation.getTime() != state['simulation_time']):
            raise RuntimeError('SUMO episode reconstruction differs; refusing approximate resume.')
        # Restore even attributes left over from a previous episode by upstream reset.
        # The wrapper/base dicts are copied together to preserve shared array aliases.
        python_state = copy.deepcopy({'delay': state['delay'], 'merging': state['merging']})
        delay.__dict__.update(python_state['delay'])
        merging.__dict__.update(python_state['merging'])
        self.action_space.np_random.bit_generator.state = state['space_rng']
        return np.asarray(observation, dtype=np.float32)


def capture_env(model, traced):
    vec = model.get_env()
    monitor = vec.envs[0]
    return {'episode': traced.capture(),
            'model_action_rng': copy.deepcopy(model.action_space.np_random.bit_generator.state),
            'monitor': plain_state(monitor, {'env', 'results_writer', '_action_space', '_observation_space'}),
            'vector': plain_state(vec, {'envs', 'observation_space', 'action_space'})}


def restore_env(model, traced, state):
    obs = traced.restore(state['episode'])
    np.testing.assert_array_equal(obs, np.asarray(model._last_obs)[0])
    vec = model.get_env()
    vec.envs[0].__dict__.update(state['monitor'])
    vec.__dict__.update(state['vector'])
    # SAC.load deserializes a separate Box. In a fresh run the model and the
    # environment share that Box; preserve the alias so warmup draws advance the
    # very same generator after a restart as they do during continuous training.
    model.action_space = traced.action_space
    for module in (model.policy, model.actor, model.critic, model.critic_target):
        module.action_space = traced.action_space
    model.action_space.np_random.bit_generator.state = state['model_action_rng']


def save_checkpoint(model, traced, callback_state, directory, cfg, environment, lineage, expected_sources):
    """Publish a checkpoint only after every file has been flushed and hashed."""
    rng = get_rng()
    if source_hashes() != expected_sources:
        raise RuntimeError('Training source changed during this run; refusing a misleading checkpoint.')
    step = model.num_timesteps
    root = Path(directory) / 'checkpoints'
    root.mkdir(exist_ok=True)
    final = root / f'step-{step:09d}'
    if final.exists():
        raise RuntimeError(f'Refusing to overwrite checkpoint: {final}')
    pending = root / f'.step-{step:09d}-{uuid4().hex[:8]}.partial'
    pending.mkdir()
    try:
        model.save(pending / 'model.zip')
        model.save_replay_buffer(pending / 'replay-buffer.pkl')
        with (pending / 'state.pkl').open('wb') as stream:
            pickle.dump({'rng': rng, 'environment': capture_env(model, traced),
                         'callback': callback_state}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(pending / 'config.json', cfg)
        files = {}
        for path in pending.iterdir():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
            files[path.name] = digest(path)
        manifest = {'schema_version': 1, 'env_steps': step, 'gradient_updates': model._n_updates,
                    'lineage': lineage, 'environment': environment, 'source_hashes': expected_sources,
                    'restore_method': 'deterministic_episode_replay', 'files': files}
        atomic_json(pending / 'manifest.json', manifest)
        pending.rename(final)
        fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        atomic_json(Path(directory) / 'latest-checkpoint.json', {'path': str(final), 'env_steps': step})
        return final
    finally:
        set_rng(rng)


def validate_checkpoint(path, environment):
    path = Path(path).resolve()
    if path.name.endswith('.partial') or not (path / 'manifest.json').exists():
        raise RuntimeError('Incomplete checkpoint: missing committed manifest.')
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['schema_version'] != 1:
        raise RuntimeError('Unsupported checkpoint schema.')
    expected = {'model.zip', 'replay-buffer.pkl', 'state.pkl', 'config.json'}
    if set(manifest['files']) != expected:
        raise RuntimeError('Checkpoint file inventory is incomplete.')
    for name, expected_hash in manifest['files'].items():
        if not (path / name).is_file() or digest(path / name) != expected_hash:
            raise RuntimeError(f'Checkpoint checksum mismatch: {name}')
    if manifest['source_hashes'] != source_hashes():
        raise RuntimeError('Training source differs from checkpoint; restore its source snapshot first.')
    if manifest['environment'] != environment:
        raise RuntimeError('Runtime/package/device settings differ from checkpoint.')
    return manifest
