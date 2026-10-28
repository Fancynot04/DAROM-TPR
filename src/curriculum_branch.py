"""Contracts for the frozen seed-43 curriculum-switch continuation pair."""

import hashlib
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback

from src import exit_e76_d3_config as config
from src.wrapper import ExitStartLaneCurriculum


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parent_files(model_path, replay_path):
    """Reject a different parent before an environment or output is created."""
    hashes = {
        'parent_model_sha256': sha256_file(model_path),
        'parent_replay_sha256': sha256_file(replay_path),
    }
    if hashes['parent_model_sha256'] != config.CURRICULUM_BRANCH_MODEL_SHA256:
        raise ValueError('Curriculum branch requires the frozen seed43 100k model')
    if hashes['parent_replay_sha256'] != config.CURRICULUM_BRANCH_REPLAY_SHA256:
        raise ValueError('Curriculum branch requires the matching seed43 100k replay')
    return hashes


def prepare_branch(model, vec_env):
    """Restore absolute curriculum time; deliberately start a new episode."""
    replay = model.replay_buffer
    if (model.num_timesteps != config.CURRICULUM_BRANCH_START
            or model.seed != 43 or model.observation_space.shape != (134,)
            or model.gamma != 0.99 or model.batch_size != 512
            or model.learning_rate != 3e-5):
        raise ValueError('Parent learner does not match the branch contract')
    if (replay.full or replay.pos != config.CURRICULUM_BRANCH_START - 1
            or replay.buffer_size != 1_000_000 or replay.n_envs != 1
            or replay.observation_space.shape != (134,)):
        raise ValueError('Parent replay does not match the frozen 100k snapshot')
    curriculum = vec_env.envs[0]
    if not isinstance(curriculum, ExitStartLaneCurriculum):
        raise ValueError('Branch continuation requires the start-lane curriculum')
    curriculum.env_steps = config.CURRICULUM_BRANCH_START
    # SAC.load(force_reset=True) normally does this too. The explicit reset
    # prevents a saved pre-step observation from crossing SUMO episode worlds.
    model._last_obs = None


def prepare_empty_replay_branch(model, vec_env, warmup_steps=100):
    """Start E09 with restored learner state but a fresh replay and episode."""
    if (model.num_timesteps != 100_000 or model.seed != 43
            or model.observation_space.shape != (134,) or model.gamma != 0.99
            or model.batch_size != 512 or model.learning_rate != 3e-5):
        raise ValueError('E09 parent learner does not match the 100k contract')
    replay = model.replay_buffer
    if (replay is None or replay.size() != 0 or replay.buffer_size != 1_000_000
            or replay.n_envs != 1 or replay.observation_space.shape != (134,)):
        raise ValueError('E09 requires a new, empty compatible replay buffer')
    curriculum = vec_env.envs[0]
    if not isinstance(curriculum, ExitStartLaneCurriculum):
        raise ValueError('E09 requires the original start-lane curriculum')
    if not isinstance(warmup_steps, int) or warmup_steps < 1:
        raise ValueError('E09 warmup must be a positive number of new steps')
    curriculum.env_steps = model.num_timesteps
    model.learning_starts = model.num_timesteps + warmup_steps
    model._last_obs = None


def restored_learner_summary(model):
    """Fingerprint restored weights and optimizer tensors without serial IDs."""
    import torch

    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, dict):
            for key in sorted(value, key=repr):
                digest.update(repr(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode())

    visit([model.policy.state_dict(), model.actor.optimizer.state_dict(),
           model.critic.optimizer.state_dict(), model.ent_coef_optimizer.state_dict(),
           model.log_ent_coef])
    return {'learner_state_sha256': digest.hexdigest(),
            'restored_updates': int(model._n_updates),
            'restored_alpha': float(model.log_ent_coef.detach().exp().cpu()),
            'restored_replay_pos': int(model.replay_buffer.pos)}


class AbsoluteCheckpointCallback(CheckpointCallback):
    """Save only the registered absolute steps, including the parent replay."""

    def __init__(self, save_path, name_prefix):
        super().__init__(save_freq=1, save_path=save_path,
                         name_prefix=name_prefix, save_replay_buffer=True, verbose=2)

    def _on_step(self):
        if self.num_timesteps not in config.CURRICULUM_BRANCH_CHECKPOINTS:
            return True
        return super()._on_step()
