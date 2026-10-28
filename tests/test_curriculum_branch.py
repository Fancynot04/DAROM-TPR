"""Guard the paired 100k continuation against fresh starts and clock drift."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np

from src import exit_e76_d3_config as config
from src.curriculum_branch import (AbsoluteCheckpointCallback, prepare_branch, prepare_empty_replay_branch,
                                   sha256_file, validate_parent_files)
from src.sync_exit_wandb import expected_run_name, validate_run
from src.wrapper import ExitStartLaneCurriculum
from train_sac_delay import parse_args


class FakeEnv(gym.Env):
    observation_space = gym.spaces.Box(-5000, 5000, shape=(134,), dtype=np.float32)
    action_space = gym.spaces.Box(-5, 5, shape=(2,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(134, dtype=np.float32), {'ego_start_lane': self.curriculum_depart_lane}

    def step(self, action):
        return np.zeros(134, dtype=np.float32), 0.0, False, False, {}


class CurriculumBranchTests(unittest.TestCase):
    def test_e09_empty_replay_warmup_and_absolute_curriculum(self):
        curriculum = ExitStartLaneCurriculum(FakeEnv(), config.URGENCY_CURRICULUM_STAGES, 43, 'unused')
        replay = SimpleNamespace(size=Mock(return_value=0), buffer_size=1000000,
                                 n_envs=1, observation_space=FakeEnv.observation_space)
        model = SimpleNamespace(num_timesteps=100000, seed=43, gamma=.99, batch_size=512,
                                learning_rate=3e-5, replay_buffer=replay,
                                observation_space=FakeEnv.observation_space, _last_obs='old')
        prepare_empty_replay_branch(model, SimpleNamespace(envs=[curriculum]))
        self.assertEqual(model.learning_starts, 100100)
        self.assertEqual(curriculum.env_steps, 100000)
        self.assertEqual(curriculum.reset()[1]['curriculum_allowed_lanes'], [1, 2, 3])
        self.assertIsNone(model._last_obs)
        replay.size.return_value = 1
        with self.assertRaises(ValueError):
            prepare_empty_replay_branch(model, SimpleNamespace(envs=[curriculum]))

    @classmethod
    def setUpClass(cls):
        cls.fixture_root = (Path(__file__).resolve().parents[1]
                            / 'tmp/ai/e76-d3/curriculum_switch/unit')
        cls.fixture_root.mkdir(parents=True, exist_ok=True)

    def test_pair_requires_parent_and_has_identical_training_contract(self):
        args_by_arm = []
        for phase in config.CURRICULUM_BRANCH_PHASES:
            argv = ['train_sac_delay.py', '--exit_phase', phase, '--seed', '43']
            with patch('sys.argv', argv), self.assertRaises(SystemExit):
                parse_args()
            with patch('sys.argv', argv + ['--load_path', 'parent.zip']), self.assertRaises(SystemExit):
                parse_args()
            with patch('sys.argv', argv + ['--load_path', 'parent.zip',
                                          '--replay_buffer_path', 'parent.pkl']):
                args_by_arm.append(parse_args())
        for args in args_by_arm:
            self.assertEqual((args.total_timesteps, args.gamma), (150000, .99))
            self.assertFalse(args.use_safety)
            self.assertFalse(args.nav_features)
            self.assertEqual(args.exit_reward_profile, config.URGENCY_REWARD_PROFILE)
        for field in ('seed', 'encoder_mode', 'gru_input_mode', 'delay_mode',
                      'task', 'scenario', 'total_timesteps', 'gamma', 'torch_threads'):
            self.assertEqual(getattr(args_by_arm[0], field), getattr(args_by_arm[1], field))

    def test_parent_hash_guard_rejects_wrong_model_or_replay(self):
        with TemporaryDirectory(dir=self.fixture_root) as folder:
            model = Path(folder) / 'parent.zip'
            replay = Path(folder) / 'parent.pkl'
            model.write_bytes(b'model')
            replay.write_bytes(b'replay')
            with self.assertRaises(ValueError):
                validate_parent_files(model, replay)
            with patch.object(config, 'CURRICULUM_BRANCH_MODEL_SHA256', sha256_file(model)):
                with self.assertRaises(ValueError):
                    validate_parent_files(model, replay)
                with patch.object(config, 'CURRICULUM_BRANCH_REPLAY_SHA256', sha256_file(replay)):
                    self.assertEqual(validate_parent_files(model, replay)['parent_model_sha256'],
                                     sha256_file(model))

    def test_resume_uses_absolute_curriculum_time_and_reset_only_stage(self):
        for phase, stages in config.CURRICULUM_BRANCH_PHASES.items():
            curriculum = ExitStartLaneCurriculum(FakeEnv(), stages, 43, 'unused')
            replay = SimpleNamespace(pos=99999, full=False, buffer_size=1000000,
                                     n_envs=1, observation_space=FakeEnv.observation_space)
            model = SimpleNamespace(num_timesteps=100000, seed=43, gamma=.99,
                                    batch_size=512, learning_rate=3e-5, replay_buffer=replay,
                                    observation_space=FakeEnv.observation_space, _last_obs='old')
            prepare_branch(model, SimpleNamespace(envs=[curriculum]))
            self.assertIs(model.replay_buffer, replay)
            self.assertIsNone(model._last_obs)
            self.assertEqual(curriculum.env_steps, 100000)
            allowed = (1, 2) if 'hold2' in phase else (1, 2, 3)
            _, info = curriculum.reset()
            self.assertEqual(tuple(info['curriculum_allowed_lanes']), allowed)
            self.assertEqual(info['curriculum_stage'], 2 if 'hold2' in phase else 3)
            self.assertEqual(info['curriculum_reset_step'], 100000)
            curriculum.env_steps = 150001
            _, _, _, _, info = curriculum.step(np.zeros(2))
            self.assertEqual(info['curriculum_reset_step'], 100000)
            self.assertEqual(set(curriculum.reset()[1]['curriculum_allowed_lanes']), set(allowed))
            model.num_timesteps = 50000
            with self.assertRaises(ValueError):
                prepare_branch(model, SimpleNamespace(envs=[curriculum]))
            model.num_timesteps = 100000
            replay.pos = 99998
            with self.assertRaises(ValueError):
                prepare_branch(model, SimpleNamespace(envs=[curriculum]))

    def test_absolute_checkpoint_schedule_saves_only_requested_models_and_replay(self):
        with TemporaryDirectory(dir=self.fixture_root) as folder:
            callback = AbsoluteCheckpointCallback(folder, 'branch')
            callback.model = SimpleNamespace(save=Mock(), save_replay_buffer=Mock(),
                                             replay_buffer=object(), _vec_normalize_env=None)
            for step in (100001, 105000, 110000, 115000, 120000, 125000, 130000, 150000):
                callback.num_timesteps = step
                callback.n_calls += 1
                self.assertTrue(callback._on_step())
            expected = [f'branch_{s}_steps.zip' for s in (110000, 125000, 150000)]
            self.assertEqual([Path(c.args[0]).name for c in callback.model.save.call_args_list], expected)
            self.assertEqual(callback.model.save_replay_buffer.call_count, 3)

    def test_forge_run_names_are_distinct_and_have_parent_provenance(self):
        names = set()
        for phase in config.CURRICULUM_BRANCH_PHASES:
            name = expected_run_name(phase, 43)
            names.add(name)
            formal = validate_run(phase, 43, name)
            self.assertEqual(formal['resume_start_step'], 100000)
            self.assertEqual(formal['continuation_budget'], 50000)
            self.assertEqual(formal['parent_model_sha256'], config.CURRICULUM_BRANCH_MODEL_SHA256)
        self.assertEqual(len(names), 2)


if __name__ == '__main__':
    unittest.main()
