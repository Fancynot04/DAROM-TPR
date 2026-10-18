"""Local regression checks; no simulator, network or output files required."""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from io import StringIO

import gymnasium as gym
import numpy as np
import torch

from experiments.run import Recorder, load_config, main
from src.delay_encoder import DelayAwareGRUEncoder
from src.merging import SetTrafficBehavior
from src.wrapper import DelayWrapper


class DummyTraffic(gym.Env):
    observation_space = gym.spaces.Box(-1000, 1000, (31, 3), np.float32)
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = np.ones((31, 3), dtype=np.float32)
        self.action = np.zeros(2, dtype=np.float32)
        return self.observation.copy(), {}

    def step(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.observation = self.observation + 1
        return self.observation.copy(), 0.0, False, False, {}


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_groups_forward_backward_and_input_boundary(self):
        env = DelayWrapper(DummyTraffic(), max_delay=20, mode='all', delay_seed=42)
        obs, _ = env.reset(seed=42)
        self.assertEqual(len(obs), 134)
        for name, group in load_config()['groups'].items():
            with self.subTest(group=name):
                encoder = DelayAwareGRUEncoder(
                    env.observation_space, act_dim=2, hist_len=group['hist_len'],
                    delay_dim=1, features_dim=group['features_dim'], gru_hidden=64)
                seen = []
                hook = encoder.gru.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
                x = torch.arange(4 * 134, dtype=torch.float32).reshape(4, 134) / 100
                x.requires_grad_()
                output = encoder(x)
                hook.remove()
                self.assertEqual(output.shape, (4, group['features_dim']))
                state_dim = group['expected_state_dim']
                torch.testing.assert_close(seen[0], x[:, state_dim:-1].reshape(4, group['hist_len'], 2))
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(float(x.grad.abs().sum()), 0)
                for parameter in encoder.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_delay_sequences_repeat_without_global_numpy_consumption(self):
        env = DelayWrapper(DummyTraffic(), max_delay=20, mode='all', delay_seed=42)

        def collect(seed):
            obs, _ = env.reset(seed=seed)
            rows = [np.asarray(obs)]
            delays = []
            for _ in range(50):
                obs, *_ = env.step(np.array([0.1, -0.2], dtype=np.float32))
                rows.append(np.asarray(obs))
                delays.append(env.current_observation_delay)
            return np.asarray(rows), delays

        before = np.random.get_state()
        first, delays = collect(42)
        repeated, repeated_delays = collect(42)
        _, changed_delays = collect(43)
        after = np.random.get_state()
        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(delays, repeated_delays)
        self.assertNotEqual(delays, changed_delays)
        self.assertTrue(all(0 <= delay <= 20 for delay in delays))
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_implicit_reset_advances_episode_randomness(self):
        env = DelayWrapper(DummyTraffic(), max_delay=20, mode='all', delay_seed=42)
        env.reset(seed=42)
        first = [env._sample_delay() for _ in range(30)]
        env.reset()
        second = [env._sample_delay() for _ in range(30)]
        env.reset(seed=42)
        repeated = [env._sample_delay() for _ in range(30)]
        self.assertNotEqual(first, second)
        self.assertEqual(first, repeated)

    def test_traffic_behavior_uses_supplied_rng_and_preserves_parameters(self):
        with patch('src.merging.traci.vehicle') as vehicle:
            first_rng = np.random.RandomState(31)
            expected_rng = np.random.RandomState(31)
            SetTrafficBehavior('f_2.0', first_rng)
            vehicle.setTau.assert_called_once_with('f_2.0', expected_rng.normal(0.7, 0.1))
            vehicle.setMaxSpeed.assert_called_once_with('f_2.0', expected_rng.uniform(8, 11))

    def test_tracking_preserves_environment_step_and_marks_verification(self):
        cfg = load_config()
        cfg.update(selected_group='D', training_seed=42, verification_only=True)
        cfg['tracking']['mode'] = 'online'
        sdk = MagicMock()
        sdk.init.return_value.id = 'verification-test'
        stream = StringIO()
        with patch.dict('sys.modules', {'wandb': sdk}), \
             patch('torch.utils.tensorboard.SummaryWriter') as writer, \
             patch.object(Path, 'open', return_value=stream), \
             patch('experiments.run.write_json'):
            recorder = Recorder(Path('/tmp/unused-tracking-test'), cfg, 'D-seed42')
            recorder.emit({'train/actor_loss': 1.2}, 200)
            recorder.emit({'eval/no_sc/success_rate': 0.5}, 200)
            self.assertEqual(sdk.init.call_args.kwargs['job_type'], 'verification')
            self.assertIn('verification', sdk.init.call_args.kwargs['group'])
            sdk.init.return_value.define_metric.assert_any_call('*', step_metric='env_steps')
            self.assertTrue(all(c.args[0]['env_steps'] == 200 for c in sdk.init.return_value.log.call_args_list))
            writer.return_value.add_scalar.assert_any_call('train/actor_loss', 1.2, 200)
            self.assertFalse(sdk.init.call_args.kwargs['save_code'])
            self.assertEqual(sdk.Settings.call_args.kwargs['console'], 'off')
            self.assertFalse(sdk.Settings.call_args.kwargs['x_save_requirements'])
            with self.assertRaises(RuntimeError):
                recorder.emit({'train/critic_loss': float('nan')}, 201)
            recorder.close(exit_code=1)
            sdk.init.return_value.finish.assert_called_once_with(exit_code=1)

    def test_smoke_requires_explicit_online_tracking(self):
        command = ['experiments.run', 'train', '--group', 'D', '--seed', '42',
                   '--smoke', '--smoke-steps', '512', '--output', '/tmp/unused-smoke-test']
        for extra, expected in [([], 'disabled'), (['--tracking', 'online'], 'online')]:
            with self.subTest(mode=expected), patch('sys.argv', command + extra), \
                 patch('experiments.run.train') as run:
                main()
                cfg = run.call_args.args[0]
                self.assertEqual(cfg['tracking']['mode'], expected)
                self.assertEqual(cfg['total_timesteps'], 512)
                self.assertEqual(cfg['evaluation']['steps'], [256, 512])


if __name__ == '__main__':
    unittest.main()
