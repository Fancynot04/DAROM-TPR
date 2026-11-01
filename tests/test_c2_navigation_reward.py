"""C2 changes only B3's final-unit reward, never its policy input."""

import io
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.c2_navigation_reward import C2Merging, navigation_contract, navigation_reward
from src.delay_encoder import DelayAwareGRUEncoder
from src.merging import Merging
from src.wrapper import DelayWrapper
from train_sac_delay import parse_args
from evaluate_sac_delay import evaluate, validate_checkpoint_setup


class C2NavigationRewardTests(unittest.TestCase):
    def make_base(self, cls=Merging):
        return cls(seed=42, scenario='easy', task='exit_e76_d3', reward_profile='B3')

    def test_final_unit_formula(self):
        for n in range(5):
            with self.subTest(n=n):
                self.assertAlmostEqual(navigation_reward(n), -0.01 * n)
        for invalid in (None, -1, 1.5):
            with self.assertRaises(ValueError):
                navigation_reward(invalid)

    def test_settled_pending_connector_and_reset_cache(self):
        env = self.make_base(C2Merging)
        try:
            settled = {'status': 'ok', 'valid_for_threshold': True,
                       'settled_minimum_lane_changes': 3, 'provisional': False}
            pending = {'status': 'ok', 'valid_for_threshold': False,
                       'minimum_lane_changes': 2, 'provisional': True, 'lane_id': 'E76_2'}
            connector = {'status': 'at_destination', 'valid_for_threshold': False,
                         'minimum_lane_changes': 0, 'provisional': True,
                         'road': ':J106_0', 'lane_id': ':J106_0_0'}
            with patch('src.c2_navigation_reward.observe_route_urgency',
                       side_effect=[settled, pending, connector, dict(settled,
                           settled_minimum_lane_changes=2)]):
                self.assertEqual(env._navigation_demand()[:3], (3, True, 'settled'))
                self.assertEqual(env._navigation_demand()[:3], (3, False, 'cached'))
                self.assertEqual(env._navigation_demand()[:3], (3, False, 'cached'))
                self.assertEqual(env._navigation_demand()[:3], (2, True, 'settled'))
            env.reset_episode_state()
            with patch('src.c2_navigation_reward.observe_route_urgency', return_value=pending):
                with self.assertRaisesRegex(ValueError, 'no settled demand cache'):
                    env._navigation_demand()
            destination = {'status': 'at_destination', 'valid_for_threshold': False,
                           'provisional': False, 'road': 'D3'}
            with patch('src.c2_navigation_reward.observe_route_urgency',
                       return_value=destination):
                self.assertEqual(env._navigation_demand()[:3], (0, True, 'destination'))
        finally:
            env.close()

    def test_terminal_event_reward_is_not_repeated(self):
        env = self.make_base(C2Merging)
        env.reset_episode_state()
        env._navigation_n = 2
        def base_step(self, was_on_road, close_on_done=True):
            self.reward = 49.97
            self.reward_components = {'base': -0.03, 'arrival': 50.0}
            return np.zeros((31, 3), dtype=np.float32), self.reward, True, False, {
                'message': 'Arrived', 'reward_components': self.reward_components.copy()}
        try:
            with patch.object(Merging, 'collect_step', base_step):
                _, reward, terminated, truncated, info = env.collect_step(True)
            self.assertEqual((terminated, truncated), (True, False))
            self.assertEqual(info['reward_components']['navigation'], 0.0)
            self.assertEqual(info['reward_components']['arrival'], 50.0)
            self.assertAlmostEqual(reward, 49.97)
            self.assertAlmostEqual(sum(info['reward_components'].values()), reward)
            self.assertEqual(info['navigation_reward_total'], 0.0)
        finally:
            env.close()

    def test_fixed_action_matches_b3_except_navigation(self):
        traces = []
        for env_type in (Merging, C2Merging):
            env = DelayWrapper(self.make_base(env_type), max_delay=20, mode='all',
                               delay_mode='uniform', seed=42)
            rows = []
            try:
                observation, reset_info = env.reset(seed=42)
                self.assertEqual(observation.shape, (134,))
                if env_type is C2Merging:
                    self.assertTrue(reset_info['navigation_valid'])
                for _ in range(100):
                    observation, reward, terminated, truncated, info = env.step([0, -5])
                    rows.append((observation.copy(), reward, terminated, truncated, info))
                    if terminated or truncated:
                        break
            finally:
                env.close()
            traces.append(rows)
        self.assertEqual(len(traces[0]), len(traces[1]))
        self.assertGreater(len(traces[0]), 20)
        pending_steps = 0
        cumulative = 0.0
        for b3, c2 in zip(*traces):
            np.testing.assert_array_equal(b3[0], c2[0])
            self.assertEqual(b3[2:4], c2[2:4])
            self.assertEqual(b3[4]['message'], c2[4]['message'])
            components = c2[4]['reward_components']
            self.assertEqual({k: v for k, v in components.items() if k != 'navigation'},
                             b3[4]['reward_components'])
            self.assertEqual(components['stage'], 0.0)
            self.assertNotIn('potential', components)
            self.assertAlmostEqual(c2[1] - b3[1], components['navigation'])
            self.assertAlmostEqual(c2[1], sum(components.values()))
            cumulative += components['navigation']
            self.assertAlmostEqual(c2[4]['navigation_reward_total'], cumulative)
            pending_steps += int(c2[4]['navigation_source'] == 'cached')
        self.assertGreater(pending_steps, 0)

    def test_training_and_checkpoint_keep_b3_encoder(self):
        with patch('sys.argv', ['train_sac_delay.py', '--exit_phase', 'basic_reward_200k',
                                '--reward_profile', 'B3', '--c2-nav-reward', '--seed', '42',
                                '--wandb_mode', 'disabled']):
            args = parse_args()
        self.assertEqual((args.total_timesteps, args.device), (200_000, 'cuda'))
        self.assertEqual(args.save_dir, 'models_experiments/c2_nav_reward/C2_seed42')
        self.assertFalse(args.use_safety)
        with patch('sys.argv', ['train_sac_delay.py', '--exit_phase', 'basic_reward_200k',
                                '--reward_profile', 'B3', '--c2-nav-reward',
                                '--load_path', 'old.zip']):
            with patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    parse_args()
        model = Mock()
        model.antwerp_env_version = 2
        model.antwerp_task = 'exit_e76_d3'
        model.ablation = 'full'
        model.exit_potential_beta = 0.0
        model.exit_reward_profile = 'B3'
        model.exit_training_config = {
            'seed': 42, 'total_timesteps': 200_000, 'task': 'exit_e76_d3',
            'scenario': 'easy', 'delay_mode': 'uniform', 'max_delay': 20,
            'use_safety': False, 'encoder_mode': 'GRU', 'gru_input_mode': 'paper_raw',
            'ablation': 'full', 'reward_profile': 'B3', 'potential_beta': 0.0,
            'c2_nav_reward': True,
            'sac': {'learning_rate': 3e-5, 'batch_size': 512, 'buffer_size': 1_000_000,
                    'learning_starts': 100, 'gamma': 0.99}}
        model.num_timesteps = 200_000
        model.observation_space.shape = (134,)
        model.policy_kwargs = {'features_extractor_class': DelayAwareGRUEncoder}
        model.c2_navigation_contract = json.loads(json.dumps(navigation_contract()))
        validate_checkpoint_setup(model, 'full', False, 'exit_e76_d3', 'B3',
                                  c2_nav_reward=True)
        with self.assertRaises(ValueError):
            validate_checkpoint_setup(model, 'full', False, 'exit_e76_d3', 'B3')
        model.c2_navigation_contract['penalty_per_change'] = 0.02
        with self.assertRaises(ValueError):
            validate_checkpoint_setup(model, 'full', False, 'exit_e76_d3', 'B3',
                                      c2_nav_reward=True)

    def test_evaluation_trace_and_episode_navigation_total(self):
        model = Mock()
        model.predict.return_value = (np.zeros(2, dtype=np.float32), None)
        env = Mock()
        env.reset.return_value = (np.zeros(134, dtype=np.float32), {'ego_start_lane': 2})
        info = {'message': 'Arrived', 'ego_teleported': False,
                'stage_reached': True, 'road': 'D3', 'lane_index': 0,
                'reward_components': {'base': -0.03, 'arrival': 50.0,
                                      'navigation': 0.0},
                'navigation_n': 2, 'navigation_valid': False,
                'navigation_source': 'terminal', 'navigation_reward_total': -0.24,
                'distance_to_d3': 0.0}
        env.step.return_value = (np.zeros(134, dtype=np.float32), 49.97, True, False, info)
        sink = io.StringIO()
        result = evaluate(model, env, 1, 200001, task='exit_e76_d3',
                          c2_nav_reward=True, trace_sink=sink)
        self.assertEqual(result['episode_results'][0]['navigation_reward_total'], -0.24)
        self.assertEqual(json.loads(sink.getvalue())['evaluation_seed'], 200001)


if __name__ == '__main__':
    unittest.main()
