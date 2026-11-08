import copy
import io
import inspect
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer

import evaluate_sac_delay as evaluation
import train_sac_delay as training
from src import exit_progress_reward
from src.merging import (ANTWERP_ENV_VERSION, Merging,
                         classify_braking_diagnostics, control_step_timeout,
                         select_terminal_event)
from src.rewards.config import load_hydra_config
from src.rewards.registry import RewardContext, settle, settle_terms
from src.rewards.terms import RewardTracker
from src.wrapper import DelayWrapper, SafetyWrapper


def checkpoint_model(config):
    return SimpleNamespace(
        exit_reward_profile=config['name'],
        exit_reward_config=config,
        exit_reward_config_sha256=exit_progress_reward.reward_identity(config),
        antwerp_env_version=ANTWERP_ENV_VERSION,
        antwerp_task='exit_e76_d3',
        observation_space=spaces.Box(-1, 1, shape=(134,), dtype=np.float32),
        policy_kwargs={'features_extractor_kwargs': {'nav_dim': 0}},
        training_safety=False,
        gamma=0.99,
    )


class EvaluationRewardContractTests(unittest.TestCase):
    def setUp(self):
        self.default = exit_progress_reward.load_experiment()

    def test_policy_return_changes_but_b0_reference_does_not(self):
        modified = copy.deepcopy(self.default)
        modified['terms']['success']['weight'] = 0.5
        events = (None, None, 'Arrived')
        default_return = sum(settle(self.default['name'], RewardContext(event, self.default))[0]
                             for event in events)
        modified_return = sum(settle(modified['name'], RewardContext(event, modified))[0]
                              for event in events)
        reference = sum(evaluation.b0_reference_reward(event) for event in events)
        self.assertEqual(default_return, 50.0)
        self.assertEqual(modified_return, 25.0)
        self.assertEqual(reference, 50.0)

    def test_success_and_reference_are_independent_of_reward_switch(self):
        disabled = copy.deepcopy(self.default)
        disabled['terms']['success']['enabled'] = False
        policy_reward = settle(disabled['name'], RewardContext('Arrived', disabled))[0]
        self.assertEqual(policy_reward, 0.0)
        self.assertEqual(evaluation.b0_reference_reward('Arrived'), 50.0)
        self.assertEqual(int('Arrived' == 'Arrived'), 1)

    def test_checkpoint_reward_provenance_is_mandatory_and_authenticated(self):
        model = checkpoint_model(self.default)
        profile, config, digest = evaluation.checkpoint_reward_setup(model)
        self.assertEqual(profile, 'events_only_v1')
        self.assertEqual(config, self.default)
        self.assertEqual(digest, model.exit_reward_config_sha256)
        del model.exit_reward_config
        with self.assertRaisesRegex(ValueError, 'missing required reward provenance'):
            evaluation.checkpoint_reward_setup(model)

        model = checkpoint_model(self.default)
        model.exit_reward_config_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            evaluation.checkpoint_reward_setup(model)

        model = checkpoint_model(self.default)
        model.exit_reward_profile = 'wrong'
        with self.assertRaisesRegex(ValueError, 'profile and configuration disagree'):
            evaluation.checkpoint_reward_setup(model)

    def test_evaluation_environment_receives_checkpoint_config(self):
        modified = copy.deepcopy(self.default)
        modified['terms']['collision']['weight'] = 0.25
        env = evaluation.make_env(43, 'easy', 'uniform', False, 'full',
                                  exit_reward_profile=modified['name'],
                                  reward_config=modified, reward_gamma=0.97)
        try:
            self.assertEqual(env.unwrapped.reward_config, modified)
            self.assertEqual(env.unwrapped.reward_tracker.gamma, 0.97)
        finally:
            env.close()


class TrainingSafetyAndMetricTests(unittest.TestCase):
    def test_cli_hydra_and_evaluation_defaults_are_aligned(self):
        with patch.object(sys, 'argv', ['train_sac_delay.py', '--wandb_mode', 'disabled']):
            args = training.parse_args()
        self.assertEqual((args.scenario, args.seed, args.gamma,
                          args.total_timesteps, args.checkpoint_freq),
                         ('easy', 42, 0.999, 200_000, 50_000))
        composed, reward, _, _ = load_hydra_config()
        self.assertEqual((composed.experiment.scenario, composed.experiment.seed,
                          composed.experiment.gamma,
                          composed.experiment.total_timesteps,
                          composed.experiment.checkpoint_freq),
                         ('easy', 42, 0.999, 200_000, 50_000))
        self.assertEqual(reward['name'], 'events_only_v1')
        evaluation_args = evaluation.parse_args(['checkpoint.zip'])
        self.assertEqual((evaluation_args.scenario, evaluation_args.episodes,
                          evaluation_args.seeds), ('easy', 50, [200001]))

    def test_git_provenance_records_commit_dirty_flag_and_porcelain(self):
        with patch('train_sac_delay.subprocess.check_output',
                   side_effect=['abc123\n', ' M train_sac_delay.py\n']):
            provenance = training.read_git_provenance()
        self.assertEqual(provenance, {
            'git_commit': 'abc123',
            'git_dirty': True,
            'git_status_porcelain': ' M train_sac_delay.py',
        })

    def test_default_training_is_safety_off_and_safety_flag_fails(self):
        with patch.object(sys, 'argv', ['train_sac_delay.py', '--wandb_mode', 'disabled']):
            self.assertFalse(training.parse_args().use_safety)
        with patch.object(sys, 'argv', ['train_sac_delay.py', '--safety']), \
                patch('sys.stderr', new=io.StringIO()) as stderr:
            with self.assertRaises(SystemExit):
                training.parse_args()
            self.assertIn('require Safety off', stderr.getvalue())

    def test_training_environment_cannot_contain_safety_wrapper(self):
        env = training.make_env(43, 'easy', 'uniform', 20, False)
        try:
            self.assertIsInstance(env, DelayWrapper)
            self.assertNotIsInstance(env, SafetyWrapper)
            self.assertFalse(env.unwrapped.runSafetyCheck)
        finally:
            env.close()
        with self.assertRaisesRegex(ValueError, 'SafetyWrapper is forbidden'):
            training.make_env(43, 'easy', 'uniform', 20, True)

    def test_wandb_metric_schema_uses_env_steps(self):
        run = Mock()
        training.configure_wandb_metrics(run)
        run.define_metric.assert_any_call('env_steps')
        for namespace in ('train/*', 'reward/*', 'diagnostic/*'):
            run.define_metric.assert_any_call(namespace, step_metric='env_steps')
        source = inspect.getsource(training.main)
        self.assertIn('sync_tensorboard=False', source)
        self.assertIn('gradient_save_freq=0', source)

    def test_emergency_metrics_come_from_info_not_terminal_outcome(self):
        callback = training.EpisodeRewardCallback()
        callback.num_timesteps = 1
        callback._log_metrics = Mock()
        callback.locals = {
            'rewards': np.array([0.0]),
            'dones': np.array([True]),
            'infos': [{
                'message': 'Timeout', 'stage_name': 'exit', 'merged': False,
                'stage_reached': False, 'is_success': False,
                'ego_teleported': False, 'ego_start_lane': None,
                'exit_actor_request_direction': 0, 'potential_reward': 0.0,
                'urgency_reward': 0.0, 'reward_episode_summary': None,
                'sumo_emergency_stopping_steps': 3,
                'sumo_emergency_stopping_episode': True,
                'hard_braking_threshold_mps2': -4.5,
                'hard_braking_steps': 2, 'hard_braking_episode': True,
            }],
        }
        self.assertTrue(callback._on_step())
        metrics = callback._log_metrics.call_args.args[0]
        self.assertEqual(metrics['train/emergency_stopping'], 1)
        self.assertEqual(metrics['train/sumo_emergency_stopping_steps_episode'], 3)
        self.assertEqual(metrics['train/sumo_emergency_stopping_episode_rate'], 1.0)
        self.assertEqual(metrics['train/hard_braking_steps_episode'], 2)
        self.assertEqual(metrics['train/hard_braking_episode_rate'], 1.0)

    @patch('train_sac_delay.wandb.log')
    def test_two_episode_logs_keep_global_monotonic_steps(self, log):
        callback = training.EpisodeRewardCallback()
        callback.num_timesteps = 4000
        callback._log_metrics({'train/episode_reward': -100,
                               'reward/total/weighted_sum': -100})
        callback.num_timesteps = 8000
        callback._log_metrics({'train/episode_reward': 50,
                               'reward/total/weighted_sum': 50})
        self.assertEqual([call.args[0]['env_steps'] for call in log.call_args_list],
                         [4000, 8000])
        callback.num_timesteps = 7999
        with self.assertRaisesRegex(RuntimeError, 'globally monotonic'):
            callback._log_metrics({'train/episode_reward': 0})


class TerminalContractTests(unittest.TestCase):
    def test_sumo_emergency_and_hard_braking_are_distinct_and_reward_free(self):
        self.assertEqual(classify_braking_diagnostics(True, -1.0), (True, False))
        self.assertEqual(classify_braking_diagnostics(False, -5.0), (False, True))
        config = exit_progress_reward.load_experiment()
        reward, components = settle(config['name'], RewardContext(None, config))
        self.assertEqual(reward, 0.0)
        self.assertTrue(all(value == 0.0 for value in components.values()))

    def test_individual_terminal_events_and_nonterminal_diagnostics(self):
        cases = {
            'Collision': dict(collision=True),
            'Ego Teleported': dict(ego_teleported=True),
            'Missed Exit': dict(missed_exit=True),
            'Arrived': dict(arrived=True),
            'Timeout': dict(timeout=True),
        }
        defaults = dict(collision=False, ego_teleported=False, missed_exit=False,
                        arrived=False, timeout=False)
        for expected, active in cases.items():
            with self.subTest(expected=expected):
                event = select_terminal_event(**(defaults | active))
                self.assertEqual(event, expected)
                self.assertEqual((event is not None, False), (True, False))
                expected_reward = 50.0 if expected == 'Arrived' else -100.0
                self.assertEqual(evaluation.b0_reference_reward(expected), expected_reward)
        event = select_terminal_event(**defaults)
        self.assertIsNone(event)
        self.assertEqual((event is not None, False), (False, False))
        self.assertEqual(evaluation.b0_reference_reward(None), 0.0)
        self.assertEqual(evaluation.b0_reference_reward('Emergency Stopping'), 0.0)

    def test_terminal_priority(self):
        all_true = dict(collision=True, ego_teleported=True, missed_exit=True,
                        arrived=True, timeout=True)
        self.assertEqual(select_terminal_event(**all_true), 'Collision')
        all_true['collision'] = False
        self.assertEqual(select_terminal_event(**all_true), 'Ego Teleported')
        all_true['ego_teleported'] = False
        self.assertEqual(select_terminal_event(**all_true), 'Missed Exit')
        all_true['missed_exit'] = False
        self.assertEqual(select_terminal_event(**all_true), 'Arrived')
        all_true['arrived'] = False
        self.assertEqual(select_terminal_event(**all_true), 'Timeout')

    def test_timeout_is_exactly_control_step_4000(self):
        self.assertFalse(control_step_timeout(3999, 4000))
        self.assertTrue(control_step_timeout(4000, 4000))
        self.assertTrue(control_step_timeout(4001, 4000))

    def test_terminal_reward_is_single_use_and_tracker_resets(self):
        config = exit_progress_reward.load_experiment()
        tracker = RewardTracker(0.99)
        for event in (None, None, 'Arrived'):
            terms = settle_terms(config['name'], RewardContext(event, config))
            tracker.add(terms, sum(term.weighted for term in terms))
        summary = tracker.summary(terminated=True)
        self.assertEqual(summary['success']['weighted_sum'], 50.0)
        self.assertEqual(summary['_meta']['weighted_total'], 50.0)
        tracker.reset()
        self.assertEqual(tracker.summary()['_meta']['steps'], 0)

        env = Merging(seed=43, scenario='easy')
        env.done = True
        with self.assertRaisesRegex(RuntimeError, 'reset'):
            env.prepare_step([0.0, 0.0])

    def test_timeout_terminal_disables_replay_bootstrap(self):
        observation = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        action = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        buffer = ReplayBuffer(4, observation, action, device='cpu', n_envs=1,
                              handle_timeout_termination=True)
        buffer.add(np.zeros((1, 2), dtype=np.float32),
                   np.ones((1, 2), dtype=np.float32),
                   np.zeros((1, 1), dtype=np.float32),
                   np.array([-100.0], dtype=np.float32), np.array([True]), [{}])
        self.assertEqual(float(buffer.timeouts[0, 0]), 0.0)
        sample = buffer._get_samples(np.array([0]))
        self.assertEqual(float(sample.dones.cpu().numpy()[0, 0]), 1.0)


if __name__ == '__main__':
    unittest.main()
