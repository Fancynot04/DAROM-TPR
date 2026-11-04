"""Compatibility checks for composable reward settlement."""
import copy
import unittest

from src import exit_progress_reward as old
from src import secrm_reward
from src.rewards.registry import RewardContext, settle


class RewardRegistryTests(unittest.TestCase):
    def context(self, config, **changes):
        values = dict(outcome='', config=config['reward'], teleported=False,
                      progress_before=150.0, progress_after=140.0,
                      odometer_before=850.0, odometer_after=860.0,
                      initial_remaining=1000.0, lane_change=0.0,
                      secrm_before={'efficiency': -0.2, 'route_reward': -0.03})
        values.update(changes)
        return RewardContext(**values)

    def test_legacy_profiles_and_events(self):
        configs = [old.load_experiment(), old.load_experiment(directional=True)]
        configs.extend(old.load_experiment(phase=phase) for phase in old.SECRM_DEFAULTS
                       if phase != 'secrm_efficiency_directional')
        d = copy.deepcopy(old.load_experiment(phase='secrm_efficiency_route'))
        d['reward']['efficiency_weight'] = 0.0
        configs.append(d)
        for config in configs:
            profile = old.reward_profile(config)
            for outcome in ('', *old.EVENTS):
                for teleported in (False, True):
                    with self.subTest(profile=profile, outcome=outcome, teleported=teleported):
                        ctx = self.context(config, outcome=outcome, teleported=teleported,
                                           lane_change=3.0 if profile == old.DIRECTIONAL_PROFILE else 0.0)
                        total, parts = settle(profile, ctx)
                        _, event = old.event_reward(outcome, config['reward'])
                        if profile in secrm_reward.PROFILES:
                            expected = (0.0 if teleported else -0.2 * config['reward']['efficiency_weight'])
                            expected += (0.0 if teleported else -0.03 * config['reward'].get('route_weight', 0.0))
                            expected += event
                        else:
                            expected = (0.0 if teleported else old.progress_reward(150, 140, 850, 860, 1000, config['reward']))
                            expected += event + ctx.lane_change
                        self.assertAlmostEqual(total, expected)
                        self.assertAlmostEqual(total, sum(parts.values()))

    def test_efficiency_directional_composition_and_identity(self):
        config = old.load_experiment(phase='secrm_efficiency_directional')
        self.assertEqual(old.reward_profile(config), secrm_reward.DIRECTIONAL_PROFILE)
        self.assertEqual((config['total_timesteps'], config['checkpoint_freq']), (500_000, 250_000))
        self.assertNotIn('route_weight', config['reward'])
        self.assertNotIn('progress_per_meter', config['reward'])
        for lane_reward in (3.0, -6.0, 0.0):
            for outcome in ('', *old.EVENTS):
                ctx = self.context(config, outcome=outcome, lane_change=lane_reward)
                total, parts = settle(secrm_reward.DIRECTIONAL_PROFILE, ctx)
                _, event = old.event_reward(outcome, config['reward'])
                self.assertAlmostEqual(total, -0.2 + event + lane_reward)
                self.assertEqual(parts['route_following'], 0.0)
                self.assertEqual(parts['progress'], 0.0)
                self.assertEqual(parts['lane_change'], lane_reward)
                self.assertAlmostEqual(sum(parts.values()), total)
        bad = copy.deepcopy(config)
        bad['reward']['alpha'] = -1
        with self.assertRaises(ValueError):
            old.validate_experiment(bad)
        changed = copy.deepcopy(config)
        changed['reward']['beta'] = 7
        self.assertNotEqual(old.reward_identity(config['reward']), old.reward_identity(changed['reward']))
        self.assertNotEqual(old.reward_identity(config['reward']),
                            old.reward_identity(old.SECRM_DEFAULTS['secrm_efficiency']['reward']))

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            settle('unknown', self.context(old.load_experiment()))

    def test_replay_identity_remains_distinct(self):
        a = old.load_experiment(phase='secrm_efficiency')
        b = old.load_experiment(phase='secrm_efficiency_route')
        self.assertNotEqual(old.reward_identity(a['reward']), old.reward_identity(b['reward']))

if __name__ == '__main__':
    unittest.main()
