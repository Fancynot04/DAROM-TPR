import copy
import unittest

from src.rewards.config import load_hydra_config, validate_reward_config
from src.rewards.terms import RewardTerm, RewardTracker


class ModularRewardTests(unittest.TestCase):
    def test_default_hydra_config_and_override_hash(self):
        _, default, default_hash, yaml_text = load_hydra_config()
        _, disabled, disabled_hash, _ = load_hydra_config(
            overrides=['reward.terms.collision.enabled=false'])
        self.assertNotEqual(default_hash, disabled_hash)
        self.assertIn('events_only_v1', yaml_text)
        self.assertTrue(default['terms']['collision']['enabled'])
        self.assertFalse(disabled['terms']['collision']['enabled'])

    def test_weight_and_enabled_are_distinct_but_payment_equivalent(self):
        term = RewardTerm('collision', -100.0, 0.5, True)
        disabled = RewardTerm('collision', -100.0, 0.5, False)
        self.assertEqual(term.weighted, -50.0)
        self.assertEqual(disabled.weighted, 0.0)

    def test_tracker_sums_discount_quantiles_and_resets(self):
        tracker = RewardTracker(gamma=0.9)
        tracker.add([RewardTerm('success', 0.0, 1.0, True, active=False)], 0.0)
        tracker.add([RewardTerm('success', 50.0, 1.0, True)], 50.0)
        summary = tracker.summary(terminated=True)
        self.assertEqual(summary['success']['raw_sum'], 50.0)
        self.assertEqual(summary['success']['weighted_sum'], 50.0)
        self.assertAlmostEqual(summary['success']['discounted_sum'], 45.0)
        self.assertEqual(summary['_meta']['steps'], 2)
        self.assertEqual(summary['success']['active_ratio'], 0.5)
        window = tracker.summary(start=1, end=2)
        self.assertEqual(window['success']['weighted_sum'], 50.0)
        self.assertEqual(window['_meta']['start'], 1)
        tracker.reset()
        self.assertEqual(tracker.summary()['_meta']['steps'], 0)

    def test_tracker_rejects_reward_mismatch_and_invalid_config(self):
        tracker = RewardTracker(0.99)
        with self.assertRaises(ValueError):
            tracker.add([RewardTerm('success', 50.0, 1.0, True)], 0.0)
        _, config, _, _ = load_hydra_config()
        bad = copy.deepcopy(config)
        bad['terms']['success']['value'] = 51
        with self.assertRaises(ValueError):
            validate_reward_config(bad)


if __name__ == '__main__':
    unittest.main()
