import unittest

import numpy as np

from src.e0_reachability import DEFAULT_LANES, audit_transition
from src.merging import parseAction


class E0ReachabilityUnitTests(unittest.TestCase):
    def test_e0_uses_the_existing_continuous_action_interface(self):
        self.assertEqual(DEFAULT_LANES, (0, 1, 2, 3, 4))
        self.assertEqual(parseAction(np.asarray([2.6, -5.0])), [2.6, -1])
        self.assertEqual(parseAction(np.asarray([2.6, 0.0])), [2.6, 0])
        self.assertEqual(parseAction(np.asarray([2.6, 5.0])), [2.6, 1])

    def test_b0_audit_accepts_only_the_contract_rewards(self):
        tracker = {"_meta": {"weighted_total": 50.0}}
        audit_transition(1, 0.0, False, False, {"message": ""})
        audit_transition(2, 50.0, True, False,
                         {"message": "Arrived", "reward_episode_summary": tracker})
        tracker["_meta"]["weighted_total"] = -100.0
        for event in ("Collision", "Ego Teleported", "Missed Exit", "Timeout"):
            audit_transition(2, -100.0, True, False,
                             {"message": event, "reward_episode_summary": tracker})

    def test_b0_audit_rejects_dense_reward_and_truncation(self):
        with self.assertRaisesRegex(RuntimeError, "non-terminal"):
            audit_transition(1, 0.1, False, False, {"message": ""})
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            audit_transition(1, -100.0, True, True,
                             {"message": "Timeout", "reward_episode_summary": {"_meta": {"weighted_total": -100.0}}})


if __name__ == "__main__":
    unittest.main()
