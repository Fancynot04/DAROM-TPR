import unittest

from scripts.compare_trajectory_preferences import choose as choose_v1
from scripts.compare_trajectory_preferences_v2_candidate import (
    audit_stable_zero,
    choose_v2,
)
from tests.test_trajectory_preferences import make


def stable_zero(trajectory, distance):
    trajectory.rows[0].update(route_state_settled='1', min_lane_changes_required='0',
                              route_remaining_to_d3_start_m=str(distance))
    return trajectory


class PreferenceRuleV2CandidateTest(unittest.TestCase):
    def test_p1_p2_p4_match_v1(self):
        pairs = [
            (make('a'), make('b', outcome='Collision')),
            (make('a', danger=0), make('b', danger=6)),
            (make('a', steps=100), make('b', steps=160)),
        ]
        for a, b in pairs:
            self.assertEqual(choose_v2(a, b, enable_stable_zero=False), choose_v1(a, b))

    def test_stable_completion_precedes_reverse_changes_when_enabled(self):
        a = stable_zero(make('a', away=2), 400)
        b = make('b', away=0)
        self.assertEqual(choose_v2(a, b, enable_stable_zero=True)[1:3], ('A', 'P3'))

    def test_earlier_completion_uses_50_m_threshold(self):
        a = stable_zero(make('a', away=2), 500)
        b = stable_zero(make('b', away=0), 449)
        self.assertEqual(choose_v2(a, b, enable_stable_zero=True)[1:3], ('A', 'P3'))
        b.rows[0]['route_remaining_to_d3_start_m'] = '451'
        self.assertEqual(choose_v2(a, b, enable_stable_zero=True)[1:3], ('B', 'P3'))

    def test_unreliable_marker_disables_stable_branches(self):
        a = stable_zero(make('a', away=2), 500)
        b = make('b', away=0)
        result = choose_v2(a, b, enable_stable_zero=False)
        self.assertEqual(result[1:3], ('B', 'P3'))

    def test_success_without_stable_zero_marks_metric_unreliable(self):
        success = make('success')
        success.rows[0].update(min_lane_changes_required='0', route_state_settled='0')
        audit = audit_stable_zero([success])
        self.assertFalse(audit['reliable_for_completion_preference'])
        self.assertEqual(audit['totals']['missing_but_has_unsettled_zero'], 1)


if __name__ == '__main__':
    unittest.main()
