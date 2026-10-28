"""Tests for E09 controller, route variants and pairing contracts."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.e09_evaluate import cases
from src.merging import ANTWERP_DIR, e09_evaluation_route_file


class E09Contracts(unittest.TestCase):
    def test_fixed_cases_pair_and_lane_coverage(self):
        for origin, lane_count in (('E76', 5), ('E77', 6), ('O1', 3)):
            rows = cases(20, origin)
            self.assertEqual(len({row['seed'] for row in rows}), 20)
            self.assertEqual({row['lane'] for row in rows}, set(range(lane_count)))
            self.assertEqual(rows, cases(20, origin))

    def test_traffic_unchanged_and_ego_origin_lane_rewritten(self):
        source = ANTWERP_DIR / 'Antwerp_easy_exit_e76_d3.rou.xml'
        original = source.read_text()
        root = Path(__file__).resolve().parents[1] / 'tmp/ai/e09/test-routes'
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root) as directory:
            for origin, lane in (('E76', 4), ('E77', 5), ('O1', 2)):
                variant = e09_evaluation_route_file(source, origin, lane, directory).read_text()
                expected = original.replace('from="E76" to="D3" departLane="random"',
                                            f'from="{origin}" to="D3" departLane="{lane}"')
                self.assertEqual(variant, expected)
            with self.assertRaises(ValueError):
                e09_evaluation_route_file(source, 'O1', 3, directory)
