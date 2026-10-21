"""Check that the transfer report preserves paired and task-specific metrics."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.report_dual_transfer import (AGENTS, EXPECTED_SEEDS, METHODS, SAFETY,
                                      SCENARIOS, aggregate, load_reports,
                                      paired_difference, report_path)


def episode(seed, entry_message, exit_message, entry_return, exit_return,
            entry_speed=None):
    def outcome(message, merged):
        return {"message": message, "is_success": message == "Arrived",
                "ego_teleported": False, "merged": merged}

    return {"seed": seed, "outcomes": {
                AGENTS[0]: outcome(entry_message, False),
                AGENTS[1]: outcome(exit_message, True)},
            "returns": {AGENTS[0]: entry_return, AGENTS[1]: exit_return},
            "motion": {AGENTS[0]: {"mean_speed_mps": entry_speed,
                                   "mean_jerk_mps3": 0.0 if entry_speed is not None else None,
                                   "mean_abs_jerk_mps3": 2.0 if entry_speed is not None else None},
                       AGENTS[1]: {"mean_speed_mps": 20.0, "mean_jerk_mps3": 0.0,
                                   "mean_abs_jerk_mps3": 3.0}},
            "both_success": entry_message == exit_message == "Arrived",
            "any_collision": entry_message == "Collision" or exit_message == "Collision"}


class TransferReportTests(unittest.TestCase):
    def test_loader_rejects_unpaired_sumo_seeds_and_missing_motion(self):
        reports = {}
        for scenario in SCENARIOS:
            for safety in SAFETY:
                for method in METHODS:
                    empty_motion = {"mean_speed_mps": None, "mean_jerk_mps3": None,
                                    "mean_abs_jerk_mps3": None}
                    records = [{"seed": seed, "sumo_seed": seed + 1,
                                "motion": {ego: empty_motion for ego in AGENTS}}
                               for seed in EXPECTED_SEEDS]
                    training = {"scenario": "easy", "delay_mode": "uniform",
                                "training_safety": False, "experience_target_per_ego": 1_000_000,
                                "seed": 42, "actor_gru_input_mode": "paper_raw",
                                "experiences": [1_000_000, 1_000_000],
                                "actor_updates": [999_000, 999_000],
                                "critic_updates": [999_000, 999_000]}
                    reports[report_path(Path("/unused"), method, scenario, safety).name] = {
                        "method": method, "scenario": scenario, "safety": safety,
                        "delay_mode": "uniform", "policy_action": "deterministic_tanh_mean",
                        "seed_start": EXPECTED_SEEDS[0],
                        "training": training, "checkpoint_sha256": method,
                        "summary": {"episodes": 200,
                                    "agents": {ego: {"mean_speed_mps": None}
                                               for ego in AGENTS}},
                        "episodes": records,
                    }

        def read(path):
            return json.dumps(reports[path.name])

        with patch("pathlib.Path.read_text", autospec=True, side_effect=read):
            self.assertEqual(len(load_reports(Path("/unused"))), 12)
            bad = reports[report_path(Path("/unused"), "ctde", "easy", False).name]
            bad["episodes"][0]["sumo_seed"] = -1
            with self.assertRaisesRegex(ValueError, "SUMO seeds differ"):
                load_reports(Path("/unused"))
            bad["episodes"][0]["sumo_seed"] = EXPECTED_SEEDS[0] + 1
            del bad["episodes"][0]["motion"]
            with self.assertRaisesRegex(ValueError, "Motion samples missing"):
                load_reports(Path("/unused"))

    def test_paired_difference_resamples_seedwise_differences(self):
        result = paired_difference([1, 0, 1], [0, 0, 0])
        self.assertAlmostEqual(result["mean"], 2 / 3)
        self.assertEqual(result["n"], 3)
        constant = paired_difference([2, 4, 6], [1, 3, 5])
        self.assertEqual(constant["ci95"], [1.0, 1.0])

    def test_entry_and_exit_timeouts_stay_separate_and_motion_n_is_explicit(self):
        reports = {}
        for scenario in SCENARIOS:
            for safety in SAFETY:
                for method in METHODS:
                    first = episode(1, "Max Steps", "Arrived", 3.0, 10.0)
                    second = episode(2, "Arrived", "Max Steps", 5.0, 0.0,
                                     entry_speed=12.0)
                    if method == "ctde":
                        second = episode(2, "Max Steps", "Max Steps", 1.0, 0.0,
                                         entry_speed=10.0)
                    reports[scenario, safety, method] = {"episodes": [first, second]}
        summary = aggregate(reports)
        entry = next(row for row in summary["rows"] if row["scenario"] == "easy"
                     and not row["safety"] and row["method"] == "independent"
                     and row["ego"] == AGENTS[0])
        exit_ = next(row for row in summary["rows"] if row["scenario"] == "easy"
                     and not row["safety"] and row["method"] == "independent"
                     and row["ego"] == AGENTS[1])
        self.assertEqual(entry["metrics"]["timeout"]["count"], 1)
        self.assertEqual(entry["metrics"]["unmerged_timeout"]["count"], 1)
        self.assertEqual(exit_["metrics"]["timeout"]["count"], 1)
        self.assertEqual(entry["metrics"]["speed_mps"]["n"], 1)
        self.assertEqual(entry["metrics"]["speed_mps"]["mean"], 12.0)
        difference = next(row for row in summary["paired_differences"]
                          if row["scenario"] == "easy" and not row["safety"]
                          and row["ego"] == AGENTS[0])
        self.assertEqual(difference["metrics"]["success"]["mean"], 0.5)
        self.assertEqual(difference["metrics"]["speed_mps"]["n"], 1)
        self.assertEqual(difference["metrics"]["speed_mps"]["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
