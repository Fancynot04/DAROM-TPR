"""Offline contract checks for full-budget W&B metric mirroring."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.sync_dual_wandb import (evaluation_metrics, initial_state,
                                 training_metric, verify_formal_config)


class DualWandbSyncTests(unittest.TestCase):
    def test_pilot_and_wrong_conditions_are_rejected(self):
        approved = {"mode": "independent", "scenario": "easy",
                    "experience_target_per_ego": 1_000_000, "seed": 42,
                    "delay_mode": "uniform", "delay_steps": [0, 20],
                    "training_safety": False}
        with patch.object(Path, "read_text", return_value=json.dumps(approved)):
            self.assertEqual(verify_formal_config(Path("unused"), "independent"), approved)
        for change in ({"experience_target_per_ego": 50_000},
                       {"training_safety": True}, {"delay_steps": [0, 0]}):
            with patch.object(Path, "read_text",
                              return_value=json.dumps({**approved, **change})):
                with self.assertRaises(ValueError):
                    verify_formal_config(Path("unused"), "independent")

    def test_training_metrics_preserve_per_ego_learning_and_rolling_outcomes(self):
        state = initial_state("ctde")
        progress = {"event": "progress", "scene_steps": 1000,
                    "experiences": [700, 1000], "actor_updates": [601, 901],
                    "critic_updates": [601, 901], "alpha": [.8, .7],
                    "latest_losses": [{"actor": 2, "critic": 3, "max_abs_target": 4},
                                      {"actor": 5, "critic": 6, "max_abs_target": 7}]}
        row = training_metric(progress, state)
        self.assertEqual(row["train/entry_experiences"], 700)
        self.assertEqual(row["train/exit_critic"], 6)
        for i in range(12):
            row = training_metric({"event": "episode", "agent": "ego_entry",
                                   "experience": i + 1, "return": float(i),
                                   "success": i >= 2}, state)
        self.assertEqual(row["episode/entry_success_rate10"], 1.0)
        self.assertEqual(row["episode/entry_return_mean10"], 6.5)
        self.assertEqual(len(state["recent"]["ego_entry"]["return"]), 10)

    def test_evaluation_requires_paired_200_seed_protocol(self):
        episodes = [{"seed": 700_001 + i,
                     "returns": {"ego_entry": 1.0, "ego_exit": -2.0},
                     "outcomes": {"ego_entry": {"is_success": True},
                                  "ego_exit": {"is_success": False}},
                     "both_success": False, "any_collision": False}
                    for i in range(200)]
        report = {"method": "independent", "scenario": "easy", "seed_start": 700_001,
                  "training": {"experience_target_per_ego": 1_000_000, "seed": 42,
                               "delay_mode": "uniform", "training_safety": False,
                               "experiences": [1_000_000, 1_000_000]},
                  "summary": {"episodes": 200,
                              "agents": {"ego_entry": {"success_rate": 1.0, "mean_return": 1.0},
                                         "ego_exit": {"success_rate": 0.0, "mean_return": -2.0}},
                              "both_success_rate": 0.0, "scene_collision_rate": 0.0},
                  "episodes": episodes}
        rows, totals = evaluation_metrics(report, "independent", "easy")
        self.assertEqual(len(rows), 200)
        self.assertEqual(rows[-1]["eval_easy/episode"], 200)
        self.assertEqual(totals["eval_easy/entry_success_rate"], 1.0)
        report["episodes"][99]["seed"] += 1
        with self.assertRaises(ValueError):
            evaluation_metrics(report, "independent", "easy")


if __name__ == "__main__":
    unittest.main()
