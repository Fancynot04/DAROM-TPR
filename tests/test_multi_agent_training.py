"""Meaningful SAC update and joint-replay checks; SUMO smoke is opt-in."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.multi_agent_sac import (AGENTS, CentralTwinQ, JointReplay,
                                 SACConfig, TwoAgentSAC)
from src.train_dual_sac import arguments, run
from src.evaluate_dual_sac import evaluate, parse_args as parse_dual_eval_args, summarize
from src.compare_dual import check_pair


class MultiAgentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_joint_replay_retains_valid_agent_indices_after_wraparound(self):
        replay = JointReplay(capacity=3, seed=42)
        zeros = np.zeros((2, 134), np.float32)
        for index, mask in enumerate(([1, 0], [1, 1], [0, 1], [0, 1])):
            obs = zeros.copy()
            obs[:, 0] = index
            replay.add(obs, np.zeros((2, 2), np.float32),
                       np.zeros(2, np.float32), obs, np.zeros(2, np.float32),
                       np.asarray(mask, np.float32), np.asarray(mask, np.float32))
        self.assertEqual(replay.size, 3)
        self.assertEqual(len(replay.valid_indices[0]), 1)
        self.assertEqual(len(replay.valid_indices[1]), 3)
        left = replay.sample(100, torch.device("cpu"), agent=0)
        self.assertTrue((left[0][:, 0, 0] == 1).all())
        right = replay.sample(100, torch.device("cpu"), agent=1)
        self.assertTrue((right[5][:, 1] == 1).all())

    def test_centralized_q_changes_when_partner_input_changes(self):
        torch.manual_seed(42)
        critic = CentralTwinQ()
        obs = torch.zeros(4, 2, 134)
        action = torch.zeros(4, 2, 2)
        mask = torch.ones(4, 2)
        original = critic(obs, action, mask)[0].detach().clone()
        action[:, 1, 0] = 0.8
        changed = critic(obs, action, mask)[0].detach()
        self.assertGreater((changed[:, 0] - original[:, 0]).abs().max().item(), 1e-8)
        mask[:, 1] = 0
        action[:, 1] = 0
        absent = critic(obs, action, mask)[0].detach()
        self.assertTrue(torch.isfinite(absent).all())

    def test_two_modes_update_both_actors_at_equal_valid_experience_budget(self):
        zero = np.zeros(134, np.float32)
        one = np.ones(134, np.float32)
        for mode in ("independent", "ctde"):
            with self.subTest(mode=mode):
                model = TwoAgentSAC(SACConfig(mode=mode, batch_size=4,
                                              replay_capacity=16, learning_starts=2))
                original = [next(actor.parameters()).detach().clone()
                            for actor in model.actors]
                for step in range(5):
                    before = {key: zero + step for key in AGENTS}
                    after = {key: one + step for key in AGENTS}
                    actions = {key: np.zeros(2, np.float32) for key in AGENTS}
                    rewards = {key: 1.0 for key in AGENTS}
                    ends = {key: step == 4 for key in AGENTS}
                    model.observe(before, actions, after, rewards, ends,
                                  dict.fromkeys(AGENTS, False), AGENTS if step < 4 else (), 5)
                self.assertEqual(model.experiences, [5, 5])
                self.assertEqual(model.actor_updates, [4, 4])
                self.assertEqual(model.critic_updates, [4, 4])
                for i in range(2):
                    self.assertFalse(torch.equal(next(model.actors[i].parameters()), original[i]))
                    self.assertTrue(np.isfinite(float(model.log_alpha[i].exp().detach())))

    def test_terminal_agent_stops_collecting_while_partner_continues(self):
        model = TwoAgentSAC(SACConfig(mode="ctde", batch_size=2,
                                      replay_capacity=8, learning_starts=2))
        obs = np.zeros(134, np.float32)
        actions = {key: np.zeros(2, np.float32) for key in AGENTS}
        rewards = {key: 0.0 for key in AGENTS}
        flags = dict.fromkeys(AGENTS, False)
        model.observe(dict.fromkeys(AGENTS, obs), actions,
                      dict.fromkeys(AGENTS, obs), rewards,
                      {AGENTS[0]: True, AGENTS[1]: False}, flags, (AGENTS[1],), 5)
        model.observe({AGENTS[1]: obs}, {AGENTS[1]: actions[AGENTS[1]]},
                      {AGENTS[1]: obs}, {AGENTS[1]: 0.0},
                      {AGENTS[1]: True}, {AGENTS[1]: False}, (), 5)
        self.assertEqual(model.experiences, [1, 2])
        self.assertEqual(len(model.replay.valid_indices[0]), 1)
        self.assertEqual(len(model.replay.valid_indices[1]), 2)

    def test_evaluation_summary_counts_joint_outcome_and_collision(self):
        def case(entry, exit_, collision):
            return {"outcomes": {
                AGENTS[0]: {"is_success": entry, "message": "Collision" if collision else "Arrived",
                            "ego_teleported": False},
                AGENTS[1]: {"is_success": exit_, "message": "Arrived",
                            "ego_teleported": not exit_}},
                "returns": {AGENTS[0]: 10.0, AGENTS[1]: -2.0},
                "both_success": entry and exit_, "any_collision": collision}
        result = summarize([case(True, True, False), case(False, False, True)])
        self.assertEqual(result["both_success_rate"], .5)
        self.assertEqual(result["scene_collision_rate"], .5)
        self.assertEqual(result["agents"][AGENTS[0]]["mean_return"], 10)

    def test_safety_evaluation_uses_controller_and_records_same_seed(self):
        env = Mock()
        env.finished = True
        env.reset.return_value = ({}, {"sumo_seed": 700001})
        with patch("src.evaluate_dual_sac.DualMerging", return_value=env) as constructor:
            records = evaluate(Mock(), "easy", 1, 700001, safety=True)
        constructor.assert_called_once_with(scenario="easy", safety=True,
                                            delay_mode="uniform")
        env.reset.assert_called_once_with(seed=700001)
        env.close.assert_called_once()
        self.assertEqual(records[0]["seed"], 700001)

    def test_safety_cli_defaults_off_and_can_be_enabled(self):
        with patch("pathlib.Path.is_file", return_value=True):
            basic = parse_dual_eval_args(["checkpoint.pt", "--output", "off.json"])
            enabled = parse_dual_eval_args(["checkpoint.pt", "--output", "on.json",
                                            "--safety"])
        self.assertFalse(basic.safety)
        self.assertTrue(enabled.safety)

    def test_comparison_rejects_unequal_training_or_evaluation_budgets(self):
        template = {"method": "independent", "scenario": "easy",
                    "delay_mode": "uniform", "safety": False, "seed_start": 700001,
                    "summary": {"episodes": 2}, "episodes": [{"seed": 700001}, {"seed": 700002}],
                    "training": {"scenario": "easy", "delay_mode": "uniform",
                                 "training_safety": False,
                                 "experience_target_per_ego": 50_000, "seed": 42,
                                 "actor_gru_input_mode": "paper_raw",
                                 "experiences": [50_000, 50_000],
                                 "actor_updates": [49_901, 49_901],
                                 "critic_updates": [49_901, 49_901]}}
        from copy import deepcopy
        other = deepcopy(template)
        other["method"] = "ctde"
        check_pair(template, other)
        other["training"]["critic_updates"][0] -= 1
        with self.assertRaisesRegex(ValueError, "budget"):
            check_pair(template, other)
        other["training"]["critic_updates"][0] += 1
        other["episodes"][1]["seed"] = 700003
        with self.assertRaisesRegex(ValueError, "seeds"):
            check_pair(template, other)


@unittest.skipUnless(__name__ == "__main__", "run opt-in SUMO training smoke")
class TrainingSmokeTests(unittest.TestCase):
    def test_both_modes_train_and_save_without_a_second_sumo_clock(self):
        from pathlib import Path
        import time
        root = Path(__file__).resolve().parents[1] / "tmp" / "ai" / "dual-sac-smoke"
        root.mkdir(parents=True, exist_ok=True)
        for mode in ("independent", "ctde"):
            name = f"smoke-{mode}-seed42-{time.time_ns()}"
            args = arguments(["--mode", mode, "--run-name", name,
                              "--experience-target", "103", "--batch-size", "8",
                              "--replay-capacity", "256", "--checkpoint-every", "100",
                              "--save-dir", str(root)])
            result = run(args)
            self.assertEqual(result["experiences"], [103, 103])
            self.assertEqual(result["actor_updates"], [4, 4])
            self.assertEqual(result["critic_updates"], [4, 4])
            checkpoint = root / name / "final.pt"
            self.assertTrue(checkpoint.exists())
            model, metadata = TwoAgentSAC.from_checkpoint(checkpoint)
            self.assertEqual(metadata["mode"], mode)
            self.assertEqual(model.actors[0].sample(torch.zeros(1, 134),
                                                     deterministic=True)[0].shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
