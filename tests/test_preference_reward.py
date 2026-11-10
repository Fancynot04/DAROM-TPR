import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.preference_reward import (FEATURE_DIM, NEXT_OBS_START, OBS_DIM,
                                   TransitionRewardMLP, fit_preference_reward,
                                   load_and_audit, normalize_trajectory,
                                   trajectory_score)


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class PreferenceRewardTest(unittest.TestCase):
    def make_source(self, root, source_id, role, seed):
        group = root / source_id
        group.mkdir()
        tids = [f"{source_id}_a", f"{source_id}_b"]
        summaries, steps, transitions = [], [], []
        vector_fields = ([f"obs_{i:03d}" for i in range(OBS_DIM)]
                         + ["action_exec_0", "action_exec_1"]
                         + [f"next_obs_{i:03d}" for i in range(OBS_DIM)])
        fields = (["trajectory_id", "step"] + vector_fields
                  + ["terminated", "truncated", "is_terminal_step", "terminal_reason",
                     "next_obs_valid"])
        for index, tid in enumerate(tids):
            summaries.append({"trajectory_id": tid, "eval_seed": seed, "steps": 2,
                              "terminal_reason": "Arrived"})
            for step in range(2):
                obs = [float(index + step)] * OBS_DIM
                nxt = [float(index + step + 1)] * OBS_DIM
                row = {"trajectory_id": tid, "step": step,
                       "action_exec_0": 0.1, "action_exec_1": -0.2,
                       "terminated": int(step == 1), "truncated": 0,
                       "is_terminal_step": int(step == 1),
                       "terminal_reason": "Arrived" if step == 1 else "",
                       "next_obs_valid": int(step == 0)}
                row.update({f"obs_{i:03d}": obs[i] for i in range(OBS_DIM)})
                row.update({f"next_obs_{i:03d}": nxt[i] for i in range(OBS_DIM)})
                transitions.append(row)
                steps.append({
                    "trajectory_id": tid, "step": step, "ego_lane_index": index,
                    "ego_lane_index_after": index, "ego_speed_mps": 20 + index,
                    "lane_change_physical": 0,
                })
        write_csv(group / "episode_summary.csv",
                  ["trajectory_id", "eval_seed", "steps", "terminal_reason"], summaries)
        write_csv(group / "trajectory_transitions.csv", fields, transitions)
        write_csv(group / "trajectory_steps.csv",
                  ["trajectory_id", "step", "ego_lane_index", "ego_lane_index_after",
                   "ego_speed_mps", "lane_change_physical"], steps)
        files = {name: hashlib.sha256((group / name).read_bytes()).hexdigest()
                 for name in ("episode_summary.csv", "trajectory_steps.csv",
                              "trajectory_transitions.csv")}
        (group / "trajectory_manifest.json").write_text(json.dumps({
            "status": "ok", "episodes": [{"complete": True}, {"complete": True}],
            "files": files,
        }))
        preferences = root / f"{source_id}_preferences.csv"
        write_csv(preferences,
                  ["pair_id", "trajectory_a", "trajectory_b", "preference",
                   "deciding_priority"],
                  [{"pair_id": source_id, "trajectory_a": tids[0],
                    "trajectory_b": tids[1], "preference": "A",
                    "deciding_priority": "P3"}])
        return {"source_id": source_id, "role": role, "groups": {"B0": str(group)},
                "preferences": str(preferences), "provenance": {
                    "experiment_id": source_id, "git_commit": "a" * 40,
                    "evaluation_seeds": [seed]}}

    def test_holdout_is_audited_but_fit_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {"schema_version": "preference-reward-data-v1",
                    "sources": [self.make_source(root, "p1", "holdout", 300001)]}
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            trajectories, pairs, audit = load_and_audit(path)
            self.assertTrue(audit["ok"])
            self.assertFalse(audit["fit_ready"])
            self.assertEqual(audit["usable_pairs_by_role"], {"holdout": 1})
            with self.assertRaisesRegex(ValueError, "success-success train and validation"):
                fit_preference_reward(trajectories, pairs, epochs=1)

    def test_seed_leakage_across_roles_fails_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {"schema_version": "preference-reward-data-v1", "sources": [
                self.make_source(root, "train", "train", 10),
                self.make_source(root, "validation", "validation", 10),
            ]}
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            _, _, audit = load_and_audit(path)
            self.assertFalse(audit["ok"])
            self.assertIn("eval_seed leakage", audit["errors"][-1])

    def test_independent_roles_can_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {"schema_version": "preference-reward-data-v1", "sources": [
                self.make_source(root, "train", "train", 10),
                self.make_source(root, "validation", "validation", 20),
                self.make_source(root, "test", "test", 30),
            ]}
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            trajectories, pairs, audit = load_and_audit(path)
            self.assertTrue(audit["fit_ready"])
            _, normalizer, history, result = fit_preference_reward(
                trajectories, pairs, epochs=2, hidden_dims=(8,), seed=7)
            self.assertEqual(len(history), 2)
            self.assertGreaterEqual(result["validation_loss"], 0.0)
            self.assertEqual(result["selected_pairs"], {"train": 1, "validation": 1})
            self.assertEqual(normalizer["mean"].shape, (FEATURE_DIM,))
            self.assertIsInstance(result["train_step_reward_mean"], float)

    def test_known_evaluation_seed_cannot_be_assigned_to_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {"schema_version": "preference-reward-data-v1", "sources": [
                self.make_source(root, "p1", "train", 300001),
            ]}
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            _, _, audit = load_and_audit(path)
            self.assertFalse(audit["ok"])
            self.assertTrue(any("reserved eval_seed" in error for error in audit["errors"]))

    def test_known_evaluation_seed_can_be_final_test_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {"schema_version": "preference-reward-data-v1", "sources": [
                self.make_source(root, "p1", "test", 300001),
            ]}
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            _, _, audit = load_and_audit(path)
            self.assertTrue(audit["ok"])
            self.assertEqual(audit["usable_pairs_by_role"], {"test": 1})

    def test_role_filter_does_not_open_p1_test_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {"schema_version": "preference-reward-data-v1", "sources": [
                self.make_source(root, "train", "train", 10),
                self.make_source(root, "validation", "validation", 20),
                {"source_id": "p1-test", "role": "test",
                 "groups": {"B0": "/must/not/be/opened"},
                 "preferences": "/must/not/be/opened/preferences.csv",
                 "provenance": {}},
            ]}
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            trajectories, pairs, audit = load_and_audit(
                path, roles={"train", "validation"},
                required_roles=("train", "validation"))
            self.assertTrue(audit["fit_ready"])
            self.assertEqual({item.role for item in trajectories.values()},
                             {"train", "validation"})
            self.assertEqual({item.role for item in pairs}, {"train", "validation"})
            self.assertEqual(audit["skipped_sources"],
                             [{"source_id": "p1-test", "role": "test"}])

    def test_invalid_successor_is_zero_after_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source(root, "train", "train", 10)
            path = root / "spec.json"
            path.write_text(json.dumps({
                "schema_version": "preference-reward-data-v1", "sources": [source]}))
            trajectories, _, audit = load_and_audit(
                path, roles={"train"}, required_roles=("train",))
            self.assertTrue(audit["ok"])
            trajectory = next(iter(trajectories.values()))
            mean = np.zeros(FEATURE_DIM, dtype=np.float32)
            scale = np.ones(FEATURE_DIM, dtype=np.float32)
            normalized = normalize_trajectory(trajectory, mean, scale)
            self.assertTrue(np.all(normalized[-1, NEXT_OBS_START:] == 0))

    def test_mean_aggregation_has_no_constant_reward_length_shortcut(self):
        model = TransitionRewardMLP(FEATURE_DIM, hidden_dims=(4,))
        for parameter in model.parameters():
            torch.nn.init.constant_(parameter, 0.0)
        model.network[-1].bias.data.fill_(2.0)
        short = torch.zeros((2, FEATURE_DIM))
        long = torch.zeros((7, FEATURE_DIM))
        self.assertAlmostEqual(float(trajectory_score(model, short, "mean").detach()),
                               float(trajectory_score(model, long, "mean").detach()))
        self.assertNotEqual(float(trajectory_score(model, short, "sum").detach()),
                            float(trajectory_score(model, long, "sum").detach()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_training_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {"schema_version": "preference-reward-data-v1", "sources": [
                self.make_source(root, "train", "train", 10),
                self.make_source(root, "validation", "validation", 20),
            ]}
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            trajectories, pairs, audit = load_and_audit(
                path, roles={"train", "validation"},
                required_roles=("train", "validation"))
            self.assertTrue(audit["fit_ready"])
            model, _, history, _ = fit_preference_reward(
                trajectories, pairs, epochs=1, hidden_dims=(8,), device="cuda")
            self.assertEqual(next(model.parameters()).device.type, "cpu")
            self.assertEqual(len(history), 1)


if __name__ == "__main__":
    unittest.main()
