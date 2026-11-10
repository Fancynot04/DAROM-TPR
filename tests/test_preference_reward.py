import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.preference_reward import OBS_DIM, fit_preference_reward, load_and_audit


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
        summaries, transitions = [], []
        vector_fields = ([f"obs_{i:03d}" for i in range(OBS_DIM)]
                         + ["action_exec_0", "action_exec_1"]
                         + [f"next_obs_{i:03d}" for i in range(OBS_DIM)])
        fields = (["trajectory_id", "step"] + vector_fields
                  + ["terminated", "truncated", "is_terminal_step", "terminal_reason",
                     "next_obs_valid"])
        for index, tid in enumerate(tids):
            summaries.append({"trajectory_id": tid, "eval_seed": seed, "steps": 2,
                              "terminal_reason": "Arrived" if index == 0 else "Collision"})
            for step in range(2):
                obs = [float(index + step)] * OBS_DIM
                nxt = [float(index + step + 1)] * OBS_DIM
                row = {"trajectory_id": tid, "step": step,
                       "action_exec_0": 0.1, "action_exec_1": -0.2,
                       "terminated": int(step == 1), "truncated": 0,
                       "is_terminal_step": int(step == 1),
                       "terminal_reason": "Arrived" if step == 1 and index == 0 else
                                          ("Collision" if step == 1 else ""),
                       "next_obs_valid": int(step == 0)}
                row.update({f"obs_{i:03d}": obs[i] for i in range(OBS_DIM)})
                row.update({f"next_obs_{i:03d}": nxt[i] for i in range(OBS_DIM)})
                transitions.append(row)
        write_csv(group / "episode_summary.csv",
                  ["trajectory_id", "eval_seed", "steps", "terminal_reason"], summaries)
        write_csv(group / "trajectory_transitions.csv", fields, transitions)
        files = {name: hashlib.sha256((group / name).read_bytes()).hexdigest()
                 for name in ("episode_summary.csv", "trajectory_transitions.csv")}
        (group / "trajectory_manifest.json").write_text(json.dumps({
            "status": "ok", "episodes": [{"complete": True}, {"complete": True}],
            "files": files,
        }))
        preferences = root / f"{source_id}_preferences.csv"
        write_csv(preferences,
                  ["pair_id", "trajectory_a", "trajectory_b", "preference"],
                  [{"pair_id": source_id, "trajectory_a": tids[0],
                    "trajectory_b": tids[1], "preference": "A"}])
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
            with self.assertRaisesRegex(ValueError, "train, validation, and test"):
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
            _, _, history, test_loss = fit_preference_reward(
                trajectories, pairs, epochs=2, hidden_dims=(8,), seed=7)
            self.assertEqual(len(history), 2)
            self.assertGreaterEqual(test_loss, 0.0)

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


if __name__ == "__main__":
    unittest.main()
