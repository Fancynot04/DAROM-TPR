import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluate_sac_delay import parse_args
from src.trajectory_export import TrajectoryDatasetWriter, TrajectoryDigestProbe
from src import trajectory_preferences
from src.trajectory_preferences import select_seeds
from src.trajectory_segments import extract_segment, load_episode


class _Simulation:
    def __init__(self):
        self.time = 0.0

    def getTime(self):
        return self.time


class _Traci:
    def __init__(self):
        self.simulation = _Simulation()


class _Base:
    def __init__(self):
        self.traci = _Traci()


class _Env:
    def __init__(self):
        self.unwrapped = _Base()


def _bundle(step, post=False):
    lane = max(0, 2 - step // 70)
    ego = {
        "valid": True, "missing_reason": None, "road_id": "E77",
        "lane_id": f"E77_{lane}", "lane_index": lane,
        "lane_position_m": float(step), "world_position_m": [float(step), 0.0],
        "speed_mps": 20.0, "acceleration_mps2": float(step % 3),
        "lateral_position_m": 0.0, "lateral_speed_mps": 0.0,
        "heading_deg": 90.0, "odometer_m": float(step),
    }
    route = {
        "valid": True, "missing_reason": None,
        "distance_to_d3_start_m": float(max(0, 300 - step)),
        "distance_to_d3_start_valid": True,
        "distance_to_d3_end_m": float(max(0, 500 - step)),
        "distance_to_d3_end_valid": True,
        "route_requirement_n_changes": lane,
        "route_requirement_valid": True,
        "has_direct_exit_route": lane == 0,
        "has_direct_exit_route_definition": "minimum_route_lane_changes_equals_zero_v1",
        "target": {"edge_id": "D3", "lane_id": "D3_0"},
        "route_edges": ["E77", "D3"], "route_index": 0,
        "next_route_edge": "D3",
    }
    surroundings = {
        "valid": True, "missing_reason": None, "range_m": 150.0,
        "candidate_lanes": [{
            "lane_id": ego["lane_id"], "lane_index": lane, "relation": "current",
            "front_vehicle": None, "rear_vehicle": None,
            "front_gap_m": 40.0, "rear_gap_m": 30.0,
            "front_relative_speed_mps": None, "rear_relative_speed_mps": None,
            "front_ttc_s": None, "rear_ttc_s": None,
            "front_min_gap_available": True, "rear_min_gap_available": True,
        }],
        "vehicle_snapshot": [],
        "safety_definition": "geometric_bumper_gap_vs_trailing_vehicle_minGap_v1",
    }
    return {"ego": ego, "route": route, "surroundings": surroundings}


def _removed_bundle():
    ego = {key: None for key in (
        "road_id", "lane_id", "lane_index", "lane_position_m",
        "world_position_m", "speed_mps", "acceleration_mps2",
        "lateral_position_m", "lateral_speed_mps", "heading_deg", "odometer_m")}
    ego.update(valid=False, missing_reason="ego_not_in_sumo_vehicle_list")
    route = {
        "valid": False, "missing_reason": "ego_not_in_sumo_vehicle_list",
        "distance_to_d3_start_m": None, "distance_to_d3_end_m": None,
        "distance_to_d3_start_valid": False, "distance_to_d3_end_valid": False,
        "route_requirement_n_changes": None, "route_requirement_valid": False,
        "has_direct_exit_route": None,
        "has_direct_exit_route_definition": "minimum_route_lane_changes_equals_zero_v1",
        "target": {"edge_id": "D3", "lane_id": "D3_0"},
        "route_edges": None, "route_index": None, "next_route_edge": None,
    }
    surroundings = {
        "valid": False, "missing_reason": "ego_not_in_sumo_vehicle_list",
        "range_m": 150.0, "candidate_lanes": [], "vehicle_snapshot": [],
        "safety_definition": "geometric_bumper_gap_vs_trailing_vehicle_minGap_v1",
    }
    return {"ego": ego, "route": route, "surroundings": surroundings}


class TrajectoryExportTests(unittest.TestCase):
    def test_pair_seed_selection_prioritizes_balanced_conversions(self):
        first = [
            {"seed": 1, "success": False}, {"seed": 2, "success": False},
            {"seed": 3, "success": True}, {"seed": 4, "success": True},
            {"seed": 5, "success": True}, {"seed": 6, "success": False},
        ]
        second = [
            {"seed": 1, "success": True}, {"seed": 2, "success": True},
            {"seed": 3, "success": False}, {"seed": 4, "success": False},
            {"seed": 5, "success": True}, {"seed": 6, "success": False},
        ]
        self.assertEqual(select_seeds(
            first, second, count=5, conversion_each=2, both_success=1),
            [1, 2, 3, 4, 5])

    def test_pair_builder_skips_short_episode_without_padding_or_abort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            assets.mkdir()
            for name in ("index.html", "app.js", "styles.css"):
                (assets / name).write_text(name, encoding="utf-8")
            episodes_a = {
                1: {"episode_id": "a-short", "steps": "80"},
                2: {"episode_id": "a-valid", "steps": "120"},
            }
            episodes_b = {
                1: {"episode_id": "b-valid", "steps": "110"},
                2: {"episode_id": "b-valid-2", "steps": "130"},
            }
            segment = {"summary": {}, "transitions": [],
                       "observations": [], "derived": []}
            with patch("src.trajectory_preferences._episode_by_seed",
                       side_effect=[episodes_a, episodes_b]), \
                    patch("src.trajectory_preferences._best_window",
                          return_value=(3.0, 10, segment, segment)) as best, \
                    patch("src.trajectory_preferences._compact",
                          return_value={"summary": {}, "steps": []}), \
                    patch("src.trajectory_preferences._comparability",
                          return_value={"status": "comparable"}):
                count = trajectory_preferences.build_pairs(
                    root / "a", root / "b", [1, 2], root / "output",
                    assets, "blind")
            self.assertEqual(count, 1)
            best.assert_called_once()
            provenance = json.loads(
                (root / "output/pair_provenance.json").read_text())
            self.assertEqual(provenance["counts"], {
                "requested_pairs": 2, "candidate_pairs": 1,
                "skipped_pairs": 1, "short_trajectories": 1,
            })
            self.assertEqual(provenance["skipped"][0]["reason"],
                             "short_episode")
            self.assertEqual(
                provenance["skipped"][0]["short_trajectories"][0]["steps"], 80)
            self.assertEqual(len(list(
                (root / "output/segments").glob("*.json.gz"))), 2)

    def test_negative_candidate_gap_is_overlap_and_has_no_valid_ttc(self):
        segment = {"transitions": [{
            "sim_time_pre_s": 12.0,
            "pre": {
                "ego": {"road_id": "E76", "lane_index": 1,
                        "speed_mps": 20.0},
                "surroundings": {"candidate_lanes": [{
                    "lane_id": "E76_0", "lane_index": 0, "relation": "right",
                    "front_vehicle": {"vehicle_id": "front", "lane_id": "E76_0",
                                      "speed_mps": 18.0},
                    "front_gap_m": -2.5, "front_ttc_s": None,
                    "rear_vehicle": {"vehicle_id": "rear", "lane_id": "E76_0",
                                     "speed_mps": 24.0},
                    "rear_gap_m": 8.0, "rear_ttc_s": 2.0,
                }]},
            },
        }]}
        audit = trajectory_preferences._gap_audit(segment)
        self.assertEqual(audit["minimum_signed_gap"]["signed_gap_m"], -2.5)
        self.assertEqual(audit["minimum_signed_gap"]["overlap_m"], 2.5)
        self.assertFalse(audit["minimum_signed_gap"]["ttc_valid"])
        self.assertEqual(audit["minimum_signed_gap"]["ttc_invalid_reason"],
                         "negative_gap_longitudinal_overlap")
        self.assertIsNone(audit["minimum_signed_gap"]["ttc_s"])
        self.assertEqual(audit["minimum_nonnegative_gap"]["signed_gap_m"], 8.0)
        self.assertEqual(audit["minimum_nonnegative_gap"]["ttc_s"], 2.0)

    def test_route_and_lane_change_events_keep_request_completion_distinct(self):
        request = {
            "sim_time_pre_s": 10.0,
            "sac": {"lateral_direction": -1},
            "controller": {"lateral": {"traci_command_called": True,
                                         "blocked_by_pending": False}},
            "execution": {"lane_change_completed": False,
                          "lane_change_completion_event": None},
            "pre": {
                "ego": {"road_id": "E76", "lane_id": "E76_2", "lane_index": 2},
                "route": {"route_requirement_n_changes": 2,
                          "distance_to_d3_start_m": 900.0},
            },
            "post": {
                "ego": {"road_id": "E76", "lane_id": "E76_2", "lane_index": 2},
                "route": {"route_requirement_n_changes": 2,
                          "distance_to_d3_start_m": 897.0},
            },
        }
        completion = {
            "sim_time_pre_s": 14.9, "sim_time_post_s": 15.0,
            "sac": {"lateral_direction": -1},
            "controller": {"lateral": {"traci_command_called": False,
                                         "blocked_by_pending": True}},
            "execution": {
                "lane_change_completed": True, "completed_direction": -1,
                "lane_change_completion_event": {
                    "actual_direction": -1, "route_direction": "toward",
                    "request_direction": -1, "request_time_s": 10.0,
                    "completion_time_s": 15.0,
                    "request_distance_to_d3_start_m": 900.0,
                },
            },
            "pre": {
                "ego": {"road_id": "E76", "lane_id": "E76_2", "lane_index": 2},
                "route": {"route_requirement_n_changes": 2,
                          "distance_to_d3_start_m": 800.0},
            },
            "post": {
                "ego": {"road_id": "E76", "lane_id": "E76_1", "lane_index": 1},
                "route": {"route_requirement_n_changes": 1,
                          "distance_to_d3_start_m": 797.0},
            },
        }
        segment = {"transitions": [request, completion]}
        lane_event = trajectory_preferences._lane_change_events(segment)[0]
        self.assertTrue(lane_event["request_in_segment"])
        self.assertTrue(lane_event["request_command_sent"])
        self.assertFalse(lane_event["request_blocked_by_pending"])
        self.assertEqual(lane_event["blocked_actor_requests_during_pending"], 1)
        route_event = trajectory_preferences._route_change_events(segment)[0]
        self.assertEqual(route_event["cause"], "actual_lane_change")
        self.assertTrue(route_event["actual_lane_change_completed"])
        self.assertFalse(route_event["road_connection_transition"])

    def test_cli_export_is_opt_in(self):
        default = parse_args(["model", "--episodes", "1"])
        self.assertIsNone(default.trajectory_export_dir)
        configured = parse_args([
            "model", "--episodes", "1", "--trajectory-export-dir", "new-data",
            "--trajectory-neighbor-range-m", "75", "--trajectory-source-commit",
            "0123456789abcdef0123456789abcdef01234567"])
        self.assertEqual(configured.trajectory_export_dir, Path("new-data"))
        self.assertEqual(configured.trajectory_neighbor_range_m, 75.0)

    def test_complete_episode_and_extract_100_and_200_steps(self):
        env = _Env()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            writer = TrajectoryDatasetWriter(output, {
                "checkpoint": "model.zip", "checkpoint_sha256": "abc",
                "scenario": "easy", "controller_mode": "no_pending"})
            with patch("src.trajectory_export._state_bundle",
                       side_effect=lambda base, radius: (
                           _removed_bundle() if int(round(
                               base.traci.simulation.time * 10)) == 200 else
                           _bundle(int(round(base.traci.simulation.time * 10))))), \
                    patch("src.trajectory_export._observation_record",
                          side_effect=lambda env, observation, step_id, episode_id,
                          eval_seed, step: {
                              "step_id": step_id, "episode_id": episode_id,
                              "eval_seed": eval_seed, "step": step,
                              "policy_observation": list(observation),
                              "policy_observation_shape": [len(observation)],
                              "policy_observation_sha256": "hash",
                              "delay_steps": step % 21,
                              "delay_seconds": (step % 21) * 0.1,
                              "received_entity_observation": [[1.0, 2.0, 3.0]],
                              "raw_action_history": [[0.0, 0.0]],
                              "masked_action_history": [[0.0, 0.0]],
                              "nav_dim": 0,
                          }), \
                    patch("src.trajectory_export._controller_record",
                          return_value={"controller_mode": "no_pending",
                                        "longitudinal": {"called": True},
                                        "lateral": {"traci_command_called": True}}):
                with writer:
                    writer.begin_episode(200051, {"ego_start_lane": 2}, env)
                    for step in range(200):
                        env.unwrapped.traci.simulation.time = step * 0.1
                        captured = writer.capture_pre(
                            env, np.array([step], dtype=np.float32),
                            np.array([0.5, -5.0], dtype=np.float32), step)
                        env.unwrapped.traci.simulation.time = (step + 1) * 0.1
                        event = ({"actual_direction": -1, "route_direction": "toward"}
                                 if step in (69, 139) else None)
                        writer.record_transition(
                            captured, env, np.array([step + 1], dtype=np.float32),
                            1.0, step == 199, False, {
                                "directional_lane_change_event": event,
                                "exit_applied_lane_change_direction": -1,
                                "reward_components": {"efficiency": 1.0},
                                "message": "Arrived" if step == 199 else "",
                                "missed_exit": False,
                            })
                    writer.finish_episode({
                        "outcome": "Arrived", "success": True, "return": 200.0,
                        "ego_teleported": False, "toward_requests": 200,
                        "away_requests": 0, "lane_change_commands_sent": 200,
                        "lane_change_counts": {"toward": 2},
                        "lane_change_direction_reversals": 0,
                        "same_direction_repeat_requests": 199,
                        "short_horizon_request_reversals_1s": 0,
                        "mean_speed_mps": 20.0,
                    })
                writer.finalize(report={"success_rate": 100.0})
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["episode_count"], 1)
            self.assertEqual(manifest["total_steps"], 200)
            with (output / "episodes.csv").open(newline="") as stream:
                episodes = list(csv.DictReader(stream))
            self.assertEqual(episodes[0]["steps"], "200")
            rows = load_episode(output, "ep-0000-seed-200051")
            self.assertTrue(all(len(value) == 200 for value in rows.values()))
            self.assertTrue(rows["transitions"][-1]["pre"]["ego"]["valid"])
            self.assertFalse(rows["transitions"][-1]["post"]["ego"]["valid"])
            self.assertEqual(rows["transitions"][-1]["events"]["terminal_reason"],
                             "Arrived")
            events = (output / "lane_change_events.jsonl").read_text().splitlines()
            self.assertEqual(len(events), 2)
            completeness = json.loads(
                (output / "field_completeness.json").read_text())
            self.assertIn("observation.delay_steps", completeness["fields"])
            self.assertIn("derived.jerk_mps3", completeness["fields"])
            ten = extract_segment(output, "ep-0000-seed-200051", 0, 100)
            twenty = extract_segment(output, "ep-0000-seed-200051", 0, 200)
            self.assertEqual(ten["summary"]["duration_s"], 10.0)
            self.assertEqual(twenty["summary"]["duration_s"], 20.0)
            self.assertEqual(twenty["summary"]["completed_lane_changes"], 2)
            self.assertEqual(twenty["summary"]["repeated_actor_requests"], 199)

    def test_terminal_transition_never_queries_closed_simulation_clock(self):
        env = _Env()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            writer = TrajectoryDatasetWriter(output, {
                "checkpoint": "model.zip", "checkpoint_sha256": "abc",
                "scenario": "easy", "controller_mode": "pending"})
            with patch("src.trajectory_export._state_bundle",
                       return_value=_removed_bundle()), \
                    patch("src.trajectory_export._observation_record",
                          return_value={
                              "step_id": "step", "episode_id": "episode",
                              "eval_seed": 200051, "step": 0,
                              "policy_observation": [],
                              "policy_observation_shape": [0],
                              "policy_observation_sha256": "hash",
                              "delay_steps": None, "delay_seconds": None,
                              "received_entity_observation": None,
                              "raw_action_history": None,
                              "masked_action_history": None, "nav_dim": None,
                          }), \
                    patch("src.trajectory_export._controller_record",
                          return_value={"controller_mode": "pending"}):
                with writer:
                    writer.begin_episode(200051, {}, env)
                    env.unwrapped.traci.simulation.time = 12.3
                    captured = writer.capture_pre(
                        env, np.array([], dtype=np.float32),
                        np.array([0.0, 0.0], dtype=np.float32), 0)
                    env.unwrapped.traci.simulation.getTime = unittest.mock.Mock(
                        side_effect=RuntimeError("Connection already closed"))
                    transition = writer.record_transition(
                        captured, env, np.array([], dtype=np.float32), 0.0,
                        True, False, {"message": "Arrived"})
                    self.assertAlmostEqual(transition["sim_time_post_s"], 12.4)
                    self.assertEqual(transition["sim_time_post_source"],
                                     "pre_plus_fixed_step")

    def test_existing_dataset_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                with TrajectoryDatasetWriter(
                        output, {"controller_mode": "no_pending"}):
                    pass

    def test_validation_probe_keeps_stepwise_ego_digest_in_memory(self):
        env = _Env()
        probe = TrajectoryDigestProbe()
        probe.begin_episode(200051)
        with patch("src.trajectory_export._ego_state",
                   side_effect=[_bundle(0)["ego"], _bundle(1)["ego"]]):
            captured = probe.capture_pre(
                env, np.array([1.0, 2.0], dtype=np.float32),
                np.array([0.5, -5.0], dtype=np.float32), 0)
            probe.record_transition(captured, env, 1.0, True, False, {
                "reward_components": {"efficiency": 1.0},
                "exit_actor_request_direction": -1,
                "exit_applied_lane_change_direction": -1,
                "directional_lane_change_event": None,
                "message": "Arrived",
            })
        probe.finish_episode()
        summary = probe.summary()
        self.assertEqual(summary[0]["eval_seed"], 200051)
        self.assertEqual(summary[0]["steps"], 1)
        self.assertEqual(len(summary[0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
