"""Opt-in, read-only trajectory export for frozen-policy evaluation.

Each transition is written as ``(s_t, a_t, command_t, r_t, s_t+1)``.  The
writer never advances SUMO and never changes an action, reward, or terminal
condition.  High-dimensional policy inputs and derived quantities are stored
separately and joined by ``step_id``.
"""

from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import exit_e76_d3_config as exit_config
from .merging import (EGO_SPEED_CAP_MPS, LANE_CHANGE_EDGES, SUMO_STEP_LENGTH,
                      parseAction)


SCHEMA_VERSION = "darom-pbrl-p0-trajectory-v1"
DERIVATION_VERSION = "darom-pbrl-p0-derived-v1"


def _json_value(value):
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _sha256_json(value):
    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _delay_wrapper(env):
    current = env
    while current is not None:
        if all(hasattr(current, name) for name in (
                "delay_of_received_observation", "action_history", "last_observation")):
            return current
        following = getattr(current, "env", None)
        if following is current:
            break
        current = following
    return None


def _valid_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _route_snapshot(base, state):
    result = {
        "valid": False,
        "missing_reason": state.get("missing_reason"),
        "distance_to_d3_start_m": None,
        "distance_to_d3_start_valid": False,
        "distance_to_d3_end_m": None,
        "distance_to_d3_end_valid": False,
        "route_requirement_n_changes": None,
        "route_requirement_valid": False,
        "has_direct_exit_route": None,
        "target": {"edge_id": exit_config.DESTINATION, "lane_id": "D3_0"},
        "route_edges": None,
        "route_index": None,
        "next_route_edge": None,
    }
    if not state["valid"]:
        return result
    vehicle, ego = base.traci.vehicle, base.ego_id
    try:
        road = vehicle.getRoadID(ego)
        lane_id = vehicle.getLaneID(ego)
        route = list(vehicle.getRoute(ego))
        route_index = int(vehicle.getRouteIndex(ego))
        if road == exit_config.DESTINATION or lane_id == ":J106_0_0":
            distance_start, required = 0.0, 0
        else:
            distance_start = _valid_number(
                vehicle.getDrivingDistance(ego, exit_config.DESTINATION, 0.0))
            if distance_start is not None and distance_start < 0:
                distance_start = None
            if road in ("E76", "E77"):
                required = int(vehicle.getLaneIndex(ego))
            elif lane_id in exit_config.J105_TARGET_LANES:
                required = int(exit_config.J105_TARGET_LANES[lane_id])
            else:
                navigation = base.exit_navigation()
                required = int(round(float(navigation[1]) * 5.0))
        distance_end = _valid_number(base._remaining_to_d3_end())
        result.update(
            valid=True,
            missing_reason=None,
            distance_to_d3_start_m=distance_start,
            distance_to_d3_start_valid=(distance_start is not None),
            distance_to_d3_end_m=distance_end,
            distance_to_d3_end_valid=(distance_end is not None),
            route_requirement_n_changes=required,
            route_requirement_valid=True,
            has_direct_exit_route=(required == 0),
            has_direct_exit_route_definition="minimum_route_lane_changes_equals_zero_v1",
            route_edges=route,
            route_index=route_index,
            next_route_edge=(route[route_index + 1]
                             if 0 <= route_index < len(route) - 1 else None),
        )
    except Exception as error:
        result["missing_reason"] = f"route_query_error:{type(error).__name__}:{error}"
    return result


def _ego_state(base):
    result = {
        "valid": False,
        "missing_reason": None,
        "road_id": None,
        "lane_id": None,
        "lane_index": None,
        "lane_position_m": None,
        "world_position_m": None,
        "speed_mps": None,
        "acceleration_mps2": None,
        "lateral_position_m": None,
        "lateral_speed_mps": None,
        "heading_deg": None,
        "odometer_m": None,
    }
    try:
        ego = getattr(base, "ego_id", None)
        if ego is None or ego not in base.traci.vehicle.getIDList():
            result["missing_reason"] = "ego_not_in_sumo_vehicle_list"
            return result
        vehicle = base.traci.vehicle
        result.update(
            valid=True,
            road_id=vehicle.getRoadID(ego),
            lane_id=vehicle.getLaneID(ego),
            lane_index=int(vehicle.getLaneIndex(ego)),
            lane_position_m=_valid_number(vehicle.getLanePosition(ego)),
            world_position_m=[_valid_number(value) for value in vehicle.getPosition(ego)],
            speed_mps=_valid_number(vehicle.getSpeed(ego)),
            acceleration_mps2=_valid_number(vehicle.getAcceleration(ego)),
            lateral_position_m=_valid_number(vehicle.getLateralLanePosition(ego)),
            lateral_speed_mps=_valid_number(vehicle.getLateralSpeed(ego)),
            heading_deg=_valid_number(vehicle.getAngle(ego)),
            odometer_m=_valid_number(vehicle.getDistance(ego)),
        )
    except Exception as error:
        result.update(valid=False,
                      missing_reason=f"ego_query_error:{type(error).__name__}:{error}")
    return result


def _vehicle_record(vehicle, vehicle_id):
    return {
        "vehicle_id": vehicle_id,
        "road_id": vehicle.getRoadID(vehicle_id),
        "lane_id": vehicle.getLaneID(vehicle_id),
        "lane_index": int(vehicle.getLaneIndex(vehicle_id)),
        "lane_position_m": _valid_number(vehicle.getLanePosition(vehicle_id)),
        "world_position_m": [_valid_number(value) for value in vehicle.getPosition(vehicle_id)],
        "speed_mps": _valid_number(vehicle.getSpeed(vehicle_id)),
        "acceleration_mps2": _valid_number(vehicle.getAcceleration(vehicle_id)),
        "lateral_position_m": _valid_number(vehicle.getLateralLanePosition(vehicle_id)),
        "lateral_speed_mps": _valid_number(vehicle.getLateralSpeed(vehicle_id)),
        "length_m": _valid_number(vehicle.getLength(vehicle_id)),
        "min_gap_m": _valid_number(vehicle.getMinGap(vehicle_id)),
    }


def _ttc(gap, closing_speed):
    if gap is None or closing_speed is None or gap < 0 or closing_speed <= 0:
        return None
    return gap / closing_speed


def _neighbors(base, ego_state, radius_m):
    result = {
        "valid": False,
        "missing_reason": ego_state.get("missing_reason"),
        "range_m": float(radius_m),
        "candidate_lanes": [],
        "vehicle_snapshot": [],
        "safety_definition": "geometric_bumper_gap_vs_trailing_vehicle_minGap_v1",
    }
    if not ego_state["valid"]:
        return result
    try:
        traci, vehicle, ego = base.traci, base.traci.vehicle, base.ego_id
        road = ego_state["road_id"]
        ego_lane = ego_state["lane_index"]
        ego_position = ego_state["lane_position_m"]
        ego_speed = ego_state["speed_mps"]
        ego_length = _valid_number(vehicle.getLength(ego))
        ego_min_gap = _valid_number(vehicle.getMinGap(ego))
        if road.startswith(":"):
            lane_indices = (ego_lane,)
        else:
            lane_count = int(traci.edge.getLaneNumber(road))
            lane_indices = tuple(index for index in (ego_lane - 1, ego_lane, ego_lane + 1)
                                 if 0 <= index < lane_count)
        for lane_index in lane_indices:
            lane_id = (ego_state["lane_id"] if road.startswith(":")
                       else f"{road}_{lane_index}")
            candidates = []
            for other_id in traci.lane.getLastStepVehicleIDs(lane_id):
                if other_id == ego:
                    continue
                other = _vehicle_record(vehicle, other_id)
                if other["lane_position_m"] is not None:
                    candidates.append(other)
            front = [item for item in candidates
                     if item["lane_position_m"] >= ego_position]
            rear = [item for item in candidates
                    if item["lane_position_m"] < ego_position]
            leader = min(front, key=lambda item: item["lane_position_m"], default=None)
            follower = max(rear, key=lambda item: item["lane_position_m"], default=None)
            front_gap = (leader["lane_position_m"] - leader["length_m"] - ego_position
                         if leader is not None else None)
            rear_gap = (ego_position - ego_length - follower["lane_position_m"]
                        if follower is not None else None)
            front_closing = (ego_speed - leader["speed_mps"]
                             if leader is not None else None)
            rear_closing = (follower["speed_mps"] - ego_speed
                            if follower is not None else None)
            result["candidate_lanes"].append({
                "lane_id": lane_id,
                "lane_index": lane_index,
                "relation": ("current" if lane_index == ego_lane
                             else "right" if lane_index < ego_lane else "left"),
                "front_vehicle": leader,
                "rear_vehicle": follower,
                "front_gap_m": _valid_number(front_gap),
                "rear_gap_m": _valid_number(rear_gap),
                "front_relative_speed_mps": (leader["speed_mps"] - ego_speed
                                              if leader is not None else None),
                "rear_relative_speed_mps": (follower["speed_mps"] - ego_speed
                                             if follower is not None else None),
                "front_ttc_s": _valid_number(_ttc(front_gap, front_closing)),
                "rear_ttc_s": _valid_number(_ttc(rear_gap, rear_closing)),
                "front_min_gap_available": (front_gap >= ego_min_gap
                                             if front_gap is not None else None),
                "rear_min_gap_available": (rear_gap >= follower["min_gap_m"]
                                            if follower is not None and rear_gap is not None else None),
            })
        ego_xy = np.asarray(ego_state["world_position_m"], dtype=float)
        for other_id in vehicle.getIDList():
            if other_id == ego:
                continue
            other = _vehicle_record(vehicle, other_id)
            if other["world_position_m"] is None:
                continue
            distance = float(np.linalg.norm(np.asarray(other["world_position_m"]) - ego_xy))
            if distance <= radius_m:
                other["euclidean_distance_to_ego_m"] = distance
                result["vehicle_snapshot"].append(other)
        result["vehicle_snapshot"].sort(
            key=lambda item: (item["euclidean_distance_to_ego_m"], item["vehicle_id"]))
        result.update(valid=True, missing_reason=None)
    except Exception as error:
        result.update(valid=False,
                      missing_reason=f"neighbor_query_error:{type(error).__name__}:{error}")
    return result


def _state_bundle(base, neighbor_range_m):
    state = _ego_state(base)
    return {"ego": state, "route": _route_snapshot(base, state),
            "surroundings": _neighbors(base, state, neighbor_range_m)}


class TrajectoryDigestProbe:
    """Validation-only in-memory Ego trace; it writes no dataset files."""

    def __init__(self):
        self.episodes = []
        self.current = None

    def begin_episode(self, eval_seed):
        if self.current is not None:
            raise RuntimeError("Previous validation-probe episode is unfinished")
        self.current = {"eval_seed": int(eval_seed), "steps": []}

    def capture_pre(self, env, observation, actor_action, step):
        if self.current is None:
            raise RuntimeError("Validation probe has no active episode")
        policy = np.asarray(observation, dtype=np.float32)
        return {
            "step": int(step),
            "ego": _ego_state(env.unwrapped),
            "policy_observation_sha256": hashlib.sha256(policy.tobytes()).hexdigest(),
            "actor_action": np.asarray(actor_action, dtype=np.float32).tolist(),
        }

    def record_transition(self, captured, env, reward, terminated, truncated, info):
        row = {
            "step": captured["step"],
            "pre_ego": captured["ego"],
            "post_ego": _ego_state(env.unwrapped),
            "policy_observation_sha256": captured["policy_observation_sha256"],
            "actor_action": captured["actor_action"],
            "reward": float(reward),
            "reward_components": _json_value(info.get("reward_components", {})),
            "actor_request_direction": int(info.get("exit_actor_request_direction", 0)),
            "command_direction": int(info.get("exit_applied_lane_change_direction", 0)),
            "lane_change_completion_event": _json_value(
                info.get("directional_lane_change_event")),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "terminal_reason": info.get("message") if terminated or truncated else None,
        }
        self.current["steps"].append(row)

    def finish_episode(self):
        if self.current is None:
            raise RuntimeError("Validation probe has no active episode")
        self.current["sha256"] = _sha256_json(self.current["steps"])
        self.current["step_count"] = len(self.current["steps"])
        self.episodes.append(self.current)
        self.current = None

    def summary(self):
        if self.current is not None:
            raise RuntimeError("Validation probe has an unfinished episode")
        return [{"eval_seed": row["eval_seed"], "steps": row["step_count"],
                 "sha256": row["sha256"]} for row in self.episodes]


def _observation_record(env, observation, step_id, episode_id, eval_seed, step):
    delay = _delay_wrapper(env)
    policy = np.asarray(observation, dtype=np.float32)
    record = {
        "step_id": step_id,
        "episode_id": episode_id,
        "eval_seed": eval_seed,
        "step": step,
        "policy_observation": policy.tolist(),
        "policy_observation_shape": list(policy.shape),
        "policy_observation_sha256": hashlib.sha256(policy.tobytes()).hexdigest(),
        "delay_steps": None,
        "delay_seconds": None,
        "received_entity_observation": None,
        "raw_action_history": None,
        "masked_action_history": None,
        "nav_dim": None,
    }
    if delay is None:
        record["missing_reason"] = "delay_wrapper_not_found"
        return record
    delay_steps = int(delay.delay_of_received_observation)
    action_history = np.asarray(list(delay.action_history), dtype=np.float32)
    masked = action_history.copy()
    if delay_steps < len(masked):
        masked[delay_steps:] = 0.0
    record.update(
        delay_steps=delay_steps,
        delay_seconds=delay_steps * SUMO_STEP_LENGTH,
        received_entity_observation=np.asarray(delay.last_observation,
                                               dtype=np.float32).tolist(),
        raw_action_history=action_history.tolist(),
        masked_action_history=masked.tolist(),
        nav_dim=int(getattr(delay, "nav_dim", 0)),
    )
    return record


def _controller_record(pre, actor_action, info):
    parsed = parseAction(actor_action)
    ego = pre["ego"]
    direction = int(parsed[1])
    eligible_edge = bool(ego["valid"] and ego["road_id"] in LANE_CHANGE_EDGES)
    target = ego["lane_index"] + direction if ego["valid"] and direction else None
    lane_count = None
    if ego["valid"] and not ego["road_id"].startswith(":"):
        try:
            lane_count = int(info["_base"].traci.edge.getLaneNumber(ego["road_id"]))
        except Exception:
            lane_count = None
    target_valid = (bool(eligible_edge and lane_count is not None
                         and target is not None and 0 <= target < lane_count)
                    if direction else None)
    applied = int(info.get("exit_applied_lane_change_direction", 0))
    called = applied != 0
    return {
        "controller_mode": "no_pending",
        "longitudinal": {
            "command": "setAcceleration",
            "called": bool(ego["valid"]),
            "acceleration_mps2": float(parsed[0]),
            "duration_s": 1.0,
            "speed_cap_setSpeed_called": bool(
                ego["valid"] and ego["speed_mps"] is not None
                and ego["speed_mps"] > EGO_SPEED_CAP_MPS),
            "speed_cap_mps": EGO_SPEED_CAP_MPS,
        },
        "lateral": {
            "actor_direction": direction,
            "eligible_lane_change_edge": eligible_edge,
            "target_lane_index": target if target_valid else None,
            "candidate_target_lane_index": target,
            "target_lane_valid": target_valid,
            "boundary_limited": bool(direction and eligible_edge and target_valid is False),
            "ineligible_edge": bool(direction and not eligible_edge),
            "traci_command": "changeLane" if called else None,
            "traci_command_called": called,
            "command_direction": applied,
            "command_duration_s": (float(exit_config.LANE_CHANGE_TIME_SECONDS)
                                    if called else None),
            "python_call_returned_without_exception": True if called else None,
            "sumo_acceptance_known": False if called else None,
        },
    }


def _flatten_completeness(value, prefix=""):
    pairs = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "_base":
                continue
            child_prefix = f"{prefix}.{key}" if prefix else key
            pairs.extend(_flatten_completeness(child, child_prefix))
    elif isinstance(value, list):
        pairs.append((prefix, True))
        for child in value:
            if isinstance(child, dict):
                pairs.extend(_flatten_completeness(child, f"{prefix}[]"))
    else:
        pairs.append((prefix, value is not None))
    return pairs


class TrajectoryDatasetWriter:
    """Write a new dataset directory without overwriting existing artifacts."""

    EPISODE_FIELDS = (
        "episode_id", "eval_seed", "steps", "outcome", "success", "return",
        "ego_teleported", "toward_requests", "away_requests",
        "lane_change_commands_sent", "toward_completed_changes",
        "away_completed_changes", "lane_change_direction_reversals",
        "same_direction_repeat_requests", "short_horizon_request_reversals_1s",
        "mean_speed_mps", "trajectory_file", "observation_file", "derived_file",
    )

    def __init__(self, output_dir, metadata, neighbor_range_m=150.0):
        self.output_dir = Path(output_dir)
        self.metadata = _json_value(metadata)
        if self.metadata.get("controller_mode") != "no_pending":
            raise ValueError("PbRL P0 writer requires controller_mode=no_pending")
        self.neighbor_range_m = float(neighbor_range_m)
        if not math.isfinite(self.neighbor_range_m) or self.neighbor_range_m <= 0:
            raise ValueError("neighbor_range_m must be finite and positive")
        self.episode_index = 0
        self.total_steps = 0
        self.completeness = Counter()
        self.present = Counter()
        self.files = []
        self.episode_provenance = []
        self.current = None
        self._finalized = False

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for name in ("trajectories", "observations", "derived"):
            (self.output_dir / name).mkdir()
        self.episodes_stream = (self.output_dir / "episodes.csv").open(
            "x", newline="", encoding="utf-8")
        self.episodes = csv.DictWriter(self.episodes_stream,
                                       fieldnames=self.EPISODE_FIELDS)
        self.episodes.writeheader()
        self.events = (self.output_dir / "lane_change_events.jsonl").open(
            "x", encoding="utf-8")
        (self.output_dir / "schema.md").write_text(self.schema_document(),
                                                   encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.current is not None:
            self._close_episode_streams()
        self.events.close()
        self.episodes_stream.close()
        if not self._finalized and exc is not None:
            self.finalize(status="failed",
                          error=f"{type(exc).__name__}: {exc}")

    def begin_episode(self, eval_seed, reset_info, env):
        if self.current is not None:
            raise RuntimeError("Previous trajectory episode was not finalized")
        episode_id = f"ep-{self.episode_index:04d}-seed-{eval_seed}"
        paths = {
            "trajectory": self.output_dir / "trajectories" / f"{episode_id}.jsonl.gz",
            "observation": self.output_dir / "observations" / f"{episode_id}.jsonl.gz",
            "derived": self.output_dir / "derived" / f"{episode_id}.jsonl.gz",
        }
        streams = {name: gzip.open(path, "xt", encoding="utf-8", newline="\n")
                   for name, path in paths.items()}
        self.current = {"episode_id": episode_id, "eval_seed": int(eval_seed),
                        "reset_info": _json_value(reset_info), "paths": paths,
                        "streams": streams, "steps": 0, "last_request": 0,
                        "last_completed_direction": None}
        self.episode_provenance.append({
            "episode_id": episode_id,
            "eval_seed": int(eval_seed),
            "reset_info": _json_value(reset_info),
        })
        self.episode_index += 1
        return episode_id

    def capture_pre(self, env, observation, actor_action, step):
        if self.current is None:
            raise RuntimeError("begin_episode must be called before capture_pre")
        base = env.unwrapped
        episode_id = self.current["episode_id"]
        eval_seed = self.current["eval_seed"]
        step_id = f"{episode_id}-step-{step:05d}"
        pre = _state_bundle(base, self.neighbor_range_m)
        observation_record = _observation_record(
            env, observation, step_id, episode_id, eval_seed, step)
        return {"step_id": step_id, "episode_id": episode_id,
                "eval_seed": eval_seed, "step": int(step),
                "sim_time_s": _valid_number(base.traci.simulation.getTime()),
                "pre": pre, "observation": observation_record,
                "actor_action": np.asarray(actor_action, dtype=float).tolist()}

    def record_transition(self, captured, env, next_observation, reward,
                          terminated, truncated, info):
        if self.current is None or captured["episode_id"] != self.current["episode_id"]:
            raise RuntimeError("Trajectory transition does not match active episode")
        base = env.unwrapped
        post = _state_bundle(base, self.neighbor_range_m)
        info_for_controller = dict(info)
        info_for_controller["_base"] = base
        actor_action = captured["actor_action"]
        parsed = parseAction(actor_action)
        lane_event = _json_value(info.get("directional_lane_change_event"))
        pre_ego, post_ego = captured["pre"]["ego"], post["ego"]
        nominal_lane_delta = None
        if (pre_ego["valid"] and post_ego["valid"]
                and pre_ego["road_id"] == post_ego["road_id"]):
            nominal_lane_delta = post_ego["lane_index"] - pre_ego["lane_index"]
        transition = {
            "schema_version": SCHEMA_VERSION,
            "step_id": captured["step_id"],
            "episode_id": captured["episode_id"],
            "eval_seed": captured["eval_seed"],
            "checkpoint": self.metadata.get("checkpoint"),
            "checkpoint_sha256": self.metadata.get("checkpoint_sha256"),
            "step": captured["step"],
            "sim_time_pre_s": captured["sim_time_s"],
            "sim_time_post_s": _valid_number(base.traci.simulation.getTime()),
            "scenario": self.metadata.get("scenario"),
            "controller_mode": "no_pending",
            "observation_step_id": captured["step_id"],
            "pre": captured["pre"],
            "post": post,
            "sac": {
                "actor_action_raw": actor_action,
                "longitudinal_acceleration_mps2": float(parsed[0]),
                "lateral_direction": int(parsed[1]),
                "delay_steps": captured["observation"]["delay_steps"],
                "delay_seconds": captured["observation"]["delay_seconds"],
                "action_history_step_id": captured["step_id"],
            },
            "controller": _controller_record(captured["pre"], actor_action,
                                               info_for_controller),
            "execution": {
                "nominal_lane_index_delta": nominal_lane_delta,
                "pre_lateral_motion": bool(
                    pre_ego["valid"] and pre_ego["lateral_speed_mps"] is not None
                    and abs(pre_ego["lateral_speed_mps"]) > 0.05),
                "post_lateral_motion": bool(
                    post_ego["valid"] and post_ego["lateral_speed_mps"] is not None
                    and abs(post_ego["lateral_speed_mps"]) > 0.05),
                "lane_change_completed": lane_event is not None,
                "completed_direction": (lane_event.get("actual_direction")
                                        if lane_event is not None else None),
                "lane_change_completion_event": lane_event,
            },
            "reward": {
                "components": _json_value(info.get("reward_components", {})),
                "step_reward": float(reward),
            },
            "events": {
                "collision": info.get("message") == "Collision",
                "arrived": info.get("message") == "Arrived",
                "missed_exit": bool(info.get("missed_exit", False)),
                "teleport_started": bool(info.get("teleport_started", False)),
                "teleport_ended": bool(info.get("teleport_ended", False)),
                "ego_teleported": bool(info.get("ego_teleported", False)),
                "timeout": info.get("message") == "Max Steps",
                "emergency_stopping": info.get("message") == "Emergency Stopping",
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "terminal_reason": (info.get("message")
                                    if terminated or truncated else None),
            },
        }
        pre_acc = pre_ego.get("acceleration_mps2")
        post_acc = post_ego.get("acceleration_mps2")
        jerk = ((post_acc - pre_acc) / SUMO_STEP_LENGTH
                if pre_acc is not None and post_acc is not None else None)
        request = int(parsed[1])
        repeated = bool(request and request == self.current["last_request"])
        completed_direction = transition["execution"]["completed_direction"]
        reversed_change = bool(
            completed_direction is not None
            and self.current["last_completed_direction"] is not None
            and completed_direction != self.current["last_completed_direction"])
        derived = {
            "derivation_version": DERIVATION_VERSION,
            "step_id": captured["step_id"],
            "episode_id": captured["episode_id"],
            "eval_seed": captured["eval_seed"],
            "step": captured["step"],
            "jerk_mps3": _valid_number(jerk),
            "actor_request_repeated_from_previous_step": repeated,
            "completed_lane_change_reversed_from_previous_completion": reversed_change,
            "pre_route_requirement_n_changes": captured["pre"]["route"][
                "route_requirement_n_changes"],
            "post_route_requirement_n_changes": post["route"][
                "route_requirement_n_changes"],
            "post_direct_exit_route": post["route"]["has_direct_exit_route"],
            "post_direct_route_stable": bool(
                post["route"]["has_direct_exit_route"] is True
                and post_ego["valid"]
                and post_ego["lateral_speed_mps"] is not None
                and abs(post_ego["lateral_speed_mps"]) <= 0.05),
        }
        self.current["last_request"] = request
        if completed_direction is not None:
            self.current["last_completed_direction"] = completed_direction
            event_record = {
                "step_id": captured["step_id"],
                "episode_id": captured["episode_id"],
                "eval_seed": captured["eval_seed"],
                "step": captured["step"],
                "event": lane_event,
            }
            self.events.write(json.dumps(event_record, allow_nan=False) + "\n")
            self.events.flush()
        for namespace, value in (("transition", transition),
                                 ("observation", captured["observation"]),
                                 ("derived", derived)):
            for key, is_present in _flatten_completeness(value, namespace):
                self.completeness[key] += 1
                self.present[key] += int(is_present)
        self.current["streams"]["trajectory"].write(
            json.dumps(_json_value(transition), allow_nan=False) + "\n")
        self.current["streams"]["observation"].write(
            json.dumps(_json_value(captured["observation"]), allow_nan=False) + "\n")
        self.current["streams"]["derived"].write(
            json.dumps(_json_value(derived), allow_nan=False) + "\n")
        self.current["steps"] += 1
        self.total_steps += 1
        return transition

    def finish_episode(self, summary):
        if self.current is None:
            raise RuntimeError("No active trajectory episode")
        paths = self.current["paths"]
        directional = summary.get("lane_change_counts", {})
        row = {
            "episode_id": self.current["episode_id"],
            "eval_seed": self.current["eval_seed"],
            "steps": self.current["steps"],
            "outcome": summary.get("outcome"),
            "success": summary.get("success"),
            "return": summary.get("return"),
            "ego_teleported": summary.get("ego_teleported"),
            "toward_requests": summary.get("toward_requests"),
            "away_requests": summary.get("away_requests"),
            "lane_change_commands_sent": summary.get("lane_change_commands_sent"),
            "toward_completed_changes": directional.get("toward", 0),
            "away_completed_changes": directional.get("away", 0),
            "lane_change_direction_reversals": summary.get(
                "lane_change_direction_reversals", 0),
            "same_direction_repeat_requests": summary.get(
                "same_direction_repeat_requests"),
            "short_horizon_request_reversals_1s": summary.get(
                "short_horizon_request_reversals_1s"),
            "mean_speed_mps": summary.get("mean_speed_mps"),
            "trajectory_file": str(paths["trajectory"].relative_to(self.output_dir)),
            "observation_file": str(paths["observation"].relative_to(self.output_dir)),
            "derived_file": str(paths["derived"].relative_to(self.output_dir)),
        }
        self.episodes.writerow(row)
        self.episodes_stream.flush()
        self._close_episode_streams()

    def _close_episode_streams(self):
        for stream in self.current["streams"].values():
            stream.close()
        self.files.extend(self.current["paths"].values())
        self.current = None

    def finalize(self, report=None, status="complete", error=None):
        if self._finalized:
            return
        completeness = {
            key: {"present": self.present[key], "total": total,
                  "fraction": self.present[key] / total if total else None}
            for key, total in sorted(self.completeness.items())
        }
        completeness_path = self.output_dir / "field_completeness.json"
        completeness_path.write_text(
            json.dumps({"schema_version": SCHEMA_VERSION,
                        "total_steps": self.total_steps,
                        "fields": completeness}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        all_files = [self.output_dir / "episodes.csv",
                     self.output_dir / "lane_change_events.jsonl",
                     self.output_dir / "schema.md", completeness_path, *self.files]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "derivation_version": DERIVATION_VERSION,
            "status": status,
            "error": error,
            "metadata": self.metadata,
            "evaluation_configuration": ({
                key: _json_value(report[key]) for key in (
                    "protocol", "scenario", "task", "stage_name", "delay_mode",
                    "ablation", "training_safety", "evaluation_safety",
                    "potential_gamma", "exit_observation_profile", "nav_features",
                    "exit_reward_profile", "road_speed_limit_mps", "e09_pending_mode",
                    "progress_experiment", "progress_reward_identity",
                    "progress_experiment_identity", "gru_input_mode",
                    "antwerp_env_version", "training_seed", "training_timesteps",
                    "deterministic", "evaluation_seeds", "episodes_per_seed",
                    "total_episodes") if report is not None and key in report
            } if report is not None else None),
            "episode_provenance": self.episode_provenance,
            "controller_mode": "no_pending",
            "simulation_step_seconds": SUMO_STEP_LENGTH,
            "episode_count": self.episode_index,
            "total_steps": self.total_steps,
            "neighbor_snapshot_range_m": self.neighbor_range_m,
            "evaluation_report_sha256": (_sha256_json(report) if report is not None else None),
            "files": [{"path": str(path.relative_to(self.output_dir)),
                       "size_bytes": path.stat().st_size,
                       "sha256": _file_sha256(path)}
                      for path in all_files if path.exists()],
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        self._finalized = True

    @staticmethod
    def schema_document():
        return """# DAROM PbRL P0 trajectory schema v1

Each trajectory row is one transition `(s_t, a_t, command_t, r_t, s_t+1)`.
`step_id` joins trajectory, observation, derived-feature, and lane-change files.
Missing or inapplicable values are JSON `null`; validity and `missing_reason`
fields explain state/route/neighbor absence. No missing number is replaced by 0.

## Files

- `episodes.csv`: one row per complete evaluation episode.
- `trajectories/*.jsonl.gz`: raw Ego, route, neighbor, action, command, reward,
  physical execution, and terminal-event transitions.
- `observations/*.jsonl.gz`: the exact policy input, received delayed entity
  observation, delay, and raw/masked action history.
- `derived/*.jsonl.gz`: versioned jerk, repeated-request, route-stability, and
  lane-change-reversal features. These are not raw simulator fields.
- `lane_change_events.jsonl`: stable physical completion events only.
- `field_completeness.json`: non-null counts for scalar transition,
  observation, and derived-feature fields.
- `manifest.json`: provenance, hashes, sizes, counts, and derivation versions.

## Units and semantics

- distance/position/gap: metres (`*_m`); speed: m/s (`*_mps`);
  acceleration: m/s² (`*_mps2`); jerk: m/s³ (`*_mps3`); time/TTC: seconds.
- SUMO headings are degrees clockwise from north.
- lateral direction follows this project: `-1` toward D3/right, `0` hold,
  `+1` away from D3/left on E76/E77.
- `python_call_returned_without_exception` means only that the TraCI call
  returned. It is not SUMO acceptance and not physical lane-change completion.
- `lane_change_completed` is emitted only by the existing stable physical
  completion detector; no fixed four-second completion inference is used.
- candidate-lane `*_min_gap_available` compares geometric bumper gap with the
  trailing vehicle's configured `minGap`. It is a diagnostic, not a SUMO
  safety-controller decision; E10 evaluates with `laneChangeMode=0`.
- `policy_observation` is the exact array passed to the actor. It is unitless
  mixed-format model input; raw physical units remain in trajectory state.
"""
