"""E0 privileged reachability diagnostic for the fixed E76-to-D3 task.

The controller is deliberately not an RL policy.  It reads current SUMO state
and the parsed route graph, but sends only the environment's ordinary two
action channels.  It never changes a vehicle position, speed limit, route, or
lane directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import exit_e76_d3_config as exit_config
from .merging import ANTWERP_DIR, LANE_CHANGE_EDGES, Merging, maxSteps, parseAction, traci
from .route_urgency import load_route_network, observe_route_urgency


DEFAULT_LANES = tuple(range(5))
DEFAULT_SEEDS = tuple(range(100001, 100011))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict[str, Any]:
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status_porcelain": subprocess.check_output(["git", "status", "--porcelain"], text=True),
    }


def finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {field}: {value!r}")
    return result


def json_safe(value: Any) -> Any:
    """Make diagnostic-only infinite gaps explicit JSON nulls."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


@dataclass
class TopologyController:
    """One-lane-at-a-time controller driven by ``route_urgency`` plans."""

    config: dict[str, Any]
    network: Any
    target_lane: int | None = None
    target_edge: str | None = None

    def _gaps(self, ego: str, direction: int) -> tuple[float, float, float, float]:
        # TraCI neighbor bits: 0/1 = left/right, 2 = leaders.  Values are
        # net longitudinal gaps, which is exactly the SUMO lane-change input.
        side = 0 if direction > 0 else 1
        leaders = traci.vehicle.getNeighbors(ego, side | 2)
        followers = traci.vehicle.getNeighbors(ego, side)
        front = min((finite(gap, "front gap") for _, gap in leaders), default=math.inf)
        rear = min((finite(gap, "rear gap") for _, gap in followers), default=math.inf)
        front_speed = min((traci.vehicle.getSpeed(veh) for veh, _ in leaders), default=math.inf)
        rear_speed = max((traci.vehicle.getSpeed(veh) for veh, _ in followers), default=0.0)
        return front, rear, front_speed, rear_speed

    def __call__(self, base: Merging) -> tuple[np.ndarray, dict[str, Any]]:
        details: dict[str, Any] = {"phase": "off_road", "target_lane": None,
                                   "front_gap_m": None, "rear_gap_m": None,
                                   "route_status": "off_road", "block_reason": ""}
        if not base.ego_on_road:
            return np.zeros(2, dtype=np.float32), details
        ego = base.ego_id
        road = traci.vehicle.getRoadID(ego)
        lane = int(traci.vehicle.getLaneIndex(ego))
        speed = finite(traci.vehicle.getSpeed(ego), "ego speed")
        urgency = observe_route_urgency(base, self.network, LANE_CHANGE_EDGES,
                                        lane_change_seconds=self.config["lane_change_seconds"])
        details["route_status"] = urgency["status"]
        if road != self.target_edge:
            self.target_edge, self.target_lane = road, None
        plan = urgency.get("plan", [])
        desired = lane
        if plan and urgency["status"] == "ok":
            desired = int(plan[0]["exit_lane"])
        # A running four-second environment maneuver owns the lateral command;
        # this controller merely waits.  This is strategy-side de-duplication,
        # not the historical Pending lock (which remains off in the environment).
        maneuver_active = bool(getattr(base, "exit_maneuver", None))
        lateral_speed = abs(finite(traci.vehicle.getLateralSpeed(ego), "lateral speed"))
        if self.target_lane == lane and lateral_speed <= 0.05:
            self.target_lane = None
        if self.target_lane is None and desired != lane:
            self.target_lane = lane + int(np.sign(desired - lane))
        target = self.target_lane if self.target_lane is not None else lane
        direction = int(np.sign(target - lane)) if road in LANE_CHANGE_EDGES else 0
        details.update(phase="keep_lane", target_lane=target, desired_exit_lane=desired,
                       maneuver_active=maneuver_active)
        front, rear, front_speed, rear_speed = self._gaps(ego, direction) if direction else (math.inf, math.inf, math.inf, 0.0)
        details.update(front_gap_m=front, rear_gap_m=rear)
        acceleration = np.clip(0.7 * (self.config["target_speed_mps"] - speed),
                               self.config["max_braking_mps2"], self.config["max_acceleration_mps2"])
        leader = traci.vehicle.getLeader(ego, 150.0)
        if leader:
            leader_speed = finite(traci.vehicle.getSpeed(leader[0]), "leader speed")
            needed = self.config["front_gap_base_m"] + self.config["headway_seconds"] * speed
            if leader[1] < needed:
                acceleration = min(acceleration, np.clip((leader_speed - speed) / 0.5,
                                                          self.config["max_braking_mps2"], 0.0))
        if direction:
            required_front = max(self.config["front_gap_base_m"] + self.config["headway_seconds"] * speed,
                                 self.config["gap_buffer_m"] + 2.0 * max(speed - front_speed, 0.0))
            required_rear = max(self.config["rear_gap_base_m"] + self.config["headway_seconds"] * rear_speed,
                                self.config["gap_buffer_m"] + 2.0 * max(rear_speed - speed, 0.0))
            if maneuver_active or lateral_speed > 0.05:
                direction, details["phase"] = 0, "maneuver_or_settling"
            elif front < required_front or rear < required_rear:
                direction, details["phase"] = 0, "waiting_gap"
                details["block_reason"] = "+".join(part for part, blocked in
                                                      (("front", front < required_front), ("rear", rear < required_rear)) if blocked)
                if rear < required_rear:
                    acceleration = min(acceleration, -1.5)
            else:
                details["phase"] = "request_lane_change"
        return np.asarray([acceleration, 5.0 * direction], dtype=np.float32), details


def _state(base: Merging) -> dict[str, Any]:
    if not base.ego_on_road:
        return {"road": "", "lane": "", "lane_index": None, "lane_position_m": None,
                "speed_mps": None, "acceleration_mps2": None, "leader_gap_m": None,
                "follower_gap_m": None, "remaining_to_d3_m": None}
    if not traci.isLoaded():
        # ``Merging.collect_step`` closes SUMO after a terminal transition.
        # Preserve its final cached state without issuing a post-close query.
        return {"road": base.lane_name, "lane": base.lane_id, "lane_index": base.lane_idx,
                "lane_position_m": None, "speed_mps": float(base.observation[0, 2]),
                "acceleration_mps2": None, "leader_gap_m": None, "follower_gap_m": None,
                "remaining_to_d3_m": None}
    ego = base.ego_id
    leader = traci.vehicle.getLeader(ego, 200.0)
    follower = traci.vehicle.getFollower(ego, 200.0)
    remaining = base._remaining_to_d3_end()
    return {"road": traci.vehicle.getRoadID(ego), "lane": traci.vehicle.getLaneID(ego),
            "lane_index": int(traci.vehicle.getLaneIndex(ego)),
            "lane_position_m": finite(traci.vehicle.getLanePosition(ego), "lane position"),
            "speed_mps": finite(traci.vehicle.getSpeed(ego), "speed"),
            "acceleration_mps2": finite(traci.vehicle.getAcceleration(ego), "acceleration"),
            "leader_gap_m": None if not leader else finite(leader[1], "leader gap"),
            "follower_gap_m": None if not follower else finite(follower[1], "follower gap"),
            "remaining_to_d3_m": None if remaining is None else finite(remaining, "remaining distance")}


def audit_transition(step: int, reward: float, terminated: bool, truncated: bool,
                     info: dict[str, Any]) -> None:
    event = info["message"]
    if truncated:
        raise RuntimeError("B0 task terminal must not use truncated=True")
    if not terminated:
        if reward != 0.0 or event:
            raise RuntimeError(f"non-terminal B0 reward/event at step {step}: {reward}, {event!r}")
        return
    expected = 50.0 if event == "Arrived" else -100.0
    if event not in {"Arrived", "Collision", "Ego Teleported", "Missed Exit", "Timeout"}:
        raise RuntimeError(f"unknown terminal event at step {step}: {event!r}")
    if reward != expected:
        raise RuntimeError(f"terminal B0 reward mismatch at step {step}: {reward} != {expected}")
    terms = info["reward_episode_summary"]
    if terms is None or terms["_meta"]["weighted_total"] != expected:
        raise RuntimeError("RewardTracker terminal total mismatch")


def run_episode(*, lane: int, seed: int, config: dict[str, Any], output_dir: Path,
                network: Any, run_label: str) -> dict[str, Any]:
    case_dir = output_dir / "episodes" / f"lane-{lane}_seed-{seed}"
    case_dir.mkdir(parents=True, exist_ok=False)
    base = Merging(seed=seed, scenario=config["scenario"], runSafetyCheck=False,
                   reward_config=config["reward_config"], reward_gamma=config["gamma"])
    base.curriculum_depart_lane = lane
    base.curriculum_route_dir = output_dir / "generated_routes"
    controller = TopologyController(config, network)
    total_reward = 0.0
    discounted_reward = 0.0
    speeds: list[float] = []
    requests = commands = completed = toward = away = reversals = 0
    last_change_direction = 0
    first_toward_distance = first_stable_distance = None
    emergency_steps = hard_braking_steps = 0
    boundary = Counter()
    outcome = "Environment Exception"
    trace_path = case_dir / "trace.jsonl"
    try:
        _, reset_info = base.reset(seed=seed)
        if reset_info["ego_start_lane"] != lane:
            raise RuntimeError(f"requested {lane}, SUMO entered {reset_info['ego_start_lane']}")
        if traci.vehicle.getLaneChangeMode(base.ego_id) != 0:
            raise RuntimeError("laneChangeMode is not 0")
        control_modes = {"laneChangeMode": traci.vehicle.getLaneChangeMode(base.ego_id),
                         "speedMode": traci.vehicle.getSpeedMode(base.ego_id)}
        with trace_path.open("x") as trace:
            for step in range(1, maxSteps + 1):
                before = _state(base)
                simulation_time = traci.simulation.getTime()
                action, decision = controller(base)
                parsed = parseAction(action)
                _, reward, terminated, truncated, info = base.step(action)
                audit_transition(step, reward, terminated, truncated, info)
                after = _state(base)
                request_direction = int(info["exit_actor_request_direction"])
                command_direction = int(info["exit_applied_lane_change_direction"])
                actual_direction = 0
                if before["road"] == after["road"] and before["lane_index"] is not None and after["lane_index"] is not None:
                    actual_direction = int(np.sign(after["lane_index"] - before["lane_index"]))
                requests += int(request_direction != 0)
                commands += int(command_direction != 0)
                completed += int(actual_direction != 0)
                toward += int(actual_direction < 0)  # D3 route plans on this network move toward lower lane indices.
                away += int(actual_direction > 0)
                if actual_direction and last_change_direction and actual_direction != last_change_direction:
                    reversals += 1
                if actual_direction:
                    last_change_direction = actual_direction
                remaining = before["remaining_to_d3_m"]
                if first_toward_distance is None and actual_direction < 0:
                    first_toward_distance = remaining
                if first_stable_distance is None and after["road"] in {"E76", "E77"} and after["lane_index"] == 0:
                    first_stable_distance = after["remaining_to_d3_m"]
                speeds.extend(value for value in (before["speed_mps"], after["speed_mps"]) if value is not None)
                emergency_steps += int(info["sumo_emergency_stopping_this_step"])
                hard_braking_steps += int(info["hard_braking_this_step"])
                status = (info.get("road_boundary") or {}).get("status", "unknown")
                boundary[status] += 1
                row = {"step": step, "simulation_time_s": simulation_time + 0.1, "before": before,
                       "after": after, "rule_action": [float(action[0]), float(action[1])],
                       "parsed_action": [float(parsed[0]), int(parsed[1])], "decision": decision,
                       "lane_change_request_direction": request_direction,
                       "lane_change_command_direction": command_direction,
                       "actual_lane_change_direction": actual_direction,
                       "pending_or_maneuver_active": bool(info["exit_maneuver_active"]),
                       "reward": reward, "reward_terms": info["reward_terms"],
                       "terminal_event": info["message"], "road_boundary": info.get("road_boundary"),
                       "sumo_emergency_stopping": info["sumo_emergency_stopping_this_step"],
                       "hard_braking": info["hard_braking_this_step"]}
                trace.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
                total_reward += reward
                discounted_reward += (config["gamma"] ** (step - 1)) * reward
                if terminated or truncated:
                    outcome = info["message"]
                    break
            else:
                raise RuntimeError("environment did not terminate by its configured step limit")
        if outcome == "Timeout" and step != config["episode_limit_steps"]:
            raise RuntimeError(f"timeout at {step}, expected {config['episode_limit_steps']}")
        summary = {"experiment_id": run_label, "scenario": config["scenario"], "seed": seed,
                   "ego_start_lane": f"E76_{lane}", "final_outcome": outcome,
                   "success": outcome == "Arrived" and not info["ego_teleported"], "episode_steps": step,
                   "total_reward": total_reward, "discounted_reward": discounted_reward,
                   "final_road": after["road"], "final_lane": after["lane"],
                   "mean_speed_mps": float(np.mean(speeds)) if speeds else None,
                   "max_speed_mps": max(speeds, default=None), "lane_change_requests": requests,
                   "lane_change_commands_sent": commands, "completed_lane_changes": completed,
                   "toward_lane_changes": toward, "away_lane_changes": away,
                   "direction_reversals": reversals, "first_toward_lane_change_distance": first_toward_distance,
                   "first_stable_exit_lane_distance": first_stable_distance,
                   "sumo_emergency_stopping_steps": emergency_steps, "hard_braking_steps": hard_braking_steps,
                   "road_boundary_status": dict(boundary), "reward_tracker": info["reward_episode_summary"],
                   "sumo_seed": reset_info["sumo_seed"], "sumo_version": reset_info["sumo_version"],
                   "control_modes": control_modes, "trace": str(trace_path)}
        summary = json_safe(summary)
        (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return summary
    finally:
        base.close()


def write_aggregate(results: list[dict[str, Any]], output_dir: Path) -> None:
    fields = list(results[0]) if results else []
    with (output_dir / "episodes.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    counts: dict[str, dict[str, int]] = {}
    for lane in DEFAULT_LANES:
        selected = [row for row in results if row["ego_start_lane"] == f"E76_{lane}"]
        counts[f"E76_{lane}"] = {key: sum(row["final_outcome"] == key for row in selected)
                                  for key in ("Arrived", "Collision", "Missed Exit", "Ego Teleported", "Timeout", "Environment Exception")}
        counts[f"E76_{lane}"]["episodes"] = len(selected)
    (output_dir / "outcome_counts.json").write_text(json.dumps(counts, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/e0_reachability_v1.json"))
    parser.add_argument("--lanes", type=int, nargs="+", default=list(DEFAULT_LANES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--run-label", default="e0-reachability")
    args = parser.parse_args()
    if any(lane not in DEFAULT_LANES for lane in args.lanes) or len(set(args.lanes)) != len(args.lanes):
        parser.error("--lanes must be unique integers in [0, 4]")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be nonempty and unique")
    config = json.loads(args.config.read_text())
    if config["reward_config"] != "configs/rewards/events_only_v1.json":
        raise ValueError("E0 accepts only the B0 events_only_v1 reward configuration")
    if config["episode_limit_steps"] != maxSteps or config["gamma"] != 0.999:
        raise ValueError("E0 configuration must retain the B0 time and gamma contract")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    network_file = ANTWERP_DIR / "Antwerp.net.xml"
    route_file = ANTWERP_DIR / "Antwerp_easy_exit_e76_d3.rou.xml"
    metadata = {**git_state(), "config": config, "config_sha256": sha256(args.config),
                "network_sha256": sha256(network_file), "route_sha256": sha256(route_file),
                "lanes": args.lanes, "seeds": args.seeds, "run_label": args.run_label}
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    network = load_route_network(network_file)
    results = [run_episode(lane=lane, seed=seed, config=config, output_dir=args.output_dir,
                           network=network, run_label=args.run_label)
               for lane in args.lanes for seed in args.seeds]
    write_aggregate(results, args.output_dir)


if __name__ == "__main__":
    main()
