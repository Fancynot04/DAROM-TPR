"""Stream evaluation-only TraCI observations without changing policy inputs."""

import gzip
import json
from math import isfinite

from traci.exceptions import TraCIException

from src.experiment_tracking import sha256


TRAJECTORY_FILENAME = "eval_trajectory.jsonl.gz"


class EvaluationTrajectory:
    def __init__(self, temporary_path, agent_ids):
        self.path = temporary_path
        self.agent_ids = agent_ids
        self.stream = gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=3)
        self.episodes = 0
        self.global_step = 0
        self.seed = None

    def _write(self, **payload):
        self.stream.write(json.dumps({
            "phase": "eval", "episode": self.episodes, "seed": self.seed,
            "global_step": self.global_step, **payload,
        }, separators=(",", ":"), allow_nan=False) + "\n")

    def reset(self, seed, sumo_version):
        self.episodes += 1
        self.seed = seed
        self._write(event="reset", sumo=sumo_version)

    @staticmethod
    def positions(vehicle, ego_ids):
        positions = {}
        for ego_id in ego_ids:
            try:
                xy = vehicle.getPosition(ego_id)
                along = vehicle.getLanePosition(ego_id)
                offset = vehicle.getLateralLanePosition(ego_id)
                speed = vehicle.getLateralSpeed(ego_id)
            except TraCIException:
                continue
            if all(isfinite(value) for value in (*xy, along, offset, speed)):
                positions[ego_id] = {
                    "xy_m": xy, "lane_position_m": along,
                    "lateral_position_m": offset, "lateral_speed_mps": speed,
                }
        return positions

    def step(self, env, step, actions, before, rewards, infos, prior_outcomes, *, decision_ids=None):
        if decision_ids is None:
            decision_ids = actions
        vehicle = env.world.conn.vehicle
        after = self.positions(vehicle, (ego for ego in self.agent_ids if infos[ego]["status"] == "active"))
        self.global_step += 1
        self._write(event="step", step=step, team_reward=env.team_reward, egos={
            ego: {
                "action": actions.get(ego), "actor_decision": ego in decision_ids,
                "pre_position": before.get(ego),
                "post_position": after.get(ego),
                "metrics": infos[ego].get("evaluation_metrics"),
                "control": infos[ego].get("control_trace"),
                "following": infos[ego].get("following_risks"),
                "lane_changes": infos[ego].get("lane_change_events", []),
                "route_preparation": infos[ego].get("route_preparation"),
                "stalled_shadow": infos[ego].get("stalled_shadow"),
                "remaining_goal_distance_m": infos[ego].get("remaining_goal_distance_m"),
                "signed_min_lane_changes": infos[ego].get("signed_min_lane_changes"),
                "reward": rewards[ego], "components": infos[ego].get("reward_components"),
                "terminal": infos[ego].get("terminal_reason") if ego not in prior_outcomes else None,
            } for ego in self.agent_ids
        })

    def close(self):
        self.stream.close()

    def publish(self, output):
        target = output.with_name(TRAJECTORY_FILENAME)
        digest = sha256(self.path)
        self.path.replace(target)
        return {"file": target.name, "sha256": digest, "format": "dual-ego-eval-jsonl-gzip-v1",
                "episodes": self.episodes, "step_records": self.global_step}
