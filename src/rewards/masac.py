"""Modular reward accounting for the O3 -> D3 MASAC environment."""

from dataclasses import dataclass, fields
from math import inf

from .events import event_reward


TASK_EVENTS = {
    "success": 50.0,
    "collision": -500.0,
    "missed_exit": -400.0,
    "timeout": -400.0,
    "teleport": 0.0,
}
TARGET_EXIT_DIM = 2
# Shared reference for gap scoring, independent of SUMO vehicle braking limits.
LANE_CHANGE_REFERENCE_DECEL = 4.5


@dataclass(frozen=True)
class MASACRewardConfig:
    task: bool = True
    merge: bool = True
    following: bool = True
    lane_change: bool = True
    efficiency: bool = True
    comfort: bool = True
    team: bool = True


def gap_deficit(gap, required):
    return min(1.0, max(0.0, 1.0 - gap / required))


def required_gap(behind_speed, ahead_speed):
    # Use the same 1.5 s threshold for the lane-change reaction time.
    return 2.0 + max(
        0.0,
        1.5 * behind_speed
        + (behind_speed ** 2 - ahead_speed ** 2) / (2 * LANE_CHANGE_REFERENCE_DECEL),
    )


class MASACRewards:
    def __init__(self, config):
        self.config = config
        self.components = tuple(field.name for field in fields(config) if field.name != "team")
        self.reset(())

    def reset(self, ego_ids):
        self.merged = set()
        self.pending_changes = {}
        self.lateral_directions = {}
        self.samples = {}
        self.totals = {ego_id: self.empty() for ego_id in ego_ids}
        self.team_return = 0.0

    def empty(self):
        return dict.fromkeys(self.components, 0.0)

    def terminal(self, event):
        return event_reward(event, {"events": TASK_EVENTS})[1] if self.config.task else 0.0

    def before_step(self, conn, ego_id, control):
        vehicle = conn.vehicle
        speed = (vehicle.getSpeed(ego_id) if any((
            self.config.efficiency, self.config.following, self.config.lane_change,
        )) else 0.0)
        values = self.empty()
        if self.config.efficiency:
            values["efficiency"] = 0.01 * min(1.0, speed / 20.0)
        if self.config.following:
            leader = vehicle.getLeader(ego_id, 1.5 * speed)
            if leader and leader[0]:
                gap = leader[1] + vehicle.getMinGap(ego_id)
                values["following"] = -0.1 * gap_deficit(gap, max(1.5 * speed, 1e-6))

        road = vehicle.getRoadID(ego_id) if self.config.lane_change else None
        lane = vehicle.getLaneIndex(ego_id) if self.config.lane_change else None
        risk = 0.0
        direction = 0
        if self.config.lane_change:
            now = conn.simulation.getTime()
            if control.target_lane is not None:
                self.pending_changes[ego_id] = (
                    road, control.target_lane, now + control.lane_duration,
                )
            pending = self.pending_changes.get(ego_id)
            if pending:
                source_road, target_lane, deadline = pending
                if road != source_road or now > deadline or target_lane == lane:
                    self.pending_changes.pop(ego_id)
                else:
                    direction = 1 if target_lane > lane else -1
                    risk = self._lane_change_risk(vehicle, ego_id, direction, speed)
        acceleration = vehicle.getAcceleration(ego_id) if self.config.comfort else None
        self.samples[ego_id] = values, acceleration, road, lane, direction, risk

    @staticmethod
    def _lane_change_risk(vehicle, ego_id, direction, speed):
        leaders = vehicle.getLeftLeaders if direction > 0 else vehicle.getRightLeaders
        followers = vehicle.getLeftFollowers if direction > 0 else vehicle.getRightFollowers
        front = min((
            (gap + vehicle.getMinGap(ego_id)) / required_gap(speed, vehicle.getSpeed(other))
            for other, gap in leaders(ego_id)
        ), default=inf)
        rear = min((
            (gap + vehicle.getMinGap(other)) / required_gap(vehicle.getSpeed(other), speed)
            for other, gap in followers(ego_id)
        ), default=inf)
        return max(0.0, 1.0 - min(front, rear))

    def after_step(self, conn, ego_id, on_road, needs_merge):
        values, acceleration, road, lane, direction, risk = self.samples.pop(ego_id)
        if not on_road:
            return values

        vehicle = conn.vehicle
        if self.config.comfort:
            delta = abs(vehicle.getAcceleration(ego_id) - acceleration)
            values["comfort"] = -0.01 * min(1.0, delta)

        if self.config.lane_change:
            lateral_speed = vehicle.getLateralSpeed(ego_id)
            motion = (lateral_speed > 1e-6) - (lateral_speed < -1e-6)
            previous = self.lateral_directions.get(ego_id, 0)
            if direction and vehicle.getRoadID(ego_id) == road:
                crossed = (vehicle.getLaneIndex(ego_id) - lane) * direction > 0
                if previous != direction and (motion == direction or crossed):
                    values["lane_change"] = -0.1 * risk
                    self.pending_changes.pop(ego_id, None)
            self.lateral_directions[ego_id] = motion

        if self.config.merge and needs_merge and ego_id not in self.merged:
            current_road = vehicle.getRoadID(ego_id)
            # E75_0 is the auxiliary lane. Entering E76 also completes the
            # merge via its downstream connection from E75_0.
            merged = current_road == "E76"
            if current_road == "E75" and vehicle.getLaneIndex(ego_id) > 0:
                lane_id = vehicle.getLaneID(ego_id)
                clearance = (conn.lane.getWidth(lane_id) - vehicle.getWidth(ego_id)) / 2
                merged = abs(vehicle.getLateralLanePosition(ego_id)) <= clearance + 1e-3
            if merged:
                values["merge"] = 25.0
                self.merged.add(ego_id)
        return values

    def settle_team(self, reasons):
        if self.config.team and not self.team_return and all(
            reason == "Arrived" for reason in reasons.values()
        ):
            self.team_return = 100.0
            return self.team_return
        return 0.0
