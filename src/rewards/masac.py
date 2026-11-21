"""Modular reward accounting for the MASAC environment."""

from dataclasses import dataclass
from math import inf, isfinite

from .events import STALLED_EVENT, STALLED_PENALTY, event_reward


TASK_EVENTS = {
    "success": 50.0,
    "collision": -500.0,
    "missed_exit": -400.0,
    "timeout": -400.0,
    "stalled": STALLED_PENALTY,
}
TARGET_EXIT_DIM = 2
LEGACY_MASAC_ENV_VERSION = 2
ROUTE_AWARE_MASAC_ENV_VERSION = 3
STALL_GRACE_MASAC_ENV_VERSION = 4
MASAC_ENV_VERSION = 5
SUPPORTED_MASAC_ENV_VERSIONS = (
    LEGACY_MASAC_ENV_VERSION, ROUTE_AWARE_MASAC_ENV_VERSION, STALL_GRACE_MASAC_ENV_VERSION, MASAC_ENV_VERSION,
)
# Shared reference for gap scoring, independent of SUMO vehicle braking limits.
LANE_CHANGE_REFERENCE_DECEL = 4.5
FOLLOWING_WEIGHT = 0.1


@dataclass(frozen=True)
class MASACRewardConfig:
    task: bool = True
    merge: bool = True
    following: bool = True
    lane_change: bool = True
    efficiency: bool = True
    comfort: bool = True
    team: bool = True
    route_guidance: bool = False
    ttc_threshold: float = 3.0
    following_lookahead: float = 120.0
    lane_change_weight: float = 1.0
    rear_following: bool = True

    def __post_init__(self):
        for name in ("ttc_threshold", "following_lookahead", "lane_change_weight"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class MASACRewardProtocol:
    merge_on_success: bool = True
    efficiency_weight: float = 0.01
    route_guidance_weight: float = 5.0

    def __post_init__(self):
        for name in ("efficiency_weight", "route_guidance_weight"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


LEGACY_MASAC_REWARD_PROTOCOL = MASACRewardProtocol(
    merge_on_success=False, efficiency_weight=0.05, route_guidance_weight=0.0,
)


@dataclass
class LaneChange:
    road: str
    source_lane: int
    target_lane: int
    direction: int
    deadline: float
    last_motion: float | None = None
    max_risk: float = 0.0


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
    def __init__(self, config, protocol=MASACRewardProtocol()):
        self.config = config
        self.protocol = protocol
        self.components = (
            "task", "merge", "following", "rear_following", "lane_change",
            "efficiency", "comfort", "route_guidance",
        )
        self.reset(())

    def reset(self, ego_ids):
        self.merged = set()
        self.guidance_paid = set()
        self.pending_changes = {}
        self.lane_change_events = []
        self.following_risks = {}
        self.samples = {}
        self.totals = {ego_id: self.empty() for ego_id in ego_ids}
        self.team_return = 0.0

    def empty(self):
        return dict.fromkeys(self.components, 0.0)

    def terminal(self, event):
        if self.config.task or event == STALLED_EVENT:
            return event_reward(event, {"events": TASK_EVENTS})[1]
        return 0.0

    def settle_merge(self, ego_id, event):
        if self.protocol.merge_on_success and ego_id in self.merged:
            self.merged.remove(ego_id)
            if event == "Arrived":
                return 25.0
        return 0.0

    def score_route_guidance(self, ego_id, connection, accepted):
        if self.config.route_guidance and connection in accepted and ego_id not in self.guidance_paid:
            self.guidance_paid.add(ego_id)
            return self.protocol.route_guidance_weight
        return 0.0

    def _following_risks(self, gap, behind_speed, ahead_speed):
        gap_risk = gap_deficit(gap, max(1.5 * behind_speed, 1e-6))
        closing_speed = behind_speed - ahead_speed
        ttc = max(gap, 0.0) / closing_speed if closing_speed > 0 else inf
        return gap_risk, gap_deficit(ttc, self.config.ttc_threshold)

    def before_step(self, conn, ego_id, control):
        vehicle = conn.vehicle
        speed = (vehicle.getSpeed(ego_id) if any((
            self.config.efficiency, self.config.following, self.config.rear_following,
            self.config.lane_change,
        )) else 0.0)
        values = self.empty()
        if self.config.efficiency:
            values["efficiency"] = self.protocol.efficiency_weight * min(1.0, speed / 20.0)
        gap_risk = ttc_risk = 0.0
        rear_gap_risk = rear_ttc_risk = 0.0
        if self.config.following:
            leader = vehicle.getLeader(ego_id, self.config.following_lookahead)
            if leader and leader[0]:
                # TraCI reports the bumper gap minus the following vehicle's minGap.
                gap = leader[1] + vehicle.getMinGap(ego_id)
                gap_risk, ttc_risk = self._following_risks(gap, speed, vehicle.getSpeed(leader[0]))
                values["following"] = -FOLLOWING_WEIGHT * max(gap_risk, ttc_risk)
        if self.config.rear_following:
            follower = vehicle.getFollower(ego_id, self.config.following_lookahead)
            if follower and follower[0]:
                follower_id, distance = follower
                gap = distance + vehicle.getMinGap(follower_id)
                rear_gap_risk, rear_ttc_risk = self._following_risks(
                    gap, vehicle.getSpeed(follower_id), speed,
                )
                values["rear_following"] = -FOLLOWING_WEIGHT * max(rear_gap_risk, rear_ttc_risk)
        self.following_risks[ego_id] = {
            "gap_risk": gap_risk, "ttc_risk": ttc_risk,
            "following_reward": values["following"],
            "rear_gap_risk": rear_gap_risk, "rear_ttc_risk": rear_ttc_risk,
            "rear_following_reward": values["rear_following"],
        }

        risk = 0.0
        if self.config.lane_change or control is not None:
            now = conn.simulation.getTime()
            road = vehicle.getRoadID(ego_id)
            lane = vehicle.getLaneIndex(ego_id)
            change = self.pending_changes.get(ego_id)
            if change and change.last_motion is None and (
                change.road != road or now > change.deadline + 1e-6
            ):
                self.end_change(ego_id, now, "request_expired")
                change = None
            if control.target_lane is not None and (change is None or change.last_motion is None):
                target = control.target_lane
                if target != lane:
                    change = LaneChange(
                        road, lane, target, 1 if target > lane else -1,
                        now + control.lane_duration,
                    )
                    self.pending_changes[ego_id] = change
            if self.config.lane_change and change and road == change.road:
                risk = self._lane_change_risk(conn, ego_id, change, lane, speed)

        acceleration = vehicle.getAcceleration(ego_id) if self.config.comfort else None
        self.samples[ego_id] = values, acceleration, risk

    @staticmethod
    def _lane_change_risk(conn, ego_id, change, lane, speed):
        vehicle = conn.vehicle
        if lane == change.target_lane:
            # Left/right queries are relative to the *current* lane index.
            # Once it flips, use vehicles on the fixed target lane instead.
            position = vehicle.getLanePosition(ego_id)
            ego_length = vehicle.getLength(ego_id)
            risk = 0.0
            for other in conn.lane.getLastStepVehicleIDs(f"{change.road}_{change.target_lane}"):
                if other == ego_id:
                    continue
                delta = vehicle.getLanePosition(other) - position
                other_speed = vehicle.getSpeed(other)
                if delta >= 0:
                    gap = delta - vehicle.getLength(other)
                    required = required_gap(speed, other_speed)
                else:
                    gap = -delta - ego_length
                    required = required_gap(other_speed, speed)
                risk = max(risk, gap_deficit(gap, required))
            return risk

        if lane != change.source_lane:
            return 0.0
        leaders = vehicle.getLeftLeaders if change.direction > 0 else vehicle.getRightLeaders
        followers = vehicle.getLeftFollowers if change.direction > 0 else vehicle.getRightFollowers
        front = max((
            gap_deficit(gap + vehicle.getMinGap(ego_id), required_gap(speed, vehicle.getSpeed(other)))
            for other, gap in leaders(ego_id)
        ), default=0.0)
        rear = max((
            gap_deficit(gap + vehicle.getMinGap(other), required_gap(vehicle.getSpeed(other), speed))
            for other, gap in followers(ego_id)
        ), default=0.0)
        return max(front, rear)

    def _log_change(self, ego_id, change, now, event, reason=None):
        self.lane_change_events.append({
            "ego_id": ego_id, "time": now, "event": event, "reason": reason,
            "road": change.road, "source_lane": change.source_lane,
            "target_lane": change.target_lane, "direction": change.direction,
            "max_risk": change.max_risk,
            "cumulative_penalty": -self.config.lane_change_weight * change.max_risk,
        })

    def end_change(self, ego_id, now, reason, event="interrupted"):
        change = self.pending_changes.pop(ego_id, None)
        if change is not None and change.last_motion is not None:
            self._log_change(ego_id, change, now, event, reason)

    def _settle_lane_change(self, conn, ego_id, on_road, risk, terminal_event):
        change = self.pending_changes.get(ego_id)
        if change is None:
            return 0.0

        now = conn.simulation.getTime()
        vehicle = conn.vehicle
        same_road = on_road and vehicle.getRoadID(ego_id) == change.road
        end_reason = terminal_event or ("edge_changed" if on_road else "vehicle_removed")
        end_event = "interrupted"
        if same_road:
            lane = vehicle.getLaneIndex(ego_id)
            lateral_speed = vehicle.getLateralSpeed(ego_id)
            toward_target = lateral_speed * change.direction
            if change.last_motion is None and (
                toward_target > 1e-6 or lane == change.target_lane
            ):
                change.last_motion = now
                self._log_change(ego_id, change, now, "started")

            if change.last_motion is not None:
                if self.config.lane_change:
                    risk = max(risk, self._lane_change_risk(
                        conn, ego_id, change, lane, vehicle.getSpeed(ego_id),
                    ))
                if toward_target > 1e-6:
                    change.last_motion = now
                if terminal_event is None:
                    clearance = (conn.lane.getWidth(vehicle.getLaneID(ego_id))
                                 - vehicle.getWidth(ego_id)) / 2
                    settled = (
                        abs(lateral_speed) <= 0.05
                        and abs(vehicle.getLateralLanePosition(ego_id)) <= clearance + 1e-3
                    )
                    if lane == change.target_lane and settled:
                        end_event, end_reason = "completed", "settled_in_target"
                    elif toward_target < -0.05 or (lane == change.source_lane and settled):
                        end_reason = "reversed_or_returned"
                    elif now - change.last_motion >= 2.0:
                        end_reason = "lateral_motion_stopped"
                    elif lane not in (change.source_lane, change.target_lane):
                        end_reason = "left_target_lane"
                    else:
                        end_reason = None
            elif terminal_event is None:
                end_reason = "request_expired" if now > change.deadline + 1e-6 else None

        penalty = 0.0
        if change.last_motion is not None:
            previous = change.max_risk
            change.max_risk = max(previous, risk)
            penalty = -self.config.lane_change_weight * (change.max_risk - previous)
        if end_reason is not None:
            self.end_change(ego_id, now, end_reason, end_event)
        return penalty

    def after_step(self, conn, ego_id, on_road, needs_merge, terminal_event=None):
        values, acceleration, risk = self.samples.pop(ego_id)
        values["lane_change"] = self._settle_lane_change(
            conn, ego_id, on_road, risk, terminal_event,
        )
        if not on_road:
            return values

        vehicle = conn.vehicle
        if self.config.comfort:
            delta = abs(vehicle.getAcceleration(ego_id) - acceleration)
            values["comfort"] = -0.01 * min(1.0, delta)

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
                self.merged.add(ego_id)
                if not self.protocol.merge_on_success:
                    values["merge"] = 25.0
        return values

    def settle_team(self, reasons):
        if self.config.team and not self.team_return and all(
            reason == "Arrived" for reason in reasons.values()
        ):
            self.team_return = 100.0
            return self.team_return
        return 0.0
