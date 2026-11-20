"""Read-only Stalled-v1 diagnostics for one Ego."""

from collections import deque
import math

from traci.exceptions import TraCIException


STEP_SECONDS = 0.1
WINDOW_SECONDS = 10.0
WINDOW_STEPS = round(WINDOW_SECONDS / STEP_SECONDS)
MIN_PROGRESS_M = 20.0
LEADER_RANGE_M = 50.0
JUNCTION_APPROACH_M = 20.0
TOLERANCE = 1e-6


class StalledDetector:
    """Track a sliding window of uninterrupted passage opportunity."""

    def __init__(self):
        self.samples = deque(maxlen=WINDOW_STEPS + 1)
        self.trigger_time = None

    def observe(self, time, *, road=None, lane=None, distance=None,
                leader=None, unavailable_reason=None):
        reason = unavailable_reason
        if reason is None and (distance is None or not math.isfinite(distance) or distance < 0):
            reason = "invalid_distance"

        progress = None
        window_seconds = 0.0
        if reason is not None:
            self.samples.clear()
        else:
            if self.samples:
                previous_time, previous_distance = self.samples[-1]
                if (abs(time - previous_time - STEP_SECONDS) > TOLERANCE
                        or distance < previous_distance - TOLERANCE):
                    self.samples.clear()
                    reason = "discontinuous_sample"
            self.samples.append((time, distance))
            start_time, start_distance = self.samples[0]
            progress = max(0.0, distance - start_distance)
            window_seconds = round(time - start_time, 6)
            if reason is None:
                if len(self.samples) < WINDOW_STEPS + 1:
                    reason = "collecting_window"
                elif progress >= MIN_PROGRESS_M - TOLERANCE:
                    reason = "progress_sufficient"
                else:
                    reason = "insufficient_progress"

        would_stall = reason == "insufficient_progress"
        if would_stall and self.trigger_time is None:
            self.trigger_time = time
        elif not would_stall:
            self.trigger_time = None

        return {
            "would_stall": would_stall,
            "trigger_time_s": self.trigger_time,
            "sample_time_s": time,
            "road": road,
            "lane": lane,
            "distance_m": distance,
            "window_progress_m": progress,
            "valid_window_s": window_seconds,
            "leader": leader,
            "reason": reason,
        }

    def sample(self, conn, ego_id, time):
        """Read one pre-action SUMO state. Missing information never counts as clear."""
        vehicle = conn.vehicle
        road = lane = distance = leader = None
        try:
            road = vehicle.getRoadID(ego_id)
            lane = vehicle.getLaneID(ego_id)
            raw_distance = vehicle.getDistance(ego_id)
            distance = raw_distance if math.isfinite(raw_distance) and raw_distance >= 0 else None
            min_gap = vehicle.getMinGap(ego_id)
            found = vehicle.getLeader(ego_id, LEADER_RANGE_M)
            if found and found[0]:
                # TraCI measures from the front bumper plus minGap.
                gap = found[1] + min_gap
                if gap <= LEADER_RANGE_M:
                    leader = {"id": found[0], "gap_m": gap, "speed_mps": vehicle.getSpeed(found[0])}
            reason = self.passage_block(conn, ego_id, road, lane, leader, min_gap)
        except TraCIException:
            reason = "traci_unavailable"

        return self.observe(time, road=road, lane=lane, distance=distance,
                            leader=leader, unavailable_reason=reason)

    @staticmethod
    def passage_block(conn, ego_id, road, lane, leader, min_gap):
        vehicle = conn.vehicle
        if vehicle.getStopState(ego_id) & 1:
            return "scheduled_stop"

        if leader is not None:
            gap = leader["gap_m"]
            # Estimate whether following the leader leaves room for 20 m of progress.
            available = gap - min_gap + max(0.0, leader["speed_mps"]) * WINDOW_SECONDS
            if gap <= min_gap or available < MIN_PROGRESS_M:
                return "leader_constrained"

        remaining = conn.lane.getLength(lane) - vehicle.getLanePosition(ego_id)
        if remaining > JUNCTION_APPROACH_M and not road.startswith(":"):
            return None

        if any(0 <= distance <= JUNCTION_APPROACH_M and state in "rRyYu"
               for _, _, distance, state in vehicle.getNextTLS(ego_id)):
            return "traffic_signal"

        if not road.startswith(":"):
            links = vehicle.getNextLinks(ego_id)
            if links:
                # Python TraCI returns (lane, priority, opened, foe, via, ...).
                _, priority, opened, foe, _, _, _, _ = links[0]
                if not opened or (foe and not priority):
                    return "junction_conflict"
            elif road != vehicle.getRoute(ego_id)[-1]:
                return "route_link_unavailable"

        # getNextLinks skips the current internal link. Look at the actual
        # crossing vehicle and its right of way near this junction.
        for foe_id, ego_dist, foe_dist, ego_exit, foe_exit, _, _, ego_yields, _ in (
            vehicle.getJunctionFoes(ego_id, JUNCTION_APPROACH_M)
        ):
            if ego_exit <= 0 or ego_dist > JUNCTION_APPROACH_M or foe_exit <= 0:
                continue
            if foe_dist <= 0 or (
                ego_yields and foe_dist <= JUNCTION_APPROACH_M and vehicle.getSpeed(foe_id) > 0.1
            ):
                return "junction_conflict"
        return None
