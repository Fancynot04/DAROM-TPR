"""Read-only progress detection for one Ego."""

from collections import Counter, deque
import math

from traci.exceptions import TraCIException
from src.rewards.masac import MASAC_ENV_VERSION, STALL_GRACE_MASAC_ENV_VERSION


STEP_SECONDS = 0.1
WINDOW_SECONDS = 10.0
WINDOW_STEPS = round(WINDOW_SECONDS / STEP_SECONDS)
MIN_PROGRESS_M = 20.0
LEADER_RANGE_M = 50.0
JUNCTION_APPROACH_M = 20.0
TOLERANCE = 1e-6
BLOCK_GRACE_SECONDS = 3.0  # Initial value, not calibrated on trajectories.
PASSAGE_BLOCK_REASONS = frozenset({
    "scheduled_stop", "leader_constrained", "traffic_signal", "junction_conflict",
    "route_link_unavailable",
})


def stalled_block_grace_for_version(env_version, saved_grace_s=None):
    if env_version in (STALL_GRACE_MASAC_ENV_VERSION, MASAC_ENV_VERSION):
        return BLOCK_GRACE_SECONDS if saved_grace_s is None else saved_grace_s
    return 0.0


class StalledDetector:
    """Track a sliding window of passage opportunity."""

    def __init__(self, block_grace_s=0.0):
        if not math.isfinite(block_grace_s) or block_grace_s < 0:
            raise ValueError("Stalled block grace must be finite and nonnegative")
        self.block_grace_s = block_grace_s
        self.samples = deque(maxlen=WINDOW_STEPS + 1)
        self.trigger_time = None
        self.clear_progress = deque(maxlen=WINDOW_STEPS)
        self.pending_progress = []
        self.previous_raw = None
        self.block_started = None
        self.block_reason = None
        self.first_clear_time = None
        self.block_cleared_window = False
        self.pause_count = 0
        self.resets = Counter()
        self.block_durations = Counter()
        self.censored_block_durations = Counter()
        self.raw_block_started = None
        self.raw_block_durations = Counter()

    def observe(self, time, *, road=None, lane=None, distance=None,
                leader=None, unavailable_reason=None):
        if self.block_grace_s:
            reason, progress, window_seconds = self._observe_with_grace(
                time, distance, unavailable_reason,
            )
        else:
            reason, progress, window_seconds = self._observe_legacy(
                time, distance, unavailable_reason,
            )

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

    def _observe_legacy(self, time, distance, unavailable_reason):
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

        return reason, progress, window_seconds

    def _reset_window(self, reason):
        if self.clear_progress or self.pending_progress:
            self.resets[reason] += 1
        self.clear_progress.clear()
        self.pending_progress.clear()

    def _finish_block(self, end_time, *, censored=False):
        if self.block_started is not None:
            elapsed_steps = round((end_time - self.block_started) / STEP_SECONDS)
            durations = self.censored_block_durations if censored else self.block_durations
            durations[elapsed_steps] += 1
        self.block_started = self.first_clear_time = None
        self.block_reason = None
        self.block_cleared_window = False

    def _observe_with_grace(self, time, distance, reason):
        previous = self.previous_raw
        if not math.isfinite(time):
            reason = "discontinuous_sample"
        if reason is None or reason in PASSAGE_BLOCK_REASONS:
            if distance is None or not math.isfinite(distance) or distance < 0:
                reason = "invalid_distance"
        if reason is not None and reason not in PASSAGE_BLOCK_REASONS:
            self._reset_window(reason)
            # A missing sample cannot establish how long the blocker persisted.
            if previous is not None:
                self._censor_block(previous[0])
            self.raw_block_started = None
            self.previous_raw = None
            return reason, None, 0.0

        blocked = reason is not None
        self.previous_raw = (time, distance, blocked)
        if previous is not None and (
            abs(time - previous[0] - STEP_SECONDS) > TOLERANCE
            or distance < previous[1] - TOLERANCE
        ):
            self._reset_window("discontinuous_sample")
            self._censor_block(previous[0])
            self.raw_block_started = time if blocked else None
            if blocked:
                self._start_block(time, reason)
            return "discontinuous_sample", 0.0, 0.0

        if blocked:
            if self.raw_block_started is None:
                self.raw_block_started = time
            if self.block_started is None:
                self._start_block(time, reason)
            self.first_clear_time = None
            self.pending_progress.clear()
            self._clear_long_block(time)
        elif self.block_started is not None:
            if self.first_clear_time is None:
                raw_steps = round((time - self.raw_block_started) / STEP_SECONDS)
                self.raw_block_durations[raw_steps] += 1
                self.raw_block_started = None
                self.first_clear_time = time
                self._clear_long_block(time)
            else:
                # Both endpoints are clear; credit only after recovery is confirmed.
                self.pending_progress.append(distance - previous[1])
            if time - self.first_clear_time >= self.block_grace_s - TOLERANCE:
                self.clear_progress.extend(self.pending_progress)
                self.pending_progress.clear()
                self._finish_block(self.first_clear_time)
            else:
                reason = "confirming_clear_passage"
        elif previous is not None and not previous[2]:
            self.clear_progress.append(distance - previous[1])

        progress = max(0.0, math.fsum(self.clear_progress))
        window_seconds = round(len(self.clear_progress) * STEP_SECONDS, 6)
        if reason is None:
            if len(self.clear_progress) < WINDOW_STEPS:
                reason = "collecting_window"
            elif progress >= MIN_PROGRESS_M - TOLERANCE:
                reason = "progress_sufficient"
            else:
                reason = "insufficient_progress"
        return reason, progress, window_seconds

    def _start_block(self, time, reason):
        self.block_started = time
        self.block_reason = reason
        self.pause_count += 1

    def _censor_block(self, previous_time):
        end_time = self.first_clear_time if self.first_clear_time is not None else previous_time
        self._finish_block(end_time, censored=True)

    def _clear_long_block(self, time):
        if not self.block_cleared_window and time - self.block_started >= self.block_grace_s - TOLERANCE:
            self._reset_window(f"long_block:{self.block_reason}")
            self.block_cleared_window = True

    def summary(self):
        """Snapshot per-episode counters, keeping an unresolved blocker separate."""
        open_block = None
        if self.block_started is not None:
            end_time = self.first_clear_time if self.first_clear_time is not None else self.previous_raw[0]
            open_block = round(end_time - self.block_started, 6)
        return {
            "window_resets": sum(self.resets.values()),
            "window_resets_by_reason": dict(self.resets),
            "pause_count": self.pause_count,
            "block_duration_histogram_steps": {
                str(steps): count for steps, count in sorted(self.block_durations.items())
            },
            "raw_block_duration_histogram_steps": {
                str(steps): count for steps, count in sorted(self.raw_block_durations.items())
            },
            "censored_block_duration_histogram_steps": {
                str(steps): count for steps, count in sorted(self.censored_block_durations.items())
            },
            "open_block_observed_s": open_block,
        }

    def sample(self, conn, ego_id, time, route_plan=None):
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
            reason = self.passage_block(conn, ego_id, road, lane, leader, min_gap, route_plan)
        except TraCIException:
            reason = "traci_unavailable"

        return self.observe(time, road=road, lane=lane, distance=distance,
                            leader=leader, unavailable_reason=reason)

    @staticmethod
    def passage_block(conn, ego_id, road, lane, leader, min_gap, route_plan=None):
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

        if route_plan is not None:
            reason, internal_lanes = route_plan.junction_passage(
                vehicle, ego_id, vehicle.getRouteIndex(ego_id), road, lane,
            )
            if reason is not None or not internal_lanes:
                return reason
            return StalledDetector.junction_foe_block(vehicle, ego_id, internal_lanes)

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

        return StalledDetector.junction_foe_block(vehicle, ego_id)

    @staticmethod
    def junction_foe_block(vehicle, ego_id, internal_lanes=None):
        for foe_id, ego_dist, foe_dist, ego_exit, foe_exit, ego_lane, _, ego_yields, _ in (
            vehicle.getJunctionFoes(ego_id, JUNCTION_APPROACH_M)
        ):
            if internal_lanes is not None and ego_lane not in internal_lanes:
                continue
            if ego_exit <= 0 or ego_dist > JUNCTION_APPROACH_M or foe_exit <= 0:
                continue
            if foe_dist <= 0 or (
                ego_yields and foe_dist <= JUNCTION_APPROACH_M and vehicle.getSpeed(foe_id) > 0.1
            ):
                return "junction_conflict"
        return None
