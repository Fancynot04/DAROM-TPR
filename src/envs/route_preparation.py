"""Stage-local route preparation and its independent reward accounting."""

from collections import defaultdict, deque
from dataclasses import dataclass
from math import isfinite

from traci.constants import INVALID_DOUBLE_VALUE
from traci.exceptions import TraCIException

from src.rewards.masac import adjacent_lane_risk

from .control import MAX_SPEED
from .sumo_world import STEP_LENGTH_S

DEFAULT_ROUTE_PREPARATION_STAGES = [
    {
        "route": ["O3", "E75", "E76", "E77", "D3"],
        "preparation_edge": "E75",
        "phases": [
            {"kind": "weaving", "critical_connection": ["E76", "E77"],
             "accepted_entry_lanes": ["E77_0"], "include_adjacent_connections": True},
            {"kind": "exit", "critical_connection": ["E77", "D3"],
             "accepted_entry_lanes": ["D3_0"]},
        ],
    },
]
DEFAULT_ROUTE_PREPARATION_WEIGHTS = {"weaving": 0.001, "exit": 0.002}
DEFAULT_ROUTE_PREPARATION_BONUS = 0.005
PREPARATION_BONUS = "continuous_bonus"
PREPARATION_PROGRESS_BONUS = "progress_bonus"
PREPARATION_V1 = "first_progress_staircase"
PREPARATION_LINEAR_V1 = "linear_progress_v1"
DEFAULT_LINEAR_PREPARATION_SETTINGS = {
    "mode": PREPARATION_LINEAR_V1,
    "reward_scale": 0.02,
    "distance_scale_m": 1000.0,
    "reference_step_m": 1.0,
}
PREPARATION_DISTANCE_MODES = (PREPARATION_V1, PREPARATION_PROGRESS_BONUS)
PREPARATION_BONUS_MODES = (*PREPARATION_DISTANCE_MODES, PREPARATION_BONUS)
PREPARATION_PENALTY = "legacy_penalty"
DEFAULT_REFERENCE_STEP_M = MAX_SPEED * STEP_LENGTH_S
V1_REFERENCE_STEP_M = 3.333
V1_DEMAND_REDUCTION_REWARD = 0.1
V1_PREPARATION_ZONE_WEIGHT = 0.4
V1_CRUISE_WEIGHTS = (1.0, 0.5, 0.2)
LATERAL_MOTION_MPS = 0.05


def checkpoint_preparation_settings(settings):
    if settings is None:
        return None
    return {"mode": PREPARATION_PENALTY, **settings}


@dataclass
class PreparationPhase:
    kind: str
    approach: str
    next_edge: str
    distance_m: float
    weight: float
    demand: dict
    normalization: int
    improving_lanes: dict

    @classmethod
    def build(cls, spec, weight, network, plan):
        approach, next_edge = spec["critical_connection"]
        indices = [i for i, edge in enumerate(plan.route[:-1])
                   if edge == approach and plan.route[i + 1] == next_edge]
        if len(indices) != 1:
            raise ValueError(f"Preparation boundary {approach, next_edge} must appear once in {plan.route}")
        index = indices[0]
        allowed_entry = set(spec["accepted_entry_lanes"])
        entry_indices = {
            lane["id"]: lane_index for lane_index, lane in network.lanes[next_edge].items()
        }
        if not allowed_entry or not allowed_entry <= entry_indices.keys():
            raise ValueError(f"Invalid preparation entry lanes for {next_edge}: {allowed_entry}")
        reachable_entries = {lane for lane in allowed_entry
                             if (index + 1, entry_indices[lane]) in plan.signed_changes}
        zero = [
            lane_index for lane_index, lane in network.lanes[approach].items()
            if any(target in reachable_entries for target, _ in plan.passage_links.get((index, lane["id"]), {}))
        ]
        if not zero:
            raise ValueError(f"No permitted preparation connection for {plan.route} at {approach}")
        return index, cls.from_demand(
            spec["kind"], index, weight, network, plan, dict.fromkeys(zero, 0),
            spec.get("preparation_distance_m"),
        )

    @classmethod
    def from_demand(cls, kind, index, weight, network, plan, demand, distance_m=None):
        approach, next_edge = plan.route[index:index + 2]
        lanes = network.lanes[approach]
        if distance_m is None:
            distance_m = max(float(lane["length"]) for lane in lanes.values())
        reverse = defaultdict(list)
        for lane_index in lanes:
            for _, neighbor in plan.lateral_neighbors.get((index, lane_index), ()):
                reverse[neighbor].append(lane_index)
        queue = deque(demand)
        while queue:
            lane_index = queue.popleft()
            for previous in reverse[lane_index]:
                candidate = demand[lane_index] + 1
                if previous not in demand or candidate < demand[previous]:
                    demand[previous] = candidate
                    queue.append(previous)
        improving = {
            lane_index: tuple(neighbor for _, neighbor in plan.lateral_neighbors.get((index, lane_index), ())
                              if neighbor in demand and demand[neighbor] < count)
            for lane_index, count in demand.items()
        }
        return cls(kind, approach, next_edge, float(distance_m),
                   weight, demand, max(demand.values(), default=0), improving)

    @classmethod
    def build_preparation(cls, index, weaving_index, weaving, weight, network, plan):
        downstream = weaving.demand
        for i in range(weaving_index - 1, index - 1, -1):
            next_lanes = {lane["id"]: j for j, lane in network.lanes[plan.route[i + 1]].items()}
            demand = {}
            for lane_index, lane in network.lanes[plan.route[i]].items():
                links = plan.passage_links.get((i, lane["id"]), {})
                costs = [downstream[next_lanes[target]] for target, _ in links
                         if next_lanes[target] in downstream]
                if costs:
                    demand[lane_index] = min(costs)
            phase = cls.from_demand("preparation", i, weight, network, plan, demand)
            downstream = phase.demand
        return phase

    def penalty(self, distance, demand, safe_opportunity, moving):
        if not demand or not safe_opportunity or moving is not False or not self.weight:
            return 0.0
        urgency = min(1.0, max(0.0, 1.0 - distance / self.distance_m))
        return -self.weight * urgency * min(demand / self.normalization, 1.0)

    def reduces_demand(self, source, target):
        return target in self.improving_lanes.get(source, ())

    def readiness(self, demand):
        if demand is None:
            return None
        if self.normalization == 0:
            return 1.0
        return min(1.0, max(0.0, 1.0 - demand / self.normalization))


@dataclass
class PreparationSample:
    phase: PreparationPhase
    lane: int
    distance_m: float
    demand: int | None
    safe_opportunity: bool
    moving: bool
    time_s: float
    road: str
    connector: bool = False


@dataclass
class PreparationMotion:
    route_index: int
    road: str
    lane: int
    lane_id: str
    along: float | None
    speed: float | None
    lateral_speed: float | None
    time_s: float | None
    current_risk: float | None


@dataclass
class BonusSample:
    motion: PreparationMotion
    phase: PreparationPhase | None
    demand: int | None = None
    connection: tuple[str, str] | None = None


@dataclass
class LinearSample:
    motion: PreparationMotion
    distance_m: float
    demand: int | None


class RoutePreparation:
    def __init__(self, settings):
        self.mode = settings.get("mode", PREPARATION_PENALTY)
        self.linear = self.mode == PREPARATION_LINEAR_V1
        self.compiled = {}
        self.internal_offsets = {}
        self.reset(())
        if self.linear:
            if {**DEFAULT_LINEAR_PREPARATION_SETTINGS, **settings} != DEFAULT_LINEAR_PREPARATION_SETTINGS:
                raise ValueError("linear_progress_v1 requires its fixed reward settings")
            return
        if self.mode not in (*PREPARATION_BONUS_MODES, PREPARATION_PENALTY):
            raise ValueError(f"Invalid route preparation mode: {self.mode}")
        self.bonus_cap = settings.get("bonus_cap", DEFAULT_ROUTE_PREPARATION_BONUS)
        if not isfinite(self.bonus_cap) or self.bonus_cap < 0:
            raise ValueError("Preparation bonus cap must be finite and nonnegative")
        self.v1 = self.mode == PREPARATION_V1
        default_step = V1_REFERENCE_STEP_M if self.v1 else DEFAULT_REFERENCE_STEP_M
        self.reference_step_m = settings.get("reference_step_m", default_step)
        if not isfinite(self.reference_step_m) or self.reference_step_m <= 0:
            raise ValueError("Preparation reference step must be finite and positive")
        self.weights = (dict.fromkeys(DEFAULT_ROUTE_PREPARATION_WEIGHTS, self.bonus_cap)
                        if self.mode in PREPARATION_BONUS_MODES else settings["weights"])
        self.stages = {}
        self.preparation_edges = {}
        if set(self.weights) != set(DEFAULT_ROUTE_PREPARATION_WEIGHTS):
            raise ValueError("Preparation weights require exactly weaving and exit")
        if any(not isfinite(w) or w < 0 for w in self.weights.values()):
            raise ValueError("Preparation weights must be finite and nonnegative")
        for rule in settings["stages"]:
            route = tuple(rule["route"])
            if route in self.stages:
                raise ValueError(f"Duplicate route preparation rule: {route}")
            kinds = set()
            for spec in rule["phases"]:
                kind = spec["kind"]
                distance = spec.get("preparation_distance_m")
                if (kind not in self.weights or kind in kinds
                        or (distance is not None and (not isfinite(distance) or distance <= 0))):
                    raise ValueError(f"Invalid or duplicate preparation phase: {spec}")
                kinds.add(kind)
            self.stages[route] = rule["phases"]
            preparation_edge = rule.get("preparation_edge")
            if preparation_edge is not None:
                if preparation_edge not in route or any(
                    spec["critical_connection"][0] == preparation_edge for spec in rule["phases"]
                ):
                    raise ValueError(f"Invalid preparation edge: {preparation_edge}")
                self.preparation_edges[route] = preparation_edge

    def reset(self, ego_ids):
        self.ego_phases = {}
        self.ego_connectors = {}
        self.ego_preparation_edges = {}
        self.ego_plans = {}
        self.ego_linear_distances = {}
        self.unsettled_changes = {}
        self.progress_frontiers = {}
        self.minimum_demands = defaultdict(dict)
        self.diagnostics = {ego: {"enabled": False, "reason": "not_departed", "phases": {}}
                            for ego in ego_ids}
        self.step_info = {}

    def configure(self, ego_id, network, plan):
        if self.linear:
            self._configure_linear(ego_id, network, plan)
            return
        if plan not in self.compiled:
            phases = {}
            include_connections = {}
            for spec in self.stages.get(plan.route, ()):
                index, phase = PreparationPhase.build(spec, self.weights[spec["kind"]], network, plan)
                if index in phases:
                    raise ValueError(f"Preparation phases overlap at {phase.approach}")
                phases[index] = phase
                if spec.get("include_adjacent_connections", False):
                    include_connections[index] = set(spec["accepted_entry_lanes"])
            connectors = {}
            for (source, lane_id), destination in plan.internal.items():
                if source in include_connections:
                    phase = phases[source]
                    _, downstream_lane_id = plan.internal_connections[source, lane_id]
                    demand = 0 if downstream_lane_id in include_connections[source] else None
                    connectors[destination, lane_id] = (phase, demand, None,
                                                        plan.internal_connections[source, lane_id])
                elif destination[0] in include_connections:
                    phase = phases[destination[0]]
                    lane = destination[1]
                    length = float(network.lanes[phase.approach][lane]["length"])
                    connectors[destination, lane_id] = (phase, phase.demand.get(lane), length,
                                                        plan.internal_connections[source, lane_id])
            preparation_edge = self.preparation_edges.get(plan.route)
            if self.v1 and preparation_edge is not None:
                prep_index = plan.route.index(preparation_edge)
                for index, phase in tuple(phases.items()):
                    if phase.kind == "weaving" and index > prep_index:
                        phases[prep_index] = PreparationPhase.build_preparation(
                            prep_index, index, phase, self.bonus_cap, network, plan,
                        )
                        break
            self.compiled[plan] = phases, connectors
            if self.mode in PREPARATION_DISTANCE_MODES:
                offsets = {}
                for (index, _), connections in plan.passage_links.items():
                    for via in connections.values():
                        offset = 0.0
                        for lane_id in via:
                            offsets[index, lane_id] = offset
                            road, lane_index = lane_id.rsplit("_", 1)
                            offset += float(network.lanes[road][int(lane_index)]["length"])
                self.internal_offsets[plan] = offsets
        phases, connectors = self.compiled[plan]
        self.ego_phases[ego_id] = phases
        self.ego_connectors[ego_id] = connectors
        self.ego_plans[ego_id] = plan
        preparation_edge = self.preparation_edges.get(plan.route)
        self.ego_preparation_edges[ego_id] = preparation_edge
        enabled = bool(phases) if self.v1 else any(p.weight > 0 for p in phases.values())
        phase_stats = {}
        for phase in phases.values():
            stats = {
                "enabled": self.v1 or phase.weight > 0, "weight": phase.weight,
                "normalization": phase.normalization, "preparation_zone_steps": 0,
                "unreachable_steps": 0, "last_necessary_completion": None,
            }
            if self.mode in PREPARATION_BONUS_MODES:
                stats.update(bonus_steps=0, cumulative_bonus=0.0, safe_progress_steps=0)
            else:
                stats.update(safe_opportunity_steps=0, penalty_steps=0, cumulative_penalty=0.0, first_penalty=None)
            phase_stats[phase.kind] = stats
            if self.mode in PREPARATION_DISTANCE_MODES:
                stats["rewarded_progress_m"] = 0.0
            if self.v1:
                stats.update(cumulative_event_bonus=0.0, cumulative_cruise_bonus=0.0)
        self.diagnostics[ego_id] = {
            "enabled": enabled, "reason": "enabled" if enabled else "zero_weights" if phases else "unconfigured_route",
            "route": list(plan.route), "phases": phase_stats,
        }
        if self.mode in PREPARATION_BONUS_MODES:
            self.diagnostics[ego_id]["mode"] = self.mode
        if preparation_edge is not None:
            self.diagnostics[ego_id]["preparation_edge_steps"] = 0

    def _configure_linear(self, ego_id, network, plan):
        if plan not in self.compiled:
            distances = {}
            if plan.route[-2:] == ("E77", "D3"):
                remaining = 0.0
                for index in reversed(range(len(plan.route) - 1)):
                    lanes = network.lanes[plan.route[index]]
                    lengths = {lane["id"]: float(lane["length"]) for lane in lanes.values()}
                    distances[index] = remaining, lengths
                    remaining += max(lengths.values())
            self.compiled[plan] = distances
        distances = self.compiled[plan]
        self.ego_plans[ego_id] = plan
        self.ego_linear_distances[ego_id] = distances
        self.diagnostics[ego_id] = {
            "mode": self.mode, "route": list(plan.route), "enabled": bool(distances),
            "reason": "enabled" if distances else "unconfigured_route",
            "phases": {"preparation": {
                "preparation_zone_steps": 0, "unreachable_steps": 0,
                "bonus_steps": 0, "cumulative_bonus": 0.0,
            }} if distances else {},
        }

    def before_linear(self, conn, ego_id):
        distances = self.ego_linear_distances.get(ego_id)
        if not distances:
            return None
        motion = self._motion(conn, ego_id)
        if motion is None or motion.along is None or motion.route_index not in distances:
            return None
        plan = self.ego_plans[ego_id]
        remaining, lengths = distances[motion.route_index]
        if motion.road.startswith(":"):
            position = plan.internal[motion.route_index, motion.lane_id]
        else:
            position = motion.route_index, motion.lane
            remaining += max(0.0, lengths[motion.lane_id] - motion.along)
        signed_demand = plan.signed_changes.get(position)
        return LinearSample(motion, remaining, abs(signed_demand) if signed_demand is not None else None)

    def after_linear(self, conn, ego_id, sample, unavailable_reason):
        after = None if unavailable_reason else self._motion(conn, ego_id)
        before = sample.motion
        movement = self._progress(conn, self.ego_plans[ego_id], before, after) if after is not None else None
        progress = movement[0] if movement is not None else None
        if progress is not None and before.speed == after.speed == 0:
            progress = 0.0
        settings = DEFAULT_LINEAR_PREPARATION_SETTINGS
        distance_factor = min(1.0, sample.distance_m / settings["distance_scale_m"])
        progress_factor = min(1.0, (progress or 0.0) / settings["reference_step_m"])
        readiness = {0: 1.0, 1: 0.5}.get(sample.demand, 0.0)
        reward = settings["reward_scale"] * readiness * distance_factor * progress_factor
        self.step_info[ego_id] = {
            "mode": self.mode, "phase": "preparation", "time_s": before.time_s,
            "road": before.road, "lane": before.lane, "lane_id": before.lane_id,
            "post_lane_id": after.lane_id if after is not None else None,
            "local_lane_changes": sample.demand, "remaining_distance_m": sample.distance_m,
            "readiness": readiness, "distance_factor": distance_factor,
            "progress_m": progress, "progress_factor": progress_factor, "reward": reward,
            "unavailable_reason": unavailable_reason or ("unverified_progress" if progress is None else None),
        }
        stats = self.diagnostics[ego_id]["phases"]["preparation"]
        stats["preparation_zone_steps"] += 1
        stats["unreachable_steps"] += int(sample.demand is None)
        stats["bonus_steps"] += int(reward > 0)
        stats["cumulative_bonus"] += reward
        return reward

    @staticmethod
    def _safe_opportunity(vehicle, ego_id, lane, neighbors):
        try:
            speed = vehicle.getSpeed(ego_id)
        except TraCIException:
            return False
        if speed is None or not isfinite(speed) or speed < 0:
            return False
        for target in neighbors:
            try:
                if adjacent_lane_risk(vehicle, ego_id, target - lane, speed, require_valid=True) == 0:
                    return True
            except (TraCIException, ValueError):
                continue
        return False

    @staticmethod
    def _measurement(query, *, nonnegative=False):
        try:
            value = query()
            if (value is not None and value != INVALID_DOUBLE_VALUE and isfinite(value)
                    and (not nonnegative or value >= 0)):
                return value
        except (TraCIException, ValueError, TypeError):
            pass
        return None

    def _motion(self, conn, ego_id, rewards=None, road=None):
        vehicle = conn.vehicle
        try:
            plan = self.ego_plans[ego_id]
            if tuple(vehicle.getRoute(ego_id)) != plan.route:
                return None
            index = vehicle.getRouteIndex(ego_id)
            if road is None:
                road = vehicle.getRoadID(ego_id)
            lane_id = vehicle.getLaneID(ego_id)
            lane = vehicle.getLaneIndex(ego_id)
            if road is None or lane_id is None or index is None or lane is None:
                return None
            if road.startswith(":"):
                if (index, lane_id) not in plan.internal:
                    return None
            elif not (0 <= index < len(plan.route) and road == plan.route[index]
                      and (index, lane_id) in plan.passage_links):
                return None
        except (TraCIException, TypeError):
            return None
        along = self._measurement(lambda: vehicle.getLanePosition(ego_id), nonnegative=True)
        speed = self._measurement(lambda: vehicle.getSpeed(ego_id), nonnegative=True)
        lateral = self._measurement(lambda: vehicle.getLateralSpeed(ego_id)) if rewards is not None else None
        risk = (self._measurement(lambda: rewards.current_lane_risk(vehicle, ego_id, speed))
                if rewards is not None and speed is not None else None)
        now = self._measurement(conn.simulation.getTime, nonnegative=True)
        return PreparationMotion(index, road, lane, lane_id, along, speed, lateral, now, risk)

    def before_bonus(self, conn, ego_id, rewards):
        phases = self.ego_phases.get(ego_id, {})
        preparation_edge = self.ego_preparation_edges.get(ego_id)
        if not phases and preparation_edge is None:
            return None
        try:
            road = conn.vehicle.getRoadID(ego_id)
        except TraCIException:
            return None
        if road is None or not (road == preparation_edge or road.startswith(":")
                                or any(road == phase.approach for phase in phases.values())):
            return None
        motion = self._motion(conn, ego_id, rewards, road)
        if motion is None:
            return None
        plan = self.ego_plans[ego_id]
        index, lane_id = motion.route_index, motion.lane_id
        if road == preparation_edge and index not in phases:
            return BonusSample(motion, None)
        if motion.road.startswith(":"):
            position = plan.internal.get((index, lane_id))
            connector = self.ego_connectors[ego_id].get((position, lane_id))
            if connector is not None:
                phase, demand, _, connection = connector
                return BonusSample(motion, phase, demand, connection)
        else:
            phase = phases.get(index)
            if phase is not None and road == phase.approach:
                return BonusSample(motion, phase, phase.demand.get(motion.lane))
        return None

    def target_risk(self, conn, ego_id, motion, change, rewards):
        if change is None or motion.road != change.road or motion.speed is None:
            return None
        return self._measurement(lambda: rewards.lane_change_risk(
            conn, ego_id, change, motion.lane, motion.speed, require_valid=True,
        ))

    @staticmethod
    def _progress(conn, plan, before, after):
        if None in (before.along, after.along, before.speed, after.speed, before.time_s, after.time_s):
            return None
        dt = after.time_s - before.time_s
        if not isfinite(dt) or dt <= 0:
            return None
        source = before.route_index, before.lane_id
        destination = after.route_index, after.lane_id
        if source == destination or (
            before.route_index == after.route_index and before.road == after.road
            and not before.road.startswith(":")
            and (after.route_index, after.lane) in plan.lateral_neighbors.get(
                (before.route_index, before.lane), (),
            )
        ):
            distance = after.along - before.along
            spans = [(source, before.along, after.along)]
        else:
            via = plan.forward_passage(source, destination)
            if via is None:
                return None
            lengths = [RoutePreparation._measurement(lambda: conn.lane.getLength(lane), nonnegative=True)
                       for lane in (before.lane_id, *via)]
            if any(length is None for length in lengths) or before.along > lengths[0] + 1e-3:
                return None
            distance = sum(lengths) - before.along + after.along
            spans = [(source, before.along, lengths[0])]
            spans.extend(((before.route_index, lane), 0.0, length)
                         for lane, length in zip(via, lengths[1:]))
            spans.append((destination, 0.0, after.along))
        # Bound route displacement by the observed movement envelope, allowing
        # a small lane-coordinate projection difference during lateral motion.
        if distance < -1e-6 or distance > max(before.speed, after.speed) * dt + 0.15:
            return None
        return max(0.0, distance), spans

    def _first_progress(self, ego_id, before, after, spans):
        offsets = self.internal_offsets[self.ego_plans[ego_id]]

        def position(node, along):
            index, lane_id = node
            return index, lane_id.startswith(":"), offsets.get(node, 0.0) + along

        frontier = self.progress_frontiers.get(ego_id)
        if before.along is not None:
            start = position((before.route_index, before.lane_id), before.along)
            frontier = start if frontier is None else max(frontier, start)
        fresh = 0.0
        for node, start, end in spans:
            key = position(node, start)
            if frontier is None or key[:2] > frontier[:2]:
                fresh += max(0.0, end - start)
            elif key[:2] == frontier[:2]:
                fresh += max(0.0, end - start - max(0.0, frontier[2] - key[2]))
        if after is not None and after.along is not None:
            end = position((after.route_index, after.lane_id), after.along)
            frontier = end if frontier is None else max(frontier, end)
        if frontier is not None:
            self.progress_frontiers[ego_id] = frontier
        return fresh

    def bonus_change(self, ego_id, change):
        return self.unsettled_changes.get(ego_id, change)

    def _settle_bonus_change(self, conn, ego_id, after, rewards, change, completed):
        if completed is not None or after.road != change.road:
            self.unsettled_changes.pop(ego_id, None)
            return completed
        # A tracker interruption can precede physical settling. Keep the source
        # lane until the vehicle has settled in either lane of the maneuver.
        self.unsettled_changes[ego_id] = change
        settled = self._measurement(lambda: rewards.lane_change_settled(conn, ego_id, after.lateral_speed))
        if settled and after.lane in (change.source_lane, change.target_lane):
            self.unsettled_changes.pop(ego_id, None)
            if after.lane == change.target_lane:
                return {"road": change.road, "source_lane": change.source_lane,
                        "target_lane": change.target_lane, "time": after.time_s}
        return None

    def _first_demand_reduction(self, ego_id, sample, after, change, completed):
        phase, before = sample.phase, sample.motion
        if phase is None:
            return 0.0, None, None
        minima = self.minimum_demands[ego_id]
        observed = sample.demand
        if change is not None and before.road == change.road and change.last_motion is not None:
            observed = phase.demand.get(change.source_lane)
        if observed is not None:
            minima[phase.kind] = min(minima.get(phase.kind, observed), observed)
        previous = minima.get(phase.kind)
        if (completed is None or after is None or sample.connection is not None
                or before.road != after.road or before.road != completed["road"]
                or before.route_index != after.route_index or after.lane != completed["target_lane"]):
            return 0.0, previous, previous
        source = phase.demand.get(completed["source_lane"])
        target = phase.demand.get(completed["target_lane"])
        if source is None or target is None:
            return 0.0, previous, previous
        previous = source if previous is None else min(previous, source)
        minima[phase.kind] = min(previous, target)
        return V1_DEMAND_REDUCTION_REWARD * (previous - minima[phase.kind]), previous, minima[phase.kind]

    def after_bonus(self, conn, ego_id, sample, rewards, change, target_before,
                    completed_change, unavailable_reason):
        before, phase, demand, connection = sample.motion, sample.phase, sample.demand, sample.connection
        after = None if unavailable_reason else self._motion(conn, ego_id, rewards)
        moving = None
        progress = None
        spans = ()
        target_after = None
        if after is not None:
            movement = self._progress(conn, self.ego_plans[ego_id], before, after)
            if movement is not None:
                progress, spans = movement
            if before.lateral_speed is not None and after.lateral_speed is not None:
                moving = (abs(before.lateral_speed) > 1e-6 or abs(after.lateral_speed) > 1e-6
                          or (not before.road.startswith(":") and before.road == after.road
                              and before.lane != after.lane)
                          or (change is not None and change.last_motion is not None))
            if moving:
                if change is None or change.last_motion is None:
                    target_before = None
                else:
                    target_after = self.target_risk(conn, ego_id, after, change, rewards)
                    completed_change = self._settle_bonus_change(conn, ego_id, after, rewards, change, completed_change)
                    if phase is not None and connection is None and before.road == change.road:
                        settled_lane = (change.target_lane if completed_change is not None
                                        else change.source_lane)
                        demand = phase.demand.get(settled_lane)
        risks = (before.current_risk, after.current_risk if after is not None else None)
        if moving:
            risks += target_before, target_after
        safety = ("unsafe" if any(risk is not None and risk > 0 for risk in risks)
                  else "unknown" if moving is None or None in risks else "safe")
        readiness = phase.readiness(demand) if phase is not None else None
        advancing = progress is not None and (
            progress > 0 if self.v1 else progress > 1e-6 and after.speed > 1e-6
        )
        distance_scale = 1.0
        first_progress = None
        if self.mode in PREPARATION_DISTANCE_MODES:
            fresh = self._first_progress(ego_id, before, after, spans)
            if progress is not None:
                first_progress = min(fresh, progress)
            distance_scale = min(1.0, (first_progress or 0.0) / self.reference_step_m)
        event_bonus = 0.0
        zone_weight = 1.0
        if self.v1:
            event_bonus, minimum_before, minimum_after = self._first_demand_reduction(
                ego_id, sample, after, change, completed_change,
            )
            zone_weight = (0.0 if phase is None else V1_PREPARATION_ZONE_WEIGHT
                           if phase.kind == "preparation" else 1.0)
            readiness = (None if demand is None else V1_CRUISE_WEIGHTS[demand]
                         if demand < len(V1_CRUISE_WEIGHTS) else 0.0)
        cruise_bonus = 0.0
        if readiness is not None and advancing and safety == "safe":
            cruise_bonus = self.bonus_cap * zone_weight * readiness * distance_scale
        reward = event_bonus + cruise_bonus
        row = {
            "mode": self.mode, "phase": phase.kind if phase is not None else "preparation",
            "time_s": before.time_s, "road": before.road, "lane": before.lane, "lane_id": before.lane_id,
            "connection": connection, "post_lane_id": after.lane_id if after is not None else None,
            "local_lane_changes": demand, "normalization": phase.normalization if phase is not None else None,
            "readiness": readiness, "safety": safety, "current_lane_risk_before": risks[0],
            "current_lane_risk_after": risks[1], "target_lane_risk_before": target_before if moving else None,
            "target_lane_risk_after": target_after, "actual_lateral_motion": moving,
            "settled_change": completed_change,
            "progress_m": progress, "advancing": advancing,
            "unavailable_reason": unavailable_reason or ("unverified_progress" if progress is None else None),
            "reward": reward,
        }
        if self.mode in PREPARATION_DISTANCE_MODES:
            row["first_progress_m"] = first_progress
        if self.v1:
            row.update(zone_weight=zone_weight, minimum_demand_before=minimum_before,
                       minimum_demand_after=minimum_after, event_bonus=event_bonus, cruise_bonus=cruise_bonus)
        self.step_info[ego_id] = row
        if phase is None or phase.kind == "preparation":
            self.diagnostics[ego_id]["preparation_edge_steps"] += 1
        if phase is not None:
            stats = self.diagnostics[ego_id]["phases"][phase.kind]
            self._record_completion(conn, ego_id, phase, stats, completed_change)
            stats["preparation_zone_steps"] += 1
            stats["unreachable_steps"] += int(demand is None)
            stats["safe_progress_steps"] += int(advancing and safety == "safe")
            stats["bonus_steps"] += int(reward > 0)
            stats["cumulative_bonus"] += reward
            if self.mode in PREPARATION_DISTANCE_MODES and cruise_bonus > 0:
                stats["rewarded_progress_m"] += min(first_progress, self.reference_step_m)
            if self.v1:
                stats["cumulative_event_bonus"] += event_bonus
                stats["cumulative_cruise_bonus"] += cruise_bonus
        return reward

    def before_step(self, conn, ego_id, position, active_change):
        self.step_info.pop(ego_id, None)
        if position is None:
            return None
        index, lane = position
        phase = self.ego_phases.get(ego_id, {}).get(index)
        preparation_edge = self.ego_preparation_edges.get(ego_id)
        connectors = self.ego_connectors.get(ego_id, {})
        if phase is None and preparation_edge is None and not connectors:
            return None
        vehicle = conn.vehicle
        try:
            road = vehicle.getRoadID(ego_id)
            if preparation_edge is not None and road == preparation_edge:
                self.diagnostics[ego_id]["preparation_edge_steps"] += 1
                self.step_info[ego_id] = {"phase": "preparation", "road": road, "lane": lane,
                                          "time_s": conn.simulation.getTime(), "reward": 0.0}
                return None
            lane_id = vehicle.getLaneID(ego_id) if connectors else None
            connector = connectors.get((position, lane_id))
            if connector is not None:
                phase, demand, downstream_length, _ = connector
                distance = 0.0
                if downstream_length is not None:
                    remaining = conn.lane.getLength(lane_id) - vehicle.getLanePosition(ego_id)
                    if not isfinite(remaining):
                        return None
                    distance = downstream_length + max(0.0, remaining)
                return PreparationSample(phase, lane, distance, demand, False, False,
                                         conn.simulation.getTime(), road, connector=True)
            if phase is None or road != phase.approach:
                return None
            distance = conn.lane.getLength(lane_id or vehicle.getLaneID(ego_id)) - vehicle.getLanePosition(ego_id)
            lateral_speed = vehicle.getLateralSpeed(ego_id)
            if not isfinite(distance) or not isfinite(lateral_speed):
                return None
        except TraCIException:
            return None
        distance = max(0.0, distance)
        demand = phase.demand.get(lane)
        neighbors = phase.improving_lanes.get(lane, ())
        movement = any(lateral_speed * (target - lane) > LATERAL_MOTION_MPS for target in neighbors)
        if active_change is not None and active_change.road == phase.approach:
            movement |= (active_change.last_motion is not None
                         and lane in (active_change.source_lane, active_change.target_lane)
                         and phase.reduces_demand(active_change.source_lane, active_change.target_lane)
                         and lateral_speed * active_change.direction > LATERAL_MOTION_MPS)
        opportunity = (distance <= phase.distance_m and bool(neighbors)
                       and self._safe_opportunity(vehicle, ego_id, lane, neighbors))
        return PreparationSample(phase, lane, distance, demand, opportunity, movement, conn.simulation.getTime(), road)

    @staticmethod
    def _moving_after(vehicle, ego_id, sample, on_road):
        if sample.moving:
            return True
        if not on_road:
            return None
        try:
            if vehicle.getRoadID(ego_id) != sample.phase.approach:
                return None
            lane = vehicle.getLaneIndex(ego_id)
            lateral_speed = vehicle.getLateralSpeed(ego_id)
            if not isfinite(lateral_speed):
                return None
        except TraCIException:
            return None
        neighbors = sample.phase.improving_lanes.get(sample.lane, ())
        return lane in neighbors or any(lateral_speed * (target - sample.lane) > LATERAL_MOTION_MPS
                                        for target in neighbors)

    def _record_completion(self, conn, ego_id, phase, stats, completed_change):
        if (completed_change is not None and completed_change["road"] == phase.approach
                and phase.reduces_demand(completed_change["source_lane"], completed_change["target_lane"])
                and phase.demand[completed_change["target_lane"]] == 0):
            vehicle = conn.vehicle
            try:
                remaining = conn.lane.getLength(vehicle.getLaneID(ego_id)) - vehicle.getLanePosition(ego_id)
                remaining = max(0.0, remaining) if isfinite(remaining) else None
            except TraCIException:
                remaining = None
            stats["last_necessary_completion"] = {"time_s": completed_change["time"],
                                                  "distance_m": remaining}

    def after_step(self, conn, ego_id, sample, on_road, completed_change):
        phase = sample.phase
        stats = self.diagnostics[ego_id]["phases"][phase.kind]
        self._record_completion(conn, ego_id, phase, stats, completed_change)
        if sample.distance_m > phase.distance_m and not sample.connector:
            return 0.0

        moving = False if sample.connector else self._moving_after(conn.vehicle, ego_id, sample, on_road)
        penalty = phase.penalty(sample.distance_m, sample.demand, sample.safe_opportunity, moving)
        row = {"phase": phase.kind, "road": sample.road, "next_edge": phase.next_edge,
               "time_s": sample.time_s, "distance_m": sample.distance_m, "lane": sample.lane,
               "local_lane_changes": sample.demand, "normalization": phase.normalization,
               "safe_opportunity": sample.safe_opportunity, "executing_necessary_change": moving,
               "reward": penalty}
        self.step_info[ego_id] = row
        stats["preparation_zone_steps"] += 1
        stats["safe_opportunity_steps"] += int(sample.safe_opportunity)
        stats["unreachable_steps"] += int(sample.demand is None)
        if penalty < 0:
            stats["penalty_steps"] += 1
            stats["cumulative_penalty"] += penalty
            if stats["first_penalty"] is None:
                stats["first_penalty"] = row
        return penalty
