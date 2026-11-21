"""Stage-local route preparation and its independent reward accounting."""

from collections import defaultdict, deque
from dataclasses import dataclass
from math import isfinite

from traci.exceptions import TraCIException

from src.rewards.masac import adjacent_lane_risk


DEFAULT_ROUTE_PREPARATION_STAGES = [
    {
        "route": ["O3", "E75", "E76", "E77", "D3"],
        "phases": [
            {"kind": "weaving", "critical_connection": ["E76", "E77"],
             "accepted_entry_lanes": ["E77_0"], "preparation_distance_m": 300.0},
            {"kind": "exit", "critical_connection": ["E77", "D3"],
             "accepted_entry_lanes": ["D3_0"], "preparation_distance_m": 172.35},
        ],
    },
]
DEFAULT_ROUTE_PREPARATION_WEIGHTS = {"weaving": 0.001, "exit": 0.002}
LATERAL_MOTION_MPS = 0.05


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
        lane_indices = network.lanes[approach]
        zero = [
            lane_index for lane_index, lane in lane_indices.items()
            if any(target in reachable_entries for target, _ in plan.passage_links.get((index, lane["id"]), {}))
        ]
        if not zero:
            raise ValueError(f"No permitted preparation connection for {plan.route} at {approach}")

        reverse = defaultdict(list)
        for lane_index in lane_indices:
            for _, neighbor in plan.lateral_neighbors.get((index, lane_index), ()):
                reverse[neighbor].append(lane_index)
        demand = dict.fromkeys(zero, 0)
        queue = deque(zero)
        while queue:
            lane_index = queue.popleft()
            for previous in reverse[lane_index]:
                if previous not in demand:
                    demand[previous] = demand[lane_index] + 1
                    queue.append(previous)
        improving = {
            lane_index: tuple(neighbor for _, neighbor in plan.lateral_neighbors.get((index, lane_index), ())
                              if neighbor in demand and demand[neighbor] < count)
            for lane_index, count in demand.items()
        }
        return index, cls(spec["kind"], approach, next_edge, float(spec["preparation_distance_m"]),
                          weight, demand, max(demand.values()), improving)

    def penalty(self, distance, demand, safe_opportunity, moving):
        if not demand or not safe_opportunity or moving is not False or not self.weight:
            return 0.0
        urgency = min(1.0, max(0.0, 1.0 - distance / self.distance_m))
        return -self.weight * urgency * min(demand / self.normalization, 1.0)

    def reduces_demand(self, source, target):
        return target in self.improving_lanes.get(source, ())


@dataclass
class PreparationSample:
    phase: PreparationPhase
    lane: int
    distance_m: float
    demand: int | None
    safe_opportunity: bool
    moving: bool
    time_s: float


class RoutePreparation:
    def __init__(self, settings):
        self.weights = settings["weights"]
        self.stages = {}
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
                distance = spec["preparation_distance_m"]
                if kind not in self.weights or kind in kinds or not isfinite(distance) or distance <= 0:
                    raise ValueError(f"Invalid or duplicate preparation phase: {spec}")
                kinds.add(kind)
            self.stages[route] = rule["phases"]
        self.compiled = {}
        self.reset(())

    def reset(self, ego_ids):
        self.ego_phases = {}
        self.diagnostics = {ego: {"enabled": False, "reason": "not_departed", "phases": {}}
                            for ego in ego_ids}
        self.step_info = {}

    def configure(self, ego_id, network, plan):
        if plan not in self.compiled:
            phases = {}
            for spec in self.stages.get(plan.route, ()):
                index, phase = PreparationPhase.build(spec, self.weights[spec["kind"]], network, plan)
                if index in phases:
                    raise ValueError(f"Preparation phases overlap at {phase.approach}")
                phases[index] = phase
            self.compiled[plan] = phases
        phases = self.compiled[plan]
        self.ego_phases[ego_id] = phases
        enabled = any(p.weight > 0 for p in phases.values())
        self.diagnostics[ego_id] = {
            "enabled": enabled, "reason": "enabled" if enabled else "zero_weights" if phases else "unconfigured_route",
            "route": list(plan.route), "phases": {
                phase.kind: {
                    "enabled": phase.weight > 0, "weight": phase.weight,
                    "normalization": phase.normalization, "preparation_zone_steps": 0,
                    "safe_opportunity_steps": 0, "penalty_steps": 0, "cumulative_penalty": 0.0,
                    "unreachable_steps": 0, "first_penalty": None, "last_necessary_completion": None,
                }
                for phase in phases.values()
            },
        }

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

    def before_step(self, conn, ego_id, position, active_change):
        self.step_info.pop(ego_id, None)
        if position is None:
            return None
        index, lane = position
        phase = self.ego_phases.get(ego_id, {}).get(index)
        if phase is None:
            return None
        vehicle = conn.vehicle
        try:
            if vehicle.getRoadID(ego_id) != phase.approach:
                return None
            distance = conn.lane.getLength(vehicle.getLaneID(ego_id)) - vehicle.getLanePosition(ego_id)
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
        return PreparationSample(phase, lane, distance, demand, opportunity, movement, conn.simulation.getTime())

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

    def after_step(self, conn, ego_id, sample, on_road, completed_change):
        phase = sample.phase
        stats = self.diagnostics[ego_id]["phases"][phase.kind]
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
        if sample.distance_m > phase.distance_m:
            return 0.0

        moving = self._moving_after(conn.vehicle, ego_id, sample, on_road)
        penalty = phase.penalty(sample.distance_m, sample.demand, sample.safe_opportunity, moving)
        row = {"phase": phase.kind, "road": phase.approach, "next_edge": phase.next_edge,
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
