"""Read-only lane-route urgency estimate for Antwerp diagnostics.

The estimate is an optimistic schedule: it ignores traffic gaps and assumes
successive lane changes can each finish in ``lane_change_seconds``. It is not
an action constraint, a reward, or a probability of reaching the destination.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Lane:
    length: float
    speed_limit: float


@dataclass(frozen=True)
class RouteNetwork:
    lanes: dict[str, dict[int, Lane]]
    connections: dict[tuple[str, str, int], tuple[tuple[int, str | None], ...]]


@lru_cache(maxsize=4)
def load_route_network(path: str | Path) -> RouteNetwork:
    """Load the lane graph, including the connector chosen at each junction."""
    root = ET.parse(path).getroot()
    lanes = {}
    for edge in root.findall('edge'):
        if edge.get('function') == 'internal':
            continue
        edge_lanes = {}
        for lane in edge.findall('lane'):
            length = float(lane.attrib['length'])
            speed_limit = float(lane.attrib['speed'])
            if (not math.isfinite(length) or not math.isfinite(speed_limit)
                    or length <= 0 or speed_limit <= 0):
                raise ValueError(f'Invalid lane geometry: {lane.attrib["id"]}')
            edge_lanes[int(lane.attrib['index'])] = Lane(length, speed_limit)
        if edge_lanes:
            lanes[edge.attrib['id']] = edge_lanes
    connections = {}
    for connection in root.findall('connection'):
        source, target = connection.get('from'), connection.get('to')
        if source not in lanes or target not in lanes:
            continue
        key = (source, target, int(connection.attrib['fromLane']))
        connections.setdefault(key, []).append(
            (int(connection.attrib['toLane']), connection.get('via')))
    return RouteNetwork(lanes, {key: tuple(value) for key, value in connections.items()})


def _edge_options(network, edge, next_edge, entry_lane, first_position, speed,
                  vehicle_length, lane_change_edges, lane_change_seconds,
                  speed_floor_fraction, end_buffer_m, min_time_s):
    lanes = network.lanes[edge]
    end_lanes = lanes if edge in lane_change_edges else (entry_lane,)
    for exit_lane in end_lanes:
        crossed = range(min(entry_lane, exit_lane), max(entry_lane, exit_lane) + 1)
        if any(index not in lanes for index in crossed):
            continue
        changes = abs(exit_lane - entry_lane)
        remaining = min(lanes[index].length for index in crossed) - first_position
        usable = max(0.0, remaining - vehicle_length - end_buffer_m)
        speed_limit = min(lanes[index].speed_limit for index in crossed)
        estimated_speed = max(speed, speed_floor_fraction * speed_limit)
        available_s = usable / estimated_speed
        load = (changes * lane_change_seconds / max(available_s, min_time_s)
                if changes else 0.0)
        for next_lane, connector in network.connections.get((edge, next_edge, exit_lane), ()):
            if next_lane in network.lanes[next_edge]:
                yield next_lane, changes, {
                    'edge': edge, 'entry_lane': entry_lane, 'exit_lane': exit_lane,
                    'connector': connector, 'next_lane': next_lane,
                    'lane_changes': changes, 'usable_distance_m': usable,
                    'available_s': available_s,
                    'estimated_speed_mps': estimated_speed, 'load': load,
                }


def route_urgency(network: RouteNetwork, route, edge_index, lane_index,
                  lane_position_m, speed_mps, vehicle_length_m,
                  lane_change_edges, lane_change_seconds=4.0,
                  speed_floor_fraction=0.7, end_buffer_m=5.0,
                  min_time_s=0.1):
    """Find the least urgent lane plan over all valid connections on a route.

    For a plan, urgency is the maximum of required/available lane-change time
    over its edges. The result minimizes that maximum. A separate minimum
    lane-change count is returned because the least urgent plan can use more
    changes than the shortest lane plan. The final route edge is the goal.
    """
    route = tuple(route)
    values = (lane_position_m, speed_mps, vehicle_length_m, lane_change_seconds,
              speed_floor_fraction, end_buffer_m, min_time_s)
    if (not route or not 0 <= edge_index < len(route)
            or any(edge not in network.lanes for edge in route)
            or lane_index not in network.lanes[route[edge_index]]
            or any(not math.isfinite(value) for value in values)
            or lane_position_m < 0 or speed_mps < 0 or vehicle_length_m < 0
            or lane_change_seconds <= 0 or not 0 < speed_floor_fraction <= 1
            or end_buffer_m < 0 or min_time_s <= 0
            or lane_position_m > network.lanes[route[edge_index]][lane_index].length + 1e-3):
        raise ValueError('Invalid route or nonfinite lane-urgency input')
    if edge_index == len(route) - 1:
        return {'status': 'at_destination', 'urgency': 0.0,
                'minimum_lane_changes': 0, 'selected_lane_changes': 0,
                'bottleneck_edge': None, 'plan': []}

    # One best peak-load path per lane is sufficient for the urgency minimum.
    # A second DP tracks the fewest changes independently of that path.
    states = {lane_index: (0.0, 0, ())}
    fewest = {lane_index: 0}
    allowed = frozenset(lane_change_edges)
    for index in range(edge_index, len(route) - 1):
        edge, next_edge = route[index:index + 2]
        following = {}
        following_fewest = {}
        for entry_lane, (peak, total_changes, plan) in states.items():
            for target_lane, changes, segment in _edge_options(
                    network, edge, next_edge, entry_lane,
                    lane_position_m if index == edge_index else 0.0,
                    speed_mps, vehicle_length_m, allowed, lane_change_seconds,
                    speed_floor_fraction, end_buffer_m, min_time_s):
                candidate = (max(peak, segment['load']), total_changes + changes,
                             plan + (segment,))
                previous = following.get(target_lane)
                if previous is None or candidate[:2] < previous[:2]:
                    following[target_lane] = candidate
                following_fewest[target_lane] = min(
                    following_fewest.get(target_lane, math.inf),
                    fewest[entry_lane] + changes)
        if not following:
            return {'status': 'unreachable', 'urgency': None,
                    'minimum_lane_changes': None, 'selected_lane_changes': None,
                    'bottleneck_edge': None, 'plan': []}
        states, fewest = following, following_fewest

    peak, total_changes, plan = min(states.values(), key=lambda state: state[:2])
    bottleneck = max(plan, key=lambda segment: segment['load'])
    return {'status': 'ok', 'urgency': peak,
            'minimum_lane_changes': min(fewest.values()),
            'selected_lane_changes': total_changes,
            'bottleneck_edge': bottleneck['edge'] if peak > 0 else None,
            'plan': list(plan)}


def observe_route_urgency(base, network: RouteNetwork, lane_change_edges,
                          lane_change_seconds=4.0):
    """Read SUMO state without advancing time or changing vehicle control."""
    if not getattr(base, 'ego_on_road', False):
        return {'status': 'off_road', 'urgency': None, 'pending': False}
    vehicle = base.traci.vehicle
    ego = base.ego_id
    route = tuple(vehicle.getRoute(ego))
    road = vehicle.getRoadID(ego)
    lane_id = vehicle.getLaneID(ego)
    route_index = int(vehicle.getRouteIndex(ego))
    connector = road.startswith(':')
    if connector:
        # A connector is identified by its actual `via` lane, irrespective of
        # the source and destination lane numbering conventions.
        matches = [(index + 1, target_lane)
                   for index in range(len(route) - 1)
                   for source_lane in network.lanes[route[index]]
                   for target_lane, via in network.connections.get(
                       (route[index], route[index + 1], source_lane), ())
                   if via == lane_id]
        matches = [match for match in matches if match[0] >= max(0, route_index - 1)]
        if len(matches) > 1 and route_index >= 0:
            closest = min(abs(index - route_index) for index, _ in matches)
            matches = [match for match in matches
                       if abs(match[0] - route_index) == closest]
        if len(matches) != 1:
            return {'status': 'unmapped_connector', 'urgency': None,
                    'pending': False, 'lane_id': lane_id}
        edge_index, lane_index = matches[0]
        position = 0.0
    else:
        if 0 <= route_index < len(route) and route[route_index] == road:
            edge_index = route_index
        else:
            matches = [index for index, edge in enumerate(route) if edge == road]
            if len(matches) != 1:
                return {'status': 'off_route', 'urgency': None,
                        'pending': False, 'lane_id': lane_id}
            edge_index = matches[0]
        lane_index = int(vehicle.getLaneIndex(ego))
        position = float(vehicle.getLanePosition(ego))
    lateral_speed = abs(float(vehicle.getLateralSpeed(ego)))
    clearance = ((base.traci.lane.getWidth(lane_id) - vehicle.getWidth(ego)) / 2.0)
    settled = (clearance >= 0
               and abs(float(vehicle.getLateralLanePosition(ego))) <= clearance + 0.001
               and lateral_speed <= 0.05)
    pending = not settled or getattr(base, 'exit_maneuver', None) is not None
    result = route_urgency(network, route, edge_index, lane_index, position,
                           float(vehicle.getSpeed(ego)), float(vehicle.getLength(ego)),
                           lane_change_edges, lane_change_seconds=lane_change_seconds)
    result.update(pending=pending, provisional=pending or connector,
                  valid_for_threshold=(not pending and not connector
                                       and result['status'] == 'ok'),
                  road=road, lane_id=lane_id, projected_edge=route[edge_index],
                  projected_lane=lane_index, speed_mps=float(vehicle.getSpeed(ego)))
    return result
