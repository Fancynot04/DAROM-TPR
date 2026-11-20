"""Minimum commanded lane changes along a fixed SUMO route."""

from collections import defaultdict, deque
from xml.etree import ElementTree

from sumolib.net.lane import get_allowed

from .control import LANE_CHANGE_EDGES


# The actor's existing lane slot is scaled by 5. Reserve the next integer
# for unreachable states, and validate the range when a route is built.
UNREACHABLE_LANE_CHANGES = 6

# Match the full SUMO route, including the approach to the destination.
DEFAULT_ROUTE_GUIDANCE_RULES = [
    {
        "route": ["O3", "E75", "E76", "E77", "D3"],
        "critical_connection": ["E76", "E77"],
        "accepted_connections": [["E76_0", "E77_0"]],
    },
]


class RouteLaneNetwork:
    def __init__(self, net_file):
        root = ElementTree.parse(net_file).getroot()
        self.lanes = {
            edge.attrib["id"]: {int(lane.attrib["index"]): lane.attrib for lane in edge.findall("lane")}
            for edge in root.findall("edge")
        }
        self.connections = defaultdict(list)
        self.internal_successor = {}
        for node in root.findall("connection"):
            link = node.attrib
            source, target = link["from"], link["to"]
            source_lane, target_lane = int(link["fromLane"]), int(link["toLane"])
            via = link.get("via")
            if source.startswith(":"):
                self.internal_successor[(self.lanes[source][source_lane]["id"], target, target_lane)] = via
            else:
                self.connections[source, target].append((source_lane, target_lane, via))
        self.permissions = {
            lane["id"]: get_allowed(
                None if lane.get("allow") == "all" else lane.get("allow"), lane.get("disallow"),
            )
            for lanes in self.lanes.values() for lane in lanes.values()
        }
        self.plans = {}

    def plan(self, route, vehicle_class):
        key = route, vehicle_class
        if key not in self.plans:
            self.plans[key] = RouteLanePlan(self, route, vehicle_class)
        return self.plans[key]


class RouteLanePlan:
    def __init__(self, network, route, vehicle_class):
        self.route = route
        permitted = [
            {index: lane for index, lane in network.lanes[edge].items()
             if vehicle_class in network.permissions[lane["id"]]}
            for edge in route
        ]
        lateral = defaultdict(list)
        forward = defaultdict(list)
        reverse = defaultdict(list)
        self.internal = {}
        self.internal_connections = {}
        self.lane_connections = set()
        for index, (edge, lanes) in enumerate(zip(route, permitted)):
            if edge in LANE_CHANGE_EDGES:
                for lane_index, lane in lanes.items():
                    for offset, attribute in ((-1, "changeRight"), (1, "changeLeft")):
                        neighbor = lane_index + offset
                        permission = lane.get(attribute, "all").split()
                        if neighbor in lanes and ("all" in permission or vehicle_class in permission):
                            lateral[index, lane_index].append((index, neighbor))
                            reverse[index, neighbor].append(((index, lane_index), 1))

            if index + 1 == len(route):
                continue
            next_edge = route[index + 1]
            for source, target, via in network.connections.get((edge, next_edge), ()):
                if source not in lanes or target not in permitted[index + 1]:
                    continue
                internal_lanes = []
                while via:
                    internal_lanes.append(via)
                    via = network.internal_successor.get((via, next_edge, target))
                if any(vehicle_class not in network.permissions[lane] for lane in internal_lanes):
                    continue
                destination = index + 1, target
                forward[index, source].append(destination)
                reverse[destination].append(((index, source), 0))
                pair = lanes[source]["id"], permitted[index + 1][target]["id"]
                self.lane_connections.add((index, pair))
                for lane in internal_lanes:
                    self.internal[index, lane] = destination
                    self.internal_connections[index, lane] = pair

        distance = {(len(route) - 1, lane): 0 for lane in permitted[-1]}
        queue = deque(distance)
        while queue:
            node = queue.popleft()
            for predecessor, cost in reverse[node]:
                candidate = distance[node] + cost
                if predecessor not in distance or candidate < distance[predecessor]:
                    distance[predecessor] = candidate
                    (queue.append if cost else queue.appendleft)(predecessor)

        if max(distance.values(), default=0) >= UNREACHABLE_LANE_CHANGES:
            raise ValueError(f"Lane-change count exceeds the actor encoding range for {route}")

        self.signed_changes = {}
        self.forward_lanes = set()
        # Resolve ties from the end of the route, preferring natural progress.
        for index in reversed(range(len(route))):
            for lane in sorted(permitted[index]):
                node = index, lane
                exits = [successor for successor in forward[node] if successor in distance]
                if exits:
                    self.forward_lanes.add(node)
                if node not in distance:
                    continue
                cost = distance[node]
                if cost == 0:
                    self.signed_changes[node] = 0
                    continue
                natural = [successor for successor in exits if distance[successor] == cost]
                if natural:
                    self.signed_changes[node] = self.signed_changes[min(natural)]
                    continue
                first = min(neighbor for neighbor in lateral[node]
                            if distance.get(neighbor) == cost - 1)
                self.signed_changes[node] = cost if first[1] > lane else -cost

        self.lateral_neighbors = lateral

    def position(self, vehicle, ego_id):
        """Return the graph node, mapping a committed connector downstream."""
        index = vehicle.getRouteIndex(ego_id)
        road = vehicle.getRoadID(ego_id)
        if road.startswith(":"):
            return self.internal.get((index, vehicle.getLaneID(ego_id)))
        if 0 <= index < len(self.route) and road == self.route[index]:
            return index, vehicle.getLaneIndex(ego_id)
        return None

    def completed_connection(self, previous, vehicle, ego_id, critical_connection):
        """Resolve the traversed link. A missing vehicle indicates a confirmed arrival."""
        index, lane = previous
        if self.route[index:index + 2] != critical_connection:
            return None
        internal_pair = self.internal_connections.get(previous)
        if vehicle is None:
            # SUMO can remove an Ego on the same step it enters the final edge.
            return internal_pair if index + 2 == len(self.route) else None
        if (
            vehicle.getRouteIndex(ego_id) != index + 1
            or vehicle.getRoadID(ego_id) != critical_connection[1]
            or tuple(vehicle.getRoute(ego_id)) != self.route
        ):
            return None
        # The internal lane identifies the chosen link even if the downstream
        # lane has changed by the time it is sampled.
        pair = internal_pair or (lane, vehicle.getLaneID(ego_id))
        return pair if (index, pair) in self.lane_connections else None
