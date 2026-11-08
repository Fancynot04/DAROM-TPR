"""SECRM route penalty; topology helper remains shared."""
import math

def route_penalty(required, remaining, distance_scale_m=200.0):
    """Bound route deficit using the remaining legal lane-change window.

    ``remaining`` is the distance through the last route edge where a needed
    lane change may still occur, not distance to D3's endpoint.  A lane deficit
    of n is paid as ``-min(1, n * scale / max(remaining, scale))``.
    """
    if (type(required) is not int or required < 0 or not math.isfinite(remaining)
            or remaining < 0 or not math.isfinite(distance_scale_m) or distance_scale_m <= 0):
        raise ValueError('Invalid settled route requirement/distance')
    if required == 0:
        return 0.0
    return -min(1.0, required * distance_scale_m / max(remaining, distance_scale_m))


def executable_distance(network, route, index, lane, lane_position, change_edges):
    """Distance through the last on-route edge that can host a lane change.

    This deliberately excludes D3 endpoint distance.  It is undefined on an
    internal connector; callers must mark that state unavailable rather than
    treating it as a zero-deficit route.
    """
    required = minimum_changes(network, route, index, lane, change_edges)
    if required == 0:
        return required, 0.0
    if not (0 <= index < len(route) - 1 and route[index] in network.lanes
            and lane in network.lanes[route[index]] and math.isfinite(lane_position)):
        raise ValueError('Invalid route state for executable distance')
    last = max((i for i in range(index, len(route) - 1) if route[i] in change_edges), default=None)
    if last is None:
        raise ValueError('No legal lane-change edge remains for a nonzero deficit')
    distance = network.lanes[route[index]][lane].length - lane_position
    for edge in route[index + 1:last + 1]:
        distance += min(item.length for item in network.lanes[edge].values())
    if not math.isfinite(distance) or distance < 0:
        raise ValueError('Invalid executable lane-change distance')
    return required, distance

def minimum_changes(network, route, index, lane, change_edges):
    """Topology only: no gap/urgency/time feasibility assumptions."""
    if index < 0 or index >= len(route) or lane not in network.lanes[route[index]]:
        raise ValueError('Invalid stable lane/route index')
    costs={lane:0}
    for edge,next_edge in zip(route[index:],route[index+1:]):
        following={}
        for start,cost in costs.items():
            exits=network.lanes[edge] if edge in change_edges else (start,)
            for end in exits:
                if any(i not in network.lanes[edge] for i in range(min(start,end),max(start,end)+1)):continue
                for target,_ in network.connections.get((edge,next_edge,end),()):
                    value=cost+abs(end-start)
                    following[target]=min(value,following.get(target,value))
        costs=following
    if not costs:raise ValueError('No continuation to route endpoint')
    return min(costs.values())
